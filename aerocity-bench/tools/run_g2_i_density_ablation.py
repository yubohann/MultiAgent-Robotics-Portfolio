"""Execute frozen G2-I atlas-density calibration inputs at L0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import BASELINES
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.inspection_atlas import inspection_sampling_policy
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config

try:
    from tools.build_g2_i_density_ablation_manifest import (
        DENSITY_MANIFEST_SCHEMA,
        _private_truth_projection,
    )
    from tools.run_g2_i_l0_calibration import (
        _calibration_implementation_hash,
        _local_path,
        _run_record,
        _validate_frozen_episode,
    )
except ModuleNotFoundError:
    from build_g2_i_density_ablation_manifest import (
        DENSITY_MANIFEST_SCHEMA,
        _private_truth_projection,
    )
    from run_g2_i_l0_calibration import (
        _calibration_implementation_hash,
        _local_path,
        _run_record,
        _validate_frozen_episode,
    )


REPORT_SCHEMA = "org.aerocity.bench.g2-i-atlas-density-ablation.v1"
PUBLIC_METHODS = ("atlas-surface-inspector", "atlas-region-greedy")


def _density_implementation_hash() -> str:
    return content_hash(
        {
            "base_calibration": _calibration_implementation_hash(),
            "density_runner": file_hash(Path(__file__)),
        }
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--policy-id", action="append", dest="policy_ids")
    parser.add_argument("--method", action="append", choices=PUBLIC_METHODS, dest="methods")
    args = parser.parse_args(argv)
    if args.policy_ids and len(set(args.policy_ids)) != len(args.policy_ids):
        parser.error("--policy-id values must not repeat")
    if args.methods and len(set(args.methods)) != len(args.methods):
        parser.error("--method values must not repeat")
    args.methods = tuple(
        method_id
        for method_id in PUBLIC_METHODS
        if not args.methods or method_id in set(args.methods)
    )
    return args


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError(f"cannot aggregate empty density condition: {field}")
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_density_results(
    raw_results: list[dict[str, Any]],
    *,
    policy_ids: tuple[str, ...],
    method_ids: tuple[str, ...] = PUBLIC_METHODS,
) -> list[dict[str, Any]]:
    """Report ancestor means; process replicates do not become samples."""

    if not raw_results:
        raise ValueError("density ablation has no execution results")
    reports = []
    for policy_id in policy_ids:
        policy_rows = [row for row in raw_results if row["sampling_policy_id"] == policy_id]
        if not policy_rows:
            raise ValueError("density ablation lacks a declared policy condition")
        for method_id in method_ids:
            rows = [row for row in policy_rows if row["method_id"] == method_id]
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row["layout_ancestor"]), []).append(row)
            if not grouped or any(not values for values in grouped.values()):
                raise ValueError("density ablation lacks method ancestor rows")
            ancestors = [
                {
                    "layout_ancestor_hash": content_hash(ancestor),
                    "episode_replicate_count": len(values),
                    "mean_confirmation_count": _mean(values, "confirmation_count"),
                    "mean_final_confirmed_recall": _mean(values, "final_confirmed_recall"),
                    "all_returned_home": all(value["returned_home_all"] for value in values),
                    "collision_count": sum(int(value["collision_count"]) for value in values),
                    "out_of_bounds_actions": sum(
                        int(value["out_of_bounds_actions"]) for value in values
                    ),
                    "deadline_misses": sum(int(value["deadline_misses"]) for value in values),
                    "deadline_failure_replicate_count": sum(
                        int(value["deadline_misses"]) > 0 for value in values
                    ),
                    "maximum_deadline_misses_per_replicate": max(
                        int(value["deadline_misses"]) for value in values
                    ),
                }
                for ancestor, values in sorted(grouped.items())
            ]
            reports.append(
                {
                    "sampling_policy_id": policy_id,
                    "method_id": method_id,
                    "observation_profile": BASELINES[method_id].observation_profile,
                    "requires_private_truth": False,
                    "independent_ancestor_count": len(ancestors),
                    "nonzero_ancestor_count": sum(
                        item["mean_confirmation_count"] > 0.0 for item in ancestors
                    ),
                    "mean_final_confirmed_recall": _mean(
                        ancestors, "mean_final_confirmed_recall"
                    ),
                    "all_returned_home": all(
                        item["all_returned_home"] for item in ancestors
                    ),
                    "collision_count": sum(item["collision_count"] for item in ancestors),
                    "out_of_bounds_actions": sum(
                        item["out_of_bounds_actions"] for item in ancestors
                    ),
                    "deadline_misses": sum(item["deadline_misses"] for item in ancestors),
                    "ancestors": ancestors,
                }
            )
    return reports


def run_density_ablation(
    manifest_path: Path,
    *,
    max_steps: int | None,
    selected_policy_ids: tuple[str, ...] | None = None,
    method_ids: tuple[str, ...] = PUBLIC_METHODS,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != DENSITY_MANIFEST_SCHEMA:
        raise ValueError("unsupported G2-I atlas-density manifest")
    if manifest.get("purpose") != "method-independent-atlas-density-calibration":
        raise ValueError("density ablation requires method-independent inputs")
    if manifest.get("self_method_results_used") is not False:
        raise ValueError("density ablation cannot consume self-method results")
    if manifest.get("frozen_before_execution") is not True:
        raise ValueError("density inputs must be frozen before execution")
    if (
        manifest.get("public_region_cohort_matched_across_density_conditions") is not True
        or manifest.get("private_target_realizations_matched_across_density_conditions")
        is not True
        or manifest.get("private_targets_reused_from_source_manifest") is not True
        or manifest.get("source_sector_frozen_before_original_private_sampling") is not True
    ):
        raise ValueError(
            "density inputs do not freeze source regions and paired private truth"
        )
    declared_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if not isinstance(declared_hash, str) or content_hash(unhashed) != declared_hash:
        raise ValueError("density manifest hash differs")
    root = manifest_path.parent
    config = load_ordinary_config(
        _local_path(root, manifest["release_config_path"], "release_config_path")
    )
    declared_policy_ids = tuple(str(item) for item in manifest.get("policy_ids", []))
    if not declared_policy_ids or len(set(declared_policy_ids)) != len(
        declared_policy_ids
    ):
        raise ValueError("density manifest policy IDs are invalid")
    requested = set(selected_policy_ids or declared_policy_ids)
    if not requested or not requested.issubset(declared_policy_ids):
        raise ValueError("density shard requests an undeclared policy ID")
    policy_ids = tuple(
        policy_id for policy_id in declared_policy_ids if policy_id in requested
    )
    if not method_ids or not set(method_ids).issubset(PUBLIC_METHODS):
        raise ValueError("density shard requests an unknown public method")
    expected_processes = set(config.target_processes("calibration"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("density manifest has no records")
    raw: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    common_region_hashes: dict[str, str] = {}
    paired_truth_hashes: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("density record must be an object")
        policy_id = str(record.get("sampling_policy_id", ""))
        ancestor = str(record.get("layout_ancestor", ""))
        if policy_id not in declared_policy_ids or not ancestor:
            raise ValueError("density record policy or ancestor is invalid")
        if policy_id not in policy_ids:
            continue
        city = read_json(_local_path(root, record["city_path"], "city_path"))
        episode = read_json(
            _local_path(root, record["private_episode_path"], "private_episode_path")
        )
        if str(city.get("split")) in FORMAL_SPLITS:
            raise ValueError("density ablation must not inspect formal split cities")
        if content_hash(city) != record.get("city_hash"):
            raise ValueError("density record city hash differs")
        task_spec = compile_g2_i_task_spec(
            city,
            config.raw["execution_contract"],
            config.raw["fleet"],
            inspection_sampling_policy=inspection_sampling_policy(policy_id),
        )
        if task_spec["task_spec_hash"] != record.get("task_spec_hash"):
            raise ValueError("density task spec differs from its frozen input")
        if task_spec["inspection_atlas"]["atlas_hash"] != record.get("atlas_hash"):
            raise ValueError("density atlas differs from its frozen input")
        if episode.get("episode_hash") != record.get("private_episode_hash"):
            raise ValueError("density private episode hash differs from its record")
        _validate_frozen_episode(episode, city, task_spec, config)
        if episode.get("mission_sector_hash") != record.get("mission_sector_hash"):
            raise ValueError("density mission sector hash differs from its record")
        sector = episode.get("mission_sector")
        if not isinstance(sector, dict):
            raise ValueError("density episode lacks its frozen public mission sector")
        region_hash = content_hash(sorted(str(value) for value in sector["selected_region_ids"]))
        if region_hash != record.get("common_region_set_hash"):
            raise ValueError("density record common-region hash differs")
        prior_region_hash = common_region_hashes.setdefault(ancestor, region_hash)
        if region_hash != prior_region_hash:
            raise ValueError("density conditions changed the public region cohort")
        process = str(episode.get("target_process", ""))
        truth_hash = content_hash(_private_truth_projection(episode))
        if truth_hash != record.get("paired_private_truth_hash"):
            raise ValueError("density record paired private-truth hash differs")
        prior_truth_hash = paired_truth_hashes.setdefault((ancestor, process), truth_hash)
        if truth_hash != prior_truth_hash:
            raise ValueError("density conditions changed paired private truth")
        key = (policy_id, ancestor, process)
        if process not in expected_processes or key in seen:
            raise ValueError("density records must have one target process per condition")
        seen.add(key)
        for method_id in method_ids:
            result = _run_record(
                config,
                city,
                episode,
                task_spec,
                method_id=method_id,
                max_steps=max_steps,
            )
            raw.append(
                {
                    **result,
                    "sampling_policy_id": policy_id,
                    "layout_ancestor": ancestor,
                }
            )
    expected_rows = len(policy_ids) * len({key[1] for key in seen}) * len(
        expected_processes
    )
    if len(seen) != expected_rows:
        raise ValueError("density manifest lacks complete target-process condition pairs")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_ONLY",
        "manifest_hash": content_hash(manifest),
        "source_manifest_hash": str(manifest["source_manifest_hash"]),
        "base_calibration_implementation_hash": _calibration_implementation_hash(),
        "density_implementation_hash": _density_implementation_hash(),
        "execution_level": "L0",
        "contract": {
            "declared_policy_ids": list(declared_policy_ids),
            "selected_policy_ids": list(policy_ids),
            "selected_method_ids": list(method_ids),
            "complete_condition_set": (
                policy_ids == declared_policy_ids and method_ids == PUBLIC_METHODS
            ),
            "selected_city_panel_matched_across_density_conditions": True,
            "policy_specific_sector_frozen_before_private_sampling": True,
            "public_region_cohort_matched_across_density_conditions": True,
            "private_target_realizations_matched_across_density_conditions": True,
            "private_targets_reused_from_source_manifest": True,
            "target_truth_visible_to_methods": False,
            "episode_replicates_are_not_independent": True,
            "density_comparison_is_task_calibration_not_method_ranking": True,
            "l0_not_a_native_or_formal_score": True,
        },
        "aggregate": {
            "sampling_policy_ids": list(policy_ids),
            "independent_ancestor_count": len({key[1] for key in seen}),
            "target_processes": sorted(expected_processes),
            "method_count": len(method_ids),
            "condition_count": len(policy_ids) * len(method_ids),
            "record_count": len(raw),
        },
        "method_reports": aggregate_density_results(
            raw, policy_ids=policy_ids, method_ids=method_ids
        ),
        "raw_record_set_hash": content_hash(raw),
        "raw_records_omitted_for_privacy": True,
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite density evidence: {args.output}")
    write_json(
        args.output,
        run_density_ablation(
            args.manifest,
            max_steps=args.max_steps,
            selected_policy_ids=(tuple(args.policy_ids) if args.policy_ids else None),
            method_ids=args.methods,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
