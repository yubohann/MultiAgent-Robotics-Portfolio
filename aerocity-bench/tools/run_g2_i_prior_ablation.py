"""Run a paired coarse-region versus full-cell G2-I L0 ablation.

This is a development-only representation ablation.  Both public policies run
against the same regenerated private episode and strict full-atlas evaluator,
while the coarse policy receives no cell, pose, surface point, or normal data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import create_baseline
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.inspection_atlas import (
    ATLAS_PRIOR_COARSE,
    validate_public_mission_sector,
)
from aerocity_bench.metrics import evaluate_run
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.targets_v3 import public_episode_projection

try:
    from tools.run_g2_i_l0_calibration import _calibration_implementation_hash
except ModuleNotFoundError:
    from run_g2_i_l0_calibration import _calibration_implementation_hash

MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"
REPORT_SCHEMA = "org.aerocity.bench.g2-i-prior-ablation.v1"
PRIOR_LEVELS = ("coarse-regions", "full-cells")


def _prior_implementation_hash() -> str:
    return content_hash(
        {
            "base_calibration": _calibration_implementation_hash(),
            "prior_runner": file_hash(Path(__file__)),
        }
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--prior-level", action="append", choices=PRIOR_LEVELS, dest="prior_levels"
    )
    args = parser.parse_args(argv)
    if args.prior_levels and len(set(args.prior_levels)) != len(args.prior_levels):
        parser.error("--prior-level values must not repeat")
    args.prior_levels = tuple(
        prior_level
        for prior_level in PRIOR_LEVELS
        if not args.prior_levels or prior_level in set(args.prior_levels)
    )
    return args


def _local_path(root: Path, value: object, field: str) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be relative to the manifest directory")
    resolved = (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{field} escapes the manifest directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {candidate}")
    return resolved


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema",
        "purpose",
        "self_method_results_used",
        "release_config_path",
        "records",
    }
    if manifest.get("schema") != MANIFEST_SCHEMA or not required.issubset(manifest):
        raise ValueError("unsupported G2-I calibration manifest")
    if manifest["purpose"] != "method-independent-task-calibration":
        raise ValueError("prior ablation requires method-independent calibration inputs")
    if manifest["self_method_results_used"] is not False:
        raise ValueError("prior ablation cannot consume self-method results")
    declared_hash = manifest.get("manifest_hash")
    if declared_hash is not None:
        payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if content_hash(payload) != declared_hash:
            raise ValueError("calibration manifest hash mismatch")


def _run_policy(
    config: Any,
    city: dict[str, Any],
    private_episode: dict[str, Any],
    full_task: dict[str, Any],
    runtime_public_episode: dict[str, Any],
    policy: Any,
    *,
    prior_level: str,
    method_id: str,
    max_steps: int | None,
) -> dict[str, Any]:
    runtime = L0FleetRuntime(
        config,
        city,
        private_episode,
        receipt_secret=b"g2-i-prior-ablation-development-only-v1",
        public_task_spec=full_task,
        public_episode=runtime_public_episode,
    )
    result = runtime.run_policy(policy, max_steps=max_steps)
    metrics = evaluate_run(
        result,
        private_episode,
        float(config.raw["execution_contract"]["episode"]["duration_s"]),
    )
    return {
        "method_id": method_id,
        "prior_level": prior_level,
        "confirmation_count": int(metrics["quality"]["confirmed_count"]),
        "confirmed_recall_auc": float(metrics["quality"]["confirmed_recall_auc"]),
        "inspection_footprint_final": float(
            metrics["coverage_diagnostics"]["inspection_footprint_final"] or 0.0
        ),
        "inspection_footprint_auc": float(
            metrics["coverage_diagnostics"]["inspection_footprint_auc"] or 0.0
        ),
        "returned_home_all": all(bool(value) for value in result["returned_home"].values()),
        "collision_count": int(result["budget_ledger"]["collisions"]),
        "out_of_bounds_actions": int(result["budget_ledger"]["out_of_bounds_actions"]),
        "deadline_misses": int(result["budget_ledger"]["deadline_misses"]),
        "path_distance_m": float(result["budget_ledger"]["path_distance_m"]),
        "formal_score_eligible": False,
    }


def _validate_frozen_episode(
    episode: dict[str, Any], city: dict[str, Any], task_spec: dict[str, Any], config: Any
) -> None:
    """Fail closed instead of regenerating a manifest-bound private process."""

    if episode.get("schema") != "org.aerocity.bench.episode-private.ordinary.v3":
        raise ValueError("prior ablation episode schema differs")
    if (
        episode.get("layout_id") != city.get("layout_id")
        or episode.get("layout_hash") != city.get("layout_hash")
    ):
        raise ValueError("prior ablation episode is not bound to its city")
    if episode.get("execution_contract_hash") != content_hash(
        config.raw["execution_contract"]
    ):
        raise ValueError("prior ablation episode execution contract differs")
    stored_hash = episode.get("episode_hash")
    unhashed = dict(episode)
    unhashed.pop("episode_hash", None)
    if not isinstance(stored_hash, str) or content_hash(unhashed) != stored_hash:
        raise ValueError("prior ablation episode hash does not match its contents")
    sector = episode.get("mission_sector")
    if not isinstance(sector, dict):
        raise ValueError("prior ablation episode lacks a frozen mission sector")
    validate_public_mission_sector(
        sector,
        task_spec["inspection_atlas"],
        episode.get("starts"),
        config.raw["execution_contract"],
    )


def aggregate_prior_results(
    raw_results: list[dict[str, Any]], *, prior_levels: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Aggregate by independent ancestor while retaining every safety failure."""

    aggregate = []
    method_by_prior = {
        "coarse-regions": "atlas-coarse-region-inspector",
        "full-cells": "atlas-region-greedy",
    }
    for prior_level in prior_levels:
        method_id = method_by_prior[prior_level]
        rows = [row for row in raw_results if row["prior_level"] == prior_level]
        by_ancestor: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_ancestor.setdefault(row["layout_ancestor_hash"], []).append(row)
        if not by_ancestor:
            raise ValueError(f"prior ablation has no rows for {prior_level}")
        ancestor_means = [
            sum(item["confirmation_count"] for item in items) / len(items)
            for items in by_ancestor.values()
        ]
        aggregate.append(
            {
                "prior_level": prior_level,
                "method_id": method_id,
                "independent_ancestor_count": len(by_ancestor),
                "nonzero_ancestor_count": sum(value > 0.0 for value in ancestor_means),
                "mean_confirmation_count": sum(ancestor_means) / len(ancestor_means),
                "mean_inspection_footprint_final": sum(
                    sum(item["inspection_footprint_final"] for item in items)
                    / len(items)
                    for items in by_ancestor.values()
                )
                / len(by_ancestor),
                "collision_count": sum(int(item["collision_count"]) for item in rows),
                "out_of_bounds_actions": sum(
                    int(item["out_of_bounds_actions"]) for item in rows
                ),
                "all_returned_home": all(item["returned_home_all"] for item in rows),
                "deadline_misses": sum(int(item["deadline_misses"]) for item in rows),
            }
        )
    return aggregate


def run_ablation(
    manifest_path: Path,
    *,
    max_steps: int | None,
    prior_levels: tuple[str, ...] = PRIOR_LEVELS,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("G2-I calibration manifest must be an object")
    _validate_manifest(manifest)
    if not prior_levels or not set(prior_levels).issubset(PRIOR_LEVELS):
        raise ValueError("prior ablation requests an unknown prior level")
    root = manifest_path.parent
    config = load_ordinary_config(
        _local_path(root, manifest["release_config_path"], "release_config_path")
    )
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("G2-I calibration manifest has no records")

    city_cache: dict[str, dict[str, Any]] = {}
    task_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    raw: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("calibration record must be an object")
        split = str(record["split_label"])
        if split in FORMAL_SPLITS:
            raise ValueError("prior ablation must not inspect a formal split")
        city_path = _local_path(root, record["city_path"], "city_path")
        episode_path = _local_path(
            root, record["private_episode_path"], "private_episode_path"
        )
        city = city_cache.setdefault(str(city_path), read_json(city_path))
        private_episode = read_json(episode_path)
        if not isinstance(private_episode, dict):
            raise ValueError("calibration private episode must be an object")
        city_hash = content_hash(city)
        if city_hash not in task_cache:
            task_cache[city_hash] = (
                compile_g2_i_task_spec(
                    city, config.raw["execution_contract"], config.raw["fleet"]
                ),
                compile_g2_i_task_spec(
                    city,
                    config.raw["execution_contract"],
                    config.raw["fleet"],
                    inspection_prior_level=ATLAS_PRIOR_COARSE,
                ),
            )
        full_task, coarse_task = task_cache[city_hash]
        _validate_frozen_episode(private_episode, city, full_task, config)

        full_public = public_episode_projection(private_episode)
        full_policy = create_baseline(
            "atlas-region-greedy", config, full_task, full_public
        )
        coarse_public = {
            key: value
            for key, value in full_public.items()
            if key not in {"mission_sector", "mission_sector_hash"}
        }
        coarse_public["coarse_region_ids"] = list(
            full_public["mission_sector"]["selected_region_ids"]
        )
        coarse_policy = create_baseline(
            "atlas-coarse-region-inspector", config, coarse_task, coarse_public
        )
        ancestor_hash = content_hash(str(record["layout_ancestor"]))
        for prior_level in prior_levels:
            if prior_level == "coarse-regions":
                result = _run_policy(
                    config,
                    city,
                    private_episode,
                    full_task,
                    full_public,
                    coarse_policy,
                    prior_level="coarse-regions",
                    method_id="atlas-coarse-region-inspector",
                    max_steps=max_steps,
                )
            else:
                result = _run_policy(
                    config,
                    city,
                    private_episode,
                    full_task,
                    full_public,
                    full_policy,
                    prior_level="full-cells",
                    method_id="atlas-region-greedy",
                    max_steps=max_steps,
                )
            raw.append({**result, "layout_ancestor_hash": ancestor_hash})

    aggregate = aggregate_prior_results(raw, prior_levels=prior_levels)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "DIAGNOSTIC_ONLY",
        "manifest_hash": content_hash(manifest),
        "base_calibration_implementation_hash": _calibration_implementation_hash(),
        "prior_implementation_hash": _prior_implementation_hash(),
        "execution_level": "L0",
        "contract": {
            "declared_prior_levels": list(PRIOR_LEVELS),
            "selected_prior_levels": list(prior_levels),
            "complete_prior_set": prior_levels == PRIOR_LEVELS,
            "same_private_episode_per_pair": True,
            "same_strict_full_atlas_evaluator_per_pair": True,
            "coarse_policy_receives_cells_or_poses": False,
            "target_truth_visible_to_public_policy": False,
            "paired_execution_ablation_not_method_ranking": True,
            "l0_not_a_native_or_formal_score": True,
        },
        "aggregate": aggregate,
        "raw_record_set_hash": content_hash(raw),
        "raw_records_omitted_for_privacy": True,
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence: {args.output}")
    report = run_ablation(
        args.manifest,
        max_steps=args.max_steps,
        prior_levels=args.prior_levels,
    )
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
