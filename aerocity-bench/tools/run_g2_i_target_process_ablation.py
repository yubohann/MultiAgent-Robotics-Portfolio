"""Measure public G2-I methods across frozen paired target processes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import BASELINES
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config

try:
    # Package import is used by the regression suite; direct execution keeps
    # working for the documented command-line workflow.
    from tools.run_g2_i_l0_calibration import (
        MANIFEST_SCHEMA,
        _calibration_implementation_hash,
        _local_path,
        _run_record,
        _validate_frozen_episode,
    )
except ModuleNotFoundError:
    from run_g2_i_l0_calibration import (
        MANIFEST_SCHEMA,
        _calibration_implementation_hash,
        _local_path,
        _run_record,
        _validate_frozen_episode,
    )

REPORT_SCHEMA = "org.aerocity.bench.g2-i-target-process-performance-ablation.v1"
PUBLIC_METHODS = ("atlas-surface-inspector", "atlas-region-greedy")


def _target_process_implementation_hash() -> str:
    return content_hash(
        {
            "base_calibration": _calibration_implementation_hash(),
            "target_process_runner": file_hash(Path(__file__)),
        }
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--method", action="append", choices=PUBLIC_METHODS, dest="methods")
    args = parser.parse_args(argv)
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
        raise ValueError(f"cannot aggregate an empty target-process group: {field}")
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_target_process_results(
    raw_results: list[dict[str, Any]],
    *,
    expected_processes: tuple[str, ...],
    method_ids: tuple[str, ...] = PUBLIC_METHODS,
) -> list[dict[str, Any]]:
    """Aggregate paired layout results; episode replicates never become samples."""

    if not raw_results:
        raise ValueError("target-process ablation has no execution results")
    expected = set(expected_processes)
    if not expected:
        raise ValueError("target-process ablation requires declared processes")
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    ancestors: set[str] = set()
    for row in raw_results:
        key = (
            str(row["method_id"]),
            str(row["layout_ancestor"]),
            str(row["target_process"]),
        )
        if key in indexed:
            raise ValueError("target-process ablation has duplicate method-condition rows")
        indexed[key] = row
        ancestors.add(key[1])
    for method_id in method_ids:
        if method_id not in BASELINES or BASELINES[method_id].requires_private_truth:
            raise ValueError("target-process ablation requires public methods")
        for ancestor in ancestors:
            observed = {
                process
                for candidate_method, candidate_ancestor, process in indexed
                if candidate_method == method_id and candidate_ancestor == ancestor
            }
            if observed != expected:
                raise ValueError("every ancestor must contain one row per target process")

    reports = []
    for method_id in method_ids:
        per_process = []
        for process in sorted(expected):
            rows = [
                indexed[(method_id, ancestor, process)] for ancestor in sorted(ancestors)
            ]
            per_process.append(
                {
                    "target_process": process,
                    "independent_ancestor_count": len(rows),
                    "nonzero_ancestor_count": sum(
                        int(row["confirmation_count"]) > 0 for row in rows
                    ),
                    "mean_confirmation_count": _mean(rows, "confirmation_count"),
                    "mean_final_confirmed_recall": _mean(
                        rows, "final_confirmed_recall"
                    ),
                    "mean_confirmed_recall_auc": _mean(rows, "confirmed_recall_auc"),
                    "mean_inspection_footprint_final": _mean(
                        rows, "inspection_footprint_final"
                    ),
                    "all_returned_home": all(row["returned_home_all"] for row in rows),
                    "collision_count": sum(int(row["collision_count"]) for row in rows),
                    "out_of_bounds_actions": sum(
                        int(row["out_of_bounds_actions"]) for row in rows
                    ),
                    "deadline_misses": sum(int(row["deadline_misses"]) for row in rows),
                }
            )
        reference = "uniform_surface" if "uniform_surface" in expected else sorted(expected)[0]
        paired_deltas = []
        for process in sorted(expected - {reference}):
            deltas = []
            for ancestor in sorted(ancestors):
                condition = indexed[(method_id, ancestor, process)]
                baseline = indexed[(method_id, ancestor, reference)]
                deltas.append(
                    float(condition["final_confirmed_recall"])
                    - float(baseline["final_confirmed_recall"])
                )
            paired_deltas.append(
                {
                    "comparison": f"{process}_minus_{reference}",
                    "independent_ancestor_count": len(deltas),
                    "mean_final_confirmed_recall_delta": sum(deltas) / len(deltas),
                }
            )
        reports.append(
            {
                "method_id": method_id,
                "observation_profile": BASELINES[method_id].observation_profile,
                "requires_private_truth": False,
                "by_target_process": per_process,
                "paired_final_recall_deltas": paired_deltas,
            }
        )
    return reports


def run_target_process_ablation(
    manifest_path: Path,
    *,
    max_steps: int | None,
    method_ids: tuple[str, ...] = PUBLIC_METHODS,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported G2-I calibration manifest")
    if manifest.get("purpose") != "method-independent-task-calibration":
        raise ValueError("target-process ablation requires method-independent inputs")
    if manifest.get("self_method_results_used") is not False:
        raise ValueError("target-process ablation cannot consume self-method results")
    if not method_ids or not set(method_ids).issubset(PUBLIC_METHODS):
        raise ValueError("target-process ablation requests an unknown public method")
    root = manifest_path.parent
    config = load_ordinary_config(
        _local_path(root, manifest["release_config_path"], "release_config_path")
    )
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("target-process ablation manifest has no records")
    task_cache: dict[str, dict[str, Any]] = {}
    raw: list[dict[str, Any]] = []
    observed_processes: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("target-process ablation record must be an object")
        city = read_json(_local_path(root, record["city_path"], "city_path"))
        episode = read_json(
            _local_path(root, record["private_episode_path"], "private_episode_path")
        )
        if str(city.get("split")) in FORMAL_SPLITS:
            raise ValueError("target-process ablation must not inspect a formal split")
        city_hash = content_hash(city)
        task_spec = task_cache.get(city_hash)
        if task_spec is None:
            task_spec = compile_g2_i_task_spec(
                city, config.raw["execution_contract"], config.raw["fleet"]
            )
            task_cache[city_hash] = task_spec
        _validate_frozen_episode(episode, city, task_spec, config)
        target_process = str(episode.get("target_process", ""))
        if not target_process:
            raise ValueError("frozen episode lacks a target process label")
        observed_processes.add(target_process)
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
                    "layout_ancestor": str(record["layout_ancestor"]),
                    "target_process": target_process,
                }
            )
    expected_processes = tuple(config.target_processes("calibration"))
    if observed_processes != set(expected_processes):
        raise ValueError("frozen calibration manifest lacks complete target-process pairs")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_ONLY",
        "manifest_hash": content_hash(manifest),
        "base_calibration_implementation_hash": _calibration_implementation_hash(),
        "target_process_implementation_hash": _target_process_implementation_hash(),
        "execution_level": "L0",
        "contract": {
            "declared_method_ids": list(PUBLIC_METHODS),
            "selected_method_ids": list(method_ids),
            "complete_method_set": method_ids == PUBLIC_METHODS,
            "frozen_private_episode_replayed": True,
            "target_truth_visible_to_methods": False,
            "target_process_labels_are_internal_calibration_conditions": True,
            "paired_by_layout_ancestor": True,
            "episode_replicates_are_not_independent": True,
            "l0_not_a_native_or_formal_score": True,
        },
        "aggregate": {
            "independent_ancestor_count": len(
                {str(row["layout_ancestor"]) for row in raw}
            ),
            "target_processes": sorted(observed_processes),
            "method_count": len(method_ids),
            "record_count": len(raw),
        },
        "method_reports": aggregate_target_process_results(
            raw,
            expected_processes=expected_processes,
            method_ids=method_ids,
        ),
        "raw_record_set_hash": content_hash(raw),
        "raw_records_omitted_for_privacy": True,
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite target-process evidence: {args.output}"
        )
    write_json(
        args.output,
        run_target_process_ablation(
            args.manifest,
            max_steps=args.max_steps,
            method_ids=args.methods,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
