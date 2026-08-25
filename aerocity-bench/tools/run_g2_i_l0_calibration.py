"""Run method-independent G2-I L0 public-searchability calibration.

The output is an aggregate diagnostic only. It deliberately omits episode IDs,
target IDs, target counts, confirmation handles, and route coordinates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from aerocity_bench import baselines as baselines_module
from aerocity_bench import compiler as compiler_module
from aerocity_bench import contracts as contracts_module
from aerocity_bench import evaluator as evaluator_module
from aerocity_bench import geometry as geometry_module
from aerocity_bench import inspection_atlas as inspection_atlas_module
from aerocity_bench import metrics as metrics_module
from aerocity_bench import ordinary_config as ordinary_config_module
from aerocity_bench import runtime as runtime_module
from aerocity_bench import targets_v3 as targets_module
from aerocity_bench.baselines import BASELINES, create_baseline
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.metrics import evaluate_run
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.targets_v3 import (
    public_episode_projection,
    validate_frozen_g2_i_episode,
)

REPORT_SCHEMA = "org.aerocity.bench.g2-i-l0-searchability-calibration.v1"
MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"
METHODS = ("atlas-surface-inspector", "atlas-region-greedy", "centralized-oracle")


def _calibration_implementation_hash() -> str:
    paths = {
        "runner": Path(__file__),
        "baselines": Path(str(baselines_module.__file__)),
        "compiler": Path(str(compiler_module.__file__)),
        "contracts": Path(str(contracts_module.__file__)),
        "evaluator": Path(str(evaluator_module.__file__)),
        "geometry": Path(str(geometry_module.__file__)),
        "inspection_atlas": Path(str(inspection_atlas_module.__file__)),
        "metrics": Path(str(metrics_module.__file__)),
        "ordinary_config": Path(str(ordinary_config_module.__file__)),
        "runtime": Path(str(runtime_module.__file__)),
        "targets": Path(str(targets_module.__file__)),
    }
    return content_hash({name: file_hash(path) for name, path in sorted(paths.items())})


def _local_path(root: Path, value: object, field: str) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be a relative path inside the manifest root")
    resolved = (root / candidate).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"{field} escapes the manifest root")
    return resolved


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--method", action="append", choices=METHODS, dest="methods")
    args = parser.parse_args(argv)
    if args.methods and len(set(args.methods)) != len(args.methods):
        parser.error("--method values must not repeat")
    args.methods = tuple(
        method_id
        for method_id in METHODS
        if not args.methods or method_id in set(args.methods)
    )
    return args


def _run_record(
    config: Any,
    city: dict[str, Any],
    private_episode: dict[str, Any],
    task_spec: dict[str, Any],
    *,
    method_id: str,
    max_steps: int | None,
) -> dict[str, Any]:
    public_episode = public_episode_projection(private_episode)
    descriptor = BASELINES[method_id]
    policy = create_baseline(
        method_id,
        config,
        task_spec,
        public_episode,
        private_episode=private_episode if descriptor.requires_private_truth else None,
    )
    runtime = L0FleetRuntime(
        config,
        city,
        private_episode,
        receipt_secret=b"g2-i-method-independent-l0-calibration-v1",
        public_task_spec=task_spec,
        public_episode=public_episode,
    )
    result = runtime.run_policy(policy, max_steps=max_steps)
    duration = float(config.raw["execution_contract"]["episode"]["duration_s"])
    metrics = evaluate_run(result, private_episode, duration)
    ledger = result["budget_ledger"]
    returned_home = all(bool(value) for value in result["returned_home"].values())
    selection = getattr(policy, "public_selection_contract", None)
    return {
        "method_id": method_id,
        "layout_hash": content_hash(city),
        "atlas_hash": task_spec["inspection_atlas"]["atlas_hash"],
        "execution_contract_hash": result["execution_contract_hash"],
        "confirmation_count": int(metrics["quality"]["confirmed_count"]),
        "final_confirmed_recall": float(metrics["quality"]["final_confirmed_recall"]),
        "confirmed_recall_auc": float(metrics["quality"]["confirmed_recall_auc"]),
        "inspection_footprint_final": metrics["coverage_diagnostics"][
            "inspection_footprint_final"
        ],
        "inspection_footprint_auc": metrics["coverage_diagnostics"][
            "inspection_footprint_auc"
        ],
        "returned_home_all": returned_home,
        "collision_count": int(ledger["collisions"]),
        "out_of_bounds_actions": int(ledger["out_of_bounds_actions"]),
        "deadline_misses": int(ledger["deadline_misses"]),
        "task_time_s": float(result["task_time_s"]),
        "wall_clock_s": float(result["wall_clock_s"] or 0.0),
        "selected_observe_pose_count": (
            int(selection["selected_observe_pose_count"])
            if isinstance(selection, dict)
            else sum(len(indices) for indices in getattr(policy, "observe_indices", {}).values())
        ),
        "formal_score_eligible": False,
    }


def _validate_frozen_episode(
    episode: dict[str, Any],
    city: dict[str, Any],
    task_spec: dict[str, Any],
    config: Any,
) -> None:
    """Validate a manifest episode without re-sampling private truth.

    A calibration manifest is a content-addressed replay input.  Re-sampling an
    episode at run time makes a previously admitted manifest depend on current
    generator code and can reject or silently replace the frozen private
    process.  The runner therefore validates the stored episode and executes it
    verbatim.
    """

    validate_frozen_g2_i_episode(
        episode, city, task_spec, config.raw["execution_contract"]
    )


def _assemble_report(
    *,
    calibration_manifest_hash: str,
    calibration_implementation_hash: str,
    episode_duration_s: float,
    max_steps: int | None,
    record_count: int,
    method_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    reports_by_method = {
        str(report["method_id"]): report for report in method_reports
    }
    methods = tuple(
        method_id for method_id in METHODS if method_id in reports_by_method
    )
    if len(methods) != len(method_reports):
        raise ValueError("calibration method reports repeat or use an unknown method")
    ancestor_counts = {int(report["ancestor_count"]) for report in method_reports}
    if len(ancestor_counts) != 1:
        raise ValueError("calibration method shards use different ancestor counts")
    independent_ancestor_count = next(iter(ancestor_counts))
    stable_ancestor_threshold = max(1, math.ceil(independent_ancestor_count * 0.8))
    non_oracle = [
        reports_by_method[method_id]
        for method_id in methods
        if not reports_by_method[method_id]["requires_private_truth"]
    ]
    stable_non_oracle_methods = [
        str(report["method_id"])
        for report in non_oracle
        if int(report["nonzero_ancestor_count"]) >= stable_ancestor_threshold
        and 0.0 < float(report["mean_final_confirmed_recall"]) < 0.95
    ]
    oracle = next(
        (
            reports_by_method[method_id]
            for method_id in methods
            if reports_by_method[method_id]["requires_private_truth"]
        ),
        None,
    )
    oracle_feasible = (
        all(
            ancestor["all_returned_home"] and ancestor["collision_count"] == 0
            for ancestor in oracle["ancestors"]
        )
        if oracle is not None
        else None
    )
    complete_method_set = methods == METHODS
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_ONLY",
        "method_independence": {
            "purpose": "method-independent-task-calibration",
            "self_method_results_used": False,
        },
        "contract": {
            "calibration_manifest_hash": calibration_manifest_hash,
            "calibration_implementation_hash": calibration_implementation_hash,
            "episode_duration_s": episode_duration_s,
            "max_steps": max_steps,
            "methods": list(methods),
            "target_count_visible_to_method": False,
            "l0_not_a_native_or_formal_score": True,
            "mission_sector_required": True,
            "mission_sector_frozen_before_private_sampling": True,
        },
        "aggregate": {
            "record_count": record_count,
            "independent_ancestor_count": independent_ancestor_count,
            "method_count": len(methods),
        },
        "method_reports": [reports_by_method[method_id] for method_id in methods],
        "searchability_gate": {
            "complete_method_set": complete_method_set,
            "oracle_feasible_and_returns": oracle_feasible,
            "non_oracle_has_nonzero_confirmation": any(
                int(report["nonzero_ancestor_count"]) > 0 for report in non_oracle
            ),
            "stable_ancestor_threshold": stable_ancestor_threshold,
            "stable_nonsaturated_non_oracle_methods": stable_non_oracle_methods,
            "stable_nonsaturated_non_oracle": bool(stable_non_oracle_methods),
            "requires_stable_nonsaturated_non_oracle": True,
            "formal_gate_status": (
                "OPEN_L1_EXTERNAL_AND_FREEZE_REQUIRED"
                if complete_method_set
                else "SHARD_ONLY_MERGE_REQUIRED"
            ),
        },
    }
    report["report_hash"] = content_hash(report)
    return report


def run_calibration(
    manifest_path: Path,
    *,
    max_steps: int | None,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("calibration manifest schema differs")
    if manifest.get("purpose") != "method-independent-task-calibration":
        raise ValueError("calibration manifest purpose differs")
    if manifest.get("self_method_results_used") is not False:
        raise ValueError("calibration must not use self-method results")
    root = manifest_path.parent.resolve()
    config = load_ordinary_config(
        _local_path(root, manifest["release_config_path"], "release_config_path")
    )
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("calibration manifest has no records")
    raw_results: list[dict[str, Any]] = []
    task_spec_cache: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("calibration record must be an object")
        city = read_json(_local_path(root, record["city_path"], "city_path"))
        private_episode = read_json(
            _local_path(root, record["private_episode_path"], "private_episode_path")
        )
        if str(city.get("split")) in FORMAL_SPLITS:
            raise ValueError("G2-I task calibration must not inspect a formal split")
        city_hash = content_hash(city)
        task_spec = task_spec_cache.get(city_hash)
        if task_spec is None:
            task_spec = compile_g2_i_task_spec(
                city, config.raw["execution_contract"], config.raw["fleet"]
            )
            task_spec_cache[city_hash] = task_spec
        _validate_frozen_episode(private_episode, city, task_spec, config)
        for method_id in methods:
            raw_results.append(
                _run_record(
                    config,
                    city,
                    private_episode,
                    task_spec,
                    method_id=method_id,
                    max_steps=max_steps,
                )
            )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for result in raw_results:
        grouped.setdefault(result["method_id"], {}).setdefault(
            result["layout_hash"], []
        ).append(result)
    method_reports = []
    for method_id in methods:
        ancestors = []
        for layout_hash, rows in sorted(grouped[method_id].items()):
            ancestors.append(
                {
                    "layout_hash": layout_hash,
                    "episode_count": len(rows),
                    "mean_confirmation_count": sum(
                        row["confirmation_count"] for row in rows
                    )
                    / len(rows),
                    "mean_final_confirmed_recall": sum(
                        row["final_confirmed_recall"] for row in rows
                    )
                    / len(rows),
                    "mean_confirmed_recall_auc": sum(
                        row["confirmed_recall_auc"] for row in rows
                    )
                    / len(rows),
                    "mean_inspection_footprint_final": sum(
                        float(row["inspection_footprint_final"] or 0.0) for row in rows
                    )
                    / len(rows),
                    "mean_inspection_footprint_auc": sum(
                        float(row["inspection_footprint_auc"] or 0.0) for row in rows
                    )
                    / len(rows),
                    "all_returned_home": all(row["returned_home_all"] for row in rows),
                    "collision_count": sum(row["collision_count"] for row in rows),
                    "out_of_bounds_actions": sum(
                        row["out_of_bounds_actions"] for row in rows
                    ),
                    "deadline_misses": sum(row["deadline_misses"] for row in rows),
                    "mean_task_time_s": sum(row["task_time_s"] for row in rows)
                    / len(rows),
                    "mean_wall_clock_s": sum(row["wall_clock_s"] for row in rows)
                    / len(rows),
                    "mean_selected_observe_pose_count": sum(
                        row["selected_observe_pose_count"] for row in rows
                    )
                    / len(rows),
                }
            )
        confirmation_means = [
            ancestor["mean_confirmation_count"] for ancestor in ancestors
        ]
        method_reports.append(
            {
                "method_id": method_id,
                "observation_profile": BASELINES[method_id].observation_profile,
                "requires_private_truth": BASELINES[method_id].requires_private_truth,
                "ancestor_count": len(ancestors),
                "ancestors": ancestors,
                "nonzero_ancestor_count": sum(value > 0.0 for value in confirmation_means),
                "mean_confirmation_count": sum(confirmation_means) / len(confirmation_means),
                "mean_final_confirmed_recall": sum(
                    float(ancestor["mean_final_confirmed_recall"])
                    for ancestor in ancestors
                )
                / len(ancestors),
            }
        )
    return _assemble_report(
        calibration_manifest_hash=content_hash(manifest),
        calibration_implementation_hash=_calibration_implementation_hash(),
        episode_duration_s=float(
            config.raw["execution_contract"]["episode"]["duration_s"]
        ),
        max_steps=max_steps,
        record_count=len(records),
        method_reports=method_reports,
    )


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {args.output}")
    report = run_calibration(
        args.manifest, max_steps=args.max_steps, methods=args.methods
    )
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
