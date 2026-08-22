"""Audit collected P07 train-outcome records before RL/QD use.

This is a development-side data-quality audit.  It does not compute or
validate new paper metrics, but it aggregates the fields that decide whether
the next checkpoint can enter RL/QD analysis: four-agent utilization,
candidate diversity, vertical participation, real path length, safety,
physics/wall time, and coverage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import write_json_atomic  # noqa: E402
from aerocity_method.evaluation.hm3d_evidence_classification import (  # noqa: E402
    normalize_p07_record_purpose,
)

AUDIT_SCHEMA_VERSION = "hm3d-p07-outcome-audit-v1"


def _record_concise(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    failed = str(payload.get("status", "")).endswith("FAILED")
    mobility = payload.get("mobility_summary")
    mobility = mobility if isinstance(mobility, dict) else {}
    metrics = payload.get("metric_report")
    metrics = metrics if isinstance(metrics, dict) else {}
    runtime = payload.get("runtime_performance")
    runtime = runtime if isinstance(runtime, dict) else {}
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        if failed:
            decisions = []
        else:
            raise ValueError(f"record lacks decisions: {path}")
    transitions = payload.get("single_rl_training_transitions")
    if not isinstance(transitions, list):
        if failed:
            transitions = []
        else:
            raise ValueError(f"record lacks single_rl transitions: {path}")
    reservation = payload.get("task_reservation")
    reservation = reservation if isinstance(reservation, dict) else {}
    reservation_decisions = reservation.get("decisions")
    if not isinstance(reservation_decisions, list):
        if failed:
            reservation_decisions = []
        else:
            raise ValueError(f"record lacks task reservation decisions: {path}")

    selected_ids: list[str] = []
    pool_hashes: set[str] = set()
    four_agent_decisions = 0
    safety_failures = 0
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"record decision is malformed: {path}")
        selection = decision.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"record selection is malformed: {path}")
        selected_ids.append(str(selection.get("selected_candidate_id")))
        pool_hash = decision.get("public_candidate_pool_hash")
        if isinstance(pool_hash, str):
            pool_hashes.add(pool_hash)
        execution = decision.get("execution")
        if isinstance(execution, dict):
            safety_failures += int(
                execution.get("collision_count", 0)
                + execution.get("failed_fragment_count", 0)
                + execution.get("inter_agent_separation_violation_count", 0)
                + execution.get("out_of_bounds_count", 0)
                + execution.get("static_clearance_contract_violation_count", 0)
            )
        rd = reservation_decisions[index] if index < len(reservation_decisions) else None
        if isinstance(rd, dict):
            reservations_after = rd.get("reservations_after")
            if isinstance(reservations_after, list) and len(reservations_after) == 4:
                four_agent_decisions += 1

    return {
        "path": str(path),
        "runtime_record_sha256": payload.get("runtime_record_sha256"),
        "scene_id": payload.get("scene_id"),
        "strategy": payload.get("strategy"),
        "status": payload.get("status"),
        "failure_reason": payload.get("status_reason") or payload.get("terminal_outcome"),
        "record_purpose": normalize_p07_record_purpose(payload),
        "terminal_outcome": payload.get("terminal_outcome"),
        "decision_count": len(decisions),
        "transition_count": len(transitions),
        "elapsed_physics_s": float(payload.get("elapsed_physics_s", 0.0)),
        "wall_s": float(runtime.get("total_wall_s", 0.0)),
        "auc": float(metrics.get("explored_free_flight_volume_auc_time", 0.0)),
        "final_coverage": float(metrics.get("final_coverage_at_budget", 0.0)),
        "final_explored_volume_m3": float(
            metrics.get("final_explored_free_volume_m3", 0.0)
        ),
        "planned_fleet_path_m": float(mobility.get("planned_fleet_path_length_m", 0.0)),
        "realised_fleet_path_m": float(mobility.get("realised_fleet_path_length_m", 0.0)),
        "mean_realised_path_per_agent_m": float(
            mobility.get("mean_realised_path_length_per_agent_m", 0.0)
        ),
        "completed_vertical_agent_count": int(
            mobility.get("completed_vertical_agent_count", 0)
        ),
        "cross_height_band_agent_count": int(
            mobility.get("cross_height_band_agent_count", 0)
        ),
        "transit_completed_count": int(mobility.get("transit_completed_count", 0)),
        "transit_completion_fraction": float(
            mobility.get("transit_completion_fraction", 0.0)
        ),
        "four_agent_decision_count": four_agent_decisions,
        "unique_candidate_pool_hash_count": len(pool_hashes),
        "unique_selected_candidate_id_count": len(set(selected_ids)),
        "safety_failure_count": safety_failures,
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = sorted({record["strategy"] for record in records})
    summaries: dict[str, Any] = {}
    for strategy in strategies:
        rows = [record for record in records if record["strategy"] == strategy]
        summaries[strategy] = {
            "episode_count": len(rows),
            "decision_count": sum(int(record["decision_count"]) for record in rows),
            "transition_count": sum(int(record["transition_count"]) for record in rows),
            "mean_auc": _mean([record["auc"] for record in rows]),
            "mean_final_coverage": _mean([record["final_coverage"] for record in rows]),
            "mean_realised_fleet_path_m": _mean(
                [record["realised_fleet_path_m"] for record in rows]
            ),
            "mean_planned_fleet_path_m": _mean(
                [record["planned_fleet_path_m"] for record in rows]
            ),
            "completed_vertical_agent_episodes": sum(
                record["completed_vertical_agent_count"] > 0 for record in rows
            ),
            "cross_height_band_episodes": sum(
                record["cross_height_band_agent_count"] > 0 for record in rows
            ),
            "four_agent_decision_fraction": _mean(
                [
                    record["four_agent_decision_count"] / record["decision_count"]
                    for record in rows
                    if record["decision_count"] > 0
                ]
            ),
            "unique_candidate_pool_hash_sum": sum(
                record["unique_candidate_pool_hash_count"] for record in rows
            ),
            "unique_selected_candidate_id_sum": sum(
                record["unique_selected_candidate_id_count"] for record in rows
            ),
            "safety_failure_count": sum(
                record["safety_failure_count"] for record in rows
            ),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "episode_count": len(records),
        "real_decision_count": sum(int(record["decision_count"]) for record in records),
        "transition_count": sum(int(record["transition_count"]) for record in records),
        "total_physics_s": sum(float(record["elapsed_physics_s"]) for record in records),
        "total_wall_s": sum(float(record["wall_s"]) for record in records),
        "failed_episode_count": sum(
            record["status"] != "P07_EXECUTION_SMOKE_COMPLETE"
            or record["terminal_outcome"] != "budget_exhausted"
            for record in records
        ),
        "strategy_summaries": summaries,
        "records": sorted(records, key=lambda record: str(record["path"])),
    }


def _record_paths(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        name = path.name
        if (
            name.startswith(".")
            or "manifest" in name
            or "batch_smoke" in name
            or "audit" in name
            or "progress" in name
            or "verified_fix" in name
        ):
            continue
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _record_paths(args.record_dir)
    if not paths:
        raise ValueError("no P07 outcome JSON files found in record-dir")
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(_record_concise(path, payload))
    report = _build_report(records)
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "episode_count": report["episode_count"],
                "real_decision_count": report["real_decision_count"],
                "transition_count": report["transition_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
