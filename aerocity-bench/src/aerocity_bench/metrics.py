"""Transparent quality, safety, resource, and collaboration metrics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from .errors import ValidationError
from .isaac_bridge import FormalExecutionContext, assert_formal_receipts


def confirmed_recall_auc(
    confirmation_times_s: Iterable[float], target_count: int, duration_s: float
) -> float:
    if target_count <= 0 or duration_s <= 0:
        raise ValueError("target count and duration must be positive")
    times = sorted(max(0.0, min(duration_s, float(value))) for value in confirmation_times_s)
    if len(times) > target_count:
        raise ValueError("confirmation count exceeds the frozen target count")
    return sum(duration_s - value for value in times) / (target_count * duration_s)


def time_to_recall(
    confirmation_times_s: Iterable[float],
    target_count: int,
    fraction: float,
    duration_s: float,
) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("recall fraction must lie in (0, 1]")
    required = math.ceil(target_count * fraction)
    times = sorted(float(value) for value in confirmation_times_s)
    if len(times) >= required:
        return {"time_s": times[required - 1], "right_censored": False, "required_count": required}
    return {"time_s": duration_s, "right_censored": True, "required_count": required}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _coverage_auc(
    trace: list[list[float]], index: int, duration_s: float, denominator: float
) -> float:
    if not trace or duration_s <= 0 or denominator <= 0:
        return 0.0
    area = 0.0
    previous_time = 0.0
    previous_value = 0.0
    for item in trace:
        timestamp = max(previous_time, min(duration_s, float(item[0])))
        value = max(previous_value, min(1.0, float(item[index]) / denominator))
        area += (timestamp - previous_time) * previous_value
        previous_time, previous_value = timestamp, value
    area += max(0.0, duration_s - previous_time) * previous_value
    return area / duration_s


def _group_recall(
    targets: list[dict[str, Any]],
    confirmed_target_ids: set[str],
    group_for_target: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    counts: dict[str, list[int]] = {}
    for target in targets:
        target_id = str(target["target_id"])
        group = group_for_target[target_id]
        bucket = counts.setdefault(group, [0, 0])
        bucket[1] += 1
        if target_id in confirmed_target_ids:
            bucket[0] += 1
    return {
        group: {
            "confirmed": confirmed,
            "total": total,
            "recall": confirmed / total,
        }
        for group, (confirmed, total) in sorted(counts.items())
    }


def _jain(values: list[float]) -> float:
    if not values or all(value == 0 for value in values):
        return 1.0
    squared_sum = sum(values) ** 2
    return squared_sum / (len(values) * sum(value * value for value in values))


def _gini(values: list[float]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    count = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (count * sum(ordered)) - (count + 1.0) / count


def evaluate_run(
    run_result: dict[str, Any],
    private_episode: dict[str, Any],
    duration_s: float,
    *,
    formal_context: FormalExecutionContext | None = None,
) -> dict[str, Any]:
    audit = run_result["evaluator_private_audit"]
    if audit["episode_id"] != private_episode["episode_id"]:
        raise ValueError("run and private episode IDs differ")
    validity = private_episode["target_validity"]
    if audit["validity_hash"] != validity["validity_hash"]:
        raise ValueError("run used another target-validity denominator")
    private_episode_id = str(private_episode["episode_id"])
    private_layout_id = str(private_episode.get("layout_id", ""))
    starts = private_episode.get("starts")
    if not isinstance(starts, list) or not starts:
        raise ValidationError("private episode has no trusted start roster")
    expected_drone_ids = {str(start.get("drone_id", "")) for start in starts}
    if not expected_drone_ids or "" in expected_drone_ids or len(expected_drone_ids) != len(starts):
        raise ValidationError("private episode start roster is invalid")
    target_count = int(private_episode["target_count"])
    confirmation_times = [float(value) for value in audit["confirmation_times_s"]]
    confirmed_count = len(confirmation_times)
    confirmation_records = list(audit.get("confirmation_records_private", []))
    if len(confirmation_records) != confirmed_count:
        raise ValueError("private confirmation attribution is incomplete")
    confirmed_target_ids = {str(record["target_id"]) for record in confirmation_records}
    if len(confirmed_target_ids) != confirmed_count:
        raise ValueError("private confirmation attribution contains duplicates")
    valid_target_ids = {str(target["target_id"]) for target in private_episode["targets"]}
    if not confirmed_target_ids <= valid_target_ids:
        raise ValueError("private confirmation attribution names an invalid target")
    final_recall = confirmed_count / target_count
    receipts = run_result["execution_receipts"]
    latencies = [float(receipt["planning_latency_s"]) for receipt in receipts]
    ledger = run_result["budget_ledger"]
    returned = run_result["returned_home"]
    collision_count = int(ledger["collisions"])
    formal_claimed = run_result["execution_level"] == "L1" and bool(
        run_result["formal_score_eligible"]
    )
    if formal_claimed:
        if formal_context is None:
            raise ValidationError(
                "formal L1 scoring requires trusted in-memory native execution context"
            )
        if (
            formal_context.episode_id != private_episode_id
            or formal_context.layout_id != private_layout_id
        ):
            raise ValidationError("formal native context does not match private episode")
        expected_bindings = {
            "episode_id": formal_context.episode_id,
            "layout_id": formal_context.layout_id,
            "execution_contract_hash": formal_context.execution_contract_hash,
            "native_gate_hash": formal_context.native_gate_hash,
            "runtime_fingerprint_hash": formal_context.runtime_fingerprint_hash,
            "execution_receipt_set_hash": formal_context.execution_receipt_set_hash,
        }
        mismatched = [
            key for key, value in expected_bindings.items() if run_result.get(key) != value
        ]
        if mismatched:
            raise ValidationError(f"formal L1 run differs from trusted context: {mismatched}")
        assert_formal_receipts(
            receipts,
            context=formal_context,
            expected_drone_ids=expected_drone_ids,
            expected_confirmation_ids={str(value) for value in audit["confirmation_ids"]},
            expected_task_time_s=float(run_result["task_time_s"]),
            ledger=ledger,
        )
    formal_eligible = formal_claimed and collision_count == 0
    targets = list(private_episode["targets"])
    support_groups = _group_recall(
        targets,
        confirmed_target_ids,
        {str(target["target_id"]): str(target["support_class"]) for target in targets},
    )
    altitude_groups = _group_recall(
        targets,
        confirmed_target_ids,
        {str(target["target_id"]): str(target["altitude_band"]) for target in targets},
    )
    witness_counts = [
        int(target.get("legal_witness_count", len(target["legal_witnesses"]))) for target in targets
    ]
    lower_witness = float(_percentile([float(value) for value in witness_counts], 1.0 / 3.0) or 0)
    upper_witness = float(_percentile([float(value) for value in witness_counts], 2.0 / 3.0) or 0)
    opportunity_labels = {}
    for target, count in zip(targets, witness_counts, strict=True):
        label = "low" if count <= lower_witness else "mid" if count <= upper_witness else "high"
        opportunity_labels[str(target["target_id"])] = label
    opportunity_groups = _group_recall(targets, confirmed_target_ids, opportunity_labels)
    group_recalls = [
        float(item["recall"])
        for groups in (support_groups, altitude_groups, opportunity_groups)
        for item in groups.values()
    ]
    coverage_denominators = run_result.get("coverage_denominators", {})
    two_d_denominator = int(coverage_denominators.get("coverage_2d_cells", 0))
    three_d_denominator = int(coverage_denominators.get("coverage_3d_free_cells", 0))
    if two_d_denominator <= 0 or three_d_denominator <= 0:
        raise ValueError("run lacks frozen positive coverage denominators")
    inspection_cell_denominator = int(
        coverage_denominators.get("inspection_atlas_cells", 0)
    )
    inspection_area_denominator = float(
        coverage_denominators.get("inspection_atlas_area_m2", 0.0)
    )
    inspection_trace = list(run_result.get("inspection_coverage_trace", []))
    inspection_cell_trace = list(run_result.get("inspection_cell_count_trace", []))
    if (inspection_cell_denominator > 0) != (inspection_area_denominator > 0.0):
        raise ValueError("G2-I run has inconsistent inspection denominators")
    if inspection_area_denominator > 0.0 and (
        not inspection_trace or not inspection_cell_trace
    ):
        raise ValueError("G2-I run lacks an inspection-footprint coverage trace")
    returned_ids = {str(value) for value in returned}
    if returned_ids != expected_drone_ids:
        raise ValidationError("run returned-home roster differs from private start roster")
    drone_ids = sorted(expected_drone_ids)
    distance_by_drone = {drone_id: 0.0 for drone_id in drone_ids}
    confirmations_by_drone = {drone_id: 0 for drone_id in drone_ids}
    simultaneous_confirmation_ids = {
        str(value) for value in audit.get("simultaneous_confirmation_ids", [])
    }
    for receipt in receipts:
        distance_by_drone[str(receipt["drone_id"])] += float(receipt["distance_m"])
    for record in confirmation_records:
        if str(record["confirmation_id"]) not in simultaneous_confirmation_ids:
            confirmations_by_drone[str(record["drone_id"])] += 1
    return {
        "schema": "org.aerocity.bench.metric-report.v1",
        "episode_id": private_episode["episode_id"],
        "target_validity_hash": validity["validity_hash"],
        "execution_level": run_result["execution_level"],
        "formal_score_eligible": formal_eligible,
        "quality": {
            "confirmed_recall_auc": confirmed_recall_auc(
                confirmation_times, target_count, duration_s
            ),
            "final_confirmed_recall": final_recall,
            "confirmed_count": confirmed_count,
            "target_count_private": target_count,
            "time_to_first": time_to_recall(
                confirmation_times, target_count, 1.0 / target_count, duration_s
            ),
            "time_to_25_percent": time_to_recall(
                confirmation_times, target_count, 0.25, duration_s
            ),
            "time_to_50_percent": time_to_recall(
                confirmation_times, target_count, 0.50, duration_s
            ),
            "time_to_75_percent": time_to_recall(
                confirmation_times, target_count, 0.75, duration_s
            ),
            "task_complete": confirmed_count == target_count,
            "stopping_protocol": {
                "method_target_count_visible": False,
                "method_task_complete_signal_visible": False,
                "episode_termination": "fixed_duration_or_runtime_terminal_with_return_requirement",
                "task_complete_is_evaluator_only": True,
            },
        },
        "coverage_diagnostics": {
            "coverage_2d_auc": _coverage_auc(
                run_result["coverage_trace"], 1, duration_s, two_d_denominator
            ),
            "coverage_3d_auc": _coverage_auc(
                run_result["coverage_trace"], 2, duration_s, three_d_denominator
            ),
            "coverage_semantics": run_result.get("coverage_semantics", "unspecified"),
            "denominators": dict(coverage_denominators),
            "inspection_footprint_auc": (
                _coverage_auc(
                    inspection_trace, 1, duration_s, inspection_area_denominator
                )
                if inspection_area_denominator > 0.0
                else None
            ),
            "inspection_footprint_final": (
                min(
                    1.0,
                    float(inspection_trace[-1][1]) / inspection_area_denominator,
                )
                if inspection_area_denominator > 0.0
                else None
            ),
            "inspection_cell_auc": (
                _coverage_auc(
                    inspection_cell_trace,
                    1,
                    duration_s,
                    float(inspection_cell_denominator),
                )
                if inspection_cell_denominator > 0
                else None
            ),
            "inspection_cell_final": (
                min(
                    1.0,
                    float(inspection_cell_trace[-1][1]) / inspection_cell_denominator,
                )
                if inspection_cell_denominator > 0
                else None
            ),
            "inspection_footprint_semantics": (
                "area_weighted_public_atlas_cell_after_accepted_observe_fov_facing_los_and_dwell"
                if inspection_area_denominator > 0.0
                else None
            ),
            "inspection_cell_semantics": (
                "auxiliary_unweighted_cell_count_not_a_primary_quality_metric"
                if inspection_cell_denominator > 0
                else None
            ),
        },
        "safety": {
            "collision_count": collision_count,
            "out_of_bounds_actions": int(ledger["out_of_bounds_actions"]),
            "safety_interventions": int(ledger["safety_interventions"]),
            "clearance_interventions": int(ledger.get("clearance_interventions", 0)),
            "minimum_clearance_m": ledger.get("minimum_clearance_m"),
            "all_survivors_returned_home": all(bool(value) for value in returned.values()),
            "rank_ineligible_due_to_collision": collision_count > 0,
        },
        "resources": {
            "path_distance_m": float(ledger["path_distance_m"]),
            "energy_used_j": float(ledger["energy_used_j"]),
            "distance_per_confirmation_m": (
                float(ledger["path_distance_m"]) / confirmed_count if confirmed_count else None
            ),
            "energy_per_confirmation_j": (
                float(ledger["energy_used_j"]) / confirmed_count if confirmed_count else None
            ),
            "wall_clock_s": run_result.get("wall_clock_s"),
        },
        "compute": {
            "planning_latency_median_s": statistics.median(latencies) if latencies else None,
            "planning_latency_p95_s": _percentile(latencies, 0.95),
            "planning_latency_p99_s": _percentile(latencies, 0.99),
            "deadline_misses": int(ledger["deadline_misses"]),
        },
        "communication": {
            "bytes_sent": int(ledger["communication_bytes_sent"]),
            "bytes_delivered": int(ledger["communication_bytes_delivered"]),
            "bytes_dropped": int(ledger["communication_bytes_dropped"]),
            "packets_sent": int(ledger.get("communication_packets_sent", 0)),
            "packets_delivered": int(ledger.get("communication_packets_delivered", 0)),
            "packets_dropped": int(ledger.get("communication_packets_dropped", 0)),
            "stale_messages_rejected": int(ledger.get("stale_messages_rejected", 0)),
            "duplicate_messages_rejected": int(ledger.get("duplicate_messages_rejected", 0)),
            "bandwidth_messages_rejected": int(ledger.get("bandwidth_messages_rejected", 0)),
        },
        "collaboration": {
            "distance_by_drone_m": distance_by_drone,
            "confirmations_by_drone": confirmations_by_drone,
            "simultaneous_confirmation_ties_excluded_from_agent_attribution": len(
                simultaneous_confirmation_ids
            ),
            "agent_confirmation_attribution_semantics": (
                "diagnostic_only_excludes_simultaneous_dwell_ties"
            ),
            "distance_jain_fairness": _jain(list(distance_by_drone.values())),
            "distance_gini": _gini(list(distance_by_drone.values())),
            "redundant_target_agent_pair_count": int(
                audit.get("redundant_target_agent_pair_count", 0)
            ),
        },
        "private_group_metrics": {
            "support_class": support_groups,
            "altitude_band": altitude_groups,
            "opportunity_tercile": opportunity_groups,
            "worst_group_recall": min(group_recalls) if group_recalls else None,
        },
    }
