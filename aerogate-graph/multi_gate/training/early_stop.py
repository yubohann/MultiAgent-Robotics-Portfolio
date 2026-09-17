from __future__ import annotations

"""Training helpers for the multi-agent 2D gate experiment."""


import json
import math
import subprocess
import sys
from typing import Literal

import numpy as np
import torch

from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv


MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv

def _assess_eval_thresholds(
    *,
    eval_summary: dict[str, object],
    thresholds: dict[str, float | None],
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    threshold_specs = (
        ("success_rate", "min_success_rate", "min"),
        ("team_success_rate", "min_team_success_rate", "min"),
        ("per_agent_success_rate", "min_per_agent_success_rate", "min"),
        ("height_contract_passed_rate", "min_height_contract_passed_rate", "min"),
        ("corridor_through_success_rate", "min_corridor_through_success_rate", "min"),
        ("gate_post_collision_rate", "max_gate_post_collision_rate", "max"),
        ("obstacle_collision_rate", "max_obstacle_collision_rate", "max"),
        ("dynamic_gate_collision_rate", "max_dynamic_gate_collision_rate", "max"),
        ("agent_collision_rate", "max_agent_collision_rate", "max"),
        ("out_of_bounds_rate", "max_out_of_bounds_rate", "max"),
        ("timeout_rate", "max_timeout_rate", "max"),
        ("hard_failure_rate", "max_hard_failure_rate", "max"),
        ("safety_violation_rate", "max_safety_violation_rate", "max"),
        ("height_escape_failure_rate", "max_height_escape_failure_rate", "max"),
        ("side_bypass_failure_rate", "max_side_bypass_failure_rate", "max"),
        ("corridor_miss_failure_rate", "max_corridor_miss_failure_rate", "max"),
        ("formation_line_collapse_failure_rate", "max_formation_line_collapse_failure_rate", "max"),
        ("dispersed_termination_rate", "max_dispersed_termination_rate", "max"),
        ("min_bucket_success_rate", "min_bucket_success_rate", "min"),
        ("mean_slot_error_m", "max_mean_slot_error_m", "max"),
        ("mean_guidance_tracking_error_m", "max_mean_guidance_tracking_error_m", "max"),
    )
    for metric_name, threshold_key, mode in threshold_specs:
        expected_raw = thresholds.get(threshold_key)
        if expected_raw is None:
            continue
        expected = float(expected_raw)
        actual_raw = eval_summary.get(metric_name)
        actual = None if actual_raw is None else float(actual_raw)
        passed = False
        if actual is not None:
            passed = actual >= expected if mode == "min" else actual <= expected
        checks[metric_name] = {
            "metric": metric_name,
            "threshold_key": threshold_key,
            "mode": mode,
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
    failed_checks = [name for name, result in checks.items() if not bool(result["passed"])]
    return {
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "checks": checks,
    }

def _assess_failure_stop_thresholds(
    *,
    eval_summary: dict[str, object],
    thresholds: dict[str, float | None],
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    threshold_specs = (
        ("success_rate", "max_success_rate", "max"),
        ("gate_post_collision_rate", "min_gate_post_collision_rate", "min"),
        ("dynamic_gate_collision_rate", "min_dynamic_gate_collision_rate", "min"),
        ("agent_collision_rate", "min_agent_collision_rate", "min"),
        ("out_of_bounds_rate", "min_out_of_bounds_rate", "min"),
        ("timeout_rate", "min_timeout_rate", "min"),
        ("hard_failure_rate", "min_hard_failure_rate", "min"),
        ("safety_violation_rate", "min_safety_violation_rate", "min"),
        ("mean_goal_distance_m", "min_mean_goal_distance_m", "min"),
    )
    for metric_name, threshold_key, mode in threshold_specs:
        expected_raw = thresholds.get(threshold_key)
        if expected_raw is None:
            continue
        expected = float(expected_raw)
        actual_raw = eval_summary.get(metric_name)
        actual = None if actual_raw is None else float(actual_raw)
        passed = False
        if actual is not None:
            passed = actual >= expected if mode == "min" else actual <= expected
        checks[metric_name] = {
            "metric": metric_name,
            "threshold_key": threshold_key,
            "mode": mode,
            "expected": expected,
            "actual": actual,
            "passed": bool(passed),
        }
    success_check = checks.get("success_rate")
    success_passed = True if success_check is None else bool(success_check["passed"])
    failure_checks = {
        name: result
        for name, result in checks.items()
        if name != "success_rate"
    }
    any_failure_threshold_passed = (
        True if not failure_checks else any(bool(result["passed"]) for result in failure_checks.values())
    )
    passed = bool(success_passed and any_failure_threshold_passed)
    failed_checks = []
    if not success_passed:
        failed_checks.append("success_rate")
    if failure_checks and not any_failure_threshold_passed:
        failed_checks.extend(name for name in failure_checks)
    return {
        "passed": bool(checks) and passed,
        "failed_checks": failed_checks,
        "checks": checks,
        "failure_threshold_logic": "success_rate_and_any_failure_mode",
    }

def _analyze_failure_stop_window(
    *,
    checkpoint_selection_records: list[dict[str, object]],
    thresholds: dict[str, float | None],
    min_window_length: int,
    min_transition: int,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for record_index, raw_record in enumerate(checkpoint_selection_records):
        if not isinstance(raw_record, dict):
            continue
        eval_summary = raw_record.get("selection_eval_summary")
        if not isinstance(eval_summary, dict):
            continue
        transition = int(raw_record.get("step") or 0)
        if transition < int(min_transition):
            continue
        assessment = _assess_failure_stop_thresholds(eval_summary=eval_summary, thresholds=thresholds)
        candidates.append(
            {
                "record_index": int(record_index),
                "transition": int(transition),
                "checkpoint_path": str(raw_record.get("checkpoint_path") or ""),
                "assessment": assessment,
                "passed": bool(assessment["passed"]),
                "selection_eval_summary": eval_summary,
            }
        )
    window_length = max(int(min_window_length), 1)
    latest_window = candidates[-window_length:] if len(candidates) >= window_length else []
    passed = len(latest_window) == window_length and all(bool(item["passed"]) for item in latest_window)
    return {
        "passed": bool(passed),
        "reason": "latest_window_failed_thresholds" if passed else "no_consecutive_failure_window",
        "min_window_length": int(window_length),
        "min_transition": int(min_transition),
        "candidate_count": len(candidates),
        "latest_window": latest_window,
    }

def _early_stop_check_margin(check_result: dict[str, object]) -> float:
    mode = str(check_result.get("mode") or "")
    actual = check_result.get("actual")
    expected = check_result.get("expected")
    if actual is None or expected is None:
        return float("-inf")
    actual_value = float(actual)
    expected_value = float(expected)
    if mode == "min":
        return actual_value - expected_value
    if mode == "max":
        return expected_value - actual_value
    return float("-inf")

def _early_stop_assessment_margin(assessment: dict[str, object]) -> float:
    checks = assessment.get("checks")
    if not isinstance(checks, dict) or not checks:
        return float("-inf")
    margins = [
        _early_stop_check_margin(check_result)
        for check_result in checks.values()
        if isinstance(check_result, dict)
    ]
    if not margins:
        return float("-inf")
    return min(margins)

def _collect_early_stop_window_candidates(
    *,
    checkpoint_selection_records: list[dict[str, object]],
    thresholds: dict[str, float | None],
    planned_total_transitions: int,
    min_window_length: int,
    late_half_only: bool,
) -> dict[str, object]:
    late_half_transition_floor = (float(planned_total_transitions) / 2.0) if bool(late_half_only) else 0.0
    all_candidates: list[dict[str, object]] = []
    eligible_candidates: list[dict[str, object]] = []

    for record_index, raw_record in enumerate(checkpoint_selection_records):
        if not isinstance(raw_record, dict):
            continue
        checkpoint_path = str(raw_record.get("checkpoint_path") or "")
        eval_summary = raw_record.get("selection_eval_summary")
        if not checkpoint_path or not isinstance(eval_summary, dict):
            continue
        transition = int(raw_record.get("step") or 0)
        assessment = _assess_eval_thresholds(eval_summary=eval_summary, thresholds=thresholds)
        candidate = {
            "record_index": int(record_index),
            "transition": int(transition),
            "checkpoint_path": checkpoint_path,
            "selection_eval_summary": eval_summary,
            "assessment": assessment,
            "passed": bool(assessment["passed"]),
            "margin": float(_early_stop_assessment_margin(assessment)),
        }
        all_candidates.append(candidate)
        if float(transition) >= late_half_transition_floor:
            eligible_candidates.append(candidate)

    eligible_candidates.sort(key=lambda item: (int(item["transition"]), int(item["record_index"])))
    return {
        "planned_total_transitions": int(planned_total_transitions),
        "late_half_transition_floor": float(late_half_transition_floor),
        "min_window_length": max(int(min_window_length), 1),
        "candidate_count": len(all_candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "eligible_candidates": eligible_candidates,
    }

def _find_early_stop_windows(
    *,
    eligible_candidates: list[dict[str, object]],
    min_window_length: int,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    current_window: list[dict[str, object]] = []

    def _finalize_window(window_records: list[dict[str, object]]) -> None:
        if len(window_records) < int(min_window_length):
            return
        margins = [float(record["margin"]) for record in window_records]
        windows.append(
            {
                "length": len(window_records),
                "start_transition": int(window_records[0]["transition"]),
                "end_transition": int(window_records[-1]["transition"]),
                "record_indices": [int(record["record_index"]) for record in window_records],
                "checkpoint_paths": [str(record["checkpoint_path"]) for record in window_records],
                "margin_floor": min(margins),
                "margin_mean": sum(margins) / float(len(margins)),
                "records": [dict(record) for record in window_records],
            }
        )

    for candidate in eligible_candidates:
        if bool(candidate.get("passed")):
            current_window.append(candidate)
            continue
        _finalize_window(current_window)
        current_window = []

    _finalize_window(current_window)
    return windows

def _select_early_stop_window(windows: list[dict[str, object]]) -> dict[str, object] | None:
    if not windows:
        return None
    return max(
        windows,
        key=lambda window: (
            int(window["length"]),
            int(window["end_transition"]),
            float(window["margin_floor"]),
            float(window["margin_mean"]),
        ),
    )

def _build_early_stop_window_metadata(window: dict[str, object] | None) -> dict[str, object] | None:
    if window is None:
        return None
    return {
        "length": int(window["length"]),
        "start_transition": int(window["start_transition"]),
        "end_transition": int(window["end_transition"]),
        "record_indices": list(window["record_indices"]),
        "checkpoint_paths": list(window["checkpoint_paths"]),
        "margin_floor": float(window["margin_floor"]),
        "margin_mean": float(window["margin_mean"]),
    }

def _analyze_early_stop_stable_window(
    *,
    checkpoint_selection_records: list[dict[str, object]],
    thresholds: dict[str, float | None],
    planned_total_transitions: int,
    min_window_length: int,
    late_half_only: bool,
) -> dict[str, object]:
    candidate_analysis = _collect_early_stop_window_candidates(
        checkpoint_selection_records=checkpoint_selection_records,
        thresholds=thresholds,
        planned_total_transitions=int(planned_total_transitions),
        min_window_length=int(min_window_length),
        late_half_only=bool(late_half_only),
    )
    windows = _find_early_stop_windows(
        eligible_candidates=list(candidate_analysis["eligible_candidates"]),
        min_window_length=int(candidate_analysis["min_window_length"]),
    )
    selected_window = _select_early_stop_window(windows)
    window_metadata = _build_early_stop_window_metadata(selected_window)
    result = {
        "passed": False,
        "reason": "no_qualifying_window",
        "planned_total_transitions": int(candidate_analysis["planned_total_transitions"]),
        "late_half_transition_floor": float(candidate_analysis["late_half_transition_floor"]),
        "min_window_length": int(candidate_analysis["min_window_length"]),
        "candidate_count": int(candidate_analysis["candidate_count"]),
        "eligible_candidate_count": int(candidate_analysis["eligible_candidate_count"]),
        "window": window_metadata,
        "checkpoint_path": None,
        "assessment": None,
    }
    if int(candidate_analysis["candidate_count"]) == 0:
        result["reason"] = "no_checkpoint_selection_records"
        return result
    if int(candidate_analysis["eligible_candidate_count"]) == 0:
        result["reason"] = "no_late_half_candidates" if bool(late_half_only) else "no_eligible_candidates"
        return result
    if selected_window is None:
        return result

    selected_record = dict(selected_window["records"][-1])
    result.update(
        {
            "passed": True,
            "reason": "stable_window_found",
            "checkpoint_path": str(selected_record["checkpoint_path"]),
            "assessment": dict(selected_record["assessment"]),
        }
    )
    return result