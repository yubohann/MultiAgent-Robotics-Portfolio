from __future__ import annotations

"""Training helpers for the multi-agent 2D gate experiment."""


import json
import math
import subprocess
import sys
from typing import Literal

import numpy as np
import torch

from multi_gate.configs.experiment_config import MultiExperimentConfig
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv


MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv

def _serialize_multi_episode_summary(
    *,
    episode_idx: int,
    step_count: int,
    episode_reward: float,
    info: dict[str, object],
    extra_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    snapshot = info.get("snapshot")
    virtual_center_xy = None
    mean_slot_error_m = None
    max_slot_error_m = None
    goal_distance_m = None
    path_index = None
    guidance_tracking_error_m = info.get("guidance_tracking_error_m")
    route_guidance_tracking_error_m = info.get("route_guidance_tracking_error_m")
    if snapshot is not None:
        virtual_center_xy = list(getattr(snapshot, "virtual_center_xy", ()))
        mean_slot_error_m = float(getattr(snapshot, "mean_slot_error_m", 0.0))
        max_slot_error_m = float(getattr(snapshot, "max_slot_error_m", info.get("max_slot_error_m") or 0.0))
        goal_distance_m = float(getattr(snapshot, "goal_distance_m", 0.0))
        path_index = int(getattr(snapshot, "path_index", 0))
    dynamic_gate_count = int(info.get("dynamic_gate_count") or 0)
    height_contract_passed = bool(info.get("height_contract_passed", True))
    height_escape_failure = bool(info.get("height_escape_failure", False))
    side_bypass_failure = bool(info.get("side_bypass_failure", False))
    corridor_miss_failure = bool(info.get("corridor_miss_failure", False))
    formation_line_collapse_failure = bool(info.get("formation_line_collapse_failure", False))
    corridor_completed = bool(info.get("corridor_completed", dynamic_gate_count <= 0))
    corridor_through_success = _multi_corridor_through_success_from_info(info)
    summary = {
        "episode_index": int(episode_idx),
        "steps": int(step_count),
        "episode_reward": float(episode_reward),
        "done_reason": str(info.get("done_reason") or "unknown"),
        "num_agents": int(info.get("num_agents") or 0),
        "virtual_center_xy": virtual_center_xy,
        "mean_slot_error_m": mean_slot_error_m,
        "max_slot_error_m": max_slot_error_m,
        "goal_distance_m": goal_distance_m,
        "initial_goal_distance_m": _finite_float_or_none(info.get("initial_goal_distance_m")),
        "goal_distance_improvement_m": _finite_float_or_none(info.get("goal_distance_improvement_m")),
        "goal_progress_ratio": _finite_float_or_none(info.get("goal_progress_ratio")),
        "per_agent_success_fraction": _multi_per_agent_success_fraction_from_info(info),
        "dispersed_termination": _multi_dispersed_termination_from_info(info),
        "guidance_tracking_error_m": (
            _finite_float_or_none(guidance_tracking_error_m)
        ),
        "route_guidance_tracking_error_m": (
            _finite_float_or_none(route_guidance_tracking_error_m)
        ),
        "route_guidance_source": (
            None if info.get("route_guidance_source") is None else str(info.get("route_guidance_source"))
        ),
        "guidance_latency_ms": _finite_float_or_none(info.get("guidance_latency_ms")),
        "guidance_cache_hit": (
            None if info.get("guidance_cache_hit") is None else bool(info.get("guidance_cache_hit"))
        ),
        "path_index": path_index,
        "min_clearance_m": _finite_float_or_none(info.get("min_clearance_m")),
        "min_pair_distance_m": _finite_float_or_none(info.get("min_pair_distance_m")),
        "dynamic_gate_enabled": bool(info.get("dynamic_gate_enabled", False)),
        "dynamic_gate_collision": bool(info.get("dynamic_gate_collision", False)),
        "dynamic_gate_count": dynamic_gate_count,
        "fixed_height_m": _finite_float_or_none(info.get("fixed_height_m")),
        "gate_bottom_height_m": _finite_float_or_none(info.get("gate_bottom_height_m")),
        "gate_top_height_m": _finite_float_or_none(info.get("gate_top_height_m")),
        "gate_center_height_m": _finite_float_or_none(info.get("gate_center_height_m")),
        "height_contract_passed": height_contract_passed,
        "height_escape_failure": height_escape_failure,
        "side_bypass_failure": side_bypass_failure,
        "corridor_miss_failure": corridor_miss_failure,
        "formation_shape_active": bool(info.get("formation_shape_active", False)),
        "formation_lateral_band_count": _finite_float_or_none(info.get("formation_lateral_band_count")),
        "formation_required_lateral_bands": _finite_float_or_none(info.get("formation_required_lateral_bands")),
        "formation_lateral_span_m": _finite_float_or_none(info.get("formation_lateral_span_m")),
        "formation_line_collapse_score": _finite_float_or_none(info.get("formation_line_collapse_score")),
        "formation_line_collapse_failure": formation_line_collapse_failure,
        "corridor_completed": corridor_completed,
        "corridor_through_success": corridor_through_success,
        "moving_gate_speed_mps": _finite_float_or_none(info.get("moving_gate_speed_mps")),
        "moving_gate_amplitude_m": _finite_float_or_none(info.get("moving_gate_amplitude_m")),
        "actual_gate_motion_range_m": _finite_float_or_none(info.get("actual_gate_motion_range_m")),
    }
    if extra_metrics:
        summary.update(extra_metrics)
    return summary

def _multi_corridor_through_success_from_info(info: dict[str, object]) -> bool:
    """Return the shared 2D corridor-through audit result for one terminal info dict."""

    dynamic_gate_count = int(info.get("dynamic_gate_count") or 0)
    if dynamic_gate_count <= 0:
        return True
    return bool(
        bool(info.get("height_contract_passed", True))
        and not bool(info.get("height_escape_failure", False))
        and bool(info.get("corridor_completed", False))
        and not bool(info.get("side_bypass_failure", False))
        and not bool(info.get("corridor_miss_failure", False))
    )

def _multi_episode_success_from_info(
    info: dict[str, object],
    *,
    timeout_counts_as_success: bool,
) -> bool:
    """Success requires the raw terminal reason plus the height/corridor contract."""

    reason = str(info.get("done_reason") or "")
    raw_success = reason == "goal_reached" or bool(timeout_counts_as_success and reason == "timeout")
    if not raw_success:
        return False
    return bool(
        bool(info.get("height_contract_passed", True))
        and not bool(info.get("height_escape_failure", False))
        and not bool(info.get("side_bypass_failure", False))
        and not bool(info.get("corridor_miss_failure", False))
        and not bool(info.get("formation_line_collapse_failure", False))
        and _multi_corridor_through_success_from_info(info)
    )

def _multi_per_agent_success_fraction_from_info(info: dict[str, object]) -> float:
    """Return the fraction of drones that individually completed the full route contract."""

    if _multi_episode_success_from_info(
        info,
        timeout_counts_as_success=bool(info.get("timeout_counts_as_success", False)),
    ):
        return 1.0
    hard_failure_reasons = {
        "gate_post_collision",
        "agent_collision",
        "out_of_bounds",
        "height_escape_failure",
        "side_bypass_failure",
        "corridor_miss_failure",
        "formation_line_collapse_failure",
    }
    if str(info.get("done_reason") or "") in hard_failure_reasons:
        return 0.0
    if (
        not bool(info.get("height_contract_passed", True))
        or bool(info.get("height_escape_failure", False))
        or bool(info.get("side_bypass_failure", False))
        or bool(info.get("corridor_miss_failure", False))
        or bool(info.get("formation_line_collapse_failure", False))
    ):
        return 0.0
    if int(info.get("dynamic_gate_count") or 0) > 0 and not bool(info.get("corridor_completed", False)):
        return 0.0
    positions = info.get("agent_positions_xy")
    if positions is None:
        return 0.0
    try:
        positions_array = np.asarray(positions, dtype=np.float32)
    except (TypeError, ValueError):
        return 0.0
    if positions_array.ndim != 2 or positions_array.shape[1] != 2 or positions_array.shape[0] <= 0:
        return 0.0
    goal_xy_raw = info.get("goal_xy")
    if goal_xy_raw is None:
        path_waypoints = info.get("path_waypoints")
        if path_waypoints:
            goal_xy_raw = list(path_waypoints)[-1]
    if goal_xy_raw is None:
        return 0.0
    goal_xy = np.asarray(goal_xy_raw, dtype=np.float32)
    if goal_xy.shape != (2,):
        return 0.0
    goal_radius_m = max(float(info.get("goal_radius_m") or 0.0), 1.0e-6)
    distances = np.linalg.norm(positions_array - goal_xy.reshape(1, 2), axis=1)
    return float(np.count_nonzero(distances <= goal_radius_m) / max(int(positions_array.shape[0]), 1))

def _multi_dispersed_termination_from_info(info: dict[str, object]) -> bool:
    """Return whether this episode terminated because the formation dispersed."""

    reason = str(info.get("done_reason") or "").lower()
    return any(token in reason for token in ("dispers", "formation", "slot_error", "max_slot", "line_collapse"))

def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        return None
    return resolved

def _multi_speed_samples_from_info(info: dict[str, object]) -> list[float]:
    velocities = info.get("agent_velocities_xy")
    if velocities is None:
        return []
    try:
        array = np.asarray(velocities, dtype=np.float32)
    except (TypeError, ValueError):
        return []
    if array.ndim != 2 or array.shape[-1] != 2:
        return []
    speeds = np.linalg.norm(array, axis=1)
    return [float(value) for value in speeds.reshape(-1) if np.isfinite(value)]

def _collect_finite_metric_values(
    episode_summaries: list[dict[str, object]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for summary in episode_summaries:
        resolved = _finite_float_or_none(summary.get(key))
        if resolved is not None:
            values.append(resolved)
    return values

def _finite_stat_or_none(values: list[float], *, reducer: str) -> float | None:
    if not values:
        return None
    if reducer == "mean":
        return float(np.mean(values))
    if reducer == "min":
        return float(np.min(values))
    if reducer == "max":
        return float(np.max(values))
    raise ValueError(f"Unsupported finite-stat reducer: {reducer}")

def _select_team_size(
    *,
    seed: int,
    num_agents: int | None,
    max_sampled_agents: int | None,
    experiment_config: MultiExperimentConfig,
) -> int:
    if num_agents is not None:
        return int(num_agents)
    rng = np.random.default_rng(seed)
    max_agents = experiment_config.max_agents_soft if max_sampled_agents is None else int(max_sampled_agents)
    max_agents = max(experiment_config.min_agents, min(max_agents, experiment_config.max_agents_soft))
    size_config = experiment_config.size_invariance
    if bool(size_config.enabled) and str(size_config.team_size_sampling_mode).lower() == "uniform_buckets":
        buckets = [
            int(size)
            for size in size_config.bucket_team_sizes
            if experiment_config.min_agents <= int(size) <= max_agents
        ]
        if buckets:
            return int(rng.choice(np.asarray(buckets, dtype=np.int64)))
    return int(rng.integers(experiment_config.min_agents, max_agents + 1))

def _select_team_sizes(
    *,
    seed: int,
    num_envs: int,
    num_agents: int | None,
    max_sampled_agents: int | None,
    experiment_config: MultiExperimentConfig,
) -> list[int]:
    return [
        _select_team_size(
            seed=int(seed) + env_idx,
            num_agents=num_agents,
            max_sampled_agents=max_sampled_agents,
            experiment_config=experiment_config,
        )
        for env_idx in range(max(int(num_envs), 1))
    ]

def _derive_failure_replay_metadata(
    *,
    info: dict[str, object],
    terminated: bool,
    truncated: bool,
    experiment_config: MultiExperimentConfig,
) -> dict[str, object]:
    failure_config = experiment_config.failure_replay
    env_config = experiment_config.environment
    formation_config = experiment_config.formation
    done_reason = str(info.get("done_reason") or "")
    min_clearance = float(info.get("min_clearance_m") or 0.0)
    min_pair_distance = float(info.get("min_pair_distance_m") or float("inf"))
    snapshot = info.get("snapshot")
    mean_slot_error = float(getattr(snapshot, "mean_slot_error_m", 0.0)) if snapshot is not None else 0.0

    reasons: list[str] = []
    hard_failures = set(str(reason) for reason in failure_config.hard_failure_reasons)
    if done_reason in hard_failures:
        reasons.append(done_reason)
    if min_clearance < float(failure_config.near_miss_clearance_m):
        reasons.append("near_miss_gate_post")
    if min_pair_distance < float(failure_config.near_miss_pair_distance_m):
        reasons.append("near_miss_agent")
    if mean_slot_error > float(failure_config.slot_error_spike_m):
        reasons.append("slot_error_spike")
    if bool(info.get("formation_line_collapse_failure", False)):
        reasons.append("formation_line_collapse_failure")
    if truncated and done_reason == "timeout":
        reasons.append("poor_progress")

    gate_post_cost = max(0.0, float(env_config.safety_clearance_m) - min_clearance) / max(
        float(env_config.safety_clearance_m),
        1e-6,
    )
    pair_cost = max(0.0, float(env_config.inter_agent_safe_distance_m) - min_pair_distance) / max(
        float(env_config.inter_agent_safe_distance_m),
        1e-6,
    )
    slot_cost = max(0.0, mean_slot_error - float(formation_config.goal_slot_tolerance_m)) / max(
        float(failure_config.slot_error_spike_m),
        1e-6,
    )
    terminal_cost = 1.0 if terminated and done_reason in hard_failures else 0.0
    safety_cost = float(failure_config.safety_cost_scale) * min(gate_post_cost + pair_cost + 0.25 * slot_cost + terminal_cost, 5.0)
    return {
        "failure_tag": bool(reasons),
        "failure_reason": "|".join(reasons),
        "safety_cost": float(safety_cost),
    }

def _count_safety_violating_episodes(
    eval_summary: dict[str, object],
    *,
    min_clearance_threshold_m: float = 0.5,
    min_pair_distance_threshold_m: float = 1.2,
) -> int:
    count = 0
    for episode in list(eval_summary.get("episode_summaries") or []):
        if not isinstance(episode, dict):
            continue
        reason = str(episode.get("done_reason") or "")
        min_clearance = _finite_float_or_none(episode.get("min_clearance_m"))
        min_pair_distance = _finite_float_or_none(episode.get("min_pair_distance_m"))
        if reason in {"gate_post_collision", "agent_collision", "out_of_bounds", "formation_line_collapse_failure"}:
            count += 1
        elif bool(episode.get("formation_line_collapse_failure", False)):
            count += 1
        elif (min_clearance is not None and min_clearance < float(min_clearance_threshold_m)) or (
            min_pair_distance is not None and min_pair_distance < float(min_pair_distance_threshold_m)
        ):
            count += 1
    return count