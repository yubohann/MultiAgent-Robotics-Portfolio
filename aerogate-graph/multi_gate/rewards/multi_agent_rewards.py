"""Reward and termination logic for the multi-agent 2D gate task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multi_gate.configs.experiment_config import MultiGateEnvConfig


@dataclass(frozen=True)
class MultiTerminationStatus:
    """Termination outcome for one team step."""

    terminated: bool
    truncated: bool
    reason: str | None = None


def _slot_safety_gate(current_mean_slot_error_m: float, config: MultiGateEnvConfig) -> float:
    scale_m = max(float(config.inter_agent_safe_distance_m), 1.0e-6)
    ratio = float(current_mean_slot_error_m) / scale_m
    return float(1.0 / (1.0 + ratio * ratio))


def _max_slot_safety_gate(current_max_slot_error_m: float, config: MultiGateEnvConfig) -> float:
    hard_stop_ratio = max(float(getattr(config, "progress_max_slot_hard_stop_ratio", 0.0) or 0.0), 0.0)
    if hard_stop_ratio <= 0.0:
        return 1.0
    scale_m = max(float(config.inter_agent_safe_distance_m), 1.0e-6)
    if float(current_max_slot_error_m) >= (scale_m * hard_stop_ratio):
        return 0.0
    return 1.0


def _boundary_safety_gate(boundary_proximity_deficit_m: float, config: MultiGateEnvConfig) -> float:
    margin_m = max(float(config.boundary_soft_margin_m), 1.0e-6)
    ratio = np.clip(float(boundary_proximity_deficit_m) / margin_m, 0.0, 1.0)
    return float(1.0 - ratio)


def _separation_safety_gate(min_pair_distance_m: float, config: MultiGateEnvConfig) -> float:
    separation_warning_ratio = max(float(getattr(config, "separation_warning_ratio", 1.0) or 1.0), 1.0)
    safe_distance_m = float(config.inter_agent_safe_distance_m)
    hard_stop_margin_m = max(
        float(getattr(config, "progress_separation_hard_stop_margin_m", 0.0) or 0.0),
        0.0,
    )
    if float(min_pair_distance_m) < (safe_distance_m + hard_stop_margin_m):
        return 0.0
    warning_distance_m = safe_distance_m * separation_warning_ratio
    warning_span_m = max(warning_distance_m - safe_distance_m, 1.0e-6)
    ratio = np.clip((warning_distance_m - float(min_pair_distance_m)) / warning_span_m, 0.0, 1.0)
    return float(1.0 - ratio)


def _guidance_safety_gate(current_guidance_tracking_error_m: float, config: MultiGateEnvConfig) -> float:
    guidance_escape_penalty_scale = float(getattr(config, "guidance_escape_penalty_scale", 0.0) or 0.0)
    goal_proximity_reward_scale = float(getattr(config, "goal_proximity_reward_scale", 0.0) or 0.0)
    if guidance_escape_penalty_scale <= 0.0 and goal_proximity_reward_scale <= 0.0:
        return 1.0
    soft_margin_m = max(float(getattr(config, "guidance_escape_soft_margin_m", 0.0) or 0.0), 1.0e-6)
    if float(current_guidance_tracking_error_m) <= soft_margin_m:
        return 1.0
    excess_ratio = (float(current_guidance_tracking_error_m) - soft_margin_m) / soft_margin_m
    return float(1.0 / (1.0 + excess_ratio * excess_ratio))


def evaluate_multi_agent_termination(
    *,
    gate_post_collision: bool,
    agent_collision: bool,
    reached_goal: bool,
    out_of_bounds: bool,
    step_count: int,
    config: MultiGateEnvConfig,
    height_escape_failure: bool = False,
    side_bypass_failure: bool = False,
    corridor_miss_failure: bool = False,
    formation_line_collapse_failure: bool = False,
) -> MultiTerminationStatus:
    """Evaluate whether the team episode should end."""

    if height_escape_failure:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="height_escape_failure")
    if side_bypass_failure:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="side_bypass_failure")
    if corridor_miss_failure:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="corridor_miss_failure")
    if gate_post_collision:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="gate_post_collision")
    if agent_collision:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="agent_collision")
    if formation_line_collapse_failure and bool(getattr(config, "formation_line_collapse_terminal", False)):
        return MultiTerminationStatus(terminated=True, truncated=False, reason="formation_line_collapse_failure")
    if reached_goal:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="goal_reached")
    if out_of_bounds:
        return MultiTerminationStatus(terminated=True, truncated=False, reason="out_of_bounds")
    if step_count >= config.max_episode_steps:
        return MultiTerminationStatus(terminated=False, truncated=True, reason="timeout")
    return MultiTerminationStatus(terminated=False, truncated=False, reason=None)


def compute_multi_agent_reward(
    *,
    previous_goal_distance_m: float,
    current_goal_distance_m: float,
    previous_mean_slot_error_m: float,
    current_mean_slot_error_m: float,
    current_max_slot_error_m: float,
    current_guidance_tracking_error_m: float,
    boundary_proximity_deficit_m: float,
    min_clearance_m: float,
    min_pair_distance_m: float,
    formation_line_collapse_score: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    termination: MultiTerminationStatus,
    config: MultiGateEnvConfig,
) -> tuple[float, dict[str, float]]:
    """Compute the team reward and a detailed breakdown."""

    raw_progress_reward = config.progress_reward_scale * (previous_goal_distance_m - current_goal_distance_m)
    slot_progress_gate = _slot_safety_gate(current_mean_slot_error_m, config)
    max_slot_progress_gate = _max_slot_safety_gate(current_max_slot_error_m, config)
    boundary_progress_gate = _boundary_safety_gate(boundary_proximity_deficit_m, config)
    separation_progress_gate = _separation_safety_gate(min_pair_distance_m, config)
    guidance_progress_gate = _guidance_safety_gate(current_guidance_tracking_error_m, config)
    progress_gate = (
        slot_progress_gate
        * max_slot_progress_gate
        * boundary_progress_gate
        * separation_progress_gate
        * guidance_progress_gate
    )
    progress_reward = raw_progress_reward
    if raw_progress_reward > 0.0:
        ungated_fraction = float(np.clip(getattr(config, "ungated_progress_reward_fraction", 0.0), 0.0, 1.0))
        progress_reward *= ungated_fraction + (1.0 - ungated_fraction) * progress_gate
    goal_proximity_reward_scale = float(getattr(config, "goal_proximity_reward_scale", 0.0) or 0.0)
    goal_proximity_sigma_m = max(float(getattr(config, "goal_proximity_sigma_m", 1.0) or 1.0), 1.0e-6)
    goal_proximity_factor = (
        float(np.exp(-float(current_goal_distance_m) / goal_proximity_sigma_m))
        if goal_proximity_reward_scale > 0.0
        else 0.0
    )
    goal_proximity_reward = progress_gate * goal_proximity_reward_scale * goal_proximity_factor
    near_goal_slot_scale = 1.0 + goal_proximity_factor
    slot_improvement_reward = config.slot_improvement_scale * near_goal_slot_scale * (
        previous_mean_slot_error_m - current_mean_slot_error_m
    )
    slot_improvement_gate = (
        max_slot_progress_gate
        * boundary_progress_gate
        * separation_progress_gate
        * guidance_progress_gate
    )
    if slot_improvement_reward > 0.0:
        slot_improvement_reward *= slot_improvement_gate
    slot_error_penalty = -config.slot_error_penalty_scale * current_mean_slot_error_m
    max_slot_error_penalty = -config.max_slot_error_penalty_scale * float(current_max_slot_error_m)
    max_slot_escape_soft_margin_ratio = max(
        float(getattr(config, "max_slot_escape_soft_margin_ratio", 0.0) or 0.0),
        0.0,
    )
    max_slot_escape_soft_margin_m = float(config.inter_agent_safe_distance_m) * max_slot_escape_soft_margin_ratio
    max_slot_escape_deficit = (
        max(0.0, float(current_max_slot_error_m) - max_slot_escape_soft_margin_m)
        if max_slot_escape_soft_margin_ratio > 0.0
        else 0.0
    )
    max_slot_escape_penalty = -float(getattr(config, "max_slot_escape_penalty_scale", 0.0) or 0.0) * (
        max_slot_escape_deficit ** 2
    )
    guidance_tracking_penalty = -config.guidance_tracking_penalty_scale * float(current_guidance_tracking_error_m)
    guidance_escape_deficit = max(
        0.0,
        float(current_guidance_tracking_error_m) - float(getattr(config, "guidance_escape_soft_margin_m", 0.0)),
    )
    guidance_escape_penalty = -float(getattr(config, "guidance_escape_penalty_scale", 0.0)) * (
        guidance_escape_deficit ** 2
    )
    survival_reward = config.survival_reward
    boundary_soft_margin_m = max(float(config.boundary_soft_margin_m), 1.0e-6)
    boundary_proximity_ratio = max(0.0, float(boundary_proximity_deficit_m)) / boundary_soft_margin_m
    boundary_proximity_penalty = -config.boundary_proximity_penalty_scale * (boundary_proximity_ratio ** 2)
    clearance_deficit = max(0.0, config.safety_clearance_m - float(min_clearance_m))
    clearance_penalty = -config.clearance_penalty_scale * (clearance_deficit ** 2)
    separation_warning_ratio = max(float(getattr(config, "separation_warning_ratio", 1.0) or 1.0), 1.0)
    separation_warning_distance_m = float(config.inter_agent_safe_distance_m) * separation_warning_ratio
    separation_warning_span_m = max(
        separation_warning_distance_m - float(config.inter_agent_safe_distance_m),
        1.0e-6,
    )
    separation_proximity_ratio = np.clip(
        (
            separation_warning_distance_m - float(min_pair_distance_m)
        )
        / separation_warning_span_m,
        0.0,
        1.0,
    )
    separation_proximity_penalty = -config.separation_proximity_penalty_scale * float(
        separation_proximity_ratio ** 2
    )
    separation_deficit = max(0.0, config.inter_agent_safe_distance_m - float(min_pair_distance_m))
    separation_penalty = -config.separation_penalty_scale * (separation_deficit ** 2)
    formation_line_collapse_penalty = -float(
        getattr(config, "formation_line_collapse_penalty_scale", 0.0) or 0.0
    ) * float(max(0.0, formation_line_collapse_score) ** 2)
    action_penalty = -config.action_l2_penalty_scale * float(np.mean(action ** 2))
    smoothness_penalty = -config.action_smoothness_penalty_scale * float(np.mean((action - previous_action) ** 2))

    terminal_reward = 0.0
    if termination.reason == "goal_reached":
        terminal_reward += config.goal_bonus
    elif termination.reason == "gate_post_collision":
        terminal_reward += config.gate_post_collision_penalty
    elif termination.reason == "agent_collision":
        terminal_reward += config.agent_collision_penalty
    elif termination.reason == "out_of_bounds":
        terminal_reward += config.out_of_bounds_penalty
    elif termination.reason == "height_escape_failure":
        terminal_reward += float(getattr(config, "height_escape_penalty", config.out_of_bounds_penalty))
    elif termination.reason == "side_bypass_failure":
        terminal_reward += float(getattr(config, "side_bypass_penalty", config.out_of_bounds_penalty))
    elif termination.reason == "corridor_miss_failure":
        terminal_reward += float(getattr(config, "corridor_miss_penalty", config.out_of_bounds_penalty))
    elif termination.reason == "formation_line_collapse_failure":
        terminal_reward += float(
            getattr(config, "formation_line_collapse_terminal_penalty", config.out_of_bounds_penalty)
        )
    elif termination.reason == "timeout":
        terminal_reward += config.timeout_penalty
        terminal_reward -= float(getattr(config, "timeout_goal_distance_penalty_scale", 0.0) or 0.0) * float(
            current_goal_distance_m
        )

    total_reward = (
        progress_reward
        + goal_proximity_reward
        + slot_improvement_reward
        + slot_error_penalty
        + max_slot_error_penalty
        + max_slot_escape_penalty
        + guidance_tracking_penalty
        + guidance_escape_penalty
        + survival_reward
        + boundary_proximity_penalty
        + clearance_penalty
        + separation_proximity_penalty
        + separation_penalty
        + formation_line_collapse_penalty
        + action_penalty
        + smoothness_penalty
        + terminal_reward
    )
    return float(total_reward), {
        "raw_progress": float(raw_progress_reward),
        "progress_gate_slot": float(slot_progress_gate),
        "progress_gate_max_slot": float(max_slot_progress_gate),
        "progress_gate_boundary": float(boundary_progress_gate),
        "progress_gate_separation": float(separation_progress_gate),
        "progress_gate_guidance": float(guidance_progress_gate),
        "progress_gate": float(progress_gate),
        "ungated_progress_reward_fraction": float(
            np.clip(getattr(config, "ungated_progress_reward_fraction", 0.0), 0.0, 1.0)
        ),
        "progress": float(progress_reward),
        "goal_proximity_factor": float(goal_proximity_factor),
        "goal_proximity_reward": float(goal_proximity_reward),
        "slot_improvement_gate": float(slot_improvement_gate),
        "slot_improvement": float(slot_improvement_reward),
        "slot_error_penalty": float(slot_error_penalty),
        "max_slot_error_penalty": float(max_slot_error_penalty),
        "max_slot_escape_penalty": float(max_slot_escape_penalty),
        "guidance_tracking_penalty": float(guidance_tracking_penalty),
        "guidance_escape_penalty": float(guidance_escape_penalty),
        "survival": float(survival_reward),
        "boundary_proximity_penalty": float(boundary_proximity_penalty),
        "clearance_penalty": float(clearance_penalty),
        "separation_proximity_penalty": float(separation_proximity_penalty),
        "separation_penalty": float(separation_penalty),
        "formation_line_collapse_penalty": float(formation_line_collapse_penalty),
        "action_penalty": float(action_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "terminal": float(terminal_reward),
        "total": float(total_reward),
    }

