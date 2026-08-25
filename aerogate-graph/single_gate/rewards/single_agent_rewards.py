"""Reward and termination logic for the single-agent 2D gate task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from single_gate.configs.experiment_config import SingleGateEnvConfig


@dataclass(frozen=True)
class SingleTerminationStatus:
    """Termination outcome for one environment step."""

    terminated: bool
    truncated: bool
    reason: str | None = None


def evaluate_single_agent_termination(
    *,
    collided: bool,
    reached_goal: bool,
    out_of_bounds: bool,
    step_count: int,
    config: SingleGateEnvConfig,
) -> SingleTerminationStatus:
    """Evaluate whether the current step should end the episode."""

    if collided:
        return SingleTerminationStatus(terminated=True, truncated=False, reason="collision")
    if reached_goal:
        return SingleTerminationStatus(terminated=True, truncated=False, reason="goal_reached")
    if out_of_bounds:
        return SingleTerminationStatus(terminated=True, truncated=False, reason="out_of_bounds")
    if step_count >= config.max_episode_steps:
        return SingleTerminationStatus(terminated=False, truncated=True, reason="timeout")
    return SingleTerminationStatus(terminated=False, truncated=False, reason=None)


def compute_single_agent_reward(
    *,
    previous_goal_distance_m: float,
    current_goal_distance_m: float,
    signed_clearance_m: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    termination: SingleTerminationStatus,
    config: SingleGateEnvConfig,
) -> tuple[float, dict[str, float]]:
    """Compute the step reward and a detailed breakdown."""

    progress_delta = float(previous_goal_distance_m - current_goal_distance_m)
    progress_reward = config.progress_reward_scale * progress_delta
    survival_reward = config.survival_reward
    clearance_deficit = max(0.0, config.safety_clearance_m - float(signed_clearance_m))
    clearance_penalty = -config.clearance_penalty_scale * (clearance_deficit ** 2)
    action_penalty = -config.action_l2_penalty_scale * float(np.dot(action, action))
    delta_action = action - previous_action
    smoothness_penalty = -config.action_smoothness_penalty_scale * float(np.dot(delta_action, delta_action))

    terminal_reward = 0.0
    if termination.reason == "goal_reached":
        terminal_reward += config.goal_bonus
    elif termination.reason == "collision":
        terminal_reward += config.collision_penalty
    elif termination.reason == "out_of_bounds":
        terminal_reward += config.out_of_bounds_penalty
    elif termination.reason == "timeout":
        terminal_reward += config.timeout_penalty

    total_reward = (
        progress_reward
        + survival_reward
        + clearance_penalty
        + action_penalty
        + smoothness_penalty
        + terminal_reward
    )
    breakdown = {
        "progress": progress_reward,
        "survival": survival_reward,
        "clearance_penalty": clearance_penalty,
        "action_penalty": action_penalty,
        "smoothness_penalty": smoothness_penalty,
        "terminal": terminal_reward,
        "total": total_reward,
    }
    return float(total_reward), breakdown

