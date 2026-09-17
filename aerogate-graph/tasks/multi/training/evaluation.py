"""Training helpers for the multi-agent 2D gate experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from tasks.multi.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG,
    MultiExperimentConfig,
    is_exp3_kinematic_3d_scene_mode,
)
from tasks.multi.env.multi_gate_env import MultiGate2DEnv
from tasks.multi.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from tasks.multi.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent

from .metrics import (
    _collect_finite_metric_values,
    _count_safety_violating_episodes,
    _finite_float_or_none,
    _finite_stat_or_none,
    _multi_episode_success_from_info,
    _multi_speed_samples_from_info,
    _serialize_multi_episode_summary,
)

MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int = 4,
    seed: int = 0,
    device: str | None = None,
    num_agents: int | None = None,
    experiment_config: MultiExperimentConfig | None = None,
) -> dict[str, object]:
    """Evaluate a saved Graph-FlashSAC checkpoint deterministically."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    env_cls = _select_multi_env_class(selected_config)
    env = env_cls(
        multi_config=selected_config,
        env_config=selected_config.environment,
        observation_config=selected_config.observation,
        formation_config=selected_config.formation,
        planner_config=selected_config.planner,
    )
    agent = GraphMASACAgent.from_defaults(
        obs_shapes=env.observation_shapes,
        device=device,
        seed=seed,
        obs_config=selected_config.observation,
        masac_config=selected_config.algorithm,
        max_agents_soft=selected_config.max_agents_soft,
        build_replay_buffer=False,
    )
    validate_multi_checkpoint_compatibility(
        checkpoint_path=checkpoint_path,
        env=env,
        experiment_config=selected_config,
    )
    metadata = agent.load_checkpoint(checkpoint_path)
    active_num_agents = selected_config.default_agents if num_agents is None else int(num_agents)
    timeout_counts_as_success = bool(getattr(selected_config.environment, "timeout_counts_as_success", False))

    total_rewards: list[float] = []
    done_reason_counts: dict[str, int] = {}
    episode_summaries: list[dict[str, object]] = []
    successes = 0

    for episode_idx in range(int(episodes)):
        obs, _ = env.reset(seed=seed + episode_idx, num_agents=active_num_agents)
        episode_reward = 0.0
        step_count = 0
        previous_center_xy = np.asarray(env.snapshot().virtual_center_xy, dtype=np.float32)
        path_length_m = 0.0
        episode_speed_samples_mps: list[float] = []
        shield_active_steps = 0
        shield_intervention_norms: list[float] = []
        guidance_query_count = 0
        last_step_info: dict[str, object] = {}
        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            step_count += 1
            last_step_info = info
            episode_speed_samples_mps.extend(_multi_speed_samples_from_info(info))
            current_center_raw = info.get("virtual_center_xy")
            if current_center_raw is None and info.get("snapshot") is not None:
                current_center_raw = getattr(info["snapshot"], "virtual_center_xy", None)
            if current_center_raw is not None:
                current_center_xy = np.asarray(current_center_raw, dtype=np.float32)
                if current_center_xy.shape == (2,):
                    path_length_m += float(np.linalg.norm(current_center_xy - previous_center_xy))
                    previous_center_xy = current_center_xy
            shield_info = info.get("action_safety_shield")
            if isinstance(shield_info, dict):
                if bool(shield_info.get("active", False)):
                    shield_active_steps += 1
                intervention_norm = _finite_float_or_none(shield_info.get("mean_intervention_norm"))
                if intervention_norm is not None:
                    shield_intervention_norms.append(intervention_norm)
            guidance_meta = info.get("route_guidance_meta")
            if isinstance(guidance_meta, dict) and bool(guidance_meta.get("request_submitted", False)):
                guidance_query_count += 1
            if terminated or truncated:
                total_rewards.append(episode_reward)
                reason = str(info.get("done_reason") or "unknown")
                done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                if _multi_episode_success_from_info(
                    info,
                    timeout_counts_as_success=timeout_counts_as_success,
                ):
                    successes += 1
                episode_summaries.append(
                    _serialize_multi_episode_summary(
                        episode_idx=episode_idx,
                        step_count=step_count,
                        episode_reward=episode_reward,
                        info=info,
                        extra_metrics={
                            "path_length_m": float(path_length_m),
                            "flight_time_s": float(step_count * float(selected_config.environment.dt_s)),
                            "mean_speed_mps": _finite_stat_or_none(episode_speed_samples_mps, reducer="mean"),
                            "max_speed_mps": _finite_stat_or_none(episode_speed_samples_mps, reducer="max"),
                            "shield_activation_count": int(shield_active_steps),
                            "shield_activation_ratio": float(shield_active_steps / max(step_count, 1)),
                            "shield_intervention_norm_mean": (
                                float(np.mean(np.asarray(shield_intervention_norms, dtype=np.float32)))
                                if shield_intervention_norms
                                else None
                            ),
                            "guidance_query_count": int(guidance_query_count),
                            "planner_call_count": int(last_step_info.get("planner_call_count") or 0),
                            "planner_latency_ms_total": _finite_float_or_none(
                                last_step_info.get("planner_latency_ms_total")
                            ),
                            "planner_latency_ms_mean": _finite_float_or_none(
                                last_step_info.get("planner_latency_ms_mean")
                            ),
                        },
                    )
                )
                break

    resolved_episodes = max(int(episodes), 1)
    step_values = [float(summary["steps"]) for summary in episode_summaries]
    goal_distance_values = [float(summary["goal_distance_m"]) for summary in episode_summaries]
    slot_error_values = [float(summary["mean_slot_error_m"]) for summary in episode_summaries]
    max_slot_error_values = _collect_finite_metric_values(episode_summaries, "max_slot_error_m")
    per_agent_success_values = _collect_finite_metric_values(episode_summaries, "per_agent_success_fraction")
    progress_distance_values = _collect_finite_metric_values(episode_summaries, "goal_distance_improvement_m")
    guidance_error_values = _collect_finite_metric_values(episode_summaries, "guidance_tracking_error_m")
    route_guidance_error_values = _collect_finite_metric_values(episode_summaries, "route_guidance_tracking_error_m")
    guidance_latency_values = _collect_finite_metric_values(episode_summaries, "guidance_latency_ms")
    guidance_cache_hit_values = [
        1.0
        for summary in episode_summaries
        if summary.get("guidance_cache_hit") is True
    ]
    guidance_cache_known_values = [
        1.0
        for summary in episode_summaries
        if summary.get("guidance_cache_hit") is not None
    ]
    guidance_non_fallback_values = [
        1.0
        for summary in episode_summaries
        if str(summary.get("route_guidance_source") or "").startswith("guidance")
    ]
    guidance_source_known_values = [
        1.0
        for summary in episode_summaries
        if summary.get("route_guidance_source") is not None
    ]
    clearance_values = _collect_finite_metric_values(episode_summaries, "min_clearance_m")
    pair_distance_values = _collect_finite_metric_values(episode_summaries, "min_pair_distance_m")
    path_length_values = _collect_finite_metric_values(episode_summaries, "path_length_m")
    flight_time_values = _collect_finite_metric_values(episode_summaries, "flight_time_s")
    mean_speed_values = _collect_finite_metric_values(episode_summaries, "mean_speed_mps")
    max_speed_values = _collect_finite_metric_values(episode_summaries, "max_speed_mps")
    shield_activation_count_values = _collect_finite_metric_values(episode_summaries, "shield_activation_count")
    shield_activation_ratio_values = _collect_finite_metric_values(episode_summaries, "shield_activation_ratio")
    shield_intervention_norm_values = _collect_finite_metric_values(episode_summaries, "shield_intervention_norm_mean")
    guidance_query_count_values = _collect_finite_metric_values(episode_summaries, "guidance_query_count")
    planner_call_count_values = _collect_finite_metric_values(episode_summaries, "planner_call_count")
    planner_latency_values = _collect_finite_metric_values(episode_summaries, "planner_latency_ms_mean")
    dynamic_gate_collision_count = sum(1 for summary in episode_summaries if summary.get("dynamic_gate_collision") is True)
    obstacle_collision_count = sum(
        1
        for summary in episode_summaries
        if str(summary.get("done_reason") or "") == "gate_post_collision"
        or summary.get("dynamic_gate_collision") is True
    )
    dispersed_termination_count = sum(
        1 for summary in episode_summaries if summary.get("dispersed_termination") is True
    )
    gate_motion_values = _collect_finite_metric_values(episode_summaries, "actual_gate_motion_range_m")
    formation_lateral_band_values = _collect_finite_metric_values(episode_summaries, "formation_lateral_band_count")
    formation_line_collapse_score_values = _collect_finite_metric_values(
        episode_summaries, "formation_line_collapse_score"
    )
    height_contract_passed_rate = sum(1 for summary in episode_summaries if summary.get("height_contract_passed") is True) / resolved_episodes
    corridor_through_success_rate = sum(1 for summary in episode_summaries if summary.get("corridor_through_success") is True) / resolved_episodes
    side_bypass_failure_rate = sum(1 for summary in episode_summaries if summary.get("side_bypass_failure") is True) / resolved_episodes
    height_escape_failure_rate = sum(1 for summary in episode_summaries if summary.get("height_escape_failure") is True) / resolved_episodes
    corridor_miss_failure_rate = sum(1 for summary in episode_summaries if summary.get("corridor_miss_failure") is True) / resolved_episodes
    formation_line_collapse_failure_rate = sum(
        1 for summary in episode_summaries if summary.get("formation_line_collapse_failure") is True
    ) / resolved_episodes
    gate_post_collision_rate = float(done_reason_counts.get("gate_post_collision", 0)) / resolved_episodes
    agent_collision_rate = float(done_reason_counts.get("agent_collision", 0)) / resolved_episodes
    out_of_bounds_rate = float(done_reason_counts.get("out_of_bounds", 0)) / resolved_episodes
    timeout_rate = float(done_reason_counts.get("timeout", 0)) / resolved_episodes
    safety_violation_rate = _count_safety_violating_episodes(
        {"episode_summaries": episode_summaries},
        min_clearance_threshold_m=0.5,
        min_pair_distance_threshold_m=float(selected_config.environment.inter_agent_safe_distance_m),
    ) / resolved_episodes
    contract_failure_rate = (
        height_escape_failure_rate
        + side_bypass_failure_rate
        + corridor_miss_failure_rate
        + formation_line_collapse_failure_rate
    )
    hard_failure_rate = min(
        1.0,
        gate_post_collision_rate + agent_collision_rate + out_of_bounds_rate + contract_failure_rate,
    )
    return {
        "episodes": int(episodes),
        "num_agents": active_num_agents,
        "success_rate": successes / resolved_episodes,
        "team_success_rate": successes / resolved_episodes,
        "per_agent_success_rate": _finite_stat_or_none(per_agent_success_values, reducer="mean"),
        "gate_post_collision_rate": gate_post_collision_rate,
        "obstacle_collision_rate": float(obstacle_collision_count) / resolved_episodes,
        "dynamic_gate_collision_rate": float(dynamic_gate_collision_count) / resolved_episodes,
        "agent_collision_rate": agent_collision_rate,
        "out_of_bounds_rate": out_of_bounds_rate,
        "timeout_rate": timeout_rate,
        "height_contract_passed_rate": float(height_contract_passed_rate),
        "corridor_through_success_rate": float(corridor_through_success_rate),
        "side_bypass_failure_rate": float(side_bypass_failure_rate),
        "height_escape_failure_rate": float(height_escape_failure_rate),
        "corridor_miss_failure_rate": float(corridor_miss_failure_rate),
        "formation_line_collapse_failure_rate": float(formation_line_collapse_failure_rate),
        "contract_failure_rate": float(contract_failure_rate),
        "hard_failure_rate": float(hard_failure_rate),
        "dispersed_termination_rate": float(dispersed_termination_count) / resolved_episodes,
        "safety_violation_rate": safety_violation_rate,
        "mean_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_steps": float(np.mean(step_values)) if step_values else 0.0,
        "mean_goal_distance_m": float(np.mean(goal_distance_values)) if goal_distance_values else 0.0,
        "progress_distance_mean_m": _finite_stat_or_none(progress_distance_values, reducer="mean"),
        "mean_slot_error_m": float(np.mean(slot_error_values)) if slot_error_values else 0.0,
        "mean_max_slot_error_m": _finite_stat_or_none(max_slot_error_values, reducer="mean"),
        "max_max_slot_error_m": _finite_stat_or_none(max_slot_error_values, reducer="max"),
        "mean_guidance_tracking_error_m": _finite_stat_or_none(guidance_error_values, reducer="mean"),
        "mean_route_guidance_tracking_error_m": (
            _finite_stat_or_none(route_guidance_error_values, reducer="mean")
        ),
        "mean_guidance_latency_ms": _finite_stat_or_none(guidance_latency_values, reducer="mean"),
        "guidance_cache_hit_rate": (
            float(sum(guidance_cache_hit_values) / max(len(guidance_cache_known_values), 1))
            if guidance_cache_known_values
            else None
        ),
        "guidance_non_fallback_rate": (
            float(sum(guidance_non_fallback_values) / max(len(guidance_source_known_values), 1))
            if guidance_source_known_values
            else None
        ),
        "mean_min_clearance_m": _finite_stat_or_none(clearance_values, reducer="mean"),
        "min_min_clearance_m": _finite_stat_or_none(clearance_values, reducer="min"),
        "mean_min_pair_distance_m": _finite_stat_or_none(pair_distance_values, reducer="mean"),
        "min_min_pair_distance_m": _finite_stat_or_none(pair_distance_values, reducer="min"),
        "path_length_m_mean": _finite_stat_or_none(path_length_values, reducer="mean"),
        "flight_time_s_mean": _finite_stat_or_none(flight_time_values, reducer="mean"),
        "mean_speed_mps": _finite_stat_or_none(mean_speed_values, reducer="mean"),
        "max_speed_mps": _finite_stat_or_none(max_speed_values, reducer="max"),
        "shield_activation_count_mean": _finite_stat_or_none(shield_activation_count_values, reducer="mean"),
        "shield_activation_ratio_mean": _finite_stat_or_none(shield_activation_ratio_values, reducer="mean"),
        "shield_intervention_norm_mean": _finite_stat_or_none(shield_intervention_norm_values, reducer="mean"),
        "guidance_query_count_mean": _finite_stat_or_none(guidance_query_count_values, reducer="mean"),
        "planner_call_count_mean": _finite_stat_or_none(planner_call_count_values, reducer="mean"),
        "planner_latency_ms_mean": _finite_stat_or_none(planner_latency_values, reducer="mean"),
        "mean_actual_gate_motion_range_m": _finite_stat_or_none(gate_motion_values, reducer="mean"),
        "max_actual_gate_motion_range_m": _finite_stat_or_none(gate_motion_values, reducer="max"),
        "mean_formation_lateral_band_count": _finite_stat_or_none(formation_lateral_band_values, reducer="mean"),
        "min_formation_lateral_band_count": _finite_stat_or_none(formation_lateral_band_values, reducer="min"),
        "mean_formation_line_collapse_score": _finite_stat_or_none(
            formation_line_collapse_score_values, reducer="mean"
        ),
        "done_reason_counts": done_reason_counts,
        "episode_summaries": episode_summaries,
        "metadata": metadata,
        "timeout_counts_as_success": timeout_counts_as_success,
        "experiment_id": selected_config.experiment_id,
    }

def evaluate_actor_checkpoint(
    *,
    actor_checkpoint_path: str | Path,
    episodes: int = 4,
    seed: int = 0,
    device: str | None = None,
    num_agents: int | None = None,
    experiment_config: MultiExperimentConfig | None = None,
) -> dict[str, object]:
    """Evaluate an actor-only BC/DAgger checkpoint deterministically."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    env_cls = _select_multi_env_class(selected_config)
    env = env_cls(
        multi_config=selected_config,
        env_config=selected_config.environment,
        observation_config=selected_config.observation,
        formation_config=selected_config.formation,
        planner_config=selected_config.planner,
    )
    agent = GraphMASACAgent.from_defaults(
        obs_shapes=env.observation_shapes,
        device=device,
        seed=seed,
        obs_config=selected_config.observation,
        masac_config=selected_config.algorithm,
        max_agents_soft=selected_config.max_agents_soft,
        build_replay_buffer=False,
    )
    metadata = agent.load_actor_checkpoint(actor_checkpoint_path)
    active_num_agents = selected_config.default_agents if num_agents is None else int(num_agents)
    timeout_counts_as_success = bool(getattr(selected_config.environment, "timeout_counts_as_success", False))

    total_rewards: list[float] = []
    done_reason_counts: dict[str, int] = {}
    episode_summaries: list[dict[str, object]] = []
    successes = 0

    for episode_idx in range(int(episodes)):
        obs, _ = env.reset(seed=seed + episode_idx, num_agents=active_num_agents)
        episode_reward = 0.0
        step_count = 0
        episode_speed_samples_mps: list[float] = []
        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            step_count += 1
            episode_speed_samples_mps.extend(_multi_speed_samples_from_info(info))
            if terminated or truncated:
                total_rewards.append(episode_reward)
                reason = str(info.get("done_reason") or "unknown")
                done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                if _multi_episode_success_from_info(
                    info,
                    timeout_counts_as_success=timeout_counts_as_success,
                ):
                    successes += 1
                episode_summaries.append(
                    _serialize_multi_episode_summary(
                        episode_idx=episode_idx,
                        step_count=step_count,
                        episode_reward=episode_reward,
                        info=info,
                        extra_metrics={
                            "mean_speed_mps": _finite_stat_or_none(episode_speed_samples_mps, reducer="mean"),
                            "max_speed_mps": _finite_stat_or_none(episode_speed_samples_mps, reducer="max"),
                        },
                    )
                )
                break

    resolved_episodes = max(int(episodes), 1)
    step_values = [float(summary["steps"]) for summary in episode_summaries]
    goal_distance_values = [float(summary["goal_distance_m"]) for summary in episode_summaries]
    slot_error_values = [float(summary["mean_slot_error_m"]) for summary in episode_summaries]
    max_slot_error_values = _collect_finite_metric_values(episode_summaries, "max_slot_error_m")
    per_agent_success_values = _collect_finite_metric_values(episode_summaries, "per_agent_success_fraction")
    progress_distance_values = _collect_finite_metric_values(episode_summaries, "goal_distance_improvement_m")
    clearance_values = _collect_finite_metric_values(episode_summaries, "min_clearance_m")
    pair_distance_values = _collect_finite_metric_values(episode_summaries, "min_pair_distance_m")
    mean_speed_values = _collect_finite_metric_values(episode_summaries, "mean_speed_mps")
    max_speed_values = _collect_finite_metric_values(episode_summaries, "max_speed_mps")
    gate_motion_values = _collect_finite_metric_values(episode_summaries, "actual_gate_motion_range_m")
    formation_lateral_band_values = _collect_finite_metric_values(episode_summaries, "formation_lateral_band_count")
    formation_line_collapse_score_values = _collect_finite_metric_values(
        episode_summaries, "formation_line_collapse_score"
    )
    dynamic_gate_collision_count = sum(1 for summary in episode_summaries if summary.get("dynamic_gate_collision") is True)
    obstacle_collision_count = sum(
        1
        for summary in episode_summaries
        if str(summary.get("done_reason") or "") == "gate_post_collision"
        or summary.get("dynamic_gate_collision") is True
    )
    dispersed_termination_count = sum(
        1 for summary in episode_summaries if summary.get("dispersed_termination") is True
    )
    height_contract_passed_rate = sum(1 for summary in episode_summaries if summary.get("height_contract_passed") is True) / resolved_episodes
    corridor_through_success_rate = sum(1 for summary in episode_summaries if summary.get("corridor_through_success") is True) / resolved_episodes
    side_bypass_failure_rate = sum(1 for summary in episode_summaries if summary.get("side_bypass_failure") is True) / resolved_episodes
    height_escape_failure_rate = sum(1 for summary in episode_summaries if summary.get("height_escape_failure") is True) / resolved_episodes
    corridor_miss_failure_rate = sum(1 for summary in episode_summaries if summary.get("corridor_miss_failure") is True) / resolved_episodes
    formation_line_collapse_failure_rate = sum(
        1 for summary in episode_summaries if summary.get("formation_line_collapse_failure") is True
    ) / resolved_episodes
    gate_post_collision_rate = float(done_reason_counts.get("gate_post_collision", 0)) / resolved_episodes
    agent_collision_rate = float(done_reason_counts.get("agent_collision", 0)) / resolved_episodes
    out_of_bounds_rate = float(done_reason_counts.get("out_of_bounds", 0)) / resolved_episodes
    timeout_rate = float(done_reason_counts.get("timeout", 0)) / resolved_episodes
    safety_violation_rate = _count_safety_violating_episodes(
        {"episode_summaries": episode_summaries},
        min_clearance_threshold_m=0.5,
        min_pair_distance_threshold_m=float(selected_config.environment.inter_agent_safe_distance_m),
    ) / resolved_episodes
    contract_failure_rate = (
        height_escape_failure_rate
        + side_bypass_failure_rate
        + corridor_miss_failure_rate
        + formation_line_collapse_failure_rate
    )
    hard_failure_rate = min(
        1.0,
        gate_post_collision_rate + agent_collision_rate + out_of_bounds_rate + contract_failure_rate,
    )
    return {
        "episodes": int(episodes),
        "num_agents": active_num_agents,
        "success_rate": successes / resolved_episodes,
        "team_success_rate": successes / resolved_episodes,
        "per_agent_success_rate": _finite_stat_or_none(per_agent_success_values, reducer="mean"),
        "gate_post_collision_rate": gate_post_collision_rate,
        "obstacle_collision_rate": float(obstacle_collision_count) / resolved_episodes,
        "dynamic_gate_collision_rate": float(dynamic_gate_collision_count) / resolved_episodes,
        "agent_collision_rate": agent_collision_rate,
        "out_of_bounds_rate": out_of_bounds_rate,
        "timeout_rate": timeout_rate,
        "height_contract_passed_rate": float(height_contract_passed_rate),
        "corridor_through_success_rate": float(corridor_through_success_rate),
        "side_bypass_failure_rate": float(side_bypass_failure_rate),
        "height_escape_failure_rate": float(height_escape_failure_rate),
        "corridor_miss_failure_rate": float(corridor_miss_failure_rate),
        "formation_line_collapse_failure_rate": float(formation_line_collapse_failure_rate),
        "contract_failure_rate": float(contract_failure_rate),
        "hard_failure_rate": float(hard_failure_rate),
        "dispersed_termination_rate": float(dispersed_termination_count) / resolved_episodes,
        "safety_violation_rate": safety_violation_rate,
        "mean_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_steps": float(np.mean(step_values)) if step_values else 0.0,
        "mean_goal_distance_m": float(np.mean(goal_distance_values)) if goal_distance_values else 0.0,
        "progress_distance_mean_m": _finite_stat_or_none(progress_distance_values, reducer="mean"),
        "mean_slot_error_m": float(np.mean(slot_error_values)) if slot_error_values else 0.0,
        "mean_max_slot_error_m": _finite_stat_or_none(max_slot_error_values, reducer="mean"),
        "max_max_slot_error_m": _finite_stat_or_none(max_slot_error_values, reducer="max"),
        "mean_min_clearance_m": _finite_stat_or_none(clearance_values, reducer="mean"),
        "min_min_clearance_m": _finite_stat_or_none(clearance_values, reducer="min"),
        "mean_min_pair_distance_m": _finite_stat_or_none(pair_distance_values, reducer="mean"),
        "min_min_pair_distance_m": _finite_stat_or_none(pair_distance_values, reducer="min"),
        "mean_speed_mps": _finite_stat_or_none(mean_speed_values, reducer="mean"),
        "max_speed_mps": _finite_stat_or_none(max_speed_values, reducer="max"),
        "mean_actual_gate_motion_range_m": _finite_stat_or_none(gate_motion_values, reducer="mean"),
        "max_actual_gate_motion_range_m": _finite_stat_or_none(gate_motion_values, reducer="max"),
        "mean_formation_lateral_band_count": _finite_stat_or_none(formation_lateral_band_values, reducer="mean"),
        "min_formation_lateral_band_count": _finite_stat_or_none(formation_lateral_band_values, reducer="min"),
        "mean_formation_line_collapse_score": _finite_stat_or_none(
            formation_line_collapse_score_values, reducer="mean"
        ),
        "done_reason_counts": done_reason_counts,
        "episode_summaries": episode_summaries,
        "metadata": metadata,
        "timeout_counts_as_success": timeout_counts_as_success,
        "experiment_id": selected_config.experiment_id,
        "actor_checkpoint_path": str(actor_checkpoint_path),
    }

def evaluate_size_buckets(
    *,
    checkpoint_path: str | Path,
    episodes_per_bucket: int = 2,
    seed: int = 0,
    device: str | None = None,
    team_sizes: tuple[int, ...] | list[int] | None = None,
    experiment_config: MultiExperimentConfig | None = None,
) -> dict[str, object]:
    """Evaluate one checkpoint across explicit team-size buckets."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    requested_team_sizes = tuple(team_sizes or selected_config.size_invariance.bucket_team_sizes)
    valid_team_sizes = tuple(
        int(size)
        for size in requested_team_sizes
        if selected_config.min_agents <= int(size) <= selected_config.max_agents_soft
    )
    if not valid_team_sizes:
        raise ValueError("No valid team sizes are available for bucketed evaluation.")

    bucket_summaries: dict[str, dict[str, object]] = {}
    total_episodes = 0
    weighted_successes = 0.0
    weighted_rewards = 0.0
    weighted_steps = 0.0
    weighted_goal_distance = 0.0
    weighted_slot_error = 0.0
    weighted_gate_post_collision_rate = 0.0
    weighted_agent_collision_rate = 0.0
    weighted_out_of_bounds_rate = 0.0
    weighted_timeout_rate = 0.0
    weighted_guidance_error = 0.0
    weighted_route_guidance_error = 0.0
    weighted_guidance_latency_ms = 0.0
    weighted_guidance_cache_hit_rate = 0.0
    weighted_guidance_non_fallback_rate = 0.0
    guidance_error_episodes = 0
    route_guidance_error_episodes = 0
    guidance_latency_episodes = 0
    guidance_cache_hit_episodes = 0
    guidance_non_fallback_episodes = 0
    weighted_clearance = 0.0
    weighted_pair_distance = 0.0
    done_reason_counts: dict[str, int] = {}
    safety_violations = 0
    min_clearances: list[float] = []
    min_pair_distances: list[float] = []

    for bucket_idx, team_size in enumerate(valid_team_sizes):
        summary = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=int(episodes_per_bucket),
            seed=seed + bucket_idx * 1000,
            device=device,
            num_agents=team_size,
            experiment_config=selected_config,
        )
        bucket_summaries[str(team_size)] = summary
        episodes = max(int(summary.get("episodes") or 0), 0)
        total_episodes += episodes
        weighted_successes += float(summary.get("success_rate") or 0.0) * episodes
        weighted_rewards += float(summary.get("mean_episode_reward") or 0.0) * episodes
        weighted_steps += float(summary.get("mean_steps") or 0.0) * episodes
        weighted_goal_distance += float(summary.get("mean_goal_distance_m") or 0.0) * episodes
        weighted_slot_error += float(summary.get("mean_slot_error_m") or 0.0) * episodes
        weighted_gate_post_collision_rate += float(summary.get("gate_post_collision_rate") or 0.0) * episodes
        weighted_agent_collision_rate += float(summary.get("agent_collision_rate") or 0.0) * episodes
        weighted_out_of_bounds_rate += float(summary.get("out_of_bounds_rate") or 0.0) * episodes
        weighted_timeout_rate += float(summary.get("timeout_rate") or 0.0) * episodes
        if summary.get("mean_guidance_tracking_error_m") is not None:
            weighted_guidance_error += float(summary.get("mean_guidance_tracking_error_m") or 0.0) * episodes
            guidance_error_episodes += episodes
        if summary.get("mean_route_guidance_tracking_error_m") is not None:
            weighted_route_guidance_error += float(summary.get("mean_route_guidance_tracking_error_m") or 0.0) * episodes
            route_guidance_error_episodes += episodes
        if summary.get("mean_guidance_latency_ms") is not None:
            weighted_guidance_latency_ms += float(summary.get("mean_guidance_latency_ms") or 0.0) * episodes
            guidance_latency_episodes += episodes
        if summary.get("guidance_cache_hit_rate") is not None:
            weighted_guidance_cache_hit_rate += float(summary.get("guidance_cache_hit_rate") or 0.0) * episodes
            guidance_cache_hit_episodes += episodes
        if summary.get("guidance_non_fallback_rate") is not None:
            weighted_guidance_non_fallback_rate += float(summary.get("guidance_non_fallback_rate") or 0.0) * episodes
            guidance_non_fallback_episodes += episodes
        mean_clearance = _finite_float_or_none(summary.get("mean_min_clearance_m"))
        if mean_clearance is not None:
            weighted_clearance += mean_clearance * episodes
        mean_pair_distance = _finite_float_or_none(summary.get("mean_min_pair_distance_m"))
        if mean_pair_distance is not None:
            weighted_pair_distance += mean_pair_distance * episodes
        for reason, count in dict(summary.get("done_reason_counts") or {}).items():
            done_reason_counts[str(reason)] = done_reason_counts.get(str(reason), 0) + int(count)
        summary_safety_rate = _finite_float_or_none(summary.get("safety_violation_rate"))
        if summary_safety_rate is None:
            safety_violations += _count_safety_violating_episodes(summary)
        else:
            safety_violations += int(round(summary_safety_rate * episodes))
        min_clearance = _finite_float_or_none(summary.get("min_min_clearance_m"))
        if min_clearance is not None:
            min_clearances.append(min_clearance)
        min_pair_distance = _finite_float_or_none(summary.get("min_min_pair_distance_m"))
        if min_pair_distance is not None:
            min_pair_distances.append(min_pair_distance)

    min_bucket_success_rate = min(float(summary.get("success_rate") or 0.0) for summary in bucket_summaries.values())
    return {
        "episodes": int(total_episodes),
        "episodes_per_bucket": int(episodes_per_bucket),
        "team_sizes": list(valid_team_sizes),
        "success_rate": weighted_successes / max(total_episodes, 1),
        "min_bucket_success_rate": float(min_bucket_success_rate),
        "mean_episode_reward": weighted_rewards / max(total_episodes, 1),
        "mean_steps": weighted_steps / max(total_episodes, 1),
        "mean_goal_distance_m": weighted_goal_distance / max(total_episodes, 1),
        "mean_slot_error_m": weighted_slot_error / max(total_episodes, 1),
        "gate_post_collision_rate": weighted_gate_post_collision_rate / max(total_episodes, 1),
        "agent_collision_rate": weighted_agent_collision_rate / max(total_episodes, 1),
        "out_of_bounds_rate": weighted_out_of_bounds_rate / max(total_episodes, 1),
        "timeout_rate": weighted_timeout_rate / max(total_episodes, 1),
        "hard_failure_rate": (
            weighted_gate_post_collision_rate + weighted_agent_collision_rate + weighted_out_of_bounds_rate
        )
        / max(total_episodes, 1),
        "mean_guidance_tracking_error_m": (
            weighted_guidance_error / max(guidance_error_episodes, 1) if guidance_error_episodes > 0 else None
        ),
        "mean_route_guidance_tracking_error_m": (
            weighted_route_guidance_error / max(route_guidance_error_episodes, 1)
            if route_guidance_error_episodes > 0
            else None
        ),
        "mean_guidance_latency_ms": (
            weighted_guidance_latency_ms / max(guidance_latency_episodes, 1)
            if guidance_latency_episodes > 0
            else None
        ),
        "guidance_cache_hit_rate": (
            weighted_guidance_cache_hit_rate / max(guidance_cache_hit_episodes, 1)
            if guidance_cache_hit_episodes > 0
            else None
        ),
        "guidance_non_fallback_rate": (
            weighted_guidance_non_fallback_rate / max(guidance_non_fallback_episodes, 1)
            if guidance_non_fallback_episodes > 0
            else None
        ),
        "mean_min_clearance_m": (
            weighted_clearance / max(total_episodes, 1) if min_clearances else None
        ),
        "min_min_clearance_m": float(min(min_clearances)) if min_clearances else None,
        "mean_min_pair_distance_m": (
            weighted_pair_distance / max(total_episodes, 1) if min_pair_distances else None
        ),
        "min_min_pair_distance_m": float(min(min_pair_distances)) if min_pair_distances else None,
        "done_reason_counts": done_reason_counts,
        "safety_violation_rate": safety_violations / max(total_episodes, 1),
        "bucket_evaluation": bucket_summaries,
        "metadata": {"checkpoint_path": str(checkpoint_path)},
        "timeout_counts_as_success": bool(
            getattr(selected_config.environment, "timeout_counts_as_success", False)
        ),
        "experiment_id": selected_config.experiment_id,
    }

def validate_multi_checkpoint_compatibility(
    *,
    checkpoint_path: str | Path,
    env: MultiEnvType,
    experiment_config: MultiExperimentConfig,
) -> dict[str, object]:
    """Validate one inference/resume checkpoint against the active multi-agent setup."""

    from .checkpoint import _load_checkpoint_metadata, _multi_resume_compatibility_findings

    metadata = _load_checkpoint_metadata(checkpoint_path)
    findings = _multi_resume_compatibility_findings(
        metadata=metadata,
        env=env,
        experiment_config=experiment_config,
    )
    incompatible = [name for name, result in findings.items() if not bool(result["compatible"])]
    if incompatible:
        raise ValueError(
            "Multi-agent checkpoint is incompatible with the current experiment: "
            f"{Path(checkpoint_path)} | failed checks: {', '.join(incompatible)}"
        )
    return metadata

def _select_multi_env_class(experiment_config: MultiExperimentConfig) -> type[MultiGate2DEnv]:
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "")).strip().lower()
    if is_exp3_kinematic_3d_scene_mode(scene_mode):
        return MultiGateKinematic3DEnv
    return MultiGate2DEnv