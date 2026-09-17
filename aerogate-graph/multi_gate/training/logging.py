from __future__ import annotations

"""Training helpers for the multi-agent 2D gate experiment."""


import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import torch

from multi_gate.configs.experiment_config import MultiExperimentConfig
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from shared.runtime.artifacts import allocate_replay_artifacts, write_json
from shared.runtime.tensorboard import (
    log_scalar,
    log_scalars
)


MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv

from .evaluation import (
    evaluate_checkpoint,
    evaluate_size_buckets
)

def _log_multi_training_scalars(
    writer,
    *,
    transitions_collected: int,
    completed_episodes: int,
    buffer_size: int,
    num_envs: int,
    last_update_metrics: dict[str, float],
    done_reason_counts: dict[str, int],
    active_num_agents: np.ndarray,
    replay_diagnostics: dict[str, object],
) -> None:
    team_sizes = np.asarray(active_num_agents, dtype=np.int32).reshape(-1)
    log_scalar(writer, "train/transitions_collected", transitions_collected, transitions_collected)
    log_scalar(writer, "train/completed_episodes", completed_episodes, transitions_collected)
    log_scalar(writer, "train/buffer_size", buffer_size, transitions_collected)
    log_scalar(writer, "train/num_envs", num_envs, transitions_collected)
    log_scalars(writer, "updates", last_update_metrics, transitions_collected)
    for reason, count in done_reason_counts.items():
        log_scalar(writer, f"done_reason_counts/{reason}", count, transitions_collected)
    if team_sizes.size > 0:
        log_scalar(writer, "teams/current_min_size", int(np.min(team_sizes)), transitions_collected)
        log_scalar(writer, "teams/current_max_size", int(np.max(team_sizes)), transitions_collected)
        log_scalar(writer, "teams/current_mean_size", float(np.mean(team_sizes)), transitions_collected)
    log_scalar(writer, "replay/size", replay_diagnostics.get("size"), transitions_collected)
    log_scalar(writer, "replay/capacity", replay_diagnostics.get("capacity"), transitions_collected)
    log_scalar(writer, "replay/failure_buffer_size", replay_diagnostics.get("failure_buffer_size"), transitions_collected)
    log_scalar(writer, "replay/failure_replay_ratio", replay_diagnostics.get("failure_replay_ratio"), transitions_collected)
    log_scalar(writer, "replay/mean_safety_cost", replay_diagnostics.get("mean_safety_cost"), transitions_collected)
    for reason, count in dict(replay_diagnostics.get("failure_reason_counts") or {}).items():
        log_scalar(writer, f"replay/failure_reason_counts/{reason}", count, transitions_collected)

def _log_periodic_eval_scalars(
    writer,
    *,
    eval_summary: dict[str, object],
    transitions_collected: int,
) -> None:
    log_scalar(writer, "periodic_eval/success_rate", eval_summary.get("success_rate"), transitions_collected)
    log_scalar(
        writer,
        "periodic_eval/mean_episode_reward",
        eval_summary.get("mean_episode_reward"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/min_bucket_success_rate",
        eval_summary.get("min_bucket_success_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/safety_violation_rate",
        eval_summary.get("safety_violation_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/hard_failure_rate",
        eval_summary.get("hard_failure_rate"),
        transitions_collected,
    )
    log_scalar(writer, "periodic_eval/timeout_rate", eval_summary.get("timeout_rate"), transitions_collected)
    log_scalar(
        writer,
        "periodic_eval/contract_failure_rate",
        eval_summary.get("contract_failure_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/corridor_through_success_rate",
        eval_summary.get("corridor_through_success_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/side_bypass_failure_rate",
        eval_summary.get("side_bypass_failure_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/height_escape_failure_rate",
        eval_summary.get("height_escape_failure_rate"),
        transitions_collected,
    )
    log_scalar(writer, "periodic_eval/mean_steps", eval_summary.get("mean_steps"), transitions_collected)
    log_scalar(
        writer,
        "periodic_eval/mean_slot_error_m",
        eval_summary.get("mean_slot_error_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/mean_max_slot_error_m",
        eval_summary.get("mean_max_slot_error_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/per_agent_success_rate",
        eval_summary.get("per_agent_success_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/obstacle_collision_rate",
        eval_summary.get("obstacle_collision_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/agent_agent_collision_rate",
        eval_summary.get("agent_collision_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/gate_post_collision_rate",
        eval_summary.get("gate_post_collision_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/dynamic_gate_collision_rate",
        eval_summary.get("dynamic_gate_collision_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/progress_distance_mean_m",
        eval_summary.get("progress_distance_mean_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/mean_speed_mps",
        eval_summary.get("mean_speed_mps"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/max_speed_mps",
        eval_summary.get("max_speed_mps"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/dispersed_termination_rate",
        eval_summary.get("dispersed_termination_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/min_pair_distance_mean_m",
        eval_summary.get("mean_min_pair_distance_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/min_pair_distance_min_m",
        eval_summary.get("min_min_pair_distance_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/formation_slot_error_mean_m",
        eval_summary.get("mean_slot_error_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/formation_slot_error_max_m",
        eval_summary.get("max_max_slot_error_m", eval_summary.get("mean_max_slot_error_m")),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/mean_guidance_tracking_error_m",
        eval_summary.get("mean_guidance_tracking_error_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/mean_route_guidance_tracking_error_m",
        eval_summary.get("mean_route_guidance_tracking_error_m"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/mean_guidance_latency_ms",
        eval_summary.get("mean_guidance_latency_ms"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/guidance_cache_hit_rate",
        eval_summary.get("guidance_cache_hit_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/guidance_non_fallback_rate",
        eval_summary.get("guidance_non_fallback_rate"),
        transitions_collected,
    )
    log_scalar(
        writer,
        "periodic_eval/mean_min_clearance_m",
        eval_summary.get("mean_min_clearance_m"),
        transitions_collected,
    )
    for reason, count in dict(eval_summary.get("done_reason_counts") or {}).items():
        log_scalar(writer, f"periodic_eval/done_reason_counts/{reason}", count, transitions_collected)

def _run_periodic_multi_eval(
    *,
    checkpoint_path: Path,
    transitions_collected: int,
    seed: int,
    device: str | None,
    episodes: int,
    num_agents: int | None,
    experiment_config: MultiExperimentConfig,
    run_label: str,
    selection_record: dict[str, object] | None,
) -> dict[str, object]:
    if selection_record is not None:
        eval_summary = dict(selection_record.get("selection_eval_summary") or {})
    elif (
        num_agents is None
        and bool(experiment_config.size_invariance.enabled)
        and int(experiment_config.size_invariance.bucket_eval_episodes) > 0
    ):
        eval_summary = evaluate_size_buckets(
            checkpoint_path=checkpoint_path,
            episodes_per_bucket=max(int(episodes), int(experiment_config.size_invariance.bucket_eval_episodes)),
            seed=seed,
            device=device,
            team_sizes=experiment_config.size_invariance.bucket_team_sizes,
            experiment_config=experiment_config,
        )
    else:
        eval_summary = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=episodes,
            seed=seed,
            device=device,
            num_agents=num_agents,
            experiment_config=experiment_config,
        )
    output_dir = allocate_replay_artifacts(
        "multi",
        run_name=f"{run_label}_step_{int(transitions_collected):06d}_periodic_eval",
    ).output_dir
    eval_summary["checkpoint_path"] = str(checkpoint_path)
    eval_summary["step"] = int(transitions_collected)
    eval_summary["summary_path"] = str(output_dir / "eval_summary.json")
    write_json(output_dir / "eval_summary.json", eval_summary)
    return eval_summary

def _run_periodic_multi_replay(
    *,
    checkpoint_path: Path,
    transitions_collected: int,
    seed: int,
    device: str | None,
    replay_mode: str,
    max_steps: int | None,
    num_agents: int | None,
    experiment_config: MultiExperimentConfig,
    run_label: str,
    render_isaaclab: bool,
    export_video: bool,
    fps: int,
) -> dict[str, object]:
    from multi_gate.replay import run_multi_replay

    output_dir = allocate_replay_artifacts(
        "multi",
        run_name=f"{run_label}_step_{int(transitions_collected):06d}_periodic_replay",
    ).output_dir
    replay_summary = run_multi_replay(
        mode=replay_mode,
        checkpoint_path=checkpoint_path if replay_mode == "checkpoint" else None,
        num_agents=num_agents,
        seed=seed,
        max_steps=max_steps,
        output_dir=output_dir,
        experiment_config=experiment_config,
        device=device,
    )
    replay_summary["step"] = int(transitions_collected)
    replay_summary["checkpoint_path"] = str(checkpoint_path)
    if render_isaaclab:
        replay_summary["isaaclab_render"] = _render_multi_replay_isaaclab(
            replay_summary=replay_summary,
            experiment_config=experiment_config,
            output_dir=output_dir / "isaaclab",
            export_video=export_video,
            fps=fps,
        )
    return replay_summary

def _render_multi_replay_isaaclab(
    *,
    replay_summary: dict[str, object],
    experiment_config: MultiExperimentConfig,
    output_dir: Path,
    export_video: bool,
    fps: int,
) -> dict[str, object]:
    from multi_gate.replay import render_multi_replay_isaaclab_from_summary

    return render_multi_replay_isaaclab_from_summary(
        replay_summary=replay_summary,
        experiment_config=experiment_config,
        output_dir=output_dir,
        export_video=export_video,
        fps=fps,
        camera_mode="picture_in_picture",
        headless=True,
    )