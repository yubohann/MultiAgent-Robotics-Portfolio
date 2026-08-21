"""Expert-data collection and BC warm-start helpers for the multi-agent line."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from multi_gate.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG,
    MultiExperimentConfig,
    is_dynamic_gate_density_scene_mode,
    is_exp3_empty_scene_mode,
    is_gate_2d_scene_mode,
)
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
from multi_gate.replay import HeuristicFormationReplayController
from multi_gate.training import run_training
from shared.runtime.artifacts import allocate_dataset_artifacts, default_run_name, write_json


def collect_expert_demonstrations(
    *,
    experiment_config: MultiExperimentConfig | None = None,
    num_agents: int | None = None,
    max_sampled_agents: int | None = None,
    teacher_actor_checkpoint: str | Path | None = None,
    teacher_device: str | None = None,
    expert_episodes: int | None = None,
    target_retained_episodes: int | None = None,
    collection_workers: int = 1,
    seed: int = 0,
    max_steps_per_episode: int | None = None,
    retain_failed_episodes: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, object]:
    """Collect heuristic expert demonstrations and save them as a dataset."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    bc_config = selected_config.imitation
    total_episodes = int(expert_episodes if expert_episodes is not None else bc_config.expert_episodes)
    early_stop_target = None
    if target_retained_episodes is not None and int(target_retained_episodes) > 0:
        early_stop_target = min(int(target_retained_episodes), total_episodes)

    if output_dir is None:
        artifacts = allocate_dataset_artifacts(
            "multi",
            run_name=run_name or default_run_name(f"{selected_config.experiment_id}_expert"),
        )
        dataset_dir = artifacts.output_dir
    else:
        dataset_dir = Path(output_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    env = MultiGate2DEnv(
        multi_config=selected_config,
        env_config=selected_config.environment,
        observation_config=selected_config.observation,
        formation_config=selected_config.formation,
        planner_config=selected_config.planner,
    )
    episode_step_limit = int(max_steps_per_episode if max_steps_per_episode is not None else bc_config.max_steps_per_episode)
    if int(collection_workers) > 1 and teacher_actor_checkpoint is None:
        return _collect_expert_demonstrations_parallel(
            selected_config=selected_config,
            dataset_dir=dataset_dir,
            num_agents=num_agents,
            max_sampled_agents=max_sampled_agents,
            total_episodes=total_episodes,
            early_stop_target=early_stop_target,
            seed=seed,
            episode_step_limit=episode_step_limit,
            retain_failed_episodes=retain_failed_episodes,
            observation_shapes=env.observation_shapes,
            action_shape=env.action_shape,
            collection_workers=int(collection_workers),
        )

    teacher_agent: GraphMASACAgent | None = None
    teacher_source = "heuristic_replay_controller"
    if teacher_actor_checkpoint is not None:
        teacher_agent = GraphMASACAgent.from_defaults(
            obs_shapes=env.observation_shapes,
            device=teacher_device,
            seed=seed,
            obs_config=selected_config.observation,
            masac_config=selected_config.algorithm,
            max_agents_soft=selected_config.max_agents_soft,
            build_replay_buffer=False,
        )
        teacher_agent.load_actor_checkpoint(teacher_actor_checkpoint)
        teacher_source = "actor_checkpoint_replay_controller"

    obs_records: dict[str, list[np.ndarray]] = {name: [] for name in env.observation_shapes}
    action_records: list[np.ndarray] = []
    episode_summaries: list[dict[str, object]] = []
    episode_records: list[dict[str, object]] = []
    done_reason_counts: dict[str, int] = {}
    stored_episodes = 0
    stored_steps = 0
    sampled_steps = 0
    team_sizes_seen: set[int] = set()
    sampled_episodes_by_team_size: dict[int, int] = defaultdict(int)
    candidate_episode_count = 0

    for episode_idx in range(total_episodes):
        episode_seed = seed + episode_idx
        active_num_agents = _sample_team_size(
            seed=episode_seed,
            num_agents=num_agents,
            max_sampled_agents=max_sampled_agents,
            experiment_config=selected_config,
        )
        team_sizes_seen.add(active_num_agents)
        sampled_episodes_by_team_size[int(active_num_agents)] += 1
        observation, _ = env.reset(seed=episode_seed, num_agents=active_num_agents)
        controller = None if teacher_agent is not None else HeuristicFormationReplayController(env)
        initial_snapshot = env.snapshot()
        initial_goal_distance_m = float(initial_snapshot.goal_distance_m)

        episode_obs: dict[str, list[np.ndarray]] = {name: [] for name in env.observation_shapes}
        episode_actions: list[np.ndarray] = []
        done_reason = "timeout"
        last_info: dict[str, object] = {}
        steps = 0
        episode_min_pair_distance_m = float("inf")
        episode_max_slot_error_m = 0.0
        episode_formation_line_collapse_failure = False
        episode_min_formation_lateral_band_count: int | None = None
        episode_max_formation_line_collapse_score = 0.0
        guidance_tracking_errors_m: list[float] = []

        for step_idx in range(episode_step_limit):
            action = (
                teacher_agent.act(observation, deterministic=True)
                if teacher_agent is not None
                else controller.act()
            )
            for name, array in observation.items():
                episode_obs[name].append(np.asarray(array, dtype=np.float32).copy())
            episode_actions.append(np.asarray(action, dtype=np.float32).copy())
            sampled_steps += 1
            observation, _, terminated, truncated, info = env.step(action)
            last_info = dict(info)
            steps = step_idx + 1
            min_pair_distance = info.get("min_pair_distance_m")
            if min_pair_distance is not None:
                episode_min_pair_distance_m = min(episode_min_pair_distance_m, float(min_pair_distance))
            max_slot_error = info.get("max_slot_error_m")
            if max_slot_error is not None:
                episode_max_slot_error_m = max(episode_max_slot_error_m, float(max_slot_error))
            if bool(info.get("formation_shape_active", False)):
                lateral_band_count = info.get("formation_lateral_band_count")
                if lateral_band_count is not None:
                    resolved_band_count = int(lateral_band_count)
                    episode_min_formation_lateral_band_count = (
                        resolved_band_count
                        if episode_min_formation_lateral_band_count is None
                        else min(episode_min_formation_lateral_band_count, resolved_band_count)
                    )
            episode_max_formation_line_collapse_score = max(
                episode_max_formation_line_collapse_score,
                float(info.get("formation_line_collapse_score") or 0.0),
            )
            episode_formation_line_collapse_failure = bool(
                episode_formation_line_collapse_failure
                or info.get("formation_line_collapse_failure", False)
            )
            guidance_tracking_error = info.get("guidance_tracking_error_m")
            if guidance_tracking_error is not None:
                guidance_tracking_errors_m.append(float(guidance_tracking_error))
            if terminated or truncated:
                done_reason = str(info.get("done_reason") or "unknown")
                break

        done_reason_counts[done_reason] = done_reason_counts.get(done_reason, 0) + 1
        final_snapshot = env.snapshot()
        final_goal_distance_m = float(final_snapshot.goal_distance_m)
        goal_distance_improvement_m = max(0.0, initial_goal_distance_m - final_goal_distance_m)
        goal_progress_ratio = float(
            np.clip(
                goal_distance_improvement_m / max(initial_goal_distance_m, 1.0e-6),
                0.0,
                1.0,
            )
        )
        episode_mean_guidance_tracking_error_m = (
            float(np.mean(np.asarray(guidance_tracking_errors_m, dtype=np.float32)))
            if guidance_tracking_errors_m
            else None
        )
        final_guidance_tracking_error_m = env._guidance_tracking_error_m(final_snapshot.virtual_center_xy)
        episode_metrics = {
            "initial_goal_distance_m": initial_goal_distance_m,
            "final_goal_distance_m": final_goal_distance_m,
            "goal_distance_improvement_m": goal_distance_improvement_m,
            "goal_progress_ratio": goal_progress_ratio,
            "episode_min_pair_distance_m": (
                None if not np.isfinite(episode_min_pair_distance_m) else float(episode_min_pair_distance_m)
            ),
            "episode_max_slot_error_m": float(episode_max_slot_error_m),
            "episode_mean_guidance_tracking_error_m": episode_mean_guidance_tracking_error_m,
            "final_guidance_tracking_error_m": (
                None if final_guidance_tracking_error_m is None else float(final_guidance_tracking_error_m)
            ),
            "height_contract_passed": bool(last_info.get("height_contract_passed", True)),
            "height_escape_failure": bool(last_info.get("height_escape_failure", False)),
            "side_bypass_failure": bool(last_info.get("side_bypass_failure", False)),
            "corridor_miss_failure": bool(last_info.get("corridor_miss_failure", False)),
            "formation_line_collapse_failure": bool(
                episode_formation_line_collapse_failure
                or last_info.get("formation_line_collapse_failure", False)
            ),
            "episode_min_formation_lateral_band_count": episode_min_formation_lateral_band_count,
            "episode_max_formation_line_collapse_score": float(episode_max_formation_line_collapse_score),
            "corridor_completed": bool(
                last_info.get("corridor_completed", int(last_info.get("dynamic_gate_count") or 0) <= 0)
            ),
            "corridor_through_success": _expert_corridor_through_success_from_info(last_info),
            "final_virtual_center_xy": [
                float(final_snapshot.virtual_center_xy[0]),
                float(final_snapshot.virtual_center_xy[1]),
            ],
            "path_index": int(last_info.get("path_index") or final_snapshot.path_index),
            "min_clearance_m": _optional_float(last_info.get("min_clearance_m")),
            "dynamic_gate_collision": bool(last_info.get("dynamic_gate_collision", False)),
            "dynamic_gate_count": int(last_info.get("dynamic_gate_count") or 0),
            "moving_gate_speed_mps": _optional_float(last_info.get("moving_gate_speed_mps")),
            "moving_gate_amplitude_m": _optional_float(last_info.get("moving_gate_amplitude_m")),
        }
        retention = _classify_expert_episode_retention(
            done_reason=done_reason,
            num_agents=active_num_agents,
            episode_metrics=episode_metrics,
            experiment_config=selected_config,
            retain_failed_episodes=retain_failed_episodes,
        )
        fragment_step_count = 0
        training_fragment_only = False
        # Formal expert collection must not turn a collision episode into BC
        # data.  Collision prefixes can be collected by a separate debug or
        # DAgger-fragment pipeline, but they cannot satisfy the expert quality
        # gate for the current full-route curriculum.
        episode_summary = {
            "episode_index": episode_idx,
            "seed": episode_seed,
            "num_agents": active_num_agents,
            "steps": steps,
            "done_reason": done_reason,
            "retained": False,
            "retention_candidate": bool(retention["keep"]),
            "retention_reason": str(retention["reason"]),
            "retention_score": float(retention["score"]),
            "training_fragment_only": bool(training_fragment_only),
            "fragment_steps": int(fragment_step_count),
            **episode_metrics,
        }
        episode_summaries.append(episode_summary)
        episode_records.append(
            {
                "summary": episode_summary,
                "observations": episode_obs,
                "actions": episode_actions,
                "keep_candidate": bool(retention["keep"]),
                "retention_reason": str(retention["reason"]),
                "retention_score": float(retention["score"]),
                "fragment_steps": int(fragment_step_count),
            }
        )
        if bool(retention["keep"]):
            candidate_episode_count += 1
        if early_stop_target is not None and candidate_episode_count >= early_stop_target:
            break

    if retain_failed_episodes:
        selected_episode_indices = [
            idx for idx, record in enumerate(episode_records) if bool(record["keep_candidate"])
        ]
        selection_stats = {
            "policy": {
                "strategy": "retain_quality_filtered_non_collision",
                "target_floor_by_team_size": {
                    str(int(team_size)): int(count)
                    for team_size, count in sorted(sampled_episodes_by_team_size.items())
                },
                "soft_cap_by_team_size": {
                    str(int(team_size)): int(count)
                    for team_size, count in sorted(sampled_episodes_by_team_size.items())
                },
            },
            "rejected_selected_reason_by_index": {},
        }
    else:
        selected_episode_indices, selection_stats = _select_balanced_expert_episode_indices(episode_records)
    selected_episode_set = set(int(index) for index in selected_episode_indices)
    stored_episodes_by_team_size: dict[int, int] = defaultdict(int)
    stored_steps_by_team_size: dict[int, int] = defaultdict(int)
    candidate_episodes_by_team_size: dict[int, int] = defaultdict(int)
    rejected_reason_counts_by_team_size: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for episode_idx, episode_record in enumerate(episode_records):
        summary = episode_record["summary"]
        team_size = int(summary["num_agents"])
        episode_actions = episode_record["actions"]
        if bool(episode_record["keep_candidate"]):
            candidate_episodes_by_team_size[team_size] += 1
        keep_episode = episode_idx in selected_episode_set
        summary["retained"] = bool(keep_episode)
        if keep_episode and episode_actions:
            store_steps = int(episode_record.get("fragment_steps") or len(episode_actions))
            store_steps = max(0, min(store_steps, len(episode_actions)))
            stored_episodes += 1
            stored_steps += store_steps
            stored_episodes_by_team_size[team_size] += 1
            stored_steps_by_team_size[team_size] += store_steps
            for name in env.observation_shapes:
                obs_records[name].extend(episode_record["observations"][name][:store_steps])
            action_records.extend(episode_actions[:store_steps])
        else:
            rejected_reason = (
                str(episode_record["retention_reason"])
                if not bool(episode_record["keep_candidate"])
                else str(selection_stats["rejected_selected_reason_by_index"].get(episode_idx, "balance_soft_cap"))
            )
            summary["rejected_reason"] = rejected_reason
            rejected_reason_counts_by_team_size[team_size][rejected_reason] += 1

    if not action_records:
        summary_path = dataset_dir / "expert_collection_summary.json"
        write_json(
            summary_path,
            {
                "experiment_id": selected_config.experiment_id,
                "source": teacher_source,
                "teacher_actor_checkpoint": None if teacher_actor_checkpoint is None else str(teacher_actor_checkpoint),
                "seed": seed,
                "total_episodes": total_episodes,
                "sampled_episodes": len(episode_records),
                "target_retained_episodes": early_stop_target,
                "stored_episodes": 0,
                "stored_steps": 0,
                "sampled_steps": sampled_steps,
                "done_reason_counts": done_reason_counts,
                "team_sizes_seen": sorted(team_sizes_seen),
                "sampled_episodes_by_team_size": {
                    str(team_size): int(count)
                    for team_size, count in sorted(sampled_episodes_by_team_size.items())
                },
                "candidate_episodes_by_team_size": {
                    str(team_size): int(count)
                    for team_size, count in sorted(candidate_episodes_by_team_size.items())
                },
                "stored_episodes_by_team_size": {},
                "stored_steps_by_team_size": {},
                "rejected_reason_counts_by_team_size": {
                    str(team_size): {
                        str(reason): int(count)
                        for reason, count in sorted(reason_counts.items())
                    }
                    for team_size, reason_counts in sorted(rejected_reason_counts_by_team_size.items())
                },
                "team_size_retention_policy": selection_stats["policy"],
                "retain_failed_episodes": retain_failed_episodes,
                "num_agents": num_agents,
                "max_sampled_agents": max_sampled_agents,
                "observation_shapes": env.observation_shapes,
                "action_shape": list(env.action_shape),
                "imitation_config": asdict(selected_config.imitation),
                "episode_summaries": episode_summaries,
                "failure": "no_retained_expert_demonstrations",
                "summary_path": str(summary_path),
            },
        )
        raise RuntimeError(
            "No expert demonstrations were retained after quality and team-balance filtering. "
            "Try increasing expert_episodes or fixing the expert/layout; collision, bypass, fly-over, and "
            "no-progress episodes are never retained."
        )

    observations = {
        name: np.asarray(records, dtype=np.float32)
        for name, records in obs_records.items()
    }
    actions = np.asarray(action_records, dtype=np.float32)

    metadata = {
        "experiment_id": selected_config.experiment_id,
        "source": teacher_source,
        "teacher_actor_checkpoint": None if teacher_actor_checkpoint is None else str(teacher_actor_checkpoint),
        "seed": seed,
        "total_episodes": total_episodes,
        "sampled_episodes": len(episode_records),
        "target_retained_episodes": early_stop_target,
        "stored_episodes": stored_episodes,
        "stored_steps": stored_steps,
        "sampled_steps": sampled_steps,
        "done_reason_counts": done_reason_counts,
        "team_sizes_seen": sorted(team_sizes_seen),
        "sampled_episodes_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(sampled_episodes_by_team_size.items())
        },
        "candidate_episodes_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(candidate_episodes_by_team_size.items())
        },
        "stored_episodes_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(stored_episodes_by_team_size.items())
        },
        "stored_steps_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(stored_steps_by_team_size.items())
        },
        "rejected_reason_counts_by_team_size": {
            str(team_size): {
                str(reason): int(count)
                for reason, count in sorted(reason_counts.items())
            }
            for team_size, reason_counts in sorted(rejected_reason_counts_by_team_size.items())
        },
        "team_size_retention_policy": selection_stats["policy"],
        "retain_failed_episodes": retain_failed_episodes,
        "num_agents": num_agents,
        "max_sampled_agents": max_sampled_agents,
        "observation_shapes": env.observation_shapes,
        "action_shape": list(actions.shape[1:]),
        "imitation_config": asdict(selected_config.imitation),
        "episode_summaries": episode_summaries,
    }
    dataset_path = dataset_dir / selected_config.imitation.dataset_name
    _save_expert_dataset(dataset_path, observations=observations, actions=actions, metadata=metadata)
    summary_path = dataset_dir / "expert_collection_summary.json"
    metadata["summary_path"] = str(summary_path)
    write_json(summary_path, metadata)
    return {
        "dataset_path": str(dataset_path),
        "summary_path": str(summary_path),
        "stored_steps": stored_steps,
        "stored_episodes": stored_episodes,
        "team_sizes_seen": sorted(team_sizes_seen),
        "done_reason_counts": done_reason_counts,
        "output_dir": str(dataset_dir),
    }


def run_actor_behavior_cloning(
    *,
    dataset_path: str | Path,
    experiment_config: MultiExperimentConfig | None = None,
    device: str | None = None,
    seed: int = 0,
    initial_actor_checkpoint: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    validation_split: float | None = None,
    target_log_std: float | None = None,
    log_std_penalty_scale: float | None = None,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    actor_checkpoint_name: str | None = None,
) -> dict[str, object]:
    """Train the actor with supervised BC on collected expert data."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    bc_config = selected_config.imitation

    if output_dir is None:
        artifacts = allocate_dataset_artifacts(
            "multi",
            run_name=run_name or default_run_name(f"{selected_config.experiment_id}_bc"),
        )
        bc_dir = artifacts.output_dir
    else:
        bc_dir = Path(output_dir)
        bc_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_expert_dataset(dataset_path)
    observations = dataset["observations"]
    actions = dataset["actions"]
    obs_shapes = {name: tuple(array.shape[1:]) for name, array in observations.items()}
    sample_count = int(actions.shape[0])
    if sample_count <= 0:
        raise RuntimeError(f"Expert dataset is empty: {dataset_path}")

    agent = GraphMASACAgent.from_defaults(
        obs_shapes=obs_shapes,
        device=device,
        seed=seed,
        obs_config=selected_config.observation,
        masac_config=selected_config.algorithm,
        max_agents_soft=selected_config.max_agents_soft,
    )
    initial_actor_metadata: dict[str, object] | None = None
    if initial_actor_checkpoint is not None:
        initial_actor_metadata = agent.load_actor_checkpoint(initial_actor_checkpoint)
    optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=float(learning_rate if learning_rate is not None else bc_config.learning_rate),
        weight_decay=float(weight_decay if weight_decay is not None else bc_config.weight_decay),
    )
    total_epochs = int(epochs if epochs is not None else bc_config.epochs)
    train_batch_size = int(batch_size if batch_size is not None else bc_config.batch_size)
    val_fraction = float(validation_split if validation_split is not None else bc_config.validation_split)
    target_log_std = bc_config.target_log_std if target_log_std is None else float(target_log_std)
    log_std_penalty_scale = float(
        bc_config.log_std_penalty_scale if log_std_penalty_scale is None else log_std_penalty_scale
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    val_count = int(round(sample_count * val_fraction)) if sample_count >= 10 else 0
    if val_count >= sample_count:
        val_count = max(sample_count - 1, 0)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if train_indices.size == 0:
        train_indices = indices
        val_indices = np.asarray([], dtype=np.int64)
    train_indices, risk_sampling_summary = _expand_dynamic_gate_risk_train_indices(
        train_indices=train_indices,
        observations=observations,
        experiment_config=selected_config,
    )

    history: list[dict[str, float]] = []
    best_metric = float("inf")
    best_epoch = 0
    best_actor_state = {
        key: value.detach().cpu().clone()
        for key, value in agent.actor.state_dict().items()
    }

    for epoch_idx in range(1, total_epochs + 1):
        rng.shuffle(train_indices)
        batch_losses: list[float] = []
        batch_bc_losses: list[float] = []
        batch_std_penalties: list[float] = []

        for start in range(0, int(train_indices.size), train_batch_size):
            batch_ids = train_indices[start : start + train_batch_size]
            batch_obs = _batch_observations(observations, batch_ids, agent.device)
            batch_actions = torch.as_tensor(actions[batch_ids], dtype=torch.float32, device=agent.device)
            loss, metrics = agent.actor_behavior_clone_loss(
                batch_obs,
                batch_actions,
                target_log_std=target_log_std,
                log_std_penalty_scale=log_std_penalty_scale,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(metrics["total_loss"])
            batch_bc_losses.append(metrics["bc_loss"])
            batch_std_penalties.append(metrics["log_std_penalty"])

        train_total_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        train_bc_loss = float(np.mean(batch_bc_losses)) if batch_bc_losses else 0.0
        train_std_penalty = float(np.mean(batch_std_penalties)) if batch_std_penalties else 0.0
        val_total_loss, val_bc_loss, val_std_penalty = _evaluate_bc_epoch(
            agent=agent,
            observations=observations,
            actions=actions,
            indices=val_indices,
            target_log_std=target_log_std,
            log_std_penalty_scale=log_std_penalty_scale,
        )
        selection_metric = val_total_loss if val_indices.size > 0 else train_total_loss
        if selection_metric <= best_metric:
            best_metric = selection_metric
            best_epoch = epoch_idx
            best_actor_state = {
                key: value.detach().cpu().clone()
                for key, value in agent.actor.state_dict().items()
            }
        history.append(
            {
                "epoch": float(epoch_idx),
                "train_total_loss": train_total_loss,
                "train_bc_loss": train_bc_loss,
                "train_log_std_penalty": train_std_penalty,
                "val_total_loss": val_total_loss,
                "val_bc_loss": val_bc_loss,
                "val_log_std_penalty": val_std_penalty,
            }
        )

    agent.actor.load_state_dict(best_actor_state)
    actor_checkpoint_path = agent.save_actor_checkpoint(
        bc_dir / (actor_checkpoint_name or bc_config.actor_checkpoint_name),
        metadata={
            "source": "behavior_cloning",
            "experiment_id": selected_config.experiment_id,
            "dataset_path": str(dataset_path),
            "initial_actor_checkpoint": (
                str(initial_actor_checkpoint) if initial_actor_checkpoint is not None else None
            ),
            "initial_actor_metadata": initial_actor_metadata or {},
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "sample_count": sample_count,
            "train_sample_count": int(train_indices.size),
            "val_sample_count": int(val_indices.size),
            "risk_sampling": risk_sampling_summary,
            "history": history,
        },
    )
    summary = {
        "experiment_id": selected_config.experiment_id,
        "dataset_path": str(dataset_path),
        "actor_checkpoint_path": str(actor_checkpoint_path),
        "initial_actor_checkpoint": (
            str(initial_actor_checkpoint) if initial_actor_checkpoint is not None else None
        ),
        "initial_actor_metadata": initial_actor_metadata or {},
        "epochs": total_epochs,
        "batch_size": train_batch_size,
        "device": str(agent.device),
        "sample_count": sample_count,
        "train_sample_count": int(train_indices.size),
        "val_sample_count": int(val_indices.size),
        "risk_sampling": risk_sampling_summary,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
    }
    summary_path = bc_dir / "bc_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    summary["output_dir"] = str(bc_dir)
    return summary


def _collect_expert_demonstrations_parallel(
    *,
    selected_config: MultiExperimentConfig,
    dataset_dir: Path,
    num_agents: int | None,
    max_sampled_agents: int | None,
    total_episodes: int,
    early_stop_target: int | None,
    seed: int,
    episode_step_limit: int,
    retain_failed_episodes: bool,
    observation_shapes: dict[str, tuple[int, ...]],
    action_shape: tuple[int, ...],
    collection_workers: int,
) -> dict[str, object]:
    worker_count = max(1, min(int(collection_workers), int(total_episodes), os.cpu_count() or 1))
    obs_records: dict[str, list[np.ndarray]] = {name: [] for name in observation_shapes}
    action_records: list[np.ndarray] = []
    episode_summaries: list[dict[str, object]] = []
    episode_records: list[dict[str, object]] = []
    done_reason_counts: dict[str, int] = {}
    stored_episodes = 0
    stored_steps = 0
    sampled_steps = 0
    team_sizes_seen: set[int] = set()
    sampled_episodes_by_team_size: dict[int, int] = defaultdict(int)
    candidate_episode_count = 0

    executor_backend = "process"
    executor_fallback_error: str | None = None

    progress_path = dataset_dir / "collection_progress.json"

    def _write_collection_progress(backend_name: str) -> None:
        write_json(
            progress_path,
            {
                "backend": backend_name,
                "worker_count": worker_count,
                "sampled_episodes": len(episode_records),
                "candidate_episodes": int(candidate_episode_count),
                "target_retained_episodes": early_stop_target,
                "sampled_steps": int(sampled_steps),
                "done_reason_counts": done_reason_counts,
                "fallback_error": executor_fallback_error,
            },
        )

    def _ingest_episode_result(result: dict[str, object]) -> None:
        nonlocal sampled_steps
        nonlocal candidate_episode_count
        sampled_steps += int(result["steps"])
        active_num_agents = int(result["num_agents"])
        team_sizes_seen.add(active_num_agents)
        sampled_episodes_by_team_size[active_num_agents] += 1
        done_reason = str(result["done_reason"])
        done_reason_counts[done_reason] = done_reason_counts.get(done_reason, 0) + 1
        retention = _classify_expert_episode_retention(
            done_reason=done_reason,
            num_agents=active_num_agents,
            episode_metrics=dict(result["episode_metrics"]),
            experiment_config=selected_config,
            retain_failed_episodes=retain_failed_episodes,
        )
        episode_summary = {
            "episode_index": int(result["episode_index"]),
            "seed": int(result["seed"]),
            "num_agents": active_num_agents,
            "steps": int(result["steps"]),
            "done_reason": done_reason,
            "retained": False,
            "retention_candidate": bool(retention["keep"]),
            "retention_reason": str(retention["reason"]),
            "retention_score": float(retention["score"]),
            **dict(result["episode_metrics"]),
        }
        episode_summaries.append(episode_summary)
        episode_records.append(
            {
                "summary": episode_summary,
                "observations": result["observations"],
                "actions": result["actions"],
                "keep_candidate": bool(retention["keep"]),
                "retention_reason": str(retention["reason"]),
                "retention_score": float(retention["score"]),
            }
        )
        if bool(retention["keep"]):
            candidate_episode_count += 1

    def _run_collection_batches(
        executor_factory: type[ProcessPoolExecutor] | type[ThreadPoolExecutor],
        backend_name: str,
    ) -> None:
        nonlocal next_episode_idx
        nonlocal sampled_steps
        nonlocal candidate_episode_count
        next_episode_idx = 0
        with executor_factory(max_workers=worker_count) as executor:
            while next_episode_idx < int(total_episodes):
                batch_count = min(worker_count, int(total_episodes) - next_episode_idx)
                futures = [
                    executor.submit(
                        _collect_expert_episode_worker,
                        selected_config,
                        num_agents,
                        max_sampled_agents,
                        int(seed),
                        int(next_episode_idx + offset),
                        int(episode_step_limit),
                    )
                    for offset in range(batch_count)
                ]
                next_episode_idx += batch_count
                for future in as_completed(futures):
                    _ingest_episode_result(dict(future.result()))
                _write_collection_progress(backend_name)
                if early_stop_target is not None and candidate_episode_count >= int(early_stop_target):
                    break

    def _run_collection_batches_external_processes(backend_name: str) -> None:
        nonlocal next_episode_idx
        worker_dir = dataset_dir / "external_worker_results"
        worker_dir.mkdir(parents=True, exist_ok=True)
        payload_path = worker_dir / "payload.pkl"
        with payload_path.open("wb") as payload_file:
            pickle.dump(
                {
                    "experiment_config": selected_config,
                    "num_agents": num_agents,
                    "max_sampled_agents": max_sampled_agents,
                    "seed": int(seed),
                    "episode_step_limit": int(episode_step_limit),
                },
                payload_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        worker_code = (
            "import pickle, pathlib; "
            "from multi_gate.imitation import _collect_expert_episode_worker; "
            "payload_path=pathlib.Path(r'{payload}'); "
            "result_path=pathlib.Path(r'{result}'); "
            "episode_idx={episode_idx}; "
            "payload=pickle.load(payload_path.open('rb')); "
            "result=_collect_expert_episode_worker("
            "payload['experiment_config'], payload['num_agents'], payload['max_sampled_agents'], "
            "payload['seed'], episode_idx, payload['episode_step_limit']); "
            "pickle.dump(result, result_path.open('wb'), protocol=pickle.HIGHEST_PROTOCOL)"
        )
        next_episode_idx = 0
        while next_episode_idx < int(total_episodes):
            batch_count = min(worker_count, int(total_episodes) - next_episode_idx)
            launched: list[tuple[subprocess.Popen[bytes], Path, Path, Path, int]] = []
            for offset in range(batch_count):
                episode_idx = int(next_episode_idx + offset)
                result_path = worker_dir / f"episode_{episode_idx:05d}.pkl"
                stdout_path = worker_dir / f"episode_{episode_idx:05d}.stdout.log"
                stderr_path = worker_dir / f"episode_{episode_idx:05d}.stderr.log"
                cmd = [
                    sys.executable,
                    "-c",
                    worker_code.format(
                        payload=str(payload_path),
                        result=str(result_path),
                        episode_idx=episode_idx,
                    ),
                ]
                stdout_file = stdout_path.open("wb")
                stderr_file = stderr_path.open("wb")
                process = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                stdout_file.close()
                stderr_file.close()
                launched.append((process, result_path, stdout_path, stderr_path, episode_idx))
            next_episode_idx += batch_count
            for process, result_path, _stdout_path, stderr_path, episode_idx in launched:
                return_code = process.wait()
                if return_code != 0 or not result_path.exists():
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
                    raise RuntimeError(
                        f"External expert worker failed for episode {episode_idx} "
                        f"with return code {return_code}: {stderr_text[-2000:]}"
                    )
                with result_path.open("rb") as result_file:
                    _ingest_episode_result(dict(pickle.load(result_file)))
                _write_collection_progress(backend_name)
                if early_stop_target is not None and candidate_episode_count >= int(early_stop_target):
                    break
            if early_stop_target is not None and candidate_episode_count >= int(early_stop_target):
                break

    next_episode_idx = 0
    try:
        _run_collection_batches(ProcessPoolExecutor, executor_backend)
    except (PermissionError, OSError) as exc:
        winerror = getattr(exc, "winerror", None)
        if isinstance(exc, PermissionError) or winerror == 5:
            executor_backend = "external_process_after_process_pool_permission_error"
            executor_fallback_error = f"{type(exc).__name__}: {exc}"
            obs_records = {name: [] for name in observation_shapes}
            action_records = []
            episode_summaries = []
            episode_records = []
            done_reason_counts = {}
            stored_episodes = 0
            stored_steps = 0
            sampled_steps = 0
            team_sizes_seen = set()
            sampled_episodes_by_team_size = defaultdict(int)
            candidate_episode_count = 0
            _write_collection_progress(executor_backend)
            _run_collection_batches_external_processes(executor_backend)
        else:
            raise

    episode_records.sort(key=lambda record: int(record["summary"]["episode_index"]))
    episode_summaries.sort(key=lambda summary: int(summary["episode_index"]))
    if retain_failed_episodes:
        selected_episode_indices = [
            idx for idx, record in enumerate(episode_records) if bool(record["keep_candidate"])
        ]
        selection_stats = {
            "policy": {
                "strategy": "retain_quality_filtered_non_collision",
                "target_floor_by_team_size": {
                    str(int(team_size)): int(count)
                    for team_size, count in sorted(sampled_episodes_by_team_size.items())
                },
                "soft_cap_by_team_size": {
                    str(int(team_size)): int(count)
                    for team_size, count in sorted(sampled_episodes_by_team_size.items())
                },
            },
            "rejected_selected_reason_by_index": {},
        }
    else:
        selected_episode_indices, selection_stats = _select_balanced_expert_episode_indices(episode_records)
    selected_episode_set = set(int(index) for index in selected_episode_indices)
    stored_episodes_by_team_size: dict[int, int] = defaultdict(int)
    stored_steps_by_team_size: dict[int, int] = defaultdict(int)
    candidate_episodes_by_team_size: dict[int, int] = defaultdict(int)
    rejected_reason_counts_by_team_size: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for episode_idx, episode_record in enumerate(episode_records):
        summary = episode_record["summary"]
        team_size = int(summary["num_agents"])
        episode_actions = list(episode_record["actions"])
        if bool(episode_record["keep_candidate"]):
            candidate_episodes_by_team_size[team_size] += 1
        keep_episode = episode_idx in selected_episode_set
        summary["retained"] = bool(keep_episode)
        if keep_episode and episode_actions:
            stored_episodes += 1
            stored_steps += len(episode_actions)
            stored_episodes_by_team_size[team_size] += 1
            stored_steps_by_team_size[team_size] += len(episode_actions)
            for name in observation_shapes:
                obs_records[name].extend(episode_record["observations"][name])
            action_records.extend(episode_actions)
        else:
            rejected_reason = (
                str(episode_record["retention_reason"])
                if not bool(episode_record["keep_candidate"])
                else str(selection_stats["rejected_selected_reason_by_index"].get(episode_idx, "balance_soft_cap"))
            )
            summary["rejected_reason"] = rejected_reason
            rejected_reason_counts_by_team_size[team_size][rejected_reason] += 1

    if not action_records:
        raise RuntimeError(
            "No expert demonstrations were retained after quality and team-balance filtering. "
            "Try increasing expert_episodes or fixing the expert/layout; collision, bypass, fly-over, and "
            "no-progress episodes are never retained."
        )

    observations = {
        name: np.asarray(records, dtype=np.float32)
        for name, records in obs_records.items()
    }
    actions = np.asarray(action_records, dtype=np.float32)
    metadata = {
        "experiment_id": selected_config.experiment_id,
        "source": "heuristic_replay_controller",
        "teacher_actor_checkpoint": None,
        "seed": seed,
        "total_episodes": int(total_episodes),
        "sampled_episodes": len(episode_records),
        "target_retained_episodes": early_stop_target,
        "collection_workers": worker_count,
        "collection_backend": executor_backend,
        "collection_fallback_error": executor_fallback_error,
        "stored_episodes": stored_episodes,
        "stored_steps": stored_steps,
        "sampled_steps": sampled_steps,
        "done_reason_counts": done_reason_counts,
        "team_sizes_seen": sorted(team_sizes_seen),
        "sampled_episodes_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(sampled_episodes_by_team_size.items())
        },
        "candidate_episodes_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(candidate_episodes_by_team_size.items())
        },
        "stored_episodes_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(stored_episodes_by_team_size.items())
        },
        "stored_steps_by_team_size": {
            str(team_size): int(count)
            for team_size, count in sorted(stored_steps_by_team_size.items())
        },
        "rejected_reason_counts_by_team_size": {
            str(team_size): {
                str(reason): int(count)
                for reason, count in sorted(reason_counts.items())
            }
            for team_size, reason_counts in sorted(rejected_reason_counts_by_team_size.items())
        },
        "team_size_retention_policy": selection_stats["policy"],
        "retain_failed_episodes": retain_failed_episodes,
        "num_agents": num_agents,
        "max_sampled_agents": max_sampled_agents,
        "observation_shapes": observation_shapes,
        "action_shape": list(action_shape),
        "imitation_config": asdict(selected_config.imitation),
        "episode_summaries": episode_summaries,
    }
    dataset_path = dataset_dir / selected_config.imitation.dataset_name
    _save_expert_dataset(dataset_path, observations=observations, actions=actions, metadata=metadata)
    summary_path = dataset_dir / "expert_collection_summary.json"
    metadata["summary_path"] = str(summary_path)
    write_json(summary_path, metadata)
    return {
        "dataset_path": str(dataset_path),
        "summary_path": str(summary_path),
        "source": metadata.get("source"),
        "teacher_actor_checkpoint": metadata.get("teacher_actor_checkpoint"),
        "stored_steps": stored_steps,
        "stored_episodes": stored_episodes,
        "team_sizes_seen": sorted(team_sizes_seen),
        "done_reason_counts": done_reason_counts,
        "output_dir": str(dataset_dir),
        "collection_workers": worker_count,
    }


def _collect_expert_episode_worker(
    experiment_config: MultiExperimentConfig,
    num_agents: int | None,
    max_sampled_agents: int | None,
    seed: int,
    episode_idx: int,
    episode_step_limit: int,
) -> dict[str, object]:
    env = MultiGate2DEnv(
        multi_config=experiment_config,
        env_config=experiment_config.environment,
        observation_config=experiment_config.observation,
        formation_config=experiment_config.formation,
        planner_config=experiment_config.planner,
    )
    episode_seed = int(seed) + int(episode_idx)
    active_num_agents = _sample_team_size(
        seed=episode_seed,
        num_agents=num_agents,
        max_sampled_agents=max_sampled_agents,
        experiment_config=experiment_config,
    )
    observation, _ = env.reset(seed=episode_seed, num_agents=active_num_agents)
    controller = HeuristicFormationReplayController(env)
    initial_snapshot = env.snapshot()
    initial_goal_distance_m = float(initial_snapshot.goal_distance_m)
    episode_obs: dict[str, list[np.ndarray]] = {name: [] for name in env.observation_shapes}
    episode_actions: list[np.ndarray] = []
    done_reason = "timeout"
    last_info: dict[str, object] = {}
    steps = 0
    episode_min_pair_distance_m = float("inf")
    episode_max_slot_error_m = 0.0
    episode_formation_line_collapse_failure = False
    episode_min_formation_lateral_band_count: int | None = None
    episode_max_formation_line_collapse_score = 0.0
    guidance_tracking_errors_m: list[float] = []

    for step_idx in range(int(episode_step_limit)):
        action = controller.act()
        for name, array in observation.items():
            episode_obs[name].append(np.asarray(array, dtype=np.float32).copy())
        episode_actions.append(np.asarray(action, dtype=np.float32).copy())
        observation, _, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        steps = step_idx + 1
        min_pair_distance = info.get("min_pair_distance_m")
        if min_pair_distance is not None:
            episode_min_pair_distance_m = min(episode_min_pair_distance_m, float(min_pair_distance))
        max_slot_error = info.get("max_slot_error_m")
        if max_slot_error is not None:
            episode_max_slot_error_m = max(episode_max_slot_error_m, float(max_slot_error))
        if bool(info.get("formation_shape_active", False)):
            lateral_band_count = info.get("formation_lateral_band_count")
            if lateral_band_count is not None:
                resolved_band_count = int(lateral_band_count)
                episode_min_formation_lateral_band_count = (
                    resolved_band_count
                    if episode_min_formation_lateral_band_count is None
                    else min(episode_min_formation_lateral_band_count, resolved_band_count)
                )
        episode_max_formation_line_collapse_score = max(
            episode_max_formation_line_collapse_score,
            float(info.get("formation_line_collapse_score") or 0.0),
        )
        episode_formation_line_collapse_failure = bool(
            episode_formation_line_collapse_failure
            or info.get("formation_line_collapse_failure", False)
        )
        guidance_tracking_error = info.get("guidance_tracking_error_m")
        if guidance_tracking_error is not None:
            guidance_tracking_errors_m.append(float(guidance_tracking_error))
        if terminated or truncated:
            done_reason = str(info.get("done_reason") or "unknown")
            break

    final_snapshot = env.snapshot()
    final_goal_distance_m = float(final_snapshot.goal_distance_m)
    goal_distance_improvement_m = max(0.0, initial_goal_distance_m - final_goal_distance_m)
    goal_progress_ratio = float(
        np.clip(
            goal_distance_improvement_m / max(initial_goal_distance_m, 1.0e-6),
            0.0,
            1.0,
        )
    )
    episode_mean_guidance_tracking_error_m = (
        float(np.mean(np.asarray(guidance_tracking_errors_m, dtype=np.float32)))
        if guidance_tracking_errors_m
        else None
    )
    final_guidance_tracking_error_m = env._guidance_tracking_error_m(final_snapshot.virtual_center_xy)
    return {
        "episode_index": int(episode_idx),
        "seed": episode_seed,
        "num_agents": int(active_num_agents),
        "steps": int(steps),
        "done_reason": done_reason,
        "episode_metrics": {
            "initial_goal_distance_m": initial_goal_distance_m,
            "final_goal_distance_m": final_goal_distance_m,
            "goal_distance_improvement_m": goal_distance_improvement_m,
            "goal_progress_ratio": goal_progress_ratio,
            "episode_min_pair_distance_m": (
                None if not np.isfinite(episode_min_pair_distance_m) else float(episode_min_pair_distance_m)
            ),
            "episode_max_slot_error_m": float(episode_max_slot_error_m),
            "episode_mean_guidance_tracking_error_m": episode_mean_guidance_tracking_error_m,
            "final_guidance_tracking_error_m": (
                None if final_guidance_tracking_error_m is None else float(final_guidance_tracking_error_m)
            ),
            "height_contract_passed": bool(last_info.get("height_contract_passed", True)),
            "height_escape_failure": bool(last_info.get("height_escape_failure", False)),
            "side_bypass_failure": bool(last_info.get("side_bypass_failure", False)),
            "corridor_miss_failure": bool(last_info.get("corridor_miss_failure", False)),
            "formation_line_collapse_failure": bool(
                episode_formation_line_collapse_failure
                or last_info.get("formation_line_collapse_failure", False)
            ),
            "episode_min_formation_lateral_band_count": episode_min_formation_lateral_band_count,
            "episode_max_formation_line_collapse_score": float(episode_max_formation_line_collapse_score),
            "corridor_completed": bool(
                last_info.get("corridor_completed", int(last_info.get("dynamic_gate_count") or 0) <= 0)
            ),
            "corridor_through_success": _expert_corridor_through_success_from_info(last_info),
        },
        "observations": episode_obs,
        "actions": episode_actions,
    }


def _expert_corridor_through_success_from_info(info: dict[str, object]) -> bool:
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


def _classify_expert_episode_retention(
    *,
    done_reason: str,
    num_agents: int,
    episode_metrics: dict[str, float | None],
    experiment_config: MultiExperimentConfig,
    retain_failed_episodes: bool,
) -> dict[str, object]:
    env_config = experiment_config.environment
    formation_config = experiment_config.formation
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "")).strip().lower()
    dynamic_gate_scene = is_dynamic_gate_density_scene_mode(scene_mode)
    safe_distance_m = float(env_config.inter_agent_safe_distance_m)
    max_slot_error_limit_m = max(
        float(formation_config.goal_slot_tolerance_m) * 2.0,
        3.2 if num_agents <= 3 else (6.5 if dynamic_gate_scene else 3.8),
    )
    guidance_error_limit_m = 0.65 if num_agents <= 3 else (2.25 if dynamic_gate_scene else 0.90)
    episode_min_pair_distance_m = _optional_float(episode_metrics.get("episode_min_pair_distance_m"))
    episode_max_slot_error_m = float(episode_metrics.get("episode_max_slot_error_m") or 0.0)
    episode_mean_guidance_tracking_error_m = _optional_float(
        episode_metrics.get("episode_mean_guidance_tracking_error_m")
    )
    goal_distance_improvement_m = float(episode_metrics.get("goal_distance_improvement_m") or 0.0)
    goal_progress_ratio = float(episode_metrics.get("goal_progress_ratio") or 0.0)

    audit_failures: list[str] = []
    if not bool(episode_metrics.get("height_contract_passed", True)):
        audit_failures.append("height_contract_failed")
    if bool(episode_metrics.get("height_escape_failure", False)):
        audit_failures.append("height_escape_failure")
    if bool(episode_metrics.get("side_bypass_failure", False)):
        audit_failures.append("side_bypass_failure")
    if bool(episode_metrics.get("corridor_miss_failure", False)):
        audit_failures.append("corridor_miss_failure")
    # formation_line_collapse_failure is not an audit failure for expert
    # retention: the protocol allows temporary formation deformation during
    # gate passage (task-first priority).  Episodes that reach the goal
    # with formation collapse are retained with a warning note instead.
    # Episodes that TERMINATE with formation_line_collapse_failure as
    # done_reason are still rejected by the hard-failure check below.
    if dynamic_gate_scene and not bool(episode_metrics.get("corridor_through_success", False)):
        audit_failures.append("corridor_through_failed")
    if audit_failures:
        return {
            "keep": False,
            "reason": "audit_failure_" + "+".join(audit_failures),
            "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
        }
    if done_reason in {"gate_post_collision", "agent_collision", "out_of_bounds", "formation_line_collapse_failure"}:
        return {
            "keep": False,
            "reason": f"hard_failure_{done_reason}",
            "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
        }
    if retain_failed_episodes:
        if done_reason == "goal_reached":
            retention_reason = "goal_reached"
        elif done_reason == "timeout" and goal_distance_improvement_m >= 0.50:
            retention_reason = "retain_non_collision_timeout"
        else:
            return {
                "keep": False,
                "reason": f"retain_failed_rejected_{done_reason or 'unknown'}",
                "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
            }
        return {
            "keep": True,
            "reason": retention_reason,
            "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
        }

    safe_pair_timeout = (
        done_reason == "timeout"
        and (episode_min_pair_distance_m is None or episode_min_pair_distance_m >= safe_distance_m * 0.95)
    )
    safe_timeout = (
        safe_pair_timeout
        and episode_max_slot_error_m <= max_slot_error_limit_m
        and (
            episode_mean_guidance_tracking_error_m is None
            or episode_mean_guidance_tracking_error_m <= guidance_error_limit_m
        )
    )

    if done_reason == "goal_reached":
        if dynamic_gate_scene:
            dynamic_goal_min_pair_m = safe_distance_m
            dynamic_goal_max_slot_error_m = min(max_slot_error_limit_m, 4.75 if num_agents > 3 else 1.20)
            dynamic_goal_max_guidance_error_m = min(guidance_error_limit_m, 2.20 if num_agents > 3 else 0.65)
            if episode_min_pair_distance_m is None:
                return {
                    "keep": False,
                    "reason": "goal_reached_missing_min_pair_distance",
                    "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
                }
            if episode_min_pair_distance_m < dynamic_goal_min_pair_m:
                return {
                    "keep": False,
                    "reason": "goal_reached_min_pair_distance_below_dynamic_bc_target",
                    "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
                }
            dynamic_goal_warnings: list[str] = []
            if episode_max_slot_error_m > dynamic_goal_max_slot_error_m:
                dynamic_goal_warnings.append("max_slot_error_above_dynamic_bc_target")
            if (
                episode_mean_guidance_tracking_error_m is not None
                and episode_mean_guidance_tracking_error_m > dynamic_goal_max_guidance_error_m
            ):
                dynamic_goal_warnings.append("guidance_error_above_dynamic_bc_target")
            if bool(episode_metrics.get("formation_line_collapse_failure", False)):
                dynamic_goal_warnings.append("formation_line_collapse")
            if dynamic_goal_warnings:
                return {
                    "keep": True,
                    "reason": "goal_reached_dynamic_task_first_" + "+".join(dynamic_goal_warnings),
                    "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
                }
        return {
            "keep": True,
            "reason": "goal_reached",
            "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
        }
    if dynamic_gate_scene and done_reason == "timeout":
        return {
            "keep": False,
            "reason": "strict_dynamic_gate_timeout_full_route_required",
            "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
        }
    if bool(getattr(env_config, "timeout_counts_as_success", False)) and safe_timeout:
        return {
            "keep": True,
            "reason": "safe_timeout_success",
            "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
        }
    if _supports_safe_progress_timeout_retention(experiment_config=experiment_config, num_agents=num_agents) and (
        safe_timeout or (dynamic_gate_scene and safe_pair_timeout)
    ):
        initial_goal_distance_m = float(episode_metrics.get("initial_goal_distance_m") or 0.0)
        progress_threshold_m, progress_ratio_threshold = _safe_progress_timeout_thresholds(
            experiment_config=experiment_config,
            num_agents=num_agents,
            initial_goal_distance_m=initial_goal_distance_m,
        )
        task_first_timeout = dynamic_gate_scene and safe_pair_timeout and (
            goal_distance_improvement_m >= max(progress_threshold_m, float(initial_goal_distance_m) * 0.72)
            or goal_progress_ratio >= max(progress_ratio_threshold, 0.72)
        )
        if safe_timeout and (
            goal_distance_improvement_m >= progress_threshold_m or goal_progress_ratio >= progress_ratio_threshold
        ):
            return {
                "keep": True,
                "reason": "safe_progress_timeout",
                "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
            }
        if task_first_timeout:
            return {
                "keep": True,
                "reason": "dynamic_gate_task_first_safe_timeout",
                "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
            }

    if done_reason in {"gate_post_collision", "agent_collision", "out_of_bounds", "formation_line_collapse_failure"}:
        rejection_reason = f"hard_failure_{done_reason}"
    elif done_reason != "timeout":
        rejection_reason = f"unsupported_done_reason_{done_reason}"
    elif episode_min_pair_distance_m is not None and episode_min_pair_distance_m < safe_distance_m * 0.95:
        rejection_reason = "timeout_min_pair_distance_below_target"
    elif episode_max_slot_error_m > max_slot_error_limit_m:
        rejection_reason = "timeout_max_slot_error_too_large"
    elif (
        episode_mean_guidance_tracking_error_m is not None
        and episode_mean_guidance_tracking_error_m > guidance_error_limit_m
    ):
        rejection_reason = "timeout_guidance_tracking_too_large"
    else:
        rejection_reason = "timeout_low_progress"
    return {
        "keep": False,
        "reason": rejection_reason,
        "score": _expert_episode_quality_score(done_reason=done_reason, episode_metrics=episode_metrics),
    }


def _select_balanced_expert_episode_indices(
    episode_records: list[dict[str, object]],
) -> tuple[list[int], dict[str, object]]:
    candidates_by_team_size: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for episode_idx, episode_record in enumerate(episode_records):
        if not bool(episode_record["keep_candidate"]):
            continue
        team_size = int(episode_record["summary"]["num_agents"])
        candidates_by_team_size[team_size].append((episode_idx, float(episode_record["retention_score"])))

    if not candidates_by_team_size:
        return [], {
            "policy": {
                "strategy": "quality_only",
                "target_floor_by_team_size": {},
                "soft_cap_by_team_size": {},
            },
            "rejected_selected_reason_by_index": {},
        }

    for team_size in list(candidates_by_team_size.keys()):
        candidates_by_team_size[team_size].sort(key=lambda item: (-item[1], item[0]))

    if len(candidates_by_team_size) == 1:
        selected_indices = [idx for entries in candidates_by_team_size.values() for idx, _ in entries]
        return selected_indices, {
            "policy": {
                "strategy": "quality_only",
                "target_floor_by_team_size": {
                    str(team_size): len(entries)
                    for team_size, entries in sorted(candidates_by_team_size.items())
                },
                "soft_cap_by_team_size": {
                    str(team_size): len(entries)
                    for team_size, entries in sorted(candidates_by_team_size.items())
                },
            },
            "rejected_selected_reason_by_index": {},
        }

    target_floor_by_team_size = {
        team_size: min(3, len(entries))
        for team_size, entries in candidates_by_team_size.items()
    }
    minimum_floor = max(min(target_floor_by_team_size.values()), 1)
    soft_cap_by_team_size = {
        team_size: max(target_floor_by_team_size[team_size], min(len(entries), minimum_floor * 2))
        for team_size, entries in candidates_by_team_size.items()
    }

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    selected_count_by_team_size: dict[int, int] = defaultdict(int)
    rejected_selected_reason_by_index: dict[int, str] = {}

    for team_size in sorted(candidates_by_team_size):
        floor_count = target_floor_by_team_size[team_size]
        for episode_idx, _score in candidates_by_team_size[team_size][:floor_count]:
            selected_indices.append(episode_idx)
            selected_set.add(episode_idx)
            selected_count_by_team_size[team_size] += 1

    remaining_candidates: list[tuple[float, int, int]] = []
    for team_size, entries in candidates_by_team_size.items():
        for episode_idx, score in entries[target_floor_by_team_size[team_size] :]:
            remaining_candidates.append((score, team_size, episode_idx))
    remaining_candidates.sort(key=lambda item: (-item[0], item[2]))

    for _score, team_size, episode_idx in remaining_candidates:
        if episode_idx in selected_set:
            continue
        if selected_count_by_team_size[team_size] >= soft_cap_by_team_size[team_size]:
            rejected_selected_reason_by_index[episode_idx] = "balance_soft_cap"
            continue
        selected_indices.append(episode_idx)
        selected_set.add(episode_idx)
        selected_count_by_team_size[team_size] += 1

    selected_indices.sort()
    return selected_indices, {
        "policy": {
            "strategy": "balanced_quality_filter",
            "target_floor_by_team_size": {
                str(team_size): int(count)
                for team_size, count in sorted(target_floor_by_team_size.items())
            },
            "soft_cap_by_team_size": {
                str(team_size): int(count)
                for team_size, count in sorted(soft_cap_by_team_size.items())
            },
        },
        "rejected_selected_reason_by_index": rejected_selected_reason_by_index,
    }


def _expert_episode_quality_score(
    *,
    done_reason: str,
    episode_metrics: dict[str, float | None],
) -> float:
    goal_distance_improvement_m = float(episode_metrics.get("goal_distance_improvement_m") or 0.0)
    goal_progress_ratio = float(episode_metrics.get("goal_progress_ratio") or 0.0)
    episode_min_pair_distance_m = _optional_float(episode_metrics.get("episode_min_pair_distance_m")) or 0.0
    episode_max_slot_error_m = float(episode_metrics.get("episode_max_slot_error_m") or 0.0)
    episode_mean_guidance_tracking_error_m = _optional_float(
        episode_metrics.get("episode_mean_guidance_tracking_error_m")
    ) or 0.0
    score = 0.0
    if done_reason == "goal_reached":
        score += 100.0
    elif done_reason == "timeout":
        score += 40.0
    score += 2.5 * goal_distance_improvement_m
    score += 35.0 * goal_progress_ratio
    score += 4.0 * episode_min_pair_distance_m
    score -= 4.5 * episode_max_slot_error_m
    score -= 12.0 * episode_mean_guidance_tracking_error_m
    return float(score)


def _supports_safe_progress_timeout_retention(
    *,
    experiment_config: MultiExperimentConfig,
    num_agents: int,
) -> bool:
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "")).strip().lower()
    if is_dynamic_gate_density_scene_mode(scene_mode):
        return False
    team_sizes = tuple(sorted(int(size) for size in experiment_config.size_invariance.bucket_team_sizes))
    env_config = experiment_config.environment
    corridor_length_m = abs(float(getattr(env_config, "goal_x_m", 0.0)) - float(getattr(env_config, "start_x_m", 0.0)))
    forward_speed_value = getattr(env_config, "max_command_forward_speed_mps", None)
    if forward_speed_value is None:
        forward_speed_value = getattr(env_config, "max_command_speed_mps", 0.0)
    forward_speed_mps = float(forward_speed_value or 0.0)
    max_team_size = max(team_sizes, default=0)
    stage_name = str(getattr(experiment_config.reasoning, "guidance_stage_name", "")).strip().lower()
    stage_specific_mixed_team_exception = (
        stage_name == "stage02_empty_235"
        and team_sizes == (2, 3, 5)
        and int(num_agents) == 3
    )
    stage_specific_seven_drone_bridge = stage_name in {
        "stage02a2_empty_7_d13",
        "stage02a3_empty_7_d23",
        "stage02a4_empty_7_full",
        "stage02a5_empty_2357_entry",
        "stage02a6_empty_2357_safe",
        "stage02_empty_2357",
    }
    stage_specific_seven_drone_mixed_exception = (
        stage_name in {"stage02a5_empty_2357_entry", "stage02a6_empty_2357_safe", "stage02_empty_2357"}
        and team_sizes == (2, 3, 5, 7)
    )
    stage_specific_eight_drone_bridge = stage_name in {
        "stage02c_empty_8_d13",
        "stage02d_empty_8_full",
        "stage02e_prep_hold_78",
        "stage02f_empty_78_micro",
        "stage02g_empty_78_d06_entry",
        "stage02e_empty_2358_entry",
        "stage02f_empty_2358_safe",
        "stage02e_empty_23578_entry",
        "stage02f_empty_23578_safe",
        "stage03_empty_23578_entry",
        "stage03a_empty_23578",
    }
    stage_specific_eight_drone_mixed_exception = (
        stage_name in {
            "stage02e_prep_hold_78",
            "stage02f_empty_78_micro",
            "stage02g_empty_78_d06_entry",
            "stage02e_empty_2358_entry",
            "stage02f_empty_2358_safe",
            "stage02e_empty_23578_entry",
            "stage02f_empty_23578_safe",
            "stage03_empty_23578_entry",
            "stage03a_empty_23578",
        }
        and team_sizes in {(7, 8), (2, 3, 5, 8), (2, 3, 5, 7, 8)}
    )
    stage_specific_nine_drone_bridge = stage_name in {
        "stage03b_empty_9_full",
        "stage03c_empty_235789_entry",
        "stage03d_empty_235789_safe",
    }
    stage_specific_nine_drone_mixed_exception = (
        stage_name in {"stage03c_empty_235789_entry", "stage03d_empty_235789_safe"}
        and team_sizes == (2, 3, 5, 7, 8, 9)
    )
    if stage_specific_nine_drone_bridge:
        max_supported_team_size = 9
    elif stage_specific_eight_drone_bridge:
        max_supported_team_size = 8
    elif stage_specific_seven_drone_bridge:
        max_supported_team_size = 7
    else:
        max_supported_team_size = 5
    max_supported_corridor_length_m = 48.0 if max_supported_team_size >= 7 else 32.0
    if is_gate_2d_scene_mode(scene_mode) and max_team_size <= 3:
        # The fixed 2/3-agent gate_2d presets use the full -46m -> 46m route.
        # A short BC smoke collection may time out despite being a clean,
        # high-progress expert segment, so retain it for imitation data only.
        max_supported_corridor_length_m = 96.0
    return (
        (is_exp3_empty_scene_mode(scene_mode) or is_gate_2d_scene_mode(scene_mode))
        and not bool(getattr(env_config, "preparation_hold_mode", False))
        and max_team_size <= max_supported_team_size
        and corridor_length_m <= max_supported_corridor_length_m
        and forward_speed_mps <= 3.50
        and (
            max_team_size <= 3
            or int(num_agents) >= max_team_size
            or stage_specific_mixed_team_exception
            or stage_specific_seven_drone_mixed_exception
            or stage_specific_eight_drone_mixed_exception
            or stage_specific_nine_drone_mixed_exception
        )
    )


def _safe_progress_timeout_thresholds(
    *,
    experiment_config: MultiExperimentConfig,
    num_agents: int,
    initial_goal_distance_m: float,
) -> tuple[float, float]:
    max_team_size = max((int(size) for size in experiment_config.size_invariance.bucket_team_sizes), default=0)
    if max_team_size <= 3:
        return max(5.0, min(8.0, float(initial_goal_distance_m) * 0.18)), 0.18
    if int(num_agents) >= max_team_size:
        return max(8.0, min(14.0, float(initial_goal_distance_m) * 0.40)), 0.45
    return max(6.0, min(10.0, float(initial_goal_distance_m) * 0.25)), 0.30


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not np.isfinite(resolved):
        return None
    return resolved


def run_bc_warmstart_then_finetune(
    *,
    experiment_config: MultiExperimentConfig | None = None,
    train_steps: int = 512,
    num_envs: int = 1,
    seed: int = 0,
    device: str | None = None,
    num_agents: int | None = None,
    max_sampled_agents: int | None = None,
    initial_actor_checkpoint: str | Path | None = None,
    expert_teacher_actor_checkpoint: str | Path | None = None,
    expert_episodes: int | None = None,
    expert_target_retained_episodes: int | None = None,
    expert_collection_workers: int = 1,
    max_steps_per_episode: int | None = None,
    retain_failed_episodes: bool = False,
    bc_epochs: int | None = None,
    bc_batch_size: int | None = None,
    bc_learning_rate: float | None = None,
    bc_weight_decay: float | None = None,
    bc_validation_split: float | None = None,
    bc_target_log_std: float | None = None,
    bc_log_std_penalty_scale: float | None = None,
    bc_output_dir: str | Path | None = None,
    bc_run_name: str | None = None,
    save_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_name: str | None = None,
    learning_starts: int | None = None,
    batch_size: int | None = None,
    updates_per_step: int | None = None,
    log_every: int = 64,
    checkpoint_interval_steps: int | None = None,
    selection_eval_episodes: int | None = None,
    periodic_eval_episodes: int = 0,
    periodic_eval_interval_steps: int | None = None,
    periodic_replay_mode: str = "skip",
    periodic_replay_interval_steps: int | None = None,
    periodic_replay_max_steps: int | None = None,
    periodic_replay_render_isaaclab: bool = False,
    periodic_replay_export_video: bool = False,
    periodic_replay_fps: int = 10,
    early_stop_eval_thresholds: dict[str, object] | None = None,
    early_stop_min_transitions: int = 0,
    early_stop_stable_window_min_length: int | None = None,
    early_stop_late_half_only: bool = False,
    early_stop_planned_total_transitions: int | None = None,
    failure_stop_eval_thresholds: dict[str, float | None] | None = None,
    failure_stop_min_transitions: int = 0,
    failure_stop_stable_window_min_length: int | None = None,
    live_preview_isaaclab: bool = False,
    live_preview_headless: bool = False,
    live_preview_interval_steps: int = 0,
    live_preview_follow_agent_index: int = 0,
) -> dict[str, object]:
    """Collect expert data, run BC warm start, then fine-tune with Graph-FlashSAC."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    collection_summary = collect_expert_demonstrations(
        experiment_config=selected_config,
        num_agents=num_agents,
        max_sampled_agents=max_sampled_agents,
        teacher_actor_checkpoint=expert_teacher_actor_checkpoint,
        teacher_device=device,
        expert_episodes=expert_episodes,
        target_retained_episodes=expert_target_retained_episodes,
        collection_workers=int(expert_collection_workers),
        seed=seed,
        max_steps_per_episode=max_steps_per_episode,
        retain_failed_episodes=retain_failed_episodes,
        output_dir=bc_output_dir,
        run_name=bc_run_name,
    )
    bc_summary = run_actor_behavior_cloning(
        dataset_path=collection_summary["dataset_path"],
        experiment_config=selected_config,
        device=device,
        seed=seed,
        initial_actor_checkpoint=initial_actor_checkpoint,
        epochs=bc_epochs,
        batch_size=bc_batch_size,
        learning_rate=bc_learning_rate,
        weight_decay=bc_weight_decay,
        validation_split=bc_validation_split,
        target_log_std=bc_target_log_std,
        log_std_penalty_scale=bc_log_std_penalty_scale,
        output_dir=bc_output_dir,
        run_name=bc_run_name,
    )
    training_summary = run_training(
        train_steps=train_steps,
        num_envs=num_envs,
        seed=seed,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        num_agents=num_agents,
        max_sampled_agents=max_sampled_agents,
        learning_starts=learning_starts,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        log_every=log_every,
        experiment_config=selected_config,
        warmstart_actor_checkpoint=bc_summary["actor_checkpoint_path"],
        checkpoint_interval_steps=checkpoint_interval_steps,
        selection_eval_episodes=selection_eval_episodes,
        periodic_eval_episodes=periodic_eval_episodes,
        periodic_eval_interval_steps=periodic_eval_interval_steps,
        periodic_replay_mode=periodic_replay_mode,
        periodic_replay_interval_steps=periodic_replay_interval_steps,
        periodic_replay_max_steps=periodic_replay_max_steps,
        periodic_replay_render_isaaclab=periodic_replay_render_isaaclab,
        periodic_replay_export_video=periodic_replay_export_video,
        periodic_replay_fps=periodic_replay_fps,
        early_stop_eval_thresholds=early_stop_eval_thresholds,
        early_stop_min_transitions=int(early_stop_min_transitions),
        early_stop_stable_window_min_length=early_stop_stable_window_min_length,
        early_stop_late_half_only=bool(early_stop_late_half_only),
        early_stop_planned_total_transitions=early_stop_planned_total_transitions,
        failure_stop_eval_thresholds=failure_stop_eval_thresholds,
        failure_stop_min_transitions=int(failure_stop_min_transitions),
        failure_stop_stable_window_min_length=failure_stop_stable_window_min_length,
        live_preview_isaaclab=live_preview_isaaclab,
        live_preview_headless=live_preview_headless,
        live_preview_interval_steps=live_preview_interval_steps,
        live_preview_follow_agent_index=live_preview_follow_agent_index,
    )
    combined_summary = {
        "experiment_id": selected_config.experiment_id,
        "initial_actor_checkpoint": (
            str(initial_actor_checkpoint) if initial_actor_checkpoint is not None else None
        ),
        "collection": collection_summary,
        "behavior_cloning": bc_summary,
        "fine_tuning": training_summary,
    }
    summary_dir = Path(training_summary.get("log_dir") or bc_summary["output_dir"])
    combined_summary_path = summary_dir / "bc_then_finetune_summary.json"
    combined_summary["summary_path"] = str(combined_summary_path)
    write_json(combined_summary_path, combined_summary)
    return combined_summary


def load_expert_dataset(path: str | Path) -> dict[str, object]:
    """Load one saved expert dataset."""

    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as payload:
        metadata_raw = payload["metadata_json"].item()
        observations = {
            key.removeprefix("obs__"): np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key.startswith("obs__")
        }
        actions = np.asarray(payload["actions"], dtype=np.float32)
    return {
        "dataset_path": str(dataset_path),
        "observations": observations,
        "actions": actions,
        "metadata": json.loads(metadata_raw),
    }


def _save_expert_dataset(
    path: Path,
    *,
    observations: dict[str, np.ndarray],
    actions: np.ndarray,
    metadata: dict[str, object],
) -> Path:
    payload = {
        "actions": np.asarray(actions, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    for name, array in observations.items():
        payload[f"obs__{name}"] = np.asarray(array, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


def _batch_observations(
    observations: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(values[indices], dtype=torch.float32, device=device)
        for name, values in observations.items()
    }


def _expand_dynamic_gate_risk_train_indices(
    *,
    train_indices: np.ndarray,
    observations: dict[str, np.ndarray],
    experiment_config: MultiExperimentConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "") or "")
    if "dynamic_gate_density" not in scene_mode:
        return train_indices, {"enabled": False, "reason": "non_dynamic_gate_scene"}
    node_features = observations.get("node_features")
    if node_features is None or node_features.ndim != 3 or node_features.shape[-1] < 15:
        return train_indices, {"enabled": False, "reason": "missing_node_features"}
    if train_indices.size == 0:
        return train_indices, {"enabled": False, "reason": "empty_train_indices"}

    features = np.asarray(node_features, dtype=np.float32)
    obstacle_mask = features[:, :, 3] > 0.5
    dynamic_velocity_norm = np.linalg.norm(features[:, :, 9:11], axis=-1)
    dynamic_obstacle_mask = obstacle_mask & (dynamic_velocity_norm > 1.0e-5)
    clearance_feature = np.where(dynamic_obstacle_mask, features[:, :, 14], np.inf)
    min_dynamic_clearance_feature = np.min(clearance_feature, axis=1)

    selected_clearance = min_dynamic_clearance_feature[train_indices]
    repeat_counts = np.ones((train_indices.size,), dtype=np.int64)
    repeat_counts[selected_clearance <= 0.32] = 2
    repeat_counts[selected_clearance <= 0.20] = 3
    repeat_counts[selected_clearance <= 0.12] = 5
    repeat_counts[selected_clearance <= 0.08] = 7
    expanded_indices = np.repeat(train_indices, repeat_counts)
    return expanded_indices.astype(np.int64, copy=False), {
        "enabled": True,
        "strategy": "dynamic_gate_low_clearance_oversampling",
        "original_train_sample_count": int(train_indices.size),
        "expanded_train_sample_count": int(expanded_indices.size),
        "near_0p08_count": int(np.sum(selected_clearance <= 0.08)),
        "near_0p12_count": int(np.sum(selected_clearance <= 0.12)),
        "near_0p20_count": int(np.sum(selected_clearance <= 0.20)),
        "near_0p32_count": int(np.sum(selected_clearance <= 0.32)),
        "min_dynamic_clearance_feature": (
            None
            if not np.isfinite(selected_clearance).any()
            else float(np.min(selected_clearance[np.isfinite(selected_clearance)]))
        ),
    }


def _evaluate_bc_epoch(
    *,
    agent: GraphMASACAgent,
    observations: dict[str, np.ndarray],
    actions: np.ndarray,
    indices: np.ndarray,
    target_log_std: float | None,
    log_std_penalty_scale: float,
) -> tuple[float, float, float]:
    if indices.size == 0:
        return (0.0, 0.0, 0.0)
    with torch.no_grad():
        batch_obs = _batch_observations(observations, indices, agent.device)
        batch_actions = torch.as_tensor(actions[indices], dtype=torch.float32, device=agent.device)
        _, metrics = agent.actor_behavior_clone_loss(
            batch_obs,
            batch_actions,
            target_log_std=target_log_std,
            log_std_penalty_scale=log_std_penalty_scale,
        )
    return (metrics["total_loss"], metrics["bc_loss"], metrics["log_std_penalty"])


def _sample_team_size(
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

