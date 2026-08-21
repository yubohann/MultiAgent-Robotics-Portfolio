"""Training helpers for the multi-agent 2D gate experiment."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import torch

from multi_gate.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG,
    MultiExperimentConfig,
    is_exp3_empty_scene_mode,
    is_exp3_gate_scene_mode,
    is_exp3_kinematic_3d_scene_mode,
    is_dynamic_gate_density_scene_mode,
)
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from multi_gate.env.vector_multi_gate_env import VectorMultiGate2DEnv
from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
from shared.runtime.artifacts import allocate_replay_artifacts, write_json
from shared.runtime.tensorboard import (
    close_summary_writer,
    create_summary_writer,
    event_file_paths,
    log_scalar,
    log_scalars,
)
from shared.runtime.training_controls import (
    build_checkpoint_selection_details,
    refresh_best_checkpoint_alias,
)
from shared.runtime.vector_training_utils import (
    resolve_updates_per_collect,
    should_checkpoint_now,
)


MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv


def _select_training_action(
    *,
    agent: GraphMASACAgent,
    env: VectorMultiGate2DEnv,
    obs: dict[str, np.ndarray],
    transitions_collected: int,
    learning_starts: int,
    use_policy_prefill: bool,
) -> np.ndarray:
    """Choose one action batch for training collection.

    Warm-started stages should prefill replay with the current actor rather than
    random actions, otherwise delayed optimization starts on mostly off-policy
    noise and can immediately destabilize the actor.
    """

    if int(transitions_collected) < int(learning_starts):
        if bool(use_policy_prefill):
            return agent.act_batch(obs, deterministic=True)
        return env.sample_random_action()
    return agent.act_batch(obs, deterministic=False)


def run_training(
    *,
    train_steps: int = 512,
    num_envs: int = 1,
    seed: int = 0,
    device: str | None = None,
    save_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_name: str | None = None,
    num_agents: int | None = None,
    max_sampled_agents: int | None = None,
    learning_starts: int | None = None,
    batch_size: int | None = None,
    updates_per_step: int | None = None,
    log_every: int = 64,
    experiment_config: MultiExperimentConfig | None = None,
    warmstart_actor_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_mode: MultiResumeMode | str | None = None,
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
    early_stop_eval_thresholds: dict[str, float | None] | None = None,
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
    """Run a vectorized Graph-FlashSAC training loop with checkpoint selection and safe resume."""

    if warmstart_actor_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("Use either warmstart_actor_checkpoint or resume_checkpoint, not both.")

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    resolved_num_envs = max(int(num_envs), 1)
    env_cls = _select_multi_env_class(selected_config)
    template_env = env_cls(
        multi_config=selected_config,
        env_config=selected_config.environment,
        observation_config=selected_config.observation,
        formation_config=selected_config.formation,
        planner_config=selected_config.planner,
    )
    env = VectorMultiGate2DEnv(
        num_envs=resolved_num_envs,
        multi_config=selected_config,
        env_config=selected_config.environment,
        observation_config=selected_config.observation,
        formation_config=selected_config.formation,
        planner_config=selected_config.planner,
        env_cls=env_cls,
    )
    active_num_agents = _select_team_sizes(
        seed=seed,
        num_envs=resolved_num_envs,
        num_agents=num_agents,
        max_sampled_agents=max_sampled_agents,
        experiment_config=selected_config,
    )
    obs, _ = env.reset(seed=seed, num_agents=active_num_agents)
    agent = GraphMASACAgent.from_defaults(
        obs_shapes=template_env.observation_shapes,
        device=device,
        seed=seed,
        obs_config=selected_config.observation,
        masac_config=selected_config.algorithm,
        max_agents_soft=selected_config.max_agents_soft,
    )
    agent.configure_replay_sampling(
        enabled=bool(selected_config.failure_replay.enabled),
        failure_replay_ratio=float(selected_config.failure_replay.failure_replay_ratio),
    )
    checkpoint_policy = selected_config.checkpoint_policy
    algorithm = selected_config.algorithm
    learning_starts = int(learning_starts if learning_starts is not None else algorithm.learning_starts)
    batch_size = int(batch_size if batch_size is not None else algorithm.batch_size)
    updates_per_step = int(updates_per_step if updates_per_step is not None else algorithm.updates_per_step)
    checkpoint_name = checkpoint_name or algorithm.checkpoint_name
    checkpoint_interval_transitions = max(
        int(
            checkpoint_interval_steps
            if checkpoint_interval_steps is not None
            else checkpoint_policy.checkpoint_interval_steps
        ),
        0,
    )
    selection_eval_episodes = max(
        int(
            selection_eval_episodes
            if selection_eval_episodes is not None
            else checkpoint_policy.selection_eval_episodes
        ),
        0,
    )
    periodic_eval_episodes = max(int(periodic_eval_episodes), 0)
    resolved_periodic_replay_mode = str(periodic_replay_mode or "skip").strip().lower()
    if resolved_periodic_replay_mode not in {"skip", "heuristic", "checkpoint"}:
        raise ValueError(f"Unsupported periodic replay mode: {periodic_replay_mode}")
    resolved_log_dir, resolved_checkpoint_dir = _resolve_output_dirs(
        save_dir=save_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )
    training_signature = _build_training_signature(env=template_env, experiment_config=selected_config)

    warmstart_metadata: dict[str, object] | None = None
    if warmstart_actor_checkpoint is not None:
        warmstart_metadata = agent.load_actor_checkpoint(warmstart_actor_checkpoint)

    resume_context = _maybe_resume_training(
        agent=agent,
        env=template_env,
        experiment_config=selected_config,
        checkpoint_path=resume_checkpoint,
        resume_mode=resume_mode,
        seed=seed,
    )
    use_policy_prefill = warmstart_metadata is not None or resume_context is not None
    if use_policy_prefill and float(getattr(selected_config.algorithm, "behavior_anchor_loss_scale", 0.0) or 0.0) > 0.0:
        agent.capture_behavior_reference()
    updates_per_collect = resolve_updates_per_collect(resolved_num_envs, updates_per_step)
    periodic_eval_interval_transitions = _resolve_review_interval(
        requested_interval=periodic_eval_interval_steps,
        fallback_interval=checkpoint_interval_transitions,
        enabled=periodic_eval_episodes > 0,
    )
    periodic_replay_interval_transitions = _resolve_review_interval(
        requested_interval=periodic_replay_interval_steps,
        fallback_interval=checkpoint_interval_transitions,
        enabled=resolved_periodic_replay_mode != "skip",
    )
    tensorboard_writer, tensorboard_dir = (
        create_summary_writer(resolved_log_dir) if resolved_log_dir is not None else (None, None)
    )
    live_preview_snapshot_path: Path | None = None
    live_preview_process: subprocess.Popen | None = None
    next_live_preview_transition = None
    if bool(live_preview_isaaclab) and resolved_log_dir is not None:
        live_preview_snapshot_path = resolved_log_dir / "live_isaaclab_snapshot.json"
        _write_live_preview_snapshot(
            snapshot_path=live_preview_snapshot_path,
            env=env.envs[0],
            experiment_config=selected_config,
            transitions_collected=0,
        )
        live_preview_process = _start_live_isaaclab_preview(
            snapshot_path=live_preview_snapshot_path,
            experiment_config=selected_config,
            follow_agent_index=live_preview_follow_agent_index,
            headless=live_preview_headless,
        )
        resolved_live_interval = max(int(live_preview_interval_steps), 0)
        next_live_preview_transition = resolved_live_interval if resolved_live_interval > 0 else resolved_num_envs

    try:
        completed_episodes = 0
        collector_iterations_completed = 0
        last_update_metrics: dict[str, float] = {}
        done_reason_counts: dict[str, int] = {}
        team_sizes_seen = set(int(size) for size in active_num_agents)
        checkpoint_paths: list[str] = []
        checkpoint_selection_records: list[dict[str, object]] = []
        periodic_evaluations: list[dict[str, object]] = []
        periodic_replays: list[dict[str, object]] = []
        selected_checkpoint_path: str | None = None
        best_alias_path: str | None = None
        final_checkpoint_path: str | None = None
        latest_alias_path: str | None = None
        best_score = float("-inf")
        transitions_collected = 0
        resolved_early_stop_thresholds = (
            None if not early_stop_eval_thresholds else dict(early_stop_eval_thresholds)
        )
        resolved_failure_stop_thresholds = (
            None if not failure_stop_eval_thresholds else dict(failure_stop_eval_thresholds)
        )
        resolved_early_stop_min_transitions = max(int(early_stop_min_transitions), 0)
        resolved_failure_stop_min_transitions = max(int(failure_stop_min_transitions), 0)
        resolved_early_stop_late_half_only = bool(early_stop_late_half_only)
        resolved_early_stop_stable_window_min_length = (
            None
            if early_stop_stable_window_min_length is None
            else max(int(early_stop_stable_window_min_length), 1)
        )
        resolved_failure_stop_stable_window_min_length = max(
            1,
            int(
                failure_stop_stable_window_min_length
                if failure_stop_stable_window_min_length is not None
                else 2
            ),
        )
        resolved_early_stop_planned_total_transitions = max(
            int(early_stop_planned_total_transitions)
            if early_stop_planned_total_transitions is not None
            else int(train_steps) * resolved_num_envs,
            0,
        )
        early_stop_triggered = False
        early_stop_reason: str | None = None
        early_stop_assessment: dict[str, object] | None = None
        early_stop_checkpoint_path: str | None = None
        early_stop_transition: int | None = None
        early_stop_window: dict[str, object] | None = None
        early_stop_window_analysis: dict[str, object] | None = None
        failure_stop_triggered = False
        failure_stop_reason: str | None = None
        failure_stop_assessment: dict[str, object] | None = None
        failure_stop_transition: int | None = None
        failure_stop_window_analysis: dict[str, object] | None = None
        next_checkpoint_transition = (
            checkpoint_interval_transitions
            if resolved_checkpoint_dir is not None and checkpoint_interval_transitions > 0
            else None
        )
        next_periodic_eval_transition = (
            periodic_eval_interval_transitions
            if resolved_checkpoint_dir is not None and periodic_eval_interval_transitions is not None
            else None
        )
        next_periodic_replay_transition = (
            periodic_replay_interval_transitions
            if resolved_checkpoint_dir is not None and periodic_replay_interval_transitions is not None
            else None
        )

        for collector_iteration in range(1, int(train_steps) + 1):
            collector_iterations_completed = int(collector_iteration)
            action = _select_training_action(
                agent=agent,
                env=env,
                obs=obs,
                transitions_collected=transitions_collected,
                learning_starts=learning_starts,
                use_policy_prefill=use_policy_prefill,
            )

            next_obs, reward, terminated, truncated, infos = env.step(action)
            done = np.logical_or(terminated, truncated)
            failure_tags = np.zeros((resolved_num_envs,), dtype=np.float32)
            safety_costs = np.zeros((resolved_num_envs,), dtype=np.float32)
            failure_reasons = np.asarray([""] * resolved_num_envs, dtype=object)
            for env_idx, info in enumerate(infos):
                failure_metadata = _derive_failure_replay_metadata(
                    info=info,
                    terminated=bool(terminated[env_idx]),
                    truncated=bool(truncated[env_idx]),
                    experiment_config=selected_config,
                )
                failure_tags[env_idx] = float(bool(failure_metadata["failure_tag"]))
                safety_costs[env_idx] = float(failure_metadata["safety_cost"])
                failure_reasons[env_idx] = str(failure_metadata["failure_reason"])
            agent.replay_buffer.add_batch(
                obs,
                action,
                reward,
                next_obs,
                done,
                failure_tag=failure_tags,
                safety_cost=safety_costs,
                failure_reason=failure_reasons,
            )
            transitions_collected += resolved_num_envs
            obs = next_obs

            if (
                live_preview_snapshot_path is not None
                and next_live_preview_transition is not None
                and transitions_collected >= next_live_preview_transition
            ):
                _write_live_preview_snapshot(
                    snapshot_path=live_preview_snapshot_path,
                    env=env.envs[0],
                    experiment_config=selected_config,
                    transitions_collected=transitions_collected,
                )
                while next_live_preview_transition is not None and transitions_collected >= next_live_preview_transition:
                    next_live_preview_transition += max(int(live_preview_interval_steps), resolved_num_envs)

            if len(agent.replay_buffer) >= max(batch_size, learning_starts):
                for _ in range(updates_per_collect):
                    batch = agent.replay_buffer.sample(batch_size, agent.device)
                    last_update_metrics = agent.update(batch)

            if bool(done.any()):
                done_indices = np.flatnonzero(done)
                for env_idx in done_indices.tolist():
                    info = infos[int(env_idx)]
                    reason = str(info.get("done_reason") or "unknown")
                    done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                reset_team_sizes = _select_team_sizes(
                    seed=seed + completed_episodes + 1,
                    num_envs=int(done_indices.size),
                    num_agents=num_agents,
                    max_sampled_agents=max_sampled_agents,
                    experiment_config=selected_config,
                )
                reset_result = env.reset_done(
                    done,
                    seed=seed + completed_episodes + 1,
                    num_agents=reset_team_sizes,
                )
                completed_episodes += int(done_indices.size)
                team_sizes_seen.update(int(size) for size in reset_team_sizes)
                obs = env.replace_done_observations(obs, reset_result)
                for env_idx, team_size in zip(done_indices.tolist(), reset_team_sizes):
                    active_num_agents[int(env_idx)] = int(team_size)

            checkpoint_due = (
                resolved_checkpoint_dir is not None
                and collector_iteration < int(train_steps)
                and should_checkpoint_now(
                    transitions_collected=transitions_collected,
                    next_checkpoint_transition=next_checkpoint_transition,
                )
            )
            periodic_eval_due = (
                resolved_checkpoint_dir is not None
                and collector_iteration < int(train_steps)
                and should_checkpoint_now(
                    transitions_collected=transitions_collected,
                    next_checkpoint_transition=next_periodic_eval_transition,
                )
            )
            periodic_replay_due = (
                resolved_checkpoint_dir is not None
                and collector_iteration < int(train_steps)
                and should_checkpoint_now(
                    transitions_collected=transitions_collected,
                    next_checkpoint_transition=next_periodic_replay_transition,
                )
            )
            stop_training_now = False

            if resolved_checkpoint_dir is not None and (checkpoint_due or periodic_eval_due or periodic_replay_due):
                candidate_path = _candidate_checkpoint_path(
                    checkpoint_dir=resolved_checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                    step=transitions_collected,
                )
                record, best_score, selected_checkpoint_path, best_alias_path = _save_candidate_checkpoint(
                    agent=agent,
                    checkpoint_path=candidate_path,
                    checkpoint_dir=resolved_checkpoint_dir,
                    alias_name=checkpoint_policy.best_alias_name,
                    selection_eval_episodes=selection_eval_episodes,
                    seed=seed,
                    device=device,
                    step=transitions_collected,
                    kind="interval",
                    training_signature=training_signature,
                    resume_context=resume_context,
                    current_best_score=best_score,
                    current_selected_checkpoint_path=selected_checkpoint_path,
                    current_best_alias_path=best_alias_path,
                    num_agents=num_agents,
                    experiment_config=selected_config,
                )
                checkpoint_paths.append(str(candidate_path))
                checkpoint_selection_records.append(record)
                while next_checkpoint_transition is not None and transitions_collected >= next_checkpoint_transition:
                    next_checkpoint_transition += checkpoint_interval_transitions
                while next_periodic_eval_transition is not None and transitions_collected >= next_periodic_eval_transition:
                    next_periodic_eval_transition += int(periodic_eval_interval_transitions or 0)
                while next_periodic_replay_transition is not None and transitions_collected >= next_periodic_replay_transition:
                    next_periodic_replay_transition += int(periodic_replay_interval_transitions or 0)

                if periodic_eval_due:
                    eval_summary = _run_periodic_multi_eval(
                        checkpoint_path=candidate_path,
                        transitions_collected=transitions_collected,
                        seed=seed,
                        device=device,
                        episodes=periodic_eval_episodes,
                        num_agents=num_agents,
                        experiment_config=selected_config,
                        run_label=_run_label(resolved_log_dir, resolved_checkpoint_dir),
                        selection_record=record if int(record["selection_eval_episodes"]) == periodic_eval_episodes else None,
                    )
                    periodic_evaluations.append(eval_summary)
                    _log_periodic_eval_scalars(
                        tensorboard_writer,
                        eval_summary=eval_summary,
                        transitions_collected=transitions_collected,
                    )

                if periodic_replay_due and resolved_periodic_replay_mode != "skip":
                    replay_summary = _run_periodic_multi_replay(
                        checkpoint_path=candidate_path,
                        transitions_collected=transitions_collected,
                        seed=seed,
                        device=device,
                        replay_mode=resolved_periodic_replay_mode,
                        max_steps=periodic_replay_max_steps,
                        num_agents=num_agents,
                        experiment_config=selected_config,
                        run_label=_run_label(resolved_log_dir, resolved_checkpoint_dir),
                        render_isaaclab=bool(periodic_replay_render_isaaclab),
                        export_video=bool(periodic_replay_export_video),
                        fps=int(periodic_replay_fps),
                    )
                    periodic_replays.append(replay_summary)
                    log_scalar(
                        tensorboard_writer,
                        "periodic_replay/success",
                        replay_summary.get("success"),
                        transitions_collected,
                    )
                    log_scalar(
                        tensorboard_writer,
                        "periodic_replay/steps",
                        replay_summary.get("steps"),
                        transitions_collected,
                    )

                if (
                    resolved_early_stop_thresholds is not None
                    and int(record.get("selection_eval_episodes") or 0) > 0
                    and transitions_collected >= resolved_early_stop_min_transitions
                ):
                    early_stop_assessment = _assess_eval_thresholds(
                        eval_summary=dict(record.get("selection_eval_summary") or {}),
                        thresholds=resolved_early_stop_thresholds,
                    )
                    if bool(early_stop_assessment["passed"]):
                        stable_window_required = bool(resolved_early_stop_late_half_only) or (
                            resolved_early_stop_stable_window_min_length is not None
                            and int(resolved_early_stop_stable_window_min_length) > 1
                        )
                        if stable_window_required:
                            early_stop_window_analysis = _analyze_early_stop_stable_window(
                                checkpoint_selection_records=checkpoint_selection_records,
                                thresholds=resolved_early_stop_thresholds,
                                planned_total_transitions=resolved_early_stop_planned_total_transitions,
                                min_window_length=(
                                    1
                                    if resolved_early_stop_stable_window_min_length is None
                                    else int(resolved_early_stop_stable_window_min_length)
                                ),
                                late_half_only=resolved_early_stop_late_half_only,
                            )
                            early_stop_window = early_stop_window_analysis.get("window")
                            if (
                                bool(early_stop_window_analysis["passed"])
                                and str(early_stop_window_analysis["checkpoint_path"]) == str(candidate_path)
                            ):
                                early_stop_triggered = True
                                early_stop_reason = "selection_eval_stable_window_met"
                                early_stop_checkpoint_path = str(candidate_path)
                                early_stop_transition = int(transitions_collected)
                                stop_training_now = True
                        else:
                            early_stop_triggered = True
                            early_stop_reason = "selection_eval_thresholds_met"
                            early_stop_checkpoint_path = str(candidate_path)
                            early_stop_transition = int(transitions_collected)
                            stop_training_now = True
                if (
                    resolved_failure_stop_thresholds is not None
                    and int(record.get("selection_eval_episodes") or 0) > 0
                    and transitions_collected >= resolved_failure_stop_min_transitions
                    and not stop_training_now
                ):
                    failure_stop_assessment = _assess_failure_stop_thresholds(
                        eval_summary=dict(record.get("selection_eval_summary") or {}),
                        thresholds=resolved_failure_stop_thresholds,
                    )
                    failure_stop_window_analysis = _analyze_failure_stop_window(
                        checkpoint_selection_records=checkpoint_selection_records,
                        thresholds=resolved_failure_stop_thresholds,
                        min_window_length=resolved_failure_stop_stable_window_min_length,
                        min_transition=resolved_failure_stop_min_transitions,
                    )
                    if bool(failure_stop_window_analysis["passed"]):
                        failure_stop_triggered = True
                        failure_stop_reason = "consecutive_selection_eval_failures"
                        failure_stop_transition = int(transitions_collected)
                        stop_training_now = True

            _log_multi_training_scalars(
                tensorboard_writer,
                transitions_collected=transitions_collected,
                completed_episodes=completed_episodes,
                buffer_size=len(agent.replay_buffer),
                num_envs=resolved_num_envs,
                last_update_metrics=last_update_metrics,
                done_reason_counts=done_reason_counts,
                active_num_agents=active_num_agents,
                replay_diagnostics=agent.replay_buffer.stats(),
            )

            if log_every > 0 and collector_iteration % log_every == 0:
                print(
                    f"[train_multi] collect={collector_iteration} transitions={transitions_collected} "
                    f"buffer={len(agent.replay_buffer)} episodes={completed_episodes} "
                    f"num_envs={resolved_num_envs} active_teams={sorted(set(int(size) for size in active_num_agents))}"
                )
            if stop_training_now:
                reason = early_stop_reason if early_stop_triggered else failure_stop_reason
                print(
                    f"[train_multi] early_stop transitions={transitions_collected} "
                    f"reason={reason} checkpoint={early_stop_checkpoint_path} failed_checks=[]"
                )
                log_scalar(tensorboard_writer, "train/early_stop_triggered", 1.0 if early_stop_triggered else 0.0, transitions_collected)
                log_scalar(tensorboard_writer, "train/failure_stop_triggered", 1.0 if failure_stop_triggered else 0.0, transitions_collected)
                break

        if live_preview_snapshot_path is not None:
            _write_live_preview_snapshot(
                snapshot_path=live_preview_snapshot_path,
                env=env.envs[0],
                experiment_config=selected_config,
                transitions_collected=transitions_collected,
            )

        summary: dict[str, object] = {
            "experiment_id": selected_config.experiment_id,
            "train_steps": int(train_steps),
            "collector_iterations": int(collector_iterations_completed),
            "num_envs": resolved_num_envs,
            "transitions_collected": int(transitions_collected),
            "completed_episodes": completed_episodes,
            "buffer_size": len(agent.replay_buffer),
            "replay_diagnostics": agent.replay_buffer.stats(),
            "device": str(agent.device),
            "done_reason_counts": done_reason_counts,
            "last_update_metrics": last_update_metrics,
            "team_sizes_seen": sorted(team_sizes_seen),
            "active_team_sizes_final": [int(size) for size in active_num_agents],
            "updates_per_collect": int(updates_per_collect),
            "updates_per_step": int(updates_per_step),
            "learning_starts_transitions": int(learning_starts),
            "checkpoint_interval_transitions": int(checkpoint_interval_transitions),
            "periodic_eval_episodes": int(periodic_eval_episodes),
            "periodic_eval_interval_transitions": periodic_eval_interval_transitions,
            "periodic_evaluations": periodic_evaluations,
            "periodic_replay_mode": resolved_periodic_replay_mode,
            "periodic_replay_interval_transitions": periodic_replay_interval_transitions,
            "periodic_replay_max_steps": (
                None if periodic_replay_max_steps is None else int(periodic_replay_max_steps)
            ),
            "periodic_replay_render_isaaclab": bool(periodic_replay_render_isaaclab),
            "periodic_replay_export_video": bool(periodic_replay_export_video),
            "periodic_replay_fps": int(periodic_replay_fps),
            "periodic_replays": periodic_replays,
            "early_stop_eval_thresholds": resolved_early_stop_thresholds,
            "early_stop_min_transitions": int(resolved_early_stop_min_transitions),
            "early_stop_stable_window_min_length": resolved_early_stop_stable_window_min_length,
            "early_stop_late_half_only": bool(resolved_early_stop_late_half_only),
            "early_stop_planned_total_transitions": int(resolved_early_stop_planned_total_transitions),
            "early_stop_triggered": bool(early_stop_triggered),
            "early_stop_reason": early_stop_reason,
            "early_stop_assessment": early_stop_assessment,
            "early_stop_checkpoint_path": early_stop_checkpoint_path,
            "early_stop_transition": early_stop_transition,
            "early_stop_window": early_stop_window,
            "early_stop_window_analysis": early_stop_window_analysis,
            "failure_stop_eval_thresholds": resolved_failure_stop_thresholds,
            "failure_stop_min_transitions": int(resolved_failure_stop_min_transitions),
            "failure_stop_stable_window_min_length": int(resolved_failure_stop_stable_window_min_length),
            "failure_stop_triggered": bool(failure_stop_triggered),
            "failure_stop_reason": failure_stop_reason,
            "failure_stop_assessment": failure_stop_assessment,
            "failure_stop_transition": failure_stop_transition,
            "failure_stop_window_analysis": failure_stop_window_analysis,
            "min_agents": selected_config.min_agents,
            "default_agents": selected_config.default_agents,
            "max_agents_soft": selected_config.max_agents_soft,
            "paper_track": selected_config.paper_track,
            "paper_variant": selected_config.paper_variant,
            "notes": selected_config.notes,
            "scene": asdict(selected_config.scene),
            "reasoning": asdict(selected_config.reasoning),
            "algorithm": asdict(selected_config.algorithm),
            "imitation": asdict(selected_config.imitation),
            "failure_replay": asdict(selected_config.failure_replay),
            "size_invariance": asdict(selected_config.size_invariance),
            "dagger": asdict(selected_config.dagger),
            "benchmark": asdict(selected_config.benchmark),
            "checkpoint_policy": asdict(selected_config.checkpoint_policy),
            "evaluation_gate": asdict(selected_config.evaluation_gate),
            "resume_policy": asdict(selected_config.resume_policy),
            "environment": asdict(selected_config.environment),
            "formation": asdict(selected_config.formation),
            "planner": asdict(selected_config.planner),
            "warmstart_actor_checkpoint": (
                str(warmstart_actor_checkpoint) if warmstart_actor_checkpoint is not None else None
            ),
            "warmstart_actor_metadata": warmstart_metadata,
            "resume_context": resume_context,
            "policy_prefill_used": bool(use_policy_prefill),
            "training_signature": training_signature,
        }

        if live_preview_snapshot_path is not None:
            summary["live_preview_snapshot_path"] = str(live_preview_snapshot_path)

        if resolved_log_dir is not None and resolved_checkpoint_dir is not None:
            resolved_log_dir.mkdir(parents=True, exist_ok=True)
            resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            final_checkpoint = resolved_checkpoint_dir / checkpoint_name
            record, best_score, selected_checkpoint_path, best_alias_path = _save_candidate_checkpoint(
                agent=agent,
                checkpoint_path=final_checkpoint,
                checkpoint_dir=resolved_checkpoint_dir,
                alias_name=checkpoint_policy.best_alias_name,
                selection_eval_episodes=selection_eval_episodes,
                seed=seed,
                device=device,
                step=int(transitions_collected),
                kind="final",
                training_signature=training_signature,
                resume_context=resume_context,
                current_best_score=best_score,
                current_selected_checkpoint_path=selected_checkpoint_path,
                current_best_alias_path=best_alias_path,
                num_agents=num_agents,
                experiment_config=selected_config,
            )
            final_checkpoint_path = str(final_checkpoint)
            if selected_checkpoint_path is None:
                selected_checkpoint_path = final_checkpoint_path
            if best_alias_path is None:
                best_alias_path = str(
                    refresh_best_checkpoint_alias(
                        selected_checkpoint_path,
                        checkpoint_dir=resolved_checkpoint_dir,
                        alias_name=checkpoint_policy.best_alias_name,
                    )
                )
            latest_alias_path = str(
                refresh_best_checkpoint_alias(
                    final_checkpoint,
                    checkpoint_dir=resolved_checkpoint_dir,
                    alias_name="latest_agent.pt",
                )
            )
            checkpoint_paths.append(final_checkpoint_path)
            checkpoint_selection_records.append(record)
            if periodic_eval_episodes > 0:
                final_eval_summary = _run_periodic_multi_eval(
                    checkpoint_path=final_checkpoint,
                    transitions_collected=transitions_collected,
                    seed=seed,
                    device=device,
                    episodes=periodic_eval_episodes,
                    num_agents=num_agents,
                    experiment_config=selected_config,
                    run_label=_run_label(resolved_log_dir, resolved_checkpoint_dir),
                    selection_record=record if int(record["selection_eval_episodes"]) == periodic_eval_episodes else None,
                )
                periodic_evaluations.append(final_eval_summary)
                _log_periodic_eval_scalars(
                    tensorboard_writer,
                    eval_summary=final_eval_summary,
                    transitions_collected=transitions_collected,
                )
            if resolved_periodic_replay_mode != "skip":
                final_replay_summary = _run_periodic_multi_replay(
                    checkpoint_path=final_checkpoint,
                    transitions_collected=transitions_collected,
                    seed=seed,
                    device=device,
                    replay_mode=resolved_periodic_replay_mode,
                    max_steps=periodic_replay_max_steps,
                    num_agents=num_agents,
                    experiment_config=selected_config,
                    run_label=_run_label(resolved_log_dir, resolved_checkpoint_dir),
                    render_isaaclab=bool(periodic_replay_render_isaaclab),
                    export_video=bool(periodic_replay_export_video),
                    fps=int(periodic_replay_fps),
                )
                periodic_replays.append(final_replay_summary)
                log_scalar(
                    tensorboard_writer,
                    "periodic_replay/success",
                    final_replay_summary.get("success"),
                    transitions_collected,
                )
                log_scalar(
                    tensorboard_writer,
                    "periodic_replay/steps",
                    final_replay_summary.get("steps"),
                    transitions_collected,
                )
            for selection_record in checkpoint_selection_records:
                selection_record["selected"] = (
                    str(selection_record["checkpoint_path"]) == str(selected_checkpoint_path)
                )

            summary["checkpoint_paths"] = checkpoint_paths
            summary["checkpoint_selection_records"] = checkpoint_selection_records
            summary["selected_checkpoint_path"] = selected_checkpoint_path
            summary["best_checkpoint_path"] = selected_checkpoint_path
            summary["best_alias_path"] = best_alias_path
            summary["final_checkpoint_path"] = final_checkpoint_path
            summary["latest_alias_path"] = latest_alias_path
            summary["checkpoint_path"] = best_alias_path or selected_checkpoint_path or latest_alias_path or final_checkpoint_path
            summary["best_alias_reselection"] = None
            summary["best_alias_reselection_report_path"] = None
            summary["log_dir"] = str(resolved_log_dir)
            summary["checkpoint_dir"] = str(resolved_checkpoint_dir)
            if tensorboard_dir is not None:
                summary["tensorboard_dir"] = str(tensorboard_dir)
                summary["tensorboard_event_files"] = event_file_paths(tensorboard_dir)

            summary_path = resolved_log_dir / "training_summary.json"
            summary["summary_path"] = str(summary_path)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            summary["checkpoint_paths"] = checkpoint_paths
            summary["checkpoint_selection_records"] = checkpoint_selection_records
            summary["selected_checkpoint_path"] = selected_checkpoint_path
            summary["best_checkpoint_path"] = selected_checkpoint_path
            summary["best_alias_path"] = best_alias_path
            summary["final_checkpoint_path"] = final_checkpoint_path
            summary["latest_alias_path"] = latest_alias_path
            summary["checkpoint_path"] = latest_alias_path or final_checkpoint_path or selected_checkpoint_path
            summary["best_alias_reselection"] = None
            summary["best_alias_reselection_report_path"] = None

        return summary
    finally:
        if live_preview_process is not None:
            live_preview_process.terminate()
        close_summary_writer(tensorboard_writer)


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


def _start_live_isaaclab_preview(
    *,
    snapshot_path: Path,
    experiment_config: MultiExperimentConfig,
    follow_agent_index: int,
    headless: bool,
) -> subprocess.Popen | None:
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "")).strip().lower()
    if not is_exp3_kinematic_3d_scene_mode(scene_mode):
        return None
    script_path = Path(__file__).resolve().parent / "scripts" / "live_preview_multi_isaaclab.py"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--snapshot",
        str(snapshot_path),
        "--config-name",
        _resolve_live_preview_config_name(experiment_config),
        "--scene-mode",
        scene_mode,
        "--render-real-gate",
        "1" if bool(experiment_config.scene.render_real_gate) else "0",
        "--render-real-drone-shell",
        "1" if bool(experiment_config.scene.render_real_drone_shell) else "0",
        "--follow-agent-index",
        str(max(int(follow_agent_index), 0)),
    ]
    if bool(headless):
        command.append("--headless")
    return subprocess.Popen(command)


def _resolve_live_preview_config_name(experiment_config: MultiExperimentConfig) -> str:
    paper_variant = str(getattr(experiment_config, "paper_variant", "")).strip().lower()
    if paper_variant in {"e3_baseline", "e3_main", "e3_guidance"}:
        return paper_variant
    return "variable"


def _write_live_preview_snapshot(
    *,
    snapshot_path: Path,
    env: MultiEnvType,
    experiment_config: MultiExperimentConfig,
    transitions_collected: int,
) -> None:
    positions_xy = env.active_positions_xy()
    velocities_xy = env.active_velocities_xy()
    snapshot = env.snapshot()
    payload = {
        "experiment_id": experiment_config.experiment_id,
        "paper_track": experiment_config.paper_track,
        "paper_variant": experiment_config.paper_variant,
        "scene_mode": experiment_config.scene.scene_mode,
        "render_real_gate": bool(experiment_config.scene.render_real_gate),
        "render_real_drone_shell": bool(experiment_config.scene.render_real_drone_shell),
        "transitions_collected": int(transitions_collected),
        "num_agents": int(snapshot.num_agents),
        "max_agents": int(experiment_config.max_agents_soft),
        "fixed_height_m": float(env.env_config.fixed_height_m),
        "positions_xy": positions_xy.tolist(),
        "velocities_xy": velocities_xy.tolist(),
        "yaws_rad": [float(state.yaw_rad) for state in env._states],
        "desired_slots_xy": env.desired_slots_xy().tolist(),
        "virtual_center_xy": list(snapshot.virtual_center_xy),
        "lookahead_heading_xy": list(env.current_heading_xy()),
        "path_waypoints": [list(point) for point in env.path_waypoints()],
        "path_index": int(snapshot.path_index),
        "start_xy": list(env._start_center_xy),
        "goal_xy": list(env.path_waypoints()[-1]),
        "world_x_bounds_m": list(env.env_config.world_x_bounds_m),
        "world_y_bounds_m": list(env.env_config.world_y_bounds_m),
        "route_plan_guidance": env._route_plan_guidance_summary(snapshot.virtual_center_xy),
        "route_guidance": env._route_guidance_summary(snapshot.virtual_center_xy),
        "route_guidance_meta": dict(getattr(env, "_route_guidance_meta", {})),
    }
    write_json(snapshot_path, payload)


def _resolve_output_dirs(
    *,
    save_dir: str | Path | None,
    log_dir: str | Path | None,
    checkpoint_dir: str | Path | None,
) -> tuple[Path | None, Path | None]:
    if log_dir is not None or checkpoint_dir is not None:
        resolved_log_dir = Path(log_dir) if log_dir is not None else None
        resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if resolved_log_dir is None or resolved_checkpoint_dir is None:
            raise ValueError("log_dir and checkpoint_dir must be provided together")
        return resolved_log_dir, resolved_checkpoint_dir
    if save_dir is not None:
        base_dir = Path(save_dir)
        return base_dir / "logs", base_dir / "checkpoints"
    return None, None


def _resolve_review_interval(
    *,
    requested_interval: int | None,
    fallback_interval: int,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    if requested_interval is not None:
        resolved = max(int(requested_interval), 0)
        return resolved if resolved > 0 else None
    fallback = max(int(fallback_interval), 0)
    return fallback if fallback > 0 else None


def _run_label(log_dir: Path | None, checkpoint_dir: Path | None) -> str:
    if log_dir is not None and log_dir.name != "logs":
        return log_dir.name
    if checkpoint_dir is not None and checkpoint_dir.name != "checkpoints":
        return checkpoint_dir.name
    if log_dir is not None:
        return log_dir.parent.name or "multi_run"
    if checkpoint_dir is not None:
        return checkpoint_dir.parent.name or "multi_run"
    return "multi_run"


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


def _load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, object]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    metadata = payload.get("metadata")
    resolved_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    actor_state = payload.get("actor") if isinstance(payload, dict) else None
    if isinstance(actor_state, dict):
        node_embed_weight = actor_state.get("encoder.node_embed.0.weight")
        mean_weight = actor_state.get("mean_layer.weight")
        actor_signature: dict[str, object] = {}
        if hasattr(node_embed_weight, "shape") and len(node_embed_weight.shape) == 2:
            actor_signature["node_feature_dim"] = int(node_embed_weight.shape[1])
            actor_signature["graph_hidden_dim"] = int(node_embed_weight.shape[0])
        if hasattr(mean_weight, "shape") and len(mean_weight.shape) == 2:
            actor_signature["action_dim"] = int(mean_weight.shape[0])
            actor_signature["actor_hidden_dim"] = int(mean_weight.shape[1])
        if actor_signature:
            actor_signature["actor_only_checkpoint"] = not any(
                key in payload for key in ("critic_1", "critic_2", "target_critic_1", "target_critic_2")
            )
            resolved_metadata["_actor_state_signature"] = actor_signature
    return resolved_metadata


def _build_training_signature(
    *,
    env: MultiEnvType,
    experiment_config: MultiExperimentConfig,
) -> dict[str, object]:
    algorithm = experiment_config.algorithm
    return {
        "experiment_id": experiment_config.experiment_id,
        "paper_track": experiment_config.paper_track,
        "paper_variant": experiment_config.paper_variant,
        "control_mode": experiment_config.control_mode,
        "planner_mode": experiment_config.planner_mode,
        "scene_mode": experiment_config.scene.scene_mode,
        "render_backend": experiment_config.scene.render_backend,
        "render_real_gate": bool(experiment_config.scene.render_real_gate),
        "render_real_drone_shell": bool(experiment_config.scene.render_real_drone_shell),
        "disable_motors": bool(experiment_config.scene.disable_motors),
        "global_planner_enabled": bool(experiment_config.reasoning.global_planner_enabled),
        "route_guidance_enabled": bool(experiment_config.reasoning.route_guidance_enabled),
        "guidance_shadow_mode": bool(getattr(experiment_config.reasoning, "guidance_shadow_mode", False)),
        "guidance_provider": str(getattr(experiment_config.reasoning, "guidance_provider", "none")),
        "guidance_model_name": str(getattr(experiment_config.reasoning, "guidance_model_name", "")),
        "guidance_prompt_version": str(getattr(experiment_config.reasoning, "guidance_prompt_version", "")),
        "guidance_stage_name": str(getattr(experiment_config.reasoning, "guidance_stage_name", "")),
        "min_agents": int(experiment_config.min_agents),
        "default_agents": int(experiment_config.default_agents),
        "max_agents_soft": int(experiment_config.max_agents_soft),
        "env_class": env.__class__.__name__,
        "observation_shapes": {name: list(shape) for name, shape in env.observation_shapes.items()},
        "algorithm_name": str(getattr(algorithm, "algorithm_name", "graph_flashsac")),
        "action_dim": int(algorithm.action_dim),
        "log_std_min": float(algorithm.log_std_min),
        "log_std_max": float(algorithm.log_std_max),
        "actor_head_mode": str(getattr(algorithm, "actor_head_mode", "single")),
        "enable_safety_critic": bool(getattr(algorithm, "enable_safety_critic", False)),
    }


def _maybe_resume_training(
    *,
    agent: GraphMASACAgent,
    env: MultiEnvType,
    experiment_config: MultiExperimentConfig,
    checkpoint_path: str | Path | None,
    resume_mode: MultiResumeMode | str | None,
    seed: int,
) -> dict[str, object] | None:
    if checkpoint_path is None:
        return None

    resolved_path = Path(checkpoint_path)
    metadata = agent.load_checkpoint(resolved_path)
    findings = _multi_resume_compatibility_findings(
        metadata=metadata,
        env=env,
        experiment_config=experiment_config,
    )
    incompatible = [name for name, result in findings.items() if not bool(result["compatible"])]
    if incompatible:
        raise ValueError(
            "Multi-agent resume checkpoint is incompatible with the current experiment: "
            f"{resolved_path} | failed checks: {', '.join(incompatible)}"
        )

    policy = experiment_config.resume_policy
    resolved_mode = str(resume_mode or policy.default_mode).strip().lower()
    if resolved_mode == "reset_train_state":
        applied_resets = agent.reset_training_state(
            reset_optimizer_state=bool(policy.reset_optimizer_state),
            reset_entropy_state=bool(policy.reset_entropy_state),
            reset_replay_buffer=True,
            reset_update_step=True,
            seed=seed,
        )
    elif resolved_mode == "keep_optimizer_state":
        applied_resets = agent.reset_training_state(
            reset_optimizer_state=False,
            reset_entropy_state=False,
            reset_replay_buffer=True,
            reset_update_step=False,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported multi-agent resume mode: {resolved_mode}")

    return {
        "resume_checkpoint_path": str(resolved_path),
        "resume_mode": resolved_mode,
        "compatibility_findings": findings,
        "applied_resets": applied_resets,
        "limitations": [
            "Replay buffer snapshots are not persisted in aerogate_graph; resume always starts with an empty replay buffer.",
        ],
        "checkpoint_metadata": metadata,
    }


def _multi_resume_compatibility_findings(
    *,
    metadata: dict[str, object],
    env: MultiEnvType,
    experiment_config: MultiExperimentConfig,
) -> dict[str, dict[str, object]]:
    policy = experiment_config.resume_policy
    signature = dict(metadata.get("training_signature") or {})
    expected_shapes = {name: list(shape) for name, shape in env.observation_shapes.items()}
    actor_state_signature = dict(metadata.get("_actor_state_signature") or {})
    expected_node_feature_dim = int(expected_shapes.get("node_features", [0, 0])[1])
    actor_state_observation_compatible = (
        bool(actor_state_signature.get("actor_only_checkpoint", False))
        and int(actor_state_signature.get("node_feature_dim", -1)) == expected_node_feature_dim
        and int(actor_state_signature.get("action_dim", -1)) == int(experiment_config.algorithm.action_dim)
    )
    actual_experiment_id = str(
        signature.get("experiment_id")
        or metadata.get("experiment_id")
        or metadata.get("summary", {}).get("experiment_id")
        or metadata.get("experiment_config", {}).get("experiment_id")
        or ""
    ).strip()
    actual_shapes = signature.get("observation_shapes") or metadata.get("observation_shapes")
    actual_action_dim = signature.get("action_dim")
    actual_max_agents_soft = signature.get("max_agents_soft") or metadata.get("experiment_config", {}).get("max_agents_soft")
    algorithm = experiment_config.algorithm
    expected_actor_head_mode = str(getattr(algorithm, "actor_head_mode", "single"))
    actual_actor_head_mode = signature.get("actor_head_mode")
    actor_head_compatible = (
        actual_actor_head_mode == expected_actor_head_mode
        or (actual_actor_head_mode is None and expected_actor_head_mode == "single")
    )
    expected_scene_mode = str(getattr(experiment_config.scene, "scene_mode", "")).strip().lower()
    actual_scene_mode = str(signature.get("scene_mode") or "").strip().lower()
    scene_mode_compatible = (
        signature.get("scene_mode") is None
        or actual_scene_mode == expected_scene_mode
        or (
            is_exp3_empty_scene_mode(actual_scene_mode)
            and is_exp3_empty_scene_mode(expected_scene_mode)
        )
        or (
            is_exp3_gate_scene_mode(actual_scene_mode)
            and is_exp3_gate_scene_mode(expected_scene_mode)
        )
        or (
            is_exp3_empty_scene_mode(actual_scene_mode)
            and is_dynamic_gate_density_scene_mode(expected_scene_mode)
        )
    )
    actual_control_mode = signature.get("control_mode")
    expected_control_mode = experiment_config.control_mode
    legacy_control_modes = {
        "graph_masac",
        "graph_masac_dynamic_gate_density_2d",
        "graph_masac_kinematic_3d",
    }
    control_mode_compatible = actual_control_mode in {None, expected_control_mode} or (
        str(expected_control_mode).startswith("graph_flashsac")
        and actual_control_mode in legacy_control_modes
    ) or (
        is_dynamic_gate_density_scene_mode(expected_scene_mode)
        and actual_control_mode in legacy_control_modes
    )

    findings = {
        "experiment_id": {
            "required": bool(policy.strict_experiment_id),
            "expected": experiment_config.experiment_id,
            "actual": actual_experiment_id,
            "compatible": (
                True
                if not policy.strict_experiment_id
                else bool(actual_experiment_id) and actual_experiment_id == experiment_config.experiment_id
            ),
        },
        "observation_shapes": {
            "required": bool(policy.strict_observation_shapes),
            "expected": expected_shapes,
            "actual": actual_shapes if actual_shapes is not None else actor_state_signature,
            "compatible": (
                True
                if not policy.strict_observation_shapes
                else actual_shapes == expected_shapes
                or (actual_shapes is None and actor_state_observation_compatible)
            ),
        },
        "action_dim": {
            "required": True,
            "expected": int(algorithm.action_dim),
            "actual": actual_action_dim,
            "compatible": actual_action_dim in {None, int(algorithm.action_dim)},
        },
        "max_agents_soft": {
            "required": True,
            "expected": int(experiment_config.max_agents_soft),
            "actual": actual_max_agents_soft,
            "compatible": actual_max_agents_soft in {None, int(experiment_config.max_agents_soft)},
        },
        "control_mode": {
            "required": True,
            "expected": expected_control_mode,
            "actual": actual_control_mode,
            "compatible": control_mode_compatible,
        },
        "scene_mode": {
            "required": True,
            "expected": experiment_config.scene.scene_mode,
            "actual": signature.get("scene_mode"),
            "compatible": scene_mode_compatible,
        },
        "actor_head_mode": {
            "required": True,
            "expected": expected_actor_head_mode,
            "actual": actual_actor_head_mode,
            "compatible": actor_head_compatible,
        },
    }
    return findings


def _candidate_checkpoint_path(
    *,
    checkpoint_dir: Path,
    checkpoint_name: str,
    step: int,
) -> Path:
    base = Path(checkpoint_name)
    return checkpoint_dir / f"{base.stem}_step_{int(step):06d}{base.suffix}"


def _checkpoint_metadata(
    *,
    step: int,
    kind: str,
    seed: int,
    training_signature: dict[str, object],
    resume_context: dict[str, object] | None,
    experiment_config: MultiExperimentConfig,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_config.experiment_id,
        "algorithm_name": str(getattr(experiment_config.algorithm, "algorithm_name", "graph_flashsac")),
        "seed": int(seed),
        "checkpoint_step": int(step),
        "checkpoint_kind": kind,
        "training_signature": training_signature,
        "resume_context": resume_context,
        "experiment_config": {
            "experiment_id": experiment_config.experiment_id,
            "min_agents": int(experiment_config.min_agents),
            "default_agents": int(experiment_config.default_agents),
            "max_agents_soft": int(experiment_config.max_agents_soft),
            "notes": experiment_config.notes,
        },
        "failure_replay": asdict(experiment_config.failure_replay),
        "size_invariance": asdict(experiment_config.size_invariance),
    }


def _save_candidate_checkpoint(
    *,
    agent: GraphMASACAgent,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    alias_name: str,
    selection_eval_episodes: int,
    seed: int,
    device: str | None,
    step: int,
    kind: str,
    training_signature: dict[str, object],
    resume_context: dict[str, object] | None,
    current_best_score: float,
    current_selected_checkpoint_path: str | None,
    current_best_alias_path: str | None,
    num_agents: int | None,
    experiment_config: MultiExperimentConfig,
) -> tuple[dict[str, object], float, str | None, str | None]:
    agent.save_checkpoint(
        checkpoint_path,
        metadata=_checkpoint_metadata(
            step=step,
            kind=kind,
            seed=seed,
            training_signature=training_signature,
            resume_context=resume_context,
            experiment_config=experiment_config,
        ),
    )
    if selection_eval_episodes > 0:
        if (
            num_agents is None
            and bool(experiment_config.size_invariance.enabled)
            and int(experiment_config.size_invariance.bucket_eval_episodes) > 0
        ):
            eval_summary = evaluate_size_buckets(
                checkpoint_path=checkpoint_path,
                episodes_per_bucket=max(
                    int(selection_eval_episodes),
                    int(experiment_config.size_invariance.bucket_eval_episodes),
                ),
                seed=seed,
                device=device,
                team_sizes=experiment_config.size_invariance.bucket_team_sizes,
                experiment_config=experiment_config,
            )
        else:
            eval_summary = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=selection_eval_episodes,
                seed=seed,
                device=device,
                num_agents=num_agents,
                experiment_config=experiment_config,
            )
        selection_details = build_checkpoint_selection_details(eval_summary)
        score = float(selection_details["score"])
    else:
        eval_summary = {
            "episodes": 0,
            "num_agents": experiment_config.default_agents if num_agents is None else int(num_agents),
            "success_rate": 0.0,
            "mean_episode_reward": 0.0,
            "done_reason_counts": {},
            "episode_summaries": [],
        }
        selection_details = {
            "task_type": "multi",
            "score": float(step),
            "metrics": {"episodes": 0},
        }
        score = float(step)

    is_selected = current_selected_checkpoint_path is None or score > current_best_score
    best_score = current_best_score
    selected_checkpoint_path = current_selected_checkpoint_path
    best_alias_path = current_best_alias_path
    if is_selected:
        best_score = score
        selected_checkpoint_path = str(checkpoint_path)
        best_alias_path = str(
            refresh_best_checkpoint_alias(
                checkpoint_path,
                checkpoint_dir=checkpoint_dir,
                alias_name=alias_name,
            )
        )

    record = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": kind,
        "step": int(step),
        "selection_eval_episodes": int(selection_eval_episodes),
        "selection_score": float(score),
        "selection_details": selection_details,
        "selection_eval_summary": eval_summary,
        "selected": is_selected,
    }
    return record, best_score, selected_checkpoint_path, best_alias_path


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

