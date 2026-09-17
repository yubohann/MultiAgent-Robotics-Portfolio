"""Training helpers for the multi-agent 2D gate experiment."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from multi_gate.configs.experiment_config import MULTI_EXPERIMENT_CONFIG, MultiExperimentConfig
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from multi_gate.env.vector_multi_gate_env import VectorMultiGate2DEnv
from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
from shared.runtime.tensorboard import close_summary_writer, create_summary_writer, event_file_paths, log_scalar
from shared.runtime.training_controls import refresh_best_checkpoint_alias
from shared.runtime.vector_training_utils import (
    resolve_updates_per_collect,
    should_checkpoint_now,
)

from .checkpoint import (
    _build_training_signature,
    _candidate_checkpoint_path,
    _maybe_resume_training,
    _save_candidate_checkpoint,
)
from .early_stop import (
    _analyze_early_stop_stable_window,
    _analyze_failure_stop_window,
    _assess_eval_thresholds,
    _assess_failure_stop_thresholds,
)
from .evaluation import _select_multi_env_class
from .live_preview import _start_live_isaaclab_preview, _write_live_preview_snapshot
from .logging import (
    _log_multi_training_scalars,
    _log_periodic_eval_scalars,
    _run_periodic_multi_eval,
    _run_periodic_multi_replay,
)
from .metrics import _derive_failure_replay_metadata, _select_team_sizes
from .paths import _resolve_output_dirs, _resolve_review_interval, _run_label

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
    """Choose one action batch for training collection, with warm-started stages prefilling replay from
    the current actor.
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
                for env_idx, team_size in zip(done_indices.tolist(), reset_team_sizes, strict=False):
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