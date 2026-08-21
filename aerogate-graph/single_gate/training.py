"""Training helpers for the single-agent 2D gate experiment."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal

import numpy as np
import torch

from single_gate.configs.experiment_config import SINGLE_EXPERIMENT_CONFIG
from single_gate.env.vector_single_gate_env import VectorSingleGate2DEnv
from single_gate.env.single_gate_env import SingleGate2DEnv
from single_gate.graph_rl.graph_flashsac import GraphFlashSACAgent
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
    reselect_best_checkpoint_alias,
)
from shared.runtime.vector_training_utils import (
    resolve_updates_per_collect,
    should_checkpoint_now,
)


SingleResumeMode = Literal["reset_train_state", "keep_optimizer_state"]


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
    learning_starts: int | None = None,
    batch_size: int | None = None,
    updates_per_step: int | None = None,
    log_every: int = 64,
    resume_checkpoint: str | Path | None = None,
    resume_mode: SingleResumeMode | str | None = None,
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
) -> dict[str, object]:
    """Run a vectorized single-agent Graph-FlashSAC training loop."""

    resolved_num_envs = max(int(num_envs), 1)
    env_config = SINGLE_EXPERIMENT_CONFIG.environment
    observation_config = SINGLE_EXPERIMENT_CONFIG.observation
    template_env = SingleGate2DEnv(
        env_config=env_config,
        observation_config=observation_config,
    )
    env = VectorSingleGate2DEnv(
        num_envs=resolved_num_envs,
        env_config=env_config,
        observation_config=observation_config,
    )
    obs, _ = env.reset(seed=seed)
    agent = GraphFlashSACAgent.from_defaults(
        obs_shapes=template_env.observation_shapes,
        device=device,
        seed=seed,
    )
    algorithm = SINGLE_EXPERIMENT_CONFIG.algorithm
    checkpoint_policy = SINGLE_EXPERIMENT_CONFIG.checkpoint_policy
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
    training_signature = _build_training_signature(template_env)
    resume_context = _maybe_resume_training(
        agent=agent,
        env=template_env,
        checkpoint_path=resume_checkpoint,
        resume_mode=resume_mode,
        seed=seed,
    )
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

    completed_episodes = 0
    last_update_metrics: dict[str, float] = {}
    done_reason_counts: dict[str, int] = {}
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
        if transitions_collected < learning_starts:
            action = env.sample_random_action()
        else:
            action = agent.act_batch(obs, deterministic=False)

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done = np.logical_or(terminated, truncated)
        agent.replay_buffer.add_batch(obs, action, reward, next_obs, done)
        transitions_collected += resolved_num_envs
        obs = next_obs

        if len(agent.replay_buffer) >= max(batch_size, learning_starts):
            for _ in range(updates_per_collect):
                batch = agent.replay_buffer.sample(batch_size, agent.device)
                last_update_metrics = agent.update(batch)

        if bool(done.any()):
            done_count = int(done.sum())
            for info in [infos[idx] for idx in np.flatnonzero(done)]:
                reason = str(info.get("done_reason") or "unknown")
                done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
            reset_result = env.reset_done(
                done,
                seed=seed + completed_episodes + 1,
            )
            completed_episodes += done_count
            obs = env.replace_done_observations(obs, reset_result)

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
                eval_summary = _run_periodic_single_eval(
                    checkpoint_path=candidate_path,
                    transitions_collected=transitions_collected,
                    seed=seed,
                    device=device,
                    episodes=periodic_eval_episodes,
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
                replay_summary = _run_periodic_single_replay(
                    checkpoint_path=candidate_path,
                    transitions_collected=transitions_collected,
                    seed=seed,
                    device=device,
                    replay_mode=resolved_periodic_replay_mode,
                    max_steps=periodic_replay_max_steps,
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

        _log_single_training_scalars(
            tensorboard_writer,
            transitions_collected=transitions_collected,
            completed_episodes=completed_episodes,
            buffer_size=len(agent.replay_buffer),
            num_envs=resolved_num_envs,
            last_update_metrics=last_update_metrics,
            done_reason_counts=done_reason_counts,
        )

        if log_every > 0 and collector_iteration % log_every == 0:
            print(
                f"[train_single] collect={collector_iteration} transitions={transitions_collected} "
                f"buffer={len(agent.replay_buffer)} episodes={completed_episodes} num_envs={resolved_num_envs}"
            )

    summary: dict[str, object] = {
        "experiment_id": SINGLE_EXPERIMENT_CONFIG.experiment_id,
        "train_steps": int(train_steps),
        "collector_iterations": int(train_steps),
        "num_envs": resolved_num_envs,
        "transitions_collected": int(transitions_collected),
        "completed_episodes": completed_episodes,
        "buffer_size": len(agent.replay_buffer),
        "device": str(agent.device),
        "done_reason_counts": done_reason_counts,
        "last_update_metrics": last_update_metrics,
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
        "algorithm": asdict(SINGLE_EXPERIMENT_CONFIG.algorithm),
        "checkpoint_policy": asdict(SINGLE_EXPERIMENT_CONFIG.checkpoint_policy),
        "evaluation_gate": asdict(SINGLE_EXPERIMENT_CONFIG.evaluation_gate),
        "resume_policy": asdict(SINGLE_EXPERIMENT_CONFIG.resume_policy),
        "environment": asdict(SINGLE_EXPERIMENT_CONFIG.environment),
        "resume_context": resume_context,
        "training_signature": training_signature,
    }

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
        )
        final_checkpoint_path = str(final_checkpoint)
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
            final_eval_summary = _run_periodic_single_eval(
                checkpoint_path=final_checkpoint,
                transitions_collected=transitions_collected,
                seed=seed,
                device=device,
                episodes=periodic_eval_episodes,
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
            final_replay_summary = _run_periodic_single_replay(
                checkpoint_path=final_checkpoint,
                transitions_collected=transitions_collected,
                seed=seed,
                device=device,
                replay_mode=resolved_periodic_replay_mode,
                max_steps=periodic_replay_max_steps,
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
        best_alias_reselection = reselect_best_checkpoint_alias(
            checkpoint_records=checkpoint_selection_records,
            checkpoint_dir=resolved_checkpoint_dir,
            alias_name=checkpoint_policy.best_alias_name,
            shortlist_size=min(max(len(checkpoint_selection_records), 1), 8),
            final_eval_episodes=max(
                int(SINGLE_EXPERIMENT_CONFIG.evaluation_gate.eval_episodes),
                int(selection_eval_episodes) * 2,
                16,
            ),
            evaluate_summary_fn=lambda checkpoint_path, episodes: evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=episodes,
                seed=seed,
                device=device,
            ),
            report_path=resolved_log_dir / "best_alias_reselection_report.json",
        )
        if best_alias_reselection is not None:
            selected_checkpoint_path = str(best_alias_reselection["selected_checkpoint_path"])
            best_alias_path = str(best_alias_reselection["best_alias_path"])
        for selection_record in checkpoint_selection_records:
            selection_record["selected"] = str(selection_record["checkpoint_path"]) == str(selected_checkpoint_path)

        summary["checkpoint_paths"] = checkpoint_paths
        summary["checkpoint_selection_records"] = checkpoint_selection_records
        summary["selected_checkpoint_path"] = selected_checkpoint_path
        summary["best_checkpoint_path"] = selected_checkpoint_path
        summary["best_alias_path"] = best_alias_path
        summary["final_checkpoint_path"] = final_checkpoint_path
        summary["latest_alias_path"] = latest_alias_path
        summary["checkpoint_path"] = best_alias_path or selected_checkpoint_path or final_checkpoint_path
        summary["best_alias_reselection"] = best_alias_reselection
        summary["best_alias_reselection_report_path"] = (
            None if best_alias_reselection is None else best_alias_reselection.get("report_path")
        )
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
        summary["checkpoint_path"] = selected_checkpoint_path
        summary["best_alias_reselection"] = None
        summary["best_alias_reselection_report_path"] = None

    close_summary_writer(tensorboard_writer)
    return summary


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int = 5,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, object]:
    """Evaluate a saved Graph-FlashSAC checkpoint deterministically."""

    env = SingleGate2DEnv()
    agent = GraphFlashSACAgent.from_defaults(
        obs_shapes=env.observation_shapes,
        device=device,
        seed=seed,
    )
    validate_single_checkpoint_compatibility(
        checkpoint_path=checkpoint_path,
        env=env,
    )
    metadata = agent.load_checkpoint(checkpoint_path)

    total_rewards: list[float] = []
    done_reason_counts: dict[str, int] = {}
    episode_summaries: list[dict[str, object]] = []
    successes = 0

    for episode_idx in range(int(episodes)):
        obs, _ = env.reset(seed=seed + episode_idx)
        episode_reward = 0.0
        step_count = 0
        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            step_count += 1
            if terminated or truncated:
                total_rewards.append(episode_reward)
                reason = str(info.get("done_reason") or "unknown")
                done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                if reason == "goal_reached":
                    successes += 1
                episode_summaries.append(
                    _serialize_single_episode_summary(
                        episode_idx=episode_idx,
                        step_count=step_count,
                        episode_reward=episode_reward,
                        info=info,
                    )
                )
                break

    resolved_episodes = max(int(episodes), 1)
    step_values = [float(summary["steps"]) for summary in episode_summaries]
    goal_distance_values = [float(summary["goal_distance_m"]) for summary in episode_summaries]
    clearance_values = [float(summary["signed_clearance_m"]) for summary in episode_summaries]
    success_step_values = [
        float(summary["steps"])
        for summary in episode_summaries
        if str(summary.get("done_reason") or "") == "goal_reached"
    ]
    success_clearance_values = [
        float(summary["signed_clearance_m"])
        for summary in episode_summaries
        if str(summary.get("done_reason") or "") == "goal_reached"
    ]
    collision_rate = float(done_reason_counts.get("collision", 0)) / resolved_episodes
    out_of_bounds_rate = float(done_reason_counts.get("out_of_bounds", 0)) / resolved_episodes
    timeout_rate = float(done_reason_counts.get("timeout", 0)) / resolved_episodes
    return {
        "episodes": int(episodes),
        "success_rate": successes / resolved_episodes,
        "collision_rate": collision_rate,
        "out_of_bounds_rate": out_of_bounds_rate,
        "timeout_rate": timeout_rate,
        "collision_out_of_bounds_rate": collision_rate + out_of_bounds_rate,
        "mean_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_steps": float(np.mean(step_values)) if step_values else 0.0,
        "mean_success_steps": float(np.mean(success_step_values)) if success_step_values else 0.0,
        "mean_goal_distance_m": float(np.mean(goal_distance_values)) if goal_distance_values else 0.0,
        "mean_signed_clearance_m": float(np.mean(clearance_values)) if clearance_values else 0.0,
        "min_signed_clearance_m": float(np.min(clearance_values)) if clearance_values else 0.0,
        "success_mean_clearance_m": (
            float(np.mean(success_clearance_values)) if success_clearance_values else 0.0
        ),
        "done_reason_counts": done_reason_counts,
        "episode_summaries": episode_summaries,
        "metadata": metadata,
    }


def validate_single_checkpoint_compatibility(
    *,
    checkpoint_path: str | Path,
    env: SingleGate2DEnv,
) -> dict[str, object]:
    """Validate one inference/resume checkpoint against the active single-agent setup."""

    metadata = _load_checkpoint_metadata(checkpoint_path)
    findings = _single_resume_compatibility_findings(metadata=metadata, env=env)
    incompatible = [name for name, result in findings.items() if not bool(result["compatible"])]
    if incompatible:
        raise ValueError(
            "Single-agent checkpoint is incompatible with the current experiment: "
            f"{Path(checkpoint_path)} | failed checks: {', '.join(incompatible)}"
        )
    return metadata


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
        return log_dir.parent.name or "single_run"
    if checkpoint_dir is not None:
        return checkpoint_dir.parent.name or "single_run"
    return "single_run"


def _log_single_training_scalars(
    writer,
    *,
    transitions_collected: int,
    completed_episodes: int,
    buffer_size: int,
    num_envs: int,
    last_update_metrics: dict[str, float],
    done_reason_counts: dict[str, int],
) -> None:
    log_scalar(writer, "train/transitions_collected", transitions_collected, transitions_collected)
    log_scalar(writer, "train/completed_episodes", completed_episodes, transitions_collected)
    log_scalar(writer, "train/buffer_size", buffer_size, transitions_collected)
    log_scalar(writer, "train/num_envs", num_envs, transitions_collected)
    log_scalars(writer, "updates", last_update_metrics, transitions_collected)
    for reason, count in done_reason_counts.items():
        log_scalar(writer, f"done_reason_counts/{reason}", count, transitions_collected)


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
        "periodic_eval/collision_out_of_bounds_rate",
        eval_summary.get("collision_out_of_bounds_rate"),
        transitions_collected,
    )
    log_scalar(writer, "periodic_eval/mean_steps", eval_summary.get("mean_steps"), transitions_collected)
    log_scalar(
        writer,
        "periodic_eval/mean_signed_clearance_m",
        eval_summary.get("mean_signed_clearance_m"),
        transitions_collected,
    )
    for reason, count in dict(eval_summary.get("done_reason_counts") or {}).items():
        log_scalar(writer, f"periodic_eval/done_reason_counts/{reason}", count, transitions_collected)


def _run_periodic_single_eval(
    *,
    checkpoint_path: Path,
    transitions_collected: int,
    seed: int,
    device: str | None,
    episodes: int,
    run_label: str,
    selection_record: dict[str, object] | None,
) -> dict[str, object]:
    if selection_record is not None:
        eval_summary = dict(selection_record.get("selection_eval_summary") or {})
    else:
        eval_summary = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=episodes,
            seed=seed,
            device=device,
        )
    output_dir = allocate_replay_artifacts(
        "single",
        run_name=f"{run_label}_step_{int(transitions_collected):06d}_periodic_eval",
    ).output_dir
    eval_summary["checkpoint_path"] = str(checkpoint_path)
    eval_summary["step"] = int(transitions_collected)
    eval_summary["summary_path"] = str(output_dir / "eval_summary.json")
    write_json(output_dir / "eval_summary.json", eval_summary)
    return eval_summary


def _run_periodic_single_replay(
    *,
    checkpoint_path: Path,
    transitions_collected: int,
    seed: int,
    device: str | None,
    replay_mode: str,
    max_steps: int | None,
    run_label: str,
    render_isaaclab: bool,
    export_video: bool,
    fps: int,
) -> dict[str, object]:
    from single_gate.replay import run_single_replay

    output_dir = allocate_replay_artifacts(
        "single",
        run_name=f"{run_label}_step_{int(transitions_collected):06d}_periodic_replay",
    ).output_dir
    replay_summary = run_single_replay(
        mode=replay_mode,
        checkpoint_path=checkpoint_path if replay_mode == "checkpoint" else None,
        seed=seed,
        max_steps=max_steps,
        output_dir=output_dir,
        device=device,
    )
    replay_summary["step"] = int(transitions_collected)
    replay_summary["checkpoint_path"] = str(checkpoint_path)
    if render_isaaclab:
        replay_summary["isaaclab_render"] = _render_single_replay_isaaclab(
            replay_summary=replay_summary,
            output_dir=output_dir / "isaaclab",
            export_video=export_video,
            fps=fps,
        )
    return replay_summary


def _render_single_replay_isaaclab(
    *,
    replay_summary: dict[str, object],
    output_dir: Path,
    export_video: bool,
    fps: int,
) -> dict[str, object]:
    script_path = Path(__file__).resolve().parent / "scripts" / "replay_single_isaaclab.py"
    command = [
        sys.executable,
        str(script_path),
        "--trajectory",
        str(replay_summary["trajectory_path"]),
        "--report",
        str(replay_summary["report_path"]),
        "--output-dir",
        str(output_dir),
        "--fps",
        str(int(fps)),
        "--headless",
    ]
    mp4_path = None
    if export_video:
        mp4_path = output_dir / "isaaclab_replay.mp4"
        command.extend(["--mp4-path", str(mp4_path)])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Single-agent IsaacLab replay rendering failed.\n"
            f"command={' '.join(command)}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    summary_path = output_dir / "isaaclab_replay_summary.json"
    rendered_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {
            "summary_path": str(summary_path),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    if mp4_path is not None:
        rendered_summary["mp4_path"] = str(mp4_path)
    return rendered_summary


def _load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, object]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    metadata = payload.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _build_training_signature(env: SingleGate2DEnv) -> dict[str, object]:
    algorithm = SINGLE_EXPERIMENT_CONFIG.algorithm
    return {
        "experiment_id": SINGLE_EXPERIMENT_CONFIG.experiment_id,
        "control_mode": SINGLE_EXPERIMENT_CONFIG.control_mode,
        "algorithm_name": str(getattr(algorithm, "algorithm_name", "graph_flashsac")),
        "planner_mode": SINGLE_EXPERIMENT_CONFIG.planner_mode,
        "observation_shapes": {name: list(shape) for name, shape in env.observation_shapes.items()},
        "action_dim": int(algorithm.action_dim),
        "log_std_min": float(algorithm.log_std_min),
        "log_std_max": float(algorithm.log_std_max),
    }


def _maybe_resume_training(
    *,
    agent: GraphFlashSACAgent,
    env: SingleGate2DEnv,
    checkpoint_path: str | Path | None,
    resume_mode: SingleResumeMode | str | None,
    seed: int,
) -> dict[str, object] | None:
    if checkpoint_path is None:
        return None

    resolved_path = Path(checkpoint_path)
    metadata = agent.load_checkpoint(resolved_path)
    findings = _single_resume_compatibility_findings(metadata=metadata, env=env)
    incompatible = [name for name, result in findings.items() if not bool(result["compatible"])]
    if incompatible:
        raise ValueError(
            "Single-agent resume checkpoint is incompatible with the current experiment: "
            f"{resolved_path} | failed checks: {', '.join(incompatible)}"
        )

    policy = SINGLE_EXPERIMENT_CONFIG.resume_policy
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
        raise ValueError(f"Unsupported single-agent resume mode: {resolved_mode}")

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


def _single_resume_compatibility_findings(
    *,
    metadata: dict[str, object],
    env: SingleGate2DEnv,
) -> dict[str, dict[str, object]]:
    policy = SINGLE_EXPERIMENT_CONFIG.resume_policy
    signature = dict(metadata.get("training_signature") or {})
    expected_shapes = {name: list(shape) for name, shape in env.observation_shapes.items()}
    actual_experiment_id = str(
        signature.get("experiment_id")
        or metadata.get("experiment_id")
        or metadata.get("summary", {}).get("experiment_id")
        or ""
    ).strip()
    actual_shapes = signature.get("observation_shapes") or metadata.get("observation_shapes")
    actual_action_dim = signature.get("action_dim")
    algorithm = SINGLE_EXPERIMENT_CONFIG.algorithm
    actual_control_mode = signature.get("control_mode")
    legacy_control_modes = {None, SINGLE_EXPERIMENT_CONFIG.control_mode, "flashsac", "graph_sac"}

    findings = {
        "experiment_id": {
            "required": bool(policy.strict_experiment_id),
            "expected": SINGLE_EXPERIMENT_CONFIG.experiment_id,
            "actual": actual_experiment_id,
            "compatible": (
                True
                if not policy.strict_experiment_id
                else bool(actual_experiment_id) and actual_experiment_id == SINGLE_EXPERIMENT_CONFIG.experiment_id
            ),
        },
        "observation_shapes": {
            "required": bool(policy.strict_observation_shapes),
            "expected": expected_shapes,
            "actual": actual_shapes,
            "compatible": (
                True
                if not policy.strict_observation_shapes
                else actual_shapes == expected_shapes
            ),
        },
        "action_dim": {
            "required": True,
            "expected": int(algorithm.action_dim),
            "actual": actual_action_dim,
            "compatible": actual_action_dim in {None, int(algorithm.action_dim)},
        },
        "control_mode": {
            "required": True,
            "expected": SINGLE_EXPERIMENT_CONFIG.control_mode,
            "actual": signature.get("control_mode"),
            "compatible": actual_control_mode in legacy_control_modes,
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
) -> dict[str, object]:
    return {
        "experiment_id": SINGLE_EXPERIMENT_CONFIG.experiment_id,
        "algorithm_name": str(getattr(SINGLE_EXPERIMENT_CONFIG.algorithm, "algorithm_name", "graph_flashsac")),
        "seed": int(seed),
        "checkpoint_step": int(step),
        "checkpoint_kind": kind,
        "training_signature": training_signature,
        "resume_context": resume_context,
    }


def _save_candidate_checkpoint(
    *,
    agent: GraphFlashSACAgent,
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
) -> tuple[dict[str, object], float, str | None, str | None]:
    agent.save_checkpoint(
        checkpoint_path,
        metadata=_checkpoint_metadata(
            step=step,
            kind=kind,
            seed=seed,
            training_signature=training_signature,
            resume_context=resume_context,
        ),
    )
    if selection_eval_episodes > 0:
        eval_summary = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=selection_eval_episodes,
            seed=seed,
            device=device,
        )
        selection_details = build_checkpoint_selection_details(eval_summary)
        score = float(selection_details["score"])
    else:
        eval_summary = {
            "episodes": 0,
            "success_rate": 0.0,
            "mean_episode_reward": 0.0,
            "done_reason_counts": {},
            "episode_summaries": [],
        }
        selection_details = {
            "task_type": "single",
            "score": float(step),
            "metrics": {"episodes": 0},
        }
        score = float(step)

    is_selected = current_selected_checkpoint_path is None or score > current_best_score
    best_score = current_best_score
    selected_checkpoint_path = current_selected_checkpoint_path
    best_alias_path = current_best_alias_path
    if is_selected:
        refreshed_alias = refresh_best_checkpoint_alias(
            checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            alias_name=alias_name,
        )
        best_alias_path = str(refreshed_alias)
        best_score = score
        selected_checkpoint_path = str(checkpoint_path)

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


def _serialize_single_episode_summary(
    *,
    episode_idx: int,
    step_count: int,
    episode_reward: float,
    info: dict[str, object],
) -> dict[str, object]:
    state = info.get("state")
    final_position_xy = None
    final_velocity_xy = None
    if state is not None:
        final_position_xy = list(getattr(state, "position_xy", ()))
        final_velocity_xy = list(getattr(state, "velocity_xy", ()))
    return {
        "episode_index": int(episode_idx),
        "steps": int(step_count),
        "episode_reward": float(episode_reward),
        "done_reason": str(info.get("done_reason") or "unknown"),
        "goal_distance_m": float(info.get("goal_distance_m") or 0.0),
        "signed_clearance_m": float(info.get("signed_clearance_m") or 0.0),
        "final_position_xy": final_position_xy,
        "final_velocity_xy": final_velocity_xy,
    }

