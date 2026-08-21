"""DAgger-style online correction for multi-agent BC warm start."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multi_gate.configs.experiment_config import MULTI_EXPERIMENT_CONFIG, MultiExperimentConfig
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
from multi_gate.imitation import (
    _save_expert_dataset,
    collect_expert_demonstrations,
    load_expert_dataset,
    run_actor_behavior_cloning,
)
from multi_gate.replay import HeuristicFormationReplayController
from multi_gate.training import _build_training_signature, evaluate_actor_checkpoint, run_training
from shared.runtime.artifacts import allocate_dataset_artifacts, default_run_name, write_json
from shared.runtime.training_controls import refresh_best_checkpoint_alias


def run_dagger_warmstart_then_finetune(
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
    initial_bc_epochs: int | None = None,
    initial_bc_batch_size: int | None = None,
    refresh_initial_bc: bool = False,
    dagger_iterations: int | None = None,
    dagger_rollout_episodes: int | None = None,
    dagger_bc_epochs: int | None = None,
    dagger_bc_batch_size: int | None = None,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    save_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
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
    early_stop_eval_thresholds: dict[str, float | None] | None = None,
    early_stop_min_transitions: int = 0,
    early_stop_stable_window_min_length: int | None = None,
    early_stop_late_half_only: bool = False,
    early_stop_planned_total_transitions: int | None = None,
    failure_stop_eval_thresholds: dict[str, float | None] | None = None,
    failure_stop_min_transitions: int = 0,
    failure_stop_stable_window_min_length: int | None = None,
    actor_gate_eval_episodes: int = 0,
    actor_gate_eval_seed: int | None = None,
    actor_gate_thresholds: dict[str, float | None] | None = None,
    skip_rl_after_actor_gate_pass: bool = False,
) -> dict[str, object]:
    """Run initial BC, DAgger corrections, then Graph-FlashSAC fine-tuning."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    dagger_config = selected_config.dagger
    if output_dir is None:
        artifacts = allocate_dataset_artifacts(
            "multi",
            run_name=run_name or default_run_name(f"{selected_config.experiment_id}_dagger"),
        )
        dagger_dir = artifacts.output_dir
    else:
        dagger_dir = Path(output_dir)
        dagger_dir.mkdir(parents=True, exist_ok=True)

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
        retain_failed_episodes=False,
        output_dir=dagger_dir,
    )
    dataset_path = Path(collection_summary["dataset_path"])
    if initial_actor_checkpoint is None or bool(refresh_initial_bc):
        bc_summary = run_actor_behavior_cloning(
            dataset_path=dataset_path,
            experiment_config=selected_config,
            device=device,
            seed=seed,
            initial_actor_checkpoint=initial_actor_checkpoint,
            epochs=initial_bc_epochs,
            batch_size=initial_bc_batch_size,
            output_dir=dagger_dir,
        )
        initial_bc_summary = dict(bc_summary)
        actor_checkpoint_path = str(bc_summary["actor_checkpoint_path"])
    else:
        bc_summary = {
            "actor_checkpoint_path": str(initial_actor_checkpoint),
            "source": "external_actor_warmstart",
            "dataset_path": str(dataset_path),
            "output_dir": str(dagger_dir),
        }
        initial_bc_summary = dict(bc_summary)
        actor_checkpoint_path = str(initial_actor_checkpoint)

    iteration_count = int(dagger_iterations if dagger_iterations is not None else dagger_config.iterations)
    rollout_episodes = int(
        dagger_rollout_episodes
        if dagger_rollout_episodes is not None
        else dagger_config.rollout_episodes_per_iteration
    )
    bc_epochs_per_iteration = int(
        dagger_bc_epochs if dagger_bc_epochs is not None else dagger_config.bc_epochs_per_iteration
    )
    iteration_summaries: list[dict[str, object]] = []
    for iteration_idx in range(1, iteration_count + 1):
        corrections = collect_dagger_corrections(
            actor_checkpoint_path=actor_checkpoint_path,
            experiment_config=selected_config,
            teacher_actor_checkpoint=expert_teacher_actor_checkpoint,
            num_agents=num_agents,
            max_sampled_agents=max_sampled_agents,
            episodes=rollout_episodes,
            seed=seed + iteration_idx * 10_000,
            device=device,
        )
        if corrections["sample_count"] > 0:
            dataset_path = append_dagger_corrections(
                dataset_path=dataset_path,
                corrections=corrections,
                output_path=dagger_dir / selected_config.imitation.dataset_name,
            )
            bc_summary = run_actor_behavior_cloning(
                dataset_path=dataset_path,
                experiment_config=selected_config,
                device=device,
                seed=seed + iteration_idx,
                initial_actor_checkpoint=actor_checkpoint_path,
                epochs=bc_epochs_per_iteration,
                batch_size=dagger_bc_batch_size,
                output_dir=dagger_dir,
            )
            actor_checkpoint_path = str(bc_summary["actor_checkpoint_path"])
        iteration_summaries.append(
            {
                "iteration": iteration_idx,
                "correction_sample_count": int(corrections["sample_count"]),
                "risk_reason_counts": corrections["risk_reason_counts"],
                "teacher_source": corrections.get("teacher_source"),
                "teacher_actor_checkpoint": corrections.get("teacher_actor_checkpoint"),
                "dataset_path": str(dataset_path),
                "actor_checkpoint_path": actor_checkpoint_path,
            }
        )

    actor_gate_eval = None
    actor_gate_assessment = None
    if int(actor_gate_eval_episodes) > 0:
        actor_gate_eval = evaluate_actor_checkpoint(
            actor_checkpoint_path=actor_checkpoint_path,
            episodes=int(actor_gate_eval_episodes),
            seed=int(seed if actor_gate_eval_seed is None else actor_gate_eval_seed),
            device=device,
            num_agents=num_agents,
            experiment_config=selected_config,
        )
        actor_gate_assessment = _assess_actor_gate(
            eval_summary=actor_gate_eval,
            thresholds=dict(actor_gate_thresholds or {}),
        )
        if bool(actor_gate_assessment["passed"]) and bool(skip_rl_after_actor_gate_pass):
            training_summary = _promote_actor_gate_checkpoint(
                actor_checkpoint_path=actor_checkpoint_path,
                base_full_checkpoint_path=initial_actor_checkpoint,
                experiment_config=selected_config,
                checkpoint_dir=checkpoint_dir,
                device=device,
                seed=seed,
            )
            summary = {
                "experiment_id": selected_config.experiment_id,
                "collection": collection_summary,
                "initial_behavior_cloning": initial_bc_summary,
                "final_behavior_cloning": bc_summary,
                "dagger_iterations": iteration_summaries,
                "final_actor_checkpoint_path": actor_checkpoint_path,
                "initial_actor_checkpoint_path": (
                    None if initial_actor_checkpoint is None else str(initial_actor_checkpoint)
                ),
                "refresh_initial_bc": bool(refresh_initial_bc),
                "actor_gate_eval": actor_gate_eval,
                "actor_gate_assessment": actor_gate_assessment,
                "fine_tuning": training_summary,
            }
            summary_path = Path(training_summary.get("log_dir") or dagger_dir) / "dagger_then_finetune_summary.json"
            summary["summary_path"] = str(summary_path)
            write_json(summary_path, summary)
            return summary
        if not bool(actor_gate_assessment["passed"]) and bool(skip_rl_after_actor_gate_pass):
            training_summary = {
                "skipped": True,
                "skip_reason": "actor_gate_failed",
                "actor_gate_eval": actor_gate_eval,
                "actor_gate_assessment": actor_gate_assessment,
            }
            summary = {
                "experiment_id": selected_config.experiment_id,
                "collection": collection_summary,
                "initial_behavior_cloning": initial_bc_summary,
                "final_behavior_cloning": bc_summary,
                "dagger_iterations": iteration_summaries,
                "final_actor_checkpoint_path": actor_checkpoint_path,
                "initial_actor_checkpoint_path": (
                    None if initial_actor_checkpoint is None else str(initial_actor_checkpoint)
                ),
                "refresh_initial_bc": bool(refresh_initial_bc),
                "actor_gate_eval": actor_gate_eval,
                "actor_gate_assessment": actor_gate_assessment,
                "fine_tuning": training_summary,
            }
            summary_path = dagger_dir / "dagger_then_finetune_summary.json"
            summary["summary_path"] = str(summary_path)
            write_json(summary_path, summary)
            return summary

    training_summary = run_training(
        train_steps=train_steps,
        num_envs=num_envs,
        seed=seed,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        num_agents=num_agents,
        max_sampled_agents=max_sampled_agents,
        learning_starts=learning_starts,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        log_every=log_every,
        experiment_config=selected_config,
        warmstart_actor_checkpoint=actor_checkpoint_path,
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
    )
    summary = {
        "experiment_id": selected_config.experiment_id,
        "collection": collection_summary,
        "initial_behavior_cloning": initial_bc_summary,
        "final_behavior_cloning": bc_summary,
        "dagger_iterations": iteration_summaries,
        "final_actor_checkpoint_path": actor_checkpoint_path,
        "initial_actor_checkpoint_path": (
            None if initial_actor_checkpoint is None else str(initial_actor_checkpoint)
        ),
        "refresh_initial_bc": bool(refresh_initial_bc),
        "actor_gate_eval": actor_gate_eval,
        "actor_gate_assessment": actor_gate_assessment,
        "fine_tuning": training_summary,
    }
    summary_dir = Path(training_summary.get("log_dir") or dagger_dir)
    summary_path = summary_dir / "dagger_then_finetune_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def _promote_actor_gate_checkpoint(
    *,
    actor_checkpoint_path: str | Path,
    base_full_checkpoint_path: str | Path | None,
    experiment_config: MultiExperimentConfig,
    checkpoint_dir: str | Path | None,
    device: str | None,
    seed: int,
) -> dict[str, object]:
    if base_full_checkpoint_path is None:
        raise ValueError("base_full_checkpoint_path is required to promote actor-gate checkpoints")
    target_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path(actor_checkpoint_path).parent / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    env = MultiGate2DEnv(
        multi_config=experiment_config,
        env_config=experiment_config.environment,
        observation_config=experiment_config.observation,
        formation_config=experiment_config.formation,
        planner_config=experiment_config.planner,
    )
    agent = GraphMASACAgent.from_defaults(
        obs_shapes=env.observation_shapes,
        device=device,
        seed=seed,
        obs_config=experiment_config.observation,
        masac_config=experiment_config.algorithm,
        max_agents_soft=experiment_config.max_agents_soft,
    )
    base_metadata = agent.load_checkpoint(base_full_checkpoint_path)
    actor_metadata = agent.load_actor_checkpoint(actor_checkpoint_path)
    checkpoint_path = target_dir / experiment_config.algorithm.checkpoint_name
    agent.save_checkpoint(
        checkpoint_path,
        metadata={
            "source": "actor_gate_promoted_bc_dagger_actor",
            "experiment_id": experiment_config.experiment_id,
            "training_signature": _build_training_signature(env=env, experiment_config=experiment_config),
            "base_full_checkpoint_path": str(base_full_checkpoint_path),
            "actor_checkpoint_path": str(actor_checkpoint_path),
            "base_metadata": base_metadata,
            "actor_metadata": actor_metadata,
        },
    )
    best_alias_path = refresh_best_checkpoint_alias(
        checkpoint_path,
        checkpoint_dir=target_dir,
        alias_name=experiment_config.checkpoint_policy.best_alias_name,
    )
    latest_alias_path = refresh_best_checkpoint_alias(
        checkpoint_path,
        checkpoint_dir=target_dir,
        alias_name="latest_agent.pt",
    )
    return {
        "skipped": True,
        "skip_reason": "actor_gate_passed_skip_rl",
        "transitions_collected": 0,
        "completed_episodes": 0,
        "checkpoint_path": str(latest_alias_path),
        "best_checkpoint_path": str(checkpoint_path),
        "best_alias_path": str(best_alias_path),
        "final_checkpoint_path": str(checkpoint_path),
        "latest_alias_path": str(latest_alias_path),
        "checkpoint_dir": str(target_dir),
        "log_dir": str(target_dir.parent / "logs"),
    }


def _assess_actor_gate(
    *,
    eval_summary: dict[str, object],
    thresholds: dict[str, float | None],
) -> dict[str, object]:
    specs = (
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
    )
    checks: dict[str, dict[str, object]] = {}
    for metric_name, threshold_key, mode in specs:
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
        "passed": bool(checks) and len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "checks": checks,
    }


def collect_dagger_corrections(
    *,
    actor_checkpoint_path: str | Path,
    experiment_config: MultiExperimentConfig,
    teacher_actor_checkpoint: str | Path | None = None,
    episodes: int,
    seed: int,
    device: str | None,
    num_agents: int | None,
    max_sampled_agents: int | None,
) -> dict[str, object]:
    """Collect teacher actions only at policy-visited high-risk states."""

    env = MultiGate2DEnv(
        multi_config=experiment_config,
        env_config=experiment_config.environment,
        observation_config=experiment_config.observation,
        formation_config=experiment_config.formation,
        planner_config=experiment_config.planner,
    )
    agent = GraphMASACAgent.from_defaults(
        obs_shapes=env.observation_shapes,
        device=device,
        seed=seed,
        obs_config=experiment_config.observation,
        masac_config=experiment_config.algorithm,
        max_agents_soft=experiment_config.max_agents_soft,
    )
    agent.load_actor_checkpoint(actor_checkpoint_path)
    teacher_agent: GraphMASACAgent | None = None
    if teacher_actor_checkpoint is not None:
        teacher_agent = GraphMASACAgent.from_defaults(
            obs_shapes=env.observation_shapes,
            device=device,
            seed=seed + 17,
            obs_config=experiment_config.observation,
            masac_config=experiment_config.algorithm,
            max_agents_soft=experiment_config.max_agents_soft,
            build_replay_buffer=False,
        )
        teacher_agent.load_actor_checkpoint(teacher_actor_checkpoint)

    obs_records: dict[str, list[np.ndarray]] = {name: [] for name in env.observation_shapes}
    action_records: list[np.ndarray] = []
    risk_reason_counts: dict[str, int] = {}
    team_sizes_seen: set[int] = set()
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "") or "").strip().lower()
    dynamic_gate_scene = "dynamic_gate_density" in scene_mode

    for episode_idx in range(int(episodes)):
        active_agents = _sample_team_size(
            seed=seed + episode_idx,
            num_agents=num_agents,
            max_sampled_agents=max_sampled_agents,
            experiment_config=experiment_config,
        )
        team_sizes_seen.add(active_agents)
        observation, info = env.reset(seed=seed + episode_idx, num_agents=active_agents)
        teacher = None if teacher_agent is not None else HeuristicFormationReplayController(env)
        for step_idx in range(int(experiment_config.dagger.max_steps_per_episode)):
            risk_reasons = _risk_reasons(info=info, experiment_config=experiment_config)
            scheduled_query = step_idx % max(int(experiment_config.dagger.query_every_n_steps), 1) == 0
            should_query = bool(risk_reasons) or (scheduled_query and not dynamic_gate_scene)
            if should_query:
                teacher_action = (
                    teacher_agent.act(observation, deterministic=True)
                    if teacher_agent is not None
                    else teacher.act()
                )
                for name, array in observation.items():
                    obs_records[name].append(np.asarray(array, dtype=np.float32).copy())
                action_records.append(np.asarray(teacher_action, dtype=np.float32).copy())
                for reason in risk_reasons or ["scheduled_teacher_query"]:
                    risk_reason_counts[reason] = risk_reason_counts.get(reason, 0) + 1
                if len(action_records) >= int(experiment_config.dagger.max_corrections_per_iteration):
                    break
            policy_action = agent.act(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(policy_action)
            if terminated or truncated:
                break
        if len(action_records) >= int(experiment_config.dagger.max_corrections_per_iteration):
            break

    return {
        "observations": {
            name: _stack_or_empty(records, env.observation_shapes[name])
            for name, records in obs_records.items()
        },
        "actions": _stack_or_empty(action_records, env.action_shape),
        "sample_count": int(len(action_records)),
        "risk_reason_counts": risk_reason_counts,
        "team_sizes_seen": sorted(team_sizes_seen),
        "teacher_source": "actor_checkpoint" if teacher_actor_checkpoint is not None else "heuristic_replay_controller",
        "teacher_actor_checkpoint": None if teacher_actor_checkpoint is None else str(teacher_actor_checkpoint),
    }


def append_dagger_corrections(
    *,
    dataset_path: str | Path,
    corrections: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Append one DAgger correction batch to an existing expert dataset."""

    dataset = load_expert_dataset(dataset_path)
    if int(corrections.get("sample_count") or 0) <= 0:
        return Path(dataset_path)
    old_obs = dataset["observations"]
    old_actions = dataset["actions"]
    new_obs = corrections["observations"]
    new_actions = corrections["actions"]
    merged_obs = {
        name: np.concatenate([old_obs[name], new_obs[name]], axis=0).astype(np.float32, copy=False)
        for name in old_obs
    }
    merged_actions = np.concatenate([old_actions, new_actions], axis=0).astype(np.float32, copy=False)
    metadata = dict(dataset["metadata"])
    metadata["dagger_appended_steps"] = int(metadata.get("dagger_appended_steps", 0)) + int(corrections["sample_count"])
    metadata["dagger_risk_reason_counts"] = corrections["risk_reason_counts"]
    return _save_expert_dataset(Path(output_path), observations=merged_obs, actions=merged_actions, metadata=metadata)


def _risk_reasons(*, info: dict[str, object], experiment_config: MultiExperimentConfig) -> list[str]:
    dagger_config = experiment_config.dagger
    snapshot = info.get("snapshot")
    mean_slot_error = float(getattr(snapshot, "mean_slot_error_m", 0.0)) if snapshot is not None else 0.0
    min_clearance = float(info.get("min_clearance_m") or 0.0)
    min_pair_distance = float(info.get("min_pair_distance_m") or float("inf"))
    reasons: list[str] = []
    if min_clearance < float(dagger_config.risk_clearance_m):
        reasons.append("low_clearance")
    if min_pair_distance < float(dagger_config.risk_pair_distance_m):
        reasons.append("low_pair_distance")
    if mean_slot_error > float(dagger_config.risk_slot_error_m):
        reasons.append("slot_error")
    if not bool(info.get("height_contract_passed", True)):
        reasons.append("height_contract_failed")
    if bool(info.get("height_escape_failure", False)):
        reasons.append("height_escape_failure")
    if bool(info.get("side_bypass_failure", False)):
        reasons.append("side_bypass_failure")
    if bool(info.get("corridor_miss_failure", False)):
        reasons.append("corridor_miss_failure")
    if bool(info.get("formation_line_collapse_failure", False)):
        reasons.append("formation_line_collapse_failure")
    done_reason = str(info.get("done_reason") or "")
    dynamic_gate_count = int(info.get("dynamic_gate_count") or 0)
    if done_reason and dynamic_gate_count > 0 and not bool(info.get("corridor_completed", False)):
        reasons.append("terminal_without_corridor_completion")
    if done_reason in {
        "gate_post_collision",
        "agent_collision",
        "out_of_bounds",
        "height_escape_failure",
        "side_bypass_failure",
        "corridor_miss_failure",
        "formation_line_collapse_failure",
    }:
        reasons.append(done_reason)
    return reasons


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
    buckets = [
        int(size)
        for size in experiment_config.size_invariance.bucket_team_sizes
        if experiment_config.min_agents <= int(size) <= max_agents
    ]
    if buckets:
        return int(rng.choice(np.asarray(buckets, dtype=np.int64)))
    return int(rng.integers(experiment_config.min_agents, max_agents + 1))


def _stack_or_empty(records: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not records:
        return np.zeros((0,) + tuple(shape), dtype=np.float32)
    return np.asarray(records, dtype=np.float32)

