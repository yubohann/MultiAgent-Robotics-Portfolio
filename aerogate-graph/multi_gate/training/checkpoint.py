from __future__ import annotations

"""Training helpers for the multi-agent 2D gate experiment."""


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
    MultiExperimentConfig,
    is_exp3_empty_scene_mode,
    is_exp3_gate_scene_mode,
    is_dynamic_gate_density_scene_mode
)
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
from shared.runtime.training_controls import (
    build_checkpoint_selection_details,
    refresh_best_checkpoint_alias,
)


MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv

from .evaluation import (
    evaluate_checkpoint,
    evaluate_size_buckets
)

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
            "Replay buffer snapshots are not persisted in gate_graph_2d_minimal; resume always starts with an empty replay buffer.",
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