"""Continue-train an 8-drone dynamic-gate checkpoint across gate densities.

The runner trains separate static and dynamic branches and keeps a trained
checkpoint only when post-eval does not regress hard safety metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any


def _bootstrap_imports() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ROOT = _bootstrap_imports()

from multi_gate.configs import get_multi_experiment_config  # noqa: E402
from multi_gate.dagger import run_dagger_warmstart_then_finetune  # noqa: E402
from multi_gate.scripts.run_dynamic_gate_density_8d_curriculum import (  # noqa: E402
    DynamicGateCurriculumStage,
    _eval_score,
    _hard_metric_regression_failure,
    _promotion_gate_failure,
    _stage_config,
    _stages,
)
from multi_gate.training import evaluate_checkpoint, run_training  # noqa: E402


DEFAULT_C6A_110_R3 = (
    ROOT
    / "runtime"
    / "supervised_e5_nested_20260506_c6a_gate36_speed110_amp090_drone240_retry3_from1075"
    / "stages"
    / "06_C6a_gate36_speed110_amp090_drone240_retry3_from1075"
    / "checkpoints"
    / "best_agent.pt"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _template_for_gate(gate_count: int) -> DynamicGateCurriculumStage:
    candidates = _stages()
    best = min(
        candidates,
        key=lambda stage: (
            abs(int(stage.gate_count) - int(gate_count)),
            -int(stage.gate_count),
        ),
    )
    return best


def _speed_for_dynamic_gate(gate_count: int) -> float:
    if gate_count <= 6:
        return 0.35
    if gate_count <= 12:
        return 0.50
    if gate_count <= 18:
        return 0.80
    if gate_count <= 24:
        return 1.00
    if gate_count <= 30:
        return 1.10
    if gate_count <= 36:
        return 1.20
    if gate_count <= 42:
        return 1.40
    if gate_count <= 48:
        return 1.60
    if gate_count <= 54:
        return 1.80
    return 2.00


def _amplitude_for_dynamic_gate(gate_count: int) -> float:
    if gate_count <= 6:
        return 0.45
    if gate_count <= 12:
        return 0.60
    if gate_count <= 18:
        return 0.75
    if gate_count <= 24:
        return 0.85
    if gate_count <= 30:
        return 0.90
    if gate_count <= 36:
        return 0.95
    if gate_count <= 42:
        return 1.00
    if gate_count <= 48:
        return 1.05
    if gate_count <= 54:
        return 1.10
    return 1.20


def _build_stages(
    mode: str,
    gates: list[int],
    steps_scale: float,
    stage_label: str,
    dynamic_speed_scale: float,
    dynamic_amplitude_scale: float,
) -> list[DynamicGateCurriculumStage]:
    stages: list[DynamicGateCurriculumStage] = []
    for gate in gates:
        template = _template_for_gate(gate)
        is_dynamic = mode == "dynamic"
        speed = (_speed_for_dynamic_gate(gate) * float(dynamic_speed_scale)) if is_dynamic else 0.0
        amplitude = (_amplitude_for_dynamic_gate(gate) * float(dynamic_amplitude_scale)) if is_dynamic else 0.0
        base_steps = int(template.train_steps)
        if gate <= 18:
            base_steps = max(base_steps, 3072)
        elif gate >= 48:
            base_steps = max(base_steps, 6144)
        train_steps = max(256, int(round(base_steps * float(steps_scale))))
        stages.append(
            replace(
                template,
                name=f"{mode}_gate{gate:02d}_from_{stage_label}",
                gate_count=int(gate),
                speed_mps=float(speed),
                amplitude_m=float(amplitude),
                train_steps=int(train_steps),
                notes=(
                    f"{mode} continuation from {stage_label}; gate_count={gate}, "
                    f"speed={speed:.2f}, amplitude={amplitude:.2f}."
                ),
            )
        )
    return stages


def _select_checkpoint(
    *,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    current_checkpoint: Path,
    trained_checkpoint: Path,
    config: Any,
    stage: DynamicGateCurriculumStage,
    min_promotion_success_rate: float,
    max_promotion_dynamic_collision_rate: float,
    reject_zero_success_promotion: bool,
) -> tuple[Path, dict[str, Any]]:
    pre_score = _eval_score(before)
    post_score = _eval_score(after)
    record = {
        "pre_score": float(pre_score),
        "post_score": float(post_score),
        "trained_checkpoint_path": str(trained_checkpoint),
    }
    promotion_failure = _promotion_gate_failure(after, config, stage)
    hard_regression_failure = _hard_metric_regression_failure(before, after)
    if after:
        post_success = max(
            float(after.get("success_rate") or 0.0),
            float(after.get("team_success_rate") or 0.0),
        )
        post_dynamic_collision = max(
            float(after.get("gate_post_collision_rate") or 0.0),
            float(after.get("obstacle_collision_rate") or 0.0),
            float(after.get("dynamic_gate_collision_rate") or 0.0),
        )
        if reject_zero_success_promotion and post_success <= 0.0:
            record["selected"] = "input_checkpoint"
            record["reason"] = "zero_success_promotion_rejected"
            return current_checkpoint, record
        if (
            post_success < float(min_promotion_success_rate)
            and post_dynamic_collision >= float(max_promotion_dynamic_collision_rate)
        ):
            record["selected"] = "input_checkpoint"
            record["reason"] = (
                "catastrophic_dynamic_collision_rejected: "
                f"success={post_success:.3f}, dynamic_collision={post_dynamic_collision:.3f}"
            )
            return current_checkpoint, record
    if promotion_failure:
        record["selected"] = "input_checkpoint"
        record["reason"] = f"promotion_gate_failed: {promotion_failure}"
        return current_checkpoint, record
    if hard_regression_failure:
        record["selected"] = "input_checkpoint"
        record["reason"] = f"hard_metric_regression: {hard_regression_failure}"
        return current_checkpoint, record
    record["selected"] = "trained_checkpoint"
    record["reason"] = "post_eval_accepted" if post_score >= pre_score else "soft_score_regression_hard_metrics_clean"
    return trained_checkpoint, record


def _run_branch(args: argparse.Namespace, mode: str, base_checkpoint: Path, output_root: Path) -> dict[str, Any]:
    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    branch_dir = output_root / mode
    branch_dir.mkdir(parents=True, exist_ok=True)
    gates = [int(item) for item in (args.static_gates if mode == "static" else args.dynamic_gates)]
    stages = _build_stages(
        mode,
        gates,
        args.steps_scale,
        str(args.stage_label),
        float(args.dynamic_speed_scale),
        float(args.dynamic_amplitude_scale),
    )
    checkpoint = base_checkpoint
    records: list[dict[str, Any]] = []

    for index, stage in enumerate(stages):
        stage_dir = branch_dir / f"{index:02d}_{stage.name}"
        config = _stage_config(base_config, stage)
        if (
            args.train_gate_post_radius_m is not None
            or args.train_drone_radius_m is not None
            or args.train_gate_half_width_m is not None
            or args.train_gate_half_width_scale is not None
        ):
            gate_radius = (
                float(args.train_gate_post_radius_m)
                if args.train_gate_post_radius_m is not None
                else float(config.dynamic_gate_density.gate_post_radius_m)
            )
            drone_radius = (
                float(args.train_drone_radius_m)
                if args.train_drone_radius_m is not None
                else float(config.environment.drone_radius_m)
            )
            gate_half_width = (
                float(args.train_gate_half_width_m)
                if args.train_gate_half_width_m is not None
                else float(config.dynamic_gate_density.gate_half_width_m)
            )
            if args.train_gate_half_width_scale is not None:
                gate_half_width *= float(args.train_gate_half_width_scale)
            config = replace(
                config,
                dynamic_gate_density=replace(
                    config.dynamic_gate_density,
                    gate_post_radius_m=gate_radius,
                    drone_radius_m=drone_radius,
                    gate_half_width_m=gate_half_width,
                ),
                environment=replace(
                    config.environment,
                    drone_radius_m=drone_radius,
                ),
            )
        if (
            args.train_safety_clearance_m is not None
            or args.train_obstacle_shield_margin_m is not None
            or args.train_separation_shield_margin_m is not None
            or args.train_inter_agent_safe_distance_m is not None
            or args.train_post_gate_cruise_min_pair_distance_m is not None
        ):
            config = replace(
                config,
                environment=replace(
                    config.environment,
                    inter_agent_safe_distance_m=(
                        float(args.train_inter_agent_safe_distance_m)
                        if args.train_inter_agent_safe_distance_m is not None
                        else float(config.environment.inter_agent_safe_distance_m)
                    ),
                    safety_clearance_m=(
                        float(args.train_safety_clearance_m)
                        if args.train_safety_clearance_m is not None
                        else float(config.environment.safety_clearance_m)
                    ),
                    action_safety_shield_obstacle_margin_m=(
                        float(args.train_obstacle_shield_margin_m)
                        if args.train_obstacle_shield_margin_m is not None
                        else float(config.environment.action_safety_shield_obstacle_margin_m)
                    ),
                    action_safety_shield_separation_margin_m=(
                        float(args.train_separation_shield_margin_m)
                        if args.train_separation_shield_margin_m is not None
                        else float(config.environment.action_safety_shield_separation_margin_m)
                    ),
                    action_safety_shield_post_gate_cruise_min_pair_distance_m=(
                        float(args.train_post_gate_cruise_min_pair_distance_m)
                        if args.train_post_gate_cruise_min_pair_distance_m is not None
                        else float(config.environment.action_safety_shield_post_gate_cruise_min_pair_distance_m)
                    ),
                ),
            )
        if args.train_lateral_spacing_m is not None or args.train_longitudinal_spacing_m is not None:
            config = replace(
                config,
                formation=replace(
                    config.formation,
                    lateral_spacing_m=(
                        float(args.train_lateral_spacing_m)
                        if args.train_lateral_spacing_m is not None
                        else float(config.formation.lateral_spacing_m)
                    ),
                    longitudinal_spacing_m=(
                        float(args.train_longitudinal_spacing_m)
                        if args.train_longitudinal_spacing_m is not None
                        else float(config.formation.longitudinal_spacing_m)
                    ),
                ),
            )
        if not bool(args.enable_guidance_runtime):
            config = replace(
                config,
                reasoning=replace(
                    config.reasoning,
                    route_guidance_enabled=False,
                    guidance_shadow_mode=False,
                    guidance_async_enabled=False,
                    guidance_cache_enabled=False,
                    guidance_provider="none",
                ),
            )
        before = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            episodes=int(args.pre_eval_episodes),
            seed=int(args.seed) + index * 100,
            device=args.eval_device,
            num_agents=int(args.num_agents),
            experiment_config=config,
        )
        if str(args.pipeline) == "dagger":
            summary = run_dagger_warmstart_then_finetune(
                experiment_config=config,
                train_steps=int(stage.train_steps),
                num_envs=int(args.num_envs),
                seed=int(args.seed) + index * 1000,
                device=args.train_device,
                num_agents=int(args.num_agents),
                initial_actor_checkpoint=checkpoint,
                expert_episodes=int(args.bc_expert_episodes),
                expert_target_retained_episodes=int(args.bc_target_retained_episodes),
                expert_collection_workers=int(args.bc_collection_workers),
                max_steps_per_episode=None,
                initial_bc_epochs=int(args.bc_epochs),
                initial_bc_batch_size=int(args.bc_batch_size),
                refresh_initial_bc=True,
                dagger_iterations=int(args.dagger_iterations),
                dagger_rollout_episodes=int(args.dagger_rollout_episodes),
                dagger_bc_epochs=int(args.dagger_bc_epochs),
                dagger_bc_batch_size=int(args.dagger_bc_batch_size),
                output_dir=stage_dir / "dagger",
                save_dir=stage_dir,
                log_dir=stage_dir / "logs",
                checkpoint_dir=stage_dir / "checkpoints",
                learning_starts=int(args.learning_starts),
                batch_size=int(args.batch_size),
                updates_per_step=int(args.updates_per_step),
                log_every=int(args.log_every),
                checkpoint_interval_steps=int(args.checkpoint_interval_transitions),
                selection_eval_episodes=int(args.selection_eval_episodes),
                periodic_eval_episodes=int(args.periodic_eval_episodes),
                periodic_eval_interval_steps=int(args.quick_eval_interval_transitions),
                early_stop_min_transitions=int(args.early_stop_min_transitions),
                early_stop_stable_window_min_length=int(args.pass_window),
                actor_gate_eval_episodes=int(args.actor_gate_eval_episodes),
                actor_gate_eval_seed=int(args.seed) + index * 100 + 17,
                actor_gate_thresholds=None,
                skip_rl_after_actor_gate_pass=False,
            )
            fine_tuning = dict(summary.get("fine_tuning") or {})
            trained = Path(
                str(
                    fine_tuning.get("best_alias_path")
                    or fine_tuning.get("best_checkpoint_path")
                    or fine_tuning.get("checkpoint_path")
                    or fine_tuning.get("final_checkpoint_path")
                )
            )
        else:
            summary = run_training(
                train_steps=int(stage.train_steps),
                num_envs=int(args.num_envs),
                seed=int(args.seed) + index * 1000,
                device=args.train_device,
                save_dir=stage_dir,
                log_dir=stage_dir / "logs",
                checkpoint_dir=stage_dir / "checkpoints",
                num_agents=int(args.num_agents),
                learning_starts=int(args.learning_starts),
                batch_size=int(args.batch_size),
                updates_per_step=int(args.updates_per_step),
                log_every=int(args.log_every),
                experiment_config=config,
                resume_checkpoint=checkpoint,
                resume_mode="reset_train_state",
                checkpoint_interval_steps=int(args.checkpoint_interval_transitions),
                selection_eval_episodes=int(args.selection_eval_episodes),
                periodic_eval_episodes=int(args.periodic_eval_episodes),
                periodic_eval_interval_steps=int(args.quick_eval_interval_transitions),
                early_stop_min_transitions=int(args.early_stop_min_transitions),
                early_stop_stable_window_min_length=int(args.pass_window),
            )
            trained = Path(
                str(
                    summary.get("best_alias_path")
                    or summary.get("best_checkpoint_path")
                    or summary.get("checkpoint_path")
                    or summary.get("final_checkpoint_path")
                )
            )
        if not str(trained) or str(trained) == "None":
            raise RuntimeError(f"Training did not produce a checkpoint for {mode} gate={stage.gate_count}")
        after = evaluate_checkpoint(
            checkpoint_path=trained,
            episodes=int(args.post_eval_episodes),
            seed=int(args.seed) + index * 100 + 33,
            device=args.eval_device,
            num_agents=int(args.num_agents),
            experiment_config=config,
        )
        selected, selection = _select_checkpoint(
            before=before,
            after=after,
            current_checkpoint=checkpoint,
            trained_checkpoint=trained,
            config=config,
            stage=stage,
            min_promotion_success_rate=float(args.min_promotion_success_rate),
            max_promotion_dynamic_collision_rate=float(args.max_promotion_dynamic_collision_rate),
            reject_zero_success_promotion=bool(args.reject_zero_success_promotion),
        )
        record = {
            "stage_index": index,
            "mode": mode,
            "stage": asdict(stage),
            "input_checkpoint": str(checkpoint),
            "train_summary": summary,
            "pre_eval": before,
            "post_eval": after,
            "checkpoint_selection": selection,
            "output_checkpoint": str(selected),
        }
        records.append(_json_safe(record))
        checkpoint = selected
        (branch_dir / "branch_summary.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "start_checkpoint": str(base_checkpoint),
                    "current_checkpoint": str(checkpoint),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{mode}] gate={stage.gate_count} speed={stage.speed_mps:.2f} "
            f"pre={before.get('team_success_rate')} post={after.get('team_success_rate')} "
            f"selected={selection.get('selected')}",
            flush=True,
        )

    return {
        "mode": mode,
        "start_checkpoint": str(base_checkpoint),
        "final_checkpoint": str(checkpoint),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-checkpoint", type=Path, default=DEFAULT_C6A_110_R3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage-label", type=str, default="checkpoint")
    parser.add_argument("--mode", choices=["static", "dynamic", "both"], default="both")
    parser.add_argument("--static-gates", type=int, nargs="+", default=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
    parser.add_argument("--dynamic-gates", type=int, nargs="+", default=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
    parser.add_argument("--steps-scale", type=float, default=0.20)
    parser.add_argument("--pipeline", choices=["dagger", "rl"], default="dagger")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--train-device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--pre-eval-episodes", type=int, default=2)
    parser.add_argument("--post-eval-episodes", type=int, default=3)
    parser.add_argument("--selection-eval-episodes", type=int, default=2)
    parser.add_argument("--periodic-eval-episodes", type=int, default=1)
    parser.add_argument("--checkpoint-interval-transitions", type=int, default=2048)
    parser.add_argument("--quick-eval-interval-transitions", type=int, default=2048)
    parser.add_argument("--pass-window", type=int, default=1)
    parser.add_argument("--early-stop-min-transitions", type=int, default=2048)
    parser.add_argument("--learning-starts", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=128)
    parser.add_argument("--bc-expert-episodes", type=int, default=6)
    parser.add_argument("--bc-target-retained-episodes", type=int, default=3)
    parser.add_argument("--bc-collection-workers", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=2)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--dagger-iterations", type=int, default=1)
    parser.add_argument("--dagger-rollout-episodes", type=int, default=3)
    parser.add_argument("--dagger-bc-epochs", type=int, default=1)
    parser.add_argument("--dagger-bc-batch-size", type=int, default=512)
    parser.add_argument("--actor-gate-eval-episodes", type=int, default=0)
    parser.add_argument("--train-gate-post-radius-m", type=float, default=None)
    parser.add_argument("--train-drone-radius-m", type=float, default=None)
    parser.add_argument("--train-gate-half-width-m", type=float, default=None)
    parser.add_argument("--train-gate-half-width-scale", type=float, default=None)
    parser.add_argument("--train-safety-clearance-m", type=float, default=None)
    parser.add_argument("--train-obstacle-shield-margin-m", type=float, default=None)
    parser.add_argument("--train-separation-shield-margin-m", type=float, default=None)
    parser.add_argument("--train-inter-agent-safe-distance-m", type=float, default=None)
    parser.add_argument("--train-post-gate-cruise-min-pair-distance-m", type=float, default=None)
    parser.add_argument("--train-lateral-spacing-m", type=float, default=None)
    parser.add_argument("--train-longitudinal-spacing-m", type=float, default=None)
    parser.add_argument("--dynamic-speed-scale", type=float, default=1.0)
    parser.add_argument("--dynamic-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--enable-guidance-runtime", action="store_true")
    parser.add_argument("--min-promotion-success-rate", type=float, default=0.01)
    parser.add_argument("--max-promotion-dynamic-collision-rate", type=float, default=0.999)
    parser.add_argument("--reject-zero-success-promotion", action="store_true", default=True)
    args = parser.parse_args()

    checkpoint = args.resume_checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir or ROOT / "runtime" / f"multi_{args.stage_label}_static_dynamic_continuation"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_modes = ["static", "dynamic"] if args.mode == "both" else [args.mode]
    branches = [_run_branch(args, mode, checkpoint, output_dir) for mode in requested_modes]
    payload = {
        "start_checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "stage_label": str(args.stage_label),
        "train_gate_post_radius_m": args.train_gate_post_radius_m,
        "train_drone_radius_m": args.train_drone_radius_m,
        "train_gate_half_width_m": args.train_gate_half_width_m,
        "train_gate_half_width_scale": args.train_gate_half_width_scale,
        "train_safety_clearance_m": args.train_safety_clearance_m,
        "train_obstacle_shield_margin_m": args.train_obstacle_shield_margin_m,
        "train_separation_shield_margin_m": args.train_separation_shield_margin_m,
        "train_inter_agent_safe_distance_m": args.train_inter_agent_safe_distance_m,
        "train_post_gate_cruise_min_pair_distance_m": args.train_post_gate_cruise_min_pair_distance_m,
        "train_lateral_spacing_m": args.train_lateral_spacing_m,
        "train_longitudinal_spacing_m": args.train_longitudinal_spacing_m,
        "dynamic_speed_scale": args.dynamic_speed_scale,
        "dynamic_amplitude_scale": args.dynamic_amplitude_scale,
        "enable_guidance_runtime": bool(args.enable_guidance_runtime),
        "branches": branches,
    }
    (output_dir / "continuation_summary.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

