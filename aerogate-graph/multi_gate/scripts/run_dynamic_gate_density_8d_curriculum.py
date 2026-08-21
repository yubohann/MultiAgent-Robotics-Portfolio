"""Continuation curriculum from demo8 formation morphing to dynamic gates.

This runner deliberately starts from the validated 8-drone demo8 checkpoint
that can execute line/triangle/rectangle/diamond/circle route morphing.  The
curriculum keeps C0 as a no-training preservation gate, then progressively
adds static and moving gates while making obstacle avoidance the primary task
and rejecting one/two-lane line-collapse as a hard formation-shape failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import sys
from typing import Any


def _bootstrap_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


@dataclass(frozen=True)
class DynamicGateCurriculumStage:
    name: str
    gate_count: int
    speed_mps: float
    amplitude_m: float
    train_steps: int
    goal_slot_tolerance_m: float
    slot_error_penalty_scale: float
    slot_improvement_scale: float
    max_slot_error_penalty_scale: float
    inter_agent_safe_distance_m: float
    shield_margin_m: float
    drone_speed_mps: float
    drone_accel_mps2: float
    notes: str


def _stages() -> tuple[DynamicGateCurriculumStage, ...]:
    return (
        DynamicGateCurriculumStage(
            "C0_demo8_route_no_gate_preserve",
            0,
            0.0,
            0.0,
            0,
            2.4,
            2.2,
            3.0,
            0.8,
            1.2,
            0.62,
            1.15,
            0.75,
            "No-gradient preservation gate: the demo8 checkpoint must still finish the full mixed-shape route.",
        ),
        DynamicGateCurriculumStage(
            "C0p1_micro_gate01_speed020_amp010_drone115",
            1,
            0.20,
            0.10,
            4096,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.15,
            0.75,
            "Training-only bridge: one slow moving gate with task-first three-lane pass slots before paper-axis gate counts.",
        ),
        DynamicGateCurriculumStage(
            "C0p2_micro_gate02_static_drone115",
            2,
            0.0,
            0.0,
            2048,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.15,
            0.75,
            "Bridge: 2 static gates. Scale up from 1-gate proficiency.",
        ),
        DynamicGateCurriculumStage(
            "C0p3_micro_gate03_static_drone115",
            3,
            0.0,
            0.0,
            2048,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.15,
            0.75,
            "Bridge: 3 static gates.",
        ),
        DynamicGateCurriculumStage(
            "C0p4_micro_gate04_static_drone115",
            4,
            0.0,
            0.0,
            3072,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.15,
            0.75,
            "Bridge: 4 static gates.",
        ),
        DynamicGateCurriculumStage(
            "C0p5_micro_gate05_static_drone115",
            5,
            0.0,
            0.0,
            3072,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.15,
            0.75,
            "Bridge: 5 static gates. Final bridge before 6-gate C1.",
        ),
        DynamicGateCurriculumStage(
            "C1_sparse_static_gate_task_first",
            6,
            0.0,
            0.0,
            4096,
            4.8,
            0.45,
            0.6,
            0.0,
            0.60,
            0.58,
            1.15,
            0.75,
            "First static gates. Hold the entry speed while obstacle contact and route progress become real.",
        ),
        DynamicGateCurriculumStage(
            "C1p5_micro_dynamic_gate08_speed020_amp010_drone115",
            8,
            0.20,
            0.10,
            3072,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.15,
            0.75,
            "Bridge: first moving gates (very slow) before C2. Gradual intro to gate motion.",
        ),
        DynamicGateCurriculumStage(
            "C2_sparse_slow_dynamic_gate",
            12,
            0.50,
            0.60,
            3072,
            5.2,
            0.35,
            0.5,
            0.0,
            0.60,
            0.60,
            1.45,
            1.00,
            "Slow moving gates. The team may deform while passing through openings.",
        ),
        DynamicGateCurriculumStage(
            "C3_low_density_dynamic_gate",
            18,
            0.80,
            0.75,
            4096,
            5.5,
            0.30,
            0.45,
            0.0,
            0.60,
            0.62,
            1.75,
            1.20,
            "Low-density dynamic gates. Learn active lateral avoidance instead of waiting.",
        ),
        DynamicGateCurriculumStage(
            "C4_medium_density_dynamic_gate",
            24,
            1.00,
            0.85,
            5120,
            5.8,
            0.25,
            0.40,
            0.0,
            0.60,
            0.64,
            2.05,
            1.40,
            "Medium density. Formation integrity becomes a secondary recovery metric.",
        ),
        DynamicGateCurriculumStage(
            "C5_medium_plus_dynamic_gate",
            30,
            1.10,
            0.90,
            6144,
            6.0,
            0.22,
            0.35,
            0.0,
            0.60,
            0.66,
            2.40,
            1.65,
            "Medium-plus density. Bridge gate count before returning to faster motion.",
        ),
        DynamicGateCurriculumStage(
            "C6_high_density_dynamic_gate",
            36,
            1.20,
            0.95,
            7168,
            6.2,
            0.20,
            0.30,
            0.0,
            0.60,
            0.68,
            2.75,
            1.90,
            "Stress scene. The paper curve should now show lower success and clearance.",
        ),
        DynamicGateCurriculumStage(
            "C7_high_density_fast_dynamic_gate",
            42,
            1.40,
            1.00,
            7680,
            6.3,
            0.19,
            0.28,
            0.0,
            0.60,
            0.69,
            3.10,
            2.15,
            "High density and faster moving gates. Success may begin to drop.",
        ),
        DynamicGateCurriculumStage(
            "C8_stress_dynamic_gate",
            48,
            1.60,
            1.05,
            8192,
            6.4,
            0.18,
            0.26,
            0.0,
            0.60,
            0.70,
            3.50,
            2.45,
            "Stress scene. The paper curve should now show lower success and clearance.",
        ),
        DynamicGateCurriculumStage(
            "C9_near_extreme_dynamic_gate",
            54,
            1.80,
            1.10,
            8704,
            6.5,
            0.18,
            0.25,
            0.0,
            0.60,
            0.70,
            3.50,
            2.45,
            "Near-extreme endpoint bridge before the validated 2 m/s cap.",
        ),
        DynamicGateCurriculumStage(
            "C10_extreme_dynamic_gate",
            60,
            2.00,
            1.20,
            9216,
            6.5,
            0.18,
            0.25,
            0.0,
            0.60,
            0.70,
            3.50,
            2.45,
            "Extreme gate-density endpoint for E2D-2/E2D-3: 60 moving gates at 2 m/s.",
        ),
    )


def _uses_dynamic_task_layout(stage: DynamicGateCurriculumStage) -> bool:
    """Use the straight dynamic-gate task layout even for paper eval's 0-gate baseline."""

    return int(stage.gate_count) > 0 or str(stage.name).startswith("E2D2_")


def _stage_config(base_config: Any, stage: DynamicGateCurriculumStage) -> Any:
    gate_cfg = replace(
        base_config.dynamic_gate_density,
        gate_count=int(stage.gate_count),
        moving_gate_speed_mps=float(stage.speed_mps),
        moving_gate_amplitude_m=float(stage.amplitude_m),
    )
    environment = replace(
        base_config.environment,
        max_command_speed_mps=float(stage.drone_speed_mps),
        max_command_forward_speed_mps=float(stage.drone_speed_mps),
        max_command_lateral_speed_mps=float(max(0.55, min(1.25, 0.50 * float(stage.drone_speed_mps)))),
        max_accel_mps2=float(stage.drone_accel_mps2),
        max_forward_accel_mps2=float(stage.drone_accel_mps2),
        max_lateral_accel_mps2=float(max(0.45, min(1.00, 0.55 * float(stage.drone_accel_mps2)))),
        inter_agent_safe_distance_m=float(stage.inter_agent_safe_distance_m),
        action_safety_shield_separation_margin_m=float(stage.shield_margin_m),
        slot_error_penalty_scale=float(stage.slot_error_penalty_scale),
        slot_improvement_scale=float(stage.slot_improvement_scale),
        max_slot_error_penalty_scale=float(stage.max_slot_error_penalty_scale),
        goal_proximity_reward_scale=2.5
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.goal_proximity_reward_scale,
        goal_proximity_sigma_m=45.0
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.goal_proximity_sigma_m,
        progress_reward_scale=14.0
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.progress_reward_scale,
        ungated_progress_reward_fraction=0.80
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.ungated_progress_reward_fraction,
        clearance_penalty_scale=1.5
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.clearance_penalty_scale,
        separation_penalty_scale=4.0
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.separation_penalty_scale,
        separation_proximity_penalty_scale=1.0
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.separation_proximity_penalty_scale,
        safety_clearance_m=0.55
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.safety_clearance_m,
        goal_requires_slot_tolerance=False
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.goal_requires_slot_tolerance,
        timeout_penalty=-80.0 if _uses_dynamic_task_layout(stage) else base_config.environment.timeout_penalty,
        timeout_goal_distance_penalty_scale=4.0
        if _uses_dynamic_task_layout(stage)
        else base_config.environment.timeout_goal_distance_penalty_scale,
    )
    formation = replace(
        base_config.formation,
        goal_slot_tolerance_m=float(stage.goal_slot_tolerance_m),
    )
    if _uses_dynamic_task_layout(stage):
        environment = replace(
            environment,
            start_y_range_m=(0.0, 0.0),
            goal_y_range_m=(0.0, 0.0),
            fixed_team_start_goal_y_m=((8, 0.0, 0.0),),
            path_waypoints_xy=((float(environment.start_x_m), 0.0), (float(environment.goal_x_m), 0.0)),
            max_episode_steps=1500,
            goal_radius_m=6.0,
            inter_agent_safe_distance_m=max(float(stage.inter_agent_safe_distance_m), 0.70),
            slot_anchor_blend=0.0,
            guidance_tracking_penalty_scale=5.0,
            guidance_escape_soft_margin_m=3.0,
            guidance_escape_penalty_scale=0.85,
            formation_line_collapse_terminal=False,
            action_safety_shield_separation_margin_m=max(float(stage.shield_margin_m), 1.20),
            action_safety_shield_brake_scale=1.75,
            action_safety_shield_pair_closing_brake_only=True,
            action_safety_shield_pair_time_horizon_s=0.35,
            action_safety_shield_repulsion_scale=5.00,
            action_safety_shield_outward_slot_bias_scale=1.15,
            max_lateral_accel_mps2=float(stage.drone_accel_mps2),
            action_safety_shield_priority_team_size_limit=9,
            action_safety_shield_boundary_margin_m=5.0,
            action_safety_shield_boundary_brake_scale=1.0,
            action_safety_shield_boundary_inward_scale=1.25,
            action_safety_shield_guidance_margin_m=3.5,
            action_safety_shield_guidance_inward_scale=2.2,
            action_safety_shield_gate_channel_enabled=True,
            action_safety_shield_gate_channel_lookahead_m=15.0,
            action_safety_shield_gate_channel_behind_m=5.5,
            action_safety_shield_gate_channel_lateral_gain=0.55,
            action_safety_shield_gate_channel_max_lateral_mps=1.15,
            action_safety_shield_gate_channel_slowdown_scale=0.36,
            action_safety_shield_post_gate_cruise_enabled=True,
            action_safety_shield_post_gate_cruise_min_forward_mps=min(0.78, float(stage.drone_speed_mps)),
            action_safety_shield_post_gate_cruise_gate_behind_m=7.0,
            action_safety_shield_post_gate_cruise_goal_margin_m=6.0,
            action_safety_shield_post_gate_cruise_min_pair_distance_m=1.55,
            action_safety_shield_post_gate_cruise_min_clearance_m=5.0,
            action_safety_shield_obstacle_margin_m=2.00,
            action_safety_shield_obstacle_brake_scale=1.35,
            action_safety_shield_obstacle_repulsion_scale=2.20,
            action_safety_shield_obstacle_time_horizon_s=1.00,
        )
        formation = replace(
            formation,
            lateral_spacing_m=1.9,
            longitudinal_spacing_m=2.4,
            bootstrap_shape_name=None,
            bootstrap_initial_shape_name=None,
            bootstrap_route_shape_names=(),
            bootstrap_route_slot_permutations=(),
            bootstrap_slot_permutation=(),
            bootstrap_morph_paths_xy=(),
            bootstrap_route_morph_paths_xy=(),
        )
    imitation = base_config.imitation
    dagger = base_config.dagger
    algorithm = base_config.algorithm
    if _uses_dynamic_task_layout(stage):
        algorithm = replace(
            algorithm,
            actor_lr=min(float(algorithm.actor_lr), 1.0e-5),
            critic_lr=min(float(algorithm.critic_lr), 3.0e-5),
            alpha_lr=min(float(algorithm.alpha_lr), 3.0e-6),
            init_alpha=min(float(algorithm.init_alpha), 0.01),
            batch_size=max(int(getattr(algorithm, "batch_size", 0)), 512),
            learning_starts=max(int(getattr(algorithm, "learning_starts", 0)), 4096),
            flash_update_interval=max(int(getattr(algorithm, "flash_update_interval", 1)), 8),
            target_value_clip=min(float(getattr(algorithm, "target_value_clip", 350.0)), 300.0),
            max_grad_norm=min(float(getattr(algorithm, "max_grad_norm", 10.0)), 8.0),
            behavior_anchor_loss_scale=max(float(getattr(algorithm, "behavior_anchor_loss_scale", 0.0)), 320.0),
            behavior_anchor_non_failure_only=False,
        )
        imitation = replace(
            imitation,
            max_steps_per_episode=max(int(imitation.max_steps_per_episode), int(environment.max_episode_steps)),
            batch_size=max(int(imitation.batch_size), 512),
            epochs=min(int(imitation.epochs), 3),
            learning_rate=min(float(imitation.learning_rate), 2.0e-5),
        )
        dagger = replace(
            dagger,
            max_steps_per_episode=max(int(dagger.max_steps_per_episode), int(environment.max_episode_steps)),
            bc_epochs_per_iteration=min(int(dagger.bc_epochs_per_iteration), 1),
        )
    return replace(
        base_config,
        dynamic_gate_density=gate_cfg,
        environment=environment,
        formation=formation,
        algorithm=algorithm,
        imitation=imitation,
        dagger=dagger,
        notes=f"{base_config.notes}\nCurriculum stage {stage.name}: {stage.notes}",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _eval_score(summary: dict[str, Any] | None) -> float:
    if not summary:
        return float("-inf")
    success_rate = float(summary.get("success_rate") or 0.0)
    gate_post_collision_rate = float(summary.get("gate_post_collision_rate") or 0.0)
    dynamic_gate_collision_rate = float(summary.get("dynamic_gate_collision_rate") or 0.0)
    agent_collision_rate = float(summary.get("agent_collision_rate") or 0.0)
    out_of_bounds_rate = float(summary.get("out_of_bounds_rate") or 0.0)
    timeout_rate = float(summary.get("timeout_rate") or 0.0)
    height_escape_failure_rate = float(summary.get("height_escape_failure_rate") or 0.0)
    side_bypass_failure_rate = float(summary.get("side_bypass_failure_rate") or 0.0)
    corridor_miss_failure_rate = float(summary.get("corridor_miss_failure_rate") or 0.0)
    goal_distance_m = float(summary.get("mean_goal_distance_m") or 0.0)
    slot_error_m = float(summary.get("mean_slot_error_m") or 0.0)
    clearance_m = float(summary.get("mean_min_clearance_m") or 0.0)
    safety_violation_rate = max(
        gate_post_collision_rate,
        dynamic_gate_collision_rate,
        agent_collision_rate,
        out_of_bounds_rate,
        height_escape_failure_rate,
        side_bypass_failure_rate,
        corridor_miss_failure_rate,
        float(summary.get("formation_line_collapse_failure_rate") or 0.0),
    )
    return float(
        success_rate * 1_000_000.0
        - safety_violation_rate * 2_000_000.0
        - timeout_rate * 50_000.0
        - goal_distance_m * 100.0
        - slot_error_m * 10.0
        + max(min(clearance_m, 5.0), -5.0) * 100.0
    )


def _promotion_gate_failure(summary: dict[str, Any] | None, config: Any, stage: DynamicGateCurriculumStage | None = None) -> str | None:
    if not summary:
        return "missing_post_train_eval"
    gate = config.evaluation_gate
    if not bool(getattr(gate, "enabled", False)):
        return None
    # Use graduated thresholds based on gate count: harder for low gates, relaxed for high gates
    gc = int(stage.gate_count) if stage is not None else 0
    if gc <= 1:
        min_success, max_gate_post = 1.0, 0.0
    elif gc <= 2:
        min_success, max_gate_post = 0.50, 0.50
    elif gc <= 4:
        min_success, max_gate_post = 0.20, 0.80
    elif gc <= 8:
        min_success, max_gate_post = 0.05, 0.95
    else:
        min_success, max_gate_post = 0.0, 1.0
    min_requirements = {
        "success_rate": min_success,
        "team_success_rate": 0.0,
        "per_agent_success_rate": 0.0,
        "height_contract_passed_rate": 1.0,
        "corridor_through_success_rate": 0.0,
    }
    for key, required in min_requirements.items():
        value = float(summary.get(key) or 0.0)
        if value < required:
            return f"{key} {value:.3f} < required {required:.3f}"

    max_requirements = {
        "gate_post_collision_rate": max_gate_post,
        "obstacle_collision_rate": max_gate_post,
        "dynamic_gate_collision_rate": max_gate_post,
        "agent_collision_rate": 1.0,
        "out_of_bounds_rate": 0.0,
        "timeout_rate": 1.0,
        "hard_failure_rate": 1.0,
        "safety_violation_rate": 1.0,
        "height_escape_failure_rate": 0.0,
        "side_bypass_failure_rate": 0.0,
        "corridor_miss_failure_rate": 0.0,
        "formation_line_collapse_failure_rate": 1.0,
        "dispersed_termination_rate": 1.0,
    }
    for key, allowed in max_requirements.items():
        value = float(summary.get(key) or 0.0)
        if value > allowed:
            return f"{key} {value:.3f} > allowed {allowed:.3f}"
    return None


def _hard_metric_regression_failure(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> str | None:
    if not before or not after:
        return "missing_eval_for_hard_metric_comparison"
    eps = 1e-9

    # Allow up to 20% regression margin on success metrics for intermediate stages
    regression_margin = 0.20
    min_keys = (
        "success_rate",
        "team_success_rate",
        "per_agent_success_rate",
        "corridor_through_success_rate",
        "height_contract_passed_rate",
    )
    for key in min_keys:
        before_value = float(before.get(key) or 0.0)
        after_value = float(after.get(key) or 0.0)
        if after_value + regression_margin + eps < before_value:
            return f"{key} regressed {after_value:.3f} < {before_value:.3f} (margin={regression_margin})"

    max_keys = (
        "gate_post_collision_rate",
        "obstacle_collision_rate",
        "dynamic_gate_collision_rate",
        "agent_collision_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "hard_failure_rate",
        "safety_violation_rate",
        "height_escape_failure_rate",
        "side_bypass_failure_rate",
        "corridor_miss_failure_rate",
        "formation_line_collapse_failure_rate",
        "dispersed_termination_rate",
    )
    for key in max_keys:
        before_value = float(before.get(key) or 0.0)
        after_value = float(after.get(key) or 0.0)
        if after_value > before_value + regression_margin + eps:
            return f"{key} regressed {after_value:.3f} > {before_value:.3f}"

    return None


def _promotion_thresholds(config: Any) -> dict[str, float | None]:
    return {
        "min_success_rate": 1.0,
        "min_team_success_rate": 1.0,
        "min_per_agent_success_rate": 1.0,
        "min_height_contract_passed_rate": 1.0,
        "min_corridor_through_success_rate": 1.0,
        "max_gate_post_collision_rate": 0.0,
        "max_obstacle_collision_rate": 0.0,
        "max_dynamic_gate_collision_rate": 0.0,
        "max_agent_collision_rate": 0.0,
        "max_out_of_bounds_rate": 0.0,
        "max_timeout_rate": 0.0,
        "max_hard_failure_rate": 0.0,
        "max_safety_violation_rate": 0.0,
        "max_height_escape_failure_rate": 0.0,
        "max_side_bypass_failure_rate": 0.0,
        "max_corridor_miss_failure_rate": 0.0,
        "max_formation_line_collapse_failure_rate": 0.0,
        "max_dispersed_termination_rate": 0.0,
    }


def _failure_stop_thresholds(args: argparse.Namespace) -> dict[str, float | None]:
    return {
        "max_success_rate": float(args.failure_stop_max_success_rate),
        "min_hard_failure_rate": float(args.failure_stop_min_safety_violation_rate),
        "min_safety_violation_rate": float(args.failure_stop_min_safety_violation_rate),
    }


def main() -> None:
    _bootstrap_imports()
    from multi_gate.configs import get_multi_experiment_config
    from multi_gate.dagger import run_dagger_warmstart_then_finetune
    from multi_gate.imitation import run_bc_warmstart_then_finetune
    from multi_gate.sanity import assert_dynamic_gate_density_environment_sane
    from multi_gate.training import evaluate_checkpoint, run_training
    from shared.core.dynamic_gate_density_2d import MAX_DRONE_COMMAND_ACCEL_MPS2, MAX_DRONE_COMMAND_SPEED_MPS

    default_checkpoint = (
        Path(__file__).resolve().parents[2]
        / ".."
        / "rt8"
        / "paper_runs"
        / "exp3_demo8_branch_from_stage02d_20260426_stage00"
        / "stages"
        / "35_demo8_35_full_route_mixed_isaaclab_render"
        / "checkpoints"
        / "latest_agent.pt"
    ).resolve()
    env_checkpoint = os.environ.get("GATE2D_RESUME_CHECKPOINT")
    if env_checkpoint:
        default_checkpoint = Path(env_checkpoint)

    parser = argparse.ArgumentParser(description="Run the demo8 -> dynamic-gate 8-drone curriculum.")
    parser.add_argument("--resume-checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--start-stage-index", type=int, default=0)
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", help="Legacy alias for the training device.")
    parser.add_argument("--train-device", type=str, default=None)
    parser.add_argument("--eval-device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--selection-eval-episodes", type=int, default=3)
    parser.add_argument("--pre-train-eval-episodes", type=int, default=1)
    parser.add_argument("--post-train-eval-episodes", type=int, default=None)
    parser.add_argument("--periodic-eval-episodes", type=int, default=2)
    parser.add_argument("--checkpoint-interval-transitions", type=int, default=2048)
    parser.add_argument("--quick-eval-interval-transitions", type=int, default=2048)
    parser.add_argument("--pass-window", type=int, default=2)
    parser.add_argument("--early-stop-min-transitions", type=int, default=2048)
    parser.add_argument("--failure-stop-window", type=int, default=2)
    parser.add_argument("--failure-stop-min-transitions", type=int, default=4096)
    parser.add_argument("--failure-stop-max-success-rate", type=float, default=0.0)
    parser.add_argument("--failure-stop-min-safety-violation-rate", type=float, default=0.75)
    parser.add_argument("--actor-gate-eval-episodes", type=int, default=3)
    parser.add_argument(
        "--actor-gate-eval-seed",
        type=int,
        default=None,
        help="Optional explicit seed for the BC/DAgger actor gate eval.",
    )
    parser.add_argument("--force-rl-after-actor-gate-pass", action="store_true")
    parser.add_argument("--learning-starts", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=128)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run scene sanity checks for the selected stages, then exit without checkpoint eval or training.",
    )
    parser.add_argument("--no-bc", action="store_true", help="Ablation mode: skip BC and DAgger, then RL fine-tune only.")
    parser.add_argument("--no-dagger", action="store_true", help="Ablation mode: run BC warm start, then RL fine-tune without DAgger.")
    parser.add_argument("--bc-expert-episodes", type=int, default=None)
    parser.add_argument("--bc-target-retained-episodes", type=int, default=None)
    parser.add_argument("--bc-collection-workers", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)
    parser.add_argument("--bc-max-steps-per-episode", type=int, default=None)
    parser.add_argument("--dagger-iterations", type=int, default=None)
    parser.add_argument("--dagger-rollout-episodes", type=int, default=None)
    parser.add_argument("--dagger-bc-epochs", type=int, default=None)
    parser.add_argument("--dagger-bc-batch-size", type=int, default=None)
    parser.add_argument(
        "--refresh-initial-bc",
        action="store_true",
        help="Rebuild the initial BC actor before DAgger. Off by default for continuation chunks.",
    )
    parser.add_argument(
        "--use-resume-checkpoint-as-expert",
        action="store_true",
        help=(
            "Use the selected resume checkpoint as the BC/DAgger teacher for bridge stages. "
            "This is intended for cases where the heuristic expert fails the audit but the "
            "incoming checkpoint already has clean full-route rollouts."
        ),
    )
    parser.add_argument(
        "--skip-scene-sanity",
        action="store_true",
        help="Developer escape hatch only: skip dynamic gate motion/progress/collision sanity checks.",
    )
    parser.add_argument(
        "--override-stage-name",
        type=str,
        default=None,
        help="One-stage bridge name override. Requires --stage-limit 1.",
    )
    parser.add_argument(
        "--override-stage-gate-count",
        type=int,
        default=None,
        help="One-stage bridge gate-count override. Requires --stage-limit 1.",
    )
    parser.add_argument(
        "--override-stage-speed-mps",
        type=float,
        default=None,
        help="One-stage bridge speed override in m/s. Requires --stage-limit 1.",
    )
    parser.add_argument(
        "--override-stage-amplitude-m",
        type=float,
        default=None,
        help="One-stage bridge amplitude override in meters. Requires --stage-limit 1.",
    )
    parser.add_argument(
        "--override-stage-drone-speed-mps",
        type=float,
        default=None,
        help="One-stage bridge drone forward speed limit override in m/s. Requires --stage-limit 1.",
    )
    parser.add_argument(
        "--override-stage-drone-accel-mps2",
        type=float,
        default=None,
        help="One-stage bridge drone acceleration limit override in m/s^2. Requires --stage-limit 1.",
    )
    parser.add_argument(
        "--override-stage-train-steps",
        type=int,
        default=None,
        help="One-stage bridge train-step override. Requires --stage-limit 1.",
    )
    args = parser.parse_args()
    if bool(args.no_bc):
        args.no_dagger = True
    if int(args.num_agents) < 2 or int(args.num_agents) > 9:
        raise SystemExit("--num-agents must be in the formal multi-agent range [2, 9]")
    if args.override_stage_gate_count is not None and (
        int(args.override_stage_gate_count) < 0 or int(args.override_stage_gate_count) > 60
    ):
        raise SystemExit("--override-stage-gate-count must be in [0, 60]")
    if args.override_stage_drone_speed_mps is not None and (
        float(args.override_stage_drone_speed_mps) <= 0.0
        or float(args.override_stage_drone_speed_mps) > MAX_DRONE_COMMAND_SPEED_MPS
    ):
        raise SystemExit(f"--override-stage-drone-speed-mps must be in (0, {MAX_DRONE_COMMAND_SPEED_MPS}]")
    if args.override_stage_drone_accel_mps2 is not None and (
        float(args.override_stage_drone_accel_mps2) <= 0.0
        or float(args.override_stage_drone_accel_mps2) > MAX_DRONE_COMMAND_ACCEL_MPS2
    ):
        raise SystemExit(f"--override-stage-drone-accel-mps2 must be in (0, {MAX_DRONE_COMMAND_ACCEL_MPS2}]")
    train_device = str(args.train_device or args.device or "cuda")
    eval_device = str(args.eval_device or "cpu")

    root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "runtime" / "dynamic_gate_density_8d_from_demo8_curriculum"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "curriculum_summary.json"

    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    if args.post_train_eval_episodes is None:
        args.post_train_eval_episodes = int(base_config.evaluation_gate.eval_episodes)
    all_stages = _stages()
    start_idx = max(0, int(args.start_stage_index))
    stop_idx = len(all_stages) if args.stage_limit is None else min(len(all_stages), start_idx + max(0, int(args.stage_limit)))
    stages = all_stages[start_idx:stop_idx]
    has_stage_override = any(
        value is not None
        for value in (
            args.override_stage_name,
            args.override_stage_gate_count,
            args.override_stage_speed_mps,
            args.override_stage_amplitude_m,
            args.override_stage_drone_speed_mps,
            args.override_stage_drone_accel_mps2,
            args.override_stage_train_steps,
        )
    )
    if has_stage_override:
        if len(stages) != 1:
            raise ValueError("Stage overrides are only allowed with exactly one selected stage.")
        stage = stages[0]
        matched_stage_index = None
        if args.override_stage_name:
            override_name = str(args.override_stage_name)
            for candidate_index, candidate_stage in enumerate(all_stages):
                if str(candidate_stage.name) == override_name:
                    stage = candidate_stage
                    matched_stage_index = int(candidate_index)
                    break
        if matched_stage_index is not None:
            start_idx = int(matched_stage_index)
        stages = (
            replace(
                stage,
                name=str(args.override_stage_name or stage.name),
                gate_count=(
                    int(args.override_stage_gate_count)
                    if args.override_stage_gate_count is not None
                    else int(stage.gate_count)
                ),
                speed_mps=(
                    float(args.override_stage_speed_mps)
                    if args.override_stage_speed_mps is not None
                    else float(stage.speed_mps)
                ),
                amplitude_m=(
                    float(args.override_stage_amplitude_m)
                    if args.override_stage_amplitude_m is not None
                    else float(stage.amplitude_m)
                ),
                drone_speed_mps=(
                    float(args.override_stage_drone_speed_mps)
                    if args.override_stage_drone_speed_mps is not None
                    else float(stage.drone_speed_mps)
                ),
                drone_accel_mps2=(
                    float(args.override_stage_drone_accel_mps2)
                    if args.override_stage_drone_accel_mps2 is not None
                    else float(stage.drone_accel_mps2)
                ),
                train_steps=(
                    int(args.override_stage_train_steps)
                    if args.override_stage_train_steps is not None
                    else int(stage.train_steps)
                ),
            ),
        )
    checkpoint_path = Path(args.resume_checkpoint)
    records: list[dict[str, Any]] = []

    for absolute_index, stage in enumerate(stages, start=start_idx):
        stage_dir = output_dir / "stages" / f"{absolute_index:02d}_{stage.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        config = _stage_config(base_config, stage)
        stage_input_checkpoint = Path(str(checkpoint_path))
        sanity_report = None
        if not bool(args.skip_scene_sanity):
            sanity_report = assert_dynamic_gate_density_environment_sane(
                experiment_config=config,
                seed=int(args.seed) + absolute_index * 100,
                num_agents=int(args.num_agents),
            )
            (stage_dir / "scene_sanity.json").write_text(
                json.dumps(_json_safe(sanity_report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if bool(args.sanity_only):
            record = {
                "stage_index": absolute_index,
                "stage": asdict(stage),
                "input_checkpoint": str(checkpoint_path),
                "scene_sanity": sanity_report,
                "pre_train_eval": None,
                "train_summary": None,
                "post_train_eval": None,
                "output_checkpoint": str(checkpoint_path),
            }
            records.append(record)
            summary_path.write_text(
                json.dumps(
                    {
                        "resume_checkpoint": str(args.resume_checkpoint),
                        "current_checkpoint": str(checkpoint_path),
                        "sanity_only": True,
                        "records": _json_safe(records),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[{absolute_index:02d}] {stage.name} "
                f"gate={stage.gate_count} gate_speed={stage.speed_mps:.2f} "
                f"drone_speed={stage.drone_speed_mps:.2f} scene_sanity=pass",
                flush=True,
            )
            continue
        eval_summary = None
        if int(args.pre_train_eval_episodes) > 0:
            eval_summary = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=int(args.pre_train_eval_episodes),
                seed=int(args.seed) + absolute_index * 100,
                device=eval_device,
                num_agents=int(args.num_agents),
                experiment_config=config,
            )
        record: dict[str, Any] = {
            "stage_index": absolute_index,
            "stage": asdict(stage),
            "input_checkpoint": str(checkpoint_path),
            "scene_sanity": sanity_report,
            "pre_train_eval": eval_summary,
        }
        if int(stage.train_steps) > 0 and not bool(args.eval_only):
            train_steps = max(1, int(round(int(stage.train_steps) * float(args.steps_scale))))
            checkpoint_interval_transitions = max(int(args.checkpoint_interval_transitions), int(args.num_envs))
            quick_eval_interval_transitions = max(int(args.quick_eval_interval_transitions), int(args.num_envs))
            promotion_thresholds = _promotion_thresholds(config)
            failure_thresholds = _failure_stop_thresholds(args)
            planned_total_transitions = int(train_steps) * max(int(args.num_envs), 1)
            expert_teacher_checkpoint = str(checkpoint_path) if bool(args.use_resume_checkpoint_as_expert) else None
            if bool(args.no_bc):
                train_pipeline = "rl_only_no_bc_no_dagger"
                train_summary = run_training(
                    train_steps=train_steps,
                    num_envs=int(args.num_envs),
                    seed=int(args.seed) + absolute_index * 1000,
                    device=train_device,
                    save_dir=str(stage_dir),
                    log_dir=stage_dir / "logs",
                    checkpoint_dir=stage_dir / "checkpoints",
                    num_agents=int(args.num_agents),
                    learning_starts=int(args.learning_starts),
                    batch_size=int(args.batch_size),
                    updates_per_step=int(args.updates_per_step),
                    log_every=int(args.log_every),
                    experiment_config=config,
                    resume_checkpoint=str(checkpoint_path),
                    resume_mode="reset_train_state" if len(records) == 0 else "keep_optimizer_state",
                    checkpoint_interval_steps=checkpoint_interval_transitions,
                    selection_eval_episodes=max(1, int(args.selection_eval_episodes)),
                    periodic_eval_episodes=max(1, int(args.periodic_eval_episodes)),
                    periodic_eval_interval_steps=quick_eval_interval_transitions,
                    early_stop_eval_thresholds=promotion_thresholds,
                    early_stop_min_transitions=int(args.early_stop_min_transitions),
                    early_stop_stable_window_min_length=max(1, int(args.pass_window)),
                    early_stop_planned_total_transitions=planned_total_transitions,
                    failure_stop_eval_thresholds=failure_thresholds,
                    failure_stop_min_transitions=int(args.failure_stop_min_transitions),
                    failure_stop_stable_window_min_length=max(1, int(args.failure_stop_window)),
                )
                fine_tuning_summary = train_summary
            elif bool(args.no_dagger):
                train_pipeline = "bc_then_rl_no_dagger"
                train_summary = run_bc_warmstart_then_finetune(
                    experiment_config=config,
                    train_steps=train_steps,
                    num_envs=int(args.num_envs),
                    seed=int(args.seed) + absolute_index * 1000,
                    device=train_device,
                    num_agents=int(args.num_agents),
                    initial_actor_checkpoint=str(checkpoint_path),
                    expert_teacher_actor_checkpoint=expert_teacher_checkpoint,
                    expert_episodes=args.bc_expert_episodes,
                    expert_target_retained_episodes=args.bc_target_retained_episodes,
                    expert_collection_workers=int(args.bc_collection_workers),
                    max_steps_per_episode=args.bc_max_steps_per_episode,
                    bc_epochs=args.bc_epochs,
                    bc_batch_size=args.bc_batch_size,
                    bc_output_dir=stage_dir / "bc",
                    save_dir=str(stage_dir),
                    log_dir=stage_dir / "logs",
                    checkpoint_dir=stage_dir / "checkpoints",
                    learning_starts=int(args.learning_starts),
                    batch_size=int(args.batch_size),
                    updates_per_step=int(args.updates_per_step),
                    log_every=int(args.log_every),
                    checkpoint_interval_steps=checkpoint_interval_transitions,
                    selection_eval_episodes=max(1, int(args.selection_eval_episodes)),
                    periodic_eval_episodes=max(1, int(args.periodic_eval_episodes)),
                    periodic_eval_interval_steps=quick_eval_interval_transitions,
                    early_stop_eval_thresholds=promotion_thresholds,
                    early_stop_min_transitions=int(args.early_stop_min_transitions),
                    early_stop_stable_window_min_length=max(1, int(args.pass_window)),
                    early_stop_planned_total_transitions=planned_total_transitions,
                    failure_stop_eval_thresholds=failure_thresholds,
                    failure_stop_min_transitions=int(args.failure_stop_min_transitions),
                    failure_stop_stable_window_min_length=max(1, int(args.failure_stop_window)),
                    actor_gate_eval_episodes=max(0, int(args.actor_gate_eval_episodes)),
                    actor_gate_eval_seed=(
                        int(args.actor_gate_eval_seed)
                        if args.actor_gate_eval_seed is not None
                        else int(args.seed) + absolute_index * 100 + 17
                    ),
                    actor_gate_thresholds=promotion_thresholds,
                )
                fine_tuning_summary = dict(train_summary.get("fine_tuning") or {})
            else:
                train_pipeline = (
                    "bc_dagger_then_rl"
                    if bool(args.refresh_initial_bc)
                    else "dagger_bc_corrections_then_rl_no_initial_bc_refresh"
                )
                train_summary = run_dagger_warmstart_then_finetune(
                    experiment_config=config,
                    train_steps=train_steps,
                    num_envs=int(args.num_envs),
                    seed=int(args.seed) + absolute_index * 1000,
                    device=train_device,
                    num_agents=int(args.num_agents),
                    initial_actor_checkpoint=str(checkpoint_path),
                    expert_teacher_actor_checkpoint=expert_teacher_checkpoint,
                    expert_episodes=args.bc_expert_episodes,
                    expert_target_retained_episodes=args.bc_target_retained_episodes,
                    expert_collection_workers=int(args.bc_collection_workers),
                    max_steps_per_episode=args.bc_max_steps_per_episode,
                    initial_bc_epochs=args.bc_epochs,
                    initial_bc_batch_size=args.bc_batch_size,
                    refresh_initial_bc=bool(args.refresh_initial_bc),
                    dagger_iterations=args.dagger_iterations,
                    dagger_rollout_episodes=args.dagger_rollout_episodes,
                    dagger_bc_epochs=args.dagger_bc_epochs,
                    dagger_bc_batch_size=args.dagger_bc_batch_size,
                    output_dir=stage_dir / "dagger",
                    save_dir=str(stage_dir),
                    log_dir=stage_dir / "logs",
                    checkpoint_dir=stage_dir / "checkpoints",
                    learning_starts=int(args.learning_starts),
                    batch_size=int(args.batch_size),
                    updates_per_step=int(args.updates_per_step),
                    log_every=int(args.log_every),
                    checkpoint_interval_steps=checkpoint_interval_transitions,
                    selection_eval_episodes=max(1, int(args.selection_eval_episodes)),
                    periodic_eval_episodes=max(1, int(args.periodic_eval_episodes)),
                    periodic_eval_interval_steps=quick_eval_interval_transitions,
                    early_stop_eval_thresholds=promotion_thresholds,
                    early_stop_min_transitions=int(args.early_stop_min_transitions),
                    early_stop_stable_window_min_length=max(1, int(args.pass_window)),
                    early_stop_planned_total_transitions=planned_total_transitions,
                    failure_stop_eval_thresholds=failure_thresholds,
                    failure_stop_min_transitions=int(args.failure_stop_min_transitions),
                    failure_stop_stable_window_min_length=max(1, int(args.failure_stop_window)),
                    actor_gate_eval_episodes=max(0, int(args.actor_gate_eval_episodes)),
                    actor_gate_eval_seed=(
                        int(args.actor_gate_eval_seed)
                        if args.actor_gate_eval_seed is not None
                        else int(args.seed) + absolute_index * 100 + 17
                    ),
                    actor_gate_thresholds=promotion_thresholds,
                    skip_rl_after_actor_gate_pass=not bool(args.force_rl_after_actor_gate_pass),
                )
                fine_tuning_summary = dict(train_summary.get("fine_tuning") or {})
            record["train_pipeline"] = train_pipeline
            record["train_summary"] = train_summary
            next_checkpoint = (
                fine_tuning_summary.get("best_alias_path")
                or fine_tuning_summary.get("best_checkpoint_path")
                or fine_tuning_summary.get("checkpoint_path")
            )
            actor_gate_failed_without_full_checkpoint = (
                bool(fine_tuning_summary.get("skipped", False))
                and str(fine_tuning_summary.get("skip_reason") or "") == "actor_gate_failed"
                and not next_checkpoint
            )
            if next_checkpoint:
                checkpoint_path = Path(str(next_checkpoint))
            trained_checkpoint_path = Path(str(checkpoint_path))
            if int(args.post_train_eval_episodes) > 0 and not actor_gate_failed_without_full_checkpoint:
                record["post_train_eval"] = evaluate_checkpoint(
                    checkpoint_path=checkpoint_path,
                    episodes=int(args.post_train_eval_episodes),
                    seed=int(args.seed) + absolute_index * 100 + 33,
                    device=eval_device,
                    num_agents=int(args.num_agents),
                    experiment_config=config,
                )
                pre_score = _eval_score(eval_summary)
                post_score = _eval_score(record["post_train_eval"])
                record["checkpoint_selection"] = {
                    "pre_score": float(pre_score),
                    "post_score": float(post_score),
                    "trained_checkpoint_path": str(trained_checkpoint_path),
                }
                promotion_failure = _promotion_gate_failure(record["post_train_eval"], config, stage)
                hard_regression_failure = _hard_metric_regression_failure(eval_summary, record["post_train_eval"])
                if promotion_failure:
                    checkpoint_path = stage_input_checkpoint
                    record["checkpoint_selection"]["selected"] = "input_checkpoint"
                    record["checkpoint_selection"]["reason"] = f"promotion_gate_failed: {promotion_failure}"
                elif hard_regression_failure:
                    checkpoint_path = stage_input_checkpoint
                    record["checkpoint_selection"]["selected"] = "input_checkpoint"
                    record["checkpoint_selection"]["reason"] = f"hard_metric_regression: {hard_regression_failure}"
                else:
                    record["checkpoint_selection"]["selected"] = "trained_checkpoint"
                    if post_score < pre_score:
                        record["checkpoint_selection"]["reason"] = "soft_score_regression_hard_metrics_clean"
            else:
                record["post_train_eval"] = None
                record["checkpoint_selection"] = {
                    "trained_checkpoint_path": str(trained_checkpoint_path),
                }
                if actor_gate_failed_without_full_checkpoint:
                    checkpoint_path = stage_input_checkpoint
                    record["checkpoint_selection"]["selected"] = "input_checkpoint"
                    record["checkpoint_selection"]["reason"] = "actor_gate_failed_no_full_checkpoint"
                elif bool(fine_tuning_summary.get("early_stop_triggered", False)):
                    record["checkpoint_selection"]["selected"] = "trained_checkpoint"
                    record["checkpoint_selection"]["reason"] = "early_stop_gate_passed_without_post_eval"
                else:
                    checkpoint_path = stage_input_checkpoint
                    record["checkpoint_selection"]["selected"] = "input_checkpoint"
                    record["checkpoint_selection"]["reason"] = "no_post_eval_and_no_early_stop_gate_pass"
        else:
            record["train_summary"] = None
            record["post_train_eval"] = None
        record["output_checkpoint"] = str(checkpoint_path)
        records.append(record)
        summary_path.write_text(
            json.dumps(
                {
                    "resume_checkpoint": str(args.resume_checkpoint),
                    "current_checkpoint": str(checkpoint_path),
                    "records": _json_safe(records),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{absolute_index:02d}] {stage.name} "
            f"gate={stage.gate_count} speed={stage.speed_mps:.2f} "
            f"pre_success={None if eval_summary is None else eval_summary.get('success_rate')} next={checkpoint_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

