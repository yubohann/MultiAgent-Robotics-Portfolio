"""Run single-drone gate-density scoring across gate counts and yaw seeds."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from gate_density_single.core.episode_runner import run_episode
from gate_density_single.core.eval_support import (
    DRONE_RADIUS_M,
    GOAL_RADIUS_M,
    MAX_EPISODE_STEPS,
    MAX_GATE_COUNT,
    SAFETY_MARGIN_M,
    build_gate_obstacle_map,
    clamp01,
    percentile,
    write_json,
)
from gate_density_single.core.gate_layout import (
    ALLOWED_GATE_LAYOUT_VERSIONS,
    GATE_BOTTOM_HEIGHT_M,
    GATE_CENTER_HEIGHT_M,
    GATE_GATE_CLEARANCE_MARGIN_M,
    GATE_GATE_FRAME_CLEARANCE_MARGIN_M,
    GATE_HALF_WIDTH_M,
    GATE_LAYOUT_VERSION,
    GATE_NATIVE_VISUAL_HEIGHT_M,
    GATE_POST_RADIUS_M,
    GATE_TOP_HEIGHT_M,
    GATE_VISUAL_FRAME_HALF_DEPTH_M,
    GATE_VISUAL_SCALE_Z,
    MAX_MOVING_GATE_SPEED_MPS,
    _gate_gate_clearance_stats,
    _gate_gate_frame_clearance_stats,
    _generate_gate_layout,
    _layout_profile,
    _resolve_moving_gate_speed_hz,
)
from gate_density_single.core.guidance_client import LocalGateGuidanceClient
from shared.runtime.paths import ensure_project_on_path

ALLOWED_GATE_COUNTS = tuple(range(0, MAX_GATE_COUNT + 1))
ALLOWED_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
DEFAULT_GUIDANCE_PROVIDER = "local_http"
DEFAULT_GUIDANCE_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_GUIDANCE_MODEL = "local-guidance-model"
DEFAULT_GUIDANCE_PROMPT_VERSION = "gate_density_single_v1"


def _density_adaptive_controller_profile(gate_count: int) -> dict[str, float | int]:
    """Return the adaptive controller profile for 24-60 gates."""

    count = max(int(gate_count), 0)
    density_24_to_30 = clamp01((float(count) - 24.0) / 6.0)
    density_30_to_60 = clamp01((float(count) - 30.0) / 30.0)
    interval = int(round(12.0 - 4.0 * density_24_to_30 - 4.0 * density_30_to_60))
    threshold = 0.45 + 0.10 * density_24_to_30 + 0.05 * density_30_to_60
    speed_base = 0.70 + 0.15 * density_24_to_30 + 0.20 * density_30_to_60
    speed_gain = 0.75 + 0.15 * density_24_to_30 + 0.15 * density_30_to_60
    shield_rollout = int(round(6.0 + 2.0 * density_24_to_30))
    inflation = 0.40 + 0.05 * density_24_to_30 - 0.05 * density_30_to_60
    final_bias_start = 22.5 - 5.5 * density_30_to_60 if count > 24 else 0.0
    final_bias_strength = 0.35 * density_24_to_30 + 0.25 * density_30_to_60 if count > 24 else 0.0
    if count > 30:
        final_bias_start = 9.0 - 2.0 * density_30_to_60
        final_bias_strength = max(final_bias_strength, 0.42 + 0.10 * density_30_to_60)
    return {
        "dynamic_replan_interval_steps": max(4, interval),
        "dynamic_replan_clearance_threshold_m": float(min(threshold, 0.60)),
        "dynamic_gate_speed_cap_base_mps": float(min(speed_base, 1.05)),
        "dynamic_gate_speed_cap_gain": float(min(speed_gain, 1.05)),
        "dynamic_shield_rollout_steps": max(6, shield_rollout),
        "dynamic_planner_inflation_extra_m": float(max(inflation, 0.40)),
        "dynamic_final_goal_bias_start_x_m": (
            float(final_bias_start) if count > 30 else (float(max(final_bias_start, 17.0)) if count > 24 else 0.0)
        ),
        "dynamic_final_goal_bias_strength": float(min(final_bias_strength, 0.60)),
    }


def _bootstrap_imports() -> None:
    ensure_project_on_path()


def _summarize_episode_metrics(metrics_list: list[dict[str, Any]], *, safety_shield_enabled: bool = True) -> dict[str, Any]:
    count = max(len(metrics_list), 1)
    success_rate = sum(1 for item in metrics_list if item["success"]) / count
    collision_rate = sum(1 for item in metrics_list if item["collision"]) / count
    out_of_bounds_rate = sum(1 for item in metrics_list if item["out_of_bounds"]) / count
    timeout_rate = sum(1 for item in metrics_list if item["timeout"]) / count
    numeric_fields = [
        "episode_reward",
        "path_length_m",
        "flight_time_s",
        "min_clearance_m",
        "mean_clearance_m",
        "mean_speed_mps",
        "max_speed_mps",
        "mean_goal_tracking_error_m",
        "max_goal_tracking_error_m",
        "initial_goal_distance_m",
        "final_goal_distance_m",
        "progress_distance_m",
        "guidance_tracking_error_mean_m",
        "guidance_tracking_error_max_m",
        "trajectory_smoothness",
        "planner_call_count",
        "planner_failure_count",
        "planner_latency_ms_mean",
        "planner_latency_ms_p95",
        "global_planner_trigger_count",
        "global_planner_latency_ms_mean",
        "global_planner_latency_ms_p95",
        "guidance_query_count",
        "guidance_success_count",
        "guidance_failure_count",
        "guidance_fallback_count",
        "guidance_cache_hit_count",
        "route_guidance_used_count",
        "guidance_latency_ms_mean",
        "guidance_latency_ms_p95",
        "guidance_cache_hit_rate",
        "guidance_non_fallback_rate",
        "route_guidance_tracking_error_m",
        "guidance_replan_urgency",
        "guidance_waypoint_bias_y",
        "guidance_dynamic_clearance_margin_m",
        "shield_activation_count",
        "shield_activation_ratio",
        "gate_gate_min_clearance_m",
        "gate_gate_overlap_pair_count_max",
        "gate_gate_frame_min_clearance_m",
        "gate_gate_frame_overlap_pair_count_max",
        "moving_gate_swept_clearance_m_min",
        "dynamic_swept_collision_count",
        "actual_gate_motion_range_m",
        "actual_gate_motion_range_x_m",
        "actual_gate_motion_range_y_m",
        "actual_gate_motion_range_mean_m",
        "actual_gate_max_displacement_m",
        "height_contract_passed",
        "corridor_through_success",
        "side_bypass_failure",
        "height_escape_failure",
        "corridor_miss_failure",
        "drone_top_clearance_to_gate_top_m",
    ]
    summary: dict[str, Any] = {
        "episodes": len(metrics_list),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "out_of_bounds_rate": float(out_of_bounds_rate),
        "timeout_rate": float(timeout_rate),
        "height_contract_passed_rate": float(
            sum(1 for item in metrics_list if item.get("height_contract_passed") is True) / count
        ),
        "corridor_through_success_rate": float(
            sum(1 for item in metrics_list if item.get("corridor_through_success") is True) / count
        ),
        "side_bypass_failure_rate": float(
            sum(1 for item in metrics_list if item.get("side_bypass_failure") is True) / count
        ),
        "height_escape_failure_rate": float(
            sum(1 for item in metrics_list if item.get("height_escape_failure") is True) / count
        ),
        "corridor_miss_failure_rate": float(
            sum(1 for item in metrics_list if item.get("corridor_miss_failure") is True) / count
        ),
        "bucket_success": {"all": float(success_rate)},
    }
    for field in numeric_fields:
        values = [float(item[field]) for item in metrics_list if item.get(field) is not None and math.isfinite(float(item[field]))]
        summary[f"{field}_mean"] = float(np.mean(values) if values else 0.0)
        summary[f"{field}_p95"] = percentile(values, 95.0)
    shield_values = [
        float(item["shield_activation_ratio"])
        for item in metrics_list
        if item.get("shield_activation_ratio") is not None and math.isfinite(float(item["shield_activation_ratio"]))
    ]
    summary["shield_activation_ratio"] = float(np.mean(shield_values) if shield_values else 0.0)
    if bool(safety_shield_enabled):
        summary["shield_note"] = "Local one-step gate-density action shield enabled in this adapter."
    else:
        summary["shield_note"] = "Local one-step gate-density action shield disabled for decomposition ablation."
    summary["morph_min_distance_m"] = None
    summary["morph_note"] = "multi-vehicle scenarios only"
    return summary


def main() -> None:
    _bootstrap_imports()

    from shared.core.dynamic_gate_density_2d import MAX_DRONE_COMMAND_ACCEL_MPS2, MAX_DRONE_COMMAND_SPEED_MPS
    from single_gate.configs.experiment_config import SINGLE_EXPERIMENT_CONFIG
    from single_gate.env.single_gate_env import SingleGate2DEnv
    from single_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphSACAgent
    from single_gate.training import validate_single_checkpoint_compatibility

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--random-yaw", action="store_true")
    parser.add_argument("--gate-layout-version", type=str, default=GATE_LAYOUT_VERSION, choices=ALLOWED_GATE_LAYOUT_VERSIONS)
    parser.add_argument("--enable-agent-policy", action="store_true")
    parser.add_argument("--enable-global-planner", action="store_true")
    parser.add_argument("--enable-path-planner", action="store_true")
    parser.add_argument("--disable-safety-shield", action="store_true")
    parser.add_argument("--shield-max-activations", type=int, default=-1)
    parser.add_argument("--planner-grid-resolution-m", type=float, default=0.0)
    parser.add_argument("--planner-time-budget-ms", type=float, default=0.0)
    parser.add_argument("--dynamic-replan-interval-steps", type=int, default=0)
    parser.add_argument("--dynamic-replan-clearance-threshold-m", type=float, default=0.0)
    parser.add_argument("--dynamic-gate-speed-cap-base-mps", type=float, default=0.45)
    parser.add_argument("--dynamic-gate-speed-cap-gain", type=float, default=0.60)
    parser.add_argument("--dynamic-shield-rollout-steps", type=int, default=6)
    parser.add_argument("--dynamic-planner-inflation-extra-m", type=float, default=0.0)
    parser.add_argument("--dynamic-final-goal-bias-start-x-m", type=float, default=0.0)
    parser.add_argument("--dynamic-final-goal-bias-strength", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-controller-profile",
        type=str,
        default="none",
        choices=("none", "density_adaptive_v1"),
        help="Use one density-adaptive dynamic-gate controller instead of hand-tuned per-gate knobs.",
    )
    parser.add_argument("--moving-gates", action="store_true")
    parser.add_argument("--moving-gate-amplitude-m", type=float, default=0.0)
    parser.add_argument("--moving-gate-speed-hz", type=float, default=0.0)
    parser.add_argument("--moving-gate-speed-mps", type=float, default=0.0)
    parser.add_argument("--drone-speed-mps", type=float, default=SINGLE_EXPERIMENT_CONFIG.environment.max_command_speed_mps)
    parser.add_argument("--drone-accel-mps2", type=float, default=SINGLE_EXPERIMENT_CONFIG.environment.max_accel_mps2)
    parser.add_argument("--gate-post-radius-scale", type=float, default=1.0)
    parser.add_argument("--gate-half-width-scale", type=float, default=1.0)
    parser.add_argument("--enable-route-guidance", action="store_true")
    parser.add_argument("--guidance-provider", "--guidance-provider", dest="guidance_provider", type=str, default=DEFAULT_GUIDANCE_PROVIDER)
    parser.add_argument("--guidance-base-url", "--guidance-base-url", dest="guidance_base_url", type=str, default=DEFAULT_GUIDANCE_BASE_URL)
    parser.add_argument("--guidance-model", "--guidance-model", dest="guidance_model", type=str, default=DEFAULT_GUIDANCE_MODEL)
    parser.add_argument("--guidance-timeout-s", type=float, default=30.0)
    parser.add_argument("--guidance-temperature", type=float, default=0.1)
    parser.add_argument("--guidance-prompt-version", "--guidance-prompt-version", dest="guidance_prompt_version", type=str, default=DEFAULT_GUIDANCE_PROMPT_VERSION)
    parser.add_argument("--guidance-shadow-mode", action="store_true")
    parser.add_argument("--guidance-visible", action="store_true")
    parser.add_argument("--guidance-cache-enabled", action="store_true")
    parser.add_argument("--guidance-async-enabled", action="store_true")
    parser.add_argument("--guidance-query-interval-steps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    args = parser.parse_args()

    if str(args.dynamic_controller_profile) == "density_adaptive_v1":
        for name, value in _density_adaptive_controller_profile(int(args.gate_count)).items():
            setattr(args, name, value)

    if int(args.gate_count) < 0 or int(args.gate_count) > MAX_GATE_COUNT:
        raise SystemExit(f"--gate-count must be in [0, {MAX_GATE_COUNT}]; got {args.gate_count}")
    if int(args.seed) not in ALLOWED_SEEDS:
        raise SystemExit(f"--seed must be one of {ALLOWED_SEEDS}; got {args.seed}")
    if not bool(args.random_yaw):
        raise SystemExit("--random-yaw is required for this formal gate-density branch.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint is missing: {args.checkpoint}")
    if bool(args.enable_route_guidance) and str(args.guidance_provider).strip().lower() != "local_http":
        raise SystemExit("Only --guidance-provider local_http is implemented for gate-density route guidance.")
    if bool(args.guidance_visible) and not bool(args.enable_route_guidance):
        raise SystemExit("--guidance-visible requires --enable-route-guidance")
    if bool(args.guidance_shadow_mode) and bool(args.guidance_visible):
        raise SystemExit("Use either --guidance-shadow-mode or --guidance-visible, not both.")
    if float(args.moving_gate_speed_mps) < 0.0 or float(args.moving_gate_speed_mps) > MAX_MOVING_GATE_SPEED_MPS:
        raise SystemExit(f"--moving-gate-speed-mps must be in [0, {MAX_MOVING_GATE_SPEED_MPS}]")
    if float(args.drone_speed_mps) <= 0.0:
        raise SystemExit("--drone-speed-mps must be positive")
    if float(args.drone_speed_mps) > MAX_DRONE_COMMAND_SPEED_MPS:
        raise SystemExit(f"--drone-speed-mps must be <= {MAX_DRONE_COMMAND_SPEED_MPS}")
    if float(args.drone_accel_mps2) <= 0.0:
        raise SystemExit("--drone-accel-mps2 must be positive")
    if float(args.drone_accel_mps2) > MAX_DRONE_COMMAND_ACCEL_MPS2:
        raise SystemExit(f"--drone-accel-mps2 must be <= {MAX_DRONE_COMMAND_ACCEL_MPS2}")
    if float(args.gate_post_radius_scale) <= 0.0:
        raise SystemExit("--gate-post-radius-scale must be positive")
    if float(args.gate_half_width_scale) <= 0.0:
        raise SystemExit("--gate-half-width-scale must be positive")
    if int(args.dynamic_replan_interval_steps) < 0:
        raise SystemExit("--dynamic-replan-interval-steps must be >= 0")
    if float(args.dynamic_replan_clearance_threshold_m) < 0.0:
        raise SystemExit("--dynamic-replan-clearance-threshold-m must be >= 0")
    if float(args.dynamic_gate_speed_cap_base_mps) <= 0.0:
        raise SystemExit("--dynamic-gate-speed-cap-base-mps must be positive")
    if float(args.dynamic_gate_speed_cap_gain) < 0.0:
        raise SystemExit("--dynamic-gate-speed-cap-gain must be >= 0")
    if int(args.dynamic_shield_rollout_steps) <= 0:
        raise SystemExit("--dynamic-shield-rollout-steps must be positive")
    if float(args.dynamic_planner_inflation_extra_m) < 0.0:
        raise SystemExit("--dynamic-planner-inflation-extra-m must be >= 0")
    if float(args.dynamic_final_goal_bias_start_x_m) < 0.0:
        raise SystemExit("--dynamic-final-goal-bias-start-x-m must be >= 0")
    if not (0.0 <= float(args.dynamic_final_goal_bias_strength) <= 1.0):
        raise SystemExit("--dynamic-final-goal-bias-strength must be in [0, 1]")

    gate_post_radius_m = float(GATE_POST_RADIUS_M) * float(args.gate_post_radius_scale)
    gate_half_width_m = float(GATE_HALF_WIDTH_M) * float(args.gate_half_width_scale)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guidance_client: LocalGateGuidanceClient | None = None
    if bool(args.enable_route_guidance):
        guidance_client = LocalGateGuidanceClient(
            base_url=str(args.guidance_base_url),
            model=str(args.guidance_model),
            timeout_s=float(args.guidance_timeout_s),
            temperature=float(args.guidance_temperature),
            prompt_version=str(args.guidance_prompt_version),
            cache_enabled=bool(args.guidance_cache_enabled),
        )
        # Visible guidance runs need a live local endpoint call.
        guidance_client.health_check()

    gate_centers_xy, gate_yaws = _generate_gate_layout(
        gate_count=int(args.gate_count),
        seed=int(args.seed),
        random_yaw=bool(args.random_yaw),
        layout_version=str(args.gate_layout_version),
    )
    layout_profile = _layout_profile(str(args.gate_layout_version))
    initial_gate_gate_stats = _gate_gate_clearance_stats(gate_centers_xy, gate_yaws)
    initial_gate_gate_frame_stats = _gate_gate_frame_clearance_stats(gate_centers_xy, gate_yaws)
    obstacle_map = build_gate_obstacle_map(gate_centers_xy, gate_yaws, post_radius_m=gate_post_radius_m)
    env_config = replace(
        SINGLE_EXPERIMENT_CONFIG.environment,
        fixed_height_m=layout_profile.start_xyz[2],
        drone_radius_m=DRONE_RADIUS_M,
        max_command_speed_mps=float(args.drone_speed_mps),
        max_accel_mps2=float(args.drone_accel_mps2),
        goal_radius_m=GOAL_RADIUS_M,
        max_episode_steps=int(args.max_steps),
        start_x_m=layout_profile.start_xyz[0],
        goal_x_m=layout_profile.goal_xyz[0],
        start_y_range_m=(layout_profile.start_xyz[1], layout_profile.start_xyz[1]),
        goal_y_range_m=(layout_profile.goal_xyz[1], layout_profile.goal_xyz[1]),
        world_x_bounds_m=layout_profile.world_x_bounds_m,
        world_y_bounds_m=layout_profile.world_y_bounds_m,
    )
    env = SingleGate2DEnv(env_config=env_config, obstacle_map=obstacle_map)
    validate_single_checkpoint_compatibility(checkpoint_path=args.checkpoint, env=env)
    agent = GraphSACAgent.from_defaults(obs_shapes=env.observation_shapes, device=args.device, seed=int(args.seed))
    if bool(args.enable_agent_policy):
        agent.load_checkpoint(args.checkpoint)
    else:
        agent = None

    task_payload = {
        "task_id": f"gate_density_count_{int(args.gate_count):02d}_seed_{int(args.seed)}",
        "branch": "gate_density_single",
        "implementation_type": "gate_density_adapter_v1",
        "gate_layout_version": str(args.gate_layout_version),
        "checkpoint": str(args.checkpoint),
        "gate_count": int(args.gate_count),
        "seed": int(args.seed),
        "random_yaw": bool(args.random_yaw),
        "start_xyz": list(layout_profile.start_xyz),
        "goal_xyz": list(layout_profile.goal_xyz),
        "world_x_bounds_m": list(layout_profile.world_x_bounds_m),
        "world_y_bounds_m": list(env_config.world_y_bounds_m),
        "gate_region_x_m": list(layout_profile.gate_region_x_m),
        "gate_region_y_m": list(layout_profile.gate_region_y_m),
        "fixed_height_m": float(layout_profile.start_xyz[2]),
        "gate_bottom_height_m": float(GATE_BOTTOM_HEIGHT_M),
        "gate_top_height_m": float(GATE_TOP_HEIGHT_M),
        "gate_center_height_m": float(GATE_CENTER_HEIGHT_M),
        "gate_native_visual_height_m": float(GATE_NATIVE_VISUAL_HEIGHT_M),
        "gate_visual_scale_xyz": [1.0, 1.0, float(GATE_VISUAL_SCALE_Z)],
        "height_contract": (
            "single and multi fixed-height 2D flight plane is locked to the gate geometric center; "
            "success is invalid if a rollout uses a z_m different from gate_center_height_m"
        ),
        "corridor_through_contract": (
            "success requires passing through the gate/obstacle corridor; side bypass, over-gate escape, "
            "static-gate masquerading, and non-terminal gate contact are invalid"
        ),
        "drone_radius_m": DRONE_RADIUS_M,
        "drone_shell_top_m": float(GATE_CENTER_HEIGHT_M + DRONE_RADIUS_M),
        "drone_shell_bottom_m": float(GATE_CENTER_HEIGHT_M - DRONE_RADIUS_M),
        "drone_top_clearance_to_gate_top_m": float(GATE_TOP_HEIGHT_M - (GATE_CENTER_HEIGHT_M + DRONE_RADIUS_M)),
        "drone_bottom_clearance_to_gate_bottom_m": float((GATE_CENTER_HEIGHT_M - DRONE_RADIUS_M) - GATE_BOTTOM_HEIGHT_M),
        "safety_margin_m": SAFETY_MARGIN_M,
        "gate_half_width_m": float(gate_half_width_m),
        "gate_post_radius_m": float(gate_post_radius_m),
        "gate_yaw_policy": (
            "random_uniform_minus5_to_plus5_deg_formation_facing"
            if str(args.gate_layout_version) == "irregular_centerline_v7_large_arena_dynamic"
            else "layout_default"
        ),
        "gate_requested_yaws_rad": list(gate_yaws),
        "gate_requested_yaws_deg": [float(math.degrees(value)) for value in gate_yaws],
        "gate_asset_yaw_correction_deg": 0.0,
        "gate_orientation_contract": (
            "all v7 obstacles use random yaw in [-5,+5] degrees so each hollow gate faces "
            "the -X start / +X formation travel direction; visual replay and collision posts "
            "use the same yaw"
            if str(args.gate_layout_version) == "irregular_centerline_v7_large_arena_dynamic"
            else "visual replay and collision posts use gate_yaws_deg directly"
        ),
        "gate_centers_xy": [list(item) for item in gate_centers_xy],
        "gate_yaws_rad": list(gate_yaws),
        "gate_yaws_deg": [float(math.degrees(value)) for value in gate_yaws],
        "enable_agent_policy": bool(args.enable_agent_policy),
        "enable_global_planner": bool(args.enable_global_planner),
        "enable_path_planner": bool(args.enable_path_planner),
        "safety_shield_enabled": not bool(args.disable_safety_shield),
        "shield_max_activations": int(args.shield_max_activations),
        "planner_grid_resolution_m": float(args.planner_grid_resolution_m),
        "planner_time_budget_ms": float(args.planner_time_budget_ms),
        "dynamic_replan_interval_steps": int(args.dynamic_replan_interval_steps),
        "dynamic_replan_clearance_threshold_m": float(args.dynamic_replan_clearance_threshold_m),
        "dynamic_gate_speed_cap_base_mps": float(args.dynamic_gate_speed_cap_base_mps),
        "dynamic_gate_speed_cap_gain": float(args.dynamic_gate_speed_cap_gain),
        "dynamic_shield_rollout_steps": int(args.dynamic_shield_rollout_steps),
        "dynamic_planner_inflation_extra_m": float(args.dynamic_planner_inflation_extra_m),
        "dynamic_final_goal_bias_start_x_m": float(args.dynamic_final_goal_bias_start_x_m),
        "dynamic_final_goal_bias_strength": float(args.dynamic_final_goal_bias_strength),
        "dynamic_controller_profile": str(args.dynamic_controller_profile),
        "moving_gates_enabled": bool(args.moving_gates),
        "moving_gate_amplitude_m": float(args.moving_gate_amplitude_m),
        "moving_gate_speed_hz": _resolve_moving_gate_speed_hz(
            amplitude_m=float(args.moving_gate_amplitude_m),
            speed_hz=float(args.moving_gate_speed_hz),
            speed_mps=float(args.moving_gate_speed_mps),
        ),
        "moving_gate_speed_mps": float(args.moving_gate_speed_mps),
        "drone_speed_mps": float(args.drone_speed_mps),
        "drone_accel_mps2": float(args.drone_accel_mps2),
        "moving_gate_motion_profile": (
            "large_motion: lateral_sweep + diagonal_lissajous + adjacent_antiphase_channel_opening"
            if str(args.gate_layout_version)
            in {"irregular_centerline_v6_large_motion_dynamic", "irregular_centerline_v7_large_arena_dynamic"}
            else "v5_historical_small_motion"
        ),
        "moving_gate_expected_main_amplitude_m": 0.8
        if str(args.gate_layout_version)
        in {"irregular_centerline_v6_large_motion_dynamic", "irregular_centerline_v7_large_arena_dynamic"}
        else None,
        "moving_gate_expected_pressure_amplitude_m": 1.1
        if str(args.gate_layout_version)
        in {"irregular_centerline_v6_large_motion_dynamic", "irregular_centerline_v7_large_arena_dynamic"}
        else None,
        "training_render_policy": layout_profile.training_render_policy,
        "obstacle_dynamics_policy": layout_profile.obstacle_dynamics_policy,
        "collision_policy": layout_profile.collision_policy,
        "gate_gate_collision_policy": (
            "post-disk non-overlap projection before every dynamic obstacle update; "
            "thin visual-frame SAT non-overlap projection before every dynamic obstacle update; "
            "gate-gate penetration is treated as invalid geometry, not physical toppling"
        ),
        "gate_gate_clearance_margin_m": float(GATE_GATE_CLEARANCE_MARGIN_M),
        "gate_gate_frame_clearance_margin_m": float(GATE_GATE_FRAME_CLEARANCE_MARGIN_M),
        "gate_visual_frame_half_depth_m": float(GATE_VISUAL_FRAME_HALF_DEPTH_M),
        "initial_gate_gate_min_clearance_m": float(initial_gate_gate_stats["gate_gate_min_clearance_m"]),
        "initial_gate_gate_overlap_pair_count": int(initial_gate_gate_stats["gate_gate_overlap_pair_count"]),
        "initial_gate_gate_frame_min_clearance_m": float(
            initial_gate_gate_frame_stats["gate_gate_frame_min_clearance_m"]
        ),
        "initial_gate_gate_frame_overlap_pair_count": int(
            initial_gate_gate_frame_stats["gate_gate_frame_overlap_pair_count"]
        ),
        "drone_dynamic_collision_policy": (
            "terminal crash on current live posts, next live posts, and one-step moving-post swept clearance"
        ),
        "gate_distribution_policy": layout_profile.distribution_policy,
        "route_guidance_enabled": bool(args.enable_route_guidance),
        "guidance_shadow_mode": bool(args.guidance_shadow_mode),
        "guidance_visible": bool(args.guidance_visible),
        "guidance_async_enabled": bool(args.guidance_async_enabled),
        "guidance_cache_enabled": bool(args.guidance_cache_enabled),
        "guidance_provider": str(args.guidance_provider),
        "guidance_base_url": str(args.guidance_base_url),
        "guidance_model_name": str(args.guidance_model),
        "guidance_timeout_s": float(args.guidance_timeout_s),
        "guidance_temperature": float(args.guidance_temperature),
        "guidance_prompt_version": str(args.guidance_prompt_version),
        "guidance_query_interval_steps": int(args.guidance_query_interval_steps),
    }
    write_json(output_dir / "stage_manifest.json", task_payload)

    metrics_list = [
        run_episode(
            episode_index=episode_idx,
            env=env,
            agent=agent,
            args=args,
            task_payload=task_payload,
            output_dir=output_dir,
            guidance_client=guidance_client,
        )
        for episode_idx in range(int(args.episodes))
    ]

    aggregate = _summarize_episode_metrics(
        metrics_list,
        safety_shield_enabled=not bool(args.disable_safety_shield),
    )
    stage_summary = task_payload | aggregate | {
        "stage_status": "completed",
        "curve_status": "single_point_pending_batch",
        "continue_or_stop": "continue_batch_if_base_gate_checks_pass",
        "stop_root_cause": None,
        "checkpoint_handoff": {
            "source_checkpoint": str(args.checkpoint),
            "used_for_all_gate_counts": True,
            "handoff_ok": True,
        },
        "isaaclab_replay_behavior": {
            "visual_preview_command": None,
            "checked_in_this_run": False,
            "note": "2D rollout metrics generated; visual replay output is not part of this package.",
        },
    }
    write_json(output_dir / "stage_summary.json", stage_summary)
    write_json(output_dir / "summary.json", stage_summary)
    write_json(
        output_dir / "best_stage_checkpoint_map.json",
        {
            "gate_density_single": {
                "checkpoint": str(args.checkpoint),
                "gate_count": int(args.gate_count),
                "seed": int(args.seed),
                "handoff_ok": True,
            }
        },
    )

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(stage_summary.keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(stage_summary)

    print("gate-density single evaluation complete")
    print(f"stage_summary_path={output_dir / 'stage_summary.json'}")
    print(f"stage_manifest_path={output_dir / 'stage_manifest.json'}")
    print(f"best_stage_checkpoint_map_path={output_dir / 'best_stage_checkpoint_map.json'}")
    print(f"success_rate={stage_summary['success_rate']}")
    print(f"collision_rate={stage_summary['collision_rate']}")
    print(f"out_of_bounds_rate={stage_summary['out_of_bounds_rate']}")
    print(f"timeout_rate={stage_summary['timeout_rate']}")


if __name__ == "__main__":
    main()


