"""Episode runner for single-drone gate-density evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_RUNTIME_NAMES = (
    "DRONE_RADIUS_M",
    "GATE_BOTTOM_HEIGHT_M",
    "GATE_CENTER_HEIGHT_M",
    "GATE_LAYOUT_VERSION",
    "GATE_REGION_X",
    "GATE_REGION_Y",
    "GATE_TOP_HEIGHT_M",
    "GOAL_XYZ",
    "START_XYZ",
    "GateDensityController",
    "LocalGateGuidanceClient",
    "_build_gate_obstacle_map",
    "_gate_gate_clearance_stats",
    "_gate_gate_frame_clearance_stats",
    "_moving_gate_centers",
    "_moving_gate_swept_clearance_m",
    "_path_length",
    "_percentile",
    "_resolve_moving_gate_speed_hz",
    "_write_json",
)


def bind_episode_runner_runtime(namespace: dict[str, Any]) -> None:
    """Bind helpers that remain in the CLI module during incremental refactors."""

    for name in _RUNTIME_NAMES:
        if name in namespace:
            globals()[name] = namespace[name]


def run_episode(
    *,
    episode_index: int,
    env,
    agent,
    args: argparse.Namespace,
    task_payload: dict[str, Any],
    output_dir: Path,
    guidance_client: LocalGateGuidanceClient | None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=int(args.seed) * 1000 + int(episode_index))
    controller = GateDensityController(
        env=env,
        agent=agent,
        args=args,
        gate_count=int(args.gate_count),
        enable_agent_policy=bool(args.enable_agent_policy),
        enable_global_planner=bool(args.enable_global_planner),
        enable_path_planner=bool(args.enable_path_planner),
        guidance_client=guidance_client,
    )
    episode_dir = output_dir / f"episode_{episode_index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    _write_json(episode_dir / "task.json", task_payload | {"episode_index": int(episode_index)})
    layout_version = str(task_payload.get("gate_layout_version", GATE_LAYOUT_VERSION))
    base_gate_centers_xy = tuple(tuple(map(float, center)) for center in task_payload.get("gate_centers_xy", ()))
    gate_yaws = tuple(float(value) for value in task_payload.get("gate_yaws_rad", ()))
    moving_gates_enabled = bool(getattr(args, "moving_gates", False))
    moving_gate_amplitude_m = float(getattr(args, "moving_gate_amplitude_m", 0.0) or 0.0)
    moving_gate_speed_mps = float(getattr(args, "moving_gate_speed_mps", 0.0) or 0.0)
    moving_gate_speed_hz = _resolve_moving_gate_speed_hz(
        amplitude_m=moving_gate_amplitude_m,
        speed_hz=float(getattr(args, "moving_gate_speed_hz", 0.0) or 0.0),
        speed_mps=moving_gate_speed_mps,
    )
    controller.set_dynamic_gate_context(
        base_centers_xy=base_gate_centers_xy,
        gate_yaws=gate_yaws,
        seed=int(args.seed),
        layout_version=layout_version,
        amplitude_m=moving_gate_amplitude_m,
        speed_hz=moving_gate_speed_hz,
    )
    controller.reset()
    dynamic_obstacle_updates = 0
    gate_motion_bounds = [
        {
            "base_x": float(center[0]),
            "base_y": float(center[1]),
            "min_x": float("inf"),
            "max_x": float("-inf"),
            "min_y": float("inf"),
            "max_y": float("-inf"),
            "max_displacement_m": 0.0,
        }
        for center in base_gate_centers_xy
    ]

    rows: list[dict[str, Any]] = []
    clearances: list[float] = []
    gate_gate_min_clearances: list[float] = []
    gate_gate_overlap_pair_counts: list[int] = []
    gate_gate_frame_min_clearances: list[float] = []
    gate_gate_frame_overlap_pair_counts: list[int] = []
    moving_gate_swept_clearances: list[float] = []
    speeds: list[float] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    done_reason = "max_steps_exhausted"
    dynamic_swept_collision_count = 0
    guidance_base_query_count = int(guidance_client.query_count if guidance_client is not None else 0)
    guidance_base_success_count = int(guidance_client.success_count if guidance_client is not None else 0)
    guidance_base_failure_count = int(guidance_client.failure_count if guidance_client is not None else 0)
    guidance_base_fallback_count = int(guidance_client.fallback_count if guidance_client is not None else 0)
    guidance_base_cache_hit_count = int(guidance_client.cache_hit_count if guidance_client is not None else 0)
    guidance_base_latency_count = len(guidance_client.latencies_ms) if guidance_client is not None else 0
    guidance_base_record_count = len(guidance_client.guidance_records) if guidance_client is not None else 0

    for step in range(int(args.max_steps)):
        t_for_obstacles = float(getattr(env._state, "t_sec", step * env.env_config.dt_s))
        if moving_gates_enabled:
            moved_centers = _moving_gate_centers(
                base_centers_xy=base_gate_centers_xy,
                gate_yaws=gate_yaws,
                seed=int(args.seed),
                t_sec=t_for_obstacles,
                enabled=True,
                amplitude_m=moving_gate_amplitude_m,
                speed_hz=moving_gate_speed_hz,
                layout_version=layout_version,
            )
            env.obstacle_map = _build_gate_obstacle_map(moved_centers, gate_yaws)
            observation = env._build_observation()
            dynamic_obstacle_updates += 1
        else:
            moved_centers = base_gate_centers_xy
        gate_gate_stats = _gate_gate_clearance_stats(moved_centers, gate_yaws)
        gate_gate_frame_stats = _gate_gate_frame_clearance_stats(moved_centers, gate_yaws)
        if math.isfinite(float(gate_gate_stats["gate_gate_min_clearance_m"])):
            gate_gate_min_clearances.append(float(gate_gate_stats["gate_gate_min_clearance_m"]))
        gate_gate_overlap_pair_counts.append(int(gate_gate_stats["gate_gate_overlap_pair_count"]))
        if math.isfinite(float(gate_gate_frame_stats["gate_gate_frame_min_clearance_m"])):
            gate_gate_frame_min_clearances.append(float(gate_gate_frame_stats["gate_gate_frame_min_clearance_m"]))
        gate_gate_frame_overlap_pair_counts.append(int(gate_gate_frame_stats["gate_gate_frame_overlap_pair_count"]))
        state = env.current_state()
        position_xy = state.position_xy
        clearance = float(env.obstacle_map.min_signed_distance(position_xy, drone_radius_m=DRONE_RADIUS_M))
        action = controller.act(observation, step=step)
        live_gate_centers_xy = [[float(cx), float(cy)] for cx, cy in moved_centers]
        live_gate_fields: dict[str, Any] = {}
        for gate_idx, (cx, cy) in enumerate(moved_centers):
            cx_f = float(cx)
            cy_f = float(cy)
            live_gate_fields[f"live_gate_center_{gate_idx}_x_m"] = cx_f
            live_gate_fields[f"live_gate_center_{gate_idx}_y_m"] = cy_f
            if gate_idx < len(gate_motion_bounds):
                bounds = gate_motion_bounds[gate_idx]
                bounds["min_x"] = min(float(bounds["min_x"]), cx_f)
                bounds["max_x"] = max(float(bounds["max_x"]), cx_f)
                bounds["min_y"] = min(float(bounds["min_y"]), cy_f)
                bounds["max_y"] = max(float(bounds["max_y"]), cy_f)
                displacement_m = math.hypot(cx_f - float(bounds["base_x"]), cy_f - float(bounds["base_y"]))
                bounds["max_displacement_m"] = max(float(bounds["max_displacement_m"]), float(displacement_m))
        rows.append(
            {
                "step": int(step),
                "t_sec": float(getattr(env._state, "t_sec", step * env.env_config.dt_s)),
                "x_m": float(position_xy[0]),
                "y_m": float(position_xy[1]),
                "z_m": float(env.env_config.fixed_height_m),
                "vx_mps": float(state.velocity_xy[0]),
                "vy_mps": float(state.velocity_xy[1]),
                "yaw_rad": float(state.yaw_rad),
                "speed_mps": float(np.linalg.norm(np.asarray(state.velocity_xy, dtype=np.float32))),
                "clearance_m": clearance,
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "goal_x_m": float(state.goal_xy[0]),
                "goal_y_m": float(state.goal_xy[1]),
                "route_guidance_source": str((controller._last_route_guidance or {}).get("route_guidance_source", "")),
                "guidance_heading_x": float((controller._last_route_guidance or {}).get("heading_x", 0.0)),
                "guidance_heading_y": float((controller._last_route_guidance or {}).get("heading_y", 0.0)),
                "guidance_speed_scale": float((controller._last_route_guidance or {}).get("speed_scale", 0.0)),
                "guidance_confidence": float((controller._last_route_guidance or {}).get("confidence", 0.0)),
                "guidance_risk_level": float((controller._last_route_guidance or {}).get("risk_level", 0.0)),
                "guidance_replan_urgency": float((controller._last_route_guidance or {}).get("replan_urgency", 0.0)),
                "guidance_waypoint_bias_y": float((controller._last_route_guidance or {}).get("waypoint_bias_y", 0.0)),
                "guidance_dynamic_clearance_margin_m": float(
                    (controller._last_route_guidance or {}).get("dynamic_clearance_margin_m", 0.0)
                ),
                "moving_gates_enabled": bool(moving_gates_enabled),
                "moving_gate_amplitude_m": float(moving_gate_amplitude_m),
                "moving_gate_speed_hz": float(moving_gate_speed_hz),
                "moving_gate_speed_mps": float(moving_gate_speed_mps),
                "live_gate_center_0_x_m": float(moved_centers[0][0]) if moved_centers else 0.0,
                "live_gate_center_0_y_m": float(moved_centers[0][1]) if moved_centers else 0.0,
                "live_gate_centers_xy_json": json.dumps(live_gate_centers_xy, ensure_ascii=False, separators=(",", ":")),
                "gate_gate_min_clearance_m": float(gate_gate_stats["gate_gate_min_clearance_m"]),
                "gate_gate_overlap_pair_count": int(gate_gate_stats["gate_gate_overlap_pair_count"]),
                "gate_gate_frame_min_clearance_m": float(gate_gate_frame_stats["gate_gate_frame_min_clearance_m"]),
                "gate_gate_frame_overlap_pair_count": int(gate_gate_frame_stats["gate_gate_frame_overlap_pair_count"]),
                "dynamic_swept_collision": False,
                "moving_gate_swept_clearance_m": float("inf"),
                "moving_gate_endpoint_clearance_next_m": float("inf"),
                **live_gate_fields,
            }
        )
        clearances.append(clearance)
        speeds.append(float(np.linalg.norm(np.asarray(state.velocity_xy, dtype=np.float32))))
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if moving_gates_enabled:
            next_state = env.current_state()
            next_position_xy = next_state.position_xy
            next_t_for_obstacles = float(getattr(env._state, "t_sec", t_for_obstacles + env.env_config.dt_s))
            next_moved_centers = _moving_gate_centers(
                base_centers_xy=base_gate_centers_xy,
                gate_yaws=gate_yaws,
                seed=int(args.seed),
                t_sec=next_t_for_obstacles,
                enabled=True,
                amplitude_m=moving_gate_amplitude_m,
                speed_hz=moving_gate_speed_hz,
                layout_version=layout_version,
            )
            swept_clearance_m = _moving_gate_swept_clearance_m(
                drone_start_xy=position_xy,
                drone_end_xy=next_position_xy,
                gate_centers_start_xy=moved_centers,
                gate_centers_end_xy=next_moved_centers,
                gate_yaws=gate_yaws,
                drone_radius_m=DRONE_RADIUS_M,
            )
            endpoint_clearance_next_m = float(
                _build_gate_obstacle_map(next_moved_centers, gate_yaws).min_signed_distance(
                    next_position_xy,
                    drone_radius_m=DRONE_RADIUS_M,
                )
            )
            dynamic_clearance_m = min(float(swept_clearance_m), float(endpoint_clearance_next_m))
            moving_gate_swept_clearances.append(dynamic_clearance_m)
            rows[-1]["moving_gate_swept_clearance_m"] = float(swept_clearance_m)
            rows[-1]["moving_gate_endpoint_clearance_next_m"] = float(endpoint_clearance_next_m)
            if dynamic_clearance_m <= 0.0:
                rows[-1]["dynamic_swept_collision"] = True
                dynamic_swept_collision_count += 1
                if str(info.get("done_reason") or "") != "collision":
                    rewards[-1] = float(rewards[-1] + env.env_config.collision_penalty)
                done_reason = "collision"
                break
        if terminated or truncated:
            done_reason = str(info.get("done_reason") or "unknown")
            break

    final_state = env.current_state()
    points_xy = [(float(row["x_m"]), float(row["y_m"])) for row in rows]
    gate_region_x = tuple(float(v) for v in task_payload.get("gate_region_x_m", GATE_REGION_X))
    gate_region_y = tuple(float(v) for v in task_payload.get("gate_region_y_m", GATE_REGION_Y))
    fixed_height_m = float(task_payload.get("fixed_height_m", env.env_config.fixed_height_m))
    gate_center_height_m = float(task_payload.get("gate_center_height_m", GATE_CENTER_HEIGHT_M))
    gate_top_height_m = float(task_payload.get("gate_top_height_m", GATE_TOP_HEIGHT_M))
    gate_bottom_height_m = float(task_payload.get("gate_bottom_height_m", GATE_BOTTOM_HEIGHT_M))
    drone_shell_top_m = float(fixed_height_m + DRONE_RADIUS_M)
    drone_shell_bottom_m = float(fixed_height_m - DRONE_RADIUS_M)
    height_contract_passed = bool(
        abs(fixed_height_m - gate_center_height_m) <= 1.0e-5
        and drone_shell_top_m < gate_top_height_m
        and drone_shell_bottom_m > gate_bottom_height_m
    )
    points_inside_gate_x = [
        (x_m, y_m)
        for x_m, y_m in points_xy
        if min(gate_region_x) <= float(x_m) <= max(gate_region_x)
    ]
    side_bypass_failure = bool(
        any(float(y_m) < min(gate_region_y) or float(y_m) > max(gate_region_y) for _x_m, y_m in points_inside_gate_x)
    )
    corridor_through_success = bool(
        int(args.gate_count) <= 0
        or (
            bool(points_inside_gate_x)
            and not side_bypass_failure
            and any(min(gate_region_y) <= float(y_m) <= max(gate_region_y) for _x_m, y_m in points_inside_gate_x)
        )
    )
    height_escape_failure = not height_contract_passed
    corridor_miss_failure = bool(int(args.gate_count) > 0 and not corridor_through_success)
    raw_goal_reached = done_reason == "goal_reached"
    success = bool(
        raw_goal_reached
        and height_contract_passed
        and corridor_through_success
        and not side_bypass_failure
        and not height_escape_failure
    )
    collision = done_reason == "collision"
    out_of_bounds = done_reason == "out_of_bounds"
    timeout = done_reason in {"timeout", "max_steps_exhausted"}
    min_clearance_observed = float(min(clearances) if clearances else float("inf"))
    min_swept_clearance_observed = float(
        min(moving_gate_swept_clearances) if moving_gate_swept_clearances else float("inf")
    )
    max_gate_gate_overlap_pairs = int(max(gate_gate_overlap_pair_counts) if gate_gate_overlap_pair_counts else 0)
    max_gate_gate_frame_overlap_pairs = int(
        max(gate_gate_frame_overlap_pair_counts) if gate_gate_frame_overlap_pair_counts else 0
    )
    collision_contract_passed = bool(
        max_gate_gate_overlap_pairs == 0
        and max_gate_gate_frame_overlap_pairs == 0
        and (min_clearance_observed > 0.0 or collision)
        and (min_swept_clearance_observed > 0.0 or collision)
    )
    goal_errors = [math.hypot(float(row["goal_x_m"]) - float(row["x_m"]), float(row["goal_y_m"]) - float(row["y_m"])) for row in rows]
    initial_goal_distance_m = (
        math.hypot(float(rows[0]["goal_x_m"]) - float(rows[0]["x_m"]), float(rows[0]["goal_y_m"]) - float(rows[0]["y_m"]))
        if rows
        else 0.0
    )
    final_goal_distance_m = float(goal_errors[-1]) if goal_errors else 0.0
    progress_distance_m = max(0.0, float(initial_goal_distance_m) - float(final_goal_distance_m))
    action_deltas = [
        float(np.linalg.norm(actions[idx] - actions[idx - 1])) for idx in range(1, len(actions))
    ]
    gate_motion_ranges_x = []
    gate_motion_ranges_y = []
    gate_motion_ranges = []
    gate_max_displacements = []
    for bounds in gate_motion_bounds:
        if math.isfinite(float(bounds["min_x"])) and math.isfinite(float(bounds["min_y"])):
            range_x = float(bounds["max_x"]) - float(bounds["min_x"])
            range_y = float(bounds["max_y"]) - float(bounds["min_y"])
            gate_motion_ranges_x.append(range_x)
            gate_motion_ranges_y.append(range_y)
            gate_motion_ranges.append(float(math.hypot(range_x, range_y)))
            gate_max_displacements.append(float(bounds["max_displacement_m"]))
    metrics = {
        "episode_index": int(episode_index),
        "success": bool(success),
        "collision": bool(collision),
        "out_of_bounds": bool(out_of_bounds),
        "timeout": bool(timeout),
        "done_reason": done_reason,
        "raw_goal_reached": bool(raw_goal_reached),
        "height_contract_passed": bool(height_contract_passed),
        "height_escape_failure": bool(height_escape_failure),
        "side_bypass_failure": bool(side_bypass_failure),
        "corridor_miss_failure": bool(corridor_miss_failure),
        "corridor_through_success": bool(corridor_through_success),
        "fixed_height_m": float(fixed_height_m),
        "gate_bottom_height_m": float(gate_bottom_height_m),
        "gate_top_height_m": float(gate_top_height_m),
        "gate_center_height_m": float(gate_center_height_m),
        "drone_shell_top_m": float(drone_shell_top_m),
        "drone_shell_bottom_m": float(drone_shell_bottom_m),
        "drone_top_clearance_to_gate_top_m": float(gate_top_height_m - drone_shell_top_m),
        "drone_bottom_clearance_to_gate_bottom_m": float(drone_shell_bottom_m - gate_bottom_height_m),
        "crash_on_contact": bool(collision),
        "collision_policy": str(task_payload.get("collision_policy", "")),
        "obstacle_dynamics_policy": str(task_payload.get("obstacle_dynamics_policy", "")),
        "gate_gate_non_overlap_projection_enabled": bool(
            layout_version in {"irregular_centerline_v6_large_motion_dynamic", "irregular_centerline_v7_large_arena_dynamic"}
        ),
        "gate_gate_min_clearance_m": float(min(gate_gate_min_clearances) if gate_gate_min_clearances else float("inf")),
        "gate_gate_overlap_pair_count_max": int(max_gate_gate_overlap_pairs),
        "gate_gate_frame_min_clearance_m": float(
            min(gate_gate_frame_min_clearances) if gate_gate_frame_min_clearances else float("inf")
        ),
        "gate_gate_frame_overlap_pair_count_max": int(max_gate_gate_frame_overlap_pairs),
        "moving_gate_swept_clearance_m_min": float(min_swept_clearance_observed),
        "dynamic_swept_collision_count": int(dynamic_swept_collision_count),
        "collision_contract_passed": bool(collision_contract_passed),
        "collision_contract": (
            "live gate map refreshed before policy action; drone segment and swept dynamic-gate contacts "
            "terminate as collision; gate-gate post and visual-frame overlaps must be zero"
        ),
        "steps": len(rows),
        "episode_reward": float(sum(rewards)),
        "path_length_m": _path_length(points_xy),
        "initial_goal_distance_m": float(initial_goal_distance_m),
        "final_goal_distance_m": float(final_goal_distance_m),
        "progress_distance_m": float(progress_distance_m),
        "flight_time_s": float(len(rows) * env.env_config.dt_s),
        "min_clearance_m": float(min_clearance_observed),
        "mean_clearance_m": float(np.mean(clearances) if clearances else float("inf")),
        "mean_speed_mps": float(np.mean(speeds) if speeds else 0.0),
        "max_speed_mps": float(max(speeds) if speeds else 0.0),
        "mean_goal_tracking_error_m": float(np.mean(goal_errors) if goal_errors else 0.0),
        "max_goal_tracking_error_m": float(max(goal_errors) if goal_errors else 0.0),
        "guidance_tracking_error_mean_m": float(np.mean(controller.guidance_tracking_errors) if controller.guidance_tracking_errors else 0.0),
        "guidance_tracking_error_max_m": float(max(controller.guidance_tracking_errors) if controller.guidance_tracking_errors else 0.0),
        "trajectory_smoothness": float(sum(action_deltas)),
        "planner_call_count": int(controller.planner_call_count),
        "planner_failure_count": int(controller.planner_failure_count),
        "planner_latency_ms_mean": float(np.mean(controller.planner_latencies_ms) if controller.planner_latencies_ms else 0.0),
        "planner_latency_ms_p95": _percentile(controller.planner_latencies_ms, 95.0),
        "global_planner_trigger_count": int(controller.global_planner_trigger_count),
        "global_planner_latency_ms_mean": float(np.mean(controller.global_planner_latencies_ms) if controller.global_planner_latencies_ms else 0.0),
        "global_planner_latency_ms_p95": _percentile(controller.global_planner_latencies_ms, 95.0),
        "route_guidance_enabled": bool(args.enable_route_guidance),
        "guidance_shadow_mode": bool(args.guidance_shadow_mode),
        "guidance_visible": bool(args.guidance_visible),
        "guidance_provider": str(args.guidance_provider),
        "guidance_base_url": str(args.guidance_base_url),
        "guidance_model_name": str(args.guidance_model),
        "guidance_prompt_version": str(args.guidance_prompt_version),
        "guidance_query_count": int((guidance_client.query_count - guidance_base_query_count) if guidance_client is not None else 0),
        "guidance_success_count": int((guidance_client.success_count - guidance_base_success_count) if guidance_client is not None else 0),
        "guidance_failure_count": int((guidance_client.failure_count - guidance_base_failure_count) if guidance_client is not None else 0),
        "guidance_fallback_count": int((guidance_client.fallback_count - guidance_base_fallback_count) if guidance_client is not None else 0),
        "guidance_cache_hit_count": int((guidance_client.cache_hit_count - guidance_base_cache_hit_count) if guidance_client is not None else 0),
        "route_guidance_used_count": int(controller.route_guidance_used_count),
        "guidance_latency_ms_mean": float(
            np.mean(guidance_client.latencies_ms[guidance_base_latency_count:])
            if guidance_client is not None and guidance_client.latencies_ms[guidance_base_latency_count:]
            else 0.0
        ),
        "guidance_latency_ms_p95": _percentile(
            guidance_client.latencies_ms[guidance_base_latency_count:] if guidance_client is not None else [], 95.0
        ),
        "guidance_cache_hit_rate": float(
            (
                (guidance_client.cache_hit_count - guidance_base_cache_hit_count)
                / max(
                    (guidance_client.query_count - guidance_base_query_count)
                    + (guidance_client.cache_hit_count - guidance_base_cache_hit_count),
                    1,
                )
            )
            if guidance_client is not None
            else 0.0
        ),
        "guidance_non_fallback_rate": float(
            (
                (
                    (guidance_client.success_count - guidance_base_success_count)
                    + (guidance_client.cache_hit_count - guidance_base_cache_hit_count)
                )
                / max(
                    (guidance_client.query_count - guidance_base_query_count)
                    + (guidance_client.cache_hit_count - guidance_base_cache_hit_count),
                    1,
                )
            )
            if guidance_client is not None
            else 0.0
        ),
        "route_guidance_tracking_error_m": float(
            np.mean(controller.route_guidance_tracking_errors) if controller.route_guidance_tracking_errors else 0.0
        ),
        "route_guidance_source": str((controller._last_route_guidance or {}).get("route_guidance_source", "disabled")),
        "guidance_replan_urgency": float((controller._last_route_guidance or {}).get("replan_urgency", 0.0)),
        "guidance_waypoint_bias_y": float((controller._last_route_guidance or {}).get("waypoint_bias_y", 0.0)),
        "guidance_dynamic_clearance_margin_m": float(
            (controller._last_route_guidance or {}).get("dynamic_clearance_margin_m", 0.0)
        ),
        "shield_activation_count": int(controller.shield_activation_count),
        "shield_activation_ratio": float(controller.shield_activation_count / max(len(rows), 1)),
        "moving_gates_enabled": bool(moving_gates_enabled),
        "moving_gate_amplitude_m": float(moving_gate_amplitude_m),
        "moving_gate_speed_hz": float(moving_gate_speed_hz),
        "moving_gate_speed_mps": float(moving_gate_speed_mps),
        "dynamic_obstacle_update_count": int(dynamic_obstacle_updates),
        "actual_gate_motion_range_m": float(max(gate_motion_ranges) if gate_motion_ranges else 0.0),
        "actual_gate_motion_range_x_m": float(max(gate_motion_ranges_x) if gate_motion_ranges_x else 0.0),
        "actual_gate_motion_range_y_m": float(max(gate_motion_ranges_y) if gate_motion_ranges_y else 0.0),
        "actual_gate_motion_range_mean_m": float(np.mean(gate_motion_ranges) if gate_motion_ranges else 0.0),
        "actual_gate_max_displacement_m": float(max(gate_max_displacements) if gate_max_displacements else 0.0),
        "morph_min_distance_m": None,
        "morph_note": "单机场景不适用",
        "final_state": asdict(final_state),
    }

    with (episode_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        with (episode_dir / "live_gate_centers_timeseries.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        {
                            "step": int(row["step"]),
                            "t_sec": float(row["t_sec"]),
                            "gate_centers_xy": json.loads(str(row.get("live_gate_centers_xy_json") or "[]")),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    _write_json(episode_dir / "metrics.json", metrics)
    if guidance_client is not None:
        guidance_jsonl_path = episode_dir / "route_guidance_timeseries.jsonl"
        with guidance_jsonl_path.open("w", encoding="utf-8") as stream:
            for record in guidance_client.guidance_records[guidance_base_record_count:]:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_json(
        episode_dir / "replay_summary.json",
        {
            "replay_type": "2d_gate_density_rollout",
            "isaaclab_visual_replay_checked": False,
            "isaaclab_replay_note": "This script saves numeric 2D rollout files only.",
            "start_xyz": list(task_payload.get("start_xyz", START_XYZ)),
            "goal_xyz": list(task_payload.get("goal_xyz", GOAL_XYZ)),
            "gate_count": int(args.gate_count),
            "random_yaw": bool(args.random_yaw),
            "seed": int(args.seed),
            "episode_index": int(episode_index),
            "done_reason": done_reason,
            "success": bool(success),
            "height_contract_passed": bool(height_contract_passed),
            "corridor_through_success": bool(corridor_through_success),
            "side_bypass_failure": bool(side_bypass_failure),
            "height_escape_failure": bool(height_escape_failure),
            "corridor_miss_failure": bool(corridor_miss_failure),
            "route_guidance_enabled": bool(args.enable_route_guidance),
            "guidance_shadow_mode": bool(args.guidance_shadow_mode),
            "guidance_visible": bool(args.guidance_visible),
            "guidance_query_count": metrics["guidance_query_count"],
            "guidance_failure_count": metrics["guidance_failure_count"],
            "actual_gate_motion_range_m": metrics["actual_gate_motion_range_m"],
            "actual_gate_max_displacement_m": metrics["actual_gate_max_displacement_m"],
        },
    )
    return metrics

