"""Live IsaacLab preview helpers for the paper Experiment 3 training loop."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from shared.visualization.scene_isaaclab import (
    REPLAY_DRONE_BEACON_Z_OFFSET_M,
    REPLAY_DRONE_HALO_Z_OFFSET_M,
    REPLAY_DRONE_MAST_Z_OFFSET_M,
    apply_pose,
    setup_replay_scene,
    update_overview_replay_camera,
    update_replay_camera,
)


def run_live_snapshot_preview(
    *,
    snapshot_path: str | Path,
    experiment_config,
    sim,
    poll_interval_s: float = 0.1,
    camera_mode: str = "picture_in_picture",
    follow_agent_index: int | None = 0,
) -> None:
    """Open one real-3D scene and continuously mirror the latest training snapshot."""

    import isaaclab.sim as sim_utils

    resolved_snapshot_path = Path(snapshot_path)
    payload = _wait_for_snapshot(resolved_snapshot_path)
    env_config = experiment_config.environment
    fixed_height_m = float(payload.get("fixed_height_m") or env_config.fixed_height_m)
    max_agents = int(payload.get("max_agents") or experiment_config.max_agents_soft)
    start_xy = tuple(payload.get("start_xy") or (env_config.start_x_m, 0.0))
    goal_xy = tuple(payload.get("goal_xy") or (env_config.goal_x_m, 0.0))
    route_waypoints_xy = tuple(tuple(point) for point in payload.get("path_waypoints") or ())
    route_waypoint_names = tuple(f"P{idx + 1}" for idx, _point in enumerate(route_waypoints_xy))

    sim_utils.create_new_stage()
    scene_rig = setup_replay_scene(
        sim=sim,
        drone_count=max_agents,
        fixed_height_m=fixed_height_m,
        drone_scale=(2.8, 2.8, 2.8),
        render_real_gate=bool(experiment_config.scene.render_real_gate),
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=env_config.world_x_bounds_m,
        world_y_bounds_m=env_config.world_y_bounds_m,
        gate_post_collision_zone_radius_m=(
            env_config.drone_radius_m + 0.8 if bool(experiment_config.scene.render_real_gate) else None
        ),
        gate_post_safety_zone_radius_m=(
            env_config.drone_radius_m + env_config.safety_clearance_m + 0.8
            if bool(experiment_config.scene.render_real_gate)
            else None
        ),
        start_zone_radius_m=3.4,
        goal_zone_radius_m=env_config.goal_radius_m,
        route_waypoints_xy=route_waypoints_xy if len(route_waypoints_xy) >= 3 else None,
        route_waypoint_names=route_waypoint_names if len(route_waypoint_names) >= 3 else None,
        drone_highlight_radius_m=0.68,
        drone_safety_radius_m=env_config.inter_agent_safe_distance_m,
    )

    sim.reset()
    for _ in range(4):
        sim.step(render=True)

    previous_yaws = [0.0 for _ in range(max_agents)]
    hidden_position = (0.0, 0.0, -20.0)
    hidden_beacon_position = (0.0, 0.0, -22.0)
    camera_focus_xy = np.asarray(start_xy, dtype=np.float32)
    camera_heading_xy = np.asarray((1.0, 0.0), dtype=np.float32)
    last_mtime_ns = -1

    while True:
        try:
            stat = resolved_snapshot_path.stat()
            if stat.st_mtime_ns != last_mtime_ns:
                payload = _read_snapshot_json(resolved_snapshot_path)
                if payload is None:
                    time.sleep(max(float(poll_interval_s), 0.01))
                    sim.step(render=True)
                    continue
                previous_yaws, camera_focus_xy, camera_heading_xy = _apply_live_snapshot(
                    payload=payload,
                    scene_rig=scene_rig,
                    max_agents=max_agents,
                    fixed_height_m=fixed_height_m,
                    previous_yaws=previous_yaws,
                    hidden_position=hidden_position,
                    hidden_beacon_position=hidden_beacon_position,
                    camera_focus_xy=camera_focus_xy,
                    camera_heading_xy=camera_heading_xy,
                    camera_mode=str(camera_mode or "picture_in_picture"),
                    follow_agent_index=follow_agent_index,
                    sim=sim,
                )
                last_mtime_ns = stat.st_mtime_ns
        except FileNotFoundError:
            pass

        sim.step(render=True)
        time.sleep(max(float(poll_interval_s), 0.01))


def _wait_for_snapshot(snapshot_path: Path, timeout_s: float = 30.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if snapshot_path.exists():
            payload = _read_snapshot_json(snapshot_path)
            if payload is not None:
                return payload
        time.sleep(0.1)
    raise FileNotFoundError(f"Live preview snapshot did not appear within {timeout_s:.1f}s: {snapshot_path}")


def _read_snapshot_json(snapshot_path: Path) -> dict[str, object] | None:
    try:
        raw_text = snapshot_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if not raw_text.strip():
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _apply_live_snapshot(
    *,
    payload: dict[str, object],
    scene_rig,
    max_agents: int,
    fixed_height_m: float,
    previous_yaws: list[float],
    hidden_position: tuple[float, float, float],
    hidden_beacon_position: tuple[float, float, float],
    camera_focus_xy: np.ndarray,
    camera_heading_xy: np.ndarray,
    camera_mode: str,
    follow_agent_index: int | None,
    sim,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    num_agents = int(payload.get("num_agents") or 0)
    positions_xy = list(payload.get("positions_xy") or [])
    velocities_xy = list(payload.get("velocities_xy") or [])
    yaws_rad = list(payload.get("yaws_rad") or [])
    next_yaws = list(previous_yaws)
    for agent_idx in range(max_agents):
        if agent_idx < num_agents:
            position_xy = positions_xy[agent_idx]
            velocity_xy = velocities_xy[agent_idx] if agent_idx < len(velocities_xy) else (0.0, 0.0)
            yaw_rad = (
                float(yaws_rad[agent_idx])
                if agent_idx < len(yaws_rad)
                else _resolve_yaw_from_velocity(velocity_xy, previous_yaws[agent_idx])
            )
            next_yaws[agent_idx] = yaw_rad
            position_xyz = (
                float(position_xy[0]),
                float(position_xy[1]),
                float(payload.get("fixed_height_m") or fixed_height_m),
            )
            apply_pose(scene_rig.drone_handles[agent_idx], position_xyz=position_xyz, yaw_rad=yaw_rad)
            apply_pose(
                scene_rig.drone_halo_handles[agent_idx],
                position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_HALO_Z_OFFSET_M),
                yaw_rad=0.0,
            )
            apply_pose(
                scene_rig.drone_mast_handles[agent_idx],
                position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_MAST_Z_OFFSET_M),
                yaw_rad=0.0,
            )
            apply_pose(
                scene_rig.drone_beacon_handles[agent_idx],
                position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_BEACON_Z_OFFSET_M),
                yaw_rad=0.0,
            )
            if scene_rig.drone_safety_handles:
                apply_pose(scene_rig.drone_safety_handles[agent_idx], position_xyz=position_xyz, yaw_rad=0.0)
        else:
            apply_pose(scene_rig.drone_handles[agent_idx], position_xyz=hidden_position, yaw_rad=0.0)
            apply_pose(scene_rig.drone_halo_handles[agent_idx], position_xyz=hidden_position, yaw_rad=0.0)
            apply_pose(scene_rig.drone_mast_handles[agent_idx], position_xyz=hidden_beacon_position, yaw_rad=0.0)
            apply_pose(scene_rig.drone_beacon_handles[agent_idx], position_xyz=hidden_beacon_position, yaw_rad=0.0)
            if scene_rig.drone_safety_handles:
                apply_pose(scene_rig.drone_safety_handles[agent_idx], position_xyz=hidden_position, yaw_rad=0.0)

    _apply_guidance_pose(
        handle_list=scene_rig.slow_guidance_handles,
        guidance=payload.get("route_plan_guidance"),
        focus_xy=payload.get("virtual_center_xy"),
        fixed_height_m=float(payload.get("fixed_height_m") or fixed_height_m),
        hidden_position=hidden_position,
    )
    _apply_guidance_pose(
        handle_list=scene_rig.route_guidance_handles,
        guidance=payload.get("route_guidance"),
        focus_xy=payload.get("virtual_center_xy"),
        fixed_height_m=float(payload.get("fixed_height_m") or fixed_height_m),
        hidden_position=hidden_position,
    )

    focus_xy = tuple(payload.get("virtual_center_xy") or payload.get("start_xy") or (0.0, 0.0))
    heading_xy = tuple(payload.get("lookahead_heading_xy") or (1.0, 0.0))
    if follow_agent_index is not None and num_agents > 0:
        tracked_index = max(0, min(int(follow_agent_index), num_agents - 1))
        focus_xy = tuple(positions_xy[tracked_index])
    next_focus_xy = 0.9 * camera_focus_xy + 0.1 * np.asarray(focus_xy, dtype=np.float32)
    next_heading_xy = _blend_heading(camera_heading_xy, heading_xy)
    subject_scale_m = _subject_scale_m(positions_xy[:num_agents]) if camera_mode != "follow" else 6.0
    update_replay_camera(
        sim,
        camera_prim_path=scene_rig.follow_camera_prim_path,
        focus_xy=(float(next_focus_xy[0]), float(next_focus_xy[1])),
        heading_xy=(float(next_heading_xy[0]), float(next_heading_xy[1])),
        fixed_height_m=float(payload.get("fixed_height_m") or fixed_height_m),
        subject_scale_m=subject_scale_m,
    )
    update_overview_replay_camera(
        sim,
        camera_prim_path=scene_rig.overview_camera_prim_path,
        focus_xy=(float(next_focus_xy[0]), float(next_focus_xy[1])),
        fixed_height_m=float(payload.get("fixed_height_m") or fixed_height_m),
        world_x_bounds_m=tuple(payload.get("world_x_bounds_m") or (-55.0, 55.0)),
        world_y_bounds_m=tuple(payload.get("world_y_bounds_m") or (-20.0, 20.0)),
    )
    return next_yaws, next_focus_xy, next_heading_xy


def _resolve_yaw_from_velocity(velocity_xy: tuple[float, float] | list[float], fallback_yaw_rad: float) -> float:
    vx = float(velocity_xy[0]) if len(velocity_xy) > 0 else 0.0
    vy = float(velocity_xy[1]) if len(velocity_xy) > 1 else 0.0
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return float(fallback_yaw_rad)
    return float(np.arctan2(vy, vx))


def _blend_heading(previous_heading_xy: np.ndarray, current_heading_xy: tuple[float, float] | list[float]) -> np.ndarray:
    target = np.asarray(current_heading_xy, dtype=np.float32)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-6:
        return previous_heading_xy
    target = target / norm
    blended = 0.85 * previous_heading_xy + 0.15 * target
    blended_norm = float(np.linalg.norm(blended))
    if blended_norm <= 1e-6:
        return target
    return blended / blended_norm


def _subject_scale_m(positions_xy: list[object]) -> float:
    if not positions_xy:
        return 12.0
    points = np.asarray(positions_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] <= 1:
        return 8.0
    spread = points.max(axis=0) - points.min(axis=0)
    return float(max(8.0, np.linalg.norm(spread) + 6.0))


def _apply_guidance_pose(
    *,
    handle_list,
    guidance,
    focus_xy,
    fixed_height_m: float,
    hidden_position: tuple[float, float, float],
) -> None:
    if not handle_list:
        return
    handle = handle_list[0]
    if not isinstance(guidance, dict) or focus_xy is None:
        apply_pose(handle, position_xyz=hidden_position, yaw_rad=0.0)
        return
    center_xy = tuple(float(value) for value in focus_xy)
    apply_pose(
        handle,
        position_xyz=(
            center_xy[0] + float(guidance.get("target_rel_x", 0.0)) * 50.0,
            center_xy[1] + float(guidance.get("target_rel_y", 0.0)) * 50.0,
            float(fixed_height_m) + 0.3,
        ),
        yaw_rad=0.0,
    )

