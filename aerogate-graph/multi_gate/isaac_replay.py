"""IsaacLab replay renderer for multi-agent aerogate_graph trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from multi_gate.configs.experiment_config import MULTI_EXPERIMENT_CONFIG, MultiExperimentConfig
from shared.core.collision_2d import GateObstacleMap2D
from shared.runtime.artifacts import write_json
from shared.configs.global_config import GLOBAL_CONFIG
from shared.visualization.scene_isaaclab import (
    REPLAY_DRONE_BEACON_Z_OFFSET_M,
    REPLAY_DRONE_HALO_Z_OFFSET_M,
    REPLAY_DRONE_MAST_Z_OFFSET_M,
    apply_pose,
    build_pose_handle,
    create_rgb_annotator,
    destroy_rgb_annotator,
    setup_replay_scene,
    update_overview_replay_camera,
    update_replay_camera,
)


DEFAULT_GATE_USD = Path(__file__).resolve().parents[1] / "assets" / "gate" / "gate.usd"
# Measured from assets/gate/gate.usd via IsaacLab BBoxCache. The prior code
# passed scale to UsdFileCfg.func(), which IsaacLab ignores; the unscaled gate
# rendered as roughly 2.13 m tall, making fixed-height drones at z=4 visibly
# fly over the gates.
GATE_NATIVE_VISUAL_HEIGHT_M = 2.1335996309757235
GATE_NATIVE_VISUAL_HALF_WIDTH_M = 2.0


def render_multi_trajectory_isaaclab(
    *,
    trajectory_path: str | Path,
    report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    mp4_path: str | Path | None = None,
    fps: int = 10,
    resolution: tuple[int, int] = (1280, 720),
    hold_initial_frames: int = 18,
    hold_final_frames: int = 12,
    real_time: bool = False,
    camera_mode: str = "picture_in_picture",
    follow_agent_index: int | None = None,
    route_waypoints_xy: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None,
    route_waypoint_names: tuple[str, ...] | list[str] | None = None,
    experiment_config: MultiExperimentConfig | None = None,
) -> dict[str, object]:
    """Render one saved multi-agent trajectory inside an IsaacLab scene."""

    import isaaclab.sim as sim_utils

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    trajectory_file = Path(trajectory_path)
    payload = _read_json(trajectory_file)
    trajectory = list(payload.get("trajectory") or [])
    if not trajectory:
        raise ValueError(f"Trajectory file does not contain any frames: {trajectory_file}")

    report = _read_json(Path(report_path)) if report_path is not None else None
    fixed_height_m = float(payload.get("fixed_height_m") or selected_config.environment.fixed_height_m)
    max_agents = max(int(frame.get("num_agents") or 0) for frame in trajectory)
    if max_agents <= 0:
        raise ValueError("Multi-agent trajectory does not record any active agents.")

    start_xy, goal_xy = _resolve_start_goal_xy(trajectory=trajectory, report=report)
    resolved_route_waypoints_xy = _coerce_route_waypoints(route_waypoints_xy)
    if not resolved_route_waypoints_xy:
        resolved_route_waypoints_xy = _resolve_recorded_route_waypoints(
            payload=payload,
            report=report,
            trajectory=trajectory,
        )
    resolved_route_waypoint_names = _resolve_route_waypoint_names(
        route_waypoint_names=route_waypoint_names,
        route_waypoints_xy=resolved_route_waypoints_xy,
        payload=payload,
        report=report,
    )
    env_config = selected_config.environment
    resolved_camera_mode = _resolve_camera_mode(camera_mode)
    resolved_follow_agent_index = _resolve_follow_agent_index(
        follow_agent_index=follow_agent_index,
        max_agents=max_agents,
        report=report,
    )
    goal_radius_m = float(payload.get("goal_radius_m") or env_config.goal_radius_m)
    render_real_gate = bool(selected_config.scene.render_real_gate)
    gate_post_collision_zone_radius_m = None
    gate_post_safety_zone_radius_m = None
    nearest_gate_post_distance_to_goal_m = None
    if render_real_gate:
        gate_post_collision_zone_radius_m = (
            GLOBAL_CONFIG.default_gate_post_collision_radius_m * float(env_config.gate_post_radius_scale)
            + env_config.drone_radius_m
        )
        gate_post_safety_zone_radius_m = gate_post_collision_zone_radius_m + env_config.safety_clearance_m
        obstacle_map = GateObstacleMap2D.from_gate(gate_post_radius_scale=env_config.gate_post_radius_scale)
        nearest_gate_post_distance_to_goal_m = min(
            float(np.hypot(goal_xy[0] - obstacle.center_xy[0], goal_xy[1] - obstacle.center_xy[1]))
            for obstacle in obstacle_map.obstacles
        )

    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / max(int(fps), 1), render_interval=1))
    scene_rig = setup_replay_scene(
        sim=sim,
        drone_count=max_agents,
        fixed_height_m=fixed_height_m,
        drone_scale=(0.9, 0.9, 0.9),
        render_real_gate=render_real_gate,
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=env_config.world_x_bounds_m,
        world_y_bounds_m=env_config.world_y_bounds_m,
        gate_post_collision_zone_radius_m=gate_post_collision_zone_radius_m,
        gate_post_safety_zone_radius_m=gate_post_safety_zone_radius_m,
        start_zone_radius_m=3.4,
        goal_zone_radius_m=goal_radius_m,
        route_waypoints_xy=resolved_route_waypoints_xy,
        route_waypoint_names=resolved_route_waypoint_names,
        drone_highlight_radius_m=0.68,
        drone_safety_radius_m=None,
        show_drone_debug_overlays=False,
        show_scene_markers=False,
        show_high_scene_markers=False,
    )
    gate_handles, dynamic_gate_summary = _spawn_dynamic_gate_assets(payload)

    resolved_output_dir = Path(output_dir) if output_dir is not None else trajectory_file.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_mp4_path = Path(mp4_path) if mp4_path is not None else None
    if resolved_mp4_path is not None:
        resolved_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    overview_annotator = None
    overview_render_product = None
    follow_annotator = None
    follow_render_product = None
    video_writer = None
    frames_requested = int(len(trajectory) + max(int(hold_initial_frames), 0) + max(int(hold_final_frames), 0))
    frames_written = 0
    latest_overlay_frame = None
    top_global_orthographic_size_m = _compute_top_global_orthographic_size(
        trajectory=trajectory,
        payload=payload,
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=env_config.world_x_bounds_m,
        world_y_bounds_m=env_config.world_y_bounds_m,
        aspect_ratio=float(resolution[0]) / max(float(resolution[1]), 1.0),
    )
    if resolved_camera_mode == "top_global":
        _set_camera_orthographic(
            scene_rig.overview_camera_prim_path,
            orthographic_size_m=top_global_orthographic_size_m,
        )
    elif resolved_camera_mode == "top_centroid_follow":
        _set_camera_orthographic(
            scene_rig.follow_camera_prim_path,
            orthographic_size_m=_top_centroid_orthographic_size_m(trajectory[0]),
        )
    try:
        if resolved_mp4_path is not None:
            import imageio.v2 as imageio

            overview_annotator, overview_render_product = create_rgb_annotator(
                camera_prim_path=(
                    scene_rig.follow_camera_prim_path
                    if resolved_camera_mode in {"follow", "height_audit", "top_centroid_follow"}
                    else scene_rig.overview_camera_prim_path
                ),
                resolution=resolution,
            )
            if resolved_camera_mode == "picture_in_picture":
                follow_annotator, follow_render_product = create_rgb_annotator(
                    camera_prim_path=scene_rig.follow_camera_prim_path,
                    resolution=resolution,
                )
            video_writer = imageio.get_writer(
                str(resolved_mp4_path),
                fps=max(int(fps), 1),
                codec="libx264",
            )

        sim.reset()
        for _ in range(5):
            sim.step(render=True)

        previous_yaws = [0.0 for _ in range(max_agents)]
        hidden_position = (0.0, 0.0, -20.0)
        hidden_beacon_position = (0.0, 0.0, -22.0)
        camera_focus_xy = np.asarray(start_xy, dtype=np.float32)
        camera_heading_xy = np.asarray((1.0, 0.0), dtype=np.float32)
        top_global_pose = _compute_top_global_camera_pose(
            trajectory=trajectory,
            payload=payload,
            start_xy=start_xy,
            goal_xy=goal_xy,
            fixed_height_m=fixed_height_m,
            world_x_bounds_m=env_config.world_x_bounds_m,
            world_y_bounds_m=env_config.world_y_bounds_m,
        )
        first_frame = trajectory[0]
        _apply_dynamic_gate_frame_pose(
            frame=first_frame,
            payload=payload,
            gate_handles=gate_handles,
            hidden_position=(0.0, 0.0, -30.0),
        )
        previous_yaws, camera_focus_xy, camera_heading_xy = _apply_multi_frame_pose(
            frame=first_frame,
            scene_rig=scene_rig,
            fixed_height_m=fixed_height_m,
            max_agents=max_agents,
            previous_yaws=previous_yaws,
            hidden_position=hidden_position,
            hidden_beacon_position=hidden_beacon_position,
            camera_focus_xy=camera_focus_xy,
            camera_heading_xy=camera_heading_xy,
            follow_agent_index=resolved_follow_agent_index if resolved_camera_mode == "follow" else None,
            sim=sim,
        )
        if resolved_camera_mode == "height_audit":
            _update_height_audit_camera(
                sim=sim,
                camera_prim_path=scene_rig.follow_camera_prim_path,
                frame=first_frame,
                fixed_height_m=fixed_height_m,
            )
        elif resolved_camera_mode == "top_global":
            _set_camera_pose(sim, scene_rig.overview_camera_prim_path, *top_global_pose)
        elif resolved_camera_mode == "top_centroid_follow":
            _update_top_centroid_follow_camera(
                sim=sim,
                camera_prim_path=scene_rig.follow_camera_prim_path,
                frame=first_frame,
                focus_xy=(float(camera_focus_xy[0]), float(camera_focus_xy[1])),
                fixed_height_m=fixed_height_m,
                orthographic_size_m=_top_centroid_orthographic_size_m(first_frame),
            )
        for _ in range(max(int(hold_initial_frames), 0)):
            sim.step(render=True)
            frame_written, latest_overlay_frame = _append_rgb_frame(
                video_writer=video_writer,
                annotator=overview_annotator,
                overlay_annotator=follow_annotator,
                overlay_follow_view=resolved_camera_mode == "picture_in_picture",
                latest_overlay_frame=latest_overlay_frame,
            )
            frames_written += int(frame_written)
            if real_time:
                time.sleep(1.0 / max(int(fps), 1))
        for frame in trajectory:
            _apply_dynamic_gate_frame_pose(
                frame=frame,
                payload=payload,
                gate_handles=gate_handles,
                hidden_position=(0.0, 0.0, -30.0),
            )
            previous_yaws, camera_focus_xy, camera_heading_xy = _apply_multi_frame_pose(
                frame=frame,
                scene_rig=scene_rig,
                fixed_height_m=fixed_height_m,
                max_agents=max_agents,
                previous_yaws=previous_yaws,
                hidden_position=hidden_position,
                hidden_beacon_position=hidden_beacon_position,
                camera_focus_xy=camera_focus_xy,
                camera_heading_xy=camera_heading_xy,
                follow_agent_index=resolved_follow_agent_index if resolved_camera_mode == "follow" else None,
                sim=sim,
            )
            if resolved_camera_mode == "height_audit":
                _update_height_audit_camera(
                    sim=sim,
                    camera_prim_path=scene_rig.follow_camera_prim_path,
                    frame=frame,
                    fixed_height_m=fixed_height_m,
                )
            elif resolved_camera_mode == "top_global":
                _set_camera_pose(sim, scene_rig.overview_camera_prim_path, *top_global_pose)
            elif resolved_camera_mode == "top_centroid_follow":
                _update_top_centroid_follow_camera(
                    sim=sim,
                    camera_prim_path=scene_rig.follow_camera_prim_path,
                    frame=frame,
                    focus_xy=(float(camera_focus_xy[0]), float(camera_focus_xy[1])),
                    fixed_height_m=fixed_height_m,
                    orthographic_size_m=_top_centroid_orthographic_size_m(frame),
                )
            sim.step(render=True)
            frame_written, latest_overlay_frame = _append_rgb_frame(
                video_writer=video_writer,
                annotator=overview_annotator,
                overlay_annotator=follow_annotator,
                overlay_follow_view=resolved_camera_mode == "picture_in_picture",
                latest_overlay_frame=latest_overlay_frame,
            )
            frames_written += int(frame_written)
            if real_time:
                time.sleep(1.0 / max(int(fps), 1))

        for _ in range(max(int(hold_final_frames), 0)):
            sim.step(render=True)
            frame_written, latest_overlay_frame = _append_rgb_frame(
                video_writer=video_writer,
                annotator=overview_annotator,
                overlay_annotator=follow_annotator,
                overlay_follow_view=resolved_camera_mode == "picture_in_picture",
                latest_overlay_frame=latest_overlay_frame,
            )
            frames_written += int(frame_written)
            if real_time:
                time.sleep(1.0 / max(int(fps), 1))
    finally:
        if video_writer is not None:
            video_writer.close()
        if overview_annotator is not None and overview_render_product is not None:
            destroy_rgb_annotator(overview_annotator, overview_render_product)
        if follow_annotator is not None and follow_render_product is not None:
            destroy_rgb_annotator(follow_annotator, follow_render_product)

    summary = {
        "mode": "multi_isaaclab_replay",
        "trajectory_path": str(trajectory_file),
        "report_path": None if report_path is None else str(Path(report_path)),
        "output_dir": str(resolved_output_dir),
        "mp4_path": None if resolved_mp4_path is None else str(resolved_mp4_path),
        "frames_rendered": frames_requested,
        "frames_written": int(frames_written),
        "frame_drops": int(max(frames_requested - frames_written, 0)),
        "fps": int(fps),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "hold_initial_frames": int(hold_initial_frames),
        "fixed_height_m": float(fixed_height_m),
        "drone_asset_path": str(GLOBAL_CONFIG.drone_asset_file),
        "gate_layout_path": str(GLOBAL_CONFIG.gate_layout_file),
        "camera_mode": resolved_camera_mode,
        "overlay_follow_view": resolved_camera_mode == "picture_in_picture",
        "height_audit_camera": resolved_camera_mode == "height_audit",
        "top_down_camera": resolved_camera_mode in {"top_global", "top_centroid_follow"},
        "formal_drone_debug_overlays": False,
        "formal_scene_markers": False,
        "formal_high_scene_markers": False,
        "formal_drone_safety_overlays": False,
        "follow_agent_index": (
            int(resolved_follow_agent_index) if resolved_camera_mode in {"follow", "picture_in_picture"} else None
        ),
        "drone_prim_paths": [handle.prim_path for handle in scene_rig.drone_handles],
        "world_x_bounds_m": list(env_config.world_x_bounds_m),
        "world_y_bounds_m": list(env_config.world_y_bounds_m),
        "goal_radius_m": float(goal_radius_m),
        "render_real_gate": bool(render_real_gate),
        "gate_post_collision_zone_radius_m": (
            None if gate_post_collision_zone_radius_m is None else float(gate_post_collision_zone_radius_m)
        ),
        "gate_post_safety_zone_radius_m": (
            None if gate_post_safety_zone_radius_m is None else float(gate_post_safety_zone_radius_m)
        ),
        "nearest_gate_post_distance_to_goal_m": (
            None if nearest_gate_post_distance_to_goal_m is None else float(nearest_gate_post_distance_to_goal_m)
        ),
        "start_xy": list(start_xy),
        "goal_xy": list(goal_xy),
        "route_waypoints_xy": [list(point) for point in resolved_route_waypoints_xy],
        "route_waypoint_names": list(resolved_route_waypoint_names),
        "max_agents": int(max_agents),
        "dynamic_gate_replay": dynamic_gate_summary,
        "done_reason": None if report is None else report.get("done_reason"),
        "success": None if report is None else report.get("success"),
    }
    summary_path = resolved_output_dir / "isaaclab_replay_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def _spawn_dynamic_gate_assets(payload: dict[str, object]) -> tuple[list[object], dict[str, object]]:
    """Spawn replay Gate USDs from the same metadata recorded by multi replay."""

    metadata = payload.get("dynamic_gate_density")
    if not isinstance(metadata, dict) or not bool(metadata.get("enabled")):
        return [], {"enabled": False, "gate_count": 0, "reason": "no_dynamic_gate_metadata"}
    gates = list(metadata.get("gates") or [])
    gate_count = int(metadata.get("gate_count") or len(gates))
    if gate_count <= 0:
        return [], {"enabled": True, "gate_count": 0, "reason": "zero_gate_scene"}

    import isaaclab.sim as sim_utils

    gate_usd = Path(str(metadata.get("gate_asset_path") or DEFAULT_GATE_USD))
    if not gate_usd.exists():
        raise FileNotFoundError(f"Gate USD asset is missing: {gate_usd}")
    base_centers = _resolve_gate_centers_from_metadata(metadata)
    yaws_rad = _resolve_gate_yaws(metadata, count=gate_count)
    gate_scale = metadata.get("gate_visual_scale_xyz")
    if isinstance(gate_scale, (list, tuple)) and len(gate_scale) >= 2:
        recorded_scale_xy = (float(gate_scale[0]), float(gate_scale[1]))
        recorded_scale_z = float(gate_scale[2]) if len(gate_scale) >= 3 else None
    else:
        recorded_scale_xy = None
        recorded_scale_z = None
    metadata_config = dict(metadata.get("config") or {})
    gate_half_width_m = float(
        metadata.get("gate_half_width_m")
        or metadata_config.get("gate_half_width_m")
        or GATE_NATIVE_VISUAL_HALF_WIDTH_M
    )
    gate_bottom_height_m = float(
        metadata.get("gate_bottom_height_m")
        or metadata_config.get("gate_opening_bottom_height_m")
        or 0.0
    )
    gate_top_height_m = float(
        metadata.get("gate_top_height_m")
        or metadata_config.get("gate_opening_top_height_m")
        or 8.0
    )
    gate_center_height_m = float(
        metadata.get("gate_center_height_m")
        or metadata_config.get("gate_center_height_m")
        or 4.0
    )
    target_gate_height_m = max(gate_top_height_m - gate_bottom_height_m, 1.0e-6)
    resolved_gate_scale_xy = float(gate_half_width_m / max(GATE_NATIVE_VISUAL_HALF_WIDTH_M, 1.0e-6))
    resolved_gate_scale = (
        resolved_gate_scale_xy,
        resolved_gate_scale_xy,
        float(target_gate_height_m / GATE_NATIVE_VISUAL_HEIGHT_M),
    )
    scale_recomputed = (
        recorded_scale_xy is None
        or recorded_scale_z is None
        or abs(recorded_scale_xy[0] - resolved_gate_scale[0]) > 1.0e-4
        or abs(recorded_scale_xy[1] - resolved_gate_scale[1]) > 1.0e-4
        or abs(recorded_scale_z - resolved_gate_scale[2]) > 1.0e-4
    )
    gate_cfg = sim_utils.UsdFileCfg(usd_path=str(gate_usd), scale=resolved_gate_scale)
    handles: list[object] = []
    for gate_idx in range(gate_count):
        rig_path = f"/World/DynamicGates/GateRig_{gate_idx:02d}"
        asset_path = f"{rig_path}/Asset"
        _define_clean_xform(rig_path)
        gate_cfg.func(
            asset_path,
            gate_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        authored_scale = _force_xform_scale(asset_path, resolved_gate_scale)
        _hide_collision_geometry(rig_path)
        _lock_replay_obstacles_static(rig_path)
        handle = build_pose_handle(rig_path)
        center_xy = base_centers[gate_idx] if gate_idx < len(base_centers) else (0.0, 0.0)
        yaw_rad = yaws_rad[gate_idx] if gate_idx < len(yaws_rad) else 0.0
        apply_pose(handle, position_xyz=(float(center_xy[0]), float(center_xy[1]), 0.0), yaw_rad=float(yaw_rad))
        handles.append(handle)
    sim_utils.update_stage()

    motion = _dynamic_gate_motion_summary(payload, metadata, gate_count=gate_count)
    speed_mps = float(metadata.get("moving_gate_speed_mps") or dict(metadata.get("config") or {}).get("moving_gate_speed_mps") or 0.0)
    if gate_count > 0 and speed_mps > 1.0e-6 and float(motion["max_center_motion_m"]) < 0.005:
        raise RuntimeError(
            "Recorded multi-agent dynamic-gate replay has moving_gate_speed_mps > 0 "
            "but live_gate_centers_xy do not move. Refusing to write fake dynamic MP4."
        )
    return handles, {
        "enabled": True,
        "gate_count": int(gate_count),
        "gate_asset_path": str(gate_usd),
        "gate_prim_paths": [str(handle.prim_path) for handle in handles],
        "gate_bottom_height_m": gate_bottom_height_m,
        "gate_top_height_m": gate_top_height_m,
        "gate_center_height_m": gate_center_height_m,
        "gate_half_width_m": gate_half_width_m,
        "gate_native_visual_half_width_m": float(GATE_NATIVE_VISUAL_HALF_WIDTH_M),
        "gate_native_visual_height_m": float(GATE_NATIVE_VISUAL_HEIGHT_M),
        "gate_visual_scale_xyz": [float(value) for value in resolved_gate_scale],
        "authored_gate_visual_scale_xyz": [float(value) for value in authored_scale],
        "gate_visual_scale_recomputed_from_height_contract": bool(scale_recomputed),
        "recorded_gate_visual_scale_xy": None if recorded_scale_xy is None else [float(value) for value in recorded_scale_xy],
        "recorded_gate_visual_scale_z": recorded_scale_z,
        "moving_gate_speed_mps": speed_mps,
        "moving_gate_amplitude_m": float(
            metadata.get("moving_gate_amplitude_m") or dict(metadata.get("config") or {}).get("moving_gate_amplitude_m") or 0.0
        ),
        **motion,
    }


def _apply_dynamic_gate_frame_pose(
    *,
    frame: dict[str, object],
    payload: dict[str, object],
    gate_handles: list[object],
    hidden_position: tuple[float, float, float],
) -> None:
    if not gate_handles:
        return
    metadata = payload.get("dynamic_gate_density")
    metadata = metadata if isinstance(metadata, dict) else {}
    centers_xy = _resolve_gate_centers_for_frame(frame, metadata)
    yaws_rad = _resolve_gate_yaws_for_frame(frame, metadata, count=len(gate_handles))
    for gate_idx, handle in enumerate(gate_handles):
        if gate_idx >= len(centers_xy):
            apply_pose(handle, position_xyz=hidden_position, yaw_rad=0.0)
            continue
        yaw_rad = yaws_rad[gate_idx] if gate_idx < len(yaws_rad) else 0.0
        center_xy = centers_xy[gate_idx]
        apply_pose(handle, position_xyz=(float(center_xy[0]), float(center_xy[1]), 0.0), yaw_rad=float(yaw_rad))


def _resolve_gate_centers_for_frame(
    frame: dict[str, object],
    metadata: dict[str, object],
) -> list[tuple[float, float]]:
    centers = frame.get("live_gate_centers_xy")
    if not centers:
        centers = metadata.get("gate_base_centers_xy")
    resolved: list[tuple[float, float]] = []
    if isinstance(centers, list):
        for item in centers:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            resolved.append((float(item[0]), float(item[1])))
    return resolved


def _resolve_gate_centers_from_metadata(metadata: dict[str, object]) -> list[tuple[float, float]]:
    centers = metadata.get("gate_base_centers_xy")
    if centers:
        return _resolve_gate_centers_for_frame({"live_gate_centers_xy": centers}, {})
    gates = list(metadata.get("gates") or [])
    resolved: list[tuple[float, float]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        center = gate.get("base_center_xy")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            resolved.append((float(center[0]), float(center[1])))
    return resolved


def _resolve_gate_yaws(metadata: dict[str, object], *, count: int) -> list[float]:
    yaws = metadata.get("gate_yaws_rad")
    resolved: list[float] = []
    if isinstance(yaws, list):
        resolved.extend(float(value) for value in yaws[:count])
    if len(resolved) < count:
        gates = list(metadata.get("gates") or [])
        for gate in gates[len(resolved) : count]:
            if isinstance(gate, dict):
                resolved.append(float(gate.get("yaw_rad") or 0.0))
    while len(resolved) < count:
        resolved.append(0.0)
    return resolved


def _resolve_gate_yaws_for_frame(
    frame: dict[str, object],
    metadata: dict[str, object],
    *,
    count: int,
) -> list[float]:
    frame_yaws = frame.get("gate_yaws_rad")
    if isinstance(frame_yaws, list) and frame_yaws:
        values = [float(value) for value in frame_yaws[:count]]
        while len(values) < count:
            values.append(0.0)
        return values
    return _resolve_gate_yaws(metadata, count=count)


def _dynamic_gate_motion_summary(
    payload: dict[str, object],
    metadata: dict[str, object],
    *,
    gate_count: int,
) -> dict[str, object]:
    trajectory = list(payload.get("trajectory") or [])
    first_centers: list[tuple[float, float]] | None = None
    max_motion = 0.0
    frame_count = 0
    for frame in trajectory:
        if not isinstance(frame, dict):
            continue
        centers = _resolve_gate_centers_for_frame(frame, metadata)
        if len(centers) < gate_count:
            continue
        frame_count += 1
        if first_centers is None:
            first_centers = centers
            continue
        for gate_idx in range(min(gate_count, len(first_centers), len(centers))):
            dx = float(centers[gate_idx][0]) - float(first_centers[gate_idx][0])
            dy = float(centers[gate_idx][1]) - float(first_centers[gate_idx][1])
            max_motion = max(max_motion, float(math.hypot(dx, dy)))
    return {
        "live_gate_frame_count": int(frame_count),
        "max_center_motion_m": float(max_motion),
    }


def _define_clean_xform(prim_path: str) -> None:
    from pxr import UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    current = ""
    for part in [item for item in str(prim_path).split("/") if item]:
        current += f"/{part}"
        prim = stage.GetPrimAtPath(current)
        if not prim or not prim.IsValid():
            stage.DefinePrim(current, "Xform")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Failed to define Xform: {prim_path}")
    UsdGeom.Xformable(prim).ClearXformOpOrder()


def _force_xform_scale(prim_path: str, scale_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Author an explicit USD scale op so replay visuals match 2D gate geometry."""

    from pxr import Gf, UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Failed to resolve prim for scale authoring: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xformable.AddScaleOp()
    authored = tuple(float(value) for value in scale_xyz)
    scale_op.Set(Gf.Vec3f(*authored))
    return authored


def _hide_collision_geometry(root_prim_path: str) -> None:
    from pxr import Usd, UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        name_lower = prim.GetName().lower()
        if not any(token in name_lower for token in ("collider", "collision", "col_", "_col")):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()


def _lock_replay_obstacles_static(root_prim_path: str) -> None:
    from pxr import Usd

    import omni.usd

    try:
        from pxr import UsdPhysics
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        try:
            api.CreateRigidBodyEnabledAttr(False)
        except Exception:
            pass
        try:
            api.CreateKinematicEnabledAttr(True)
        except Exception:
            pass


def _apply_multi_frame_pose(
    *,
    frame: dict[str, object],
    scene_rig,
    fixed_height_m: float,
    max_agents: int,
    previous_yaws: list[float],
    hidden_position: tuple[float, float, float],
    hidden_beacon_position: tuple[float, float, float],
    camera_focus_xy: np.ndarray,
    camera_heading_xy: np.ndarray,
    follow_agent_index: int | None,
    sim,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    num_agents = int(frame.get("num_agents") or 0)
    positions_xy = list(frame.get("positions_xy") or [])
    velocities_xy = list(frame.get("velocities_xy") or [])
    yaws_rad = list(frame.get("yaws_rad") or [])
    next_yaws = list(previous_yaws)
    for agent_idx in range(max_agents):
        if agent_idx < num_agents:
            position_xy = positions_xy[agent_idx]
            velocity_xy = velocities_xy[agent_idx] if agent_idx < len(velocities_xy) else (0.0, 0.0)
            if agent_idx < len(yaws_rad):
                yaw_rad = float(yaws_rad[agent_idx])
            else:
                yaw_rad = _resolve_yaw_from_velocity(
                    velocity_xy=velocity_xy,
                    fallback_yaw_rad=previous_yaws[agent_idx],
                )
            next_yaws[agent_idx] = yaw_rad
            position_xyz = (
                float(position_xy[0]),
                float(position_xy[1]),
                float(frame.get("fixed_height_m") or fixed_height_m),
            )
            apply_pose(
                scene_rig.drone_handles[agent_idx],
                position_xyz=position_xyz,
                yaw_rad=yaw_rad,
            )
            if scene_rig.drone_halo_handles:
                apply_pose(
                    scene_rig.drone_halo_handles[agent_idx],
                    position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_HALO_Z_OFFSET_M),
                    yaw_rad=0.0,
                )
            if scene_rig.drone_mast_handles:
                apply_pose(
                    scene_rig.drone_mast_handles[agent_idx],
                    position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_MAST_Z_OFFSET_M),
                    yaw_rad=0.0,
                )
            if scene_rig.drone_beacon_handles:
                apply_pose(
                    scene_rig.drone_beacon_handles[agent_idx],
                    position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_BEACON_Z_OFFSET_M),
                    yaw_rad=0.0,
                )
            if scene_rig.drone_safety_handles:
                apply_pose(
                    scene_rig.drone_safety_handles[agent_idx],
                    position_xyz=position_xyz,
                    yaw_rad=0.0,
                )
        else:
            apply_pose(scene_rig.drone_handles[agent_idx], position_xyz=hidden_position, yaw_rad=0.0)
            if scene_rig.drone_halo_handles:
                apply_pose(scene_rig.drone_halo_handles[agent_idx], position_xyz=hidden_position, yaw_rad=0.0)
            if scene_rig.drone_mast_handles:
                apply_pose(scene_rig.drone_mast_handles[agent_idx], position_xyz=hidden_beacon_position, yaw_rad=0.0)
            if scene_rig.drone_beacon_handles:
                apply_pose(
                    scene_rig.drone_beacon_handles[agent_idx],
                    position_xyz=hidden_beacon_position,
                    yaw_rad=0.0,
                )
            if scene_rig.drone_safety_handles:
                apply_pose(
                    scene_rig.drone_safety_handles[agent_idx],
                    position_xyz=hidden_position,
                    yaw_rad=0.0,
                )

    _apply_guidance_pose(
        handle_list=scene_rig.slow_guidance_handles,
        guidance=frame.get("route_plan_guidance"),
        focus_xy=frame.get("virtual_center_xy"),
        fixed_height_m=float(frame.get("fixed_height_m") or fixed_height_m),
        hidden_position=hidden_position,
    )
    _apply_guidance_pose(
        handle_list=scene_rig.route_guidance_handles,
        guidance=frame.get("route_guidance"),
        focus_xy=frame.get("virtual_center_xy"),
        fixed_height_m=float(frame.get("fixed_height_m") or fixed_height_m),
        hidden_position=hidden_position,
    )

    focus_xy = _resolve_focus_xy(
        frame=frame,
        positions_xy=positions_xy,
        fallback_focus_xy=tuple(camera_focus_xy.tolist()),
    )
    heading_xy = _resolve_heading_xy(
        frame=frame,
        positions_xy=positions_xy,
        velocities_xy=velocities_xy,
        fallback_heading_xy=tuple(camera_heading_xy.tolist()),
    )
    if follow_agent_index is not None and num_agents > 0:
        tracked_agent_index = max(0, min(int(follow_agent_index), num_agents - 1))
        tracked_position_xy = positions_xy[tracked_agent_index]
        focus_xy = (float(tracked_position_xy[0]), float(tracked_position_xy[1]))
        tracked_velocity_xy = velocities_xy[tracked_agent_index] if tracked_agent_index < len(velocities_xy) else (0.0, 0.0)
        heading_xy = _resolve_yaw_heading(
            velocity_xy=tracked_velocity_xy,
            fallback_yaw_rad=next_yaws[tracked_agent_index],
            fallback_heading_xy=heading_xy,
        )
    next_focus_xy = 0.9 * camera_focus_xy + 0.1 * np.asarray(focus_xy, dtype=np.float32)
    next_heading_xy = _blend_heading(camera_heading_xy, heading_xy)
    subject_scale_m = 6.0 if follow_agent_index is not None else _subject_scale_m(positions_xy, num_agents)
    update_replay_camera(
        sim,
        camera_prim_path=scene_rig.follow_camera_prim_path,
        focus_xy=(float(next_focus_xy[0]), float(next_focus_xy[1])),
        heading_xy=(float(next_heading_xy[0]), float(next_heading_xy[1])),
        fixed_height_m=fixed_height_m,
        subject_scale_m=subject_scale_m,
    )
    update_overview_replay_camera(
        sim,
        camera_prim_path=scene_rig.overview_camera_prim_path,
        focus_xy=(float(next_focus_xy[0]), float(next_focus_xy[1])),
        fixed_height_m=fixed_height_m,
        world_x_bounds_m=scene_rig.world_x_bounds_m,
        world_y_bounds_m=scene_rig.world_y_bounds_m,
        reference_xy=scene_rig.overview_reference_xy,
        subject_span_m=max(float(subject_scale_m), 18.0),
    )
    return next_yaws, next_focus_xy, next_heading_xy


def _append_rgb_frame(
    *,
    video_writer,
    annotator,
    overlay_annotator=None,
    overlay_follow_view: bool = False,
    latest_overlay_frame: np.ndarray | None = None,
) -> tuple[bool, np.ndarray | None]:
    if video_writer is None or annotator is None:
        return False, latest_overlay_frame
    frame = _try_extract_rgb_frame(annotator)
    if frame is None:
        return False, latest_overlay_frame
    next_overlay_frame = latest_overlay_frame
    if overlay_follow_view and overlay_annotator is not None:
        overlay_frame = _try_extract_rgb_frame(overlay_annotator)
        if overlay_frame is not None:
            next_overlay_frame = overlay_frame
        if next_overlay_frame is not None:
            frame = _compose_picture_in_picture(base_frame=frame, inset_frame=next_overlay_frame)
    video_writer.append_data(frame)
    return True, next_overlay_frame


def _try_extract_rgb_frame(annotator) -> np.ndarray | None:
    try:
        rgb = np.asarray(annotator.get_data())
    except Exception:
        return None
    if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0 or rgb.shape[-1] < 3:
        return None
    frame = rgb[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_camera_mode(camera_mode: str) -> str:
    normalized = str(camera_mode or "global").strip().lower()
    if normalized not in {
        "global",
        "follow",
        "picture_in_picture",
        "height_audit",
        "top_global",
        "top_centroid_follow",
    }:
        return "global"
    return normalized


def _set_camera_pose(
    sim,
    camera_prim_path: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    sim.set_camera_view(eye, target, camera_prim_path=camera_prim_path)


def _set_camera_orthographic(camera_prim_path: str, *, orthographic_size_m: float) -> None:
    from pxr import UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        raise RuntimeError(f"Replay camera prim does not exist: {camera_prim_path}")
    camera = UsdGeom.Camera(camera_prim)
    # This Isaac/Usd build does not expose orthographicSize on UsdGeom.Camera.
    # Keep the camera in perspective mode and remove top-down parallax through
    # the almost-vertical eye/target pair below instead of relying on a
    # non-portable camera attribute.
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)


def _collect_top_bounds(
    *,
    trajectory: list[dict[str, object]],
    payload: dict[str, object],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    world_x_bounds_m: tuple[float, float],
    world_y_bounds_m: tuple[float, float],
) -> tuple[list[float], list[float]]:
    x_values = [float(world_x_bounds_m[0]), float(world_x_bounds_m[1]), float(start_xy[0]), float(goal_xy[0])]
    y_values = [float(world_y_bounds_m[0]), float(world_y_bounds_m[1]), float(start_xy[1]), float(goal_xy[1])]
    metadata = payload.get("dynamic_gate_density")
    metadata = metadata if isinstance(metadata, dict) else {}
    for frame in trajectory[:: max(1, len(trajectory) // 64)]:
        for point in list(frame.get("positions_xy") or []):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x_values.append(float(point[0]))
                y_values.append(float(point[1]))
        for point in _resolve_gate_centers_for_frame(frame, metadata):
            x_values.append(float(point[0]))
            y_values.append(float(point[1]))
    return x_values, y_values


def _compute_top_global_orthographic_size(
    *,
    trajectory: list[dict[str, object]],
    payload: dict[str, object],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    world_x_bounds_m: tuple[float, float],
    world_y_bounds_m: tuple[float, float],
    aspect_ratio: float,
) -> float:
    x_values, y_values = _collect_top_bounds(
        trajectory=trajectory,
        payload=payload,
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=world_x_bounds_m,
        world_y_bounds_m=world_y_bounds_m,
    )
    span_x = max(max(x_values) - min(x_values), 24.0)
    span_y = max(max(y_values) - min(y_values), 16.0)
    vertical_size = max(1.22 * span_y, 1.16 * span_x / max(float(aspect_ratio), 1.0e-6), 30.0)
    return float(vertical_size)


def _compute_top_global_camera_pose(
    *,
    trajectory: list[dict[str, object]],
    payload: dict[str, object],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    fixed_height_m: float,
    world_x_bounds_m: tuple[float, float],
    world_y_bounds_m: tuple[float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute a near top-down fixed camera that frames the whole route."""

    x_values, y_values = _collect_top_bounds(
        trajectory=trajectory,
        payload=payload,
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=world_x_bounds_m,
        world_y_bounds_m=world_y_bounds_m,
    )
    center_x = 0.5 * (min(x_values) + max(x_values))
    center_y = 0.5 * (min(y_values) + max(y_values))
    span_x = max(max(x_values) - min(x_values), 24.0)
    span_y = max(max(y_values) - min(y_values), 16.0)
    framing_span = max(span_x, 1.8 * span_y, 42.0)
    height_m = max(58.0, 1.32 * framing_span)
    # Keep the top camera almost vertical. Larger offsets made the tall gate
    # frame project sideways over the drones and looked like clipping.
    eye = (float(center_x), float(center_y - 0.001), float(fixed_height_m + height_m))
    target = (float(center_x), float(center_y), float(fixed_height_m))
    return eye, target


def _top_centroid_orthographic_size_m(frame: dict[str, object]) -> float:
    positions_xy = list(frame.get("positions_xy") or [])
    num_agents = int(frame.get("num_agents") or len(positions_xy) or 1)
    subject_scale_m = _subject_scale_m(positions_xy, num_agents)
    return float(min(24.0, max(17.5, 1.10 * subject_scale_m + 7.0)))


def _update_top_centroid_follow_camera(
    *,
    sim,
    camera_prim_path: str,
    frame: dict[str, object],
    focus_xy: tuple[float, float],
    fixed_height_m: float,
    orthographic_size_m: float | None = None,
) -> None:
    positions_xy = list(frame.get("positions_xy") or [])
    num_agents = int(frame.get("num_agents") or len(positions_xy) or 1)
    subject_scale_m = _subject_scale_m(positions_xy, num_agents)
    follow_height_m = min(30.0, max(20.0, 18.0 + 0.55 * subject_scale_m))
    focus_x, focus_y = float(focus_xy[0]), float(focus_xy[1])
    if orthographic_size_m is not None:
        _set_camera_orthographic(camera_prim_path, orthographic_size_m=float(orthographic_size_m))
    eye = (focus_x, focus_y - 0.001, float(fixed_height_m + follow_height_m))
    target = (focus_x, focus_y, float(fixed_height_m))
    sim.set_camera_view(eye, target, camera_prim_path=camera_prim_path)


def _update_height_audit_camera(
    *,
    sim,
    camera_prim_path: str,
    frame: dict[str, object],
    fixed_height_m: float,
) -> None:
    """Use a low side camera so gate top/bottom and drone height are visually auditable."""

    focus_x, focus_y = _resolve_height_audit_focus_xy(frame)
    eye = (
        float(focus_x),
        float(focus_y - 28.0),
        float(fixed_height_m + 0.55),
    )
    target = (
        float(focus_x),
        float(focus_y),
        float(fixed_height_m),
    )
    sim.set_camera_view(eye, target, camera_prim_path=camera_prim_path)


def _resolve_height_audit_focus_xy(frame: dict[str, object]) -> tuple[float, float]:
    virtual_center = frame.get("virtual_center_xy")
    if isinstance(virtual_center, (list, tuple)) and len(virtual_center) >= 2:
        focus_x = float(virtual_center[0])
        focus_y = float(virtual_center[1])
    else:
        positions_xy = list(frame.get("positions_xy") or [])
        if positions_xy:
            focus_x = float(np.mean([float(point[0]) for point in positions_xy]))
            focus_y = float(np.mean([float(point[1]) for point in positions_xy]))
        else:
            focus_x = 0.0
            focus_y = 0.0
    gate_centers = list(frame.get("live_gate_centers_xy") or [])
    if gate_centers:
        nearest_gate = min(
            gate_centers,
            key=lambda point: abs(float(point[0]) - focus_x) if isinstance(point, (list, tuple)) and len(point) >= 2 else 1.0e9,
        )
        if isinstance(nearest_gate, (list, tuple)) and len(nearest_gate) >= 2:
            focus_x = 0.72 * focus_x + 0.28 * float(nearest_gate[0])
            focus_y = 0.72 * focus_y + 0.28 * float(nearest_gate[1])
    return focus_x, focus_y


def _compose_picture_in_picture(*, base_frame: np.ndarray, inset_frame: np.ndarray) -> np.ndarray:
    from PIL import Image

    base_image = Image.fromarray(base_frame)
    inset_image = Image.fromarray(inset_frame)
    max_inset_width = max(base_frame.shape[1] - 24, 1)
    max_inset_height = max(base_frame.shape[0] - 24, 1)
    inset_width = min(max(int(base_frame.shape[1] * 0.40), 320), max_inset_width)
    inset_height = min(max(int(base_frame.shape[0] * 0.40), 180), max_inset_height)
    resized_inset = inset_image.resize((inset_width, inset_height), Image.Resampling.BILINEAR)
    canvas = base_image.copy()
    margin_x_px = min(18, max(canvas.width - inset_width, 0))
    margin_y_px = min(18, max(canvas.height - inset_height, 0))
    inset_x = canvas.width - inset_width - margin_x_px
    inset_y = canvas.height - inset_height - margin_y_px
    canvas.paste(resized_inset, (inset_x, inset_y))

    border = np.array(canvas, dtype=np.uint8, copy=True)
    border_y0 = max(inset_y - 4, 0)
    border_x0 = max(inset_x - 4, 0)
    border_y1 = min(inset_y + inset_height + 4, border.shape[0])
    border_x1 = min(inset_x + inset_width + 4, border.shape[1])
    border[border_y0:border_y1, border_x0:inset_x] = np.array([255, 225, 120], dtype=np.uint8)
    border[border_y0:border_y1, inset_x + inset_width:border_x1] = np.array([255, 225, 120], dtype=np.uint8)
    border[border_y0:inset_y, border_x0:border_x1] = np.array([255, 225, 120], dtype=np.uint8)
    border[inset_y + inset_height:border_y1, border_x0:border_x1] = np.array([255, 225, 120], dtype=np.uint8)
    return border


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
    target_rel_x = float(guidance.get("target_rel_x", 0.0)) * 50.0
    target_rel_y = float(guidance.get("target_rel_y", 0.0)) * 50.0
    apply_pose(
        handle,
        position_xyz=(
            center_xy[0] + target_rel_x,
            center_xy[1] + target_rel_y,
            float(fixed_height_m) + 0.3,
        ),
        yaw_rad=0.0,
    )


def _resolve_follow_agent_index(
    *,
    follow_agent_index: int | None,
    max_agents: int,
    report: dict[str, object] | None,
) -> int:
    if max_agents <= 0:
        return 0
    if follow_agent_index is not None and int(follow_agent_index) >= 0:
        return min(int(follow_agent_index), max_agents - 1)
    seed_value = 0
    if report is not None:
        seed_value = int(report.get("seed") or 0)
    rng = np.random.default_rng(seed_value + max_agents)
    return int(rng.integers(0, max_agents))


def _coerce_route_waypoints(route_waypoints_xy) -> tuple[tuple[float, float], ...]:
    if route_waypoints_xy is None:
        return tuple()
    points: list[tuple[float, float]] = []
    for raw_point in route_waypoints_xy:
        if isinstance(raw_point, dict):
            raw_x = raw_point.get("x_m", raw_point.get("x"))
            raw_y = raw_point.get("y_m", raw_point.get("y"))
            if raw_x is None or raw_y is None:
                continue
            points.append((float(raw_x), float(raw_y)))
            continue
        if raw_point is None:
            continue
        try:
            if len(raw_point) < 2:
                continue
            points.append((float(raw_point[0]), float(raw_point[1])))
        except (TypeError, ValueError):
            continue
    return tuple(points)


def _resolve_recorded_route_waypoints(
    *,
    payload: dict[str, object],
    report: dict[str, object] | None,
    trajectory: list[dict[str, object]],
) -> tuple[tuple[float, float], ...]:
    candidates: list[object] = [
        payload.get("route_waypoints_xy"),
        payload.get("path_waypoints"),
        payload.get("waypoints_xy"),
    ]
    if report is not None:
        candidates.extend(
            [
                report.get("route_waypoints_xy"),
                report.get("path_waypoints"),
                report.get("waypoints_xy"),
                dict(report.get("snapshot") or {}).get("path_waypoints"),
            ]
        )
    if trajectory:
        candidates.extend(
            [
                trajectory[0].get("route_waypoints_xy"),
                trajectory[0].get("path_waypoints"),
                trajectory[0].get("waypoints_xy"),
            ]
        )
    for candidate in candidates:
        points = _coerce_route_waypoints(candidate)
        if len(points) >= 3:
            return points
    return tuple()


def _resolve_route_waypoint_names(
    *,
    route_waypoint_names,
    route_waypoints_xy: tuple[tuple[float, float], ...],
    payload: dict[str, object],
    report: dict[str, object] | None,
) -> tuple[str, ...]:
    if not route_waypoints_xy:
        return tuple()
    candidates: list[object] = [route_waypoint_names, payload.get("route_waypoint_names"), payload.get("waypoint_names")]
    if report is not None:
        candidates.extend([report.get("route_waypoint_names"), report.get("waypoint_names")])
    for candidate in candidates:
        if not candidate:
            continue
        names = tuple(str(name) for name in candidate)
        if len(names) == len(route_waypoints_xy):
            return names
    return tuple(f"P{idx + 1}" for idx in range(len(route_waypoints_xy)))


def _resolve_start_goal_xy(
    *,
    trajectory: list[dict[str, object]],
    report: dict[str, object] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    first = trajectory[0]
    center_xy = list(first.get("virtual_center_xy") or [])
    if len(center_xy) < 2:
        positions_xy = list(first.get("positions_xy") or [])
        if not positions_xy:
            raise ValueError("Multi-agent trajectory does not contain a virtual center or agent positions.")
        positions_np = np.asarray(positions_xy, dtype=np.float32)
        start_xy = (float(np.mean(positions_np[:, 0])), float(np.mean(positions_np[:, 1])))
    else:
        start_xy = (float(center_xy[0]), float(center_xy[1]))
    goal_xy = list(first.get("goal_xy") or [])
    if len(goal_xy) >= 2:
        return start_xy, (float(goal_xy[0]), float(goal_xy[1]))
    if report is not None:
        snapshot = dict(report.get("snapshot") or {})
        if "goal_xy" in snapshot and isinstance(snapshot["goal_xy"], list | tuple) and len(snapshot["goal_xy"]) >= 2:
            return start_xy, (float(snapshot["goal_xy"][0]), float(snapshot["goal_xy"][1]))
    raise ValueError("Multi-agent trajectory does not provide goal coordinates.")


def _resolve_yaw_from_velocity(
    *,
    velocity_xy: list[float] | tuple[float, float],
    fallback_yaw_rad: float,
) -> float:
    vx = float(velocity_xy[0])
    vy = float(velocity_xy[1])
    speed = float(np.hypot(vx, vy))
    if speed < 1.0e-6:
        return float(fallback_yaw_rad)
    return float(np.arctan2(vy, vx))


def _resolve_yaw_heading(
    *,
    velocity_xy: list[float] | tuple[float, float],
    fallback_yaw_rad: float,
    fallback_heading_xy: tuple[float, float],
) -> tuple[float, float]:
    vx = float(velocity_xy[0])
    vy = float(velocity_xy[1])
    speed = float(np.hypot(vx, vy))
    if speed > 1.0e-6:
        return (vx / speed, vy / speed)
    yaw_rad = float(fallback_yaw_rad)
    if abs(yaw_rad) > 1.0e-6:
        return (float(np.cos(yaw_rad)), float(np.sin(yaw_rad)))
    return fallback_heading_xy


def _resolve_focus_xy(
    *,
    frame: dict[str, object],
    positions_xy: list[list[float]] | list[tuple[float, float]],
    fallback_focus_xy: tuple[float, float],
) -> tuple[float, float]:
    virtual_center_xy = list(frame.get("virtual_center_xy") or [])
    if len(virtual_center_xy) >= 2:
        return (float(virtual_center_xy[0]), float(virtual_center_xy[1]))
    if positions_xy:
        positions_np = np.asarray(positions_xy, dtype=np.float32)
        return (float(np.mean(positions_np[:, 0])), float(np.mean(positions_np[:, 1])))
    return fallback_focus_xy


def _resolve_heading_xy(
    *,
    frame: dict[str, object],
    positions_xy: list[list[float]] | list[tuple[float, float]],
    velocities_xy: list[list[float]] | list[tuple[float, float]],
    fallback_heading_xy: tuple[float, float],
) -> tuple[float, float]:
    if velocities_xy:
        velocity_np = np.asarray(velocities_xy, dtype=np.float32)
        mean_velocity_xy = np.mean(velocity_np, axis=0)
        norm = float(np.linalg.norm(mean_velocity_xy))
        if norm > 1.0e-6:
            return (float(mean_velocity_xy[0] / norm), float(mean_velocity_xy[1] / norm))
    desired_slots_xy = list(frame.get("desired_slots_xy") or [])
    if positions_xy and desired_slots_xy:
        positions_np = np.asarray(positions_xy, dtype=np.float32)
        slots_np = np.asarray(desired_slots_xy[: len(positions_xy)], dtype=np.float32)
        mean_delta_xy = np.mean(slots_np - positions_np, axis=0)
        norm = float(np.linalg.norm(mean_delta_xy))
        if norm > 1.0e-6:
            return (float(mean_delta_xy[0] / norm), float(mean_delta_xy[1] / norm))
    return fallback_heading_xy


def _subject_scale_m(
    positions_xy: list[list[float]] | list[tuple[float, float]],
    num_agents: int,
) -> float:
    if not positions_xy:
        return 10.0
    positions_np = np.asarray(positions_xy, dtype=np.float32)
    span_x = float(np.max(positions_np[:, 0]) - np.min(positions_np[:, 0]))
    span_y = float(np.max(positions_np[:, 1]) - np.min(positions_np[:, 1]))
    return max(10.0, span_x, span_y, 6.0 + 0.5 * math.sqrt(max(num_agents, 1)))


def _blend_heading(previous_heading_xy: np.ndarray, heading_xy: tuple[float, float]) -> np.ndarray:
    blended = 0.86 * np.asarray(previous_heading_xy, dtype=np.float32) + 0.14 * np.asarray(
        heading_xy,
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(blended))
    if norm <= 1.0e-6:
        return np.asarray((1.0, 0.0), dtype=np.float32)
    return blended / norm

