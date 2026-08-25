"""IsaacLab replay renderer for single-agent aerogate_graph trajectories."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from single_gate.configs.experiment_config import SINGLE_EXPERIMENT_CONFIG
from shared.core.collision_2d import GateObstacleMap2D
from shared.runtime.artifacts import write_json
from shared.configs.global_config import GLOBAL_CONFIG
from shared.visualization.scene_isaaclab import (
    REPLAY_DRONE_BEACON_Z_OFFSET_M,
    REPLAY_DRONE_HALO_Z_OFFSET_M,
    REPLAY_DRONE_MAST_Z_OFFSET_M,
    apply_pose,
    create_rgb_annotator,
    destroy_rgb_annotator,
    setup_replay_scene,
    update_overview_replay_camera,
    update_replay_camera,
)


def render_single_trajectory_isaaclab(
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
    overlay_follow_view: bool = False,
) -> dict[str, object]:
    """Render one saved single-agent trajectory inside an IsaacLab scene."""

    import isaaclab.sim as sim_utils

    trajectory_file = Path(trajectory_path)
    payload = _read_json(trajectory_file)
    trajectory = list(payload.get("trajectory") or [])
    if not trajectory:
        raise ValueError(f"Trajectory file does not contain any frames: {trajectory_file}")

    report = _read_json(Path(report_path)) if report_path is not None else None
    fixed_height_m = float(payload.get("fixed_height_m") or SINGLE_EXPERIMENT_CONFIG.environment.fixed_height_m)
    start_xy, goal_xy = _resolve_start_goal_xy(trajectory=trajectory, report=report)
    env_config = SINGLE_EXPERIMENT_CONFIG.environment
    resolved_camera_mode = _resolve_camera_mode(camera_mode, overlay_follow_view)
    goal_radius_m = float(payload.get("goal_radius_m") or env_config.goal_radius_m)
    gate_post_collision_zone_radius_m = GLOBAL_CONFIG.default_gate_post_collision_radius_m + env_config.drone_radius_m
    gate_post_safety_zone_radius_m = gate_post_collision_zone_radius_m + env_config.safety_clearance_m
    obstacle_map = GateObstacleMap2D.from_gate()
    goal_center_clearance_m = obstacle_map.min_signed_distance(goal_xy, drone_radius_m=env_config.drone_radius_m)
    goal_zone_clearance_m = float(goal_center_clearance_m - goal_radius_m)
    nearest_gate_post_distance_to_goal_m = min(
        float(np.hypot(goal_xy[0] - obstacle.center_xy[0], goal_xy[1] - obstacle.center_xy[1]))
        for obstacle in obstacle_map.obstacles
    )

    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / max(int(fps), 1), render_interval=1))
    scene_rig = setup_replay_scene(
        sim=sim,
        drone_count=1,
        fixed_height_m=fixed_height_m,
        drone_scale=(2.8, 2.8, 2.8),
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=env_config.world_x_bounds_m,
        world_y_bounds_m=env_config.world_y_bounds_m,
        gate_post_collision_zone_radius_m=gate_post_collision_zone_radius_m,
        gate_post_safety_zone_radius_m=gate_post_safety_zone_radius_m,
        start_zone_radius_m=2.6,
        goal_zone_radius_m=goal_radius_m,
        drone_highlight_radius_m=0.72,
        drone_safety_radius_m=env_config.drone_radius_m + env_config.safety_clearance_m,
    )

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
    try:
        if resolved_mp4_path is not None:
            import imageio.v2 as imageio

            overview_annotator, overview_render_product = create_rgb_annotator(
                camera_prim_path=(
                    scene_rig.follow_camera_prim_path
                    if resolved_camera_mode == "follow"
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

        previous_yaw = 0.0
        camera_focus_xy = np.asarray(start_xy, dtype=np.float32)
        camera_heading_xy = np.asarray((1.0, 0.0), dtype=np.float32)
        first_frame = trajectory[0]
        previous_yaw, camera_focus_xy, camera_heading_xy = _apply_single_frame_pose(
            frame=first_frame,
            fixed_height_m=fixed_height_m,
            scene_rig=scene_rig,
            previous_yaw=previous_yaw,
            camera_focus_xy=camera_focus_xy,
            camera_heading_xy=camera_heading_xy,
            sim=sim,
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
            previous_yaw, camera_focus_xy, camera_heading_xy = _apply_single_frame_pose(
                frame=frame,
                fixed_height_m=fixed_height_m,
                scene_rig=scene_rig,
                previous_yaw=previous_yaw,
                camera_focus_xy=camera_focus_xy,
                camera_heading_xy=camera_heading_xy,
                sim=sim,
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
        "mode": "single_isaaclab_replay",
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
        "drone_prim_paths": [handle.prim_path for handle in scene_rig.drone_handles],
        "world_x_bounds_m": list(env_config.world_x_bounds_m),
        "world_y_bounds_m": list(env_config.world_y_bounds_m),
        "goal_radius_m": float(goal_radius_m),
        "gate_post_collision_zone_radius_m": float(gate_post_collision_zone_radius_m),
        "gate_post_safety_zone_radius_m": float(gate_post_safety_zone_radius_m),
        "goal_center_clearance_m": float(goal_center_clearance_m),
        "goal_zone_clearance_m": float(goal_zone_clearance_m),
        "nearest_gate_post_distance_to_goal_m": float(nearest_gate_post_distance_to_goal_m),
        "start_xy": list(start_xy),
        "goal_xy": list(goal_xy),
        "done_reason": None if report is None else report.get("done_reason"),
        "success": None if report is None else report.get("success"),
    }
    summary_path = resolved_output_dir / "isaaclab_replay_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def _apply_single_frame_pose(
    *,
    frame: dict[str, object],
    fixed_height_m: float,
    scene_rig,
    previous_yaw: float,
    camera_focus_xy: np.ndarray,
    camera_heading_xy: np.ndarray,
    sim,
) -> tuple[float, np.ndarray, np.ndarray]:
    yaw_rad = _resolve_yaw_rad(frame, fallback_yaw_rad=previous_yaw)
    position_xyz = (
        float(frame["x_m"]),
        float(frame["y_m"]),
        float(frame.get("z_m") or fixed_height_m),
    )
    apply_pose(
        scene_rig.drone_handles[0],
        position_xyz=position_xyz,
        yaw_rad=yaw_rad,
    )
    apply_pose(
        scene_rig.drone_halo_handles[0],
        position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_HALO_Z_OFFSET_M),
        yaw_rad=0.0,
    )
    apply_pose(
        scene_rig.drone_mast_handles[0],
        position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_MAST_Z_OFFSET_M),
        yaw_rad=0.0,
    )
    apply_pose(
        scene_rig.drone_beacon_handles[0],
        position_xyz=(position_xyz[0], position_xyz[1], position_xyz[2] + REPLAY_DRONE_BEACON_Z_OFFSET_M),
        yaw_rad=0.0,
    )
    if scene_rig.drone_safety_handles:
        apply_pose(scene_rig.drone_safety_handles[0], position_xyz=position_xyz, yaw_rad=0.0)
    position_xy = np.asarray((position_xyz[0], position_xyz[1]), dtype=np.float32)
    heading_xy = _heading_xy_from_frame(frame, fallback_heading_xy=tuple(camera_heading_xy.tolist()))
    next_focus_xy = 0.88 * camera_focus_xy + 0.12 * position_xy
    next_heading_xy = _blend_heading(camera_heading_xy, heading_xy)
    update_replay_camera(
        sim,
        camera_prim_path=scene_rig.follow_camera_prim_path,
        focus_xy=(float(next_focus_xy[0]), float(next_focus_xy[1])),
        heading_xy=(float(next_heading_xy[0]), float(next_heading_xy[1])),
        fixed_height_m=fixed_height_m,
        subject_scale_m=7.0,
    )
    update_overview_replay_camera(
        sim,
        camera_prim_path=scene_rig.overview_camera_prim_path,
        focus_xy=(float(next_focus_xy[0]), float(next_focus_xy[1])),
        fixed_height_m=fixed_height_m,
        world_x_bounds_m=scene_rig.world_x_bounds_m,
        world_y_bounds_m=scene_rig.world_y_bounds_m,
        reference_xy=scene_rig.overview_reference_xy,
        subject_span_m=18.0,
    )
    return yaw_rad, next_focus_xy, next_heading_xy


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
        return _extract_rgb_frame(annotator)
    except Exception:
        return None


def _extract_rgb_frame(annotator) -> np.ndarray:
    rgb = np.asarray(annotator.get_data())
    if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0 or rgb.shape[-1] < 3:
        raise RuntimeError(f"Unexpected RGB annotator output shape: {rgb.shape}")
    frame = rgb[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


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


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_camera_mode(camera_mode: str, overlay_follow_view: bool) -> str:
    normalized = str(camera_mode or "global").strip().lower()
    if normalized not in {"global", "follow", "picture_in_picture"}:
        normalized = "picture_in_picture" if overlay_follow_view else "global"
    elif normalized == "global" and overlay_follow_view:
        normalized = "picture_in_picture"
    return normalized


def _resolve_start_goal_xy(
    *,
    trajectory: list[dict[str, object]],
    report: dict[str, object] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    first = trajectory[0]
    start_xy = (float(first["x_m"]), float(first["y_m"]))
    if "goal_x_m" in first and "goal_y_m" in first:
        return start_xy, (float(first["goal_x_m"]), float(first["goal_y_m"]))
    if report is not None:
        final_state = dict(report.get("final_state") or {})
        goal_xy = final_state.get("goal_xy")
        if isinstance(goal_xy, list | tuple) and len(goal_xy) >= 2:
            return start_xy, (float(goal_xy[0]), float(goal_xy[1]))
    raise ValueError("Single-agent trajectory does not provide goal coordinates.")


def _resolve_yaw_rad(frame: dict[str, object], *, fallback_yaw_rad: float) -> float:
    if frame.get("yaw_rad") is not None:
        return float(frame["yaw_rad"])
    vx = float(frame.get("vx_mps") or 0.0)
    vy = float(frame.get("vy_mps") or 0.0)
    speed = float(np.hypot(vx, vy))
    if speed < 1.0e-6:
        return float(fallback_yaw_rad)
    return float(np.arctan2(vy, vx))


def _heading_xy_from_frame(
    frame: dict[str, object],
    *,
    fallback_heading_xy: tuple[float, float],
) -> tuple[float, float]:
    vx = float(frame.get("vx_mps") or 0.0)
    vy = float(frame.get("vy_mps") or 0.0)
    norm = float(np.hypot(vx, vy))
    if norm > 1.0e-6:
        return (vx / norm, vy / norm)
    yaw_rad = frame.get("yaw_rad")
    if yaw_rad is not None:
        return (float(np.cos(float(yaw_rad))), float(np.sin(float(yaw_rad))))
    return fallback_heading_xy


def _blend_heading(previous_heading_xy: np.ndarray, heading_xy: tuple[float, float]) -> np.ndarray:
    blended = 0.85 * np.asarray(previous_heading_xy, dtype=np.float32) + 0.15 * np.asarray(
        heading_xy,
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(blended))
    if norm <= 1.0e-6:
        return np.asarray((1.0, 0.0), dtype=np.float32)
    return blended / norm

