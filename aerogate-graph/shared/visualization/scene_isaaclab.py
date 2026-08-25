"""IsaacLab scene helpers for aerogate_graph replay rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

DEFAULT_DRONE_SCALE = (1.0, 1.0, 1.0)


REPLAY_OVERVIEW_CAMERA_FOCAL_LENGTH_MM = 24.0
REPLAY_FOLLOW_CAMERA_FOCAL_LENGTH_MM = 26.0
REPLAY_CAMERA_HORIZONTAL_APERTURE_MM = 20.955
REPLAY_CAMERA_VERTICAL_APERTURE_MM = 11.7871875
REPLAY_CAMERA_CLIP_RANGE_M = (0.1, 2000.0)
REPLAY_OVERVIEW_CAMERA_EYE = (-14.0, -108.0, 30.0)
REPLAY_OVERVIEW_CAMERA_TARGET = (10.0, 0.0, 4.5)
REPLAY_FOLLOW_CAMERA_EYE = (-16.0, -14.0, 6.8)
REPLAY_FOLLOW_CAMERA_TARGET = (0.0, 0.0, 4.6)
REPLAY_DRONE_HALO_Z_OFFSET_M = 2.1
REPLAY_DRONE_MAST_HEIGHT_M = 8.0
REPLAY_DRONE_MAST_Z_OFFSET_M = REPLAY_DRONE_MAST_HEIGHT_M * 0.5
REPLAY_DRONE_BEACON_Z_OFFSET_M = 6.8


@dataclass(frozen=True)
class PoseHandle:
    """Resolved USD xform attributes used to drive one prim pose."""

    prim_path: str
    translate_attr: Any
    orient_attr: Any


@dataclass(frozen=True)
class ReplaySceneRig:
    """Replay scene prim handles for drones, highlights, and the camera."""

    drone_handles: list[PoseHandle]
    drone_halo_handles: list[PoseHandle]
    drone_mast_handles: list[PoseHandle]
    drone_beacon_handles: list[PoseHandle]
    drone_safety_handles: list[PoseHandle]
    slow_guidance_handles: list[PoseHandle]
    route_guidance_handles: list[PoseHandle]
    overview_camera_prim_path: str
    follow_camera_prim_path: str
    overview_reference_xy: tuple[float, float]
    world_x_bounds_m: tuple[float, float] | None
    world_y_bounds_m: tuple[float, float] | None


def ensure_project_and_source_paths() -> None:
    """Add project and IsaacLab source roots to ``sys.path``."""

    experiment_root = Path(__file__).resolve().parents[2]
    project_root = experiment_root.parents[1]
    isaaclab_root = project_root.parent
    source_root = isaaclab_root / "source"
    candidates = [project_root]
    if source_root.is_dir():
        candidates.extend(path for path in source_root.iterdir() if path.is_dir())
    for path in candidates:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def setup_replay_scene(
    *,
    sim: Any,
    drone_count: int,
    fixed_height_m: float,
    drone_scale: tuple[float, float, float] = DEFAULT_DRONE_SCALE,
    render_real_gate: bool = True,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    overview_camera_eye: tuple[float, float, float] = REPLAY_OVERVIEW_CAMERA_EYE,
    overview_camera_target: tuple[float, float, float] = REPLAY_OVERVIEW_CAMERA_TARGET,
    overview_camera_prim_path: str = "/World/OverviewCamera",
    follow_camera_eye: tuple[float, float, float] = REPLAY_FOLLOW_CAMERA_EYE,
    follow_camera_target: tuple[float, float, float] = REPLAY_FOLLOW_CAMERA_TARGET,
    follow_camera_prim_path: str = "/World/ReplayFollowCamera",
    world_x_bounds_m: tuple[float, float] | None = None,
    world_y_bounds_m: tuple[float, float] | None = None,
    gate_post_collision_zone_radius_m: float | None = None,
    gate_post_safety_zone_radius_m: float | None = None,
    start_zone_radius_m: float | None = None,
    goal_zone_radius_m: float | None = None,
    route_waypoints_xy: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None,
    route_waypoint_names: tuple[str, ...] | list[str] | None = None,
    drone_highlight_radius_m: float = 0.68,
    drone_safety_radius_m: float | None = None,
    show_drone_debug_overlays: bool = True,
    show_scene_markers: bool = True,
    show_high_scene_markers: bool = True,
) -> ReplaySceneRig:
    """Spawn the replay scene, optionally with the fixed gate assets."""

    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils

    drone_positions_xyz = [
        (
            float(start_xy[0]) if start_xy is not None else 0.0,
            float(start_xy[1]) if start_xy is not None else 0.0,
            float(fixed_height_m),
        )
        for _ in range(int(drone_count))
    ]
    prim_utils.create_prim(overview_camera_prim_path, "Camera")
    prim_utils.create_prim(follow_camera_prim_path, "Camera")
    _configure_replay_camera(
        overview_camera_prim_path,
        focal_length_mm=REPLAY_OVERVIEW_CAMERA_FOCAL_LENGTH_MM,
    )
    _configure_replay_camera(
        follow_camera_prim_path,
        focal_length_mm=REPLAY_FOLLOW_CAMERA_FOCAL_LENGTH_MM,
    )
    overview_reference_xy = _resolve_overview_reference_xy(
        start_xy=start_xy,
        goal_xy=goal_xy,
        world_x_bounds_m=world_x_bounds_m,
        world_y_bounds_m=world_y_bounds_m,
    )
    _spawn_ground_and_lights()
    drone_prim_paths = _spawn_replay_drone_prims(
        positions_xyz=drone_positions_xyz,
        drone_scale=drone_scale,
    )
    _disable_replay_scene_physics(
        stage_roots=(
            [
                "/World/Gate",
            ]
            if bool(render_real_gate)
            else []
        )
        + [
            "/World/Drones",
            "/World/ReplayHighlights",
            "/World/Markers",
            "/World/Debug",
        ]
    )
    _apply_replay_drone_materials(drone_prim_paths)
    update_overview_replay_camera(
        sim,
        camera_prim_path=overview_camera_prim_path,
        focus_xy=overview_reference_xy,
        fixed_height_m=fixed_height_m,
        world_x_bounds_m=world_x_bounds_m,
        world_y_bounds_m=world_y_bounds_m,
        reference_xy=overview_reference_xy,
    )
    sim.set_camera_view(
        follow_camera_eye,
        follow_camera_target,
        camera_prim_path=follow_camera_prim_path,
    )
    if bool(show_scene_markers):
        _spawn_start_goal_markers(
            start_xy=start_xy,
            goal_xy=goal_xy,
            fixed_height_m=fixed_height_m,
            start_zone_radius_m=start_zone_radius_m,
            goal_zone_radius_m=goal_zone_radius_m,
            show_high_markers=show_high_scene_markers,
        )
        _spawn_route_waypoint_markers(
            route_waypoints_xy=route_waypoints_xy,
            route_waypoint_names=route_waypoint_names,
            fixed_height_m=fixed_height_m,
            show_high_markers=show_high_scene_markers,
        )
    if gate_post_collision_zone_radius_m is not None:
        _spawn_gate_post_zone_cylinders(
            prim_root="/World/Debug/GatePostCollisionZones",
            radius_m=float(gate_post_collision_zone_radius_m),
            height_m=14.0,
            color_rgb=(1.0, 0.12, 0.1),
            opacity=0.48,
        )
    if gate_post_safety_zone_radius_m is not None:
        _spawn_gate_post_zone_cylinders(
            prim_root="/World/Debug/GatePostSafetyZones",
            radius_m=float(gate_post_safety_zone_radius_m),
            height_m=14.0,
            color_rgb=(0.1, 0.95, 0.95),
            opacity=0.28,
        )
    if world_x_bounds_m is not None and world_y_bounds_m is not None:
        _spawn_world_bounds_debug(
            prim_root="/World/Debug/WorldBounds",
            world_x_bounds_m=world_x_bounds_m,
            world_y_bounds_m=world_y_bounds_m,
        )
    # Hide high-altitude debug markers in paper replays; they read as drones
    # flying over gates even when the mesh stays on the 4 m plane.
    drone_halo_handles = []
    drone_mast_handles = []
    drone_beacon_handles = []
    if bool(show_drone_debug_overlays):
        drone_halo_handles = _spawn_replay_spheres(
            prim_root="/World/ReplayHighlights/Halos",
            count=int(drone_count),
            radius_m=float(drone_highlight_radius_m),
            color_rgb=(1.0, 0.92, 0.15),
            opacity=0.98,
            positions_xyz=drone_positions_xyz,
            z_offset_m=REPLAY_DRONE_HALO_Z_OFFSET_M,
        )
        drone_mast_handles = _spawn_replay_cylinders(
            prim_root="/World/ReplayHighlights/Masts",
            count=int(drone_count),
            radius_m=0.18,
            height_m=REPLAY_DRONE_MAST_HEIGHT_M,
            color_rgb=(1.0, 0.15, 0.28),
            opacity=0.96,
            positions_xyz=[
                (position[0], position[1], float(position[2]) + REPLAY_DRONE_MAST_Z_OFFSET_M)
                for position in drone_positions_xyz
            ],
        )
        drone_beacon_handles = _spawn_replay_spheres(
            prim_root="/World/ReplayHighlights/Beacons",
            count=int(drone_count),
            radius_m=0.95,
            color_rgb=(1.0, 0.12, 0.28),
            opacity=0.96,
            positions_xyz=[
                (position[0], position[1], float(position[2]) + REPLAY_DRONE_BEACON_Z_OFFSET_M)
                for position in drone_positions_xyz
            ],
        )
    drone_safety_handles = []
    if drone_safety_radius_m is not None and drone_safety_radius_m > 0.0:
        drone_safety_handles = _spawn_replay_spheres(
            prim_root="/World/ReplayHighlights/Safety",
            count=int(drone_count),
            radius_m=float(drone_safety_radius_m),
            color_rgb=(1.0, 0.58, 0.08),
            opacity=0.12,
            positions_xyz=drone_positions_xyz,
        )
    slow_guidance_handles = _spawn_replay_spheres(
        prim_root="/World/ReplayHighlights/SlowGuidance",
        count=1,
        radius_m=0.72,
        color_rgb=(0.12, 0.95, 0.36),
        opacity=0.92,
        positions_xyz=[drone_positions_xyz[0]],
        z_offset_m=0.35,
    )
    route_guidance_handles = _spawn_replay_spheres(
        prim_root="/World/ReplayHighlights/LlmGuidance",
        count=1,
        radius_m=0.82,
        color_rgb=(0.16, 0.55, 1.0),
        opacity=0.9,
        positions_xyz=[drone_positions_xyz[0]],
        z_offset_m=0.55,
    )
    sim_utils.update_stage()
    return ReplaySceneRig(
        drone_handles=[build_pose_handle(prim_path) for prim_path in drone_prim_paths],
        drone_halo_handles=drone_halo_handles,
        drone_mast_handles=drone_mast_handles,
        drone_beacon_handles=drone_beacon_handles,
        drone_safety_handles=drone_safety_handles,
        slow_guidance_handles=slow_guidance_handles,
        route_guidance_handles=route_guidance_handles,
        overview_camera_prim_path=overview_camera_prim_path,
        follow_camera_prim_path=follow_camera_prim_path,
        overview_reference_xy=overview_reference_xy,
        world_x_bounds_m=None if world_x_bounds_m is None else tuple(float(value) for value in world_x_bounds_m),
        world_y_bounds_m=None if world_y_bounds_m is None else tuple(float(value) for value in world_y_bounds_m),
    )


def update_replay_camera(
    sim: Any,
    *,
    camera_prim_path: str,
    focus_xy: tuple[float, float],
    heading_xy: tuple[float, float],
    fixed_height_m: float,
    subject_scale_m: float = 8.0,
) -> None:
    """Move the replay camera to a shallow chase view that keeps the drone in frame."""

    heading_x, heading_y = _normalize_heading(heading_xy)
    side_x, side_y = (-heading_y, heading_x)
    resolved_scale = max(float(subject_scale_m), 6.0)
    distance_m = min(15.5, 8.8 + 0.30 * resolved_scale)
    side_offset_m = min(4.5, 2.4 + 0.06 * resolved_scale)
    height_m = min(6.2, 2.6 + 0.07 * resolved_scale)
    lookahead_m = min(9.0, 5.2 + 0.12 * resolved_scale)
    eye = (
        float(focus_xy[0] - heading_x * distance_m + side_x * side_offset_m),
        float(focus_xy[1] - heading_y * distance_m + side_y * side_offset_m),
        float(fixed_height_m + height_m),
    )
    target = (
        float(focus_xy[0] + heading_x * lookahead_m),
        float(focus_xy[1] + heading_y * lookahead_m),
        float(fixed_height_m + 1.1),
    )
    sim.set_camera_view(eye, target, camera_prim_path=camera_prim_path)


def compute_overview_camera_pose(
    *,
    focus_xy: tuple[float, float],
    fixed_height_m: float,
    world_x_bounds_m: tuple[float, float] | None,
    world_y_bounds_m: tuple[float, float] | None,
    reference_xy: tuple[float, float] | None = None,
    subject_span_m: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute a global replay camera that keeps the gate in frame while biasing toward the drone."""

    if world_x_bounds_m is None or world_y_bounds_m is None:
        return REPLAY_OVERVIEW_CAMERA_EYE, REPLAY_OVERVIEW_CAMERA_TARGET

    x_min, x_max = [float(value) for value in world_x_bounds_m]
    y_min, y_max = [float(value) for value in world_y_bounds_m]
    world_center_xy = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    resolved_reference_xy = world_center_xy if reference_xy is None else (
        float(reference_xy[0]),
        float(reference_xy[1]),
    )
    resolved_focus_xy = (float(focus_xy[0]), float(focus_xy[1]))
    blended_focus_xy = (
        0.68 * resolved_reference_xy[0] + 0.32 * resolved_focus_xy[0],
        0.68 * resolved_reference_xy[1] + 0.32 * resolved_focus_xy[1],
    )
    world_span_x = abs(x_max - x_min)
    world_span_y = abs(y_max - y_min)
    travel_span_x = abs(resolved_focus_xy[0] - world_center_xy[0]) * 2.0 + 42.0
    travel_span_y = abs(resolved_focus_xy[1] - world_center_xy[1]) * 2.0 + 42.0
    framing_span_m = max(
        world_span_x,
        world_span_y,
        float(subject_span_m) + 26.0,
        travel_span_x,
        travel_span_y,
        54.0,
    )
    eye = (
        float(blended_focus_xy[0] + 0.64 * framing_span_m),
        float(blended_focus_xy[1] - 0.88 * framing_span_m),
        float(fixed_height_m + 0.40 * framing_span_m),
    )
    target = (
        float(blended_focus_xy[0] - 0.08 * framing_span_m),
        float(blended_focus_xy[1] + 0.08 * framing_span_m),
        float(fixed_height_m + 1.8),
    )
    return eye, target


def update_overview_replay_camera(
    sim: Any,
    *,
    camera_prim_path: str,
    focus_xy: tuple[float, float],
    fixed_height_m: float,
    world_x_bounds_m: tuple[float, float] | None,
    world_y_bounds_m: tuple[float, float] | None,
    reference_xy: tuple[float, float] | None = None,
    subject_span_m: float = 0.0,
) -> None:
    """Update the overview camera with a persistent global framing and a mild subject-follow bias."""

    eye, target = compute_overview_camera_pose(
        focus_xy=focus_xy,
        fixed_height_m=fixed_height_m,
        world_x_bounds_m=world_x_bounds_m,
        world_y_bounds_m=world_y_bounds_m,
        reference_xy=reference_xy,
        subject_span_m=subject_span_m,
    )
    sim.set_camera_view(eye, target, camera_prim_path=camera_prim_path)


def create_rgb_annotator(
    *,
    camera_prim_path: str,
    resolution: tuple[int, int],
) -> tuple[Any, Any]:
    """Create a Replicator RGB annotator for one camera prim."""

    import omni.replicator.core as rep

    render_product = rep.create.render_product(camera_prim_path, resolution=resolution)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach(render_product)
    return annotator, render_product


def destroy_rgb_annotator(annotator: Any, render_product: Any) -> None:
    """Detach and destroy an RGB annotator/render product pair."""

    try:
        annotator.detach(render_product)
    except Exception:
        pass


def build_pose_handle(prim_path: str) -> PoseHandle:
    """Resolve a prim's translate/orient xform ops for repeated updates."""

    from pxr import UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Replay prim does not exist: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    translate_attr = None
    orient_attr = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_attr is None:
            translate_attr = op.GetAttr()
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_attr is None:
            orient_attr = op.GetAttr()
    if translate_attr is None:
        translate_attr = xformable.AddTranslateOp().GetAttr()
    if orient_attr is None:
        orient_attr = xformable.AddOrientOp().GetAttr()
    return PoseHandle(
        prim_path=prim_path,
        translate_attr=translate_attr,
        orient_attr=orient_attr,
    )


def apply_pose(handle: PoseHandle, *, position_xyz: tuple[float, float, float], yaw_rad: float) -> None:
    """Update one replay prim pose."""

    from pxr import Gf

    handle.translate_attr.Set(Gf.Vec3d(*[float(value) for value in position_xyz]))
    quat_wxyz = yaw_to_quat_wxyz(yaw_rad)
    try:
        handle.orient_attr.Set(Gf.Quatd(*quat_wxyz))
    except Exception:
        handle.orient_attr.Set(Gf.Quatf(*[float(value) for value in quat_wxyz]))


def yaw_to_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    """Convert planar yaw to an IsaacLab-compatible world quaternion."""

    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def planar_speed_to_spin_scale(speed_mps: float, *, max_speed_mps: float) -> float:
    """Convert planar speed to a mild rotor-spin multiplier for replay visuals."""

    normalized = max(0.0, min(float(speed_mps) / max(float(max_speed_mps), 1e-6), 1.0))
    return 0.75 + 0.5 * normalized


def _configure_replay_camera(camera_prim_path: str, *, focal_length_mm: float) -> None:
    """Configure one replay camera prim for a clear follow-style gate view."""

    from pxr import Gf, UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        raise RuntimeError(f"Replay camera prim does not exist: {camera_prim_path}")
    camera = UsdGeom.Camera(camera_prim)
    camera.CreateFocalLengthAttr().Set(float(focal_length_mm))
    camera.CreateHorizontalApertureAttr().Set(float(REPLAY_CAMERA_HORIZONTAL_APERTURE_MM))
    camera.CreateVerticalApertureAttr().Set(float(REPLAY_CAMERA_VERTICAL_APERTURE_MM))
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*REPLAY_CAMERA_CLIP_RANGE_M))


def _apply_replay_drone_materials(drone_prim_paths: list[str]) -> None:
    """Bind bright replay-only materials so the real drone mesh stays visible on video."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    def _make_material(
        name: str,
        *,
        diffuse_rgb: tuple[float, float, float],
        emissive_rgb: tuple[float, float, float],
        roughness: float,
    ):
        mat_path = f"/World/ReplayMaterials/{name}"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_rgb))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    body_material = _make_material(
        "DroneBody",
        diffuse_rgb=(0.08, 0.92, 0.88),
        emissive_rgb=(0.06, 0.40, 0.36),
        roughness=0.16,
    )
    prop_material = _make_material(
        "DroneProp",
        diffuse_rgb=(1.0, 0.46, 0.08),
        emissive_rgb=(0.30, 0.10, 0.02),
        roughness=0.12,
    )

    for drone_prim_path in drone_prim_paths:
        for desc in _iter_replay_drone_mesh_prims(stage, drone_prim_path):
            name_lower = desc.GetName().lower()
            material = prop_material if any(token in name_lower for token in ("prop", "rotor")) else body_material
            UsdShade.MaterialBindingAPI(desc).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            display_rgb = (1.0, 0.46, 0.08) if material is prop_material else (0.08, 0.92, 0.88)
            gprim = UsdGeom.Gprim(desc)
            gprim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*display_rgb)])
            gprim.CreateDisplayOpacityAttr().Set([1.0])
            UsdGeom.Mesh(desc).CreateDoubleSidedAttr().Set(True)


def _spawn_ground_and_lights() -> None:
    import isaaclab.sim as sim_utils

    ground_cfg = sim_utils.CuboidCfg(
        size=(160.0, 160.0, 0.18),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.065, 0.07, 0.075)),
    )
    ground_cfg.func("/World/Ground", ground_cfg, translation=(0.0, 0.0, -0.09))

    dome_cfg = sim_utils.DomeLightCfg(intensity=2800.0, color=(0.86, 0.88, 0.92))
    dome_cfg.func("/World/DomeLight", dome_cfg, translation=(0.0, 0.0, 10.0))

    sun_cfg = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.96, 0.84))
    sun_cfg.func(
        "/World/SunLight",
        sun_cfg,
        translation=(40.0, -10.0, 45.0),
        orientation=(0.707, 0.0, 0.707, 0.0),
    )


def _spawn_replay_drone_prims(
    *,
    positions_xyz: list[tuple[float, float, float]],
    drone_scale: tuple[float, float, float],
) -> list[str]:
    import isaaclab.sim as sim_utils

    from assets.five_in_drone import DEFAULT_FIVE_IN_DRONE_USD

    drone_cfg = sim_utils.UsdFileCfg(
        usd_path=str(DEFAULT_FIVE_IN_DRONE_USD),
        scale=tuple(float(value) for value in drone_scale),
    )
    prim_paths: list[str] = []
    for idx, position_xyz in enumerate(positions_xyz):
        prim_path = f"/World/Drones/Drone_{idx:02d}"
        drone_cfg.func(
            prim_path,
            drone_cfg,
            translation=tuple(float(value) for value in position_xyz),
        )
        prim_paths.append(prim_path)
    return prim_paths


def _spawn_start_goal_markers(
    *,
    start_xy: tuple[float, float] | None,
    goal_xy: tuple[float, float] | None,
    fixed_height_m: float,
    start_zone_radius_m: float | None,
    goal_zone_radius_m: float | None,
    show_high_markers: bool = True,
) -> None:
    import isaaclab.sim as sim_utils

    if start_xy is not None:
        start_radius_m = float(start_zone_radius_m if start_zone_radius_m is not None else 2.2)
        start_zone_cfg = sim_utils.CylinderCfg(
            radius=start_radius_m,
            height=0.18,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.62, 0.12),
                opacity=0.24,
            ),
        )
        start_zone_cfg.func(
            "/World/Markers/StartZone",
            start_zone_cfg,
            translation=(float(start_xy[0]), float(start_xy[1]), 0.09),
        )
        start_core_cfg = sim_utils.SphereCfg(
            radius=max(0.7, 0.24 * start_radius_m),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.72, 0.18)),
        )
        start_core_cfg.func(
            "/World/Markers/StartCore",
            start_core_cfg,
            translation=(float(start_xy[0]), float(start_xy[1]), 0.55),
        )
        if bool(show_high_markers):
            start_mast_cfg = sim_utils.CylinderCfg(
                radius=0.26,
                height=8.0,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.72, 0.18),
                    opacity=0.92,
                ),
            )
            start_mast_cfg.func(
                "/World/Markers/StartMast",
                start_mast_cfg,
                translation=(float(start_xy[0]), float(start_xy[1]), 4.0),
            )
            start_beacon_cfg = sim_utils.SphereCfg(
                radius=1.25,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.72, 0.18),
                    opacity=0.96,
                ),
            )
            start_beacon_cfg.func(
                "/World/Markers/StartBeacon",
                start_beacon_cfg,
                translation=(float(start_xy[0]), float(start_xy[1]), 8.5),
            )
    if goal_xy is not None:
        goal_radius_m = float(goal_zone_radius_m if goal_zone_radius_m is not None else 2.4)
        goal_zone_cfg = sim_utils.CylinderCfg(
            radius=goal_radius_m,
            height=0.18,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.92, 0.32),
                opacity=0.22,
            ),
        )
        goal_zone_cfg.func(
            "/World/Markers/GoalZone",
            goal_zone_cfg,
            translation=(float(goal_xy[0]), float(goal_xy[1]), 0.09),
        )
        goal_core_cfg = sim_utils.SphereCfg(
            radius=max(0.8, 0.24 * goal_radius_m),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.88, 0.32)),
        )
        goal_core_cfg.func(
            "/World/Markers/GoalCore",
            goal_core_cfg,
            translation=(float(goal_xy[0]), float(goal_xy[1]), 0.6),
        )
        if bool(show_high_markers):
            goal_mast_cfg = sim_utils.CylinderCfg(
                radius=0.28,
                height=12.0,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.12, 0.88, 0.32),
                    opacity=0.94,
                ),
            )
            goal_mast_cfg.func(
                "/World/Markers/GoalMast",
                goal_mast_cfg,
                translation=(float(goal_xy[0]), float(goal_xy[1]), 6.0),
            )
            goal_beacon_cfg = sim_utils.SphereCfg(
                radius=1.35,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.12, 0.88, 0.32),
                    opacity=0.98,
                ),
            )
            goal_beacon_cfg.func(
                "/World/Markers/GoalBeacon",
                goal_beacon_cfg,
                translation=(float(goal_xy[0]), float(goal_xy[1]), 12.5),
            )


def _spawn_route_waypoint_markers(
    *,
    route_waypoints_xy: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None,
    route_waypoint_names: tuple[str, ...] | list[str] | None,
    fixed_height_m: float,
    show_high_markers: bool = True,
) -> None:
    """Spawn persistent route beacons so full-route demos show every waypoint."""

    import isaaclab.sim as sim_utils

    if not route_waypoints_xy:
        return

    resolved_points: list[tuple[float, float]] = []
    for point in route_waypoints_xy:
        if point is None or len(point) < 2:
            continue
        resolved_points.append((float(point[0]), float(point[1])))
    if not resolved_points:
        return

    names = list(route_waypoint_names or [])
    last_index = len(resolved_points) - 1
    for waypoint_idx, point_xy in enumerate(resolved_points):
        name = names[waypoint_idx] if waypoint_idx < len(names) and names[waypoint_idx] else f"P{waypoint_idx + 1}"
        safe_name = _sanitize_marker_name(str(name))
        is_start = waypoint_idx == 0
        is_goal = waypoint_idx == last_index
        if is_start:
            color_rgb = (1.0, 0.72, 0.16)
            radius_m = 1.55
            mast_height_m = 7.2
            beacon_radius_m = 0.92
        elif is_goal:
            color_rgb = (0.12, 0.92, 0.32)
            radius_m = 1.65
            mast_height_m = 8.2
            beacon_radius_m = 1.0
        else:
            color_rgb = (0.0, 0.78, 1.0)
            radius_m = 1.15
            mast_height_m = 5.4
            beacon_radius_m = 0.62

        ring_cfg = sim_utils.CylinderCfg(
            radius=radius_m,
            height=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color_rgb,
                opacity=0.34 if (is_start or is_goal) else 0.26,
            ),
        )
        ring_cfg.func(
            f"/World/Markers/RouteWaypoints/WP_{waypoint_idx:02d}_{safe_name}/GroundRing",
            ring_cfg,
            translation=(point_xy[0], point_xy[1], 0.06),
        )
        core_cfg = sim_utils.SphereCfg(
            radius=0.45 if (is_start or is_goal) else 0.34,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color_rgb,
                opacity=0.94,
            ),
        )
        core_cfg.func(
            f"/World/Markers/RouteWaypoints/WP_{waypoint_idx:02d}_{safe_name}/Core",
            core_cfg,
            translation=(point_xy[0], point_xy[1], 0.76),
        )
        if bool(show_high_markers):
            mast_cfg = sim_utils.CylinderCfg(
                radius=0.16 if (is_start or is_goal) else 0.11,
                height=mast_height_m,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color_rgb,
                    opacity=0.88,
                ),
            )
            mast_cfg.func(
                f"/World/Markers/RouteWaypoints/WP_{waypoint_idx:02d}_{safe_name}/Mast",
                mast_cfg,
                translation=(point_xy[0], point_xy[1], 0.5 * mast_height_m),
            )
            beacon_cfg = sim_utils.SphereCfg(
                radius=beacon_radius_m,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color_rgb,
                    opacity=0.96,
                ),
            )
            beacon_cfg.func(
                f"/World/Markers/RouteWaypoints/WP_{waypoint_idx:02d}_{safe_name}/Beacon",
                beacon_cfg,
                translation=(point_xy[0], point_xy[1], float(fixed_height_m) + mast_height_m * 0.72),
            )


def _spawn_gate_post_zone_cylinders(
    *,
    prim_root: str,
    radius_m: float,
    height_m: float,
    color_rgb: tuple[float, float, float],
    opacity: float,
) -> None:
    _ = (prim_root, radius_m, height_m, color_rgb, opacity)


def _spawn_world_bounds_debug(
    *,
    prim_root: str,
    world_x_bounds_m: tuple[float, float],
    world_y_bounds_m: tuple[float, float],
) -> None:
    import isaaclab.sim as sim_utils

    x_min, x_max = [float(value) for value in world_x_bounds_m]
    y_min, y_max = [float(value) for value in world_y_bounds_m]
    pole_cfg = sim_utils.CylinderCfg(
        radius=0.3,
        height=6.0,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.15), opacity=0.94),
    )

    positions: list[tuple[float, float, float]] = []
    step_m = 4.0
    x_value = x_min
    while x_value <= x_max + 1.0e-6:
        positions.append((x_value, y_min, 2.0))
        positions.append((x_value, y_max, 2.0))
        x_value += step_m
    y_value = y_min + step_m
    while y_value < y_max - 1.0e-6:
        positions.append((x_min, y_value, 2.0))
        positions.append((x_max, y_value, 2.0))
        y_value += step_m

    for pole_idx, position_xyz in enumerate(positions):
        pole_cfg.func(
            f"{prim_root}/Pole_{pole_idx:03d}",
            pole_cfg,
            translation=(position_xyz[0], position_xyz[1], 3.0),
        )


def _spawn_replay_spheres(
    *,
    prim_root: str,
    count: int,
    radius_m: float,
    color_rgb: tuple[float, float, float],
    opacity: float,
    positions_xyz: list[tuple[float, float, float]],
    z_offset_m: float = 0.0,
) -> list[PoseHandle]:
    import isaaclab.sim as sim_utils

    sphere_cfg = sim_utils.SphereCfg(
        radius=float(radius_m),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_rgb, opacity=float(opacity)),
    )
    handles: list[PoseHandle] = []
    for idx in range(int(count)):
        prim_path = f"{prim_root}/Sphere_{idx:02d}"
        sphere_cfg.func(
            prim_path,
            sphere_cfg,
            translation=(
                float(positions_xyz[idx][0]),
                float(positions_xyz[idx][1]),
                float(positions_xyz[idx][2]) + float(z_offset_m),
            ),
        )
        handles.append(build_pose_handle(prim_path))
    return handles


def _spawn_replay_cylinders(
    *,
    prim_root: str,
    count: int,
    radius_m: float,
    height_m: float,
    color_rgb: tuple[float, float, float],
    opacity: float,
    positions_xyz: list[tuple[float, float, float]],
) -> list[PoseHandle]:
    import isaaclab.sim as sim_utils

    cylinder_cfg = sim_utils.CylinderCfg(
        radius=float(radius_m),
        height=float(height_m),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_rgb, opacity=float(opacity)),
    )
    handles: list[PoseHandle] = []
    for idx in range(int(count)):
        prim_path = f"{prim_root}/Cylinder_{idx:02d}"
        cylinder_cfg.func(
            prim_path,
            cylinder_cfg,
            translation=tuple(float(value) for value in positions_xyz[idx]),
        )
        handles.append(build_pose_handle(prim_path))
    return handles


def _sanitize_marker_name(name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in str(name))
    return safe or "Waypoint"


def _normalize_heading(heading_xy: tuple[float, float]) -> tuple[float, float]:
    heading_x = float(heading_xy[0])
    heading_y = float(heading_xy[1])
    norm = math.hypot(heading_x, heading_y)
    if norm <= 1.0e-6:
        return (1.0, 0.0)
    return (heading_x / norm, heading_y / norm)


def _resolve_overview_reference_xy(
    *,
    start_xy: tuple[float, float] | None,
    goal_xy: tuple[float, float] | None,
    world_x_bounds_m: tuple[float, float] | None,
    world_y_bounds_m: tuple[float, float] | None,
) -> tuple[float, float]:
    anchor_points: list[tuple[float, float]] = []
    if world_x_bounds_m is not None and world_y_bounds_m is not None:
        anchor_points.append(
            (
                0.5 * (float(world_x_bounds_m[0]) + float(world_x_bounds_m[1])),
                0.5 * (float(world_y_bounds_m[0]) + float(world_y_bounds_m[1])),
            )
        )
    if start_xy is not None:
        anchor_points.append((float(start_xy[0]), float(start_xy[1])))
    if goal_xy is not None:
        anchor_points.append((float(goal_xy[0]), float(goal_xy[1])))
    if not anchor_points:
        return (0.0, 0.0)
    xs, ys = zip(*anchor_points)
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def _iter_replay_drone_mesh_prims(stage: Any, drone_prim_path: str):
    from pxr import Usd, UsdGeom

    drone_prim = stage.GetPrimAtPath(drone_prim_path)
    if not drone_prim or not drone_prim.IsValid():
        return
    for desc in Usd.PrimRange(drone_prim):
        if not desc.IsA(UsdGeom.Mesh):
            continue
        name_lower = desc.GetName().lower()
        if any(token in name_lower for token in ("collider", "collision")):
            continue
        yield desc


def _disable_replay_scene_physics(*, stage_roots: list[str]) -> None:
    """Make replay/live-preview stages visual-only to avoid unnecessary PhysX load."""

    from pxr import Usd, UsdPhysics

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    attr_names_to_disable = (
        "physics:collisionEnabled",
        "physics:rigidBodyEnabled",
        "physxRigidBody:disableGravity",
        "physxArticulation:enabledSelfCollisions",
        "physxDeformable:simulationHexahedralResolution",
    )

    for root_path in stage_roots:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if not prim or not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
            for attr_name in attr_names_to_disable:
                attr = prim.GetAttribute(attr_name)
                if not attr or not attr.IsValid():
                    continue
                type_name = str(attr.GetTypeName())
                if type_name in {"bool", "Bool"}:
                    attr.Set(False)

