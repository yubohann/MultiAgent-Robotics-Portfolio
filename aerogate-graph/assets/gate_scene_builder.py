"""Reusable gate-course scene builder for the local gate package."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from assets.five_in_drone import DEFAULT_DRONE_USD, spawn_real_drones
from assets.gate_scene_layouts import DEFAULT_GATE_USD, GateCourseLayout2D, gate_visual_instances


DEFAULT_GATE_CAMERA_EYE = (0.0, -78.0, 26.0)
DEFAULT_GATE_CAMERA_TARGET = (0.0, 0.0, 3.0)


def build_gate_course_scene(
    *,
    sim,
    layout: GateCourseLayout2D,
    drone_positions_xyz: Sequence[Sequence[float]],
    drone_usd_path: str | Path | None = DEFAULT_DRONE_USD,
    drone_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gate_usd_path: str | Path = DEFAULT_GATE_USD,
    camera_eye: tuple[float, float, float] | None = DEFAULT_GATE_CAMERA_EYE,
    camera_target: tuple[float, float, float] | None = DEFAULT_GATE_CAMERA_TARGET,
    camera_prim_path: str | None = None,
) -> dict[str, list[str]]:
    """Build a visual shell scene with the local gate USD and 5_in_drone."""

    _spawn_ground_and_lights()
    gate_prim_paths = spawn_gate_course_assets(layout=layout, gate_usd_path=gate_usd_path)
    drone_prim_paths = spawn_real_drones(
        drone_positions_xyz=drone_positions_xyz,
        drone_usd_path=drone_usd_path,
        drone_scale=drone_scale,
    )
    if camera_eye is not None and camera_target is not None:
        if camera_prim_path is None:
            sim.set_camera_view(camera_eye, camera_target)
        else:
            sim.set_camera_view(camera_eye, camera_target, camera_prim_path=camera_prim_path)
    return {
        "gate_prim_paths": gate_prim_paths,
        "drone_prim_paths": drone_prim_paths,
    }


def spawn_gate_course_assets(
    *,
    layout: GateCourseLayout2D,
    gate_usd_path: str | Path = DEFAULT_GATE_USD,
) -> list[str]:
    """Spawn the gate USD instances for one fixed slalom layout."""

    import isaaclab.sim as sim_utils

    resolved_gate_path = Path(gate_usd_path)
    if not resolved_gate_path.exists():
        raise FileNotFoundError(f"Gate USD asset is missing: {resolved_gate_path}")

    prim_paths: list[str] = []
    gate_cfg = sim_utils.UsdFileCfg(usd_path=str(resolved_gate_path))
    for instance in gate_visual_instances(layout):
        prim_path = f"/World/Gates/{instance.prim_name}"
        gate_cfg.func(
            prim_path,
            gate_cfg,
            translation=instance.position_xyz,
            orientation=_yaw_to_quat_wxyz(instance.yaw_rad),
            scale=instance.scale_xyz,
        )
        prim_paths.append(prim_path)
    return prim_paths


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


def _yaw_to_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, 0.0, math.sin(half))
