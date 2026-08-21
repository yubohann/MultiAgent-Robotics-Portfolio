"""Shared gate-course layouts for experiment-1/2 planner evaluation and replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


ASSETS_ROOT = Path(__file__).resolve().parent
DEFAULT_GATE_USD = ASSETS_ROOT / "gate" / "gate.usd"
UNIFIED_GATE_BOTTOM_HEIGHT_M = 0.0
UNIFIED_GATE_TOP_HEIGHT_M = 8.0
UNIFIED_GATE_CENTER_HEIGHT_M = 4.0
GATE_NATIVE_VISUAL_HEIGHT_M = 4.2
UNIFIED_GATE_VISUAL_SCALE_Z = UNIFIED_GATE_TOP_HEIGHT_M / GATE_NATIVE_VISUAL_HEIGHT_M


@dataclass(frozen=True)
class CircularObstacleSpec:
    """Generic 2D circular obstacle used by planner-side gate approximations."""

    center_xy: tuple[float, float]
    radius_m: float
    label: str
    usd_path: Path


@dataclass(frozen=True)
class GateVisualInstance:
    """One visual gate instance placed in the real 3D shell scene."""

    prim_name: str
    position_xyz: tuple[float, float, float]
    yaw_rad: float
    scale_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class GateCourseLayout2D:
    """Gate slalom layout shared by planner tasks and 3D IsaacLab visualization."""

    name: str
    start_x_m: float
    goal_x_m: float
    start_y_range_m: tuple[float, float]
    goal_y_range_m: tuple[float, float]
    world_x_bounds_m: tuple[float, float]
    world_y_bounds_m: tuple[float, float]
    gate_centers_xy: tuple[tuple[float, float], ...]
    gate_yaw_rad: tuple[float, ...]
    gate_half_width_m: float = 2.0
    gate_post_radius_m: float = 0.55
    gate_height_m: float = UNIFIED_GATE_TOP_HEIGHT_M
    gate_scale_xyz: tuple[float, float, float] = (1.0, 1.0, UNIFIED_GATE_VISUAL_SCALE_Z)
    gate_base_z_m: float = 0.0


EXP2_INTERNAL_GATE_LAYOUT = GateCourseLayout2D(
    name="exp2_internal_gate_slalom",
    start_x_m=-38.0,
    goal_x_m=38.0,
    start_y_range_m=(-4.5, 4.5),
    goal_y_range_m=(-4.5, 4.5),
    world_x_bounds_m=(-42.0, 42.0),
    world_y_bounds_m=(-18.0, 18.0),
    gate_centers_xy=((-28.0, -4.0), (-14.0, 5.0), (0.0, -5.0), (14.0, 4.0), (28.0, 0.0)),
    gate_yaw_rad=(0.0, 0.04, -0.03, 0.03, 0.0),
    gate_half_width_m=2.0,
    gate_post_radius_m=0.55,
    gate_height_m=UNIFIED_GATE_TOP_HEIGHT_M,
    gate_scale_xyz=(1.25, 1.25, UNIFIED_GATE_VISUAL_SCALE_Z),
)

EXP1_EXTERNAL_GATE_LAYOUT = GateCourseLayout2D(
    name="exp1_external_gate_slalom",
    start_x_m=-40.0,
    goal_x_m=40.0,
    start_y_range_m=(-6.0, 6.0),
    goal_y_range_m=(-6.0, 6.0),
    world_x_bounds_m=(-46.0, 46.0),
    world_y_bounds_m=(-22.0, 22.0),
    gate_centers_xy=((-31.0, 6.0), (-18.0, -7.0), (-5.0, 7.0), (8.0, -6.0), (22.0, 5.0), (34.0, -2.0)),
    gate_yaw_rad=(0.08, -0.10, 0.07, -0.06, 0.05, 0.0),
    gate_half_width_m=1.9,
    gate_post_radius_m=0.60,
    gate_height_m=UNIFIED_GATE_TOP_HEIGHT_M,
    gate_scale_xyz=(1.20, 1.20, UNIFIED_GATE_VISUAL_SCALE_Z),
)

RACE50_MULTI_YAW_GATE_LAYOUT = GateCourseLayout2D(
    name="race50_multi_yaw_gate_course",
    start_x_m=-25.0,
    goal_x_m=25.0,
    start_y_range_m=(-3.0, 3.0),
    goal_y_range_m=(-3.0, 3.0),
    world_x_bounds_m=(-30.0, 30.0),
    world_y_bounds_m=(-16.0, 16.0),
    gate_centers_xy=(
        (-20.0, -3.5),
        (-13.0, 4.5),
        (-6.0, -4.0),
        (1.0, 5.2),
        (8.0, -5.0),
        (15.0, 3.0),
        (21.0, -1.5),
    ),
    gate_yaw_rad=(
        math.radians(0.0),
        math.radians(18.0),
        math.radians(-24.0),
        math.radians(32.0),
        math.radians(-34.0),
        math.radians(22.0),
        math.radians(-12.0),
    ),
    gate_half_width_m=1.85,
    gate_post_radius_m=0.52,
    gate_height_m=UNIFIED_GATE_TOP_HEIGHT_M,
    gate_scale_xyz=(1.15, 1.15, UNIFIED_GATE_VISUAL_SCALE_Z),
)


def gate_post_obstacle_specs(
    layout: GateCourseLayout2D,
    *,
    label_prefix: str = "gate_post",
    usd_path: Path = DEFAULT_GATE_USD,
) -> tuple[CircularObstacleSpec, ...]:
    """Approximate each gate as two circular posts in the fixed-height planner plane."""

    obstacles: list[CircularObstacleSpec] = []
    for gate_index, center_xy in enumerate(layout.gate_centers_xy):
        yaw_rad = layout.gate_yaw_rad[gate_index] if gate_index < len(layout.gate_yaw_rad) else 0.0
        side_xy = (-math.sin(yaw_rad), math.cos(yaw_rad))
        for side_name, sign in (("left", 1.0), ("right", -1.0)):
            offset_xy = (
                float(center_xy[0] + sign * layout.gate_half_width_m * side_xy[0]),
                float(center_xy[1] + sign * layout.gate_half_width_m * side_xy[1]),
            )
            obstacles.append(
                CircularObstacleSpec(
                    center_xy=offset_xy,
                    radius_m=float(layout.gate_post_radius_m),
                    label=f"{label_prefix}_{gate_index:02d}_{side_name}",
                    usd_path=usd_path,
                )
            )
    return tuple(obstacles)


def gate_visual_instances(
    layout: GateCourseLayout2D,
    *,
    gate_scale_xyz: tuple[float, float, float] | None = None,
) -> tuple[GateVisualInstance, ...]:
    """Return the visual gate transforms used by the IsaacLab gate scene builder."""

    scale_xyz = layout.gate_scale_xyz if gate_scale_xyz is None else tuple(float(value) for value in gate_scale_xyz)
    instances = []
    for gate_index, center_xy in enumerate(layout.gate_centers_xy):
        yaw_rad = layout.gate_yaw_rad[gate_index] if gate_index < len(layout.gate_yaw_rad) else 0.0
        instances.append(
            GateVisualInstance(
                prim_name=f"Gate_{gate_index:02d}",
                position_xyz=(float(center_xy[0]), float(center_xy[1]), float(layout.gate_base_z_m)),
                yaw_rad=float(yaw_rad),
                scale_xyz=scale_xyz,
            )
        )
    return tuple(instances)

