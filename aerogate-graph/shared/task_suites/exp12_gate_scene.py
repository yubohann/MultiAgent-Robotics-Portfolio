"""Gate-based task suites for experiment-1 external generalization and experiment-2 internal evaluation."""

from __future__ import annotations

from assets.gate_scene_layouts import (
    CircularObstacleSpec,
    EXP1_EXTERNAL_GATE_LAYOUT,
    EXP2_INTERNAL_GATE_LAYOUT,
    GateCourseLayout2D,
    RACE50_MULTI_YAW_GATE_LAYOUT,
    gate_post_obstacle_specs,
)
from single_internal_gate.configs.experiment_config import Exp2SingleInternalConfig
from single_internal_gate.planners.interfaces import PlannerTask2D
from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D


TASK_SUITE_NAMES: tuple[str, ...] = ("gate", "race50", "race50_gate", "gate50", "legacy_gate")


def task_suite_names() -> tuple[str, ...]:
    return TASK_SUITE_NAMES


def gate_obstacle_map(
    layout: GateCourseLayout2D,
    *,
    fixed_height_m: float,
) -> GateObstacleMap2D:
    """Convert one visual gate layout into the unified planner obstacle map."""

    return GateObstacleMap2D(
        tuple(
            GatePostObstacle2D(
                species="gate_post",
                center_xy=spec.center_xy,
                collision_radius_m=float(spec.radius_m),
                canopy_height_m=float(fixed_height_m),
                description=f"{layout.name} {spec.label}",
                usd_path=str(spec.usd_path),
            )
            for spec in gate_post_obstacle_specs(layout)
        )
    )


def exp1_gate_obstacle_specs() -> tuple[CircularObstacleSpec, ...]:
    """Return the post obstacles used by the experiment-1 gate generalization scene."""

    return gate_post_obstacle_specs(EXP1_EXTERNAL_GATE_LAYOUT)


def make_exp2_gate_tasks(
    count: int,
    config: Exp2SingleInternalConfig,
) -> tuple[PlannerTask2D, ...]:
    """Build the fixed internal gate slalom tasks for experiment 2."""

    layout = EXP2_INTERNAL_GATE_LAYOUT
    obstacle_map = gate_obstacle_map(layout, fixed_height_m=config.environment.fixed_height_m)
    tasks = []
    for index in range(max(int(count), 0)):
        start_y = _interpolate(layout.start_y_range_m, index=index, total=count)
        goal_y = _interpolate((layout.goal_y_range_m[1], layout.goal_y_range_m[0]), index=index, total=count)
        tasks.append(
            PlannerTask2D(
                start_xy=(float(layout.start_x_m), start_y),
                goal_xy=(float(layout.goal_x_m), goal_y),
                obstacles_2d=obstacle_map,
                fixed_height_m=config.environment.fixed_height_m,
                task_id=f"exp2_gate_{index:03d}",
                drone_radius_m=config.environment.drone_radius_m,
                world_x_bounds_m=layout.world_x_bounds_m,
                world_y_bounds_m=layout.world_y_bounds_m,
            )
        )
    return tuple(tasks)


def make_race50_gate_tasks(
    count: int,
    config: Exp2SingleInternalConfig,
) -> tuple[PlannerTask2D, ...]:
    """Build the Race50-inspired multi-yaw gate tasks for speed stress tests."""

    layout = RACE50_MULTI_YAW_GATE_LAYOUT
    obstacle_map = gate_obstacle_map(layout, fixed_height_m=config.environment.fixed_height_m)
    tasks = []
    for index in range(max(int(count), 0)):
        start_y = _interpolate(layout.start_y_range_m, index=index, total=count)
        goal_y = _interpolate((layout.goal_y_range_m[1], layout.goal_y_range_m[0]), index=index, total=count)
        tasks.append(
            PlannerTask2D(
                start_xy=(float(layout.start_x_m), start_y),
                goal_xy=(float(layout.goal_x_m), goal_y),
                obstacles_2d=obstacle_map,
                fixed_height_m=config.environment.fixed_height_m,
                task_id=f"race50_gate_{index:03d}",
                drone_radius_m=config.environment.drone_radius_m,
                world_x_bounds_m=layout.world_x_bounds_m,
                world_y_bounds_m=layout.world_y_bounds_m,
            )
        )
    return tuple(tasks)


def _interpolate(range_m: tuple[float, float], *, index: int, total: int) -> float:
    total = max(int(total), 1)
    if total == 1:
        return 0.5 * (float(range_m[0]) + float(range_m[1]))
    alpha = float(index) / float(total - 1)
    return float(range_m[0]) + (float(range_m[1]) - float(range_m[0])) * alpha

