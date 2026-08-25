"""Shared planner contracts for experiment-2 internal 2D methods."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shared.core.collision_2d import GateObstacleMap2D


@dataclass(frozen=True)
class PlannerTask2D:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    obstacles_2d: GateObstacleMap2D
    fixed_height_m: float
    task_id: str
    drone_radius_m: float
    world_x_bounds_m: tuple[float, float]
    world_y_bounds_m: tuple[float, float]


@dataclass(frozen=True)
class PlannerResult:
    planner_name: str
    success: bool
    path_xy: tuple[tuple[float, float], ...]
    path_length_m: float
    planning_time_ms: float
    num_replans: int = 0


def path_length(path_xy: tuple[tuple[float, float], ...]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path_xy[:-1], path_xy[1:])))

