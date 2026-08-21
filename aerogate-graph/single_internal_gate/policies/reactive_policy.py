"""Reactive local 2D obstacle-avoidance policy for experiment-2."""

from __future__ import annotations

import math

from single_internal_gate.configs.experiment_config import Exp2MethodConfig
from single_internal_gate.planners.interfaces import PlannerTask2D


class ReactivePolicy2D:
    def __init__(self, config: Exp2MethodConfig) -> None:
        self.config = config

    def command(
        self,
        *,
        position_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        task: PlannerTask2D,
    ) -> tuple[float, float]:
        gx, gy = _unit(goal_xy[0] - position_xy[0], goal_xy[1] - position_xy[1])
        rx = 0.0
        ry = 0.0
        for obstacle in task.obstacles_2d.query_local(position_xy, radius_m=self.config.obstacle_query_radius_m):
            dx = position_xy[0] - obstacle.center_xy[0]
            dy = position_xy[1] - obstacle.center_xy[1]
            distance = max(math.hypot(dx, dy), 1.0e-6)
            clearance = distance - obstacle.collision_radius_m - task.drone_radius_m
            if clearance >= self.config.obstacle_query_radius_m:
                continue
            weight = (self.config.obstacle_query_radius_m - clearance) / self.config.obstacle_query_radius_m
            rx += dx / distance * weight * self.config.reactive_repulsion_gain
            ry += dy / distance * weight * self.config.reactive_repulsion_gain
        return _clip_speed(gx + rx, gy + ry, self.config.max_speed_mps)


def _unit(x: float, y: float) -> tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= 1.0e-6:
        return (0.0, 0.0)
    return (x / norm, y / norm)


def _clip_speed(x: float, y: float, max_speed: float) -> tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= 1.0e-6:
        return (0.0, 0.0)
    scale = min(float(max_speed), norm) / norm
    return (x * scale, y * scale)

