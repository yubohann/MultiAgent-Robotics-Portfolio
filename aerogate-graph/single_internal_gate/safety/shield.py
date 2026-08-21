"""One-step safety shield for experiment-2 2D closed-loop rollouts."""

from __future__ import annotations

import math

from single_internal_gate.configs.experiment_config import Exp2MethodConfig
from single_internal_gate.planners.interfaces import PlannerTask2D


class SafetyShield2D:
    def __init__(self, config: Exp2MethodConfig) -> None:
        self.config = config

    def filter_command(
        self,
        *,
        position_xy: tuple[float, float],
        command_xy: tuple[float, float],
        task: PlannerTask2D,
    ) -> tuple[tuple[float, float], bool]:
        dt = self.config.dt_s
        next_xy = (position_xy[0] + command_xy[0] * dt, position_xy[1] + command_xy[1] * dt)
        if not task.obstacles_2d.segment_collides(position_xy, next_xy, drone_radius_m=task.drone_radius_m):
            return command_xy, False
        speed = min(math.hypot(command_xy[0], command_xy[1]), self.config.max_speed_mps)
        base_angle = math.atan2(command_xy[1], command_xy[0])
        for delta in (math.pi / 2.0, -math.pi / 2.0, math.pi / 3.0, -math.pi / 3.0, math.pi):
            candidate = (math.cos(base_angle + delta) * speed * 0.65, math.sin(base_angle + delta) * speed * 0.65)
            candidate_next = (position_xy[0] + candidate[0] * dt, position_xy[1] + candidate[1] * dt)
            if not task.obstacles_2d.segment_collides(position_xy, candidate_next, drone_radius_m=task.drone_radius_m):
                return candidate, True
        return (0.0, 0.0), True

