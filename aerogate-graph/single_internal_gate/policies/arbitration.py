"""Uncertainty-aware planner/reactive arbitration for experiment-2."""

from __future__ import annotations

from dataclasses import dataclass
import math

from single_internal_gate.configs.experiment_config import Exp2MethodConfig
from single_internal_gate.planners.interfaces import PlannerTask2D


@dataclass(frozen=True)
class ArbitrationDecision:
    planner_weight: float
    should_replan: bool
    shield_only: bool = False


class UncertaintyAwareArbitrator:
    def __init__(self, config: Exp2MethodConfig) -> None:
        self.config = config

    def decide(
        self,
        *,
        task: PlannerTask2D,
        position_xy: tuple[float, float],
        planner_target_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        step_index: int,
        previous_goal_distance_m: float,
    ) -> ArbitrationDecision:
        clearance = task.obstacles_2d.min_signed_distance(position_xy, drone_radius_m=task.drone_radius_m)
        planner_weight = 0.78 if clearance > 2.0 else 0.42
        current_goal_distance = math.hypot(goal_xy[0] - position_xy[0], goal_xy[1] - position_xy[1])
        progress_stalled = current_goal_distance > previous_goal_distance_m - 0.02
        target_blocked = task.obstacles_2d.segment_collides(position_xy, planner_target_xy, drone_radius_m=task.drone_radius_m)
        should_replan = bool((step_index % 25 == 0 and step_index > 0 and progress_stalled) or target_blocked)
        return ArbitrationDecision(
            planner_weight=float(planner_weight),
            should_replan=should_replan,
            shield_only=clearance < 0.55,
        )

