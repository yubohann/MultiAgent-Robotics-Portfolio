"""Simple deterministic planners used by experiment-2 method ablations."""

from __future__ import annotations

from single_internal_gate.configs.experiment_config import Exp2PlannerConfig
from single_internal_gate.planners.classic_planners import (
    AStarPlanner,
    HeuristicPlanner,
    InformedRRTStarPlanner,
    RRTStarPlanner,
    ThetaStarPlanner,
)
from single_internal_gate.planners.interfaces import PlannerResult, PlannerTask2D, path_length
from single_internal_gate.planners.strong_planners import EgoPlanner, FastPlanner


class _StraightPlanner:
    name = "straight"

    def __init__(self, config: Exp2PlannerConfig) -> None:
        self.config = config

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        path = (task.start_xy, task.goal_xy)
        collision = task.obstacles_2d.segment_collides(task.start_xy, task.goal_xy, drone_radius_m=task.drone_radius_m)
        return PlannerResult(
            planner_name=self.name,
            success=not collision,
            path_xy=path,
            path_length_m=path_length(path),
            planning_time_ms=float(self.config.straight_latency_ms),
        )


class _DetourPlanner:
    name = "detour"

    def __init__(self, config: Exp2PlannerConfig) -> None:
        self.config = config

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        if not task.obstacles_2d.segment_collides(task.start_xy, task.goal_xy, drone_radius_m=task.drone_radius_m):
            path = (task.start_xy, task.goal_xy)
            return PlannerResult(self.name, True, path, path_length(path), float(self.config.detour_latency_ms))

        lane_y = _choose_lane_y(task)
        mid_x = 0.5 * (float(task.start_xy[0]) + float(task.goal_xy[0]))
        path = (
            task.start_xy,
            (task.start_xy[0], lane_y),
            (mid_x, lane_y),
            (task.goal_xy[0], lane_y),
            task.goal_xy,
        )
        collision = any(
            task.obstacles_2d.segment_collides(a, b, drone_radius_m=task.drone_radius_m)
            for a, b in zip(path[:-1], path[1:])
        )
        return PlannerResult(self.name, not collision, path, path_length(path), float(self.config.detour_latency_ms))


def _choose_lane_y(task: PlannerTask2D) -> float:
    ymin, ymax = (float(task.world_y_bounds_m[0]), float(task.world_y_bounds_m[1]))
    candidates = (0.72 * ymax, 0.72 * ymin, 0.45 * ymax, 0.45 * ymin, 0.0)
    best_y = candidates[0]
    best_clearance = -float("inf")
    mid_x = 0.5 * (float(task.start_xy[0]) + float(task.goal_xy[0]))
    for candidate_y in candidates:
        points = ((task.start_xy[0], candidate_y), (mid_x, candidate_y), (task.goal_xy[0], candidate_y))
        clearance = min(task.obstacles_2d.min_signed_distance(point, drone_radius_m=task.drone_radius_m) for point in points)
        if clearance > best_clearance:
            best_clearance = clearance
            best_y = candidate_y
    return float(max(ymin + 0.5, min(ymax - 0.5, best_y)))


def planner_names() -> tuple[str, ...]:
    return (
        "straight",
        "detour",
        "astar",
        "theta_star",
        "rrt_star",
        "informed_rrt_star",
        "heuristic",
        "ego_planner",
        "fast_planner",
    )


def create_planner(name: str, config: Exp2PlannerConfig):
    normalized = str(name).strip().lower()
    if normalized == "straight":
        return _StraightPlanner(config)
    if normalized in {"detour", "visibility"}:
        return _DetourPlanner(config)
    if normalized == "astar":
        return AStarPlanner(config)
    if normalized in {"theta_star", "theta*", "theta"}:
        return ThetaStarPlanner(config)
    if normalized in {"rrt_star", "rrt*"}:
        return RRTStarPlanner(config)
    if normalized in {"informed_rrt_star", "informed_rrt*", "irrt_star"}:
        return InformedRRTStarPlanner(config)
    if normalized == "heuristic":
        return HeuristicPlanner(config)
    if normalized in {"ego_planner", "ego-planner", "ego"}:
        return EgoPlanner(config)
    if normalized in {"fast_planner", "fast-planner", "fast"}:
        return FastPlanner(config)
    raise KeyError(f"Unknown experiment-2 planner: {name}")

