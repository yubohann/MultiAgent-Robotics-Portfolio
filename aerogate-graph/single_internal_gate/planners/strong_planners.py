"""Eval-only EGO-Planner and Fast-Planner baselines."""

from __future__ import annotations

import math
import time

from single_internal_gate.configs.experiment_config import Exp2PlannerConfig
from single_internal_gate.planners.classic_planners import AStarPlanner, ThetaStarPlanner
from single_internal_gate.planners.interfaces import PlannerResult, PlannerTask2D, path_length


_Point = tuple[float, float]


def _distance(a: _Point, b: _Point) -> float:
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _collides(task: PlannerTask2D, a: _Point, b: _Point, margin_m: float) -> bool:
    return task.obstacles_2d.segment_collides(a, b, drone_radius_m=float(task.drone_radius_m) + float(max(margin_m, 0.0)))


def _point_clearance(task: PlannerTask2D, point: _Point, margin_m: float) -> float:
    return float(task.obstacles_2d.min_signed_distance(point, drone_radius_m=float(task.drone_radius_m) + float(max(margin_m, 0.0))))


def _valid_path(task: PlannerTask2D, path: tuple[_Point, ...]) -> bool:
    return len(path) >= 2 and not any(
        task.obstacles_2d.segment_collides(a, b, drone_radius_m=task.drone_radius_m)
        for a, b in zip(path[:-1], path[1:])
    )


def _dedupe(path: list[_Point] | tuple[_Point, ...]) -> tuple[_Point, ...]:
    out: list[_Point] = []
    for point in path:
        candidate = (float(point[0]), float(point[1]))
        if not out or _distance(out[-1], candidate) > 1.0e-6:
            out.append(candidate)
    return tuple(out)


def _sample_polyline(path: tuple[_Point, ...], spacing_m: float) -> list[_Point]:
    if len(path) <= 1:
        return list(path)
    points = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        seg = _distance(a, b)
        steps = max(1, int(math.ceil(seg / max(float(spacing_m), 1.0e-6))))
        for idx in range(1, steps + 1):
            t = idx / steps
            points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return points


def _shortcut(task: PlannerTask2D, path: tuple[_Point, ...], margin_m: float) -> tuple[_Point, ...]:
    if len(path) <= 2:
        return path
    out = [path[0]]
    idx = 0
    while idx < len(path) - 1:
        nxt = len(path) - 1
        while nxt > idx + 1 and _collides(task, path[idx], path[nxt], margin_m):
            nxt -= 1
        out.append(path[nxt])
        idx = nxt
    return _dedupe(out)


def _smooth(task: PlannerTask2D, path: tuple[_Point, ...], margin_m: float, iterations: int, obstacle_gain: float) -> tuple[_Point, ...]:
    if len(path) <= 2:
        return path
    pts = _sample_polyline(path, spacing_m=1.1)
    if len(pts) <= 2:
        return path
    xmin, xmax = task.world_x_bounds_m
    ymin, ymax = task.world_y_bounds_m
    obstacle_influence = 2.0 + margin_m
    for _ in range(int(iterations)):
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            prev_p = pts[i - 1]
            curr = pts[i]
            next_p = pts[i + 1]
            smooth_x = 0.58 * curr[0] + 0.21 * (prev_p[0] + next_p[0])
            smooth_y = 0.58 * curr[1] + 0.21 * (prev_p[1] + next_p[1])
            clearance = _point_clearance(task, curr, margin_m=0.0)
            push_x = 0.0
            push_y = 0.0
            if clearance < obstacle_influence:
                for obstacle in task.obstacles_2d.query_local(curr, radius_m=obstacle_influence + float(task.drone_radius_m) + 1.0):
                    dx = curr[0] - obstacle.center_xy[0]
                    dy = curr[1] - obstacle.center_xy[1]
                    dist = max(math.hypot(dx, dy), 1.0e-6)
                    signed = dist - obstacle.collision_radius_m - float(task.drone_radius_m)
                    if signed < obstacle_influence:
                        scale = obstacle_gain * (obstacle_influence - signed) / obstacle_influence
                        push_x += scale * dx / dist
                        push_y += scale * dy / dist
            candidate = (
                float(max(xmin, min(xmax, smooth_x + push_x))),
                float(max(ymin, min(ymax, smooth_y + push_y))),
            )
            if _collides(task, prev_p, candidate, margin_m) or _collides(task, candidate, next_p, margin_m):
                candidate = curr
            new_pts.append(candidate)
        new_pts.append(pts[-1])
        pts = new_pts
    return _shortcut(task, _dedupe(pts), margin_m=max(0.0, 0.5 * margin_m))


def _finish(name: str, task: PlannerTask2D, path: tuple[_Point, ...], start_time: float, num_replans: int = 0) -> PlannerResult:
    path = _dedupe(path)
    return PlannerResult(
        planner_name=name,
        success=_valid_path(task, path),
        path_xy=path,
        path_length_m=path_length(path),
        planning_time_ms=float((time.perf_counter() - start_time) * 1000.0),
        num_replans=int(num_replans),
    )


class EgoPlanner:
    name = "ego_planner"

    def __init__(self, config: Exp2PlannerConfig) -> None:
        self.config = config
        self.seed_planner = ThetaStarPlanner(config, resolution_m=0.42)

    @property
    def safety_margin_m(self) -> float:
        return 0.45 * float(getattr(self.config, "safety_margin_m", 0.0) or 0.0)

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        start = time.perf_counter()
        seed = self.seed_planner.plan(task)
        if not seed.success:
            seed = AStarPlanner(self.config, resolution_m=0.36).plan(task)
        if not seed.success:
            return _finish(self.name, task, tuple(seed.path_xy), start)
        path = _smooth(task, seed.path_xy, margin_m=self.safety_margin_m, iterations=10, obstacle_gain=0.38)
        path = _shortcut(task, path, margin_m=0.15 * self.safety_margin_m)
        return _finish(self.name, task, path, start, num_replans=seed.num_replans)


class FastPlanner:
    name = "fast_planner"

    def __init__(self, config: Exp2PlannerConfig) -> None:
        self.config = config
        self.seed_planner = AStarPlanner(config, resolution_m=0.34)

    @property
    def safety_margin_m(self) -> float:
        return 0.30 * float(getattr(self.config, "safety_margin_m", 0.0) or 0.0)

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        start = time.perf_counter()
        seed = self.seed_planner.plan(task)
        if not seed.success:
            seed = ThetaStarPlanner(self.config, resolution_m=0.42).plan(task)
        if not seed.success:
            return _finish(self.name, task, tuple(seed.path_xy), start)
        path = _shortcut(task, seed.path_xy, margin_m=self.safety_margin_m)
        path = _smooth(task, path, margin_m=self.safety_margin_m, iterations=5, obstacle_gain=0.22)
        return _finish(self.name, task, path, start, num_replans=seed.num_replans)





