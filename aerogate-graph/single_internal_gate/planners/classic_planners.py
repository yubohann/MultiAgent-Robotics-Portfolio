"""Planner-only baselines for 2D gate tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
import random
import time
from typing import Iterable

from single_internal_gate.configs.experiment_config import Exp2PlannerConfig
from single_internal_gate.planners.interfaces import PlannerResult, PlannerTask2D, path_length


_Point = tuple[float, float]
_Cell = tuple[int, int]


def _distance(a: _Point, b: _Point) -> float:
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _collides(task: PlannerTask2D, a: _Point, b: _Point, safety_margin_m: float) -> bool:
    radius = float(task.drone_radius_m) + float(max(safety_margin_m, 0.0))
    return task.obstacles_2d.segment_collides(a, b, drone_radius_m=radius)


def _point_collides(task: PlannerTask2D, point: _Point, safety_margin_m: float) -> bool:
    radius = float(task.drone_radius_m) + float(max(safety_margin_m, 0.0))
    return task.obstacles_2d.collides_point(point, drone_radius_m=radius)


def _clamp_point(task: PlannerTask2D, point: _Point) -> _Point:
    xmin, xmax = task.world_x_bounds_m
    ymin, ymax = task.world_y_bounds_m
    return (
        float(max(xmin, min(xmax, point[0]))),
        float(max(ymin, min(ymax, point[1]))),
    )


def _dedupe_path(path: Iterable[_Point]) -> tuple[_Point, ...]:
    result: list[_Point] = []
    for point in path:
        candidate = (float(point[0]), float(point[1]))
        if not result or _distance(result[-1], candidate) > 1.0e-6:
            result.append(candidate)
    return tuple(result)


def _validate_path(task: PlannerTask2D, path: tuple[_Point, ...]) -> bool:
    if len(path) < 2:
        return False
    return not any(task.obstacles_2d.segment_collides(a, b, drone_radius_m=task.drone_radius_m) for a, b in zip(path[:-1], path[1:]))


def _finish(name: str, task: PlannerTask2D, path: tuple[_Point, ...], start_time: float, replans: int = 0) -> PlannerResult:
    path = _dedupe_path(path)
    success = _validate_path(task, path)
    if not success:
        path = tuple(path)
    return PlannerResult(
        planner_name=name,
        success=bool(success),
        path_xy=path,
        path_length_m=path_length(path),
        planning_time_ms=float((time.perf_counter() - start_time) * 1000.0),
        num_replans=int(replans),
    )


@dataclass(frozen=True)
class _GridSpec:
    resolution_m: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    nx: int
    ny: int

    @classmethod
    def from_task(cls, task: PlannerTask2D, resolution_m: float) -> "_GridSpec":
        xmin, xmax = (float(task.world_x_bounds_m[0]), float(task.world_x_bounds_m[1]))
        ymin, ymax = (float(task.world_y_bounds_m[0]), float(task.world_y_bounds_m[1]))
        res = float(max(resolution_m, 0.05))
        return cls(
            resolution_m=res,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            nx=int(round((xmax - xmin) / res)) + 1,
            ny=int(round((ymax - ymin) / res)) + 1,
        )

    def point_to_cell(self, point: _Point) -> _Cell:
        x = int(round((float(point[0]) - self.xmin) / self.resolution_m))
        y = int(round((float(point[1]) - self.ymin) / self.resolution_m))
        return (max(0, min(self.nx - 1, x)), max(0, min(self.ny - 1, y)))

    def cell_to_point(self, cell: _Cell) -> _Point:
        return (self.xmin + float(cell[0]) * self.resolution_m, self.ymin + float(cell[1]) * self.resolution_m)

    def in_bounds(self, cell: _Cell) -> bool:
        return 0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny


class _GridPlanner:
    def __init__(self, config: Exp2PlannerConfig, *, resolution_m: float = 0.45) -> None:
        self.config = config
        self.resolution_m = float(resolution_m)

    @property
    def safety_margin_m(self) -> float:
        return float(getattr(self.config, "safety_margin_m", 0.0) or 0.0)

    def _neighbors(self, cell: _Cell) -> tuple[_Cell, ...]:
        result: list[_Cell] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                result.append((cell[0] + dx, cell[1] + dy))
        return tuple(result)

    def _cell_free(self, task: PlannerTask2D, spec: _GridSpec, cell: _Cell) -> bool:
        return spec.in_bounds(cell) and not _point_collides(task, spec.cell_to_point(cell), self.safety_margin_m)

    def _reconstruct(self, came_from: dict[_Cell, _Cell], current: _Cell, spec: _GridSpec, task: PlannerTask2D) -> tuple[_Point, ...]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        points = [spec.cell_to_point(cell) for cell in cells]
        if points:
            points[0] = task.start_xy
            points[-1] = task.goal_xy
        return _dedupe_path(points)


class AStarPlanner(_GridPlanner):
    name = "astar"

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        start_time = time.perf_counter()
        if not _collides(task, task.start_xy, task.goal_xy, self.safety_margin_m):
            return _finish(self.name, task, (task.start_xy, task.goal_xy), start_time)
        spec = _GridSpec.from_task(task, self.resolution_m)
        start = spec.point_to_cell(task.start_xy)
        goal = spec.point_to_cell(task.goal_xy)
        path = self._astar(task, spec, start, goal)
        return _finish(self.name, task, path, start_time)

    def _astar(self, task: PlannerTask2D, spec: _GridSpec, start: _Cell, goal: _Cell) -> tuple[_Point, ...]:
        open_heap: list[tuple[float, float, _Cell]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))
        came_from: dict[_Cell, _Cell] = {}
        g_score: dict[_Cell, float] = {start: 0.0}
        visited: set[_Cell] = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                return self._reconstruct(came_from, current, spec, task)

            current_point = spec.cell_to_point(current)
            for neighbor in self._neighbors(current):
                if not self._cell_free(task, spec, neighbor):
                    continue
                neighbor_point = spec.cell_to_point(neighbor)
                if _collides(task, current_point, neighbor_point, self.safety_margin_m):
                    continue
                tentative = current_g + _distance(current_point, neighbor_point)
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + _distance(neighbor_point, task.goal_xy)
                heapq.heappush(open_heap, (priority, tentative, neighbor))
        return ()


class ThetaStarPlanner(AStarPlanner):
    name = "theta_star"

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        start_time = time.perf_counter()
        if not _collides(task, task.start_xy, task.goal_xy, self.safety_margin_m):
            return _finish(self.name, task, (task.start_xy, task.goal_xy), start_time)
        spec = _GridSpec.from_task(task, self.resolution_m)
        raw = self._astar(task, spec, spec.point_to_cell(task.start_xy), spec.point_to_cell(task.goal_xy))
        if raw:
            raw = self._shortcut(task, raw)
        return _finish(self.name, task, raw, start_time)

    def _shortcut(self, task: PlannerTask2D, path: tuple[_Point, ...]) -> tuple[_Point, ...]:
        if len(path) <= 2:
            return path
        result = [path[0]]
        index = 0
        while index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > index + 1:
                if not _collides(task, path[index], path[next_index], self.safety_margin_m):
                    break
                next_index -= 1
            result.append(path[next_index])
            index = next_index
        return _dedupe_path(result)


class HeuristicPlanner:
    name = "heuristic"

    def __init__(self, config: Exp2PlannerConfig) -> None:
        self.config = config

    @property
    def safety_margin_m(self) -> float:
        return float(getattr(self.config, "safety_margin_m", 0.0) or 0.0)

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        start_time = time.perf_counter()
        if not _collides(task, task.start_xy, task.goal_xy, self.safety_margin_m):
            return _finish(self.name, task, (task.start_xy, task.goal_xy), start_time)
        candidates = self._candidate_paths(task)
        best: tuple[_Point, ...] = ()
        best_length = float("inf")
        for candidate in candidates:
            candidate = _dedupe_path(_clamp_point(task, point) for point in candidate)
            if not _validate_path(task, candidate):
                continue
            length = path_length(candidate)
            if length < best_length:
                best = candidate
                best_length = length
        return _finish(self.name, task, best, start_time)

    def _candidate_paths(self, task: PlannerTask2D) -> tuple[tuple[_Point, ...], ...]:
        xmin, xmax = task.world_x_bounds_m
        ymin, ymax = task.world_y_bounds_m
        mid_x = 0.5 * (float(task.start_xy[0]) + float(task.goal_xy[0]))
        margin = float(task.drone_radius_m) + self.safety_margin_m + 0.45
        lane_candidates = (
            0.0,
            ymax - margin,
            ymin + margin,
            0.65 * ymax,
            0.65 * ymin,
            0.38 * ymax,
            0.38 * ymin,
        )
        paths: list[tuple[_Point, ...]] = []
        for lane_y in lane_candidates:
            lane = float(max(ymin + margin, min(ymax - margin, lane_y)))
            paths.append(
                (
                    task.start_xy,
                    (float(task.start_xy[0]), lane),
                    (mid_x, lane),
                    (float(task.goal_xy[0]), lane),
                    task.goal_xy,
                )
            )
            paths.append(
                (
                    task.start_xy,
                    (float(xmin) + margin, lane),
                    (mid_x, lane),
                    (float(xmax) - margin, lane),
                    task.goal_xy,
                )
            )
        return tuple(paths)


@dataclass
class _RrtNode:
    point: _Point
    parent: int | None
    cost: float


class RRTStarPlanner:
    name = "rrt_star"

    def __init__(
        self,
        config: Exp2PlannerConfig,
        *,
        max_iterations: int = 160,
        step_size_m: float = 1.35,
        goal_sample_rate: float = 0.10,
        rewire_radius_m: float = 2.75,
        seed_offset: int = 0,
        time_budget_ms: float | None = None,
    ) -> None:
        self.config = config
        self.max_iterations = int(max_iterations)
        self.step_size_m = float(step_size_m)
        self.goal_sample_rate = float(goal_sample_rate)
        self.rewire_radius_m = float(rewire_radius_m)
        self.seed_offset = int(seed_offset)
        self.time_budget_ms = float(time_budget_ms if time_budget_ms is not None else max(20.0, config.latency_budget_ms))

    @property
    def safety_margin_m(self) -> float:
        return float(getattr(self.config, "safety_margin_m", 0.0) or 0.0)

    def plan(self, task: PlannerTask2D) -> PlannerResult:
        start_time = time.perf_counter()
        if not _collides(task, task.start_xy, task.goal_xy, self.safety_margin_m):
            return _finish(self.name, task, (task.start_xy, task.goal_xy), start_time)
        seed_material = f"{task.task_id}|{self.name}|{self.seed_offset}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") & 0xFFFFFFFF
        rng = random.Random(seed)
        path = self._rrt_star(task, rng, start_time)
        return _finish(self.name, task, path, start_time)

    def _sample(self, task: PlannerTask2D, rng: random.Random, nodes: list[_RrtNode], best_goal: int | None) -> _Point:
        if rng.random() < self.goal_sample_rate:
            return task.goal_xy
        xmin, xmax = task.world_x_bounds_m
        ymin, ymax = task.world_y_bounds_m
        return (rng.uniform(float(xmin), float(xmax)), rng.uniform(float(ymin), float(ymax)))

    def _rrt_star(self, task: PlannerTask2D, rng: random.Random, start_time: float) -> tuple[_Point, ...]:
        if _point_collides(task, task.start_xy, self.safety_margin_m) or _point_collides(task, task.goal_xy, self.safety_margin_m):
            return ()
        nodes = [_RrtNode(point=task.start_xy, parent=None, cost=0.0)]
        best_goal: int | None = None
        for iteration in range(self.max_iterations):
            if iteration and iteration % 8 == 0 and (time.perf_counter() - start_time) * 1000.0 >= self.time_budget_ms:
                break
            sample = self._sample(task, rng, nodes, best_goal)
            nearest_index = min(range(len(nodes)), key=lambda idx: _distance(nodes[idx].point, sample))
            new_point = self._steer(nodes[nearest_index].point, sample, task)
            if _point_collides(task, new_point, self.safety_margin_m):
                continue
            if _collides(task, nodes[nearest_index].point, new_point, self.safety_margin_m):
                continue
            near = [
                idx for idx, node in enumerate(nodes)
                if _distance(node.point, new_point) <= self.rewire_radius_m
                and not _collides(task, node.point, new_point, self.safety_margin_m)
            ]
            parent = nearest_index
            cost = nodes[nearest_index].cost + _distance(nodes[nearest_index].point, new_point)
            for idx in near:
                candidate_cost = nodes[idx].cost + _distance(nodes[idx].point, new_point)
                if candidate_cost < cost:
                    parent = idx
                    cost = candidate_cost
            nodes.append(_RrtNode(point=new_point, parent=parent, cost=cost))
            new_index = len(nodes) - 1
            for idx in near:
                if idx == parent or self._is_ancestor(nodes, idx, new_index):
                    continue
                candidate_cost = cost + _distance(new_point, nodes[idx].point)
                if candidate_cost < nodes[idx].cost:
                    nodes[idx].parent = new_index
                    nodes[idx].cost = candidate_cost
            if _distance(new_point, task.goal_xy) <= self.step_size_m * 1.35 and not _collides(task, new_point, task.goal_xy, self.safety_margin_m):
                goal_cost = cost + _distance(new_point, task.goal_xy)
                if best_goal is None or goal_cost < nodes[best_goal].cost:
                    nodes.append(_RrtNode(point=task.goal_xy, parent=new_index, cost=goal_cost))
                    best_goal = len(nodes) - 1
        if best_goal is None:
            return ()
        return self._extract(nodes, best_goal)

    def _steer(self, source: _Point, target: _Point, task: PlannerTask2D) -> _Point:
        dist = _distance(source, target)
        if dist <= self.step_size_m:
            return _clamp_point(task, target)
        scale = self.step_size_m / max(dist, 1.0e-9)
        return _clamp_point(task, (source[0] + (target[0] - source[0]) * scale, source[1] + (target[1] - source[1]) * scale))

    def _extract(self, nodes: list[_RrtNode], index: int) -> tuple[_Point, ...]:
        points: list[_Point] = []
        current: int | None = index
        visited: set[int] = set()
        while current is not None and current not in visited and 0 <= current < len(nodes):
            visited.add(current)
            node = nodes[current]
            points.append(node.point)
            current = node.parent
        if current is not None:
            return ()
        points.reverse()
        return _dedupe_path(points)

    def _is_ancestor(self, nodes: list[_RrtNode], ancestor: int, node_index: int) -> bool:
        current: int | None = node_index
        visited: set[int] = set()
        while current is not None and current not in visited and 0 <= current < len(nodes):
            if current == ancestor:
                return True
            visited.add(current)
            current = nodes[current].parent
        return False


class InformedRRTStarPlanner(RRTStarPlanner):
    name = "informed_rrt_star"

    def __init__(self, config: Exp2PlannerConfig) -> None:
        super().__init__(
            config,
            max_iterations=180,
            step_size_m=1.35,
            goal_sample_rate=0.08,
            rewire_radius_m=2.9,
            seed_offset=9173,
            time_budget_ms=max(25.0, config.latency_budget_ms),
        )

    def _sample(self, task: PlannerTask2D, rng: random.Random, nodes: list[_RrtNode], best_goal: int | None) -> _Point:
        if best_goal is None:
            return super()._sample(task, rng, nodes, best_goal)
        if rng.random() < self.goal_sample_rate:
            return task.goal_xy
        best_cost = max(float(nodes[best_goal].cost), _distance(task.start_xy, task.goal_xy) + 1.0e-6)
        c_min = _distance(task.start_xy, task.goal_xy)
        center = ((task.start_xy[0] + task.goal_xy[0]) * 0.5, (task.start_xy[1] + task.goal_xy[1]) * 0.5)
        a = best_cost * 0.5
        b = max(math.sqrt(max(best_cost * best_cost - c_min * c_min, 1.0e-6)) * 0.5, self.step_size_m)
        theta = math.atan2(task.goal_xy[1] - task.start_xy[1], task.goal_xy[0] - task.start_xy[0])
        for _ in range(24):
            radius = math.sqrt(rng.random())
            angle = rng.uniform(0.0, 2.0 * math.pi)
            local_x = a * radius * math.cos(angle)
            local_y = b * radius * math.sin(angle)
            x = center[0] + math.cos(theta) * local_x - math.sin(theta) * local_y
            y = center[1] + math.sin(theta) * local_x + math.cos(theta) * local_y
            point = _clamp_point(task, (x, y))
            if not _point_collides(task, point, self.safety_margin_m):
                return point
        return super()._sample(task, rng, nodes, best_goal)

