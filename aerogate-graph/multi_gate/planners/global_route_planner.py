"""Global route planner based on A* over an obstacle-inflated grid."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from multi_gate.configs.experiment_config import MultiGateEnvConfig, MultiPlannerConfig
from shared.core.collision_2d import GateObstacleMap2D


GridCell = tuple[int, int]


@dataclass(frozen=True)
class GlobalRoutePlan2D:
    """Global planner output used by the multi-agent environment."""

    waypoints_xy: tuple[tuple[float, float], ...]

    def heading_at(self, index: int) -> tuple[float, float]:
        if len(self.waypoints_xy) == 1:
            return (1.0, 0.0)
        idx = max(0, min(index, len(self.waypoints_xy) - 1))
        if idx >= len(self.waypoints_xy) - 1:
            start = self.waypoints_xy[idx - 1]
            end = self.waypoints_xy[idx]
        else:
            start = self.waypoints_xy[idx]
            end = self.waypoints_xy[idx + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return (1.0, 0.0)
        return (dx / norm, dy / norm)


class GlobalRoutePlanner2D:
    """Plan a centerline path that respects gate_post inflation and world bounds."""

    def __init__(
        self,
        *,
        obstacle_map: GateObstacleMap2D | None = None,
        env_config: MultiGateEnvConfig | None = None,
        planner_config: MultiPlannerConfig | None = None,
    ) -> None:
        self.obstacle_map = obstacle_map or GateObstacleMap2D.from_gate()
        self.env_config = env_config or MultiGateEnvConfig()
        self.planner_config = planner_config or MultiPlannerConfig()
        self._x_min, self._x_max = self.env_config.world_x_bounds_m
        self._y_min, self._y_max = self.env_config.world_y_bounds_m
        self._resolution = float(self.planner_config.grid_resolution_m)
        self._x_count = int(round((self._x_max - self._x_min) / self._resolution)) + 1
        self._y_count = int(round((self._y_max - self._y_min) / self._resolution)) + 1
        self._path_cache: dict[tuple[GridCell, GridCell, int], tuple[GridCell, ...]] = {}

    def plan(
        self,
        *,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        inflation_radius_m: float,
    ) -> GlobalRoutePlan2D:
        """Run A* and simplify the resulting corridor path."""

        start_cell = self._nearest_free_cell(self._to_grid(start_xy), inflation_radius_m)
        goal_cell = self._nearest_free_cell(self._to_grid(goal_xy), inflation_radius_m)
        cache_key = (
            start_cell,
            goal_cell,
            int(round(float(inflation_radius_m) * 1000.0)),
        )
        cached_path = self._path_cache.get(cache_key)
        if cached_path is None:
            path_cells = tuple(self._astar(start_cell, goal_cell, inflation_radius_m))
            self._path_cache[cache_key] = path_cells
        else:
            path_cells = list(cached_path)
        world_path = [start_xy] + [self._to_world(cell) for cell in path_cells[1:-1]] + [goal_xy]
        simplified = self._simplify_path(world_path, inflation_radius_m)
        return GlobalRoutePlan2D(waypoints_xy=tuple(simplified))

    def _astar(
        self,
        start_cell: GridCell,
        goal_cell: GridCell,
        inflation_radius_m: float,
    ) -> list[GridCell]:
        frontier: list[tuple[float, GridCell]] = [(0.0, start_cell)]
        came_from: dict[GridCell, GridCell | None] = {start_cell: None}
        g_score: dict[GridCell, float] = {start_cell: 0.0}
        visited = 0

        while frontier:
            _, current = heapq.heappop(frontier)
            visited += 1
            if visited > self.planner_config.max_search_iterations:
                break
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)
            for neighbor in self._neighbors(current):
                if self._is_occupied(neighbor, inflation_radius_m):
                    continue
                tentative = g_score[current] + self._cell_distance(current, neighbor)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    priority = tentative + self._heuristic(neighbor, goal_cell)
                    heapq.heappush(frontier, (priority, neighbor))

        return [start_cell, goal_cell]

    def _simplify_path(
        self,
        waypoints_xy: list[tuple[float, float]],
        inflation_radius_m: float,
    ) -> list[tuple[float, float]]:
        if len(waypoints_xy) <= 2:
            return waypoints_xy
        simplified = [waypoints_xy[0]]
        anchor_index = 0
        stride = max(1, int(self.planner_config.waypoint_stride))
        while anchor_index < len(waypoints_xy) - 1:
            candidate_index = len(waypoints_xy) - 1
            while candidate_index > anchor_index + stride:
                if not self.obstacle_map.segment_collides(
                    waypoints_xy[anchor_index],
                    waypoints_xy[candidate_index],
                    drone_radius_m=inflation_radius_m,
                ):
                    break
                candidate_index -= stride
            if candidate_index <= anchor_index:
                candidate_index = anchor_index + 1
            simplified.append(waypoints_xy[candidate_index])
            anchor_index = candidate_index
        return simplified

    def _nearest_free_cell(self, start_cell: GridCell, inflation_radius_m: float) -> GridCell:
        if not self._is_occupied(start_cell, inflation_radius_m):
            return start_cell
        visited = {start_cell}
        frontier = [start_cell]
        while frontier:
            cell = frontier.pop(0)
            for neighbor in self._neighbors(cell):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if not self._is_occupied(neighbor, inflation_radius_m):
                    return neighbor
                frontier.append(neighbor)
        return start_cell

    def _is_occupied(self, cell: GridCell, inflation_radius_m: float) -> bool:
        x, y = self._to_world(cell)
        margin = self._resolution * 0.25
        if x <= self._x_min + margin or x >= self._x_max - margin:
            return True
        if y <= self._y_min + margin or y >= self._y_max - margin:
            return True
        return self.obstacle_map.collides_point((x, y), drone_radius_m=inflation_radius_m)

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        return self._cell_distance(a, b)

    @staticmethod
    def _cell_distance(a: GridCell, b: GridCell) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def _neighbors(self, cell: GridCell) -> list[GridCell]:
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = cell[0] + dx
                ny = cell[1] + dy
                if 0 <= nx < self._x_count and 0 <= ny < self._y_count:
                    neighbors.append((nx, ny))
        return neighbors

    def _reconstruct_path(self, came_from: dict[GridCell, GridCell | None], current: GridCell) -> list[GridCell]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path

    def _to_grid(self, point_xy: tuple[float, float]) -> GridCell:
        x_idx = int(round((point_xy[0] - self._x_min) / self._resolution))
        y_idx = int(round((point_xy[1] - self._y_min) / self._resolution))
        x_idx = max(0, min(self._x_count - 1, x_idx))
        y_idx = max(0, min(self._y_count - 1, y_idx))
        return (x_idx, y_idx)

    def _to_world(self, cell: GridCell) -> tuple[float, float]:
        return (
            self._x_min + cell[0] * self._resolution,
            self._y_min + cell[1] * self._resolution,
        )

