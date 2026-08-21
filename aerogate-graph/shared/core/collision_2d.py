"""2D circular-obstacle collision helpers for gate-only experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GatePostObstacle2D:
    """Planar collision approximation for one gate post or circular obstacle."""

    center_xy: tuple[float, float]
    collision_radius_m: float
    species: str = "gate_post"
    canopy_height_m: float = 8.0
    description: str = "gate post obstacle"
    usd_path: str = ""
    velocity_xy: tuple[float, float] = (0.0, 0.0)


def _distance_point_to_segment(
    point_xy: tuple[float, float],
    seg_start_xy: tuple[float, float],
    seg_end_xy: tuple[float, float],
) -> float:
    px, py = point_xy
    ax, ay = seg_start_xy
    bx, by = seg_end_xy
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


class GateObstacleMap2D:
    """Planar circular-obstacle map used by the 2D gate experiments."""

    def __init__(self, obstacles: tuple[GatePostObstacle2D, ...]) -> None:
        self.obstacles = obstacles

    @classmethod
    def empty(cls) -> "GateObstacleMap2D":
        """Construct an obstacle-free map for empty-scene curricula."""

        return cls(())

    @classmethod
    def from_gate(cls, *, gate_post_radius_scale: float = 1.0) -> "GateObstacleMap2D":
        """Return the default gate-only map."""

        _ = gate_post_radius_scale
        return cls.empty()

    @classmethod
    def from_legacy_gate(cls, *, gate_post_radius_scale: float = 1.0) -> "GateObstacleMap2D":
        """Compatibility alias for older call sites."""

        return cls.from_gate(gate_post_radius_scale=gate_post_radius_scale)

    def __len__(self) -> int:
        return len(self.obstacles)

    def min_signed_distance(self, point_xy: tuple[float, float], drone_radius_m: float = 0.0) -> float:
        """Return the smallest signed clearance to any obstacle."""

        if not self.obstacles:
            return float("inf")
        clearance = []
        for obstacle in self.obstacles:
            distance = math.hypot(
                float(point_xy[0]) - obstacle.center_xy[0],
                float(point_xy[1]) - obstacle.center_xy[1],
            )
            clearance.append(distance - obstacle.collision_radius_m - float(drone_radius_m))
        return min(clearance)

    def collides_point(self, point_xy: tuple[float, float], drone_radius_m: float = 0.0) -> bool:
        """Check whether a planar point intersects any obstacle disk."""

        return self.min_signed_distance(point_xy, drone_radius_m=drone_radius_m) <= 0.0

    def colliding_obstacles(
        self,
        point_xy: tuple[float, float],
        drone_radius_m: float = 0.0,
    ) -> tuple[GatePostObstacle2D, ...]:
        """Return all obstacles intersecting the planar point."""

        matches = []
        for obstacle in self.obstacles:
            distance = math.hypot(
                float(point_xy[0]) - obstacle.center_xy[0],
                float(point_xy[1]) - obstacle.center_xy[1],
            )
            if distance <= obstacle.collision_radius_m + float(drone_radius_m):
                matches.append(obstacle)
        return tuple(matches)

    def query_local(self, center_xy: tuple[float, float], radius_m: float) -> tuple[GatePostObstacle2D, ...]:
        """Return obstacles whose centers lie within a query radius."""

        matches = []
        for obstacle in self.obstacles:
            distance = math.hypot(
                float(center_xy[0]) - obstacle.center_xy[0],
                float(center_xy[1]) - obstacle.center_xy[1],
            )
            if distance <= float(radius_m):
                matches.append(obstacle)
        return tuple(matches)

    def segment_collides(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        drone_radius_m: float = 0.0,
    ) -> bool:
        """Check line-of-motion collision against the obstacle disks."""

        for obstacle in self.obstacles:
            distance = _distance_point_to_segment(obstacle.center_xy, start_xy, end_xy)
            if distance <= obstacle.collision_radius_m + float(drone_radius_m):
                return True
        return False
