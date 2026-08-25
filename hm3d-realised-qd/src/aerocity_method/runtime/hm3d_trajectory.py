"""Continuous-time high-level trajectory timing for CF2X exploration candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aerocity_method.contracts.io import finite_number, require_identifier

Point3 = tuple[float, float, float]


def _point(point: Point3, name: str) -> Point3:
    if len(point) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(finite_number(value, f"{name}[{index}]") for index, value in enumerate(point))  # type: ignore[return-value]


def segment_length_m(start: Point3, end: Point3) -> float:
    return math.sqrt(sum((start[index] - end[index]) ** 2 for index in range(3)))


def minimum_rest_to_rest_duration_s(
    distance_m: float,
    *,
    cruise_speed_mps: float,
    max_accel_mps2: float,
) -> float:
    """Return the triangular/trapezoidal travel time for one line segment."""

    distance = finite_number(distance_m, "distance_m")
    speed = finite_number(cruise_speed_mps, "cruise_speed_mps")
    acceleration = finite_number(max_accel_mps2, "max_accel_mps2")
    if distance < 0.0:
        raise ValueError("distance_m must be non-negative")
    if speed <= 0.0 or acceleration <= 0.0:
        raise ValueError("cruise speed and acceleration must be positive")
    if distance == 0.0:
        return 0.0
    acceleration_distance_m = speed**2 / acceleration
    if distance <= acceleration_distance_m:
        return 2.0 * math.sqrt(distance / acceleration)
    return distance / speed + speed / acceleration


def maximum_rest_to_rest_distance_m(
    duration_s: float,
    *,
    cruise_speed_mps: float,
    max_accel_mps2: float,
) -> float:
    """Invert the shared line profile for a fixed motion-time budget."""

    duration = finite_number(duration_s, "duration_s")
    speed = finite_number(cruise_speed_mps, "cruise_speed_mps")
    acceleration = finite_number(max_accel_mps2, "max_accel_mps2")
    if duration < 0.0:
        raise ValueError("duration_s must be non-negative")
    if speed <= 0.0 or acceleration <= 0.0:
        raise ValueError("cruise speed and acceleration must be positive")
    acceleration_time_s = speed / acceleration
    if duration <= 2.0 * acceleration_time_s:
        return 0.25 * acceleration * duration**2
    return speed * (duration - acceleration_time_s)


@dataclass(frozen=True, slots=True)
class TrajectoryTimingConfig:
    cruise_speed_mps: float
    max_accel_mps2: float
    tracking_margin_s: float = 0.0
    waypoint_blend_radius_m: float = 0.15

    def __post_init__(self) -> None:
        for name in ("cruise_speed_mps", "max_accel_mps2"):
            value = finite_number(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("tracking_margin_s", "waypoint_blend_radius_m"):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class TimedWaypoint:
    position_m: Point3
    timestamp_s: float
    stop_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _point(self.position_m, "position_m"))
        timestamp = finite_number(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        if not isinstance(self.stop_required, bool):
            raise ValueError("stop_required must be boolean")


@dataclass(frozen=True, slots=True)
class ContinuousTrajectory:
    agent_id: str
    waypoints: tuple[TimedWaypoint, ...]
    distance_m: float
    duration_s: float
    intermediate_stops: int

    def __post_init__(self) -> None:
        require_identifier(self.agent_id, "agent_id")
        if len(self.waypoints) < 2:
            raise ValueError("continuous trajectory requires at least two waypoints")
        if any(
            right.timestamp_s <= left.timestamp_s
            for left, right in zip(self.waypoints[:-1], self.waypoints[1:], strict=True)
        ):
            raise ValueError("waypoint timestamps must strictly increase")
        if self.distance_m <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("trajectory distance and duration must be positive")
        if self.intermediate_stops != sum(row.stop_required for row in self.waypoints[:-1]):
            raise ValueError("intermediate stop count disagrees with waypoints")

    @property
    def path_m(self) -> tuple[Point3, ...]:
        return tuple(row.position_m for row in self.waypoints)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "waypoints": [
                {
                    "position_m": row.position_m,
                    "timestamp_s": row.timestamp_s,
                    "stop_required": row.stop_required,
                }
                for row in self.waypoints
            ],
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "intermediate_stops": self.intermediate_stops,
        }


def _segment_duration(distance_m: float, config: TrajectoryTimingConfig) -> float:
    return minimum_rest_to_rest_duration_s(
        distance_m,
        cruise_speed_mps=config.cruise_speed_mps,
        max_accel_mps2=config.max_accel_mps2,
    )


def plan_continuous_trajectory(
    agent_id: str,
    path_m: tuple[Point3, ...],
    *,
    start_time_s: float,
    config: TrajectoryTimingConfig,
    stop_at_final: bool = True,
) -> ContinuousTrajectory:
    """Parameterize a guarded polyline as rest-to-rest waypoint segments."""

    require_identifier(agent_id, "agent_id")
    if len(path_m) < 2:
        raise ValueError("path requires at least two points")
    path = tuple(_point(point, "path_m") for point in path_m)
    start = finite_number(start_time_s, "start_time_s")
    if start < 0.0:
        raise ValueError("start_time_s must be non-negative")
    cumulative = start
    total_distance = 0.0
    rows = [TimedWaypoint(path[0], cumulative, stop_required=False)]
    for index, (left, right) in enumerate(zip(path[:-1], path[1:], strict=True), start=1):
        distance = segment_length_m(left, right)
        if distance <= 1.0e-9:
            raise ValueError("trajectory contains a zero-length segment")
        total_distance += distance
        cumulative += _segment_duration(distance, config)
        is_final = index == len(path) - 1
        rows.append(
            TimedWaypoint(
                right,
                cumulative,
                stop_required=(stop_at_final if is_final else True),
            )
        )
    cumulative += config.tracking_margin_s
    if config.tracking_margin_s:
        rows[-1] = TimedWaypoint(rows[-1].position_m, cumulative, stop_required=stop_at_final)
    return ContinuousTrajectory(
        agent_id=agent_id,
        waypoints=tuple(rows),
        distance_m=total_distance,
        duration_s=cumulative - start,
        intermediate_stops=sum(row.stop_required for row in rows[:-1]),
    )


__all__ = [
    "ContinuousTrajectory",
    "Point3",
    "TimedWaypoint",
    "TrajectoryTimingConfig",
    "maximum_rest_to_rest_distance_m",
    "minimum_rest_to_rest_duration_s",
    "plan_continuous_trajectory",
    "segment_length_m",
]
