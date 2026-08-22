"""Conservative evaluator-side safety checks for real HM3D CF2X execution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Point3 = tuple[float, float, float]


def _point(values: Sequence[float], name: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must be finite")
    return point  # type: ignore[return-value]


def required_segment_sample_clearance_m(
    required_clearance_m: float, maximum_sample_spacing_m: float
) -> float:
    """Return the point-clearance threshold that proves a linear segment safe.

    Distance to a fixed collision mesh is 1-Lipschitz.  If samples are at
    most ``maximum_sample_spacing_m`` apart, every point on a line segment is
    within half that spacing of one sample.  Requiring this returned value at
    every sample therefore proves the requested continuous centreline
    clearance without assuming that the ESDF is exact.
    """

    required = float(required_clearance_m)
    spacing = float(maximum_sample_spacing_m)
    if not math.isfinite(required) or required < 0.0:
        raise ValueError("required clearance must be finite and non-negative")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("maximum sample spacing must be finite and positive")
    return required + spacing / 2.0


@dataclass(frozen=True, slots=True)
class ClearanceAssessment:
    """One conservative point-clearance result without exposing the field."""

    in_field_bounds: bool
    sampled_distance_m: float | None
    conservative_distance_m: float | None
    discretization_margin_m: float

    def admits(self, required_clearance_m: float) -> bool:
        required = float(required_clearance_m)
        if not math.isfinite(required) or required < 0.0:
            raise ValueError("required clearance must be finite and non-negative")
        return bool(
            self.in_field_bounds
            and self.conservative_distance_m is not None
            and self.conservative_distance_m + 1.0e-12 >= required
        )


@dataclass(frozen=True, slots=True)
class ConservativeVoxelClearance:
    """A lower-bound distance query over an evaluator-private collision ESDF.

    The input distance grid is derived from the exact collision mesh used by
    PhysX.  A query takes the minimum of the local lattice cells and subtracts
    one voxel diagonal, covering both the query-to-grid-centre and the
    surface-to-occupied-voxel discretization error.  It is therefore a
    conservative admission test, rather than a method-visible map feature.
    """

    collision_distance_m: np.ndarray
    origin_center_m: Point3
    resolution_m: float

    def __post_init__(self) -> None:
        distance = np.asarray(self.collision_distance_m, dtype=np.float64)
        if distance.ndim != 3 or min(distance.shape) < 1 or not np.all(np.isfinite(distance)):
            raise ValueError("collision distance field must be a finite non-empty 3-D array")
        resolution = float(self.resolution_m)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("ESDF resolution must be finite and positive")
        object.__setattr__(self, "collision_distance_m", distance)
        object.__setattr__(self, "origin_center_m", _point(self.origin_center_m, "origin_center_m"))
        object.__setattr__(self, "resolution_m", resolution)

    @property
    def discretization_margin_m(self) -> float:
        return math.sqrt(3.0) * self.resolution_m

    def assess(self, point: Sequence[float]) -> ClearanceAssessment:
        query = _point(point, "clearance_query_point")
        relative = tuple(
            (query[axis] - self.origin_center_m[axis]) / self.resolution_m for axis in range(3)
        )
        shape = self.collision_distance_m.shape
        if any(relative[axis] < -0.5 or relative[axis] > shape[axis] - 0.5 for axis in range(3)):
            return ClearanceAssessment(False, None, None, self.discretization_margin_m)
        neighboring_indices: list[tuple[int, int, int]] = []
        for ix in {math.floor(relative[0]), math.ceil(relative[0])}:
            for iy in {math.floor(relative[1]), math.ceil(relative[1])}:
                for iz in {math.floor(relative[2]), math.ceil(relative[2])}:
                    if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                        neighboring_indices.append((ix, iy, iz))
        if not neighboring_indices:
            return ClearanceAssessment(False, None, None, self.discretization_margin_m)
        sampled = min(float(self.collision_distance_m[index]) for index in neighboring_indices)
        return ClearanceAssessment(
            True,
            sampled,
            max(0.0, sampled - self.discretization_margin_m),
            self.discretization_margin_m,
        )


@dataclass(frozen=True, slots=True)
class TimedPolyline:
    """A transit centreline with an explicit planned time window."""

    agent_id: str
    path_m: tuple[Point3, ...]
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("timed polyline agent ID must be non-empty")
        path = tuple(_point(point, "timed_polyline_point") for point in self.path_m)
        if len(path) < 2:
            raise ValueError("timed polyline needs at least two points")
        if any(
            math.dist(left, right) <= 1.0e-9
            for left, right in zip(path[:-1], path[1:], strict=True)
        ):
            raise ValueError("timed polyline cannot contain zero-length segments")
        start = float(self.start_s)
        end = float(self.end_s)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise ValueError("timed polyline requires 0 <= start < end")
        object.__setattr__(self, "path_m", path)
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)

    @property
    def _segment_lengths(self) -> tuple[float, ...]:
        return tuple(
            math.dist(left, right)
            for left, right in zip(self.path_m[:-1], self.path_m[1:], strict=True)
        )

    @property
    def _total_length_m(self) -> float:
        return sum(self._segment_lengths)

    def _breakpoints(self) -> tuple[float, ...]:
        elapsed = 0.0
        times = [self.start_s]
        for length in self._segment_lengths:
            elapsed += length / self._total_length_m * (self.end_s - self.start_s)
            times.append(self.start_s + elapsed)
        times[-1] = self.end_s
        return tuple(times)

    def position_at(self, timestamp_s: float) -> Point3:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("trajectory timestamp must be finite")
        if timestamp <= self.start_s:
            return self.path_m[0]
        if timestamp >= self.end_s:
            return self.path_m[-1]
        times = self._breakpoints()
        for index, (start, end) in enumerate(zip(times[:-1], times[1:], strict=True)):
            if timestamp <= end + 1.0e-12:
                fraction = (timestamp - start) / (end - start)
                left, right = self.path_m[index], self.path_m[index + 1]
                return tuple(
                    left[axis] + fraction * (right[axis] - left[axis]) for axis in range(3)
                )  # type: ignore[return-value]
        raise RuntimeError("trajectory position lookup exhausted its own segments")


@dataclass(frozen=True, slots=True)
class TimedStationary:
    """A stationary agent that still occupies space over a planned interval."""

    agent_id: str
    position_m: Point3
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("timed stationary agent ID must be non-empty")
        position = _point(self.position_m, "timed_stationary_position")
        start = float(self.start_s)
        end = float(self.end_s)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise ValueError("timed stationary route requires 0 <= start < end")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)

    def _breakpoints(self) -> tuple[float, float]:
        return (self.start_s, self.end_s)

    def position_at(self, timestamp_s: float) -> Point3:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("trajectory timestamp must be finite")
        return self.position_m


@dataclass(frozen=True, slots=True)
class SynchronizedSeparationAssessment:
    """Exact piecewise-linear minimum separation over all simultaneous routes."""

    required_separation_m: float
    minimum_separation_m: float
    minimum_time_s: float
    closest_agent_pair: tuple[str, str]

    @property
    def admitted(self) -> bool:
        return self.minimum_separation_m + 1.0e-12 >= self.required_separation_m

    def to_public_dict(self) -> dict[str, object]:
        return {
            "required_separation_m": self.required_separation_m,
            "minimum_separation_m": self.minimum_separation_m,
            "minimum_time_s": self.minimum_time_s,
            "closest_agent_pair": list(self.closest_agent_pair),
            "admitted": self.admitted,
        }


@dataclass(frozen=True, slots=True)
class RouteTubeSeparationAssessment:
    """Minimum spatial separation of every pair of planned route centrelines.

    This is intentionally independent of the nominal timing model.  It is a
    conservative admission certificate for an executor that advances waypoint
    references after physical settling, rather than an enforceable temporal
    reservation system.  A moving route and a stationary relay are both
    represented as spatial segments.
    """

    required_separation_m: float
    minimum_route_separation_m: float
    closest_agent_pair: tuple[str, str]
    closest_segment_indices: tuple[int, int]
    closest_points_m: tuple[Point3, Point3]

    @property
    def admitted(self) -> bool:
        return self.minimum_route_separation_m + 1.0e-12 >= self.required_separation_m

    def to_public_dict(self) -> dict[str, object]:
        return {
            "required_separation_m": self.required_separation_m,
            "minimum_route_separation_m": self.minimum_route_separation_m,
            "closest_agent_pair": list(self.closest_agent_pair),
            "closest_segment_indices": list(self.closest_segment_indices),
            "closest_points_m": [list(point) for point in self.closest_points_m],
            "admitted": self.admitted,
        }


@dataclass(frozen=True, slots=True)
class CollisionAvoidanceRecoveryAssessment:
    """Certificate for a one-vehicle escape from a planning-envelope overlap.

    This is intentionally narrower than normal joint-route admission.  It is
    only for a physically safe fleet that has entered the tracking envelope
    but not the collision envelope.  One vehicle follows a diverging route
    while every other vehicle remains stationary; the endpoint must restore
    the normal planning margin before ordinary candidate generation resumes.
    """

    physical_minimum_separation_m: float
    planned_minimum_separation_m: float
    recovery_endpoint_minimum_separation_m: float
    boundary_speed_limit_mps: float
    recovery_agent_id: str
    initial_minimum_separation_m: float
    endpoint_minimum_separation_m: float
    maximum_boundary_speed_mps: float
    synchronized_physical_assessment: SynchronizedSeparationAssessment
    route_tube_physical_assessment: RouteTubeSeparationAssessment
    nonconverging: bool
    rejection_reasons: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return not self.rejection_reasons

    def to_public_dict(self) -> dict[str, object]:
        return {
            "physical_minimum_separation_m": self.physical_minimum_separation_m,
            "planned_minimum_separation_m": self.planned_minimum_separation_m,
            "recovery_endpoint_minimum_separation_m": self.recovery_endpoint_minimum_separation_m,
            "boundary_speed_limit_mps": self.boundary_speed_limit_mps,
            "recovery_agent_id": self.recovery_agent_id,
            "initial_minimum_separation_m": self.initial_minimum_separation_m,
            "endpoint_minimum_separation_m": self.endpoint_minimum_separation_m,
            "maximum_boundary_speed_mps": self.maximum_boundary_speed_mps,
            "synchronized_physical_assessment": self.synchronized_physical_assessment.to_public_dict(),
            "route_tube_physical_assessment": self.route_tube_physical_assessment.to_public_dict(),
            "nonconverging": self.nonconverging,
            "rejection_reasons": list(self.rejection_reasons),
            "admitted": self.admitted,
        }


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _closest_points_on_segments(
    left_start: Point3,
    left_end: Point3,
    right_start: Point3,
    right_end: Point3,
) -> tuple[Point3, Point3]:
    """Return the exact closest pair on two closed 3-D line segments."""

    left_direction = tuple(
        left_end[axis] - left_start[axis] for axis in range(3)
    )
    right_direction = tuple(
        right_end[axis] - right_start[axis] for axis in range(3)
    )
    offset = tuple(left_start[axis] - right_start[axis] for axis in range(3))
    left_length_squared = sum(value * value for value in left_direction)
    right_length_squared = sum(value * value for value in right_direction)
    offset_left = sum(left_direction[axis] * offset[axis] for axis in range(3))
    offset_right = sum(right_direction[axis] * offset[axis] for axis in range(3))
    epsilon = 1.0e-18

    if left_length_squared <= epsilon and right_length_squared <= epsilon:
        left_fraction = 0.0
        right_fraction = 0.0
    elif left_length_squared <= epsilon:
        left_fraction = 0.0
        right_fraction = _clamp_unit(offset_right / right_length_squared)
    elif right_length_squared <= epsilon:
        right_fraction = 0.0
        left_fraction = _clamp_unit(-offset_left / left_length_squared)
    else:
        directions_dot = sum(
            left_direction[axis] * right_direction[axis] for axis in range(3)
        )
        denominator = left_length_squared * right_length_squared - directions_dot**2
        left_fraction = (
            _clamp_unit(
                (directions_dot * offset_right - offset_left * right_length_squared)
                / denominator
            )
            if denominator > epsilon
            else 0.0
        )
        right_numerator = directions_dot * left_fraction + offset_right
        if right_numerator <= 0.0:
            right_fraction = 0.0
            left_fraction = _clamp_unit(-offset_left / left_length_squared)
        elif right_numerator >= right_length_squared:
            right_fraction = 1.0
            left_fraction = _clamp_unit(
                (directions_dot - offset_left) / left_length_squared
            )
        else:
            right_fraction = right_numerator / right_length_squared

    left_point = tuple(
        left_start[axis] + left_fraction * left_direction[axis] for axis in range(3)
    )
    right_point = tuple(
        right_start[axis] + right_fraction * right_direction[axis] for axis in range(3)
    )
    return left_point, right_point  # type: ignore[return-value]


def _spatial_segments(
    route: TimedPolyline | TimedStationary,
) -> tuple[tuple[int, Point3, Point3], ...]:
    if isinstance(route, TimedStationary):
        return ((0, route.position_m, route.position_m),)
    return tuple(
        (index, start, end)
        for index, (start, end) in enumerate(
            zip(route.path_m[:-1], route.path_m[1:], strict=True)
        )
    )


def assess_route_tube_separation(
    routes: Sequence[TimedPolyline | TimedStationary], *, minimum_separation_m: float
) -> RouteTubeSeparationAssessment:
    """Reject routes whose occupied spatial tubes overlap before execution.

    The assessment is deliberately stricter than
    :func:`assess_synchronized_separation`: two paths crossing at different
    *planned* times still conflict because waypoint settling makes those times
    non-binding in the current CF2X executor.  It is not claimed to prove
    safety for arbitrary asynchronous execution; that requires enforced
    temporal reservations or online replanning.
    """

    required = float(minimum_separation_m)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("minimum separation must be finite and positive")
    rows = tuple(routes)
    if len(rows) < 2:
        raise ValueError("route-tube admission requires at least two routes")
    if len({row.agent_id for row in rows}) != len(rows):
        raise ValueError("route-tube admission requires distinct agent IDs")

    best_distance = math.inf
    best_pair = ("", "")
    best_segment_indices = (-1, -1)
    best_points: tuple[Point3, Point3] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            for left_segment_index, left_start, left_end in _spatial_segments(left):
                for right_segment_index, right_start, right_end in _spatial_segments(right):
                    left_point, right_point = _closest_points_on_segments(
                        left_start, left_end, right_start, right_end
                    )
                    distance = math.dist(left_point, right_point)
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = tuple(sorted((left.agent_id, right.agent_id)))
                        best_segment_indices = (left_segment_index, right_segment_index)
                        best_points = (left_point, right_point)
    if not math.isfinite(best_distance):
        raise RuntimeError("route-tube admission did not examine any route segments")
    return RouteTubeSeparationAssessment(
        required,
        best_distance,
        best_pair,
        best_segment_indices,
        best_points,
    )


def assess_synchronized_separation(
    routes: Sequence[TimedPolyline | TimedStationary], *, minimum_separation_m: float
) -> SynchronizedSeparationAssessment:
    """Check continuous planned separation, including wait-at-endpoint intervals."""

    required = float(minimum_separation_m)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("minimum separation must be finite and positive")
    rows = tuple(routes)
    if len(rows) < 2:
        raise ValueError("separation admission requires at least two routes")
    if len({row.agent_id for row in rows}) != len(rows):
        raise ValueError("separation admission requires distinct agent IDs")

    best_distance = math.inf
    best_time = 0.0
    best_pair = ("", "")
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            timeline = sorted(
                {
                    *left._breakpoints(),
                    *right._breakpoints(),
                    left.start_s,
                    right.start_s,
                    max(left.end_s, right.end_s),
                }
            )
            for start, end in zip(timeline[:-1], timeline[1:], strict=True):
                if end <= start:
                    continue
                left_start = left.position_at(start)
                left_end = left.position_at(end)
                right_start = right.position_at(start)
                right_end = right.position_at(end)
                relative_start = tuple(left_start[axis] - right_start[axis] for axis in range(3))
                relative_delta = tuple(
                    (left_end[axis] - left_start[axis]) - (right_end[axis] - right_start[axis])
                    for axis in range(3)
                )
                denominator = sum(value * value for value in relative_delta)
                fraction = (
                    0.0
                    if denominator <= 1.0e-18
                    else min(
                        1.0,
                        max(
                            0.0,
                            -sum(relative_start[axis] * relative_delta[axis] for axis in range(3))
                            / denominator,
                        ),
                    )
                )
                distance = math.sqrt(
                    sum(
                        (relative_start[axis] + fraction * relative_delta[axis]) ** 2
                        for axis in range(3)
                    )
                )
                if distance < best_distance:
                    best_distance = distance
                    best_time = start + fraction * (end - start)
                    best_pair = tuple(sorted((left.agent_id, right.agent_id)))
    if not math.isfinite(best_distance):
        raise RuntimeError("separation admission did not examine an overlapping time interval")
    return SynchronizedSeparationAssessment(required, best_distance, best_time, best_pair)


def _minimum_pairwise_distance(points: Mapping[str, Point3]) -> float:
    if len(points) < 2:
        raise ValueError("pairwise separation requires at least two points")
    rows = tuple(points.values())
    return min(
        math.dist(left, right)
        for index, left in enumerate(rows)
        for right in rows[index + 1 :]
    )


def assess_collision_avoidance_recovery(
    routes: Sequence[TimedPolyline | TimedStationary],
    *,
    recovery_agent_id: str,
    boundary_linear_speeds_mps: Mapping[str, float],
    physical_minimum_separation_m: float,
    planned_minimum_separation_m: float,
    recovery_endpoint_minimum_separation_m: float,
    boundary_speed_limit_mps: float,
) -> CollisionAvoidanceRecoveryAssessment:
    """Certify a stationary-fleet, strictly non-converging envelope recovery.

    It never relaxes the physical CF2X separation requirement.  The exception
    is only that the *initial* planning envelope may be entered, provided the
    fleet is already near rest and the sole moving vehicle can monotonically
    increase its distance from every stationary neighbour.
    """

    physical = float(physical_minimum_separation_m)
    planned = float(planned_minimum_separation_m)
    endpoint_required = float(recovery_endpoint_minimum_separation_m)
    speed_limit = float(boundary_speed_limit_mps)
    if (
        not recovery_agent_id
        or not all(math.isfinite(value) and value > 0.0 for value in (physical, planned, endpoint_required))
        or not math.isfinite(speed_limit)
        or speed_limit < 0.0
        or physical >= planned
        or endpoint_required < planned
    ):
        raise ValueError("invalid collision-avoidance recovery contract")

    rows = tuple(routes)
    if len(rows) < 2:
        raise ValueError("collision-avoidance recovery requires at least two routes")
    ids = tuple(route.agent_id for route in rows)
    if recovery_agent_id not in ids or len(set(ids)) != len(ids):
        raise ValueError("recovery routes require distinct agents including the recovery agent")
    if set(boundary_linear_speeds_mps) != set(ids):
        raise ValueError("recovery boundary speeds must cover exactly the recovery routes")

    starts = {
        route.agent_id: (
            route.path_m[0] if isinstance(route, TimedPolyline) else route.position_m
        )
        for route in rows
    }
    endpoints = {
        route.agent_id: (
            route.path_m[-1] if isinstance(route, TimedPolyline) else route.position_m
        )
        for route in rows
    }
    initial_minimum = _minimum_pairwise_distance(starts)
    endpoint_minimum = _minimum_pairwise_distance(endpoints)
    speeds = {agent_id: float(speed) for agent_id, speed in boundary_linear_speeds_mps.items()}
    if not all(math.isfinite(speed) and speed >= 0.0 for speed in speeds.values()):
        raise ValueError("recovery boundary speeds must be finite and non-negative")
    maximum_speed = max(speeds.values())

    recovery_route = next(route for route in rows if route.agent_id == recovery_agent_id)
    stationary_routes = tuple(route for route in rows if route.agent_id != recovery_agent_id)
    nonconverging = isinstance(recovery_route, TimedPolyline) and all(
        isinstance(route, TimedStationary) for route in stationary_routes
    )
    if nonconverging:
        for segment_start, segment_end in zip(
            recovery_route.path_m[:-1], recovery_route.path_m[1:], strict=True
        ):
            displacement = tuple(
                segment_end[axis] - segment_start[axis] for axis in range(3)
            )
            for stationary in stationary_routes:
                assert isinstance(stationary, TimedStationary)
                separation = tuple(
                    segment_start[axis] - stationary.position_m[axis] for axis in range(3)
                )
                if sum(separation[axis] * displacement[axis] for axis in range(3)) < -1.0e-12:
                    nonconverging = False
                    break
            if not nonconverging:
                break

    synchronized = assess_synchronized_separation(
        rows, minimum_separation_m=physical
    )
    route_tube = assess_route_tube_separation(rows, minimum_separation_m=physical)
    reasons: list[str] = []
    if initial_minimum + 1.0e-12 < physical:
        reasons.append("initial_physical_separation_violation")
    if initial_minimum + 1.0e-12 >= planned:
        reasons.append("initial_state_not_inside_planning_envelope")
    if maximum_speed > speed_limit + 1.0e-12:
        reasons.append("boundary_not_near_stationary")
    if not nonconverging:
        reasons.append("recovery_route_converges_or_multiple_agents_move")
    if not synchronized.admitted:
        reasons.append("physical_synchronized_separation_violation")
    if not route_tube.admitted:
        reasons.append("physical_route_tube_separation_violation")
    if endpoint_minimum + 1.0e-12 < endpoint_required:
        reasons.append("recovery_endpoint_does_not_restore_planning_margin")
    return CollisionAvoidanceRecoveryAssessment(
        physical,
        planned,
        endpoint_required,
        speed_limit,
        recovery_agent_id,
        initial_minimum,
        endpoint_minimum,
        maximum_speed,
        synchronized,
        route_tube,
        nonconverging,
        tuple(reasons),
    )


__all__ = [
    "CollisionAvoidanceRecoveryAssessment",
    "ClearanceAssessment",
    "ConservativeVoxelClearance",
    "RouteTubeSeparationAssessment",
    "SynchronizedSeparationAssessment",
    "TimedPolyline",
    "TimedStationary",
    "assess_route_tube_separation",
    "assess_collision_avoidance_recovery",
    "assess_synchronized_separation",
    "required_segment_sample_clearance_m",
]
