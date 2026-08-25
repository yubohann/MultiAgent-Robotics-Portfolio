"""Small deterministic geometry kernel used by the L0 evaluator and audits."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import Pose3D

Vec3 = tuple[float, float, float]
EPSILON = 1.0e-9


def add(first: Vec3, second: Vec3) -> Vec3:
    return tuple(a + b for a, b in zip(first, second, strict=True))  # type: ignore[return-value]


def subtract(first: Vec3, second: Vec3) -> Vec3:
    return tuple(a - b for a, b in zip(first, second, strict=True))  # type: ignore[return-value]


def scale(vector: Vec3, factor: float) -> Vec3:
    return tuple(value * factor for value in vector)  # type: ignore[return-value]


def dot(first: Vec3, second: Vec3) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def norm(vector: Vec3) -> float:
    return math.sqrt(dot(vector, vector))


def unit(vector: Vec3) -> Vec3:
    magnitude = norm(vector)
    if magnitude <= EPSILON:
        raise ValueError("zero vector has no direction")
    return scale(vector, 1.0 / magnitude)


def distance(first: Vec3, second: Vec3) -> float:
    return norm(subtract(first, second))


def forward_vector(pose: Pose3D) -> Vec3:
    yaw = math.radians(pose.yaw_deg)
    pitch = math.radians(pose.pitch_deg)
    return (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )


def right_vector(pose: Pose3D) -> Vec3:
    yaw = math.radians(pose.yaw_deg)
    return (-math.sin(yaw), math.cos(yaw), 0.0)


def up_vector(pose: Pose3D) -> Vec3:
    forward = forward_vector(pose)
    right = right_vector(pose)
    return (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )


def sensor_pose(
    body: Pose3D,
    translation_body_m: Iterable[float],
    *,
    sensor_pitch_deg: float | None = None,
) -> Pose3D:
    """Return a mounted inspection-camera pose.

    The mounting translation always follows the measured vehicle body. When an
    execution contract declares a bounded gimbal, only the camera optical pitch
    is replaced; the vehicle itself remains governed by its physical attitude.
    """
    tx, ty, tz = (float(value) for value in translation_body_m)
    forward = forward_vector(body)
    right = right_vector(body)
    up = up_vector(body)
    offset = add(add(scale(forward, tx), scale(right, ty)), scale(up, tz))
    return Pose3D(
        position=add(body.position, offset),
        yaw_deg=body.yaw_deg,
        pitch_deg=body.pitch_deg if sensor_pitch_deg is None else float(sensor_pitch_deg),
        roll_deg=body.roll_deg,
    )


@dataclass(frozen=True)
class AABB:
    collider_id: str
    minimum: Vec3
    maximum: Vec3
    semantic: str = "obstacle"

    def __post_init__(self) -> None:
        if not self.collider_id:
            raise ValueError("collider_id cannot be empty")
        if any(low > high for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("AABB minimum exceeds maximum")

    @classmethod
    def from_center_size(
        cls,
        collider_id: str,
        center: Iterable[float],
        size: Iterable[float],
        semantic: str = "obstacle",
    ) -> AABB:
        center_values = tuple(float(value) for value in center)
        size_values = tuple(float(value) for value in size)
        if len(center_values) != 3 or len(size_values) != 3 or any(v <= 0 for v in size_values):
            raise ValueError("center and positive size must be three-vectors")
        half = tuple(value / 2.0 for value in size_values)
        minimum = tuple(c - h for c, h in zip(center_values, half, strict=True))
        maximum = tuple(c + h for c, h in zip(center_values, half, strict=True))
        return cls(collider_id, minimum, maximum, semantic)  # type: ignore[arg-type]

    def expanded(self, margin: float) -> AABB:
        return AABB(
            self.collider_id,
            tuple(value - margin for value in self.minimum),  # type: ignore[arg-type]
            tuple(value + margin for value in self.maximum),  # type: ignore[arg-type]
            self.semantic,
        )

    def contains(self, point: Vec3, margin: float = 0.0) -> bool:
        return all(
            low - margin <= value <= high + margin
            for value, low, high in zip(point, self.minimum, self.maximum, strict=True)
        )

    def point_distance(self, point: Vec3) -> float:
        squared = 0.0
        for value, low, high in zip(point, self.minimum, self.maximum, strict=True):
            delta = low - value if value < low else value - high if value > high else 0.0
            squared += delta * delta
        return math.sqrt(squared)


def colliders_from_city(city: dict[str, Any], *, include_ground: bool = False) -> list[AABB]:
    result: list[AABB] = []
    for building in city.get("buildings", []):
        for component in building.get("components", []):
            result.append(
                AABB.from_center_size(
                    f"{building['id']}/{component['id']}",
                    component["center"],
                    component["size"],
                    "building",
                )
            )
    for obstacle in city.get("obstacles", []):
        result.append(
            AABB.from_center_size(
                str(obstacle["id"]),
                obstacle["center"],
                obstacle["size"],
                str(obstacle.get("kind", "obstacle")),
            )
        )
    if include_ground:
        size = float(city["size_m"])
        result.append(AABB.from_center_size("ground", (0.0, 0.0, -0.1), (size, size, 0.2)))
    return result


def segment_intersection_fraction(start: Vec3, end: Vec3, box: AABB) -> float | None:
    """Return the first segment/AABB intersection in [0, 1], using the slab test."""

    direction = subtract(end, start)
    t_min, t_max = 0.0, 1.0
    for origin, delta, low, high in zip(start, direction, box.minimum, box.maximum, strict=True):
        if abs(delta) <= EPSILON:
            if origin < low or origin > high:
                return None
            continue
        inverse = 1.0 / delta
        first = (low - origin) * inverse
        second = (high - origin) * inverse
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
        if t_min > t_max:
            return None
    return t_min


def segment_intersects_expanded_aabb(
    start: Vec3, end: Vec3, box: AABB, margin: float
) -> bool:
    """Conservatively test a segment against an AABB expanded by ``margin``.

    The expanded box contains the Euclidean margin around the original AABB.
    A negative result therefore proves that the exact segment/AABB clearance is
    greater than ``margin``.  A positive result is only a broad-phase candidate
    and must not be treated as an exact collision result.
    """

    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("expanded AABB margin must be finite and non-negative")
    direction = subtract(end, start)
    t_min, t_max = 0.0, 1.0
    for origin, delta, low, high in zip(
        start, direction, box.minimum, box.maximum, strict=True
    ):
        expanded_low = low - margin
        expanded_high = high + margin
        if abs(delta) <= EPSILON:
            if origin < expanded_low or origin > expanded_high:
                return False
            continue
        inverse = 1.0 / delta
        first = (expanded_low - origin) * inverse
        second = (expanded_high - origin) * inverse
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
        if t_min > t_max:
            return False
    return True


def segment_aabb_clearance(start: Vec3, end: Vec3, box: AABB) -> float:
    """Return the exact minimum center-line distance from a segment to an AABB."""

    direction = subtract(end, start)
    breakpoints = {0.0, 1.0}
    for origin, delta, low, high in zip(start, direction, box.minimum, box.maximum, strict=True):
        if abs(delta) <= EPSILON:
            continue
        for boundary in (low, high):
            crossing = (boundary - origin) / delta
            if 0.0 < crossing < 1.0:
                breakpoints.add(crossing)
    ordered = sorted(breakpoints)
    best_squared = math.inf
    for left, right in zip(ordered, ordered[1:], strict=False):
        midpoint = (left + right) / 2.0
        quadratic = 0.0
        linear = 0.0
        constant = 0.0
        for origin, delta, low, high in zip(
            start, direction, box.minimum, box.maximum, strict=True
        ):
            coordinate = origin + delta * midpoint
            if coordinate < low:
                intercept, slope = low - origin, -delta
            elif coordinate > high:
                intercept, slope = origin - high, delta
            else:
                continue
            quadratic += slope * slope
            linear += 2.0 * intercept * slope
            constant += intercept * intercept
        candidates = [left, right]
        if quadratic > EPSILON:
            stationary = -linear / (2.0 * quadratic)
            if left <= stationary <= right:
                candidates.append(stationary)
        for parameter in candidates:
            squared = quadratic * parameter * parameter + linear * parameter + constant
            best_squared = min(best_squared, max(0.0, squared))
    return math.sqrt(best_squared)


def minimum_segment_clearance(
    start: Vec3, end: Vec3, colliders: Iterable[AABB]
) -> tuple[float, str | None]:
    """Return the closest obstacle distance anywhere along a straight segment."""

    best_distance = math.inf
    best_id: str | None = None
    for collider in colliders:
        candidate = segment_aabb_clearance(start, end, collider)
        if candidate < best_distance:
            best_distance = candidate
            best_id = collider.collider_id
    return best_distance, best_id


def segment_segment_distance(
    first_start: Vec3,
    first_end: Vec3,
    second_start: Vec3,
    second_end: Vec3,
) -> float:
    """Return the minimum distance between two 3D line segments."""

    first = subtract(first_end, first_start)
    second = subtract(second_end, second_start)
    offset = subtract(first_start, second_start)
    first_norm = dot(first, first)
    second_norm = dot(second, second)
    cross = dot(first, second)
    first_offset = dot(first, offset)
    second_offset = dot(second, offset)
    denominator = first_norm * second_norm - cross * cross

    if first_norm <= EPSILON and second_norm <= EPSILON:
        return distance(first_start, second_start)
    if first_norm <= EPSILON:
        first_parameter = 0.0
        second_parameter = max(0.0, min(1.0, second_offset / second_norm))
    elif second_norm <= EPSILON:
        second_parameter = 0.0
        first_parameter = max(0.0, min(1.0, -first_offset / first_norm))
    else:
        if denominator > EPSILON:
            first_parameter = max(
                0.0,
                min(1.0, (cross * second_offset - first_offset * second_norm) / denominator),
            )
        else:
            first_parameter = 0.0
        second_parameter = (cross * first_parameter + second_offset) / second_norm
        if second_parameter < 0.0:
            second_parameter = 0.0
            first_parameter = max(0.0, min(1.0, -first_offset / first_norm))
        elif second_parameter > 1.0:
            second_parameter = 1.0
            first_parameter = max(0.0, min(1.0, (cross - first_offset) / first_norm))
    first_point = add(first_start, scale(first, first_parameter))
    second_point = add(second_start, scale(second, second_parameter))
    return distance(first_point, second_point)


def line_of_sight(
    start: Vec3,
    end: Vec3,
    colliders: Iterable[AABB],
    *,
    ignored_ids: frozenset[str] = frozenset(),
    endpoint_tolerance: float = 1.0e-4,
) -> tuple[bool, str | None]:
    for collider in colliders:
        if collider.collider_id in ignored_ids:
            continue
        fraction = segment_intersection_fraction(start, end, collider)
        if fraction is not None and endpoint_tolerance < fraction < 1.0 - endpoint_tolerance:
            return False, collider.collider_id
        if fraction is not None and collider.contains(start):
            return False, collider.collider_id
    return True, None


def in_field_of_view(
    pose: Pose3D,
    point: Vec3,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[bool, float, float]:
    relative = subtract(point, pose.position)
    forward = forward_vector(pose)
    right = right_vector(pose)
    up = up_vector(pose)
    forward_component = dot(relative, forward)
    if forward_component <= EPSILON:
        return False, 180.0, 180.0
    horizontal = math.degrees(math.atan2(dot(relative, right), forward_component))
    vertical = math.degrees(math.atan2(dot(relative, up), forward_component))
    visible = (
        abs(horizontal) <= horizontal_fov_deg / 2.0 and abs(vertical) <= vertical_fov_deg / 2.0
    )
    return visible, horizontal, vertical


def surface_facing(
    sensor_position: Vec3,
    target_position: Vec3,
    outward_normal: Vec3,
    minimum_cosine: float,
) -> tuple[bool, float]:
    target_to_sensor = unit(subtract(sensor_position, target_position))
    cosine = dot(unit(outward_normal), target_to_sensor)
    return cosine >= minimum_cosine, cosine


def minimum_clearance(point: Vec3, colliders: Iterable[AABB]) -> tuple[float, str | None]:
    best_distance = math.inf
    best_id: str | None = None
    for collider in colliders:
        candidate = collider.point_distance(point)
        if candidate < best_distance:
            best_distance = candidate
            best_id = collider.collider_id
    return best_distance, best_id


def pose_looking_at(position: Vec3, target: Vec3) -> Pose3D:
    delta = subtract(target, position)
    horizontal = math.hypot(delta[0], delta[1])
    return Pose3D(
        position=position,
        yaw_deg=math.degrees(math.atan2(delta[1], delta[0])),
        pitch_deg=math.degrees(math.atan2(delta[2], horizontal)),
    )


def review_camera_pose(
    points: Iterable[Iterable[float]],
    city: dict[str, Any],
) -> tuple[Vec3, Vec3]:
    """Choose a nearby review camera that is clear of authoritative colliders.

    The function is intentionally renderer-independent so camera placement can be
    regression-tested without starting Isaac Sim.
    """

    positions = [tuple(float(value) for value in point) for point in points]
    if not positions or any(len(position) != 3 for position in positions):
        raise ValueError("review camera requires at least one three-dimensional point")
    size = float(city["size_m"])
    if size <= 0.0:
        raise ValueError("city size must be positive")
    center: Vec3 = tuple(  # type: ignore[assignment]
        sum(position[axis] for position in positions) / len(positions) for axis in range(3)
    )
    look_at: Vec3 = (center[0], center[1], max(1.0, center[2]))
    colliders = colliders_from_city(city)
    maximum_height = float(city.get("metrics", {}).get("height_max_m", 0.0))
    horizontal_spread = max(
        (math.hypot(position[0] - center[0], position[1] - center[1]) for position in positions),
        default=0.0,
    )
    # Start markers are a compact four-UAV audit group.  Prefer a local camera so
    # each identity remains readable; fall back to city-scale rings only when
    # nearby candidates intersect authoritative colliders.
    camera_z = max(center[2] + 5.0, min(14.0, maximum_height * 0.25))
    radii = (
        max(8.0, horizontal_spread * 3.5),
        max(12.0, horizontal_spread * 5.0),
        max(18.0, size * 0.22),
        size * 0.55,
    )

    candidates: list[tuple[tuple[int, float, float, float], Vec3]] = []
    for radius in dict.fromkeys(radii):
        for index in range(16):
            angle = -math.pi / 2.0 + index * math.tau / 16.0
            candidate: Vec3 = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                camera_z,
            )
            point_clearance, _ = minimum_clearance(candidate, colliders)
            if point_clearance < 2.0:
                continue
            visible = sum(
                line_of_sight(candidate, position, colliders)[0] for position in positions
            )
            sight_clearance, _ = minimum_segment_clearance(candidate, look_at, colliders)
            inside_city = abs(candidate[0]) <= size / 2.0 and abs(candidate[1]) <= size / 2.0
            score = (
                visible,
                min(sight_clearance, 10.0),
                1.0 if inside_city else 0.0,
                -radius,
            )
            candidates.append((score, candidate))
    if not candidates:
        return (center[0], -size / 2.0 - 8.0, camera_z), look_at
    return max(candidates, key=lambda item: item[0])[1], look_at
