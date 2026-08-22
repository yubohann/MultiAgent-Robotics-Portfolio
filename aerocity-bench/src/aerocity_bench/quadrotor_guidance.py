"""Bounded, target-free guidance for internal CF2X preflight runs.

This module deliberately sits below benchmark methods.  A method supplies a
public waypoint; this helper converts it into a conservative velocity reference
for the shared low-level controller.  It never receives evaluator-private
targets, witnesses, or score state.

The previous vertical-slice fixture combined a moving look-ahead position
target with a high velocity reference.  With the candidate controller this
caused the position and velocity terms to demand acceleration in the same
direction, which made a long vertical leg oscillate around its waypoint.  The
guidance below anchors the position term at the measured position during
transit.  The controller therefore acts as a damped velocity servo; the
position-to-speed law reduces the requested velocity continuously near the
public waypoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class VelocityGuidanceLimits:
    """Public, anisotropic transit limits for a candidate preflight controller."""

    horizontal_speed_mps: float
    vertical_speed_mps: float
    horizontal_position_to_speed_gain_s: float = 0.5
    vertical_position_to_speed_gain_s: float = 0.5

    def validate(self) -> None:
        for name, value in (
            ("horizontal_speed_mps", self.horizontal_speed_mps),
            ("vertical_speed_mps", self.vertical_speed_mps),
            (
                "horizontal_position_to_speed_gain_s",
                self.horizontal_position_to_speed_gain_s,
            ),
            ("vertical_position_to_speed_gain_s", self.vertical_position_to_speed_gain_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def to_dict(self) -> dict[str, float | str]:
        self.validate()
        return {
            "control_law": "position_anchored_anisotropic_velocity_servo",
            "horizontal_speed_mps": self.horizontal_speed_mps,
            "vertical_speed_mps": self.vertical_speed_mps,
            "horizontal_position_to_speed_gain_s": self.horizontal_position_to_speed_gain_s,
            "vertical_position_to_speed_gain_s": self.vertical_position_to_speed_gain_s,
        }


@dataclass(frozen=True)
class VerticalBoundaryGuard:
    """Candidate-only lower/upper flight-bound braking envelope.

    This guard receives measured flight state and public flight bounds only. It
    never reads target truth, CitySpec internals, or a baseline identifier.
    """

    minimum_safe_altitude_m: float
    maximum_safe_altitude_m: float
    guaranteed_braking_deceleration_mps2: float
    response_horizon_s: float
    reserve_distance_m: float

    def validate(self) -> None:
        values = (
            ("minimum_safe_altitude_m", self.minimum_safe_altitude_m),
            ("maximum_safe_altitude_m", self.maximum_safe_altitude_m),
            ("guaranteed_braking_deceleration_mps2", self.guaranteed_braking_deceleration_mps2),
            ("response_horizon_s", self.response_horizon_s),
            ("reserve_distance_m", self.reserve_distance_m),
        )
        for name, value in values:
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.maximum_safe_altitude_m <= self.minimum_safe_altitude_m:
            raise ValueError("vertical safety bounds must be ordered")
        if self.guaranteed_braking_deceleration_mps2 <= 0.0:
            raise ValueError("guaranteed_braking_deceleration_mps2 must be positive")
        if self.response_horizon_s < 0.0 or self.reserve_distance_m < 0.0:
            raise ValueError("vertical guard horizon and reserve must be non-negative")

    def to_dict(self) -> dict[str, float | str]:
        self.validate()
        return {
            "control_law": "public_boundary_aware_vertical_brake",
            "minimum_safe_altitude_m": self.minimum_safe_altitude_m,
            "maximum_safe_altitude_m": self.maximum_safe_altitude_m,
            "guaranteed_braking_deceleration_mps2": self.guaranteed_braking_deceleration_mps2,
            "response_horizon_s": self.response_horizon_s,
            "reserve_distance_m": self.reserve_distance_m,
        }


@dataclass(frozen=True)
class YawAlignmentGuard:
    """Latch a rotate-before-translate phase for heading reversals."""

    activation_yaw_error_rad: float
    release_yaw_error_rad: float
    release_yaw_rate_rad_s: float

    def validate(self) -> None:
        if (
            not math.isfinite(self.activation_yaw_error_rad)
            or not 0.0 < self.activation_yaw_error_rad < math.pi
        ):
            raise ValueError("activation_yaw_error_rad must be finite and between zero and pi")
        if (
            not math.isfinite(self.release_yaw_error_rad)
            or not 0.0 < self.release_yaw_error_rad < self.activation_yaw_error_rad
        ):
            raise ValueError("release_yaw_error_rad must be below the activation threshold")
        if not math.isfinite(self.release_yaw_rate_rad_s) or self.release_yaw_rate_rad_s <= 0.0:
            raise ValueError("release_yaw_rate_rad_s must be finite and positive")

    def to_dict(self) -> dict[str, float | str]:
        self.validate()
        return {
            "control_law": "latched_heading_reversal_then_translate",
            "activation_yaw_error_rad": self.activation_yaw_error_rad,
            "release_yaw_error_rad": self.release_yaw_error_rad,
            "release_yaw_rate_rad_s": self.release_yaw_rate_rad_s,
        }


def yaw_aligned_translation_goal(
    current_position_w_m: Vec3,
    goal_position_w_m: Vec3,
    current_yaw_rad: float,
    goal_yaw_rad: float,
    current_yaw_rate_rad_s: float,
    *,
    alignment_active: bool = False,
    guard: YawAlignmentGuard,
) -> tuple[Vec3, bool]:
    """Return a translation goal only after the measured yaw motion has settled.

    A Crazyflie-class quadrotor is underactuated.  A simultaneous near-180-degree
    yaw step and horizontal acceleration can consume attitude authority that the
    translational loop assumes is available.  The guard uses only public measured
    state and the public waypoint, and it preserves the requested yaw target.
    """

    guard.validate()
    values = (
        *current_position_w_m,
        *goal_position_w_m,
        current_yaw_rad,
        goal_yaw_rad,
        current_yaw_rate_rad_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("yaw-alignment guard inputs must be finite")
    displacement = math.sqrt(
        sum(
            (float(goal) - float(current)) ** 2
            for current, goal in zip(current_position_w_m, goal_position_w_m, strict=True)
        )
    )
    if displacement <= 1.0e-9:
        return tuple(float(value) for value in goal_position_w_m), False  # type: ignore[return-value]
    yaw_error = abs(
        (float(goal_yaw_rad) - float(current_yaw_rad) + math.pi) % (2.0 * math.pi)
        - math.pi
    )
    alignment_active = bool(alignment_active) or yaw_error > guard.activation_yaw_error_rad
    if alignment_active and (
        yaw_error > guard.release_yaw_error_rad
        or abs(float(current_yaw_rate_rad_s)) > guard.release_yaw_rate_rad_s
    ):
        return tuple(float(value) for value in current_position_w_m), True  # type: ignore[return-value]
    return tuple(float(value) for value in goal_position_w_m), False  # type: ignore[return-value]


def boundary_aware_vertical_velocity(
    desired_velocity_mps: float,
    current_altitude_m: float,
    current_vertical_velocity_mps: float,
    *,
    guard: VerticalBoundaryGuard,
) -> tuple[float, bool]:
    """Keep enough measured stopping distance before a public flight boundary.

    This changes a velocity reference passed to the existing rotor controller;
    it is never a root-state clamp or a post-hoc score correction.
    """

    guard.validate()
    values = (desired_velocity_mps, current_altitude_m, current_vertical_velocity_mps)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("vertical guard inputs must be finite")

    def stopping_distance(directed_speed: float) -> float:
        return (
            directed_speed * guard.response_horizon_s
            + directed_speed * directed_speed / (2.0 * guard.guaranteed_braking_deceleration_mps2)
            + guard.reserve_distance_m
        )

    def maximum_stoppable_speed(available_distance: float) -> float:
        # Solve v * response_horizon + v^2 / (2a) + reserve <= distance.
        usable_distance = max(0.0, available_distance - guard.reserve_distance_m)
        acceleration = guard.guaranteed_braking_deceleration_mps2
        return max(
            0.0,
            acceleration
            * (
                math.sqrt(guard.response_horizon_s * guard.response_horizon_s
                          + 2.0 * usable_distance / acceleration)
                - guard.response_horizon_s
            ),
        )

    descending_speed = max(0.0, -float(current_vertical_velocity_mps))
    ascending_speed = max(0.0, float(current_vertical_velocity_mps))
    lower_distance = float(current_altitude_m) - guard.minimum_safe_altitude_m
    upper_distance = guard.maximum_safe_altitude_m - float(current_altitude_m)
    lower_speed_limit = maximum_stoppable_speed(lower_distance)
    upper_speed_limit = maximum_stoppable_speed(upper_distance)
    if lower_distance <= stopping_distance(descending_speed):
        # Measured descent is already too fast for the conservative envelope.
        # Remove a downward reference so the existing velocity servo brakes.
        return max(0.0, float(desired_velocity_mps)), True
    if upper_distance <= stopping_distance(ascending_speed):
        return min(0.0, float(desired_velocity_mps)), True
    if desired_velocity_mps < 0.0 and -desired_velocity_mps > lower_speed_limit:
        return -lower_speed_limit, True
    if desired_velocity_mps > 0.0 and desired_velocity_mps > upper_speed_limit:
        return upper_speed_limit, True
    return float(desired_velocity_mps), False


def position_anchored_velocity_guidance(
    current_position_w_m: Vec3,
    goal_position_w_m: Vec3,
    goal_yaw_rad: float,
    *,
    limits: VelocityGuidanceLimits,
    current_linear_velocity_w_mps: Vec3 | None = None,
    vertical_guard: VerticalBoundaryGuard | None = None,
) -> tuple[Vec3, Vec3, float]:
    """Return a current-position anchor and a bounded public velocity reference.

    Axis-wise limits are intentional: a scalar three-dimensional cap changes
    the public timing semantics whenever a route has both horizontal and
    vertical motion.  The returned position equals the current measured
    position during transit, so the low-level position term cannot add an
    unbounded look-ahead acceleration on top of the requested velocity.
    """

    limits.validate()
    values = (*current_position_w_m, *goal_position_w_m, goal_yaw_rad)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("guidance positions and yaw must be finite")

    dx = float(goal_position_w_m[0]) - float(current_position_w_m[0])
    dy = float(goal_position_w_m[1]) - float(current_position_w_m[1])
    dz = float(goal_position_w_m[2]) - float(current_position_w_m[2])
    horizontal_distance = math.hypot(dx, dy)
    if horizontal_distance <= 1.0e-12:
        horizontal_velocity = (0.0, 0.0)
    else:
        speed = min(
            limits.horizontal_speed_mps,
            limits.horizontal_position_to_speed_gain_s * horizontal_distance,
        )
        horizontal_velocity = (speed * dx / horizontal_distance, speed * dy / horizontal_distance)
    vertical_velocity = max(
        -limits.vertical_speed_mps,
        min(
            limits.vertical_speed_mps,
            limits.vertical_position_to_speed_gain_s * dz,
        ),
    )
    if vertical_guard is not None:
        if current_linear_velocity_w_mps is None:
            raise ValueError("vertical_guard requires measured linear velocity")
        if len(current_linear_velocity_w_mps) != 3 or not all(
            math.isfinite(float(value)) for value in current_linear_velocity_w_mps
        ):
            raise ValueError("current_linear_velocity_w_mps must be a finite three-vector")
        vertical_velocity, _ = boundary_aware_vertical_velocity(
            vertical_velocity,
            float(current_position_w_m[2]),
            float(current_linear_velocity_w_mps[2]),
            guard=vertical_guard,
        )
    anchor = tuple(float(value) for value in current_position_w_m)
    return (
        anchor,
        (horizontal_velocity[0], horizontal_velocity[1], vertical_velocity),
        float(goal_yaw_rad),
    )


def anisotropic_route_time_lower_bound_s(
    route_positions_w_m: tuple[Vec3, ...], *, limits: VelocityGuidanceLimits
) -> float:
    """Return a kinematic lower bound for an ordered public route.

    Each leg is charged separately for horizontal and vertical motion.  This is
    intentionally a lower bound, not an execution prediction: it excludes
    acceleration, attitude settling, observation dwell, and obstacle avoidance.
    It prevents a preflight invocation from pretending that its requested
    simulation budget can complete a route even at its own declared speed caps.
    """

    limits.validate()
    if len(route_positions_w_m) < 2:
        raise ValueError("route_positions_w_m must contain at least two positions")
    total = 0.0
    starts = route_positions_w_m[:-1]
    ends = route_positions_w_m[1:]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if not all(math.isfinite(float(value)) for value in (*start, *end)):
            raise ValueError(f"route position {index} is not finite")
        total += math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1])) / (
            limits.horizontal_speed_mps
        )
        total += abs(float(end[2]) - float(start[2])) / limits.vertical_speed_mps
    return total


def three_leg_sky_route_waypoint_yaw(
    route_positions_w_m: tuple[Vec3, Vec3, Vec3],
    waypoint_index: int,
    *,
    terminal_yaw_rad: float | None = None,
) -> float:
    """Return a fixed yaw target for a vertical-horizontal-vertical route.

    The first two waypoints span the route's one horizontal transit leg.  The
    ascent and descent consequently keep that transit heading instead of
    recomputing it from a small, noisy residual horizontal error.  An outbound
    terminal witness may explicitly require a final observation yaw; return
    descents omit it and therefore retain the stable transit heading.
    """

    if waypoint_index not in {0, 1, 2}:
        raise ValueError("waypoint_index must select one of the three route waypoints")
    flattened = tuple(value for position in route_positions_w_m for value in position)
    if not all(math.isfinite(float(value)) for value in flattened):
        raise ValueError("route positions must be finite")
    if terminal_yaw_rad is not None and not math.isfinite(float(terminal_yaw_rad)):
        raise ValueError("terminal_yaw_rad must be finite when provided")

    first, second, _terminal = route_positions_w_m
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    if math.hypot(dx, dy) <= 1.0e-12:
        if terminal_yaw_rad is None:
            raise ValueError("route must include a non-zero horizontal transit leg")
        transit_yaw_rad = float(terminal_yaw_rad)
    else:
        transit_yaw_rad = math.atan2(dy, dx)

    if waypoint_index == 2 and terminal_yaw_rad is not None:
        return float(terminal_yaw_rad)
    return transit_yaw_rad
