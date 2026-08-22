from __future__ import annotations

import math
from pathlib import Path

import pytest

from aerocity_bench.quadrotor_guidance import (
    VelocityGuidanceLimits,
    VerticalBoundaryGuard,
    YawAlignmentGuard,
    anisotropic_route_time_lower_bound_s,
    position_anchored_velocity_guidance,
    three_leg_sky_route_waypoint_yaw,
    yaw_aligned_translation_goal,
)
from tools.quadrotor_l1_vertical_slice import _validated_output_paths


def test_guidance_anchors_position_and_enforces_axis_speed_caps() -> None:
    limits = VelocityGuidanceLimits(horizontal_speed_mps=3.0, vertical_speed_mps=1.0)
    anchor, velocity, yaw = position_anchored_velocity_guidance(
        (1.0, -2.0, 3.0), (101.0, -2.0, 53.0), 0.7, limits=limits
    )
    assert anchor == (1.0, -2.0, 3.0)
    assert math.hypot(velocity[0], velocity[1]) == pytest.approx(3.0)
    assert velocity[2] == pytest.approx(1.0)
    assert yaw == pytest.approx(0.7)


def test_guidance_decelerates_continuously_at_the_public_waypoint() -> None:
    limits = VelocityGuidanceLimits(horizontal_speed_mps=3.0, vertical_speed_mps=1.0)
    _, velocity, _ = position_anchored_velocity_guidance(
        (0.0, 0.0, 0.0), (0.2, 0.0, -0.4), 0.0, limits=limits
    )
    assert velocity == pytest.approx((0.1, 0.0, -0.2))
    _, stationary, _ = position_anchored_velocity_guidance(
        (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 0.0, limits=limits
    )
    assert stationary == (0.0, 0.0, 0.0)


def test_yaw_alignment_guard_rotates_before_translating() -> None:
    guard = YawAlignmentGuard(
        activation_yaw_error_rad=math.radians(90.0),
        release_yaw_error_rad=math.radians(5.0),
        release_yaw_rate_rad_s=math.radians(10.0),
    )
    current = (7.5693, -28.5963, 3.3864)
    goal = (9.5, -28.596, 3.4092)
    held_goal, held = yaw_aligned_translation_goal(
        current,
        goal,
        math.radians(-158.1),
        0.0,
        math.radians(44.7),
        guard=guard,
    )
    assert held is True
    assert held_goal == current

    latched_goal, held = yaw_aligned_translation_goal(
        current,
        goal,
        math.radians(-45.0),
        0.0,
        math.radians(3.0),
        alignment_active=held,
        guard=guard,
    )
    assert held is True
    assert latched_goal == current

    translating_goal, held = yaw_aligned_translation_goal(
        current,
        goal,
        math.radians(-2.0),
        0.0,
        math.radians(3.0),
        alignment_active=held,
        guard=guard,
    )
    assert held is False
    assert translating_goal == goal


def test_yaw_alignment_guard_does_not_block_an_in_place_turn() -> None:
    guard = YawAlignmentGuard(
        activation_yaw_error_rad=math.radians(90.0),
        release_yaw_error_rad=math.radians(5.0),
        release_yaw_rate_rad_s=math.radians(10.0),
    )
    goal, held = yaw_aligned_translation_goal(
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
        math.radians(-180.0),
        0.0,
        0.0,
        guard=guard,
    )
    assert held is False
    assert goal == (1.0, 2.0, 3.0)


def test_yaw_alignment_guard_does_not_latch_ordinary_route_turns() -> None:
    guard = YawAlignmentGuard(
        activation_yaw_error_rad=math.radians(90.0),
        release_yaw_error_rad=math.radians(5.0),
        release_yaw_rate_rad_s=math.radians(10.0),
    )
    goal, held = yaw_aligned_translation_goal(
        (0.0, 0.0, 4.0),
        (10.0, 0.0, 4.0),
        math.radians(-45.0),
        0.0,
        math.radians(40.0),
        guard=guard,
    )
    assert held is False
    assert goal == (10.0, 0.0, 4.0)


def test_guidance_rejects_invalid_limits_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="vertical_speed_mps"):
        VelocityGuidanceLimits(horizontal_speed_mps=3.0, vertical_speed_mps=0.0).validate()
    with pytest.raises(ValueError, match="finite"):
        position_anchored_velocity_guidance(
            (0.0, 0.0, 0.0),
            (math.inf, 0.0, 0.0),
            0.0,
            limits=VelocityGuidanceLimits(horizontal_speed_mps=3.0, vertical_speed_mps=1.0),
        )


def test_guidance_brakes_measured_descent_before_the_lower_safe_boundary() -> None:
    limits = VelocityGuidanceLimits(horizontal_speed_mps=1.5, vertical_speed_mps=2.0)
    guard = VerticalBoundaryGuard(
        minimum_safe_altitude_m=1.32,
        maximum_safe_altitude_m=69.68,
        guaranteed_braking_deceleration_mps2=0.25,
        response_horizon_s=0.2,
        reserve_distance_m=0.5,
    )
    _, velocity, _ = position_anchored_velocity_guidance(
        (0.0, 0.0, 10.0),
        (0.0, 0.0, 2.5),
        0.0,
        limits=limits,
        current_linear_velocity_w_mps=(0.0, 0.0, -3.0),
        vertical_guard=guard,
    )
    assert velocity[2] == 0.0
    _, safe_velocity, _ = position_anchored_velocity_guidance(
        (0.0, 0.0, 30.0),
        (0.0, 0.0, 2.5),
        0.0,
        limits=limits,
        current_linear_velocity_w_mps=(0.0, 0.0, -1.0),
        vertical_guard=guard,
    )
    assert safe_velocity[2] == pytest.approx(-2.0)
    _, low_approach_velocity, _ = position_anchored_velocity_guidance(
        (0.0, 0.0, 2.5),
        (0.0, 0.0, 2.0),
        0.0,
        limits=limits,
        current_linear_velocity_w_mps=(0.0, 0.0, 0.0),
        vertical_guard=guard,
    )
    assert -0.6 < low_approach_velocity[2] < 0.0
    with pytest.raises(ValueError, match="requires measured linear velocity"):
        position_anchored_velocity_guidance(
            (0.0, 0.0, 10.0),
            (0.0, 0.0, 2.5),
            0.0,
            limits=limits,
            vertical_guard=guard,
        )


def test_anisotropic_route_lower_bound_charges_axes_separately() -> None:
    limits = VelocityGuidanceLimits(horizontal_speed_mps=3.0, vertical_speed_mps=1.0)
    duration = anisotropic_route_time_lower_bound_s(
        ((0.0, 0.0, 0.0), (6.0, 0.0, 2.0), (6.0, 3.0, 0.0)), limits=limits
    )
    assert duration == pytest.approx(7.0)
    with pytest.raises(ValueError, match="at least two"):
        anisotropic_route_time_lower_bound_s(((0.0, 0.0, 0.0),), limits=limits)


def test_three_leg_sky_route_freezes_transit_heading_through_return_descent() -> None:
    route = ((-17.35, -32.48, 45.9), (1.31, -38.88, 45.9), (1.31, -38.88, 2.5))
    expected_heading = math.atan2(-38.88 + 32.48, 1.31 + 17.35)

    assert three_leg_sky_route_waypoint_yaw(route, 0) == pytest.approx(expected_heading)
    assert three_leg_sky_route_waypoint_yaw(route, 1) == pytest.approx(expected_heading)
    assert three_leg_sky_route_waypoint_yaw(route, 2) == pytest.approx(expected_heading)


def test_three_leg_sky_route_uses_terminal_yaw_only_when_observation_requires_it() -> None:
    route = ((0.0, 0.0, 12.0), (8.0, 4.0, 12.0), (8.0, 4.0, 5.0))
    assert three_leg_sky_route_waypoint_yaw(route, 0, terminal_yaw_rad=0.4) == pytest.approx(
        math.atan2(4.0, 8.0)
    )
    assert three_leg_sky_route_waypoint_yaw(route, 2, terminal_yaw_rad=0.4) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="waypoint_index"):
        three_leg_sky_route_waypoint_yaw(route, 3)
    with pytest.raises(ValueError, match="non-zero horizontal"):
        three_leg_sky_route_waypoint_yaw(((0.0, 0.0, 3.0), (0.0, 0.0, 7.0), (0.0, 0.0, 2.0)), 0)


def test_vertical_slice_evidence_outputs_are_fresh_distinct_json_files(tmp_path: Path) -> None:
    public, private = _validated_output_paths(tmp_path / "slice.public.json", None)
    assert public == tmp_path / "slice.public.json"
    assert private == tmp_path / "slice.public.private.json"

    with pytest.raises(ValueError, match="must be a .json"):
        _validated_output_paths(tmp_path / "slice", None)
    with pytest.raises(ValueError, match="must differ"):
        _validated_output_paths(tmp_path / "slice.public.json", tmp_path / "slice.public.json")

    public.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exist"):
        _validated_output_paths(public, None)
