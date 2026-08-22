from __future__ import annotations

import numpy as np
import pytest

from aerocity_method.evaluation.hm3d_safety import (
    ConservativeVoxelClearance,
    TimedPolyline,
    TimedStationary,
    assess_collision_avoidance_recovery,
    assess_route_tube_separation,
    assess_synchronized_separation,
    required_segment_sample_clearance_m,
)


def test_conservative_voxel_field_rejects_a_diagonal_nearby_obstacle() -> None:
    distance = np.full((5, 5, 5), 5.0, dtype=np.float64)
    distance[2, 2, 2] = 0.0
    field = ConservativeVoxelClearance(distance, (0.0, 0.0, 0.0), 1.0)
    assessment = field.assess((1.49, 1.49, 1.49))
    assert assessment.in_field_bounds
    assert assessment.admits(0.3) is False
    assert assessment.discretization_margin_m == pytest.approx(3.0**0.5)


def test_continuous_time_aligned_crossing_routes_are_rejected() -> None:
    assessment = assess_synchronized_separation(
        (
            TimedPolyline("uav0", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)), 0.0, 2.0),
            TimedPolyline("uav1", ((1.0, -1.0, 1.0), (1.0, 1.0, 1.0)), 0.0, 2.0),
        ),
        minimum_separation_m=0.5,
    )
    assert assessment.admitted is False
    assert assessment.minimum_separation_m == pytest.approx(0.0)
    assert assessment.minimum_time_s == pytest.approx(1.0)


def test_same_geometry_at_nonoverlapping_times_remains_admissible() -> None:
    routes = (
        TimedPolyline("uav0", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)), 0.0, 1.0),
        TimedPolyline("uav1", ((1.0, -1.0, 1.0), (1.0, 1.0, 1.0)), 2.0, 3.0),
    )
    synchronized = assess_synchronized_separation(routes, minimum_separation_m=0.5)
    route_tube = assess_route_tube_separation(routes, minimum_separation_m=0.5)

    assert synchronized.admitted is True
    assert synchronized.minimum_separation_m == pytest.approx(1.0)
    assert route_tube.admitted is False
    assert route_tube.minimum_route_separation_m == pytest.approx(0.0)
    assert route_tube.closest_agent_pair == ("uav0", "uav1")


def test_route_tube_allows_parallel_paths_outside_physical_collision_distance() -> None:
    assessment = assess_route_tube_separation(
        (
            TimedPolyline("uav0", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)), 0.0, 2.0),
            TimedPolyline("uav1", ((0.0, 0.70, 1.0), (2.0, 0.70, 1.0)), 4.0, 6.0),
        ),
        minimum_separation_m=0.5,
    )

    assert assessment.admitted is True
    assert assessment.minimum_route_separation_m == pytest.approx(0.70)


def test_normal_endpoint_requires_a_robust_margin_beyond_the_tracking_envelope() -> None:
    endpoints = (
        TimedStationary("uav0", (0.0, 0.0, 1.0), 0.0, 1.0),
        TimedStationary("uav1", (0.901, 0.0, 1.0), 0.0, 1.0),
    )

    assert assess_synchronized_separation(
        endpoints, minimum_separation_m=0.90
    ).admitted
    assert not assess_synchronized_separation(
        endpoints, minimum_separation_m=0.95
    ).admitted


def test_route_tube_rejects_the_failed_00626_shared_waypoint() -> None:
    shared_waypoint = (-6.625, -3.125, 1.375)
    assessment = assess_route_tube_separation(
        (
            TimedPolyline(
                "uav0",
                (
                    (-6.62663, -1.85269, 1.37446),
                    shared_waypoint,
                    (-6.625, -3.125, 1.625),
                ),
                0.0,
                4.0,
            ),
            TimedPolyline(
                "uav1",
                (
                    (-6.75, -3.00023, 1.27096),
                    shared_waypoint,
                    (-6.625, -4.875, 1.375),
                ),
                0.0,
                6.0,
            ),
        ),
        minimum_separation_m=0.5,
    )

    assert assessment.admitted is False
    assert assessment.minimum_route_separation_m == pytest.approx(0.0, abs=1.0e-12)
    assert assessment.closest_agent_pair == ("uav0", "uav1")
    assert shared_waypoint in assessment.closest_points_m


def test_moving_route_is_checked_against_a_stationary_relay() -> None:
    assessment = assess_synchronized_separation(
        (
            TimedPolyline("uav0", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)), 0.0, 2.0),
            TimedStationary("uav1", (1.0, 0.2, 1.0), 0.0, 2.0),
        ),
        minimum_separation_m=0.5,
    )
    assert assessment.admitted is False
    assert assessment.minimum_separation_m == pytest.approx(0.2)


def _recovery_assessment(*, route_end_x: float, initial_x: float = 0.70, speed: float = 0.01):
    return assess_collision_avoidance_recovery(
        (
            TimedPolyline("uav0", ((initial_x, 0.0, 1.0), (route_end_x, 0.0, 1.0)), 0.0, 1.0),
            TimedStationary("uav1", (0.0, 0.0, 1.0), 0.0, 1.0),
        ),
        recovery_agent_id="uav0",
        boundary_linear_speeds_mps={"uav0": speed, "uav1": speed},
        physical_minimum_separation_m=0.50,
        planned_minimum_separation_m=0.90,
        recovery_endpoint_minimum_separation_m=0.95,
        boundary_speed_limit_mps=0.05,
    )


def test_envelope_recovery_allows_a_near_rest_diverging_route() -> None:
    assessment = _recovery_assessment(route_end_x=1.00)

    assert assessment.admitted is True
    assert assessment.initial_minimum_separation_m == pytest.approx(0.70)
    assert assessment.endpoint_minimum_separation_m == pytest.approx(1.00)
    assert assessment.synchronized_physical_assessment.admitted is True


def test_envelope_recovery_accepts_the_exact_endpoint_margin() -> None:
    assessment = _recovery_assessment(route_end_x=0.95)

    assert assessment.admitted is True
    assert assessment.endpoint_minimum_separation_m == pytest.approx(0.95)


def test_envelope_recovery_rejects_a_nonstationary_boundary() -> None:
    assessment = _recovery_assessment(route_end_x=1.00, speed=0.051)

    assert assessment.admitted is False
    assert "boundary_not_near_stationary" in assessment.rejection_reasons


def test_envelope_recovery_rejects_multiple_moving_vehicles() -> None:
    assessment = assess_collision_avoidance_recovery(
        (
            TimedPolyline("uav0", ((0.70, 0.0, 1.0), (1.00, 0.0, 1.0)), 0.0, 1.0),
            TimedPolyline("uav1", ((0.0, 0.0, 1.0), (-0.10, 0.0, 1.0)), 0.0, 1.0),
        ),
        recovery_agent_id="uav0",
        boundary_linear_speeds_mps={"uav0": 0.01, "uav1": 0.01},
        physical_minimum_separation_m=0.50,
        planned_minimum_separation_m=0.90,
        recovery_endpoint_minimum_separation_m=0.95,
        boundary_speed_limit_mps=0.05,
    )

    assert assessment.admitted is False
    assert "recovery_route_converges_or_multiple_agents_move" in assessment.rejection_reasons


def test_envelope_recovery_rejects_a_converging_route() -> None:
    assessment = _recovery_assessment(route_end_x=0.60)

    assert assessment.admitted is False
    assert "recovery_route_converges_or_multiple_agents_move" in assessment.rejection_reasons


def test_envelope_recovery_rejects_a_physical_separation_violation() -> None:
    assessment = _recovery_assessment(route_end_x=1.00, initial_x=0.49)

    assert assessment.admitted is False
    assert "initial_physical_separation_violation" in assessment.rejection_reasons


def test_envelope_recovery_rejects_an_endpoint_that_does_not_restore_margin() -> None:
    assessment = _recovery_assessment(route_end_x=0.92)

    assert assessment.admitted is False
    assert "recovery_endpoint_does_not_restore_planning_margin" in assessment.rejection_reasons
    assert assessment.endpoint_minimum_separation_m == pytest.approx(0.92)


def test_segment_sampling_threshold_preserves_continuous_clearance() -> None:
    assert required_segment_sample_clearance_m(0.45, 0.15) == pytest.approx(0.525)
    assert required_segment_sample_clearance_m(0.30, 0.15) == pytest.approx(0.375)


@pytest.mark.parametrize("required, spacing", [(float("nan"), 0.1), (0.1, 0.0)])
def test_segment_sampling_threshold_rejects_invalid_inputs(required: float, spacing: float) -> None:
    with pytest.raises(ValueError):
        required_segment_sample_clearance_m(required, spacing)
