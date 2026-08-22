from __future__ import annotations

import concurrent.futures
import math
import threading

import numpy as np
import pytest
import torch

from aerocity_method.evaluation.hm3d_safety import (
    ClearanceAssessment,
    ConservativeVoxelClearance,
)
from aerocity_method.runtime.hm3d_cf2x_execution import (
    CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
    CF2X_MAX_REFERENCE_ACCELERATION_MPS2,
    CF2X_MAX_REFERENCE_SPEED_MPS,
    FLIGHT_CLEARANCE_M,
    PLANNED_CONTINUOUS_CLEARANCE_M,
    REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
    REQUIRED_TERMINAL_CLEARANCE_M,
    _clear_static_collision_los,
    _controller_tracking_profile,
    _desired_rotation_from_force_and_yaw,
    _EvaluatorStaticClearance,
    _first_static_scene_hit,
    _line_guard,
    _line_profile_state,
    _minimum_observation_dwell_completed,
    _minimum_time_line_reference,
    _minimum_time_line_reference_with_boundary_speeds,
    _observation_failure_reason,
    _observation_source_identity,
    _raycast_guard_diagnostic,
    _routed_guard,
    _route_corner_speed_mps,
    _scheduled_observation_completed,
    _so3_attitude_error,
    _sparse_range_sampling_phase,
    _waypoint_reached,
)


class _Clearance:
    def admits_many(self, points, required_clearance_m):
        del required_clearance_m
        return tuple(True for _ in points)

    def exact_static_distances_m(self, points):
        return tuple(1.25 for _ in points)


def test_exact_clearance_serializes_shared_native_rtree_queries() -> None:
    """The Windows ``rtree`` backend faults if one mesh is queried concurrently."""

    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=3)
    oracle = _EvaluatorStaticClearance(
        field=ConservativeVoxelClearance(
            collision_distance_m=np.ones((1, 1, 1), dtype=np.float64),
            origin_center_m=(0.0, 0.0, 0.0),
            resolution_m=0.25,
        ),
        collision_mesh=mesh,
    )
    points = tuple(
        (float(index) * 0.01, 0.3, 1.1) for index in range(96)
    )
    ready = threading.Barrier(4)

    def query() -> tuple[float, ...]:
        ready.wait(timeout=5.0)
        return oracle.exact_static_distances_m(points)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = tuple(executor.submit(query) for _ in range(4))
        results = tuple(future.result(timeout=30.0) for future in futures)

    assert all(result == pytest.approx(results[0]) for result in results[1:])
    assert oracle.exact_batch_call_count == 1
    assert len(oracle._exact_cache_m) == len(points)


class _DeepRejectedClearanceField:
    def __init__(self, sampled_distance_m: float) -> None:
        self.sampled_distance_m = float(sampled_distance_m)
        self.discretization_margin_m = math.sqrt(3.0) * 0.25

    def assess(self, point):
        del point
        sampled = self.sampled_distance_m
        return ClearanceAssessment(
            in_field_bounds=True,
            sampled_distance_m=sampled,
            conservative_distance_m=max(0.0, sampled - self.discretization_margin_m),
            discretization_margin_m=self.discretization_margin_m,
        )


def test_clearance_skips_exact_when_esdf_upper_bound_cannot_meet_requirement() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=3)
    oracle = _EvaluatorStaticClearance(
        field=_DeepRejectedClearanceField(0.0),
        collision_mesh=mesh,
    )

    admitted = oracle.admits_many(((0.0, 0.0, 0.0),), PLANNED_CONTINUOUS_CLEARANCE_M)

    assert admitted == (False,)
    assert oracle.exact_fallback_count == 0
    assert oracle.esdf_upper_bound_skip_count == 1
    assert oracle.exact_rejection_count == 1
    assert oracle._exact_cache_m == {}


def test_clearance_still_uses_exact_when_esdf_upper_bound_could_meet_requirement() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=3)
    oracle = _EvaluatorStaticClearance(
        field=_DeepRejectedClearanceField(0.2),
        collision_mesh=mesh,
    )

    admitted = oracle.admits_many(((0.0, 0.0, 0.0),), PLANNED_CONTINUOUS_CLEARANCE_M)

    assert oracle.exact_fallback_count == 1
    assert oracle.esdf_upper_bound_skip_count == 0
    assert isinstance(admitted[0], bool)


def test_batched_required_clearances_use_one_exact_fallback_path() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=3)
    oracle = _EvaluatorStaticClearance(
        field=_DeepRejectedClearanceField(0.2),
        collision_mesh=mesh,
    )

    admitted = oracle.admits_many_with_required_clearances(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        (PLANNED_CONTINUOUS_CLEARANCE_M, PLANNED_CONTINUOUS_CLEARANCE_M),
    )

    assert len(admitted) == 2
    assert oracle.exact_fallback_count == 2
    assert oracle.esdf_upper_bound_skip_count == 0


def test_cell_index_exact_distances_match_reference_within_query_margin() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    oracle = _EvaluatorStaticClearance(
        field=ConservativeVoxelClearance(
            collision_distance_m=np.ones((1, 1, 1), dtype=np.float64),
            origin_center_m=(0.0, 0.0, 0.0),
            resolution_m=0.25,
        ),
        collision_mesh=mesh,
    )
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.1, 0.1, 1.1],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    custom = oracle._exact_distances_with_cell_index(points)
    reference = trimesh.proximity.closest_point(mesh, points)[1]

    for point, custom_distance, reference_distance in zip(
        points, custom, reference, strict=True
    ):
        if reference_distance <= oracle._LOCAL_MESH_MARGIN_M:
            assert custom_distance == pytest.approx(reference_distance, abs=1.0e-6)
        else:
            assert custom_distance == pytest.approx(oracle._LOCAL_MESH_MARGIN_M)
    assert oracle._face_cell_index_build_wall_s >= 0.0
    assert oracle._face_cell_index_query_count == len(points)


def test_observation_uses_absolute_decision_boundary_after_early_arrival() -> None:
    assert not _scheduled_observation_completed(
        timestamp_s=6.0,
        planned_end_s=8.0,
        actual_start_s=4.0,
        minimum_dwell_s=1.0,
        final_physics_timestamp_s=8.0,
    )
    assert _scheduled_observation_completed(
        timestamp_s=8.0,
        planned_end_s=8.0,
        actual_start_s=4.0,
        minimum_dwell_s=1.0,
        final_physics_timestamp_s=8.0,
    )


def test_observation_does_not_claim_unsimulated_substep_or_short_dwell() -> None:
    assert _scheduled_observation_completed(
        timestamp_s=7.991666666666666,
        planned_end_s=8.0,
        actual_start_s=6.0,
        minimum_dwell_s=1.0,
        final_physics_timestamp_s=7.991666666666666,
    )


def test_event_driven_observation_finishes_after_its_actual_minimum_dwell() -> None:
    assert not _minimum_observation_dwell_completed(
        timestamp_s=4.99,
        actual_start_s=4.0,
        minimum_dwell_s=1.0,
    )
    assert _minimum_observation_dwell_completed(
        timestamp_s=5.0,
        actual_start_s=4.0,
        minimum_dwell_s=1.0,
    )
    assert not _scheduled_observation_completed(
        timestamp_s=7.991666666666666,
        planned_end_s=8.0,
        actual_start_s=7.5,
        minimum_dwell_s=1.0,
        final_physics_timestamp_s=7.991666666666666,
    )


@pytest.mark.parametrize(
    ("transit_completed", "observation_completed", "failed", "reservation_waiting", "expected"),
    (
        (False, False, False, False, "transit"),
        (True, False, False, False, "dwell"),
        (False, False, False, True, None),
        (False, False, True, False, None),
        (True, True, False, False, None),
    ),
)
def test_sparse_range_sampling_phase_only_accepts_real_transit_or_dwell(
    transit_completed: bool,
    observation_completed: bool,
    failed: bool,
    reservation_waiting: bool,
    expected: str | None,
) -> None:
    assert _sparse_range_sampling_phase(
        transit_completed=transit_completed,
        observation_completed=observation_completed,
        failed=failed,
        reservation_waiting=reservation_waiting,
    ) == expected


def test_waypoint_requires_a_settle_before_the_next_rest_to_rest_segment() -> None:
    assert not _waypoint_reached(
        error_m=0.05,
        speed_mps=0.8,
        requires_settle=True,
        arrival_tolerance_m=0.1,
    )
    assert not _waypoint_reached(
        error_m=0.05,
        speed_mps=0.2,
        requires_settle=True,
        arrival_tolerance_m=0.1,
    )
    assert not _waypoint_reached(
        error_m=0.02,
        speed_mps=0.06,
        requires_settle=True,
        arrival_tolerance_m=0.1,
    )
    assert _waypoint_reached(
        error_m=0.02,
        speed_mps=0.04,
        requires_settle=True,
        arrival_tolerance_m=0.1,
    )
    assert _waypoint_reached(
        error_m=0.05,
        speed_mps=0.8,
        requires_settle=False,
        arrival_tolerance_m=0.1,
    )


def test_short_line_reference_accelerates_then_stops_at_the_endpoint() -> None:
    start = (0.0, 0.0, 0.0)
    end = (0.4, 0.0, 0.0)
    beginning = _minimum_time_line_reference(start, end, 0.0)
    midpoint = _minimum_time_line_reference(start, end, beginning.duration_s / 2.0)
    finished = _minimum_time_line_reference(start, end, beginning.duration_s)

    assert beginning.acceleration_mps2 == pytest.approx((0.8, 0.0, 0.0))
    assert midpoint.position_m == pytest.approx((0.2, 0.0, 0.0))
    assert midpoint.velocity_mps[0] == pytest.approx((0.4 * 0.8) ** 0.5)
    assert finished.position_m == pytest.approx(end)
    assert finished.velocity_mps == pytest.approx((0.0, 0.0, 0.0))
    assert finished.acceleration_mps2 == pytest.approx((0.0, 0.0, 0.0))


def test_long_line_reference_contains_a_speed_limited_cruise_phase() -> None:
    reference = _minimum_time_line_reference((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 1.5)

    assert reference.duration_s == pytest.approx(3.25)
    assert reference.position_m == pytest.approx((0.875, 0.0, 0.0))
    assert reference.velocity_mps == pytest.approx((1.0, 0.0, 0.0))
    assert reference.acceleration_mps2 == pytest.approx((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite and non-negative"):
        _minimum_time_line_reference((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.1)


def test_route_corner_speed_is_bounded_by_the_shortest_segment() -> None:
    assert _route_corner_speed_mps(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    ) == pytest.approx(0.35)
    short_route_speed = _route_corner_speed_mps(
        ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0))
    )
    assert 0.0 < short_route_speed < 0.35


def test_route_corner_speed_accepts_explicit_hold_and_rejects_mixed_zero_segment() -> None:
    position = (1.0, 2.0, 1.2)

    assert _route_corner_speed_mps((position, position)) == 0.0
    with pytest.raises(ValueError, match="zero-length"):
        _route_corner_speed_mps(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        )


def test_pass_through_line_reference_keeps_intermediate_speed() -> None:
    start = (0.0, 0.0, 0.0)
    end = (1.0, 0.0, 0.0)
    intermediate = _minimum_time_line_reference_with_boundary_speeds(
        start,
        end,
        0.5,
        initial_speed_mps=0.35,
        terminal_speed_mps=0.35,
    )
    finished = _minimum_time_line_reference_with_boundary_speeds(
        start,
        end,
        intermediate.duration_s,
        initial_speed_mps=0.35,
        terminal_speed_mps=0.35,
    )
    assert intermediate.velocity_mps[0] == pytest.approx(0.35)
    assert finished.position_m == pytest.approx(end)
    assert finished.velocity_mps[0] == pytest.approx(0.35)

    stopped = _minimum_time_line_reference_with_boundary_speeds(
        start,
        end,
        10.0,
        initial_speed_mps=0.35,
        terminal_speed_mps=0.0,
    )
    assert stopped.position_m == pytest.approx(end)
    assert stopped.velocity_mps == pytest.approx((0.0, 0.0, 0.0))


def test_controller_profile_records_racer_scale_speed_and_acceleration() -> None:
    profile = _controller_tracking_profile()
    assert profile["speed_profile"] == "time-parameterized-trapezoid-so3-guarded-v8"
    assert profile["intermediate_waypoint_requires_settle"] is False
    assert profile["waypoint_pass_through_speed_mps"] == pytest.approx(0.35)
    assert profile["waypoint_settle_position_tolerance_m"] == pytest.approx(0.03)
    assert profile["attitude_control"] == "force-rate-limited-yaw-so3-v2"
    assert profile["maximum_yaw_rate_deg_s"] == 10.0
    assert profile["maximum_reference_speed_mps"] == pytest.approx(1.0)
    assert profile["maximum_reference_acceleration_mps2"] == pytest.approx(0.8)
    assert profile["maximum_feedback_acceleration_mps2"] == pytest.approx(3.0)
    assert profile["position_error_gain_per_s2"] == pytest.approx(4.0)
    assert profile["velocity_error_gain_per_s"] == pytest.approx(7.0)
    assert CF2X_MAX_REFERENCE_SPEED_MPS == pytest.approx(1.0)
    assert CF2X_MAX_REFERENCE_ACCELERATION_MPS2 == pytest.approx(0.8)
    assert CF2X_MAX_FEEDBACK_ACCELERATION_MPS2 == pytest.approx(3.0)


def test_force_aligned_attitude_keeps_world_thrust_direction_across_yaw() -> None:
    force = torch.tensor(((1.0, 0.0, 9.81), (1.0, 0.0, 9.81)))
    yaw = torch.tensor((0.0, math.pi / 2.0))

    desired_rotation = _desired_rotation_from_force_and_yaw(force, yaw)

    assert desired_rotation[0, :, 2].tolist() == pytest.approx(desired_rotation[1, :, 2].tolist())
    identity_error = _so3_attitude_error(desired_rotation, desired_rotation)
    torch.testing.assert_close(identity_error, torch.zeros_like(identity_error))


def test_observation_without_a_real_frame_keeps_identity_absent_and_records_failure() -> None:
    assert _observation_source_identity(
        None,
        episode_id="episode",
        agent_id="uav0",
    ) == (None, None, None)
    assert (
        _observation_failure_reason(
            completed=False,
            collided=True,
            out_of_bounds=False,
            source_id=None,
        )
        == "observation_collision"
    )
    assert (
        _observation_failure_reason(
            completed=True,
            collided=False,
            out_of_bounds=False,
            source_id=None,
        )
        == "observation_no_valid_range_frame"
    )


def test_observation_with_a_real_frame_preserves_complete_source_identity() -> None:
    assert _observation_source_identity(
        "range-frame-0",
        episode_id="episode",
        agent_id="uav0",
    ) == ("range-frame-0", "episode", "uav0")
    assert (
        _observation_failure_reason(
            completed=True,
            collided=False,
            out_of_bounds=False,
            source_id="range-frame-0",
        )
        == ""
    )


@pytest.mark.parametrize(
    ("agent_id", "collider", "expected"),
    (
        ("uav0", "/World/P07Agents/Env_0/Robot/body/collisions", "self_cf2x"),
        ("uav0", "/World/P07Agents/Env_2/Robot/body/collisions", "other_cf2x"),
        (
            "uav0",
            "/World/envs/env_1/P07Agents/Agent_0/Robot/body/collisions",
            "other_cf2x",
        ),
        ("uav0", "/World/HM3DCollision/mesh", "static_hm3d"),
        ("uav0", "/World/envs/env_1/HM3DCollision/mesh", "static_hm3d"),
        ("uav0", "/World/Unexpected/collider", "unknown_scene_prim"),
    ),
)
def test_raycast_guard_diagnostic_classifies_hit_prim(
    agent_id: str, collider: str, expected: str
) -> None:
    row = _raycast_guard_diagnostic(
        agent_id=agent_id,
        start=(0.0, 0.0, 0.0),
        end=(1.0, 0.0, 0.0),
        requested_distance_m=1.0,
        raycast_distance_m=0.95,
        hit={
            "hit": True,
            "distance": 0.03,
            "position": (0.03, 0.0, 0.0),
            "rigidBody": collider.rsplit("/", 1)[0],
            "collider": collider,
        },
    )

    assert row["hit_class"] == expected
    assert row["hit_prim_path"] == collider
    assert row["hit_distance_m"] == pytest.approx(0.03)
    assert row["hit_position_m"] == (0.03, 0.0, 0.0)
    assert "exact_static_start_clearance_m" not in row


def test_static_query_skips_cf2x_and_returns_later_hm3d_hit() -> None:
    class _Query:
        def __init__(self):
            self.calls = 0

        def raycast_closest(self, origin, direction, distance):
            del origin, direction, distance
            self.calls += 1
            if self.calls == 1:
                return {
                    "hit": True,
                    "distance": 0.07,
                    "position": (0.07, 0.0, 0.0),
                    "rigidBody": "/World/envs/env_1/P07Agents/Agent_1/Robot/body",
                    "collider": ("/World/envs/env_1/P07Agents/Agent_1/Robot/body/collisions"),
                }
            return {
                "hit": True,
                "distance": 0.4,
                "position": (0.48, 0.0, 0.0),
                "rigidBody": "/World/envs/env_1/HM3DCollision/geometry",
                "collider": "/World/envs/env_1/HM3DCollision/geometry/mesh",
            }

    query = _Query()
    hit = _first_static_scene_hit(query, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert hit is not None
    assert hit["distance"] == pytest.approx(0.48)
    assert hit["ignored_dynamic_hit_count"] == 1
    assert query.calls == 2


def test_static_los_ignores_only_cf2x_colliders() -> None:
    class _DynamicOnlyQuery:
        def __init__(self):
            self.calls = 0

        def raycast_closest(self, origin, direction, distance):
            del origin, direction, distance
            self.calls += 1
            if self.calls == 1:
                return {
                    "hit": True,
                    "distance": 0.07,
                    "rigidBody": "/World/P07Agents/Env_0/Robot/body",
                    "collider": "/World/P07Agents/Env_0/Robot/body/collisions",
                }
            return {"hit": False}

    assert _clear_static_collision_los(_DynamicOnlyQuery(), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_line_guard_emits_static_evidence_after_skipping_self_cf2x() -> None:
    class _Query:
        def __init__(self):
            self.calls = 0

        def raycast_closest(self, origin, direction, distance):
            del origin, direction, distance
            self.calls += 1
            if self.calls == 1:
                return {
                    "hit": True,
                    "distance": 0.07,
                    "position": (0.07, 0.0, 0.0),
                    "rigidBody": "/World/P07Agents/Env_1/Robot/body",
                    "collider": "/World/P07Agents/Env_1/Robot/body/collisions",
                }
            return {
                "hit": True,
                "distance": 0.4,
                "position": (0.48, 0.0, 0.0),
                "rigidBody": "/World/HM3DCollision/geometry",
                "collider": "/World/HM3DCollision/geometry/mesh",
            }

    events = []
    guarded = _line_guard(
        _Query(),
        _Clearance(),
        "uav1",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        events.append,
    )

    assert guarded.legal is False
    assert guarded.reason == "segment_blocked"
    assert len(events) == 1
    assert events[0]["hit_class"] == "static_hm3d"
    assert events[0]["ignored_dynamic_hit_count"] == 1
    assert events[0]["segment_start_m"] == (0.0, 0.0, 0.0)
    assert events[0]["segment_end_m"] == (1.0, 0.0, 0.0)


def test_line_guard_accepts_route_when_only_hit_is_self_cf2x() -> None:
    class _Query:
        def __init__(self):
            self.calls = 0

        def raycast_closest(self, origin, direction, distance):
            del origin, direction, distance
            self.calls += 1
            if self.calls == 1:
                return {
                    "hit": True,
                    "distance": 0.07,
                    "rigidBody": "/World/P07Agents/Env_1/Robot/body",
                    "collider": "/World/P07Agents/Env_1/Robot/body/collisions",
                }
            return {"hit": False}

    guarded = _line_guard(
        _Query(),
        _Clearance(),
        "uav1",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )

    assert guarded.legal is True


def test_short_horizon_guard_can_reject_without_private_reroute_search() -> None:
    class _StaticHitQuery:
        def __init__(self):
            self.calls = 0

        def raycast_closest(self, origin, direction, distance):
            del origin, direction, distance
            self.calls += 1
            return {
                "hit": True,
                "distance": 0.2,
                "position": (0.2, 0.0, 0.0),
                "rigidBody": "/World/HM3DCollision/geometry",
                "collider": "/World/HM3DCollision/geometry/mesh",
            }

    query = _StaticHitQuery()
    guarded = _routed_guard(
        query,
        _Clearance(),
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        "uav0",
        ((0.0, 0.0, 0.0), (0.4, 0.0, 0.0)),
        allow_public_reroute=False,
    )

    assert guarded.legal is False
    assert guarded.reason == "segment_blocked"
    assert query.calls == 1


def test_exact_endpoints_do_not_receive_the_interior_sampling_reserve() -> None:
    class _NoHitQuery:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    class _ThresholdClearance:
        def __init__(self):
            self.requirements = []

        def admits_many(self, points, required_clearance_m):
            self.requirements.append((tuple(points), required_clearance_m))
            return tuple(True for _ in points)

    clearance = _ThresholdClearance()
    guarded = _line_guard(
        _NoHitQuery(),
        clearance,
        "uav0",
        ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
    )

    assert guarded.legal is True
    assert REQUIRED_TERMINAL_CLEARANCE_M == pytest.approx(PLANNED_CONTINUOUS_CLEARANCE_M)
    assert REQUIRED_ROUTE_SAMPLE_CLEARANCE_M > REQUIRED_TERMINAL_CLEARANCE_M
    assert clearance.requirements[0][1] == pytest.approx(FLIGHT_CLEARANCE_M)
    assert clearance.requirements[1][1] == pytest.approx(REQUIRED_TERMINAL_CLEARANCE_M)
    assert clearance.requirements[2][1] == pytest.approx(REQUIRED_ROUTE_SAMPLE_CLEARANCE_M)


def test_stationary_hold_uses_exact_endpoint_clearance_without_a_raycast() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            raise AssertionError("a stationary hold must not issue a zero-length raycast")

    guarded = _line_guard(
        _Query(),
        _Clearance(),
        "uav0",
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )

    assert guarded.legal is True
    assert guarded.reason == "stationary_hold"


def test_stationary_hold_accepts_measured_pose_inside_planning_reserve() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            raise AssertionError("a stationary hold must not issue a zero-length raycast")

    class _MeasuredClearance:
        def __init__(self, distance_m: float):
            self.distance_m = distance_m
            self.requirements: list[float] = []

        def admits_many(self, points, required_clearance_m):
            self.requirements.append(required_clearance_m)
            return tuple(self.distance_m >= required_clearance_m for _ in points)

    safe = _MeasuredClearance(0.35)
    unsafe = _MeasuredClearance(0.29)
    safe_hold = _line_guard(_Query(), safe, "uav0", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    unsafe_hold = _line_guard(_Query(), unsafe, "uav0", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    assert safe_hold.legal is True
    assert safe.requirements == [pytest.approx(FLIGHT_CLEARANCE_M)]
    assert unsafe_hold.legal is False
    assert unsafe.requirements == [pytest.approx(FLIGHT_CLEARANCE_M)]


def test_new_endpoint_inside_planning_reserve_is_still_rejected() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    start = (0.0, 0.0, 0.0)
    end = (0.30, 0.0, 0.0)

    class _PointClearance:
        def admits_many(self, points, required_clearance_m):
            distances = {start: 0.35, end: 0.35}
            return tuple(
                distances.get(tuple(point), 1.0) >= required_clearance_m for point in points
            )

    guarded = _line_guard(_Query(), _PointClearance(), "uav0", (start, end))

    assert guarded.legal is False
    assert guarded.reason == "insufficient_continuous_collision_clearance"


@pytest.mark.parametrize(
    ("distances", "path", "expected_stage", "expected_point"),
    (
        (
            (
                {(0.0, 0.0, 0.0): 0.29},
                ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
                "start",
                (0.0, 0.0, 0.0),
            ),
            (
                {(0.0, 0.0, 0.0): 0.60, (0.25, 0.0, 0.0): 0.35},
                ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
                "endpoint",
                (0.25, 0.0, 0.0),
            ),
            (
                {(0.0, 0.0, 0.0): 0.60, (0.50, 0.0, 0.0): 0.35, (1.0, 0.0, 0.0): 0.60},
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                "interior",
                (0.50, 0.0, 0.0),
            ),
        )
    ),
)
def test_clearance_guard_diagnostic_identifies_rejected_route_stage(
    distances, path, expected_stage, expected_point
) -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    class _DistanceClearance:
        def admits_many(self, points, required_clearance_m):
            return tuple(
                distances.get(tuple(point), 0.60) >= required_clearance_m for point in points
            )

        def exact_static_distances_m(self, points):
            return tuple(distances.get(tuple(point), 0.60) for point in points)

    events: list[dict[str, object]] = []
    guarded = _line_guard(
        _Query(),
        _DistanceClearance(),
        "uav3",
        path,
        events.append,
    )

    assert guarded.legal is False
    assert guarded.reason == "insufficient_continuous_collision_clearance"
    assert len(events) == 1
    assert events[0]["event_type"] == "static_clearance_rejection"
    assert events[0]["stage"] == expected_stage
    assert events[0]["minimum_clearance_position_m"] == expected_point
    assert events[0]["minimum_static_mesh_clearance_m"] == pytest.approx(distances[expected_point])


def test_routed_guard_checks_each_supplied_public_polyline_segment() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    start = (0.0, 0.0, 0.0)
    unsafe_waypoint = (0.5, 0.0, 0.0)
    end = (1.0, 0.0, 0.0)

    class _WaypointClearance:
        def admits_many(self, points, required_clearance_m):
            return tuple(
                tuple(point) != unsafe_waypoint or required_clearance_m <= FLIGHT_CLEARANCE_M
                for point in points
            )

    guarded = _routed_guard(
        _Query(),
        _WaypointClearance(),
        (),
        "uav0",
        (start, unsafe_waypoint, end),
        allow_public_reroute=False,
    )

    assert guarded.legal is False
    assert guarded.path_m == (start, unsafe_waypoint, end)
    assert guarded.reason == "insufficient_continuous_collision_clearance"


def test_routed_guard_shortened_polyline_keeps_the_original_first_point() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    start = (0.0, 0.0, 0.0)
    intermediate = (1.0, 1.0, 0.0)
    end = (2.0, 2.0, 0.0)

    guarded = _routed_guard(
        _Query(),
        _Clearance(),
        (),
        "uav0",
        (start, intermediate, end),
        allow_public_reroute=False,
    )

    # The shortcut collapses the three-point polyline onto the legal direct
    # leg, but the admitted command must still begin at the vehicle pose.
    # Regressing to the final leg's GuardedPath would start the command at
    # ``intermediate`` and send the vehicle across an unguarded gap.
    assert guarded.legal is True
    assert guarded.path_m[0] == start
    assert guarded.path_m[-1] == end
    assert all(math.dist(a, b) <= 1.0e-9 for a, b in ((guarded.path_m[0], start), (guarded.path_m[-1], end)))


def test_line_profile_pass_through_to_rest_terminates_at_arrival() -> None:
    """A corner pass-through leg must end with zero reference acceleration.

    The pass-through boundary-speed profile (initial speed above zero) used to
    keep returning the braking acceleration forever after arrival, which
    pushed the vehicle away from the terminal waypoint and made it impossible
    to satisfy the settle contract.  After ``duration_s`` the reference must
    be stationary with zero acceleration.
    """

    _, _, _, duration_s = _line_profile_state(
        distance_m=1.43,
        elapsed_s=1.0,
        initial_speed_mps=0.35,
        terminal_speed_mps=0.0,
        cruise_speed_mps=CF2X_MAX_REFERENCE_SPEED_MPS,
        max_accel_mps2=CF2X_MAX_REFERENCE_ACCELERATION_MPS2,
    )
    braking_travelled_m, braking_speed_mps, braking_acceleration_mps2, _ = (
        _line_profile_state(
            distance_m=1.43,
            elapsed_s=duration_s - 0.1,
            initial_speed_mps=0.35,
            terminal_speed_mps=0.0,
            cruise_speed_mps=CF2X_MAX_REFERENCE_SPEED_MPS,
            max_accel_mps2=CF2X_MAX_REFERENCE_ACCELERATION_MPS2,
        )
    )
    assert braking_acceleration_mps2 < 0.0
    assert braking_speed_mps < 0.35
    travelled_m, speed_mps, acceleration_mps2, ended_duration_s = _line_profile_state(
        distance_m=1.43,
        elapsed_s=duration_s + 5.0,
        initial_speed_mps=0.35,
        terminal_speed_mps=0.0,
        cruise_speed_mps=CF2X_MAX_REFERENCE_SPEED_MPS,
        max_accel_mps2=CF2X_MAX_REFERENCE_ACCELERATION_MPS2,
    )
    assert ended_duration_s == duration_s
    assert abs(travelled_m - 1.43) < 1.0e-9
    assert abs(speed_mps) < 1.0e-9
    assert abs(acceleration_mps2) < 1.0e-9


def test_measured_start_outside_tightened_margin_can_hold_or_move_inward() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    bounds_min = (-13.25, -20.75, -2.75)
    bounds_max = (21.0, 1.5, 3.25)
    measured_start = (-2.878859758, -8.654204369, -2.352415085)

    hold = _routed_guard(
        _Query(),
        _Clearance(),
        (),
        "uav0",
        (measured_start, measured_start),
        bounds_min,
        bounds_max,
        allow_public_reroute=False,
    )
    inward = _routed_guard(
        _Query(),
        _Clearance(),
        (),
        "uav0",
        (measured_start, (-2.5, -8.7, -2.3)),
        bounds_min,
        bounds_max,
        allow_public_reroute=False,
    )

    assert hold.legal is True
    assert hold.reason == "stationary_hold"
    assert inward.legal is True


def test_new_command_endpoint_still_requires_control_boundary_margin() -> None:
    class _Query:
        @staticmethod
        def raycast_closest(origin, direction, distance):
            del origin, direction, distance
            return {"hit": False}

    guarded = _routed_guard(
        _Query(),
        _Clearance(),
        (),
        "uav0",
        ((0.0, 0.0, 0.0), (0.0, 0.0, -2.5)),
        (-13.25, -20.75, -2.75),
        (21.0, 1.5, 3.25),
        allow_public_reroute=False,
    )

    assert guarded.legal is False
    assert guarded.reason == "insufficient_control_boundary_margin"


def test_unexecuted_observation_timeout_uses_a_zero_width_execution_window() -> None:
    from aerocity_method.adapters.hm3d_execution import FragmentExecutionSample

    cutoff_s = 1.5
    sample = FragmentExecutionSample(
        planned_fragment_hash="a" * 64,
        executed=False,
        actual_start_s=cutoff_s,
        actual_end_s=cutoff_s,
        execution_trace_hash="b" * 64,
        failure_reason="observation_not_reached",
    )

    assert sample.actual_start_s == cutoff_s
    assert sample.actual_end_s == cutoff_s
