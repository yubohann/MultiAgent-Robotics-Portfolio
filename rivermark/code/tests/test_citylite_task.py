from __future__ import annotations

import pytest

from rivermark_benchmark.citylite_scene import (
    PUBLIC_ROUTE_FAMILIES_W_M,
    PUBLIC_ROUTES_W_M,
)
from rivermark_benchmark.citylite_task import (
    CAMERA_HEADING_MODEL_SEGMENT_YAW_LIMITED,
    TARGET_VISIBILITY_DIRECT,
    TargetSamplingError,
    measure_target_visibility,
    route_timing_requirements,
    sample_private_targets,
    target_visibility_execution_window,
    validate_route_timing_feasibility,
)
from rivermark_benchmark.collection_protocol import native_t2_v3_motion_contract


def _sample(*, seed: int, count: int = 2) -> tuple[dict[str, object], ...]:
    return sample_private_targets(
        seed=seed,
        target_count=count,
        target_region_id="citylite-target-region-a-v1",
        visibility_bucket="direct-visible-v1",
        routes_w_m=PUBLIC_ROUTES_W_M,
        structural_aabbs=(),
    )


def test_private_sampler_is_digest_ordered_and_reproducible() -> None:
    first = _sample(seed=7)
    second = _sample(seed=7)
    assert first == second
    assert all("seed" not in target and "candidate_pool" not in target for target in first)
    assert all(target["visibility_bucket"] == "direct-visible-v1" for target in first)
    assert all(target["visibility_evidence"]["visible_witness_count"] >= 1 for target in first)  # type: ignore[index]


def test_private_sampler_changes_selection_without_changing_hard_contract() -> None:
    first = _sample(seed=7, count=1)
    second = _sample(seed=8, count=1)
    assert first[0]["position_w_m"] != second[0]["position_w_m"]
    assert first[0]["visibility_bucket"] == second[0]["visibility_bucket"] == "direct-visible-v1"


def test_private_sampler_does_not_downgrade_an_unrealizable_visibility_bucket() -> None:
    with pytest.raises(TargetSamplingError, match="without relaxing gates"):
        sample_private_targets(
            seed=7,
            target_count=1,
            target_region_id="citylite-target-region-a-v1",
            visibility_bucket="partial-visible-v1",
            routes_w_m=PUBLIC_ROUTES_W_M,
            structural_aabbs=(),
        )


def test_target_visibility_rejects_witnesses_only_reached_after_the_capture_window() -> None:
    # The normal 12.6 s capture reaches only the beginning of the third
    # 6-second segment. The target is visible only once the fourth segment is
    # reached by a longer otherwise-identical rollout.
    route = (((0.0, 0.0, 10.0), (20.0, 0.0, 10.0), (40.0, 0.0, 10.0), (60.0, 0.0, 10.0), (80.0, 0.0, 10.0)),)
    target = (80.0, 0.0, 4.7)
    retained = measure_target_visibility(
        target, routes_w_m=route, structural_aabbs=()
    )
    full_route = measure_target_visibility(
        target,
        routes_w_m=route,
        structural_aabbs=(),
        execution_window=target_visibility_execution_window(rollout_steps=4800),
    )
    assert retained.visibility_bucket is None
    assert full_route.visibility_bucket == TARGET_VISIBILITY_DIRECT


def test_target_visibility_requires_a_nontrivial_projected_instance_area() -> None:
    route = (((0.0, 0.0, 10.0), (20.0, 0.0, 10.0)),)
    target = (20.0, 0.0, 4.7)
    visible = measure_target_visibility(
        target, routes_w_m=route, structural_aabbs=(), radius_m=0.30
    )
    undersized = measure_target_visibility(
        target, routes_w_m=route, structural_aabbs=(), radius_m=0.01
    )
    assert visible.visibility_bucket == TARGET_VISIBILITY_DIRECT
    assert undersized.visibility_bucket is None
    assert undersized.undersized_witness_count > 0


def test_target_visibility_rejects_a_nominal_only_witness_under_tracking_error() -> None:
    # This is a public synthetic route/target fixture.  The centreline camera
    # can see the target, but a bounded physical tracking displacement pushes
    # at least one probe outside the pinhole frustum.  Sampling must reject
    # such a target before a native Isaac rollout can produce an all-zero
    # semantic visibility stream.
    route = (((0.0, 0.0, 10.0), (20.0, 0.0, 10.0)),)
    target = (12.0, -3.0, 7.0)
    nominal = measure_target_visibility(
        target,
        routes_w_m=route,
        structural_aabbs=(),
        tracking_envelope_m=0.0,
    )
    robust = measure_target_visibility(
        target,
        routes_w_m=route,
        structural_aabbs=(),
        tracking_envelope_m=1.5,
    )
    assert nominal.visibility_bucket == TARGET_VISIBILITY_DIRECT
    assert robust.visibility_bucket is None
    assert robust.tracking_envelope_m == 1.5


def test_target_visibility_execution_window_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        measure_target_visibility(
            (20.0, 0.0, 4.7),
            routes_w_m=(((0.0, 0.0, 10.0), (20.0, 0.0, 10.0)),),
            structural_aabbs=(),
            execution_window={},
        )


def test_t2_segment_heading_keeps_turn_visibility_separate_from_t1_initial_heading() -> None:
    # The second segment faces +Y.  Its target is deliberately outside the
    # first +X camera view, so the historical T1 heading model cannot claim a
    # witness while the yaw-aware T2 schedule can after its public settle time.
    route = (((0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (10.0, 20.0, 10.0)),)
    target = (10.0, 16.0, 5.6)
    window = target_visibility_execution_window(rollout_steps=4800)
    initial = measure_target_visibility(
        target, routes_w_m=route, structural_aabbs=(), tracking_envelope_m=0.0, execution_window=window
    )
    segment = measure_target_visibility(
        target,
        routes_w_m=route,
        structural_aabbs=(),
        tracking_envelope_m=0.0,
        execution_window=window,
        camera_heading_model=CAMERA_HEADING_MODEL_SEGMENT_YAW_LIMITED,
        max_yaw_rate_rad_s=0.8,
        yaw_feedback_gain=1.2,
        yaw_stability_error_rad=0.2,
        yaw_settle_margin_s=0.4,
    )
    assert initial.visibility_bucket is None
    assert segment.visibility_bucket == TARGET_VISIBILITY_DIRECT


def test_route_timing_feasibility_rejects_the_legacy_t2_action_envelope() -> None:
    routes = (((0.0, 0.0, 10.0), (10.0, 0.0, 11.9)),)
    requirements = route_timing_requirements(routes, waypoint_segment_seconds=6.0)
    assert requirements["maximum_required_horizontal_speed_mps"] == pytest.approx(10.0 / 6.0)
    assert requirements["maximum_required_vertical_speed_mps"] == pytest.approx(1.9 / 6.0)
    with pytest.raises(ValueError, match="horizontal speed"):
        validate_route_timing_feasibility(
            routes,
            waypoint_segment_seconds=6.0,
            max_horizontal_speed_mps=0.75,
            max_vertical_speed_mps=0.10,
            utilization_limit=1.0,
        )
    feasible = validate_route_timing_feasibility(
        routes,
        waypoint_segment_seconds=6.0,
        max_horizontal_speed_mps=2.0,
        max_vertical_speed_mps=0.4,
        utilization_limit=0.9,
    )
    assert feasible["horizontal_speed_budget_mps"] == pytest.approx(1.8)


def test_v3_time_scaled_contract_is_feasible_for_its_public_route() -> None:
    motion = native_t2_v3_motion_contract()
    feasible = validate_route_timing_feasibility(
        PUBLIC_ROUTE_FAMILIES_W_M["citylite-route-family-a-v1"],
        waypoint_segment_seconds=float(motion["waypoint_segment_seconds"]),
        max_horizontal_speed_mps=float(motion["max_horizontal_speed_mps"]),
        max_vertical_speed_mps=float(motion["max_vertical_speed_mps"]),
        utilization_limit=float(motion["route_speed_utilization_limit"]),
    )
    assert feasible["maximum_required_vertical_speed_mps"] == pytest.approx(0.32025)
    assert feasible["vertical_speed_budget_mps"] == pytest.approx(0.36)
