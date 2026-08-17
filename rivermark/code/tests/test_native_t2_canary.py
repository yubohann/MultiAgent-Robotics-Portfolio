from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.citylite_scene import AGENT_COUNT
from rivermark_benchmark.native_t2_canary import (
    PublicRouteCoveragePolicy,
    SpatialCandidateDeduplicator,
    native_rgbd_world_points,
    native_semantic_rgbd_candidates,
)
from rivermark_benchmark.t2_policy_abi import (
    T2CandidateDetection,
    T2PolicyAbiError,
    T2PublicFleetObservation,
)


def _observation(*, time_ns: int = 0, x_positions: np.ndarray | None = None) -> T2PublicFleetObservation:
    if x_positions is None:
        x_positions = np.arange(AGENT_COUNT, dtype=np.float64)
    return T2PublicFleetObservation.from_rigid_body_state(
        physics_step=0,
        command_time_ns=time_ns,
        position_w_m=np.column_stack((x_positions, np.zeros(AGENT_COUNT), np.ones(AGENT_COUNT))),
        linear_velocity_w_mps=np.zeros((AGENT_COUNT, 3)),
        quaternion_wxyz=np.tile(np.array((1.0, 0.0, 0.0, 0.0)), (AGENT_COUNT, 1)),
        angular_velocity_b_radps=np.zeros((AGENT_COUNT, 3)),
    )


def test_public_route_policy_uses_only_static_routes_and_public_state() -> None:
    routes = np.zeros((AGENT_COUNT, 3, 3), dtype=np.float64)
    routes[:, :, 0] = np.array((0.0, 2.0, 4.0))
    routes[:, :, 2] = 1.0
    policy = PublicRouteCoveragePolicy(routes, waypoint_segment_seconds=2.0)
    action = policy(_observation(time_ns=1_000_000_000, x_positions=np.zeros(AGENT_COUNT)))
    assert action.shape == (AGENT_COUNT, 4)
    assert np.all(np.isfinite(action))
    assert np.all(action[:, 0] > 0.0)
    provenance = policy.provenance()
    assert provenance["private_evaluator_inputs"] is False
    assert len(provenance["route_sha256"]) == 64


def test_public_route_policy_corrects_toward_the_public_waypoint() -> None:
    routes = np.zeros((AGENT_COUNT, 3, 3), dtype=np.float64)
    routes[:, :, 0] = np.array((0.0, 2.0, 4.0))
    routes[:, :, 2] = 1.0
    policy = PublicRouteCoveragePolicy(routes, waypoint_segment_seconds=2.0)
    action = policy(_observation(time_ns=1_000_000_000, x_positions=np.full(AGENT_COUNT, 5.0)))
    assert np.all(action[:, 0] < 0.0)


def test_public_route_policy_uses_explicit_rollout_time_origin() -> None:
    routes = np.zeros((AGENT_COUNT, 3, 3), dtype=np.float64)
    routes[:, :, 0] = np.array((0.0, 2.0, 4.0))
    routes[:, :, 2] = 1.0
    policy = PublicRouteCoveragePolicy(
        routes,
        waypoint_segment_seconds=2.0,
        route_start_time_ns=10_000_000_000,
    )
    action = policy(_observation(time_ns=10_000_000_000, x_positions=np.zeros(AGENT_COUNT)))
    assert np.all(action[:, 0] > 0.0)
    assert policy.provenance()["route_start_time_ns"] == 10_000_000_000


def test_native_detector_uses_camera_local_semantics_and_world_points() -> None:
    labels = np.zeros((AGENT_COUNT, 2, 2, 1), dtype=np.int32)
    labels[2, :, :, 0] = 7
    points = np.zeros((AGENT_COUNT, 4, 3), dtype=np.float64)
    points[2] = np.array(((3.0, 4.0, 5.0),) * 4)
    metadata = {
        "per_camera": [
            {"id_to_labels": {"0": {"class": "background"}}}
            for _ in range(AGENT_COUNT)
        ]
    }
    metadata["per_camera"][2] = {
        "id_to_labels": {"7": {"class": "search_target_slot_000"}}
    }
    candidates = native_semantic_rgbd_candidates(labels, metadata, points, minimum_pixels=4)
    assert len(candidates) == AGENT_COUNT
    assert candidates[2] == (
        T2CandidateDetection(
            agent_id=2,
            position_w_m=(3.0, 4.0, 5.0),
            confidence=0.5,
            deduplication_key="search_target_slot_000",
        ),
    )
    assert all(not rows for index, rows in enumerate(candidates) if index != 2)


def test_native_detector_matches_isaaclab_unproject_depth_pixel_order() -> None:
    """The native detector must follow IsaacLab's u-major point ordering."""

    labels = np.zeros((AGENT_COUNT, 2, 3, 1), dtype=np.int32)
    # This is pixel (v=1, u=2) in the H/W semantic image. IsaacLab's
    # unproject_depth returns it at flat index u * H + v = 5.
    labels[0, 1, 2, 0] = 9
    points = np.zeros((AGENT_COUNT, 6, 3), dtype=np.float64)
    points[0, 5] = (7.0, 8.0, 9.0)
    points[0, 4] = (99.0, 98.0, 97.0)
    metadata = {
        "per_camera": [
            {"id_to_labels": {"9": {"class": "search_target_slot_000"}}}
        ]
        + [{"id_to_labels": {"0": {"class": "background"}}}] * (AGENT_COUNT - 1)
    }

    candidates = native_semantic_rgbd_candidates(labels, metadata, points, minimum_pixels=1)

    assert candidates[0] == (
        T2CandidateDetection(
            agent_id=0,
            position_w_m=(7.0, 8.0, 9.0),
            confidence=0.5,
            deduplication_key="search_target_slot_000",
        ),
    )


def test_native_rgbd_world_points_uses_retained_frame_contract_for_nontrivial_pose() -> None:
    """Catch geometry drift between capture-side and validator-side replay."""

    depth = np.full((AGENT_COUNT, 2, 3, 1), 4.0, dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], AGENT_COUNT, axis=0)
    intrinsics[:, 0, 0] = 2.0
    intrinsics[:, 1, 1] = 4.0
    intrinsics[:, 0, 2] = 1.0
    intrinsics[:, 1, 2] = 0.5
    positions = np.zeros((AGENT_COUNT, 3), dtype=np.float32)
    positions[0] = (10.0, 20.0, 30.0)
    quaternions = np.zeros((AGENT_COUNT, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    quaternions[0] = np.array((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0), dtype=np.float32)

    points = native_rgbd_world_points(depth, intrinsics, positions, quaternions)

    # Pixel (u=2, v=1) has camera point (2, 0.5, 4).  The non-identity
    # quaternion rotates it to (2, -4, 0.5) before the world translation.
    np.testing.assert_allclose(points[0, 5], (12.0, 16.0, 30.5), rtol=0.0, atol=2.0e-6)
    from rivermark_benchmark import native_t2_validate

    assert np.array_equal(
        points,
        native_t2_validate._native_world_points(depth, intrinsics, positions, quaternions),
    )


def test_native_detector_and_deduplicator_fail_closed() -> None:
    labels = np.zeros((AGENT_COUNT, 2, 2, 1), dtype=np.int32)
    metadata = {"per_camera": []}
    with pytest.raises(T2PolicyAbiError, match="one mapping"):
        native_semantic_rgbd_candidates(labels, metadata, np.zeros((AGENT_COUNT, 4, 3)), minimum_pixels=1)
    deduplicator = SpatialCandidateDeduplicator(merge_radius_m=0.5)
    first = T2CandidateDetection(agent_id=0, position_w_m=(1.0, 2.0, 3.0), confidence=1.0)
    repeat = T2CandidateDetection(agent_id=1, position_w_m=(1.1, 2.0, 3.0), confidence=1.0)
    distant = T2CandidateDetection(agent_id=1, position_w_m=(2.0, 2.0, 3.0), confidence=1.0)
    assert deduplicator.filter((first, repeat, distant)) == (first, distant)


def test_semantic_slot_deduplication_suppresses_distant_repeat_without_serializing_identity() -> None:
    deduplicator = SpatialCandidateDeduplicator(merge_radius_m=0.5)
    first = T2CandidateDetection(
        agent_id=2,
        position_w_m=(3.0, 4.0, 5.0),
        confidence=1.0,
        deduplication_key="search_target_slot_000",
    )
    repeat_from_new_view = T2CandidateDetection(
        agent_id=6,
        position_w_m=(9.0, 4.0, 5.0),
        confidence=1.0,
        deduplication_key="search_target_slot_000",
    )
    other_slot = T2CandidateDetection(
        agent_id=6,
        position_w_m=(9.0, 4.0, 5.0),
        confidence=1.0,
        deduplication_key="search_target_slot_001",
    )
    assert deduplicator.filter((first, repeat_from_new_view, other_slot)) == (first, other_slot)
