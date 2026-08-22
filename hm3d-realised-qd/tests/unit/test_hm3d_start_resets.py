"""Unit checks for deterministic, environment-side P07 start candidates."""

from __future__ import annotations

import numpy as np
import pytest

from aerocity_method.runtime.hm3d_start_resets import (
    largest_component_departure_witnesses,
    largest_component_clearance_points,
    select_local_spread_positions,
)


def test_local_spread_positions_are_deterministic_and_separated() -> None:
    x_axis, y_axis, z_axis = np.meshgrid(
        np.arange(5, dtype=np.float64),
        np.arange(5, dtype=np.float64),
        np.arange(2, dtype=np.float64),
        indexing="ij",
    )
    points = np.column_stack((x_axis.ravel(), y_axis.ravel(), z_axis.ravel()))

    first = select_local_spread_positions(
        points,
        count=12,
        seed=20260803,
        cluster_radius_m=3.5,
        minimum_separation_m=0.75,
    )
    second = select_local_spread_positions(
        points,
        count=12,
        seed=20260803,
        cluster_radius_m=3.5,
        minimum_separation_m=0.75,
    )

    assert np.array_equal(first, second)
    assert len(first) == 12
    distances = np.linalg.norm(first[:, None, :] - first[None, :, :], axis=2)
    assert np.all(distances[np.triu_indices(len(first), k=1)] >= 0.75)


def test_local_spread_positions_reject_an_impossible_cluster() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])

    with pytest.raises(ValueError, match="no local collision-admitted cluster"):
        select_local_spread_positions(
            points,
            count=3,
            seed=1,
            cluster_radius_m=1.0,
            minimum_separation_m=0.75,
        )


def test_largest_component_clearance_points_excludes_body_only_voxels() -> None:
    arrays = {
        "free_mask": np.asarray(
            [
                [[True, True]],
                [[True, True]],
            ]
        ),
        "component_labels": np.asarray(
            [
                [[1, 1]],
                [[1, 2]],
            ],
            dtype=np.int32,
        ),
        "collision_distance_m": np.asarray(
            [
                [[0.30, 0.50]],
                [[0.75, 1.00]],
            ]
        ),
        "origin_center_m": np.asarray((1.0, 2.0, 3.0)),
        "resolution_m": np.asarray(0.25),
    }

    points = largest_component_clearance_points(arrays, minimum_clearance_m=0.50)

    assert {tuple(point) for point in points} == {(1.0, 2.0, 3.25), (1.25, 2.0, 3.0)}


def test_departure_witnesses_exclude_isolated_high_clearance_voxels() -> None:
    arrays = {
        "free_mask": np.ones((4, 1, 1), dtype=bool),
        "component_labels": np.ones((4, 1, 1), dtype=np.int32),
        # With 0.25 m voxels and a 0.55 m route-sample requirement, a launch
        # witness needs 0.675 m at both ends.  The last cell is safe to hold
        # but has no qualifying first hop and must not become a reset point.
        "collision_distance_m": np.asarray([[[0.80]], [[0.80]], [[0.80]], [[0.60]]]),
        "origin_center_m": np.asarray((0.0, 0.0, 0.0)),
        "resolution_m": np.asarray(0.25),
    }

    starts, endpoints, grid_tube_clearance_m = largest_component_departure_witnesses(
        arrays,
        minimum_route_sample_clearance_m=0.55,
    )

    assert grid_tube_clearance_m == pytest.approx(0.675)
    assert {tuple(point) for point in starts} == {(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0)}
    assert len(endpoints) == len(starts)
    assert np.all(np.linalg.norm(endpoints - starts, axis=1) == pytest.approx(0.25))


def test_departure_witnesses_reject_component_without_a_route_sample_clear_hop() -> None:
    arrays = {
        "free_mask": np.ones((2, 1, 1), dtype=bool),
        "component_labels": np.ones((2, 1, 1), dtype=np.int32),
        "collision_distance_m": np.asarray([[[0.80]], [[0.60]]]),
        "origin_center_m": np.asarray((0.0, 0.0, 0.0)),
        "resolution_m": np.asarray(0.25),
    }

    with pytest.raises(ValueError, match="no route-sample-clear first departure"):
        largest_component_departure_witnesses(
            arrays,
            minimum_route_sample_clearance_m=0.55,
        )
