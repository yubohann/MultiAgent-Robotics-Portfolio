from __future__ import annotations

import math

import numpy as np
import pytest

from shared.core.team_geometry import (
    boundary_proximity_deficit_m,
    count_lateral_bands,
    pairwise_separation_stats,
    slot_error_stats,
)


def test_boundary_proximity_deficit_is_zero_outside_the_soft_margin() -> None:
    deficit_m = boundary_proximity_deficit_m(
        np.asarray(((3.0, 5.0), (7.0, 5.0)), dtype=np.float32),
        world_x_bounds_m=(0.0, 10.0),
        world_y_bounds_m=(0.0, 10.0),
        agent_radius_m=0.5,
        soft_margin_m=1.0,
    )
    assert deficit_m == 0.0


def test_boundary_proximity_deficit_grows_for_margin_and_bounds_violations() -> None:
    near_boundary_m = boundary_proximity_deficit_m(
        np.asarray(((1.0, 5.0),), dtype=np.float32),
        world_x_bounds_m=(0.0, 10.0),
        world_y_bounds_m=(0.0, 10.0),
        agent_radius_m=0.5,
        soft_margin_m=1.0,
    )
    out_of_bounds_m = boundary_proximity_deficit_m(
        np.asarray(((-0.5, 5.0),), dtype=np.float32),
        world_x_bounds_m=(0.0, 10.0),
        world_y_bounds_m=(0.0, 10.0),
        agent_radius_m=0.5,
        soft_margin_m=1.0,
    )
    assert near_boundary_m == pytest.approx(0.5)
    assert out_of_bounds_m == pytest.approx(2.0)


def test_pairwise_separation_tracks_all_pairs_and_uses_an_inclusive_threshold() -> None:
    stats = pairwise_separation_stats(
        np.asarray(((0.0, 0.0), (3.0, 4.0), (3.0, 0.0)), dtype=np.float32),
        safe_distance_m=3.0,
    )
    assert stats.collision
    assert stats.min_distance_m == pytest.approx(3.0)
    assert stats.pair_count == 3


def test_pairwise_separation_has_no_pairs_for_a_single_agent() -> None:
    stats = pairwise_separation_stats(
        np.asarray(((0.0, 0.0),), dtype=np.float32),
        safe_distance_m=1.0,
    )
    assert not stats.collision
    assert math.isinf(stats.min_distance_m)
    assert stats.pair_count == 0


def test_count_lateral_bands_uses_running_cluster_centers() -> None:
    assert count_lateral_bands(np.asarray((-1.0, -0.7, 0.2, 0.35, 1.2), dtype=np.float32), band_width_m=0.4) == 3
    assert count_lateral_bands(np.zeros((0,), dtype=np.float32), band_width_m=0.4) == 0


def test_slot_error_stats_aggregates_mean_and_maximum_distances() -> None:
    mean_error_m, max_error_m = slot_error_stats(
        np.asarray(((0.0, 0.0), (3.0, 4.0)), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    assert mean_error_m == pytest.approx(2.5)
    assert max_error_m == pytest.approx(5.0)
    assert slot_error_stats(np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)) == (0.0, 0.0)


def test_team_geometry_rejects_ambiguous_position_shapes() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        pairwise_separation_stats(np.asarray((1.0, 2.0), dtype=np.float32), safe_distance_m=1.0)
