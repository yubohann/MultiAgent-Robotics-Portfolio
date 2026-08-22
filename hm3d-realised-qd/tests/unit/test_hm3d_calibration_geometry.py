from __future__ import annotations

import numpy as np
import pytest

from aerocity_method.runtime.hm3d_calibration_geometry import (
    densest_height_slice_index,
    farthest_spread_indices,
)


def test_farthest_spread_selection_is_deterministic_and_unique() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    first = farthest_spread_indices(points, count=3, seed=7)
    second = farthest_spread_indices(points, count=3, seed=7)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 3


def test_densest_height_slice_avoids_a_narrow_median_connector() -> None:
    component = np.zeros((4, 4, 5), dtype=bool)
    component[0:3, 0:3, 0] = True
    component[0, 0, 2] = True
    component[0:2, 0:2, 4] = True
    assert densest_height_slice_index(component) == 0


def test_densest_height_slice_rejects_invalid_or_empty_masks() -> None:
    with pytest.raises(ValueError, match="3-D"):
        densest_height_slice_index(np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="no active"):
        densest_height_slice_index(np.zeros((2, 2, 2), dtype=bool))
