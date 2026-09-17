"""Evaluator-side geometry helpers for target-free HM3D calibration."""

from __future__ import annotations

import numpy as np


def farthest_spread_indices(points: np.ndarray, *, count: int, seed: int) -> np.ndarray:
    """Select deterministic, geometry-only spread points from 3-D free space."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 1:
        raise ValueError("points must be a non-empty (N, 3) array")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    if not 1 <= count <= len(values):
        raise ValueError("count must be in [1, len(points)]")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(values)))
    selected = [first]
    nearest_sq = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest_sq))
        selected.append(index)
        nearest_sq = np.minimum(nearest_sq, np.sum((values - values[index]) ** 2, axis=1))
    return np.asarray(selected, dtype=np.int64)


def densest_height_slice_index(component: np.ndarray) -> int:
    """Return the fixed-height slice with the most connected free-flight cells."""

    mask = np.asarray(component, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("component must be a 3-D boolean mask")
    counts = np.count_nonzero(mask, axis=(0, 1))
    if not np.any(counts):
        raise ValueError("component has no active height slice")
    return int(np.argmax(counts))


__all__ = ["densest_height_slice_index", "farthest_spread_indices"]
