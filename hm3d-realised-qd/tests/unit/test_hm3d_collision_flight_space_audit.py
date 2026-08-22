from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "audit_hm3d_collision_flight_space.py"
    spec = importlib.util.spec_from_file_location("hm3d_collision_flight_space_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vertical_statistics_does_not_treat_a_tiny_stair_sliver_as_a_floor_band() -> None:
    module = _module()
    free = np.zeros((8, 8, 5), dtype=bool)
    free[:, :, 0] = True
    free[:, :, 1] = True
    free[0, 0, 2] = True
    free[:, :, 3] = True
    free[:, :, 4] = True

    report = module._vertical_statistics({"free_mask": free}, minimum_voxels=1)

    assert report["substantial_height_slice_minimum_voxels"] == 4
    assert report["active_height_slice_indices"] == [0, 1, 3, 4]
    assert report["connected_height_band_count"] == 2


def test_vertical_statistics_preserves_one_continuous_height_band() -> None:
    module = _module()
    free = np.ones((4, 4, 4), dtype=bool)

    report = module._vertical_statistics({"free_mask": free}, minimum_voxels=1)

    assert report["connected_height_band_count"] == 1
