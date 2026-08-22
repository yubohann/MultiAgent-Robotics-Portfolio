from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "materialize_hm3d_two_sided_collision_usd.py"
    spec = importlib.util.spec_from_file_location("hm3d_two_sided_collision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_sided_triangle_indices_preserve_then_reverse_every_face() -> None:
    module = _module()

    counts, indices = module._two_sided_triangle_indices(
        np.asarray([3, 3]), np.asarray([0, 1, 2, 2, 3, 0])
    )

    assert counts.tolist() == [3, 3, 3, 3]
    assert indices.tolist() == [0, 1, 2, 2, 3, 0, 0, 2, 1, 2, 0, 3]


def test_two_sided_triangle_indices_reject_non_triangles_and_bad_lengths() -> None:
    module = _module()
    with pytest.raises(ValueError, match="triangle"):
        module._two_sided_triangle_indices(np.asarray([4]), np.asarray([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="does not match"):
        module._two_sided_triangle_indices(np.asarray([3]), np.asarray([0, 1]))
