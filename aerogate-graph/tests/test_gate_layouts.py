from __future__ import annotations

from gate_density_single.scripts.run_gate_density_eval import (
    ALLOWED_GATE_LAYOUT_VERSIONS,
    _generate_gate_layout,
    _layout_profile,
)


def test_gate_layout_versions_generate_expected_counts() -> None:
    for version in ALLOWED_GATE_LAYOUT_VERSIONS:
        centers, yaws = _generate_gate_layout(
            gate_count=8,
            seed=3,
            random_yaw=True,
            layout_version=version,
        )
        assert len(centers) == 8
        assert len(yaws) == 8
        profile = _layout_profile(version)
        x_min, x_max = profile.world_x_bounds_m
        y_min, y_max = profile.world_y_bounds_m
        for x_value, y_value in centers:
            assert x_min <= x_value <= x_max
            assert y_min <= y_value <= y_max


def test_empty_layout_is_empty() -> None:
    centers, yaws = _generate_gate_layout(gate_count=0, seed=0, random_yaw=True)
    assert centers == ()
    assert yaws == ()
