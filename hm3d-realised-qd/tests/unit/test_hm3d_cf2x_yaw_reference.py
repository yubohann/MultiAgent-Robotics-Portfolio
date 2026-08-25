from __future__ import annotations

import pytest

from aerocity_method.runtime.hm3d_cf2x_execution import (
    _rate_limited_yaw_reference_deg,
    _shortest_angular_delta_deg,
)


def test_yaw_reference_crosses_wrap_boundary_by_shortest_turn() -> None:
    next_heading = _rate_limited_yaw_reference_deg(
        179.0,
        -170.0,
        0.1,
        maximum_rate_deg_s=10.0,
    )

    assert _shortest_angular_delta_deg(179.0, next_heading) == pytest.approx(1.0)


def test_yaw_reference_never_exceeds_rate_limit() -> None:
    next_heading = _rate_limited_yaw_reference_deg(
        0.0,
        135.0,
        1.0 / 120.0,
        maximum_rate_deg_s=10.0,
    )

    assert abs(_shortest_angular_delta_deg(0.0, next_heading)) <= 10.0 / 120.0 + 1e-12


def test_stationary_heading_target_keeps_existing_reference() -> None:
    current = -73.25

    assert _rate_limited_yaw_reference_deg(current, current, 1.0 / 120.0) == pytest.approx(
        current
    )


@pytest.mark.parametrize("dt_s", [0.0, -0.1, float("nan")])
def test_yaw_reference_rejects_invalid_time_step(dt_s: float) -> None:
    with pytest.raises(ValueError, match="time step"):
        _rate_limited_yaw_reference_deg(0.0, 10.0, dt_s)
