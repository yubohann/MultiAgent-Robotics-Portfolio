from __future__ import annotations

import pytest

from aerocity_bench.hover_stability import (
    candidate_long_horizon_hover_thresholds,
    long_horizon_hover_checks,
    long_horizon_hover_metrics,
)


def test_candidate_long_horizon_hover_gate_accepts_a_bounded_30_second_trace() -> None:
    thresholds = candidate_long_horizon_hover_thresholds()
    times = tuple(float(index) for index in range(1, 31))
    altitudes = tuple(1.5 - 1.0e-5 * time_s for time_s in times)
    metrics = long_horizon_hover_metrics(
        times,
        altitudes,
        initial_altitude_m=1.5,
        terminal_vertical_velocity_mps=-1.0e-5,
        warmup_s=thresholds.warmup_s,
    )

    assert metrics.late_altitude_slope_mps == pytest.approx(-1.0e-5)
    assert all(long_horizon_hover_checks(metrics, thresholds, max_contact_force_n=0.0).values())


def test_candidate_long_horizon_hover_gate_rejects_a_continuing_descent() -> None:
    thresholds = candidate_long_horizon_hover_thresholds()
    times = tuple(float(index) for index in range(1, 31))
    altitudes = tuple(1.5 - 1.0e-3 * time_s for time_s in times)
    metrics = long_horizon_hover_metrics(
        times,
        altitudes,
        initial_altitude_m=1.5,
        terminal_vertical_velocity_mps=-1.0e-3,
        warmup_s=thresholds.warmup_s,
    )
    checks = long_horizon_hover_checks(metrics, thresholds, max_contact_force_n=0.0)

    assert checks["long_hold_late_altitude_slope_satisfied"] is False
    assert checks["long_hold_max_abs_altitude_error_satisfied"] is False


def test_long_horizon_metrics_reject_reordered_or_too_short_traces() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        long_horizon_hover_metrics(
            (1.0, 2.0, 2.0),
            (1.5, 1.5, 1.5),
            initial_altitude_m=1.5,
            terminal_vertical_velocity_mps=0.0,
            warmup_s=0.5,
        )

    with pytest.raises(ValueError, match="post-warm-up samples"):
        long_horizon_hover_metrics(
            (1.0, 2.0, 3.0),
            (1.5, 1.5, 1.5),
            initial_altitude_m=1.5,
            terminal_vertical_velocity_mps=0.0,
            warmup_s=2.5,
        )
