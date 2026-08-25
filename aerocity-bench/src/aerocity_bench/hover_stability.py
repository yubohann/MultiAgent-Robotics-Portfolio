"""Pure-Python long-horizon CF2X hover metrics and candidate gate.

Short controller smoke tests can hide a small persistent altitude trend.  This
module deliberately measures the late part of a native flight trace, so a
vehicle that slowly descends (or climbs) cannot pass merely because its first
few seconds look stable.  The threshold set is an engineering preflight
candidate only; it never promotes the CF2X model to formal-score eligibility.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class LongHorizonHoverMetrics:
    """Measured altitude behavior of a single native hover trace."""

    trace_samples: int
    simulated_duration_s: float
    initial_altitude_m: float
    final_altitude_m: float
    final_altitude_error_m: float
    max_abs_altitude_error_m: float
    terminal_vertical_velocity_mps: float
    late_window_start_s: float
    late_window_samples: int
    late_min_altitude_error_m: float
    late_max_altitude_error_m: float
    late_altitude_slope_mps: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "trace_samples": self.trace_samples,
            "simulated_duration_s": self.simulated_duration_s,
            "initial_altitude_m": self.initial_altitude_m,
            "final_altitude_m": self.final_altitude_m,
            "final_altitude_error_m": self.final_altitude_error_m,
            "max_abs_altitude_error_m": self.max_abs_altitude_error_m,
            "terminal_vertical_velocity_mps": self.terminal_vertical_velocity_mps,
            "late_window_start_s": self.late_window_start_s,
            "late_window_samples": self.late_window_samples,
            "late_min_altitude_error_m": self.late_min_altitude_error_m,
            "late_max_altitude_error_m": self.late_max_altitude_error_m,
            "late_altitude_slope_mps": self.late_altitude_slope_mps,
        }


@dataclass(frozen=True)
class LongHorizonHoverThresholds:
    """Candidate, non-formal bounds for a 30-second stationary hover.

    The slope bound limits a continuing late-window trend to 5 mm across the
    final 20 seconds.  It is materially stricter than merely staying above the
    ground, while still leaving a large margin above the observed native
    quantization-level drift.
    """

    minimum_duration_s: float = 30.0
    warmup_s: float = 10.0
    max_abs_altitude_error_m: float = 0.02
    max_abs_final_altitude_error_m: float = 0.01
    max_abs_terminal_vertical_velocity_mps: float = 0.01
    max_abs_late_altitude_slope_mps: float = 2.5e-4
    max_contact_force_n: float = 1.0e-6

    def validate(self) -> None:
        duration = _finite(self.minimum_duration_s, "minimum_duration_s")
        warmup = _finite(self.warmup_s, "warmup_s")
        if duration <= 0.0:
            raise ValueError("minimum_duration_s must be positive")
        if warmup < 0.0 or warmup >= duration:
            raise ValueError("warmup_s must be non-negative and below the duration")
        for name, value in {
            "max_abs_altitude_error_m": self.max_abs_altitude_error_m,
            "max_abs_final_altitude_error_m": self.max_abs_final_altitude_error_m,
            "max_abs_terminal_vertical_velocity_mps": self.max_abs_terminal_vertical_velocity_mps,
            "max_abs_late_altitude_slope_mps": self.max_abs_late_altitude_slope_mps,
            "max_contact_force_n": self.max_contact_force_n,
        }.items():
            if _finite(value, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def fingerprint_payload(self) -> dict[str, float | str]:
        self.validate()
        return {
            "schema": "org.aerocity.bench.long-horizon-hover-thresholds.v1",
            "status": "candidate_preflight_only",
            "minimum_duration_s": self.minimum_duration_s,
            "warmup_s": self.warmup_s,
            "max_abs_altitude_error_m": self.max_abs_altitude_error_m,
            "max_abs_final_altitude_error_m": self.max_abs_final_altitude_error_m,
            "max_abs_terminal_vertical_velocity_mps": self.max_abs_terminal_vertical_velocity_mps,
            "max_abs_late_altitude_slope_mps": self.max_abs_late_altitude_slope_mps,
            "max_contact_force_n": self.max_contact_force_n,
        }


def candidate_long_horizon_hover_thresholds() -> LongHorizonHoverThresholds:
    """Return the reviewed non-formal long-horizon preflight candidate."""

    thresholds = LongHorizonHoverThresholds()
    thresholds.validate()
    return thresholds


def long_horizon_hover_metrics(
    sample_times_s: Iterable[float],
    sample_altitudes_m: Iterable[float],
    *,
    initial_altitude_m: float,
    terminal_vertical_velocity_mps: float,
    warmup_s: float,
) -> LongHorizonHoverMetrics:
    """Compute a late-window least-squares altitude trend from a flight trace.

    Input timestamps must be strictly increasing and the terminal sample must be
    included.  This prevents a sparse or reordered receipt from concealing a
    late descent.
    """

    times = tuple(_finite(value, "sample time") for value in sample_times_s)
    altitudes = tuple(_finite(value, "sample altitude") for value in sample_altitudes_m)
    initial = _finite(initial_altitude_m, "initial_altitude_m")
    terminal_velocity = _finite(terminal_vertical_velocity_mps, "terminal_vertical_velocity_mps")
    warmup = _finite(warmup_s, "warmup_s")
    if len(times) != len(altitudes) or len(times) < 3:
        raise ValueError("hover trace needs at least three paired samples")
    if times[0] <= 0.0 or any(
        right <= left for left, right in zip(times[:-1], times[1:], strict=True)
    ):
        raise ValueError("hover trace timestamps must be positive and strictly increasing")
    if warmup < 0.0 or warmup >= times[-1]:
        raise ValueError("warmup_s must be non-negative and precede the final sample")

    errors = tuple(altitude - initial for altitude in altitudes)
    late_pairs = tuple(
        (time_s, error_m)
        for time_s, error_m in zip(times, errors, strict=True)
        if time_s >= warmup
    )
    if len(late_pairs) < 3:
        raise ValueError("hover trace needs at least three post-warm-up samples")
    late_time_mean = sum(time_s for time_s, _ in late_pairs) / len(late_pairs)
    late_error_mean = sum(error_m for _, error_m in late_pairs) / len(late_pairs)
    denominator = sum((time_s - late_time_mean) ** 2 for time_s, _ in late_pairs)
    if denominator <= 1.0e-12:
        raise ValueError("post-warm-up timestamps cannot determine a slope")
    slope = (
        sum(
            (time_s - late_time_mean) * (error_m - late_error_mean)
            for time_s, error_m in late_pairs
        )
        / denominator
    )

    return LongHorizonHoverMetrics(
        trace_samples=len(times),
        simulated_duration_s=times[-1],
        initial_altitude_m=initial,
        final_altitude_m=altitudes[-1],
        final_altitude_error_m=errors[-1],
        max_abs_altitude_error_m=max(abs(error_m) for error_m in errors),
        terminal_vertical_velocity_mps=terminal_velocity,
        late_window_start_s=warmup,
        late_window_samples=len(late_pairs),
        late_min_altitude_error_m=min(error_m for _, error_m in late_pairs),
        late_max_altitude_error_m=max(error_m for _, error_m in late_pairs),
        late_altitude_slope_mps=slope,
    )


def long_horizon_hover_checks(
    metrics: LongHorizonHoverMetrics,
    thresholds: LongHorizonHoverThresholds,
    *,
    max_contact_force_n: float,
) -> dict[str, bool]:
    """Return explicit pass/fail checks without hiding a downward trend."""

    thresholds.validate()
    contact_force = _finite(max_contact_force_n, "max_contact_force_n")
    if contact_force < 0.0:
        raise ValueError("max_contact_force_n must be non-negative")
    return {
        "long_hold_minimum_simulated_duration_satisfied": (
            metrics.simulated_duration_s >= thresholds.minimum_duration_s
        ),
        "long_hold_max_abs_altitude_error_satisfied": (
            metrics.max_abs_altitude_error_m <= thresholds.max_abs_altitude_error_m
        ),
        "long_hold_final_altitude_error_satisfied": (
            abs(metrics.final_altitude_error_m) <= thresholds.max_abs_final_altitude_error_m
        ),
        "long_hold_terminal_vertical_velocity_satisfied": (
            abs(metrics.terminal_vertical_velocity_mps)
            <= thresholds.max_abs_terminal_vertical_velocity_mps
        ),
        "long_hold_late_altitude_slope_satisfied": (
            abs(metrics.late_altitude_slope_mps)
            <= thresholds.max_abs_late_altitude_slope_mps
        ),
        "long_hold_no_contact_satisfied": contact_force <= thresholds.max_contact_force_n,
    }
