"""Independent pure-Python reference for the public Search3D metric.

This module intentionally does not import :mod:`numpy` or call the production
implementation.  It is used only to cross-check the versioned metric on the
small public fixture; the production metric definition remains in
``rivermark_benchmark.metrics``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def normalized_confirmed_auc_reference(
    timestamps_s: Sequence[Any],
    confirmed_counts: Sequence[Any],
    *,
    target_count: int,
    time_budget_s: float,
) -> float:
    """Compute the metric using only Python lists and trapezoids."""

    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    if isinstance(time_budget_s, bool) or not isinstance(time_budget_s, (int, float)):
        raise ValueError("time_budget_s must be positive and finite")
    budget = float(time_budget_s)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("time_budget_s must be positive and finite")
    if not isinstance(timestamps_s, Sequence) or not isinstance(confirmed_counts, Sequence):
        raise ValueError("metric traces must be sequences")
    if len(timestamps_s) == 0 or len(timestamps_s) != len(confirmed_counts):
        raise ValueError("metric traces must be non-empty and have equal length")

    times = [float(value) for value in timestamps_s]
    counts = [float(value) for value in confirmed_counts]
    if any(not math.isfinite(value) for value in times + counts):
        raise ValueError("metric traces must be finite")
    if times[0] < 0.0 or times[-1] > budget:
        raise ValueError("timestamps_s must lie within [0, time_budget_s]")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps_s must be strictly increasing")
    if any(value < 0.0 or value > target_count or value != math.floor(value) for value in counts):
        raise ValueError("confirmed_counts must be non-decreasing integer counts within target_count")
    if any(right < left for left, right in zip(counts, counts[1:])):
        raise ValueError("confirmed_counts must be non-decreasing")
    if times[0] > 0.0:
        times.insert(0, 0.0)
        counts.insert(0, 0.0)
    if times[-1] < budget:
        times.append(budget)
        counts.append(counts[-1])
    area_terms = [
        ((left_count + right_count) / (2.0 * target_count)) * (right_time - left_time)
        for left_time, right_time, left_count, right_count in zip(times, times[1:], counts, counts[1:])
    ]
    return max(0.0, min(1.0, math.fsum(area_terms) / budget))


def score_search_episode_reference(
    timestamps_s: Sequence[Any],
    confirmed_counts: Sequence[Any],
    *,
    target_count: int,
    time_budget_s: float,
    false_confirmations: int = 0,
    truncated: bool = False,
) -> dict[str, Any]:
    """Return the fixture-shaped score without importing production code."""

    if not isinstance(false_confirmations, int) or isinstance(false_confirmations, bool) or false_confirmations < 0:
        raise ValueError("false_confirmations must be a non-negative integer")
    if not isinstance(truncated, bool):
        raise ValueError("truncated must be boolean")
    auc = normalized_confirmed_auc_reference(
        timestamps_s,
        confirmed_counts,
        target_count=target_count,
        time_budget_s=time_budget_s,
    )
    counts = [int(value) for value in confirmed_counts]
    times = [float(value) for value in timestamps_s]
    reached = [time for time, count in zip(times, counts) if count >= target_count]
    return {
        "normalized_confirmed_auc": auc,
        "final_recall": counts[-1] / float(target_count),
        "false_confirmations": false_confirmations,
        "time_to_all_targets_s": reached[0] if reached else None,
        "truncated": truncated,
    }


__all__ = ["normalized_confirmed_auc_reference", "score_search_episode_reference"]
