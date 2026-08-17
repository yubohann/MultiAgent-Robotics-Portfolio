"""Versioned, public Search3D metrics and bootstrap summaries.

This module scores evaluator-produced confirmation traces.  It never accepts
target coordinates and therefore cannot replace the private truth service for
blind evaluation.  It provides the public metric definition and statistical
aggregation needed for train/validation reports.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


METRIC_VERSION = "org.rivermark.benchmark.metrics.v1"


class MetricError(ValueError):
    """Raised when a metric input violates the public metric contract."""


@dataclass(frozen=True)
class SearchMetrics:
    normalized_confirmed_auc: float
    final_recall: float
    false_confirmations: int
    time_to_all_targets_s: float | None
    truncated: bool
    metric_version: str = METRIC_VERSION


@dataclass(frozen=True)
class BootstrapSummary:
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    sample_count: int
    confidence: float
    resamples: int
    seed: int
    metric_version: str = METRIC_VERSION


def _series(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{name} must be numeric") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise MetricError(f"{name} must be a non-empty finite one-dimensional series")
    return array


def normalized_confirmed_auc(
    timestamps_s: Any,
    confirmed_counts: Any,
    *,
    target_count: int,
    time_budget_s: float,
) -> float:
    """Compute normalized area under confirmed-target recall over time.

    The trace is sampled at evaluator timestamps.  A missing initial sample is
    treated as zero confirmations at time zero; a trace ending before the
    budget is held constant through the budget.  This convention is explicit
    so decimated sensor streams cannot silently change the score.
    """

    times = _series(timestamps_s, "timestamps_s")
    counts = _series(confirmed_counts, "confirmed_counts")
    if times.shape != counts.shape:
        raise MetricError("timestamps_s and confirmed_counts must have the same length")
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 1:
        raise MetricError("target_count must be a positive integer")
    if (
        isinstance(time_budget_s, bool)
        or not isinstance(time_budget_s, (int, float))
        or not np.isfinite(time_budget_s)
        or time_budget_s <= 0
    ):
        raise MetricError("time_budget_s must be positive and finite")
    if times[0] < 0.0 or times[-1] > float(time_budget_s):
        raise MetricError("timestamps_s must lie within [0, time_budget_s]")
    if np.any(np.diff(times) <= 0.0):
        raise MetricError("timestamps_s must be strictly increasing")
    if np.any(np.diff(counts) < 0.0) or np.any(counts < 0.0) or np.any(counts > target_count):
        raise MetricError("confirmed_counts must be non-decreasing and within target_count")
    if not np.all(np.equal(counts, np.floor(counts))):
        raise MetricError("confirmed_counts must contain integers")
    if times[0] > 0.0:
        times = np.concatenate((np.asarray([0.0]), times))
        counts = np.concatenate((np.asarray([0.0]), counts))
    if times[-1] < float(time_budget_s):
        times = np.concatenate((times, np.asarray([float(time_budget_s)])))
        counts = np.concatenate((counts, np.asarray([counts[-1]])))
    recall = counts / float(target_count)
    # NumPy 2 removed ``trapz``. Do not put it in ``getattr``'s default,
    # because Python evaluates that default even when ``trapezoid`` exists.
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    score = float(integrate(recall, times) / float(time_budget_s))
    return float(np.clip(score, 0.0, 1.0))


def score_search_episode(
    timestamps_s: Any,
    confirmed_counts: Any,
    *,
    target_count: int,
    time_budget_s: float,
    false_confirmations: int = 0,
    truncated: bool = False,
) -> SearchMetrics:
    """Score one evaluator confirmation trace without exposing target truth."""

    if not isinstance(false_confirmations, int) or isinstance(false_confirmations, bool) or false_confirmations < 0:
        raise MetricError("false_confirmations must be a non-negative integer")
    if not isinstance(truncated, bool):
        raise MetricError("truncated must be boolean")
    times = _series(timestamps_s, "timestamps_s")
    counts = _series(confirmed_counts, "confirmed_counts")
    auc = normalized_confirmed_auc(
        times,
        counts,
        target_count=target_count,
        time_budget_s=time_budget_s,
    )
    final_recall = float(counts[-1] / float(target_count))
    reached = np.flatnonzero(counts >= target_count)
    time_to_all = float(times[int(reached[0])]) if reached.size else None
    return SearchMetrics(auc, final_recall, false_confirmations, time_to_all, truncated)


def bootstrap_summary(
    values: Any,
    *,
    metric: str,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapSummary:
    """Return a reproducible percentile bootstrap CI for the sample mean."""

    observations = _series(values, "values")
    if not isinstance(metric, str) or not metric.strip():
        raise MetricError("metric must be a non-empty string")
    if not 0.0 < confidence < 1.0:
        raise MetricError("confidence must be in (0, 1)")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise MetricError("resamples must be an integer >= 100")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise MetricError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(np.mean(observations[rng.integers(0, observations.size, observations.size)]))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapSummary(
        metric=metric,
        mean=float(np.mean(observations)),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
        sample_count=int(observations.size),
        confidence=float(confidence),
        resamples=resamples,
        seed=seed,
    )


def paired_bootstrap_difference(
    first: Any,
    second: Any,
    *,
    metric: str,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapSummary:
    """Summarize paired per-episode differences ``first - second``."""

    left = _series(first, "first")
    right = _series(second, "second")
    if left.shape != right.shape:
        raise MetricError("paired samples must have the same length")
    return bootstrap_summary(
        left - right,
        metric=metric,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="JSON array of numeric episode scores")
    parser.add_argument("--metric", default="normalized_confirmed_auc")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        values = json.loads(args.scores.read_text(encoding="utf-8"))
        result = bootstrap_summary(
            values,
            metric=args.metric,
            confidence=args.confidence,
            resamples=args.resamples,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, MetricError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
