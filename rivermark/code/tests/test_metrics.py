from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.metrics import (
    MetricError,
    bootstrap_summary,
    normalized_confirmed_auc,
    paired_bootstrap_difference,
    score_search_episode,
)


class MetricsTests(unittest.TestCase):
    def test_normalized_auc_uses_budget_hold(self) -> None:
        score = normalized_confirmed_auc([0.0, 1.0, 2.0], [0, 1, 2], target_count=2, time_budget_s=4.0)
        self.assertAlmostEqual(score, 0.75)

    def test_episode_score_reports_recall_and_completion_time(self) -> None:
        score = score_search_episode(
            [0.0, 1.0, 2.0],
            [0, 1, 2],
            target_count=2,
            time_budget_s=2.0,
            false_confirmations=1,
            truncated=False,
        )
        self.assertEqual(score.final_recall, 1.0)
        self.assertEqual(score.time_to_all_targets_s, 2.0)
        self.assertEqual(score.false_confirmations, 1)

    def test_non_monotonic_trace_is_rejected(self) -> None:
        with self.assertRaises(MetricError):
            normalized_confirmed_auc([0.0, 1.0], [1, 0], target_count=2, time_budget_s=2.0)

    def test_bootstrap_is_reproducible_and_paired(self) -> None:
        first = np.asarray([0.2, 0.4, 0.6, 0.8])
        second = np.asarray([0.1, 0.3, 0.5, 0.7])
        summary_a = bootstrap_summary(first, metric="auc", resamples=200, seed=17)
        summary_b = bootstrap_summary(first, metric="auc", resamples=200, seed=17)
        self.assertEqual(summary_a, summary_b)
        difference = paired_bootstrap_difference(first, second, metric="auc_delta", resamples=200, seed=17)
        self.assertAlmostEqual(difference.mean, 0.1)
        self.assertLessEqual(difference.ci_low, difference.mean)
        self.assertGreaterEqual(difference.ci_high, difference.mean)

    def test_bootstrap_requires_enough_resamples(self) -> None:
        with self.assertRaises(MetricError):
            bootstrap_summary([1.0], metric="auc", resamples=99)

    def test_invalid_numeric_input_and_seed_fail_with_metric_error(self) -> None:
        with self.assertRaises(MetricError):
            normalized_confirmed_auc([0.0, "bad"], [0, 1], target_count=1, time_budget_s=1.0)
        with self.assertRaises(MetricError):
            bootstrap_summary([1.0], metric="auc", resamples=100, seed=-1)


if __name__ == "__main__":
    unittest.main()
