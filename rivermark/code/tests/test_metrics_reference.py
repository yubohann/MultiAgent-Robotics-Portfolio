from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.metrics import METRIC_VERSION, score_search_episode  # noqa: E402
from rivermark_benchmark.metrics_reference import score_search_episode_reference  # noqa: E402


class MetricsReferenceTests(unittest.TestCase):
    def test_reference_agrees_with_production_on_shared_fixture(self) -> None:
        fixture = json.loads((ROOT / "tests" / "fixtures" / "metrics_validation_fixture.json").read_text(encoding="utf-8"))
        self.assertEqual(fixture["metric_version"], METRIC_VERSION)
        for case in fixture["cases"]:
            production = score_search_episode(
                case["timestamps_s"],
                case["confirmed_counts"],
                target_count=case["target_count"],
                time_budget_s=case["time_budget_s"],
                false_confirmations=case["false_confirmations"],
                truncated=case["truncated"],
            )
            reference = score_search_episode_reference(
                case["timestamps_s"],
                case["confirmed_counts"],
                target_count=case["target_count"],
                time_budget_s=case["time_budget_s"],
                false_confirmations=case["false_confirmations"],
                truncated=case["truncated"],
            )
            expected = case["expected"]
            self.assertAlmostEqual(production.normalized_confirmed_auc, expected["normalized_confirmed_auc"])
            self.assertAlmostEqual(reference["normalized_confirmed_auc"], expected["normalized_confirmed_auc"])
            self.assertAlmostEqual(production.final_recall, expected["final_recall"])
            self.assertAlmostEqual(reference["final_recall"], expected["final_recall"])
            self.assertEqual(production.time_to_all_targets_s, expected["time_to_all_targets_s"])
            self.assertEqual(reference["time_to_all_targets_s"], expected["time_to_all_targets_s"])
            self.assertEqual(production.false_confirmations, case["false_confirmations"])
            self.assertEqual(production.truncated, case["truncated"])


if __name__ == "__main__":
    unittest.main()
