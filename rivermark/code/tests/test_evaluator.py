from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.evaluator import (  # noqa: E402
    MAX_SUBMISSION_EPISODES,
    MAX_TRACE_SAMPLES,
    RESULT_SCHEMA,
    SUBMISSION_SCHEMA,
    EvaluatorSubmissionError,
    evaluate_submission,
    evaluate_submission_file,
    validate_submission,
)


def _submission() -> dict:
    trace = {
        "timestamps_s": [0.0, 1.0, 2.0],
        "confirmed_counts": [0, 1, 2],
        "target_count": 2,
        "time_budget_s": 2.0,
        "false_confirmations": 0,
        "truncated": False,
    }
    return {
        "schema": SUBMISSION_SCHEMA,
        "dataset_version": "0.1.0",
        "dataset_index_sha256": "a" * 64,
        "split": "validation",
        "evaluator": {
            "evaluator_id": "public-search3d",
            "evaluator_version": "1.0.0",
            "evaluator_sha256": "b" * 64,
            "metric_schema": "org.rivermark.benchmark.metrics.v1",
        },
        "policy": {
            "method_id": "classical-route",
            "code_revision": "c" * 40,
            "checkpoint_sha256": "d" * 64,
            "seed": 7,
        },
        "episodes": [
            {"episode_id": "validation-001", "split": "validation", "trace": trace},
        ],
    }


class EvaluatorTests(unittest.TestCase):
    def test_valid_submission_is_scored_without_truth(self) -> None:
        report = evaluate_submission(_submission())
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(report.schema, RESULT_SCHEMA)
        self.assertEqual(report.dataset_index_sha256, "a" * 64)
        self.assertEqual(report.evaluator_sha256, "b" * 64)
        self.assertEqual(report.code_revision, "c" * 40)
        self.assertEqual(report.checkpoint_sha256, "d" * 64)
        self.assertEqual(report.episode_count, 1)
        self.assertEqual(report.scores[0].final_recall, 1.0)
        self.assertEqual(report.scores[0].time_to_all_targets_s, 2.0)

    def test_schema_file_matches_runtime_contract(self) -> None:
        schema = json.loads((ROOT / "schemas/evaluator_submission_v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"], SUBMISSION_SCHEMA)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["episodes"]["maxItems"], MAX_SUBMISSION_EPISODES)
        trace_schema = schema["properties"]["episodes"]["items"]["properties"]["trace"]["properties"]
        self.assertEqual(trace_schema["timestamps_s"]["maxItems"], MAX_TRACE_SAMPLES)
        result_schema = json.loads((ROOT / "schemas/evaluator_result_v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(result_schema["properties"]["schema"]["const"], RESULT_SCHEMA)

    def test_private_truth_and_unknown_fields_are_rejected(self) -> None:
        payload = _submission()
        payload["target_coordinates"] = [[1.0, 2.0, 3.0]]
        payload["episodes"][0]["trace"]["reward"] = 1.0
        codes = {issue.code for issue in validate_submission(payload)}
        self.assertIn("unknown_field", codes)
        self.assertIn("private_field", codes)

    def test_dataset_and_split_bindings_are_checked(self) -> None:
        payload = _submission()
        issues = validate_submission(payload, expected_dataset_version="0.2.0", expected_split="blind_test")
        paths = {issue.path for issue in issues}
        self.assertIn("$.dataset_version", paths)
        self.assertIn("$.split", paths)

    def test_dataset_index_binding_and_resource_limits_are_fail_closed(self) -> None:
        payload = _submission()
        issues = validate_submission(payload, expected_dataset_index_sha256="c" * 64)
        self.assertIn("dataset_index_sha256", {issue.code for issue in issues})

        too_many_episodes = _submission()
        too_many_episodes["episodes"] = too_many_episodes["episodes"] * (MAX_SUBMISSION_EPISODES + 1)
        issues = validate_submission(too_many_episodes)
        self.assertIn("resource_budget", {issue.code for issue in issues})

        too_many_samples = _submission()
        too_many_samples["episodes"][0]["trace"]["timestamps_s"] = [float(index) for index in range(MAX_TRACE_SAMPLES + 1)]
        issues = validate_submission(too_many_samples)
        self.assertIn("resource_budget", {issue.code for issue in issues})

    def test_duplicate_episode_and_metric_trace_are_fail_closed(self) -> None:
        payload = _submission()
        payload["episodes"].append(dict(payload["episodes"][0]))
        report = evaluate_submission(payload)
        self.assertFalse(report.valid)
        codes = {issue.code for issue in report.issues}
        self.assertIn("duplicate_episode", codes)

        payload = _submission()
        payload["episodes"][0]["trace"]["confirmed_counts"] = [0, 2, 1]
        report = evaluate_submission(payload)
        self.assertFalse(report.valid)
        codes = {issue.code for issue in report.issues}
        self.assertIn("metric", codes)

    def test_file_report_hashes_input_and_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission = root / "submission.json"
            submission.write_text(json.dumps(_submission()), encoding="utf-8")
            output = root / "report.json"
            report = evaluate_submission_file(submission, output=output)
            self.assertTrue(report.valid)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["submission_sha256"], report.submission_sha256)
            with self.assertRaises(EvaluatorSubmissionError):
                evaluate_submission_file(submission, output=submission)


if __name__ == "__main__":
    unittest.main()
