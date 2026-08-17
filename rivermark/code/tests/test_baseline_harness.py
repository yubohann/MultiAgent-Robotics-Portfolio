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

from rivermark_benchmark.baseline_harness import (  # noqa: E402
    BASELINE_SUITE_SCHEMA,
    BaselineConfigError,
    run_baseline_suite,
    validate_baseline_config,
    verify_baseline_report,
)


def _config() -> dict:
    return {
        "schema": BASELINE_SUITE_SCHEMA,
        "suite_id": "test_suite",
        "backend": "rivermark-kinematic-pilot-v1",
        "formal_benchmark_admission": False,
        "agent_count": 2,
        "runtime": {"dt_s": 0.2, "world_size_xy_m": [32.0, 24.0], "camera_width": 32, "camera_height": 24},
        "train": {"enabled": False, "seed": 1, "episodes": 0},
        "tune": {"enabled": False, "seed": 2, "max_trials": 0},
        "evaluate": {"seeds": [7], "episodes_per_seed": 1},
        "methods": [
            {"method_id": "astar_mpc_pilot", "family": "classical", "information_profile": "geometry_state"},
            {"method_id": "actor_critic_rl_pilot", "family": "rl", "information_profile": "state_only"},
        ],
        "budget": {"max_steps": 4, "timeout_s": 30.0, "max_failures": 0},
    }


class BaselineHarnessTests(unittest.TestCase):
    def test_config_rejects_external_method_and_inconsistent_family(self) -> None:
        config = _config()
        config["methods"][0]["method_id"] = "openvla_checkpoint"
        with self.assertRaises(BaselineConfigError):
            validate_baseline_config(config)
        config = _config()
        config["methods"][0]["family"] = "rl"
        with self.assertRaises(BaselineConfigError):
            validate_baseline_config(config)

    def test_suite_runs_real_public_pilot_rollouts_without_private_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            report_path = root / "report.json"
            config_path.write_text(json.dumps(_config()), encoding="utf-8")
            report = run_baseline_suite(config_path, report_path)
            self.assertEqual(report["attempt_count"], 2)
            self.assertEqual(report["passed_count"], 2)
            self.assertEqual(report["failed_count"], 0)
            self.assertFalse(report["formal_benchmark_admission"])
            self.assertTrue(all("evaluator_truth_sha256" not in row.get("metrics", {}) for row in report["attempts"]))
            self.assertTrue(all(row["metrics"]["private_truth_digest_emitted"] is False for row in report["attempts"]))
            self.assertEqual(verify_baseline_report(report_path)["config_sha256"], report["config_sha256"])
            serialized = report_path.read_text(encoding="utf-8")
            self.assertNotIn("evaluator_truth_sha256", serialized)
            self.assertEqual(report_path.read_text(encoding="utf-8"), report_path.read_text(encoding="utf-8"))

    def test_existing_report_is_preserved_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            report_path = root / "report.json"
            config_path.write_text(json.dumps(_config()), encoding="utf-8")
            report_path.write_text("keep\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_baseline_suite(config_path, report_path)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "keep\n")

    def test_failure_budget_writes_partial_report_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config()
            config["methods"] = [config["methods"][0]]
            config["budget"] = {"max_steps": 1, "timeout_s": 1e-12, "max_failures": 0}
            config_path = root / "config.json"
            report_path = root / "report.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report = run_baseline_suite(config_path, report_path)
            self.assertEqual(report["status"], "stopped_failure_budget")
            self.assertEqual(report["attempt_count"], 1)
            self.assertEqual(report["failed_count"], 1)
            self.assertTrue(report_path.is_file())

    def test_report_verifier_rejects_config_hash_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            report_path = root / "report.json"
            config_path.write_text(json.dumps(_config()), encoding="utf-8")
            run_baseline_suite(config_path, report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["config_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_baseline_report(report_path)


if __name__ == "__main__":
    unittest.main()
