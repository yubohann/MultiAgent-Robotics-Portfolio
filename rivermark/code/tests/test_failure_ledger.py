from __future__ import annotations

import sys
import tempfile
import time
import unittest
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.failure_ledger import (
    FailureLedgerError,
    FailureRecord,
    append_failure_record,
    append_failure_record_once,
    load_failure_ledger,
    recover_crash_left_attempts,
    summarize_failure_ledger,
    validate_failure_record,
)


class FailureLedgerTests(unittest.TestCase):
    def test_append_and_summarize_preserves_attempt_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "failure_ledger.jsonl"
            append_failure_record(
                ledger,
                FailureRecord("attempt-001", "admitted", "none", "admission", "2026-07-24T00:00:00Z", split="train"),
            )
            append_failure_record(
                ledger,
                FailureRecord("attempt-002", "quarantined", "sensor_failure", "capture", "2026-07-24T00:01:00Z", split="train", reason_code="missing_frame"),
            )
            summary = summarize_failure_ledger(ledger)
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["admitted_count"], 1)
            self.assertEqual(summary["quarantined_count"], 1)
            self.assertEqual(summary["failure_categories"], {"sensor_failure": 1})
            self.assertEqual(len(load_failure_ledger(ledger)), 2)

    def test_private_text_and_duplicate_attempts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "failure_ledger.jsonl"
            private = FailureRecord("attempt-001", "failed", "infrastructure_failure", "capture", "2026-07-24T00:00:00Z", reason_code="private manifest path")
            with self.assertRaises(FailureLedgerError):
                append_failure_record(ledger, private)
            public = FailureRecord("attempt-001", "failed", "infrastructure_failure", "capture", "2026-07-24T00:00:00Z", reason_code="host_process_exit")
            append_failure_record(ledger, public)
            append_failure_record(ledger, public)
            with self.assertRaises(FailureLedgerError):
                summarize_failure_ledger(ledger)

    def test_terminal_append_once_is_idempotent_and_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "failure_ledger.jsonl"
            original = FailureRecord(
                "attempt-001",
                "failed",
                "capture_failure",
                "isaac_capture",
                "2026-07-25T00:00:00Z",
                split="train",
                receipt_sha256="a" * 64,
                source_capture_sha256="a" * 64,
                reason_code="runtimeerror",
            )
            self.assertEqual(append_failure_record_once(ledger, original), "appended")
            retried = FailureRecord(
                "attempt-001",
                "failed",
                "capture_failure",
                "isaac_capture",
                "2026-07-25T00:01:00Z",
                split="train",
                receipt_sha256="a" * 64,
                source_capture_sha256="a" * 64,
                reason_code="runtimeerror",
            )
            self.assertEqual(append_failure_record_once(ledger, retried), "already_recorded")
            self.assertEqual(summarize_failure_ledger(ledger)["attempt_count"], 1)
            conflicting = FailureRecord(
                "attempt-001",
                "failed",
                "sensor_failure",
                "isaac_capture",
                "2026-07-25T00:02:00Z",
                split="train",
                receipt_sha256="a" * 64,
                source_capture_sha256="a" * 64,
                reason_code="runtimeerror",
            )
            with self.assertRaisesRegex(FailureLedgerError, "conflicting terminal record"):
                append_failure_record_once(ledger, conflicting)

    def test_admitted_record_cannot_carry_failure_category(self) -> None:
        record = FailureRecord("attempt-001", "admitted", "sensor_failure", "admission", "2026-07-24T00:00:00Z")
        self.assertIn("outcome_category", {issue.code for issue in validate_failure_record(record.as_dict())})

    def test_timestamp_and_split_are_explicitly_validated(self) -> None:
        record = FailureRecord("attempt-001", "failed", "task_failure", "capture", "2026-07-24T00:00:00")
        codes = {issue.code for issue in validate_failure_record(record.as_dict())}
        self.assertIn("recorded_at", codes)
        record = FailureRecord("attempt-001", "failed", "task_failure", "capture", "2026-07-24T00:00:00Z", split="unknown")
        self.assertIn("split", {issue.code for issue in validate_failure_record(record.as_dict())})

    def test_collection_binding_is_complete_and_typed(self) -> None:
        partial = FailureRecord(
            "attempt-001",
            "failed",
            "quality_failure",
            "capture",
            "2026-07-24T00:00:00Z",
            collection_protocol_id="citylite-coverage-v1",
        )
        self.assertIn("collection_binding", {issue.code for issue in validate_failure_record(partial.as_dict())})
        bound = FailureRecord(
            "attempt-002",
            "failed",
            "quality_failure",
            "capture",
            "2026-07-24T00:00:00Z",
            collection_protocol_id="citylite-coverage-v1",
            collection_protocol_sha256="a" * 64,
            collection_cell_id="validation-route-2",
            collection_episode_index=0,
            episode_seed=42,
        )
        self.assertEqual(validate_failure_record(bound.as_dict()), ())
        invalid_seed = {**bound.as_dict(), "episode_seed": 2**32}
        self.assertIn("episode_seed", {issue.code for issue in validate_failure_record(invalid_seed)})
        malformed_key = {**bound.as_dict(), 7: "unexpected"}
        self.assertIn("unknown_field", {issue.code for issue in validate_failure_record(malformed_key)})

    def test_crash_left_recovery_records_stale_start_marker_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rivermark-runs"
            capture = root / "crash-left"
            capture.mkdir(parents=True)
            started = time.time_ns() - 48 * 3600 * 1_000_000_000
            marker = {
                "schema": "org.rivermark.isaac-capture-start.v1",
                "attempt_id": "attempt-" + "a" * 32,
                "started_wall_time_ns": started,
                "source_revision": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_worktree_dirty": False,
                "task_kind": "search3d",
                "control_mode": "fixed_public_route",
                "agent_count_requested": 8,
                "collection_binding": {
                    "protocol_id": "citylite-coverage-v1",
                    "protocol_sha256": "c" * 64,
                    "cell_id": "train-route-0",
                    "split": "train",
                    "episode_index": 3,
                    "episode_seed": 42,
                },
            }
            (capture / "capture_start.json").write_text(json.dumps(marker), encoding="utf-8")
            (capture / "capture_progress.json").write_text(
                json.dumps({"schema": "org.rivermark.isaac-capture-progress.v1", "stage": "app_launched"}),
                encoding="utf-8",
            )
            now = time.time_ns()
            old_seconds = (now - started) / 1_000_000_000
            for item in (capture / "capture_start.json", capture / "capture_progress.json"):
                item.touch()
                item_mtime = time.time() - old_seconds
                os.utime(item, (item_mtime, item_mtime))
            records = recover_crash_left_attempts(root, now_ns=now, min_age_hours=24)
            self.assertEqual([record.status for record in records], ["recovered"])
            summary = summarize_failure_ledger(root / "failure_ledger.jsonl")
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["failure_categories"], {"infrastructure_failure": 1})
            recovered_record = load_failure_ledger(root / "failure_ledger.jsonl")[0]
            self.assertEqual(recovered_record["collection_cell_id"], "train-route-0")
            self.assertEqual(recovered_record["collection_episode_index"], 3)
            self.assertEqual(recovered_record["episode_seed"], 42)
            self.assertEqual(recovered_record["split"], "train")
            again = recover_crash_left_attempts(root, now_ns=now, min_age_hours=24)
            self.assertEqual([record.status for record in again], ["already_recorded"])
            self.assertEqual(summarize_failure_ledger(root / "failure_ledger.jsonl")["attempt_count"], 1)

    def test_crash_left_recovery_protects_recent_and_terminal_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rivermark-runs"
            recent = root / "recent"
            terminal = root / "terminal"
            recent.mkdir(parents=True)
            terminal.mkdir(parents=True)
            base = {
                "schema": "org.rivermark.isaac-capture-start.v1",
                "started_wall_time_ns": time.time_ns(),
                "source_revision": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_worktree_dirty": False,
                "task_kind": "search3d",
                "control_mode": "fixed_public_route",
                "agent_count_requested": 8,
            }
            for directory, suffix in ((recent, "c"), (terminal, "d")):
                marker = {**base, "attempt_id": "attempt-" + suffix * 32}
                (directory / "capture_start.json").write_text(json.dumps(marker), encoding="utf-8")
            (terminal / "capture_receipt.json").write_text(json.dumps({"status": "captured"}), encoding="utf-8")
            records = recover_crash_left_attempts(root, min_age_hours=24)
            statuses = {record.attempt_id: record.status for record in records}
            self.assertEqual(statuses["attempt-" + "c" * 32], "protected_recent")
            self.assertEqual(statuses["attempt-" + "d" * 32], "recovered_terminal")
            self.assertEqual(
                summarize_failure_ledger(root / "failure_ledger.jsonl")["quarantined_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
