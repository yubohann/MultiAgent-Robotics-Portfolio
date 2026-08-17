from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.cleanup_history import cleanup_completed_runs


class CleanupHistoryTests(unittest.TestCase):
    def test_terminal_receipts_require_explicit_opt_in_before_archival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = root / "isaac-failed"
            failed.mkdir()
            (failed / "capture_receipt.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            (failed / "payload.bin").write_bytes(b"x" * 32)
            captured = root / "isaac-captured"
            captured.mkdir()
            (captured / "capture_receipt.json").write_text(
                json.dumps({"status": "captured"}), encoding="utf-8"
            )
            (captured / "payload.bin").write_bytes(b"x" * 32)
            now = time.time_ns()
            old_ns = now - 3 * 24 * 3600 * 1_000_000_000
            for path in (*failed.iterdir(), *captured.iterdir()):
                import os

                os.utime(path, ns=(old_ns, old_ns))
            records = cleanup_completed_runs(
                root,
                min_age_hours=24,
                min_size_bytes=1,
                dry_run=True,
                now_ns=now,
            )
            self.assertEqual(records, ())
            self.assertTrue(failed.exists())
            self.assertTrue(captured.exists())

            records = cleanup_completed_runs(
                root,
                min_age_hours=24,
                min_size_bytes=1,
                dry_run=True,
                include_terminal_receipts=True,
                now_ns=now,
            )
            self.assertEqual({record.status for record in records}, {"captured", "failed"})
            self.assertTrue(all(record.action == "dry_run" for record in records))
            self.assertTrue((root / "cleanup_history.jsonl").is_file())

    def test_dry_run_selects_stale_crash_left_progress_without_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orphan = root / "isaac-interrupted"
            orphan.mkdir()
            (orphan / "capture_progress.json").write_text(
                json.dumps(
                    {
                        "schema": "org.rivermark.isaac-capture-progress.v1",
                        "stage": "sensors_constructed",
                    }
                ),
                encoding="utf-8",
            )
            (orphan / "payload.bin").write_bytes(b"x" * 32)
            now = time.time_ns()
            old_ns = now - 3 * 24 * 3600 * 1_000_000_000
            import os

            for path in orphan.iterdir():
                os.utime(path, ns=(old_ns, old_ns))
            records = cleanup_completed_runs(
                root,
                min_age_hours=24,
                min_size_bytes=1,
                dry_run=True,
                now_ns=now,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "orphaned")
            self.assertEqual(records[0].action, "dry_run")
            self.assertTrue(orphan.exists())

    def test_dry_run_selects_stale_start_marker_without_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orphan = root / "isaac-before-preflight"
            orphan.mkdir()
            (orphan / "capture_start.json").write_text(
                json.dumps(
                    {
                        "schema": "org.rivermark.isaac-capture-start.v1",
                        "attempt_id": "attempt-" + "a" * 32,
                        "started_wall_time_ns": 1,
                        "source_revision": "a" * 40,
                        "source_tree_sha256": "b" * 64,
                        "source_worktree_dirty": False,
                        "task_kind": "search3d",
                        "control_mode": "fixed_public_route",
                        "agent_count_requested": 8,
                    }
                ),
                encoding="utf-8",
            )
            (orphan / "payload.bin").write_bytes(b"x" * 32)
            now = time.time_ns()
            old_ns = now - 3 * 24 * 3600 * 1_000_000_000
            import os

            for path in orphan.iterdir():
                os.utime(path, ns=(old_ns, old_ns))
            records = cleanup_completed_runs(
                root,
                min_age_hours=24,
                min_size_bytes=1,
                dry_run=True,
                now_ns=now,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "orphaned")


if __name__ == "__main__":
    unittest.main()
