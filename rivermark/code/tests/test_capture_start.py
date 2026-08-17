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

from rivermark_benchmark.isaac_capture import _write_capture_start_marker  # noqa: E402


class CaptureStartMarkerTests(unittest.TestCase):
    def test_marker_is_path_free_and_deterministically_recoverable(self) -> None:
        receipt = {
            "created_wall_time_ns": 1721779200000000000,
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "source_worktree_dirty": False,
            "task_kind": "search3d",
            "command": {"control_mode": "fixed_public_route"},
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            root.mkdir()
            attempt_id = _write_capture_start_marker(root, receipt)
            payload = json.loads((root / "capture_start.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["attempt_id"], attempt_id)
            self.assertEqual(payload["schema"], "org.rivermark.isaac-capture-start.v1")
            self.assertNotIn(str(root), json.dumps(payload))
            self.assertNotIn("private", json.dumps(payload).lower())
            self.assertEqual(payload["collection_binding"], receipt["collection_binding"])


if __name__ == "__main__":
    unittest.main()
