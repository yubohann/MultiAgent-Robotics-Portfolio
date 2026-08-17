from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.formal_dataset import rebuild_dataset_index
from rivermark_benchmark.release_gate import audit_release_dataset, main


class ReleaseGateTests(unittest.TestCase):
    def test_structurally_valid_empty_dataset_is_not_releaseable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            report = rebuild_dataset_index(root, write=True)
            self.assertTrue(report.valid, report.issues)
            gate = audit_release_dataset(root)
            self.assertFalse(gate.valid)
            self.assertEqual(gate.episode_count, 0)
            self.assertIn("minimum_episode_count", {issue.code for issue in gate.issues})
            self.assertEqual(main([str(root)]), 1)

    def test_corrupt_index_fails_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            (root / "manifests").mkdir(parents=True)
            (root / "manifests" / "dataset_index.json").write_text("[]\n", encoding="utf-8")
            gate = audit_release_dataset(root)
            self.assertFalse(gate.valid)
            self.assertIn("dataset_index", {issue.code for issue in gate.issues})

    def test_minimum_episode_count_cannot_disable_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                audit_release_dataset(Path(temporary), minimum_episodes=0)


if __name__ == "__main__":
    unittest.main()
