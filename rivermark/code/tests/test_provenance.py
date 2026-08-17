from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.provenance import detect_source_provenance, require_clean_source


class SourceProvenanceTests(unittest.TestCase):
    def _repository(self, root: Path) -> str:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "test@rivermark.invalid"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "Rivermark Test"], check=True)
        (root / "tracked.txt").write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True)
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()

    def test_clean_repository_records_commit_and_tree_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = self._repository(root)
            first = require_clean_source(root)
            second = detect_source_provenance(root)
            self.assertEqual(first.source_revision, revision)
            self.assertEqual(first, second)
            self.assertEqual(len(first.source_tree_sha256), 64)

    def test_tracked_or_untracked_change_marks_repository_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repository(root)
            (root / "untracked.txt").write_text("local\n", encoding="utf-8")
            provenance = detect_source_provenance(root)
            self.assertTrue(provenance.source_worktree_dirty)
            with self.assertRaisesRegex(RuntimeError, "clean Git worktree"):
                require_clean_source(root)


if __name__ == "__main__":
    unittest.main()
