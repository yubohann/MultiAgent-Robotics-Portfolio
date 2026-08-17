from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.clean_room_smoke import (  # noqa: E402
    CLEAN_ROOM_SMOKE_SCHEMA,
    run_clean_room_smoke,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


class CleanRoomSmokeTests(unittest.TestCase):
    def _clean_fixture_repo(self, root: Path) -> Path:
        repo = root / "source"
        shutil.copytree(SRC, repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.invalid")
        _git(repo, "config", "user.name", "Test Runner")
        _git(repo, "add", "src")
        _git(repo, "commit", "-m", "fixture")
        return repo

    def test_clean_clone_runs_public_cpu_entry_and_leaves_only_report(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            repo = self._clean_fixture_repo(Path(temporary))
            output = Path(temporary) / "report"
            result = run_clean_room_smoke(output, source_root=repo, timeout_s=60.0)
            self.assertEqual(result["schema"], CLEAN_ROOM_SMOKE_SCHEMA)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["source"]["revision"], result["clone"]["revision"])
            self.assertTrue(result["checks"]["cpu_researcher_smoke"])
            self.assertFalse(result["formal_benchmark_admission"])
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["clean_room_report.json"])
            persisted = json.loads((output / "clean_room_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("evaluator_truth_sha256", json.dumps(persisted))
            self.assertNotIn(str(repo), json.dumps(persisted))

    def test_dirty_source_fails_closed_without_cloning(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            repo = self._clean_fixture_repo(Path(temporary))
            (repo / "untracked.txt").write_text("dirty", encoding="utf-8")
            output = Path(temporary) / "report"
            result = run_clean_room_smoke(output, source_root=repo)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_code"], "source_worktree_dirty")
            self.assertFalse(result["checks"]["source_clean"])
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["clean_room_report.json"])


if __name__ == "__main__":
    unittest.main()
