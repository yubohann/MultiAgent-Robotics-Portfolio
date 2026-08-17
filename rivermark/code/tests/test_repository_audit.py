from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.repository_audit import (
    audit_git_history,
    audit_repository,
    audit_source_distribution,
)


class RepositoryAuditTests(unittest.TestCase):
    def _repo(self, root: Path, relative: str, payload: bytes) -> None:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "audit@rivermark.invalid"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "Rivermark Audit"], check=True)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        subprocess.run(["git", "-C", str(root), "add", "--", relative], check=True)

    def test_clean_source_files_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repo(root, "src/example.py", b"print('ok')\n")
            report = audit_repository(root, max_file_bytes=1024)
            self.assertEqual(report.status, "passed")

    def test_forbidden_asset_path_and_extension_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repo(root, "evidence/capture.mp4", b"video")
            codes = {issue.code for issue in audit_repository(root, max_file_bytes=1024).issues}
            self.assertEqual(codes, {"forbidden_path", "forbidden_extension"})

    def test_operator_prompt_is_not_a_release_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repo(root, "docs/isaac_execution_prompt.md", b"operator instructions")
            self.assertEqual(audit_repository(root, max_file_bytes=1024).issues[0].code, "operator_prompt")

    def test_oversized_tracked_file_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repo(root, "docs/notes.md", b"x" * 11)
            report = audit_repository(root, max_file_bytes=10)
            self.assertEqual(report.issues[0].code, "oversized_file")

    def test_missing_index_file_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repo(root, "src/example.py", b"print('ok')\n")
            (root / "src/example.py").unlink()
            self.assertEqual(audit_repository(root, max_file_bytes=1024).issues[0].code, "missing_tracked_file")

    def test_history_audit_finds_deleted_forbidden_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repo(root, "evidence/withdrawn.mp4", b"historical video")
            subprocess.run(["git", "-C", str(root), "config", "user.email", "audit@rivermark.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Rivermark Audit"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "add historical artifact"], check=True)
            (root / "evidence/withdrawn.mp4").unlink()
            (root / "README.md").write_text("kept source\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "withdraw artifact"], check=True)

            current = audit_repository(root, max_file_bytes=1024)
            self.assertEqual(current.status, "passed")
            history = audit_git_history(root, max_file_bytes=1024)
            self.assertEqual(history.status, "blocked")
            self.assertIn("historical_forbidden_path", {issue.code for issue in history.issues})
            self.assertIn("historical_forbidden_extension", {issue.code for issue in history.issues})
            self.assertTrue(any(issue.path == "evidence/withdrawn.mp4" for issue in history.issues))
            self.assertGreaterEqual(history.reachable_entry_count, history.reachable_blob_count)
            self.assertEqual(history.as_dict()["reachable_entry_count"], history.reachable_entry_count)
            self.assertEqual(history.as_dict()["reachable_blob_count"], history.reachable_blob_count)

    def test_source_distribution_audit_rejects_ignored_evidence_and_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "package.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                for name, payload in (
                    ("package-1.0/src/example.py", b"print('ok')\n"),
                    ("package-1.0/evidence/old.mp4", b"video"),
                    ("package-1.0/docs/isaac_execution_prompt.md", b"operator instructions"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target.addfile(info, BytesIO(payload))
            report = audit_source_distribution(archive, max_file_bytes=1024)
            self.assertEqual(report.status, "blocked")
            self.assertIn("forbidden_path", {issue.code for issue in report.issues})
            self.assertIn("operator_prompt", {issue.code for issue in report.issues})

    def test_source_distribution_audit_rejects_unsafe_member_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "package.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                info = tarfile.TarInfo("../outside.txt")
                info.size = 1
                target.addfile(info, BytesIO(b"x"))
            report = audit_source_distribution(archive, max_file_bytes=1024)
            self.assertEqual(report.status, "blocked")
            self.assertEqual(report.issues[0].code, "unsafe_archive_path")


if __name__ == "__main__":
    unittest.main()
