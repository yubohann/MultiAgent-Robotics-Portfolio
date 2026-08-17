from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.formal_dataset import rebuild_dataset_index
from rivermark_benchmark.release_readiness import audit_release_readiness


class ReleaseReadinessTests(unittest.TestCase):
    def test_empty_index_and_missing_manifests_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self.assertTrue(rebuild_dataset_index(root, write=True).valid)
            report = audit_release_readiness(
                root,
                root / "release-manifest.json",
                root / "supply-chain.json",
            )
            codes = {issue.code for issue in report.issues}
            self.assertFalse(report.valid)
            self.assertEqual(report.episode_count, 0)
            self.assertIn("minimum_episode_count", codes)
            self.assertIn("release_manifest", codes)
            self.assertIn("manifest_read", codes)

    def test_local_payload_hash_and_bindings_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            (root / "manifests").mkdir()
            (root / "manifests" / "dataset_index.json").write_text(
                '{"dataset_version":"0.1.0","episode_count":1,"episodes":[],"schema":"org.rivermark.benchmark.dataset-index.v1"}\n',
                encoding="utf-8",
            )
            payload = root / "validation" / "episode-001" / "state.bin"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"immutable state")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            release = {
                "release_id": "pilot-test-001",
                "dataset_version": "0.1.0",
                "source_revision": "a" * 40,
                "supply_chain_manifest_sha256": "c" * 64,
                "shards": [{
                    "shard_id": "episode-001-state",
                    "path": "validation/episode-001/state.bin",
                    "size_bytes": payload.stat().st_size,
                    "sha256": digest,
                }],
            }
            integrity = SimpleNamespace(issues=(), episode_count=1)
            supply = {"status": "valid", "manifest_sha256": "c" * 64, "release_id": "pilot-test-001", "issues": []}
            with patch("rivermark_benchmark.release_readiness.verify_dataset_integrity", return_value=integrity), \
                patch("rivermark_benchmark.release_readiness.load_release_manifest", return_value=release), \
                patch("rivermark_benchmark.release_readiness.verify_supply_chain_manifest", return_value=supply):
                report = audit_release_readiness(root, root / "release.json", root / "supply.json")
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.shard_count, 1)
            self.assertEqual(report.checks["release_bindings"], "passed")

            payload.write_bytes(b"tampered state")
            with patch("rivermark_benchmark.release_readiness.verify_dataset_integrity", return_value=integrity), \
                patch("rivermark_benchmark.release_readiness.load_release_manifest", return_value=release), \
                patch("rivermark_benchmark.release_readiness.verify_supply_chain_manifest", return_value=supply):
                tampered = audit_release_readiness(root, root / "release.json", root / "supply.json")
            self.assertFalse(tampered.valid)
            self.assertIn("local_payload_hash", {issue.code for issue in tampered.issues})

    def test_failure_ledger_is_verified_without_a_shard_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            (root / "manifests").mkdir(parents=True)
            (root / "manifests" / "dataset_index.json").write_text(
                '{"dataset_version":"0.1.0","episode_count":1,"episodes":[],"schema":"org.rivermark.benchmark.dataset-index.v1"}\n',
                encoding="utf-8",
            )
            ledger = root / "manifests" / "failure_ledger.jsonl"
            ledger.write_bytes(b'{"schema":"org.rivermark.benchmark.failure-ledger.v1"}\n')
            digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
            release = {
                "release_id": "pilot-test-001",
                "dataset_version": "0.1.0",
                "source_revision": "a" * 40,
                "supply_chain_manifest_sha256": "c" * 64,
                "shards": [{
                    "shard_id": "episode-001-state",
                    "path": "manifests/failure_ledger.jsonl",
                    "size_bytes": ledger.stat().st_size,
                    "sha256": digest,
                }],
                "accounting": {"failure_ledger": {
                    "path": "manifests/failure_ledger.jsonl",
                    "size_bytes": ledger.stat().st_size,
                    "sha256": digest,
                }},
            }
            integrity = SimpleNamespace(issues=(), episode_count=1)
            supply = {"status": "valid", "manifest_sha256": "c" * 64, "release_id": "pilot-test-001", "issues": []}
            with patch("rivermark_benchmark.release_readiness.verify_dataset_integrity", return_value=integrity), \
                patch("rivermark_benchmark.release_readiness.load_release_manifest", return_value=release), \
                patch("rivermark_benchmark.release_readiness.verify_supply_chain_manifest", return_value=supply):
                report = audit_release_readiness(root, root / "release.json", root / "supply.json")
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.shard_count, 2)


if __name__ == "__main__":
    unittest.main()
