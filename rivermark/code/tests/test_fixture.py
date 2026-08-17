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

from rivermark_benchmark.fixture import (  # noqa: E402
    FIXTURE_SCHEMA,
    FixtureError,
    create_cpu_fixture,
    verify_cpu_fixture,
)


class CpuFixtureTests(unittest.TestCase):
    def test_fixture_is_small_hash_bound_and_explicitly_non_formal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-fixture-") as temporary:
            result = create_cpu_fixture(Path(temporary) / "fixture", seed=7)
            payload = json.loads(result.fixture_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], FIXTURE_SCHEMA)
            self.assertTrue(payload["derived_sample"])
            self.assertFalse(payload["formal_benchmark_admission"])
            self.assertEqual(payload["claim_boundary"], "cpu_loader_smoke_only")
            self.assertEqual(payload["episode_manifest_sha256"], result.episode_manifest_sha256)
            self.assertEqual(result.frame_count, 5)
            self.assertEqual(result.agent_count, 2)
            verification = verify_cpu_fixture(result.fixture_manifest_path)
            self.assertTrue(verification.valid, verification.issues)

    def test_fixture_refuses_non_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-fixture-") as temporary:
            root = Path(temporary) / "fixture"
            root.mkdir()
            (root / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FixtureError):
                create_cpu_fixture(root)
            self.assertEqual((root / "existing.txt").read_text(encoding="utf-8"), "preserve")

    def test_verifier_rejects_hash_tampering_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-fixture-") as temporary:
            result = create_cpu_fixture(Path(temporary) / "fixture")
            payload = json.loads(result.fixture_manifest_path.read_text(encoding="utf-8"))
            payload["episode_manifest_sha256"] = "0" * 64
            result.fixture_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn("episode_manifest_sha256:mismatch", verify_cpu_fixture(result.fixture_manifest_path).issues)
            payload["episode_manifest"] = "../outside.json"
            result.fixture_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn("episode_manifest:unsafe", verify_cpu_fixture(result.fixture_manifest_path).issues)


if __name__ == "__main__":
    unittest.main()
