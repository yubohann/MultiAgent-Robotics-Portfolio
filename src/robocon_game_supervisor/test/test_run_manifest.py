import json
import tempfile
import unittest
from pathlib import Path

from robocon_game_supervisor.run_manifest import build_manifest, write_manifest


class RunManifestTests(unittest.TestCase):
    def test_missing_measurements_remain_unknown(self):
        manifest = build_manifest(
            run_id="test-run",
            evidence_level="synthetic",
            source_specs=["workspace=C:/does-not-exist"],
        )
        self.assertEqual(manifest["evidence_level"], "synthetic")
        self.assertEqual(manifest["metrics"]["pose_rate_hz"], "unknown")
        self.assertEqual(manifest["hardware"]["target_computer_product"], "TBD")
        self.assertEqual(manifest["software"]["sources"]["workspace"]["git_revision"], "unknown")

    def test_invalid_evidence_level_is_rejected(self):
        with self.assertRaises(ValueError):
            build_manifest(run_id="test-run", evidence_level="measured")

    def test_gazebo_simulation_is_an_explicit_evidence_level(self):
        manifest = build_manifest(run_id="gazebo-run", evidence_level="gazebo_simulation")
        self.assertEqual(manifest["evidence_level"], "gazebo_simulation")
        self.assertEqual(manifest["metrics"]["position_error_m"], "unknown")

    def test_explicit_vendor_revision_overrides_missing_git_directory(self):
        manifest = build_manifest(
            run_id="test-run",
            evidence_level="synthetic",
            source_specs=["fast_lio=C:/vendored/fast_lio"],
            source_revision_specs=["fast_lio=c8c20962"],
        )
        self.assertEqual(manifest["software"]["sources"]["fast_lio"]["git_revision"], "c8c20962")

    def test_revision_without_source_is_rejected(self):
        with self.assertRaises(ValueError):
            build_manifest(
                run_id="test-run",
                evidence_level="synthetic",
                source_revision_specs=["fast_lio=c8c20962"],
            )

    def test_map_hash_and_json_output_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            map_path = Path(directory) / "map.pcd"
            map_path.write_bytes(b"synthetic map")
            output = Path(directory) / "run.json"
            manifest = build_manifest(
                run_id="test-run",
                evidence_level="bag_replay",
                map_path=map_path,
                started_at_utc="2026-08-19T00:00:00Z",
            )
            write_manifest(output, manifest)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["map"]["sha256"], manifest["map"]["sha256"])
            self.assertEqual(loaded["run_id"], "test-run")


if __name__ == "__main__":
    unittest.main()
