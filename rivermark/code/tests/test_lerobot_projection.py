from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_parquet_projection import _capture_fixture

from rivermark_benchmark.lerobot_projection import (
    LEROBOT_FORMAT_VERSION,
    LeRobotProjectionError,
    project_development_parquet_to_lerobot,
    verify_lerobot_projection,
    verify_with_upstream_lerobot,
)
from rivermark_benchmark.parquet_projection import (
    project_development_capture_to_parquet,
)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional parquet dependency is not installed")
class LeRobotProjectionTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        capture = _capture_fixture(root / "capture")
        source = root / "parquet"
        project_development_capture_to_parquet(capture, source, row_group_size=4)
        return source

    def test_projection_preserves_fleet_grouping_and_native_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            output = root / "lerobot"
            result = project_development_parquet_to_lerobot(source, output)

            self.assertEqual(result.agent_episode_count, 2)
            self.assertEqual(result.frame_count, 6)
            self.assertEqual(result.fps, 10)
            report = verify_lerobot_projection(output, source_root=source)
            self.assertEqual(report["status"], "valid")
            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], LEROBOT_FORMAT_VERSION)
            self.assertEqual(info["total_episodes"], 2)
            self.assertIsNone(info["video_path"])
            serialized = json.dumps(
                json.loads((output / "meta" / "rivermark_group_manifest.json").read_text(encoding="utf-8")),
                sort_keys=True,
            )
            self.assertNotIn(str(source), serialized)
            self.assertNotIn("private_evaluator", serialized)
            self.assertIn("development-only", serialized)

    def test_source_and_output_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            with (source / "state_action.parquet").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(LeRobotProjectionError, "manifest binding"):
                project_development_parquet_to_lerobot(source, root / "rejected")
            self.assertFalse((root / "rejected").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            output = root / "lerobot"
            project_development_parquet_to_lerobot(source, output)
            info_path = output / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["total_frames"] = 999
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(LeRobotProjectionError, "manifest binding"):
                verify_lerobot_projection(output)

    def test_wrong_fps_and_nested_output_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            with self.assertRaisesRegex(LeRobotProjectionError, "do not match 20 fps"):
                project_development_parquet_to_lerobot(source, root / "wrong-fps", fps=20)
            with self.assertRaisesRegex(LeRobotProjectionError, "must not be inside"):
                project_development_parquet_to_lerobot(source, source / "nested")

    def test_upstream_reader_requires_the_reviewed_source_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            output = root / "lerobot"
            project_development_parquet_to_lerobot(source, output)
            with self.assertRaisesRegex(LeRobotProjectionError, "upstream_source must contain"):
                verify_with_upstream_lerobot(output, root / "not-lerobot")


if __name__ == "__main__":
    unittest.main()
