from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.frame_archive import write_chunked_frame_archive
from rivermark_benchmark.isaac_dataset import IsaacCapture


class IsaacDatasetLazyAccessTests(unittest.TestCase):
    def test_chunked_capture_reads_selected_frames_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sensors = root / "sensors"
            sensors.mkdir()
            timestamps = np.asarray([10, 20, 30, 40], dtype=np.int64)
            rgb = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
            depth = np.ones((4, 2, 3, 1), dtype=np.float32)
            write_chunked_frame_archive(
                sensors / "overview_rgb.npz",
                timestamps_ns=timestamps,
                inline_fields={"camera_pos_w_m": np.zeros((4, 3), dtype=np.float64)},
                frame_fields={"rgb": rgb, "distance_to_image_plane_m": depth},
            )
            (root / "capture_receipt.json").write_text(
                json.dumps({"status": "captured", "ok": True}), encoding="utf-8"
            )
            (root / "independent_validation.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            capture = IsaacCapture(root)
            self.assertEqual(capture.frame_count, 4)
            self.assertEqual(capture.modalities, ("overview",))
            rows = list(
                capture.iter_frames(
                    "overview", fields=("rgb",), start=1, stop=4, stride=2
                )
            )
            self.assertEqual([row.index for row in rows], [1, 3])
            self.assertEqual([row.timestamp_ns for row in rows], [20, 40])
            np.testing.assert_array_equal(rows[0].values["rgb"], rgb[1])
            np.testing.assert_array_equal(capture.read_frame("overview", 2, fields=("rgb",)).values["rgb"], rgb[2])

    def test_failed_capture_is_not_usable_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "capture_receipt.json").write_text(
                json.dumps({"status": "failed", "ok": False}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                IsaacCapture(root)


if __name__ == "__main__":
    unittest.main()
