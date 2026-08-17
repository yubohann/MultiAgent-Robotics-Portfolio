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

from rivermark_benchmark.open_aoe import (
    OPEN_AOE_EXTERNAL_PROVENANCE_SCHEMA,
    OpenAoeError,
    inspect_open_aoe_segment,
    main,
    scan_open_aoe_root,
    write_open_aoe_manifest,
)


def _camera_info(*, undistorted: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "deviceInfo": {"brand": "fixture", "model": "fixture-phone"},
        "cameraParams": {
            "resolution": "16x9",
            "fx_pixels": 10.0,
            "fy_pixels": 11.0,
            "cx_pixels": 8.0,
            "cy_pixels": 4.5,
            "lensDistortion": "[0.0, 0.0, 0.0, 0.0, 0.0]",
        },
    }
    if undistorted:
        payload["video_filename"] = "raw_video_undistorted.mp4"
        payload["fps"] = 30
    return payload


def _segment(root: Path, *, frame_count: int = 4) -> Path:
    segment = root / "dataset" / "segment-001"
    hands_root = segment / "ego_process" / "ego_hands_reconstruction"
    undistorted_root = segment / "ego_process" / "ego_undistorted_video"
    annotation_root = segment / "ego_annotation"
    hands_root.mkdir(parents=True)
    undistorted_root.mkdir(parents=True)
    annotation_root.mkdir()
    (segment / "raw_video.mp4").write_bytes(b"raw fixture video")
    (segment / "video_info.json").write_text(json.dumps(_camera_info(undistorted=False)), encoding="utf-8")
    (undistorted_root / "raw_video_undistorted.mp4").write_bytes(b"undistorted fixture video")
    (undistorted_root / "undistorted_video_info.json").write_text(
        json.dumps(_camera_info(undistorted=True)), encoding="utf-8"
    )
    annotations = [
        {
            "id": 1,
            "start_frame": "0",
            "end_frame": "2",
            "atomic_action": [
                {"verb": "grasp", "object": "cup", "hand": "right", "description": "Grasp the cup."}
            ],
        },
        {
            "id": 2,
            "start_frame": "2",
            "end_frame": str(frame_count),
            "atomic_action": [
                {"verb": "place", "object": "cup", "hand": "right", "description": "Place the cup."}
            ],
        },
    ]
    (annotation_root / "ego_action_annotation.json").write_text(json.dumps(annotations), encoding="utf-8")
    identity = np.tile(np.eye(3, dtype=np.float32), (frame_count, 1, 1))
    zeros_3 = np.zeros((frame_count, 3), dtype=np.float32)
    hands = {
        "R_w2c": identity,
        "t_w2c": zeros_3,
        "R_c2w": identity,
        "t_c2w": zeros_3,
        "pred_trans": np.zeros((2, frame_count, 3), dtype=np.float32),
        "pred_rot": np.zeros((2, frame_count, 3), dtype=np.float32),
        "pred_trans_cam": np.zeros((2, frame_count, 3), dtype=np.float32),
        "pred_rot_cam": np.zeros((2, frame_count, 3), dtype=np.float32),
        "pred_hand_pose": np.zeros((2, frame_count, 45), dtype=np.float32),
        "pred_betas": np.zeros((2, frame_count, 10), dtype=np.float32),
        "pred_valid": np.ones((2, frame_count), dtype=np.float32),
        "focal": np.asarray(10.0, dtype=np.float64),
    }
    np.savez(hands_root / "hands.npz", **hands)
    cam_c2w = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
    intrinsic = np.asarray([[10.0, 0.0, 8.0], [0.0, 11.0, 4.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savez(hands_root / "camera_traj.npz", cam_c2w=cam_c2w, intrinsic=intrinsic)
    return segment


class OpenAoeTests(unittest.TestCase):
    def test_valid_segment_produces_path_free_external_pretraining_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = _segment(root)
            report = inspect_open_aoe_segment(segment)
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.frame_count, 4)
            self.assertEqual(report.annotation_segment_count, 2)
            manifest = scan_open_aoe_root(root / "dataset", repository_root=root / "rivermark")
            self.assertEqual(manifest["schema"], OPEN_AOE_EXTERNAL_PROVENANCE_SCHEMA)
            self.assertEqual(manifest["purpose"], "external_pretraining_only")
            self.assertFalse(manifest["formal_rivermark_admission"])
            self.assertFalse(manifest["isaac_execution_evidence"])
            self.assertEqual(manifest["valid_segment_count"], 1)
            encoded = json.dumps(manifest, sort_keys=True)
            self.assertNotIn(str(root), encoded)
            output = write_open_aoe_manifest(root / "open-aoe-provenance.json", manifest)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["manifest_sha256"], manifest["manifest_sha256"])

    def test_intrinsic_mismatch_and_annotation_gap_are_retained_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = _segment(root)
            info_path = segment / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["cameraParams"]["fx_pixels"] = 999.0
            info_path.write_text(json.dumps(info), encoding="utf-8")
            annotation_path = segment / "ego_annotation" / "ego_action_annotation.json"
            annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotations[1]["start_frame"] = "3"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
            report = inspect_open_aoe_segment(segment)
            self.assertFalse(report.valid)
            codes = {issue.code for issue in report.issues}
            self.assertIn("intrinsic_mismatch", codes)
            self.assertIn("annotation_coverage", codes)
            with self.assertRaisesRegex(OpenAoeError, "no Open-AoE segment passed"):
                scan_open_aoe_root(root / "dataset", repository_root=root / "rivermark")

    def test_payload_inside_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "rivermark"
            _segment(repository)
            with self.assertRaisesRegex(OpenAoeError, "outside the Rivermark repository"):
                scan_open_aoe_root(repository / "dataset", repository_root=repository)

    def test_cli_writes_manifest_and_missing_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _segment(root)
            output = root / "output.json"
            self.assertEqual(
                main(
                    [
                        "--open-aoe-root",
                        str(root / "dataset"),
                        "--repository-root",
                        str(root / "rivermark"),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(
                main(
                    [
                        "--open-aoe-root",
                        str(root / "missing"),
                        "--repository-root",
                        str(root / "rivermark"),
                        "--output",
                        str(root / "other.json"),
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
