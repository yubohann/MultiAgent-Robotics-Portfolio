from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import cv2  # noqa: F401
    import imageio_ffmpeg  # noqa: F401
except ImportError:
    cv2 = None

from rivermark_benchmark.demo import Mp4Writer
from rivermark_benchmark.frame_archive import write_chunked_frame_archive
from rivermark_benchmark.video import (
    STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
    VideoAudit,
    audit_video,
    encode_isaac_composite,
    encode_isaac_overview,
    require_release_video,
    sha256_file,
)


def _audit(**changes) -> VideoAudit:
    value = VideoAudit(
        path="demo.mp4",
        sha256="0" * 64,
        bytes=4096,
        width=16,
        height=16,
        fps=30.0,
        frame_count=3,
        first_frame_nonconstant=True,
        ffmpeg_full_decode=True,
        h264_signature_present=True,
        pixel_format="yuv420p",
        faststart=True,
    )
    return replace(value, **changes)


@unittest.skipIf(cv2 is None, "OpenCV or imageio-ffmpeg is not installed")
class ReleaseVideoTests(unittest.TestCase):
    def _frames(self) -> np.ndarray:
        frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        for index in range(len(frames)):
            frames[index, :, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 8
            frames[index, :, :, 1] = index * 70
        return frames

    def _write_passing_validation(self, root: Path, **updates: object) -> Path:
        receipt_path = root / "capture_receipt.json"
        validation = {
            "schema": "org.rivermark.isaac-independent-validation.v1",
            "status": "passed",
            "issues": [],
            "capture_receipt_sha256": sha256_file(receipt_path),
        }
        validation.update(updates)
        destination = root / "independent_validation.json"
        destination.write_text(json.dumps(validation), encoding="utf-8")
        return destination

    def _write_state_only_transfer_validation(self, root: Path) -> Path:
        receipt_path = root / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update(
            {
                "task_kind": "state_only_control_transfer_smoke",
                "command": {"control_mode": "sb3_state_only_transfer"},
                "claim_boundary": {
                    "formal_benchmark_admission": False,
                    "development_control_transfer": True,
                },
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return self._write_passing_validation(
            root,
            schema=STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
            formal_benchmark_admission=False,
        )

    def _composite_capture(self, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sensors = root / "sensors"
        sensors.mkdir(parents=True)
        timestamps = np.array([0, 50_000_000, 100_000_000], dtype=np.int64)
        overview = np.zeros((3, 24, 32, 3), dtype=np.uint8)
        overview[..., 0] = np.arange(32, dtype=np.uint8)[None, None, :] * 6
        overview[..., 1] = 80
        onboard = np.zeros((3, 8, 20, 28, 3), dtype=np.uint8)
        palette = np.array(
            [
                [220, 30, 30],
                [30, 220, 30],
                [30, 30, 220],
                [220, 220, 30],
                [220, 30, 220],
                [30, 220, 220],
                [180, 100, 30],
                [100, 30, 180],
            ],
            dtype=np.uint8,
        )
        for agent, color in enumerate(palette):
            onboard[:, agent] = color
            onboard[:, agent, :, agent + 2 : agent + 4] = 255 - color
        overview_path = sensors / "overview_rgb.npz"
        onboard_path = sensors / "onboard_rgbd.npz"
        np.savez_compressed(overview_path, timestamps_ns=timestamps, rgb=overview)
        np.savez_compressed(onboard_path, timestamps_ns=timestamps, rgb=onboard)
        receipt = {
            "schema": "org.rivermark.isaac-swarm-capture.v1",
            "ok": True,
            "artifact_hashes": {
                "sensors/overview_rgb.npz": {
                    "sha256": sha256_file(overview_path),
                    "bytes": overview_path.stat().st_size,
                },
                "sensors/onboard_rgbd.npz": {
                    "sha256": sha256_file(onboard_path),
                    "bytes": onboard_path.stat().st_size,
                },
            },
        }
        (root / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        self._write_passing_validation(root)
        return timestamps, overview, onboard

    def test_release_gate_reports_every_portability_failure(self) -> None:
        invalid = _audit(
            h264_signature_present=False,
            pixel_format="yuv444p",
            faststart=False,
            codec_name="mpeg4",
            moov_before_mdat=False,
            opencv_full_decode=False,
            first_frame_nonconstant=False,
            frame_count=2,
        )
        with self.assertRaises(RuntimeError) as raised:
            require_release_video(invalid, expected_frames=3)
        message = str(raised.exception)
        for expected in ("mpeg4", "not H.264", "yuv444p", "moov", "OpenCV", "constant", "decoded 2 frames"):
            self.assertIn(expected, message)

    def test_mp4_writer_produces_decodable_h264_yuv420p_faststart_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "demo.mp4"
            writer = Mp4Writer(destination, fps=30, frame_size=(16, 16))
            for rgb in self._frames():
                writer.write(np.ascontiguousarray(rgb[..., ::-1]))
            writer.close()
            writer.close()
            audit = audit_video(destination)
            require_release_video(audit, expected_frames=3)
            self.assertEqual((audit.width, audit.height), (16, 16))
            self.assertGreater(audit.bytes, 1024)
            self.assertEqual(audit.codec_name, "h264")
            self.assertEqual(audit.pixel_format, "yuv420p")
            self.assertTrue(audit.moov_before_mdat)
            self.assertTrue(audit.opencv_full_decode)
            self.assertEqual(audit.ffmpeg_frame_count, 3)
            self.assertEqual(audit.opencv_reported_frame_count, 3)

    def test_isaac_overview_encoding_is_bound_to_capture_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            source = root / "sensors" / "overview_rgb.npz"
            source.parent.mkdir(parents=True)
            timestamps = np.array([0, 50_000_000, 100_000_000], dtype=np.int64)
            np.savez_compressed(source, timestamps_ns=timestamps, rgb=self._frames())
            receipt_path = root / "capture_receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema": "org.rivermark.isaac-swarm-capture.v1",
                        "ok": True,
                        "artifact_hashes": {"sensors/overview_rgb.npz": {"sha256": sha256_file(source)}},
                    }
                ),
                encoding="utf-8",
            )
            validation_path = self._write_passing_validation(root)
            destination = Path(temporary) / "isaac-demo.mp4"
            result = encode_isaac_overview(root, destination)
            self.assertEqual(result["capture_receipt_sha256"], sha256_file(receipt_path))
            self.assertEqual(result["independent_validation_sha256"], sha256_file(validation_path))
            self.assertEqual(result["overview_npz_sha256"], sha256_file(source))
            self.assertAlmostEqual(result["fps"], 20.0)
            self.assertEqual(result["audit"]["frame_count"], 3)
            self.assertTrue(destination.with_suffix(".mp4.receipt.json").is_file())

    def test_overview_tamper_and_invalid_fps_fail_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            source = root / "sensors" / "overview_rgb.npz"
            source.parent.mkdir(parents=True)
            np.savez_compressed(
                source,
                timestamps_ns=np.array([0, 1], dtype=np.int64),
                rgb=self._frames()[:2],
            )
            receipt = {
                "schema": "org.rivermark.isaac-swarm-capture.v1",
                "ok": True,
                "artifact_hashes": {"sensors/overview_rgb.npz": {"sha256": "0" * 64}},
            }
            (root / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(root)
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                encode_isaac_overview(root, Path(temporary) / "tampered.mp4")
            receipt["artifact_hashes"]["sensors/overview_rgb.npz"]["sha256"] = sha256_file(source)
            (root / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(root)
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                encode_isaac_overview(root, Path(temporary) / "bad-fps.mp4", fps=float("nan"))

    def test_overview_malformed_artifact_inventory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            source = root / "sensors" / "overview_rgb.npz"
            source.parent.mkdir(parents=True)
            np.savez_compressed(
                source,
                timestamps_ns=np.array([0, 1], dtype=np.int64),
                rgb=self._frames()[:2],
            )
            (root / "capture_receipt.json").write_text(
                json.dumps(
                    {
                        "schema": "org.rivermark.isaac-swarm-capture.v1",
                        "ok": True,
                        "artifact_hashes": [],
                    }
                ),
                encoding="utf-8",
            )
            self._write_passing_validation(root)
            with self.assertRaisesRegex(RuntimeError, "artifact inventory"):
                encode_isaac_overview(root, Path(temporary) / "malformed.mp4")

    def test_isaac_encoders_require_a_passing_bound_independent_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            self._composite_capture(root)
            validation_path = root / "independent_validation.json"
            validation_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "independent validation"):
                encode_isaac_overview(root, Path(temporary) / "missing-validation.mp4")

            for updates in (
                {"schema": "unknown"},
                {"schema": []},
                {"status": "failed"},
                {"issues": [{"code": "capture_integrity"}]},
            ):
                self._write_passing_validation(root, **updates)
                with self.assertRaisesRegex(RuntimeError, "passing independent validation"):
                    encode_isaac_overview(root, Path(temporary) / "invalid-validation.mp4")

            self._write_passing_validation(root, capture_receipt_sha256="0" * 64)
            with self.assertRaisesRegex(RuntimeError, "does not bind"):
                encode_isaac_composite(root, Path(temporary) / "wrong-capture-binding.mp4")

    def test_isaac_encoder_accepts_only_compatible_state_only_transfer_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            self._composite_capture(root)
            validation_path = self._write_state_only_transfer_validation(root)
            result = encode_isaac_overview(root, Path(temporary) / "transfer-overview.mp4")
            self.assertEqual(result["independent_validation_sha256"], sha256_file(validation_path))

            self._write_passing_validation(root)
            with self.assertRaisesRegex(RuntimeError, "state-only transfer capture requires"):
                encode_isaac_overview(root, Path(temporary) / "wrong-transfer-validator.mp4")

            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.update(
                {
                    "task_kind": "search3d",
                    "command": {"control_mode": "fixed_public_route"},
                    "claim_boundary": {"formal_benchmark_admission": False},
                }
            )
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(
                root,
                schema=STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
                formal_benchmark_admission=False,
            )
            with self.assertRaisesRegex(RuntimeError, "state-only transfer validation requires"):
                encode_isaac_overview(root, Path(temporary) / "wrong-fixed-route-validator.mp4")

    def test_isaac_composite_encodes_overview_and_eight_onboard_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            timestamps, _, onboard = self._composite_capture(root)
            destination = Path(temporary) / "isaac-composite.mp4"
            result = encode_isaac_composite(root, destination)

            self.assertEqual(result["schema"], "org.rivermark.isaac-swarm-composite-video.v1")
            self.assertTrue(result["ok"])
            self.assertEqual(result["layout"]["onboard_rows"], 4)
            self.assertEqual(result["layout"]["onboard_columns"], 2)
            self.assertEqual(len(result["layout"]["slots"]), 9)
            self.assertEqual(result["layout"]["slots"][0]["width"], 32)
            self.assertEqual(result["layout"]["slots"][0]["height"], 24)
            self.assertEqual(result["timestamps"]["count"], len(timestamps))
            self.assertEqual(len(set(result["timestamp_bindings"].values())), 1)
            self.assertEqual(result["video_sha256"], sha256_file(destination))
            self.assertEqual(
                result["independent_validation_sha256"],
                sha256_file(root / "independent_validation.json"),
            )
            self.assertEqual(result["audit"]["path"], destination.name)
            self.assertEqual(result["audit"]["codec_name"], "h264")
            self.assertEqual(result["audit"]["pixel_format"], "yuv420p")
            self.assertTrue(result["audit"]["moov_before_mdat"])
            self.assertTrue(result["audit"]["opencv_full_decode"])
            self.assertEqual(result["audit"]["frame_count"], len(timestamps))

            receipt_path = destination.with_suffix(".mp4.receipt.json")
            self.assertEqual(json.loads(receipt_path.read_text(encoding="utf-8")), result)
            capture = cv2.VideoCapture(str(destination))
            ok, decoded_bgr = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(decoded_bgr.shape[:2], (24, 48))
            decoded_rgb = decoded_bgr[..., ::-1]
            overview_height, overview_width = 24, 32
            cell_height, cell_width = 6, 8
            overview_nonblack = np.count_nonzero(
                np.max(decoded_rgb[:overview_height, :overview_width], axis=-1) > 8
            )
            onboard_nonblack = np.count_nonzero(
                np.max(decoded_rgb[:cell_height, overview_width : overview_width + cell_width], axis=-1) > 8
            )
            self.assertGreater(overview_nonblack, 2 * onboard_nonblack)
            for agent in range(8):
                row = agent // 2
                column = agent % 2
                sample = decoded_rgb[
                    row * cell_height + cell_height // 2,
                    overview_width + column * cell_width + cell_width // 2,
                ].astype(np.int16)
                expected = onboard[0, agent, 10, 14].astype(np.int16)
                self.assertLess(np.abs(sample - expected).max(), 45, f"agent {agent} view missing")

    def test_isaac_composite_streams_chunked_sensor_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            timestamps, overview, onboard = self._composite_capture(root)
            overview_path = root / "sensors" / "overview_rgb.npz"
            onboard_path = root / "sensors" / "onboard_rgbd.npz"
            write_chunked_frame_archive(
                overview_path,
                timestamps_ns=timestamps,
                inline_fields={},
                frame_fields={"rgb": overview},
            )
            write_chunked_frame_archive(
                onboard_path,
                timestamps_ns=timestamps,
                inline_fields={},
                frame_fields={"rgb": onboard},
            )
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for relative, path in (
                ("sensors/overview_rgb.npz", overview_path),
                ("sensors/onboard_rgbd.npz", onboard_path),
            ):
                receipt["artifact_hashes"][relative] = {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(root)
            result = encode_isaac_composite(root, Path(temporary) / "chunked.mp4")
            self.assertEqual(result["timestamps"]["count"], len(timestamps))
            self.assertEqual(result["audit"]["frame_count"], len(timestamps))

    def test_isaac_composite_maps_low_rate_overview_to_exact_onboard_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            timestamps, overview, onboard = self._composite_capture(root)
            overview_path = root / "sensors" / "overview_rgb.npz"
            selected = np.asarray([0, 2], dtype=np.int64)
            np.savez_compressed(
                overview_path,
                timestamps_ns=timestamps[selected],
                rgb=overview[selected],
            )
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifact_hashes"]["sensors/overview_rgb.npz"] = {
                "sha256": sha256_file(overview_path),
                "bytes": overview_path.stat().st_size,
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(root)

            result = encode_isaac_composite(root, Path(temporary) / "low-rate.mp4")
            self.assertEqual(result["timestamps"]["count"], len(selected))
            self.assertEqual(
                result["onboard_frame_mapping"]["entries"],
                [
                    {"overview_frame_index": 0, "onboard_frame_index": 0, "timestamp_ns": 0},
                    {
                        "overview_frame_index": 1,
                        "onboard_frame_index": 2,
                        "timestamp_ns": 100_000_000,
                    },
                ],
            )

    def test_isaac_composite_rejects_tamper_timestamp_mismatch_and_wrong_agent_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            timestamps, _, onboard = self._composite_capture(root)
            onboard_path = root / "sensors" / "onboard_rgbd.npz"
            destination = Path(temporary) / "invalid.mp4"

            with onboard_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                encode_isaac_composite(root, destination)
            self.assertFalse(destination.exists())

            np.savez_compressed(onboard_path, timestamps_ns=timestamps + 1, rgb=onboard)
            receipt = json.loads((root / "capture_receipt.json").read_text(encoding="utf-8"))
            receipt["artifact_hashes"]["sensors/onboard_rgbd.npz"] = {
                "sha256": sha256_file(onboard_path),
                "bytes": onboard_path.stat().st_size,
            }
            (root / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(root)
            with self.assertRaisesRegex(RuntimeError, "exact ordered subset"):
                encode_isaac_composite(root, destination)

            np.savez_compressed(onboard_path, timestamps_ns=timestamps, rgb=onboard[:, :7])
            receipt["artifact_hashes"]["sensors/onboard_rgbd.npz"] = {
                "sha256": sha256_file(onboard_path),
                "bytes": onboard_path.stat().st_size,
            }
            (root / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            self._write_passing_validation(root)
            with self.assertRaisesRegex(RuntimeError, r"\[T,8,H,W"):
                encode_isaac_composite(root, destination)


if __name__ == "__main__":
    unittest.main()
