from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.isaac_evidence import (  # noqa: E402
    EVIDENCE_MANIFEST_SCHEMA,
    EVIDENCE_RECEIPT_SCHEMA,
    pack_isaac_development_evidence,
)
from rivermark_benchmark.video import (  # noqa: E402
    STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
    VideoAudit,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _audit(path: Path) -> VideoAudit:
    return VideoAudit(
        path=str(path),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        width=960,
        height=540,
        fps=20.0,
        frame_count=180,
        first_frame_nonconstant=True,
        ffmpeg_full_decode=True,
        h264_signature_present=True,
        pixel_format="yuv420p",
        faststart=True,
        codec_name="h264",
        moov_before_mdat=True,
        opencv_full_decode=True,
        ffmpeg_frame_count=180,
        opencv_reported_frame_count=180,
    )


def _fixture(
    root: Path,
    *,
    state_only_transfer: bool = False,
) -> tuple[Path, Path, list[Path]]:
    capture = root / "raw-capture"
    capture.mkdir()
    task_kind = "state_only_control_transfer_smoke" if state_only_transfer else "search3d"
    claim_boundary = {"formal_benchmark_admission": False}
    if state_only_transfer:
        claim_boundary["development_control_transfer"] = True
    _write_json(
        capture / "capture_receipt.json",
        {
            "schema": "org.rivermark.isaac-swarm-capture.v1",
            "status": "captured",
            "ok": True,
            "task_kind": task_kind,
            "claim_boundary": claim_boundary,
            **(
                {"command": {"control_mode": "sb3_state_only_transfer"}}
                if state_only_transfer
                else {}
            ),
            "evaluator_manifest_sha256": "f" * 64,
        },
    )
    capture_sha256 = sha256_file(capture / "capture_receipt.json")
    validation = root / "independent_validation.json"
    _write_json(
        validation,
        {
            "schema": (
                STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA
                if state_only_transfer
                else "org.rivermark.isaac-independent-validation.v1"
            ),
            "status": "passed",
            "issues": [],
            "formal_benchmark_admission": False,
            "capture_receipt_sha256": capture_sha256,
            "validator_id": "independent-test-validator",
            "validator_source_sha256": "a" * 64,
            "checks": {"evaluator_manifest_sha256": "f" * 64},
        },
    )
    validation_sha256 = sha256_file(validation)
    videos: list[Path] = []
    for name, schema in (
        ("overview.mp4", "org.rivermark.isaac-demo-video.v1"),
        ("composite.mp4", "org.rivermark.isaac-swarm-composite-video.v1"),
    ):
        video = root / "source-videos" / name
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes((name + " verified video bytes").encode("ascii"))
        audit = _audit(video)
        receipt = {
            "schema": schema,
            "ok": True,
            "capture_receipt_sha256": capture_sha256,
            "independent_validation_sha256": validation_sha256,
            "video_sha256": audit.sha256,
            "timestamps_sha256": "b" * 64,
            "timestamps": {"sha256": "b" * 64, "count": 180},
            "audit": {**asdict(audit), "path": video.name},
        }
        _write_json(video.with_suffix(".mp4.receipt.json"), receipt)
        videos.append(video)
    return capture, validation, videos


class IsaacDevelopmentEvidenceTests(unittest.TestCase):
    def _pack(self, capture: Path, validation: Path, destination: Path, videos: list[Path]):
        with patch("rivermark_benchmark.isaac_evidence.audit_video", side_effect=_audit), patch(
            "rivermark_benchmark.isaac_evidence.require_release_video"
        ):
            return pack_isaac_development_evidence(capture, validation, destination, videos)

    def test_external_bundle_is_hash_bound_and_explicitly_not_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            before = sha256_file(capture / "capture_receipt.json")
            destination = root / "evidence-bundle"
            result = self._pack(capture, validation, destination, videos)

            self.assertTrue(result.valid, result.issues)
            self.assertEqual(before, sha256_file(capture / "capture_receipt.json"))
            self.assertFalse((destination / "capture_receipt.json").exists())
            self.assertFalse((destination / "independent_validation.json").exists())
            self.assertFalse((destination / "evaluator_private").exists())
            manifest = json.loads((destination / "evidence_manifest.json").read_text(encoding="utf-8"))
            receipt = json.loads((destination / "evidence_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], EVIDENCE_MANIFEST_SCHEMA)
            self.assertEqual(receipt["schema"], EVIDENCE_RECEIPT_SCHEMA)
            for payload in (manifest, receipt):
                self.assertTrue(payload["development_only"])
                self.assertFalse(payload["formal_benchmark_admission"])
                self.assertFalse(payload["dataset_episode"])
            self.assertEqual(manifest["capture"]["capture_receipt_sha256"], before)
            self.assertEqual(
                manifest["independent_validation"]["validation_receipt_sha256"], sha256_file(validation)
            )
            self.assertEqual(receipt["evidence_manifest_sha256"], sha256_file(destination / "evidence_manifest.json"))
            self.assertEqual(len(manifest["videos"]), 2)
            for record in manifest["videos"]:
                copied = destination / record["path"]
                self.assertTrue(copied.is_file())
                self.assertEqual(record["sha256"], sha256_file(copied))

    def test_state_only_transfer_validation_is_projected_as_development_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root, state_only_transfer=True)
            destination = root / "transfer-evidence-bundle"
            result = self._pack(capture, validation, destination, videos)

            self.assertTrue(result.valid, result.issues)
            manifest = json.loads((destination / "evidence_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["independent_validation"]["schema"],
                STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
            )
            self.assertEqual(manifest["capture"]["task_kind"], "state_only_control_transfer_smoke")
            self.assertTrue(manifest["development_only"])
            self.assertFalse(manifest["formal_benchmark_admission"])
            self.assertFalse(manifest["dataset_episode"])

    def test_state_only_transfer_validation_rejects_a_fixed_route_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            payload = json.loads(validation.read_text(encoding="utf-8"))
            payload["schema"] = STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA
            _write_json(validation, payload)
            result = self._pack(capture, validation, root / "evidence-bundle", videos)

            self.assertFalse(result.valid)
            self.assertIn("validation_capture_contract", {issue.code for issue in result.issues})

    def test_rejects_unbound_validation_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            payload = json.loads(validation.read_text(encoding="utf-8"))
            payload["capture_receipt_sha256"] = "0" * 64
            _write_json(validation, payload)
            destination = root / "evidence-bundle"
            result = self._pack(capture, validation, destination, videos)
            self.assertFalse(result.valid)
            self.assertIn("validation_binding", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_rejects_unsuccessful_validation_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            payload = json.loads(validation.read_text(encoding="utf-8"))
            payload["status"] = "failed"
            payload["issues"] = [{"code": "capture_integrity"}]
            _write_json(validation, payload)
            destination = root / "evidence-bundle"
            result = self._pack(capture, validation, destination, videos)
            self.assertFalse(result.valid)
            self.assertIn("validation_status", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_rejects_nonstring_validation_schema_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            payload = json.loads(validation.read_text(encoding="utf-8"))
            payload["schema"] = []
            _write_json(validation, payload)
            destination = root / "evidence-bundle"
            result = self._pack(capture, validation, destination, videos)
            self.assertFalse(result.valid)
            self.assertIn("validation_schema", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_rejects_video_receipt_with_wrong_validation_hash_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            video_receipt = videos[0].with_suffix(".mp4.receipt.json")
            payload = json.loads(video_receipt.read_text(encoding="utf-8"))
            payload["independent_validation_sha256"] = "0" * 64
            _write_json(video_receipt, payload)
            destination = root / "evidence-bundle"
            result = self._pack(capture, validation, destination, videos)
            self.assertFalse(result.valid)
            self.assertIn("video_validation_binding", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_rejects_video_receipt_with_wrong_video_hash_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            video_receipt = videos[0].with_suffix(".mp4.receipt.json")
            payload = json.loads(video_receipt.read_text(encoding="utf-8"))
            payload["video_sha256"] = "0" * 64
            _write_json(video_receipt, payload)
            destination = root / "evidence-bundle"
            result = self._pack(capture, validation, destination, videos)
            self.assertFalse(result.valid)
            self.assertIn("video_hash", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_rejects_destination_inside_raw_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, videos = _fixture(root)
            result = self._pack(capture, validation, capture / "evidence", videos)
            self.assertFalse(result.valid)
            self.assertIn("destination_boundary", {issue.code for issue in result.issues})
            self.assertFalse((capture / "evidence").exists())


if __name__ == "__main__":
    unittest.main()
