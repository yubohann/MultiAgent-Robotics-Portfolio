"""Create a standalone, development-only Isaac pilot evidence bundle.

This module deliberately has no path into formal candidate packing or dataset
admission.  It projects only hash-bound, independently validated MP4 evidence
into a new external directory.  Raw capture and validation receipts are used
for verification but are never copied, which keeps evaluator/private material
outside the public bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import is_sha256
from .video import (
    STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA,
    SUPPORTED_INDEPENDENT_VALIDATION_SCHEMAS,
    VideoAudit,
    audit_video,
    independent_validation_schema_for_capture,
    require_release_video,
    sha256_file,
)


CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
EVIDENCE_MANIFEST_SCHEMA = "org.rivermark.isaac-development-evidence-manifest.v1"
EVIDENCE_RECEIPT_SCHEMA = "org.rivermark.isaac-development-evidence-receipt.v1"

_ALLOWED_VIDEO_RECEIPT_SCHEMAS = frozenset(
    {
        "org.rivermark.isaac-demo-video.v1",
        "org.rivermark.isaac-swarm-composite-video.v1",
    }
)
_FORBIDDEN_SOURCE_PARTS = frozenset(
    {
        "evaluator-private",
        "evaluator_private",
        "hidden",
        "hidden_truth",
        "private",
        "target_truth",
    }
)


@dataclass(frozen=True)
class IsaacEvidenceIssue:
    """One fail-closed reason why a development evidence pack was rejected."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class IsaacEvidenceResult:
    """The externally materialized bundle, if all input bindings passed."""

    bundle_root: Path | None
    manifest_sha256: str | None
    receipt_sha256: str | None
    issues: tuple[IsaacEvidenceIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues and self.bundle_root is not None


class _EvidenceError(RuntimeError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.issue = IsaacEvidenceIssue(code=code, path=path, message=message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _EvidenceError("invalid_json", label, str(exc)) from exc
    if not isinstance(value, Mapping):
        raise _EvidenceError("json_type", label, "expected a JSON object")
    return value


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_external_destination(capture_root: Path, destination: Path) -> None:
    if destination.exists():
        raise _EvidenceError("destination_exists", str(destination), "destination must not already exist")
    if _is_within(destination, capture_root) or _is_within(capture_root, destination):
        raise _EvidenceError(
            "destination_boundary",
            str(destination),
            "development evidence must be materialized outside the raw capture root",
        )


def _require_public_source(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise _EvidenceError("missing_source", label, "source must be an existing regular file")
    if any(part.lower() in _FORBIDDEN_SOURCE_PARTS for part in resolved.parts):
        raise _EvidenceError(
            "private_source",
            label,
            "evaluator/private/hidden source paths cannot be projected into public evidence",
        )
    return resolved


def _capture_contract(capture_root: Path) -> tuple[Mapping[str, Any], str]:
    receipt_path = capture_root / "capture_receipt.json"
    receipt = _read_object(receipt_path, label="capture_receipt.json")
    if receipt.get("schema") != CAPTURE_SCHEMA:
        raise _EvidenceError("capture_schema", "capture_receipt.json", "unsupported capture receipt schema")
    if receipt.get("status") != "captured" or receipt.get("ok") is not True:
        raise _EvidenceError("capture_status", "capture_receipt.json", "capture did not complete successfully")
    claim_boundary = receipt.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or claim_boundary.get("formal_benchmark_admission") is not False:
        raise _EvidenceError(
            "capture_claim_boundary",
            "capture_receipt.json.claim_boundary.formal_benchmark_admission",
            "raw capture must explicitly remain outside formal benchmark admission",
        )
    task_kind = receipt.get("task_kind")
    if not isinstance(task_kind, str) or not task_kind:
        raise _EvidenceError("capture_task_kind", "capture_receipt.json.task_kind", "must be a non-empty string")
    return receipt, sha256_file(receipt_path)


def _validation_contract(
    validation_receipt: Path,
    capture: Mapping[str, Any],
    capture_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    validation = _read_object(validation_receipt, label="independent_validation")
    if (
        not isinstance(validation.get("schema"), str)
        or validation.get("schema") not in SUPPORTED_INDEPENDENT_VALIDATION_SCHEMAS
    ):
        raise _EvidenceError("validation_schema", "independent_validation.schema", "unsupported validation receipt schema")
    if validation.get("status") != "passed" or validation.get("issues") != []:
        raise _EvidenceError("validation_status", "independent_validation", "independent validation did not pass cleanly")
    if validation.get("formal_benchmark_admission") is not False:
        raise _EvidenceError(
            "validation_claim_boundary",
            "independent_validation.formal_benchmark_admission",
            "independent validation must not claim formal admission",
        )
    if validation.get("capture_receipt_sha256") != capture_sha256:
        raise _EvidenceError(
            "validation_binding",
            "independent_validation.capture_receipt_sha256",
            "validation receipt does not bind this raw capture receipt",
        )
    validator_id = validation.get("validator_id")
    if not isinstance(validator_id, str) or not validator_id.strip():
        raise _EvidenceError("validator_id", "independent_validation.validator_id", "must be a non-empty string")
    if not is_sha256(validation.get("validator_source_sha256")):
        raise _EvidenceError(
            "validator_source",
            "independent_validation.validator_source_sha256",
            "must be a SHA-256 digest",
        )
    if not isinstance(validation.get("checks"), Mapping):
        raise _EvidenceError("validation_checks", "independent_validation.checks", "must be an object")
    try:
        independent_validation_schema_for_capture(validation, capture)
    except ValueError as exc:
        raise _EvidenceError(
            "validation_capture_contract",
            "independent_validation.schema",
            str(exc),
        ) from exc
    return validation, sha256_file(validation_receipt)


def _audit_matches_receipt(audit: VideoAudit, receipt_audit: Mapping[str, Any], *, label: str) -> None:
    expected: dict[str, Any] = {
        "sha256": audit.sha256,
        "bytes": audit.bytes,
        "width": audit.width,
        "height": audit.height,
        "frame_count": audit.frame_count,
        "first_frame_nonconstant": audit.first_frame_nonconstant,
        "ffmpeg_full_decode": audit.ffmpeg_full_decode,
        "h264_signature_present": audit.h264_signature_present,
        "pixel_format": audit.pixel_format,
        "faststart": audit.faststart,
        "codec_name": audit.codec_name,
        "moov_before_mdat": audit.moov_before_mdat,
        "opencv_full_decode": audit.opencv_full_decode,
        "ffmpeg_frame_count": audit.ffmpeg_frame_count,
        "opencv_reported_frame_count": audit.opencv_reported_frame_count,
    }
    for key, value in expected.items():
        if receipt_audit.get(key) != value:
            raise _EvidenceError(
                "video_audit_binding",
                f"{label}.audit.{key}",
                "video receipt audit does not match a fresh decode audit",
            )
    receipt_fps = receipt_audit.get("fps")
    if (
        isinstance(receipt_fps, bool)
        or not isinstance(receipt_fps, (int, float))
        or not math.isfinite(float(receipt_fps))
        or not math.isclose(float(receipt_fps), audit.fps, rel_tol=1.0e-9, abs_tol=1.0e-9)
    ):
        raise _EvidenceError(
            "video_audit_binding",
            f"{label}.audit.fps",
            "video receipt fps does not match a fresh decode audit",
        )


def _video_contract(
    video_path: Path,
    *,
    capture_sha256: str,
    validation_sha256: str,
) -> dict[str, Any]:
    source = _require_public_source(video_path, label=str(video_path))
    if source.suffix.lower() != ".mp4":
        raise _EvidenceError("video_extension", str(source), "evidence inputs must use the .mp4 extension")
    receipt_path = _require_public_source(
        source.with_suffix(source.suffix + ".receipt.json"),
        label=f"{source.name}.receipt.json",
    )
    receipt = _read_object(receipt_path, label=f"{source.name}.receipt.json")
    if receipt.get("schema") not in _ALLOWED_VIDEO_RECEIPT_SCHEMAS or receipt.get("ok") is not True:
        raise _EvidenceError("video_status", receipt_path.name, "unsupported or unsuccessful Isaac video receipt")
    actual_sha256 = sha256_file(source)
    if receipt.get("video_sha256") != actual_sha256:
        raise _EvidenceError("video_hash", receipt_path.name, "video receipt does not bind the MP4 bytes")
    if receipt.get("capture_receipt_sha256") != capture_sha256:
        raise _EvidenceError("video_capture_binding", receipt_path.name, "video receipt does not bind this capture")
    if receipt.get("independent_validation_sha256") != validation_sha256:
        raise _EvidenceError(
            "video_validation_binding",
            receipt_path.name,
            "video receipt does not bind this passing validation receipt",
        )
    timestamp_sha256 = receipt.get("timestamps_sha256")
    timestamps = receipt.get("timestamps")
    if (
        not is_sha256(timestamp_sha256)
        or not isinstance(timestamps, Mapping)
        or timestamps.get("sha256") != timestamp_sha256
    ):
        raise _EvidenceError("video_timestamps", receipt_path.name, "video receipt lacks a self-consistent timestamp digest")
    receipt_audit = receipt.get("audit")
    if not isinstance(receipt_audit, Mapping) or receipt_audit.get("path") != source.name:
        raise _EvidenceError("video_audit", receipt_path.name, "video receipt audit is missing or names another file")
    try:
        audit = audit_video(source)
        require_release_video(audit)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _EvidenceError("video_decode", str(source), str(exc)) from exc
    _audit_matches_receipt(audit, receipt_audit, label=receipt_path.name)
    return {
        "source": source,
        "source_receipt": receipt_path,
        "sha256": actual_sha256,
        "bytes": source.stat().st_size,
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_schema": receipt["schema"],
        "timestamps_sha256": timestamp_sha256,
        "audit": {
            "width": audit.width,
            "height": audit.height,
            "fps": audit.fps,
            "frame_count": audit.frame_count,
            "codec_name": audit.codec_name,
            "pixel_format": audit.pixel_format,
        },
    }


def _public_manifest(
    capture: Mapping[str, Any],
    capture_sha256: str,
    validation: Mapping[str, Any],
    validation_sha256: str,
    videos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_MANIFEST_SCHEMA,
        "development_only": True,
        "formal_benchmark_admission": False,
        "dataset_episode": False,
        "capture": {
            "capture_receipt_sha256": capture_sha256,
            "schema": CAPTURE_SCHEMA,
            "status": "captured",
            "task_kind": capture["task_kind"],
        },
        "independent_validation": {
            "validation_receipt_sha256": validation_sha256,
            "schema": validation["schema"],
            "status": "passed",
            "validator_id": validation["validator_id"],
            "validator_source_sha256": validation["validator_source_sha256"],
        },
        "videos": [
            {
                "path": video["path"],
                "sha256": video["sha256"],
                "bytes": video["bytes"],
                "receipt_sha256": video["receipt_sha256"],
                "receipt_schema": video["receipt_schema"],
                "timestamps_sha256": video["timestamps_sha256"],
                "audit": video["audit"],
            }
            for video in videos
        ],
    }


def _public_receipt(manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    capture = manifest["capture"]
    validation = manifest["independent_validation"]
    videos = manifest["videos"]
    assert isinstance(capture, Mapping)
    assert isinstance(validation, Mapping)
    assert isinstance(videos, list)
    return {
        "schema": EVIDENCE_RECEIPT_SCHEMA,
        "status": "packed",
        "ok": True,
        "development_only": True,
        "formal_benchmark_admission": False,
        "dataset_episode": False,
        "evidence_manifest_sha256": manifest_sha256,
        "capture_receipt_sha256": capture["capture_receipt_sha256"],
        "independent_validation_sha256": validation["validation_receipt_sha256"],
        "videos": [
            {
                "path": video["path"],
                "sha256": video["sha256"],
                "receipt_sha256": video["receipt_sha256"],
            }
            for video in videos
        ],
    }


def _verify_staged_bundle(root: Path) -> None:
    manifest_path = root / "evidence_manifest.json"
    receipt_path = root / "evidence_receipt.json"
    manifest = _read_object(manifest_path, label="evidence_manifest.json")
    receipt = _read_object(receipt_path, label="evidence_receipt.json")
    if manifest.get("schema") != EVIDENCE_MANIFEST_SCHEMA or receipt.get("schema") != EVIDENCE_RECEIPT_SCHEMA:
        raise _EvidenceError("bundle_schema", str(root), "evidence manifest or receipt schema is invalid")
    for payload, label in ((manifest, "evidence_manifest.json"), (receipt, "evidence_receipt.json")):
        if (
            payload.get("development_only") is not True
            or payload.get("formal_benchmark_admission") is not False
            or payload.get("dataset_episode") is not False
        ):
            raise _EvidenceError("bundle_claim_boundary", label, "evidence bundle claim boundary is invalid")
    if receipt.get("status") != "packed" or receipt.get("ok") is not True:
        raise _EvidenceError("bundle_status", "evidence_receipt.json", "evidence receipt must be successful")
    if receipt.get("evidence_manifest_sha256") != sha256_file(manifest_path):
        raise _EvidenceError("manifest_binding", "evidence_receipt.json", "receipt does not bind manifest bytes")
    capture = manifest.get("capture")
    validation = manifest.get("independent_validation")
    videos = manifest.get("videos")
    if not isinstance(capture, Mapping) or not isinstance(validation, Mapping) or not isinstance(videos, list) or not videos:
        raise _EvidenceError("bundle_structure", "evidence_manifest.json", "bundle manifest is incomplete")
    validation_schema = validation.get("schema")
    if (
        not isinstance(validation_schema, str)
        or validation_schema not in SUPPORTED_INDEPENDENT_VALIDATION_SCHEMAS
    ):
        raise _EvidenceError("validation_schema", "evidence_manifest.json", "bundle validation schema is unsupported")
    if (
        validation_schema == STATE_ONLY_TRANSFER_INDEPENDENT_VALIDATION_SCHEMA
        and capture.get("task_kind") != "state_only_control_transfer_smoke"
    ):
        raise _EvidenceError(
            "validation_capture_contract",
            "evidence_manifest.json",
            "state-only transfer validation is bound to the wrong capture task kind",
        )
    if receipt.get("capture_receipt_sha256") != capture.get("capture_receipt_sha256"):
        raise _EvidenceError("capture_binding", "evidence_receipt.json", "receipt/capture binding mismatch")
    if receipt.get("independent_validation_sha256") != validation.get("validation_receipt_sha256"):
        raise _EvidenceError("validation_binding", "evidence_receipt.json", "receipt/validation binding mismatch")
    receipt_videos = receipt.get("videos")
    if not isinstance(receipt_videos, list) or len(receipt_videos) != len(videos):
        raise _EvidenceError("video_inventory", "evidence_receipt.json", "receipt video inventory is incomplete")
    expected_files = {"evidence_manifest.json", "evidence_receipt.json"}
    for index, video in enumerate(videos):
        if not isinstance(video, Mapping):
            raise _EvidenceError("video_inventory", f"evidence_manifest.json.videos[{index}]", "must be an object")
        relative = video.get("path")
        if not isinstance(relative, str) or not relative.startswith("videos/") or "/" in relative[7:]:
            raise _EvidenceError("video_path", f"evidence_manifest.json.videos[{index}]", "invalid bundle video path")
        video_path = root / relative
        expected_files.add(relative)
        if not video_path.is_file() or video.get("sha256") != sha256_file(video_path):
            raise _EvidenceError("video_hash", relative, "bundle video digest does not match")
        bound = receipt_videos[index]
        if not isinstance(bound, Mapping) or any(bound.get(key) != video.get(key) for key in ("path", "sha256", "receipt_sha256")):
            raise _EvidenceError("video_binding", f"evidence_receipt.json.videos[{index}]", "receipt video binding mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise _EvidenceError("bundle_inventory", str(root), "bundle contains unexpected or missing files")


def pack_isaac_development_evidence(
    capture_root: Path,
    validation_receipt: Path,
    destination: Path,
    videos: Sequence[Path],
) -> IsaacEvidenceResult:
    """Copy validated MP4 evidence into one external development-only bundle.

    The function never modifies ``capture_root`` and never invokes formal
    candidate packing, admission, or an Isaac runtime.
    """

    temporary: Path | None = None
    try:
        capture_root = capture_root.expanduser().resolve()
        if not capture_root.is_dir():
            raise _EvidenceError("capture_root", str(capture_root), "capture root must be a directory")
        destination = destination.expanduser().resolve()
        _require_external_destination(capture_root, destination)
        if not videos:
            raise _EvidenceError("video_inventory", "videos", "at least one explicit MP4 input is required")

        capture, capture_sha256 = _capture_contract(capture_root)
        validation_path = _require_public_source(validation_receipt, label="independent_validation")
        validation, validation_sha256 = _validation_contract(
            validation_path,
            capture,
            capture_sha256,
        )

        video_contracts = [
            _video_contract(
                video,
                capture_sha256=capture_sha256,
                validation_sha256=validation_sha256,
            )
            for video in videos
        ]
        source_paths = [contract["source"] for contract in video_contracts]
        output_names = [source.name for source in source_paths]
        if len(source_paths) != len(set(source_paths)):
            raise _EvidenceError("duplicate_video", "videos", "each explicit MP4 input must be unique")
        if len(output_names) != len(set(output_names)):
            raise _EvidenceError("video_name_collision", "videos", "MP4 basenames must be unique")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        public_videos: list[dict[str, Any]] = []
        for contract in video_contracts:
            source = contract["source"]
            assert isinstance(source, Path)
            relative = f"videos/{source.name}"
            copied = temporary / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            if sha256_file(copied) != contract["sha256"]:
                raise _EvidenceError("video_copy", relative, "copied video digest does not match the verified source")
            public_videos.append({**contract, "path": relative})

        manifest = _public_manifest(capture, capture_sha256, validation, validation_sha256, public_videos)
        manifest_path = temporary / "evidence_manifest.json"
        _write_json(manifest_path, manifest)
        receipt = _public_receipt(manifest, sha256_file(manifest_path))
        receipt_path = temporary / "evidence_receipt.json"
        _write_json(receipt_path, receipt)
        _verify_staged_bundle(temporary)
        os.replace(temporary, destination)
        temporary = None
        return IsaacEvidenceResult(
            bundle_root=destination,
            manifest_sha256=sha256_file(destination / "evidence_manifest.json"),
            receipt_sha256=sha256_file(destination / "evidence_receipt.json"),
            issues=(),
        )
    except _EvidenceError as exc:
        return IsaacEvidenceResult(None, None, None, (exc.issue,))
    except (OSError, TypeError, ValueError) as exc:
        return IsaacEvidenceResult(None, None, None, (IsaacEvidenceIssue("pack_error", str(destination), str(exc)),))
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("validation_receipt", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("videos", nargs="+", type=Path, metavar="video.mp4")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = pack_isaac_development_evidence(
        args.capture_root,
        args.validation_receipt,
        args.destination,
        args.videos,
    )
    print(
        json.dumps(
            {
                "valid": result.valid,
                "bundle_root": str(result.bundle_root) if result.bundle_root else None,
                "manifest_sha256": result.manifest_sha256,
                "receipt_sha256": result.receipt_sha256,
                "issues": [asdict(issue) for issue in result.issues],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
