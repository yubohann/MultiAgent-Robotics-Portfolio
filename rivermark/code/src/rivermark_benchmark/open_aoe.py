"""Read-only provenance and consistency checks for external Open-AoE segments.

Open-AoE is a human egocentric manipulation corpus.  It is valuable as an
external pretraining source for visual, language-conditioned, and world-model
representations, but its MANO hand actions are not CF2X flight actions.  This
module intentionally never creates a Rivermark episode, never copies source
media, and emits a path-free manifest with that boundary frozen in its schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

OPEN_AOE_EXTERNAL_PROVENANCE_SCHEMA = "org.rivermark.external-pretraining.open-aoe.v1"
OPEN_AOE_REPOSITORY = "https://github.com/ant-research/Open-AoE"
OPEN_AOE_DATASET = "https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h"
OPEN_AOE_TECHNICAL_REPORT = "arXiv:2607.14183v2"
OPEN_AOE_PURPOSE = "external_pretraining_only"
_ABSOLUTE_TOLERANCE = 1e-3
_RELATIVE_TOLERANCE = 5e-3

_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("raw_rgb_video", "raw_video.mp4"),
    ("raw_camera_info", "video_info.json"),
    ("atomic_action_annotations", "ego_annotation/ego_action_annotation.json"),
    ("mano_hands", "ego_process/ego_hands_reconstruction/hands.npz"),
    ("camera_trajectory", "ego_process/ego_hands_reconstruction/camera_traj.npz"),
    ("undistorted_rgb_video", "ego_process/ego_undistorted_video/raw_video_undistorted.mp4"),
    ("undistorted_camera_info", "ego_process/ego_undistorted_video/undistorted_video_info.json"),
)
_HAND_SHAPES: dict[str, tuple[object, ...]] = {
    "R_w2c": ("T", 3, 3),
    "t_w2c": ("T", 3),
    "R_c2w": ("T", 3, 3),
    "t_c2w": ("T", 3),
    "pred_trans": (2, "T", 3),
    "pred_rot": (2, "T", 3),
    "pred_trans_cam": (2, "T", 3),
    "pred_rot_cam": (2, "T", 3),
    "pred_hand_pose": (2, "T", 45),
    "pred_betas": (2, "T", 10),
    "pred_valid": (2, "T"),
}


class OpenAoeError(ValueError):
    """Raised when an external Open-AoE input cannot be audited safely."""


@dataclass(frozen=True)
class OpenAoeIssue:
    code: str
    artifact: str
    message: str


@dataclass(frozen=True)
class OpenAoeSegmentReport:
    """Internal report; ``public_record`` deliberately omits local paths."""

    segment_root: Path
    frame_count: int | None
    fps: float | None
    annotation_segment_count: int | None
    valid_hand_frame_ratio: float | None
    artifact_records: tuple[dict[str, object], ...]
    issues: tuple[OpenAoeIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def public_record(self, ordinal: int) -> dict[str, object]:
        return {
            "segment_id": f"open-aoe-{ordinal:06d}",
            "status": "valid" if self.valid else "invalid",
            "frame_count": self.frame_count,
            "fps": self.fps,
            "annotation_segment_count": self.annotation_segment_count,
            "valid_hand_frame_ratio": self.valid_hand_frame_ratio,
            "artifacts": [dict(value) for value in self.artifact_records],
            "issues": [asdict(issue) for issue in self.issues],
        }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append(issues: list[OpenAoeIssue], code: str, artifact: str, message: str) -> None:
    issues.append(OpenAoeIssue(code=code, artifact=artifact, message=message))


def _json_mapping(path: Path, artifact: str, issues: list[OpenAoeIssue]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _append(issues, "json", artifact, f"cannot parse JSON: {exc}")
        return None
    if not isinstance(value, dict):
        _append(issues, "schema", artifact, "must contain a JSON object")
        return None
    return dict(value)


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _camera_parameters(
    value: Mapping[str, Any] | None,
    *,
    artifact: str,
    issues: list[OpenAoeIssue],
    require_fps: bool,
) -> tuple[float, float, float, float, float | None] | None:
    if value is None:
        return None
    parameters = value.get("cameraParams")
    if not isinstance(parameters, Mapping):
        _append(issues, "schema", artifact, "cameraParams must be an object")
        return None
    parsed: list[float] = []
    for name in ("fx_pixels", "fy_pixels", "cx_pixels", "cy_pixels"):
        number = _finite_positive(parameters.get(name)) if name.startswith("f") else _finite_nonnegative(parameters.get(name))
        if number is None:
            _append(issues, "camera_intrinsic", artifact, f"cameraParams.{name} must be finite and valid")
        else:
            parsed.append(number)
    fps: float | None = None
    if require_fps:
        fps = _finite_positive(value.get("fps"))
        if fps is None:
            _append(issues, "fps", artifact, "fps must be finite and positive")
        filename = value.get("video_filename")
        if filename != "raw_video_undistorted.mp4":
            _append(issues, "schema", artifact, "video_filename must be raw_video_undistorted.mp4")
    if len(parsed) != 4:
        return None
    return (*parsed, fps)


def _shape_matches(value: np.ndarray, expected: tuple[object, ...], frames: int) -> bool:
    resolved = tuple(frames if dimension == "T" else dimension for dimension in expected)
    return tuple(value.shape) == resolved


def _near(left: np.ndarray | float, right: np.ndarray | float) -> bool:
    return bool(np.allclose(left, right, rtol=_RELATIVE_TOLERANCE, atol=_ABSOLUTE_TOLERANCE))


def _check_rotations(values: np.ndarray, *, artifact: str, field: str, issues: list[OpenAoeIssue]) -> None:
    if not np.isfinite(values).all():
        _append(issues, "nonfinite", artifact, f"{field} contains non-finite values")
        return
    identity = np.eye(3, dtype=values.dtype)
    products = np.matmul(np.swapaxes(values, -1, -2), values)
    if not _near(products, identity):
        _append(issues, "rotation", artifact, f"{field} is not orthonormal")
    determinants = np.linalg.det(values)
    if not _near(determinants, np.ones_like(determinants)):
        _append(issues, "rotation", artifact, f"{field} determinant is not +1")


def _load_hands_and_trajectory(
    hands_path: Path,
    trajectory_path: Path,
    undistorted_intrinsic: tuple[float, float, float, float, float | None] | None,
    issues: list[OpenAoeIssue],
) -> tuple[int | None, float | None]:
    artifact = "mano_hands"
    frame_count: int | None = None
    valid_ratio: float | None = None
    try:
        with np.load(hands_path, allow_pickle=False) as hands:
            missing = sorted(set(_HAND_SHAPES) - set(hands.files))
            if missing:
                _append(issues, "npz_keys", artifact, f"missing required fields: {', '.join(missing)}")
                return None, None
            r_w2c = np.asarray(hands["R_w2c"])
            if r_w2c.ndim != 3 or tuple(r_w2c.shape[1:]) != (3, 3) or r_w2c.shape[0] <= 0:
                _append(issues, "shape", artifact, "R_w2c must have shape (T, 3, 3) with T > 0")
                return None, None
            frame_count = int(r_w2c.shape[0])
            arrays = {name: np.asarray(hands[name]) for name in _HAND_SHAPES}
            for name, expected in _HAND_SHAPES.items():
                if not _shape_matches(arrays[name], expected, frame_count):
                    _append(issues, "shape", artifact, f"{name} has shape {tuple(arrays[name].shape)}, expected {expected}")
                elif not np.isfinite(arrays[name]).all():
                    _append(issues, "nonfinite", artifact, f"{name} contains non-finite values")
            focal = np.asarray(hands["focal"])
            if focal.shape != () or not np.isfinite(focal).all() or float(focal) <= 0.0:
                _append(issues, "focal", artifact, "focal must be a finite positive scalar")
            elif undistorted_intrinsic is not None and not _near(float(focal), undistorted_intrinsic[0]):
                _append(issues, "intrinsic_mismatch", artifact, "focal disagrees with undistorted fx_pixels")
            if all(_shape_matches(arrays[name], expected, frame_count) for name, expected in _HAND_SHAPES.items()):
                _check_rotations(arrays["R_w2c"], artifact=artifact, field="R_w2c", issues=issues)
                _check_rotations(arrays["R_c2w"], artifact=artifact, field="R_c2w", issues=issues)
                if not _near(np.matmul(arrays["R_w2c"], arrays["R_c2w"]), np.eye(3)):
                    _append(issues, "transform_closure", artifact, "R_w2c and R_c2w do not compose to identity")
                translated = np.einsum("tij,tj->ti", arrays["R_w2c"], arrays["t_c2w"]) + arrays["t_w2c"]
                if not _near(translated, np.zeros_like(translated)):
                    _append(issues, "transform_closure", artifact, "world-to-camera translation does not close")
                valid = arrays["pred_valid"]
                if not np.all(np.logical_or(np.isclose(valid, 0.0), np.isclose(valid, 1.0))):
                    _append(issues, "validity_mask", artifact, "pred_valid must contain only 0 or 1")
                else:
                    valid_ratio = float(np.mean(valid > 0.5))
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:  # type: ignore[name-defined]
        _append(issues, "npz", artifact, f"cannot load hands.npz: {exc}")
        return None, None

    trajectory_artifact = "camera_trajectory"
    try:
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            if {"cam_c2w", "intrinsic"} - set(trajectory.files):
                _append(issues, "npz_keys", trajectory_artifact, "must contain cam_c2w and intrinsic")
                return frame_count, valid_ratio
            cam_c2w = np.asarray(trajectory["cam_c2w"])
            intrinsic = np.asarray(trajectory["intrinsic"])
            if frame_count is not None and tuple(cam_c2w.shape) != (frame_count, 4, 4):
                _append(issues, "shape", trajectory_artifact, "cam_c2w must have shape (T, 4, 4) matching hands.npz")
            elif not np.isfinite(cam_c2w).all():
                _append(issues, "nonfinite", trajectory_artifact, "cam_c2w contains non-finite values")
            elif not _near(cam_c2w[:, 3, :], np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (cam_c2w.shape[0], 1))):
                _append(issues, "transform", trajectory_artifact, "cam_c2w homogeneous row is invalid")
            else:
                _check_rotations(cam_c2w[:, :3, :3], artifact=trajectory_artifact, field="cam_c2w", issues=issues)
            if tuple(intrinsic.shape) != (3, 3) or not np.isfinite(intrinsic).all():
                _append(issues, "intrinsic", trajectory_artifact, "intrinsic must be a finite 3x3 matrix")
            elif not _near(intrinsic[2], np.asarray([0.0, 0.0, 1.0])):
                _append(issues, "intrinsic", trajectory_artifact, "intrinsic last row must be [0, 0, 1]")
            elif undistorted_intrinsic is not None:
                expected = np.asarray(
                    [[undistorted_intrinsic[0], 0.0, undistorted_intrinsic[2]], [0.0, undistorted_intrinsic[1], undistorted_intrinsic[3]], [0.0, 0.0, 1.0]]
                )
                if not _near(intrinsic, expected):
                    _append(issues, "intrinsic_mismatch", trajectory_artifact, "intrinsic disagrees with undistorted camera metadata")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:  # type: ignore[name-defined]
        _append(issues, "npz", trajectory_artifact, f"cannot load camera_traj.npz: {exc}")
    return frame_count, valid_ratio


def _as_frame(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number != round(number):
        return None
    return int(number)


def _validate_annotations(
    path: Path,
    *,
    frame_count: int | None,
    issues: list[OpenAoeIssue],
) -> int | None:
    artifact = "atomic_action_annotations"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _append(issues, "json", artifact, f"cannot parse annotations: {exc}")
        return None
    if not isinstance(value, list) or not value:
        _append(issues, "schema", artifact, "must be a non-empty JSON array")
        return None
    previous_end: int | None = None
    for index, segment in enumerate(value):
        location = f"{artifact}[{index}]"
        if not isinstance(segment, Mapping):
            _append(issues, "schema", location, "segment must be an object")
            continue
        start = _as_frame(segment.get("start_frame"))
        end = _as_frame(segment.get("end_frame"))
        if start is None or end is None or start < 0 or end <= start:
            _append(issues, "annotation_range", location, "start_frame/end_frame must be non-negative increasing integers")
        else:
            if previous_end is None and start != 0:
                _append(issues, "annotation_coverage", location, "first action segment must start at frame 0")
            if previous_end is not None and start != previous_end:
                _append(issues, "annotation_coverage", location, "action segments must be contiguous and non-overlapping")
            if frame_count is not None and end > frame_count:
                _append(issues, "annotation_range", location, "end_frame exceeds synchronized tensor frame count")
            previous_end = end
        actions = segment.get("atomic_action")
        if not isinstance(actions, list) or not actions:
            _append(issues, "schema", location, "atomic_action must be a non-empty array")
            continue
        for action_index, action in enumerate(actions):
            action_location = f"{location}.atomic_action[{action_index}]"
            if not isinstance(action, Mapping):
                _append(issues, "schema", action_location, "action must be an object")
                continue
            for field in ("verb", "object", "hand", "description"):
                if not isinstance(action.get(field), str) or not str(action[field]).strip():
                    _append(issues, "schema", action_location, f"{field} must be a non-empty string")
            if action.get("hand") not in {"left", "right", "both", "none"}:
                _append(issues, "hand", action_location, "hand must be left, right, both, or none")
    if frame_count is not None and previous_end not in {frame_count - 1, frame_count}:
        _append(issues, "annotation_coverage", artifact, "final action segment does not cover the final synchronized frame")
    return len(value)


def _artifact_records(segment_root: Path, *, include_hashes: bool, issues: list[OpenAoeIssue]) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for role, relative in _ARTIFACTS:
        path = segment_root / relative
        if not path.is_file():
            _append(issues, "missing_artifact", role, f"required artifact is absent: {relative}")
            continue
        size = path.stat().st_size
        if size <= 0:
            _append(issues, "empty_artifact", role, f"required artifact is empty: {relative}")
            continue
        record: dict[str, object] = {"role": role, "size_bytes": size}
        if include_hashes:
            record["sha256"] = _sha256_file(path)
        records.append(record)
    return tuple(records)


def inspect_open_aoe_segment(segment_root: Path, *, include_hashes: bool = True) -> OpenAoeSegmentReport:
    """Validate one documented Open-AoE segment without decoding its video.

    The absence of a decoder keeps the adapter CPU-only and portable, but means
    that media frame-count agreement is deliberately *not* claimed here.  The
    synchronized NPZ arrays, metadata intrinsics, transforms, and annotation
    boundaries are all checked directly.
    """

    root = Path(segment_root).expanduser().resolve()
    if not root.is_dir():
        raise OpenAoeError(f"Open-AoE segment directory does not exist: {root}")
    issues: list[OpenAoeIssue] = []
    records = _artifact_records(root, include_hashes=include_hashes, issues=issues)
    paths = {role: root / relative for role, relative in _ARTIFACTS}
    raw_info = _json_mapping(paths["raw_camera_info"], "raw_camera_info", issues) if paths["raw_camera_info"].is_file() else None
    _camera_parameters(raw_info, artifact="raw_camera_info", issues=issues, require_fps=False)
    undistorted_info = (
        _json_mapping(paths["undistorted_camera_info"], "undistorted_camera_info", issues)
        if paths["undistorted_camera_info"].is_file()
        else None
    )
    undistorted_intrinsic = _camera_parameters(
        undistorted_info,
        artifact="undistorted_camera_info",
        issues=issues,
        require_fps=True,
    )
    frame_count: int | None = None
    valid_ratio: float | None = None
    if paths["mano_hands"].is_file() and paths["camera_trajectory"].is_file():
        frame_count, valid_ratio = _load_hands_and_trajectory(
            paths["mano_hands"], paths["camera_trajectory"], undistorted_intrinsic, issues
        )
    annotation_count = (
        _validate_annotations(paths["atomic_action_annotations"], frame_count=frame_count, issues=issues)
        if paths["atomic_action_annotations"].is_file()
        else None
    )
    fps = undistorted_intrinsic[4] if undistorted_intrinsic is not None else None
    return OpenAoeSegmentReport(
        segment_root=root,
        frame_count=frame_count,
        fps=fps,
        annotation_segment_count=annotation_count,
        valid_hand_frame_ratio=valid_ratio,
        artifact_records=records,
        issues=tuple(issues),
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _segment_roots(dataset_root: Path, *, max_segments: int | None) -> tuple[Path, ...]:
    if max_segments is not None and (isinstance(max_segments, bool) or not isinstance(max_segments, int) or max_segments <= 0):
        raise OpenAoeError("max_segments must be a positive integer when provided")
    roots: list[Path] = []
    if (dataset_root / "raw_video.mp4").is_file():
        roots.append(dataset_root)
    else:
        for raw_video in sorted(dataset_root.rglob("raw_video.mp4")):
            if raw_video.is_file() and _is_within(raw_video, dataset_root):
                roots.append(raw_video.parent)
                if max_segments is not None and len(roots) >= max_segments:
                    break
    if max_segments is not None:
        roots = roots[:max_segments]
    if not roots:
        raise OpenAoeError(f"no Open-AoE segments containing raw_video.mp4 were found under {dataset_root}")
    return tuple(roots)


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def scan_open_aoe_root(
    dataset_root: Path,
    *,
    repository_root: Path | None = None,
    max_segments: int | None = None,
    include_hashes: bool = True,
    access_basis: str = "user_authorized_local_use",
) -> dict[str, object]:
    """Create a path-free external-pretraining manifest for valid source segments.

    Invalid segments are retained in the manifest's denominator.  A scan with
    no valid segments fails closed, because a provenance file with only failed
    sources must not look like usable training evidence.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise OpenAoeError(f"Open-AoE dataset root does not exist: {root}")
    if repository_root is not None and _is_within(root, Path(repository_root)):
        raise OpenAoeError("external Open-AoE payload must remain outside the Rivermark repository")
    if access_basis not in {"user_authorized_local_use", "upstream_terms_verified"}:
        raise OpenAoeError("access_basis must be user_authorized_local_use or upstream_terms_verified")
    reports = tuple(inspect_open_aoe_segment(path, include_hashes=include_hashes) for path in _segment_roots(root, max_segments=max_segments))
    valid_count = sum(report.valid for report in reports)
    if valid_count == 0:
        raise OpenAoeError("no Open-AoE segment passed metadata, geometry, and annotation validation")
    records = [report.public_record(index) for index, report in enumerate(reports, start=1)]
    total_bytes = sum(
        int(artifact["size_bytes"])
        for report in reports
        for artifact in report.artifact_records
    )
    manifest: dict[str, object] = {
        "schema": OPEN_AOE_EXTERNAL_PROVENANCE_SCHEMA,
        "source": {
            "dataset_name": "Open-AoE-2000h",
            "repository": OPEN_AOE_REPOSITORY,
            "dataset_landing_page": OPEN_AOE_DATASET,
            "technical_report": OPEN_AOE_TECHNICAL_REPORT,
            "code_license": "Apache-2.0",
            "access_basis": access_basis,
            "payload_copied_into_rivermark": False,
            "local_source_path_redacted": True,
        },
        "purpose": OPEN_AOE_PURPOSE,
        "claim_boundary": (
            "Human egocentric manipulation source for external pretraining only; not a Rivermark UAV "
            "episode, not a CF2X action source, and not evidence of native Isaac closed-loop execution."
        ),
        "formal_rivermark_admission": False,
        "isaac_execution_evidence": False,
        "video_decode_validation": "not_run_by_this_cpu_only_adapter",
        "segment_count": len(records),
        "valid_segment_count": valid_count,
        "invalid_segment_count": len(records) - valid_count,
        "total_artifact_bytes": total_bytes,
        "segments": records,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def write_open_aoe_manifest(path: Path, manifest: Mapping[str, object], *, overwrite: bool = False) -> Path:
    """Atomically write a previously validated path-free manifest."""

    if manifest.get("schema") != OPEN_AOE_EXTERNAL_PROVENANCE_SCHEMA:
        raise OpenAoeError("manifest does not use the Open-AoE external provenance schema")
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise OpenAoeError("manifest hash is missing or does not bind its content")
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise OpenAoeError(f"refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary_name = stream.name
            stream.write(_canonical_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate external Open-AoE segments and write a path-free pretraining provenance manifest."
    )
    parser.add_argument("--open-aoe-root", type=Path, required=True, help="external root containing Open-AoE segments")
    parser.add_argument("--output", type=Path, required=True, help="path-free manifest output path")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-segments", type=int, default=None, help="bounded number of source segments to inspect")
    parser.add_argument("--skip-file-hashes", action="store_true", help="omit expensive content hashes; not suitable for release evidence")
    parser.add_argument(
        "--access-basis",
        choices=("user_authorized_local_use", "upstream_terms_verified"),
        default="user_authorized_local_use",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = scan_open_aoe_root(
            args.open_aoe_root,
            repository_root=args.repository_root,
            max_segments=args.max_segments,
            include_hashes=not args.skip_file_hashes,
            access_basis=args.access_basis,
        )
        output = write_open_aoe_manifest(args.output, manifest, overwrite=args.overwrite)
    except OpenAoeError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=True, sort_keys=True))
        return 2
    print(json.dumps({"status": "passed", "output": str(output), "manifest_sha256": manifest["manifest_sha256"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
