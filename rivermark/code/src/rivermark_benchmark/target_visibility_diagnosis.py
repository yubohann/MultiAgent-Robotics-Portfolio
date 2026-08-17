"""Private, aggregate-only diagnosis for a failed native target-visibility gate.

This module exists to answer one narrow question after a failed Isaac run:
did a target fail because the *recorded* onboard cameras never had a usable
view, or because a target that should have been visible was absent from native
semantic pixels?  It is deliberately not a sampler and never produces target
coordinates, evaluator IDs, seeds, or image crops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .citylite_scene import segment_intersects_aabb
from .citylite_task import (
    ONBOARD_FOCAL_LENGTH_MM,
    ONBOARD_HORIZONTAL_APERTURE_MM,
    ONBOARD_IMAGE_HEIGHT,
    ONBOARD_IMAGE_WIDTH,
    TARGET_VISIBILITY_FRUSTUM_MARGIN,
    TARGET_VISIBILITY_MAX_DISTANCE_M,
    TARGET_VISIBILITY_MIN_DISTANCE_M,
    TARGET_VISIBILITY_MIN_PROJECTED_INSTANCE_PIXELS,
)
from .private_evaluator_manifest import load_native_geometry_catalog
from .frame_archive import ChunkedFrameArchive, FrameArchiveError


DIAGNOSIS_SCHEMA = "org.rivermark.private-target-visibility-diagnosis.v3"
_SEMANTIC_SLOT_PREFIX = "search_target_slot_"
_CALIBRATED_PIXEL_PROBE_SCHEMA = "org.rivermark.private-target-pixel-probe.v1"


class TargetVisibilityDiagnosisError(ValueError):
    """Raised when a failed capture cannot be diagnosed without guessing."""


@dataclass(frozen=True)
class _Target:
    position_w_m: tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True)
class _PixelCalibration:
    centroid_residual_q99_px: float
    color_median_rgb: np.ndarray
    color_residual_q99: float
    depth_residual_q99_m: float
    positive_observation_count: int
    positive_pixel_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _as_target(value: Any) -> _Target:
    if not isinstance(value, Mapping):
        raise TargetVisibilityDiagnosisError("private manifest target must be an object")
    position = value.get("position_w_m")
    radius = value.get("radius_m")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 3
        or not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in position)
    ):
        raise TargetVisibilityDiagnosisError("private manifest target position must be finite xyz")
    if not isinstance(radius, (int, float)) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
        raise TargetVisibilityDiagnosisError("private manifest target radius must be finite and positive")
    return _Target(tuple(float(item) for item in position), float(radius))


def _load_private_targets(path: Path) -> tuple[_Target, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetVisibilityDiagnosisError("private manifest is not readable JSON") from exc
    targets = payload.get("targets") if isinstance(payload, Mapping) else None
    if not isinstance(targets, list) or not targets:
        raise TargetVisibilityDiagnosisError("private manifest has no targets")
    return tuple(_as_target(target) for target in targets)


def _load_pose_stream(capture_root: Path) -> tuple[np.ndarray, np.ndarray]:
    spool = capture_root / ".sensor_spool_v1"
    positions_path = spool / "camera_observed_pos_w_m.npy"
    quaternions_path = spool / "camera_observed_quat_wxyz.npy"
    if not positions_path.is_file() or not quaternions_path.is_file():
        raise TargetVisibilityDiagnosisError(
            "failed capture has no retained observed onboard camera poses"
        )
    positions = np.load(positions_path, mmap_mode="r", allow_pickle=False)
    quaternions = np.load(quaternions_path, mmap_mode="r", allow_pickle=False)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 3
        or quaternions.ndim != 3
        or quaternions.shape[-1] != 4
        or positions.shape[:2] != quaternions.shape[:2]
        or positions.shape[0] <= 0
        or positions.shape[1] <= 0
        or not np.isfinite(positions).all()
        or not np.isfinite(quaternions).all()
    ):
        raise TargetVisibilityDiagnosisError("observed onboard camera pose stream is malformed")
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.any(norms <= 1.0e-9):
        raise TargetVisibilityDiagnosisError("observed onboard camera quaternions are degenerate")
    return positions, quaternions


def _observed_pose_provenance(capture_root: Path) -> dict[str, str]:
    """Identify the retained camera-pose authority without guessing for old runs.

    Recent captures bind ``camera_observed_*`` to the render-facing USD
    hierarchy after the render/read fence. Historical failed captures used the
    same array names for Camera Fabric values. The spool alone cannot
    distinguish the two, so absence of a matching receipt is explicitly
    reported as unknown rather than upgrading historical data by implication.
    """

    receipt_path = capture_root / "capture_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "source": "unknown_unverified_camera_observed_stream",
            "evidence": "matching_capture_receipt_unavailable",
        }
    calibration = receipt.get("calibration") if isinstance(receipt, Mapping) else None
    onboard = calibration.get("onboard_camera") if isinstance(calibration, Mapping) else None
    fabric = onboard.get("fabric_pose_closure") if isinstance(onboard, Mapping) else None
    usd = onboard.get("usd_pose_closure") if isinstance(onboard, Mapping) else None
    if (
        isinstance(fabric, Mapping)
        and fabric.get("authority") == "diagnostic_only_camera_fabric_cache"
        and fabric.get("acceptance_authority") == "render_facing_usd_hierarchy"
        and isinstance(usd, Mapping)
    ):
        artifacts = receipt.get("artifact_hashes")
        relative_paths = (
            ".sensor_spool_v1/camera_observed_pos_w_m.npy",
            ".sensor_spool_v1/camera_observed_quat_wxyz.npy",
        )
        if not isinstance(artifacts, Mapping):
            return {
                "source": "unknown_unverified_camera_observed_stream",
                "evidence": "matching_capture_receipt_has_no_spool_hash_binding",
            }
        for relative in relative_paths:
            binding = artifacts.get(relative)
            path = capture_root / Path(relative)
            if (
                not isinstance(binding, Mapping)
                or binding.get("sha256") != _sha256_file(path)
                or binding.get("bytes") != path.stat().st_size
            ):
                return {
                    "source": "unknown_unverified_camera_observed_stream",
                    "evidence": "matching_capture_receipt_spool_hash_binding_failed",
                }
        return {
            "source": "verified_render_facing_usd_hierarchy_pose_in_isaaclab_world_convention",
            "evidence": "capture_receipt.usd_pose_closure_and_spool_hash_binding",
        }
    return {
        "source": "unknown_unverified_camera_observed_stream",
        "evidence": "matching_capture_receipt_has_no_render_pose_authority_declaration",
    }


def _rotate_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise TargetVisibilityDiagnosisError("observed camera quaternion is degenerate")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _camera_frustum_membership(
    camera_w_m: Sequence[float], camera_quat_wxyz: Sequence[float], target_w_m: Sequence[float]
) -> tuple[bool, bool, float]:
    """Return complete-FOV, conservative-FOV membership and target distance.

    IsaacLab's Camera world convention maps its optical axis to camera +X.
    The paired USD closure in ``isaac_capture`` verifies this same basis, so
    this replay is tied to the render-facing pose instead of an ideal route yaw.

    The conservative margin is a target-sampling safety gate, not a claim that
    the renderer has a narrower camera model.  Keeping both memberships lets a
    private failure diagnosis distinguish a genuine off-camera target from a
    target that appeared only in the outer rendered FOV.
    """

    delta = tuple(float(target_w_m[axis]) - float(camera_w_m[axis]) for axis in range(3))
    distance = math.sqrt(sum(value * value for value in delta))
    if not TARGET_VISIBILITY_MIN_DISTANCE_M <= distance <= TARGET_VISIBILITY_MAX_DISTANCE_M:
        return False, False, distance
    forward = _rotate_wxyz(camera_quat_wxyz, (1.0, 0.0, 0.0))
    # Camera world convention: image-right is local -Y and image-up is +Z.
    right = _rotate_wxyz(camera_quat_wxyz, (0.0, -1.0, 0.0))
    up = _rotate_wxyz(camera_quat_wxyz, (0.0, 0.0, 1.0))
    optical = sum(delta[axis] * forward[axis] for axis in range(3))
    if optical <= 0.0:
        return False, False, distance
    horizontal = math.atan2(sum(delta[axis] * right[axis] for axis in range(3)), optical)
    vertical = math.atan2(sum(delta[axis] * up[axis] for axis in range(3)), optical)
    horizontal_half_fov = math.atan(
        ONBOARD_HORIZONTAL_APERTURE_MM / (2.0 * ONBOARD_FOCAL_LENGTH_MM)
    )
    vertical_aperture = ONBOARD_HORIZONTAL_APERTURE_MM * ONBOARD_IMAGE_HEIGHT / ONBOARD_IMAGE_WIDTH
    vertical_half_fov = math.atan(vertical_aperture / (2.0 * ONBOARD_FOCAL_LENGTH_MM))
    in_render_fov = (
        abs(horizontal) <= horizontal_half_fov
        and abs(vertical) <= vertical_half_fov
    )
    in_conservative_fov = in_render_fov and (
        abs(horizontal) <= horizontal_half_fov * TARGET_VISIBILITY_FRUSTUM_MARGIN
        and abs(vertical) <= vertical_half_fov * TARGET_VISIBILITY_FRUSTUM_MARGIN
    )
    return in_render_fov, in_conservative_fov, distance


def _camera_witness(
    camera_w_m: Sequence[float], camera_quat_wxyz: Sequence[float], target_w_m: Sequence[float]
) -> tuple[bool, float]:
    """Return conservative recorded-pose frustum membership and distance.

    This compatibility helper deliberately retains the historical conservative
    contract.  New diagnostic code should use ``_camera_frustum_membership``
    when it must distinguish the rendered field of view from the sampling
    safety margin.
    """

    _, conservative, distance = _camera_frustum_membership(
        camera_w_m, camera_quat_wxyz, target_w_m
    )
    return conservative, distance


def _project_target_center_px(
    camera_w_m: Sequence[float], camera_quat_wxyz: Sequence[float], target_w_m: Sequence[float]
) -> tuple[float, float, float] | None:
    """Project private target truth only inside the local diagnostic process.

    The implementation deliberately mirrors the recorded-pose camera axes used
    by ``_camera_witness``.  Callers receive only aggregate residuals and probe
    counts; neither projected coordinates nor target truth enter JSON output.
    """

    delta = tuple(float(target_w_m[axis]) - float(camera_w_m[axis]) for axis in range(3))
    forward = _rotate_wxyz(camera_quat_wxyz, (1.0, 0.0, 0.0))
    right = _rotate_wxyz(camera_quat_wxyz, (0.0, -1.0, 0.0))
    up = _rotate_wxyz(camera_quat_wxyz, (0.0, 0.0, 1.0))
    optical = sum(delta[axis] * forward[axis] for axis in range(3))
    if not math.isfinite(optical) or optical <= 0.0:
        return None
    focal_length_px = ONBOARD_FOCAL_LENGTH_MM / ONBOARD_HORIZONTAL_APERTURE_MM * ONBOARD_IMAGE_WIDTH
    horizontal = sum(delta[axis] * right[axis] for axis in range(3))
    vertical = sum(delta[axis] * up[axis] for axis in range(3))
    column = 0.5 * (ONBOARD_IMAGE_WIDTH - 1) + focal_length_px * horizontal / optical
    row = 0.5 * (ONBOARD_IMAGE_HEIGHT - 1) - focal_length_px * vertical / optical
    if not all(math.isfinite(value) for value in (row, column)):
        return None
    return row, column, optical


def _load_finalized_pose_stream(capture_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = capture_root / "sensors" / "camera_poses.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["camera_observed_pos_w_m"], dtype=np.float64)
            quaternions = np.asarray(archive["camera_observed_quat_wxyz"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise TargetVisibilityDiagnosisError("reference capture has no usable finalized camera poses") from exc
    if (
        positions.ndim != 3
        or positions.shape[-1] != 3
        or quaternions.ndim != 3
        or quaternions.shape[-1] != 4
        or positions.shape[:2] != quaternions.shape[:2]
        or positions.shape[0] <= 0
        or not np.isfinite(positions).all()
        or not np.isfinite(quaternions).all()
    ):
        raise TargetVisibilityDiagnosisError("reference camera pose stream is malformed")
    return positions, quaternions


def _semantic_ids_by_slot(semantic_metadata: Any, target_count: int) -> dict[str, tuple[tuple[int, ...], ...]]:
    if isinstance(semantic_metadata, Mapping):
        direct = semantic_metadata.get("per_camera")
        legacy = semantic_metadata.get("replicator_info")
        per_camera = direct if isinstance(direct, list) else (
            legacy.get("per_camera") if isinstance(legacy, Mapping) else None
        )
    else:
        per_camera = None
    if not isinstance(per_camera, list) or len(per_camera) <= 0:
        raise TargetVisibilityDiagnosisError("reference semantic metadata has no per-camera mappings")
    found: dict[str, list[set[int]]] = {
        f"{_SEMANTIC_SLOT_PREFIX}{index:03d}": [set() for _ in range(len(per_camera))]
        for index in range(target_count)
    }
    for camera_index, metadata in enumerate(per_camera):
        if not isinstance(metadata, Mapping):
            continue
        labels = metadata.get("id_to_labels", metadata.get("idToLabels"))
        if not isinstance(labels, Mapping):
            continue
        for raw_id, raw_label in labels.items():
            try:
                semantic_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            class_value = raw_label.get("class") if isinstance(raw_label, Mapping) else None
            classes = {item.strip().lower() for item in str(class_value).split(",") if item.strip()}
            for slot in found:
                if slot.lower() in classes:
                    found[slot][camera_index].add(semantic_id)
    return {slot: tuple(tuple(sorted(ids)) for ids in per_camera_ids) for slot, per_camera_ids in found.items()}


def _semantic_ids_by_frame(
    reference_capture_root: Path,
    *,
    frame_count: int,
    target_count: int,
) -> tuple[dict[str, tuple[tuple[int, ...], ...]], ...]:
    """Read the public per-frame semantic ID namespaces for a reference capture.

    Replicator IDs are render-product local and may change between frames.  A
    diagnostic calibrated from a single static mapping can therefore report a
    false absence.  The v2 index points at the JSONL stream; each row is kept
    local and reduced to anonymous target-slot -> camera-local IDs before use.
    """

    metadata_path = reference_capture_root / "learning_labels" / "semantic_metadata.json"
    frame_path = reference_capture_root / "learning_labels" / "semantic_frame_metadata.jsonl"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetVisibilityDiagnosisError("reference semantic metadata is unreadable") from exc
    frame_contract = metadata.get("frame_metadata") if isinstance(metadata, Mapping) else None
    if not isinstance(frame_contract, Mapping) or frame_contract.get("path") != (
        "learning_labels/semantic_frame_metadata.jsonl"
    ):
        raise TargetVisibilityDiagnosisError("reference semantic metadata has no frame-aligned mapping contract")
    if frame_contract.get("frame_count") != frame_count:
        raise TargetVisibilityDiagnosisError("reference semantic frame count does not match camera poses")
    rows: list[dict[str, Any]] = []
    try:
        with frame_path.open("r", encoding="utf-8", newline="") as stream:
            for raw_line in stream:
                if raw_line.strip():
                    record = json.loads(raw_line)
                    if isinstance(record, Mapping):
                        rows.append(dict(record))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetVisibilityDiagnosisError("reference semantic frame metadata is unreadable") from exc
    if len(rows) != frame_count:
        raise TargetVisibilityDiagnosisError("reference semantic frame metadata is incomplete")
    result: list[dict[str, tuple[tuple[int, ...], ...]]] = []
    for frame_index, row in enumerate(rows):
        if row.get("frame_index") != frame_index:
            raise TargetVisibilityDiagnosisError("reference semantic frame metadata is out of order")
        result.append(_semantic_ids_by_slot(row.get("onboard_replicator_info"), target_count))
    return tuple(result)


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise TargetVisibilityDiagnosisError("reference capture contains no native target semantic pixels")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _calibrate_pixel_probe(
    reference_capture_root: Path,
    *,
    reference_private_manifest: Path,
) -> _PixelCalibration:
    """Calibrate an image-space probe from native semantic positives in r7-like evidence."""

    targets = _load_private_targets(reference_private_manifest)
    positions, quaternions = _load_finalized_pose_stream(reference_capture_root)
    semantic_path = reference_capture_root / "learning_labels" / "semantic_segmentation.npz"
    rgb_path = reference_capture_root / "sensors" / "onboard_rgbd.npz"
    residuals: list[float] = []
    colors: list[np.ndarray] = []
    depths: list[float] = []
    depth_residuals: list[float] = []
    positive_observations = 0
    positive_pixels = 0
    semantic_id_maps = _semantic_ids_by_frame(
        reference_capture_root,
        frame_count=positions.shape[0],
        target_count=len(targets),
    )
    try:
        with ChunkedFrameArchive(semantic_path) as semantic, ChunkedFrameArchive(rgb_path) as rgbd:
            rgb_descriptor = rgbd.descriptor("rgb")
            depth_descriptor = rgbd.descriptor("distance_to_image_plane_m")
            rgb_dtype, rgb_shape = rgb_descriptor.dtype, rgb_descriptor.shape
            depth_dtype, depth_shape = depth_descriptor.dtype, depth_descriptor.shape
            if (
                semantic.frame_count != positions.shape[0]
                or rgbd.frame_count != positions.shape[0]
                or rgb_shape[:2] != positions.shape[:2]
                or depth_shape[:2] != positions.shape[:2]
                or rgb_shape[2:4] != (ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH)
                or depth_shape[2:4] != (ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH)
                or rgb_shape[-1] != 3
                or depth_shape[-1] != 1
                or rgb_dtype != np.dtype(np.uint8)
                or not np.issubdtype(depth_dtype, np.floating)
            ):
                raise TargetVisibilityDiagnosisError("reference sensor streams disagree on frame geometry")
            for frame_index in range(semantic.frame_count):
                labels = semantic.frame("semantic_segmentation", frame_index)[..., 0]
                rgb = rgbd.frame("rgb", frame_index)
                depth = rgbd.frame("distance_to_image_plane_m", frame_index)
                semantic_ids = semantic_id_maps[frame_index]
                for target_index, target in enumerate(targets):
                    slot = f"{_SEMANTIC_SLOT_PREFIX}{target_index:03d}"
                    for camera_index, ids in enumerate(semantic_ids[slot]):
                        if not ids:
                            continue
                        mask = np.isin(labels[camera_index], np.asarray(ids, dtype=labels.dtype))
                        if not np.any(mask):
                            continue
                        projected = _project_target_center_px(
                            positions[frame_index, camera_index],
                            quaternions[frame_index, camera_index],
                            target.position_w_m,
                        )
                        if projected is None:
                            continue
                        row, column, optical_depth = projected
                        rows, columns = np.nonzero(mask)
                        centroid_row, centroid_column = float(np.mean(rows)), float(np.mean(columns))
                        residuals.append(math.hypot(centroid_row - row, centroid_column - column))
                        pixel_rgb = rgb[camera_index][mask].reshape(-1, 3)
                        pixel_depth = depth[camera_index, ..., 0][mask]
                        finite = np.isfinite(pixel_depth) & (pixel_depth > 0.0)
                        if pixel_rgb.size:
                            colors.append(np.median(pixel_rgb, axis=0).astype(np.float64))
                        if np.any(finite):
                            observed_depth = float(np.median(pixel_depth[finite]))
                            depths.append(observed_depth)
                            depth_residuals.append(abs(observed_depth - optical_depth))
                        positive_observations += 1
                        positive_pixels += int(np.count_nonzero(mask))
    except TargetVisibilityDiagnosisError:
        raise
    except (FrameArchiveError, OSError, KeyError, ValueError) as exc:
        raise TargetVisibilityDiagnosisError("reference target sensor archive is unreadable") from exc
    if not colors or not depths:
        raise TargetVisibilityDiagnosisError("reference capture has no usable target RGB-D positives")
    color_median = np.median(np.stack(colors), axis=0)
    color_residuals = [float(np.linalg.norm(color - color_median)) for color in colors]
    return _PixelCalibration(
        centroid_residual_q99_px=_quantile(residuals, 0.99),
        color_median_rgb=color_median,
        color_residual_q99=_quantile(color_residuals, 0.99),
        depth_residual_q99_m=_quantile(depth_residuals, 0.99),
        positive_observation_count=positive_observations,
        positive_pixel_count=positive_pixels,
    )


def _probe_failed_semantic_neighborhoods(
    capture_root: Path,
    *,
    targets: Sequence[_Target],
    positions: np.ndarray,
    quaternions: np.ndarray,
    calibration: _PixelCalibration,
) -> dict[str, dict[str, int]]:
    """Search calibrated private projections in retained raw failed-capture frames.

    The probe is diagnostic only.  It neither changes a gate nor makes a
    visibility claim: it counts evidence categories inside a radius learned
    from a separate native-positive capture.
    """

    spool = capture_root / ".sensor_spool_v1"
    try:
        rgb = np.load(spool / "onboard_rgb.npy", mmap_mode="r", allow_pickle=False)
        depth = np.load(spool / "depth_m.npy", mmap_mode="r", allow_pickle=False)
        semantic = np.load(spool / "semantic.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TargetVisibilityDiagnosisError("failed capture has no readable raw RGB-D-semantic spool") from exc
    if (
        rgb.shape[:2] != positions.shape[:2]
        or depth.shape[:2] != positions.shape[:2]
        or semantic.shape[:2] != positions.shape[:2]
        or rgb.shape[2:4] != (ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH)
        or depth.shape[2:4] != (ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH)
        or semantic.shape[2:4] != (ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH)
    ):
        raise TargetVisibilityDiagnosisError("failed capture raw sensor spool disagrees with camera poses")
    radius_px = max(2, int(math.ceil(calibration.centroid_residual_q99_px + 2.0)))
    color_threshold = calibration.color_residual_q99 + 1.0
    depth_threshold = calibration.depth_residual_q99_m + 0.05
    result: dict[str, dict[str, int]] = {}
    for target_index, target in enumerate(targets):
        projected_windows = 0
        in_image_windows = 0
        color_matches = 0
        depth_matches = 0
        joint_matches = 0
        nonzero_semantic_windows = 0
        for frame_index in range(positions.shape[0]):
            for camera_index in range(positions.shape[1]):
                projected = _project_target_center_px(
                    positions[frame_index, camera_index], quaternions[frame_index, camera_index], target.position_w_m
                )
                if projected is None:
                    continue
                row, column, optical_depth = projected
                projected_windows += 1
                row0, row1 = max(0, int(math.floor(row - radius_px))), min(ONBOARD_IMAGE_HEIGHT, int(math.ceil(row + radius_px + 1)))
                column0, column1 = max(0, int(math.floor(column - radius_px))), min(ONBOARD_IMAGE_WIDTH, int(math.ceil(column + radius_px + 1)))
                if row0 >= row1 or column0 >= column1:
                    continue
                in_image_windows += 1
                rgb_window = rgb[frame_index, camera_index, row0:row1, column0:column1].astype(np.float64)
                depth_window = depth[frame_index, camera_index, row0:row1, column0:column1, 0]
                semantic_window = semantic[frame_index, camera_index, row0:row1, column0:column1, 0]
                color_mask = np.linalg.norm(rgb_window - calibration.color_median_rgb, axis=-1) <= color_threshold
                depth_mask = np.isfinite(depth_window) & (np.abs(depth_window - optical_depth) <= depth_threshold)
                color_matches += int(np.count_nonzero(color_mask))
                depth_matches += int(np.count_nonzero(depth_mask))
                joint_matches += int(np.count_nonzero(color_mask & depth_mask))
                nonzero_semantic_windows += int(np.count_nonzero(semantic_window))
        result[f"{_SEMANTIC_SLOT_PREFIX}{target_index:03d}"] = {
            "projected_windows": projected_windows,
            "in_image_windows": in_image_windows,
            "color_match_pixels": color_matches,
            "depth_match_pixels": depth_matches,
            "joint_color_depth_match_pixels": joint_matches,
            "nonzero_semantic_pixels": nonzero_semantic_windows,
        }
    return result


def _native_visibility_summary(capture_root: Path, target_count: int) -> dict[str, dict[str, int]] | None:
    path = capture_root / "capture_progress.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    visibility = payload.get("target_visibility") if isinstance(payload, Mapping) else None
    rows = visibility.get("per_target_slot") if isinstance(visibility, Mapping) else None
    if not isinstance(rows, Mapping):
        return None
    result: dict[str, dict[str, int]] = {}
    for index in range(target_count):
        slot = f"{_SEMANTIC_SLOT_PREFIX}{index:03d}"
        row = rows.get(slot)
        if not isinstance(row, Mapping):
            return None
        frames, pixels = row.get("visible_frames"), row.get("max_pixels")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            return None
        if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 0:
            return None
        result[slot] = {"visible_sensor_frames": int(frames), "maximum_pixels": int(pixels)}
    return result


def _diagnostic_outcome(
    *,
    render_fov_samples: int,
    conservative_frustum_samples: int,
    render_fov_unoccluded_samples: int,
    render_fov_projected_samples: int,
    native_visible_frames: int | None,
) -> str:
    if render_fov_samples == 0:
        return "no_recorded_render_fov_witness"
    if conservative_frustum_samples == 0:
        if native_visible_frames and native_visible_frames > 0:
            return "native_semantic_visible_only_at_render_fov_edge"
        return "only_recorded_render_fov_edge_witnesses"
    if render_fov_unoccluded_samples == 0:
        return "all_recorded_render_fov_witnesses_structurally_blocked"
    if render_fov_projected_samples == 0:
        return "all_recorded_unblocked_render_fov_witnesses_below_projected_area_gate"
    if native_visible_frames == 0:
        return "native_semantic_absent_despite_recorded_geometry_witness"
    if native_visible_frames is None:
        return "recorded_geometry_witness_native_summary_unavailable"
    return "native_semantic_visible"


def diagnose_failed_target_visibility(
    capture_root: Path,
    *,
    private_manifest: Path,
    geometry_scan: Path,
    calibration_reference_capture: Path | None = None,
    calibration_reference_private_manifest: Path | None = None,
) -> dict[str, Any]:
    """Diagnose a terminal visibility failure without emitting evaluator truth.

    ``private_manifest`` is read locally and its coordinates stay in process.
    The returned JSON contains only anonymous slot identifiers and aggregate
    counts.  It is therefore useful for deciding whether a sampler needs a
    measured camera trajectory without turning target truth into public data.
    """

    capture_root = Path(capture_root).expanduser().resolve()
    private_manifest = Path(private_manifest).expanduser().resolve()
    geometry_scan = Path(geometry_scan).expanduser().resolve()
    if not capture_root.is_dir():
        raise TargetVisibilityDiagnosisError("capture root is missing")
    if not private_manifest.is_file():
        raise TargetVisibilityDiagnosisError("private manifest is missing")
    targets = _load_private_targets(private_manifest)
    positions, quaternions = _load_pose_stream(capture_root)
    catalog = load_native_geometry_catalog(geometry_scan)
    native = _native_visibility_summary(capture_root, len(targets))
    pose_provenance = _observed_pose_provenance(capture_root)
    focal_length_pixels = ONBOARD_FOCAL_LENGTH_MM / ONBOARD_HORIZONTAL_APERTURE_MM * ONBOARD_IMAGE_WIDTH
    total_pose_samples = int(positions.shape[0] * positions.shape[1])
    per_slot: dict[str, dict[str, Any]] = {}
    for target_index, target in enumerate(targets):
        slot = f"{_SEMANTIC_SLOT_PREFIX}{target_index:03d}"
        render_fov_samples = 0
        conservative_frustum_samples = 0
        render_fov_occluded_samples = 0
        render_fov_unoccluded_samples = 0
        render_fov_projected_samples = 0
        conservative_occluded_samples = 0
        conservative_unoccluded_samples = 0
        conservative_projected_samples = 0
        maximum_projected_pixels = 0.0
        for frame_index in range(positions.shape[0]):
            for camera_index in range(positions.shape[1]):
                camera = positions[frame_index, camera_index]
                quaternion = quaternions[frame_index, camera_index]
                in_render_fov, in_conservative_fov, distance = _camera_frustum_membership(
                    camera, quaternion, target.position_w_m
                )
                if not in_render_fov:
                    continue
                render_fov_samples += 1
                if in_conservative_fov:
                    conservative_frustum_samples += 1
                projected_pixels = math.pi * (focal_length_pixels * target.radius_m / distance) ** 2
                maximum_projected_pixels = max(maximum_projected_pixels, projected_pixels)
                structurally_blocked = any(
                    segment_intersects_aabb(camera, target.position_w_m, box, clearance_m=0.0)
                    for box in catalog.structural_aabbs
                )
                if structurally_blocked:
                    render_fov_occluded_samples += 1
                    if in_conservative_fov:
                        conservative_occluded_samples += 1
                    continue
                render_fov_unoccluded_samples += 1
                if in_conservative_fov:
                    conservative_unoccluded_samples += 1
                if projected_pixels >= TARGET_VISIBILITY_MIN_PROJECTED_INSTANCE_PIXELS:
                    render_fov_projected_samples += 1
                    if in_conservative_fov:
                        conservative_projected_samples += 1
        native_row = native.get(slot) if native is not None else None
        native_frames = None if native_row is None else native_row["visible_sensor_frames"]
        row: dict[str, Any] = {
            "recorded_pose_sample_count": total_pose_samples,
            "recorded_render_fov_samples": render_fov_samples,
            "recorded_conservative_frustum_samples": conservative_frustum_samples,
            "recorded_render_fov_edge_samples": render_fov_samples - conservative_frustum_samples,
            "recorded_render_fov_structurally_blocked_samples": render_fov_occluded_samples,
            "recorded_render_fov_unblocked_samples": render_fov_unoccluded_samples,
            "recorded_render_fov_projected_area_eligible_samples": render_fov_projected_samples,
            # Retained aliases preserve v2 consumers; their values are always
            # the strict inner-FOV gate, never the full rendered FOV.
            "recorded_frustum_eligible_samples": conservative_frustum_samples,
            "recorded_structurally_blocked_samples": conservative_occluded_samples,
            "recorded_unblocked_frustum_samples": conservative_unoccluded_samples,
            "recorded_projected_area_eligible_samples": conservative_projected_samples,
            "maximum_projected_instance_pixels": maximum_projected_pixels,
            "outcome": _diagnostic_outcome(
                render_fov_samples=render_fov_samples,
                conservative_frustum_samples=conservative_frustum_samples,
                render_fov_unoccluded_samples=render_fov_unoccluded_samples,
                render_fov_projected_samples=render_fov_projected_samples,
                native_visible_frames=native_frames,
            ),
        }
        if native_row is not None:
            row["native_semantic_visible_sensor_frames"] = native_row["visible_sensor_frames"]
            row["native_semantic_maximum_pixels"] = native_row["maximum_pixels"]
        per_slot[slot] = row
    spool = capture_root / ".sensor_spool_v1"
    report: dict[str, Any] = {
        "schema": DIAGNOSIS_SCHEMA,
        "claim_boundary": "aggregate_private_diagnosis_not_dataset_evidence",
        "capture": {
            "camera_pose_sha256": {
                "positions": _sha256_file(spool / "camera_observed_pos_w_m.npy"),
                "quaternions": _sha256_file(spool / "camera_observed_quat_wxyz.npy"),
            },
            "semantic_spool_present": (spool / "semantic.npy").is_file(),
            "native_semantic_summary_source": (
                "capture_progress_terminal_summary" if native is not None else "unavailable"
            ),
        },
        "private_inputs": {
            "manifest_sha256": _sha256_file(private_manifest),
            "target_count": len(targets),
        },
        "geometry": {
            "aabb_geometry_sha256": catalog.aabb_geometry_sha256,
            "native_scan_sha256": catalog.scan_sha256,
            "structural_aabb_count": len(catalog.structural_aabbs),
        },
        "camera_contract": {
            "observed_pose_source": pose_provenance["source"],
            "observed_pose_evidence": pose_provenance["evidence"],
            "optical_axis": "camera_local_plus_x",
            "image_right_axis": "camera_local_minus_y",
            "image_up_axis": "camera_local_plus_z",
            "render_fov": "physical_camera_intrinsics_full_field_of_view",
            "conservative_frustum_margin": TARGET_VISIBILITY_FRUSTUM_MARGIN,
            "legacy_frustum_fields_mean": "conservative_inner_fov_only",
            "minimum_distance_m": TARGET_VISIBILITY_MIN_DISTANCE_M,
            "maximum_distance_m": TARGET_VISIBILITY_MAX_DISTANCE_M,
            "minimum_projected_instance_pixels": TARGET_VISIBILITY_MIN_PROJECTED_INSTANCE_PIXELS,
        },
        "per_target_slot": per_slot,
    }
    if calibration_reference_capture is None and calibration_reference_private_manifest is None:
        return report
    if calibration_reference_capture is None or calibration_reference_private_manifest is None:
        raise TargetVisibilityDiagnosisError(
            "pixel probe requires both a reference capture and its private manifest"
        )
    reference_capture = Path(calibration_reference_capture).expanduser().resolve()
    reference_private = Path(calibration_reference_private_manifest).expanduser().resolve()
    if not reference_capture.is_dir() or not reference_private.is_file():
        raise TargetVisibilityDiagnosisError("pixel probe reference evidence is missing")
    calibration = _calibrate_pixel_probe(
        reference_capture, reference_private_manifest=reference_private
    )
    probe = _probe_failed_semantic_neighborhoods(
        capture_root,
        targets=targets,
        positions=positions,
        quaternions=quaternions,
        calibration=calibration,
    )
    report["calibrated_pixel_probe"] = {
        "schema": _CALIBRATED_PIXEL_PROBE_SCHEMA,
        "claim_boundary": "aggregate_diagnostic_not_visibility_or_rendering_evidence",
        "reference": {
            "capture_receipt_sha256": _sha256_file(reference_capture / "capture_receipt.json"),
            "private_manifest_sha256": _sha256_file(reference_private),
        },
        "calibration": {
            "native_positive_observation_count": calibration.positive_observation_count,
            "native_positive_pixel_count": calibration.positive_pixel_count,
            "centroid_residual_q99_px": calibration.centroid_residual_q99_px,
            "color_residual_q99": calibration.color_residual_q99,
            "depth_residual_q99_m": calibration.depth_residual_q99_m,
            "probe_radius_px": max(2, int(math.ceil(calibration.centroid_residual_q99_px + 2.0))),
        },
        "per_target_slot": probe,
    }
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--geometry-scan", type=Path, required=True)
    parser.add_argument("--calibration-reference-capture", type=Path)
    parser.add_argument("--calibration-reference-private-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = diagnose_failed_target_visibility(
            args.capture_root,
            private_manifest=args.private_manifest,
            geometry_scan=args.geometry_scan,
            calibration_reference_capture=args.calibration_reference_capture,
            calibration_reference_private_manifest=args.calibration_reference_private_manifest,
        )
        _atomic_json(args.output.expanduser().resolve(), report)
    except (OSError, TargetVisibilityDiagnosisError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "ok", "schema": DIAGNOSIS_SCHEMA}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
