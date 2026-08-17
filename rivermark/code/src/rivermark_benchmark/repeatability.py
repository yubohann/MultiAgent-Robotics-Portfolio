"""Compare two independently validated, same-seed native Isaac captures.

The report measures bounded run-to-run variation.  It does not require
bitwise-identical RTX output and it does not admit either capture into the
formal dataset.
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

from .frame_archive import ChunkedFrameArchive, FrameArchiveError

REPEATABILITY_PROFILE_SCHEMA = "org.rivermark.isaac-repeatability-profile.v2"
REPEATABILITY_REPORT_SCHEMA = "org.rivermark.isaac-repeatability-report.v2"
SEMANTIC_FRAME_METADATA_SCHEMA = "org.rivermark.isaac-semantic-frame-metadata.v1"

_STATE_FIELDS = frozenset(
    {
        "command_time_ns",
        "effective_time_ns",
        "root_pos_w_m",
        "root_quat_wxyz",
        "root_lin_vel_w_mps",
        "root_ang_vel_b_radps",
        "desired_pos_w_m",
        "desired_vel_w_mps",
        "target_thrust_n",
        "applied_thrust_n",
    }
)
_IMU_FIELDS = frozenset(
    {
        "timestamps_ns",
        "pos_w_m",
        "quat_wxyz",
        "linear_acceleration_b_mps2",
        "angular_velocity_b_radps",
    }
)
_LIDAR_FIELDS = frozenset({"timestamps_ns", "pos_w_m", "quat_wxyz", "ranges_m"})
_BINDING_FIELDS = (
    "protocol_id",
    "protocol_sha256",
    "cell_id",
    "split",
    "episode_index",
    "episode_seed",
)
_CONFIGURATION_FIELDS = (
    "agent_count_requested",
    "command",
    "condition_request",
    "information_profile",
    "modalities",
    "city_lite_scene",
    "runtime_live",
    "target_visibility_execution_window",
)
_THRESHOLD_FIELDS = frozenset(
    {
        "root_position_max_error_m",
        "root_orientation_max_error_rad",
        "root_linear_velocity_max_error_mps",
        "root_angular_velocity_max_error_radps",
        "command_position_max_error_m",
        "command_velocity_max_error_mps",
        "thrust_max_abs_error_n",
        "imu_linear_acceleration_max_error_mps2",
        "imu_angular_velocity_max_error_radps",
        "lidar_mean_abs_error_m",
        "lidar_frame_p95_abs_error_m",
        "lidar_finite_mask_agreement_min",
        "rgb_frame_mean_abs_error_uint8",
        "rgb_frame_p95_abs_error_uint8",
        "depth_frame_mean_abs_error_m",
        "depth_frame_p95_abs_error_m",
        "depth_finite_mask_agreement_min",
        "semantic_frame_agreement_min",
        "target_visible_frame_delta_max",
    }
)
_USED_ARTIFACTS = (
    "streams/state_action.npz",
    "sensors/imu.npz",
    "sensors/lidar.npz",
    "sensors/onboard_rgbd.npz",
    "learning_labels/semantic_segmentation.npz",
    "learning_labels/semantic_frame_metadata.jsonl",
    "sensors/overview_rgb.npz",
    "task_outcome.json",
)


class RepeatabilityError(RuntimeError):
    """Raised when inputs cannot form an auditable repeatability pair."""


@dataclass(frozen=True)
class _Capture:
    root: Path
    receipt: Mapping[str, Any]
    receipt_sha256: str
    validation_sha256: str
    task_outcome: Mapping[str, Any]
    semantic_metadata_by_timestamp: Mapping[int, Mapping[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepeatabilityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RepeatabilityError(f"{label} must be a JSON object")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_profile(path: Path) -> tuple[Mapping[str, Any], str]:
    profile = _load_json(path, "repeatability profile")
    if set(profile) != {
        "schema",
        "profile_id",
        "claim",
        "semantic_identity",
        "thresholds",
    }:
        raise RepeatabilityError("repeatability profile has missing or unknown fields")
    if profile.get("schema") != REPEATABILITY_PROFILE_SCHEMA:
        raise RepeatabilityError("repeatability profile schema is unsupported")
    if not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]:
        raise RepeatabilityError("repeatability profile_id is invalid")
    if (
        profile.get("claim")
        != "bounded_same_seed_variation_with_canonical_semantics_not_bitwise_determinism"
    ):
        raise RepeatabilityError("repeatability profile claim boundary is invalid")
    if profile.get("semantic_identity") != {
        "source": "frame_aligned_public_id_to_labels",
        "key_fields": ["class", "agent_id"],
        "camera_local": True,
        "unmapped_id_policy": "fail_closed",
    }:
        raise RepeatabilityError("repeatability semantic identity contract is invalid")
    thresholds = profile.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != _THRESHOLD_FIELDS:
        raise RepeatabilityError("repeatability threshold set is incomplete or unknown")
    for name, raw in thresholds.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RepeatabilityError(f"repeatability threshold {name} must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise RepeatabilityError(f"repeatability threshold {name} is invalid")
        if name.endswith("_agreement_min") and value > 1.0:
            raise RepeatabilityError(f"repeatability threshold {name} exceeds one")
    return profile, _sha256(path)


def _verify_artifact(root: Path, receipt: Mapping[str, Any], relative: str) -> None:
    inventory = receipt.get("artifact_hashes")
    if not isinstance(inventory, Mapping) or not isinstance(
        inventory.get(relative), Mapping
    ):
        raise RepeatabilityError(f"capture receipt does not bind {relative}")
    path = root / relative
    entry = inventory[relative]
    if not path.is_file():
        raise RepeatabilityError(f"capture artifact is missing: {relative}")
    if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != _sha256(
        path
    ):
        raise RepeatabilityError(f"capture artifact is stale or modified: {relative}")


def _semantic_camera_maps(
    value: Any, *, expected_camera_count: int, label: str
) -> tuple[Mapping[int, tuple[str, str | None]], ...]:
    if not isinstance(value, Mapping) or set(value) != {"per_camera"}:
        raise RepeatabilityError(f"{label} must contain exactly per_camera")
    cameras = value.get("per_camera")
    if not isinstance(cameras, list) or len(cameras) != expected_camera_count:
        raise RepeatabilityError(
            f"{label} must contain exactly {expected_camera_count} camera maps"
        )
    result: list[Mapping[int, tuple[str, str | None]]] = []
    for camera_index, camera in enumerate(cameras):
        camera_label = f"{label}.per_camera[{camera_index}]"
        if not isinstance(camera, Mapping) or set(camera) != {"id_to_labels"}:
            raise RepeatabilityError(
                f"{camera_label} must contain exactly id_to_labels"
            )
        raw_mapping = camera.get("id_to_labels")
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            raise RepeatabilityError(f"{camera_label}.id_to_labels is empty")
        mapping: dict[int, tuple[str, str | None]] = {}
        for raw_id, raw_identity in raw_mapping.items():
            try:
                semantic_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise RepeatabilityError(
                    f"{camera_label} contains a non-integer semantic ID"
                ) from exc
            if (
                not isinstance(raw_id, str)
                or semantic_id < 0
                or raw_id != str(semantic_id)
                or not isinstance(raw_identity, Mapping)
                or set(raw_identity) - {"class", "agent_id"}
                or not isinstance(raw_identity.get("class"), str)
            ):
                raise RepeatabilityError(
                    f"{camera_label} contains a malformed public semantic identity"
                )
            agent_id = raw_identity.get("agent_id")
            if agent_id is not None and (
                not isinstance(agent_id, str)
                or not agent_id.isdecimal()
                or not 0 <= int(agent_id) < 8
            ):
                raise RepeatabilityError(
                    f"{camera_label} contains a malformed public agent_id"
                )
            mapping[semantic_id] = (raw_identity["class"], agent_id)
        result.append(mapping)
    return tuple(result)


def _load_semantic_metadata(
    path: Path,
) -> Mapping[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    expected_keys = {
        "schema",
        "frame_index",
        "timestamp_ns",
        "onboard_replicator_info",
        "overview_replicator_info",
    }
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        raise RepeatabilityError(f"cannot read semantic frame metadata: {exc}") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise RepeatabilityError("semantic frame metadata contains a blank row")
            try:
                record = json.loads(
                    raw_line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(value)
                    ),
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RepeatabilityError(
                    f"semantic frame metadata row {line_number} is invalid: {exc}"
                ) from exc
            if (
                not isinstance(record, Mapping)
                or set(record) != expected_keys
                or record.get("schema") != SEMANTIC_FRAME_METADATA_SCHEMA
                or isinstance(record.get("frame_index"), bool)
                or record.get("frame_index") != line_number - 1
                or isinstance(record.get("timestamp_ns"), bool)
                or not isinstance(record.get("timestamp_ns"), int)
                or record["timestamp_ns"] < 0
            ):
                raise RepeatabilityError(
                    f"semantic frame metadata row {line_number} has an invalid contract"
                )
            timestamp_ns = int(record["timestamp_ns"])
            if timestamp_ns in rows:
                raise RepeatabilityError("semantic frame metadata timestamps are not unique")
            _semantic_camera_maps(
                record["onboard_replicator_info"],
                expected_camera_count=8,
                label=f"semantic row {line_number}.onboard_replicator_info",
            )
            _semantic_camera_maps(
                record["overview_replicator_info"],
                expected_camera_count=1,
                label=f"semantic row {line_number}.overview_replicator_info",
            )
            rows[timestamp_ns] = record
    if not rows:
        raise RepeatabilityError("semantic frame metadata is empty")
    return rows


def _load_capture(path: Path) -> _Capture:
    root = Path(path).expanduser().resolve()
    receipt_path = root / "capture_receipt.json"
    validation_path = root / "independent_validation.json"
    receipt = _load_json(receipt_path, "capture receipt")
    validation = _load_json(validation_path, "independent validation")
    receipt_sha256 = _sha256(receipt_path)
    if (
        receipt.get("schema") != "org.rivermark.isaac-swarm-capture.v1"
        or receipt.get("status") != "captured"
        or receipt.get("ok") is not True
        or receipt.get("source_worktree_dirty") is not False
    ):
        raise RepeatabilityError(
            "repeatability requires a successful clean-source capture"
        )
    validation_passed = (
        validation.get("valid") is True or validation.get("status") == "passed"
    )
    if (
        not validation_passed
        or validation.get("issues") != []
        or validation.get("capture_receipt_sha256") != receipt_sha256
    ):
        raise RepeatabilityError("independent validation is absent, failed, or stale")
    for relative in _USED_ARTIFACTS:
        _verify_artifact(root, receipt, relative)
    return _Capture(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        validation_sha256=_sha256(validation_path),
        task_outcome=_load_json(root / "task_outcome.json", "task outcome"),
        semantic_metadata_by_timestamp=_load_semantic_metadata(
            root / "learning_labels/semantic_frame_metadata.jsonl"
        ),
    )


def _public_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    binding = receipt.get("collection_binding")
    if not isinstance(binding, Mapping):
        raise RepeatabilityError("capture collection binding is missing")
    result = {name: binding.get(name) for name in _BINDING_FIELDS}
    if any(value is None for value in result.values()):
        raise RepeatabilityError("capture collection binding is incomplete")
    return result


def _same_capture_contract(reference: _Capture, candidate: _Capture) -> dict[str, Any]:
    left = reference.receipt
    right = candidate.receipt
    binding = _public_binding(left)
    if binding != _public_binding(right):
        raise RepeatabilityError(
            "captures do not share the same protocol cell/index/seed binding"
        )
    required_pairs = {
        "source_revision": (left.get("source_revision"), right.get("source_revision")),
        "source_tree_sha256": (
            left.get("source_tree_sha256"),
            right.get("source_tree_sha256"),
        ),
        "evaluator_manifest_sha256": (
            left.get("evaluator_manifest_sha256"),
            right.get("evaluator_manifest_sha256"),
        ),
        "runtime_lock": (left.get("runtime_lock"), right.get("runtime_lock")),
        "city_lite_authority": (
            left.get("city_lite_authority"),
            right.get("city_lite_authority"),
        ),
        "simulator": (left.get("simulator"), right.get("simulator")),
        **{
            name: (left.get(name), right.get(name))
            for name in _CONFIGURATION_FIELDS
        },
    }
    for label, (first, second) in required_pairs.items():
        if first is None or first != second:
            raise RepeatabilityError(f"captures disagree on {label}")
    if not isinstance(left.get("evaluator_manifest_sha256"), str):
        raise RepeatabilityError("capture evaluator manifest commitment is missing")
    configuration = {name: left[name] for name in _CONFIGURATION_FIELDS}
    return {
        "collection_binding": binding,
        "source_revision": left["source_revision"],
        "source_tree_sha256": left["source_tree_sha256"],
        "evaluator_manifest_sha256": left["evaluator_manifest_sha256"],
        "runtime_lock": left["runtime_lock"],
        "capture_configuration_sha256": _canonical_sha256(configuration),
    }


def _load_npz(path: Path, expected_fields: frozenset[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields:
                raise RepeatabilityError(
                    f"{path.name} fields differ from the repeatability allow-list"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except RepeatabilityError:
        raise
    except (OSError, EOFError, ValueError) as exc:
        raise RepeatabilityError(f"cannot load {path.name}: {exc}") from exc
    if any(
        value.dtype.hasobject or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise RepeatabilityError(f"{path.name} contains object or non-finite arrays")
    return arrays


def _matching_contract(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RepeatabilityError(f"{label} shape or dtype differs between captures")


def _limit(
    value: float, threshold: float, *, lower_bound: bool = False
) -> dict[str, Any]:
    passed = value >= threshold if lower_bound else value <= threshold
    return {
        "value": float(value),
        "threshold": float(threshold),
        "comparison": "greater_than_or_equal" if lower_bound else "less_than_or_equal",
        "passed": bool(passed),
    }


def _vector_error(
    left: np.ndarray, right: np.ndarray, threshold: float, label: str
) -> dict[str, Any]:
    _matching_contract(left, right, label)
    delta = np.linalg.norm(left.astype(np.float64) - right.astype(np.float64), axis=-1)
    return {
        **_limit(float(np.max(delta)), threshold),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _scalar_error(
    left: np.ndarray, right: np.ndarray, threshold: float, label: str
) -> dict[str, Any]:
    _matching_contract(left, right, label)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        **_limit(float(np.max(delta)), threshold),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _quaternion_error(
    left: np.ndarray, right: np.ndarray, threshold: float, label: str
) -> dict[str, Any]:
    _matching_contract(left, right, label)
    if left.shape[-1] != 4:
        raise RepeatabilityError(f"{label} must contain WXYZ quaternions")
    first = left.astype(np.float64)
    second = right.astype(np.float64)
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(first_norm <= 0.0) or np.any(second_norm <= 0.0):
        raise RepeatabilityError(f"{label} contains a zero-norm quaternion")
    first /= first_norm
    second /= second_norm
    dots = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return {
        **_limit(float(np.max(angles)), threshold),
        "rmse": float(np.sqrt(np.mean(np.square(angles)))),
    }


def _compare_state(
    reference: _Capture, candidate: _Capture, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    left = _load_npz(reference.root / "streams/state_action.npz", _STATE_FIELDS)
    right = _load_npz(candidate.root / "streams/state_action.npz", _STATE_FIELDS)
    for name in ("command_time_ns", "effective_time_ns"):
        _matching_contract(left[name], right[name], f"state.{name}")
        if not np.array_equal(left[name], right[name]):
            raise RepeatabilityError(f"state.{name} is not identical")
    return {
        "root_position": _vector_error(
            left["root_pos_w_m"],
            right["root_pos_w_m"],
            float(thresholds["root_position_max_error_m"]),
            "state.root_pos_w_m",
        ),
        "root_orientation": _quaternion_error(
            left["root_quat_wxyz"],
            right["root_quat_wxyz"],
            float(thresholds["root_orientation_max_error_rad"]),
            "state.root_quat_wxyz",
        ),
        "root_linear_velocity": _vector_error(
            left["root_lin_vel_w_mps"],
            right["root_lin_vel_w_mps"],
            float(thresholds["root_linear_velocity_max_error_mps"]),
            "state.root_lin_vel_w_mps",
        ),
        "root_angular_velocity": _vector_error(
            left["root_ang_vel_b_radps"],
            right["root_ang_vel_b_radps"],
            float(thresholds["root_angular_velocity_max_error_radps"]),
            "state.root_ang_vel_b_radps",
        ),
        "desired_position": _vector_error(
            left["desired_pos_w_m"],
            right["desired_pos_w_m"],
            float(thresholds["command_position_max_error_m"]),
            "state.desired_pos_w_m",
        ),
        "desired_velocity": _vector_error(
            left["desired_vel_w_mps"],
            right["desired_vel_w_mps"],
            float(thresholds["command_velocity_max_error_mps"]),
            "state.desired_vel_w_mps",
        ),
        "target_thrust": _scalar_error(
            left["target_thrust_n"],
            right["target_thrust_n"],
            float(thresholds["thrust_max_abs_error_n"]),
            "state.target_thrust_n",
        ),
        "applied_thrust": _scalar_error(
            left["applied_thrust_n"],
            right["applied_thrust_n"],
            float(thresholds["thrust_max_abs_error_n"]),
            "state.applied_thrust_n",
        ),
    }


def _compare_imu(
    reference: _Capture, candidate: _Capture, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    left = _load_npz(reference.root / "sensors/imu.npz", _IMU_FIELDS)
    right = _load_npz(candidate.root / "sensors/imu.npz", _IMU_FIELDS)
    if not np.array_equal(left["timestamps_ns"], right["timestamps_ns"]):
        raise RepeatabilityError("IMU timestamps differ")
    return {
        "position": _vector_error(
            left["pos_w_m"],
            right["pos_w_m"],
            float(thresholds["root_position_max_error_m"]),
            "imu.pos_w_m",
        ),
        "orientation": _quaternion_error(
            left["quat_wxyz"],
            right["quat_wxyz"],
            float(thresholds["root_orientation_max_error_rad"]),
            "imu.quat_wxyz",
        ),
        "linear_acceleration": _vector_error(
            left["linear_acceleration_b_mps2"],
            right["linear_acceleration_b_mps2"],
            float(thresholds["imu_linear_acceleration_max_error_mps2"]),
            "imu.linear_acceleration_b_mps2",
        ),
        "angular_velocity": _vector_error(
            left["angular_velocity_b_radps"],
            right["angular_velocity_b_radps"],
            float(thresholds["imu_angular_velocity_max_error_radps"]),
            "imu.angular_velocity_b_radps",
        ),
    }


def _finite_frame_error(
    left: np.ndarray,
    right: np.ndarray,
    *,
    mean_threshold: float,
    p95_threshold: float,
    agreement_threshold: float | None = None,
) -> tuple[float, float, float | None]:
    _matching_contract(left, right, "frame field")
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    agreement = float(np.mean(left_finite == right_finite))
    common = left_finite & right_finite
    if not np.any(common):
        raise RepeatabilityError("frame pair has no common finite samples")
    delta = np.abs(left[common].astype(np.float64) - right[common].astype(np.float64))
    mean = float(np.mean(delta))
    p95 = float(np.percentile(delta, 95.0))
    if mean_threshold < 0.0 or p95_threshold < 0.0:
        raise RepeatabilityError("frame thresholds must be nonnegative")
    if agreement_threshold is not None and agreement_threshold > 1.0:
        raise RepeatabilityError("frame agreement threshold exceeds one")
    return mean, p95, agreement if agreement_threshold is not None else None


def _compare_lidar(
    reference: _Capture, candidate: _Capture, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    left = _load_npz(reference.root / "sensors/lidar.npz", _LIDAR_FIELDS)
    right = _load_npz(candidate.root / "sensors/lidar.npz", _LIDAR_FIELDS)
    if not np.array_equal(left["timestamps_ns"], right["timestamps_ns"]):
        raise RepeatabilityError("LiDAR timestamps differ")
    mean, p95, agreement = _finite_frame_error(
        left["ranges_m"],
        right["ranges_m"],
        mean_threshold=float(thresholds["lidar_mean_abs_error_m"]),
        p95_threshold=float(thresholds["lidar_frame_p95_abs_error_m"]),
        agreement_threshold=float(thresholds["lidar_finite_mask_agreement_min"]),
    )
    return {
        "position": _vector_error(
            left["pos_w_m"],
            right["pos_w_m"],
            float(thresholds["root_position_max_error_m"]),
            "lidar.pos_w_m",
        ),
        "orientation": _quaternion_error(
            left["quat_wxyz"],
            right["quat_wxyz"],
            float(thresholds["root_orientation_max_error_rad"]),
            "lidar.quat_wxyz",
        ),
        "range_mean_abs_error": _limit(
            mean, float(thresholds["lidar_mean_abs_error_m"])
        ),
        "range_p95_abs_error": _limit(
            p95, float(thresholds["lidar_frame_p95_abs_error_m"])
        ),
        "finite_mask_agreement": _limit(
            float(agreement),
            float(thresholds["lidar_finite_mask_agreement_min"]),
            lower_bound=True,
        ),
    }


def _frame_archive_pair(
    reference_path: Path,
    candidate_path: Path,
    *,
    expected_fields: set[str],
) -> tuple[ChunkedFrameArchive, ChunkedFrameArchive]:
    try:
        left = ChunkedFrameArchive(reference_path)
        right = ChunkedFrameArchive(candidate_path)
    except (OSError, ValueError, FrameArchiveError) as exc:
        raise RepeatabilityError(f"cannot open frame archive pair: {exc}") from exc
    try:
        if left.fields != expected_fields or right.fields != expected_fields:
            raise RepeatabilityError(
                "frame archive fields differ from the repeatability allow-list"
            )
        if left.frame_count != right.frame_count or not np.array_equal(
            left.timestamps_ns, right.timestamps_ns
        ):
            raise RepeatabilityError("frame archive counts or timestamps differ")
        for field in expected_fields - {"timestamps_ns"}:
            first = left.descriptor(field)
            second = right.descriptor(field)
            if first != second:
                raise RepeatabilityError(f"frame archive contract differs for {field}")
    except BaseException:
        left.close()
        right.close()
        raise
    return left, right


def _frame_abs_metrics(
    left: ChunkedFrameArchive,
    right: ChunkedFrameArchive,
    field: str,
) -> tuple[float, float]:
    max_frame_mean = 0.0
    max_frame_p95 = 0.0
    for frame_index in range(left.frame_count):
        mean, p95, _ = _finite_frame_error(
            left.frame(field, frame_index),
            right.frame(field, frame_index),
            mean_threshold=0.0,
            p95_threshold=0.0,
        )
        max_frame_mean = max(max_frame_mean, mean)
        max_frame_p95 = max(max_frame_p95, p95)
    return max_frame_mean, max_frame_p95


def _frame_finite_metrics(
    left: ChunkedFrameArchive,
    right: ChunkedFrameArchive,
    field: str,
) -> tuple[float, float, float]:
    max_frame_mean = 0.0
    max_frame_p95 = 0.0
    minimum_agreement = 1.0
    for frame_index in range(left.frame_count):
        mean, p95, agreement = _finite_frame_error(
            left.frame(field, frame_index),
            right.frame(field, frame_index),
            mean_threshold=0.0,
            p95_threshold=0.0,
            agreement_threshold=0.0,
        )
        max_frame_mean = max(max_frame_mean, mean)
        max_frame_p95 = max(max_frame_p95, p95)
        minimum_agreement = min(minimum_agreement, float(agreement))
    return max_frame_mean, max_frame_p95, minimum_agreement


def _canonical_semantic_codes(
    raw: np.ndarray,
    mapping: Mapping[int, tuple[str, str | None]],
    identity_codes: Mapping[tuple[str, str | None], int],
) -> np.ndarray:
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0):
        raise RepeatabilityError("semantic frame contains non-integer or negative IDs")
    unique, inverse = np.unique(raw, return_inverse=True)
    codes: list[int] = []
    for raw_id in unique:
        semantic_id = int(raw_id)
        if semantic_id not in mapping:
            raise RepeatabilityError(
                f"semantic frame contains ID {semantic_id} absent from frame metadata"
            )
        codes.append(identity_codes[mapping[semantic_id]])
    return np.asarray(codes, dtype=np.int32)[inverse]


def _semantic_label_agreement(
    left: ChunkedFrameArchive,
    right: ChunkedFrameArchive,
    reference: _Capture,
    candidate: _Capture,
    *,
    metadata_key: str,
    camera_count: int,
) -> float:
    minimum = 1.0
    for frame_index, raw_timestamp in enumerate(left.timestamps_ns):
        timestamp_ns = int(raw_timestamp)
        left_record = reference.semantic_metadata_by_timestamp.get(timestamp_ns)
        right_record = candidate.semantic_metadata_by_timestamp.get(timestamp_ns)
        if left_record is None or right_record is None:
            raise RepeatabilityError(
                "semantic archive timestamp has no frame-aligned metadata"
            )
        left_maps = _semantic_camera_maps(
            left_record.get(metadata_key),
            expected_camera_count=camera_count,
            label=f"reference semantic timestamp {timestamp_ns}.{metadata_key}",
        )
        right_maps = _semantic_camera_maps(
            right_record.get(metadata_key),
            expected_camera_count=camera_count,
            label=f"candidate semantic timestamp {timestamp_ns}.{metadata_key}",
        )
        first = left.frame("semantic_segmentation", frame_index)
        second = right.frame("semantic_segmentation", frame_index)
        _matching_contract(first, second, "semantic_segmentation")
        if camera_count == 1:
            first = first[np.newaxis, ...]
            second = second[np.newaxis, ...]
        elif first.ndim < 2 or first.shape[0] != camera_count:
            raise RepeatabilityError(
                "semantic frame camera dimension differs from frame metadata"
            )
        for camera_index in range(camera_count):
            identities = sorted(
                set(left_maps[camera_index].values())
                | set(right_maps[camera_index].values()),
                key=repr,
            )
            identity_codes = {
                identity: code for code, identity in enumerate(identities)
            }
            left_codes = _canonical_semantic_codes(
                first[camera_index], left_maps[camera_index], identity_codes
            )
            right_codes = _canonical_semantic_codes(
                second[camera_index], right_maps[camera_index], identity_codes
            )
            minimum = min(minimum, float(np.mean(left_codes == right_codes)))
    return minimum


def _compare_frame_archives(
    reference: _Capture, candidate: _Capture, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    onboard_left, onboard_right = _frame_archive_pair(
        reference.root / "sensors/onboard_rgbd.npz",
        candidate.root / "sensors/onboard_rgbd.npz",
        expected_fields={"timestamps_ns", "rgb", "distance_to_image_plane_m"},
    )
    try:
        rgb_mean, rgb_p95 = _frame_abs_metrics(onboard_left, onboard_right, "rgb")
        depth_mean, depth_p95, depth_agreement = _frame_finite_metrics(
            onboard_left, onboard_right, "distance_to_image_plane_m"
        )
    finally:
        onboard_left.close()
        onboard_right.close()

    semantic_left, semantic_right = _frame_archive_pair(
        reference.root / "learning_labels/semantic_segmentation.npz",
        candidate.root / "learning_labels/semantic_segmentation.npz",
        expected_fields={"timestamps_ns", "semantic_segmentation"},
    )
    try:
        semantic_agreement = _semantic_label_agreement(
            semantic_left,
            semantic_right,
            reference,
            candidate,
            metadata_key="onboard_replicator_info",
            camera_count=8,
        )
    finally:
        semantic_left.close()
        semantic_right.close()

    overview_left, overview_right = _frame_archive_pair(
        reference.root / "sensors/overview_rgb.npz",
        candidate.root / "sensors/overview_rgb.npz",
        expected_fields={
            "timestamps_ns",
            "camera_pos_w_m",
            "camera_quat_wxyz",
            "target_w_m",
            "rgb",
            "semantic_segmentation",
        },
    )
    try:
        for field in ("camera_pos_w_m", "camera_quat_wxyz", "target_w_m"):
            first = np.asarray(overview_left.array(field))
            second = np.asarray(overview_right.array(field))
            _matching_contract(first, second, f"overview.{field}")
            if not np.array_equal(first, second):
                raise RepeatabilityError(f"fixed overview field differs: {field}")
        overview_rgb_mean, overview_rgb_p95 = _frame_abs_metrics(
            overview_left, overview_right, "rgb"
        )
        overview_semantic_agreement = _semantic_label_agreement(
            overview_left,
            overview_right,
            reference,
            candidate,
            metadata_key="overview_replicator_info",
            camera_count=1,
        )
    finally:
        overview_left.close()
        overview_right.close()

    return {
        "onboard_rgb_frame_mean_abs_error": _limit(
            rgb_mean, float(thresholds["rgb_frame_mean_abs_error_uint8"])
        ),
        "onboard_rgb_frame_p95_abs_error": _limit(
            rgb_p95, float(thresholds["rgb_frame_p95_abs_error_uint8"])
        ),
        "onboard_depth_frame_mean_abs_error": _limit(
            depth_mean, float(thresholds["depth_frame_mean_abs_error_m"])
        ),
        "onboard_depth_frame_p95_abs_error": _limit(
            depth_p95, float(thresholds["depth_frame_p95_abs_error_m"])
        ),
        "onboard_depth_finite_mask_agreement": _limit(
            depth_agreement,
            float(thresholds["depth_finite_mask_agreement_min"]),
            lower_bound=True,
        ),
        "onboard_semantic_label_frame_agreement": _limit(
            semantic_agreement,
            float(thresholds["semantic_frame_agreement_min"]),
            lower_bound=True,
        ),
        "overview_rgb_frame_mean_abs_error": _limit(
            overview_rgb_mean, float(thresholds["rgb_frame_mean_abs_error_uint8"])
        ),
        "overview_rgb_frame_p95_abs_error": _limit(
            overview_rgb_p95, float(thresholds["rgb_frame_p95_abs_error_uint8"])
        ),
        "overview_semantic_label_frame_agreement": _limit(
            overview_semantic_agreement,
            float(thresholds["semantic_frame_agreement_min"]),
            lower_bound=True,
        ),
    }


def _compare_target_visibility(
    reference: _Capture, candidate: _Capture, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    left = reference.task_outcome.get("target_observability")
    right = candidate.task_outcome.get("target_observability")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise RepeatabilityError("T1 target observability summary is missing")
    if left.get("passed") is not True or right.get("passed") is not True:
        raise RepeatabilityError(
            "repeatability inputs must both pass target observability"
        )
    left_slots = left.get("per_target_slot")
    right_slots = right.get("per_target_slot")
    if not isinstance(left_slots, Mapping) or set(left_slots) != set(right_slots or {}):
        raise RepeatabilityError("target observability slots differ")
    deltas: dict[str, int] = {}
    for slot in sorted(left_slots):
        first = left_slots[slot]
        second = right_slots[slot]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise RepeatabilityError("target observability slot is malformed")
        first_count = first.get("visible_frames")
        second_count = second.get("visible_frames")
        if (
            isinstance(first_count, bool)
            or not isinstance(first_count, int)
            or isinstance(second_count, bool)
            or not isinstance(second_count, int)
            or first_count < 0
            or second_count < 0
        ):
            raise RepeatabilityError("target visible frame count is malformed")
        deltas[str(slot)] = abs(first_count - second_count)
    maximum = max(deltas.values(), default=0)
    return {
        **_limit(float(maximum), float(thresholds["target_visible_frame_delta_max"])),
        "per_target_slot_absolute_delta": deltas,
    }


def _resource_summary(capture: _Capture) -> dict[str, Any]:
    receipt = capture.receipt
    resource = receipt.get("resource_telemetry")
    maxima = resource.get("maxima") if isinstance(resource, Mapping) else None
    storage = receipt.get("capture_storage_budget")
    created = receipt.get("created_wall_time_ns")
    finished = receipt.get("finished_wall_time_ns")
    duration = None
    if isinstance(created, int) and isinstance(finished, int) and finished >= created:
        duration = (finished - created) / 1_000_000_000.0
    return {
        "capture_bytes": sum(
            path.stat().st_size for path in capture.root.rglob("*") if path.is_file()
        ),
        "duration_s": duration,
        "peak_system_commit_percent": maxima.get("commit_percent")
        if isinstance(maxima, Mapping)
        else None,
        "peak_process_private_commit_bytes": maxima.get("private_commit_bytes")
        if isinstance(maxima, Mapping)
        else None,
        "declared_required_storage_bytes": storage.get("required_bytes")
        if isinstance(storage, Mapping)
        else None,
    }


def _all_metric_results(value: Any) -> list[bool]:
    results: list[bool] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("passed"), bool):
            results.append(bool(value["passed"]))
        for nested in value.values():
            results.extend(_all_metric_results(nested))
    elif isinstance(value, list):
        for nested in value:
            results.extend(_all_metric_results(nested))
    return results


def build_repeatability_report(
    reference_root: Path,
    candidate_root: Path,
    *,
    profile_path: Path,
) -> dict[str, Any]:
    profile, profile_sha256 = _load_profile(Path(profile_path).expanduser().resolve())
    reference = _load_capture(reference_root)
    candidate = _load_capture(candidate_root)
    if reference.root == candidate.root:
        raise RepeatabilityError(
            "reference and candidate capture must be distinct directories"
        )
    binding = _same_capture_contract(reference, candidate)
    thresholds = profile["thresholds"]
    metrics = {
        "state_action": _compare_state(reference, candidate, thresholds),
        "imu": _compare_imu(reference, candidate, thresholds),
        "lidar": _compare_lidar(reference, candidate, thresholds),
        "frame_archives": _compare_frame_archives(reference, candidate, thresholds),
        "target_visibility": _compare_target_visibility(
            reference, candidate, thresholds
        ),
    }
    failures = [passed for passed in _all_metric_results(metrics) if not passed]
    status = "passed" if not failures else "failed"
    report = {
        "schema": REPEATABILITY_REPORT_SCHEMA,
        "status": status,
        "formal": False,
        "t2_score_permitted": False,
        "claim": profile["claim"],
        "analyzer": {
            "implementation": "rivermark_benchmark.repeatability",
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "semantic_comparison": "camera_local_frame_aligned_class_and_public_agent_id",
        },
        "profile": {
            "profile_id": profile["profile_id"],
            "sha256": profile_sha256,
            "semantic_identity": dict(profile["semantic_identity"]),
            "thresholds": dict(thresholds),
        },
        "binding": binding,
        "reference": {
            "capture_attempt_id": reference.receipt.get("capture_attempt_id"),
            "capture_receipt_sha256": reference.receipt_sha256,
            "independent_validation_sha256": reference.validation_sha256,
            "resources": _resource_summary(reference),
        },
        "candidate": {
            "capture_attempt_id": candidate.receipt.get("capture_attempt_id"),
            "capture_receipt_sha256": candidate.receipt_sha256,
            "independent_validation_sha256": candidate.validation_sha256,
            "resources": _resource_summary(candidate),
        },
        "metrics": metrics,
        "failed_metric_count": len(failures),
        "report_payload_sha256": "",
    }
    report["report_payload_sha256"] = _canonical_sha256(report)
    return report


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise RepeatabilityError(
            f"refusing to overwrite repeatability report: {destination}"
        )
    payload = dict(report)
    expected_digest = _canonical_sha256({**payload, "report_payload_sha256": ""})
    if payload.get("report_payload_sha256") != expected_digest:
        raise RepeatabilityError("repeatability report payload digest is stale")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = args.output.expanduser().resolve()
        for capture_root in (
            args.reference.expanduser().resolve(),
            args.candidate.expanduser().resolve(),
        ):
            try:
                output.relative_to(capture_root)
            except ValueError:
                continue
            raise RepeatabilityError(
                "repeatability output must remain outside both capture directories"
            )
        report = build_repeatability_report(
            args.reference, args.candidate, profile_path=args.profile
        )
        _write_new_report(output, report)
    except (OSError, ValueError, RepeatabilityError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_metric_count": report["failed_metric_count"],
                "output": str(args.output),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
