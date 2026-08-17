"""Fail-closed policy-observation index for validated native T1 captures.

The raw capture is an audit bundle and contains streams that a policy must not
read.  This module writes an external, hash-bound index over an explicit
allow-list.  It copies no sensor payload and is deliberately only an
accidental-leakage guard; it is not an operating-system sandbox for hostile
policy code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .citylite_task import LIDAR_RAY_COUNT, ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH
from .formal_dataset import sha256_file
from .frame_archive import ChunkedFrameArchive, FrameArchiveError

POLICY_PROJECTION_SCHEMA = "org.rivermark.policy-observation-projection.v1"
POLICY_OBSERVATION_SCHEMA = "org.rivermark.policy-observation.v1"
T1_DATA_TRACK_ID = "t1-expert-coverage-multisensor-v1"
_CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
_VALIDATION_SCHEMA = "org.rivermark.isaac-independent-validation.v1"
_AGENT_COUNT = 8
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SPLITS = frozenset({"train", "inner_dev", "validation", "blind_test", "ood_test"})
_FORBIDDEN_PUBLIC_TOKENS = (
    "semantic",
    "learning_label",
    "target",
    "evaluator",
    "private",
    "ground_truth",
    "overview",
    "camera_pose",
    "contact",
)

_ONBOARD_PATH = "sensors/onboard_rgbd.npz"
_LIDAR_PATH = "sensors/lidar.npz"
_IMU_PATH = "sensors/imu.npz"
_STATE_PATH = "streams/state_action.npz"
_PUBLIC_TASK_PATH = "streams/public_task.npz"
_PUBLIC_MESSAGES_PATH = "streams/public_messages.npz"
_PARTITION_METADATA_PATH = "learning_labels/semantic_metadata.json"

_ONBOARD_FIELDS = frozenset({"rgb", "distance_to_image_plane_m"})
_LIDAR_FIELDS = frozenset({"timestamps_ns", "pos_w_m", "quat_wxyz", "ranges_m"})
_IMU_FIELDS = frozenset(
    {
        "timestamps_ns",
        "pos_w_m",
        "quat_wxyz",
        "linear_acceleration_b_mps2",
        "angular_velocity_b_radps",
    }
)
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
_PUBLIC_TASK_FIELDS = frozenset(
    {
        "timestamps_ns",
        "waypoint_index",
        "waypoint_progress",
        "desired_waypoint_w_m",
        "distance_to_waypoint_m",
        "waypoint_reached",
        "action_mode",
        "coverage_cell_id",
        "task_time_s",
    }
)
_PUBLIC_MESSAGE_FIELDS = frozenset(
    {
        "timestamps_ns",
        "sender_agent_id",
        "message_sequence",
        "message_waypoint_index",
        "message_position_w_m",
        "message_velocity_w_mps",
        "message_flags",
    }
)
_PUBLIC_BINDING_FIELDS = (
    "protocol_id",
    "protocol_sha256",
    "cell_id",
    "split",
    "episode_index",
    "episode_seed",
)


class PolicyProjectionError(ValueError):
    """Raised when a policy projection cannot prove its public boundary."""


@dataclass(frozen=True)
class PolicyProjectionResult:
    output_root: Path
    manifest_sha256: str
    observations_sha256: str
    observation_count: int


@dataclass(frozen=True)
class CandidateAbiIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PolicySourceInspection:
    """Read-only evidence summary for the frozen policy-visible source set."""

    capture_receipt_sha256: str
    independent_validation_sha256: str
    source_revision: str
    collection_binding: Mapping[str, Any]
    frame_count: int
    state_sample_count: int
    source_artifacts: tuple[Mapping[str, Any], ...]
    streams: Mapping[str, Any]


@dataclass(frozen=True)
class _ArrayDescriptor:
    dtype: np.dtype[Any]
    shape: tuple[int, ...]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value))


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyProjectionError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PolicyProjectionError(f"{label} must be a JSON object")
    return value


def _contained_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise PolicyProjectionError(f"required capture artifact is missing: {relative}")
    return candidate


def _verify_source_hash(
    receipt: Mapping[str, Any], root: Path, relative: str
) -> dict[str, Any]:
    path = _contained_file(root, relative)
    artifact_hashes = receipt.get("artifact_hashes")
    expected = artifact_hashes.get(relative) if isinstance(artifact_hashes, Mapping) else None
    if not isinstance(expected, Mapping):
        raise PolicyProjectionError(f"capture receipt does not bind {relative}")
    expected_hash = expected.get("sha256")
    expected_bytes = expected.get("bytes")
    actual_hash = sha256_file(path)
    if (
        not isinstance(expected_hash, str)
        or not _SHA256.fullmatch(expected_hash)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != path.stat().st_size
        or expected_hash != actual_hash
    ):
        raise PolicyProjectionError(f"capture artifact differs from its receipt: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": actual_hash}


def _public_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    raw = receipt.get("collection_binding")
    if not isinstance(raw, Mapping):
        raise PolicyProjectionError("capture has no protocol binding")
    result: dict[str, Any] = {}
    for field in _PUBLIC_BINDING_FIELDS:
        value = raw.get(field)
        if field == "episode_index":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PolicyProjectionError("protocol binding episode_index must be non-negative")
        elif field == "episode_seed":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 0xFFFFFFFF
            ):
                raise PolicyProjectionError("protocol binding episode_seed must be uint32")
        elif field == "protocol_sha256":
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise PolicyProjectionError("protocol binding hash is invalid")
        elif field == "split":
            if value not in _SPLITS:
                raise PolicyProjectionError("protocol binding split is invalid")
        elif (
            not isinstance(value, str)
            or not _ID.fullmatch(value)
            or any(token in value.lower() for token in _FORBIDDEN_PUBLIC_TOKENS)
        ):
            raise PolicyProjectionError(f"protocol binding {field} is not public-safe")
        result[field] = value
    return result


def _verify_capture_boundary(
    root: Path,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    receipt_path = _contained_file(root, "capture_receipt.json")
    validation_path = _contained_file(root, "independent_validation.json")
    receipt = dict(_load_json(receipt_path, "capture_receipt.json"))
    validation = _load_json(validation_path, "independent_validation.json")
    if (
        receipt.get("schema") != _CAPTURE_SCHEMA
        or receipt.get("status") != "captured"
        or receipt.get("ok") is not True
    ):
        raise PolicyProjectionError("policy projection requires a successful native capture")
    if receipt.get("source_worktree_dirty") is not False:
        raise PolicyProjectionError("policy projection requires a clean capture source")
    if not isinstance(receipt.get("source_revision"), str) or not _REVISION.fullmatch(
        receipt["source_revision"]
    ):
        raise PolicyProjectionError("capture source revision is not a Git commit hash")
    boundary = receipt.get("claim_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("formal_benchmark_admission") is not False
    ):
        raise PolicyProjectionError("policy projection is development-only")
    task = receipt.get("task")
    if (
        not isinstance(task, Mapping)
        or task.get("track") != T1_DATA_TRACK_ID
        or task.get("task_kind") != "expert_coverage_dataset"
        or task.get("scoring_status") != "not_scored"
    ):
        raise PolicyProjectionError("source capture is not the frozen T1 data track")
    physics = receipt.get("physics")
    if (
        not isinstance(physics, Mapping)
        or physics.get("same_world_agent_count") != _AGENT_COUNT
    ):
        raise PolicyProjectionError("policy projection requires exactly eight agents")
    receipt_hash = sha256_file(receipt_path)
    if (
        validation.get("schema") != _VALIDATION_SCHEMA
        or validation.get("status") != "passed"
        or validation.get("issues") != []
        or validation.get("capture_receipt_sha256") != receipt_hash
    ):
        raise PolicyProjectionError("independent validation is absent, failed, or stale")
    validation_hash = sha256_file(validation_path)
    partition_path = _contained_file(root, _PARTITION_METADATA_PATH)
    _verify_source_hash(receipt, root, _PARTITION_METADATA_PATH)
    partition = _load_json(partition_path, "partition metadata")
    if partition.get("partition") != "learning_labels" or partition.get("policy_visible") is not False:
        raise PolicyProjectionError("capture label partition is not explicitly isolated")
    return receipt, receipt_hash, validation_hash, _public_binding(receipt)


def _npz_descriptors(
    path: Path, expected_fields: frozenset[str]
) -> dict[str, _ArrayDescriptor]:
    try:
        with zipfile.ZipFile(path) as archive:
            member_names = [info.filename for info in archive.infolist()]
            expected_members = {f"{field}.npy" for field in expected_fields}
            if len(member_names) != len(set(member_names)) or set(member_names) != expected_members:
                raise PolicyProjectionError(
                    f"{path.name} members differ from the canonical allow-list: "
                    f"expected {sorted(expected_members)}, got {sorted(member_names)}"
                )
            names = {
                info.filename[:-4]: info.filename
                for info in archive.infolist()
                if info.filename.endswith(".npy") and "/" not in info.filename
            }
            if set(names) != expected_fields:
                raise PolicyProjectionError(
                    f"{path.name} fields differ from the allow-list: "
                    f"expected {sorted(expected_fields)}, got {sorted(names)}"
                )
            result: dict[str, _ArrayDescriptor] = {}
            for field, member_name in names.items():
                with archive.open(member_name) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version in {(2, 0), (3, 0)}:
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise PolicyProjectionError(
                            f"{path.name}.{field} uses unsupported NPY version {version}"
                        )
                normalized = np.dtype(dtype)
                if normalized.hasobject:
                    raise PolicyProjectionError(f"{path.name}.{field} has object dtype")
                result[field] = _ArrayDescriptor(normalized, tuple(shape))
            return result
    except PolicyProjectionError:
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise PolicyProjectionError(f"cannot inspect {path.name}: {exc}") from exc


def _npz_array(path: Path, field: str) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive[field]).copy()
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise PolicyProjectionError(f"cannot read {path.name}.{field}: {exc}") from exc


def _require_agent_shape(
    descriptor: _ArrayDescriptor,
    *,
    label: str,
    leading_count: int,
    tail: tuple[int, ...] | None = None,
) -> None:
    if len(descriptor.shape) < 2 or descriptor.shape[:2] != (leading_count, _AGENT_COUNT):
        raise PolicyProjectionError(
            f"{label} must begin with [{leading_count},{_AGENT_COUNT}], got {descriptor.shape}"
        )
    if tail is not None and descriptor.shape[2:] != tail:
        raise PolicyProjectionError(f"{label} must end with {tail}, got {descriptor.shape}")


def _require_dtype(
    descriptor: _ArrayDescriptor,
    *,
    label: str,
    expected: np.dtype[Any] | type[np.generic],
) -> None:
    expected_dtype = np.dtype(expected)
    if descriptor.dtype != expected_dtype:
        raise PolicyProjectionError(
            f"{label} must use {expected_dtype.name}, got {descriptor.dtype.name}"
        )


def _array_contract(descriptor: _ArrayDescriptor) -> dict[str, Any]:
    return {
        "dtype": descriptor.dtype.str,
        "shape": list(descriptor.shape),
    }


def _inspect_sources(
    root: Path, receipt: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    source_paths = {
        "onboard_rgbd": _ONBOARD_PATH,
        "lidar": _LIDAR_PATH,
        "imu": _IMU_PATH,
        "state_action": _STATE_PATH,
    }
    artifacts = {
        name: _verify_source_hash(receipt, root, relative)
        for name, relative in source_paths.items()
    }

    onboard_path = root / _ONBOARD_PATH
    try:
        with ChunkedFrameArchive(onboard_path) as onboard:
            if onboard.fields != {"timestamps_ns", *_ONBOARD_FIELDS}:
                raise PolicyProjectionError("onboard RGB-D fields differ from the allow-list")
            with zipfile.ZipFile(onboard_path) as raw_archive:
                member_names = [info.filename for info in raw_archive.infolist()]
            expected_members = {
                "__rivermark_chunked_frame_archive_v1__.npy",
                "__rivermark_frame_count__.npy",
                "timestamps_ns.npy",
                *(
                    f"{field}__frame__{frame_index:06d}.npy"
                    for field in _ONBOARD_FIELDS
                    for frame_index in range(onboard.frame_count)
                ),
            }
            if len(member_names) != len(set(member_names)) or set(member_names) != expected_members:
                raise PolicyProjectionError(
                    "onboard RGB-D members differ from the canonical allow-list"
                )
            timestamps = np.asarray(onboard.timestamps_ns, dtype=np.int64).copy()
            rgb = _ArrayDescriptor(
                onboard.descriptor("rgb").dtype,
                onboard.descriptor("rgb").shape,
            )
            depth = _ArrayDescriptor(
                onboard.descriptor("distance_to_image_plane_m").dtype,
                onboard.descriptor("distance_to_image_plane_m").shape,
            )
            for frame_index in range(onboard.frame_count):
                onboard.frame("rgb", frame_index)
                onboard.frame("distance_to_image_plane_m", frame_index)
    except FrameArchiveError as exc:
        raise PolicyProjectionError(f"invalid onboard RGB-D archive: {exc}") from exc
    if timestamps.ndim != 1 or len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0):
        raise PolicyProjectionError("sensor timestamps must be non-empty and increasing")
    frame_count = len(timestamps)
    _require_agent_shape(
        rgb,
        label="rgb",
        leading_count=frame_count,
        tail=(ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH, 3),
    )
    _require_dtype(rgb, label="rgb", expected=np.uint8)
    _require_agent_shape(
        depth,
        label="distance",
        leading_count=frame_count,
        tail=(ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH, 1),
    )
    _require_dtype(depth, label="distance", expected=np.float32)

    lidar_path = root / _LIDAR_PATH
    lidar = _npz_descriptors(lidar_path, _LIDAR_FIELDS)
    imu_path = root / _IMU_PATH
    imu = _npz_descriptors(imu_path, _IMU_FIELDS)
    for label, path in (("lidar", lidar_path), ("imu", imu_path)):
        source_timestamps = _npz_array(path, "timestamps_ns")
        if source_timestamps.dtype != np.int64 or not np.array_equal(source_timestamps, timestamps):
            raise PolicyProjectionError(f"{label} timestamps do not match onboard RGB-D")
    _require_agent_shape(
        lidar["ranges_m"],
        label="lidar ranges",
        leading_count=frame_count,
        tail=(LIDAR_RAY_COUNT,),
    )
    _require_agent_shape(lidar["pos_w_m"], label="lidar pose", leading_count=frame_count, tail=(3,))
    _require_agent_shape(lidar["quat_wxyz"], label="lidar orientation", leading_count=frame_count, tail=(4,))
    for field in ("pos_w_m", "quat_wxyz", "ranges_m"):
        _require_dtype(lidar[field], label=f"lidar {field}", expected=np.float32)
    _require_agent_shape(
        imu["linear_acceleration_b_mps2"],
        label="imu acceleration",
        leading_count=frame_count,
        tail=(3,),
    )
    _require_agent_shape(
        imu["angular_velocity_b_radps"],
        label="imu angular velocity",
        leading_count=frame_count,
        tail=(3,),
    )
    _require_agent_shape(imu["pos_w_m"], label="imu pose", leading_count=frame_count, tail=(3,))
    _require_agent_shape(imu["quat_wxyz"], label="imu orientation", leading_count=frame_count, tail=(4,))
    for field in (
        "pos_w_m",
        "quat_wxyz",
        "linear_acceleration_b_mps2",
        "angular_velocity_b_radps",
    ):
        _require_dtype(imu[field], label=f"imu {field}", expected=np.float32)

    state_path = root / _STATE_PATH
    state = _npz_descriptors(state_path, _STATE_FIELDS)
    state_effective = _npz_array(state_path, "effective_time_ns")
    state_command = _npz_array(state_path, "command_time_ns")
    if (
        state_effective.dtype != np.int64
        or state_command.dtype != np.int64
        or state_effective.ndim != 1
        or state_command.shape != state_effective.shape
        or np.any(np.diff(state_effective) <= 0)
        or np.any(state_command >= state_effective)
    ):
        raise PolicyProjectionError("state/action timing is invalid")
    state_indices = np.searchsorted(state_effective, timestamps)
    if (
        np.any(state_indices >= len(state_effective))
        or not np.array_equal(state_effective[state_indices], timestamps)
    ):
        raise PolicyProjectionError("sensor frames do not have exact post-step state rows")
    state_count = len(state_effective)
    for field, tail in (
        ("root_pos_w_m", (3,)),
        ("root_quat_wxyz", (4,)),
        ("root_lin_vel_w_mps", (3,)),
        ("root_ang_vel_b_radps", (3,)),
        ("desired_pos_w_m", (3,)),
        ("desired_vel_w_mps", (3,)),
        ("target_thrust_n", (4,)),
        ("applied_thrust_n", (4,)),
    ):
        _require_agent_shape(state[field], label=f"state {field}", leading_count=state_count, tail=tail)
        _require_dtype(state[field], label=f"state {field}", expected=np.float32)

    stream_contract = {
        "onboard_rgbd": {
            "path": _ONBOARD_PATH,
            "fields": sorted(_ONBOARD_FIELDS),
            "arrays": {
                "distance_to_image_plane_m": _array_contract(depth),
                "rgb": _array_contract(rgb),
            },
        },
        "lidar": {
            "path": _LIDAR_PATH,
            "fields": ["ranges_m"],
            "arrays": {"ranges_m": _array_contract(lidar["ranges_m"])},
        },
        "imu": {
            "path": _IMU_PATH,
            "fields": ["angular_velocity_b_radps", "linear_acceleration_b_mps2"],
            "arrays": {
                "angular_velocity_b_radps": _array_contract(
                    imu["angular_velocity_b_radps"]
                ),
                "linear_acceleration_b_mps2": _array_contract(
                    imu["linear_acceleration_b_mps2"]
                ),
            },
        },
        "state": {
            "path": _STATE_PATH,
            "fields": [
                "root_ang_vel_b_radps",
                "root_lin_vel_w_mps",
                "root_pos_w_m",
                "root_quat_wxyz",
            ],
            "arrays": {
                field: _array_contract(state[field])
                for field in (
                    "root_ang_vel_b_radps",
                    "root_lin_vel_w_mps",
                    "root_pos_w_m",
                    "root_quat_wxyz",
                )
            },
            "state_timing": "after_step_available_before_next_command",
        },
    }
    public_artifacts = [artifacts[name] for name in sorted(artifacts)]
    return timestamps, state_indices.astype(np.int64), public_artifacts, stream_contract


def _observation_id(
    *, receipt_sha256: str, timestamp_ns: int, frame_index: int, agent_id: int
) -> str:
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "schema": POLICY_OBSERVATION_SCHEMA,
                "receipt_sha256": receipt_sha256,
                "timestamp_ns": timestamp_ns,
                "frame_index": frame_index,
                "agent_id": agent_id,
            }
        )
    ).hexdigest()
    return f"obs-{digest[:32]}"


def inspect_policy_observation_sources(capture_root: Path) -> PolicySourceInspection:
    """Validate the T1 public source boundary without creating an index.

    This is the same byte-, field-, shape-, dtype-, timing-, and partition
    inspection used by :func:`project_policy_observations`.  It deliberately
    returns no arrays and never exposes learning-label or evaluator-private
    content.
    """

    capture = Path(capture_root).expanduser().resolve()
    if not capture.is_dir():
        raise PolicyProjectionError(f"capture directory is missing: {capture}")
    receipt, receipt_hash, validation_hash, binding = _verify_capture_boundary(capture)
    timestamps, _, artifacts, streams = _inspect_sources(capture, receipt)
    state_shape = streams["state"]["arrays"]["root_pos_w_m"]["shape"]
    return PolicySourceInspection(
        capture_receipt_sha256=receipt_hash,
        independent_validation_sha256=validation_hash,
        source_revision=str(receipt["source_revision"]),
        collection_binding=dict(binding),
        frame_count=len(timestamps),
        state_sample_count=int(state_shape[0]),
        source_artifacts=tuple(dict(item) for item in artifacts),
        streams={key: dict(value) for key, value in streams.items()},
    )


def inspect_candidate_pack_streams(
    capture_root: Path,
    *,
    inspection: PolicySourceInspection | None = None,
) -> dict[str, Any]:
    """Return the exact eight-stream T1 candidate contract without writing data."""

    capture = Path(capture_root).expanduser().resolve()
    if inspection is None:
        inspection = inspect_policy_observation_sources(capture)
    receipt = _load_json(_contained_file(capture, "capture_receipt.json"), "capture_receipt.json")
    if sha256_file(capture / "capture_receipt.json") != inspection.capture_receipt_sha256:
        raise PolicyProjectionError("source inspection belongs to another capture receipt")

    paths = {
        "task": _PUBLIC_TASK_PATH,
        "messages": _PUBLIC_MESSAGES_PATH,
        "state": _STATE_PATH,
    }
    for relative in paths.values():
        _verify_source_hash(receipt, capture, relative)
    task = _npz_descriptors(capture / _PUBLIC_TASK_PATH, _PUBLIC_TASK_FIELDS)
    messages = _npz_descriptors(capture / _PUBLIC_MESSAGES_PATH, _PUBLIC_MESSAGE_FIELDS)
    state = _npz_descriptors(capture / _STATE_PATH, _STATE_FIELDS)
    lidar = _npz_descriptors(capture / _LIDAR_PATH, _LIDAR_FIELDS)
    imu = _npz_descriptors(capture / _IMU_PATH, _IMU_FIELDS)

    frame_count = inspection.frame_count
    state_count = inspection.state_sample_count
    task_timestamps = _npz_array(capture / _PUBLIC_TASK_PATH, "timestamps_ns")
    message_timestamps = _npz_array(capture / _PUBLIC_MESSAGES_PATH, "timestamps_ns")
    command_times = _npz_array(capture / _STATE_PATH, "command_time_ns")
    effective_times = _npz_array(capture / _STATE_PATH, "effective_time_ns")
    if (
        task_timestamps.dtype != np.int64
        or task_timestamps.shape != (frame_count,)
        or message_timestamps.dtype != np.int64
        or message_timestamps.shape != (frame_count,)
        or not np.array_equal(task_timestamps, message_timestamps)
        or np.any(np.diff(task_timestamps) <= 0)
    ):
        raise PolicyProjectionError("public task/message timestamps are invalid or unequal")
    if (
        command_times.dtype != np.int64
        or command_times.shape != (state_count,)
        or effective_times.dtype != np.int64
        or effective_times.shape != (state_count,)
        or np.any(command_times >= effective_times)
    ):
        raise PolicyProjectionError("action command timestamps do not precede effective timestamps")

    for field, dtype, tail in (
        ("waypoint_index", np.int64, (8,)),
        ("waypoint_progress", np.float32, (8,)),
        ("desired_waypoint_w_m", np.float32, (8, 3)),
        ("distance_to_waypoint_m", np.float32, (8,)),
        ("waypoint_reached", np.bool_, (8,)),
        ("action_mode", np.int8, (8,)),
        ("coverage_cell_id", np.int64, (8,)),
        ("task_time_s", np.float32, (8,)),
    ):
        _require_agent_shape(task[field], label=f"task {field}", leading_count=frame_count, tail=tail[1:])
        _require_dtype(task[field], label=f"task {field}", expected=dtype)
    for field, dtype, tail in (
        ("sender_agent_id", np.int64, (8,)),
        ("message_sequence", np.int64, (8,)),
        ("message_waypoint_index", np.int64, (8,)),
        ("message_position_w_m", np.float32, (8, 3)),
        ("message_velocity_w_mps", np.float32, (8, 3)),
        ("message_flags", np.uint8, (8,)),
    ):
        _require_agent_shape(messages[field], label=f"message {field}", leading_count=frame_count, tail=tail[1:])
        _require_dtype(messages[field], label=f"message {field}", expected=dtype)
    for field in ("desired_pos_w_m", "desired_vel_w_mps"):
        _require_agent_shape(state[field], label=f"action {field}", leading_count=state_count, tail=(3,))
        _require_dtype(state[field], label=f"action {field}", expected=np.float32)

    observed = inspection.streams
    onboard = observed["onboard_rgbd"]
    return {
        "actions": {
            "path": _STATE_PATH,
            "modality": "high_level_action_history",
            "timestamp_field": "command_time_ns",
            "fields": [
                "command_time_ns",
                "effective_time_ns",
                "desired_pos_w_m",
                "desired_vel_w_mps",
            ],
            "arrays": {
                field: _array_contract(state[field])
                for field in (
                    "command_time_ns",
                    "effective_time_ns",
                    "desired_pos_w_m",
                    "desired_vel_w_mps",
                )
            },
        },
        "state": {
            **dict(observed["state"]),
            "modality": "proprioception",
            "timestamp_field": "effective_time_ns",
            "fields": [
                "effective_time_ns",
                "root_ang_vel_b_radps",
                "root_lin_vel_w_mps",
                "root_pos_w_m",
                "root_quat_wxyz",
            ],
            "arrays": {
                "effective_time_ns": _array_contract(state["effective_time_ns"]),
                **dict(observed["state"]["arrays"]),
            },
        },
        "task": {
            "path": _PUBLIC_TASK_PATH,
            "modality": "public_task_state",
            "timestamp_field": "timestamps_ns",
            "fields": sorted(_PUBLIC_TASK_FIELDS),
            "arrays": {
                field: _array_contract(task[field]) for field in sorted(_PUBLIC_TASK_FIELDS)
            },
        },
        "messages": {
            "path": _PUBLIC_MESSAGES_PATH,
            "modality": "public_team_messages",
            "timestamp_field": "timestamps_ns",
            "fields": sorted(_PUBLIC_MESSAGE_FIELDS),
            "arrays": {
                field: _array_contract(messages[field])
                for field in sorted(_PUBLIC_MESSAGE_FIELDS)
            },
        },
        "rgb": {
            "path": onboard["path"],
            "modality": "rgb",
            "timestamp_field": "timestamps_ns",
            "fields": ["timestamps_ns", "rgb"],
            "arrays": {
                "timestamps_ns": {"dtype": np.dtype(np.int64).str, "shape": [frame_count]},
                "rgb": onboard["arrays"]["rgb"],
            },
        },
        "depth": {
            "path": onboard["path"],
            "modality": "distance_to_image_plane",
            "timestamp_field": "timestamps_ns",
            "fields": ["timestamps_ns", "distance_to_image_plane_m"],
            "arrays": {
                "timestamps_ns": {"dtype": np.dtype(np.int64).str, "shape": [frame_count]},
                "distance_to_image_plane_m": onboard["arrays"]["distance_to_image_plane_m"]
            },
        },
        "lidar": {
            **dict(observed["lidar"]),
            "modality": "lidar",
            "timestamp_field": "timestamps_ns",
            "fields": ["timestamps_ns", "ranges_m"],
            "arrays": {
                "timestamps_ns": _array_contract(lidar["timestamps_ns"]),
                "ranges_m": _array_contract(lidar["ranges_m"]),
            },
        },
        "imu": {
            **dict(observed["imu"]),
            "modality": "imu",
            "timestamp_field": "timestamps_ns",
            "fields": [
                "timestamps_ns",
                "angular_velocity_b_radps",
                "linear_acceleration_b_mps2",
            ],
            "arrays": {
                "timestamps_ns": _array_contract(imu["timestamps_ns"]),
                "angular_velocity_b_radps": _array_contract(
                    imu["angular_velocity_b_radps"]
                ),
                "linear_acceleration_b_mps2": _array_contract(
                    imu["linear_acceleration_b_mps2"]
                ),
            },
        },
    }


def validate_candidate_abi_sources(
    abi: Mapping[str, Any],
    source_streams: Mapping[str, Any],
) -> tuple[CandidateAbiIssue, ...]:
    """Require exact agreement between a formal ABI and inspected source arrays."""

    issues: list[CandidateAbiIssue] = []
    streams = abi.get("streams")
    if not isinstance(streams, list):
        return (
            CandidateAbiIssue(
                "abi_streams",
                "observation_abi.streams",
                "ABI streams must be an array",
            ),
        )
    stream_ids = [
        stream.get("stream_id")
        for stream in streams
        if isinstance(stream, Mapping) and isinstance(stream.get("stream_id"), str)
    ]
    by_id = {
        stream.get("stream_id"): stream
        for stream in streams
        if isinstance(stream, Mapping) and isinstance(stream.get("stream_id"), str)
    }
    if len(stream_ids) != len(streams) or len(stream_ids) != len(set(stream_ids)):
        issues.append(
            CandidateAbiIssue(
                "abi_stream_set",
                "observation_abi.streams",
                "ABI stream IDs must be unique strings",
            )
        )
    if set(by_id) != set(source_streams):
        issues.append(
            CandidateAbiIssue(
                "abi_stream_set",
                "observation_abi.streams",
                "ABI stream IDs must exactly match the inspected source streams",
            )
        )
    for stream_id, contract in source_streams.items():
        abi_stream = by_id.get(stream_id)
        if not isinstance(abi_stream, Mapping):
            issues.append(
                CandidateAbiIssue(
                    "abi_stream_missing",
                    f"observation_abi.streams.{stream_id}",
                    "public source stream is not described",
                )
            )
            continue
        if abi_stream.get("partition") != "policy_visible":
            issues.append(
                CandidateAbiIssue(
                    "abi_partition_mismatch",
                    f"observation_abi.streams.{stream_id}.partition",
                    "candidate streams must be policy_visible",
                )
            )
        if abi_stream.get("modality") != contract.get("modality"):
            issues.append(
                CandidateAbiIssue(
                    "abi_modality_mismatch",
                    f"observation_abi.streams.{stream_id}.modality",
                    "ABI modality differs from the inspected source stream",
                )
            )
        fields = abi_stream.get("fields")
        if not isinstance(fields, list):
            issues.append(
                CandidateAbiIssue(
                    "abi_field_set",
                    f"observation_abi.streams.{stream_id}.fields",
                    "ABI fields must be an array",
                )
            )
            continue
        field_names = [
            field.get("name")
            for field in fields
            if isinstance(field, Mapping) and isinstance(field.get("name"), str)
        ]
        abi_fields = {
            field.get("name"): field
            for field in fields
            if isinstance(field, Mapping) and isinstance(field.get("name"), str)
        }
        arrays = contract.get("arrays") if isinstance(contract, Mapping) else None
        if not isinstance(arrays, Mapping):
            continue
        if (
            len(field_names) != len(fields)
            or len(field_names) != len(set(field_names))
            or set(abi_fields) != set(arrays)
        ):
            issues.append(
                CandidateAbiIssue(
                    "abi_field_set",
                    f"observation_abi.streams.{stream_id}.fields",
                    "ABI fields must exactly match the inspected source arrays",
                )
            )
        for field_name, descriptor in arrays.items():
            path = f"observation_abi.streams.{stream_id}.{field_name}"
            abi_field = abi_fields.get(field_name)
            if not isinstance(abi_field, Mapping):
                issues.append(
                    CandidateAbiIssue(
                        "abi_field_missing",
                        path,
                        "public source array is not described",
                    )
                )
                continue
            if not isinstance(descriptor, Mapping):
                issues.append(CandidateAbiIssue("abi_source_descriptor", path, "source descriptor is invalid"))
                continue
            try:
                actual_dtype = np.dtype(str(descriptor.get("dtype"))).name
            except TypeError:
                issues.append(CandidateAbiIssue("abi_source_dtype", path, "source dtype is invalid"))
                continue
            if abi_field.get("dtype") != actual_dtype:
                issues.append(
                    CandidateAbiIssue(
                        "abi_dtype_mismatch",
                        path,
                        f"ABI declares {abi_field.get('dtype')!r}, source is {actual_dtype!r}",
                    )
                )
            if abi_field.get("timestamp_field") != contract.get("timestamp_field"):
                issues.append(
                    CandidateAbiIssue(
                        "abi_timestamp_mismatch",
                        path,
                        "ABI timestamp reference differs from the inspected source stream",
                    )
                )
            declared_shape = abi_field.get("shape")
            actual_shape = descriptor.get("shape")
            if (
                not isinstance(declared_shape, list)
                or not isinstance(actual_shape, list)
                or len(declared_shape) != len(actual_shape)
            ):
                issues.append(
                    CandidateAbiIssue(
                        "abi_shape_mismatch",
                        path,
                        "ABI and source ranks differ",
                    )
                )
                continue
            leading_dimension = (
                "physics_step" if stream_id in {"actions", "state"} else "sensor_frame"
            )
            expected_shape = [leading_dimension, *actual_shape[1:]]
            if declared_shape != expected_shape:
                issues.append(
                    CandidateAbiIssue(
                        "abi_shape_mismatch",
                        path,
                        "ABI symbolic leading dimension or fixed source dimensions differ",
                    )
                )
    return tuple(issues)


def _assert_public_output(value: Any, *, label: str) -> None:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True).lower()
    found = sorted(token for token in _FORBIDDEN_PUBLIC_TOKENS if token in serialized)
    if found:
        raise PolicyProjectionError(f"{label} contains forbidden public tokens: {found}")


def project_policy_observations(
    capture_root: Path,
    output_root: Path,
) -> PolicyProjectionResult:
    """Write a deterministic external index over the policy allow-list.

    The source capture remains untouched and no sensor array is copied.  A
    policy runner must still receive only the selected fields; giving arbitrary
    policy code filesystem access to the raw capture bypasses this guard.
    """

    capture = Path(capture_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not capture.is_dir():
        raise PolicyProjectionError(f"capture directory is missing: {capture}")
    if capture == destination or destination.is_relative_to(capture):
        raise PolicyProjectionError("projection output must be outside the source capture")
    if destination.exists():
        raise PolicyProjectionError(f"projection output already exists: {destination}")

    receipt, receipt_hash, validation_hash, binding = _verify_capture_boundary(capture)
    timestamps, state_indices, artifacts, streams = _inspect_sources(capture, receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        observation_path = staging / "observations.jsonl"
        observation_count = 0
        with observation_path.open("wb") as handle:
            for frame_index, timestamp_ns in enumerate(timestamps):
                for agent_id in range(_AGENT_COUNT):
                    record = {
                        "schema": POLICY_OBSERVATION_SCHEMA,
                        "observation_id": _observation_id(
                            receipt_sha256=receipt_hash,
                            timestamp_ns=int(timestamp_ns),
                            frame_index=frame_index,
                            agent_id=agent_id,
                        ),
                        "timestamp_ns": int(timestamp_ns),
                        "frame_index": frame_index,
                        "agent_id": agent_id,
                        "source_indices": {
                            "onboard_frame_index": frame_index,
                            "lidar_frame_index": frame_index,
                            "imu_frame_index": frame_index,
                            "state_step_index": int(state_indices[frame_index]),
                        },
                    }
                    _assert_public_output(record, label="observation record")
                    handle.write(_canonical_bytes(record))
                    observation_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        observations_hash = sha256_file(observation_path)
        manifest = {
            "schema": POLICY_PROJECTION_SCHEMA,
            "status": "projected",
            "development_only": True,
            "formal_benchmark_admission": False,
            "t2_score_permitted": False,
            "enforcement_scope": "allow-list provenance only; not an operating-system sandbox",
            "source_capture_receipt_sha256": receipt_hash,
            "independent_validation_sha256": validation_hash,
            "source_revision": receipt.get("source_revision"),
            "collection_binding": binding,
            "agent_count": _AGENT_COUNT,
            "frame_count": len(timestamps),
            "observation_count": observation_count,
            "observations": {
                "path": "observations.jsonl",
                "sha256": observations_hash,
                "bytes": observation_path.stat().st_size,
            },
            "source_artifacts": artifacts,
            "streams": streams,
            "causal_contract": {
                "observation_phase": "after_step",
                "next_command_phase": "after_observation",
                "source_id_use": "event_provenance_only",
            },
        }
        _assert_public_output(manifest, label="projection manifest")
        _write_json(staging / "projection_manifest.json", manifest)
        os.replace(staging, destination)
        staging = None  # type: ignore[assignment]
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return PolicyProjectionResult(
        output_root=destination,
        manifest_sha256=sha256_file(destination / "projection_manifest.json"),
        observations_sha256=observations_hash,
        observation_count=observation_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = project_policy_observations(args.capture_root, args.output_root)
    except (OSError, PolicyProjectionError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "projected",
                "development_only": True,
                "t2_score_permitted": False,
                "output_root": str(result.output_root),
                "manifest_sha256": result.manifest_sha256,
                "observations_sha256": result.observations_sha256,
                "observation_count": result.observation_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "POLICY_OBSERVATION_SCHEMA",
    "POLICY_PROJECTION_SCHEMA",
    "PolicyProjectionError",
    "PolicyProjectionResult",
    "PolicySourceInspection",
    "inspect_policy_observation_sources",
    "project_policy_observations",
]
