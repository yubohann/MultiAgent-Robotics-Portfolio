"""Independently validate a raw eight-agent Isaac capture.

The validator never imports Isaac, trusts no capture self-report, and does not
issue formal benchmark admission. It reopens every bound file, audits the
numeric contracts, and writes a hash-bound validation receipt beside the raw
capture only when all required checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .citylite_scene import (
    AUTHORITY_SHA256,
    CITY_LITE_COMMAND_VOLUME_W_M,
    CITY_LITE_FLIGHT_VOLUME_W_M,
    CityLiteAuthorityError,
    ENVIRONMENT_ID,
    EXPECTED_NATIVE_COLLISION_COUNTS,
    EXPECTED_UPSTREAM_PERMISSIONS,
    ROUTE_CLEARANCE_M,
    SCENE_CONTRACT_GATE_STATUS,
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SCHEMA,
    SCENE_CONTRACT_SHA256,
    SELECTIVE_REFERENCES,
    TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M,
    AABB,
    aabb_geometry_sha256,
    segment_intersects_aabb,
    validate_city_task_obstacle_material_closure_receipt,
    validate_rivermark_layer_inventory_receipt,
)
from .isaac_capture import (
    HOVER_THRUST_PER_ROTOR_N,
    IDENTITY_MARKER_RADIUS_M,
    INITIAL_HOVER_RPS,
    LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE,
    LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE,
    LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N,
    LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD,
    LITERAL_SPAWN_POSITION_TOLERANCE_M,
    LITERAL_USD_SPAWN_BASIS_LENGTH_TOLERANCE,
    LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD,
    LITERAL_USD_SPAWN_POSITION_TOLERANCE_M,
    MAX_CF2X_ANGULAR_VELOCITY_RADPS,
    MAX_CF2X_LINEAR_VELOCITY_MPS,
    ONBOARD_CAMERA_CLIPPING_RANGE_M,
    ONBOARD_CONTENT_GATE_SCHEMA,
    OVERVIEW_ARCHIVE_SCHEMA,
    OVERVIEW_ARCHIVE_STRIDE,
    OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M,
    OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
    OVERVIEW_WITNESS_POSITION_TOLERANCE_M,
    OVERVIEW_WITNESS_TRACKED_AGENT_ID,
    PrivateEvaluatorManifestError,
    SWARM_AGENT_LITERAL_PRIM_PATHS,
    T1_DATA_TRACK_ID,
    T1_OBSERVABILITY_OUTCOME_SCHEMA,
    TARGET_COUNT,
    TARGET_SEMANTIC_INSTANCE_PREFIX,
    PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS,
    PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES,
    RUNTIME_TARGET_USD_BOUND_EXTENT_TOLERANCE_M,
    RUNTIME_TARGET_USD_POSITION_TOLERANCE_M,
    RUNTIME_TARGET_USD_RADIUS_TOLERANCE_M,
    SEMANTIC_FRAME_METADATA_RELATIVE_PATH,
    SEMANTIC_FRAME_METADATA_SCHEMA,
    SEMANTIC_METADATA_SCHEMA,
    TASK_VARIANT_ID,
    VISUAL_INTRUSION_GATE_SCHEMA,
    _city_lite_spawn_states,
    _captured_frame_indices,
    _overview_archive_frame_indices,
    _onboard_visual_intrusion_evidence,
    _onboard_content_gate_contract,
    _onboard_scene_content_evidence,
    _overview_tracked_agent_visibility_evidence,
    _public_route_witness_schedule,
    _public_route_witness_view_at_time_ns,
    _visual_intrusion_gate_contract,
    validate_external_private_evaluator_manifest,
    validate_private_target_execution_window,
    validate_private_target_geometry,
)
from .citylite_task import (
    PUBLIC_ROUTE_WAYPOINT_SEGMENT_SECONDS,
    target_visibility_execution_window,
)
from .frame_archive import (
    LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES,
    ChunkedFrameArchive,
    FrameArchiveError,
    is_chunked_frame_archive,
    oversized_legacy_frame_members,
)
from .collection_protocol import validate_collection_binding
from .condition_realization import (
    CONDITION_REALIZATION_SCHEMA,
    evaluate_condition_realization,
    validate_condition_request,
)
from .private_evaluator_manifest import (
    PRIVATE_MANIFEST_RETENTION_KIND,
    PRIVATE_MANIFEST_RETENTION_MAX_BYTES,
)
from .isaac_runtime_safety import (
    CF2X_RUNTIME_GUARD_RADIUS_M,
    CONTACT_ABORT_FORCE_N,
    CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
    INTER_AGENT_BODY_ENVELOPE_SEPARATION_M,
    INTER_AGENT_MINIMUM_CENTER_SEPARATION_M,
    INTER_AGENT_PAIR_COUNT,
    INTER_AGENT_SAFETY_PROVENANCE,
    RUNTIME_SAFETY_SCHEMA,
    RUNTIME_SAFETY_FRAME_OUTCOME_CODES,
    RUNTIME_SAFETY_PHASE_CODES,
    RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
    RUNTIME_SAFETY_TRACE_SCHEMA,
    SENSOR_PHASE_EVENT_CODES,
    SENSOR_PHASE_EVENT_SEQUENCE,
    SENSOR_PHASE_SENSOR_NAMES,
    SENSOR_PHASE_TRACE_RELATIVE_PATH,
    SENSOR_PHASE_TRACE_SCHEMA,
    physics_time_ns,
    sensor_phase_array_digest,
)
from .runtime_lock import (
    RuntimeLockError,
    compare_live_simulation,
    load_runtime_lock,
    runtime_lock_sha256,
)
from .schema import (
    forbidden_policy_key,
    forbidden_policy_value_token,
    is_sha256,
    iter_tree,
    normalized_key,
)
from .video import sha256_file


VALIDATION_SCHEMA = "org.rivermark.isaac-independent-validation.v1"
CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
AGENT_COUNT = 8
COLLISION_PROXY_PRIM_ROOT = "/World/StaticScene/CollisionProxies"
COLLISION_PROXY_REPRESENTATION = "conservative_world_aabb"
EXPECTED_STAGE_UNITS = {
    "meters_per_unit": 1.0,
    "up_axis": "Z",
    "time_codes_per_second": 60.0,
    "frames_per_second": 60.0,
}
EXPECTED_FLIGHT_VOLUME = {"x": [-46.0, 46.0], "y": [-48.0, 44.0], "z": [8.9, 15.0]}
EXPECTED_COMMAND_VOLUME = {
    "x": [-46.0, 46.0],
    "y": [-48.0, 44.0],
    "z": [9.0, 14.25],
}
MIN_LIDAR_MAX_DISTANCE_M = 1.0
MAX_LIDAR_MAX_DISTANCE_M = 500.0
ONBOARD_CAMERA_USD_POSITION_TOLERANCE_M = 0.05
ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD = 0.01
ONBOARD_CAMERA_USD_FORWARD_COSINE_MIN = math.cos(ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD)
OVERVIEW_CONTENT_GATE_SCHEMA = "org.rivermark.isaac-overview-city-content-gate.v1"
OVERVIEW_CAMERA_NEAR_CLIP_M = 0.05
OVERVIEW_CAMERA_FAR_CLIP_M = 200.0
OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION = 0.99
OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION = 0.03
OVERVIEW_CONTENT_NEAR_SURFACE_M = 2.0
OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION = 0.20
OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M = 1.5
OVERVIEW_CONTENT_RGB_EDGE_DELTA = 8.0
OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION = 0.003
OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION = 0.001
OVERVIEW_STRUCTURAL_LABEL_TOKENS = (
    "building",
    "structure",
    "facade",
    "wall",
    "tower",
    "rubble",
    "debris",
)
_SAFE_COMMITMENT_KEYS = frozenset(
    {
        "evaluator_manifest_sha256",
        "evaluator_manifest_retention",
        "private_evaluator_manifest_sha256",
        "private_manifest_commitment_sha256",
    }
)
_PUBLIC_PRIVATE_TRUTH_KEYS = frozenset(
    {
        "candidate",
        "candidates",
        "candidate_pool",
        "candidate_pools",
        "private_target",
        "private_targets",
        "target",
        "targets",
    }
)
_CAPTURE_RECEIPT_NON_POLICY_METADATA_PATHS = frozenset(
    {
        "$.command.seed",
        "$.collection_binding.episode_seed",
        "$.provenance.legacy_route_target_trace_or_evaluator_migrated",
    }
)
_PUBLIC_PRIVATE_ARTIFACT_KEYS = frozenset(
    {
        "target_id",
        "target_ids",
        "position_w_m",
        "target_w_m",
        "evaluator_manifest_path",
        "private_manifest_path",
        "evaluator_private_path",
    }
)
_PUBLIC_PRIVATE_ARTIFACT_PATH_TOKENS = (
    "/world/searchtargets/",
    "evaluator_private",
    "evaluator-private",
    "private_manifest",
)
EXPECTED_ARTIFACTS = frozenset(
    {
        "calibration.json",
        "capture_progress.json",
        "learning_labels/semantic_metadata.json",
        SEMANTIC_FRAME_METADATA_RELATIVE_PATH,
        "learning_labels/semantic_segmentation.npz",
        "public_task.json",
        "scene.json",
        "task_outcome.json",
        "sensors/camera_poses.npz",
        "sensors/contact.npz",
        "sensors/imu.npz",
        "sensors/lidar.npz",
        "sensors/onboard_rgbd.npz",
        "sensors/overview_rgb.npz",
        "sensors/runtime_safety.npz",
        "sensors/sensor_phase.npz",
        "streams/public_messages.npz",
        "streams/public_task.npz",
        "streams/state_action.npz",
    }
)
# ``capture_start.json`` is a path-free crash-recovery/ledger control marker,
# not a public payload stream.  Native capture binds it for provenance, so the
# closed-world audit must allow this one control artifact without weakening the
# payload inventory below.
CONTROL_ARTIFACTS = frozenset({"capture_start.json"})
VIDEO_ARTIFACT_ROOT = "videos"
VIDEO_RECEIPT_SUFFIX = ".mp4.receipt.json"
VIDEO_RECEIPT_SCHEMAS = frozenset(
    {
        "org.rivermark.isaac-demo-video.v1",
        "org.rivermark.isaac-swarm-composite-video.v1",
    }
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class IsaacValidationReport:
    root: Path
    receipt_sha256: str | None
    checks: Mapping[str, Any]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def _issue(issues: list[ValidationIssue], code: str, path: str, message: str) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message))


def _read_json(path: Path, issues: list[ValidationIssue]) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "invalid_json", path.name, str(exc))
        return None
    if not isinstance(value, Mapping):
        _issue(issues, "json_type", path.name, "expected object")
        return None
    return value


_SEMANTIC_FRAME_RECORD_KEYS = frozenset(
    {
        "schema",
        "frame_index",
        "timestamp_ns",
        "onboard_replicator_info",
        "overview_replicator_info",
    }
)


def _validate_semantic_frame_id_mapping(
    value: Any,
    *,
    expected_camera_count: int,
    relative: str,
    issues: list[ValidationIssue],
) -> bool:
    """Validate the redacted, camera-local ID map stored for one frame."""

    valid = True
    if not isinstance(value, Mapping) or set(value) != {"per_camera"}:
        _issue(
            issues,
            "semantic_frame_metadata",
            relative,
            "semantic mapping must contain exactly per_camera",
        )
        return False
    per_camera = value.get("per_camera")
    if not isinstance(per_camera, list) or len(per_camera) != expected_camera_count:
        _issue(
            issues,
            "semantic_frame_metadata",
            relative,
            f"semantic mapping must contain exactly {expected_camera_count} camera maps",
        )
        return False
    for camera_index, camera_mapping in enumerate(per_camera):
        camera_path = f"{relative}.per_camera[{camera_index}]"
        if not isinstance(camera_mapping, Mapping) or set(camera_mapping) != {
            "id_to_labels"
        }:
            _issue(
                issues,
                "semantic_frame_metadata",
                camera_path,
                "camera mapping must contain only id_to_labels",
            )
            valid = False
            continue
        labels = camera_mapping.get("id_to_labels")
        if not isinstance(labels, Mapping) or not labels:
            _issue(
                issues,
                "semantic_frame_metadata",
                f"{camera_path}.id_to_labels",
                "camera ID mapping must be a non-empty object",
            )
            valid = False
            continue
        for raw_id, label in labels.items():
            try:
                semantic_id = int(raw_id)
            except (TypeError, ValueError):
                semantic_id = -1
            if (
                not isinstance(raw_id, str)
                or semantic_id < 0
                or raw_id != str(semantic_id)
            ):
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{camera_path}.id_to_labels",
                    "semantic IDs must be canonical non-negative decimal strings",
                )
                valid = False
            if not isinstance(label, Mapping) or set(label) - {"class", "agent_id"} or "class" not in label:
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{camera_path}.id_to_labels.{raw_id}",
                    "semantic labels may contain only class and public agent_id",
                )
                valid = False
                continue
            class_value = label.get("class")
            if not isinstance(class_value, str):
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{camera_path}.id_to_labels.{raw_id}.class",
                    "semantic class must be a string",
                )
                valid = False
            if "agent_id" in label:
                agent_id = label.get("agent_id")
                try:
                    agent_index = int(agent_id)
                except (TypeError, ValueError):
                    agent_index = -1
                if (
                    not isinstance(agent_id, (str, int))
                    or isinstance(agent_id, bool)
                    or not isinstance(class_value, str)
                    or "agent_identity" not in class_value.lower()
                    or not 0 <= agent_index < AGENT_COUNT
                    or str(agent_index) != str(agent_id)
                ):
                    _issue(
                        issues,
                        "semantic_frame_metadata",
                        f"{camera_path}.id_to_labels.{raw_id}.agent_id",
                        "agent_id must be a public integer identity marker in [0, 7]",
                    )
                    valid = False
    return valid


def _read_semantic_frame_metadata(
    path: Path,
    *,
    expected_timestamps: np.ndarray,
    private_target_ids: Sequence[str],
    issues: list[ValidationIssue],
) -> list[Mapping[str, Any]] | None:
    """Read and independently bind one semantic ID map to every raw frame."""

    rows: list[Mapping[str, Any]] = []
    valid = True
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        _issue(issues, "semantic_frame_metadata", path.as_posix(), str(exc))
        return None
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{path.as_posix()}:{line_number}",
                    "JSONL cannot contain blank records",
                )
                valid = False
                continue
            try:
                record = json.loads(
                    raw_line,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{path.as_posix()}:{line_number}",
                    f"invalid JSONL record: {exc}",
                )
                valid = False
                continue
            if not isinstance(record, Mapping):
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{path.as_posix()}:{line_number}",
                    "JSONL record must be an object",
                )
                valid = False
                continue
            rows.append(record)
            if set(record) != _SEMANTIC_FRAME_RECORD_KEYS:
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{path.as_posix()}:{line_number}",
                    "JSONL record fields are not exact",
                )
                valid = False
            if record.get("schema") != SEMANTIC_FRAME_METADATA_SCHEMA:
                _issue(
                    issues,
                    "semantic_frame_metadata",
                    f"{path.as_posix()}:{line_number}.schema",
                    "unexpected semantic frame metadata schema",
                )
                valid = False
            frame_index = record.get("frame_index")
            timestamp_ns = record.get("timestamp_ns")
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index != line_number - 1
                or frame_index >= len(expected_timestamps)
            ):
                _issue(
                    issues,
                    "semantic_frame_alignment",
                    f"{path.as_posix()}:{line_number}.frame_index",
                    "frame indices must be contiguous, ordered, and zero-based",
                )
                valid = False
            elif (
                isinstance(timestamp_ns, bool)
                or not isinstance(timestamp_ns, int)
                or timestamp_ns != int(expected_timestamps[frame_index])
            ):
                _issue(
                    issues,
                    "semantic_frame_alignment",
                    f"{path.as_posix()}:{line_number}.timestamp_ns",
                    "semantic metadata timestamp must equal the retained sensor timestamp",
                )
                valid = False
            valid = _validate_semantic_frame_id_mapping(
                record.get("onboard_replicator_info"),
                expected_camera_count=AGENT_COUNT,
                relative=f"{path.as_posix()}:{line_number}.onboard_replicator_info",
                issues=issues,
            ) and valid
            valid = _validate_semantic_frame_id_mapping(
                record.get("overview_replicator_info"),
                expected_camera_count=1,
                relative=f"{path.as_posix()}:{line_number}.overview_replicator_info",
                issues=issues,
            ) and valid
            if _private_target_id_leaks(record, private_target_ids):
                _issue(
                    issues,
                    "semantic_private_id_leakage",
                    f"{path.as_posix()}:{line_number}",
                    "semantic frame metadata contains an evaluator-private target identifier",
                )
                valid = False
    if len(rows) != len(expected_timestamps):
        _issue(
            issues,
            "semantic_frame_alignment",
            path.as_posix(),
            "semantic frame metadata row count does not match the retained sensor frame count",
        )
        valid = False
    return rows if valid else None


def _calibrated_lidar_max_distance_m(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> float | None:
    """Return the capture-bound LiDAR no-hit range, never a validator constant."""

    if calibration is None:
        return None
    lidar = calibration.get("lidar")
    if not isinstance(lidar, Mapping):
        _issue(
            issues,
            "lidar_calibration",
            "calibration.json.lidar",
            "LiDAR calibration with max_distance_m is required",
        )
        return None
    value = lidar.get("max_distance_m")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not (MIN_LIDAR_MAX_DISTANCE_M <= float(value) <= MAX_LIDAR_MAX_DISTANCE_M)
    ):
        _issue(
            issues,
            "lidar_calibration",
            "calibration.json.lidar.max_distance_m",
            f"LiDAR max_distance_m must be finite and in [{MIN_LIDAR_MAX_DISTANCE_M}, {MAX_LIDAR_MAX_DISTANCE_M}]",
        )
        return None
    return float(value)


def _calibrated_visual_intrusion_gate(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    """Read the declared RGB-D/LiDAR gate; raw inputs are rechecked later."""

    onboard = calibration.get("onboard_camera") if isinstance(calibration, Mapping) else None
    gate = onboard.get("visual_intrusion_gate") if isinstance(onboard, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("schema") != VISUAL_INTRUSION_GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("contract") != _visual_intrusion_gate_contract()
    ):
        _issue(
            issues,
            "visual_intrusion_calibration",
            "calibration.json.onboard_camera.visual_intrusion_gate",
            "RGB-D/LiDAR visual intrusion gate declaration is missing or inconsistent",
        )
        return None
    return gate


def _calibrated_onboard_content_gate(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    """Read the onboard scene-content declaration; raw frames are authoritative."""

    onboard = calibration.get("onboard_camera") if isinstance(calibration, Mapping) else None
    clipping = onboard.get("clipping_range_m") if isinstance(onboard, Mapping) else None
    if (
        not isinstance(clipping, list)
        or len(clipping) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in clipping)
        or not all(math.isfinite(float(value)) for value in clipping)
        or not math.isclose(float(clipping[0]), ONBOARD_CAMERA_CLIPPING_RANGE_M[0], abs_tol=1.0e-9)
        or not math.isclose(float(clipping[1]), ONBOARD_CAMERA_CLIPPING_RANGE_M[1], abs_tol=1.0e-9)
    ):
        _issue(
            issues,
            "onboard_content_calibration",
            "calibration.json.onboard_camera.clipping_range_m",
            "onboard clipping range must be the configured [0.05, 100.0] m",
        )
    gate = onboard.get("content_gate") if isinstance(onboard, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("schema") != ONBOARD_CONTENT_GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("contract") != _onboard_content_gate_contract()
    ):
        _issue(
            issues,
            "onboard_content_calibration",
            "calibration.json.onboard_camera.content_gate",
            "onboard scene-content gate declaration is missing or inconsistent",
        )
        return None
    return gate


def _calibrated_route_witness_schedule(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    """Require the immutable public schedule used by the Isaac route witness."""

    overview = calibration.get("overview_camera") if isinstance(calibration, Mapping) else None
    schedule = (
        overview.get("route_witness_schedule") if isinstance(overview, Mapping) else None
    )
    if not isinstance(schedule, Mapping) or schedule != _public_route_witness_schedule():
        _issue(
            issues,
            "route_witness_camera_calibration",
            "calibration.json.overview_camera.route_witness_schedule",
            "overview camera must use the exact public fixed-route witness schedule",
        )
        return None
    return schedule


def _calibrated_route_witness_visibility_gate(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    """Read the declared visibility contract; raw semantic frames are rechecked later."""

    overview = calibration.get("overview_camera") if isinstance(calibration, Mapping) else None
    gate = overview.get("tracked_agent_visibility_gate") if isinstance(overview, Mapping) else None
    initial = gate.get("initial_post_render") if isinstance(gate, Mapping) else None
    expected = {
        "schema": "org.rivermark.isaac-route-witness-agent-visibility.v1",
        "status": "passed",
        "tracked_agent_id": OVERVIEW_WITNESS_TRACKED_AGENT_ID,
        "minimum_tracked_agent_pixels": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
    }
    if (
        not isinstance(gate, Mapping)
        or any(gate.get(key) != value for key, value in expected.items())
        or not isinstance(initial, Mapping)
        or initial.get("schema")
        != "org.rivermark.isaac-route-witness-agent-visibility.v1"
        or initial.get("effective_time_ns") != 0
        or initial.get("witness_shot_index") != 0
        or initial.get("passed") is not True
        or initial.get("failures") != []
    ):
        _issue(
            issues,
            "route_witness_visibility_calibration",
            "calibration.json.overview_camera.tracked_agent_visibility_gate",
            "overview camera must declare the tracked-CF2X semantic visibility gate",
        )
        return None
    return gate


def _overview_content_gate_contract() -> dict[str, float]:
    """Return the immutable public thresholds for the overview evidence gate."""

    return {
        "minimum_finite_depth_fraction": OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION,
        "minimum_geometry_fraction": OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION,
        "near_surface_m": OVERVIEW_CONTENT_NEAR_SURFACE_M,
        "maximum_near_surface_fraction": OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION,
        "minimum_geometry_depth_span_m": OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M,
        "rgb_edge_delta": OVERVIEW_CONTENT_RGB_EDGE_DELTA,
        "minimum_rgb_edge_fraction": OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION,
        "minimum_structural_pixel_fraction": OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION,
    }


def _calibrated_overview_content_gate(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> tuple[float, Mapping[str, Any] | None]:
    """Read the overview render declaration but never trust its pass result."""

    if calibration is None:
        return OVERVIEW_CAMERA_FAR_CLIP_M, None
    overview = calibration.get("overview_camera")
    if not isinstance(overview, Mapping):
        _issue(
            issues,
            "overview_calibration",
            "calibration.json.overview_camera",
            "overview camera calibration is required",
        )
        return OVERVIEW_CAMERA_FAR_CLIP_M, None
    clipping = overview.get("clipping_range_m")
    far_clip_m = OVERVIEW_CAMERA_FAR_CLIP_M
    if (
        not isinstance(clipping, list)
        or len(clipping) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in clipping)
        or not all(math.isfinite(float(value)) for value in clipping)
        or float(clipping[0]) >= float(clipping[1])
        or not math.isclose(float(clipping[0]), OVERVIEW_CAMERA_NEAR_CLIP_M, abs_tol=1.0e-9)
        or not math.isclose(float(clipping[1]), OVERVIEW_CAMERA_FAR_CLIP_M, abs_tol=1.0e-9)
    ):
        _issue(
            issues,
            "overview_calibration",
            "calibration.json.overview_camera.clipping_range_m",
            "overview clipping range must be the configured [0.05, 200.0] m",
        )
    else:
        far_clip_m = float(clipping[1])
    if overview.get("data_types") != [
        "rgb",
        "distance_to_image_plane",
        "semantic_segmentation",
    ]:
        _issue(
            issues,
            "overview_calibration",
            "calibration.json.overview_camera.data_types",
            "overview must declare RGB, depth, and semantic render products",
        )
    gate = overview.get("content_gate")
    expected_contract = _overview_content_gate_contract()
    initial = gate.get("initial_post_render") if isinstance(gate, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("schema") != OVERVIEW_CONTENT_GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("contract") != expected_contract
        or not isinstance(initial, Mapping)
        or initial.get("schema") != OVERVIEW_CONTENT_GATE_SCHEMA
        or initial.get("passed") is not True
        or initial.get("failures") != []
    ):
        _issue(
            issues,
            "overview_content_calibration",
            "calibration.json.overview_camera.content_gate",
            "overview content-gate declaration is missing or inconsistent",
        )
        return far_clip_m, None
    return far_clip_m, gate


def _calibrated_overview_evidence_archive(
    calibration: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    """Validate the new low-rate overview declaration without accepting selection.

    Legacy full-rate captures have no declaration and remain readable.  A new
    archive must name its deterministic frame-index rule and explicitly state
    that overview depth was checked live but not retained.
    """

    overview = calibration.get("overview_camera") if isinstance(calibration, Mapping) else None
    archive = overview.get("evidence_archive") if isinstance(overview, Mapping) else None
    if archive is None:
        return None
    expected_fields = [
        "rgb",
        "semantic_segmentation",
        "camera_pos_w_m",
        "camera_quat_wxyz",
        "target_w_m",
    ]
    valid = bool(
        isinstance(archive, Mapping)
        and archive.get("schema") == OVERVIEW_ARCHIVE_SCHEMA
        and archive.get("selection_rule")
        == "first_each_fixed_retained_frame_stride_and_final"
        and archive.get("frame_index_stride") == OVERVIEW_ARCHIVE_STRIDE
        and isinstance(archive.get("source_frame_count"), int)
        and not isinstance(archive.get("source_frame_count"), bool)
        and archive["source_frame_count"] >= 2
        and isinstance(archive.get("source_frame_indices"), list)
        and archive.get("stored_fields") == expected_fields
        and archive.get("runtime_only_render_products")
        == ["distance_to_image_plane"]
        and archive.get("selection_uses_content_or_outcome") is False
    )
    if not valid:
        _issue(
            issues,
            "overview_archive_calibration",
            "calibration.json.overview_camera.evidence_archive",
            "overview archive must declare the frozen low-rate RGB/semantic witness schedule",
        )
        return None
    return archive


def _semantic_label_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_semantic_label_text(item)}" for key, item in value.items()
        ).lower()
    if isinstance(value, (list, tuple)):
        return " ".join(_semantic_label_text(item) for item in value).lower()
    return str(value).lower()


def _target_slots(target_count: int) -> tuple[str, ...]:
    """Build public capture-local slots without consulting private target IDs."""

    return tuple(
        f"{TARGET_SEMANTIC_INSTANCE_PREFIX}{index:03d}"
        for index in range(target_count)
    )


def _target_visibility_rollout_summary(
    target_slots: Sequence[str], evidence_samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the capture v2 public summary without reading its declaration.

    This deliberately duplicates the small public ABI reducer in the capture
    process.  The independent validator must be able to produce an identical
    result from reopened raw frames even if the capture process or its outcome
    writer is wrong.
    """

    observed: dict[str, dict[str, int]] = {
        target_slot: {"max_pixels": 0, "visible_frames": 0}
        for target_slot in target_slots
    }
    for evidence in evidence_samples:
        per_target_slot = evidence.get("per_target_slot")
        if not isinstance(per_target_slot, Mapping):
            continue
        for target_slot, row in per_target_slot.items():
            if target_slot not in observed or not isinstance(row, Mapping):
                continue
            observed[target_slot]["max_pixels"] = max(
                observed[target_slot]["max_pixels"],
                int(row.get("maximum_pixels_in_one_camera", 0)),
            )
            observed[target_slot]["visible_frames"] += int(
                row.get("visible_sensor_frames", 0)
            )
    failed_target_slots = [
        target_slot
        for target_slot, row in observed.items()
        if row["visible_frames"] < PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES
    ]
    return {
        "schema": "org.rivermark.isaac-target-visibility-summary.v2",
        "target_count": len(target_slots),
        "targets_meeting_visibility": len(target_slots) - len(failed_target_slots),
        "minimum_visible_sensor_frames_per_target": PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES,
        "minimum_visible_instance_pixels": PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS,
        "passed": not failed_target_slots,
        "failed_target_count": len(failed_target_slots),
        "failed_target_slots": failed_target_slots,
        "per_target_slot": observed,
    }


def _target_slot_semantic_ids(
    semantic_metadata: Any, target_slots: Sequence[str]
) -> dict[str, tuple[tuple[int, ...], ...]] | None:
    """Resolve slots in the distinct semantic-ID namespace of each camera."""

    per_camera = (
        semantic_metadata.get("per_camera")
        if isinstance(semantic_metadata, Mapping)
        else None
    )
    if not isinstance(per_camera, (list, tuple)) or len(per_camera) != AGENT_COUNT:
        return None
    found: dict[str, list[set[int]]] = {
        slot: [set() for _ in range(AGENT_COUNT)] for slot in target_slots
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
            class_labels = {
                item.strip().lower()
                for item in str(class_value).split(",")
                if item.strip()
            }
            for slot in target_slots:
                if slot.lower() in class_labels:
                    found[slot][camera_index].add(semantic_id)
    return {
        slot: tuple(tuple(sorted(ids)) for ids in per_camera_ids)
        for slot, per_camera_ids in found.items()
    }


def _recompute_target_observability(
    semantic: _FramePayload | None,
    semantic_frame_metadata: Sequence[Mapping[str, Any]] | None,
    *,
    target_count: int,
) -> dict[str, Any]:
    """Recompute the public T1 visibility summary from raw semantic frames.

    Replicator allocates numeric semantic IDs independently for each camera and
    may reassign them after a camera update.  The validator therefore resolves
    every slot inside each camera namespace for the matching retained frame,
    then builds the same public v2 rollout ABI from reopened sensor bytes.  It
    never consumes the capture's declared outcome while doing so.
    """

    slots = _target_slots(target_count)
    evidence_samples: list[dict[str, Any]] = []
    if semantic is None or "semantic_segmentation" not in semantic.fields:
        return _target_visibility_rollout_summary(slots, evidence_samples)
    elif semantic_frame_metadata is None or len(semantic_frame_metadata) != len(semantic.timestamps_ns):
        return _target_visibility_rollout_summary(slots, evidence_samples)
    else:
        dtype, shape = semantic.descriptor("semantic_segmentation")
        if (
            not np.issubdtype(dtype, np.integer)
            or len(shape) != 5
            or shape[1] != AGENT_COUNT
            or shape[-1] != 1
        ):
            return _target_visibility_rollout_summary(slots, evidence_samples)
        else:
            for frame_id in range(shape[0]):
                frame = semantic.frame("semantic_segmentation", frame_id)[..., 0]
                frame_metadata = semantic_frame_metadata[frame_id]
                semantic_ids = _target_slot_semantic_ids(
                    frame_metadata.get("onboard_replicator_info"), slots
                )
                per_target_slot: dict[str, dict[str, int]] = {}
                for slot in slots:
                    ids_by_camera = (
                        semantic_ids.get(slot)
                        if semantic_ids is not None
                        else tuple(() for _ in range(AGENT_COUNT))
                    )
                    pixel_counts = np.asarray(
                        [
                            np.count_nonzero(
                                np.isin(
                                    frame[camera_index],
                                    np.asarray(ids, dtype=frame.dtype)
                                    if ids
                                    else np.asarray([], dtype=frame.dtype),
                                )
                            )
                            if ids
                            else 0
                            for camera_index, ids in enumerate(ids_by_camera)
                        ],
                        dtype=np.int64,
                    )
                    maximum = int(np.max(pixel_counts)) if pixel_counts.size else 0
                    per_target_slot[slot] = {
                        "maximum_pixels_in_one_camera": maximum,
                        "visible_sensor_frames": int(
                            maximum >= PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS
                        ),
                    }
                evidence_samples.append({"per_target_slot": per_target_slot})
    return _target_visibility_rollout_summary(slots, evidence_samples)


def _private_target_id_leaks(value: Any, private_target_ids: Sequence[str]) -> bool:
    """Reject private evaluator identifiers copied into public semantic metadata."""

    tokens = tuple(
        token.lower()
        for token in private_target_ids
        if isinstance(token, str) and token.strip()
    )
    if not tokens:
        return False
    for _path, key, item in iter_tree(value):
        candidates = (key, item) if key is not None else (item,)
        if any(
            isinstance(candidate, str)
            and any(token in candidate.lower() for token in tokens)
            for candidate in candidates
        ):
            return True
    return False


def _overview_structural_semantic_ids(metadata: Any) -> tuple[tuple[int, ...], bool]:
    """Extract structural label IDs from Replicator's stable metadata variants."""

    structural_ids: set[int] = set()
    id_labels_seen = False

    def visit(value: Any) -> None:
        nonlocal id_labels_seen
        if isinstance(value, Mapping):
            for key in ("id_to_labels", "idToLabels"):
                labels = value.get(key)
                if not isinstance(labels, Mapping):
                    continue
                id_labels_seen = True
                for raw_id, label in labels.items():
                    try:
                        semantic_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    label_text = _semantic_label_text(label)
                    if any(token in label_text for token in OVERVIEW_STRUCTURAL_LABEL_TOKENS):
                        structural_ids.add(semantic_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(metadata)
    return tuple(sorted(structural_ids)), id_labels_seen


def _overview_city_content_evidence(
    rgb: np.ndarray,
    depth: np.ndarray,
    semantic: np.ndarray,
    semantic_metadata: Any,
    *,
    far_clip_m: float,
) -> dict[str, Any]:
    """Independently recompute City-Lite visual evidence from raw frame data."""

    luma = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )
    horizontal = np.abs(np.diff(luma, axis=1)) >= OVERVIEW_CONTENT_RGB_EDGE_DELTA
    vertical = np.abs(np.diff(luma, axis=0)) >= OVERVIEW_CONTENT_RGB_EDGE_DELTA
    rgb_edge_fraction = float(
        (np.count_nonzero(horizontal) + np.count_nonzero(vertical))
        / (horizontal.size + vertical.size)
    )
    plane_depth = depth[..., 0]
    finite_depth = np.isfinite(plane_depth)
    finite_depth_fraction = float(np.mean(finite_depth))
    background_margin_m = max(0.05, far_clip_m * 1.0e-3)
    geometry = finite_depth & (plane_depth >= 0.0) & (
        plane_depth < far_clip_m - background_margin_m
    )
    geometry_fraction = float(np.mean(geometry))
    near_surface_fraction = float(
        np.mean(geometry & (plane_depth <= OVERVIEW_CONTENT_NEAR_SURFACE_M))
    )
    if np.any(geometry):
        geometry_values = plane_depth[geometry]
        geometry_depth_span_m = float(
            np.percentile(geometry_values, 95) - np.percentile(geometry_values, 5)
        )
    else:
        geometry_depth_span_m = 0.0
    structural_ids, id_labels_seen = _overview_structural_semantic_ids(
        semantic_metadata
    )
    structural_pixel_fraction: float | None = None
    if structural_ids:
        structural_pixel_fraction = float(
            np.mean(np.isin(semantic[..., 0], np.asarray(structural_ids)))
        )
    city_evidence_passed = bool(
        finite_depth_fraction >= OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION
        and geometry_fraction >= OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION
        and near_surface_fraction <= OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION
        and geometry_depth_span_m >= OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M
        and rgb_edge_fraction >= OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION
    )
    structural_evidence_passed = bool(
        not structural_ids
        or structural_pixel_fraction is not None
        and structural_pixel_fraction >= OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION
    )
    return {
        "finite_depth_fraction": finite_depth_fraction,
        "non_background_geometry_fraction": geometry_fraction,
        "near_surface_fraction": near_surface_fraction,
        "geometry_depth_span_m": geometry_depth_span_m,
        "rgb_edge_fraction": rgb_edge_fraction,
        "semantic_id_metadata_available": id_labels_seen,
        "structural_semantic_ids": list(structural_ids),
        "structural_pixel_fraction": structural_pixel_fraction,
        "city_evidence_passed": city_evidence_passed,
        "structural_evidence_passed": structural_evidence_passed,
    }


def _overview_archived_visual_evidence(
    rgb: np.ndarray,
    semantic: np.ndarray,
    semantic_metadata: Any,
) -> dict[str, Any]:
    """Check only evidence that is intentionally retained in a low-rate overview.

    Overview depth is checked live at every retained main sensor frame, but it
    is not stored in the evidence archive.  This function must therefore not
    manufacture a geometry/depth claim from RGB or semantic labels.
    """

    luma = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )
    horizontal = np.abs(np.diff(luma, axis=1)) >= OVERVIEW_CONTENT_RGB_EDGE_DELTA
    vertical = np.abs(np.diff(luma, axis=0)) >= OVERVIEW_CONTENT_RGB_EDGE_DELTA
    rgb_edge_fraction = float(
        (np.count_nonzero(horizontal) + np.count_nonzero(vertical))
        / (horizontal.size + vertical.size)
    )
    structural_ids, id_labels_seen = _overview_structural_semantic_ids(
        semantic_metadata
    )
    structural_pixel_fraction: float | None = None
    if structural_ids:
        structural_pixel_fraction = float(
            np.mean(np.isin(semantic[..., 0], np.asarray(structural_ids)))
        )
    return {
        "rgb_edge_fraction": rgb_edge_fraction,
        "rgb_evidence_passed": rgb_edge_fraction
        >= OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION,
        "semantic_id_metadata_available": id_labels_seen,
        "structural_semantic_ids": list(structural_ids),
        "structural_pixel_fraction": structural_pixel_fraction,
        "structural_evidence_passed": bool(
            not structural_ids
            or structural_pixel_fraction is not None
            and structural_pixel_fraction
            >= OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION
        ),
    }


def _load_npz(path: Path, issues: list[ValidationIssue]) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        _issue(issues, "npz_decode", path.relative_to(path.parents[1]).as_posix(), str(exc))
        return None


@dataclass
class _FramePayload:
    """Uniform access to legacy arrays and bounded-memory frame archives."""

    arrays: Mapping[str, np.ndarray] | None = None
    archive: ChunkedFrameArchive | None = None

    @property
    def fields(self) -> set[str]:
        if self.archive is not None:
            return self.archive.fields
        return set(self.arrays or {})

    @property
    def timestamps_ns(self) -> np.ndarray:
        return self.array("timestamps_ns")

    def descriptor(self, field: str) -> tuple[np.dtype[Any], tuple[int, ...]]:
        if self.archive is not None:
            descriptor = self.archive.descriptor(field)
            return descriptor.dtype, descriptor.shape
        value = self.array(field)
        return value.dtype, tuple(value.shape)

    def array(self, field: str) -> np.ndarray:
        if self.archive is not None:
            return self.archive.array(field)
        if self.arrays is None or field not in self.arrays:
            raise KeyError(field)
        return self.arrays[field]

    def frame(self, field: str, frame_index: int) -> np.ndarray:
        if self.archive is not None:
            return self.archive.frame(field, frame_index)
        return self.array(field)[frame_index]

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


def _load_frame_payload(path: Path, issues: list[ValidationIssue]) -> _FramePayload | None:
    relative = path.relative_to(path.parents[1]).as_posix()
    if not is_chunked_frame_archive(path):
        oversized = oversized_legacy_frame_members(
            path,
            ("rgb", "distance_to_image_plane_m", "semantic_segmentation"),
        )
        if oversized:
            limit_mib = LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES // (1024 * 1024)
            _issue(
                issues,
                "resource_unsafe_legacy_frame_archive",
                relative,
                f"legacy members exceed the {limit_mib} MiB bounded-memory limit: {oversized}",
            )
            return None
        arrays = _load_npz(path, issues)
        return _FramePayload(arrays=arrays) if arrays is not None else None
    try:
        return _FramePayload(archive=ChunkedFrameArchive(path))
    except (FrameArchiveError, OSError, ValueError, EOFError) as exc:
        _issue(issues, "npz_decode", relative, str(exc))
        return None


def _exact_frame_fields(
    payload: _FramePayload | None,
    expected: set[str],
    relative: str,
    issues: list[ValidationIssue],
) -> bool:
    if payload is None:
        return False
    actual = payload.fields
    if actual != expected:
        _issue(issues, "npz_fields", relative, f"expected {sorted(expected)}, got {sorted(actual)}")
        return False
    return True


def _frame_field_is_finite(payload: _FramePayload, field: str) -> bool:
    return all(np.isfinite(payload.frame(field, index)).all() for index in range(len(payload.timestamps_ns)))


def _exact_arrays(
    payload: Mapping[str, np.ndarray] | None,
    expected: set[str],
    relative: str,
    issues: list[ValidationIssue],
) -> bool:
    if payload is None:
        return False
    actual = set(payload)
    if actual != expected:
        _issue(issues, "npz_fields", relative, f"expected {sorted(expected)}, got {sorted(actual)}")
        return False
    return True


def _finite(name: str, value: np.ndarray, issues: list[ValidationIssue]) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        _issue(issues, "nonfinite", name, "numeric array must contain only finite values")


def _timestamps(
    payloads: Mapping[str, Mapping[str, np.ndarray]], issues: list[ValidationIssue]
) -> np.ndarray | None:
    reference: np.ndarray | None = None
    for relative, payload in payloads.items():
        value = payload.get("timestamps_ns")
        if value is None or value.dtype != np.int64 or value.ndim != 1 or len(value) < 2:
            _issue(issues, "timestamps", relative, "timestamps_ns must be int64 [T] with T >= 2")
            continue
        if not np.all(np.diff(value) > 0):
            _issue(issues, "timestamp_order", relative, "timestamps must be strictly increasing")
        if reference is None:
            reference = value
        elif not np.array_equal(reference, value):
            _issue(issues, "timestamp_alignment", relative, "sensor timestamps do not match")
    return reference


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _scan_policy_json(
    payload: Mapping[str, Any] | None,
    relative: str,
    issues: list[ValidationIssue],
    *,
    non_policy_metadata_paths: frozenset[str] = frozenset(),
) -> None:
    if payload is None:
        return
    for tree_path, key, value in iter_tree(payload):
        if (
            key is not None
            and tree_path not in non_policy_metadata_paths
            and key not in _SAFE_COMMITMENT_KEYS
            and (
                forbidden_policy_key(key)
                or normalized_key(key) in _PUBLIC_PRIVATE_TRUTH_KEYS
            )
        ):
            _issue(
                issues,
                "policy_truth_leakage",
                f"{relative}{tree_path[1:]}",
                f"forbidden policy-visible key: {key}",
            )
        if isinstance(value, str):
            token = forbidden_policy_value_token(value)
            if token is not None:
                _issue(
                    issues,
                    "policy_truth_leakage",
                    f"{relative}{tree_path[1:]}",
                    f"forbidden policy-visible provenance token: {token}",
                )


def _scan_public_private_artifact_json(
    payload: Mapping[str, Any] | None,
    relative: str,
    issues: list[ValidationIssue],
    *,
    private_target_ids: Sequence[str] = (),
    private_target_positions: Sequence[Sequence[float]] = (),
) -> None:
    """Reject private IDs, target geometry and evaluator paths in public JSON.

    Policy-key scanning intentionally permits aggregate counts and opaque
    commitment hashes.  This second boundary scan covers control-plane files
    such as ``capture_progress.json`` where a future checkpoint could otherwise
    copy a private object identifier or coordinate without becoming a policy
    input.
    """

    if payload is None:
        return
    normalized_private_ids = tuple(
        token.casefold()
        for token in private_target_ids
        if isinstance(token, str) and token.strip()
    )
    private_positions = tuple(
        tuple(float(coordinate) for coordinate in position)
        for position in private_target_positions
        if isinstance(position, Sequence)
        and not isinstance(position, (str, bytes))
        and len(position) == 3
    )
    for tree_path, key, value in iter_tree(payload):
        path = f"{relative}{tree_path[1:]}"
        normalized_key_name = normalized_key(key) if key is not None else ""
        if key is not None and normalized_key_name in _PUBLIC_PRIVATE_ARTIFACT_KEYS:
            # A public overview camera uses a key named target_w_m for its
            # fixed witness pose, so only reject it when the value exactly
            # equals evaluator geometry below; the field-name check still
            # rejects all other target-coordinate-shaped fields.
            if normalized_key_name != "target_w_m":
                _issue(issues, "public_private_leakage", path, "private target field is not public")
        if isinstance(value, str):
            lowered = value.casefold()
            if any(token in lowered for token in _PUBLIC_PRIVATE_ARTIFACT_PATH_TOKENS):
                _issue(issues, "public_private_leakage", path, "private evaluator path/token is not public")
            if any(token in lowered for token in normalized_private_ids):
                _issue(issues, "public_private_leakage", path, "private target identifier is not public")
        if (
            key is not None
            and normalized_key_name in {"position_w_m", "target_w_m", "coordinates"}
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 3
        ):
            try:
                candidate = tuple(float(coordinate) for coordinate in value)
            except (TypeError, ValueError):
                candidate = ()
            if candidate and any(
                all(math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9) for left, right in zip(candidate, expected, strict=True))
                for expected in private_positions
            ):
                _issue(issues, "public_private_leakage", path, "private target coordinate is not public")


def _validate_post_validation_video_artifacts(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
    bound_artifacts: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> frozenset[str]:
    """Validate the small post-validation video evidence extension.

    Video encoding intentionally happens after the independent validator, so
    videos cannot be part of the raw capture artifact hash inventory.  Once
    present, however, they are not arbitrary extra files: each MP4 must have
    the receipt emitted by the encoder, bind this capture receipt, and bind
    only source artifacts already covered by the capture receipt.
    """

    video_root = root / VIDEO_ARTIFACT_ROOT
    if not video_root.is_dir():
        return frozenset()
    allowed: set[str] = set()
    validation_path = root / "independent_validation.json"
    validation_sha256: str | None = None
    validation_bound = False
    if validation_path.is_file():
        try:
            validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            validation_payload = None
        if isinstance(validation_payload, Mapping):
            validation_bound = (
                validation_payload.get("schema") == VALIDATION_SCHEMA
                and validation_payload.get("status") == "passed"
                and validation_payload.get("capture_receipt_sha256") == receipt_sha256
            )
            validation_sha256 = sha256_file(validation_path)
    for path in sorted(video_root.rglob("*")):
        if not path.is_file():
            _issue(issues, "video_artifact", str(path.relative_to(root)), "video directory may contain regular files only")
            continue
        relative = path.relative_to(root).as_posix()
        if not relative.startswith(f"{VIDEO_ARTIFACT_ROOT}/"):
            _issue(issues, "video_artifact", relative, "video artifact must be beneath videos/")
            continue
        if relative.endswith(VIDEO_RECEIPT_SUFFIX):
            continue
        if path.suffix.casefold() != ".mp4":
            _issue(issues, "video_artifact", relative, "only encoder-produced .mp4 files are allowed under videos/")
            continue
        receipt_path = Path(f"{path}{'.receipt.json'}")
        receipt_relative = receipt_path.relative_to(root).as_posix()
        allowed.update((relative, receipt_relative))
        if not receipt_path.is_file():
            _issue(issues, "video_artifact", relative, "video receipt is missing")
            continue
        try:
            video_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _issue(issues, "video_artifact", receipt_relative, f"video receipt is invalid: {exc}")
            continue
        if not isinstance(video_receipt, Mapping):
            _issue(issues, "video_artifact", receipt_relative, "video receipt must be an object")
            continue
        if video_receipt.get("schema") not in VIDEO_RECEIPT_SCHEMAS:
            _issue(issues, "video_artifact", receipt_relative, "video receipt schema is not supported")
        if video_receipt.get("ok") is not True:
            _issue(issues, "video_artifact", receipt_relative, "video receipt must declare ok=true")
        if video_receipt.get("capture_receipt_sha256") != receipt_sha256:
            _issue(issues, "video_artifact", receipt_relative, "video receipt is bound to a different capture receipt")
        if not validation_bound or video_receipt.get("independent_validation_sha256") != validation_sha256:
            _issue(issues, "video_artifact", receipt_relative, "video receipt is not bound to the passing independent validation receipt")
        actual_sha256 = sha256_file(path)
        if video_receipt.get("video_sha256") != actual_sha256:
            _issue(issues, "video_artifact", relative, "video SHA-256 does not match its receipt")
        audit = video_receipt.get("audit")
        if not isinstance(audit, Mapping) or audit.get("sha256") != actual_sha256 or audit.get("bytes") != path.stat().st_size:
            _issue(issues, "video_artifact", receipt_relative, "video audit does not match the MP4")
        input_artifacts = video_receipt.get("input_artifacts")
        if not isinstance(input_artifacts, Mapping) or not input_artifacts:
            _issue(issues, "video_artifact", receipt_relative, "video receipt must bind input capture artifacts")
        else:
            for input_relative, input_binding in input_artifacts.items():
                if not isinstance(input_relative, str) or input_relative not in bound_artifacts:
                    _issue(issues, "video_artifact", f"{receipt_relative}.input_artifacts", "video input is not a capture-bound artifact")
                    continue
                source = root / PurePosixPath(input_relative)
                if not source.is_file() or not isinstance(input_binding, Mapping):
                    _issue(issues, "video_artifact", f"{receipt_relative}.input_artifacts.{input_relative}", "video input binding is missing")
                    continue
                if input_binding.get("sha256") != bound_artifacts[input_relative].get("sha256") or input_binding.get("bytes") != bound_artifacts[input_relative].get("bytes"):
                    _issue(issues, "video_artifact", f"{receipt_relative}.input_artifacts.{input_relative}", "video input binding disagrees with capture receipt")
        _scan_public_private_artifact_json(video_receipt, receipt_relative, issues)
    for path in sorted(video_root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix().endswith(VIDEO_RECEIPT_SUFFIX):
            mp4 = Path(str(path)[: -len(".receipt.json")])
            if not mp4.is_file():
                _issue(issues, "video_artifact", path.relative_to(root).as_posix(), "video receipt has no MP4 sibling")
            allowed.add(path.relative_to(root).as_posix())
    return frozenset(allowed)


def _exact_zero(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _exact_axis_volume(value: object, expected: Mapping[str, list[float]]) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        return False
    for axis, bounds in expected.items():
        actual = value.get(axis)
        if not isinstance(actual, list) or len(actual) != 2:
            return False
        for coordinate, expected_coordinate in zip(actual, bounds):
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                or float(coordinate) != expected_coordinate
            ):
                return False
    return True


def _exact_stage_units(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_STAGE_UNITS):
        return False
    if value.get("up_axis") != "Z":
        return False
    for key in ("meters_per_unit", "time_codes_per_second", "frames_per_second"):
        coordinate = value.get(key)
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
            or float(coordinate) != EXPECTED_STAGE_UNITS[key]
        ):
            return False
    return True


def _exact_collision_counts(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(EXPECTED_NATIVE_COLLISION_COUNTS)
        and all(
            isinstance(value.get(key), int)
            and not isinstance(value.get(key), bool)
            and value.get(key) == expected
            for key, expected in EXPECTED_NATIVE_COLLISION_COUNTS.items()
        )
    )


def _parse_structural_aabbs(
    scene: Mapping[str, Any], issues: list[ValidationIssue]
) -> tuple[AABB, ...]:
    raw_boxes = scene.get("structural_aabbs")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        _issue(
            issues,
            "structural_aabbs",
            "scene.json.structural_aabbs",
            "City-Lite capture requires nonempty structural/task-obstacle AABBs",
        )
        return ()
    boxes: list[AABB] = []
    seen_paths: set[str] = set()
    admitted_roots = tuple(destination for _, destination in SELECTIVE_REFERENCES)
    for index, item in enumerate(raw_boxes):
        path = f"scene.json.structural_aabbs[{index}]"
        if not isinstance(item, Mapping):
            _issue(issues, "structural_aabb", path, "AABB must be an object")
            continue
        prim_path = item.get("path")
        source_kind = item.get("source_kind")
        if (
            not isinstance(prim_path, str)
            or not prim_path.startswith("/World/")
            or not any(
                prim_path == root or prim_path.startswith(root + "/")
                for root in admitted_roots
            )
        ):
            _issue(
                issues,
                "structural_aabb_source",
                f"{path}.path",
                "source prim must descend from one of the two selective City-Lite roots",
            )
            continue
        if prim_path in seen_paths:
            _issue(issues, "structural_aabb_source", f"{path}.path", "source prim path is duplicated")
            continue
        if not isinstance(source_kind, str) or not source_kind.strip():
            _issue(issues, "structural_aabb_source", f"{path}.source_kind", "source_kind is required")
            continue
        minimum, maximum = item.get("min"), item.get("max")
        try:
            box = AABB(
                tuple(minimum),  # type: ignore[arg-type]
                tuple(maximum),  # type: ignore[arg-type]
                source_prim=prim_path,
                category=source_kind,
            )
        except (TypeError, ValueError):
            _issue(issues, "structural_aabb", path, "min/max must be finite ordered xyz coordinates")
            continue
        seen_paths.add(prim_path)
        boxes.append(box)
    return tuple(boxes)


def _validate_city_lite_scene(
    scene: Mapping[str, Any],
    *,
    evaluator_sha256: object,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
) -> tuple[AABB, ...]:
    if scene.get("environment_id") != ENVIRONMENT_ID:
        _issue(issues, "environment_id", "scene.json.environment_id", f"must be {ENVIRONMENT_ID}")
    if scene.get("static_scene_authority_verified") is not True:
        _issue(issues, "scene_authority", "scene.json", "City-Lite authority was not verified before capture")
    if not _exact_zero(scene.get("unresolved_reference_count")):
        _issue(issues, "unresolved_reference", "scene.json", "active selective composition must have zero unresolved references")
    if not _exact_zero(scene.get("legacy_prim_count")):
        _issue(issues, "legacy_prims", "scene.json", "legacy Mission/Drones prim count must be exactly zero")
    if not _exact_zero(scene.get("forbidden_decoration_prim_count")):
        _issue(issues, "decorative_prims", "scene.json", "foliage, grass, and traffic-sign prim count must be exactly zero")

    material_closure_ok = False
    material_closure = scene.get("city_task_obstacle_material_closure")
    if not isinstance(material_closure, Mapping):
        _issue(
            issues,
            "city_task_obstacle_material_closure",
            "scene.json.city_task_obstacle_material_closure",
            "the eight local CityTaskObstacles material bindings are required",
        )
    else:
        try:
            validate_city_task_obstacle_material_closure_receipt(material_closure)
        except CityLiteAuthorityError as exc:
            _issue(
                issues,
                "city_task_obstacle_material_closure",
                "scene.json.city_task_obstacle_material_closure",
                str(exc),
            )
        else:
            material_closure_ok = True

    contract = scene.get("scene_contract")
    expected_contract = {
        "sha256": SCENE_CONTRACT_SHA256,
        "payload_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
        "schema": SCENE_CONTRACT_SCHEMA,
        "gate_status": SCENE_CONTRACT_GATE_STATUS,
        "permissions": dict(EXPECTED_UPSTREAM_PERMISSIONS),
    }
    contract_ok = isinstance(contract, Mapping) and all(
        contract.get(key) == expected for key, expected in expected_contract.items()
    )
    if contract_ok:
        assert isinstance(contract, Mapping)
        permissions = contract.get("permissions")
        contract_ok = isinstance(permissions, Mapping) and all(
            permissions.get(key) is expected
            for key, expected in EXPECTED_UPSTREAM_PERMISSIONS.items()
        )
    if not contract_ok:
        _issue(issues, "scene_authority", "scene.json.scene_contract", "scene contract does not bind exact md_qd_swarm v1_r2 authority")

    assets = scene.get("authority_assets")
    if not isinstance(assets, Mapping) or set(assets) != set(AUTHORITY_SHA256):
        _issue(issues, "authority_assets", "scene.json.authority_assets", "authority asset inventory must be exact")
    else:
        for filename, expected_sha256 in AUTHORITY_SHA256.items():
            item = assets.get(filename)
            if not isinstance(item, Mapping) or item.get("sha256") != expected_sha256:
                _issue(issues, "authority_assets", f"scene.json.authority_assets.{filename}", "authority SHA-256 mismatch")

    expected_references = [
        {"source_prim": source, "destination_prim": destination}
        for source, destination in SELECTIVE_REFERENCES
    ]
    if scene.get("selective_references") != expected_references:
        _issue(issues, "selective_references", "scene.json.selective_references", "exactly the two admitted City-Lite prim references are required")

    layer_inventory_ok = False
    layer_inventory = scene.get("rivermark_layer_inventory")
    if not isinstance(layer_inventory, Mapping):
        _issue(
            issues,
            "rivermark_layer_inventory",
            "scene.json.rivermark_layer_inventory",
            "a hash-bound RivermarkSrc51 layer inventory receipt is required",
        )
    else:
        try:
            validate_rivermark_layer_inventory_receipt(layer_inventory)
        except CityLiteAuthorityError as exc:
            _issue(
                issues,
                "rivermark_layer_inventory",
                "scene.json.rivermark_layer_inventory",
                str(exc),
            )
        else:
            layer_inventory_ok = True

    if not _exact_stage_units(scene.get("stage_units")):
        _issue(issues, "stage_units", "scene.json.stage_units", "metres, Z-up, and 60 Hz stage metadata must be exact")
    if not _exact_collision_counts(scene.get("native_collision_counts")):
        _issue(issues, "native_collision_counts", "scene.json.native_collision_counts", "native collision audit is missing or stale")
    if not _exact_axis_volume(scene.get("flight_volume_m"), EXPECTED_FLIGHT_VOLUME):
        _issue(issues, "flight_volume", "scene.json.flight_volume_m", "City-Lite flight volume is missing or stale")
    if not _exact_axis_volume(scene.get("command_volume_m"), EXPECTED_COMMAND_VOLUME):
        _issue(issues, "command_volume", "scene.json.command_volume_m", "City-Lite command volume is missing or stale")
    clearance = scene.get("route_clearance_m")
    if isinstance(clearance, bool) or not isinstance(clearance, (int, float)) or float(clearance) != ROUTE_CLEARANCE_M:
        _issue(issues, "route_clearance_contract", "scene.json.route_clearance_m", f"must remain {ROUTE_CLEARANCE_M} m")
    if scene.get("private_evaluator_manifest_sha256") != evaluator_sha256:
        _issue(issues, "evaluator_binding", "scene.json.private_evaluator_manifest_sha256", "scene does not bind capture evaluator commitment")
    if scene.get("formal_benchmark_admission") is not False:
        _issue(issues, "scene_claim_boundary", "scene.json.formal_benchmark_admission", "raw City-Lite capture must not self-admit")

    boxes = _parse_structural_aabbs(scene, issues)
    geometry_sha256 = aabb_geometry_sha256(boxes) if boxes else None
    proxies = scene.get("collision_proxies")
    proxy_ok = isinstance(proxies, Mapping)
    if proxy_ok:
        assert isinstance(proxies, Mapping)
        proxy_ok = (
            isinstance(proxies.get("count"), int)
            and not isinstance(proxies.get("count"), bool)
            and proxies.get("count") == len(boxes)
            and len(boxes) > 0
            and proxies.get("aabb_geometry_sha256") == geometry_sha256
            and proxies.get("source_aabb_geometry_sha256") == geometry_sha256
            and proxies.get("representation") == COLLISION_PROXY_REPRESENTATION
            and proxies.get("prim_root") == COLLISION_PROXY_PRIM_ROOT
            and proxies.get("collision_enabled") is True
            and proxies.get("visible") is False
        )
    if not proxy_ok:
        _issue(issues, "collision_proxies", "scene.json.collision_proxies", "one invisible conservative collision proxy per structural AABB is required")

    lidar_coverage = scene.get("lidar_geometry_coverage")
    lidar_ok = isinstance(lidar_coverage, Mapping) and all(
        lidar_coverage.get(key) is True
        for key in ("includes_city", "includes_city_task_obstacles", "includes_collision_proxies")
    )
    if lidar_ok:
        assert isinstance(lidar_coverage, Mapping)
        lidar_ok = lidar_coverage.get("geometry_aabb_sha256") == geometry_sha256
    if not lidar_ok:
        _issue(issues, "lidar_geometry_coverage", "scene.json.lidar_geometry_coverage", "LiDAR must cover City-Lite, task obstacles, and the exact proxy geometry")

    checks["environment_id"] = scene.get("environment_id")
    checks["rivermark_layer_inventory_verified"] = layer_inventory_ok
    checks["city_task_obstacle_material_closure_verified"] = material_closure_ok
    checks["city_lite_authority_verified"] = not any(
        issue.code
        in {
            "environment_id",
            "scene_authority",
            "authority_assets",
            "selective_references",
            "city_task_obstacle_material_closure",
        }
        for issue in issues
    )
    checks["selective_reference_count"] = len(expected_references) if scene.get("selective_references") == expected_references else 0
    checks["structural_aabb_count"] = len(boxes)
    checks["structural_aabb_geometry_sha256"] = geometry_sha256
    checks["collision_proxy_count"] = proxies.get("count") if isinstance(proxies, Mapping) else 0
    checks["collision_proxy_geometry_verified"] = proxy_ok
    checks["lidar_geometry_coverage_verified"] = lidar_ok
    return boxes


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _validate_runtime_target_usd_closure(
    receipt: Mapping[str, Any],
    *,
    required: bool,
    expected_target_count: int | None,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
) -> bool:
    """Audit the public aggregate for pre/post-reset target USD closure.

    The capture-side USD query sees evaluator-private geometry, but its public
    receipt is restricted to this fixed aggregate schema.  A validator must
    reject missing, weakened, or expanded records rather than trusting that a
    capture process performed the query.
    """

    expected_keys = {
        "schema",
        "target_count",
        "all_targets_active",
        "all_targets_visible",
        "all_targets_renderable",
        "all_targets_have_expected_class_label",
        "all_target_transforms_rigid",
        "maximum_world_position_error_m",
        "maximum_radius_error_m",
        "maximum_bound_extent_error_m",
        "position_tolerance_m",
        "radius_tolerance_m",
        "bound_extent_tolerance_m",
    }
    phases = ("runtime_target_usd_pre_reset", "runtime_target_usd_post_reset")
    phase_results: dict[str, bool] = {}
    for phase in phases:
        closure = receipt.get(phase)
        if closure is None and not required:
            phase_results[phase] = False
            continue
        valid = isinstance(closure, Mapping) and set(closure) == expected_keys
        if not valid:
            _issue(
                issues,
                "runtime_target_usd_closure",
                f"capture_receipt.json.{phase}",
                "protocol-bound capture requires the exact path-free target USD closure aggregate",
            )
            phase_results[phase] = False
            continue
        assert isinstance(closure, Mapping)
        target_count = closure.get("target_count")
        count_valid = (
            isinstance(target_count, int)
            and not isinstance(target_count, bool)
            and target_count > 0
            and (expected_target_count is None or target_count == expected_target_count)
        )
        flags_valid = all(
            closure.get(key) is True
            for key in (
                "all_targets_active",
                "all_targets_visible",
                "all_targets_renderable",
                "all_targets_have_expected_class_label",
                "all_target_transforms_rigid",
            )
        )
        tolerances = (
            ("position_tolerance_m", RUNTIME_TARGET_USD_POSITION_TOLERANCE_M),
            ("radius_tolerance_m", RUNTIME_TARGET_USD_RADIUS_TOLERANCE_M),
            ("bound_extent_tolerance_m", RUNTIME_TARGET_USD_BOUND_EXTENT_TOLERANCE_M),
        )
        tolerance_valid = all(
            _finite_float_or_none(closure.get(key)) == expected for key, expected in tolerances
        )
        errors = (
            ("maximum_world_position_error_m", RUNTIME_TARGET_USD_POSITION_TOLERANCE_M),
            ("maximum_radius_error_m", RUNTIME_TARGET_USD_RADIUS_TOLERANCE_M),
            ("maximum_bound_extent_error_m", RUNTIME_TARGET_USD_BOUND_EXTENT_TOLERANCE_M),
        )
        error_valid = all(
            (value := _finite_float_or_none(closure.get(key))) is not None
            and 0.0 <= value <= maximum
            for key, maximum in errors
        )
        valid = (
            closure.get("schema") == "org.rivermark.runtime-target-usd-closure.v1"
            and count_valid
            and flags_valid
            and tolerance_valid
            and error_valid
        )
        if not valid:
            _issue(
                issues,
                "runtime_target_usd_closure",
                f"capture_receipt.json.{phase}",
                "target USD closure does not prove active, visible, renderable, semantic-labelled, rigid private-manifest authoring",
            )
        phase_results[phase] = valid
    present_phases = tuple(phase for phase in phases if receipt.get(phase) is not None)
    verified = all(phase_results.values()) if required else bool(present_phases) and all(
        phase_results[phase] for phase in present_phases
    )
    checks["runtime_target_usd_closure_required"] = required
    checks["runtime_target_usd_closure_verified"] = verified
    return verified


def _validate_literal_city_lite_fleet_spawn(
    receipt: Mapping[str, Any],
    scene: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
    *,
    routes_w_m: Sequence[Sequence[Sequence[float]]] | None = None,
) -> bool:
    """Require auditable literal-CF2X authoring and reset evidence.

    The live state immediately after ``sim.reset()`` is allowed to contain a
    recorded, bounded physical settling velocity.  It cannot substitute for
    the separate USD-transform and resolved-default-state proofs, and it never
    replaces the post-reset runtime safety trace checked elsewhere.
    """

    physics = receipt.get("physics")
    ok = isinstance(physics, Mapping)
    if not isinstance(physics, Mapping):
        physics = {}
    expected_paths = list(SWARM_AGENT_LITERAL_PRIM_PATHS)
    if (
        physics.get("same_world_agent_count") != AGENT_COUNT
        or physics.get("multirotor_prim_expression") != "/World/Swarm/Agent_.*/Robot"
        or physics.get("literal_agent_prim_paths") != expected_paths
    ):
        ok = False

    literal = physics.get("literal_fleet_spawn")
    expected_literal_keys = {
        "literal_prim_paths",
        "authored_usd_transform",
        "authored_defaults",
        "post_reset_physics_settling",
        "post_reset_root_pose_rewrite",
        "post_reset_root_velocity_rewrite",
        "anchor_contract",
    }
    if not isinstance(literal, Mapping) or set(literal) != expected_literal_keys:
        literal = {}
        ok = False
    if (
        literal.get("literal_prim_paths") != expected_paths
        or literal.get("post_reset_root_pose_rewrite") is not False
        or literal.get("post_reset_root_velocity_rewrite") is not False
        or literal.get("anchor_contract") != "rivermark_public_route_initial_waypoints"
    ):
        ok = False

    usd = literal.get("authored_usd_transform")
    expected_usd_keys = {
        "source",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "per_agent",
        "max_position_error_m",
        "max_orientation_error_rad",
    }
    if not isinstance(usd, Mapping) or set(usd) != expected_usd_keys:
        usd = {}
        ok = False
    if (
        usd.get("source") != "fresh_stage_usd_xform_cache_before_sim_reset"
        or _finite_float_or_none(usd.get("position_tolerance_m"))
        != LITERAL_USD_SPAWN_POSITION_TOLERANCE_M
        or _finite_float_or_none(usd.get("orientation_tolerance_rad"))
        != LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD
    ):
        ok = False
    rows = usd.get("per_agent")
    expected_usd_row_keys = {
        "agent_id",
        "prim_path",
        "position_error_m",
        "orientation_error_rad",
        "rigid_transform_determinant",
        "basis_axis_lengths",
    }
    if not isinstance(rows, list) or len(rows) != AGENT_COUNT:
        ok = False
    else:
        for agent_id, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected_usd_row_keys:
                ok = False
                continue
            position_error = _finite_float_or_none(row.get("position_error_m"))
            orientation_error = _finite_float_or_none(row.get("orientation_error_rad"))
            determinant = _finite_float_or_none(row.get("rigid_transform_determinant"))
            basis_axis_lengths = row.get("basis_axis_lengths")
            basis_lengths_valid = isinstance(basis_axis_lengths, list) and len(
                basis_axis_lengths
            ) == 3
            if basis_lengths_valid:
                basis_lengths_valid = all(
                    length is not None
                    and math.isclose(
                        length,
                        1.0,
                        rel_tol=0.0,
                        abs_tol=LITERAL_USD_SPAWN_BASIS_LENGTH_TOLERANCE,
                    )
                    for length in (
                        _finite_float_or_none(value) for value in basis_axis_lengths
                    )
                )
            if (
                row.get("agent_id") != agent_id
                or row.get("prim_path") != expected_paths[agent_id]
                or position_error is None
                or not 0.0 <= position_error <= LITERAL_USD_SPAWN_POSITION_TOLERANCE_M
                or orientation_error is None
                or not 0.0 <= orientation_error <= LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD
                or determinant is None
                or not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
                or not basis_lengths_valid
            ):
                ok = False
    for key, maximum in (
        ("max_position_error_m", LITERAL_USD_SPAWN_POSITION_TOLERANCE_M),
        ("max_orientation_error_rad", LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD),
    ):
        value = _finite_float_or_none(usd.get(key))
        if value is None or not 0.0 <= value <= maximum:
            ok = False

    defaults = literal.get("authored_defaults")
    expected_default_keys = {
        "root_state_shape",
        "thruster_rps_shape",
        "thrust_target_shape",
        "root_state_max_abs_error",
        "thruster_rps_max_abs_error",
        "thrust_target_max_abs_error_n",
        "root_state_tolerance",
        "thruster_rps_tolerance",
        "thrust_target_tolerance_n",
    }
    if not isinstance(defaults, Mapping) or set(defaults) != expected_default_keys:
        defaults = {}
        ok = False
    if (
        defaults.get("root_state_shape") != [AGENT_COUNT, 13]
        or defaults.get("thruster_rps_shape") != [AGENT_COUNT, 4]
        or defaults.get("thrust_target_shape") != [AGENT_COUNT, 4]
        or _finite_float_or_none(defaults.get("root_state_tolerance"))
        != LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE
        or _finite_float_or_none(defaults.get("thruster_rps_tolerance"))
        != LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE
        or _finite_float_or_none(defaults.get("thrust_target_tolerance_n"))
        != LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N
    ):
        ok = False
    for key, maximum in (
        ("root_state_max_abs_error", LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE),
        ("thruster_rps_max_abs_error", LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE),
        ("thrust_target_max_abs_error_n", LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N),
    ):
        value = _finite_float_or_none(defaults.get(key))
        if value is None or not 0.0 <= value <= maximum:
            ok = False

    settling = literal.get("post_reset_physics_settling")
    expected_settling_keys = {
        "classification",
        "max_position_delta_m",
        "max_orientation_delta_rad",
        "max_linear_velocity_mps",
        "max_angular_velocity_radps",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "linear_velocity_hard_limit_mps",
        "angular_velocity_hard_limit_radps",
    }
    if not isinstance(settling, Mapping) or set(settling) != expected_settling_keys:
        settling = {}
        ok = False
    if (
        settling.get("classification") != "observed_after_sim_reset_before_first_command"
        or _finite_float_or_none(settling.get("position_tolerance_m"))
        != LITERAL_SPAWN_POSITION_TOLERANCE_M
        or _finite_float_or_none(settling.get("orientation_tolerance_rad"))
        != LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD
        or _finite_float_or_none(settling.get("linear_velocity_hard_limit_mps"))
        != MAX_CF2X_LINEAR_VELOCITY_MPS
        or _finite_float_or_none(settling.get("angular_velocity_hard_limit_radps"))
        != MAX_CF2X_ANGULAR_VELOCITY_RADPS
    ):
        ok = False
    for key, maximum in (
        ("max_position_delta_m", LITERAL_SPAWN_POSITION_TOLERANCE_M),
        ("max_orientation_delta_rad", LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD),
        ("max_linear_velocity_mps", MAX_CF2X_LINEAR_VELOCITY_MPS),
        ("max_angular_velocity_radps", MAX_CF2X_ANGULAR_VELOCITY_RADPS),
    ):
        value = _finite_float_or_none(settling.get(key))
        if value is None or not 0.0 <= value <= maximum:
            ok = False

    try:
        expected_states = (
            _city_lite_spawn_states(routes_w_m)
            if routes_w_m is not None
            else _city_lite_spawn_states()
        )
    except (TypeError, ValueError, RuntimeError):
        # Route validation reports malformed public tasks separately. Keep
        # this evidence gate deterministic and fail closed on the default
        # route family rather than masking the root issue with an exception.
        expected_states = _city_lite_spawn_states()
    expected_root_poses = np.asarray(
        [(*position, *quaternion) for position, quaternion in expected_states],
        dtype=np.float64,
    )
    if not isinstance(scene, Mapping):
        ok = False
    else:
        try:
            scene_root_poses = np.asarray(scene.get("initial_root_poses_wxyz"), dtype=np.float64)
        except (TypeError, ValueError):
            scene_root_poses = np.empty((0, 0), dtype=np.float64)
        if (
            scene_root_poses.shape != (AGENT_COUNT, 7)
            or not np.isfinite(scene_root_poses).all()
            or not np.allclose(scene_root_poses, expected_root_poses, rtol=0.0, atol=1.0e-6)
            or scene.get("literal_fleet") != literal
        ):
            ok = False

    trim = physics.get("cf2x_hover_trim")
    if not isinstance(trim, Mapping) or (
        _finite_float_or_none(trim.get("hover_thrust_per_rotor_n"))
        != HOVER_THRUST_PER_ROTOR_N
        or _finite_float_or_none(trim.get("initial_hover_rps")) != INITIAL_HOVER_RPS
    ):
        ok = False

    if not ok:
        _issue(
            issues,
            "literal_fleet_spawn",
            "capture_receipt.json.physics.literal_fleet_spawn",
            "eight literal CF2X assets require exact pre-reset USD, resolved-default, and bounded settling evidence",
        )
    checks["literal_fleet_spawn_verified"] = ok
    return ok


def _runtime_safety_trace_aabb_violation(
    positions: np.ndarray, structural_aabbs: Sequence[AABB]
) -> tuple[int, int, AABB] | None:
    """Return the first expanded-AABB intersection in a full physical trace."""

    # Row zero is the post-reset point check. Every later row is a swept
    # segment from its preceding physical state. The slab computation is
    # vectorized over time and agents, while retaining the frozen AABB order
    # for a deterministic first violation.
    starts = np.concatenate((positions[:1], positions[:-1]), axis=0)
    ends = positions
    delta = ends - starts
    shape = positions.shape[:2]
    for box in structural_aabbs:
        minimum = np.asarray(box.minimum, dtype=np.float64) - ROUTE_CLEARANCE_M
        maximum = np.asarray(box.maximum, dtype=np.float64) + ROUTE_CLEARANCE_M
        low = np.zeros(shape, dtype=np.float64)
        high = np.ones(shape, dtype=np.float64)
        missed = np.zeros(shape, dtype=bool)
        for axis in range(3):
            first = starts[..., axis]
            movement = delta[..., axis]
            static = np.abs(movement) <= 1.0e-12
            missed |= static & ((first < minimum[axis]) | (first > maximum[axis]))
            moving = ~static
            if np.any(moving):
                left = np.zeros(shape, dtype=np.float64)
                right = np.zeros(shape, dtype=np.float64)
                left[moving] = (minimum[axis] - first[moving]) / movement[moving]
                right[moving] = (maximum[axis] - first[moving]) / movement[moving]
                low[moving] = np.maximum(low[moving], np.minimum(left[moving], right[moving]))
                high[moving] = np.minimum(high[moving], np.maximum(left[moving], right[moving]))
        intersections = ~missed & (low <= high + 1.0e-12)
        if np.any(intersections):
            frame_id, agent_id = (int(value) for value in np.argwhere(intersections)[0])
            return frame_id, agent_id, box
    return None


def _runtime_safety_trace_inter_agent_violation(
    positions: np.ndarray,
) -> tuple[float, tuple[int, int, int, float] | None]:
    """Independently replay simultaneous swept separation for every CF2X pair."""

    minimum = math.inf
    for frame_id in range(positions.shape[0]):
        previous = positions[frame_id] if frame_id == 0 else positions[frame_id - 1]
        current = positions[frame_id]
        for left_agent_id in range(AGENT_COUNT - 1):
            left_start = tuple(float(value) for value in previous[left_agent_id])
            left_end = tuple(float(value) for value in current[left_agent_id])
            for right_agent_id in range(left_agent_id + 1, AGENT_COUNT):
                right_start = tuple(float(value) for value in previous[right_agent_id])
                right_end = tuple(float(value) for value in current[right_agent_id])
                relative_start = tuple(
                    left_start[axis] - right_start[axis] for axis in range(3)
                )
                relative_delta = tuple(
                    (left_end[axis] - left_start[axis])
                    - (right_end[axis] - right_start[axis])
                    for axis in range(3)
                )
                denominator = sum(component * component for component in relative_delta)
                if denominator <= 0.0:
                    closest_time = 0.0
                else:
                    closest_time = -sum(
                        relative_start[axis] * relative_delta[axis] for axis in range(3)
                    ) / denominator
                    closest_time = min(1.0, max(0.0, closest_time))
                separation = math.sqrt(
                    sum(
                        (relative_start[axis] + closest_time * relative_delta[axis]) ** 2
                        for axis in range(3)
                    )
                )
                minimum = min(minimum, separation)
                if separation <= INTER_AGENT_MINIMUM_CENTER_SEPARATION_M:
                    return minimum, (
                        frame_id,
                        left_agent_id,
                        right_agent_id,
                        closest_time,
                    )
    return minimum, None


def _validate_runtime_safety_trace(
    trace: Mapping[str, np.ndarray] | None,
    receipt: Mapping[str, Any],
    structural_aabbs: Sequence[AABB],
    *,
    state: Mapping[str, np.ndarray] | None,
    captured_contact: Mapping[str, np.ndarray] | None,
    sensor_timestamps: np.ndarray | None,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
) -> tuple[int | None, float | None]:
    """Recompute full-step guard facts and bind every capture sample to time."""

    relative = RUNTIME_SAFETY_TRACE_RELATIVE_PATH
    expected_fields = {
        "physics_step",
        "physics_time_ns",
        "phase_code",
        "frame_outcome_code",
        "root_pos_w_m",
        "net_contact_forces_w_n",
        "max_contact_force_n",
    }
    if not _exact_arrays(trace, expected_fields, relative, issues):
        checks["runtime_safety_trace_verified"] = False
        return None, None
    assert trace is not None
    command = receipt.get("command")
    if not isinstance(command, Mapping):
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            "capture command is required for full-step trace validation",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    rollout_steps = command.get("steps")
    warmup_steps = command.get("warmup_steps")
    capture_stride = command.get("capture_stride")
    dt_s = command.get("dt_s")
    if (
        isinstance(rollout_steps, bool)
        or not isinstance(rollout_steps, int)
        or rollout_steps < 1
        or isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
        or isinstance(capture_stride, bool)
        or not isinstance(capture_stride, int)
        or capture_stride < 1
        or isinstance(dt_s, bool)
        or not isinstance(dt_s, (int, float))
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
    ):
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            "capture command has invalid physical-step timing",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    frame_count = 1 + warmup_steps + rollout_steps
    physics_step = trace["physics_step"]
    physics_time = trace["physics_time_ns"]
    phase_code = trace["phase_code"]
    frame_outcome = trace["frame_outcome_code"]
    positions = trace["root_pos_w_m"]
    forces = trace["net_contact_forces_w_n"]
    maxima = trace["max_contact_force_n"]
    shape_ok = (
        physics_step.dtype == np.int64
        and physics_step.shape == (frame_count,)
        and physics_time.dtype == np.int64
        and physics_time.shape == (frame_count,)
        and phase_code.dtype == np.int8
        and phase_code.shape == (frame_count,)
        and frame_outcome.dtype == np.uint8
        and frame_outcome.shape == (frame_count,)
        and positions.dtype == np.float32
        and positions.shape == (frame_count, AGENT_COUNT, 3)
        and forces.dtype == np.float32
        and forces.shape == (frame_count, AGENT_COUNT, 1, 3)
        and maxima.dtype == np.float32
        and maxima.shape == (frame_count,)
    )
    if not shape_ok:
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            "full-step trace must use int64 steps/times, int8 phases, uint8 outcomes, float32 [F,8,3] roots, float32 [F,8,1,3] normal forces, and float32 [F] maxima",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    expected_steps = np.arange(frame_count, dtype=np.int64)
    try:
        expected_times = np.asarray(
            [physics_time_ns(step, float(dt_s)) for step in range(frame_count)],
            dtype=np.int64,
        )
    except (OverflowError, ValueError):
        _issue(
            issues,
            "runtime_safety_timing",
            relative,
            "capture command cannot produce an int64 physical-time schedule",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    expected_phases = np.asarray(
        [RUNTIME_SAFETY_PHASE_CODES["post_reset"]]
        + [RUNTIME_SAFETY_PHASE_CODES["warmup"]] * warmup_steps
        + [RUNTIME_SAFETY_PHASE_CODES["rollout"]] * rollout_steps,
        dtype=np.int8,
    )
    expected_outcomes = np.full(
        frame_count,
        RUNTIME_SAFETY_FRAME_OUTCOME_CODES["passed"],
        dtype=np.uint8,
    )
    if not np.array_equal(physics_step, expected_steps) or not np.array_equal(
        phase_code, expected_phases
    ):
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            "physical-step or phase sequence is not complete",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    if not np.array_equal(physics_time, expected_times):
        _issue(
            issues,
            "runtime_safety_timing",
            relative,
            "physical frame timestamps do not match the capture dt schedule",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    if not np.array_equal(frame_outcome, expected_outcomes):
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            "a successful capture must contain only passed runtime safety frames",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    if not (
        np.isfinite(positions).all()
        and np.isfinite(forces).all()
        and np.isfinite(maxima).all()
    ):
        _issue(issues, "runtime_safety_trace", relative, "full-step safety trace contains non-finite values")
        checks["runtime_safety_trace_verified"] = False
        return None, None
    observed_maxima = np.max(np.linalg.norm(forces, axis=-1), axis=(1, 2))
    if np.any(maxima < 0.0) or not np.allclose(maxima, observed_maxima, rtol=1.0e-6, atol=1.0e-8):
        _issue(issues, "runtime_safety_trace", relative, "per-frame contact maxima do not match raw normal forces")
        checks["runtime_safety_trace_verified"] = False
        return None, None
    trace_maximum = float(np.max(observed_maxima))
    if trace_maximum >= CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N:
        _issue(issues, "runtime_safety_trace", relative, "full-step contact force reaches the abort threshold")
        checks["runtime_safety_trace_verified"] = False
        return None, None
    minimum = np.asarray(CITY_LITE_FLIGHT_VOLUME_W_M.minimum) + CF2X_RUNTIME_GUARD_RADIUS_M
    maximum = np.asarray(CITY_LITE_FLIGHT_VOLUME_W_M.maximum) - CF2X_RUNTIME_GUARD_RADIUS_M
    in_volume = bool(np.all(positions >= minimum) and np.all(positions <= maximum))
    if not in_volume:
        _issue(issues, "runtime_safety_trace", relative, "full-step trace leaves the CF2X-shrunk flight volume")
        checks["runtime_safety_trace_verified"] = False
        return None, None
    violation = _runtime_safety_trace_aabb_violation(positions, structural_aabbs)
    if violation is not None:
        frame_id, agent_id, box = violation
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            f"physics frame {frame_id}, agent {agent_id} intersects protected AABB {box.source_prim}",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None
    minimum_pair_separation, pair_violation = _runtime_safety_trace_inter_agent_violation(
        positions
    )
    if pair_violation is not None:
        frame_id, left_agent_id, right_agent_id, closest_time = pair_violation
        _issue(
            issues,
            "runtime_safety_trace",
            relative,
            "physics frame "
            f"{frame_id}, agents {left_agent_id}/{right_agent_id} breach the "
            f"{INTER_AGENT_MINIMUM_CENTER_SEPARATION_M:.3f} m simultaneous swept separation "
            f"at t={closest_time:.6f}",
        )
        checks["runtime_safety_trace_verified"] = False
        return None, None

    expected_command_times = expected_times[warmup_steps : warmup_steps + rollout_steps]
    expected_effective_times = expected_times[
        1 + warmup_steps : 1 + warmup_steps + rollout_steps
    ]
    state_bound = False
    state_timing_bound = False
    contact_bound = False
    sensor_timing_bound = False
    if (
        state is not None
        and state.get("root_pos_w_m") is not None
        and state.get("command_time_ns") is not None
        and state.get("effective_time_ns") is not None
    ):
        state_positions = state["root_pos_w_m"]
        state_command = state["command_time_ns"]
        state_effective = state["effective_time_ns"]
        expected_rollout_shape = (rollout_steps, AGENT_COUNT, 3)
        state_timing_bound = (
            state_command.dtype == np.int64
            and state_command.shape == (rollout_steps,)
            and state_effective.dtype == np.int64
            and state_effective.shape == (rollout_steps,)
            and np.array_equal(state_command, expected_command_times)
            and np.array_equal(state_effective, expected_effective_times)
        )
        if (
            state_positions.dtype == np.float32
            and state_positions.shape == expected_rollout_shape
            and state_timing_bound
            and np.array_equal(
                positions[1 + warmup_steps : 1 + warmup_steps + rollout_steps],
                state_positions,
            )
        ):
            state_bound = True
    if not state_timing_bound:
        _issue(
            issues,
            "capture_timing",
            "streams/state_action.npz",
            "state command/effective timestamps do not match the full physical-frame schedule",
        )
    if not state_bound:
        _issue(
            issues,
            "runtime_safety_trace_binding",
            relative,
            "rollout root positions and timestamps are not bitwise bound to the recorded state/action stream",
        )
        checks["runtime_safety_trace_verified"] = False
        checks["runtime_safety_trace_state_bound"] = False
        checks["runtime_safety_trace_contact_bound"] = False
        checks["runtime_safety_trace_timing_bound"] = False
        return None, None
    checks["runtime_safety_trace_state_bound"] = True

    # Keep the validator's retained-frame schedule identical to the capture
    # loop.  The final partial stride is intentionally retained, so a plain
    # arange(capture_stride - 1, steps, capture_stride) would omit it.
    expected_sensor_indices = np.asarray(
        _captured_frame_indices(rollout_steps, capture_stride), dtype=np.int64
    )
    expected_sensor_timestamps = expected_effective_times[expected_sensor_indices]
    if sensor_timestamps is not None:
        sensor_timing_bound = (
            sensor_timestamps.dtype == np.int64
            and sensor_timestamps.shape == expected_sensor_timestamps.shape
            and np.array_equal(sensor_timestamps, expected_sensor_timestamps)
        )
    if not sensor_timing_bound:
        _issue(
            issues,
            "capture_timing",
            "sensors/*.npz",
            "sensor timestamps do not match the configured capture-stride physical frames",
        )
    if (
        captured_contact is not None
        and sensor_timing_bound
        and captured_contact.get("timestamps_ns") is not None
        and captured_contact.get("net_forces_w_n") is not None
    ):
        contact_timestamps = captured_contact["timestamps_ns"]
        sampled_forces = captured_contact["net_forces_w_n"]
        expected_contact_rows = 1 + warmup_steps + expected_sensor_indices
        if (
            contact_timestamps.dtype == np.int64
            and np.array_equal(contact_timestamps, expected_sensor_timestamps)
            and sampled_forces.dtype == np.float32
            and sampled_forces.shape
            == (len(expected_sensor_timestamps), AGENT_COUNT, 1, 3)
            and np.array_equal(forces[expected_contact_rows], sampled_forces)
        ):
            contact_bound = True
    if not contact_bound:
        _issue(
            issues,
            "runtime_safety_trace_binding",
            relative,
            "capture-stride contact samples are not bitwise bound to their full-step safety trace rows",
        )
        checks["runtime_safety_trace_verified"] = False
        checks["runtime_safety_trace_contact_bound"] = False
        checks["runtime_safety_trace_timing_bound"] = False
        return None, None
    checks["runtime_safety_trace_contact_bound"] = True
    checks["runtime_safety_trace_timing_bound"] = True
    checks["runtime_safety_trace_verified"] = True
    checks["runtime_safety_trace_frames"] = frame_count
    checks["runtime_safety_trace_max_contact_force_n"] = trace_maximum
    checks["runtime_safety_trace_minimum_inter_agent_swept_separation_m"] = (
        minimum_pair_separation
    )
    return frame_count, trace_maximum


def _validate_sensor_phase_trace(
    trace: Mapping[str, np.ndarray] | None,
    receipt: Mapping[str, Any],
    *,
    state: Mapping[str, np.ndarray] | None,
    contact: Mapping[str, np.ndarray] | None,
    timestamps: np.ndarray | None,
    root: Path,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
) -> bool:
    """Verify physical-frame order and retained contact bytes independently."""

    relative = SENSOR_PHASE_TRACE_RELATIVE_PATH
    if trace is None:
        _issue(issues, "sensor_phase_trace", relative, "sensor phase trace is required")
        checks["sensor_phase_trace_verified"] = False
        return False
    expected_fields = {
        "schema",
        "sensor_names",
        "physics_step",
        "physics_time_ns",
        "event_codes",
        "retained_contact_sha256",
        "archive_frame_index",
    }
    if set(trace) != expected_fields:
        _issue(
            issues,
            "sensor_phase_trace",
            relative,
            f"trace fields must be exactly {sorted(expected_fields)}",
        )
        checks["sensor_phase_trace_verified"] = False
        return False

    schema = trace["schema"]
    names = trace["sensor_names"]
    schema_ok = (
        schema.ndim == 1
        and schema.shape == (1,)
        and schema.dtype.kind in {"U", "S"}
        and str(schema[0]) == SENSOR_PHASE_TRACE_SCHEMA
    )
    names_ok = (
        names.ndim == 1
        and names.dtype.kind in {"U", "S"}
        and tuple(str(value) for value in names.tolist())
        == SENSOR_PHASE_SENSOR_NAMES
    )
    if not schema_ok or not names_ok:
        _issue(
            issues,
            "sensor_phase_trace",
            relative,
            "trace schema or sensor family ordering is invalid",
        )

    physics_step = trace["physics_step"]
    physics_time = trace["physics_time_ns"]
    event_codes = trace["event_codes"]
    digests = trace["retained_contact_sha256"]
    archive_indices = trace["archive_frame_index"]
    count = len(timestamps) if timestamps is not None else 0
    shape_ok = (
        physics_step.dtype == np.int64
        and physics_time.dtype == np.int64
        and archive_indices.dtype == np.int64
        and physics_step.shape == (count,)
        and physics_time.shape == (count,)
        and archive_indices.shape == (count,)
        and event_codes.dtype == np.uint8
        and event_codes.shape == (count, len(SENSOR_PHASE_EVENT_SEQUENCE))
        and digests.dtype == np.uint8
        and digests.shape == (count, 32)
    )
    if not shape_ok:
        _issue(
            issues,
            "sensor_phase_trace",
            relative,
            "trace arrays do not have the declared frame-aligned dtypes and shapes",
        )
        checks["sensor_phase_trace_verified"] = False
        return False

    command = receipt.get("command")
    steps = command.get("steps") if isinstance(command, Mapping) else None
    warmup = command.get("warmup_steps") if isinstance(command, Mapping) else None
    stride = command.get("capture_stride") if isinstance(command, Mapping) else None
    try:
        indices = np.asarray(_captured_frame_indices(int(steps), int(stride)), dtype=np.int64)
    except (TypeError, ValueError):
        indices = np.asarray([], dtype=np.int64)
        _issue(
            issues,
            "sensor_phase_trace",
            "capture_receipt.json.command",
            "steps, warmup_steps, and capture_stride are required to replay phase rows",
        )
    if indices.shape != (count,):
        _issue(
            issues,
            "sensor_phase_trace",
            relative,
            "trace frame count does not match the capture cadence",
        )
    expected_step = (
        int(warmup) + indices + 1
        if isinstance(warmup, (int, np.integer)) and not isinstance(warmup, bool)
        else np.asarray([], dtype=np.int64)
    )
    expected_time = None
    if state is not None and "effective_time_ns" in state:
        effective = state["effective_time_ns"]
        if effective.dtype == np.int64 and effective.ndim == 1 and indices.size:
            expected_time = effective[indices]
    if expected_time is None and timestamps is not None:
        expected_time = timestamps
    timing_ok = (
        expected_step.shape == physics_step.shape
        and np.array_equal(physics_step, expected_step)
        and expected_time is not None
        and np.array_equal(physics_time, expected_time)
        and timestamps is not None
        and np.array_equal(physics_time, timestamps)
    )
    if not timing_ok:
        _issue(
            issues,
            "sensor_phase_timing",
            relative,
            "physical steps and absolute timestamps do not match state/action and retained frames",
        )

    expected_events = np.asarray(SENSOR_PHASE_EVENT_SEQUENCE, dtype=np.uint8)
    events_ok = bool(np.array_equal(event_codes, np.repeat(expected_events[None, :], count, axis=0)))
    if not events_ok:
        _issue(
            issues,
            "sensor_phase_order",
            relative,
            "one or more retained frames has a reordered or incomplete sensor event sequence",
        )
    archive_ok = bool(np.array_equal(archive_indices, np.arange(count, dtype=np.int64)))
    if not archive_ok:
        _issue(
            issues,
            "sensor_phase_binding",
            relative,
            "archive frame indices must be contiguous and zero-based",
        )

    contact_ok = True
    if contact is None or timestamps is None:
        contact_ok = False
    else:
        values = contact.get("net_forces_w_n")
        contact_ok = (
            isinstance(values, np.ndarray)
            and values.shape[0] == count
            and all(
                np.array_equal(
                    digests[index],
                    np.frombuffer(sensor_phase_array_digest(values[index]), dtype=np.uint8),
                )
                for index in range(count)
            )
        )
    if not contact_ok:
        _issue(
            issues,
            "sensor_phase_contact_binding",
            relative,
            "retained contact digests are not byte-for-byte bound to sensors/contact.npz",
        )

    binding = receipt.get("sensor_phase_trace")
    binding_ok = (
        isinstance(binding, Mapping)
        and binding.get("schema") == SENSOR_PHASE_TRACE_SCHEMA
        and binding.get("path") == SENSOR_PHASE_TRACE_RELATIVE_PATH
        and binding.get("frame_count") == count
        and binding.get("sensor_names") == list(SENSOR_PHASE_SENSOR_NAMES)
        and binding.get("event_codes") == list(SENSOR_PHASE_EVENT_SEQUENCE)
        and is_sha256(binding.get("sha256"))
        and (root / SENSOR_PHASE_TRACE_RELATIVE_PATH).is_file()
        and binding.get("sha256") == sha256_file(root / SENSOR_PHASE_TRACE_RELATIVE_PATH)
    )
    if not binding_ok:
        _issue(
            issues,
            "sensor_phase_binding",
            "capture_receipt.json.sensor_phase_trace",
            "receipt does not bind the phase trace path, hash, schema, and frame count",
        )
    verified = bool(schema_ok and names_ok and shape_ok and timing_ok and events_ok and archive_ok and contact_ok and binding_ok)
    checks["sensor_phase_trace_verified"] = verified
    checks["sensor_phase_trace_frames"] = count
    checks["sensor_phase_trace_event_sequence"] = list(SENSOR_PHASE_EVENT_SEQUENCE)
    return verified


def _validate_runtime_safety_guard(
    receipt: Mapping[str, Any],
    scene: Mapping[str, Any] | None,
    structural_aabbs: Sequence[AABB],
    *,
    physics_steps: int | None,
    captured_contact_force_max_n: float | None,
    runtime_trace_frame_count: int | None,
    runtime_trace_max_contact_force_n: float | None,
    root: Path,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
) -> bool:
    """Check capture-time guard accounting without replacing replay checks."""

    guard = receipt.get("runtime_safety_guard")
    if not isinstance(guard, Mapping):
        _issue(
            issues,
            "runtime_safety_guard",
            "capture_receipt.json.runtime_safety_guard",
            "every City-Lite capture requires a fail-closed runtime safety receipt",
        )
        checks["runtime_safety_guard_verified"] = False
        return False
    expected_keys = {
        "schema",
        "enabled",
        "fail_closed",
        "status",
        "agent_center_radius_m",
        "flight_volume_m",
        "structural_aabb_count",
        "structural_aabb_geometry_sha256",
        "swept_aabb_clearance_m",
        "inter_agent",
        "contact",
        "evidence",
        "checks",
        "first_violation",
    }
    ok = set(guard) == expected_keys
    if guard.get("schema") != RUNTIME_SAFETY_SCHEMA:
        ok = False
    if guard.get("enabled") is not True or guard.get("fail_closed") is not True:
        ok = False
    if guard.get("status") != "passed" or guard.get("first_violation") is not None:
        ok = False
    if _finite_float_or_none(guard.get("agent_center_radius_m")) != CF2X_RUNTIME_GUARD_RADIUS_M:
        ok = False
    if not _exact_axis_volume(guard.get("flight_volume_m"), EXPECTED_FLIGHT_VOLUME):
        ok = False
    geometry_sha256 = aabb_geometry_sha256(structural_aabbs) if structural_aabbs else None
    if (
        guard.get("structural_aabb_count") != len(structural_aabbs)
        or guard.get("structural_aabb_geometry_sha256") != geometry_sha256
        or _finite_float_or_none(guard.get("swept_aabb_clearance_m")) != ROUTE_CLEARANCE_M
    ):
        ok = False
    inter_agent = guard.get("inter_agent")
    expected_inter_agent_keys = {
        "pair_count",
        "body_envelope_separation_m",
        "minimum_swept_center_separation_m",
        "provenance",
    }
    if not isinstance(inter_agent, Mapping) or set(inter_agent) != expected_inter_agent_keys:
        ok = False
    elif (
        inter_agent.get("pair_count") != INTER_AGENT_PAIR_COUNT
        or _finite_float_or_none(inter_agent.get("body_envelope_separation_m"))
        != INTER_AGENT_BODY_ENVELOPE_SEPARATION_M
        or _finite_float_or_none(inter_agent.get("minimum_swept_center_separation_m"))
        != INTER_AGENT_MINIMUM_CENTER_SEPARATION_M
        or inter_agent.get("provenance") != INTER_AGENT_SAFETY_PROVENANCE
    ):
        ok = False

    command = receipt.get("command")
    if not isinstance(command, Mapping):
        command = {}
        ok = False
    warmup_steps = command.get("warmup_steps")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        warmup_steps = None
        ok = False
    command_steps = command.get("steps")
    if isinstance(command_steps, bool) or not isinstance(command_steps, int) or command_steps < 1:
        command_steps = None
        ok = False
    dt_s = _finite_float_or_none(command.get("dt_s"))
    if dt_s is None or dt_s <= 0.0:
        ok = False
    if physics_steps is None or command_steps != physics_steps:
        ok = False

    contact = guard.get("contact")
    expected_contact_keys = {
        "prim_expression",
        "update_period_s",
        "every_physics_step",
        "force_abort_threshold_n",
        "force_abort_float32_cutoff_n",
        "body_count",
        "counterpart_attribution",
    }
    if not isinstance(contact, Mapping) or set(contact) != expected_contact_keys:
        ok = False
    else:
        if contact.get("prim_expression") != "/World/Swarm/Agent_.*/Robot/body":
            ok = False
        if _finite_float_or_none(contact.get("update_period_s")) != dt_s:
            ok = False
        if contact.get("every_physics_step") is not True or contact.get("body_count") != 1:
            ok = False
        if _finite_float_or_none(contact.get("force_abort_threshold_n")) != CONTACT_ABORT_FORCE_N:
            ok = False
        if (
            _finite_float_or_none(contact.get("force_abort_float32_cutoff_n"))
            != CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N
        ):
            ok = False
        if contact.get("counterpart_attribution") != (
            "unfiltered_root_body_net_normal_force; static_city_guarded_by_structural_aabb_sweep"
        ):
            ok = False

    evidence = guard.get("evidence")
    expected_evidence_keys = {"schema", "path", "sha256", "physics_frame_count"}
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_keys:
        ok = False
    else:
        trace_path = root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH
        if (
            evidence.get("schema") != RUNTIME_SAFETY_TRACE_SCHEMA
            or evidence.get("path") != RUNTIME_SAFETY_TRACE_RELATIVE_PATH
            or not is_sha256(evidence.get("sha256"))
            or not trace_path.is_file()
            or evidence.get("sha256") != sha256_file(trace_path)
            or evidence.get("physics_frame_count") != runtime_trace_frame_count
        ):
            ok = False

    guard_checks = guard.get("checks")
    expected_check_keys = {
        "post_reset_agent_center_checks",
        "post_reset_point_geometry_checks",
        "post_reset_inter_agent_pair_checks",
        "warmup_physics_steps_checked",
        "rollout_physics_steps_checked",
        "agent_center_checks",
        "swept_segments_checked",
        "inter_agent_pair_checks",
        "minimum_inter_agent_swept_separation_m",
        "contact_samples_checked",
        "max_contact_force_n",
        "contact_abort_count",
    }
    if not isinstance(guard_checks, Mapping) or set(guard_checks) != expected_check_keys:
        ok = False
    elif warmup_steps is not None and physics_steps is not None:
        expected_counts = {
            "post_reset_agent_center_checks": AGENT_COUNT,
            "post_reset_point_geometry_checks": AGENT_COUNT,
            "post_reset_inter_agent_pair_checks": INTER_AGENT_PAIR_COUNT,
            "warmup_physics_steps_checked": warmup_steps,
            "rollout_physics_steps_checked": physics_steps,
            "agent_center_checks": AGENT_COUNT * (1 + warmup_steps + physics_steps),
            "swept_segments_checked": AGENT_COUNT * (warmup_steps + physics_steps),
            "inter_agent_pair_checks": INTER_AGENT_PAIR_COUNT
            * (1 + warmup_steps + physics_steps),
            "contact_samples_checked": 1 + warmup_steps + physics_steps,
            "contact_abort_count": 0,
        }
        for key, expected in expected_counts.items():
            value = guard_checks.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                ok = False
        maximum_force = _finite_float_or_none(guard_checks.get("max_contact_force_n"))
        if (
            maximum_force is None
            or maximum_force < 0.0
            or maximum_force >= CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N
        ):
            ok = False
        if (
            maximum_force is not None
            and captured_contact_force_max_n is not None
            and maximum_force + 1.0e-9 < captured_contact_force_max_n
        ):
            ok = False
        minimum_pair_separation = _finite_float_or_none(
            guard_checks.get("minimum_inter_agent_swept_separation_m")
        )
        trace_minimum_pair_separation = _finite_float_or_none(
            checks.get("runtime_safety_trace_minimum_inter_agent_swept_separation_m")
        )
        if (
            minimum_pair_separation is None
            or minimum_pair_separation <= INTER_AGENT_MINIMUM_CENTER_SEPARATION_M
            or trace_minimum_pair_separation is None
            or not math.isclose(
                minimum_pair_separation,
                trace_minimum_pair_separation,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
        ):
            ok = False
        if (
            maximum_force is None
            or runtime_trace_max_contact_force_n is None
            or not math.isclose(
                maximum_force,
                runtime_trace_max_contact_force_n,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
        ):
            ok = False

    if not isinstance(scene, Mapping) or scene.get("runtime_safety_guard") != guard:
        ok = False
    if not ok:
        _issue(
            issues,
            "runtime_safety_guard",
            "capture_receipt.json.runtime_safety_guard",
            "runtime safety guard must match frozen City-Lite geometry, timing, and successful full-step counts",
        )
    checks["runtime_safety_guard_verified"] = ok
    return ok


def _validate_private_manifest(
    root: Path,
    manifest_path: Path | None,
    expected_sha256: object,
    issues: list[ValidationIssue],
    expected_task_variant_id: str = TASK_VARIANT_ID,
) -> tuple[bool, Mapping[str, Any] | None]:
    if not is_sha256(expected_sha256):
        _issue(issues, "evaluator_commitment", "capture_receipt.json", "missing evaluator SHA-256 commitment")
        return False, None
    if manifest_path is None:
        _issue(
            issues,
            "evaluator_manifest_required",
            "evaluator_private",
            "independent validation requires the external private evaluator manifest",
        )
        return False, None
    resolved = manifest_path.expanduser().resolve()
    if _is_within(resolved, root):
        _issue(issues, "evaluator_manifest_location", str(resolved), "private manifest must remain outside capture root")
        return False, None
    if not resolved.is_file():
        _issue(issues, "evaluator_manifest_missing", str(resolved), "external private manifest is missing")
        return False, None
    if sha256_file(resolved) != expected_sha256:
        _issue(issues, "evaluator_manifest_hash", str(resolved), "private manifest does not match capture commitment")
        return False, None
    payload = _read_json(resolved, issues)
    if payload is None:
        return False, None
    try:
        validate_external_private_evaluator_manifest(
            payload,
            city_lite_scene_contract_sha256=SCENE_CONTRACT_SHA256,
            city_lite_scene_payload_sha256=SCENE_CONTRACT_PAYLOAD_SHA256,
            expected_task_variant_id=expected_task_variant_id,
        )
    except PrivateEvaluatorManifestError as error:
        _issue(issues, "evaluator_manifest_contract", str(resolved), str(error))
        return False, None
    return True, payload


def _validate_runtime_lock_binding(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    runtime_lock_path: Path | None,
    issues: list[ValidationIssue],
    checks: dict[str, Any],
) -> bool:
    """Verify locked captures against the external v2 lock and live receipt."""

    binding = receipt.get("runtime_lock")
    if binding is None:
        checks["runtime_lock_verified"] = False
        return False
    if not isinstance(binding, Mapping):
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock", "runtime lock binding must be an object")
        checks["runtime_lock_verified"] = False
        return False
    expected_fields = {"path", "profile_id", "sha256"}
    if set(binding) != expected_fields:
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock", "runtime lock binding fields are incomplete")
    bound_path = runtime_lock_path.expanduser().resolve() if runtime_lock_path else None
    if bound_path is None and isinstance(binding.get("path"), str):
        bound_path = Path(binding["path"]).expanduser().resolve()
    if bound_path is None:
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock.path", "runtime lock path is required for independent validation")
        checks["runtime_lock_verified"] = False
        return False
    try:
        bound_path.relative_to(root)
    except ValueError:
        pass
    else:
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock.path", "runtime lock must remain outside the capture directory")
    if not bound_path.is_file():
        _issue(issues, "runtime_lock", str(bound_path), "bound runtime lock file is missing")
        checks["runtime_lock_verified"] = False
        return False
    try:
        lock = load_runtime_lock(bound_path)
    except (OSError, RuntimeLockError, ValueError) as error:
        _issue(issues, "runtime_lock", str(bound_path), f"runtime lock cannot be loaded: {error}")
        checks["runtime_lock_verified"] = False
        return False
    digest = runtime_lock_sha256(lock)
    if binding.get("sha256") != digest:
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock.sha256", "runtime lock hash does not match the external lock")
    if binding.get("profile_id") != lock.get("profile_id"):
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock.profile_id", "runtime lock profile does not match the external lock")

    preflight = receipt.get("preflight")
    runtime_check = None
    if isinstance(preflight, Mapping) and isinstance(preflight.get("checks"), list):
        runtime_check = next(
            (item for item in preflight["checks"] if isinstance(item, Mapping) and item.get("name") == "runtime_lock"),
            None,
        )
    preflight_ok = bool(
        isinstance(runtime_check, Mapping)
        and runtime_check.get("passed") is True
        and isinstance(runtime_check.get("value"), Mapping)
        and runtime_check["value"].get("status") == "passed"
        and runtime_check["value"].get("runtime_lock_sha256") == digest
        and runtime_check["value"].get("profile_id") == lock.get("profile_id")
    )
    if not preflight_ok:
        _issue(issues, "runtime_lock_preflight", "capture_receipt.json.preflight", "preflight does not contain a passed hash-bound runtime lock audit")

    runtime_live = receipt.get("runtime_live")
    live_issues: tuple[Any, ...] = ()
    live_shape_ok = isinstance(runtime_live, Mapping)
    if not isinstance(runtime_live, Mapping):
        _issue(issues, "runtime_lock_live", "capture_receipt.json.runtime_live", "locked capture must record live SimulationContext configuration")
    else:
        try:
            live_issues = compare_live_simulation(lock, runtime_live)
        except (KeyError, TypeError, ValueError) as error:
            live_shape_ok = False
            _issue(issues, "runtime_lock_live", "capture_receipt.json.runtime_live", f"live runtime observation is malformed: {error}")
        for live_issue in live_issues:
            _issue(issues, "runtime_lock_live", f"capture_receipt.json.runtime_live.{live_issue.path.lstrip('$.')}", live_issue.message)
        if runtime_live.get("configuration_observation") != "public_simulation_context_and_locked_cfg":
            _issue(issues, "runtime_lock_live", "capture_receipt.json.runtime_live.configuration_observation", "live runtime observation provenance is missing")
    verified = bool(
        set(binding) == expected_fields
        and binding.get("sha256") == digest
        and binding.get("profile_id") == lock.get("profile_id")
        and preflight_ok
        and live_shape_ok
        and not live_issues
    )
    checks["runtime_lock_verified"] = verified
    checks["runtime_lock_profile_id"] = lock.get("profile_id")
    checks["runtime_lock_sha256"] = digest
    checks["runtime_lock_preflight_verified"] = preflight_ok
    checks["runtime_lock_live_verified"] = live_shape_ok and not live_issues
    return verified


def validate_isaac_capture(
    root: Path,
    *,
    evaluator_manifest: Path | None = None,
    require_clean_source: bool = False,
    runtime_lock_path: Path | None = None,
    cf2x_runtime_calibration: Path | None = None,
    pose_threshold_m: float = 1.0e-4,
    orientation_threshold_rad: float = 2.0e-3,
) -> IsaacValidationReport:
    """Reopen and mechanically audit one raw capture directory."""

    root = root.resolve()
    issues: list[ValidationIssue] = []
    checks: dict[str, Any] = {}
    receipt_path = root / "capture_receipt.json"
    receipt = _read_json(receipt_path, issues)
    receipt_hash = sha256_file(receipt_path) if receipt_path.is_file() else None
    if receipt is None:
        return IsaacValidationReport(root, receipt_hash, checks, tuple(issues))

    # Native T2 has a deliberately different closed-world artifact inventory
    # and replays the command-to-actuator and RGB-D event chains.  Dispatch
    # before the legacy T1 checks so a native canary cannot be interpreted as
    # a normal Search3D capture merely by sharing a receipt schema.
    command = receipt.get("command")
    if (
        receipt.get("task_kind") == "native_t2_search_canary"
        or isinstance(command, Mapping)
        and command.get("control_mode") == "native_t2_canary"
    ):
        from .native_t2_validate import validate_native_t2_capture

        native = validate_native_t2_capture(
            root,
            evaluator_manifest=evaluator_manifest,
            cf2x_runtime_calibration=cf2x_runtime_calibration,
            runtime_lock_path=runtime_lock_path,
            require_clean_source=require_clean_source,
        )
        checks = {
            "validation_profile": "native_t2_canary",
            **dict(native.checks),
        }
        return IsaacValidationReport(
            root,
            receipt_hash,
            checks,
            tuple(ValidationIssue(issue.code, issue.path, issue.message) for issue in native.issues),
        )
    if receipt.get("schema") != CAPTURE_SCHEMA:
        _issue(issues, "capture_schema", "capture_receipt.json", "unsupported capture schema")
    if receipt.get("status") != "captured" or receipt.get("ok") is not True:
        _issue(issues, "capture_status", "capture_receipt.json", "capture did not complete successfully")
    if receipt.get("task_kind") != "search3d":
        _issue(issues, "task_kind", "capture_receipt.json", "formal Isaac capture must be Search3D")
    if receipt.get("information_profile") != "multisensor_rgbd_lidar_imu_state":
        _issue(issues, "information_profile", "capture_receipt.json", "capture must use the no-radar multisensor profile")
    if require_clean_source and receipt.get("source_worktree_dirty") is not False:
        _issue(issues, "dirty_source", "capture_receipt.json", "formal validation requires a clean source worktree")
    revision = receipt.get("source_revision")
    if not isinstance(revision, str) or not 7 <= len(revision) <= 64 or any(char not in "0123456789abcdef" for char in revision):
        _issue(issues, "source_revision", "capture_receipt.json", "source revision must be a Git hex object ID")
    if not is_sha256(receipt.get("source_tree_sha256")):
        _issue(issues, "source_tree", "capture_receipt.json", "tracked source-tree SHA-256 is required")
    collection_binding = receipt.get("collection_binding")
    collection_binding_ok = False
    condition_request = receipt.get("condition_request")
    condition_request_ok = collection_binding is None
    if collection_binding is not None:
        binding_issues = validate_collection_binding(collection_binding)
        for binding_issue in binding_issues:
            suffix = binding_issue.path.lstrip("$").lstrip(".")
            _issue(
                issues,
                f"collection_{binding_issue.code}",
                f"capture_receipt.json.collection_binding.{suffix}"
                if suffix
                else "capture_receipt.json.collection_binding",
                binding_issue.message,
            )
        command = receipt.get("command")
        seed_matches = (
            isinstance(command, Mapping)
            and isinstance(collection_binding, Mapping)
            and command.get("seed") == collection_binding.get("episode_seed")
        )
        if not seed_matches:
            _issue(
                issues,
                "collection_seed",
                "capture_receipt.json.command.seed",
                "runtime seed must equal the predeclared collection episode seed",
            )
        collection_binding_ok = not binding_issues and seed_matches
        # Older pilot receipts remain readable, but they are explicitly not
        # eligible for protocol-bound packing without this request.
        condition_issues = (
            validate_condition_request(condition_request, binding=collection_binding)
            if condition_request is not None
            else ()
        )
        for condition_issue in condition_issues:
            _issue(issues, condition_issue["code"], condition_issue["path"], condition_issue["message"])
        condition_request_ok = condition_request is not None and not condition_issues
    elif condition_request is not None:
        _issue(
            issues,
            "condition_request_without_binding",
            "capture_receipt.json.condition_request",
            "condition request requires a collection binding",
        )
    checks["collection_binding_present"] = collection_binding is not None
    checks["collection_binding_verified"] = collection_binding_ok
    checks["condition_request_present"] = condition_request is not None
    checks["condition_request_verified"] = condition_request_ok
    evaluator_sha256 = receipt.get("evaluator_manifest_sha256")
    retention = receipt.get("evaluator_manifest_retention")
    if retention is not None:
        retention_ok = (
            isinstance(retention, Mapping)
            and set(retention)
            == {"kind", "sha256", "bytes", "path_released", "payload_released"}
            and retention.get("kind") == PRIVATE_MANIFEST_RETENTION_KIND
            and retention.get("sha256") == evaluator_sha256
            and isinstance(retention.get("bytes"), int)
            and not isinstance(retention.get("bytes"), bool)
            and 0 < retention["bytes"] <= PRIVATE_MANIFEST_RETENTION_MAX_BYTES
            and retention.get("path_released") is False
            and retention.get("payload_released") is False
        )
        if not retention_ok:
            _issue(
                issues,
                "evaluator_manifest_retention",
                "capture_receipt.json.evaluator_manifest_retention",
                "retention commitment must be exact, hash-bound, bounded, and path/payload-free",
            )
    evaluator_verified, private_evaluator_payload = _validate_private_manifest(
        root, evaluator_manifest, evaluator_sha256, issues
    )
    checks["evaluator_manifest_sha256"] = evaluator_sha256
    checks["evaluator_manifest_verified"] = evaluator_verified
    checks["private_target_geometry_verified"] = False
    checks["private_target_region_verified"] = False
    checks["private_target_visibility_verified"] = False
    expected_private_target_count: int | None = None
    private_target_ids: tuple[str, ...] = ()
    private_target_positions: tuple[tuple[float, float, float], ...] = ()
    if isinstance(private_evaluator_payload, Mapping):
        private_targets = private_evaluator_payload.get("targets")
        if isinstance(private_targets, Sequence) and not isinstance(private_targets, (str, bytes)):
            expected_private_target_count = len(private_targets)
            private_target_ids = tuple(
                str(target["target_id"])
                for target in private_targets
                if isinstance(target, Mapping) and isinstance(target.get("target_id"), str)
            )
            private_target_positions = tuple(
                tuple(float(coordinate) for coordinate in target["position_w_m"])
                for target in private_targets
                if (
                    isinstance(target, Mapping)
                    and isinstance(target.get("position_w_m"), Sequence)
                    and not isinstance(target.get("position_w_m"), (str, bytes))
                    and len(target["position_w_m"]) == 3
                )
            )
    _validate_runtime_target_usd_closure(
        receipt,
        required=collection_binding is not None,
        expected_target_count=expected_private_target_count,
        issues=issues,
        checks=checks,
    )
    capture_integrity = receipt.get("capture_integrity")
    if not isinstance(capture_integrity, Mapping):
        _issue(issues, "capture_integrity", "capture_receipt.json", "capture integrity declaration is missing")
        capture_integrity = {}
    for key, expected in (
        ("online_capture", True),
        ("queue_used", False),
        ("queue_overflow", False),
        ("silent_frame_drop", False),
        ("synchronous_sensor_reads", True),
    ):
        if capture_integrity.get(key) is not expected:
            _issue(issues, "capture_integrity", f"capture_receipt.json.capture_integrity.{key}", f"must be {expected}")
    _validate_runtime_lock_binding(
        receipt,
        root=root,
        runtime_lock_path=runtime_lock_path,
        issues=issues,
        checks=checks,
    )
    if receipt.get("runtime_lock") is not None:
        locked_sensor_contract = tuple(SENSOR_PHASE_EVENT_CODES)
        if capture_integrity.get("sensor_step_order") != list(locked_sensor_contract):
            _issue(
                issues,
                "capture_integrity",
                "capture_receipt.json.capture_integrity.sensor_step_order",
                "locked captures must distinguish the per-step safety contact read from the retained sensor phase",
            )
        for key in (
            "per_physics_step_safety_contact_reads",
            "retained_contact_read_in_synchronous_sensor_phase",
        ):
            if capture_integrity.get(key) is not True:
                _issue(
                    issues,
                    "capture_integrity",
                    f"capture_receipt.json.capture_integrity.{key}",
                    "locked captures must bind both safety and retained contact reads",
                )
    claim_boundary = receipt.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        _issue(issues, "claim_boundary", "capture_receipt.json", "claim boundary must be an object")
        claim_boundary = {}
    if claim_boundary.get("formal_benchmark_admission") is not False:
        _issue(issues, "claim_boundary", "capture_receipt.json", "raw capture must not self-admit")
    if claim_boundary.get("radar_profile_eligible") is not False:
        _issue(issues, "radar_claim", "capture_receipt.json", "unvalidated radar profile must remain ineligible")
    for key in ("hardware_validated", "foundation_model_executed", "semantic_labels_policy_visible"):
        if claim_boundary.get(key) is not False:
            _issue(issues, "claim_boundary", f"capture_receipt.json.claim_boundary.{key}", "must remain false")

    bound = receipt.get("artifact_hashes")
    if not isinstance(bound, Mapping) or set(bound) not in (EXPECTED_ARTIFACTS, EXPECTED_ARTIFACTS | CONTROL_ARTIFACTS):
        _issue(issues, "artifact_inventory", "capture_receipt.json", "capture artifact inventory is not exact")
        bound = {}
    video_artifacts = _validate_post_validation_video_artifacts(
        root,
        receipt=receipt,
        receipt_sha256=receipt_hash,
        bound_artifacts=bound,
        issues=issues,
    )
    expected_artifact_sets = (
        EXPECTED_ARTIFACTS,
        EXPECTED_ARTIFACTS | CONTROL_ARTIFACTS,
        EXPECTED_ARTIFACTS | video_artifacts,
        EXPECTED_ARTIFACTS | CONTROL_ARTIFACTS | video_artifacts,
    )
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"capture_receipt.json", "capture_receipt.sha256", "independent_validation.json"}
    }
    if present not in expected_artifact_sets:
        expected_for_diff = EXPECTED_ARTIFACTS | (
            CONTROL_ARTIFACTS if "capture_start.json" in present else frozenset()
        ) | video_artifacts
        _issue(issues, "closed_world", ".", f"unexpected/missing capture files: {sorted(present ^ expected_for_diff)}")
    for relative in EXPECTED_ARTIFACTS:
        path = root / PurePosixPath(relative)
        item = bound.get(relative) if isinstance(bound, Mapping) else None
        if not path.is_file() or not isinstance(item, Mapping):
            _issue(issues, "missing_artifact", relative, "bound artifact is missing")
            continue
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            _issue(issues, "artifact_hash", relative, "size or SHA-256 does not match capture receipt")
    checksum_path = root / "capture_receipt.sha256"
    if not checksum_path.is_file() or checksum_path.read_text(encoding="ascii", errors="replace") != f"{receipt_hash}  capture_receipt.json\n":
        _issue(issues, "receipt_checksum", "capture_receipt.sha256", "receipt checksum file is missing or stale")

    scene = _read_json(root / "scene.json", issues)
    public_task = _read_json(root / "public_task.json", issues)
    outcome = _read_json(root / "task_outcome.json", issues)
    calibration = _read_json(root / "calibration.json", issues)
    progress = _read_json(root / "capture_progress.json", issues)
    _scan_policy_json(
        receipt,
        "capture_receipt.json",
        issues,
        non_policy_metadata_paths=_CAPTURE_RECEIPT_NON_POLICY_METADATA_PATHS,
    )
    _scan_policy_json(scene, "scene.json", issues)
    _scan_policy_json(public_task, "public_task.json", issues)
    _scan_policy_json(outcome, "task_outcome.json", issues)
    _scan_policy_json(calibration, "calibration.json", issues)
    for payload, relative in (
        (receipt, "capture_receipt.json"),
        (scene, "scene.json"),
        (public_task, "public_task.json"),
        (outcome, "task_outcome.json"),
        (calibration, "calibration.json"),
        (progress, "capture_progress.json"),
    ):
        _scan_public_private_artifact_json(
            payload,
            relative,
            issues,
            private_target_ids=private_target_ids,
            private_target_positions=private_target_positions,
        )
    structural_aabbs: tuple[AABB, ...] = ()
    if scene is not None:
        structural_aabbs = _validate_city_lite_scene(
            scene,
            evaluator_sha256=evaluator_sha256,
            issues=issues,
            checks=checks,
        )
        if scene.get("agent_count") != AGENT_COUNT or scene.get("agent_prim_expression") != "/World/Swarm/Agent_.*/Robot":
            _issue(issues, "scene_agents", "scene.json", "scene must bind eight swarm articulations")
        if scene.get("fresh_stage") is not True or scene.get("legacy_route_or_target_imported") is not False:
            _issue(issues, "scene_provenance", "scene.json", "capture must use a fresh public-only stage")
        if (
            scene.get("search_object_prim_count") != TARGET_COUNT
            or scene.get("search_object_paths_listed") is not False
            or scene.get("object_coordinates_in_policy_inputs") is not False
        ):
            _issue(issues, "scene_privacy", "scene.json", "public scene must omit private object coordinates and paths")
        markers = scene.get("identity_markers")
        if not isinstance(markers, list) or len(markers) != AGENT_COUNT:
            _issue(issues, "identity_markers", "scene.json", "scene must bind one visible identity marker per agent")
        marker_provenance = scene.get("identity_marker_provenance")
        expected_marker_paths = markers if isinstance(markers, list) else []
        marker_records = (
            marker_provenance.get("markers")
            if isinstance(marker_provenance, Mapping)
            else None
        )
        marker_provenance_valid = bool(
            isinstance(marker_provenance, Mapping)
            and marker_provenance.get("schema")
            == "org.rivermark.isaac-cf2x-identity-marker.v1"
            and marker_provenance.get("shape") == "sphere"
            and marker_provenance.get("radius_m") == IDENTITY_MARKER_RADIUS_M
            and marker_provenance.get("collision_enabled") is False
            and marker_provenance.get("body_relative_translation_m")
            == [-0.045, 0.0, 0.075]
            and marker_provenance.get("root_semantic_tags")
            == [["class", "cf2x"], ["class", "agent_identity"]]
            and isinstance(marker_records, list)
            and len(marker_records) == AGENT_COUNT
            and len(expected_marker_paths) == AGENT_COUNT
            and all(
                isinstance(record, Mapping)
                and record.get("agent_id") == agent_id
                and record.get("prim_path") == expected_marker_paths[agent_id]
                and record.get("semantic_tags")
                == [["class", "agent_identity"], ["agent_id", str(agent_id)]]
                for agent_id, record in enumerate(marker_records)
            )
        )
        if not marker_provenance_valid:
            _issue(
                issues,
                "identity_marker_provenance",
                "scene.json.identity_marker_provenance",
                "identity markers must bind collision-free geometry and CF2X semantic identity provenance",
            )
        if scene.get("overview_route_witness_schedule") != _public_route_witness_schedule():
            _issue(
                issues,
                "route_witness_scene_contract",
                "scene.json.overview_route_witness_schedule",
                "scene must bind the exact public fixed route-witness schedule",
            )
        if scene.get("public_task_sha256") != sha256_file(root / "public_task.json"):
            _issue(issues, "public_task_binding", "scene.json", "scene does not bind public task")
    routes: np.ndarray | None = None
    if public_task is not None:
        if (
            public_task.get("schema") != "org.rivermark.public-search-task.v1"
            or public_task.get("task_kind") != "search3d"
            or public_task.get("task_variant_id") != TASK_VARIANT_ID
            or public_task.get("agent_count") != AGENT_COUNT
            or public_task.get("route_conditioning") != "public_only"
            or public_task.get("object_coordinates_in_policy_inputs") is not False
        ):
            _issue(issues, "public_task_contract", "public_task.json", "public Search3D task contract is invalid")
        try:
            routes = np.asarray(public_task.get("routes_w_m"), dtype=np.float64)
        except (TypeError, ValueError):
            routes = None
        if routes is None or routes.ndim != 3 or routes.shape[0] != AGENT_COUNT or routes.shape[1] < 3 or routes.shape[2] != 3 or not np.isfinite(routes).all():
            _issue(issues, "public_routes", "public_task.json", "routes must be finite [8,W,3] with W >= 3")
            routes = None
        if routes is not None:
            outside = [
                (agent_id, waypoint_id)
                for agent_id in range(AGENT_COUNT)
                for waypoint_id, point in enumerate(routes[agent_id])
                if not CITY_LITE_COMMAND_VOLUME_W_M.contains(point)
            ]
            if outside:
                _issue(
                    issues,
                    "route_command_volume",
                    "public_task.json.routes_w_m",
                    f"waypoints outside City-Lite command volume: {outside[:8]}",
                )
            route_family = public_task.get("route_family_id")
            expected_starts = (
                TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M.get(route_family)
                if isinstance(route_family, str)
                else None
            )
            bad_starts = (
                list(range(AGENT_COUNT))
                if expected_starts is None
                else [
                    agent_id
                    for agent_id in range(AGENT_COUNT)
                    if not np.allclose(
                        routes[agent_id, 0],
                        np.asarray(expected_starts[agent_id]),
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                ]
            )
            if bad_starts:
                _issue(
                    issues,
                    "route_start",
                    "public_task.json.routes_w_m",
                    f"routes do not start at target-free City-Lite anchors: {bad_starts}",
                )
            if not structural_aabbs:
                _issue(issues, "route_clearance", "public_task.json.routes_w_m", "route sweep requires structural AABBs")
            else:
                clearance_violation: tuple[int, int, str] | None = None
                for agent_id in range(AGENT_COUNT):
                    for segment_id in range(routes.shape[1] - 1):
                        start, end = routes[agent_id, segment_id : segment_id + 2]
                        if float(np.linalg.norm(end - start)) <= 1.0e-12:
                            clearance_violation = (agent_id, segment_id, "zero-length segment")
                            break
                        for box in structural_aabbs:
                            if segment_intersects_aabb(
                                start,
                                end,
                                box,
                                clearance_m=ROUTE_CLEARANCE_M,
                            ):
                                clearance_violation = (
                                    agent_id,
                                    segment_id,
                                    box.source_prim,
                                )
                                break
                        if clearance_violation is not None:
                            break
                    if clearance_violation is not None:
                        break
                if clearance_violation is not None:
                    agent_id, segment_id, source = clearance_violation
                    _issue(
                        issues,
                        "route_clearance",
                        "public_task.json.routes_w_m",
                        f"agent {agent_id} segment {segment_id} violates {ROUTE_CLEARANCE_M} m clearance from {source}",
                    )

            geometry_sha256 = aabb_geometry_sha256(structural_aabbs) if structural_aabbs else None
            route_contract = public_task.get("route_contract")
            route_contract_ok = isinstance(route_contract, Mapping) and (
                route_contract.get("geometry_source") == "citylite_structural_aabb_v1"
                and route_contract.get("clearance_m") == ROUTE_CLEARANCE_M
                and route_contract.get("aabb_geometry_sha256") == geometry_sha256
                and route_contract.get("all_waypoints_in_command_volume") is True
                and route_contract.get("all_segments_clear") is True
            )
            if not route_contract_ok:
                _issue(issues, "route_contract", "public_task.json.route_contract", "route contract is missing, stale, or not bound to structural geometry")
            checks["route_clearance_m"] = ROUTE_CLEARANCE_M
            checks["public_route_segment_count"] = int(AGENT_COUNT * (routes.shape[1] - 1))
    _validate_literal_city_lite_fleet_spawn(
        receipt,
        scene,
        issues,
        checks,
        routes_w_m=routes.tolist() if routes is not None else None,
    )
    if private_evaluator_payload is not None and routes is not None and structural_aabbs:
        try:
            command = receipt.get("command")
            execution_window: Mapping[str, Any] | None = None
            if receipt.get("target_visibility_execution_window") is not None:
                if not isinstance(command, Mapping):
                    raise PrivateEvaluatorManifestError(
                        "capture receipt command is required to reconstruct target visibility"
                    )
                waypoint_segment_seconds = (
                    public_task.get("waypoint_segment_seconds")
                    if isinstance(public_task, Mapping)
                    else None
                )
                if waypoint_segment_seconds is None:
                    waypoint_segment_seconds = PUBLIC_ROUTE_WAYPOINT_SEGMENT_SECONDS
                execution_window = target_visibility_execution_window(
                    dt_s=float(command["dt_s"]),
                    warmup_steps=command["warmup_steps"],
                    rollout_steps=command["steps"],
                    capture_stride=command["capture_stride"],
                    waypoint_segment_seconds=float(waypoint_segment_seconds),
                )
                if dict(receipt["target_visibility_execution_window"]) != dict(execution_window):
                    raise PrivateEvaluatorManifestError(
                        "capture receipt target-visibility execution window does not match its command"
                    )
                validate_private_target_execution_window(
                    private_evaluator_payload, execution_window=execution_window
                )
            placement = validate_private_target_geometry(
                private_evaluator_payload,
                structural_aabbs=structural_aabbs,
                public_routes_w_m=routes.tolist(),
                city_lite_scene_contract_sha256=SCENE_CONTRACT_SHA256,
                city_lite_scene_payload_sha256=SCENE_CONTRACT_PAYLOAD_SHA256,
                execution_window=execution_window,
            )
        except (KeyError, TypeError, ValueError, PrivateEvaluatorManifestError) as error:
            _issue(issues, "evaluator_target_geometry", "evaluator_private", str(error))
        else:
            checks["private_target_geometry_verified"] = True
            checks["private_target_minimum_route_separation_m"] = placement[
                "minimum_route_separation_m"
            ]
            checks["private_target_minimum_pairwise_separation_m"] = placement[
                "minimum_pairwise_separation_m"
            ]
            checks["private_target_region_verified"] = True
            checks["private_target_region_id"] = placement["target_region_id"]
            checks["private_target_visibility_verified"] = True
            checks["private_target_visibility_bucket"] = placement[
                "visibility_bucket"
            ]
    elif private_evaluator_payload is not None:
        _issue(
            issues,
            "evaluator_target_geometry",
            "evaluator_private",
            "private target geometry cannot be audited without valid City-Lite AABBs and public routes",
        )
    if outcome is not None:
        observability = outcome.get("target_observability")
        if (
            outcome.get("schema") != T1_OBSERVABILITY_OUTCOME_SCHEMA
            or outcome.get("track") != T1_DATA_TRACK_ID
            or outcome.get("task_variant_id") != TASK_VARIANT_ID
            or outcome.get("scoring_status") != "not_scored"
            or outcome.get("search_score") is not None
            or outcome.get("object_count") != TARGET_COUNT
            or not isinstance(observability, Mapping)
            or outcome.get("policy_confirmation_events_present") is not False
            or outcome.get("closed_loop_scoring_eligible") is not False
            or outcome.get("private_coordinates_released") is not False
        ):
            _issue(
                issues,
                "task_outcome",
                "task_outcome.json",
                "T1 outcome must record observability and remain explicitly unscored",
            )
        if outcome.get("private_manifest_commitment_sha256") != evaluator_sha256:
            _issue(issues, "evaluator_binding", "task_outcome.json", "outcome does not bind capture evaluator commitment")
        if outcome.get("state_action_sha256") != sha256_file(root / "streams/state_action.npz"):
            _issue(issues, "outcome_binding", "task_outcome.json", "outcome does not bind state/action stream")
    if calibration is not None:
        radar = calibration.get("radar")
        if not isinstance(radar, Mapping) or radar.get("status") != "not_captured" or radar.get("fail_closed") is not True:
            _issue(issues, "radar_calibration", "calibration.json", "radar must remain explicitly unavailable")
    lidar_max_distance_m = _calibrated_lidar_max_distance_m(calibration, issues)
    checks["lidar_max_distance_m"] = lidar_max_distance_m
    visual_intrusion_gate_declaration = _calibrated_visual_intrusion_gate(
        calibration, issues
    )
    checks["visual_intrusion_gate_declared"] = visual_intrusion_gate_declaration is not None
    onboard_content_gate_declaration = _calibrated_onboard_content_gate(
        calibration, issues
    )
    checks["onboard_content_gate_declared"] = onboard_content_gate_declaration is not None
    checks["onboard_scene_content_verified"] = False
    route_witness_declaration = _calibrated_route_witness_schedule(calibration, issues)
    checks["route_witness_camera_declared"] = route_witness_declaration is not None
    route_witness_visibility_declaration = _calibrated_route_witness_visibility_gate(
        calibration, issues
    )
    checks["route_witness_visibility_gate_declared"] = (
        route_witness_visibility_declaration is not None
    )
    overview_far_clip_m, overview_gate_declaration = _calibrated_overview_content_gate(
        calibration, issues
    )
    checks["overview_content_gate_declared"] = overview_gate_declaration is not None
    overview_archive_declaration = _calibrated_overview_evidence_archive(
        calibration, issues
    )
    checks["overview_archive_declared"] = overview_archive_declaration is not None

    paths = {
        "camera": root / "sensors/camera_poses.npz",
        "contact": root / "sensors/contact.npz",
        "imu": root / "sensors/imu.npz",
        "lidar": root / "sensors/lidar.npz",
        "camera_data": root / "sensors/onboard_rgbd.npz",
        "semantic": root / "learning_labels/semantic_segmentation.npz",
        "overview": root / "sensors/overview_rgb.npz",
        "runtime_safety": root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
        "sensor_phase": root / SENSOR_PHASE_TRACE_RELATIVE_PATH,
        "task": root / "streams/public_task.npz",
        "messages": root / "streams/public_messages.npz",
        "state": root / "streams/state_action.npz",
    }
    frame_payload_names = {"camera_data", "semantic", "overview"}
    payloads = {
        name: _load_npz(path, issues)
        for name, path in paths.items()
        if name not in frame_payload_names
    }
    frame_payloads = {
        name: _load_frame_payload(paths[name], issues)
        for name in frame_payload_names
    }
    sensor_payloads = {
        name: value
        for name, value in payloads.items()
        if name not in {"state", "runtime_safety", "sensor_phase"} and value is not None
    }
    sensor_payloads.update(
        {
            name: {"timestamps_ns": value.timestamps_ns}
            for name, value in frame_payloads.items()
            if value is not None and name != "overview"
        }
    )
    timestamps = _timestamps(sensor_payloads, issues)
    state = payloads["state"]
    expected_state = {
        "command_time_ns", "effective_time_ns", "root_pos_w_m", "root_quat_wxyz", "root_lin_vel_w_mps",
        "root_ang_vel_b_radps", "desired_pos_w_m", "desired_vel_w_mps", "target_thrust_n", "applied_thrust_n",
    }
    if _exact_arrays(state, expected_state, "streams/state_action.npz", issues):
        assert state is not None
        command, effective = state["command_time_ns"], state["effective_time_ns"]
        steps = command.shape[0] if command.ndim == 1 else 0
        if command.dtype != np.int64 or effective.dtype != np.int64 or command.ndim != 1 or effective.shape != (steps,):
            _issue(issues, "action_time", "streams/state_action.npz", "action times must be int64 [S]")
        elif not (np.all(np.diff(command) > 0) and np.all(np.diff(effective) > 0) and np.all(command < effective)):
            _issue(issues, "action_causality", "streams/state_action.npz", "command times must precede effective times")
        for key in expected_state - {"command_time_ns", "effective_time_ns"}:
            value = state[key]
            _finite(f"state_action.{key}", value, issues)
            expected_tail = (AGENT_COUNT, 4) if key.endswith("thrust_n") or key == "root_quat_wxyz" else (AGENT_COUNT, 3)
            if value.shape != (steps, *expected_tail):
                _issue(issues, "state_shape", f"state_action.{key}", f"expected {(steps, *expected_tail)}, got {value.shape}")
        applied = state["applied_thrust_n"]
        target = state["target_thrust_n"]
        if not np.any(applied > 0.0) or float(np.max(applied)) <= 1.0e-4:
            _issue(issues, "zero_applied_thrust", "state_action.applied_thrust_n", "applied thrust is not physically active")
        if not np.any(np.abs(np.diff(applied, axis=0)) > 1.0e-6):
            _issue(issues, "static_applied_thrust", "state_action.applied_thrust_n", "actuator response is static")
        if np.any(applied < -1.0e-7) or np.any(applied > 0.180001) or np.any(target < 0.0) or np.any(target > 0.180001):
            _issue(issues, "thrust_range", "streams/state_action.npz", "thrust is outside configured bounds")
        positions = state["root_pos_w_m"]
        if positions.shape == (steps, AGENT_COUNT, 3) and steps >= 2:
            flight_minimum = (
                np.asarray(CITY_LITE_FLIGHT_VOLUME_W_M.minimum)
                + CF2X_RUNTIME_GUARD_RADIUS_M
            )
            flight_maximum = (
                np.asarray(CITY_LITE_FLIGHT_VOLUME_W_M.maximum)
                - CF2X_RUNTIME_GUARD_RADIUS_M
            )
            trajectory_in_volume = bool(
                np.all(positions >= flight_minimum) and np.all(positions <= flight_maximum)
            )
            checks["trajectory_in_city_lite_flight_volume"] = trajectory_in_volume
            if not trajectory_in_volume:
                _issue(
                    issues,
                    "trajectory_flight_volume",
                    "state_action.root_pos_w_m",
                    "physical trajectory leaves the frozen City-Lite flight volume after the CF2X body margin",
                )
            trajectory_clearance_verified = False
            if structural_aabbs:
                trajectory_clearance_verified = True
                clearance_violation: tuple[int, int, int, str] | None = None
                for step_id in range(steps - 1):
                    for agent_id in range(AGENT_COUNT):
                        start, end = positions[step_id : step_id + 2, agent_id]
                        for box_id, box in enumerate(structural_aabbs):
                            if segment_intersects_aabb(
                                start,
                                end,
                                box,
                                clearance_m=ROUTE_CLEARANCE_M,
                            ):
                                clearance_violation = (
                                    step_id,
                                    agent_id,
                                    box_id,
                                    box.source_prim,
                                )
                                break
                        if clearance_violation is not None:
                            break
                    if clearance_violation is not None:
                        break
                if clearance_violation is not None:
                    trajectory_clearance_verified = False
                    step_id, agent_id, _, source = clearance_violation
                    _issue(
                        issues,
                        "trajectory_clearance",
                        "state_action.root_pos_w_m",
                        f"step {step_id} agent {agent_id} violates {ROUTE_CLEARANCE_M} m clearance from {source}",
                    )
            checks["trajectory_segment_clearance_verified"] = trajectory_clearance_verified
            path_lengths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=-1), axis=0)
            max_displacements = np.max(np.linalg.norm(positions - positions[0:1], axis=-1), axis=0)
            checks["minimum_agent_path_length_m"] = float(np.min(path_lengths))
            checks["minimum_agent_max_displacement_m"] = float(np.min(max_displacements))
            if np.any(path_lengths <= 0.15) or np.any(max_displacements <= 0.10):
                _issue(issues, "insufficient_search_movement", "state_action.root_pos_w_m", "every agent must execute nontrivial 3D search movement")
        desired = state["desired_pos_w_m"]
        if desired.shape == (steps, AGENT_COUNT, 3) and not np.any(np.abs(np.diff(desired, axis=0)) > 1.0e-5):
            _issue(issues, "static_high_level_action", "state_action.desired_pos_w_m", "public high-level route did not change")
        checks["physics_steps"] = steps
        checks["applied_thrust_max_n"] = float(np.max(applied))

    task_stream = payloads["task"]
    expected_task = {
        "timestamps_ns", "waypoint_index", "waypoint_progress", "desired_waypoint_w_m",
        "distance_to_waypoint_m", "waypoint_reached", "action_mode", "coverage_cell_id", "task_time_s",
    }
    if _exact_arrays(task_stream, expected_task, "streams/public_task.npz", issues) and timestamps is not None:
        assert task_stream is not None
        sample_count = len(timestamps)
        waypoint_index = task_stream["waypoint_index"]
        if waypoint_index.dtype != np.int64 or waypoint_index.shape != (sample_count, AGENT_COUNT):
            _issue(issues, "waypoint_shape", "streams/public_task.npz", "waypoint_index must be int64 [T,8]")
        else:
            route_count = int(routes.shape[1]) if routes is not None else 0
            if route_count and (np.any(waypoint_index < 1) or np.any(waypoint_index >= route_count)):
                _issue(issues, "waypoint_range", "streams/public_task.npz", "waypoint index is outside public routes")
            if len(np.unique(waypoint_index)) < 2:
                _issue(issues, "waypoint_static", "streams/public_task.npz", "Search3D route never advances")
        for key in ("waypoint_progress", "distance_to_waypoint_m", "task_time_s"):
            value = task_stream[key]
            _finite(f"public_task.{key}", value, issues)
            if value.shape != (sample_count, AGENT_COUNT):
                _issue(issues, "task_shape", f"public_task.{key}", "expected [T,8]")
        desired_waypoint = task_stream["desired_waypoint_w_m"]
        _finite("public_task.desired_waypoint_w_m", desired_waypoint, issues)
        if desired_waypoint.shape != (sample_count, AGENT_COUNT, 3):
            _issue(issues, "task_shape", "public_task.desired_waypoint_w_m", "expected [T,8,3]")
        elif routes is not None and waypoint_index.shape == (sample_count, AGENT_COUNT):
            expected_waypoints = np.empty_like(desired_waypoint, dtype=np.float64)
            for agent_id in range(AGENT_COUNT):
                expected_waypoints[:, agent_id] = routes[agent_id, waypoint_index[:, agent_id]]
            if not np.allclose(desired_waypoint, expected_waypoints, rtol=0.0, atol=1.0e-5):
                _issue(issues, "waypoint_binding", "streams/public_task.npz", "desired waypoints do not match public task")
        reached = task_stream["waypoint_reached"]
        if reached.dtype != np.bool_ or reached.shape != (sample_count, AGENT_COUNT):
            _issue(issues, "task_shape", "public_task.waypoint_reached", "expected bool [T,8]")
        action_mode = task_stream["action_mode"]
        if action_mode.shape != (sample_count, AGENT_COUNT) or not np.issubdtype(action_mode.dtype, np.integer) or np.any((action_mode < 0) | (action_mode > 1)):
            _issue(issues, "task_shape", "public_task.action_mode", "expected integer transit/hold flags [T,8]")
        coverage = task_stream["coverage_cell_id"]
        coverage_ok = coverage.shape == (sample_count, AGENT_COUNT) and np.issubdtype(coverage.dtype, np.integer)
        if coverage_ok:
            expected_agents = np.repeat(np.arange(AGENT_COUNT, dtype=coverage.dtype)[None, :], sample_count, axis=0)
            coverage_ok = np.array_equal(coverage, expected_agents)
        if not coverage_ok:
            _issue(issues, "coverage_identity", "public_task.coverage_cell_id", "coverage cells must remain agent-partitioned")
        checks["waypoint_indices_observed"] = int(len(np.unique(waypoint_index)))

    messages = payloads["messages"]
    expected_messages = {
        "timestamps_ns", "sender_agent_id", "message_sequence", "message_waypoint_index",
        "message_position_w_m", "message_velocity_w_mps", "message_flags",
    }
    if _exact_arrays(messages, expected_messages, "streams/public_messages.npz", issues) and timestamps is not None:
        assert messages is not None
        sample_count = len(timestamps)
        sender = messages["sender_agent_id"]
        expected_sender = np.repeat(np.arange(AGENT_COUNT, dtype=np.int64)[None, :], sample_count, axis=0)
        if sender.dtype != np.int64 or sender.shape != expected_sender.shape or not np.array_equal(sender, expected_sender):
            _issue(issues, "message_sender", "streams/public_messages.npz", "each sample must contain senders 0..7")
        sequence = messages["message_sequence"]
        expected_sequence = np.repeat(np.arange(sample_count, dtype=np.int64)[:, None], AGENT_COUNT, axis=1)
        if sequence.dtype != np.int64 or sequence.shape != expected_sequence.shape or not np.array_equal(sequence, expected_sequence):
            _issue(issues, "message_sequence", "streams/public_messages.npz", "message sequences must be complete and contiguous")
        for key in ("message_position_w_m", "message_velocity_w_mps"):
            value = messages[key]
            _finite(f"messages.{key}", value, issues)
            if value.shape != (sample_count, AGENT_COUNT, 3):
                _issue(issues, "message_shape", f"messages.{key}", "expected [T,8,3]")
        flags = messages["message_flags"]
        if flags.shape != (sample_count, AGENT_COUNT) or not np.issubdtype(flags.dtype, np.integer) or not np.all(flags == 1):
            _issue(issues, "message_flags", "streams/public_messages.npz", "every synchronous broadcast must be marked delivered")
        if task_stream is not None and "waypoint_index" in task_stream:
            if not np.array_equal(messages["message_waypoint_index"], task_stream["waypoint_index"]):
                _issue(issues, "message_task_alignment", "streams/public_messages.npz", "messages do not bind current public waypoint")
        if state is not None and "effective_time_ns" in state and state["effective_time_ns"].ndim == 1:
            indices = np.searchsorted(state["effective_time_ns"], timestamps)
            aligned = indices.shape == timestamps.shape and np.all(indices < len(state["effective_time_ns"]))
            if aligned:
                aligned = np.array_equal(state["effective_time_ns"][indices], timestamps)
            if not aligned:
                _issue(issues, "message_state_alignment", "streams/public_messages.npz", "message times are absent from physical state stream")
            else:
                for message_key, state_key in (
                    ("message_position_w_m", "root_pos_w_m"),
                    ("message_velocity_w_mps", "root_lin_vel_w_mps"),
                ):
                    if not np.allclose(messages[message_key], state[state_key][indices], rtol=0.0, atol=1.0e-6):
                        _issue(issues, "message_state_alignment", f"messages.{message_key}", "broadcast payload differs from same-time physical state")
        checks["public_message_samples"] = sample_count

    if collection_binding is not None and condition_request_ok and isinstance(condition_request, Mapping):
        realization = evaluate_condition_realization(
            condition_request,
            receipt=receipt,
            scene=scene,
            public_task=public_task,
            state=state,
            messages=messages,
            checks=checks,
        )
        checks["condition_realization_schema"] = CONDITION_REALIZATION_SCHEMA
        checks["condition_realization"] = realization
        checks["condition_realization_verified"] = realization["status"] == "passed"
        for condition_issue in realization["issues"]:
            _issue(issues, condition_issue["code"], condition_issue["path"], condition_issue["message"])
    else:
        checks["condition_realization_verified"] = collection_binding is None

    camera = payloads["camera"]
    if _exact_arrays(camera, {"timestamps_ns", "camera_expected_pos_w_m", "camera_expected_quat_wxyz", "camera_observed_pos_w_m", "camera_observed_quat_wxyz", "camera_position_error_m", "camera_orientation_error_rad", "camera_usd_position_error_m", "camera_usd_forward_alignment_cosine", "camera_usd_orientation_error_rad", "camera_fabric_observed_pos_w_m", "camera_fabric_observed_quat_wxyz", "camera_fabric_position_error_m", "camera_fabric_orientation_error_rad", "camera_render_read_pre_frame_index", "camera_render_read_post_frame_index"}, "sensors/camera_poses.npz", issues) and timestamps is not None:
        assert camera is not None
        fabric_diagnostic_keys = {
            "camera_fabric_observed_pos_w_m",
            "camera_fabric_observed_quat_wxyz",
            "camera_fabric_position_error_m",
            "camera_fabric_orientation_error_rad",
        }
        for key, value in camera.items():
            if key != "timestamps_ns" and key not in fabric_diagnostic_keys:
                _finite(f"camera_poses.{key}", value, issues)
        position_error = camera["camera_position_error_m"]
        orientation_error = camera["camera_orientation_error_rad"]
        if position_error.shape != (len(timestamps), AGENT_COUNT) or float(np.max(position_error)) > pose_threshold_m:
            _issue(issues, "pose_closure", "sensors/camera_poses.npz", "camera position closure exceeds threshold")
        if orientation_error.shape != (len(timestamps), AGENT_COUNT) or float(np.max(orientation_error)) > orientation_threshold_rad:
            _issue(issues, "orientation_closure", "sensors/camera_poses.npz", "camera orientation closure exceeds threshold")
        checks["pose_closure_max_error_m"] = float(np.max(position_error))
        checks["orientation_closure_max_error_rad"] = float(np.max(orientation_error))
        usd_position_error = camera["camera_usd_position_error_m"]
        usd_forward_alignment = camera["camera_usd_forward_alignment_cosine"]
        usd_orientation_error = camera["camera_usd_orientation_error_rad"]
        if (
            usd_position_error.shape != (len(timestamps), AGENT_COUNT)
            or float(np.max(usd_position_error)) > ONBOARD_CAMERA_USD_POSITION_TOLERANCE_M
        ):
            _issue(
                issues,
                "camera_usd_pose_closure",
                "sensors/camera_poses.npz",
                "render-facing USD camera position closure exceeds threshold",
            )
        if (
            usd_forward_alignment.shape != (len(timestamps), AGENT_COUNT)
            or float(np.min(usd_forward_alignment)) < ONBOARD_CAMERA_USD_FORWARD_COSINE_MIN
        ):
            _issue(
                issues,
                "camera_usd_orientation_closure",
                "sensors/camera_poses.npz",
                "render-facing USD camera optical-axis closure exceeds threshold",
            )
        if (
            usd_orientation_error.shape != (len(timestamps), AGENT_COUNT)
            or float(np.max(usd_orientation_error)) > ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD
        ):
            _issue(
                issues,
                "camera_usd_full_orientation_closure",
                "sensors/camera_poses.npz",
                "render-facing USD camera full-orientation closure exceeds threshold",
            )
        checks["camera_usd_pose_closure_max_error_m"] = float(np.max(usd_position_error))
        checks["camera_usd_forward_alignment_min_cosine"] = float(np.min(usd_forward_alignment))
        checks["camera_usd_orientation_closure_max_error_rad"] = float(np.max(usd_orientation_error))
        fabric_position_error = camera["camera_fabric_position_error_m"]
        fabric_orientation_error = camera["camera_fabric_orientation_error_rad"]
        fabric_pose_shape = (len(timestamps), AGENT_COUNT)
        if (
            camera["camera_fabric_observed_pos_w_m"].shape
            != (*fabric_pose_shape, 3)
            or camera["camera_fabric_observed_quat_wxyz"].shape
            != (*fabric_pose_shape, 4)
            or fabric_position_error.shape != fabric_pose_shape
            or fabric_orientation_error.shape != fabric_pose_shape
        ):
            _issue(
                issues,
                "camera_fabric_diagnostic_shape",
                "sensors/camera_poses.npz",
                "non-authoritative Camera Fabric diagnostics have an invalid shape",
            )
        checks["camera_fabric_pose_authority"] = "diagnostic_only"
        fabric_position_max = float(np.max(fabric_position_error))
        fabric_orientation_max = float(np.max(fabric_orientation_error))
        checks["camera_fabric_pose_finite"] = bool(
            all(np.isfinite(camera[key]).all() for key in fabric_diagnostic_keys)
        )
        checks["camera_fabric_pose_max_error_m"] = (
            fabric_position_max if math.isfinite(fabric_position_max) else None
        )
        checks["camera_fabric_orientation_max_error_rad"] = (
            fabric_orientation_max if math.isfinite(fabric_orientation_max) else None
        )
        frame_before = camera["camera_render_read_pre_frame_index"]
        frame_after = camera["camera_render_read_post_frame_index"]
        frame_shape = (len(timestamps), AGENT_COUNT)
        frame_fence_ok = bool(
            frame_before.dtype == np.int64
            and frame_after.dtype == np.int64
            and frame_before.shape == frame_shape
            and frame_after.shape == frame_shape
            and np.array_equal(frame_after - frame_before, np.ones(frame_shape, dtype=np.int64))
        )
        if not frame_fence_ok:
            _issue(
                issues,
                "camera_render_read_fence",
                "sensors/camera_poses.npz",
                "each retained camera sample must bind exactly one post-render Camera buffer update",
            )
        checks["camera_render_read_fence_verified"] = frame_fence_ok

    camera_depth_valid = False
    lidar_ranges_valid = False
    camera_data = frame_payloads["camera_data"]
    camera_rgb_shape: tuple[int, ...] | None = None
    if _exact_frame_fields(camera_data, {"timestamps_ns", "rgb", "distance_to_image_plane_m"}, "sensors/onboard_rgbd.npz", issues) and timestamps is not None:
        assert camera_data is not None
        rgb_dtype, rgb_shape = camera_data.descriptor("rgb")
        depth_dtype, depth_shape = camera_data.descriptor("distance_to_image_plane_m")
        if rgb_dtype != np.uint8 or len(rgb_shape) != 5 or rgb_shape[:2] != (len(timestamps), AGENT_COUNT) or rgb_shape[-1] != 3:
            _issue(issues, "rgb_shape", "sensors/onboard_rgbd.npz", "RGB must be uint8 [T,8,H,W,3]")
        if depth_dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or depth_shape != (*rgb_shape[:-1], 1):
            _issue(issues, "depth_shape", "sensors/onboard_rgbd.npz", "depth must be float [T,8,H,W,1]")
        elif not _frame_field_is_finite(camera_data, "distance_to_image_plane_m"):
            _issue(issues, "nonfinite", "camera.depth", "numeric array must contain only finite values")
        camera_rgb_shape = rgb_shape
        camera_depth_valid = bool(
            depth_dtype in (np.dtype(np.float32), np.dtype(np.float64))
            and depth_shape == (*rgb_shape[:-1], 1)
            and _frame_field_is_finite(camera_data, "distance_to_image_plane_m")
        )
        rgb_min = min(int(np.min(camera_data.frame("rgb", frame_id))) for frame_id in range(len(timestamps)))
        rgb_max = max(int(np.max(camera_data.frame("rgb", frame_id))) for frame_id in range(len(timestamps)))
        if rgb_min == rgb_max:
            _issue(issues, "blank_rgb", "sensors/onboard_rgbd.npz", "RGB capture is constant")

    semantic = frame_payloads["semantic"]
    if _exact_frame_fields(semantic, {"timestamps_ns", "semantic_segmentation"}, "learning_labels/semantic_segmentation.npz", issues) and timestamps is not None:
        assert semantic is not None
        labels_dtype, labels_shape = semantic.descriptor("semantic_segmentation")
        if camera_rgb_shape is None:
            _issue(issues, "semantic_shape", "learning_labels/semantic_segmentation.npz", "RGB reference is unavailable")
        else:
            if labels_shape != (*camera_rgb_shape[:-1], 1) or not np.issubdtype(labels_dtype, np.integer):
                _issue(issues, "semantic_shape", "learning_labels/semantic_segmentation.npz", "semantic labels must be integer [T,8,H,W,1]")
    private_target_ids: tuple[str, ...] = ()
    if isinstance(private_evaluator_payload, Mapping):
        raw_targets = private_evaluator_payload.get("targets")
        if isinstance(raw_targets, list):
            private_target_ids = tuple(
                target.get("target_id")
                for target in raw_targets
                if isinstance(target, Mapping)
                and isinstance(target.get("target_id"), str)
            )
    semantic_metadata = _read_json(root / "learning_labels/semantic_metadata.json", issues)
    semantic_frame_metadata: list[Mapping[str, Any]] | None = None
    if semantic_metadata is not None:
        expected_metadata = {
            "schema": SEMANTIC_METADATA_SCHEMA,
            "partition": "learning_labels",
            "policy_visible": False,
            "frame_metadata": {
                "schema": SEMANTIC_FRAME_METADATA_SCHEMA,
                "path": SEMANTIC_FRAME_METADATA_RELATIVE_PATH,
                "frame_count": len(timestamps) if timestamps is not None else None,
                "onboard_camera_count": AGENT_COUNT,
                "overview_camera_count": 1,
                "record_fields": [
                    "schema",
                    "frame_index",
                    "timestamp_ns",
                    "onboard_replicator_info",
                    "overview_replicator_info",
                ],
            },
        }
        if dict(semantic_metadata) != expected_metadata:
            _issue(
                issues,
                "semantic_metadata",
                "learning_labels/semantic_metadata.json",
                "semantic metadata must declare the exact frame-aligned mapping contract",
            )
        if timestamps is not None:
            semantic_frame_metadata = _read_semantic_frame_metadata(
                root / SEMANTIC_FRAME_METADATA_RELATIVE_PATH,
                expected_timestamps=timestamps,
                private_target_ids=private_target_ids,
                issues=issues,
            )
    if _private_target_id_leaks(semantic_metadata, private_target_ids):
        _issue(
            issues,
            "semantic_private_id_leakage",
            "learning_labels/semantic_metadata.json",
            "public semantic metadata contains an evaluator-private target identifier",
        )
    target_observability = _recompute_target_observability(
        semantic,
        semantic_frame_metadata,
        target_count=TARGET_COUNT,
    )
    checks["target_observability"] = target_observability
    checks["target_observability_verified"] = bool(target_observability["passed"])
    if not target_observability["passed"]:
        _issue(
            issues,
            "target_observability",
            "learning_labels/semantic_segmentation.npz",
            "anonymous target slots do not meet the native semantic observability contract",
        )
    if isinstance(outcome, Mapping):
        declared_observability = outcome.get("target_observability")
        if declared_observability != target_observability:
            _issue(
                issues,
                "target_observability_binding",
                "task_outcome.json.target_observability",
                "T1 outcome does not match independently recomputed native semantic observability",
            )

    lidar = payloads["lidar"]
    if _exact_arrays(lidar, {"timestamps_ns", "pos_w_m", "quat_wxyz", "ranges_m"}, "sensors/lidar.npz", issues) and timestamps is not None:
        assert lidar is not None
        ranges = lidar["ranges_m"]
        _finite("lidar.ranges_m", ranges, issues)
        if ranges.ndim != 3 or ranges.shape[:2] != (len(timestamps), AGENT_COUNT) or ranges.shape[2] < 32:
            _issue(issues, "lidar_shape", "sensors/lidar.npz", "LiDAR must be [T,8,R] with R >= 32")
        elif lidar_max_distance_m is None:
            _issue(issues, "lidar_range", "sensors/lidar.npz", "LiDAR range cannot be checked without calibration")
        else:
            range_tolerance = max(1.0e-3, lidar_max_distance_m * 1.0e-5)
            if (
                np.any(ranges < 0.0)
                or np.any(ranges > lidar_max_distance_m + range_tolerance)
                or not np.any(ranges < lidar_max_distance_m - range_tolerance)
            ):
                _issue(
                    issues,
                    "lidar_range",
                    "sensors/lidar.npz",
                    "LiDAR ranges exceed the captured max distance or contain no geometry hit",
                )
            else:
                lidar_ranges_valid = bool(np.isfinite(ranges).all())

    if (
        camera_depth_valid
        and lidar_ranges_valid
        and camera_data is not None
        and lidar is not None
        and timestamps is not None
        and lidar_max_distance_m is not None
    ):
        visual_intrusion_evidence = [
            _onboard_visual_intrusion_evidence(
                camera_data.frame("distance_to_image_plane_m", frame_id),
                lidar["ranges_m"][frame_id],
                lidar_max_distance_m=lidar_max_distance_m,
            )
            for frame_id in range(len(timestamps))
        ]
        visual_intrusion_failures = [
            frame_id
            for frame_id, evidence in enumerate(visual_intrusion_evidence)
            if evidence.get("passed") is not True
        ]
        if visual_intrusion_failures:
            _issue(
                issues,
                "visual_intrusion",
                "sensors/onboard_rgbd.npz",
                "raw RGB-D/LiDAR frames contain near-geometry intrusion: "
                f"{visual_intrusion_failures[:8]}",
            )
        if visual_intrusion_gate_declaration is not None:
            declared_frames = visual_intrusion_gate_declaration.get("capture_frames")
            declared_count = visual_intrusion_gate_declaration.get("capture_frame_count")
            declared_valid = bool(
                isinstance(declared_frames, list)
                and declared_count == len(timestamps)
                and len(declared_frames) == len(timestamps)
                and all(
                    isinstance(item, Mapping)
                    and item.get("schema") == VISUAL_INTRUSION_GATE_SCHEMA
                    and item.get("passed") is True
                    and item.get("failures") == []
                    and isinstance(item.get("per_agent"), list)
                    and len(item["per_agent"]) == AGENT_COUNT
                    for item in declared_frames
                )
            )
            if not declared_valid:
                _issue(
                    issues,
                    "visual_intrusion_calibration",
                    "calibration.json.onboard_camera.visual_intrusion_gate.capture_frames",
                    "visual intrusion declarations do not bind every raw sensor frame",
                )
        checks["visual_intrusion_evidence"] = visual_intrusion_evidence
        checks["visual_intrusion_verified"] = not visual_intrusion_failures
        onboard_scene_content_evidence: list[dict[str, Any]] = []
        onboard_scene_content_failures: list[int] = []
        if camera_data is not None and semantic is not None:
            onboard_scene_content_evidence = [
                _onboard_scene_content_evidence(
                    camera_data.frame("distance_to_image_plane_m", frame_id),
                    semantic.frame("semantic_segmentation", frame_id),
                    (
                        semantic_frame_metadata[frame_id].get(
                            "onboard_replicator_info", {}
                        )
                        if semantic_frame_metadata is not None
                        else {}
                    ),
                    far_clip_m=ONBOARD_CAMERA_CLIPPING_RANGE_M[1],
                )
                for frame_id in range(len(timestamps))
            ]
            onboard_scene_content_failures = [
                frame_id
                for frame_id, evidence in enumerate(onboard_scene_content_evidence)
                if evidence.get("passed") is not True
            ]
            if onboard_scene_content_failures:
                _issue(
                    issues,
                    "onboard_scene_content",
                    "sensors/onboard_rgbd.npz",
                    "raw onboard frames are dominated by far-clip/background content: "
                    f"{onboard_scene_content_failures[:8]}",
                )
            if onboard_content_gate_declaration is not None:
                declared_frames = onboard_content_gate_declaration.get("capture_frames")
                declared_count = onboard_content_gate_declaration.get("capture_frame_count")
                declared_valid = bool(
                    isinstance(declared_frames, list)
                    and declared_count == len(timestamps)
                    and len(declared_frames) == len(timestamps)
                    and all(
                        isinstance(item, Mapping)
                        and item.get("schema") == ONBOARD_CONTENT_GATE_SCHEMA
                        and item.get("passed") is True
                        and item.get("failures") == []
                        and isinstance(item.get("per_agent"), list)
                        and len(item["per_agent"]) == AGENT_COUNT
                        for item in declared_frames
                    )
                )
                if not declared_valid:
                    _issue(
                        issues,
                        "onboard_content_calibration",
                        "calibration.json.onboard_camera.content_gate.capture_frames",
                        "onboard scene-content declarations do not bind every raw sensor frame",
                    )
            checks["onboard_scene_content_evidence"] = onboard_scene_content_evidence
            checks["onboard_scene_content_verified"] = not onboard_scene_content_failures

    imu = payloads["imu"]
    if _exact_arrays(imu, {"timestamps_ns", "pos_w_m", "quat_wxyz", "linear_acceleration_b_mps2", "angular_velocity_b_radps"}, "sensors/imu.npz", issues) and timestamps is not None:
        assert imu is not None
        for key in ("pos_w_m", "linear_acceleration_b_mps2", "angular_velocity_b_radps"):
            _finite(f"imu.{key}", imu[key], issues)
            if imu[key].shape != (len(timestamps), AGENT_COUNT, 3):
                _issue(issues, "imu_shape", f"imu.{key}", "expected [T,8,3]")

    captured_contact_force_max_n: float | None = None
    contact = payloads["contact"]
    if _exact_arrays(contact, {"timestamps_ns", "net_forces_w_n"}, "sensors/contact.npz", issues) and timestamps is not None:
        assert contact is not None
        force = contact["net_forces_w_n"]
        _finite("contact.net_forces_w_n", force, issues)
        if (
            not np.issubdtype(force.dtype, np.floating)
            or force.shape != (len(timestamps), AGENT_COUNT, 1, 3)
        ):
            _issue(issues, "contact_shape", "sensors/contact.npz", "contact normal forces must be float [T,8,1,3]")
        else:
            contact_free = bool(np.all(force == 0.0))
            checks["contact_free"] = contact_free
            captured_contact_force_max_n = float(np.max(np.linalg.norm(force, axis=-1)))
            checks["contact_net_normal_force_max_n"] = captured_contact_force_max_n
            checks["contact_net_normal_force_below_abort_threshold"] = (
                captured_contact_force_max_n < CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N
            )
            if captured_contact_force_max_n >= CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N:
                _issue(
                    issues,
                    "contact_abort_threshold",
                    "sensors/contact.npz",
                    "captured root-body net normal force reaches the runtime abort threshold",
                )

    physics_steps_for_guard: int | None = None
    if state is not None and "root_pos_w_m" in state and state["root_pos_w_m"].ndim >= 1:
        physics_steps_for_guard = int(state["root_pos_w_m"].shape[0])
    runtime_trace_frame_count, runtime_trace_max_contact_force_n = _validate_runtime_safety_trace(
        payloads["runtime_safety"],
        receipt,
        structural_aabbs,
        state=state,
        captured_contact=contact,
        sensor_timestamps=timestamps,
        issues=issues,
        checks=checks,
    )
    _validate_runtime_safety_guard(
        receipt,
        scene,
        structural_aabbs,
        physics_steps=physics_steps_for_guard,
        captured_contact_force_max_n=captured_contact_force_max_n,
        runtime_trace_frame_count=runtime_trace_frame_count,
        runtime_trace_max_contact_force_n=runtime_trace_max_contact_force_n,
        root=root,
        issues=issues,
        checks=checks,
    )
    _validate_sensor_phase_trace(
        payloads["sensor_phase"],
        receipt,
        state=state,
        contact=contact,
        timestamps=timestamps,
        root=root,
        issues=issues,
        checks=checks,
    )

    overview = frame_payloads["overview"]
    overview_metadata_by_timestamp = {
        int(row["timestamp_ns"]): row.get("overview_replicator_info", {})
        for row in semantic_frame_metadata or ()
        if isinstance(row.get("timestamp_ns"), int)
    }
    low_rate_overview = overview_archive_declaration is not None
    overview_expected_fields = {
        "timestamps_ns",
        "rgb",
        "semantic_segmentation",
        "camera_pos_w_m",
        "camera_quat_wxyz",
        "target_w_m",
    }
    if not low_rate_overview:
        # Preserve validation of already-recorded full-rate development
        # evidence.  New captures must use the explicit low-rate declaration
        # above and cannot silently omit their archive contract.
        overview_expected_fields.add("distance_to_image_plane_m")
    if _exact_frame_fields(
        overview,
        overview_expected_fields,
        "sensors/overview_rgb.npz",
        issues,
    ) and timestamps is not None:
        assert overview is not None
        overview_timestamps = overview.timestamps_ns
        if (
            overview_timestamps.dtype != np.int64
            or overview_timestamps.ndim != 1
            or len(overview_timestamps) < 2
            or not np.all(np.diff(overview_timestamps) > 0)
        ):
            _issue(
                issues,
                "overview_timestamp_schedule",
                "sensors/overview_rgb.npz.timestamps_ns",
                "overview timestamps must be a strictly increasing int64 sequence with first and final frames",
            )
        elif low_rate_overview:
            assert overview_archive_declaration is not None
            expected_indices = _overview_archive_frame_indices(len(timestamps))
            expected_timestamps = timestamps[np.asarray(expected_indices)]
            declared_indices = overview_archive_declaration["source_frame_indices"]
            if (
                overview_archive_declaration["source_frame_count"] != len(timestamps)
                or declared_indices != list(expected_indices)
                or not np.array_equal(overview_timestamps, expected_timestamps)
            ):
                _issue(
                    issues,
                    "overview_timestamp_schedule",
                    "sensors/overview_rgb.npz.timestamps_ns",
                    "overview frames must be the exact first/stride/final subset of retained sensor timestamps",
                )
            checks["overview_archive_frame_count"] = int(len(overview_timestamps))
            checks["overview_archive_source_frame_count"] = int(len(timestamps))
            checks["overview_archive_frame_indices"] = list(expected_indices)
        elif not np.array_equal(overview_timestamps, timestamps):
            _issue(
                issues,
                "timestamp_alignment",
                "sensors/overview_rgb.npz",
                "legacy overview timestamps must match every retained sensor timestamp",
            )
        rgb_dtype, rgb_shape = overview.descriptor("rgb")
        if low_rate_overview:
            depth_dtype: np.dtype[Any] | None = None
            depth_shape: tuple[int, ...] | None = None
        else:
            depth_dtype, depth_shape = overview.descriptor("distance_to_image_plane_m")
        semantic_dtype, semantic_shape = overview.descriptor("semantic_segmentation")
        camera_pos_w_m = overview.array("camera_pos_w_m")
        camera_quat_wxyz = overview.array("camera_quat_wxyz")
        target_w_m = overview.array("target_w_m")
        rgb_shape_valid = bool(
            rgb_dtype == np.uint8
            and len(rgb_shape) == 4
            and rgb_shape[0] == len(overview_timestamps)
            and rgb_shape[1] >= 2
            and rgb_shape[2] >= 2
            and rgb_shape[-1] == 3
        )
        if not rgb_shape_valid:
            _issue(
                issues,
                "overview_shape",
                "sensors/overview_rgb.npz",
                "overview RGB must be uint8 [T,H,W,3] with H,W >= 2",
            )
        depth_shape_valid = bool(
            not low_rate_overview
            and rgb_shape_valid
            and depth_dtype is not None
            and depth_shape is not None
            and np.issubdtype(depth_dtype, np.floating)
            and depth_shape == (*rgb_shape[:-1], 1)
        )
        if not low_rate_overview and not depth_shape_valid:
            _issue(
                issues,
                "overview_depth_shape",
                "sensors/overview_rgb.npz",
                "overview depth must be floating [T,H,W,1] aligned to RGB",
            )
        elif not low_rate_overview:
            if not _frame_field_is_finite(overview, "distance_to_image_plane_m"):
                _issue(issues, "nonfinite", "overview.distance_to_image_plane_m", "numeric array must contain only finite values")
        semantic_shape_valid = bool(
            rgb_shape_valid
            and np.issubdtype(semantic_dtype, np.integer)
            and semantic_shape == (*rgb_shape[:-1], 1)
        )
        if not semantic_shape_valid:
            _issue(
                issues,
                "overview_semantic_shape",
                "sensors/overview_rgb.npz",
                "overview semantic labels must be integer [T,H,W,1] aligned to RGB",
            )
        route_witness_pose_valid = bool(
            np.issubdtype(camera_pos_w_m.dtype, np.floating)
            and np.issubdtype(camera_quat_wxyz.dtype, np.floating)
            and np.issubdtype(target_w_m.dtype, np.floating)
            and camera_pos_w_m.shape == (len(overview_timestamps), 3)
            and camera_quat_wxyz.shape == (len(overview_timestamps), 4)
            and target_w_m.shape == (len(overview_timestamps), 3)
            and np.isfinite(camera_pos_w_m).all()
            and np.isfinite(camera_quat_wxyz).all()
            and np.isfinite(target_w_m).all()
        )
        if not route_witness_pose_valid:
            _issue(
                issues,
                "route_witness_camera_pose",
                "sensors/overview_rgb.npz",
                "route-witness pose, quaternion, and target must be finite [T,3]/[T,4]/[T,3] arrays",
            )
        elif route_witness_declaration is not None:
            expected_views = [
                _public_route_witness_view_at_time_ns(int(timestamp))
                for timestamp in overview_timestamps
            ]
            expected_pos = np.asarray(
                [view["eye_w_m"] for view in expected_views], dtype=np.float64
            )
            expected_target = np.asarray(
                [view["target_w_m"] for view in expected_views], dtype=np.float64
            )
            expected_quat = np.asarray(
                [view["orientation_wxyz"] for view in expected_views], dtype=np.float64
            )
            position_error = np.linalg.norm(camera_pos_w_m - expected_pos, axis=-1)
            target_error = np.linalg.norm(target_w_m - expected_target, axis=-1)
            observed_norm = np.linalg.norm(camera_quat_wxyz, axis=-1)
            normalized_quat = camera_quat_wxyz / observed_norm[:, None]
            alignment = np.abs(np.sum(normalized_quat * expected_quat, axis=-1))
            min_alignment = float(np.min(alignment))
            checks["route_witness_camera_pose_closure_max_error_m"] = float(
                np.max(position_error)
            )
            checks["route_witness_camera_target_closure_max_error_m"] = float(
                np.max(target_error)
            )
            checks["route_witness_camera_orientation_min_abs_dot"] = min_alignment
            if (
                not np.isfinite(observed_norm).all()
                or np.any(observed_norm <= 1.0e-8)
                or float(np.max(position_error)) > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
                or float(np.max(target_error)) > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
            ):
                _issue(
                    issues,
                    "route_witness_camera_pose",
                    "sensors/overview_rgb.npz",
                    "route-witness camera position/target does not match the public world-frame contract",
                )
            if min_alignment < math.cos(0.01 / 2.0):
                _issue(
                    issues,
                    "route_witness_camera_orientation",
                    "sensors/overview_rgb.npz.camera_quat_wxyz",
                    "route-witness camera orientation does not match the public look-at target",
                )
            if state is None or "effective_time_ns" not in state:
                _issue(
                    issues,
                    "route_witness_state_alignment",
                    "sensors/overview_rgb.npz",
                    "route witness cannot be bound without the physical state stream",
                )
            else:
                indices = np.searchsorted(state["effective_time_ns"], overview_timestamps)
                aligned = (
                    indices.shape == overview_timestamps.shape
                    and np.all(indices < len(state["effective_time_ns"]))
                    and np.array_equal(state["effective_time_ns"][indices], overview_timestamps)
                )
                if not aligned:
                    _issue(
                        issues,
                        "route_witness_state_alignment",
                        "sensors/overview_rgb.npz.timestamps_ns",
                        "route-witness frames must align to physical effective-time state samples",
                    )
                else:
                    tracked_positions = state["root_pos_w_m"][
                        indices, OVERVIEW_WITNESS_TRACKED_AGENT_ID
                    ]
                    tracked_displacement_m = float(
                        np.max(
                            np.linalg.norm(
                                tracked_positions - tracked_positions[0:1], axis=-1
                            )
                        )
                    )
                    checks["route_witness_tracked_agent_max_displacement_m"] = (
                        tracked_displacement_m
                    )
                    if tracked_displacement_m < OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M:
                        _issue(
                            issues,
                            "route_witness_tracked_agent_motion",
                            "streams/state_action.npz.root_pos_w_m",
                            "tracked CF2X did not move enough for the route-witness demonstration",
                        )
        if rgb_shape_valid and int(overview.frame("rgb", 0).max()) == int(overview.frame("rgb", 0).min()):
            _issue(
                issues,
                "blank_overview",
                "sensors/overview_rgb.npz",
                "overview first frame is constant",
            )
        if rgb_shape_valid and semantic_shape_valid:
            if low_rate_overview:
                overview_evidence = [
                    _overview_archived_visual_evidence(
                        overview.frame("rgb", frame_id),
                        overview.frame("semantic_segmentation", frame_id),
                        overview_metadata_by_timestamp.get(
                            int(overview_timestamps[frame_id]), {}
                        ),
                    )
                    for frame_id in range(len(overview_timestamps))
                ]
                city_failures = [
                    frame_id
                    for frame_id, evidence in enumerate(overview_evidence)
                    if not evidence["rgb_evidence_passed"]
                ]
            elif depth_shape_valid:
                overview_evidence = [
                    _overview_city_content_evidence(
                        overview.frame("rgb", frame_id),
                        overview.frame("distance_to_image_plane_m", frame_id),
                        overview.frame("semantic_segmentation", frame_id),
                        overview_metadata_by_timestamp.get(
                            int(overview_timestamps[frame_id]), {}
                        ),
                        far_clip_m=overview_far_clip_m,
                    )
                    for frame_id in range(len(overview_timestamps))
                ]
                city_failures = [
                    frame_id
                    for frame_id, evidence in enumerate(overview_evidence)
                    if not evidence["city_evidence_passed"]
                ]
            else:
                overview_evidence = []
                city_failures = []
            structural_failures = [
                frame_id
                for frame_id, evidence in enumerate(overview_evidence)
                if not evidence["structural_evidence_passed"]
            ]
            if city_failures:
                _issue(
                    issues,
                    "overview_archive_visual" if low_rate_overview else "overview_city_content",
                    "sensors/overview_rgb.npz",
                    (
                        "low-rate overview frames fail retained RGB/semantic evidence: "
                        if low_rate_overview
                        else "overview frames fail RGB/depth City-Lite evidence: "
                    )
                    + f"{city_failures[:8]}",
                )
            if structural_failures:
                _issue(
                    issues,
                    "overview_structural_semantics",
                    "sensors/overview_rgb.npz",
                    "overview frames lack pixels for structural IDs declared by Replicator: "
                    f"{structural_failures[:8]}",
                )
            route_witness_visibility = [
                _overview_tracked_agent_visibility_evidence(
                    overview.frame("semantic_segmentation", frame_id),
                    overview_metadata_by_timestamp.get(
                        int(overview_timestamps[frame_id]), {}
                    ),
                )
                for frame_id in range(len(overview_timestamps))
            ]
            route_witness_visibility_failures = [
                frame_id
                for frame_id, evidence in enumerate(route_witness_visibility)
                if not evidence["passed"]
            ]
            if route_witness_visibility_failures:
                _issue(
                    issues,
                    "route_witness_tracked_agent_visibility",
                    "sensors/overview_rgb.npz.semantic_segmentation",
                    "route-witness frames do not visibly contain the tracked CF2X identity marker: "
                    f"{route_witness_visibility_failures[:8]}",
                )
            if route_witness_visibility_declaration is not None:
                declared_frames = route_witness_visibility_declaration.get("capture_frames")
                declared_count = route_witness_visibility_declaration.get("capture_frame_count")
                declared_frames_valid = bool(
                    isinstance(declared_frames, list)
                    and declared_count == len(timestamps)
                    and len(declared_frames) == len(timestamps)
                    and all(
                        isinstance(item, Mapping)
                        and item.get("schema")
                        == "org.rivermark.isaac-route-witness-agent-visibility.v1"
                        and item.get("tracked_agent_id") == OVERVIEW_WITNESS_TRACKED_AGENT_ID
                        and item.get("passed") is True
                        and item.get("failures") == []
                        and isinstance(
                            item.get("tracked_agent_pixel_count"), (int, np.integer)
                        )
                        and not isinstance(
                            item.get("tracked_agent_pixel_count"), (bool, np.bool_)
                        )
                        and int(item["tracked_agent_pixel_count"])
                        >= OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS
                        and item.get("effective_time_ns") == int(timestamp)
                        and item.get("witness_shot_index")
                        == _public_route_witness_view_at_time_ns(int(timestamp))["shot_index"]
                        for item, timestamp in zip(declared_frames, timestamps)
                    )
                )
                if not declared_frames_valid:
                    _issue(
                        issues,
                        "route_witness_visibility_calibration",
                        "calibration.json.overview_camera.tracked_agent_visibility_gate.capture_frames",
                        "route-witness visibility declarations do not bind every raw overview frame",
                    )
            if overview_gate_declaration is not None:
                declared_frames = overview_gate_declaration.get("capture_frames")
                declared_count = overview_gate_declaration.get("capture_frame_count")
                declared_frames_valid = bool(
                    isinstance(declared_frames, list)
                    and declared_count == len(timestamps)
                    and len(declared_frames) == len(timestamps)
                    and all(
                        isinstance(item, Mapping)
                        and item.get("schema") == OVERVIEW_CONTENT_GATE_SCHEMA
                        and item.get("passed") is True
                        and item.get("failures") == []
                        and (
                            not low_rate_overview
                            or (
                                item.get("effective_time_ns") == int(timestamp)
                                and item.get("witness_shot_index")
                                == _public_route_witness_view_at_time_ns(
                                    int(timestamp)
                                )["shot_index"]
                                and item.get("city_evidence_passed") is True
                                and item.get("structural_evidence_passed") is True
                            )
                        )
                        for item, timestamp in zip(declared_frames, timestamps)
                    )
                )
                if not declared_frames_valid:
                    _issue(
                        issues,
                        "overview_content_calibration",
                        "calibration.json.overview_camera.content_gate.capture_frames",
                        "capture-frame content-gate declarations do not bind every raw overview frame",
                    )
            else:
                declared_frames_valid = False
            checks["overview_city_content_evidence"] = overview_evidence
            checks["overview_city_content_verified"] = (
                declared_frames_valid if low_rate_overview else not city_failures
            )
            checks["overview_live_depth_gate_persisted"] = not low_rate_overview
            if low_rate_overview:
                checks["overview_live_depth_gate_verified"] = declared_frames_valid
                checks["overview_archive_visual_evidence"] = overview_evidence
                checks["overview_archive_visual_verified"] = not city_failures
            checks["overview_structural_semantics_verified"] = not structural_failures
            checks["overview_structural_semantics_required"] = any(
                bool(evidence["structural_semantic_ids"])
                for evidence in overview_evidence
            )
            checks["route_witness_visibility_evidence"] = route_witness_visibility
            checks["route_witness_visibility_verified"] = not route_witness_visibility_failures
    checks["sensor_samples"] = int(len(timestamps)) if timestamps is not None else 0
    checks["agent_count"] = AGENT_COUNT
    checks["radar_captured"] = False
    checks["real_flight_captured"] = False
    checks["online_capture"] = capture_integrity.get("online_capture") is True
    checks["queue_overflow"] = capture_integrity.get("queue_overflow")
    checks["silent_frame_drop"] = capture_integrity.get("silent_frame_drop")
    checks["pose_closure_threshold_m"] = float(pose_threshold_m)
    timestamp_codes = {
        "timestamps",
        "timestamp_order",
        "timestamp_alignment",
        "overview_timestamp_schedule",
        "capture_timing",
        "runtime_safety_timing",
    }
    pose_codes = {
        "pose_closure",
        "orientation_closure",
        "camera_usd_pose_closure",
        "camera_usd_orientation_closure",
        "camera_usd_full_orientation_closure",
    }
    action_codes = {
        "action_time", "action_causality", "capture_timing", "zero_applied_thrust", "static_applied_thrust",
        "thrust_range", "static_high_level_action", "waypoint_static", "waypoint_binding",
        "message_task_alignment", "message_state_alignment",
    }
    sensor_codes = {
        "npz_decode", "npz_fields", "nonfinite", "rgb_shape", "depth_shape", "semantic_shape",
        "blank_rgb", "lidar_shape", "lidar_range", "lidar_calibration", "imu_shape", "contact_shape", "overview_shape",
        "overview_depth_shape", "overview_semantic_shape", "blank_overview", "semantic_metadata",
        "semantic_frame_metadata", "semantic_frame_alignment", "semantic_private_id_leakage",
        "overview_semantic_metadata", "overview_calibration", "overview_content_calibration",
        "overview_archive_calibration", "overview_archive_visual", "overview_city_content",
        "overview_structural_semantics",
        "visual_intrusion_calibration", "visual_intrusion",
        "route_witness_camera_calibration", "route_witness_camera_pose",
        "route_witness_camera_orientation", "route_witness_state_alignment",
        "route_witness_tracked_agent_motion", "route_witness_visibility_calibration",
        "route_witness_tracked_agent_visibility",
    }
    issue_codes = {issue.code for issue in issues}
    checks["timestamp_audit_passed"] = not bool(issue_codes & timestamp_codes)
    checks["pose_closure_audit_passed"] = not bool(issue_codes & pose_codes)
    checks["action_causality_audit_passed"] = not bool(issue_codes & action_codes)
    checks["sensor_decode_audit_passed"] = not bool(issue_codes & sensor_codes)
    checks["policy_leakage_audit_passed"] = not bool(
        issue_codes & {"policy_truth_leakage", "public_private_leakage"}
    )
    city_lite_codes = {
        "environment_id", "scene_authority", "authority_assets", "selective_references",
        "rivermark_layer_inventory",
        "unresolved_reference", "legacy_prims", "decorative_prims", "stage_units",
        "native_collision_counts", "flight_volume", "command_volume", "structural_aabbs",
        "structural_aabb", "structural_aabb_source", "collision_proxies",
        "route_clearance_contract", "route_command_volume", "route_start", "route_clearance",
        "route_contract", "trajectory_flight_volume", "trajectory_clearance", "contact_abort_threshold",
        "literal_fleet_spawn",
        "runtime_safety_guard", "runtime_safety_trace", "runtime_safety_timing", "runtime_safety_trace_binding",
        "capture_timing",
        "lidar_geometry_coverage", "evaluator_binding", "scene_claim_boundary",
        "evaluator_target_geometry", "semantic_private_id_leakage",
        "runtime_target_usd_closure",
        "target_observability", "target_observability_binding",
        "public_private_leakage",
    }
    checks["city_lite_scene_audit_passed"] = not bool(issue_codes & city_lite_codes)
    checks["evaluator_binding_verified"] = evaluator_verified and "evaluator_binding" not in issue_codes
    checks["route_geometry_audit_passed"] = not bool(
        issue_codes
        & {
            "public_routes",
            "route_clearance_contract",
            "route_command_volume",
            "route_start",
            "route_clearance",
            "route_contract",
        }
    )
    checks["t1_scoring_status"] = (
        outcome.get("scoring_status") if isinstance(outcome, Mapping) else None
    )
    for frame_payload in frame_payloads.values():
        if frame_payload is not None:
            frame_payload.close()
    return IsaacValidationReport(root, receipt_hash, checks, tuple(issues))


def _validator_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _native_t2_validator_sha256() -> str:
    """Hash the complete native-T2 validation implementation bundle.

    A native validation receipt is only meaningful when both the CLI dispatch
    and CPU replay implementation are fixed.  Keep the legacy single-file
    hash for normal T1 receipts so downstream formal-pack compatibility stays
    unchanged.
    """

    bundle = {
        "isaac_validate.py": _validator_sha256(),
        "native_t2_validate.py": sha256_file(
            Path(__file__).with_name("native_t2_validate.py")
        ),
    }
    encoded = json.dumps(
        bundle, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_validation_receipt(report: IsaacValidationReport, destination: Path) -> Path:
    if not report.valid or report.receipt_sha256 is None:
        raise RuntimeError("cannot write a passing validation receipt for an invalid capture")
    is_native_t2 = report.checks.get("validation_profile") == "native_t2_canary"
    payload = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed",
        "formal_benchmark_admission": False,
        "capture_receipt_sha256": report.receipt_sha256,
        "validator_id": (
            "rivermark-independent-native-t2-canary-validator-v1"
            if is_native_t2
            else "rivermark-independent-isaac-validator-v1"
        ),
        "validator_source_sha256": (
            _native_t2_validator_sha256() if is_native_t2 else _validator_sha256()
        ),
        "checks": dict(report.checks),
        "issues": [],
    }
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--evaluator-private-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        help="external v2 runtime lock; required when the capture receipt binds one",
    )
    parser.add_argument(
        "--cf2x-runtime-calibration",
        type=Path,
        help="external CF2X runtime calibration report; required for native T2 canaries",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = validate_isaac_capture(
        args.capture_root,
        evaluator_manifest=args.evaluator_private_manifest,
        require_clean_source=args.require_clean_source,
        runtime_lock_path=args.runtime_lock,
        cf2x_runtime_calibration=args.cf2x_runtime_calibration,
    )
    payload: dict[str, Any] = {
        "valid": report.valid,
        "capture_receipt_sha256": report.receipt_sha256,
        "checks": dict(report.checks),
        "issues": [asdict(issue) for issue in report.issues],
    }
    if report.valid and args.output is not None:
        payload["validation_receipt"] = str(write_validation_receipt(report, args.output))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
