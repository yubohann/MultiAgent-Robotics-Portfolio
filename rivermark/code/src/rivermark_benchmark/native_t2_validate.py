"""Independent CPU-only validation for a development native Isaac T2 canary.

This path intentionally does not reuse the capture's self-reported outcome.
It reopens the raw bounded artifacts, recreates public RGB-D semantic
candidates, verifies the command-to-actuator chain, and invokes the private
event evaluator only in memory.  It is a development canary validator, not a
formal-episode admission path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .cf2x_runtime_calibration import validate_calibration_report
from .citylite_task import validate_route_timing_feasibility
from .citylite_scene import AGENT_COUNT
from .collection_protocol import native_t2_v2_motion_contract, native_t2_v3_motion_contract
from .frame_archive import ChunkedFrameArchive, FrameArchiveError
from .isaac_capture import (
    CONTROL_MODE_NATIVE_T2_CANARY,
    NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH,
    NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
    NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
    NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
    NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
    NATIVE_T2_TASK_KIND,
    OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M,
    OVERVIEW_WITNESS_POSITION_TOLERANCE_M,
    PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS,
    SEMANTIC_FRAME_METADATA_RELATIVE_PATH,
    PrivateEvaluatorManifestError,
    _onboard_scene_content_evidence,
    _onboard_visual_intrusion_evidence,
    _overview_archive_frame_indices,
    _overview_tracked_agent_visibility_evidence,
    _public_route_witness_view_at_time_ns,
    bind_native_t2_calibration,
    validate_external_private_evaluator_manifest,
)
from .isaac_runtime_safety import physics_time_ns
from .isaac_transfer import (
    FixedDecisionCadence,
    WorldCommandBounds,
    derive_physical_state_8d,
)
from .native_t2_canary import (
    NATIVE_T2_EVENTS_SCHEMA,
    NATIVE_T2_TRACE_SCHEMA,
    PublicRouteCoveragePolicy,
    SpatialCandidateDeduplicator,
    native_rgbd_world_points,
    native_semantic_rgbd_candidates,
)
from .private_evaluator_manifest import (
    NATIVE_T2_TASK_VARIANT_ID,
    NATIVE_T2_V2_TASK_VARIANT_ID,
    NATIVE_T2_V3_TASK_VARIANT_ID,
)
from .runtime_lock import RuntimeLockError, load_runtime_lock, runtime_lock_sha256
from .schema import iter_tree, normalized_key
from .search_event_evaluator import PRIVATE_TASK_SCHEMA, evaluate_search_events
from .t2_policy_abi import (
    T2_NATIVE_STEP_EVIDENCE_SCHEMA,
    T2CandidateEventJournal,
    T2PublicFleetObservation,
    T2PublicSensorObservation,
)
from .video import sha256_file

NATIVE_T2_EXPECTED_ARTIFACTS = frozenset(
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
        "sensors/native_t2_camera_extrinsics.npz",
        "sensors/onboard_rgbd.npz",
        "sensors/overview_rgb.npz",
        "sensors/runtime_safety.npz",
        "sensors/sensor_phase.npz",
        NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
        NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
        "streams/state_action.npz",
    }
)
_CONTROL_ARTIFACTS = frozenset({"capture_start.json"})
_HEX = frozenset("0123456789abcdef")
_NATIVE_T2_PROTOCOL_TASK_VARIANTS = {
    "citylite-native-t2-canary-v1": NATIVE_T2_TASK_VARIANT_ID,
    "citylite-native-t2-canary-v2": NATIVE_T2_V2_TASK_VARIANT_ID,
    "citylite-native-t2-canary-v3": NATIVE_T2_V3_TASK_VARIANT_ID,
}


def _native_t2_task_variant_from_receipt(receipt: Mapping[str, Any]) -> str | None:
    binding = receipt.get("collection_binding")
    if not isinstance(binding, Mapping):
        return None
    return _NATIVE_T2_PROTOCOL_TASK_VARIANTS.get(binding.get("protocol_id"))


def _native_t2_motion_contract_from_receipt(
    receipt: Mapping[str, Any], *, task_variant_id: str | None
) -> Mapping[str, Any] | None:
    """Return a revision-bound public motion contract, never a default."""

    expected_contract_by_variant = {
        NATIVE_T2_V2_TASK_VARIANT_ID: native_t2_v2_motion_contract,
        NATIVE_T2_V3_TASK_VARIANT_ID: native_t2_v3_motion_contract,
    }
    expected_contract_factory = expected_contract_by_variant.get(task_variant_id)
    if expected_contract_factory is None:
        return None
    command = receipt.get("command")
    native = command.get("native_t2_canary") if isinstance(command, Mapping) else None
    motion = native.get("motion_contract") if isinstance(native, Mapping) else None
    required = {
        "schema",
        "waypoint_segment_seconds",
        "dt_s",
        "warmup_steps",
        "rollout_steps",
        "capture_stride",
        "decision_stride_physics_steps",
        "max_horizontal_speed_mps",
        "max_vertical_speed_mps",
        "max_yaw_rate_rad_s",
        "position_feedback_gain",
        "yaw_feedback_gain",
        "route_speed_utilization_limit",
        "camera_heading_model",
        "yaw_stability_error_rad",
        "yaw_settle_margin_s",
    }
    if not isinstance(motion, Mapping) or set(motion) != required:
        return None
    try:
        numeric = (
            "waypoint_segment_seconds",
            "dt_s",
            "max_horizontal_speed_mps",
            "max_vertical_speed_mps",
            "max_yaw_rate_rad_s",
            "position_feedback_gain",
            "yaw_feedback_gain",
            "route_speed_utilization_limit",
            "yaw_stability_error_rad",
            "yaw_settle_margin_s",
        )
        if motion["schema"] != "org.rivermark.native-t2-motion-contract.v1":
            return None
        if any(not math.isfinite(float(motion[key])) or float(motion[key]) <= 0.0 for key in numeric):
            return None
        if float(motion["route_speed_utilization_limit"]) > 1.0:
            return None
        if any(
            isinstance(motion[key], bool) or not isinstance(motion[key], int) or motion[key] < 1
            for key in ("warmup_steps", "rollout_steps", "capture_stride", "decision_stride_physics_steps")
        ):
            return None
        if motion["camera_heading_model"] != "segment_horizontal_heading_yaw_limited_v1":
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    # A receipt can commit to the frozen contract, but it cannot author a new
    # one. Otherwise a self-consistent artifact could lower bounds or alter
    # timing after capture.
    if dict(motion) != expected_contract_factory():
        return None
    return motion


def _validate_motion_command_binding(
    command: Mapping[str, Any],
    motion_contract: Mapping[str, Any],
    issues: list[NativeT2ValidationIssue],
) -> None:
    """Require receipt command fields to exactly realize its frozen revision."""

    for receipt_key, contract_key in {
        "steps": "rollout_steps",
        "warmup_steps": "warmup_steps",
        "capture_stride": "capture_stride",
    }.items():
        actual = command.get(receipt_key)
        expected = int(motion_contract[contract_key])
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            _issue(
                issues,
                "native_t2_motion_command",
                f"capture_receipt.json.command.{receipt_key}",
                "native T2 command timing does not match the frozen motion contract",
            )
    actual_dt_s = command.get("dt_s")
    try:
        dt_matches = not isinstance(actual_dt_s, bool) and math.isclose(
            float(actual_dt_s),
            float(motion_contract["dt_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    except (TypeError, ValueError, OverflowError):
        dt_matches = False
    if not dt_matches:
        _issue(
            issues,
            "native_t2_motion_command",
            "capture_receipt.json.command.dt_s",
            "native T2 command timing does not match the frozen motion contract",
        )

    native = command.get("native_t2_canary")
    expected_bounds = {
        "max_horizontal_speed_mps": float(motion_contract["max_horizontal_speed_mps"]),
        "max_vertical_speed_mps": float(motion_contract["max_vertical_speed_mps"]),
        "max_yaw_rate_rad_s": float(motion_contract["max_yaw_rate_rad_s"]),
    }
    if (
        not isinstance(native, Mapping)
        or native.get("decision_stride_physics_steps")
        != int(motion_contract["decision_stride_physics_steps"])
        or native.get("world_command_bounds") != expected_bounds
    ):
        _issue(
            issues,
            "native_t2_motion_command",
            "capture_receipt.json.command.native_t2_canary",
            "native T2 decision cadence or bounded action fields do not match the frozen motion contract",
        )


@dataclass(frozen=True)
class NativeT2ValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class NativeT2ValidationResult:
    checks: Mapping[str, Any]
    issues: tuple[NativeT2ValidationIssue, ...]


def _issue(issues: list[NativeT2ValidationIssue], code: str, path: str, message: str) -> None:
    issues.append(NativeT2ValidationIssue(code, path, message))


def _canonical_sha256(value: Any) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _read_json(path: Path, issues: list[NativeT2ValidationIssue]) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "invalid_json", path.as_posix(), str(exc))
        return None
    if not isinstance(payload, Mapping):
        _issue(issues, "json_type", path.as_posix(), "expected a JSON object")
        return None
    return payload


def _load_npz(path: Path, issues: list[NativeT2ValidationIssue]) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as source:
            return {name: source[name] for name in source.files}
    except (OSError, ValueError, EOFError) as exc:
        _issue(issues, "npz_decode", path.as_posix(), str(exc))
        return None


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _allclose(left: Any, right: Any, *, atol: float = 1.0e-5) -> bool:
    try:
        lhs = np.asarray(left, dtype=np.float64)
        rhs = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return lhs.shape == rhs.shape and np.isfinite(lhs).all() and np.isfinite(rhs).all() and np.allclose(
        lhs, rhs, rtol=0.0, atol=atol
    )


def _native_world_points(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    pos_w_m: np.ndarray,
    quat_w_ros: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper for the shared retained-frame geometry contract."""

    return native_rgbd_world_points(depth_m, intrinsics, pos_w_m, quat_w_ros)


def _semantic_slot_pixels(metadata: Mapping[str, Any], labels: np.ndarray, slot: str) -> tuple[int, ...]:
    per_camera = metadata.get("per_camera")
    if not isinstance(per_camera, list) or len(per_camera) != AGENT_COUNT:
        return ()
    output: list[int] = []
    for agent_id, mapping in enumerate(per_camera):
        ids: list[int] = []
        if isinstance(mapping, Mapping) and isinstance(mapping.get("id_to_labels"), Mapping):
            for raw_id, details in mapping["id_to_labels"].items():
                if isinstance(details, Mapping) and str(details.get("class", "")).casefold() == slot:
                    try:
                        ids.append(int(raw_id))
                    except (TypeError, ValueError):
                        pass
        output.append(int(np.count_nonzero(np.isin(labels[agent_id, ..., 0], ids)))) if ids else output.append(0)
    return tuple(output)


def _load_semantic_rows(
    path: Path, timestamps_ns: np.ndarray, issues: list[NativeT2ValidationIssue]
) -> list[Mapping[str, Any]] | None:
    rows: list[Mapping[str, Any]] = []
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        _issue(issues, "semantic_frame_metadata", path.as_posix(), str(exc))
        return None
    with stream:
        for index, line in enumerate(stream):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _issue(issues, "semantic_frame_metadata", f"{path}:{index + 1}", str(exc))
                continue
            if (
                not isinstance(record, Mapping)
                or record.get("frame_index") != index
                or record.get("timestamp_ns") != int(timestamps_ns[index])
                or not isinstance(record.get("onboard_replicator_info"), Mapping)
            ):
                _issue(issues, "semantic_frame_alignment", f"{path}:{index + 1}", "invalid frame mapping")
                continue
            rows.append(record)
    if len(rows) != len(timestamps_ns):
        _issue(issues, "semantic_frame_alignment", path.as_posix(), "metadata does not cover every raw frame")
        return None
    return rows


def _private_target_facts(
    manifest: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]:
    """Extract only the private values needed to audit public artifacts."""

    targets = manifest.get("targets") if isinstance(manifest, Mapping) else None
    if not isinstance(targets, list):
        return (), ()
    identifiers: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        target_id = target.get("target_id")
        position = target.get("position_w_m")
        if isinstance(target_id, str) and target_id:
            identifiers.append(target_id)
        if (
            isinstance(position, Sequence)
            and not isinstance(position, (str, bytes))
            and len(position) == 3
        ):
            try:
                positions.append(tuple(float(value) for value in position))
            except (TypeError, ValueError):
                pass
    return tuple(identifiers), tuple(positions)


def _audit_native_t2_public_artifact_privacy(
    *,
    receipt: Mapping[str, Any],
    public_json: Sequence[tuple[Mapping[str, Any] | None, str]],
    semantic_rows: Sequence[Mapping[str, Any]] | None,
    event_payload: Mapping[str, Any] | None,
    private_target_ids: Sequence[str],
    private_target_positions: Sequence[Sequence[float]],
    issues: list[NativeT2ValidationIssue],
) -> None:
    """Apply the shared public-artifact boundary without hiding T2 candidates.

    T2 candidate positions are public RGB-D estimates and therefore cannot use
    the generic coordinate scan.  Everything else, including every ordinary
    public JSON document and the semantic metadata, keeps the mature T1 scan.
    The event journal has its own narrow schema: evaluator-owned identifiers
    and paths are forbidden, while anonymous candidate positions remain valid.
    """

    from .isaac_validate import (
        _PUBLIC_PRIVATE_ARTIFACT_KEYS,
        _PUBLIC_PRIVATE_ARTIFACT_PATH_TOKENS,
        _private_target_id_leaks,
        _scan_public_private_artifact_json,
    )

    # The existing scanner owns the canonical private-ID/path/coordinate
    # semantics.  Translate its diagnostics so this validator has one result
    # type and one fail-closed exit condition.
    shared_issues: list[Any] = []
    for payload, relative in public_json:
        _scan_public_private_artifact_json(
            payload,
            relative,
            shared_issues,
            private_target_ids=private_target_ids,
            private_target_positions=private_target_positions,
        )
    for index, row in enumerate(semantic_rows or ()):
        relative = f"{SEMANTIC_FRAME_METADATA_RELATIVE_PATH}:{index + 1}"
        _scan_public_private_artifact_json(
            row,
            relative,
            shared_issues,
            private_target_ids=private_target_ids,
            private_target_positions=private_target_positions,
        )
        if _private_target_id_leaks(row, private_target_ids):
            _issue(
                issues,
                "semantic_private_id_leakage",
                relative,
                "semantic frame metadata contains an evaluator-private target identifier",
            )
    for shared_issue in shared_issues:
        _issue(issues, shared_issue.code, shared_issue.path, shared_issue.message)

    if event_payload is None:
        return
    # A candidate event legitimately includes ``position_w_m``.  All other
    # evaluator-owned artifact fields are forbidden, including their spelling
    # variants via ``normalized_key``.
    forbidden_event_keys = _PUBLIC_PRIVATE_ARTIFACT_KEYS - {"position_w_m"}
    normalized_private_ids = tuple(
        value.casefold()
        for value in private_target_ids
        if isinstance(value, str) and value.strip()
    )
    for tree_path, key, value in iter_tree(event_payload):
        path = f"{NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH}{tree_path[1:]}"
        if key is not None and normalized_key(key) in forbidden_event_keys:
            _issue(
                issues,
                "public_private_leakage",
                path,
                "candidate event journal contains an evaluator-owned field",
            )
        if isinstance(value, str):
            lowered = value.casefold()
            if any(token in lowered for token in _PUBLIC_PRIVATE_ARTIFACT_PATH_TOKENS):
                _issue(
                    issues,
                    "public_private_leakage",
                    path,
                    "candidate event journal contains a private evaluator path/token",
                )
            if any(token in lowered for token in normalized_private_ids):
                _issue(
                    issues,
                    "public_private_leakage",
                    path,
                    "candidate event journal contains an evaluator-private target identifier",
                )
    if _private_target_id_leaks(event_payload, private_target_ids):
        _issue(
            issues,
            "public_private_leakage",
            NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
            "candidate event journal contains an evaluator-private target identifier",
        )


def _audit_native_evidence_binding(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    decisions: Sequence[Mapping[str, Any]],
    physical_steps: Sequence[Mapping[str, Any]],
    event_payload: Mapping[str, Any] | None,
    native_extrinsics: Mapping[str, np.ndarray] | None,
    issues: list[NativeT2ValidationIssue],
) -> None:
    """Cross-check the receipt's concise T2 index against raw artifacts.

    The complete artifact inventory is already content-addressed, but this
    additional binding catches stale counts or a receipt summary that points
    to a different raw trace.  It deliberately contains no target truth.
    """

    evidence = receipt.get("native_t2_evidence")
    trace_path = root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH
    event_path = root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH
    extrinsics_path = root / NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH
    source_observations = (
        event_payload.get("source_observations")
        if isinstance(event_payload, Mapping)
        else None
    )
    journal = (
        event_payload.get("candidate_event_journal")
        if isinstance(event_payload, Mapping)
        else None
    )
    submission = journal.get("submission") if isinstance(journal, Mapping) else None
    events = submission.get("events") if isinstance(submission, Mapping) else None
    frame_count = (
        len(native_extrinsics["timestamps_ns"])
        if isinstance(native_extrinsics, Mapping)
        and isinstance(native_extrinsics.get("timestamps_ns"), np.ndarray)
        else None
    )
    expected_task_variant = _native_t2_task_variant_from_receipt(receipt)
    expected = {
        "task_variant_id": expected_task_variant,
        "claim_boundary": "development_native_t2_canary_only",
        "decision_trace": {
            "path": NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
            "sha256": sha256_file(trace_path) if trace_path.is_file() else None,
            "decision_count": len(decisions),
            "physical_step_count": len(physical_steps),
        },
        "candidate_events": {
            "path": NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
            "sha256": sha256_file(event_path) if event_path.is_file() else None,
            "source_observation_count": len(source_observations)
            if isinstance(source_observations, list)
            else None,
            "event_count": len(events) if isinstance(events, list) else None,
        },
        "camera_extrinsics": {
            "path": NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH,
            "sha256": sha256_file(extrinsics_path)
            if extrinsics_path.is_file()
            else None,
            "frame_count": frame_count,
            "world_camera_closure": "T_world_camera_from_verified_render_facing_usd_pose_converted_to_ros",
        },
    }
    if evidence != expected:
        _issue(
            issues,
            "native_t2_evidence_binding",
            "capture_receipt.json.native_t2_evidence",
            "native T2 receipt index does not match the retained trace, events, and extrinsics",
        )


def _check_external_inputs(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    evaluator_manifest: Path | None,
    cf2x_runtime_calibration: Path | None,
    runtime_lock_path: Path | None,
    issues: list[NativeT2ValidationIssue],
) -> tuple[Mapping[str, Any] | None, np.ndarray | None]:
    manifest: Mapping[str, Any] | None = None
    expected_manifest_hash = receipt.get("evaluator_manifest_sha256")
    expected_collection_binding = receipt.get("collection_binding")
    expected_task_variant = _native_t2_task_variant_from_receipt(receipt)
    motion_contract = _native_t2_motion_contract_from_receipt(
        receipt, task_variant_id=expected_task_variant
    )
    if not isinstance(expected_collection_binding, Mapping):
        _issue(
            issues,
            "native_t2_collection_binding_required",
            "capture_receipt.json.collection_binding",
            "native T2 validation requires a public canary protocol binding",
        )
    elif expected_task_variant is None:
        _issue(
            issues,
            "native_t2_collection_protocol",
            "capture_receipt.json.collection_binding.protocol_id",
            "native T2 validation requires the dedicated native T2 canary protocol",
        )
    elif expected_task_variant in (
        NATIVE_T2_V2_TASK_VARIANT_ID,
        NATIVE_T2_V3_TASK_VARIANT_ID,
    ) and motion_contract is None:
        _issue(
            issues,
            "native_t2_motion_contract",
            "capture_receipt.json.command.native_t2_canary.motion_contract",
            "revisioned native T2 validation requires a complete receipt-bound motion contract",
        )
    if evaluator_manifest is None:
        _issue(issues, "evaluator_manifest_required", "evaluator_private", "native T2 validation requires an external private manifest")
    else:
        manifest_path = evaluator_manifest.expanduser().resolve()
        if _within(manifest_path, root):
            _issue(issues, "evaluator_manifest_location", str(manifest_path), "private manifest must remain outside the capture")
        elif not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_hash:
            _issue(issues, "evaluator_manifest_hash", str(manifest_path), "private manifest does not match capture commitment")
        else:
            manifest = _read_json(manifest_path, issues)
            if manifest is not None:
                try:
                    validate_external_private_evaluator_manifest(
                        manifest,
                        city_lite_scene_contract_sha256=receipt.get("city_lite_scene", {}).get("scene_contract_sha256"),
                        city_lite_scene_payload_sha256=receipt.get("city_lite_scene", {}).get("scene_contract_payload_sha256"),
                        expected_collection_binding=(
                            expected_collection_binding
                            if isinstance(expected_collection_binding, Mapping)
                            else None
                        ),
                        expected_task_variant_id=(
                            expected_task_variant
                            if expected_task_variant is not None
                            else NATIVE_T2_TASK_VARIANT_ID
                        ),
                        expected_native_t2_motion_contract=motion_contract,
                    )
                except (PrivateEvaluatorManifestError, AttributeError) as exc:
                    _issue(issues, "evaluator_manifest_contract", str(manifest_path), str(exc))
                    manifest = None

    allocation: np.ndarray | None = None
    if runtime_lock_path is None or cf2x_runtime_calibration is None:
        _issue(issues, "native_t2_calibration_required", "capture_receipt.json", "native T2 validation requires external runtime lock and calibration report")
        return manifest, allocation
    lock_path = runtime_lock_path.expanduser().resolve()
    calibration_path = cf2x_runtime_calibration.expanduser().resolve()
    try:
        lock = load_runtime_lock(lock_path)
        lock_hash = runtime_lock_sha256(lock)
    except (OSError, RuntimeLockError, ValueError) as exc:
        _issue(issues, "runtime_lock", str(lock_path), str(exc))
        return manifest, allocation
    if receipt.get("runtime_lock", {}).get("sha256") != lock_hash:
        _issue(issues, "runtime_lock", "capture_receipt.json.runtime_lock", "runtime lock does not match the external lock")
    report = _read_json(calibration_path, issues)
    command = receipt.get("command")
    if report is None or not isinstance(command, Mapping):
        return manifest, allocation
    report_issues = validate_calibration_report(report)
    if report_issues or report.get("status") != "passed":
        _issue(issues, "cf2x_runtime_calibration", str(calibration_path), "; ".join(report_issues) or "report did not pass")
        return manifest, allocation
    try:
        expected = bind_native_t2_calibration(
            report,
            expected_usd_sha256=str(command.get("drone_usd_sha256")),
            expected_runtime_lock_sha256=lock_hash,
            expected_control_dt_s=float(command.get("dt_s")),
        )
    except (TypeError, ValueError) as exc:
        _issue(issues, "cf2x_runtime_calibration", str(calibration_path), str(exc))
        return manifest, allocation
    if receipt.get("cf2x_runtime_calibration") != expected:
        _issue(issues, "cf2x_runtime_calibration_binding", "capture_receipt.json", "calibration binding does not match external report/runtime")
    try:
        allocation = np.asarray(report["runtime"]["allocation_matrix"], dtype=np.float64)
        if allocation.shape != (6, 4) or not np.isfinite(allocation).all():
            raise ValueError("invalid allocation matrix")
    except (KeyError, TypeError, ValueError):
        _issue(issues, "cf2x_allocation", str(calibration_path), "calibration allocation matrix is invalid")
    return manifest, allocation


def _audit_native_runtime_safety_and_overview(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    scene: Mapping[str, Any] | None,
    state: Mapping[str, np.ndarray] | None,
    runtime_safety: Mapping[str, np.ndarray] | None,
    sensor_phase: Mapping[str, np.ndarray] | None,
    contact: Mapping[str, np.ndarray] | None,
    sensor_timestamps: np.ndarray | None,
    semantic_rows: Sequence[Mapping[str, Any]] | None,
    issues: list[NativeT2ValidationIssue],
    checks: dict[str, Any],
) -> None:
    """Reuse the proven physical trace gates and verify the fixed overview.

    Native T2 differs from T1 in control and candidate evidence, not in the
    City-Lite physical, synchronized-sensor, or route-witness requirements.
    Keeping those audits shared prevents a canary-specific weakened version of
    collision, timing, or fixed-world-camera validation.
    """

    # These CPU-only helpers are deliberately imported lazily.  The public
    # CLI owns the native dispatch, while both tracks retain one identical
    # implementation for full-step safety and sensor-phase replay.
    from .isaac_validate import (
        _overview_archived_visual_evidence,
        _validate_city_lite_scene,
        _validate_runtime_safety_guard,
        _validate_runtime_safety_trace,
        _validate_sensor_phase_trace,
    )

    structural_aabbs = ()
    if scene is None:
        _issue(issues, "scene", "scene.json", "native T2 capture is missing its City-Lite scene evidence")
    else:
        structural_aabbs = _validate_city_lite_scene(
            scene,
            evaluator_sha256=receipt.get("evaluator_manifest_sha256"),
            issues=issues,  # type: ignore[arg-type]
            checks=checks,
        )
    if state is None or "root_pos_w_m" not in state:
        physics_steps: int | None = None
    else:
        physics_steps = int(state["root_pos_w_m"].shape[0])
    captured_force_max: float | None = None
    if contact is not None and isinstance(contact.get("net_forces_w_n"), np.ndarray):
        force = contact["net_forces_w_n"]
        if force.ndim != 4 or force.shape[1:] != (AGENT_COUNT, 1, 3):
            _issue(issues, "contact_shape", "sensors/contact.npz", "native T2 contact must be [T,8,1,3]")
        elif not np.isfinite(force).all():
            _issue(issues, "contact_shape", "sensors/contact.npz", "native T2 contact contains non-finite values")
        else:
            captured_force_max = float(np.max(np.linalg.norm(force, axis=-1)))

    runtime_frames, runtime_force_max = _validate_runtime_safety_trace(
        runtime_safety,
        receipt,
        structural_aabbs,
        state=state,
        captured_contact=contact,
        sensor_timestamps=sensor_timestamps,
        issues=issues,  # type: ignore[arg-type]
        checks=checks,
    )
    _validate_runtime_safety_guard(
        receipt,
        scene,
        structural_aabbs,
        physics_steps=physics_steps,
        captured_contact_force_max_n=captured_force_max,
        runtime_trace_frame_count=runtime_frames,
        runtime_trace_max_contact_force_n=runtime_force_max,
        root=root,
        issues=issues,  # type: ignore[arg-type]
        checks=checks,
    )
    _validate_sensor_phase_trace(
        sensor_phase,
        receipt,
        state=state,
        contact=contact,
        timestamps=sensor_timestamps,
        root=root,
        issues=issues,  # type: ignore[arg-type]
        checks=checks,
    )

    if sensor_timestamps is None or state is None:
        _issue(issues, "route_witness_state_alignment", "sensors/overview_rgb.npz", "cannot bind overview without raw sensor times and state")
        return
    overview_path = root / "sensors/overview_rgb.npz"
    try:
        with ChunkedFrameArchive(overview_path) as overview:
            expected_fields = {
                "timestamps_ns",
                "rgb",
                "semantic_segmentation",
                "camera_pos_w_m",
                "camera_quat_wxyz",
                "target_w_m",
            }
            if overview.fields != expected_fields:
                raise ValueError("overview archive fields are not the fixed low-rate witness contract")
            overview_times = overview.timestamps_ns
            indices = _overview_archive_frame_indices(len(sensor_timestamps))
            expected_times = sensor_timestamps[np.asarray(indices, dtype=np.int64)]
            if not np.array_equal(overview_times, expected_times):
                raise ValueError("overview timestamps are not the fixed first/stride/final schedule")
            positions = overview.array("camera_pos_w_m")
            quaternions = overview.array("camera_quat_wxyz")
            targets = overview.array("target_w_m")
            if (
                positions.shape != (len(overview_times), 3)
                or quaternions.shape != (len(overview_times), 4)
                or targets.shape != (len(overview_times), 3)
                or not all(np.isfinite(value).all() for value in (positions, quaternions, targets))
            ):
                raise ValueError("overview pose arrays are malformed")
            expected_views = [_public_route_witness_view_at_time_ns(int(timestamp)) for timestamp in overview_times]
            expected_positions = np.asarray([view["eye_w_m"] for view in expected_views], dtype=np.float64)
            expected_targets = np.asarray([view["target_w_m"] for view in expected_views], dtype=np.float64)
            expected_quaternions = np.asarray([view["orientation_wxyz"] for view in expected_views], dtype=np.float64)
            norms = np.linalg.norm(quaternions, axis=-1)
            alignment = np.abs(np.sum(quaternions / norms[:, None] * expected_quaternions, axis=-1))
            if (
                np.any(norms <= 1.0e-8)
                or float(np.max(np.linalg.norm(positions - expected_positions, axis=-1))) > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
                or float(np.max(np.linalg.norm(targets - expected_targets, axis=-1))) > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
                or float(np.min(alignment)) < math.cos(0.01 / 2.0)
            ):
                raise ValueError("overview is not the frozen world-frame route witness")
            if state.get("effective_time_ns") is None or state.get("root_pos_w_m") is None:
                raise ValueError("state stream cannot bind the overview witness")
            state_times = state["effective_time_ns"]
            state_indices = np.searchsorted(state_times, overview_times)
            if (
                state_times.dtype != np.int64
                or np.any(state_indices >= len(state_times))
                or not np.array_equal(state_times[state_indices], overview_times)
            ):
                raise ValueError("overview timestamps are absent from the physical state stream")
            tracked = state["root_pos_w_m"][state_indices, 2]
            displacement = float(np.max(np.linalg.norm(tracked - tracked[0:1], axis=-1)))
            checks["route_witness_tracked_agent_max_displacement_m"] = displacement
            if displacement < OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M:
                raise ValueError("Agent 2 did not move enough in the fixed overview")
            metadata_by_time = {
                int(row["timestamp_ns"]): row.get("overview_replicator_info", {})
                for row in semantic_rows or ()
                if isinstance(row.get("timestamp_ns"), int)
            }
            for frame_index, timestamp in enumerate(overview_times):
                rgb = overview.frame("rgb", frame_index)
                semantic = overview.frame("semantic_segmentation", frame_index)
                metadata = metadata_by_time.get(int(timestamp), {})
                visual = _overview_archived_visual_evidence(rgb, semantic, metadata)
                if visual.get("rgb_evidence_passed") is not True or visual.get("structural_evidence_passed") is not True:
                    raise ValueError(f"overview frame {frame_index} lacks retained City-Lite visual evidence")
                identity = _overview_tracked_agent_visibility_evidence(semantic, metadata)
                if identity.get("passed") is not True:
                    raise ValueError(f"overview frame {frame_index} lacks the Agent 2 identity marker")
    except (OSError, ValueError, FrameArchiveError, np.linalg.LinAlgError) as exc:
        _issue(issues, "route_witness", "sensors/overview_rgb.npz", str(exc))


def validate_native_t2_capture(
    root: Path,
    *,
    evaluator_manifest: Path | None,
    cf2x_runtime_calibration: Path | None,
    runtime_lock_path: Path | None,
    require_clean_source: bool = False,
) -> NativeT2ValidationResult:
    """Validate a completed native T2 capture without importing Isaac or Torch."""

    root = root.resolve()
    issues: list[NativeT2ValidationIssue] = []
    checks: dict[str, Any] = {
        "claim_boundary": "development_native_t2_canary_only",
        "formal_benchmark_admission": False,
        "native_t2_independent_replay": False,
    }
    receipt_path = root / "capture_receipt.json"
    receipt = _read_json(receipt_path, issues)
    if receipt is None:
        return NativeT2ValidationResult(checks, tuple(issues))
    receipt_hash = sha256_file(receipt_path)
    if receipt.get("status") != "captured" or receipt.get("ok") is not True:
        _issue(issues, "capture_status", "capture_receipt.json", "capture did not complete")
    if receipt.get("task_kind") != NATIVE_T2_TASK_KIND or receipt.get("information_profile") != "state_only_control_plus_rgbd_semantic_events":
        _issue(issues, "native_t2_task", "capture_receipt.json", "receipt is not a native T2 capture")
    if require_clean_source and receipt.get("source_worktree_dirty") is not False:
        _issue(issues, "dirty_source", "capture_receipt.json", "native T2 validation requires clean source")
    command = receipt.get("command")
    if not isinstance(command, Mapping) or command.get("control_mode") != CONTROL_MODE_NATIVE_T2_CANARY:
        _issue(issues, "native_t2_command", "capture_receipt.json", "receipt does not bind the native T2 control mode")
        command = {}
    integrity = receipt.get("capture_integrity")
    if not isinstance(integrity, Mapping) or any(
        integrity.get(key) is not expected
        for key, expected in (("online_capture", True), ("queue_used", False), ("queue_overflow", False), ("silent_frame_drop", False), ("synchronous_sensor_reads", True))
    ):
        _issue(issues, "capture_integrity", "capture_receipt.json", "native T2 capture integrity contract failed")
    boundary = receipt.get("claim_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("formal_benchmark_admission") is not False or boundary.get("development_native_t2_canary") is not True:
        _issue(issues, "claim_boundary", "capture_receipt.json", "native canary claim boundary is invalid")

    bound = receipt.get("artifact_hashes")
    if not isinstance(bound, Mapping) or set(bound) not in (NATIVE_T2_EXPECTED_ARTIFACTS, NATIVE_T2_EXPECTED_ARTIFACTS | _CONTROL_ARTIFACTS):
        _issue(issues, "artifact_inventory", "capture_receipt.json", "native T2 artifact inventory is not exact")
        bound = {}
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"capture_receipt.json", "capture_receipt.sha256", "independent_validation.json"}
    }
    expected = NATIVE_T2_EXPECTED_ARTIFACTS | (_CONTROL_ARTIFACTS if "capture_start.json" in present else frozenset())
    if present != expected:
        _issue(issues, "closed_world", ".", f"unexpected/missing native T2 files: {sorted(present ^ expected)}")
    for relative in NATIVE_T2_EXPECTED_ARTIFACTS:
        path = root / PurePosixPath(relative)
        item = bound.get(relative) if isinstance(bound, Mapping) else None
        if not path.is_file() or not isinstance(item, Mapping) or item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            _issue(issues, "artifact_hash", relative, "capture-bound artifact is missing or modified")
    checksum = root / "capture_receipt.sha256"
    if not checksum.is_file() or checksum.read_text(encoding="ascii", errors="replace") != f"{receipt_hash}  capture_receipt.json\n":
        _issue(issues, "receipt_checksum", "capture_receipt.sha256", "receipt checksum is missing or stale")

    manifest, allocation = _check_external_inputs(
        root,
        receipt,
        evaluator_manifest=evaluator_manifest,
        cf2x_runtime_calibration=cf2x_runtime_calibration,
        runtime_lock_path=runtime_lock_path,
        issues=issues,
    )
    checks["evaluator_manifest_verified"] = manifest is not None
    checks["cf2x_runtime_calibration_verified"] = allocation is not None

    state = _load_npz(root / "streams/state_action.npz", issues)
    native_extrinsics = _load_npz(root / NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH, issues)
    lidar = _load_npz(root / "sensors/lidar.npz", issues)
    contact = _load_npz(root / "sensors/contact.npz", issues)
    runtime_safety = _load_npz(root / "sensors/runtime_safety.npz", issues)
    sensor_phase = _load_npz(root / "sensors/sensor_phase.npz", issues)
    calibration = _read_json(root / "calibration.json", issues)
    public_task = _read_json(root / "public_task.json", issues)
    scene = _read_json(root / "scene.json", issues)
    outcome = _read_json(root / "task_outcome.json", issues)
    progress = _read_json(root / "capture_progress.json", issues)
    semantic_metadata = _read_json(root / "learning_labels/semantic_metadata.json", issues)
    event_payload = _read_json(root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH, issues)

    required_state = {
        "command_time_ns", "effective_time_ns", "root_pos_w_m", "root_quat_wxyz", "root_lin_vel_w_mps", "root_ang_vel_b_radps", "desired_pos_w_m", "desired_vel_w_mps", "target_thrust_n", "applied_thrust_n", "pre_command_root_pos_w_m", "pre_command_root_quat_wxyz", "pre_command_root_lin_vel_w_mps", "pre_command_root_ang_vel_b_radps", "emitted_world_velocity_yaw_command",
    }
    steps = 0
    timestamps: np.ndarray | None = None
    if state is None or set(state) != required_state:
        _issue(issues, "state_action_schema", "streams/state_action.npz", "native T2 state/action fields are not exact")
    else:
        command_times, effective_times = state["command_time_ns"], state["effective_time_ns"]
        if command_times.dtype != np.int64 or effective_times.dtype != np.int64 or command_times.ndim != 1 or effective_times.shape != command_times.shape:
            _issue(issues, "action_time", "streams/state_action.npz", "action timestamps must be int64 vectors")
        else:
            steps = len(command_times)
            if steps < 1 or not (np.all(np.diff(command_times) > 0) and np.all(np.diff(effective_times) > 0) and np.all(command_times < effective_times)):
                _issue(issues, "action_causality", "streams/state_action.npz", "commands must precede strictly ordered post-step state")
        shapes = {
            "root_pos_w_m": (steps, AGENT_COUNT, 3), "root_quat_wxyz": (steps, AGENT_COUNT, 4), "root_lin_vel_w_mps": (steps, AGENT_COUNT, 3), "root_ang_vel_b_radps": (steps, AGENT_COUNT, 3), "desired_pos_w_m": (steps, AGENT_COUNT, 3), "desired_vel_w_mps": (steps, AGENT_COUNT, 3), "target_thrust_n": (steps, AGENT_COUNT, 4), "applied_thrust_n": (steps, AGENT_COUNT, 4), "pre_command_root_pos_w_m": (steps, AGENT_COUNT, 3), "pre_command_root_quat_wxyz": (steps, AGENT_COUNT, 4), "pre_command_root_lin_vel_w_mps": (steps, AGENT_COUNT, 3), "pre_command_root_ang_vel_b_radps": (steps, AGENT_COUNT, 3), "emitted_world_velocity_yaw_command": (steps, AGENT_COUNT, 4),
        }
        for name, shape in shapes.items():
            value = state[name]
            if value.shape != shape or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                _issue(issues, "state_shape", f"streams/state_action.npz.{name}", f"expected finite {shape}")
        if not np.any(state["applied_thrust_n"] > 0.0):
            _issue(issues, "zero_applied_thrust", "streams/state_action.npz", "no nonzero applied rotor thrust")

    expected_task_variant = _native_t2_task_variant_from_receipt(receipt)
    motion_contract = _native_t2_motion_contract_from_receipt(
        receipt, task_variant_id=expected_task_variant
    )
    if motion_contract is not None:
        _validate_motion_command_binding(command, motion_contract, issues)
    if (
        public_task is None
        or public_task.get("task_kind") != NATIVE_T2_TASK_KIND
        or public_task.get("task_variant_id") != expected_task_variant
    ):
        _issue(issues, "public_task_contract", "public_task.json", "native T2 public task contract is invalid")
        public_task = None
    evaluator_contract: Mapping[str, Any] | None = None
    routes: np.ndarray | None = None
    if public_task is not None:
        if expected_task_variant in (
            NATIVE_T2_V2_TASK_VARIANT_ID,
            NATIVE_T2_V3_TASK_VARIANT_ID,
        ):
            if motion_contract is None or public_task.get("motion_contract") != motion_contract:
                _issue(
                    issues,
                    "native_t2_motion_contract",
                    "public_task.json.motion_contract",
                    "public task motion contract does not match the receipt-bound native T2 contract",
                )
        try:
            routes = np.asarray(public_task["routes_w_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            routes = None
        evaluator_contract = public_task.get("evaluator_contract") if isinstance(public_task.get("evaluator_contract"), Mapping) else None
        if routes is None or routes.ndim != 3 or routes.shape[0] != AGENT_COUNT or routes.shape[2] != 3 or not np.isfinite(routes).all():
            _issue(issues, "public_routes", "public_task.json", "T2 public routes are invalid")
            routes = None
        elif motion_contract is not None:
            native_command = command.get("native_t2_canary")
            recorded_feasibility = (
                native_command.get("route_timing_feasibility")
                if isinstance(native_command, Mapping)
                else None
            )
            try:
                expected_feasibility = validate_route_timing_feasibility(
                    routes,
                    waypoint_segment_seconds=float(motion_contract["waypoint_segment_seconds"]),
                    max_horizontal_speed_mps=float(motion_contract["max_horizontal_speed_mps"]),
                    max_vertical_speed_mps=float(motion_contract["max_vertical_speed_mps"]),
                    utilization_limit=float(motion_contract["route_speed_utilization_limit"]),
                )
            except ValueError as exc:
                _issue(
                    issues,
                    "native_t2_route_timing_feasibility",
                    "public_task.json.routes_w_m",
                    f"public route is not feasible under the frozen motion contract: {exc}",
                )
            else:
                if recorded_feasibility != expected_feasibility:
                    _issue(
                        issues,
                        "native_t2_route_timing_feasibility",
                        "capture_receipt.json.command.native_t2_canary.route_timing_feasibility",
                        "receipt route timing feasibility does not match the frozen contract and public route",
                    )
        expected_contract = {
            "schema": "org.rivermark.native-t2-private-evaluation-contract.v1",
            "event_time_origin": "post_warmup_physics_time",
            "time_budget_s": float(command.get("steps", 0)) * float(command.get("dt_s", 0.0)),
            "match_radius_m": NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
            "maximum_false_confirmations": 0,
            "minimum_verified_matches": 1,
            "observation_time_tolerance_s": 0.0,
            "target_count_source": "external_private_evaluator_manifest",
        }
        if evaluator_contract != expected_contract:
            _issue(issues, "evaluator_contract", "public_task.json", "native T2 evaluator contract is not frozen")
            evaluator_contract = None

    trace_path = root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH
    trace_records: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
            record = json.loads(line)
            if not isinstance(record, Mapping) or record.get("schema") != NATIVE_T2_TRACE_SCHEMA:
                raise ValueError(f"line {line_number} has an invalid native T2 trace record")
            trace_records.append(record)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "native_t2_trace", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, str(exc))
    provenance = trace_records[0] if trace_records else None
    if not isinstance(provenance, Mapping) or provenance.get("record_type") != "provenance" or provenance.get("capture_attempt_id") != receipt.get("capture_attempt_id"):
        _issue(issues, "native_t2_trace", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, "missing bound provenance record")
    decisions = [record for record in trace_records if record.get("record_type") == "decision"]
    physical = [record for record in trace_records if record.get("record_type") == "physical_step"]
    if len(trace_records) != 1 + len(decisions) + len(physical) or len(physical) != steps:
        _issue(issues, "native_t2_trace_coverage", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, "trace must contain one provenance record and one physical record per state step")

    _audit_native_evidence_binding(
        root,
        receipt,
        decisions=decisions,
        physical_steps=physical,
        event_payload=event_payload,
        native_extrinsics=native_extrinsics,
        issues=issues,
    )

    decision_by_step: dict[int, Mapping[str, Any]] = {}
    if provenance is not None and routes is not None and state is not None:
        policy = provenance.get("policy")
        policy_abi = provenance.get("policy_abi")
        try:
            bounds_record = policy_abi["world_command_bounds"]
            bounds = WorldCommandBounds(
                max_horizontal_speed_mps=float(bounds_record["max_horizontal_speed_mps"]),
                max_vertical_speed_mps=float(bounds_record["max_vertical_speed_mps"]),
                max_yaw_rate_rad_s=float(bounds_record["max_yaw_rate_rad_s"]),
            )
            stride = int(policy_abi["decision_cadence_physics_steps"])
            route_start = physics_time_ns(int(command["warmup_steps"]), float(command["dt_s"]))
            policy_kwargs: dict[str, float] = {}
            if motion_contract is not None:
                if (
                    float(public_task["waypoint_segment_seconds"])
                    != float(motion_contract["waypoint_segment_seconds"])
                    or stride != int(motion_contract["decision_stride_physics_steps"])
                    or bounds_record != {
                        "max_horizontal_speed_mps": float(motion_contract["max_horizontal_speed_mps"]),
                        "max_vertical_speed_mps": float(motion_contract["max_vertical_speed_mps"]),
                        "max_yaw_rate_rad_s": float(motion_contract["max_yaw_rate_rad_s"]),
                    }
                ):
                    raise ValueError("policy bounds or timing do not match the receipt-bound native T2 motion contract")
                policy_kwargs = {
                    "position_feedback_gain": float(motion_contract["position_feedback_gain"]),
                    "yaw_feedback_gain": float(motion_contract["yaw_feedback_gain"]),
                }
            route_policy = PublicRouteCoveragePolicy(
                routes,
                waypoint_segment_seconds=float(public_task["waypoint_segment_seconds"]),
                route_start_time_ns=route_start,
                **policy_kwargs,
            )
            if policy != route_policy.provenance() or policy_abi.get("claim_boundary") != "development_native_t2_canary_only":
                raise ValueError("policy provenance does not bind the public route and rollout origin")
            cadence = FixedDecisionCadence(stride)
        except (KeyError, TypeError, ValueError) as exc:
            _issue(issues, "native_t2_policy_provenance", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, str(exc))
            bounds = None
            cadence = None
        for expected_index, record in enumerate(decisions):
            decision = record.get("decision")
            step = record.get("rollout_physics_step")
            if not isinstance(decision, Mapping) or isinstance(step, bool) or not isinstance(step, int):
                _issue(issues, "native_t2_decision", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, "malformed decision record")
                continue
            if cadence is None or step != expected_index * cadence.every_physics_steps or step >= steps:
                _issue(issues, "native_t2_decision_cadence", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, "decision cadence has a gap or unexpected step")
                continue
            if record.get("decision_sha256") != _canonical_sha256(decision):
                _issue(issues, "native_t2_decision_hash", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, "decision hash is stale")
            try:
                observation = T2PublicFleetObservation.from_rigid_body_state(
                    physics_step=step,
                    command_time_ns=int(state["command_time_ns"][step]),
                    position_w_m=state["pre_command_root_pos_w_m"][step],
                    linear_velocity_w_mps=state["pre_command_root_lin_vel_w_mps"][step],
                    quaternion_wxyz=state["pre_command_root_quat_wxyz"][step],
                    angular_velocity_b_radps=state["pre_command_root_ang_vel_b_radps"][step],
                )
                expected_raw = route_policy(observation)
                expected_action = bounds.apply(expected_raw[:, :3], expected_raw[:, 3]) if bounds is not None else None
                if decision.get("observation") != observation.public_dict() or decision.get("observation_sha256") != observation.sha256:
                    raise ValueError("decision observation is not the pre-command native state")
                action = decision.get("action")
                emitted = np.concatenate((expected_action[0], expected_action[1][:, None]), axis=1) if expected_action is not None else None
                if not isinstance(action, Mapping) or not _allclose(action.get("raw_velocity_yaw_command"), expected_raw) or not _allclose(action.get("emitted_velocity_yaw_command"), emitted):
                    raise ValueError("decision action is not the bounded public-route action")
                if not _allclose(state["emitted_world_velocity_yaw_command"][step], emitted):
                    raise ValueError("state stream does not carry the decision's emitted command")
            except (KeyError, TypeError, ValueError) as exc:
                _issue(issues, "native_t2_decision", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, str(exc))
                continue
            decision_by_step[step] = record

    if allocation is not None and state is not None:
        current_decision: Mapping[str, Any] | None = None
        for step, record in enumerate(physical):
            if step in decision_by_step:
                current_decision = decision_by_step[step]
            evidence = record.get("evidence")
            if not isinstance(evidence, Mapping) or record.get("rollout_physics_step") != step or record.get("global_applied_physics_step") != int(command.get("warmup_steps", 0)) + step + 1:
                _issue(issues, "native_t2_physical_trace", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} is malformed")
                continue
            if current_decision is None or evidence.get("decision_sha256") != current_decision.get("decision_sha256"):
                _issue(issues, "native_t2_action_causality", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} is not bound to its current decision")
            decision_payload = current_decision.get("decision") if isinstance(current_decision, Mapping) else None
            decision_observation = decision_payload.get("observation") if isinstance(decision_payload, Mapping) else None
            if (
                evidence.get("schema") != T2_NATIVE_STEP_EVIDENCE_SCHEMA
                or evidence.get("applied_physics_step") != step + 1
                or evidence.get("decision_command_time_ns") != (decision_observation.get("command_time_ns") if isinstance(decision_observation, Mapping) else None)
                or evidence.get("physical_command_time_ns") != int(state["command_time_ns"][step])
                or evidence.get("effective_time_ns") != int(state["effective_time_ns"][step])
            ):
                _issue(issues, "native_t2_action_causality", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} timing is invalid")
            elif not (
                int(evidence["decision_command_time_ns"])
                <= int(evidence["physical_command_time_ns"])
                < int(evidence["effective_time_ns"])
            ):
                _issue(issues, "native_t2_action_causality", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} causal time order is invalid")
            requested = evidence.get("requested_thrust_n")
            applied = evidence.get("applied_thrust_n")
            wrench = evidence.get("applied_wrench_body")
            if not _allclose(requested, state["target_thrust_n"][step]) or not _allclose(applied, state["applied_thrust_n"][step]):
                _issue(issues, "native_t2_actuator_binding", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} thrust differs from state stream")
            elif not _allclose(np.asarray(applied, dtype=np.float64) @ allocation.T, wrench):
                _issue(issues, "native_t2_allocation_wrench", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} wrench does not equal calibration allocation * applied thrust")
            try:
                post = derive_physical_state_8d(
                    state["root_pos_w_m"][step], state["root_lin_vel_w_mps"][step], state["root_quat_wxyz"][step], state["root_ang_vel_b_radps"][step], agent_ids=range(AGENT_COUNT)
                ).values
                if not _allclose(evidence.get("post_step_state_8d"), post):
                    raise ValueError
            except (TypeError, ValueError):
                _issue(issues, "native_t2_post_state", NATIVE_T2_DECISION_TRACE_RELATIVE_PATH, f"physical step {step} post-state is not the recorded native state")

    # Raw RGB-D/semantic replay and event evaluation are deliberately after all
    # trace checks: a malformed trace cannot be rescued by a good-looking event
    # journal.
    source_observations: list[dict[str, Any]] = []
    visible_by_slot: dict[str, list[str]] = {}
    replay_submission: Mapping[str, Any] | None = None
    semantic_rows: list[Mapping[str, Any]] | None = None
    if state is not None and native_extrinsics is not None and lidar is not None and calibration is not None and event_payload is not None:
        try:
            with ChunkedFrameArchive(root / "sensors/onboard_rgbd.npz") as rgbd, ChunkedFrameArchive(root / "learning_labels/semantic_segmentation.npz") as semantic:
                timestamps = rgbd.timestamps_ns
                if set(rgbd.fields) != {"timestamps_ns", "rgb", "distance_to_image_plane_m"} or set(semantic.fields) != {"timestamps_ns", "semantic_segmentation"} or not np.array_equal(timestamps, semantic.timestamps_ns):
                    raise ValueError("raw RGB-D and semantic archive fields/timestamps are invalid")
                for key, shape in (("timestamps_ns", (len(timestamps),)), ("pos_w_m", (len(timestamps), AGENT_COUNT, 3)), ("quat_w_ros", (len(timestamps), AGENT_COUNT, 4)), ("intrinsic_matrices", (len(timestamps), AGENT_COUNT, 3, 3))):
                    if key not in native_extrinsics or native_extrinsics[key].shape != shape:
                        raise ValueError(f"native camera extrinsics {key} is invalid")
                if native_extrinsics["timestamps_ns"].dtype != np.int64 or not np.array_equal(timestamps, native_extrinsics["timestamps_ns"]):
                    raise ValueError("native camera extrinsics timestamps are not frame-aligned")
                if lidar.get("timestamps_ns") is None or not np.array_equal(timestamps, lidar["timestamps_ns"]):
                    raise ValueError("LiDAR timestamps are not frame-aligned")
                metadata = _load_semantic_rows(root / SEMANTIC_FRAME_METADATA_RELATIVE_PATH, timestamps, issues)
                if metadata is None:
                    raise ValueError("semantic metadata cannot be replayed")
                semantic_rows = metadata
                event_journal = event_payload.get("candidate_event_journal")
                origin = event_journal.get("event_time_origin_ns") if isinstance(event_journal, Mapping) else None
                expected_origin = physics_time_ns(int(command["warmup_steps"]), float(command["dt_s"]))
                if origin != expected_origin:
                    raise ValueError("candidate event journal has the wrong rollout time origin")
                journal = T2CandidateEventJournal(episode_id=str(receipt["capture_attempt_id"]), event_time_origin_ns=int(origin))
                deduplicator = SpatialCandidateDeduplicator(NATIVE_T2_CANDIDATE_MERGE_RADIUS_M)
                slot_count = len(manifest.get("targets", [])) if manifest is not None else 0
                visible_by_slot = {f"search_target_slot_{index:03d}": [] for index in range(slot_count)}
                lidar_max = float(calibration["lidar"]["max_distance_m"])
                for frame_index, timestamp in enumerate(timestamps):
                    depth = rgbd.frame("distance_to_image_plane_m", frame_index)
                    labels = semantic.frame("semantic_segmentation", frame_index)
                    raw_metadata = metadata[frame_index]["onboard_replicator_info"]
                    visual = _onboard_visual_intrusion_evidence(depth, lidar["ranges_m"][frame_index], lidar_max_distance_m=lidar_max)
                    content = _onboard_scene_content_evidence(depth, labels, raw_metadata)
                    if visual.get("passed") is not True:
                        _issue(issues, "visual_intrusion", "sensors/onboard_rgbd.npz", f"raw frame {frame_index} fails the RGB-D/LiDAR gate")
                    if content.get("passed") is not True:
                        _issue(issues, "onboard_scene_content", "sensors/onboard_rgbd.npz", f"raw frame {frame_index} fails the scene-content gate")
                    points = _native_world_points(depth, native_extrinsics["intrinsic_matrices"][frame_index], native_extrinsics["pos_w_m"][frame_index], native_extrinsics["quat_w_ros"][frame_index])
                    candidates = native_semantic_rgbd_candidates(labels, raw_metadata, points, minimum_pixels=NATIVE_T2_CANDIDATE_MINIMUM_PIXELS)
                    for slot in visible_by_slot:
                        for agent_id, pixels in enumerate(_semantic_slot_pixels(raw_metadata, labels, slot)):
                            if pixels >= PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS:
                                visible_by_slot[slot].append(f"obs-a{agent_id:02d}-f{frame_index:08d}")
                    for agent_id, rows in enumerate(candidates):
                        observation = T2PublicSensorObservation(agent_id=agent_id, capture_frame_index=frame_index, sensor_time_ns=int(timestamp))
                        source_observations.append(observation.public_dict())
                        journal.append(observation, deduplicator.filter(rows))
                replay = journal.public_dict()
                replay_submission = replay["submission"]
                if event_payload.get("schema") != NATIVE_T2_EVENTS_SCHEMA or event_payload.get("decision_trace_sha256") != sha256_file(root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH) or event_payload.get("source_observations") != source_observations or event_payload.get("candidate_event_journal") != replay:
                    _issue(issues, "native_t2_event_replay", NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH, "event journal is not an exact replay of raw public RGB-D semantics")
        except (KeyError, TypeError, ValueError, FrameArchiveError, np.linalg.LinAlgError) as exc:
            _issue(issues, "native_t2_sensor_replay", "sensors/onboard_rgbd.npz", str(exc))

    _audit_native_runtime_safety_and_overview(
        root,
        receipt,
        scene=scene,
        state=state,
        runtime_safety=runtime_safety,
        sensor_phase=sensor_phase,
        contact=contact,
        sensor_timestamps=timestamps,
        semantic_rows=semantic_rows,
        issues=issues,
        checks=checks,
    )

    private_target_ids, private_target_positions = _private_target_facts(manifest)
    _audit_native_t2_public_artifact_privacy(
        receipt=receipt,
        public_json=(
            (receipt, "capture_receipt.json"),
            (scene, "scene.json"),
            (public_task, "public_task.json"),
            (outcome, "task_outcome.json"),
            (calibration, "calibration.json"),
            (progress, "capture_progress.json"),
            (semantic_metadata, "learning_labels/semantic_metadata.json"),
        ),
        semantic_rows=semantic_rows,
        event_payload=event_payload,
        private_target_ids=private_target_ids,
        private_target_positions=private_target_positions,
        issues=issues,
    )

    if manifest is not None and evaluator_contract is not None and replay_submission is not None:
        targets = manifest.get("targets")
        if isinstance(targets, list) and len(targets) == len(visible_by_slot) and all(visible_by_slot.values()):
            safety = {
                "collision": 0,
                "geofence": 0,
                "visual_intrusion": sum(issue.code == "visual_intrusion" for issue in issues),
            }
            if contact is not None and isinstance(contact.get("net_forces_w_n"), np.ndarray):
                safety["collision"] = int(np.any(np.linalg.norm(contact["net_forces_w_n"], axis=-1) >= 0.01))
            private_task = {
                "schema": PRIVATE_TASK_SCHEMA,
                "episode_id": str(receipt["capture_attempt_id"]),
                "agent_count": AGENT_COUNT,
                "time_budget_s": evaluator_contract["time_budget_s"],
                "match_radius_m": evaluator_contract["match_radius_m"],
                "maximum_false_confirmations": evaluator_contract["maximum_false_confirmations"],
                "observation_time_tolerance_s": evaluator_contract["observation_time_tolerance_s"],
                "observations": [
                    {"observation_id": row["observation_id"], "agent_id": row["agent_id"], "timestamp_s": (int(row["sensor_time_ns"]) - int(physics_time_ns(int(command["warmup_steps"]), float(command["dt_s"])))) / 1_000_000_000.0}
                    for row in source_observations
                ],
                "targets": [
                    {"target_id": row["target_id"], "position_w_m": row["position_w_m"], "visible_observation_ids": visible_by_slot[f"search_target_slot_{index:03d}"]}
                    for index, row in enumerate(targets)
                ],
                "safety_violations": safety,
            }
            try:
                evaluation = evaluate_search_events(replay_submission, private_task=private_task)
                checks["private_event_evaluation"] = evaluation.public_dict()
                minimum_verified_matches = evaluator_contract["minimum_verified_matches"]
                if not evaluation.eligible:
                    _issue(
                        issues,
                        "native_t2_private_evaluation",
                        NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
                        "private evaluator rejected the replayed candidate submission",
                    )
                elif evaluation.matched_count < minimum_verified_matches:
                    _issue(
                        issues,
                        "native_t2_minimum_verified_matches",
                        NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
                        "replayed candidate submission has fewer verified matches than the frozen canary minimum",
                    )
                else:
                    checks["private_event_evaluation_verified"] = True
            except ValueError as exc:
                _issue(issues, "native_t2_private_evaluation", NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH, str(exc))
        else:
            _issue(issues, "native_t2_visibility", NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH, "private targets have no recomputed sensor-visible observation")

    checks["physics_steps"] = steps
    checks["native_t2_decision_count"] = len(decisions)
    checks["native_t2_physical_step_count"] = len(physical)
    checks["native_t2_independent_replay"] = not issues
    return NativeT2ValidationResult(checks, tuple(issues))


__all__ = [
    "NATIVE_T2_EXPECTED_ARTIFACTS",
    "NativeT2ValidationIssue",
    "NativeT2ValidationResult",
    "validate_native_t2_capture",
]
