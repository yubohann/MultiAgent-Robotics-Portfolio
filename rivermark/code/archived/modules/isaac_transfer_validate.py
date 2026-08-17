"""Independently validate a development-only SB3-to-Isaac control transfer.

This validator deliberately imports neither Isaac Sim, Torch, Gymnasium, nor
Stable-Baselines3.  It only replays the persisted public state-to-command ABI
for a City-Lite control-wiring smoke capture.  A passing result is evidence of
trace integrity, not Isaac training, physical training, a foundation-model
integration, formal benchmark admission, or a dataset episode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .citylite_scene import AGENT_COUNT
from .isaac_runtime_safety import (
    RUNTIME_SAFETY_FRAME_OUTCOME_CODES,
    RUNTIME_SAFETY_PHASE_CODES,
    RUNTIME_SAFETY_SCHEMA,
    RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
    RUNTIME_SAFETY_TRACE_SCHEMA,
    physics_time_ns,
)
from .isaac_transfer import (
    CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1,
    DEVELOPMENT_CLAIM_BOUNDARY,
    EXCLUDED_POLICY_INPUTS,
    STATE_FIELDS,
    STATE_ONLY_PROFILE,
    TRANSFER_SCHEMA,
    TRANSFER_SOURCE,
    CityLiteRouteAnchorTransform,
    StateOnlyTransferError,
    WorldCommandBounds,
    derive_physical_state_8d,
)


TRANSFER_VALIDATION_SCHEMA = (
    "org.rivermark.isaac-state-only-transfer-independent-validation.v1"
)
CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
CONTROL_MODE = "sb3_state_only_transfer"
TASK_KIND = "state_only_control_transfer_smoke"
TASK_VARIANT_ID = "isaac-eight-agent-sb3-state-only-control-transfer-smoke-v1"
TRACE_SCHEMA = "org.rivermark.isaac-sb3-state-transfer-trace.v1"
TRACE_PATH = "streams/sb3_state_only_transfer.npz"
TRACE_PROVENANCE_PATH = "streams/sb3_state_only_transfer_provenance.json"
STATE_ACTION_PATH = "streams/state_action.npz"
SCENE_PATH = "scene.json"
RUNTIME_SAFETY_PATH = RUNTIME_SAFETY_TRACE_RELATIVE_PATH

STATE_ACTION_FIELDS = (
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
    "pre_command_root_pos_w_m",
    "pre_command_root_quat_wxyz",
    "pre_command_root_lin_vel_w_mps",
    "pre_command_root_ang_vel_b_radps",
    "emitted_world_velocity_yaw_command",
)
TRACE_FIELDS = (
    "rollout_physics_step",
    "command_time_ns",
    "effective_time_ns",
    "decision_index",
    "physical_state_8d",
    "pilot_state_8d",
    "normalized_observation_8d",
    "raw_action",
    "normalized_action",
    "local_velocity_yaw_command",
    "prebound_world_velocity_yaw_command",
    "emitted_world_velocity_yaw_command",
    "altitude_reference_w_m",
)
RUNTIME_SAFETY_FIELDS = (
    "physics_step",
    "physics_time_ns",
    "phase_code",
    "frame_outcome_code",
    "root_pos_w_m",
    "net_contact_forces_w_n",
    "max_contact_force_n",
)
REQUIRED_ARTIFACTS = (
    SCENE_PATH,
    STATE_ACTION_PATH,
    TRACE_PATH,
    TRACE_PROVENANCE_PATH,
    RUNTIME_SAFETY_PATH,
)
_FLOAT_ATOL = 1.0e-6
_FLOAT_RTOL = 1.0e-6


@dataclass(frozen=True)
class TransferValidationIssue:
    """One fail-closed problem found in a transfer capture."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class IsaacTransferValidationReport:
    """Result of independently replaying one state-only transfer capture."""

    capture_root: Path
    receipt_sha256: str | None
    checks: Mapping[str, Any]
    issues: tuple[TransferValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class _TransferContract:
    cadence_steps: int
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_scale: np.ndarray
    transform: CityLiteRouteAnchorTransform
    bounds: WorldCommandBounds


def _issue(
    issues: list[TransferValidationIssue], code: str, path: str, message: str
) -> None:
    issues.append(TransferValidationIssue(code=code, path=path, message=message))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(
    path: Path, *, relative: str, issues: list[TransferValidationIssue]
) -> Mapping[str, Any] | None:
    if not path.is_file():
        _issue(issues, "missing_file", relative, "required JSON artifact is missing")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _issue(issues, "json_decode", relative, str(exc))
        return None
    if not isinstance(payload, Mapping):
        _issue(issues, "json_object", relative, "JSON payload must be an object")
        return None
    return payload


def _load_npz(
    path: Path, *, relative: str, issues: list[TransferValidationIssue]
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        _issue(issues, "missing_file", relative, "required NPZ artifact is missing")
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        _issue(issues, "npz_decode", relative, str(exc))
        return None


def _is_int(value: Any, *, minimum: int | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return False
    return minimum is None or int(value) >= minimum


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_vector(
    value: Any,
    *,
    shape: tuple[int, ...],
    path: str,
    issues: list[TransferValidationIssue],
    positive: bool = False,
) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _issue(issues, "transfer_vector", path, f"must be a finite array with shape {shape}")
        return None
    if result.shape != shape or not np.all(np.isfinite(result)):
        _issue(issues, "transfer_vector", path, f"must be a finite array with shape {shape}")
        return None
    if positive and np.any(result <= 0.0):
        _issue(issues, "transfer_vector", path, "must contain strictly positive values")
        return None
    return result


def _exact_fields(
    payload: Mapping[str, np.ndarray] | None,
    *,
    expected: Sequence[str],
    relative: str,
    issues: list[TransferValidationIssue],
) -> bool:
    if payload is None:
        return False
    actual = set(payload)
    required = set(expected)
    if actual != required:
        _issue(
            issues,
            "npz_fields",
            relative,
            f"fields must be exactly {list(expected)}; got {sorted(actual)}",
        )
        return False
    return True


def _integer_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    path: str,
    issues: list[TransferValidationIssue],
) -> bool:
    if value.shape != shape or not np.issubdtype(value.dtype, np.integer):
        _issue(issues, "array_shape", path, f"must be an integer array with shape {shape}")
        return False
    return True


def _float_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    path: str,
    issues: list[TransferValidationIssue],
) -> bool:
    if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
        _issue(issues, "array_shape", path, f"must be a floating array with shape {shape}")
        return False
    if not np.all(np.isfinite(value)):
        _issue(issues, "nonfinite", path, "must contain only finite values")
        return False
    return True


def _allclose(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=_FLOAT_RTOL, atol=_FLOAT_ATOL, equal_nan=False))


def _verify_capture_receipt_hash(
    root: Path, issues: list[TransferValidationIssue], checks: dict[str, Any]
) -> str | None:
    receipt_path = root / "capture_receipt.json"
    binding_path = root / "capture_receipt.sha256"
    if not receipt_path.is_file():
        _issue(issues, "missing_file", "capture_receipt.json", "capture receipt is missing")
        return None
    actual = _sha256_file(receipt_path)
    checks["capture_receipt_sha256"] = actual
    if not binding_path.is_file():
        _issue(issues, "capture_receipt_binding", "capture_receipt.sha256", "receipt hash file is missing")
        return actual
    try:
        tokens = binding_path.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeDecodeError) as exc:
        _issue(issues, "capture_receipt_binding", "capture_receipt.sha256", str(exc))
        return actual
    if len(tokens) != 2 or tokens[1] != "capture_receipt.json" or not _is_sha256(tokens[0]):
        _issue(
            issues,
            "capture_receipt_binding",
            "capture_receipt.sha256",
            "must contain '<sha256>  capture_receipt.json'",
        )
    elif tokens[0] != actual:
        _issue(
            issues,
            "capture_receipt_binding",
            "capture_receipt.sha256",
            "does not match capture_receipt.json",
        )
    else:
        checks["capture_receipt_hash_verified"] = True
    return actual


def _verify_artifact_hashes(
    root: Path,
    receipt: Mapping[str, Any],
    issues: list[TransferValidationIssue],
    checks: dict[str, Any],
) -> dict[str, str]:
    artifact_hashes = receipt.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        _issue(issues, "artifact_hashes", "capture_receipt.json.artifact_hashes", "must be an object")
        return {}
    actual_hashes: dict[str, str] = {}
    verified: dict[str, bool] = {}
    for relative in REQUIRED_ARTIFACTS:
        artifact = root / Path(*relative.split("/"))
        binding = artifact_hashes.get(relative)
        if not artifact.is_file():
            _issue(issues, "missing_file", relative, "required artifact is missing")
            verified[relative] = False
            continue
        actual = _sha256_file(artifact)
        actual_hashes[relative] = actual
        if not isinstance(binding, Mapping):
            _issue(issues, "artifact_hash", relative, "artifact is missing from capture receipt")
            verified[relative] = False
            continue
        expected_hash = binding.get("sha256")
        expected_bytes = binding.get("bytes")
        if not _is_sha256(expected_hash):
            _issue(issues, "artifact_hash", relative, "artifact receipt SHA-256 is invalid")
            verified[relative] = False
            continue
        if not _is_int(expected_bytes, minimum=0):
            _issue(issues, "artifact_bytes", relative, "artifact receipt byte count is invalid")
            verified[relative] = False
            continue
        if expected_hash != actual:
            _issue(issues, "artifact_hash", relative, "artifact SHA-256 does not match receipt")
            verified[relative] = False
            continue
        if int(expected_bytes) != artifact.stat().st_size:
            _issue(issues, "artifact_bytes", relative, "artifact byte count does not match receipt")
            verified[relative] = False
            continue
        verified[relative] = True
    checks["required_artifact_hashes"] = verified
    checks["required_artifact_hashes_verified"] = bool(verified) and all(verified.values())
    return actual_hashes


def _validate_capture_contract(
    receipt: Mapping[str, Any], issues: list[TransferValidationIssue], checks: dict[str, Any]
) -> tuple[int, int, float, Mapping[str, Any]] | None:
    if receipt.get("schema") != CAPTURE_SCHEMA:
        _issue(issues, "capture_schema", "capture_receipt.json.schema", "unsupported capture schema")
    if receipt.get("status") != "captured" or receipt.get("ok") is not True:
        _issue(issues, "capture_status", "capture_receipt.json", "capture must have status=captured and ok=true")
    if receipt.get("task_kind") != TASK_KIND:
        _issue(issues, "capture_task", "capture_receipt.json.task_kind", "must be a state-only control transfer smoke")
    if receipt.get("information_profile") != STATE_ONLY_PROFILE:
        _issue(issues, "capture_profile", "capture_receipt.json.information_profile", "must be state_only")

    command = receipt.get("command")
    if not isinstance(command, Mapping):
        _issue(issues, "capture_command", "capture_receipt.json.command", "must be an object")
        return None
    if command.get("control_mode") != CONTROL_MODE:
        _issue(issues, "capture_control_mode", "capture_receipt.json.command.control_mode", "must be sb3_state_only_transfer")
    steps = command.get("steps")
    warmup = command.get("warmup_steps")
    dt_s = _finite_float(command.get("dt_s"))
    if not _is_int(steps, minimum=1) or not _is_int(warmup, minimum=0) or dt_s is None or dt_s <= 0.0:
        _issue(issues, "capture_timing", "capture_receipt.json.command", "steps, warmup_steps, and dt_s are invalid")
        return None
    transfer_command = command.get("sb3_state_only_transfer")
    if not isinstance(transfer_command, Mapping):
        _issue(
            issues,
            "capture_transfer_command",
            "capture_receipt.json.command.sb3_state_only_transfer",
            "must be an object",
        )
    else:
        stride = transfer_command.get("decision_stride_physics_steps")
        if not _is_int(stride, minimum=1):
            _issue(
                issues,
                "capture_transfer_command",
                "capture_receipt.json.command.sb3_state_only_transfer.decision_stride_physics_steps",
                "must be a positive integer",
            )
        bounds = transfer_command.get("world_command_bounds")
        if not isinstance(bounds, Mapping):
            _issue(
                issues,
                "capture_transfer_command",
                "capture_receipt.json.command.sb3_state_only_transfer.world_command_bounds",
                "must be an object",
            )

    if "evaluator_manifest_sha256" in receipt:
        _issue(
            issues,
            "private_evaluator",
            "capture_receipt.json.evaluator_manifest_sha256",
            "development transfer capture must not contain an evaluator manifest",
        )
    task = receipt.get("task")
    if not isinstance(task, Mapping):
        _issue(issues, "capture_task", "capture_receipt.json.task", "must be an object")
    else:
        if task.get("task_kind") != TASK_KIND or task.get("task_variant_id") != TASK_VARIANT_ID:
            _issue(issues, "capture_task", "capture_receipt.json.task", "task contract is not the transfer smoke")
        if task.get("evaluation") != "not_a_search_result" or task.get("private_targets_present") is not False:
            _issue(issues, "private_evaluator", "capture_receipt.json.task", "must contain no search evaluation or private targets")
        if task.get("decision_trace") != TRACE_PATH:
            _issue(issues, "capture_trace_binding", "capture_receipt.json.task.decision_trace", "must bind the transfer trace path")

    claim = receipt.get("claim_boundary")
    if not isinstance(claim, Mapping):
        _issue(issues, "claim_boundary", "capture_receipt.json.claim_boundary", "must be an object")
    else:
        expected_claims = {
            "formal_benchmark_admission": False,
            "development_control_transfer": True,
            "isaac_training": False,
            "physical_training": False,
            "hardware_validated": False,
            "radar_profile_eligible": False,
            "foundation_model_executed": False,
            "semantic_labels_policy_visible": False,
        }
        for key, expected in expected_claims.items():
            if claim.get(key) is not expected:
                _issue(issues, "claim_boundary", f"capture_receipt.json.claim_boundary.{key}", f"must be {expected}")

    modalities = receipt.get("modalities")
    if not isinstance(modalities, Mapping):
        _issue(issues, "capture_modalities", "capture_receipt.json.modalities", "must be an object")
    else:
        for field in ("rtx_radar", "hardware_radar", "real_flight"):
            if modalities.get(field) != "not_captured":
                _issue(issues, "claim_boundary", f"capture_receipt.json.modalities.{field}", "must be not_captured")
        if modalities.get("body_state") != "captured_state_only_policy_input":
            _issue(issues, "capture_profile", "capture_receipt.json.modalities.body_state", "must identify the state-only policy input")

    transfer = receipt.get("state_only_transfer")
    if not isinstance(transfer, Mapping):
        _issue(issues, "capture_transfer_provenance", "capture_receipt.json.state_only_transfer", "must be an object")
        return None
    checks["physics_steps"] = int(steps)
    checks["warmup_physics_steps"] = int(warmup)
    checks["physics_dt_s"] = float(dt_s)
    return int(steps), int(warmup), float(dt_s), transfer


def _validate_scene_contract(
    scene: Mapping[str, Any] | None,
    *,
    trace_sha256: str | None,
    provenance_sha256: str | None,
    issues: list[TransferValidationIssue],
) -> None:
    if scene is None:
        return
    if scene.get("capture_control_mode") != CONTROL_MODE:
        _issue(issues, "scene_transfer_contract", f"{SCENE_PATH}.capture_control_mode", "must be sb3_state_only_transfer")
    if scene.get("control_transfer_task_kind") != TASK_KIND:
        _issue(issues, "scene_transfer_contract", f"{SCENE_PATH}.control_transfer_task_kind", "must be the state-only transfer task")
    if scene.get("control_transfer_state_phase") != "pre_sim_command_state":
        _issue(issues, "scene_transfer_contract", f"{SCENE_PATH}.control_transfer_state_phase", "must be pre_sim_command_state")
    if scene.get("control_transfer_policy_input") != "state_only_8d":
        _issue(issues, "scene_transfer_contract", f"{SCENE_PATH}.control_transfer_policy_input", "must be state_only_8d")
    if trace_sha256 is not None and scene.get("control_transfer_trace_sha256") != trace_sha256:
        _issue(issues, "scene_transfer_binding", f"{SCENE_PATH}.control_transfer_trace_sha256", "does not bind the trace bytes")
    if provenance_sha256 is not None and scene.get("control_transfer_provenance_sha256") != provenance_sha256:
        _issue(issues, "scene_transfer_binding", f"{SCENE_PATH}.control_transfer_provenance_sha256", "does not bind the provenance bytes")


def _validate_transfer_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    receipt_transfer: Mapping[str, Any] | None,
    receipt: Mapping[str, Any],
    actual_hashes: Mapping[str, str],
    issues: list[TransferValidationIssue],
    checks: dict[str, Any],
) -> _TransferContract | None:
    if provenance is None:
        return None
    if provenance.get("schema") != TRACE_SCHEMA:
        _issue(issues, "transfer_provenance_schema", TRACE_PROVENANCE_PATH, "unsupported trace provenance schema")
    expected_scalar_fields = {
        "claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        "formal_benchmark_admission": False,
        "dataset_episode": False,
        "task_kind": TASK_KIND,
        "task_variant_id": TASK_VARIANT_ID,
        "control_mode": CONTROL_MODE,
        "state_phase": "pre_sim_command_state",
        "state_action_state_phase": "pre_sim_command_state",
        "state_action_path": STATE_ACTION_PATH,
        "trace_path": TRACE_PATH,
    }
    for key, expected in expected_scalar_fields.items():
        if provenance.get(key) != expected:
            _issue(issues, "transfer_provenance_contract", f"{TRACE_PROVENANCE_PATH}.{key}", f"must be {expected!r}")
    for field, relative in (("state_action_sha256", STATE_ACTION_PATH), ("trace_sha256", TRACE_PATH)):
        value = provenance.get(field)
        if not _is_sha256(value):
            _issue(issues, "transfer_provenance_hash", f"{TRACE_PROVENANCE_PATH}.{field}", "must be a SHA-256 digest")
        elif relative in actual_hashes and value != actual_hashes[relative]:
            _issue(issues, "transfer_provenance_hash", f"{TRACE_PROVENANCE_PATH}.{field}", "does not bind the artifact bytes")
    if receipt_transfer is not None and provenance.get("transfer") != receipt_transfer:
        _issue(issues, "transfer_receipt_binding", f"{TRACE_PROVENANCE_PATH}.transfer", "does not equal capture receipt state_only_transfer")

    transfer = provenance.get("transfer")
    if not isinstance(transfer, Mapping):
        _issue(issues, "transfer_contract", f"{TRACE_PROVENANCE_PATH}.transfer", "must be an object")
        return None
    expected_transfer_fields = {
        "schema": TRANSFER_SCHEMA,
        "claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        "formal_benchmark_admission": False,
        "physical_training": False,
        "isaac_training": False,
        "information_profile": STATE_ONLY_PROFILE,
        "policy_input_fields": list(STATE_FIELDS),
        "excluded_policy_inputs": list(EXCLUDED_POLICY_INPUTS),
        "action_source": TRANSFER_SOURCE,
    }
    for key, expected in expected_transfer_fields.items():
        if transfer.get(key) != expected:
            _issue(issues, "transfer_contract", f"{TRACE_PROVENANCE_PATH}.transfer.{key}", f"must be {expected!r}")

    cadence = transfer.get("decision_cadence_physics_steps")
    if not _is_int(cadence, minimum=1):
        _issue(issues, "transfer_cadence", f"{TRACE_PROVENANCE_PATH}.transfer.decision_cadence_physics_steps", "must be a positive integer")
        return None
    command = receipt.get("command")
    capture_transfer = command.get("sb3_state_only_transfer") if isinstance(command, Mapping) else None
    if isinstance(capture_transfer, Mapping) and capture_transfer.get("decision_stride_physics_steps") != int(cadence):
        _issue(issues, "transfer_cadence", f"{TRACE_PROVENANCE_PATH}.transfer.decision_cadence_physics_steps", "does not match capture command cadence")

    mean = _finite_vector(
        transfer.get("observation_mean"),
        shape=(8,),
        path=f"{TRACE_PROVENANCE_PATH}.transfer.observation_mean",
        issues=issues,
    )
    standard_deviation = _finite_vector(
        transfer.get("observation_std"),
        shape=(8,),
        path=f"{TRACE_PROVENANCE_PATH}.transfer.observation_std",
        issues=issues,
        positive=True,
    )
    action_scale = _finite_vector(
        transfer.get("action_scale"),
        shape=(4,),
        path=f"{TRACE_PROVENANCE_PATH}.transfer.action_scale",
        issues=issues,
        positive=True,
    )

    expected_transform = CityLiteRouteAnchorTransform.from_public_routes()
    coordinate_transform = transfer.get("coordinate_transform")
    if not isinstance(coordinate_transform, Mapping) or dict(coordinate_transform) != expected_transform.provenance():
        _issue(
            issues,
            "transfer_coordinate_transform",
            f"{TRACE_PROVENANCE_PATH}.transfer.coordinate_transform",
            "must exactly match the public City-Lite route-anchor/heading transform",
        )
    elif coordinate_transform.get("coordinate_contract") != CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1:
        _issue(
            issues,
            "transfer_coordinate_transform",
            f"{TRACE_PROVENANCE_PATH}.transfer.coordinate_transform.coordinate_contract",
            "has an unsupported coordinate contract",
        )

    bounds_payload = transfer.get("world_command_bounds")
    bounds_values: list[float] = []
    if not isinstance(bounds_payload, Mapping):
        _issue(issues, "transfer_bounds", f"{TRACE_PROVENANCE_PATH}.transfer.world_command_bounds", "must be an object")
    else:
        for key in ("max_horizontal_speed_mps", "max_vertical_speed_mps", "max_yaw_rate_rad_s"):
            number = _finite_float(bounds_payload.get(key))
            if number is None or number <= 0.0:
                _issue(issues, "transfer_bounds", f"{TRACE_PROVENANCE_PATH}.transfer.world_command_bounds.{key}", "must be finite and positive")
            else:
                bounds_values.append(number)
        capture_bounds = capture_transfer.get("world_command_bounds") if isinstance(capture_transfer, Mapping) else None
        if isinstance(capture_bounds, Mapping):
            for key in ("max_horizontal_speed_mps", "max_vertical_speed_mps", "max_yaw_rate_rad_s"):
                left = _finite_float(bounds_payload.get(key))
                right = _finite_float(capture_bounds.get(key))
                if left is None or right is None or not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12):
                    _issue(issues, "transfer_bounds", f"{TRACE_PROVENANCE_PATH}.transfer.world_command_bounds.{key}", "does not match capture command bounds")
    policy = transfer.get("policy")
    if not isinstance(policy, Mapping):
        _issue(issues, "transfer_policy", f"{TRACE_PROVENANCE_PATH}.transfer.policy", "must be an object")
    else:
        expected_policy = {
            "method_id": "sb3_checkpoint_policy",
            "implementation_kind": "trained_sb3_pilot_checkpoint",
            "external_dependency": "stable_baselines3",
            "parameter_sharing": "independent_shared_policy_per_agent",
        }
        for key, expected in expected_policy.items():
            if policy.get(key) != expected:
                _issue(issues, "transfer_policy", f"{TRACE_PROVENANCE_PATH}.transfer.policy.{key}", f"must be {expected!r}")
        if policy.get("algorithm") not in {"ppo", "sac"}:
            _issue(issues, "transfer_policy", f"{TRACE_PROVENANCE_PATH}.transfer.policy.algorithm", "must be ppo or sac")
        for key in ("checkpoint_sha256", "adapter_metadata_sha256"):
            if not _is_sha256(policy.get(key)):
                _issue(issues, "transfer_policy", f"{TRACE_PROVENANCE_PATH}.transfer.policy.{key}", "must be a SHA-256 digest")

    trace_count = provenance.get("trace_decision_count")
    if not _is_int(trace_count, minimum=1):
        _issue(issues, "trace_decision_count", f"{TRACE_PROVENANCE_PATH}.trace_decision_count", "must be a positive integer")
    if provenance.get("trace_fields") != list(TRACE_FIELDS):
        _issue(issues, "trace_fields", f"{TRACE_PROVENANCE_PATH}.trace_fields", "must preserve the exact ordered trace ABI")

    if mean is None or standard_deviation is None or action_scale is None or len(bounds_values) != 3:
        return None
    try:
        bounds = WorldCommandBounds(*bounds_values)
    except StateOnlyTransferError as exc:
        _issue(issues, "transfer_bounds", f"{TRACE_PROVENANCE_PATH}.transfer.world_command_bounds", str(exc))
        return None
    checks["transfer_cadence_physics_steps"] = int(cadence)
    checks["transfer_trace_declared_decision_count"] = int(trace_count) if _is_int(trace_count, minimum=1) else None
    return _TransferContract(
        cadence_steps=int(cadence),
        observation_mean=mean,
        observation_std=standard_deviation,
        action_scale=action_scale,
        transform=expected_transform,
        bounds=bounds,
    )


def _validate_state_action(
    state: Mapping[str, np.ndarray] | None,
    *,
    steps: int,
    warmup_steps: int,
    dt_s: float,
    issues: list[TransferValidationIssue],
    checks: dict[str, Any],
) -> bool:
    if not _exact_fields(
        state,
        expected=STATE_ACTION_FIELDS,
        relative=STATE_ACTION_PATH,
        issues=issues,
    ):
        return False
    assert state is not None
    time_valid = True
    for field in ("command_time_ns", "effective_time_ns"):
        time_valid &= _integer_array(
            state[field],
            shape=(steps,),
            path=f"{STATE_ACTION_PATH}.{field}",
            issues=issues,
        )
    expected_command_times = np.asarray(
        [physics_time_ns(warmup_steps + step, dt_s) for step in range(steps)], dtype=np.int64
    )
    expected_effective_times = np.asarray(
        [physics_time_ns(warmup_steps + step + 1, dt_s) for step in range(steps)], dtype=np.int64
    )
    if time_valid and not np.array_equal(state["command_time_ns"], expected_command_times):
        _issue(issues, "state_action_timing", f"{STATE_ACTION_PATH}.command_time_ns", "does not follow the physics clock")
    if time_valid and not np.array_equal(state["effective_time_ns"], expected_effective_times):
        _issue(issues, "state_action_timing", f"{STATE_ACTION_PATH}.effective_time_ns", "does not follow the physics clock")

    valid = time_valid
    shapes = {
        "root_pos_w_m": (steps, AGENT_COUNT, 3),
        "root_quat_wxyz": (steps, AGENT_COUNT, 4),
        "root_lin_vel_w_mps": (steps, AGENT_COUNT, 3),
        "root_ang_vel_b_radps": (steps, AGENT_COUNT, 3),
        "desired_pos_w_m": (steps, AGENT_COUNT, 3),
        "desired_vel_w_mps": (steps, AGENT_COUNT, 3),
        "target_thrust_n": (steps, AGENT_COUNT, 4),
        "applied_thrust_n": (steps, AGENT_COUNT, 4),
        "pre_command_root_pos_w_m": (steps, AGENT_COUNT, 3),
        "pre_command_root_quat_wxyz": (steps, AGENT_COUNT, 4),
        "pre_command_root_lin_vel_w_mps": (steps, AGENT_COUNT, 3),
        "pre_command_root_ang_vel_b_radps": (steps, AGENT_COUNT, 3),
        "emitted_world_velocity_yaw_command": (steps, AGENT_COUNT, 4),
    }
    for field, shape in shapes.items():
        valid &= _float_array(state[field], shape=shape, path=f"{STATE_ACTION_PATH}.{field}", issues=issues)
    checks["state_action_physics_steps"] = int(steps)
    checks["state_action_timing_verified"] = time_valid and (
        np.array_equal(state["command_time_ns"], expected_command_times)
        and np.array_equal(state["effective_time_ns"], expected_effective_times)
    )
    return valid


def _validate_trace(
    trace: Mapping[str, np.ndarray] | None,
    *,
    provenance: Mapping[str, Any] | None,
    contract: _TransferContract | None,
    state: Mapping[str, np.ndarray] | None,
    state_valid: bool,
    steps: int,
    warmup_steps: int,
    dt_s: float,
    issues: list[TransferValidationIssue],
    checks: dict[str, Any],
) -> bool:
    if not _exact_fields(trace, expected=TRACE_FIELDS, relative=TRACE_PATH, issues=issues):
        return False
    assert trace is not None
    rollout_steps = trace["rollout_physics_step"]
    if rollout_steps.ndim != 1:
        _issue(
            issues,
            "array_shape",
            f"{TRACE_PATH}.rollout_physics_step",
            "must be a one-dimensional integer decision sequence",
        )
        return False
    decision_count = int(rollout_steps.shape[0])
    expected_count = len(range(0, steps, contract.cadence_steps)) if contract is not None else None
    integer_valid = True
    for field in ("rollout_physics_step", "command_time_ns", "effective_time_ns", "decision_index"):
        integer_valid &= _integer_array(trace[field], shape=(decision_count,), path=f"{TRACE_PATH}.{field}", issues=issues)
    floating_valid = True
    for field in ("physical_state_8d", "pilot_state_8d", "normalized_observation_8d"):
        floating_valid &= _float_array(
            trace[field],
            shape=(decision_count, AGENT_COUNT, 8),
            path=f"{TRACE_PATH}.{field}",
            issues=issues,
        )
    for field in (
        "raw_action",
        "normalized_action",
        "local_velocity_yaw_command",
        "prebound_world_velocity_yaw_command",
        "emitted_world_velocity_yaw_command",
    ):
        floating_valid &= _float_array(
            trace[field],
            shape=(decision_count, AGENT_COUNT, 4),
            path=f"{TRACE_PATH}.{field}",
            issues=issues,
        )
    floating_valid &= _float_array(
        trace["altitude_reference_w_m"],
        shape=(decision_count, AGENT_COUNT),
        path=f"{TRACE_PATH}.altitude_reference_w_m",
        issues=issues,
    )
    if decision_count < 1:
        _issue(issues, "trace_decision_count", TRACE_PATH, "must contain at least one decision")
        return False
    if provenance is not None and _is_int(provenance.get("trace_decision_count"), minimum=1):
        if int(provenance["trace_decision_count"]) != decision_count:
            _issue(issues, "trace_decision_count", TRACE_PATH, "does not match provenance trace_decision_count")
    if expected_count is not None and decision_count != expected_count:
        _issue(issues, "trace_decision_count", TRACE_PATH, "does not match the configured fixed cadence")
    if contract is None or not integer_valid or not floating_valid:
        return False

    expected_steps = np.asarray(range(0, steps, contract.cadence_steps), dtype=np.int64)
    expected_indexes = expected_steps // contract.cadence_steps
    expected_command_times = np.asarray(
        [physics_time_ns(warmup_steps + int(step), dt_s) for step in expected_steps], dtype=np.int64
    )
    expected_effective_times = np.asarray(
        [physics_time_ns(warmup_steps + int(step) + 1, dt_s) for step in expected_steps], dtype=np.int64
    )
    if decision_count == len(expected_steps):
        if not np.array_equal(trace["rollout_physics_step"], expected_steps):
            _issue(issues, "trace_cadence", f"{TRACE_PATH}.rollout_physics_step", "does not follow the fixed physics-step cadence")
        if not np.array_equal(trace["decision_index"], expected_indexes):
            _issue(issues, "trace_cadence", f"{TRACE_PATH}.decision_index", "does not equal rollout_physics_step / cadence")
        if not np.array_equal(trace["command_time_ns"], expected_command_times):
            _issue(issues, "trace_timing", f"{TRACE_PATH}.command_time_ns", "does not follow the physics clock")
        if not np.array_equal(trace["effective_time_ns"], expected_effective_times):
            _issue(issues, "trace_timing", f"{TRACE_PATH}.effective_time_ns", "does not follow the physics clock")

    state_bound = state_valid and state is not None and decision_count == len(expected_steps)
    if state_bound:
        assert state is not None
        selected = expected_steps.astype(np.intp, copy=False)
        if not np.array_equal(trace["command_time_ns"], state["command_time_ns"][selected]):
            _issue(issues, "trace_state_timing_binding", f"{TRACE_PATH}.command_time_ns", "does not bind selected command-pre state rows")
        if not np.array_equal(trace["effective_time_ns"], state["effective_time_ns"][selected]):
            _issue(issues, "trace_state_timing_binding", f"{TRACE_PATH}.effective_time_ns", "does not bind selected command-pre state rows")
        expected_physical = np.empty((decision_count, AGENT_COUNT, 8), dtype=np.float64)
        expected_pilot = np.empty_like(expected_physical)
        for row, state_index in enumerate(selected):
            try:
                physical = derive_physical_state_8d(
                    state["pre_command_root_pos_w_m"][state_index],
                    state["pre_command_root_lin_vel_w_mps"][state_index],
                    state["pre_command_root_quat_wxyz"][state_index],
                    state["pre_command_root_ang_vel_b_radps"][state_index],
                    agent_ids=range(AGENT_COUNT),
                )
                expected_physical[row] = physical.values
                expected_pilot[row] = contract.transform.physical_to_pilot(physical).values
            except (StateOnlyTransferError, ValueError) as exc:
                _issue(issues, "trace_state_binding", TRACE_PATH, f"cannot derive public physical state: {exc}")
                state_bound = False
                break
        if state_bound and not _allclose(trace["physical_state_8d"], expected_physical):
            _issue(issues, "trace_physical_state", f"{TRACE_PATH}.physical_state_8d", "does not equal the selected command-pre Isaac state")
        if state_bound and not _allclose(trace["pilot_state_8d"], expected_pilot):
            _issue(issues, "trace_pilot_state", f"{TRACE_PATH}.pilot_state_8d", "does not equal the public City-Lite coordinate transform")
        if state_bound:
            expected_normalized_observation = (
                expected_pilot - contract.observation_mean[None, None, :]
            ) / contract.observation_std[None, None, :]
            if not _allclose(trace["normalized_observation_8d"], expected_normalized_observation):
                _issue(issues, "trace_normalized_observation", f"{TRACE_PATH}.normalized_observation_8d", "does not replay the persisted normalization ABI")
            expected_altitude = np.broadcast_to(
                state["pre_command_root_pos_w_m"][selected[0], :, 2],
                (decision_count, AGENT_COUNT),
            )
            if not _allclose(trace["altitude_reference_w_m"], expected_altitude):
                _issue(issues, "trace_altitude_reference", f"{TRACE_PATH}.altitude_reference_w_m", "must stay fixed at the first command-pre altitude")

    expected_normalized_action = np.clip(trace["raw_action"], -1.0, 1.0)
    if not _allclose(trace["normalized_action"], expected_normalized_action):
        _issue(issues, "trace_normalized_action", f"{TRACE_PATH}.normalized_action", "does not equal clipped raw action")
    expected_local = expected_normalized_action * contract.action_scale[None, None, :]
    if not _allclose(trace["local_velocity_yaw_command"], expected_local):
        _issue(issues, "trace_local_command", f"{TRACE_PATH}.local_velocity_yaw_command", "does not equal normalized action times action scale")

    expected_prebound = np.empty((decision_count, AGENT_COUNT, 4), dtype=np.float64)
    expected_emitted = np.empty_like(expected_prebound)
    for row in range(decision_count):
        try:
            world_velocity = contract.transform.pilot_velocity_to_world(
                expected_local[row, :, :3], agent_ids=range(AGENT_COUNT)
            )
            bounded_velocity, bounded_yaw = contract.bounds.apply(
                world_velocity, expected_local[row, :, 3]
            )
        except StateOnlyTransferError as exc:
            _issue(issues, "trace_command_replay", TRACE_PATH, str(exc))
            return False
        expected_prebound[row] = np.concatenate((world_velocity, expected_local[row, :, 3:4]), axis=1)
        expected_emitted[row] = np.concatenate((bounded_velocity, bounded_yaw[:, None]), axis=1)
    if not _allclose(trace["prebound_world_velocity_yaw_command"], expected_prebound):
        _issue(issues, "trace_prebound_command", f"{TRACE_PATH}.prebound_world_velocity_yaw_command", "does not replay pilot-to-world command rotation")
    if not _allclose(trace["emitted_world_velocity_yaw_command"], expected_emitted):
        _issue(issues, "trace_emitted_command", f"{TRACE_PATH}.emitted_world_velocity_yaw_command", "does not replay world command bounds")

    if state_bound:
        assert state is not None
        held_commands = trace["emitted_world_velocity_yaw_command"][
            np.arange(steps, dtype=np.intp) // contract.cadence_steps
        ]
        if not _allclose(state["emitted_world_velocity_yaw_command"], held_commands):
            _issue(issues, "state_action_command_hold", f"{STATE_ACTION_PATH}.emitted_world_velocity_yaw_command", "does not hold each decision command until the next cadence tick")
    checks["trace_decision_count"] = int(decision_count)
    checks["trace_replay_attempted"] = True
    return True


def _validate_runtime_safety(
    runtime: Mapping[str, np.ndarray] | None,
    *,
    receipt: Mapping[str, Any],
    runtime_sha256: str | None,
    state: Mapping[str, np.ndarray] | None,
    state_valid: bool,
    steps: int,
    warmup_steps: int,
    dt_s: float,
    issues: list[TransferValidationIssue],
    checks: dict[str, Any],
) -> bool:
    if not _exact_fields(runtime, expected=RUNTIME_SAFETY_FIELDS, relative=RUNTIME_SAFETY_PATH, issues=issues):
        return False
    assert runtime is not None
    frame_count = 1 + warmup_steps + steps
    integer_valid = True
    for field in ("physics_step", "physics_time_ns", "phase_code", "frame_outcome_code"):
        integer_valid &= _integer_array(runtime[field], shape=(frame_count,), path=f"{RUNTIME_SAFETY_PATH}.{field}", issues=issues)
    float_valid = True
    float_valid &= _float_array(runtime["root_pos_w_m"], shape=(frame_count, AGENT_COUNT, 3), path=f"{RUNTIME_SAFETY_PATH}.root_pos_w_m", issues=issues)
    float_valid &= _float_array(runtime["net_contact_forces_w_n"], shape=(frame_count, AGENT_COUNT, 1, 3), path=f"{RUNTIME_SAFETY_PATH}.net_contact_forces_w_n", issues=issues)
    float_valid &= _float_array(runtime["max_contact_force_n"], shape=(frame_count,), path=f"{RUNTIME_SAFETY_PATH}.max_contact_force_n", issues=issues)
    if not integer_valid or not float_valid:
        return False
    expected_steps = np.arange(frame_count, dtype=np.int64)
    expected_times = np.asarray([physics_time_ns(int(step), dt_s) for step in expected_steps], dtype=np.int64)
    expected_phase = np.asarray(
        [RUNTIME_SAFETY_PHASE_CODES["post_reset"]]
        + [RUNTIME_SAFETY_PHASE_CODES["warmup"]] * warmup_steps
        + [RUNTIME_SAFETY_PHASE_CODES["rollout"]] * steps,
        dtype=np.int64,
    )
    if not np.array_equal(runtime["physics_step"], expected_steps):
        _issue(issues, "runtime_safety_timing", f"{RUNTIME_SAFETY_PATH}.physics_step", "must cover every post-reset, warmup, and rollout frame")
    if not np.array_equal(runtime["physics_time_ns"], expected_times):
        _issue(issues, "runtime_safety_timing", f"{RUNTIME_SAFETY_PATH}.physics_time_ns", "does not follow the physics clock")
    if not np.array_equal(runtime["phase_code"], expected_phase):
        _issue(issues, "runtime_safety_phase", f"{RUNTIME_SAFETY_PATH}.phase_code", "does not match the expected post-reset/warmup/rollout timeline")
    if not np.array_equal(
        runtime["frame_outcome_code"],
        np.full((frame_count,), RUNTIME_SAFETY_FRAME_OUTCOME_CODES["passed"], dtype=runtime["frame_outcome_code"].dtype),
    ):
        _issue(issues, "runtime_safety_outcome", f"{RUNTIME_SAFETY_PATH}.frame_outcome_code", "a passing capture must contain only passed frames")
    if state_valid and state is not None and _allclose(runtime["root_pos_w_m"][warmup_steps + 1 :], state["root_pos_w_m"]):
        checks["runtime_safety_state_position_binding_verified"] = True
    elif state_valid and state is not None:
        _issue(issues, "runtime_safety_state_binding", RUNTIME_SAFETY_PATH, "rollout safety positions do not equal state_action root positions")

    guard = receipt.get("runtime_safety_guard")
    if not isinstance(guard, Mapping):
        _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard", "must be an object")
    else:
        if guard.get("schema") != RUNTIME_SAFETY_SCHEMA or guard.get("enabled") is not True or guard.get("fail_closed") is not True or guard.get("status") != "passed":
            _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard", "must be a passed fail-closed runtime guard")
        evidence = guard.get("evidence")
        if not isinstance(evidence, Mapping):
            _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard.evidence", "must be an object")
        else:
            if evidence.get("schema") != RUNTIME_SAFETY_TRACE_SCHEMA or evidence.get("path") != RUNTIME_SAFETY_PATH:
                _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard.evidence", "must bind the runtime safety trace ABI")
            if runtime_sha256 is not None and evidence.get("sha256") != runtime_sha256:
                _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard.evidence.sha256", "does not bind runtime safety bytes")
            if evidence.get("physics_frame_count") != frame_count:
                _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard.evidence.physics_frame_count", "does not match trace frame count")
        guard_checks = guard.get("checks")
        if not isinstance(guard_checks, Mapping):
            _issue(issues, "runtime_safety_guard", "capture_receipt.json.runtime_safety_guard.checks", "must be an object")
        else:
            expected_counts = {
                "warmup_physics_steps_checked": warmup_steps,
                "rollout_physics_steps_checked": steps,
                "contact_samples_checked": frame_count,
                "contact_abort_count": 0,
            }
            for key, expected in expected_counts.items():
                if guard_checks.get(key) != expected:
                    _issue(issues, "runtime_safety_guard", f"capture_receipt.json.runtime_safety_guard.checks.{key}", f"must equal {expected}")
    checks["runtime_safety_frame_count"] = int(frame_count)
    checks["runtime_safety_timeline_verified"] = (
        np.array_equal(runtime["physics_step"], expected_steps)
        and np.array_equal(runtime["physics_time_ns"], expected_times)
        and np.array_equal(runtime["phase_code"], expected_phase)
    )
    return True


def validate_isaac_state_only_transfer(capture_root: Path) -> IsaacTransferValidationReport:
    """Validate a hash-bound development-only state-only transfer capture.

    The function returns every discovered contract violation instead of raising
    for malformed evidence.  Callers may write a public receipt only when the
    returned report is valid.
    """

    root = capture_root.expanduser().resolve()
    issues: list[TransferValidationIssue] = []
    checks: dict[str, Any] = {
        "development_only": True,
        "formal_benchmark_admission": False,
        "dataset_episode": False,
        "validator_runtime": "pure_python_numpy",
    }
    if not root.is_dir():
        _issue(issues, "capture_root", str(root), "capture root must be a directory")
        return IsaacTransferValidationReport(root, None, checks, tuple(issues))

    receipt_sha256 = _verify_capture_receipt_hash(root, issues, checks)
    receipt = _read_json(root / "capture_receipt.json", relative="capture_receipt.json", issues=issues)
    if receipt is None:
        return IsaacTransferValidationReport(root, receipt_sha256, checks, tuple(issues))
    actual_hashes = _verify_artifact_hashes(root, receipt, issues, checks)
    timing = _validate_capture_contract(receipt, issues, checks)

    scene = _read_json(root / SCENE_PATH, relative=SCENE_PATH, issues=issues)
    provenance = _read_json(
        root / TRACE_PROVENANCE_PATH,
        relative=TRACE_PROVENANCE_PATH,
        issues=issues,
    )
    _validate_scene_contract(
        scene,
        trace_sha256=actual_hashes.get(TRACE_PATH),
        provenance_sha256=actual_hashes.get(TRACE_PROVENANCE_PATH),
        issues=issues,
    )
    task = receipt.get("task")
    if isinstance(task, Mapping) and TRACE_PATH in actual_hashes:
        if task.get("decision_trace_sha256") != actual_hashes[TRACE_PATH]:
            _issue(
                issues,
                "capture_trace_binding",
                "capture_receipt.json.task.decision_trace_sha256",
                "does not bind the transfer trace bytes",
            )

    state = _load_npz(root / STATE_ACTION_PATH, relative=STATE_ACTION_PATH, issues=issues)
    trace = _load_npz(root / TRACE_PATH, relative=TRACE_PATH, issues=issues)
    runtime = _load_npz(root / RUNTIME_SAFETY_PATH, relative=RUNTIME_SAFETY_PATH, issues=issues)
    if timing is not None:
        steps, warmup_steps, dt_s, receipt_transfer = timing
        contract = _validate_transfer_provenance(
            provenance,
            receipt_transfer=receipt_transfer,
            receipt=receipt,
            actual_hashes=actual_hashes,
            issues=issues,
            checks=checks,
        )
        state_valid = _validate_state_action(
            state,
            steps=steps,
            warmup_steps=warmup_steps,
            dt_s=dt_s,
            issues=issues,
            checks=checks,
        )
        _validate_trace(
            trace,
            provenance=provenance,
            contract=contract,
            state=state,
            state_valid=state_valid,
            steps=steps,
            warmup_steps=warmup_steps,
            dt_s=dt_s,
            issues=issues,
            checks=checks,
        )
        _validate_runtime_safety(
            runtime,
            receipt=receipt,
            runtime_sha256=actual_hashes.get(RUNTIME_SAFETY_PATH),
            state=state,
            state_valid=state_valid,
            steps=steps,
            warmup_steps=warmup_steps,
            dt_s=dt_s,
            issues=issues,
            checks=checks,
        )
    else:
        checks["state_action_timing_verified"] = False
        checks["trace_replay_attempted"] = False
        checks["runtime_safety_timeline_verified"] = False

    checks["issue_count"] = len(issues)
    return IsaacTransferValidationReport(root, receipt_sha256, checks, tuple(issues))


def _validator_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def write_transfer_validation_receipt(
    report: IsaacTransferValidationReport, destination: Path
) -> Path:
    """Atomically write the public development-only receipt for a valid report."""

    if not report.valid or report.receipt_sha256 is None:
        raise RuntimeError("cannot write a passing validation receipt for an invalid capture")
    payload = {
        "schema": TRANSFER_VALIDATION_SCHEMA,
        "status": "passed",
        "formal_benchmark_admission": False,
        "development_only": True,
        "dataset_episode": False,
        "capture_receipt_sha256": report.receipt_sha256,
        "validator_id": "rivermark-independent-isaac-state-transfer-validator-v1",
        "validator_source_sha256": _validator_sha256(),
        "checks": dict(report.checks),
        "issues": [],
    }
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


# Keep the conventional name used by the normal Isaac validator available to
# callers without making the two validation contracts interchangeable.
write_validation_receipt = write_transfer_validation_receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.capture_root.expanduser().resolve()
    report = validate_isaac_state_only_transfer(root)
    payload: dict[str, Any] = {
        "valid": report.valid,
        "capture_receipt_sha256": report.receipt_sha256,
        "checks": dict(report.checks),
        "issues": [asdict(issue) for issue in report.issues],
    }
    if report.valid:
        destination = (args.output or root / "independent_validation.json").expanduser().resolve()
        payload["validation_receipt"] = str(write_transfer_validation_receipt(report, destination))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
