"""Dependency-light validator for the field-level observation ABI.

The episode manifest binds files and provenance.  This contract binds the
meaning of values inside those files: shape, dtype, units, timing, missing
values, compression, and calibration.  It is intentionally separate from the
pilot manifest so an old pilot cannot be made to look standards-compliant by
adding a label after the fact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OBSERVATION_ABI_SCHEMA = "org.rivermark.benchmark.observation-abi.v1"
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_DIMENSION = re.compile(r"^(?:[0-9]+|[a-z][a-z0-9_{}-]{0,63})$")
_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "uint8",
        "uint16",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "string",
    }
)
_PARTITIONS = frozenset({"policy_visible", "learning_labels"})
_MISSING_POLICIES = frozenset({"not_applicable", "drop_sample", "nan", "sentinel", "mask_field"})
_COMPRESSIONS = frozenset({"none", "npz_deflate", "gzip", "zstd", "parquet", "zarr", "mp4_h264"})
_TIME_SEMANTICS = frozenset({"sensor_sample", "command_before_step", "state_after_step", "event_time"})
_CALIBRATION_STATUS = frozenset({"recorded", "unavailable"})
_FIDELITY_LEVELS = frozenset({"simulator_consistent", "noise_modeled", "hardware_calibrated"})


@dataclass(frozen=True)
class AbiIssue:
    code: str
    path: str
    message: str


class AbiError(ValueError):
    """Raised when an observation ABI cannot be read or validated."""


@dataclass(frozen=True)
class AbiCompatibilityReport:
    """Machine-readable compatibility result for a producer and reader ABI.

    ``development_readable`` is deliberately weaker than
    ``formal_admissible``: an ABI 1.0 producer can be read by a newer
    development reader, but it cannot become a formal episode until the
    producer has emitted the ABI 1.1 fidelity evidence.
    """

    producer_version: str | None
    reader_version: str | None
    development_readable: bool
    formal_admissible: bool
    issues: tuple[str, ...]


def _issue(issues: list[AbiIssue], code: str, path: str, message: str) -> None:
    issues.append(AbiIssue(code, path, message))


def _required(value: Mapping[str, Any], names: tuple[str, ...], path: str, issues: list[AbiIssue]) -> None:
    for name in names:
        if name not in value:
            _issue(issues, "required", f"{path}.{name}", "required field is missing")


def _unknown(value: Mapping[str, Any], allowed: frozenset[str], path: str, issues: list[AbiIssue]) -> None:
    for name in sorted(set(value) - allowed):
        _issue(issues, "unknown_field", f"{path}.{name}", "field is not part of observation-abi v1")


def _nonempty_string(value: Any, path: str, issues: list[AbiIssue], *, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _issue(issues, "string", path, f"must be a non-empty string of at most {maximum} characters")


def _validate_coordinate_frames(value: Any, issues: list[AbiIssue]) -> None:
    path = "$.coordinate_frames"
    if not isinstance(value, Mapping):
        _issue(issues, "type", path, "must be an object")
        return
    allowed = frozenset(
        {
            "handedness",
            "world_up_axis",
            "world_frame_convention",
            "body_frame_convention",
            "camera_optical_frame_convention",
            "length_unit",
            "angle_unit",
            "quaternion_order",
            "transform_notation",
        }
    )
    _unknown(value, allowed, path, issues)
    _required(value, tuple(sorted(allowed)), path, issues)
    expected = {
        "handedness": "right",
        "world_up_axis": "+z",
        "world_frame_convention": "x_east_y_north_z_up",
        "body_frame_convention": "flu",
        "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "m",
        "angle_unit": "rad",
        "quaternion_order": "wxyz",
        "transform_notation": "T_parent_child",
    }
    for key, expected_value in expected.items():
        if key in value and value[key] != expected_value:
            _issue(issues, "coordinate_convention", f"{path}.{key}", f"must be {expected_value!r}")


def _validate_action_timing(value: Any, issues: list[AbiIssue]) -> None:
    path = "$.action_timing"
    if not isinstance(value, Mapping):
        _issue(issues, "type", path, "must be an object")
        return
    allowed = frozenset({"command_write", "simulation_step", "state_update", "sensor_read", "storage"})
    _unknown(value, allowed, path, issues)
    _required(value, tuple(sorted(allowed)), path, issues)
    expected = {
        "command_write": "before_simulation_step",
        "simulation_step": "after_command_write",
        "state_update": "after_simulation_step",
        "sensor_read": "after_state_update",
        "storage": "after_sensor_read",
    }
    for key, expected_value in expected.items():
        if key in value and value[key] != expected_value:
            _issue(issues, "action_causality", f"{path}.{key}", f"must be {expected_value!r}")


def _validate_calibration(value: Any, issues: list[AbiIssue]) -> None:
    path = "$.calibration"
    if not isinstance(value, Mapping):
        _issue(issues, "type", path, "must be an object")
        return
    allowed = frozenset({"camera", "lidar", "imu"})
    _unknown(value, allowed, path, issues)
    _required(value, tuple(sorted(allowed)), path, issues)
    for sensor in sorted(allowed):
        sensor_path = f"{path}.{sensor}"
        record = value.get(sensor)
        if not isinstance(record, Mapping):
            _issue(issues, "type", sensor_path, "must be an object")
            continue
        record_allowed = frozenset(
            {
                "status",
                "source",
                "frame_id",
                "intrinsics",
                "extrinsics",
                "distortion_model",
                "distortion_coefficients",
                "max_range_m",
            }
        )
        _unknown(record, record_allowed, sensor_path, issues)
        _required(record, ("status", "source"), sensor_path, issues)
        if record.get("status") not in _CALIBRATION_STATUS:
            _issue(issues, "calibration_status", f"{sensor_path}.status", "must be recorded or unavailable")
        _nonempty_string(record.get("source"), f"{sensor_path}.source", issues)
        if record.get("status") == "recorded":
            if sensor == "camera":
                _validate_camera_calibration(record, sensor_path, issues)
            else:
                _nonempty_string(record.get("frame_id"), f"{sensor_path}.frame_id", issues)
        elif any(
            key in record
            for key in ("frame_id", "intrinsics", "extrinsics", "distortion_model", "distortion_coefficients", "max_range_m")
        ):
            _issue(
                issues,
                "calibration_unavailable_payload",
                sensor_path,
                "unavailable calibration cannot carry usable sensor parameters",
            )
        if "max_range_m" in record:
            value_m = record["max_range_m"]
            if not isinstance(value_m, (int, float)) or isinstance(value_m, bool) or not math.isfinite(float(value_m)) or value_m <= 0:
                _issue(issues, "calibration_range", f"{sensor_path}.max_range_m", "must be a positive finite number")


def _validate_camera_calibration(record: Mapping[str, Any], path: str, issues: list[AbiIssue]) -> None:
    intrinsics = record.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        _issue(issues, "camera_intrinsics", f"{path}.intrinsics", "recorded camera calibration requires intrinsics")
    else:
        allowed = frozenset({"model", "width_px", "height_px", "fx_px", "fy_px", "cx_px", "cy_px"})
        _unknown(intrinsics, allowed, f"{path}.intrinsics", issues)
        _required(intrinsics, tuple(sorted(allowed)), f"{path}.intrinsics", issues)
        if intrinsics.get("model") not in {"pinhole", "fisheye"}:
            _issue(issues, "camera_model", f"{path}.intrinsics.model", "must be pinhole or fisheye")
        for key in ("width_px", "height_px"):
            if not isinstance(intrinsics.get(key), int) or isinstance(intrinsics.get(key), bool) or intrinsics.get(key) <= 0:
                _issue(issues, "camera_resolution", f"{path}.intrinsics.{key}", "must be a positive integer")
        for key in ("fx_px", "fy_px", "cx_px", "cy_px"):
            value = intrinsics.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                _issue(issues, "camera_intrinsics", f"{path}.intrinsics.{key}", "must be finite")
    if record.get("extrinsics") != {"formula": "T_world_camera = T_world_body * T_body_camera", "quaternion_order": "wxyz"}:
        _issue(
            issues,
            "camera_extrinsics",
            f"{path}.extrinsics",
            "must bind T_world_camera = T_world_body * T_body_camera with wxyz quaternions",
        )
    model = record.get("distortion_model")
    if model not in {"none", "brown_conrady", "fisheye"}:
        _issue(issues, "distortion_model", f"{path}.distortion_model", "must name the distortion model")
    coefficients = record.get("distortion_coefficients")
    if not isinstance(coefficients, list) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
        for value in coefficients
    ):
        _issue(issues, "distortion_coefficients", f"{path}.distortion_coefficients", "must be a finite numeric list")


def _version_tuple(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        return None
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def _validate_fidelity(stream: Mapping[str, Any], path: str, issues: list[AbiIssue], *, required: bool) -> None:
    """Validate the evidence level and explicit unmodelled error sources."""

    if required:
        _required(stream, ("fidelity", "fidelity_limitations"), path, issues)
    if "fidelity" in stream and stream.get("fidelity") not in _FIDELITY_LEVELS:
        _issue(
            issues,
            "fidelity_level",
            f"{path}.fidelity",
            "must be simulator_consistent, noise_modeled, or hardware_calibrated",
        )
    limitations = stream.get("fidelity_limitations")
    if "fidelity_limitations" not in stream:
        return
    if not isinstance(limitations, list):
        _issue(issues, "fidelity_limitations", f"{path}.fidelity_limitations", "must be an array of strings")
        return
    seen: set[str] = set()
    for index, value in enumerate(limitations):
        item_path = f"{path}.fidelity_limitations[{index}]"
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            _issue(issues, "fidelity_limitation", item_path, "must be a non-empty string of at most 256 characters")
        elif value in seen:
            _issue(issues, "fidelity_limitation_duplicate", item_path, "limitations must be unique")
        else:
            seen.add(value)


def _validate_field(field: Any, path: str, issues: list[AbiIssue]) -> None:
    if not isinstance(field, Mapping):
        _issue(issues, "type", path, "must be an object")
        return
    allowed = frozenset(
        {
            "name",
            "dtype",
            "shape",
            "units",
            "frame_id",
            "agent_id_field",
            "timestamp_field",
            "missing",
            "valid_range",
            "compression",
            "time_semantics",
        }
    )
    _unknown(field, allowed, path, issues)
    _required(field, tuple(sorted(allowed)), path, issues)
    name = field.get("name")
    if not isinstance(name, str) or not _FIELD_NAME.fullmatch(name):
        _issue(issues, "field_name", f"{path}.name", "invalid field name")
    if field.get("dtype") not in _DTYPES:
        _issue(issues, "dtype", f"{path}.dtype", "unsupported dtype")
    shape = field.get("shape")
    if not isinstance(shape, list) or not shape or any(
        (not isinstance(dim, (str, int)) or isinstance(dim, bool) or (isinstance(dim, int) and dim < 0) or (isinstance(dim, str) and not _DIMENSION.fullmatch(dim)))
        for dim in shape
    ):
        _issue(issues, "shape", f"{path}.shape", "must be a non-empty list of dimensions")
    _nonempty_string(field.get("units"), f"{path}.units", issues, maximum=64)
    _nonempty_string(field.get("frame_id"), f"{path}.frame_id", issues, maximum=128)
    agent_id_field = field.get("agent_id_field")
    if agent_id_field is not None:
        _nonempty_string(agent_id_field, f"{path}.agent_id_field", issues, maximum=128)
    _nonempty_string(field.get("timestamp_field"), f"{path}.timestamp_field", issues, maximum=128)
    missing = field.get("missing")
    if not isinstance(missing, Mapping):
        _issue(issues, "missing", f"{path}.missing", "must be an object")
    else:
        missing_allowed = frozenset({"policy", "sentinel", "mask_field"})
        _unknown(missing, missing_allowed, f"{path}.missing", issues)
        _required(missing, tuple(sorted(missing_allowed)), f"{path}.missing", issues)
        if missing.get("policy") not in _MISSING_POLICIES:
            _issue(issues, "missing_policy", f"{path}.missing.policy", "unsupported missing-value policy")
        if missing.get("policy") == "mask_field" and not isinstance(missing.get("mask_field"), str):
            _issue(issues, "missing_mask", f"{path}.missing.mask_field", "mask_field is required for mask_field policy")
    valid_range = field.get("valid_range")
    if not isinstance(valid_range, Mapping):
        _issue(issues, "valid_range", f"{path}.valid_range", "must declare min, max, and inclusive")
    else:
        _unknown(valid_range, frozenset({"min", "max", "inclusive"}), f"{path}.valid_range", issues)
        _required(valid_range, ("min", "max", "inclusive"), f"{path}.valid_range", issues)
        for key in ("min", "max"):
            bound = valid_range.get(key)
            if bound is not None and (not isinstance(bound, (int, float)) or isinstance(bound, bool) or not math.isfinite(float(bound))):
                _issue(issues, "valid_range", f"{path}.valid_range.{key}", "must be finite numeric or null")
        lower = valid_range.get("min")
        upper = valid_range.get("max")
        if isinstance(lower, (int, float)) and not isinstance(lower, bool) and isinstance(upper, (int, float)) and not isinstance(upper, bool) and lower > upper:
            _issue(issues, "valid_range", f"{path}.valid_range", "min must not exceed max")
        if not isinstance(valid_range.get("inclusive"), bool):
            _issue(issues, "valid_range", f"{path}.valid_range.inclusive", "must be boolean")
    if field.get("compression") not in _COMPRESSIONS:
        _issue(issues, "compression", f"{path}.compression", "unsupported compression")
    if field.get("time_semantics") not in _TIME_SEMANTICS:
        _issue(issues, "time_semantics", f"{path}.time_semantics", "unsupported timestamp semantics")


def validate_observation_abi(payload: Any, *, require_fidelity: bool = False) -> tuple[AbiIssue, ...]:
    """Return every structural error in an observation ABI document.

    ABI 1.0 remains readable for development captures.  ABI 1.1 makes the
    sensor-fidelity level and explicitly unmodelled error sources mandatory;
    ``require_fidelity`` lets formal admission reject legacy documents even
    when a caller is validating an older version.
    """

    issues: list[AbiIssue] = []
    if not isinstance(payload, Mapping):
        return (AbiIssue("type", "$", "ABI must be a JSON object"),)
    allowed = frozenset({"schema", "version", "action_timing", "coordinate_frames", "calibration", "streams"})
    _unknown(payload, allowed, "$", issues)
    _required(payload, tuple(sorted(allowed)), "$", issues)
    if payload.get("schema") != OBSERVATION_ABI_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {OBSERVATION_ABI_SCHEMA!r}")
    if not isinstance(payload.get("version"), str) or not _VERSION.fullmatch(payload["version"]):
        _issue(issues, "version", "$.version", "must be a semantic version")
    _validate_action_timing(payload.get("action_timing"), issues)
    _validate_coordinate_frames(payload.get("coordinate_frames"), issues)
    _validate_calibration(payload.get("calibration"), issues)
    version = _version_tuple(payload.get("version"))
    fidelity_required = require_fidelity or (version is not None and version >= (1, 1, 0))
    if require_fidelity and (version is None or version < (1, 1, 0)):
        _issue(issues, "fidelity_required", "$.version", "formal episodes require observation ABI version >= 1.1.0")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        _issue(issues, "streams", "$.streams", "must be a non-empty list")
        return tuple(issues)
    seen: set[str] = set()
    for index, stream in enumerate(streams):
        path = f"$.streams[{index}]"
        if not isinstance(stream, Mapping):
            _issue(issues, "type", path, "must be an object")
            continue
        allowed_stream = frozenset(
            {"stream_id", "modality", "partition", "encoding", "fields", "fidelity", "fidelity_limitations"}
        )
        _unknown(stream, allowed_stream, path, issues)
        _required(stream, ("stream_id", "modality", "partition", "encoding", "fields"), path, issues)
        stream_id = stream.get("stream_id")
        if not isinstance(stream_id, str) or not _FIELD_NAME.fullmatch(stream_id):
            _issue(issues, "stream_id", f"{path}.stream_id", "invalid stream id")
        elif stream_id in seen:
            _issue(issues, "duplicate_stream", f"{path}.stream_id", "stream ids must be unique")
        else:
            seen.add(stream_id)
        _nonempty_string(stream.get("modality"), f"{path}.modality", issues, maximum=128)
        if stream.get("partition") not in _PARTITIONS:
            _issue(issues, "partition", f"{path}.partition", "must be policy_visible or learning_labels")
        _nonempty_string(stream.get("encoding"), f"{path}.encoding", issues, maximum=128)
        _validate_fidelity(stream, path, issues, required=fidelity_required)
        fields = stream.get("fields")
        if not isinstance(fields, list) or not fields:
            _issue(issues, "fields", f"{path}.fields", "must be a non-empty list")
            continue
        field_names: set[str] = set()
        for field_index, field in enumerate(fields):
            _validate_field(field, f"{path}.fields[{field_index}]", issues)
            if isinstance(field, Mapping) and isinstance(field.get("name"), str):
                if field["name"] in field_names:
                    _issue(issues, "duplicate_field", f"{path}.fields[{field_index}].name", "field names must be unique")
                field_names.add(field["name"])
                if stream.get("modality") == "high_level_action_history" and field.get("time_semantics") != "command_before_step":
                    _issue(issues, "action_causality", f"{path}.fields[{field_index}].time_semantics", "actions must be command_before_step")
    return tuple(issues)


def validate_formal_observation_abi(payload: Any) -> tuple[AbiIssue, ...]:
    """Validate the stricter ABI required by a formal release candidate."""

    return validate_observation_abi(payload, require_fidelity=True)


def _abi_semantic_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ABI meaning that must not change across a major version.

    Calibration measurements are episode data and may legitimately vary
    between cameras; their structure is validated by the ABI validator.
    Action/coordinate conventions and field-level meanings are contract
    semantics and are compared exactly. Fidelity metadata is handled
    separately because ABI 1.1 adds it to the legacy 1.0 reader boundary.
    """

    streams: list[dict[str, Any]] = []
    for stream in payload.get("streams", []):
        if not isinstance(stream, Mapping):
            continue
        streams.append(
            {
                "stream_id": stream.get("stream_id"),
                "modality": stream.get("modality"),
                "partition": stream.get("partition"),
                "encoding": stream.get("encoding"),
                "fields": stream.get("fields"),
            }
        )
    streams.sort(key=lambda item: str(item.get("stream_id")))
    return {
        "schema": payload.get("schema"),
        "action_timing": payload.get("action_timing"),
        "coordinate_frames": payload.get("coordinate_frames"),
        "streams": streams,
    }


def assess_observation_abi_compatibility(
    producer: Any,
    reader: Any,
) -> AbiCompatibilityReport:
    """Assess whether a reader can consume a producer ABI.

    Compatibility is conservative: versions must share a major number, the
    reader cannot be older than the producer, and all field-level semantics
    must match exactly. ABI 1.0 is readable in development by ABI 1.1, but
    remains formally ineligible because it lacks stream fidelity evidence.
    Invalid documents and semantic changes always fail closed.
    """

    issues: list[str] = []
    producer_issues = validate_observation_abi(producer)
    reader_issues = validate_observation_abi(reader)
    if producer_issues:
        issues.append("invalid_producer_abi")
    if reader_issues:
        issues.append("invalid_reader_abi")
    producer_version = producer.get("version") if isinstance(producer, Mapping) else None
    reader_version = reader.get("version") if isinstance(reader, Mapping) else None
    producer_tuple = _version_tuple(producer_version)
    reader_tuple = _version_tuple(reader_version)
    if producer_tuple is None:
        issues.append("invalid_producer_version")
    if reader_tuple is None:
        issues.append("invalid_reader_version")
    if producer_tuple is not None and reader_tuple is not None:
        if producer_tuple[0] != reader_tuple[0]:
            issues.append("major_version_mismatch")
        elif producer_tuple > reader_tuple:
            issues.append("producer_newer_than_reader")
        if isinstance(producer, Mapping) and isinstance(reader, Mapping):
            if _abi_semantic_projection(producer) != _abi_semantic_projection(reader):
                issues.append("semantic_contract_mismatch")
            producer_streams = {
                item.get("stream_id"): item
                for item in producer.get("streams", [])
                if isinstance(item, Mapping)
            }
            reader_streams = {
                item.get("stream_id"): item
                for item in reader.get("streams", [])
                if isinstance(item, Mapping)
            }
            for stream_id, stream in producer_streams.items():
                reader_stream = reader_streams.get(stream_id)
                if reader_stream is None:
                    continue
                if "fidelity" in stream and reader_stream.get("fidelity") != stream.get("fidelity"):
                    issues.append(f"fidelity_contract_mismatch:{stream_id}")
                if "fidelity_limitations" in stream and reader_stream.get("fidelity_limitations") != stream.get("fidelity_limitations"):
                    issues.append(f"fidelity_limitations_mismatch:{stream_id}")
    development_readable = not issues
    formal_admissible = development_readable and not validate_formal_observation_abi(producer)
    return AbiCompatibilityReport(
        producer_version=producer_version if isinstance(producer_version, str) else None,
        reader_version=reader_version if isinstance(reader_version, str) else None,
        development_readable=development_readable,
        formal_admissible=formal_admissible,
        issues=tuple(issues),
    )


def load_observation_abi(path: Path) -> dict[str, Any]:
    """Load one ABI document and fail before a reader sees unvalidated data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AbiError(f"cannot read observation ABI {path}: {exc}") from exc
    issues = validate_observation_abi(payload)
    if issues:
        raise AbiError("invalid observation ABI: " + "; ".join(f"{item.code}:{item.path}" for item in issues))
    return dict(payload)


def observation_abi_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical ABI bytes for manifest/provenance binding."""

    issues = validate_observation_abi(payload)
    if issues:
        raise AbiError("cannot hash invalid observation ABI")
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
