"""Build a post-capture ABI descriptor from audited native Isaac arrays.

The descriptor is written outside the immutable capture.  It describes public
policy-visible arrays and calibration that already exist; it does not add a
sensor claim, create a pack spec, or grant formal admission.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .abi import (
    OBSERVATION_ABI_SCHEMA,
    observation_abi_sha256,
    validate_formal_observation_abi,
)
from .policy_projection import (
    inspect_candidate_pack_streams,
    validate_candidate_abi_sources,
)

_STREAM_IDS = frozenset(
    {"actions", "state", "task", "messages", "rgb", "depth", "lidar", "imu"}
)
_ACTION_TIMING = {
    "command_write": "before_simulation_step",
    "simulation_step": "after_command_write",
    "state_update": "after_simulation_step",
    "sensor_read": "after_state_update",
    "storage": "after_sensor_read",
}
_COORDINATE_FRAMES = {
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
_FIDELITY_LIMITATIONS = {
    "actions": ["scripted_expert_commands_without_hardware_actuator_identification"],
    "state": ["simulator_rigid_body_state_without_external_tracking_noise"],
    "task": ["simulator_public_task_state_without_operator_annotation_error"],
    "messages": ["synchronous_messages_without_packet_loss_or_hardware_clock_error"],
    "rgb": ["no_lens_distortion_rolling_shutter_photon_noise_or_hardware_clock_error"],
    "depth": ["no_multipath_depth_noise_or_hardware_clock_error"],
    "lidar": ["raycaster_without_multipath_beam_divergence_or_hardware_noise"],
    "imu": ["simulator_imu_without_bias_drift_temperature_or_hardware_clock_error"],
}
_FIELD_SEMANTICS: dict[str, tuple[str, str, float | int | None, float | int | None]] = {
    "command_time_ns": ("ns", "simulation_clock", 0, None),
    "effective_time_ns": ("ns", "simulation_clock", 0, None),
    "timestamps_ns": ("ns", "simulation_clock", 0, None),
    "desired_pos_w_m": ("m", "world", None, None),
    "desired_vel_w_mps": ("m/s", "world", None, None),
    "root_ang_vel_b_radps": ("rad/s", "body_flu", None, None),
    "root_lin_vel_w_mps": ("m/s", "world", None, None),
    "root_pos_w_m": ("m", "world", None, None),
    "root_quat_wxyz": ("1", "world_from_body", -1, 1),
    "action_mode": ("1", "public_task", 0, 1),
    "coverage_cell_id": ("1", "public_task", 0, None),
    "desired_waypoint_w_m": ("m", "world", None, None),
    "distance_to_waypoint_m": ("m", "world", 0, None),
    "task_time_s": ("s", "simulation_clock", 0, None),
    "waypoint_index": ("1", "public_task", 0, None),
    "waypoint_progress": ("1", "public_task", 0, 1),
    "waypoint_reached": ("1", "public_task", 0, 1),
    "message_flags": ("bitmask", "public_team_message", 0, 255),
    "message_position_w_m": ("m", "world", None, None),
    "message_sequence": ("1", "public_team_message", 0, None),
    "message_velocity_w_mps": ("m/s", "world", None, None),
    "message_waypoint_index": ("1", "public_team_message", 0, None),
    "sender_agent_id": ("1", "public_team_message", 0, 7),
    "rgb": ("1", "camera_optical", 0, 255),
    "distance_to_image_plane_m": ("m", "camera_optical", 0.05, 100.0),
    "ranges_m": ("m", "body_flu", 0.0, 100.0),
    "angular_velocity_b_radps": ("rad/s", "body_flu", None, None),
    "linear_acceleration_b_mps2": ("m/s^2", "body_flu", None, None),
}


class IsaacPackDescriptorError(ValueError):
    """Raised when existing capture bytes cannot support a truthful ABI."""


def _object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsaacPackDescriptorError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise IsaacPackDescriptorError(f"{label} must be an object")
    return value


def _camera_calibration(
    calibration: Mapping[str, Any],
    source_streams: Mapping[str, Any],
) -> dict[str, Any]:
    camera = calibration.get("onboard_camera")
    if not isinstance(camera, Mapping):
        raise IsaacPackDescriptorError("calibration has no onboard_camera object")
    matrices = np.asarray(camera.get("intrinsic_matrices"), dtype=np.float64)
    if (
        matrices.shape != (8, 3, 3)
        or not np.all(np.isfinite(matrices))
        or not np.allclose(matrices, matrices[0], rtol=0.0, atol=1e-9)
    ):
        raise IsaacPackDescriptorError("eight onboard camera intrinsic matrices must be finite and identical")
    shape = camera.get("image_shape_hw")
    rgb_shape = source_streams["rgb"]["arrays"]["rgb"]["shape"]
    if shape != rgb_shape[-3:-1] or shape != [120, 160]:
        raise IsaacPackDescriptorError("camera calibration resolution does not match audited RGB arrays")
    matrix = matrices[0]
    return {
        "status": "recorded",
        "source": "calibration.json#onboard_camera",
        "frame_id": "camera_optical",
        "intrinsics": {
            "model": "pinhole",
            "width_px": int(shape[1]),
            "height_px": int(shape[0]),
            "fx_px": float(matrix[0, 0]),
            "fy_px": float(matrix[1, 1]),
            "cx_px": float(matrix[0, 2]),
            "cy_px": float(matrix[1, 2]),
        },
        "extrinsics": {
            "formula": "T_world_camera = T_world_body * T_body_camera",
            "quaternion_order": "wxyz",
        },
        "distortion_model": "none",
        "distortion_coefficients": [],
    }


def _calibration_payload(
    calibration: Mapping[str, Any],
    source_streams: Mapping[str, Any],
) -> dict[str, Any]:
    lidar = calibration.get("lidar")
    imu = calibration.get("imu")
    if not isinstance(lidar, Mapping):
        raise IsaacPackDescriptorError("calibration must contain a lidar object")
    if imu is not None and not isinstance(imu, Mapping):
        raise IsaacPackDescriptorError("imu calibration must be an object when present")
    if isinstance(imu, Mapping):
        implementation = imu.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise IsaacPackDescriptorError("recorded IMU calibration requires an implementation")
        if imu.get("attachment_frame") != "body_flu":
            raise IsaacPackDescriptorError(
                "recorded IMU calibration requires attachment_frame body_flu"
            )
    maximum = lidar.get("max_distance_m")
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0.0
    ):
        raise IsaacPackDescriptorError("LiDAR calibration requires a positive finite max_distance_m")
    return {
        "camera": _camera_calibration(calibration, source_streams),
        "lidar": {
            "status": "recorded",
            "source": "calibration.json#lidar",
            "frame_id": "body_flu",
            "max_range_m": float(maximum),
        },
        "imu": (
            {
                "status": "recorded",
                "source": "calibration.json#imu",
                "frame_id": "body_flu",
            }
            if isinstance(imu, Mapping)
            else {
                "status": "unavailable",
                "source": "calibration.json#imu (missing in source capture)",
            }
        ),
    }


def _field(
    stream_id: str,
    name: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    semantics = _FIELD_SEMANTICS.get(name)
    if semantics is None:
        raise IsaacPackDescriptorError(f"no reviewed ABI semantics for {stream_id}.{name}")
    units, frame_id, minimum, maximum = semantics
    try:
        dtype = np.dtype(str(descriptor["dtype"])).name
    except (KeyError, TypeError) as exc:
        raise IsaacPackDescriptorError(f"invalid dtype for {stream_id}.{name}") from exc
    raw_shape = descriptor.get("shape")
    if not isinstance(raw_shape, list) or not raw_shape:
        raise IsaacPackDescriptorError(f"invalid shape for {stream_id}.{name}")
    shape = list(raw_shape)
    shape[0] = "physics_step" if stream_id in {"actions", "state"} else "sensor_frame"
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "units": units,
        "frame_id": frame_id,
        "agent_id_field": None,
        "timestamp_field": (
            "command_time_ns"
            if stream_id == "actions"
            else "effective_time_ns"
            if stream_id == "state"
            else "timestamps_ns"
        ),
        "missing": {
            "policy": "not_applicable",
            "sentinel": None,
            "mask_field": None,
        },
        "valid_range": {"min": minimum, "max": maximum, "inclusive": True},
        "compression": "npz_deflate",
        "time_semantics": (
            "command_before_step"
            if stream_id == "actions"
            else "state_after_step"
            if stream_id == "state"
            else "sensor_sample"
        ),
    }


def build_isaac_observation_abi(
    source_streams: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and self-check one reusable ABI from audited source descriptors."""

    if set(source_streams) != _STREAM_IDS:
        raise IsaacPackDescriptorError("source contract must contain the exact eight T1 streams")
    streams = []
    for stream_id in sorted(source_streams):
        source = source_streams[stream_id]
        arrays = source.get("arrays") if isinstance(source, Mapping) else None
        if not isinstance(arrays, Mapping) or not arrays:
            raise IsaacPackDescriptorError(f"source stream {stream_id} has no array descriptors")
        streams.append(
            {
                "stream_id": stream_id,
                "modality": source["modality"],
                "partition": "policy_visible",
                "encoding": "npz",
                "fidelity": "simulator_consistent",
                "fidelity_limitations": list(_FIDELITY_LIMITATIONS[stream_id]),
                "fields": [
                    _field(stream_id, name, descriptor)
                    for name, descriptor in arrays.items()
                ],
            }
        )
    payload = {
        "schema": OBSERVATION_ABI_SCHEMA,
        "version": "1.1.0",
        "action_timing": dict(_ACTION_TIMING),
        "coordinate_frames": dict(_COORDINATE_FRAMES),
        "calibration": _calibration_payload(calibration, source_streams),
        "streams": streams,
    }
    abi_issues = validate_formal_observation_abi(payload)
    source_issues = validate_candidate_abi_sources(payload, source_streams)
    if abi_issues or source_issues:
        detail = "; ".join(
            f"{issue.code}:{issue.path}" for issue in (*abi_issues, *source_issues)
        )
        raise IsaacPackDescriptorError(f"generated ABI failed closed: {detail}")
    return payload


def descriptor_for_capture(capture_root: Path) -> dict[str, Any]:
    """Inspect one immutable capture and build its external ABI descriptor."""

    capture = capture_root.expanduser().resolve()
    source_streams = inspect_candidate_pack_streams(capture)
    calibration = _object(capture / "calibration.json", label="calibration.json")
    return build_isaac_observation_abi(source_streams, calibration)


def write_descriptor(capture_root: Path, output: Path) -> str:
    """Atomically write a new external descriptor and return its canonical hash."""

    destination = output.expanduser().resolve()
    if destination.exists():
        raise IsaacPackDescriptorError(f"refusing to overwrite descriptor: {destination}")
    payload = descriptor_for_capture(capture_root)
    serialized = (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as descriptor:
            temporary = Path(descriptor.name)
            descriptor.write(serialized)
            descriptor.flush()
            os.fsync(descriptor.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return observation_abi_sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        digest = write_descriptor(args.capture_root, args.output)
    except (IsaacPackDescriptorError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "written",
                "formal_benchmark_admission": False,
                "output": str(args.output.expanduser().resolve()),
                "observation_abi_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IsaacPackDescriptorError",
    "build_isaac_observation_abi",
    "descriptor_for_capture",
    "write_descriptor",
]
