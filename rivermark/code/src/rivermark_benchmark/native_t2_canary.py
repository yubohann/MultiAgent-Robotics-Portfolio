"""Public control and perception primitives for the native Isaac T2 canary."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._identity import IdentityAccumulator
from .cf2x_runtime_calibration import validate_calibration_report
from .citylite_scene import AGENT_COUNT
from .isaac_transfer import _wrap_angle
from .t2_policy_abi import (
    T2CandidateDetection,
    T2PolicyAbiError,
    T2PublicFleetObservation,
)

NATIVE_T2_POLICY_SCHEMA = "org.rivermark.native-t2-public-route-policy.v1"
NATIVE_T2_DETECTOR_SCHEMA = "org.rivermark.native-t2-rgbd-semantic-detector.v1"
NATIVE_T2_TRACE_SCHEMA = "org.rivermark.native-t2-trace.v1"
NATIVE_T2_EVENTS_SCHEMA = "org.rivermark.native-t2-events.v1"
NATIVE_T2_CALIBRATION_BINDING_SCHEMA = "org.rivermark.native-t2-calibration-binding.v1"
TARGET_SLOT_PREFIX = "search_target_slot_"


def _canonical_identity(value: Any) -> str:
    try:
        payload = json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise T2PolicyAbiError("native T2 provenance cannot be canonicalized") from exc
    return IdentityAccumulator(payload + b"\n").hexdigest()


def _finite_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise T2PolicyAbiError(f"{name} must be a finite numeric array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise T2PolicyAbiError(f"{name} must be finite with shape {shape}")
    return array.copy()


def native_rgbd_world_points(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    pos_w_m: np.ndarray,
    quat_w_ros: np.ndarray,
) -> np.ndarray:
    """Return canonical CPU world points for a retained native RGB-D frame."""

    depth = np.asarray(depth_m, dtype=np.float32)
    matrices = np.asarray(intrinsics, dtype=np.float32)
    positions = np.asarray(pos_w_m, dtype=np.float32)
    quaternions = np.asarray(quat_w_ros, dtype=np.float32)
    if (
        depth.ndim != 4
        or depth.shape[0] != AGENT_COUNT
        or depth.shape[-1] != 1
        or matrices.shape != (AGENT_COUNT, 3, 3)
        or positions.shape != (AGENT_COUNT, 3)
        or quaternions.shape != (AGENT_COUNT, 4)
        or not all(np.isfinite(value).all() for value in (depth, matrices, positions, quaternions))
    ):
        raise ValueError("native RGB-D extrinsics have incompatible shape or non-finite values")
    height, width = depth.shape[1:3]
    if height < 2 or width < 2:
        raise ValueError("native RGB-D frame is too small to unproject")
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="ij"
    )
    pixels = np.stack((u, v, np.ones_like(u)), axis=0).reshape(3, -1)
    rays = np.linalg.inv(matrices) @ pixels[None, :, :]
    rays = rays / rays[:, 2:3, :]
    points_camera = rays.transpose(0, 2, 1) * depth[..., 0].transpose(0, 2, 1).reshape(
        AGENT_COUNT, -1, 1
    )
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-8):
        raise ValueError("native camera quaternion has zero norm")
    w, x, y, z = (quaternions[:, index] / norms for index in range(4))
    rotation = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=1),
        ),
        axis=1,
    )
    return np.einsum("nij,npj->npi", rotation, points_camera) + positions[:, None, :]


def bind_native_t2_calibration(
    report: Any,
    *,
    expected_usd_identity: str,
    expected_runtime_lock_identity: str,
    expected_control_dt_s: float,
) -> dict[str, Any]:
    """Return a path-free, identity-bound calibration commitment for native T2."""

    issues = validate_calibration_report(report)
    if issues:
        raise T2PolicyAbiError("CF2X runtime calibration is invalid: " + "; ".join(issues))
    if not isinstance(report, Mapping) or report.get("status") != "passed":
        raise T2PolicyAbiError("CF2X runtime calibration must have passed")
    if not isinstance(expected_usd_identity, str) or len(expected_usd_identity) != 16:
        raise T2PolicyAbiError("expected CF2X USD identity must be a short identity")
    if (
        not isinstance(expected_runtime_lock_identity, str)
        or len(expected_runtime_lock_identity) != 16
    ):
        raise T2PolicyAbiError("expected runtime-lock identity must be a short identity")
    if not math.isfinite(float(expected_control_dt_s)) or float(expected_control_dt_s) <= 0.0:
        raise T2PolicyAbiError("expected control dt must be finite and positive")

    asset = report.get("asset")
    runtime = report.get("runtime")
    if not isinstance(asset, Mapping) or not isinstance(runtime, Mapping):
        raise T2PolicyAbiError("CF2X runtime calibration is missing asset or runtime evidence")
    if asset.get("usd_identity") != expected_usd_identity:
        raise T2PolicyAbiError("CF2X runtime calibration USD identity does not match capture")
    if report.get("runtime_lock_identity") != expected_runtime_lock_identity:
        raise T2PolicyAbiError("CF2X runtime calibration runtime-lock identity does not match capture")
    actuator = runtime.get("actuator")
    if not isinstance(actuator, Mapping):
        raise T2PolicyAbiError("CF2X runtime calibration actuator evidence is missing")
    for field in ("control_dt_s", "actuator_dt_s"):
        value = actuator.get(field)
        if not isinstance(value, (int, float)) or not math.isclose(
            float(value), float(expected_control_dt_s), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise T2PolicyAbiError(
                f"CF2X runtime calibration {field} does not match capture dt"
            )
    rotor_names = runtime.get("thruster_names")
    allocation = runtime.get("allocation_matrix")
    axis = runtime.get("thrust_axis")
    if (
        not isinstance(rotor_names, list)
        or not isinstance(allocation, list)
        or not isinstance(axis, Mapping)
        or axis.get("all_positive_body_z") is not True
    ):
        raise T2PolicyAbiError("CF2X runtime calibration physical evidence is incomplete")
    report_identity = report.get("report_identity")
    source = report.get("source")
    if not isinstance(report_identity, str) or len(report_identity) != 16 or not isinstance(source, Mapping):
        raise T2PolicyAbiError("CF2X runtime calibration provenance is incomplete")
    return {
        "schema": NATIVE_T2_CALIBRATION_BINDING_SCHEMA,
        "report_identity": report_identity,
        "calibration_source_revision": source.get("source_revision"),
        "calibration_source_tree_identity": source.get("source_tree_identity"),
        "cf2x_usd_identity": expected_usd_identity,
        "runtime_lock_identity": expected_runtime_lock_identity,
        "control_dt_s": float(expected_control_dt_s),
        "rotor_order_identity": _canonical_identity(rotor_names),
        "allocation_matrix_identity": _canonical_identity(allocation),
        "positive_body_z_thrust_axis": True,
        "actuator_response_identity": _canonical_identity(dict(actuator)),
        "calibration_path_released": False,
        "calibration_payload_released": False,
    }


@dataclass(frozen=True)
class PublicRouteCoveragePolicy:
    """A deterministic, target-blind route follower for the T2 canary."""

    routes_w_m: np.ndarray
    waypoint_segment_seconds: float
    route_start_time_ns: int = 0
    position_feedback_gain: float = 0.8
    yaw_feedback_gain: float = 1.2

    def __post_init__(self) -> None:
        routes = np.asarray(self.routes_w_m, dtype=np.float64)
        if (
            routes.ndim != 3
            or routes.shape[0] != AGENT_COUNT
            or routes.shape[1] < 2
            or routes.shape[2] != 3
            or not np.all(np.isfinite(routes))
        ):
            raise T2PolicyAbiError("public T2 routes must be finite [8,W,3] with W >= 2")
        for name, value in (
            ("waypoint_segment_seconds", self.waypoint_segment_seconds),
            ("position_feedback_gain", self.position_feedback_gain),
            ("yaw_feedback_gain", self.yaw_feedback_gain),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise T2PolicyAbiError(f"{name} must be finite and positive")
        if isinstance(self.route_start_time_ns, bool) or not isinstance(
            self.route_start_time_ns, (int, np.integer)
        ) or int(self.route_start_time_ns) < 0:
            raise T2PolicyAbiError("route_start_time_ns must be a non-negative integer")
        result = routes.copy()
        result.setflags(write=False)
        object.__setattr__(self, "routes_w_m", result)

    def __call__(self, observation: T2PublicFleetObservation) -> np.ndarray:
        if not isinstance(observation, T2PublicFleetObservation):
            raise T2PolicyAbiError("native T2 policy requires a public fleet observation")
        elapsed_s = max(0, observation.command_time_ns - self.route_start_time_ns) / 1_000_000_000.0
        segment_count = self.routes_w_m.shape[1] - 1
        route_time = max(0.0, elapsed_s) / float(self.waypoint_segment_seconds)
        segment = min(int(route_time), segment_count - 1)
        progress = min(max(route_time - segment, 0.0), 1.0)
        if route_time >= segment_count:
            segment = segment_count - 1
            progress = 1.0
        start, end = self.routes_w_m[:, segment], self.routes_w_m[:, segment + 1]
        desired_position = start + (end - start) * progress
        feedforward = (end - start) / float(self.waypoint_segment_seconds)
        if route_time >= segment_count:
            feedforward = np.zeros_like(feedforward)
        state = observation.state.values
        velocity = feedforward + float(self.position_feedback_gain) * (
            desired_position - state[:, :3]
        )
        horizontal = feedforward[:, :2]
        heading = np.where(
            np.linalg.norm(horizontal, axis=1) > 1.0e-9,
            np.arctan2(horizontal[:, 1], horizontal[:, 0]),
            state[:, 6],
        )
        yaw_rate = float(self.yaw_feedback_gain) * np.asarray(
            _wrap_angle(heading - state[:, 6]), dtype=np.float64
        )
        return np.concatenate((velocity, yaw_rate[:, None]), axis=1)

    def provenance(self) -> dict[str, Any]:
        routes = self.routes_w_m.tolist()
        return {
            "schema": NATIVE_T2_POLICY_SCHEMA,
            "policy_kind": "deterministic_public_route_coverage",
            "route_identity": _canonical_identity(routes),
            "route_agent_count": AGENT_COUNT,
            "route_waypoint_count": int(self.routes_w_m.shape[1]),
            "waypoint_segment_seconds": float(self.waypoint_segment_seconds),
            "route_start_time_ns": int(self.route_start_time_ns),
            "position_feedback_gain": float(self.position_feedback_gain),
            "yaw_feedback_gain": float(self.yaw_feedback_gain),
            "private_evaluator_inputs": False,
        }


def _semantic_slot_ids(metadata: Any) -> dict[int, str]:
    """Map one camera's Replicator IDs to public anonymous slot classes."""

    if not isinstance(metadata, Mapping):
        return {}
    labels = metadata.get("id_to_labels", metadata.get("idToLabels"))
    if not isinstance(labels, Mapping):
        return {}
    slots: dict[int, str] = {}
    for raw_id, raw_labels in labels.items():
        try:
            semantic_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        class_value = raw_labels.get("class") if isinstance(raw_labels, Mapping) else None
        classes = {
            value.strip().lower()
            for value in str(class_value).split(",")
            if value.strip()
        }
        for class_name in classes:
            if class_name.startswith(TARGET_SLOT_PREFIX):
                # The slot is present in public semantic metadata, but is
                # capture-local evidence only.  Candidate events retain no
                # slot or target identity.
                slots[semantic_id] = class_name
    return slots


def native_semantic_rgbd_candidates(
    semantic_segmentation: Any,
    semantic_metadata: Any,
    points_w_m: Any,
    *,
    minimum_pixels: int,
) -> tuple[tuple[T2CandidateDetection, ...], ...]:
    """Recover public candidates from synchronized semantic RGB-D point clouds."""

    if isinstance(minimum_pixels, bool) or not isinstance(minimum_pixels, int) or minimum_pixels < 1:
        raise T2PolicyAbiError("minimum_pixels must be a positive integer")
    labels = np.asarray(semantic_segmentation)
    if labels.shape == (AGENT_COUNT, *labels.shape[1:]) and labels.ndim == 4 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 3 or labels.shape[0] != AGENT_COUNT or not np.issubdtype(labels.dtype, np.integer):
        raise T2PolicyAbiError("semantic segmentation must be integer [8,H,W,1]")
    points = np.asarray(points_w_m, dtype=np.float64)
    expected_flat = (AGENT_COUNT, labels.shape[1] * labels.shape[2], 3)
    if points.shape == (AGENT_COUNT, labels.shape[1], labels.shape[2], 3):
        points = points.reshape(expected_flat)
    if points.shape != expected_flat:
        raise T2PolicyAbiError(
            "world point cloud must align to the semantic image as [8,H*W,3]"
        )
    per_camera = semantic_metadata.get("per_camera") if isinstance(semantic_metadata, Mapping) else None
    if not isinstance(per_camera, Sequence) or isinstance(per_camera, (str, bytes)) or len(per_camera) != AGENT_COUNT:
        raise T2PolicyAbiError("semantic metadata must provide one mapping for each CF2X camera")

    output: list[tuple[T2CandidateDetection, ...]] = []
    for agent_id in range(AGENT_COUNT):
        ids = _semantic_slot_ids(per_camera[agent_id])
        candidates: list[T2CandidateDetection] = []
        for semantic_id, slot in sorted(ids.items()):
            # IsaacLab ``unproject_depth`` flattens the depth image after a
            # H/W transpose (u-major / column-major pixel order).  Keep the
            # semantic mask in that exact order before indexing the returned
            # point cloud.  A plain row-major reshape happens to look correct
            # for square all-one fixtures but pairs different pixels on every
            # real non-square camera image.
            mask = labels[agent_id].transpose(1, 0).reshape(-1) == semantic_id
            observed = points[agent_id][mask]
            observed = observed[np.all(np.isfinite(observed), axis=1)]
            if len(observed) < minimum_pixels:
                continue
            center = np.median(observed, axis=0)
            confidence = min(1.0, float(len(observed)) / float(2 * minimum_pixels))
            candidates.append(
                T2CandidateDetection(
                    agent_id=agent_id,
                    position_w_m=tuple(float(value) for value in center),
                    confidence=confidence,
                    deduplication_key=slot,
                )
            )
        output.append(tuple(candidates))
    return tuple(output)


@dataclass
class SpatialCandidateDeduplicator:
    """Suppress repeated public detections without consulting evaluator truth."""

    merge_radius_m: float
    _accepted_positions_w_m: list[np.ndarray] | None = None
    _accepted_keys: set[str] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.merge_radius_m)) or float(self.merge_radius_m) <= 0.0:
            raise T2PolicyAbiError("merge_radius_m must be finite and positive")
        self._accepted_positions_w_m = []
        self._accepted_keys = set()

    def filter(self, detections: Iterable[T2CandidateDetection]) -> tuple[T2CandidateDetection, ...]:
        accepted: list[T2CandidateDetection] = []
        for detection in detections:
            if not isinstance(detection, T2CandidateDetection):
                raise T2PolicyAbiError("deduplicator requires T2CandidateDetection values")
            position = np.asarray(detection.position_w_m, dtype=np.float64)
            key = detection.deduplication_key
            if key is not None and key in (self._accepted_keys or set()):
                continue
            if all(
                float(np.linalg.norm(position - previous)) > float(self.merge_radius_m)
                for previous in self._accepted_positions_w_m or ()
            ):
                assert self._accepted_positions_w_m is not None
                self._accepted_positions_w_m.append(position)
                if key is not None:
                    assert self._accepted_keys is not None
                    self._accepted_keys.add(key)
                accepted.append(detection)
        return tuple(accepted)


__all__ = [
    "NATIVE_T2_CALIBRATION_BINDING_SCHEMA",
    "NATIVE_T2_DETECTOR_SCHEMA",
    "NATIVE_T2_EVENTS_SCHEMA",
    "NATIVE_T2_POLICY_SCHEMA",
    "NATIVE_T2_TRACE_SCHEMA",
    "PublicRouteCoveragePolicy",
    "SpatialCandidateDeduplicator",
    "bind_native_t2_calibration",
    "native_rgbd_world_points",
    "native_semantic_rgbd_candidates",
]
