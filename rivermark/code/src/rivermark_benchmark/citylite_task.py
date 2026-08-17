"""Public-geometry observability checks for private City-Lite targets.

The evaluator owns target coordinates, but their difficulty label is not
trusted metadata.  This module reconstructs conservative onboard-camera
witnesses from a frozen public route and checks their frustum and line of
sight against the same structural AABBs used by the runtime safety guard.
Native semantic frames remain the final evidence that a rendered target was
actually observable in Isaac.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .citylite_scene import (
    AABB,
    CITY_LITE_TARGET_REGION_A_ID,
    CITY_LITE_TARGET_REGION_B_ID,
    TARGET_REGIONS_W_M,
    canonical_payload_sha256,
    coerce_aabb,
    segment_intersects_aabb,
)


# ``v3`` remains the frozen fixed-initial-heading contract used by T1 and the
# first native-T2 canary.  The yaw-aware v4 form below is deliberately opt-in:
# regenerating a manifest for a historical protocol must not silently change
# the geometric claim it made.
TARGET_VISIBILITY_GEOMETRY_SCHEMA = "org.rivermark.private-target-visibility-geometry.v3"
TARGET_VISIBILITY_GEOMETRY_V4_SCHEMA = "org.rivermark.private-target-visibility-geometry.v4"
TARGET_VISIBILITY_EXECUTION_WINDOW_SCHEMA = "org.rivermark.public-route-execution-window.v1"
TARGET_VISIBILITY_DIRECT = "direct-visible-v1"
TARGET_VISIBILITY_PARTIAL = "partial-visible-v1"
TARGET_VISIBILITY_BUCKETS = frozenset(
    (TARGET_VISIBILITY_DIRECT, TARGET_VISIBILITY_PARTIAL)
)
ONBOARD_FOCAL_LENGTH_MM = 24.0
ONBOARD_HORIZONTAL_APERTURE_MM = 20.955
ONBOARD_IMAGE_WIDTH = 160
ONBOARD_IMAGE_HEIGHT = 120
LIDAR_CHANNEL_COUNT = 16
LIDAR_VERTICAL_FOV_RANGE_DEG = (-35.0, 25.0)
LIDAR_HORIZONTAL_FOV_RANGE_DEG = (-180.0, 180.0)
LIDAR_HORIZONTAL_RESOLUTION_DEG = 5.0
LIDAR_HORIZONTAL_SAMPLE_COUNT = 72
LIDAR_RAY_COUNT = LIDAR_CHANNEL_COUNT * LIDAR_HORIZONTAL_SAMPLE_COUNT
ONBOARD_PITCH_DOWN_RAD = math.radians(15.0)
ONBOARD_CAMERA_OFFSET_BODY_M = (0.12, 0.0, 0.04)
TARGET_VISIBILITY_MIN_DISTANCE_M = 2.0
TARGET_VISIBILITY_MAX_DISTANCE_M = 35.0
TARGET_VISIBILITY_FRUSTUM_MARGIN = 0.92
TARGET_VISIBILITY_ROUTE_SAMPLES_PER_SEGMENT = 5
TARGET_VISIBILITY_MIN_WITNESSES = 1
TARGET_VISIBILITY_MIN_NATIVE_FRAMES = 1
TARGET_VISIBILITY_MIN_NATIVE_PIXELS = 8
# The native semantic gate is deliberately lower than this analytic sampling
# bound. Rasterization, camera-pose tracking and exact scene geometry remain
# runtime evidence; the extra analytic margin prevents a distant point-like
# target from being accepted solely because its centre lies in the frustum.
TARGET_VISIBILITY_MIN_PROJECTED_INSTANCE_PIXELS = 12.0
# A route witness is accepted only when the target remains observable under
# the bounded physical tracking error seen in the Isaac canary.  This is a
# translation envelope in world metres, not a claim about controller error
# distributions; native semantic frames remain the final evidence.
TARGET_VISIBILITY_TRACKING_ENVELOPE_M = 1.5
TARGET_VISIBILITY_PROTOCOL_DT_S = 0.005
TARGET_VISIBILITY_PROTOCOL_WARMUP_STEPS = 120
TARGET_VISIBILITY_PROTOCOL_ROLLOUT_STEPS = 2400
TARGET_VISIBILITY_PROTOCOL_CAPTURE_STRIDE = 10
PUBLIC_ROUTE_WAYPOINT_SEGMENT_SECONDS = 6.0
CAMERA_HEADING_MODEL_FIXED_INITIAL = "fixed_initial_horizontal_heading_v1"
CAMERA_HEADING_MODEL_SEGMENT_YAW_LIMITED = "segment_horizontal_heading_yaw_limited_v1"
CAMERA_HEADING_MODELS = frozenset(
    (CAMERA_HEADING_MODEL_FIXED_INITIAL, CAMERA_HEADING_MODEL_SEGMENT_YAW_LIMITED)
)


def route_timing_requirements(
    routes_w_m: Sequence[Sequence[Sequence[float]]], *, waypoint_segment_seconds: float
) -> dict[str, float | int]:
    """Return the public velocity lower bounds imposed by a waypoint schedule.

    This is deliberately kinematic: it checks whether a bounded velocity ABI
    can even express the published route before Isaac allocates a stage.  It
    does not claim physical trackability, which remains a native canary gate.
    """

    duration = float(waypoint_segment_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("waypoint_segment_seconds must be finite and positive")
    horizontal_required: list[float] = []
    vertical_required: list[float] = []
    segment_count = 0
    for route in routes_w_m:
        for start, end in zip(route, route[1:]):
            delta_x = float(end[0]) - float(start[0])
            delta_y = float(end[1]) - float(start[1])
            delta_z = float(end[2]) - float(start[2])
            values = (delta_x, delta_y, delta_z)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("routes must contain finite world coordinates")
            horizontal_required.append(math.hypot(delta_x, delta_y) / duration)
            vertical_required.append(abs(delta_z) / duration)
            segment_count += 1
    if segment_count == 0:
        raise ValueError("routes must contain at least one segment")
    return {
        "segment_count": segment_count,
        "maximum_required_horizontal_speed_mps": max(horizontal_required),
        "maximum_required_vertical_speed_mps": max(vertical_required),
    }


def validate_route_timing_feasibility(
    routes_w_m: Sequence[Sequence[Sequence[float]]],
    *,
    waypoint_segment_seconds: float,
    max_horizontal_speed_mps: float,
    max_vertical_speed_mps: float,
    utilization_limit: float,
) -> dict[str, float | int]:
    """Fail closed when a scheduled route exceeds bounded public actions.

    ``utilization_limit`` reserves a declared fraction of command authority
    for feedback.  The result is receipt-safe public geometry; callers record
    it rather than inferring feasibility from a later successful video.
    """

    values = {
        "max_horizontal_speed_mps": max_horizontal_speed_mps,
        "max_vertical_speed_mps": max_vertical_speed_mps,
        "utilization_limit": utilization_limit,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if float(utilization_limit) > 1.0:
        raise ValueError("utilization_limit must not exceed one")
    requirements = route_timing_requirements(
        routes_w_m, waypoint_segment_seconds=waypoint_segment_seconds
    )
    # These values enter hash-bound public receipts.  Canonicalize the small
    # binary multiplication residue at the contract boundary so equivalent
    # action envelopes do not serialize as different JSON numbers.
    horizontal_budget = round(float(max_horizontal_speed_mps) * float(utilization_limit), 12)
    vertical_budget = round(float(max_vertical_speed_mps) * float(utilization_limit), 12)
    horizontal_required = float(requirements["maximum_required_horizontal_speed_mps"])
    vertical_required = float(requirements["maximum_required_vertical_speed_mps"])
    if horizontal_required > horizontal_budget + 1.0e-12:
        raise ValueError(
            "route schedule requires horizontal speed "
            f"{horizontal_required:.6f} m/s but the bounded-action budget is "
            f"{horizontal_budget:.6f} m/s"
        )
    if vertical_required > vertical_budget + 1.0e-12:
        raise ValueError(
            "route schedule requires vertical speed "
            f"{vertical_required:.6f} m/s but the bounded-action budget is "
            f"{vertical_budget:.6f} m/s"
        )
    return {
        **requirements,
        "horizontal_speed_budget_mps": horizontal_budget,
        "vertical_speed_budget_mps": vertical_budget,
        "utilization_limit": float(utilization_limit),
    }


class TargetSamplingError(ValueError):
    """Raised when a private target cohort cannot satisfy every hard gate."""


def target_visibility_execution_window(
    *,
    dt_s: float = TARGET_VISIBILITY_PROTOCOL_DT_S,
    warmup_steps: int = TARGET_VISIBILITY_PROTOCOL_WARMUP_STEPS,
    rollout_steps: int = TARGET_VISIBILITY_PROTOCOL_ROLLOUT_STEPS,
    capture_stride: int = TARGET_VISIBILITY_PROTOCOL_CAPTURE_STRIDE,
    waypoint_segment_seconds: float = PUBLIC_ROUTE_WAYPOINT_SEGMENT_SECONDS,
) -> dict[str, float | int | str]:
    """Describe the public route interval represented by retained sensor frames.

    Sampling against every waypoint in a route that a short physical rollout
    never reaches produces targets that are geometrically visible only in the
    future.  This frozen window is part of the private-target geometry
    contract and is checked again by the Isaac capture command.
    """

    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (rollout_steps, capture_stride)
    ):
        raise ValueError("rollout_steps and capture_stride must be positive integers")
    if capture_stride > rollout_steps:
        raise ValueError("capture_stride cannot exceed rollout_steps")
    if not math.isfinite(float(waypoint_segment_seconds)) or float(waypoint_segment_seconds) <= 0.0:
        raise ValueError("waypoint_segment_seconds must be finite and positive")
    first_sensor_time_s = float(warmup_steps + capture_stride) * float(dt_s)
    last_sensor_time_s = float(warmup_steps + rollout_steps) * float(dt_s)
    return {
        "schema": TARGET_VISIBILITY_EXECUTION_WINDOW_SCHEMA,
        "dt_s": float(dt_s),
        "warmup_steps": int(warmup_steps),
        "rollout_steps": int(rollout_steps),
        "capture_stride": int(capture_stride),
        "first_retained_sensor_time_s": first_sensor_time_s,
        "last_retained_sensor_time_s": last_sensor_time_s,
        "waypoint_segment_seconds": float(waypoint_segment_seconds),
    }


def _point_to_segment_distance_m(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    direction = tuple(float(end[axis]) - float(start[axis]) for axis in range(3))
    squared_length = sum(component * component for component in direction)
    if squared_length <= 1.0e-12:
        return math.dist(tuple(float(value) for value in point), tuple(float(value) for value in start))
    projection = sum(
        (float(point[axis]) - float(start[axis])) * direction[axis]
        for axis in range(3)
    ) / squared_length
    alpha = min(1.0, max(0.0, projection))
    closest = tuple(float(start[axis]) + alpha * direction[axis] for axis in range(3))
    return math.dist(tuple(float(value) for value in point), closest)


def _candidate_order_key(seed: int, index: int) -> bytes:
    """Use a documented digest order instead of implementation-dependent RNG."""

    return hashlib.sha256(f"rivermark-private-target:{seed}:{index}".encode("ascii")).digest()


def _private_candidate_grid(
    region: AABB, *, margin_m: float, spacing_m: float
) -> tuple[tuple[float, float, float], ...]:
    if spacing_m <= 0.0 or not math.isfinite(spacing_m):
        raise TargetSamplingError("candidate spacing must be finite and positive")
    minimum = tuple(float(value) + margin_m for value in region.minimum)
    maximum = tuple(float(value) - margin_m for value in region.maximum)
    if any(low > high for low, high in zip(minimum, maximum)):
        raise TargetSamplingError("target region is smaller than the placement margin")

    axes: list[tuple[float, ...]] = []
    for low, high in zip(minimum, maximum):
        values: list[float] = []
        value = low
        while value <= high + 1.0e-9:
            values.append(round(value, 6))
            value += spacing_m
        if not values or values[-1] < high - 1.0e-9:
            values.append(round(high, 6))
        axes.append(tuple(values))
    return tuple(
        (x, y, z)
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )


def sample_private_targets(
    *,
    seed: int,
    target_count: int,
    target_region_id: str,
    visibility_bucket: str,
    routes_w_m: Sequence[Sequence[Sequence[float]]],
    structural_aabbs: Sequence[AABB | Mapping[str, Any]],
    radius_m: float = 0.14,
    obstacle_clearance_m: float = 0.85,
    minimum_route_separation_m: float = 2.0,
    minimum_pairwise_separation_m: float = 1.5,
    candidate_spacing_m: float = 2.0,
    tracking_envelope_m: float = TARGET_VISIBILITY_TRACKING_ENVELOPE_M,
    execution_window: Mapping[str, Any] | None = None,
    camera_heading_model: str = CAMERA_HEADING_MODEL_FIXED_INITIAL,
    max_yaw_rate_rad_s: float | None = None,
    yaw_feedback_gain: float | None = None,
    yaw_stability_error_rad: float | None = None,
    yaw_settle_margin_s: float | None = None,
) -> tuple[dict[str, Any], ...]:
    """Sample an evaluator-private target cohort under immutable geometry gates.

    The returned coordinates and seed are private evaluator material.  This
    helper intentionally has no public-task or policy-input side effects.  It
    uses a digest-defined candidate order so a clean-room evaluator can replay
    the selection without relying on Python's random implementation.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TargetSamplingError("seed must be an integer")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count <= 0:
        raise TargetSamplingError("target_count must be a positive integer")
    if target_region_id not in TARGET_REGIONS_W_M:
        raise TargetSamplingError(f"unknown City-Lite target region: {target_region_id}")
    if visibility_bucket not in TARGET_VISIBILITY_BUCKETS:
        raise TargetSamplingError(f"unknown target visibility bucket: {visibility_bucket}")
    finite_positive = {
        "radius_m": radius_m,
        "obstacle_clearance_m": obstacle_clearance_m,
        "minimum_route_separation_m": minimum_route_separation_m,
        "minimum_pairwise_separation_m": minimum_pairwise_separation_m,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise TargetSamplingError(f"{name} must be finite and positive")
    if not math.isfinite(float(tracking_envelope_m)) or float(tracking_envelope_m) < 0.0:
        raise TargetSamplingError("tracking_envelope_m must be finite and non-negative")
    boxes = tuple(coerce_aabb(value) for value in structural_aabbs)
    window = (
        target_visibility_execution_window()
        if execution_window is None
        else dict(execution_window)
    )
    _validate_target_visibility_execution_window(window)
    _validate_camera_heading_model(
        camera_heading_model,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        yaw_feedback_gain=yaw_feedback_gain,
        yaw_stability_error_rad=yaw_stability_error_rad,
        yaw_settle_margin_s=yaw_settle_margin_s,
    )
    segments = tuple(
        (start, end)
        for route in routes_w_m
        for start, end in zip(route, route[1:])
    )
    if not segments:
        raise TargetSamplingError("routes must contain at least one segment")

    region = TARGET_REGIONS_W_M[target_region_id]
    candidates = _private_candidate_grid(
        region,
        margin_m=float(radius_m) + float(obstacle_clearance_m),
        spacing_m=float(candidate_spacing_m),
    )
    ordered = sorted(
        enumerate(candidates), key=lambda item: _candidate_order_key(seed, item[0])
    )
    chosen: list[tuple[tuple[float, float, float], TargetVisibilityEvidence]] = []
    rejected = {"obstacle": 0, "route": 0, "visibility": 0, "separation": 0}
    for _candidate_index, candidate in ordered:
        if any(
            box.expanded(float(obstacle_clearance_m) + float(radius_m)).contains(candidate)
            for box in boxes
        ):
            rejected["obstacle"] += 1
            continue
        route_distance = min(
            _point_to_segment_distance_m(candidate, start, end)
            for start, end in segments
        )
        if route_distance < float(minimum_route_separation_m):
            rejected["route"] += 1
            continue
        evidence = measure_target_visibility(
            candidate,
            routes_w_m=routes_w_m,
            structural_aabbs=boxes,
            radius_m=float(radius_m),
            tracking_envelope_m=float(tracking_envelope_m),
            execution_window=window,
            camera_heading_model=camera_heading_model,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
            yaw_feedback_gain=yaw_feedback_gain,
            yaw_stability_error_rad=yaw_stability_error_rad,
            yaw_settle_margin_s=yaw_settle_margin_s,
        )
        if evidence.visibility_bucket != visibility_bucket:
            rejected["visibility"] += 1
            continue
        if any(
            math.dist(candidate, existing) < float(minimum_pairwise_separation_m)
            for existing, _ in chosen
        ):
            rejected["separation"] += 1
            continue
        chosen.append((candidate, evidence))
        if len(chosen) == target_count:
            break
    if len(chosen) != target_count:
        raise TargetSamplingError(
            "unable to sample the requested private target cohort without relaxing gates: "
            f"requested={target_count}, accepted={len(chosen)}, rejected={rejected}"
        )
    return tuple(
        {
            "target_id": f"private-target-{index:02d}",
            "position_w_m": list(position),
            "radius_m": float(radius_m),
            "visibility_bucket": visibility_bucket,
            "visibility_evidence": evidence.as_dict(),
        }
        for index, (position, evidence) in enumerate(chosen)
    )


@dataclass(frozen=True)
class TargetVisibilityEvidence:
    eligible_witness_count: int
    visible_witness_count: int
    blocked_witness_count: int
    undersized_witness_count: int
    nearest_visible_distance_m: float | None
    maximum_projected_instance_pixels: float
    visibility_bucket: str | None
    tracking_envelope_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TARGET_VISIBILITY_GEOMETRY_SCHEMA,
            "eligible_witness_count": self.eligible_witness_count,
            "visible_witness_count": self.visible_witness_count,
            "blocked_witness_count": self.blocked_witness_count,
            "undersized_witness_count": self.undersized_witness_count,
            "nearest_visible_distance_m": self.nearest_visible_distance_m,
            "maximum_projected_instance_pixels": self.maximum_projected_instance_pixels,
            "visibility_bucket": self.visibility_bucket,
            "tracking_envelope_m": self.tracking_envelope_m,
        }


def target_visibility_geometry_contract(
    *,
    route_family_id: str,
    routes_w_m: Sequence[Sequence[Sequence[float]]],
    aabb_geometry_sha256: str,
    target_region_id: str,
    visibility_bucket: str,
    tracking_envelope_m: float = TARGET_VISIBILITY_TRACKING_ENVELOPE_M,
    execution_window: Mapping[str, Any] | None = None,
    camera_heading_model: str = CAMERA_HEADING_MODEL_FIXED_INITIAL,
    max_yaw_rate_rad_s: float | None = None,
    yaw_feedback_gain: float | None = None,
    yaw_stability_error_rad: float | None = None,
    yaw_settle_margin_s: float | None = None,
) -> dict[str, Any]:
    if target_region_id not in TARGET_REGIONS_W_M:
        raise ValueError(f"unknown City-Lite target region: {target_region_id}")
    if visibility_bucket not in TARGET_VISIBILITY_BUCKETS:
        raise ValueError(f"unknown City-Lite target visibility bucket: {visibility_bucket}")
    if not math.isfinite(float(tracking_envelope_m)) or float(tracking_envelope_m) < 0.0:
        raise ValueError("tracking_envelope_m must be finite and non-negative")
    window = (
        target_visibility_execution_window()
        if execution_window is None
        else dict(execution_window)
    )
    _validate_target_visibility_execution_window(window)
    _validate_camera_heading_model(
        camera_heading_model,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        yaw_feedback_gain=yaw_feedback_gain,
        yaw_stability_error_rad=yaw_stability_error_rad,
        yaw_settle_margin_s=yaw_settle_margin_s,
    )
    contract = {
        "schema": TARGET_VISIBILITY_GEOMETRY_SCHEMA,
        "evidence": "public-route-execution-window-pinhole-projective-area-structural-aabb-los-tracking-envelope-v3",
        "route_family_id": route_family_id,
        "routes_sha256": canonical_payload_sha256(routes_w_m),
        "aabb_geometry_sha256": aabb_geometry_sha256,
        "target_region_id": target_region_id,
        "visibility_bucket": visibility_bucket,
        "camera": {
            "focal_length_mm": ONBOARD_FOCAL_LENGTH_MM,
            "horizontal_aperture_mm": ONBOARD_HORIZONTAL_APERTURE_MM,
            "image_width": ONBOARD_IMAGE_WIDTH,
            "image_height": ONBOARD_IMAGE_HEIGHT,
            "pitch_down_rad": ONBOARD_PITCH_DOWN_RAD,
            "offset_body_m": list(ONBOARD_CAMERA_OFFSET_BODY_M),
        },
        "minimum_distance_m": TARGET_VISIBILITY_MIN_DISTANCE_M,
        "maximum_distance_m": TARGET_VISIBILITY_MAX_DISTANCE_M,
        "frustum_margin": TARGET_VISIBILITY_FRUSTUM_MARGIN,
        "route_samples_per_segment": TARGET_VISIBILITY_ROUTE_SAMPLES_PER_SEGMENT,
        "minimum_visible_witnesses_per_target": TARGET_VISIBILITY_MIN_WITNESSES,
        "minimum_projected_instance_pixels": TARGET_VISIBILITY_MIN_PROJECTED_INSTANCE_PIXELS,
        "tracking_envelope_m": float(tracking_envelope_m),
        "execution_window": window,
        "native_semantic_gate": {
            "evidence": "native_onboard_semantic_segmentation_instance_v1",
            "minimum_visible_sensor_frames_per_target": TARGET_VISIBILITY_MIN_NATIVE_FRAMES,
            "minimum_visible_instance_pixels": TARGET_VISIBILITY_MIN_NATIVE_PIXELS,
        },
    }
    if camera_heading_model == CAMERA_HEADING_MODEL_SEGMENT_YAW_LIMITED:
        # This v4 form is intentionally not emitted for the historical model.
        # It binds the analytic schedule used during private sampling to the
        # same public yaw controller parameters carried by the T2 protocol.
        contract["schema"] = TARGET_VISIBILITY_GEOMETRY_V4_SCHEMA
        contract["evidence"] = (
            "public-route-execution-window-segment-heading-yaw-settling-"
            "pinhole-projective-area-structural-aabb-los-tracking-envelope-v4"
        )
        contract["camera_heading_contract"] = {
            "model": camera_heading_model,
            "max_yaw_rate_rad_s": float(max_yaw_rate_rad_s),
            "yaw_feedback_gain": float(yaw_feedback_gain),
            "yaw_stability_error_rad": float(yaw_stability_error_rad),
            "yaw_settle_margin_s": float(yaw_settle_margin_s),
        }
    return contract


def target_region_for_positions(
    positions_w_m: Sequence[Sequence[float]],
) -> str | None:
    positions = tuple(tuple(float(value) for value in point) for point in positions_w_m)
    for region_id in (CITY_LITE_TARGET_REGION_A_ID, CITY_LITE_TARGET_REGION_B_ID):
        region = TARGET_REGIONS_W_M[region_id]
        if positions and all(region.contains(point) for point in positions):
            return region_id
    return None


def _initial_heading(route: Sequence[Sequence[float]]) -> float:
    for start, end in zip(route, route[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) > 1.0e-6:
            return math.atan2(dy, dx)
    raise ValueError("route has no horizontal segment")


def _validate_camera_heading_model(
    camera_heading_model: str,
    *,
    max_yaw_rate_rad_s: float | None,
    yaw_feedback_gain: float | None,
    yaw_stability_error_rad: float | None,
    yaw_settle_margin_s: float | None,
) -> None:
    if camera_heading_model not in CAMERA_HEADING_MODELS:
        raise ValueError("unknown camera heading model")
    values = (
        max_yaw_rate_rad_s,
        yaw_feedback_gain,
        yaw_stability_error_rad,
        yaw_settle_margin_s,
    )
    if camera_heading_model == CAMERA_HEADING_MODEL_FIXED_INITIAL:
        if any(value is not None for value in values):
            raise ValueError("fixed initial heading cannot carry yaw-settling fields")
        return
    names = (
        "max_yaw_rate_rad_s",
        "yaw_feedback_gain",
        "yaw_stability_error_rad",
        "yaw_settle_margin_s",
    )
    for name, value in zip(names, values, strict=True):
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive for segment yaw model")
    if float(yaw_stability_error_rad) >= math.pi:
        raise ValueError("yaw_stability_error_rad must be below pi")


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _segment_heading(
    start: Sequence[float], end: Sequence[float], *, fallback: float
) -> float:
    delta_x = float(end[0]) - float(start[0])
    delta_y = float(end[1]) - float(start[1])
    return math.atan2(delta_y, delta_x) if math.hypot(delta_x, delta_y) > 1.0e-6 else fallback


def _yaw_stable_after_s(
    initial_yaw: float,
    target_yaw: float,
    *,
    max_yaw_rate_rad_s: float,
    yaw_feedback_gain: float,
    yaw_stability_error_rad: float,
    yaw_settle_margin_s: float,
) -> float:
    """Bound the ideal public yaw controller's transient after a route turn.

    The route policy uses a proportional yaw command followed by a public yaw
    rate clamp.  This is an analytic *sampling* exclusion window, not a claim
    that the physical body has converged; the native semantic gate remains
    authoritative after rollout.
    """

    error = abs(_wrap_angle_rad(target_yaw - initial_yaw))
    tolerance = float(yaw_stability_error_rad)
    if error <= tolerance:
        return float(yaw_settle_margin_s)
    rate = float(max_yaw_rate_rad_s)
    gain = float(yaw_feedback_gain)
    saturated_threshold = rate / gain
    if error > saturated_threshold:
        saturated_s = (error - saturated_threshold) / rate
        exponential_start = saturated_threshold
    else:
        saturated_s = 0.0
        exponential_start = error
    exponential_s = math.log(exponential_start / tolerance) / gain
    return saturated_s + max(0.0, exponential_s) + float(yaw_settle_margin_s)


def _validate_target_visibility_execution_window(window: Mapping[str, Any]) -> None:
    if not isinstance(window, Mapping):
        raise ValueError("target visibility execution window must be an object")
    required = (
        "schema",
        "dt_s",
        "warmup_steps",
        "rollout_steps",
        "capture_stride",
        "first_retained_sensor_time_s",
        "last_retained_sensor_time_s",
        "waypoint_segment_seconds",
    )
    missing = [key for key in required if key not in window]
    if missing:
        raise ValueError(
            "target visibility execution window is missing required fields: "
            + ", ".join(missing)
        )
    try:
        expected = target_visibility_execution_window(
            dt_s=float(window["dt_s"]),
            warmup_steps=window["warmup_steps"],
            rollout_steps=window["rollout_steps"],
            capture_stride=window["capture_stride"],
            waypoint_segment_seconds=float(window["waypoint_segment_seconds"]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target visibility execution window has invalid field values") from exc
    if dict(window) != expected:
        raise ValueError("target visibility execution window is invalid or non-canonical")


def _route_camera_witnesses(
    routes_w_m: Sequence[Sequence[Sequence[float]]],
    *,
    execution_window: Mapping[str, Any],
    camera_heading_model: str = CAMERA_HEADING_MODEL_FIXED_INITIAL,
    max_yaw_rate_rad_s: float | None = None,
    yaw_feedback_gain: float | None = None,
    yaw_stability_error_rad: float | None = None,
    yaw_settle_margin_s: float | None = None,
) -> tuple[tuple[tuple[float, float, float], float], ...]:
    witnesses: list[tuple[tuple[float, float, float], float]] = []
    _validate_target_visibility_execution_window(execution_window)
    _validate_camera_heading_model(
        camera_heading_model,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        yaw_feedback_gain=yaw_feedback_gain,
        yaw_stability_error_rad=yaw_stability_error_rad,
        yaw_settle_margin_s=yaw_settle_margin_s,
    )
    first_sensor_time_s = float(execution_window["first_retained_sensor_time_s"])
    last_sensor_time_s = float(execution_window["last_retained_sensor_time_s"])
    segment_duration_s = float(execution_window["waypoint_segment_seconds"])
    for route in routes_w_m:
        initial_yaw = _initial_heading(route)
        previous_yaw = initial_yaw
        for segment_id, (start, end) in enumerate(zip(route, route[1:])):
            segment_yaw = (
                initial_yaw
                if camera_heading_model == CAMERA_HEADING_MODEL_FIXED_INITIAL
                else _segment_heading(start, end, fallback=previous_yaw)
            )
            segment_start_s = float(segment_id) * segment_duration_s
            segment_end_s = segment_start_s + segment_duration_s
            yaw_stable_after_s = 0.0
            if (
                camera_heading_model == CAMERA_HEADING_MODEL_SEGMENT_YAW_LIMITED
                and segment_id > 0
            ):
                yaw_stable_after_s = _yaw_stable_after_s(
                    previous_yaw,
                    segment_yaw,
                    max_yaw_rate_rad_s=float(max_yaw_rate_rad_s),
                    yaw_feedback_gain=float(yaw_feedback_gain),
                    yaw_stability_error_rad=float(yaw_stability_error_rad),
                    yaw_settle_margin_s=float(yaw_settle_margin_s),
                )
            overlap_start_s = max(
                first_sensor_time_s, segment_start_s + yaw_stable_after_s
            )
            overlap_end_s = min(last_sensor_time_s, segment_end_s)
            if overlap_end_s < overlap_start_s:
                previous_yaw = segment_yaw
                continue
            start_fraction = (overlap_start_s - segment_start_s) / segment_duration_s
            end_fraction = (overlap_end_s - segment_start_s) / segment_duration_s
            fractions = tuple(
                start_fraction
                + (end_fraction - start_fraction)
                * index
                / float(TARGET_VISIBILITY_ROUTE_SAMPLES_PER_SEGMENT - 1)
                for index in range(TARGET_VISIBILITY_ROUTE_SAMPLES_PER_SEGMENT)
            )
            for fraction_id, fraction in enumerate(fractions):
                if segment_id > 0 and fraction_id == 0 and start_fraction <= 1.0e-12:
                    continue
                body = tuple(
                    float(start[axis])
                    + fraction * (float(end[axis]) - float(start[axis]))
                    for axis in range(3)
                )
                camera = (
                    body[0] + ONBOARD_CAMERA_OFFSET_BODY_M[0] * math.cos(segment_yaw),
                    body[1] + ONBOARD_CAMERA_OFFSET_BODY_M[0] * math.sin(segment_yaw),
                    body[2] + ONBOARD_CAMERA_OFFSET_BODY_M[2],
                )
                witnesses.append((camera, segment_yaw))
            previous_yaw = segment_yaw
    return tuple(witnesses)


def _inside_camera_frustum(
    camera_w_m: Sequence[float], yaw_rad: float, target_w_m: Sequence[float]
) -> tuple[bool, float]:
    delta = tuple(float(target_w_m[axis]) - float(camera_w_m[axis]) for axis in range(3))
    distance = math.sqrt(sum(value * value for value in delta))
    if not TARGET_VISIBILITY_MIN_DISTANCE_M <= distance <= TARGET_VISIBILITY_MAX_DISTANCE_M:
        return False, distance
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_pitch, sin_pitch = math.cos(ONBOARD_PITCH_DOWN_RAD), math.sin(
        ONBOARD_PITCH_DOWN_RAD
    )
    forward = (cos_pitch * cos_yaw, cos_pitch * sin_yaw, -sin_pitch)
    right = (-sin_yaw, cos_yaw, 0.0)
    up = (sin_pitch * cos_yaw, sin_pitch * sin_yaw, cos_pitch)
    optical = sum(delta[axis] * forward[axis] for axis in range(3))
    if optical <= 0.0:
        return False, distance
    horizontal = math.atan2(
        sum(delta[axis] * right[axis] for axis in range(3)), optical
    )
    vertical = math.atan2(
        sum(delta[axis] * up[axis] for axis in range(3)), optical
    )
    horizontal_half_fov = math.atan(
        ONBOARD_HORIZONTAL_APERTURE_MM / (2.0 * ONBOARD_FOCAL_LENGTH_MM)
    )
    vertical_aperture = (
        ONBOARD_HORIZONTAL_APERTURE_MM * ONBOARD_IMAGE_HEIGHT / ONBOARD_IMAGE_WIDTH
    )
    vertical_half_fov = math.atan(vertical_aperture / (2.0 * ONBOARD_FOCAL_LENGTH_MM))
    margin = TARGET_VISIBILITY_FRUSTUM_MARGIN
    return bool(
        abs(horizontal) <= horizontal_half_fov * margin
        and abs(vertical) <= vertical_half_fov * margin
    ), distance


def _tracking_camera_probes(
    camera_w_m: Sequence[float], tracking_envelope_m: float
) -> tuple[tuple[float, float, float], ...]:
    """Return the nominal camera and the six axis-aligned error probes."""

    envelope = float(tracking_envelope_m)
    if not math.isfinite(envelope) or envelope < 0.0:
        raise ValueError("tracking_envelope_m must be finite and non-negative")
    offsets = (
        (0.0, 0.0, 0.0),
        (envelope, 0.0, 0.0),
        (-envelope, 0.0, 0.0),
        (0.0, envelope, 0.0),
        (0.0, -envelope, 0.0),
        (0.0, 0.0, envelope),
        (0.0, 0.0, -envelope),
    )
    return tuple(
        tuple(float(camera_w_m[axis]) + offset[axis] for axis in range(3))
        for offset in offsets
    )


def measure_target_visibility(
    target_w_m: Sequence[float],
    *,
    routes_w_m: Sequence[Sequence[Sequence[float]]],
    structural_aabbs: Sequence[AABB | Mapping[str, Any]],
    radius_m: float = 0.30,
    tracking_envelope_m: float = TARGET_VISIBILITY_TRACKING_ENVELOPE_M,
    execution_window: Mapping[str, Any] | None = None,
    camera_heading_model: str = CAMERA_HEADING_MODEL_FIXED_INITIAL,
    max_yaw_rate_rad_s: float | None = None,
    yaw_feedback_gain: float | None = None,
    yaw_stability_error_rad: float | None = None,
    yaw_settle_margin_s: float | None = None,
) -> TargetVisibilityEvidence:
    if not math.isfinite(float(radius_m)) or float(radius_m) <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if not math.isfinite(float(tracking_envelope_m)) or float(tracking_envelope_m) < 0.0:
        raise ValueError("tracking_envelope_m must be finite and non-negative")
    boxes = tuple(coerce_aabb(value) for value in structural_aabbs)
    window = (
        target_visibility_execution_window()
        if execution_window is None
        else dict(execution_window)
    )
    _validate_target_visibility_execution_window(window)
    _validate_camera_heading_model(
        camera_heading_model,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        yaw_feedback_gain=yaw_feedback_gain,
        yaw_stability_error_rad=yaw_stability_error_rad,
        yaw_settle_margin_s=yaw_settle_margin_s,
    )
    focal_length_pixels = (
        ONBOARD_FOCAL_LENGTH_MM / ONBOARD_HORIZONTAL_APERTURE_MM * ONBOARD_IMAGE_WIDTH
    )
    eligible = visible = blocked = undersized = 0
    nearest: float | None = None
    maximum_projected_pixels = 0.0
    for camera, yaw in _route_camera_witnesses(
        routes_w_m,
        execution_window=window,
        camera_heading_model=camera_heading_model,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        yaw_feedback_gain=yaw_feedback_gain,
        yaw_stability_error_rad=yaw_stability_error_rad,
        yaw_settle_margin_s=yaw_settle_margin_s,
    ):
        probe_results = tuple(
            _inside_camera_frustum(probe, yaw, target_w_m)
            for probe in _tracking_camera_probes(camera, float(tracking_envelope_m))
        )
        if not all(in_frustum for in_frustum, _distance in probe_results):
            continue
        eligible += 1
        distances = tuple(distance for _in_frustum, distance in probe_results)
        occluded = any(
            segment_intersects_aabb(probe, target_w_m, box, clearance_m=0.0)
            for probe in _tracking_camera_probes(camera, float(tracking_envelope_m))
            for box in boxes
        )
        if occluded:
            blocked += 1
        else:
            projected_pixels_by_probe = tuple(
                math.pi
                * (focal_length_pixels * float(radius_m) / distance) ** 2
                for distance in distances
            )
            maximum_projected_pixels = max(
                maximum_projected_pixels, max(projected_pixels_by_probe)
            )
            if any(
                pixels < TARGET_VISIBILITY_MIN_PROJECTED_INSTANCE_PIXELS
                for pixels in projected_pixels_by_probe
            ):
                undersized += 1
                continue
            visible += 1
            nearest_distance = min(distances)
            nearest = (
                nearest_distance
                if nearest is None
                else min(nearest, nearest_distance)
            )
    if visible < TARGET_VISIBILITY_MIN_WITNESSES:
        bucket = None
    elif blocked:
        bucket = TARGET_VISIBILITY_PARTIAL
    else:
        bucket = TARGET_VISIBILITY_DIRECT
    return TargetVisibilityEvidence(
        eligible,
        visible,
        blocked,
        undersized,
        nearest,
        maximum_projected_pixels,
        bucket,
        float(tracking_envelope_m),
    )


def verify_target_visibility_bucket(
    positions_w_m: Sequence[Sequence[float]],
    *,
    requested_bucket: str,
    routes_w_m: Sequence[Sequence[Sequence[float]]],
    structural_aabbs: Sequence[AABB | Mapping[str, Any]],
    radii_m: Sequence[float] | None = None,
    tracking_envelope_m: float = TARGET_VISIBILITY_TRACKING_ENVELOPE_M,
    execution_window: Mapping[str, Any] | None = None,
    camera_heading_model: str = CAMERA_HEADING_MODEL_FIXED_INITIAL,
    max_yaw_rate_rad_s: float | None = None,
    yaw_feedback_gain: float | None = None,
    yaw_stability_error_rad: float | None = None,
    yaw_settle_margin_s: float | None = None,
) -> tuple[bool, tuple[TargetVisibilityEvidence, ...]]:
    if radii_m is None:
        radii_m = (0.30,) * len(positions_w_m)
    if len(radii_m) != len(positions_w_m):
        raise ValueError("radii_m must align with target positions")
    evidence = tuple(
        measure_target_visibility(
            position,
            routes_w_m=routes_w_m,
            structural_aabbs=structural_aabbs,
            radius_m=float(radius_m),
            tracking_envelope_m=float(tracking_envelope_m),
            execution_window=execution_window,
            camera_heading_model=camera_heading_model,
            max_yaw_rate_rad_s=max_yaw_rate_rad_s,
            yaw_feedback_gain=yaw_feedback_gain,
            yaw_stability_error_rad=yaw_stability_error_rad,
            yaw_settle_margin_s=yaw_settle_margin_s,
        )
        for position, radius_m in zip(positions_w_m, radii_m, strict=True)
    )
    return bool(
        requested_bucket in TARGET_VISIBILITY_BUCKETS
        and evidence
        and all(item.visibility_bucket == requested_bucket for item in evidence)
    ), evidence
