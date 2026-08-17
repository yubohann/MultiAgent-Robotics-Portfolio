"""Fail-closed runtime safety checks for City-Lite Isaac CF2X captures.

The functions in this module deliberately do not import Isaac Sim or Torch.
That keeps the physical guard testable before Kit starts and makes the capture
loop's safety decision depend on the same frozen geometry contracts used by
independent validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import math
import struct
from typing import Any, Mapping, Sequence

from .citylite_scene import (
    AGENT_COUNT,
    AABB,
    CITY_LITE_FLIGHT_VOLUME_W_M,
    ROUTE_CLEARANCE_M,
    aabb_geometry_sha256,
    segment_intersects_aabb,
)


RUNTIME_SAFETY_SCHEMA = "org.rivermark.isaac-runtime-safety-guard.v2"
RUNTIME_SAFETY_TRACE_SCHEMA = "org.rivermark.isaac-runtime-safety-trace.v2"
RUNTIME_SAFETY_TRACE_RELATIVE_PATH = "sensors/runtime_safety.npz"
SENSOR_PHASE_TRACE_SCHEMA = "org.rivermark.isaac-sensor-phase-trace.v1"
SENSOR_PHASE_TRACE_RELATIVE_PATH = "sensors/sensor_phase.npz"
# A capture-frame trace records the order in which its retained sensor sample
# was produced.  Numeric codes keep the NPZ evidence compact and make the
# independent validator reject reordered self-reported strings.
SENSOR_PHASE_EVENT_CODES = {
    "command_write": 1,
    "simulation_step": 2,
    "state_update": 3,
    "safety_contact_read": 4,
    "camera_pose_update": 5,
    "render": 6,
    "rgbd_lidar_imu_read": 7,
    "retained_contact_read": 8,
    "storage": 9,
}
SENSOR_PHASE_EVENT_SEQUENCE = tuple(SENSOR_PHASE_EVENT_CODES.values())
SENSOR_PHASE_SENSOR_NAMES = ("rgb", "depth", "semantic", "lidar", "imu", "contact")


def sensor_phase_array_digest(value: Any) -> bytes:
    """Return the raw contiguous-byte SHA-256 used by phase evidence.

    Isaac tensors and validator NumPy arrays must have the same byte-level
    representation.  Conversion is intentionally local and does not import
    Isaac or Torch at module import time.
    """

    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    else:
        import numpy as np

        value = np.ascontiguousarray(value)
    if getattr(value.dtype, "hasobject", False):
        raise TypeError("sensor phase digest does not accept object arrays")
    return hashlib.sha256(memoryview(value).cast("B")).digest()
RUNTIME_SAFETY_PHASE_CODES = {
    "post_reset": 0,
    "warmup": 1,
    "rollout": 2,
}
RUNTIME_SAFETY_FRAME_OUTCOME_CODES = {
    "passed": 0,
    "aborted": 1,
}
# The upstream CF2X City-Lite audit uses this radius for the physical body
# envelope.  It applies only to the flight-volume boundary; AABB clearance is
# independently frozen as a 0.85 m root-center sweep.
CF2X_RUNTIME_GUARD_RADIUS_M = 0.08
INTER_AGENT_BODY_ENVELOPE_SEPARATION_M = 2.0 * CF2X_RUNTIME_GUARD_RADIUS_M
# The md_qd_swarm E0-hard safety interface uses a 0.32 m operational
# separation, which retains a 0.16 m margin beyond the two 0.08 m CF2X body
# envelopes.  This is intentionally stronger than merely rejecting overlap.
INTER_AGENT_MINIMUM_CENTER_SEPARATION_M = 0.32
INTER_AGENT_SAFETY_PROVENANCE = "md_qd_swarm_e0_hard_safety_min_separation_m"
INTER_AGENT_PAIR_COUNT = AGENT_COUNT * (AGENT_COUNT - 1) // 2
CONTACT_ABORT_FORCE_N = 0.01
# Isaac's contact tensor is float32.  Comparing its values after conversion to
# Python float against the decimal literal 0.01 would make exactly-float32
# 0.01 appear smaller than the stated threshold.  Freeze the actual comparison
# boundary and publish it with the guard receipt.
CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N = struct.unpack(
    "!f", struct.pack("!f", CONTACT_ABORT_FORCE_N)
)[0]


def physics_time_ns(physics_step: int, dt_s: float) -> int:
    """Return the canonical logical simulation time for one physics frame.

    The trace records the post-reset state as frame zero at zero nanoseconds.
    Every later frame is the state after its numbered ``dt_s`` simulation
    interval.  Decimal half-even rounding avoids platform-dependent binary
    floating-point drift while preserving the familiar ``round`` contract.
    """

    if (
        isinstance(physics_step, bool)
        or not isinstance(physics_step, int)
        or physics_step < 0
    ):
        raise ValueError("physics_step must be a non-negative integer")
    if isinstance(dt_s, bool):
        raise ValueError("dt_s must be a positive finite number")
    try:
        decimal_dt_s = Decimal(str(dt_s))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("dt_s must be a positive finite number") from exc
    if not decimal_dt_s.is_finite() or decimal_dt_s <= 0:
        raise ValueError("dt_s must be a positive finite number")
    return int(
        (Decimal(physics_step) * decimal_dt_s * Decimal(1_000_000_000)).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
    )


class RuntimeSafetyAbort(RuntimeError):
    """A physical state violates a non-recoverable City-Lite safety rule."""

    def __init__(self, violation: Mapping[str, Any]):
        self.violation = dict(violation)
        super().__init__(
            "City-Lite runtime safety guard aborted: "
            + str(self.violation.get("kind", "unknown_violation"))
        )


@dataclass(frozen=True)
class RuntimeSafetyCheck:
    """One successful post-reset, warmup, or rollout safety evaluation."""

    agent_center_checks: int
    initial_point_geometry_checks: int
    swept_segments_checked: int
    inter_agent_pair_checks: int
    minimum_inter_agent_swept_separation_m: float
    contact_samples_checked: int
    max_contact_force_n: float


def _minimum_inter_agent_swept_separation(
    previous: tuple[tuple[float, float, float], ...] | None,
    current: tuple[tuple[float, float, float], ...],
) -> tuple[float, int, int, float]:
    """Return the minimum simultaneous segment separation over all CF2X pairs.

    Each pair is evaluated over the same normalized physics-step interval.  A
    check of only the two endpoints misses two agents exchanging positions in
    one step, so this uses the analytic closest point of their relative
    segment.  At post-reset ``previous`` is ``None`` and the segments collapse
    to the current points.
    """

    minimum = math.inf
    minimum_left = -1
    minimum_right = -1
    minimum_time = 0.0
    for left in range(AGENT_COUNT - 1):
        left_start = current[left] if previous is None else previous[left]
        left_end = current[left]
        for right in range(left + 1, AGENT_COUNT):
            right_start = current[right] if previous is None else previous[right]
            right_end = current[right]
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
                time = 0.0
            else:
                time = -sum(
                    relative_start[axis] * relative_delta[axis] for axis in range(3)
                ) / denominator
                time = min(1.0, max(0.0, time))
            separation = math.sqrt(
                sum(
                    (relative_start[axis] + time * relative_delta[axis]) ** 2
                    for axis in range(3)
                )
            )
            if separation < minimum:
                minimum = separation
                minimum_left = left
                minimum_right = right
                minimum_time = time
    return minimum, minimum_left, minimum_right, minimum_time


def record_runtime_safety_check(
    guard: dict[str, Any], check: RuntimeSafetyCheck, *, phase: str
) -> None:
    """Account for one successful guard evaluation in its public receipt.

    The capture loop owns the raw evidence trace.  This helper only updates
    deterministic counters so the receipt cannot accidentally drift from the
    code path that made the safety decision.
    """

    if phase not in RUNTIME_SAFETY_PHASE_CODES:
        raise ValueError(f"unknown runtime safety phase: {phase}")
    if guard.get("status") != "running":
        raise ValueError("cannot record a safety check after the guard stopped")
    checks = guard.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("runtime safety guard checks must be mutable")
    if check.inter_agent_pair_checks != INTER_AGENT_PAIR_COUNT:
        raise ValueError("runtime safety check must cover every CF2X pair")
    if (
        not math.isfinite(float(check.minimum_inter_agent_swept_separation_m))
        or float(check.minimum_inter_agent_swept_separation_m)
        <= INTER_AGENT_MINIMUM_CENTER_SEPARATION_M
    ):
        raise ValueError("runtime safety pairwise separation must remain above the CF2X envelope")
    for key, value in (
        ("agent_center_checks", check.agent_center_checks),
        ("post_reset_point_geometry_checks", check.initial_point_geometry_checks),
        ("swept_segments_checked", check.swept_segments_checked),
        ("inter_agent_pair_checks", check.inter_agent_pair_checks),
        ("contact_samples_checked", check.contact_samples_checked),
    ):
        checks[key] = int(checks.get(key, 0)) + int(value)
    if phase == "post_reset":
        checks["post_reset_agent_center_checks"] = int(
            checks.get("post_reset_agent_center_checks", 0)
        ) + int(check.agent_center_checks)
        checks["post_reset_inter_agent_pair_checks"] = int(
            checks.get("post_reset_inter_agent_pair_checks", 0)
        ) + int(check.inter_agent_pair_checks)
    elif phase == "warmup":
        checks["warmup_physics_steps_checked"] = int(
            checks.get("warmup_physics_steps_checked", 0)
        ) + 1
    else:
        checks["rollout_physics_steps_checked"] = int(
            checks.get("rollout_physics_steps_checked", 0)
        ) + 1
    maximum = max(float(checks.get("max_contact_force_n", 0.0)), float(check.max_contact_force_n))
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("runtime safety contact maximum must remain finite and nonnegative")
    checks["max_contact_force_n"] = maximum
    previous_minimum = checks.get("minimum_inter_agent_swept_separation_m")
    if previous_minimum is None:
        minimum_separation = float(check.minimum_inter_agent_swept_separation_m)
    else:
        minimum_separation = min(
            float(previous_minimum), float(check.minimum_inter_agent_swept_separation_m)
        )
    checks["minimum_inter_agent_swept_separation_m"] = minimum_separation


def record_runtime_safety_abort(
    guard: dict[str, Any], error: RuntimeSafetyAbort
) -> None:
    """Freeze the first non-recoverable runtime safety violation."""

    if guard.get("status") != "running":
        raise ValueError("runtime safety guard already stopped")
    violation = dict(error.violation)
    checks = guard.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("runtime safety guard checks must be mutable")
    if violation.get("kind") == "contact_force_violation":
        checks["contact_abort_count"] = int(checks.get("contact_abort_count", 0)) + 1
    guard["status"] = "aborted"
    guard["first_violation"] = violation


def bind_runtime_safety_trace_evidence(
    guard: dict[str, Any], *, trace_sha256: str, physics_frame_count: int
) -> None:
    """Bind either a passing or aborted guard to its recorded trace."""

    if (
        not isinstance(trace_sha256, str)
        or len(trace_sha256) != 64
        or any(character not in "0123456789abcdef" for character in trace_sha256)
    ):
        raise ValueError("runtime safety trace requires a SHA-256 digest")
    if (
        isinstance(physics_frame_count, bool)
        or not isinstance(physics_frame_count, int)
        or physics_frame_count < 1
    ):
        raise ValueError("runtime safety trace requires at least one frame")
    evidence = guard.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("runtime safety guard evidence must be mutable")
    evidence["sha256"] = trace_sha256
    evidence["physics_frame_count"] = physics_frame_count


def finalize_runtime_safety_guard(
    guard: dict[str, Any], *, trace_sha256: str, physics_frame_count: int
) -> None:
    """Bind a passing guard to one immutable full-step trace artifact."""

    if guard.get("status") != "running":
        raise ValueError("only a running runtime safety guard can pass")
    bind_runtime_safety_trace_evidence(
        guard,
        trace_sha256=trace_sha256,
        physics_frame_count=physics_frame_count,
    )
    guard["status"] = "passed"


def _context(*, phase: str, physics_step: int) -> dict[str, Any]:
    if not isinstance(phase, str) or not phase:
        return {"phase": "invalid", "physics_step": int(physics_step)}
    return {"phase": phase, "physics_step": int(physics_step)}


def _finite_coordinate(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite coordinate")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite coordinate") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite coordinate")
    return number


def _positions(value: Any, *, label: str) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {AGENT_COUNT} xyz rows")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must contain {AGENT_COUNT} xyz rows") from exc
    if len(rows) != AGENT_COUNT:
        raise ValueError(f"{label} must contain {AGENT_COUNT} xyz rows")
    result: list[tuple[float, float, float]] = []
    for agent_id, raw_row in enumerate(rows):
        if isinstance(raw_row, (str, bytes)):
            raise ValueError(f"{label}[{agent_id}] must be xyz")
        try:
            row = tuple(raw_row)
        except TypeError as exc:
            raise ValueError(f"{label}[{agent_id}] must be xyz") from exc
        if len(row) != 3:
            raise ValueError(f"{label}[{agent_id}] must be xyz")
        result.append(
            tuple(
                _finite_coordinate(component, label=f"{label}[{agent_id}][{axis}]")
                for axis, component in enumerate(row)
            )
        )
    return tuple(result)


def _contact_forces(value: Any) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Normalize exactly [8, 1, 3] root-body net-contact forces."""

    if isinstance(value, (str, bytes)):
        raise ValueError("contact forces must be [8,1,3]")
    try:
        agents = tuple(value)
    except TypeError as exc:
        raise ValueError("contact forces must be [8,1,3]") from exc
    if len(agents) != AGENT_COUNT:
        raise ValueError("contact forces must be [8,1,3]")
    normalized: list[tuple[tuple[float, float, float], ...]] = []
    for agent_id, raw_bodies in enumerate(agents):
        if isinstance(raw_bodies, (str, bytes)):
            raise ValueError(f"contact forces agent {agent_id} must contain one body")
        try:
            bodies = tuple(raw_bodies)
        except TypeError as exc:
            raise ValueError(
                f"contact forces agent {agent_id} must contain one body"
            ) from exc
        if len(bodies) != 1:
            raise ValueError(f"contact forces agent {agent_id} must contain one body")
        body_rows: list[tuple[float, float, float]] = []
        for body_id, raw_force in enumerate(bodies):
            if isinstance(raw_force, (str, bytes)):
                raise ValueError(
                    f"contact forces agent {agent_id} body {body_id} must be xyz"
                )
            try:
                force = tuple(raw_force)
            except TypeError as exc:
                raise ValueError(
                    f"contact forces agent {agent_id} body {body_id} must be xyz"
                ) from exc
            if len(force) != 3:
                raise ValueError(
                    f"contact forces agent {agent_id} body {body_id} must be xyz"
                )
            body_rows.append(
                tuple(
                    _finite_coordinate(
                        component,
                        label=f"contact forces[{agent_id}][{body_id}][{axis}]",
                    )
                    for axis, component in enumerate(force)
                )
            )
        normalized.append(tuple(body_rows))
    return tuple(normalized)


def _abort(kind: str, *, phase: str, physics_step: int, **details: Any) -> RuntimeSafetyAbort:
    return RuntimeSafetyAbort({"kind": kind, **_context(phase=phase, physics_step=physics_step), **details})


def evaluate_runtime_safety(
    previous_positions_w_m: Sequence[Sequence[Any]] | None,
    current_positions_w_m: Sequence[Sequence[Any]],
    net_contact_forces_w_n: Any,
    structural_aabbs: Sequence[AABB],
    *,
    phase: str,
    physics_step: int,
) -> RuntimeSafetyCheck:
    """Validate one physical frame and raise before a bad trace is retained.

    At post-reset, pass ``None`` as ``previous_positions_w_m``.  This verifies
    the root centers and their current AABB separation without pretending a
    zero-length point check is a physical swept segment.
    """

    try:
        current = _positions(current_positions_w_m, label="current root positions")
    except ValueError as exc:
        raise _abort(
            "invalid_root_positions",
            phase=phase,
            physics_step=physics_step,
            detail=str(exc),
        ) from exc
    if not structural_aabbs:
        raise _abort(
            "missing_structural_aabbs",
            phase=phase,
            physics_step=physics_step,
        )
    try:
        boxes = tuple(AABB(box.minimum, box.maximum, box.source_prim, box.category) for box in structural_aabbs)
    except (AttributeError, TypeError, ValueError) as exc:
        raise _abort(
            "invalid_structural_aabbs",
            phase=phase,
            physics_step=physics_step,
            detail=str(exc),
        ) from exc

    previous: tuple[tuple[float, float, float], ...] | None = None
    if previous_positions_w_m is not None:
        try:
            previous = _positions(previous_positions_w_m, label="previous root positions")
        except ValueError as exc:
            raise _abort(
                "invalid_previous_root_positions",
                phase=phase,
                physics_step=physics_step,
                detail=str(exc),
            ) from exc

    for agent_id, position in enumerate(current):
        if not CITY_LITE_FLIGHT_VOLUME_W_M.contains(
            position, margin_m=CF2X_RUNTIME_GUARD_RADIUS_M
        ):
            raise _abort(
                "flight_volume_violation",
                phase=phase,
                physics_step=physics_step,
                agent_id=agent_id,
                current_pos_w_m=list(position),
                agent_center_radius_m=CF2X_RUNTIME_GUARD_RADIUS_M,
            )

        start = position if previous is None else previous[agent_id]
        for box in boxes:
            if segment_intersects_aabb(
                start,
                position,
                box,
                clearance_m=ROUTE_CLEARANCE_M,
            ):
                raise _abort(
                    "structural_aabb_clearance_violation",
                    phase=phase,
                    physics_step=physics_step,
                    agent_id=agent_id,
                    previous_pos_w_m=list(start),
                    current_pos_w_m=list(position),
                    source_prim=box.source_prim,
                    category=box.category,
                    swept_aabb_clearance_m=ROUTE_CLEARANCE_M,
                )

    (
        minimum_pair_separation,
        left_agent_id,
        right_agent_id,
        closest_time,
    ) = _minimum_inter_agent_swept_separation(previous, current)
    if minimum_pair_separation <= INTER_AGENT_MINIMUM_CENTER_SEPARATION_M:
        left_previous = current[left_agent_id] if previous is None else previous[left_agent_id]
        right_previous = current[right_agent_id] if previous is None else previous[right_agent_id]
        raise _abort(
            "inter_agent_swept_separation_violation",
            phase=phase,
            physics_step=physics_step,
            left_agent_id=left_agent_id,
            right_agent_id=right_agent_id,
            left_previous_pos_w_m=list(left_previous),
            left_current_pos_w_m=list(current[left_agent_id]),
            right_previous_pos_w_m=list(right_previous),
            right_current_pos_w_m=list(current[right_agent_id]),
            closest_segment_time=closest_time,
            minimum_center_separation_m=minimum_pair_separation,
            required_center_separation_m=INTER_AGENT_MINIMUM_CENTER_SEPARATION_M,
        )

    try:
        contact = _contact_forces(net_contact_forces_w_n)
    except ValueError as exc:
        raise _abort(
            "invalid_contact_sensor_data",
            phase=phase,
            physics_step=physics_step,
            detail=str(exc),
        ) from exc
    maximum_force = 0.0
    for agent_id, bodies in enumerate(contact):
        for body_id, force in enumerate(bodies):
            force_norm = math.sqrt(sum(component * component for component in force))
            maximum_force = max(maximum_force, force_norm)
            if force_norm >= CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N:
                raise _abort(
                    "contact_force_violation",
                    phase=phase,
                    physics_step=physics_step,
                    agent_id=agent_id,
                    body_id=body_id,
                    net_force_w_n=list(force),
                    force_norm_n=force_norm,
                    force_abort_threshold_n=CONTACT_ABORT_FORCE_N,
                    force_abort_float32_cutoff_n=CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
                )

    return RuntimeSafetyCheck(
        agent_center_checks=AGENT_COUNT,
        initial_point_geometry_checks=AGENT_COUNT if previous is None else 0,
        swept_segments_checked=0 if previous is None else AGENT_COUNT,
        inter_agent_pair_checks=INTER_AGENT_PAIR_COUNT,
        minimum_inter_agent_swept_separation_m=minimum_pair_separation,
        contact_samples_checked=1,
        max_contact_force_n=maximum_force,
    )


def runtime_safety_receipt_template(
    structural_aabbs: Sequence[AABB],
    *,
    contact_prim_expression: str,
    physics_dt_s: float,
) -> dict[str, Any]:
    """Create the immutable portion of a capture's public safety receipt."""

    if not structural_aabbs:
        raise ValueError("runtime safety receipt requires structural AABBs")
    if not math.isfinite(float(physics_dt_s)) or float(physics_dt_s) <= 0.0:
        raise ValueError("runtime safety receipt requires a positive physics dt")
    return {
        "schema": RUNTIME_SAFETY_SCHEMA,
        "enabled": True,
        "fail_closed": True,
        "status": "running",
        "agent_center_radius_m": CF2X_RUNTIME_GUARD_RADIUS_M,
        "flight_volume_m": {
            "x": list(CITY_LITE_FLIGHT_VOLUME_W_M.minimum[0:1] + CITY_LITE_FLIGHT_VOLUME_W_M.maximum[0:1]),
            "y": list(CITY_LITE_FLIGHT_VOLUME_W_M.minimum[1:2] + CITY_LITE_FLIGHT_VOLUME_W_M.maximum[1:2]),
            "z": list(CITY_LITE_FLIGHT_VOLUME_W_M.minimum[2:3] + CITY_LITE_FLIGHT_VOLUME_W_M.maximum[2:3]),
        },
        "structural_aabb_count": len(structural_aabbs),
        "structural_aabb_geometry_sha256": aabb_geometry_sha256(structural_aabbs),
        "swept_aabb_clearance_m": ROUTE_CLEARANCE_M,
        "inter_agent": {
            "pair_count": INTER_AGENT_PAIR_COUNT,
            "body_envelope_separation_m": INTER_AGENT_BODY_ENVELOPE_SEPARATION_M,
            "minimum_swept_center_separation_m": INTER_AGENT_MINIMUM_CENTER_SEPARATION_M,
            "provenance": INTER_AGENT_SAFETY_PROVENANCE,
        },
        "contact": {
            "prim_expression": str(contact_prim_expression),
            "update_period_s": float(physics_dt_s),
            "every_physics_step": True,
            "force_abort_threshold_n": CONTACT_ABORT_FORCE_N,
            "force_abort_float32_cutoff_n": CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
            "body_count": 1,
            "counterpart_attribution": (
                "unfiltered_root_body_net_normal_force; "
                "static_city_guarded_by_structural_aabb_sweep"
            ),
        },
        "evidence": {
            "schema": RUNTIME_SAFETY_TRACE_SCHEMA,
            "path": RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
            "sha256": None,
            "physics_frame_count": 0,
        },
        "checks": {
            "post_reset_agent_center_checks": 0,
            "post_reset_point_geometry_checks": 0,
            "post_reset_inter_agent_pair_checks": 0,
            "warmup_physics_steps_checked": 0,
            "rollout_physics_steps_checked": 0,
            "agent_center_checks": 0,
            "swept_segments_checked": 0,
            "inter_agent_pair_checks": 0,
            "minimum_inter_agent_swept_separation_m": None,
            "contact_samples_checked": 0,
            "max_contact_force_n": 0.0,
            "contact_abort_count": 0,
        },
        "first_violation": None,
    }
