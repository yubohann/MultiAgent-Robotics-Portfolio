"""Shared CF2X execution and geometric guard helpers for HM3D exploration."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import time
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    GuardedPath,
)
from aerocity_method.adapters.hm3d_execution import (
    FragmentExecutionSample,
)
from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
)
from aerocity_method.evaluation.hm3d_safety import (
    ConservativeVoxelClearance,
    TimedPolyline,
    assess_route_tube_separation,
    assess_synchronized_separation,
    required_segment_sample_clearance_m,
)
from aerocity_method.runtime.communication import (
    RelayGraphSnapshot,
    RelayMessage,
    RelayMessageQueue,
    build_range_los_relay_graph,
)
from aerocity_method.runtime.hm3d_belief import (
    PublicRangeObservationFrameOutcome,
    PublicRangeRayOutcome,
)
from aerocity_method.runtime.range_sensing import DENSE_26_RAY_PATTERN
from aerocity_method.runtime.range_sensing import (
    resolve_public_range_directions,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    CONTROLLER_ID as BITCRAZE_LEE_CONTROLLER_ID,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    OFFICIAL_CONTROL_RATE_HZ as BITCRAZE_LEE_OFFICIAL_CONTROL_RATE_HZ,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    POSITION_ERROR_LIMIT as BITCRAZE_LEE_POSITION_ERROR_LIMIT_M,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    SOURCE_COMMIT as BITCRAZE_LEE_SOURCE_COMMIT,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    SOURCE_FILE as BITCRAZE_LEE_SOURCE_FILE,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    SOURCE_URL as BITCRAZE_LEE_SOURCE_URL,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    VELOCITY_ERROR_LIMIT as BITCRAZE_LEE_VELOCITY_ERROR_LIMIT_MPS,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import (
    BitcrazeLeeTracker,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import (
    CONTROLLER_ID as BITCRAZE_MELLINGER_CONTROLLER_ID,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import (
    OFFICIAL_CONTROL_RATE_HZ as BITCRAZE_MELLINGER_OFFICIAL_CONTROL_RATE_HZ,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import (
    SOURCE_COMMIT as BITCRAZE_MELLINGER_SOURCE_COMMIT,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import (
    SOURCE_FILE as BITCRAZE_MELLINGER_SOURCE_FILE,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import (
    SOURCE_URL as BITCRAZE_MELLINGER_SOURCE_URL,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import (
    BitcrazeMellingerTracker,
)
from aerocity_method.runtime.hm3d_team_collaboration import (
    audit_translation_invariant_team_trajectories,
)
from aerocity_method.runtime.hm3d_trajectory import minimum_rest_to_rest_duration_s

DRONE_USD = ROOT.parents[1] / "assets" / "new" / "cf2x.usd"
DEFAULT_COMMUNICATION_CONTRACT = (
    ROOT / "configs" / "external" / "hm3d_p07_communication_contract.json"
)
HOVER_THRUST_PER_ROTOR_N = 0.06935
MAX_THRUST_PER_ROTOR_N = 0.18
CF2X_THRUST_CONSTANT_N_PER_RPS2 = 1.0e-6
CF2X_INITIAL_ROTOR_RPS = math.sqrt(HOVER_THRUST_PER_ROTOR_N / CF2X_THRUST_CONSTANT_N_PER_RPS2)
CF2X_ACTUATOR_INITIALIZATION_ID = "hover-equilibrium-rps-v1"
CF2X_THRUSTER_TAU_INC_RANGE_S = (0.04, 0.06)
CF2X_THRUSTER_TAU_DEC_RANGE_S = (0.02, 0.03)
CONTACT_HARD_FAIL_N = 0.01
FLIGHT_CLEARANCE_M = 0.30
# A guarded geometric centreline is not the physical CF2X root trace.  Reserve
# a measured 0.20 m tracking envelope above the non-negotiable 0.30 m physical
# contract; the real root trace is still checked at every physics step. A
# later short-route outcome reached 0.165 m line deviation even with the
# overdamped tracker, so the arrival tolerance alone was not a safe envelope.
# The
# previous implementation added a second 0.15 m terminal reserve on top of a
# 0.15 m tracking reserve.  That required 0.60 m from every wall even though
# only one of sixteen pre-registered indoor reset poses met it, making a four-
# vehicle episode impossible before any method selected an action.
TRACKING_CLEARANCE_MARGIN_M = 0.20
PLANNED_CONTINUOUS_CLEARANCE_M = FLIGHT_CLEARANCE_M + TRACKING_CLEARANCE_MARGIN_M
# ``CF2X_MIN_INTER_AGENT_SEPARATION_M`` is the physical root-to-root
# requirement.  A joint candidate, however, is evaluated on ideal reference
# paths while two physical vehicles can each use the tracked-path envelope.
# Reserve both envelopes before command issuance.  This is deliberately
# separate from static-mesh clearance: it prevents two individually safe
# tracks from converging in a corridor.
CF2X_MIN_INTER_AGENT_SEPARATION_M = 0.50
TRACKING_INTER_AGENT_SEPARATION_MARGIN_M = 2.0 * TRACKING_CLEARANCE_MARGIN_M
PLANNED_INTER_AGENT_SEPARATION_M = (
    CF2X_MIN_INTER_AGENT_SEPARATION_M + TRACKING_INTER_AGENT_SEPARATION_MARGIN_M
)
# A route whose terminal clearance exceeds the continuous 0.90 m envelope by
# numerical dust can leave the next decision with no recoverable planning
# margin after ordinary tracking error.  This is a command-admission reserve,
# not a relaxation of the physical or continuous separation contracts.
PLANNED_INTER_AGENT_ENDPOINT_MARGIN_M = 0.05
PLANNED_INTER_AGENT_ENDPOINT_SEPARATION_M = (
    PLANNED_INTER_AGENT_SEPARATION_M + PLANNED_INTER_AGENT_ENDPOINT_MARGIN_M
)
# Physical execution needs one legal team action.  Strategy headroom is audited
# separately: killing an otherwise safe episode merely because only one action
# remains confounds task feasibility with selector identifiability.
P07_MINIMUM_EXECUTABLE_CANDIDATES = 1
ROUTE_CLEARANCE_SAMPLE_STEP_M = 0.10
REQUIRED_ROUTE_SAMPLE_CLEARANCE_M = required_segment_sample_clearance_m(
    PLANNED_CONTINUOUS_CLEARANCE_M,
    ROUTE_CLEARANCE_SAMPLE_STEP_M,
)
# The Lipschitz sampling reserve applies only between discrete interior
# samples.  Start and terminal points are queried exactly, so adding another
# half-sample reserve there would reject a 0.40 m-clear endpoint at 0.45 m.
REQUIRED_TERMINAL_CLEARANCE_M = PLANNED_CONTINUOUS_CLEARANCE_M
# The 1 m coarse grid used by ``_grid_route`` is too sparse to certify HM3D
# corridors: the lattice frequently misses the exact-clearance centreline, so
# every connector fails.  This local fallback uses a finer lattice and still
# checks every edge with the same shared line guard.  It is evaluator-side
# route construction only; it never relaxes the frozen clearance contract.
FINE_CLEARANCE_ROUTE_RESOLUTION_M = 0.25
FINE_CLEARANCE_ROUTE_LOCAL_MARGIN_M = 1.25
FINE_CLEARANCE_ROUTE_MAX_GRID_POINTS = 20000
FINE_CLEARANCE_ROUTE_MAX_EXPANDED_NODES = 6000
FINE_CLEARANCE_ROUTE_MAX_EDGE_GUARDS = 120000
FINE_CLEARANCE_ROUTE_CONNECTOR_RADIUS_M = 1.5
FINE_CLEARANCE_ROUTE_START_CANDIDATE_LIMIT = 24
STATIC_QUERY_DYNAMIC_HIT_ADVANCE_M = 0.01
STATIC_QUERY_MAX_DYNAMIC_HITS = 32
MAX_PHYSICS_WAYPOINT_SPAN_M = 2.0
# ``0.30 m/s`` was only a low-speed arrival condition, not a true stop.  On
# 00803's certified right-angle route the v7 executor switched with 0.277 m/s
# of lateral residual velocity and left the 0.30 m physical-clearance tube.
# A rest-to-rest reference must therefore wait for an actual near-rest state.
WAYPOINT_SETTLE_SPEED_MPS = 0.05
# A 0.10 m outcome-agreement tolerance is acceptable for a final accounting
# record, but it is too large to start a new rest-to-rest segment in a narrow
# corridor.  A route admitted with a 0.30 m physical clearance can otherwise
# leave its certified centreline at a corner before the next reference starts.
WAYPOINT_SETTLE_POSITION_TOLERANCE_M = 0.03
# RACER's conservative configuration uses 1.0 m/s and 0.8 m/s^2.  These are
# trajectory limits, not feedback-authority limits: FUEL/RACER/FALCON publish
# time-indexed position, velocity and acceleration references to an SO(3)
# tracker.  Conflating both limits caused the first faster-controller probe to
# overshoot every 0.54 m route because the tracker could not correct lag.
CF2X_MAX_REFERENCE_SPEED_MPS = 1.0
CF2X_MAX_REFERENCE_ACCELERATION_MPS2 = 0.8
# marker2
# marker
# Mature waypoint explorers replan from non-static states and keep a route
# alive while an agent passes an intermediate viewpoint. This is not a
# terminal tracking relaxation: the final waypoint still requires the
# calibrated settle condition, and the static/fleet guards remain unchanged.
CF2X_WAYPOINT_PASS_THROUGH_SPEED_MPS = 0.35
# The reference profile supplies the planned acceleration.  The feedback loop
# uses a critically damped position/velocity pair (kp=4, kd=4) and has its own
# authority; it must be able to remove a 0.1 m terminal error inside a 2 s
# decision without changing the planner's 1.0 m/s speed contract.
CF2X_POSITION_ERROR_GAIN_PER_S2 = 4.0
CF2X_VELOCITY_ERROR_GAIN_PER_S = 7.0
CF2X_MAX_FEEDBACK_ACCELERATION_MPS2 = 3.0
# This version labels the execution/timing ABI, independently of the frozen
# SO(3) control law identifier below.  Any profile change requires fresh timing
# calibration and cannot be mixed with earlier outcome evidence.
CF2X_SPEED_PROFILE_ID = "time-parameterized-trapezoid-so3-guarded-v8"
CF2X_ATTITUDE_CONTROL_ID = "force-rate-limited-yaw-so3-v2"
CF2X_DEFAULT_CONTROLLER_ID = "isaac-so3-feedback-v6"
CF2X_MAXIMUM_TILT_RAD = 0.25
# FUEL's public exploration configuration limits its independently planned
# heading reference to 10 deg/s.  A direct jump to the route bearing can put a
# 180-degree yaw error into the SO(3) tracker at the first physics step and
# consume all attitude authority before translation begins.
CF2X_MAX_YAW_RATE_DEG_S = 10.0
CONTROLLER_TRACKING_TELEMETRY_HZ = 20.0
FLIGHT_CONTROL_BOUNDARY_MARGIN_M = 0.40
# This bounds outcome agreement with a scheduled fragment.  It is deliberately
# independent of the decision horizon: a 50 s budget does not authorize a
# several-second timing error to become reusable OGFR supervision.
OUTCOME_TIME_TOLERANCE_S = 0.25
# A delayed corridor entry must start after the predecessor's planned arrival
# plus this release margin, then wait again for that predecessor's measured
# settled arrival. Keep the value aligned with the common candidate authority.
TRAFFIC_RESERVATION_MINIMUM_RELEASE_MARGIN_S = 0.25
CF2X_EXECUTION_BACKEND_ID = "isaac-physx-cf2x-waypoint-executor-v1"
CF2X_EXECUTION_EVIDENCE_CLASS = "real_isaac_physx_cf2x"
ROTOR_RADIUS_M = 0.0225
AIR_DENSITY_KG_M3 = 1.225
THRUSTER_NAMES = ("m1_prop", "m2_prop", "m3_prop", "m4_prop")
CF2X_ROTOR_XY_LEVER_ARM_M = 0.031
CF2X_YAW_TORQUE_TO_THRUST_M = 0.006
CF2X_ROTOR_YAW_REACTION_SIGNS = (-1, 1, -1, 1)
CF2X_ROTOR_ALLOCATION_ID = "cf2x-usd-m1-m4-0p031m-reaction-yaw-v2"


def _cf2x_allocation_matrix() -> list[list[float]]:
    arm = CF2X_ROTOR_XY_LEVER_ARM_M
    yaw = CF2X_YAW_TORQUE_TO_THRUST_M
    return [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-yaw, yaw, -yaw, yaw],
    ]


def _waypoint_reached(
    *,
    error_m: float,
    speed_mps: float,
    requires_settle: bool,
    arrival_tolerance_m: float,
) -> bool:
    """Admit a waypoint only after the profile's required settle condition."""

    allowed_error_m = (
        min(arrival_tolerance_m, WAYPOINT_SETTLE_POSITION_TOLERANCE_M)
        if requires_settle
        else arrival_tolerance_m
    )
    if error_m > allowed_error_m:
        return False
    return not requires_settle or speed_mps <= WAYPOINT_SETTLE_SPEED_MPS


def _route_corner_speed_mps(path_m: Sequence[tuple[float, float, float]]) -> float:
    """Return a conservative pass-through speed for a guarded polyline."""

    if len(path_m) < 2:
        raise ValueError("pass-through route requires at least two points")
    segment_lengths = [
        math.dist(left, right)
        for left, right in zip(path_m, path_m[1:], strict=False)
    ]
    shortest_segment_m = min(segment_lengths)
    if shortest_segment_m <= 0.0:
        if all(length <= 0.0 for length in segment_lengths):
            # Explicit cold-start hold: no traversed segment, so the corner
            # pass-through speed is zero and only terminal settling applies.
            return 0.0
        raise ValueError("pass-through route contains a zero-length segment")
    geometry_bounded_speed_mps = math.sqrt(
        2.0 * CF2X_MAX_REFERENCE_ACCELERATION_MPS2 * shortest_segment_m
    )
    return min(CF2X_WAYPOINT_PASS_THROUGH_SPEED_MPS, geometry_bounded_speed_mps)


def _line_profile_state(
    *,
    distance_m: float,
    elapsed_s: float,
    initial_speed_mps: float,
    terminal_speed_mps: float,
    cruise_speed_mps: float,
    max_accel_mps2: float,
) -> tuple[float, float, float, float]:
    """Return (travelled_m, speed_mps, acceleration_mps2, duration_s)."""

    if distance_m <= 1.0e-9:
        return 0.0, 0.0, 0.0, 0.0
    initial_speed = min(max(0.0, initial_speed_mps), cruise_speed_mps)
    terminal_speed = min(max(0.0, terminal_speed_mps), cruise_speed_mps)
    if initial_speed == terminal_speed:
        if initial_speed <= 1.0e-12:
            duration_s = minimum_rest_to_rest_duration_s(
                distance_m,
                cruise_speed_mps=cruise_speed_mps,
                max_accel_mps2=max_accel_mps2,
            )
            peak_speed_mps = min(
                cruise_speed_mps,
                math.sqrt(max_accel_mps2 * distance_m),
            )
            acceleration_time_s = peak_speed_mps / max_accel_mps2
            if elapsed_s < acceleration_time_s:
                speed_mps = max_accel_mps2 * elapsed_s
                travelled_m = 0.5 * max_accel_mps2 * elapsed_s**2
                acceleration_mps2 = max_accel_mps2
            elif elapsed_s < duration_s - acceleration_time_s:
                speed_mps = peak_speed_mps
                travelled_m = (
                    0.5 * max_accel_mps2 * acceleration_time_s**2
                    + peak_speed_mps * (elapsed_s - acceleration_time_s)
                )
                acceleration_mps2 = 0.0
            elif elapsed_s < duration_s:
                braking_time_s = elapsed_s - duration_s + acceleration_time_s
                speed_mps = max(0.0, peak_speed_mps - max_accel_mps2 * braking_time_s)
                travelled_m = distance_m - 0.5 * max_accel_mps2 * (
                    duration_s - elapsed_s
                ) ** 2
                acceleration_mps2 = -max_accel_mps2
            else:
                speed_mps = 0.0
                travelled_m = distance_m
                acceleration_mps2 = 0.0
            return travelled_m, speed_mps, acceleration_mps2, duration_s
        duration_s = distance_m / initial_speed
        clamped_elapsed_s = min(elapsed_s, duration_s)
        return (
            initial_speed * clamped_elapsed_s,
            initial_speed,
            0.0,
            duration_s,
        )

    if initial_speed < terminal_speed:
        acceleration_distance_m = (
            terminal_speed**2 - initial_speed**2
        ) / (2.0 * max_accel_mps2)
        if distance_m < acceleration_distance_m:
            peak_speed_mps = math.sqrt(
                initial_speed**2 + 2.0 * max_accel_mps2 * distance_m
            )
            duration_s = (peak_speed_mps - initial_speed) / max_accel_mps2
            clamped_elapsed_s = min(elapsed_s, duration_s)
            speed_mps = initial_speed + max_accel_mps2 * clamped_elapsed_s
            travelled_m = (
                initial_speed * clamped_elapsed_s
                + 0.5 * max_accel_mps2 * clamped_elapsed_s**2
            )
            acceleration_mps2 = (
                max_accel_mps2 if clamped_elapsed_s < duration_s else 0.0
            )
            return travelled_m, speed_mps, acceleration_mps2, duration_s
        acceleration_time_s = (terminal_speed - initial_speed) / max_accel_mps2
        cruise_distance_m = distance_m - acceleration_distance_m
        duration_s = acceleration_time_s + cruise_distance_m / terminal_speed
        clamped_elapsed_s = min(elapsed_s, duration_s)
        if clamped_elapsed_s < acceleration_time_s:
            speed_mps = initial_speed + max_accel_mps2 * clamped_elapsed_s
            travelled_m = (
                initial_speed * clamped_elapsed_s
                + 0.5 * max_accel_mps2 * clamped_elapsed_s**2
            )
            acceleration_mps2 = max_accel_mps2
        else:
            speed_mps = terminal_speed
            travelled_m = (
                acceleration_distance_m
                + terminal_speed * (clamped_elapsed_s - acceleration_time_s)
            )
            acceleration_mps2 = 0.0
        return travelled_m, speed_mps, acceleration_mps2, duration_s

    deceleration_distance_m = (
        initial_speed**2 - terminal_speed**2
    ) / (2.0 * max_accel_mps2)
    if distance_m < deceleration_distance_m:
        duration_s = minimum_rest_to_rest_duration_s(
            distance_m,
            cruise_speed_mps=cruise_speed_mps,
            max_accel_mps2=max_accel_mps2,
        )
        peak_speed_mps = min(
            cruise_speed_mps,
            math.sqrt(max_accel_mps2 * distance_m),
        )
        acceleration_time_s = peak_speed_mps / max_accel_mps2
        if elapsed_s < acceleration_time_s:
            speed_mps = max_accel_mps2 * elapsed_s
            travelled_m = 0.5 * max_accel_mps2 * elapsed_s**2
            acceleration_mps2 = max_accel_mps2
        elif elapsed_s < duration_s - acceleration_time_s:
            speed_mps = peak_speed_mps
            travelled_m = (
                0.5 * max_accel_mps2 * acceleration_time_s**2
                + peak_speed_mps * (elapsed_s - acceleration_time_s)
            )
            acceleration_mps2 = 0.0
        elif elapsed_s < duration_s:
            braking_time_s = elapsed_s - duration_s + acceleration_time_s
            speed_mps = max(0.0, peak_speed_mps - max_accel_mps2 * braking_time_s)
            travelled_m = distance_m - 0.5 * max_accel_mps2 * (
                duration_s - elapsed_s
            ) ** 2
            acceleration_mps2 = -max_accel_mps2
        else:
            speed_mps = 0.0
            travelled_m = distance_m
            acceleration_mps2 = 0.0
        return travelled_m, speed_mps, acceleration_mps2, duration_s
    cruise_distance_m = distance_m - deceleration_distance_m
    cruise_time_s = cruise_distance_m / initial_speed
    deceleration_time_s = (initial_speed - terminal_speed) / max_accel_mps2
    duration_s = cruise_time_s + deceleration_time_s
    clamped_elapsed_s = min(elapsed_s, duration_s)
    if clamped_elapsed_s < cruise_time_s:
        speed_mps = initial_speed
        travelled_m = initial_speed * clamped_elapsed_s
        acceleration_mps2 = 0.0
    elif clamped_elapsed_s >= duration_s:
        speed_mps = terminal_speed
        travelled_m = distance_m
        acceleration_mps2 = 0.0
    else:
        braking_time_s = clamped_elapsed_s - cruise_time_s
        speed_mps = max(terminal_speed, initial_speed - max_accel_mps2 * braking_time_s)
        travelled_m = (
            cruise_distance_m
            + initial_speed * braking_time_s
            - 0.5 * max_accel_mps2 * braking_time_s**2
        )
        acceleration_mps2 = -max_accel_mps2
    return travelled_m, speed_mps, acceleration_mps2, duration_s


def _minimum_time_line_reference_with_boundary_speeds(
    start_m: tuple[float, float, float],
    end_m: tuple[float, float, float],
    elapsed_s: float,
    *,
    initial_speed_mps: float,
    terminal_speed_mps: float,
) -> LineTrajectoryReference:
    """Evaluate one line segment without forcing an intermediate stop."""

    if len(start_m) != 3 or len(end_m) != 3:
        raise ValueError("line trajectory endpoints must be three-dimensional")
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("trajectory elapsed time must be finite and non-negative")
    delta = tuple(end_m[axis] - start_m[axis] for axis in range(3))
    distance_m = math.sqrt(sum(value * value for value in delta))
    if distance_m <= 1.0e-9:
        return LineTrajectoryReference(tuple(end_m), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    travelled_m, speed_mps, acceleration_mps2, duration_s = _line_profile_state(
        distance_m=distance_m,
        elapsed_s=elapsed_s,
        initial_speed_mps=initial_speed_mps,
        terminal_speed_mps=terminal_speed_mps,
        cruise_speed_mps=CF2X_MAX_REFERENCE_SPEED_MPS,
        max_accel_mps2=CF2X_MAX_REFERENCE_ACCELERATION_MPS2,
    )
    direction = tuple(value / distance_m for value in delta)
    return LineTrajectoryReference(
        position_m=tuple(
            start_m[axis] + direction[axis] * travelled_m for axis in range(3)
        ),
        velocity_mps=tuple(direction[axis] * speed_mps for axis in range(3)),
        acceleration_mps2=tuple(direction[axis] * acceleration_mps2 for axis in range(3)),
        duration_s=duration_s,
    )


@dataclass(frozen=True, slots=True)
class LineTrajectoryReference:
    """One time-indexed position/velocity/acceleration reference on a line."""

    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    acceleration_mps2: tuple[float, float, float]
    duration_s: float


def _minimum_time_line_reference(
    start_m: tuple[float, float, float],
    end_m: tuple[float, float, float],
    elapsed_s: float,
) -> LineTrajectoryReference:
    """Evaluate a rest-to-rest triangular or trapezoidal line trajectory."""

    if len(start_m) != 3 or len(end_m) != 3:
        raise ValueError("line trajectory endpoints must be three-dimensional")
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("trajectory elapsed time must be finite and non-negative")
    delta = tuple(end_m[axis] - start_m[axis] for axis in range(3))
    distance_m = math.sqrt(sum(value * value for value in delta))
    zero = (0.0, 0.0, 0.0)
    if distance_m <= 1.0e-9:
        return LineTrajectoryReference(tuple(end_m), zero, zero, 0.0)

    acceleration_mps2 = CF2X_MAX_REFERENCE_ACCELERATION_MPS2
    cruise_speed_mps = CF2X_MAX_REFERENCE_SPEED_MPS
    nominal_acceleration_time_s = cruise_speed_mps / acceleration_mps2
    nominal_acceleration_distance_m = 0.5 * acceleration_mps2 * nominal_acceleration_time_s**2
    if 2.0 * nominal_acceleration_distance_m >= distance_m:
        acceleration_time_s = math.sqrt(distance_m / acceleration_mps2)
        peak_speed_mps = acceleration_mps2 * acceleration_time_s
        cruise_time_s = 0.0
    else:
        acceleration_time_s = nominal_acceleration_time_s
        peak_speed_mps = cruise_speed_mps
        cruise_distance_m = distance_m - 2.0 * nominal_acceleration_distance_m
        cruise_time_s = cruise_distance_m / cruise_speed_mps
    duration_s = minimum_rest_to_rest_duration_s(
        distance_m,
        cruise_speed_mps=cruise_speed_mps,
        max_accel_mps2=acceleration_mps2,
    )
    clamped_time_s = min(elapsed_s, duration_s)
    if clamped_time_s < acceleration_time_s:
        scalar_acceleration_mps2 = acceleration_mps2
        scalar_speed_mps = acceleration_mps2 * clamped_time_s
        travelled_m = 0.5 * acceleration_mps2 * clamped_time_s**2
    elif clamped_time_s < acceleration_time_s + cruise_time_s:
        scalar_acceleration_mps2 = 0.0
        scalar_speed_mps = peak_speed_mps
        travelled_m = 0.5 * acceleration_mps2 * acceleration_time_s**2 + peak_speed_mps * (
            clamped_time_s - acceleration_time_s
        )
    elif clamped_time_s < duration_s:
        braking_time_s = clamped_time_s - acceleration_time_s - cruise_time_s
        scalar_acceleration_mps2 = -acceleration_mps2
        scalar_speed_mps = max(0.0, peak_speed_mps - acceleration_mps2 * braking_time_s)
        travelled_m = distance_m - 0.5 * acceleration_mps2 * (duration_s - clamped_time_s) ** 2
    else:
        scalar_acceleration_mps2 = 0.0
        scalar_speed_mps = 0.0
        travelled_m = distance_m

    direction = tuple(value / distance_m for value in delta)
    return LineTrajectoryReference(
        position_m=tuple(start_m[axis] + direction[axis] * travelled_m for axis in range(3)),
        velocity_mps=tuple(direction[axis] * scalar_speed_mps for axis in range(3)),
        acceleration_mps2=tuple(direction[axis] * scalar_acceleration_mps2 for axis in range(3)),
        duration_s=duration_s,
    )


def _controller_tracking_profile(
    controller_id: str = CF2X_DEFAULT_CONTROLLER_ID,
    *,
    physics_dt_s: float | None = None,
) -> dict[str, object]:
    if controller_id not in {
        CF2X_DEFAULT_CONTROLLER_ID,
        BITCRAZE_LEE_CONTROLLER_ID,
        BITCRAZE_MELLINGER_CONTROLLER_ID,
    }:
        raise ValueError(f"unsupported CF2X controller_id: {controller_id}")
    if physics_dt_s is not None and (not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0):
        raise ValueError("physics_dt_s must be finite and positive")
    effective_control_rate_hz = 120.0 if physics_dt_s is None else 1.0 / physics_dt_s
    profile: dict[str, object] = {
        "controller_id": controller_id,
        "speed_profile": CF2X_SPEED_PROFILE_ID,
        "attitude_control": (
            CF2X_ATTITUDE_CONTROL_ID
            if controller_id == CF2X_DEFAULT_CONTROLLER_ID
            else (
                "bitcraze-lee-se3-decision-core-guarded-isaac-v4"
                if controller_id == BITCRAZE_LEE_CONTROLLER_ID
                else "bitcraze-mellinger-legacy-mixer-adapted-v1"
            )
        ),
        "maximum_tilt_rad": CF2X_MAXIMUM_TILT_RAD,
        "maximum_yaw_rate_deg_s": CF2X_MAX_YAW_RATE_DEG_S,
        "maximum_reference_speed_mps": CF2X_MAX_REFERENCE_SPEED_MPS,
        "maximum_reference_acceleration_mps2": CF2X_MAX_REFERENCE_ACCELERATION_MPS2,
        "waypoint_pass_through_speed_mps": CF2X_WAYPOINT_PASS_THROUGH_SPEED_MPS,
        "waypoint_settle_speed_mps": WAYPOINT_SETTLE_SPEED_MPS,
        "waypoint_settle_position_tolerance_m": WAYPOINT_SETTLE_POSITION_TOLERANCE_M,
        "intermediate_waypoint_requires_settle": False,
        "terminal_waypoint_requires_settle": True,
        "tracking_clearance_margin_m": TRACKING_CLEARANCE_MARGIN_M,
        "rotor_allocation_id": CF2X_ROTOR_ALLOCATION_ID,
        "rotor_order": list(THRUSTER_NAMES),
        "rotor_xy_lever_arm_m": CF2X_ROTOR_XY_LEVER_ARM_M,
        "yaw_torque_to_thrust_m": CF2X_YAW_TORQUE_TO_THRUST_M,
        "rotor_yaw_reaction_signs": list(CF2X_ROTOR_YAW_REACTION_SIGNS),
        "actuator_initialization_id": CF2X_ACTUATOR_INITIALIZATION_ID,
        "initial_rotor_rps": CF2X_INITIAL_ROTOR_RPS,
        "thrust_constant_n_per_rps2": CF2X_THRUST_CONSTANT_N_PER_RPS2,
        "tau_inc_range_s": list(CF2X_THRUSTER_TAU_INC_RANGE_S),
        "tau_dec_range_s": list(CF2X_THRUSTER_TAU_DEC_RANGE_S),
    }
    if controller_id == CF2X_DEFAULT_CONTROLLER_ID:
        profile.update(
            {
                "position_error_gain_per_s2": CF2X_POSITION_ERROR_GAIN_PER_S2,
                "velocity_error_gain_per_s": CF2X_VELOCITY_ERROR_GAIN_PER_S,
                "maximum_feedback_acceleration_mps2": CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
                "effective_control_rate_hz": effective_control_rate_hz,
            }
        )
    elif controller_id == BITCRAZE_LEE_CONTROLLER_ID:
        profile.update(
            {
                "position_error_gain_per_s2": 7.0,
                "velocity_error_gain_per_s": 4.0,
                "position_error_limit_m": BITCRAZE_LEE_POSITION_ERROR_LIMIT_M,
                "velocity_error_limit_mps": BITCRAZE_LEE_VELOCITY_ERROR_LIMIT_MPS,
                "maximum_feedback_acceleration_mps2": CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
                "effective_control_rate_hz": effective_control_rate_hz,
                "official_control_rate_hz": BITCRAZE_LEE_OFFICIAL_CONTROL_RATE_HZ,
                "feedback_acceleration_limit_mode": "norm_limited_isaac_adapter_v1",
                "source_url": BITCRAZE_LEE_SOURCE_URL,
                "source_commit": BITCRAZE_LEE_SOURCE_COMMIT,
                "source_file": BITCRAZE_LEE_SOURCE_FILE,
                "source_license": "MIT",
                "adaptation_scope": (
                    "Lee decision core with Isaac state, physics dt, mass calibration, "
                    "and asset-matched constrained rotor allocation"
                ),
            }
        )
    else:
        profile.update(
            {
                "official_control_rate_hz": BITCRAZE_MELLINGER_OFFICIAL_CONTROL_RATE_HZ,
                "effective_control_rate_hz": effective_control_rate_hz,
                "source_url": BITCRAZE_MELLINGER_SOURCE_URL,
                "source_commit": BITCRAZE_MELLINGER_SOURCE_COMMIT,
                "source_file": BITCRAZE_MELLINGER_SOURCE_FILE,
                "source_license": "MIT",
                "adaptation_scope": (
                    "Mellinger decision core and legacy mixer with Crazyswarm2 PWM/RPM "
                    "calibration normalized at the active Isaac hover equilibrium"
                ),
                "firmware_power_distribution": "legacy_quadrotor_m1_m4_v1",
                "actuator_translation": "crazyswarm2_pwm_rpm_hover_normalized_v1",
            }
        )
    return profile


def _transit_timing_controller_tracking_profile(
    controller_id: str = CF2X_DEFAULT_CONTROLLER_ID,
    *,
    physics_dt_s: float | None = None,
) -> dict[str, object]:
    """Return the v2 calibration ABI, not the complete outcome provenance.

    Outcomes deliberately retain source licence and adaptation prose so that a
    controller result remains auditable.  Transit calibration, however,
    records only the controller fields which its validator accepts.  Comparing
    that reduced calibration ABI to a complete outcome profile made every
    Bitcraze Mellinger calibration fail before physics started.  Keep the two
    representations explicit: this function is the sole runtime producer of
    the reduced ABI used to validate a timing calibration.
    """

    profile = _controller_tracking_profile(controller_id, physics_dt_s=physics_dt_s)
    profile.pop("source_license", None)
    if controller_id == BITCRAZE_MELLINGER_CONTROLLER_ID:
        # The calibration validator records the immutable source revision and
        # mixer identity; explanatory prose has no effect on transit timing.
        profile.pop("adaptation_scope", None)
    return profile


def _observation_source_identity(
    source_id: str | None,
    *,
    episode_id: str,
    agent_id: str,
) -> tuple[str | None, str | None, str | None]:
    """Keep the source identity tuple all present or all absent."""

    if source_id is None:
        return None, None, None
    return source_id, episode_id, agent_id


def _observation_failure_reason(
    *,
    completed: bool,
    collided: bool,
    out_of_bounds: bool,
    source_id: str | None,
) -> str:
    if collided:
        return "observation_collision"
    if out_of_bounds:
        return "observation_out_of_bounds"
    if completed and source_id is None:
        return "observation_no_valid_range_frame"
    return ""


def _scheduled_observation_completed(
    *,
    timestamp_s: float,
    planned_end_s: float,
    actual_start_s: float,
    minimum_dwell_s: float,
    final_physics_timestamp_s: float,
) -> bool:
    """Complete sensing at the shared decision boundary, not after a new timer.

    Transit timing is deliberately conservative.  Starting a full planned
    dwell again when a vehicle arrives early shortens the episode by the
    prediction slack.  The physical executor instead observes until the
    absolute fragment boundary while still requiring the calibrated minimum
    dwell.  A sub-step horizon remainder is never claimed as simulated time.
    """

    boundary_s = min(planned_end_s, final_physics_timestamp_s)
    return (
        timestamp_s + 1.0e-12 >= boundary_s
        and timestamp_s - actual_start_s + 1.0e-12 >= minimum_dwell_s
    )


def _minimum_observation_dwell_completed(
    *, timestamp_s: float, actual_start_s: float, minimum_dwell_s: float
) -> bool:
    """Return whether an event-driven action has completed its real dwell."""

    return timestamp_s - actual_start_s + 1.0e-12 >= minimum_dwell_s


def _sparse_range_sampling_phase(
    *,
    transit_completed: bool,
    observation_completed: bool,
    failed: bool,
    reservation_waiting: bool,
    team_awaiting: bool = False,
) -> Literal["transit", "dwell"] | None:
    """Return the eligible physical sensing phase for one realised CF2X tick.

    Sparse range sensing is available while a vehicle moves and while it
    completes its required observation dwell. A traffic reservation is neither
    motion nor observation activity, so it cannot generate repeated evidence
    while holding position.  While the team is still awaiting a slower
    vehicle (the synchronous hold), a completed fast vehicle keeps sampling
    in place: the waiting time produces information instead of being idle,
    which is the same dwell sensing contract the H15 entitlement allows.
    """

    if failed or reservation_waiting:
        return None
    if observation_completed:
        return "dwell" if team_awaiting else None
    if transit_completed:
        return "dwell"
    return "transit"


def _finalize_fragment_pair_into(
    ledger: list[FragmentExecutionSample],
    *,
    index: int,
    transit: FragmentInstance,
    observe: FragmentInstance,
    transit_completed: bool,
    observation_completed: bool,
    transit_trace: tuple[tuple[float, float, float], ...],
    observation_trace: tuple[tuple[float, float, float], ...],
    transit_release_s: float | None,
    transit_end_s: float | None,
    observation_start_s: float | None,
    observation_end_s: float | None,
    execution_horizon_s: float,
    energy_j: float,
    collision: bool,
    out_of_bounds: bool,
    separation_violation: bool,
    static_clearance_contract_violation: bool,
    minimum_clearance_m: float,
    connected_at_every_tick: bool,
    last_sensor_source_id: str | None,
    rolling: bool,
) -> None:
    """Append the realised transit + observation samples of one fragment pair.

    Shared by the synchronous finalisation at window end and by the async
    per-agent roll (on_agent_complete), so rolled pairs produce the exact
    same provenance fields as synchronous pairs.  ``rolling`` only affects
    the observation source binding: a rolled pair always has a sensor source
    (its dwell completed inside the window).
    """

    if transit_completed or collision or out_of_bounds:
        ledger.append(
            FragmentExecutionSample(
                planned_fragment_hash=transit.digest,
                executed=True,
                actual_start_s=transit_release_s or transit.planned_start,
                actual_end_s=transit_end_s or execution_horizon_s,
                command_path_m=transit.path,
                actual_path_m=transit_trace,
                execution_trace_hash=canonical_sha256(transit_trace),
                collision=collision,
                out_of_bounds=out_of_bounds,
                inter_agent_separation_violation=separation_violation,
                static_clearance_contract_violation=static_clearance_contract_violation,
                minimum_clearance_m=minimum_clearance_m,
                energy_used_j=energy_j,
                communication_connected_at_every_telemetry_tick=connected_at_every_tick,
            )
        )
    else:
        ledger.append(
            FragmentExecutionSample(
                planned_fragment_hash=transit.digest,
                executed=True,
                actual_start_s=transit_release_s or transit.planned_start,
                actual_end_s=execution_horizon_s,
                command_path_m=transit.path,
                actual_path_m=transit_trace,
                execution_trace_hash=canonical_sha256(transit_trace),
                static_clearance_contract_violation=static_clearance_contract_violation,
                minimum_clearance_m=minimum_clearance_m,
                inter_agent_separation_violation=separation_violation,
                energy_used_j=energy_j,
                communication_connected_at_every_telemetry_tick=connected_at_every_tick,
                failure_reason="transit_timeout",
            )
        )
    observation_trace_tuple = tuple(observation_trace)
    if observation_completed or collision or out_of_bounds:
        source_id = last_sensor_source_id
        source_id, source_episode_id, source_agent_id = _observation_source_identity(
            source_id,
            episode_id=observe.episode_id,
            agent_id=observe.agent_id,
        )
        observation_verified = observation_completed and source_id is not None
        ledger.append(
            FragmentExecutionSample(
                planned_fragment_hash=observe.digest,
                executed=True,
                actual_start_s=observation_start_s or observe.planned_start,
                actual_end_s=observation_end_s or execution_horizon_s,
                command_path_m=observe.path,
                actual_path_m=observation_trace_tuple or (observe.path[0],),
                execution_trace_hash=canonical_sha256(observation_trace_tuple),
                collision=collision,
                out_of_bounds=out_of_bounds,
                inter_agent_separation_violation=separation_violation,
                static_clearance_contract_violation=static_clearance_contract_violation,
                minimum_clearance_m=minimum_clearance_m,
                energy_used_j=0.0,
                communication_connected_at_every_telemetry_tick=connected_at_every_tick,
                source_observation_id=source_id,
                source_observation_episode_id=source_episode_id,
                source_observation_agent_id=source_agent_id,
                range_ok=observation_verified,
                fov_ok=observation_verified,
                los_ok=observation_verified,
                orientation_ok=observation_verified,
                dwell_ok=observation_verified,
                failure_reason=_observation_failure_reason(
                    completed=observation_completed,
                    collided=collision,
                    out_of_bounds=out_of_bounds,
                    source_id=source_id,
                ),
            )
        )
    else:
        ledger.append(
            FragmentExecutionSample(
                planned_fragment_hash=observe.digest,
                executed=False,
                actual_start_s=execution_horizon_s,
                actual_end_s=execution_horizon_s,
                execution_trace_hash=canonical_sha256(observation_trace_tuple),
                failure_reason="observation_not_reached",
            )
        )


class CandidateHeadroomError(ValueError):
    """Fail closed with evaluator-only counts for candidate-pool diagnosis.

    A P07 selector comparison needs two legal common candidates.  Merely
    recording that the condition failed conceals whether the issue is the
    finite decision budget, static flight admission, or synchronized fleet
    separation.  These counters are an immutable engineering denominator;
    they never become policy-visible features or task rewards.
    """

    def __init__(self, message: str, admission_audit: dict[str, int]) -> None:
        super().__init__(message)
        self.admission_audit = dict(sorted(admission_audit.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime evidence: {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _as_point(raw: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{label} must be a three-coordinate list")
    point = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label} must be finite")
    return point


def _load_collision_triangle_mesh(usd_path: Path) -> Any:
    """Load the collision mesh evaluator-side without exposing it to a method."""

    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open collision USD: {usd_path}")
    cache = UsdGeom.XformCache()
    vertices: list[Any] = []
    faces: list[Any] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"mesh lacks points: {prim.GetPath()}")
        if not len(counts) or not np.all(counts == 3) or len(indices) != int(counts.sum()):
            raise ValueError(f"mesh is not a valid triangulation: {prim.GetPath()}")
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = (np.column_stack((points, np.ones(len(points)))) @ transform)[:, :3]
        vertices.append(world)
        faces.append(indices.reshape((-1, 3)) + offset)
        offset += len(points)
    if not vertices:
        raise ValueError("collision USD contains no mesh triangles")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )


def _build_conservative_clearance_field(
    collision_usd: Path,
) -> tuple[Any, ConservativeVoxelClearance, dict[str, object]]:
    """Derive an evaluator-only clearance lower bound from the actual collider."""

    from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf

    mesh = _load_collision_triangle_mesh(collision_usd)
    arrays, report = build_enclosed_esdf(
        mesh,
        resolution_m=0.1,
        vehicle_clearance_m=0.125,
    )
    field = ConservativeVoxelClearance(
        arrays["collision_distance_m"],
        tuple(float(value) for value in arrays["origin_center_m"]),
        float(arrays["resolution_m"]),
    )
    return (
        mesh,
        field,
        {
            "source": "same_static_collision_mesh_as_physx",
            "representation": "conservative_voxel_esdf_lower_bound_v1",
            "resolution_m": field.resolution_m,
            "discretization_margin_m": field.discretization_margin_m,
            "nominal_vehicle_clearance_m": FLIGHT_CLEARANCE_M,
            "tracking_clearance_margin_m": TRACKING_CLEARANCE_MARGIN_M,
            "arrival_tracking_reserve_m": TRACKING_CLEARANCE_MARGIN_M,
            "required_continuous_flight_clearance_m": PLANNED_CONTINUOUS_CLEARANCE_M,
            "required_route_sample_clearance_m": REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
            "required_terminal_clearance_m": REQUIRED_TERMINAL_CLEARANCE_M,
            "esdf_generation_method": report["generation_method"],
        },
    )


@dataclass(slots=True)
class _EvaluatorStaticClearance:
    """Admit public routes from the exact evaluator collision geometry.

    The 0.25 m ESDF is intentionally a cheap lower-bound prefilter.  Its full
    voxel-diagonal uncertainty can reject a valid indoor CF2X corridor, so a
    failed lower bound falls back to exact nearest-triangle distance on the
    *same* collision USD supplied to PhysX.  Neither field is exposed to a
    candidate strategy; only the guard's legal/infeasible result is public.
    """

    field: ConservativeVoxelClearance
    collision_mesh: Any
    _exact_cache_m: dict[tuple[float, float, float], float] = field(default_factory=dict)
    # `trimesh` backs ``triangles_tree`` with a native rtree index. On the
    # required Windows stack it is not safe to query that index concurrently.
    # Keep the mesh and its exact-distance cache as one serialized evaluator
    # transaction; this affects neither the points checked nor their values.
    _exact_query_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    esdf_admission_count: int = 0
    exact_fallback_count: int = 0
    esdf_upper_bound_skip_count: int = 0
    exact_rejection_count: int = 0
    exact_batch_call_count: int = 0
    _exact_query_wall_s: float = 0.0
    _exact_query_point_count: int = 0
    _local_mesh_cache: dict[tuple[int, int, int, int, int, int], Any] = field(default_factory=dict)
    _face_min_m: Any = None
    _face_max_m: Any = None
    _local_mesh_cache_hit_count: int = 0
    _local_mesh_construction_count: int = 0
    _local_mesh_construction_wall_s: float = 0.0
    _local_mesh_max_selected_face_count: int = 0
    _face_triangles_m: Any = None
    _face_cell_index: dict[tuple[int, int, int], list[int]] | None = field(
        default=None, init=False, repr=False
    )
    _face_cell_index_build_wall_s: float = 0.0
    _face_cell_index_query_count: int = 0

    # ``trimesh.proximity.closest_point`` has a substantial per-call overhead for
    # HM3D meshes.  Execution traces are already materialized in memory, so use a
    # larger bounded batch while retaining a cap to avoid an unbounded temporary
    # array on unusually long episodes.  This changes only query batching, not
    # which trace poses are checked or the distance/caching semantics.
    _EXACT_QUERY_CHUNK_SIZE: ClassVar[int] = 512
    _LOCAL_MESH_MARGIN_M: ClassVar[float] = 0.7
    _LOCAL_MESH_CELL_M: ClassVar[float] = 0.25

    def _exact_distance_m(self, point: tuple[float, float, float]) -> float:
        return self._exact_distances_m((point,))[0]

    def _exact_distances_m(
        self, points: tuple[tuple[float, float, float], ...]
    ) -> tuple[float, ...]:
        import numpy as np

        # Do not narrow this critical section to only `closest_point`: concurrent
        # cache misses can otherwise build/read the shared native rtree and race
        # before a later call is serialized.
        with self._exact_query_lock:
            keys = tuple(tuple(round(value, 9) for value in point) for point in points)
            missing = tuple(key for key in dict.fromkeys(keys) if key not in self._exact_cache_m)
            # ``closest_point`` has substantial Python/rtree setup cost per call. Batch only
            # evaluator-side grid points and cap chunks so a dense HM3D room cannot spike memory.
            for offset in range(0, len(missing), self._EXACT_QUERY_CHUNK_SIZE):
                chunk = missing[offset : offset + self._EXACT_QUERY_CHUNK_SIZE]
                query_started = time.perf_counter()
                distances = self._exact_distances_with_cell_index(
                    np.asarray(chunk, dtype=np.float64)
                )
                self._exact_query_wall_s += time.perf_counter() - query_started
                self._exact_query_point_count += len(chunk)
                self.exact_batch_call_count += 1
                for key, distance in zip(chunk, distances, strict=True):
                    result = float(distance)
                    if not math.isfinite(result):
                        raise RuntimeError("exact collision-mesh clearance query was non-finite")
                    self._exact_cache_m[key] = result
            return tuple(self._exact_cache_m[key] for key in keys)

    def _local_mesh_for_missing(
        self, missing: tuple[tuple[float, float, float], ...]
    ) -> Any:
        import numpy as np
        import trimesh

        points = np.asarray(missing, dtype=np.float64)
        margin = float(self._LOCAL_MESH_MARGIN_M)
        query_min = points.min(axis=0) - margin
        query_max = points.max(axis=0) + margin
        cell_m = float(self._LOCAL_MESH_CELL_M)
        cell_min = tuple(int(math.floor(float(value) / cell_m)) for value in query_min)
        cell_max = tuple(int(math.floor(float(value) / cell_m)) for value in query_max)
        key = tuple(
            int(value) for value in (*cell_min, *cell_max)
        )
        cached = self._local_mesh_cache.get(key)
        if cached is not None:
            self._local_mesh_cache_hit_count += 1
            return cached
        mesh_min = tuple(cell_min[index] * cell_m - margin for index in range(3))
        mesh_max = tuple((cell_max[index] + 1) * cell_m + margin for index in range(3))
        if self._face_min_m is None or self._face_max_m is None:
            face_vertices = self.collision_mesh.vertices[self.collision_mesh.faces]
            self._face_min_m = face_vertices.min(axis=1)
            self._face_max_m = face_vertices.max(axis=1)
        face_min = self._face_min_m
        face_max = self._face_max_m
        selected = np.all((face_min <= mesh_max) & (face_max >= mesh_min), axis=1)
        selected_face_count = int(np.count_nonzero(selected))
        self._local_mesh_max_selected_face_count = max(
            self._local_mesh_max_selected_face_count, selected_face_count
        )
        constructed_started = time.perf_counter()
        if not np.any(selected):
            local_mesh = self.collision_mesh
        else:
            local_mesh = trimesh.Trimesh(
                vertices=self.collision_mesh.vertices,
                faces=self.collision_mesh.faces[selected],
                process=False,
            )
        self._local_mesh_construction_wall_s += time.perf_counter() - constructed_started
        self._local_mesh_construction_count += 1
        query_tree = local_mesh.triangles_tree
        self._local_mesh_cache[key] = (local_mesh, query_tree)
        return local_mesh, query_tree

    @staticmethod
    def _exact_distances_on_local_mesh(
        local_mesh: Any,
        query_tree: Any,
        points: Any,
    ) -> tuple[float, ...]:
        import numpy as np
        import trimesh

        triangles = local_mesh.triangles.view(np.ndarray)
        margin = float(_EvaluatorStaticClearance._LOCAL_MESH_MARGIN_M)
        face_counts: list[int] = []
        face_ids: list[int] = []
        expanded_points: list[Any] = []
        for point in points:
            bounds = (
                *(float(value - margin) for value in point),
                *(float(value + margin) for value in point),
            )
            candidates = tuple(int(value) for value in query_tree.intersection(bounds))
            face_counts.append(len(candidates))
            face_ids.extend(candidates)
            expanded_points.extend(tuple(point) for _ in candidates)
        if not face_ids:
            return tuple(margin for _ in points)
        candidate_triangles = triangles[np.asarray(face_ids, dtype=np.int64)]
        candidate_points = np.asarray(expanded_points, dtype=np.float64)
        closest = trimesh.triangles.closest_point(
            candidate_triangles, candidate_points
        )
        squared = ((closest - candidate_points) ** 2).sum(axis=1)
        distances: list[float] = []
        cursor = 0
        for count in face_counts:
            if count == 0:
                distances.append(margin)
                continue
            distances.append(float(np.sqrt(squared[cursor : cursor + count].min())))
            cursor += count
        return tuple(distances)

    def _face_cell_index_or_build(self) -> dict[tuple[int, int, int], list[int]]:
        import numpy as np

        if self._face_cell_index is not None:
            return self._face_cell_index
        if self._face_min_m is None or self._face_max_m is None:
            face_vertices = self.collision_mesh.vertices[self.collision_mesh.faces]
            self._face_triangles_m = face_vertices
            self._face_min_m = face_vertices.min(axis=1)
            self._face_max_m = face_vertices.max(axis=1)
        cell_m = float(self._LOCAL_MESH_CELL_M)
        min_cells = np.floor(self._face_min_m / cell_m).astype(np.int64)
        max_cells = np.floor(self._face_max_m / cell_m).astype(np.int64)
        started = time.perf_counter()
        index: dict[tuple[int, int, int], list[int]] = {}
        for face_id, (face_min, face_max) in enumerate(
            zip(min_cells, max_cells, strict=True)
        ):
            for ix in range(int(face_min[0]), int(face_max[0]) + 1):
                for iy in range(int(face_min[1]), int(face_max[1]) + 1):
                    for iz in range(int(face_min[2]), int(face_max[2]) + 1):
                        index.setdefault((ix, iy, iz), []).append(face_id)
        self._face_cell_index_build_wall_s += time.perf_counter() - started
        self._face_cell_index = index
        return index

    def _exact_distances_with_cell_index(
        self, points: Any
    ) -> tuple[float, ...]:
        import numpy as np
        import trimesh

        index = self._face_cell_index_or_build()
        triangles = self._face_triangles_m.view(np.ndarray)
        cell_m = float(self._LOCAL_MESH_CELL_M)
        margin = float(self._LOCAL_MESH_MARGIN_M)
        offsets = tuple(
            (dx, dy, dz)
            for dx in (-2, -1, 0, 1, 2)
            for dy in (-2, -1, 0, 1, 2)
            for dz in (-2, -1, 0, 1, 2)
        )
        face_counts: list[int] = []
        face_ids: list[int] = []
        for point in points:
            point_cell = tuple(
                int(math.floor(float(value) / cell_m)) for value in point
            )
            point_face_ids: list[int] = []
            for dx, dy, dz in offsets:
                cell_ids = index.get(
                    (
                        point_cell[0] + dx,
                        point_cell[1] + dy,
                        point_cell[2] + dz,
                    )
                )
                if cell_ids:
                    point_face_ids.extend(cell_ids)
            self._face_cell_index_query_count += 1
            unique_ids = tuple(dict.fromkeys(point_face_ids))
            face_counts.append(len(unique_ids))
            face_ids.extend(unique_ids)
        if not face_ids:
            return tuple(margin for _ in points)
        candidate_triangles = triangles[np.asarray(face_ids, dtype=np.int64)]
        candidate_points = np.repeat(
            np.asarray(points, dtype=np.float64),
            np.asarray(face_counts, dtype=np.int64),
            axis=0,
        )
        closest = trimesh.triangles.closest_point(
            candidate_triangles, candidate_points
        )
        squared = ((closest - candidate_points) ** 2).sum(axis=1)
        distances: list[float] = []
        cursor = 0
        for count in face_counts:
            if count == 0:
                distances.append(margin)
                continue
            distances.append(float(np.sqrt(squared[cursor : cursor + count].min())))
            cursor += count
        return tuple(distances)

    def exact_static_distances_m(
        self, points: tuple[tuple[float, float, float], ...]
    ) -> tuple[float, ...]:
        """Return evaluator-only distances to the static HM3D collision mesh.

        This deliberately bypasses the ESDF admission shortcut: execution
        telemetry needs a measured mesh distance, not a Boolean planning
        decision. The collision mesh is the same immutable USD derivative
        used to create the static PhysX world, and never reaches a strategy.
        """

        return self._exact_distances_m(points)

    def _exact_cannot_meet_clearance(
        self,
        assessment: Any,
        required_clearance_m: float,
    ) -> bool:
        sampled = assessment.sampled_distance_m
        margin = assessment.discretization_margin_m
        if sampled is None or margin is None:
            return False
        if not math.isfinite(float(sampled)) or not math.isfinite(float(margin)):
            return False
        # ESDF sampled distance is to an occupied voxel centre. The exact mesh
        # distance is at most one voxel diagonal farther, so it cannot meet a
        # stricter requirement when this upper bound is already too small.
        return float(sampled) + float(margin) + 1.0e-12 < float(required_clearance_m)

    def admits(self, point: tuple[float, float, float], required_clearance_m: float) -> bool:
        assessment = self.field.assess(point)
        if assessment.admits(required_clearance_m):
            self.esdf_admission_count += 1
            return True
        if self._exact_cannot_meet_clearance(assessment, required_clearance_m):
            self.esdf_upper_bound_skip_count += 1
            self.exact_rejection_count += 1
            return False
        self.exact_fallback_count += 1
        admitted = self._exact_distance_m(point) + 1.0e-12 >= required_clearance_m
        if not admitted:
            self.exact_rejection_count += 1
        return admitted

    def admits_many(
        self,
        points: tuple[tuple[float, float, float], ...],
        required_clearance_m: float,
    ) -> tuple[bool, ...]:
        """Use the same ESDF/exact authority as ``admits`` for public grid points."""

        assessments = tuple(self.field.assess(point) for point in points)
        accepted = [assessment.admits(required_clearance_m) for assessment in assessments]
        self.esdf_admission_count += sum(accepted)
        fallback_indices = tuple(index for index, value in enumerate(accepted) if not value)
        exact_required_indices: list[int] = []
        for index in fallback_indices:
            if self._exact_cannot_meet_clearance(
                assessments[index], required_clearance_m
            ):
                self.esdf_upper_bound_skip_count += 1
                self.exact_rejection_count += 1
            else:
                exact_required_indices.append(index)
        if exact_required_indices:
            distances = self._exact_distances_m(
                tuple(points[index] for index in exact_required_indices)
            )
            self.exact_fallback_count += len(exact_required_indices)
            for index, distance in zip(
                exact_required_indices, distances, strict=True
            ):
                accepted[index] = distance + 1.0e-12 >= required_clearance_m
                if not accepted[index]:
                    self.exact_rejection_count += 1
        return tuple(accepted)

    def admits_many_with_required_clearances(
        self,
        points: tuple[tuple[float, float, float], ...],
        required_clearances_m: Sequence[float],
    ) -> tuple[bool, ...]:
        assessments = tuple(self.field.assess(point) for point in points)
        accepted = [False] * len(points)
        exact_required_indices: list[int] = []
        for index, (assessment, required_clearance_m) in enumerate(
            zip(assessments, required_clearances_m, strict=True)
        ):
            if assessment.admits(required_clearance_m):
                self.esdf_admission_count += 1
                accepted[index] = True
            elif self._exact_cannot_meet_clearance(
                assessment, required_clearance_m
            ):
                self.esdf_upper_bound_skip_count += 1
                self.exact_rejection_count += 1
            else:
                exact_required_indices.append(index)
        if exact_required_indices:
            distances = self._exact_distances_m(
                tuple(points[index] for index in exact_required_indices)
            )
            self.exact_fallback_count += len(exact_required_indices)
            for index, distance in zip(
                exact_required_indices, distances, strict=True
            ):
                accepted[index] = (
                    distance + 1.0e-12 >= required_clearances_m[index]
                )
                if not accepted[index]:
                    self.exact_rejection_count += 1
        return tuple(accepted)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "method": "esdf_lower_bound_prefilter_then_exact_same_collision_mesh_v1",
            "nominal_vehicle_clearance_m": FLIGHT_CLEARANCE_M,
            "tracking_clearance_margin_m": TRACKING_CLEARANCE_MARGIN_M,
            "arrival_tracking_reserve_m": TRACKING_CLEARANCE_MARGIN_M,
            "required_continuous_centreline_clearance_m": PLANNED_CONTINUOUS_CLEARANCE_M,
            "required_terminal_point_clearance_m": REQUIRED_TERMINAL_CLEARANCE_M,
            "maximum_sample_spacing_m": ROUTE_CLEARANCE_SAMPLE_STEP_M,
            "required_internal_sample_clearance_m": REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
            "esdf_admission_count": self.esdf_admission_count,
            "esdf_upper_bound_skip_count": self.esdf_upper_bound_skip_count,
            "exact_fallback_count": self.exact_fallback_count,
            "exact_rejection_count": self.exact_rejection_count,
            "exact_batch_call_count": self.exact_batch_call_count,
            "exact_query_wall_s": self._exact_query_wall_s,
            "exact_query_point_count": self._exact_query_point_count,
            "exact_cached_point_count": len(self._exact_cache_m),
            "local_mesh_cache_hit_count": self._local_mesh_cache_hit_count,
            "local_mesh_construction_count": self._local_mesh_construction_count,
            "local_mesh_construction_wall_s": self._local_mesh_construction_wall_s,
            "local_mesh_max_selected_face_count": self._local_mesh_max_selected_face_count,
            "local_mesh_cache_size": len(self._local_mesh_cache),
            "face_cell_index_build_wall_s": self._face_cell_index_build_wall_s,
            "face_cell_index_query_count": self._face_cell_index_query_count,
            "face_cell_index_cell_count": (
                len(self._face_cell_index)
                if self._face_cell_index is not None
                else 0
            ),
        }


def _first_static_scene_hit(
    scene_query: Any,
    source: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    endpoint_margin_m: float = 0.05,
) -> dict[str, Any] | None:
    """Return the first non-agent collider hit on a finite segment.

    PhysX exposes one closest-hit query for the full stage.  Once CF2X assets
    have spawned, a ray that starts at an agent root commonly exits that
    agent's own body before reaching the immutable HM3D collision mesh.  The
    high-level route and radio-LOS contracts are static-geometry queries, so
    agent collider hits are advanced past and audited instead of being treated
    as walls.  Dynamic safety remains enforced by synchronized separation and
    real PhysX contacts during execution.
    """

    distance = math.dist(source, target)
    if distance <= 1.0e-9:
        return {
            "hit": True,
            "hit_class": "invalid_zero_length_segment",
            "distance": 0.0,
            "ignored_dynamic_hit_count": 0,
        }
    direction = tuple((target[axis] - source[axis]) / distance for axis in range(3))
    query_limit_m = max(distance - endpoint_margin_m, 1.0e-6)
    travelled_m = 0.0
    ignored_dynamic_hits: list[dict[str, object]] = []
    for _ in range(STATIC_QUERY_MAX_DYNAMIC_HITS + 1):
        remaining_m = query_limit_m - travelled_m
        if remaining_m <= 1.0e-6:
            return None
        origin = tuple(source[axis] + travelled_m * direction[axis] for axis in range(3))
        raw_hit = dict(scene_query.raycast_closest(origin, direction, remaining_m))
        if not bool(raw_hit.get("hit")):
            return None
        rigid_body_path = _diagnostic_prim_path(raw_hit.get("rigidBody"))
        collider_path = _diagnostic_prim_path(raw_hit.get("collider"))
        paths = tuple(path for path in (rigid_body_path, collider_path) if path is not None)
        raw_distance = raw_hit.get("distance")
        try:
            local_hit_distance_m = max(0.0, float(raw_distance))
        except (TypeError, ValueError):
            local_hit_distance_m = 0.0
        if not math.isfinite(local_hit_distance_m):
            local_hit_distance_m = 0.0
        total_hit_distance_m = travelled_m + local_hit_distance_m
        if not any("/P07Agents/" in path for path in paths):
            raw_hit["distance"] = total_hit_distance_m
            raw_hit["ignored_dynamic_hit_count"] = len(ignored_dynamic_hits)
            raw_hit["ignored_dynamic_hits"] = ignored_dynamic_hits
            return raw_hit
        ignored_dynamic_hits.append(
            {
                "rigid_body_path": rigid_body_path,
                "collider_path": collider_path,
                "segment_distance_m": total_hit_distance_m,
            }
        )
        travelled_m = total_hit_distance_m + STATIC_QUERY_DYNAMIC_HIT_ADVANCE_M
    return {
        "hit": True,
        "hit_class": "dynamic_hit_skip_limit",
        "distance": travelled_m,
        "ignored_dynamic_hit_count": len(ignored_dynamic_hits),
        "ignored_dynamic_hits": ignored_dynamic_hits,
    }


def _clear_static_collision_los(
    scene_query: Any,
    source: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    endpoint_margin_m: float = 0.05,
) -> bool:
    return (
        _first_static_scene_hit(
            scene_query,
            source,
            target,
            endpoint_margin_m=endpoint_margin_m,
        )
        is None
    )


def _initial_relay_graph(
    scene_query: Any, positions: tuple[tuple[float, float, float], ...]
) -> RelayGraphSnapshot:
    return build_range_los_relay_graph(
        positions,
        maximum_range_m=10.0,
        line_of_sight_clear=lambda source, target: _clear_static_collision_los(
            scene_query, source, target
        ),
    )


def _yaw_from_delta(start: tuple[float, float, float], end: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _shortest_angular_delta_deg(source_deg: float, target_deg: float) -> float:
    """Return the signed shortest turn from source to target in degrees."""

    source = float(source_deg)
    target = float(target_deg)
    if not math.isfinite(source) or not math.isfinite(target):
        raise ValueError("yaw references must be finite")
    return (target - source + 180.0) % 360.0 - 180.0


def _rate_limited_yaw_reference_deg(
    current_reference_deg: float,
    target_heading_deg: float,
    dt_s: float,
    *,
    maximum_rate_deg_s: float = CF2X_MAX_YAW_RATE_DEG_S,
) -> float:
    """Advance a continuous heading reference along the shortest angular path."""

    dt = float(dt_s)
    maximum_rate = float(maximum_rate_deg_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("yaw-reference time step must be finite and positive")
    if not math.isfinite(maximum_rate) or maximum_rate <= 0.0:
        raise ValueError("maximum yaw-reference rate must be finite and positive")
    delta = _shortest_angular_delta_deg(current_reference_deg, target_heading_deg)
    maximum_step = maximum_rate * dt
    bounded_delta = min(max(delta, -maximum_step), maximum_step)
    return (float(current_reference_deg) + bounded_delta + 180.0) % 360.0 - 180.0


def _euler_xyz_from_quaternion_wxyz(quaternion: Any) -> tuple[Any, Any, Any]:
    import torch

    normalized = quaternion / torch.clamp(
        torch.linalg.norm(quaternion, dim=1, keepdim=True), min=1.0e-8
    )
    w, x, y, z = normalized.unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _rotation_matrix_from_quaternion_wxyz(quaternion: Any) -> Any:
    """Return batched body-to-world rotations for Isaac wxyz quaternions."""

    import torch

    normalized = quaternion / torch.clamp(
        torch.linalg.norm(quaternion, dim=1, keepdim=True), min=1.0e-8
    )
    w, x, y, z = normalized.unbind(dim=1)
    rows = (
        torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            dim=1,
        ),
        torch.stack(
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            dim=1,
        ),
        torch.stack(
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=1,
        ),
    )
    return torch.stack(rows, dim=1)


def _desired_rotation_from_force_and_yaw(requested_force_world: Any, desired_yaw: Any) -> Any:
    """Port FUEL's force/yaw construction of one complete desired attitude."""

    import torch

    vertical_force = torch.clamp(requested_force_world[:, 2:3], min=1.0e-6)
    horizontal_force = requested_force_world[:, :2]
    horizontal_norm = torch.linalg.norm(horizontal_force, dim=1, keepdim=True)
    maximum_horizontal_force = vertical_force * math.tan(CF2X_MAXIMUM_TILT_RAD)
    horizontal_force = horizontal_force * torch.clamp(
        maximum_horizontal_force / torch.clamp(horizontal_norm, min=1.0e-6),
        max=1.0,
    )
    limited_force = torch.cat((horizontal_force, vertical_force), dim=1)
    desired_b3 = limited_force / torch.clamp(
        torch.linalg.norm(limited_force, dim=1, keepdim=True), min=1.0e-8
    )
    desired_heading = torch.stack(
        (torch.cos(desired_yaw), torch.sin(desired_yaw), torch.zeros_like(desired_yaw)),
        dim=1,
    )
    desired_b2 = torch.linalg.cross(desired_b3, desired_heading, dim=1)
    desired_b2 = desired_b2 / torch.clamp(
        torch.linalg.norm(desired_b2, dim=1, keepdim=True), min=1.0e-8
    )
    desired_b1 = torch.linalg.cross(desired_b2, desired_b3, dim=1)
    return torch.stack((desired_b1, desired_b2, desired_b3), dim=2)


def _euler_xyz_from_rotation_matrix(rotation: Any) -> tuple[Any, Any, Any]:
    import torch

    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = torch.asin(torch.clamp(-rotation[:, 2, 0], min=-1.0, max=1.0))
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return roll, pitch, yaw


def _so3_attitude_error(current_rotation: Any, desired_rotation: Any) -> Any:
    """Match the vee-map attitude error used by FUEL's SO(3) simulator."""

    error_matrix = 0.5 * (
        desired_rotation.transpose(1, 2) @ current_rotation
        - current_rotation.transpose(1, 2) @ desired_rotation
    )
    return error_matrix[:, (2, 0, 1), (1, 2, 0)]


def _largest_feasible_scale(baseline: Any, delta: Any) -> Any:
    import torch

    epsilon = 1.0e-8
    limits = torch.full_like(delta, float("inf"))
    limits = torch.where(delta > epsilon, (MAX_THRUST_PER_ROTOR_N - baseline) / delta, limits)
    limits = torch.where(delta < -epsilon, baseline / (-delta), limits)
    return torch.clamp(torch.amin(limits, dim=1, keepdim=True), min=0.0, max=1.0)


def _bounded_rotor_thrust(
    robot: Any,
    reference_positions: Any,
    reference_velocities: Any,
    reference_accelerations: Any,
    headings_deg: Any,
    diagnostics: dict[str, Any] | None = None,
    controller: BitcrazeLeeTracker | BitcrazeMellingerTracker | None = None,
    dt_s: float | None = None,
) -> Any:
    """Track time-indexed public trajectory references with bounded rotor thrust."""

    import torch

    position = robot.data.root_pos_w
    velocity = robot.data.root_lin_vel_w
    current_rotation = _rotation_matrix_from_quaternion_wxyz(robot.data.root_quat_w)
    roll, pitch, yaw = _euler_xyz_from_rotation_matrix(current_rotation)
    angular_velocity = robot.data.root_ang_vel_b
    collective = HOVER_THRUST_PER_ROTOR_N * float(robot.num_thrusters)
    vehicle_mass_kg = collective / 9.81
    direct_rotor_thrust: Any | None = None
    if controller is None:
        feedback_acceleration = CF2X_POSITION_ERROR_GAIN_PER_S2 * (
            reference_positions - position
        ) + CF2X_VELOCITY_ERROR_GAIN_PER_S * (reference_velocities - velocity)
        feedback_acceleration_norm = torch.linalg.norm(feedback_acceleration, dim=1, keepdim=True)
        feedback_acceleration = feedback_acceleration * torch.clamp(
            CF2X_MAX_FEEDBACK_ACCELERATION_MPS2
            / torch.clamp(feedback_acceleration_norm, min=1.0e-6),
            max=1.0,
        )
        desired_acceleration = reference_accelerations + feedback_acceleration
        desired_yaw = torch.deg2rad(torch.as_tensor(headings_deg, device=robot.device))
        requested_force_world = vehicle_mass_kg * desired_acceleration
        requested_force_world[:, 2] += vehicle_mass_kg * 9.81
        desired_rotation = _desired_rotation_from_force_and_yaw(requested_force_world, desired_yaw)
        attitude_error = _so3_attitude_error(current_rotation, desired_rotation)
        current_body_z_world = current_rotation[:, :, 2]
        collective = torch.sum(requested_force_world * current_body_z_world, dim=1)
        wrench = torch.zeros((int(robot.num_instances), 4), device=robot.device)
        wrench[:, 0] = torch.clamp(
            collective, min=0.0, max=MAX_THRUST_PER_ROTOR_N * float(robot.num_thrusters)
        )
        wrench[:, 1] = -0.017 * attitude_error[:, 0] - 0.0065 * angular_velocity[:, 0]
        wrench[:, 2] = -0.017 * attitude_error[:, 1] - 0.0065 * angular_velocity[:, 1]
        wrench[:, 3] = torch.clamp(
            -0.0040 * attitude_error[:, 2] - 0.0015 * angular_velocity[:, 2],
            min=-0.001,
            max=0.001,
        )
        controller_id = CF2X_DEFAULT_CONTROLLER_ID
        controller_state: dict[str, Any] = {}
    elif isinstance(controller, BitcrazeLeeTracker):
        if abs(controller.mass_kg - vehicle_mass_kg) > 1.0e-8:
            raise ValueError("Bitcraze Lee mass must match the active Isaac CF2X model")
        controller_state = controller.step(
            position=position,
            velocity=velocity,
            quaternion_wxyz=robot.data.root_quat_w,
            angular_velocity_body=angular_velocity,
            reference_positions=reference_positions,
            reference_velocities=reference_velocities,
            reference_accelerations=reference_accelerations,
            headings_deg=headings_deg,
            dt_s=dt_s,
        )
        feedback_acceleration = controller_state["feedback_accelerations_mps2"]
        desired_acceleration = controller_state["desired_accelerations_mps2"]
        requested_force_world = controller_state["requested_forces_world_n"]
        desired_rotation = controller_state["desired_attitude_matrix"]
        desired_yaw = controller_state["requested_headings_rad"]
        attitude_error = controller_state["so3_attitude_errors"]
        wrench = controller_state["requested_wrenches"].clone()
        wrench[:, 0] = torch.clamp(
            wrench[:, 0], min=0.0, max=MAX_THRUST_PER_ROTOR_N * float(robot.num_thrusters)
        )
        controller_id = str(controller_state["controller_id"])
    else:
        if abs(controller.mass_kg - vehicle_mass_kg) > 1.0e-8:
            raise ValueError("Bitcraze Mellinger mass must match the active Isaac CF2X model")
        controller_state = controller.step_for_physics(
            physics_dt_s=float(dt_s) if dt_s is not None else 1.0 / 120.0,
            position=position,
            velocity=velocity,
            quaternion_wxyz=robot.data.root_quat_w,
            angular_velocity_body=angular_velocity,
            reference_positions=reference_positions,
            reference_velocities=reference_velocities,
            reference_accelerations=reference_accelerations,
            headings_deg=headings_deg,
        )
        feedback_acceleration = controller_state["feedback_accelerations_mps2"]
        desired_acceleration = controller_state["desired_accelerations_mps2"]
        requested_force_world = controller_state["requested_forces_world_n"]
        desired_rotation = controller_state["desired_attitude_matrix"]
        desired_yaw = controller_state["requested_headings_rad"]
        attitude_error = controller_state["so3_attitude_errors"]
        direct_rotor_thrust = controller.rotor_thrust_from_pwm(
            controller_state["legacy_motor_pwm"],
            hover_thrust_per_rotor_n=HOVER_THRUST_PER_ROTOR_N,
        )
        controller_id = str(controller_state["controller_id"])
    desired_roll, desired_pitch, realised_desired_yaw = _euler_xyz_from_rotation_matrix(
        desired_rotation
    )
    allocation = robot.allocation_matrix[[2, 3, 4, 5], :].to(device=robot.device)
    if direct_rotor_thrust is None:
        collective_only = torch.zeros_like(wrench)
        collective_only[:, 0] = wrench[:, 0]
        collective_thrust = torch.linalg.solve(allocation, collective_only.T).T
        attitude_wrench = wrench.clone()
        attitude_wrench[:, 3] = 0.0
        attitude_delta = torch.linalg.solve(allocation, attitude_wrench.T).T - collective_thrust
        attitude_scale = _largest_feasible_scale(collective_thrust, attitude_delta)
        attitude_thrust = collective_thrust + attitude_scale * attitude_delta
        yaw_wrench = torch.zeros_like(wrench)
        yaw_wrench[:, 3] = wrench[:, 3]
        yaw_delta = torch.linalg.solve(allocation, yaw_wrench.T).T
        yaw_scale = _largest_feasible_scale(attitude_thrust, yaw_delta)
        thrust = attitude_thrust + yaw_scale * yaw_delta
    else:
        thrust = direct_rotor_thrust
        wrench = (allocation @ thrust.T).T
        attitude_scale = torch.ones((int(robot.num_instances), 1), device=robot.device)
        yaw_scale = torch.ones((int(robot.num_instances), 1), device=robot.device)
    if not bool(torch.isfinite(thrust).all().item()):
        raise RuntimeError("CF2X thrust allocator returned a non-finite command")
    thrust = torch.clamp(thrust, min=0.0, max=MAX_THRUST_PER_ROTOR_N)
    if diagnostics is not None:
        diagnostics.update(
            {
                "reference_positions_m": reference_positions.detach().clone(),
                "controller_id": controller_id,
                "reference_velocities_mps": reference_velocities.detach().clone(),
                "reference_accelerations_mps2": reference_accelerations.detach().clone(),
                "control_positions_m": position.detach().clone(),
                "control_velocities_mps": velocity.detach().clone(),
                "control_attitude_rpy_rad": torch.stack((roll, pitch, yaw), dim=1).detach().clone(),
                "control_angular_velocities_body_rad_s": angular_velocity.detach().clone(),
                "feedback_accelerations_mps2": feedback_acceleration.detach().clone(),
                "desired_accelerations_mps2": desired_acceleration.detach().clone(),
                "desired_attitude_rpy_rad": torch.stack(
                    (desired_roll, desired_pitch, realised_desired_yaw), dim=1
                )
                .detach()
                .clone(),
                "requested_headings_rad": desired_yaw.detach().clone(),
                "so3_attitude_errors": attitude_error.detach().clone(),
                "requested_forces_world_n": requested_force_world.detach().clone(),
                "requested_wrenches": wrench.detach().clone(),
                "attitude_allocation_scales": attitude_scale.detach().clone(),
                "yaw_allocation_scales": yaw_scale.detach().clone(),
                "rotor_thrust_targets_n": thrust.detach().clone(),
            }
        )
        if controller_state:
            diagnostics["controller_state"] = {
                "source_url": controller_state["source_url"],
                "source_commit": controller_state["source_commit"],
                "source_file": controller_state["source_file"],
                "position_error_m": controller_state["position_error_m"].detach().clone(),
                "velocity_error_mps": controller_state["velocity_error_mps"].detach().clone(),
                "attitude_integral": controller_state["attitude_integral"].detach().clone(),
            }
    return thrust


def _energy_increment_j(thrust_row: list[float], dt_s: float) -> float:
    disc_area = math.pi * ROTOR_RADIUS_M**2
    denominator = math.sqrt(2.0 * AIR_DENSITY_KG_M3 * disc_area)
    return sum(max(0.0, value) ** 1.5 / denominator * dt_s for value in thrust_row)


def _multirotor_cfg(robot_asset: Path, dt_s: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab_contrib.actuators import ThrusterCfg
    from isaaclab_contrib.assets import MultirotorCfg

    yaw_ratio = CF2X_YAW_TORQUE_TO_THRUST_M
    allocation = _cf2x_allocation_matrix()
    return MultirotorCfg(
        prim_path="/World/P07Agents/Env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_asset),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.02,
                angular_damping=0.02,
                max_linear_velocity=6.0,
                max_angular_velocity=12.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            copy_from_source=False,
        ),
        init_state=MultirotorCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            # The episode starts already airborne at rest. Match the actuator's
            # internal state to that frozen hover equilibrium so independent
            # motor rise constants do not create an artificial reset torque.
            rps={name: CF2X_INITIAL_ROTOR_RPS for name in THRUSTER_NAMES},
        ),
        actuators={
            "thrusters": ThrusterCfg(
                dt=dt_s,
                thrust_range=(0.0, MAX_THRUST_PER_ROTOR_N),
                max_thrust_rate=100000.0,
                thrust_const_range=(
                    CF2X_THRUST_CONSTANT_N_PER_RPS2,
                    CF2X_THRUST_CONSTANT_N_PER_RPS2,
                ),
                tau_inc_range=CF2X_THRUSTER_TAU_INC_RANGE_S,
                tau_dec_range=CF2X_THRUSTER_TAU_DEC_RANGE_S,
                torque_to_thrust_ratio=yaw_ratio,
                thruster_names_expr=list(THRUSTER_NAMES),
            )
        },
        allocation_matrix=allocation,
        rotor_directions=[1, -1, 1, -1],
    )


@dataclass(slots=True)
class IsaacCF2XExecutionBackend:
    """Real, target-free, multi-agent CF2X executor for a single manifest."""

    sim: Any
    robot: Any
    contact: Any
    scene_query: Any
    static_clearance_oracle: _EvaluatorStaticClearance
    agent_order: tuple[str, ...]
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    arrival_tolerance_m: float
    execution_deadline_s: float | None = None
    communication_max_range_m: float = 10.0
    communication_base_latency_s: float = 0.05
    communication_per_hop_latency_s: float = 0.02
    communication_loss_probability: float = 0.0
    communication_update_hz: float = 10.0
    sparse_range_update_hz: float = 10.0
    sparse_range_directions: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: resolve_public_range_directions(DENSE_26_RAY_PATTERN)
    )
    sparse_range_max_m: float = 20.0
    communication_message_ttl_s: float = 0.5
    minimum_observation_dwell_s: float = 1.0
    event_driven_action_completion: bool = True
    controller_id: str = CF2X_DEFAULT_CONTROLLER_ID
    # Optional per-agent rolling continuation: when a vehicle finishes its
    # transit+observe pair inside the physical execution window, this
    # callback is invoked with (agent_id, timestamp_s).  It returns the next
    # (transit, observe) fragment pair or None to wait.  The vehicle then
    # keeps flying the next pair in the same simulation window, so fast
    # vehicles never idle behind a slow one (async completion).  When None,
    # the executor behaves exactly as before (synchronous team completion).
    on_agent_complete: (
        Callable[[str, float, tuple[float, float, float]], tuple[FragmentInstance, FragmentInstance] | None]
        | None
    ) = None
    # This observer is deliberately opt-in and audit-only.  It records already
    # realised root states after PhysX steps; it never supplies data to the
    # controller, planner, public belief, safety guard, or reward path.
    visualization_trace_sample_hz: float | None = None
    backend_id: str = CF2X_EXECUTION_BACKEND_ID
    evidence_class: str = CF2X_EXECUTION_EVIDENCE_CLASS
    engineering_diagnostics: dict[str, object] = field(default_factory=dict, init=False)
    public_range_frames: tuple[PublicRangeObservationFrameOutcome, ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    public_range_outcomes: tuple[PublicRangeRayOutcome, ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    public_map_sender_ids: tuple[str, ...] = field(default_factory=tuple, init=False)
    final_root_positions_m: tuple[tuple[float, float, float], ...] = field(
        default_factory=tuple, init=False
    )
    final_root_linear_speeds_mps: tuple[float, ...] = field(default_factory=tuple, init=False)
    last_execution_samples: tuple[FragmentExecutionSample, ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _controller: BitcrazeLeeTracker | BitcrazeMellingerTracker | None = field(
        default=None, init=False, repr=False
    )

    def _communication_graph(
        self, positions: list[tuple[float, float, float]]
    ) -> RelayGraphSnapshot:
        """Measure a range-limited relay graph against static HM3D geometry.

        A direct edge ends just before the receiver position so that endpoint
        precision does not turn the receiver itself into an apparent wall.
        CF2X collider hits are skipped because the communication contract does
        not model airframe RF shadowing; walls remain authoritative blockers.
        The resulting graph is still public runtime telemetry, never target
        truth or a hidden global map.
        """

        def line_of_sight_clear(
            source: tuple[float, float, float], target: tuple[float, float, float]
        ) -> bool:
            return _clear_static_collision_los(
                self.scene_query,
                source,
                target,
            )

        return build_range_los_relay_graph(
            positions,
            maximum_range_m=self.communication_max_range_m,
            line_of_sight_clear=line_of_sight_clear,
        )

    def execute_manifest(
        self, manifest: CandidateFragmentManifest, token: Any
    ) -> tuple[FragmentExecutionSample, ...]:
        import torch

        self.last_execution_samples = ()
        _controller_tracking_profile(self.controller_id, physics_dt_s=float(self.sim.cfg.dt))
        if self.visualization_trace_sample_hz is not None and (
            not math.isfinite(self.visualization_trace_sample_hz)
            or self.visualization_trace_sample_hz <= 0.0
        ):
            raise ValueError("visualization trace sample frequency must be finite and positive")
        controller_mass_kg = (HOVER_THRUST_PER_ROTOR_N * float(self.robot.num_thrusters)) / 9.81
        if self.controller_id == BITCRAZE_LEE_CONTROLLER_ID:
            self._controller = BitcrazeLeeTracker(
                mass_kg=controller_mass_kg,
                dt_s=float(self.sim.cfg.dt),
                maximum_feedback_acceleration_mps2=CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
                maximum_tilt_rad=CF2X_MAXIMUM_TILT_RAD,
            )
        elif self.controller_id == BITCRAZE_MELLINGER_CONTROLLER_ID:
            self._controller = BitcrazeMellingerTracker(
                mass_kg=controller_mass_kg,
                dt_s=1.0 / BITCRAZE_MELLINGER_OFFICIAL_CONTROL_RATE_HZ,
            )
        else:
            self._controller = None

        by_agent: dict[str, list[FragmentInstance]] = defaultdict(list)
        for fragment in manifest.fragments:
            by_agent[fragment.agent_id].append(fragment)
        if tuple(sorted(by_agent)) != self.agent_order:
            raise ValueError("manifest agents do not match the spawned CF2X fleet")
        routes: list[tuple[FragmentInstance, FragmentInstance]] = []
        for agent_id in self.agent_order:
            fragments = sorted(by_agent[agent_id], key=lambda row: row.planned_start)
            if len(fragments) != 2 or [row.type_signature.fragment_type for row in fragments] != [
                "transit",
                "observation",
            ]:
                raise ValueError(
                    "P07 execution smoke supports one transit plus one observation per UAV"
                )
            routes.append((fragments[0], fragments[1]))
        route_corner_speed_mps = [
            _route_corner_speed_mps(transit.path) for transit, _ in routes
        ]
        agent_index_by_id = {agent_id: index for index, agent_id in enumerate(self.agent_order)}
        traffic_predecessor_index: list[int | None] = []
        traffic_reservation_delay_s: list[float] = []
        for index, (transit, _) in enumerate(routes):
            features = dict(transit.type_signature.public_features)
            delay_s = float(features.get("traffic_reservation_delay_s", 0.0))
            predecessor_agent_id = str(
                features.get("traffic_reservation_predecessor_agent_id", "")
            )
            if not math.isfinite(delay_s) or delay_s < 0.0:
                raise ValueError("traffic reservation delay must be finite and non-negative")
            if delay_s <= 1.0e-12:
                if predecessor_agent_id:
                    raise ValueError("un-delayed transit cannot name a traffic predecessor")
                traffic_predecessor_index.append(None)
                traffic_reservation_delay_s.append(0.0)
                continue
            predecessor_index = agent_index_by_id.get(predecessor_agent_id)
            if predecessor_index is None or predecessor_index == index:
                raise ValueError("traffic reservation predecessor must name another fleet agent")
            if not math.isclose(delay_s, transit.planned_start, abs_tol=1.0e-9):
                raise ValueError("traffic reservation delay must equal its transit planned start")
            predecessor_transit = routes[predecessor_index][0]
            if transit.planned_start + 1.0e-9 < (
                predecessor_transit.planned_end + TRAFFIC_RESERVATION_MINIMUM_RELEASE_MARGIN_S
            ):
                raise ValueError("traffic reservation begins before predecessor planned arrival")
            traffic_predecessor_index.append(predecessor_index)
            traffic_reservation_delay_s.append(delay_s)
        execution_horizon_s = (
            token.duration if self.execution_deadline_s is None else self.execution_deadline_s
        )
        if not 0.0 < execution_horizon_s <= token.duration:
            raise ValueError("execution deadline must lie in (0, token.duration]")
        if (
            not math.isfinite(self.minimum_observation_dwell_s)
            or self.minimum_observation_dwell_s <= 0.0
        ):
            raise ValueError("minimum observation dwell must be positive")
        if not math.isfinite(self.communication_update_hz) or self.communication_update_hz <= 0.0:
            raise ValueError("communication update frequency must be positive")
        maximum_steps = max(1, math.floor(execution_horizon_s / float(self.sim.cfg.dt)))
        final_physics_timestamp_s = maximum_steps * float(self.sim.cfg.dt)
        transit_traces = [[tuple(fragment.path[0])] for fragment, _ in routes]
        observation_traces: list[list[tuple[float, float, float]]] = [[] for _ in routes]
        energy = [0.0 for _ in routes]
        fragment_connected_at_every_telemetry_tick = [True for _ in routes]
        transit_end: list[float | None] = [None for _ in routes]
        observation_start: list[float | None] = [None for _ in routes]
        observation_end: list[float | None] = [None for _ in routes]
        transit_contact = [False for _ in routes]
        transit_oob = [False for _ in routes]
        observation_contact = [False for _ in routes]
        observation_oob = [False for _ in routes]
        inter_agent_separation_violation = [False for _ in routes]
        failed = [False for _ in routes]
        transit_waypoint_index = [1 for _ in routes]
        segment_start_s = [0.0 for _ in routes]
        segment_start_m = [tuple(fragment.path[0]) for fragment, _ in routes]
        segment_initial_speed_mps = [0.0 for _ in routes]
        transit_release_s: list[float | None] = [None for _ in routes]
        reservation_wait_steps = [0 for _ in routes]
        reservation_max_prestart_displacement_m = [0.0 for _ in routes]
        first_collision_step: list[int | None] = [None for _ in routes]
        first_collision_position: list[tuple[float, float, float] | None] = [None for _ in routes]
        first_collision_waypoint: list[tuple[float, float, float] | None] = [None for _ in routes]
        minimum_inter_agent_distance_m = math.inf
        first_inter_agent_separation_violation: dict[str, object] | None = None
        waypoint_transitions: list[list[dict[str, object]]] = [[] for _ in routes]
        maximum_contact_force_n = [0.0 for _ in routes]
        maximum_linear_speed_mps = [0.0 for _ in routes]
        maximum_linear_acceleration_mps2 = [0.0 for _ in routes]
        rolling_agent_decision_count = [0 for _ in routes]
        sample_ledger: list[FragmentExecutionSample] = []
        controller_tracking_samples: list[list[dict[str, object]]] = [[] for _ in routes]
        controller_tracking_sample_interval_s = 1.0 / CONTROLLER_TRACKING_TELEMETRY_HZ
        next_controller_tracking_sample_s = 0.0
        visualization_trace_sample_interval_s = (
            None
            if self.visualization_trace_sample_hz is None
            else 1.0 / self.visualization_trace_sample_hz
        )
        next_visualization_trace_sample_s = 0.0
        visualization_trace_samples: list[dict[str, object]] = []
        last_sensor_timestamp: list[float | None] = [None for _ in routes]
        last_sensor_source_id: list[str | None] = [None for _ in routes]
        sensor_frames_by_agent = [0 for _ in routes]
        range_frames_by_phase = {"transit": 0, "dwell": 0}
        public_range_frames: list[PublicRangeObservationFrameOutcome] = []
        public_range_outcomes: list[PublicRangeRayOutcome] = []
        source_observation_ids_by_agent = {agent_id: [] for agent_id in self.agent_order}
        message_queue = RelayMessageQueue(
            self.agent_order,
            self.communication_base_latency_s,
            self.communication_per_hop_latency_s,
            self.communication_loss_probability,
        )
        relay_measurement_count = 0
        relay_fully_connected_count = 0
        relay_direct_link_count_sum = 0
        relay_component_count_max = 0
        relay_maximum_hops_max = 0
        current_disconnect_started_s: float | None = None
        longest_disconnected_duration_s = 0.0
        partition_event_count = 0
        reconnection_count = 0
        previous_relay_connected: bool | None = None
        last_relay_graph: RelayGraphSnapshot | None = None
        communication_interval_s = 1.0 / self.communication_update_hz
        last_communication_measurement_s: float | None = None

        def record_relay_measurement(
            timestamp_s: float,
            positions_m: list[tuple[float, float, float]],
        ) -> RelayGraphSnapshot:
            nonlocal current_disconnect_started_s
            nonlocal last_communication_measurement_s
            nonlocal last_relay_graph
            nonlocal longest_disconnected_duration_s
            nonlocal partition_event_count
            nonlocal previous_relay_connected
            nonlocal reconnection_count
            nonlocal relay_component_count_max
            nonlocal relay_direct_link_count_sum
            nonlocal relay_fully_connected_count
            nonlocal relay_maximum_hops_max
            nonlocal relay_measurement_count

            graph = self._communication_graph(positions_m)
            relay_measurement_count += 1
            relay_fully_connected_count += int(graph.fully_relay_connected)
            relay_direct_link_count_sum += graph.direct_link_count
            relay_component_count_max = max(relay_component_count_max, len(graph.components))
            relay_maximum_hops_max = max(relay_maximum_hops_max, graph.maximum_relay_hops or 0)
            if graph.fully_relay_connected:
                if current_disconnect_started_s is not None:
                    longest_disconnected_duration_s = max(
                        longest_disconnected_duration_s,
                        timestamp_s - current_disconnect_started_s,
                    )
                    current_disconnect_started_s = None
                if previous_relay_connected is False:
                    reconnection_count += 1
            else:
                if current_disconnect_started_s is None:
                    # A sampled graph cannot locate the partition inside the
                    # preceding interval. Use its lower bound so the recorded
                    # duration is conservative without claiming continuous RF.
                    current_disconnect_started_s = max(0.0, timestamp_s - communication_interval_s)
                if previous_relay_connected is True:
                    partition_event_count += 1
            previous_relay_connected = graph.fully_relay_connected
            last_relay_graph = graph
            last_communication_measurement_s = timestamp_s
            return graph

        initial_positions = [
            tuple(float(value) for value in row)
            for row in self.robot.data.root_pos_w.detach().cpu().tolist()
        ]
        _, _, initial_yaw_rad = _euler_xyz_from_quaternion_wxyz(self.robot.data.root_quat_w)
        heading_references_deg = torch.rad2deg(initial_yaw_rad).detach().cpu().tolist()
        previous_linear_velocity = self.robot.data.root_lin_vel_w.detach().clone()
        initial_relay_graph = record_relay_measurement(0.0, initial_positions)
        for index, connected in enumerate(initial_relay_graph.agent_relay_reachable_to_all):
            fragment_connected_at_every_telemetry_tick[index] = connected

        def append_visualization_trace_sample(
            *,
            timestamp_s: float,
            positions_m: list[tuple[float, float, float]],
            linear_speeds_mps: Any,
            reservation_waiting: list[bool],
        ) -> None:
            """Record observed root state for a post-run visual audit only."""

            if visualization_trace_sample_interval_s is None:
                return
            root_quaternions = self.robot.data.root_quat_w.detach().cpu().tolist()
            visualization_trace_samples.append(
                {
                    "physics_timestamp_s": timestamp_s,
                    "agents": [
                        {
                            "agent_id": self.agent_order[index],
                            "position_m": list(positions_m[index]),
                            "quaternion_wxyz": [
                                float(value) for value in root_quaternions[index]
                            ],
                            "linear_speed_mps": float(linear_speeds_mps[index].item()),
                            "reservation_waiting": bool(reservation_waiting[index]),
                            "transit_completed": transit_end[index] is not None,
                            "failed": bool(failed[index]),
                        }
                        for index in range(len(routes))
                    ],
                    "minimum_inter_agent_distance_m": min(
                        (
                            math.dist(positions_m[left], positions_m[right])
                            for left in range(len(positions_m))
                            for right in range(left + 1, len(positions_m))
                        ),
                        default=math.inf,
                    ),
                }
            )

        if visualization_trace_sample_interval_s is not None:
            initial_speeds = torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=1)
            append_visualization_trace_sample(
                timestamp_s=0.0,
                positions_m=initial_positions,
                linear_speeds_mps=initial_speeds,
                reservation_waiting=[False for _ in routes],
            )
            next_visualization_trace_sample_s = visualization_trace_sample_interval_s
        for step in range(1, maximum_steps + 1):
            reference_positions = []
            reference_velocities = []
            reference_accelerations = []
            destinations = []
            reservation_waiting = [False for _ in routes]
            headings = []
            control_timestamp_s = (step - 1) * float(self.sim.cfg.dt)
            for index, (transit, observe) in enumerate(routes):
                predecessor_index = traffic_predecessor_index[index]
                predecessor_completed = (
                    predecessor_index is None or transit_end[predecessor_index] is not None
                )
                authorized_to_depart = (
                    control_timestamp_s + 1.0e-12 >= transit.planned_start
                    and predecessor_completed
                )
                reservation_waiting[index] = (
                    transit_end[index] is None and not failed[index] and not authorized_to_depart
                )
                if failed[index] or reservation_waiting[index]:
                    destination = transit_traces[index][-1]
                else:
                    destination = (
                        transit.path[transit_waypoint_index[index]]
                        if transit_end[index] is None
                        else observe.path[0]
                    )
                destinations.append(destination)
                if failed[index] or reservation_waiting[index]:
                    reference_positions.append(tuple(destination))
                    reference_velocities.append((0.0, 0.0, 0.0))
                    reference_accelerations.append((0.0, 0.0, 0.0))
                elif transit_end[index] is None:
                    if transit_release_s[index] is None:
                        transit_release_s[index] = control_timestamp_s
                        segment_start_s[index] = control_timestamp_s
                        segment_start_m[index] = tuple(transit_traces[index][-1])
                        segment_initial_speed_mps[index] = 0.0
                    terminal_segment = (
                        transit_waypoint_index[index] + 1 >= len(transit.path)
                    )
                    terminal_speed_mps = (
                        0.0 if terminal_segment else route_corner_speed_mps[index]
                    )
                    reference = _minimum_time_line_reference_with_boundary_speeds(
                        segment_start_m[index],
                        tuple(destination),
                        max(0.0, control_timestamp_s - segment_start_s[index]),
                        initial_speed_mps=segment_initial_speed_mps[index],
                        terminal_speed_mps=terminal_speed_mps,
                    )
                    reference_positions.append(reference.position_m)
                    reference_velocities.append(reference.velocity_mps)
                    reference_accelerations.append(reference.acceleration_mps2)
                else:
                    reference_positions.append(tuple(destination))
                    reference_velocities.append((0.0, 0.0, 0.0))
                    reference_accelerations.append((0.0, 0.0, 0.0))
                horizontal_delta_m = math.hypot(
                    destination[0] - transit_traces[index][-1][0],
                    destination[1] - transit_traces[index][-1][1],
                )
                target_heading_deg = (
                    _yaw_from_delta(transit_traces[index][-1], destination)
                    if horizontal_delta_m > 1.0e-9
                    else heading_references_deg[index]
                )
                heading_references_deg[index] = _rate_limited_yaw_reference_deg(
                    heading_references_deg[index],
                    target_heading_deg,
                    float(self.sim.cfg.dt),
                )
                headings.append(heading_references_deg[index])
            capture_controller_tracking = (
                control_timestamp_s + 1.0e-12 >= next_controller_tracking_sample_s
            )
            controller_step_diagnostics: dict[str, Any] = {}
            thrust = _bounded_rotor_thrust(
                self.robot,
                torch.tensor(reference_positions, device=self.robot.device, dtype=torch.float32),
                torch.tensor(reference_velocities, device=self.robot.device, dtype=torch.float32),
                torch.tensor(
                    reference_accelerations,
                    device=self.robot.device,
                    dtype=torch.float32,
                ),
                headings,
                controller_step_diagnostics if capture_controller_tracking else None,
                self._controller,
                float(self.sim.cfg.dt),
            )
            self.robot.set_thrust_target(thrust)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(float(self.sim.cfg.dt))
            self.contact.update(float(self.sim.cfg.dt), force_recompute=True)
            timestamp = step * float(self.sim.cfg.dt)
            positions = [
                tuple(float(value) for value in row)
                for row in self.robot.data.root_pos_w.detach().cpu().tolist()
            ]
            linear_velocity = self.robot.data.root_lin_vel_w
            linear_speeds = torch.linalg.norm(linear_velocity, dim=1)
            linear_accelerations = torch.linalg.norm(
                (linear_velocity - previous_linear_velocity) / float(self.sim.cfg.dt),
                dim=1,
            )
            previous_linear_velocity = linear_velocity.detach().clone()
            contact_forces = (
                torch.linalg.norm(self.contact.data.net_forces_w, dim=-1).max(dim=1).values
            )
            for left_index in range(len(positions)):
                for right_index in range(left_index + 1, len(positions)):
                    pair_distance_m = math.dist(positions[left_index], positions[right_index])
                    minimum_inter_agent_distance_m = min(
                        minimum_inter_agent_distance_m, pair_distance_m
                    )
                    if pair_distance_m + 1.0e-12 < CF2X_MIN_INTER_AGENT_SEPARATION_M:
                        inter_agent_separation_violation[left_index] = True
                        inter_agent_separation_violation[right_index] = True
                        if first_inter_agent_separation_violation is None:
                            first_inter_agent_separation_violation = {
                                "physics_timestamp_s": timestamp,
                                "agent_ids": [
                                    self.agent_order[left_index],
                                    self.agent_order[right_index],
                                ],
                                "distance_m": pair_distance_m,
                                "required_distance_m": CF2X_MIN_INTER_AGENT_SEPARATION_M,
                                "positions_m": [positions[left_index], positions[right_index]],
                            }
            communication_due = (
                last_communication_measurement_s is None
                or timestamp - last_communication_measurement_s + 1.0e-12
                >= communication_interval_s
            )
            relay_graph = (
                record_relay_measurement(timestamp, positions) if communication_due else None
            )
            if (
                visualization_trace_sample_interval_s is not None
                and timestamp + 1.0e-12 >= next_visualization_trace_sample_s
            ):
                append_visualization_trace_sample(
                    timestamp_s=timestamp,
                    positions_m=positions,
                    linear_speeds_mps=linear_speeds,
                    reservation_waiting=reservation_waiting,
                )
                next_visualization_trace_sample_s += visualization_trace_sample_interval_s
            thrust_rows = thrust.detach().cpu().tolist()
            if capture_controller_tracking:
                controller_rows = {
                    key: value.detach().cpu().tolist()
                    for key, value in controller_step_diagnostics.items()
                    if hasattr(value, "detach")
                }
                sampled_controller_id = str(controller_step_diagnostics["controller_id"])
                post_step_velocities = linear_velocity.detach().cpu().tolist()
                for index, position in enumerate(positions):
                    if transit_end[index] is not None or failed[index]:
                        continue
                    rotor_targets = controller_rows["rotor_thrust_targets_n"][index]
                    saturated_rotor_count = sum(
                        value <= 1.0e-6 or value >= MAX_THRUST_PER_ROTOR_N - 1.0e-6
                        for value in rotor_targets
                    )
                    controller_tracking_samples[index].append(
                        {
                            "controller_id": sampled_controller_id,
                            "control_timestamp_s": control_timestamp_s,
                            "physics_timestamp_s": timestamp,
                            "reference_position_m": controller_rows["reference_positions_m"][index],
                            "reference_velocity_mps": controller_rows["reference_velocities_mps"][
                                index
                            ],
                            "reference_acceleration_mps2": controller_rows[
                                "reference_accelerations_mps2"
                            ][index],
                            "control_position_m": controller_rows["control_positions_m"][index],
                            "control_velocity_mps": controller_rows["control_velocities_mps"][
                                index
                            ],
                            "post_step_position_m": position,
                            "post_step_velocity_mps": post_step_velocities[index],
                            "feedback_acceleration_mps2": controller_rows[
                                "feedback_accelerations_mps2"
                            ][index],
                            "desired_acceleration_mps2": controller_rows[
                                "desired_accelerations_mps2"
                            ][index],
                            "control_attitude_rpy_rad": controller_rows["control_attitude_rpy_rad"][
                                index
                            ],
                            "desired_attitude_rpy_rad": controller_rows["desired_attitude_rpy_rad"][
                                index
                            ],
                            "requested_heading_rad": controller_rows["requested_headings_rad"][
                                index
                            ],
                            "so3_attitude_error": controller_rows["so3_attitude_errors"][index],
                            "requested_force_world_n": controller_rows["requested_forces_world_n"][
                                index
                            ],
                            "control_angular_velocity_body_rad_s": controller_rows[
                                "control_angular_velocities_body_rad_s"
                            ][index],
                            "requested_wrench": controller_rows["requested_wrenches"][index],
                            "attitude_allocation_scale": controller_rows[
                                "attitude_allocation_scales"
                            ][index][0],
                            "yaw_allocation_scale": controller_rows["yaw_allocation_scales"][index][
                                0
                            ],
                            "rotor_thrust_targets_n": rotor_targets,
                            "saturated_rotor_count": saturated_rotor_count,
                        }
                    )
                next_controller_tracking_sample_s += controller_tracking_sample_interval_s
            for index, position in enumerate(positions):
                energy[index] += _energy_increment_j(thrust_rows[index], float(self.sim.cfg.dt))
                maximum_linear_speed_mps[index] = max(
                    maximum_linear_speed_mps[index],
                    float(linear_speeds[index].item()),
                )
                maximum_linear_acceleration_mps2[index] = max(
                    maximum_linear_acceleration_mps2[index],
                    float(linear_accelerations[index].item()),
                )
                if relay_graph is not None:
                    fragment_connected_at_every_telemetry_tick[index] = (
                        fragment_connected_at_every_telemetry_tick[index]
                        and relay_graph.agent_relay_reachable_to_all[index]
                    )
                out_of_bounds = any(
                    position[axis] < self.bounds_min_m[axis] - 1.0e-6
                    or position[axis] > self.bounds_max_m[axis] + 1.0e-6
                    for axis in range(3)
                )
                collided = float(contact_forces[index].item()) > CONTACT_HARD_FAIL_N
                maximum_contact_force_n[index] = max(
                    maximum_contact_force_n[index], float(contact_forces[index].item())
                )
                if collided and first_collision_step[index] is None:
                    first_collision_step[index] = step
                    first_collision_position[index] = position
                    first_collision_waypoint[index] = destinations[index]
                if transit_end[index] is None:
                    transit_traces[index].append(position)
                    transit_contact[index] = transit_contact[index] or collided
                    transit_oob[index] = transit_oob[index] or out_of_bounds
                    if collided or out_of_bounds or inter_agent_separation_violation[index]:
                        failed[index] = True
                        continue
                    if reservation_waiting[index]:
                        reservation_wait_steps[index] += 1
                        reservation_max_prestart_displacement_m[index] = max(
                            reservation_max_prestart_displacement_m[index],
                            math.dist(position, transit.path[0]),
                        )
                        continue
                    waypoint = routes[index][0].path[transit_waypoint_index[index]]
                    error = math.dist(position, waypoint)
                    speed_mps = float(linear_speeds[index].item())
                    # The route-level plan keeps moving through intermediate
                    # corners at a geometry-bounded pass-through speed. Only the
                    # terminal waypoint requires the calibrated settle condition.
                    terminal_segment = (
                        transit_waypoint_index[index] + 1
                        >= len(routes[index][0].path)
                    )
                    waypoint_requires_settle = terminal_segment
                    if _waypoint_reached(
                        error_m=error,
                        speed_mps=speed_mps,
                        requires_settle=waypoint_requires_settle,
                        arrival_tolerance_m=self.arrival_tolerance_m,
                    ):
                        waypoint_transitions[index].append(
                            {
                                "waypoint_index": transit_waypoint_index[index],
                                "timestamp_s": timestamp,
                                "position_m": position,
                                "error_m": error,
                                "speed_mps": speed_mps,
                                "stop_required": waypoint_requires_settle,
                                "position_tolerance_m": (
                                    WAYPOINT_SETTLE_POSITION_TOLERANCE_M
                                    if waypoint_requires_settle
                                    else self.arrival_tolerance_m
                                ),
                            }
                        )
                        if not terminal_segment:
                            transit_waypoint_index[index] += 1
                            segment_start_s[index] = timestamp
                            segment_start_m[index] = position
                            segment_initial_speed_mps[index] = min(
                                max(speed_mps, 0.0),
                                route_corner_speed_mps[index],
                            )
                        else:
                            transit_end[index] = timestamp
                            observation_start[index] = timestamp
                            observation_traces[index].append(position)
                elif observation_end[index] is None:
                    observation_traces[index].append(position)
                    observation_contact[index] = observation_contact[index] or collided
                    observation_oob[index] = observation_oob[index] or out_of_bounds
                    if collided or out_of_bounds or inter_agent_separation_violation[index]:
                        failed[index] = True
                        continue
                    observation_complete = (
                        _minimum_observation_dwell_completed(
                            timestamp_s=timestamp,
                            actual_start_s=observation_start[index],
                            minimum_dwell_s=self.minimum_observation_dwell_s,
                        )
                        if self.event_driven_action_completion
                        else _scheduled_observation_completed(
                            timestamp_s=timestamp,
                            planned_end_s=observe.planned_end,
                            actual_start_s=observation_start[index],
                            minimum_dwell_s=self.minimum_observation_dwell_s,
                            final_physics_timestamp_s=final_physics_timestamp_s,
                        )
                    )
                    if observation_complete:
                        observation_end[index] = timestamp
                        if (
                            self.on_agent_complete is not None
                            and not failed[index]
                            and not transit_contact[index]
                            and not transit_oob[index]
                            and not observation_contact[index]
                            and not observation_oob[index]
                            and not inter_agent_separation_violation[index]
                        ):
                            next_pair = self.on_agent_complete(
                                routes[index][1].agent_id,
                                timestamp,
                                position,
                            )
                            if next_pair is not None:
                                # Roll this vehicle onto its next pair inside
                                # the same physical window.  The finished pair
                                # is finalised into the outcome ledger first,
                                # then the per-agent execution state resets so
                                # the next pair starts from the measured pose.
                                # Static clearance is queried now (the batch
                                # clearance pass runs only after the loop), and
                                # the next segment starts from the measured
                                # speed, not from rest.
                                roll_trace = tuple(transit_traces[index])
                                roll_distances = (
                                    self.static_clearance_oracle.exact_static_distances_m(
                                        roll_trace
                                    )
                                )
                                roll_min_clearance = min(roll_distances)
                                roll_clearance_violation = (
                                    roll_min_clearance + 1.0e-12 < FLIGHT_CLEARANCE_M
                                )
                                _finalize_fragment_pair_into(
                                    sample_ledger,
                                    index=index,
                                    transit=routes[index][0],
                                    observe=routes[index][1],
                                    transit_completed=transit_end[index] is not None,
                                    observation_completed=True,
                                    transit_trace=roll_trace,
                                    observation_trace=tuple(observation_traces[index]),
                                    transit_release_s=transit_release_s[index],
                                    transit_end_s=transit_end[index],
                                    observation_start_s=observation_start[index],
                                    observation_end_s=observation_end[index],
                                    execution_horizon_s=execution_horizon_s,
                                    energy_j=energy[index],
                                    collision=transit_contact[index],
                                    out_of_bounds=transit_oob[index],
                                    separation_violation=inter_agent_separation_violation[index],
                                    static_clearance_contract_violation=roll_clearance_violation,
                                    minimum_clearance_m=roll_min_clearance,
                                    connected_at_every_tick=(
                                        fragment_connected_at_every_telemetry_tick[index]
                                    ),
                                    last_sensor_source_id=last_sensor_source_id[index],
                                    rolling=False,
                                )
                                new_transit, new_observe = next_pair
                                routes[index] = (new_transit, new_observe)
                                route_corner_speed_mps[index] = _route_corner_speed_mps(
                                    new_transit.path
                                )
                                transit_traces[index] = [tuple(transit_traces[index][-1])]
                                observation_traces[index] = []
                                energy[index] = 0.0
                                fragment_connected_at_every_telemetry_tick[index] = True
                                transit_end[index] = None
                                observation_start[index] = None
                                observation_end[index] = None
                                transit_contact[index] = False
                                transit_oob[index] = False
                                observation_contact[index] = False
                                observation_oob[index] = False
                                transit_waypoint_index[index] = 1
                                segment_start_s[index] = timestamp
                                segment_start_m[index] = tuple(transit_traces[index][-1])
                                segment_initial_speed_mps[index] = float(
                                    linear_speeds[index].item()
                                )
                                transit_release_s[index] = timestamp
                                waypoint_transitions[index] = []
                                rolling_agent_decision_count[index] += 1
            # Sparse range is a real (non-rendering) public sensor profile.
            # The evaluator retains source-bound pose outcomes; the shared-map
            # payload is a public digest so this network layer never reads task
            # truth or evaluator geometry.
            for index, position in enumerate(positions):
                team_awaiting = not all(
                    failed[other] or observation_end[other] is not None
                    for other in range(len(routes))
                )
                sampling_phase = _sparse_range_sampling_phase(
                    transit_completed=transit_end[index] is not None,
                    observation_completed=observation_end[index] is not None,
                    failed=failed[index],
                    reservation_waiting=reservation_waiting[index],
                    team_awaiting=team_awaiting,
                )
                if sampling_phase is None:
                    continue
                previous = last_sensor_timestamp[index]
                if (
                    previous is not None
                    and timestamp - previous + 1.0e-12 < 1.0 / self.sparse_range_update_hz
                ):
                    continue
                source_id = (
                    f"range-{manifest.manifest_hash[:12]}-{routes[index][1].agent_id}"
                    f"-{sensor_frames_by_agent[index]:04d}"
                )
                range_distances = []
                for direction_index, direction in enumerate(self.sparse_range_directions):
                    # Public sparse range is an environment sensor. A raw
                    # closest-hit query also sees CF2X bodies spawned in the
                    # same PhysX stage, making paired runs sensitive to tiny
                    # hover/contact differences. Skip dynamic P07Agents hits
                    # here; dynamic safety remains measured by contacts and
                    # synchronized separation below.
                    target = tuple(
                        position[axis] + direction[axis] * self.sparse_range_max_m
                        for axis in range(3)
                    )
                    hit = _first_static_scene_hit(
                        self.scene_query,
                        position,
                        target,
                        endpoint_margin_m=0.0,
                    )
                    hit_occupied = bool(hit is not None and hit.get("hit", False))
                    distance = (
                        float(hit.get("distance", self.sparse_range_max_m))
                        if hit_occupied
                        else self.sparse_range_max_m
                    )
                    if not math.isfinite(distance) or distance <= 0.02:
                        continue
                    range_distances.append(distance)
                    endpoint = tuple(
                        position[axis] + direction[axis] * distance for axis in range(3)
                    )
                    public_range_outcomes.append(
                        PublicRangeRayOutcome(
                            observation_id=f"{source_id}-ray{direction_index}",
                            agent_id=routes[index][1].agent_id,
                            timestamp_s=timestamp,
                            origin_m=position,
                            endpoint_m=endpoint,
                            hit_occupied=hit_occupied,
                        )
                    )
                if not range_distances:
                    continue
                public_range_frames.append(
                    PublicRangeObservationFrameOutcome(
                        observation_frame_id=source_id,
                        agent_id=routes[index][1].agent_id,
                        timestamp_s=timestamp,
                        sensor_position_m=position,
                        ray_count=len(range_distances),
                    )
                )
                source_observation_ids_by_agent[routes[index][1].agent_id].append(source_id)
                last_sensor_timestamp[index] = timestamp
                last_sensor_source_id[index] = source_id
                sensor_frames_by_agent[index] += 1
                range_frames_by_phase[sampling_phase] += 1
            if all(
                failed[index] or observation_end[index] is not None for index in range(len(routes))
            ):
                break
        final_timestamp_s = timestamp
        if last_communication_measurement_s != final_timestamp_s:
            final_relay_graph = record_relay_measurement(final_timestamp_s, positions)
            for index, connected in enumerate(final_relay_graph.agent_relay_reachable_to_all):
                fragment_connected_at_every_telemetry_tick[index] = (
                    fragment_connected_at_every_telemetry_tick[index] and connected
                )
        else:
            if last_relay_graph is None:
                raise RuntimeError("CF2X execution produced no relay telemetry")
            final_relay_graph = last_relay_graph
        if current_disconnect_started_s is not None:
            longest_disconnected_duration_s = max(
                longest_disconnected_duration_s,
                final_timestamp_s - current_disconnect_started_s,
            )
        # A PhysX scene query after vehicle spawn can hit a CF2X's own collider
        # at zero distance. Measure all actual root poses against only the
        # immutable static HM3D collision mesh instead. Query after collection
        # so the same evaluator batch/cache used at admission remains efficient.
        actual_trace_clearance_m: list[float] = []
        static_trace_clearance_by_agent: dict[str, dict[str, object]] = {}
        for index, (transit, _) in enumerate(routes):
            trace = tuple(transit_traces[index] + observation_traces[index])
            if not trace:
                raise RuntimeError(f"missing physical trace for {transit.agent_id}")
            distances = self.static_clearance_oracle.exact_static_distances_m(trace)
            minimum_index = min(range(len(distances)), key=distances.__getitem__)
            minimum_distance_m = distances[minimum_index]
            static_clearance_contract_violation = minimum_distance_m + 1.0e-12 < FLIGHT_CLEARANCE_M
            actual_trace_clearance_m.append(minimum_distance_m)
            static_trace_clearance_by_agent[transit.agent_id] = {
                "minimum_static_mesh_clearance_m": minimum_distance_m,
                "minimum_clearance_position_m": trace[minimum_index],
                "trace_pose_count": len(distances),
                "static_clearance_contract_required_m": FLIGHT_CLEARANCE_M,
                "static_clearance_contract_violation": static_clearance_contract_violation,
            }
        static_trace_clearance = {
            "method": "exact_same_static_collision_mesh_at_each_physics_trace_pose_v1",
            "scope": (
                "Root-position samples at the physics integration cadence; this is distinct from "
                "the continuous planned-centreline admission certificate."
            ),
            "vehicle_self_collider_excluded": True,
            "per_agent": static_trace_clearance_by_agent,
            "minimum_static_mesh_clearance_m": min(actual_trace_clearance_m),
            "static_clearance_contract_required_m": FLIGHT_CLEARANCE_M,
            "static_clearance_contract_passed": all(
                not bool(row["static_clearance_contract_violation"])
                for row in static_trace_clearance_by_agent.values()
            ),
        }
        samples: list[FragmentExecutionSample] = []
        for index, (transit, observe) in enumerate(routes):
            _finalize_fragment_pair_into(
                samples,
                index=index,
                transit=transit,
                observe=observe,
                transit_completed=transit_end[index] is not None,
                observation_completed=observation_end[index] is not None,
                transit_trace=tuple(transit_traces[index]),
                observation_trace=tuple(observation_traces[index]),
                transit_release_s=transit_release_s[index],
                transit_end_s=transit_end[index],
                observation_start_s=observation_start[index],
                observation_end_s=observation_end[index],
                execution_horizon_s=execution_horizon_s,
                energy_j=energy[index],
                collision=transit_contact[index] or observation_contact[index],
                out_of_bounds=transit_oob[index] or observation_oob[index],
                separation_violation=inter_agent_separation_violation[index],
                static_clearance_contract_violation=bool(
                    static_trace_clearance_by_agent[transit.agent_id][
                        "static_clearance_contract_violation"
                    ]
                ),
                minimum_clearance_m=actual_trace_clearance_m[index],
                connected_at_every_tick=fragment_connected_at_every_telemetry_tick[index],
                last_sensor_source_id=last_sensor_source_id[index],
                rolling=True,
            )
        samples.extend(sample_ledger)
        # The task has synchronous team decisions.  Sensor frames still arrive
        # at 10 Hz, but each UAV transmits one source-bound map delta only after
        # completing its decision fragment.  This avoids treating every ray as
        # an independent network packet while preserving real range/LOS/relay
        # admission for the next public belief.
        fusion_agent_id = self.agent_order[0]
        segment_delta_senders = tuple(
            agent_id for agent_id in self.agent_order if source_observation_ids_by_agent[agent_id]
        )
        for sender_id in segment_delta_senders:
            message_queue.publish(
                RelayMessage(
                    message_id=f"map-segment-{manifest.manifest_hash[:12]}-{sender_id}",
                    sender_id=sender_id,
                    source_timestamp_s=final_timestamp_s,
                    payload_digest=canonical_sha256(
                        {
                            "sender_id": sender_id,
                            "source_observation_ids": source_observation_ids_by_agent[sender_id],
                        }
                    ),
                    time_to_live_s=self.communication_message_ttl_s,
                )
            )
        boundary_delivery_timestamp_s = (
            final_timestamp_s
            + self.communication_base_latency_s
            + self.communication_per_hop_latency_s * max(1, len(self.agent_order) - 1)
        )
        message_queue.advance(
            timestamp_s=boundary_delivery_timestamp_s,
            graph=final_relay_graph,
        )
        message_queue.finalize_episode(timestamp_s=boundary_delivery_timestamp_s)
        public_map_sender_ids = {
            fusion_agent_id for agent_id in segment_delta_senders if agent_id == fusion_agent_id
        }
        public_map_sender_ids.update(
            outcome.sender_id
            for outcome in message_queue.outcomes
            if outcome.status == "DELIVERED" and outcome.receiver_id == fusion_agent_id
        )

        self.last_execution_samples = tuple(samples)
        roles_by_agent = {
            transit.agent_id: str(
                dict(transit.type_signature.public_features).get("assignment_role", "explore")
            )
            for transit, _ in routes
        }
        team_trajectory_diversity = audit_translation_invariant_team_trajectories(
            {transit.agent_id: transit_traces[index] for index, (transit, _) in enumerate(routes)},
            roles_by_agent=roles_by_agent,
            scope="realised_physx",
        )
        self.engineering_diagnostics = {
            "token_authorization_duration_s": token.duration,
            "execution_deadline_s": execution_horizon_s,
            "execution_elapsed_physics_s": final_timestamp_s,
            "calibration_only_timeout_probe": execution_horizon_s < token.duration,
            "action_completion_mode": (
                "event_driven_all_routes_completed_plus_minimum_dwell"
                if self.event_driven_action_completion
                else "legacy_planned_fragment_boundary"
            ),
            "contact_hard_fail_n": CONTACT_HARD_FAIL_N,
            "controller_tracking": {
                **_controller_tracking_profile(
                    self.controller_id,
                    physics_dt_s=float(self.sim.cfg.dt),
                ),
                "claim_limit": (
                    "Engineering executor setting. Any changed controller requires fresh "
                    "post-change transit calibration before a formal P07 budget is frozen."
                ),
            },
            "physics_visualization_trace": {
                "schema_version": "hm3d-physx-visualization-trace-v1",
                "purpose": "engineering_visual_audit_only",
                "sample_hz": self.visualization_trace_sample_hz,
                "sample_count": len(visualization_trace_samples),
                "samples": visualization_trace_samples,
                "claim_limit": (
                    "Post-step realised root telemetry for rendering and human inspection only; "
                    "it is not exposed to selection, control, sensing, safety, rewards, "
                    "training, QD, or OGFR."
                ),
            },
            "team_trajectory_diversity": team_trajectory_diversity.to_dict(),
            "inter_agent_separation": {
                "minimum_root_distance_m": minimum_inter_agent_distance_m,
                "required_root_distance_m": CF2X_MIN_INTER_AGENT_SEPARATION_M,
                "violation": first_inter_agent_separation_violation is not None,
                "first_violation": first_inter_agent_separation_violation,
                "method": "all_physx_root_pose_pairs_at_physics_cadence_v1",
            },
            "static_trace_clearance": static_trace_clearance,
            "communication": {
                "model": "range_los_undirected_relay_graph_v1",
                "maximum_range_m": self.communication_max_range_m,
                "measurement_count": relay_measurement_count,
                "fully_relay_connected_count": relay_fully_connected_count,
                "fully_relay_connected_fraction": (
                    relay_fully_connected_count / relay_measurement_count
                    if relay_measurement_count
                    else 0.0
                ),
                "telemetry_update_hz": self.communication_update_hz,
                "telemetry_sample_interval_s": communication_interval_s,
                "telemetry_sampling_claim": (
                    "contract-rate range/LOS snapshots; no unmeasured continuous-link claim"
                ),
                "relay_telemetry_sample_count": relay_measurement_count,
                "relay_connected_telemetry_sample_count": relay_fully_connected_count,
                "relay_connected_telemetry_sample_fraction": (
                    relay_fully_connected_count / relay_measurement_count
                    if relay_measurement_count
                    else 0.0
                ),
                "longest_sampled_disconnected_duration_s": longest_disconnected_duration_s,
                "partition_event_count": partition_event_count,
                "reconnection_count": reconnection_count,
                "mean_direct_link_count": (
                    relay_direct_link_count_sum / relay_measurement_count
                    if relay_measurement_count
                    else 0.0
                ),
                "maximum_component_count": relay_component_count_max,
                "maximum_relay_hops": relay_maximum_hops_max,
                "final_graph": final_relay_graph.to_dict(),
                "claim_limit": (
                    "Decision-boundary aggregate public-map delivery only. No RF propagation, "
                    "bandwidth, or connectivity between telemetry samples is represented."
                ),
            },
            "sparse_range_outcomes": {
                "profile_id": "sparse-range-3d-vfov90",
                "update_hz": self.sparse_range_update_hz,
                "source_observation_frame_count": len(public_range_frames),
                "frames_by_agent": dict(zip(self.agent_order, sensor_frames_by_agent, strict=True)),
                "frames_by_phase": dict(range_frames_by_phase),
                "outcome_hash": canonical_sha256([row.to_dict() for row in public_range_frames]),
                "ray_outcome_count": len(public_range_outcomes),
                "ray_outcome_hash": canonical_sha256(
                    [row.to_dict() for row in public_range_outcomes]
                ),
            },
            "message_delivery": {
                "model": "range-los-relay-decision-boundary-delta-v2",
                "aggregation": "one_delta_per_sender_per_decision",
                "fusion_agent_id": fusion_agent_id,
                "public_map_sender_ids": sorted(public_map_sender_ids),
                "public_map_delta_count": len(segment_delta_senders),
                "base_latency_s": self.communication_base_latency_s,
                "per_hop_latency_s": self.communication_per_hop_latency_s,
                "loss_probability": self.communication_loss_probability,
                "outcome_counts": {
                    status: sum(row.status == status for row in message_queue.outcomes)
                    for status in ("DELIVERED", "DROPPED", "EXPIRED")
                },
                "pending_recipient_count_before_close": message_queue.pending_recipient_count,
            },
            "agents": [
                {
                    "agent_id": self.agent_order[index],
                    "command_path_m": routes[index][0].path,
                    "initial_planned_position_m": transit_traces[index][0],
                    "first_simulated_position_m": (
                        transit_traces[index][1] if len(transit_traces[index]) > 1 else None
                    ),
                    "last_transit_position_m": transit_traces[index][-1],
                    "first_collision_step": first_collision_step[index],
                    "first_collision_position_m": first_collision_position[index],
                    "first_collision_commanded_waypoint_m": first_collision_waypoint[index],
                    "maximum_contact_force_n": maximum_contact_force_n[index],
                    "maximum_linear_speed_mps": maximum_linear_speed_mps[index],
                    "maximum_linear_acceleration_mps2": (maximum_linear_acceleration_mps2[index]),
                    "controller_tracking_telemetry_hz": CONTROLLER_TRACKING_TELEMETRY_HZ,
                    "controller_tracking_samples": controller_tracking_samples[index],
                    "transit_collision": transit_contact[index],
                    "transit_out_of_bounds": transit_oob[index],
                    "inter_agent_separation_violation": inter_agent_separation_violation[index],
                    "observation_collision": observation_contact[index],
                    "observation_out_of_bounds": observation_oob[index],
                    "minimum_static_mesh_clearance_m": actual_trace_clearance_m[index],
                    "minimum_clearance_position_m": static_trace_clearance_by_agent[
                        self.agent_order[index]
                    ]["minimum_clearance_position_m"],
                    "static_clearance_contract_required_m": FLIGHT_CLEARANCE_M,
                    "static_clearance_contract_violation": bool(
                        static_trace_clearance_by_agent[self.agent_order[index]][
                            "static_clearance_contract_violation"
                        ]
                    ),
                    "waypoint_settle_speed_mps": WAYPOINT_SETTLE_SPEED_MPS,
                    "waypoint_settle_position_tolerance_m": (
                        WAYPOINT_SETTLE_POSITION_TOLERANCE_M
                    ),
                    "waypoint_transitions": waypoint_transitions[index],
                    "transit_completed": transit_end[index] is not None,
                    "transit_completed_at_s": transit_end[index],
                    "transit_attempted": True,
                    "traffic_reservation": {
                        "enforced": traffic_predecessor_index[index] is not None,
                        "planned_delay_s": traffic_reservation_delay_s[index],
                        "predecessor_agent_id": (
                            None
                            if traffic_predecessor_index[index] is None
                            else self.agent_order[traffic_predecessor_index[index]]
                        ),
                        "released_at_s": transit_release_s[index],
                        "wait_physics_step_count": reservation_wait_steps[index],
                        "maximum_prestart_displacement_m": (
                            reservation_max_prestart_displacement_m[index]
                        ),
                        "release_rule": (
                            "planned_start_and_predecessor_measured_settled_completion"
                            if traffic_predecessor_index[index] is not None
                            else "immediate"
                        ),
                    },
                    "transit_attempt_actual_end_s": transit_end[index] or execution_horizon_s,
                    "transit_execution_deadline_s": execution_horizon_s,
                    "transit_failure_reason": (
                        None
                        if transit_end[index] is not None
                        else "collision"
                        if transit_contact[index]
                        else "out_of_bounds"
                        if transit_oob[index]
                        else "transit_timeout"
                    ),
                    "observation_started_at_s": observation_start[index],
                    "observation_completed_at_s": observation_end[index],
                    "realized_transit_path_length_m": _path_length_m(tuple(transit_traces[index])),
                    "next_unreached_waypoint_index": transit_waypoint_index[index],
                }
                for index in range(len(routes))
            ],
        }
        expected_recipient_outcomes = len(segment_delta_senders) * (len(self.agent_order) - 1)
        resolved_recipient_outcomes = len(message_queue.outcomes)
        if resolved_recipient_outcomes != expected_recipient_outcomes:
            raise RuntimeError(
                "relay message denominator mismatch: "
                f"expected={expected_recipient_outcomes}, resolved={resolved_recipient_outcomes}"
            )
        self.engineering_diagnostics["message_delivery"]["outcome_counts_after_close"] = {
            status: sum(row.status == status for row in message_queue.outcomes)
            for status in ("DELIVERED", "DROPPED", "EXPIRED")
        }
        self.engineering_diagnostics["message_delivery"]["expected_recipient_outcomes"] = (
            expected_recipient_outcomes
        )
        self.engineering_diagnostics["message_delivery"]["resolved_recipient_outcomes"] = (
            resolved_recipient_outcomes
        )
        delivered_ages_s = [
            row.delivery.age_seconds
            for row in message_queue.outcomes
            if row.status == "DELIVERED" and row.delivery is not None
        ]
        self.engineering_diagnostics["message_delivery"]["maximum_delivery_age_s"] = max(
            delivered_ages_s, default=0.0
        )
        self.public_range_frames = tuple(public_range_frames)
        self.public_range_outcomes = tuple(public_range_outcomes)
        self.public_map_sender_ids = tuple(sorted(public_map_sender_ids))
        self.final_root_positions_m = tuple(
            tuple(float(value) for value in row)
            for row in self.robot.data.root_pos_w.detach().cpu().tolist()
        )
        self.final_root_linear_speeds_mps = tuple(
            float(value)
            for value in torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        return tuple(samples)


def _diagnostic_point(raw: Any) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    try:
        point = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        return None
    return point


def _diagnostic_prim_path(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw)
    return value if value and value != "None" else None


def _raycast_hit_class(
    agent_id: str,
    rigid_body_path: str | None,
    collider_path: str | None,
) -> str:
    paths = tuple(path for path in (rigid_body_path, collider_path) if path is not None)
    if any("/HM3DCollision" in path for path in paths):
        return "static_hm3d"
    suffix = agent_id.removeprefix("uav")
    own_prefix = f"/World/P07Agents/Env_{suffix}/Robot" if suffix.isdigit() else None
    if own_prefix is not None and any(path.startswith(own_prefix) for path in paths):
        return "self_cf2x"
    if any("/P07Agents/" in path for path in paths):
        return "other_cf2x"
    return "unknown_scene_prim"


def _raycast_guard_diagnostic(
    *,
    agent_id: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    requested_distance_m: float,
    raycast_distance_m: float,
    hit: dict[str, Any],
) -> dict[str, object]:
    rigid_body_path = _diagnostic_prim_path(hit.get("rigidBody"))
    collider_path = _diagnostic_prim_path(hit.get("collider"))
    hit_position = _diagnostic_point(hit.get("position"))
    raw_hit_distance = hit.get("distance")
    try:
        hit_distance_m = float(raw_hit_distance)
    except (TypeError, ValueError):
        hit_distance_m = None
    if hit_distance_m is not None and not math.isfinite(hit_distance_m):
        hit_distance_m = None
    diagnostic: dict[str, object] = {
        "schema_version": "hm3d-physx-route-guard-hit-v1",
        "event_type": "raycast_hit",
        "agent_id": agent_id,
        "segment_start_m": start,
        "segment_end_m": end,
        "requested_distance_m": requested_distance_m,
        "raycast_distance_m": raycast_distance_m,
        "hit_class": _raycast_hit_class(agent_id, rigid_body_path, collider_path),
        "hit_prim_path": collider_path or rigid_body_path,
        "hit_rigid_body_path": rigid_body_path,
        "hit_collider_path": collider_path,
        "hit_distance_m": hit_distance_m,
        "hit_position_m": hit_position,
        "ignored_dynamic_hit_count": int(hit.get("ignored_dynamic_hit_count", 0)),
        "ignored_dynamic_hits": list(hit.get("ignored_dynamic_hits", ())),
        "required_continuous_clearance_m": PLANNED_CONTINUOUS_CLEARANCE_M,
    }
    return diagnostic


def _clearance_guard_diagnostic(
    *,
    clearance_oracle: Any,
    agent_id: str,
    stage: str,
    points: tuple[tuple[float, float, float], ...],
    required_clearance_m: float,
) -> dict[str, object]:
    """Describe one rejected clearance check without changing its authority.

    Candidate selection still receives only the guard's legal/infeasible
    result.  The exact mesh distances below are evaluator-side outcome
    diagnostics used to distinguish an invalid reset from a bad observation
    endpoint or an unsafe public-route interior.
    """

    exact_distance_query = getattr(clearance_oracle, "exact_static_distances_m", None)
    distances: tuple[float, ...] = ()
    if callable(exact_distance_query):
        raw_distances = exact_distance_query(points)
        distances = tuple(float(value) for value in raw_distances)
        if len(distances) != len(points) or not all(math.isfinite(value) for value in distances):
            raise RuntimeError("clearance diagnostic returned invalid exact mesh distances")
    minimum_index = min(range(len(distances)), key=distances.__getitem__) if distances else None
    return {
        "schema_version": "hm3d-physx-route-guard-clearance-v1",
        "event_type": "static_clearance_rejection",
        "agent_id": agent_id,
        "stage": stage,
        "sample_count": len(points),
        "required_clearance_m": required_clearance_m,
        "exact_static_mesh_distance_available": bool(distances),
        "minimum_static_mesh_clearance_m": (
            None if minimum_index is None else distances[minimum_index]
        ),
        "minimum_clearance_position_m": (
            None if minimum_index is None else points[minimum_index]
        ),
    }


def _line_guard(
    scene_query: Any,
    clearance_oracle: _EvaluatorStaticClearance,
    agent_id: str,
    path_m: tuple[tuple[float, float, float], ...],
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
) -> GuardedPath:
    start, end = path_m[0], path_m[-1]
    delta = tuple(end[index] - start[index] for index in range(3))
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= 1.0e-6:
        # A hold begins at measured post-execution state. Normal tracking error
        # may place it inside the planning reserve while it still satisfies the
        # frozen physical flight-clearance contract. Requiring the larger
        # new-command margin here can remove the safest recovery action.
        if not all(clearance_oracle.admits_many((start,), FLIGHT_CLEARANCE_M)):
            if diagnostic_sink is not None:
                diagnostic_sink(
                    _clearance_guard_diagnostic(
                        clearance_oracle=clearance_oracle,
                        agent_id=agent_id,
                        stage="stationary_hold_start",
                        points=(start,),
                        required_clearance_m=FLIGHT_CLEARANCE_M,
                    )
                )
            return GuardedPath(
                legal=False,
                path_m=path_m,
                reason="insufficient_continuous_collision_clearance",
            )
        return GuardedPath(legal=True, path_m=path_m, reason="stationary_hold")
    raycast_distance_m = distance - 0.05
    hit = _first_static_scene_hit(
        scene_query,
        start,
        end,
        endpoint_margin_m=0.05,
    )
    if hit is not None:
        if diagnostic_sink is not None:
            diagnostic_sink(
                _raycast_guard_diagnostic(
                    agent_id=agent_id,
                    start=start,
                    end=end,
                    requested_distance_m=distance,
                    raycast_distance_m=raycast_distance_m,
                    hit=hit,
                )
            )
        return GuardedPath(legal=False, path_m=path_m, reason="segment_blocked")
    sample_count = max(1, math.ceil(distance / ROUTE_CLEARANCE_SAMPLE_STEP_M))
    internal_sample_clearance_m = REQUIRED_ROUTE_SAMPLE_CLEARANCE_M
    # The first point is measured state, not a newly commanded waypoint. It
    # must remain physically safe at FLIGHT_CLEARANCE_M. The new endpoint and
    # internal samples retain the stricter planning/tracking reserve.
    interior_samples = tuple(
        tuple(start[axis] + sample_index / sample_count * delta[axis] for axis in range(3))
        for sample_index in range(1, sample_count)
    )
    batch_admissions = getattr(
        clearance_oracle, "admits_many_with_required_clearances", None
    )
    if callable(batch_admissions):
        admissions = batch_admissions(
            (start, end, *interior_samples),
            (
                FLIGHT_CLEARANCE_M,
                REQUIRED_TERMINAL_CLEARANCE_M,
                *((internal_sample_clearance_m,) * len(interior_samples)),
            ),
        )
    else:
        admissions = (
            all(
                clearance_oracle.admits_many((start,), FLIGHT_CLEARANCE_M)
            ),
            all(
                clearance_oracle.admits_many(
                    (end,), REQUIRED_TERMINAL_CLEARANCE_M
                )
            ),
            *tuple(
                all(
                    clearance_oracle.admits_many(
                        (point,), internal_sample_clearance_m
                    )
                )
                for point in interior_samples
            ),
        )
    if not admissions[0]:
        if diagnostic_sink is not None:
            diagnostic_sink(
                _clearance_guard_diagnostic(
                    clearance_oracle=clearance_oracle,
                    agent_id=agent_id,
                    stage="start",
                    points=(start,),
                    required_clearance_m=FLIGHT_CLEARANCE_M,
                )
            )
        return GuardedPath(
            legal=False,
            path_m=path_m,
            reason="insufficient_continuous_collision_clearance",
        )
    if not admissions[1]:
        if diagnostic_sink is not None:
            diagnostic_sink(
                _clearance_guard_diagnostic(
                    clearance_oracle=clearance_oracle,
                    agent_id=agent_id,
                    stage="endpoint",
                    points=(end,),
                    required_clearance_m=REQUIRED_TERMINAL_CLEARANCE_M,
                )
            )
        return GuardedPath(
            legal=False,
            path_m=path_m,
            reason="insufficient_continuous_collision_clearance",
        )
    if not all(admissions[2:]):
        if diagnostic_sink is not None:
            diagnostic_sink(
                _clearance_guard_diagnostic(
                    clearance_oracle=clearance_oracle,
                    agent_id=agent_id,
                    stage="interior",
                    points=interior_samples,
                    required_clearance_m=internal_sample_clearance_m,
                )
            )
        return GuardedPath(
            legal=False,
            path_m=path_m,
            reason="insufficient_continuous_collision_clearance",
        )
    return GuardedPath(legal=True, path_m=path_m)


def _line_guard_with_segment_cache(
    scene_query: Any,
    clearance_oracle: _EvaluatorStaticClearance,
    agent_id: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    segment_cache: (
        dict[
            tuple[str, tuple[float, float, float], tuple[float, float, float]],
            GuardedPath,
        ]
        | None
    ) = None,
) -> GuardedPath:
    if segment_cache is not None:
        segment_key = (agent_id, tuple(start), tuple(end))
        cached = segment_cache.get(segment_key)
        if cached is not None:
            return cached
    guarded = _line_guard(
        scene_query,
        clearance_oracle,
        agent_id,
        (start, end),
        diagnostic_sink,
    )
    if segment_cache is not None:
        segment_cache[(agent_id, tuple(start), tuple(end))] = guarded
    return guarded


def _routed_guard(
    scene_query: Any,
    clearance_oracle: _EvaluatorStaticClearance,
    public_waypoints: tuple[tuple[float, float, float], ...],
    agent_id: str,
    path_m: tuple[tuple[float, float, float], ...],
    bounds_min: tuple[float, float, float] | None = None,
    bounds_max: tuple[float, float, float] | None = None,
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    *,
    allow_public_reroute: bool = True,
    segment_cache: (
        dict[
            tuple[str, tuple[float, float, float], tuple[float, float, float]],
            GuardedPath,
        ]
        | None
    ) = None,
) -> GuardedPath:
    """Guard every supplied path segment, then a bounded public-receiver polyline.

    HM3D receiver positions can lie in separate rooms.  A direct segment is
    not a valid high-level flight command in that case, but a short polyline
    through other public receivers can be.  The route is still accepted only when every
    segment passes the same runtime PhysX ray guard.  The bounded search is a
    smoke-test convenience, not a policy-visible planner.
    """

    # ``path_m`` can already be a public-free-space polyline.  Guarding only
    # its first and last point would certify a direct chord while handing the
    # executor unchecked intermediate waypoints.  Those waypoints are real
    # rest-to-rest commands, so every adjacent pair must satisfy the exact
    # same endpoint and swept-clearance contract before the full path is
    # exposed to a candidate selector.
    supplied_polyline_legal = True
    failed_reason = ""
    last_segment: GuardedPath | None = None

    def line_guard(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> GuardedPath:
        return _line_guard_with_segment_cache(
            scene_query,
            clearance_oracle,
            agent_id,
            start,
            end,
            diagnostic_sink,
            segment_cache,
        )

    for segment_start, segment_end in zip(path_m[:-1], path_m[1:], strict=True):
        segment = line_guard(segment_start, segment_end)
        last_segment = segment
        if not segment.legal:
            supplied_polyline_legal = False
            failed_reason = segment.reason or "segment_blocked"
            break
    if supplied_polyline_legal:
        if len(path_m) > 2:
            path_m = _shortcut_guarded_polyline(
                scene_query,
                clearance_oracle,
                agent_id,
                path_m,
                diagnostic_sink,
                segment_cache=segment_cache,
            )
        # Preserve a one-leg guard's useful audit reason, notably
        # ``stationary_hold`` for a measured safe recovery pose.  The
        # admitted path must remain the guarded polyline: substituting
        # ``last_segment`` here replaces a whole shortened route with only
        # its final leg, which would start the command far from the actual
        # vehicle pose.  Keep the polyline and inherit the reason only.
        if len(path_m) == 2 and last_segment is not None:
            admitted_path = GuardedPath(
                legal=True,
                path_m=path_m,
                rewritten=last_segment.rewritten,
                reason=last_segment.reason,
            )
        else:
            admitted_path = GuardedPath(legal=True, path_m=path_m)
        return _admit_trackable_path(
            admitted_path, bounds_min, bounds_max
        )
    if not allow_public_reroute:
        return GuardedPath(legal=False, path_m=path_m, reason=failed_reason)
    start, end = path_m[0], path_m[-1]
    candidates = tuple(point for point in public_waypoints if point != start and point != end)
    for midpoint in candidates:
        first = line_guard(start, midpoint)
        second = line_guard(midpoint, end)
        if first.legal and second.legal:
            return _admit_trackable_path(
                GuardedPath(
                    legal=True,
                    path_m=(start, midpoint, end),
                    rewritten=True,
                    reason="public_waypoint_route",
                ),
                bounds_min,
                bounds_max,
            )
    for first_midpoint, second_midpoint in itertools.permutations(candidates, 2):
        first = line_guard(start, first_midpoint)
        middle = line_guard(first_midpoint, second_midpoint)
        last = line_guard(second_midpoint, end)
        if first.legal and middle.legal and last.legal:
            return _admit_trackable_path(
                GuardedPath(
                    legal=True,
                    path_m=(start, first_midpoint, second_midpoint, end),
                    rewritten=True,
                    reason="public_waypoint_route",
                ),
                bounds_min,
                bounds_max,
            )
    if bounds_min is not None and bounds_max is not None:
        grid_route_reason = "public_flight_grid_route"
        grid_route = _grid_route(
            scene_query,
            clearance_oracle,
            agent_id,
            start,
            end,
            bounds_min,
            bounds_max,
            diagnostic_sink,
            segment_cache=segment_cache,
        )
        if grid_route is None:
            fine_result = _fine_clearance_grid_route(
                scene_query,
                clearance_oracle,
                agent_id,
                start,
                end,
                bounds_min,
                bounds_max,
                diagnostic_sink,
                requested_path_m=path_m,
                segment_cache=segment_cache,
            )
            if fine_result is not None:
                grid_route, _fine_audit = fine_result
                if grid_route is not None:
                    grid_route_reason = "public_flight_fine_grid_route"
        if grid_route is not None:
            return _admit_trackable_path(
                GuardedPath(
                    legal=True,
                    path_m=grid_route,
                    rewritten=True,
                    reason=grid_route_reason,
                ),
                bounds_min,
                bounds_max,
            )
    return GuardedPath(legal=False, path_m=path_m, reason="segment_blocked")


def _axis_samples(lower: float, upper: float, step: float) -> tuple[float, ...]:
    count = max(0, math.floor((upper - lower) / step))
    return tuple(lower + step * (index + 0.5) for index in range(count))


def _point_has_clearance(
    clearance_oracle: _EvaluatorStaticClearance,
    point: tuple[float, float, float],
    clearance_m: float = PLANNED_CONTINUOUS_CLEARANCE_M,
) -> bool:
    return clearance_oracle.admits(point, clearance_m)


def _path_length_m(path_m: tuple[tuple[float, float, float], ...]) -> float:
    """Return polyline length without attaching any optimistic speed claim."""

    return sum(math.dist(start, end) for start, end in zip(path_m[:-1], path_m[1:], strict=True))


def _shortcut_guarded_polyline(
    scene_query: Any,
    clearance_oracle: _EvaluatorStaticClearance,
    agent_id: str,
    path_m: tuple[tuple[float, float, float], ...],
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    segment_cache: (
        dict[
            tuple[str, tuple[float, float, float], tuple[float, float, float]],
            GuardedPath,
        ]
        | None
    ) = None,
) -> tuple[tuple[float, float, float], ...]:
    """Remove grid turns only when the existing full clearance guard approves.

    The grid search deliberately expands a six-connected 1 m lattice, which
    creates many right-angle turns even inside a clear corridor.  Those turns
    make physical execution much slower than the high-level distance model.
    This is a conservative simplifier: every replacement segment is submitted
    to the exact same ray plus swept-clearance guard used to admit the original
    route.  It cannot tunnel through geometry or relax the CF2X envelope.
    """

    if len(path_m) <= 2:
        return path_m
    compact = [path_m[0]]
    current_index = 0
    while current_index < len(path_m) - 1:
        next_index = current_index + 1
        for candidate_index in range(len(path_m) - 1, current_index + 1, -1):
            candidate = _line_guard_with_segment_cache(
                scene_query,
                clearance_oracle,
                agent_id,
                path_m[current_index],
                path_m[candidate_index],
                diagnostic_sink,
                segment_cache,
            )
            if candidate.legal:
                next_index = candidate_index
                break
        compact.append(path_m[next_index])
        current_index = next_index
    return tuple(compact)


def _densify_for_physics_tracking(
    path_m: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """Bound controller target separation without altering guarded geometry.

    A collision-free centreline does not prove that the current CF2X waypoint
    controller can track a long segment without cutting a corner.  The added
    points are collinear samples of already guarded segments; this neither
    forms a new shortcut nor relaxes the 0.30 m clearance condition.
    """

    if len(path_m) < 2:
        return path_m
    expanded = [path_m[0]]
    for start, end in zip(path_m[:-1], path_m[1:], strict=True):
        length = math.dist(start, end)
        segment_count = max(1, math.ceil(length / MAX_PHYSICS_WAYPOINT_SPAN_M))
        for segment_index in range(1, segment_count + 1):
            fraction = segment_index / segment_count
            expanded.append(
                tuple(start[axis] + fraction * (end[axis] - start[axis]) for axis in range(3))
            )
    return tuple(expanded)


def _with_trackable_waypoints(path: GuardedPath) -> GuardedPath:
    """Preserve a legal guard decision while making its path trackable."""

    if not path.legal:
        return path
    return GuardedPath(
        legal=True,
        path_m=_densify_for_physics_tracking(path.path_m),
        rewritten=path.rewritten,
        reason=path.reason,
    )


def _within_control_bounds(
    path_m: tuple[tuple[float, float, float], ...],
    bounds_min: tuple[float, float, float] | None,
    bounds_max: tuple[float, float, float] | None,
) -> bool:
    """Keep newly commanded waypoints away from a component's outer edge.

    The route guard protects against geometry, whereas the component bounds
    protect the flight-space admission.  A public view on the component edge
    left no room for position tolerance and momentum in the r17 smoke.  This
    is an admission constraint, not an execution-time clipping rule.

    The first path point is the measured post-execution state, not a waypoint
    selected by the current method.  Tracking error can place it a few
    millimetres outside the tightened command set while it remains inside the
    frozen flight-space bounds.  Rejecting that state also rejects the safest
    recovery action, a stationary hold.  Require the measured start only to
    remain in the physical flight-space bounds; apply the control margin to
    every newly commanded point.
    """

    if bounds_min is None or bounds_max is None:
        return True
    if not path_m:
        return False

    def point_within(point: tuple[float, float, float], margin_m: float) -> bool:
        return all(
            bounds_min[axis] + margin_m <= point[axis] <= bounds_max[axis] - margin_m
            for axis in range(3)
        )

    if not point_within(path_m[0], 0.0):
        return False
    if all(math.dist(path_m[0], point) <= 1.0e-9 for point in path_m[1:]):
        return True
    return all(point_within(point, FLIGHT_CONTROL_BOUNDARY_MARGIN_M) for point in path_m[1:])


def _admit_trackable_path(
    path: GuardedPath,
    bounds_min: tuple[float, float, float] | None,
    bounds_max: tuple[float, float, float] | None,
) -> GuardedPath:
    """Fail closed when an otherwise clear path lacks control-boundary room."""

    trackable = _with_trackable_waypoints(path)
    if trackable.legal and not _within_control_bounds(trackable.path_m, bounds_min, bounds_max):
        return GuardedPath(
            legal=False,
            path_m=trackable.path_m,
            rewritten=trackable.rewritten,
            reason="insufficient_control_boundary_margin",
        )
    return trackable


def _grid_route(
    scene_query: Any,
    clearance_oracle: _EvaluatorStaticClearance,
    agent_id: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    segment_cache: (
        dict[
            tuple[str, tuple[float, float, float], tuple[float, float, float]],
            GuardedPath,
        ]
        | None
    ) = None,
) -> tuple[tuple[float, float, float], ...] | None:
    """Find a short collision-checked route on a public coarse flight grid."""

    import heapq

    resolution = 1.0
    axes = tuple(
        _axis_samples(bounds_min[index], bounds_max[index], resolution) for index in range(3)
    )
    if any(not axis for axis in axes):
        return None
    indexed_points = tuple(
        ((ix, iy, iz), (x, y, z))
        for ix, x in enumerate(axes[0])
        for iy, y in enumerate(axes[1])
        for iz, z in enumerate(axes[2])
    )
    point_admission = clearance_oracle.admits_many(
        tuple(point for _, point in indexed_points), PLANNED_CONTINUOUS_CLEARANCE_M
    )
    grid_points = {
        key: point
        for (key, point), admitted in zip(indexed_points, point_admission, strict=True)
        if admitted
    }
    if not grid_points:
        return None

    def visible(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
        return _line_guard_with_segment_cache(
            scene_query,
            clearance_oracle,
            agent_id,
            left,
            right,
            diagnostic_sink,
            segment_cache,
        ).legal

    start_candidates = sorted(
        (
            (math.dist(start, point), key, point)
            for key, point in grid_points.items()
            if math.dist(start, point) <= 2.0 and visible(start, point)
        ),
        key=lambda row: row[0],
    )
    end_candidates = {
        key
        for _, key, point in sorted(
            (
                (math.dist(end, point), key, point)
                for key, point in grid_points.items()
                if math.dist(end, point) <= 2.0 and visible(point, end)
            ),
            key=lambda row: row[0],
        )
    }
    if not start_candidates or not end_candidates:
        return None

    frontier: list[tuple[float, tuple[int, int, int]]] = []
    predecessor: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for _, key, _ in start_candidates[:32]:
        predecessor[key] = None
        heapq.heappush(frontier, (0.0, key))
    goal: tuple[int, int, int] | None = None
    while frontier and len(predecessor) <= 12000:
        _, current = heapq.heappop(frontier)
        if current in end_candidates:
            goal = current
            break
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(current)
                neighbor[axis] += direction
                neighbor_key = tuple(neighbor)
                if neighbor_key not in grid_points or neighbor_key in predecessor:
                    continue
                if not visible(grid_points[current], grid_points[neighbor_key]):
                    continue
                predecessor[neighbor_key] = current
                priority = math.dist(grid_points[neighbor_key], end)
                heapq.heappush(frontier, (priority, neighbor_key))
    if goal is None:
        return None
    keys = [goal]
    while predecessor[keys[-1]] is not None:
        keys.append(predecessor[keys[-1]])  # type: ignore[arg-type]
    keys.reverse()
    dense_route = (start, *(grid_points[key] for key in keys), end)
    return _shortcut_guarded_polyline(
        scene_query,
        clearance_oracle,
        agent_id,
        dense_route,
        diagnostic_sink,
        segment_cache,
    )


def _fine_clearance_grid_route(
    scene_query: Any,
    clearance_oracle: _EvaluatorStaticClearance,
    agent_id: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    *,
    requested_path_m: Sequence[tuple[float, float, float]] | None = None,
    resolution_m: float = FINE_CLEARANCE_ROUTE_RESOLUTION_M,
    segment_cache: (
        dict[
            tuple[str, tuple[float, float, float], tuple[float, float, float]],
            GuardedPath,
        ]
        | None
    ) = None,
) -> tuple[tuple[tuple[float, float, float], ...], dict[str, object]] | None:
    """Find a local collision-checked route on a fine exact-clearance lattice.

    The coarse 1 m grid frequently misses an indoor corridor centreline.
    This fallback searches only a small box around the requested public
    route, prefilters vertices with the same evaluator clearance oracle,
    and rechecks every lattice edge with the shared _line_guard.  The
    route is still submitted to _admit_trackable_path by the caller, so
    control-boundary and physics-waypoint constraints are not bypassed.
    """
    import heapq

    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("fine grid resolution must be finite and positive")
    anchor_points = tuple(tuple(point) for point in requested_path_m) if requested_path_m else (start, end)
    local_min = tuple(
        min(point[axis] for point in anchor_points) - FINE_CLEARANCE_ROUTE_LOCAL_MARGIN_M
        for axis in range(3)
    )
    local_max = tuple(
        max(point[axis] for point in anchor_points) + FINE_CLEARANCE_ROUTE_LOCAL_MARGIN_M
        for axis in range(3)
    )
    if bounds_min is not None:
        local_min = tuple(max(local_min[axis], bounds_min[axis]) for axis in range(3))
    if bounds_max is not None:
        local_max = tuple(min(local_max[axis], bounds_max[axis]) for axis in range(3))
    axes = tuple(_axis_samples(local_min[axis], local_max[axis], resolution_m) for axis in range(3))
    if any(not axis for axis in axes):
        return None, {"reason": "empty_local_grid"}
    points = tuple((x, y, z) for x in axes[0] for y in axes[1] for z in axes[2])
    if len(points) > FINE_CLEARANCE_ROUTE_MAX_GRID_POINTS:
        return None, {"reason": "local_grid_exceeds_budget", "grid_point_count": len(points)}
    point_admission = clearance_oracle.admits_many(points, PLANNED_CONTINUOUS_CLEARANCE_M)
    grid_points = {
        point: point
        for point, admitted in zip(points, point_admission, strict=True)
        if admitted
    }
    if not grid_points:
        return None, {"reason": "no_admitted_grid_points"}

    def visible(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
        return _line_guard_with_segment_cache(
            scene_query,
            clearance_oracle,
            agent_id,
            left,
            right,
            diagnostic_sink,
            segment_cache,
        ).legal

    start_candidates = sorted(
        (
            (math.dist(start, point), point)
            for point in grid_points
            if math.dist(start, point) <= FINE_CLEARANCE_ROUTE_CONNECTOR_RADIUS_M
            and visible(start, point)
        ),
        key=lambda row: row[0],
    )[:FINE_CLEARANCE_ROUTE_START_CANDIDATE_LIMIT]
    end_candidates = {
        point
        for point in grid_points
        if math.dist(end, point) <= FINE_CLEARANCE_ROUTE_CONNECTOR_RADIUS_M
    }
    if not start_candidates or not end_candidates:
        return None, {
            "reason": "no_connector_candidates",
            "start_candidate_count": len(start_candidates),
            "end_candidate_count": len(end_candidates),
        }

    g_score: dict[tuple[float, float, float], float] = {}
    predecessor: dict[tuple[float, float, float], tuple[float, float, float] | None] = {}
    frontier: list[tuple[float, tuple[float, float, float]]] = []
    for _, point in start_candidates:
        g_score[point] = math.dist(start, point)
        predecessor[point] = None
        heapq.heappush(frontier, (g_score[point] + math.dist(point, end), point))
    goal: tuple[float, float, float] | None = None
    expanded_nodes = 0
    edge_guards = 0
    while frontier and expanded_nodes <= FINE_CLEARANCE_ROUTE_MAX_EXPANDED_NODES:
        _, current = heapq.heappop(frontier)
        if current in end_candidates:
            goal = current
            break
        expanded_nodes += 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (
                        current[0] + resolution_m * dx,
                        current[1] + resolution_m * dy,
                        current[2] + resolution_m * dz,
                    )
                    if neighbor not in grid_points or neighbor in predecessor:
                        continue
                    edge_guards += 1
                    if edge_guards > FINE_CLEARANCE_ROUTE_MAX_EDGE_GUARDS:
                        return None, {
                            "reason": "edge_guard_budget_exceeded",
                            "expanded_node_count": expanded_nodes,
                        }
                    if not visible(current, neighbor):
                        continue
                    candidate_g = g_score[current] + math.dist(current, neighbor)
                    if candidate_g + 1.0e-12 < g_score.get(neighbor, float("inf")):
                        g_score[neighbor] = candidate_g
                        predecessor[neighbor] = current
                        heapq.heappush(
                            frontier,
                            (candidate_g + math.dist(neighbor, end), neighbor),
                        )
    if goal is None:
        return None, {
            "reason": "no_fine_clearance_route",
            "expanded_node_count": expanded_nodes,
            "edge_guard_count": edge_guards,
        }
    keys = [goal]
    while predecessor[keys[-1]] is not None:
        parent = predecessor[keys[-1]]
        assert parent is not None
        keys.append(parent)
    keys.reverse()
    terminal = goal
    if visible(goal, end):
        terminal = end
    dense_route = (start, *keys)
    if terminal != goal:
        dense_route = (*dense_route, terminal)
    compact = _shortcut_guarded_polyline(
        scene_query,
        clearance_oracle,
        agent_id,
        dense_route,
        diagnostic_sink,
        segment_cache,
    )
    return compact, {
        "reason": "admitted",
        "resolution_m": resolution_m,
        "local_bounds_min_m": local_min,
        "local_bounds_max_m": local_max,
        "grid_point_count": len(points),
        "admitted_grid_point_count": len(grid_points),
        "start_candidate_count": len(start_candidates),
        "end_candidate_count": len(end_candidates),
        "expanded_node_count": expanded_nodes,
        "edge_guard_count": edge_guards,
        "route_length_m": _path_length_m(compact),
        "terminal_point_m": terminal,
        "terminal_offset_m": math.dist(terminal, end),
        "terminal_reached": terminal == end,
    }


def _select_smoke_positions(
    route_guard: Callable[[str, tuple[tuple[float, float, float], ...]], GuardedPath],
    positions: tuple[tuple[float, float, float], ...],
    transit_timing_model: ConservativeTransitTimingModel,
    decision_duration_s: float,
    observe_dwell_s: float,
    candidate_limit: int,
    minimum_feasible_candidates: int,
    position_order_offset: int = 0,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    """Select public views with the requested guard- and budget-feasible headroom.

    The evaluator receiver audit gives valid public poses, not a route plan.  Using
    the first and last poses as waypoints made the old smoke fail closed on
    ordinary walls.  We search only the public pose set and ask the same
    runtime line guard and decision-time budget used by candidate construction
    to validate every start-to-frontier edge.  Physical execution requires one
    legal simultaneous assignment; callers that need selector-identifiability
    headroom may request more.  The candidate
    builder still retains other infeasible assignments in its failure
    denominator. A
    recorded enumeration offset is available only to collect distinct,
    public-route transit calibration probes; it is not a policy feature and
    must stay at its frozen default in comparable P07 baseline episodes. This
    does not add a planner or inspect targets; it merely prevents a
    deliberately impossible engineering smoke.
    """

    if len(positions) < FORMAL_FLEET_SIZE + candidate_limit:
        raise ValueError("not enough public view positions for the formal fleet")
    if candidate_limit < FORMAL_FLEET_SIZE:
        raise ValueError("candidate limit must cover the formal fleet")
    if minimum_feasible_candidates < 1 or minimum_feasible_candidates > candidate_limit:
        raise ValueError("invalid minimum feasible-candidate requirement")
    if decision_duration_s <= observe_dwell_s:
        raise ValueError("smoke route selector needs a positive transit-time budget")

    admission_audit: defaultdict[str, int] = defaultdict(int)

    def _admitted_leg(
        agent_id: str,
        start: tuple[float, float, float],
        frontier: tuple[float, float, float],
    ) -> GuardedPath | None:
        admission_audit["leg_attempts"] += 1
        # A guarded route cannot be shorter than its straight-line lower bound.
        # Rejecting this case before ray queries preserves the time budget and
        # avoids constructing a full flight grid for a physically impossible
        # smoke-test assignment.
        if (
            transit_timing_model.estimate_seconds((start, frontier)) + observe_dwell_s
            > decision_duration_s
        ):
            admission_audit["leg_lower_bound_budget_rejected"] += 1
            return None
        guarded = route_guard(agent_id, (start, frontier))
        travel_s = transit_timing_model.estimate_seconds(guarded.path_m)
        if not guarded.legal:
            admission_audit[f"leg_guard_rejected:{guarded.reason or 'unspecified'}"] += 1
            return None
        if travel_s + observe_dwell_s > decision_duration_s + 1.0e-9:
            admission_audit["leg_guarded_route_budget_rejected"] += 1
            return None
        admission_audit["leg_admitted"] += 1
        return guarded

    def _admitted_assignment(
        starts: tuple[tuple[float, float, float], ...],
        frontier_indices: tuple[int, ...],
    ) -> bool:
        admission_audit["joint_assignment_attempts"] += 1
        guarded_legs = tuple(
            _admitted_leg(
                f"uav{agent_index}",
                starts[agent_index],
                positions[frontier_indices[agent_index]],
            )
            for agent_index in range(FORMAL_FLEET_SIZE)
        )
        if any(leg is None for leg in guarded_legs):
            admission_audit["joint_assignment_leg_rejected"] += 1
            return False
        legal_legs = tuple(leg for leg in guarded_legs if leg is not None)
        synchronized = assess_synchronized_separation(
            tuple(
                TimedPolyline(
                    f"uav{agent_index}",
                    leg.path_m,
                    0.0,
                    transit_timing_model.estimate_seconds(leg.path_m),
                )
                for agent_index, leg in enumerate(legal_legs)
            ),
            minimum_separation_m=PLANNED_INTER_AGENT_SEPARATION_M,
        )
        if not synchronized.admitted:
            admission_audit["joint_assignment_separation_rejected"] += 1
            return False
        route_tube = assess_route_tube_separation(
            tuple(
                TimedPolyline(
                    f"uav{agent_index}",
                    leg.path_m,
                    0.0,
                    transit_timing_model.estimate_seconds(leg.path_m),
                )
                for agent_index, leg in enumerate(legal_legs)
            ),
            minimum_separation_m=CF2X_MIN_INTER_AGENT_SEPARATION_M,
        )
        if not route_tube.admitted:
            admission_audit["joint_assignment_route_tube_rejected"] += 1
            return False
        admission_audit["joint_assignment_admitted"] += 1
        return True

    indices = tuple(range(len(positions)))
    normalized_offset = position_order_offset % len(indices)
    indices = indices[normalized_offset:] + indices[:normalized_offset]
    for start_indices in itertools.combinations(indices, FORMAL_FLEET_SIZE):
        remaining = tuple(index for index in indices if index not in start_indices)
        for frontier_indices in itertools.combinations(remaining, FORMAL_FLEET_SIZE):
            starts = tuple(positions[index] for index in start_indices)
            for offset in range(FORMAL_FLEET_SIZE):
                selected_frontier_indices = tuple(
                    frontier_indices[(offset + agent_index) % FORMAL_FLEET_SIZE]
                    for agent_index in range(FORMAL_FLEET_SIZE)
                )
                if not _admitted_assignment(starts, selected_frontier_indices):
                    continue
                admission_audit["first_assignment_admitted"] += 1
                if minimum_feasible_candidates == 1:
                    ordered_frontiers = selected_frontier_indices + tuple(
                        index for index in remaining if index not in selected_frontier_indices
                    )
                    ordered_frontiers = ordered_frontiers[:candidate_limit]
                    if len(ordered_frontiers) != candidate_limit:
                        raise RuntimeError("candidate-frontier construction lost a public position")
                    return starts, tuple(positions[index] for index in ordered_frontiers)
                # Candidate 0 is the admitted assignment.  Candidate 1 in the
                # shared cyclic pool shifts every destination by one frontier.
                # When the pool has room, find a new final frontier; when it is
                # exactly fleet-sized, test its cyclic wraparound.  No private
                # evaluator state or learned policy participates in this search.
                if candidate_limit == FORMAL_FLEET_SIZE:
                    continuation_indices = (selected_frontier_indices[0],)
                else:
                    continuation_indices = tuple(
                        index for index in remaining if index not in selected_frontier_indices
                    )
                for continuation in continuation_indices:
                    admission_audit["continuation_assignment_attempts"] += 1
                    next_assignment = selected_frontier_indices[1:] + (continuation,)
                    if not _admitted_assignment(starts, next_assignment):
                        continue
                    admission_audit["continuation_assignment_admitted"] += 1
                    ordered_frontiers = selected_frontier_indices + (
                        () if candidate_limit == FORMAL_FLEET_SIZE else (continuation,)
                    )
                    ordered_frontiers += tuple(
                        index for index in remaining if index not in ordered_frontiers
                    )
                    ordered_frontiers = ordered_frontiers[:candidate_limit]
                    if len(ordered_frontiers) != candidate_limit:
                        raise RuntimeError("candidate-frontier construction lost a public position")
                    return starts, tuple(positions[index] for index in ordered_frontiers)
    raise CandidateHeadroomError(
        "no public view-position combination has the required P07 strategy headroom "
        f"(feasible>={minimum_feasible_candidates})",
        {
            "candidate_limit": candidate_limit,
            "fleet_size": FORMAL_FLEET_SIZE,
            "minimum_feasible_candidates": minimum_feasible_candidates,
            "public_position_count": len(positions),
            **admission_audit,
        },
    )
