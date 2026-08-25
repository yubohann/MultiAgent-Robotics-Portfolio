"""Freeze a conservative CF2X transit-time contract from immutable P07 outcomes.

The script accepts only real Isaac/PhysX smoke evidence and creates a new
calibration artifact without copying evaluator-private task fields, private
geometry, or complete execution traces.  It is a calibration
artifact, not a P07 baseline result.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.adapters.hm3d_baselines import ConservativeTransitTimingModel
from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.runtime.hm3d_cf2x_execution import (
    BITCRAZE_LEE_CONTROLLER_ID,
    BITCRAZE_MELLINGER_CONTROLLER_ID,
    CF2X_DEFAULT_CONTROLLER_ID,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _path_points(path: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("execution path must contain at least two points")
    points: list[tuple[float, float, float]] = []
    for point in path:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("execution path contains an invalid point")
        row = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in row):
            raise ValueError("execution path contains a non-finite point")
        points.append(row)
    return tuple(points)


def _path_length_m(path: Any) -> float:
    points = _path_points(path)
    return sum(math.dist(left, right) for left, right in zip(points[:-1], points[1:], strict=True))


def _route_geometry(path: Any) -> dict[str, object]:
    points = _path_points(path)
    horizontal_m = sum(
        math.dist(left[:2], right[:2]) for left, right in zip(points[:-1], points[1:], strict=True)
    )
    vertical_m = sum(
        abs(right[2] - left[2]) for left, right in zip(points[:-1], points[1:], strict=True)
    )
    route_classes: list[str] = []
    if horizontal_m > 0.05:
        route_classes.append("horizontal")
    if vertical_m > 0.05:
        route_classes.append("vertical")
    if len(points) > 2:
        route_classes.append("turn")
    if not route_classes:
        route_classes.append("stationary")
    return {
        "route_classes": route_classes,
        "horizontal_path_length_m": horizontal_m,
        "vertical_path_length_m": vertical_m,
        "waypoint_segment_count": len(points) - 1,
    }


def _controller_tracking_profile(
    execution: dict[str, Any], source: Path, record_id: str
) -> dict[str, object]:
    """Extract the control settings that determine outcome transit timing.

    This is deliberately a small, explicit contract rather than a hash of the
    whole smoke artifact: candidate pools, routes and task outcomes must vary
    across calibration samples, whereas an altered near-waypoint controller
    invalidates their common speed envelope.  Rejecting a missing profile also
    prevents historical, pre-change evidence from silently entering a new
    calibration.
    """

    tracking = execution.get("controller_tracking")
    if not isinstance(tracking, dict):
        raise ValueError(f"input has no controller-tracking profile: {source}#{record_id}")
    controller_id = tracking.get("controller_id")
    if not isinstance(controller_id, str) or not controller_id:
        raise ValueError(f"controller profile lacks controller_id: {source}#{record_id}")
    speed_profile = tracking.get("speed_profile")
    if speed_profile is None:
        # Historical proportional-controller evidence remains readable, but
        # its profile cannot mix with either trajectory-reference ABI.
        numeric_fields = (
            "horizontal_approach_speed_gain_mps_per_m",
            "waypoint_settle_speed_mps",
            "tracking_clearance_margin_m",
        )
    elif speed_profile == "braking-distance-limited-v1":
        numeric_fields = (
            "maximum_translational_speed_mps",
            "maximum_translational_acceleration_mps2",
            "velocity_error_gain_per_s",
            "waypoint_settle_speed_mps",
            "tracking_clearance_margin_m",
        )
    elif speed_profile in {
        "time-parameterized-trapezoid-v1",
        "time-parameterized-trapezoid-so3-v2",
        "time-parameterized-trapezoid-so3-overdamped-v3",
        "time-parameterized-trapezoid-so3-overdamped-v4",
        "time-parameterized-trapezoid-so3-guarded-v5",
        "time-parameterized-trapezoid-so3-guarded-v6",
        "time-parameterized-trapezoid-so3-guarded-v7",
        "time-parameterized-trapezoid-so3-guarded-v8",
    }:
        numeric_fields = (
            "maximum_reference_speed_mps",
            "maximum_reference_acceleration_mps2",
            "position_error_gain_per_s2",
            "velocity_error_gain_per_s",
            "waypoint_settle_speed_mps",
            "tracking_clearance_margin_m",
        )
        if controller_id == BITCRAZE_LEE_CONTROLLER_ID:
            numeric_fields = (
                *numeric_fields,
                "position_error_limit_m",
                "velocity_error_limit_mps",
                "maximum_feedback_acceleration_mps2",
                "effective_control_rate_hz",
                "official_control_rate_hz",
            )
        elif controller_id == BITCRAZE_MELLINGER_CONTROLLER_ID:
            numeric_fields = (
                "maximum_reference_speed_mps",
                "maximum_reference_acceleration_mps2",
                "waypoint_settle_speed_mps",
                "tracking_clearance_margin_m",
                "effective_control_rate_hz",
                "official_control_rate_hz",
            )
        else:
            numeric_fields = (*numeric_fields, "maximum_feedback_acceleration_mps2")
            if controller_id == CF2X_DEFAULT_CONTROLLER_ID:
                numeric_fields = (*numeric_fields, "effective_control_rate_hz")
        if speed_profile in {
            "time-parameterized-trapezoid-so3-guarded-v7",
            "time-parameterized-trapezoid-so3-guarded-v8",
        }:
            numeric_fields = (*numeric_fields, "waypoint_settle_position_tolerance_m")
        if speed_profile == "time-parameterized-trapezoid-so3-guarded-v8":
            numeric_fields = (*numeric_fields, "waypoint_pass_through_speed_mps")
        if speed_profile in {
            "time-parameterized-trapezoid-so3-v2",
            "time-parameterized-trapezoid-so3-overdamped-v3",
            "time-parameterized-trapezoid-so3-overdamped-v4",
            "time-parameterized-trapezoid-so3-guarded-v5",
            "time-parameterized-trapezoid-so3-guarded-v6",
            "time-parameterized-trapezoid-so3-guarded-v7",
            "time-parameterized-trapezoid-so3-guarded-v8",
        }:
            numeric_fields = (*numeric_fields, "maximum_tilt_rad")
    else:
        raise ValueError(
            f"unsupported controller speed profile {speed_profile!r}: {source}#{record_id}"
        )
    profile: dict[str, object] = {"controller_id": controller_id}
    if speed_profile is not None:
        profile["speed_profile"] = speed_profile
    if speed_profile in {
        "time-parameterized-trapezoid-so3-v2",
        "time-parameterized-trapezoid-so3-overdamped-v3",
        "time-parameterized-trapezoid-so3-overdamped-v4",
        "time-parameterized-trapezoid-so3-guarded-v5",
        "time-parameterized-trapezoid-so3-guarded-v6",
        "time-parameterized-trapezoid-so3-guarded-v7",
        "time-parameterized-trapezoid-so3-guarded-v8",
    }:
        attitude_control = tracking.get("attitude_control")
        supported_attitude_controls = {
            "force-yaw-so3-v1",
            "force-rate-limited-yaw-so3-v2",
        }
        if controller_id == BITCRAZE_LEE_CONTROLLER_ID:
            supported_attitude_controls.add("bitcraze-lee-se3-decision-core-guarded-isaac-v4")
        if controller_id == BITCRAZE_MELLINGER_CONTROLLER_ID:
            supported_attitude_controls.add("bitcraze-mellinger-legacy-mixer-adapted-v1")
        if attitude_control not in supported_attitude_controls:
            raise ValueError(
                f"unsupported attitude control {attitude_control!r}: {source}#{record_id}"
            )
        profile["attitude_control"] = attitude_control
        if attitude_control == "force-rate-limited-yaw-so3-v2":
            numeric_fields = (*numeric_fields, "maximum_yaw_rate_deg_s")
        elif attitude_control == "bitcraze-lee-se3-decision-core-guarded-isaac-v4":
            numeric_fields = (*numeric_fields, "maximum_yaw_rate_deg_s")
        elif attitude_control == "bitcraze-mellinger-legacy-mixer-adapted-v1":
            numeric_fields = (*numeric_fields, "maximum_yaw_rate_deg_s")
    for key in dict.fromkeys(numeric_fields):
        value = tracking.get(key)
        if isinstance(value, bool):
            raise ValueError(f"controller profile {key} must be numeric: {source}")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"controller profile {key} must be numeric: {source}") from error
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"controller profile {key} must be finite and positive: {source}")
        profile[key] = number
    if controller_id in {
        CF2X_DEFAULT_CONTROLLER_ID,
        BITCRAZE_LEE_CONTROLLER_ID,
        BITCRAZE_MELLINGER_CONTROLLER_ID,
    }:
        for key in (
            "rotor_xy_lever_arm_m",
            "yaw_torque_to_thrust_m",
            "initial_rotor_rps",
            "thrust_constant_n_per_rps2",
        ):
            value = tracking.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"controller profile {key} must be numeric: {source}") from error
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"controller profile {key} must be finite and positive: {source}")
            profile[key] = number
        for key in ("rotor_allocation_id", "actuator_initialization_id"):
            value = tracking.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"controller profile {key} must be a non-empty string: {source}")
            profile[key] = value
        rotor_order = tracking.get("rotor_order")
        if rotor_order != ["m1_prop", "m2_prop", "m3_prop", "m4_prop"]:
            raise ValueError(f"controller profile rotor_order is incompatible: {source}")
        profile["rotor_order"] = list(rotor_order)
        yaw_signs = tracking.get("rotor_yaw_reaction_signs")
        if yaw_signs != [-1, 1, -1, 1]:
            raise ValueError(
                f"controller profile rotor_yaw_reaction_signs is incompatible: {source}"
            )
        profile["rotor_yaw_reaction_signs"] = list(yaw_signs)
        for key in ("tau_inc_range_s", "tau_dec_range_s"):
            raw_range = tracking.get(key)
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError(f"controller profile {key} is incompatible: {source}")
            try:
                values = [float(value) for value in raw_range]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"controller profile {key} must contain numeric values: {source}"
                ) from error
            if (
                any(not math.isfinite(value) or value <= 0.0 for value in values)
                or values[0] > values[1]
            ):
                raise ValueError(
                    f"controller profile {key} must be a positive ordered range: {source}"
                )
            profile[key] = values
    if controller_id == BITCRAZE_LEE_CONTROLLER_ID:
        limit_mode = tracking.get("feedback_acceleration_limit_mode")
        if limit_mode != "norm_limited_isaac_adapter_v1":
            raise ValueError(
                f"Lee controller profile has unsupported feedback limit mode: {source}"
            )
        profile["feedback_acceleration_limit_mode"] = limit_mode
        for key in ("source_url", "source_commit", "source_file", "adaptation_scope"):
            value = tracking.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Lee controller profile {key} must be a non-empty string: {source}"
                )
            profile[key] = value
    if controller_id == BITCRAZE_MELLINGER_CONTROLLER_ID:
        for key in (
            "source_url",
            "source_commit",
            "source_file",
            "firmware_power_distribution",
            "actuator_translation",
        ):
            value = tracking.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Mellinger controller profile {key} must be a non-empty string: {source}"
                )
            profile[key] = value
    for key, expected in (
        (
            "intermediate_waypoint_requires_settle",
            speed_profile
            in {
                "time-parameterized-trapezoid-so3-guarded-v6",
                "time-parameterized-trapezoid-so3-guarded-v7",
            },
        ),
        ("terminal_waypoint_requires_settle", True),
    ):
        value = tracking.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"controller profile {key} must be boolean: {source}#{record_id}")
        if value is not expected:
            raise ValueError(
                "controller profile uses an unsupported waypoint-settling ABI: "
                f"{source}#{record_id}"
            )
        profile[key] = value
    return profile


def _positive_source_number(payload: dict[str, Any], key: str, source: Path) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"input {key} must be numeric: {source}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"input {key} must be numeric: {source}") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"input {key} must be finite and positive: {source}")
    return number


def _finite_source_number(value: object, label: str, source: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric: {source}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric: {source}") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative: {source}")
    return number


def _validate_source_safety(
    execution: dict[str, Any],
    source: Path,
    record_id: str,
    minimum_static_clearance_m: float,
) -> dict[str, object]:
    """Reject every calibration input lacking a successful trace-safety outcome.

    A completed maneuver is not speed evidence when it collided, left the
    admitted component, triggered an execution guard, or crossed the physical
    0.30 m static-mesh contract.  This deliberately rejects historical schema
    variants which did not write per-agent trace-clearance evidence.
    """

    trace = execution.get("static_trace_clearance")
    if not isinstance(trace, dict):
        raise ValueError(f"input has no static trace-clearance diagnostics: {source}#{record_id}")
    if trace.get("static_clearance_contract_passed") is not True:
        raise ValueError(f"input failed static clearance contract: {source}")
    recorded_required = _finite_source_number(
        trace.get("static_clearance_contract_required_m"),
        "static_trace_clearance.static_clearance_contract_required_m",
        source,
    )
    if not math.isclose(recorded_required, minimum_static_clearance_m, abs_tol=1.0e-12):
        raise ValueError(f"input uses a different static clearance contract: {source}")
    agents = execution.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError(f"input has no P07 execution-agent diagnostics: {source}")
    for agent in agents:
        if not isinstance(agent, dict):
            raise ValueError(f"execution agent diagnostic must be an object: {source}")
        for key in (
            "transit_collision",
            "transit_out_of_bounds",
            "observation_collision",
            "observation_out_of_bounds",
            "static_clearance_contract_violation",
        ):
            if agent.get(key) is not False:
                raise ValueError(f"input has unsafe {key}: {source}")
        agent_required = _finite_source_number(
            agent.get("static_clearance_contract_required_m"),
            f"agent {agent.get('agent_id')} static_clearance_contract_required_m",
            source,
        )
        if not math.isclose(agent_required, minimum_static_clearance_m, abs_tol=1.0e-12):
            raise ValueError(f"input agent uses a different static clearance contract: {source}")
        actual = _finite_source_number(
            agent.get("minimum_static_mesh_clearance_m"),
            f"agent {agent.get('agent_id')} minimum_static_mesh_clearance_m",
            source,
        )
        if actual + 1.0e-12 < minimum_static_clearance_m:
            raise ValueError(f"input agent breached static clearance contract: {source}")
    return {
        "static_clearance_contract_required_m": minimum_static_clearance_m,
        "source_trace_method": trace.get("method"),
        "source_trace_scope": trace.get("scope"),
    }


def _source_execution_profile(
    payload: dict[str, Any], execution: dict[str, Any], source: Path, record_id: str
) -> dict[str, object]:
    """Bind timing evidence to the controller, CF2X asset and integration ABI."""

    asset_hash = payload.get("cf2x_usd_sha256")
    if not isinstance(asset_hash, str) or len(asset_hash) != 64:
        raise ValueError(f"input has no CF2X asset SHA-256: {source}")
    fleet_size = payload.get("fleet_size")
    if fleet_size != FORMAL_FLEET_SIZE:
        raise ValueError(
            f"input fleet_size must equal the formal N={FORMAL_FLEET_SIZE} contract: {source}"
        )
    return {
        "cf2x_usd_sha256": asset_hash,
        "fleet_size": fleet_size,
        "physics_dt_s": _positive_source_number(payload, "physics_dt_s", source),
        "arrival_tolerance_m": _positive_source_number(payload, "arrival_tolerance_m", source),
        "outcome_time_tolerance_s": _finite_source_number(
            payload.get("outcome_time_tolerance_s"),
            "outcome_time_tolerance_s",
            source,
        ),
        "backend_id": execution.get("backend_id", "legacy-unrecorded-backend"),
        "evidence_class": execution.get("evidence_class", "legacy-unrecorded-evidence"),
        "controller_tracking": _controller_tracking_profile(execution, source, record_id),
    }


def _source_censoring_contract(
    execution: dict[str, Any], source: Path, record_id: str, action_budget_s: float
) -> tuple[float, bool]:
    """Return the real execution horizon used to censor an unfinished transit.

    A calibration-only timeout probe deliberately preserves the normal token
    and planning budget, then stops the physical executor earlier.  Its lower
    bound is that real deadline, not the larger decision budget.  Old outcome
    artifacts predate this field and are valid only when their implicit
    execution horizon equals the action budget.
    """

    probe = execution.get("calibration_only_timeout_probe", False)
    if not isinstance(probe, bool):
        raise ValueError(f"calibration_only_timeout_probe must be boolean: {source}")
    deadline = _positive_source_number(
        {"execution_deadline_s": execution.get("execution_deadline_s", action_budget_s)},
        "execution_deadline_s",
        source,
    )
    recorded_deadline = execution.get("execution_deadline_s")
    recorded_probe = execution.get("calibration_only_timeout_probe")
    if recorded_deadline is not None:
        diagnostics_deadline = _finite_source_number(
            recorded_deadline, "execution.execution_deadline_s", source
        )
        if not math.isclose(diagnostics_deadline, deadline, abs_tol=1.0e-12):
            raise ValueError(f"execution deadline disagrees with runtime diagnostics: {source}")
    if recorded_probe is not None and recorded_probe is not probe:
        raise ValueError(f"timeout-probe flag disagrees with runtime diagnostics: {source}")
    if probe:
        if not deadline < action_budget_s:
            raise ValueError(f"timeout probe must end before its action budget: {source}")
    elif not math.isclose(deadline, action_budget_s, abs_tol=1.0e-12):
        raise ValueError(f"non-probe execution deadline must equal action budget: {source}")
    return deadline, probe


def _execution_records(
    payload: dict[str, Any], source: Path
) -> Iterable[tuple[str, float, dict[str, Any]]]:
    schema = payload.get("schema_version")
    if schema == "hm3d-p07-physx-execution-smoke-v1":
        engineering = payload.get("engineering_debug")
        execution = engineering.get("execution") if isinstance(engineering, dict) else None
        if not isinstance(execution, dict):
            raise ValueError(f"input has no execution diagnostics: {source}")
        merged = dict(execution)
        merged.setdefault("execution_deadline_s", payload.get("execution_deadline_s"))
        merged.setdefault(
            "calibration_only_timeout_probe", payload.get("calibration_only_timeout_probe", False)
        )
        yield (
            "legacy-execution",
            _positive_source_number(payload, "action_budget_s", source),
            merged,
        )
        return
    if schema == "hm3d-cf2x-vectorized-outcome-probe-v1":
        clusters = payload.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ValueError(f"vectorized probe has no cluster records: {source}")
        action_budget_s = _positive_source_number(payload, "action_budget_s", source)
        for index, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                raise ValueError(f"vectorized cluster record is malformed: {source}#{index}")
            execution = cluster.get("execution_calibration")
            if not isinstance(execution, dict):
                raise ValueError(f"vectorized cluster lacks calibration evidence: {source}#{index}")
            yield f"cluster{index}", action_budget_s, execution
        return
    if schema != "hm3d-p07-exploration-execution-v1":
        raise ValueError(f"input is not a supported real P07 execution artifact: {source}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"multi-decision P07 input has no decisions: {source}")
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"P07 decision is malformed: {source}#{index}")
        record_id = str(decision.get("decision_id", f"decision{index}"))
        execution = decision.get("execution_calibration")
        if not isinstance(execution, dict):
            raise ValueError(f"P07 decision lacks calibration evidence: {source}#{record_id}")
        if execution.get("schema_version") != "hm3d-cf2x-decision-execution-calibration-v1":
            raise ValueError(f"P07 decision calibration schema mismatch: {source}#{record_id}")
        recorded_hash = execution.get("summary_sha256")
        unhashed = {key: value for key, value in execution.items() if key != "summary_sha256"}
        if recorded_hash != canonical_sha256(unhashed):
            raise ValueError(f"P07 decision calibration hash mismatch: {source}#{record_id}")
        yield (
            record_id,
            _positive_source_number(
                {"action_budget_s": execution.get("token_authorization_duration_s")},
                "action_budget_s",
                source,
            ),
            execution,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision-budget-s", required=True, type=float)
    parser.add_argument("--observation-dwell-s", required=True, type=float)
    parser.add_argument(
        "--minimum-terminal-tracking-margin-s",
        type=float,
        default=0.0,
        help=(
            "Optional non-negative lower bound for the fitted terminal tracking margin. "
            "The final value is the larger of this bound and the outcome-derived value."
        ),
    )
    parser.add_argument(
        "--minimum-intermediate-waypoint-settle-margin-s",
        type=float,
        default=0.0,
        help=(
            "Optional non-negative lower bound for the fitted per-intermediate-waypoint "
            "settle margin. The final value is the larger of this bound and the outcome fit."
        ),
    )
    parser.add_argument(
        "--uncovered-segment-reserve-s",
        type=float,
        default=0.0,
        help=(
            "Per-segment admission reserve for routes longer than the maximum "
            "completed route in this calibration set. This is not a controller "
            "command; it prevents an unvalidated long polyline from consuming "
            "the observation dwell at the action deadline."
        ),
    )
    parser.add_argument(
        "--outcome-time-tolerance-s",
        type=float,
        default=0.25,
        help=(
            "Frozen P07 outcome timing tolerance added to the physical tracking margin. "
            "It must match every source execution profile."
        ),
    )
    parser.add_argument("--calibration-id", default="cf2x-hm3d-p07-r1")
    parser.add_argument(
        "--minimum-static-clearance-m",
        type=float,
        default=0.30,
        help="Physical static-mesh clearance required of every calibration trace.",
    )
    parser.add_argument(
        "--require-route-class",
        action="append",
        choices=("horizontal", "vertical", "turn"),
        default=None,
        help="Require observed completed calibration routes of each named class.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.decision_budget_s <= 0.0 or args.observation_dwell_s <= 0.0:
        raise ValueError("decision budget and observation dwell must be positive")
    if not 0.0 <= args.minimum_terminal_tracking_margin_s < args.decision_budget_s:
        raise ValueError(
            "minimum terminal tracking margin must be non-negative and below the decision budget"
        )
    if not 0.0 <= args.minimum_intermediate_waypoint_settle_margin_s < args.decision_budget_s:
        raise ValueError(
            "minimum intermediate waypoint settle margin must be non-negative and below "
            "the decision budget"
        )
    if not math.isfinite(args.uncovered_segment_reserve_s) or args.uncovered_segment_reserve_s < 0.0:
        raise ValueError("uncovered segment reserve must be finite and non-negative")
    if not math.isfinite(args.outcome_time_tolerance_s) or args.outcome_time_tolerance_s < 0.0:
        raise ValueError("outcome time tolerance must be finite and non-negative")
    if not math.isfinite(args.minimum_static_clearance_m) or args.minimum_static_clearance_m <= 0.0:
        raise ValueError("minimum static clearance must be finite and positive")
    inputs = tuple(path.expanduser().resolve() for path in args.input)
    if len(inputs) < 2 or len(set(inputs)) != len(inputs):
        raise ValueError("provide at least two distinct immutable P07 outcome files")
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("a P07 outcome input is missing")
    rows: list[dict[str, object]] = []
    lower_bounds: list[dict[str, object]] = []
    excluded_stationary: list[dict[str, object]] = []
    execution_profile: dict[str, object] | None = None
    safety_contract: dict[str, object] | None = None
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("synthetic") is not False
            or payload.get("formal_result") is not False
        ):
            raise ValueError(f"input is not an immutable real P07 execution smoke: {path}")
        if "evaluator_private_task_probe" in payload:
            raise ValueError("transit calibration accepts target-free execution evidence only")
        source_hash = _sha256(path)
        for record_id, source_decision_budget_s, execution in _execution_records(payload, path):
            source_execution_profile = _source_execution_profile(
                payload, execution, path, record_id
            )
            if execution_profile is None:
                execution_profile = source_execution_profile
            elif source_execution_profile != execution_profile:
                raise ValueError("calibration inputs use different execution profiles")
            if not math.isclose(
                float(source_execution_profile["outcome_time_tolerance_s"]),
                args.outcome_time_tolerance_s,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "calibration outcome-time tolerance differs from the source execution profile"
                )
            source_safety_contract = _validate_source_safety(
                execution, path, record_id, args.minimum_static_clearance_m
            )
            if safety_contract is None:
                safety_contract = source_safety_contract
            elif source_safety_contract != safety_contract:
                raise ValueError("calibration inputs use different static trace-safety contracts")
            source_execution_deadline_s, calibration_only_timeout_probe = (
                _source_censoring_contract(execution, path, record_id, source_decision_budget_s)
            )
            agents = execution.get("agents")
            if not isinstance(agents, list):
                raise ValueError(f"input has no P07 execution-agent diagnostics: {path}")
            for agent in agents:
                if not isinstance(agent, dict):
                    raise ValueError("execution agent diagnostic must be an object")
                path_length_m = _path_length_m(agent.get("command_path_m"))
                route_geometry = agent.get("route_geometry")
                if not isinstance(route_geometry, dict):
                    route_geometry = _route_geometry(agent.get("command_path_m"))
                route_classes = route_geometry.get("route_classes")
                if not isinstance(route_classes, list) or any(
                    route_class not in {"horizontal", "vertical", "turn", "stationary"}
                    for route_class in route_classes
                ):
                    raise ValueError("execution route classification is malformed")
                if path_length_m <= 0.05:
                    excluded_stationary.append(
                        {
                            "source_file_sha256": source_hash,
                            "source_record_id": record_id,
                            "agent_id": agent.get("agent_id"),
                            "reason": "stationary_hold_is_not_speed_evidence",
                        }
                    )
                    continue
                transit_completed_at_s = agent.get("transit_completed_at_s")
                row = {
                    "source_file_sha256": source_hash,
                    "source_record_id": record_id,
                    "agent_id": agent.get("agent_id"),
                    "command_path_m": agent["command_path_m"],
                    "command_path_length_m": path_length_m,
                    "waypoint_segments": len(agent["command_path_m"]) - 1,
                    "route_classes": route_classes,
                    "route_geometry": route_geometry,
                    "transit_completed": agent.get("transit_completed") is True,
                    "source_decision_budget_s": source_decision_budget_s,
                    "source_execution_deadline_s": source_execution_deadline_s,
                    "calibration_only_timeout_probe": calibration_only_timeout_probe,
                }
                if agent.get("transit_completed") is True:
                    duration_s = float(transit_completed_at_s)
                    if not math.isfinite(duration_s) or duration_s <= 0.0:
                        raise ValueError("completed transit must have a positive duration")
                    if duration_s > source_execution_deadline_s + 1.0e-9:
                        raise ValueError("completed transit extends beyond its execution deadline")
                    row["observed_transit_seconds"] = duration_s
                    rows.append(row)
                else:
                    if agent.get("transit_failure_reason") != "transit_timeout":
                        raise ValueError(
                            "unfinished transit lacks an explicit transit_timeout reason"
                        )
                    if calibration_only_timeout_probe:
                        if agent.get("transit_attempted") is not True:
                            raise ValueError("timeout probe lacks a real attempted-transit outcome")
                        attempt_end_s = _finite_source_number(
                            agent.get("transit_attempt_actual_end_s"),
                            "transit_attempt_actual_end_s",
                            path,
                        )
                        if not math.isclose(
                            attempt_end_s, source_execution_deadline_s, abs_tol=1.0e-9
                        ):
                            raise ValueError(
                                "timeout probe trace does not end at its execution deadline"
                            )
                    lower_bounds.append(
                        {
                            **row,
                            "lower_bound_transit_seconds": source_execution_deadline_s,
                        }
                    )
    # Millimetre-scale coordinate noise must not masquerade as route-length
    # diversity.  Calibration needs genuinely distinct centimetre-scale paths.
    unique_success_paths = {
        (round(float(row["command_path_length_m"]), 2), int(row["waypoint_segments"]))
        for row in rows
    }
    if len(rows) < 3 or len(unique_success_paths) < 3 or not lower_bounds:
        raise ValueError(
            "calibration needs at least three distinct completed paths and one censored timeout"
        )
    if execution_profile is None:
        raise ValueError("calibration inputs do not identify an execution profile")
    controller_tracking = execution_profile["controller_tracking"]
    if not isinstance(controller_tracking, dict):
        raise ValueError("calibration execution profile has no controller timing limits")
    cruise_speed_mps = controller_tracking.get(
        "maximum_reference_speed_mps",
        controller_tracking.get("maximum_translational_speed_mps"),
    )
    max_accel_mps2 = controller_tracking.get(
        "maximum_reference_acceleration_mps2",
        controller_tracking.get("maximum_translational_acceleration_mps2"),
    )
    if cruise_speed_mps is None or max_accel_mps2 is None:
        raise ValueError(
            "transit timing v4 requires an executor profile with speed and acceleration limits"
        )
    intermediate_waypoint_requires_settle = controller_tracking.get(
        "intermediate_waypoint_requires_settle"
    )
    if not isinstance(intermediate_waypoint_requires_settle, bool):
        raise ValueError("controller timing profile omits intermediate waypoint settling semantics")
    continuous_waypoint_speed_mps = float(
        controller_tracking.get("waypoint_pass_through_speed_mps", 0.35)
    )
    required_route_classes = set(args.require_route_class or ())
    if intermediate_waypoint_requires_settle:
        # A terminal-plus-intermediate model cannot be admitted for the current
        # executor until a completed corner route has tested that exact ABI.
        required_route_classes.add("turn")
    covered_route_classes = sorted(
        {route_class for row in rows for route_class in row["route_classes"]}
    )
    missing_route_classes = sorted(required_route_classes - set(covered_route_classes))
    if missing_route_classes:
        raise ValueError(
            "calibration lacks required completed route classes: "
            + ", ".join(missing_route_classes)
        )
    reference_model = ConservativeTransitTimingModel(
        calibration_id=f"{args.calibration_id}-zero-margin-reference",
        cruise_speed_mps=cruise_speed_mps,
        max_accel_mps2=max_accel_mps2,
        terminal_tracking_margin_s=0.0,
        intermediate_waypoint_settle_margin_s=0.0,
        intermediate_waypoint_requires_settle=intermediate_waypoint_requires_settle,
        continuous_waypoint_speed_mps=continuous_waypoint_speed_mps,
    )
    direct_margin_rows = []
    for row in rows:
        path_m = _path_points(row["command_path_m"])
        segment_count = len(path_m) - 1
        if segment_count != 1:
            continue
        reference_motion_s = reference_model.estimate_seconds(path_m)
        observed_s = float(row["observed_transit_seconds"])
        outcome_adjusted_s = observed_s + args.outcome_time_tolerance_s
        direct_margin_rows.append(
            {
                "source_file_sha256": row["source_file_sha256"],
                "source_record_id": row["source_record_id"],
                "agent_id": row["agent_id"],
                "waypoint_segments": segment_count,
                "reference_motion_seconds": reference_motion_s,
                "observed_transit_seconds": observed_s,
                "outcome_adjusted_transit_seconds": outcome_adjusted_s,
                "required_terminal_tracking_margin_s": max(
                    0.0, outcome_adjusted_s - reference_motion_s
                ),
            }
        )
    if not direct_margin_rows:
        raise ValueError("calibration needs at least one completed direct route")
    outcome_fitted_terminal_tracking_margin_s = max(
        float(row["required_terminal_tracking_margin_s"]) for row in direct_margin_rows
    )
    terminal_tracking_margin_s = max(
        args.minimum_terminal_tracking_margin_s,
        outcome_fitted_terminal_tracking_margin_s,
    )
    intermediate_margin_rows = []
    for row in rows:
        path_m = _path_points(row["command_path_m"])
        intermediate_waypoint_count = len(path_m) - 2
        if intermediate_waypoint_count == 0:
            continue
        reference_motion_s = reference_model.estimate_seconds(path_m)
        observed_s = float(row["observed_transit_seconds"])
        outcome_adjusted_s = observed_s + args.outcome_time_tolerance_s
        intermediate_margin_rows.append(
            {
                "source_file_sha256": row["source_file_sha256"],
                "source_record_id": row["source_record_id"],
                "agent_id": row["agent_id"],
                "intermediate_waypoint_count": intermediate_waypoint_count,
                "reference_motion_seconds": reference_motion_s,
                "observed_transit_seconds": observed_s,
                "outcome_adjusted_transit_seconds": outcome_adjusted_s,
                "terminal_tracking_margin_s": terminal_tracking_margin_s,
                "required_intermediate_waypoint_settle_margin_s": max(
                    0.0,
                    (outcome_adjusted_s - reference_motion_s - terminal_tracking_margin_s)
                    / intermediate_waypoint_count,
                ),
            }
        )
    if intermediate_waypoint_requires_settle and not intermediate_margin_rows:
        raise ValueError("intermediate-settling executor needs a completed multi-waypoint route")
    outcome_fitted_intermediate_waypoint_settle_margin_s = max(
        (
            float(row["required_intermediate_waypoint_settle_margin_s"])
            for row in intermediate_margin_rows
        ),
        default=0.0,
    )
    intermediate_waypoint_settle_margin_s = max(
        args.minimum_intermediate_waypoint_settle_margin_s,
        outcome_fitted_intermediate_waypoint_settle_margin_s,
    )
    model = ConservativeTransitTimingModel(
        calibration_id=args.calibration_id,
        cruise_speed_mps=cruise_speed_mps,
        max_accel_mps2=max_accel_mps2,
        terminal_tracking_margin_s=terminal_tracking_margin_s,
        intermediate_waypoint_settle_margin_s=intermediate_waypoint_settle_margin_s,
        calibrated_max_segment_count=max(int(row["waypoint_segments"]) for row in rows),
        uncovered_segment_reserve_s=args.uncovered_segment_reserve_s,
        intermediate_waypoint_requires_settle=intermediate_waypoint_requires_settle,
        continuous_waypoint_speed_mps=continuous_waypoint_speed_mps,
    )
    completed_checks = [
        {
            **row,
            "predicted_transit_seconds": model.estimate_seconds(
                _path_points(row["command_path_m"])
            ),
        }
        for row in rows
    ]
    for row in completed_checks:
        if float(row["predicted_transit_seconds"]) + 1.0e-9 < float(
            row["observed_transit_seconds"]
        ):
            raise ValueError("calibrated model underestimates a completed calibration path")
    usable_completed_checks = [
        row
        for row in completed_checks
        if float(row["predicted_transit_seconds"]) + args.observation_dwell_s
        <= args.decision_budget_s + 1.0e-9
    ]
    conservatively_rejected_completed_checks = [
        {
            **row,
            "reason": "conservative_upper_envelope_exceeds_target_decision_budget",
        }
        for row in completed_checks
        if float(row["predicted_transit_seconds"]) + args.observation_dwell_s
        > args.decision_budget_s + 1.0e-9
    ]
    usable_success_lengths = {
        round(float(row["command_path_length_m"]), 2) for row in usable_completed_checks
    }
    if len(usable_success_lengths) < 3:
        raise ValueError(
            "calibrated model leaves fewer than three distinct completed paths usable "
            "under the target decision budget"
        )
    usable_route_classes = {
        route_class for row in usable_completed_checks for route_class in row["route_classes"]
    }
    missing_usable_route_classes = sorted(required_route_classes - usable_route_classes)
    if missing_usable_route_classes:
        raise ValueError(
            "calibrated model leaves no target-budget-usable completed route for required "
            "classes: " + ", ".join(missing_usable_route_classes)
        )
    timeout_checks = [
        {
            **row,
            "predicted_transit_seconds": model.estimate_seconds(
                _path_points(row["command_path_m"])
            ),
        }
        for row in lower_bounds
    ]
    if not any(
        float(row["predicted_transit_seconds"]) + args.observation_dwell_s
        > float(row["source_execution_deadline_s"]) + 1.0e-9
        for row in timeout_checks
    ):
        raise ValueError("calibrated model does not reject any observed censoring deadline")
    payload = {
        "schema_version": "hm3d-cf2x-transit-timing-calibration-v4",
        "status": "CALIBRATION_PASS",
        "formal_result": False,
        "p07_task_validity_closed": False,
        "claim_limit": (
            "Outcome-calibrated public transit-time contract only. It is not a weak-baseline "
            "score, target-search metric, QD result, OGFR result, or RL result."
        ),
        "time_model": model.to_dict(),
        "execution_profile": execution_profile,
        "execution_profile_sha256": canonical_sha256(execution_profile),
        "controller_tracking_profile": execution_profile["controller_tracking"],
        "controller_tracking_profile_sha256": canonical_sha256(
            execution_profile["controller_tracking"]
        ),
        "static_trace_safety_contract": safety_contract,
        "decision_budget_s": args.decision_budget_s,
        "observation_dwell_s": args.observation_dwell_s,
        "outcome_time_tolerance_s": args.outcome_time_tolerance_s,
        "minimum_terminal_tracking_margin_s": args.minimum_terminal_tracking_margin_s,
        "outcome_fitted_terminal_tracking_margin_s": outcome_fitted_terminal_tracking_margin_s,
        "terminal_tracking_margin_fit_rows": direct_margin_rows,
        "minimum_intermediate_waypoint_settle_margin_s": (
            args.minimum_intermediate_waypoint_settle_margin_s
        ),
        "calibrated_max_segment_count": model.calibrated_max_segment_count,
        "uncovered_segment_reserve_s": model.uncovered_segment_reserve_s,
        "intermediate_waypoint_requires_settle": (
            intermediate_waypoint_requires_settle
        ),
        "continuous_waypoint_speed_mps": continuous_waypoint_speed_mps,
        "outcome_fitted_intermediate_waypoint_settle_margin_s": (
            outcome_fitted_intermediate_waypoint_settle_margin_s
        ),
        "intermediate_waypoint_settle_margin_fit_rows": intermediate_margin_rows,
        "completed_transit_checks": completed_checks,
        "target_budget_usable_completed_checks": usable_completed_checks,
        "conservatively_rejected_completed_checks": (conservatively_rejected_completed_checks),
        "censored_timeout_checks": timeout_checks,
        "excluded_stationary_transits": excluded_stationary,
        "route_class_coverage": {
            "observed": covered_route_classes,
            "target_budget_usable": sorted(usable_route_classes),
            "required": sorted(required_route_classes),
        },
        "conservative_false_negative_audit": {
            "completed_path_count": len(completed_checks),
            "target_budget_usable_completed_path_count": len(usable_completed_checks),
            "conservatively_rejected_completed_path_count": len(
                conservatively_rejected_completed_checks
            ),
            "interpretation": (
                "A completed path may be rejected under the target decision budget when the "
                "shared upper envelope is slower than that individual trace. This is a safe "
                "false negative and is reported as candidate-availability cost."
            ),
        },
        "timeout_censoring_contract": (
            "A censored transit lower bound is the recorded physical execution deadline. "
            "Calibration-only timeout probes retain no evaluator-private fields and "
            "cannot enter replay."
        ),
        "timing_margin_contract": (
            "The terminal tracking margin is fit from direct route residuals after "
            "executor-aligned rest-to-rest motion time. The intermediate waypoint margin is "
            "fit from completed multi-waypoint residuals after subtracting that terminal "
            "margin, normalized by intermediate waypoint count. The resulting prediction "
            "remains an empirical upper envelope of every completed calibration transit."
        ),
        "source_evidence": [
            {"path": str(path), "sha256": _sha256(path)} for path in sorted(inputs)
        ],
        "identifiability": {
            "reference_speed_and_acceleration": "bound_to_executor_profile",
            "three_dimensional_segments": "executor_aligned_rest_to_rest_profile",
            "controller_terminal_convergence": "outcome_validated_terminal_tracking_margin",
            "intermediate_waypoint_settling": "outcome_validated_turn_margin",
            "multi_waypoint_route": (
                "completed_turn_required_when_executor_requires_intermediate_settling"
            ),
            "short_route_average_speed": "not_extrapolated",
        },
    }
    payload["calibration_record_sha256"] = canonical_sha256(payload)
    _write_new_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
