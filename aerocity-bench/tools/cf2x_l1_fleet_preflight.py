"""Exercise four local CF2X vehicles in one shared Isaac PhysX world.

This is an internal engineering preflight, not a benchmark runner.  It drives
a selected public reference policy through measured CF2X observations,
then lets the evaluator-private service consume only valid ``OBSERVE`` actions.
All four motor targets are written before exactly one ``sim.step()`` per
physics tick.  The resulting evidence is intentionally ineligible for a
formal score until CF2X provenance, actuator calibration, collision policy,
and the complete L1 executor are frozen.

The tool uses a short candidate budget by default.  ``--max-sim-time-s`` is
simulated time, not a host wait.  A separate complete-calibration purpose must
match the frozen episode duration exactly and remains ineligible for a formal
test score.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = BENCH_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from aerocity_bench.isaaclab_paths import discover_isaaclab_paths  # noqa: E402

_ISAACLAB_PATHS = discover_isaaclab_paths(BENCH_ROOT)
DRONE_PROJECT_ROOT = _ISAACLAB_PATHS.drone_project_root
ISAACLAB_ROOT = _ISAACLAB_PATHS.isaaclab_root
ISAACLAB_SOURCE_ROOT = _ISAACLAB_PATHS.source_root
for _path in (
    BENCH_ROOT / "src",
    DRONE_PROJECT_ROOT,
    ISAACLAB_SOURCE_ROOT / "isaaclab" if ISAACLAB_SOURCE_ROOT else None,
    ISAACLAB_SOURCE_ROOT / "isaaclab_contrib" if ISAACLAB_SOURCE_ROOT else None,
    ISAACLAB_SOURCE_ROOT / "isaaclab_assets" if ISAACLAB_SOURCE_ROOT else None,
    ISAACLAB_SOURCE_ROOT / "isaaclab_tasks" if ISAACLAB_SOURCE_ROOT else None,
):
    if _path is not None and _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aerocity_bench.cf2x_fleet_preflight_contract import (
    COMPLETE_CALIBRATION_PURPOSE,
    EXTERNAL_PROCESS_POLICY_MODE,
    FLEET_PRECHECK_SCOPE,
    FLEET_PRIVATE_SCOPE,
    PRIVATE_WITNESS_FIXTURE_MODE,
    SHORT_PREFLIGHT_PURPOSE,
    SharedWorldStepLedger,
    altitude_stability_metrics,
    assert_action_roster_complete,
    candidate_shared_hold_assessment,
    public_fleet_members,
    public_policy_progress_status,
    validate_fleet_preflight_reports,
    validate_native_run_purpose,
)
from aerocity_bench.measurement_evidence import L1MeasurementEvidence
from aerocity_bench.planning_cadence import PlanningCadenceController

PUBLIC_POLICY_METHODS = (
    "sweep-3d",
    "atlas-surface-inspector",
    "atlas-region-greedy",
)
EXTERNAL_PROCESS_INITIALIZATION_DEADLINE_S = 10.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Isaac Lab 2.3.0 probes the parser with parse_known_args() while adding
    # launcher options.  Required application arguments would make parser
    # construction (and --help) exit before the launcher options are added.
    # Register them first for Isaac Lab's collision checks, then restore their
    # required semantics after the launcher has finished extending the parser.
    required_actions = [
        parser.add_argument("--layout-root", type=Path),
        parser.add_argument("--release-config", type=Path),
        parser.add_argument("--output", type=Path),
    ]
    parser.add_argument("--private-output", type=Path)
    required_actions.append(parser.add_argument("--cf2x-usd", type=Path))
    parser.add_argument("--episode-name", default="episode-0000.json")
    parser.add_argument(
        "--execution-mode",
        choices=(
            "shared-hold",
            "public-policy",
            EXTERNAL_PROCESS_POLICY_MODE,
            PRIVATE_WITNESS_FIXTURE_MODE,
        ),
        default="shared-hold",
        help=(
            "shared-hold verifies four simultaneous CF2X altitude stability; "
            "public-policy exercises an internal public method; external-process-policy "
            "uses a pinned external G2-I process; private-witness-fixture "
            "is an evaluator-owned internal closure check, never a public method."
        ),
    )
    parser.add_argument("--method", choices=PUBLIC_POLICY_METHODS, default="sweep-3d")
    parser.add_argument(
        "--external-adapter-manifest",
        type=Path,
        help="pinned external L1 process manifest; required only for external-process-policy",
    )
    parser.add_argument(
        "--run-purpose",
        choices=(SHORT_PREFLIGHT_PURPOSE, COMPLETE_CALIBRATION_PURPOSE),
        default=SHORT_PREFLIGHT_PURPOSE,
        help=(
            "short engineering evidence or one complete development/calibration episode; "
            "neither is a formal test score"
        ),
    )
    parser.add_argument("--max-sim-time-s", type=float, default=12.0)
    parser.add_argument("--guidance-horizontal-speed-mps", type=float, default=1.5)
    parser.add_argument("--guidance-vertical-speed-mps", type=float, default=1.0)
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    parser.add_argument("--stability-sample-period-s", type=float, default=1.0)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--device", type=str, default="cpu")
        for action in required_actions:
            action.required = True
        return parser
    AppLauncher.add_app_launcher_args(parser)
    for action in required_actions:
        action.required = True
    return parser


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    from aerocity_bench.canonical import write_json_atomic

    write_json_atomic(path, payload)


def _write_large_private_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    from aerocity_bench.canonical import write_json_atomic_compact

    write_json_atomic_compact(path, payload)


def _latency_summary(samples: list[float]) -> dict[str, float]:
    """Return a stable percentile summary for non-empty wall/CPU samples."""

    if not samples:
        raise ValueError("latency summary requires at least one sample")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "p50_s": percentile(0.50),
        "p95_s": percentile(0.95),
        "p99_s": percentile(0.99),
        "max_s": ordered[-1],
    }


def _substage_latency_summary(samples: list[float]) -> dict[str, float | int | None]:
    """Summarize one optional external-planner substage without inventing samples."""

    if not samples:
        return {
            "call_count": 0,
            "p50_s": None,
            "p95_s": None,
            "p99_s": None,
            "max_s": None,
        }
    return {"call_count": len(samples), **_latency_summary(samples)}


def _external_process_substage_summary(
    timing_trace: list[dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Expose method-neutral timing attribution without exposing planner inputs."""

    fields = (
        "bridge_act_wall_clock_s",
        "projection_wall_clock_s",
        "request_public_audit_wall_clock_s",
        "request_json_serialize_wall_clock_s",
        "request_size_check_wall_clock_s",
        "request_write_flush_wall_clock_s",
        "response_wait_wall_clock_s",
        "response_json_decode_wall_clock_s",
        "response_validate_wall_clock_s",
        "action_validation_conversion_wall_clock_s",
        "bridge_internal_unattributed_wall_clock_s",
        "fleet_arbitration_wall_clock_s",
        "unattributed_wall_clock_s",
    )
    summaries: dict[str, dict[str, float | int | None]] = {}
    for field in fields:
        samples: list[float] = []
        for item in timing_trace:
            value = item.get(field)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"external planner timing {field} is not numeric")
            samples.append(float(value))
        summaries[field.removesuffix("_s")] = _substage_latency_summary(samples)
    return summaries


def _progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def _failure_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failure.json")


def _private_path(output: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    return output.with_name(f"{output.stem}.private.json")


def _validated_output_paths(output: Path, requested_private: Path | None) -> tuple[Path, Path]:
    """Refuse stale or overlapping evidence before starting an Isaac process."""

    public_path = output.resolve()
    private_path = _private_path(public_path, requested_private)
    for label, path in (("public output", public_path), ("private output", private_path)):
        if path.suffix.lower() != ".json":
            raise ValueError(f"{label} must be a .json evidence file: {path}")
    if public_path == private_path:
        raise ValueError("fleet public and private evidence outputs must differ")
    generated = (
        public_path,
        private_path,
        _progress_path(public_path),
        _failure_path(public_path),
    )
    existing = [str(path) for path in generated if path.exists()]
    if existing:
        raise FileExistsError(
            "fleet preflight evidence paths already exist; choose a new output name: "
            + ", ".join(existing)
        )
    return public_path, private_path


def _write_progress(
    output: Path,
    stage: str,
    *,
    status: str = "IN_PROGRESS",
    **details: Any,
) -> None:
    if status not in {"IN_PROGRESS", "COMPLETED", "FAILED", "REJECTED"}:
        raise ValueError(f"unsupported fleet preflight progress status: {status}")
    _write_json_atomic(
        _progress_path(output),
        {
            "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-progress.v1",
            "status": status,
            "formal_score_eligible": False,
            "evidence_scope": FLEET_PRECHECK_SCOPE,
            "stage": stage,
            "timestamp_unix_s": time.time(),
            **details,
        },
    )


def _finite_list(tensor: Any) -> list[float]:
    values = [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Isaac returned a non-finite CF2X state")
    return values


def _norm(values: tuple[float, float, float] | list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return _norm([left - right for left, right in zip(first, second, strict=True)])


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _euler_from_wxyz(quaternion: list[float]) -> tuple[float, float, float]:
    if len(quaternion) != 4:
        raise ValueError("orientation_wxyz must have four values")
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _pose_from_state(state: dict[str, Any]) -> Any:
    from aerocity_bench.contracts import Pose3D

    roll, pitch, yaw = _euler_from_wxyz(state["orientation_wxyz"])
    return Pose3D(
        position=tuple(float(value) for value in state["position"]),
        yaw_deg=math.degrees(yaw),
        pitch_deg=math.degrees(pitch),
        roll_deg=math.degrees(roll),
    )


def _requested_sensor_pitch(
    current_pitch_deg: float, action: Any, execution: dict[str, Any], *, deadline_miss: bool
) -> float:
    """Advance the contracted bounded gimbal independently of CF2X body pitch."""

    rig = execution["sensor_rig"]
    if rig["gimbal_mode"] == "fixed":
        if action.sensor_pitch_deg is not None:
            raise ValueError("fixed sensor rig cannot accept a gimbal pitch command")
        return float(current_pitch_deg)
    if deadline_miss:
        return float(current_pitch_deg)
    if action.kind == "OBSERVE" and action.sensor_pitch_deg is not None:
        raise ValueError("OBSERVE cannot move the bounded sensor gimbal")
    target = (
        current_pitch_deg if action.sensor_pitch_deg is None else float(action.sensor_pitch_deg)
    )
    lower, upper = (float(value) for value in rig["pitch_limits_deg"])
    if not lower <= target <= upper:
        raise ValueError("gimbal pitch command lies outside public contract limits")
    period = float(execution["control_period_s"])
    maximum_delta = float(rig["max_pitch_rate_deg_s"]) * period
    delta = max(-maximum_delta, min(maximum_delta, target - current_pitch_deg))
    return max(lower, min(upper, current_pitch_deg + delta))


def _state(
    robot: Any,
    *,
    rotor_reference_rad_s: tuple[float, float, float, float],
    applied_rotor_thrust_n: tuple[float, float, float, float],
    body_wrench: tuple[float, float, float, float],
    contact_force_n: float,
) -> dict[str, Any]:
    return {
        "position": _finite_list(robot.data.root_pos_w[0]),
        "orientation_wxyz": _finite_list(robot.data.root_quat_w[0]),
        "linear_velocity_mps": _finite_list(robot.data.root_lin_vel_w[0]),
        "angular_velocity_rad_s": _finite_list(robot.data.root_ang_vel_w[0]),
        "rotor_reference_rad_s": [float(value) for value in rotor_reference_rad_s],
        "applied_rotor_thrust_n": [float(value) for value in applied_rotor_thrust_n],
        "body_wrench": [float(value) for value in body_wrench],
        "contact_force_n": float(contact_force_n),
    }


def _within_bounds(
    position: tuple[float, float, float], bounds: dict[str, Any], margin: float
) -> bool:
    return all(
        float(low) + margin <= value <= float(high) - margin
        for value, low, high in zip(position, bounds["minimum"], bounds["maximum"], strict=True)
    )


def _effective_vertical_safe_bounds(
    task_spec: dict[str, Any], vehicle: dict[str, Any]
) -> tuple[float, float]:
    """Return center-of-mass altitude bounds with the contracted clearance margin.

    The flight envelope is a body-center bound, while the execution contract
    separately requires clearance from colliders.  Using only ``radius_m``
    here reproduces the historical slow-descent failure: a waypoint can be
    geometrically inside the envelope while leaving no room for controller
    tracking error or the required clearance.  This helper does not alter the
    public task; it makes the native candidate guard fail closed on that
    already-frozen contract.
    """

    radius_m = float(vehicle["radius_m"])
    clearance_m = float(vehicle["minimum_clearance_m"])
    minimum = float(task_spec["flight_bounds"]["minimum"][2]) + radius_m + clearance_m
    maximum = float(task_spec["flight_bounds"]["maximum"][2]) - radius_m - clearance_m
    if maximum <= minimum:
        raise ValueError("contracted vertical safety envelope is empty")
    return minimum, maximum


def _local_occupancy(
    position: tuple[float, float, float], colliders: list[Any]
) -> tuple[tuple[float, float, float], float, float, tuple[tuple[int, int, int], ...]]:
    """Match the documented G1 voxel shape without exposing target truth."""

    radius_m = 14.0
    resolution_m = 2.0
    origin = tuple(round(value / resolution_m) * resolution_m for value in position)
    index_limit = math.ceil(radius_m / resolution_m)
    occupied: set[tuple[int, int, int]] = set()
    for collider in colliders:
        if collider.point_distance(position) > radius_m:
            continue
        ranges: list[range] = []
        for axis in range(3):
            low = math.ceil((collider.minimum[axis] - origin[axis]) / resolution_m - 0.5)
            high = math.floor((collider.maximum[axis] - origin[axis]) / resolution_m + 0.5)
            ranges.append(range(max(-index_limit, low), min(index_limit, high) + 1))
        for x_index in ranges[0]:
            for y_index in ranges[1]:
                for z_index in ranges[2]:
                    center = (
                        origin[0] + x_index * resolution_m,
                        origin[1] + y_index * resolution_m,
                        origin[2] + z_index * resolution_m,
                    )
                    if (
                        _distance(position, center)
                        <= radius_m + math.sqrt(3.0) * resolution_m / 2.0
                    ):
                        occupied.add((x_index, y_index, z_index))
    return origin, resolution_m, radius_m, tuple(sorted(occupied))


def _observation_receipt_dict(receipt: Any) -> dict[str, Any]:
    return {
        "observation_id": receipt.observation_id,
        "drone_id": receipt.drone_id,
        "timestamp_s": receipt.timestamp_s,
        "accepted": receipt.accepted,
        "reason": receipt.reason,
        "receipt_hash": receipt.receipt_hash,
    }


def _public_summary(private: dict[str, Any]) -> dict[str, Any]:
    final = private["final"]
    summary = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight.v4",
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRECHECK_SCOPE,
        "not_a_formal_l1_episode": True,
        "complete_calibration_replay": (
            private["execution_purpose"] == COMPLETE_CALIBRATION_PURPOSE
        ),
        "reason_not_formal": (
            "development/calibration replay is not a formal test score"
            if private["execution_purpose"] == COMPLETE_CALIBRATION_PURPOSE
            else (
                "candidate CF2X parameters and controller remain parameter_audit_pending; "
                "this is a short shared-world preflight only"
            )
        ),
        "input_bindings": private["input_bindings"],
        "execution_mode": private["execution_mode"],
        "execution_purpose": private["execution_purpose"],
        "method": private["method"],
        "vehicle": private["vehicle_public"],
        "execution": {
            "control_ticks": private["execution"]["control_ticks"],
            "control_period_s": private["execution"]["control_period_s"],
            "shared_physx_step_count": private["execution"]["shared_physx_step_count"],
            "simulated_time_s": private["execution"]["simulated_time_s"],
            "wall_clock_s": private["execution"]["wall_clock_s"],
            "execution_receipt_count": len(private["execution_receipts"]),
            "observation_receipt_count": len(private["observation_receipts"]),
            "confirmed_receipt_count": len(private["confirmation_receipts"]),
            "failure_record_count": len(private["failure_records"]),
        },
        "flight_stability": private["flight_stability_public"],
        "candidate_shared_hold": private["candidate_shared_hold"],
        "route_budget_audit": private["route_budget_audit"],
        "planning_timing": private["planning_timing"],
        "policy_progress": private["policy_progress"],
        "external_adapter": private["external_adapter"],
        "final": {
            "safe_completion": final["safe_completion"],
            "collision_detected": final["collision_detected"],
            "out_of_bounds_detected": final["out_of_bounds_detected"],
            "all_returned_home": final["all_returned_home"],
        },
        "private_evaluator_commitment": private["private_evaluator_commitment"],
        "private_report_file_sha256": private["private_report_file_sha256"],
    }
    if private["execution_mode"] == PRIVATE_WITNESS_FIXTURE_MODE:
        summary["private_fixture_commitment"] = private["private_fixture_commitment"]
    return summary


def _validate_complete_calibration_public_inputs(
    args: argparse.Namespace,
    task_spec: dict[str, Any],
    public_episode: dict[str, Any],
) -> None:
    """Reject malformed complete-replay inputs before opening an Isaac process."""

    if args.run_purpose != COMPLETE_CALIBRATION_PURPOSE:
        return
    if task_spec.get("task_track") != "G2-I":
        raise ValueError("complete calibration replay requires a public G2-I task")
    atlas = task_spec.get("inspection_atlas")
    if not isinstance(atlas, dict) or not isinstance(atlas.get("atlas_hash"), str):
        raise ValueError("complete calibration replay requires the full public inspection atlas")
    sector = public_episode.get("mission_sector")
    sector_hash = public_episode.get("mission_sector_hash")
    if (
        not isinstance(sector, dict)
        or not isinstance(sector_hash, str)
        or sector.get("sector_hash") != sector_hash
    ):
        raise ValueError("complete calibration replay requires its frozen public mission sector")


def _validate_layout_execution_contract(task_spec: dict[str, Any], config: Any) -> None:
    """Bind the public task contract to the requested release before Isaac starts."""

    from aerocity_bench.canonical import content_hash
    from aerocity_bench.ordinary_config import public_execution_contract

    public_contract = public_execution_contract(config.raw["execution_contract"])
    if task_spec.get("execution_contract") != public_contract:
        raise ValueError("fleet task execution contract differs from release configuration")
    if task_spec.get("public_execution_contract_hash") != content_hash(public_contract):
        raise ValueError("fleet task public execution-contract hash is invalid")


def _cpu_only_run_contract_validation(args: argparse.Namespace) -> None:
    """Validate native-run semantics and complete inputs before AppLauncher."""

    from aerocity_bench.adapters import load_external_l1_adapter_manifest
    from aerocity_bench.canonical import read_json
    from aerocity_bench.ordinary_config import load_ordinary_config
    from aerocity_bench.public_boundary import audit_public_layout

    config = load_ordinary_config(args.release_config.resolve())
    validate_native_run_purpose(
        purpose=str(args.run_purpose),
        execution_mode=str(args.execution_mode),
        requested_sim_time_s=float(args.max_sim_time_s),
        frozen_episode_duration_s=float(config.raw["execution_contract"]["episode"]["duration_s"]),
    )
    layout_root = args.layout_root.resolve()
    audit_public_layout(layout_root)
    task_spec = read_json(layout_root / "method_public" / "task_spec.json")
    if not isinstance(task_spec, dict):
        raise ValueError("public task specification must be a JSON object")
    _validate_layout_execution_contract(task_spec, config)
    if args.execution_mode == EXTERNAL_PROCESS_POLICY_MODE:
        if args.external_adapter_manifest is None:
            raise ValueError("external-process-policy requires --external-adapter-manifest")
        load_external_l1_adapter_manifest(args.external_adapter_manifest)
        if task_spec.get("task_track") != "G2-I":
            raise ValueError("external-process-policy requires a public G2-I task")
    elif args.external_adapter_manifest is not None:
        raise ValueError("--external-adapter-manifest is only valid for external-process-policy")
    if args.run_purpose != COMPLETE_CALIBRATION_PURPOSE:
        return
    task_spec = read_json(layout_root / "method_public" / "task_spec.json")
    public_episode = read_json(layout_root / "method_public" / "episodes" / args.episode_name)
    if not isinstance(task_spec, dict) or not isinstance(public_episode, dict):
        raise ValueError("complete calibration public inputs must be JSON objects")
    _validate_complete_calibration_public_inputs(args, task_spec, public_episode)


def _public_policy_budget_audit(args: argparse.Namespace) -> dict[str, Any] | None:
    """Reject a known-impossible public route before launching Isaac.

    This is intentionally public-only: it reads the release configuration,
    public task projection, and public episode start states.  In particular it
    never opens the evaluator-private episode just to decide that an ordered
    baseline route exceeds the frozen task budget.
    """

    if args.execution_mode != "public-policy":
        return None
    from aerocity_bench.baselines import create_baseline
    from aerocity_bench.canonical import read_json
    from aerocity_bench.ordinary_config import load_ordinary_config
    from aerocity_bench.public_boundary import audit_public_layout

    layout_root = args.layout_root.resolve()
    audit_public_layout(layout_root)
    config = load_ordinary_config(args.release_config.resolve())
    task_spec = read_json(layout_root / "method_public" / "task_spec.json")
    public_episode = read_json(layout_root / "method_public" / "episodes" / args.episode_name)
    vehicle = config.raw["execution_contract"]["vehicle"]
    if args.guidance_horizontal_speed_mps > float(vehicle["horizontal_speed_mps"]):
        raise ValueError("fleet candidate horizontal guidance exceeds release speed cap")
    if args.guidance_vertical_speed_mps > float(vehicle["vertical_speed_mps"]):
        raise ValueError("fleet candidate vertical guidance exceeds release speed cap")
    policy = create_baseline(args.method, config, task_spec, public_episode)
    audit = getattr(policy, "route_budget_audit", None)
    if not callable(audit):
        raise ValueError(f"{args.method} does not expose a public route-budget audit")
    report = audit(
        horizontal_speed_mps=float(args.guidance_horizontal_speed_mps),
        vertical_speed_mps=float(args.guidance_vertical_speed_mps),
    )
    if report.get("status") not in {"LOWER_BOUND_FITS", "BUDGET_INFEASIBLE"}:
        raise ValueError("public baseline route-budget audit returned an invalid status")
    return report


def _select_private_fixture(
    private_episode: dict[str, Any], public_episode: dict[str, Any]
) -> dict[str, Any]:
    """Select one evaluator-owned, prevalidated observation witness.

    This helper is intentionally only called by the internal fixture mode.  Its
    result must remain private: the public report carries a commitment hash,
    never the selected target, witness, route, or owner identity.
    """

    starts = {str(item["drone_id"]): item for item in public_episode["starts"]}
    candidates: list[tuple[tuple[float, str, str], dict[str, Any]]] = []
    for target in private_episode.get("targets", []):
        if not isinstance(target, dict) or target.get("valid_before_run") is not True:
            continue
        for witness in target.get("legal_witnesses", []):
            if not isinstance(witness, dict):
                continue
            proof = witness.get("reachability_proof")
            if not isinstance(proof, dict):
                continue
            drone_id = str(proof.get("start_drone_id", ""))
            distance_m = float(proof.get("path_distance_upper_bound_m", math.inf))
            if drone_id not in starts or not math.isfinite(distance_m) or distance_m <= 0.0:
                continue
            candidates.append(
                (
                    (
                        distance_m,
                        str(target.get("target_id", "")),
                        str(witness.get("witness_id", "")),
                    ),
                    {"target": target, "witness": witness, "start": starts[drone_id]},
                )
            )
    if not candidates:
        raise ValueError("private episode has no prevalidated witness tied to a public start")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _safe_sky_route(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    transit_altitude_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if transit_altitude_m + 1.0e-9 < max(start[2], end[2]):
        raise ValueError("private fixture transit altitude is below a route endpoint")
    return (
        (start[0], start[1], transit_altitude_m),
        (end[0], end[1], transit_altitude_m),
        end,
    )


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_contrib.assets import Multirotor

    from aerocity_bench.adapters import (
        ExternalProcessPlannerBridge,
        arbitrate_public_fleet_actions,
        load_external_l1_adapter_manifest,
    )
    from aerocity_bench.baselines import create_baseline
    from aerocity_bench.canonical import content_hash, file_hash, read_json
    from aerocity_bench.cf2x_contract import (
        inspect_verified_cf2x_structure,
        verify_local_cf2x_asset,
    )
    from aerocity_bench.cf2x_native import (
        build_cf2x_multirotor_cfg,
        read_verified_cf2x_runtime_mass_kg,
    )
    from aerocity_bench.contracts import ActionPacket, FailureRecord, ObservationPacket, Pose3D
    from aerocity_bench.evaluator import PrivateEvaluator
    from aerocity_bench.geometry import colliders_from_city, minimum_clearance
    from aerocity_bench.isaac_bridge import build_l1_execution_receipt
    from aerocity_bench.ordinary_config import (
        load_ordinary_config,
    )
    from aerocity_bench.quadrotor_dynamics import (
        FlightCommand,
        FlightState,
        candidate_controller_spec,
        controller_step,
        hover_rotor_speed_for_mass,
        project_asset_spec,
        rotor_thrust_wrench,
    )
    from aerocity_bench.quadrotor_guidance import (
        VelocityGuidanceLimits,
        VerticalBoundaryGuard,
        YawAlignmentGuard,
        position_anchored_velocity_guidance,
        three_leg_sky_route_waypoint_yaw,
        yaw_aligned_translation_goal,
    )

    output_path, private_output_path = _validated_output_paths(args.output, args.private_output)
    layout_root = args.layout_root.resolve()
    config_path = args.release_config.resolve()
    stage_path = layout_root / "scene_authority" / "stage.usda"
    city_path = layout_root / "scene_authority" / "cityspec.json"
    task_path = layout_root / "method_public" / "task_spec.json"
    public_episode_path = layout_root / "method_public" / "episodes" / args.episode_name
    private_episode_path = layout_root / "evaluator_private" / "episodes" / args.episode_name
    required = (
        stage_path,
        city_path,
        task_path,
        public_episode_path,
        private_episode_path,
        config_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fleet preflight input is absent: {missing}")
    if args.contact_threshold_n <= 0.0 or args.stability_sample_period_s <= 0.0:
        raise ValueError("fleet contact threshold and stability sample period must be positive")

    _write_progress(output_path, "loading_public_and_private_contracts")
    config = load_ordinary_config(config_path)
    city = read_json(city_path)
    task_spec = read_json(task_path)
    public_episode = read_json(public_episode_path)
    private_episode = read_json(private_episode_path)
    if public_episode.get("episode_id") != private_episode.get("episode_id"):
        raise ValueError("fleet public/private episode IDs differ")
    if city.get("layout_hash") != private_episode.get("layout_hash"):
        raise ValueError("fleet private episode belongs to another CitySpec")
    if task_spec.get("layout_id") != city.get("layout_id"):
        raise ValueError("fleet task specification belongs to another CitySpec")
    _validate_layout_execution_contract(task_spec, config)
    external_manifest = (
        load_external_l1_adapter_manifest(args.external_adapter_manifest)
        if args.execution_mode == EXTERNAL_PROCESS_POLICY_MODE
        else None
    )
    members = public_fleet_members(public_episode)
    if int(task_spec["fleet_profile"]["count"]) != len(members):
        raise ValueError("fleet task profile does not match four public starts")
    vehicle = config.raw["execution_contract"]["vehicle"]
    if args.guidance_horizontal_speed_mps <= 0.0 or args.guidance_vertical_speed_mps <= 0.0:
        raise ValueError("fleet guidance speeds must be positive")
    if args.guidance_horizontal_speed_mps > float(vehicle["horizontal_speed_mps"]):
        raise ValueError("fleet candidate horizontal guidance exceeds release speed cap")
    if args.guidance_vertical_speed_mps > float(vehicle["vertical_speed_mps"]):
        raise ValueError("fleet candidate vertical guidance exceeds release speed cap")
    guidance = VelocityGuidanceLimits(
        horizontal_speed_mps=float(args.guidance_horizontal_speed_mps),
        vertical_speed_mps=float(args.guidance_vertical_speed_mps),
    )
    guidance.validate()
    execution = config.raw["execution_contract"]
    validate_native_run_purpose(
        purpose=str(args.run_purpose),
        execution_mode=str(args.execution_mode),
        requested_sim_time_s=float(args.max_sim_time_s),
        frozen_episode_duration_s=float(execution["episode"]["duration_s"]),
    )
    period = float(execution["control_period_s"])
    physical_dt = project_asset_spec().physics_dt_s
    physical_steps_per_control = int(round(period / physical_dt))
    if not math.isclose(physical_steps_per_control * physical_dt, period, abs_tol=1.0e-9):
        raise ValueError("fleet control period must be an integer number of physical ticks")
    effective_vertical_minimum, effective_vertical_maximum = _effective_vertical_safe_bounds(
        task_spec, vehicle
    )
    vertical_guard = VerticalBoundaryGuard(
        minimum_safe_altitude_m=effective_vertical_minimum,
        maximum_safe_altitude_m=effective_vertical_maximum,
        # Candidate-only: lower than the controller's vector cap so it leaves
        # braking authority while the vehicle is tilted.
        guaranteed_braking_deceleration_mps2=0.25,
        response_horizon_s=period,
        reserve_distance_m=0.5,
    )
    vertical_guard.validate()
    yaw_alignment_guard = YawAlignmentGuard(
        activation_yaw_error_rad=math.radians(90.0),
        release_yaw_error_rad=math.radians(5.0),
        release_yaw_rate_rad_s=math.radians(float(execution["observe"]["max_angular_speed_deg_s"])),
    )
    yaw_alignment_guard.validate()
    guidance_contract = {
        **guidance.to_dict(),
        "vertical_boundary_guard": vertical_guard.to_dict(),
        "yaw_alignment_guard": yaw_alignment_guard.to_dict(),
    }
    maximum_control_ticks = int(math.floor(args.max_sim_time_s / period))
    if maximum_control_ticks < 4:
        raise ValueError("fleet preflight needs at least four control ticks")

    fixture: dict[str, Any] | None = None
    fixture_active_drone_id: str | None = None
    fixture_witness_pose: Pose3D | None = None
    fixture_outbound_route: tuple[tuple[float, float, float], ...] = ()
    fixture_return_route: tuple[tuple[float, float, float], ...] = ()
    fixture_lower_bound_s: float | None = None
    if args.execution_mode == PRIVATE_WITNESS_FIXTURE_MODE:
        fixture = _select_private_fixture(private_episode, public_episode)
        witness = fixture["witness"]
        proof = witness["reachability_proof"]
        fixture_active_drone_id = str(proof["start_drone_id"])
        fixture_witness_pose = Pose3D.from_dict(witness["pose"])
        fixture_start = tuple(float(value) for value in fixture["start"]["position"])
        fixture_outbound_route = _safe_sky_route(
            fixture_start,
            fixture_witness_pose.position,
            float(proof["transit_altitude_m"]),
        )
        fixture_return_route = _safe_sky_route(
            fixture_witness_pose.position,
            fixture_start,
            float(proof["transit_altitude_m"]),
        )
        fixture_lower_bound_s = sum(
            math.hypot(right[0] - left[0], right[1] - left[1]) / guidance.horizontal_speed_mps
            + abs(right[2] - left[2]) / guidance.vertical_speed_mps
            for route in (fixture_outbound_route, fixture_return_route)
            for left, right in zip((fixture_start, *route[:-1]), route, strict=True)
        ) + float(execution["observe"]["continuous_dwell_s"])
        if args.max_sim_time_s <= fixture_lower_bound_s:
            raise ValueError(
                "private-witness fixture budget is no greater than its route lower bound "
                f"({fixture_lower_bound_s:.6f}s) before settling overhead"
            )

    asset = verify_local_cf2x_asset(args.cf2x_usd)
    asset_structure = inspect_verified_cf2x_structure(asset)
    spec = project_asset_spec()
    controller = candidate_controller_spec()
    colliders = colliders_from_city(city)
    input_bindings = {
        "layout_id": str(city["layout_id"]),
        "layout_hash": str(city["layout_hash"]),
        "episode_id": str(public_episode["episode_id"]),
        "task_track": str(task_spec.get("task_track", "G1-U")),
        "stage_sha256": file_hash(stage_path),
        "cityspec_sha256": file_hash(city_path),
        "task_spec_sha256": file_hash(task_path),
        "task_spec_hash": str(task_spec["task_spec_hash"]),
        "public_episode_sha256": file_hash(public_episode_path),
        "execution_contract_hash": content_hash(task_spec["execution_contract"]),
        "release_config_sha256": file_hash(config_path),
        "cf2x_usd_sha256": asset.usd_sha256,
        "cf2x_schema_sha256": asset.schema_sha256,
        "dynamics_spec_hash": content_hash(spec.fingerprint_payload()),
        "controller_spec_hash": content_hash(
            {
                "controller": controller.fingerprint_payload(),
                "guidance_profile": guidance_contract,
            }
        ),
        "baseline_source_sha256": file_hash(BENCH_ROOT / "src" / "aerocity_bench" / "baselines.py"),
        "geometry_source_sha256": file_hash(BENCH_ROOT / "src" / "aerocity_bench" / "geometry.py"),
    }
    atlas = task_spec.get("inspection_atlas")
    projection = task_spec.get("inspection_atlas_projection")
    if isinstance(atlas, dict):
        input_bindings["atlas_hash"] = str(atlas["atlas_hash"])
        input_bindings["inspection_prior_level"] = "full-cells"
    elif isinstance(projection, dict):
        input_bindings["atlas_hash"] = str(projection["source_atlas_hash"])
        input_bindings["inspection_prior_level"] = str(projection["prior_level"])
    if "mission_sector_hash" in public_episode:
        input_bindings["mission_sector_hash"] = str(public_episode["mission_sector_hash"])
    private_evaluator_commitment = content_hash(
        {
            "private_episode_sha256": file_hash(private_episode_path),
            "layout_hash": str(private_episode["layout_hash"]),
            "execution_contract_hash": input_bindings["execution_contract_hash"],
        }
    )
    private_fixture_commitment = (
        content_hash(
            {
                "target_id": fixture["target"]["target_id"],
                "witness_id": fixture["witness"]["witness_id"],
                "start_drone_id": fixture_active_drone_id,
                "private_episode_sha256": file_hash(private_episode_path),
                "execution_contract_hash": input_bindings["execution_contract_hash"],
            }
        )
        if fixture is not None
        else None
    )

    _write_progress(output_path, "opening_existing_city_stage")
    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError("Isaac USD context rejected the existing CitySpec stage")
    from omni.kit.app import get_app

    for _ in range(30):
        get_app().update()
    if context.get_stage() is None:
        raise RuntimeError("Isaac did not expose the opened CitySpec stage")
    sim = SimulationContext(
        SimulationCfg(
            dt=physical_dt,
            device=args.device,
            create_stage_in_memory=False,
            physx=sim_utils.PhysxCfg(
                # Isaac Sim 5.1 removed the pre-5.x
                # ``enable_external_forces_every_iteration`` field.  The
                # wrench application path is still advanced at every PhysX
                # step; keep only fields present in the frozen runtime API.
                min_velocity_iteration_count=1,
            ),
        )
    )
    # ``UsdFileCfg`` may create a leaf prim but does not reliably author an
    # absent intermediate Xform in an already-open CitySpec stage.  Create the
    # shared parent once, before any CF2X exists; each child remains an
    # independently named articulation below this one common PhysX world.
    from pxr import UsdGeom

    fleet_root = "/World/AeroCityFleetPreflight"
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac stage disappeared before CF2X fleet construction")
    if stage.GetPrimAtPath(fleet_root).IsValid():
        raise RuntimeError("CF2X fleet root already exists; refusing stale shared-world state")
    UsdGeom.Xform.Define(stage, fleet_root)
    get_app().update()
    robots: dict[str, Any] = {}
    sensors: dict[str, Any] = {}
    for member in members:
        yaw_rad = math.radians(member.start_yaw_deg)
        robot_cfg = build_cf2x_multirotor_cfg(
            asset,
            spec,
            dt_s=physical_dt,
            prim_path=member.prim_path,
            position_w_m=member.start_position_w_m,
            orientation_wxyz=(math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)),
        )
        robots[member.drone_id] = Multirotor(robot_cfg)
        sensors[member.drone_id] = ContactSensor(
            ContactSensorCfg(
                prim_path=f"{robot_cfg.prim_path}/body",
                update_period=0.0,
                history_length=1,
            )
        )
    _write_progress(output_path, "four_articulations_and_contact_sensors_constructed")
    sim.reset()
    for drone_id in sorted(robots):
        robot = robots[drone_id]
        sensor = sensors[drone_id]
        robot.update(physical_dt)
        sensor.update(physical_dt)
        body_ids, _ = robot.find_bodies("body", preserve_order=True)
        if len(body_ids) != 1 or tuple(robot.data.thruster_names) != (
            "m1_prop",
            "m2_prop",
            "m3_prop",
            "m4_prop",
        ):
            raise RuntimeError(f"unexpected CF2X multirotor layout for {drone_id}")
        default_root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(default_root_state[:, :7])
        robot.write_root_velocity_to_sim(default_root_state[:, 7:])
        robot.reset()
        sensor.reset()
        robot.update(physical_dt)
        sensor.update(physical_dt)

    body_masses: dict[str, list[float]] = {}
    mass_by_drone: dict[str, float] = {}
    reference_by_drone: dict[str, tuple[float, float, float, float]] = {}
    applied_by_drone: dict[str, tuple[float, float, float, float]] = {}
    wrench_by_drone: dict[str, tuple[float, float, float, float]] = {}
    for drone_id, robot in sorted(robots.items()):
        masses, total_mass = read_verified_cf2x_runtime_mass_kg(
            robot, expected_total_mass_kg=spec.mass_kg
        )
        body_masses[drone_id] = list(masses)
        mass_by_drone[drone_id] = total_mass
        hover = hover_rotor_speed_for_mass(spec, total_mass)
        reference_by_drone[drone_id] = (hover, hover, hover, hover)
        applied = tuple(float(value) for value in _finite_list(robot.data.applied_thrust[0]))
        applied_by_drone[drone_id] = applied  # type: ignore[assignment]
        wrench_by_drone[drone_id] = rotor_thrust_wrench(spec, applied)

    evaluator = PrivateEvaluator(config, city, private_episode, receipt_secret=os.urandom(32))
    # The public policy sees public task material and G1 observations only.  The
    # private episode remains evaluator-owned and is never passed to a baseline
    # or an external planner process.
    policy = (
        create_baseline(args.method, config, task_spec, public_episode)
        if args.execution_mode == "public-policy"
        else None
    )
    external_bridge: ExternalProcessPlannerBridge | None = None
    external_bridge_for_reporting: ExternalProcessPlannerBridge | None = None
    if external_manifest is not None:
        external_bridge = ExternalProcessPlannerBridge(
            external_manifest.declaration,
            external_manifest.launch_command(),
            cwd=BENCH_ROOT,
            response_timeout_s=float(execution["planning_deadline_s"]),
            initialization_timeout_s=EXTERNAL_PROCESS_INITIALIZATION_DEADLINE_S,
            maximum_line_bytes=2_000_000,
        )
        args._owned_external_planner_bridge = external_bridge
        external_bridge_for_reporting = external_bridge
        external_bridge.reset(public_episode, public_task_spec=task_spec)
    active_method = (
        external_manifest.declaration.method_id
        if external_manifest is not None
        else (
            args.method
            if policy is not None
            else (
                "internal-private-witness-fixture"
                if fixture is not None
                else "internal-shared-hold"
            )
        )
    )
    if policy is not None:
        route_budget_audit = policy.route_budget_audit(
            horizontal_speed_mps=guidance.horizontal_speed_mps,
            vertical_speed_mps=guidance.vertical_speed_mps,
        )
    elif external_manifest is not None:
        route_budget_audit = {
            "schema": "org.aerocity.bench.baseline-route-budget-audit.v1",
            "status": "NOT_APPLICABLE",
            "reason": (
                "external-process-policy owns its public route choice; "
                "the shared executor records measured deadline, safety, and return outcomes"
            ),
        }
    else:
        route_budget_audit = {
            "schema": "org.aerocity.bench.baseline-route-budget-audit.v1",
            "status": "NOT_APPLICABLE",
            "reason": (
                "shared-hold does not execute a public search route"
                if args.execution_mode == "shared-hold"
                else (
                    "private-witness-fixture uses an evaluator-owned internal route; "
                    "it is not a public search method"
                )
            ),
        }
    if route_budget_audit["status"] == "BUDGET_INFEASIBLE":
        raise RuntimeError(
            "public route became budget-infeasible after the CPU-only preflight; refusing Isaac run"
        )
    home_by_drone = {member.drone_id: member.start_position_w_m for member in members}
    action_sequence = 0
    task_time_s = 0.0
    energy_used_j = {member.drone_id: 0.0 for member in members}
    previous_receipt_hash = {member.drone_id: None for member in members}
    state_by_drone: dict[str, dict[str, Any]] = {}
    sensor_pitch_by_drone: dict[str, float] = {}
    altitude_samples = {member.drone_id: [] for member in members}
    execution_receipts: list[dict[str, Any]] = []
    # This private evidence contains the exact public packets passed through the
    # adapter.  It lets the CPU-only verifier recompute receipt bindings without
    # retaining evaluator-private targets or witnesses.
    execution_bindings_public: list[dict[str, Any]] = []
    observation_receipts: list[dict[str, Any]] = []
    confirmation_receipts: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    measured_state_trace: list[dict[str, Any]] = []
    action_kind_counts: dict[str, int] = {}
    observation_build_latency_s: list[float] = []
    policy_latency_s: list[float] = []
    # Scalar timing attribution only. The raw packets remain in the existing
    # protected execution binding trace and are never copied into this trace.
    external_planning_timing_trace: list[dict[str, float | int | None]] = []
    collision_detected = False
    out_of_bounds_detected = False
    safe_completion = True
    yaw_alignment_hold_physics_steps = {member.drone_id: 0 for member in members}
    yaw_alignment_active = {member.drone_id: False for member in members}
    shared_steps = SharedWorldStepLedger(members)
    fixture_phase = "outbound" if fixture is not None else "not_applicable"
    fixture_outbound_index = 0
    fixture_return_index = 0
    fixture_closed = False
    external_adapter_failures = 0
    planning_cadence = (
        PlanningCadenceController.from_execution_contract(execution)
        if policy is not None or external_bridge is not None
        else None
    )
    planner_invoked_by_tick: list[bool] = []
    planning_trigger_counts: dict[str, int] = {}
    return_reserve_event_emitted = False

    def contact_force(drone_id: str) -> float:
        sensor = sensors[drone_id]
        sensor.update(physical_dt)
        values = _finite_list(sensor.data.net_forces_w)
        if len(values) % 3:
            raise RuntimeError("CF2X ContactSensor returned a malformed force tensor")
        return max(
            (_norm(values[index : index + 3]) for index in range(0, len(values), 3)),
            default=0.0,
        )

    def refresh_states() -> None:
        for drone_id, robot in robots.items():
            state_by_drone[drone_id] = _state(
                robot,
                rotor_reference_rad_s=reference_by_drone[drone_id],
                applied_rotor_thrust_n=applied_by_drone[drone_id],
                body_wrench=wrench_by_drone[drone_id],
                contact_force_n=contact_force(drone_id),
            )

    def observations() -> dict[str, Any]:
        packets: dict[str, Any] = {}
        communication_range = float(execution["communication"]["range_m"])
        for member in members:
            drone_id = member.drone_id
            state = state_by_drone[drone_id]
            position = tuple(float(value) for value in state["position"])
            origin, resolution, radius, occupied = _local_occupancy(position, colliders)
            teammates = tuple(
                {
                    "drone_id": other.drone_id,
                    "position": list(state_by_drone[other.drone_id]["position"]),
                    "health": "nominal",
                }
                for other in members
                if other.drone_id != drone_id
                and _distance(position, tuple(state_by_drone[other.drone_id]["position"]))
                <= communication_range
            )
            packets[drone_id] = ObservationPacket(
                episode_id=str(public_episode["episode_id"]),
                observation_id=f"fleet-observation-{action_sequence:05d}-{drone_id}",
                drone_id=drone_id,
                sequence=action_sequence,
                timestamp_s=task_time_s,
                pose=_pose_from_state(state),
                linear_velocity_world_mps=tuple(state["linear_velocity_mps"]),
                angular_speed_deg_s=math.degrees(_norm(state["angular_velocity_rad_s"])),
                energy_remaining_j=max(
                    0.0, float(vehicle["energy_budget_j"]) - energy_used_j[drone_id]
                ),
                local_occupancy=occupied,
                local_occupancy_origin_world_m=origin,
                local_occupancy_resolution_m=resolution,
                local_occupancy_radius_m=radius,
                teammate_states=teammates,
                sensor_pitch_deg=sensor_pitch_by_drone[drone_id],
            )
        return packets

    refresh_states()
    for member in members:
        initial = state_by_drone[member.drone_id]
        initial_body_pitch_deg = math.degrees(_euler_from_wxyz(initial["orientation_wxyz"])[1])
        sensor_pitch_by_drone[member.drone_id] = initial_body_pitch_deg
        altitude_samples[member.drone_id].append(
            {
                "task_time_s": 0.0,
                "position_w_m": initial["position"],
                "linear_velocity_w_mps": initial["linear_velocity_mps"],
            }
        )
    measurement_evidence = L1MeasurementEvidence(
        city=city,
        task_spec=task_spec,
        public_episode=public_episode,
    )
    _write_progress(output_path, "beginning_shared_physx_candidate_run")
    started = time.perf_counter()
    last_sample_time_s = 0.0
    for action_sequence in range(maximum_control_ticks):
        if fixture_phase == "closed":
            break
        observation_started = time.perf_counter()
        current_observations = observations()
        observation_build_latency_s.append(time.perf_counter() - observation_started)
        if planning_cadence is not None:
            return_reserve_start_s = float(execution["episode"]["duration_s"]) - float(
                execution["episode"]["return_reserve_s"]
            )
            if not return_reserve_event_emitted and task_time_s >= return_reserve_start_s:
                planning_cadence.request_event("return_reserve_entry")
                return_reserve_event_emitted = True
            planning_due_reasons = planning_cadence.due_reasons(
                control_tick=action_sequence,
                active_drone_ids=tuple(current_observations),
            )
        else:
            planning_due_reasons = ()
        planner_invoked = bool(planning_due_reasons)
        planner_invoked_by_tick.append(planner_invoked)
        for reason in planning_due_reasons:
            planning_trigger_counts[reason] = planning_trigger_counts.get(reason, 0) + 1
        planning_started = time.perf_counter()
        bridge_wall_clock_s: float | None = None
        bridge_timing: dict[str, float] | None = None
        arbitration_wall_clock_s: float | None = None
        planning_attempt_succeeded = True
        if policy is not None and planner_invoked:
            actions = policy(current_observations)
        elif external_bridge is not None and planner_invoked:
            try:
                bridge_started = time.perf_counter()
                try:
                    actions, _ = external_bridge.act(current_observations)
                    bridge_timing = external_bridge.last_act_timing()
                    if bridge_timing is None:
                        raise RuntimeError("external planner bridge omitted action timing")
                finally:
                    bridge_wall_clock_s = time.perf_counter() - bridge_started
                arbitration_started = time.perf_counter()
                try:
                    actions = arbitrate_public_fleet_actions(
                        actions,
                        current_observations,
                        vehicle_radius_m=float(vehicle["radius_m"]),
                    )
                finally:
                    arbitration_wall_clock_s = time.perf_counter() - arbitration_started
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                planning_attempt_succeeded = False
                external_adapter_failures += 1
                failure_records.append(
                    FailureRecord(
                        episode_id=str(public_episode["episode_id"]),
                        drone_id="fleet",
                        task_time_s=task_time_s,
                        category="external_adapter_failure",
                        detail=f"{type(exc).__name__}: {exc}",
                        terminal=False,
                    ).to_dict()
                )
                external_bridge.close()
                external_bridge = None
                actions = {
                    member.drone_id: ActionPacket(
                        episode_id=str(public_episode["episode_id"]),
                        drone_id=member.drone_id,
                        sequence=action_sequence,
                        issued_at_s=task_time_s,
                        kind="HOVER",
                    )
                    for member in members
                }
        elif planning_cadence is not None and not planner_invoked:
            actions = planning_cadence.held_actions(current_observations)
        elif fixture is not None:
            # All four vehicles remain in the same PhysX world.  Only this
            # internal evaluator-owned fixture may consult a private witness;
            # the three non-selected vehicles hold their public start poses.
            actions = {
                member.drone_id: ActionPacket(
                    episode_id=str(public_episode["episode_id"]),
                    drone_id=member.drone_id,
                    sequence=action_sequence,
                    issued_at_s=task_time_s,
                    kind="HOVER",
                )
                for member in members
            }
            if fixture_active_drone_id is None or fixture_witness_pose is None:
                raise RuntimeError("private-witness fixture lost its active drone or witness pose")
            active_state = state_by_drone[fixture_active_drone_id]
            active_position = tuple(float(value) for value in active_state["position"])
            active_speed = _norm(active_state["linear_velocity_mps"])
            active_yaw = _euler_from_wxyz(active_state["orientation_wxyz"])[2]
            active_observation = current_observations[fixture_active_drone_id]
            active_action: ActionPacket
            if fixture_phase == "outbound":
                goal = fixture_outbound_route[fixture_outbound_index]
                tolerance = 0.30 if fixture_outbound_index < 2 else 0.20
                if _distance(active_position, goal) <= tolerance and active_speed <= 0.30:
                    if fixture_outbound_index < 2:
                        fixture_outbound_index += 1
                    else:
                        fixture_phase = "align"
                desired_yaw = three_leg_sky_route_waypoint_yaw(
                    fixture_outbound_route,
                    fixture_outbound_index,
                    terminal_yaw_rad=math.radians(fixture_witness_pose.yaw_deg),
                )
                active_action = ActionPacket(
                    episode_id=active_observation.episode_id,
                    drone_id=fixture_active_drone_id,
                    sequence=action_sequence,
                    issued_at_s=task_time_s,
                    kind="WAYPOINT",
                    waypoint=Pose3D(
                        position=goal,
                        yaw_deg=math.degrees(desired_yaw),
                    ),
                )
            elif fixture_phase == "align":
                yaw_error = abs(
                    _wrap_angle_rad(active_yaw - math.radians(fixture_witness_pose.yaw_deg))
                )
                if (
                    _distance(active_position, fixture_witness_pose.position) <= 0.20
                    and active_speed <= float(execution["observe"]["max_linear_speed_mps"])
                    and math.degrees(_norm(active_state["angular_velocity_rad_s"]))
                    <= float(execution["observe"]["max_angular_speed_deg_s"])
                    and yaw_error <= math.radians(5.0)
                ):
                    fixture_phase = "observe"
                active_action = ActionPacket(
                    episode_id=active_observation.episode_id,
                    drone_id=fixture_active_drone_id,
                    sequence=action_sequence,
                    issued_at_s=task_time_s,
                    kind="WAYPOINT",
                    waypoint=fixture_witness_pose,
                )
            elif fixture_phase == "observe":
                active_action = ActionPacket(
                    episode_id=active_observation.episode_id,
                    drone_id=fixture_active_drone_id,
                    sequence=action_sequence,
                    issued_at_s=task_time_s,
                    kind="OBSERVE",
                    source_observation_id=active_observation.observation_id,
                )
            elif fixture_phase == "return":
                goal = fixture_return_route[fixture_return_index]
                if _distance(active_position, goal) <= 0.30 and active_speed <= 0.30:
                    if fixture_return_index < 2:
                        fixture_return_index += 1
                    else:
                        fixture_closed = True
                        fixture_phase = "closed"
                desired_yaw = three_leg_sky_route_waypoint_yaw(
                    fixture_return_route, fixture_return_index
                )
                active_action = ActionPacket(
                    episode_id=active_observation.episode_id,
                    drone_id=fixture_active_drone_id,
                    sequence=action_sequence,
                    issued_at_s=task_time_s,
                    kind="RETURN",
                    waypoint=Pose3D(position=goal, yaw_deg=math.degrees(desired_yaw)),
                )
            else:
                raise RuntimeError(f"unknown private-witness fixture phase: {fixture_phase}")
            actions[fixture_active_drone_id] = active_action
        else:
            actions = {
                member.drone_id: ActionPacket(
                    episode_id=str(public_episode["episode_id"]),
                    drone_id=member.drone_id,
                    sequence=action_sequence,
                    issued_at_s=task_time_s,
                    kind="HOVER",
                )
                for member in members
            }
        planning_latency_s = (
            time.perf_counter() - planning_started if planner_invoked else 0.0
        )
        deadline_miss = planner_invoked and planning_latency_s > float(
            execution["planning_deadline_s"]
        )
        if planning_cadence is not None and planner_invoked:
            if planning_attempt_succeeded and not deadline_miss:
                planning_cadence.approve(actions)
            else:
                planning_cadence.reject_planning_attempt()
        if planner_invoked:
            policy_latency_s.append(planning_latency_s)
        if external_manifest is not None and planner_invoked:
            external_planning_timing_trace.append(
                {
                    "action_sequence": action_sequence,
                    "total_planner_wall_clock_s": planning_latency_s,
                    "bridge_act_wall_clock_s": (
                        bridge_timing["bridge_act_wall_clock_s"]
                        if bridge_timing is not None
                        else bridge_wall_clock_s
                    ),
                    **(
                        {
                            field: bridge_timing[field]
                            for field in (
                                "projection_wall_clock_s",
                                "request_public_audit_wall_clock_s",
                                "request_json_serialize_wall_clock_s",
                                "request_size_check_wall_clock_s",
                                "request_write_flush_wall_clock_s",
                                "response_wait_wall_clock_s",
                                "response_json_decode_wall_clock_s",
                                "response_validate_wall_clock_s",
                                "action_validation_conversion_wall_clock_s",
                                "bridge_internal_unattributed_wall_clock_s",
                            )
                        }
                        if bridge_timing is not None
                        else {}
                    ),
                    "fleet_arbitration_wall_clock_s": arbitration_wall_clock_s,
                    "unattributed_wall_clock_s": max(
                        0.0,
                        planning_latency_s
                        - (
                            bridge_timing["bridge_act_wall_clock_s"]
                            if bridge_timing is not None
                            else (bridge_wall_clock_s or 0.0)
                        )
                        - (arbitration_wall_clock_s or 0.0),
                    ),
                }
            )
        assert_action_roster_complete(actions, members)
        goals: dict[str, tuple[tuple[float, float, float], float, str, bool]] = {}
        requested_sensor_pitch_by_drone: dict[str, float] = {}
        confirmations_by_drone: dict[str, tuple[str, ...]] = {}
        observation_results: dict[str, dict[str, Any] | None] = {}
        for member in members:
            drone_id = member.drone_id
            action = actions[drone_id]
            observation = current_observations[drone_id]
            if (
                action.episode_id != observation.episode_id
                or action.drone_id != drone_id
                or action.sequence != action_sequence
                or abs(action.issued_at_s - task_time_s) > 1.0e-9
            ):
                raise ValueError(f"public policy action identity mismatch for {drone_id}")
            execution_bindings_public.append(
                {
                    "drone_id": drone_id,
                    "action_sequence": action_sequence,
                    "planner_invoked": planner_invoked,
                    "planning_trigger_reasons": list(planning_due_reasons),
                    "action": action.to_dict(),
                    "source_observation": observation.to_dict(),
                }
            )
            requested_kind = action.kind
            action_kind_counts[requested_kind] = action_kind_counts.get(requested_kind, 0) + 1
            observation_results[drone_id] = None
            confirmations_by_drone[drone_id] = ()
            state = state_by_drone[drone_id]
            current_position = tuple(float(value) for value in state["position"])
            current_yaw = _euler_from_wxyz(state["orientation_wxyz"])[2]
            executed_kind = requested_kind
            safety_intervention = deadline_miss
            current_sensor_pitch = sensor_pitch_by_drone[drone_id]
            requested_sensor_pitch = _requested_sensor_pitch(
                current_sensor_pitch,
                action,
                execution,
                deadline_miss=deadline_miss,
            )
            if deadline_miss:
                executed_kind = "HOVER"
                goal_position, goal_yaw = current_position, current_yaw
            elif action.kind == "WAYPOINT":
                assert action.waypoint is not None
                goal_position = action.waypoint.position
                goal_yaw = math.radians(action.waypoint.yaw_deg)
            elif action.kind in {"HOVER", "OBSERVE"}:
                goal_position, goal_yaw = current_position, current_yaw
            elif action.kind == "RETURN":
                if action.waypoint is None:
                    # A third-party adapter may issue a bare RETURN.  Preserve
                    # its conservative preflight fallback, but record it as an
                    # intervention instead of silently inventing a route.
                    goal_position = home_by_drone[drone_id]
                    goal_yaw = member.start_yaw_deg * math.pi / 180.0
                    safety_intervention = True
                else:
                    goal_position = action.waypoint.position
                    goal_yaw = math.radians(action.waypoint.yaw_deg)
            else:
                # The initial native adapter intentionally does not reinterpret
                # body-frame velocity actions as an extra controller surface.
                executed_kind = "HOVER"
                goal_position, goal_yaw = current_position, current_yaw
                safety_intervention = True
            requested_sensor_pitch_by_drone[drone_id] = requested_sensor_pitch
            if action.kind == "OBSERVE" and not deadline_miss:
                observation_receipt, confirmations = evaluator.process(observation, action)
                observation_result = _observation_receipt_dict(observation_receipt)
                observation_results[drone_id] = observation_result
                observation_receipts.append(observation_result)
                confirmation_ids = tuple(item.confirmation_id for item in confirmations)
                confirmations_by_drone[drone_id] = confirmation_ids
                confirmation_receipts.extend(item.to_dict() for item in confirmations)
                if fixture is not None and drone_id == fixture_active_drone_id and confirmation_ids:
                    fixture_phase = "return"
            else:
                evaluator.end_observe(drone_id, observation.timestamp_s)
            goals[drone_id] = (goal_position, goal_yaw, executed_kind, safety_intervention)

        before_states = {drone_id: dict(state) for drone_id, state in state_by_drone.items()}
        for _ in range(physical_steps_per_control):
            pending: set[str] = set()
            for member in members:
                drone_id = member.drone_id
                robot = robots[drone_id]
                goal_position, goal_yaw, _, _ = goals[drone_id]
                measured = FlightState(
                    position_w_m=tuple(_finite_list(robot.data.root_pos_w[0])),
                    orientation_wxyz=tuple(_finite_list(robot.data.root_quat_w[0])),
                    linear_velocity_w_mps=tuple(_finite_list(robot.data.root_lin_vel_w[0])),
                    angular_velocity_w_rad_s=tuple(_finite_list(robot.data.root_ang_vel_w[0])),
                )
                current_yaw = _euler_from_wxyz(measured.orientation_wxyz)[2]
                guarded_goal_position, yaw_alignment_held = yaw_aligned_translation_goal(
                    measured.position_w_m,
                    goal_position,
                    current_yaw,
                    goal_yaw,
                    measured.angular_velocity_w_rad_s[2],
                    alignment_active=yaw_alignment_active[drone_id],
                    guard=yaw_alignment_guard,
                )
                yaw_alignment_active[drone_id] = yaw_alignment_held
                if yaw_alignment_held:
                    yaw_alignment_hold_physics_steps[drone_id] += 1
                anchor, velocity, yaw = position_anchored_velocity_guidance(
                    measured.position_w_m,
                    guarded_goal_position,
                    goal_yaw,
                    limits=guidance,
                    current_linear_velocity_w_mps=measured.linear_velocity_w_mps,
                    vertical_guard=vertical_guard,
                )
                output = controller_step(
                    spec,
                    controller,
                    measured,
                    FlightCommand(
                        target_position_w_m=anchor,
                        target_velocity_w_mps=velocity,
                        target_yaw_rad=yaw,
                    ),
                    mass_kg=mass_by_drone[drone_id],
                )
                reference_by_drone[drone_id] = output.rotor_references_rad_s
                maximum_thrust = spec.thrust_coeff_n_per_rad2 * spec.max_rotor_speed_rad_s**2
                thrust = tuple(
                    min(
                        maximum_thrust,
                        max(0.0, spec.thrust_coeff_n_per_rad2 * speed * speed),
                    )
                    for speed in output.rotor_references_rad_s
                )
                import torch

                robot.set_thrust_target(
                    torch.tensor([thrust], dtype=torch.float32, device=robot.device)
                )
                robot.write_data_to_sim()
                pending.add(drone_id)
            shared_steps.record_step(pending)
            sim.step(render=False)
            for drone_id, robot in robots.items():
                robot.update(physical_dt)
                sensors[drone_id].update(physical_dt)
                applied = tuple(
                    float(value) for value in _finite_list(robot.data.applied_thrust[0])
                )
                applied_by_drone[drone_id] = applied  # type: ignore[assignment]
                wrench_by_drone[drone_id] = rotor_thrust_wrench(spec, applied)

        refresh_states()
        task_time_end_s = task_time_s + period
        pair_violations: set[str] = set()
        separation_floor = 2.0 * float(vehicle["radius_m"])
        for left_index, left in enumerate(members):
            left_position = tuple(state_by_drone[left.drone_id]["position"])
            for right in members[left_index + 1 :]:
                right_position = tuple(state_by_drone[right.drone_id]["position"])
                if _distance(left_position, right_position) < separation_floor:
                    pair_violations.update((left.drone_id, right.drone_id))
        tick_failed = False
        collision_by_drone: dict[str, bool] = {}
        out_of_bounds_by_drone: dict[str, bool] = {}
        safety_intervention_by_drone: dict[str, bool] = {}
        for member in members:
            drone_id = member.drone_id
            action = actions[drone_id]
            before = before_states[drone_id]
            after = state_by_drone[drone_id]
            after_position = tuple(float(value) for value in after["position"])
            contact_collision = float(after["contact_force_n"]) >= float(args.contact_threshold_n)
            pair_collision = drone_id in pair_violations
            collision = contact_collision or pair_collision
            out_of_bounds = not _within_bounds(
                after_position, task_spec["flight_bounds"], float(vehicle["radius_m"])
            )
            center_clearance, _ = minimum_clearance(after_position, colliders)
            body_clearance = max(0.0, center_clearance - float(vehicle["radius_m"]))
            moved = _distance(tuple(before["position"]), after_position)
            action_energy = moved * float(vehicle["energy_per_meter_j"]) + period * float(
                vehicle["hover_power_w"]
            )
            energy_used_j[drone_id] += action_energy
            _, _, executed_kind, safety_intervention = goals[drone_id]
            collision_by_drone[drone_id] = collision
            out_of_bounds_by_drone[drone_id] = out_of_bounds
            safety_intervention_by_drone[drone_id] = safety_intervention
            status = "measured_physx_executed"
            if deadline_miss:
                status = "planning_deadline_hover"
            elif action.kind == "VELOCITY":
                status = "unsupported_velocity_hover"
            elif action.kind == "RETURN":
                status = (
                    "candidate_return_waypoint"
                    if action.waypoint is not None
                    else "candidate_direct_home_return"
                )
            if collision:
                status = "native_contact_or_agent_separation_collision"
            elif out_of_bounds:
                status = "native_out_of_bounds"
            receipt = build_l1_execution_receipt(
                action=action,
                source_observation=current_observations[drone_id],
                state_before=before,
                state_after=after,
                task_time_start_s=task_time_s,
                task_time_end_s=task_time_end_s,
                planning_latency_s=planning_latency_s,
                action_executed=executed_kind,
                status=status,
                energy_used_j=action_energy,
                minimum_clearance_m=body_clearance,
                collision=collision,
                out_of_bounds=out_of_bounds,
                safety_intervention=safety_intervention,
                deadline_miss=deadline_miss,
                previous_receipt_hash=previous_receipt_hash[drone_id],
                confirmation_ids=confirmations_by_drone[drone_id],
                planner_invoked=planner_invoked,
            ).to_dict()
            execution_receipts.append(receipt)
            previous_receipt_hash[drone_id] = str(receipt["receipt_hash"])
            sensor_pitch_by_drone[drone_id] = requested_sensor_pitch_by_drone[drone_id]
            collision_detected = collision_detected or collision
            out_of_bounds_detected = out_of_bounds_detected or out_of_bounds
            if collision or out_of_bounds:
                tick_failed = True
                safe_completion = False
                failure_records.append(
                    FailureRecord(
                        episode_id=action.episode_id,
                        drone_id=drone_id,
                        task_time_s=task_time_end_s,
                        category="collision" if collision else "out_of_bounds_failure",
                        detail=(
                            f"contact_force_n={after['contact_force_n']:.6f}; "
                            f"agent_separation_violation={pair_collision}"
                            if collision
                            else "measured root position left flight bounds"
                        ),
                        terminal=True,
                    ).to_dict()
                )
        for member in members:
            drone_id = member.drone_id
            action = actions[drone_id]
            if action.kind == "OBSERVE" and not deadline_miss:
                observation_result = observation_results[drone_id]
                if observation_result is None:
                    raise RuntimeError("L1 OBSERVE action lacks an evaluator receipt")
                measurement_evidence.record_observe(
                    current_observations[drone_id],
                    evaluator_accepted=bool(observation_result["accepted"]),
                    runtime_safe=(
                        not collision_by_drone[drone_id]
                        and not out_of_bounds_by_drone[drone_id]
                        and not safety_intervention_by_drone[drone_id]
                    ),
                )
            else:
                measurement_evidence.end_observe(drone_id)
        positions_by_drone = {
            member.drone_id: tuple(
                float(value) for value in state_by_drone[member.drone_id]["position"]
            )
            for member in members
        }
        safe_drone_ids = {
            drone_id
            for drone_id in positions_by_drone
            if not collision_by_drone[drone_id]
            and not out_of_bounds_by_drone[drone_id]
            and not safety_intervention_by_drone[drone_id]
        }
        task_time_end_for_evidence = task_time_s + period
        measurement_evidence.record_measured_positions(
            task_time_end_for_evidence,
            positions_by_drone,
            safe_drone_ids=safe_drone_ids,
        )
        measured_state_trace.append(
            {
                "action_sequence": action_sequence,
                "task_time_s": round(task_time_end_for_evidence, 9),
                "positions_w_m": {
                    drone_id: list(position)
                    for drone_id, position in sorted(positions_by_drone.items())
                },
                "safe_drone_ids": sorted(safe_drone_ids),
            }
        )
        task_time_s = task_time_end_s
        if planning_cadence is not None:
            if any(confirmations_by_drone.values()):
                planning_cadence.request_event("anonymous_confirmation")
            if deadline_miss or any(safety_intervention_by_drone.values()):
                planning_cadence.request_event("safety_intervention")
        if task_time_s - last_sample_time_s + 1.0e-9 >= args.stability_sample_period_s:
            for member in members:
                state = state_by_drone[member.drone_id]
                altitude_samples[member.drone_id].append(
                    {
                        "task_time_s": task_time_s,
                        "position_w_m": state["position"],
                        "linear_velocity_w_mps": state["linear_velocity_mps"],
                    }
                )
            last_sample_time_s = task_time_s
        if tick_failed:
            break

    if task_time_s - last_sample_time_s > 1.0e-9:
        for member in members:
            state = state_by_drone[member.drone_id]
            altitude_samples[member.drone_id].append(
                {
                    "task_time_s": task_time_s,
                    "position_w_m": state["position"],
                    "linear_velocity_w_mps": state["linear_velocity_mps"],
                }
            )
    stability = {
        member.drone_id: altitude_stability_metrics(altitude_samples[member.drone_id])
        for member in members
    }
    candidate_shared_hold = (
        candidate_shared_hold_assessment(stability)
        if args.execution_mode == "shared-hold"
        else {
            "status": "NOT_APPLICABLE",
            "candidate_preflight_only": True,
            "reason": (
                "public-policy is maneuver evidence, not a shared-hold stability test"
                if args.execution_mode == "public-policy"
                else (
                    "external-process-policy is maneuver evidence, not a shared-hold stability test"
                    if args.execution_mode == EXTERNAL_PROCESS_POLICY_MODE
                    else (
                        "private-witness-fixture is closure evidence, "
                        "not a shared-hold stability test"
                    )
                )
            ),
        }
    )
    returned_home = {
        member.drone_id: _distance(
            tuple(state_by_drone[member.drone_id]["position"]), member.start_position_w_m
        )
        <= float(vehicle["home_radius_m"])
        for member in members
    }
    observe_action_count = action_kind_counts.get("OBSERVE", 0)
    deadline_miss_tick_count = sum(
        bool(receipt["deadline_miss"])
        for receipt in execution_receipts[:: len(members)]
    )

    planning_timing = {
        "schema": "org.aerocity.bench.fleet-preflight-timing.v4",
        "control_tick_count": len(observation_build_latency_s),
        "control_period_s": period,
        "planner_invocation_count": sum(planner_invoked_by_tick),
        "held_action_tick_count": len(planner_invoked_by_tick) - sum(planner_invoked_by_tick),
        "planning_cadence": dict(execution["planning"]),
        "planning_trigger_counts": dict(sorted(planning_trigger_counts.items())),
        "planning_deadline_s": float(execution["planning_deadline_s"]),
        "deadline_miss_tick_count": deadline_miss_tick_count,
        # The deadline is deliberately based on this total, not any substage.
        "policy_call": _substage_latency_summary(policy_latency_s),
        "public_observation_build": _latency_summary(observation_build_latency_s),
        "external_process_substages": (
            _external_process_substage_summary(external_planning_timing_trace)
            if external_manifest is not None
            else None
        ),
    }
    private_fixture_closed = (
        fixture is not None
        and fixture_closed
        and len(confirmation_receipts) > 0
        and action_kind_counts.get("RETURN", 0) > 0
        and all(returned_home.values())
    )
    episode_budget_completed = (
        abs(task_time_s - float(execution["episode"]["duration_s"])) <= 1.0e-9
    )
    if args.execution_mode == "shared-hold":
        progress_status = "NOT_APPLICABLE"
    elif args.execution_mode == PRIVATE_WITNESS_FIXTURE_MODE:
        progress_status = (
            "PRIVATE_FIXTURE_CLOSED" if private_fixture_closed else "PRIVATE_FIXTURE_INCOMPLETE"
        )
    else:
        progress_status = public_policy_progress_status(
            purpose=str(args.run_purpose),
            observe_action_count=observe_action_count,
            confirmation_receipt_count=len(confirmation_receipts),
            return_action_count=action_kind_counts.get("RETURN", 0),
            all_returned_home=all(returned_home.values()),
            episode_budget_completed=episode_budget_completed,
            safe_completion=safe_completion,
            deadline_miss_tick_count=deadline_miss_tick_count,
            adapter_failure_count=external_adapter_failures,
        )
    policy_progress = {
        "status": progress_status,
        "observe_action_count": observe_action_count,
        "confirmation_receipt_count": len(confirmation_receipts),
        "return_action_count": action_kind_counts.get("RETURN", 0),
        "all_returned_home": all(returned_home.values()),
        "episode_budget_completed": episode_budget_completed,
    }
    external_adapter = None
    if external_manifest is not None:
        external_adapter = {
            **external_manifest.public_provenance(),
            "adapter_tax": external_bridge_for_reporting.adapter_tax_report()
            if external_bridge_for_reporting is not None
            else None,
            "initialization": external_bridge_for_reporting.initialization_report()
            if external_bridge_for_reporting is not None
            else None,
            "failure_count": external_adapter_failures,
        }
    private_report: dict[str, Any] = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-private.v4",
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRIVATE_SCOPE,
        "input_bindings": input_bindings,
        "execution_mode": args.execution_mode,
        "execution_purpose": args.run_purpose,
        "method": active_method,
        "fleet_members_private": [
            {
                "drone_id": member.drone_id,
                "start_position_w_m": list(member.start_position_w_m),
                "start_yaw_deg": member.start_yaw_deg,
                "prim_path": member.prim_path,
            }
            for member in members
        ],
        "private_evaluator_commitment": private_evaluator_commitment,
        "vehicle_public": {
            "execution_model": (
                "cf2x_multirotor_per_rotor_thrust_geometry_allocated_root_wrench_physx"
            ),
            "asset_sha256": asset.usd_sha256,
            "schema_sha256": asset.schema_sha256,
            "asset_structure": asset_structure,
            "asset_provenance_status": spec.provenance_status,
            "controller_provenance_status": controller.provenance_status,
            "physical_dt_s": physical_dt,
            "control_period_s": period,
            "physical_steps_per_control": physical_steps_per_control,
            "contact_evidence": "IsaacLab ContactSensor.net_forces_w",
            "contact_threshold_n": float(args.contact_threshold_n),
            "direct_root_state_writes_during_loop": False,
            "rotor_joint_velocity_targets_written_during_loop": False,
            "applied_thrust_evidence": "Multirotor.data.applied_thrust",
            "wrench_application_model": "derived_geometry_allocation_to_root_body_physx",
            "prop_link_forces_applied_directly": False,
            "guidance_profile": {
                **guidance_contract,
                "yaw_alignment_hold_physics_steps_by_drone": dict(
                    sorted(yaw_alignment_hold_physics_steps.items())
                ),
                "status": "candidate_preflight_profile_not_formal_execution_envelope",
            },
            "physx_body_masses_kg": body_masses,
        },
        "execution": {
            "control_ticks": len(execution_receipts) // len(members),
            "control_period_s": period,
            "shared_physx_step_count": shared_steps.shared_physx_step_count,
            "physical_steps_per_control": physical_steps_per_control,
            "simulated_time_s": task_time_s,
            "wall_clock_s": time.perf_counter() - started,
            "action_kind_counts": dict(sorted(action_kind_counts.items())),
            "energy_used_j_estimate_by_drone": dict(sorted(energy_used_j.items())),
        },
        "observation_receipts": observation_receipts,
        "confirmation_receipts": confirmation_receipts,
        "execution_receipts": execution_receipts,
        "execution_bindings_public": execution_bindings_public,
        "measured_state_trace_private": measured_state_trace,
        "measurement_evidence": measurement_evidence.snapshot(
            measured_state_trace=measured_state_trace,
            input_bindings_hash=content_hash(input_bindings),
        ),
        "external_planning_timing_trace": (
            external_planning_timing_trace if external_manifest is not None else None
        ),
        "failure_records": failure_records,
        "evaluator_private_audit": evaluator.private_audit_snapshot(),
        "altitude_samples_private": altitude_samples,
        "flight_stability_public": stability,
        "candidate_shared_hold": candidate_shared_hold,
        "route_budget_audit": route_budget_audit,
        "planning_timing": planning_timing,
        "policy_progress": policy_progress,
        "external_adapter": external_adapter,
        "final": {
            "safe_completion": safe_completion,
            "collision_detected": collision_detected,
            "out_of_bounds_detected": out_of_bounds_detected,
            "all_returned_home": all(returned_home.values()),
            "returned_home_by_drone": returned_home,
            "states_private": state_by_drone,
        },
    }
    if fixture is not None:
        private_report["private_fixture_commitment"] = private_fixture_commitment
        private_report["private_fixture_execution"] = {
            "candidate_round_trip_lower_bound_s": fixture_lower_bound_s,
            "closure_route_finished": fixture_closed,
        }
    private_report["private_report_content_sha256"] = content_hash(private_report)
    _write_large_private_json_atomic(private_output_path, private_report)
    private_report["private_report_file_sha256"] = file_hash(private_output_path)
    public_report = _public_summary(private_report)
    public_report["public_report_sha256"] = content_hash(public_report)
    _write_json_atomic(output_path, public_report)
    return public_report, private_report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path, private_path = _validated_output_paths(args.output, args.private_output)
    try:
        _cpu_only_run_contract_validation(args)
        route_budget_audit = _public_policy_budget_audit(args)
    except BaseException as exc:
        _write_json_atomic(
            _failure_path(output_path),
            {
                "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-failure.v1",
                "status": "FAIL",
                "formal_score_eligible": False,
                "evidence_scope": FLEET_PRECHECK_SCOPE,
                "failure_stage": "cpu_only_contract_or_public_route_budget_audit",
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            "CF2X fleet preflight rejected before Isaac launch; receipt: "
            f"{_failure_path(output_path)}",
            file=sys.stderr,
        )
        return 2
    if route_budget_audit is not None and route_budget_audit["status"] == "BUDGET_INFEASIBLE":
        _write_progress(
            output_path,
            "rejected_public_route_budget",
            status="REJECTED",
            route_budget_status=route_budget_audit["status"],
        )
        _write_json_atomic(
            _failure_path(output_path),
            {
                "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-failure.v1",
                "status": "BUDGET_INFEASIBLE",
                "formal_score_eligible": False,
                "evidence_scope": FLEET_PRECHECK_SCOPE,
                "failure_stage": "public_route_budget_audit",
                "reason": (
                    "the public route exceeds the frozen episode duration even under a "
                    "kinematic lower-bound model; no Isaac process was launched"
                ),
                "route_budget_audit": route_budget_audit,
            },
        )
        print(
            f"CF2X fleet preflight rejected budget-infeasible public route; receipt: "
            f"{_failure_path(output_path)}",
            file=sys.stderr,
        )
        return 2
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "isaaclab")
        raise SystemExit(
            f"IsaacLab AppLauncher unavailable ({missing}); activate env_isaaclab first"
        ) from exc
    simulation_app = AppLauncher(args).app
    try:
        public, _ = _run(args)
        validation = validate_fleet_preflight_reports(output_path, private_path)
        if public["final"]["safe_completion"] is not True:
            _write_progress(
                output_path,
                "completed_with_safety_failure",
                status="FAILED",
                safe_completion=False,
                simulated_time_s=public["execution"]["simulated_time_s"],
            )
            print(f"CF2X fleet preflight failed safety: {output_path}", file=sys.stderr, flush=True)
            return 2
        if (
            args.execution_mode == "shared-hold"
            and public["candidate_shared_hold"]["status"] != "PASS"
        ):
            _write_progress(
                output_path,
                "completed_with_stability_failure",
                status="FAILED",
                candidate_shared_hold_status=public["candidate_shared_hold"]["status"],
                simulated_time_s=public["execution"]["simulated_time_s"],
            )
            print(
                f"CF2X fleet preflight failed candidate stability: {output_path}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        if (
            args.execution_mode == PRIVATE_WITNESS_FIXTURE_MODE
            and public["policy_progress"]["status"] != "PRIVATE_FIXTURE_CLOSED"
        ):
            _write_progress(
                output_path,
                "completed_with_fixture_closure_failure",
                status="FAILED",
                fixture_status=public["policy_progress"]["status"],
                simulated_time_s=public["execution"]["simulated_time_s"],
            )
            print(
                f"CF2X private-witness fixture failed to close: {output_path}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        if (
            args.run_purpose == COMPLETE_CALIBRATION_PURPOSE
            and public["policy_progress"]["status"] != "CALIBRATION_EPISODE_CLOSED"
        ):
            _write_progress(
                output_path,
                "completed_with_calibration_closure_failure",
                status="FAILED",
                calibration_status=public["policy_progress"]["status"],
                simulated_time_s=public["execution"]["simulated_time_s"],
            )
            print(
                f"CF2X complete calibration episode failed closure: {output_path}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        if (
            args.execution_mode == EXTERNAL_PROCESS_POLICY_MODE
            and public["policy_progress"]["status"] == "ADAPTER_FAILED"
        ):
            _write_progress(
                output_path,
                "completed_with_external_adapter_failure",
                status="FAILED",
                policy_status=public["policy_progress"]["status"],
                simulated_time_s=public["execution"]["simulated_time_s"],
            )
            print(f"CF2X external adapter failed: {output_path}", file=sys.stderr, flush=True)
            return 2
        _write_progress(
            output_path,
            (
                "completed_calibration_replay"
                if args.run_purpose == COMPLETE_CALIBRATION_PURPOSE
                else "completed_candidate_run"
            ),
            status="COMPLETED",
            safe_completion=public["final"]["safe_completion"],
            simulated_time_s=public["execution"]["simulated_time_s"],
            policy_status=public["policy_progress"]["status"],
            formal_score_eligible=False,
        )
        print(
            "CF2X shared-world fleet preflight written: "
            f"{output_path} hash={validation['public_report_file_sha256']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_progress(
            output_path,
            "failed_candidate_run",
            status="FAILED",
            exception_type=type(exc).__name__,
        )
        _write_json_atomic(
            _failure_path(output_path),
            {
                "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-failure.v1",
                "status": "FAIL",
                "formal_score_eligible": False,
                "evidence_scope": FLEET_PRECHECK_SCOPE,
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "last_progress": (
                    _progress_path(output_path).read_text(encoding="utf-8")
                    if _progress_path(output_path).is_file()
                    else None
                ),
            },
        )
        print(
            f"CF2X fleet preflight failed; receipt: {_failure_path(output_path)}",
            file=sys.stderr,
        )
        return 2
    finally:
        bridge = getattr(args, "_owned_external_planner_bridge", None)
        if bridge is not None:
            bridge.close()
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
