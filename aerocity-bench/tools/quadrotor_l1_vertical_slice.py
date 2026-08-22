"""Run one internal, non-formal L1 slice with the reviewed local CF2X model.

The historical native gate proves Isaac capabilities with DynamicCuboids.  This
tool instead loads an existing AeroCity CitySpec stage, flies a user-supplied
hash-verified local four-rotor articulation under the shared candidate
controller, and binds a
measured PhysX observation to the existing evaluator-private confirmation
protocol.  It is intentionally an internal preflight fixture: the fixture may
read one private legal witness, but no witness, target ID, target coordinate,
or private evaluator audit is written to the public report or passed through a
method-facing packet.

The output is not a benchmark score.  It remains ineligible until the vehicle
asset, actuator parameters, contact policy, and full four-UAV episode executor
are independently frozen.

Run from the IsaacLab root using the IsaacLab Python environment::

    python quadrotor_l1_vertical_slice.py --layout-root CITY_ROOT \
      --release-config ordinary-v1-mini.json --cf2x-usd $AEROCITY_CF2X_USD \
      --output ./tmp/public_report.json --headless
"""

from __future__ import annotations

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


PUBLIC_EVIDENCE_SCOPE = "quadrotor_internal_vertical_slice_preflight"
PRIVATE_EVIDENCE_SCOPE = "quadrotor_internal_vertical_slice_private_fixture"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--max-sim-time-s", type=float, default=120.0)
    parser.add_argument(
        "--guidance-horizontal-speed-mps",
        type=float,
        required=True,
        help="Candidate public horizontal speed, never above the release contract cap.",
    )
    parser.add_argument(
        "--guidance-vertical-speed-mps",
        type=float,
        required=True,
        help="Candidate public vertical speed, never above the release contract cap.",
    )
    parser.add_argument("--route-tolerance-m", type=float, default=0.35)
    parser.add_argument("--witness-tolerance-m", type=float, default=0.12)
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--device", type=str, default="cpu")
        return parser
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def _failure_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failure.json")


def _private_path(output: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    return output.with_name(f"{output.stem}.private.json")


def _validated_output_paths(
    output: Path, requested_private_output: Path | None
) -> tuple[Path, Path]:
    """Return fresh JSON evidence paths before Isaac starts.

    A vertical-slice receipt is an immutable preflight artifact, not an output
    directory or a cache.  Reusing an existing path could overwrite the only
    evidence of a failed controller run, which would make later conclusions
    unauditable.  Validate this before launching Kit so a path mistake cannot
    consume simulated execution time or write a partial runtime report.
    """

    public_output = output.resolve()
    private_output = _private_path(public_output, requested_private_output)
    for label, path in (("public output", public_output), ("private output", private_output)):
        if path.suffix.lower() != ".json":
            raise ValueError(f"{label} must be a .json evidence file: {path}")
    if public_output == private_output:
        raise ValueError("public and private evidence outputs must differ")
    generated_paths = (
        public_output,
        private_output,
        _progress_path(public_output),
        _failure_path(public_output),
    )
    existing = [str(path) for path in generated_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "vertical-slice evidence paths already exist; choose a new output name: "
            + ", ".join(existing)
        )
    return public_output, private_output


def _write_progress(output: Path, stage: str, **details: Any) -> None:
    _write_json_atomic(
        _progress_path(output),
        {
            "schema": "org.aerocity.bench.quadrotor-l1-vertical-slice-progress.v1",
            "status": "IN_PROGRESS",
            "formal_score_eligible": False,
            "stage": stage,
            "timestamp_unix_s": time.time(),
            **details,
        },
    )


def _finite_list(tensor: Any) -> list[float]:
    values = tensor.detach().cpu().reshape(-1).tolist()
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("Isaac returned a non-finite state")
    return result


def _norm(values: tuple[float, float, float] | list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return _norm([left - right for left, right in zip(first, second, strict=True)])


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _euler_from_wxyz(quaternion: list[float]) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in radians from an Isaac world quaternion."""

    if len(quaternion) != 4:
        raise ValueError("orientation_wxyz must have four values")
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
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


def _state(
    robot: Any,
    *,
    rotor_speeds_rad_s: tuple[float, float, float, float],
    rotor_references_rad_s: tuple[float, float, float, float],
    applied_rotor_thrust_n: tuple[float, float, float, float],
    wrench: tuple[float, float, float, float],
    contact_force_n: float,
) -> dict[str, Any]:
    return {
        "position": _finite_list(robot.data.root_pos_w[0]),
        "orientation_wxyz": _finite_list(robot.data.root_quat_w[0]),
        "linear_velocity_mps": _finite_list(robot.data.root_lin_vel_w[0]),
        "angular_velocity_rad_s": _finite_list(robot.data.root_ang_vel_w[0]),
        "rotor_motor_state_rad_s": [float(value) for value in rotor_speeds_rad_s],
        "rotor_reference_rad_s": [float(value) for value in rotor_references_rad_s],
        "applied_rotor_thrust_n": [float(value) for value in applied_rotor_thrust_n],
        "body_wrench": [float(value) for value in wrench],
        "contact_force_n": float(contact_force_n),
    }


def _select_private_fixture(
    private_episode: dict[str, Any], public_episode: dict[str, Any]
) -> dict[str, Any]:
    """Choose the shortest prevalidated witness deterministically.

    This helper deliberately consumes the private episode.  Callers must keep
    its result within the evaluator-owned fixture report.
    """

    public_starts = {str(item["drone_id"]): item for item in public_episode["starts"]}
    candidates: list[tuple[tuple[float, str, str], dict[str, Any]]] = []
    for target in private_episode["targets"]:
        if target.get("valid_before_run") is not True:
            continue
        for witness in target.get("legal_witnesses", []):
            proof = witness.get("reachability_proof", {})
            drone_id = str(proof.get("start_drone_id", ""))
            if drone_id not in public_starts:
                continue
            distance_m = float(proof.get("path_distance_upper_bound_m", math.inf))
            if not math.isfinite(distance_m) or distance_m <= 0.0:
                continue
            candidates.append(
                (
                    (distance_m, str(target["target_id"]), str(witness["witness_id"])),
                    {
                        "target": target,
                        "witness": witness,
                        "start": public_starts[drone_id],
                    },
                )
            )
    if not candidates:
        raise ValueError("private episode has no prevalidated witness tied to a public start")
    return min(candidates, key=lambda item: item[0])[1]


def _safe_sky_route(
    start_position: tuple[float, float, float],
    witness_position: tuple[float, float, float],
    transit_altitude_m: float,
) -> list[tuple[float, float, float]]:
    if transit_altitude_m < max(start_position[2], witness_position[2]):
        raise ValueError("private reachability proof transit altitude is below an endpoint")
    return [
        (start_position[0], start_position[1], transit_altitude_m),
        (witness_position[0], witness_position[1], transit_altitude_m),
        witness_position,
    ]


def _within_bounds(
    position: tuple[float, float, float], bounds: dict[str, Any], margin: float
) -> bool:
    return all(
        float(low) + margin <= value <= float(high) - margin
        for value, low, high in zip(position, bounds["minimum"], bounds["maximum"], strict=True)
    )


def _sanitize_public_report(private_report: dict[str, Any]) -> dict[str, Any]:
    """Emit only safe aggregate evidence for a private-fixture preflight."""

    final = private_report["final"]
    return {
        "schema": "org.aerocity.bench.quadrotor-l1-vertical-slice.v1",
        "formal_score_eligible": False,
        "evidence_scope": PUBLIC_EVIDENCE_SCOPE,
        "not_a_formal_four_uav_episode": True,
        "reason_not_formal": (
            "single-UAV evaluator-private fixture; vehicle and controller parameter audits "
            "remain pending"
        ),
        "input_bindings": private_report["input_bindings"],
        "vehicle": private_report["vehicle_public"],
        "execution": {
            "control_action_count": private_report["execution"]["control_action_count"],
            "simulated_time_s": private_report["execution"]["simulated_time_s"],
            "confirmed_receipt_count": len(private_report["confirmation_receipts"]),
            "observation_receipt_count": len(private_report["observation_receipts"]),
            "execution_receipt_count": len(private_report["execution_receipts"]),
            "collision_detected": private_report["final"]["collision_detected"],
            "out_of_bounds_detected": private_report["final"]["out_of_bounds_detected"],
            "returned_home": private_report["final"]["returned_home"],
            "closure_status": final["closure_status"],
        },
        "private_contract_validation": private_report["contract_validation"],
        "private_fixture_commitment": private_report["private_fixture_commitment"],
        "private_report_file_sha256": private_report["private_report_file_sha256"],
    }


def validate_vertical_slice_reports(public_path: Path, private_path: Path) -> dict[str, Any]:
    """Fail closed when public and private vertical-slice evidence diverge."""

    from aerocity_bench.canonical import content_hash, file_hash, read_json
    from aerocity_bench.vertical_slice_contract import validate_private_vertical_slice_report

    public = read_json(public_path)
    private = read_json(private_path)
    expected_private_hash = str(private.pop("private_report_content_sha256", ""))
    if content_hash(private) != expected_private_hash:
        raise ValueError("private vertical slice report hash mismatch")
    private_validation = validate_private_vertical_slice_report(private)
    if public.get("schema") != "org.aerocity.bench.quadrotor-l1-vertical-slice.v1":
        raise ValueError("unexpected public vertical slice report schema")
    if public.get("formal_score_eligible") is not False:
        raise ValueError("a vertical slice must not claim formal-score eligibility")
    if public.get("evidence_scope") != PUBLIC_EVIDENCE_SCOPE:
        raise ValueError("unexpected public vertical slice evidence scope")
    expected_public_hash = str(public.pop("public_report_sha256", ""))
    if content_hash(public) != expected_public_hash:
        raise ValueError("public vertical slice report hash mismatch")
    if public.get("private_report_file_sha256") != file_hash(private_path):
        raise ValueError("public report private-report file hash differs")
    if public.get("private_fixture_commitment") != private.get("private_fixture_commitment"):
        raise ValueError("public and private fixture commitments differ")
    if public.get("private_contract_validation") != private_validation:
        raise ValueError("public report private-contract validation differs")
    public_text = str(public).lower()
    for forbidden in ("target_id", "witness_id", "legal_witness", "target_position"):
        if forbidden in public_text:
            raise ValueError(f"public vertical slice report leaks {forbidden}")
    return {
        "status": "PASS",
        "public_report_sha256": file_hash(public_path),
        "private_report_file_sha256": file_hash(private_path),
        "private_fixture_commitment": private["private_fixture_commitment"],
        "execution_receipt_set_hash": private_validation["execution_receipt_set_hash"],
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_contrib.assets import Multirotor

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
        public_execution_contract,
    )
    from aerocity_bench.public_boundary import audit_public_layout
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
        anisotropic_route_time_lower_bound_s,
        position_anchored_velocity_guidance,
        three_leg_sky_route_waypoint_yaw,
    )

    output_path, private_output_path = _validated_output_paths(args.output, args.private_output)
    layout_root = args.layout_root.resolve()
    audit_public_layout(layout_root)
    release_config_path = args.release_config.resolve()
    stage_path = layout_root / "scene_authority" / "stage.usda"
    city_path = layout_root / "scene_authority" / "cityspec.json"
    task_spec_path = layout_root / "method_public" / "task_spec.json"
    public_episode_path = layout_root / "method_public" / "episodes" / "episode-0000.json"
    private_episode_path = layout_root / "evaluator_private" / "episodes" / "episode-0000.json"
    required_paths = (
        stage_path,
        city_path,
        task_spec_path,
        public_episode_path,
        private_episode_path,
        release_config_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"vertical slice input is absent: {missing}")
    if args.max_sim_time_s <= 0.0:
        raise ValueError("max-sim-time-s must be positive")
    if args.guidance_horizontal_speed_mps <= 0.0 or args.guidance_vertical_speed_mps <= 0.0:
        raise ValueError("anisotropic guidance speeds must be positive")
    if args.route_tolerance_m <= 0.0 or args.witness_tolerance_m <= 0.0:
        raise ValueError("route and witness tolerances must be positive")
    if args.contact_threshold_n <= 0.0:
        raise ValueError("contact threshold must be positive")

    _write_progress(output_path, "loading_contract_inputs", layout_root=str(layout_root))
    config = load_ordinary_config(release_config_path)
    city = read_json(city_path)
    public_episode = read_json(public_episode_path)
    private_episode = read_json(private_episode_path)
    task_spec = read_json(task_spec_path)
    if private_episode.get("layout_hash") != city.get("layout_hash"):
        raise ValueError("private episode and CitySpec layout hashes differ")
    if public_episode.get("episode_id") != private_episode.get("episode_id"):
        raise ValueError("public and private episode IDs differ")
    if task_spec.get("layout_id") != city.get("layout_id"):
        raise ValueError("public task specification and CitySpec layout IDs differ")
    public_contract = public_execution_contract(config.raw["execution_contract"])
    if task_spec.get("execution_contract") != public_contract:
        raise ValueError("public task specification and release execution contracts differ")
    if task_spec.get("public_execution_contract_hash") != content_hash(public_contract):
        raise ValueError("public task specification public execution-contract hash is invalid")
    vehicle_contract = config.raw["execution_contract"]["vehicle"]
    if args.guidance_horizontal_speed_mps > float(vehicle_contract["horizontal_speed_mps"]):
        raise ValueError("candidate horizontal guidance exceeds the release contract cap")
    if args.guidance_vertical_speed_mps > float(vehicle_contract["vertical_speed_mps"]):
        raise ValueError("candidate vertical guidance exceeds the release contract cap")
    guidance_limits = VelocityGuidanceLimits(
        horizontal_speed_mps=float(args.guidance_horizontal_speed_mps),
        vertical_speed_mps=float(args.guidance_vertical_speed_mps),
    )
    guidance_limits.validate()
    fixture = _select_private_fixture(private_episode, public_episode)
    start = fixture["start"]
    witness = fixture["witness"]
    proof = witness["reachability_proof"]
    start_position = tuple(float(value) for value in start["position"])
    witness_pose = Pose3D.from_dict(witness["pose"])
    witness_position = witness_pose.position
    transit_altitude = float(proof["transit_altitude_m"])
    outbound_route = _safe_sky_route(start_position, witness_position, transit_altitude)
    return_route = _safe_sky_route(witness_position, start_position, transit_altitude)
    candidate_outbound_lower_bound_s = anisotropic_route_time_lower_bound_s(
        (start_position, *outbound_route), limits=guidance_limits
    )
    candidate_return_lower_bound_s = anisotropic_route_time_lower_bound_s(
        (witness_position, *return_route), limits=guidance_limits
    )
    execution_contract = config.raw["execution_contract"]
    candidate_round_trip_lower_bound_s = (
        candidate_outbound_lower_bound_s
        + candidate_return_lower_bound_s
        + float(execution_contract["observe"]["continuous_dwell_s"])
    )
    if args.max_sim_time_s <= candidate_round_trip_lower_bound_s:
        raise ValueError(
            "max_sim_time_s is no greater than the candidate anisotropic round-trip lower "
            f"bound ({candidate_round_trip_lower_bound_s:.6f}s), before settling overhead"
        )
    period = float(execution_contract["control_period_s"])
    physical_dt = project_asset_spec().physics_dt_s
    physical_steps_per_action = int(round(period / physical_dt))
    if not math.isclose(physical_steps_per_action * physical_dt, period, abs_tol=1.0e-9):
        raise ValueError("control period must be an integer number of physical steps")
    max_actions = int(math.floor(args.max_sim_time_s / period))
    if max_actions < 4:
        raise ValueError("max-sim-time-s must allow at least four control actions")

    spec = project_asset_spec()
    controller = candidate_controller_spec()
    asset = verify_local_cf2x_asset(args.cf2x_usd)
    asset_structure = inspect_verified_cf2x_structure(asset)
    input_bindings = {
        "layout_id": str(city["layout_id"]),
        "episode_id": str(public_episode["episode_id"]),
        "stage_sha256": file_hash(stage_path),
        "cityspec_sha256": file_hash(city_path),
        "task_spec_sha256": file_hash(task_spec_path),
        "public_episode_sha256": file_hash(public_episode_path),
        "execution_contract_hash": content_hash(public_contract),
        "release_config_sha256": file_hash(release_config_path),
    }
    private_fixture_commitment = content_hash(
        {
            "target_id": fixture["target"]["target_id"],
            "witness_id": witness["witness_id"],
            "start_drone_id": start["drone_id"],
            "private_episode_sha256": file_hash(private_episode_path),
            "execution_contract_hash": input_bindings["execution_contract_hash"],
        }
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

    sim_cfg = SimulationCfg(
        dt=physical_dt,
        device=args.device,
        create_stage_in_memory=False,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
        ),
    )
    sim = SimulationContext(sim_cfg)
    robot_cfg = build_cf2x_multirotor_cfg(
        asset,
        spec,
        dt_s=physical_dt,
        prim_path="/World/AeroCityVerticalSlice/Drone",
        position_w_m=start_position,
        orientation_wxyz=(
            math.cos(math.radians(float(start["yaw_deg"])) / 2.0),
            0.0,
            0.0,
            math.sin(math.radians(float(start["yaw_deg"])) / 2.0),
        ),
    )
    # Initial placement is part of reset, never part of the flight loop.
    robot = Multirotor(robot_cfg)
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=f"{robot_cfg.prim_path}/body",
            update_period=0.0,
            history_length=1,
        )
    )
    _write_progress(output_path, "articulation_and_contact_sensor_constructed")
    sim.reset()
    robot.update(physical_dt)
    contact_sensor.update(physical_dt)
    body_ids, body_names = robot.find_bodies("body", preserve_order=True)
    if len(body_ids) != 1 or tuple(robot.data.thruster_names) != (
        "m1_prop",
        "m2_prop",
        "m3_prop",
        "m4_prop",
    ):
        raise RuntimeError(
            f"unexpected CF2X multirotor layout: bodies={body_names}, "
            f"thrusters={robot.data.thruster_names}"
        )
    default_root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.reset()
    contact_sensor.reset()
    robot.update(physical_dt)
    contact_sensor.update(physical_dt)
    _write_progress(output_path, "episode_reset_complete")

    body_masses_kg, articulated_mass_kg = read_verified_cf2x_runtime_mass_kg(
        robot, expected_total_mass_kg=spec.mass_kg
    )
    runtime_hover_speed = hover_rotor_speed_for_mass(spec, articulated_mass_kg)
    motor_state: tuple[float, float, float, float] = (runtime_hover_speed,) * 4
    latest_reference: tuple[float, float, float, float] = motor_state
    initial_applied = tuple(float(value) for value in _finite_list(robot.data.applied_thrust[0]))
    latest_applied_thrust: tuple[float, float, float, float] = initial_applied
    latest_wrench: tuple[float, float, float, float] = rotor_thrust_wrench(
        spec, latest_applied_thrust
    )
    colliders = colliders_from_city(city)
    evaluator = PrivateEvaluator(
        config, city, private_episode, receipt_secret=b"aerocity-internal-vertical-slice-secret-v1"
    )
    execution_receipts: list[dict[str, Any]] = []
    observation_receipts: list[dict[str, Any]] = []
    confirmation_receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    previous_receipt_hash: str | None = None
    action_sequence = 0
    task_time_s = 0.0
    energy_used_j = 0.0
    collision_detected = False
    out_of_bounds_detected = False
    max_contact_force_n = 0.0

    def contact_force_n() -> float:
        contact_sensor.update(physical_dt)
        net_forces = contact_sensor.data.net_forces_w
        if net_forces is None:
            raise RuntimeError("ContactSensor returned no net contact force tensor")
        values = _finite_list(net_forces)
        if len(values) % 3:
            raise RuntimeError("ContactSensor net force tensor is not three-dimensional")
        return max(
            (_norm(values[index : index + 3]) for index in range(0, len(values), 3)),
            default=0.0,
        )

    def current_state() -> dict[str, Any]:
        force = contact_force_n()
        return _state(
            robot,
            rotor_speeds_rad_s=motor_state,
            rotor_references_rad_s=latest_reference,
            applied_rotor_thrust_n=latest_applied_thrust,
            wrench=latest_wrench,
            contact_force_n=force,
        )

    def issue(
        *,
        kind: str,
        goal_position: tuple[float, float, float],
        goal_yaw_rad: float,
        route_phase: str,
    ) -> tuple[dict[str, Any], bool]:
        nonlocal action_sequence, task_time_s, energy_used_j, motor_state
        nonlocal latest_reference, latest_applied_thrust, latest_wrench, previous_receipt_hash
        nonlocal collision_detected, out_of_bounds_detected, max_contact_force_n
        before = current_state()
        before_position = tuple(float(value) for value in before["position"])
        source_observation = ObservationPacket(
            episode_id=str(public_episode["episode_id"]),
            observation_id=f"slice-observation-{action_sequence:05d}",
            drone_id=str(start["drone_id"]),
            sequence=action_sequence,
            timestamp_s=task_time_s,
            pose=_pose_from_state(before),
            linear_velocity_world_mps=tuple(
                float(value) for value in before["linear_velocity_mps"]
            ),
            angular_speed_deg_s=math.degrees(_norm(before["angular_velocity_rad_s"])),
            energy_remaining_j=max(
                0.0,
                float(execution_contract["vehicle"]["energy_budget_j"]) - energy_used_j,
            ),
        )
        if kind == "WAYPOINT":
            action = ActionPacket(
                episode_id=source_observation.episode_id,
                drone_id=source_observation.drone_id,
                sequence=action_sequence,
                issued_at_s=task_time_s,
                kind="WAYPOINT",
                waypoint=Pose3D(position=goal_position, yaw_deg=math.degrees(goal_yaw_rad)),
            )
        elif kind == "OBSERVE":
            action = ActionPacket(
                episode_id=source_observation.episode_id,
                drone_id=source_observation.drone_id,
                sequence=action_sequence,
                issued_at_s=task_time_s,
                kind="OBSERVE",
                source_observation_id=source_observation.observation_id,
            )
        elif kind == "RETURN":
            action = ActionPacket(
                episode_id=source_observation.episode_id,
                drone_id=source_observation.drone_id,
                sequence=action_sequence,
                issued_at_s=task_time_s,
                kind="RETURN",
            )
        else:
            raise ValueError(f"unsupported vertical-slice action kind: {kind}")

        confirmation_ids: tuple[str, ...] = ()
        observation_result: dict[str, Any] | None = None
        if action.kind == "OBSERVE":
            observation_receipt, confirmations = evaluator.process(source_observation, action)
            observation_result = {
                "observation_id": observation_receipt.observation_id,
                "drone_id": observation_receipt.drone_id,
                "timestamp_s": observation_receipt.timestamp_s,
                "accepted": observation_receipt.accepted,
                "reason": observation_receipt.reason,
                "receipt_hash": observation_receipt.receipt_hash,
            }
            observation_receipts.append(observation_result)
            confirmation_ids = tuple(item.confirmation_id for item in confirmations)
            confirmation_receipts.extend(item.to_dict() for item in confirmations)
        else:
            evaluator.end_observe(source_observation.drone_id, source_observation.timestamp_s)

        for _ in range(physical_steps_per_action):
            measured = FlightState(
                position_w_m=tuple(_finite_list(robot.data.root_pos_w[0])),
                orientation_wxyz=tuple(_finite_list(robot.data.root_quat_w[0])),
                linear_velocity_w_mps=tuple(_finite_list(robot.data.root_lin_vel_w[0])),
                angular_velocity_w_rad_s=tuple(_finite_list(robot.data.root_ang_vel_w[0])),
            )
            controller_target, controller_velocity, controller_yaw = (
                position_anchored_velocity_guidance(
                    measured.position_w_m,
                    goal_position,
                    goal_yaw_rad,
                    limits=guidance_limits,
                )
            )
            output = controller_step(
                spec,
                controller,
                measured,
                FlightCommand(
                    target_position_w_m=controller_target,
                    target_velocity_w_mps=controller_velocity,
                    target_yaw_rad=controller_yaw,
                ),
                mass_kg=articulated_mass_kg,
            )
            latest_reference = output.rotor_references_rad_s
            motor_state = latest_reference
            requested_thrust = tuple(
                min(
                    spec.thrust_coeff_n_per_rad2 * spec.max_rotor_speed_rad_s**2,
                    max(0.0, spec.thrust_coeff_n_per_rad2 * speed * speed),
                )
                for speed in latest_reference
            )
            import torch

            robot.set_thrust_target(
                torch.tensor([requested_thrust], dtype=torch.float32, device=robot.device)
            )
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(physical_dt)
            contact_sensor.update(physical_dt)
            latest_applied_thrust = tuple(
                float(value) for value in _finite_list(robot.data.applied_thrust[0])
            )
            latest_wrench = rotor_thrust_wrench(spec, latest_applied_thrust)

        after = current_state()
        after_position = tuple(float(value) for value in after["position"])
        contact_force = float(after["contact_force_n"])
        max_contact_force_n = max(max_contact_force_n, contact_force)
        collision = contact_force >= float(args.contact_threshold_n)
        out_of_bounds = not _within_bounds(
            after_position, city["flight_bounds"], float(execution_contract["vehicle"]["radius_m"])
        )
        collision_detected = collision_detected or collision
        out_of_bounds_detected = out_of_bounds_detected or out_of_bounds
        center_clearance, _ = minimum_clearance(after_position, colliders)
        body_clearance = max(
            0.0,
            center_clearance - float(execution_contract["vehicle"]["radius_m"]),
        )
        distance_m = _distance(before_position, after_position)
        action_energy = distance_m * float(
            execution_contract["vehicle"]["energy_per_meter_j"]
        ) + period * float(execution_contract["vehicle"]["hover_power_w"])
        energy_used_j += action_energy
        task_time_end_s = task_time_s + period
        status = "measured_physx_executed"
        if collision:
            status = "contact_sensor_terminal_collision"
        elif out_of_bounds:
            status = "out_of_bounds_terminal"
        receipt = build_l1_execution_receipt(
            action=action,
            source_observation=source_observation,
            state_before=before,
            state_after=after,
            task_time_start_s=task_time_s,
            task_time_end_s=task_time_end_s,
            planning_latency_s=0.0,
            action_executed=action.kind,
            status=status,
            energy_used_j=action_energy,
            minimum_clearance_m=body_clearance,
            collision=collision,
            out_of_bounds=out_of_bounds,
            safety_intervention=False,
            deadline_miss=False,
            previous_receipt_hash=previous_receipt_hash,
            confirmation_ids=confirmation_ids,
        ).to_dict()
        execution_receipts.append(receipt)
        previous_receipt_hash = str(receipt["receipt_hash"])
        trace.append(
            {
                "route_phase": route_phase,
                "action": action.to_dict(),
                "source_observation": source_observation.to_dict(),
                "observation_receipt": observation_result,
                "state_before": before,
                "state_after": after,
                "execution_receipt_hash": receipt["receipt_hash"],
                "cpu_geometric_clearance_m": body_clearance,
                "contact_sensor_force_n": contact_force,
            }
        )
        task_time_s = task_time_end_s
        action_sequence += 1
        terminal = collision or out_of_bounds
        if terminal:
            category = "collision" if collision else "out_of_bounds_failure"
            failures.append(
                FailureRecord(
                    episode_id=source_observation.episode_id,
                    drone_id=source_observation.drone_id,
                    task_time_s=task_time_s,
                    category=category,
                    detail=(
                        f"contact_force_n={contact_force:.6f}"
                        if collision
                        else "measured root position left flight bounds"
                    ),
                    terminal=True,
                ).to_dict()
            )
        return after, terminal

    _write_progress(output_path, "beginning_measured_physx_route")
    outbound_index = 0
    return_index = 0
    phase = "outbound"
    confirmed = False
    returned_home = False
    closure_status = "INCOMPLETE"
    started = time.perf_counter()
    while action_sequence < max_actions:
        state = current_state()
        position = tuple(float(value) for value in state["position"])
        linear_speed = _norm(state["linear_velocity_mps"])
        angular_speed_deg_s = math.degrees(_norm(state["angular_velocity_rad_s"]))
        if phase == "outbound":
            goal = outbound_route[outbound_index]
            tolerance = (
                float(args.witness_tolerance_m)
                if outbound_index == len(outbound_route) - 1
                else float(args.route_tolerance_m)
            )
            desired_yaw = three_leg_sky_route_waypoint_yaw(
                tuple(outbound_route),
                outbound_index,
                terminal_yaw_rad=math.radians(witness_pose.yaw_deg),
            )
            if _distance(position, goal) <= tolerance and linear_speed <= 0.30:
                if outbound_index < len(outbound_route) - 1:
                    outbound_index += 1
                    continue
                phase = "observe"
                continue
            _, terminal = issue(
                kind="WAYPOINT",
                goal_position=goal,
                goal_yaw_rad=desired_yaw,
                route_phase="outbound",
            )
        elif phase == "observe":
            yaw_error = abs(
                _wrap_angle_rad(
                    _euler_from_wxyz(state["orientation_wxyz"])[2]
                    - math.radians(witness_pose.yaw_deg)
                )
            )
            stable = (
                _distance(position, witness_position) <= float(args.witness_tolerance_m)
                and linear_speed <= float(execution_contract["observe"]["max_linear_speed_mps"])
                and angular_speed_deg_s
                <= float(execution_contract["observe"]["max_angular_speed_deg_s"])
                and yaw_error <= math.radians(5.0)
            )
            _, terminal = issue(
                kind="OBSERVE" if stable else "WAYPOINT",
                goal_position=witness_position,
                goal_yaw_rad=math.radians(witness_pose.yaw_deg),
                route_phase="observe",
            )
            if confirmation_receipts:
                confirmed = True
                phase = "return"
        elif phase == "return":
            goal = return_route[return_index]
            desired_yaw = three_leg_sky_route_waypoint_yaw(
                tuple(return_route),
                return_index,
            )
            if _distance(position, goal) <= float(args.route_tolerance_m) and linear_speed <= 0.30:
                if return_index < len(return_route) - 1:
                    return_index += 1
                    continue
                returned_home = _distance(position, start_position) <= float(
                    execution_contract["vehicle"]["home_radius_m"]
                )
                closure_status = "PASS" if confirmed and returned_home else "FAIL"
                break
            _, terminal = issue(
                kind="RETURN",
                goal_position=goal,
                goal_yaw_rad=desired_yaw,
                route_phase="return",
            )
        else:
            raise RuntimeError(f"unknown vertical-slice phase: {phase}")
        if terminal:
            closure_status = "FAIL"
            break
    else:
        failures.append(
            FailureRecord(
                episode_id=str(public_episode["episode_id"]),
                drone_id=str(start["drone_id"]),
                task_time_s=task_time_s,
                category="deadline_failure",
                detail=f"vertical slice exhausted max_sim_time_s={args.max_sim_time_s}",
                terminal=True,
            ).to_dict()
        )
        closure_status = "FAIL"

    final_state = current_state()
    private_report: dict[str, Any] = {
        "schema": "org.aerocity.bench.quadrotor-l1-vertical-slice-private.v1",
        "formal_score_eligible": False,
        "evidence_scope": PRIVATE_EVIDENCE_SCOPE,
        "input_bindings": input_bindings,
        "private_fixture": {
            "target_id": fixture["target"]["target_id"],
            "witness_id": witness["witness_id"],
            "start_drone_id": start["drone_id"],
            "witness_pose": witness["pose"],
            "reachability_proof": proof,
        },
        "private_fixture_commitment": private_fixture_commitment,
        "closure_contract": {
            "home_position": list(start_position),
            "home_radius_m": float(execution_contract["vehicle"]["home_radius_m"]),
        },
        "vehicle_public": {
            "execution_model": (
                "cf2x_multirotor_per_rotor_thrust_geometry_allocated_root_wrench_physx"
            ),
            "asset_sha256": asset.usd_sha256,
            "schema_sha256": asset.schema_sha256,
            "asset_structure": asset_structure,
            "asset_provenance_status": spec.provenance_status,
            "controller_provenance_status": controller.provenance_status,
            "articulated_mass_kg": articulated_mass_kg,
            "physx_body_masses_kg": list(body_masses_kg),
            "physical_dt_s": physical_dt,
            "control_period_s": period,
            "contact_evidence": "IsaacLab ContactSensor.net_forces_w",
            "contact_threshold_n": float(args.contact_threshold_n),
            "direct_root_state_writes_during_loop": False,
            "rotor_joint_velocity_targets_written_during_loop": False,
            "applied_thrust_evidence": "Multirotor.data.applied_thrust",
            "wrench_application_model": "derived_geometry_allocation_to_root_body_physx",
            "prop_link_forces_applied_directly": False,
            "guidance_profile": {
                **guidance_limits.to_dict(),
                "release_horizontal_speed_cap_mps": float(vehicle_contract["horizontal_speed_mps"]),
                "release_vertical_speed_cap_mps": float(vehicle_contract["vertical_speed_mps"]),
                "candidate_outbound_route_lower_bound_s": candidate_outbound_lower_bound_s,
                "candidate_return_route_lower_bound_s": candidate_return_lower_bound_s,
                "candidate_round_trip_lower_bound_s": candidate_round_trip_lower_bound_s,
                "status": "candidate_preflight_profile_not_formal_execution_envelope",
            },
        },
        "execution": {
            "control_action_count": action_sequence,
            "simulated_time_s": task_time_s,
            "wall_clock_s": time.perf_counter() - started,
            "energy_used_j_estimate": energy_used_j,
            "physical_steps_per_action": physical_steps_per_action,
        },
        "observation_receipts": observation_receipts,
        "confirmation_receipts": confirmation_receipts,
        "execution_receipts": execution_receipts,
        "failure_records": failures,
        "evaluator_private_audit": evaluator.private_audit_snapshot(),
        "trace_private": trace,
        "final": {
            "closure_status": closure_status,
            "confirmation_observed": confirmed,
            "returned_home": returned_home,
            "collision_detected": collision_detected,
            "out_of_bounds_detected": out_of_bounds_detected,
            "max_contact_force_n": max_contact_force_n,
            "final_state": final_state,
        },
    }
    from aerocity_bench.vertical_slice_contract import validate_private_vertical_slice_report

    private_report["contract_validation"] = validate_private_vertical_slice_report(private_report)
    private_report["private_report_content_sha256"] = content_hash(private_report)
    _write_json_atomic(private_output_path, private_report)
    # This is a file hash rather than a content hash so public consumers can
    # detect byte-level replacement of the private evidence artifact.
    private_report["private_report_file_sha256"] = file_hash(private_output_path)
    public_report = _sanitize_public_report(private_report)
    public_report["public_report_sha256"] = content_hash(public_report)
    _write_json_atomic(output_path, public_report)
    return public_report, private_report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path, _ = _validated_output_paths(args.output, args.private_output)
    try:
        from aerocity_bench.public_boundary import audit_public_layout

        audit_public_layout(args.layout_root.resolve())
    except BaseException as exc:
        _write_json_atomic(
            _failure_path(output_path),
            {
                "schema": "org.aerocity.bench.quadrotor-l1-vertical-slice-failure.v1",
                "status": "FAIL",
                "formal_score_eligible": False,
                "evidence_scope": PUBLIC_EVIDENCE_SCOPE,
                "failure_stage": "cpu_only_public_boundary_audit",
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
            },
        )
        print(
            "quadrotor L1 vertical slice rejected before Isaac launch: "
            f"{_failure_path(output_path)}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "isaaclab")
        raise SystemExit(
            f"IsaacLab AppLauncher unavailable ({missing}); activate env_isaaclab first"
        ) from exc
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        public, _ = _run(args)
        validation = validate_vertical_slice_reports(
            output_path, _private_path(output_path, args.private_output)
        )
        if public["execution"]["closure_status"] != "PASS":
            print(
                f"quadrotor vertical slice completed without closure: {output_path}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        print(
            "quadrotor L1 vertical slice written: "
            f"{output_path} hash={validation['public_report_sha256']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json_atomic(
            _failure_path(output_path),
            {
                "schema": "org.aerocity.bench.quadrotor-l1-vertical-slice-failure.v1",
                "status": "FAIL",
                "formal_score_eligible": False,
                "evidence_scope": PUBLIC_EVIDENCE_SCOPE,
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
            f"quadrotor L1 vertical slice failed; receipt written: {_failure_path(output_path)}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
