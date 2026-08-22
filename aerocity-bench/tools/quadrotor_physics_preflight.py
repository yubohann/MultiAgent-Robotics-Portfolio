"""Run a non-formal CF2X native multirotor preflight.

This tool is deliberately separate from ``isaac_native_gate.py``, which remains
a DynamicCuboid capability probe.  Here a reviewed local CF2X runtime asset is
verified by digest before Isaac starts.  The shared controller requests four
rotor thrusts in Newtons.  IsaacLab's ``ThrusterCfg`` advances them separately,
then ``Multirotor`` maps them through the CF2X geometry allocation to a
root-body PhysX wrench.  The receipt records the per-rotor actuator states and
contact sensor measurements; it does not claim force application at prop links.

The output is an engineering preflight receipt, not a benchmark score.  It
remains ineligible until parameter calibration, collision policy, and the full
multi-UAV evaluator loop have independent evidence.  The host timeout used by
the batch runner is a wall-clock safety guard, never a simulated task budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
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


_PROFILES = (
    "shared-hold",
    "shared-long-hold",
    "shared-long-lateral-hold",
    "shared-lateral-step",
    "shared-lateral-hold",
    "shared-altitude-hold",
    "shared-yaw-hold",
    "open-loop-hover",
    "open-loop-pitch-pulse",
    "open-loop-drop",
)

# This is deliberately only a sensor-evidence floor, not a crash-safety
# threshold.  The isolated 1.5 m CF2X drop produced 5.379 N on 2026-07-31;
# 0.05 N is well above a numerical near-zero reading while leaving the actual
# collision envelope to a later, separately calibrated city-obstacle gate.
_CONTACT_EVIDENCE_MIN_FORCE_N = 0.05


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--sample-every", type=int, default=6)
    parser.add_argument("--profile", choices=_PROFILES, default="shared-hold")
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--device", type=str, default="cpu")
        return parser
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _finite_list(tensor: Any) -> list[float]:
    values = tensor.detach().cpu().reshape(-1).tolist()
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("Isaac returned a non-finite state")
    return result


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_wxyz(orientation_wxyz: list[float]) -> float:
    if len(orientation_wxyz) != 4:
        raise ValueError("orientation_wxyz must have four values")
    w, x, y, z = (float(value) for value in orientation_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def _failure_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failure.json")


def _write_progress(output: Path, stage: str, **details: Any) -> None:
    _write_json_atomic(
        _progress_path(output),
        {
            "schema": "org.aerocity.bench.quadrotor-physx-preflight-progress.v2",
            "status": "IN_PROGRESS",
            "formal_score_eligible": False,
            "stage": stage,
            "timestamp_unix_s": time.time(),
            **details,
        },
    )


def _rotor_diagnostics(
    spec: Any,
    requested_thrust_n: tuple[float, float, float, float],
    applied_thrust_n: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "body_name": ("m1_prop", "m2_prop", "m3_prop", "m4_prop")[index],
            "joint_name": spec.rotor_joint_names[index],
            "position_body_m": [float(value) for value in spec.rotor_positions_body_m[index]],
            "spin_direction": spec.rotor_spin_directions[index],
            "requested_thrust_n": float(requested_thrust_n[index]),
            "applied_thrust_n": float(applied_thrust_n[index]),
        }
        for index in range(4)
    ]


def _state(
    robot: Any,
    spec: Any,
    requested_thrust_n: tuple[float, float, float, float],
    applied_thrust_n: tuple[float, float, float, float],
    *,
    controller_output: dict[str, Any] | None = None,
    target_position_w_m: tuple[float, float, float] | None = None,
    target_yaw_rad: float | None = None,
) -> dict[str, Any]:
    from aerocity_bench.quadrotor_dynamics import rotor_thrust_wrench

    applied_wrench = rotor_thrust_wrench(spec, applied_thrust_n)
    result: dict[str, Any] = {
        "position_w_m": _finite_list(robot.data.root_pos_w[0]),
        "orientation_wxyz": _finite_list(robot.data.root_quat_w[0]),
        "linear_velocity_w_mps": _finite_list(robot.data.root_lin_vel_w[0]),
        "angular_velocity_w_rad_s": _finite_list(robot.data.root_ang_vel_w[0]),
        "requested_rotor_thrust_n": [float(value) for value in requested_thrust_n],
        "applied_rotor_thrust_n": [float(value) for value in applied_thrust_n],
        "rotors": _rotor_diagnostics(spec, requested_thrust_n, applied_thrust_n),
        "applied_body_wrench": [float(value) for value in applied_wrench],
    }
    if controller_output is not None:
        result["controller_output"] = controller_output
    if target_position_w_m is not None:
        result["target_position_w_m"] = [float(value) for value in target_position_w_m]
    if target_yaw_rad is not None:
        result["target_yaw_rad"] = float(target_yaw_rad)
    return result


def _contact_force_n(contact_sensor: Any, dt_s: float) -> float:
    contact_sensor.update(dt_s)
    net_forces = contact_sensor.data.net_forces_w
    if net_forces is None:
        raise RuntimeError("ContactSensor returned no net contact force tensor")
    values = _finite_list(net_forces)
    if len(values) % 3:
        raise RuntimeError("ContactSensor net force tensor is not three-dimensional")
    magnitudes = (
        math.sqrt(sum(value * value for value in values[index : index + 3]))
        for index in range(0, len(values), 3)
    )
    return max(magnitudes, default=0.0)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_contrib.assets import Multirotor

    from aerocity_bench.canonical import content_hash, file_hash
    from aerocity_bench.cf2x_contract import (
        CF2X_THRUSTER_BODY_NAMES,
        inspect_verified_cf2x_structure,
        verify_local_cf2x_asset,
    )
    from aerocity_bench.cf2x_native import (
        build_cf2x_multirotor_cfg,
        read_verified_cf2x_runtime_mass_kg,
    )
    from aerocity_bench.hover_stability import (
        candidate_long_horizon_hover_thresholds,
        long_horizon_hover_checks,
        long_horizon_hover_metrics,
    )
    from aerocity_bench.quadrotor_dynamics import (
        FlightCommand,
        FlightState,
        candidate_controller_spec,
        controller_step,
        project_asset_spec,
    )

    output_path = args.output.resolve()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.sample_every <= 0:
        raise ValueError("sample-every must be positive")
    asset = verify_local_cf2x_asset(args.cf2x_usd)
    _write_progress(output_path, "cf2x_asset_hash_verified", asset=asset.fingerprint_payload())
    asset_structure = inspect_verified_cf2x_structure(asset)

    spec = project_asset_spec()
    controller = candidate_controller_spec()
    sim_cfg = SimulationCfg(
        dt=spec.physics_dt_s,
        device=args.device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
        ),
    )
    _write_progress(output_path, "creating_simulation_context")
    sim = SimulationContext(sim_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/AeroCityPreflight/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/AeroCityPreflight/Light", light_cfg)

    robot_cfg = build_cf2x_multirotor_cfg(
        asset,
        spec,
        dt_s=spec.physics_dt_s,
        prim_path="/World/AeroCityPreflight/Drone",
        position_w_m=(0.0, 0.0, 1.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    robot = Multirotor(robot_cfg)
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=f"{robot_cfg.prim_path}/body",
            update_period=0.0,
            history_length=1,
        )
    )
    _write_progress(output_path, "cf2x_multirotor_and_contact_sensor_constructed")
    sim.reset()
    robot.update(spec.physics_dt_s)
    contact_sensor.update(spec.physics_dt_s)
    body_ids, body_names = robot.find_bodies("body", preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"CF2X body layout is not singular: {body_names}")
    if tuple(robot.data.thruster_names) != CF2X_THRUSTER_BODY_NAMES:
        raise RuntimeError(
            "CF2X thruster order differs from the reviewed contract: "
            f"{tuple(robot.data.thruster_names)}"
        )

    # This is the episode reset boundary.  The flight loop below performs no
    # root pose, root velocity, or joint-state writes.
    default_root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.reset()
    contact_sensor.reset()
    robot.update(spec.physics_dt_s)
    contact_sensor.update(spec.physics_dt_s)
    _write_progress(output_path, "episode_reset_before_flight_loop")

    body_masses_kg, articulated_mass_kg = read_verified_cf2x_runtime_mass_kg(
        robot, expected_total_mass_kg=spec.mass_kg
    )
    max_thrust = spec.thrust_coeff_n_per_rad2 * spec.max_rotor_speed_rad_s**2
    hover_thrust = articulated_mass_kg * spec.gravity_mps2 / 4.0
    requested_thrust = (hover_thrust,) * 4
    applied_thrust = tuple(float(value) for value in _finite_list(robot.data.applied_thrust[0]))
    if len(applied_thrust) != 4:
        raise RuntimeError("CF2X actuator did not expose four applied thrust values")
    reset_state = _state(robot, spec, requested_thrust, applied_thrust)

    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    maneuver_steps = 0
    max_contact_force_n = 0.0
    for step in range(args.steps):
        controller_output: dict[str, Any] | None = None
        target_position: tuple[float, float, float] | None = None
        target_yaw_rad = 0.0
        if args.profile in {"open-loop-hover", "open-loop-pitch-pulse", "open-loop-drop"}:
            reference_rad_s = (spec.hover_rotor_speed_rad_s,) * 4
            if args.profile == "open-loop-pitch-pulse" and 30 <= step < 60:
                reference_rad_s = tuple(
                    spec.hover_rotor_speed_rad_s * factor for factor in (1.04, 1.04, 0.96, 0.96)
                )
                maneuver_steps += 1
            elif args.profile == "open-loop-drop":
                # The isolated contact gate intentionally commands no lift.
                # The ensuing ground contact must arise from PhysX, never from
                # a root-state write or a synthetic contact record.
                reference_rad_s = (0.0,) * 4
        else:
            target_position = (0.0, 0.0, 1.5)
            if args.profile == "shared-lateral-step" and 30 <= step < 60:
                target_position = (0.35, 0.0, 1.5)
                maneuver_steps += 1
            elif args.profile in {"shared-lateral-hold", "shared-long-lateral-hold"} and step >= 30:
                target_position = (0.35, 0.0, 1.5)
                maneuver_steps += 1
            elif args.profile == "shared-altitude-hold" and step >= 30:
                target_position = (0.0, 0.0, 1.8)
                maneuver_steps += 1
            elif args.profile == "shared-yaw-hold" and step >= 30:
                target_yaw_rad = math.pi / 2.0
                maneuver_steps += 1
            measured = FlightState(
                position_w_m=tuple(_finite_list(robot.data.root_pos_w[0])),
                orientation_wxyz=tuple(_finite_list(robot.data.root_quat_w[0])),
                linear_velocity_w_mps=tuple(_finite_list(robot.data.root_lin_vel_w[0])),
                angular_velocity_w_rad_s=tuple(_finite_list(robot.data.root_ang_vel_w[0])),
            )
            controller_result = controller_step(
                spec,
                controller,
                measured,
                FlightCommand(
                    target_position_w_m=target_position,
                    target_velocity_w_mps=(0.0, 0.0, 0.0),
                    target_yaw_rad=target_yaw_rad,
                ),
                mass_kg=articulated_mass_kg,
            )
            reference_rad_s = controller_result.rotor_references_rad_s
            controller_output = controller_result.to_dict()

        requested_thrust = tuple(
            min(max_thrust, max(0.0, spec.thrust_coeff_n_per_rad2 * speed * speed))
            for speed in reference_rad_s
        )
        robot.set_thrust_target(
            torch.tensor([requested_thrust], dtype=torch.float32, device=robot.device)
        )
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(spec.physics_dt_s)
        contact_sensor.update(spec.physics_dt_s)
        applied_thrust = tuple(float(value) for value in _finite_list(robot.data.applied_thrust[0]))
        contact_force = _contact_force_n(contact_sensor, spec.physics_dt_s)
        max_contact_force_n = max(max_contact_force_n, contact_force)
        if step % args.sample_every == 0 or step == args.steps - 1:
            sample = _state(
                robot,
                spec,
                requested_thrust,
                applied_thrust,
                controller_output=controller_output,
                target_position_w_m=target_position,
                target_yaw_rad=target_yaw_rad,
            )
            sample["step"] = step
            sample["time_s"] = (step + 1) * spec.physics_dt_s
            sample["contact_force_n"] = contact_force
            trace.append(sample)

    final = trace[-1]
    wall_clock_s = time.perf_counter() - started
    final_position = final["position_w_m"]
    final_velocity = final["linear_velocity_w_mps"]
    initial_position = tuple(float(value) for value in reset_state["position_w_m"])
    initial_yaw_rad = _yaw_from_wxyz(reset_state["orientation_wxyz"])
    max_linear_speed_mps = max(
        math.sqrt(sum(value * value for value in sample["linear_velocity_w_mps"]))
        for sample in trace
    )
    horizontal_displacements_m = [
        math.hypot(
            sample["position_w_m"][0] - initial_position[0],
            sample["position_w_m"][1] - initial_position[1],
        )
        for sample in trace
    ]
    altitude_offsets_m = [sample["position_w_m"][2] - initial_position[2] for sample in trace]
    yaw_progress_rad = [
        abs(_wrap_angle_rad(_yaw_from_wxyz(sample["orientation_wxyz"]) - initial_yaw_rad))
        for sample in trace
    ]
    max_horizontal_displacement_m = max(horizontal_displacements_m)
    final_horizontal_displacement_m = horizontal_displacements_m[-1]
    final_position_target_error_m = (
        math.sqrt(
            sum(
                (final_position[axis] - target_position[axis]) ** 2
                for axis in range(3)
            )
        )
        if target_position is not None
        else None
    )
    max_altitude_above_initial_m = max(altitude_offsets_m)
    max_yaw_progress_rad = max(yaw_progress_rad)
    max_tilt_angle_rad = max(
        math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    1.0
                    - 2.0
                    * (
                        sample["orientation_wxyz"][1] ** 2
                        + sample["orientation_wxyz"][2] ** 2
                    ),
                ),
            )
        )
        for sample in trace
    )
    yaw_errors_rad = [
        abs(_wrap_angle_rad(_yaw_from_wxyz(sample["orientation_wxyz"]) - sample["target_yaw_rad"]))
        for sample in trace
        if "target_yaw_rad" in sample
    ]
    all_applied = [value for sample in trace for value in sample["applied_rotor_thrust_n"]]
    long_hover_thresholds = candidate_long_horizon_hover_thresholds()
    long_hover_metrics = None
    long_hover_checks: dict[str, bool] = {}
    if args.profile in {"shared-long-hold", "shared-long-lateral-hold"}:
        long_hover_metrics = long_horizon_hover_metrics(
            (sample["time_s"] for sample in trace),
            (sample["position_w_m"][2] for sample in trace),
            initial_altitude_m=initial_position[2],
            terminal_vertical_velocity_mps=final_velocity[2],
            warmup_s=long_hover_thresholds.warmup_s,
        )
        long_hover_checks = long_horizon_hover_checks(
            long_hover_metrics,
            long_hover_thresholds,
            max_contact_force_n=max_contact_force_n,
        )
    report: dict[str, Any] = {
        "schema": "org.aerocity.bench.quadrotor-physx-preflight.v2",
        "formal": False,
        "benchmark_admission": "not-admitted",
        "formal_score_eligible": False,
        "vehicle_execution_model": (
            "cf2x_multirotor_per_rotor_thrust_geometry_allocated_root_wrench_physx"
        ),
        "reason_not_formal": (
            "local CF2X redistribution clearance and actuator calibration remain pending; "
            "this is not a full multi-UAV scored episode"
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": str(robot.device),
            "physics_dt_s": spec.physics_dt_s,
            "gravity_mps2": spec.gravity_mps2,
            "preflight_script_sha256": file_hash(Path(__file__).resolve()),
            "dynamics_contract_sha256": file_hash(
                BENCH_ROOT / "src" / "aerocity_bench" / "quadrotor_dynamics.py"
            ),
            "cf2x_contract_sha256": file_hash(
                BENCH_ROOT / "src" / "aerocity_bench" / "cf2x_contract.py"
            ),
            "cf2x_native_sha256": file_hash(
                BENCH_ROOT / "src" / "aerocity_bench" / "cf2x_native.py"
            ),
        },
        "asset": {
            **asset_structure,
            "root_usd_path_disclosed_for_local_debug_only": str(asset.usd_path),
            "schema_path_disclosed_for_local_debug_only": str(asset.schema_path),
        },
        "controller": {
            "shared_controller_used": args.profile.startswith("shared-"),
            "profile": args.profile,
            "candidate_fingerprint": controller.fingerprint_payload(),
            "formal_score_eligible": controller.formal_score_eligible,
        },
        "episode_reset": reset_state,
        "multirotor": {
            "body_names": list(body_names),
            "body_ids": list(body_ids),
            "physx_body_masses_kg": list(body_masses_kg),
            "thruster_names": list(robot.data.thruster_names),
            "articulated_mass_kg": articulated_mass_kg,
            "requested_thrust_unit": "N_per_rotor",
            "applied_thrust_unit": "N_per_rotor",
            "applied_thrust_semantics": "per_rotor_thruster_actuator_state",
            "wrench_application_model": "derived_geometry_allocation_to_root_body_physx",
            "prop_link_forces_applied_directly": False,
            "contact_evidence": "IsaacLab ContactSensor.net_forces_w",
            "contact_evidence_min_force_n": _CONTACT_EVIDENCE_MIN_FORCE_N,
            "direct_root_state_writes_during_loop": False,
            "joint_velocity_targets_written_during_loop": False,
        },
        "steps": args.steps,
        "trace": trace,
        "final_state": {
            "position_w_m": final_position,
            "linear_velocity_w_mps": final_velocity,
            "orientation_wxyz": final["orientation_wxyz"],
            "applied_rotor_thrust_n": final["applied_rotor_thrust_n"],
        },
        "runtime_quality": {
            "wall_clock_s": wall_clock_s,
            "simulated_time_s": args.steps * spec.physics_dt_s,
            "realtime_factor": args.steps * spec.physics_dt_s / max(wall_clock_s, 1.0e-9),
            "root_height_below_0_15m": min(sample["position_w_m"][2] for sample in trace) < 0.15,
            "max_linear_speed_mps": max_linear_speed_mps,
            "max_tilt_angle_rad": max_tilt_angle_rad,
            "max_yaw_error_rad": max(yaw_errors_rad) if yaw_errors_rad else None,
            "max_contact_force_n": max_contact_force_n,
            "max_horizontal_displacement_m": max_horizontal_displacement_m,
            "final_horizontal_displacement_m": final_horizontal_displacement_m,
            "final_position_target_error_m": final_position_target_error_m,
            "max_altitude_above_initial_m": max_altitude_above_initial_m,
            "max_yaw_progress_rad": max_yaw_progress_rad,
            "minimum_applied_rotor_thrust_n": min(all_applied),
            "maximum_applied_rotor_thrust_n": max(all_applied),
            "long_horizon_hover": (
                long_hover_metrics.to_dict() if long_hover_metrics is not None else None
            ),
            "long_horizon_hover_candidate_thresholds": (
                long_hover_thresholds.fingerprint_payload()
                if args.profile in {"shared-long-hold", "shared-long-lateral-hold"}
                else None
            ),
        },
        "checks": {
            "verified_cf2x_root_and_relative_schema": True,
            "usd_structure_matches_cf2x_contract": bool(
                asset_structure["usd_structure_inspected"]
            ),
            "four_rotor_thruster_order": (
                tuple(robot.data.thruster_names) == CF2X_THRUSTER_BODY_NAMES
            ),
            "finite_measured_state": True,
            "per_rotor_thrust_actuator_applied": all(
                math.isfinite(value) and value >= 0.0 for value in all_applied
            ),
            "derived_geometry_allocation_to_root_body_physx": True,
            "contact_sensor_instrumented": math.isfinite(max_contact_force_n),
            "no_direct_root_state_writes_during_loop": True,
            "no_joint_velocity_propulsion_during_loop": True,
            "profile_attitude_response_satisfied": (
                args.profile != "open-loop-pitch-pulse" or max_tilt_angle_rad >= 0.01
            ),
            "profile_position_response_satisfied": (
                (
                    args.profile not in {"shared-lateral-step", "shared-lateral-hold"}
                    or max_horizontal_displacement_m >= 0.01
                )
                and (
                    args.profile != "shared-altitude-hold"
                    or max_altitude_above_initial_m >= 0.01
                )
            ),
            "profile_yaw_response_satisfied": (
                args.profile != "shared-yaw-hold" or max_yaw_progress_rad >= 0.02
            ),
            "profile_maneuver_satisfied": (
                args.profile not in {
                    "shared-lateral-step",
                    "shared-lateral-hold",
                    "shared-altitude-hold",
                    "shared-yaw-hold",
                    "open-loop-pitch-pulse",
                }
                or maneuver_steps > 0
            ),
            "profile_hold_stability_satisfied": (
                args.profile not in {"shared-hold", "open-loop-hover"}
                or (
                    max_horizontal_displacement_m <= 0.05
                    and max(abs(value) for value in altitude_offsets_m) <= 0.05
                    and max_linear_speed_mps <= 0.15
                    and max_contact_force_n <= 1.0e-6
                )
            ),
            "profile_long_hold_stability_satisfied": (
                args.profile not in {"shared-long-hold", "shared-long-lateral-hold"}
                or all(long_hover_checks.values())
            ),
            "profile_long_lateral_response_satisfied": (
                args.profile != "shared-long-lateral-hold"
                or (
                    final_horizontal_displacement_m >= 0.20
                    and final_position_target_error_m is not None
                    and final_position_target_error_m <= 0.05
                )
            ),
            **long_hover_checks,
            "profile_braking_response_satisfied": (
                args.profile != "shared-lateral-step"
                or (
                    max_horizontal_displacement_m >= 0.01
                    and final_horizontal_displacement_m <= 0.08
                    and math.sqrt(sum(value * value for value in final_velocity)) <= 0.15
                    and max_contact_force_n <= 1.0e-6
                )
            ),
            "profile_contact_response_satisfied": (
                args.profile != "open-loop-drop"
                or (
                    max_contact_force_n >= _CONTACT_EVIDENCE_MIN_FORCE_N
                    and min(sample["position_w_m"][2] for sample in trace) < 0.15
                )
            ),
        },
    }
    report["preflight_hash"] = content_hash(report)
    _write_json_atomic(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = args.output.resolve()
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
        report = _run(args)
        checks = report["checks"]
        status = "PASS" if all(value is True for value in checks.values()) else "FAIL"
        print(f"CF2X preflight {status}: {output_path}", flush=True)
        return 0 if status == "PASS" else 2
    except BaseException as exc:
        _write_json_atomic(
            _failure_path(output_path),
            {
                "schema": "org.aerocity.bench.quadrotor-physx-preflight-failure.v2",
                "status": "FAIL",
                "formal_score_eligible": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"CF2X preflight failed: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        # On Windows + Isaac Sim, full cleanup can block indefinitely after a
        # valid small diagnostic.  This tool owns one isolated process, so use
        # the same non-rendering close mode as the L1 slice and let the host
        # guard reject a later process if this process does not exit.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
