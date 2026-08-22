#!/usr/bin/env python3
"""Generate frozen Mellinger-controller outputs from Bitcraze's C binding.

Run this under WSL with the frozen ``crazyflie-firmware`` build directory on
``PYTHONPATH``.  The resulting fixture constrains the Windows/Torch decision
core; it does not claim that the two simulators share the same actuator model.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "5d287434b21b9b4fd3577c51e4d90bb4c54a5145"
SOURCE_FILE = "src/modules/src/controller/controller_mellinger.c"
OFFICIAL_CONTROL_RATE_HZ = 500.0


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "hover",
        "repeat": 1,
        "position_m": [0.0, 0.0, 0.0],
        "velocity_mps": [0.0, 0.0, 0.0],
        "attitude_rpy_rad": [0.0, 0.0, 0.0],
        "angular_velocity_body_rad_s": [0.0, 0.0, 0.0],
        "reference_position_m": [0.0, 0.0, 0.0],
        "reference_velocity_mps": [0.0, 0.0, 0.0],
        "reference_acceleration_mps2": [0.0, 0.0, 0.0],
        "heading_deg": 0.0,
        "heading_rate_deg_s": 0.0,
    },
    {
        "name": "combined_tracking_after_three_steps",
        "repeat": 3,
        "position_m": [0.1, -0.2, 0.3],
        "velocity_mps": [0.15, -0.05, 0.08],
        "attitude_rpy_rad": [0.15, -0.1, 0.2],
        "angular_velocity_body_rad_s": [0.12, -0.08, 0.21],
        "reference_position_m": [0.5, -0.1, 0.7],
        "reference_velocity_mps": [0.3, -0.2, 0.1],
        "reference_acceleration_mps2": [0.4, -0.3, 0.2],
        "heading_deg": 35.0,
        "heading_rate_deg_s": 12.0,
    },
    {
        "name": "legacy_pitch_and_yaw_signs",
        "repeat": 2,
        "position_m": [-0.15, 0.25, 0.1],
        "velocity_mps": [-0.1, 0.06, -0.03],
        "attitude_rpy_rad": [-0.12, 0.09, -0.25],
        "angular_velocity_body_rad_s": [-0.2, 0.14, -0.18],
        "reference_position_m": [0.25, -0.35, 0.5],
        "reference_velocity_mps": [0.2, -0.18, 0.08],
        "reference_acceleration_mps2": [-0.3, 0.25, 0.15],
        "heading_deg": -40.0,
        "heading_rate_deg_s": -10.0,
    },
    {
        "name": "negative_collective_resets_integrals",
        "repeat": 1,
        "position_m": [0.0, 0.0, 0.0],
        "velocity_mps": [0.0, 0.0, 0.0],
        "attitude_rpy_rad": [0.0, 0.0, 0.0],
        "angular_velocity_body_rad_s": [0.0, 0.0, 0.0],
        "reference_position_m": [0.0, 0.0, 0.0],
        "reference_velocity_mps": [0.0, 0.0, 0.0],
        "reference_acceleration_mps2": [0.0, 0.0, -20.0],
        "heading_deg": 45.0,
        "heading_rate_deg_s": 0.0,
    },
)


def _set_xyz(target: Any, values: list[float]) -> None:
    target.x, target.y, target.z = (float(value) for value in values)


def _quaternion_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _vec(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _motor_values(value: Any) -> list[int]:
    return [int(value.motors.m1), int(value.motors.m2), int(value.motors.m3), int(value.motors.m4)]


def _run_case(cffirmware: Any, case: dict[str, Any]) -> dict[str, Any]:
    controller = cffirmware.controllerMellinger_t()
    cffirmware.controllerMellingerInit(controller)
    control = cffirmware.control_t()
    setpoint = cffirmware.setpoint_t()
    sensors = cffirmware.sensorData_t()
    state = cffirmware.state_t()
    uncapped = cffirmware.motors_thrust_uncapped_t()
    pwm = cffirmware.motors_thrust_pwm_t()

    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.yaw = cffirmware.modeAbs
    _set_xyz(setpoint.position, case["reference_position_m"])
    _set_xyz(setpoint.velocity, case["reference_velocity_mps"])
    _set_xyz(setpoint.acceleration, case["reference_acceleration_mps2"])
    setpoint.attitude.yaw = float(case["heading_deg"])
    setpoint.attitudeRate.yaw = float(case["heading_rate_deg_s"])

    _set_xyz(state.position, case["position_m"])
    _set_xyz(state.velocity, case["velocity_mps"])
    roll, pitch, yaw = case["attitude_rpy_rad"]
    qx, qy, qz, qw = _quaternion_xyzw_from_rpy(roll, pitch, yaw)
    state.attitudeQuaternion.x = qx
    state.attitudeQuaternion.y = qy
    state.attitudeQuaternion.z = qz
    state.attitudeQuaternion.w = qw
    state.attitude.roll = math.degrees(roll)
    state.attitude.pitch = -math.degrees(pitch)
    state.attitude.yaw = math.degrees(yaw)
    _set_xyz(sensors.gyro, [math.degrees(value) for value in case["angular_velocity_body_rad_s"]])

    for step_index in range(int(case["repeat"])):
        cffirmware.controllerMellinger(
            controller, control, setpoint, sensors, state, 2 * step_index
        )
    cffirmware.powerDistribution(control, uncapped)
    capped = bool(cffirmware.powerDistributionCap(uncapped, pwm))

    return {
        **case,
        "official_mass_kg": float(controller.mass),
        "official_mass_thrust": float(controller.massThrust),
        "official_dt_s": 1.0 / OFFICIAL_CONTROL_RATE_HZ,
        "expected": {
            "legacy_thrust": float(control.thrust),
            "legacy_roll": int(control.roll),
            "legacy_pitch": int(control.pitch),
            "legacy_yaw": int(control.yaw),
            "position_integral": [
                float(controller.i_error_x),
                float(controller.i_error_y),
                float(controller.i_error_z),
            ],
            "attitude_integral": [
                float(controller.i_error_m_x),
                float(controller.i_error_m_y),
                float(controller.i_error_m_z),
            ],
            "desired_body_z": _vec(controller.z_axis_desired),
            "motor_uncapped": _motor_values(uncapped),
            "motor_pwm": _motor_values(pwm),
            "power_distribution_capped": capped,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cffirmware  # type: ignore[import-not-found]

    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.firmware_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != SOURCE_COMMIT:
        raise RuntimeError(f"official firmware commit mismatch: {actual_commit} != {SOURCE_COMMIT}")
    payload = {
        "schema_version": "bitcraze-mellinger-official-golden-v1",
        "source_commit": SOURCE_COMMIT,
        "source_file": SOURCE_FILE,
        "official_control_rate_hz": OFFICIAL_CONTROL_RATE_HZ,
        "generator": Path(__file__).name,
        "cases": [_run_case(cffirmware, case) for case in CASES],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
