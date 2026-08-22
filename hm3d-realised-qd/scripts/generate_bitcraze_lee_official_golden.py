#!/usr/bin/env python3
"""Generate frozen Lee-controller outputs from Bitcraze's official C binding.

Run this script under WSL from the frozen crazyflie-firmware ``build``
directory, where ``cffirmware.py`` and ``_cffirmware*.so`` are importable.
The generated JSON is consumed by the Windows/Torch differential unit test.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "5d287434b21b9b4fd3577c51e4d90bb4c54a5145"
SOURCE_FILE = "src/modules/src/controller/controller_lee.c"


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
        "reference_jerk_mps3": [0.0, 0.0, 0.0],
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
        "reference_jerk_mps3": [0.8, -0.5, 0.25],
        "heading_deg": 35.0,
        "heading_rate_deg_s": 12.0,
    },
    {
        "name": "negative_collective_uses_official_fallback",
        "repeat": 1,
        "position_m": [0.0, 0.0, 0.0],
        "velocity_mps": [0.0, 0.0, 0.0],
        "attitude_rpy_rad": [0.0, 0.0, 0.0],
        "angular_velocity_body_rad_s": [0.0, 0.0, 0.0],
        "reference_position_m": [0.0, 0.0, 0.0],
        "reference_velocity_mps": [0.0, 0.0, 0.0],
        "reference_acceleration_mps2": [0.0, 0.0, -20.0],
        "reference_jerk_mps3": [0.0, 0.0, 0.0],
        "heading_deg": 45.0,
        "heading_rate_deg_s": 0.0,
    },
    {
        "name": "zero_collective_uses_official_fallback",
        "repeat": 1,
        "position_m": [0.0, 0.0, 0.0],
        "velocity_mps": [0.0, 0.0, 0.0],
        "attitude_rpy_rad": [0.0, 0.0, 0.0],
        "angular_velocity_body_rad_s": [0.0, 0.0, 0.0],
        "reference_position_m": [0.0, 0.0, 0.0],
        "reference_velocity_mps": [0.0, 0.0, 0.0],
        "reference_acceleration_mps2": [12.0, 0.0, -9.81],
        "reference_jerk_mps3": [0.0, 0.0, 0.0],
        "heading_deg": 0.0,
        "heading_rate_deg_s": 0.0,
    },
    {
        "name": "component_error_clamps",
        "repeat": 1,
        "position_m": [-250.0, 250.0, -250.0],
        "velocity_mps": [-180.0, 180.0, -180.0],
        "attitude_rpy_rad": [-0.1, 0.12, -0.2],
        "angular_velocity_body_rad_s": [-0.2, 0.15, -0.1],
        "reference_position_m": [250.0, -250.0, 250.0],
        "reference_velocity_mps": [180.0, -180.0, 180.0],
        "reference_acceleration_mps2": [0.0, 0.0, 0.0],
        "reference_jerk_mps3": [0.0, 0.0, 0.0],
        "heading_deg": -30.0,
        "heading_rate_deg_s": -8.0,
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


def _matrix_rows(cffirmware: Any, value: Any) -> list[list[float]]:
    return [_vec(cffirmware.mrow(value, index)) for index in range(3)]


def _run_case(cffirmware: Any, case: dict[str, Any]) -> dict[str, Any]:
    controller = cffirmware.controllerLee_t()
    cffirmware.controllerLeeInit(controller)
    control = cffirmware.control_t()
    setpoint = cffirmware.setpoint_t()
    sensors = cffirmware.sensorData_t()
    state = cffirmware.state_t()

    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.yaw = cffirmware.modeAbs
    _set_xyz(setpoint.position, case["reference_position_m"])
    _set_xyz(setpoint.velocity, case["reference_velocity_mps"])
    _set_xyz(setpoint.acceleration, case["reference_acceleration_mps2"])
    _set_xyz(setpoint.jerk, case["reference_jerk_mps3"])
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
    state.attitude.pitch = math.degrees(pitch)
    state.attitude.yaw = math.degrees(yaw)
    gyro_deg_s = [math.degrees(value) for value in case["angular_velocity_body_rad_s"]]
    _set_xyz(sensors.gyro, gyro_deg_s)

    for step_index in range(int(case["repeat"])):
        cffirmware.controllerLee(controller, control, setpoint, sensors, state, 2 * step_index)

    return {
        **case,
        "official_mass_kg": float(controller.mass),
        "official_dt_s": 1.0 / 500.0,
        "expected": {
            "thrust_n": float(control.thrustSi),
            "torque_nm": [
                float(control.torqueX),
                float(control.torqueY),
                float(control.torqueZ),
            ],
            "position_error_m": _vec(controller.p_error),
            "velocity_error_mps": _vec(controller.v_error),
            "position_integral": _vec(controller.i_error_pos),
            "attitude_integral": _vec(controller.i_error_att),
            "desired_rotation": _matrix_rows(cffirmware, controller.R_des),
            "angular_velocity_body_rad_s": _vec(controller.omega),
            "desired_angular_velocity_body_rad_s": _vec(controller.omega_r),
            "torque_state_nm": _vec(controller.u),
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
        "schema_version": "bitcraze-lee-official-golden-v1",
        "source_commit": SOURCE_COMMIT,
        "source_file": SOURCE_FILE,
        "generator": Path(__file__).name,
        "cases": [_run_case(cffirmware, case) for case in CASES],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
