"""Bitcraze Mellinger decision core adapted to Isaac CF2X state tensors.

The decision equations and legacy mixer are constrained by outputs from
``controller_mellinger.c`` at the pinned Bitcraze firmware revision.  The
firmware's PWM output is converted through Crazyswarm2's documented PWM/RPM
calibration and normalized at the active Isaac hover equilibrium.  It is thus
an auditable controlled transfer, not a claim of full firmware equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_URL = "https://github.com/bitcraze/crazyflie-firmware"
SOURCE_COMMIT = "5d287434b21b9b4fd3577c51e4d90bb4c54a5145"
SOURCE_FILE = "src/modules/src/controller/controller_mellinger.c"
CONTROLLER_ID = "bitcraze-mellinger-decision-core-isaac-v1"
OFFICIAL_CONTROL_RATE_HZ = 500.0
OFFICIAL_MASS_THRUST = 132000.0
LEGACY_MAX_PWM = 65535.0
LEGACY_MAX_ATTITUDE_COMMAND = 32000.0
PWM_START_THRESHOLD = 10000.0
CRAZYSWARM2_PWM_TO_RPM_SLOPE = 3.26535711e-01
CRAZYSWARM2_PWM_TO_RPM_OFFSET = 3.37495115e03


def _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: Any) -> Any:
    import torch

    quaternion = quaternion_wxyz / torch.linalg.norm(
        quaternion_wxyz, dim=1, keepdim=True
    ).clamp_min(1.0e-8)
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.stack(
        (
            torch.stack(
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                dim=1,
            ),
            torch.stack(
                (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                dim=1,
            ),
            torch.stack(
                (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
                dim=1,
            ),
        ),
        dim=1,
    )


def _normalize(vector: Any, fallback: Any) -> Any:
    import torch

    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    return torch.where(norm > 1.0e-8, vector / norm.clamp_min(1.0e-8), fallback)


def _clamped_integral(value: Any, derivative: Any, dt_s: float, limit: float) -> Any:
    import torch

    return torch.clamp(value + derivative * dt_s, min=-limit, max=limit)


@dataclass(slots=True)
class BitcrazeMellingerTracker:
    """Stateful Torch translation of the official Mellinger decision core."""

    mass_kg: float
    dt_s: float
    mass_thrust: float = OFFICIAL_MASS_THRUST
    _position_integral: Any = None
    _attitude_integral: Any = None
    _previous_omega_xy: Any = None
    _previous_setpoint_omega_xy: Any = None
    _previous_heading_rad: Any = None
    _physics_time_remainder_s: float = 0.0

    def reset(self) -> None:
        self._position_integral = None
        self._attitude_integral = None
        self._previous_omega_xy = None
        self._previous_setpoint_omega_xy = None
        self._previous_heading_rad = None
        self._physics_time_remainder_s = 0.0

    def _ensure_state(self, reference_positions: Any) -> None:
        import torch

        if self._position_integral is None or tuple(self._position_integral.shape) != tuple(
            reference_positions.shape
        ):
            self._position_integral = torch.zeros_like(reference_positions)
            self._attitude_integral = torch.zeros_like(reference_positions)
            self._previous_omega_xy = torch.zeros_like(reference_positions[:, :2])
            self._previous_setpoint_omega_xy = torch.zeros_like(reference_positions[:, :2])

    def step(
        self,
        *,
        position: Any,
        velocity: Any,
        quaternion_wxyz: Any,
        angular_velocity_body: Any,
        reference_positions: Any,
        reference_velocities: Any,
        reference_accelerations: Any,
        headings_deg: Any,
        heading_rates_deg_s: Any | None = None,
        dt_s: float | None = None,
    ) -> dict[str, Any]:
        import torch

        dt = self.dt_s if dt_s is None else float(dt_s)
        if not 0.0 < dt <= 1.0:
            raise ValueError("Bitcraze Mellinger controller dt must be in (0, 1]")
        self._ensure_state(reference_positions)
        zero = torch.zeros_like(reference_positions)
        world_z = torch.zeros_like(reference_positions)
        world_z[:, 2] = 1.0

        position_error = reference_positions - position
        velocity_error = reference_velocities - velocity
        self._position_integral[:, :2] = _clamped_integral(
            self._position_integral[:, :2], position_error[:, :2], dt, 2.0
        )
        self._position_integral[:, 2] = _clamped_integral(
            self._position_integral[:, 2], position_error[:, 2], dt, 0.4
        )
        target_force_world = torch.empty_like(reference_positions)
        target_force_world[:, :2] = (
            self.mass_kg * reference_accelerations[:, :2]
            + 0.4 * position_error[:, :2]
            + 0.2 * velocity_error[:, :2]
            + 0.05 * self._position_integral[:, :2]
        )
        target_force_world[:, 2] = (
            self.mass_kg * (reference_accelerations[:, 2] + 9.81)
            + 1.25 * position_error[:, 2]
            + 0.4 * velocity_error[:, 2]
            + 0.05 * self._position_integral[:, 2]
        )
        desired_acceleration = target_force_world / self.mass_kg - 9.81 * world_z
        feedback_acceleration = desired_acceleration - reference_accelerations

        current_rotation = _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz)
        current_body_z_world = current_rotation[:, :, 2]
        collective_command = torch.sum(target_force_world * current_body_z_world, dim=1)
        desired_body_z = _normalize(target_force_world, world_z)
        heading_rad = torch.deg2rad(
            torch.as_tensor(headings_deg, dtype=position.dtype, device=position.device)
        )
        desired_heading = torch.stack(
            (torch.cos(heading_rad), torch.sin(heading_rad), torch.zeros_like(heading_rad)), dim=1
        )
        desired_body_y = _normalize(
            torch.linalg.cross(desired_body_z, desired_heading, dim=1),
            torch.tensor((0.0, 1.0, 0.0), dtype=position.dtype, device=position.device).expand_as(
                desired_body_z
            ),
        )
        desired_body_x = torch.linalg.cross(desired_body_y, desired_body_z, dim=1)
        desired_rotation = torch.stack((desired_body_x, desired_body_y, desired_body_z), dim=2)
        error_matrix = (
            desired_rotation.transpose(1, 2) @ current_rotation
            - current_rotation.transpose(1, 2) @ desired_rotation
        )
        attitude_error = torch.stack(
            (error_matrix[:, 2, 1], -error_matrix[:, 0, 2], error_matrix[:, 1, 0]), dim=1
        )

        legacy_body_omega = angular_velocity_body * torch.tensor(
            (1.0, -1.0, 1.0), dtype=position.dtype, device=position.device
        )
        if heading_rates_deg_s is None:
            if self._previous_heading_rad is None:
                heading_rate_rad_s = torch.zeros_like(heading_rad)
            else:
                heading_delta = torch.atan2(
                    torch.sin(heading_rad - self._previous_heading_rad),
                    torch.cos(heading_rad - self._previous_heading_rad),
                )
                heading_rate_rad_s = heading_delta / dt
        else:
            heading_rate_rad_s = torch.deg2rad(
                torch.as_tensor(heading_rates_deg_s, dtype=position.dtype, device=position.device)
            )
        self._previous_heading_rad = heading_rad.detach().clone()
        setpoint_omega = torch.stack(
            (
                torch.zeros_like(heading_rate_rad_s),
                torch.zeros_like(heading_rate_rad_s),
                heading_rate_rad_s,
            ),
            dim=1,
        )
        legacy_setpoint_omega = setpoint_omega * torch.tensor(
            (1.0, -1.0, 1.0), dtype=position.dtype, device=position.device
        )
        angular_velocity_error = legacy_setpoint_omega - legacy_body_omega
        derivative_xy = (
            (legacy_setpoint_omega[:, :2] - self._previous_setpoint_omega_xy)
            - (legacy_body_omega[:, :2] - self._previous_omega_xy)
        ) / dt
        self._previous_omega_xy = legacy_body_omega[:, :2].detach().clone()
        self._previous_setpoint_omega_xy = legacy_setpoint_omega[:, :2].detach().clone()
        self._attitude_integral[:, :2] = _clamped_integral(
            self._attitude_integral[:, :2], -attitude_error[:, :2], dt, 1.0
        )
        self._attitude_integral[:, 2] = _clamped_integral(
            self._attitude_integral[:, 2], -attitude_error[:, 2], dt, 1500.0
        )
        moment = torch.empty_like(reference_positions)
        moment[:, :2] = (
            -70000.0 * attitude_error[:, :2]
            + 20000.0 * angular_velocity_error[:, :2]
            + 200.0 * derivative_xy
        )
        moment[:, 2] = (
            -60000.0 * attitude_error[:, 2]
            + 12000.0 * angular_velocity_error[:, 2]
            + 500.0 * self._attitude_integral[:, 2]
        )
        legacy_roll = torch.trunc(
            torch.clamp(moment[:, 0], -LEGACY_MAX_ATTITUDE_COMMAND, LEGACY_MAX_ATTITUDE_COMMAND)
        )
        legacy_pitch = torch.trunc(
            torch.clamp(moment[:, 1], -LEGACY_MAX_ATTITUDE_COMMAND, LEGACY_MAX_ATTITUDE_COMMAND)
        )
        legacy_yaw = torch.trunc(
            torch.clamp(-moment[:, 2], -LEGACY_MAX_ATTITUDE_COMMAND, LEGACY_MAX_ATTITUDE_COMMAND)
        )
        legacy_thrust = self.mass_thrust * collective_command
        active = legacy_thrust > 0.0
        legacy_roll = torch.where(active, legacy_roll, torch.zeros_like(legacy_roll))
        legacy_pitch = torch.where(active, legacy_pitch, torch.zeros_like(legacy_pitch))
        legacy_yaw = torch.where(active, legacy_yaw, torch.zeros_like(legacy_yaw))
        if bool((~active).any().item()):
            self._position_integral = torch.where(
                active[:, None], self._position_integral, torch.zeros_like(self._position_integral)
            )
            self._attitude_integral = torch.where(
                active[:, None], self._attitude_integral, torch.zeros_like(self._attitude_integral)
            )

        mixer_roll = torch.trunc(legacy_roll / 2.0)
        mixer_pitch = torch.trunc(legacy_pitch / 2.0)
        raw_motor_commands = torch.stack(
            (
                legacy_thrust - mixer_roll + mixer_pitch + legacy_yaw,
                legacy_thrust - mixer_roll - mixer_pitch - legacy_yaw,
                legacy_thrust + mixer_roll - mixer_pitch + legacy_yaw,
                legacy_thrust + mixer_roll + mixer_pitch - legacy_yaw,
            ),
            dim=1,
        ).to(torch.int64)
        highest = torch.maximum(
            torch.amax(raw_motor_commands, dim=1, keepdim=True),
            torch.zeros_like(raw_motor_commands[:, :1]),
        )
        reduction = torch.clamp(highest - int(LEGACY_MAX_PWM), min=0)
        motor_pwm = torch.clamp(raw_motor_commands - reduction, min=0, max=int(LEGACY_MAX_PWM))
        return {
            "controller_id": CONTROLLER_ID,
            "source_url": SOURCE_URL,
            "source_commit": SOURCE_COMMIT,
            "source_file": SOURCE_FILE,
            "reference_positions_m": reference_positions,
            "reference_velocities_mps": reference_velocities,
            "reference_accelerations_mps2": reference_accelerations,
            "control_positions_m": position,
            "control_velocities_mps": velocity,
            "feedback_accelerations_mps2": feedback_acceleration,
            "desired_accelerations_mps2": desired_acceleration,
            "control_attitude_rpy_rad": zero,
            "desired_attitude_matrix": desired_rotation,
            "requested_headings_rad": heading_rad,
            "so3_attitude_errors": attitude_error,
            "requested_forces_world_n": target_force_world,
            "control_angular_velocities_body_rad_s": angular_velocity_body,
            "legacy_thrust": legacy_thrust,
            "legacy_roll": legacy_roll.to(torch.int64),
            "legacy_pitch": legacy_pitch.to(torch.int64),
            "legacy_yaw": legacy_yaw.to(torch.int64),
            "legacy_motor_uncapped": raw_motor_commands,
            "legacy_motor_pwm": motor_pwm,
            "power_distribution_capped": reduction[:, 0] > 0,
            "position_error_m": position_error,
            "velocity_error_mps": velocity_error,
            "position_integral": self._position_integral,
            "attitude_integral": self._attitude_integral,
            "desired_body_z": desired_body_z,
        }

    def step_for_physics(
        self,
        *,
        physics_dt_s: float,
        position: Any,
        velocity: Any,
        quaternion_wxyz: Any,
        angular_velocity_body: Any,
        reference_positions: Any,
        reference_velocities: Any,
        reference_accelerations: Any,
        headings_deg: Any,
    ) -> dict[str, Any]:
        """Advance the 500 Hz official core over one lower-rate PhysX step."""

        elapsed = self._physics_time_remainder_s + float(physics_dt_s)
        update_count = int((elapsed + 1.0e-12) / self.dt_s)
        if update_count <= 0:
            update_count = 1
        self._physics_time_remainder_s = elapsed - update_count * self.dt_s
        result: dict[str, Any] | None = None
        for _ in range(update_count):
            result = self.step(
                position=position,
                velocity=velocity,
                quaternion_wxyz=quaternion_wxyz,
                angular_velocity_body=angular_velocity_body,
                reference_positions=reference_positions,
                reference_velocities=reference_velocities,
                reference_accelerations=reference_accelerations,
                headings_deg=headings_deg,
                dt_s=self.dt_s,
            )
        if result is None:
            raise RuntimeError("Mellinger control schedule produced no update")
        result["official_control_updates"] = update_count
        result["physics_time_remainder_s"] = self._physics_time_remainder_s
        return result

    def rotor_thrust_from_pwm(self, motor_pwm: Any, *, hover_thrust_per_rotor_n: float) -> Any:
        """Map official mixer PWM through the Crazyswarm2 actuator calibration.

        The mapping is normalized once at the active vehicle's hover PWM so a
        mass update in the controller does not silently change the frozen
        Isaac hover equilibrium.
        """

        import torch

        hover_pwm = self.mass_thrust * self.mass_kg * 9.81
        def rpm(pwm: Any) -> Any:
            return CRAZYSWARM2_PWM_TO_RPM_SLOPE * pwm + CRAZYSWARM2_PWM_TO_RPM_OFFSET

        numerator = rpm(motor_pwm.to(dtype=torch.float32))
        denominator = rpm(torch.as_tensor(hover_pwm, device=motor_pwm.device, dtype=torch.float32))
        relative_force = (numerator / denominator.clamp_min(1.0e-8)) ** 2
        return torch.where(
            motor_pwm >= int(PWM_START_THRESHOLD),
            float(hover_thrust_per_rotor_n) * relative_force,
            torch.zeros_like(relative_force),
        )
