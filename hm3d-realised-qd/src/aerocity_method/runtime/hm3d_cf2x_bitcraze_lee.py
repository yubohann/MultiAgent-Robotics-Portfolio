"""MIT-licensed Bitcraze Lee controller core adapted to Isaac state tensors.

The source reference is Bitcraze ``controller_lee.c`` at commit
``5d287434b21b9b4fd3577c51e4d90bb4c54a5145``.  This module implements the
state and geometric decision core only; Isaac's existing rotor allocation and
thrust limits remain the execution boundary.  It intentionally does not copy
Bitcraze's GPL ``math3d`` helpers or firmware plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_URL = "https://github.com/bitcraze/crazyflie-firmware"
SOURCE_COMMIT = "5d287434b21b9b4fd3577c51e4d90bb4c54a5145"
SOURCE_FILE = "src/modules/src/controller/controller_lee.c"
CONTROLLER_ID = "bitcraze-lee-decision-core-isaac-guarded-v4"
OFFICIAL_CONTROL_RATE_HZ = 500.0
POSITION_ERROR_LIMIT = 100.0
VELOCITY_ERROR_LIMIT = 100.0


def _rotation_matrix_from_quaternion_wxyz(quaternion: Any) -> Any:
    import torch

    q = quaternion / torch.linalg.norm(quaternion, dim=1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q.unbind(dim=1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=1),
        ),
        dim=1,
    )


def _limit_force_tilt(force_world: Any, maximum_tilt_rad: float | None) -> Any:
    """Apply the executor's geometric tilt contract without changing Lee gains.

    The firmware delegates motor saturation to its own power-distribution
    layer.  Isaac receives wrench commands directly, while the exploration
    planner reserves clearance only for its declared tilt envelope.  This
    bridge-side limit therefore belongs here rather than in a planner or a
    method-specific safety rule.
    """

    import torch

    if maximum_tilt_rad is None:
        return force_world
    if not 0.0 < maximum_tilt_rad < 0.5 * torch.pi:
        raise ValueError("Bitcraze Lee maximum tilt must lie in (0, pi / 2)")
    vertical = torch.clamp(force_world[:, 2:3], min=1.0e-6)
    horizontal = force_world[:, :2]
    horizontal_norm = torch.linalg.norm(horizontal, dim=1, keepdim=True)
    maximum_horizontal = vertical * torch.tan(
        torch.as_tensor(maximum_tilt_rad, dtype=force_world.dtype, device=force_world.device)
    )
    constrained_horizontal = horizontal * torch.clamp(
        maximum_horizontal / horizontal_norm.clamp_min(1.0e-8), max=1.0
    )
    return torch.cat((constrained_horizontal, vertical), dim=1)


def _desired_rotation(force_world: Any, yaw_rad: Any, collective_thrust: Any) -> Any:
    import torch

    batch_size = int(force_world.shape[0])
    basis_y = torch.tensor(
        (0.0, 1.0, 0.0), device=force_world.device, dtype=force_world.dtype
    ).expand(batch_size, -1)
    basis_z = torch.tensor(
        (0.0, 0.0, 1.0), device=force_world.device, dtype=force_world.dtype
    ).expand(batch_size, -1)
    force_norm = torch.linalg.norm(force_world, dim=1, keepdim=True)
    positive_collective = collective_thrust[:, None] > 0.0
    b3 = torch.where(
        positive_collective,
        force_world / force_norm.clamp_min(1.0e-8),
        basis_z,
    )
    heading = torch.stack(
        (torch.cos(yaw_rad), torch.sin(yaw_rad), torch.zeros_like(yaw_rad)), dim=1
    )
    cross = torch.linalg.cross(b3, heading, dim=1)
    cross_norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    b2 = torch.where(cross_norm > 0.0, cross / cross_norm.clamp_min(1.0e-8), basis_y)
    b1 = torch.linalg.cross(b2, b3, dim=1)
    return torch.stack((b1, b2, b3), dim=2)


@dataclass(slots=True)
class BitcrazeLeeTracker:
    """Stateful Torch translation of the official Lee decision core."""

    mass_kg: float
    dt_s: float
    position_kp: tuple[float, float, float] = (7.0, 7.0, 7.0)
    position_kd: tuple[float, float, float] = (4.0, 4.0, 4.0)
    position_ki: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attitude_kr: tuple[float, float, float] = (0.007, 0.007, 0.008)
    angular_kd: tuple[float, float, float] = (0.00115, 0.00115, 0.002)
    attitude_ki: tuple[float, float, float] = (0.03, 0.03, 0.03)
    inertia_kg_m2: tuple[float, float, float] = (16.571710e-6, 16.655602e-6, 29.261652e-6)
    maximum_feedback_acceleration_mps2: float | None = None
    maximum_tilt_rad: float | None = None
    _position_integral: Any = None
    _attitude_integral: Any = None
    _previous_heading_rad: Any = None

    def reset(self) -> None:
        self._position_integral = None
        self._attitude_integral = None
        self._previous_heading_rad = None

    def _ensure_state(self, reference_positions: Any) -> None:
        import torch

        shape = reference_positions.shape
        if self._position_integral is None or tuple(self._position_integral.shape) != tuple(shape):
            self._position_integral = torch.zeros_like(reference_positions)
            self._attitude_integral = torch.zeros_like(reference_positions)

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
        reference_jerks: Any | None = None,
        heading_rates_deg_s: Any | None = None,
        dt_s: float | None = None,
    ) -> dict[str, Any]:
        import torch

        dt = self.dt_s if dt_s is None else float(dt_s)
        if not 0.0 < dt <= 1.0:
            raise ValueError("Bitcraze Lee controller dt must be in (0, 1]")
        if (
            self.maximum_feedback_acceleration_mps2 is not None
            and self.maximum_feedback_acceleration_mps2 <= 0.0
        ):
            raise ValueError("Bitcraze Lee feedback acceleration limit must be positive")
        self._ensure_state(reference_positions)
        kp = torch.as_tensor(self.position_kp, device=position.device, dtype=position.dtype)
        kd = torch.as_tensor(self.position_kd, device=position.device, dtype=position.dtype)
        ki_pos = torch.as_tensor(self.position_ki, device=position.device, dtype=position.dtype)
        kr = torch.as_tensor(self.attitude_kr, device=position.device, dtype=position.dtype)
        kw = torch.as_tensor(self.angular_kd, device=position.device, dtype=position.dtype)
        ki_att = torch.as_tensor(self.attitude_ki, device=position.device, dtype=position.dtype)
        inertia = torch.as_tensor(self.inertia_kg_m2, device=position.device, dtype=position.dtype)

        pos_error = torch.clamp(
            reference_positions - position, -POSITION_ERROR_LIMIT, POSITION_ERROR_LIMIT
        )
        vel_error = torch.clamp(
            reference_velocities - velocity, -VELOCITY_ERROR_LIMIT, VELOCITY_ERROR_LIMIT
        )
        self._position_integral = self._position_integral + dt * pos_error
        feedback_acceleration = kp * pos_error + kd * vel_error + ki_pos * self._position_integral
        if self.maximum_feedback_acceleration_mps2 is not None:
            feedback_norm = torch.linalg.norm(feedback_acceleration, dim=1, keepdim=True)
            feedback_acceleration = feedback_acceleration * torch.clamp(
                self.maximum_feedback_acceleration_mps2 / feedback_norm.clamp_min(1.0e-8),
                max=1.0,
            )
        desired_acceleration = reference_accelerations + feedback_acceleration
        force_world = self.mass_kg * desired_acceleration
        force_world = force_world.clone()
        force_world[:, 2] += self.mass_kg * 9.81
        force_world = _limit_force_tilt(force_world, self.maximum_tilt_rad)
        desired_acceleration = force_world / self.mass_kg
        desired_acceleration[:, 2] -= 9.81

        current_rotation = _rotation_matrix_from_quaternion_wxyz(quaternion_wxyz)
        yaw_rad = torch.deg2rad(
            torch.as_tensor(headings_deg, device=position.device, dtype=position.dtype)
        )
        collective = torch.sum(force_world * current_rotation[:, :, 2], dim=1)
        desired_rotation = _desired_rotation(force_world, yaw_rad, collective)
        error_matrix = 0.5 * (
            desired_rotation.transpose(1, 2) @ current_rotation
            - current_rotation.transpose(1, 2) @ desired_rotation
        )
        attitude_error = error_matrix[:, (2, 0, 1), (1, 2, 0)]
        low_thrust = collective < 0.01
        if bool(low_thrust.any().item()):
            self._position_integral = torch.where(
                low_thrust[:, None],
                torch.zeros_like(self._position_integral),
                self._position_integral,
            )
            self._attitude_integral = torch.where(
                low_thrust[:, None],
                torch.zeros_like(self._attitude_integral),
                self._attitude_integral,
            )

        if reference_jerks is None:
            reference_jerks = torch.zeros_like(reference_positions)
        if heading_rates_deg_s is None:
            if self._previous_heading_rad is None:
                heading_rate_rad_s = torch.zeros_like(yaw_rad)
            else:
                heading_delta = torch.atan2(
                    torch.sin(yaw_rad - self._previous_heading_rad),
                    torch.cos(yaw_rad - self._previous_heading_rad),
                )
                heading_rate_rad_s = heading_delta / dt
        else:
            heading_rate_rad_s = torch.deg2rad(
                torch.as_tensor(heading_rates_deg_s, device=position.device, dtype=position.dtype)
            )
        self._previous_heading_rad = yaw_rad.detach().clone()

        desired_x = desired_rotation[:, :, 0]
        desired_y = desired_rotation[:, :, 1]
        desired_z = desired_rotation[:, :, 2]
        projected_jerk = (
            reference_jerks
            - torch.sum(desired_z * reference_jerks, dim=1, keepdim=True) * desired_z
        )
        safe_collective = collective[:, None].abs().clamp_min(1.0e-8)
        horizontal_omega = self.mass_kg * projected_jerk / safe_collective
        horizontal_omega = torch.where(
            collective[:, None] != 0.0, horizontal_omega, torch.zeros_like(horizontal_omega)
        )
        yaw_rate = heading_rate_rad_s * desired_z[:, 2]
        desired_omega = torch.stack(
            (
                -torch.sum(horizontal_omega * desired_y, dim=1),
                torch.sum(horizontal_omega * desired_x, dim=1),
                yaw_rate,
            ),
            dim=1,
        )
        desired_omega_body = (
            current_rotation.transpose(1, 2) @ desired_rotation @ desired_omega[:, :, None]
        )[:, :, 0]
        self._attitude_integral = self._attitude_integral + dt * attitude_error
        omega_error = angular_velocity_body - desired_omega_body
        torque = -kr * attitude_error - kw * omega_error - ki_att * self._attitude_integral
        torque = torque + torch.linalg.cross(
            angular_velocity_body, inertia * angular_velocity_body, dim=1
        )
        wrench = torch.cat((collective[:, None], torque), dim=1)
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
            "control_attitude_rpy_rad": torch.zeros_like(reference_positions),
            "desired_attitude_matrix": desired_rotation,
            "requested_headings_rad": yaw_rad,
            "so3_attitude_errors": attitude_error,
            "requested_forces_world_n": force_world,
            "control_angular_velocities_body_rad_s": angular_velocity_body,
            "desired_angular_velocities_body_rad_s": desired_omega_body,
            "angular_velocity_error_rad_s": omega_error,
            "requested_wrenches": wrench,
            "position_error_m": pos_error,
            "velocity_error_mps": vel_error,
            "attitude_integral": self._attitude_integral,
        }
