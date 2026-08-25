"""Planar kinematics updater for the fixed-height drone experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .math_2d import clip_vector_norm, vector_norm, yaw_from_velocity


@dataclass(frozen=True)
class Kinematics2DConfig:
    """Configuration for the 2D velocity-controlled drone model."""

    dt_s: float = 0.1
    max_speed_mps: float = 6.0
    max_speed_x_mps: float | None = None
    max_speed_y_mps: float | None = None
    max_accel_mps2: float = 4.0
    max_accel_x_mps2: float | None = None
    max_accel_y_mps2: float | None = None
    align_yaw_to_velocity: bool = True
    yaw_deadband_speed_mps: float = 0.05


@dataclass(frozen=True)
class PlanarVelocityCommand2D:
    """High-level planar velocity command produced by the policy."""

    vx_cmd_mps: float
    vy_cmd_mps: float


@dataclass(frozen=True)
class KinematicState2D:
    """Minimal planar state used by the new experiment line."""

    x_m: float
    y_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    yaw_rad: float = 0.0
    t_sec: float = 0.0


class Kinematics2DUpdater:
    """Update planar motion with clipped speed and acceleration limits."""

    def __init__(self, config: Kinematics2DConfig | None = None) -> None:
        self.config = config or Kinematics2DConfig()

    def step(self, state: KinematicState2D, command: PlanarVelocityCommand2D) -> KinematicState2D:
        """Advance the state by one fixed timestep."""

        cfg = self.config
        use_axis_speed_limits = cfg.max_speed_x_mps is not None or cfg.max_speed_y_mps is not None
        max_speed_x_mps = float(cfg.max_speed_mps if cfg.max_speed_x_mps is None else cfg.max_speed_x_mps)
        max_speed_y_mps = float(cfg.max_speed_mps if cfg.max_speed_y_mps is None else cfg.max_speed_y_mps)
        desired_velocity = (
            float(max(min(float(command.vx_cmd_mps), max_speed_x_mps), -max_speed_x_mps)),
            float(max(min(float(command.vy_cmd_mps), max_speed_y_mps), -max_speed_y_mps)),
        )
        if not use_axis_speed_limits:
            desired_velocity = clip_vector_norm(desired_velocity, cfg.max_speed_mps)
        dv = (
            desired_velocity[0] - float(state.vx_mps),
            desired_velocity[1] - float(state.vy_mps),
        )
        use_axis_accel_limits = cfg.max_accel_x_mps2 is not None or cfg.max_accel_y_mps2 is not None
        if use_axis_accel_limits:
            max_dvx = float((cfg.max_accel_mps2 if cfg.max_accel_x_mps2 is None else cfg.max_accel_x_mps2) * cfg.dt_s)
            max_dvy = float((cfg.max_accel_mps2 if cfg.max_accel_y_mps2 is None else cfg.max_accel_y_mps2) * cfg.dt_s)
            limited_dv = (
                float(max(min(dv[0], max_dvx), -max_dvx)),
                float(max(min(dv[1], max_dvy), -max_dvy)),
            )
        else:
            limited_dv = clip_vector_norm(dv, cfg.max_accel_mps2 * cfg.dt_s)
        new_velocity = (
            float(state.vx_mps) + limited_dv[0],
            float(state.vy_mps) + limited_dv[1],
        )
        if use_axis_speed_limits:
            new_velocity = (
                float(max(min(new_velocity[0], max_speed_x_mps), -max_speed_x_mps)),
                float(max(min(new_velocity[1], max_speed_y_mps), -max_speed_y_mps)),
            )
            new_speed = vector_norm(new_velocity)
        else:
            new_speed = vector_norm(new_velocity)
            if new_speed > cfg.max_speed_mps:
                new_velocity = clip_vector_norm(new_velocity, cfg.max_speed_mps)
                new_speed = vector_norm(new_velocity)

        new_yaw = float(state.yaw_rad)
        if cfg.align_yaw_to_velocity and new_speed >= cfg.yaw_deadband_speed_mps:
            new_yaw = yaw_from_velocity(new_velocity, fallback_yaw_rad=new_yaw)

        return KinematicState2D(
            x_m=float(state.x_m) + new_velocity[0] * cfg.dt_s,
            y_m=float(state.y_m) + new_velocity[1] * cfg.dt_s,
            vx_mps=new_velocity[0],
            vy_mps=new_velocity[1],
            yaw_rad=new_yaw,
            t_sec=float(state.t_sec) + cfg.dt_s,
        )

