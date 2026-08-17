"""Fixed low-level velocity/yaw controller shared by native methods."""

from __future__ import annotations

import math

import numpy as np

from .datatypes import DroneState, HighLevelAction, _clamp_norm


class FixedVelocityYawController:
    """A fixed low-level velocity/yaw tracker used by every native method."""

    def __init__(self, *, max_speed_mps: float, max_yaw_rate_rad_s: float) -> None:
        self.max_speed_mps = max_speed_mps
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s

    def desired_world_velocity(self, action: HighLevelAction, yaw_rad: float) -> np.ndarray:
        command = action.vector.copy()
        if action.frame == "body":
            c, s = math.cos(yaw_rad), math.sin(yaw_rad)
            command[:2] = np.array((c * command[0] - s * command[1], s * command[0] + c * command[1]))
        command[:2] = _clamp_norm(command[:2], self.max_speed_mps)
        command[2] = float(np.clip(command[2], -self.max_speed_mps * 0.55, self.max_speed_mps * 0.55))
        return command

    def track(self, state: DroneState, action: HighLevelAction, dt_s: float) -> tuple[np.ndarray, float]:
        desired = self.desired_world_velocity(action, state.yaw_rad)
        response = min(1.0, dt_s * 5.5)
        applied = state.velocity_mps + response * (desired - state.velocity_mps)
        yaw_rate = float(np.clip(action.yaw_rate_rad_s, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s))
        return applied, yaw_rate
