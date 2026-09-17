"""Configuration for the kinematic pilot backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PilotRuntimeConfig:
    """Configuration for the explicitly labelled kinematic pilot backend."""

    agent_count: int = 8
    seed: int = 20260722
    dt_s: float = 0.2
    max_steps: int = 72
    world_size_xy_m: tuple[float, float] = (32.0, 24.0)
    min_altitude_m: float = 1.1
    max_altitude_m: float = 5.0
    drone_radius_m: float = 0.28
    max_speed_mps: float = 2.8
    max_yaw_rate_rad_s: float = 1.5
    camera_width: int = 96
    camera_height: int = 72
    camera_hfov_deg: float = 104.0
    camera_vfov_deg: float = 74.0
    camera_pitch_rad: float = -0.45
    lidar_beams: int = 72
    lidar_max_range_m: float = 18.0
    radar_max_range_m: float = 18.0
    candidate_match_radius_m: float = 0.9
    candidate_min_frames: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.agent_count <= 32:
            raise ValueError("agent_count must be in [1, 32]")
        if self.dt_s <= 0.0 or self.max_steps <= 0:
            raise ValueError("dt_s and max_steps must be positive")
        if min(self.world_size_xy_m) <= 6.0:
            raise ValueError("world must have usable search area")
        if not 0.0 < self.min_altitude_m < self.max_altitude_m:
            raise ValueError("invalid altitude limits")
        if self.camera_width < 32 or self.camera_height < 24:
            raise ValueError("camera resolution is too small")
