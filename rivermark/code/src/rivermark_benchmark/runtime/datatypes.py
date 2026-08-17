"""Shared data types for the deterministic multi-UAV pilot runtime."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-9

def _as_vec3(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("expected three finite coordinates")
    return result

def _clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    return vector if magnitude <= maximum or magnitude <= _EPS else vector * (maximum / magnitude)

@dataclass(frozen=True)
class HighLevelAction:
    """Public policy ABI compatible with a velocity/yaw fixed controller.

    ``velocity_xyz`` is interpreted in ``frame``.  The third component is the
    MD-QD-Swarm-compatible ``dz`` command: vertical velocity in m/s, not a
    target altitude.  This keeps high-level policies independent of the
    low-level controller and simulator-specific actuator ABI.
    """

    velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rate_rad_s: float = 0.0
    mode: str = "hold"
    frame: str = "world"
    duration_s: float = 0.2
    source: str = "native_reference"

    def __post_init__(self) -> None:
        _as_vec3(self.velocity_xyz)
        if not math.isfinite(self.yaw_rate_rad_s):
            raise ValueError("yaw_rate_rad_s must be finite")
        if self.mode not in {"transit", "dwell", "hold", "return"}:
            raise ValueError("unknown action mode")
        if self.frame not in {"world", "body"}:
            raise ValueError("action frame must be world or body")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

    @property
    def vector(self) -> np.ndarray:
        return _as_vec3(self.velocity_xyz)

    @classmethod
    def hold(cls, *, source: str = "native_reference") -> HighLevelAction:
        return cls(source=source)

@dataclass
class DroneState:
    agent_id: int
    position_m: np.ndarray
    velocity_mps: np.ndarray
    yaw_rad: float
    yaw_rate_rad_s: float = 0.0

    def copy(self) -> DroneState:
        return DroneState(
            agent_id=self.agent_id,
            position_m=self.position_m.copy(),
            velocity_mps=self.velocity_mps.copy(),
            yaw_rad=float(self.yaw_rad),
            yaw_rate_rad_s=float(self.yaw_rate_rad_s),
        )

@dataclass(frozen=True)
class CylinderObstacle:
    obstacle_id: str
    center_xy_m: tuple[float, float]
    radius_m: float
    height_m: float

    def contains(self, position: np.ndarray, clearance_m: float = 0.0) -> bool:
        xy = position[:2] - np.asarray(self.center_xy_m)
        return bool(
            float(np.dot(xy, xy)) <= (self.radius_m + clearance_m) ** 2
            and position[2] <= self.height_m + clearance_m
        )

@dataclass(frozen=True)
class _HiddenTarget:
    """Evaluator-private target truth.  It is never placed in observations."""

    position_m: tuple[float, float, float]
    appearance: str = "high_visibility_marker"

@dataclass(frozen=True)
class PublicMission:
    bounds_xy_m: tuple[float, float]
    regions: tuple[dict[str, Any], ...]
    instruction: str
    time_budget_s: float
    target_count_disclosed: int

    def public_geometry(self, obstacles: Iterable[CylinderObstacle]) -> dict[str, Any]:
        return {
            "bounds_xy_m": list(self.bounds_xy_m),
            "obstacles": [
                {
                    "obstacle_id": obstacle.obstacle_id,
                    "center_xy_m": list(obstacle.center_xy_m),
                    "radius_m": obstacle.radius_m,
                    "height_m": obstacle.height_m,
                }
                for obstacle in obstacles
            ],
            "regions": list(self.regions),
        }

@dataclass(frozen=True)
class SensorPacket:
    agent_id: int
    sensor_time_ns: int
    rgb: np.ndarray
    distance_to_image_plane_m: np.ndarray
    semantic_segmentation: np.ndarray
    lidar_ranges_m: np.ndarray
    radar_detections: np.ndarray
    imu: np.ndarray
    camera_extrinsics_body: np.ndarray

@dataclass(frozen=True)
class CandidateEvent:
    agent_id: int
    sensor_time_ns: int
    estimated_xyz_m: tuple[float, float, float]
    confidence: float
    source: str = "online_rgbd_detector"

@dataclass(frozen=True)
class SafetyEvent:
    agent_id: int
    sim_time_ns: int
    kind: str

@dataclass(frozen=True)
class PublicObservation:
    """The only object passed to native policies.

    Sensor data are copied when exposed.  There is no target identifier,
    evaluator result, reward, future state, or random seed in this type.
    """

    agent_id: int
    sim_time_ns: int
    information_profile: str
    proprioception: np.ndarray
    public_task_state: Mapping[str, Any]
    public_team_messages: tuple[Mapping[str, Any], ...]
    high_level_action_history: tuple[tuple[float, float, float, float], ...]
    public_geometry: Mapping[str, Any] | None = None
    rgb: np.ndarray | None = None
    distance_to_image_plane_m: np.ndarray | None = None
    lidar_ranges_m: np.ndarray | None = None
    radar_detections: np.ndarray | None = None
    imu: np.ndarray | None = None
    language: str | None = None

@dataclass(frozen=True)
class RuntimeFrame:
    sim_time_ns: int
    step_index: int
    states: Mapping[int, DroneState]
    actions: Mapping[int, HighLevelAction]
    low_level_velocity_targets_mps: Mapping[int, np.ndarray]
    sensor_packets: Mapping[int, SensorPacket]
    candidate_events: tuple[CandidateEvent, ...]
    safety_events: tuple[SafetyEvent, ...]

@dataclass(frozen=True)
class EvaluationReport:
    confirmed_count: int
    target_count: int
    confirmation_precision: float
    false_confirmation_count: int
    normalized_confirmed_auc: float
    first_confirmation_latency_s: float | None
    collision_count: int
    evaluator_truth_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "confirmed_count": self.confirmed_count,
            "target_count": self.target_count,
            "confirmation_precision": self.confirmation_precision,
            "false_confirmation_count": self.false_confirmation_count,
            "normalized_confirmed_auc": self.normalized_confirmed_auc,
            "first_confirmation_latency_s": self.first_confirmation_latency_s,
            "collision_count": self.collision_count,
            "evaluator_truth_sha256": self.evaluator_truth_sha256,
        }
