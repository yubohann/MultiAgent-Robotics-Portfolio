"""Sensor entitlement, scheduling and throughput contracts for multi-UAV runs.

The simulator and evaluator may use private collision geometry, but policy code
only receives observations produced under a frozen public sensor entitlement.
This module deliberately separates *availability* from *consumption*: a baseline
may ignore an available modality, but it may not receive a different sensor
budget from the proposed method in the same comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
)
from aerocity_method.contracts.privacy import walk_public_payload

SENSOR_CONTRACT_SCHEMA_VERSION = "multi-uav-sensor-contract-v1"
FORMAL_H15_SENSOR_PILOT_MODES = ("physics_only", "sparse_range_3d")
# HM3D's active contract is camera-free.  Historical camera pilots live in the
# dated archive and cannot be scheduled through this module.
SENSOR_PILOT_MODES = FORMAL_H15_SENSOR_PILOT_MODES
_PHASES = frozenset({"transit", "observe", "dwell", "map_update"})
_DROP_POLICIES = frozenset({"block", "drop_oldest", "mark_missing"})
_PRIVATE_POLICY_FIELDS = frozenset(
    {
        "complete_mesh",
        "evaluator_esdf",
        "private_esdf",
        "private_geometry",
        "target_coordinates",
        "target_distance",
        "target_truth",
        "truth_map",
    }
)


def _nonnegative(value: float, name: str) -> float:
    resolved = finite_number(value, name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


@dataclass(frozen=True, slots=True)
class SensorProfile:
    """A method-visible sensor entitlement and deterministic render schedule."""

    profile_id: str
    mode: str
    update_hz: float
    allowed_phases: tuple[str, ...]
    public_fields: tuple[str, ...]
    width: int = 0
    height: int = 0
    history_frames: int = 1
    drop_policy: str = "mark_missing"
    rgb_enabled: bool = False
    depth_enabled: bool = False
    range_enabled: bool = False
    schema_version: str = SENSOR_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SENSOR_CONTRACT_SCHEMA_VERSION:
            raise ValueError("sensor contract schema version mismatch")
        require_identifier(self.profile_id, "profile_id")
        if self.mode not in SENSOR_PILOT_MODES:
            raise ValueError("unsupported sensor pilot mode")
        update_hz = _nonnegative(self.update_hz, "update_hz")
        object.__setattr__(self, "update_hz", update_hz)
        phases = tuple(sorted(set(self.allowed_phases)))
        if any(phase not in _PHASES for phase in phases):
            raise ValueError("sensor profile contains an unsupported phase")
        object.__setattr__(self, "allowed_phases", phases)
        fields = tuple(sorted(set(self.public_fields)))
        if any(not isinstance(field, str) or not field.strip() for field in fields):
            raise ValueError("public sensor fields must be non-empty strings")
        forbidden = {field.casefold() for field in fields} & _PRIVATE_POLICY_FIELDS
        if forbidden:
            raise ValueError(
                f"private geometry/truth cannot enter a sensor profile: {sorted(forbidden)}"
            )
        walk_public_payload({field: True for field in fields})
        object.__setattr__(self, "public_fields", fields)
        for name in ("width", "height", "history_frames"):
            value = getattr(self, name)
            minimum = 1 if name == "history_frames" else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.drop_policy not in _DROP_POLICIES:
            raise ValueError("unsupported sensor drop policy")
        for name in ("rgb_enabled", "depth_enabled", "range_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

        camera_enabled = self.rgb_enabled or self.depth_enabled
        if camera_enabled != (self.width > 0 and self.height > 0):
            raise ValueError("camera dimensions and RGB/depth availability must agree")
        if self.mode == "physics_only":
            if update_hz != 0.0 or phases or fields or camera_enabled or self.range_enabled:
                raise ValueError("physics_only cannot expose a method-visible sensor")
        else:
            if update_hz <= 0.0 or not phases or not fields:
                raise ValueError("active sensor profiles need rate, phase and public fields")
        if self.mode == "sparse_range_3d" and (not self.range_enabled or camera_enabled):
            raise ValueError("sparse_range_3d must use range without a camera")

    @property
    def entitlement_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensorProfile:
        expected = {
            "schema_version",
            "profile_id",
            "mode",
            "update_hz",
            "allowed_phases",
            "public_fields",
            "width",
            "height",
            "history_frames",
            "drop_policy",
            "rgb_enabled",
            "depth_enabled",
            "range_enabled",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("sensor profile fields mismatch")
        return cls(
            schema_version=str(payload["schema_version"]),
            profile_id=str(payload["profile_id"]),
            mode=str(payload["mode"]),
            update_hz=payload["update_hz"],
            allowed_phases=tuple(payload["allowed_phases"]),
            public_fields=tuple(payload["public_fields"]),
            width=payload["width"],
            height=payload["height"],
            history_frames=payload["history_frames"],
            drop_policy=str(payload["drop_policy"]),
            rgb_enabled=payload["rgb_enabled"],
            depth_enabled=payload["depth_enabled"],
            range_enabled=payload["range_enabled"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "mode": self.mode,
            "update_hz": self.update_hz,
            "allowed_phases": self.allowed_phases,
            "public_fields": self.public_fields,
            "width": self.width,
            "height": self.height,
            "history_frames": self.history_frames,
            "drop_policy": self.drop_policy,
            "rgb_enabled": self.rgb_enabled,
            "depth_enabled": self.depth_enabled,
            "range_enabled": self.range_enabled,
        }

    def due(self, *, phase: str, elapsed_since_frame_s: float) -> bool:
        """Return whether a public sensor frame is due without touching private state."""

        if phase not in _PHASES:
            raise ValueError("unsupported runtime phase")
        elapsed = _nonnegative(elapsed_since_frame_s, "elapsed_since_frame_s")
        if self.mode == "physics_only" or phase not in self.allowed_phases:
            return False
        return elapsed + 1e-12 >= 1.0 / self.update_hz


@dataclass(frozen=True, slots=True)
class SensorEntitlement:
    method_id: str
    profile_hash: str

    def __post_init__(self) -> None:
        require_identifier(self.method_id, "method_id")
        require_sha256(self.profile_hash, "profile_hash")


@dataclass(frozen=True, slots=True)
class SensorFairnessAdmission:
    """Prove equal sensor availability for every method in one comparison."""

    profile: SensorProfile
    entitlements: tuple[SensorEntitlement, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.entitlements)
        if len(rows) < 2:
            raise ValueError("sensor fairness needs at least two compared methods")
        method_ids = [row.method_id for row in rows]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("sensor entitlements must contain unique methods")
        mismatched = [
            row.method_id for row in rows if row.profile_hash != self.profile.entitlement_hash
        ]
        if mismatched:
            raise ValueError(f"methods have unequal sensor entitlements: {sorted(mismatched)}")


@dataclass(frozen=True, slots=True)
class SensorThroughputRecord:
    """One measured H15 throughput row; it is not a task-quality result."""

    comparison_id: str
    scene_id: str
    episode_id: str
    fleet_size: int
    profile: SensorProfile
    physics_dt_s: float
    planned_episodes: int
    executed_episodes: int
    failed_episodes: int
    physics_real_time_factor: float
    environment_steps_per_s: float
    sensor_frames_per_s: float
    render_time_s: float
    transfer_time_s: float
    gpu_memory_mb: float
    cpu_memory_mb: float
    dropped_frames: int
    observations_per_agent: tuple[int, ...]
    measurement_scope: str
    wall_clock_s: float

    def __post_init__(self) -> None:
        for name in ("comparison_id", "scene_id", "episode_id"):
            require_identifier(getattr(self, name), name)
        if self.fleet_size != FORMAL_FLEET_SIZE:
            raise ValueError(f"formal sensor pilot fleet_size must be {FORMAL_FLEET_SIZE}")
        for name in ("planned_episodes", "executed_episodes", "failed_episodes", "dropped_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.executed_episodes + self.failed_episodes != self.planned_episodes:
            raise ValueError("sensor pilot failure denominator is incomplete")
        if self.planned_episodes < 1:
            raise ValueError("sensor pilot must contain at least one planned episode")
        for name in (
            "physics_dt_s",
            "physics_real_time_factor",
            "environment_steps_per_s",
            "sensor_frames_per_s",
            "render_time_s",
            "transfer_time_s",
            "gpu_memory_mb",
            "cpu_memory_mb",
            "wall_clock_s",
        ):
            value = _nonnegative(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        if self.measurement_scope != "throughput_only":
            raise ValueError("H15 records must be limited to throughput_only measurement")
        observations = tuple(self.observations_per_agent)
        if len(observations) != self.fleet_size or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in observations
        ):
            raise ValueError("observations_per_agent must cover every UAV")
        if self.profile.mode == "physics_only":
            if any(observations) or self.sensor_frames_per_s != 0.0:
                raise ValueError("physics_only pilot cannot report sensor frames")
        elif self.executed_episodes > 0 and not any(observations):
            raise ValueError("an executed active-sensor pilot needs actual observations")
        object.__setattr__(self, "observations_per_agent", observations)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensorThroughputRecord:
        expected = {
            "comparison_id",
            "scene_id",
            "episode_id",
            "fleet_size",
            "profile",
            "physics_dt_s",
            "planned_episodes",
            "executed_episodes",
            "failed_episodes",
            "physics_real_time_factor",
            "environment_steps_per_s",
            "sensor_frames_per_s",
            "render_time_s",
            "transfer_time_s",
            "gpu_memory_mb",
            "cpu_memory_mb",
            "dropped_frames",
            "observations_per_agent",
            "measurement_scope",
            "wall_clock_s",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("sensor throughput record fields mismatch")
        profile_payload = payload["profile"]
        if not isinstance(profile_payload, dict):
            raise ValueError("sensor throughput profile must be an object")
        return cls(
            comparison_id=str(payload["comparison_id"]),
            scene_id=str(payload["scene_id"]),
            episode_id=str(payload["episode_id"]),
            fleet_size=payload["fleet_size"],
            profile=SensorProfile.from_dict(profile_payload),
            physics_dt_s=payload["physics_dt_s"],
            planned_episodes=payload["planned_episodes"],
            executed_episodes=payload["executed_episodes"],
            failed_episodes=payload["failed_episodes"],
            physics_real_time_factor=payload["physics_real_time_factor"],
            environment_steps_per_s=payload["environment_steps_per_s"],
            sensor_frames_per_s=payload["sensor_frames_per_s"],
            render_time_s=payload["render_time_s"],
            transfer_time_s=payload["transfer_time_s"],
            gpu_memory_mb=payload["gpu_memory_mb"],
            cpu_memory_mb=payload["cpu_memory_mb"],
            dropped_frames=payload["dropped_frames"],
            observations_per_agent=tuple(payload["observations_per_agent"]),
            measurement_scope=str(payload["measurement_scope"]),
            wall_clock_s=payload["wall_clock_s"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "fleet_size": self.fleet_size,
            "profile": self.profile.to_dict(),
            "physics_dt_s": self.physics_dt_s,
            "planned_episodes": self.planned_episodes,
            "executed_episodes": self.executed_episodes,
            "failed_episodes": self.failed_episodes,
            "physics_real_time_factor": self.physics_real_time_factor,
            "environment_steps_per_s": self.environment_steps_per_s,
            "sensor_frames_per_s": self.sensor_frames_per_s,
            "render_time_s": self.render_time_s,
            "transfer_time_s": self.transfer_time_s,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cpu_memory_mb": self.cpu_memory_mb,
            "dropped_frames": self.dropped_frames,
            "observations_per_agent": self.observations_per_agent,
            "measurement_scope": self.measurement_scope,
            "wall_clock_s": self.wall_clock_s,
        }


def audit_sensor_throughput_pilot(
    records: tuple[SensorThroughputRecord, ...],
    *,
    modes: tuple[str, ...] = SENSOR_PILOT_MODES,
) -> dict[str, Any]:
    """Require the complete paired H15 matrix before freezing a sensor profile."""

    rows = tuple(records)
    modes_are_supported = all(mode in SENSOR_PILOT_MODES for mode in modes)
    if not modes or len(set(modes)) != len(modes) or not modes_are_supported:
        raise ValueError("H15 modes must be a non-empty unique subset of supported sensor modes")
    required = {(FORMAL_FLEET_SIZE, mode) for mode in modes}
    actual = {(row.fleet_size, row.profile.mode) for row in rows}
    duplicates = len(actual) != len(rows)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    comparison_ids = {row.comparison_id for row in rows}
    scene_episode = {(row.scene_id, row.episode_id) for row in rows}
    physics_steps = {row.physics_dt_s for row in rows}
    profile_hashes_by_mode = {
        mode: {row.profile.entitlement_hash for row in rows if row.profile.mode == mode}
        for mode in modes
    }
    reasons: list[str] = []
    if duplicates:
        reasons.append("DUPLICATE_FLEET_MODE_ROWS")
    if missing:
        reasons.append("MISSING_FLEET_MODE_ROWS")
    if extra:
        reasons.append("UNEXPECTED_FLEET_MODE_ROWS")
    if len(comparison_ids) != 1:
        reasons.append("COMPARISON_ID_MISMATCH")
    if len(scene_episode) != 1:
        reasons.append("SCENE_OR_EPISODE_MISMATCH")
    if len(physics_steps) != 1:
        reasons.append("PHYSICS_STEP_MISMATCH")
    if any(len(hashes) > 1 for hashes in profile_hashes_by_mode.values()):
        reasons.append("PROFILE_CHANGED_ACROSS_FORMAL_FLEET")
    return {
        "schema_version": SENSOR_CONTRACT_SCHEMA_VERSION,
        "status": "PASS" if not reasons else "RUNTIME_NOT_READY",
        "rows": len(rows),
        "required_rows": len(required),
        "missing": [{"fleet_size": fleet, "mode": mode} for fleet, mode in missing],
        "extra": [{"fleet_size": fleet, "mode": mode} for fleet, mode in extra],
        "reasons": reasons,
    }


__all__ = [
    "SENSOR_CONTRACT_SCHEMA_VERSION",
    "FORMAL_H15_SENSOR_PILOT_MODES",
    "SENSOR_PILOT_MODES",
    "SensorEntitlement",
    "SensorFairnessAdmission",
    "SensorProfile",
    "SensorThroughputRecord",
    "audit_sensor_throughput_pilot",
]
