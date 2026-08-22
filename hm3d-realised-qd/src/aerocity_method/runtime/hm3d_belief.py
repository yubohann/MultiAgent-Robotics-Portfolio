"""Public sparse 3D occupancy belief for HM3D exploration.

The evaluator may own a complete mesh or ESDF.  This module only stores voxels
that were produced by public sensor outcomes.  Replaying the same outcome is
idempotent, which is required for outcome-grounded replay and fragment reuse.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from aerocity_method.contracts.exploration import BeliefVersion
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier

Point3 = tuple[float, float, float]
VoxelKey = tuple[int, int, int]
UNKNOWN = 0
FREE = 1
OCCUPIED = 2
_STATE_NAMES = {UNKNOWN: "unknown", FREE: "free", OCCUPIED: "occupied"}


def public_free_voxel_transition(
    before: Iterable[VoxelKey], after: Iterable[VoxelKey]
) -> tuple[frozenset[VoxelKey], frozenset[VoxelKey]]:
    """Return newly public-free voxels and prior free voxels revised away.

    Public occupancy fusion is not monotone in the FREE state: a later
    occupied observation is allowed to override an earlier free ray.  The
    decision ledger therefore has to compare the fused maps on both sides of
    the execution boundary instead of counting only this segment's raw rays.
    """

    before_keys = frozenset(tuple(int(value) for value in key) for key in before)
    after_keys = frozenset(tuple(int(value) for value in key) for key in after)
    return after_keys - before_keys, before_keys - after_keys


def _point(values: tuple[float, float, float], name: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(finite_number(value, f"{name}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


def _distance(left: Point3, right: Point3) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


@dataclass(frozen=True, slots=True)
class PublicRangeRayOutcome:
    """One public sparse-range ray update with a stable source observation ID."""

    observation_id: str
    agent_id: str
    timestamp_s: float
    origin_m: Point3
    endpoint_m: Point3
    hit_occupied: bool

    def __post_init__(self) -> None:
        require_identifier(self.observation_id, "observation_id")
        require_identifier(self.agent_id, "agent_id")
        timestamp = finite_number(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "origin_m", _point(self.origin_m, "origin_m"))
        object.__setattr__(self, "endpoint_m", _point(self.endpoint_m, "endpoint_m"))
        if _distance(self.origin_m, self.endpoint_m) <= 1.0e-9:
            raise ValueError("range ray must have non-zero length")
        if not isinstance(self.hit_occupied, bool):
            raise ValueError("hit_occupied must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "agent_id": self.agent_id,
            "timestamp_s": self.timestamp_s,
            "origin_m": self.origin_m,
            "endpoint_m": self.endpoint_m,
            "hit_occupied": self.hit_occupied,
        }


@dataclass(frozen=True, slots=True)
class PublicRangeObservationFrameOutcome:
    """One emitted sparse-range frame, without target or image semantics."""

    observation_frame_id: str
    agent_id: str
    timestamp_s: float
    sensor_position_m: Point3
    ray_count: int

    def __post_init__(self) -> None:
        require_identifier(self.observation_frame_id, "observation_frame_id")
        require_identifier(self.agent_id, "agent_id")
        timestamp = finite_number(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        if (
            not isinstance(self.ray_count, int)
            or isinstance(self.ray_count, bool)
            or self.ray_count <= 0
        ):
            raise ValueError("ray_count must be a positive integer")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(
            self,
            "sensor_position_m",
            _point(self.sensor_position_m, "sensor_position_m"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_frame_id": self.observation_frame_id,
            "agent_id": self.agent_id,
            "timestamp_s": self.timestamp_s,
            "sensor_position_m": self.sensor_position_m,
            "ray_count": self.ray_count,
        }


@dataclass(slots=True)
class SparseVoxelBelief:
    """Method-visible sparse occupancy map derived only from public outcomes."""

    scene_id: str
    agent_id: str
    resolution_m: float
    reset_epoch: int = 0
    origin_m: Point3 = (0.0, 0.0, 0.0)
    _states: dict[VoxelKey, int] = field(default_factory=dict, init=False, repr=False)
    _outcomes: set[str] = field(default_factory=set, init=False, repr=False)
    _last_timestamp_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        require_identifier(self.scene_id, "scene_id")
        require_identifier(self.agent_id, "agent_id")
        resolution = finite_number(self.resolution_m, "resolution_m")
        if resolution <= 0.0:
            raise ValueError("resolution_m must be positive")
        if (
            not isinstance(self.reset_epoch, int)
            or isinstance(self.reset_epoch, bool)
            or self.reset_epoch < 0
        ):
            raise ValueError("reset_epoch must be a non-negative integer")
        self.resolution_m = resolution
        self.origin_m = _point(self.origin_m, "origin_m")

    def world_to_voxel(self, point_m: Point3) -> VoxelKey:
        point = _point(point_m, "point_m")
        return tuple(
            math.floor((point[index] - self.origin_m[index]) / self.resolution_m)
            for index in range(3)
        )  # type: ignore[return-value]

    def voxel_center(self, key: VoxelKey) -> Point3:
        if len(key) != 3:
            raise ValueError("voxel key must be 3D")
        return tuple(
            self.origin_m[index] + (int(key[index]) + 0.5) * self.resolution_m for index in range(3)
        )  # type: ignore[return-value]

    def state(self, key: VoxelKey) -> int:
        return self._states.get(tuple(key), UNKNOWN)

    def set_state(self, key: VoxelKey, state: int) -> None:
        if state not in {FREE, OCCUPIED}:
            raise ValueError("public belief only stores observed free/occupied voxels")
        frozen_key = tuple(int(value) for value in key)
        current = self._states.get(frozen_key, UNKNOWN)
        if current == OCCUPIED and state == FREE:
            return
        if current == FREE and state == OCCUPIED:
            # A sparse ray can graze an obstacle edge and report an occupied
            # terminal inside a voxel that an earlier pass-through ray already
            # proved free and that the continuous PhysX route guard physically
            # admitted. Exploration knowledge is monotone: a confirmed-free
            # voxel stays free so the explored-volume metric cannot shrink.
            # The exact static guard remains the authority for route safety.
            return
        self._states[frozen_key] = state

    def _ray_voxels(self, origin_m: Point3, endpoint_m: Point3) -> tuple[VoxelKey, ...]:
        distance_m = _distance(origin_m, endpoint_m)
        steps = max(1, int(math.ceil(distance_m / (self.resolution_m * 0.5))))
        keys: list[VoxelKey] = []
        seen: set[VoxelKey] = set()
        for step in range(steps + 1):
            alpha = step / steps
            point = tuple(
                origin_m[index] + alpha * (endpoint_m[index] - origin_m[index])
                for index in range(3)
            )  # type: ignore[assignment]
            key = self.world_to_voxel(point)
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return tuple(keys)

    def integrate_ray(self, outcome: PublicRangeRayOutcome) -> bool:
        """Integrate one public ray.  Returns False for an idempotent replay."""

        if outcome.observation_id in self._outcomes:
            return False
        if outcome.agent_id != self.agent_id:
            raise ValueError("outcome agent_id does not match this local belief")
        keys = self._ray_voxels(outcome.origin_m, outcome.endpoint_m)
        terminal = keys[-1]
        free_keys = keys[:-1] if outcome.hit_occupied else keys
        for key in free_keys:
            self.set_state(key, FREE)
        if outcome.hit_occupied:
            self.set_state(terminal, OCCUPIED)
        self._outcomes.add(outcome.observation_id)
        self._last_timestamp_s = max(self._last_timestamp_s, outcome.timestamp_s)
        return True

    def merge_public(self, other: SparseVoxelBelief) -> None:
        """Merge another public belief without importing private evaluator state."""

        if (
            self.scene_id != other.scene_id
            or self.resolution_m != other.resolution_m
            or self.reset_epoch != other.reset_epoch
            or self.origin_m != other.origin_m
        ):
            raise ValueError("belief maps are not compatible for public merge")
        for key, state in other._states.items():
            self.set_state(key, state)
        self._outcomes.update(other._outcomes)
        self._last_timestamp_s = max(self._last_timestamp_s, other._last_timestamp_s)

    @property
    def observed_free_count(self) -> int:
        return sum(state == FREE for state in self._states.values())

    @property
    def observed_occupied_count(self) -> int:
        return sum(state == OCCUPIED for state in self._states.values())

    @property
    def outcome_count(self) -> int:
        return len(self._outcomes)

    def occupied_keys(self) -> tuple[VoxelKey, ...]:
        return tuple(sorted(key for key, state in self._states.items() if state == OCCUPIED))

    def free_keys(self) -> tuple[VoxelKey, ...]:
        return tuple(sorted(key for key, state in self._states.items() if state == FREE))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "agent_id": self.agent_id,
            "reset_epoch": self.reset_epoch,
            "resolution_m": self.resolution_m,
            "origin_m": self.origin_m,
            "last_timestamp_s": self._last_timestamp_s,
            "outcome_count": self.outcome_count,
            "voxels": [
                {"key": key, "state": _STATE_NAMES[state]}
                for key, state in sorted(self._states.items())
            ],
        }

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_public_dict())

    def version(self) -> BeliefVersion:
        return BeliefVersion(
            scene_id=self.scene_id,
            agent_id=self.agent_id,
            reset_epoch=self.reset_epoch,
            timestamp_s=self._last_timestamp_s,
            resolution_m=self.resolution_m,
            content_sha256=self.content_sha256,
        )


__all__ = [
    "FREE",
    "OCCUPIED",
    "UNKNOWN",
    "Point3",
    "PublicRangeRayOutcome",
    "SparseVoxelBelief",
    "VoxelKey",
]
