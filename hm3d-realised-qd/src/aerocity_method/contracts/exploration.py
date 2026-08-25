"""Public contracts for HM3D multi-UAV online exploration.

These contracts are deliberately independent from evaluator geometry.  They
carry hashes, public map summaries, guarded trajectories, and execution
outcomes, but never complete meshes, private ESDFs, or truth coverage maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
)
from aerocity_method.contracts.privacy import walk_public_payload

EXPLORATION_SCHEMA_VERSION = "hm3d-multi-uav-exploration-v1"
Point3 = tuple[float, float, float]
NumericPairs = tuple[tuple[str, float], ...]
_ROLES = frozenset({"explore", "relay", "rendezvous", "return", "hold"})
_TERMINAL_STATUSES = frozenset({"RUNNING", "SUCCESS", "BUDGET_EXHAUSTED", "UNRECOVERABLE_FAILURE"})


def _schema(value: str) -> None:
    if value != EXPLORATION_SCHEMA_VERSION:
        raise ValueError("exploration schema version mismatch")


def _point(values: tuple[float, ...], name: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(finite_number(value, name) for value in values)  # type: ignore[return-value]


def _points(values: tuple[tuple[float, ...], ...], name: str) -> tuple[Point3, ...]:
    return tuple(_point(tuple(value), f"{name}[{index}]") for index, value in enumerate(values))


def _nonnegative(value: float, name: str) -> float:
    resolved = finite_number(value, name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _numeric_pairs(values: NumericPairs, name: str) -> NumericPairs:
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    for key, raw_value in values:
        require_identifier(key, f"{name} key")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        seen.add(key)
        rows.append((key, finite_number(raw_value, f"{name}.{key}")))
    return tuple(sorted(rows))


@dataclass(frozen=True, slots=True)
class BeliefVersion:
    scene_id: str
    agent_id: str
    reset_epoch: int
    timestamp_s: float
    resolution_m: float
    content_sha256: str
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.scene_id, "scene_id")
        require_identifier(self.agent_id, "agent_id")
        if (
            not isinstance(self.reset_epoch, int)
            or isinstance(self.reset_epoch, bool)
            or self.reset_epoch < 0
        ):
            raise ValueError("reset_epoch must be a non-negative integer")
        timestamp = _nonnegative(self.timestamp_s, "timestamp_s")
        resolution = finite_number(self.resolution_m, "resolution_m")
        if resolution <= 0.0:
            raise ValueError("resolution_m must be positive")
        require_sha256(self.content_sha256, "content_sha256")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "resolution_m", resolution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "agent_id": self.agent_id,
            "reset_epoch": self.reset_epoch,
            "timestamp_s": self.timestamp_s,
            "resolution_m": self.resolution_m,
            "content_sha256": self.content_sha256,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MapDeltaMessage:
    message_id: str
    source_agent_id: str
    destination_agent_id: str
    created_timestamp_s: float
    time_to_live_s: float
    payload_bytes: int
    belief_version_sha256: str
    delta_sha256: str
    delivered_timestamp_s: float | None = None
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("message_id", "source_agent_id", "destination_agent_id"):
            require_identifier(getattr(self, name), name)
        if self.source_agent_id == self.destination_agent_id:
            raise ValueError("map delta sender and receiver must differ")
        created = _nonnegative(self.created_timestamp_s, "created_timestamp_s")
        ttl = finite_number(self.time_to_live_s, "time_to_live_s")
        if ttl <= 0.0:
            raise ValueError("time_to_live_s must be positive")
        if (
            not isinstance(self.payload_bytes, int)
            or isinstance(self.payload_bytes, bool)
            or self.payload_bytes < 0
        ):
            raise ValueError("payload_bytes must be a non-negative integer")
        for name in ("belief_version_sha256", "delta_sha256"):
            require_sha256(getattr(self, name), name)
        delivered = self.delivered_timestamp_s
        if delivered is not None:
            delivered = _nonnegative(delivered, "delivered_timestamp_s")
            if delivered < created:
                raise ValueError("map delta cannot arrive before creation")
            object.__setattr__(self, "delivered_timestamp_s", delivered)
        object.__setattr__(self, "created_timestamp_s", created)
        object.__setattr__(self, "time_to_live_s", ttl)

    @property
    def delivered(self) -> bool:
        return self.delivered_timestamp_s is not None

    @property
    def age_of_information_s(self) -> float | None:
        if self.delivered_timestamp_s is None:
            return None
        return self.delivered_timestamp_s - self.created_timestamp_s

    @property
    def expired(self) -> bool:
        age = self.age_of_information_s
        return age is not None and age > self.time_to_live_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "message_id": self.message_id,
            "source_agent_id": self.source_agent_id,
            "destination_agent_id": self.destination_agent_id,
            "created_timestamp_s": self.created_timestamp_s,
            "delivered_timestamp_s": self.delivered_timestamp_s,
            "time_to_live_s": self.time_to_live_s,
            "payload_bytes": self.payload_bytes,
            "belief_version_sha256": self.belief_version_sha256,
            "delta_sha256": self.delta_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrontierCluster:
    frontier_id: str
    belief_version_sha256: str
    centroid_m: Point3
    outward_normal: Point3
    viewpoint_candidates_m: tuple[Point3, ...]
    unknown_voxel_count: int
    expected_gain_m3: float
    relative_height_band: int
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.frontier_id, "frontier_id")
        require_sha256(self.belief_version_sha256, "belief_version_sha256")
        object.__setattr__(self, "centroid_m", _point(self.centroid_m, "centroid_m"))
        object.__setattr__(self, "outward_normal", _point(self.outward_normal, "outward_normal"))
        viewpoints = _points(self.viewpoint_candidates_m, "viewpoint_candidates_m")
        if not viewpoints:
            raise ValueError("frontier cluster requires at least one public viewpoint")
        object.__setattr__(self, "viewpoint_candidates_m", viewpoints)
        if (
            not isinstance(self.unknown_voxel_count, int)
            or isinstance(self.unknown_voxel_count, bool)
            or self.unknown_voxel_count < 1
        ):
            raise ValueError("unknown_voxel_count must be a positive integer")
        gain = finite_number(self.expected_gain_m3, "expected_gain_m3")
        if gain <= 0.0:
            raise ValueError("expected_gain_m3 must be positive")
        if not isinstance(self.relative_height_band, int) or isinstance(
            self.relative_height_band, bool
        ):
            raise ValueError("relative_height_band must be an integer")
        object.__setattr__(self, "expected_gain_m3", gain)
        walk_public_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frontier_id": self.frontier_id,
            "belief_version_sha256": self.belief_version_sha256,
            "centroid_m": self.centroid_m,
            "outward_normal": self.outward_normal,
            "viewpoint_candidates_m": self.viewpoint_candidates_m,
            "unknown_voxel_count": self.unknown_voxel_count,
            "expected_gain_m3": self.expected_gain_m3,
            "relative_height_band": self.relative_height_band,
        }


@dataclass(frozen=True, slots=True)
class AgentExplorationPlan:
    agent_id: str
    role: str
    trajectory_m: tuple[Point3, ...]
    duration_s: float
    expected_gain_m3: float
    risk: float
    energy_j: float
    frontier_id: str | None = None
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.agent_id, "agent_id")
        if self.role not in _ROLES:
            raise ValueError("unsupported exploration role")
        path = _points(self.trajectory_m, "trajectory_m")
        if not path:
            raise ValueError("agent plan requires a trajectory")
        if self.role not in {"hold", "relay"} and len(path) < 2:
            raise ValueError("moving exploration roles require at least two trajectory points")
        object.__setattr__(self, "trajectory_m", path)
        duration = finite_number(self.duration_s, "duration_s")
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(
            self, "expected_gain_m3", _nonnegative(self.expected_gain_m3, "expected_gain_m3")
        )
        risk = finite_number(self.risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "energy_j", _nonnegative(self.energy_j, "energy_j"))
        if self.frontier_id is not None:
            require_identifier(self.frontier_id, "frontier_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "role": self.role,
            "trajectory_m": self.trajectory_m,
            "duration_s": self.duration_s,
            "expected_gain_m3": self.expected_gain_m3,
            "risk": self.risk,
            "energy_j": self.energy_j,
            "frontier_id": self.frontier_id,
        }


@dataclass(frozen=True, slots=True)
class TeamExplorationCandidate:
    candidate_id: str
    context_sha256: str
    belief_version_sha256s: tuple[str, ...]
    agent_plans: tuple[AgentExplorationPlan, ...]
    planned_descriptor: tuple[float, ...]
    feasible: bool
    admission_reasons: tuple[str, ...] = ()
    source: str = "public-frontier"
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.candidate_id, "candidate_id")
        require_identifier(self.source, "source")
        require_sha256(self.context_sha256, "context_sha256")
        belief_hashes = tuple(sorted(set(self.belief_version_sha256s)))
        if not belief_hashes:
            raise ValueError("team candidate requires at least one belief version")
        for digest in belief_hashes:
            require_sha256(digest, "belief_version_sha256")
        object.__setattr__(self, "belief_version_sha256s", belief_hashes)
        plans = tuple(sorted(self.agent_plans, key=lambda row: row.agent_id))
        if not plans or len({row.agent_id for row in plans}) != len(plans):
            raise ValueError("team candidate requires unique agent plans")
        object.__setattr__(self, "agent_plans", plans)
        descriptor = tuple(
            finite_number(value, "planned_descriptor") for value in self.planned_descriptor
        )
        if len(descriptor) < 2:
            raise ValueError("planned_descriptor requires at least two dimensions")
        object.__setattr__(self, "planned_descriptor", descriptor)
        if not isinstance(self.feasible, bool):
            raise ValueError("feasible must be boolean")
        reasons = tuple(sorted(set(self.admission_reasons)))
        for reason in reasons:
            require_identifier(reason, "admission reason")
        if self.feasible and reasons:
            raise ValueError("a feasible candidate cannot contain rejection reasons")
        object.__setattr__(self, "admission_reasons", reasons)
        walk_public_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "context_sha256": self.context_sha256,
            "belief_version_sha256s": self.belief_version_sha256s,
            "agent_plans": [row.to_dict() for row in self.agent_plans],
            "planned_descriptor": self.planned_descriptor,
            "feasible": self.feasible,
            "admission_reasons": self.admission_reasons,
            "source": self.source,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class AgentExecutionOutcome:
    agent_id: str
    applied_trajectory_m: tuple[Point3, ...]
    distance_m: float
    energy_j: float
    collision_count: int
    guard_interventions: int
    aborted: bool
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.agent_id, "agent_id")
        object.__setattr__(
            self, "applied_trajectory_m", _points(self.applied_trajectory_m, "applied_trajectory_m")
        )
        object.__setattr__(self, "distance_m", _nonnegative(self.distance_m, "distance_m"))
        object.__setattr__(self, "energy_j", _nonnegative(self.energy_j, "energy_j"))
        for name in ("collision_count", "guard_interventions"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.aborted, bool):
            raise ValueError("aborted must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "applied_trajectory_m": self.applied_trajectory_m,
            "distance_m": self.distance_m,
            "energy_j": self.energy_j,
            "collision_count": self.collision_count,
            "guard_interventions": self.guard_interventions,
            "aborted": self.aborted,
        }


@dataclass(frozen=True, slots=True)
class ExplorationExecutionOutcome:
    outcome_id: str
    episode_id: str
    decision_id: str
    candidate_sha256: str
    started_timestamp_s: float
    ended_timestamp_s: float
    agent_outcomes: tuple[AgentExecutionOutcome, ...]
    observation_ids: tuple[str, ...]
    delivered_message_ids: tuple[str, ...]
    observed_free_delta: int
    observed_occupied_delta: int
    realised_descriptor: tuple[float, ...]
    unrecoverable_failure: bool = False
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("outcome_id", "episode_id", "decision_id"):
            require_identifier(getattr(self, name), name)
        require_sha256(self.candidate_sha256, "candidate_sha256")
        start = _nonnegative(self.started_timestamp_s, "started_timestamp_s")
        end = _nonnegative(self.ended_timestamp_s, "ended_timestamp_s")
        if end <= start:
            raise ValueError("outcome end must be after start")
        object.__setattr__(self, "started_timestamp_s", start)
        object.__setattr__(self, "ended_timestamp_s", end)
        outcomes = tuple(sorted(self.agent_outcomes, key=lambda row: row.agent_id))
        if not outcomes or len({row.agent_id for row in outcomes}) != len(outcomes):
            raise ValueError("execution outcome requires unique agent outcomes")
        object.__setattr__(self, "agent_outcomes", outcomes)
        for name in ("observation_ids", "delivered_message_ids"):
            values = tuple(sorted(set(getattr(self, name))))
            for value in values:
                require_identifier(value, name)
            object.__setattr__(self, name, values)
        for name in ("observed_free_delta", "observed_occupied_delta"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        descriptor = tuple(
            finite_number(value, "realised_descriptor") for value in self.realised_descriptor
        )
        if len(descriptor) < 2:
            raise ValueError("realised_descriptor requires at least two dimensions")
        object.__setattr__(self, "realised_descriptor", descriptor)
        if not isinstance(self.unrecoverable_failure, bool):
            raise ValueError("unrecoverable_failure must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome_id": self.outcome_id,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "candidate_sha256": self.candidate_sha256,
            "started_timestamp_s": self.started_timestamp_s,
            "ended_timestamp_s": self.ended_timestamp_s,
            "agent_outcomes": [row.to_dict() for row in self.agent_outcomes],
            "observation_ids": self.observation_ids,
            "delivered_message_ids": self.delivered_message_ids,
            "observed_free_delta": self.observed_free_delta,
            "observed_occupied_delta": self.observed_occupied_delta,
            "realised_descriptor": self.realised_descriptor,
            "unrecoverable_failure": self.unrecoverable_failure,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExplorationTransition:
    episode_id: str
    decision_id: str
    state_sha256: str
    candidate_set_sha256: str
    selected_candidate_sha256: str
    outcome_sha256: str
    next_state_sha256: str
    duration_s: float
    reward_features: NumericPairs
    cost_features: NumericPairs
    done: bool
    terminal_status: str = "RUNNING"
    schema_version: str = EXPLORATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("episode_id", "decision_id"):
            require_identifier(getattr(self, name), name)
        for name in (
            "state_sha256",
            "candidate_set_sha256",
            "selected_candidate_sha256",
            "outcome_sha256",
            "next_state_sha256",
        ):
            require_sha256(getattr(self, name), name)
        duration = finite_number(self.duration_s, "duration_s")
        if duration <= 0.0:
            raise ValueError("transition duration must be positive")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(
            self, "reward_features", _numeric_pairs(self.reward_features, "reward_features")
        )
        object.__setattr__(
            self, "cost_features", _numeric_pairs(self.cost_features, "cost_features")
        )
        if not isinstance(self.done, bool):
            raise ValueError("done must be boolean")
        if self.terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("unsupported terminal status")
        if self.done != (self.terminal_status != "RUNNING"):
            raise ValueError("done and terminal_status disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "state_sha256": self.state_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "outcome_sha256": self.outcome_sha256,
            "next_state_sha256": self.next_state_sha256,
            "duration_s": self.duration_s,
            "reward_features": dict(self.reward_features),
            "cost_features": dict(self.cost_features),
            "done": self.done,
            "terminal_status": self.terminal_status,
        }


__all__ = [
    "EXPLORATION_SCHEMA_VERSION",
    "AgentExecutionOutcome",
    "AgentExplorationPlan",
    "BeliefVersion",
    "ExplorationExecutionOutcome",
    "ExplorationTransition",
    "FrontierCluster",
    "MapDeltaMessage",
    "Point3",
    "TeamExplorationCandidate",
]
