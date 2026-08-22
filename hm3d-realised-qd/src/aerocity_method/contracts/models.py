"""Immutable public contracts for target-free HM3D exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
    to_primitive,
)
from aerocity_method.contracts.privacy import walk_public_payload

ABI_VERSION = "aerocity-hm3d-exploration-v1"
FRAGMENT_TYPES = frozenset({"transit", "observation", "hold", "communication"})
INTERACTION_KINDS = frozenset(
    {
        "temporal_overlap",
        "collision",
        "communication",
        "redundant_observation",
        "resource_competition",
    }
)

Scalar = str | int | float | bool
FeaturePairs = tuple[tuple[str, Scalar], ...]
NumericPairs = tuple[tuple[str, float], ...]
Point3 = tuple[float, float, float]


def _schema(value: str) -> str:
    if value != ABI_VERSION:
        raise ValueError(f"unsupported schema_version {value!r}; expected {ABI_VERSION!r}")
    return value


def _scalar_pairs(values: Any, name: str) -> FeaturePairs:
    pairs: list[tuple[str, Scalar]] = []
    seen: set[str] = set()
    for raw_key, raw_value in tuple(values):
        key = require_identifier(raw_key, f"{name} key")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        if isinstance(raw_value, float):
            raw_value = finite_number(raw_value, f"{name}.{key}")
        elif not isinstance(raw_value, (str, int, bool)):
            raise ValueError(f"{name}.{key} has unsupported scalar type")
        seen.add(key)
        pairs.append((key, raw_value))
    return tuple(sorted(pairs, key=lambda item: item[0]))


def _numeric_pairs(values: Any, name: str) -> NumericPairs:
    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_key, raw_value in tuple(values):
        key = require_identifier(raw_key, f"{name} key")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        seen.add(key)
        pairs.append((key, finite_number(raw_value, f"{name}.{key}")))
    return tuple(sorted(pairs, key=lambda item: item[0]))


def _points(values: Any, name: str) -> tuple[Point3, ...]:
    points: list[Point3] = []
    for index, raw in enumerate(tuple(values)):
        if len(raw) != 3:
            raise ValueError(f"{name}[{index}] must contain exactly three coordinates")
        points.append(
            tuple(finite_number(value, f"{name}[{index}]") for value in raw)  # type: ignore[arg-type]
        )
    return tuple(points)


@dataclass(frozen=True, slots=True)
class FragmentTypeSignature:
    fragment_type: str
    public_features: FeaturePairs = ()
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if self.fragment_type not in FRAGMENT_TYPES:
            raise ValueError(f"unsupported fragment_type {self.fragment_type!r}")
        object.__setattr__(
            self, "public_features", _scalar_pairs(self.public_features, "public_features")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fragment_type": self.fragment_type,
            "public_features": dict(self.public_features),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FragmentInstance:
    instance_fragment_id: str
    type_signature: FragmentTypeSignature
    episode_id: str
    decision_id: str
    agent_id: str
    planned_start: float
    planned_end: float
    path: tuple[Point3, ...] = ()
    pose_mode: str = ""
    source_observation_id: str | None = None
    sender_id: str | None = None
    receiver_id: str | None = None
    message_digest: str | None = None
    context_bucket: str = "default"
    executed: bool = False
    guard_rewritten: bool = False
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("instance_fragment_id", "episode_id", "decision_id", "agent_id"):
            require_identifier(getattr(self, name), name)
        start = finite_number(self.planned_start, "planned_start")
        end = finite_number(self.planned_end, "planned_end")
        if start < 0.0 or end < start:
            raise ValueError("fragment time window must satisfy 0 <= start <= end")
        object.__setattr__(self, "planned_start", start)
        object.__setattr__(self, "planned_end", end)
        object.__setattr__(self, "path", _points(self.path, "path"))
        require_identifier(self.context_bucket, "context_bucket")
        if self.pose_mode:
            require_identifier(self.pose_mode, "pose_mode")
        for name in ("source_observation_id", "sender_id", "receiver_id"):
            value = getattr(self, name)
            if value is not None:
                require_identifier(value, name)
        fragment_type = self.type_signature.fragment_type
        if fragment_type == "transit" and len(self.path) < 2:
            raise ValueError("transit fragment requires at least two path points")
        if fragment_type == "hold" and len(self.path) != 1:
            raise ValueError("hold fragment requires exactly one position")
        if fragment_type == "observation" and len(self.path) != 1:
            raise ValueError("observation fragment requires exactly one observation position")
        if fragment_type == "communication":
            if self.sender_id is None or self.receiver_id is None or self.message_digest is None:
                raise ValueError(
                    "communication fragment requires sender, receiver and message digest"
                )
            if self.sender_id == self.receiver_id:
                raise ValueError("communication sender and receiver must differ")
            require_sha256(self.message_digest, "message_digest")
        elif any(
            value is not None for value in (self.sender_id, self.receiver_id, self.message_digest)
        ):
            raise ValueError("communication-only fields used by non-communication fragment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_fragment_id": self.instance_fragment_id,
            "type_signature": self.type_signature.to_dict(),
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "planned_start": self.planned_start,
            "planned_end": self.planned_end,
            "path": self.path,
            "pose_mode": self.pose_mode,
            "source_observation_id": self.source_observation_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_digest": self.message_digest,
            "context_bucket": self.context_bucket,
            "executed": self.executed,
            "guard_rewritten": self.guard_rewritten,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class PublicMethodContext:
    context_id: str
    episode_id: str
    decision_id: str
    agent_features: tuple[tuple[str, tuple[float, ...]], ...]
    public_features: FeaturePairs = ()
    preferences: NumericPairs = ()
    budget: NumericPairs = ()
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("context_id", "episode_id", "decision_id"):
            require_identifier(getattr(self, name), name)
        agents: list[tuple[str, tuple[float, ...]]] = []
        seen: set[str] = set()
        for raw_agent, raw_values in tuple(self.agent_features):
            agent = require_identifier(raw_agent, "agent_id")
            if agent in seen:
                raise ValueError(f"duplicate agent_id {agent!r}")
            values = tuple(
                finite_number(value, f"agent_features.{agent}") for value in tuple(raw_values)
            )
            if not values:
                raise ValueError("agent feature vectors must not be empty")
            seen.add(agent)
            agents.append((agent, values))
        if not agents:
            raise ValueError("PublicMethodContext requires at least one agent")
        object.__setattr__(self, "agent_features", tuple(sorted(agents)))
        object.__setattr__(
            self, "public_features", _scalar_pairs(self.public_features, "public_features")
        )
        object.__setattr__(self, "preferences", _numeric_pairs(self.preferences, "preferences"))
        object.__setattr__(self, "budget", _numeric_pairs(self.budget, "budget"))
        walk_public_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_id": self.context_id,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "agent_features": [
                {"agent_id": agent, "features": features} for agent, features in self.agent_features
            ],
            "public_features": dict(self.public_features),
            "preferences": dict(self.preferences),
            "budget": dict(self.budget),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CandidateFragmentManifest:
    candidate_id: str
    context_hash: str
    fragments: tuple[FragmentInstance, ...]
    planned_descriptor: tuple[float, ...]
    feasible: bool
    quality_hint: float = 0.0
    cost_hint: float = 0.0
    source: str = "deterministic"
    admission_reasons: tuple[str, ...] = ()
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.candidate_id, "candidate_id")
        require_sha256(self.context_hash, "context_hash")
        require_identifier(self.source, "source")
        if not isinstance(self.feasible, bool):
            raise ValueError("candidate feasibility must be boolean")
        fragments = tuple(self.fragments)
        if not fragments:
            raise ValueError("candidate manifest requires at least one fragment")
        ids = [fragment.instance_fragment_id for fragment in fragments]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate manifest fragment IDs must be unique")
        episode_decisions = {(fragment.episode_id, fragment.decision_id) for fragment in fragments}
        if len(episode_decisions) != 1:
            raise ValueError("all candidate fragments must belong to one episode decision")
        object.__setattr__(self, "fragments", fragments)
        descriptor = tuple(
            finite_number(value, "planned_descriptor") for value in self.planned_descriptor
        )
        if len(descriptor) < 2:
            raise ValueError("planned_descriptor must contain at least two dimensions")
        object.__setattr__(self, "planned_descriptor", descriptor)
        object.__setattr__(self, "quality_hint", finite_number(self.quality_hint, "quality_hint"))
        cost = finite_number(self.cost_hint, "cost_hint")
        if cost < 0.0:
            raise ValueError("cost_hint must be non-negative")
        object.__setattr__(self, "cost_hint", cost)
        reasons = tuple(sorted(set(self.admission_reasons)))
        for reason in reasons:
            require_identifier(reason, "candidate admission reason")
        object.__setattr__(self, "admission_reasons", reasons)
        walk_public_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "context_hash": self.context_hash,
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "planned_descriptor": self.planned_descriptor,
            "feasible": self.feasible,
            "quality_hint": self.quality_hint,
            "cost_hint": self.cost_hint,
            "source": self.source,
            "admission_reasons": self.admission_reasons,
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class InteractionEdge:
    source_fragment_hash: str
    target_fragment_hash: str
    kind: str
    weight: float
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_sha256(self.source_fragment_hash, "source_fragment_hash")
        require_sha256(self.target_fragment_hash, "target_fragment_hash")
        if self.source_fragment_hash == self.target_fragment_hash:
            raise ValueError("interaction edge must connect different fragments")
        if self.kind not in INTERACTION_KINDS:
            raise ValueError(f"unsupported interaction kind {self.kind!r}")
        weight = finite_number(self.weight, "weight")
        if weight < 0.0:
            raise ValueError("interaction weight must be non-negative")
        object.__setattr__(self, "weight", weight)
        if self.source_fragment_hash > self.target_fragment_hash:
            source = self.target_fragment_hash
            target = self.source_fragment_hash
            object.__setattr__(self, "source_fragment_hash", source)
            object.__setattr__(self, "target_fragment_hash", target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fragment_hash": self.source_fragment_hash,
            "target_fragment_hash": self.target_fragment_hash,
            "kind": self.kind,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class CandidateGraphBatch:
    candidate_hashes: tuple[str, ...]
    fragment_hashes: tuple[str, ...]
    membership_edges: tuple[tuple[str, str, int], ...]
    interaction_edges: tuple[InteractionEdge, ...]
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        candidates = tuple(sorted(set(self.candidate_hashes)))
        fragments = tuple(sorted(set(self.fragment_hashes)))
        if not candidates or not fragments:
            raise ValueError("candidate graph must contain candidates and fragments")
        for digest in candidates:
            require_sha256(digest, "candidate_hash")
        for digest in fragments:
            require_sha256(digest, "fragment_hash")
        edges = tuple(sorted(tuple(self.membership_edges)))
        for candidate_hash, fragment_hash, order in edges:
            if candidate_hash not in candidates or fragment_hash not in fragments:
                raise ValueError("membership edge references an unknown node")
            if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                raise ValueError("membership edge order must be a non-negative integer")
        interactions = tuple(
            sorted(
                tuple(self.interaction_edges),
                key=lambda edge: (
                    edge.source_fragment_hash,
                    edge.target_fragment_hash,
                    edge.kind,
                    edge.weight,
                ),
            )
        )
        object.__setattr__(self, "candidate_hashes", candidates)
        object.__setattr__(self, "fragment_hashes", fragments)
        object.__setattr__(self, "membership_edges", edges)
        object.__setattr__(self, "interaction_edges", interactions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_hashes": self.candidate_hashes,
            "fragment_hashes": self.fragment_hashes,
            "membership_edges": self.membership_edges,
            "interaction_edges": [edge.to_dict() for edge in self.interaction_edges],
        }

    @property
    def graph_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActionToken:
    token_id: str
    episode_id: str
    decision_id: str
    context_hash: str
    manifest_hash: str
    legal_mask_hash: str
    planned_fragment_hashes: tuple[str, ...]
    issued_at: float
    duration: float
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("token_id", "episode_id", "decision_id"):
            require_identifier(getattr(self, name), name)
        for name in ("context_hash", "manifest_hash", "legal_mask_hash"):
            require_sha256(getattr(self, name), name)
        fragment_hashes = tuple(self.planned_fragment_hashes)
        if not fragment_hashes:
            raise ValueError("ActionToken requires planned fragment hashes")
        for digest in fragment_hashes:
            require_sha256(digest, "planned_fragment_hash")
        object.__setattr__(self, "planned_fragment_hashes", fragment_hashes)
        issued_at = finite_number(self.issued_at, "issued_at")
        duration = finite_number(self.duration, "duration")
        if issued_at < 0.0 or duration <= 0.0:
            raise ValueError("ActionToken requires issued_at >= 0 and duration > 0")
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "duration", duration)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(
            {
                "schema_version": self.schema_version,
                "token_id": self.token_id,
                "episode_id": self.episode_id,
                "decision_id": self.decision_id,
                "context_hash": self.context_hash,
                "manifest_hash": self.manifest_hash,
                "legal_mask_hash": self.legal_mask_hash,
                "planned_fragment_hashes": self.planned_fragment_hashes,
                "issued_at": self.issued_at,
                "duration": self.duration,
            }
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FragmentOutcome:
    outcome_id: str
    token_hash: str
    manifest_hash: str
    episode_id: str
    decision_id: str
    agent_id: str
    planned_fragment_hash: str
    executed: bool
    actual_start: float
    actual_end: float
    applied_fragment: FragmentInstance | None = None
    outcome_fields: NumericPairs = ()
    cost_fields: NumericPairs = ()
    source_observation_id: str | None = None
    source_observation_episode_id: str | None = None
    source_observation_agent_id: str | None = None
    range_ok: bool | None = None
    fov_ok: bool | None = None
    los_ok: bool | None = None
    orientation_ok: bool | None = None
    dwell_ok: bool | None = None
    link_window_ok: bool | None = None
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("outcome_id", "episode_id", "decision_id", "agent_id"):
            require_identifier(getattr(self, name), name)
        for name in ("token_hash", "manifest_hash", "planned_fragment_hash"):
            require_sha256(getattr(self, name), name)
        start = finite_number(self.actual_start, "actual_start")
        end = finite_number(self.actual_end, "actual_end")
        if start < 0.0 or end < start:
            raise ValueError("outcome time window must satisfy 0 <= start <= end")
        object.__setattr__(self, "actual_start", start)
        object.__setattr__(self, "actual_end", end)
        object.__setattr__(self, "outcome_fields", _numeric_pairs(self.outcome_fields, "outcome"))
        object.__setattr__(self, "cost_fields", _numeric_pairs(self.cost_fields, "cost"))
        if self.executed and self.applied_fragment is None:
            raise ValueError("executed outcome requires applied_fragment")
        if not self.executed and self.applied_fragment is not None:
            raise ValueError("unexecuted outcome must not include applied_fragment")
        if not self.executed and (self.outcome_fields or self.cost_fields):
            raise ValueError("unexecuted outcome must not contain outcome labels")
        observation_binding = (
            self.source_observation_id,
            self.source_observation_episode_id,
            self.source_observation_agent_id,
        )
        if self.source_observation_id is not None:
            for name, value in zip(
                (
                    "source_observation_id",
                    "source_observation_episode_id",
                    "source_observation_agent_id",
                ),
                observation_binding,
                strict=True,
            ):
                require_identifier(value, name)  # type: ignore[arg-type]
        elif any(value is not None for value in observation_binding[1:]):
            raise ValueError("source observation identity requires source_observation_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome_id": self.outcome_id,
            "token_hash": self.token_hash,
            "manifest_hash": self.manifest_hash,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "planned_fragment_hash": self.planned_fragment_hash,
            "executed": self.executed,
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "applied_fragment": (
                None if self.applied_fragment is None else self.applied_fragment.to_dict()
            ),
            "outcome_fields": dict(self.outcome_fields),
            "cost_fields": dict(self.cost_fields),
            "source_observation_id": self.source_observation_id,
            "source_observation_episode_id": self.source_observation_episode_id,
            "source_observation_agent_id": self.source_observation_agent_id,
            "range_ok": self.range_ok,
            "fov_ok": self.fov_ok,
            "los_ok": self.los_ok,
            "orientation_ok": self.orientation_ok,
            "dwell_ok": self.dwell_ok,
            "link_window_ok": self.link_window_ok,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FragmentReplayRecord:
    instance_fragment_id: str
    fragment_type_hash: str
    outcome_hash: str
    context_hash: str
    labels: NumericPairs
    costs: NumericPairs = ()
    weight: float = 1.0
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.instance_fragment_id, "instance_fragment_id")
        for name in ("fragment_type_hash", "outcome_hash", "context_hash"):
            require_sha256(getattr(self, name), name)
        labels = _numeric_pairs(self.labels, "labels")
        if not labels:
            raise ValueError("fragment replay record requires at least one label")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "costs", _numeric_pairs(self.costs, "costs"))
        weight = finite_number(self.weight, "weight")
        if weight <= 0.0:
            raise ValueError("weight must be positive")
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_fragment_id": self.instance_fragment_id,
            "fragment_type_hash": self.fragment_type_hash,
            "outcome_hash": self.outcome_hash,
            "context_hash": self.context_hash,
            "labels": dict(self.labels),
            "costs": dict(self.costs),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FragmentReplayRecord:
        return cls(
            instance_fragment_id=payload["instance_fragment_id"],
            fragment_type_hash=payload["fragment_type_hash"],
            outcome_hash=payload["outcome_hash"],
            context_hash=payload["context_hash"],
            labels=tuple(payload["labels"].items()),
            costs=tuple(payload.get("costs", {}).items()),
            weight=payload.get("weight", 1.0),
            schema_version=payload.get("schema_version", ABI_VERSION),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceDecision:
    allowed: bool
    reason_code: str
    evidence_hash: str | None = None
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        require_identifier(self.reason_code, "reason_code")
        if self.allowed and self.reason_code != "ALLOW":
            raise ValueError("allowed provenance decision must use ALLOW reason")
        if not self.allowed and self.reason_code == "ALLOW":
            raise ValueError("denied provenance decision cannot use ALLOW reason")
        if self.evidence_hash is not None:
            require_sha256(self.evidence_hash, "evidence_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    limits: NumericPairs
    used: NumericPairs = ()
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        limits = _numeric_pairs(self.limits, "limits")
        used = _numeric_pairs(self.used, "used")
        if any(value < 0.0 for _, value in limits + used):
            raise ValueError("budget limits and usage must be non-negative")
        limit_map = dict(limits)
        for key, value in used:
            if key not in limit_map:
                raise ValueError(f"budget usage {key!r} has no registered limit")
            if value > limit_map[key]:
                raise ValueError(f"budget usage {key!r} exceeds its limit")
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "used", used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "limits": dict(self.limits),
            "used": dict(self.used),
        }
