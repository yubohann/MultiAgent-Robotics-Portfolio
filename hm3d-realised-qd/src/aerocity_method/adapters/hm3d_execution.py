"""Concurrent fragment execution contract for the HM3D exploration runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from aerocity_method.contracts.io import finite_number, require_identifier
from aerocity_method.contracts.models import (
    ActionToken,
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentOutcome,
    FragmentReplayRecord,
    ReplayDecision,
)

Point3 = tuple[float, float, float]
PHYSICAL_FAILURE_OUTCOMES = frozenset(
    {
        "collision",
        "out_of_bounds",
        "guard_intervention",
        "static_clearance_contract_violation",
        "inter_agent_separation_violation",
    }
)


def _point3(values: Sequence[float], name: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    return tuple(finite_number(value, f"{name}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


def _points(values: Sequence[Sequence[float]], name: str) -> tuple[Point3, ...]:
    return tuple(_point3(point, f"{name}[{index}]") for index, point in enumerate(values))


@dataclass(frozen=True, slots=True)
class FragmentExecutionSample:
    """Actual result for one fragment returned by a concurrent backend."""

    planned_fragment_id: str
    executed: bool
    actual_start_s: float
    actual_end_s: float
    command_path_m: tuple[Point3, ...] = ()
    actual_path_m: tuple[Point3, ...] = ()
    execution_trace_id: str = ""
    collision: bool = False
    out_of_bounds: bool = False
    guard_intervened: bool = False
    static_clearance_contract_violation: bool = False
    inter_agent_separation_violation: bool = False
    minimum_clearance_m: float = 0.0
    energy_used_j: float = 0.0
    communication_connected_at_every_telemetry_tick: bool = True
    source_observation_id: str | None = None
    source_observation_episode_id: str | None = None
    source_observation_agent_id: str | None = None
    range_ok: bool | None = None
    fov_ok: bool | None = None
    los_ok: bool | None = None
    orientation_ok: bool | None = None
    dwell_ok: bool | None = None
    link_window_ok: bool | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        require_identifier(self.planned_fragment_id, "planned_fragment_id")
        if not isinstance(self.executed, bool):
            raise ValueError("executed must be boolean")
        start = finite_number(self.actual_start_s, "actual_start_s")
        end = finite_number(self.actual_end_s, "actual_end_s")
        if start < 0.0 or end < start:
            raise ValueError("execution time window must satisfy 0 <= start <= end")
        object.__setattr__(self, "actual_start_s", start)
        object.__setattr__(self, "actual_end_s", end)
        object.__setattr__(self, "command_path_m", _points(self.command_path_m, "command_path_m"))
        object.__setattr__(self, "actual_path_m", _points(self.actual_path_m, "actual_path_m"))
        if self.execution_trace_id:
            require_identifier(self.execution_trace_id, "execution_trace_id")
        for name in ("minimum_clearance_m", "energy_used_j"):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.source_observation_id is None:
            for name in ("source_observation_episode_id", "source_observation_agent_id"):
                if getattr(self, name) is not None:
                    raise ValueError("source observation identity requires source_observation_id")
        else:
            for name in (
                "source_observation_id",
                "source_observation_episode_id",
                "source_observation_agent_id",
            ):
                require_identifier(getattr(self, name), name)  # type: ignore[arg-type]
        if self.failure_reason:
            require_identifier(self.failure_reason, "failure_reason")

    @property
    def actual_path_id(self) -> str:
        # A readable path label: which fragment flew and how many waypoints it has.
        return f"{self.planned_fragment_id}:path-{len(self.actual_path_m)}"


@runtime_checkable
class HM3DManifestExecutionBackend(Protocol):
    """Backend boundary: execute every UAV command concurrently once per decision."""

    backend_id: str
    evidence_class: str

    def execute_manifest(
        self,
        manifest: CandidateFragmentManifest,
        token: ActionToken,
    ) -> Sequence[FragmentExecutionSample]:
        """Return exactly one sample for every planned fragment."""


@dataclass(frozen=True, slots=True)
class HM3DExecutionLedger:
    """Complete execution accounting for one concurrent manifest."""

    backend_id: str
    evidence_class: str
    manifest_id: str
    token_id: str
    outcomes: tuple[FragmentOutcome, ...]
    replay_decisions: tuple[ReplayDecision, ...]
    replay_records: tuple[FragmentReplayRecord | None, ...]
    trace_ids: tuple[tuple[str, str], ...]
    actual_path_ids: tuple[tuple[str, str], ...]
    collision_count: int
    out_of_bounds_count: int
    guard_intervention_count: int
    static_clearance_contract_violation_count: int
    inter_agent_separation_violation_count: int
    executed_fragment_count: int
    failed_fragment_count: int
    total_energy_used_j: float
    minimum_clearance_m: float
    fragment_connected_at_all_telemetry_samples_fraction: float
    engineering_only: bool

    def __post_init__(self) -> None:
        require_identifier(self.backend_id, "backend_id")
        require_identifier(self.evidence_class, "evidence_class")
        require_identifier(self.manifest_id, "manifest_id")
        require_identifier(self.token_id, "token_id")
        outcomes = tuple(self.outcomes)
        decisions = tuple(self.replay_decisions)
        records = tuple(self.replay_records)
        if not outcomes or len(outcomes) != len(decisions) or len(outcomes) != len(records):
            raise ValueError("execution ledger must align each outcome, decision and replay record")
        if self.executed_fragment_count + self.failed_fragment_count != len(outcomes):
            raise ValueError("execution ledger failure denominator is incomplete")

    @property
    def reusable_fragment_count(self) -> int:
        return sum(record is not None for record in self.replay_records)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "evidence_class": self.evidence_class,
            "manifest_id": self.manifest_id,
            "token_id": self.token_id,
            "outcome_ids": [outcome.outcome_id for outcome in self.outcomes],
            "replay": [decision.to_dict() for decision in self.replay_decisions],
            "trace_ids": dict(self.trace_ids),
            "actual_path_ids": dict(self.actual_path_ids),
            "collision_count": self.collision_count,
            "out_of_bounds_count": self.out_of_bounds_count,
            "guard_intervention_count": self.guard_intervention_count,
            "static_clearance_contract_violation_count": (
                self.static_clearance_contract_violation_count
            ),
            "inter_agent_separation_violation_count": (
                self.inter_agent_separation_violation_count
            ),
            "executed_fragment_count": self.executed_fragment_count,
            "failed_fragment_count": self.failed_fragment_count,
            "reusable_fragment_count": self.reusable_fragment_count,
            "total_energy_used_j": self.total_energy_used_j,
            "minimum_clearance_m": self.minimum_clearance_m,
            "fragment_connected_at_all_telemetry_samples_fraction": (
                self.fragment_connected_at_all_telemetry_samples_fraction
            ),
            "engineering_only": self.engineering_only,
        }


def _unexecuted_sample(planned: FragmentInstance, reason: str) -> FragmentExecutionSample:
    return FragmentExecutionSample(
        planned_fragment_id=planned.instance_fragment_id,
        executed=False,
        actual_start_s=planned.planned_start,
        actual_end_s=planned.planned_start,
        execution_trace_id=f"{planned.instance_fragment_id}:{reason}",
        failure_reason=reason,
    )


def _outcomes(sample: FragmentExecutionSample) -> tuple[tuple[str, float], ...]:
    return (
        ("collision", float(sample.collision)),
        (
            "static_clearance_contract_violation",
            float(sample.static_clearance_contract_violation),
        ),
        ("inter_agent_separation_violation", float(sample.inter_agent_separation_violation)),
        (
            "execution_success",
            float(
                sample.executed
                and not sample.failure_reason
                and not sample.collision
                and not sample.out_of_bounds
                and not sample.guard_intervened
                and not sample.static_clearance_contract_violation
                and not sample.inter_agent_separation_violation
            ),
        ),
        ("guard_intervention", float(sample.guard_intervened)),
        ("out_of_bounds", float(sample.out_of_bounds)),
        ("transit_timeout", float(sample.failure_reason == "transit_timeout")),
    )


def _costs(sample: FragmentExecutionSample) -> tuple[tuple[str, float], ...]:
    return (
        ("energy_used_j", sample.energy_used_j),
        ("minimum_clearance_m", sample.minimum_clearance_m),
    )


def _replay_decision(planned: FragmentInstance, outcome: FragmentOutcome) -> ReplayDecision:
    """Decide whether a real outcome may label a reusable fragment."""
    if not outcome.executed:
        return ReplayDecision(False, "NOT_EXECUTED")
    fields = dict(outcome.outcome_fields)
    # A timeout is a real attempt whose timing cannot label a reusable transit.
    if fields.get("transit_timeout", 0.0) > 0.0:
        return ReplayDecision(False, "TRANSIT_TIMEOUT")
    if fields.get("collision", 0.0) > 0.0:
        return ReplayDecision(False, "PHYSICAL_COLLISION")
    if fields.get("out_of_bounds", 0.0) > 0.0:
        return ReplayDecision(False, "FLIGHT_BOUNDS_VIOLATION")
    if fields.get("guard_intervention", 0.0) > 0.0:
        return ReplayDecision(False, "GUARD_REWRITTEN")
    if fields.get("static_clearance_contract_violation", 0.0) > 0.0:
        return ReplayDecision(False, "STATIC_CLEARANCE_CONTRACT_VIOLATION")
    if fields.get("inter_agent_separation_violation", 0.0) > 0.0:
        return ReplayDecision(False, "INTER_AGENT_SEPARATION_VIOLATION")
    if planned.type_signature.fragment_type == "observation":
        if outcome.source_observation_id is None:
            return ReplayDecision(False, "MISSING_SOURCE_OBSERVATION_ID")
        evidence = (
            outcome.range_ok,
            outcome.fov_ok,
            outcome.los_ok,
            outcome.orientation_ok,
            outcome.dwell_ok,
        )
        if any(value is not True for value in evidence):
            return ReplayDecision(False, "OBSERVATION_EVIDENCE_INCOMPLETE")
    if planned.type_signature.fragment_type == "communication" and outcome.link_window_ok is not True:
        return ReplayDecision(False, "COMMUNICATION_LINK_UNVERIFIED")
    return ReplayDecision(True, "ALLOW")


def _replay_record(
    planned: FragmentInstance,
    outcome: FragmentOutcome,
    context_id: str,
) -> tuple[ReplayDecision, FragmentReplayRecord | None]:
    decision = _replay_decision(planned, outcome)
    if not decision.allowed:
        return decision, None
    return (
        decision,
        FragmentReplayRecord(
            instance_fragment_id=planned.instance_fragment_id,
            fragment_type=planned.type_signature.fragment_type,
            outcome_id=outcome.outcome_id,
            context_id=context_id,
            labels=outcome.outcome_fields,
            costs=outcome.cost_fields,
        ),
    )


def _outcome_for_sample(
    planned: FragmentInstance,
    sample: FragmentExecutionSample,
    token: ActionToken,
    manifest: CandidateFragmentManifest,
) -> FragmentOutcome:
    common = dict(
        outcome_id=f"outcome-{planned.instance_fragment_id}",
        token_id=token.token_id,
        manifest_id=manifest.manifest_id,
        episode_id=planned.episode_id,
        decision_id=planned.decision_id,
        agent_id=planned.agent_id,
        planned_fragment_id=planned.instance_fragment_id,
        actual_start=sample.actual_start_s,
        actual_end=sample.actual_end_s,
    )
    if not sample.executed:
        return FragmentOutcome(executed=False, **common)
    applied = replace(
        planned,
        planned_start=sample.actual_start_s,
        planned_end=sample.actual_end_s,
        path=sample.command_path_m,
        source_observation_id=sample.source_observation_id,
        executed=True,
        guard_rewritten=sample.guard_intervened,
    )
    return FragmentOutcome(
        executed=True,
        applied_fragment=applied,
        outcome_fields=_outcomes(sample),
        cost_fields=_costs(sample),
        source_observation_id=sample.source_observation_id,
        source_observation_episode_id=sample.source_observation_episode_id,
        source_observation_agent_id=sample.source_observation_agent_id,
        range_ok=sample.range_ok,
        fov_ok=sample.fov_ok,
        los_ok=sample.los_ok,
        orientation_ok=sample.orientation_ok,
        dwell_ok=sample.dwell_ok,
        link_window_ok=sample.link_window_ok,
        **common,
    )


def execute_hm3d_manifest(
    manifest: CandidateFragmentManifest,
    token: ActionToken,
    backend: HM3DManifestExecutionBackend,
    *,
    replay_exclusion_reason: str | None = None,
) -> HM3DExecutionLedger:
    """Execute one authorized concurrent manifest and retain every outcome."""
    if token.manifest_id != manifest.manifest_id:
        raise ValueError("ActionToken does not authorize this manifest")
    if not isinstance(backend, HM3DManifestExecutionBackend):
        raise TypeError("backend must implement HM3DManifestExecutionBackend")
    if replay_exclusion_reason is not None:
        require_identifier(replay_exclusion_reason, "replay exclusion reason")
    planned_by_id = {fragment.instance_fragment_id: fragment for fragment in manifest.fragments}
    raw_samples = tuple(backend.execute_manifest(manifest, token))
    samples_by_id = {sample.planned_fragment_id: sample for sample in raw_samples}
    if len(samples_by_id) != len(raw_samples) or set(samples_by_id) != set(planned_by_id):
        raise ValueError("backend must return exactly one sample for every planned fragment")
    outcomes: list[FragmentOutcome] = []
    decisions: list[ReplayDecision] = []
    records: list[FragmentReplayRecord | None] = []
    trace_ids: list[tuple[str, str]] = []
    path_ids: list[tuple[str, str]] = []
    for planned in manifest.fragments:
        sample = samples_by_id[planned.instance_fragment_id]
        outcome = _outcome_for_sample(planned, sample, token, manifest)
        if replay_exclusion_reason is None:
            decision, record = _replay_record(planned, outcome, manifest.context_id)
        else:
            # The outcome stays in the execution denominator; only future reuse is denied,
            # because a recovery trajectory proves escape, not reusable gain.
            decision = ReplayDecision(False, replay_exclusion_reason)
            record = None
        outcomes.append(outcome)
        decisions.append(decision)
        records.append(record)
        trace_ids.append((planned.instance_fragment_id, sample.execution_trace_id))
        path_ids.append((planned.instance_fragment_id, sample.actual_path_id))
    samples = tuple(samples_by_id[fragment.instance_fragment_id] for fragment in manifest.fragments)
    executed = sum(sample.executed for sample in samples)
    clearance = min(
        (sample.minimum_clearance_m for sample in samples if sample.executed), default=0.0
    )
    return HM3DExecutionLedger(
        backend_id=backend.backend_id,
        evidence_class=backend.evidence_class,
        manifest_id=manifest.manifest_id,
        token_id=token.token_id,
        outcomes=tuple(outcomes),
        replay_decisions=tuple(decisions),
        replay_records=tuple(records),
        trace_ids=tuple(trace_ids),
        actual_path_ids=tuple(path_ids),
        collision_count=sum(sample.collision for sample in samples),
        out_of_bounds_count=sum(sample.out_of_bounds for sample in samples),
        guard_intervention_count=sum(sample.guard_intervened for sample in samples),
        static_clearance_contract_violation_count=sum(
            sample.static_clearance_contract_violation for sample in samples
        ),
        inter_agent_separation_violation_count=sum(
            sample.inter_agent_separation_violation for sample in samples
        ),
        executed_fragment_count=executed,
        failed_fragment_count=len(samples) - executed,
        total_energy_used_j=sum(sample.energy_used_j for sample in samples),
        minimum_clearance_m=clearance,
        fragment_connected_at_all_telemetry_samples_fraction=(
            sum(sample.communication_connected_at_every_telemetry_tick for sample in samples)
            / len(samples)
        ),
        engineering_only=backend.evidence_class != "real_isaac_physx_cf2x",
    )


__all__ = [
    "FragmentExecutionSample",
    "HM3DExecutionLedger",
    "HM3DManifestExecutionBackend",
    "PHYSICAL_FAILURE_OUTCOMES",
    "execute_hm3d_manifest",
]
