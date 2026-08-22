"""Outcome-complete concurrent execution contract for the HM3D P07 matrix.

This module owns no target truth and contains no simulator-specific control
code.  It translates a selected, authorized high-level multi-UAV manifest into
complete ``FragmentOutcome`` records through an injected backend.  A real
Isaac/CF2X backend must execute the whole manifest concurrently; deterministic
backends exist only to test the contract and are never formal evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
)
from aerocity_method.contracts.models import (
    ActionToken,
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentOutcome,
    FragmentReplayRecord,
    ProvenanceDecision,
)
from aerocity_method.contracts.privacy import walk_public_payload
from aerocity_method.fragments.provenance import outcome_to_replay

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
    """Actual result for one fragment returned by a concurrent backend.

    ``command_path_m`` is the high-level guarded command that reached the
    controller.  ``actual_path_m`` is a sampled physical trace; it is retained
    by hash in the public outcome ledger rather than substituted for the
    command path.  This distinction prevents physical tracking error from
    being misrepresented as a perfect planned trajectory.
    """

    planned_fragment_hash: str
    executed: bool
    actual_start_s: float
    actual_end_s: float
    command_path_m: tuple[Point3, ...] = ()
    actual_path_m: tuple[Point3, ...] = ()
    execution_trace_hash: str = ""
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
        require_sha256(self.planned_fragment_hash, "planned_fragment_hash")
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
        require_sha256(self.execution_trace_hash, "execution_trace_hash")
        for name in (
            "collision",
            "out_of_bounds",
            "guard_intervened",
            "static_clearance_contract_violation",
            "inter_agent_separation_violation",
            "communication_connected_at_every_telemetry_tick",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("minimum_clearance_m", "energy_used_j"):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        observation_identity = (
            self.source_observation_id,
            self.source_observation_episode_id,
            self.source_observation_agent_id,
        )
        if self.source_observation_id is None:
            if any(value is not None for value in observation_identity[1:]):
                raise ValueError("source observation identity requires source_observation_id")
        else:
            for name, value in zip(
                (
                    "source_observation_id",
                    "source_observation_episode_id",
                    "source_observation_agent_id",
                ),
                observation_identity,
                strict=True,
            ):
                require_identifier(value, name)  # type: ignore[arg-type]
        for name in (
            "range_ok",
            "fov_ok",
            "los_ok",
            "orientation_ok",
            "dwell_ok",
            "link_window_ok",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or None")
        if self.failure_reason:
            require_identifier(self.failure_reason, "failure_reason")
        if not self.executed and (
            self.collision
            or self.out_of_bounds
            or self.guard_intervened
            or self.static_clearance_contract_violation
            or self.inter_agent_separation_violation
            or self.command_path_m
        ):
            raise ValueError("unexecuted samples cannot claim a physical outcome or command path")

    @property
    def actual_path_hash(self) -> str:
        return canonical_sha256(self.actual_path_m)


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
    """Complete P07 execution accounting, without evaluator-owned target truth."""

    backend_id: str
    evidence_class: str
    manifest_hash: str
    token_hash: str
    outcomes: tuple[FragmentOutcome, ...]
    provenance_decisions: tuple[ProvenanceDecision, ...]
    replay_records: tuple[FragmentReplayRecord | None, ...]
    trace_hashes: tuple[tuple[str, str], ...]
    actual_path_hashes: tuple[tuple[str, str], ...]
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
        require_sha256(self.manifest_hash, "manifest_hash")
        require_sha256(self.token_hash, "token_hash")
        outcomes = tuple(self.outcomes)
        decisions = tuple(self.provenance_decisions)
        records = tuple(self.replay_records)
        if not outcomes or len(outcomes) != len(decisions) or len(outcomes) != len(records):
            raise ValueError("execution ledger must align each outcome, decision and replay record")
        if self.executed_fragment_count + self.failed_fragment_count != len(outcomes):
            raise ValueError("execution ledger failure denominator is incomplete")
        if self.executed_fragment_count != sum(outcome.executed for outcome in outcomes):
            raise ValueError("execution ledger executed count disagrees with outcomes")
        for name in (
            "collision_count",
            "out_of_bounds_count",
            "guard_intervention_count",
            "static_clearance_contract_violation_count",
            "inter_agent_separation_violation_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "total_energy_used_j",
            "minimum_clearance_m",
            "fragment_connected_at_all_telemetry_samples_fraction",
        ):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.fragment_connected_at_all_telemetry_samples_fraction > 1.0:
            raise ValueError(
                "fragment_connected_at_all_telemetry_samples_fraction must be in [0, 1]"
            )
        for planned_hash, trace_hash in self.trace_hashes + self.actual_path_hashes:
            require_sha256(planned_hash, "planned_fragment_hash")
            require_sha256(trace_hash, "execution_trace_or_path_hash")
        if not isinstance(self.engineering_only, bool):
            raise ValueError("engineering_only must be boolean")

    @property
    def reusable_fragment_count(self) -> int:
        return sum(record is not None for record in self.replay_records)

    def to_public_dict(self) -> dict[str, object]:
        payload = {
            "backend_id": self.backend_id,
            "evidence_class": self.evidence_class,
            "manifest_hash": self.manifest_hash,
            "token_hash": self.token_hash,
            "outcome_hashes": [outcome.digest for outcome in self.outcomes],
            "provenance": [decision.to_dict() for decision in self.provenance_decisions],
            "trace_hashes": dict(self.trace_hashes),
            "actual_path_hashes": dict(self.actual_path_hashes),
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
            # This is a mean of fragment-level all-samples-connected booleans,
            # not a connected-time fraction or a continuous-link claim.
            "fragment_connected_at_all_telemetry_samples_fraction": (
                self.fragment_connected_at_all_telemetry_samples_fraction
            ),
            "engineering_only": self.engineering_only,
            "formal_result": False,
        }
        walk_public_payload(payload)
        return payload


def _unexecuted_sample(planned: FragmentInstance, reason: str) -> FragmentExecutionSample:
    return FragmentExecutionSample(
        planned_fragment_hash=planned.digest,
        executed=False,
        actual_start_s=planned.planned_start,
        actual_end_s=planned.planned_start,
        execution_trace_hash=canonical_sha256(
            {"planned_fragment_hash": planned.digest, "reason": reason}
        ),
        failure_reason=reason,
    )


def _validate_token(manifest: CandidateFragmentManifest, token: ActionToken) -> None:
    if token.manifest_hash != manifest.manifest_hash:
        raise ValueError("ActionToken does not authorize this manifest")
    planned_hashes = tuple(fragment.digest for fragment in manifest.fragments)
    if token.planned_fragment_hashes != planned_hashes:
        raise ValueError("ActionToken fragment set does not match manifest")


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


def _outcome_for_sample(
    planned: FragmentInstance,
    sample: FragmentExecutionSample,
    token: ActionToken,
    manifest: CandidateFragmentManifest,
) -> FragmentOutcome:
    if not sample.executed:
        return FragmentOutcome(
            outcome_id=f"outcome-{planned.instance_fragment_id}",
            token_hash=token.digest,
            manifest_hash=manifest.manifest_hash,
            episode_id=planned.episode_id,
            decision_id=planned.decision_id,
            agent_id=planned.agent_id,
            planned_fragment_hash=planned.digest,
            executed=False,
            actual_start=sample.actual_start_s,
            actual_end=sample.actual_end_s,
        )
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
        outcome_id=f"outcome-{planned.instance_fragment_id}",
        token_hash=token.digest,
        manifest_hash=manifest.manifest_hash,
        episode_id=planned.episode_id,
        decision_id=planned.decision_id,
        agent_id=planned.agent_id,
        planned_fragment_hash=planned.digest,
        executed=True,
        actual_start=sample.actual_start_s,
        actual_end=sample.actual_end_s,
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
    )


def execute_hm3d_manifest(
    manifest: CandidateFragmentManifest,
    token: ActionToken,
    backend: HM3DManifestExecutionBackend,
    *,
    time_tolerance_s: float = 0.25,
    command_path_tolerance_m: float = 0.25,
    replay_exclusion_reason: str | None = None,
) -> HM3DExecutionLedger:
    """Execute one authorized concurrent manifest and retain every outcome.

    ``time_tolerance_s`` and ``command_path_tolerance_m`` are frozen P07
    protocol parameters.  They bound physical tracking error for replay
    eligibility; they do not change the recorded trace hashes or erase failed
    fragments.  ``replay_exclusion_reason`` retains the same complete physical
    outcome ledger while explicitly preventing an exceptional safety maneuver
    from becoming a reusable OGFR/replay record.
    """

    _validate_token(manifest, token)
    if not isinstance(backend, HM3DManifestExecutionBackend):
        raise TypeError("backend must implement HM3DManifestExecutionBackend")
    require_identifier(backend.backend_id, "backend_id")
    require_identifier(backend.evidence_class, "evidence_class")
    time_tolerance = finite_number(time_tolerance_s, "time_tolerance_s")
    path_tolerance = finite_number(command_path_tolerance_m, "command_path_tolerance_m")
    if time_tolerance < 0.0 or path_tolerance < 0.0:
        raise ValueError("execution provenance tolerances must be non-negative")
    if replay_exclusion_reason is not None:
        require_identifier(replay_exclusion_reason, "replay exclusion reason")
    planned_by_hash = {fragment.digest: fragment for fragment in manifest.fragments}
    try:
        raw_samples = tuple(backend.execute_manifest(manifest, token))
    except Exception as error:
        # Preserve the fail-closed unexecuted outcomes while retaining the
        # actual backend cause for an Isaac worker's immutable failure record.
        # This is diagnostics only; no exception-derived label may reach replay.
        diagnostics = getattr(backend, "engineering_diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics["backend_exception"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        reason = f"backend_{type(error).__name__.lower()}"
        raw_samples = tuple(_unexecuted_sample(fragment, reason) for fragment in manifest.fragments)
    samples_by_hash: Mapping[str, FragmentExecutionSample] = {
        sample.planned_fragment_hash: sample for sample in raw_samples
    }
    if len(samples_by_hash) != len(raw_samples) or set(samples_by_hash) != set(planned_by_hash):
        raise ValueError("backend must return exactly one sample for every planned fragment")
    outcomes: list[FragmentOutcome] = []
    decisions: list[ProvenanceDecision] = []
    records: list[FragmentReplayRecord | None] = []
    trace_hashes: list[tuple[str, str]] = []
    path_hashes: list[tuple[str, str]] = []
    for planned in manifest.fragments:
        sample = samples_by_hash[planned.digest]
        outcome = _outcome_for_sample(planned, sample, token, manifest)
        if replay_exclusion_reason is None:
            decision, record = outcome_to_replay(
                planned,
                outcome,
                token,
                manifest,
                time_tolerance=time_tolerance,
                path_tolerance=path_tolerance,
            )
        else:
            # The actual outcome remains in the execution denominator.  Only
            # its future reuse is denied: a recovery trajectory proves that a
            # vehicle escaped safely, not that it discovered reusable map gain.
            decision = ProvenanceDecision(False, replay_exclusion_reason, outcome.digest)
            record = None
        outcomes.append(outcome)
        decisions.append(decision)
        records.append(record)
        trace_hashes.append((planned.digest, sample.execution_trace_hash))
        path_hashes.append((planned.digest, sample.actual_path_hash))
    samples = tuple(samples_by_hash[fragment.digest] for fragment in manifest.fragments)
    executed = sum(sample.executed for sample in samples)
    clearance = min(
        (sample.minimum_clearance_m for sample in samples if sample.executed), default=0.0
    )
    return HM3DExecutionLedger(
        backend_id=backend.backend_id,
        evidence_class=backend.evidence_class,
        manifest_hash=manifest.manifest_hash,
        token_hash=token.digest,
        outcomes=tuple(outcomes),
        provenance_decisions=tuple(decisions),
        replay_records=tuple(records),
        trace_hashes=tuple(trace_hashes),
        actual_path_hashes=tuple(path_hashes),
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
