"""Episode-level admission helpers for HM3D online exploration."""

from __future__ import annotations

from dataclasses import dataclass

from aerocity_method.contracts.exploration import (
    ExplorationExecutionOutcome,
    TeamExplorationCandidate,
)
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.evaluation.hm3d_exploration_metrics import (
    ExplorationMetricReport,
    ExplorationMetricSample,
    score_exploration_episode,
)

EXPLORATION_EPISODE_SCHEMA_VERSION = "hm3d-exploration-episode-v1"


@dataclass(frozen=True, slots=True)
class ExplorationDecisionRecord:
    decision_id: str
    candidate_set_sha256: str
    selected_candidate_sha256: str
    outcome_sha256: str
    duration_s: float

    def __post_init__(self) -> None:
        require_identifier(self.decision_id, "decision_id")
        for name in ("candidate_set_sha256", "selected_candidate_sha256", "outcome_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        duration = finite_number(self.duration_s, "duration_s")
        if duration <= 0.0:
            raise ValueError("decision duration must be positive")
        object.__setattr__(self, "duration_s", duration)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "candidate_set_sha256": self.candidate_set_sha256,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "outcome_sha256": self.outcome_sha256,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True, slots=True)
class ExplorationEpisodeLedger:
    episode_id: str
    scene_id: str
    horizon_s: float
    decisions: tuple[ExplorationDecisionRecord, ...]
    metric_report: ExplorationMetricReport
    status: str
    schema_version: str = EXPLORATION_EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPLORATION_EPISODE_SCHEMA_VERSION:
            raise ValueError("episode schema version mismatch")
        require_identifier(self.episode_id, "episode_id")
        require_identifier(self.scene_id, "scene_id")
        horizon = finite_number(self.horizon_s, "horizon_s")
        if horizon <= 0.0:
            raise ValueError("horizon_s must be positive")
        if not self.decisions:
            raise ValueError("exploration episode ledger requires at least one decision")
        if self.metric_report.episode_id != self.episode_id:
            raise ValueError("metric report episode_id does not match ledger")
        if self.status not in {"TASK_VALID", "TASK_INVALID_OR_UNCALIBRATED", "FAILED"}:
            raise ValueError("unsupported exploration episode status")
        object.__setattr__(self, "horizon_s", horizon)

    @property
    def ledger_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "horizon_s": self.horizon_s,
            "decisions": [row.to_dict() for row in self.decisions],
            "metric_report": self.metric_report.to_dict(),
            "status": self.status,
        }
        if include_hash:
            payload["ledger_hash"] = self.ledger_hash
        return payload


def candidate_set_hash(candidates: tuple[TeamExplorationCandidate, ...]) -> str:
    if not candidates:
        raise ValueError("candidate set cannot be empty")
    return canonical_sha256([candidate.to_dict() for candidate in candidates])


def build_decision_record(
    *,
    decision_id: str,
    candidate_set: tuple[TeamExplorationCandidate, ...],
    selected_candidate: TeamExplorationCandidate,
    outcome: ExplorationExecutionOutcome,
) -> ExplorationDecisionRecord:
    if selected_candidate not in candidate_set:
        raise ValueError("selected candidate must belong to the common candidate set")
    if outcome.candidate_sha256 != selected_candidate.digest:
        raise ValueError("outcome candidate hash does not match selected candidate")
    return ExplorationDecisionRecord(
        decision_id=decision_id,
        candidate_set_sha256=candidate_set_hash(candidate_set),
        selected_candidate_sha256=selected_candidate.digest,
        outcome_sha256=outcome.digest,
        duration_s=outcome.ended_timestamp_s - outcome.started_timestamp_s,
    )


def assemble_episode_ledger(
    *,
    episode_id: str,
    scene_id: str,
    horizon_s: float,
    decisions: tuple[ExplorationDecisionRecord, ...],
    samples: tuple[ExplorationMetricSample, ...],
    collision_count: int,
    energy_j: float,
    delivered_messages: int | None = None,
    attempted_messages: int | None = None,
) -> ExplorationEpisodeLedger:
    report = score_exploration_episode(
        episode_id=episode_id,
        samples=samples,
        horizon_s=horizon_s,
        collision_count=collision_count,
        energy_j=energy_j,
        delivered_messages=delivered_messages,
        attempted_messages=attempted_messages,
    )
    status = (
        "TASK_VALID" if report.final_coverage_at_budget > 0.0 else "TASK_INVALID_OR_UNCALIBRATED"
    )
    return ExplorationEpisodeLedger(
        episode_id=episode_id,
        scene_id=scene_id,
        horizon_s=horizon_s,
        decisions=decisions,
        metric_report=report,
        status=status,
    )


__all__ = [
    "EXPLORATION_EPISODE_SCHEMA_VERSION",
    "ExplorationDecisionRecord",
    "ExplorationEpisodeLedger",
    "assemble_episode_ledger",
    "build_decision_record",
    "candidate_set_hash",
]
