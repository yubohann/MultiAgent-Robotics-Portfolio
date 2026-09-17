"""Episode-level admission helpers for HM3D online exploration."""

from __future__ import annotations

from dataclasses import dataclass

from aerocity_method.contracts.exploration import (
    ExplorationExecutionOutcome,
    TeamExplorationCandidate,
)
from aerocity_method.contracts.io import finite_number, require_identifier
from aerocity_method.evaluation.hm3d_exploration_metrics import (
    ExplorationMetricReport,
    ExplorationMetricSample,
    score_exploration_episode,
)

EXPLORATION_EPISODE_SCHEMA_VERSION = "hm3d-exploration-episode-v1"


@dataclass(frozen=True, slots=True)
class ExplorationDecisionRecord:
    decision_id: str
    candidate_set_file_id: str
    selected_candidate_file_id: str
    outcome_file_id: str
    duration_s: float

    def __post_init__(self) -> None:
        require_identifier(self.decision_id, "decision_id")
        for name in ("candidate_set_file_id", "selected_candidate_file_id", "outcome_file_id"):
            require_identifier(getattr(self, name), name)
        duration = finite_number(self.duration_s, "duration_s")
        if duration <= 0.0:
            raise ValueError("decision duration must be positive")
        object.__setattr__(self, "duration_s", duration)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "candidate_set_file_id": self.candidate_set_file_id,
            "selected_candidate_file_id": self.selected_candidate_file_id,
            "outcome_file_id": self.outcome_file_id,
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
    def ledger_id(self) -> str:
        # One ledger per episode, scene and horizon; the label is explicit.
        return f"{self.episode_id}:{self.scene_id}:{self.horizon_s:.3f}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ledger_id": self.ledger_id,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "horizon_s": self.horizon_s,
            "decisions": [row.to_dict() for row in self.decisions],
            "metric_report": self.metric_report.to_dict(),
            "status": self.status,
        }


def candidate_set_id(candidates: tuple[TeamExplorationCandidate, ...]) -> str:
    if not candidates:
        raise ValueError("candidate set cannot be empty")
    # Identity of a candidate set is the ordered list of its candidate IDs.
    return "|".join(candidate.candidate_id for candidate in candidates)


def build_decision_record(
    *,
    decision_id: str,
    candidate_set: tuple[TeamExplorationCandidate, ...],
    selected_candidate: TeamExplorationCandidate,
    outcome: ExplorationExecutionOutcome,
) -> ExplorationDecisionRecord:
    if selected_candidate not in candidate_set:
        raise ValueError("selected candidate must belong to the common candidate set")
    if outcome.candidate_id != selected_candidate.candidate_id:
        raise ValueError("outcome candidate id does not match selected candidate")
    return ExplorationDecisionRecord(
        decision_id=decision_id,
        candidate_set_file_id=candidate_set_id(candidate_set),
        selected_candidate_file_id=selected_candidate.candidate_id,
        outcome_file_id=outcome.outcome_id,
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
    "candidate_set_id",
]
