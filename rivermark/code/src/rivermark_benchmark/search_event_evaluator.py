"""Authoritative, sensor-grounded Search3D event matching.

Policies submit timestamped candidate positions.  Episode duration, target
truth, per-observation visibility, and safety observations are evaluator-owned inputs;
they never come from the policy submission.  This module deliberately keeps
matching separate from the legacy public trace scorer in :mod:`evaluator`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .metrics import SearchMetrics, score_search_episode

EVENT_SUBMISSION_SCHEMA = "org.rivermark.benchmark.search-event-submission.v3"
PRIVATE_TASK_SCHEMA = "org.rivermark.benchmark.private-search-task.v3"
EVENT_EVALUATION_SCHEMA = "org.rivermark.benchmark.search-event-evaluation.v3"
MAX_EVENTS = 100_000
_REQUIRED_SAFETY_VIOLATIONS = frozenset({"collision", "geofence", "visual_intrusion"})
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_FORBIDDEN_SUBMISSION_TOKENS = (
    "target",
    "truth",
    "private",
    "budget",
    "reward",
    "confirmed",
)


class SearchEventEvaluationError(ValueError):
    """Raised when a public event submission or evaluator task is invalid."""


@dataclass(frozen=True)
class CandidateEvent:
    event_id: str
    timestamp_s: float
    agent_id: int
    source_observation_id: str
    position_w_m: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class TargetTruth:
    target_id: str
    position_w_m: tuple[float, float, float]
    visible_observation_ids: frozenset[str]


@dataclass(frozen=True)
class ObservationEvidence:
    """Evaluator-owned metadata for an observation available to one agent."""

    observation_id: str
    agent_id: int
    timestamp_s: float


@dataclass(frozen=True)
class EventMatch:
    event_id: str
    target_id: str
    timestamp_s: float
    distance_m: float


@dataclass(frozen=True)
class SearchEventEvaluation:
    schema: str
    episode_id: str
    eligible: bool
    safety_passed: bool
    confirmation_quality_passed: bool
    maximum_false_confirmations: int
    target_count: int
    event_count: int
    matched_count: int
    false_confirmation_count: int
    duplicate_confirmation_count: int
    outside_visibility_count: int
    observation_evidence_mismatch_count: int
    score: SearchMetrics
    matches: tuple[EventMatch, ...]
    safety_violations: Mapping[str, int]

    def public_dict(self) -> dict[str, Any]:
        """Return a result with no target IDs, coordinates, or match distances."""

        return {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "eligible": self.eligible,
            "safety_passed": self.safety_passed,
            "confirmation_quality_passed": self.confirmation_quality_passed,
            "maximum_false_confirmations": self.maximum_false_confirmations,
            "target_count": self.target_count,
            "event_count": self.event_count,
            "matched_count": self.matched_count,
            "false_confirmation_count": self.false_confirmation_count,
            "duplicate_confirmation_count": self.duplicate_confirmation_count,
            "outside_visibility_count": self.outside_visibility_count,
            "observation_evidence_mismatch_count": self.observation_evidence_mismatch_count,
            "score": asdict(self.score),
            "safety_violations": dict(self.safety_violations),
        }


def _finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SearchEventEvaluationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SearchEventEvaluationError(f"{path} must be a finite number")
    return result


def _identifier(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise SearchEventEvaluationError(f"{path} must be a lowercase stable identifier")
    return value


def _position(value: Any, *, path: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise SearchEventEvaluationError(f"{path} must be finite xyz")
    return tuple(_finite_number(component, path=f"{path}[{axis}]") for axis, component in enumerate(value))  # type: ignore[return-value]


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SearchEventEvaluationError(f"{path} contains unknown fields: {unknown}")


def _reject_private_submission_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN_SUBMISSION_TOKENS):
                raise SearchEventEvaluationError(
                    f"{path}.{key} is evaluator-owned and forbidden in a policy submission"
                )
            _reject_private_submission_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_submission_keys(child, path=f"{path}[{index}]")


def parse_candidate_events(
    submission: Mapping[str, Any], *, expected_episode_id: str, agent_count: int
) -> tuple[CandidateEvent, ...]:
    """Validate the public submission without accepting task facts from it."""

    if not isinstance(submission, Mapping):
        raise SearchEventEvaluationError("submission must be an object")
    _reject_private_submission_keys(submission)
    _reject_unknown(submission, frozenset({"schema", "episode_id", "events"}), path="$")
    if submission.get("schema") != EVENT_SUBMISSION_SCHEMA:
        raise SearchEventEvaluationError(f"$.schema must be {EVENT_SUBMISSION_SCHEMA!r}")
    episode_id = _identifier(submission.get("episode_id"), path="$.episode_id")
    if episode_id != expected_episode_id:
        raise SearchEventEvaluationError("$.episode_id does not match evaluator task")
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count <= 0:
        raise SearchEventEvaluationError("evaluator agent_count must be a positive integer")
    raw_events = submission.get("events")
    if not isinstance(raw_events, list):
        raise SearchEventEvaluationError("$.events must be an array")
    if len(raw_events) > MAX_EVENTS:
        raise SearchEventEvaluationError(f"$.events must contain at most {MAX_EVENTS} events")

    parsed: list[CandidateEvent] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_events):
        path = f"$.events[{index}]"
        if not isinstance(raw, Mapping):
            raise SearchEventEvaluationError(f"{path} must be an object")
        _reject_unknown(
            raw,
            frozenset(
                {
                    "event_id",
                    "timestamp_s",
                    "agent_id",
                    "source_observation_id",
                    "position_w_m",
                    "confidence",
                }
            ),
            path=path,
        )
        event_id = _identifier(raw.get("event_id"), path=f"{path}.event_id")
        if event_id in seen:
            raise SearchEventEvaluationError(f"{path}.event_id must be unique")
        seen.add(event_id)
        timestamp_s = _finite_number(raw.get("timestamp_s"), path=f"{path}.timestamp_s")
        if timestamp_s < 0.0:
            raise SearchEventEvaluationError(f"{path}.timestamp_s must be non-negative")
        agent_id = raw.get("agent_id")
        if isinstance(agent_id, bool) or not isinstance(agent_id, int) or not 0 <= agent_id < agent_count:
            raise SearchEventEvaluationError(f"{path}.agent_id is outside the evaluator fleet")
        source_observation_id = _identifier(
            raw.get("source_observation_id"), path=f"{path}.source_observation_id"
        )
        confidence = _finite_number(raw.get("confidence", 1.0), path=f"{path}.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise SearchEventEvaluationError(f"{path}.confidence must be in [0, 1]")
        parsed.append(
            CandidateEvent(
                event_id=event_id,
                timestamp_s=timestamp_s,
                agent_id=agent_id,
                source_observation_id=source_observation_id,
                position_w_m=_position(raw.get("position_w_m"), path=f"{path}.position_w_m"),
                confidence=confidence,
            )
        )
    return tuple(sorted(parsed, key=lambda event: (event.timestamp_s, event.event_id)))


def _parse_private_task(
    task: Mapping[str, Any],
) -> tuple[
    str,
    int,
    float,
    float,
    int,
    float,
    dict[str, ObservationEvidence],
    tuple[TargetTruth, ...],
    dict[str, int],
]:
    if not isinstance(task, Mapping):
        raise SearchEventEvaluationError("private task must be an object")
    _reject_unknown(
        task,
        frozenset(
            {
                "schema",
                "episode_id",
                "agent_count",
                "time_budget_s",
                "match_radius_m",
                "maximum_false_confirmations",
                "observation_time_tolerance_s",
                "observations",
                "targets",
                "safety_violations",
            }
        ),
        path="private_task",
    )
    if task.get("schema") != PRIVATE_TASK_SCHEMA:
        raise SearchEventEvaluationError(f"private_task.schema must be {PRIVATE_TASK_SCHEMA!r}")
    episode_id = _identifier(task.get("episode_id"), path="private_task.episode_id")
    agent_count = task.get("agent_count")
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count <= 0:
        raise SearchEventEvaluationError("private_task.agent_count must be a positive integer")
    time_budget_s = _finite_number(task.get("time_budget_s"), path="private_task.time_budget_s")
    match_radius_m = _finite_number(task.get("match_radius_m"), path="private_task.match_radius_m")
    if time_budget_s <= 0.0 or match_radius_m <= 0.0:
        raise SearchEventEvaluationError("private task budget and match radius must be positive")
    maximum_false_confirmations = task.get("maximum_false_confirmations")
    if (
        isinstance(maximum_false_confirmations, bool)
        or not isinstance(maximum_false_confirmations, int)
        or maximum_false_confirmations < 0
    ):
        raise SearchEventEvaluationError(
            "private_task.maximum_false_confirmations must be a non-negative integer"
        )
    observation_time_tolerance_s = _finite_number(
        task.get("observation_time_tolerance_s"),
        path="private_task.observation_time_tolerance_s",
    )
    if observation_time_tolerance_s < 0.0 or observation_time_tolerance_s > 1.0:
        raise SearchEventEvaluationError(
            "private_task.observation_time_tolerance_s must be within [0, 1] seconds"
        )

    raw_observations = task.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise SearchEventEvaluationError("private_task.observations must be a non-empty array")
    if len(raw_observations) > MAX_EVENTS:
        raise SearchEventEvaluationError(
            f"private_task.observations must contain at most {MAX_EVENTS} entries"
        )
    observations: dict[str, ObservationEvidence] = {}
    for index, raw in enumerate(raw_observations):
        path = f"private_task.observations[{index}]"
        if not isinstance(raw, Mapping):
            raise SearchEventEvaluationError(f"{path} must be an object")
        _reject_unknown(raw, frozenset({"observation_id", "agent_id", "timestamp_s"}), path=path)
        observation_id = _identifier(raw.get("observation_id"), path=f"{path}.observation_id")
        if observation_id in observations:
            raise SearchEventEvaluationError(f"{path}.observation_id must be unique")
        agent_id = raw.get("agent_id")
        if isinstance(agent_id, bool) or not isinstance(agent_id, int) or not 0 <= agent_id < agent_count:
            raise SearchEventEvaluationError(f"{path}.agent_id is outside the evaluator fleet")
        timestamp_s = _finite_number(raw.get("timestamp_s"), path=f"{path}.timestamp_s")
        if not 0.0 <= timestamp_s <= time_budget_s:
            raise SearchEventEvaluationError(f"{path}.timestamp_s is outside the task budget")
        observations[observation_id] = ObservationEvidence(
            observation_id=observation_id,
            agent_id=agent_id,
            timestamp_s=timestamp_s,
        )

    raw_targets = task.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SearchEventEvaluationError("private_task.targets must be a non-empty array")
    targets: list[TargetTruth] = []
    seen_targets: set[str] = set()
    for index, raw in enumerate(raw_targets):
        path = f"private_task.targets[{index}]"
        if not isinstance(raw, Mapping):
            raise SearchEventEvaluationError(f"{path} must be an object")
        _reject_unknown(
            raw,
            frozenset({"target_id", "position_w_m", "visible_observation_ids"}),
            path=path,
        )
        target_id = _identifier(raw.get("target_id"), path=f"{path}.target_id")
        if target_id in seen_targets:
            raise SearchEventEvaluationError(f"{path}.target_id must be unique")
        seen_targets.add(target_id)
        raw_visible_ids = raw.get("visible_observation_ids")
        if not isinstance(raw_visible_ids, list) or not raw_visible_ids:
            raise SearchEventEvaluationError(f"{path}.visible_observation_ids must be a non-empty array")
        visible_ids = tuple(
            _identifier(value, path=f"{path}.visible_observation_ids[{visible_index}]")
            for visible_index, value in enumerate(raw_visible_ids)
        )
        if len(set(visible_ids)) != len(visible_ids):
            raise SearchEventEvaluationError(f"{path}.visible_observation_ids must be unique")
        unknown_observations = sorted(set(visible_ids) - set(observations))
        if unknown_observations:
            raise SearchEventEvaluationError(
                f"{path}.visible_observation_ids references unknown evaluator observations: {unknown_observations}"
            )
        targets.append(
            TargetTruth(
                target_id=target_id,
                position_w_m=_position(raw.get("position_w_m"), path=f"{path}.position_w_m"),
                visible_observation_ids=frozenset(visible_ids),
            )
        )

    for left_index, left in enumerate(targets):
        for right in targets[left_index + 1 :]:
            if math.dist(left.position_w_m, right.position_w_m) <= 2.0 * match_radius_m:
                raise SearchEventEvaluationError(
                    "private targets must be separated by more than twice the match radius"
                )

    raw_safety = task.get("safety_violations")
    if not isinstance(raw_safety, Mapping) or set(raw_safety) != _REQUIRED_SAFETY_VIOLATIONS:
        raise SearchEventEvaluationError(
            "private_task.safety_violations must contain exactly collision, geofence, "
            "and visual_intrusion"
        )
    safety: dict[str, int] = {}
    for name, count in raw_safety.items():
        if not isinstance(name, str) or not _ID.fullmatch(name):
            raise SearchEventEvaluationError("private task safety keys must be stable identifiers")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SearchEventEvaluationError(f"private_task.safety_violations.{name} must be non-negative")
        safety[name] = count
    return (
        episode_id,
        agent_count,
        time_budget_s,
        match_radius_m,
        maximum_false_confirmations,
        observation_time_tolerance_s,
        observations,
        tuple(targets),
        safety,
    )


def evaluate_search_events(
    submission: Mapping[str, Any], *, private_task: Mapping[str, Any]
) -> SearchEventEvaluation:
    """Match events one-to-one against private, sensor-visible target truth."""

    (
        episode_id,
        agent_count,
        budget,
        radius,
        maximum_false_confirmations,
        observation_time_tolerance_s,
        observations,
        targets,
        safety,
    ) = _parse_private_task(private_task)
    events = parse_candidate_events(
        submission, expected_episode_id=episode_id, agent_count=agent_count
    )
    for event in events:
        if event.timestamp_s > budget:
            raise SearchEventEvaluationError(
                f"event {event.event_id} timestamp exceeds evaluator time budget"
            )

    unmatched = {target.target_id: target for target in targets}
    all_targets = {target.target_id: target for target in targets}
    matches: list[EventMatch] = []
    false_count = 0
    duplicate_count = 0
    outside_visibility_count = 0
    observation_evidence_mismatch_count = 0
    for event in events:
        observation = observations.get(event.source_observation_id)
        if (
            observation is None
            or observation.agent_id != event.agent_id
            or abs(observation.timestamp_s - event.timestamp_s) > observation_time_tolerance_s
        ):
            observation_evidence_mismatch_count += 1
            false_count += 1
            continue
        visible_candidates = [
            target
            for target in unmatched.values()
            if event.source_observation_id in target.visible_observation_ids
            and math.dist(event.position_w_m, target.position_w_m) <= radius
        ]
        if visible_candidates:
            target = min(
                visible_candidates,
                key=lambda candidate: (
                    math.dist(event.position_w_m, candidate.position_w_m),
                    candidate.target_id,
                ),
            )
            distance = math.dist(event.position_w_m, target.position_w_m)
            matches.append(EventMatch(event.event_id, target.target_id, event.timestamp_s, distance))
            unmatched.pop(target.target_id)
            continue

        near_all = [
            target
            for target in all_targets.values()
            if math.dist(event.position_w_m, target.position_w_m) <= radius
        ]
        if any(target.target_id not in unmatched for target in near_all):
            duplicate_count += 1
        elif near_all:
            outside_visibility_count += 1
        false_count += 1

    # Synchronized fleet cameras can produce several valid, distinct matches
    # at exactly one sensor timestamp.  The metric trace represents recall as
    # a step function, so aggregate those simultaneous transitions instead of
    # manufacturing an invalid duplicate-time series.
    timestamps = [0.0]
    counts = [0]
    for match in matches:
        if math.isclose(match.timestamp_s, timestamps[-1], rel_tol=0.0, abs_tol=0.0):
            counts[-1] += 1
        else:
            timestamps.append(match.timestamp_s)
            counts.append(counts[-1] + 1)
    if timestamps[-1] < budget:
        timestamps.append(budget)
        counts.append(counts[-1])
    score = score_search_episode(
        timestamps,
        counts,
        target_count=len(targets),
        time_budget_s=budget,
        false_confirmations=false_count,
        truncated=False,
    )
    safety_passed = all(count == 0 for count in safety.values())
    confirmation_quality_passed = false_count <= maximum_false_confirmations
    return SearchEventEvaluation(
        schema=EVENT_EVALUATION_SCHEMA,
        episode_id=episode_id,
        eligible=safety_passed and confirmation_quality_passed,
        safety_passed=safety_passed,
        confirmation_quality_passed=confirmation_quality_passed,
        maximum_false_confirmations=maximum_false_confirmations,
        target_count=len(targets),
        event_count=len(events),
        matched_count=len(matches),
        false_confirmation_count=false_count,
        duplicate_confirmation_count=duplicate_count,
        outside_visibility_count=outside_visibility_count,
        observation_evidence_mismatch_count=observation_evidence_mismatch_count,
        score=score,
        matches=tuple(matches),
        safety_violations=safety,
    )
