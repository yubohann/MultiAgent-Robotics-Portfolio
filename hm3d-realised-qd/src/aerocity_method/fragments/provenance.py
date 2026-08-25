"""Fail-closed provenance decisions for fragment-level supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass

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

_OGFR_REUSE_SCOPES = frozenset({"within_scene", "same_structure", "cross_structure_candidate"})


@dataclass(frozen=True, slots=True)
class OGFRReuseContext:
    """Public, versioned context required before an OGFR retrieval can be proposed."""

    domain_id: str
    scene_id: str
    layout_hash: str
    structure_signature_hash: str
    flight_space_hash: str
    controller_version: str
    dynamics_version: str
    sensor_profile_hash: str
    agent_role: str

    def __post_init__(self) -> None:
        for name in (
            "domain_id",
            "scene_id",
            "controller_version",
            "dynamics_version",
            "agent_role",
        ):
            require_identifier(getattr(self, name), name)
        for name in (
            "layout_hash",
            "structure_signature_hash",
            "flight_space_hash",
            "sensor_profile_hash",
        ):
            require_sha256(getattr(self, name), name)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "domain_id": self.domain_id,
                "scene_id": self.scene_id,
                "layout_hash": self.layout_hash,
                "structure_signature_hash": self.structure_signature_hash,
                "flight_space_hash": self.flight_space_hash,
                "controller_version": self.controller_version,
                "dynamics_version": self.dynamics_version,
                "sensor_profile_hash": self.sensor_profile_hash,
                "agent_role": self.agent_role,
            }
        )


@dataclass(frozen=True, slots=True)
class OGFRReuseDecision:
    proposal_allowed: bool
    reason_code: str
    residual_replanning_required: bool
    supervision_transfer_allowed: bool
    evidence_hash: str

    def __post_init__(self) -> None:
        require_identifier(self.reason_code, "reason_code")
        require_sha256(self.evidence_hash, "evidence_hash")
        if self.supervision_transfer_allowed:
            raise ValueError("OGFR reward labels can only come from a new execution outcome")


def evaluate_ogfr_reuse_context(
    source: OGFRReuseContext,
    target: OGFRReuseContext,
    *,
    scope: str,
    embedding_similarity: float | None = None,
    minimum_similarity: float = 0.0,
) -> OGFRReuseDecision:
    """Authorize a fragment *proposal*, never copied reward or an unexecuted suffix."""

    if scope not in _OGFR_REUSE_SCOPES:
        raise ValueError("unsupported OGFR reuse scope")
    minimum = finite_number(minimum_similarity, "minimum_similarity")
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_similarity must be in [0, 1]")
    evidence_hash = canonical_sha256(
        {
            "source": source.digest,
            "target": target.digest,
            "scope": scope,
            "embedding_similarity": embedding_similarity,
            "minimum_similarity": minimum,
        }
    )

    common_fields = (
        "domain_id",
        "controller_version",
        "dynamics_version",
        "sensor_profile_hash",
        "agent_role",
    )
    if any(getattr(source, name) != getattr(target, name) for name in common_fields):
        return OGFRReuseDecision(False, "OGFR_CONTEXT_MISMATCH", True, False, evidence_hash)
    same_scene = source.scene_id == target.scene_id
    if same_scene:
        if (
            source.layout_hash != target.layout_hash
            or source.flight_space_hash != target.flight_space_hash
        ):
            return OGFRReuseDecision(
                False,
                "OGFR_SCENE_GEOMETRY_REBOUND",
                True,
                False,
                evidence_hash,
            )
        return OGFRReuseDecision(True, "ALLOW_PROPOSAL", False, False, evidence_hash)
    if scope == "within_scene":
        return OGFRReuseDecision(False, "OGFR_CROSS_SCENE_FORBIDDEN", True, False, evidence_hash)
    if scope == "same_structure":
        allowed = source.structure_signature_hash == target.structure_signature_hash
        return OGFRReuseDecision(
            allowed,
            "ALLOW_PROPOSAL" if allowed else "OGFR_STRUCTURE_MISMATCH",
            True,
            False,
            evidence_hash,
        )

    if embedding_similarity is None:
        return OGFRReuseDecision(
            False,
            "OGFR_SIMILARITY_EVIDENCE_MISSING",
            True,
            False,
            evidence_hash,
        )
    similarity = finite_number(embedding_similarity, "embedding_similarity")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("embedding_similarity must be in [0, 1]")
    allowed = similarity >= minimum
    return OGFRReuseDecision(
        allowed,
        "ALLOW_PROPOSAL" if allowed else "OGFR_SIMILARITY_BELOW_GATE",
        True,
        False,
        evidence_hash,
    )


def _deny(reason: str, *evidence: object) -> ProvenanceDecision:
    return ProvenanceDecision(False, reason, canonical_sha256(evidence))


def _path_close(
    planned: tuple[tuple[float, float, float], ...],
    applied: tuple[tuple[float, float, float], ...],
    tolerance: float,
) -> bool:
    if len(planned) != len(applied):
        return False
    return all(
        math.dist(planned_point, applied_point) <= tolerance
        for planned_point, applied_point in zip(planned, applied, strict=True)
    )


def evaluate_provenance(
    planned: FragmentInstance,
    outcome: FragmentOutcome,
    token: ActionToken,
    manifest: CandidateFragmentManifest,
    *,
    time_tolerance: float = 1e-6,
    path_tolerance: float = 1e-6,
) -> ProvenanceDecision:
    time_tolerance = finite_number(time_tolerance, "time_tolerance")
    path_tolerance = finite_number(path_tolerance, "path_tolerance")
    if time_tolerance < 0.0 or path_tolerance < 0.0:
        raise ValueError("provenance tolerances must be non-negative")
    if token.manifest_hash != manifest.manifest_hash:
        return _deny("TOKEN_MANIFEST_MISMATCH", token.digest, manifest.manifest_hash)
    if token.context_hash != manifest.context_hash:
        return _deny("TOKEN_CONTEXT_MISMATCH", token.digest, manifest.context_hash)
    expected_fragment_hashes = tuple(fragment.digest for fragment in manifest.fragments)
    if token.planned_fragment_hashes != expected_fragment_hashes:
        return _deny("TOKEN_FRAGMENT_SET_MISMATCH", token.digest, manifest.manifest_hash)
    if outcome.token_hash != token.digest:
        return _deny("OUTCOME_TOKEN_MISMATCH", outcome.digest, token.digest)
    if outcome.manifest_hash != manifest.manifest_hash:
        return _deny("OUTCOME_MANIFEST_MISMATCH", outcome.digest, manifest.manifest_hash)
    if planned.digest not in token.planned_fragment_hashes:
        return _deny("FRAGMENT_NOT_AUTHORIZED", planned.digest, token.digest)
    if outcome.planned_fragment_hash != planned.digest:
        return _deny("PLANNED_FRAGMENT_HASH_MISMATCH", outcome.digest, planned.digest)
    if not outcome.executed:
        return _deny("NOT_EXECUTED", outcome.digest)
    applied = outcome.applied_fragment
    if applied is None:
        return _deny("MISSING_APPLIED_FRAGMENT", outcome.digest)
    if not applied.executed:
        return _deny("APPLIED_FRAGMENT_NOT_MARKED_EXECUTED", applied.digest)
    # A guarded route can still be physically attempted and then time out.
    # Prefer that runtime outcome to the pre-execution guard label so a
    # calibration ledger cannot disguise a censored trace as merely rewritten.
    if dict(outcome.outcome_fields).get("transit_timeout", 0.0) > 0.0:
        return _deny("TRANSIT_TIMEOUT", outcome.digest)
    if applied.guard_rewritten or planned.guard_rewritten:
        return _deny("GUARD_REWRITTEN", planned.digest, applied.digest)
    if (
        applied.instance_fragment_id != planned.instance_fragment_id
        or applied.type_signature.digest != planned.type_signature.digest
    ):
        return _deny("FRAGMENT_ID_OR_TYPE_MISMATCH", planned.digest, applied.digest)
    if (
        outcome.episode_id != token.episode_id
        or outcome.decision_id != token.decision_id
        or applied.episode_id != planned.episode_id
        or applied.decision_id != planned.decision_id
        or applied.agent_id != planned.agent_id
        or outcome.agent_id != planned.agent_id
    ):
        return _deny("EXECUTION_IDENTITY_MISMATCH", outcome.digest, planned.digest)
    # These outcomes have more diagnostic value than a later timing mismatch.
    # In particular, a deadline-truncated transit is a real attempted trace,
    # not an absent action, but it must never enter replay.
    early_outcomes = dict(outcome.outcome_fields)
    if early_outcomes.get("collision", 0.0) > 0.0:
        return _deny("PHYSICAL_COLLISION", outcome.digest)
    if early_outcomes.get("out_of_bounds", 0.0) > 0.0:
        return _deny("FLIGHT_BOUNDS_VIOLATION", outcome.digest)
    if early_outcomes.get("guard_intervention", 0.0) > 0.0:
        return _deny("GUARD_INTERVENED", outcome.digest)
    if early_outcomes.get("static_clearance_contract_violation", 0.0) > 0.0:
        return _deny("STATIC_CLEARANCE_CONTRACT_VIOLATION", outcome.digest)
    if early_outcomes.get("inter_agent_separation_violation", 0.0) > 0.0:
        return _deny("INTER_AGENT_SEPARATION_VIOLATION", outcome.digest)
    # Planned start/end are timing-model predictions.  Event-driven execution
    # records the real applied window and may finish early without changing the
    # command path, so replay eligibility is bound to the authorized token
    # window instead of requiring the controller to match a predicted schedule.
    # Timing-model error remains auditable through the calibrated transit
    # contract and the recorded planned/actual durations.
    token_window_start = token.issued_at
    token_window_end = token.issued_at + token.duration
    if (
        outcome.actual_start < token_window_start - time_tolerance
        or outcome.actual_end > token_window_end + time_tolerance
    ):
        return _deny("TOKEN_TIME_WINDOW_MISMATCH", outcome.digest, token.digest)
    if (
        abs(applied.planned_start - outcome.actual_start) > time_tolerance
        or abs(applied.planned_end - outcome.actual_end) > time_tolerance
    ):
        return _deny("APPLIED_TIME_MISMATCH", outcome.digest, applied.digest)
    if not _path_close(planned.path, applied.path, path_tolerance):
        return _deny("APPLIED_PATH_MISMATCH", planned.digest, applied.digest)
    if planned.pose_mode != applied.pose_mode or planned.context_bucket != applied.context_bucket:
        return _deny("POSE_OR_CONTEXT_MISMATCH", planned.digest, applied.digest)

    fragment_type = planned.type_signature.fragment_type
    if fragment_type == "observation":
        if outcome.source_observation_id is None:
            return _deny("MISSING_SOURCE_OBSERVATION_ID", outcome.digest)
        if (
            outcome.source_observation_episode_id != outcome.episode_id
            or outcome.source_observation_agent_id != outcome.agent_id
        ):
            return _deny("SOURCE_OBSERVATION_IDENTITY_MISMATCH", outcome.digest)
        if applied.source_observation_id != outcome.source_observation_id:
            return _deny("SOURCE_OBSERVATION_MISMATCH", outcome.digest, applied.digest)
        evidence = (
            outcome.range_ok,
            outcome.fov_ok,
            outcome.los_ok,
            outcome.orientation_ok,
            outcome.dwell_ok,
        )
        if any(value is not True for value in evidence):
            return _deny("OBSERVATION_EVIDENCE_INCOMPLETE", outcome.digest)
    elif outcome.source_observation_id is not None:
        return _deny("UNEXPECTED_SOURCE_OBSERVATION", outcome.digest)

    if fragment_type == "communication":
        if outcome.link_window_ok is not True:
            return _deny("COMMUNICATION_LINK_UNVERIFIED", outcome.digest)
        if (
            applied.sender_id != planned.sender_id
            or applied.receiver_id != planned.receiver_id
            or applied.message_digest != planned.message_digest
        ):
            return _deny("COMMUNICATION_PROVENANCE_MISMATCH", planned.digest, applied.digest)

    if not outcome.outcome_fields:
        return _deny("MISSING_ADDITIVE_OUTCOME", outcome.digest)
    evidence_hash = canonical_sha256(
        {
            "planned": planned.to_dict(),
            "applied": applied.to_dict(),
            "outcome": outcome.to_dict(),
            "token": token.to_dict(),
            "manifest_hash": manifest.manifest_hash,
        }
    )
    return ProvenanceDecision(True, "ALLOW", evidence_hash)


def outcome_to_replay(
    planned: FragmentInstance,
    outcome: FragmentOutcome,
    token: ActionToken,
    manifest: CandidateFragmentManifest,
    *,
    time_tolerance: float = 1e-6,
    path_tolerance: float = 1e-6,
) -> tuple[ProvenanceDecision, FragmentReplayRecord | None]:
    decision = evaluate_provenance(
        planned,
        outcome,
        token,
        manifest,
        time_tolerance=time_tolerance,
        path_tolerance=path_tolerance,
    )
    if not decision.allowed:
        return decision, None
    return (
        decision,
        FragmentReplayRecord(
            instance_fragment_id=planned.instance_fragment_id,
            fragment_type_hash=planned.type_signature.digest,
            outcome_hash=outcome.digest,
            context_hash=manifest.context_hash,
            labels=outcome.outcome_fields,
            costs=outcome.cost_fields,
        ),
    )
