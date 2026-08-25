from __future__ import annotations

from dataclasses import replace

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentOutcome,
    FragmentTypeSignature,
)
from aerocity_method.fragments.provenance import (
    OGFRReuseContext,
    evaluate_provenance,
    evaluate_ogfr_reuse_context,
    outcome_to_replay,
)
from aerocity_method.runtime.tokens import authorize_manifest


def test_transit_outcome_is_allowed(manifests, outcomes, token):
    decision = evaluate_provenance(manifests[0].fragments[0], outcomes[0], token, manifests[0])
    assert decision.allowed
    assert decision.reason_code == "ALLOW"


def test_observation_requires_complete_evidence(manifests, outcomes, token):
    planned = manifests[0].fragments[1]
    assert evaluate_provenance(planned, outcomes[1], token, manifests[0]).allowed
    denied = evaluate_provenance(planned, replace(outcomes[1], los_ok=False), token, manifests[0])
    assert denied.reason_code == "OBSERVATION_EVIDENCE_INCOMPLETE"


def test_missing_source_observation_fails_closed(manifests, outcomes, token):
    planned = manifests[0].fragments[1]
    denied = evaluate_provenance(
        planned,
        replace(
            outcomes[1],
            source_observation_id=None,
            source_observation_episode_id=None,
            source_observation_agent_id=None,
        ),
        token,
        manifests[0],
    )
    assert denied.reason_code == "MISSING_SOURCE_OBSERVATION_ID"


def test_token_hash_mismatch_is_denied(manifests, outcomes, token):
    denied = evaluate_provenance(
        manifests[0].fragments[0],
        replace(outcomes[0], token_hash=canonical_sha256({"wrong": 1})),
        token,
        manifests[0],
    )
    assert denied.reason_code == "OUTCOME_TOKEN_MISMATCH"


def test_unexecuted_fragment_has_no_replay(manifests, token):
    planned = manifests[0].fragments[0]
    outcome = FragmentOutcome(
        outcome_id="not-executed",
        token_hash=token.digest,
        manifest_hash=manifests[0].manifest_hash,
        episode_id=planned.episode_id,
        decision_id=planned.decision_id,
        agent_id=planned.agent_id,
        planned_fragment_hash=planned.digest,
        executed=False,
        actual_start=0,
        actual_end=0,
    )
    decision, replay = outcome_to_replay(planned, outcome, token, manifests[0])
    assert decision.reason_code == "NOT_EXECUTED"
    assert replay is None


def test_guard_rewrite_denies_planned_fragment(manifests, outcomes, token):
    applied = replace(outcomes[0].applied_fragment, guard_rewritten=True)
    denied = evaluate_provenance(
        manifests[0].fragments[0],
        replace(outcomes[0], applied_fragment=applied),
        token,
        manifests[0],
    )
    assert denied.reason_code == "GUARD_REWRITTEN"


def test_path_mismatch_is_denied_and_event_driven_early_finish_is_allowed(
    manifests, outcomes, token
):
    applied = replace(outcomes[0].applied_fragment, path=((0, 0, 1), (99, 0, 1)))
    assert (
        evaluate_provenance(
            manifests[0].fragments[0],
            replace(outcomes[0], applied_fragment=applied),
            token,
            manifests[0],
        ).reason_code
        == "APPLIED_PATH_MISMATCH"
    )
    early_applied = replace(outcomes[0].applied_fragment, planned_end=0.6)
    decision = evaluate_provenance(
        manifests[0].fragments[0],
        replace(outcomes[0], actual_end=0.6, applied_fragment=early_applied),
        token,
        manifests[0],
    )
    assert decision.allowed


def test_outcome_outside_token_window_is_denied(manifests, outcomes, token):
    assert (
        evaluate_provenance(
            manifests[0].fragments[0],
            replace(outcomes[0], actual_end=token.duration + 0.5),
            token,
            manifests[0],
        ).reason_code
        == "TOKEN_TIME_WINDOW_MISMATCH"
    )


def test_agent_identity_mismatch_is_denied(manifests, outcomes, token):
    denied = evaluate_provenance(
        manifests[0].fragments[0],
        replace(outcomes[0], agent_id="uav-other"),
        token,
        manifests[0],
    )
    assert denied.reason_code == "EXECUTION_IDENTITY_MISMATCH"


def test_unauthorized_fragment_is_denied(context, manifests, outcomes):
    token = authorize_manifest(
        context,
        manifests,
        (False, True, False),
        1,
        token_id="token-B",
        issued_at=0,
        duration=2,
    )
    forged = replace(
        outcomes[0],
        token_hash=token.digest,
        manifest_hash=manifests[1].manifest_hash,
    )
    denied = evaluate_provenance(manifests[0].fragments[0], forged, token, manifests[1])
    assert denied.reason_code == "FRAGMENT_NOT_AUTHORIZED"


def test_allowed_outcome_creates_replay_record(manifests, outcomes, token):
    decision, replay = outcome_to_replay(
        manifests[0].fragments[0], outcomes[0], token, manifests[0]
    )
    assert decision.allowed
    assert replay.labels == outcomes[0].outcome_fields


def test_communication_requires_measured_link(context):
    signature = FragmentTypeSignature("communication")
    planned = FragmentInstance(
        instance_fragment_id="comm-1",
        type_signature=signature,
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        agent_id="uav-1",
        planned_start=0,
        planned_end=1,
        sender_id="uav-1",
        receiver_id="uav-2",
        message_digest=canonical_sha256({"message": "x"}),
    )
    manifest = CandidateFragmentManifest(
        candidate_id="comm-candidate",
        context_hash=context.digest,
        fragments=(planned,),
        planned_descriptor=(0.5, 0.5),
        feasible=True,
    )
    token = authorize_manifest(
        context, (manifest,), (True,), 0, token_id="comm-token", issued_at=0, duration=1
    )
    applied = replace(planned, executed=True)
    outcome = FragmentOutcome(
        outcome_id="comm-outcome",
        token_hash=token.digest,
        manifest_hash=manifest.manifest_hash,
        episode_id=planned.episode_id,
        decision_id=planned.decision_id,
        agent_id=planned.agent_id,
        planned_fragment_hash=planned.digest,
        executed=True,
        actual_start=0,
        actual_end=1,
        applied_fragment=applied,
        outcome_fields=(("message_delivery", 1.0),),
        link_window_ok=False,
    )
    denied = evaluate_provenance(planned, outcome, token, manifest)
    allowed = evaluate_provenance(planned, replace(outcome, link_window_ok=True), token, manifest)
    assert denied.reason_code == "COMMUNICATION_LINK_UNVERIFIED"
    assert allowed.allowed


def _reuse_context(scene_id: str, **changes) -> OGFRReuseContext:
    values = dict(
        domain_id="hm3d-indoor",
        scene_id=scene_id,
        layout_hash=canonical_sha256({"layout": scene_id}),
        structure_signature_hash=canonical_sha256({"structure": "office"}),
        flight_space_hash=canonical_sha256({"flight": scene_id}),
        controller_version="controller-v1",
        dynamics_version="quadrotor-v1",
        sensor_profile_hash=canonical_sha256({"sensor": 1}),
        agent_role="searcher",
    )
    values.update(changes)
    return OGFRReuseContext(**values)


def test_ogfr_same_scene_reuse_rejects_geometry_rebinding():
    source = _reuse_context("scene-a")
    rebound = _reuse_context("scene-a", flight_space_hash=canonical_sha256({"forged": 1}))
    decision = evaluate_ogfr_reuse_context(source, rebound, scope="within_scene")
    assert not decision.proposal_allowed
    assert decision.reason_code == "OGFR_SCENE_GEOMETRY_REBOUND"
    assert not decision.supervision_transfer_allowed


def test_ogfr_cross_scene_proposal_never_transfers_reward_labels():
    source = _reuse_context("scene-a")
    target = _reuse_context("scene-b")
    decision = evaluate_ogfr_reuse_context(source, target, scope="same_structure")
    assert decision.proposal_allowed
    assert decision.residual_replanning_required
    assert not decision.supervision_transfer_allowed


def test_ogfr_cross_structure_needs_similarity_and_matching_runtime_context():
    source = _reuse_context("scene-a")
    target = _reuse_context(
        "scene-b",
        structure_signature_hash=canonical_sha256({"structure": "warehouse"}),
    )
    missing = evaluate_ogfr_reuse_context(source, target, scope="cross_structure_candidate")
    assert missing.reason_code == "OGFR_SIMILARITY_EVIDENCE_MISSING"
    allowed = evaluate_ogfr_reuse_context(
        source,
        target,
        scope="cross_structure_candidate",
        embedding_similarity=0.9,
        minimum_similarity=0.8,
    )
    assert allowed.proposal_allowed
    assert not allowed.supervision_transfer_allowed
    mismatched_sensor = _reuse_context(
        "scene-b",
        sensor_profile_hash=canonical_sha256({"sensor": 2}),
    )
    denied = evaluate_ogfr_reuse_context(
        source,
        mismatched_sensor,
        scope="cross_structure_candidate",
        embedding_similarity=1.0,
    )
    assert denied.reason_code == "OGFR_CONTEXT_MISMATCH"
