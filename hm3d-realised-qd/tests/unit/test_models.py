from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import (
    ABI_VERSION,
    BudgetLedger,
    CandidateGraphBatch,
    FragmentInstance,
    FragmentOutcome,
    FragmentReplayRecord,
    FragmentTypeSignature,
    InteractionEdge,
    ProvenanceDecision,
    PublicMethodContext,
)
from aerocity_method.contracts.privacy import PublicBoundaryError


def test_fragment_type_rejects_unknown_kind():
    with pytest.raises(ValueError):
        FragmentTypeSignature("latent")


def test_feature_pairs_are_sorted_and_duplicate_keys_rejected():
    signature = FragmentTypeSignature("hold", (("z", 1), ("a", "x")))
    assert tuple(dict(signature.public_features)) == ("a", "z")
    with pytest.raises(ValueError):
        FragmentTypeSignature("hold", (("a", 1), ("a", 2)))


def test_transit_requires_two_points(context):
    with pytest.raises(ValueError):
        FragmentInstance(
            instance_fragment_id="f",
            type_signature=FragmentTypeSignature("transit"),
            episode_id=context.episode_id,
            decision_id=context.decision_id,
            agent_id="uav-1",
            planned_start=0,
            planned_end=1,
            path=((0, 0, 0),),
        )


def test_communication_requires_distinct_endpoints_and_digest(context):
    digest = canonical_sha256({"message": 1})
    fragment = FragmentInstance(
        instance_fragment_id="communication-1",
        type_signature=FragmentTypeSignature("communication"),
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        agent_id="uav-1",
        planned_start=0,
        planned_end=1,
        sender_id="uav-1",
        receiver_id="uav-2",
        message_digest=digest,
    )
    assert fragment.message_digest == digest
    with pytest.raises(ValueError):
        replace(fragment, receiver_id="uav-1")


def test_public_context_is_order_canonical(context):
    reversed_context = replace(context, agent_features=tuple(reversed(context.agent_features)))
    assert context.digest == reversed_context.digest


def test_public_context_rejects_private_field():
    with pytest.raises(PublicBoundaryError):
        PublicMethodContext(
            context_id="c",
            episode_id="e",
            decision_id="d",
            agent_features=(("uav", (1.0,)),),
            public_features=(("target_id", "secret"),),
        )


def test_manifest_requires_two_descriptor_dimensions(manifests):
    with pytest.raises(ValueError):
        replace(manifests[0], planned_descriptor=(0.5,))


def test_manifest_rejects_duplicate_fragment_ids(manifests):
    first = manifests[0].fragments[0]
    conflicting = replace(
        manifests[0].fragments[1], instance_fragment_id=first.instance_fragment_id
    )
    with pytest.raises(ValueError):
        replace(manifests[0], fragments=(first, conflicting))


def test_interaction_edge_is_canonical():
    left = canonical_sha256({"x": 1})
    right = canonical_sha256({"x": 2})
    edge = InteractionEdge(max(left, right), min(left, right), "collision", 1.0)
    assert edge.source_fragment_hash < edge.target_fragment_hash


def test_candidate_graph_rejects_unknown_membership(manifests):
    candidate = manifests[0].manifest_hash
    fragment = manifests[0].fragments[0].digest
    with pytest.raises(ValueError):
        CandidateGraphBatch(
            candidate_hashes=(candidate,),
            fragment_hashes=(fragment,),
            membership_edges=((candidate, canonical_sha256({"unknown": 1}), 0),),
            interaction_edges=(),
        )


def test_action_token_rejects_invalid_time(token):
    with pytest.raises(ValueError):
        replace(token, duration=0.0)


def test_unexecuted_outcome_cannot_carry_labels(manifests, token):
    planned = manifests[0].fragments[0]
    with pytest.raises(ValueError):
        FragmentOutcome(
            outcome_id="r",
            token_hash=token.digest,
            manifest_hash=manifests[0].manifest_hash,
            episode_id=planned.episode_id,
            decision_id=planned.decision_id,
            agent_id=planned.agent_id,
            planned_fragment_hash=planned.digest,
            executed=False,
            actual_start=0,
            actual_end=0,
            outcome_fields=(("coverage", 1.0),),
        )


def test_replay_requires_label(outcomes, manifests):
    outcome = outcomes[0]
    with pytest.raises(ValueError):
        FragmentReplayRecord(
            instance_fragment_id=manifests[0].fragments[0].instance_fragment_id,
            fragment_type_hash=manifests[0].fragments[0].type_signature.digest,
            outcome_hash=outcome.digest,
            context_hash=manifests[0].context_hash,
            labels=(),
        )


def test_provenance_decision_reason_consistency():
    with pytest.raises(ValueError):
        ProvenanceDecision(True, "DENY")
    with pytest.raises(ValueError):
        ProvenanceDecision(False, "ALLOW")


def test_budget_ledger_rejects_unknown_or_excess_usage():
    ledger = BudgetLedger(limits=(("wall", 10.0),), used=(("wall", 3.0),))
    assert dict(ledger.used)["wall"] == 3.0
    with pytest.raises(ValueError):
        BudgetLedger(limits=(("wall", 1.0),), used=(("wall", 2.0),))


def test_schema_version_is_strict(context):
    with pytest.raises(ValueError):
        replace(context, schema_version="future")
    assert context.schema_version == ABI_VERSION
