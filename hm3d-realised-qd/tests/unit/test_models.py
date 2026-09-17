from __future__ import annotations

from dataclasses import replace

import pytest

from realised_qd.contracts.models import (
    ABI_VERSION,
    BudgetLedger,
    CandidateGraphBatch,
    FragmentInstance,
    FragmentOutcome,
    FragmentReplayRecord,
    FragmentTypeSignature,
    InteractionEdge,
    PublicMethodContext,
    ReplayDecision,
)


def test_fragment_type_rejects_unknown_kind():
    with pytest.raises(ValueError):
        FragmentTypeSignature("latent")


def test_feature_pairs_are_sorted(context):
    signature = FragmentTypeSignature("hold", (("z", 1), ("a", "x")))
    assert tuple(dict(signature.public_features)) == ("a", "z")


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


def test_communication_requires_distinct_endpoints_and_message_id(context):
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
        message_id="message-1",
    )
    assert fragment.message_id == "message-1"
    with pytest.raises(ValueError):
        replace(fragment, receiver_id="uav-1")
    with pytest.raises(ValueError):
        replace(fragment, message_id=None)


def test_public_context_agent_order_is_canonical(context):
    reversed_context = replace(context, agent_features=tuple(reversed(context.agent_features)))
    assert context.to_dict() == reversed_context.to_dict()
    assert context.context_id == "hm3d-exploration-test-context"


def test_public_context_rejects_non_numeric_features():
    with pytest.raises(ValueError):
        PublicMethodContext(
            context_id="c",
            episode_id="e",
            decision_id="d",
            agent_features=(("uav", ("not-a-number",)),),
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
    edge = InteractionEdge("fragment-b", "fragment-a", "collision", 1.0)
    assert edge.source_fragment_id < edge.target_fragment_id
    assert edge.source_fragment_id == "fragment-a"


def test_candidate_graph_rejects_unknown_membership(manifests):
    candidate = manifests[0].manifest_id
    fragment = manifests[0].fragments[0].instance_fragment_id
    with pytest.raises(ValueError):
        CandidateGraphBatch(
            candidate_ids=(candidate,),
            fragment_ids=(fragment,),
            membership_edges=((candidate, "unknown-fragment", 0),),
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
            token_id=token.token_id,
            manifest_id=manifests[0].manifest_id,
            episode_id=planned.episode_id,
            decision_id=planned.decision_id,
            agent_id=planned.agent_id,
            planned_fragment_id=planned.instance_fragment_id,
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
            fragment_type=manifests[0].fragments[0].type_signature.fragment_type,
            outcome_id=outcome.outcome_id,
            context_id=manifests[0].context_id,
            labels=(),
        )


def test_replay_decision_reason_consistency():
    with pytest.raises(ValueError):
        ReplayDecision(True, "DENY")
    with pytest.raises(ValueError):
        ReplayDecision(False, "ALLOW")
    assert ReplayDecision(True, "ALLOW").allowed


def test_budget_ledger_rejects_unknown_or_excess_usage():
    ledger = BudgetLedger(limits=(("wall", 10.0),), used=(("wall", 3.0),))
    assert dict(ledger.used)["wall"] == 3.0
    with pytest.raises(ValueError):
        BudgetLedger(limits=(("wall", 1.0),), used=(("wall", 2.0),))


def test_schema_version_is_strict(context):
    with pytest.raises(ValueError):
        replace(context, schema_version="future")
    assert context.schema_version == ABI_VERSION
