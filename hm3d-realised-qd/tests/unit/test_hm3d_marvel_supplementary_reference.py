from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    build_public_candidate_pool,
    identity_path_guard,
)
from aerocity_method.adapters.hm3d_marvel import (
    MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_STATE_KEY,
    MARVEL_SUPPLEMENTARY_REFERENCE_STATE_KEY,
    MarvelSupplementaryReferenceConfig,
    MarvelSupplementaryReferencePolicy,
    MarvelSupplementaryReferenceTrainingRow,
    build_marvel_checkpoint_payload,
    public_marvel_graph_observation,
    select_marvel_supplementary_reference,
)
from aerocity_method.contracts.models import PublicMethodContext
from aerocity_method.contracts.hm3d_public_schema import public_schema_fields

try:
    import torch
except ModuleNotFoundError:
    pytest.fail("PyTorch is required for MARVEL supplementary reference tests", pytrace=False)


SPLIT_HASH = "b" * 64


def _state() -> PublicSearchState:
    context = PublicMethodContext(
        context_id="marvel-context",
        episode_id="marvel-episode",
        decision_id="decision0",
        agent_features=(("uav0", (1.0, 1.0)), ("uav1", (1.0, 1.0))),
        public_features=(("sparse_range_schedule_hz", 10.0),),
        budget=(("time_remaining_s", 40.0),),
    )
    return PublicSearchState(
        context=context,
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 1),
            PublicAgentPose("uav1", (3.0, 0.0, 2.0), 0.9, 1),
        ),
        frontiers=(
            PublicFrontier("f0", (1.0, 1.0, 1.0), 0.8, 0.1),
            PublicFrontier("f1", (4.0, -1.0, 2.5), 0.9, 0.2),
            PublicFrontier("f2", (4.0, 1.0, 3.0), 0.7, 0.1),
        ),
        decision_start_s=0.0,
        decision_duration_s=20.0,
        transit_timing_model=ConservativeTransitTimingModel("unit", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
    )


def _checkpoint(tmp_path) -> object:
    model = MarvelSupplementaryReferencePolicy(MarvelSupplementaryReferenceConfig(hidden_dim=8), seed=12)
    payload = build_marvel_checkpoint_payload(
        model,
        training_scene_ids=("00244-E64sjs3Dyfd",),
        training_updates=1,
        training_provenance={
            "real_runtime_outcomes": ["a" * 64],
            "split_manifest_sha256": SPLIT_HASH,
            **public_schema_fields(),
        },
        split_manifest_sha256=SPLIT_HASH,
    )
    path = tmp_path / "marvel-supplementary-reference.pt"
    torch.save(payload, path)
    return path


def test_marvel_supplementary_reference_masks_illegal_actions_and_records_train_provenance(tmp_path) -> None:
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    selected, selection = select_marvel_supplementary_reference(
        state,
        pool,
        checkpoint_path=_checkpoint(tmp_path),
        expected_split_manifest_sha256=SPLIT_HASH,
    )
    assert selected.feasible
    assert selection.to_dict()["strategy"] == "marvel_supplementary_reference"
    assert sum(score for _, score in selection.scores) == pytest.approx(1.0)


def test_marvel_supplementary_reference_executes_a_real_gradient_update_on_public_graph_rows() -> None:
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    model = MarvelSupplementaryReferencePolicy(MarvelSupplementaryReferenceConfig(hidden_dim=8), seed=3)
    observation = public_marvel_graph_observation(state, pool)
    row = MarvelSupplementaryReferenceTrainingRow(
        observation=observation,
        next_observation=observation,
        action=0,
        reward=0.25,
        duration_s=5.0,
        done=False,
    )
    diagnostics = model.update((row,))
    assert {"policy_loss", "q1_loss", "q2_loss", "alpha", "entropy"} <= set(diagnostics)
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())


def test_marvel_author_graph_and_policy_respond_to_vertical_candidate_change() -> None:
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    candidate = pool[0]
    transit_index = next(
        index
        for index, fragment in enumerate(candidate.fragments)
        if fragment.type_signature.fragment_type == "transit"
    )
    fragment = candidate.fragments[transit_index]
    endpoint = fragment.path[-1]
    lifted_fragment = replace(
        fragment,
        path=(*fragment.path[:-1], (endpoint[0], endpoint[1], endpoint[2] + 2.0)),
    )
    fragments = list(candidate.fragments)
    fragments[transit_index] = lifted_fragment
    lifted_pool = (replace(candidate, fragments=tuple(fragments)), *pool[1:])
    base_graph = public_marvel_graph_observation(state, pool)
    lifted_graph = public_marvel_graph_observation(state, lifted_pool)
    assert base_graph.node_inputs[1] != lifted_graph.node_inputs[1]
    model = MarvelSupplementaryReferencePolicy(MarvelSupplementaryReferenceConfig(hidden_dim=8), seed=9)
    assert model.action_probabilities(base_graph) != pytest.approx(
        model.action_probabilities(lifted_graph)
    )


def test_marvel_supplementary_reference_rejects_a_checkpoint_from_another_frozen_split(tmp_path) -> None:
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    with pytest.raises(ValueError, match="different frozen scene split"):
        select_marvel_supplementary_reference(
            state,
            pool,
            checkpoint_path=_checkpoint(tmp_path),
            expected_split_manifest_sha256="c" * 64,
        )


def test_marvel_supplementary_reference_loads_legacy_checkpoint_state_key(tmp_path) -> None:
    model = MarvelSupplementaryReferencePolicy(MarvelSupplementaryReferenceConfig(hidden_dim=8), seed=14)
    payload = build_marvel_checkpoint_payload(
        model,
        training_scene_ids=("00244-E64sjs3Dyfd",),
        training_updates=1,
        training_provenance={
            "real_runtime_outcomes": ["a" * 64],
            "split_manifest_sha256": SPLIT_HASH,
            **public_schema_fields(),
        },
        split_manifest_sha256=SPLIT_HASH,
    )
    payload[MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_STATE_KEY] = payload.pop(
        MARVEL_SUPPLEMENTARY_REFERENCE_STATE_KEY
    )
    path = tmp_path / "legacy-marvel-port.pt"
    torch.save(payload, path)

    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    selected, _ = select_marvel_supplementary_reference(
        state,
        pool,
        checkpoint_path=path,
        expected_split_manifest_sha256=SPLIT_HASH,
    )

    assert selected.feasible
