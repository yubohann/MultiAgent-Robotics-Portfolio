from __future__ import annotations

import pytest

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    build_public_candidate_pool,
    identity_path_guard,
)
from aerocity_method.adapters.hm3d_single_rl import (
    SINGLE_RL_CHECKPOINT_SCHEMA_VERSION,
    build_single_rl_checkpoint_payload,
    public_candidate_features,
    public_context_features,
    select_single_rl,
)
from aerocity_method.contracts.models import PublicMethodContext
from aerocity_method.contracts.hm3d_public_schema import public_schema_fields
from aerocity_method.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig

try:
    import torch
except ModuleNotFoundError:  # no silent skip: the real Isaac worker needs this dependency
    pytest.fail("PyTorch is required for the single-RL baseline tests", pytrace=False)


SPLIT_HASH = "b" * 64


def _state() -> PublicSearchState:
    context = PublicMethodContext(
        context_id="p07-single-rl-context",
        episode_id="p07-single-rl-episode",
        decision_id="decision0",
        agent_features=(("uav0", (1.0, 1.0)), ("uav1", (1.0, 1.0))),
        public_features=(("sparse_range_schedule_hz", 10.0),),
        budget=(("time_remaining_s", 40.0),),
    )
    return PublicSearchState(
        context=context,
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 1),
            PublicAgentPose("uav1", (2.0, 0.0, 1.5), 1.0, 1),
        ),
        frontiers=(
            PublicFrontier("frontier0", (1.0, 1.0, 1.0), 0.8, 0.1),
            PublicFrontier("frontier1", (3.0, -1.0, 2.0), 0.9, 0.2),
            PublicFrontier("frontier2", (4.0, 1.0, 2.5), 0.6, 0.1),
        ),
        decision_start_s=0.0,
        decision_duration_s=40.0,
        transit_timing_model=ConservativeTransitTimingModel("unit", 2.0, 2.0, 0.1),
        observe_dwell_s=0.5,
    )


def _checkpoint(tmp_path):
    model = RBSFSAC(RBSFSACConfig(context_dim=4, candidate_dim=5, hidden_dim=8), seed=7)
    payload = build_single_rl_checkpoint_payload(
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
    path = tmp_path / "single-rl.pt"
    torch.save(payload, path)
    return path


def test_single_rl_loads_a_train_provenanced_checkpoint_and_selects_only_public_features(tmp_path):
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    selected, selection = select_single_rl(
        state,
        pool,
        checkpoint_path=_checkpoint(tmp_path),
        expected_split_manifest_sha256=SPLIT_HASH,
    )
    assert selected.feasible
    assert selection.to_dict()["strategy"] == "single_rl"
    assert len(public_context_features(state)) == 4
    assert len(public_candidate_features(pool[0])) == 5
    assert sum(score for _, score in selection.scores) == pytest.approx(1.0)


def test_single_rl_checkpoint_builder_rejects_an_untrained_payload():
    model = RBSFSAC(RBSFSACConfig(context_dim=4, candidate_dim=5, hidden_dim=8), seed=8)
    with pytest.raises(ValueError, match="training_updates"):
        build_single_rl_checkpoint_payload(
            model,
            training_scene_ids=("00244-E64sjs3Dyfd",),
            training_updates=0,
            training_provenance={
                "real_runtime_outcomes": ["a" * 64],
                "split_manifest_sha256": SPLIT_HASH,
            },
            split_manifest_sha256=SPLIT_HASH,
        )


def test_single_rl_rejects_a_checkpoint_from_another_frozen_split(tmp_path) -> None:
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=3)
    with pytest.raises(ValueError, match="different frozen scene split"):
        select_single_rl(
            state,
            pool,
            checkpoint_path=_checkpoint(tmp_path),
            expected_split_manifest_sha256="c" * 64,
        )


def test_single_rl_checkpoint_schema_is_explicit():
    assert SINGLE_RL_CHECKPOINT_SCHEMA_VERSION == "hm3d-p07-single-rl-checkpoint-v2"
