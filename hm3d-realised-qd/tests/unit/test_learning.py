from __future__ import annotations

import pytest

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import FragmentReplayRecord
from aerocity_method.learning.masked_ppo import MaskedPPO, MaskedPPOConfig
from aerocity_method.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig
from aerocity_method.learning.replay import (
    CandidateTransition,
    FragmentReplayBuffer,
    ReplayBuffer,
    pad_candidate_batch,
)
from aerocity_method.learning.vanilla_sac import VanillaMaskedDiscreteSAC, vanilla_sac_config

try:
    import torch
except ModuleNotFoundError:  # no silent skip: formal G0 requires the dependency
    pytest.fail("PyTorch dependency is required for the learning test suite", pytrace=False)


def transition(**changes):
    values = dict(
        context=(0.0, 1.0),
        candidates=((0.0, 0.0), (1.0, 1.0)),
        legal_mask=(True, True),
        action=1,
        reward=1.0,
        cost=0.1,
        preference=(),
        behavior_features=(),
        next_context=(0.1, 0.9),
        next_candidates=((0.2, 0.1), (0.8, 0.9)),
        next_legal_mask=(True, True),
        next_preference=(),
        done=False,
        duration=2.0,
        outcome_hash=canonical_sha256({"transition": 1}),
    )
    values.update(changes)
    return CandidateTransition(**values)


def test_transition_rejects_illegal_selected_action():
    with pytest.raises(ValueError):
        transition(legal_mask=(True, False), action=1)


def test_transition_rejects_all_masked_next_state():
    with pytest.raises(ValueError):
        transition(next_legal_mask=(False, False))


def test_transition_rejects_dimension_drift():
    with pytest.raises(ValueError):
        transition(next_context=(1.0,))


def test_pad_candidate_batch_marks_padding_illegal():
    batch = pad_candidate_batch(
        [(1.0,), (2.0,)],
        [((1.0, 2.0),), ((3.0, 4.0), (5.0, 6.0))],
        [(True,), (True, False)],
    )
    assert batch.counts == (1, 2)
    assert batch.legal_masks[0] == (True, False)
    assert batch.candidates[0][1] == (0.0, 0.0)


def test_replay_is_bounded_and_rng_restorable():
    replay = ReplayBuffer[int](2, seed=3)
    replay.add(1)
    replay.add(2)
    state = replay.rng_state()
    first = replay.sample(2)
    replay.restore_rng_state(state)
    assert replay.sample(2) == first
    replay.add(3)
    assert len(replay) == 2
    assert 1 not in replay.snapshot()


def test_fragment_replay_deduplicates_outcome_binding():
    record = FragmentReplayRecord(
        instance_fragment_id="f",
        fragment_type_hash=canonical_sha256({"type": 1}),
        outcome_hash=canonical_sha256({"outcome": 1}),
        context_hash=canonical_sha256({"context": 1}),
        labels=(("coverage", 1.0),),
    )
    replay = FragmentReplayBuffer(4)
    replay.add(record)
    replay.add(record)
    assert len(replay) == 1


def test_config_validation():
    with pytest.raises(ValueError):
        RBSFSACConfig(context_dim=0, candidate_dim=2)
    with pytest.raises(ValueError):
        RBSFSACConfig(context_dim=2, candidate_dim=2, gamma=1.1)
    with pytest.raises(ValueError):
        RBSFSACConfig(context_dim=2, candidate_dim=2, cost_limit=-1.0)
    with pytest.raises(ValueError):
        RBSFSACConfig(
            context_dim=2,
            candidate_dim=2,
            cost_multiplier_learning_rate=0.0,
        )
    with pytest.raises(ValueError):
        RBSFSACConfig(
            context_dim=2,
            candidate_dim=2,
            enable_cost_critics=False,
            cost_weight=0.1,
        )


def test_masked_actor_assigns_exact_zero_probability():
    method = RBSFSAC(RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16), seed=1)
    probabilities = method.action_probabilities((0.0, 1.0), ((0.0, 0.0), (1.0, 1.0)), (True, False))
    assert probabilities == (1.0, 0.0)


def test_candidate_permutation_equivariance():
    method = RBSFSAC(RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16), seed=2)
    left = method.action_probabilities((0.0, 1.0), ((0.0, 0.0), (1.0, 1.0)), (True, True))
    right = method.action_probabilities((0.0, 1.0), ((1.0, 1.0), (0.0, 0.0)), (True, True))
    assert right == pytest.approx(tuple(reversed(left)), abs=1e-7)


def test_update_returns_finite_task_cost_diagnostics():
    method = RBSFSAC(RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16), seed=3)
    diagnostics = method.update((transition(), transition(outcome_hash=canonical_sha256({"t": 2}))))
    assert diagnostics["critic_loss"] >= 0.0
    assert diagnostics["cost_loss"] >= 0.0
    assert diagnostics["alpha"] > 0.0


def test_vanilla_masked_discrete_sac_has_no_cost_or_sf_heads():
    config = vanilla_sac_config(context_dim=2, candidate_dim=2, hidden_dim=16)
    method = VanillaMaskedDiscreteSAC(config, seed=13)
    diagnostics = method.update((transition(),))
    assert method.cost1 is None
    assert method.cost2 is None
    assert diagnostics["cost_loss"] == 0.0
    assert diagnostics["cost_multiplier"] == 0.0
    restored = VanillaMaskedDiscreteSAC(config, seed=14)
    restored.load_state_dict(method.state_dict())
    assert restored.cost1 is None


def test_masked_ppo_updates_on_the_same_candidate_transition_contract():
    method = MaskedPPO(MaskedPPOConfig(context_dim=2, candidate_dim=2, hidden_dim=16), seed=15)
    probabilities = method.action_probabilities((0.0, 1.0), ((0.0, 0.0), (1.0, 1.0)), (True, False))
    assert probabilities == (1.0, 0.0)
    diagnostics = method.update((transition(), transition(outcome_hash=canonical_sha256({"p": 2}))))
    assert diagnostics["entropy"] >= 0.0


def test_adaptive_cost_multiplier_is_checkpointed_and_reported():
    config = RBSFSACConfig(
        context_dim=2,
        candidate_dim=2,
        hidden_dim=16,
        cost_limit=0.0,
        initial_cost_multiplier=0.2,
    )
    method = RBSFSAC(config, seed=31)
    with torch.no_grad():
        for network in (method.cost1, method.cost2, method.target_cost1, method.target_cost2):
            for parameter in network.parameters():
                parameter.zero_()
            network.network[-1].bias.fill_(1.0)
    diagnostics = method.update(
        (
            transition(cost=1.0),
            transition(cost=1.0, outcome_hash=canonical_sha256({"t": 31})),
        )
    )
    assert diagnostics["cost_multiplier"] > 0.2
    assert diagnostics["cost_violation"] > 0.0
    restored = RBSFSAC(config, seed=32)
    restored.load_state_dict(method.state_dict())
    assert restored.cost_multiplier is not None
    assert float(restored.cost_multiplier.detach()) == pytest.approx(
        diagnostics["cost_multiplier"], rel=1e-6
    )


def test_sf_compatibility_head_updates_without_direct_actor_projection():
    config = RBSFSACConfig(context_dim=2, candidate_dim=2, sf_dim=2, hidden_dim=16)
    method = RBSFSAC(config, seed=4)
    row = transition(behavior_features=(0.2, 0.8))
    before = [parameter.detach().clone() for parameter in method.actor.parameters()]
    diagnostics = method.update((row,))
    after = list(method.actor.parameters())
    assert diagnostics["sf_loss"] >= 0.0
    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_duration_changes_bootstrap_target():
    config = RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16)
    short = RBSFSAC(config, seed=5)
    long = RBSFSAC(config, seed=5)
    short_target = short.update((transition(duration=0.0),))["task_target_mean"]
    long_target = long.update((transition(duration=4.0),))["task_target_mean"]
    assert short_target != pytest.approx(long_target)


def test_checkpoint_restores_action_rng_and_update_step():
    config = RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16)
    method = RBSFSAC(config, seed=6)
    method.update((transition(),))
    state = method.state_dict()
    expected = [method.select_action((0, 1), ((0, 0), (1, 1)), (True, True)) for _ in range(5)]
    restored = RBSFSAC(config, seed=999)
    restored.load_state_dict(state)
    actual = [restored.select_action((0, 1), ((0, 0), (1, 1)), (True, True)) for _ in range(5)]
    assert actual == expected
    assert restored.update_step == method.update_step


def test_checkpoint_rejects_config_mismatch():
    source = RBSFSAC(RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16))
    target = RBSFSAC(RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=32))
    with pytest.raises(ValueError):
        target.load_state_dict(source.state_dict())
