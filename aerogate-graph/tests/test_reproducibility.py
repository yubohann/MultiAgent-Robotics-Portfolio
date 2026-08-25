from __future__ import annotations

import math

import pytest

from aerogate import DEFAULT_REPRODUCIBILITY_SEEDS, run_rollout, verify_reproducibility


def test_structured_rollout_preserves_the_public_smoke_contract() -> None:
    summary = run_rollout("single-static", seed=4, steps=2)
    payload = summary.to_dict()
    assert payload["scenario"] == "single-static"
    assert payload["seed"] == 4
    assert payload["steps_executed"] == 2
    assert math.isfinite(float(payload["clearance_m"]))
    assert payload["min_pair_distance_m"] is None
    assert payload["mean_slot_error_m"] is None
    assert payload["max_slot_error_m"] is None
    assert payload["finite_reward"]


def test_seeded_multi_agent_rollouts_are_reproducible() -> None:
    report = verify_reproducibility("multi-static", agents=4, seeds=(3, 7), steps=3)
    assert report.deterministic
    assert report.mismatched_seeds == ()
    assert [rollout.seed for rollout in report.rollouts] == [3, 7]
    assert report.to_dict()["seeds"] == [3, 7]
    assert all(rollout.min_pair_distance_m is not None for rollout in report.rollouts)
    assert all(rollout.mean_slot_error_m is not None for rollout in report.rollouts)
    assert all(rollout.max_slot_error_m is not None for rollout in report.rollouts)


def test_dynamic_rollout_uses_null_for_an_unbounded_clearance_metric() -> None:
    summary = run_rollout("multi-dynamic", agents=8, seed=7, steps=2)
    assert summary.clearance_m is None


def test_reproducibility_check_requires_at_least_one_seed() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        verify_reproducibility("single-static", seeds=())


def test_default_reproducibility_seed_set_is_stable() -> None:
    assert DEFAULT_REPRODUCIBILITY_SEEDS == (3, 7, 11)
