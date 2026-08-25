from __future__ import annotations

from large_scale_50v50_battle.config import BattleConfig, DEFAULT_THETA
from large_scale_50v50_battle.sim import LargeScaleBattle50v50


def test_large_scale_simulator_is_seed_deterministic_for_short_episode():
    config = BattleConfig(agents_per_team=4, max_steps=8)
    simulator = LargeScaleBattle50v50(config)
    first = simulator.run_episode(DEFAULT_THETA, DEFAULT_THETA, seed=17)
    second = simulator.run_episode(DEFAULT_THETA, DEFAULT_THETA, seed=17)
    assert first == second
    assert first["yellow_alive"] <= config.agents_per_team
    assert first["blue_alive"] <= config.agents_per_team
    assert first["steps"] <= config.max_steps
