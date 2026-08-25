from __future__ import annotations

import numpy as np

from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv, TACTICAL_ACTION_DIM


def test_selfplay_reset_and_zero_action_step_are_well_formed():
    env = RoboCupVisionRLSelfPlayEnv(max_time_s=1.0)
    observations, infos = env.reset(seed=7)
    assert set(observations) == set(AGENTS)
    assert set(infos) == set(AGENTS)
    actions = {team: np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32) for team in AGENTS}
    next_observations, rewards, terminations, truncations, step_infos = env.step(actions)
    assert set(next_observations) == set(AGENTS)
    assert set(rewards) == set(AGENTS)
    assert set(terminations) == set(AGENTS)
    assert set(truncations) == set(AGENTS)
    assert set(step_infos) == set(AGENTS)
    assert all(np.isfinite(value) for value in rewards.values())
