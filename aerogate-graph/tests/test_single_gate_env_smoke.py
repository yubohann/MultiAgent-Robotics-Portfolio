from __future__ import annotations

import numpy as np

from single_gate.env.single_gate_env import SingleGate2DEnv


def test_single_gate_env_reset_and_step_smoke() -> None:
    env = SingleGate2DEnv()
    obs, info = env.reset(seed=5)
    assert "node_features" in obs
    assert "adjacency" in obs
    assert "node_mask" in obs
    assert "goal_distance_m" in info
    action = np.zeros(env.action_shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    assert "node_features" in next_obs
    assert np.isfinite(float(reward))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "goal_distance_m" in next_info
