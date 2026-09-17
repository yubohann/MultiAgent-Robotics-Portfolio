from __future__ import annotations

import numpy as np

from tasks.multi.env.multi_gate_env import MultiGate2DEnv


def test_multi_gate_env_reset_and_step_smoke() -> None:
    env = MultiGate2DEnv()
    try:
        obs, info = env.reset(seed=7, num_agents=2)
        assert "node_features" in obs
        assert info["num_agents"] == 2
        action = np.zeros(env.action_shape, dtype=np.float32)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        assert "node_features" in next_obs
        assert np.isfinite(float(reward))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert next_info["num_agents"] == 2
    finally:
        env.close()

