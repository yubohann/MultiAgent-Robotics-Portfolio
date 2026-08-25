from __future__ import annotations

import numpy as np

from multi_gate.configs.experiment_config import build_dynamic_gate_density_8d_config
from multi_gate.env.multi_gate_env import MultiGate2DEnv


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


def test_dynamic_gate_action_shield_reports_structured_diagnostics() -> None:
    env = MultiGate2DEnv(multi_config=build_dynamic_gate_density_8d_config())
    try:
        env.reset(seed=7, num_agents=8)
        action = np.zeros(env.action_shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        diagnostics = info["action_safety_shield"]
        assert diagnostics["enabled"] is True
        assert isinstance(diagnostics["active"], bool)
        assert np.isfinite(float(diagnostics["mean_intervention_norm"]))
        assert np.isfinite(float(diagnostics["max_pair_closeness"]))
        assert info["live_gate_centers_xy"].shape == (info["dynamic_gate_count"], 2)
        assert info["live_gate_post_positions_xy"].shape == (2 * info["dynamic_gate_count"], 2)
    finally:
        env.close()
