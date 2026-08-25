from __future__ import annotations

import importlib

from multi_gate.configs.experiment_config import build_fixed_team_experiment_config
from shared.configs.global_config import GLOBAL_CONFIG


def test_fixed_team_presets_cover_original_gate_sizes() -> None:
    expected_sizes = (2, 3, 5, 7, 8, 9, 12, 13, 21, 34)
    assert GLOBAL_CONFIG.max_fixed_team_agents >= max(expected_sizes)
    for team_size in expected_sizes:
        module = importlib.import_module(f"multi_gate.configs.experiment_config_fixed_{team_size:02d}")
        config = module.MULTI_EXPERIMENT_CONFIG
        assert config.default_agents == team_size
        assert config.max_agents_soft == team_size


def test_fixed_team_builder_rejects_out_of_range_values() -> None:
    for bad_size in (0, 1, GLOBAL_CONFIG.max_fixed_team_agents + 1):
        try:
            build_fixed_team_experiment_config(int(bad_size))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for fixed team size {bad_size}")
