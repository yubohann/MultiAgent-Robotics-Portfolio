"""Fixed 8-agent config preset for the multi-agent gate experiment."""

from __future__ import annotations

from multi_gate.configs.experiment_config import build_fixed_team_experiment_config


TEAM_SIZE = 8
MULTI_EXPERIMENT_CONFIG = build_fixed_team_experiment_config(TEAM_SIZE)

