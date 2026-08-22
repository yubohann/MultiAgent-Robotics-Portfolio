from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.ordinary_config import load_ordinary_config, public_execution_contract


def _tool():
    path = Path("tools/cf2x_l1_fleet_preflight.py")
    spec = importlib.util.spec_from_file_location("cf2x_l1_fleet_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layout_contract_mismatch_is_rejected_without_opening_isaac() -> None:
    tool = _tool()
    config = load_ordinary_config(Path("configs/releases/ordinary-v1-mini.json"))
    public_contract = public_execution_contract(config.raw["execution_contract"])
    task = {
        "execution_contract": copy.deepcopy(public_contract),
        "public_execution_contract_hash": content_hash(public_contract),
    }

    tool._validate_layout_execution_contract(task, config)

    task["execution_contract"]["sensor_rig"]["gimbal_mode"] = "fixed"
    with pytest.raises(ValueError, match="differs from release configuration"):
        tool._validate_layout_execution_contract(task, config)
