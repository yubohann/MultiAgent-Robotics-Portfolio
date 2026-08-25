from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aerogate import available_scenarios, build_environment, run_smoke
from aerogate.cli import build_parser, main
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from single_gate.env.single_gate_env import SingleGate2DEnv


def test_public_scenarios_create_expected_environment_families() -> None:
    assert {scenario.name for scenario in available_scenarios()} == {
        "single-static",
        "multi-static",
        "multi-dynamic",
    }
    assert isinstance(build_environment("single-static"), SingleGate2DEnv)
    static_environment = build_environment("multi-static", agents=2)
    dynamic_environment = build_environment("multi-dynamic", agents=8)
    assert isinstance(static_environment, MultiGate2DEnv)
    assert isinstance(dynamic_environment, MultiGate2DEnv)
    static_environment.close()
    dynamic_environment.close()


def test_public_scenarios_reject_invalid_team_sizes_early() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_environment("single-static", agents=2)
    with pytest.raises(ValueError, match="at least two"):
        build_environment("multi-static", agents=1)


def test_public_smoke_returns_finite_json_safe_summary() -> None:
    summary = run_smoke("single-static", seed=4, steps=2)
    assert summary["scenario"] == "single-static"
    assert summary["steps_executed"] == 2
    assert summary["observation_nodes"] > 0
    assert summary["observation_feature_dim"] > 0
    assert math.isfinite(float(summary["clearance_m"]))
    assert summary["min_pair_distance_m"] is None
    assert math.isfinite(float(summary["reward_total"]))


def test_multi_agent_public_smoke_returns_episode_progress() -> None:
    summary = run_smoke("multi-static", agents=2, seed=4, steps=2)
    assert summary["scenario"] == "multi-static"
    assert summary["num_agents"] == 2
    assert summary["steps_executed"] == 2
    assert math.isfinite(float(summary["clearance_m"]))
    assert math.isfinite(float(summary["min_pair_distance_m"]))
    assert math.isfinite(float(summary["mean_slot_error_m"]))
    assert math.isfinite(float(summary["max_slot_error_m"]))
    assert math.isfinite(float(summary["reward_total"]))


def test_public_cli_reports_the_package_version(capsys: object) -> None:
    try:
        build_parser().parse_args(["--version"])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("--version should exit after printing the package version")
    assert "aerogate 0.2.0" in capsys.readouterr().out


def test_reproducibility_cli_writes_the_same_evidence_to_stdout_and_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "reports" / "reproducibility.json"
    main(
        [
            "reproduce",
            "--scenario",
            "multi-static",
            "--agents",
            "4",
            "--seeds",
            "3",
            "--steps",
            "2",
            "--output",
            str(output_path),
        ]
    )
    printed = capsys.readouterr().out
    assert output_path.read_text(encoding="utf-8") == printed
    payload = json.loads(printed)
    assert payload["deterministic"]
    assert payload["provenance"]["aerogate_version"] == "0.2.0"
    assert payload["schema_version"] == 1
    assert "Infinity" not in printed
