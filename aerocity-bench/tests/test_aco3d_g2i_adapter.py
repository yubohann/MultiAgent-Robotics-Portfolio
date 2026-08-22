from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_adapter():
    path = Path("tools/aco3d_g2i_process_adapter.py")
    spec = importlib.util.spec_from_file_location("aco3d_g2i_process_adapter_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _public_reset_inputs() -> tuple[dict[str, object], dict[str, object]]:
    cells = [
        {
            "cell_id": f"public-cell-{index:02d}",
            "pose": {
                "position": [2.0 + index, float(index % 2), 2.0 + 0.25 * index],
                "yaw_deg": float(index * 15),
                "pitch_deg": -30.0,
            },
            "represented_area_m2": 1.0 + index,
        }
        for index in range(3)
    ]
    task: dict[str, object] = {
        "task_track": "G2-I",
        "inspection_atlas": {
            "atlas_hash": "a" * 64,
            "regions": [{"region_id": "r", "cells": cells}],
        },
        "execution_contract": {
            "control_period_s": 0.2,
            "episode": {"duration_s": 300.0},
            "observe": {
                "continuous_dwell_s": 0.5,
                "max_linear_speed_mps": 0.25,
                "max_angular_speed_deg_s": 8.0,
            },
            "vehicle": {
                "horizontal_speed_mps": 3.0,
                "vertical_speed_mps": 2.0,
                "radius_m": 0.1,
                "minimum_clearance_m": 0.1,
            },
        },
        "public_transit_contract": {"safe_sky_altitude_m": 8.0},
        "flight_bounds": {"maximum": [20.0, 20.0, 20.0]},
    }
    episode: dict[str, object] = {
        "starts": [{"drone_id": "uav-00", "position": [0.0, 0.0, 2.0]}],
        "mission_sector": {
            "truth_independent": True,
            "frozen_before_sampling": True,
            "atlas_hash": "a" * 64,
            "selected_cell_ids": [cell["cell_id"] for cell in cells],
            "cell_assignment_by_drone": {"uav-00": [cell["cell_id"] for cell in cells]},
            "capacity_certificate": {
                "return_reserve_s": 35.0,
                "horizontal_speed_mps": 1.5,
                "vertical_speed_mps": 1.0,
            },
        },
    }
    return episode, task


def test_source_lock_is_pinned_and_honestly_marks_translation() -> None:
    module = _load_adapter()
    lock = module._source_lock()

    assert lock["upstream"]["commit"] == module.UPSTREAM_COMMIT
    assert lock["upstream"]["license"] == "MIT"
    assert lock["upstream"]["paper_doi"] == "10.1109/SII58957.2024.10417512"
    assert lock["adapter"]["upstream_runtime_executed"] is False
    assert lock["adapter"]["fixed_parameters"]["iterations"] == 350
    assert set(lock["upstream"]["source_file_sha256"]) == {
        "CreateModel.m",
        "TourCost.m",
        "main.m",
    }


def test_source_cost_formula_matches_create_model_and_tour_cost() -> None:
    module = _load_adapter()
    points = [(0.0, 0.0, 0.0), (3.0, 4.0, 1.0), (60.0, 0.0, 0.0)]
    matrix = module._source_distance_matrix(points)

    assert matrix[0][1] == pytest.approx(1.2 * (26.0**0.5) + 5.0 + 2.0)
    assert matrix[0][2] == pytest.approx(1.2 * 60.0 + 4.0 * 60.0)
    expected = 2.0 * matrix[0][1] + 2.0 * matrix[1][2]
    assert module._source_tour_cost([0, 1, 2], points, matrix) == pytest.approx(expected)


def test_source_aco_order_is_public_seeded_permutation() -> None:
    module = _load_adapter()
    cells = [
        SimpleNamespace(cell_id=f"cell-{index}", position=(float(index), float(index % 2), 2.0))
        for index in range(6)
    ]

    first = module._source_aco_order("uav-00", cells)
    second = module._source_aco_order("uav-00", cells)

    assert first == second
    assert first[0] == "cell-0"
    assert set(first) == {cell.cell_id for cell in cells}
    assert len(first) == len(cells)


def test_public_reset_and_jsonl_actions_need_no_target_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_adapter()
    episode, task = _public_reset_inputs()
    reset = {
        "schema": module.REQUEST_SCHEMA,
        "request_id": "reset-public-only",
        "kind": "reset",
        "public_episode": episode,
        "public_task_spec": task,
    }
    act = {
        "schema": module.REQUEST_SCHEMA,
        "request_id": "act-public-only",
        "kind": "act",
        "observations": {
            "uav-00": {
                "timestamp_s": 0.0,
                "observation_id": "public-observation",
                "self_state": {
                    "pose": {"position": [0.0, 0.0, 2.0], "yaw_deg": 0.0},
                    "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                    "angular_speed_deg_s": 0.0,
                },
            }
        },
    }

    module.serve([json.dumps(reset), json.dumps(act)])
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert [response["status"] for response in responses] == ["ok", "ok"]
    assert responses[1]["actions"]["uav-00"]["kind"] == "WAYPOINT"


def test_private_field_is_rejected_before_aco_can_run(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_adapter()
    episode, task = _public_reset_inputs()
    episode["target_count"] = 3
    module.serve(
        [
            json.dumps(
                {
                    "schema": module.REQUEST_SCHEMA,
                    "request_id": "private-input",
                    "kind": "reset",
                    "public_episode": episode,
                    "public_task_spec": task,
                }
            )
        ]
    )

    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "error:ValueError"


def test_upstream_source_verifier_rejects_tampered_locked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_adapter()
    source = tmp_path / "source"
    source.mkdir()
    for name, content in {
        "CreateModel.m": "create\n",
        "TourCost.m": "cost\n",
        "main.m": "main\n",
    }.items():
        (source / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "locked"], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/aco.git"],
        cwd=source,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    filenames = ("CreateModel.m", "TourCost.m", "main.m")
    hashes = {name: module._file_hash(source / name) for name in filenames}
    lock = {
        "upstream": {
            "url": "https://example.invalid/aco.git",
            "commit": revision,
            "license": "MIT",
            "source_file_sha256": hashes,
        },
        "adapter": {
            "fixed_parameters": {
                "iterations": 350,
                "ants": 50,
                "q": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "rho": 0.05,
            }
        },
    }
    monkeypatch.setattr(module, "UPSTREAM_URL", "https://example.invalid/aco.git")
    monkeypatch.setattr(module, "UPSTREAM_COMMIT", revision)
    monkeypatch.setattr(module, "_source_lock", lambda: lock)

    module._verify_upstream_source(source)
    (source / "TourCost.m").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source checkout must be clean|source hash differs"):
        module._verify_upstream_source(source)


def test_smoke_runner_has_an_explicit_translation_boundary() -> None:
    path = Path("tools/run_aco3d_g2i_l0_smoke.py")
    spec = importlib.util.spec_from_file_location("run_aco3d_g2i_l0_smoke_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    assert runner.UPSTREAM_URL == "https://github.com/duynamrcv/aco_3d_ipp.git"
    assert runner.UPSTREAM_COMMIT == "c395f5b61f6746b2d39310dbc55a7ec3e1eae2d5"
    assert runner.UPSTREAM_LICENSE == "MIT"
    assert runner.ADAPTER_VERSION == "aco3d-source-translation-v1"
    assert runner.MAXIMUM_RESET_BYTES == 2_000_000
    assert "coverage and confirmation are never success criteria" in runner.PASS_SEMANTICS
    assert "return_closure_required=true" in runner.PASS_SEMANTICS


def test_smoke_summary_separates_public_coverage_from_private_confirmation() -> None:
    path = Path("tools/run_aco3d_g2i_l0_smoke.py")
    spec = importlib.util.spec_from_file_location("run_aco3d_g2i_l0_smoke_summary", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    summary = runner._summary(
        {
            "task_time_s": 1.0,
            "execution_receipts": [{"action_requested": "OBSERVE"}, {"action_requested": "HOVER"}],
            "confirmations": [],
            "failures": [],
            "budget_ledger": {"collisions": 0, "out_of_bounds_actions": 0, "deadline_misses": 0},
            "returned_home": {"uav-00": True},
            "formal_score_eligible": False,
            "coverage_denominators": {"inspection_atlas_area_m2": 8.0, "inspection_atlas_cells": 4},
            "inspection_coverage_trace": [[1.0, 2.0]],
            "inspection_cell_count_trace": [[1.0, 1]],
        }
    )

    assert summary["confirmation_count"] == 0
    assert summary["observe_request_count"] == 1
    assert summary["inspection_coverage"]["area_fraction"] == 0.25
    assert summary["inspection_coverage"]["cell_fraction"] == 0.25
