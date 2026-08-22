from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aerocity_bench.contracts import Pose3D


def _load_adapter():
    path = Path("tools/ortools_g2i_process_adapter.py")
    spec = importlib.util.spec_from_file_location("ortools_g2i_process_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_external_solver_source_lock_is_complete_and_pinned() -> None:
    source_lock = json.loads(Path("external/ortools/source-lock.json").read_text(encoding="utf-8"))

    assert source_lock["upstream"]["url"] == "https://github.com/google/or-tools.git"
    assert source_lock["upstream"]["commit"] == "98c165af62df62b3056c2ee0fca66b24e79097cb"
    assert source_lock["upstream"]["license"] == "Apache-2.0"
    assert source_lock["python_distribution"]["version"] == "9.15.6755"
    assert source_lock["boundary"]["core_package_dependency"] is False


def test_adapter_is_importable_without_installing_external_solver() -> None:
    module = _load_adapter()

    assert module.REQUEST_SCHEMA.endswith("external-planner-request.v1")
    assert module.UPSTREAM_COMMIT == "98c165af62df62b3056c2ee0fca66b24e79097cb"
    assert module.ORTOOLS_VERSION == "9.15.6755"


def test_adapter_rejects_private_truth_before_solver_import() -> None:
    module = _load_adapter()

    with pytest.raises(ValueError, match="forbidden field"):
        module._reject_non_public({"evaluator_private": {"targets": []}})


def test_adapter_rejects_an_unlocked_upstream_source(tmp_path) -> None:
    module = _load_adapter()
    source = tmp_path / "unlocked-source"
    source.mkdir()

    with pytest.raises(ValueError, match="not a readable Git checkout"):
        module._verify_upstream_source(source)


def test_version_mode_validates_an_explicit_upstream_source(tmp_path) -> None:
    module = _load_adapter()
    source = tmp_path / "unlocked-source"
    source.mkdir()

    completed = subprocess.run(
        [sys.executable, str(Path(module.__file__)), "--upstream-source", str(source), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "not a readable Git checkout" in completed.stderr


def test_adapter_rejects_act_before_a_public_reset(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_adapter()
    module.serve(
        [
            json.dumps(
                {
                    "schema": module.REQUEST_SCHEMA,
                    "request_id": "before-reset",
                    "kind": "act",
                    "observations": {},
                }
            )
        ]
    )

    response = json.loads(capsys.readouterr().out)
    assert response["request_id"] == "before-reset"
    assert response["status"] == "error:ValueError"


def test_runner_constants_match_the_external_source_lock() -> None:
    path = Path("tools/run_ortools_g2i_l0_smoke.py")
    spec = importlib.util.spec_from_file_location("run_ortools_g2i_l0_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.UPSTREAM_URL == "https://github.com/google/or-tools.git"
    assert module.UPSTREAM_COMMIT == "98c165af62df62b3056c2ee0fca66b24e79097cb"
    assert module.UPSTREAM_LICENSE == "Apache-2.0"
    assert module.ORTOOLS_VERSION == "9.15.6755"
    assert module.MAXIMUM_RESET_BYTES == 2_000_000


def test_smoke_summary_separates_public_inspection_from_private_confirmation() -> None:
    path = Path("tools/run_ortools_g2i_l0_smoke.py")
    spec = importlib.util.spec_from_file_location("run_ortools_g2i_l0_smoke_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    summary = module._summary(
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


def _public_reset_inputs() -> tuple[dict[str, object], dict[str, object]]:
    task = {
        "task_track": "G2-I",
        "inspection_atlas": {
            "atlas_hash": "a" * 64,
            "regions": [
                {
                    "cells": [
                        {
                            "cell_id": "public-cell-00",
                            "pose": {
                                "position": [2.0, 0.0, 2.0],
                                "yaw_deg": 0.0,
                                "pitch_deg": 0.0,
                            },
                            "represented_area_m2": 1.0,
                        }
                    ]
                }
            ],
        },
        "execution_contract": {
            "control_period_s": 0.2,
            "episode": {"duration_s": 300.0},
            "observe": {
                "continuous_dwell_s": 0.5,
                "max_linear_speed_mps": 0.25,
                "max_angular_speed_deg_s": 8.0,
            },
            "vehicle": {"horizontal_speed_mps": 3.0, "vertical_speed_mps": 2.0},
        },
        "public_transit_contract": {"safe_sky_altitude_m": 8.0},
        "flight_bounds": {"maximum": [10.0, 10.0, 20.0]},
    }
    episode = {
        "starts": [{"drone_id": "uav-00", "position": [0.0, 0.0, 2.0]}],
        "mission_sector": {
            "truth_independent": True,
            "frozen_before_sampling": True,
            "atlas_hash": "a" * 64,
            "selected_cell_ids": ["public-cell-00"],
            "cell_assignment_by_drone": {"uav-00": ["public-cell-00"]},
            "capacity_certificate": {
                "return_reserve_s": 35.0,
                "horizontal_speed_mps": 1.5,
                "vertical_speed_mps": 1.0,
            },
        },
    }
    return episode, task


def test_adapter_uses_frozen_sector_transit_rates_not_vehicle_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter()
    monkeypatch.setattr(module.ORToolsInspectionPlanner, "_solve_sector_route", lambda *_: [])
    episode, task = _public_reset_inputs()

    planner = module.ORToolsInspectionPlanner.from_public_reset(episode, task)

    assert planner.horizontal_speed_mps == 1.5
    assert planner.vertical_speed_mps == 1.0
    assert planner.return_reserve_s == 35.0


def test_adapter_rejects_capacity_rate_above_public_vehicle_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_adapter()
    monkeypatch.setattr(module.ORToolsInspectionPlanner, "_solve_sector_route", lambda *_: [])
    episode, task = _public_reset_inputs()
    certificate = episode["mission_sector"]["capacity_certificate"]  # type: ignore[index]
    certificate["horizontal_speed_mps"] = 3.1  # type: ignore[index]

    with pytest.raises(ValueError, match="capacity-certificate speed exceeds"):
        module.ORToolsInspectionPlanner.from_public_reset(episode, task)


def test_adapter_routes_public_cell_pitch_to_bounded_sensor_gimbal() -> None:
    module = _load_adapter()
    cell = module.PublicCell(
        cell_id="public-roof-cell",
        position=(1.0, 2.0, 3.0),
        yaw_deg=27.0,
        pitch_deg=-90.0,
        represented_area_m2=1.0,
    )
    planner = module.ORToolsInspectionPlanner(
        duration_s=300.0,
        control_period_s=0.2,
        dwell_s=0.5,
        safe_sky_altitude_m=8.0,
        return_reserve_s=20.0,
        horizontal_speed_mps=1.5,
        vertical_speed_mps=1.0,
        starts={"uav-00": (0.0, 0.0, 1.0)},
        cells={cell.cell_id: cell},
        assignments={"uav-00": (cell.cell_id,)},
        routes={},
        completed={"uav-00": set()},
    )
    route = module.RouteState(
        ordered_cell_ids=[], cell_id=cell.cell_id, phase="descend"
    )

    action = planner._route_action(
        "uav-00",
        {
            "observation_id": "public-observation",
            "self_state": {"pose": {"position": [1.0, 2.0, 4.0], "yaw_deg": 0.0}},
        },
        route,
    )

    assert action["kind"] == "WAYPOINT"
    waypoint = Pose3D.from_dict(action["waypoint"])
    assert waypoint.position == cell.position
    assert waypoint.yaw_deg == cell.yaw_deg
    assert waypoint.pitch_deg == 0.0
    assert action["sensor_pitch_deg"] == cell.pitch_deg


def test_adapter_schedules_enough_samples_for_continuous_dwell() -> None:
    module = _load_adapter()
    cell = module.PublicCell(
        cell_id="public-wall-cell",
        position=(1.0, 2.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        represented_area_m2=1.0,
    )
    planner = module.ORToolsInspectionPlanner(
        duration_s=300.0,
        control_period_s=0.2,
        dwell_s=0.5,
        safe_sky_altitude_m=8.0,
        return_reserve_s=20.0,
        horizontal_speed_mps=1.5,
        vertical_speed_mps=1.0,
        starts={"uav-00": (0.0, 0.0, 1.0)},
        cells={cell.cell_id: cell},
        assignments={"uav-00": (cell.cell_id,)},
        routes={},
        completed={"uav-00": set()},
    )
    route = module.RouteState(
        ordered_cell_ids=[], cell_id=cell.cell_id, phase="descend"
    )
    observation = {
        "observation_id": "public-observation",
        "self_state": {
            "pose": {"position": list(cell.position), "yaw_deg": cell.yaw_deg},
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_speed_deg_s": 0.0,
        },
    }
    actions = [
        planner._route_action("uav-00", observation, route)
        for _ in range(6)
    ]

    # One hover absorbs arrival velocity and one follows the first public
    # settled packet.  The first OBSERVE sample then starts the dwell window.
    assert [action["kind"] for action in actions] == ["HOVER", "HOVER", *(["OBSERVE"] * 4)]
    assert (len(actions[2:]) - 1) * planner.control_period_s >= planner.dwell_s


def test_adapter_waits_for_public_velocity_to_settle_before_observe() -> None:
    module = _load_adapter()
    cell = module.PublicCell(
        cell_id="public-wall-cell",
        position=(1.0, 2.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        represented_area_m2=1.0,
    )
    planner = module.ORToolsInspectionPlanner(
        duration_s=300.0,
        control_period_s=0.2,
        dwell_s=0.5,
        safe_sky_altitude_m=8.0,
        return_reserve_s=20.0,
        horizontal_speed_mps=1.5,
        vertical_speed_mps=1.0,
        starts={"uav-00": (0.0, 0.0, 1.0)},
        cells={cell.cell_id: cell},
        assignments={"uav-00": (cell.cell_id,)},
        routes={},
        completed={"uav-00": set()},
    )
    route = module.RouteState(ordered_cell_ids=[], cell_id=cell.cell_id, phase="settle")
    moving = {
        "observation_id": "public-observation-moving",
        "self_state": {
            "pose": {"position": list(cell.position), "yaw_deg": cell.yaw_deg},
            "linear_velocity_world_mps": [0.26, 0.0, 0.0],
            "angular_speed_deg_s": 0.0,
        },
    }
    settled = {
        "observation_id": "public-observation-settled",
        "self_state": {
            "pose": {"position": list(cell.position), "yaw_deg": cell.yaw_deg},
            "linear_velocity_world_mps": [0.24, 0.0, 0.0],
            "angular_speed_deg_s": 7.9,
        },
    }

    assert planner._route_action("uav-00", moving, route)["kind"] == "HOVER"
    assert route.phase == "settle"
    assert planner._route_action("uav-00", settled, route)["kind"] == "HOVER"
    assert route.phase == "observe"


def test_adapter_keeps_same_region_inspection_motion_below_safe_sky() -> None:
    module = _load_adapter()
    first = module.PublicCell(
        cell_id="same-region-00",
        position=(1.0, 2.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=-30.0,
        represented_area_m2=1.0,
        region_id="region-a",
    )
    second = module.PublicCell(
        cell_id="same-region-01",
        position=(2.0, 2.0, 3.0),
        yaw_deg=15.0,
        pitch_deg=-30.0,
        represented_area_m2=1.0,
        region_id="region-a",
    )
    planner = module.ORToolsInspectionPlanner(
        duration_s=300.0,
        control_period_s=0.2,
        dwell_s=0.5,
        safe_sky_altitude_m=40.0,
        return_reserve_s=35.0,
        horizontal_speed_mps=1.5,
        vertical_speed_mps=1.0,
        starts={"uav-00": (0.0, 0.0, 2.0)},
        cells={first.cell_id: first, second.cell_id: second},
        assignments={"uav-00": (first.cell_id, second.cell_id)},
        routes={
            "uav-00": module.RouteState(
                ordered_cell_ids=[second.cell_id], previous_region_id=first.region_id
            )
        },
        completed={"uav-00": {first.cell_id}},
    )

    action = planner.action(
        "uav-00",
        {
            "timestamp_s": 10.0,
            "observation_id": "public-observation",
            "self_state": {"pose": {"position": list(first.position), "yaw_deg": first.yaw_deg}},
        },
    )

    assert action["kind"] == "WAYPOINT"
    assert Pose3D.from_dict(action["waypoint"]).position == second.position
    assert action["sensor_pitch_deg"] == second.pitch_deg
    assert planner.routes["uav-00"].phase == "direct"
    assert planner._direct_scan_time_s(first.position, second.position) < planner._transit_time_s(
        first.position, second.position
    )


def test_adapter_uses_safe_sky_when_same_region_segment_crosses_public_building() -> None:
    module = _load_adapter()
    first = module.PublicCell(
        cell_id="same-region-00",
        position=(-2.0, 0.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=-30.0,
        represented_area_m2=1.0,
        region_id="region-a",
    )
    second = module.PublicCell(
        cell_id="same-region-01",
        position=(2.0, 0.0, 3.0),
        yaw_deg=15.0,
        pitch_deg=-30.0,
        represented_area_m2=1.0,
        region_id="region-a",
    )
    planner = module.ORToolsInspectionPlanner(
        duration_s=300.0,
        control_period_s=0.2,
        dwell_s=0.5,
        safe_sky_altitude_m=40.0,
        return_reserve_s=35.0,
        horizontal_speed_mps=1.5,
        vertical_speed_mps=1.0,
        starts={"uav-00": (0.0, 0.0, 2.0)},
        cells={first.cell_id: first, second.cell_id: second},
        assignments={"uav-00": (first.cell_id, second.cell_id)},
        coarse_colliders=(
            module.PublicAABB(minimum=(-0.5, -1.0, 0.0), maximum=(0.5, 1.0, 20.0)),
        ),
        direct_scan_clearance_m=0.25,
        routes={
            "uav-00": module.RouteState(
                ordered_cell_ids=[second.cell_id], previous_region_id=first.region_id
            )
        },
        completed={"uav-00": {first.cell_id}},
    )

    action = planner.action(
        "uav-00",
        {
            "timestamp_s": 10.0,
            "observation_id": "public-observation",
            "self_state": {"pose": {"position": list(first.position), "yaw_deg": first.yaw_deg}},
        },
    )

    assert action["kind"] == "WAYPOINT"
    assert Pose3D.from_dict(action["waypoint"]).position == (
        first.position[0],
        first.position[1],
        planner.safe_sky_altitude_m,
    )
    assert planner.routes["uav-00"].phase == "ascend"
    assert not planner._direct_scan_is_publicly_clear(first.position, second.position)
