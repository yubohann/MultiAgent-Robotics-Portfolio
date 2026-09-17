from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from aerocity_bench.contracts import Pose3D


def _load_adapter():
    path = Path("tools/ortools_g2i_process_adapter_v10_grouped_safe_sky.py")
    spec = importlib.util.spec_from_file_location("ortools_g2i_adapter_v10", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ancestor_00_public_inputs() -> tuple[dict[str, object], dict[str, object]]:
    import tempfile

    from aerocity_bench.canonical import write_json
    from aerocity_bench.compiler import compile_g2_i_task_spec
    from aerocity_bench.generator_v3 import generate_city_v3
    from aerocity_bench.ordinary_config import load_ordinary_config
    from aerocity_bench.targets_v3 import (
        derive_support_sites_v3,
        public_episode_projection,
        sample_episode_v3,
    )

    config_root = Path(__file__).parents[1] / "configs" / "releases" / "ordinary-v1-mini.json"
    raw = json.loads(config_root.read_text(encoding="utf-8"))
    raw["admission"]["maximum_single_observation_target_fraction"] = 1.0
    workspace = Path(tempfile.mkdtemp(prefix="ortools-v10-public-"))
    config_path = workspace / "ordinary.json"
    write_json(config_path, raw)
    config = load_ordinary_config(config_path)
    from aerocity_bench.errors import GenerationRejected

    for attempt in range(32):
        try:
            city = generate_city_v3(
                config, "calibration", 0, attempt, list(config.raw["assets"]["allowlist"])
            )
            task = compile_g2_i_task_spec(
                city, config.raw["execution_contract"], config.raw["fleet"]
            )
            sites = derive_support_sites_v3(city, config)
            episode = sample_episode_v3(config, city, sites, 0, public_task_spec=task)
            return public_episode_projection(episode), task
        except GenerationRejected:
            continue
    raise AssertionError("no admissible calibration episode for the OR-Tools probe")


def test_v10_locks_legacy_source_and_exposes_a_distinct_grouped_route_model() -> None:
    module = _load_adapter()

    assert module.ADAPTER_ID == "ortools-public-atlas-routing-v10-grouped-safe-sky"
    assert module.GROUPED_ROUTE_MODEL == "public-fixed-assignment-grouped-safe-sky-route-v1"
    assert module._legacy_path().is_file()


def test_v10_public_probe_keeps_every_assigned_cell_for_the_two_previously_empty_lanes(
    monkeypatch,
) -> None:
    module = _load_adapter()
    episode, task = _ancestor_00_public_inputs()

    # Use the frozen canonical order so the route topology is testable without
    # the separate OR-Tools wheel used by the production process boundary.
    monkeypatch.setattr(
        module.GroupedSafeSkyORToolsPlanner,
        "_solve_local_group",
        lambda self, *, origin, group: list(group),
    )
    planner = module.GroupedSafeSkyORToolsPlanner.from_public_reset(episode, task)
    assignments = episode["mission_sector"]["cell_assignment_by_drone"]

    for drone_id in ("uav-02", "uav-03"):
        ordered = planner.routes[drone_id].ordered_cell_ids
        assert ordered
        assert set(ordered) == set(assignments[drone_id])
        assert len(ordered) == len(assignments[drone_id])


def test_v10_uses_direct_waypoint_only_for_a_public_facade_group() -> None:
    module = _load_adapter()
    first = module.PublicCell(
        cell_id="facade-00",
        position=(1.0, 2.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=-20.0,
        represented_area_m2=1.0,
        region_id="facade-region",
    )
    second = module.PublicCell(
        cell_id="facade-01",
        position=(2.0, 2.0, 3.0),
        yaw_deg=10.0,
        pitch_deg=-20.0,
        represented_area_m2=1.0,
        region_id="facade-region",
    )
    planner = module.GroupedSafeSkyORToolsPlanner(
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
        routes={"uav-00": module.RouteState(ordered_cell_ids=[second.cell_id])},
        completed={"uav-00": {first.cell_id}},
    )
    planner.last_completed_cell_by_drone = {"uav-00": first.cell_id}
    planner.direct_successors_by_drone = {"uav-00": {first.cell_id: second.cell_id}}

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
    assert planner.routes["uav-00"].phase == "local-transit"


def test_v10_top_down_cell_remains_an_individual_safe_sky_group() -> None:
    module = _load_adapter()
    facade = module.PublicCell(
        cell_id="facade-00",
        position=(1.0, 2.0, 3.0),
        yaw_deg=0.0,
        pitch_deg=-20.0,
        represented_area_m2=1.0,
        region_id="shared-region",
    )
    roof = module.PublicCell(
        cell_id="roof-00",
        position=(2.0, 2.0, 5.0),
        yaw_deg=0.0,
        pitch_deg=-90.0,
        represented_area_m2=1.0,
        region_id="shared-region",
    )
    planner = module.GroupedSafeSkyORToolsPlanner(
        duration_s=300.0,
        control_period_s=0.2,
        dwell_s=0.5,
        safe_sky_altitude_m=40.0,
        return_reserve_s=35.0,
        horizontal_speed_mps=1.5,
        vertical_speed_mps=1.0,
        starts={"uav-00": (0.0, 0.0, 2.0)},
        cells={facade.cell_id: facade, roof.cell_id: roof},
        assignments={"uav-00": (facade.cell_id, roof.cell_id)},
        routes={},
        completed={"uav-00": set()},
    )

    assert planner._grouped_assignment("uav-00") == [[facade.cell_id], [roof.cell_id]]
