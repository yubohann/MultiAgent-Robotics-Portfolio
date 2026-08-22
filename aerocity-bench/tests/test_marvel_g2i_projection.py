from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from aerocity_bench.adapters import _external_process_episode_projection
from aerocity_bench.marvel_g2i_projection import MarvelG2IProjection


def _public_reset() -> tuple[dict, dict]:
    cells = [
        {
            "cell_id": f"cell-{index}",
            "pose": {"position": [float(index + 2), 0.0, 2.5], "yaw_deg": 0.0},
            "represented_area_m2": 1.0 + index,
        }
        for index in range(3)
    ]
    episode = {
        "starts": [{"drone_id": "uav-00", "position": [0.0, 0.0, 2.5]}],
        "mission_sector": {
            "truth_independent": True,
            "frozen_before_sampling": True,
            "atlas_hash": "public-atlas",
            "selected_cell_ids": [cell["cell_id"] for cell in cells],
            "cell_assignment_by_drone": {"uav-00": [cell["cell_id"] for cell in cells]},
            "capacity_certificate": {"return_reserve_s": 2.0},
        },
    }
    task = {
        "task_track": "G2-I",
        "flight_bounds": {"maximum": [20.0, 20.0, 12.0]},
        "public_transit_contract": {"safe_sky_altitude_m": 8.0},
        "execution_contract": {
            "control_period_s": 0.2,
            "episode": {"duration_s": 10.0},
            "observe": {"continuous_dwell_s": 0.5},
            "vehicle": {"horizontal_speed_mps": 3.0, "vertical_speed_mps": 2.0},
        },
        "inspection_atlas": {"atlas_hash": "public-atlas", "regions": [{"cells": cells}]},
    }
    return episode, task


def _observation(*, position: list[float], timestamp_s: float = 0.0) -> dict:
    return {
        "observation_id": "obs-0",
        "timestamp_s": timestamp_s,
        "self_state": {"pose": {"position": position, "yaw_deg": 0.0}},
    }


def _choose_first(graph: object) -> int:
    del graph
    return 0


def test_projection_has_marvel_compatible_public_graph_shapes() -> None:
    episode, task = _public_reset()
    task["target_count_public"] = False
    task["target_process_public"] = False
    task["formal_split_label_public"] = False
    projection = MarvelG2IProjection.from_public_reset(episode, task)

    graph = projection.graph_input("uav-00", _observation(position=[0.0, 0.0, 2.5]))

    assert len(graph.node_inputs) == 360
    assert len(graph.node_inputs[0]) == 6
    assert len(graph.edge_mask) == len(graph.edge_mask[0]) == 360
    assert len(graph.current_edge) == len(graph.edge_padding_mask) == 25
    assert len(graph.neighbor_best_headings) == 25
    assert len(graph.neighbor_best_headings[0]) == 3
    assert len(graph.neighbor_best_headings[0][0]) == 36
    assert graph.candidate_cell_ids == ["cell-0", "cell-1", "cell-2"]
    assert graph.edge_padding_mask[:4] == [1, 0, 0, 0]


def test_projection_rejects_private_field_before_policy_can_receive_it() -> None:
    episode, task = _public_reset()
    task["private_evaluator"] = {"secret": "no"}

    with pytest.raises(ValueError, match="non-public"):
        MarvelG2IProjection.from_public_reset(episode, task)


def test_process_wire_drops_all_false_information_boundary_sentinels() -> None:
    wire = _external_process_episode_projection(
        {
            "target_count_public": False,
            "target_process_public": False,
            "formal_split_label_public": False,
            "public_value": 1,
        }
    )

    assert wire == {"public_value": 1}


def test_projection_routes_public_cell_observes_then_returns() -> None:
    episode, task = _public_reset()
    projection = MarvelG2IProjection.from_public_reset(episode, task)
    first = projection.action(
        "uav-00", _observation(position=[0.0, 0.0, 2.5]), _choose_first
    )
    assert first["kind"] == "WAYPOINT"
    assert first["waypoint"]["position"] == [0.0, 0.0, 8.0]
    projection.action("uav-00", _observation(position=[0.0, 0.0, 8.0]), _choose_first)
    projection.action("uav-00", _observation(position=[2.0, 0.0, 8.0]), _choose_first)
    observe = projection.action("uav-00", _observation(position=[2.0, 0.0, 2.5]), _choose_first)
    assert observe == {"kind": "OBSERVE", "source_observation_id": "obs-0"}
    for _ in range(2):
        projection.action("uav-00", _observation(position=[2.0, 0.0, 2.5]), _choose_first)
    assert "cell-0" in projection.completed["uav-00"]
    returning = projection.action(
        "uav-00", _observation(position=[2.0, 0.0, 2.5], timestamp_s=8.0), _choose_first
    )
    assert returning["kind"] == "WAYPOINT"
    assert returning["waypoint"]["position"] == [2.0, 0.0, 8.0]


def test_projection_uses_public_safe_sky_return_bound_not_only_fixed_time() -> None:
    episode, task = _public_reset()
    projection = MarvelG2IProjection.from_public_reset(episode, task)

    action = projection.action(
        "uav-00",
        _observation(position=[12.0, 0.0, 2.5], timestamp_s=1.0),
        _choose_first,
    )

    assert action["kind"] == "WAYPOINT"
    assert action["waypoint"]["position"] == [12.0, 0.0, 8.0]
    assert projection.routes["uav-00"].phase == "return-ascend"


def test_process_adapter_keeps_upstream_model_import_lazy() -> None:
    path = Path("tools/marvel_g2i_process_adapter.py")
    spec = importlib.util.spec_from_file_location("marvel_g2i_process_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REQUEST_SCHEMA.endswith("external-planner-request.v1")


def test_process_adapter_warms_a_fixed_task_free_graph_before_reset() -> None:
    path = Path("tools/marvel_g2i_process_adapter.py")
    spec = importlib.util.spec_from_file_location("marvel_g2i_process_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured = []

    class Recorder:
        def choose_slot(self, graph):
            captured.append(graph)
            return 0

    module.FrozenMarvelPolicy._warm_up(Recorder())

    assert len(captured) == 1
    graph = captured[0]
    assert graph.candidate_cell_ids == ["warmup-cell"]
    assert len(graph.node_inputs) == 360
    assert len(graph.node_inputs[0]) == 6
    assert len(graph.edge_mask) == 360
    assert len(graph.edge_mask[0]) == 360
    assert graph.current_edge == [0] * 25
    assert len(graph.edge_padding_mask) == 25
    assert len(graph.neighbor_best_headings) == 25
    assert len(graph.neighbor_best_headings[0]) == 3
    assert len(graph.neighbor_best_headings[0][0]) == 36
