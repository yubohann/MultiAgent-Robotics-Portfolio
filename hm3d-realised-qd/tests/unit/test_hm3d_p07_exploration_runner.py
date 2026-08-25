"""Static contract tests for the isolated target-free P07 Isaac worker."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aerocity_method.evaluation.hm3d_exploration_metrics import ExplorationMetricSample
from aerocity_method.runtime.hm3d_belief import FREE, PublicRangeRayOutcome, SparseVoxelBelief
from aerocity_method.runtime.hm3d_cf2x_execution import (
    BITCRAZE_MELLINGER_CONTROLLER_ID,
    CF2X_DEFAULT_CONTROLLER_ID,
)

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_hm3d_p07_exploration_episode.py"


def _load_runner_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("hm3d_p07_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def _p0_departure_reset_source(runner):
    candidates = []
    for index in range(4):
        start = (float(index), 0.0, 1.0)
        candidates.append(
            {
                "candidate_id": f"start-{index}",
                "position_w_m": list(start),
                "static_departure_witnesses": [
                    {
                        "schema_version": runner.P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
                        "witness_id": "six-neighbour-00",
                        "path_m": [list(start), [start[0] + 0.25, start[1], start[2]]],
                        "offline_exact_admitted": True,
                    }
                ],
            }
        )
    return {
        "departure_witness_contract": {
            "schema_version": runner.P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
            "selection_rule": "six-neighbour-grid-tube+exact-static-samples-v1",
            "required_internal_sample_clearance_m": runner.cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
        },
        "candidates": candidates,
    }


def test_p0_eligibility_contract_normalizes_receipt_tolerance() -> None:
    runner = _load_runner_module()

    normalized = runner._normalized_p0_eligibility_contract(
        {
            "schema_version": "hm3d-p07-start-eligibility-contract-v1",
            "strategy": "frontier_3d",
            "candidate_limit": 8,
            "action_budget_s": 40.0,
            "physics_dt_s": 1.0 / 120.0,
            "arrival_tolerance_m": 0.1,
            "receipt_time_tolerance_s": 0.25,
            "random_key": 20260806,
        }
    )

    assert normalized == {
        "schema_version": "hm3d-p07-start-eligibility-contract-v1",
        "candidate_limit": 8,
        "action_budget_s": 40.0,
        "physics_dt_s": 1.0 / 120.0,
        "arrival_tolerance_m": 0.1,
        "outcome_time_tolerance_s": 0.25,
    }


def test_online_p07_loads_only_passed_transit_timing_calibration(tmp_path: Path) -> None:
    runner = _load_runner_module()
    path = tmp_path / "timing.json"
    path.write_text(
        '{"schema_version":"hm3d-cf2x-transit-timing-calibration-v4",'
        '"status":"CALIBRATION_PASS","time_model":{'
        '"schema_version":"hm3d-kinematic-transit-timing-v4",'
        '"calibration_id":"unit","cruise_speed_mps":0.5,'
        '"max_accel_mps2":0.4,"terminal_tracking_margin_s":0.2,'
        '"intermediate_waypoint_settle_margin_s":0.1,'
        '"calibrated_max_segment_count":2,"uncovered_segment_reserve_s":1.0,'
        '"intermediate_waypoint_requires_settle":true,'
        '"continuous_waypoint_speed_mps":0.35},'
        '"calibrated_max_segment_count":2,"uncovered_segment_reserve_s":1.0,'
        '"intermediate_waypoint_requires_settle":true,'
        '"continuous_waypoint_speed_mps":0.35,'
        '"observation_dwell_s":1.0}',
        encoding="utf-8",
    )

    model = runner._load_transit_timing_model(path)

    assert model.calibration_id == "unit"
    assert model.cruise_speed_mps == 0.5
    assert model.max_accel_mps2 == 0.4


def test_online_p07_rejects_v4_timing_artifact_without_long_route_reserve(tmp_path: Path) -> None:
    runner = _load_runner_module()
    path = tmp_path / "stale-timing.json"
    path.write_text(
        '{"schema_version":"hm3d-cf2x-transit-timing-calibration-v4",'
        '"status":"CALIBRATION_PASS","time_model":{'
        '"schema_version":"hm3d-kinematic-transit-timing-v4",'
        '"calibration_id":"stale","cruise_speed_mps":0.5,'
        '"max_accel_mps2":0.4,"terminal_tracking_margin_s":0.2,'
        '"intermediate_waypoint_settle_margin_s":0.1},'
        '"observation_dwell_s":1.0}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required v4 fields"):
        runner._load_transit_timing_contract(path)


def test_online_p07_uses_a_physical_budget_not_fixed_decision_windows() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '"--max-decision-count"' in source
    assert "while elapsed_s < args.action_budget_s - 1.0e-9:" in source
    assert "round_duration_s" not in source
    assert "args.decision_count" not in source
    assert '"--maximum-frontier-step-m"' not in source
    assert "maximum_frontier_step_m" not in source
    assert "effective_frontier_step_m = reachable_path_length_m" in source
    assert '"--lattice-step-m"' not in source
    assert "def parse_args(argv: Sequence[str] | None = None)" in source
    assert "if argv is None:" in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "@functools.lru_cache(maxsize=8)" in source


def test_online_p07_frontier_horizon_is_derived_and_rejects_retired_manual_cap() -> None:
    runner = _load_runner_module()
    required = (
        "--scene-id",
        "scene0",
        "--split",
        "train",
        "--collision-usd",
        "collision.usd",
        "--start-reset-json",
        "starts.json",
        "--flight-space-audit",
        "flight-space.json",
        "--p03-artifact",
        "p03.json",
        "--p04-artifact",
        "p04.json",
        "--p05-artifact",
        "p05.json",
        "--p06-artifact",
        "p06.json",
        "--transit-time-model-json",
        "timing.json",
        "--output",
        "output.json",
        "--strategy",
        "frontier_3d",
    )

    parsed = runner.parse_args(required)
    assert not hasattr(parsed, "maximum_frontier_step_m")
    with pytest.raises(SystemExit):
        runner.parse_args((*required, "--maximum-frontier-step-m", "3"))


def test_explicit_p0_start_selection_keeps_the_requested_order_and_relay_contract() -> None:
    runner = _load_runner_module()
    candidates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    selected = runner._select_explicit_initial_positions(
        candidates,
        ("start-0", "start-1", "start-2", "start-3"),
        ("start-3", "start-1", "start-2", "start-0"),
        lambda _positions: type("Graph", (), {"fully_relay_connected": True})(),
    )

    assert selected == (candidates[3], candidates[1], candidates[2], candidates[0])
    with pytest.raises(ValueError, match="unknown candidate IDs"):
        runner._select_explicit_initial_positions(
            candidates,
            ("start-0", "start-1", "start-2", "start-3"),
            ("start-3", "start-1", "start-2", "missing"),
            lambda _positions: type("Graph", (), {"fully_relay_connected": True})(),
        )
    with pytest.raises(ValueError, match="not relay connected"):
        runner._select_explicit_initial_positions(
            candidates,
            ("start-0", "start-1", "start-2", "start-3"),
            ("start-3", "start-1", "start-2", "start-0"),
            lambda _positions: type("Graph", (), {"fully_relay_connected": False})(),
        )


def test_relay_connected_start_combination_audit_only_returns_admitted_quartets() -> None:
    runner = _load_runner_module()
    candidates = tuple((float(index), 0.0, 0.0) for index in range(5))
    candidate_ids = tuple(f"start-{index}" for index in range(5))

    admitted = runner._relay_connected_start_id_combinations(
        candidates,
        candidate_ids,
        lambda positions: type(
            "Graph",
            (),
            {"fully_relay_connected": positions[0] == candidates[0]},
        )(),
    )

    assert admitted == (
        ("start-0", "start-1", "start-2", "start-3"),
        ("start-0", "start-1", "start-2", "start-4"),
        ("start-0", "start-1", "start-3", "start-4"),
        ("start-0", "start-2", "start-3", "start-4"),
    )


def test_p0_full_qualification_requires_matching_all_active_audit_evidence() -> None:
    runner = _load_runner_module()
    eligibility_contract = {
        "schema_version": "hm3d-p07-start-eligibility-contract-v1",
        "strategy": "frontier_3d",
        "candidate_limit": 8,
        "action_budget_s": 40.0,
        "physics_dt_s": 0.01,
        "arrival_tolerance_m": 0.03,
        "outcome_time_tolerance_s": 0.05,
        "random_key": 20260806,
    }
    payload = {
        "schema_version": "hm3d-p07-start-eligibility-audit-v1",
        "status": "P07_START_ELIGIBILITY_AUDIT_COMPLETE",
        "scene_id": "scene",
        "controller_id": "isaac-so3-feedback-v6",
        "transit_time_model_sha256": "t" * 64,
        "p0_eligibility_contract": eligibility_contract,
        "start_reset_manifest_sha256": "s" * 64,
        "initial_start_reset": {
            "selected_start_candidate_ids": ["start-0", "start-1", "start-2", "start-3"],
            "p0_live_departure_qualification": {
                "schema_version": "hm3d-p07-live-start-departure-qualification-v1",
                "selected_candidate_ids": ["start-0", "start-1", "start-2", "start-3"],
                "passed": True,
            },
        },
        "first_pool": {"all_agents_active_candidate_exists": True},
    }
    payload["audit_record_sha256"] = runner.canonical_sha256(payload)

    selected = runner._validated_p0_start_eligibility_evidence(
        payload,
        scene_id="scene",
        start_reset_manifest_sha256="s" * 64,
        controller_id="isaac-so3-feedback-v6",
        transit_time_model_sha256="t" * 64,
        p0_eligibility_contract=eligibility_contract,
        requested_candidate_ids=("start-0", "start-1", "start-2", "start-3"),
    )

    assert selected == ("start-0", "start-1", "start-2", "start-3")
    with pytest.raises(ValueError, match="all-active"):
        failing = dict(payload)
        failing["first_pool"] = {"all_agents_active_candidate_exists": False}
        failing.pop("audit_record_sha256")
        failing["audit_record_sha256"] = runner.canonical_sha256(failing)
        runner._validated_p0_start_eligibility_evidence(
            failing,
            scene_id="scene",
            start_reset_manifest_sha256="s" * 64,
            controller_id="isaac-so3-feedback-v6",
            transit_time_model_sha256="t" * 64,
            p0_eligibility_contract=eligibility_contract,
            requested_candidate_ids=("start-0", "start-1", "start-2", "start-3"),
        )


    failed_departure = dict(payload)
    failed_departure["initial_start_reset"] = {
        **payload["initial_start_reset"],
        "p0_live_departure_qualification": {
            "schema_version": "hm3d-p07-live-start-departure-qualification-v1",
            "selected_candidate_ids": ["start-0", "start-1", "start-2", "start-3"],
            "passed": False,
        },
    }
    failed_departure.pop("audit_record_sha256")
    failed_departure["audit_record_sha256"] = runner.canonical_sha256(failed_departure)
    with pytest.raises(ValueError, match="live departure"):
        runner._validated_p0_start_eligibility_evidence(
            failed_departure,
            scene_id="scene",
            start_reset_manifest_sha256="s" * 64,
            controller_id="isaac-so3-feedback-v6",
            transit_time_model_sha256="t" * 64,
            p0_eligibility_contract=eligibility_contract,
            requested_candidate_ids=("start-0", "start-1", "start-2", "start-3"),
        )

    mismatched_contract = {**eligibility_contract, "candidate_limit": 9}
    with pytest.raises(ValueError, match="execution contract"):
        runner._validated_p0_start_eligibility_evidence(
            payload,
            scene_id="scene",
            start_reset_manifest_sha256="s" * 64,
            controller_id="isaac-so3-feedback-v6",
            transit_time_model_sha256="t" * 64,
            p0_eligibility_contract=mismatched_contract,
            requested_candidate_ids=("start-0", "start-1", "start-2", "start-3"),
        )


def test_p0_eligibility_rejects_infeasible_all_active_candidate_rows() -> None:
    runner = _load_runner_module()
    rows = [
        {"feasible": False, "moving_explorer_count": 4},
        {"feasible": True, "moving_explorer_count": 4},
        {"feasible": True, "moving_explorer_count": 3},
    ]

    assert runner._feasible_all_active_candidate_count(rows, fleet_size=4) == 1
    assert (
        runner._feasible_all_active_candidate_count(
            [rows[0]],
            fleet_size=4,
        )
        == 0
    )


def test_p0_departure_envelope_rejects_terminal_only_reset_generation() -> None:
    runner = _load_runner_module()
    legacy = {
        "selection_rule": "largest-component-terminal-clearance-local-farthest-spread-v2",
        "start_mobility_clearance_m": runner.cf2x.REQUIRED_TERMINAL_CLEARANCE_M,
    }

    with pytest.raises(ValueError, match="terminal clearance"):
        runner._p0_departure_envelope_audit(legacy)

    source = {
        "selection_rule": runner.P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE,
        "start_mobility_clearance_m": runner.cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
        "candidate_count": 1,
        "departure_witness_contract": {
            "schema_version": runner.P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
            "selection_rule": "six-neighbour-grid-tube+exact-static-samples-v1",
            "required_internal_sample_clearance_m": runner.cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
        },
        "candidates": [
            {
                "static_departure_witnesses": [
                    {
                        "schema_version": runner.P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
                        "offline_exact_admitted": True,
                        "path_m": [[0.0, 0.0, 1.0], [0.25, 0.0, 1.0]],
                    }
                ]
            }
        ],
    }
    admitted = runner._p0_departure_envelope_audit(source)

    assert admitted["admitted"] is True
    assert admitted["required_route_sample_clearance_m"] == pytest.approx(
        runner.cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M
    )
    assert admitted["departure_witness_count"] == 1

    source["candidates"][0]["static_departure_witnesses"] = []
    with pytest.raises(ValueError, match="lacks a static departure witness"):
        runner._p0_departure_envelope_audit(source)


def test_p0_live_departure_reuses_shared_guard_and_keeps_probe_geometry_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    source = _p0_departure_reset_source(runner)
    calls = []

    def fake_guard(
        scene_query,
        clearance_oracle,
        public_waypoints,
        agent_id,
        path_m,
        bounds_min,
        bounds_max,
        diagnostic_sink,
        *,
        allow_public_reroute,
    ):
        calls.append(
            {
                "public_waypoints": public_waypoints,
                "agent_id": agent_id,
                "path_m": path_m,
                "allow_public_reroute": allow_public_reroute,
            }
        )
        return SimpleNamespace(legal=True, path_m=path_m, reason="")

    monkeypatch.setattr(runner.cf2x, "_routed_guard", fake_guard)
    starts = tuple((float(index), 0.0, 1.0) for index in range(4))
    qualification = runner._p0_live_departure_qualification(
        source,
        selected_candidate_ids=tuple(f"start-{index}" for index in range(4)),
        selected_positions=starts,
        observed_positions=starts,
        scene_query=object(),
        clearance_oracle=object(),
        bounds_min=(-10.0, -10.0, -10.0),
        bounds_max=(10.0, 10.0, 10.0),
        collision_usd_sha256="c" * 64,
        start_reset_manifest_sha256="s" * 64,
    )

    assert qualification["passed"] is True
    assert len(calls) == 4
    assert all(call["public_waypoints"] == () for call in calls)
    assert all(call["allow_public_reroute"] is False for call in calls)
    assert all(row["passed"] is True for row in qualification["candidates"])
    encoded = json.dumps(qualification, sort_keys=True)
    assert "path_m" not in encoded
    assert "position_w_m" not in encoded


def test_p0_live_departure_requires_one_guard_legal_nonzero_hop_per_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    source = _p0_departure_reset_source(runner)

    def fake_guard(
        scene_query,
        clearance_oracle,
        public_waypoints,
        agent_id,
        path_m,
        bounds_min,
        bounds_max,
        diagnostic_sink,
        *,
        allow_public_reroute,
    ):
        legal = not agent_id.endswith("start-2")
        return SimpleNamespace(
            legal=legal,
            path_m=path_m,
            reason="" if legal else "segment_blocked",
        )

    monkeypatch.setattr(runner.cf2x, "_routed_guard", fake_guard)
    starts = tuple((float(index), 0.0, 1.0) for index in range(4))
    qualification = runner._p0_live_departure_qualification(
        source,
        selected_candidate_ids=tuple(f"start-{index}" for index in range(4)),
        selected_positions=starts,
        observed_positions=starts,
        scene_query=object(),
        clearance_oracle=object(),
        bounds_min=(-10.0, -10.0, -10.0),
        bounds_max=(10.0, 10.0, 10.0),
        collision_usd_sha256="c" * 64,
        start_reset_manifest_sha256="s" * 64,
    )

    assert qualification["passed"] is False
    by_id = {row["candidate_id"]: row for row in qualification["candidates"]}
    assert by_id["start-2"]["passed"] is False
    assert by_id["start-2"]["live_legal_witness_count"] == 0
    assert all(by_id[f"start-{index}"]["passed"] for index in (0, 1, 3))


def test_p0_live_departure_rejects_zero_length_manifest_witness() -> None:
    runner = _load_runner_module()
    source = _p0_departure_reset_source(runner)
    source["candidates"][0]["static_departure_witnesses"][0]["path_m"] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    starts = tuple((float(index), 0.0, 1.0) for index in range(4))

    with pytest.raises(ValueError, match="zero-length"):
        runner._p0_live_departure_qualification(
            source,
            selected_candidate_ids=tuple(f"start-{index}" for index in range(4)),
            selected_positions=starts,
            observed_positions=starts,
            scene_query=object(),
            clearance_oracle=object(),
            bounds_min=(-10.0, -10.0, -10.0),
            bounds_max=(10.0, 10.0, 10.0),
            collision_usd_sha256="c" * 64,
            start_reset_manifest_sha256="s" * 64,
        )


def test_unrouteable_final_budget_uses_a_shared_physical_tail() -> None:
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="context",
        episode_id="episode",
        decision_id="budget_tail0",
        agent_features=tuple((f"uav{index}", (1.0, 1.0)) for index in range(4)),
        budget=(("time_remaining_s", 1.125),),
    )
    positions = tuple((float(index), 0.0, 1.0) for index in range(4))

    manifest = runner._budget_tail_manifest(
        context,
        positions,
        duration_s=1.125,
        observe_dwell_s=1.0,
    )

    assert manifest.candidate_id == "hm3d-public-budget-tail"
    assert manifest.source == "hm3d-public-budget-tail-v1"
    assert len(manifest.fragments) == 8
    for index in range(4):
        transit, observation = manifest.fragments[index * 2 : index * 2 + 2]
        assert transit.type_signature.fragment_type == "transit"
        assert transit.path == (positions[index], positions[index])
        assert transit.planned_end == pytest.approx(0.125)
        assert observation.type_signature.fragment_type == "observation"
        assert observation.path == (positions[index],)
        assert observation.planned_start == pytest.approx(0.125)
        assert observation.planned_end == pytest.approx(1.125)
    with pytest.raises(ValueError, match="frozen observation dwell"):
        runner._budget_tail_manifest(
            context,
            positions,
            duration_s=0.999,
            observe_dwell_s=1.0,
        )


def test_sub_dwell_final_budget_is_recorded_without_a_fake_outcome() -> None:
    runner = _load_runner_module()

    record = runner._unexecuted_budget_tail_record(
        duration_s=0.4833,
        observe_dwell_s=1.0,
    )

    assert record == {
        "manifest_hash": None,
        "elapsed_physics_s": 0.0,
        "unexecuted_remainder_s": pytest.approx(0.4833),
        "scheduled_completion_mode": "unexecuted_budget_remainder_below_observation_dwell",
        "execution": None,
    }
    with pytest.raises(ValueError, match="shorter than dwell"):
        runner._unexecuted_budget_tail_record(duration_s=1.0, observe_dwell_s=1.0)


def test_min_stationary_budget_tail_does_not_require_transit_settle_margin() -> None:
    runner = _load_runner_module()
    minimum = runner._min_stationary_budget_tail_s(
        observation_dwell_s=1.0,
        physics_dt_s=1.0 / 120.0,
    )
    assert minimum == pytest.approx(1.0 + 1.0 / 120.0)

def test_episode_mobility_summary_reports_realised_speed_distance_and_height_bands() -> None:
    runner = _load_runner_module()
    decisions = [
        {
            "candidate_reachability": {
                "vertical_opportunity": {
                    "schema_version": "hm3d-p07-vertical-opportunity-v3",
                    "raw_exposed_upward_frontier_count": 2,
                    "raw_exposed_downward_frontier_count": 1,
                    "public_free_path_reachable_upward_edge_count": 2,
                    "public_free_path_reachable_downward_edge_count": 1,
                    "static_guard_admitted_upward_edge_count": 1,
                    "static_guard_admitted_downward_edge_count": 0,
                    "team_feasible_upward_edge_count": 1,
                    "team_feasible_downward_edge_count": 0,
                    "selected_upward_edge_count": 1,
                    "selected_downward_edge_count": 0,
                    "completed_vertical_agent_count": 1,
                }
            },
            "execution_calibration": {
                "agents": [
                    {
                        "agent_id": "uav0",
                        "route_geometry": {"command_path_length_m": 2.0},
                        "realized_transit_path_length_m": 1.8,
                        "maximum_linear_speed_mps": 1.1,
                        "transit_attempted": True,
                        "transit_completed": True,
                        "controller_tracking_samples": [
                            {"post_step_position_m": [0.0, 0.0, 1.2]},
                            {"post_step_position_m": [0.0, 0.0, 2.2]},
                        ],
                    }
                ]
            },
        }
    ]

    summary = runner._episode_mobility_summary(decisions, ((0.0, 0.0, 1.0),))

    assert summary["realised_fleet_path_length_m"] == pytest.approx(1.8)
    assert summary["maximum_realised_speed_mps"] == pytest.approx(1.1)
    assert summary["transit_completion_fraction"] == pytest.approx(1.0)
    assert summary["raw_exposed_vertical_frontier_count"] == 3
    assert summary["public_free_path_reachable_vertical_edge_count"] == 3
    assert summary["static_guard_admitted_vertical_edge_count"] == 1
    assert summary["team_feasible_vertical_edge_count"] == 1
    assert summary["selected_vertical_edge_count"] == 1
    assert summary["completed_vertical_agent_count"] == 1
    assert summary["cross_height_band_agent_count"] == 1


def test_empty_public_observation_cooldown_blocks_only_the_immediate_retry() -> None:
    runner = _load_runner_module()
    cooldown = runner._PublicObservationCooldown()
    target = (3, -1, 5)

    cooldown.begin_decision(0)
    update = cooldown.observe_empty_targets(
        [target], decision_index=0, public_new_free_voxel_count=0
    )
    assert update["applied"] is True

    cooldown.begin_decision(1)
    assert cooldown.blocks(target, decision_index=1) is True
    audit = cooldown.audit(decision_index=1)
    assert audit["active_target_voxel_keys"] == [[3, -1, 5]]
    assert audit["filtered_viewpoint_count"] == 1

    cooldown.begin_decision(2)
    assert cooldown.blocks(target, decision_index=2) is False


def test_stationarity_supervision_rejects_a_completed_micro_exploration() -> None:
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="stationarity",
        episode_id="stationarity-episode",
        decision_id="stationarity-decision",
        agent_features=(("uav0", (1.0,)),),
    )
    fragment = runner.FragmentInstance(
        instance_fragment_id="stationarity-transit",
        type_signature=runner.FragmentTypeSignature(
            "transit", (("assignment_role", "explore"), ("hold_reason", ""))
        ),
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        agent_id="uav0",
        planned_start=0.0,
        planned_end=1.0,
        path=((0.0, 0.0, 1.0), (0.5, 0.0, 1.0)),
        pose_mode="unit",
    )
    manifest = runner.CandidateFragmentManifest(
        candidate_id="stationarity-candidate",
        context_hash=context.digest,
        fragments=(fragment,),
        planned_descriptor=(0.0, 0.0, 0.0),
        feasible=True,
        quality_hint=1.0,
        cost_hint=1.0,
        source="unit",
    )
    audit = runner._decision_stationarity_supervision(
        manifest,
        {
            "execution_elapsed_physics_s": 2.0,
            "agents": [
                {
                    "agent_id": "uav0",
                    "route_geometry": {"command_path_length_m": 0.5},
                    "realized_transit_path_length_m": 0.02,
                    "transit_completed": True,
                    "transit_completed_at_s": 0.5,
                    "observation_started_at_s": 0.5,
                    "observation_completed_at_s": 1.5,
                }
            ],
        },
    )

    assert audit["status"] == "STATIONARITY_SUPERVISION_NOT_ADMITTED"
    assert audit["violations"] == ["uav0:subthreshold_realised_exploration"]
    assert audit["synchronization_wait_agent_seconds"] == pytest.approx(0.5)


def test_stationarity_supervision_accepts_a_completed_outcome_backtrack() -> None:
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="stationarity-backtrack",
        episode_id="stationarity-backtrack-episode",
        decision_id="stationarity-backtrack-decision",
        agent_features=(("uav1", (1.0,)),),
    )
    fragment = runner.FragmentInstance(
        instance_fragment_id="stationarity-backtrack-transit",
        type_signature=runner.FragmentTypeSignature(
            "transit", (("assignment_role", "backtrack"), ("hold_reason", ""))
        ),
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        agent_id="uav1",
        planned_start=0.0,
        planned_end=1.0,
        path=((0.6, 0.0, 1.0), (0.0, 0.0, 1.0)),
        pose_mode="unit",
    )
    manifest = runner.CandidateFragmentManifest(
        candidate_id="stationarity-backtrack-candidate",
        context_hash=context.digest,
        fragments=(fragment,),
        planned_descriptor=(0.0, 0.0, 0.0),
        feasible=True,
        quality_hint=0.0,
        cost_hint=1.0,
        source="unit",
    )

    audit = runner._decision_stationarity_supervision(
        manifest,
        {
            "execution_elapsed_physics_s": 2.0,
            "agents": [
                {
                    "agent_id": "uav1",
                    "route_geometry": {"command_path_length_m": 0.6},
                    "realized_transit_path_length_m": 0.55,
                    "transit_completed": True,
                    "transit_completed_at_s": 0.8,
                    "observation_started_at_s": 0.8,
                    "observation_completed_at_s": 1.8,
                }
            ],
        },
    )

    assert audit["status"] == "STATIONARITY_SUPERVISION_ADMITTED"
    assert audit["violations"] == []
    assert audit["agents"][0]["meaningful_realised_backtrack"] is True


def test_relative_height_bands_ignore_small_controller_drift() -> None:
    runner = _load_runner_module()

    assert runner._relative_height_band(-0.04, 1.0) == 0
    assert runner._relative_height_band(0.49, 1.0) == 0
    assert runner._relative_height_band(0.51, 1.0) == 1
    assert runner._relative_height_band(-0.51, 1.0) == -1


def test_vertical_completion_uses_realised_not_only_commanded_displacement() -> None:
    runner = _load_runner_module()
    frontier = runner.PublicFrontier("frontier-up", (0.0, 0.0, 1.0), 1.0, 0.0, "uav0")
    catalog = {
        "agents": [
            {
                "agent_id": "uav0",
                "frontier_edges": [
                    {
                        "task_kind": "explore",
                        "endpoint_m": [0.0, 0.0, 1.0],
                        "public_route_status": "admitted",
                        "individual_exploration_edge_admitted": True,
                        "appears_in_feasible_team_candidate": True,
                        "selected": True,
                    }
                ],
            }
        ]
    }
    execution = {
        "agents": [
            {
                "agent_id": "uav0",
                "command_path_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                "controller_tracking_samples": [
                    {"post_step_position_m": [0.0, 0.0, 0.0]},
                    {"post_step_position_m": [0.0, 0.0, 0.2]},
                ],
                "transit_completed": True,
            }
        ]
    }

    summary = runner._vertical_opportunity_summary(
        [frontier], [(0.0, 0.0, 0.0)], execution, catalog
    )

    assert summary["raw_exposed_upward_frontier_count"] == 1
    assert summary["static_guard_admitted_upward_edge_count"] == 1
    assert summary["team_feasible_upward_edge_count"] == 1
    assert summary["selected_upward_edge_count"] == 1
    assert summary["completed_vertical_agent_count"] == 0
    assert summary["maximum_realised_upward_displacement_m"] == pytest.approx(0.2)


def test_vertical_completion_excludes_unselected_recovery_or_backtrack_motion() -> None:
    runner = _load_runner_module()
    frontier = runner.PublicFrontier("frontier-up", (0.0, 0.0, 1.0), 1.0, 0.0, "uav0")
    catalog = {
        "agents": [
            {
                "agent_id": "uav0",
                "frontier_edges": [
                    {
                        "task_kind": "explore",
                        "endpoint_m": [0.0, 0.0, 1.0],
                        "public_route_status": "admitted",
                        "individual_exploration_edge_admitted": True,
                        "appears_in_feasible_team_candidate": True,
                        "selected": False,
                    }
                ],
            }
        ]
    }
    execution = {
        "agents": [
            {
                "agent_id": "uav0",
                "command_path_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
                "controller_tracking_samples": [
                    {"post_step_position_m": [0.0, 0.0, 0.0]},
                    {"post_step_position_m": [0.0, 0.0, 1.5]},
                ],
                "transit_completed": True,
            }
        ]
    }

    summary = runner._vertical_opportunity_summary(
        [frontier], [(0.0, 0.0, 0.0)], execution, catalog
    )

    assert summary["team_feasible_upward_edge_count"] == 1
    assert summary["selected_vertical_explore_agent_count"] == 0
    assert summary["completed_vertical_agent_count"] == 0
    assert summary["maximum_realised_upward_displacement_m"] == 0.0


def test_vertical_completion_keeps_directional_extrema_separate() -> None:
    runner = _load_runner_module()
    frontier = runner.PublicFrontier("frontier-down", (0.0, 0.0, -1.0), 1.0, 0.0, "uav0")
    catalog = {
        "agents": [
            {
                "agent_id": "uav0",
                "frontier_edges": [
                    {
                        "task_kind": "explore",
                        "endpoint_m": [0.0, 0.0, -1.0],
                        "public_route_status": "admitted",
                        "individual_exploration_edge_admitted": True,
                        "appears_in_feasible_team_candidate": True,
                        "selected": True,
                    }
                ],
            }
        ]
    }
    execution = {
        "agents": [
            {
                "agent_id": "uav0",
                "command_path_m": [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                "controller_tracking_samples": [
                    {"post_step_position_m": [0.0, 0.0, 0.0]},
                    {"post_step_position_m": [0.0, 0.0, -0.7]},
                ],
                "transit_completed": True,
            }
        ]
    }

    summary = runner._vertical_opportunity_summary(
        [frontier], [(0.0, 0.0, 0.0)], execution, catalog
    )

    assert summary["completed_downward_explore_agent_count"] == 1
    assert summary["maximum_raw_exposed_upward_endpoint_delta_m"] == 0.0
    assert summary["maximum_raw_exposed_downward_endpoint_delta_m"] == pytest.approx(-1.0)
    assert summary["maximum_realised_upward_displacement_m"] == 0.0
    assert summary["maximum_realised_downward_displacement_m"] == pytest.approx(-0.7)


def test_vertical_summary_reports_inconsistent_execution_path_without_raising() -> None:
    runner = _load_runner_module()
    frontier = runner.PublicFrontier("frontier-up", (0.0, 0.0, 1.0), 1.0, 0.0, "uav0")
    catalog = {
        "agents": [
            {
                "agent_id": "uav0",
                "frontier_edges": [
                    {
                        "task_kind": "explore",
                        "endpoint_m": [0.0, 0.0, 1.0],
                        "public_route_status": "admitted",
                        "individual_exploration_edge_admitted": True,
                        "appears_in_feasible_team_candidate": True,
                        "selected": True,
                    }
                ],
            }
        ]
    }
    execution = {
        "agents": [
            {
                "agent_id": "uav0",
                "command_path_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
                "controller_tracking_samples": [
                    {"post_step_position_m": [0.0, 0.0, 0.0]},
                    {"post_step_position_m": [0.0, 0.0, 0.15]},
                ],
                "transit_completed": True,
            }
        ]
    }

    summary = runner._vertical_opportunity_summary(
        [frontier], [(0.0, 0.0, 0.0)], execution, catalog
    )

    assert summary["execution_path_inconsistent_agent_ids"] == ["uav0"]
    assert summary["selected_vertical_explore_agent_count"] == 1
    assert summary["completed_vertical_agent_count"] == 0


def test_candidate_role_summary_exposes_holds_in_the_shared_pool() -> None:
    runner = _load_runner_module()
    fragments = tuple(
        runner.FragmentInstance(
            instance_fragment_id=f"fragment-{agent_id}",
            type_signature=runner.FragmentTypeSignature(
                "transit",
                (
                    ("assignment_role", role),
                    ("viewpoint_kind", "observation" if role == "explore" else "hold"),
                    ("hold_reason", "" if role == "explore" else "no_reachable_viewpoint"),
                ),
            ),
            episode_id="episode",
            decision_id="decision",
            agent_id=agent_id,
            planned_start=0.0,
            planned_end=1.0,
            path=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
            pose_mode="unit",
        )
        for agent_id, role in (("uav0", "explore"), ("uav1", "hold"))
    )
    candidate = runner.CandidateFragmentManifest(
        candidate_id="candidate",
        context_hash="a" * 64,
        fragments=fragments,
        planned_descriptor=(0.0, 0.0, 0.0),
        feasible=True,
        quality_hint=2.0,
        cost_hint=1.0,
        source="unit",
    )

    rows = runner._candidate_role_summary(
        [candidate], selected_manifest_hash=candidate.manifest_hash
    )

    assert len(rows) == 1
    row = rows[0]
    assert {
        key: row[key]
        for key in (
            "candidate_id",
            "feasible",
            "selected",
            "moving_explorer_count",
            "backtrack_count",
            "backtrack_agent_ids",
            "moving_agent_count",
            "hold_count",
            "hold_agent_ids",
            "hold_reasons_by_agent",
            "viewpoint_kinds_by_agent",
            "quality_hint",
            "cost_hint",
        )
    } == {
        "candidate_id": "candidate",
        "feasible": True,
        "selected": True,
        "moving_explorer_count": 1,
        "backtrack_count": 0,
        "backtrack_agent_ids": [],
        "moving_agent_count": 1,
        "hold_count": 1,
        "hold_agent_ids": ["uav1"],
        "hold_reasons_by_agent": {"uav1": "no_reachable_viewpoint"},
        "viewpoint_kinds_by_agent": {"uav0": "observation", "uav1": "hold"},
        "quality_hint": 2.0,
        "cost_hint": 1.0,
    }
    assert row["planned_path_length_m_by_agent"] == {"uav0": 0.1, "uav1": 0.1}
    assert row["team_planned_path_length_m"] == pytest.approx(0.2)
    assert row["expected_public_gain_proxy_by_agent"] == {"uav0": 0.0, "uav1": 0.0}
    assert row["team_expected_public_gain_proxy"] == pytest.approx(0.0)
    assert row["task_reservation_match_by_agent"] == {"uav0": False, "uav1": False}
    assert row["task_reservation_heading_alignment_by_agent"] == {"uav0": 0.0, "uav1": 0.0}
    assert row["task_reservation_switch_cost_by_agent"] == {"uav0": 0.0, "uav1": 0.0}
    assert row["task_reservation_switch_cost_total"] == pytest.approx(0.0)
    assert row["predicted_physical_makespan_s"] == pytest.approx(0.0)


def test_joint_guard_rejects_team_separation_before_building_relay_graph() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    guard_start = source.index("        def _joint_guard(")
    guard_end = source.index("        def joint_guard(", guard_start)
    guard_source = source[guard_start:guard_end]

    assert guard_source.index("assess_synchronized_separation(") < guard_source.index(
        "cf2x._initial_relay_graph("
    )
    assert guard_source.index("assess_route_tube_separation(") < guard_source.index(
        "cf2x._initial_relay_graph("
    )
    assert 'return "synchronized_fleet_separation"' in guard_source
    assert 'return "route_tube_separation"' in guard_source
    assert '"separation_assessment": assessment.to_public_dict()' in guard_source
    assert '"route_tube_assessment": route_tube_assessment.to_public_dict()' in guard_source


def test_joint_guard_has_a_narrow_audited_collision_envelope_recovery_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    guard_start = source.index("        def _joint_guard(")
    guard_end = source.index("        def joint_guard(", guard_start)
    guard_source = source[guard_start:guard_end]

    assert "assess_collision_avoidance_recovery(" in guard_source
    assert "cf2x.PLANNED_INTER_AGENT_ENDPOINT_SEPARATION_M" in guard_source
    assert "cf2x.WAYPOINT_SETTLE_SPEED_MPS" in guard_source
    assert "current_boundary_linear_speeds_mps" in guard_source
    assert 'return "planned_endpoint_separation_margin"' in guard_source
    assert 'return "malformed_collision_avoidance_recovery_metadata"' in guard_source
    assert 'return "collision_avoidance_recovery_rejected"' in guard_source


def test_recovery_execution_is_excluded_from_exploration_belief_qd_and_fragment_reuse() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "replay_exclusion_reason=(" in source
    assert '"COLLISION_AVOIDANCE_RECOVERY"' in source
    assert "integrate_into_belief=not selected_is_collision_avoidance_recovery" in source
    assert "AdmissionDecision(" in source
    assert '"metric_auc_contribution"' in source
    assert "0.0 if selected_is_collision_avoidance_recovery" in source


def test_public_frontier_selection_uses_interior_viewpoints_and_public_free_paths() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "FRONTIER_OBSERVATION_STANDOFF_M = 1.5" in source
    assert "FRONTIER_OBSERVATION_STANDOFF_VARIANTS_M = (1.5, 2.0, 2.5)" in source
    assert "def _known_free_observation_points(" in source
    assert "def _public_free_space_path(" in source
    assert 'reason="no_public_free_path"' in source


def test_online_p07_seeds_actuator_randomization_and_enables_physx_determinism() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "random.seed(args.random_key)" in source
    assert "np.random.seed(args.random_key % (2**32))" in source
    assert "torch.manual_seed(args.random_key)" in source
    assert "torch.cuda.manual_seed_all(args.random_key)" in source
    assert "PhysxCfg(enable_enhanced_determinism=True)" in source
    assert '"physx_enhanced_determinism": True' in source


def test_sparse_range_skips_dynamic_airframes_for_paired_method_determinism() -> None:
    execution_source = (
        RUNNER.parents[1] / "src" / "aerocity_method" / "runtime" / "hm3d_cf2x_execution.py"
    ).read_text(encoding="utf-8")

    assert "hit = _first_static_scene_hit(" in execution_source
    assert "endpoint_margin_m=0.0" in execution_source
    assert (
        "hit = self.scene_query.raycast_closest(position, direction, 20.0)" not in execution_source
    )


def test_online_p07_loads_the_calibrated_observation_dwell(tmp_path: Path) -> None:
    runner = _load_runner_module()
    path = tmp_path / "timing.json"
    path.write_text(
        '{"schema_version":"hm3d-cf2x-transit-timing-calibration-v4",'
        '"status":"CALIBRATION_PASS","time_model":{'
        '"schema_version":"hm3d-kinematic-transit-timing-v4",'
        '"calibration_id":"unit","cruise_speed_mps":0.5,'
        '"max_accel_mps2":0.4,"terminal_tracking_margin_s":0.2,'
        '"intermediate_waypoint_settle_margin_s":0.1,'
        '"calibrated_max_segment_count":2,"uncovered_segment_reserve_s":1.0,'
        '"intermediate_waypoint_requires_settle":true,'
        '"continuous_waypoint_speed_mps":0.35},'
        '"calibrated_max_segment_count":2,"uncovered_segment_reserve_s":1.0,'
        '"intermediate_waypoint_requires_settle":true,'
        '"continuous_waypoint_speed_mps":0.35,'
        '"observation_dwell_s":1.25}',
        encoding="utf-8",
    )

    _, observation_dwell_s = runner._load_transit_timing_contract(path)

    assert observation_dwell_s == 1.25


def test_online_p07_rejects_unpassed_transit_timing_calibration(tmp_path: Path) -> None:
    runner = _load_runner_module()
    path = tmp_path / "timing.json"
    path.write_text(
        '{"schema_version":"hm3d-cf2x-transit-timing-calibration-v2",'
        '"status":"CALIBRATION_FAILED","time_model":{}}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not a passed calibration or analytical pass-through artifact",
    ):
        runner._load_transit_timing_model(path)


def test_online_p07_rejects_a_timing_model_from_a_different_execution_profile(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    cf2x_path = tmp_path / "cf2x.usd"
    cf2x_path.write_text("unit-cf2x", encoding="utf-8")
    expected = runner._current_transit_execution_profile(
        cf2x_usd_path=cf2x_path,
        fleet_size=4,
        physics_dt_s=1.0 / 120.0,
        arrival_tolerance_m=0.1,
        outcome_time_tolerance_s=0.25,
    )
    calibrated = {**expected, "fleet_size": 2}
    path = tmp_path / "timing.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "hm3d-cf2x-transit-timing-calibration-v4",
                "status": "CALIBRATION_PASS",
                "time_model": {
                    "schema_version": "hm3d-kinematic-transit-timing-v4",
                    "calibration_id": "unit",
                    "cruise_speed_mps": 0.5,
                    "max_accel_mps2": 0.4,
                    "terminal_tracking_margin_s": 0.2,
                    "intermediate_waypoint_settle_margin_s": 0.1,
                    "calibrated_max_segment_count": 2,
                    "uncovered_segment_reserve_s": 1.0,
                    "intermediate_waypoint_requires_settle": True,
                    "continuous_waypoint_speed_mps": 0.35,
                },
                "calibrated_max_segment_count": 2,
                "uncovered_segment_reserve_s": 1.0,
                "intermediate_waypoint_requires_settle": True,
                "continuous_waypoint_speed_mps": 0.35,
                "observation_dwell_s": 1.0,
                "execution_profile": calibrated,
                "execution_profile_sha256": runner.canonical_sha256(calibrated),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fleet_size"):
        runner._load_transit_timing_contract(path, expected_execution_profile=expected)


def test_online_p07_accepts_legacy_receipt_tolerance_alias(tmp_path: Path) -> None:
    runner = _load_runner_module()
    cf2x_path = tmp_path / "cf2x.usd"
    cf2x_path.write_text("unit-cf2x", encoding="utf-8")
    expected = runner._current_transit_execution_profile(
        cf2x_usd_path=cf2x_path,
        fleet_size=4,
        physics_dt_s=1.0 / 120.0,
        arrival_tolerance_m=0.1,
        outcome_time_tolerance_s=0.25,
    )
    calibrated = {
        key: value
        for key, value in expected.items()
        if key != "outcome_time_tolerance_s"
    }
    calibrated["receipt_time_tolerance_s"] = 0.25

    def write_contract(tolerance_s: float) -> Path:
        path = tmp_path / f"timing-{tolerance_s}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "hm3d-cf2x-transit-timing-calibration-v4",
                    "status": "CALIBRATION_PASS",
                    "time_model": {
                        "schema_version": "hm3d-kinematic-transit-timing-v4",
                        "calibration_id": "unit-alias",
                        "cruise_speed_mps": 0.5,
                        "max_accel_mps2": 0.4,
                        "terminal_tracking_margin_s": 0.2,
                        "intermediate_waypoint_settle_margin_s": 0.1,
                        "calibrated_max_segment_count": 2,
                        "uncovered_segment_reserve_s": 1.0,
                        "intermediate_waypoint_requires_settle": True,
                        "continuous_waypoint_speed_mps": 0.35,
                    },
                    "calibrated_max_segment_count": 2,
                    "uncovered_segment_reserve_s": 1.0,
                    "intermediate_waypoint_requires_settle": True,
                    "continuous_waypoint_speed_mps": 0.35,
                    "observation_dwell_s": 1.0,
                    "execution_profile": {
                        **calibrated,
                        "receipt_time_tolerance_s": tolerance_s,
                    },
                    "execution_profile_sha256": runner.canonical_sha256(
                        {
                            **calibrated,
                            "receipt_time_tolerance_s": tolerance_s,
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path

    runner._load_transit_timing_contract(
        write_contract(0.25),
        expected_execution_profile=expected,
    )
    with pytest.raises(ValueError, match="outcome_time_tolerance_s"):
        runner._load_transit_timing_contract(
            write_contract(0.3),
            expected_execution_profile=expected,
        )


def test_online_p07_matches_calibration_abi_but_rejects_a_different_controller(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    cf2x_path = tmp_path / "cf2x.usd"
    cf2x_path.write_text("unit-cf2x", encoding="utf-8")
    mellinger_profile = runner._current_transit_execution_profile(
        cf2x_usd_path=cf2x_path,
        fleet_size=4,
        physics_dt_s=1.0 / 120.0,
        arrival_tolerance_m=0.1,
        outcome_time_tolerance_s=0.25,
        controller_id=BITCRAZE_MELLINGER_CONTROLLER_ID,
    )
    path = tmp_path / "mellinger-timing.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "hm3d-cf2x-transit-timing-calibration-v4",
                "status": "CALIBRATION_PASS",
                "time_model": {
                    "schema_version": "hm3d-kinematic-transit-timing-v4",
                    "calibration_id": "unit-mellinger",
                    "cruise_speed_mps": 0.5,
                    "max_accel_mps2": 0.4,
                    "terminal_tracking_margin_s": 0.2,
                    "intermediate_waypoint_settle_margin_s": 0.1,
                    "calibrated_max_segment_count": 2,
                    "uncovered_segment_reserve_s": 1.0,
                    "intermediate_waypoint_requires_settle": True,
                    "continuous_waypoint_speed_mps": 0.35,
                },
                "calibrated_max_segment_count": 2,
                "uncovered_segment_reserve_s": 1.0,
                "intermediate_waypoint_requires_settle": True,
                "continuous_waypoint_speed_mps": 0.35,
                "observation_dwell_s": 1.0,
                "execution_profile": mellinger_profile,
                "execution_profile_sha256": runner.canonical_sha256(mellinger_profile),
            }
        ),
        encoding="utf-8",
    )

    runner._load_transit_timing_contract(path, expected_execution_profile=mellinger_profile)
    v6_profile = runner._current_transit_execution_profile(
        cf2x_usd_path=cf2x_path,
        fleet_size=4,
        physics_dt_s=1.0 / 120.0,
        arrival_tolerance_m=0.1,
        outcome_time_tolerance_s=0.25,
        controller_id=CF2X_DEFAULT_CONTROLLER_ID,
    )
    with pytest.raises(ValueError, match="controller_tracking"):
        runner._load_transit_timing_contract(path, expected_execution_profile=v6_profile)


def test_online_p07_uses_connected_reset_but_allows_audited_runtime_disconnection() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "def _select_connected_initial_positions(" in source
    assert "public initial-position source has no relay-connected fleet reset" in source
    assert "initial_public_relay_graph" in source
    assert 'parser.add_argument("--start-reset-json", required=True, type=Path)' in source
    assert "P07 requires a dedicated start-reset manifest, never P04 calibration views" in source
    assert '"initial_start_reset": initial_start_reset_witness' in source
    assert "P07 CF2X reset does not match the pre-registered shared start poses" in source
    assert 'else "final_relay_disconnected"' in source
    assert 'return "final_relay_disconnected"' not in source
    assert '"communication_warning": (' in source
    assert '"joint_guard_records": joint_guard_records' in source
    assert '"selected_joint_safety": selected_joint_safety' in source
    assert '"completed_decision_count": len(decisions)' in source


def test_undelivered_range_updates_cannot_enter_the_shared_belief() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "undelivered range-map updates cannot enter the shared belief" in source
    assert "communication_audit=bootstrap_communication_audit" in source
    assert "communication_audit=round_communication_audit" in source


def test_public_frontiers_are_team_shared_not_required_per_source_agent() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "too few reachable interior observation viewpoints" in source
    assert "no_public_free_path" in source
    assert "if len(candidates) < len(positions):" in source
    assert "extract_frontier_clusters" in source


def test_public_frontier_view_set_is_bounded_per_agent() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "uav0", 0.25)
    # The production generator requires multiple received-free voxels in a
    # 0.50 m support window. Build an actual 3-D interior rather than a sheet.
    for x_index in range(-2, 14):
        for y_index in range(-2, 4):
            for z_index in range(-10, 11):
                belief.set_state((x_index, y_index, z_index), FREE)

    rows = runner._public_frontiers_from_belief(
        ((0.125, 0.125, 0.125),),
        belief,
        decision_index=0,
        maximum_step_m=3.0,
        maximum_frontiers_per_agent=3,
    )

    assert 2 <= len(rows) <= 6
    starts = {"uav0": (0.125, 0.125, 0.125), "uav1": (2.125, 0.125, 0.125)}
    assert all(
        runner._public_free_space_path(
            belief,
            starts[row.source_agent_id],
            row.position_m,
            maximum_path_length_m=3.0,
        )
        is not None
        for row in rows
    )


def test_public_frontier_deduplication_never_rounds_past_the_time_budget() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    for x_index in range(-2, 14):
        for y_index in range(-2, 4):
            for z_index in range(-10, 11):
                belief.set_state((x_index, y_index, z_index), FREE)
    start = (0.125, 0.125, 0.125)
    maximum_step_m = 3.0

    rows = runner._public_frontiers_from_belief(
        (start,),
        belief,
        decision_index=6,
        maximum_step_m=maximum_step_m,
        maximum_frontiers_per_agent=3,
    )

    assert rows
    assert all(
        runner._public_free_space_path(
            belief,
            start,
            row.position_m,
            maximum_path_length_m=maximum_step_m,
        )
        is not None
        for row in rows
    )


def test_public_frontiers_include_route_level_region_access() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    # A free volume that extends far beyond the 2.5 m frontier standoff is the
    # minimal case where the old near-frontier viewpoints collapse into
    # micro-routes even though a connected access route exists.
    for x_index in range(-2, 34):
        for y_index in range(-2, 8):
            for z_index in range(-10, 11):
                belief.set_state((x_index, y_index, z_index), FREE)

    rows = runner._public_frontiers_from_belief(
        ((0.125, 0.125, 0.125), (2.125, 0.125, 0.125)),
        belief,
        decision_index=0,
        maximum_step_m=20.0,
        maximum_frontiers_per_agent=3,
    )

    region_rows = [row for row in rows if row.viewpoint_kind == "region_access"]
    assert region_rows
    assert any(
        sum(
            math.dist(left, right)
            for left, right in zip(path, path[1:], strict=False)
        )
        >= runner.PUBLIC_REGION_ACCESS_MIN_ADVANCE_M
        for row in region_rows
        for _agent_id, path in row.access_paths_m
    )


def test_region_access_has_reserved_search_budget_when_observation_budget_exhausted() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    # A free volume that exposes a long corridor is the same public-map
    # condition as the normal region-access test. The difference is that the
    # observation standoff budget is deliberately exhausted before generation,
    # which previously prevented the region-access loop from running at all.
    for x_index in range(-2, 34):
        for y_index in range(-2, 8):
            for z_index in range(-10, 11):
                belief.set_state((x_index, y_index, z_index), FREE)

    cache = runner._PublicFreeReachabilityCache(belief)
    cache._bounded_path_search_count = (
        runner.PUBLIC_FRONTIER_PATH_SEARCH_BUDGET_PER_DECISION
        - runner.PUBLIC_REGION_ACCESS_PATH_SEARCH_RESERVE_PER_DECISION
    )

    rows = runner._public_frontiers_from_belief(
        ((0.125, 0.125, 0.125),),
        belief,
        decision_index=0,
        maximum_step_m=20.0,
        maximum_frontiers_per_agent=3,
        reachability_cache=cache,
    )

    region_rows = [row for row in rows if row.viewpoint_kind == "region_access"]
    assert region_rows
    assert cache._region_access_attempt_count > 0
    assert cache._region_access_generated_count > 0
    audit = cache.audit()
    assert audit["region_access_attempt_count"] == cache._region_access_attempt_count
    assert audit["region_access_generated_count"] == cache._region_access_generated_count


def test_per_agent_edge_diagnostics_distinguish_time_path_and_anchor_failures() -> None:
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="edge-diagnostics",
        episode_id="edge-diagnostics-episode",
        decision_id="edge-diagnostics-decision",
        agent_features=(("uav0", (1.0,)), ("uav1", (1.0,))),
        budget=(("time_remaining_s", 10.0),),
    )
    state = runner.PublicSearchState(
        context=context,
        agents=(
            runner.PublicAgentPose("uav0", (0.125, 0.125, 0.125), 1.0, 1),
            runner.PublicAgentPose("uav1", (4.125, 0.125, 0.125), 1.0, 1),
        ),
        frontiers=(
            runner.PublicFrontier("near", (1.125, 0.125, 0.125), 1.0, 0.0),
            runner.PublicFrontier("far", (9.125, 0.125, 0.125), 1.0, 0.0),
        ),
        decision_start_s=0.0,
        decision_duration_s=3.0,
        transit_timing_model=runner.ConservativeTransitTimingModel("edge-test", 1.0, 10.0, 0.0),
        observe_dwell_s=0.5,
    )
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    belief.set_state((0, 0, 0), FREE)
    records = [
        {
            "agent_id": "uav0",
            "cache_hit": False,
            "requested_path_m": ((0.125, 0.125, 0.125), (1.125, 0.125, 0.125)),
            "guarded_path_m": ((0.125, 0.125, 0.125), (1.125, 0.125, 0.125)),
            "legal": True,
            "reason": None,
        },
        {
            "agent_id": "uav1",
            "cache_hit": False,
            "requested_path_m": ((4.125, 0.125, 0.125), (1.125, 0.125, 0.125)),
            "guarded_path_m": ((4.125, 0.125, 0.125), (1.125, 0.125, 0.125)),
            "legal": False,
            "reason": "no_public_free_path",
        },
    ]

    rows = {
        row["agent_id"]: row
        for row in runner._per_agent_candidate_edge_diagnostics(state, belief, records)
    }

    assert rows["uav0"]["public_free_start_anchor_present"] is True
    assert rows["uav0"]["legal_frontier_edge_count"] == 1
    assert rows["uav0"]["guard_legal_frontier_edge_count"] == 1
    assert rows["uav0"]["meaningful_guard_legal_exploration_edge_count"] == 1
    assert rows["uav0"]["non_alias_guard_legal_exploration_edge_count"] == 1
    assert rows["uav0"]["endpoint_alias_rejected_edge_count"] == 0
    assert rows["uav0"]["legacy_subhalfmetre_guard_legal_exploration_edge_count"] == 0
    assert rows["uav0"]["nearest_legal_frontier_id"] == "near"
    assert rows["uav1"]["public_free_start_anchor_present"] is False
    assert rows["uav1"]["legal_frontier_edge_count"] == 0
    assert rows["uav1"]["route_guard_reason_counts"] == {"no_public_free_path": 1}
    assert rows["uav1"]["public_route_status_counts"] == {"not_applicable": 1}
    assert rows["uav1"]["time_lower_bound_rejected_edge_count"] == 2


def test_per_agent_edge_diagnostics_expose_terminal_pullback_audit() -> None:
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="pullback-audit",
        episode_id="pullback-audit-episode",
        decision_id="pullback-audit-decision",
        agent_features=(("uav0", (1.0,)),),
        budget=(("time_remaining_s", 10.0),),
    )
    state = runner.PublicSearchState(
        context=context,
        agents=(runner.PublicAgentPose("uav0", (0.125, 0.125, 0.125), 1.0, 1),),
        frontiers=(
            runner.PublicFrontier("near", (1.125, 0.125, 0.125), 1.0, 0.0),
            runner.PublicFrontier("blocked", (2.125, 0.125, 0.125), 1.0, 0.0),
        ),
        decision_start_s=0.0,
        decision_duration_s=3.0,
        transit_timing_model=runner.ConservativeTransitTimingModel("edge-test", 1.0, 10.0, 0.0),
        observe_dwell_s=0.5,
    )
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    belief.set_state((0, 0, 0), FREE)
    belief.set_state((1, 0, 0), FREE)
    belief.set_state((2, 0, 0), FREE)
    records = [
        {
            "agent_id": "uav0",
            "cache_hit": False,
            "requested_path_m": ((0.125, 0.125, 0.125), (2.125, 0.125, 0.125)),
            "guarded_path_m": ((0.125, 0.125, 0.125), (2.125, 0.125, 0.125)),
            "legal": False,
            "reason": "insufficient_continuous_collision_clearance",
            "public_route_status": "admitted",
            "clearance_rejection_stage_counts": {"endpoint": 1},
            "terminal_pullback_attempted": True,
            "terminal_pullback_admitted": False,
            "terminal_pullback_failure_reason": "no_terminal_clearance_pullback_guard_admitted",
        }
    ]

    row = runner._per_agent_candidate_edge_diagnostics(state, belief, records)[0]

    assert row["terminal_pullback_attempted_count"] == 1
    assert row["terminal_pullback_admitted_count"] == 0
    assert row["terminal_pullback_failure_reason_counts"] == {
        "no_terminal_clearance_pullback_guard_admitted": 1
    }


def test_per_agent_edge_diagnostics_distinguish_short_non_alias_progress_from_endpoint_alias():
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="micro-edge-diagnostics",
        episode_id="micro-edge-episode",
        decision_id="micro-edge-decision",
        agent_features=(("uav0", (1.0,)),),
        budget=(("time_remaining_s", 10.0),),
    )
    state = runner.PublicSearchState(
        context=context,
        agents=(runner.PublicAgentPose("uav0", (0.125, 0.125, 0.125), 1.0, 1),),
        frontiers=(
            runner.PublicFrontier("micro", (0.375, 0.125, 0.125), 1.0, 0.0),
        ),
        decision_start_s=0.0,
        decision_duration_s=3.0,
        transit_timing_model=runner.ConservativeTransitTimingModel("micro-edge", 1.0, 10.0, 0.0),
        observe_dwell_s=0.5,
    )
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    belief.set_state((0, 0, 0), FREE)
    records = [
        {
            "agent_id": "uav0",
            "cache_hit": False,
            "requested_path_m": ((0.125, 0.125, 0.125), (0.375, 0.125, 0.125)),
            "guarded_path_m": ((0.125, 0.125, 0.125), (0.375, 0.125, 0.125)),
            "legal": True,
            "reason": None,
        }
    ]

    row = runner._per_agent_candidate_edge_diagnostics(state, belief, records)[0]

    assert row["guard_legal_frontier_edge_count"] == 1
    assert row["meaningful_guard_legal_exploration_edge_count"] == 0
    assert row["non_alias_guard_legal_exploration_edge_count"] == 1
    assert row["endpoint_alias_rejected_edge_count"] == 0
    assert row["legacy_subhalfmetre_guard_legal_exploration_edge_count"] == 1
    assert row["nearest_guard_legal_path_length_m"] == pytest.approx(0.25)
    assert row["nearest_meaningful_guard_legal_frontier_id"] is None


def test_candidate_route_opportunity_catalog_reports_longest_nonreverse_without_selection():
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="route-catalog-context",
        episode_id="route-catalog-episode",
        decision_id="route-catalog-decision",
        agent_features=(("uav0", (1.0,)),),
        budget=(("time_remaining_s", 10.0),),
    )
    reservation = runner.PublicTaskReservation.from_completed_public_exploration_transit(
        agent_id="uav0",
        source_decision_id="prior-decision",
        source_manifest_hash="0" * 64,
        source_transit_outcome_sha256="1" * 64,
        public_path_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    state = runner.PublicSearchState(
        context=context,
        agents=(runner.PublicAgentPose("uav0", (1.0, 0.0, 0.0), 1.0, 1),),
        frontiers=(
            runner.PublicFrontier("forward", (1.75, 0.0, 0.0), 1.0, 0.0),
            runner.PublicFrontier("reverse", (-0.25, 0.0, 0.0), 1.0, 0.0),
            runner.PublicFrontier("blocked", (2.5, 0.0, 0.0), 1.0, 0.0),
        ),
        decision_start_s=0.0,
        decision_duration_s=10.0,
        transit_timing_model=runner.ConservativeTransitTimingModel(
            "route-catalog", 1.0, 10.0, 0.0
        ),
        observe_dwell_s=0.1,
        task_reservations=(reservation,),
    )
    records = [
        {
                "agent_id": "uav0",
                "cache_hit": True,
                "public_access_frontier_id": "forward",
                "requested_path_m": ((1.0, 0.0, 0.0), (1.75, 0.0, 0.0)),
                "guarded_path_m": ((1.0, 0.0, 0.0), (1.75, 0.0, 0.0)),
            "legal": True,
            "reason": "",
            "public_route_status": "revalidated_public_access_plan",
        },
        {
            "agent_id": "uav0",
            "cache_hit": False,
            "public_access_frontier_id": "blocked",
            "requested_path_m": ((1.0, 0.0, 0.0), (2.5, 0.0, 0.0)),
            "guarded_path_m": ((1.0, 0.0, 0.0), (2.5, 0.0, 0.0)),
            "legal": False,
            "reason": "insufficient_continuous_collision_clearance",
            "clearance_rejection_stage_counts": {"endpoint": 1},
            "public_route_status": "revalidated_public_access_plan",
        },
        {
                "agent_id": "uav0",
                "cache_hit": False,
                "public_access_frontier_id": "reverse",
                "requested_path_m": ((1.0, 0.0, 0.0), (-0.25, 0.0, 0.0)),
                "guarded_path_m": ((1.0, 0.0, 0.0), (-0.25, 0.0, 0.0)),
            "legal": True,
            "reason": "",
            "public_route_status": "revalidated_public_access_plan",
        },
    ]

    catalog = runner._candidate_route_opportunity_catalog(state, records, ())

    summary = catalog["agents"][0]["summary"]
    assert summary["individual_exploration_edge_count"] == 2
    assert summary["individual_reserved_task_edge_count"] == 1
    assert summary["team_matching_lost_exploration_edge_count"] == 2
    assert summary["static_guard_rejected_frontier_edge_count"] == 1
    assert summary["static_guard_rejection_reason_counts"] == {
        "insufficient_continuous_collision_clearance": 1
    }
    assert summary["static_guard_rejection_stage_counts"] == {"endpoint": 1}
    assert summary["longest_individual_exploration_edge"]["frontier_id"] == "reverse"
    assert summary["longest_individual_reserved_task_edge"]["frontier_id"] == "forward"
    assert summary["selected_edge"] is None
    forward_row = next(
        row for row in catalog["agents"][0]["frontier_edges"]
        if row["frontier_id"] == "forward"
    )
    assert forward_row["route_guard_cache_hit"] is True
    assert forward_row["feasible_team_candidate_ids"] == []
    assert forward_row["feasible_team_manifest_hashes"] == []
    assert forward_row["selected_candidate_contains_edge"] is False
    assert catalog["schema_version"] == "hm3d-candidate-route-opportunity-catalog-v3"
    assert len(catalog["catalog_sha256"]) == 64


def test_candidate_route_opportunity_catalog_binds_selected_frontier_to_manifest():
    runner = _load_runner_module()
    context = runner.PublicMethodContext(
        context_id="route-catalog-selected-context",
        episode_id="route-catalog-selected-episode",
        decision_id="route-catalog-selected-decision",
        agent_features=(("uav0", (1.0,)),),
        budget=(("time_remaining_s", 10.0),),
    )
    state = runner.PublicSearchState(
        context=context,
        agents=(runner.PublicAgentPose("uav0", (0.0, 0.0, 0.0), 1.0, 1),),
        frontiers=(runner.PublicFrontier("view", (1.0, 0.0, 0.0), 1.0, 0.0),),
        decision_start_s=0.0,
        decision_duration_s=10.0,
        transit_timing_model=runner.ConservativeTransitTimingModel(
            "route-catalog-selected", 1.0, 10.0, 0.0
        ),
        observe_dwell_s=0.1,
    )

    def guard(_agent_id, path_m):
        return runner.GuardedPath(legal=True, path_m=tuple(path_m))

    pool = runner.build_public_candidate_pool(state, guard, candidate_limit=1)
    assert len(pool) == 1
    selected = pool[0]
    records = [
        {
            "agent_id": "uav0",
            "cache_hit": False,
            "public_access_frontier_id": "view",
            "requested_path_m": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            "guarded_path_m": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            "legal": True,
            "reason": "",
            "public_route_status": "revalidated_public_access_plan",
        }
    ]

    catalog = runner._candidate_route_opportunity_catalog(
        state,
        records,
        pool,
        selected=selected,
    )
    row = catalog["agents"][0]["frontier_edges"][0]
    assert row["appears_in_feasible_team_candidate"] is True
    assert row["feasible_team_candidate_ids"] == [selected.candidate_id]
    assert row["feasible_team_manifest_hashes"] == [selected.manifest_hash]
    assert row["selected_candidate_contains_edge"] is True
    assert row["selected"] is True


def test_public_endpoint_alias_tolerance_matches_cf2x_settle_contract() -> None:
    runner = _load_runner_module()

    assert runner.PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M == pytest.approx(
        runner.cf2x.WAYPOINT_SETTLE_POSITION_TOLERANCE_M
    )


def test_outcome_backtrack_frontier_preserves_only_a_completed_owned_route() -> None:
    runner = _load_runner_module()
    route = runner._OutcomeBacktrackRoute(
        route_id="decision0-uav1-0123456789ab",
        agent_id="uav1",
        source_decision_id="decision0",
        source_manifest_hash="a" * 64,
        source_transit_outcome_sha256="b" * 64,
        source_minimum_static_mesh_clearance_m=0.52,
        source_static_clearance_contract_required_m=0.40,
        path_m=((0.0, 0.0, 1.0), (0.6, 0.0, 1.0)),
    )

    result = runner._outcome_backtrack_frontier(
        current_position_m=(0.61, 0.0, 1.0),
        route=route,
        arrival_tolerance_m=0.1,
        occupied_endpoints_m=(),
    )

    assert result is not None
    frontier, reverse_path = result
    assert frontier.task_kind == "backtrack"
    assert frontier.exclusive_agent_id == "uav1"
    assert frontier.information_gain == 0.0
    assert reverse_path == ((0.61, 0.0, 1.0), (0.0, 0.0, 1.0))
    clearance = runner._outcome_backtrack_clearance_reuse_audit(
        current_position_m=(0.61, 0.0, 1.0),
        route=route,
        requested_path_m=reverse_path,
    )
    assert clearance["admitted"] is True
    assert clearance["source_clearance_slack_m"] == pytest.approx(0.12)
    insufficient_clearance = runner._outcome_backtrack_clearance_reuse_audit(
        current_position_m=(0.75, 0.0, 1.0),
        route=route,
        requested_path_m=((0.75, 0.0, 1.0), (0.0, 0.0, 1.0)),
    )
    assert insufficient_clearance["admitted"] is False
    assert insufficient_clearance["reason"] == "endpoint_offset_exceeds_source_clearance_slack"
    assert runner._outcome_backtrack_frontier(
        current_position_m=(0.3, 0.0, 1.0),
        route=route,
        arrival_tolerance_m=0.1,
        occupied_endpoints_m=(),
    ) is None
    assert runner._outcome_backtrack_frontier(
        current_position_m=(0.61, 0.0, 1.0),
        route=route,
        arrival_tolerance_m=0.1,
        occupied_endpoints_m=((0.0, 0.0, 1.0),),
    ) is None


def test_route_guard_record_retains_per_stage_exact_clearance_minima() -> None:
    runner = _load_runner_module()
    start = (0.0, 0.0, 0.0)
    end = (1.0, 0.0, 0.0)
    guarded = runner.GuardedPath(
        legal=False,
        path_m=(start, end),
        reason="insufficient_continuous_collision_clearance",
    )
    record = runner._route_guard_record(
        agent_id="uav3",
        requested_path_m=(start, end),
        guarded=guarded,
        events=[
            {
                "event_type": "static_clearance_rejection",
                "stage": "endpoint",
                "required_clearance_m": 0.4,
                "minimum_static_mesh_clearance_m": 0.36,
                "minimum_clearance_position_m": (1.0, 0.0, 0.0),
            },
            {
                "event_type": "static_clearance_rejection",
                "stage": "endpoint",
                "required_clearance_m": 0.4,
                "minimum_static_mesh_clearance_m": 0.31,
                "minimum_clearance_position_m": (0.8, 0.0, 0.0),
            },
            {
                "event_type": "static_clearance_rejection",
                "stage": "interior",
                "required_clearance_m": 0.45,
                "minimum_static_mesh_clearance_m": 0.28,
                "minimum_clearance_position_m": (0.5, 0.0, 0.0),
            },
        ],
    )

    assert record["clearance_rejection_stage_counts"] == {"endpoint": 2, "interior": 1}
    assert record["clearance_rejections_by_stage"] == {
        "endpoint": {
            "rejection_count": 2,
            "required_clearance_m": 0.4,
            "minimum_static_mesh_clearance_m": 0.31,
            "minimum_clearance_position_m": (0.8, 0.0, 0.0),
        },
        "interior": {
            "rejection_count": 1,
            "required_clearance_m": 0.45,
            "minimum_static_mesh_clearance_m": 0.28,
            "minimum_clearance_position_m": (0.5, 0.0, 0.0),
        },
    }


def test_public_free_path_result_distinguishes_horizon_from_disconnected_map() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    for key in ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (8, 0, 0)):
        belief.set_state(key, FREE)

    bounded = runner._public_free_space_path_result(
        belief,
        (0.125, 0.125, 0.125),
        (0.875, 0.125, 0.125),
        maximum_path_length_m=0.5,
    )
    disconnected = runner._public_free_space_path_result(
        belief,
        (0.125, 0.125, 0.125),
        (2.125, 0.125, 0.125),
        maximum_path_length_m=4.0,
    )

    assert bounded.path_m is None
    assert bounded.status == "path_exceeds_step_budget"
    assert disconnected.path_m is None
    assert disconnected.status == "public_free_component_disconnected"


def test_public_free_reachability_cache_preserves_path_result_semantics() -> None:
    """The cache only avoids repeated diagnostic floods; route outcomes stay exact."""

    runner = _load_runner_module()

    def paired_result(
        belief: SparseVoxelBelief,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        **kwargs: float,
    ) -> None:
        uncached = runner._public_free_space_path_result(belief, start, goal, **kwargs)
        cached = runner._public_free_space_path_result(
            belief,
            start,
            goal,
            reachability_cache=runner._PublicFreeReachabilityCache(belief),
            **kwargs,
        )
        assert cached == uncached

    start = (0.125, 0.125, 0.125)
    chain_belief = SparseVoxelBelief("scene0", "team", 0.25)
    for key in ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)):
        chain_belief.set_state(key, FREE)
    paired_result(chain_belief, start, (0.875, 0.125, 0.125), maximum_path_length_m=2.0)
    paired_result(chain_belief, start, (0.875, 0.125, 0.125), maximum_path_length_m=0.5)

    disconnected_belief = SparseVoxelBelief("scene0", "team", 0.25)
    for key in ((0, 0, 0), (1, 0, 0), (8, 0, 0)):
        disconnected_belief.set_state(key, FREE)
    paired_result(
        disconnected_belief,
        start,
        (2.125, 0.125, 0.125),
        maximum_path_length_m=4.0,
    )

    unsupported_belief = SparseVoxelBelief("scene0", "team", 0.25)
    for key in ((0, 0, 0), (1, 0, 0)):
        unsupported_belief.set_state(key, FREE)
    paired_result(
        unsupported_belief,
        start,
        (0.375, 0.125, 0.125),
        maximum_path_length_m=1.0,
        minimum_received_free_support_m=0.25,
    )


def test_public_free_reachability_cache_avoids_repeated_component_floods(monkeypatch) -> None:
    """Many disconnected targets must not each rescan the same source component."""

    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    for x_index in range(33):
        belief.set_state((x_index, 0, 0), FREE)
    belief.set_state((48, 0, 0), FREE)
    belief.set_state((58, 0, 0), FREE)
    cache = runner._PublicFreeReachabilityCache(belief)
    original_neighbors_26 = runner.neighbors_26
    neighbor_call_count = 0

    def counted_neighbors_26(key):
        nonlocal neighbor_call_count
        neighbor_call_count += 1
        return original_neighbors_26(key)

    monkeypatch.setattr(runner, "neighbors_26", counted_neighbors_26)
    start = (0.125, 0.125, 0.125)
    first = runner._public_free_space_path_result(
        belief,
        start,
        (12.125, 0.125, 0.125),
        maximum_path_length_m=0.25,
        reachability_cache=cache,
    )
    calls_after_first = neighbor_call_count
    second = runner._public_free_space_path_result(
        belief,
        start,
        (14.625, 0.125, 0.125),
        maximum_path_length_m=0.25,
        reachability_cache=cache,
    )

    assert first.status == "public_free_component_disconnected"
    assert second.status == "public_free_component_disconnected"
    assert neighbor_call_count - calls_after_first == 1
    assert cache.audit()["component_flood_count"] == 1
    assert cache.audit()["component_cache_free_voxel_count"] == 33
    assert cache.audit()["component_cached_disconnected_rejections"] == 1


def test_public_free_path_follows_a_diagonal_received_free_ray() -> None:
    """A received diagonal corridor must reach the common static route guard."""

    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "uav0", 0.25)
    start = (0.125, 0.125, 0.125)
    goal = (1.125, 1.125, 1.125)
    assert belief.integrate_ray(
        PublicRangeRayOutcome("diagonal-free", "uav0", 1.0, start, goal, False)
    )

    result = runner._public_free_space_path_result(
        belief,
        start,
        goal,
        maximum_path_length_m=2.0,
    )

    assert result.status == "admitted"
    assert result.path_m == (start, goal)


def test_public_free_path_requires_received_interior_support_when_requested() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    start = (0.125, 0.125, 0.125)
    goal = (0.375, 0.125, 0.125)
    belief.set_state((0, 0, 0), FREE)
    belief.set_state((1, 0, 0), FREE)

    unsupported = runner._public_free_space_path_result(
        belief,
        start,
        goal,
        maximum_path_length_m=1.0,
        minimum_received_free_support_m=0.25,
    )

    assert unsupported.path_m is None
    assert unsupported.status == "public_free_interior_support_missing"

    for neighbor in ((1, 0, 0), (0, 0, 0), (2, 0, 0), (1, -1, 0), (1, 1, 0), (1, 0, -1), (1, 0, 1)):
        belief.set_state(neighbor, FREE)
    supported = runner._public_free_space_path_result(
        belief,
        start,
        goal,
        maximum_path_length_m=1.0,
        minimum_received_free_support_m=0.25,
    )

    assert supported.status == "admitted"
    assert supported.path_m == (start, goal)


def test_component_progress_advances_inside_known_free_space_toward_disconnected_goal() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    start_m = (0.125, 0.125, 0.125)
    goal_m = (12.125, 0.125, 0.125)
    # A known corridor approaches the frontier, but the last span is not yet
    # observed, so the direct public path is disconnected.
    for x_index in range(0, 33):
        belief.set_state((x_index, 0, 0), FREE)
    belief.set_state((48, 0, 0), FREE)

    direct = runner._public_free_space_path_result(
        belief,
        start_m,
        goal_m,
        maximum_path_length_m=8.0,
    )
    progress = runner._public_component_progress_path_result(
        belief,
        start_m,
        goal_m,
        maximum_path_length_m=8.0,
        minimum_advance_m=0.5,
    )

    assert direct.status == "public_free_component_disconnected"
    assert progress.status == "public_component_progress"
    assert progress.path_m is not None
    assert len(progress.path_m) >= 2
    assert math.dist(progress.path_m[-1], goal_m) < math.dist(start_m, goal_m)
    assert all(
        belief.state(belief.world_to_voxel(point)) == FREE for point in progress.path_m
    )
    assert sum(
        math.dist(left, right)
        for left, right in zip(progress.path_m, progress.path_m[1:], strict=False)
    ) >= 1.0


def test_component_progress_rejects_stationary_or_retreating_route() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    start_m = (0.125, 0.125, 0.125)
    goal_m = (8.125, 0.125, 0.125)
    # The source component has no received-free extension, so there is no
    # public-map movement that can count as progress toward the frontier.
    belief.set_state((0, 0, 0), FREE)

    progress = runner._public_component_progress_path_result(
        belief,
        start_m,
        goal_m,
        maximum_path_length_m=4.0,
        minimum_advance_m=0.5,
    )

    assert progress.path_m is None
    assert progress.status == "no_public_component_progress"


def test_observation_viewpoints_prefer_multiaxis_received_free_support() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    frontier_point = (10.125, 0.125, 0.125)
    weak_key = (34, 0, 0)
    robust_key = (34, 3, 0)

    # The nominal standoff is only supported along one received-free line.
    for key in ((33, 0, 0), weak_key, (35, 0, 0)):
        belief.set_state(key, FREE)
    # This alternative is farther from the nominal standoff but has free
    # evidence on both signs of all three axes.
    for key in (
        robust_key,
        (33, 3, 0),
        (35, 3, 0),
        (34, 2, 0),
        (34, 4, 0),
        (34, 3, -1),
        (34, 3, 1),
    ):
        belief.set_state(key, FREE)

    points = runner._known_free_observation_points(
        belief,
        frontier_point_m=frontier_point,
        outward_normal=(1.0, 0.0, 0.0),
        minimum_received_free_support_m=0.3,
    )

    assert points[0] == belief.voxel_center(robust_key)
    assert belief.voxel_center(weak_key) in points


def test_path_arc_helpers_truncate_without_removing_turns() -> None:
    runner = _load_runner_module()
    path = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0))

    midpoint = runner._point_at_path_arc_length_m(path, 1.5)
    prefix = runner._path_prefix_to_arc_length_m(path, 1.5)

    assert midpoint == pytest.approx((1.0, 0.5, 0.0))
    assert prefix == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0))
    assert runner._path_length_m(prefix) == pytest.approx(1.5)


def test_terminal_clearance_pullback_keeps_farthest_safe_public_prefix(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    path = ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    belief = SimpleNamespace(resolution_m=0.25)

    def admits_many(points, clearance_m):
        assert clearance_m == pytest.approx(runner.cf2x.REQUIRED_TERMINAL_CLEARANCE_M)
        return all(points[0][0] <= 4.0 for point in points)

    clearance_oracle = SimpleNamespace(admits_many=admits_many)

    def fake_routed_guard(
            scene_query,
            clearance_oracle,
            public_waypoints,
            agent_id,
            path_m,
            bounds_min,
            bounds_max,
            diagnostic_sink=None,
            *,
            allow_public_reroute=True,
            segment_cache=None,
        ):
            return runner.GuardedPath(legal=True, path_m=path_m)

    monkeypatch.setattr(runner.cf2x, "_routed_guard", fake_routed_guard)
    guarded, audit = runner._terminal_clearance_pullback_guarded_path(
        belief,
        object(),
        clearance_oracle,
        "uav0",
        path,
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
    )

    assert guarded is not None
    assert audit["admitted"] is True
    assert audit["pullback_route_length_m"] == pytest.approx(4.0)
    assert audit["pullback_retained_fraction"] == pytest.approx(0.8)


def test_terminal_clearance_pullback_fails_closed_without_safe_endpoint(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    path = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    belief = SimpleNamespace(resolution_m=0.25)
    clearance_oracle = SimpleNamespace(
        admits_many=lambda points, clearance_m: False,
    )

    guarded, audit = runner._terminal_clearance_pullback_guarded_path(
        belief,
        object(),
        clearance_oracle,
        "uav0",
        path,
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
    )

    assert guarded is None
    assert audit["admitted"] is False
    assert audit["reason"] == "no_terminal_clearance_pullback_guard_admitted"


def test_terminal_clearance_pullback_uses_public_voxel_chain_prefix(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    keys = tuple((index, 0, 0) for index in range(13))
    for key in keys:
        belief.set_state(key, FREE)
        for neighbor in runner.neighbors_26(key):
            if belief.state(neighbor) == FREE:
                continue
            if max(abs(neighbor[axis] - key[axis]) for axis in range(3)) == 1:
                belief.set_state(neighbor, FREE)
    path = (
        belief.voxel_center(keys[0]),
        belief.voxel_center(keys[4]),
        belief.voxel_center(keys[8]),
        belief.voxel_center(keys[12]),
    )

    def admits_many(points, clearance_m):
        return all(point[0] <= 1.75 for point in points)

    clearance_oracle = SimpleNamespace(admits_many=admits_many)

    def guarded_route(
        scene_query,
        clearance_oracle,
        public_waypoints,
        agent_id,
        path_m,
        bounds_min,
        bounds_max,
        diagnostic_sink=None,
        *,
        allow_public_reroute=True,
        segment_cache=None,
    ):
        endpoint = path_m[-1]
        return runner.GuardedPath(
            legal=endpoint[0] <= 1.75,
            path_m=path_m,
            reason=None if endpoint[0] <= 1.75 else "insufficient_continuous_collision_clearance",
        )

    monkeypatch.setattr(runner.cf2x, "_routed_guard", guarded_route)
    guarded, audit = runner._terminal_clearance_pullback_guarded_path(
        belief,
        object(),
        clearance_oracle,
        "uav0",
        path,
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
        voxel_keys=keys,
    )

    assert guarded is not None
    assert audit["admitted"] is True
    assert audit["pullback_source"] == "public_voxel_chain"
    assert audit["pullback_endpoint_m"][0] <= 1.75
    assert audit["pullback_route_length_m"] >= runner.MINIMUM_MEANINGFUL_EXPLORATION_PATH_M


def test_terminal_clearance_pullback_uses_exact_grid_rescue_when_public_prefix_fails(
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    path = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    belief = SimpleNamespace(resolution_m=0.25)
    clearance_oracle = SimpleNamespace(
        admits_many=lambda points, clearance_m: all(point[0] <= 4.0 for point in points)
    )

    def failing_public_prefix_guard(
        scene_query,
        clearance_oracle,
        public_waypoints,
        agent_id,
        path_m,
        bounds_min,
        bounds_max,
        diagnostic_sink=None,
        *,
        allow_public_reroute=True,
        segment_cache=None,
    ):
        return runner.GuardedPath(
            legal=False,
            path_m=path_m,
            reason="insufficient_continuous_collision_clearance",
        )

    def exact_grid_route(
        scene_query,
        clearance_oracle,
        agent_id,
        start,
        end,
        bounds_min,
        bounds_max,
        diagnostic_sink=None,
        segment_cache=None,
    ):
        assert math.dist(start, end) >= 2.0
        return (start, end)

    monkeypatch.setattr(runner.cf2x, "_routed_guard", failing_public_prefix_guard)
    monkeypatch.setattr(runner.cf2x, "_grid_route", exact_grid_route)
    monkeypatch.setattr(
        runner.cf2x,
        "_admit_trackable_path",
        lambda guarded, bounds_min, bounds_max: guarded,
    )

    guarded, audit = runner._terminal_clearance_pullback_guarded_path(
        belief,
        object(),
        clearance_oracle,
        "uav0",
        path,
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
    )

    assert guarded is not None
    assert audit["admitted"] is True
    assert audit["pullback_source"] == "exact_clearance_grid_route"
    assert audit["pullback_route_length_m"] >= 2.0


def test_supported_public_route_prefixes_provide_nonterminal_progress_targets() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    for x_index in range(11):
        belief.set_state((x_index, 0, 0), FREE)
    for x_index in range(3, 9):
        for key in (
            (x_index, -1, 0),
            (x_index, 1, 0),
            (x_index, 0, -1),
            (x_index, 0, 1),
        ):
            belief.set_state(key, FREE)
    path = (belief.voxel_center((0, 0, 0)), belief.voxel_center((10, 0, 0)))

    progress = runner._public_route_progress_points(
        belief,
        path,
        minimum_received_free_support_m=0.3,
    )

    assert progress
    assert all(point != path[-1] for point in progress)
    assert all(
        runner._received_free_support(
            belief,
            belief.world_to_voxel(point),
            radius_m=0.3,
        ).balanced_axis_count
        >= runner.PUBLIC_ROUTE_PROGRESS_MIN_BALANCED_AXES
        for point in progress
    )
    assert all(
        runner._polyline_prefix_distance_m(path, point)
        >= runner.PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M
        for point in progress
    )


def test_long_public_route_prefix_receives_full_cluster_gain_ranking() -> None:
    runner = _load_runner_module()

    long_gain = runner._public_route_progress_gain(
        cluster_gain=0.8,
        progress_length_m=3.5,
        route_length_m=5.0,
    )
    short_gain = runner._public_route_progress_gain(
        cluster_gain=0.8,
        progress_length_m=1.0,
        route_length_m=5.0,
    )

    assert long_gain == pytest.approx(0.8)
    assert short_gain == pytest.approx(0.8 * 0.25)

    with pytest.raises(ValueError, match="positive route evidence"):
        runner._public_route_progress_gain(
            cluster_gain=0.8,
            progress_length_m=1.0,
            route_length_m=0.0,
        )
def test_route_progress_retention_preserves_efficient_long_and_vertical_alternatives() -> None:
    runner = _load_runner_module()
    source = (0.125, 0.125, 0.125)
    rows = (
        (
            (2, 0, 0),
            runner._PublicFrontierViewpoint(
                information_gain=3.0,
                traversal_risk=0.05,
                position_m=(0.625, 0.125, 0.125),
                source_agent_index=0,
                route_lengths_m=(0.5,),
                route_paths_m=(((0.125, 0.125, 0.125), (0.625, 0.125, 0.125)),),
                viewpoint_kind="route_progress",
            ),
        ),
        (
            (10, 0, 0),
            runner._PublicFrontierViewpoint(
                information_gain=1.0,
                traversal_risk=0.05,
                position_m=(2.625, 0.125, 0.125),
                source_agent_index=0,
                route_lengths_m=(2.5,),
                route_paths_m=(((0.125, 0.125, 0.125), (2.625, 0.125, 0.125)),),
                viewpoint_kind="route_progress",
            ),
        ),
        (
            (4, 0, 5),
            runner._PublicFrontierViewpoint(
                information_gain=0.4,
                traversal_risk=0.05,
                position_m=(1.125, 0.125, 1.375),
                source_agent_index=0,
                route_lengths_m=(1.6,),
                route_paths_m=(((0.125, 0.125, 0.125), (1.125, 0.125, 1.375)),),
                viewpoint_kind="route_progress",
            ),
        ),
        (
            (4, 6, 0),
            runner._PublicFrontierViewpoint(
                information_gain=0.4,
                traversal_risk=0.05,
                position_m=(1.125, 1.625, 0.125),
                source_agent_index=0,
                route_lengths_m=(1.6,),
                route_paths_m=(((0.125, 0.125, 0.125), (1.125, 1.625, 0.125)),),
                viewpoint_kind="route_progress",
            ),
        ),
    )

    retained = runner._retain_route_progress_viewpoints(
        rows,
        source_position_m=source,
        source_agent_index=0,
        maximum_count=3,
        resolution_m=0.25,
    )

    assert [key for key, _row in retained] == [(2, 0, 0), (10, 0, 0), (4, 0, 5)]


def test_online_p07_emits_outcome_grounded_qd_diagnostics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "realised_descriptor_from_public_outcomes" in source
    assert "audit_realised_qd_richness" in source
    assert "audit_public_candidate_intent_richness" in source
    assert "audit_intent_realised_alignment" in source
    assert '"candidate_intent_audits"' in source
    assert '"quality_source": "public_sparse_range_outcomes"' in source
    assert "realised-QD selection gain until the paired planned-QD/no-QD matrix is run" in source


def test_online_p07_emits_per_decision_cf2x_calibration_evidence() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"execution_calibration": execution_calibration' in source

    runner = _load_runner_module()
    summary = runner._route_geometry_summary(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)))

    assert summary["route_classes"] == ["horizontal", "vertical", "turn"]
    assert summary["waypoint_segment_count"] == 2
    source = RUNNER.read_text(encoding="utf-8")
    assert '"execution_calibration": execution_calibration' in source
    assert '"arrival_tolerance_m": args.arrival_tolerance_m' in source


def test_metric_scores_no_hit_public_free_voxels_against_the_evaluator_grid() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    belief.integrate_ray(
        PublicRangeRayOutcome(
            observation_id="ray0",
            agent_id="team",
            timestamp_s=1.0,
            origin_m=(0.01, 0.01, 0.01),
            endpoint_m=(0.49, 0.01, 0.01),
            hit_occupied=False,
        )
    )
    component = np.ones((2, 1, 1), dtype=bool)

    sample = runner._metric_sample(
        timestamp_s=1.0,
        component=component,
        grid_origin=np.asarray((0.125, 0.125, 0.125)),
        resolution_m=0.25,
        denominator_volume_m3=2 * 0.25**3,
        team_belief=belief,
    )

    assert sample.explored_free_volume_m3 == pytest.approx(2 * 0.25**3)
    assert sample.predicted_free_volume_m3 == pytest.approx(2 * 0.25**3)
    assert sample.hallucinated_free_volume_m3 == 0.0


def test_metric_counts_only_evaluator_inconsistent_public_voxels_as_hallucinated() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    belief.integrate_ray(
        PublicRangeRayOutcome(
            observation_id="ray0",
            agent_id="team",
            timestamp_s=1.0,
            origin_m=(0.01, 0.01, 0.01),
            endpoint_m=(0.74, 0.01, 0.01),
            hit_occupied=False,
        )
    )
    component = np.asarray([True, True, False], dtype=bool).reshape((3, 1, 1))

    sample = runner._metric_sample(
        timestamp_s=1.0,
        component=component,
        grid_origin=np.asarray((0.125, 0.125, 0.125)),
        resolution_m=0.25,
        denominator_volume_m3=2 * 0.25**3,
        team_belief=belief,
    )

    assert sample.explored_free_volume_m3 == pytest.approx(2 * 0.25**3)
    assert sample.predicted_free_volume_m3 == pytest.approx(3 * 0.25**3)
    assert sample.hallucinated_free_volume_m3 == pytest.approx(0.25**3)


def test_metric_conserves_volume_when_public_and_evaluator_grids_are_half_voxel_shifted() -> None:
    runner = _load_runner_module()
    belief = SparseVoxelBelief("scene0", "team", 0.25)
    belief.set_state((0, 0, 0), FREE)
    belief.set_state((1, 0, 0), FREE)
    component = np.ones((4, 1, 1), dtype=bool)

    overlap = runner._evaluator_consistent_public_free(
        component=component,
        grid_origin=np.asarray((-0.25, 0.125, 0.125)),
        resolution_m=0.25,
        team_belief=belief,
    )

    assert overlap.grid_phase_offset_fraction == pytest.approx((0.5, 0.0, 0.0))
    assert overlap.touched_evaluator_voxel_count == 3
    assert overlap.consistent_volume_m3 == pytest.approx(2 * 0.25**3)
    assert overlap.inconsistent_volume_m3 == 0.0
    assert overlap.conservation_error_m3 == pytest.approx(0.0, abs=1.0e-12)


def test_online_p07_emits_wall_timing_for_throughput_root_cause_analysis() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '"runtime_performance"' in source
    assert '"decision_execution_wall_s"' in source
    assert '"wall_to_physics_ratio"' in source
    assert '"route_guard_unique_wall_s"' in source
    assert '"candidate_assignment_and_manifest_wall_s"' in source


def test_execution_status_rejects_outcome_backed_terminal_failure() -> None:
    runner = _load_runner_module()

    assert runner._classify_execution_status(
        terminal_outcome="budget_exhausted", failed_fragment_count=0
    ) == (runner.P07_EXECUTION_SMOKE_COMPLETE_STATUS, None)
    status, reason = runner._classify_execution_status(
        terminal_outcome="executed_terminal_safety_failure", failed_fragment_count=1
    )
    assert status == runner.P07_EXECUTION_SMOKE_FAILED_STATUS
    assert reason == (
        "terminal_outcome=executed_terminal_safety_failure;failed_fragment_count=1"
    )


def test_execution_status_does_not_call_budget_episode_complete_when_fragments_failed() -> None:
    runner = _load_runner_module()

    status, reason = runner._classify_execution_status(
        terminal_outcome="budget_exhausted", failed_fragment_count=1
    )

    assert status == runner.P07_EXECUTION_SMOKE_FAILED_STATUS
    assert reason == "terminal_outcome=budget_exhausted;failed_fragment_count=1"


def test_completed_decision_progress_is_atomic_non_trainable_recovery_evidence(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output = tmp_path / "episode.json"
    sample = ExplorationMetricSample(
        timestamp_s=2.0,
        explored_free_volume_m3=1.0,
        true_free_volume_m3=10.0,
        predicted_free_volume_m3=1.2,
        hallucinated_free_volume_m3=0.2,
    )

    runner._write_decision_progress(
        output,
        scene_id="scene0",
        strategy="frontier_3d",
        action_budget_s=40.0,
        elapsed_physics_s=2.0,
        decisions=[{"decision_id": "decision0", "reason_counts": {"admitted": 1}}],
        samples=[sample],
    )

    progress = json.loads(runner._progress_path(output).read_text(encoding="utf-8"))
    supplied_hash = progress.pop("progress_record_sha256")
    assert runner.canonical_sha256(progress) == supplied_hash
    assert progress["formal_result"] is False
    assert progress["trainable"] is False
    assert progress["decision_count"] == 1


def test_calibration_timeout_probe_is_train_only_and_never_emits_replay() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'parser.add_argument(\n        "--calibration-timeout-probe-s"' in source
    assert 'parser.add_argument(\n        "--record-purpose"' in source
    assert 'if args.record_purpose == "train_outcome":' in source
    assert '"calibration_only_timeout_probe": timeout_probe' in source
    assert "calibration-only timeout probes cannot enter QD history" in source


def test_qd_strategies_require_train_outcomes_and_fail_closed_candidate_variety() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "def _load_train_qd_history(" in source
    assert 'payload.get("selection_partition") != "train"' in source
    assert 'payload.get("split_manifest_sha256") != split_manifest_sha256' in source
    assert '"split_manifest_sha256": split_manifest_sha256' in source
    assert "MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION" in source
    assert "feasible train execution outcomes" in source
    assert "audit_pre_registered_qd_descriptor_families" in source
    assert '"descriptor_family_screen"' in source
    assert '"candidate_descriptor_features": candidate_descriptor_features.to_dict()' in source
    assert '"planned_qd"' in source
    assert '"realised_qd"' in source
    assert "QD_INTENT_FALLBACK_TO_PUBLIC_VALUE" in source
    assert "candidate_intent_audit.status != \"QD_CANDIDATE_INTENT_ADMITTED\"" in source


def test_qd_replay_calibration_is_train_only_and_not_a_ranked_p07_strategy() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "QD_CALIBRATION_INTENT_MODES" in source
    assert '"qd_calibration"' in source
    assert "QD replay calibration may only run on the train partition" in source
    assert "train_only_qd_replay_calibration" in source
    assert "not a P07 baseline or formal result" in source


def test_online_p07_binds_the_worker_to_the_frozen_p05_scene_split() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'parser.add_argument("--p05-artifact", required=True, type=Path)' in source
    assert "def _frozen_split_manifest_hash(" in source
    assert "P07 scene and requested partition disagree with P05 freeze" in source


def test_online_qd_archive_requires_safe_execution_outcomes() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "execution_feasible=qd_feasible" in source
    assert '"executed": execution_complete' in source
    assert '"candidate_manifest_sha256": selected.manifest_hash' in source
    assert '"execution_outcome_sha256": behavior_hash' in source


def test_qd_archive_uses_only_publicly_new_voxels_not_a_repeat_scan() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "public_free_keys_before = frozenset(team_belief.free_keys())" in source
    assert "public_free_voxel_transition(" in source
    assert "public_free_keys_before, team_belief.free_keys()" in source
    assert '"public_revised_free_voxel_count"' in source
    assert '"NO_NEW_PUBLIC_FREE_VOXELS"' in source
    assert '"executed": execution_complete' in source


def test_periodic_supervision_records_ten_second_boundaries_and_cumulative_auc() -> None:
    runner = _load_runner_module()
    start = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (3.0, 0.0, 1.0),
    )
    ledger = runner._PeriodicSupervisionLedger(
        interval_s=10.0,
        agent_ids=("uav0", "uav1", "uav2", "uav3"),
        start_positions_m=start,
        next_timestamp_s=10.0,
    )
    samples_10 = [
        ExplorationMetricSample(
            timestamp_s=0.0,
            explored_free_volume_m3=0.0,
            true_free_volume_m3=100.0,
            predicted_free_volume_m3=0.0,
        ),
        ExplorationMetricSample(
            timestamp_s=5.0,
            explored_free_volume_m3=20.0,
            true_free_volume_m3=100.0,
            predicted_free_volume_m3=20.0,
        ),
        ExplorationMetricSample(
            timestamp_s=10.0,
            explored_free_volume_m3=30.0,
            true_free_volume_m3=100.0,
            predicted_free_volume_m3=30.0,
        ),
    ]
    positions_10 = [(0.2, 0.0, 1.0), (1.5, 0.0, 1.0), (2.5, 0.0, 1.0), (3.5, 0.0, 1.0)]
    ledger.emit_until(
        elapsed_s=10.0,
        positions_m=positions_10,
        linear_speeds_mps=(0.1, 0.2, 0.3, 0.4),
        samples=samples_10,
        horizon_s=20.0,
        total_energy_j=3.0,
        collision_count=0,
        separation_violation_count=0,
        out_of_bounds_count=0,
        static_clearance_violation_count=0,
        executed_fragment_count=8,
        failed_fragment_count=0,
        decision_count=2,
    )
    assert [row["timestamp_s"] for row in ledger.samples] == [10.0]
    assert math.isclose(
        ledger.samples[-1]["explored_free_flight_volume_auc_time"], 0.0875
    )
    assert ledger.samples[-1]["moving_agent_count"] == 4

    samples_20 = samples_10 + [
        ExplorationMetricSample(
            timestamp_s=20.0,
            explored_free_volume_m3=50.0,
            true_free_volume_m3=100.0,
            predicted_free_volume_m3=50.0,
        )
    ]
    positions_20 = [(0.2, 0.0, 1.0), (1.8, 0.0, 1.0), (2.8, 0.0, 1.0), (3.8, 0.0, 1.0)]
    ledger.emit_until(
        elapsed_s=20.0,
        positions_m=positions_20,
        linear_speeds_mps=(0.05, 0.15, 0.25, 0.35),
        samples=samples_20,
        horizon_s=20.0,
        total_energy_j=6.0,
        collision_count=0,
        separation_violation_count=0,
        out_of_bounds_count=0,
        static_clearance_violation_count=0,
        executed_fragment_count=16,
        failed_fragment_count=0,
        decision_count=3,
    )
    assert [row["timestamp_s"] for row in ledger.samples] == [10.0, 20.0]
    assert math.isclose(
        ledger.samples[-1]["explored_free_flight_volume_auc_time"], 0.2875
    )
    assert ledger.samples[-1]["decision_count_so_far"] == 3


def test_periodic_supervision_prefers_physics_visualization_trace_timestamps() -> None:
    runner = _load_runner_module()
    start = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (3.0, 0.0, 1.0),
    )
    ledger = runner._PeriodicSupervisionLedger(
        interval_s=10.0,
        agent_ids=("uav0", "uav1", "uav2", "uav3"),
        start_positions_m=start,
        next_timestamp_s=10.0,
    )
    ledger.accumulate_trace(
        {
            "samples": [
                {
                    "physics_timestamp_s": 0.0,
                    "minimum_inter_agent_distance_m": 1.0,
                    "agents": [
                        {"agent_id": f"uav{index}", "position_m": list(start[index]), "linear_speed_mps": 0.0}
                        for index in range(4)
                    ],
                },
                {
                    "physics_timestamp_s": 10.0,
                    "minimum_inter_agent_distance_m": 1.0,
                    "agents": [
                        {
                            "agent_id": f"uav{index}",
                            "position_m": [float(index) + 0.5, 0.0, 1.5],
                            "linear_speed_mps": 0.4 + float(index) * 0.1,
                        }
                        for index in range(4)
                    ],
                },
            ]
        },
        start_s=0.0,
    )
    samples_10 = [
        ExplorationMetricSample(
            timestamp_s=0.0,
            explored_free_volume_m3=0.0,
            true_free_volume_m3=100.0,
            predicted_free_volume_m3=0.0,
        ),
        ExplorationMetricSample(
            timestamp_s=10.0,
            explored_free_volume_m3=30.0,
            true_free_volume_m3=100.0,
            predicted_free_volume_m3=30.0,
        ),
    ]
    ledger.emit_until(
        elapsed_s=10.0,
        positions_m=[(9.0, 9.0, 9.0) for _ in range(4)],
        linear_speeds_mps=(0.0, 0.0, 0.0, 0.0),
        samples=samples_10,
        horizon_s=20.0,
        total_energy_j=0.0,
        collision_count=0,
        separation_violation_count=0,
        out_of_bounds_count=0,
        static_clearance_violation_count=0,
        executed_fragment_count=0,
        failed_fragment_count=0,
        decision_count=1,
    )
    row = ledger.samples[-1]
    assert row["position_source"] == "physics_visualization_trace"
    assert row["agents"][0]["position_m"] == [0.5, 0.0, 1.5]
    assert math.isclose(row["agents"][3]["linear_speed_mps"], 0.7)
