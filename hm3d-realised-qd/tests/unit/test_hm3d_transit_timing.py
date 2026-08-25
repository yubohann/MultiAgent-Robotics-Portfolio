from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    _manifest_for_assignment,
    build_public_candidate_pool,
    identity_path_guard,
)
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import PublicMethodContext

ROOT = Path(__file__).resolve().parents[2]
CALIBRATOR = ROOT / "scripts" / "calibrate_hm3d_cf2x_transit_timing.py"
VECTORIZED_PROBE = ROOT / "scripts" / "run_hm3d_cf2x_vectorized_outcome_probe.py"


def _load_vectorized_probe():
    spec = importlib.util.spec_from_file_location("hm3d_vectorized_probe", VECTORIZED_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load vectorized CF2X outcome probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context() -> PublicMethodContext:
    return PublicMethodContext(
        context_id="timing-context",
        episode_id="timing-episode",
        decision_id="timing-decision",
        agent_features=(("uav0", (0.0,)),),
    )


def test_public_candidate_timing_uses_the_outcome_calibrated_upper_envelope() -> None:
    model = ConservativeTransitTimingModel("unit-timing", 1.0, 1.0, 0.5)
    state = PublicSearchState(
        context=_context(),
        agents=(PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 0),),
        frontiers=(PublicFrontier("frontier0", (4.0, 0.0, 1.0), 1.0, 0.0),),
        decision_start_s=0.0,
        decision_duration_s=6.0,
        transit_timing_model=model,
        observe_dwell_s=0.5,
    )
    manifest = build_public_candidate_pool(state, identity_path_guard, candidate_limit=1)[0]
    transit = next(
        row for row in manifest.fragments if row.type_signature.fragment_type == "transit"
    )
    assert transit.planned_end == pytest.approx(5.5)
    assert manifest.cost_hint == pytest.approx(5.5)
    assert "transit_timing_model" in state.to_dict()


def test_transit_timing_charges_terminal_and_intermediate_settling_separately() -> None:
    model = ConservativeTransitTimingModel("turn-aware", 1.0, 1.0, 0.5, 0.5)
    direct = model.estimate_seconds(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    turning = model.estimate_seconds(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))

    assert direct == pytest.approx(model.motion_seconds_for_distance(2.0) + 0.5)
    assert turning == pytest.approx(2.0 * model.motion_seconds_for_distance(1.0) + 1.0)
    assert turning > direct


def test_pass_through_timing_accepts_explicit_hold_with_terminal_settle() -> None:
    model = ConservativeTransitTimingModel(
        "pass-through-hold",
        1.0,
        1.0,
        0.4,
        intermediate_waypoint_requires_settle=False,
        continuous_waypoint_speed_mps=0.35,
    )
    position = (1.0, 2.0, 1.2)

    assert model.estimate_seconds((position, position)) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="zero-length"):
        model.continuous_polyline_seconds((position, position))


def test_uncovered_route_segments_reserve_observation_dwell() -> None:
    path = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (3.0, 0.0, 1.0),
    )
    base = ConservativeTransitTimingModel(
        "base",
        1.0,
        1.0,
        0.0,
        calibrated_max_segment_count=2,
        uncovered_segment_reserve_s=0.0,
    )
    guarded = ConservativeTransitTimingModel(
        "guarded",
        1.0,
        1.0,
        0.0,
        calibrated_max_segment_count=2,
        uncovered_segment_reserve_s=1.0,
    )
    assert guarded.estimate_seconds(path) == pytest.approx(base.estimate_seconds(path) + 1.0)

    frontier = PublicFrontier(
        "long-frontier",
        path[-1],
        1.0,
        0.0,
        access_paths_m=(("uav0", path),),
    )
    common = dict(
        context=_context(),
        agents=(PublicAgentPose("uav0", path[0], 1.0, 0),),
        frontiers=(frontier,),
        decision_start_s=0.0,
        decision_duration_s=7.0,
        observe_dwell_s=1.0,
    )
    admitted = build_public_candidate_pool(
        PublicSearchState(transit_timing_model=base, **common),
        identity_path_guard,
        candidate_limit=1,
    )[0]
    rejected = _manifest_for_assignment(
        PublicSearchState(transit_timing_model=guarded, **common),
        (0,),
        identity_path_guard,
        candidate_index=0,
    )
    assert admitted.feasible is True
    assert rejected.feasible is False
    assert "decision_window_exceeded" in rejected.admission_reasons


def test_vectorized_probe_preserves_an_explicit_multi_waypoint_command_path() -> None:
    probe = _load_vectorized_probe()
    first_path = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    paths = (
        first_path,
        ((2.0, 0.0, 0.0), (2.0, 0.0, 0.5)),
        ((4.0, 0.0, 0.0), (4.3, 0.0, 0.0)),
        ((6.0, 0.0, 0.0), (6.0, 0.4, 0.0)),
    )

    _, manifest = probe._manifest(
        cluster_id=0,
        transit_paths_m=paths,
        transit_end_s=6.0,
        horizon_s=8.0,
    )
    transit = next(
        fragment
        for fragment in manifest.fragments
        if fragment.agent_id == "uav0" and fragment.type_signature.fragment_type == "transit"
    )

    assert transit.path == first_path
    assert manifest.cost_hint == pytest.approx(3.2)


def _agent(
    *,
    agent_id: str,
    endpoint_x: float,
    completed_at_s: float | None,
    incomplete_reason: str | None = "transit_timeout",
    minimum_static_mesh_clearance_m: float = 0.40,
    static_clearance_contract_violation: bool = False,
    command_path_m: list[list[float]] | None = None,
) -> dict[str, object]:
    path = command_path_m or [[0.0, 0.0, 0.0], [endpoint_x, 0.0, 0.0]]
    return {
        "agent_id": agent_id,
        "command_path_m": path,
        "transit_completed": completed_at_s is not None,
        "transit_completed_at_s": completed_at_s,
        "transit_failure_reason": (None if completed_at_s is not None else incomplete_reason),
        "transit_collision": False,
        "transit_out_of_bounds": False,
        "observation_collision": False,
        "observation_out_of_bounds": False,
        "minimum_static_mesh_clearance_m": minimum_static_mesh_clearance_m,
        "static_clearance_contract_required_m": 0.30,
        "static_clearance_contract_violation": static_clearance_contract_violation,
    }


def _outcome_payload(
    agents: list[dict[str, object]],
    *,
    maximum_reference_speed_mps: float = 0.65,
    waypoint_settle_speed_mps: float = 0.3,
    tracking_clearance_margin_m: float = 0.15,
    execution_deadline_s: float = 20.0,
    calibration_only_timeout_probe: bool = False,
    intermediate_waypoint_requires_settle: bool = False,
) -> dict[str, object]:
    # These rows exercise the still-readable pre-v8 direct-route ABI.  They
    # deliberately omit turn settling; the v8 multi-waypoint ABI is covered by
    # _decision_calibration below with an explicit completed turn.
    return {
        "schema_version": "hm3d-p07-physx-execution-smoke-v1",
        "synthetic": False,
        "formal_result": False,
        "cf2x_usd_sha256": "f" * 64,
        "fleet_size": 4,
        "action_budget_s": 20.0,
        "execution_deadline_s": execution_deadline_s,
        "calibration_only_timeout_probe": calibration_only_timeout_probe,
        "arrival_tolerance_m": 0.1,
        "outcome_time_tolerance_s": 0.25,
        "physics_dt_s": 1.0 / 120.0,
        "engineering_debug": {
            "execution": {
                "execution_deadline_s": execution_deadline_s,
                "calibration_only_timeout_probe": calibration_only_timeout_probe,
                "agents": agents,
                "controller_tracking": {
                    "controller_id": "isaac-so3-feedback-v5",
                    "speed_profile": "time-parameterized-trapezoid-so3-guarded-v5",
                    "attitude_control": "force-rate-limited-yaw-so3-v2",
                    "maximum_reference_speed_mps": maximum_reference_speed_mps,
                    "maximum_reference_acceleration_mps2": 0.8,
                    "position_error_gain_per_s2": 4.0,
                    "velocity_error_gain_per_s": 7.0,
                    "maximum_feedback_acceleration_mps2": 3.0,
                    "waypoint_settle_speed_mps": waypoint_settle_speed_mps,
                    "maximum_tilt_rad": 0.25,
                    "maximum_yaw_rate_deg_s": 10.0,
                    "intermediate_waypoint_requires_settle": intermediate_waypoint_requires_settle,
                    "terminal_waypoint_requires_settle": True,
                    "tracking_clearance_margin_m": tracking_clearance_margin_m,
                },
                "static_trace_clearance": {
                    "method": "unit-exact-static-mesh-trace-v1",
                    "scope": "unit physics trace",
                    "static_clearance_contract_required_m": 0.30,
                    "static_clearance_contract_passed": all(
                        agent["static_clearance_contract_violation"] is False for agent in agents
                    ),
                },
            }
        },
    }


def test_calibrator_keeps_censored_timeout_and_rejects_it_under_the_shared_budget(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "timing.json"
    first.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5),
                    _agent(agent_id="uav1", endpoint_x=8.0, completed_at_s=13.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=10.0, completed_at_s=16.0),
                    _agent(agent_id="uav1", endpoint_x=12.0, completed_at_s=None),
                ]
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--decision-budget-s",
            "20",
            "--observation-dwell-s",
            "0.5",
            "--minimum-terminal-tracking-margin-s",
            "0.5",
            "--calibration-id",
            "unit-calibration",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CALIBRATION_PASS" in result.stdout
    calibration = json.loads(output.read_text(encoding="utf-8"))
    assert calibration["status"] == "CALIBRATION_PASS"
    assert len(calibration["completed_transit_checks"]) == 3
    timeout = calibration["censored_timeout_checks"]
    assert len(timeout) == 1
    assert timeout[0]["predicted_transit_seconds"] + 0.5 > 20.0
    assert calibration["identifiability"]["short_route_average_speed"] == "not_extrapolated"
    assert calibration["controller_tracking_profile"] == {
        "controller_id": "isaac-so3-feedback-v5",
        "speed_profile": "time-parameterized-trapezoid-so3-guarded-v5",
        "attitude_control": "force-rate-limited-yaw-so3-v2",
        "maximum_reference_speed_mps": 0.65,
        "maximum_reference_acceleration_mps2": 0.8,
        "position_error_gain_per_s2": 4.0,
        "velocity_error_gain_per_s": 7.0,
        "maximum_feedback_acceleration_mps2": 3.0,
        "waypoint_settle_speed_mps": 0.3,
        "maximum_tilt_rad": 0.25,
        "maximum_yaw_rate_deg_s": 10.0,
        "intermediate_waypoint_requires_settle": False,
        "terminal_waypoint_requires_settle": True,
        "tracking_clearance_margin_m": 0.15,
    }
    assert calibration["execution_profile"] == {
        "cf2x_usd_sha256": "f" * 64,
        "fleet_size": 4,
        "physics_dt_s": 1.0 / 120.0,
        "arrival_tolerance_m": 0.1,
        "outcome_time_tolerance_s": 0.25,
        "backend_id": "legacy-unrecorded-backend",
        "evidence_class": "legacy-unrecorded-evidence",
        "controller_tracking": calibration["controller_tracking_profile"],
    }
    assert calibration["static_trace_safety_contract"] == {
        "static_clearance_contract_required_m": 0.30,
        "source_trace_method": "unit-exact-static-mesh-trace-v1",
        "source_trace_scope": "unit physics trace",
    }
    assert calibration["minimum_terminal_tracking_margin_s"] == pytest.approx(0.5)
    assert calibration["outcome_time_tolerance_s"] == pytest.approx(0.25)
    assert calibration["time_model"]["terminal_tracking_margin_s"] >= 0.5


def test_calibrator_uses_a_real_short_probe_deadline_as_the_timeout_lower_bound(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "timeout-probe.json"
    third = tmp_path / "third.json"
    output = tmp_path / "timing.json"
    first.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5),
                    _agent(agent_id="uav1", endpoint_x=8.0, completed_at_s=13.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    timed_out = _agent(agent_id="uav1", endpoint_x=12.0, completed_at_s=None)
    timed_out["transit_attempted"] = True
    timed_out["transit_attempt_actual_end_s"] = 10.0
    second.write_text(
        json.dumps(
            _outcome_payload(
                [
                    timed_out,
                ],
                execution_deadline_s=10.0,
                calibration_only_timeout_probe=True,
            )
        ),
        encoding="utf-8",
    )
    third.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=10.0, completed_at_s=16.0),
                    _agent(agent_id="uav1", endpoint_x=11.0, completed_at_s=18.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--input",
            str(third),
            "--output",
            str(output),
            "--decision-budget-s",
            "21",
            "--observation-dwell-s",
            "0.5",
            "--minimum-terminal-tracking-margin-s",
            "0.5",
            "--calibration-id",
            "unit-short-deadline",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CALIBRATION_PASS" in result.stdout
    calibration = json.loads(output.read_text(encoding="utf-8"))
    timeout = calibration["censored_timeout_checks"]
    assert timeout[0]["source_decision_budget_s"] == pytest.approx(20.0)
    assert timeout[0]["source_execution_deadline_s"] == pytest.approx(10.0)
    assert timeout[0]["lower_bound_transit_seconds"] == pytest.approx(10.0)
    assert timeout[0]["predicted_transit_seconds"] + 0.5 > 10.0
    assert timeout[0]["calibration_only_timeout_probe"] is True


def test_calibrator_reports_safe_false_negatives_without_rejecting_the_model(
    tmp_path: Path,
) -> None:
    completed = tmp_path / "completed.json"
    timeout_path = tmp_path / "timeout.json"
    output = tmp_path / "timing.json"
    completed.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=0.1, completed_at_s=1.2),
                    _agent(agent_id="uav1", endpoint_x=0.2, completed_at_s=1.5),
                    _agent(agent_id="uav2", endpoint_x=1.0, completed_at_s=3.0),
                    _agent(agent_id="uav3", endpoint_x=2.5, completed_at_s=5.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    timeout = _agent(agent_id="uav0", endpoint_x=3.0, completed_at_s=None)
    timeout["transit_attempted"] = True
    timeout["transit_attempt_actual_end_s"] = 4.0
    timeout_path.write_text(
        json.dumps(
            _outcome_payload(
                [timeout],
                execution_deadline_s=4.0,
                calibration_only_timeout_probe=True,
            )
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(completed),
            "--input",
            str(timeout_path),
            "--output",
            str(output),
            "--decision-budget-s",
            "5",
            "--observation-dwell-s",
            "0.5",
            "--minimum-terminal-tracking-margin-s",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    calibration = json.loads(output.read_text(encoding="utf-8"))
    assert len(calibration["target_budget_usable_completed_checks"]) == 3
    rejected = calibration["conservatively_rejected_completed_checks"]
    assert len(rejected) == 1
    assert rejected[0]["agent_id"] == "uav3"
    assert calibration["conservative_false_negative_audit"] == {
        "completed_path_count": 4,
        "target_budget_usable_completed_path_count": 3,
        "conservatively_rejected_completed_path_count": 1,
        "interpretation": (
            "A completed path may be rejected under the target decision budget when the "
            "shared upper envelope is slower than that individual trace. This is a safe "
            "false negative and is reported as candidate-availability cost."
        ),
    }


def test_calibrator_rejects_mixed_controller_tracking_profiles(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "timing.json"
    first.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5),
                    _agent(agent_id="uav1", endpoint_x=8.0, completed_at_s=13.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=10.0, completed_at_s=16.0),
                    _agent(agent_id="uav1", endpoint_x=12.0, completed_at_s=None),
                ],
                maximum_reference_speed_mps=0.8,
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--decision-budget-s",
            "20",
            "--observation-dwell-s",
            "0.5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "different execution profiles" in result.stderr
    assert not output.exists()


def test_calibrator_rejects_the_old_intermediate_waypoint_settling_abi(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "timing.json"
    first.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5),
                    _agent(agent_id="uav1", endpoint_x=8.0, completed_at_s=13.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=10.0, completed_at_s=16.0),
                    _agent(agent_id="uav1", endpoint_x=12.0, completed_at_s=None),
                ],
                intermediate_waypoint_requires_settle=True,
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--decision-budget-s",
            "20",
            "--observation-dwell-s",
            "0.5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unsupported waypoint-settling ABI" in result.stderr
    assert not output.exists()


def _route_geometry(path: list[list[float]], classes: list[str]) -> dict[str, object]:
    return {
        "command_path_length_m": sum(
            sum((right[axis] - left[axis]) ** 2 for axis in range(3)) ** 0.5
            for left, right in zip(path[:-1], path[1:], strict=True)
        ),
        "route_classes": classes,
    }


def _decision_calibration(
    decision_id: str,
    agents: list[dict[str, object]],
    *,
    execution_deadline_s: float = 20.0,
    timeout_probe: bool = False,
) -> dict[str, object]:
    for agent in agents:
        path = agent["command_path_m"]
        assert isinstance(path, list)
        if path == [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]]:
            classes = ["vertical"]
        elif len(path) == 3:
            classes = ["horizontal", "turn"]
        elif path[0] == path[-1]:
            classes = ["stationary"]
        else:
            classes = ["horizontal"]
        agent["route_geometry"] = _route_geometry(path, classes)
    summary = {
        "schema_version": "hm3d-cf2x-decision-execution-calibration-v1",
        "decision_id": decision_id,
        "backend_id": "isaac-physx-cf2x-waypoint-executor-v1",
        "evidence_class": "real_isaac_physx_cf2x",
        "token_authorization_duration_s": 20.0,
        "execution_deadline_s": execution_deadline_s,
        "calibration_only_timeout_probe": timeout_probe,
        "controller_tracking": {
            "controller_id": "isaac-so3-feedback-v6",
            "speed_profile": "time-parameterized-trapezoid-so3-guarded-v8",
            "attitude_control": "force-rate-limited-yaw-so3-v2",
            "maximum_reference_speed_mps": 0.65,
            "maximum_reference_acceleration_mps2": 0.8,
            "position_error_gain_per_s2": 4.0,
            "velocity_error_gain_per_s": 7.0,
            "maximum_feedback_acceleration_mps2": 3.0,
            "effective_control_rate_hz": 120.0,
            "waypoint_settle_speed_mps": 0.3,
            "waypoint_settle_position_tolerance_m": 0.03,
            "maximum_tilt_rad": 0.25,
            "maximum_yaw_rate_deg_s": 10.0,
            "waypoint_pass_through_speed_mps": 0.35,
            "intermediate_waypoint_requires_settle": False,
            "terminal_waypoint_requires_settle": True,
            "tracking_clearance_margin_m": 0.15,
            "rotor_allocation_id": "cf2x-usd-m1-m4-0p031m-reaction-yaw-v2",
            "rotor_order": ["m1_prop", "m2_prop", "m3_prop", "m4_prop"],
            "rotor_xy_lever_arm_m": 0.031,
            "yaw_torque_to_thrust_m": 0.006,
            "rotor_yaw_reaction_signs": [-1, 1, -1, 1],
            "actuator_initialization_id": "hover-equilibrium-rps-v1",
            "initial_rotor_rps": 263.3438816452738,
            "thrust_constant_n_per_rps2": 1.0e-6,
            "tau_inc_range_s": [0.04, 0.06],
            "tau_dec_range_s": [0.02, 0.03],
        },
        "static_trace_clearance": {
            "method": "unit-exact-static-mesh-trace-v1",
            "scope": "unit physics trace",
            "static_clearance_contract_required_m": 0.30,
            "static_clearance_contract_passed": True,
        },
        "agents": agents,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _multi_decision_payload(decisions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "hm3d-p07-exploration-execution-v1",
        "synthetic": False,
        "formal_result": False,
        "cf2x_usd_sha256": "f" * 64,
        "fleet_size": 4,
        "action_budget_s": 40.0,
        "arrival_tolerance_m": 0.1,
        "physics_dt_s": 1.0 / 120.0,
        "outcome_time_tolerance_s": 0.25,
        "decisions": decisions,
    }


def test_calibrator_reads_multi_decision_routes_and_requires_3d_class_coverage(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "timing.json"
    horizontal = _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5)
    vertical = _agent(
        agent_id="uav1",
        endpoint_x=0.0,
        completed_at_s=10.0,
        command_path_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
    )
    turning = _agent(
        agent_id="uav2",
        endpoint_x=0.0,
        completed_at_s=8.0,
        command_path_m=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]],
    )
    stationary = _agent(
        agent_id="uav3",
        endpoint_x=0.0,
        completed_at_s=1.0,
        command_path_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    first.write_text(
        json.dumps(
            _multi_decision_payload(
                [
                    {
                        "decision_id": "decision0",
                        "execution_calibration": _decision_calibration(
                            "decision0", [horizontal, vertical]
                        ),
                    },
                    {
                        "decision_id": "decision1",
                        "execution_calibration": _decision_calibration(
                            "decision1", [turning, stationary]
                        ),
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    timeout = _agent(agent_id="uav0", endpoint_x=12.0, completed_at_s=None)
    timeout["transit_attempted"] = True
    timeout["transit_attempt_actual_end_s"] = 10.0
    second.write_text(
        json.dumps(
            _multi_decision_payload(
                [
                    {
                        "decision_id": "decision2",
                        "execution_calibration": _decision_calibration(
                            "decision2",
                            [timeout],
                            execution_deadline_s=10.0,
                            timeout_probe=True,
                        ),
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--decision-budget-s",
            "25",
            "--observation-dwell-s",
            "0.5",
            "--minimum-terminal-tracking-margin-s",
            "0.5",
            "--require-route-class",
            "horizontal",
            "--require-route-class",
            "vertical",
            "--require-route-class",
            "turn",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CALIBRATION_PASS" in result.stdout
    calibration = json.loads(output.read_text(encoding="utf-8"))
    assert calibration["route_class_coverage"] == {
        "observed": ["horizontal", "turn", "vertical"],
        "target_budget_usable": ["horizontal", "turn", "vertical"],
        "required": ["horizontal", "turn", "vertical"],
    }
    assert {row["source_record_id"] for row in calibration["completed_transit_checks"]} == {
        "decision0",
        "decision1",
    }
    turning_row = next(
        row for row in calibration["completed_transit_checks"] if row["agent_id"] == "uav2"
    )
    model = ConservativeTransitTimingModel.from_dict(calibration["time_model"])
    turning_path = tuple(tuple(point) for point in turning_row["command_path_m"])
    straight_path = ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    assert turning_row["predicted_transit_seconds"] == pytest.approx(
        model.estimate_seconds(turning_path)
    )
    assert turning_row["predicted_transit_seconds"] > model.estimate_seconds(straight_path)
    assert calibration["excluded_stationary_transits"][0]["agent_id"] == "uav3"


def test_calibrator_rejects_non_timeout_as_a_speed_censor(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "timing.json"
    first.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5),
                    _agent(agent_id="uav1", endpoint_x=8.0, completed_at_s=13.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=10.0, completed_at_s=16.0),
                    _agent(
                        agent_id="uav1",
                        endpoint_x=12.0,
                        completed_at_s=None,
                        incomplete_reason="collision",
                    ),
                ]
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--decision-budget-s",
            "20",
            "--observation-dwell-s",
            "0.5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "explicit transit_timeout reason" in result.stderr
    assert not output.exists()


def test_calibrator_rejects_a_completed_trace_that_breaches_static_clearance(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "timing.json"
    first.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(agent_id="uav0", endpoint_x=4.0, completed_at_s=7.5),
                    _agent(agent_id="uav1", endpoint_x=8.0, completed_at_s=13.0),
                ]
            )
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            _outcome_payload(
                [
                    _agent(
                        agent_id="uav0",
                        endpoint_x=10.0,
                        completed_at_s=16.0,
                        minimum_static_mesh_clearance_m=0.25,
                        static_clearance_contract_violation=True,
                    ),
                    _agent(agent_id="uav1", endpoint_x=12.0, completed_at_s=None),
                ]
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--decision-budget-s",
            "20",
            "--observation-dwell-s",
            "0.5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "failed static clearance contract" in result.stderr
    assert not output.exists()
