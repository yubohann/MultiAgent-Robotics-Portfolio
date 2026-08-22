from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash, file_hash, write_json
from aerocity_bench.cf2x_fleet_preflight_contract import (
    COMPLETE_CALIBRATION_PURPOSE,
    FLEET_PRECHECK_SCOPE,
    FLEET_PRIVATE_SCOPE,
    SHORT_PREFLIGHT_PURPOSE,
    SharedWorldStepLedger,
    _validate_planning_receipt_consistency,
    _validate_planning_timing,
    altitude_stability_metrics,
    assert_action_roster_complete,
    assert_canonical_fleet_receipts,
    assert_fleet_confirmation_bindings,
    assert_fleet_execution_bindings,
    assert_public_report_has_no_private_truth,
    candidate_shared_hold_assessment,
    public_fleet_members,
    public_policy_progress_status,
    validate_complete_calibration_summary,
    validate_fleet_preflight_reports,
    validate_native_run_purpose,
)
from aerocity_bench.errors import ValidationError


def _episode() -> dict[str, object]:
    return {
        "starts": [
            {"drone_id": f"drone-{index:03d}", "position": [index, 0.0, 2.0], "yaw_deg": 0.0}
            for index in range(4)
        ]
    }


def test_fleet_preflight_progress_replaces_in_progress_with_terminal_status(
    tmp_path: Path,
) -> None:
    """Completed evidence must never look like a still-running Isaac process."""
    tool = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "cf2x_l1_fleet_preflight.py"))
    output = tmp_path / "fleet.public.json"
    write_progress = tool["_write_progress"]
    write_progress(output, "beginning_shared_physx_candidate_run")
    write_progress(
        output,
        "completed_candidate_run",
        status="COMPLETED",
        safe_completion=True,
    )
    progress = json.loads((tmp_path / "fleet.public.progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "COMPLETED"
    assert progress["stage"] == "completed_candidate_run"
    assert progress["safe_completion"] is True
    with pytest.raises(ValueError, match="unsupported fleet preflight progress status"):
        write_progress(output, "invalid", status="UNKNOWN")


def test_fleet_preflight_exposes_only_frozen_public_policy_choices() -> None:
    tool = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "cf2x_l1_fleet_preflight.py"))
    parser = tool["_parser"]()
    required = {action.dest for action in parser._actions if getattr(action, "required", False)}
    assert {
        "layout_root",
        "release_config",
        "output",
        "cf2x_usd",
    }.issubset(required)
    method_action = next(action for action in parser._actions if action.dest == "method")
    assert tuple(method_action.choices) == (
        "sweep-3d",
        "atlas-surface-inspector",
        "atlas-region-greedy",
    )
    purpose_action = next(action for action in parser._actions if action.dest == "run_purpose")
    assert tuple(purpose_action.choices) == (
        SHORT_PREFLIGHT_PURPOSE,
        COMPLETE_CALIBRATION_PURPOSE,
    )


def test_external_substage_summary_preserves_empty_stages_and_counts_samples() -> None:
    tool = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "cf2x_l1_fleet_preflight.py"))
    summary = tool["_external_process_substage_summary"](
        [
            {
                "action_sequence": 0,
                "bridge_act_wall_clock_s": 0.02,
                "bridge_act_process_cpu_s": 0.01,
                "fleet_arbitration_wall_clock_s": 0.003,
                "fleet_arbitration_process_cpu_s": 0.002,
                "unattributed_wall_clock_s": 0.001,
                "unattributed_process_cpu_s": 0.001,
            },
            {
                "action_sequence": 1,
                "bridge_act_wall_clock_s": None,
                "bridge_act_process_cpu_s": None,
                "fleet_arbitration_wall_clock_s": None,
                "fleet_arbitration_process_cpu_s": None,
                "unattributed_wall_clock_s": 0.004,
                "unattributed_process_cpu_s": 0.003,
            },
        ]
    )
    assert summary["bridge_act_wall_clock"]["call_count"] == 1
    assert summary["fleet_arbitration_wall_clock"]["max_s"] == 0.003
    assert summary["unattributed_wall_clock"]["call_count"] == 2


def test_v2_external_timing_requires_complete_public_substage_summaries() -> None:
    one_sample = {"call_count": 1, "p50_s": 0.001, "p95_s": 0.001, "p99_s": 0.001, "max_s": 0.001}
    timing = {
        "schema": "org.aerocity.bench.fleet-preflight-timing.v2",
        "control_tick_count": 1,
        "planning_deadline_s": 0.15,
        "deadline_miss_tick_count": 0,
        "policy_call": {"p50_s": 0.002, "p95_s": 0.002, "p99_s": 0.002, "max_s": 0.002},
        "public_observation_build": {
            "p50_s": 0.001,
            "p95_s": 0.001,
            "p99_s": 0.001,
            "max_s": 0.001,
        },
        "external_process_substages": {
            "bridge_act_wall_clock": one_sample,
            "bridge_act_process_cpu": one_sample,
            "fleet_arbitration_wall_clock": one_sample,
            "fleet_arbitration_process_cpu": one_sample,
            "unattributed_wall_clock": one_sample,
            "unattributed_process_cpu": one_sample,
        },
    }
    _validate_planning_timing(
        timing, expected_control_ticks=1, execution_mode="external-process-policy"
    )
    timing["external_process_substages"] = None
    with pytest.raises(ValidationError, match="lacks timing substages"):
        _validate_planning_timing(
            timing, expected_control_ticks=1, execution_mode="external-process-policy"
        )


def test_v3_external_timing_replaces_ambiguous_process_cpu_with_bridge_stages() -> None:
    one_sample = {"call_count": 1, "p50_s": 0.001, "p95_s": 0.001, "p99_s": 0.001, "max_s": 0.001}
    timing = {
        "schema": "org.aerocity.bench.fleet-preflight-timing.v3",
        "control_tick_count": 1,
        "planning_deadline_s": 0.15,
        "deadline_miss_tick_count": 0,
        "policy_call": {"p50_s": 0.002, "p95_s": 0.002, "p99_s": 0.002, "max_s": 0.002},
        "public_observation_build": {
            "p50_s": 0.001,
            "p95_s": 0.001,
            "p99_s": 0.001,
            "max_s": 0.001,
        },
        "external_process_substages": {
            field: one_sample
            for field in (
                "bridge_act_wall_clock",
                "projection_wall_clock",
                "request_public_audit_wall_clock",
                "request_json_serialize_wall_clock",
                "request_size_check_wall_clock",
                "request_write_flush_wall_clock",
                "response_wait_wall_clock",
                "response_json_decode_wall_clock",
                "response_validate_wall_clock",
                "action_validation_conversion_wall_clock",
                "bridge_internal_unattributed_wall_clock",
                "fleet_arbitration_wall_clock",
                "unattributed_wall_clock",
            )
        },
    }
    _validate_planning_timing(
        timing, expected_control_ticks=1, execution_mode="external-process-policy"
    )


def _v4_timing() -> dict[str, object]:
    return {
        "schema": "org.aerocity.bench.fleet-preflight-timing.v4",
        "control_tick_count": 10,
        "control_period_s": 0.2,
        "planner_invocation_count": 2,
        "held_action_tick_count": 8,
        "planning_cadence": {
            "schema": "org.aerocity.bench.planning-cadence.v1",
            "mode": "fixed-rate-with-public-events",
            "period_s": 1.0,
            "event_triggers": [
                "anonymous_confirmation",
                "safety_intervention",
                "fleet_roster_change",
                "return_reserve_entry",
            ],
            "held_action_rebinding": "latest-public-observation",
            "retransmit_messages_on_hold": False,
        },
        "planning_trigger_counts": {"fixed_period": 2, "initial": 1},
        "planning_deadline_s": 0.15,
        "deadline_miss_tick_count": 0,
        "policy_call": {
            "call_count": 2,
            "p50_s": 0.01,
            "p95_s": 0.01,
            "p99_s": 0.01,
            "max_s": 0.01,
        },
        "public_observation_build": {
            "p50_s": 0.001,
            "p95_s": 0.001,
            "p99_s": 0.001,
            "max_s": 0.001,
        },
        "external_process_substages": None,
    }


def test_v4_timing_counts_only_real_planner_invocations() -> None:
    timing = _v4_timing()
    _validate_planning_timing(
        timing, expected_control_ticks=10, execution_mode="public-policy"
    )
    timing["planner_invocation_count"] = 10
    timing["held_action_tick_count"] = 0
    with pytest.raises(ValidationError, match="sample count"):
        _validate_planning_timing(
            timing, expected_control_ticks=10, execution_mode="public-policy"
        )


def test_v4_timing_rejects_forged_fixed_period_trigger_count() -> None:
    timing = _v4_timing()
    timing["planning_trigger_counts"] = {"fixed_period": 1, "initial": 1}
    with pytest.raises(ValidationError, match="fixed trigger counts"):
        _validate_planning_timing(
            timing, expected_control_ticks=10, execution_mode="public-policy"
        )


def test_native_run_purpose_separates_short_and_complete_calibration_runs() -> None:
    validate_native_run_purpose(
        purpose=SHORT_PREFLIGHT_PURPOSE,
        execution_mode="shared-hold",
        requested_sim_time_s=299.8,
        frozen_episode_duration_s=300.0,
    )
    validate_native_run_purpose(
        purpose=COMPLETE_CALIBRATION_PURPOSE,
        execution_mode="public-policy",
        requested_sim_time_s=300.0,
        frozen_episode_duration_s=300.0,
    )
    with pytest.raises(ValidationError, match="remain below"):
        validate_native_run_purpose(
            purpose=SHORT_PREFLIGHT_PURPOSE,
            execution_mode="public-policy",
            requested_sim_time_s=300.0,
            frozen_episode_duration_s=300.0,
        )
    with pytest.raises(ValidationError, match="requires a public policy"):
        validate_native_run_purpose(
            purpose=COMPLETE_CALIBRATION_PURPOSE,
            execution_mode="shared-hold",
            requested_sim_time_s=300.0,
            frozen_episode_duration_s=300.0,
        )
    with pytest.raises(ValidationError, match="frozen 300-second"):
        validate_native_run_purpose(
            purpose=COMPLETE_CALIBRATION_PURPOSE,
            execution_mode="public-policy",
            requested_sim_time_s=240.0,
            frozen_episode_duration_s=240.0,
        )
    with pytest.raises(ValidationError, match="exactly"):
        validate_native_run_purpose(
            purpose=COMPLETE_CALIBRATION_PURPOSE,
            execution_mode="public-policy",
            requested_sim_time_s=299.8,
            frozen_episode_duration_s=300.0,
        )


def test_complete_calibration_public_inputs_require_full_g2i_sector() -> None:
    tool = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "cf2x_l1_fleet_preflight.py"))
    validate_inputs = tool["_validate_complete_calibration_public_inputs"]
    args = tool["_parser"]().parse_args(
        [
            "--layout-root",
            "layout",
            "--release-config",
            "release.json",
            "--output",
            "output.json",
            "--cf2x-usd",
            "cf2x.usd",
            "--run-purpose",
            COMPLETE_CALIBRATION_PURPOSE,
            "--execution-mode",
            "public-policy",
        ]
    )
    task = {"task_track": "G2-I", "inspection_atlas": {"atlas_hash": "a" * 64}}
    public_episode = {
        "mission_sector_hash": "b" * 64,
        "mission_sector": {"sector_hash": "b" * 64},
    }
    validate_inputs(args, task, public_episode)
    with pytest.raises(ValueError, match="full public inspection atlas"):
        validate_inputs(args, {"task_track": "G2-I"}, public_episode)
    with pytest.raises(ValueError, match="frozen public mission sector"):
        validate_inputs(args, task, {})


def _complete_progress(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "purpose": COMPLETE_CALIBRATION_PURPOSE,
        "observe_action_count": 1,
        "confirmation_receipt_count": 0,
        "return_action_count": 4,
        "all_returned_home": True,
        "episode_budget_completed": True,
        "safe_completion": True,
        "deadline_miss_tick_count": 0,
    }
    values.update(overrides)
    return values


def test_zero_confirmation_is_a_closed_scientific_calibration_outcome() -> None:
    assert public_policy_progress_status(**_complete_progress()) == ("CALIBRATION_EPISODE_CLOSED")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observe_action_count", 0),
        ("return_action_count", 0),
        ("all_returned_home", False),
        ("episode_budget_completed", False),
        ("safe_completion", False),
        ("deadline_miss_tick_count", 1),
    ),
)
def test_complete_progress_fails_closed_when_any_execution_condition_is_missing(
    field: str, value: object
) -> None:
    assert public_policy_progress_status(**_complete_progress(**{field: value})) == (
        "CALIBRATION_EPISODE_INCOMPLETE"
    )


def test_complete_progress_rejects_non_boolean_flags() -> None:
    with pytest.raises(ValidationError, match="flags must be booleans"):
        public_policy_progress_status(**_complete_progress(safe_completion=1))


_COMPLETE_BINDING_HASHES = (
    "layout_hash",
    "stage_sha256",
    "cityspec_sha256",
    "task_spec_sha256",
    "task_spec_hash",
    "public_episode_sha256",
    "mission_sector_hash",
    "execution_contract_hash",
    "release_config_sha256",
    "cf2x_usd_sha256",
    "cf2x_schema_sha256",
    "dynamics_spec_hash",
    "controller_spec_hash",
    "baseline_source_sha256",
    "geometry_source_sha256",
    "atlas_hash",
)


def _complete_summary() -> dict[str, object]:
    drone_ids = {f"drone-{index:03d}" for index in range(4)}
    private_execution = {
        "control_ticks": 1500,
        "control_period_s": 0.2,
        "shared_physx_step_count": 15000,
        "simulated_time_s": 300.0,
    }
    public_execution = {
        **private_execution,
        "failure_record_count": 0,
    }
    private_final = {
        "safe_completion": True,
        "collision_detected": False,
        "out_of_bounds_detected": False,
        "all_returned_home": True,
        "returned_home_by_drone": {drone_id: True for drone_id in drone_ids},
    }
    public_final = {
        key: private_final[key]
        for key in (
            "safe_completion",
            "collision_detected",
            "out_of_bounds_detected",
            "all_returned_home",
        )
    }
    bindings = {
        **{field: "a" * 64 for field in _COMPLETE_BINDING_HASHES},
        "layout_id": "calibration-ancestor-04",
        "episode_id": "episode-calibration-04",
        "task_track": "G2-I",
        "inspection_prior_level": "full-cells",
    }
    return {
        "execution_mode": "public-policy",
        "policy_progress": {
            "status": "CALIBRATION_EPISODE_CLOSED",
            "observe_action_count": 1,
            "confirmation_receipt_count": 0,
            "return_action_count": 4,
            "all_returned_home": True,
            "episode_budget_completed": True,
        },
        "planning_timing": {"deadline_miss_tick_count": 0},
        "private_final": private_final,
        "public_final": public_final,
        "private_execution": private_execution,
        "public_execution": public_execution,
        "input_bindings": bindings,
        "public_input_bindings": deepcopy(bindings),
        "method": "atlas-region-greedy",
        "public_method": "atlas-region-greedy",
        "expected_drone_ids": drone_ids,
        "failure_records": [],
    }


def test_complete_calibration_summary_independently_closes_300_second_replay() -> None:
    validate_complete_calibration_summary(**_complete_summary())


@pytest.mark.parametrize("missing_hash", _COMPLETE_BINDING_HASHES)
def test_complete_calibration_summary_rejects_every_missing_hash(
    missing_hash: str,
) -> None:
    summary = _complete_summary()
    private_bindings = summary["input_bindings"]
    public_bindings = summary["public_input_bindings"]
    assert isinstance(private_bindings, dict) and isinstance(public_bindings, dict)
    private_bindings.pop(missing_hash)
    public_bindings.pop(missing_hash)
    with pytest.raises(ValidationError, match="immutable input hashes"):
        validate_complete_calibration_summary(**summary)


def test_complete_calibration_summary_rejects_truncated_or_unsafe_evidence() -> None:
    truncated = _complete_summary()
    truncated["private_execution"]["control_ticks"] = 1499  # type: ignore[index]
    truncated["public_execution"]["control_ticks"] = 1499  # type: ignore[index]
    with pytest.raises(ValidationError, match="tick timing"):
        validate_complete_calibration_summary(**truncated)

    unsafe = _complete_summary()
    unsafe["private_final"]["safe_completion"] = False  # type: ignore[index]
    unsafe["public_final"]["safe_completion"] = False  # type: ignore[index]
    with pytest.raises(ValidationError, match="safety state"):
        validate_complete_calibration_summary(**unsafe)

    failed = _complete_summary()
    failed["failure_records"] = [{"category": "collision"}]
    with pytest.raises(ValidationError, match="failure records"):
        validate_complete_calibration_summary(**failed)


def test_measurement_path_retains_a_receipt_complete_planner_timeout() -> None:
    """A full failed replay remains evidence, but never closes the run gate."""

    timed_out = _complete_summary()
    timed_out["policy_progress"]["status"] = "CALIBRATION_EPISODE_INCOMPLETE"  # type: ignore[index]
    timed_out["planning_timing"]["deadline_miss_tick_count"] = 1  # type: ignore[index]

    with pytest.raises(ValidationError, match="did not close"):
        validate_complete_calibration_summary(**timed_out)
    validate_complete_calibration_summary(**timed_out, allow_execution_failure=True)


def test_candidate_vertical_guard_includes_contract_clearance_margin() -> None:
    tool = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "cf2x_l1_fleet_preflight.py"))
    minimum, maximum = tool["_effective_vertical_safe_bounds"](
        {"flight_bounds": {"minimum": [-1.0, -1.0, 1.0], "maximum": [1.0, 1.0, 10.0]}},
        {"radius_m": 0.32, "minimum_clearance_m": 0.75},
    )
    assert minimum == pytest.approx(2.07)
    assert maximum == pytest.approx(8.93)


def _binding(drone_id: str, sequence: int) -> dict[str, object]:
    observation = {
        "schema": "org.aerocity.bench.observation-packet.v2",
        "episode_id": "fleet-test",
        "observation_id": f"observation-{sequence}-{drone_id}",
        "drone_id": drone_id,
        "sequence": sequence,
        "timestamp_s": float(sequence),
    }
    action = {
        "schema": "org.aerocity.bench.action-packet.v1",
        "episode_id": "fleet-test",
        "drone_id": drone_id,
        "sequence": sequence,
        "issued_at_s": float(sequence),
        "kind": "HOVER",
        "source_observation_id": None,
    }
    return {
        "drone_id": drone_id,
        "action_sequence": sequence,
        "action": action,
        "source_observation": observation,
    }


def _receipt(
    binding: dict[str, object], previous: str | None, previous_state: str | None
) -> dict[str, object]:
    drone_id = str(binding["drone_id"])
    sequence = int(binding["action_sequence"])
    action = binding["action"]
    observation = binding["source_observation"]
    assert isinstance(action, dict) and isinstance(observation, dict)
    state_after = content_hash({"drone_id": drone_id, "sequence": sequence, "state": "after"})
    payload: dict[str, object] = {
        "schema": "org.aerocity.bench.execution-receipt.v2",
        "episode_id": "fleet-test",
        "drone_id": drone_id,
        "action_sequence": sequence,
        "task_time_start_s": float(sequence),
        "task_time_end_s": float(sequence + 1),
        "planning_latency_s": 0.01,
        "action_requested": action["kind"],
        "action_executed": action["kind"],
        "status": "measured_physx_executed",
        "distance_m": 0.0,
        "energy_used_j": 1.0,
        "minimum_clearance_m": 1.0,
        "collision": False,
        "out_of_bounds": False,
        "safety_intervention": False,
        "deadline_miss": False,
        "execution_level": "L1",
        "action_packet_hash": content_hash(action),
        "source_observation_id": observation["observation_id"],
        "source_observation_hash": content_hash(observation),
        "state_before_hash": previous_state
        or content_hash({"drone_id": drone_id, "state": "initial"}),
        "state_after_hash": state_after,
        "previous_receipt_hash": previous,
        "confirmation_ids": [],
    }
    payload["receipt_hash"] = content_hash(payload)
    return payload


def _valid_evidence(ticks: int = 2) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    members = public_fleet_members(_episode())
    receipts: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    prior_hash = {member.drone_id: None for member in members}
    prior_state = {member.drone_id: None for member in members}
    for sequence in range(ticks):
        for member in members:
            binding = _binding(member.drone_id, sequence)
            receipt = _receipt(binding, prior_hash[member.drone_id], prior_state[member.drone_id])
            prior_hash[member.drone_id] = str(receipt["receipt_hash"])
            prior_state[member.drone_id] = str(receipt["state_after_hash"])
            bindings.append(binding)
            receipts.append(receipt)
    return receipts, bindings


def _v4_receipt_evidence() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    receipts, bindings = _valid_evidence(ticks=10)
    for receipt, binding in zip(receipts, bindings, strict=True):
        sequence = int(receipt["action_sequence"])
        planner_invoked = sequence % 5 == 0
        receipt["schema"] = "org.aerocity.bench.execution-receipt.v3"
        receipt["planner_invoked"] = planner_invoked
        receipt["planning_latency_s"] = 0.01 if planner_invoked else 0.0
        binding["planner_invoked"] = planner_invoked
        binding["planning_trigger_reasons"] = (
            ["fixed_period", "initial"]
            if sequence == 0
            else ["fixed_period"]
            if planner_invoked
            else []
        )
    return receipts, bindings


def test_v4_timing_is_reconciled_against_every_fleet_receipt() -> None:
    receipts, bindings = _v4_receipt_evidence()
    _validate_planning_receipt_consistency(_v4_timing(), receipts, bindings)

    held_receipt = next(
        receipt for receipt in receipts if receipt["planner_invoked"] is False
    )
    held_receipt["planning_latency_s"] = 0.01
    with pytest.raises(ValidationError, match="held-action tick invents"):
        _validate_planning_receipt_consistency(_v4_timing(), receipts, bindings)


def test_v4_timing_rejects_receipt_summary_or_trigger_disagreement() -> None:
    receipts, bindings = _v4_receipt_evidence()
    forged_timing = _v4_timing()
    forged_timing["planner_invocation_count"] = 3
    forged_timing["held_action_tick_count"] = 7
    with pytest.raises(ValidationError, match="differs from execution receipts"):
        _validate_planning_receipt_consistency(forged_timing, receipts, bindings)

    receipts, bindings = _v4_receipt_evidence()
    first_tick_binding = next(
        binding for binding in bindings if binding["action_sequence"] == 0
    )
    first_tick_binding["planning_trigger_reasons"] = ["initial"]
    with pytest.raises(ValidationError, match="differs within a fleet tick"):
        _validate_planning_receipt_consistency(_v4_timing(), receipts, bindings)


def test_public_members_require_exactly_four_unique_safe_ids_and_paths() -> None:
    members = public_fleet_members(_episode())

    assert [member.drone_id for member in members] == [
        "drone-000",
        "drone-001",
        "drone-002",
        "drone-003",
    ]
    assert len({member.prim_path for member in members}) == 4
    assert all("-" not in member.prim_path.rsplit("/", maxsplit=1)[-1] for member in members)
    with pytest.raises(ValueError, match="exactly 4"):
        public_fleet_members({"starts": _episode()["starts"][:3]})
    invalid = _episode()
    invalid["starts"][1]["drone_id"] = "../unsafe"  # type: ignore[index]
    with pytest.raises(ValueError, match="unsafe"):
        public_fleet_members(invalid)


def test_shared_step_requires_all_pending_thrust_targets_and_action_roster() -> None:
    members = public_fleet_members(_episode())
    ids = {member.drone_id for member in members}
    ledger = SharedWorldStepLedger(members)
    ledger.record_step(ids)
    assert ledger.shared_physx_step_count == 1
    with pytest.raises(ValueError, match="every fleet member"):
        ledger.record_step({"drone-000"})
    assert_action_roster_complete({drone_id: object() for drone_id in ids}, members)
    with pytest.raises(ValueError, match="roster"):
        assert_action_roster_complete({"drone-000": object()}, members)


def test_fleet_receipts_must_be_canonical_complete_and_per_drone_chained() -> None:
    members = public_fleet_members(_episode())
    receipts, bindings = _valid_evidence()
    assert_canonical_fleet_receipts(receipts, members, expected_control_ticks=2)
    assert_fleet_execution_bindings(receipts, bindings)
    with pytest.raises(ValidationError, match="canonical"):
        assert_canonical_fleet_receipts(list(reversed(receipts)), members, expected_control_ticks=2)
    corrupted = deepcopy(receipts)
    corrupted[4]["previous_receipt_hash"] = None
    corrupted[4]["receipt_hash"] = content_hash(
        {key: value for key, value in corrupted[4].items() if key != "receipt_hash"}
    )
    with pytest.raises(ValidationError, match="discontinuous"):
        assert_canonical_fleet_receipts(corrupted, members, expected_control_ticks=2)
    corrupt_hash = deepcopy(receipts)
    corrupt_hash[0]["distance_m"] = 999.0
    with pytest.raises(ValidationError, match="content hash"):
        assert_canonical_fleet_receipts(corrupt_hash, members, expected_control_ticks=2)


def test_execution_binding_rejects_substituted_action_or_observation() -> None:
    receipts, bindings = _valid_evidence()
    substituted_action = deepcopy(bindings)
    substituted_action[0]["action"]["kind"] = "RETURN"  # type: ignore[index]
    with pytest.raises(ValidationError, match="does not bind"):
        assert_fleet_execution_bindings(receipts, substituted_action)
    substituted_observation = deepcopy(bindings)
    substituted_observation[0]["source_observation"]["observation_id"] = "other"  # type: ignore[index]
    with pytest.raises(ValidationError, match="does not bind"):
        assert_fleet_execution_bindings(receipts, substituted_observation)


def test_confirmation_requires_an_accepted_observe_and_matching_execution_receipt() -> None:
    receipts, bindings = _valid_evidence()
    observe_binding = bindings[0]
    observe_action = observe_binding["action"]
    observe_observation = observe_binding["source_observation"]
    assert isinstance(observe_action, dict) and isinstance(observe_observation, dict)
    observe_action["kind"] = "OBSERVE"
    observe_action["source_observation_id"] = observe_observation["observation_id"]
    confirmation_id = "confirmation-fixture"
    receipts[0]["confirmation_ids"] = [confirmation_id]
    receipts[0]["receipt_hash"] = content_hash(
        {key: value for key, value in receipts[0].items() if key != "receipt_hash"}
    )
    observation_receipts = [
        {
            "observation_id": observe_observation["observation_id"],
            "drone_id": observe_binding["drone_id"],
            "timestamp_s": 0.0,
            "accepted": True,
            "reason": "accepted",
            "receipt_hash": "irrelevant-to-binding-test",
        }
    ]
    confirmations = [
        {
            "schema": "org.aerocity.bench.confirmation-receipt.v1",
            "confirmation_id": confirmation_id,
            "anonymous_target_handle": "anonymous",
            "drone_id": observe_binding["drone_id"],
            "confirmed_at_s": 0.0,
            "source_observation_id": observe_observation["observation_id"],
            "receipt_token": "token",
        }
    ]
    assert_fleet_confirmation_bindings(receipts, bindings, observation_receipts, confirmations)
    confirmations[0]["source_observation_id"] = "not-an-observe"
    with pytest.raises(ValidationError, match="accepted OBSERVE"):
        assert_fleet_confirmation_bindings(receipts, bindings, observation_receipts, confirmations)


def test_altitude_metrics_detect_continuing_sink_not_just_final_altitude() -> None:
    samples = [
        {
            "task_time_s": float(index),
            "position_w_m": [0.0, 0.0, 2.0 - 0.02 * index],
            "linear_velocity_w_mps": [0.0, 0.0, -0.02],
        }
        for index in range(6)
    ]
    metrics = altitude_stability_metrics(samples)
    assert metrics["late_altitude_slope_mps"] == pytest.approx(-0.02)
    assert metrics["terminal_vertical_velocity_mps"] == pytest.approx(-0.02)
    with pytest.raises(ValueError, match="strictly increasing"):
        altitude_stability_metrics(samples[:2] + [samples[1]] + samples[3:])


def test_candidate_shared_hold_gate_rejects_short_or_drifting_flight() -> None:
    passing_metrics = {
        f"drone-{index:03d}": {
            "duration_s": 30.0,
            "initial_altitude_m": 2.5,
            "final_altitude_m": 2.5,
            "altitude_span_m": 0.0,
            "late_altitude_slope_mps": 0.0,
            "terminal_vertical_velocity_mps": 0.0,
        }
        for index in range(4)
    }
    assert candidate_shared_hold_assessment(passing_metrics)["status"] == "PASS"
    short = deepcopy(passing_metrics)
    short["drone-000"]["duration_s"] = 12.0
    assert candidate_shared_hold_assessment(short)["failed_checks_by_drone"] == {
        "drone-000": ["duration"]
    }
    sinking = deepcopy(passing_metrics)
    sinking["drone-002"]["late_altitude_slope_mps"] = -0.01
    assert candidate_shared_hold_assessment(sinking)["failed_checks_by_drone"] == {
        "drone-002": ["late_altitude_slope"]
    }


def test_public_summary_rejects_private_target_or_witness_keys() -> None:
    public = {
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRECHECK_SCOPE,
        "private_evaluator_commitment": "a" * 64,
        "execution_mode": "shared-hold",
        "candidate_shared_hold": {
            "status": "PASS",
            "candidate_preflight_only": True,
            "thresholds": {
                "minimum_duration_s": 30.0,
                "maximum_altitude_span_m": 0.02,
                "maximum_terminal_altitude_error_m": 0.01,
                "maximum_terminal_vertical_velocity_mps": 0.01,
                "maximum_late_altitude_slope_mps": 2.5e-4,
            },
            "failed_checks_by_drone": {},
        },
        "policy_progress": {
            "status": "NOT_APPLICABLE",
            "observe_action_count": 0,
            "confirmation_receipt_count": 0,
            "return_action_count": 0,
            "all_returned_home": True,
            "episode_budget_completed": False,
        },
        "private_report_file_sha256": "b" * 64,
    }
    assert_public_report_has_no_private_truth(public)
    leaked = {**public, "witness_id": "must-not-leak"}
    with pytest.raises(ValidationError, match="leaks private"):
        assert_public_report_has_no_private_truth(leaked)
    local_path = {**public, "diagnostic": r"E:\private_fixture\evidence.json"}
    with pytest.raises(ValidationError, match="local path or runtime URI"):
        assert_public_report_has_no_private_truth(local_path)
    assert FLEET_PRIVATE_SCOPE != FLEET_PRECHECK_SCOPE


def test_report_validator_rejects_shared_step_mismatch_and_public_leak(tmp_path) -> None:
    receipts, bindings = _valid_evidence()
    private_path = tmp_path / "fleet.private.json"
    public_path = tmp_path / "fleet.public.json"
    private: dict[str, object] = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-private.v4",
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRIVATE_SCOPE,
        "private_evaluator_commitment": "a" * 64,
        "execution_mode": "shared-hold",
        "candidate_shared_hold": {
            "status": "PASS",
            "candidate_preflight_only": True,
            "thresholds": {
                "minimum_duration_s": 30.0,
                "maximum_altitude_span_m": 0.02,
                "maximum_terminal_altitude_error_m": 0.01,
                "maximum_terminal_vertical_velocity_mps": 0.01,
                "maximum_late_altitude_slope_mps": 2.5e-4,
            },
            "failed_checks_by_drone": {},
        },
        "policy_progress": {
            "status": "NOT_APPLICABLE",
            "observe_action_count": 0,
            "confirmation_receipt_count": 0,
            "return_action_count": 0,
            "all_returned_home": True,
            "episode_budget_completed": False,
        },
        "route_budget_audit": {
            "schema": "org.aerocity.bench.baseline-route-budget-audit.v1",
            "status": "NOT_APPLICABLE",
            "reason": "shared-hold does not execute a public search route",
        },
        "planning_timing": {
            "schema": "org.aerocity.bench.fleet-preflight-timing.v1",
            "control_tick_count": 2,
            "planning_deadline_s": 0.15,
            "deadline_miss_tick_count": 0,
            "policy_call": {"p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0, "max_s": 0.0},
            "public_observation_build": {
                "p50_s": 0.0,
                "p95_s": 0.0,
                "p99_s": 0.0,
                "max_s": 0.0,
            },
        },
        "fleet_members_private": [
            {
                "drone_id": member.drone_id,
                "start_position_w_m": list(member.start_position_w_m),
                "start_yaw_deg": member.start_yaw_deg,
            }
            for member in public_fleet_members(_episode())
        ],
        "execution": {
            "control_ticks": 2,
            "physical_steps_per_control": 3,
            "shared_physx_step_count": 6,
        },
        "execution_receipts": receipts,
        "execution_bindings_public": bindings,
        "observation_receipts": [],
        "confirmation_receipts": [],
    }
    private["private_report_content_sha256"] = content_hash(private)
    write_json(private_path, private)
    public: dict[str, object] = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight.v4",
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRECHECK_SCOPE,
        "private_evaluator_commitment": private["private_evaluator_commitment"],
        "private_report_file_sha256": file_hash(private_path),
        "candidate_shared_hold": private["candidate_shared_hold"],
        "route_budget_audit": private["route_budget_audit"],
        "planning_timing": private["planning_timing"],
        "policy_progress": private["policy_progress"],
    }
    public["public_report_sha256"] = content_hash(public)
    write_json(public_path, public)
    assert validate_fleet_preflight_reports(public_path, private_path)["status"] == "PASS"
    mismatched_private = deepcopy(private)
    mismatched_private["execution"]["shared_physx_step_count"] = 5  # type: ignore[index]
    mismatched_private["private_report_content_sha256"] = content_hash(
        {
            key: value
            for key, value in mismatched_private.items()
            if key != "private_report_content_sha256"
        }
    )
    write_json(private_path, mismatched_private)
    public["private_report_file_sha256"] = file_hash(private_path)
    public["public_report_sha256"] = content_hash(
        {key: value for key, value in public.items() if key != "public_report_sha256"}
    )
    write_json(public_path, public)
    with pytest.raises(ValidationError, match="shared PhysX"):
        validate_fleet_preflight_reports(public_path, private_path)
    write_json(private_path, private)
    public["private_report_file_sha256"] = file_hash(private_path)
    leaking_public = deepcopy(public)
    leaking_public["target_position"] = [1.0, 2.0, 3.0]
    leaking_public["public_report_sha256"] = content_hash(
        {key: value for key, value in leaking_public.items() if key != "public_report_sha256"}
    )
    write_json(public_path, leaking_public)
    with pytest.raises(ValidationError, match="leaks private"):
        validate_fleet_preflight_reports(public_path, private_path)


def test_private_fixture_cannot_claim_closure_without_return_receipts(tmp_path) -> None:
    receipts, bindings = _valid_evidence()
    first_binding = bindings[0]
    first_action = first_binding["action"]
    first_observation = first_binding["source_observation"]
    assert isinstance(first_action, dict) and isinstance(first_observation, dict)
    first_action["kind"] = "OBSERVE"
    first_action["source_observation_id"] = first_observation["observation_id"]
    receipts[0]["action_requested"] = "OBSERVE"
    receipts[0]["action_executed"] = "OBSERVE"
    receipts[0]["action_packet_hash"] = content_hash(first_action)
    receipts[0]["confirmation_ids"] = ["confirmation-fixture"]
    receipts[0]["receipt_hash"] = content_hash(
        {key: value for key, value in receipts[0].items() if key != "receipt_hash"}
    )
    receipts[4]["previous_receipt_hash"] = receipts[0]["receipt_hash"]
    receipts[4]["receipt_hash"] = content_hash(
        {key: value for key, value in receipts[4].items() if key != "receipt_hash"}
    )
    private_path = tmp_path / "fixture.private.json"
    public_path = tmp_path / "fixture.public.json"
    private: dict[str, object] = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-private.v4",
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRIVATE_SCOPE,
        "private_evaluator_commitment": "a" * 64,
        "private_fixture_commitment": "c" * 64,
        "execution_mode": "private-witness-fixture",
        "candidate_shared_hold": {
            "status": "NOT_APPLICABLE",
            "candidate_preflight_only": True,
            "reason": "closure fixture",
        },
        "policy_progress": {
            "status": "PRIVATE_FIXTURE_CLOSED",
            "observe_action_count": 1,
            "confirmation_receipt_count": 1,
            "return_action_count": 1,
            "all_returned_home": True,
            "episode_budget_completed": False,
        },
        "route_budget_audit": {
            "schema": "org.aerocity.bench.baseline-route-budget-audit.v1",
            "status": "NOT_APPLICABLE",
            "reason": (
                "private-witness-fixture uses an evaluator-owned internal route; "
                "it is not a public search method"
            ),
        },
        "planning_timing": {
            "schema": "org.aerocity.bench.fleet-preflight-timing.v1",
            "control_tick_count": 2,
            "planning_deadline_s": 0.15,
            "deadline_miss_tick_count": 0,
            "policy_call": {"p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0, "max_s": 0.0},
            "public_observation_build": {
                "p50_s": 0.0,
                "p95_s": 0.0,
                "p99_s": 0.0,
                "max_s": 0.0,
            },
        },
        "fleet_members_private": [
            {
                "drone_id": member.drone_id,
                "start_position_w_m": list(member.start_position_w_m),
                "start_yaw_deg": member.start_yaw_deg,
            }
            for member in public_fleet_members(_episode())
        ],
        "execution": {
            "control_ticks": 2,
            "physical_steps_per_control": 3,
            "shared_physx_step_count": 6,
        },
        "execution_receipts": receipts,
        "execution_bindings_public": bindings,
        "observation_receipts": [
            {
                "observation_id": first_observation["observation_id"],
                "drone_id": first_binding["drone_id"],
                "timestamp_s": 0.0,
                "accepted": True,
                "reason": "accepted",
                "receipt_hash": "fixture-observation",
            }
        ],
        "confirmation_receipts": [
            {
                "schema": "org.aerocity.bench.confirmation-receipt.v1",
                "confirmation_id": "confirmation-fixture",
                "anonymous_target_handle": "anonymous",
                "drone_id": first_binding["drone_id"],
                "confirmed_at_s": 0.0,
                "source_observation_id": first_observation["observation_id"],
                "receipt_token": "fixture-token",
            }
        ],
    }
    private["private_report_content_sha256"] = content_hash(private)
    write_json(private_path, private)
    public = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight.v4",
        "formal_score_eligible": False,
        "evidence_scope": FLEET_PRECHECK_SCOPE,
        "private_evaluator_commitment": private["private_evaluator_commitment"],
        "private_fixture_commitment": private["private_fixture_commitment"],
        "private_report_file_sha256": file_hash(private_path),
        "candidate_shared_hold": private["candidate_shared_hold"],
        "route_budget_audit": private["route_budget_audit"],
        "planning_timing": private["planning_timing"],
        "policy_progress": private["policy_progress"],
    }
    public["public_report_sha256"] = content_hash(public)
    write_json(public_path, public)
    with pytest.raises(ValidationError, match="RETURN count"):
        validate_fleet_preflight_reports(public_path, private_path)
