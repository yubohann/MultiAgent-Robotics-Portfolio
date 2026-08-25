from __future__ import annotations

from copy import deepcopy

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.contracts import ActionPacket, ObservationPacket, Pose3D
from aerocity_bench.errors import ValidationError
from aerocity_bench.isaac_bridge import build_l1_execution_receipt
from aerocity_bench.vertical_slice_contract import validate_private_vertical_slice_report


def _state(x: float = 0.0) -> dict[str, object]:
    return {
        "position": [x, 0.0, 1.5],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_mps": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }


def _observation(sequence: int, timestamp_s: float) -> ObservationPacket:
    return ObservationPacket(
        episode_id="episode-vertical-slice",
        observation_id=f"obs-{sequence}",
        drone_id="drone-000",
        sequence=sequence,
        timestamp_s=timestamp_s,
        pose=Pose3D((0.0, 0.0, 1.5), yaw_deg=0.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=100.0 - timestamp_s,
    )


def _observation_receipt(observation: ObservationPacket) -> dict[str, object]:
    payload: dict[str, object] = {
        "observation_id": observation.observation_id,
        "drone_id": observation.drone_id,
        "timestamp_s": observation.timestamp_s,
        "accepted": True,
        "reason": "accepted",
    }
    return {**payload, "receipt_hash": content_hash(payload)}


def _private_report() -> dict[str, object]:
    first_observation = _observation(0, 0.0)
    second_observation = _observation(1, 0.2)
    observe = ActionPacket(
        episode_id=first_observation.episode_id,
        drone_id=first_observation.drone_id,
        sequence=0,
        issued_at_s=0.0,
        kind="OBSERVE",
        source_observation_id=first_observation.observation_id,
    )
    returning = ActionPacket(
        episode_id=second_observation.episode_id,
        drone_id=second_observation.drone_id,
        sequence=1,
        issued_at_s=0.2,
        kind="RETURN",
    )
    before = _state()
    after = _state()
    confirmation_id = "confirmation-fixture"
    first_receipt = build_l1_execution_receipt(
        action=observe,
        source_observation=first_observation,
        state_before=before,
        state_after=after,
        task_time_start_s=0.0,
        task_time_end_s=0.2,
        planning_latency_s=0.0,
        action_executed="OBSERVE",
        status="measured_physx_executed",
        energy_used_j=1.0,
        minimum_clearance_m=1.0,
        collision=False,
        out_of_bounds=False,
        safety_intervention=False,
        deadline_miss=False,
        previous_receipt_hash=None,
        confirmation_ids=(confirmation_id,),
    ).to_dict()
    second_receipt = build_l1_execution_receipt(
        action=returning,
        source_observation=second_observation,
        state_before=after,
        state_after=after,
        task_time_start_s=0.2,
        task_time_end_s=0.4,
        planning_latency_s=0.0,
        action_executed="RETURN",
        status="measured_physx_executed",
        energy_used_j=1.0,
        minimum_clearance_m=1.0,
        collision=False,
        out_of_bounds=False,
        safety_intervention=False,
        deadline_miss=False,
        previous_receipt_hash=str(first_receipt["receipt_hash"]),
    ).to_dict()
    confirmation = {
        "schema": "org.aerocity.bench.confirmation-receipt.v1",
        "confirmation_id": confirmation_id,
        "anonymous_target_handle": "found-opaque",
        "drone_id": first_observation.drone_id,
        "confirmed_at_s": 0.0,
        "source_observation_id": first_observation.observation_id,
        "receipt_token": "opaque-test-token",
    }
    trace = [
        {
            "route_phase": "observe",
            "action": observe.to_dict(),
            "source_observation": first_observation.to_dict(),
            "observation_receipt": _observation_receipt(first_observation),
            "state_before": before,
            "state_after": after,
            "execution_receipt_hash": first_receipt["receipt_hash"],
        },
        {
            "route_phase": "return",
            "action": returning.to_dict(),
            "source_observation": second_observation.to_dict(),
            "observation_receipt": None,
            "state_before": after,
            "state_after": after,
            "execution_receipt_hash": second_receipt["receipt_hash"],
        },
    ]
    return {
        "schema": "org.aerocity.bench.quadrotor-l1-vertical-slice-private.v1",
        "formal_score_eligible": False,
        "evidence_scope": "quadrotor_internal_vertical_slice_private_fixture",
        "input_bindings": {"episode_id": first_observation.episode_id},
        "private_fixture": {"start_drone_id": first_observation.drone_id},
        "private_fixture_commitment": "a" * 64,
        "closure_contract": {"home_position": [0.0, 0.0, 1.5], "home_radius_m": 0.1},
        "execution": {"control_action_count": 2, "simulated_time_s": 0.4},
        "trace_private": trace,
        "execution_receipts": [first_receipt, second_receipt],
        "observation_receipts": [_observation_receipt(first_observation)],
        "confirmation_receipts": [confirmation],
        "failure_records": [],
        "evaluator_private_audit": {
            "episode_id": first_observation.episode_id,
            "confirmed_count": 1,
            "confirmation_ids": [confirmation_id],
        },
        "final": {
            "closure_status": "PASS",
            "confirmation_observed": True,
            "returned_home": True,
            "collision_detected": False,
            "out_of_bounds_detected": False,
            "final_state": after,
        },
    }


def test_private_vertical_slice_contract_binds_native_and_evaluator_evidence() -> None:
    report = _private_report()

    result = validate_private_vertical_slice_report(report)

    assert result["status"] == "PASS"
    assert result["formal_score_eligible"] is False
    assert result["control_action_count"] == 2


def test_private_vertical_slice_contract_rejects_a_deleted_receipt() -> None:
    report = _private_report()
    report["execution_receipts"] = report["execution_receipts"][1:]

    with pytest.raises(ValidationError, match="counts differ"):
        validate_private_vertical_slice_report(report)


def test_private_vertical_slice_contract_rejects_confirmation_attached_to_non_observe() -> None:
    report = deepcopy(_private_report())
    report["confirmation_receipts"][0]["source_observation_id"] = "obs-1"

    with pytest.raises(ValidationError, match="accepted OBSERVE"):
        validate_private_vertical_slice_report(report)


def test_private_vertical_slice_contract_rejects_pass_without_return_action() -> None:
    report = deepcopy(_private_report())
    report["trace_private"][1]["action"]["kind"] = "HOVER"
    report["execution_receipts"][1]["action_requested"] = "HOVER"
    report["execution_receipts"][1]["action_executed"] = "HOVER"
    report["execution_receipts"][1]["action_packet_hash"] = content_hash(
        report["trace_private"][1]["action"]
    )
    payload = dict(report["execution_receipts"][1])
    payload.pop("receipt_hash")
    report["execution_receipts"][1]["receipt_hash"] = content_hash(payload)
    report["trace_private"][1]["execution_receipt_hash"] = report["execution_receipts"][1][
        "receipt_hash"
    ]

    with pytest.raises(ValidationError, match="PASS closure"):
        validate_private_vertical_slice_report(report)
