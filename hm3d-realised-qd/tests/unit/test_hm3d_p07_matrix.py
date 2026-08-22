from __future__ import annotations

from copy import deepcopy

import pytest

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.hm3d_public_schema import public_schema_fields
from aerocity_method.evaluation.hm3d_communication_contract import HM3DCommunicationContract
from aerocity_method.evaluation.hm3d_p07_matrix import (
    P07ProbeRecord,
    assemble_p07_task_validity_pilot,
)
from aerocity_method.evaluation.hm3d_preflight import TASK_VALIDITY_METHODS


def _communication_contract() -> dict[str, object]:
    return {
        "schema_version": "hm3d-public-communication-contract-v3",
        "contract_id": "p07-test-intermittent",
        "mode": "intermittent_rendezvous",
        "claim_scope": "benchmark_networking_condition_not_rf_propagation",
        "network": {
            "model": "range_los_undirected_relay_graph_v1",
            "maximum_range_m": 10.0,
            "telemetry_update_hz": 10.0,
            "line_of_sight_required": True,
            "base_latency_s": 0.05,
            "per_hop_latency_s": 0.02,
            "loss_probability": 0.0,
        },
        "message_policy": {
            "message_type": "public_sparse_range_segment_delta",
            "aggregation": "one_delta_per_sender_per_decision",
            "time_to_live_s": 0.5,
        },
        "admission": {
            "require_all_recipient_outcomes_resolved": True,
            "all_telemetry_samples_connected_required": False,
            "maximum_disconnected_duration_s": None,
        },
    }


def _raw(method: str, *, metric: float = 0.25, timeout: bool = False) -> dict[str, object]:
    public_context = {
        "schema_version": "aerocity-hm3d-exploration-v1",
        "context_id": "p07-paired-context",
        "episode_id": "p07-paired-episode",
        "decision_id": "decision0",
        "agent_features": [
            {"agent_id": "uav0", "features": [1.0, 1.0]},
            {"agent_id": "uav1", "features": [1.0, 1.0]},
        ],
        "public_features": {"sparse_range_schedule_hz": 10.0},
        "preferences": {},
        "budget": {"time_remaining_s": 40.0},
    }
    contract_payload = _communication_contract()
    contract = HM3DCommunicationContract(contract_payload)
    payload: dict[str, object] = {
        "schema_version": "hm3d-p07-exploration-execution-v1",
        "status": "P07_EXECUTION_SMOKE_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "p07_task_validity_closed": False,
        "strategy": method,
        "selection_partition": "train",
        "scene_id": "00244-E64sjs3Dyfd",
        "public_episode_id": "p07-paired-episode",
        "public_context": public_context,
        "public_context_hash": canonical_sha256(public_context),
        "public_candidate_pool_hash": "b" * 64,
        **public_schema_fields(),
        "public_contract_sha256": "c" * 64,
        "evaluation_denominator_sha256": "d" * 64,
        "evaluation_geometry_denominator_sha256": "1" * 64,
        "evaluation_denominator": {
            "schema_version": "hm3d-reachable-evaluation-denominator-v1",
            "denominator_sha256": "d" * 64,
            "metadata_sha256": "e" * 64,
            "mask_sha256": "f" * 64,
            "geometry_evaluation_denominator_sha256": "1" * 64,
            "flight_space_manifest_hash": "2" * 64,
            "source_geometry_sha256": "3" * 64,
            "collision_geometry_sha256": "4" * 64,
            "start_reset_manifest_sha256": "5" * 64,
            "component_ids": [7],
            "start_component_ids": [7, 7, 7, 7],
            "component_voxel_counts": {"7": 64},
            "reachable_voxel_count": 64,
            "reachable_volume_m3": 1.0,
            "resolution_m": 0.25,
        },
        "fleet_size": 4,
        "decision_count": 2,
        "action_budget_s": 40.0,
        "elapsed_physics_s": 40.0,
        "action_budget_utilization": 1.0,
        "candidate_limit": 4,
        "physics_dt_s": 1.0 / 120.0,
        "outcome_time_tolerance_s": 0.25,
        "communication_contract": contract_payload,
        "communication_contract_sha256": contract.digest,
        "engineering_debug": {
            "execution": {
                "communication": {
                    "telemetry_update_hz": 10.0,
                    "relay_telemetry_sample_count": 100,
                    "relay_connected_telemetry_sample_count": 100,
                    "relay_connected_telemetry_sample_fraction": 1.0,
                    "longest_sampled_disconnected_duration_s": 0.0,
                    "final_graph": {"fully_relay_connected": True},
                },
                "message_delivery": {
                    "outcome_counts_after_close": {"DELIVERED": 10, "DROPPED": 0, "EXPIRED": 0},
                    "expected_recipient_outcomes": 10,
                    "resolved_recipient_outcomes": 10,
                    "maximum_delivery_age_s": 0.1,
                },
            }
        },
        "metric_report": {
            "schema_version": "hm3d-exploration-metrics-v2",
            "explored_free_flight_volume_auc_time": metric,
            "final_coverage_at_budget": min(1.0, metric + 0.02),
            "energy_j": 2.5,
        },
        "execution": {
            "collision_count": 0,
            "out_of_bounds_count": 0,
            "failed_fragment_count": 1 if timeout else 0,
            "executed_fragment_count": 3 if timeout else 4,
            "outcome_count": 4,
            "total_energy_used_j": 2.5,
        },
    }
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    return payload


def _probes() -> tuple[P07ProbeRecord, ...]:
    return tuple(
        P07ProbeRecord.from_raw(method, _raw(method, metric=0.1 + index * 0.05))
        for index, method in enumerate(TASK_VALIDITY_METHODS)
    )


def test_p07_pilot_uses_complete_public_paired_order_and_cannot_close() -> None:
    payload = assemble_p07_task_validity_pilot(
        _probes(),
        matrix_run_id="p07-pilot-001",
        sensor_profile_sha256="e" * 64,
        communication_contract_sha256=canonical_sha256(_communication_contract()),
    )
    candidate = payload["preflight_payload_candidate"]
    assert payload["status"] == "P07_TASK_VALIDITY_PILOT_COMPLETE"
    assert payload["p07_task_validity_closed"] is False
    assert candidate["task_validity_passed"] is False
    assert [row["method_id"] for row in candidate["rows"]] == list(TASK_VALIDITY_METHODS)
    assert all(
        row["reads_private_truth"] is False and row["ranked"] is True for row in candidate["rows"]
    )
    assert candidate["primary_metric"] == "Explored-Free-Flight-Volume-AUC_time"


def test_p07_pilot_rejects_task_identity_drift() -> None:
    probes = list(_probes())
    changed = deepcopy(_raw("frontier_3d"))
    changed["public_candidate_pool_hash"] = "1" * 64
    changed["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "runtime_record_sha256"}
    )
    probes[1] = P07ProbeRecord.from_raw("frontier_3d", changed)
    with pytest.raises(ValueError, match="not a paired task matrix"):
        assemble_p07_task_validity_pilot(
            probes,
            matrix_run_id="p07-pilot-drift",
            sensor_profile_sha256="e" * 64,
            communication_contract_sha256=canonical_sha256(_communication_contract()),
        )


def test_p07_rejects_private_target_geometry() -> None:
    raw = _raw("random")
    raw["target_positions"] = [[1.0, 2.0, 3.0]]
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="forbidden private field"):
        P07ProbeRecord.from_raw("random", raw)


def test_p07_rejects_unhashed_public_context() -> None:
    raw = _raw("random")
    context = raw["public_context"]
    assert isinstance(context, dict)
    context["context_id"] = "drifted-context"
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="does not match public_context_hash"):
        P07ProbeRecord.from_raw("random", raw)


def test_failed_timeout_remains_an_episode_denominator() -> None:
    raw = _raw("random", timeout=True)
    raw["status"] = "P07_EXECUTION_SMOKE_FAILED"
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )
    record = P07ProbeRecord.from_raw("random", raw)
    assert (record.planned, record.executed, record.failed, record.timeout) == (1, 0, 0, 1)


def test_complete_status_cannot_mask_a_failed_fragment() -> None:
    raw = _raw("random", timeout=True)
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="COMPLETE status conflicts"):
        P07ProbeRecord.from_raw("random", raw)


def test_p07_rejects_single_decision_engineering_smoke() -> None:
    raw = _raw("random")
    raw["decision_count"] = 1
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="at least two online decisions"):
        P07ProbeRecord.from_raw("random", raw)


def test_p07_rejects_a_worker_that_did_not_use_its_physical_time_budget() -> None:
    raw = _raw("random")
    raw["elapsed_physics_s"] = 10.21
    raw["action_budget_utilization"] = 10.21 / 40.0
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="did not execute the required fraction"):
        P07ProbeRecord.from_raw("random", raw)


def test_p07_keeps_a_outcome_backed_early_terminal_failure_in_the_denominator() -> None:
    raw = _raw("random")
    raw["status"] = "P07_EXECUTION_SMOKE_FAILED"
    raw["decision_count"] = 1
    raw["elapsed_physics_s"] = 10.21
    raw["action_budget_utilization"] = 10.21 / 40.0
    raw["terminal_outcome"] = "executed_terminal_safety_failure"
    execution = raw["execution"]
    assert isinstance(execution, dict)
    execution["collision_count"] = 1
    execution["failed_fragment_count"] = 1
    raw["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "runtime_record_sha256"}
    )

    record = P07ProbeRecord.from_raw("random", raw)

    assert record.terminal_outcome == "executed_terminal_safety_failure"
    assert (record.planned, record.executed, record.failed, record.timeout) == (1, 0, 1, 0)
