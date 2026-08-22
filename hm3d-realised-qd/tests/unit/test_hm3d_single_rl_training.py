from __future__ import annotations

from copy import deepcopy
import json

import pytest

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.hm3d_public_schema import public_schema_fields
from aerocity_method.adapters.hm3d_marvel import (
    MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY,
    MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY,
    MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION,
)
from aerocity_method.evaluation.hm3d_communication_contract import (
    HM3DCommunicationContract,
)
from aerocity_method.evaluation.hm3d_evidence_classification import (
    audit_p07_record_evidence,
    build_current_evidence_integrity_contract,
    normalize_p07_record_purpose,
)
from aerocity_method.evaluation.hm3d_marl_ipp_training import (
    sample_from_p07_training_record as sample_marl_ipp_training_record,
)
from aerocity_method.evaluation.hm3d_outcome_dataset import (
    OUTCOME_DATASET_SCHEMA_VERSION,
    build_outcome_dataset_manifest,
)
from aerocity_method.evaluation.hm3d_marvel_training import (
    sample_from_p07_training_record as sample_marvel_training_record,
)
from aerocity_method.evaluation.hm3d_single_rl_training import (
    sample_from_p07_training_record,
    train_single_rl_baseline,
    training_scene_ids_from_split_manifest,
)


def _split_manifest() -> dict[str, object]:
    assignments = [
        {"scene_id": "00000-kfPV7w3FaU5", "split": "train", "asset_sha256": "a" * 64},
        {"scene_id": "00002-FxCkHAfgh7A", "split": "train", "asset_sha256": "b" * 64},
        {"scene_id": "00807-rsggHU7g7dh", "split": "validation", "asset_sha256": "c" * 64},
    ]
    return {
        "scene_assignments": assignments,
        "split_manifest_sha256": canonical_sha256(assignments),
    }


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


def _raw(
    *,
    scene_id: str = "00000-kfPV7w3FaU5",
    episode_id: str = "rl-train-episode-0",
    action: int = 0,
    strategy: str = "random",
) -> dict[str, object]:
    schema_fields = public_schema_fields()
    public_context = {
        "schema_version": "aerocity-hm3d-exploration-v1",
        "context_id": "rl-train-context",
        "episode_id": episode_id,
        "decision_id": "decision0",
        "agent_features": [
            {"agent_id": "uav0", "features": [1.0, 1.0]},
            {"agent_id": "uav1", "features": [1.0, 1.0]},
        ],
        "public_features": {"sparse_range_schedule_hz": 10.0},
        "preferences": {},
        "budget": {"time_remaining_s": 40.0},
    }
    selection = {
        "strategy": strategy,
        "runner_version": "hm3d-p07-runner-current",
        "selected_candidate_id": f"candidate-{action}",
        "selected_manifest_hash": ("d" if action == 0 else "e") * 64,
    }
    execution = {
        "evidence_class": "real_isaac_physx_cf2x",
        "collision_count": 0,
        "out_of_bounds_count": 0,
        "executed_fragment_count": 4,
        "failed_fragment_count": 0,
        "reusable_fragment_count": 0,
        "provenance": [{"reason_code": "EXECUTION_MATCHED", "allowed": True}],
        "manifest_hash": selection["selected_manifest_hash"],
        "outcome_hashes": [("1" if action == 0 else "2") * 64],
        "trace_hashes": [("4" if action == 0 else "5") * 64],
        "total_energy_used_j": 12.5 + action,
    }
    second_context_hash = canonical_sha256(
        {"episode_id": episode_id, "decision_id": "decision1"}
    )
    second_pool_hash = "a" * 64
    emitted: dict[str, object] = {
        "schema_version": "hm3d-p07-single-rl-train-transition-v4",
        "claim_limit": "test-only public decision transition",
        "scene_id": scene_id,
        "decision_id": "decision0",
        "public_context_hash": canonical_sha256(public_context),
        "public_candidate_pool_hash": "b" * 64,
        **schema_fields,
        "context_features": [0.5, 1.0, 1.0, 2.0 / 6.0],
        "candidate_features": [[0.1, 0.2, 0.3, 1.0, 2.0], [0.2, 0.3, 0.4, 0.5, 3.0]],
        "legal_mask": [True, True],
        "selected_action_index": action,
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_manifest_hash": selection["selected_manifest_hash"],
        "reward_explored_free_flight_volume_auc_time_contribution": 0.10,
        "cost_energy_j": execution["total_energy_used_j"],
        "duration_s": 20.0,
        "terminated": False,
        "truncated": False,
        "next_public_context_hash": second_context_hash,
        "next_public_candidate_pool_hash": second_pool_hash,
        "next_context_features": [0.25, 1.0, 1.0, 2.0 / 6.0],
        "next_candidate_features": [
            [0.1, 0.2, 0.3, 1.0, 2.0],
            [0.2, 0.3, 0.4, 0.5, 3.0],
        ],
        "next_legal_mask": [True, True],
        "outcome_hash": canonical_sha256(
            {
                "manifest_hash": selection["selected_manifest_hash"],
                "outcome_hashes": execution["outcome_hashes"],
            }
        ),
    }
    emitted["transition_sha256"] = canonical_sha256(emitted)
    second_selection = {
        "strategy": strategy,
        "selected_candidate_id": "candidate-later",
        "selected_manifest_hash": "f" * 64,
    }
    second_execution = {
        "evidence_class": "real_isaac_physx_cf2x",
        "manifest_hash": "f" * 64,
        "outcome_hashes": ["3" * 64],
        "trace_hashes": ["6" * 64],
        "total_energy_used_j": 1.0,
    }
    emitted_second = deepcopy(emitted)
    emitted_second.update(
        {
            "decision_id": "decision1",
            "public_context_hash": second_context_hash,
            "public_candidate_pool_hash": second_pool_hash,
            "selected_action_index": 1,
            "selected_candidate_id": second_selection["selected_candidate_id"],
            "selected_manifest_hash": second_selection["selected_manifest_hash"],
            "reward_explored_free_flight_volume_auc_time_contribution": 0.15,
            "cost_energy_j": second_execution["total_energy_used_j"],
            "terminated": False,
            "truncated": True,
            "next_public_context_hash": second_context_hash,
            "next_public_candidate_pool_hash": second_pool_hash,
            "outcome_hash": canonical_sha256(
                {
                    "manifest_hash": second_selection["selected_manifest_hash"],
                    "outcome_hashes": second_execution["outcome_hashes"],
                }
            ),
        }
    )
    emitted_second["transition_sha256"] = canonical_sha256(
        {key: value for key, value in emitted_second.items() if key != "transition_sha256"}
    )
    evaluation_denominator: dict[str, object] = {
        "schema_version": "hm3d-reachable-evaluation-denominator-v1",
        "connectivity": 26,
        "origin_center_m": [0.0, 0.0, 0.0],
        "resolution_m": 0.5,
        "start_positions_m": [[0.25, 0.25, 0.25]] * 4,
        "start_voxel_indices": [[0, 0, 0]] * 4,
        "start_component_ids": [7] * 4,
        "component_ids": [7],
        "component_voxel_counts": {"7": 8},
        "reachable_voxel_count": 8,
        "reachable_volume_m3": 1.0,
        "mask_sha256": "0" * 64,
        "metadata_sha256": "f" * 64,
        "geometry_evaluation_denominator_sha256": "1" * 64,
        "flight_space_manifest_hash": "2" * 64,
        "source_geometry_sha256": "3" * 64,
        "collision_geometry_sha256": "4" * 64,
        "start_reset_manifest_sha256": "5" * 64,
    }
    evaluation_denominator["denominator_sha256"] = canonical_sha256(evaluation_denominator)
    payload: dict[str, object] = {
        "schema_version": "hm3d-p07-exploration-execution-v1",
        "status": "P07_EXECUTION_SMOKE_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "record_purpose": "train_outcome",
        "p07_task_validity_closed": False,
        "evidence_integrity_contract": build_current_evidence_integrity_contract(
            runner_source_sha256="7" * 64,
            execution_source_sha256="8" * 64,
        ),
        "strategy": strategy,
        "selection_partition": "train",
        "scene_id": scene_id,
        "public_episode_id": episode_id,
        "public_context": public_context,
        "public_context_hash": canonical_sha256(public_context),
        "public_candidate_pool_hash": "b" * 64,
        **schema_fields,
        "public_contract_sha256": "c" * 64,
        "evaluation_denominator_sha256": evaluation_denominator["denominator_sha256"],
        "evaluation_geometry_denominator_sha256": "1" * 64,
        "evaluation_denominator": evaluation_denominator,
        "fleet_size": 4,
        "decision_count": 2,
        "action_budget_s": 40.0,
        "elapsed_physics_s": 40.0,
        "action_budget_utilization": 1.0,
        "terminal_outcome": "budget_exhausted",
        "candidate_limit": 4,
        "physics_dt_s": 1.0 / 120.0,
        "arrival_tolerance_m": 0.1,
        "outcome_time_tolerance_s": 0.25,
        "controller_id": "isaac-so3-feedback-v6",
        "action_completion_mode": "event_driven_all_routes_completed_plus_minimum_dwell",
        "execution_profile_sha256": "9" * 64,
        "cf2x_usd_sha256": "a" * 64,
        "sensor_profile_sha256": "b" * 64,
        "split_manifest_sha256": "6" * 64,
        "transit_time_model_sha256": "c" * 64,
        "communication_contract": _communication_contract(),
        "communication_contract_sha256": canonical_sha256(_communication_contract()),
        "engineering_debug": {
            "execution": {
                "communication": {
                    "telemetry_update_hz": 10.0,
                    "relay_telemetry_sample_count": 100,
                    "relay_connected_telemetry_sample_count": 95,
                    "relay_connected_telemetry_sample_fraction": 0.95,
                    "longest_sampled_disconnected_duration_s": 0.05,
                    "final_graph": {"fully_relay_connected": True},
                },
                "message_delivery": {
                    "outcome_counts_after_close": {
                        "DELIVERED": 10,
                        "DROPPED": 0,
                        "EXPIRED": 0,
                    },
                    "expected_recipient_outcomes": 10,
                    "resolved_recipient_outcomes": 10,
                    "maximum_delivery_age_s": 0.1,
                },
            }
        },
        "decisions": [
            {
                "decision_id": "decision0",
                "public_context_hash": canonical_sha256(public_context),
                "public_candidate_pool_hash": "b" * 64,
                **schema_fields,
                "selection": selection,
                "execution": execution,
                "reward_explored_free_flight_volume_auc_time_contribution": 0.10,
                "duration_s": 20.0,
            },
            {
                "decision_id": "decision1",
                "public_context_hash": second_context_hash,
                "public_candidate_pool_hash": second_pool_hash,
                **schema_fields,
                "selection": second_selection,
                "execution": second_execution,
                "reward_explored_free_flight_volume_auc_time_contribution": 0.15,
                "duration_s": 20.0,
            },
        ],
        "execution": {
            "collision_count": 0,
            "out_of_bounds_count": 0,
            "executed_fragment_count": 8,
            "failed_fragment_count": 0,
            "planned_fragment_count": 12,
            "outcome_hashes": execution["outcome_hashes"] + ["3" * 64],
            "total_energy_used_j": execution["total_energy_used_j"] + 1.0,
        },
        "stationarity_supervision": {
            "schema_version": "hm3d-episode-stationarity-supervision-v1",
            "status": "EPISODE_STATIONARITY_SUPERVISION_ADMITTED",
            "violations": [],
        },
        "single_rl_training_transitions": [emitted, emitted_second],
        "metric_report": {
            "schema_version": "hm3d-exploration-metrics-v2",
            "explored_free_flight_volume_auc_time": 0.25,
            "final_coverage_at_budget": 0.30,
            "energy_j": execution["total_energy_used_j"],
        },
    }
    contract = HM3DCommunicationContract(payload["communication_contract"])
    execution_debug = payload["engineering_debug"]["execution"]
    payload["communication_contract_audit"] = contract.audit_worker_evidence(
        execution_debug["communication"], execution_debug["message_delivery"]
    )
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    return payload


def _rewrite_record_hash(payload: dict[str, object]) -> None:
    payload["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "runtime_record_sha256"}
    )


def test_outcome_dataset_indexes_event_driven_reachable_component_records(tmp_path) -> None:
    split_manifest = _split_manifest()
    payload = _raw()
    payload["split_manifest_sha256"] = split_manifest["split_manifest_sha256"]
    _rewrite_record_hash(payload)
    record_path = tmp_path / "p07_event_driven.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_outcome_dataset_manifest([record_path], split_manifest=split_manifest)

    assert dataset["schema_version"] == OUTCOME_DATASET_SCHEMA_VERSION
    assert dataset["real_decision_count"] == 2
    row = dataset["records"][0]
    assert row["evaluation_denominator_sha256"] == payload["evaluation_denominator_sha256"]
    assert (
        row["evaluation_geometry_denominator_sha256"]
        == payload["evaluation_geometry_denominator_sha256"]
    )
    assert row["reachable_component_ids"] == [7]


def test_outcome_dataset_rejects_legacy_completion_and_split_mismatch(tmp_path) -> None:
    split_manifest = _split_manifest()
    payload = _raw()
    payload["split_manifest_sha256"] = split_manifest["split_manifest_sha256"]
    payload["action_completion_mode"] = "legacy_planned_fragment_boundary"
    _rewrite_record_hash(payload)
    record_path = tmp_path / "legacy_completion.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="event-driven"):
        build_outcome_dataset_manifest([record_path], split_manifest=split_manifest)

    payload["action_completion_mode"] = "event_driven_all_routes_completed_plus_minimum_dwell"
    payload["split_manifest_sha256"] = "0" * 64
    _rewrite_record_hash(payload)
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split manifest differs"):
        build_outcome_dataset_manifest([record_path], split_manifest=split_manifest)


def test_outcome_dataset_rejects_mixed_execution_profiles(tmp_path) -> None:
    split_manifest = _split_manifest()
    first = _raw(episode_id="p07-outcome-a")
    second = _raw(episode_id="p07-outcome-b")
    for payload in (first, second):
        payload["split_manifest_sha256"] = split_manifest["split_manifest_sha256"]
        _rewrite_record_hash(payload)
    second["execution_profile_sha256"] = "7" * 64
    _rewrite_record_hash(second)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(ValueError, match="different frozen collection contracts"):
        build_outcome_dataset_manifest([first_path, second_path], split_manifest=split_manifest)


def test_field_level_evidence_keeps_dynamics_when_qd_is_degenerate() -> None:
    payload = _raw()
    audit = audit_p07_record_evidence(payload, known_defects=["QD_DESCRIPTOR_DEGENERATE"])
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is True
    assert audit["use_cases"]["dynamics_calibration_evidence"]["eligible"] is True
    assert audit["fields"]["realised_qd_descriptor"]["eligible"] is False
    assert audit["fields"]["coverage_metric"]["eligible"] is True


def test_field_level_evidence_rejects_sensor_contamination_from_training_only() -> None:
    payload = _raw()
    audit = audit_p07_record_evidence(
        payload, known_defects=["PUBLIC_RANGE_DYNAMIC_BODY_CONTAMINATION"]
    )
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is False
    assert audit["use_cases"]["dynamics_calibration_evidence"]["eligible"] is True
    assert audit["use_cases"]["engineering_diagnostic_evidence"]["eligible"] is True


def test_missing_stationarity_supervision_stays_diagnostic_only() -> None:
    payload = _raw()
    payload.pop("stationarity_supervision")
    _rewrite_record_hash(payload)

    audit = audit_p07_record_evidence(payload)

    assert audit["use_cases"]["dynamics_calibration_evidence"]["eligible"] is True
    assert audit["fields"]["coverage_metric"]["eligible"] is False
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is False
    assert "STATIONARITY_SUPERVISION_MISSING" in audit["fields"]["coverage_metric"]["reasons"]


def test_engineering_smoke_is_preserved_but_excluded_from_train_and_qd() -> None:
    payload = _raw()
    payload["record_purpose"] = "engineering_smoke"
    _rewrite_record_hash(payload)

    audit = audit_p07_record_evidence(payload)

    assert audit["use_cases"]["dynamics_calibration_evidence"]["eligible"] is True
    assert audit["use_cases"]["engineering_diagnostic_evidence"]["eligible"] is True
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is False
    assert "RECORD_PURPOSE_NOT_TRAIN_OUTCOME" in audit["use_cases"][
        "trainable_real_outcome"
    ]["reasons"]
    assert audit["fields"]["realised_qd_descriptor"]["eligible"] is False
    assert "RECORD_PURPOSE_EXCLUDES_QD_HISTORY" in audit["fields"][
        "realised_qd_descriptor"
    ]["reasons"]


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("train_receipt", "train_outcome"),
        ("engineering_receipt", "engineering_smoke"),
        ("qd_receipt", "qd_calibration"),
    ],
)
def test_legacy_record_purpose_aliases_are_normalized(
    legacy: str, expected: str
) -> None:
    assert normalize_p07_record_purpose({"record_purpose": legacy}) == expected


def test_outcome_backed_failed_status_keeps_dynamics_but_not_score_or_learning() -> None:
    """A physical failure is diagnostic evidence, never a successful sample."""

    payload = _raw()
    payload["status"] = "P07_EXECUTION_SMOKE_FAILED"
    payload["terminal_outcome"] = "executed_terminal_safety_failure"
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["failed_fragment_count"] = 1
    _rewrite_record_hash(payload)

    audit = audit_p07_record_evidence(payload)

    assert audit["use_cases"]["dynamics_calibration_evidence"]["eligible"] is True
    assert audit["fields"]["execution_dynamics"]["eligible"] is True
    assert audit["fields"]["formal_performance"]["eligible"] is False
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is False
    assert audit["fields"]["realised_qd_descriptor"]["eligible"] is False
    assert "P07_EXECUTION_NOT_COMPLETE" in audit["fields"][
        "formal_performance"
    ]["reasons"]
    assert "FAILED_FRAGMENTS_PRESENT" in audit["use_cases"][
        "trainable_real_outcome"
    ]["reasons"]


def test_legacy_complete_status_cannot_mask_a_failed_terminal_outcome() -> None:
    """Old hard-coded COMPLETE statuses must fail closed for downstream use."""

    payload = _raw()
    payload["terminal_outcome"] = "executed_terminal_safety_failure"
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["failed_fragment_count"] = 1
    _rewrite_record_hash(payload)

    audit = audit_p07_record_evidence(payload)

    assert audit["use_cases"]["dynamics_calibration_evidence"]["eligible"] is True
    assert audit["fields"]["formal_performance"]["eligible"] is False
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is False
    assert "TERMINAL_OUTCOME_NOT_BUDGET_EXHAUSTED" in audit["fields"][
        "formal_performance"
    ]["reasons"]
    assert "FAILED_FRAGMENTS_PRESENT" in audit["fields"][
        "formal_performance"
    ]["reasons"]


@pytest.mark.parametrize("record_purpose", ("engineering_smoke", None))
def test_outcome_dataset_rejects_non_training_record_purpose(
    tmp_path, record_purpose: str | None
) -> None:
    """Legacy and engineering records cannot cross the real replay boundary."""

    split_manifest = _split_manifest()
    payload = _raw()
    payload["split_manifest_sha256"] = split_manifest["split_manifest_sha256"]
    if record_purpose is None:
        payload.pop("record_purpose")
    else:
        payload["record_purpose"] = record_purpose
    _rewrite_record_hash(payload)
    record_path = tmp_path / "non_training_record.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="RECORD_PURPOSE_NOT_TRAIN_OUTCOME"):
        build_outcome_dataset_manifest([record_path], split_manifest=split_manifest)


def test_qd_calibration_purpose_allows_qd_but_not_rl_transitions() -> None:
    payload = _raw()
    payload["record_purpose"] = "qd_calibration"
    _rewrite_record_hash(payload)

    audit = audit_p07_record_evidence(payload)

    assert audit["fields"]["realised_qd_descriptor"]["eligible"] is True
    assert audit["use_cases"]["trainable_real_outcome"]["eligible"] is False


def _with_marvel_transition(payload: dict[str, object], *, action: int) -> dict[str, object]:
    decisions = payload["decisions"]
    assert isinstance(decisions, list)
    agent_features = [
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.1, 0.0, 0.2, 1.0, 1.0],
        [-0.1, 0.0, 0.4, 1.0, 1.0],
        [0.0, 0.1, 0.6, 1.0, 1.0],
    ]
    adjacency = [
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, True],
        [False, False, True, True],
    ]
    candidate_features = [
        [0.1, 0.2, 0.3, 1.0, 0.2, 0.1, 0.0, 0.4],
        [0.2, 0.3, 0.8, 0.5, 0.4, 0.2, 0.1, 0.6],
    ]
    graph_observation = {
        "node_inputs": [
            [0.0] * 16,
            [1.0, 0.5, 1.0, 1.0, *candidate_features[0], 0.25, 0.25, 0.25, 0.25],
            [1.0, 0.5, 1.0, 1.0, *candidate_features[1], 0.25, 0.25, 0.25, 0.25],
        ],
        "edge_mask": [
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ],
        "current_index": 0,
        "current_edge": [1, 2],
        "edge_padding_mask": [False, False],
        "frontier_distribution": [[0.0] * 36 for _ in range(3)],
        "heading_occupancy": [[0.0] * 36 for _ in range(3)],
        "neighbor_best_headings": [[[0.0] * 36], [[0.0] * 36]],
    }
    transitions: list[dict[str, object]] = []
    for index, raw_decision in enumerate(decisions):
        assert isinstance(raw_decision, dict)
        selection = raw_decision["selection"]
        execution = raw_decision["execution"]
        assert isinstance(selection, dict)
        assert isinstance(execution, dict)
        next_decision = decisions[min(index + 1, len(decisions) - 1)]
        assert isinstance(next_decision, dict)
        transition: dict[str, object] = {
            "schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION,
            "author_model_commit": "318c2a6016d0f2d1dbb0dd08b3f8f8224b361e4c",
            "scene_id": payload["scene_id"],
            "decision_id": raw_decision["decision_id"],
            "public_context_hash": raw_decision["public_context_hash"],
            "public_candidate_pool_hash": raw_decision["public_candidate_pool_hash"],
            **public_schema_fields(),
            "context_features": [0.5, 1.0, 1.0, 0.5],
            "agent_features": agent_features,
            "communication_adjacency": adjacency,
            "candidate_features": candidate_features,
            "marvel_graph_observation": graph_observation,
            "legal_mask": [True, True],
            "selected_action_index": action if index == 0 else 1,
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_manifest_hash": selection["selected_manifest_hash"],
            "reward_explored_free_flight_volume_auc_time_contribution": raw_decision[
                "reward_explored_free_flight_volume_auc_time_contribution"
            ],
            "cost_energy_j": execution["total_energy_used_j"],
            "duration_s": raw_decision["duration_s"],
            "terminated": False,
            "truncated": index == len(decisions) - 1,
            "next_public_context_hash": next_decision["public_context_hash"],
            "next_public_candidate_pool_hash": next_decision["public_candidate_pool_hash"],
            "next_context_features": [0.5, 1.0, 1.0, 0.5],
            "next_agent_features": agent_features,
            "next_communication_adjacency": adjacency,
            "next_candidate_features": candidate_features,
            "next_marvel_graph_observation": graph_observation,
            "next_legal_mask": [True, True],
            "outcome_hash": canonical_sha256(
                {
                    "manifest_hash": selection["selected_manifest_hash"],
                    "outcome_hashes": execution["outcome_hashes"],
                }
            ),
        }
        transition["transition_sha256"] = canonical_sha256(transition)
        transitions.append(transition)
    payload[MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY] = transitions
    payload["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "runtime_record_sha256"}
    )
    return payload


def _with_legacy_marvel_transition(payload: dict[str, object], *, action: int) -> dict[str, object]:
    payload = _with_marvel_transition(payload, action=action)
    payload[MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY] = payload.pop(
        MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY
    )
    _rewrite_record_hash(payload)
    return payload


def _with_marl_ipp_transition(payload: dict[str, object], *, action: int) -> dict[str, object]:
    decisions = payload["decisions"]
    assert isinstance(decisions, list)
    nodes = [
        [0.0] * 8,
        [0.1, 0.2, 0.3, 1.0, 0.2, 0.1, 0.0, 0.4],
        [0.2, 0.3, 0.8, 0.5, 0.4, 0.2, 0.1, 0.6],
    ]
    adjacency = [[True, True, True], [True, True, True], [True, True, True]]
    budgets = [[1.0], [0.7], [0.5]]
    positions = [[0.0] * 32, [0.1] * 32, [-0.1] * 32]
    transitions: list[dict[str, object]] = []
    for index, raw_decision in enumerate(decisions):
        assert isinstance(raw_decision, dict)
        selection = raw_decision["selection"]
        execution = raw_decision["execution"]
        assert isinstance(selection, dict)
        assert isinstance(execution, dict)
        next_decision = decisions[min(index + 1, len(decisions) - 1)]
        assert isinstance(next_decision, dict)
        transition: dict[str, object] = {
            "schema_version": "hm3d-marl-ipp-train-transition-v4",
            "scene_id": payload["scene_id"],
            "decision_id": raw_decision["decision_id"],
            "public_context_hash": raw_decision["public_context_hash"],
            "public_candidate_pool_hash": raw_decision["public_candidate_pool_hash"],
            **public_schema_fields(),
            "node_features": nodes,
            "adjacency": adjacency,
            "budget_features": budgets,
            "position_encoding": positions,
            "legal_mask": [False, True, True],
            "selected_action_index": action if index == 0 else 1,
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_manifest_hash": selection["selected_manifest_hash"],
            "reward_explored_free_flight_volume_auc_time_contribution": raw_decision[
                "reward_explored_free_flight_volume_auc_time_contribution"
            ],
            "cost_energy_j": execution["total_energy_used_j"],
            "duration_s": raw_decision["duration_s"],
            "terminated": False,
            "truncated": index == len(decisions) - 1,
            "next_public_context_hash": next_decision["public_context_hash"],
            "next_public_candidate_pool_hash": next_decision["public_candidate_pool_hash"],
            "next_node_features": nodes,
            "next_adjacency": adjacency,
            "next_budget_features": budgets,
            "next_position_encoding": positions,
            "next_legal_mask": [False, True, True],
            "execution_outcome_hashes": execution["outcome_hashes"],
            "outcome_hash": canonical_sha256(
                {
                    "manifest_hash": selection["selected_manifest_hash"],
                    "outcome_hashes": execution["outcome_hashes"],
                }
            ),
        }
        transition["transition_sha256"] = canonical_sha256(transition)
        transitions.append(transition)
    payload["marl_ipp_training_transitions"] = transitions
    payload["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "runtime_record_sha256"}
    )
    return payload


def test_training_ingestion_binds_actual_outcome_reward_and_frozen_train_split() -> None:
    manifest = _split_manifest()
    samples = sample_from_p07_training_record(
        _raw(), allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest)
    )
    assert len(samples) == 2
    assert sum(sample.transition.reward for sample in samples) == pytest.approx(0.25)
    assert samples[0].transition.cost == pytest.approx(12.5)
    assert samples[0].transition.done is False
    assert samples[1].transition.done is True


def test_training_ingestion_binds_each_real_decision_not_episode_aggregate() -> None:
    manifest = _split_manifest()
    payload = _raw(action=1)
    assert "selection" not in payload
    aggregate = payload["execution"]
    assert isinstance(aggregate, dict)
    assert "manifest_hash" not in aggregate
    samples = sample_from_p07_training_record(
        payload, allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest)
    )
    assert samples[0].transition.action == 1
    assert samples[0].transition.cost == pytest.approx(13.5)
    assert samples[1].transition.cost == pytest.approx(1.0)


def test_training_ingestion_rejects_tampered_first_decision_binding() -> None:
    manifest = _split_manifest()
    payload = deepcopy(_raw())
    decisions = payload["decisions"]
    assert isinstance(decisions, list)
    anchor = decisions[0]
    assert isinstance(anchor, dict)
    selection = anchor["selection"]
    assert isinstance(selection, dict)
    selection["selected_candidate_id"] = "tampered-candidate"
    payload["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="candidate differs from execution"):
        sample_from_p07_training_record(
            payload,
            allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
        )


def test_training_ingestion_rejects_missing_public_task_schema_even_with_new_hash() -> None:
    """A rehashed legacy outcome must not cross the current training boundary."""

    manifest = _split_manifest()
    payload = deepcopy(_raw())
    payload.pop("candidate_pool_schema_version")
    _rewrite_record_hash(payload)

    with pytest.raises(ValueError, match="PUBLIC_TASK_LIFECYCLE_SCHEMA_MISMATCH"):
        sample_from_p07_training_record(
            payload,
            allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
        )


def test_training_ingestion_rejects_decision_schema_drift() -> None:
    """Every decision must remain in the same public-task semantic domain as its root outcome."""

    manifest = _split_manifest()
    payload = deepcopy(_raw())
    decisions = payload["decisions"]
    assert isinstance(decisions, list)
    first = decisions[0]
    assert isinstance(first, dict)
    first["task_reservation_schema_version"] = "hm3d-public-task-reservation-legacy"
    _rewrite_record_hash(payload)

    with pytest.raises(ValueError, match="PUBLIC_TASK_DECISION_SCHEMA_MISMATCH"):
        sample_from_p07_training_record(
            payload,
            allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
        )


def test_training_ingestion_rejects_transition_schema_drift() -> None:
    """A transition cannot silently bind a different task-lifecycle schema."""

    manifest = _split_manifest()
    payload = deepcopy(_raw())
    transitions = payload["single_rl_training_transitions"]
    assert isinstance(transitions, list)
    first = transitions[0]
    assert isinstance(first, dict)
    first["candidate_pool_schema_version"] = "hm3d-public-candidate-pool-legacy"
    first["transition_sha256"] = canonical_sha256(
        {key: value for key, value in first.items() if key != "transition_sha256"}
    )
    _rewrite_record_hash(payload)

    with pytest.raises(ValueError, match="RL_TRANSITION_DECISION_BINDING_MISMATCH"):
        sample_from_p07_training_record(
            payload,
            allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
        )


@pytest.mark.parametrize(
    ("builder", "reader"),
    [
        (_with_marvel_transition, sample_marvel_training_record),
        (_with_legacy_marvel_transition, sample_marvel_training_record),
        (_with_marl_ipp_transition, sample_marl_ipp_training_record),
    ],
)
@pytest.mark.parametrize("action", [0, 1])
def test_external_learning_ports_bind_the_first_real_four_agent_decision(
    builder, reader, action: int
) -> None:
    manifest = _split_manifest()
    payload = builder(_raw(action=action), action=action)
    samples = reader(
        payload,
        allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
    )
    assert len(samples) == 2
    assert sum(sample.row.reward for sample in samples) == pytest.approx(0.25)
    assert samples[-1].done is True


def test_marvel_supplementary_reference_rejects_current_and_legacy_transition_keys_together() -> None:
    manifest = _split_manifest()
    payload = _with_marvel_transition(_raw(), action=0)
    payload[MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY] = payload[
        MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY
    ]
    _rewrite_record_hash(payload)

    with pytest.raises(ValueError, match="both current and legacy transition keys"):
        sample_marvel_training_record(
            payload,
            allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
        )


@pytest.mark.parametrize(
    ("builder", "reader", "transition_key", "error"),
    [
        (
            _with_marvel_transition,
            sample_marvel_training_record,
            MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY,
            "actual decision contribution",
        ),
        (
            _with_legacy_marvel_transition,
            sample_marvel_training_record,
            MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY,
            "actual decision contribution",
        ),
        (
            _with_marl_ipp_transition,
            sample_marl_ipp_training_record,
            "marl_ipp_training_transitions",
            "actual decision contribution",
        ),
    ],
)
def test_external_learning_ports_reject_planner_reward_substitution(
    builder, reader, transition_key: str, error: str
) -> None:
    manifest = _split_manifest()
    payload = builder(_raw(), action=0)
    transitions = payload[transition_key]
    assert isinstance(transitions, list)
    transition = transitions[0]
    assert isinstance(transition, dict)
    transition["reward_explored_free_flight_volume_auc_time_contribution"] = 0.99
    transition["transition_sha256"] = canonical_sha256(
        {key: value for key, value in transition.items() if key != "transition_sha256"}
    )
    payload["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match=error):
        reader(
            payload,
            allowed_train_scene_ids=training_scene_ids_from_split_manifest(manifest),
        )


def test_training_ingestion_rejects_mutated_reward_or_non_train_scene() -> None:
    manifest = _split_manifest()
    allowed = training_scene_ids_from_split_manifest(manifest)
    altered = deepcopy(_raw())
    transitions = altered["single_rl_training_transitions"]
    assert isinstance(transitions, list)
    transition = transitions[0]
    assert isinstance(transition, dict)
    transition["reward_explored_free_flight_volume_auc_time_contribution"] = 0.99
    transition["transition_sha256"] = canonical_sha256(
        {key: value for key, value in transition.items() if key != "transition_sha256"}
    )
    altered["runtime_record_sha256"] = canonical_sha256(
        {key: value for key, value in altered.items() if key != "runtime_record_sha256"}
    )
    with pytest.raises(ValueError, match="actual decision AUC contribution"):
        sample_from_p07_training_record(altered, allowed_train_scene_ids=allowed)
    with pytest.raises(ValueError, match="absent from the frozen P05 train split"):
        sample_from_p07_training_record(
            _raw(scene_id="00807-rsggHU7g7dh"), allowed_train_scene_ids=allowed
        )


def test_training_performs_real_updates_and_records_only_train_rollout_provenance() -> None:
    manifest = _split_manifest()
    allowed = training_scene_ids_from_split_manifest(manifest)
    episodes = (
        sample_from_p07_training_record(
            _raw(episode_id="episode-a0", action=0), allowed_train_scene_ids=allowed
        ),
        sample_from_p07_training_record(
            _raw(episode_id="episode-a1", action=1), allowed_train_scene_ids=allowed
        ),
        sample_from_p07_training_record(
            _raw(scene_id="00002-FxCkHAfgh7A", episode_id="episode-b0", action=0),
            allowed_train_scene_ids=allowed,
        ),
        sample_from_p07_training_record(
            _raw(scene_id="00002-FxCkHAfgh7A", episode_id="episode-b1", action=1),
            allowed_train_scene_ids=allowed,
        ),
    )
    samples = tuple(sample for episode in episodes for sample in episode)
    checkpoint, provenance = train_single_rl_baseline(
        samples,
        split_manifest_sha256=str(manifest["split_manifest_sha256"]),
        updates=2,
        hidden_dim=8,
        seed=17,
        minimum_transitions=8,
        minimum_scenes=2,
    )
    assert checkpoint["training_partition"] == "train"
    assert checkpoint["training_updates"] == 2
    assert provenance["episode_count"] == 4
    assert provenance["transition_count"] == 8
    assert provenance["model"]["archive"] is False
    assert provenance["model"]["ogfr"] is False
