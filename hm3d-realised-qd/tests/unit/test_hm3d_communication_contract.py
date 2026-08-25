from __future__ import annotations

from copy import deepcopy

import pytest

from aerocity_method.evaluation.hm3d_communication_contract import (
    HM3DCommunicationContract,
)


def _contract(*, mode: str = "intermittent_rendezvous") -> dict[str, object]:
    return {
        "schema_version": "hm3d-public-communication-contract-v3",
        "contract_id": f"unit-{mode}",
        "mode": mode,
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
            "all_telemetry_samples_connected_required": mode == "continuous_relay",
            "maximum_disconnected_duration_s": None,
        },
    }


def _evidence() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "telemetry_update_hz": 10.0,
            "relay_telemetry_sample_count": 100,
            "relay_connected_telemetry_sample_count": 95,
            "relay_connected_telemetry_sample_fraction": 0.95,
            "longest_sampled_disconnected_duration_s": 0.05,
            "final_graph": {"fully_relay_connected": True},
        },
        {
            "outcome_counts_after_close": {"DELIVERED": 10, "DROPPED": 0, "EXPIRED": 0},
            "expected_recipient_outcomes": 10,
            "resolved_recipient_outcomes": 10,
            "maximum_delivery_age_s": 0.1,
        },
    )


def test_intermittent_contract_accepts_brief_partition_with_complete_delivery() -> None:
    communication, delivery = _evidence()
    audit = HM3DCommunicationContract(_contract()).audit_worker_evidence(communication, delivery)
    assert audit["passed"] is True
    assert audit["relay_connected_telemetry_sample_fraction"] == pytest.approx(0.95)


def test_continuous_contract_rejects_the_same_brief_partition() -> None:
    communication, delivery = _evidence()
    audit = HM3DCommunicationContract(_contract(mode="continuous_relay")).audit_worker_evidence(
        communication, delivery
    )
    assert audit["passed"] is False
    assert audit["checks"]["all_telemetry_samples_connected"] is False


def test_contract_rejects_a_fraction_that_hides_raw_count_drift() -> None:
    communication, delivery = _evidence()
    communication["relay_connected_telemetry_sample_fraction"] = 0.0
    with pytest.raises(ValueError, match="disagrees with raw counts"):
        HM3DCommunicationContract(_contract()).audit_worker_evidence(communication, delivery)


def test_contract_records_an_expired_delta_without_discarding_the_episode() -> None:
    communication, delivery = _evidence()
    changed = deepcopy(delivery)
    changed["outcome_counts_after_close"] = {
        "DELIVERED": 9,
        "DROPPED": 0,
        "EXPIRED": 1,
    }
    audit = HM3DCommunicationContract(_contract()).audit_worker_evidence(communication, changed)
    assert audit["passed"] is True
    assert audit["expired_recipient_count"] == 1
    assert audit["delivered_recipient_fraction"] == pytest.approx(0.9)


def test_contract_refuses_to_mislabel_range_los_as_real_rf() -> None:
    contract = _contract()
    contract["claim_scope"] = "real_indoor_radio_propagation"
    with pytest.raises(ValueError, match="non-RF claim boundary"):
        HM3DCommunicationContract(contract)
