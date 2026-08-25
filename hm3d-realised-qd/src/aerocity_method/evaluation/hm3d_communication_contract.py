"""Machine-auditable public communication contract for HM3D P07 episodes.

HM3D contains geometry, not radio measurements.  This contract therefore
defines a benchmark networking condition and deliberately avoids claims about
real RF propagation.  It binds the worker's range/LOS graph, packet timing and
acceptance rule to the raw episode record so a detached hash cannot authorize a
run whose communication evidence was never checked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    read_json_object,
    require_identifier,
)

COMMUNICATION_CONTRACT_SCHEMA_VERSION = "hm3d-public-communication-contract-v3"
COMMUNICATION_MODES = frozenset(
    {"intermittent_rendezvous", "continuous_relay", "disconnected_stress"}
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class HM3DCommunicationContract:
    """Frozen public network model and episode admission rule."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        raw = dict(self.payload)
        if raw.get("schema_version") != COMMUNICATION_CONTRACT_SCHEMA_VERSION:
            raise ValueError("HM3D communication contract schema mismatch")
        require_identifier(raw.get("contract_id"), "contract_id")
        mode = raw.get("mode")
        if mode not in COMMUNICATION_MODES:
            raise ValueError("unsupported HM3D communication mode")
        if raw.get("claim_scope") != "benchmark_networking_condition_not_rf_propagation":
            raise ValueError("communication contract must retain its non-RF claim boundary")

        network = _mapping(raw.get("network"), "network")
        if network.get("model") != "range_los_undirected_relay_graph_v1":
            raise ValueError("unsupported HM3D public relay model")
        if finite_number(network.get("maximum_range_m"), "network.maximum_range_m") <= 0.0:
            raise ValueError("network.maximum_range_m must be positive")
        if finite_number(network.get("telemetry_update_hz"), "network.telemetry_update_hz") <= 0.0:
            raise ValueError("network.telemetry_update_hz must be positive")
        _boolean(network.get("line_of_sight_required"), "network.line_of_sight_required")
        for name in ("base_latency_s", "per_hop_latency_s", "loss_probability"):
            value = finite_number(network.get(name), f"network.{name}")
            if value < 0.0:
                raise ValueError(f"network.{name} must be non-negative")
        if float(network["loss_probability"]) > 1.0:
            raise ValueError("network.loss_probability must be in [0, 1]")

        message = _mapping(raw.get("message_policy"), "message_policy")
        if message.get("message_type") != "public_sparse_range_segment_delta":
            raise ValueError("communication requires decision-boundary sparse-map deltas")
        if message.get("aggregation") != "one_delta_per_sender_per_decision":
            raise ValueError("communication requires one aggregate delta per sender and decision")
        if finite_number(message.get("time_to_live_s"), "message_policy.time_to_live_s") <= 0.0:
            raise ValueError("message_policy.time_to_live_s must be positive")

        admission = _mapping(raw.get("admission"), "admission")
        for name in (
            "require_all_recipient_outcomes_resolved",
            "all_telemetry_samples_connected_required",
        ):
            _boolean(admission.get(name), f"admission.{name}")
        maximum_gap = admission.get("maximum_disconnected_duration_s")
        if (
            maximum_gap is not None
            and finite_number(maximum_gap, "maximum disconnected duration") < 0
        ):
            raise ValueError("maximum disconnected duration must be non-negative or null")
        if (
            mode == "continuous_relay"
            and admission.get("all_telemetry_samples_connected_required") is not True
        ):
            raise ValueError("continuous_relay must require all telemetry samples connected")
        if (
            mode == "intermittent_rendezvous"
            and admission.get("all_telemetry_samples_connected_required") is not False
        ):
            raise ValueError("intermittent_rendezvous must permit transient partitions")
        object.__setattr__(self, "payload", raw)

    @classmethod
    def from_path(cls, path: str | Path) -> HM3DCommunicationContract:
        return cls(read_json_object(path))

    @property
    def digest(self) -> str:
        return canonical_sha256(self.payload)

    @property
    def mode(self) -> str:
        return str(self.payload["mode"])

    @property
    def network(self) -> Mapping[str, Any]:
        return _mapping(self.payload["network"], "network")

    @property
    def message_policy(self) -> Mapping[str, Any]:
        return _mapping(self.payload["message_policy"], "message_policy")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def audit_worker_evidence(
        self, communication: Mapping[str, Any], message_delivery: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate raw denominators and evaluate the frozen admission rule."""

        telemetry_update_hz = finite_number(
            communication.get("telemetry_update_hz"), "telemetry_update_hz"
        )
        expected_update_hz = float(self.network["telemetry_update_hz"])
        if not math.isclose(
            telemetry_update_hz, expected_update_hz, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("worker telemetry update rate disagrees with communication contract")
        total = _integer(
            communication.get("relay_telemetry_sample_count"),
            "relay_telemetry_sample_count",
            minimum=1,
        )
        connected = _integer(
            communication.get("relay_connected_telemetry_sample_count"),
            "relay_connected_telemetry_sample_count",
        )
        if connected > total:
            raise ValueError("connected relay telemetry samples exceed total samples")
        fraction = finite_number(
            communication.get("relay_connected_telemetry_sample_fraction"),
            "relay_connected_telemetry_sample_fraction",
        )
        if not math.isclose(fraction, connected / total, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("relay connected step fraction disagrees with raw counts")
        final_graph = _mapping(communication.get("final_graph"), "communication.final_graph")
        final_connected = _boolean(
            final_graph.get("fully_relay_connected"),
            "communication.final_graph.fully_relay_connected",
        )
        longest_gap = finite_number(
            communication.get("longest_sampled_disconnected_duration_s"),
            "longest_sampled_disconnected_duration_s",
        )
        if longest_gap < 0.0:
            raise ValueError("longest disconnected duration must be non-negative")

        outcomes = _mapping(
            message_delivery.get("outcome_counts_after_close"),
            "message_delivery.outcome_counts_after_close",
        )
        delivered = _integer(outcomes.get("DELIVERED"), "delivered recipient count")
        dropped = _integer(outcomes.get("DROPPED"), "dropped recipient count")
        expired = _integer(outcomes.get("EXPIRED"), "expired recipient count")
        expected = _integer(
            message_delivery.get("expected_recipient_outcomes"),
            "expected recipient outcomes",
            minimum=0,
        )
        resolved = _integer(
            message_delivery.get("resolved_recipient_outcomes"),
            "resolved recipient outcomes",
        )
        if delivered + dropped + expired != resolved:
            raise ValueError("message outcome categories do not sum to the resolved denominator")
        maximum_delivery_age_s = finite_number(
            message_delivery.get("maximum_delivery_age_s"), "maximum_delivery_age_s"
        )
        if maximum_delivery_age_s < 0.0:
            raise ValueError("maximum delivery age must be non-negative")

        admission = _mapping(self.payload["admission"], "admission")
        delivery_fraction = 1.0 if expected == 0 else delivered / expected
        checks = {
            "all_recipient_outcomes_resolved": (
                resolved == expected
                if admission["require_all_recipient_outcomes_resolved"]
                else True
            ),
            "all_telemetry_samples_connected": (
                connected == total
                if admission["all_telemetry_samples_connected_required"]
                else True
            ),
            "maximum_disconnected_duration": (
                longest_gap <= float(admission["maximum_disconnected_duration_s"])
                if admission["maximum_disconnected_duration_s"] is not None
                else True
            ),
        }
        return {
            "contract_id": self.payload["contract_id"],
            "contract_sha256": self.digest,
            "mode": self.mode,
            "telemetry_update_hz": telemetry_update_hz,
            "relay_telemetry_sample_count": total,
            "relay_connected_telemetry_sample_count": connected,
            "relay_connected_telemetry_sample_fraction": fraction,
            "longest_sampled_disconnected_duration_s": longest_gap,
            "final_relay_connected": final_connected,
            "expected_recipient_outcomes": expected,
            "resolved_recipient_outcomes": resolved,
            "delivered_recipient_count": delivered,
            "dropped_recipient_count": dropped,
            "expired_recipient_count": expired,
            "delivered_recipient_fraction": delivery_fraction,
            "maximum_delivery_age_s": maximum_delivery_age_s,
            "checks": checks,
            "passed": all(checks.values()),
        }


__all__ = [
    "COMMUNICATION_CONTRACT_SCHEMA_VERSION",
    "COMMUNICATION_MODES",
    "HM3DCommunicationContract",
]
