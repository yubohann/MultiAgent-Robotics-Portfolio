"""Fail-closed assembly of the target-free HM3D P07 exploration matrix.

P07 compares public methods on the same scene, reset, candidate pool, sensor
entitlement and time budget.  The evaluator may own the complete collision
field, but a worker record may expose only aggregate exploration metrics and
outcome identities. Records carrying fields from a retired task contract are
deliberately rejected by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.contracts.hm3d_public_schema import (
    PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    require_current_public_schema,
)
from aerocity_method.evaluation.hm3d_communication_contract import HM3DCommunicationContract
from aerocity_method.evaluation.hm3d_preflight import PRIMARY_METRIC, TASK_VALIDITY_METHODS

P07_MATRIX_PILOT_SCHEMA_VERSION = "hm3d-p07-exploration-pilot-v3"
MINIMUM_ACTION_BUDGET_UTILIZATION = 0.95
_RAW_SCHEMA_VERSIONS = frozenset(
    {
        "hm3d-p07-exploration-execution-v1",
    }
)
_COMPLETE_STATUS = "P07_EXECUTION_SMOKE_COMPLETE"
_FAILED_STATUS = "P07_EXECUTION_SMOKE_FAILED"
_BUDGET_EXHAUSTED = "budget_exhausted"
_EXECUTED_TERMINAL_SAFETY_FAILURE = "executed_terminal_safety_failure"
_TERMINAL_OUTCOMES = frozenset({_BUDGET_EXHAUSTED, _EXECUTED_TERMINAL_SAFETY_FAILURE})
# These names are fail-closed input guards, not a compatibility path.
_FORBIDDEN_RETIRED_FIELDS = frozenset(
    {
        "target",
        "targets",
        "target_id",
        "target_ids",
        "target_position",
        "target_positions",
        "target_count",
        "target_denominator",
        "target_generation",
        "evaluator_private_task_probe",
        "evaluator_oracle",
        "oracle_only",
        "confirmed_recall_auc_time",
        "final_confirmed_recall",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    int(value, 16)
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unit_interval(value: Any, name: str) -> float:
    resolved = finite_number(value, name)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return resolved


def _assert_no_private_truth(value: Any, path: str = "worker_record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string key")
            lowered = key.casefold()
            if lowered in _FORBIDDEN_RETIRED_FIELDS or lowered.startswith("target_"):
                raise ValueError(f"{path} serializes forbidden private field {key}")
            _assert_no_private_truth(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_no_private_truth(child, f"{path}[{index}]")


def _metric_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metric = payload.get("metric_report")
    if metric is None:
        metric = payload.get("exploration_metric_report")
    metric = _mapping(metric, "metric_report")
    if metric.get("schema_version") != "hm3d-exploration-metrics-v2":
        raise ValueError("metric_report must use hm3d-exploration-metrics-v2")
    return metric


def _execution_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = payload.get("execution")
    if isinstance(execution, Mapping):
        return execution
    outcome = _mapping(payload.get("execution_outcome"), "execution_outcome")
    outcomes = outcome.get("agent_outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("execution_outcome.agent_outcomes is required")
    return {
        "collision_count": sum(
            int(bool(row.get("collision_count", 0))) for row in outcomes if isinstance(row, Mapping)
        ),
        "out_of_bounds_count": sum(
            int(bool(row.get("out_of_bounds", False)))
            for row in outcomes
            if isinstance(row, Mapping)
        ),
        "failed_fragment_count": sum(
            int(bool(row.get("aborted", False))) for row in outcomes if isinstance(row, Mapping)
        ),
        "executed_fragment_count": len(outcomes),
        "total_energy_used_j": sum(
            float(row.get("energy_j", 0.0)) for row in outcomes if isinstance(row, Mapping)
        ),
        "outcome_hashes": [outcome.get("outcome_id")],
    }


def _communication_payload(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    debug = payload.get("engineering_debug")
    if isinstance(debug, Mapping):
        execution = debug.get("execution")
        if isinstance(execution, Mapping):
            return _mapping(execution.get("communication"), "communication"), _mapping(
                execution.get("message_delivery"), "message_delivery"
            )
    communication = _mapping(payload.get("communication"), "communication")
    delivered = _integer(communication.get("delivered_count", 0), "delivered_count")
    dropped = _integer(communication.get("dropped_count", 0), "dropped_count")
    expired = _integer(communication.get("expired_count", 0), "expired_count")
    outcome_count = _integer(
        communication.get("outcome_count", delivered + dropped + expired), "outcome_count"
    )
    return communication, {
        "outcome_counts_after_close": {
            "DELIVERED": delivered,
            "DROPPED": dropped,
            "EXPIRED": expired,
        },
        "expected_recipient_outcomes": outcome_count,
        "resolved_recipient_outcomes": outcome_count,
    }


@dataclass(frozen=True, slots=True)
class P07ProbeRecord:
    """One target-free public worker result."""

    method_id: str
    raw_record_sha256: str
    partition: str
    scene_id: str
    public_episode_id: str
    public_context_hash: str
    public_candidate_pool_hash: str
    candidate_pool_schema_version: str
    task_reservation_schema_version: str
    public_contract_sha256: str
    evaluation_denominator_sha256: str
    evaluation_geometry_denominator_sha256: str
    fleet_size: int
    action_budget_s: float
    elapsed_physics_s: float
    action_budget_utilization: float
    terminal_outcome: str
    candidate_limit: int
    physics_dt_s: float
    outcome_time_tolerance_s: float
    communication_contract_sha256: str
    communication_mode: str
    relay_telemetry_sample_count: int
    relay_connected_telemetry_sample_count: int
    relay_connected_telemetry_sample_fraction: float
    delivered_recipient_count: int
    expected_recipient_outcomes: int
    communication_contract_passed: bool
    planned: int
    executed: int
    failed: int
    timeout: int
    oom: int
    other_failed: int
    explored_free_flight_volume_auc_time: float
    final_coverage_at_budget: float
    collision_count: int
    communication_failure_count: int
    energy_used_j: float
    reads_private_truth: bool
    oracle_only: bool
    ranked: bool
    fragment_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        require_identifier(self.method_id, "method_id")
        _sha(self.raw_record_sha256, "raw_record_sha256")
        if self.partition not in {"train", "validation"}:
            raise ValueError("P07 pilot may only use train or validation")
        for name in (
            "scene_id",
            "public_episode_id",
            "public_context_hash",
            "public_candidate_pool_hash",
            "public_contract_sha256",
            "evaluation_denominator_sha256",
            "evaluation_geometry_denominator_sha256",
            "communication_contract_sha256",
        ):
            value = getattr(self, name)
            if name.endswith("hash") or name.endswith("sha256"):
                _sha(value, name)
            else:
                require_identifier(value, name)
        if self.candidate_pool_schema_version != PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION:
            raise ValueError("P07 probes require the current public candidate-pool schema")
        if self.task_reservation_schema_version != PUBLIC_TASK_RESERVATION_SCHEMA_VERSION:
            raise ValueError("P07 probes require the current public task-reservation schema")
        if _integer(self.fleet_size, "fleet_size", minimum=1) != FORMAL_FLEET_SIZE:
            raise ValueError(f"P07 probes must use the formal N={FORMAL_FLEET_SIZE} fleet")
        _integer(self.candidate_limit, "candidate_limit", minimum=self.fleet_size)
        for name in ("action_budget_s", "elapsed_physics_s", "physics_dt_s"):
            if finite_number(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        utilization = _unit_interval(self.action_budget_utilization, "action_budget_utilization")
        expected_utilization = self.elapsed_physics_s / self.action_budget_s
        if abs(utilization - expected_utilization) > 1.0e-9:
            raise ValueError("action budget utilization disagrees with elapsed physics time")
        if self.terminal_outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("P07 terminal outcome is not an allowed physical episode outcome")
        if finite_number(self.outcome_time_tolerance_s, "outcome_time_tolerance_s") < 0.0:
            raise ValueError("outcome_time_tolerance_s must be non-negative")
        if self.communication_mode not in {
            "intermittent_rendezvous",
            "continuous_relay",
            "disconnected_stress",
        }:
            raise ValueError("unsupported communication mode")
        total = _integer(
            self.relay_telemetry_sample_count, "relay_telemetry_sample_count", minimum=1
        )
        connected = _integer(
            self.relay_connected_telemetry_sample_count,
            "relay_connected_telemetry_sample_count",
        )
        if connected > total:
            raise ValueError("connected relay telemetry samples exceed total samples")
        fraction = _unit_interval(
            self.relay_connected_telemetry_sample_fraction,
            "relay_connected_telemetry_sample_fraction",
        )
        if abs(fraction - connected / total) > 1e-12:
            raise ValueError("relay connected fraction disagrees with raw counts")
        delivered = _integer(self.delivered_recipient_count, "delivered_recipient_count")
        expected = _integer(
            self.expected_recipient_outcomes, "expected_recipient_outcomes", minimum=1
        )
        if delivered > expected:
            raise ValueError("delivered recipients exceed expected recipients")
        if not isinstance(self.communication_contract_passed, bool):
            raise ValueError("communication_contract_passed must be boolean")
        counts = {
            name: _integer(getattr(self, name), name)
            for name in ("planned", "executed", "failed", "timeout", "oom", "other_failed")
        }
        if counts["planned"] != 1 or sum(counts[name] for name in counts if name != "planned") != 1:
            raise ValueError("a P07 probe must contribute exactly one complete episode denominator")
        if self.terminal_outcome == _BUDGET_EXHAUSTED:
            if utilization < MINIMUM_ACTION_BUDGET_UTILIZATION:
                raise ValueError(
                    "P07 worker did not execute the required fraction of its physical-time budget"
                )
        elif counts["executed"] != 0 or counts["failed"] + counts["timeout"] != 1:
            raise ValueError(
                "an early P07 terminal outcome requires an executed failed-fragment denominator"
            )
        _unit_interval(
            self.explored_free_flight_volume_auc_time, "explored_free_flight_volume_auc_time"
        )
        _unit_interval(self.final_coverage_at_budget, "final_coverage_at_budget")
        _integer(self.collision_count, "collision_count")
        _integer(self.communication_failure_count, "communication_failure_count")
        if finite_number(self.energy_used_j, "energy_used_j") < 0.0:
            raise ValueError("energy_used_j must be non-negative")
        if self.reads_private_truth or self.oracle_only or not self.ranked:
            raise ValueError("P07 ranked methods must be public and ranked")
        if not self.fragment_counts:
            raise ValueError("fragment_counts must be non-empty")

    @classmethod
    def from_raw(cls, method_id: str, payload: Mapping[str, Any]) -> P07ProbeRecord:
        require_identifier(method_id, "method_id")
        _assert_no_private_truth(payload)
        if payload.get("schema_version") not in _RAW_SCHEMA_VERSIONS:
            raise ValueError("P07 probe schema mismatch")
        if payload.get("synthetic") is not False or payload.get("formal_result") is not False:
            raise ValueError("P07 pilot requires a non-synthetic non-formal worker record")
        if payload.get("p07_task_validity_closed") is not False:
            raise ValueError("worker records may not close P07 directly")
        if payload.get("strategy") != method_id:
            raise ValueError("probe method_id differs from worker strategy")
        terminal_outcome = payload.get("terminal_outcome", _BUDGET_EXHAUSTED)
        if not isinstance(terminal_outcome, str):
            raise ValueError("P07 terminal_outcome must be a string")
        decision_count = _integer(payload.get("decision_count"), "decision_count")
        minimum_decisions = 1 if terminal_outcome == _EXECUTED_TERMINAL_SAFETY_FAILURE else 2
        if decision_count < minimum_decisions:
            raise ValueError("P07 exploration probes require at least two online decisions")
        context = _mapping(payload.get("public_context"), "public_context")
        context_hash = _sha(payload.get("public_context_hash"), "public_context_hash")
        if canonical_sha256(context) != context_hash:
            raise ValueError("public context payload does not match public_context_hash")
        raw_hash = _sha(payload.get("runtime_record_sha256"), "runtime_record_sha256")
        unsigned = dict(payload)
        unsigned.pop("runtime_record_sha256", None)
        if canonical_sha256(unsigned) != raw_hash:
            raise ValueError("P07 worker record SHA-256 does not match its contents")
        require_current_public_schema(payload, context="P07 worker record")
        status = payload.get("status")
        if status not in {_COMPLETE_STATUS, _FAILED_STATUS}:
            raise ValueError("P07 worker status is not a recognized execution status")
        denominator = _mapping(payload.get("evaluation_denominator"), "evaluation_denominator")
        if denominator.get("schema_version") != "hm3d-reachable-evaluation-denominator-v1":
            raise ValueError("P07 worker does not use the reachable-component denominator schema")
        denominator_hash = _sha(
            payload.get("evaluation_denominator_sha256"), "evaluation_denominator_sha256"
        )
        if denominator.get("denominator_sha256") != denominator_hash:
            raise ValueError("P07 reachable denominator payload does not match its hash")
        for field in (
            "metadata_sha256",
            "mask_sha256",
            "geometry_evaluation_denominator_sha256",
            "flight_space_manifest_hash",
            "source_geometry_sha256",
            "collision_geometry_sha256",
            "start_reset_manifest_sha256",
        ):
            _sha(denominator.get(field), f"evaluation_denominator.{field}")
        geometry_denominator_hash = _sha(
            payload.get("evaluation_geometry_denominator_sha256"),
            "evaluation_geometry_denominator_sha256",
        )
        if denominator.get("geometry_evaluation_denominator_sha256") != geometry_denominator_hash:
            raise ValueError("P07 episode denominator does not bind its P04 geometry denominator")
        component_ids = denominator.get("component_ids")
        start_component_ids = denominator.get("start_component_ids")
        component_counts = denominator.get("component_voxel_counts")
        if (
            not isinstance(component_ids, list)
            or not component_ids
            or not isinstance(start_component_ids, list)
            or not start_component_ids
            or not isinstance(component_counts, Mapping)
        ):
            raise ValueError("P07 reachable denominator component provenance is incomplete")
        if set(component_ids) != set(start_component_ids):
            raise ValueError("P07 start components disagree with the reachable denominator")
        reachable_voxels = _integer(
            denominator.get("reachable_voxel_count"), "reachable denominator voxel count", minimum=1
        )
        reachable_volume_m3 = finite_number(
            denominator.get("reachable_volume_m3"), "reachable denominator volume"
        )
        resolution_m = finite_number(denominator.get("resolution_m"), "reachable denominator resolution")
        if resolution_m <= 0.0 or reachable_volume_m3 <= 0.0:
            raise ValueError("P07 reachable denominator geometry must be positive")
        if abs(reachable_volume_m3 - reachable_voxels * resolution_m**3) > 1.0e-9:
            raise ValueError("P07 reachable denominator volume disagrees with voxel count")
        contract_payload = _mapping(payload.get("communication_contract"), "communication_contract")
        contract = HM3DCommunicationContract(contract_payload)
        contract_hash = _sha(
            payload.get("communication_contract_sha256"), "communication_contract_sha256"
        )
        if contract.digest != contract_hash:
            raise ValueError("communication contract payload does not match its SHA-256")
        communication, delivery = _communication_payload(payload)
        audit = contract.audit_worker_evidence(communication, delivery)
        if audit["passed"] is not True:
            raise ValueError("worker record fails its frozen communication contract")
        metric = _metric_payload(payload)
        execution = _execution_payload(payload)
        collision_count = _integer(execution.get("collision_count", 0), "collision_count")
        out_of_bounds = _integer(execution.get("out_of_bounds_count", 0), "out_of_bounds_count")
        failed_fragments = _integer(
            execution.get("failed_fragment_count", 0), "failed_fragment_count"
        )
        has_failure = terminal_outcome != _BUDGET_EXHAUSTED or failed_fragments > 0
        if status == _COMPLETE_STATUS and has_failure:
            raise ValueError(
                "P07 COMPLETE status conflicts with terminal outcome or failed fragments"
            )
        if status == _FAILED_STATUS and not has_failure:
            raise ValueError(
                "P07 FAILED status lacks a terminal failure or failed fragment"
            )
        timeout = int(failed_fragments > 0 and out_of_bounds == 0 and collision_count == 0)
        failed = int(
            collision_count > 0 or out_of_bounds > 0 or (failed_fragments > 0 and not timeout)
        )
        executed = int(not (failed or timeout))
        return cls(
            method_id=method_id,
            raw_record_sha256=raw_hash,
            partition=str(payload.get("selection_partition")),
            scene_id=str(payload.get("scene_id")),
            public_episode_id=str(payload.get("public_episode_id") or context.get("episode_id")),
            public_context_hash=context_hash,
            public_candidate_pool_hash=_sha(
                payload.get("public_candidate_pool_hash"), "public_candidate_pool_hash"
            ),
            candidate_pool_schema_version=str(payload.get("candidate_pool_schema_version")),
            task_reservation_schema_version=str(payload.get("task_reservation_schema_version")),
            public_contract_sha256=_sha(
                payload.get("public_contract_sha256"), "public_contract_sha256"
            ),
            evaluation_denominator_sha256=_sha(
                payload.get("evaluation_denominator_sha256"), "evaluation_denominator_sha256"
            ),
            evaluation_geometry_denominator_sha256=geometry_denominator_hash,
            fleet_size=_integer(payload.get("fleet_size"), "fleet_size", minimum=1),
            action_budget_s=finite_number(payload.get("action_budget_s"), "action_budget_s"),
            elapsed_physics_s=finite_number(payload.get("elapsed_physics_s"), "elapsed_physics_s"),
            action_budget_utilization=_unit_interval(
                payload.get("action_budget_utilization"), "action_budget_utilization"
            ),
            terminal_outcome=terminal_outcome,
            candidate_limit=_integer(payload.get("candidate_limit"), "candidate_limit", minimum=1),
            physics_dt_s=finite_number(payload.get("physics_dt_s"), "physics_dt_s"),
            outcome_time_tolerance_s=finite_number(
                payload.get("outcome_time_tolerance_s", 0.25), "outcome_time_tolerance_s"
            ),
            communication_contract_sha256=contract_hash,
            communication_mode=str(audit["mode"]),
            relay_telemetry_sample_count=_integer(
                audit["relay_telemetry_sample_count"],
                "relay_telemetry_sample_count",
                minimum=1,
            ),
            relay_connected_telemetry_sample_count=_integer(
                audit["relay_connected_telemetry_sample_count"],
                "relay_connected_telemetry_sample_count",
            ),
            relay_connected_telemetry_sample_fraction=_unit_interval(
                audit["relay_connected_telemetry_sample_fraction"],
                "relay_connected_telemetry_sample_fraction",
            ),
            delivered_recipient_count=_integer(
                audit["delivered_recipient_count"], "delivered_recipient_count"
            ),
            expected_recipient_outcomes=_integer(
                audit["expected_recipient_outcomes"], "expected_recipient_outcomes", minimum=1
            ),
            communication_contract_passed=True,
            planned=1,
            executed=executed,
            failed=failed,
            timeout=timeout,
            oom=0,
            other_failed=0,
            explored_free_flight_volume_auc_time=_unit_interval(
                metric.get("explored_free_flight_volume_auc_time"), "metric AUC"
            ),
            final_coverage_at_budget=_unit_interval(
                metric.get("final_coverage_at_budget"), "metric final coverage"
            ),
            collision_count=collision_count,
            communication_failure_count=_integer(
                _mapping(payload.get("communication"), "communication").get("dropped_count", 0)
                if isinstance(payload.get("communication"), Mapping)
                else 0,
                "communication_failure_count",
            ),
            energy_used_j=max(
                0.0,
                finite_number(
                    metric.get("energy_j", execution.get("total_energy_used_j", 0.0)),
                    "energy_used_j",
                ),
            ),
            reads_private_truth=False,
            oracle_only=False,
            ranked=True,
            fragment_counts=(
                (
                    "planned_fragments",
                    _integer(
                        execution.get("planned_fragment_count", payload.get("candidate_limit", 1)),
                        "planned_fragment_count",
                        minimum=1,
                    ),
                ),
                (
                    "executed_fragments",
                    _integer(
                        execution.get("executed_fragment_count", 0), "executed_fragment_count"
                    ),
                ),
                ("failed_fragments", failed_fragments),
                (
                    "outcome_count",
                    _integer(
                        execution.get(
                            "outcome_count",
                            len(execution.get("outcome_hashes", []))
                            if isinstance(execution.get("outcome_hashes"), list)
                            else 0,
                        ),
                        "outcome_count",
                    ),
                ),
            ),
        )

    def to_preflight_row(
        self, *, budget_sha256: str, sensor_profile_sha256: str
    ) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "deployed": True,
            "reads_private_truth": False,
            "oracle_only": False,
            "ranked": True,
            "budget_sha256": _sha(budget_sha256, "budget_sha256"),
            "sensor_profile_sha256": _sha(sensor_profile_sha256, "sensor_profile_sha256"),
            "public_contract_sha256": self.public_contract_sha256,
            "candidate_pool_schema_version": self.candidate_pool_schema_version,
            "task_reservation_schema_version": self.task_reservation_schema_version,
            "evaluation_denominator_sha256": self.evaluation_denominator_sha256,
            "evaluation_geometry_denominator_sha256": self.evaluation_geometry_denominator_sha256,
            "communication_contract_sha256": self.communication_contract_sha256,
            "communication_contract_passed": self.communication_contract_passed,
            "relay_connected_telemetry_sample_fraction": (
                self.relay_connected_telemetry_sample_fraction
            ),
            "delivered_recipient_fraction": self.delivered_recipient_count
            / self.expected_recipient_outcomes,
            "planned": self.planned,
            "executed": self.executed,
            "failed": self.failed,
            "timeout": self.timeout,
            "oom": self.oom,
            "other_failed": self.other_failed,
            "explored_free_flight_volume_auc_time": self.explored_free_flight_volume_auc_time,
            "final_coverage_at_budget": self.final_coverage_at_budget,
            "collision_count": self.collision_count,
            "communication_failure_count": self.communication_failure_count,
            "energy_used_j": self.energy_used_j,
            "elapsed_physics_s": self.elapsed_physics_s,
            "action_budget_utilization": self.action_budget_utilization,
        }

    def public_diagnostics(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "raw_record_sha256": self.raw_record_sha256,
            "public_contract_sha256": self.public_contract_sha256,
            "candidate_pool_schema_version": self.candidate_pool_schema_version,
            "task_reservation_schema_version": self.task_reservation_schema_version,
            "evaluation_denominator_sha256": self.evaluation_denominator_sha256,
            "evaluation_geometry_denominator_sha256": self.evaluation_geometry_denominator_sha256,
            "metric": PRIMARY_METRIC,
            "auc": self.explored_free_flight_volume_auc_time,
            "final_coverage": self.final_coverage_at_budget,
            "elapsed_physics_s": self.elapsed_physics_s,
            "action_budget_utilization": self.action_budget_utilization,
            "terminal_outcome": self.terminal_outcome,
            "fragment_counts": dict(self.fragment_counts),
        }


def assemble_p07_task_validity_pilot(
    probes: Sequence[P07ProbeRecord],
    *,
    matrix_run_id: str,
    sensor_profile_sha256: str,
    communication_contract_sha256: str,
) -> dict[str, object]:
    """Assemble one paired development matrix; it never closes P07 by itself."""

    require_identifier(matrix_run_id, "matrix_run_id")
    sensor_hash = _sha(sensor_profile_sha256, "sensor_profile_sha256")
    communication_hash = _sha(communication_contract_sha256, "communication_contract_sha256")
    rows = tuple(probes)
    by_method = {row.method_id: row for row in rows}
    if len(by_method) != len(rows) or tuple(by_method) != TASK_VALIDITY_METHODS:
        raise ValueError("P07 pilot requires each frozen method exactly once and in frozen order")
    pairing_fields = (
        "partition",
        "scene_id",
        "public_episode_id",
        "public_context_hash",
        "public_candidate_pool_hash",
        "candidate_pool_schema_version",
        "task_reservation_schema_version",
        "public_contract_sha256",
        "evaluation_denominator_sha256",
        "evaluation_geometry_denominator_sha256",
        "fleet_size",
        "action_budget_s",
        "candidate_limit",
        "physics_dt_s",
        "outcome_time_tolerance_s",
        "communication_contract_sha256",
        "communication_mode",
    )
    anchor = rows[0]
    for row in rows[1:]:
        drift = [name for name in pairing_fields if getattr(row, name) != getattr(anchor, name)]
        if drift:
            raise ValueError(f"P07 pilot is not a paired task matrix; drifted={drift}")
    if anchor.communication_contract_sha256 != communication_hash:
        raise ValueError("P07 assembler communication hash differs from worker contract")
    budget_payload = {
        "fleet_size": anchor.fleet_size,
        "action_budget_s": anchor.action_budget_s,
        "candidate_limit": anchor.candidate_limit,
        "physics_dt_s": anchor.physics_dt_s,
        "outcome_time_tolerance_s": anchor.outcome_time_tolerance_s,
        "sensor_profile_sha256": sensor_hash,
        "communication_contract_sha256": communication_hash,
    }
    budget_sha256 = canonical_sha256(budget_payload)
    preflight_payload = {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": matrix_run_id,
        "runtime_command_sha256": canonical_sha256(
            {
                "matrix_run_id": matrix_run_id,
                "raw_record_sha256": [row.raw_record_sha256 for row in rows],
            }
        ),
        "partition": anchor.partition,
        "budget_sha256": budget_sha256,
        "sensor_profile_sha256": sensor_hash,
        "public_contract_sha256": anchor.public_contract_sha256,
        "evaluation_denominator_sha256": anchor.evaluation_denominator_sha256,
        "evaluation_geometry_denominator_sha256": anchor.evaluation_geometry_denominator_sha256,
        "primary_metric": PRIMARY_METRIC,
        "rows": [
            row.to_preflight_row(budget_sha256=budget_sha256, sensor_profile_sha256=sensor_hash)
            for row in rows
        ],
        "task_validity_passed": False,
    }
    return {
        "schema_version": P07_MATRIX_PILOT_SCHEMA_VERSION,
        "status": "P07_TASK_VALIDITY_PILOT_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "p07_task_validity_closed": False,
        "claim_limit": (
            "Paired public-observation pilot only; independent multi-scene evidence "
            "is required before P07 closure."
        ),
        "matrix_identity": {
            "matrix_run_id": matrix_run_id,
            "scene_id": anchor.scene_id,
            "partition": anchor.partition,
            "public_episode_id": anchor.public_episode_id,
            "public_context_hash": anchor.public_context_hash,
            "public_candidate_pool_hash": anchor.public_candidate_pool_hash,
            "candidate_pool_schema_version": anchor.candidate_pool_schema_version,
            "task_reservation_schema_version": anchor.task_reservation_schema_version,
            "public_contract_sha256": anchor.public_contract_sha256,
            "evaluation_denominator_sha256": anchor.evaluation_denominator_sha256,
            "evaluation_geometry_denominator_sha256": anchor.evaluation_geometry_denominator_sha256,
        },
        "budget_contract": {**budget_payload, "budget_sha256": budget_sha256},
        "preflight_payload_candidate": preflight_payload,
        "raw_probe_diagnostics": [row.public_diagnostics() for row in rows],
    }


__all__ = ["P07_MATRIX_PILOT_SCHEMA_VERSION", "P07ProbeRecord", "assemble_p07_task_validity_pilot"]
