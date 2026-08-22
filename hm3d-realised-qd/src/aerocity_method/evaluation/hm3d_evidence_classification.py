"""Classify P07 runtime evidence by field and permitted downstream use.

One runtime file can remain useful for controller calibration while being
ineligible for coverage, QD, or RL claims.  This module keeps those decisions
explicit and prevents a single defect from either discarding all evidence or
silently contaminating training.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from aerocity_method.contracts.io import canonical_sha256, require_sha256
from aerocity_method.contracts.hm3d_public_schema import (
    PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    require_current_public_schema,
)

P07_EVIDENCE_INTEGRITY_SCHEMA_VERSION = "hm3d-p07-evidence-integrity-v1"
P07_EVIDENCE_CLASSIFICATION_SCHEMA_VERSION = "hm3d-p07-evidence-classification-v4"

PUBLIC_RANGE_QUERY_CONTRACT = "static_hm3d_collision_only_dynamic_agents_skipped_v1"
PUBLIC_CANDIDATE_SOURCE_CONTRACT = "public_sparse_voxel_frontier_and_guard_v1"
DECISION_REWARD_CONTRACT = "time_integral_increment_conservation_v1"
REALISED_QD_SOURCE_CONTRACT = "post_execution_public_outcome_v1"
INITIAL_MAP_CONTRACT = "episode_local_empty_then_bootstrap_outcomes_v1"
REAL_EXECUTION_EVIDENCE_CLASS = "real_isaac_physx_cf2x"
P07_RECORD_PURPOSES = frozenset(
    {"engineering_smoke", "train_outcome", "qd_calibration"}
)
P07_RECORD_PURPOSE_ALIASES = {
    "engineering_receipt": "engineering_smoke",
    "train_receipt": "train_outcome",
    "qd_receipt": "qd_calibration",
}
P07_RECORD_PURPOSE_ALIASES = {
    "engineering_receipt": "engineering_smoke",
    "train_receipt": "train_outcome",
    "qd_receipt": "qd_calibration",
}

_KNOWN_DEFECT_FIELDS = {
    "PUBLIC_RANGE_DYNAMIC_BODY_CONTAMINATION": {
        "coverage_metric",
        "decision_reward",
        "realised_qd_descriptor",
        "rl_transition",
        "formal_performance",
    },
    "CANDIDATE_ENDPOINT_NOT_PUBLICLY_OBSERVED": {
        "decision_reward",
        "realised_qd_descriptor",
        "rl_transition",
        "formal_performance",
    },
    "INITIAL_PUBLIC_MAP_UNPAIRED": {"paired_method_comparison", "formal_performance"},
    "DEVELOPMENT_40_SECOND_BUDGET": {"formal_performance"},
    "QD_DESCRIPTOR_DEGENERATE": {"realised_qd_descriptor", "qd_mechanism_claim"},
    "RUNTIME_RECORD_HASH_INVALID": {
        "coverage_metric",
        "decision_reward",
        "realised_qd_descriptor",
        "rl_transition",
        "execution_dynamics",
        "formal_performance",
        "paired_method_comparison",
        "qd_mechanism_claim",
    },
    "NO_REAL_PHYSX_EXECUTION": {
        "coverage_metric",
        "decision_reward",
        "realised_qd_descriptor",
        "rl_transition",
        "execution_dynamics",
        "formal_performance",
        "paired_method_comparison",
        "qd_mechanism_claim",
    },
}


def build_current_evidence_integrity_contract(
    *, runner_source_sha256: str, execution_source_sha256: str
) -> dict[str, Any]:
    """Bind a new worker record to the current auditable implementation contracts."""

    return {
        "schema_version": P07_EVIDENCE_INTEGRITY_SCHEMA_VERSION,
        "real_execution_evidence_class": REAL_EXECUTION_EVIDENCE_CLASS,
        "public_range_query_contract": PUBLIC_RANGE_QUERY_CONTRACT,
        "public_candidate_source_contract": PUBLIC_CANDIDATE_SOURCE_CONTRACT,
        "decision_reward_contract": DECISION_REWARD_CONTRACT,
        "realised_qd_source_contract": REALISED_QD_SOURCE_CONTRACT,
        "initial_map_contract": INITIAL_MAP_CONTRACT,
        "candidate_pool_schema_version": PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
        "task_reservation_schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
        "runner_source_sha256": require_sha256(
            runner_source_sha256, "runner_source_sha256"
        ),
        "execution_source_sha256": require_sha256(
            execution_source_sha256, "execution_source_sha256"
        ),
        "known_defects": [],
    }


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_hash_valid(payload: Mapping[str, Any]) -> bool:
    supplied = payload.get("runtime_record_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        return False
    unsigned = dict(payload)
    unsigned.pop("runtime_record_sha256", None)
    return canonical_sha256(unsigned) == supplied


def _integrity_contract_reasons(payload: Mapping[str, Any]) -> list[str]:
    contract = _mapping(payload.get("evidence_integrity_contract"))
    if contract is None:
        return ["MISSING_EVIDENCE_INTEGRITY_CONTRACT"]
    expected = {
        "schema_version": P07_EVIDENCE_INTEGRITY_SCHEMA_VERSION,
        "real_execution_evidence_class": REAL_EXECUTION_EVIDENCE_CLASS,
        "public_range_query_contract": PUBLIC_RANGE_QUERY_CONTRACT,
        "public_candidate_source_contract": PUBLIC_CANDIDATE_SOURCE_CONTRACT,
        "decision_reward_contract": DECISION_REWARD_CONTRACT,
        "realised_qd_source_contract": REALISED_QD_SOURCE_CONTRACT,
        "initial_map_contract": INITIAL_MAP_CONTRACT,
        "candidate_pool_schema_version": PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
        "task_reservation_schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    }
    reasons = [
        f"EVIDENCE_CONTRACT_MISMATCH:{name}"
        for name, value in expected.items()
        if contract.get(name) != value
    ]
    for name in ("runner_source_sha256", "execution_source_sha256"):
        try:
            require_sha256(contract.get(name), name)
        except (TypeError, ValueError):
            reasons.append(f"INVALID_EVIDENCE_SOURCE_HASH:{name}")
    embedded_defects = contract.get("known_defects")
    if not isinstance(embedded_defects, list) or any(
        not isinstance(item, str) for item in embedded_defects
    ):
        reasons.append("INVALID_EMBEDDED_KNOWN_DEFECTS")
    return reasons


def _public_schema_reasons(payload: Mapping[str, Any]) -> list[str]:
    """Keep old lifecycle outcomes useful for dynamics, but out of selection evidence."""

    reasons: list[str] = []
    try:
        require_current_public_schema(payload, context="P07 record")
    except ValueError:
        reasons.append("PUBLIC_TASK_LIFECYCLE_SCHEMA_MISMATCH")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return reasons + ["PUBLIC_TASK_DECISION_SCHEMA_MISSING"]
    for index, raw_decision in enumerate(decisions):
        if not isinstance(raw_decision, Mapping):
            reasons.append(f"PUBLIC_TASK_DECISION_SCHEMA_MALFORMED:{index}")
            continue
        try:
            require_current_public_schema(raw_decision, context=f"decision[{index}]")
        except ValueError:
            reasons.append(f"PUBLIC_TASK_DECISION_SCHEMA_MISMATCH:{index}")
    return reasons


def _real_execution_reasons(payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    # A failed worker can still provide valid controller, trajectory, speed,
    # collision and sensor evidence.  Completion is deliberately checked by
    # _successful_execution_reasons for score/training uses instead of here.
    if payload.get("status") not in {
        "P07_EXECUTION_SMOKE_COMPLETE",
        "P07_EXECUTION_SMOKE_FAILED",
    }:
        reasons.append("UNKNOWN_P07_EXECUTION_STATUS")
    if payload.get("synthetic") is not False:
        reasons.append("SYNTHETIC_OR_UNKNOWN_EXECUTION")
    elapsed = _finite(payload.get("elapsed_physics_s"))
    if elapsed is None or elapsed <= 0.0:
        reasons.append("MISSING_POSITIVE_PHYSICS_TIME")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        reasons.append("MISSING_EXECUTED_DECISIONS")
        return reasons
    for index, raw_decision in enumerate(decisions):
        decision = _mapping(raw_decision)
        execution = None if decision is None else _mapping(decision.get("execution"))
        if execution is None:
            reasons.append(f"MISSING_DECISION_EXECUTION:{index}")
            continue
        if execution.get("evidence_class") != REAL_EXECUTION_EVIDENCE_CLASS:
            reasons.append(f"NON_PHYSX_DECISION_EXECUTION:{index}")
        outcome_hashes = execution.get("outcome_hashes")
        trace_hashes = execution.get("trace_hashes")
        if not isinstance(outcome_hashes, list) or not outcome_hashes:
            reasons.append(f"MISSING_DECISION_OUTCOMES:{index}")
        if not isinstance(trace_hashes, (list, Mapping)) or not trace_hashes:
            reasons.append(f"MISSING_DECISION_TRACES:{index}")
    return reasons


def _successful_execution_reasons(
    payload: Mapping[str, Any], physical_reasons: Sequence[str]
) -> list[str]:
    """Return reasons a real execution cannot enter score or learning data.

    This is intentionally stricter than _real_execution_reasons.  A legacy
    record may claim COMPLETE while its terminal outcome contains a failure;
    the outcome remains useful for engineering diagnostics but is rejected
    from coverage, QD, replay and formal comparisons.
    """

    reasons = list(physical_reasons)
    if payload.get("status") != "P07_EXECUTION_SMOKE_COMPLETE":
        reasons.append("P07_EXECUTION_NOT_COMPLETE")
    if payload.get("terminal_outcome") != "budget_exhausted":
        reasons.append("TERMINAL_OUTCOME_NOT_BUDGET_EXHAUSTED")

    execution = _mapping(payload.get("execution"))
    if execution is None:
        reasons.append("MISSING_EPISODE_EXECUTION_SUMMARY")
    else:
        failed = _finite(execution.get("failed_fragment_count"))
        if failed is None:
            reasons.append("MISSING_FAILED_FRAGMENT_COUNT")
        elif failed != 0.0:
            reasons.append("FAILED_FRAGMENTS_PRESENT")

    stationarity = _mapping(payload.get("stationarity_supervision"))
    if stationarity is None:
        # A successful status without the episode-level stationarity audit is
        # an incomplete/legacy outcome.  It may still support dynamics
        # diagnostics, but it cannot cross the score or replay boundary.
        reasons.append("STATIONARITY_SUPERVISION_MISSING")
    elif stationarity.get("status") != (
        "EPISODE_STATIONARITY_SUPERVISION_ADMITTED"
    ):
        reasons.append("STATIONARITY_SUPERVISION_NOT_ADMITTED")
    return reasons


def _reward_conservation_reasons(payload: Mapping[str, Any]) -> list[str]:
    decisions = payload.get("decisions")
    metric = _mapping(payload.get("metric_report"))
    if not isinstance(decisions, list) or not decisions or metric is None:
        return ["MISSING_REWARD_CONSERVATION_INPUT"]
    target = _finite(metric.get("explored_free_flight_volume_auc_time"))
    if target is None:
        return ["MISSING_EPISODE_AUC"]
    contributions: list[float] = []
    for index, raw_decision in enumerate(decisions):
        decision = _mapping(raw_decision)
        value = None if decision is None else _finite(
            decision.get("reward_explored_free_flight_volume_auc_time_contribution")
        )
        if value is None:
            return [f"MISSING_DECISION_AUC_CONTRIBUTION:{index}"]
        contributions.append(value)
    if not math.isclose(sum(contributions), target, rel_tol=0.0, abs_tol=1.0e-9):
        return ["DECISION_AUC_CONTRIBUTIONS_DO_NOT_SUM_TO_EPISODE_AUC"]
    return []


def _transition_reasons(payload: Mapping[str, Any]) -> list[str]:
    decisions = payload.get("decisions")
    transitions = payload.get("single_rl_training_transitions")
    if not isinstance(decisions, list) or not isinstance(transitions, list):
        return ["MISSING_DECISION_LEVEL_RL_TRANSITIONS"]
    if len(transitions) != len(decisions):
        return ["RL_TRANSITION_COUNT_DIFFERS_FROM_DECISIONS"]
    reasons: list[str] = []
    for index, (raw_transition, raw_decision) in enumerate(
        zip(transitions, decisions, strict=True)
    ):
        transition = _mapping(raw_transition)
        decision = _mapping(raw_decision)
        if transition is None or decision is None:
            reasons.append(f"MALFORMED_RL_TRANSITION:{index}")
            continue
        supplied = transition.get("transition_sha256")
        unsigned = dict(transition)
        unsigned.pop("transition_sha256", None)
        if not isinstance(supplied, str) or canonical_sha256(unsigned) != supplied:
            reasons.append(f"INVALID_RL_TRANSITION_HASH:{index}")
        for field in (
            "decision_id",
            "public_context_hash",
            "public_candidate_pool_hash",
            "candidate_pool_schema_version",
            "task_reservation_schema_version",
        ):
            if transition.get(field) != decision.get(field):
                reasons.append(f"RL_TRANSITION_DECISION_BINDING_MISMATCH:{index}:{field}")
    return reasons


def _record_purpose(payload: Mapping[str, Any]) -> str | None:
    """Return the declared downstream-use class without guessing from a claim string."""

    return normalize_p07_record_purpose(payload)


def normalize_p07_record_purpose(payload: Mapping[str, Any]) -> str | None:
    """Normalize legacy record purpose names without mutating the source record."""

    purpose = payload.get("record_purpose")
    purpose = P07_RECORD_PURPOSE_ALIASES.get(purpose, purpose)
    return purpose if purpose in P07_RECORD_PURPOSES else None


def _entry(eligible: bool, reasons: Sequence[str]) -> dict[str, Any]:
    return {"eligible": bool(eligible), "reasons": sorted(set(reasons))}


def audit_p07_record_evidence(
    payload: Mapping[str, Any], *, known_defects: Sequence[str] = ()
) -> dict[str, Any]:
    """Return field-level and use-level eligibility for one P07 record."""

    defects = set(known_defects)
    contract = _mapping(payload.get("evidence_integrity_contract"))
    if contract is not None and isinstance(contract.get("known_defects"), list):
        defects.update(item for item in contract["known_defects"] if isinstance(item, str))
    unknown_defects = sorted(defects.difference(_KNOWN_DEFECT_FIELDS))
    hash_reasons = [] if _record_hash_valid(payload) else ["RUNTIME_RECORD_HASH_INVALID"]
    contract_reasons = _integrity_contract_reasons(payload)
    public_schema_reasons = _public_schema_reasons(payload)
    execution_reasons = _real_execution_reasons(payload)
    successful_execution_reasons = _successful_execution_reasons(
        payload, execution_reasons
    )
    reward_reasons = _reward_conservation_reasons(payload)
    transition_reasons = _transition_reasons(payload)
    record_purpose = _record_purpose(payload)
    train_purpose_reasons = (
        []
        if record_purpose == "train_outcome"
        else ["RECORD_PURPOSE_NOT_TRAIN_OUTCOME"]
    )
    qd_purpose_reasons = (
        []
        if record_purpose in {"train_outcome", "qd_calibration"}
        else ["RECORD_PURPOSE_EXCLUDES_QD_HISTORY"]
    )

    common_metric_reasons = (
        hash_reasons + contract_reasons + successful_execution_reasons
    )
    fields: dict[str, dict[str, Any]] = {}
    base_field_reasons = {
        "coverage_metric": common_metric_reasons,
        "decision_reward": common_metric_reasons + reward_reasons,
        "realised_qd_descriptor": (
            common_metric_reasons
            + reward_reasons
            + qd_purpose_reasons
            + public_schema_reasons
        ),
        "rl_transition": (
            common_metric_reasons
            + reward_reasons
            + transition_reasons
            + train_purpose_reasons
            + public_schema_reasons
        ),
        "execution_dynamics": hash_reasons + execution_reasons,
        "paired_method_comparison": common_metric_reasons + public_schema_reasons,
        "formal_performance": common_metric_reasons + public_schema_reasons,
        "qd_mechanism_claim": (
            common_metric_reasons + qd_purpose_reasons + public_schema_reasons
        ),
    }
    for name, reasons in base_field_reasons.items():
        affected = sorted(defect for defect in defects if name in _KNOWN_DEFECT_FIELDS[defect])
        combined = list(reasons) + affected + [f"UNKNOWN_DEFECT:{row}" for row in unknown_defects]
        fields[name] = _entry(not combined, combined)

    formal_reasons = list(fields["formal_performance"]["reasons"])
    if payload.get("formal_result") is not True:
        formal_reasons.append("NOT_A_FROZEN_FORMAL_RESULT")
    if _finite(payload.get("action_budget_s")) != 240.0:
        formal_reasons.append("NOT_THE_FROZEN_240_SECOND_FORMAL_BUDGET")
    fields["formal_performance"] = _entry(not formal_reasons, formal_reasons)

    train_reasons = list(fields["rl_transition"]["reasons"])
    if payload.get("selection_partition") != "train":
        train_reasons.append("NOT_A_TRAIN_PARTITION_RECORD")
    if payload.get("calibration_only_timeout_probe") is True:
        train_reasons.append("TIMEOUT_PROBE_CANNOT_TRAIN")
    use_cases = {
        "formal_performance_evidence": fields["formal_performance"],
        "trainable_real_outcome": _entry(not train_reasons, train_reasons),
        "dynamics_calibration_evidence": fields["execution_dynamics"],
        "engineering_diagnostic_evidence": _entry(not hash_reasons, hash_reasons),
    }
    completely_unusable = not any(row["eligible"] for row in use_cases.values())
    result = {
        "schema_version": P07_EVIDENCE_CLASSIFICATION_SCHEMA_VERSION,
        "runtime_record_sha256": payload.get("runtime_record_sha256"),
        "record_purpose": record_purpose,
        "known_defects": sorted(defects),
        "unknown_defects": unknown_defects,
        "fields": fields,
        "use_cases": use_cases,
        "completely_unusable": completely_unusable,
    }
    result["classification_sha256"] = canonical_sha256(result)
    return result


def require_trainable_p07_outcome(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the record is eligible for decision-level training."""

    audit = audit_p07_record_evidence(payload)
    eligibility = audit["use_cases"]["trainable_real_outcome"]
    if eligibility["eligible"] is not True:
        raise ValueError(
            "P07 record is not eligible for real-outcome training: "
            + ", ".join(eligibility["reasons"])
        )
    return audit


def require_p07_evidence_field(
    payload: Mapping[str, Any], field_name: str
) -> dict[str, Any]:
    """Fail closed unless one specific evidence field is independently eligible."""

    audit = audit_p07_record_evidence(payload)
    fields = audit["fields"]
    if field_name not in fields:
        raise ValueError(f"unknown P07 evidence field: {field_name}")
    eligibility = fields[field_name]
    if eligibility["eligible"] is not True:
        raise ValueError(
            f"P07 {field_name} evidence is ineligible: "
            + ", ".join(eligibility["reasons"])
        )
    return audit


__all__ = [
    "DECISION_REWARD_CONTRACT",
    "INITIAL_MAP_CONTRACT",
    "P07_EVIDENCE_CLASSIFICATION_SCHEMA_VERSION",
    "P07_EVIDENCE_INTEGRITY_SCHEMA_VERSION",
    "P07_RECORD_PURPOSES",
    "P07_RECORD_PURPOSE_ALIASES",
    "PUBLIC_CANDIDATE_SOURCE_CONTRACT",
    "PUBLIC_RANGE_QUERY_CONTRACT",
    "REALISED_QD_SOURCE_CONTRACT",
    "audit_p07_record_evidence",
    "build_current_evidence_integrity_contract",
    "normalize_p07_record_purpose",
    "require_p07_evidence_field",
    "require_trainable_p07_outcome",
]
