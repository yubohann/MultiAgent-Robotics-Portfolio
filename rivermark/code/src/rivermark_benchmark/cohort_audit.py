"""Receipt-bound audit for a frozen development capture cohort.

This module is deliberately read-only. It summarizes independently validated
development captures without admitting episodes, copying payloads, or exposing
evaluator-private paths and target coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .collection_protocol import (
    COLLECTION_BINDING_KEYS,
    COLLECTION_SPLITS,
    CollectionProtocolError,
    coverage_report,
    load_collection_protocol,
    protocol_sha256,
    resolve_collection_binding,
)
from .failure_ledger import FailureLedgerError, load_failure_ledger
from .formal_dataset import sha256_file

COHORT_AUDIT_SCHEMA = "org.rivermark.benchmark.development-cohort-audit.v1"
_CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
_VALIDATION_SCHEMA = "org.rivermark.isaac-independent-validation.v1"
_TASK_OUTCOME_SCHEMA = "org.rivermark.t1-target-observability.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_WINDOWS_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\)")
_REPORT_KEYS = frozenset(
    {
        "schema",
        "status",
        "formal",
        "policy_ranking",
        "audit_provenance",
        "protocol",
        "source",
        "accounting",
        "candidates",
        "aggregate",
        "admission_readiness",
        "report_payload_sha256",
    }
)
_PROTOCOL_KEYS = frozenset(
    {
        "protocol_id",
        "protocol_sha256",
        "candidate_index_start",
        "expected_candidate_count",
        "target_quota_by_cell",
    }
)
_SOURCE_KEYS = frozenset({"revision", "all_worktrees_clean", "source_tree_sha256_values"})
_ACCOUNTING_KEYS = frozenset(
    {
        "failure_ledger_sha256",
        "ledger_record_count",
        "protocol_attempt_count",
        "candidate_attempt_count",
        "noncandidate_protocol_attempt_count",
        "candidate_outcomes",
        "candidate_reason_codes",
        "candidate_selection_sha256",
        "coverage",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "capture_attempt_id",
        "binding",
        "source_revision",
        "source_tree_sha256",
        "capture_receipt_sha256",
        "independent_validation_sha256",
        "task_outcome_sha256",
        "quality_gates",
        "target_visibility",
        "contact_free",
        "minimum_agent_max_displacement_m",
        "route_witness_displacement_m",
        "resources",
        "ledger",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "capture_bytes",
        "duration_s",
        "peak_system_commit_percent",
        "peak_process_private_commit_bytes",
    }
)
_AGGREGATE_KEYS = frozenset(
    {
        "candidate_count",
        "candidate_count_by_split",
        "quality_gate_pass_count",
        "contact_free_count",
        "target_count",
        "targets_meeting_visibility",
        "target_visibility_rate",
        "capture_failure_rate",
        "quarantine_rate",
        "formal_admission_rate",
        "capture_bytes",
        "duration_s",
        "peak_system_commit_percent",
        "peak_process_private_commit_bytes",
        "minimum_agent_max_displacement_m",
        "route_witness_displacement_m",
    }
)
_READINESS_KEYS = frozenset(
    {
        "native_evidence_ready_for_review",
        "formal_admission_complete",
        "release_ready",
        "blocking_reason_codes",
        "not_evaluated_by_this_audit",
    }
)
_LEDGER_KEYS = frozenset({"outcome", "category", "stage", "reason_code"})
_DISTRIBUTION_KEYS = frozenset({"count", "minimum", "median", "mean", "maximum"})
_COVERAGE_BASE_KEYS = frozenset(
    {
        "schema",
        "protocol_id",
        "protocol_sha256",
        "failure_ledger_schema",
        "ledger_record_count",
        "excluded_ledger_record_count",
        "excluded_protocol_id_count",
        "excluded_protocol_hash_count",
        "attempts_sha256",
        "attempt_count",
        "admitted_count",
        "quarantined_count",
        "failed_count",
        "exclusion_reasons",
        "cells",
        "complete",
    }
)
_COVERAGE_CELL_KEYS = frozenset(
    {
        "cell_id",
        "split",
        "conditions",
        "attempt_count",
        "admitted_count",
        "quarantined_count",
        "failed_count",
        "exclusion_reasons",
        "minimum_attempts",
        "minimum_admitted",
        "status",
    }
)
_T1_QUOTA_KEYS = frozenset(
    {
        "basis",
        "statistical_unit",
        "policy_ranking",
        "initial_admitted_episode_target",
        "admitted_episodes",
        "quota_target_met",
    }
)
_POWER_ANALYSIS_KEYS = frozenset(
    {
        "method",
        "primary_metric",
        "evaluation_split",
        "required_evaluation_episodes",
        "admitted_evaluation_episodes",
        "power_target_met",
    }
)
_NOT_EVALUATED = frozenset(
    {
        "distribution_clearance",
        "formal_candidate_pack",
        "operator_receipt_allowlist",
        "release_supply_chain",
    }
)
# Collection protocol v2 encodes the numeric quota in a condition identifier.
# Keep the supported frozen vocabulary explicit until a future protocol adds a
# structured numeric field.
_TARGET_COUNT_BY_CONDITION = {"object-count-4-v1": 4}
_ROUTE_WITNESS_MIN_DISPLACEMENT_M = 3.0
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "capture_path",
        "capture_root",
        "evaluator_private_manifest",
        "private_manifest_path",
        "target_coordinates",
        "target_position_w",
    }
)
_QUALITY_GATE_CHECKS = {
    "independent_validation_passed": (),
    "sensor_timestamps_synchronized": (
        "timestamp_audit_passed",
        "sensor_phase_trace_verified",
    ),
    "action_before_step_causality": ("action_causality_audit_passed",),
    "camera_pose_closure": ("pose_closure_audit_passed",),
    "visual_intrusion_absent": (
        "visual_intrusion_verified",
        "onboard_scene_content_verified",
    ),
    "physical_safety_passed": (
        "contact_free",
        "runtime_safety_trace_verified",
        "trajectory_segment_clearance_verified",
    ),
    "condition_realization_passed": ("condition_realization_verified",),
    "artifact_hash_binding_passed": (),
}
_DEVELOPMENT_LEDGER_STATE = {
    "outcome": "quarantined",
    "category": "quality_failure",
    "stage": "isaac_capture",
    "reason_code": "development_evidence_not_formal",
}
_TARGET_VISIBILITY_KEYS = {
    "target_count",
    "targets_meeting_visibility",
    "failed_target_count",
    "minimum_visible_frames",
    "maximum_visible_frames",
    "minimum_max_pixels",
    "maximum_max_pixels",
}


class CohortAuditError(ValueError):
    """Raised when development evidence cannot support a cohort audit."""


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise CohortAuditError(f"{label} structure is malformed")


def _protocol_target_quotas(protocol: Mapping[str, Any]) -> dict[str, dict[str, int | str]]:
    quotas: dict[str, dict[str, int | str]] = {}
    for cell in protocol.get("cells", []):
        if not isinstance(cell, Mapping) or not isinstance(cell.get("conditions"), Mapping):
            raise CohortAuditError("collection protocol cell conditions are malformed")
        cell_id = cell.get("cell_id")
        condition_id = cell["conditions"].get("target_count")
        if not isinstance(cell_id, str) or condition_id not in _TARGET_COUNT_BY_CONDITION:
            raise CohortAuditError("collection protocol target-count condition is unsupported")
        quotas[cell_id] = {
            "condition_id": str(condition_id),
            "target_count": _TARGET_COUNT_BY_CONDITION[str(condition_id)],
        }
    if not quotas:
        raise CohortAuditError("collection protocol target quotas are missing")
    return dict(sorted(quotas.items()))


def _reason_counts(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise CohortAuditError(f"{label} is malformed")
    counts: dict[str, int] = {}
    for reason, count in value.items():
        if (
            not isinstance(reason, str)
            or not _PUBLIC_ID.fullmatch(reason)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise CohortAuditError(f"{label} is malformed")
        counts[reason] = count
    return counts


def _verify_embedded_coverage(
    coverage: Mapping[str, Any],
    *,
    protocol_id: str,
    protocol_digest: str,
    target_quotas: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    schema = coverage.get("schema")
    if schema == "org.rivermark.benchmark.t1-coverage-report.v2":
        _require_exact_keys(
            coverage,
            _COVERAGE_BASE_KEYS | {"quota_analysis"},
            label="cohort coverage",
        )
    elif schema == "org.rivermark.benchmark.coverage-report.v1":
        _require_exact_keys(
            coverage,
            _COVERAGE_BASE_KEYS | {"power_analysis"},
            label="cohort coverage",
        )
    else:
        raise CohortAuditError("cohort coverage schema is unsupported")
    if (
        coverage.get("protocol_id") != protocol_id
        or coverage.get("protocol_sha256") != protocol_digest
        or coverage.get("failure_ledger_schema")
        != "org.rivermark.benchmark.failure-ledger.v1"
        or not isinstance(coverage.get("attempts_sha256"), str)
        or not _SHA256.fullmatch(str(coverage["attempts_sha256"]))
    ):
        raise CohortAuditError("cohort coverage identity is inconsistent")
    count_keys = (
        "ledger_record_count",
        "excluded_ledger_record_count",
        "excluded_protocol_id_count",
        "excluded_protocol_hash_count",
        "attempt_count",
        "admitted_count",
        "quarantined_count",
        "failed_count",
    )
    counts = {
        key: _non_negative_int(coverage.get(key), label=f"coverage.{key}")
        for key in count_keys
    }
    if (
        counts["excluded_ledger_record_count"]
        != counts["excluded_protocol_id_count"] + counts["excluded_protocol_hash_count"]
        or counts["ledger_record_count"]
        != counts["attempt_count"] + counts["excluded_ledger_record_count"]
        or counts["attempt_count"]
        != counts["admitted_count"]
        + counts["quarantined_count"]
        + counts["failed_count"]
    ):
        raise CohortAuditError("cohort coverage global counts are inconsistent")
    cells = coverage.get("cells")
    if not isinstance(cells, list) or not cells or any(
        not isinstance(cell, Mapping) for cell in cells
    ):
        raise CohortAuditError("cohort coverage cells are malformed")
    global_reasons: Counter[str] = Counter()
    cell_totals = Counter[str]()
    cell_splits: dict[str, str] = {}
    for cell in cells:
        _require_exact_keys(cell, _COVERAGE_CELL_KEYS, label="cohort coverage cell")
        cell_id = cell.get("cell_id")
        split = cell.get("split")
        conditions = cell.get("conditions")
        if (
            not isinstance(cell_id, str)
            or not _PUBLIC_ID.fullmatch(cell_id)
            or cell_id in cell_splits
            or split not in COLLECTION_SPLITS
            or not isinstance(conditions, Mapping)
            or not conditions
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not _PUBLIC_ID.fullmatch(value)
                for key, value in conditions.items()
            )
        ):
            raise CohortAuditError("cohort coverage cell identity is malformed")
        quota = target_quotas.get(cell_id)
        if not isinstance(quota, Mapping) or conditions.get("target_count") != quota.get(
            "condition_id"
        ):
            raise CohortAuditError("cohort coverage target quota is inconsistent")
        cell_splits[cell_id] = str(split)
        cell_counts = {
            key: _non_negative_int(cell.get(key), label=f"coverage.cell.{key}")
            for key in ("attempt_count", "admitted_count", "quarantined_count", "failed_count")
        }
        minimum_attempts = _non_negative_int(
            cell.get("minimum_attempts"), label="coverage.cell.minimum_attempts"
        )
        minimum_admitted = _non_negative_int(
            cell.get("minimum_admitted"), label="coverage.cell.minimum_admitted"
        )
        if minimum_attempts <= 0 or minimum_admitted <= 0:
            raise CohortAuditError("cohort coverage cell quota is malformed")
        reasons = _reason_counts(
            cell.get("exclusion_reasons"), label="cohort coverage cell exclusion reasons"
        )
        expected_status = (
            "passed"
            if cell_counts["attempt_count"] >= minimum_attempts
            and cell_counts["admitted_count"] >= minimum_admitted
            else "under_quota"
        )
        if (
            cell_counts["attempt_count"]
            != cell_counts["admitted_count"]
            + cell_counts["quarantined_count"]
            + cell_counts["failed_count"]
            or sum(reasons.values())
            != cell_counts["quarantined_count"] + cell_counts["failed_count"]
            or cell.get("status") != expected_status
        ):
            raise CohortAuditError("cohort coverage cell counts are inconsistent")
        global_reasons.update(reasons)
        cell_totals.update(cell_counts)
    if set(cell_splits) != set(target_quotas):
        raise CohortAuditError("cohort coverage cells disagree with protocol quotas")
    for candidate in candidates:
        cell_id = str(candidate["binding"]["cell_id"])
        if cell_splits.get(cell_id) != candidate["binding"]["split"]:
            raise CohortAuditError("cohort coverage candidate split is inconsistent")
    candidate_counts = Counter(str(item["binding"]["cell_id"]) for item in candidates)
    quarantined_by_cell = {
        str(cell["cell_id"]): int(cell["quarantined_count"]) for cell in cells
    }
    if any(quarantined_by_cell.get(cell_id, 0) < count for cell_id, count in candidate_counts.items()):
        raise CohortAuditError("cohort coverage omits audited quarantined candidates")
    if (
        any(cell_totals[key] != counts[key] for key in ("attempt_count", "admitted_count", "quarantined_count", "failed_count"))
        or dict(sorted(global_reasons.items()))
        != dict(sorted(_reason_counts(coverage.get("exclusion_reasons"), label="cohort coverage exclusion reasons").items()))
    ):
        raise CohortAuditError("cohort coverage cell and global counts are inconsistent")
    all_cells_passed = all(cell.get("status") == "passed" for cell in cells)
    if schema == "org.rivermark.benchmark.t1-coverage-report.v2":
        quota_analysis = coverage.get("quota_analysis")
        if not isinstance(quota_analysis, Mapping):
            raise CohortAuditError("cohort coverage quota analysis is malformed")
        _require_exact_keys(quota_analysis, _T1_QUOTA_KEYS, label="cohort coverage quota analysis")
        target = _non_negative_int(
            quota_analysis.get("initial_admitted_episode_target"),
            label="coverage.initial_admitted_episode_target",
        )
        target_met = counts["admitted_count"] >= target
        if (
            target <= 0
            or not isinstance(quota_analysis.get("basis"), str)
            or not isinstance(quota_analysis.get("statistical_unit"), str)
            or quota_analysis.get("policy_ranking") is not False
            or quota_analysis.get("admitted_episodes") != counts["admitted_count"]
            or quota_analysis.get("quota_target_met") is not target_met
            or coverage.get("complete") is not (target_met and all_cells_passed)
        ):
            raise CohortAuditError("cohort coverage quota analysis is inconsistent")
    else:
        power = coverage.get("power_analysis")
        if not isinstance(power, Mapping):
            raise CohortAuditError("cohort coverage power analysis is malformed")
        _require_exact_keys(power, _POWER_ANALYSIS_KEYS, label="cohort coverage power analysis")
        evaluation_split = power.get("evaluation_split")
        required = _non_negative_int(
            power.get("required_evaluation_episodes"),
            label="coverage.required_evaluation_episodes",
        )
        admitted_evaluation = sum(
            int(cell["admitted_count"]) for cell in cells if cell["split"] == evaluation_split
        )
        target_met = admitted_evaluation >= required
        if (
            evaluation_split not in {"validation", "blind_test", "ood_test"}
            or required < 2
            or power.get("admitted_evaluation_episodes") != admitted_evaluation
            or power.get("power_target_met") is not target_met
            or coverage.get("complete") is not (target_met and all_cells_passed)
        ):
            raise CohortAuditError("cohort coverage power analysis is inconsistent")
    return counts


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CohortAuditError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CohortAuditError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise CohortAuditError(f"{label} must be a finite number >= {minimum}")
    return float(value)


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CohortAuditError(f"{label} must be a non-negative integer")
    return value


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise CohortAuditError("cannot summarize an empty metric distribution")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise CohortAuditError("metric distribution contains a non-finite value")
    return {
        "count": len(normalized),
        "minimum": min(normalized),
        "median": statistics.median(normalized),
        "mean": statistics.fmean(normalized),
        "maximum": max(normalized),
    }


def _verify_receipt_sidecar(root: Path, receipt_sha256: str) -> None:
    sidecar = root / "capture_receipt.sha256"
    try:
        content = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise CohortAuditError(f"capture receipt sidecar is missing or unreadable: {root.name}") from exc
    if content != f"{receipt_sha256}  capture_receipt.json":
        raise CohortAuditError(f"capture receipt sidecar is stale: {root.name}")


def _verify_receipt_artifacts(
    root: Path,
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, int | str]]:
    inventory = receipt.get("artifact_hashes")
    if not isinstance(inventory, Mapping) or not inventory:
        raise CohortAuditError(f"capture receipt artifact inventory is missing: {root.name}")
    verified: dict[str, dict[str, int | str]] = {}
    resolved_inventory_paths: set[str] = set()
    for relative, binding in sorted(inventory.items(), key=lambda item: str(item[0])):
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or ":" in relative
        ):
            raise CohortAuditError(f"capture receipt artifact path is malformed: {root.name}")
        normalized = PurePosixPath(relative)
        if (
            normalized.is_absolute()
            or normalized.as_posix() != relative
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            raise CohortAuditError(f"capture receipt artifact path is unsafe: {relative}")
        if not isinstance(binding, Mapping):
            raise CohortAuditError(f"{relative} is not bound by the capture receipt: {root.name}")
        expected_sha256 = binding.get("sha256")
        expected_bytes = binding.get("bytes")
        if (
            not isinstance(expected_sha256, str)
            or not _SHA256.fullmatch(expected_sha256)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise CohortAuditError(f"{relative} has a malformed capture receipt binding")
        unresolved = root / Path(*normalized.parts)
        probe = root
        for part in normalized.parts:
            probe /= part
            if probe.is_symlink():
                raise CohortAuditError(f"capture artifact is missing or unsafe: {relative}: {root.name}")
        try:
            path = unresolved.resolve(strict=True)
        except OSError as exc:
            raise CohortAuditError(
                f"capture artifact is missing or unsafe: {relative}: {root.name}"
            ) from exc
        if not path.is_relative_to(root) or not path.is_file():
            raise CohortAuditError(f"capture artifact is missing or unsafe: {relative}: {root.name}")
        normalized_path = os.path.normcase(str(path))
        if normalized_path in resolved_inventory_paths:
            raise CohortAuditError(f"capture receipt has a duplicate artifact path: {relative}")
        resolved_inventory_paths.add(normalized_path)
        actual_sha256 = sha256_file(path)
        actual_bytes = path.stat().st_size
        if expected_sha256 != actual_sha256 or expected_bytes != actual_bytes:
            raise CohortAuditError(f"{relative} disagrees with the capture receipt: {root.name}")
        verified[relative] = {"sha256": actual_sha256, "bytes": actual_bytes}
    return verified


def _quality_gates(
    validation: Mapping[str, Any], *, artifact_hash_binding_passed: bool
) -> dict[str, bool]:
    checks = validation.get("checks")
    if not isinstance(checks, Mapping):
        raise CohortAuditError("independent validation checks are missing")
    gates: dict[str, bool] = {}
    for gate, check_names in _QUALITY_GATE_CHECKS.items():
        if gate == "independent_validation_passed":
            gates[gate] = validation.get("status") == "passed" and validation.get("issues") == []
        elif gate == "artifact_hash_binding_passed":
            gates[gate] = artifact_hash_binding_passed
        else:
            gates[gate] = all(checks.get(name) is True for name in check_names)
    return gates


def _target_visibility(checks: Mapping[str, Any]) -> dict[str, Any]:
    visibility = checks.get("target_observability")
    if not isinstance(visibility, Mapping) or visibility.get("passed") is not True:
        raise CohortAuditError("target observability is missing or failed")
    target_count = _non_negative_int(visibility.get("target_count"), label="target_count")
    targets_meeting = _non_negative_int(
        visibility.get("targets_meeting_visibility"), label="targets_meeting_visibility"
    )
    failed_count = _non_negative_int(
        visibility.get("failed_target_count"), label="failed_target_count"
    )
    slots = visibility.get("per_target_slot")
    if (
        target_count <= 0
        or targets_meeting != target_count
        or failed_count != 0
        or not isinstance(slots, Mapping)
        or len(slots) != target_count
    ):
        raise CohortAuditError("target observability counts are inconsistent")
    visible_frames: list[int] = []
    maximum_pixels: list[int] = []
    for slot, value in sorted(slots.items(), key=lambda item: str(item[0])):
        if not isinstance(slot, str) or not isinstance(value, Mapping):
            raise CohortAuditError("target observability slot is malformed")
        visible_frames.append(
            _non_negative_int(value.get("visible_frames"), label=f"{slot}.visible_frames")
        )
        maximum_pixels.append(
            _non_negative_int(value.get("max_pixels"), label=f"{slot}.max_pixels")
        )
    return {
        "target_count": target_count,
        "targets_meeting_visibility": targets_meeting,
        "failed_target_count": failed_count,
        "minimum_visible_frames": min(visible_frames),
        "maximum_visible_frames": max(visible_frames),
        "minimum_max_pixels": min(maximum_pixels),
        "maximum_max_pixels": max(maximum_pixels),
    }


def _resource_summary(
    receipt: Mapping[str, Any], *, receipt_bound_artifact_bytes: int
) -> dict[str, Any]:
    telemetry = receipt.get("resource_telemetry")
    maxima = telemetry.get("maxima") if isinstance(telemetry, Mapping) else None
    if not isinstance(maxima, Mapping):
        raise CohortAuditError("capture resource maxima are missing")
    created = receipt.get("created_wall_time_ns")
    finished = receipt.get("finished_wall_time_ns")
    if (
        isinstance(created, bool)
        or not isinstance(created, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or finished < created
    ):
        raise CohortAuditError("capture wall-time interval is malformed")
    return {
        "capture_bytes": receipt_bound_artifact_bytes,
        "duration_s": (finished - created) / 1_000_000_000.0,
        "peak_system_commit_percent": _finite_number(
            maxima.get("commit_percent"),
            label="peak_system_commit_percent",
        ),
        "peak_process_private_commit_bytes": _non_negative_int(
            maxima.get("private_commit_bytes"),
            label="peak_process_private_commit_bytes",
        ),
    }


def _assert_public_report(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise CohortAuditError(f"report key is not a string at {path}")
            lowered = key.lower()
            if lowered in _FORBIDDEN_REPORT_KEYS or (
                "target" in lowered and ("coordinate" in lowered or "position" in lowered)
            ):
                raise CohortAuditError(f"report contains a forbidden private field at {path}.{key}")
            _assert_public_report(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_public_report(nested, path=f"{path}[{index}]")
    elif isinstance(value, str) and (_WINDOWS_PATH.search(value) or value.startswith("/")):
        raise CohortAuditError(f"report contains an absolute local path at {path}")


def _load_candidate(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    ledger_by_attempt: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    capture_root = root.expanduser().resolve()
    if not capture_root.is_dir():
        raise CohortAuditError(f"capture directory is missing: {root}")
    receipt_path = capture_root / "capture_receipt.json"
    validation_path = capture_root / "independent_validation.json"
    outcome_path = capture_root / "task_outcome.json"
    receipt = _load_json(receipt_path, label="capture receipt")
    validation = _load_json(validation_path, label="independent validation")
    outcome = _load_json(outcome_path, label="task outcome")
    receipt_sha256 = sha256_file(receipt_path)
    validation_sha256 = sha256_file(validation_path)
    _verify_receipt_sidecar(capture_root, receipt_sha256)
    if (
        receipt.get("schema") != _CAPTURE_SCHEMA
        or receipt.get("status") != "captured"
        or receipt.get("ok") is not True
        or receipt.get("source_worktree_dirty") is not False
    ):
        raise CohortAuditError(f"capture is not clean successful evidence: {capture_root.name}")
    verified_artifacts = _verify_receipt_artifacts(capture_root, receipt)
    if "task_outcome.json" not in verified_artifacts:
        raise CohortAuditError(
            f"task_outcome.json is not bound by the capture receipt: {capture_root.name}"
        )
    outcome_sha256 = str(verified_artifacts["task_outcome.json"]["sha256"])
    if (
        outcome.get("schema") != _TASK_OUTCOME_SCHEMA
        or outcome.get("scoring_status") != "not_scored"
    ):
        raise CohortAuditError(f"task outcome identity is invalid: {capture_root.name}")
    if (
        validation.get("schema") != _VALIDATION_SCHEMA
        or validation.get("formal_benchmark_admission") is not False
    ):
        raise CohortAuditError(f"independent validation identity is invalid: {capture_root.name}")
    binding = receipt.get("collection_binding")
    if not isinstance(binding, Mapping):
        raise CohortAuditError(f"capture collection binding is missing: {capture_root.name}")
    cell_id = binding.get("cell_id")
    episode_index = binding.get("episode_index")
    if not isinstance(cell_id, str) or isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise CohortAuditError(f"capture collection binding is malformed: {capture_root.name}")
    expected_binding = resolve_collection_binding(
        protocol,
        cell_id=cell_id,
        episode_index=episode_index,
    )
    if dict(binding) != expected_binding:
        raise CohortAuditError(f"capture collection binding is stale: {capture_root.name}")
    artifact_binding = validation.get("capture_receipt_sha256") == receipt_sha256
    gates = _quality_gates(validation, artifact_hash_binding_passed=artifact_binding)
    if not all(gates.values()):
        failed = ", ".join(sorted(name for name, passed in gates.items() if not passed))
        raise CohortAuditError(f"capture quality gates failed for {capture_root.name}: {failed}")
    checks = validation.get("checks")
    assert isinstance(checks, Mapping)
    outcome_visibility = outcome.get("target_observability")
    validation_visibility = checks.get("target_observability")
    if (
        not isinstance(outcome_visibility, Mapping)
        or not isinstance(validation_visibility, Mapping)
        or _canonical_sha256(outcome_visibility) != _canonical_sha256(validation_visibility)
    ):
        raise CohortAuditError(
            f"task outcome target observability disagrees with independent validation: "
            f"{capture_root.name}"
        )
    attempt_id = receipt.get("capture_attempt_id")
    source_revision = receipt.get("source_revision")
    source_tree_sha256 = receipt.get("source_tree_sha256")
    if not isinstance(attempt_id, str) or attempt_id not in ledger_by_attempt:
        raise CohortAuditError(f"capture attempt is absent from the failure ledger: {capture_root.name}")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise CohortAuditError(f"capture source revision is not a full Git hash: {capture_root.name}")
    if not isinstance(source_tree_sha256, str) or not _SHA256.fullmatch(source_tree_sha256):
        raise CohortAuditError(f"capture source-tree hash is malformed: {capture_root.name}")
    record = ledger_by_attempt[attempt_id]
    expected_record_fields = {
        "collection_protocol_id": binding["protocol_id"],
        "collection_protocol_sha256": binding["protocol_sha256"],
        "collection_cell_id": binding["cell_id"],
        "collection_episode_index": binding["episode_index"],
        "episode_seed": binding["episode_seed"],
        "split": binding["split"],
        "receipt_sha256": receipt_sha256,
        "source_capture_sha256": receipt_sha256,
    }
    mismatches = [
        key for key, expected in expected_record_fields.items() if record.get(key) != expected
    ]
    if mismatches:
        raise CohortAuditError(
            f"capture ledger binding is stale for {capture_root.name}: {', '.join(mismatches)}"
        )
    state_mismatches = [
        key for key, expected in _DEVELOPMENT_LEDGER_STATE.items() if record.get(key) != expected
    ]
    if state_mismatches:
        raise CohortAuditError(
            f"candidate is not a development quarantine for {capture_root.name}: "
            f"{', '.join(state_mismatches)}"
        )
    visibility = _target_visibility(checks)
    resources = _resource_summary(
        receipt,
        receipt_bound_artifact_bytes=sum(
            int(binding["bytes"]) for binding in verified_artifacts.values()
        ),
    )
    minimum_displacement = _finite_number(
        checks.get("minimum_agent_max_displacement_m"),
        label="minimum_agent_max_displacement_m",
    )
    witness_displacement = _finite_number(
        checks.get("route_witness_tracked_agent_max_displacement_m"),
        label="route_witness_tracked_agent_max_displacement_m",
        minimum=_ROUTE_WITNESS_MIN_DISPLACEMENT_M,
    )
    return {
        "capture_attempt_id": attempt_id,
        "binding": expected_binding,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "capture_receipt_sha256": receipt_sha256,
        "independent_validation_sha256": validation_sha256,
        "task_outcome_sha256": outcome_sha256,
        "quality_gates": gates,
        "target_visibility": visibility,
        "contact_free": checks.get("contact_free") is True,
        "minimum_agent_max_displacement_m": minimum_displacement,
        "route_witness_displacement_m": witness_displacement,
        "resources": resources,
        "ledger": {
            "outcome": record["outcome"],
            "category": record["category"],
            "stage": record["stage"],
            "reason_code": record.get("reason_code"),
        },
    }


def build_development_cohort_audit(
    protocol_path: Path,
    failure_ledger_path: Path,
    capture_roots: Sequence[Path],
    *,
    candidate_index_start: int = 1,
) -> dict[str, Any]:
    """Build a public, receipt-bound audit without changing capture state."""

    if (
        isinstance(candidate_index_start, bool)
        or not isinstance(candidate_index_start, int)
        or candidate_index_start < 0
    ):
        raise CohortAuditError("candidate_index_start must be a non-negative integer")
    try:
        protocol = load_collection_protocol(protocol_path)
        ledger_records = load_failure_ledger(failure_ledger_path)
        coverage = coverage_report(protocol, ledger_records)
    except (CollectionProtocolError, FailureLedgerError, OSError, ValueError) as exc:
        raise CohortAuditError(str(exc)) from exc
    ledger_by_attempt = {str(record["attempt_id"]): record for record in ledger_records}
    candidates = [
        _load_candidate(root, protocol=protocol, ledger_by_attempt=ledger_by_attempt)
        for root in capture_roots
    ]
    if not candidates:
        raise CohortAuditError("at least one capture is required")
    attempt_ids = [str(item["capture_attempt_id"]) for item in candidates]
    receipt_hashes = [str(item["capture_receipt_sha256"]) for item in candidates]
    if len(set(attempt_ids)) != len(attempt_ids) or len(set(receipt_hashes)) != len(receipt_hashes):
        raise CohortAuditError("candidate captures must have unique attempts and receipts")
    cells = {str(cell["cell_id"]): cell for cell in protocol["cells"]}
    target_quotas = _protocol_target_quotas(protocol)
    expected_bindings = {
        (cell_id, index)
        for cell_id, cell in cells.items()
        for index in range(
            candidate_index_start,
            candidate_index_start + int(cell["minimum_admitted"]),
        )
    }
    actual_bindings = {
        (str(item["binding"]["cell_id"]), int(item["binding"]["episode_index"]))
        for item in candidates
    }
    if actual_bindings != expected_bindings or len(candidates) != len(expected_bindings):
        missing = sorted(expected_bindings - actual_bindings)
        unexpected = sorted(actual_bindings - expected_bindings)
        raise CohortAuditError(
            f"candidate binding set is incomplete or duplicated; missing={missing}, unexpected={unexpected}"
        )
    for candidate in candidates:
        cell_id = str(candidate["binding"]["cell_id"])
        if int(candidate["target_visibility"]["target_count"]) != int(
            target_quotas[cell_id]["target_count"]
        ):
            raise CohortAuditError(f"candidate target quota disagrees with protocol cell: {cell_id}")
    revisions = sorted({str(item["source_revision"]) for item in candidates})
    if len(revisions) != 1:
        raise CohortAuditError("candidate captures do not share one clean Git revision")
    candidates.sort(key=lambda item: (str(item["binding"]["split"]), int(item["binding"]["episode_index"])))
    outcomes = Counter(str(item["ledger"]["outcome"]) for item in candidates)
    reasons = Counter(
        str(item["ledger"]["reason_code"])
        for item in candidates
        if item["ledger"]["reason_code"] is not None
    )
    target_total = sum(int(item["target_visibility"]["target_count"]) for item in candidates)
    target_visible = sum(
        int(item["target_visibility"]["targets_meeting_visibility"]) for item in candidates
    )
    candidate_selection_payload = [
        {
            "capture_attempt_id": item["capture_attempt_id"],
            "cell_id": item["binding"]["cell_id"],
            "episode_index": item["binding"]["episode_index"],
            "capture_receipt_sha256": item["capture_receipt_sha256"],
        }
        for item in candidates
    ]
    report = {
        "schema": COHORT_AUDIT_SCHEMA,
        "status": "passed",
        "formal": False,
        "policy_ranking": False,
        "audit_provenance": {
            "analyzer_module": "rivermark_benchmark.cohort_audit",
            "analyzer_source_sha256": sha256_file(Path(__file__).resolve()),
            "construction_scope": "source_protocol_ledger_and_capture_artifacts",
            "offline_verification_scope": "internal_structure_and_unkeyed_digest_only",
        },
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256(protocol),
            "candidate_index_start": candidate_index_start,
            "expected_candidate_count": len(expected_bindings),
            "target_quota_by_cell": target_quotas,
        },
        "source": {
            "revision": revisions[0],
            "all_worktrees_clean": True,
            "source_tree_sha256_values": sorted(
                {str(item["source_tree_sha256"]) for item in candidates}
            ),
        },
        "accounting": {
            "failure_ledger_sha256": sha256_file(failure_ledger_path),
            "ledger_record_count": len(ledger_records),
            "protocol_attempt_count": int(coverage["attempt_count"]),
            "candidate_attempt_count": len(candidates),
            "noncandidate_protocol_attempt_count": int(coverage["attempt_count"]) - len(candidates),
            "candidate_outcomes": dict(sorted(outcomes.items())),
            "candidate_reason_codes": dict(sorted(reasons.items())),
            "candidate_selection_sha256": _canonical_sha256(candidate_selection_payload),
            "coverage": coverage,
        },
        "candidates": candidates,
        "aggregate": {
            "candidate_count": len(candidates),
            "candidate_count_by_split": dict(
                sorted(Counter(str(item["binding"]["split"]) for item in candidates).items())
            ),
            "quality_gate_pass_count": sum(
                1 for item in candidates if all(item["quality_gates"].values())
            ),
            "contact_free_count": sum(1 for item in candidates if item["contact_free"] is True),
            "target_count": target_total,
            "targets_meeting_visibility": target_visible,
            "target_visibility_rate": target_visible / target_total,
            "capture_failure_rate": int(coverage["failed_count"]) / int(coverage["attempt_count"]),
            "quarantine_rate": int(coverage["quarantined_count"]) / int(coverage["attempt_count"]),
            "formal_admission_rate": int(coverage["admitted_count"]) / int(coverage["attempt_count"]),
            "capture_bytes": _distribution(
                [int(item["resources"]["capture_bytes"]) for item in candidates]
            ),
            "duration_s": _distribution(
                [float(item["resources"]["duration_s"]) for item in candidates]
            ),
            "peak_system_commit_percent": _distribution(
                [float(item["resources"]["peak_system_commit_percent"]) for item in candidates]
            ),
            "peak_process_private_commit_bytes": _distribution(
                [int(item["resources"]["peak_process_private_commit_bytes"]) for item in candidates]
            ),
            "minimum_agent_max_displacement_m": _distribution(
                [float(item["minimum_agent_max_displacement_m"]) for item in candidates]
            ),
            "route_witness_displacement_m": _distribution(
                [float(item["route_witness_displacement_m"]) for item in candidates]
            ),
        },
        "admission_readiness": {
            "native_evidence_ready_for_review": True,
            "formal_admission_complete": False,
            "release_ready": False,
            "blocking_reason_codes": sorted(reasons),
            "not_evaluated_by_this_audit": [
                "distribution_clearance",
                "formal_candidate_pack",
                "operator_receipt_allowlist",
                "release_supply_chain",
            ],
        },
        "report_payload_sha256": "",
    }
    report["report_payload_sha256"] = _canonical_sha256(report)
    _assert_public_report(report)
    return report


def _verify_development_cohort_audit_payload(
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_exact_keys(report, _REPORT_KEYS, label="cohort audit")
    if (
        report.get("schema") != COHORT_AUDIT_SCHEMA
        or report.get("status") != "passed"
        or report.get("formal") is not False
        or report.get("policy_ranking") is not False
    ):
        raise CohortAuditError("cohort audit identity or claim boundary is invalid")
    declared_digest = report.get("report_payload_sha256")
    if not isinstance(declared_digest, str) or not _SHA256.fullmatch(declared_digest):
        raise CohortAuditError("cohort audit payload digest is missing or malformed")
    if _canonical_sha256({**report, "report_payload_sha256": ""}) != declared_digest:
        raise CohortAuditError("cohort audit payload digest is stale")
    _assert_public_report(report)
    candidates = report.get("candidates")
    aggregate = report.get("aggregate")
    accounting = report.get("accounting")
    protocol = report.get("protocol")
    source = report.get("source")
    readiness = report.get("admission_readiness")
    provenance = report.get("audit_provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (aggregate, accounting, protocol, source, readiness, provenance)
    ) or not isinstance(candidates, list):
        raise CohortAuditError("cohort audit sections are missing")
    _require_exact_keys(protocol, _PROTOCOL_KEYS, label="cohort protocol")
    _require_exact_keys(source, _SOURCE_KEYS, label="cohort source")
    _require_exact_keys(accounting, _ACCOUNTING_KEYS, label="cohort accounting")
    _require_exact_keys(aggregate, _AGGREGATE_KEYS, label="cohort aggregate")
    _require_exact_keys(readiness, _READINESS_KEYS, label="cohort admission readiness")
    if (
        set(provenance)
        != {
            "analyzer_module",
            "analyzer_source_sha256",
            "construction_scope",
            "offline_verification_scope",
        }
        or provenance.get("analyzer_module") != "rivermark_benchmark.cohort_audit"
        or not isinstance(provenance.get("analyzer_source_sha256"), str)
        or not _SHA256.fullmatch(str(provenance["analyzer_source_sha256"]))
        or provenance.get("construction_scope")
        != "source_protocol_ledger_and_capture_artifacts"
        or provenance.get("offline_verification_scope")
        != "internal_structure_and_unkeyed_digest_only"
    ):
        raise CohortAuditError("cohort audit provenance or verification scope is invalid")
    if not candidates or any(not isinstance(item, Mapping) for item in candidates):
        raise CohortAuditError("cohort audit candidates are missing or malformed")
    expected_count = protocol.get("expected_candidate_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(candidates)
        or aggregate.get("candidate_count") != len(candidates)
        or accounting.get("candidate_attempt_count") != len(candidates)
    ):
        raise CohortAuditError("cohort candidate counts are inconsistent")
    attempt_ids = [item.get("capture_attempt_id") for item in candidates]
    receipt_hashes = [item.get("capture_receipt_sha256") for item in candidates]
    bindings = [item.get("binding") for item in candidates]
    if (
        len(set(attempt_ids)) != len(candidates)
        or len(set(receipt_hashes)) != len(candidates)
        or any(not isinstance(binding, Mapping) for binding in bindings)
    ):
        raise CohortAuditError("cohort candidate identities are not unique")
    protocol_id = protocol.get("protocol_id")
    protocol_digest = protocol.get("protocol_sha256")
    if (
        not isinstance(protocol_id, str)
        or not _PUBLIC_ID.fullmatch(protocol_id)
        or not isinstance(protocol_digest, str)
        or not _SHA256.fullmatch(protocol_digest)
    ):
        raise CohortAuditError("cohort protocol identity is malformed")
    candidate_index_start = protocol.get("candidate_index_start")
    target_quotas = protocol.get("target_quota_by_cell")
    if (
        isinstance(candidate_index_start, bool)
        or not isinstance(candidate_index_start, int)
        or candidate_index_start < 0
        or not isinstance(target_quotas, Mapping)
        or not target_quotas
    ):
        raise CohortAuditError("cohort protocol quota summary is malformed")
    for cell_id, quota in target_quotas.items():
        if (
            not isinstance(cell_id, str)
            or not _PUBLIC_ID.fullmatch(cell_id)
            or not isinstance(quota, Mapping)
            or set(quota) != {"condition_id", "target_count"}
            or quota.get("condition_id") not in _TARGET_COUNT_BY_CONDITION
            or quota.get("target_count")
            != _TARGET_COUNT_BY_CONDITION[str(quota.get("condition_id"))]
        ):
            raise CohortAuditError("cohort protocol target quota is malformed")
    source_revision = source.get("revision")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise CohortAuditError("cohort source revision is malformed")
    for item in candidates:
        _require_exact_keys(item, _CANDIDATE_KEYS, label="cohort candidate")
        binding = item["binding"]
        if (
            set(binding) != COLLECTION_BINDING_KEYS
            or binding.get("protocol_id") != protocol_id
            or binding.get("protocol_sha256") != protocol_digest
            or not isinstance(binding.get("cell_id"), str)
            or not _PUBLIC_ID.fullmatch(str(binding.get("cell_id")))
            or isinstance(binding.get("episode_index"), bool)
            or not isinstance(binding.get("episode_index"), int)
            or int(binding["episode_index"]) < 0
            or isinstance(binding.get("episode_seed"), bool)
            or not isinstance(binding.get("episode_seed"), int)
            or int(binding["episode_seed"]) < 0
            or binding.get("split") not in COLLECTION_SPLITS
        ):
            raise CohortAuditError("cohort candidate binding is malformed")
        attempt_id = item.get("capture_attempt_id")
        if not isinstance(attempt_id, str) or not _PUBLIC_ID.fullmatch(attempt_id):
            raise CohortAuditError("cohort candidate attempt identity is malformed")
        cell_quota = target_quotas.get(binding["cell_id"])
        if not isinstance(cell_quota, Mapping):
            raise CohortAuditError("cohort candidate target quota is missing")
        if item.get("source_revision") != source_revision:
            raise CohortAuditError("cohort candidate source revision is inconsistent")
        for key in (
            "source_tree_sha256",
            "capture_receipt_sha256",
            "independent_validation_sha256",
            "task_outcome_sha256",
        ):
            value = item.get(key)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise CohortAuditError(f"cohort candidate {key} is malformed")
        gates = item.get("quality_gates")
        if (
            not isinstance(gates, Mapping)
            or set(gates) != set(_QUALITY_GATE_CHECKS)
            or any(value is not True for value in gates.values())
        ):
            raise CohortAuditError("cohort candidate quality gates are missing or failed")
        visibility = item.get("target_visibility")
        if not isinstance(visibility, Mapping) or set(visibility) != _TARGET_VISIBILITY_KEYS:
            raise CohortAuditError("cohort candidate target visibility is malformed")
        target_count_item = _non_negative_int(
            visibility.get("target_count"), label="candidate.target_count"
        )
        targets_visible_item = _non_negative_int(
            visibility.get("targets_meeting_visibility"),
            label="candidate.targets_meeting_visibility",
        )
        failed_targets_item = _non_negative_int(
            visibility.get("failed_target_count"), label="candidate.failed_target_count"
        )
        minimum_frames = _non_negative_int(
            visibility.get("minimum_visible_frames"), label="candidate.minimum_visible_frames"
        )
        maximum_frames = _non_negative_int(
            visibility.get("maximum_visible_frames"), label="candidate.maximum_visible_frames"
        )
        minimum_pixels = _non_negative_int(
            visibility.get("minimum_max_pixels"), label="candidate.minimum_max_pixels"
        )
        maximum_pixels = _non_negative_int(
            visibility.get("maximum_max_pixels"), label="candidate.maximum_max_pixels"
        )
        if target_count_item != cell_quota.get("target_count"):
            raise CohortAuditError("cohort candidate target quota is inconsistent")
        if (
            target_count_item <= 0
            or targets_visible_item != target_count_item
            or failed_targets_item != 0
            or minimum_frames <= 0
            or minimum_frames > maximum_frames
            or minimum_pixels <= 0
            or minimum_pixels > maximum_pixels
        ):
            raise CohortAuditError("cohort candidate target visibility is inconsistent")
        if item.get("contact_free") is not True:
            raise CohortAuditError("cohort candidate contact-free evidence is missing")
        _finite_number(
            item.get("minimum_agent_max_displacement_m"),
            label="candidate.minimum_agent_max_displacement_m",
        )
        _finite_number(
            item.get("route_witness_displacement_m"),
            label="candidate.route_witness_displacement_m",
            minimum=_ROUTE_WITNESS_MIN_DISPLACEMENT_M,
        )
        resources = item.get("resources")
        if not isinstance(resources, Mapping):
            raise CohortAuditError("cohort candidate resource evidence is missing")
        _require_exact_keys(resources, _RESOURCE_KEYS, label="cohort candidate resources")
        if _non_negative_int(resources.get("capture_bytes"), label="candidate.capture_bytes") <= 0:
            raise CohortAuditError("cohort candidate capture size is invalid")
        if _finite_number(resources.get("duration_s"), label="candidate.duration_s") <= 0.0:
            raise CohortAuditError("cohort candidate duration is invalid")
        commit_percent = _finite_number(
            resources.get("peak_system_commit_percent"),
            label="candidate.peak_system_commit_percent",
        )
        if commit_percent > 100.0:
            raise CohortAuditError("cohort candidate system commit percentage is invalid")
        _non_negative_int(
            resources.get("peak_process_private_commit_bytes"),
            label="candidate.peak_process_private_commit_bytes",
        )
        ledger = item.get("ledger")
        if (
            not isinstance(ledger, Mapping)
            or set(ledger) != _LEDGER_KEYS
            or dict(ledger) != _DEVELOPMENT_LEDGER_STATE
        ):
            raise CohortAuditError("cohort candidate is not a development quarantine")
    if set(target_quotas) != {str(item["binding"]["cell_id"]) for item in candidates}:
        raise CohortAuditError("cohort protocol target quota cells are inconsistent")
    selection = [
        {
            "capture_attempt_id": item["capture_attempt_id"],
            "cell_id": item["binding"]["cell_id"],
            "episode_index": item["binding"]["episode_index"],
            "capture_receipt_sha256": item["capture_receipt_sha256"],
        }
        for item in candidates
    ]
    if accounting.get("candidate_selection_sha256") != _canonical_sha256(selection):
        raise CohortAuditError("cohort candidate selection digest is stale")
    split_counts = dict(
        sorted(Counter(str(item["binding"]["split"]) for item in candidates).items())
    )
    quality_count = sum(1 for item in candidates if all(item["quality_gates"].values()))
    contact_count = sum(1 for item in candidates if item.get("contact_free") is True)
    target_count = sum(int(item["target_visibility"]["target_count"]) for item in candidates)
    target_visible = sum(
        int(item["target_visibility"]["targets_meeting_visibility"])
        for item in candidates
    )
    if (
        aggregate.get("candidate_count_by_split") != split_counts
        or aggregate.get("quality_gate_pass_count") != quality_count
        or aggregate.get("contact_free_count") != contact_count
        or aggregate.get("target_count") != target_count
        or aggregate.get("targets_meeting_visibility") != target_visible
        or aggregate.get("target_visibility_rate") != target_visible / target_count
    ):
        raise CohortAuditError("cohort aggregate evidence counts are inconsistent")
    expected_distributions = {
        "capture_bytes": _distribution(
            [int(item["resources"]["capture_bytes"]) for item in candidates]
        ),
        "duration_s": _distribution(
            [float(item["resources"]["duration_s"]) for item in candidates]
        ),
        "peak_system_commit_percent": _distribution(
            [float(item["resources"]["peak_system_commit_percent"]) for item in candidates]
        ),
        "peak_process_private_commit_bytes": _distribution(
            [int(item["resources"]["peak_process_private_commit_bytes"]) for item in candidates]
        ),
        "minimum_agent_max_displacement_m": _distribution(
            [float(item["minimum_agent_max_displacement_m"]) for item in candidates]
        ),
        "route_witness_displacement_m": _distribution(
            [float(item["route_witness_displacement_m"]) for item in candidates]
        ),
    }
    if any(
        aggregate.get(name) != distribution
        or not isinstance(aggregate.get(name), Mapping)
        or set(aggregate[name]) != _DISTRIBUTION_KEYS
        for name, distribution in expected_distributions.items()
    ):
        raise CohortAuditError("cohort aggregate distributions are inconsistent")
    candidate_outcomes = dict(
        sorted(Counter(str(item["ledger"]["outcome"]) for item in candidates).items())
    )
    candidate_reasons = dict(
        sorted(Counter(str(item["ledger"]["reason_code"]) for item in candidates).items())
    )
    if (
        accounting.get("candidate_outcomes") != candidate_outcomes
        or accounting.get("candidate_reason_codes") != candidate_reasons
    ):
        raise CohortAuditError("cohort candidate ledger accounting is inconsistent")
    coverage = accounting.get("coverage")
    if not isinstance(coverage, Mapping):
        raise CohortAuditError("cohort coverage report is missing")
    failure_ledger_sha256 = accounting.get("failure_ledger_sha256")
    ledger_record_count = accounting.get("ledger_record_count")
    if (
        not isinstance(failure_ledger_sha256, str)
        or not _SHA256.fullmatch(failure_ledger_sha256)
        or isinstance(ledger_record_count, bool)
        or not isinstance(ledger_record_count, int)
        or ledger_record_count < 0
    ):
        raise CohortAuditError("cohort ledger accounting is malformed")
    coverage_counts = _verify_embedded_coverage(
        coverage,
        protocol_id=protocol_id,
        protocol_digest=protocol_digest,
        target_quotas=target_quotas,
        candidates=candidates,
    )
    attempts = coverage_counts["attempt_count"]
    if attempts <= 0:
        raise CohortAuditError("cohort protocol attempt count is invalid")
    if ledger_record_count != coverage_counts["ledger_record_count"]:
        raise CohortAuditError("cohort ledger accounting is inconsistent")
    if (
        accounting.get("protocol_attempt_count") != attempts
        or accounting.get("noncandidate_protocol_attempt_count") != attempts - len(candidates)
        or aggregate.get("capture_failure_rate") != coverage_counts["failed_count"] / attempts
        or aggregate.get("quarantine_rate") != coverage_counts["quarantined_count"] / attempts
        or aggregate.get("formal_admission_rate") != coverage_counts["admitted_count"] / attempts
    ):
        raise CohortAuditError("cohort attempt accounting is inconsistent")
    source_revisions = {item.get("source_revision") for item in candidates}
    source_tree_hashes = sorted({str(item["source_tree_sha256"]) for item in candidates})
    if (
        source_revisions != {source_revision}
        or source.get("source_tree_sha256_values") != source_tree_hashes
        or source.get("all_worktrees_clean") is not True
    ):
        raise CohortAuditError("cohort source revision summary is inconsistent")
    if (
        readiness.get("native_evidence_ready_for_review") is not True
        or readiness.get("formal_admission_complete") is not False
        or readiness.get("release_ready") is not False
        or readiness.get("blocking_reason_codes") != sorted(candidate_reasons)
        or not isinstance(readiness.get("not_evaluated_by_this_audit"), list)
        or len(readiness["not_evaluated_by_this_audit"]) != len(_NOT_EVALUATED)
        or set(readiness["not_evaluated_by_this_audit"]) != _NOT_EVALUATED
    ):
        raise CohortAuditError("cohort admission-readiness summary is inconsistent")
    return report


def verify_development_cohort_audit(report_path: Path) -> Mapping[str, Any]:
    """Verify a saved public report without access to the source captures."""

    report = _load_json(report_path.expanduser().resolve(), label="development cohort audit")
    try:
        return _verify_development_cohort_audit_payload(report)
    except CohortAuditError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise CohortAuditError("cohort audit structure is malformed") from exc


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise CohortAuditError(f"refusing to overwrite cohort audit: {destination}")
    expected = _canonical_sha256({**report, "report_payload_sha256": ""})
    if report.get("report_payload_sha256") != expected:
        raise CohortAuditError("cohort audit payload digest is stale")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path, nargs="?")
    parser.add_argument("--failure-ledger", type=Path)
    parser.add_argument("--capture", type=Path, action="append")
    parser.add_argument("--candidate-index-start", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_report is not None:
            if any(
                value is not None
                for value in (args.protocol, args.failure_ledger, args.capture, args.output)
            ):
                raise CohortAuditError(
                    "--verify-report cannot be combined with build arguments"
                )
            report = verify_development_cohort_audit(args.verify_report)
            print(
                json.dumps(
                    {
                        "schema": COHORT_AUDIT_SCHEMA,
                        "status": "internally_consistent",
                        "analyzer_source_matches_current": report["audit_provenance"]
                        ["analyzer_source_sha256"]
                        == sha256_file(Path(__file__).resolve()),
                        "candidate_count": report["aggregate"]["candidate_count"],
                        "report_payload_sha256": report["report_payload_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if (
            args.protocol is None
            or args.failure_ledger is None
            or not args.capture
            or args.output is None
        ):
            raise CohortAuditError(
                "build requires protocol, --failure-ledger, at least one --capture, and --output"
            )
        report = build_development_cohort_audit(
            args.protocol,
            args.failure_ledger,
            args.capture,
            candidate_index_start=args.candidate_index_start,
        )
        _write_new_report(args.output, report)
    except (CohortAuditError, OSError, ValueError) as exc:
        print(json.dumps({"schema": COHORT_AUDIT_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "schema": COHORT_AUDIT_SCHEMA,
                "status": report["status"],
                "candidate_count": report["aggregate"]["candidate_count"],
                "protocol_attempt_count": report["accounting"]["protocol_attempt_count"],
                "formal_admission_complete": report["admission_readiness"]["formal_admission_complete"],
                "report_payload_sha256": report["report_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
