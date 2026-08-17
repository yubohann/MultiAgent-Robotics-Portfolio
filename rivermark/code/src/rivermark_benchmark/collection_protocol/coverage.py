"""Coverage reports built from failure-ledger records bound to one protocol."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..failure_ledger import FAILURE_LEDGER_SCHEMA, validate_failure_record
from .common import (
    CollectionProtocolError,
    _canonical_bytes,
    protocol_sha256,
)
from .constants import (
    _ID,
    _VALUE,
    COVERAGE_REPORT_SCHEMA,
    NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
    T1_COLLECTION_PROTOCOL_SCHEMA,
    T1_COVERAGE_REPORT_SCHEMA,
)
from .seeds import derive_episode_seed
from .validate import validate_collection_protocol


def coverage_report(protocol: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build coverage from failure-ledger records bound to one protocol."""

    if protocol.get("schema") == NATIVE_T2_CANARY_PROTOCOL_SCHEMA:
        raise CollectionProtocolError(
            "coverage report is not defined for development-only native T2 canary protocols"
        )
    protocol_issues = validate_collection_protocol(protocol)
    if protocol_issues:
        raise CollectionProtocolError("invalid collection protocol: " + "; ".join(issue.code for issue in protocol_issues))
    expected_protocol_hash = protocol_sha256(protocol)
    cells = {str(cell["cell_id"]): cell for cell in protocol["cells"]}
    seed_start = int(protocol["randomization"]["episode_seed_start"])
    seen_attempts: set[str] = set()
    seen_episodes: set[str] = set()
    seen_admitted_cell_indices: set[tuple[str, int]] = set()
    counts = {
        cell_id: {
            "attempt_count": 0,
            "admitted_count": 0,
            "quarantined_count": 0,
            "failed_count": 0,
            "exclusion_reasons": Counter(),
        }
        for cell_id in cells
    }
    canonical_attempts: list[dict[str, Any]] = []
    excluded_protocol_id_count = 0
    excluded_protocol_hash_count = 0
    for index, attempt in enumerate(attempts):
        path = f"$[{index}]"
        if not isinstance(attempt, Mapping):
            raise CollectionProtocolError(f"attempt {path} must be an object")
        record_issues = validate_failure_record(attempt)
        if record_issues:
            detail = ", ".join(f"{issue.code}:{issue.path}" for issue in record_issues)
            raise CollectionProtocolError(f"attempt {path} is not a valid public failure-ledger record: {detail}")
        attempt_id = str(attempt["attempt_id"])
        if attempt_id in seen_attempts:
            raise CollectionProtocolError(f"attempt {path} has a duplicate attempt_id")
        seen_attempts.add(attempt_id)
        # The ledger is append-only across protocol eras. Coverage has to select
        # the exact frozen (ID, hash) pair, while exposing how many valid
        # records were excluded. This prevents legacy records and prior frozen
        # revisions from contaminating the current cohort without hiding that
        # they exist in the public denominator.
        if attempt.get("collection_protocol_id") != protocol["protocol_id"]:
            excluded_protocol_id_count += 1
            continue
        if attempt.get("collection_protocol_sha256") != expected_protocol_hash:
            excluded_protocol_hash_count += 1
            continue
        cell_id = attempt.get("collection_cell_id")
        if not isinstance(cell_id, str) or cell_id not in cells:
            raise CollectionProtocolError(f"attempt {path} references an unknown collection cell")
        cell = cells[cell_id]
        if attempt.get("split") != cell["split"]:
            raise CollectionProtocolError(f"attempt {path} split does not match its collection cell")
        episode_index = attempt.get("collection_episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise CollectionProtocolError(f"attempt {path} lacks a valid collection episode index")
        cell_index = (cell_id, episode_index)
        expected_seed = derive_episode_seed(
            protocol_id=str(protocol["protocol_id"]),
            cell_id=cell_id,
            episode_seed_start=seed_start,
            episode_index=episode_index,
        )
        if attempt.get("episode_seed") != expected_seed:
            raise CollectionProtocolError(f"attempt {path} does not match the deterministic episode seed")
        outcome = str(attempt["outcome"])
        if outcome == "admitted":
            episode_id = attempt.get("episode_id")
            if not isinstance(episode_id, str) or not _ID.fullmatch(episode_id) or episode_id in seen_episodes:
                raise CollectionProtocolError(f"attempt {path} has a missing or duplicate admitted episode_id")
            if cell_index in seen_admitted_cell_indices:
                raise CollectionProtocolError(f"attempt {path} repeats an admitted collection cell episode index")
            seen_episodes.add(episode_id)
            seen_admitted_cell_indices.add(cell_index)
        else:
            reason_code = attempt.get("reason_code")
            if not isinstance(reason_code, str) or not _VALUE.fullmatch(reason_code):
                raise CollectionProtocolError(f"attempt {path} lacks a public exclusion reason")
            counts[cell_id]["exclusion_reasons"][reason_code] += 1
        counts[cell_id]["attempt_count"] += 1
        counts[cell_id][f"{outcome}_count"] += 1
        canonical_attempts.append(dict(attempt))

    cell_reports: list[dict[str, Any]] = []
    all_reasons: Counter[str] = Counter()
    for cell_id, cell in sorted(cells.items()):
        count = counts[cell_id]
        minimum_attempts = int(cell["minimum_attempts"])
        minimum_admitted = int(cell["minimum_admitted"])
        reasons = dict(sorted(count["exclusion_reasons"].items()))
        all_reasons.update(reasons)
        cell_reports.append(
            {
                "cell_id": cell_id,
                "split": cell["split"],
                "conditions": dict(cell["conditions"]),
                "attempt_count": count["attempt_count"],
                "admitted_count": count["admitted_count"],
                "quarantined_count": count["quarantined_count"],
                "failed_count": count["failed_count"],
                "exclusion_reasons": reasons,
                "minimum_attempts": minimum_attempts,
                "minimum_admitted": minimum_admitted,
                "status": "passed"
                if count["attempt_count"] >= minimum_attempts and count["admitted_count"] >= minimum_admitted
                else "under_quota",
            }
        )
    is_t1 = protocol.get("schema") == T1_COLLECTION_PROTOCOL_SCHEMA
    report = {
        "schema": T1_COVERAGE_REPORT_SCHEMA if is_t1 else COVERAGE_REPORT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": expected_protocol_hash,
        "failure_ledger_schema": FAILURE_LEDGER_SCHEMA,
        "ledger_record_count": len(attempts),
        "excluded_ledger_record_count": (
            excluded_protocol_id_count + excluded_protocol_hash_count
        ),
        "excluded_protocol_id_count": excluded_protocol_id_count,
        "excluded_protocol_hash_count": excluded_protocol_hash_count,
        "attempts_sha256": hashlib.sha256(_canonical_bytes(canonical_attempts)).hexdigest(),
        "attempt_count": len(canonical_attempts),
        "admitted_count": sum(item["admitted_count"] for item in cell_reports),
        "quarantined_count": sum(item["quarantined_count"] for item in cell_reports),
        "failed_count": sum(item["failed_count"] for item in cell_reports),
        "exclusion_reasons": dict(sorted(all_reasons.items())),
        "cells": cell_reports,
    }
    if is_t1:
        target = int(protocol["analysis_plan"]["initial_admitted_episode_target"])
        report["quota_analysis"] = {
            "basis": protocol["analysis_plan"]["quota_basis"],
            "statistical_unit": protocol["statistical_unit"]["unit"],
            "policy_ranking": False,
            "initial_admitted_episode_target": target,
            "admitted_episodes": report["admitted_count"],
            "quota_target_met": report["admitted_count"] >= target,
        }
        report["complete"] = bool(
            report["quota_analysis"]["quota_target_met"]
            and all(cell["status"] == "passed" for cell in cell_reports)
        )
    else:
        power = protocol["power_analysis"]
        evaluation_split = str(power["evaluation_split"])
        evaluation_admitted = sum(
            item["admitted_count"]
            for item in cell_reports
            if item["split"] == evaluation_split
        )
        required_evaluation = int(power["required_evaluation_episodes"])
        report["power_analysis"] = {
            "method": power["method"],
            "primary_metric": power["primary_metric"],
            "evaluation_split": evaluation_split,
            "required_evaluation_episodes": required_evaluation,
            "admitted_evaluation_episodes": evaluation_admitted,
            "power_target_met": evaluation_admitted >= required_evaluation,
        }
        report["complete"] = bool(
            report["power_analysis"]["power_target_met"]
            and all(cell["status"] == "passed" for cell in cell_reports)
        )
    return report
