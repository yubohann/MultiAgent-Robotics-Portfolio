"""Shared integrity helpers for the CF2X B-gate L0/L1 pairing evidence."""

from __future__ import annotations

import re
from math import isfinite
from typing import Any

from aerocity_bench.canonical import content_hash

L0_PAIRING_SCHEMA = "org.aerocity.bench.cf2x-b-gate-l0-pairing.v1"
L0_PAIRING_SCOPE = "g2-i-l0-paired-cf2x-b-gate-calibration"
L0_PAIRING_METHODS = (
    "sweep-3d",
    "atlas-surface-inspector",
    "atlas-region-greedy",
)
SHARED_BINDING_FIELDS = frozenset(
    {
        "layout_hash",
        "stage_sha256",
        "cityspec_sha256",
        "task_spec_sha256",
        "task_spec_hash",
        "public_episode_sha256",
        "mission_sector_hash",
        "atlas_hash",
        "execution_contract_hash",
        "release_config_sha256",
        "baseline_source_sha256",
        "geometry_source_sha256",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PAIRING_REPORT_FIELDS = frozenset(
    {
        "schema",
        "evidence_scope",
        "formal_score_eligible",
        "status",
        "b_gate_manifest_report_hash",
        "b_gate_manifest_file_sha256",
        "l0_implementation_hash",
        "layout_ancestors",
        "method_ids",
        "expected_input_bindings",
        "records",
        "report_hash",
    }
)
PAIRING_RECORD_FIELDS = frozenset(
    {
        "layout_ancestor",
        "method_id",
        "score",
        "execution_level",
        "input_bindings",
        "private_episode_sha256",
        "private_evaluator_commitment",
        "execution",
        "evidence_hash",
    }
)
PAIRING_EXECUTION_FIELDS = frozenset(
    {
        "all_returned_home",
        "collision_count",
        "out_of_bounds_actions",
        "deadline_miss_tick_count",
        "task_time_s",
    }
)


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def private_evaluator_commitment(
    private_episode_sha256: str,
    layout_hash: str,
    execution_contract_hash: str,
) -> str:
    """Recompute the private commitment published by CF2X L1 receipts."""

    return content_hash(
        {
            "private_episode_sha256": private_episode_sha256,
            "layout_hash": layout_hash,
            "execution_contract_hash": execution_contract_hash,
        }
    )


def l0_pair_record_evidence_hash(record: dict[str, Any], manifest_report_hash: str) -> str:
    """Bind an L0 score to immutable public inputs and the private commitment."""

    return content_hash(
        {
            "b_gate_manifest_report_hash": manifest_report_hash,
            "layout_ancestor": record["layout_ancestor"],
            "method_id": record["method_id"],
            "score": record["score"],
            "execution_level": record["execution_level"],
            "input_bindings": record["input_bindings"],
            "private_episode_sha256": record["private_episode_sha256"],
            "private_evaluator_commitment": record["private_evaluator_commitment"],
            "execution": record["execution"],
        }
    )


def validate_l0_pairing_header(
    pairing: object,
    *,
    manifest_report_hash: str,
    manifest_file_sha256: str,
    method_ids: tuple[str, ...],
    layout_ancestors: tuple[str, ...],
    expected_input_bindings: dict[str, str],
) -> list[dict[str, Any]]:
    """Fail closed before L1 execution or final B-gate aggregation."""

    if not isinstance(pairing, dict) or set(pairing) != PAIRING_REPORT_FIELDS:
        raise ValueError("L0 pairing report fields differ")
    claimed_hash = pairing["report_hash"]
    unhashed = {key: value for key, value in pairing.items() if key != "report_hash"}
    if (
        pairing["schema"] != L0_PAIRING_SCHEMA
        or pairing["evidence_scope"] != L0_PAIRING_SCOPE
        or pairing["formal_score_eligible"] is not False
        or pairing["status"] != "VERIFIED_L0_PAIRING"
        or not is_sha256(claimed_hash)
        or claimed_hash != content_hash(unhashed)
        or pairing["b_gate_manifest_report_hash"] != manifest_report_hash
        or pairing["b_gate_manifest_file_sha256"] != manifest_file_sha256
        or not is_sha256(pairing["l0_implementation_hash"])
        or tuple(pairing["method_ids"]) != method_ids
        or tuple(pairing["layout_ancestors"]) != layout_ancestors
        or pairing["expected_input_bindings"] != expected_input_bindings
    ):
        raise ValueError("L0 pairing report does not bind the frozen B-gate panel")
    records = pairing["records"]
    expected_pairs = {(ancestor, method) for ancestor in layout_ancestors for method in method_ids}
    if not isinstance(records, list) or len(records) != len(expected_pairs):
        raise ValueError("L0 pairing does not contain one record for every B-gate pair")
    seen_pairs: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != PAIRING_RECORD_FIELDS:
            raise ValueError("L0 pairing record fields differ")
        pair = (str(record["layout_ancestor"]), str(record["method_id"]))
        bindings = record["input_bindings"]
        execution = record["execution"]
        if (
            pair not in expected_pairs
            or pair in seen_pairs
            or record["execution_level"] != "L0"
            or isinstance(record["score"], bool)
            or not isinstance(record["score"], (int, float))
            or not isfinite(float(record["score"]))
            or not isinstance(bindings, dict)
            or set(bindings) != SHARED_BINDING_FIELDS
            or any(not is_sha256(value) for value in bindings.values())
            or any(
                bindings[field] != expected_input_bindings[field]
                for field in expected_input_bindings
                if field in bindings
            )
            or not is_sha256(record["private_episode_sha256"])
            or not is_sha256(record["private_evaluator_commitment"])
            or not isinstance(execution, dict)
            or set(execution) != PAIRING_EXECUTION_FIELDS
            or not isinstance(execution["all_returned_home"], bool)
            or any(
                isinstance(execution[field], bool)
                or not isinstance(execution[field], (int, float))
                or not isfinite(float(execution[field]))
                for field in PAIRING_EXECUTION_FIELDS - {"all_returned_home"}
            )
            or record["private_evaluator_commitment"]
            != private_evaluator_commitment(
                str(record["private_episode_sha256"]),
                str(bindings["layout_hash"]),
                str(bindings["execution_contract_hash"]),
            )
            or record["evidence_hash"]
            != l0_pair_record_evidence_hash(record, manifest_report_hash)
        ):
            raise ValueError("L0 pairing record is invalid or no longer bound to private truth")
        seen_pairs.add(pair)
        normalized.append(record)
    if seen_pairs != expected_pairs:
        raise ValueError("L0 pairing pair set differs from the B-gate panel")
    return normalized
