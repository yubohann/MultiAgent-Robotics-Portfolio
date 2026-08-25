"""Verify and freeze the development-only G2-I scientific task contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, read_json, write_json

SEARCH_SCHEMA = "org.aerocity.bench.g2-i-l0-searchability-calibration.v1"
DENSITY_SCHEMA = "org.aerocity.bench.g2-i-atlas-density-ablation.v1"
TARGET_PROCESS_SCHEMA = (
    "org.aerocity.bench.g2-i-target-process-performance-ablation.v1"
)
PRIOR_SCHEMA = "org.aerocity.bench.g2-i-prior-ablation.v1"
SCIENTIFIC_SCHEMA = "org.aerocity.bench.g2-i-scientific-gate-report.v1"
REPORT_SCHEMA = "org.aerocity.bench.g2-i-a-gate-freeze.v1"
PUBLIC_METHODS = ("atlas-surface-inspector", "atlas-region-greedy")
ORACLE_METHOD = "centralized-oracle"
NOMINAL_POLICY = "g2-i-geometric-sampling-calibration-candidate-v2"
DENSITY_POLICIES = (
    "g2-i-geometric-sampling-density-sparse-v1",
    NOMINAL_POLICY,
    "g2-i-geometric-sampling-density-dense-v1",
)
TARGET_PROCESSES = (
    "clustered_surface",
    "height_stratified",
    "uniform_surface",
)
PRIOR_METHODS = {
    "coarse-regions": "atlas-coarse-region-inspector",
    "full-cells": "atlas-region-greedy",
}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--searchability", type=Path, required=True)
    parser.add_argument("--density", type=Path, required=True)
    parser.add_argument("--target-process", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--scientific-audit", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _validated_report(path: Path, schema: str) -> dict[str, Any]:
    report = read_json(path.resolve())
    if not isinstance(report, dict) or report.get("schema") != schema:
        raise ValueError(f"evidence schema differs: {path}")
    supplied_hash = str(report.get("report_hash", ""))
    payload = dict(report)
    payload.pop("report_hash", None)
    if content_hash(payload) != supplied_hash:
        raise ValueError(f"evidence report hash mismatch: {path}")
    if report.get("formal_score_eligible") is not False:
        raise ValueError(f"A-gate input must be development-only: {path}")
    return report


def _safety_pass(report: dict[str, Any]) -> bool:
    return (
        report.get("all_returned_home") is True
        and int(report.get("collision_count", -1)) == 0
        and int(report.get("out_of_bounds_actions", -1)) == 0
        and int(report.get("deadline_misses", -1)) == 0
    )


def _exact_ids(values: object, expected: tuple[str, ...] | set[str]) -> bool:
    if not isinstance(values, list):
        return False
    normalized = [str(value) for value in values]
    return len(normalized) == len(expected) and set(normalized) == set(expected)


def _searchability_checks(report: dict[str, Any]) -> dict[str, bool]:
    rows = report.get("method_reports", [])
    method_ids = [str(item.get("method_id", "")) for item in rows]
    method_reports = {
        method_id: item for method_id, item in zip(method_ids, rows, strict=True)
    }
    expected_methods = {*PUBLIC_METHODS, ORACLE_METHOD}
    public_complete = set(PUBLIC_METHODS).issubset(method_reports)
    oracle = method_reports.get(ORACLE_METHOD, {})
    public = [method_reports.get(method_id, {}) for method_id in PUBLIC_METHODS]
    public_safety = all(
        item
        and len(item.get("ancestors", [])) >= 5
        and all(_safety_pass(ancestor) for ancestor in item.get("ancestors", []))
        for item in public
    )
    public_nonzero = all(
        int(item.get("nonzero_ancestor_count", 0)) >= 4 for item in public
    )
    public_recall = [float(item.get("mean_final_confirmed_recall", -1.0)) for item in public]
    vectors = [
        tuple(float(row["mean_confirmation_count"]) for row in item.get("ancestors", []))
        for item in public
    ]
    oracle_safety = (
        bool(oracle)
        and len(oracle.get("ancestors", [])) >= 5
        and all(_safety_pass(ancestor) for ancestor in oracle.get("ancestors", []))
    )
    contract = report.get("contract", {})
    return {
        "complete_canonical_method_set": (
            report.get("searchability_gate", {}).get("complete_method_set") is True
            and public_complete
            and ORACLE_METHOD in method_reports
            and len(method_ids) == len(expected_methods)
            and set(method_ids) == expected_methods
            and _exact_ids(contract.get("methods"), expected_methods)
        ),
        "canonical_task_contract_used": (
            float(contract.get("episode_duration_s", -1.0)) == 300.0
            and contract.get("max_steps") is None
            and contract.get("target_count_visible_to_method") is False
            and contract.get("l0_not_a_native_or_formal_score") is True
            and contract.get("mission_sector_required") is True
            and contract.get("mission_sector_frozen_before_private_sampling") is True
        ),
        "two_public_methods_stable_nonzero": public_nonzero,
        "strong_public_recall_nonsaturated": bool(public_recall)
        and 0.10 <= max(public_recall) <= 0.90,
        "public_ancestor_vectors_distinct": len(vectors) == 2
        and len(vectors[0]) >= 5
        and vectors[0] != vectors[1],
        "public_methods_safe_and_return": public_safety,
        "oracle_five_ancestor_feasible_and_returns": (
            int(oracle.get("nonzero_ancestor_count", 0)) >= 5
            and int(oracle.get("ancestor_count", 0)) >= 5
            and oracle_safety
        ),
    }


def _density_checks(report: dict[str, Any]) -> dict[str, bool]:
    rows = report.get("method_reports", [])
    observed_grid = [
        (str(row.get("sampling_policy_id", "")), str(row.get("method_id", "")))
        for row in rows
    ]
    expected_grid = {
        (policy_id, method_id)
        for policy_id in DENSITY_POLICIES
        for method_id in PUBLIC_METHODS
    }
    policies = {policy_id for policy_id, _ in observed_grid}
    nominal_rows = [
        row for row in rows if row.get("sampling_policy_id") == NOMINAL_POLICY
    ]
    contract = report.get("contract", {})
    return {
        "complete_three_by_two_condition_grid": (
            contract.get("complete_condition_set") is True
            and int(report.get("aggregate", {}).get("condition_count", 0)) == 6
            and len(observed_grid) == len(expected_grid)
            and len(set(observed_grid)) == len(observed_grid)
            and set(observed_grid) == expected_grid
            and _exact_ids(contract.get("declared_policy_ids"), DENSITY_POLICIES)
            and _exact_ids(contract.get("selected_policy_ids"), DENSITY_POLICIES)
            and _exact_ids(contract.get("selected_method_ids"), PUBLIC_METHODS)
        ),
        "nominal_policy_present": NOMINAL_POLICY in policies
        and len(nominal_rows) == len(PUBLIC_METHODS),
        "density_is_paired_private_safe_calibration": (
            contract.get("target_truth_visible_to_methods") is False
            and contract.get("l0_not_a_native_or_formal_score") is True
            and contract.get("selected_city_panel_matched_across_density_conditions")
            is True
            and contract.get("public_region_cohort_matched_across_density_conditions")
            is True
            and contract.get("private_target_realizations_matched_across_density_conditions")
            is True
        ),
        "all_density_conditions_have_complete_denominators": all(
            int(row.get("independent_ancestor_count", 0)) >= 5
            and "nonzero_ancestor_count" in row
            and "collision_count" in row
            and "deadline_misses" in row
            for row in rows
        ),
        "nominal_density_conditions_searchable": all(
            int(row.get("nonzero_ancestor_count", 0)) >= 4
            for row in nominal_rows
        ),
        "nominal_density_conditions_safe_and_return": all(
            _safety_pass(row) for row in nominal_rows
        ),
        "density_failures_are_reported_not_dropped": all(
            int(row.get("collision_count", -1)) >= 0
            and int(row.get("deadline_misses", -1)) >= 0
            for row in rows
        ),
    }


def _target_process_checks(report: dict[str, Any]) -> dict[str, bool]:
    methods = report.get("method_reports", [])
    processes = tuple(report.get("aggregate", {}).get("target_processes", []))
    rows = [row for method in methods for row in method.get("by_target_process", [])]
    method_ids = [str(method.get("method_id", "")) for method in methods]
    contract = report.get("contract", {})
    expected_comparisons = {
        "clustered_surface_minus_uniform_surface",
        "height_stratified_minus_uniform_surface",
    }
    return {
        "complete_paired_process_panel": (
            contract.get("complete_method_set") is True
            and len(method_ids) == len(PUBLIC_METHODS)
            and len(set(method_ids)) == len(method_ids)
            and set(method_ids) == set(PUBLIC_METHODS)
            and _exact_ids(contract.get("declared_method_ids"), PUBLIC_METHODS)
            and _exact_ids(contract.get("selected_method_ids"), PUBLIC_METHODS)
            and len(processes) == len(TARGET_PROCESSES)
            and len(set(processes)) == len(processes)
            and set(processes) == set(TARGET_PROCESSES)
            and len(rows) == len(PUBLIC_METHODS) * len(TARGET_PROCESSES)
            and all(
                len(method.get("by_target_process", [])) == len(TARGET_PROCESSES)
                and {
                    str(row.get("target_process", ""))
                    for row in method.get("by_target_process", [])
                }
                == set(TARGET_PROCESSES)
                for method in methods
            )
            and all(int(row.get("independent_ancestor_count", 0)) >= 5 for row in rows)
        ),
        "process_panel_safe_and_return": all(_safety_pass(row) for row in rows),
        "target_process_is_paired_private_safe_calibration": (
            contract.get("target_truth_visible_to_methods") is False
            and contract.get("l0_not_a_native_or_formal_score") is True
            and contract.get("paired_by_layout_ancestor") is True
            and contract.get("frozen_private_episode_replayed") is True
        ),
        "process_effect_is_measured_not_assumed": all(
            {
                str(row.get("comparison", ""))
                for row in method.get("paired_final_recall_deltas", [])
            }
            == expected_comparisons
            and len(method.get("paired_final_recall_deltas", []))
            == len(expected_comparisons)
            for method in methods
        ),
    }


def _prior_checks(report: dict[str, Any]) -> dict[str, bool]:
    aggregate = report.get("aggregate", [])
    prior_levels = [str(row.get("prior_level", "")) for row in aggregate]
    rows = {
        prior_level: row
        for prior_level, row in zip(prior_levels, aggregate, strict=True)
    }
    full = rows.get("full-cells", {})
    contract = report.get("contract", {})
    return {
        "complete_coarse_full_pair": (
            contract.get("complete_prior_set") is True
            and len(prior_levels) == len(PRIOR_METHODS)
            and len(set(prior_levels)) == len(prior_levels)
            and set(prior_levels) == set(PRIOR_METHODS)
            and _exact_ids(contract.get("declared_prior_levels"), set(PRIOR_METHODS))
            and _exact_ids(contract.get("selected_prior_levels"), set(PRIOR_METHODS))
            and all(
                rows[prior_level].get("method_id") == method_id
                for prior_level, method_id in PRIOR_METHODS.items()
            )
        ),
        "coarse_has_no_cells_or_poses": (
            contract.get("coarse_policy_receives_cells_or_poses") is False
        ),
        "prior_is_paired_private_safe_calibration": (
            contract.get("target_truth_visible_to_public_policy") is False
            and contract.get("l0_not_a_native_or_formal_score") is True
            and contract.get("same_private_episode_per_pair") is True
            and contract.get("same_strict_full_atlas_evaluator_per_pair") is True
        ),
        "full_prior_is_searchable": int(full.get("nonzero_ancestor_count", 0)) >= 4,
        "prior_pair_safe_and_return": all(_safety_pass(row) for row in rows.values()),
    }


def _scientific_checks(report: dict[str, Any], *, require_split: bool) -> dict[str, bool]:
    gates = report.get("gate_checks", {})
    leakage = report.get("leakage_report", {})
    checks = {
        "cpu_geometry_all_pass": gates.get("cpu_geometry_all_pass") is True,
        "paired_leakage_probe_pass": gates.get("paired_leakage_probe_pass") is True,
        "sector_process_leakage_probe_pass": (
            gates.get("sector_process_leakage_probe_pass") is True
        ),
        "sampling_policy_frozen": gates.get("sampling_policy_frozen") is True,
    }
    if require_split:
        checks.update(
            {
                "nine_independent_ancestors": int(
                    report.get("aggregate", {}).get("independent_ancestor_count", 0)
                )
                >= 9,
                "atlas_split_probe_pass": (
                    leakage.get("split_label_probe", {}).get("status")
                    == "PASS_NO_DETECTED_SIGNAL"
                ),
                "sector_split_probe_pass": (
                    leakage.get("sector_split_label_probe", {}).get("status")
                    == "PASS_NO_DETECTED_SIGNAL"
                ),
            }
        )
    else:
        checks["five_independent_ancestors"] = int(
            report.get("aggregate", {}).get("independent_ancestor_count", 0)
        ) >= 5
    return checks


def verify_a_gate(
    *,
    searchability: dict[str, Any],
    density: dict[str, Any],
    target_process: dict[str, Any],
    prior: dict[str, Any],
    scientific_audit: dict[str, Any],
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    manifest_hash = str(searchability["contract"]["calibration_manifest_hash"])
    implementation_hash = str(
        searchability["contract"]["calibration_implementation_hash"]
    )
    split_manifest_hash = str(split_audit.get("manifest_hash", ""))
    scientific_execution_hashes = {
        str(row.get("execution_contract_hash", ""))
        for report in (scientific_audit, split_audit)
        for row in report.get("geometry_reports", [])
    }
    scientific_execution_hashes.discard("")
    execution_contract_hash = (
        next(iter(scientific_execution_hashes))
        if len(scientific_execution_hashes) == 1
        else ""
    )
    binding_checks = {
        "same_calibration_manifest": (
            target_process.get("manifest_hash") == manifest_hash
            and prior.get("manifest_hash") == manifest_hash
            and scientific_audit.get("manifest_hash") == manifest_hash
            and density.get("source_manifest_hash") == manifest_hash
        ),
        "split_audit_is_independent_panel": split_manifest_hash != manifest_hash,
        "same_base_calibration_implementation": (
            density.get("base_calibration_implementation_hash") == implementation_hash
            and target_process.get("base_calibration_implementation_hash")
            == implementation_hash
            and prior.get("base_calibration_implementation_hash") == implementation_hash
        ),
        "same_nonempty_execution_contract": bool(execution_contract_hash),
    }
    check_groups = {
        "bindings": binding_checks,
        "searchability": _searchability_checks(searchability),
        "density": _density_checks(density),
        "target_process": _target_process_checks(target_process),
        "prior": _prior_checks(prior),
        "scientific": _scientific_checks(scientific_audit, require_split=False),
        "split_shortcuts": _scientific_checks(split_audit, require_split=True),
    }
    passed = all(value for group in check_groups.values() for value in group.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "gate": "A_TASK_SCIENTIFIC_CONTRACT",
        "status": "VERIFIED" if passed else "NO_GO",
        "formal_score_eligible": False,
        "authorizes_formal_test_access": False,
        "authorizes_next_gate": passed,
        "frozen_contract": {
            "calibration_manifest_hash": manifest_hash,
            "calibration_implementation_hash": implementation_hash,
            "sampling_policy_id": NOMINAL_POLICY,
            "mission_sector_capacity_fraction": 1.0,
            "episode_duration_s": 300.0,
            "execution_contract_hash": execution_contract_hash,
        },
        "checks": check_groups,
        "input_report_hashes": {
            "searchability": searchability["report_hash"],
            "density": density["report_hash"],
            "target_process": target_process["report_hash"],
            "prior": prior["report_hash"],
            "scientific_audit": scientific_audit["report_hash"],
            "split_audit": split_audit["report_hash"],
        },
        "failure_count": sum(
            not value for group in check_groups.values() for value in group.values()
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite A-gate evidence: {args.output}")
    report = verify_a_gate(
        searchability=_validated_report(args.searchability, SEARCH_SCHEMA),
        density=_validated_report(args.density, DENSITY_SCHEMA),
        target_process=_validated_report(args.target_process, TARGET_PROCESS_SCHEMA),
        prior=_validated_report(args.prior, PRIOR_SCHEMA),
        scientific_audit=_validated_report(args.scientific_audit, SCIENTIFIC_SCHEMA),
        split_audit=_validated_report(args.split_audit, SCIENTIFIC_SCHEMA),
    )
    write_json(args.output, report)
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
