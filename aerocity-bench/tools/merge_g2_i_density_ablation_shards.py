"""Merge complete, implementation-bound G2-I density calibration shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, read_json, write_json

try:
    from tools.run_g2_i_density_ablation import PUBLIC_METHODS, REPORT_SCHEMA
except ModuleNotFoundError:
    from run_g2_i_density_ablation import PUBLIC_METHODS, REPORT_SCHEMA

DENSITY_PAIRING_FIELDS = (
    "policy_specific_sector_frozen_before_private_sampling",
    "private_target_realizations_matched_across_density_conditions",
    "private_targets_reused_from_source_manifest",
    "public_region_cohort_matched_across_density_conditions",
    "selected_city_panel_matched_across_density_conditions",
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def merge_density_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one density shard is required")
    manifest_hashes: set[str] = set()
    source_manifest_hashes: set[str] = set()
    base_implementation_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    declared_policy_sets: set[tuple[str, ...]] = set()
    condition_reports: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes = []
    for shard in shards:
        if shard.get("schema") != REPORT_SCHEMA:
            raise ValueError("density shard schema differs")
        supplied_hash = str(shard.get("report_hash", ""))
        payload = dict(shard)
        payload.pop("report_hash", None)
        if content_hash(payload) != supplied_hash:
            raise ValueError("density shard report hash mismatch")
        if shard.get("formal_score_eligible") is not False:
            raise ValueError("density L0 shard cannot be formally score eligible")
        contract = shard.get("contract")
        reports = shard.get("method_reports")
        if not isinstance(contract, dict) or not isinstance(reports, list):
            raise ValueError("density shard contract or reports are missing")
        if contract.get("target_truth_visible_to_methods") is not False or not all(
            contract.get(field) is True for field in DENSITY_PAIRING_FIELDS
        ):
            raise ValueError("density shard lacks the paired private-safe calibration contract")
        manifest_hashes.add(str(shard.get("manifest_hash", "")))
        source_manifest_hashes.add(str(shard.get("source_manifest_hash", "")))
        base_implementation_hashes.add(
            str(shard.get("base_calibration_implementation_hash", ""))
        )
        implementation_hashes.add(
            str(shard.get("density_implementation_hash", ""))
        )
        declared_policy_sets.add(tuple(contract.get("declared_policy_ids", [])))
        for report in reports:
            key = (str(report["sampling_policy_id"]), str(report["method_id"]))
            if key in condition_reports:
                raise ValueError("density condition appears in multiple shards")
            condition_reports[key] = report
        source_hashes.append(supplied_hash)
    if "" in manifest_hashes or len(manifest_hashes) != 1:
        raise ValueError("density shards do not bind the same manifest")
    if "" in source_manifest_hashes or len(source_manifest_hashes) != 1:
        raise ValueError("density shards do not bind the same source manifest")
    if "" in base_implementation_hashes or len(base_implementation_hashes) != 1:
        raise ValueError("density shards do not bind the same base implementation")
    if "" in implementation_hashes or len(implementation_hashes) != 1:
        raise ValueError("density shards do not bind the same implementation")
    if len(declared_policy_sets) != 1:
        raise ValueError("density shards declare different policy sets")
    declared_policy_ids = next(iter(declared_policy_sets))
    expected = {
        (policy_id, method_id)
        for policy_id in declared_policy_ids
        for method_id in PUBLIC_METHODS
    }
    if set(condition_reports) != expected:
        raise ValueError("density shards do not contain every policy-method condition")
    ordered_reports = [
        condition_reports[(policy_id, method_id)]
        for policy_id in declared_policy_ids
        for method_id in PUBLIC_METHODS
    ]
    ancestor_counts = {
        int(report["independent_ancestor_count"]) for report in ordered_reports
    }
    if len(ancestor_counts) != 1:
        raise ValueError("density conditions use different ancestor counts")
    merged: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_ONLY",
        "manifest_hash": next(iter(manifest_hashes)),
        "source_manifest_hash": next(iter(source_manifest_hashes)),
        "base_calibration_implementation_hash": next(iter(base_implementation_hashes)),
        "density_implementation_hash": next(iter(implementation_hashes)),
        "execution_level": "L0",
        "contract": {
            "declared_policy_ids": list(declared_policy_ids),
            "selected_policy_ids": list(declared_policy_ids),
            "selected_method_ids": list(PUBLIC_METHODS),
            "complete_condition_set": True,
            "target_truth_visible_to_methods": False,
            **{field: True for field in DENSITY_PAIRING_FIELDS},
            "episode_replicates_are_not_independent": True,
            "density_comparison_is_task_calibration_not_method_ranking": True,
            "l0_not_a_native_or_formal_score": True,
        },
        "aggregate": {
            "sampling_policy_ids": list(declared_policy_ids),
            "independent_ancestor_count": next(iter(ancestor_counts)),
            "method_count": len(PUBLIC_METHODS),
            "condition_count": len(ordered_reports),
        },
        "method_reports": ordered_reports,
        "raw_records_omitted_for_privacy": True,
        "shard_merge": {
            "source_report_hashes": sorted(source_hashes),
            "complete_policy_method_grid": True,
        },
    }
    merged["report_hash"] = content_hash(merged)
    return merged


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite density evidence: {args.output}")
    write_json(
        args.output, merge_density_shards([read_json(path) for path in args.shards])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
