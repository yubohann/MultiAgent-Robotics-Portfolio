"""Merge complete, implementation-bound target-process ablation shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, read_json, write_json

try:
    from tools.run_g2_i_target_process_ablation import PUBLIC_METHODS, REPORT_SCHEMA
except ModuleNotFoundError:
    from run_g2_i_target_process_ablation import PUBLIC_METHODS, REPORT_SCHEMA


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def merge_target_process_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one target-process shard is required")
    manifest_hashes: set[str] = set()
    base_implementation_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    process_sets: set[tuple[str, ...]] = set()
    reports: dict[str, dict[str, Any]] = {}
    source_hashes: list[str] = []
    ancestor_counts: set[int] = set()
    for shard in shards:
        if shard.get("schema") != REPORT_SCHEMA:
            raise ValueError("target-process shard schema differs")
        supplied_hash = str(shard.get("report_hash", ""))
        payload = dict(shard)
        payload.pop("report_hash", None)
        if content_hash(payload) != supplied_hash:
            raise ValueError("target-process shard report hash mismatch")
        if shard.get("formal_score_eligible") is not False:
            raise ValueError("target-process L0 shard cannot be formally score eligible")
        contract = shard.get("contract")
        aggregate = shard.get("aggregate")
        method_reports = shard.get("method_reports")
        if not isinstance(contract, dict) or not isinstance(aggregate, dict):
            raise ValueError("target-process shard contract or aggregate is missing")
        if not isinstance(method_reports, list) or not method_reports:
            raise ValueError("target-process shard has no method reports")
        manifest_hashes.add(str(shard.get("manifest_hash", "")))
        base_implementation_hashes.add(
            str(shard.get("base_calibration_implementation_hash", ""))
        )
        implementation_hashes.add(
            str(shard.get("target_process_implementation_hash", ""))
        )
        process_sets.add(tuple(aggregate.get("target_processes", [])))
        ancestor_counts.add(int(aggregate["independent_ancestor_count"]))
        for report in method_reports:
            method_id = str(report.get("method_id", ""))
            if method_id in reports:
                raise ValueError("target-process method appears in multiple shards")
            reports[method_id] = report
        source_hashes.append(supplied_hash)
    if "" in manifest_hashes or len(manifest_hashes) != 1:
        raise ValueError("target-process shards do not bind the same manifest")
    if "" in base_implementation_hashes or len(base_implementation_hashes) != 1:
        raise ValueError(
            "target-process shards do not bind the same base implementation"
        )
    if "" in implementation_hashes or len(implementation_hashes) != 1:
        raise ValueError("target-process shards do not bind the same implementation")
    if len(process_sets) != 1 or len(ancestor_counts) != 1:
        raise ValueError("target-process shard condition panels differ")
    if set(reports) != set(PUBLIC_METHODS):
        raise ValueError("target-process shards lack the complete public method set")
    merged: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_ONLY",
        "manifest_hash": next(iter(manifest_hashes)),
        "base_calibration_implementation_hash": next(iter(base_implementation_hashes)),
        "target_process_implementation_hash": next(iter(implementation_hashes)),
        "execution_level": "L0",
        "contract": {
            "declared_method_ids": list(PUBLIC_METHODS),
            "selected_method_ids": list(PUBLIC_METHODS),
            "complete_method_set": True,
            "frozen_private_episode_replayed": True,
            "target_truth_visible_to_methods": False,
            "target_process_labels_are_internal_calibration_conditions": True,
            "paired_by_layout_ancestor": True,
            "episode_replicates_are_not_independent": True,
            "l0_not_a_native_or_formal_score": True,
        },
        "aggregate": {
            "independent_ancestor_count": next(iter(ancestor_counts)),
            "target_processes": list(next(iter(process_sets))),
            "method_count": len(PUBLIC_METHODS),
            "record_count": (
                next(iter(ancestor_counts)) * len(next(iter(process_sets))) * len(PUBLIC_METHODS)
            ),
        },
        "method_reports": [reports[method_id] for method_id in PUBLIC_METHODS],
        "raw_records_omitted_for_privacy": True,
        "shard_merge": {
            "source_report_hashes": sorted(source_hashes),
            "complete_public_method_set": True,
        },
    }
    merged["report_hash"] = content_hash(merged)
    return merged


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite merged target-process evidence: {args.output}"
        )
    write_json(
        args.output,
        merge_target_process_shards([read_json(path) for path in args.shards]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
