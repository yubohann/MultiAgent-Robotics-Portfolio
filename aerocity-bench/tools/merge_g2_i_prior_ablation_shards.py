"""Merge complete, implementation-bound G2-I prior ablation shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, read_json, write_json

try:
    from tools.run_g2_i_prior_ablation import PRIOR_LEVELS, REPORT_SCHEMA
except ModuleNotFoundError:
    from run_g2_i_prior_ablation import PRIOR_LEVELS, REPORT_SCHEMA


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def merge_prior_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one prior shard is required")
    manifest_hashes: set[str] = set()
    base_implementation_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    aggregate: dict[str, dict[str, Any]] = {}
    source_hashes: list[str] = []
    for shard in shards:
        if shard.get("schema") != REPORT_SCHEMA:
            raise ValueError("prior shard schema differs")
        supplied_hash = str(shard.get("report_hash", ""))
        payload = dict(shard)
        payload.pop("report_hash", None)
        if content_hash(payload) != supplied_hash:
            raise ValueError("prior shard report hash mismatch")
        if shard.get("formal_score_eligible") is not False:
            raise ValueError("prior L0 shard cannot be formally score eligible")
        reports = shard.get("aggregate")
        if not isinstance(reports, list) or not reports:
            raise ValueError("prior shard has no aggregate report")
        manifest_hashes.add(str(shard.get("manifest_hash", "")))
        base_implementation_hashes.add(
            str(shard.get("base_calibration_implementation_hash", ""))
        )
        implementation_hashes.add(str(shard.get("prior_implementation_hash", "")))
        for report in reports:
            prior_level = str(report.get("prior_level", ""))
            if prior_level in aggregate:
                raise ValueError("prior level appears in multiple shards")
            aggregate[prior_level] = report
        source_hashes.append(supplied_hash)
    if "" in manifest_hashes or len(manifest_hashes) != 1:
        raise ValueError("prior shards do not bind the same manifest")
    if "" in base_implementation_hashes or len(base_implementation_hashes) != 1:
        raise ValueError("prior shards do not bind the same base implementation")
    if "" in implementation_hashes or len(implementation_hashes) != 1:
        raise ValueError("prior shards do not bind the same implementation")
    if set(aggregate) != set(PRIOR_LEVELS):
        raise ValueError("prior shards lack the complete prior-level set")
    merged: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "DIAGNOSTIC_ONLY",
        "manifest_hash": next(iter(manifest_hashes)),
        "base_calibration_implementation_hash": next(iter(base_implementation_hashes)),
        "prior_implementation_hash": next(iter(implementation_hashes)),
        "execution_level": "L0",
        "contract": {
            "declared_prior_levels": list(PRIOR_LEVELS),
            "selected_prior_levels": list(PRIOR_LEVELS),
            "complete_prior_set": True,
            "same_private_episode_per_pair": True,
            "same_strict_full_atlas_evaluator_per_pair": True,
            "coarse_policy_receives_cells_or_poses": False,
            "target_truth_visible_to_public_policy": False,
            "paired_execution_ablation_not_method_ranking": True,
            "l0_not_a_native_or_formal_score": True,
        },
        "aggregate": [aggregate[prior_level] for prior_level in PRIOR_LEVELS],
        "raw_records_omitted_for_privacy": True,
        "shard_merge": {
            "source_report_hashes": sorted(source_hashes),
            "complete_prior_level_set": True,
        },
    }
    merged["report_hash"] = content_hash(merged)
    return merged


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite merged prior evidence: {args.output}")
    write_json(args.output, merge_prior_shards([read_json(path) for path in args.shards]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
