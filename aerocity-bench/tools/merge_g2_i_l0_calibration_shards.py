"""Merge independently executed G2-I L0 method shards fail closed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import read_json, write_json
from tools.run_g2_i_l0_calibration import METHODS, REPORT_SCHEMA, _assemble_report


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def merge_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one calibration shard is required")
    manifests = set()
    implementations = set()
    episode_durations = set()
    max_steps_values = set()
    record_counts = set()
    method_reports: list[dict[str, Any]] = []
    seen_methods: set[str] = set()
    for shard in shards:
        if shard.get("schema") != REPORT_SCHEMA:
            raise ValueError("calibration shard schema differs")
        if shard.get("formal_score_eligible") is not False:
            raise ValueError("L0 calibration shard cannot be formally score eligible")
        contract = shard.get("contract")
        aggregate = shard.get("aggregate")
        reports = shard.get("method_reports")
        if not isinstance(contract, dict) or not isinstance(aggregate, dict):
            raise ValueError("calibration shard contract or aggregate is missing")
        if not isinstance(reports, list) or not reports:
            raise ValueError("calibration shard has no method reports")
        manifests.add(str(contract.get("calibration_manifest", "")))
        implementations.add(str(contract.get("calibration_implementation", "")))
        episode_durations.add(float(contract["episode_duration_s"]))
        max_steps_values.add(contract.get("max_steps"))
        record_counts.add(int(aggregate["record_count"]))
        for report in reports:
            method_id = str(report.get("method_id", ""))
            if method_id in seen_methods:
                raise ValueError(f"calibration method appears in multiple shards: {method_id}")
            seen_methods.add(method_id)
            method_reports.append(report)
    if "" in manifests or len(manifests) != 1:
        raise ValueError("calibration shards do not bind the same manifest")
    if "" in implementations or len(implementations) != 1:
        raise ValueError("calibration shards do not bind the same implementation")
    if len(episode_durations) != 1 or len(max_steps_values) != 1 or len(record_counts) != 1:
        raise ValueError("calibration shard execution contracts differ")
    if seen_methods != set(METHODS):
        raise ValueError("calibration shards do not contain the complete canonical method set")
    merged = _assemble_report(
        calibration_manifest=next(iter(manifests)),
        calibration_implementation=next(iter(implementations)),
        episode_duration_s=next(iter(episode_durations)),
        max_steps=next(iter(max_steps_values)),
        record_count=next(iter(record_counts)),
        method_reports=method_reports,
    )
    merged["shard_merge"] = {
        "source_methods": sorted(seen_methods),
        "complete_canonical_method_set": True,
    }
    return merged


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite merged evidence: {args.output}")
    merged = merge_shards([read_json(path) for path in args.shards])
    write_json(args.output, merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
