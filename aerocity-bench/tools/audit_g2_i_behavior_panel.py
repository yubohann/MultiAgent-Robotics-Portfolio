"""Audit cheap L0 profiles for exact public-action equivalence before L1."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerocity_bench.behavioral_distinctness import (
    audit_method_panel_behavior,
    audit_method_panel_behavior_cohort,
)
from aerocity_bench.canonical import read_json, write_json


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def behavior_distinct_count(result: dict[str, object]) -> int:
    """Return the comparable distinct-count field for either audit schema."""

    if "distinct_mechanism_lower_bound" in result:
        return int(result["distinct_mechanism_lower_bound"])
    if "distinct_deterministic_behavior_count" in result:
        return int(result["distinct_deterministic_behavior_count"])
    raise ValueError("behavior panel audit lacks a distinct-count field")


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    reports = [read_json(path.resolve()) for path in args.profiles]
    contexts = {
        (str(report.get("layout_hash", "")), str(report.get("episode_hash", "")))
        for report in reports
    }
    result = (
        audit_method_panel_behavior_cohort(reports)
        if len(contexts) > 1
        else audit_method_panel_behavior(reports)
    )
    write_json(args.output, result)
    print(
        "behavior panel audit written: "
        f"status={result['status']} methods={result['method_count']} "
        f"distinct={behavior_distinct_count(result)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
