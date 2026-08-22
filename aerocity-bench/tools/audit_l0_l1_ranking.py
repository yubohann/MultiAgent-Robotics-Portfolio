"""Compare evaluator-produced L0/L1 score records by layout ancestor."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerocity_bench.canonical import read_json, write_json
from aerocity_bench.fidelity_audit import compare_l0_l1_rankings


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l0-records", type=Path, required=True)
    parser.add_argument("--l1-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    l0 = read_json(args.l0_records)
    l1 = read_json(args.l1_records)
    if not isinstance(l0, list) or not isinstance(l1, list):
        raise ValueError("L0/L1 ranking inputs must be JSON arrays")
    write_json(args.output, compare_l0_l1_rankings(l0, l1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
