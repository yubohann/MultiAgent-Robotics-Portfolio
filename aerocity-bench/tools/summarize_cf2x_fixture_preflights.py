"""Create a public-only aggregate for three local CF2X closure fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerocity_bench.cf2x_fixture_aggregate import write_closed_private_fixture_aggregate
from aerocity_bench.errors import ValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True, help="Public calibration report")
    parser.add_argument("--train", type=Path, required=True, help="Public train report")
    parser.add_argument("--validation", type=Path, required=True, help="Public validation report")
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh local public aggregate path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        aggregate = write_closed_private_fixture_aggregate(
            {
                "calibration": args.calibration,
                "train": args.train,
                "validation": args.validation,
            },
            args.output,
        )
    except (OSError, ValidationError, ValueError) as exc:
        print(f"fixture aggregate rejected: {exc}")
        return 2
    print(
        json.dumps(
            {
                "result": aggregate["result"],
                "fixture_count": aggregate["fixture_count"],
                "aggregate_sha256": aggregate["aggregate_sha256"],
                "formal_score_eligible": aggregate["formal_score_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
