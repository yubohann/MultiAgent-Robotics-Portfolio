"""Create a content-addressed, ancestor-level G2-I statistical planning report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import read_json, write_json  # noqa: E402
from aerocity_bench.statistical_protocol import build_statistical_planning_report  # noqa: E402


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite statistical evidence: {args.output}")
    report = build_statistical_planning_report(
        read_json(args.protocol), read_json(args.calibration_report)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"G2I_STATISTICAL_PLANNING={report['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
