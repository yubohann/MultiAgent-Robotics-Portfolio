"""Audit a resumable, balanced 10--20 city development scene cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import read_json, write_json_atomic  # noqa: E402
from aerocity_bench.ordinary_config import load_ordinary_config  # noqa: E402
from aerocity_bench.scene_audit import (  # noqa: E402
    audit_development_layout,
    development_scene_audit_plan,
    summarize_development_scene_audit_cohort,
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--layouts-per-split",
        type=int,
        default=4,
        help="4--6 yields the required 12--18 development-city audit cohort",
    )
    parser.add_argument("--receipt-directory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if not 4 <= args.layouts_per_split <= 6:
        raise ValueError("layouts-per-split must be in [4, 6] for a 12--18 city cohort")
    if args.summary.exists():
        raise FileExistsError(f"refusing to overwrite cohort evidence: {args.summary}")

    config = load_ordinary_config(args.config)
    assets = list(config.raw["assets"]["allowlist"])
    receipts: dict[tuple[str, int], object] = {}
    for split, index in development_scene_audit_plan(args.layouts_per_split):
        receipt_path = args.receipt_directory / split / f"layout-{index:03d}.json"
        if receipt_path.exists():
            receipts[(split, index)] = read_json(receipt_path)
            print(f"RESUME split={split} index={index}")
            continue
        report = audit_development_layout(
            config,
            split,
            index,
            assets,
            max_attempts=args.max_attempts,
        )
        write_json_atomic(receipt_path, report)
        receipts[(split, index)] = report
        print(f"{report['status']} split={split} index={index}")

    summary = summarize_development_scene_audit_cohort(
        config,
        receipts,
        per_split=args.layouts_per_split,
    )
    write_json_atomic(args.summary, summary)
    print(
        f"SCENE_AUDIT_COHORT={summary['status']} "
        f"layouts={summary['sampling']['layout_count']}"
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
