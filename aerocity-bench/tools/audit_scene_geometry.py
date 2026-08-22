"""Write one private-safe static scene-admission report for a development layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerocity_bench.canonical import write_json_atomic
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.scene_audit import audit_development_layout


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config = load_ordinary_config(args.config)
    report = audit_development_layout(
        config,
        args.split,
        args.index,
        list(config.raw["assets"]["allowlist"]),
        max_attempts=args.max_attempts,
    )
    write_json_atomic(args.output, report)
    print(
        f"{report['status']} split={report['split']} "
        f"layout={report.get('layout_id', 'none')} hash={report['report_hash']}"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
