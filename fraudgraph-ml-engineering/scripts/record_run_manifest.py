"""Record a reproducible experiment manifest without launching model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_ml_engineering.run_artifacts import build_run_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path, usually under artifacts/.")
    parser.add_argument("--dataset", required=True, help="Dataset identifier used by the planned experiment.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, if the run uses one.")
    parser.add_argument("--config", type=Path, default=None, help="Optional experiment configuration path.")
    parser.add_argument("--note", default="", help="Optional protocol or data-revision note.")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to record. Put `--` before it so argparse preserves every flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    manifest = build_run_manifest(
        dataset=args.dataset,
        seed=args.seed,
        command=command,
        config_path=args.config,
        notes=args.note,
    )
    write_json(args.output, manifest)
    print(json.dumps({"manifest_path": str(args.output), "repository_commit": manifest["repository_commit"]}, indent=2))


if __name__ == "__main__":
    main()
