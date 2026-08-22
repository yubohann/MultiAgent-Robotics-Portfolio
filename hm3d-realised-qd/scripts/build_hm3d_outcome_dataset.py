"""Build the shared train-only decision outcome dataset manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic  # noqa: E402
from aerocity_method.evaluation.hm3d_outcome_dataset import (  # noqa: E402
    build_outcome_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p05-artifact", required=True, type=Path)
    parser.add_argument("--record", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_outcome_dataset_manifest(
        args.record,
        split_manifest=read_json_object(args.p05_artifact),
    )
    write_json_atomic(args.output, manifest)
    print(
        f"indexed {manifest['real_decision_count']} real decisions from "
        f"{manifest['physical_episode_count']} episodes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
