"""Summarize the official Matterport metadata without treating area as length."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.adapters.hm3d_runtime import (
    load_official_metadata,
    summarize_official_metadata,
)
from aerocity_method.contracts.io import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_official_metadata(load_official_metadata(args.metadata))
    write_json_atomic(args.output, report)
    print(json.dumps({"scene_rows": report["scene_rows"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
