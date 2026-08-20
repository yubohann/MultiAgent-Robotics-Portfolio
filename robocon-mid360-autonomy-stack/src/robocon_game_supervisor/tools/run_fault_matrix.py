#!/usr/bin/env python3
"""Run the hardware-independent competition safety/fault matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocon_game_supervisor.fault_matrix import run_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fault_matrix.json"))
    args = parser.parse_args()
    result = run_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
