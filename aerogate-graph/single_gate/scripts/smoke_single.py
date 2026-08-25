"""Smoke entry for the single-agent Graph-FlashSAC experiment."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _bootstrap_imports()
    from single_gate.smoke import run_single_smoke_test

    summary = run_single_smoke_test()
    print("single-agent smoke test complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

