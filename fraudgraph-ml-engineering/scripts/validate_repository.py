#!/usr/bin/env python3
"""Lightweight framework validator for this repository.

Checks structure, hygiene, and imports without requiring the graph-learning
runtime (dgl / torch are only needed by the withheld core).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
CORE = REPO_ROOT / "core"

NEUTRAL_MODULES = [
    "fraud_ml_engineering",
    "fraud_ml_engineering.paths",
    "fraud_ml_engineering.run_artifacts",
    "fraud_ml_engineering.experiment_protocol",
]


def _check(name: str, ok: bool) -> None:
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def main() -> int:
    results = []
    results.append(_check("structure: src package exists", SRC.is_dir()))
    results.append(_check("structure: core with withheld notice", (CORE / "__init__.py").exists()))
    results.append(_check("structure: NOTICE.md present", (REPO_ROOT / "NOTICE.md").exists()))
    results.append(_check("structure: README present", (REPO_ROOT / "README.md").exists()))

    ok = True
    for py in SRC.rglob("*.py"):
        try:
            ast.parse(py.read_text(encoding="utf-8-sig"))
        except SyntaxError as exc:
            ok = False
            print(f"  syntax error in {py.relative_to(REPO_ROOT)}: {exc}")
    results.append(_check("source hygiene: all src modules parse", ok))

    try:
        sys.path.insert(0, str(SRC))
        import fraud_ml_engineering  # noqa: F401
        import fraud_ml_engineering.paths  # noqa: F401
        import fraud_ml_engineering.run_artifacts  # noqa: F401
        import fraud_ml_engineering.experiment_protocol  # noqa: F401
        results.append(_check("imports: neutral modules import", True))
    except Exception as exc:  # pragma: no cover
        results.append(_check(f"imports: neutral modules import ({exc})", False))

    print("Framework repository validation passed." if all(results) else "Validation FAILED.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())