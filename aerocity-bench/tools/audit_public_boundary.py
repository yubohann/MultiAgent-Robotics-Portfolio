"""Audit a materialized method-public layout before an external-method run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, write_json  # noqa: E402
from aerocity_bench.public_boundary import audit_public_layout  # noqa: E402


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        report = audit_public_layout(args.layout_root)
    except (OSError, TypeError, ValueError) as exc:
        report = {
            "schema": "org.aerocity.bench.public-boundary-audit.v1",
            "status": "FAIL",
            "formal_score_eligible": False,
            "failure_stage": "public_artifact_validation",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
    report["tool_source_sha256"] = content_hash(Path(__file__).read_text(encoding="utf-8"))
    report["report_hash"] = content_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"PUBLIC_BOUNDARY_AUDIT={report['status']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
