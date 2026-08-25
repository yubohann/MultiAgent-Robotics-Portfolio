"""Audit P07 JSON records by field and permitted downstream evidence use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic  # noqa: E402
from aerocity_method.evaluation.hm3d_evidence_classification import (  # noqa: E402
    audit_p07_record_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, action="append", type=Path)
    parser.add_argument(
        "--known-defect",
        action="append",
        default=[],
        help="Apply one documented defect code to all supplied records.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for raw_path in args.record:
        path = raw_path.expanduser().resolve()
        payload = read_json_object(path)
        rows.append(
            {
                "path": str(path),
                "classification": audit_p07_record_evidence(
                    payload, known_defects=tuple(args.known_defect)
                ),
            }
        )
    report = {
        "schema_version": "hm3d-p07-evidence-eligibility-report-v1",
        "claim_limit": (
            "Field-level evidence routing only. Eligibility does not convert development "
            "episodes into frozen formal performance results."
        ),
        "record_count": len(rows),
        "records": rows,
    }
    write_json_atomic(args.output.expanduser().resolve(), report)
    print(json.dumps({"status": "AUDIT_COMPLETE", "record_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
