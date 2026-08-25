"""Audit the P01-P10 HM3D formal-experiment preflight.

Exit codes:
  0: preventive contract passed, formal start is ready, or formal results are ready
  2: contract exists but required real-runtime evidence is incomplete
  8: malformed protocol or evidence manifest
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import write_json_atomic
from aerocity_method.evaluation.hm3d_preflight import (
    audit_hm3d_formal_preflight,
    audit_preflight_contract,
    load_preflight_evidence,
    load_preflight_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed P01-P10 audit for HM3D formal experiments"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json",
    )
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        protocol = load_preflight_protocol(args.protocol)
        contract = audit_preflight_contract(protocol)
        if args.contract_only:
            report = contract
            exit_code = 0
        else:
            evidence = None if args.evidence is None else load_preflight_evidence(args.evidence)
            evidence_root = ROOT if args.evidence is None else args.evidence.resolve().parent
            runtime = audit_hm3d_formal_preflight(
                protocol,
                evidence,
                evidence_root=evidence_root,
            )
            report = {"contract": contract, **runtime}
            exit_code = 0 if runtime["status"] != "RUNTIME_NOT_READY" else 2
        write_json_atomic(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return exit_code
    except (OSError, TypeError, ValueError) as exc:
        report = {
            "status": "RUNTIME_NOT_READY",
            "formal_experiment_start_authorized": False,
            "formal_results_authorized": False,
            "error": str(exc),
        }
        write_json_atomic(args.output, report)
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 8


if __name__ == "__main__":
    raise SystemExit(main())
