"""Admit an assembled real H15 matrix as immutable P06 preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic  # noqa: E402
from aerocity_method.evaluation.hm3d_preflight import (  # noqa: E402
    PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
    audit_hm3d_formal_preflight,
    load_preflight_evidence,
    load_preflight_protocol,
)
from aerocity_method.runtime.sensors import FORMAL_H15_SENSOR_PILOT_MODES  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p06-payload", type=Path, required=True)
    parser.add_argument("--assembly-audit", type=Path, required=True)
    parser.add_argument("--prior-evidence", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json",
    )
    return parser.parse_args()


def _require_new(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {label}: {path}")


def _validate_assembly(p06: dict[str, Any], audit: dict[str, Any]) -> None:
    if audit.get("status") != "H15_ASSEMBLY_PASS":
        raise ValueError("H15 assembly audit did not pass")
    matrix = audit.get("matrix")
    required_rows = len(FORMAL_H15_SENSOR_PILOT_MODES) * 3
    if (
        not isinstance(matrix, dict)
        or matrix.get("status") != "PASS"
        or matrix.get("rows") != required_rows
    ):
        raise ValueError("H15 assembly audit does not contain the complete camera-free matrix")
    selected = p06.get("selected_profile")
    if not isinstance(selected, dict) or selected != audit.get("selected_profile"):
        raise ValueError("P06 selected sensor profile differs from the assembly audit")
    row_files = audit.get("row_files")
    if not isinstance(row_files, list) or len(row_files) != required_rows:
        raise ValueError("H15 assembly audit lacks all immutable worker-row hashes")
    for row in row_files:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
        ):
            raise ValueError("H15 assembly audit row reference is malformed")
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"H15 worker row hash mismatch: {path}")
    ledger = audit.get("matrix_ledger")
    if not isinstance(ledger, dict) or ledger.get("status") != "H15_MATRIX_COMPLETE":
        raise ValueError("H15 assembly audit lacks a complete worker-exit ledger")
    ledger_path = ledger.get("path")
    ledger_hash = ledger.get("sha256")
    if not isinstance(ledger_path, str) or not isinstance(ledger_hash, str):
        raise ValueError("H15 matrix ledger reference is malformed")
    if not Path(ledger_path).is_file() or _sha256(Path(ledger_path)) != ledger_hash:
        raise ValueError("H15 matrix ledger hash mismatch")


def main() -> int:
    args = parse_args()
    p06_path = args.p06_payload.expanduser().resolve()
    audit_path = args.assembly_audit.expanduser().resolve()
    prior_path = args.prior_evidence.expanduser().resolve()
    artifact_path = args.artifact_output.expanduser().resolve()
    evidence_path = args.evidence_output.expanduser().resolve()
    report_path = args.report_output.expanduser().resolve()
    required_inputs = (
        (p06_path, "P06 payload"),
        (audit_path, "assembly audit"),
        (prior_path, "prior evidence"),
        (args.protocol.expanduser().resolve(), "protocol"),
    )
    for path, label in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    required_outputs = (
        (artifact_path, "P06 artifact"),
        (evidence_path, "P06 evidence manifest"),
        (report_path, "P06 preflight report"),
    )
    for path, label in required_outputs:
        _require_new(path, label)

    p06 = read_json_object(p06_path)
    assembly_audit = read_json_object(audit_path)
    _validate_assembly(p06, assembly_audit)
    prior = read_json_object(prior_path)
    artifacts = prior.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("prior preflight evidence has no artifact list")
    if any(row.get("phase_id") == "P06" for row in artifacts if isinstance(row, dict)):
        raise ValueError("prior evidence already includes P06")

    artifact = {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "phase_id": "P06",
        "kind": "sensor_h15_pilot",
        "origin": "real_runtime",
        "measured": True,
        "synthetic": False,
        "denominator_complete": True,
        "payload": p06,
    }
    write_json_atomic(artifact_path, artifact)
    try:
        combined = dict(prior)
        combined["artifacts"] = [
            *artifacts,
            {
                "phase_id": "P06",
                "kind": "sensor_h15_pilot",
                "origin": "real_runtime",
                "path": str(artifact_path),
                "sha256": _sha256(artifact_path),
            },
        ]
        write_json_atomic(evidence_path, combined)
        protocol = load_preflight_protocol(args.protocol)
        evidence = load_preflight_evidence(evidence_path)
        report = audit_hm3d_formal_preflight(protocol, evidence, evidence_root=evidence_path.parent)
        p06_row = next(row for row in report["phases"] if row["phase_id"] == "P06")
        if p06_row["status"] != "READY":
            raise ValueError(f"P06 admission failed: {p06_row['reasons']}")
        write_json_atomic(report_path, report)
    except BaseException:
        evidence_path.unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "P06_READY",
                "artifact": str(artifact_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
