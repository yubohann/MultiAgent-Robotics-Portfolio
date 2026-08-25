"""Create a new immutable P01--P05 manifest for the current protocol.

This tool never edits historical runtime evidence.  It can only bind existing
phase envelopes to the hash of the currently checked-in preflight protocol.
The formal preflight audit remains the authority that validates each payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.evaluation.hm3d_preflight import (  # noqa: E402
    METHOD_CORE,
    PHASE_SPECS,
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    load_preflight_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_envelope(path: Path, *, phase_id: str, kind: str, origin: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{phase_id} is not a JSON object: {path}")
    required = {
        "schema_version",
        "phase_id",
        "kind",
        "origin",
        "measured",
        "synthetic",
        "denominator_complete",
        "payload",
    }
    if set(payload) != required:
        raise ValueError(f"{phase_id} envelope fields do not match the preflight schema")
    if (payload["phase_id"], payload["kind"], payload["origin"]) != (phase_id, kind, origin):
        raise ValueError(f"{phase_id} envelope identity does not match its requested binding")
    if payload["measured"] is not True or payload["synthetic"] is not False:
        raise ValueError(f"{phase_id} is not admissible measured evidence")
    if payload["denominator_complete"] is not True:
        raise ValueError(f"{phase_id} denominator is incomplete")
    if not isinstance(payload["payload"], dict):
        raise ValueError(f"{phase_id} envelope payload is not an object")


def build_manifest(protocol_path: Path, references: dict[str, Path]) -> dict[str, Any]:
    protocol = load_preflight_protocol(protocol_path)
    required_specs = PHASE_SPECS[:5]
    if tuple(references) != tuple(phase for phase, _, _ in required_specs):
        raise ValueError("P01--P05 references must be supplied in phase order")
    artifacts: list[dict[str, str]] = []
    for phase_id, kind, origin in required_specs:
        path = references[phase_id].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{phase_id} evidence is missing: {path}")
        _read_envelope(path, phase_id=phase_id, kind=kind, origin=origin)
        artifacts.append(
            {
                "phase_id": phase_id,
                "kind": kind,
                "origin": origin,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "protocol_hash": protocol.protocol_hash,
        "requested_gate": "formal_experiment_start",
        "method_core": METHOD_CORE,
        "aerocity_bench_accesses": [],
        "artifacts": artifacts,
    }


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json",
    )
    for phase_id, _, _ in PHASE_SPECS[:5]:
        parser.add_argument(f"--{phase_id.lower()}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = {phase_id: getattr(args, phase_id.lower()) for phase_id, _, _ in PHASE_SPECS[:5]}
    manifest = build_manifest(args.protocol.expanduser().resolve(), references)
    output = args.output.expanduser().resolve()
    _write_new(output, manifest)
    print(
        json.dumps(
            {
                "status": "CURRENT_PROTOCOL_EVIDENCE_MANIFEST_CREATED",
                "output": str(output),
                "protocol_hash": manifest["protocol_hash"],
                "phases": [row["phase_id"] for row in manifest["artifacts"]],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
