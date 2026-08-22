"""Assemble seven paired real Isaac P07 probes without claiming P07 closure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic
from aerocity_method.evaluation.hm3d_p07_matrix import (
    P07ProbeRecord,
    assemble_p07_task_validity_pilot,
)


def _parse_probe(value: str) -> tuple[str, Path]:
    method_id, separator, raw_path = value.partition("=")
    if separator != "=" or not method_id or not raw_path:
        raise argparse.ArgumentTypeError("--probe must be METHOD_ID=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"probe file is missing: {path}")
    return method_id, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--sensor-profile-sha256", required=True)
    parser.add_argument("--communication-contract-sha256", required=True)
    parser.add_argument("--probe", type=_parse_probe, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite task-pilot evidence: {output}")
    methods = [method for method, _ in args.probe]
    if len(methods) != len(set(methods)):
        raise ValueError("each P07 method may supply only one immutable raw probe")
    probes = tuple(
        P07ProbeRecord.from_raw(method, read_json_object(path)) for method, path in args.probe
    )
    payload = assemble_p07_task_validity_pilot(
        probes,
        matrix_run_id=args.matrix_run_id,
        sensor_profile_sha256=args.sensor_profile_sha256,
        communication_contract_sha256=args.communication_contract_sha256,
    )
    write_json_atomic(output, payload)
    print(f"P07_TASK_VALIDITY_PILOT_COMPLETE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
