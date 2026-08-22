"""Assemble immutable P07 no-QD/planned-QD/realised-QD validation records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.evaluation.hm3d_p08_qd_matrix import (  # noqa: E402
    P08QDUnit,
    assemble_p08_qd_paired_evidence,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _parse_record(value: str) -> tuple[str, Path]:
    unit_id, separator, raw_path = value.partition("=")
    path = Path(raw_path).expanduser().resolve()
    if separator != "=" or not unit_id or not path.is_file():
        raise argparse.ArgumentTypeError("record must be UNIT_ID=EXISTING_JSON_PATH")
    return unit_id, path


def _write_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable P08 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-qd", type=_parse_record, action="append", required=True)
    parser.add_argument("--planned-qd", type=_parse_record, action="append", required=True)
    parser.add_argument("--realised-qd", type=_parse_record, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    branches = {
        "no_qd": dict(args.no_qd),
        "planned_qd": dict(args.planned_qd),
        "realised_qd": dict(args.realised_qd),
    }
    unit_ids = set(branches["no_qd"])
    if not unit_ids or any(set(branch) != unit_ids for branch in branches.values()):
        raise ValueError("all three QD branches must provide exactly the same unit IDs")
    units = tuple(
        P08QDUnit(
            unit_id=unit_id,
            no_qd=_read(branches["no_qd"][unit_id]),
            planned_qd=_read(branches["planned_qd"][unit_id]),
            realised_qd=_read(branches["realised_qd"][unit_id]),
        )
        for unit_id in sorted(unit_ids)
    )
    payload = assemble_p08_qd_paired_evidence(units)
    _write_new(args.output.expanduser().resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if payload["status"] == "P08_QD_PILOT_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
