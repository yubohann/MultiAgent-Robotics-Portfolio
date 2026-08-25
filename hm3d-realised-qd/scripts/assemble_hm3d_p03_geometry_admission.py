"""Assemble measured P03 geometry evidence before P04 observation collection.

P03 proves a scene is a valid 3D CF2X flight space.  P04 then uses that fixed
geometry to collect public sparse-range outcomes.  This explicit stage breaks
the former P03<->P04 circular dependency without treating a planned route or
an observation-free synthetic record as runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_hm3d_p03_p05_admission import build_p03_scene_row  # noqa: E402

from aerocity_method.contracts.io import write_json_atomic  # noqa: E402
from aerocity_method.evaluation.hm3d_preflight import (  # noqa: E402
    PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _command_sha256() -> str:
    return hashlib.sha256("\0".join(sys.argv).encode("utf-8")).hexdigest()


def _file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p01", type=Path, required=True)
    parser.add_argument("--p02", type=Path, required=True)
    parser.add_argument("--train-flight-space", type=Path, required=True)
    parser.add_argument("--validation-flight-space", type=Path, required=True)
    parser.add_argument("--train-vertical-counterfactual", type=Path, required=True)
    parser.add_argument("--validation-vertical-counterfactual", type=Path, required=True)
    parser.add_argument("--train-collision-replay", type=Path, required=True)
    parser.add_argument("--validation-collision-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p01_path = _file(args.p01, "P01")
    p02_path = _file(args.p02, "P02")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P03 artifact: {output}")

    p01 = _read(p01_path)
    p02 = _read(p02_path)
    if p01.get("phase_id") != "P01" or p02.get("phase_id") != "P02":
        raise ValueError("P01/P02 artifact identity mismatch")
    p01_payload = p01.get("payload")
    if not isinstance(p01_payload, dict) or not isinstance(p01_payload.get("scenes"), list):
        raise ValueError("P01 asset lock lacks its scene list")
    locked_rows = p01_payload["scenes"]
    locked = {
        row.get("scene_id"): row
        for row in locked_rows
        if isinstance(row, dict) and isinstance(row.get("scene_id"), str)
    }
    if len(locked) != len(locked_rows):
        raise ValueError("P01 asset lock contains malformed or duplicate scene rows")

    pair_paths = (
        (
            args.train_flight_space,
            args.train_vertical_counterfactual,
            args.train_collision_replay,
        ),
        (
            args.validation_flight_space,
            args.validation_vertical_counterfactual,
            args.validation_collision_replay,
        ),
    )
    rows: list[dict[str, Any]] = []
    for index, (flight, vertical, replay) in enumerate(pair_paths, start=1):
        rows.append(
            build_p03_scene_row(
                locked=locked,
                flight=_read(_file(flight, f"cohort {index} flight-space audit")),
                vertical_counterfactual=_read(
                    _file(vertical, f"cohort {index} vertical counterfactual")
                ),
                replay=_read(_file(replay, f"cohort {index} collision replay")),
            )
        )
    rows.sort(key=lambda row: str(row["scene_id"]))
    if len({row["scene_id"] for row in rows}) != len(rows):
        raise ValueError("P03 cohort must contain distinct scenes")
    splits = {locked[row["scene_id"]].get("split") for row in rows}
    if splits != {"train", "validation"}:
        raise ValueError("P03 bootstrap cohort requires one train and one validation scene")

    payload = {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": "isaac-hm3d-stratified-cohort-flight-space-vfov90-20260803",
        "runtime_command_sha256": _command_sha256(),
        "navmesh_authorizes_flight": False,
        "admission_scope": "stratified_development_cohort",
        "scenes": rows,
    }
    artifact = {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "phase_id": "P03",
        "kind": "flight_space_3d",
        "origin": "real_runtime",
        "measured": True,
        "synthetic": False,
        "denominator_complete": True,
        "payload": payload,
    }
    write_json_atomic(output, artifact)
    print(json.dumps({"status": "P03_GEOMETRY_READY", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
