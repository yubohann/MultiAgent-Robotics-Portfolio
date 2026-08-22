"""Assemble three independent real CF2X reset probes into A-B-A evidence.

This tool consumes the JSON written by ``probe_hm3d_cf2x_reset.py``.  It does
not execute a simulator itself and refuses fabricated, failed, or mismatched
probes.  Its output is development evidence until P01/P05 lock the complete
official HM3D split manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_probe(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "measured",
        "synthetic",
        "passed",
        "scene_id",
        "collision_usd_sha256",
        "coordinate_transform_sha256",
        "cf2x_usd_sha256",
        "runtime",
        "controller",
        "robot",
        "fingerprint_components",
        "reset_fingerprint",
        "reset_state",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"reset probe is incomplete: {resolved}")
    if (
        payload["schema_version"] != "hm3d-cf2x-real-reset-probe-v1"
        or payload["status"] != "RESET_PROBE_PASSED"
        or payload["measured"] is not True
        or payload["synthetic"] is not False
        or payload["passed"] is not True
    ):
        raise ValueError(f"reset probe is not a passing real-runtime witness: {resolved}")
    if payload["reset_state"].get("position_within_tolerance") is not True:
        raise ValueError(f"reset probe position check failed: {resolved}")
    if payload["reset_state"].get("speed_within_tolerance") is not True:
        raise ValueError(f"reset probe velocity check failed: {resolved}")
    components = payload["fingerprint_components"]
    expected_components = {
        "scene",
        "collider",
        "contact",
        "sensor",
        "rng",
        "controller",
        "reset_state",
    }
    if set(components) != expected_components:
        raise ValueError(f"reset fingerprint components are incomplete: {resolved}")
    if payload["reset_fingerprint"] != _canonical_sha256(components):
        raise ValueError(f"reset fingerprint does not match its components: {resolved}")
    if payload["runtime"].get("meters_per_unit") != 1.0:
        raise ValueError(f"reset probe does not use metre units: {resolved}")
    if payload["runtime"].get("stage_up_axis") != "Z":
        raise ValueError(f"reset probe does not use Z-up Isaac coordinates: {resolved}")
    return payload, {"path": str(resolved), "sha256": _sha256(resolved)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite A-B-A evidence: {output}")
    a1, a1_ref = _read_probe(args.a1)
    b, b_ref = _read_probe(args.b)
    a2, a2_ref = _read_probe(args.a2)
    if a1["scene_id"] != a2["scene_id"]:
        raise ValueError("A1 and A2 must use the same scene")
    if a1["scene_id"] == b["scene_id"]:
        raise ValueError("A and B must use distinct official scenes")
    if a1["reset_fingerprint"] != a2["reset_fingerprint"]:
        raise ValueError("A1 and A2 reset fingerprints differ")
    if b["reset_fingerprint"] == a1["reset_fingerprint"]:
        raise ValueError("B reset fingerprint is not independent from A")
    for field in ("cf2x_usd_sha256", "coordinate_transform_sha256"):
        if len({row[field] for row in (a1, b, a2)}) != 1:
            raise ValueError(f"A-B-A changed {field}")
    for field in ("runtime", "controller", "robot"):
        if len({_canonical_sha256(row[field]) for row in (a1, b, a2)}) != 1:
            raise ValueError(f"A-B-A changed {field}")

    runtime = a1["runtime"]
    controller_hash = _canonical_sha256(a1["controller"])
    dynamics_hash = _canonical_sha256(
        {
            "cf2x_usd_sha256": a1["cf2x_usd_sha256"],
            "runtime": runtime,
            "robot": a1["robot"],
        }
    )
    p02_candidate = {
        "evidence_class": "real_runtime",
        "runtime_run_id": "hm3d-cf2x-development-aba-reset-20260801",
        "runtime_command_sha256": _sha256(Path(__file__).resolve()),
        "length_unit_m": 1.0,
        "source_up_axis": "Y",
        "runtime_up_axis": "Z",
        "coordinate_transform_sha256": a1["coordinate_transform_sha256"],
        "gravity_m_s2": runtime["gravity_m_s2"],
        "vehicle_envelope_m": a1["robot"]["vehicle_envelope_m"],
        "simulator_id": "isaac_sim_5_1_isaaclab",
        "simulator_version": "5_1",
        "controller_sha256": controller_hash,
        "dynamics_sha256": dynamics_hash,
        "vehicle_collider_sha256": a1["cf2x_usd_sha256"],
        "aba_reset": {
            "scene_a_id": a1["scene_id"],
            "scene_b_id": b["scene_id"],
            "a1_fingerprint": a1["reset_fingerprint"],
            "b_fingerprint": b["reset_fingerprint"],
            "a2_fingerprint": a2["reset_fingerprint"],
            "components": sorted(a1["fingerprint_components"]),
            "passed": True,
        },
    }
    payload = {
        "schema_version": "hm3d-cf2x-aba-reset-development-evidence-v1",
        "status": "DEVELOPMENT_ABA_RESET_PASSED_NOT_FORMAL_P02",
        "measured": True,
        "synthetic": False,
        "raw_probes": {"a1": a1_ref, "b": b_ref, "a2": a2_ref},
        "p02_candidate": p02_candidate,
        "formal_runtime_admission": False,
        "caveat": (
            "This verifies actual A-B-A reset over locally available official validation assets. "
            "It cannot close formal P02 until P01/P05 lock the complete "
            "train/validation/test scene set."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "a1_equals_a2": True,
                "b_differs_a1": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
