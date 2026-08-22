"""Admit independently recorded HM3D A-B-A reset probes after P01 asset lock.

The script does not run Isaac Sim.  It verifies the original real-runtime
probe files, binds their scene/asset hashes to a P01 asset lock, and writes a
P02 preflight artifact.  This keeps an existing measured A-B-A run auditable
without relabelling it as a fresh simulator execution.
"""

# ruff: noqa: E402

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

from aerocity_method.evaluation.hm3d_preflight import (
    PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    load_preflight_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _locked_scenes(p01_artifact: Path) -> dict[str, dict[str, Any]]:
    envelope = _read_object(p01_artifact)
    payload = envelope.get("payload")
    if (
        envelope.get("phase_id") != "P01"
        or envelope.get("kind") != "asset_lock"
        or envelope.get("origin") != "source_license_audit"
        or not isinstance(payload, dict)
        or not isinstance(payload.get("scenes"), list)
    ):
        raise ValueError("P01 artifact has an unexpected shape")
    rows: dict[str, dict[str, Any]] = {}
    for scene in payload["scenes"]:
        if not isinstance(scene, dict) or not isinstance(scene.get("scene_id"), str):
            raise ValueError("P01 scene rows are malformed")
        rows[scene["scene_id"]] = scene
    return rows


def _verified_p02_payload(
    *, development_evidence_path: Path, p01_artifact_path: Path
) -> dict[str, Any]:
    development = _read_object(development_evidence_path)
    if (
        development.get("status") != "DEVELOPMENT_ABA_RESET_PASSED_NOT_FORMAL_P02"
        or development.get("measured") is not True
        or development.get("synthetic") is not False
    ):
        raise ValueError("development A-B-A evidence is not a passing real-runtime witness")
    candidate = development.get("p02_candidate")
    raw_probes = development.get("raw_probes")
    if not isinstance(candidate, dict) or not isinstance(raw_probes, dict):
        raise ValueError("development A-B-A evidence lacks candidate or raw probe references")
    locked = _locked_scenes(p01_artifact_path)
    expected_raw = {"a1", "b", "a2"}
    if set(raw_probes) != expected_raw:
        raise ValueError("A-B-A evidence must contain a1, b, and a2 probes")

    probes: dict[str, dict[str, Any]] = {}
    for key, reference in raw_probes.items():
        if not isinstance(reference, dict):
            raise ValueError(f"raw {key} probe reference is malformed")
        path = Path(reference.get("path", "")).resolve()
        expected_hash = reference.get("sha256")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"raw {key} probe is missing or changed")
        probe = _read_object(path)
        if (
            probe.get("schema_version") != "hm3d-cf2x-real-reset-probe-v1"
            or probe.get("status") != "RESET_PROBE_PASSED"
            or probe.get("measured") is not True
            or probe.get("synthetic") is not False
            or probe.get("passed") is not True
        ):
            raise ValueError(f"raw {key} probe is not a passing real-runtime probe")
        scene_id = probe.get("scene_id")
        if scene_id not in locked:
            raise ValueError(f"raw {key} probe scene is absent from P01: {scene_id}")
        if probe.get("scene_source_glb_sha256") != locked[scene_id].get("sha256"):
            raise ValueError(f"raw {key} probe geometry hash does not match P01")
        probes[key] = probe

    reset = candidate.get("aba_reset")
    if not isinstance(reset, dict):
        raise ValueError("A-B-A candidate lacks reset section")
    if (
        reset.get("scene_a_id") != probes["a1"].get("scene_id")
        or reset.get("scene_b_id") != probes["b"].get("scene_id")
        or probes["a2"].get("scene_id") != probes["a1"].get("scene_id")
        or reset.get("a1_fingerprint") != probes["a1"].get("reset_fingerprint")
        or reset.get("b_fingerprint") != probes["b"].get("reset_fingerprint")
        or reset.get("a2_fingerprint") != probes["a2"].get("reset_fingerprint")
    ):
        raise ValueError("A-B-A candidate does not bind to its raw probes")
    if any(
        probes[key].get("cf2x_usd_sha256") != candidate.get("vehicle_collider_sha256")
        for key in expected_raw
    ):
        raise ValueError("raw probe vehicle collider changed across A-B-A")
    return {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": "hm3d-cf2x-aba-reset-admitted-20260801",
        "runtime_command_sha256": _sha256(ROOT / "scripts" / "probe_hm3d_cf2x_reset.py"),
        "length_unit_m": candidate["length_unit_m"],
        "source_up_axis": candidate["source_up_axis"],
        "runtime_up_axis": candidate["runtime_up_axis"],
        "coordinate_transform_sha256": candidate["coordinate_transform_sha256"],
        "gravity_m_s2": candidate["gravity_m_s2"],
        "vehicle_envelope_m": candidate["vehicle_envelope_m"],
        "simulator_id": candidate["simulator_id"],
        "simulator_version": candidate["simulator_version"],
        "controller_sha256": candidate["controller_sha256"],
        "dynamics_sha256": candidate["dynamics_sha256"],
        "vehicle_collider_sha256": candidate["vehicle_collider_sha256"],
        "aba_reset": reset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--p01-artifact", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--p01-reference", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    development = args.development_evidence.expanduser().resolve()
    p01_artifact = args.p01_artifact.expanduser().resolve()
    p02_artifact = args.output_artifact.expanduser().resolve()
    p01_reference = args.p01_reference.expanduser().resolve()
    output_evidence = args.output_evidence.expanduser().resolve()
    if not p01_reference.is_file():
        raise FileNotFoundError(f"P01 reference is missing: {p01_reference}")
    p02_payload = _verified_p02_payload(
        development_evidence_path=development, p01_artifact_path=p01_artifact
    )
    p02_envelope = {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "phase_id": "P02",
        "kind": "runtime_aba_reset",
        "origin": "real_runtime",
        "measured": True,
        "synthetic": False,
        "denominator_complete": True,
        "payload": p02_payload,
    }
    _write_json_new(p02_artifact, p02_envelope)
    p01_reference_payload = _read_object(p01_reference)
    artifacts = p01_reference_payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("P01 partial evidence must contain exactly one artifact")
    artifacts = [*artifacts]
    artifacts.append(
        {
            "phase_id": "P02",
            "kind": "runtime_aba_reset",
            "origin": "real_runtime",
            "path": str(p02_artifact),
            "sha256": _sha256(p02_artifact),
        }
    )
    protocol = load_preflight_protocol(
        ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json"
    )
    evidence = {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "protocol_hash": protocol.protocol_hash,
        "requested_gate": "formal_experiment_start",
        "method_core": p01_reference_payload.get("method_core"),
        "aerocity_bench_accesses": [],
        "artifacts": artifacts,
    }
    _write_json_new(output_evidence, evidence)
    print(
        json.dumps(
            {
                "status": "P02_ABA_RESET_ADMITTED",
                "p02_artifact": str(p02_artifact),
                "evidence": str(output_evidence),
                "scenes": [
                    p02_payload["aba_reset"]["scene_a_id"],
                    p02_payload["aba_reset"]["scene_b_id"],
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
