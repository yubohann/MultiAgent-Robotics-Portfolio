"""Select evaluator-only sparse-range receiver positions from admitted P03 geometry.

This preparation step deliberately has no camera dependency.  It rebuilds the
frozen collision-derived ESDF only to select spread-out calibration positions;
the positions and the ESDF remain evaluator-side.  The eventual method receives
only the real range outcomes measured by ``run_hm3d_public_observation_admission``.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_hm3d_collision_flight_space import (
    _canonical_sha256 as _flight_space_sha256,
)
from audit_hm3d_collision_flight_space import (
    _load_and_validate_manifest,
    _load_triangle_mesh,
)

from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.runtime.hm3d_calibration_geometry import farthest_spread_indices

SCHEMA_VERSION = "hm3d-p04-range-receiver-position-calibration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _largest_component_points(arrays: dict[str, Any]) -> np.ndarray:
    free = np.asarray(arrays["free_mask"], dtype=bool)
    labels = np.asarray(arrays["component_labels"], dtype=np.int32)
    counts = np.bincount(labels[free])
    counts[0] = 0
    component = int(np.argmax(counts))
    if component == 0:
        raise ValueError("P03 ESDF has no retained free-flight component")
    resolution_m = float(np.asarray(arrays["resolution_m"]).item())
    origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
    indices = np.argwhere(free & (labels == component))
    return origin + indices.astype(np.float64) * resolution_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--collision-derivative-manifest", type=Path, required=True)
    parser.add_argument("--flight-space-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewpoint-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.viewpoint_count < 1 or args.seed < 0:
        raise ValueError("viewpoint count and seed must be non-negative")
    source = args.source_glb.expanduser().resolve()
    collision = args.collision_usd.expanduser().resolve()
    manifest_path = args.collision_manifest.expanduser().resolve()
    derivative_path = args.collision_derivative_manifest.expanduser().resolve()
    flight_path = args.flight_space_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (source, collision, manifest_path, derivative_path, flight_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output}")

    _, derivative = _load_and_validate_manifest(
        manifest_path,
        args.scene_id,
        source,
        collision,
        collision_derivative_manifest=derivative_path,
    )
    if derivative is None:
        raise ValueError("P04 calibration requires the frozen two-sided collision derivative")
    flight = _read_object(flight_path)
    if flight.get("scene_id") != args.scene_id:
        raise ValueError("flight-space audit scene mismatch")
    if flight.get("source_glb_sha256") != _sha256(source):
        raise ValueError("flight-space source GLB differs from calibration input")
    if flight.get("collision_usd_sha256") != _sha256(collision):
        raise ValueError("flight-space collision USD differs from calibration input")
    if not isinstance(flight.get("flight_space"), dict):
        raise ValueError("flight-space audit lacks ESDF payload")
    # P03 persists this hash with the flight-space audit's canonicalizer.  Use
    # that exact implementation rather than treating a serialization change as
    # a geometry change.
    expected_flight_hash = _flight_space_sha256(flight["flight_space"])
    if flight.get("flight_space_manifest_hash") != expected_flight_hash:
        raise ValueError("flight-space audit hash is invalid")

    mesh = _load_triangle_mesh(collision)
    arrays, rebuilt_flight = build_enclosed_esdf(
        mesh,
        resolution_m=float(flight["resolution_m"]),
        vehicle_clearance_m=float(flight["vehicle_clearance_m"]),
    )
    if _flight_space_sha256(rebuilt_flight) != expected_flight_hash:
        raise ValueError("rebuilt ESDF differs from frozen P03 flight-space audit")
    candidates = _largest_component_points(arrays)
    if len(candidates) < args.viewpoint_count:
        raise ValueError("largest free-flight component has too few receiver positions")
    selected = candidates[
        farthest_spread_indices(candidates, count=args.viewpoint_count, seed=args.seed)
    ]
    views = [
        {
            "receiver_id": f"p04-calibration-receiver-{index:02d}",
            "receiver_position_w_m": [float(value) for value in point],
        }
        for index, point in enumerate(selected)
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "P04_RANGE_RECEIVER_POSITIONS_READY",
        "synthetic": False,
        "formal_result": False,
        "evidence_class": "evaluator_geometry_calibration",
        "scene_id": args.scene_id,
        "source_glb_sha256": _sha256(source),
        "collision_usd_sha256": _sha256(collision),
        "flight_space_manifest_hash": expected_flight_hash,
        "collision_derivative_manifest_sha256": derivative["manifest_sha256"],
        "selection_rule": "largest-component-deterministic-farthest-spread-v1",
        "selection_seed": args.seed,
        "receiver_position_count": len(views),
        "method_visible": False,
        "claim_limit": (
            "Evaluator-side P04 receiver calibration only. These positions are not "
            "method-visible waypoints, visual-camera observations, or exploration results."
        ),
        "views": views,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    _write_new(output, payload)
    print(
        json.dumps(
            {
                "scene_id": args.scene_id,
                "receiver_position_count": len(views),
                "output": str(output),
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
