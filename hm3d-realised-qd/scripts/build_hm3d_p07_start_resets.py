"""Pre-register collision-admitted candidate resets for HM3D P07 fleets.

This script deliberately separates environment initialisation from P04 sensor
calibration.  It produces a geometry-only candidate set from the already
admitted P03 free-flight component.  P07 later chooses a range/LOS-connected
subset in Isaac/PhysX and records the actual reset witness in its outcome.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_hm3d_collision_flight_space import (  # noqa: E402
    _canonical_sha256 as _flight_space_sha256,
)
from audit_hm3d_collision_flight_space import (
    _load_and_validate_manifest,
    _load_triangle_mesh,
)

from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf  # noqa: E402
from aerocity_method.contracts.io import canonical_sha256  # noqa: E402
from aerocity_method.runtime.hm3d_start_resets import (  # noqa: E402
    P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
    P07_START_RESET_SCHEMA_VERSION,
    P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE,
    largest_component_departure_witnesses,
    select_local_spread_positions,
)
from aerocity_method.runtime.hm3d_cf2x_execution import (  # noqa: E402
    FLIGHT_CLEARANCE_M,
    REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
    REQUIRED_TERMINAL_CLEARANCE_M,
    ROUTE_CLEARANCE_SAMPLE_STEP_M,
)


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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite start-reset evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _offline_departure_witness(
    mesh: Any,
    start_m: tuple[float, float, float],
    end_m: tuple[float, float, float],
    *,
    grid_tube_clearance_m: float,
) -> dict[str, Any]:
    """Certify one pre-registered first hop against the collision mesh.

    This is an environment-side reset filter, not a replacement for the
    runtime PhysX guard.  It checks exactly the start, endpoint and internal
    clearance samples used by a nonzero `_line_guard` command; the subsequent
    P0 first-pool audit replays that witness with the actual scene query.
    """

    import numpy as np
    import trimesh

    distance_m = math.dist(start_m, end_m)
    if distance_m <= 1.0e-9:
        raise ValueError("departure witness cannot be a stationary hold")
    sample_count = max(1, math.ceil(distance_m / ROUTE_CLEARANCE_SAMPLE_STEP_M))
    samples = tuple(
        tuple(
            start_m[axis] + sample_index / sample_count * (end_m[axis] - start_m[axis])
            for axis in range(3)
        )
        for sample_index in range(sample_count + 1)
    )
    _, raw_distances, _ = trimesh.proximity.closest_point(
        mesh, np.asarray(samples, dtype=np.float64)
    )
    distances = tuple(float(value) for value in raw_distances)
    if len(distances) != len(samples) or not all(math.isfinite(value) for value in distances):
        raise RuntimeError("offline departure witness produced invalid mesh clearances")
    interior = distances[1:-1]
    admitted = (
        distances[0] + 1.0e-12 >= FLIGHT_CLEARANCE_M
        and distances[-1] + 1.0e-12 >= REQUIRED_TERMINAL_CLEARANCE_M
        and all(value + 1.0e-12 >= REQUIRED_ROUTE_SAMPLE_CLEARANCE_M for value in interior)
    )
    if not admitted:
        raise ValueError(
            "route-sample-selected reset has no exact-mesh-admitted first departure"
        )
    witness_id = canonical_sha256(
        {
            "start_m": start_m,
            "end_m": end_m,
            "route_sample_count": sample_count,
            "route_sample_spacing_m": ROUTE_CLEARANCE_SAMPLE_STEP_M,
            "required_start_clearance_m": FLIGHT_CLEARANCE_M,
            "required_terminal_clearance_m": REQUIRED_TERMINAL_CLEARANCE_M,
            "required_internal_sample_clearance_m": REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
        }
    )
    return {
        "schema_version": P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
        "witness_id": witness_id,
        "path_m": [list(start_m), list(end_m)],
        "length_m": distance_m,
        "route_sample_count": sample_count,
        "minimum_exact_static_mesh_clearance_m": min(distances),
        "exact_start_clearance_m": distances[0],
        "exact_terminal_clearance_m": distances[-1],
        "exact_internal_sample_minimum_clearance_m": (
            None if not interior else min(interior)
        ),
        "offline_exact_admitted": True,
        "offline_grid_tube_clearance_m": grid_tube_clearance_m,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--collision-derivative-manifest", type=Path, required=True)
    parser.add_argument("--flight-space-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--cluster-radius-m", type=float, default=3.0)
    parser.add_argument("--minimum-separation-m", type=float, default=0.75)
    parser.add_argument(
        "--start-mobility-clearance-m",
        type=float,
        default=REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
        help=(
            "minimum P03 ESDF clearance for a reset candidate; defaults to the shared "
            "runtime interior route-sample clearance contract"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_count < 2 or args.seed < 0:
        raise ValueError("candidate count must be at least two and seed must be non-negative")
    if args.start_mobility_clearance_m <= 0.0:
        raise ValueError("start mobility clearance must be positive")
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
        raise FileExistsError(f"refusing to overwrite start-reset evidence: {output}")

    _, derivative = _load_and_validate_manifest(
        manifest_path,
        args.scene_id,
        source,
        collision,
        collision_derivative_manifest=derivative_path,
    )
    if derivative is None:
        raise ValueError(
            "P07 start reset candidates require the frozen two-sided collision derivative"
        )
    flight = _read_object(flight_path)
    if flight.get("scene_id") != args.scene_id:
        raise ValueError("flight-space audit scene mismatch")
    if flight.get("source_glb_sha256") != _sha256(source):
        raise ValueError("flight-space source GLB differs from start-reset input")
    if flight.get("collision_usd_sha256") != _sha256(collision):
        raise ValueError("flight-space collision USD differs from start-reset input")
    if not isinstance(flight.get("flight_space"), dict):
        raise ValueError("flight-space audit lacks ESDF payload")
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
    departure_points, departure_endpoints, grid_tube_clearance_m = (
        largest_component_departure_witnesses(
            arrays,
            minimum_route_sample_clearance_m=args.start_mobility_clearance_m,
        )
    )
    selected = select_local_spread_positions(
        departure_points,
        count=args.candidate_count,
        seed=args.seed,
        cluster_radius_m=args.cluster_radius_m,
        minimum_separation_m=args.minimum_separation_m,
    )
    endpoint_by_start = {
        tuple(float(value) for value in start): tuple(float(value) for value in endpoint)
        for start, endpoint in zip(departure_points, departure_endpoints, strict=True)
    }
    candidates = []
    for index, point in enumerate(selected):
        start_m = tuple(float(value) for value in point)
        end_m = endpoint_by_start.get(start_m)
        if end_m is None:
            raise RuntimeError("selected departure point lacks its deterministic witness")
        candidates.append(
            {
                "candidate_id": f"p07-start-candidate-{index:02d}",
                "position_w_m": list(start_m),
                "static_departure_witnesses": [
                    _offline_departure_witness(
                        mesh,
                        start_m,
                        end_m,
                        grid_tube_clearance_m=grid_tube_clearance_m,
                    )
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": P07_START_RESET_SCHEMA_VERSION,
        "status": "P07_START_RESET_CANDIDATES_READY",
        "synthetic": False,
        "formal_result": False,
        "evidence_class": "environment_reset_pre_registration",
        "scene_id": args.scene_id,
        "source_glb_sha256": _sha256(source),
        "collision_usd_sha256": _sha256(collision),
        "flight_space_manifest_hash": expected_flight_hash,
        "collision_derivative_manifest_sha256": derivative["manifest_sha256"],
        "selection_rule": P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE,
        "selection_seed": args.seed,
        "candidate_count": len(candidates),
        "cluster_radius_m": args.cluster_radius_m,
        "minimum_pairwise_separation_m": args.minimum_separation_m,
        "start_mobility_clearance_m": args.start_mobility_clearance_m,
        "departure_witness_contract": {
            "schema_version": P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
            "selection_rule": "six-neighbour-grid-tube+exact-static-samples-v1",
            "candidate_witness_count": len(candidates),
            "route_sample_spacing_m": ROUTE_CLEARANCE_SAMPLE_STEP_M,
            "required_start_clearance_m": FLIGHT_CLEARANCE_M,
            "required_terminal_clearance_m": REQUIRED_TERMINAL_CLEARANCE_M,
            "required_internal_sample_clearance_m": REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
            "offline_grid_tube_clearance_m": grid_tube_clearance_m,
            "offline_static_mesh": "same_immutable_collision_usd_as_physx",
            "runtime_authority": "P0 replays a selected witness through the active PhysX route guard",
        },
        "method_visible": False,
        "claim_limit": (
            "Environment reset candidates only. P04 range receiver calibration is separate; "
            "the P07 worker must still verify the chosen fleet and each selected first-hop "
            "witness by actual PhysX range/LOS and route guards before exploration begins."
        ),
        "candidates": candidates,
    }
    payload["start_reset_sha256"] = canonical_sha256(payload)
    _write_new(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scene_id": args.scene_id,
                "candidate_count": len(candidates),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
