"""Audit a converted HM3D collision USD as a conservative 3-D flight-space candidate.

The script reads the converted static collision mesh, checks its provenance against
the original GLB and derives a bounded voxel ESDF.  It intentionally does not close
the formal HM3D P03 phase: controller counterfactuals, sparse-range ray consistency
and runtime collision replay remain separate evidence requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


GENERATOR_VERSION = "hm3d-collision-usd-esdf-audit-v1"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--collision-derivative-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-m", type=float, default=0.25)
    parser.add_argument("--vehicle-clearance-m", type=float, default=0.30)
    parser.add_argument("--vertical-slice-min-voxels", type=int, default=8)
    return parser.parse_args()


def _load_triangle_mesh(usd_path: Path) -> Any:
    """Load all USD mesh triangles in world coordinates without materials."""

    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open collision USD: {usd_path}")
    xform_cache = UsdGeom.XformCache()
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"mesh has no valid points: {prim.GetPath()}")
        if not len(counts) or not np.all(counts == 3):
            raise ValueError(f"collision mesh must be triangulated: {prim.GetPath()}")
        if len(indices) != int(counts.sum()):
            raise ValueError(f"mesh indices are malformed: {prim.GetPath()}")
        matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = (np.concatenate((points, np.ones((len(points), 1))), axis=1) @ matrix)[:, :3]
        vertices.append(world)
        faces.append(indices.reshape((-1, 3)) + vertex_offset)
        vertex_offset += len(points)
    if not vertices:
        raise ValueError(f"collision USD has no meshes: {usd_path}")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )


def _load_and_validate_manifest(
    path: Path,
    scene_id: str,
    source: Path,
    collision: Path,
    collision_derivative_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("collision manifest must be an object")
    if payload.get("scene_id") != scene_id:
        raise ValueError("collision manifest scene_id mismatch")
    if Path(payload.get("source_glb", "")).resolve() != source:
        raise ValueError("collision manifest source GLB mismatch")
    if payload.get("source_glb_sha256") != _sha256(source):
        raise ValueError("source GLB changed after collision conversion")
    source_collision = Path(payload.get("output_usd", "")).resolve()
    source_collision_sha256 = payload.get("output_usd_sha256")
    derivative_provenance = None
    if source_collision == collision:
        if source_collision_sha256 != _sha256(collision):
            raise ValueError("collision USD changed after conversion manifest")
    else:
        if collision_derivative_manifest is None:
            raise ValueError("collision manifest collision USD mismatch")
        derivative = json.loads(collision_derivative_manifest.read_text(encoding="utf-8"))
        if not isinstance(derivative, dict):
            raise ValueError("collision derivative manifest must be an object")
        if Path(derivative.get("source_collision_usd", "")).resolve() != source_collision:
            raise ValueError("collision derivative source USD path mismatch")
        if derivative.get("source_collision_usd_sha256") != source_collision_sha256:
            raise ValueError("collision derivative source USD hash mismatch")
        if Path(derivative.get("output_usd", "")).resolve() != collision:
            raise ValueError("collision derivative output USD path mismatch")
        if derivative.get("output_usd_sha256") != _sha256(collision):
            raise ValueError("collision derivative output USD hash mismatch")
        operation = derivative.get("derivative")
        if not isinstance(operation, dict):
            raise ValueError("collision derivative lacks operation provenance")
        if operation.get("changes_vertex_positions") is not False:
            raise ValueError("collision derivative changes vertex positions")
        if operation.get("changes_world_transform") is not False:
            raise ValueError("collision derivative changes world transform")
        derivative_provenance = {
            "manifest": str(collision_derivative_manifest.resolve()),
            "manifest_sha256": _sha256(collision_derivative_manifest),
            "operation": operation.get("operation"),
        }
    collision_info = payload.get("collision")
    if not isinstance(collision_info, dict):
        raise ValueError("collision manifest lacks collision inspection")
    if collision_info.get("runtime_up_axis") != "Z" or collision_info.get("meters_per_unit") != 1.0:
        raise ValueError("collision USD is not Z-up at one metre per unit")
    meshes = collision_info.get("meshes")
    if (
        not isinstance(meshes, list)
        or not meshes
        or any(
            row.get("collision_enabled") is not True
            or row.get("physx_triangle_mesh_collision") is not True
            for row in meshes
        )
    ):
        raise ValueError("collision manifest does not prove enabled triangle collision")
    return payload, derivative_provenance


def _vertical_statistics(arrays: dict[str, Any], minimum_voxels: int) -> dict[str, Any]:
    free = np.asarray(arrays["free_mask"], dtype=bool)
    counts = np.bincount(np.argwhere(free)[:, 2], minlength=free.shape[2])
    # Treat a few staircase or reconstruction slivers differently from a
    # substantial floor band.  This matches build_enclosed_esdf's 5% rule
    # while retaining an explicit absolute lower bound for small scenes.
    substantial_threshold = max(
        minimum_voxels,
        int(math.ceil(float(counts.max()) * 0.05)),
    )
    active = np.flatnonzero(counts >= substantial_threshold)
    if len(active) == 0:
        raise ValueError("ESDF has no substantial free-space height slice")
    gaps = np.diff(active)
    bands = 1 + int(np.count_nonzero(gaps > 1))
    volume_by_slice = counts.astype(np.float64)
    dominant = int(volume_by_slice.max())
    vertical_fraction = float(
        volume_by_slice[volume_by_slice < dominant].sum() / max(volume_by_slice.sum(), 1.0)
    )
    return {
        "active_height_slice_count": int(len(active)),
        "active_height_slice_indices": active.tolist(),
        "connected_height_band_count": bands,
        "substantial_height_slice_minimum_voxels": substantial_threshold,
        "height_slice_voxel_counts": counts.tolist(),
        "dominant_height_slice_voxels": dominant,
        "vertical_opportunity_fraction_proxy": vertical_fraction,
    }


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf

    args = parse_args()
    source = args.source_glb.expanduser().resolve()
    collision = args.collision_usd.expanduser().resolve()
    manifest_path = args.collision_manifest.expanduser().resolve()
    for path in (source, collision, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not math.isfinite(args.resolution_m) or not 0.05 <= args.resolution_m <= 0.5:
        raise ValueError("resolution must be in [0.05, 0.5] m")
    if (
        not math.isfinite(args.vehicle_clearance_m)
        or args.vehicle_clearance_m < args.resolution_m / 2.0
    ):
        raise ValueError("vehicle clearance must be at least half a voxel")
    derivative_manifest = None
    if args.collision_derivative_manifest is not None:
        derivative_manifest = args.collision_derivative_manifest.expanduser().resolve()
        if not derivative_manifest.is_file():
            raise FileNotFoundError(derivative_manifest)
    manifest, derivative_provenance = _load_and_validate_manifest(
        manifest_path,
        args.scene_id,
        source,
        collision,
        collision_derivative_manifest=derivative_manifest,
    )
    mesh = _load_triangle_mesh(collision)
    arrays, flight_space = build_enclosed_esdf(
        mesh,
        resolution_m=args.resolution_m,
        vehicle_clearance_m=args.vehicle_clearance_m,
    )
    vertical = _vertical_statistics(arrays, args.vertical_slice_min_voxels)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    report: dict[str, Any] = {
        "schema_version": "hm3d-collision-flight-space-audit-v1",
        "status": "ESDF_AUDIT_COMPLETE_NOT_P03_ADMITTED",
        "synthetic": False,
        "scene_id": args.scene_id,
        "source_glb": str(source),
        "source_glb_sha256": _sha256(source),
        "collision_usd": str(collision),
        "collision_usd_sha256": _sha256(collision),
        "collision_manifest_sha256": _sha256(manifest_path),
        "collision_manifest_coordinate_transform_sha256": manifest["coordinate_transform_sha256"],
        "source_mesh_bounds_min_m": bounds[0].tolist(),
        "source_mesh_bounds_max_m": bounds[1].tolist(),
        "source_mesh_extents_m": (bounds[1] - bounds[0]).tolist(),
        "representation": "voxel_esdf_3d",
        "dimension": 3,
        "generator_version": GENERATOR_VERSION,
        "resolution_m": args.resolution_m,
        "vehicle_clearance_m": args.vehicle_clearance_m,
        "navmesh_authorizes_flight": False,
        "flight_space": flight_space,
        "vertical_statistics": vertical,
        "formal_p03_fields_not_claimed": {
            "fixed_altitude_control_run": False,
            "sparse_range_receiver_count": 0,
            "collision_sparse_range_consistent": False,
            "collision_replay_passed": False,
        },
        "required_followups": [
            "sparse-range ray/collision consistency audit",
            "fixed-altitude counterfactual with identical budget",
            "runtime collision replay binding",
            "public sparse-range receiver admission on the flight space",
        ],
    }
    report["flight_space_manifest_hash"] = _canonical_sha256(report["flight_space"])
    if derivative_provenance is not None:
        report["collision_derivative_provenance"] = derivative_provenance
    report["audit_sha256"] = _canonical_sha256(report)
    _write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
