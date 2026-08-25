"""Measure evaluator-side conservative ESDF clearance at frozen public HM3D views."""

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

from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf
from aerocity_method.evaluation.hm3d_safety import ConservativeVoxelClearance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mesh(usd_path: Path) -> Any:
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open collision USD: {usd_path}")
    cache = UsdGeom.XformCache()
    vertices: list[Any] = []
    faces: list[Any] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"mesh lacks points: {prim.GetPath()}")
        if not len(counts) or not np.all(counts == 3) or len(indices) != int(counts.sum()):
            raise ValueError(f"mesh is not a valid triangulation: {prim.GetPath()}")
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = (np.column_stack((points, np.ones(len(points)))) @ transform)[:, :3]
        vertices.append(world)
        faces.append(indices.reshape((-1, 3)) + offset)
        offset += len(points)
    if not vertices:
        raise ValueError("collision USD contains no triangles")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--receiver-positions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-m", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    import numpy as np
    import trimesh

    args = parse_args()
    collision = args.collision_usd.expanduser().resolve()
    poses = args.receiver_positions_json.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not collision.is_file() or not poses.is_file():
        raise FileNotFoundError("collision USD and receiver-position evidence must exist")
    source = json.loads(poses.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("views"), list):
        raise ValueError("receiver-position evidence lacks a public views list")
    mesh = _load_mesh(collision)
    arrays, report = build_enclosed_esdf(
        mesh,
        resolution_m=args.resolution_m,
        vehicle_clearance_m=args.resolution_m / 2.0,
    )
    field = ConservativeVoxelClearance(
        arrays["collision_distance_m"],
        tuple(float(value) for value in arrays["origin_center_m"]),
        float(arrays["resolution_m"]),
    )
    public_positions = np.asarray(
        [view.get("receiver_position_w_m") for view in source["views"]], dtype=np.float64
    )
    if public_positions.ndim != 2 or public_positions.shape[1:] != (3,):
        raise ValueError("receiver-position evidence contains an invalid public position")
    _, exact_distances, _ = trimesh.proximity.closest_point(mesh, public_positions)
    rows = []
    for index, view in enumerate(source["views"]):
        if not isinstance(view, dict) or not isinstance(view.get("receiver_position_w_m"), list):
            raise ValueError(f"view {index} lacks a public position")
        position = tuple(float(value) for value in view["receiver_position_w_m"])
        assessment = field.assess(position)
        rows.append(
            {
                "public_view_index": index,
                "position_m": position,
                "in_field_bounds": assessment.in_field_bounds,
                "sampled_distance_m": assessment.sampled_distance_m,
                "conservative_distance_m": assessment.conservative_distance_m,
                "exact_triangle_distance_m": float(exact_distances[index]),
                "admitted_at_030_m": assessment.admits(0.30),
                "admitted_at_040_m": assessment.admits(0.40),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "hm3d-p07-clearance-probe-v1",
        "status": "ENGINEERING_DIAGNOSTIC_NOT_P07_RESULT",
        "claim_limit": (
            "Evaluator-side static-clearance diagnostic only; no targets or result scores."
        ),
        "scene_id": source.get("scene_id"),
        "collision_usd_sha256": _sha256(collision),
        "receiver_position_source_sha256": _sha256(poses),
        "esdf_resolution_m": field.resolution_m,
        "esdf_discretization_margin_m": field.discretization_margin_m,
        "esdf_generation_method": report["generation_method"],
        "public_view_clearance": rows,
    }
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
