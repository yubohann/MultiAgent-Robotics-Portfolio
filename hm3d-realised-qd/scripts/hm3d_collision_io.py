"""Shared HM3D collision-asset loading and explicit identity labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "hm3d-collision-io-v1"


def file_id(path: Path) -> str:
    """Asset identity from file name and size."""
    return f"{path.name}:{path.stat().st_size}"


def flight_space_id(flight: dict[str, Any]) -> str:
    """Readable flight-space label retained next to the raw grid arrays."""
    return (
        f"flight-space:{flight.get('resolution_m')}m:"
        f"{flight.get('free_voxels')}voxels:{flight.get('retained_component_count')}components"
    )


def load_triangle_mesh(usd_path: Path) -> Any:
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


def load_and_validate_manifest(
    path: Path,
    scene_id: str,
    source: Path,
    collision: Path,
    collision_derivative_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load a conversion manifest and check its declared scene and asset paths."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("collision manifest must be an object")
    if payload.get("scene_id") != scene_id:
        raise ValueError("collision manifest scene_id mismatch")
    if Path(payload.get("source_glb", "")).resolve() != source:
        raise ValueError("collision manifest source GLB mismatch")
    derivative_provenance = None
    if Path(payload.get("output_usd", "")).resolve() != collision:
        if collision_derivative_manifest is None:
            raise ValueError("collision manifest collision USD mismatch")
        derivative = json.loads(collision_derivative_manifest.read_text(encoding="utf-8"))
        if not isinstance(derivative, dict):
            raise ValueError("collision derivative manifest must be an object")
        if Path(derivative.get("output_usd", "")).resolve() != collision:
            raise ValueError("collision derivative output USD path mismatch")
        operation = derivative.get("derivative")
        if not isinstance(operation, dict):
            raise ValueError("collision derivative lacks operation provenance")
        derivative_provenance = {
            "manifest": str(collision_derivative_manifest.resolve()),
            "operation": operation.get("operation"),
        }
    return payload, derivative_provenance


__all__ = [
    "SCHEMA_VERSION",
    "file_id",
    "flight_space_id",
    "load_and_validate_manifest",
    "load_triangle_mesh",
]
