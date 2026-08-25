"""Create a local two-sided PhysX collision derivative for one HM3D scan.

HM3D meshes are surface scans rather than watertight solids.  A sparse-range
ray can meet a nearby back-facing scan triangle that a one-sided PhysX query
passes through.  This tool makes a local, conservative collision derivative by
duplicating every triangle with reverse winding.  It never modifies the source
GLB or the original collision USD, and records exact provenance for the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "hm3d-two-sided-collision-derivative-v1"


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


def _two_sided_triangle_indices(
    face_vertex_counts: np.ndarray, face_vertex_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return original triangles followed by their reverse-winding copies."""

    counts = np.asarray(face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1)
    if not len(counts) or not np.all(counts == 3):
        raise ValueError("two-sided collision requires a non-empty triangle mesh")
    if len(indices) != int(counts.sum()):
        raise ValueError("face-vertex index count does not match face counts")
    triangles = indices.reshape((-1, 3))
    reversed_triangles = triangles[:, [0, 2, 1]]
    return (
        np.concatenate((counts, counts), axis=0),
        np.concatenate((triangles.reshape(-1), reversed_triangles.reshape(-1)), axis=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--source-collision-usd", type=Path, required=True)
    parser.add_argument("--source-collision-manifest", type=Path, required=True)
    parser.add_argument("--output-usd", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def _validate_source_manifest(
    path: Path, *, scene_id: str, source_glb: Path, source_collision_usd: Path
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source collision manifest must be an object")
    if payload.get("scene_id") != scene_id:
        raise ValueError("source collision manifest scene mismatch")
    if Path(payload.get("source_glb", "")).resolve() != source_glb:
        raise ValueError("source collision manifest GLB path mismatch")
    if Path(payload.get("output_usd", "")).resolve() != source_collision_usd:
        raise ValueError("source collision manifest USD path mismatch")
    if payload.get("source_glb_sha256") != _sha256(source_glb):
        raise ValueError("source GLB hash changed after collision conversion")
    if payload.get("output_usd_sha256") != _sha256(source_collision_usd):
        raise ValueError("source collision USD hash changed after conversion")
    return payload


def main() -> int:
    # This post-processing step deliberately uses only the public USD Python
    # bindings.  PhysX schema registration itself requires a live Isaac app,
    # so the runtime audit remains responsible for proving the derived mesh is
    # actually accepted by PhysX.
    from pxr import Usd, UsdGeom, UsdPhysics

    args = parse_args()
    source_glb = args.source_glb.expanduser().resolve()
    source_usd = args.source_collision_usd.expanduser().resolve()
    source_manifest = args.source_collision_manifest.expanduser().resolve()
    output_usd = args.output_usd.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    for path in (source_glb, source_usd, source_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_usd.exists() or output_manifest.exists():
        raise FileExistsError("two-sided derivative outputs must be new evidence files")
    _validate_source_manifest(
        source_manifest,
        scene_id=args.scene_id,
        source_glb=source_glb,
        source_collision_usd=source_usd,
    )

    source_stage = Usd.Stage.Open(str(source_usd))
    if source_stage is None:
        raise RuntimeError(f"could not open source collision USD: {source_usd}")
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if not source_stage.GetRootLayer().Export(str(output_usd)):
        raise RuntimeError(f"could not export two-sided derivative: {output_usd}")

    output_stage = Usd.Stage.Open(str(output_usd))
    if output_stage is None:
        raise RuntimeError(f"could not reopen derivative USD: {output_usd}")
    mesh_rows: list[dict[str, Any]] = []
    for prim in output_stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        applied_schemas = {str(schema) for schema in prim.GetAppliedSchemas()}
        api_schemas = prim.GetMetadata("apiSchemas")
        if api_schemas is not None:
            applied_schemas.update(str(schema) for schema in api_schemas.GetAddedOrExplicitItems())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not any(
            "TriangleMeshCollision" in schema for schema in applied_schemas
        ):
            raise RuntimeError(f"source collision APIs missing on {prim.GetPath()}")
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        doubled_counts, doubled_indices = _two_sided_triangle_indices(counts, indices)
        mesh.GetFaceVertexCountsAttr().Set(doubled_counts.tolist())
        mesh.GetFaceVertexIndicesAttr().Set(doubled_indices.tolist())
        mesh_rows.append(
            {
                "prim_path": str(prim.GetPath()),
                "face_count_before": int(len(counts)),
                "face_count_after": int(len(doubled_counts)),
                "point_count": int(len(mesh.GetPointsAttr().Get())),
                "collision_enabled": bool(
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                ),
                "applied_schemas": sorted(applied_schemas),
            }
        )
    if not mesh_rows:
        raise RuntimeError("source collision USD contains no mesh")
    output_stage.GetRootLayer().Save()
    if not output_usd.is_file():
        raise RuntimeError("two-sided derivative did not persist")

    transform = {
        "operation": "duplicate_each_triangle_with_reverse_winding",
        "source_collision_usd_sha256": _sha256(source_usd),
        "source_glb_sha256": _sha256(source_glb),
        "changes_vertex_positions": False,
        "changes_world_transform": False,
        "collision_semantics": "two_sided_conservative_static_triangle_mesh",
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "TWO_SIDED_COLLISION_DERIVATIVE_COMPLETE_NOT_RUNTIME_ADMITTED",
        "synthetic": False,
        "scene_id": args.scene_id,
        "source_glb": str(source_glb),
        "source_glb_sha256": _sha256(source_glb),
        "source_collision_usd": str(source_usd),
        "source_collision_usd_sha256": _sha256(source_usd),
        "source_collision_manifest": str(source_manifest),
        "source_collision_manifest_sha256": _sha256(source_manifest),
        "output_usd": str(output_usd),
        "output_usd_sha256": _sha256(output_usd),
        "derivative": transform,
        "derivative_sha256": _canonical_sha256(transform),
        "meshes": mesh_rows,
        "formal_runtime_admission": False,
        "required_followups": [
            "real sparse-range ray / PhysX distance consistency audit",
            "CF2X collision replay",
            "fixed-altitude counterfactual",
        ],
    }
    report["audit_sha256"] = _canonical_sha256(report)
    _write_json(output_manifest, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
