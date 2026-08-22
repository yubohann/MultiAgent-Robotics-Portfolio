"""Convert one official HM3D GLB to a static triangle-mesh collision USD.

This is intentionally separate from the archived visual-preview converter. The
latter creates review-only visual USD files for videos; this tool creates a
separate candidate for physical-runtime admission and verifies that every
converted mesh has collision enabled and a PhysX triangle-mesh collider.

Successful conversion is necessary but not sufficient for P02/P03.  A real
CF2X reset and collision replay must still be recorded before the asset can
authorize flight-space or experiment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--usd-file-name", default="hm3d_collision_scene.usd")
    parser.add_argument(
        "--source-up-axis",
        choices=("Y", "Z"),
        default="Y",
        help="Documented source convention; runtime stage convention is measured after conversion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS)
SIMULATION_APP = APP.app


def _inspect_static_triangle_mesh_collision(usd_path: Path) -> dict[str, Any]:
    """Fail if a converted geometry mesh lacks static triangle collision."""

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not reopen generated USD: {usd_path}")
    mesh_rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not indices:
            raise RuntimeError(f"empty converted mesh: {prim.GetPath()}")
        collision = UsdPhysics.CollisionAPI(prim)
        collision_enabled = collision.GetCollisionEnabledAttr().Get()
        triangle_api = prim.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
        if collision_enabled is not True or not triangle_api:
            raise RuntimeError(
                "converted mesh lacks enabled static triangle collision: "
                f"{prim.GetPath()} collision_enabled={collision_enabled!r} "
                f"triangle_api={triangle_api!r}"
            )
        mesh_rows.append(
            {
                "prim_path": str(prim.GetPath()),
                "point_count": len(points),
                "face_index_count": len(indices),
                "collision_enabled": True,
                "physx_triangle_mesh_collision": True,
            }
        )
    if not mesh_rows:
        raise RuntimeError("converted USD has no geometry meshes")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = bbox.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    minimum = [float(value) for value in aligned.GetMin()]
    maximum = [float(value) for value in aligned.GetMax()]
    if any(upper <= lower for lower, upper in zip(minimum, maximum, strict=True)):
        raise RuntimeError(f"generated USD has invalid world bounds: {minimum} to {maximum}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if meters_per_unit != 1.0:
        raise RuntimeError(
            "HM3D collision candidate must be metres in Isaac runtime; "
            f"stage meters-per-unit is {meters_per_unit}"
        )
    runtime_up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if runtime_up_axis != "Z":
        raise RuntimeError(
            "HM3D collision candidate must use Isaac Z-up runtime coordinates; "
            f"stage up-axis is {runtime_up_axis!r}"
        )
    return {
        "runtime_up_axis": runtime_up_axis,
        "meters_per_unit": meters_per_unit,
        "world_bounds_min_m": minimum,
        "world_bounds_max_m": maximum,
        "mesh_count": len(mesh_rows),
        "meshes": mesh_rows,
    }


def main() -> int:
    # Omniverse modules are imported after SimulationApp creation.
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import CollisionPropertiesCfg, TriangleMeshPropertiesCfg

    source = ARGS.source_glb.expanduser().resolve()
    output_dir = ARGS.output_dir.expanduser().resolve()
    output = output_dir / ARGS.usd_file_name
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not ARGS.overwrite:
        raise FileExistsError(f"USD already exists; inspect it or use --overwrite: {output}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = MeshConverterCfg(
        asset_path=str(source),
        usd_dir=str(output_dir),
        usd_file_name=ARGS.usd_file_name,
        force_usd_conversion=True,
        make_instanceable=False,
        collision_props=CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=TriangleMeshPropertiesCfg(),
    )
    converter = MeshConverter(cfg)
    usd_path = Path(converter.usd_path).resolve()
    if usd_path != output or not usd_path.is_file():
        raise RuntimeError(f"MeshConverter did not create the requested USD: {usd_path}")

    inspection = _inspect_static_triangle_mesh_collision(usd_path)
    transform = {
        "source_declared_up_axis": ARGS.source_up_axis,
        "runtime_stage_up_axis": inspection["runtime_up_axis"],
        "runtime_meters_per_unit": inspection["meters_per_unit"],
        "converter": "isaaclab.MeshConverter",
        "collision_representation": "static_triangle_mesh",
        "translation": [0.0, 0.0, 0.0],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }
    _write_json(
        output_dir / "collision_conversion_manifest.json",
        {
            "schema_version": "hm3d-isaac-collision-usd-conversion-v1",
            "status": "COLLISION_CONVERSION_COMPLETE_NOT_RUNTIME_ADMITTED",
            "scene_id": ARGS.scene_id,
            "split": ARGS.split,
            "source_glb": str(source),
            "source_glb_sha256": _sha256(source),
            "output_usd": str(usd_path),
            "output_usd_sha256": _sha256(usd_path),
            "coordinate_transform": transform,
            "coordinate_transform_sha256": _canonical_sha256(transform),
            "collision": inspection,
            "formal_runtime_admission": False,
            "required_next_evidence": [
                "real CF2X A-B-A reset",
                "PhysX collision replay",
                "mesh/collider/sparse-range consistency audit",
                "3D flight-space admission",
            ],
        },
    )
    print(
        json.dumps(
            {
                "usd": str(usd_path),
                "mesh_count": inspection["mesh_count"],
                "world_bounds_min_m": inspection["world_bounds_min_m"],
                "world_bounds_max_m": inspection["world_bounds_max_m"],
            },
            indent=2,
        )
    )
    return 0


try:
    raise SystemExit(main())
finally:
    SIMULATION_APP.close()
