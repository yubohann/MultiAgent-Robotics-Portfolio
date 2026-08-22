"""Measure a target-free vertical-exploration counterfactual in Isaac Sim.

This P03 probe is deliberately independent from the retired target-search
pipeline.  It uses matched, target-free ESDF routes and PhysX range outcomes to
compare free-height sensing with a fixed-height control.  The ESDF mask is used
only by the evaluator to score real ray-confirmed free cells; policies never
receive it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf  # noqa: E402
from aerocity_method.contracts.io import canonical_sha256  # noqa: E402
from aerocity_method.evaluation.hm3d_exploration_metrics import (  # noqa: E402
    ExplorationMetricSample,
    score_exploration_episode,
)
from aerocity_method.runtime.hm3d_belief import (  # noqa: E402
    PublicRangeRayOutcome,
    SparseVoxelBelief,
)
from aerocity_method.runtime.hm3d_calibration_geometry import (  # noqa: E402
    densest_height_slice_index,
    farthest_spread_indices,
)

RUNNER_VERSION = "hm3d-p03-target-free-vertical-counterfactual-v2"
_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite measured evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--collision-derivative-manifest", type=Path, required=True)
    parser.add_argument("--flight-space-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--route-pose-count", type=int, default=36)
    parser.add_argument("--dwell-seconds", type=float, default=0.4)
    parser.add_argument("--resolution-m", type=float, default=0.25)
    parser.add_argument("--vehicle-clearance-m", type=float, default=0.30)
    parser.add_argument("--maximum-range-m", type=float, default=20.0)
    parser.add_argument("--physics-dt-s", type=float, default=1.0 / 120.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _load_collision_provenance(
    *,
    scene_id: str,
    source_glb: Path,
    collision_usd: Path,
    collision_manifest: Path,
    derivative_manifest: Path,
) -> dict[str, str]:
    source = _read_object(collision_manifest)
    derivative = _read_object(derivative_manifest)
    if source.get("scene_id") != scene_id:
        raise ValueError("collision conversion manifest scene mismatch")
    if Path(str(source.get("source_glb", ""))).resolve() != source_glb:
        raise ValueError("collision conversion source GLB path mismatch")
    if source.get("source_glb_sha256") != _sha256(source_glb):
        raise ValueError("source GLB changed after collision conversion")
    if (
        Path(str(derivative.get("source_collision_usd", ""))).resolve()
        != Path(str(source.get("output_usd", ""))).resolve()
    ):
        raise ValueError("derivative source collider mismatch")
    if derivative.get("source_collision_usd_sha256") != source.get("output_usd_sha256"):
        raise ValueError("derivative source collider hash mismatch")
    if Path(str(derivative.get("output_usd", ""))).resolve() != collision_usd:
        raise ValueError("derivative output collider mismatch")
    if derivative.get("output_usd_sha256") != _sha256(collision_usd):
        raise ValueError("derivative collider hash mismatch")
    operation = derivative.get("derivative")
    if not isinstance(operation, dict):
        raise ValueError("derivative operation is missing")
    if operation.get("changes_vertex_positions") is not False:
        raise ValueError("derivative changes vertex positions")
    if operation.get("changes_world_transform") is not False:
        raise ValueError("derivative changes world transform")
    return {
        "source_glb_sha256": _sha256(source_glb),
        "collision_usd_sha256": _sha256(collision_usd),
        "collision_manifest_sha256": _sha256(collision_manifest),
        "collision_derivative_manifest_sha256": _sha256(derivative_manifest),
    }


def _load_triangle_mesh(usd_path: Path) -> Any:
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open collision USD: {usd_path}")
    cache = UsdGeom.XformCache()
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
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
        raise ValueError("collision USD contains no mesh triangles")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )


def _largest_component(arrays: dict[str, Any]) -> np.ndarray:
    free = np.asarray(arrays["free_mask"], dtype=bool)
    labels = np.asarray(arrays["component_labels"], dtype=np.int32)
    counts = np.bincount(labels[free])
    counts[0] = 0
    component = int(np.argmax(counts))
    if component == 0:
        raise ValueError("ESDF has no retained free-flight component")
    return free & (labels == component)


def _grid_points(arrays: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
    resolution = float(np.asarray(arrays["resolution_m"]).item())
    return origin + np.argwhere(mask).astype(np.float64) * resolution


def _matched_routes(
    arrays: dict[str, Any], *, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    component = _largest_component(arrays)
    free_points = _grid_points(arrays, component)
    if len(free_points) < count:
        raise ValueError("largest free-flight component cannot supply the requested route")
    free_route = free_points[farthest_spread_indices(free_points, count=count, seed=seed)]
    # Use the strongest feasible fixed-height control. A median active slice can
    # be a narrow connector and can reject otherwise valid multi-level scenes.
    fixed_z_index = densest_height_slice_index(component)
    fixed_mask = component & (np.indices(component.shape)[2] == fixed_z_index)
    fixed_points = _grid_points(arrays, fixed_mask)
    if len(fixed_points) < count:
        raise ValueError("fixed-height slice cannot supply the matched route")
    fixed_route = fixed_points[farthest_spread_indices(fixed_points, count=count, seed=seed)]
    return free_route, fixed_route, float(fixed_points[0, 2])


def _world_to_index(
    point: np.ndarray, *, origin: np.ndarray, resolution_m: float, shape: tuple[int, int, int]
) -> tuple[int, int, int] | None:
    index = np.rint((point - origin) / resolution_m).astype(np.int64)
    if np.any(index < 0) or np.any(index >= np.asarray(shape, dtype=np.int64)):
        return None
    return tuple(int(value) for value in index)


def _evaluator_free_indices(
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    hit_distance_m: float,
    component: np.ndarray,
    grid_origin: np.ndarray,
    resolution_m: float,
) -> set[tuple[int, int, int]]:
    # Exact hit endpoints are occupied.  The half-cell offset avoids awarding
    # the terminal surface cell as free while retaining conservative free rays.
    free_length = max(0.0, hit_distance_m - resolution_m * 0.5)
    steps = max(1, int(math.ceil(free_length / (resolution_m * 0.5))))
    output: set[tuple[int, int, int]] = set()
    for step in range(steps + 1):
        point = origin + direction * (free_length * step / steps)
        index = _world_to_index(
            point,
            origin=grid_origin,
            resolution_m=resolution_m,
            shape=tuple(int(value) for value in component.shape),
        )
        if index is not None and component[index]:
            output.add(index)
    return output


def _run_route(
    *,
    route_name: str,
    route: np.ndarray,
    scene_id: str,
    seed: int,
    dwell_seconds: float,
    maximum_range_m: float,
    component: np.ndarray,
    grid_origin: np.ndarray,
    resolution_m: float,
    interface: Any,
    sim: Any,
) -> tuple[dict[str, Any], set[tuple[int, int, int]]]:
    denominator_volume = float(component.sum()) * resolution_m**3
    if denominator_volume <= 0.0:
        raise ValueError("counterfactual evaluator denominator is empty")
    belief = SparseVoxelBelief(
        scene_id=scene_id,
        agent_id=f"p03-{route_name}",
        resolution_m=resolution_m,
    )
    confirmed: set[tuple[int, int, int]] = set()
    accepted_outcomes: list[dict[str, Any]] = []
    samples = [
        ExplorationMetricSample(
            timestamp_s=0.0,
            explored_free_volume_m3=0.0,
            true_free_volume_m3=denominator_volume,
            predicted_free_volume_m3=0.0,
        )
    ]
    query_count = 0
    for pose_index, raw_origin in enumerate(route):
        origin = tuple(float(value) for value in raw_origin)
        for direction_index, direction_values in enumerate(_DIRECTIONS):
            query_count += 1
            direction = np.asarray(direction_values, dtype=np.float64)
            hit = interface.raycast_closest(
                origin,
                tuple(float(value) for value in direction),
                maximum_range_m,
            )
            if not isinstance(hit, dict) or not bool(hit.get("hit", False)):
                continue
            distance = float(hit.get("distance", 0.0))
            if not 0.02 < distance <= maximum_range_m:
                continue
            endpoint = tuple(origin[axis] + direction_values[axis] * distance for axis in range(3))
            outcome = PublicRangeRayOutcome(
                observation_id=(
                    f"p03-{scene_id}-{seed}-{route_name}-{pose_index:03d}-{direction_index}"
                ),
                agent_id=f"p03-{route_name}",
                timestamp_s=(pose_index + 1) * dwell_seconds,
                origin_m=origin,
                endpoint_m=endpoint,
                hit_occupied=True,
            )
            if belief.integrate_ray(outcome):
                accepted_outcomes.append(outcome.to_dict())
                confirmed.update(
                    _evaluator_free_indices(
                        origin=np.asarray(origin, dtype=np.float64),
                        direction=direction,
                        hit_distance_m=distance,
                        component=component,
                        grid_origin=grid_origin,
                        resolution_m=resolution_m,
                    )
                )
        sim.step(render=False)
        explored_volume = len(confirmed) * resolution_m**3
        samples.append(
            ExplorationMetricSample(
                timestamp_s=(pose_index + 1) * dwell_seconds,
                explored_free_volume_m3=explored_volume,
                true_free_volume_m3=denominator_volume,
                predicted_free_volume_m3=belief.observed_free_count * resolution_m**3,
                hallucinated_free_volume_m3=max(
                    0.0, (belief.observed_free_count - len(confirmed)) * resolution_m**3
                ),
            )
        )
    report = score_exploration_episode(
        episode_id=f"p03-{route_name}-{scene_id}-{seed}",
        samples=tuple(samples),
        horizon_s=len(route) * dwell_seconds,
    )
    return (
        {
            "route_hash": canonical_sha256(route.tolist()),
            "route_pose_count": len(route),
            "physx_query_count": query_count,
            "accepted_range_outcomes_total": len(accepted_outcomes),
            "outcome_aggregate_sha256": canonical_sha256(accepted_outcomes),
            "public_belief_sha256": belief.content_sha256,
            "evaluator_confirmed_free_voxel_count": len(confirmed),
            "metric": report.to_dict(),
        },
        confirmed,
    )


def main(args: argparse.Namespace, simulation_app: Any) -> int:
    import omni.physx
    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import UsdGeom

    if args.seed < 0 or args.route_pose_count < 2 or args.dwell_seconds <= 0.0:
        raise ValueError("invalid seed, route-pose-count, or dwell-seconds")
    if not 0.05 <= args.resolution_m <= 0.5 or args.vehicle_clearance_m < args.resolution_m / 2.0:
        raise ValueError("invalid ESDF resolution or vehicle clearance")
    if args.maximum_range_m <= 0.02 or args.physics_dt_s <= 0.0:
        raise ValueError("invalid maximum-range or physics timestep")
    paths = {
        "source_glb": args.source_glb.expanduser().resolve(),
        "collision": args.collision_usd.expanduser().resolve(),
        "manifest": args.collision_manifest.expanduser().resolve(),
        "derivative": args.collision_derivative_manifest.expanduser().resolve(),
        "flight": args.flight_space_audit.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    for name in ("source_glb", "collision", "manifest", "derivative", "flight"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name}: {paths[name]}")
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite measured evidence: {paths['output']}")
    provenance = _load_collision_provenance(
        scene_id=args.scene_id,
        source_glb=paths["source_glb"],
        collision_usd=paths["collision"],
        collision_manifest=paths["manifest"],
        derivative_manifest=paths["derivative"],
    )
    prior = _read_object(paths["flight"])
    if prior.get("scene_id") != args.scene_id:
        raise ValueError("flight-space audit scene mismatch")
    if prior.get("source_glb_sha256") != provenance["source_glb_sha256"]:
        raise ValueError("flight-space source geometry mismatch")
    if prior.get("collision_usd_sha256") != provenance["collision_usd_sha256"]:
        raise ValueError("flight-space collision geometry mismatch")

    mesh = _load_triangle_mesh(paths["collision"])
    arrays, rebuilt = build_enclosed_esdf(
        mesh, resolution_m=args.resolution_m, vehicle_clearance_m=args.vehicle_clearance_m
    )
    if rebuilt["flight_space_manifest_hash"] != prior.get("flight_space", {}).get(
        "flight_space_manifest_hash"
    ):
        raise ValueError("rebuilt ESDF differs from the independently audited flight space")
    component = _largest_component(arrays)
    free_route, fixed_route, fixed_altitude_m = _matched_routes(
        arrays, count=args.route_pose_count, seed=args.seed
    )

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    root = UsdGeom.Xform.Define(stage, "/World/HM3D")
    root.GetPrim().GetReferences().AddReference(str(paths["collision"]))
    for _ in range(12):
        simulation_app.update()
    sim = SimulationContext(
        SimulationCfg(dt=args.physics_dt_s, device=args.device, enable_scene_query_support=True)
    )
    sim.reset()
    for _ in range(3):
        sim.step(render=False)
    interface = omni.physx.get_physx_scene_query_interface()
    grid_origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
    free_result, free_confirmed = _run_route(
        route_name="free_height",
        route=free_route,
        scene_id=args.scene_id,
        seed=args.seed,
        dwell_seconds=args.dwell_seconds,
        maximum_range_m=args.maximum_range_m,
        component=component,
        grid_origin=grid_origin,
        resolution_m=args.resolution_m,
        interface=interface,
        sim=sim,
    )
    fixed_result, fixed_confirmed = _run_route(
        route_name="fixed_height",
        route=fixed_route,
        scene_id=args.scene_id,
        seed=args.seed,
        dwell_seconds=args.dwell_seconds,
        maximum_range_m=args.maximum_range_m,
        component=component,
        grid_origin=grid_origin,
        resolution_m=args.resolution_m,
        interface=interface,
        sim=sim,
    )
    free_metric = free_result["metric"]
    fixed_metric = fixed_result["metric"]
    free_only = free_confirmed - fixed_confirmed
    vertical_fraction = len(free_only) / len(free_confirmed) if free_confirmed else 0.0
    payload = {
        "schema_version": "hm3d-p03-target-free-vertical-counterfactual-v2",
        "status": "P03_VERTICAL_COUNTERFACTUAL_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "evidence_class": "real_runtime",
        "runtime_run_id": f"isaac-hm3d-p03-vertical-{uuid.uuid4().hex}",
        "runtime_command_sha256": hashlib.sha256(
            "\0".join(str(value) for value in sys.argv).encode("utf-8")
        ).hexdigest(),
        "runner_version": RUNNER_VERSION,
        "scene_id": args.scene_id,
        "source_glb_sha256": provenance["source_glb_sha256"],
        "collision_usd_sha256": provenance["collision_usd_sha256"],
        "flight_space_manifest_hash": prior["flight_space_manifest_hash"],
        "sensor_profile": "sparse-range-3d-vfov90",
        "ray_pattern": "six-axis-range-rays",
        "counterfactual_type": "matched-target-free-ESDF-routes-with-identical-physx-range-budget",
        "fixed_altitude_selection_rule": "densest-slice-in-largest-connected-component",
        "fixed_altitude_m": fixed_altitude_m,
        "dwell_seconds": args.dwell_seconds,
        "observation_horizon_seconds": args.route_pose_count * args.dwell_seconds,
        "free_height": free_result,
        "fixed_height": fixed_result,
        "fixed_altitude_counterfactual": {
            "run": True,
            "free_height_explored_free_volume_auc_time": free_metric[
                "explored_free_flight_volume_auc_time"
            ],
            "fixed_height_explored_free_volume_auc_time": fixed_metric[
                "explored_free_flight_volume_auc_time"
            ],
            "explored_free_volume_auc_delta": (
                free_metric["explored_free_flight_volume_auc_time"]
                - fixed_metric["explored_free_flight_volume_auc_time"]
            ),
            "policy_claim": "not-a-learned-policy-result",
        },
        "vertical_geometry_probe": {
            "probe_family": "target-free-real-PhysX-range-outcomes",
            "free_height_confirmed_free_voxels": len(free_confirmed),
            "fixed_height_confirmed_free_voxels": len(fixed_confirmed),
            "free_height_only_confirmed_free_voxels": len(free_only),
            "vertical_opportunity_fraction": vertical_fraction,
        },
        "claim_limit": (
            "P03 geometry admission only. This is not an exploration-policy, baseline, "
            "QD, OGFR, RL, or formal-result comparison."
        ),
    }
    payload["counterfactual_sha256"] = canonical_sha256(payload)
    _write_new(paths["output"], payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scene_id": args.scene_id,
                "free_auc": free_metric["explored_free_flight_volume_auc_time"],
                "fixed_auc": fixed_metric["explored_free_flight_volume_auc_time"],
                "delta": payload["fixed_altitude_counterfactual"]["explored_free_volume_auc_delta"],
                "vertical_opportunity_fraction": vertical_fraction,
                "output": str(paths["output"]),
            },
            sort_keys=True,
        )
    )
    return 0


def _entrypoint() -> int:
    args = parse_args()
    app = AppLauncher(args)
    exit_code = main(args, app.app)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
