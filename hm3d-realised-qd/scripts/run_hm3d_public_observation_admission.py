"""Measure the target-free P04 public sparse-range observation contract in Isaac Sim.

This is an admission measurement, not an exploration-policy rollout. It places
a calibration receiver at previously audited free-flight positions and uses
PhysX scene queries to generate the only ray outcomes an eventual method may
consume. The P03 ESDF, collision mesh, and denominator membership remain
evaluator-side throughout.
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

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256  # noqa: E402
from aerocity_method.evaluation.hm3d_exploration_contract import (  # noqa: E402
    DEFAULT_PATH as DEFAULT_EXPLORATION_CONTRACT,
)
from aerocity_method.evaluation.hm3d_exploration_contract import (  # noqa: E402
    load_exploration_observation_contract,
)
from aerocity_method.evaluation.hm3d_exploration_metrics import (  # noqa: E402
    evaluation_denominator_sha256,
)
from aerocity_method.runtime.hm3d_belief import (  # noqa: E402
    PublicRangeRayOutcome,
    SparseVoxelBelief,
)
from aerocity_method.runtime.range_sensing import (  # noqa: E402
    resolve_public_range_directions,
)

RUNNER_VERSION = "hm3d-p04-public-sparse-range-v1"


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
        raise FileExistsError(f"refusing to overwrite P04 runtime evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _point(raw: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates")
    point = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label} must be finite")
    return point


def _receiver_position(row: dict[str, Any], label: str) -> tuple[float, float, float]:
    """Read a range receiver pose without requiring a camera audit artifact."""

    return _point(row.get("receiver_position_w_m"), label)


def _cohort_rows(p03_artifact: dict[str, Any]) -> tuple[dict[str, object], ...]:
    payload = p03_artifact.get("payload")
    if not isinstance(payload, dict) or p03_artifact.get("phase_id") != "P03":
        raise ValueError("--p03-artifact must be a P03 preflight artifact")
    rows = payload.get("scenes")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("P03 artifact needs non-empty scene rows")
    return tuple(rows)


def _matching_p03_row(rows: tuple[dict[str, object], ...], scene_id: str) -> dict[str, object]:
    matches = [row for row in rows if row.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError("P03 artifact must contain this scene exactly once")
    return matches[0]


def _public_payload(
    *,
    scene_id: str,
    source_geometry_sha256: str,
    collision_usd_sha256: str,
    flight_space_manifest_hash: str,
    public_contract_sha256: str,
    evaluation_denominator_digest: str,
    episode_id: str,
    source_observation_ids_total: int,
    accepted_outcome_count: int,
    observed_free_voxels_total: int,
    observation_voxel_resolution_m: float,
    outcome_aggregate_sha256: str,
    belief_sha256: str,
    receiver_position_source_sha256: str,
    receiver_position_count: int,
) -> dict[str, Any]:
    if source_observation_ids_total < 1 or accepted_outcome_count < 1:
        raise ValueError("P04 requires at least one real accepted range outcome")
    if observed_free_voxels_total < 1:
        raise ValueError("P04 range outcomes did not observe any free voxel")
    return {
        "schema_version": "hm3d-p04-public-observation-ledger-v1",
        "status": "P04_PUBLIC_SPARSE_RANGE_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "evidence_class": "real_runtime",
        "runtime_run_id": f"isaac-hm3d-p04-{uuid.uuid4().hex}",
        "runtime_command_sha256": hashlib.sha256(
            "\0".join(str(value) for value in sys.argv).encode("utf-8")
        ).hexdigest(),
        "runner_version": RUNNER_VERSION,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "source_geometry_sha256": source_geometry_sha256,
        "collision_usd_sha256": collision_usd_sha256,
        "flight_space_manifest_hash": flight_space_manifest_hash,
        "public_contract_sha256": public_contract_sha256,
        "evaluation_denominator_sha256": evaluation_denominator_digest,
        "sensor_profile": "sparse-range-3d-vfov90",
        "ray_pattern": "six-axis-range-rays",
        "source_observation_ids_total": source_observation_ids_total,
        "accepted_range_outcomes_total": accepted_outcome_count,
        "observed_free_voxels_total": observed_free_voxels_total,
        "observation_voxel_resolution_m": observation_voxel_resolution_m,
        "source_observation_binding": True,
        "method_private_truth_fields": [],
        "public_outcome_aggregate_sha256": outcome_aggregate_sha256,
        "public_belief_sha256": belief_sha256,
        "receiver_position_source_sha256": receiver_position_source_sha256,
        "receiver_position_count": receiver_position_count,
        "calibration_scope": (
            "P04 sensor-contract admission only: audited receiver positions are evaluator "
            "calibration inputs and are not method-visible exploration waypoints."
        ),
        "claim_limit": (
            "Real PhysX public sparse-range outcomes only. This artifact is not an "
            "exploration-policy result, target-search result, or coverage score."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--receiver-positions-json", type=Path, required=True)
    parser.add_argument("--flight-space-audit", type=Path, required=True)
    parser.add_argument("--p03-artifact", type=Path, required=True)
    parser.add_argument("--exploration-contract", type=Path, default=DEFAULT_EXPLORATION_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewpoint-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--physics-dt-s", type=float, default=1.0 / 120.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace, simulation_app: Any) -> int:
    import omni.physx
    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import UsdGeom

    if args.viewpoint_count < 1 or args.seed < 0 or args.physics_dt_s <= 0.0:
        raise ValueError("invalid P04 viewpoint count, seed, or physics timestep")
    paths = {
        "collision": args.collision_usd.expanduser().resolve(),
        "positions": args.receiver_positions_json.expanduser().resolve(),
        "flight": args.flight_space_audit.expanduser().resolve(),
        "p03": args.p03_artifact.expanduser().resolve(),
        "contract": args.exploration_contract.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    for name in ("collision", "positions", "flight", "p03", "contract"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name}: {paths[name]}")
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite P04 runtime evidence: {paths['output']}")

    contract = load_exploration_observation_contract(paths["contract"])
    flight = _read_object(paths["flight"])
    position_source = _read_object(paths["positions"])
    p03_rows = _cohort_rows(_read_object(paths["p03"]))
    p03_row = _matching_p03_row(p03_rows, args.scene_id)
    if flight.get("scene_id") != args.scene_id or position_source.get("scene_id") != args.scene_id:
        raise ValueError("P04 scene ID disagrees with flight-space or receiver-position evidence")
    expected_collision_sha = str(p03_row["collision_geometry_sha256"])
    if _sha256(paths["collision"]) != expected_collision_sha:
        raise ValueError("collision USD hash differs from frozen P03 flight space")
    if flight.get("source_glb_sha256") != p03_row["source_geometry_sha256"]:
        raise ValueError("flight-space source geometry differs from frozen P03")
    if flight.get("flight_space_manifest_hash") != p03_row["flight_space_manifest_hash"]:
        raise ValueError("flight-space manifest differs from frozen P03")
    raw_views = position_source.get("views")
    if not isinstance(raw_views, list) or len(raw_views) < args.viewpoint_count:
        raise ValueError("P04 needs enough independently audited receiver positions")
    positions = tuple(
        _receiver_position(row, f"views[{index}].receiver_position_w_m")
        for index, row in enumerate(raw_views[: args.viewpoint_count])
        if isinstance(row, dict)
    )
    if len(positions) != args.viewpoint_count:
        raise ValueError("P04 receiver views must all be objects")
    denominator_digest = evaluation_denominator_sha256(p03_rows)
    sensor = contract.payload["sensor_profile"]
    resolution_m = float(contract.payload["public_belief"]["resolution_m"])
    maximum_range_m = float(sensor["maximum_range_m"])
    update_hz = float(sensor["update_hz"])

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    collision_root = UsdGeom.Xform.Define(stage, "/World/HM3DCollision")
    collision_root.GetPrim().GetReferences().AddReference(str(paths["collision"]))
    UsdGeom.Imageable(collision_root).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    for _ in range(12):
        simulation_app.update()
    sim = SimulationContext(
        SimulationCfg(dt=args.physics_dt_s, device=args.device, enable_scene_query_support=True)
    )
    sim.reset()
    for _ in range(3):
        sim.step(render=False)
    interface = omni.physx.get_physx_scene_query_interface()
    belief = SparseVoxelBelief(
        scene_id=args.scene_id,
        agent_id="p04-calibration-receiver",
        resolution_m=resolution_m,
    )
    accepted_outcomes: list[dict[str, Any]] = []
    source_observation_count = 0
    ray_directions = resolve_public_range_directions(str(sensor["ray_pattern"]))
    for view_index, origin in enumerate(positions):
        for direction_index, direction in enumerate(ray_directions):
            source_observation_count += 1
            observation_id = f"p04-{args.scene_id}-{args.seed}-{view_index:03d}-{direction_index}"
            hit = interface.raycast_closest(origin, direction, maximum_range_m)
            if not isinstance(hit, dict) or not bool(hit.get("hit", False)):
                continue
            distance = float(hit.get("distance", 0.0))
            if not 0.02 < distance <= maximum_range_m:
                continue
            endpoint = tuple(origin[axis] + direction[axis] * distance for axis in range(3))
            outcome = PublicRangeRayOutcome(
                observation_id=observation_id,
                agent_id="p04-calibration-receiver",
                timestamp_s=source_observation_count / update_hz,
                origin_m=origin,
                endpoint_m=endpoint,
                hit_occupied=True,
            )
            if belief.integrate_ray(outcome):
                accepted_outcomes.append(outcome.to_dict())
        sim.step(render=False)
    payload = _public_payload(
        scene_id=args.scene_id,
        source_geometry_sha256=str(p03_row["source_geometry_sha256"]),
        collision_usd_sha256=expected_collision_sha,
        flight_space_manifest_hash=str(p03_row["flight_space_manifest_hash"]),
        public_contract_sha256=contract.digest,
        evaluation_denominator_digest=denominator_digest,
        episode_id=f"p04-public-range-{args.scene_id}-{args.seed}",
        source_observation_ids_total=source_observation_count,
        accepted_outcome_count=len(accepted_outcomes),
        observed_free_voxels_total=belief.observed_free_count,
        observation_voxel_resolution_m=resolution_m,
        outcome_aggregate_sha256=canonical_sha256(accepted_outcomes),
        belief_sha256=belief.content_sha256,
        receiver_position_source_sha256=_sha256(paths["positions"]),
        receiver_position_count=len(positions),
    )
    _write_new(paths["output"], payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scene_id": args.scene_id,
                "accepted_range_outcomes_total": payload["accepted_range_outcomes_total"],
                "observed_free_voxels_total": payload["observed_free_voxels_total"],
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
    # Windows Kit shutdown can hang after large static scene-query stages.
    # The result is atomically persisted before process isolation exits.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
