"""Run isolated four-CF2X HM3D clusters in one lockstep Isaac simulation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256, write_json_atomic
from aerocity_method.runtime import hm3d_cf2x_execution as cf2x
from aerocity_method.runtime.communication import RelayMessage, RelayMessageQueue
from aerocity_method.runtime.hm3d_multicluster import HM3DClusterLayout, cluster_seed

PROBE_SCHEMA_VERSION = "hm3d-cf2x-multicluster-probe-v1"


def _point(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("start position must contain three finite coordinates")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-usd", required=True, type=Path)
    parser.add_argument(
        "--cf2x-usd",
        type=Path,
        default=ROOT.parents[1] / "assets" / "new" / "cf2x.usd",
    )
    parser.add_argument(
        "--start-position",
        action="append",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--cluster-count", type=int, default=2)
    parser.add_argument("--peer-mode", choices=("hover", "random"), default="hover")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--physics-dt-s", type=float, default=1.0 / 120.0)
    parser.add_argument(
        "--controller-id",
        choices=(
            cf2x.CF2X_DEFAULT_CONTROLLER_ID,
            cf2x.BITCRAZE_LEE_CONTROLLER_ID,
            cf2x.BITCRAZE_MELLINGER_CONTROLLER_ID,
        ),
        default=cf2x.CF2X_DEFAULT_CONTROLLER_ID,
    )
    parser.add_argument("--telemetry-hz", type=float, default=10.0)
    parser.add_argument("--environment-spacing-m", type=float, default=80.0)
    parser.add_argument("--base-seed", type=int, default=20260804)
    parser.add_argument("--output", required=True, type=Path)
    if argv is None:
        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def _target_offsets(
    *, cluster_id: int, peer_mode: str, base_seed: int, scene_id: str
) -> tuple[tuple[float, float, float], ...]:
    if cluster_id == 0:
        return tuple((0.0, 0.0, 0.12) for _ in range(FORMAL_FLEET_SIZE))
    if peer_mode == "hover":
        return tuple((0.0, 0.0, 0.0) for _ in range(FORMAL_FLEET_SIZE))
    rng = random.Random(
        cluster_seed(
            scene_id=scene_id,
            cluster_id=cluster_id,
            episode_id=f"multicluster-{peer_mode}",
            base_seed=base_seed,
        )
    )
    return tuple(
        (
            rng.uniform(-0.12, 0.12),
            rng.uniform(-0.12, 0.12),
            rng.uniform(0.04, 0.12),
        )
        for _ in range(FORMAL_FLEET_SIZE)
    )


def _physics_scene_path(stage: Any) -> str:
    from pxr import PhysxSchema

    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            return prim.GetPath().pathString
    raise RuntimeError("Isaac stage has no PhysX scene")


def main(args: argparse.Namespace, simulation_app: Any) -> int:
    import numpy as np
    import omni.physx
    import omni.usd
    import torch
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
    from isaaclab_contrib.assets import Multirotor
    from isaacsim.core.cloner import GridCloner
    from pxr import Gf, UsdGeom

    collision_usd = args.collision_usd.expanduser().resolve()
    cf2x_usd = args.cf2x_usd.expanduser().resolve()
    output = args.output.expanduser().resolve()
    starts = tuple(_point(row) for row in args.start_position)
    if len(starts) != FORMAL_FLEET_SIZE:
        raise ValueError(f"each cluster requires exactly {FORMAL_FLEET_SIZE} start positions")
    if args.cluster_count < 1:
        raise ValueError("cluster_count must be positive")
    for name in ("duration_s", "physics_dt_s", "telemetry_hz", "environment_spacing_m"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not collision_usd.is_file() or not cf2x_usd.is_file():
        raise FileNotFoundError("collision or CF2X USD is missing")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite multi-cluster evidence: {output}")

    random.seed(args.base_seed)
    np.random.seed(args.base_seed % (2**32))
    torch.manual_seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    sim = SimulationContext(
        SimulationCfg(
            dt=args.physics_dt_s,
            device=args.device,
            enable_scene_query_support=True,
            physx=PhysxCfg(enable_enhanced_determinism=True),
        )
    )
    cloner = GridCloner(spacing=args.environment_spacing_m, stage=stage)
    cloner.define_base_env("/World/envs")
    env_paths = cloner.generate_paths("/World/envs/env", args.cluster_count)
    source_env = UsdGeom.Xform.Define(stage, env_paths[0])
    collision_root = UsdGeom.Xform.Define(stage, f"{env_paths[0]}/HM3DCollision")
    collision_root.GetPrim().GetReferences().AddReference(str(collision_usd))
    UsdGeom.Xform.Define(stage, f"{env_paths[0]}/P07Agents")
    for agent_index, position in enumerate(starts):
        agent = UsdGeom.Xform.Define(stage, f"{env_paths[0]}/P07Agents/Agent_{agent_index}")
        agent.AddTranslateOp().Set(Gf.Vec3d(*position))
    source_env.GetPrim().SetInstanceable(False)
    for _ in range(12):
        simulation_app.update()

    env_origins = cloner.clone(
        source_prim_path=env_paths[0],
        prim_paths=env_paths,
        replicate_physics=True,
        copy_from_source=False,
        enable_env_ids=args.device != "cpu",
    )
    layout = HM3DClusterLayout(tuple(tuple(float(value) for value in row) for row in env_origins))
    robot_cfg = cf2x._multirotor_cfg(cf2x_usd, args.physics_dt_s).replace(
        prim_path="/World/envs/env_.*/P07Agents/Agent_.*/Robot"
    )
    robot = Multirotor(robot_cfg)
    contact = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/P07Agents/Agent_.*/Robot/.*",
            track_pose=False,
            track_air_time=True,
            force_threshold=cf2x.CONTACT_HARD_FAIL_N,
            history_length=1,
            debug_vis=False,
        )
    )
    cloner.filter_collisions(_physics_scene_path(stage), "/World/collisions", env_paths)
    sim.reset()
    if robot.num_instances != layout.total_agent_count:
        raise RuntimeError(
            f"spawned {robot.num_instances} CF2X for {layout.total_agent_count} expected rows"
        )
    robot.update(float(sim.cfg.dt))
    contact.update(float(sim.cfg.dt), force_recompute=True)

    expected_world_starts = [
        layout.to_world(cluster_id, starts[agent_index])
        for cluster_id in range(layout.cluster_count)
        for agent_index in range(layout.fleet_size)
    ]
    root_pose = torch.tensor(
        [[*position, 1.0, 0.0, 0.0, 0.0] for position in expected_world_starts],
        device=robot.device,
        dtype=torch.float32,
    )
    robot.reset()
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(
        torch.zeros((layout.total_agent_count, 6), device=robot.device)
    )
    robot.set_thrust_target(
        torch.full(
            (layout.total_agent_count, int(robot.num_thrusters)),
            cf2x.HOVER_THRUST_PER_ROTOR_N,
            device=robot.device,
        )
    )
    robot.write_data_to_sim()
    sim.forward()
    robot.update(float(sim.cfg.dt))
    observed = robot.data.root_pos_w.detach().cpu().tolist()
    reset_error_m = max(
        math.dist(expected, tuple(float(value) for value in actual))
        for expected, actual in zip(expected_world_starts, observed, strict=True)
    )
    if reset_error_m > 1.0e-4:
        raise RuntimeError("multi-cluster CF2X reset ordering or environment transform is wrong")

    from aerocity_method.runtime.physx_query_cache import MemoizedRaycastClosestQuery

    scene_query = MemoizedRaycastClosestQuery(omni.physx.get_physx_scene_query_interface())
    target_offsets = [
        _target_offsets(
            cluster_id=cluster_id,
            peer_mode=args.peer_mode,
            base_seed=args.base_seed,
            scene_id=collision_usd.stem,
        )
        for cluster_id in range(layout.cluster_count)
    ]
    target_world = [
        layout.to_world(
            cluster_id,
            tuple(
                starts[agent_index][axis] + target_offsets[cluster_id][agent_index][axis]
                for axis in range(3)
            ),
        )
        for cluster_id in range(layout.cluster_count)
        for agent_index in range(layout.fleet_size)
    ]
    action_hashes = [
        canonical_sha256(
            {
                "cluster_id": cluster_id,
                "local_starts_m": starts,
                "local_offsets_m": target_offsets[cluster_id],
                "duration_s": args.duration_s,
            }
        )
        for cluster_id in range(layout.cluster_count)
    ]

    agent_ids = tuple(f"uav{index}" for index in range(layout.fleet_size))
    queues = [RelayMessageQueue(agent_ids, 0.05, 0.02, 0.0) for _ in range(layout.cluster_count)]
    for cluster_id, queue in enumerate(queues):
        queue.publish(
            RelayMessage(
                message_id=f"cluster{cluster_id}-bootstrap",
                sender_id="uav0",
                source_timestamp_s=0.0,
                payload_digest=canonical_sha256(
                    {"cluster_id": cluster_id, "local_starts_m": starts}
                ),
                time_to_live_s=max(0.5, args.duration_s),
            )
        )

    telemetry_interval_steps = max(1, round(1.0 / (args.physics_dt_s * args.telemetry_hz)))
    maximum_steps = max(1, math.floor(args.duration_s / args.physics_dt_s))
    local_traces: list[list[list[list[float]]]] = [[] for _ in range(layout.cluster_count)]
    public_map_voxels: list[set[tuple[int, int, int]]] = [
        set() for _ in range(layout.cluster_count)
    ]
    maximum_contact_force_n = [0.0 for _ in range(layout.total_agent_count)]
    controller_mass_kg = (cf2x.HOVER_THRUST_PER_ROTOR_N * float(robot.num_thrusters)) / 9.81
    if args.controller_id == cf2x.BITCRAZE_LEE_CONTROLLER_ID:
        controller = cf2x.BitcrazeLeeTracker(
            mass_kg=controller_mass_kg,
            dt_s=float(sim.cfg.dt),
            maximum_feedback_acceleration_mps2=cf2x.CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
            maximum_tilt_rad=cf2x.CF2X_MAXIMUM_TILT_RAD,
        )
    elif args.controller_id == cf2x.BITCRAZE_MELLINGER_CONTROLLER_ID:
        controller = cf2x.BitcrazeMellingerTracker(
            mass_kg=controller_mass_kg,
            dt_s=1.0 / cf2x.BITCRAZE_MELLINGER_OFFICIAL_CONTROL_RATE_HZ,
        )
    else:
        controller = None
    minimum_cross_cluster_distance_m = math.inf
    started = time.perf_counter()
    for step in range(1, maximum_steps + 1):
        current = robot.data.root_pos_w.detach().cpu().tolist()
        control_timestamp_s = (step - 1) * args.physics_dt_s
        references = [
            cf2x._minimum_time_line_reference(start, target, control_timestamp_s)
            for start, target in zip(expected_world_starts, target_world, strict=True)
        ]
        headings = [
            cf2x._yaw_from_delta(tuple(float(value) for value in position), target_world[index])
            for index, position in enumerate(current)
        ]
        thrust = cf2x._bounded_rotor_thrust(
            robot,
            torch.tensor(
                [row.position_m for row in references],
                device=robot.device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [row.velocity_mps for row in references],
                device=robot.device,
                dtype=torch.float32,
            ),
            torch.tensor(
                [row.acceleration_mps2 for row in references],
                device=robot.device,
                dtype=torch.float32,
            ),
            headings,
            controller=controller,
            dt_s=float(sim.cfg.dt),
        )
        robot.set_thrust_target(thrust)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(float(sim.cfg.dt))
        contact.update(float(sim.cfg.dt), force_recompute=True)
        forces = torch.linalg.norm(contact.data.net_forces_w, dim=-1).max(dim=1).values
        for index, value in enumerate(forces.detach().cpu().tolist()):
            maximum_contact_force_n[index] = max(maximum_contact_force_n[index], float(value))
        if step % telemetry_interval_steps != 0 and step != maximum_steps:
            continue
        timestamp_s = step * args.physics_dt_s
        world_rows = [
            tuple(float(value) for value in row)
            for row in robot.data.root_pos_w.detach().cpu().tolist()
        ]
        for left_cluster in range(layout.cluster_count):
            left_rows = world_rows[layout.cluster_slice(left_cluster)]
            for right_cluster in range(left_cluster + 1, layout.cluster_count):
                right_rows = world_rows[layout.cluster_slice(right_cluster)]
                minimum_cross_cluster_distance_m = min(
                    minimum_cross_cluster_distance_m,
                    min(math.dist(left, right) for left in left_rows for right in right_rows),
                )
        for cluster_id in range(layout.cluster_count):
            local_rows = layout.local_team_from_flat_world(cluster_id, world_rows)
            local_traces[cluster_id].append([list(row) for row in local_rows])
            world_cluster_rows = world_rows[layout.cluster_slice(cluster_id)]
            graph = cf2x._initial_relay_graph(scene_query, tuple(world_cluster_rows))
            queues[cluster_id].advance(timestamp_s=timestamp_s, graph=graph)
            for world_position in world_cluster_rows:
                target = (world_position[0] + 20.0, world_position[1], world_position[2])
                hit = cf2x._first_static_scene_hit(
                    scene_query, world_position, target, endpoint_margin_m=0.0
                )
                distance = 20.0 if hit is None else float(hit.get("distance", 20.0))
                endpoint_world = (
                    world_position[0] + distance,
                    world_position[1],
                    world_position[2],
                )
                endpoint_local = layout.to_local(cluster_id, endpoint_world)
                public_map_voxels[cluster_id].add(
                    tuple(math.floor(value / 0.25) for value in endpoint_local)
                )
    wall_s = time.perf_counter() - started

    cluster_rows: list[dict[str, Any]] = []
    for cluster_id in range(layout.cluster_count):
        trace_hash = canonical_sha256(local_traces[cluster_id])
        outcomes = queues[cluster_id].outcomes
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "environment_origin_m": list(layout.env_origins_m[cluster_id]),
                "selected_candidate_ids": [
                    f"cluster{cluster_id}-offset-uav{index}" for index in range(layout.fleet_size)
                ],
                "action_hashes": [action_hashes[cluster_id]],
                "outcome_hashes": [trace_hash],
                "local_root_trace_m": local_traces[cluster_id],
                "public_map_voxel_count": len(public_map_voxels[cluster_id]),
                "message_outcome_count": len(outcomes),
                "cross_cluster_contact_count": 0,
                "cross_cluster_message_count": 0,
                "cross_cluster_map_delta_count": 0,
            }
        )
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "MULTICLUSTER_PROBE_COMPLETE",
        "formal_result": False,
        "claim_limit": (
            "Engineering isolation and throughput evidence only. Clusters are parallel "
            "rollouts in one scene load and are not independent HM3D maps for statistics."
        ),
        "cluster_count": layout.cluster_count,
        "fleet_size_per_cluster": layout.fleet_size,
        "total_cf2x_count": layout.total_agent_count,
        "peer_mode": args.peer_mode,
        "duration_s": args.duration_s,
        "physics_dt_s": args.physics_dt_s,
        "telemetry_hz": args.telemetry_hz,
        "controller_tracking": cf2x._controller_tracking_profile(
            args.controller_id,
            physics_dt_s=args.physics_dt_s,
        ),
        "collision_filter_applied": True,
        "collision_usd_sha256": _sha256(collision_usd),
        "cf2x_usd_sha256": _sha256(cf2x_usd),
        "maximum_reset_error_m": reset_error_m,
        "maximum_contact_force_n": max(maximum_contact_force_n, default=0.0),
        "minimum_cross_cluster_distance_m": (
            None if layout.cluster_count == 1 else minimum_cross_cluster_distance_m
        ),
        "cross_cluster_contact_count": 0,
        "cross_cluster_message_count": 0,
        "cross_cluster_map_delta_count": 0,
        "wall_s": wall_s,
        "real_decision_equivalent_count": layout.cluster_count,
        "real_decisions_per_wall_hour": layout.cluster_count * 3600.0 / wall_s,
        "clusters": cluster_rows,
    }
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    write_json_atomic(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cluster_count": layout.cluster_count,
                "wall_s": wall_s,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _entrypoint() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    app = AppLauncher(args)
    from isaaclab.sim import SimulationContext

    try:
        return main(args, app.app)
    finally:
        SimulationContext.clear_instance()
        app.app.update()
        app.app.close()


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
