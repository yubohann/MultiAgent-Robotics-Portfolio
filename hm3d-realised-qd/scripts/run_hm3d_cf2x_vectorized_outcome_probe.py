"""Exercise the full outcome backend for cloned four-CF2X HM3D clusters."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROBE_PROCESS_WALL_STARTED = time.perf_counter()

from aerocity_method.adapters.hm3d_execution import execute_hm3d_manifest
from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256, write_json_atomic
from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentTypeSignature,
    PublicMethodContext,
)
from aerocity_method.evaluation.hm3d_communication_contract import (
    HM3DCommunicationContract,
)
from aerocity_method.runtime import hm3d_cf2x_execution as cf2x
from aerocity_method.runtime.hm3d_cf2x_vectorized_execution import (
    IsaacCF2XVectorizedExecutionBackend,
)
from aerocity_method.runtime.hm3d_multicluster import (
    HM3DClusterLayout,
    cluster_seed,
    validate_cluster_start_sets,
)
from aerocity_method.runtime.tokens import authorize_manifest

SCHEMA_VERSION = "hm3d-cf2x-vectorized-outcome-probe-v1"


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


def _physics_scene_path(stage: Any) -> str:
    from pxr import PhysxSchema

    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            return prim.GetPath().pathString
    raise RuntimeError("Isaac stage has no PhysX scene")


def _manifest(
    *,
    cluster_id: int,
    transit_paths_m: tuple[tuple[tuple[float, float, float], ...], ...],
    transit_end_s: float,
    horizon_s: float,
) -> tuple[PublicMethodContext, CandidateFragmentManifest]:
    context = PublicMethodContext(
        context_id=f"vectorized-probe-cluster{cluster_id}",
        episode_id=f"vectorized-probe-cluster{cluster_id}",
        decision_id="decision0",
        agent_features=tuple((f"uav{index}", (1.0, 1.0)) for index in range(len(transit_paths_m))),
        public_features=(("sparse_range_schedule_hz", 10.0),),
        budget=(("time_remaining_s", horizon_s),),
    )
    fragments: list[FragmentInstance] = []
    if len(transit_paths_m) != FORMAL_FLEET_SIZE:
        raise ValueError("transit paths must match the formal four-CF2X probe fleet")
    for index, path_m in enumerate(transit_paths_m):
        if len(path_m) < 2:
            raise ValueError("transit probe path requires at least two points")
        agent_id = f"uav{index}"
        endpoint = path_m[-1]
        fragments.extend(
            (
                FragmentInstance(
                    instance_fragment_id=f"cluster{cluster_id}-{agent_id}-transit",
                    type_signature=FragmentTypeSignature(
                        "transit", (("probe", "configured_3d_move"),)
                    ),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=0.0,
                    planned_end=transit_end_s,
                    path=path_m,
                    pose_mode="guarded_waypoint",
                    context_bucket="vectorized-outcome-probe",
                ),
                FragmentInstance(
                    instance_fragment_id=f"cluster{cluster_id}-{agent_id}-observe",
                    type_signature=FragmentTypeSignature(
                        "observation", (("probe", "sparse_range_3d"),)
                    ),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=transit_end_s,
                    planned_end=horizon_s,
                    path=(endpoint,),
                    pose_mode="dwell",
                    context_bucket="vectorized-outcome-probe",
                ),
            )
        )
    return context, CandidateFragmentManifest(
        candidate_id=f"vectorized-probe-cluster{cluster_id}",
        context_hash=context.digest,
        fragments=tuple(fragments),
        planned_descriptor=(
            sum(
                sum(math.dist(left, right) for left, right in zip(path[:-1], path[1:], strict=True))
                for path in transit_paths_m
            )
            / len(transit_paths_m),
            1.0,
            0.0,
        ),
        feasible=True,
        quality_hint=0.0,
        cost_hint=sum(
            sum(math.dist(left, right) for left, right in zip(path[:-1], path[1:], strict=True))
            for path in transit_paths_m
        ),
        source="vectorized-outcome-probe-v1",
        admission_reasons=(),
    )


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
        "--communication-contract",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_p07_communication_contract.json",
    )
    parser.add_argument(
        "--start-position",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Legacy shared four-UAV start set. Multi-cluster use requires the explicit "
            "isolation-probe flag; training should use --cluster-start-position."
        ),
    )
    parser.add_argument(
        "--cluster-start-position",
        action="append",
        nargs=4,
        type=float,
        metavar=("CLUSTER", "X", "Y", "Z"),
        help="Per-cluster UAV start; provide four rows for every cluster.",
    )
    parser.add_argument("--cluster-count", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--transit-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--calibration-timeout-probe-s",
        type=float,
        default=None,
        help=(
            "Calibration-only physical cutoff below --duration-s. The normal token and "
            "planning budget remain unchanged; the resulting outcome cannot enter QD/RL."
        ),
    )
    parser.add_argument(
        "--transit-delta-m",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.12),
        metavar=("DX", "DY", "DZ"),
        help="Shared local-frame transit displacement for every probe vehicle.",
    )
    parser.add_argument(
        "--agent-transit-delta-m",
        action="append",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="Per-agent displacement; provide exactly four rows to override the shared delta.",
    )
    parser.add_argument(
        "--agent-intermediate-waypoint-m",
        action="append",
        nargs=4,
        type=float,
        metavar=("AGENT", "X", "Y", "Z"),
        help=(
            "Optional local-frame waypoint inserted before one UAV's endpoint. Repeat in "
            "path order to create an exact real multi-segment calibration route."
        ),
    )
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
    parser.add_argument("--environment-spacing-m", type=float, default=80.0)
    parser.add_argument("--scene-id", default="00626-XiJhRLvpKpX")
    parser.add_argument("--base-seed", type=int, default=20260804)
    parser.add_argument(
        "--allow-identical-cluster-starts-for-isolation-probe",
        action="store_true",
    )
    parser.add_argument("--output", required=True, type=Path)
    if argv is None:
        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def main(args: argparse.Namespace, simulation_app: Any) -> int:
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
    if args.start_position and args.cluster_start_position:
        raise ValueError(
            "use either shared --start-position or per-cluster --cluster-start-position"
        )
    if args.cluster_start_position:
        grouped_starts: list[list[tuple[float, float, float]]] = [
            [] for _ in range(args.cluster_count)
        ]
        for raw_cluster_id, x, y, z in args.cluster_start_position:
            cluster_id = int(raw_cluster_id)
            if float(cluster_id) != raw_cluster_id or not 0 <= cluster_id < args.cluster_count:
                raise ValueError("cluster start row has an invalid cluster id")
            grouped_starts[cluster_id].append(_point([x, y, z]))
        cluster_starts = validate_cluster_start_sets(grouped_starts)
    else:
        if not args.start_position:
            raise ValueError("probe requires shared or per-cluster start positions")
        shared_starts = tuple(_point(row) for row in args.start_position)
        cluster_starts = validate_cluster_start_sets(
            tuple(shared_starts for _ in range(args.cluster_count)),
            allow_identical_for_isolation_probe=(
                args.allow_identical_cluster_starts_for_isolation_probe or args.cluster_count == 1
            ),
        )
    shared_transit_delta_m = _point(list(args.transit_delta_m))
    starts = cluster_starts[0]
    if args.agent_transit_delta_m is None:
        transit_deltas_m = tuple(shared_transit_delta_m for _ in starts)
    else:
        transit_deltas_m = tuple(_point(row) for row in args.agent_transit_delta_m)
        if len(transit_deltas_m) != FORMAL_FLEET_SIZE:
            raise ValueError(f"agent transit deltas require exactly {FORMAL_FLEET_SIZE} rows")
    if any(math.dist((0.0, 0.0, 0.0), row) <= 1.0e-9 for row in transit_deltas_m):
        raise ValueError("every transit delta must be non-zero")
    intermediate_waypoints: list[list[tuple[float, float, float]]] = [
        [] for _ in range(FORMAL_FLEET_SIZE)
    ]
    for raw_agent_index, x, y, z in args.agent_intermediate_waypoint_m or ():
        agent_index = int(raw_agent_index)
        if float(agent_index) != raw_agent_index or not 0 <= agent_index < FORMAL_FLEET_SIZE:
            raise ValueError("intermediate waypoint has an invalid UAV index")
        intermediate_waypoints[agent_index].append(_point([x, y, z]))
    if args.cluster_count < 1:
        raise ValueError("cluster_count must be positive")
    if not 0.0 < args.transit_duration_s < args.duration_s:
        raise ValueError("transit duration must lie inside the probe horizon")
    if args.calibration_timeout_probe_s is not None and not (
        0.0 < args.calibration_timeout_probe_s < args.duration_s
    ):
        raise ValueError("calibration timeout probe must be positive and below the action budget")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite vectorized outcome evidence: {output}")
    contract = HM3DCommunicationContract.from_path(args.communication_contract)

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
    robot = Multirotor(
        cf2x._multirotor_cfg(cf2x_usd, args.physics_dt_s).replace(
            prim_path="/World/envs/env_.*/P07Agents/Agent_.*/Robot"
        )
    )
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
    robot.update(float(sim.cfg.dt))
    contact.update(float(sim.cfg.dt), force_recompute=True)
    expected_starts = [
        layout.to_world(cluster_id, cluster_starts[cluster_id][agent_index])
        for cluster_id in range(layout.cluster_count)
        for agent_index in range(layout.fleet_size)
    ]
    robot.reset()
    robot.write_root_pose_to_sim(
        torch.tensor(
            [[*position, 1.0, 0.0, 0.0, 0.0] for position in expected_starts],
            device=robot.device,
            dtype=torch.float32,
        )
    )
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
    contact.update(float(sim.cfg.dt), force_recompute=True)

    from aerocity_method.runtime.physx_query_cache import MemoizedRaycastClosestQuery

    scene_query = MemoizedRaycastClosestQuery(omni.physx.get_physx_scene_query_interface())
    collision_mesh = cf2x._load_collision_triangle_mesh(collision_usd)
    clearance_oracle = cf2x._EvaluatorStaticClearance(None, collision_mesh)
    backend = IsaacCF2XVectorizedExecutionBackend(
        sim=sim,
        robot=robot,
        contact=contact,
        scene_query=scene_query,
        static_clearance_oracle=clearance_oracle,
        layout=layout,
        bounds_min_m=(-1.0e6, -1.0e6, -1.0e6),
        bounds_max_m=(1.0e6, 1.0e6, 1.0e6),
        arrival_tolerance_m=0.10,
        communication_max_range_m=float(contract.network["maximum_range_m"]),
        communication_base_latency_s=float(contract.network["base_latency_s"]),
        communication_per_hop_latency_s=float(contract.network["per_hop_latency_s"]),
        communication_loss_probability=float(contract.network["loss_probability"]),
        communication_update_hz=float(contract.network["telemetry_update_hz"]),
        communication_message_ttl_s=float(contract.message_policy["time_to_live_s"]),
        calibration_timeout_probe_s=args.calibration_timeout_probe_s,
        controller_id=args.controller_id,
    )

    def _cluster_transit_paths(
        cluster_start_set: tuple[tuple[float, float, float], ...],
    ) -> tuple[tuple[tuple[float, float, float], ...], ...]:
        paths: list[tuple[tuple[float, float, float], ...]] = []
        for index, (start, delta_m) in enumerate(
            zip(cluster_start_set, transit_deltas_m, strict=True)
        ):
            endpoint = tuple(start[axis] + delta_m[axis] for axis in range(3))
            path_m = (start, *intermediate_waypoints[index], endpoint)
            if any(
                math.dist(left, right) <= 1.0e-9
                for left, right in zip(path_m[:-1], path_m[1:], strict=True)
            ):
                raise ValueError("transit probe path contains a zero-length segment")
            paths.append(path_m)
        return tuple(paths)

    contexts_and_manifests = tuple(
        _manifest(
            cluster_id=cluster_id,
            transit_paths_m=_cluster_transit_paths(cluster_starts[cluster_id]),
            transit_end_s=args.transit_duration_s,
            horizon_s=args.duration_s,
        )
        for cluster_id in range(layout.cluster_count)
    )
    manifests = tuple(row[1] for row in contexts_and_manifests)
    tokens = tuple(
        authorize_manifest(
            context_row,
            (manifest,),
            (True,),
            0,
            token_id=f"vector-probe-token-{cluster_id}-{uuid.uuid4().hex}",
            issued_at=0.0,
            duration=args.duration_s,
        )
        for cluster_id, (context_row, manifest) in enumerate(contexts_and_manifests)
    )
    batch_results = backend.execute_manifests(manifests, tokens)
    shared_wall_clock_timing = batch_results[0].engineering_diagnostics.get(
        "shared_wall_clock_timing_s"
    )
    if not isinstance(shared_wall_clock_timing, dict):
        raise RuntimeError("vectorized backend omitted shared wall-clock timing")
    if any(
        result.engineering_diagnostics.get("shared_wall_clock_timing_s") != shared_wall_clock_timing
        for result in batch_results[1:]
    ):
        raise RuntimeError("vectorized clusters disagree on shared wall-clock timing")
    cluster_rows: list[dict[str, object]] = []
    for cluster_id, (manifest, token, result) in enumerate(
        zip(manifests, tokens, batch_results, strict=True)
    ):
        ledger = execute_hm3d_manifest(
            manifest,
            token,
            result.precomputed_backend(),
            time_tolerance_s=0.25,
            command_path_tolerance_m=0.25,
        )
        communication = result.engineering_diagnostics["communication"]
        delivery = result.engineering_diagnostics["message_delivery"]
        assert isinstance(communication, dict) and isinstance(delivery, dict)
        audit = contract.audit_worker_evidence(communication, delivery)
        if audit["passed"] is not True:
            raise RuntimeError(f"cluster {cluster_id} communication evidence failed")
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "manifest_hash": manifest.manifest_hash,
                "token_hash": token.digest,
                "execution": ledger.to_public_dict(),
                "communication_audit": audit,
                "public_range_frame_count": len(result.public_range_frames),
                "public_range_ray_count": len(result.public_range_outcomes),
                "public_map_sender_ids": list(result.public_map_sender_ids),
                "final_root_positions_m": result.final_root_positions_m,
                "controller_tracking": result.engineering_diagnostics.get("controller_tracking"),
                "team_trajectory_diversity": result.engineering_diagnostics.get(
                    "team_trajectory_diversity"
                ),
                "shared_wall_clock_timing_s": shared_wall_clock_timing,
                "agent_execution_calibration": result.engineering_diagnostics.get("agents"),
                "execution_calibration": {
                    key: result.engineering_diagnostics[key]
                    for key in (
                        "backend_id",
                        "evidence_class",
                        "token_authorization_duration_s",
                        "execution_deadline_s",
                        "calibration_only_timeout_probe",
                        "controller_tracking",
                        "team_trajectory_diversity",
                        "static_trace_clearance",
                        "agents",
                    )
                },
                "result_sha256": canonical_sha256(
                    {
                        "samples": [sample.actual_path_hash for sample in result.samples],
                        "frames": [row.to_dict() for row in result.public_range_frames],
                        "rays": [row.to_dict() for row in result.public_range_outcomes],
                    }
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "VECTORIZED_OUTCOME_PROBE_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "claim_limit": (
            "Engineering proof of lockstep candidate-to-PhysX-to-outcome isolation only; "
            "clusters are not independent HM3D scenes for statistical reporting."
        ),
        "cluster_count": layout.cluster_count,
        "cluster_start_positions_m": [
            [list(point) for point in start_set] for start_set in cluster_starts
        ],
        "cluster_seeds": [
            cluster_seed(
                scene_id=args.scene_id,
                cluster_id=cluster_id,
                episode_id=f"vectorized-probe-cluster{cluster_id}",
                base_seed=args.base_seed,
            )
            for cluster_id in range(layout.cluster_count)
        ],
        "identical_start_sets_allowed_for_isolation_probe": bool(
            args.allow_identical_cluster_starts_for_isolation_probe
        ),
        "fleet_size_per_cluster": layout.fleet_size,
        "fleet_size": layout.fleet_size,
        "single_physx_step_loop": True,
        "collision_filter_applied": True,
        "transit_deltas_m": transit_deltas_m,
        "transit_intermediate_waypoints_m": intermediate_waypoints,
        "action_budget_s": args.duration_s,
        "execution_deadline_s": (args.calibration_timeout_probe_s or args.duration_s),
        "calibration_only_timeout_probe": (args.calibration_timeout_probe_s is not None),
        "physics_dt_s": args.physics_dt_s,
        "arrival_tolerance_m": 0.10,
        "outcome_time_tolerance_s": 0.25,
        "shared_backend_wall_clock_timing_s": shared_wall_clock_timing,
        "probe_process_wall_elapsed_before_write_s": (
            time.perf_counter() - PROBE_PROCESS_WALL_STARTED
        ),
        "probe_process_wall_timing_claim_limit": (
            "Includes script import, Isaac application startup, stage construction, asset "
            "loading, execution, outcome assembly, and payload construction up to JSON write."
        ),
        "cross_cluster_message_count": 0,
        "cross_cluster_map_delta_count": 0,
        "collision_usd_sha256": _sha256(collision_usd),
        "cf2x_usd_sha256": _sha256(cf2x_usd),
        "communication_contract_sha256": contract.digest,
        "clusters": cluster_rows,
    }
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    write_json_atomic(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}))
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
