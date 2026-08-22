"""Run a controlled real-PhysX CF2X collision replay against HM3D geometry.

The probe uses the static triangle-mesh collider produced by
``convert_hm3d_glb_to_collision_usd.py``.  It selects a collision direction
through a real PhysX ray query, then records contact-sensor force while an
actual CF2X articulation moves into that specific mesh.  This validates
collision wiring only; a deliberate impact is never a policy or safety score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
DRONE_USD = ROOT.parents[1] / "assets" / "new" / "cf2x.usd"
AXIS_DIRECTIONS = (
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _as_float_list(value: Any) -> list[float]:
    return [float(component) for component in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--collision-derivative-manifest", type=Path)
    parser.add_argument("--cf2x-usd", type=Path, default=DRONE_USD)
    parser.add_argument("--origin", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--minimum-ray-distance-m", type=float, default=0.50)
    parser.add_argument("--maximum-ray-distance-m", type=float, default=5.0)
    parser.add_argument("--speed-mps", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS)
SIMULATION_APP = APP.app


def _load_conversion(
    path: Path,
    scene_id: str,
    collision_usd: Path,
    collision_derivative_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("scene_id") != scene_id:
        raise ValueError("collision manifest does not match the requested scene")
    original_usd = Path(payload.get("output_usd", "")).resolve()
    original_sha256 = payload.get("output_usd_sha256")
    derivative_provenance = None
    if original_usd == collision_usd:
        if original_sha256 != _sha256(collision_usd):
            raise ValueError("collision USD changed after collision conversion")
    else:
        if collision_derivative_manifest is None:
            raise ValueError("collision manifest USD path does not match the replay input")
        derivative = json.loads(collision_derivative_manifest.read_text(encoding="utf-8"))
        if not isinstance(derivative, dict):
            raise ValueError("collision derivative manifest must be an object")
        if Path(derivative.get("source_collision_usd", "")).resolve() != original_usd:
            raise ValueError("collision derivative source USD path mismatch")
        if derivative.get("source_collision_usd_sha256") != original_sha256:
            raise ValueError("collision derivative source USD hash mismatch")
        if Path(derivative.get("output_usd", "")).resolve() != collision_usd:
            raise ValueError("collision derivative output USD path mismatch")
        if derivative.get("output_usd_sha256") != _sha256(collision_usd):
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
    collision = payload.get("collision", {})
    if collision.get("runtime_up_axis") != "Z" or collision.get("meters_per_unit") != 1.0:
        raise ValueError("collision candidate is not Z-up metres")
    mesh_rows = collision.get("meshes", [])
    if not mesh_rows or not all(
        row.get("collision_enabled") is True and row.get("physx_triangle_mesh_collision") is True
        for row in mesh_rows
    ):
        raise ValueError("collision candidate lacks verified static triangle-mesh collider")
    return payload, derivative_provenance


def _select_hm3d_raycast(
    *,
    interface: Any,
    origin: tuple[float, float, float],
    minimum_distance_m: float,
    maximum_distance_m: float,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    candidates: list[tuple[float, tuple[float, float, float], dict[str, Any]]] = []
    for direction in AXIS_DIRECTIONS:
        hit = interface.raycast_closest(origin, direction, maximum_distance_m)
        if not hit.get("hit"):
            continue
        distance = float(hit["distance"])
        rigid_body = str(hit.get("rigidBody", ""))
        if distance >= minimum_distance_m and rigid_body.startswith("/World/HM3D/"):
            candidates.append((distance, direction, hit))
    if not candidates:
        raise RuntimeError(
            "no HM3D triangle-mesh ray hit met the requested distance bounds; "
            "choose another independently ESDF-audited origin"
        )
    _, direction, hit = min(candidates, key=lambda row: row[0])
    return direction, hit


def main() -> int:
    # Omniverse modules must only be imported after the app is running.
    import isaaclab.sim as sim_utils
    import omni.physx
    import omni.usd
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import UsdGeom

    scene_usd = ARGS.collision_usd.expanduser().resolve()
    manifest_path = ARGS.collision_manifest.expanduser().resolve()
    drone_usd = ARGS.cf2x_usd.expanduser().resolve()
    output = ARGS.output.expanduser().resolve()
    for path, label in (
        (scene_usd, "collision USD"),
        (manifest_path, "collision manifest"),
        (drone_usd, "CF2X USD"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite collision replay evidence: {output}")
    if (
        ARGS.minimum_ray_distance_m <= 0.0
        or ARGS.maximum_ray_distance_m <= ARGS.minimum_ray_distance_m
        or ARGS.speed_mps <= 0.0
        or ARGS.max_steps < 1
    ):
        raise ValueError(
            "collision replay bounds, speed, and step count must be positive and ordered"
        )
    derivative_manifest = None
    if ARGS.collision_derivative_manifest is not None:
        derivative_manifest = ARGS.collision_derivative_manifest.expanduser().resolve()
        if not derivative_manifest.is_file():
            raise FileNotFoundError(derivative_manifest)
    conversion, derivative_provenance = _load_conversion(
        manifest_path,
        ARGS.scene_id,
        scene_usd,
        collision_derivative_manifest=derivative_manifest,
    )
    origin = tuple(float(value) for value in ARGS.origin)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene_root = UsdGeom.Xform.Define(stage, "/World/HM3D")
    scene_root.GetPrim().GetReferences().AddReference(str(scene_usd))
    for _ in range(8):
        SIMULATION_APP.update()

    sim = SimulationContext(
        SimulationCfg(
            dt=1.0 / 120.0,
            device=ARGS.device,
            enable_scene_query_support=True,
        )
    )
    sim.reset()
    interface = omni.physx.get_physx_scene_query_interface()
    direction, ray_hit = _select_hm3d_raycast(
        interface=interface,
        origin=origin,
        minimum_distance_m=ARGS.minimum_ray_distance_m,
        maximum_distance_m=ARGS.maximum_ray_distance_m,
    )

    drone_cfg = ArticulationCfg(
        prim_path="/World/CF2X",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(drone_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # This is a deterministic collider probe, not a flight rollout.
                # Gravity is disabled only for the direct, horizontal impact.
                disable_gravity=True,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=origin),
        actuators={
            "passive_collision_probe": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0
            )
        },
    )
    robot = Articulation(drone_cfg)
    contact = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/CF2X/.*",
            track_pose=False,
            track_air_time=True,
            force_threshold=0.01,
            history_length=1,
            debug_vis=False,
        )
    )
    sim.reset()
    robot.update(sim.cfg.dt)
    pose = torch.tensor([[*origin, 1.0, 0.0, 0.0, 0.0]], device=robot.device)
    velocity = torch.tensor(
        [
            [
                direction[0] * ARGS.speed_mps,
                direction[1] * ARGS.speed_mps,
                direction[2] * ARGS.speed_mps,
                0.0,
                0.0,
                0.0,
            ]
        ],
        device=robot.device,
    )
    robot.reset()
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(velocity)
    robot.write_data_to_sim()
    sim.forward()

    maximum_force_n = 0.0
    first_contact_step: int | None = None
    trace: list[dict[str, Any]] = []
    for step in range(ARGS.max_steps):
        sim.step(render=False)
        robot.update(sim.cfg.dt)
        contact.update(sim.cfg.dt, force_recompute=True)
        forces = contact.data.net_forces_w
        maximum_force_n = max(
            maximum_force_n,
            float(torch.linalg.norm(forces, dim=-1).max().item()),
        )
        position = robot.data.root_pos_w.detach().cpu().tolist()[0]
        velocity_row = robot.data.root_lin_vel_w.detach().cpu().tolist()[0]
        if step % 12 == 0:
            trace.append(
                {
                    "step": step,
                    "position_m": _as_float_list(position),
                    "linear_velocity_mps": _as_float_list(velocity_row),
                    "maximum_contact_force_n": maximum_force_n,
                }
            )
        if maximum_force_n >= 0.01 and first_contact_step is None:
            first_contact_step = step
            break

    hit_path = str(ray_hit.get("rigidBody", ""))
    passed = bool(first_contact_step is not None and hit_path.startswith("/World/HM3D/"))
    payload = {
        "schema_version": "hm3d-cf2x-physx-collision-replay-v1",
        "status": "COLLISION_REPLAY_PASSED" if passed else "COLLISION_REPLAY_FAILED",
        "measured": True,
        "synthetic": False,
        "scene_id": ARGS.scene_id,
        "collision_usd": str(scene_usd),
        "collision_usd_sha256": _sha256(scene_usd),
        "source_glb_sha256": conversion["source_glb_sha256"],
        "cf2x_usd": str(drone_usd),
        "cf2x_usd_sha256": _sha256(drone_usd),
        "origin_m": list(origin),
        "ray_query": {
            "minimum_distance_m": ARGS.minimum_ray_distance_m,
            "maximum_distance_m": ARGS.maximum_ray_distance_m,
            "direction": list(direction),
            "hit_distance_m": float(ray_hit["distance"]),
            "hit_position_m": _as_float_list(ray_hit["position"]),
            "hit_normal": _as_float_list(ray_hit["normal"]),
            "hit_rigid_body": hit_path,
        },
        "replay": {
            "physics_dt_s": float(sim.cfg.dt),
            "gravity_disabled_for_probe": True,
            "initial_speed_mps": ARGS.speed_mps,
            "max_steps": ARGS.max_steps,
            "first_contact_step": first_contact_step,
            "maximum_contact_force_n": maximum_force_n,
            "trace": trace,
        },
        "passed": passed,
        "formal_runtime_admission": False,
        "caveat": (
            "This deliberate, gravity-disabled collision replay verifies PhysX collider "
            "interaction only. "
            "It is not a safe-flight result, a controller validation, or P03 completion."
        ),
    }
    if derivative_provenance is not None:
        payload["collision_derivative_provenance"] = derivative_provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "ray_distance_m": payload["ray_query"]["hit_distance_m"],
                "first_contact_step": first_contact_step,
                "maximum_force_n": maximum_force_n,
            }
        )
    )
    return 0 if passed else 2


try:
    raise SystemExit(main())
finally:
    SIMULATION_APP.close()
