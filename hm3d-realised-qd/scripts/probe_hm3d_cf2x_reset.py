"""Record one real Isaac Sim reset witness for a collision-admitted HM3D USD.

The output is deliberately a *single-scene probe*.  Three independently run
probes (A, B and A again) are required before constructing A-B-A reset
evidence.  This prevents a source-file hash from being presented as a physics
reset result.
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _round_rows(value: Any, digits: int = 7) -> Any:
    if isinstance(value, list):
        return [_round_rows(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--collision-manifest", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, default=DRONE_USD)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, default=12031)
    parser.add_argument("--spawn-x", type=float, required=True)
    parser.add_argument("--spawn-y", type=float, required=True)
    parser.add_argument("--spawn-z", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS)
SIMULATION_APP = APP.app


def _load_collision_manifest(path: Path, scene_id: str, collision_usd: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "scene_id",
        "source_glb_sha256",
        "output_usd",
        "output_usd_sha256",
        "coordinate_transform_sha256",
        "collision",
        "status",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("collision manifest lacks required conversion evidence")
    if payload["scene_id"] != scene_id:
        raise ValueError("collision manifest scene_id does not match reset probe")
    if Path(payload["output_usd"]).resolve() != collision_usd:
        raise ValueError("collision manifest USD path does not match reset probe")
    if payload["output_usd_sha256"] != _sha256(collision_usd):
        raise ValueError("collision USD changed after conversion manifest was written")
    collision = payload["collision"]
    if (
        collision.get("runtime_up_axis") != "Z"
        or collision.get("meters_per_unit") != 1.0
        or collision.get("mesh_count", 0) < 1
    ):
        raise ValueError("collision conversion is not a Z-up metre-scale mesh")
    if not all(
        row.get("collision_enabled") is True and row.get("physx_triangle_mesh_collision") is True
        for row in collision.get("meshes", [])
    ):
        raise ValueError("collision conversion does not prove static triangle collision")
    return payload


def main() -> int:
    # All Omniverse imports must happen after Application creation.
    import isaaclab.sim as sim_utils
    import omni.usd
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import UsdGeom

    scene_usd = ARGS.collision_usd.expanduser().resolve()
    collision_manifest_path = ARGS.collision_manifest.expanduser().resolve()
    drone_usd = ARGS.cf2x_usd.expanduser().resolve()
    output = ARGS.output.expanduser().resolve()
    for path, label in (
        (scene_usd, "collision USD"),
        (collision_manifest_path, "collision manifest"),
        (drone_usd, "CF2X USD"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a reset witness: {output}")
    collision_manifest = _load_collision_manifest(collision_manifest_path, ARGS.scene_id, scene_usd)
    spawn = [ARGS.spawn_x, ARGS.spawn_y, ARGS.spawn_z]
    if not all(-1.0e4 < value < 1.0e4 for value in spawn):
        raise ValueError("CF2X spawn coordinates are implausible")

    torch.manual_seed(ARGS.seed)
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
    # Finish reference composition before adding the dynamic articulation.
    for _ in range(8):
        SIMULATION_APP.update()

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=ARGS.device))
    drone_cfg = ArticulationCfg(
        prim_path="/World/CF2X",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(drone_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(spawn)),
        actuators={
            "passive_reset": ImplicitActuatorCfg(
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
    reset_pose = torch.tensor([[*spawn, 1.0, 0.0, 0.0, 0.0]], device=robot.device)
    reset_velocity = torch.zeros((1, 6), device=robot.device)
    robot.reset()
    robot.write_root_pose_to_sim(reset_pose)
    robot.write_root_velocity_to_sim(reset_velocity)
    robot.write_data_to_sim()
    # Refresh Fabric/RTX without advancing physics.  Advancing one frame lets
    # gravity change the reset state, which is not a reset equivalence test.
    sim.forward()
    sim.render()
    robot.update(sim.cfg.dt)
    contact.update(sim.cfg.dt, force_recompute=True)

    observed_pos = robot.data.root_pos_w.detach().cpu().tolist()
    observed_velocity = robot.data.root_lin_vel_w.detach().cpu().tolist()
    target_pos = torch.tensor([spawn], device=robot.device)
    position_error = float(
        torch.linalg.norm(robot.data.root_pos_w - target_pos, dim=1).max().item()
    )
    speed = float(torch.linalg.norm(robot.data.root_lin_vel_w, dim=1).max().item())
    contact_forces = contact.data.net_forces_w.detach().cpu().tolist()
    contact_shape = list(contact.data.net_forces_w.shape)
    controller = {
        "controller_id": "cf2x-reset-zero-velocity-passive-joints-v1",
        "physics_advanced_during_reset": False,
        "joint_command": "passive_zero_stiffness_zero_damping",
    }
    sensor = {
        "sensor_id": "isaaclab-contact-sensor-v1",
        "prim_path": "/World/CF2X/.*",
        "force_threshold_n": 0.01,
        "history_length": 1,
        "observed_shape": contact_shape,
    }
    reset_state = {
        "spawn_position_m": spawn,
        "spawn_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "observed_root_position_m": _round_rows(observed_pos),
        "observed_root_linear_velocity_mps": _round_rows(observed_velocity),
        "position_error_m": position_error,
        "linear_speed_mps": speed,
        "position_within_tolerance": position_error <= 1.0e-4,
        "speed_within_tolerance": speed <= 1.0e-5,
    }
    components = {
        "scene": collision_manifest["source_glb_sha256"],
        "collider": collision_manifest["output_usd_sha256"],
        "contact": _canonical_sha256(sensor),
        "sensor": _canonical_sha256(sensor),
        "rng": _canonical_sha256({"torch_seed": ARGS.seed}),
        "controller": _canonical_sha256(controller),
        "reset_state": _canonical_sha256(reset_state),
    }
    reset_fingerprint = _canonical_sha256(components)
    passed = bool(
        int(robot.num_instances) == 1
        and position_error <= 1.0e-4
        and speed <= 1.0e-5
        and contact_shape
    )
    payload = {
        "schema_version": "hm3d-cf2x-real-reset-probe-v1",
        "status": "RESET_PROBE_PASSED" if passed else "RESET_PROBE_FAILED",
        "measured": True,
        "synthetic": False,
        "runtime": {
            "simulator_id": "isaac-sim-5.1+isaaclab",
            "device": str(robot.device),
            "physics_dt_s": float(sim.cfg.dt),
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "stage_up_axis": "Z",
            "meters_per_unit": 1.0,
        },
        "run_tag": ARGS.run_tag,
        "scene_id": ARGS.scene_id,
        "scene_source_glb_sha256": collision_manifest["source_glb_sha256"],
        "collision_usd": str(scene_usd),
        "collision_usd_sha256": collision_manifest["output_usd_sha256"],
        "coordinate_transform_sha256": collision_manifest["coordinate_transform_sha256"],
        "cf2x_usd": str(drone_usd),
        "cf2x_usd_sha256": _sha256(drone_usd),
        "robot": {
            "num_instances": int(robot.num_instances),
            "num_bodies": int(robot.num_bodies),
            "num_joints": int(robot.num_joints),
            "body_names": list(robot.body_names),
            "vehicle_envelope_m": [0.11, 0.11, 0.04],
        },
        "controller": controller,
        "sensor": sensor,
        "reset_state": reset_state,
        "contact_force_w_n": _round_rows(contact_forces),
        "fingerprint_components": components,
        "reset_fingerprint": reset_fingerprint,
        "passed": passed,
        "formal_runtime_admission": False,
        "caveat": (
            "A single real reset witness is development evidence only. "
            "It is not A-B-A equivalence, collision replay, or formal P02/P03 admission."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "output": str(output), "fingerprint": reset_fingerprint}
        )
    )
    return 0 if passed else 2


try:
    raise SystemExit(main())
finally:
    SIMULATION_APP.close()
