"""Run the fail-closed native Isaac L1 capability gate for one public episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cityspec", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--public-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-count", type=int, default=3)
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.step_count <= 0:
    raise ValueError("step-count must be positive")
ARGS.output.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(status: str, **details: object) -> dict[str, object]:
    return {"status": status, **details}


def _collision_paths(stage: object, collision_api: object) -> list[str]:
    paths: list[str] = []
    for prim in stage.Traverse():  # type: ignore[attr-defined]
        if prim.HasAPI(collision_api):  # type: ignore[attr-defined]
            paths.append(str(prim.GetPath()))  # type: ignore[attr-defined]
    return sorted(paths)


def _raycast(
    query: object,
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
) -> dict[str, object]:
    direction = tuple(target[i] - origin[i] for i in range(3))
    distance = math.sqrt(sum(value * value for value in direction))
    if distance <= 0.0:
        raise ValueError("raycast endpoints must differ")
    direction = tuple(value / distance for value in direction)
    hit = query.raycast_closest(origin, direction, distance)  # type: ignore[attr-defined]
    if not hit:
        return {"hit": False, "distance_m": distance}
    prim_path = next(
        (
            hit.get(key)
            for key in ("collision", "collider", "rigid_body", "rigidBody")
            if hit.get(key)
        ),
        "",
    )
    return {
        "hit": bool(hit.get("hit", True)),
        "distance_m": float(hit.get("distance", distance)),
        "prim_path": str(prim_path),
        "native_result_keys": sorted(str(key) for key in hit),
    }


def _vector(values: object) -> list[float]:
    return [float(value) for value in values]  # type: ignore[union-attr]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _distance(first: list[float], second: list[float]) -> float:
    return _norm([a - b for a, b in zip(first, second, strict=True)])


def _yaw_deg(orientation_wxyz: list[float]) -> float:
    w, x, y, z = (float(value) for value in orientation_wxyz)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _world_velocity_to_body(velocity: list[float], yaw_deg: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * float(velocity[0]) + sine * float(velocity[1]),
        -sine * float(velocity[0]) + cosine * float(velocity[1]),
        float(velocity[2]),
    )


def _camera_fov_check(cameras: dict[str, object], contract: dict[str, Any]) -> dict[str, object]:
    from pxr import Gf, UsdGeom

    observe = contract["observe"]
    sensor = contract["sensor_rig"]
    expected_translation = [float(value) for value in sensor["translation_body_m"]]
    records: dict[str, dict[str, object]] = {}
    maximum_fov_error = 0.0
    maximum_translation_error = 0.0
    maximum_axis_error = 0.0
    for drone_id, raw_camera in sorted(cameras.items()):
        camera = UsdGeom.Camera(raw_camera.GetPrim())  # type: ignore[attr-defined]
        focal = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        horizontal_fov = math.degrees(2.0 * math.atan(horizontal_aperture / (2.0 * focal)))
        vertical_fov = math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * focal)))
        fov_error = max(
            abs(horizontal_fov - float(observe["horizontal_fov_deg"])),
            abs(vertical_fov - float(observe["vertical_fov_deg"])),
        )
        local = UsdGeom.Xformable(camera.GetPrim()).GetLocalTransformation()
        translation = _vector(local.Transform(Gf.Vec3d(0.0, 0.0, 0.0)))
        forward = _vector(local.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized())
        up = _vector(local.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized())
        translation_error = _distance(translation, expected_translation)
        axis_error = max(_distance(forward, [1.0, 0.0, 0.0]), _distance(up, [0.0, 0.0, 1.0]))
        maximum_fov_error = max(maximum_fov_error, fov_error)
        maximum_translation_error = max(maximum_translation_error, translation_error)
        maximum_axis_error = max(maximum_axis_error, axis_error)
        records[drone_id] = {
            "horizontal_fov_deg": horizontal_fov,
            "vertical_fov_deg": vertical_fov,
            "translation_body_m": translation,
            "forward_axis_body": forward,
            "up_axis_body": up,
        }
    passed = (
        len(records) == 4
        and maximum_fov_error <= 1.0e-4
        and maximum_translation_error <= 1.0e-6
        and maximum_axis_error <= 1.0e-6
        and sensor["forward_axis"] == "+X"
        and sensor["up_axis"] == "+Z"
        and sensor["gimbal_mode"] == "fixed"
    )
    return _check(
        "PASS" if passed else "FAIL",
        method="authored_usd_camera_intrinsics_and_fixed_body_extrinsics",
        camera_count=len(records),
        maximum_fov_error_deg=maximum_fov_error,
        maximum_translation_error_m=maximum_translation_error,
        maximum_axis_vector_error=maximum_axis_error,
        cameras=records,
        rgb_required=False,
    )


def _create_dynamic_rig(
    stage: object,
    episode: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[object, dict[str, object], dict[str, object]]:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from pxr import Gf, UsdGeom

    physics_dt = 1.0 / 60.0
    World.clear_instance()
    world = World(
        physics_dt=physics_dt,
        rendering_dt=physics_dt,
        stage_units_in_meters=1.0,
    )
    world.get_physics_context().set_gravity(0.0)
    radius = float(contract["vehicle"]["radius_m"])
    cameras: dict[str, object] = {}
    drones: dict[str, object] = {}
    focal_length_mm = 20.0
    horizontal_fov = math.radians(float(contract["observe"]["horizontal_fov_deg"]))
    vertical_fov = math.radians(float(contract["observe"]["vertical_fov_deg"]))
    translation = contract["sensor_rig"]["translation_body_m"]
    camera_orientation = Gf.Quatf(0.5, Gf.Vec3f(0.5, -0.5, -0.5))
    colors = (
        (0.85, 0.20, 0.18),
        (0.12, 0.55, 0.86),
        (0.18, 0.72, 0.32),
        (0.88, 0.63, 0.12),
    )
    for index, start in enumerate(sorted(episode["starts"], key=lambda item: item["drone_id"])):
        drone_id = str(start["drone_id"])
        prim_name = drone_id.replace("-", "_")
        path = f"/World/AeroCityNativeGate/{prim_name}"
        yaw = math.radians(float(start["yaw_deg"]))
        orientation = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        position = np.array(start["position"], dtype=np.float64)
        drone = world.scene.add(
            DynamicCuboid(
                prim_path=path,
                name=f"native_gate_{prim_name}",
                position=position,
                orientation=orientation,
                size=2.0 * radius,
                color=np.array(colors[index], dtype=np.float64),
                mass=1.5,
            )
        )
        drone.set_default_state(
            position=position,
            orientation=orientation,
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        camera = UsdGeom.Camera.Define(stage, f"{path}/FixedCamera")
        camera.CreateFocalLengthAttr(focal_length_mm)
        camera.CreateHorizontalApertureAttr(
            2.0 * focal_length_mm * math.tan(horizontal_fov / 2.0)
        )
        camera.CreateVerticalApertureAttr(
            2.0 * focal_length_mm * math.tan(vertical_fov / 2.0)
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in translation]))
        camera_xform.AddOrientOp().Set(camera_orientation)
        drones[drone_id] = drone
        cameras[drone_id] = camera
    world.reset()
    return world, drones, cameras


def _drone_state(drone: object) -> dict[str, list[float]]:
    position, orientation = drone.get_world_pose()  # type: ignore[attr-defined]
    return {
        "position": _vector(position),
        "orientation_wxyz": _vector(orientation),
        "linear_velocity_mps": _vector(drone.get_linear_velocity()),  # type: ignore[attr-defined]
        "angular_velocity_rad_s": _vector(drone.get_angular_velocity()),  # type: ignore[attr-defined]
    }


def _run_transcript(
    world: object,
    drones: dict[str, object],
    transcript: list[dict[str, Any]],
    control_period_s: float,
) -> list[dict[str, Any]]:
    import numpy as np

    physics_dt = 1.0 / 60.0
    physics_steps = round(control_period_s / physics_dt)
    if not math.isclose(physics_steps * physics_dt, control_period_s, abs_tol=1.0e-9):
        raise ValueError("control period must be an integer multiple of the native physics step")
    samples: list[dict[str, Any]] = []
    task_time_s = 0.0
    for step in transcript:
        before = {drone_id: _drone_state(drone) for drone_id, drone in drones.items()}
        for drone_id, drone in sorted(drones.items()):
            command = step["commands"][drone_id]
            drone.set_linear_velocity(  # type: ignore[attr-defined]
                np.array(command["linear_velocity_world_mps"], dtype=np.float64)
            )
            drone.set_angular_velocity(  # type: ignore[attr-defined]
                np.array(
                    [0.0, 0.0, math.radians(float(command["yaw_rate_deg_s"]))],
                    dtype=np.float64,
                )
            )
        for _ in range(physics_steps):
            world.step(render=False, update_fabric=True)  # type: ignore[attr-defined]
        task_time_s += control_period_s
        for drone_id, drone in sorted(drones.items()):
            state = _drone_state(drone)
            average_velocity = [
                (end - start) / control_period_s
                for start, end in zip(
                    before[drone_id]["position"], state["position"], strict=True
                )
            ]
            linear_speed = _norm(state["linear_velocity_mps"])
            angular_speed_deg_s = math.degrees(_norm(state["angular_velocity_rad_s"]))
            samples.append(
                {
                    "command_index": int(step["index"]),
                    "phase": str(step["phase"]),
                    "observe_case": step["observe_case"],
                    "drone_id": drone_id,
                    "task_time_s": task_time_s,
                    "position_before": before[drone_id]["position"],
                    "state_before": before[drone_id],
                    **state,
                    "average_velocity_mps": average_velocity,
                    "linear_speed_mps": linear_speed,
                    "angular_speed_deg_s": angular_speed_deg_s,
                    "commanded_linear_velocity_mps": step["commands"][drone_id][
                        "linear_velocity_world_mps"
                    ],
                    "commanded_yaw_rate_deg_s": step["commands"][drone_id][
                        "yaw_rate_deg_s"
                    ],
                }
            )
    return samples


def _build_capability_receipts(
    samples: list[dict[str, Any]],
    *,
    episode_id: str,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compile measured native samples into non-formal L1 capability receipts."""

    from aerocity_bench.contracts import ActionPacket, ObservationPacket, Pose3D
    from aerocity_bench.isaac_bridge import build_l1_execution_receipt

    period = float(contract["control_period_s"])
    energy_budget = float(contract["vehicle"]["energy_budget_j"])
    energy_per_meter = float(contract["vehicle"]["energy_per_meter_j"])
    hover_power = float(contract["vehicle"]["hover_power_w"])
    previous_hash: dict[str, str | None] = {}
    cumulative_energy: dict[str, float] = {}
    receipts: list[dict[str, Any]] = []
    ordered_samples = sorted(
        samples, key=lambda item: (int(item["command_index"]), str(item["drone_id"]))
    )
    for sample in ordered_samples:
        drone_id = str(sample["drone_id"])
        command = sample["commanded_linear_velocity_mps"]
        command_speed = _norm([float(value) for value in command])
        phase = str(sample["phase"])
        observe_case = sample.get("observe_case")
        observation_id = f"native-gate-observation-{drone_id}-{int(sample['command_index']):03d}"
        timestamp = float(sample["task_time_s"]) - period
        before = dict(sample["state_before"])
        yaw_deg = _yaw_deg(before["orientation_wxyz"])
        after = {
            key: sample[key]
            for key in (
                "position",
                "orientation_wxyz",
                "linear_velocity_mps",
                "angular_velocity_rad_s",
            )
        }
        observation = ObservationPacket(
            episode_id=episode_id,
            observation_id=observation_id,
            drone_id=drone_id,
            sequence=int(sample["command_index"]),
            timestamp_s=timestamp,
            pose=Pose3D(position=tuple(before["position"]), yaw_deg=yaw_deg),
            linear_velocity_world_mps=tuple(before["linear_velocity_mps"]),
            angular_speed_deg_s=math.degrees(_norm(before["angular_velocity_rad_s"])),
            energy_remaining_j=max(0.0, energy_budget - cumulative_energy.get(drone_id, 0.0)),
        )
        if observe_case == "positive" and phase == "observe_stable":
            kind = "OBSERVE"
            velocity = None
            source_observation_id = observation_id
        elif command_speed > 1.0e-12 or abs(float(sample["commanded_yaw_rate_deg_s"])) > 1.0e-12:
            kind = "VELOCITY"
            velocity = _world_velocity_to_body(command, yaw_deg)
            source_observation_id = None
        else:
            kind = "HOVER"
            velocity = None
            source_observation_id = None
        action = ActionPacket(
            episode_id=episode_id,
            drone_id=drone_id,
            sequence=int(sample["command_index"]),
            issued_at_s=timestamp,
            kind=kind,
            velocity_body_mps=velocity,
            yaw_rate_deg_s=float(sample["commanded_yaw_rate_deg_s"]),
            source_observation_id=source_observation_id,
        )
        distance_m = _distance(before["position"], after["position"])
        energy_used = distance_m * energy_per_meter + hover_power * period
        cumulative_energy[drone_id] = cumulative_energy.get(drone_id, 0.0) + energy_used
        receipt = build_l1_execution_receipt(
            action=action,
            source_observation=observation,
            state_before=before,
            state_after=after,
            task_time_start_s=timestamp,
            task_time_end_s=float(sample["task_time_s"]),
            planning_latency_s=0.0,
            action_executed=kind,
            status="capability_probe_measured",
            energy_used_j=energy_used,
            minimum_clearance_m=None,
            collision=False,
            out_of_bounds=False,
            safety_intervention=False,
            deadline_miss=False,
            previous_receipt_hash=previous_hash.get(drone_id),
        ).to_dict()
        previous_hash[drone_id] = str(receipt["receipt_hash"])
        receipts.append(receipt)
    return receipts


def _velocity_check(samples: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, object]:
    period = float(contract["control_period_s"])
    vehicle = contract["vehicle"]
    maximum_velocity_error = 0.0
    maximum_average_error = 0.0
    maximum_acceleration = 0.0
    maximum_horizontal_speed = 0.0
    maximum_vertical_speed = 0.0
    maximum_yaw_error = 0.0
    maximum_yaw_speed = 0.0
    previous: dict[str, list[float]] = {}
    for sample in samples:
        actual = sample["linear_velocity_mps"]
        command = sample["commanded_linear_velocity_mps"]
        average = sample["average_velocity_mps"]
        maximum_velocity_error = max(maximum_velocity_error, _distance(actual, command))
        maximum_average_error = max(maximum_average_error, _distance(average, command))
        drone_id = str(sample["drone_id"])
        prior = previous.get(drone_id, [0.0, 0.0, 0.0])
        maximum_acceleration = max(
            maximum_acceleration, _distance(actual, prior) / period
        )
        previous[drone_id] = actual
        maximum_horizontal_speed = max(maximum_horizontal_speed, _norm(actual[:2]))
        maximum_vertical_speed = max(maximum_vertical_speed, abs(float(actual[2])))
        maximum_yaw_speed = max(maximum_yaw_speed, float(sample["angular_speed_deg_s"]))
        maximum_yaw_error = max(
            maximum_yaw_error,
            abs(
                float(sample["angular_speed_deg_s"])
                - abs(float(sample["commanded_yaw_rate_deg_s"]))
            ),
        )
    passed = (
        maximum_velocity_error <= 0.08
        and maximum_average_error <= 0.12
        and maximum_acceleration <= float(vehicle["acceleration_mps2"]) + 0.10
        and maximum_horizontal_speed <= float(vehicle["horizontal_speed_mps"]) + 0.05
        and maximum_vertical_speed <= float(vehicle["vertical_speed_mps"]) + 0.05
        and maximum_yaw_speed <= float(vehicle["yaw_rate_deg_s"]) + 0.5
        and maximum_yaw_error <= 2.0
    )
    return _check(
        "PASS" if passed else "FAIL",
        method="four_dynamic_rigid_bodies_with_acceleration_limited_velocity_commands",
        drone_count=len({str(sample["drone_id"]) for sample in samples}),
        maximum_velocity_tracking_error_mps=maximum_velocity_error,
        maximum_interval_average_error_mps=maximum_average_error,
        maximum_measured_acceleration_mps2=maximum_acceleration,
        maximum_horizontal_speed_mps=maximum_horizontal_speed,
        maximum_vertical_speed_mps=maximum_vertical_speed,
        maximum_yaw_speed_deg_s=maximum_yaw_speed,
        maximum_yaw_tracking_error_deg_s=maximum_yaw_error,
    )


def _braking_check(
    samples: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, object]:
    from aerocity_bench.native_gate_contract import commanded_braking_distance

    period = float(contract["control_period_s"])
    expected = commanded_braking_distance(transcript, period)
    vehicle = contract["vehicle"]
    continuous_upper = (
        float(vehicle["horizontal_speed_mps"]) ** 2
        / (2.0 * float(vehicle["acceleration_mps2"]))
        + float(vehicle["horizontal_speed_mps"]) * period
    )
    observed: dict[str, float] = {}
    for drone_id in sorted({str(sample["drone_id"]) for sample in samples}):
        braking = [
            sample
            for sample in samples
            if sample["drone_id"] == drone_id and sample["phase"] == "horizontal_braking"
        ]
        observed[drone_id] = _distance(braking[0]["position_before"], braking[-1]["position"])
    maximum_error = max(abs(value - expected) for value in observed.values())
    passed = (
        len(observed) == 4
        and maximum_error <= 0.12
        and max(observed.values()) <= continuous_upper + 1.0e-9
    )
    return _check(
        "PASS" if passed else "FAIL",
        method="measured_physx_stop_distance_from_canonical_max_horizontal_speed",
        expected_discrete_distance_m=expected,
        conservative_upper_bound_m=continuous_upper,
        observed_distance_by_drone_m=observed,
        maximum_distance_error_m=maximum_error,
    )


def _dwell_check(samples: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, object]:
    from aerocity_bench.native_gate_contract import evaluate_native_dwell_samples

    by_drone = {}
    for drone_id in sorted({str(sample["drone_id"]) for sample in samples}):
        by_drone[drone_id] = evaluate_native_dwell_samples(
            [sample for sample in samples if sample["drone_id"] == drone_id],
            contract["observe"],
        )
    passed = len(by_drone) == 4 and all(item["status"] == "PASS" for item in by_drone.values())
    return _check(
        "PASS" if passed else "FAIL",
        method="measured_physx_pose_velocity_continuous_dwell_state_machine",
        drone_count=len(by_drone),
        by_drone=by_drone,
    )


def _reset_check(
    first: dict[str, dict[str, list[float]]],
    second: dict[str, dict[str, list[float]]],
    collision_paths_before: list[str],
    collision_paths_after: list[str],
) -> dict[str, object]:
    position_error = 0.0
    velocity_after_reset = 0.0
    angular_velocity_after_reset = 0.0
    for drone_id in sorted(first):
        position_error = max(
            position_error, _distance(first[drone_id]["position"], second[drone_id]["position"])
        )
        velocity_after_reset = max(
            velocity_after_reset, _norm(second[drone_id]["linear_velocity_mps"])
        )
        angular_velocity_after_reset = max(
            angular_velocity_after_reset,
            _norm(second[drone_id]["angular_velocity_rad_s"]),
        )
    passed = (
        set(first) == set(second)
        and position_error <= 1.0e-5
        and velocity_after_reset <= 1.0e-5
        and angular_velocity_after_reset <= 1.0e-5
        and collision_paths_before == collision_paths_after
    )
    return _check(
        "PASS" if passed else "FAIL",
        method="world_episode_reset_with_registered_dynamic_default_states",
        drone_count=len(second),
        maximum_reset_position_error_m=position_error,
        maximum_reset_linear_speed_mps=velocity_after_reset,
        maximum_reset_angular_speed_rad_s=angular_velocity_after_reset,
        collider_paths_stable=collision_paths_before == collision_paths_after,
    )


def _runtime_version() -> str:
    try:
        from isaacsim.core.version import get_version

        return ".".join(str(value) for value in get_version())
    except (ImportError, TypeError):
        return "loaded-version-api-unavailable"


def main() -> int:
    from pxr import Usd, UsdGeom, UsdPhysics

    from aerocity_bench.canonical import content_hash, file_hash, write_json
    from aerocity_bench.isaac_bridge import REQUIRED_NATIVE_CHECKS, write_native_gate_report
    from aerocity_bench.native_gate_contract import (
        build_native_action_transcript,
        compare_native_replays,
        load_native_gate_inputs,
        select_native_test_directions,
    )

    config, _, episode, city, input_bindings = load_native_gate_inputs(
        ARGS.release_config,
        ARGS.task_spec,
        ARGS.public_episode,
        ARGS.cityspec,
    )
    contract = config["execution_contract"]
    checks: dict[str, dict[str, object]] = {}
    context = None
    world = None
    capability_receipt_set_hash: str | None = None
    capability_receipt_list_hash: str | None = None
    try:
        import omni.physx
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(ARGS.stage.resolve())):
            raise RuntimeError("Isaac USD context rejected the stage")
        for _ in range(30):
            APP.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac USD context returned no stage")
        checks["stage_load"] = _check(
            "PASS",
            stage_path=str(ARGS.stage.resolve()),
            default_prim=str(stage.GetDefaultPrim().GetPath()),
            input_bindings=input_bindings,
        )

        base_collision_paths = _collision_paths(stage, UsdPhysics.CollisionAPI)
        expected_count = 1 + sum(
            len(building.get("components", [])) for building in city.get("buildings", [])
        ) + len(city.get("obstacles", []))
        checks["collider_count"] = _check(
            "PASS" if len(base_collision_paths) == expected_count else "FAIL",
            actual_count=len(base_collision_paths),
            expected_count=expected_count,
            collision_paths=base_collision_paths,
        )
        visual_collision_paths = [
            path
            for path in base_collision_paths
            if path.startswith("/World/VisualDecorations/")
            or path.startswith("/World/UrbanGroundDetail/")
        ]
        checks["visual_collision_isolation"] = _check(
            "PASS" if not visual_collision_paths else "FAIL",
            visual_collision_paths=visual_collision_paths,
        )

        try:
            physx = omni.physx.get_physx_interface()
            for index in range(ARGS.step_count):
                physx.update_simulation(1.0 / 60.0, (index + 1) / 60.0)
                physx.update_transformations(True, True, True, False)
            query = omni.physx.get_physx_scene_query_interface()
            collider_prim = next(
                prim
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and str(prim.GetPath()) != "/World/Ground/ground"
            )
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            bounds = cache.ComputeWorldBound(collider_prim).ComputeAlignedBox()
            minimum = bounds.GetMin()
            maximum = bounds.GetMax()
            center = tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))
            origin = (center[0] - 100.0, center[1], center[2])
            result = _raycast(query, origin, center)
            checks["ray_los_agreement"] = _check(
                "PASS" if result["hit"] and result["prim_path"] else "FAIL",
                blocked_ray=result,
                expected="hit_non_ground_collider",
            )
        except Exception as exc:  # noqa: BLE001
            checks["ray_los_agreement"] = _check(
                "FAIL", error_type=type(exc).__name__, error=str(exc)[-2000:]
            )

        horizontal_limit = float(contract["vehicle"]["horizontal_speed_mps"])
        acceleration = float(contract["vehicle"]["acceleration_mps2"])
        travel_distance = (
            horizontal_limit * horizontal_limit / acceleration
            + horizontal_limit * float(contract["control_period_s"])
            + 1.0
        )
        body_clearance = float(contract["vehicle"]["radius_m"]) + float(
            contract["vehicle"]["minimum_clearance_m"]
        )
        directions = select_native_test_directions(
            city,
            list(episode["starts"]),
            travel_distance_m=travel_distance,
            clearance_m=body_clearance,
            body_radius_m=float(contract["vehicle"]["radius_m"]),
        )
        transcript = build_native_action_transcript(contract, directions)
        world, drones, cameras = _create_dynamic_rig(stage, episode, contract)
        initial_first = {drone_id: _drone_state(drone) for drone_id, drone in drones.items()}
        dynamic_collision_paths = _collision_paths(stage, UsdPhysics.CollisionAPI)
        checks["fov_agreement"] = _camera_fov_check(cameras, contract)
        first_samples = _run_transcript(
            world,
            drones,
            transcript,
            float(contract["control_period_s"]),
        )
        checks["physics_step"] = _check(
            "PASS" if first_samples else "FAIL",
            physics_dt_s=1.0 / 60.0,
            control_steps=len(transcript),
            physics_steps=len(transcript)
            * round(float(contract["control_period_s"]) / (1.0 / 60.0)),
            dynamic_rigid_body_count=len(drones),
        )
        checks["observe_dwell"] = _dwell_check(first_samples, contract)
        checks["velocity_tracking"] = _velocity_check(first_samples, contract)
        checks["braking_distance"] = _braking_check(first_samples, transcript, contract)

        world.reset()
        initial_second = {drone_id: _drone_state(drone) for drone_id, drone in drones.items()}
        collision_paths_after_reset = _collision_paths(stage, UsdPhysics.CollisionAPI)
        checks["reset_isolation"] = _reset_check(
            initial_first,
            initial_second,
            dynamic_collision_paths,
            collision_paths_after_reset,
        )
        second_samples = _run_transcript(
            world,
            drones,
            transcript,
            float(contract["control_period_s"]),
        )
        checks["deterministic_replay_tolerance"] = compare_native_replays(
            first_samples,
            second_samples,
            position_tolerance_m=1.0e-4,
            velocity_tolerance_mps=1.0e-4,
            orientation_tolerance=1.0e-5,
        )
        capability_receipts = _build_capability_receipts(
            first_samples,
            episode_id=str(episode["episode_id"]),
            contract=contract,
        )
        capability_receipt_set = {
            "schema": "org.aerocity.bench.capability-receipt-set.v1",
            "execution_level": "L1",
            "formal_score_eligible": False,
            "evidence_scope": (
                "canonical_l1_dynamic_vertical_slice_not_formal_episode_score"
            ),
            "input_bindings": input_bindings,
            "energy_semantics": (
                "contract_proxy_from_measured_distance_plus_hover_power_not_native_battery"
            ),
            "safety_semantics": (
                "capability_probe_has_no_injected_failure_not_formal_contact_evidence"
            ),
            "receipts": capability_receipts,
        }
        capability_receipt_set["receipt_set_hash"] = content_hash(capability_receipts)
        capability_receipt_list_hash = str(capability_receipt_set["receipt_set_hash"])
        capability_receipt_set["capability_receipt_set_hash"] = content_hash(
            capability_receipt_set
        )
        capability_receipt_path = ARGS.output / "native_capability_receipts.json"
        write_json(capability_receipt_path, capability_receipt_set)
        capability_receipt_set_hash = str(capability_receipt_set["capability_receipt_set_hash"])
        dynamic_evidence = {
            "schema": "org.aerocity.bench.native-dynamic-evidence.v1",
            "input_bindings": input_bindings,
            "directions": {key: list(value) for key, value in sorted(directions.items())},
            "transcript": transcript,
            "first_replay_samples": first_samples,
            "second_replay_samples": second_samples,
            "capability_receipt_set_hash": capability_receipt_set_hash,
            "checks": {
                name: checks[name]
                for name in (
                    "fov_agreement",
                    "observe_dwell",
                    "velocity_tracking",
                    "braking_distance",
                    "reset_isolation",
                    "deterministic_replay_tolerance",
                )
            },
        }
        dynamic_evidence["dynamic_evidence_hash"] = content_hash(dynamic_evidence)
        evidence_path = ARGS.output / "native_dynamic_evidence.json"
        write_json(evidence_path, dynamic_evidence)
        evidence_sha256 = file_hash(evidence_path)
        for name in (
            "fov_agreement",
            "observe_dwell",
            "velocity_tracking",
            "braking_distance",
            "reset_isolation",
            "deterministic_replay_tolerance",
        ):
            checks[name]["dynamic_evidence_sha256"] = evidence_sha256
    except Exception as exc:  # noqa: BLE001
        checks.setdefault(
            "stage_load", _check("FAIL", error_type=type(exc).__name__, error=str(exc))
        )
        for name in REQUIRED_NATIVE_CHECKS:
            checks.setdefault(name, _check("FAIL", reason="native gate aborted before check"))
        (ARGS.output / "native_gate_exception.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    finally:
        if world is not None:
            try:
                world.stop()
            except Exception as cleanup_exc:  # noqa: BLE001
                (ARGS.output / "native_gate_cleanup.log").write_text(
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}\n", encoding="utf-8"
                )
        if context is not None:
            context.close_stage()

    report = write_native_gate_report(
        ARGS.output / "native_gate.json",
        stage_path=ARGS.stage,
        execution_level="L1",
        runtime_fingerprint={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "isaac_sim": _runtime_version(),
            "physics_dt_s": "0.016666666666666666",
            "native_gate_script_sha256": _sha256(Path(__file__)),
        },
        checks=checks,
        input_bindings=input_bindings,
    )
    # This historical capability probe intentionally uses DynamicCuboid and
    # direct velocity writes.  Keep the executor identity in the signed report
    # so no later paper table can mistake it for the formal quadrotor backend.
    report["vehicle_execution_model"] = "dynamic_cuboid_kinematic_capability_probe"
    report["quadrotor_dynamics_contract"] = {
        "status": "not_connected",
        "formal_score_eligible": False,
        "reason": (
            "formal L1 requires hash-verified local CF2X Multirotor, gravity, "
            "per-rotor motor response, and validated allocation"
        ),
    }
    failed = sorted(name for name, check in checks.items() if check.get("status") != "PASS")
    report["formal_score_eligible"] = False
    report["evidence_scope"] = "canonical_l1_dynamic_vertical_slice_not_formal_episode_score"
    if capability_receipt_set_hash is not None:
        report["capability_receipt_set"] = {
            "path": str((ARGS.output / "native_capability_receipts.json").resolve()),
            "sha256": _sha256(ARGS.output / "native_capability_receipts.json"),
            "receipt_set_hash": capability_receipt_list_hash,
            "capability_receipt_set_hash": capability_receipt_set_hash,
            "formal_score_eligible": False,
        }
    report["failed_checks"] = failed
    report.pop("native_gate_hash", None)
    from aerocity_bench.canonical import content_hash, write_json

    report["native_gate_hash"] = content_hash(report)
    write_json(ARGS.output / "native_gate.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


APP = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
exit_code = 1
try:
    exit_code = main()
except BaseException:  # noqa: BLE001
    (ARGS.output / "native_gate_exception.log").write_text(
        traceback.format_exc(), encoding="utf-8"
    )
finally:
    APP.close()
raise SystemExit(exit_code)
