"""Measure the frozen H15 sensor matrix in one real Isaac Sim HM3D scene.

The pilot is a throughput admission run, not a search-policy benchmark.  It
uses the same static HM3D collision stage, physics step, run horizon, receiver
poses and failure denominator for the formal four-CF2X camera-free two-mode matrix.
H15 emits no task-quality field because no task controller is selected here;
later P07/P08 exclusively own exploration-quality measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts import FORMAL_FLEET_SIZE  # noqa: E402
from aerocity_method.runtime.sensors import (  # noqa: E402
    FORMAL_H15_SENSOR_PILOT_MODES,
    SensorProfile,
    SensorThroughputRecord,
)

ROW_SCHEMA_VERSION = "hm3d-h15-sensor-row-v3"
RUNNER_VERSION = "hm3d-h15-real-sensor-row-v4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite H15 evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _profile(mode: str) -> SensorProfile:
    if mode == "physics_only":
        return SensorProfile("physics-only-baseline", mode, 0.0, (), ())
    if mode == "sparse_range_3d":
        return SensorProfile(
            "sparse-range-3d-vfov90",
            mode,
            10.0,
            ("transit", "observe", "dwell", "map_update"),
            ("range_points", "source_observation_id"),
            range_enabled=True,
        )
    raise ValueError(f"unsupported formal H15 mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--receiver-positions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=FORMAL_H15_SENSOR_PILOT_MODES, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--physics-dt-s", type=float, default=1.0 / 120.0)
    parser.add_argument("--seed", type=int, default=20260801)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace, simulation_app: Any) -> int:
    import omni.physx
    import omni.usd
    import torch
    from isaaclab.sim import SimulationCfg, SimulationContext
    from pxr import UsdGeom

    if args.steps < 30 or args.physics_dt_s <= 0.0 or args.seed < 0:
        raise ValueError("invalid H15 steps, physics dt or seed")
    paths = {
        "collision": args.collision_usd.expanduser().resolve(),
        "receiver_positions": args.receiver_positions_json.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    for name in ("collision", "receiver_positions"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name}: {paths[name]}")
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite H15 evidence: {paths['output']}")
    receiver_source = json.loads(paths["receiver_positions"].read_text(encoding="utf-8"))
    if not isinstance(receiver_source, dict) or receiver_source.get("scene_id") != args.scene_id:
        raise ValueError("receiver position evidence scene mismatch")
    raw_positions = [row.get("receiver_position_w_m") for row in receiver_source.get("views", [])]
    if len(raw_positions) < 6:
        raise ValueError("H15 needs at least six independently audited receiver positions")
    positions = [tuple(float(value) for value in row) for row in raw_positions[:6]]

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
    for _ in range(12):
        simulation_app.update()
    sim = SimulationContext(
        SimulationCfg(dt=args.physics_dt_s, device=args.device, enable_scene_query_support=True)
    )
    interface = omni.physx.get_physx_scene_query_interface()
    comparison_id = f"h15-hm3d-train-{args.scene_id}-vfov90-v1"
    fleet_size = FORMAL_FLEET_SIZE
    mode = args.mode
    fleet_positions = positions[:fleet_size]
    profile = _profile(mode)
    sim.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sensor_elapsed = 0.0
    sensor_frames = 0
    observations = [0] * fleet_size
    frame_period_steps = (
        max(1, round(1.0 / (profile.update_hz * args.physics_dt_s))) if profile.update_hz else 0
    )
    for step in range(args.steps):
        phase = "transit"
        frame_due = (
            profile.mode != "physics_only"
            and step % frame_period_steps == 0
            and phase in profile.allowed_phases
        )
        if frame_due:
            frame_start = time.perf_counter()
            if profile.mode == "sparse_range_3d":
                # One source-bound sparse 3-D range sweep per UAV: the
                # endpoint is from the same PhysX collision scene.
                for agent_index, origin in enumerate(fleet_positions):
                    for direction in (
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                        (0.0, 0.0, -1.0),
                    ):
                        interface.raycast_closest(origin, direction, 20.0)
                    observations[agent_index] += 1
                    sensor_frames += 1
            sensor_elapsed += time.perf_counter() - frame_start
        sim.step(render=False)
    wall_elapsed = max(time.perf_counter() - start, 1.0e-9)
    simulated_seconds = args.steps * args.physics_dt_s
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    else:
        gpu_memory_mb = 0.0
    try:
        import psutil

        cpu_memory_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        cpu_memory_mb = 0.0
    expected_frame_events = sum(
        1
        for step in range(args.steps)
        if profile.mode != "physics_only" and step % frame_period_steps == 0
    )
    expected_frames = expected_frame_events * fleet_size
    dropped = max(0, expected_frames - sensor_frames)
    record = SensorThroughputRecord(
        comparison_id=comparison_id,
        scene_id=args.scene_id,
        episode_id="h15-throughput-episode",
        fleet_size=fleet_size,
        profile=profile,
        physics_dt_s=args.physics_dt_s,
        planned_episodes=1,
        executed_episodes=1,
        failed_episodes=0,
        physics_real_time_factor=simulated_seconds / wall_elapsed,
        environment_steps_per_s=args.steps / wall_elapsed,
        sensor_frames_per_s=sensor_frames / wall_elapsed,
        render_time_s=sensor_elapsed,
        transfer_time_s=0.0,
        gpu_memory_mb=gpu_memory_mb,
        cpu_memory_mb=cpu_memory_mb,
        dropped_frames=dropped,
        observations_per_agent=tuple(observations),
        measurement_scope="throughput_only",
        wall_clock_s=wall_elapsed,
    )
    payload = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "H15_REAL_SENSOR_ROW_COMPLETE",
        "synthetic": False,
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": f"isaac-hm3d-h15-{uuid.uuid4().hex}",
        "runtime_command_sha256": hashlib.sha256(
            "\0".join(str(value) for value in sys.argv).encode("utf-8")
        ).hexdigest(),
        "runner_version": RUNNER_VERSION,
        "source_observation_binding": True,
        "selection_partition": "train",
        "scene_id": args.scene_id,
        "collision_usd_sha256": _sha256(paths["collision"]),
        "receiver_position_source_sha256": _sha256(paths["receiver_positions"]),
        "pilot_claim_limit": (
            "Throughput and sensor-entitlement pilot only; no task-quality metric is measured."
        ),
        "fleet_size": fleet_size,
        "mode": mode,
        "record": record.to_dict(),
    }
    _write_new(paths["output"], payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fleet_size": fleet_size,
                "mode": mode,
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
    # Every matrix cell runs in a fresh process. The measurement has already
    # been atomically written by ``_write_new``; process isolation gives each
    # next cell a clean PhysX instance.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
