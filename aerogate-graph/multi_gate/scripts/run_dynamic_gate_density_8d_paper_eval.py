"""Paper E2D dynamic gate-density evaluation for the multi_gate 8-drone line.

This script evaluates a fixed checkpoint on the paper gate-count axis
0,6,12,...,60.  It reuses the same curriculum stage builder used for training
so evaluation sees the same dynamic gate layout, live centers, velocity
observation, action shield, and collision logic.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


GATE_AXIS: tuple[int, ...] = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
DEFAULT_DRONE_SPEED_MPS = 3.50
SPEED_BY_GATE_COUNT_MPS: dict[int, float] = {
    0: 0.0,
    6: 0.0,
    12: 0.5,
    18: 0.8,
    24: 1.0,
    30: 1.1,
    36: 1.2,
    42: 1.4,
    48: 1.6,
    54: 1.8,
    60: 2.0,
}
AMPLITUDE_BY_GATE_COUNT_M: dict[int, float] = {
    0: 0.0,
    6: 0.0,
    12: 0.60,
    18: 0.75,
    24: 0.85,
    30: 0.90,
    36: 0.95,
    42: 1.00,
    48: 1.05,
    54: 1.10,
    60: 1.20,
}


CSV_FIELDS: tuple[str, ...] = (
    "gate_count",
    "speed_mps",
    "amplitude_m",
    "drone_speed_mps",
    "drone_accel_mps2",
    "seed",
    "episodes",
    "success_rate",
    "gate_post_collision_rate",
    "dynamic_gate_collision_rate",
    "agent_collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "hard_failure_rate",
    "safety_violation_rate",
    "mean_goal_distance_m",
    "mean_steps",
    "mean_slot_error_m",
    "mean_guidance_tracking_error_m",
    "mean_route_guidance_tracking_error_m",
    "mean_min_clearance_m",
    "min_min_clearance_m",
    "mean_min_pair_distance_m",
    "min_min_pair_distance_m",
    "mean_actual_gate_motion_range_m",
    "max_actual_gate_motion_range_m",
    "guidance_non_fallback_rate",
    "mean_guidance_latency_ms",
    "guidance_cache_hit_rate",
    "done_reason_counts",
    "checkpoint_path",
)


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _eval_drone_speed_axis_mps(root: Path) -> tuple[float, ...]:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import eval_drone_speed_axis_mps

    return eval_drone_speed_axis_mps()


def _drone_accel_for_speed(speed_mps: float, override_mps2: float | None = None) -> float:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import drone_accel_limit_for_speed_mps2

    return drone_accel_limit_for_speed_mps2(speed_mps, override_mps2)


def _validate_drone_command_limits(root: Path, speeds_mps: tuple[float, ...], accel_mps2: float | None) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import MAX_DRONE_COMMAND_ACCEL_MPS2, MAX_DRONE_COMMAND_SPEED_MPS

    bad_speeds = [float(speed) for speed in speeds_mps if float(speed) <= 0.0 or float(speed) > MAX_DRONE_COMMAND_SPEED_MPS]
    if bad_speeds:
        raise SystemExit(f"drone speed values must be in (0, {MAX_DRONE_COMMAND_SPEED_MPS}]; got {bad_speeds}")
    if accel_mps2 is not None and (float(accel_mps2) <= 0.0 or float(accel_mps2) > MAX_DRONE_COMMAND_ACCEL_MPS2):
        raise SystemExit(f"--drone-accel-mps2 must be in (0, {MAX_DRONE_COMMAND_ACCEL_MPS2}]")


def _load_runner(root: Path) -> Any:
    runner_path = root / "multi_gate" / "scripts" / "run_dynamic_gate_density_8d_curriculum.py"
    spec = importlib.util.spec_from_file_location("dynamic_gate_curriculum_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_gate_curriculum_runner"] = module
    spec.loader.exec_module(module)
    return module


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _flatten_summary(
    *,
    summary: dict[str, Any],
    gate_count: int,
    speed_mps: float,
    amplitude_m: float,
    drone_speed_mps: float,
    drone_accel_mps2: float,
    seed: int,
    episodes: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gate_count": int(gate_count),
        "speed_mps": float(speed_mps),
        "amplitude_m": float(amplitude_m),
        "drone_speed_mps": float(drone_speed_mps),
        "drone_accel_mps2": float(drone_accel_mps2),
        "seed": int(seed),
        "episodes": int(episodes),
        "checkpoint_path": str(checkpoint_path),
    }
    for key in CSV_FIELDS:
        if key in row:
            continue
        value = summary.get(key)
        if key == "done_reason_counts":
            row[key] = json.dumps(value or {}, ensure_ascii=False, sort_keys=True)
        else:
            row[key] = value
    return row


def _existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def main() -> None:
    root = _bootstrap()
    runner = _load_runner(root)

    from multi_gate.configs import get_multi_experiment_config
    from multi_gate.training import evaluate_checkpoint

    default_checkpoint = (
        root
        / "runtime"
        / "dynamic_gate_density_8d_c4a_24g_speed05_typefix_dagger_retry9_from_retry7_20260504"
        / "stages"
        / "04_C4a_bridge_24_gate_speed05_typefix_dagger_from_retry7"
        / "checkpoints"
        / "best_agent.pt"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "paper_2d" / "E2_dynamic_gate_density_8d" / "quick_curve",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20263021, 20264021])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--drone-speed-mps", type=float, default=DEFAULT_DRONE_SPEED_MPS)
    parser.add_argument("--drone-accel-mps2", type=float, default=None)
    parser.add_argument("--sweep-drone-speed", action="store_true")
    parser.add_argument("--drone-speed-axis-mps", type=float, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e2d_dynamic_gate_density_8d_rows.json"
    csv_path = output_dir / "e2d_dynamic_gate_density_8d_rows.csv"

    rows = [] if args.overwrite else _existing_rows(json_path)
    completed = {
        (
            int(row["gate_count"]),
            int(row["seed"]),
            int(row.get("episodes") or args.episodes),
            float(row.get("drone_speed_mps") or DEFAULT_DRONE_SPEED_MPS),
        )
        for row in rows
    }

    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    stage_template = runner._stages()[5]
    drone_speed_axis = (
        tuple(float(value) for value in args.drone_speed_axis_mps)
        if args.drone_speed_axis_mps is not None
        else (_eval_drone_speed_axis_mps(root) if bool(args.sweep_drone_speed) else (float(args.drone_speed_mps),))
    )
    _validate_drone_command_limits(root, drone_speed_axis, args.drone_accel_mps2)
    launched = 0
    for gate_count in GATE_AXIS:
        speed_mps = float(SPEED_BY_GATE_COUNT_MPS[gate_count])
        amplitude_m = float(AMPLITUDE_BY_GATE_COUNT_M[gate_count])
        for drone_speed_mps in drone_speed_axis:
            drone_accel_mps2 = _drone_accel_for_speed(float(drone_speed_mps), args.drone_accel_mps2)
            stage = replace(
                stage_template,
                name=(
                    f"E2D2_gate{gate_count:02d}_speed{int(round(speed_mps * 10)):02d}"
                    f"_drone{int(round(float(drone_speed_mps) * 100)):03d}"
                ),
                gate_count=int(gate_count),
                speed_mps=float(speed_mps),
                amplitude_m=float(amplitude_m),
                drone_speed_mps=float(drone_speed_mps),
                drone_accel_mps2=float(drone_accel_mps2),
            )
            config = runner._stage_config(base_config, stage)
            for seed in args.seeds:
                key = (int(gate_count), int(seed), int(args.episodes), float(drone_speed_mps))
                if key in completed:
                    print(
                        f"[skip] gate={gate_count} gate_speed={speed_mps:.2f} "
                        f"drone_speed={float(drone_speed_mps):.2f} seed={seed}",
                        flush=True,
                    )
                    continue
                if args.max_rows is not None and launched >= int(args.max_rows):
                    _write_json(
                        json_path,
                        {
                            "checkpoint": str(checkpoint_path),
                            "gate_axis": list(GATE_AXIS),
                            "speed_by_gate_count_mps": SPEED_BY_GATE_COUNT_MPS,
                            "amplitude_by_gate_count_m": AMPLITUDE_BY_GATE_COUNT_M,
                            "drone_speed_axis_mps": list(drone_speed_axis),
                            "rows": rows,
                        },
                    )
                    _write_csv(csv_path, rows)
                    print("[stop] max_rows reached", flush=True)
                    return

                print(
                    f"[eval] gate={gate_count} gate_speed={speed_mps:.2f} amp={amplitude_m:.2f} "
                    f"drone_speed={float(drone_speed_mps):.2f} seed={seed} episodes={args.episodes}",
                    flush=True,
                )
                summary = evaluate_checkpoint(
                    checkpoint_path=checkpoint_path,
                    episodes=int(args.episodes),
                    seed=int(seed),
                    device=str(args.device),
                    num_agents=8,
                    experiment_config=config,
                )
                row = _flatten_summary(
                    summary=summary,
                    gate_count=gate_count,
                    speed_mps=speed_mps,
                    amplitude_m=amplitude_m,
                    drone_speed_mps=float(drone_speed_mps),
                    drone_accel_mps2=float(drone_accel_mps2),
                    seed=int(seed),
                    episodes=int(args.episodes),
                    checkpoint_path=checkpoint_path,
                )
                rows.append(row)
                launched += 1
                _write_json(
                    json_path,
                    {
                        "checkpoint": str(checkpoint_path),
                        "gate_axis": list(GATE_AXIS),
                        "speed_by_gate_count_mps": SPEED_BY_GATE_COUNT_MPS,
                        "amplitude_by_gate_count_m": AMPLITUDE_BY_GATE_COUNT_M,
                        "drone_speed_axis_mps": list(drone_speed_axis),
                        "rows": rows,
                    },
                )
                _write_csv(csv_path, rows)
                print(
                    f"[done] gate={gate_count} seed={seed} drone_speed={float(drone_speed_mps):.2f} "
                    f"success={row.get('success_rate')} dyn={row.get('dynamic_gate_collision_rate')} "
                    f"agent={row.get('agent_collision_rate')} timeout={row.get('timeout_rate')}",
                    flush=True,
                )

    print(f"[complete] rows={len(rows)} json={json_path} csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()

