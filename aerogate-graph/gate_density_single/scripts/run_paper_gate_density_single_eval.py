"""Run resumable paper E1/E2 single-drone gate-density evaluations.

This is an orchestration wrapper around ``run_gate_density_eval.py``.  It does
not duplicate layout, live-gate motion, or collision logic; every episode is
still produced by the existing single-drone evaluator so training, evaluation,
and replay stay on the same gate-density contract.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


GATE_AXIS: tuple[int, ...] = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
DEFAULT_SINGLE_DRONE_SPEED_MPS = 3.50
HISTORICAL_CHECKPOINT_PATH_MARKERS: tuple[str, ...] = (
    "gate_density_imitation_bridge_C4_C8_v1",
    "dynamic_gate_density_8d_c4a_24g_speed05_typefix",
)

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
    "experiment",
    "scenario",
    "method",
    "gate_count",
    "seed",
    "episodes",
    "moving_gate_enabled",
    "moving_gate_speed_mps",
    "moving_gate_amplitude_m",
    "drone_speed_mps",
    "drone_accel_mps2",
    "gate_post_radius_scale",
    "gate_half_width_scale",
    "team_success_rate",
    "per_agent_success_rate",
    "agent_agent_collision_rate",
    "obstacle_collision_rate",
    "min_pair_distance_mean_m",
    "min_pair_distance_min_m",
    "formation_slot_error_mean_m",
    "formation_slot_error_max_m",
    "progress_distance_mean_m",
    "dispersed_termination_rate",
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "height_contract_passed_rate",
    "corridor_through_success_rate",
    "side_bypass_failure_rate",
    "height_escape_failure_rate",
    "corridor_miss_failure_rate",
    "path_length_m_mean",
    "progress_distance_m_mean",
    "flight_time_s_mean",
    "min_clearance_m_mean",
    "mean_clearance_m_mean",
    "mean_speed_mps_mean",
    "max_speed_mps_mean",
    "mean_goal_tracking_error_m_mean",
    "max_goal_tracking_error_m_mean",
    "guidance_tracking_error_mean_m_mean",
    "guidance_tracking_error_max_m_mean",
    "planner_call_count_mean",
    "planner_failure_count_mean",
    "planner_latency_ms_mean_mean",
    "global_planner_trigger_count_mean",
    "global_planner_latency_ms_mean_mean",
    "shield_activation_count_mean",
    "shield_activation_ratio_mean",
    "guidance_query_count_mean",
    "guidance_latency_ms_mean_mean",
    "guidance_non_fallback_rate_mean",
    "actual_gate_motion_range_m_mean",
    "actual_gate_motion_range_x_m_mean",
    "actual_gate_motion_range_y_m_mean",
    "moving_gate_swept_clearance_m_min_mean",
    "dynamic_swept_collision_count_mean",
    "gate_gate_overlap_pair_count_max_mean",
    "gate_gate_frame_overlap_pair_count_max_mean",
    "stage_status",
    "checkpoint",
    "output_dir",
    "returncode",
    "duration_s",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _eval_drone_speed_axis_mps(root: Path) -> tuple[float, ...]:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import eval_drone_speed_axis_mps

    return eval_drone_speed_axis_mps()


def _drone_accel_for_speed(speed_mps: float, override_mps2: float | None = None) -> float:
    root = _repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import drone_accel_limit_for_speed_mps2

    return drone_accel_limit_for_speed_mps2(speed_mps, override_mps2)


def _validate_drone_command_limits(root: Path, speeds_mps: list[float], accel_mps2: float | None) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import MAX_DRONE_COMMAND_ACCEL_MPS2, MAX_DRONE_COMMAND_SPEED_MPS

    bad_speeds = [float(speed) for speed in speeds_mps if float(speed) <= 0.0 or float(speed) > MAX_DRONE_COMMAND_SPEED_MPS]
    if bad_speeds:
        raise SystemExit(f"drone speed values must be in (0, {MAX_DRONE_COMMAND_SPEED_MPS}]; got {bad_speeds}")
    if accel_mps2 is not None and (float(accel_mps2) <= 0.0 or float(accel_mps2) > MAX_DRONE_COMMAND_ACCEL_MPS2):
        raise SystemExit(f"--drone-accel-mps2 must be in (0, {MAX_DRONE_COMMAND_ACCEL_MPS2}]")


def _validate_checkpoint_for_formal_eval(args: argparse.Namespace) -> None:
    checkpoint_text = str(args.checkpoint).replace("\\", "/").lower()
    matched = [marker for marker in HISTORICAL_CHECKPOINT_PATH_MARKERS if marker.lower() in checkpoint_text]
    if matched and bool(getattr(args, "allow_historical_checkpoint", False)):
        return
    if matched:
        raise SystemExit(
            "Refusing historical diagnostic checkpoint for formal paper eval: "
            f"{args.checkpoint}. Historical checkpoint eval is disabled on the active aerogate_graph mainline."
        )


def _is_dynamic_job(experiment: str, scenario: str, gate_count: int) -> bool:
    return (
        (experiment == "E2_dynamic_single_gate_density" and int(gate_count) > 0)
        or (experiment == "E8_single_geometry_pressure" and str(scenario).startswith("dynamic") and int(gate_count) > 0)
        or (experiment == "E9_single_drone_speed_gradient" and str(scenario).startswith("dynamic") and int(gate_count) > 0)
    )


def _moving_gate_params_for_job(args: argparse.Namespace, experiment: str, scenario: str, gate_count: int) -> tuple[float, float]:
    if not _is_dynamic_job(experiment, scenario, gate_count):
        return 0.0, 0.0
    speed = (
        float(args.fixed_dynamic_gate_speed_mps)
        if args.fixed_dynamic_gate_speed_mps is not None
        else float(SPEED_BY_GATE_COUNT_MPS[int(gate_count)])
    )
    amplitude = (
        float(args.fixed_dynamic_gate_amplitude_m)
        if args.fixed_dynamic_gate_amplitude_m is not None
        else float(AMPLITUDE_BY_GATE_COUNT_M[int(gate_count)])
    )
    return speed, amplitude


def _method_args(method: str) -> list[str]:
    if method == "full":
        return ["--enable-agent-policy", "--enable-global-planner", "--enable-path-planner"]
    if method == "no_shield":
        return [
            "--enable-agent-policy",
            "--enable-global-planner",
            "--enable-path-planner",
            "--disable-safety-shield",
        ]
    if method == "planner_only":
        return ["--enable-global-planner", "--enable-path-planner"]
    if method == "reactive_only":
        return ["--enable-agent-policy"]
    if method == "fast_only_no_shield":
        return ["--enable-agent-policy", "--disable-safety-shield"]
    raise ValueError(f"unknown method: {method}")


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_summary(
    *,
    experiment: str,
    scenario: str,
    method: str,
    gate_post_radius_scale: float,
    gate_half_width_scale: float,
    output_dir: Path,
    summary: dict[str, Any],
    returncode: int,
    duration_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": experiment,
        "scenario": scenario,
        "method": method,
        "output_dir": str(output_dir),
        "returncode": int(returncode),
        "duration_s": float(duration_s),
        "gate_post_radius_scale": float(gate_post_radius_scale),
        "gate_half_width_scale": float(gate_half_width_scale),
    }
    for field in CSV_FIELDS:
        if field in row:
            continue
        row[field] = summary.get(field)
    row["moving_gate_enabled"] = summary.get("moving_gates_enabled")
    row["team_success_rate"] = summary.get("success_rate")
    row["per_agent_success_rate"] = summary.get("success_rate")
    row["agent_agent_collision_rate"] = 0.0
    row["obstacle_collision_rate"] = summary.get("collision_rate")
    row["min_pair_distance_mean_m"] = None
    row["min_pair_distance_min_m"] = None
    row["formation_slot_error_mean_m"] = None
    row["formation_slot_error_max_m"] = None
    row["progress_distance_mean_m"] = summary.get("progress_distance_m_mean")
    row["dispersed_termination_rate"] = 0.0
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, float, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("returncode") or 0) != 0:
            continue
        key = (
            str(row["experiment"]),
            str(row.get("scenario") or ""),
            str(row["method"]),
            int(row["gate_count"]),
            float(row.get("gate_post_radius_scale") or 1.0),
            float(row.get("gate_half_width_scale") or 1.0),
            float(row.get("drone_speed_mps") or DEFAULT_SINGLE_DRONE_SPEED_MPS),
            float(row.get("drone_accel_mps2") or _drone_accel_for_speed(float(row.get("drone_speed_mps") or DEFAULT_SINGLE_DRONE_SPEED_MPS))),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    metrics = (
        "success_rate",
        "collision_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "height_contract_passed_rate",
        "corridor_through_success_rate",
        "side_bypass_failure_rate",
        "height_escape_failure_rate",
        "corridor_miss_failure_rate",
        "team_success_rate",
        "per_agent_success_rate",
        "agent_agent_collision_rate",
        "obstacle_collision_rate",
        "progress_distance_mean_m",
        "dispersed_termination_rate",
        "path_length_m_mean",
        "progress_distance_m_mean",
        "flight_time_s_mean",
        "min_clearance_m_mean",
        "shield_activation_ratio_mean",
        "actual_gate_motion_range_m_mean",
        "moving_gate_swept_clearance_m_min_mean",
        "dynamic_swept_collision_count_mean",
    )
    for (experiment, scenario, method, gate_count, radius_scale, width_scale, drone_speed, drone_accel), items in sorted(grouped.items(), key=lambda item: item[0]):
        summary: dict[str, Any] = {
            "experiment": experiment,
            "scenario": scenario,
            "method": method,
            "gate_count": gate_count,
            "gate_post_radius_scale": radius_scale,
            "gate_half_width_scale": width_scale,
            "drone_speed_mps": drone_speed,
            "drone_accel_mps2": drone_accel,
            "seed_count": len({int(item["seed"]) for item in items}),
            "episodes_total": sum(int(item.get("episodes") or 0) for item in items),
            "moving_gate_speed_mps": items[0].get("moving_gate_speed_mps"),
            "moving_gate_amplitude_m": items[0].get("moving_gate_amplitude_m"),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_min"] = min(values) if values else None
            summary[f"{metric}_max"] = max(values) if values else None
        out.append(summary)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one(
    *,
    root: Path,
    python_exe: Path,
    checkpoint: Path,
    experiment: str,
    scenario: str,
    method: str,
    gate_count: int,
    seed: int,
    episodes: int,
    output_dir: Path,
    max_steps: int,
    layout_version: str,
    gate_post_radius_scale: float,
    gate_half_width_scale: float,
    drone_speed_mps: float,
    drone_accel_mps2: float,
    moving_gate_speed_mps: float,
    moving_gate_amplitude_m: float,
    dynamic_controller_profile: str,
    device: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "stage_summary.json"
    if summary_path.exists() and not overwrite:
        summary = _read_summary(summary_path)
        return _row_from_summary(
            experiment=experiment,
            scenario=scenario,
            method=method,
            gate_post_radius_scale=gate_post_radius_scale,
            gate_half_width_scale=gate_half_width_scale,
            output_dir=output_dir,
            summary=summary,
            returncode=0,
            duration_s=0.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    script = root / "gate_density_single" / "scripts" / "run_gate_density_eval.py"
    cmd = [
        str(python_exe),
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--gate-count",
        str(gate_count),
        "--seed",
        str(seed),
        "--random-yaw",
        "--gate-layout-version",
        layout_version,
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--gate-post-radius-scale",
        str(gate_post_radius_scale),
        "--gate-half-width-scale",
        str(gate_half_width_scale),
        "--drone-speed-mps",
        str(drone_speed_mps),
        "--drone-accel-mps2",
        str(drone_accel_mps2),
        "--output-dir",
        str(output_dir),
    ]
    cmd.extend(_method_args(method))
    if dynamic_controller_profile and dynamic_controller_profile != "none":
        cmd.extend(["--dynamic-controller-profile", str(dynamic_controller_profile)])
    if device:
        cmd.extend(["--device", device])

    moving = _is_dynamic_job(experiment, scenario, gate_count)
    if moving:
        cmd.append("--moving-gates")
        cmd.extend(["--moving-gate-speed-mps", str(float(moving_gate_speed_mps))])
        cmd.extend(["--moving-gate-amplitude-m", str(float(moving_gate_amplitude_m))])

    stdout_path = output_dir / "run_stdout.log"
    stderr_path = output_dir / "run_stderr.log"
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(root), stdout=stdout, stderr=stderr, text=True)
    duration_s = time.perf_counter() - start

    if summary_path.exists():
        summary = _read_summary(summary_path)
    else:
        summary = {
            "gate_count": gate_count,
            "seed": seed,
            "episodes": episodes,
            "checkpoint": str(checkpoint),
            "stage_status": "failed_no_summary",
        }
    return _row_from_summary(
        experiment=experiment,
        scenario=scenario,
        method=method,
        gate_post_radius_scale=gate_post_radius_scale,
        gate_half_width_scale=gate_half_width_scale,
        output_dir=output_dir,
        summary=summary,
        returncode=int(proc.returncode),
        duration_s=duration_s,
    )


def _build_jobs(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    default_drone_speed = float(args.drone_speed_mps)
    default_drone_accel = _drone_accel_for_speed(default_drone_speed, args.drone_accel_mps2)
    for experiment in args.experiments:
        for method in args.methods:
            if experiment == "E8_single_geometry_pressure":
                for gate_count in args.e8_gate_counts:
                    for mode in args.e8_modes:
                        scenario = f"{mode}_gate{int(gate_count):02d}"
                        moving_speed, moving_amplitude = _moving_gate_params_for_job(args, experiment, scenario, int(gate_count))
                        for radius_scale in args.geometry_scales:
                            for seed in args.seeds:
                                out = (
                                    args.output_root
                                    / experiment
                                    / method
                                    / scenario
                                    / f"radius_{float(radius_scale):.2f}_seed_{seed}"
                                )
                                jobs.append(
                                    {
                                        "root": root,
                                        "python_exe": args.python,
                                        "checkpoint": args.checkpoint,
                                        "experiment": experiment,
                                        "scenario": scenario,
                                        "method": method,
                                        "gate_count": int(gate_count),
                                        "seed": int(seed),
                                        "episodes": int(args.episodes),
                                        "output_dir": out,
                                        "max_steps": int(args.max_steps),
                                        "layout_version": str(args.gate_layout_version),
                                        "gate_post_radius_scale": float(radius_scale),
                                        "gate_half_width_scale": 1.0,
                                        "drone_speed_mps": default_drone_speed,
                                        "drone_accel_mps2": default_drone_accel,
                                        "moving_gate_speed_mps": moving_speed,
                                        "moving_gate_amplitude_m": moving_amplitude,
                                        "dynamic_controller_profile": str(args.dynamic_controller_profile),
                                        "device": args.device,
                                        "overwrite": bool(args.overwrite),
                                    }
                                )
                continue
            if experiment == "E9_single_drone_speed_gradient":
                scenarios = (
                    ("static_30", 30, 0.0, 0.0),
                    ("dynamic_42_speed14", 42, 1.4, AMPLITUDE_BY_GATE_COUNT_M[42]),
                    ("dynamic_60_speed20", 60, 2.0, AMPLITUDE_BY_GATE_COUNT_M[60]),
                )
                for scenario, gate_count, _gate_speed, _gate_amplitude in scenarios:
                    moving_speed, moving_amplitude = _moving_gate_params_for_job(args, experiment, scenario, int(gate_count))
                    for drone_speed in args.drone_speed_axis_mps:
                        drone_accel = _drone_accel_for_speed(float(drone_speed), args.drone_accel_mps2)
                        for seed in args.seeds:
                            out = (
                                args.output_root
                                / experiment
                                / method
                                / scenario
                                / f"drone_{float(drone_speed):.2f}_seed_{seed}"
                            )
                            jobs.append(
                                {
                                    "root": root,
                                    "python_exe": args.python,
                                    "checkpoint": args.checkpoint,
                                    "experiment": experiment,
                                    "scenario": scenario,
                                    "method": method,
                                    "gate_count": int(gate_count),
                                    "seed": int(seed),
                                    "episodes": int(args.episodes),
                                    "output_dir": out,
                                    "max_steps": int(args.max_steps),
                                    "layout_version": str(args.gate_layout_version),
                                    "gate_post_radius_scale": 1.0,
                                    "gate_half_width_scale": 1.0,
                                    "drone_speed_mps": float(drone_speed),
                                    "drone_accel_mps2": float(drone_accel),
                                    "moving_gate_speed_mps": moving_speed,
                                    "moving_gate_amplitude_m": moving_amplitude,
                                    "dynamic_controller_profile": str(args.dynamic_controller_profile),
                                    "device": args.device,
                                    "overwrite": bool(args.overwrite),
                                }
                            )
                continue
            for gate_count in args.gate_counts:
                if experiment == "E1_static_single_gate_density" and gate_count not in GATE_AXIS:
                    raise SystemExit(f"E1 gate-count must be in {GATE_AXIS}; got {gate_count}")
                if experiment == "E2_dynamic_single_gate_density" and gate_count not in GATE_AXIS:
                    raise SystemExit(f"E2 gate-count must be in {GATE_AXIS}; got {gate_count}")
                scenario = "static" if experiment == "E1_static_single_gate_density" else "dynamic"
                moving_speed, moving_amplitude = _moving_gate_params_for_job(args, experiment, scenario, int(gate_count))
                for seed in args.seeds:
                    out = args.output_root / experiment / method / f"gate_{gate_count:02d}_seed_{seed}"
                    jobs.append(
                        {
                            "root": root,
                            "python_exe": args.python,
                            "checkpoint": args.checkpoint,
                            "experiment": experiment,
                            "scenario": scenario,
                            "method": method,
                            "gate_count": int(gate_count),
                            "seed": int(seed),
                            "episodes": int(args.episodes),
                            "output_dir": out,
                            "max_steps": int(args.max_steps),
                            "layout_version": str(args.gate_layout_version),
                            "gate_post_radius_scale": 1.0,
                            "gate_half_width_scale": 1.0,
                            "drone_speed_mps": default_drone_speed,
                            "drone_accel_mps2": default_drone_accel,
                            "moving_gate_speed_mps": moving_speed,
                            "moving_gate_amplitude_m": moving_amplitude,
                            "dynamic_controller_profile": str(args.dynamic_controller_profile),
                            "device": args.device,
                            "overwrite": bool(args.overwrite),
                        }
                    )
    if args.max_runs is not None:
        jobs = jobs[: int(args.max_runs)]
    return jobs


def main() -> None:
    root = _repo_root()
    default_python = Path(sys.executable)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E1_static_single_gate_density", "E2_dynamic_single_gate_density"],
        choices=["E1_static_single_gate_density", "E2_dynamic_single_gate_density", "E8_single_geometry_pressure", "E9_single_drone_speed_gradient"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full"],
        choices=["full", "no_shield", "planner_only", "reactive_only", "fast_only_no_shield"],
    )
    parser.add_argument("--gate-counts", type=int, nargs="+", default=list(GATE_AXIS))
    parser.add_argument("--e8-gate-counts", type=int, nargs="+", default=[30, 60])
    parser.add_argument("--e8-modes", nargs="+", default=["static", "dynamic"], choices=["static", "dynamic"])
    parser.add_argument("--geometry-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--drone-speed-mps", type=float, default=DEFAULT_SINGLE_DRONE_SPEED_MPS)
    parser.add_argument("--drone-accel-mps2", type=float, default=None)
    parser.add_argument("--drone-speed-axis-mps", type=float, nargs="+", default=None)
    parser.add_argument("--fixed-dynamic-gate-speed-mps", type=float, default=None)
    parser.add_argument("--fixed-dynamic-gate-amplitude-m", type=float, default=None)
    parser.add_argument("--dynamic-controller-profile", type=str, default="none", choices=["none", "density_adaptive_v1"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", type=Path, default=default_python)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--allow-historical-checkpoint", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results" / "paper_2d",
    )
    parser.add_argument(
        "--gate-layout-version",
        type=str,
        default="irregular_centerline_v7_large_arena_dynamic",
    )
    args = parser.parse_args()
    if args.drone_speed_axis_mps is None:
        args.drone_speed_axis_mps = list(_eval_drone_speed_axis_mps(root))
    _validate_drone_command_limits(
        root,
        [float(args.drone_speed_mps), *(float(value) for value in args.drone_speed_axis_mps)],
        args.drone_accel_mps2,
    )
    _validate_checkpoint_for_formal_eval(args)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.python.exists():
        raise FileNotFoundError(args.python)

    jobs = _build_jobs(args, root)
    print(
        f"[launch] jobs={len(jobs)} workers={args.workers} experiments={args.experiments} "
        f"methods={args.methods} episodes={args.episodes}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
            pool.submit(_run_one, **job): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "experiment": job["experiment"],
                    "method": job["method"],
                    "gate_count": job["gate_count"],
                    "seed": job["seed"],
                    "episodes": job["episodes"],
                    "moving_gate_enabled": (
                        job["experiment"] == "E2_dynamic_single_gate_density"
                        or str(job.get("scenario") or "").startswith("dynamic")
                    ),
                    "moving_gate_speed_mps": job.get("moving_gate_speed_mps"),
                    "moving_gate_amplitude_m": job.get("moving_gate_amplitude_m"),
                    "drone_speed_mps": job.get("drone_speed_mps"),
                    "drone_accel_mps2": job.get("drone_accel_mps2"),
                    "stage_status": f"failed_exception:{type(exc).__name__}",
                    "checkpoint": str(args.checkpoint),
                    "output_dir": str(job["output_dir"]),
                    "returncode": -1,
                    "duration_s": 0.0,
                }
            rows.append(row)
            print(
                "[done] "
                f"{row.get('experiment')} {row.get('method')} gate={row.get('gate_count')} "
                f"seed={row.get('seed')} rc={row.get('returncode')} "
                f"succ={row.get('success_rate')} coll={row.get('collision_rate')} "
                f"timeout={row.get('timeout_rate')}",
                flush=True,
            )

    for experiment in args.experiments:
        experiment_rows = [row for row in rows if row.get("experiment") == experiment]
        if not experiment_rows:
            continue
        exp_root = args.output_root / experiment
        merged_csv = exp_root / "formal_merged_latest.csv"
        merged_json = exp_root / "formal_merged_latest.json"
        summary_json = exp_root / "formal_summary_latest.json"
        _write_csv(
            merged_csv,
            sorted(
                experiment_rows,
                key=lambda r: (
                    str(r["method"]),
                    str(r.get("scenario") or ""),
                    int(r["gate_count"]),
                    float(r.get("drone_speed_mps") or DEFAULT_SINGLE_DRONE_SPEED_MPS),
                    int(r["seed"]),
                ),
            ),
        )
        _write_json(
            merged_json,
            {
                "gate_axis": list(GATE_AXIS),
                "speed_by_gate_count_mps": SPEED_BY_GATE_COUNT_MPS,
                "amplitude_by_gate_count_m": AMPLITUDE_BY_GATE_COUNT_M,
                "fixed_dynamic_gate_speed_mps": args.fixed_dynamic_gate_speed_mps,
                "fixed_dynamic_gate_amplitude_m": args.fixed_dynamic_gate_amplitude_m,
                "drone_speed_axis_mps": list(args.drone_speed_axis_mps),
                "rows": experiment_rows,
            },
        )
        _write_json(summary_json, _summarize(experiment_rows))
        print(f"[summary] {experiment} csv={merged_csv} json={summary_json}", flush=True)


if __name__ == "__main__":
    main()

