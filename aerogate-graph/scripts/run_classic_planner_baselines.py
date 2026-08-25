"""Run planner-only baselines against completed mainline scenarios.

Scope:
- eval_only, classic_python_planner
- no imitation learning, no DAgger, no RL, no checkpoint warm start
- reads completed scenario/results metadata from the configured results root
- writes planner baseline rows and planner-vs-completed-mainline tables
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.runtime.paths import RESULTS_ROOT, ensure_project_on_path

ROOT = ensure_project_on_path()

from single_internal_gate.configs.experiment_config import EXP2_SINGLE_INTERNAL_CONFIG
from single_internal_gate.planners import create_planner
from single_internal_gate.planners.interfaces import PlannerTask2D
from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D
from shared.core.dynamic_gate_density_2d import (
    DynamicGate2D,
    DynamicGateDensity2DConfig,
    default_dynamic_gate_density_config,
    gate_posts,
    generate_gate_layout,
    live_gate_centers,
    resolved_corridor_half_width_m,
)


CLASSIC_PLANNERS = ("astar", "theta_star", "rrt_star", "informed_rrt_star", "heuristic")
STRONG_PLANNERS = ("ego_planner", "fast_planner")
DEFAULT_PLANNERS = CLASSIC_PLANNERS + STRONG_PLANNERS
DEFAULT_EXPERIMENTS = (
    "E1_static_single_gate_density",
    "E2_dynamic_single_gate_density",
    "E4_static_multi_8d",
    "E5_dynamic_multi_8d",
)
_SINGLE_RE = re.compile(r"gate_(?P<gate>\d+)_seed_(?P<seed>\d+)$")
_MULTI_RE = re.compile(r"gate_(?P<gate>\d+)_team_(?P<team>\d+)_seed_(?P<seed>\d+)$")


class GateObstacleProvider:
    def __init__(
        self,
        *,
        gate_count: int,
        seed: int,
        fixed_height_m: float,
        drone_radius_m: float,
        world_x_bounds_m: tuple[float, float],
        world_y_bounds_m: tuple[float, float],
        gate_region_x_m: tuple[float, float],
        gate_region_y_m: tuple[float, float],
        gate_half_width_m: float,
        gate_post_radius_m: float,
        base_centers_xy: tuple[tuple[float, float], ...] = (),
        gate_yaws_rad: tuple[float, ...] = (),
        moving_enabled: bool = False,
        moving_gate_amplitude_m: float = 0.0,
        moving_gate_speed_mps: float = 0.0,
        timeseries_path: Path | None = None,
        generated_config: DynamicGateDensity2DConfig | None = None,
        generated_gates: tuple[DynamicGate2D, ...] | None = None,
        generated_static_layout: bool = True,
    ) -> None:
        self.gate_count = int(gate_count)
        self.seed = int(seed)
        self.fixed_height_m = float(fixed_height_m)
        self.drone_radius_m = float(drone_radius_m)
        self.world_x_bounds_m = world_x_bounds_m
        self.world_y_bounds_m = world_y_bounds_m
        self.gate_region_x_m = gate_region_x_m
        self.gate_region_y_m = gate_region_y_m
        self.gate_half_width_m = float(gate_half_width_m)
        self.gate_post_radius_m = float(gate_post_radius_m)
        self.base_centers_xy = base_centers_xy
        self.gate_yaws_rad = gate_yaws_rad
        self.moving_enabled = bool(moving_enabled)
        self.moving_gate_amplitude_m = float(moving_gate_amplitude_m)
        self.moving_gate_speed_mps = float(moving_gate_speed_mps)
        self.generated_config = generated_config
        self.generated_static_layout = bool(generated_static_layout)
        self._timeseries = _load_gate_timeseries(timeseries_path) if timeseries_path else ()
        self._generated_gates = (
            list(generated_gates)
            if generated_gates is not None
            else generate_gate_layout(
                gate_count=self.gate_count,
                seed=self.seed,
                config=generated_config,
                static_layout=generated_static_layout,
            )
            if generated_config is not None
            else None
        )
        self._cache: dict[int, GateObstacleMap2D] = {}
        self.dynamic_motion_source = self._resolve_motion_source()
        self.actual_gate_motion_range_m = self._estimate_motion_range()
        self.dynamic_gate_really_moves = bool(
            self.moving_enabled
            and self.moving_gate_amplitude_m > 1.0e-6
            and self.moving_gate_speed_mps > 1.0e-6
            and self.actual_gate_motion_range_m > 0.05
        )
        self.corridor_half_width_m = self._resolve_corridor_half_width()

    def _resolve_motion_source(self) -> str:
        if self._timeseries:
            return "completed_episode_live_gate_centers_timeseries"
        if self.generated_config is not None:
            return "shared_dynamic_gate_density_2d_live_gate_centers"
        if self.base_centers_xy:
            return "completed_task_static_gate_centers"
        return "empty_gate_layout"

    def _resolve_corridor_half_width(self) -> float:
        if self.gate_count <= 0:
            return float("inf")
        if self.generated_config is not None:
            return float(resolved_corridor_half_width_m(self.generated_config))
        y_abs = max(abs(float(self.gate_region_y_m[0])), abs(float(self.gate_region_y_m[1])))
        world_abs = max(abs(float(self.world_y_bounds_m[0])), abs(float(self.world_y_bounds_m[1])))
        return float(min(max(0.0, world_abs - 0.05), y_abs + self.gate_half_width_m + self.drone_radius_m + 0.4))

    def _estimate_motion_range(self) -> float:
        centers = []
        if self._timeseries:
            stride = max(1, len(self._timeseries) // 40)
            centers = [entry[1] for entry in self._timeseries[::stride]]
        elif self.generated_config is not None and self._generated_gates is not None:
            centers = [
                tuple(tuple(float(v) for v in row) for row in live_gate_centers(
                    self._generated_gates,
                    t_sec=float(t),
                    amplitude_m=self.moving_gate_amplitude_m,
                    speed_mps=self.moving_gate_speed_mps,
                    config=self.generated_config,
                ))
                for t in (0.0, 1.0, 2.5, 5.0, 8.0, 12.0)
            ]
        elif self.base_centers_xy:
            centers = [self.base_centers_xy]
        if len(centers) <= 1 or not centers[0]:
            return 0.0
        count = min(len(centers[0]), *(len(value) for value in centers))
        best = 0.0
        for idx in range(count):
            xs = [float(value[idx][0]) for value in centers]
            ys = [float(value[idx][1]) for value in centers]
            best = max(best, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        return float(best)

    def obstacle_map_at(self, t_sec: float) -> GateObstacleMap2D:
        key = int(round(float(t_sec) * 10.0))
        if key not in self._cache:
            self._cache[key] = _obstacle_map_from_posts(
                self.centers_at(float(t_sec)),
                self.yaws_at(),
                self.gate_half_width_m,
                self.gate_post_radius_m,
                self.fixed_height_m,
            )
            if len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def centers_at(self, t_sec: float) -> tuple[tuple[float, float], ...]:
        if self._timeseries:
            return self._centers_from_timeseries(t_sec)
        if self.generated_config is not None and self._generated_gates is not None:
            if not self.moving_enabled or self.moving_gate_amplitude_m <= 1.0e-6 or self.moving_gate_speed_mps <= 1.0e-6:
                return tuple((float(gate.base_center_xy[0]), float(gate.base_center_xy[1])) for gate in self._generated_gates)
            centers = live_gate_centers(
                self._generated_gates,
                t_sec=float(t_sec),
                amplitude_m=self.moving_gate_amplitude_m,
                speed_mps=self.moving_gate_speed_mps,
                config=self.generated_config,
            )
            return tuple((float(row[0]), float(row[1])) for row in centers)
        return self.base_centers_xy

    def yaws_at(self) -> tuple[float, ...]:
        if self._generated_gates is not None:
            return tuple(float(gate.yaw_rad) for gate in self._generated_gates)
        return self.gate_yaws_rad

    def _centers_from_timeseries(self, t_sec: float) -> tuple[tuple[float, float], ...]:
        if not self._timeseries:
            return self.base_centers_xy
        if len(self._timeseries) == 1:
            return self._timeseries[0][1]
        step = int(round(float(t_sec) / max(self._timeseries[1][0] - self._timeseries[0][0], 0.1)))
        step = max(0, min(len(self._timeseries) - 1, step))
        return self._timeseries[step][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT / "paper_2d")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--planner", action="append", choices=DEFAULT_PLANNERS)
    parser.add_argument("--experiment", action="append", choices=DEFAULT_EXPERIMENTS)
    parser.add_argument("--gate-count", action="append", type=int, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--limit-per-experiment", type=int, default=0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--max-speed-mps", type=float, default=3.5)
    parser.add_argument("--dynamic-replan-period-s", type=float, default=1.0)
    parser.add_argument(
        "--fixed-dynamic-gate-speed-mps",
        type=float,
        default=None,
        help="Override every dynamic scenario to this moving gate speed and ignore completed live-gate timeseries.",
    )
    parser.add_argument(
        "--fixed-dynamic-gate-amplitude-m",
        type=float,
        default=None,
        help="Override every dynamic scenario to this moving gate amplitude and ignore completed live-gate timeseries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel planner worker processes. Use >1 for CPU-bound planner-only sweeps.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    planners = tuple(args.planner) if args.planner else DEFAULT_PLANNERS
    experiments = tuple(args.experiment) if args.experiment else DEFAULT_EXPERIMENTS
    results_root = args.results_root.resolve()
    output_dir = args.output_dir or RESULTS_ROOT / "planner_baselines" / f"planner_only_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(results_root, experiments)
    for scenario in scenarios:
        scenario["completed_mainline_source"] = str(results_root)
    if args.fixed_dynamic_gate_speed_mps is not None or args.fixed_dynamic_gate_amplitude_m is not None:
        if args.fixed_dynamic_gate_speed_mps is None or args.fixed_dynamic_gate_amplitude_m is None:
            raise ValueError("fixed dynamic gate speed and amplitude must be provided together")
        for scenario in scenarios:
            if bool(scenario.get("dynamic")):
                scenario["fixed_dynamic_gate_speed_mps"] = float(args.fixed_dynamic_gate_speed_mps)
                scenario["fixed_dynamic_gate_amplitude_m"] = float(args.fixed_dynamic_gate_amplitude_m)
    if args.gate_count:
        allowed_gates = {int(value) for value in args.gate_count}
        scenarios = [row for row in scenarios if int(row["gate_count"]) in allowed_gates]
    if args.seed:
        allowed_seeds = {int(value) for value in args.seed}
        scenarios = [row for row in scenarios if int(row["seed"]) in allowed_seeds]
    if args.mode == "smoke":
        scenarios = _smoke_selection(scenarios)
    elif args.limit_per_experiment > 0:
        scenarios = _limit_per_experiment(scenarios, args.limit_per_experiment)

    worker_count = int(args.workers)
    if worker_count <= 0:
        worker_count = max(1, min(8, (os.cpu_count() or 2) - 1))
    worker_count = max(1, worker_count)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        provider = build_provider(scenario)
        base_audit = {
            "experiment": scenario["experiment"],
            "scenario_id": scenario["scenario_id"],
            "gate_count": scenario["gate_count"],
            "team_size": scenario.get("team_size", 1),
            "seed": scenario["seed"],
            "dynamic_expected": bool(scenario["dynamic"]),
            "dynamic_gate_really_moves": bool(provider.dynamic_gate_really_moves),
            "actual_gate_motion_range_m": provider.actual_gate_motion_range_m,
            "moving_gate_speed_mps": provider.moving_gate_speed_mps,
            "moving_gate_amplitude_m": provider.moving_gate_amplitude_m,
            "dynamic_motion_source": provider.dynamic_motion_source,
            "source_result_path": str(scenario["source_dir"]),
        }
        audit_rows.append(base_audit)

    jobs = [
        (
            planner_name,
            scenario,
            float(args.dt_s),
            float(args.max_speed_mps),
            float(args.dynamic_replan_period_s),
        )
        for scenario in scenarios
        for planner_name in planners
    ]
    partial_rows_path = output_dir / "planner_baseline_rows.partial.jsonl"
    if partial_rows_path.exists():
        partial_rows_path.unlink()
    progress_interval = max(1, int(args.progress_interval))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "scenarios": len(scenarios),
                "jobs": len(jobs),
                "workers": worker_count,
                "planners": planners,
                "experiments": experiments,
                "max_speed_mps": float(args.max_speed_mps),
                "dynamic_replan_period_s": float(args.dynamic_replan_period_s),
                "fixed_dynamic_gate_speed_mps": args.fixed_dynamic_gate_speed_mps,
                "fixed_dynamic_gate_amplitude_m": args.fixed_dynamic_gate_amplitude_m,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if worker_count == 1 or len(jobs) <= 1:
        for index, job in enumerate(jobs, start=1):
            row = _evaluate_planner_job(job)
            rows.append(row)
            append_jsonl(partial_rows_path, row)
            if index == 1 or index % progress_interval == 0 or index == len(jobs):
                print(json.dumps({"completed_jobs": index, "total_jobs": len(jobs)}, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {executor.submit(_evaluate_planner_job, job): job for job in jobs}
            for index, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    planner_name, scenario, *_ = job
                    raise RuntimeError(
                        f"Planner job failed: planner={planner_name} scenario={scenario.get('scenario_id')}"
                    ) from exc
                rows.append(row)
                append_jsonl(partial_rows_path, row)
                if index == 1 or index % progress_interval == 0 or index == len(jobs):
                    print(json.dumps({"completed_jobs": index, "total_jobs": len(jobs)}, ensure_ascii=False), flush=True)

    summary_rows = summarize_rows(rows)
    comparison_rows = summarize_comparison(rows)
    write_jsonl(output_dir / "planner_baseline_rows.jsonl", rows)
    write_csv(output_dir / "planner_baseline_summary.csv", summary_rows)
    write_csv(output_dir / "planner_vs_completed_mainline.csv", comparison_rows)
    write_json(output_dir / "planner_baseline_audit.json", {"scenarios": audit_rows})
    write_json(output_dir / "planner_baseline_metric_contract.json", metric_contract())
    write_json(
        output_dir / "planner_baseline_run_manifest.json",
        {
            "category": "classic/strong planner-only baseline",
            "algorithm_type": "classic_python_planner / eval_only",
            "training": "none",
            "learning": "none",
            "policy_checkpoint": "none",
            "checkpoint_warm_start": "none",
            "actor_critic_replay_buffer": "none",
            "execution": "deterministic path follower",
            "source_completed_results_root": str(results_root),
            "forbidden_sources": ["runtime", "currently running training", "checkpoint state"],
            "mode": args.mode,
            "planners": planners,
            "experiments": experiments,
            "scenario_count": len(scenarios),
            "row_count": len(rows),
            "workers": worker_count,
            "max_speed_mps": float(args.max_speed_mps),
            "dynamic_replan_period_s": float(args.dynamic_replan_period_s),
            "fixed_dynamic_gate_speed_mps": args.fixed_dynamic_gate_speed_mps,
            "fixed_dynamic_gate_amplitude_m": args.fixed_dynamic_gate_amplitude_m,
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "scenarios": len(scenarios)}, ensure_ascii=False, indent=2))


def metric_contract() -> dict[str, Any]:
    return {
        "distance_semantics": {
            "full_route_distance_m": "Required start-to-goal full-route distance for the scenario; never replaced by failed partial travel.",
            "progress_distance_mean_m": "Actual forward progress before success or true terminal failure.",
            "progress_ratio": "progress_distance_mean_m / full_route_distance_m.",
            "flown_path_length_m": "Actual executed travel length, including short failed rollouts.",
            "failed_episode_flown_path_length_m": "Actual executed travel length only for failed episodes; diagnostic only.",
            "path_length_m": "Success-only executed path length; blank for failed full-route episodes.",
            "path_length_ratio_success_only": "Success-only path_length_m / full_route_distance_m; blank for failed full-route episodes.",
        },
        "single_agent_metrics": [
            "success_rate",
            "collision_rate",
            "crash_rate",
            "obstacle_collision_rate",
            "timeout_rate",
            "planning_failure_rate",
            "no_path_rate",
            "out_of_bounds_rate",
            "hard_failure_rate",
            "safety_violation_rate",
            "progress_distance_mean_m",
            "progress_ratio",
            "full_route_distance_m",
            "remaining_goal_distance_m",
            "path_length_ratio_success_only",
            "mean_latency_ms",
            "p95_latency_ms",
            "min_clearance_m",
            "replan_count",
            "flight_time_s",
            "mean_speed_mps",
            "max_speed_mps",
            "configured_max_speed_mps",
            "drone_speed_limit_mps",
            "full_route_success_rate",
            "start_to_goal_complete_rate",
            "corridor_through_success_rate",
            "side_bypass_failure_rate",
            "height_contract_passed_rate",
            "height_out_of_bounds_rate",
            "dynamic_gate_really_moves",
            "moving_gate_speed_mps",
            "moving_gate_amplitude_m",
            "actual_gate_motion_range_m",
            "gate_contact_terminates_episode",
            "collision_terminates_episode",
            "mp4_generated",
            "video_accepted",
            "done_reason",
        ],
        "multi_agent_metrics": [
            "team_success_rate",
            "per_agent_success_rate",
            "agent_agent_collision_rate",
            "obstacle_collision_rate",
            "collision_rate",
            "timeout_rate",
            "planning_failure_rate",
            "no_path_rate",
            "out_of_bounds_rate",
            "hard_failure_rate",
            "safety_violation_rate",
            "min_pair_distance_mean_m",
            "formation_slot_error_mean_m",
            "formation_slot_error_max_m",
            "progress_distance_mean_m",
            "progress_ratio",
            "dispersed_termination_rate",
            "flight_time_s_mean",
            "mean_speed_mps",
            "max_speed_mps",
            "corridor_through_success_rate",
            "side_bypass_failure_rate",
            "height_contract_passed_rate",
            "height_out_of_bounds_rate",
            "dynamic_gate_really_moves",
            "actual_gate_motion_range_m",
            "gate_contact_terminates_episode",
            "collision_terminates_episode",
            "mp4_generated",
            "video_accepted",
            "done_reason",
        ],
        "baseline_contract": {
            "training": "none",
            "policy_checkpoint": "none",
            "learning": "none",
            "execution": "deterministic path follower",
            "source_completed_mainline": "configured results root",
        },
    }


def _evaluate_planner_job(job: tuple[str, dict[str, Any], float, float, float]) -> dict[str, Any]:
    planner_name, scenario, dt_s, max_speed_mps, dynamic_replan_period_s = job
    provider = build_provider(scenario)
    if int(scenario.get("team_size", 1)) <= 1:
        row = evaluate_single_scenario(
            planner_name=planner_name,
            scenario=scenario,
            provider=provider,
            dt_s=float(dt_s),
            max_speed_mps=float(max_speed_mps),
            dynamic_replan_period_s=float(dynamic_replan_period_s),
        )
    else:
        row = evaluate_multi_scenario(
            planner_name=planner_name,
            scenario=scenario,
            provider=provider,
            dt_s=float(dt_s),
            max_speed_mps=float(max_speed_mps),
            dynamic_replan_period_s=float(dynamic_replan_period_s),
        )
    row.update(_mainline_prefixed(scenario["mainline_metrics"]))
    row["completed_mainline_source"] = str(scenario.get("completed_mainline_source", ""))
    return row


def discover_scenarios(results_root: Path, experiments: tuple[str, ...]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for experiment in experiments:
        full_dir = results_root / experiment / "full"
        if not full_dir.exists():
            continue
        for stage_dir in sorted(path for path in full_dir.iterdir() if path.is_dir()):
            if experiment.startswith(("E1_", "E2_")):
                match = _SINGLE_RE.match(stage_dir.name)
                if not match:
                    continue
                summary_path = stage_dir / "summary.json"
                task_path = stage_dir / "episode_000" / "task.json"
                if not summary_path.exists() or not task_path.exists():
                    continue
                summary = read_json(summary_path)
                gate_count = int(match.group("gate"))
                seed = int(match.group("seed"))
                scenarios.append(
                    {
                        "experiment": experiment,
                        "scenario_kind": "single_dynamic" if experiment.startswith("E2_") else "single_static",
                        "scenario_id": stage_dir.name,
                        "gate_count": gate_count,
                        "team_size": 1,
                        "seed": seed,
                        "dynamic": experiment.startswith("E2_"),
                        "source_dir": stage_dir,
                        "task_path": task_path,
                        "timeseries_path": stage_dir / "episode_000" / "live_gate_centers_timeseries.jsonl",
                        "mainline_metrics": _mainline_metrics_from_single(summary),
                    }
                )
            elif experiment.startswith(("E4_", "E5_")):
                match = _MULTI_RE.match(stage_dir.name)
                if not match:
                    continue
                row_path = stage_dir / "row.json"
                if not row_path.exists():
                    continue
                row = read_json(row_path)
                scenarios.append(
                    {
                        "experiment": experiment,
                        "scenario_kind": "multi_dynamic" if experiment.startswith("E5_") else "multi_static",
                        "scenario_id": stage_dir.name,
                        "gate_count": int(match.group("gate")),
                        "team_size": int(match.group("team")),
                        "seed": int(match.group("seed")),
                        "dynamic": experiment.startswith("E5_"),
                        "source_dir": stage_dir,
                        "row_path": row_path,
                        "mainline_row": row,
                        "mainline_metrics": _mainline_metrics_from_multi(row),
                    }
                )
    return scenarios


def build_provider(scenario: dict[str, Any]) -> GateObstacleProvider:
    if int(scenario.get("team_size", 1)) <= 1:
        task_data = read_json(Path(scenario["task_path"]))
        return _provider_from_single_task(
            task_data,
            Path(scenario.get("timeseries_path")),
            fixed_dynamic_gate_speed_mps=scenario.get("fixed_dynamic_gate_speed_mps"),
            fixed_dynamic_gate_amplitude_m=scenario.get("fixed_dynamic_gate_amplitude_m"),
            dynamic_expected=bool(scenario.get("dynamic")),
        )
    return _provider_from_multi_row(scenario["mainline_row"])


def _provider_from_single_task(
    task_data: dict[str, Any],
    timeseries_path: Path,
    *,
    fixed_dynamic_gate_speed_mps: Any = None,
    fixed_dynamic_gate_amplitude_m: Any = None,
    dynamic_expected: bool = False,
) -> GateObstacleProvider:
    centers = tuple((float(row[0]), float(row[1])) for row in task_data.get("gate_centers_xy", []))
    yaws = tuple(float(value) for value in task_data.get("gate_yaws_rad", task_data.get("gate_requested_yaws_rad", [])))
    if len(yaws) < len(centers):
        yaws = yaws + (0.0,) * (len(centers) - len(yaws))
    start_xyz = task_data.get("start_xyz", [-27.0, 0.0, 4.0])
    goal_xyz = task_data.get("goal_xyz", [27.0, 0.0, 4.0])
    override_dynamic = fixed_dynamic_gate_speed_mps is not None or fixed_dynamic_gate_amplitude_m is not None
    moving_speed = (
        float(fixed_dynamic_gate_speed_mps)
        if fixed_dynamic_gate_speed_mps is not None
        else float(task_data.get("moving_gate_speed_mps", 0.0) or 0.0)
    )
    moving_amplitude = (
        float(fixed_dynamic_gate_amplitude_m)
        if fixed_dynamic_gate_amplitude_m is not None
        else float(task_data.get("moving_gate_amplitude_m", 0.0) or 0.0)
    )
    moving_enabled = bool(task_data.get("moving_gates_enabled", False) or dynamic_expected)
    if int(task_data.get("gate_count", len(centers))) <= 0:
        moving_enabled = False
        moving_speed = 0.0
        moving_amplitude = 0.0
    generated_config = None
    generated_gates = None
    generated_static_layout = True
    use_timeseries = timeseries_path.exists() and not override_dynamic
    if override_dynamic and moving_enabled and moving_speed > 1.0e-6 and moving_amplitude > 1.0e-6:
        generated_config = _single_task_dynamic_config(
            task_data,
            centers=centers,
            start_xyz=start_xyz,
            goal_xyz=goal_xyz,
            moving_gate_speed_mps=moving_speed,
            moving_gate_amplitude_m=moving_amplitude,
        )
        generated_gates = _dynamic_gates_from_completed_layout(centers, yaws)
        generated_static_layout = False
    return GateObstacleProvider(
        gate_count=int(task_data.get("gate_count", len(centers))),
        seed=int(task_data.get("seed", 0)),
        fixed_height_m=float(task_data.get("fixed_height_m", start_xyz[2] if len(start_xyz) > 2 else 4.0)),
        drone_radius_m=float(task_data.get("drone_radius_m", 0.35)),
        world_x_bounds_m=_tuple2(task_data.get("world_x_bounds_m", [-30.0, 30.0])),
        world_y_bounds_m=_tuple2(task_data.get("world_y_bounds_m", [-10.0, 10.0])),
        gate_region_x_m=_tuple2(task_data.get("gate_region_x_m", [min(start_xyz[0], goal_xyz[0]), max(start_xyz[0], goal_xyz[0])])),
        gate_region_y_m=_tuple2(task_data.get("gate_region_y_m", task_data.get("world_y_bounds_m", [-10.0, 10.0]))),
        gate_half_width_m=float(task_data.get("gate_half_width_m", 1.05)),
        gate_post_radius_m=float(task_data.get("gate_post_radius_m", 0.32)),
        base_centers_xy=centers,
        gate_yaws_rad=yaws,
        moving_enabled=moving_enabled,
        moving_gate_amplitude_m=moving_amplitude,
        moving_gate_speed_mps=moving_speed,
        timeseries_path=timeseries_path if use_timeseries else None,
        generated_config=generated_config,
        generated_gates=generated_gates,
        generated_static_layout=generated_static_layout,
    )


def _single_task_dynamic_config(
    task_data: dict[str, Any],
    *,
    centers: tuple[tuple[float, float], ...],
    start_xyz: list[Any],
    goal_xyz: list[Any],
    moving_gate_speed_mps: float,
    moving_gate_amplitude_m: float,
) -> DynamicGateDensity2DConfig:
    base = default_dynamic_gate_density_config()
    world_x = _tuple2(task_data.get("world_x_bounds_m", base.world_x_bounds_m))
    world_y = _tuple2(task_data.get("world_y_bounds_m", base.world_y_bounds_m))
    gate_region_x = _tuple2(
        task_data.get("gate_region_x_m", [min(float(start_xyz[0]), float(goal_xyz[0])), max(float(start_xyz[0]), float(goal_xyz[0]))])
    )
    if centers:
        center_y_abs = max(abs(float(row[1])) for row in centers)
    else:
        center_y_abs = 0.0
    gate_half_width = float(task_data.get("gate_half_width_m", base.gate_half_width_m))
    drone_radius = float(task_data.get("drone_radius_m", base.drone_radius_m))
    corridor_half_width = max(float(base.corridor_half_width_m), center_y_abs + gate_half_width + drone_radius + 0.4)
    return replace(
        base,
        world_x_bounds_m=world_x,
        world_y_bounds_m=world_y,
        start_x_m=float(start_xyz[0]),
        goal_x_m=float(goal_xyz[0]),
        fixed_height_m=float(task_data.get("fixed_height_m", start_xyz[2] if len(start_xyz) > 2 else base.fixed_height_m)),
        gate_count=int(task_data.get("gate_count", len(centers))),
        moving_gate_speed_mps=float(moving_gate_speed_mps),
        moving_gate_amplitude_m=float(moving_gate_amplitude_m),
        gate_region_x_m=gate_region_x,
        moving_clip_x_m=world_x,
        moving_clip_y_m=world_y,
        gate_half_width_m=gate_half_width,
        gate_post_radius_m=float(task_data.get("gate_post_radius_m", base.gate_post_radius_m)),
        drone_radius_m=drone_radius,
        corridor_half_width_m=float(corridor_half_width),
    )


def _dynamic_gates_from_completed_layout(
    centers: tuple[tuple[float, float], ...],
    yaws: tuple[float, ...],
) -> tuple[DynamicGate2D, ...]:
    gates: list[DynamicGate2D] = []
    for index, center in enumerate(centers):
        y = float(center[1])
        if abs(y) < 2.5:
            lane_index = 0
        elif y < 0.0:
            lane_index = 1
        else:
            lane_index = 2
        if lane_index == 0:
            motion_mode = "antiphase" if index % 2 else "lateral"
        else:
            motion_mode = "lissajous"
        gates.append(
            DynamicGate2D(
                base_center_xy=(float(center[0]), float(center[1])),
                yaw_rad=float(yaws[index]) if index < len(yaws) else 0.0,
                motion_phase=float(0.71 * index + 0.43 * lane_index),
                motion_mode=motion_mode,
                lane_index=int(lane_index),
                column_index=int(index),
            )
        )
    return tuple(gates)


def _provider_from_multi_row(row: dict[str, Any]) -> GateObstacleProvider:
    base = default_dynamic_gate_density_config()
    cfg = replace(
        base,
        gate_count=int(row.get("gate_count", 0)),
        moving_gate_speed_mps=float(row.get("speed_mps", row.get("moving_gate_speed_mps", 0.0)) or 0.0),
        moving_gate_amplitude_m=float(row.get("amplitude_m", row.get("moving_gate_amplitude_m", 0.0)) or 0.0),
        gate_post_radius_m=float(base.gate_post_radius_m) * float(row.get("gate_post_radius_scale", 1.0) or 1.0),
        gate_half_width_m=float(base.gate_half_width_m) * float(row.get("gate_half_width_scale", 1.0) or 1.0),
    )
    dynamic = str(row.get("experiment", "")).startswith("E5_") or "dynamic" in str(row.get("scenario", ""))
    static_layout = not dynamic or cfg.moving_gate_amplitude_m <= 1.0e-6 or cfg.moving_gate_speed_mps <= 1.0e-6
    return GateObstacleProvider(
        gate_count=int(row.get("gate_count", 0)),
        seed=int(row.get("seed", 0)),
        fixed_height_m=float(cfg.fixed_height_m),
        drone_radius_m=float(cfg.drone_radius_m),
        world_x_bounds_m=tuple(float(v) for v in cfg.world_x_bounds_m),
        world_y_bounds_m=tuple(float(v) for v in cfg.world_y_bounds_m),
        gate_region_x_m=tuple(float(v) for v in cfg.gate_region_x_m),
        gate_region_y_m=(-float(resolved_corridor_half_width_m(cfg)), float(resolved_corridor_half_width_m(cfg))),
        gate_half_width_m=float(cfg.gate_half_width_m),
        gate_post_radius_m=float(cfg.gate_post_radius_m),
        moving_enabled=dynamic,
        moving_gate_amplitude_m=float(cfg.moving_gate_amplitude_m),
        moving_gate_speed_mps=float(cfg.moving_gate_speed_mps),
        generated_config=cfg,
        generated_static_layout=static_layout,
    )


def evaluate_single_scenario(
    *,
    planner_name: str,
    scenario: dict[str, Any],
    provider: GateObstacleProvider,
    dt_s: float,
    max_speed_mps: float,
    dynamic_replan_period_s: float,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    task_suffix: str = "",
) -> dict[str, Any]:
    task_data = read_json(Path(scenario["task_path"])) if "task_path" in scenario else None
    if task_data is not None:
        raw_start = task_data.get("start_xyz", [provider.world_x_bounds_m[0], 0.0])
        raw_goal = task_data.get("goal_xyz", [provider.world_x_bounds_m[1], 0.0])
        base_start = (float(raw_start[0]), float(raw_start[1]))
        base_goal = (float(raw_goal[0]), float(raw_goal[1]))
    else:
        base_start = (float(provider.world_x_bounds_m[0]) + 5.0, 0.0)
        base_goal = (float(provider.world_x_bounds_m[1]) - 5.0, 0.0)
    start = start_xy or base_start
    goal = goal_xy or base_goal
    result = _rollout_planner(
        planner_name=planner_name,
        provider=provider,
        start_xy=start,
        goal_xy=goal,
        task_id=f"{scenario['scenario_id']}{task_suffix}",
        dt_s=dt_s,
        max_speed_mps=max_speed_mps,
        dynamic_replan_period_s=dynamic_replan_period_s,
    )
    straight = max(_distance(start, goal), 1.0e-6)
    scenario_kind = scenario["scenario_kind"]
    is_multi_agent_proxy = task_suffix != ""
    success_rate = 1.0 if result["success"] else 0.0
    collision_rate = 1.0 if result["collision"] else 0.0
    timeout_rate = 1.0 if result["timeout"] else 0.0
    planning_failure_rate = 1.0 if result["planning_failure"] else 0.0
    no_path_rate = 1.0 if result["no_path"] else 0.0
    side_bypass_rate = 1.0 if result["side_bypass_failure"] else 0.0
    out_of_bounds_rate = 1.0 if result["out_of_bounds"] else 0.0
    hard_failure_rate = 0.0 if result["success"] else 1.0
    flown_path_length_m = float(result["path_length_m"])
    path_length_m_success_only = flown_path_length_m if result["success"] and flown_path_length_m > 0.0 else None
    path_length_ratio_success_only = path_length_m_success_only / straight if path_length_m_success_only is not None else None
    return {
        "method": planner_name,
        "training": "none",
        "policy_checkpoint": "none",
        "learning": "none",
        "execution": "deterministic path follower",
        "category": "classic/strong planner-only baseline",
        "experiment": scenario["experiment"],
        "scenario_kind": scenario_kind,
        "scenario_id": scenario["scenario_id"],
        "gate_count": int(scenario["gate_count"]),
        "team_size": int(scenario.get("team_size", 1)),
        "seed": int(scenario["seed"]),
        "success_rate": success_rate,
        "team_success_rate": success_rate,
        "per_agent_success_rate": success_rate,
        "collision_rate": collision_rate,
        "crash_rate": collision_rate,
        "obstacle_collision_rate": collision_rate,
        "agent_agent_collision_rate": 0.0,
        "timeout_rate": timeout_rate,
        "planning_failure_rate": planning_failure_rate,
        "no_path_rate": no_path_rate,
        "out_of_bounds_rate": out_of_bounds_rate,
        "hard_failure_rate": hard_failure_rate,
        "safety_violation_rate": max(collision_rate, side_bypass_rate, out_of_bounds_rate),
        "progress_distance": result["progress_distance_m"],
        "progress_distance_mean_m": result["progress_distance_m"],
        "progress_ratio": result["progress_ratio"],
        "full_route_distance_m": result["route_distance_m"],
        "required_route_distance_m": result["route_distance_m"],
        "remaining_goal_distance_m": result["remaining_goal_distance_m"],
        "path_length_m": path_length_m_success_only,
        "path_length_m_success_only": path_length_m_success_only,
        "flown_path_length_m": flown_path_length_m,
        "failed_episode_flown_path_length_m": None if result["success"] else flown_path_length_m,
        "flown_path_length_ratio": flown_path_length_m / straight if flown_path_length_m > 0.0 else 0.0,
        "path_length_ratio": path_length_ratio_success_only,
        "path_length_ratio_success_only": path_length_ratio_success_only,
        "mean_latency_ms": result["mean_latency_ms"],
        "p95_latency_ms": result["p95_latency_ms"],
        "min_clearance": result["min_clearance_m"],
        "min_clearance_m": result["min_clearance_m"],
        "replan_count": result["replan_count"],
        "planner_call_count": result["planner_call_count"],
        "planner_failure_count": result["planner_failure_count"],
        "flight_time_s": result["flight_time_s"],
        "mean_speed_mps": result["mean_speed_mps"],
        "max_speed_mps": result["max_speed_mps"],
        "configured_max_speed_mps": float(max_speed_mps),
        "drone_speed_limit_mps": float(max_speed_mps),
        "full_route_success_rate": success_rate,
        "start_to_goal_complete_rate": success_rate,
        "corridor_through_success_rate": 1.0 if result["success"] and not result["side_bypass_failure"] else 0.0,
        "side_bypass_failure_rate": side_bypass_rate,
        "height_contract_passed_rate": 1.0,
        "height_out_of_bounds_rate": 0.0,
        "dynamic_gate_really_moves": provider.dynamic_gate_really_moves,
        "actual_gate_motion_range_m": provider.actual_gate_motion_range_m,
        "moving_gate_speed_mps": provider.moving_gate_speed_mps,
        "moving_gate_amplitude_m": provider.moving_gate_amplitude_m,
        "dynamic_motion_source": provider.dynamic_motion_source,
        "gate_contact_terminates_episode": True,
        "collision_terminates_episode": True,
        "mp4_generated": False,
        "video_accepted": False,
        "done_reason": result["done_reason"],
        "source_result_path": str(scenario["source_dir"]),
        "_trajectory_xy": result["trajectory_xy"] if is_multi_agent_proxy else None,
    }


def evaluate_multi_scenario(
    *,
    planner_name: str,
    scenario: dict[str, Any],
    provider: GateObstacleProvider,
    dt_s: float,
    max_speed_mps: float,
    dynamic_replan_period_s: float,
) -> dict[str, Any]:
    team_size = int(scenario.get("team_size", 8))
    slots = _formation_slots(team_size, spacing_m=0.85)
    leader_start = (float(provider.generated_config.start_x_m if provider.generated_config else -27.0), 0.0)
    leader_goal = (float(provider.generated_config.goal_x_m if provider.generated_config else 27.0), 0.0)
    leader_row = evaluate_single_scenario(
        planner_name=planner_name,
        scenario=scenario,
        provider=provider,
        dt_s=dt_s,
        max_speed_mps=max_speed_mps,
        dynamic_replan_period_s=dynamic_replan_period_s,
        start_xy=leader_start,
        goal_xy=leader_goal,
        task_suffix="_leader",
    )
    leader_trajectory = leader_row.pop("_trajectory_xy") or []
    if float(leader_row.get("planning_failure_rate", 0.0)) > 0.0 or float(leader_row.get("no_path_rate", 0.0)) > 0.0:
        return _multi_planning_failure_row(
            planner_name=planner_name,
            scenario=scenario,
            provider=provider,
            team_size=team_size,
            slots=slots,
            leader_start=leader_start,
            leader_goal=leader_goal,
            leader_row=leader_row,
        )
    trajectories = [
        [(float(point[0]) + slot[0], float(point[1]) + slot[1]) for point in leader_trajectory]
        for slot in slots
    ]
    agent_rows = []
    for index, slot in enumerate(slots):
        start = (leader_start[0] + slot[0], leader_start[1] + slot[1])
        goal = (leader_goal[0] + slot[0], leader_goal[1] + slot[1])
        agent_rows.append(
            _audit_shifted_agent_trajectory(
                provider=provider,
                trajectory=trajectories[index],
                start_xy=start,
                goal_xy=goal,
                dt_s=dt_s,
                leader_row=leader_row,
            )
        )
    pair = _pair_distance_metrics(trajectories, provider.drone_radius_m)
    slot_metrics = _formation_slot_metrics(trajectories, slots)
    per_agent_success = [row["success_rate"] >= 1.0 for row in agent_rows]
    obstacle_collision = any(row["collision_rate"] > 0.0 for row in agent_rows)
    timeout = any(row["timeout_rate"] > 0.0 for row in agent_rows)
    planning_failure = any(row["planning_failure_rate"] > 0.0 for row in agent_rows)
    no_path = any(row["no_path_rate"] > 0.0 for row in agent_rows)
    out_of_bounds = any(row["out_of_bounds_rate"] > 0.0 for row in agent_rows)
    side_bypass = any(row["side_bypass_failure_rate"] > 0.0 for row in agent_rows)
    dispersed = bool(slot_metrics["formation_slot_error_max_m"] > max(8.0, 1.6 * team_size * 0.85))
    team_success = all(per_agent_success) and not pair["agent_agent_collision"] and not dispersed
    min_clearance = min(float(row["min_clearance_m"]) for row in agent_rows)
    progress_mean = _mean(float(row["progress_distance_mean_m"]) for row in agent_rows)
    flown_path_length_mean = _mean(float(row["flown_path_length_m"]) for row in agent_rows)
    flight_time_mean = _mean(float(row["flight_time_s"]) for row in agent_rows)
    mean_speed = _mean(float(row["mean_speed_mps"]) for row in agent_rows)
    max_speed = max(float(row["max_speed_mps"]) for row in agent_rows)
    return {
        "method": planner_name,
        "training": "none",
        "policy_checkpoint": "none",
        "learning": "none",
        "execution": "deterministic path follower",
        "category": "classic/strong planner-only baseline",
        "experiment": scenario["experiment"],
        "scenario_kind": scenario["scenario_kind"],
        "scenario_id": scenario["scenario_id"],
        "gate_count": int(scenario["gate_count"]),
        "team_size": team_size,
        "seed": int(scenario["seed"]),
        "team_success_rate": 1.0 if team_success else 0.0,
        "per_agent_success_rate": sum(per_agent_success) / max(team_size, 1),
        "success_rate": 1.0 if team_success else 0.0,
        "collision_rate": 1.0 if obstacle_collision or pair["agent_agent_collision"] else 0.0,
        "crash_rate": 1.0 if obstacle_collision or pair["agent_agent_collision"] else 0.0,
        "obstacle_collision_rate": 1.0 if obstacle_collision else 0.0,
        "agent_agent_collision_rate": 1.0 if pair["agent_agent_collision"] else 0.0,
        "timeout_rate": 1.0 if timeout else 0.0,
        "planning_failure_rate": 1.0 if planning_failure else 0.0,
        "no_path_rate": 1.0 if no_path else 0.0,
        "out_of_bounds_rate": 1.0 if out_of_bounds else 0.0,
        "hard_failure_rate": 0.0 if team_success else 1.0,
        "safety_violation_rate": 1.0 if obstacle_collision or pair["agent_agent_collision"] or side_bypass or out_of_bounds else 0.0,
        "progress_distance": progress_mean,
        "progress_distance_mean_m": progress_mean,
        "progress_ratio": _mean(float(row["progress_ratio"]) for row in agent_rows),
        "full_route_distance_m": _mean(float(row["full_route_distance_m"]) for row in agent_rows),
        "required_route_distance_m": _mean(float(row["required_route_distance_m"]) for row in agent_rows),
        "remaining_goal_distance_m": _mean(float(row["remaining_goal_distance_m"]) for row in agent_rows),
        "path_length_m": _mean_or_none(row.get("path_length_m") for row in agent_rows) if team_success else None,
        "path_length_m_success_only": _mean_or_none(row.get("path_length_m_success_only") for row in agent_rows) if team_success else None,
        "path_length_m_mean": _mean_or_none(row.get("path_length_m") for row in agent_rows) if team_success else None,
        "flown_path_length_m": flown_path_length_mean,
        "flown_path_length_m_mean": flown_path_length_mean,
        "failed_episode_flown_path_length_m": None if team_success else flown_path_length_mean,
        "flown_path_length_ratio": _mean(float(row["flown_path_length_ratio"]) for row in agent_rows),
        "path_length_ratio": _mean_or_none(row.get("path_length_ratio") for row in agent_rows) if team_success else None,
        "path_length_ratio_success_only": _mean_or_none(row.get("path_length_ratio_success_only") for row in agent_rows) if team_success else None,
        "mean_latency_ms": float(leader_row["mean_latency_ms"]),
        "p95_latency_ms": float(leader_row["p95_latency_ms"]),
        "min_clearance": min_clearance,
        "min_clearance_m": min_clearance,
        "replan_count": float(leader_row["replan_count"]),
        "planner_call_count": float(leader_row["planner_call_count"]),
        "planner_failure_count": float(leader_row["planner_failure_count"]),
        "flight_time_s": flight_time_mean,
        "flight_time_s_mean": flight_time_mean,
        "mean_speed_mps": mean_speed,
        "max_speed_mps": max_speed,
        "min_pair_distance": pair["min_pair_distance_m"],
        "min_pair_distance_mean_m": pair["min_pair_distance_m"],
        "formation_slot_error_mean": slot_metrics["formation_slot_error_mean_m"],
        "formation_slot_error_max": slot_metrics["formation_slot_error_max_m"],
        "formation_slot_error_mean_m": slot_metrics["formation_slot_error_mean_m"],
        "formation_slot_error_max_m": slot_metrics["formation_slot_error_max_m"],
        "dispersed_termination_rate": 1.0 if dispersed else 0.0,
        "full_route_success_rate": 1.0 if team_success else 0.0,
        "corridor_through_success_rate": 1.0 if team_success else 0.0,
        "side_bypass_failure_rate": 1.0 if side_bypass else 0.0,
        "height_contract_passed_rate": 1.0,
        "height_out_of_bounds_rate": 0.0,
        "dynamic_gate_really_moves": provider.dynamic_gate_really_moves,
        "actual_gate_motion_range_m": provider.actual_gate_motion_range_m,
        "dynamic_motion_source": provider.dynamic_motion_source,
        "gate_contact_terminates_episode": True,
        "collision_terminates_episode": True,
        "mp4_generated": False,
        "video_accepted": False,
        "done_reason": _multi_done_reason(team_success, obstacle_collision, pair["agent_agent_collision"], timeout, dispersed, planning_failure, side_bypass, out_of_bounds),
        "source_result_path": str(scenario["source_dir"]),
    }


def _multi_planning_failure_row(
    *,
    planner_name: str,
    scenario: dict[str, Any],
    provider: GateObstacleProvider,
    team_size: int,
    slots: tuple[tuple[float, float], ...],
    leader_start: tuple[float, float],
    leader_goal: tuple[float, float],
    leader_row: dict[str, Any],
) -> dict[str, Any]:
    start_positions = [(leader_start[0] + slot[0], leader_start[1] + slot[1]) for slot in slots]
    route_distances = [_distance(start, (leader_goal[0] + slot[0], leader_goal[1] + slot[1])) for start, slot in zip(start_positions, slots)]
    obstacle_map = provider.obstacle_map_at(0.0)
    clearances = [obstacle_map.min_signed_distance(point, drone_radius_m=provider.drone_radius_m) for point in start_positions]
    initial_collision = any(value <= 0.0 for value in clearances)
    pair = _pair_distance_metrics([[point] for point in start_positions], provider.drone_radius_m)
    full_route_distance = _mean(route_distances)
    return {
        "method": planner_name,
        "training": "none",
        "policy_checkpoint": "none",
        "learning": "none",
        "execution": "deterministic path follower",
        "category": "classic/strong planner-only baseline",
        "experiment": scenario["experiment"],
        "scenario_kind": scenario["scenario_kind"],
        "scenario_id": scenario["scenario_id"],
        "gate_count": int(scenario["gate_count"]),
        "team_size": team_size,
        "seed": int(scenario["seed"]),
        "team_success_rate": 0.0,
        "per_agent_success_rate": 0.0,
        "success_rate": 0.0,
        "collision_rate": 1.0 if initial_collision or pair["agent_agent_collision"] else 0.0,
        "crash_rate": 1.0 if initial_collision or pair["agent_agent_collision"] else 0.0,
        "obstacle_collision_rate": 1.0 if initial_collision else 0.0,
        "agent_agent_collision_rate": 1.0 if pair["agent_agent_collision"] else 0.0,
        "timeout_rate": 1.0,
        "planning_failure_rate": 1.0,
        "no_path_rate": float(leader_row.get("no_path_rate", 1.0)),
        "out_of_bounds_rate": 0.0,
        "hard_failure_rate": 1.0,
        "safety_violation_rate": 1.0 if initial_collision or pair["agent_agent_collision"] else 0.0,
        "progress_distance": 0.0,
        "progress_distance_mean_m": 0.0,
        "progress_ratio": 0.0,
        "full_route_distance_m": full_route_distance,
        "required_route_distance_m": full_route_distance,
        "remaining_goal_distance_m": full_route_distance,
        "path_length_m": None,
        "path_length_m_success_only": None,
        "path_length_m_mean": None,
        "flown_path_length_m": 0.0,
        "flown_path_length_m_mean": 0.0,
        "failed_episode_flown_path_length_m": 0.0,
        "flown_path_length_ratio": 0.0,
        "path_length_ratio": None,
        "path_length_ratio_success_only": None,
        "mean_latency_ms": float(leader_row["mean_latency_ms"]),
        "p95_latency_ms": float(leader_row["p95_latency_ms"]),
        "min_clearance": min(float(value) for value in clearances),
        "min_clearance_m": min(float(value) for value in clearances),
        "replan_count": float(leader_row["replan_count"]),
        "planner_call_count": float(leader_row["planner_call_count"]),
        "planner_failure_count": float(leader_row["planner_failure_count"]),
        "flight_time_s": float(leader_row.get("flight_time_s", 0.0)),
        "flight_time_s_mean": float(leader_row.get("flight_time_s", 0.0)),
        "mean_speed_mps": 0.0,
        "max_speed_mps": 0.0,
        "min_pair_distance": pair["min_pair_distance_m"],
        "min_pair_distance_mean_m": pair["min_pair_distance_m"],
        "formation_slot_error_mean": None,
        "formation_slot_error_max": None,
        "formation_slot_error_mean_m": None,
        "formation_slot_error_max_m": None,
        "dispersed_termination_rate": 0.0,
        "full_route_success_rate": 0.0,
        "corridor_through_success_rate": 0.0,
        "side_bypass_failure_rate": 0.0,
        "height_contract_passed_rate": 1.0,
        "height_out_of_bounds_rate": 0.0,
        "dynamic_gate_really_moves": provider.dynamic_gate_really_moves,
        "actual_gate_motion_range_m": provider.actual_gate_motion_range_m,
        "dynamic_motion_source": provider.dynamic_motion_source,
        "gate_contact_terminates_episode": True,
        "collision_terminates_episode": True,
        "mp4_generated": False,
        "video_accepted": False,
        "done_reason": "planning_failure_timeout",
        "source_result_path": str(scenario["source_dir"]),
    }


def _audit_shifted_agent_trajectory(
    *,
    provider: GateObstacleProvider,
    trajectory: list[tuple[float, float]],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    dt_s: float,
    leader_row: dict[str, Any],
) -> dict[str, Any]:
    collision = False
    side_bypass = False
    out_of_bounds = False
    min_clearance = float("inf")
    for step, point in enumerate(trajectory):
        t_sec = step * dt_s
        obstacle_map = provider.obstacle_map_at(t_sec)
        min_clearance = min(min_clearance, obstacle_map.min_signed_distance(point, drone_radius_m=provider.drone_radius_m))
        side_bypass = side_bypass or _side_bypass(provider, point)
        out_of_bounds = out_of_bounds or _out_of_bounds(provider, point)
        if step > 0:
            prev = trajectory[step - 1]
            mid_map = provider.obstacle_map_at(t_sec - 0.5 * dt_s)
            collision = collision or obstacle_map.segment_collides(prev, point, drone_radius_m=provider.drone_radius_m)
            collision = collision or mid_map.segment_collides(prev, point, drone_radius_m=provider.drone_radius_m)
    path_len = sum(_distance(a, b) for a, b in zip(trajectory[:-1], trajectory[1:]))
    final_dist = _distance(trajectory[-1], goal_xy) if trajectory else _distance(start_xy, goal_xy)
    initial_dist = _distance(start_xy, goal_xy)
    timeout = bool(leader_row.get("timeout_rate", 0.0))
    planning_failure = bool(leader_row.get("planning_failure_rate", 0.0))
    no_path = bool(leader_row.get("no_path_rate", 0.0))
    success = (not collision) and (not out_of_bounds) and (not side_bypass) and final_dist <= 0.85 and not timeout
    path_ratio_success = float(path_len) / max(initial_dist, 1.0e-6) if success and path_len > 0.0 else None
    return {
        "success_rate": 1.0 if success else 0.0,
        "collision_rate": 1.0 if collision else 0.0,
        "timeout_rate": 1.0 if timeout else 0.0,
        "planning_failure_rate": 1.0 if planning_failure else 0.0,
        "no_path_rate": 1.0 if no_path else 0.0,
        "out_of_bounds_rate": 1.0 if out_of_bounds else 0.0,
        "side_bypass_failure_rate": 1.0 if side_bypass else 0.0,
        "progress_distance_mean_m": max(0.0, initial_dist - final_dist),
        "progress_ratio": max(0.0, initial_dist - final_dist) / max(initial_dist, 1.0e-6),
        "full_route_distance_m": initial_dist,
        "required_route_distance_m": initial_dist,
        "remaining_goal_distance_m": final_dist,
        "path_length_m": float(path_len) if success else None,
        "path_length_m_success_only": float(path_len) if success else None,
        "flown_path_length_m": float(path_len),
        "failed_episode_flown_path_length_m": None if success else float(path_len),
        "flown_path_length_ratio": float(path_len) / max(initial_dist, 1.0e-6) if path_len > 0.0 else 0.0,
        "path_length_ratio": path_ratio_success,
        "path_length_ratio_success_only": path_ratio_success,
        "flight_time_s": float(leader_row.get("flight_time_s", 0.0)),
        "mean_speed_mps": float(leader_row.get("mean_speed_mps", 0.0)),
        "max_speed_mps": float(leader_row.get("max_speed_mps", 0.0)),
        "min_clearance_m": float(min_clearance),
    }


def _rollout_planner(
    *,
    planner_name: str,
    provider: GateObstacleProvider,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    task_id: str,
    dt_s: float,
    max_speed_mps: float,
    dynamic_replan_period_s: float,
) -> dict[str, Any]:
    max_steps = int(max(240, min(2000, math.ceil((_distance(start_xy, goal_xy) / max(max_speed_mps, 1.0e-6)) / dt_s * 3.0 + 180))))
    goal_tolerance_m = 0.85
    waypoint_tolerance_m = 0.55
    position = (float(start_xy[0]), float(start_xy[1]))
    trajectory = [position]
    planner = create_planner(planner_name, EXP2_SINGLE_INTERNAL_CONFIG.planner)
    latencies: list[float] = []
    planner_failures = 0
    planner_calls = 0
    replan_count = 0
    side_bypass_failure = False
    min_clearance = provider.obstacle_map_at(0.0).min_signed_distance(position, drone_radius_m=provider.drone_radius_m)
    path: list[tuple[float, float]] = []
    waypoint_index = 0
    next_replan_t = 0.0

    def run_plan(t_sec: float, from_xy: tuple[float, float], suffix: str) -> bool:
        nonlocal path, waypoint_index, planner_calls, planner_failures, latencies
        task = _task_from_provider(provider, from_xy, goal_xy, f"{task_id}_{suffix}", t_sec)
        result = planner.plan(task)
        planner_calls += 1
        latencies.append(float(result.planning_time_ms))
        if not result.success or len(result.path_xy) < 2:
            planner_failures += 1
            return False
        path = list(result.path_xy)
        path[0] = from_xy
        waypoint_index = 1
        return True

    if not run_plan(0.0, position, "initial"):
        full_timeout_trajectory = [position for _ in range(max_steps + 1)]
        return _rollout_result(
            success=False,
            collision=False,
            timeout=True,
            side_bypass_failure=False,
            done_reason="planning_failure_timeout",
            start_xy=start_xy,
            goal_xy=goal_xy,
            trajectory=full_timeout_trajectory,
            dt_s=dt_s,
            latencies=latencies,
            planner_calls=planner_calls,
            planner_failures=planner_failures,
            replan_count=replan_count,
            min_clearance=min_clearance,
        )

    for step in range(max_steps):
        t_sec = step * dt_s
        if _distance(position, goal_xy) <= goal_tolerance_m:
            return _rollout_result(
                success=not side_bypass_failure,
                collision=False,
                timeout=False,
                side_bypass_failure=side_bypass_failure,
                done_reason="goal_reached" if not side_bypass_failure else "side_bypass_failure",
                start_xy=start_xy,
                goal_xy=goal_xy,
                trajectory=trajectory,
                dt_s=dt_s,
                latencies=latencies,
                planner_calls=planner_calls,
                planner_failures=planner_failures,
                replan_count=replan_count,
                min_clearance=min_clearance,
            )
        if provider.dynamic_gate_really_moves and t_sec >= next_replan_t + max(dynamic_replan_period_s, dt_s) - 1.0e-9:
            if run_plan(t_sec, position, f"replan_{replan_count + 1:03d}"):
                replan_count += 1
            next_replan_t = t_sec
        while waypoint_index < len(path) and _distance(position, path[waypoint_index]) <= waypoint_tolerance_m:
            waypoint_index += 1
        target = goal_xy if waypoint_index >= len(path) else path[waypoint_index]
        next_position = _advance(position, target, max_speed_mps * dt_s)
        start_map = provider.obstacle_map_at(t_sec)
        mid_map = provider.obstacle_map_at(t_sec + 0.5 * dt_s)
        end_map = provider.obstacle_map_at(t_sec + dt_s)
        collision = any(
            obstacle_map.segment_collides(position, next_position, drone_radius_m=provider.drone_radius_m)
            for obstacle_map in (start_map, mid_map, end_map)
        )
        trajectory.append(next_position)
        position = next_position
        min_clearance = min(min_clearance, end_map.min_signed_distance(position, drone_radius_m=provider.drone_radius_m))
        side_bypass_failure = side_bypass_failure or _side_bypass(provider, position)
        if collision:
            return _rollout_result(
                success=False,
                collision=True,
                timeout=False,
                side_bypass_failure=side_bypass_failure,
                done_reason="obstacle_collision",
                start_xy=start_xy,
                goal_xy=goal_xy,
                trajectory=trajectory,
                dt_s=dt_s,
                latencies=latencies,
                planner_calls=planner_calls,
                planner_failures=planner_failures,
                replan_count=replan_count,
                min_clearance=min_clearance,
            )
        if _out_of_bounds(provider, position):
            return _rollout_result(
                success=False,
                collision=False,
                timeout=False,
                side_bypass_failure=side_bypass_failure,
                done_reason="out_of_bounds",
                start_xy=start_xy,
                goal_xy=goal_xy,
                trajectory=trajectory,
                dt_s=dt_s,
                latencies=latencies,
                planner_calls=planner_calls,
                planner_failures=planner_failures,
                replan_count=replan_count,
                min_clearance=min_clearance,
            )

    return _rollout_result(
        success=False,
        collision=False,
        timeout=True,
        side_bypass_failure=side_bypass_failure,
        done_reason="timeout",
        start_xy=start_xy,
        goal_xy=goal_xy,
        trajectory=trajectory,
        dt_s=dt_s,
        latencies=latencies,
        planner_calls=planner_calls,
        planner_failures=planner_failures,
        replan_count=replan_count,
        min_clearance=min_clearance,
    )


def _rollout_result(
    *,
    success: bool,
    collision: bool,
    timeout: bool,
    side_bypass_failure: bool,
    done_reason: str,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    trajectory: list[tuple[float, float]],
    dt_s: float,
    latencies: list[float],
    planner_calls: int,
    planner_failures: int,
    replan_count: int,
    min_clearance: float,
) -> dict[str, Any]:
    path_len = sum(_distance(a, b) for a, b in zip(trajectory[:-1], trajectory[1:]))
    speeds = [_distance(a, b) / max(dt_s, 1.0e-9) for a, b in zip(trajectory[:-1], trajectory[1:])]
    final_dist = _distance(trajectory[-1], goal_xy)
    initial_dist = _distance(start_xy, goal_xy)
    planning_failure = str(done_reason).startswith("planning_failure") or str(done_reason) == "no_path"
    no_path = planning_failure and int(planner_calls) > 0 and int(planner_failures) >= int(planner_calls)
    out_of_bounds = str(done_reason) == "out_of_bounds"
    progress = max(0.0, initial_dist - final_dist)
    return {
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "planning_failure": bool(planning_failure),
        "no_path": bool(no_path),
        "out_of_bounds": bool(out_of_bounds),
        "side_bypass_failure": bool(side_bypass_failure),
        "done_reason": done_reason,
        "route_distance_m": float(initial_dist),
        "remaining_goal_distance_m": float(final_dist),
        "progress_distance_m": progress,
        "progress_ratio": progress / max(initial_dist, 1.0e-6),
        "path_length_m": float(path_len),
        "flight_time_s": max(0, len(trajectory) - 1) * float(dt_s),
        "mean_speed_mps": _mean(speeds),
        "max_speed_mps": max(speeds, default=0.0),
        "mean_latency_ms": _mean(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "planner_call_count": int(planner_calls),
        "planner_failure_count": int(planner_failures),
        "replan_count": int(replan_count),
        "min_clearance_m": float(min_clearance),
        "trajectory_xy": trajectory,
    }


def _task_from_provider(
    provider: GateObstacleProvider,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    task_id: str,
    t_sec: float,
) -> PlannerTask2D:
    return PlannerTask2D(
        start_xy=start_xy,
        goal_xy=goal_xy,
        obstacles_2d=provider.obstacle_map_at(t_sec),
        fixed_height_m=provider.fixed_height_m,
        task_id=task_id,
        drone_radius_m=provider.drone_radius_m,
        world_x_bounds_m=provider.world_x_bounds_m,
        world_y_bounds_m=provider.world_y_bounds_m,
    )


def _obstacle_map_from_posts(
    centers_xy: tuple[tuple[float, float], ...],
    yaws_rad: tuple[float, ...],
    half_width_m: float,
    post_radius_m: float,
    fixed_height_m: float,
) -> GateObstacleMap2D:
    obstacles: list[GatePostObstacle2D] = []
    for gate_index, center in enumerate(centers_xy):
        yaw = float(yaws_rad[gate_index]) if gate_index < len(yaws_rad) else 0.0
        axis = (-math.sin(yaw) * float(half_width_m), math.cos(yaw) * float(half_width_m))
        for post_index, sign in enumerate((-1.0, 1.0)):
            obstacles.append(
                GatePostObstacle2D(
                    species="gate_post",
                    center_xy=(float(center[0]) + sign * axis[0], float(center[1]) + sign * axis[1]),
                    collision_radius_m=float(post_radius_m),
                    canopy_height_m=float(fixed_height_m),
                    description=f"gate_{gate_index:03d}_post_{post_index}",
                    usd_path="",
                )
            )
    return GateObstacleMap2D(tuple(obstacles))


def _load_gate_timeseries(path: Path | None) -> tuple[tuple[float, tuple[tuple[float, float], ...]], ...]:
    if path is None or not path.exists():
        return ()
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            centers = tuple((float(row[0]), float(row[1])) for row in data.get("gate_centers_xy", []))
            result.append((float(data.get("t_sec", len(result) * 0.1)), centers))
    return tuple(result)


def _advance(position: tuple[float, float], target: tuple[float, float], max_step_m: float) -> tuple[float, float]:
    dist = _distance(position, target)
    if dist <= max_step_m or dist <= 1.0e-9:
        return (float(target[0]), float(target[1]))
    scale = max_step_m / dist
    return (float(position[0] + (target[0] - position[0]) * scale), float(position[1] + (target[1] - position[1]) * scale))


def _side_bypass(provider: GateObstacleProvider, position: tuple[float, float]) -> bool:
    if provider.gate_count <= 0 or not math.isfinite(provider.corridor_half_width_m):
        return False
    x = float(position[0])
    if not (float(provider.gate_region_x_m[0]) <= x <= float(provider.gate_region_x_m[1])):
        return False
    return abs(float(position[1])) > float(provider.corridor_half_width_m)


def _out_of_bounds(provider: GateObstacleProvider, position: tuple[float, float]) -> bool:
    return not (
        float(provider.world_x_bounds_m[0]) <= float(position[0]) <= float(provider.world_x_bounds_m[1])
        and float(provider.world_y_bounds_m[0]) <= float(position[1]) <= float(provider.world_y_bounds_m[1])
    )


def _formation_slots(team_size: int, spacing_m: float) -> tuple[tuple[float, float], ...]:
    if team_size <= 1:
        return ((0.0, 0.0),)
    return tuple((0.0, (index - 0.5 * (team_size - 1)) * float(spacing_m)) for index in range(team_size))


def _pair_distance_metrics(trajectories: list[list[tuple[float, float]]], drone_radius_m: float) -> dict[str, Any]:
    max_len = max((len(traj) for traj in trajectories), default=0)
    min_pair = float("inf")
    collision = False
    threshold = 2.0 * float(drone_radius_m)
    for step in range(max_len):
        points = [traj[min(step, len(traj) - 1)] for traj in trajectories if traj]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = _distance(points[i], points[j])
                min_pair = min(min_pair, dist)
                collision = collision or dist <= threshold
    return {"min_pair_distance_m": float(min_pair), "agent_agent_collision": bool(collision)}


def _formation_slot_metrics(trajectories: list[list[tuple[float, float]]], slots: tuple[tuple[float, float], ...]) -> dict[str, float]:
    max_len = max((len(traj) for traj in trajectories), default=0)
    errors: list[float] = []
    max_error = 0.0
    for step in range(max_len):
        points = [traj[min(step, len(traj) - 1)] for traj in trajectories if traj]
        if not points:
            continue
        center = (_mean(point[0] for point in points), _mean(point[1] for point in points))
        for idx, point in enumerate(points):
            if idx >= len(slots):
                continue
            target = (center[0] + slots[idx][0], center[1] + slots[idx][1])
            error = _distance(point, target)
            errors.append(error)
            max_error = max(max_error, error)
    return {"formation_slot_error_mean_m": _mean(errors), "formation_slot_error_max_m": float(max_error)}


def _multi_done_reason(
    team_success: bool,
    obstacle_collision: bool,
    agent_collision: bool,
    timeout: bool,
    dispersed: bool,
    planning_failure: bool,
    side_bypass: bool,
    out_of_bounds: bool,
) -> str:
    if team_success:
        return "goal_reached"
    if obstacle_collision:
        return "obstacle_collision"
    if agent_collision:
        return "agent_collision"
    if dispersed:
        return "dispersed_termination"
    if planning_failure:
        return "planning_failure_timeout"
    if side_bypass:
        return "side_bypass_failure"
    if out_of_bounds:
        return "out_of_bounds"
    if timeout:
        return "timeout"
    return "failed"


def _mainline_metrics_from_single(summary: dict[str, Any]) -> dict[str, Any]:
    start = summary.get("start_xyz", [-27.0, 0.0])
    goal = summary.get("goal_xyz", [27.0, 0.0])
    initial = abs(float(goal[0]) - float(start[0]))
    success_rate = _float_or_nan(summary.get("success_rate"))
    mainline_path_length = _float_or_nan(summary.get("path_length_m_mean"))
    progress = _mainline_single_progress_distance(summary, initial, success_rate, mainline_path_length)
    collision_rate = _float_or_nan(summary.get("collision_rate"))
    timeout_rate = _float_or_nan(summary.get("timeout_rate"))
    hard_failure_rate = 1.0 - success_rate if math.isfinite(success_rate) else None
    out_of_bounds_rate = _float_or_nan(summary.get("out_of_bounds_rate"))
    side_bypass_rate = _float_or_nan(summary.get("side_bypass_failure_rate"))
    safety_parts = [value for value in (collision_rate, out_of_bounds_rate, side_bypass_rate) if math.isfinite(value)]
    return {
        "success_rate": summary.get("success_rate"),
        "team_success_rate": summary.get("success_rate"),
        "per_agent_success_rate": summary.get("success_rate"),
        "collision_rate": summary.get("collision_rate"),
        "crash_rate": summary.get("collision_rate"),
        "obstacle_collision_rate": summary.get("collision_rate"),
        "agent_agent_collision_rate": 0.0,
        "timeout_rate": summary.get("timeout_rate"),
        "planning_failure_rate": 0.0,
        "no_path_rate": 0.0,
        "out_of_bounds_rate": summary.get("out_of_bounds_rate"),
        "hard_failure_rate": hard_failure_rate,
        "safety_violation_rate": max(safety_parts) if safety_parts else None,
        "full_route_success_rate": summary.get("success_rate"),
        "start_to_goal_complete_rate": summary.get("success_rate"),
        "required_route_distance_m": initial,
        "full_route_distance_m": initial,
        "progress_distance_mean_m": progress,
        "progress_ratio": progress / max(initial, 1.0e-6) if progress is not None else None,
        "path_length_m_mean": summary.get("path_length_m_mean") if math.isfinite(success_rate) and success_rate >= 0.999 else None,
        "flown_path_length_m_mean": summary.get("path_length_m_mean"),
        "path_length_ratio_success_only": (
            mainline_path_length / max(initial, 1.0e-6)
            if math.isfinite(mainline_path_length) and math.isfinite(success_rate) and success_rate >= 0.999
            else None
        ),
        "flight_time_s_mean": summary.get("flight_time_s_mean"),
        "mean_speed_mps": summary.get("mean_speed_mps_mean"),
        "max_speed_mps": summary.get("max_speed_mps_mean"),
        "moving_gate_speed_mps": summary.get("moving_gate_speed_mps"),
        "moving_gate_amplitude_m": summary.get("moving_gate_amplitude_m"),
        "corridor_through_success_rate": summary.get("corridor_through_success_rate"),
        "side_bypass_failure_rate": summary.get("side_bypass_failure_rate"),
        "actual_gate_motion_range_m": summary.get("actual_gate_motion_range_m_mean"),
        "height_contract_passed_rate": summary.get("height_contract_passed_rate", 1.0),
        "height_out_of_bounds_rate": summary.get("height_escape_failure_rate", 0.0),
        "dispersed_termination_rate": 0.0,
    }


def _mainline_single_progress_distance(
    summary: dict[str, Any],
    full_route_distance_m: float,
    success_rate: float,
    path_length_m_mean: float,
) -> float | None:
    for key in ("progress_distance_mean_m", "progress_distance_m_mean", "progress_distance_m"):
        value = _float_or_nan(summary.get(key))
        if math.isfinite(value):
            return max(0.0, min(float(full_route_distance_m), value))
    goal_distance = _float_or_nan(summary.get("mean_goal_distance_m"))
    if math.isfinite(goal_distance):
        return max(0.0, min(float(full_route_distance_m), float(full_route_distance_m) - goal_distance))
    if math.isfinite(success_rate) and success_rate >= 0.999:
        return float(full_route_distance_m)
    # Older paper_2d summaries do not store final goal distance.  Use executed
    # path length as a bounded fallback so missing data is not reported as zero
    # progress.  This is diagnostic, not a replacement for full-route success.
    if math.isfinite(path_length_m_mean):
        return max(0.0, min(float(full_route_distance_m), path_length_m_mean))
    return None


def _mainline_metrics_from_multi(row: dict[str, Any]) -> dict[str, Any]:
    obstacle_collision_rate = _float_or_nan(row.get("obstacle_collision_rate"))
    agent_agent_collision_rate = _float_or_nan(row.get("agent_agent_collision_rate"))
    out_of_bounds_rate = _float_or_nan(row.get("out_of_bounds_rate"))
    side_bypass_rate = _float_or_nan(row.get("side_bypass_failure_rate"))
    collision_parts = [value for value in (obstacle_collision_rate, agent_agent_collision_rate) if math.isfinite(value)]
    safety_parts = [value for value in (obstacle_collision_rate, agent_agent_collision_rate, out_of_bounds_rate, side_bypass_rate) if math.isfinite(value)]
    team_success_rate = _float_or_nan(row.get("team_success_rate", row.get("success_rate")))
    mainline_path_length = _float_or_nan(row.get("path_length_m_mean"))
    return {
        "success_rate": row.get("team_success_rate", row.get("success_rate")),
        "team_success_rate": row.get("team_success_rate"),
        "per_agent_success_rate": row.get("per_agent_success_rate"),
        "collision_rate": max(collision_parts) if collision_parts else None,
        "crash_rate": max(collision_parts) if collision_parts else None,
        "obstacle_collision_rate": row.get("obstacle_collision_rate"),
        "agent_agent_collision_rate": row.get("agent_agent_collision_rate"),
        "timeout_rate": row.get("timeout_rate"),
        "planning_failure_rate": 0.0,
        "no_path_rate": 0.0,
        "progress_distance_mean_m": row.get("progress_distance_mean_m"),
        "progress_ratio": (
            _float_or_nan(row.get("progress_distance_mean_m")) / 54.0
            if math.isfinite(_float_or_nan(row.get("progress_distance_mean_m")))
            else None
        ),
        "full_route_distance_m": 54.0,
        "required_route_distance_m": 54.0,
        "path_length_m_mean": row.get("path_length_m_mean") if math.isfinite(team_success_rate) and team_success_rate >= 0.999 else None,
        "flown_path_length_m_mean": row.get("path_length_m_mean"),
        "path_length_ratio_success_only": (
            mainline_path_length / 54.0
            if math.isfinite(mainline_path_length) and math.isfinite(team_success_rate) and team_success_rate >= 0.999
            else None
        ),
        "flight_time_s_mean": row.get("flight_time_s_mean"),
        "mean_speed_mps": row.get("mean_speed_mps"),
        "max_speed_mps": row.get("max_speed_mps"),
        "corridor_through_success_rate": row.get("corridor_through_success_rate"),
        "side_bypass_failure_rate": row.get("side_bypass_failure_rate"),
        "out_of_bounds_rate": row.get("out_of_bounds_rate"),
        "hard_failure_rate": row.get("hard_failure_rate"),
        "safety_violation_rate": row.get("safety_violation_rate", max(safety_parts) if safety_parts else None),
        "actual_gate_motion_range_m": row.get("actual_gate_motion_range_m_mean"),
        "height_contract_passed_rate": row.get("height_contract_passed_rate"),
        "height_out_of_bounds_rate": row.get("height_escape_failure_rate"),
        "min_pair_distance_mean_m": row.get("min_pair_distance_mean_m"),
        "formation_slot_error_mean_m": row.get("formation_slot_error_mean_m"),
        "formation_slot_error_max_m": row.get("formation_slot_error_max_m"),
        "dispersed_termination_rate": row.get("dispersed_termination_rate"),
    }


def _mainline_prefixed(metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"mainline_{key}": value for key, value in metrics.items()}


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["experiment"], row["scenario_kind"], row["method"], row["team_size"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (experiment, scenario_kind, method, team_size), group in sorted(groups.items()):
        summary.append(
            {
                "experiment": experiment,
                "scenario_kind": scenario_kind,
                "method": method,
                "team_size": team_size,
                "scenario_count": len(group),
                "training": "none",
                "policy_checkpoint": "none",
                "learning": "none",
                "success_rate": _mean(float(row.get("success_rate", 0.0)) for row in group),
                "team_success_rate": _mean(float(row.get("team_success_rate", row.get("success_rate", 0.0))) for row in group),
                "per_agent_success_rate": _mean(float(row.get("per_agent_success_rate", row.get("success_rate", 0.0))) for row in group),
                "collision_rate": _mean(float(row.get("collision_rate", 0.0)) for row in group),
                "crash_rate": _mean(float(row.get("crash_rate", row.get("collision_rate", 0.0))) for row in group),
                "obstacle_collision_rate": _mean(float(row.get("obstacle_collision_rate", row.get("collision_rate", 0.0))) for row in group),
                "agent_agent_collision_rate": _mean(float(row.get("agent_agent_collision_rate", 0.0)) for row in group),
                "timeout_rate": _mean(float(row.get("timeout_rate", 0.0)) for row in group),
                "planning_failure_rate": _mean(float(row.get("planning_failure_rate", 0.0)) for row in group),
                "no_path_rate": _mean(float(row.get("no_path_rate", 0.0)) for row in group),
                "out_of_bounds_rate": _mean(float(row.get("out_of_bounds_rate", 0.0)) for row in group),
                "hard_failure_rate": _mean(float(row.get("hard_failure_rate", 1.0 - float(row.get("success_rate", 0.0)))) for row in group),
                "safety_violation_rate": _mean(float(row.get("safety_violation_rate", 0.0)) for row in group),
                "progress_distance_mean_m": _mean(float(row.get("progress_distance_mean_m", 0.0)) for row in group),
                "progress_ratio": _mean(float(row.get("progress_ratio", 0.0)) for row in group),
                "full_route_distance_m": _mean(float(row.get("full_route_distance_m", 0.0)) for row in group),
                "remaining_goal_distance_m": _mean(float(row.get("remaining_goal_distance_m", 0.0)) for row in group),
                "path_length_m_mean": _mean_or_none(row.get("path_length_m", row.get("path_length_m_mean")) for row in group),
                "path_length_m_success_only": _mean_or_none(row.get("path_length_m_success_only", row.get("path_length_m")) for row in group),
                "flown_path_length_m_mean": _mean(float(row.get("flown_path_length_m", row.get("flown_path_length_m_mean", 0.0))) for row in group),
                "failed_episode_flown_path_length_m": _mean_or_none(row.get("failed_episode_flown_path_length_m") for row in group),
                "flown_path_length_ratio": _mean(float(row.get("flown_path_length_ratio", 0.0)) for row in group),
                "path_length_ratio": _mean_or_none(row.get("path_length_ratio") for row in group),
                "path_length_ratio_success_only": _mean_or_none(row.get("path_length_ratio_success_only") for row in group),
                "flight_time_s_mean": _mean(float(row.get("flight_time_s", row.get("flight_time_s_mean", 0.0))) for row in group),
                "mean_speed_mps": _mean(float(row.get("mean_speed_mps", 0.0)) for row in group),
                "max_speed_mps": _mean(float(row.get("max_speed_mps", 0.0)) for row in group),
                "configured_max_speed_mps": _mean(float(row.get("configured_max_speed_mps", row.get("drone_speed_limit_mps", 0.0))) for row in group),
                "drone_speed_limit_mps": _mean(float(row.get("drone_speed_limit_mps", row.get("configured_max_speed_mps", 0.0))) for row in group),
                "mean_latency_ms": _mean(float(row.get("mean_latency_ms", 0.0)) for row in group),
                "p95_latency_ms": _percentile([float(row.get("p95_latency_ms", 0.0)) for row in group], 0.95),
                "min_clearance_m": min(float(row.get("min_clearance_m", float("inf"))) for row in group),
                "replan_count": _mean(float(row.get("replan_count", 0.0)) for row in group),
                "planner_call_count": _mean(float(row.get("planner_call_count", 0.0)) for row in group),
                "planner_failure_count": _mean(float(row.get("planner_failure_count", 0.0)) for row in group),
                "full_route_success_rate": _mean(float(row.get("full_route_success_rate", row.get("success_rate", 0.0))) for row in group),
                "corridor_through_success_rate": _mean(float(row.get("corridor_through_success_rate", 0.0)) for row in group),
                "side_bypass_failure_rate": _mean(float(row.get("side_bypass_failure_rate", 0.0)) for row in group),
                "height_contract_passed_rate": _mean(float(row.get("height_contract_passed_rate", 1.0)) for row in group),
                "height_out_of_bounds_rate": _mean(float(row.get("height_out_of_bounds_rate", 0.0)) for row in group),
                "min_pair_distance_mean_m": _mean_or_none(row.get("min_pair_distance_mean_m") for row in group),
                "formation_slot_error_mean_m": _mean_or_none(row.get("formation_slot_error_mean_m") for row in group),
                "formation_slot_error_max_m": _mean_or_none(row.get("formation_slot_error_max_m") for row in group),
                "dispersed_termination_rate": _mean(float(row.get("dispersed_termination_rate", 0.0)) for row in group),
                "dynamic_moving_scenario_rate": _mean(1.0 if row.get("dynamic_gate_really_moves") else 0.0 for row in group),
                "actual_gate_motion_range_m": _mean(float(row.get("actual_gate_motion_range_m", 0.0)) for row in group),
                "moving_gate_speed_mps": _mean(float(row.get("moving_gate_speed_mps", 0.0)) for row in group),
                "moving_gate_amplitude_m": _mean(float(row.get("moving_gate_amplitude_m", 0.0)) for row in group),
                "gate_contact_terminates_episode_rate": _mean(1.0 if row.get("gate_contact_terminates_episode") else 0.0 for row in group),
                "mp4_generated_rate": _mean(1.0 if row.get("mp4_generated") else 0.0 for row in group),
                "video_accepted_rate": _mean(1.0 if row.get("video_accepted") else 0.0 for row in group),
            }
        )
    return summary


def summarize_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["experiment"], row["scenario_kind"], row["method"], row["gate_count"], row["team_size"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (experiment, scenario_kind, method, gate_count, team_size), group in sorted(groups.items()):
        planner_success = _mean(float(row.get("success_rate", 0.0)) for row in group)
        mainline_success = _mean(_float_or_nan(row.get("mainline_success_rate")) for row in group)
        planner_collision = _mean(float(row.get("collision_rate", 0.0)) for row in group)
        mainline_collision = _mean(_float_or_nan(row.get("mainline_collision_rate")) for row in group)
        planner_hard_failure = _mean(float(row.get("hard_failure_rate", 1.0 - float(row.get("success_rate", 0.0)))) for row in group)
        mainline_hard_failure = _mean(_float_or_nan(row.get("mainline_hard_failure_rate")) for row in group)
        summary.append(
            {
                "experiment": experiment,
                "scenario_kind": scenario_kind,
                "gate_count": gate_count,
                "team_size": team_size,
                "method": method,
                "training": "none",
                "policy_checkpoint": "none",
                "learning": "none",
                "completed_mainline_source": str(group[0].get("completed_mainline_source", "")),
                "scenario_count": len(group),
                "planner_success_rate": planner_success,
                "completed_mainline_success_rate": mainline_success,
                "success_rate_delta_planner_minus_mainline": planner_success - mainline_success if math.isfinite(mainline_success) else None,
                "planner_collision_rate": planner_collision,
                "completed_mainline_collision_rate": mainline_collision,
                "collision_rate_delta_planner_minus_mainline": planner_collision - mainline_collision if math.isfinite(mainline_collision) else None,
                "planner_crash_rate": _mean(float(row.get("crash_rate", row.get("collision_rate", 0.0))) for row in group),
                "completed_mainline_crash_rate": _mean(_float_or_nan(row.get("mainline_crash_rate")) for row in group),
                "planner_obstacle_collision_rate": _mean(float(row.get("obstacle_collision_rate", row.get("collision_rate", 0.0))) for row in group),
                "completed_mainline_obstacle_collision_rate": _mean(_float_or_nan(row.get("mainline_obstacle_collision_rate")) for row in group),
                "planner_agent_agent_collision_rate": _mean(float(row.get("agent_agent_collision_rate", 0.0)) for row in group),
                "completed_mainline_agent_agent_collision_rate": _mean(_float_or_nan(row.get("mainline_agent_agent_collision_rate")) for row in group),
                "planner_timeout_rate": _mean(float(row.get("timeout_rate", 0.0)) for row in group),
                "completed_mainline_timeout_rate": _mean(_float_or_nan(row.get("mainline_timeout_rate")) for row in group),
                "planner_planning_failure_rate": _mean(float(row.get("planning_failure_rate", 0.0)) for row in group),
                "planner_no_path_rate": _mean(float(row.get("no_path_rate", 0.0)) for row in group),
                "planner_out_of_bounds_rate": _mean(float(row.get("out_of_bounds_rate", 0.0)) for row in group),
                "planner_hard_failure_rate": planner_hard_failure,
                "completed_mainline_hard_failure_rate": mainline_hard_failure,
                "hard_failure_delta_planner_minus_mainline": planner_hard_failure - mainline_hard_failure if math.isfinite(mainline_hard_failure) else None,
                "planner_safety_violation_rate": _mean(float(row.get("safety_violation_rate", 0.0)) for row in group),
                "completed_mainline_safety_violation_rate": _mean(_float_or_nan(row.get("mainline_safety_violation_rate")) for row in group),
                "planner_progress_distance_mean_m": _mean(float(row.get("progress_distance_mean_m", 0.0)) for row in group),
                "completed_mainline_progress_distance_mean_m": _mean(_float_or_nan(row.get("mainline_progress_distance_mean_m")) for row in group),
                "planner_progress_ratio": _mean(float(row.get("progress_ratio", 0.0)) for row in group),
                "completed_mainline_progress_ratio": _mean(_float_or_nan(row.get("mainline_progress_ratio")) for row in group),
                "planner_full_route_distance_m": _mean(float(row.get("full_route_distance_m", 0.0)) for row in group),
                "completed_mainline_full_route_distance_m": _mean(_float_or_nan(row.get("mainline_full_route_distance_m")) for row in group),
                "planner_path_length_m_success_only": _mean_or_none(row.get("path_length_m_success_only", row.get("path_length_m")) for row in group),
                "completed_mainline_path_length_m_success_only": _mean_or_none(row.get("mainline_path_length_m_mean") for row in group),
                "planner_flown_path_length_m_mean": _mean(float(row.get("flown_path_length_m", row.get("flown_path_length_m_mean", 0.0))) for row in group),
                "completed_mainline_flown_path_length_m_mean": _mean_or_none(row.get("mainline_flown_path_length_m_mean") for row in group),
                "planner_failed_episode_flown_path_length_m": _mean_or_none(row.get("failed_episode_flown_path_length_m") for row in group),
                "planner_path_length_ratio_success_only": _mean_or_none(row.get("path_length_ratio_success_only") for row in group),
                "completed_mainline_path_length_ratio_success_only": _mean_or_none(row.get("mainline_path_length_ratio_success_only") for row in group),
                "planner_flight_time_s_mean": _mean(float(row.get("flight_time_s", row.get("flight_time_s_mean", 0.0))) for row in group),
                "completed_mainline_flight_time_s_mean": _mean(_float_or_nan(row.get("mainline_flight_time_s_mean")) for row in group),
                "planner_mean_speed_mps": _mean(float(row.get("mean_speed_mps", 0.0)) for row in group),
                "completed_mainline_mean_speed_mps": _mean(_float_or_nan(row.get("mainline_mean_speed_mps")) for row in group),
                "planner_max_speed_mps": _mean(float(row.get("max_speed_mps", 0.0)) for row in group),
                "completed_mainline_max_speed_mps": _mean(_float_or_nan(row.get("mainline_max_speed_mps")) for row in group),
                "planner_configured_max_speed_mps": _mean(float(row.get("configured_max_speed_mps", row.get("drone_speed_limit_mps", 0.0))) for row in group),
                "planner_drone_speed_limit_mps": _mean(float(row.get("drone_speed_limit_mps", row.get("configured_max_speed_mps", 0.0))) for row in group),
                "planner_mean_latency_ms": _mean(float(row.get("mean_latency_ms", 0.0)) for row in group),
                "planner_p95_latency_ms": _percentile([float(row.get("p95_latency_ms", 0.0)) for row in group], 0.95),
                "planner_min_clearance_m": min(float(row.get("min_clearance_m", float("inf"))) for row in group),
                "planner_replan_count": _mean(float(row.get("replan_count", 0.0)) for row in group),
                "planner_full_route_success_rate": _mean(float(row.get("full_route_success_rate", row.get("success_rate", 0.0))) for row in group),
                "completed_mainline_full_route_success_rate": _mean(_float_or_nan(row.get("mainline_full_route_success_rate")) for row in group),
                "planner_start_to_goal_complete_rate": _mean(float(row.get("start_to_goal_complete_rate", row.get("success_rate", 0.0))) for row in group),
                "completed_mainline_start_to_goal_complete_rate": _mean(_float_or_nan(row.get("mainline_start_to_goal_complete_rate")) for row in group),
                "planner_corridor_through_success_rate": _mean(float(row.get("corridor_through_success_rate", 0.0)) for row in group),
                "completed_mainline_corridor_through_success_rate": _mean(_float_or_nan(row.get("mainline_corridor_through_success_rate")) for row in group),
                "planner_side_bypass_failure_rate": _mean(float(row.get("side_bypass_failure_rate", 0.0)) for row in group),
                "completed_mainline_side_bypass_failure_rate": _mean(_float_or_nan(row.get("mainline_side_bypass_failure_rate")) for row in group),
                "planner_height_contract_passed_rate": _mean(float(row.get("height_contract_passed_rate", 1.0)) for row in group),
                "completed_mainline_height_contract_passed_rate": _mean(_float_or_nan(row.get("mainline_height_contract_passed_rate")) for row in group),
                "planner_height_out_of_bounds_rate": _mean(float(row.get("height_out_of_bounds_rate", 0.0)) for row in group),
                "completed_mainline_height_out_of_bounds_rate": _mean(_float_or_nan(row.get("mainline_height_out_of_bounds_rate")) for row in group),
                "planner_min_pair_distance_mean_m": _mean_or_none(row.get("min_pair_distance_mean_m") for row in group),
                "completed_mainline_min_pair_distance_mean_m": _mean_or_none(row.get("mainline_min_pair_distance_mean_m") for row in group),
                "planner_formation_slot_error_mean_m": _mean_or_none(row.get("formation_slot_error_mean_m") for row in group),
                "completed_mainline_formation_slot_error_mean_m": _mean_or_none(row.get("mainline_formation_slot_error_mean_m") for row in group),
                "planner_formation_slot_error_max_m": _mean_or_none(row.get("formation_slot_error_max_m") for row in group),
                "completed_mainline_formation_slot_error_max_m": _mean_or_none(row.get("mainline_formation_slot_error_max_m") for row in group),
                "planner_dispersed_termination_rate": _mean(float(row.get("dispersed_termination_rate", 0.0)) for row in group),
                "completed_mainline_dispersed_termination_rate": _mean(_float_or_nan(row.get("mainline_dispersed_termination_rate")) for row in group),
                "dynamic_gate_really_moves_rate": _mean(1.0 if row.get("dynamic_gate_really_moves") else 0.0 for row in group),
                "planner_actual_gate_motion_range_m": _mean(float(row.get("actual_gate_motion_range_m", 0.0)) for row in group),
                "completed_mainline_actual_gate_motion_range_m": _mean(_float_or_nan(row.get("mainline_actual_gate_motion_range_m")) for row in group),
                "planner_moving_gate_speed_mps": _mean(float(row.get("moving_gate_speed_mps", 0.0)) for row in group),
                "planner_moving_gate_amplitude_m": _mean(float(row.get("moving_gate_amplitude_m", 0.0)) for row in group),
                "planner_gate_contact_terminates_episode_rate": _mean(1.0 if row.get("gate_contact_terminates_episode") else 0.0 for row in group),
                "planner_collision_terminates_episode_rate": _mean(1.0 if row.get("collision_terminates_episode") else 0.0 for row in group),
                "planner_mp4_generated_rate": _mean(1.0 if row.get("mp4_generated") else 0.0 for row in group),
                "planner_video_accepted_rate": _mean(1.0 if row.get("video_accepted") else 0.0 for row in group),
            }
        )
    return summary


def _smoke_selection(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    desired = {
        ("E1_static_single_gate_density", 6, 0),
        ("E2_dynamic_single_gate_density", 42, 0),
        ("E4_static_multi_8d", 6, 0),
        ("E5_dynamic_multi_8d", 42, 0),
    }
    selected = [row for row in scenarios if (row["experiment"], row["gate_count"], row["seed"]) in desired]
    if selected:
        return selected
    return scenarios[: min(4, len(scenarios))]


def _limit_per_experiment(scenarios: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    result = []
    for scenario in scenarios:
        experiment = str(scenario["experiment"])
        if counts.get(experiment, 0) >= limit:
            continue
        counts[experiment] = counts.get(experiment, 0) + 1
        result.append(scenario)
    return result


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({key: value for key, value in row.items() if not key.startswith("_")}, ensure_ascii=False, allow_nan=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: value for key, value in row.items() if not key.startswith("_")}, ensure_ascii=False, allow_nan=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=True)
    return value


def _tuple2(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _mean(values: Any) -> float:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.fmean(data)) if data else 0.0


def _mean_or_none(values: Any) -> float | None:
    data = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            data.append(number)
    return float(statistics.fmean(data)) if data else None


def _percentile(values: list[float], q: float) -> float:
    data = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not data:
        return 0.0
    index = min(len(data) - 1, max(0, int(round((len(data) - 1) * float(q)))))
    return float(data[index])


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()

