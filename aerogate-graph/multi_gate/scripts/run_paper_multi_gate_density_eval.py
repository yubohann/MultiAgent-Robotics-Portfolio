"""Run resumable paper multi-drone E4/E7/E8 gate-density evaluations.

The row worker builds its environment through
``run_dynamic_gate_density_8d_curriculum._stage_config`` so static, dynamic,
team-size, and geometry-pressure evaluations share the same live gate layout
and collision implementation as E5 training/eval/replay.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


GATE_AXIS: tuple[int, ...] = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
TEAM_SIZE_AXIS: tuple[int, ...] = (2, 3, 5, 7, 8, 9)
DEFAULT_DRONE_SPEED_MPS = 3.50
HISTORICAL_CHECKPOINT_PATH_MARKERS: tuple[str, ...] = (
    "dynamic_gate_density_8d_c4a_24g_speed05_typefix",
    "gate_density_imitation_bridge_C4_C8_v1",
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
    "team_size",
    "seed",
    "episodes",
    "speed_mps",
    "amplitude_m",
    "drone_speed_mps",
    "drone_accel_mps2",
    "drone_radius_m",
    "gate_post_radius_scale",
    "gate_half_width_scale",
    "team_success_rate",
    "per_agent_success_rate",
    "obstacle_collision_rate",
    "gate_post_collision_rate",
    "dynamic_gate_collision_rate",
    "agent_agent_collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "height_contract_passed_rate",
    "corridor_through_success_rate",
    "side_bypass_failure_rate",
    "height_escape_failure_rate",
    "corridor_miss_failure_rate",
    "formation_line_collapse_failure_rate",
    "hard_failure_rate",
    "dispersed_termination_rate",
    "safety_violation_rate",
    "progress_distance_mean_m",
    "mean_goal_distance_m",
    "mean_steps",
    "path_length_m_mean",
    "flight_time_s_mean",
    "mean_speed_mps",
    "max_speed_mps",
    "formation_slot_error_mean_m",
    "formation_slot_error_max_m",
    "formation_lateral_band_count_min",
    "formation_line_collapse_score_mean",
    "min_pair_distance_mean_m",
    "min_pair_distance_min_m",
    "min_clearance_mean_m",
    "min_clearance_min_m",
    "actual_gate_motion_range_m_mean",
    "actual_gate_motion_range_m_max",
    "planner_call_count_mean",
    "planner_latency_ms_mean",
    "shield_activation_count_mean",
    "shield_activation_ratio_mean",
    "shield_intervention_norm_mean",
    "guidance_query_count_mean",
    "guidance_non_fallback_rate",
    "mean_guidance_latency_ms",
    "guidance_cache_hit_rate",
    "done_reason_counts",
    "checkpoint_path",
    "output_dir",
    "returncode",
    "duration_s",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _shared_eval_drone_speed_axis_mps() -> tuple[float, ...]:
    root = _root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import eval_drone_speed_axis_mps

    return eval_drone_speed_axis_mps()


def _drone_accel_for_speed(speed_mps: float, override_mps2: float | None = None) -> float:
    root = _root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import drone_accel_limit_for_speed_mps2

    return drone_accel_limit_for_speed_mps2(speed_mps, override_mps2)


def _validate_drone_command_limits(args: argparse.Namespace) -> None:
    root = _root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from shared.core.dynamic_gate_density_2d import MAX_DRONE_COMMAND_ACCEL_MPS2, MAX_DRONE_COMMAND_SPEED_MPS

    speeds = [float(args.drone_speed_mps), *(float(value) for value in args.drone_speed_axis_mps)]
    bad_speeds = [speed for speed in speeds if speed <= 0.0 or speed > MAX_DRONE_COMMAND_SPEED_MPS]
    if bad_speeds:
        raise SystemExit(f"drone speed values must be in (0, {MAX_DRONE_COMMAND_SPEED_MPS}]; got {bad_speeds}")
    if args.drone_accel_mps2 is not None and (
        float(args.drone_accel_mps2) <= 0.0 or float(args.drone_accel_mps2) > MAX_DRONE_COMMAND_ACCEL_MPS2
    ):
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


def _load_runner(root: Path) -> Any:
    runner_path = root / "multi_gate" / "scripts" / "run_dynamic_gate_density_8d_curriculum.py"
    spec = importlib.util.spec_from_file_location("paper_dynamic_gate_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["paper_dynamic_gate_runner"] = module
    spec.loader.exec_module(module)
    return module


def _scenario_for_e7(name: str) -> tuple[int, float, float]:
    if name == "empty":
        return 0, 0.0, 0.0
    if name == "static_30":
        return 30, 0.0, 0.0
    if name == "static_60":
        return 60, 0.0, 0.0
    if name == "dynamic_30_speed20":
        return 30, 2.0, AMPLITUDE_BY_GATE_COUNT_M[30]
    if name == "dynamic_24_speed10":
        return 24, 1.0, AMPLITUDE_BY_GATE_COUNT_M[24]
    if name == "dynamic_30_speed11":
        return 30, 1.1, AMPLITUDE_BY_GATE_COUNT_M[30]
    if name == "dynamic_60_speed20":
        return 60, 2.0, AMPLITUDE_BY_GATE_COUNT_M[60]
    raise ValueError(f"unknown E7 scenario: {name}")


def _scenario_for_e6(name: str) -> tuple[int, float, float]:
    if name == "static_60":
        return 60, 0.0, 0.0
    if name == "dynamic_24_speed10":
        return 24, 1.0, AMPLITUDE_BY_GATE_COUNT_M[24]
    if name == "dynamic_30_speed20":
        return 30, 2.0, AMPLITUDE_BY_GATE_COUNT_M[30]
    if name == "dynamic_60_speed20":
        return 60, 2.0, AMPLITUDE_BY_GATE_COUNT_M[60]
    raise ValueError(f"unknown E6 scenario: {name}")


def _scenario_for_e9(name: str) -> tuple[int, float, float]:
    if name == "empty":
        return 0, 0.0, 0.0
    if name == "static_30":
        return 30, 0.0, 0.0
    if name == "static_60":
        return 60, 0.0, 0.0
    if name == "dynamic_30_speed20":
        return 30, 2.0, AMPLITUDE_BY_GATE_COUNT_M[30]
    if name == "dynamic_42_speed14":
        return 42, 1.4, AMPLITUDE_BY_GATE_COUNT_M[42]
    if name == "dynamic_60_speed20":
        return 60, 2.0, AMPLITUDE_BY_GATE_COUNT_M[60]
    raise ValueError(f"unknown E9 speed-gradient scenario: {name}")


def _method_overrides(config: Any, method: str) -> Any:
    if method == "full":
        return config
    env = config.environment
    reasoning = config.reasoning
    if method == "no_guidance":
        return replace(
            config,
            reasoning=replace(
                reasoning,
                route_guidance_enabled=False,
                guidance_shadow_mode=False,
                guidance_provider="none",
            ),
        )
    if method == "guidance_shadow":
        return replace(
            config,
            reasoning=replace(
                reasoning,
                route_guidance_enabled=False,
                guidance_shadow_mode=True,
                guidance_async_enabled=True,
                guidance_cache_enabled=True,
                guidance_provider="local_http",
                guidance_timeout_s=10.0,
            ),
        )
    if method == "guidance_visible":
        return replace(
            config,
            reasoning=replace(
                reasoning,
                route_guidance_enabled=True,
                guidance_shadow_mode=False,
                guidance_async_enabled=True,
                guidance_cache_enabled=True,
                guidance_provider="local_http",
                guidance_timeout_s=10.0,
            ),
        )
    if method == "no_shield":
        return replace(
            config,
            environment=replace(
                env,
                action_safety_shield_enabled=False,
                action_safety_shield_separation_margin_m=0.0,
                action_safety_shield_boundary_margin_m=0.0,
                action_safety_shield_guidance_margin_m=0.0,
                action_safety_shield_obstacle_margin_m=0.0,
            ),
        )
    if method == "no_slow_planner":
        return replace(config, reasoning=replace(reasoning, global_planner_enabled=False))
    if method == "fast_only":
        return replace(
            config,
            reasoning=replace(
                reasoning,
                global_planner_enabled=False,
                route_guidance_enabled=False,
                guidance_shadow_mode=False,
                guidance_provider="none",
            ),
            environment=replace(
                env,
                action_safety_shield_enabled=False,
                action_safety_shield_separation_margin_m=0.0,
                action_safety_shield_boundary_margin_m=0.0,
                action_safety_shield_guidance_margin_m=0.0,
                action_safety_shield_obstacle_margin_m=0.0,
            ),
        )
    if method == "planner_only":
        return replace(
            config,
            reasoning=replace(
                reasoning,
                global_planner_enabled=True,
                route_guidance_enabled=False,
                guidance_shadow_mode=False,
                guidance_provider="none",
            ),
        )
    raise ValueError(f"unknown method: {method}")


def _build_config(
    *,
    runner: Any,
    experiment: str,
    scenario: str,
    method: str,
    gate_count: int,
    speed_mps: float,
    amplitude_m: float,
    drone_speed_mps: float,
    drone_accel_mps2: float,
    drone_radius_m: float | None,
    gate_post_radius_scale: float,
    gate_half_width_scale: float,
    min_agents: int,
) -> Any:
    from multi_gate.configs import get_multi_experiment_config

    base = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    base = replace(base, min_agents=min_agents, default_agents=max(8, min_agents))
    gate_cfg = replace(
        base.dynamic_gate_density,
        gate_post_radius_m=float(base.dynamic_gate_density.gate_post_radius_m) * float(gate_post_radius_scale),
        gate_half_width_m=float(base.dynamic_gate_density.gate_half_width_m) * float(gate_half_width_scale),
        drone_radius_m=(
            float(base.dynamic_gate_density.drone_radius_m)
            if drone_radius_m is None
            else float(drone_radius_m)
        ),
    )
    env_cfg = replace(
        base.environment,
        drone_radius_m=(
            float(base.environment.drone_radius_m)
            if drone_radius_m is None
            else float(drone_radius_m)
        ),
    )
    base = replace(base, dynamic_gate_density=gate_cfg, environment=env_cfg)
    stage = runner.DynamicGateCurriculumStage(
        # The E2D2 prefix is intentional: the curriculum runner uses it to keep
        # the 0-gate baseline on the straight dynamic-gate paper route instead
        # of silently switching back to the demo8 morph route.
        f"E2D2_{experiment}_{scenario}_gate{gate_count:02d}",
        int(gate_count),
        float(speed_mps),
        float(amplitude_m),
        0,
        6.0,
        0.22,
        0.35,
        0.0,
        0.60,
        0.66,
        float(drone_speed_mps),
        float(drone_accel_mps2),
        "paper eval row",
    )
    config = runner._stage_config(base, stage)
    return _method_overrides(config, method)


def _flatten_row(
    *,
    experiment: str,
    scenario: str,
    method: str,
    gate_count: int,
    team_size: int,
    seed: int,
    episodes: int,
    speed_mps: float,
    amplitude_m: float,
    drone_speed_mps: float,
    drone_accel_mps2: float,
    drone_radius_m: float | None,
    gate_post_radius_scale: float,
    gate_half_width_scale: float,
    checkpoint_path: Path,
    output_dir: Path,
    summary: dict[str, Any],
    returncode: int,
    duration_s: float,
) -> dict[str, Any]:
    start_x = -27.0
    goal_x = 27.0
    mean_goal_distance = summary.get("mean_goal_distance_m")
    progress_distance = None
    if mean_goal_distance is not None:
        progress_distance = max(0.0, abs(goal_x - start_x) - float(mean_goal_distance))
    row = {
        "experiment": experiment,
        "scenario": scenario,
        "method": method,
        "gate_count": int(gate_count),
        "team_size": int(team_size),
        "seed": int(seed),
        "episodes": int(episodes),
        "speed_mps": float(speed_mps),
        "amplitude_m": float(amplitude_m),
        "drone_speed_mps": float(drone_speed_mps),
        "drone_accel_mps2": float(drone_accel_mps2),
        "drone_radius_m": None if drone_radius_m is None else float(drone_radius_m),
        "gate_post_radius_scale": float(gate_post_radius_scale),
        "gate_half_width_scale": float(gate_half_width_scale),
        "team_success_rate": summary.get("success_rate"),
        "per_agent_success_rate": summary.get("per_agent_success_rate", summary.get("success_rate")),
        "obstacle_collision_rate": summary.get("obstacle_collision_rate", summary.get("gate_post_collision_rate")),
        "gate_post_collision_rate": summary.get("gate_post_collision_rate"),
        "dynamic_gate_collision_rate": summary.get("dynamic_gate_collision_rate"),
        "agent_agent_collision_rate": summary.get("agent_collision_rate"),
        "out_of_bounds_rate": summary.get("out_of_bounds_rate"),
        "timeout_rate": summary.get("timeout_rate"),
        "height_contract_passed_rate": summary.get("height_contract_passed_rate"),
        "corridor_through_success_rate": summary.get("corridor_through_success_rate"),
        "side_bypass_failure_rate": summary.get("side_bypass_failure_rate"),
        "height_escape_failure_rate": summary.get("height_escape_failure_rate"),
        "corridor_miss_failure_rate": summary.get("corridor_miss_failure_rate"),
        "formation_line_collapse_failure_rate": summary.get("formation_line_collapse_failure_rate"),
        "hard_failure_rate": summary.get("hard_failure_rate"),
        "dispersed_termination_rate": summary.get("dispersed_termination_rate"),
        "safety_violation_rate": summary.get("safety_violation_rate"),
        "progress_distance_mean_m": summary.get("progress_distance_mean_m", progress_distance),
        "mean_goal_distance_m": summary.get("mean_goal_distance_m"),
        "mean_steps": summary.get("mean_steps"),
        "path_length_m_mean": summary.get("path_length_m_mean"),
        "flight_time_s_mean": summary.get("flight_time_s_mean"),
        "mean_speed_mps": summary.get("mean_speed_mps"),
        "max_speed_mps": summary.get("max_speed_mps"),
        "formation_slot_error_mean_m": summary.get("mean_slot_error_m"),
        "formation_slot_error_max_m": summary.get(
            "max_max_slot_error_m",
            summary.get("mean_max_slot_error_m", summary.get("max_slot_error_m_mean")),
        ),
        "formation_lateral_band_count_min": summary.get("min_formation_lateral_band_count"),
        "formation_line_collapse_score_mean": summary.get("mean_formation_line_collapse_score"),
        "min_pair_distance_mean_m": summary.get("mean_min_pair_distance_m"),
        "min_pair_distance_min_m": summary.get("min_min_pair_distance_m"),
        "min_clearance_mean_m": summary.get("mean_min_clearance_m"),
        "min_clearance_min_m": summary.get("min_min_clearance_m"),
        "actual_gate_motion_range_m_mean": summary.get("mean_actual_gate_motion_range_m"),
        "actual_gate_motion_range_m_max": summary.get("max_actual_gate_motion_range_m"),
        "planner_call_count_mean": summary.get("planner_call_count_mean"),
        "planner_latency_ms_mean": summary.get("planner_latency_ms_mean"),
        "shield_activation_count_mean": summary.get("shield_activation_count_mean"),
        "shield_activation_ratio_mean": summary.get("shield_activation_ratio_mean"),
        "shield_intervention_norm_mean": summary.get("shield_intervention_norm_mean"),
        "guidance_query_count_mean": summary.get("guidance_query_count_mean"),
        "guidance_non_fallback_rate": summary.get("guidance_non_fallback_rate"),
        "mean_guidance_latency_ms": summary.get("mean_guidance_latency_ms"),
        "guidance_cache_hit_rate": summary.get("guidance_cache_hit_rate"),
        "done_reason_counts": json.dumps(summary.get("done_reason_counts") or {}, ensure_ascii=False, sort_keys=True),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "returncode": int(returncode),
        "duration_s": float(duration_s),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "team_success_rate",
        "per_agent_success_rate",
        "obstacle_collision_rate",
        "gate_post_collision_rate",
        "dynamic_gate_collision_rate",
        "agent_agent_collision_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "height_contract_passed_rate",
        "corridor_through_success_rate",
        "side_bypass_failure_rate",
        "height_escape_failure_rate",
        "corridor_miss_failure_rate",
        "formation_line_collapse_failure_rate",
        "dispersed_termination_rate",
        "progress_distance_mean_m",
        "path_length_m_mean",
        "flight_time_s_mean",
        "mean_speed_mps",
        "max_speed_mps",
        "formation_slot_error_mean_m",
        "formation_slot_error_max_m",
        "formation_lateral_band_count_min",
        "formation_line_collapse_score_mean",
        "min_pair_distance_min_m",
        "min_clearance_min_m",
        "actual_gate_motion_range_m_mean",
        "planner_call_count_mean",
        "planner_latency_ms_mean",
        "shield_activation_count_mean",
        "shield_activation_ratio_mean",
        "shield_intervention_norm_mean",
        "guidance_query_count_mean",
        "guidance_non_fallback_rate",
        "mean_guidance_latency_ms",
    )
    grouped: dict[tuple[str, str, str, int, int, float, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("returncode") or 0) != 0:
            continue
        key = (
            str(row["experiment"]),
            str(row["scenario"]),
            str(row["method"]),
            int(row["gate_count"]),
            int(row["team_size"]),
            float(row["gate_post_radius_scale"]),
            float(row["gate_half_width_scale"]),
            float(row.get("drone_radius_m") or 0.0),
            float(row.get("drone_speed_mps") or DEFAULT_DRONE_SPEED_MPS),
            float(row.get("drone_accel_mps2") or _drone_accel_for_speed(float(row.get("drone_speed_mps") or DEFAULT_DRONE_SPEED_MPS))),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        experiment, scenario, method, gate_count, team_size, radius_scale, width_scale, drone_radius, drone_speed, drone_accel = key
        summary: dict[str, Any] = {
            "experiment": experiment,
            "scenario": scenario,
            "method": method,
            "gate_count": gate_count,
            "team_size": team_size,
            "gate_post_radius_scale": radius_scale,
            "gate_half_width_scale": width_scale,
            "drone_radius_m": drone_radius if drone_radius > 0.0 else None,
            "drone_speed_mps": drone_speed,
            "drone_accel_mps2": drone_accel,
            "seed_count": len({int(item["seed"]) for item in items}),
            "episodes_total": sum(int(item.get("episodes") or 0) for item in items),
            "speed_mps": items[0].get("speed_mps"),
            "amplitude_m": items[0].get("amplitude_m"),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in items if item.get(metric) is not None]
            summary[f"{metric}_mean"] = _mean(vals)
            summary[f"{metric}_min"] = min(vals) if vals else None
            summary[f"{metric}_max"] = max(vals) if vals else None
        out.append(summary)

    # Bucket summaries for E7.
    e7_rows = [row for row in rows if str(row.get("experiment")) == "E7_team_size_buckets" and int(row.get("returncode") or 0) == 0]
    by_scenario: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in e7_rows:
        by_scenario.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)
    for (scenario, method), items in sorted(by_scenario.items()):
        bucket_values: list[float] = []
        for team_size in sorted({int(item["team_size"]) for item in items}):
            team_items = [item for item in items if int(item["team_size"]) == team_size]
            vals = [float(item["team_success_rate"]) for item in team_items if item.get("team_success_rate") is not None]
            if vals:
                bucket_values.append(sum(vals) / len(vals))
        if bucket_values:
            out.append(
                {
                    "experiment": "E7_team_size_buckets_bucket_summary",
                    "scenario": scenario,
                    "method": method,
                    "gate_count": None,
                    "team_size": None,
                    "seed_count": len({int(item["seed"]) for item in items}),
                    "episodes_total": sum(int(item.get("episodes") or 0) for item in items),
                    "min_bucket_success_rate": min(bucket_values),
                    "max_bucket_success_rate": max(bucket_values),
                    "bucket_gap": max(bucket_values) - min(bucket_values),
                }
            )
    return out


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _finite_mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else None


def _speed_samples_from_info(info: dict[str, Any]) -> list[float]:
    velocities = info.get("agent_velocities_xy")
    if velocities is None:
        return []
    try:
        array = np.asarray(velocities, dtype=np.float32)
    except (TypeError, ValueError):
        return []
    if array.ndim != 2 or array.shape[-1] != 2:
        return []
    speeds = np.linalg.norm(array, axis=1)
    return [float(value) for value in speeds.reshape(-1) if np.isfinite(value)]


def _corridor_through_success_from_info(info: dict[str, Any]) -> bool:
    dynamic_gate_count = int(info.get("dynamic_gate_count") or 0)
    if dynamic_gate_count <= 0:
        return True
    return bool(
        bool(info.get("height_contract_passed", True))
        and not bool(info.get("height_escape_failure", False))
        and bool(info.get("corridor_completed", False))
        and not bool(info.get("side_bypass_failure", False))
        and not bool(info.get("corridor_miss_failure", False))
    )


def _episode_success_from_info(info: dict[str, Any], *, timeout_counts_as_success: bool) -> bool:
    reason = str(info.get("done_reason") or "")
    raw_success = reason == "goal_reached" or bool(timeout_counts_as_success and reason == "timeout")
    if not raw_success:
        return False
    return bool(
        bool(info.get("height_contract_passed", True))
        and not bool(info.get("height_escape_failure", False))
        and not bool(info.get("side_bypass_failure", False))
        and not bool(info.get("corridor_miss_failure", False))
        and not bool(info.get("formation_line_collapse_failure", False))
        and _corridor_through_success_from_info(info)
    )


def _per_agent_success_fraction_from_info(info: dict[str, Any], *, timeout_counts_as_success: bool) -> float:
    if _episode_success_from_info(info, timeout_counts_as_success=timeout_counts_as_success):
        return 1.0
    hard_failure_reasons = {
        "gate_post_collision",
        "agent_collision",
        "out_of_bounds",
        "height_escape_failure",
        "side_bypass_failure",
        "corridor_miss_failure",
        "formation_line_collapse_failure",
    }
    if str(info.get("done_reason") or "") in hard_failure_reasons:
        return 0.0
    if (
        not bool(info.get("height_contract_passed", True))
        or bool(info.get("height_escape_failure", False))
        or bool(info.get("side_bypass_failure", False))
        or bool(info.get("corridor_miss_failure", False))
        or bool(info.get("formation_line_collapse_failure", False))
    ):
        return 0.0
    if int(info.get("dynamic_gate_count") or 0) > 0 and not bool(info.get("corridor_completed", False)):
        return 0.0
    positions = info.get("agent_positions_xy")
    if positions is None:
        return 0.0
    try:
        positions_array = np.asarray(positions, dtype=np.float32)
    except (TypeError, ValueError):
        return 0.0
    if positions_array.ndim != 2 or positions_array.shape[1] != 2 or positions_array.shape[0] <= 0:
        return 0.0
    goal_xy_raw = info.get("goal_xy")
    if goal_xy_raw is None:
        path_waypoints = info.get("path_waypoints")
        if path_waypoints:
            goal_xy_raw = list(path_waypoints)[-1]
    if goal_xy_raw is None:
        return 0.0
    goal_xy = np.asarray(goal_xy_raw, dtype=np.float32)
    if goal_xy.shape != (2,):
        return 0.0
    goal_radius_m = max(float(info.get("goal_radius_m") or 0.0), 1.0e-6)
    distances = np.linalg.norm(positions_array - goal_xy.reshape(1, 2), axis=1)
    return float(np.count_nonzero(distances <= goal_radius_m) / max(int(positions_array.shape[0]), 1))


def _dispersed_termination_from_info(info: dict[str, Any]) -> bool:
    reason = str(info.get("done_reason") or "").lower()
    return any(token in reason for token in ("dispers", "formation", "slot_error", "max_slot", "line_collapse"))


def _evaluate_planner_only(
    *,
    experiment_config: Any,
    episodes: int,
    seed: int,
    num_agents: int,
) -> dict[str, Any]:
    from multi_gate.env.multi_gate_env import MultiGate2DEnv

    env = MultiGate2DEnv(
        multi_config=experiment_config,
        env_config=experiment_config.environment,
        observation_config=experiment_config.observation,
        formation_config=experiment_config.formation,
        planner_config=experiment_config.planner,
    )
    done_reason_counts: dict[str, int] = {}
    episode_summaries: list[dict[str, Any]] = []
    successes = 0
    timeout_counts_as_success = bool(getattr(experiment_config.environment, "timeout_counts_as_success", False))
    dt_s = float(getattr(experiment_config.environment, "dt_s", 0.1) or 0.1)

    try:
        for episode_idx in range(int(episodes)):
            env.reset(seed=int(seed) + episode_idx, num_agents=int(num_agents))
            previous_center_xy = np.asarray(env.snapshot().virtual_center_xy, dtype=np.float32)
            path_length_m = 0.0
            episode_speed_samples_mps: list[float] = []
            shield_active_steps = 0
            shield_intervention_norms: list[float] = []
            total_reward = 0.0
            step_count = 0
            last_info: dict[str, Any] = {}
            while True:
                positions = env.active_positions_xy()
                slots = env.desired_slots_xy()
                heading = np.asarray(env.current_heading_xy(), dtype=np.float32)
                heading_norm = float(np.linalg.norm(heading))
                if heading_norm <= 1.0e-6:
                    heading = np.asarray((1.0, 0.0), dtype=np.float32)
                else:
                    heading = heading / heading_norm
                forward_speed = float(
                    min(
                        getattr(env, "_resolved_forward_command_speed_mps")(),
                        getattr(env, "_resolved_max_command_speed_mps")(),
                    )
                )
                slot_gain = 0.65
                desired_velocities = heading.reshape(1, 2) * forward_speed + (slots - positions) * slot_gain
                max_speed = max(float(getattr(env, "_resolved_max_command_speed_mps")()), 1.0e-6)
                action = np.zeros(env.action_shape, dtype=np.float32)
                for idx, velocity in enumerate(desired_velocities[: int(num_agents)]):
                    speed = float(np.linalg.norm(velocity))
                    clipped_velocity = velocity if speed <= max_speed else velocity * (max_speed / speed)
                    action[idx] = env.desired_velocity_to_action(clipped_velocity)

                _, reward, terminated, truncated, info = env.step(action)
                last_info = info
                total_reward += float(reward)
                step_count += 1
                episode_speed_samples_mps.extend(_speed_samples_from_info(info))

                current_center_raw = info.get("virtual_center_xy")
                if current_center_raw is not None:
                    current_center_xy = np.asarray(current_center_raw, dtype=np.float32)
                    if current_center_xy.shape == (2,):
                        path_length_m += float(np.linalg.norm(current_center_xy - previous_center_xy))
                        previous_center_xy = current_center_xy
                shield_info = info.get("action_safety_shield")
                if isinstance(shield_info, dict):
                    if bool(shield_info.get("active", False)):
                        shield_active_steps += 1
                    intervention_norm = _finite_float(shield_info.get("mean_intervention_norm"))
                    if intervention_norm is not None:
                        shield_intervention_norms.append(intervention_norm)

                if terminated or truncated:
                    reason = str(info.get("done_reason") or "unknown")
                    done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                    if _episode_success_from_info(info, timeout_counts_as_success=timeout_counts_as_success):
                        successes += 1
                    snapshot = info.get("snapshot")
                    max_slot_error_m = (
                        float(getattr(snapshot, "max_slot_error_m", info.get("max_slot_error_m") or 0.0))
                        if snapshot is not None
                        else _finite_float(info.get("max_slot_error_m"))
                    )
                    episode_summaries.append(
                        {
                            "episode_index": int(episode_idx),
                            "steps": int(step_count),
                            "episode_reward": float(total_reward),
                            "done_reason": reason,
                            "num_agents": int(info.get("num_agents") or num_agents),
                            "goal_distance_m": (
                                float(getattr(snapshot, "goal_distance_m", 0.0)) if snapshot is not None else 0.0
                            ),
                            "mean_slot_error_m": (
                                float(getattr(snapshot, "mean_slot_error_m", 0.0)) if snapshot is not None else 0.0
                            ),
                            "max_slot_error_m": max_slot_error_m,
                            "goal_distance_improvement_m": _finite_float(info.get("goal_distance_improvement_m")),
                            "goal_progress_ratio": _finite_float(info.get("goal_progress_ratio")),
                            "per_agent_success_fraction": _per_agent_success_fraction_from_info(
                                info,
                                timeout_counts_as_success=timeout_counts_as_success,
                            ),
                            "dispersed_termination": _dispersed_termination_from_info(info),
                            "guidance_tracking_error_m": _finite_float(info.get("guidance_tracking_error_m")),
                            "route_guidance_tracking_error_m": _finite_float(info.get("route_guidance_tracking_error_m")),
                            "guidance_latency_ms": _finite_float(info.get("guidance_latency_ms")),
                            "route_guidance_source": info.get("route_guidance_source"),
                            "guidance_cache_hit": info.get("guidance_cache_hit"),
                            "min_clearance_m": _finite_float(info.get("min_clearance_m")),
                            "min_pair_distance_m": _finite_float(info.get("min_pair_distance_m")),
                            "dynamic_gate_collision": bool(info.get("dynamic_gate_collision", False)),
                            "height_contract_passed": bool(info.get("height_contract_passed", True)),
                            "height_escape_failure": bool(info.get("height_escape_failure", False)),
                            "side_bypass_failure": bool(info.get("side_bypass_failure", False)),
                            "corridor_miss_failure": bool(info.get("corridor_miss_failure", False)),
                            "formation_line_collapse_failure": bool(
                                info.get("formation_line_collapse_failure", False)
                            ),
                            "formation_lateral_band_count": _finite_float(
                                info.get("formation_lateral_band_count")
                            ),
                            "formation_line_collapse_score": _finite_float(
                                info.get("formation_line_collapse_score")
                            ),
                            "corridor_completed": bool(info.get("corridor_completed", int(info.get("dynamic_gate_count") or 0) <= 0)),
                            "corridor_through_success": _corridor_through_success_from_info(info),
                            "actual_gate_motion_range_m": _finite_float(info.get("actual_gate_motion_range_m")),
                            "path_length_m": float(path_length_m),
                            "flight_time_s": float(step_count * dt_s),
                            "mean_speed_mps": _finite_mean(episode_speed_samples_mps),
                            "max_speed_mps": max(episode_speed_samples_mps) if episode_speed_samples_mps else None,
                            "shield_activation_count": int(shield_active_steps),
                            "shield_activation_ratio": float(shield_active_steps / max(step_count, 1)),
                            "shield_intervention_norm_mean": _finite_mean(shield_intervention_norms),
                            "guidance_query_count": 0,
                            "planner_call_count": int(last_info.get("planner_call_count") or 0),
                            "planner_latency_ms_mean": _finite_float(last_info.get("planner_latency_ms_mean")),
                        }
                    )
                    break
    finally:
        env.close()

    resolved_episodes = max(int(episodes), 1)
    collect = lambda key: [
        value
        for value in (_finite_float(summary.get(key)) for summary in episode_summaries)
        if value is not None
    ]
    max_slot_error_values = collect("max_slot_error_m")
    per_agent_success_values = collect("per_agent_success_fraction")
    progress_distance_values = collect("goal_distance_improvement_m")
    obstacle_collision_count = sum(
        1
        for summary in episode_summaries
        if str(summary.get("done_reason") or "") == "gate_post_collision"
        or summary.get("dynamic_gate_collision") is True
    )
    dispersed_termination_count = sum(
        1 for summary in episode_summaries if summary.get("dispersed_termination") is True
    )
    formation_line_collapse_failure_count = sum(
        1 for summary in episode_summaries if summary.get("formation_line_collapse_failure") is True
    )
    return {
        "episodes": int(episodes),
        "num_agents": int(num_agents),
        "success_rate": float(successes / resolved_episodes),
        "team_success_rate": float(successes / resolved_episodes),
        "per_agent_success_rate": _finite_mean(per_agent_success_values),
        "obstacle_collision_rate": float(obstacle_collision_count / resolved_episodes),
        "gate_post_collision_rate": float(done_reason_counts.get("gate_post_collision", 0) / resolved_episodes),
        "dynamic_gate_collision_rate": float(
            sum(1 for summary in episode_summaries if summary.get("dynamic_gate_collision") is True)
            / resolved_episodes
        ),
        "agent_collision_rate": float(done_reason_counts.get("agent_collision", 0) / resolved_episodes),
        "out_of_bounds_rate": float(done_reason_counts.get("out_of_bounds", 0) / resolved_episodes),
        "timeout_rate": float(done_reason_counts.get("timeout", 0) / resolved_episodes),
        "height_contract_passed_rate": float(
            sum(1 for summary in episode_summaries if summary.get("height_contract_passed") is True)
            / resolved_episodes
        ),
        "corridor_through_success_rate": float(
            sum(1 for summary in episode_summaries if summary.get("corridor_through_success") is True)
            / resolved_episodes
        ),
        "side_bypass_failure_rate": float(
            sum(1 for summary in episode_summaries if summary.get("side_bypass_failure") is True)
            / resolved_episodes
        ),
        "height_escape_failure_rate": float(
            sum(1 for summary in episode_summaries if summary.get("height_escape_failure") is True)
            / resolved_episodes
        ),
        "corridor_miss_failure_rate": float(
            sum(1 for summary in episode_summaries if summary.get("corridor_miss_failure") is True)
            / resolved_episodes
        ),
        "formation_line_collapse_failure_rate": float(
            formation_line_collapse_failure_count / resolved_episodes
        ),
        "hard_failure_rate": float(
            (
                done_reason_counts.get("gate_post_collision", 0)
                + done_reason_counts.get("agent_collision", 0)
                + done_reason_counts.get("out_of_bounds", 0)
                + formation_line_collapse_failure_count
            )
            / resolved_episodes
        ),
        "dispersed_termination_rate": float(dispersed_termination_count / resolved_episodes),
        "safety_violation_rate": 0.0,
        "mean_episode_reward": _finite_mean(collect("episode_reward")),
        "mean_steps": _finite_mean(collect("steps")),
        "mean_goal_distance_m": _finite_mean(collect("goal_distance_m")),
        "progress_distance_mean_m": _finite_mean(progress_distance_values),
        "mean_slot_error_m": _finite_mean(collect("mean_slot_error_m")),
        "mean_max_slot_error_m": _finite_mean(max_slot_error_values),
        "max_max_slot_error_m": max(max_slot_error_values) if max_slot_error_values else None,
        "mean_guidance_tracking_error_m": _finite_mean(collect("guidance_tracking_error_m")),
        "mean_route_guidance_tracking_error_m": _finite_mean(collect("route_guidance_tracking_error_m")),
        "mean_guidance_latency_ms": _finite_mean(collect("guidance_latency_ms")),
        "guidance_cache_hit_rate": 0.0,
        "guidance_non_fallback_rate": 0.0,
        "mean_min_clearance_m": _finite_mean(collect("min_clearance_m")),
        "min_min_clearance_m": min(collect("min_clearance_m")) if collect("min_clearance_m") else None,
        "mean_min_pair_distance_m": _finite_mean(collect("min_pair_distance_m")),
        "min_min_pair_distance_m": min(collect("min_pair_distance_m")) if collect("min_pair_distance_m") else None,
        "path_length_m_mean": _finite_mean(collect("path_length_m")),
        "flight_time_s_mean": _finite_mean(collect("flight_time_s")),
        "mean_speed_mps": _finite_mean(collect("mean_speed_mps")),
        "max_speed_mps": max(collect("max_speed_mps")) if collect("max_speed_mps") else None,
        "shield_activation_count_mean": _finite_mean(collect("shield_activation_count")),
        "shield_activation_ratio_mean": _finite_mean(collect("shield_activation_ratio")),
        "shield_intervention_norm_mean": _finite_mean(collect("shield_intervention_norm_mean")),
        "guidance_query_count_mean": _finite_mean(collect("guidance_query_count")),
        "planner_call_count_mean": _finite_mean(collect("planner_call_count")),
        "planner_latency_ms_mean": _finite_mean(collect("planner_latency_ms_mean")),
        "mean_actual_gate_motion_range_m": _finite_mean(collect("actual_gate_motion_range_m")),
        "max_actual_gate_motion_range_m": max(collect("actual_gate_motion_range_m"))
        if collect("actual_gate_motion_range_m")
        else None,
        "mean_formation_lateral_band_count": _finite_mean(collect("formation_lateral_band_count")),
        "min_formation_lateral_band_count": min(collect("formation_lateral_band_count"))
        if collect("formation_lateral_band_count")
        else None,
        "mean_formation_line_collapse_score": _finite_mean(collect("formation_line_collapse_score")),
        "done_reason_counts": done_reason_counts,
        "episode_summaries": episode_summaries,
        "timeout_counts_as_success": timeout_counts_as_success,
        "experiment_id": experiment_config.experiment_id,
    }


def _single_row(args: argparse.Namespace) -> None:
    root = _root()
    sys.path.insert(0, str(root))
    runner = _load_runner(root)
    from multi_gate.training import evaluate_checkpoint

    output_dir = Path(args.output_dir)
    row_path = output_dir / "row.json"
    summary_path = output_dir / "eval_summary.json"
    if row_path.exists() and not args.overwrite:
        print(row_path)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    drone_speed_mps = float(args.drone_speed_mps)
    drone_accel_mps2 = _drone_accel_for_speed(drone_speed_mps, args.drone_accel_mps2)

    config = _build_config(
        runner=runner,
        experiment=str(args.experiment),
        scenario=str(args.scenario),
        method=str(args.method),
        gate_count=int(args.gate_count),
        speed_mps=float(args.speed_mps),
        amplitude_m=float(args.amplitude_m),
        gate_post_radius_scale=float(args.gate_post_radius_scale),
        gate_half_width_scale=float(args.gate_half_width_scale),
        drone_radius_m=args.drone_radius_m,
        min_agents=int(args.min_agents),
        drone_speed_mps=drone_speed_mps,
        drone_accel_mps2=drone_accel_mps2,
    )
    start = time.perf_counter()
    try:
        if str(args.method) == "planner_only":
            summary = _evaluate_planner_only(
                experiment_config=config,
                episodes=int(args.episodes),
                seed=int(args.seed),
                num_agents=int(args.team_size),
            )
        else:
            summary = evaluate_checkpoint(
                checkpoint_path=Path(args.checkpoint),
                episodes=int(args.episodes),
                seed=int(args.seed),
                device=str(args.device) if args.device else None,
                num_agents=int(args.team_size),
                experiment_config=config,
            )
        returncode = 0
    except Exception as exc:
        summary = {
            "success_rate": None,
            "done_reason_counts": {},
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        returncode = 1
    duration_s = time.perf_counter() - start
    _write_json(summary_path, summary)
    row = _flatten_row(
        experiment=str(args.experiment),
        scenario=str(args.scenario),
        method=str(args.method),
        gate_count=int(args.gate_count),
        team_size=int(args.team_size),
        seed=int(args.seed),
        episodes=int(args.episodes),
        speed_mps=float(args.speed_mps),
        amplitude_m=float(args.amplitude_m),
        drone_speed_mps=drone_speed_mps,
        drone_accel_mps2=drone_accel_mps2,
        drone_radius_m=args.drone_radius_m,
        gate_post_radius_scale=float(args.gate_post_radius_scale),
        gate_half_width_scale=float(args.gate_half_width_scale),
        checkpoint_path=Path(args.checkpoint),
        output_dir=output_dir,
        summary=summary,
        returncode=returncode,
        duration_s=duration_s,
    )
    _write_json(row_path, row)
    print(row_path)
    if returncode != 0:
        raise SystemExit(returncode)


def _build_jobs(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    default_drone_speed = float(args.drone_speed_mps)
    default_drone_accel = _drone_accel_for_speed(default_drone_speed, args.drone_accel_mps2)
    for experiment in args.experiments:
        if experiment == "E4_static_multi_8d":
            for method in args.methods:
                for gate_count in args.gate_counts:
                    for seed in args.seeds:
                        out = args.output_root / experiment / method / f"gate_{gate_count:02d}_team_08_seed_{seed}"
                        jobs.append(
                            {
                                "experiment": experiment,
                                "scenario": "static_gate_density_8d",
                                "method": method,
                                "gate_count": int(gate_count),
                                "team_size": 8,
                                "seed": int(seed),
                                "speed_mps": 0.0,
                                "amplitude_m": 0.0,
                                "drone_speed_mps": default_drone_speed,
                                "drone_accel_mps2": default_drone_accel,
                                "gate_post_radius_scale": 1.0,
                                "gate_half_width_scale": 1.0,
                                "output_dir": out,
                            }
                        )
        elif experiment == "E5_dynamic_multi_8d":
            for method in args.methods:
                for gate_count in args.gate_counts:
                    speed_mps = float(SPEED_BY_GATE_COUNT_MPS[int(gate_count)])
                    amplitude_m = float(AMPLITUDE_BY_GATE_COUNT_M[int(gate_count)])
                    for seed in args.seeds:
                        out = args.output_root / experiment / method / f"gate_{gate_count:02d}_team_08_seed_{seed}"
                        jobs.append(
                            {
                                "experiment": experiment,
                                "scenario": "dynamic_gate_density_8d",
                                "method": method,
                                "gate_count": int(gate_count),
                                "team_size": 8,
                                "seed": int(seed),
                                "speed_mps": speed_mps,
                                "amplitude_m": amplitude_m,
                                "drone_speed_mps": default_drone_speed,
                                "drone_accel_mps2": default_drone_accel,
                                "gate_post_radius_scale": 1.0,
                                "gate_half_width_scale": 1.0,
                                "output_dir": out,
                            }
                        )
        elif experiment == "E6_ablation":
            for method in args.methods:
                for scenario in args.e6_scenarios:
                    gate_count, speed_mps, amplitude_m = _scenario_for_e6(str(scenario))
                    for seed in args.seeds:
                        out = (
                            args.output_root
                            / experiment
                            / method
                            / str(scenario)
                            / f"team_08_seed_{seed}"
                        )
                        jobs.append(
                            {
                                "experiment": experiment,
                                "scenario": str(scenario),
                                "method": method,
                                "gate_count": int(gate_count),
                                "team_size": 8,
                                "seed": int(seed),
                                "speed_mps": float(speed_mps),
                                "amplitude_m": float(amplitude_m),
                                "drone_speed_mps": default_drone_speed,
                                "drone_accel_mps2": default_drone_accel,
                                "gate_post_radius_scale": 1.0,
                                "gate_half_width_scale": 1.0,
                                "output_dir": out,
                            }
                        )
        elif experiment == "E7_team_size_buckets":
            for method in args.methods:
                for scenario in args.e7_scenarios:
                    gate_count, speed_mps, amplitude_m = _scenario_for_e7(str(scenario))
                    for team_size in args.team_sizes:
                        for seed in args.seeds:
                            out = (
                                args.output_root
                                / experiment
                                / method
                                / str(scenario)
                                / f"team_{int(team_size):02d}_seed_{seed}"
                            )
                            jobs.append(
                                {
                                    "experiment": experiment,
                                    "scenario": str(scenario),
                                    "method": method,
                                    "gate_count": int(gate_count),
                                    "team_size": int(team_size),
                                    "seed": int(seed),
                                    "speed_mps": float(speed_mps),
                                    "amplitude_m": float(amplitude_m),
                                    "drone_speed_mps": default_drone_speed,
                                    "drone_accel_mps2": default_drone_accel,
                                    "gate_post_radius_scale": 1.0,
                                    "gate_half_width_scale": 1.0,
                                    "output_dir": out,
                                }
                            )
        elif experiment == "E8_multi_geometry_pressure":
            for method in args.methods:
                for gate_count in args.e8_gate_counts:
                    for dynamic in args.e8_modes:
                        speed_mps = 2.0 if dynamic == "dynamic" and int(gate_count) > 0 else 0.0
                        amplitude_m = AMPLITUDE_BY_GATE_COUNT_M[int(gate_count)] if speed_mps > 0.0 else 0.0
                        scenario = f"{dynamic}_gate{int(gate_count):02d}"
                        for radius_scale in args.geometry_scales:
                            for team_size in args.team_sizes:
                                for seed in args.seeds:
                                    out = (
                                        args.output_root
                                        / experiment
                                        / method
                                        / scenario
                                        / f"radius_{float(radius_scale):.2f}_team_{int(team_size):02d}_seed_{seed}"
                                    )
                                    jobs.append(
                                        {
                                            "experiment": experiment,
                                            "scenario": scenario,
                                            "method": method,
                                            "gate_count": int(gate_count),
                                            "team_size": int(team_size),
                                            "seed": int(seed),
                                            "speed_mps": float(speed_mps),
                                            "amplitude_m": float(amplitude_m),
                                            "drone_speed_mps": default_drone_speed,
                                            "drone_accel_mps2": default_drone_accel,
                                            "gate_post_radius_scale": float(radius_scale),
                                            "gate_half_width_scale": 1.0,
                                            "output_dir": out,
                                        }
                                    )
        elif experiment == "E9_drone_speed_gradient":
            for method in args.methods:
                for scenario in args.e9_scenarios:
                    gate_count, speed_mps, amplitude_m = _scenario_for_e9(str(scenario))
                    for drone_speed in args.drone_speed_axis_mps:
                        drone_accel = _drone_accel_for_speed(float(drone_speed), args.drone_accel_mps2)
                        for seed in args.seeds:
                            out = (
                                args.output_root
                                / experiment
                                / method
                                / str(scenario)
                                / f"drone_{float(drone_speed):.2f}_seed_{seed}"
                            )
                            jobs.append(
                                {
                                    "experiment": experiment,
                                    "scenario": str(scenario),
                                    "method": method,
                                    "gate_count": int(gate_count),
                                    "team_size": 8,
                                    "seed": int(seed),
                                    "speed_mps": float(speed_mps),
                                    "amplitude_m": float(amplitude_m),
                                    "drone_speed_mps": float(drone_speed),
                                    "drone_accel_mps2": float(drone_accel),
                                    "gate_post_radius_scale": 1.0,
                                    "gate_half_width_scale": 1.0,
                                    "output_dir": out,
                                }
                            )
        else:
            raise ValueError(f"unknown experiment: {experiment}")
    if args.max_runs is not None:
        jobs = jobs[: int(args.max_runs)]
    return jobs


def _run_job(args: argparse.Namespace, job: dict[str, Any]) -> dict[str, Any]:
    row_path = Path(job["output_dir"]) / "row.json"
    if row_path.exists() and not args.overwrite:
        cached = json.loads(row_path.read_text(encoding="utf-8"))
        if int(cached.get("returncode", -1)) == 0 and cached.get("team_success_rate") is not None:
            return cached
    cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--single-row",
        "--checkpoint",
        str(args.checkpoint),
        "--experiment",
        str(job["experiment"]),
        "--scenario",
        str(job["scenario"]),
        "--method",
        str(job["method"]),
        "--gate-count",
        str(job["gate_count"]),
        "--team-size",
        str(job["team_size"]),
        "--seed",
        str(job["seed"]),
        "--episodes",
        str(args.episodes),
        "--speed-mps",
        str(job["speed_mps"]),
        "--amplitude-m",
        str(job["amplitude_m"]),
        "--drone-speed-mps",
        str(job["drone_speed_mps"]),
        "--drone-accel-mps2",
        str(job["drone_accel_mps2"]),
        "--gate-post-radius-scale",
        str(job["gate_post_radius_scale"]),
        "--gate-half-width-scale",
        str(job["gate_half_width_scale"]),
        "--min-agents",
        str(min(int(min(args.team_sizes or [8])), int(job["team_size"]), 8)),
        "--output-dir",
        str(job["output_dir"]),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.drone_radius_m is not None:
        cmd.extend(["--drone-radius-m", str(args.drone_radius_m)])
    if args.overwrite:
        cmd.append("--overwrite")
    if bool(getattr(args, "allow_historical_checkpoint", False)):
        cmd.append("--allow-historical-checkpoint")
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    if bool(getattr(args, "guidance_trace_console", False)):
        env = dict(os.environ)
        env["GATE2D_GUIDANCE_TRACE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(cmd, cwd=str(_root()), text=True, env=env)
    else:
        with (output_dir / "run_stdout.log").open("w", encoding="utf-8") as stdout, (
            output_dir / "run_stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(cmd, cwd=str(_root()), stdout=stdout, stderr=stderr, text=True)
    if row_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))
    return {
        "experiment": job["experiment"],
        "scenario": job["scenario"],
        "method": job["method"],
        "gate_count": job["gate_count"],
        "team_size": job["team_size"],
        "seed": job["seed"],
        "episodes": args.episodes,
        "speed_mps": job["speed_mps"],
        "amplitude_m": job["amplitude_m"],
        "drone_speed_mps": job.get("drone_speed_mps"),
        "drone_accel_mps2": job.get("drone_accel_mps2"),
        "drone_radius_m": args.drone_radius_m,
        "gate_post_radius_scale": job["gate_post_radius_scale"],
        "gate_half_width_scale": job["gate_half_width_scale"],
        "checkpoint_path": str(args.checkpoint),
        "output_dir": str(output_dir),
        "returncode": int(proc.returncode),
        "duration_s": 0.0,
    }


def main() -> None:
    root = _root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-row", action="store_true")
    parser.add_argument("--experiments", nargs="+", default=["E4_static_multi_8d"], choices=["E4_static_multi_8d", "E5_dynamic_multi_8d", "E6_ablation", "E7_team_size_buckets", "E8_multi_geometry_pressure", "E9_drone_speed_gradient"])
    parser.add_argument("--methods", nargs="+", default=["full"], choices=["full", "no_guidance", "guidance_shadow", "guidance_visible", "no_shield", "no_slow_planner", "fast_only", "planner_only"])
    parser.add_argument("--gate-counts", type=int, nargs="+", default=list(GATE_AXIS))
    parser.add_argument("--team-sizes", type=int, nargs="+", default=list(TEAM_SIZE_AXIS))
    parser.add_argument("--e6-scenarios", nargs="+", default=["dynamic_24_speed10", "dynamic_30_speed20", "static_60"])
    parser.add_argument("--e7-scenarios", nargs="+", default=["empty", "static_30", "static_60", "dynamic_30_speed20"])
    parser.add_argument("--e9-scenarios", nargs="+", default=["static_30", "dynamic_42_speed14", "dynamic_60_speed20"])
    parser.add_argument("--e8-gate-counts", type=int, nargs="+", default=[30, 60])
    parser.add_argument("--e8-modes", nargs="+", default=["static", "dynamic"], choices=["static", "dynamic"])
    parser.add_argument("--geometry-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--drone-speed-mps", type=float, default=DEFAULT_DRONE_SPEED_MPS)
    parser.add_argument("--drone-accel-mps2", type=float, default=None)
    parser.add_argument("--drone-radius-m", type=float, default=None)
    parser.add_argument("--drone-speed-axis-mps", type=float, nargs="+", default=list(_shared_eval_drone_speed_axis_mps()))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--guidance-trace-console", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--allow-historical-checkpoint", action="store_true")
    parser.add_argument("--output-root", type=Path, default=root / "results" / "paper_2d")

    # Single-row arguments.
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--scenario", type=str, default="")
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument("--gate-count", type=int, default=0)
    parser.add_argument("--team-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-mps", type=float, default=0.0)
    parser.add_argument("--amplitude-m", type=float, default=0.0)
    parser.add_argument("--gate-post-radius-scale", type=float, default=1.0)
    parser.add_argument("--gate-half-width-scale", type=float, default=1.0)
    parser.add_argument("--min-agents", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    _validate_drone_command_limits(args)
    _validate_checkpoint_for_formal_eval(args)

    if args.single_row:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --single-row")
        _single_row(args)
        return

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
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
            pool.submit(_run_job, args, job): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "experiment": job["experiment"],
                    "scenario": job["scenario"],
                    "method": job["method"],
                    "gate_count": job["gate_count"],
                    "team_size": job["team_size"],
                    "seed": job["seed"],
                    "episodes": args.episodes,
                    "stage_status": f"failed_exception:{type(exc).__name__}",
                    "checkpoint_path": str(args.checkpoint),
                    "output_dir": str(job["output_dir"]),
                    "returncode": -1,
                    "duration_s": 0.0,
                }
            rows.append(row)
            print(
                "[done] "
                f"{row.get('experiment')} {row.get('scenario')} {row.get('method')} "
                f"gate={row.get('gate_count')} team={row.get('team_size')} seed={row.get('seed')} "
                f"rc={row.get('returncode')} succ={row.get('team_success_rate')} "
                f"dyn={row.get('dynamic_gate_collision_rate')} agent={row.get('agent_agent_collision_rate')} "
                f"timeout={row.get('timeout_rate')}",
                flush=True,
            )

    for experiment in args.experiments:
        exp_rows = [row for row in rows if row.get("experiment") == experiment]
        if not exp_rows:
            continue
        exp_root = args.output_root / experiment
        _write_csv(
            exp_root / "formal_merged_latest.csv",
            sorted(exp_rows, key=lambda row: (str(row.get("scenario")), str(row.get("method")), int(row.get("gate_count") or 0), int(row.get("team_size") or 0), int(row.get("seed") or 0))),
        )
        _write_json(
            exp_root / "formal_merged_latest.json",
            {
                "gate_axis": list(GATE_AXIS),
                "team_size_axis": list(TEAM_SIZE_AXIS),
                "speed_by_gate_count_mps": SPEED_BY_GATE_COUNT_MPS,
                "amplitude_by_gate_count_m": AMPLITUDE_BY_GATE_COUNT_M,
                "rows": exp_rows,
            },
        )
        _write_json(exp_root / "formal_summary_latest.json", _summarize(exp_rows))
        print(f"[summary] {experiment} root={exp_root}", flush=True)


if __name__ == "__main__":
    main()

