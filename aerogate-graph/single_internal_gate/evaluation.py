"""Planner-only evaluation metrics for experiment 2."""

from __future__ import annotations

from dataclasses import dataclass
import statistics

from single_internal_gate.configs.experiment_config import EXP2_SINGLE_INTERNAL_CONFIG, Exp2SingleInternalConfig
from single_internal_gate.planners.interfaces import PlannerResult, PlannerTask2D
from shared.core.collision_2d import GateObstacleMap2D
from shared.task_suites.exp12_gate_scene import make_exp2_gate_tasks, make_race50_gate_tasks, task_suite_names


@dataclass(frozen=True)
class PlannerMetrics:
    planner_name: str
    episodes: int
    success_rate: float
    collision_rate: float
    normalized_path_length: float
    normalized_travel_time: float
    min_clearance: float
    mean_latency: float
    p95_latency: float
    replan_count: float
    latency_violation_rate: float


def make_tasks(
    count: int,
    config: Exp2SingleInternalConfig = EXP2_SINGLE_INTERNAL_CONFIG,
    *,
    task_suite: str = "gate",
) -> tuple[PlannerTask2D, ...]:
    normalized_task_suite = str(task_suite).strip().lower().replace("-", "_")
    if normalized_task_suite == "gate":
        return make_exp2_gate_tasks(count, config)
    if normalized_task_suite in {"race50", "race50_gate", "gate50"}:
        return make_race50_gate_tasks(count, config)
    if normalized_task_suite != "gate":
        raise ValueError(f"Unsupported experiment-2 task suite: {task_suite}. Expected one of {task_suite_names()}.")

    obstacle_map = GateObstacleMap2D.from_gate()
    start_min, start_max = config.environment.start_y_range_m
    goal_min, goal_max = config.environment.goal_y_range_m
    tasks = []
    for index in range(count):
        denom = max(count - 1, 1)
        start_y = start_min + (start_max - start_min) * index / denom
        goal_y = goal_max - (goal_max - goal_min) * index / denom
        tasks.append(
            PlannerTask2D(
                start_xy=(config.environment.start_x_m, start_y),
                goal_xy=(config.environment.goal_x_m, goal_y),
                obstacles_2d=obstacle_map,
                fixed_height_m=config.environment.fixed_height_m,
                task_id=f"exp2_default_{index:03d}",
                drone_radius_m=config.environment.drone_radius_m,
                world_x_bounds_m=config.environment.world_x_bounds_m,
                world_y_bounds_m=config.environment.world_y_bounds_m,
            )
        )
    return tuple(tasks)


def make_default_tasks(count: int, config: Exp2SingleInternalConfig = EXP2_SINGLE_INTERNAL_CONFIG) -> tuple[PlannerTask2D, ...]:
    return make_tasks(count, config, task_suite="gate")


def summarize_results(
    planner_name: str,
    tasks: tuple[PlannerTask2D, ...],
    results: tuple[PlannerResult, ...],
    *,
    latency_budget_ms: float,
) -> PlannerMetrics:
    if len(tasks) != len(results):
        raise ValueError("tasks and results must have the same length")
    successes = [result for result in results if result.success]
    straight_lengths = [max(_straight_distance(task), 1e-6) for task in tasks]
    normalized_lengths = [
        result.path_length_m / straight_lengths[index]
        for index, result in enumerate(results)
        if result.success
    ]
    latencies = [result.planning_time_ms for result in results]
    clearances = [_path_min_clearance(task, result) for task, result in zip(tasks, results) if result.success]
    return PlannerMetrics(
        planner_name=planner_name,
        episodes=len(results),
        success_rate=len(successes) / max(len(results), 1),
        collision_rate=sum(_path_collides(task, result) for task, result in zip(tasks, results)) / max(len(results), 1),
        normalized_path_length=_mean_or_inf(normalized_lengths),
        normalized_travel_time=_mean_or_inf(normalized_lengths),
        min_clearance=min(clearances) if clearances else float("inf"),
        mean_latency=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency=_percentile(latencies, 0.95),
        replan_count=statistics.fmean([result.num_replans for result in results]) if results else 0.0,
        latency_violation_rate=sum(latency > latency_budget_ms for latency in latencies) / max(len(latencies), 1),
    )


def _straight_distance(task: PlannerTask2D) -> float:
    return ((task.goal_xy[0] - task.start_xy[0]) ** 2 + (task.goal_xy[1] - task.start_xy[1]) ** 2) ** 0.5


def _path_collides(task: PlannerTask2D, result: PlannerResult) -> bool:
    if not result.path_xy:
        return False
    return any(
        task.obstacles_2d.segment_collides(a, b, drone_radius_m=task.drone_radius_m)
        for a, b in zip(result.path_xy[:-1], result.path_xy[1:])
    )


def _path_min_clearance(task: PlannerTask2D, result: PlannerResult) -> float:
    if not result.path_xy:
        return float("inf")
    return min(task.obstacles_2d.min_signed_distance(point, drone_radius_m=task.drone_radius_m) for point in result.path_xy)


def _mean_or_inf(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("inf")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]

