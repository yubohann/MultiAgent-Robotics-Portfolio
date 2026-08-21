"""Closed-loop method evaluation for experiment-2 variants."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from single_internal_gate.ablation import MethodVariant
from single_internal_gate.configs.experiment_config import EXP2_SINGLE_INTERNAL_CONFIG, Exp2SingleInternalConfig
from single_internal_gate.evaluation import make_tasks
from single_internal_gate.planners import create_planner
from single_internal_gate.planners.interfaces import PlannerTask2D
from single_internal_gate.policies.arbitration import UncertaintyAwareArbitrator
from single_internal_gate.policies.reactive_policy import ReactivePolicy2D
from single_internal_gate.safety.shield import SafetyShield2D


@dataclass(frozen=True)
class MethodEpisodeResult:
    method_name: str
    task_id: str
    success: bool
    collision: bool
    path_xy: tuple[tuple[float, float], ...]
    travel_time_s: float
    path_length_m: float
    min_clearance_m: float
    mean_latency_ms: float
    p95_latency_ms: float
    replan_count: int
    shield_interventions: int
    latency_violation: bool


@dataclass(frozen=True)
class MethodMetrics:
    method_name: str
    episodes: int
    success_rate: float
    collision_rate: float
    normalized_path_length: float
    normalized_travel_time: float
    min_clearance: float
    mean_latency: float
    p95_latency: float
    replan_count: float
    shield_interventions: float
    latency_violation_rate: float


def evaluate_method_variant(
    variant: MethodVariant,
    *,
    episodes: int,
    config: Exp2SingleInternalConfig = EXP2_SINGLE_INTERNAL_CONFIG,
    task_suite: str = "gate",
) -> MethodMetrics:
    tasks = make_tasks(episodes, config, task_suite=task_suite)
    return evaluate_method_variant_on_tasks(variant, tasks, config=config)


def evaluate_method_variant_on_tasks(
    variant: MethodVariant,
    tasks: tuple[PlannerTask2D, ...],
    *,
    config: Exp2SingleInternalConfig = EXP2_SINGLE_INTERNAL_CONFIG,
) -> MethodMetrics:
    results = tuple(_run_episode(task, variant, config) for task in tasks)
    return summarize_method_results(variant.name, tasks, results, latency_budget_ms=config.planner.latency_budget_ms)


def summarize_method_results(
    method_name: str,
    tasks: tuple[PlannerTask2D, ...],
    results: tuple[MethodEpisodeResult, ...],
    *,
    latency_budget_ms: float,
) -> MethodMetrics:
    straight_lengths = [max(_distance(task.start_xy, task.goal_xy), 1e-6) for task in tasks]
    straight_times = [length / max(EXP2_SINGLE_INTERNAL_CONFIG.method.max_speed_mps, 1e-6) for length in straight_lengths]
    successful = [result for result in results if result.success]
    normalized_lengths = [
        result.path_length_m / straight_lengths[index]
        for index, result in enumerate(results)
        if result.success
    ]
    normalized_times = [
        result.travel_time_s / straight_times[index]
        for index, result in enumerate(results)
        if result.success
    ]
    return MethodMetrics(
        method_name=method_name,
        episodes=len(results),
        success_rate=len(successful) / max(len(results), 1),
        collision_rate=sum(result.collision for result in results) / max(len(results), 1),
        normalized_path_length=_mean_or_inf(normalized_lengths),
        normalized_travel_time=_mean_or_inf(normalized_times),
        min_clearance=min((result.min_clearance_m for result in results), default=float("inf")),
        mean_latency=statistics.fmean([result.mean_latency_ms for result in results]) if results else 0.0,
        p95_latency=_percentile([result.p95_latency_ms for result in results], 0.95),
        replan_count=statistics.fmean([result.replan_count for result in results]) if results else 0.0,
        shield_interventions=statistics.fmean([result.shield_interventions for result in results]) if results else 0.0,
        latency_violation_rate=sum(result.latency_violation for result in results) / max(len(results), 1),
    )


def _run_episode(task: PlannerTask2D, variant: MethodVariant, config: Exp2SingleInternalConfig) -> MethodEpisodeResult:
    method_cfg = config.method
    reactive = ReactivePolicy2D(method_cfg)
    shield = SafetyShield2D(method_cfg)
    arbitrator = UncertaintyAwareArbitrator(method_cfg)
    planner_result = create_planner(variant.planner_name, config.planner).plan(task) if variant.use_planner else None
    planner_path = list(planner_result.path_xy) if planner_result and planner_result.success else [task.start_xy, task.goal_xy]
    position = task.start_xy
    path = [position]
    waypoint_index = 1 if len(planner_path) > 1 else 0
    previous_goal_distance = _distance(position, task.goal_xy)
    latencies = [planner_result.planning_time_ms] if planner_result is not None else []
    replan_count = 0
    shield_interventions = 0
    min_clearance = task.obstacles_2d.min_signed_distance(position, drone_radius_m=task.drone_radius_m)
    collision = False

    for step in range(method_cfg.max_steps):
        if _distance(position, task.goal_xy) <= method_cfg.goal_tolerance_m:
            return _episode_result(variant.name, task, path, step, method_cfg.dt_s, min_clearance, latencies, replan_count, shield_interventions, True, False, config.planner.latency_budget_ms)

        planner_target = _current_waypoint(position, planner_path, waypoint_index, method_cfg.waypoint_tolerance_m)
        waypoint_index = planner_target[1]
        planner_command = _command_towards(position, planner_target[0], method_cfg.max_speed_mps)
        reactive_command = reactive.command(position_xy=position, goal_xy=task.goal_xy, task=task)

        decision = arbitrator.decide(
            task=task,
            position_xy=position,
            planner_target_xy=planner_target[0],
            goal_xy=task.goal_xy,
            step_index=step,
            previous_goal_distance_m=previous_goal_distance,
        )
        if variant.event_triggered_replanning and decision.should_replan and variant.use_planner:
            replan_count += 1
            replanned = create_planner(variant.planner_name, config.planner).plan(_task_from_position(task, position, replan_count))
            latencies.append(replanned.planning_time_ms)
            if replanned.success:
                planner_path = list(replanned.path_xy)
                waypoint_index = 1 if len(planner_path) > 1 else 0

        if variant.fixed_planner_weight is not None:
            planner_weight = variant.fixed_planner_weight
        elif variant.use_uncertainty_arbitration:
            planner_weight = decision.planner_weight
        else:
            planner_weight = 0.5
        if not variant.use_planner:
            planner_weight = 0.0
        if not variant.use_reactive:
            planner_weight = 1.0
        command = (
            planner_command[0] * planner_weight + reactive_command[0] * (1.0 - planner_weight),
            planner_command[1] * planner_weight + reactive_command[1] * (1.0 - planner_weight),
        )
        if decision.shield_only and variant.use_shield:
            command = reactive_command
        if variant.use_shield:
            command, intervened = shield.filter_command(position_xy=position, command_xy=command, task=task)
            shield_interventions += int(intervened)
        latency = method_cfg.latency_arbitration_ms
        latency += method_cfg.latency_reactive_ms if variant.use_reactive else 0.0
        latency += method_cfg.latency_shield_ms if variant.use_shield else 0.0
        latencies.append(latency)

        next_position = (position[0] + command[0] * method_cfg.dt_s, position[1] + command[1] * method_cfg.dt_s)
        collision = task.obstacles_2d.segment_collides(position, next_position, drone_radius_m=task.drone_radius_m)
        path.append(next_position)
        position = next_position
        min_clearance = min(min_clearance, task.obstacles_2d.min_signed_distance(position, drone_radius_m=task.drone_radius_m))
        if collision:
            return _episode_result(variant.name, task, path, step + 1, method_cfg.dt_s, min_clearance, latencies, replan_count, shield_interventions, False, True, config.planner.latency_budget_ms)
        previous_goal_distance = _distance(position, task.goal_xy)

    return _episode_result(variant.name, task, path, method_cfg.max_steps, method_cfg.dt_s, min_clearance, latencies, replan_count, shield_interventions, False, False, config.planner.latency_budget_ms)


def _episode_result(
    method_name: str,
    task: PlannerTask2D,
    path: list[tuple[float, float]],
    steps: int,
    dt_s: float,
    min_clearance: float,
    latencies: list[float],
    replan_count: int,
    shield_interventions: int,
    success: bool,
    collision: bool,
    latency_budget_ms: float,
) -> MethodEpisodeResult:
    return MethodEpisodeResult(
        method_name=method_name,
        task_id=task.task_id,
        success=success,
        collision=collision,
        path_xy=tuple(path),
        travel_time_s=steps * dt_s,
        path_length_m=sum(_distance(a, b) for a, b in zip(path[:-1], path[1:])),
        min_clearance_m=min_clearance,
        mean_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 0.95),
        replan_count=replan_count,
        shield_interventions=shield_interventions,
        latency_violation=any(latency > latency_budget_ms for latency in latencies),
    )


def _current_waypoint(position: tuple[float, float], path: list[tuple[float, float]], index: int, tolerance: float) -> tuple[tuple[float, float], int]:
    if not path:
        return position, 0
    current = min(max(index, 0), len(path) - 1)
    while current < len(path) - 1 and _distance(position, path[current]) <= tolerance:
        current += 1
    return path[current], current


def _command_towards(position: tuple[float, float], target: tuple[float, float], max_speed: float) -> tuple[float, float]:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return (0.0, 0.0)
    speed = min(max_speed, distance / max(EXP2_SINGLE_INTERNAL_CONFIG.method.dt_s, 1e-6))
    return (dx / distance * speed, dy / distance * speed)


def _task_from_position(task: PlannerTask2D, position: tuple[float, float], replan_count: int) -> PlannerTask2D:
    return PlannerTask2D(
        start_xy=position,
        goal_xy=task.goal_xy,
        obstacles_2d=task.obstacles_2d,
        fixed_height_m=task.fixed_height_m,
        task_id=f"{task.task_id}_replan_{replan_count}",
        drone_radius_m=task.drone_radius_m,
        world_x_bounds_m=task.world_x_bounds_m,
        world_y_bounds_m=task.world_y_bounds_m,
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _mean_or_inf(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("inf")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]

