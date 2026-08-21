"""Configuration for the lightweight experiment-2 internal 2D method stack."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.configs.global_config import GLOBAL_CONFIG


@dataclass(frozen=True)
class Exp2EnvironmentConfig:
    fixed_height_m: float = GLOBAL_CONFIG.fixed_flight_height_m
    drone_radius_m: float = 0.35
    start_x_m: float = -18.0
    goal_x_m: float = 18.0
    start_y_range_m: tuple[float, float] = (-6.0, 6.0)
    goal_y_range_m: tuple[float, float] = (6.0, -6.0)
    world_x_bounds_m: tuple[float, float] = (-24.0, 24.0)
    world_y_bounds_m: tuple[float, float] = (-12.0, 12.0)


@dataclass(frozen=True)
class Exp2PlannerConfig:
    latency_budget_ms: float = 50.0
    straight_latency_ms: float = 2.0
    detour_latency_ms: float = 8.0
    safety_margin_m: float = 0.7


@dataclass(frozen=True)
class Exp2MethodConfig:
    dt_s: float = 0.1
    max_speed_mps: float = 3.5
    max_steps: int = 240
    goal_tolerance_m: float = 0.85
    waypoint_tolerance_m: float = 0.75
    obstacle_query_radius_m: float = 3.2
    reactive_repulsion_gain: float = 1.8
    shield_lookahead_s: float = 0.35
    latency_arbitration_ms: float = 1.0
    latency_reactive_ms: float = 1.5
    latency_shield_ms: float = 0.8


@dataclass(frozen=True)
class Exp2SingleInternalConfig:
    environment: Exp2EnvironmentConfig = field(default_factory=Exp2EnvironmentConfig)
    planner: Exp2PlannerConfig = field(default_factory=Exp2PlannerConfig)
    method: Exp2MethodConfig = field(default_factory=Exp2MethodConfig)


EXP2_SINGLE_INTERNAL_CONFIG = Exp2SingleInternalConfig()

