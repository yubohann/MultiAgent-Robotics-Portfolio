"""Public scenario registry and dependency-light environment smoke runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from multi_gate.configs.experiment_config import MULTI_EXPERIMENT_CONFIG, build_dynamic_gate_density_8d_config
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from single_gate.env.single_gate_env import SingleGate2DEnv

ScenarioName = Literal["single-static", "multi-static", "multi-dynamic"]


@dataclass(frozen=True)
class Scenario:
    """A public task entry with its environment family and intended use."""

    name: ScenarioName
    mode: Literal["single", "multi"]
    dynamic_gates: bool
    description: str


@dataclass(frozen=True)
class RolloutSummary:
    """JSON-safe diagnostics from one deterministic, zero-action rollout."""

    scenario: ScenarioName
    seed: int
    steps_requested: int
    steps_executed: int
    reward_total: float
    terminated: bool
    truncated: bool
    done_reason: str | None
    observation_nodes: int
    observation_feature_dim: int
    num_agents: int
    goal_distance_m: float
    clearance_m: float | None
    min_pair_distance_m: float | None
    mean_slot_error_m: float | None
    max_slot_error_m: float | None
    finite_reward: bool

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON representation for CLIs, CI, and experiment logs."""

        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "steps_requested": self.steps_requested,
            "steps_executed": self.steps_executed,
            "reward_total": self.reward_total,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done_reason": self.done_reason,
            "observation_nodes": self.observation_nodes,
            "observation_feature_dim": self.observation_feature_dim,
            "num_agents": self.num_agents,
            "goal_distance_m": self.goal_distance_m,
            "clearance_m": self.clearance_m,
            "min_pair_distance_m": self.min_pair_distance_m,
            "mean_slot_error_m": self.mean_slot_error_m,
            "max_slot_error_m": self.max_slot_error_m,
            "finite_reward": self.finite_reward,
        }


_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="single-static",
        mode="single",
        dynamic_gates=False,
        description="One drone traverses the default static gate course.",
    ),
    Scenario(
        name="multi-static",
        mode="multi",
        dynamic_gates=False,
        description="A variable-size team traverses a static gate course with formation slots.",
    ),
    Scenario(
        name="multi-dynamic",
        mode="multi",
        dynamic_gates=True,
        description="An eight-drone dynamic gate-density scenario with moving posts.",
    ),
)


def available_scenarios() -> tuple[Scenario, ...]:
    """Return the stable public scenario catalog."""

    return _SCENARIOS


def build_environment(name: ScenarioName, *, agents: int | None = None) -> SingleGate2DEnv | MultiGate2DEnv:
    """Create a core environment without importing PyTorch or Isaac Lab."""

    scenario = _scenario_by_name(name)
    if scenario.mode == "single":
        if agents not in (None, 1):
            raise ValueError("single-static supports exactly one agent")
        return SingleGate2DEnv()
    _validate_multi_agent_count(agents)
    if scenario.dynamic_gates:
        return MultiGate2DEnv(multi_config=build_dynamic_gate_density_8d_config())
    return MultiGate2DEnv(multi_config=MULTI_EXPERIMENT_CONFIG)


def run_rollout(
    name: ScenarioName,
    *,
    agents: int | None = None,
    seed: int = 7,
    steps: int = 8,
) -> RolloutSummary:
    """Advance a scenario with zero actions and collect deterministic diagnostics.

    This intentionally small rollout is a dependency-light contract check, not a policy
    evaluation. It is used by the CLI, tests, and reproducibility checks.
    """

    if steps < 1:
        raise ValueError("steps must be at least one")
    environment = build_environment(name, agents=agents)
    try:
        observation, info = _reset_environment(environment, name=name, seed=seed, agents=agents)
        action = np.zeros(environment.action_shape, dtype=np.float32)
        reward_total = 0.0
        terminated = False
        truncated = False
        for _ in range(steps):
            observation, reward, terminated, truncated, info = environment.step(action)
            reward_total += float(reward)
            if terminated or truncated:
                break
        return _summarize_rollout(
            environment,
            name=name,
            seed=seed,
            steps=steps,
            observation=observation,
            info=info,
            reward_total=reward_total,
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def run_smoke(
    name: ScenarioName,
    *,
    agents: int | None = None,
    seed: int = 7,
    steps: int = 8,
) -> dict[str, object]:
    """Return the JSON-safe form of a short deterministic scenario rollout."""

    return run_rollout(name, agents=agents, seed=seed, steps=steps).to_dict()


def _reset_environment(
    environment: SingleGate2DEnv | MultiGate2DEnv,
    *,
    name: ScenarioName,
    seed: int,
    agents: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reset the scenario-specific environment with an explicit seed."""

    if isinstance(environment, SingleGate2DEnv):
        return environment.reset(seed=seed)
    requested_agents = 8 if name == "multi-dynamic" and agents is None else agents
    return environment.reset(seed=seed, num_agents=requested_agents)


def _summarize_rollout(
    environment: SingleGate2DEnv | MultiGate2DEnv,
    *,
    name: ScenarioName,
    seed: int,
    steps: int,
    observation: dict[str, np.ndarray],
    info: dict[str, Any],
    reward_total: float,
    terminated: bool,
    truncated: bool,
) -> RolloutSummary:
    """Normalize environment-family diagnostics into the public result schema."""

    node_features = observation["node_features"]
    step_count = _extract_step_count(environment, info)
    return RolloutSummary(
        scenario=name,
        seed=seed,
        steps_requested=steps,
        steps_executed=step_count,
        reward_total=round(reward_total, 6),
        terminated=bool(terminated),
        truncated=bool(truncated),
        done_reason=info.get("done_reason"),
        observation_nodes=int(node_features.shape[0]),
        observation_feature_dim=int(node_features.shape[1]),
        num_agents=int(info.get("num_agents", 1)),
        goal_distance_m=round(float(info["goal_distance_m"]), 6),
        clearance_m=_finite_optional_metric(info, "min_clearance_m", fallback_name="signed_clearance_m"),
        min_pair_distance_m=_finite_optional_metric(info, "min_pair_distance_m"),
        mean_slot_error_m=_finite_optional_metric(info, "mean_slot_error_m"),
        max_slot_error_m=_finite_optional_metric(info, "max_slot_error_m"),
        finite_reward=math.isfinite(reward_total),
    )


def _validate_multi_agent_count(agents: int | None) -> None:
    """Reject team sizes that cannot describe a multi-agent task."""

    if agents is not None and agents < 2:
        raise ValueError("multi-agent scenarios require at least two agents")


def _extract_step_count(environment: SingleGate2DEnv | MultiGate2DEnv, info: dict[str, Any]) -> int:
    """Read the environment-family-specific step counter from runtime metadata."""

    if isinstance(environment, SingleGate2DEnv):
        return int(info["state"].step_count)
    return int(info["step_count"])


def _finite_optional_metric(
    info: dict[str, Any],
    metric_name: str,
    *,
    fallback_name: str | None = None,
) -> float | None:
    """Return an optional, finite metric rounded for a stable JSON report."""

    raw_value = info.get(metric_name, info.get(fallback_name) if fallback_name is not None else None)
    if raw_value is None:
        return None
    value = float(raw_value)
    return round(value, 6) if math.isfinite(value) else None


def _scenario_by_name(name: str) -> Scenario:
    normalized_name = str(name).strip().lower()
    for scenario in _SCENARIOS:
        if scenario.name == normalized_name:
            return scenario
    choices = ", ".join(scenario.name for scenario in _SCENARIOS)
    raise ValueError(f"unknown scenario {name!r}; choose one of: {choices}")
