"""Public HM3D exploration candidate generation.

The candidate generator consumes only public beliefs, public frontier clusters
and public vehicle states.  It does not know target coordinates, private ESDFs
or future evaluator coverage.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from aerocity_method.contracts.exploration import (
    AgentExplorationPlan,
    FrontierCluster,
    TeamExplorationCandidate,
)
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.runtime.hm3d_trajectory import (
    Point3,
    TrajectoryTimingConfig,
    plan_continuous_trajectory,
    segment_length_m,
)


@dataclass(frozen=True, slots=True)
class PublicExplorationAgentState:
    agent_id: str
    position_m: Point3
    remaining_energy_j: float
    communication_degree: int

    def __post_init__(self) -> None:
        require_identifier(self.agent_id, "agent_id")
        if len(self.position_m) != 3:
            raise ValueError("position_m must be 3D")
        object.__setattr__(
            self,
            "position_m",
            tuple(finite_number(value, "position_m") for value in self.position_m),
        )
        energy = finite_number(self.remaining_energy_j, "remaining_energy_j")
        if energy < 0.0:
            raise ValueError("remaining_energy_j must be non-negative")
        object.__setattr__(self, "remaining_energy_j", energy)
        if (
            not isinstance(self.communication_degree, int)
            or isinstance(self.communication_degree, bool)
            or self.communication_degree < 0
        ):
            raise ValueError("communication_degree must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ExplorationCandidateBudget:
    decision_start_s: float
    decision_duration_s: float
    reserve_energy_j: float = 0.0
    energy_per_meter_j: float = 1.0
    risk_per_meter: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "decision_start_s",
            "decision_duration_s",
            "reserve_energy_j",
            "energy_per_meter_j",
            "risk_per_meter",
        ):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.decision_duration_s <= 0.0 or self.energy_per_meter_j <= 0.0:
            raise ValueError("duration and energy_per_meter_j must be positive")


PathGuard = Callable[[str, tuple[Point3, ...]], tuple[bool, tuple[Point3, ...], str]]


def permissive_path_guard(
    agent_id: str, path_m: tuple[Point3, ...]
) -> tuple[bool, tuple[Point3, ...], str]:
    require_identifier(agent_id, "agent_id")
    return True, path_m, ""


def _distance(left: Point3, right: Point3) -> float:
    return segment_length_m(left, right)


def _rank_frontiers_for_agent(
    agent: PublicExplorationAgentState, frontiers: Sequence[FrontierCluster]
) -> tuple[FrontierCluster, ...]:
    return tuple(
        sorted(
            frontiers,
            key=lambda frontier: (
                -frontier.expected_gain_m3
                / max(_distance(agent.position_m, frontier.centroid_m), 0.5),
                frontier.frontier_id,
            ),
        )
    )


def _height_descriptor(plans: Sequence[AgentExplorationPlan]) -> float:
    endpoints = [plan.trajectory_m[-1][2] for plan in plans]
    if len(endpoints) < 2:
        return 0.0
    spread = max(endpoints) - min(endpoints)
    return spread / (spread + 1.0)


def _workload_balance(plans: Sequence[AgentExplorationPlan]) -> float:
    durations = [plan.duration_s for plan in plans]
    mean = sum(durations) / len(durations)
    if mean <= 1.0e-12:
        return 1.0
    variance = sum((value - mean) ** 2 for value in durations) / len(durations)
    return 1.0 / (1.0 + math.sqrt(variance) / mean)


def _plan_for_frontier(
    agent: PublicExplorationAgentState,
    frontier: FrontierCluster,
    *,
    start_s: float,
    budget: ExplorationCandidateBudget,
    timing: TrajectoryTimingConfig,
    guard: PathGuard,
) -> AgentExplorationPlan:
    viewpoint = frontier.viewpoint_candidates_m[0]
    legal, guarded_path, reason = guard(agent.agent_id, (agent.position_m, viewpoint))
    if not legal:
        return AgentExplorationPlan(
            agent_id=agent.agent_id,
            role="hold",
            trajectory_m=(agent.position_m,),
            duration_s=max(1.0e-6, budget.decision_duration_s),
            expected_gain_m3=0.0,
            risk=1.0,
            energy_j=0.0,
            frontier_id=None,
        )
    trajectory = plan_continuous_trajectory(
        agent.agent_id,
        guarded_path,
        start_time_s=start_s,
        config=timing,
    )
    energy_j = trajectory.distance_m * budget.energy_per_meter_j
    risk = min(1.0, trajectory.distance_m * budget.risk_per_meter)
    feasible_time = trajectory.duration_s <= budget.decision_duration_s + 1.0e-9
    feasible_energy = energy_j + budget.reserve_energy_j <= agent.remaining_energy_j + 1.0e-9
    if not feasible_time or not feasible_energy:
        role = "hold"
        gain = 0.0
        risk = 1.0
        path = (agent.position_m,)
        duration = max(1.0e-6, budget.decision_duration_s)
        frontier_id = None
    else:
        role = "explore"
        gain = frontier.expected_gain_m3
        path = trajectory.path_m
        duration = trajectory.duration_s
        frontier_id = frontier.frontier_id
    _ = reason
    return AgentExplorationPlan(
        agent_id=agent.agent_id,
        role=role,
        trajectory_m=path,
        duration_s=duration,
        expected_gain_m3=gain,
        risk=risk,
        energy_j=energy_j if role == "explore" else 0.0,
        frontier_id=frontier_id,
    )


def build_exploration_candidate_pool(
    *,
    context_payload: dict[str, object],
    belief_version_sha256s: Sequence[str],
    agents: Sequence[PublicExplorationAgentState],
    frontiers: Sequence[FrontierCluster],
    budget: ExplorationCandidateBudget,
    timing: TrajectoryTimingConfig,
    candidate_limit: int = 8,
    guard: PathGuard = permissive_path_guard,
) -> tuple[TeamExplorationCandidate, ...]:
    """Build common team candidates for weak baselines and RL selectors."""

    rows = tuple(sorted(agents, key=lambda row: row.agent_id))
    clusters = tuple(sorted(frontiers, key=lambda row: row.frontier_id))
    if not rows:
        raise ValueError("candidate generation requires at least one agent")
    if not clusters:
        raise ValueError("candidate generation requires public frontier clusters")
    if (
        not isinstance(candidate_limit, int)
        or isinstance(candidate_limit, bool)
        or candidate_limit < 1
    ):
        raise ValueError("candidate_limit must be a positive integer")
    context_sha256 = canonical_sha256(context_payload)
    candidates: list[TeamExplorationCandidate] = []
    for offset in range(min(candidate_limit, len(clusters))):
        plans: list[AgentExplorationPlan] = []
        selected_frontiers: set[str] = set()
        for agent_index, agent in enumerate(rows):
            ranked = _rank_frontiers_for_agent(agent, clusters)
            frontier = ranked[(offset + agent_index) % len(ranked)]
            selected_frontiers.add(frontier.frontier_id)
            plans.append(
                _plan_for_frontier(
                    agent,
                    frontier,
                    start_s=budget.decision_start_s,
                    budget=budget,
                    timing=timing,
                    guard=guard,
                )
            )
        feasible = all(plan.role != "hold" for plan in plans)
        reasons = () if feasible else ("candidate_guard_or_budget_rejected",)
        total_gain = sum(plan.expected_gain_m3 for plan in plans)
        total_energy = sum(plan.energy_j for plan in plans)
        average_risk = sum(plan.risk for plan in plans) / len(plans)
        descriptor = (
            _height_descriptor(plans),
            len(selected_frontiers) / len(plans),
            _workload_balance(plans),
            total_gain / max(1.0, total_gain + total_energy),
            average_risk,
        )
        candidates.append(
            TeamExplorationCandidate(
                candidate_id=f"hm3d-exploration-candidate-{offset}",
                context_sha256=context_sha256,
                belief_version_sha256s=tuple(belief_version_sha256s),
                agent_plans=tuple(plans),
                planned_descriptor=descriptor,
                feasible=feasible,
                admission_reasons=reasons,
                source="hm3d-public-frontier-exploration-v1",
            )
        )
    if not any(candidate.feasible for candidate in candidates):
        raise ValueError("no feasible exploration candidate survived public guards")
    return tuple(candidates)


__all__ = [
    "ExplorationCandidateBudget",
    "PathGuard",
    "PublicExplorationAgentState",
    "build_exploration_candidate_pool",
    "permissive_path_guard",
]
