"""Target-free, high-level reinforcement-learning wrappers for G2-I."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .adapters import project_g1
from .canonical import content_hash
from .contracts import ActionPacket, ObservationPacket
from .geometry import distance
from .planning_cadence import PlanningCadenceController
from .public_boundary import assert_public_fields, validate_public_episode
from .runtime import L0FleetRuntime, RuntimeStep

G2_I_RL_OBSERVATION_SCHEMA = "org.aerocity.bench.g2-i-rl-observation.v1"
G2_I_RL_CONTEXT_SCHEMA = "org.aerocity.bench.g2-i-rl-public-context.v1"
G2_I_REWARD_SCHEMA = "org.aerocity.bench.g2-i-rl-reward.v1"
_REWARD_COMPONENTS = (
    "inspection_area_fraction_delta",
    "anonymous_confirmation_count",
    "elapsed_fraction",
    "energy_fraction",
    "collision_count",
    "out_of_bounds_count",
    "safety_intervention_count",
    "return_completion_fraction",
)


@dataclass(frozen=True)
class G2IRewardContract:
    """Versioned public reward decomposition; never a benchmark score."""

    contract_id: str
    weights: dict[str, float]
    credit_assignment: str = "team_shared"

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("G2-I reward contract ID cannot be empty")
        if set(self.weights) != set(_REWARD_COMPONENTS):
            raise ValueError("G2-I reward component fields differ")
        if any(not math.isfinite(float(value)) for value in self.weights.values()):
            raise ValueError("G2-I reward weights must be finite")
        if self.credit_assignment != "team_shared":
            raise ValueError("G2-I v1 reward uses team-shared credit assignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": G2_I_REWARD_SCHEMA,
            "contract_id": self.contract_id,
            "credit_assignment": self.credit_assignment,
            "weights": dict(self.weights),
            "uses_private_truth": False,
            "is_benchmark_score": False,
        }


G2_I_INSPECTION_SHAPED_REWARD_V1 = G2IRewardContract(
    contract_id="g2-i-inspection-shaped-v1",
    weights={
        "inspection_area_fraction_delta": 1.0,
        "anonymous_confirmation_count": 1.0,
        "elapsed_fraction": -0.05,
        "energy_fraction": -0.05,
        "collision_count": -1.0,
        "out_of_bounds_count": -0.5,
        "safety_intervention_count": -0.1,
        "return_completion_fraction": 0.25,
    },
)
G2_I_CONFIRMATION_ONLY_REWARD_V1 = G2IRewardContract(
    contract_id="g2-i-confirmation-only-v1",
    weights={
        component: float(component == "anonymous_confirmation_count")
        for component in _REWARD_COMPONENTS
    },
)


def _cell_semantics(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "pose": cell.get("pose"),
        "surface_point": cell.get("surface_point"),
        "surface_normal": cell.get("surface_normal"),
        "represented_area_m2": cell.get("represented_area_m2"),
        "pose_envelope": cell.get("pose_envelope"),
    }


def build_g2_i_rl_public_context(runtime: L0FleetRuntime) -> tuple[dict[str, Any], dict[str, str]]:
    """Encode selected public cells without generator IDs or padding-order shortcuts."""

    task = runtime.public_task_spec
    episode = runtime.public_episode
    if (
        not isinstance(task, dict)
        or task.get("task_track") != "G2-I"
        or not isinstance(episode, dict)
    ):
        raise ValueError("G2-I RL wrapper requires a public G2-I task and episode")
    validate_public_episode(episode, task)
    atlas = task.get("inspection_atlas")
    sector = episode.get("mission_sector")
    if not isinstance(atlas, dict) or not isinstance(sector, dict):
        raise ValueError("G2-I RL wrapper requires a selected public mission sector")
    selected_ids = {str(value) for value in sector["selected_cell_ids"]}
    encoded: list[tuple[str, str, dict[str, Any]]] = []
    for region in atlas["regions"]:
        for cell in region["cells"]:
            cell_id = str(cell["cell_id"])
            if cell_id not in selected_ids:
                continue
            semantics = _cell_semantics(cell)
            handle = content_hash(semantics)
            encoded.append((handle, cell_id, semantics))
    if {cell_id for _, cell_id, _ in encoded} != selected_ids:
        raise ValueError("G2-I RL context does not resolve the complete mission sector")
    encoded.sort(key=lambda item: item[0])
    if len({handle for handle, _, _ in encoded}) != len(encoded):
        raise ValueError("G2-I RL semantic cell handles are not unique")
    context: dict[str, Any] = {
        "schema": G2_I_RL_CONTEXT_SCHEMA,
        "layout_id": str(task["layout_id"]),
        "episode_id": str(episode["episode_id"]),
        "public_task_spec_hash": content_hash(task),
        "public_episode_hash": content_hash(episode),
        "execution_contract": deepcopy(task["execution_contract"]),
        "starts": deepcopy(episode["starts"]),
        "cell_ordering": "semantic-content-sha256-v1",
        "cell_handles": [handle for handle, _, _ in encoded],
        "cell_features": [semantics for _, _, semantics in encoded],
        "private_truth_available": False,
    }
    # Avoid even false sentinel keys containing private/target terms on the RL wire.
    context.pop("private_truth_available")
    assert_public_fields(context, path="g2_i_rl_context")
    context["context_hash"] = content_hash(context)
    id_by_handle = {handle: cell_id for handle, cell_id, _ in encoded}
    return context, id_by_handle


class G2IGymnasiumFleetWrapper:
    """One RL step equals one high-level decision, not one motor-control tick."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        runtime: L0FleetRuntime,
        *,
        reward_contract: G2IRewardContract = G2_I_INSPECTION_SHAPED_REWARD_V1,
    ) -> None:
        self.runtime = runtime
        self.reward_contract = reward_contract
        self.public_context, self._cell_id_by_handle = build_g2_i_rl_public_context(runtime)
        self._handle_by_cell_id = {
            cell_id: handle for handle, cell_id in self._cell_id_by_handle.items()
        }
        self._cadence = PlanningCadenceController.from_execution_contract(
            runtime.config.raw["execution_contract"]
        )
        self._control_tick = 0
        self._latest_observations: dict[str, ObservationPacket] = {}
        self._anonymous_confirmation_handles: set[str] = set()
        self._return_requested: set[str] = set()
        self._return_rewarded: set[str] = set()
        self._return_reserve_event_emitted = False
        self._homes = {
            str(start["drone_id"]): tuple(float(value) for value in start["position"])
            for start in runtime.public_episode["starts"]
        }

    def _inspection_history(self) -> dict[str, Any]:
        state = self.runtime.public_inspection_state()
        visited = set(str(value) for value in state["visited_cell_ids"])
        ordered_ids = [
            self._cell_id_by_handle[handle]
            for handle in self.public_context["cell_handles"]
        ]
        return {
            "schema": "org.aerocity.bench.g2-i-rl-inspection-history.v1",
            "visited_cell_mask": [cell_id in visited for cell_id in ordered_ids],
            "visited_area_m2": float(state["visited_area_m2"]),
            "total_area_m2": float(state["total_area_m2"]),
            "area_fraction": float(state["area_fraction"]),
            "anonymous_confirmation_handles": sorted(self._anonymous_confirmation_handles),
        }

    def _project(self, observations: dict[str, ObservationPacket]) -> dict[str, dict[str, Any]]:
        history = self._inspection_history()
        duration = float(self.runtime.config.raw["execution_contract"]["episode"]["duration_s"])
        return {
            drone_id: {
                "schema": G2_I_RL_OBSERVATION_SCHEMA,
                "public_context_ref": self.public_context["context_hash"],
                "agent": project_g1(observation),
                "inspection_history": deepcopy(history),
                "task_time_fraction": min(1.0, observation.timestamp_s / duration),
            }
            for drone_id, observation in sorted(observations.items())
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._cadence = PlanningCadenceController.from_execution_contract(
            self.runtime.config.raw["execution_contract"]
        )
        self._control_tick = 0
        self._anonymous_confirmation_handles.clear()
        self._return_requested.clear()
        self._return_rewarded.clear()
        self._return_reserve_event_emitted = False
        self._latest_observations = self.runtime.reset()
        return self._project(self._latest_observations), {
            "execution_level": self.runtime.execution_level,
            "public_context": deepcopy(self.public_context),
            "reward_contract": self.reward_contract.to_dict(),
            "high_level_decision_period_s": self._cadence.planning_period_s,
        }

    def _request_public_events(self, result: RuntimeStep) -> None:
        if result.confirmations:
            self._cadence.request_event("anonymous_confirmation")
            self._anonymous_confirmation_handles.update(
                str(value["anonymous_target_handle"]) for value in result.confirmations
            )
        if any(receipt.safety_intervention for receipt in result.execution_receipts):
            self._cadence.request_event("safety_intervention")
        episode = self.runtime.config.raw["execution_contract"]["episode"]
        reserve_start = float(episode["duration_s"]) - float(episode["return_reserve_s"])
        if not self._return_reserve_event_emitted and result.task_time_s >= reserve_start:
            self._cadence.request_event("return_reserve_entry")
            self._return_reserve_event_emitted = True

    def _return_completion_fraction(self) -> float:
        radius = float(self.runtime.config.raw["execution_contract"]["vehicle"]["home_radius_m"])
        completed = 0
        for drone_id in sorted(self._return_requested - self._return_rewarded):
            observation = self._latest_observations.get(drone_id)
            if (
                observation is not None
                and distance(observation.pose.position, self._homes[drone_id]) <= radius
            ):
                self._return_rewarded.add(drone_id)
                completed += 1
        return completed / max(1, len(self._homes))

    def step(self, actions: dict[str, ActionPacket]):
        due = self._cadence.due_reasons(
            control_tick=self._control_tick,
            active_drone_ids=tuple(self._latest_observations),
        )
        if not due:
            raise RuntimeError("G2-I RL step was called before the next high-level decision")
        self._cadence.approve(actions)
        self._return_requested.update(
            drone_id for drone_id, action in actions.items() if action.kind == "RETURN"
        )
        inspection_before = self.runtime.public_inspection_state()
        task_time_before = self.runtime.task_time_s
        all_receipts = []
        all_confirmations = []
        result: RuntimeStep | None = None
        planner_invoked = True
        while True:
            control_actions = (
                actions
                if planner_invoked
                else self._cadence.held_actions(self._latest_observations)
            )
            result = self.runtime.step(
                control_actions,
                planning_latencies_s={drone_id: 0.0 for drone_id in control_actions},
                planner_invoked_by_drone={
                    drone_id: planner_invoked for drone_id in control_actions
                },
            )
            all_receipts.extend(result.execution_receipts)
            all_confirmations.extend(result.confirmations)
            self._latest_observations = result.observations
            self._control_tick += 1
            self._request_public_events(result)
            if result.done:
                break
            next_due = self._cadence.due_reasons(
                control_tick=self._control_tick,
                active_drone_ids=tuple(self._latest_observations),
            )
            if next_due:
                break
            planner_invoked = False
        assert result is not None
        inspection_after = self.runtime.public_inspection_state()
        execution = self.runtime.config.raw["execution_contract"]
        energy_denominator = float(execution["vehicle"]["energy_budget_j"]) * len(self._homes)
        components = {
            "inspection_area_fraction_delta": max(
                0.0,
                float(inspection_after["area_fraction"])
                - float(inspection_before["area_fraction"]),
            ),
            "anonymous_confirmation_count": float(len(all_confirmations)),
            "elapsed_fraction": (result.task_time_s - task_time_before)
            / float(execution["episode"]["duration_s"]),
            "energy_fraction": sum(receipt.energy_used_j for receipt in all_receipts)
            / energy_denominator,
            "collision_count": float(sum(receipt.collision for receipt in all_receipts)),
            "out_of_bounds_count": float(sum(receipt.out_of_bounds for receipt in all_receipts)),
            "safety_intervention_count": float(
                sum(receipt.safety_intervention for receipt in all_receipts)
            ),
            "return_completion_fraction": self._return_completion_fraction(),
        }
        team_reward = sum(
            components[name] * self.reward_contract.weights[name]
            for name in _REWARD_COMPONENTS
        )
        observations = self._project(self._latest_observations)
        active = set(observations)
        agents = set(actions)
        terminated = {
            drone_id: result.done or drone_id not in active for drone_id in agents
        }
        truncated = {drone_id: False for drone_id in agents}
        info = {
            "task_time_s": result.task_time_s,
            "control_ticks_executed": len(all_receipts) // max(1, len(actions)),
            "planning_trigger_reasons": list(due),
            "reward_components": components,
            "reward_contract": self.reward_contract.to_dict(),
        }
        rewards = {drone_id: team_reward for drone_id in agents}
        return observations, rewards, terminated, truncated, info


class G2IPettingZooParallelWrapper:
    """PettingZoo-compatible view over the same high-level G2-I transition."""

    metadata = {"name": "aerocity_g2_i_v1", "is_parallelizable": True}

    def __init__(
        self,
        runtime: L0FleetRuntime,
        *,
        reward_contract: G2IRewardContract = G2_I_INSPECTION_SHAPED_REWARD_V1,
    ) -> None:
        self._wrapper = G2IGymnasiumFleetWrapper(runtime, reward_contract=reward_contract)
        self.possible_agents = sorted(runtime.reset())
        self.agents = list(self.possible_agents)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, common = self._wrapper.reset(seed=seed, options=options)
        self.agents = sorted(observations)
        return observations, {agent: deepcopy(common) for agent in self.agents}

    def step(self, actions: dict[str, ActionPacket]):
        acting_agents = list(self.agents)
        observations, rewards, terminations, truncations, common = self._wrapper.step(actions)
        infos = {agent: deepcopy(common) for agent in acting_agents}
        self.agents = [] if all(terminations.values()) else sorted(observations)
        return observations, rewards, terminations, truncations, infos
