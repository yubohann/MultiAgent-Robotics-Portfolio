"""Author-network MARVEL SAC core for the public three-dimensional candidate graph."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from aerocity_method.adapters.hm3d_baselines import PublicSearchState
from aerocity_method.adapters.hm3d_single_rl import (
    public_candidate_features,
    public_context_features,
)
from aerocity_method.contracts.io import finite_number
from aerocity_method.contracts.models import CandidateFragmentManifest

try:
    import torch
    from torch import Tensor, nn

    from aerocity_method.vendor.marvel_icra2025 import PolicyNet as AuthorMarvelPolicyNet
    from aerocity_method.vendor.marvel_icra2025 import QNet as AuthorMarvelQNet
except ModuleNotFoundError:  # pragma: no cover - explicit at runtime
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]
    AuthorMarvelPolicyNet = Any  # type: ignore[assignment,misc]
    AuthorMarvelQNet = Any  # type: ignore[assignment,misc]

MARVEL_AUTHOR_MODEL_COMMIT = "318c2a6016d0f2d1dbb0dd08b3f8f8224b361e4c"
CONTEXT_DIM = 4
AGENT_DIM = 5
CANDIDATE_DIM = 8
TEAM_STAT_DIM = 3
NODE_DIM = 1 + CONTEXT_DIM + CANDIDATE_DIM + TEAM_STAT_DIM
NUM_ANGLES_BIN = 36


def public_marvel_agent_features(state: PublicSearchState) -> tuple[tuple[float, ...], ...]:
    """Return public, translation-stable three-dimensional team-node features."""

    centroid = tuple(
        sum(agent.position_m[axis] for agent in state.agents) / len(state.agents)
        for axis in range(3)
    )
    reference = state.communication_range_m
    max_degree = max(1, len(state.agents) - 1)
    return tuple(
        (
            (agent.position_m[0] - centroid[0]) / reference,
            (agent.position_m[1] - centroid[1]) / reference,
            (agent.position_m[2] - centroid[2]) / reference,
            agent.remaining_energy_fraction,
            agent.communication_degree / max_degree,
        )
        for agent in state.agents
    )


def public_marvel_adjacency(state: PublicSearchState) -> tuple[tuple[bool, ...], ...]:
    """Build a self-looped public communication graph."""

    return tuple(
        tuple(
            index == other_index
            or math.dist(agent.position_m, other.position_m) <= state.communication_range_m
            for other_index, other in enumerate(state.agents)
        )
        for index, agent in enumerate(state.agents)
    )


def _candidate_orientation_features(
    state: PublicSearchState,
    manifest: CandidateFragmentManifest,
) -> tuple[float, float, float]:
    starts = {agent.agent_id: agent.position_m for agent in state.agents}
    directions: list[tuple[float, float, float]] = []
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        start = starts.get(fragment.agent_id)
        if start is None:
            raise ValueError("MARVEL manifest references an unknown public agent")
        endpoint = tuple(fragment.path[-1])
        delta = tuple(endpoint[axis] - start[axis] for axis in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        directions.append(
            (0.0, 0.0, 0.0)
            if length <= 1.0e-12
            else tuple(value / length for value in delta)
        )
    if not directions:
        raise ValueError("MARVEL candidate needs transit fragments")
    return tuple(sum(row[axis] for row in directions) / len(directions) for axis in range(3))


def public_marvel_candidate_features(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> tuple[tuple[float, ...], ...]:
    """Fuse the common candidate descriptor with explicit three-dimensional direction."""

    return tuple(
        (*public_candidate_features(manifest), *_candidate_orientation_features(state, manifest))
        for manifest in pool
    )


@dataclass(frozen=True, slots=True)
class MarvelSupplementaryReferenceConfig:
    context_dim: int = CONTEXT_DIM
    agent_dim: int = AGENT_DIM
    candidate_dim: int = CANDIDATE_DIM
    node_dim: int = NODE_DIM
    num_angles_bin: int = NUM_ANGLES_BIN
    hidden_dim: int = 128
    learning_rate: float = 1.0e-5
    alpha_learning_rate: float = 1.0e-4
    gamma_reference: float = 0.99
    discount_reference_s: float = 5.0
    initial_log_alpha: float = -2.0
    target_entropy_scale: float = 0.05
    target_update_interval: int = 64
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "context_dim",
            "agent_dim",
            "candidate_dim",
            "node_dim",
            "num_angles_bin",
            "hidden_dim",
            "target_update_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by the author's four attention heads")
        if (self.context_dim, self.agent_dim, self.candidate_dim) != (
            CONTEXT_DIM,
            AGENT_DIM,
            CANDIDATE_DIM,
        ) or (self.node_dim, self.num_angles_bin) != (NODE_DIM, NUM_ANGLES_BIN):
            raise ValueError("MARVEL controlled-transfer dimensions are frozen")
        for name in (
            "learning_rate",
            "alpha_learning_rate",
            "gamma_reference",
            "discount_reference_s",
            "initial_log_alpha",
            "target_entropy_scale",
        ):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))
        if self.learning_rate <= 0.0 or self.alpha_learning_rate <= 0.0:
            raise ValueError("MARVEL learning rates must be positive")
        if not 0.0 < self.gamma_reference <= 1.0 or self.discount_reference_s <= 0.0:
            raise ValueError("MARVEL duration discount is invalid")
        if self.target_entropy_scale < 0.0:
            raise ValueError("target_entropy_scale must be non-negative")


@dataclass(frozen=True, slots=True)
class MarvelGraphObservation:
    """One author-network observation assembled only from the public contract."""

    node_inputs: tuple[tuple[float, ...], ...]
    edge_mask: tuple[tuple[bool, ...], ...]
    current_edge: tuple[int, ...]
    edge_padding_mask: tuple[bool, ...]
    frontier_distribution: tuple[tuple[float, ...], ...]
    heading_occupancy: tuple[tuple[float, ...], ...]
    neighbor_best_headings: tuple[tuple[tuple[float, ...], ...], ...]
    current_index: int = 0

    def __post_init__(self) -> None:
        nodes = tuple(
            tuple(finite_number(value, "MARVEL node feature") for value in row)
            for row in self.node_inputs
        )
        if len(nodes) < 2 or any(len(row) != NODE_DIM for row in nodes):
            raise ValueError("MARVEL graph needs a root and fixed-width candidate nodes")
        count = len(nodes)
        edge_mask = tuple(tuple(row) for row in self.edge_mask)
        if len(edge_mask) != count or any(
            len(row) != count or any(not isinstance(value, bool) for value in row)
            for row in edge_mask
        ):
            raise ValueError("MARVEL graph edge mask must be square and boolean")
        if any(edge_mask[index][index] for index in range(count)):
            raise ValueError("MARVEL graph nodes require unmasked self edges")
        current_edge = tuple(self.current_edge)
        padding = tuple(self.edge_padding_mask)
        if (
            len(current_edge) != count - 1
            or current_edge != tuple(range(1, count))
            or len(padding) != len(current_edge)
            or any(not isinstance(value, bool) for value in padding)
            or all(padding)
        ):
            raise ValueError("MARVEL root-to-candidate action mask is malformed")
        if self.current_index != 0:
            raise ValueError("MARVEL controlled transfer freezes the team root at node zero")
        normalized_matrices: list[tuple[tuple[float, ...], ...]] = []
        for name, matrix in (
            ("frontier_distribution", self.frontier_distribution),
            ("heading_occupancy", self.heading_occupancy),
        ):
            rows = tuple(
                tuple(finite_number(value, name) for value in row) for row in matrix
            )
            if len(rows) != count or any(len(row) != NUM_ANGLES_BIN for row in rows):
                raise ValueError(f"{name} has the wrong shape")
            if any(value < 0.0 for row in rows for value in row):
                raise ValueError(f"{name} must be non-negative")
            normalized_matrices.append(rows)
        neighbor = tuple(
            tuple(
                tuple(finite_number(value, "neighbor_best_headings") for value in bin_row)
                for bin_row in row
            )
            for row in self.neighbor_best_headings
        )
        if len(neighbor) != len(current_edge) or any(
            len(row) != 1 or len(row[0]) != NUM_ANGLES_BIN for row in neighbor
        ):
            raise ValueError("MARVEL best-heading tensor has the wrong shape")
        if any(value < 0.0 for row in neighbor for bin_row in row for value in bin_row):
            raise ValueError("MARVEL best-heading values must be non-negative")
        object.__setattr__(self, "node_inputs", nodes)
        object.__setattr__(self, "edge_mask", edge_mask)
        object.__setattr__(self, "current_edge", current_edge)
        object.__setattr__(self, "edge_padding_mask", padding)
        object.__setattr__(self, "frontier_distribution", normalized_matrices[0])
        object.__setattr__(self, "heading_occupancy", normalized_matrices[1])
        object.__setattr__(self, "neighbor_best_headings", neighbor)

    @property
    def legal_mask(self) -> tuple[bool, ...]:
        return tuple(not value for value in self.edge_padding_mask)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_inputs": [list(row) for row in self.node_inputs],
            "edge_mask": [list(row) for row in self.edge_mask],
            "current_index": self.current_index,
            "current_edge": list(self.current_edge),
            "edge_padding_mask": list(self.edge_padding_mask),
            "frontier_distribution": [list(row) for row in self.frontier_distribution],
            "heading_occupancy": [list(row) for row in self.heading_occupancy],
            "neighbor_best_headings": [
                [list(bin_row) for bin_row in row] for row in self.neighbor_best_headings
            ],
        }


def marvel_graph_observation_from_dict(payload: Mapping[str, Any]) -> MarvelGraphObservation:
    """Validate and reconstruct one serialized author-network input."""

    def matrix(name: str) -> tuple[tuple[Any, ...], ...]:
        value = payload.get(name)
        if not isinstance(value, list):
            raise ValueError(f"MARVEL {name} must be a list")
        if any(not isinstance(row, list) for row in value):
            raise ValueError(f"MARVEL {name} rows must be lists")
        return tuple(tuple(row) for row in value)

    raw_neighbor = payload.get("neighbor_best_headings")
    if not isinstance(raw_neighbor, list) or any(not isinstance(row, list) for row in raw_neighbor):
        raise ValueError("MARVEL neighbor_best_headings must be a list")
    neighbor: list[tuple[tuple[Any, ...], ...]] = []
    for row in raw_neighbor:
        if any(not isinstance(bin_row, list) for bin_row in row):
            raise ValueError("MARVEL neighbor heading rows must be lists")
        neighbor.append(tuple(tuple(bin_row) for bin_row in row))
    current_edge = payload.get("current_edge")
    edge_padding = payload.get("edge_padding_mask")
    if not isinstance(current_edge, list) or not isinstance(edge_padding, list):
        raise ValueError("MARVEL action arrays must be lists")
    current_index = payload.get("current_index")
    if not isinstance(current_index, int) or isinstance(current_index, bool):
        raise ValueError("MARVEL current_index must be an integer")
    return MarvelGraphObservation(
        node_inputs=matrix("node_inputs"),
        edge_mask=matrix("edge_mask"),
        current_index=current_index,
        current_edge=tuple(current_edge),
        edge_padding_mask=tuple(edge_padding),
        frontier_distribution=matrix("frontier_distribution"),
        heading_occupancy=matrix("heading_occupancy"),
        neighbor_best_headings=tuple(neighbor),
    )


def _orientation_bin(direction: tuple[float, float, float]) -> int:
    angle = math.atan2(direction[1], direction[0])
    return int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * NUM_ANGLES_BIN)) % NUM_ANGLES_BIN


def _one_hot(index: int) -> tuple[float, ...]:
    return tuple(1.0 if item == index else 0.0 for item in range(NUM_ANGLES_BIN))


def _normalise_histogram(values: Sequence[float]) -> tuple[float, ...]:
    total = sum(values)
    return tuple(0.0 for _ in values) if total <= 0.0 else tuple(value / total for value in values)


def _candidate_endpoint_centroid(manifest: CandidateFragmentManifest) -> tuple[float, float, float]:
    endpoints = [
        tuple(fragment.path[-1])
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    ]
    if not endpoints:
        raise ValueError("MARVEL candidate needs transit endpoints")
    return tuple(sum(point[axis] for point in endpoints) / len(endpoints) for axis in range(3))


def public_marvel_graph_observation(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> MarvelGraphObservation:
    """Map the common candidate pool to the author PolicyNet/QNet tensor contract."""

    rows = tuple(pool)
    if not rows or not any(row.feasible for row in rows):
        raise ValueError("MARVEL graph requires at least one legal candidate")
    context = public_context_features(state)
    candidate_features = public_marvel_candidate_features(state, rows)
    agent_count = len(state.agents)
    adjacency = public_marvel_adjacency(state)
    possible_edges = max(1, agent_count * max(1, agent_count - 1))
    communication_density = sum(
        1
        for index, row in enumerate(adjacency)
        for other, connected in enumerate(row)
        if index != other and connected
    ) / possible_edges
    vertical_span = (
        max(agent.position_m[2] for agent in state.agents)
        - min(agent.position_m[2] for agent in state.agents)
    ) / state.communication_range_m
    team_stats = (
        sum(agent.remaining_energy_fraction for agent in state.agents) / agent_count,
        communication_density,
        min(1.0, vertical_span),
    )
    root = (0.0, *context, *((0.0,) * CANDIDATE_DIM), *team_stats)
    nodes = (root,) + tuple(
        (1.0, *context, *features, *team_stats) for features in candidate_features
    )
    endpoints = tuple(_candidate_endpoint_centroid(row) for row in rows)
    node_count = len(nodes)
    edge_mask_rows: list[tuple[bool, ...]] = []
    for source in range(node_count):
        mask_row: list[bool] = []
        for target in range(node_count):
            if source == target or source == 0 or target == 0:
                mask_row.append(False)
            else:
                mask_row.append(
                    math.dist(endpoints[source - 1], endpoints[target - 1])
                    > state.communication_range_m
                )
        edge_mask_rows.append(tuple(mask_row))
    candidate_directions = tuple(_candidate_orientation_features(state, row) for row in rows)
    candidate_bins = tuple(_orientation_bin(direction) for direction in candidate_directions)
    root_frontier = [0.0] * NUM_ANGLES_BIN
    for bin_index, row in zip(candidate_bins, rows, strict=True):
        root_frontier[bin_index] += max(0.0, float(row.quality_hint)) + 1.0e-6
    frontier_distribution = (_normalise_histogram(root_frontier),) + tuple(
        _one_hot(index) for index in candidate_bins
    )
    centroid = tuple(
        sum(agent.position_m[axis] for agent in state.agents) / agent_count for axis in range(3)
    )
    root_heading = [0.0] * NUM_ANGLES_BIN
    for agent in state.agents:
        delta = tuple(agent.position_m[axis] - centroid[axis] for axis in range(3))
        if math.hypot(delta[0], delta[1]) > 1.0e-9:
            root_heading[_orientation_bin(delta)] += 1.0
    heading_occupancy = (_normalise_histogram(root_heading),) + tuple(
        _one_hot(index) for index in candidate_bins
    )
    return MarvelGraphObservation(
        node_inputs=nodes,
        edge_mask=tuple(edge_mask_rows),
        current_edge=tuple(range(1, node_count)),
        edge_padding_mask=tuple(not row.feasible for row in rows),
        frontier_distribution=frontier_distribution,
        heading_occupancy=heading_occupancy,
        neighbor_best_headings=tuple((_one_hot(index),) for index in candidate_bins),
    )


@dataclass(frozen=True, slots=True)
class MarvelSupplementaryReferenceTrainingRow:
    observation: MarvelGraphObservation
    next_observation: MarvelGraphObservation
    action: int
    reward: float
    duration_s: float
    done: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observation, MarvelGraphObservation) or not isinstance(
            self.next_observation, MarvelGraphObservation
        ):
            raise TypeError("MARVEL training rows require current and next graph observations")
        if not isinstance(self.action, int) or not 0 <= self.action < len(
            self.observation.current_edge
        ):
            raise ValueError("MARVEL training action is outside the candidate pool")
        if not self.observation.legal_mask[self.action]:
            raise ValueError("MARVEL training action is not legal")
        reward = finite_number(self.reward, "MARVEL reward")
        duration = finite_number(self.duration_s, "MARVEL duration_s")
        if not 0.0 <= reward <= 1.0 or duration <= 0.0 or not isinstance(self.done, bool):
            raise ValueError("MARVEL reward, duration or terminal flag is invalid")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "duration_s", duration)


class MarvelSupplementaryReferencePolicy:
    """Author PolicyNet/QNet trained with the author's discrete SAC structure."""

    def __init__(self, config: MarvelSupplementaryReferenceConfig, *, seed: int = 0) -> None:
        if torch is None or nn is None:
            raise RuntimeError("MARVEL baseline requires PyTorch")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(seed)
        self.policy = AuthorMarvelPolicyNet(
            config.node_dim, config.hidden_dim, config.num_angles_bin
        ).to(self.device)
        self.q1 = AuthorMarvelQNet(
            config.node_dim, config.hidden_dim, config.num_angles_bin, 0
        ).to(self.device)
        self.q2 = AuthorMarvelQNet(
            config.node_dim, config.hidden_dim, config.num_angles_bin, 0
        ).to(self.device)
        self.target_q1 = AuthorMarvelQNet(
            config.node_dim, config.hidden_dim, config.num_angles_bin, 0
        ).to(self.device)
        self.target_q2 = AuthorMarvelQNet(
            config.node_dim, config.hidden_dim, config.num_angles_bin, 0
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()
        self.log_alpha = torch.tensor(
            [config.initial_log_alpha], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_learning_rate
        )
        self.update_count = 0

    def _batch(
        self, observations: Sequence[MarvelGraphObservation]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not observations:
            raise ValueError("MARVEL tensor batch cannot be empty")
        batch_size = len(observations)
        max_nodes = max(len(row.node_inputs) for row in observations)
        max_actions = max(len(row.current_edge) for row in observations)
        node_inputs = torch.zeros(
            (batch_size, max_nodes, self.config.node_dim),
            dtype=torch.float32,
            device=self.device,
        )
        node_padding = torch.ones(
            (batch_size, 1, max_nodes), dtype=torch.bool, device=self.device
        )
        edge_mask = torch.ones(
            (batch_size, max_nodes, max_nodes), dtype=torch.bool, device=self.device
        )
        current_index = torch.zeros(
            (batch_size, 1, 1), dtype=torch.long, device=self.device
        )
        current_edge = torch.zeros(
            (batch_size, max_actions, 1), dtype=torch.long, device=self.device
        )
        action_padding = torch.ones(
            (batch_size, 1, max_actions), dtype=torch.bool, device=self.device
        )
        frontier = torch.zeros(
            (batch_size, max_nodes, self.config.num_angles_bin),
            dtype=torch.float32,
            device=self.device,
        )
        headings = torch.zeros_like(frontier)
        best_headings = torch.zeros(
            (batch_size, max_actions, 1, self.config.num_angles_bin),
            dtype=torch.float32,
            device=self.device,
        )
        for batch_index, observation in enumerate(observations):
            node_count = len(observation.node_inputs)
            action_count = len(observation.current_edge)
            node_inputs[batch_index, :node_count] = torch.tensor(
                observation.node_inputs, dtype=torch.float32, device=self.device
            )
            node_padding[batch_index, :, :node_count] = False
            edge_mask[batch_index, :node_count, :node_count] = torch.tensor(
                observation.edge_mask, dtype=torch.bool, device=self.device
            )
            current_index[batch_index, 0, 0] = observation.current_index
            current_edge[batch_index, :action_count, 0] = torch.tensor(
                observation.current_edge, dtype=torch.long, device=self.device
            )
            action_padding[batch_index, 0, :action_count] = torch.tensor(
                observation.edge_padding_mask, dtype=torch.bool, device=self.device
            )
            frontier[batch_index, :node_count] = torch.tensor(
                observation.frontier_distribution, dtype=torch.float32, device=self.device
            )
            headings[batch_index, :node_count] = torch.tensor(
                observation.heading_occupancy, dtype=torch.float32, device=self.device
            )
            best_headings[batch_index, :action_count] = torch.tensor(
                observation.neighbor_best_headings,
                dtype=torch.float32,
                device=self.device,
            )
        return (
            node_inputs,
            node_padding,
            edge_mask,
            current_index,
            current_edge,
            action_padding,
            frontier,
            headings,
            best_headings,
        )

    def action_probabilities(
        self, observation: MarvelGraphObservation
    ) -> tuple[float, ...]:
        with torch.no_grad():
            log_probabilities = self.policy(*self._batch((observation,)))
            probabilities = log_probabilities.exp()[0, : len(observation.current_edge)]
        result = tuple(float(value) for value in probabilities.cpu().tolist())
        if not math.isclose(sum(result), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6):
            raise FloatingPointError("MARVEL author policy produced invalid probabilities")
        return result

    def update(self, rows: Sequence[MarvelSupplementaryReferenceTrainingRow]) -> dict[str, float]:
        if not rows:
            raise ValueError("MARVEL update requires real training rows")
        state = self._batch(tuple(row.observation for row in rows))
        next_state = self._batch(tuple(row.next_observation for row in rows))
        actions = torch.tensor([row.action for row in rows], dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [row.reward for row in rows], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([row.done for row in rows], dtype=torch.float32, device=self.device)
        durations = torch.tensor(
            [row.duration_s for row in rows], dtype=torch.float32, device=self.device
        )
        discounts = torch.exp(
            math.log(self.config.gamma_reference)
            * durations
            / self.config.discount_reference_s
        )

        logp = self.policy(*state)
        probabilities = logp.exp()
        with torch.no_grad():
            q_values = torch.minimum(self.q1(*state), self.q2(*state)).squeeze(-1)
        policy_loss = (
            probabilities * (self.log_alpha.exp().detach() * logp - q_values.detach())
        ).sum(dim=1).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=100.0
        )
        self.policy_optimizer.step()

        with torch.no_grad():
            next_logp = self.policy(*next_state)
            next_probabilities = next_logp.exp()
            next_q = torch.minimum(
                self.target_q1(*next_state), self.target_q2(*next_state)
            ).squeeze(-1)
            next_value = (
                next_probabilities * (next_q - self.log_alpha.exp() * next_logp)
            ).sum(dim=1)
            target = rewards + discounts * (1.0 - dones) * next_value
        q1_values = self.q1(*state).squeeze(-1)
        q1_selected = q1_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q1_loss = torch.nn.functional.mse_loss(q1_selected, target)
        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=20000.0)
        self.q1_optimizer.step()
        q2_values = self.q2(*state).squeeze(-1)
        q2_selected = q2_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_loss = torch.nn.functional.mse_loss(q2_selected, target)
        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=20000.0)
        self.q2_optimizer.step()

        negative_entropy = (logp * probabilities).sum(dim=-1)
        legal_counts = torch.tensor(
            [sum(row.observation.legal_mask) for row in rows],
            dtype=torch.float32,
            device=self.device,
        )
        target_entropy = self.config.target_entropy_scale * torch.log(legal_counts)
        alpha_loss = -(self.log_alpha * (negative_entropy.detach() + target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_q1.load_state_dict(self.q1.state_dict())
            self.target_q2.load_state_dict(self.q2.state_dict())
        diagnostics = {
            "policy_loss": float(policy_loss.detach().cpu()),
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.log_alpha.exp().detach().cpu()),
            "entropy": float((-negative_entropy).mean().detach().cpu()),
            "policy_grad_norm": float(policy_grad_norm.detach().cpu()),
            "q1_grad_norm": float(q1_grad_norm.detach().cpu()),
            "q2_grad_norm": float(q2_grad_norm.detach().cpu()),
            "target_mean": float(target.mean().detach().cpu()),
        }
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("MARVEL SAC update produced non-finite diagnostics")
        return diagnostics

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "author_model_commit": MARVEL_AUTHOR_MODEL_COMMIT,
            "policy_model": self.policy.state_dict(),
            "q_net1_model": self.q1.state_dict(),
            "q_net2_model": self.q2.state_dict(),
            "target_q_net1_model": self.target_q1.state_dict(),
            "target_q_net2_model": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        config = payload.get("config")
        required_models = (
            "policy_model",
            "q_net1_model",
            "q_net2_model",
            "target_q_net1_model",
            "target_q_net2_model",
        )
        if not isinstance(config, Mapping) or any(
            not isinstance(payload.get(name), Mapping) for name in required_models
        ):
            raise ValueError("MARVEL state is incomplete")
        if payload.get("author_model_commit") != MARVEL_AUTHOR_MODEL_COMMIT:
            raise ValueError("MARVEL checkpoint uses an unapproved author model source")
        if MarvelSupplementaryReferenceConfig(**dict(config)) != self.config:
            raise ValueError("MARVEL checkpoint config differs from instantiated model")
        self.policy.load_state_dict(dict(payload["policy_model"]))
        self.q1.load_state_dict(dict(payload["q_net1_model"]))
        self.q2.load_state_dict(dict(payload["q_net2_model"]))
        self.target_q1.load_state_dict(dict(payload["target_q_net1_model"]))
        self.target_q2.load_state_dict(dict(payload["target_q_net2_model"]))
        log_alpha = payload.get("log_alpha")
        if not isinstance(log_alpha, Tensor) or log_alpha.numel() != 1:
            raise ValueError("MARVEL checkpoint lacks learned entropy temperature")
        self.log_alpha.data.copy_(log_alpha.to(self.device))
        update_count = payload.get("update_count")
        if not isinstance(update_count, int) or isinstance(update_count, bool) or update_count < 0:
            raise ValueError("MARVEL checkpoint update count is invalid")
        self.update_count = update_count


__all__ = [
    "AGENT_DIM",
    "CANDIDATE_DIM",
    "CONTEXT_DIM",
    "MARVEL_AUTHOR_MODEL_COMMIT",
    "MarvelGraphObservation",
    "MarvelSupplementaryReferenceConfig",
    "MarvelSupplementaryReferencePolicy",
    "MarvelSupplementaryReferenceTrainingRow",
    "marvel_graph_observation_from_dict",
    "public_marvel_adjacency",
    "public_marvel_agent_features",
    "public_marvel_candidate_features",
    "public_marvel_graph_observation",
]
