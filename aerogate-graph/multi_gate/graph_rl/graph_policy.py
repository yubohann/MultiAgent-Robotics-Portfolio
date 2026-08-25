"""Graph policy shared across all active agents."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Normal

from multi_gate.configs.experiment_config import MultiGraphMASACConfig, MultiGraphObservationConfig


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _bounded_features(features: Tensor, bound: float) -> Tensor:
    if bound <= 0.0:
        return features
    norm = features.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    scale = torch.clamp(float(bound) / norm, max=1.0)
    return features * scale


class GraphEncoder(nn.Module):
    """Message-passing encoder that returns node and pooled graph embeddings."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        message_passing_steps: int,
        feature_norm_bound: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.feature_norm_bound = float(feature_norm_bound)

    def forward(self, node_features: Tensor, adjacency: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor]:
        mask = node_mask.unsqueeze(-1)
        hidden = _bounded_features(self.node_embed(node_features), self.feature_norm_bound) * mask
        masked_adjacency = adjacency * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        degree = masked_adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)

        for layer in self.message_layers:
            aggregated = torch.bmm(masked_adjacency / degree, hidden)
            hidden = _bounded_features(F.relu(layer(torch.cat([hidden, aggregated], dim=-1))), self.feature_norm_bound) * mask

        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden, _bounded_features(pooled, self.feature_norm_bound)


class GraphPolicy(nn.Module):
    """Shared-parameter actor head producing one action per active agent."""

    def __init__(
        self,
        *,
        obs_config: MultiGraphObservationConfig,
        masac_config: MultiGraphMASACConfig,
        max_agents_soft: int,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_feature_dim=obs_config.node_feature_dim,
            hidden_dim=masac_config.graph_hidden_dim,
            message_passing_steps=masac_config.message_passing_steps,
            feature_norm_bound=float(getattr(masac_config, "feature_norm_bound", 0.0)),
        )
        self.agent_backbone = _mlp(
            masac_config.graph_hidden_dim * 2,
            masac_config.actor_hidden_dim,
            masac_config.actor_hidden_dim,
        )
        self.mean_layer = nn.Linear(masac_config.actor_hidden_dim, masac_config.action_dim)
        self.log_std_layer = nn.Linear(masac_config.actor_hidden_dim, masac_config.action_dim)
        self.actor_head_mode = str(getattr(masac_config, "actor_head_mode", "single") or "single").lower()
        if self.actor_head_mode not in {"single", "modular"}:
            raise ValueError(f"Unsupported actor_head_mode: {self.actor_head_mode}")
        if self.actor_head_mode == "modular":
            self.path_mean_layer = nn.Linear(masac_config.actor_hidden_dim, masac_config.action_dim)
            self.slot_mean_layer = nn.Linear(masac_config.actor_hidden_dim, masac_config.action_dim)
            self.avoid_mean_layer = nn.Linear(masac_config.actor_hidden_dim, masac_config.action_dim)
            self.gate_layer = nn.Linear(masac_config.actor_hidden_dim, 3)
        self.last_gate_weights: Tensor | None = None
        self.log_std_min = masac_config.log_std_min
        self.log_std_max = masac_config.log_std_max
        self.max_agents_soft = int(max_agents_soft)

    def forward(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        node_embeddings, pooled = self.encoder(
            observation["node_features"],
            observation["adjacency"],
            observation["node_mask"],
        )
        agent_embeddings = node_embeddings[:, : self.max_agents_soft, :]
        pooled_repeated = pooled.unsqueeze(1).expand(-1, self.max_agents_soft, -1)
        hidden = self.agent_backbone(torch.cat([agent_embeddings, pooled_repeated], dim=-1))
        if self.actor_head_mode == "modular":
            head_means = torch.stack(
                [
                    self.path_mean_layer(hidden),
                    self.slot_mean_layer(hidden),
                    self.avoid_mean_layer(hidden),
                ],
                dim=-2,
            )
            gate_weights = torch.softmax(self.gate_layer(hidden), dim=-1)
            mean = (head_means * gate_weights.unsqueeze(-1)).sum(dim=-2)
            self.last_gate_weights = gate_weights
        else:
            mean = self.mean_layer(hidden)
            self.last_gate_weights = None
        log_std = torch.clamp(self.log_std_layer(hidden), self.log_std_min, self.log_std_max)
        return mean, log_std, pooled

    def sample(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, log_std, pooled = self.forward(observation)
        mask = observation["action_mask"].unsqueeze(-1)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh) * mask
        log_prob = distribution.log_prob(pre_tanh) - torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + 1e-6)
        log_prob = (log_prob * mask).sum(dim=(1, 2)).unsqueeze(-1)
        deterministic_action = torch.tanh(mean) * mask
        return action, log_prob, deterministic_action, pooled

