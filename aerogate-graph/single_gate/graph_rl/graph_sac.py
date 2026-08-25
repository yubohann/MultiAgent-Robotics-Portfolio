"""A lightweight single-agent Graph-FlashSAC implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Normal

from single_gate.configs.experiment_config import (
    SINGLE_EXPERIMENT_CONFIG,
    SingleGraphObservationConfig,
    SingleGraphSACConfig,
)
from single_gate.graph_rl.replay_buffer import GraphReplayBuffer


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


def _project_module_weights(module: nn.Module, bound: float) -> None:
    if bound <= 0.0:
        return
    with torch.no_grad():
        for parameter in module.parameters():
            norm = parameter.data.norm().clamp_min(1.0e-6)
            if float(norm.item()) > float(bound):
                parameter.data.mul_(float(bound) / norm)


class GraphEncoder(nn.Module):
    """Simple message-passing encoder over fixed-size graph observations."""

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

    def forward(self, node_features: Tensor, adjacency: Tensor, node_mask: Tensor) -> Tensor:
        mask = node_mask.unsqueeze(-1)
        hidden = _bounded_features(self.node_embed(node_features), self.feature_norm_bound) * mask
        masked_adjacency = adjacency * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        degree = masked_adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)

        for layer in self.message_layers:
            aggregated = torch.bmm(masked_adjacency / degree, hidden)
            hidden = _bounded_features(F.relu(layer(torch.cat([hidden, aggregated], dim=-1))), self.feature_norm_bound) * mask

        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return _bounded_features(pooled, self.feature_norm_bound)


class SquashedGaussianGraphActor(nn.Module):
    """Policy network that maps graph observations to bounded planar actions."""

    def __init__(self, obs_config: SingleGraphObservationConfig, sac_config: SingleGraphSACConfig) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_feature_dim=obs_config.node_feature_dim,
            hidden_dim=sac_config.graph_hidden_dim,
            message_passing_steps=sac_config.message_passing_steps,
            feature_norm_bound=float(getattr(sac_config, "feature_norm_bound", 0.0)),
        )
        self.backbone = _mlp(sac_config.graph_hidden_dim, sac_config.actor_hidden_dim, sac_config.actor_hidden_dim)
        self.mean_layer = nn.Linear(sac_config.actor_hidden_dim, sac_config.action_dim)
        self.log_std_layer = nn.Linear(sac_config.actor_hidden_dim, sac_config.action_dim)
        self.log_std_min = sac_config.log_std_min
        self.log_std_max = sac_config.log_std_max

    def forward(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        graph_embedding = self.encoder(
            observation["node_features"],
            observation["adjacency"],
            observation["node_mask"],
        )
        hidden = self.backbone(graph_embedding)
        mean = self.mean_layer(hidden)
        log_std = torch.clamp(self.log_std_layer(hidden), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mean)
        return action, log_prob, deterministic_action


class GraphCritic(nn.Module):
    """Q-network over graph observations and continuous actions."""

    def __init__(self, obs_config: SingleGraphObservationConfig, sac_config: SingleGraphSACConfig) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_feature_dim=obs_config.node_feature_dim,
            hidden_dim=sac_config.graph_hidden_dim,
            message_passing_steps=sac_config.message_passing_steps,
            feature_norm_bound=float(getattr(sac_config, "feature_norm_bound", 0.0)),
        )
        self.q_network = _mlp(
            sac_config.graph_hidden_dim + sac_config.action_dim,
            sac_config.critic_hidden_dim,
            1,
        )

    def forward(self, observation: dict[str, Tensor], action: Tensor) -> Tensor:
        graph_embedding = self.encoder(
            observation["node_features"],
            observation["adjacency"],
            observation["node_mask"],
        )
        return self.q_network(torch.cat([graph_embedding, action], dim=-1))


@dataclass(frozen=True)
class GraphSACBuildContext:
    """Metadata required to build a Graph-FlashSAC agent."""

    obs_config: SingleGraphObservationConfig
    sac_config: SingleGraphSACConfig
    obs_shapes: dict[str, tuple[int, ...]]


class GraphSACAgent:
    """Minimal single-agent Graph-FlashSAC agent with replay and checkpointing."""

    def __init__(
        self,
        *,
        build_context: GraphSACBuildContext,
        device: str | torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self.obs_config = build_context.obs_config
        self.sac_config = build_context.sac_config
        self.obs_shapes = build_context.obs_shapes
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = SquashedGaussianGraphActor(self.obs_config, self.sac_config).to(self.device)
        self.critic_1 = GraphCritic(self.obs_config, self.sac_config).to(self.device)
        self.critic_2 = GraphCritic(self.obs_config, self.sac_config).to(self.device)
        self.target_critic_1 = GraphCritic(self.obs_config, self.sac_config).to(self.device)
        self.target_critic_2 = GraphCritic(self.obs_config, self.sac_config).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for module in (self.target_critic_1, self.target_critic_2):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.actor_optimizer = self._build_actor_optimizer()
        self.critic_optimizer = self._build_critic_optimizer()
        self.log_alpha = torch.tensor(
            np.log(self.sac_config.init_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = self._build_alpha_optimizer()
        self.replay_buffer = self._build_replay_buffer(seed=self.seed)
        self.update_step = 0
        self.update_call_step = 0
        self.reward_scale_ema = torch.tensor(1.0, dtype=torch.float32, device=self.device)

    @classmethod
    def from_defaults(
        cls,
        *,
        obs_shapes: dict[str, tuple[int, ...]],
        device: str | torch.device | None = None,
        seed: int = 0,
        obs_config: SingleGraphObservationConfig | None = None,
        sac_config: SingleGraphSACConfig | None = None,
    ) -> "GraphSACAgent":
        return cls(
            build_context=GraphSACBuildContext(
                obs_config=obs_config or SINGLE_EXPERIMENT_CONFIG.observation,
                sac_config=sac_config or SINGLE_EXPERIMENT_CONFIG.algorithm,
                obs_shapes=obs_shapes,
            ),
            device=device,
            seed=seed,
        )

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def _build_actor_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.sac_config.actor_lr,
            weight_decay=float(getattr(self.sac_config, "actor_weight_decay", 0.0)),
        )

    def _build_critic_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.sac_config.critic_lr,
            weight_decay=float(getattr(self.sac_config, "critic_weight_decay", 0.0)),
        )

    def _build_alpha_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [self.log_alpha],
            lr=self.sac_config.alpha_lr,
            weight_decay=float(getattr(self.sac_config, "alpha_weight_decay", 0.0)),
        )

    def _build_replay_buffer(self, *, seed: int | None = None) -> GraphReplayBuffer:
        return GraphReplayBuffer(
            capacity=self.sac_config.replay_buffer_capacity,
            obs_shapes=self.obs_shapes,
            action_dim=self.sac_config.action_dim,
            seed=self.seed if seed is None else int(seed),
        )

    def _obs_batch_to_tensor(self, observation: dict[str, np.ndarray]) -> dict[str, Tensor]:
        return {
            name: torch.as_tensor(array, dtype=torch.float32, device=self.device)
            for name, array in observation.items()
        }

    def _obs_to_tensor(self, observation: dict[str, np.ndarray]) -> dict[str, Tensor]:
        return {
            name: torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)
            for name, array in observation.items()
        }

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        obs_tensor = self._obs_to_tensor(observation)
        action, _, deterministic_action = self.actor.sample(obs_tensor)
        chosen = deterministic_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act_batch(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        obs_tensor = self._obs_batch_to_tensor(observation)
        action, _, deterministic_action = self.actor.sample(obs_tensor)
        chosen = deterministic_action if deterministic else action
        return chosen.cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, object]) -> dict[str, float]:
        self.update_call_step += 1
        update_interval = max(int(getattr(self.sac_config, "flash_update_interval", 1) or 1), 1)
        if self.update_call_step % update_interval != 0:
            return {
                "flash_update_skipped": 1.0,
                "flash_update_interval": float(update_interval),
                "alpha": float(self.alpha.item()),
                "reward_scale": float(self.reward_scale_ema.item()),
            }

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards, reward_scale = self._normalize_rewards(batch["rewards"])
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1 = self.target_critic_1(next_obs, next_action)
            target_q2 = self.target_critic_2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = rewards + (1.0 - dones) * self.sac_config.gamma * target_q
            target_value_clip = float(getattr(self.sac_config, "target_value_clip", 0.0) or 0.0)
            if target_value_clip > 0.0:
                target_value = target_value.clamp(-target_value_clip, target_value_clip)

        current_q1 = self.critic_1(obs, actions)
        current_q2 = self.critic_2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            float(getattr(self.sac_config, "max_grad_norm", 0.0) or 1.0e9),
        )
        self.critic_optimizer.step()
        self._project_weights()

        new_action, log_prob, _ = self.actor.sample(obs)
        q1_pi = self.critic_1(obs, new_action)
        q2_pi = self.critic_2(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            float(getattr(self.sac_config, "max_grad_norm", 0.0) or 1.0e9),
        )
        self.actor_optimizer.step()
        self._project_weights()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.sac_config.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
            [self.log_alpha],
            float(getattr(self.sac_config, "max_grad_norm", 0.0) or 1.0e9),
        )
        self.alpha_optimizer.step()
        self._clamp_alpha()

        self._soft_update_targets()
        self.update_step += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q_pi.mean().item()),
            "reward_scale": float(reward_scale.item()),
            "flash_update_skipped": 0.0,
            "flash_update_interval": float(update_interval),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "alpha_grad_norm": float(alpha_grad_norm.item()),
        }

    def _normalize_rewards(self, rewards: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            batch_scale = rewards.detach().abs().mean().clamp_min(float(getattr(self.sac_config, "reward_scale_min", 1.0)))
            decay = float(getattr(self.sac_config, "reward_scale_ema_decay", 0.995))
            self.reward_scale_ema.mul_(decay).add_((1.0 - decay) * batch_scale)
            self.reward_scale_ema.clamp_(
                float(getattr(self.sac_config, "reward_scale_min", 1.0)),
                float(getattr(self.sac_config, "reward_scale_max", 1.0e6)),
            )
        return rewards / self.reward_scale_ema.detach(), self.reward_scale_ema.detach()

    def _project_weights(self) -> None:
        bound = float(getattr(self.sac_config, "weight_norm_bound", 0.0) or 0.0)
        for module in (self.actor, self.critic_1, self.critic_2):
            _project_module_weights(module, bound)

    def _clamp_alpha(self) -> None:
        min_alpha = float(getattr(self.sac_config, "min_alpha", 1.0e-8))
        max_alpha = float(getattr(self.sac_config, "max_alpha", 1.0e6))
        with torch.no_grad():
            self.log_alpha.clamp_(np.log(min_alpha), np.log(max_alpha))

    def _soft_update_targets(self) -> None:
        tau = self.sac_config.tau
        for source, target in (
            (self.critic_1, self.target_critic_1),
            (self.critic_2, self.target_critic_2),
        ):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, object] | None = None) -> Path:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "update_step": int(self.update_step),
                "update_call_step": int(self.update_call_step),
                "reward_scale_ema": self.reward_scale_ema.detach().cpu(),
                "algorithm_name": str(getattr(self.sac_config, "algorithm_name", "graph_flashsac")),
                "metadata": {
                    "algorithm_name": str(getattr(self.sac_config, "algorithm_name", "graph_flashsac")),
                    **(metadata or {}),
                },
            },
            target_path,
        )
        return target_path

    def load_checkpoint(self, path: str | Path) -> dict[str, object]:
        payload = torch.load(Path(path), map_location=self.device)
        metadata = dict(payload.get("metadata") or {})
        payload_algorithm = str(
            payload.get("algorithm_name")
            or metadata.get("algorithm_name")
            or metadata.get("training_signature", {}).get("algorithm_name")
            or ""
        ).strip()
        current_algorithm = str(getattr(self.sac_config, "algorithm_name", "graph_flashsac"))
        if payload_algorithm and payload_algorithm != current_algorithm:
            self.actor.load_state_dict(payload["actor"])
            reset_metadata = self.reset_training_state(
                reset_optimizer_state=True,
                reset_entropy_state=True,
                reset_replay_buffer=True,
                reset_update_step=True,
                seed=self.seed,
            )
            metadata["legacy_actor_warmstart_only"] = True
            metadata["legacy_algorithm_name"] = payload_algorithm
            metadata["applied_resets"] = reset_metadata
            return metadata
        if not payload_algorithm:
            self.actor.load_state_dict(payload["actor"])
            reset_metadata = self.reset_training_state(
                reset_optimizer_state=True,
                reset_entropy_state=True,
                reset_replay_buffer=True,
                reset_update_step=True,
                seed=self.seed,
            )
            metadata["legacy_actor_warmstart_only"] = True
            metadata["legacy_algorithm_name"] = "unknown_graph_sac"
            metadata["applied_resets"] = reset_metadata
            return metadata
        self.actor.load_state_dict(payload["actor"])
        self.critic_1.load_state_dict(payload["critic_1"])
        self.critic_2.load_state_dict(payload["critic_2"])
        self.target_critic_1.load_state_dict(payload["target_critic_1"])
        self.target_critic_2.load_state_dict(payload["target_critic_2"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.update_step = int(payload.get("update_step", 0))
        self.update_call_step = int(payload.get("update_call_step", self.update_step))
        if payload.get("reward_scale_ema") is not None:
            self.reward_scale_ema.data.copy_(payload["reward_scale_ema"].to(self.device))
        return metadata

    def reset_training_state(
        self,
        *,
        reset_optimizer_state: bool = True,
        reset_entropy_state: bool = True,
        reset_replay_buffer: bool = True,
        reset_update_step: bool = True,
        seed: int | None = None,
    ) -> dict[str, object]:
        """Reset mutable training state after a conservative resume."""

        applied_resets: dict[str, object] = {
            "reset_optimizer_state": bool(reset_optimizer_state),
            "reset_entropy_state": bool(reset_entropy_state),
            "reset_replay_buffer": bool(reset_replay_buffer),
            "reset_update_step": bool(reset_update_step),
        }
        if seed is not None:
            self.seed = int(seed)

        if reset_optimizer_state:
            self.actor_optimizer = self._build_actor_optimizer()
            self.critic_optimizer = self._build_critic_optimizer()
            self.alpha_optimizer = self._build_alpha_optimizer()

        if reset_entropy_state:
            self.log_alpha = torch.tensor(
                np.log(self.sac_config.init_alpha),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = self._build_alpha_optimizer()

        if reset_replay_buffer:
            self.replay_buffer = self._build_replay_buffer(seed=self.seed)

        if reset_update_step:
            self.update_step = 0
            self.update_call_step = 0

        self.reward_scale_ema = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        return applied_resets

GraphFlashSACAgent = GraphSACAgent
FlashSACAgent = GraphSACAgent

