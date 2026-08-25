"""A lightweight Graph-FlashSAC implementation for the multi-agent task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from multi_gate.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG,
    MultiGraphMASACConfig,
    MultiGraphObservationConfig,
)
from multi_gate.graph_rl.graph_policy import GraphPolicy, GraphEncoder, _mlp
from multi_gate.graph_rl.replay_buffer import MultiGraphReplayBuffer


def _project_module_weights(module: nn.Module, bound: float) -> None:
    if bound <= 0.0:
        return
    with torch.no_grad():
        for parameter in module.parameters():
            norm = parameter.data.norm().clamp_min(1.0e-6)
            if float(norm.item()) > float(bound):
                parameter.data.mul_(float(bound) / norm)


class CentralizedGraphCritic(nn.Module):
    """Centralized double-Q critic over pooled graph state and joint actions."""

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
        critic_input_dim = (
            masac_config.graph_hidden_dim
            + max_agents_soft * masac_config.action_dim
            + max_agents_soft
        )
        self.q_network = _mlp(critic_input_dim, masac_config.critic_hidden_dim, 1)
        self.max_agents_soft = int(max_agents_soft)

    def forward(self, observation: dict[str, Tensor], action: Tensor) -> Tensor:
        _, pooled = self.encoder(
            observation["node_features"],
            observation["adjacency"],
            observation["node_mask"],
        )
        action_mask = observation["action_mask"]
        masked_action = action * action_mask.unsqueeze(-1)
        critic_input = torch.cat(
            [
                pooled,
                masked_action.reshape(masked_action.shape[0], -1),
                action_mask,
            ],
            dim=-1,
        )
        return self.q_network(critic_input)


class CentralizedSafetyCritic(CentralizedGraphCritic):
    """Predict discounted safety cost-to-go for constrained actor updates."""


@dataclass(frozen=True)
class GraphMASACBuildContext:
    """Metadata required to build a Graph-FlashSAC agent."""

    obs_config: MultiGraphObservationConfig
    masac_config: MultiGraphMASACConfig
    obs_shapes: dict[str, tuple[int, ...]]
    max_agents_soft: int


class GraphMASACAgent:
    """Minimal Graph-FlashSAC agent with shared graph actor and centralized critics."""

    def __init__(
        self,
        *,
        build_context: GraphMASACBuildContext,
        device: str | torch.device | None = None,
        seed: int = 0,
        build_replay_buffer: bool = True,
    ) -> None:
        self.obs_config = build_context.obs_config
        self.masac_config = build_context.masac_config
        self.obs_shapes = build_context.obs_shapes
        self.max_agents_soft = build_context.max_agents_soft
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = int(seed)
        self._failure_replay_enabled = False
        self._failure_replay_ratio = 0.0
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = GraphPolicy(
            obs_config=self.obs_config,
            masac_config=self.masac_config,
            max_agents_soft=self.max_agents_soft,
        ).to(self.device)
        self.behavior_reference_actor: GraphPolicy | None = None
        self.critic_1 = CentralizedGraphCritic(
            obs_config=self.obs_config,
            masac_config=self.masac_config,
            max_agents_soft=self.max_agents_soft,
        ).to(self.device)
        self.critic_2 = CentralizedGraphCritic(
            obs_config=self.obs_config,
            masac_config=self.masac_config,
            max_agents_soft=self.max_agents_soft,
        ).to(self.device)
        self.target_critic_1 = CentralizedGraphCritic(
            obs_config=self.obs_config,
            masac_config=self.masac_config,
            max_agents_soft=self.max_agents_soft,
        ).to(self.device)
        self.target_critic_2 = CentralizedGraphCritic(
            obs_config=self.obs_config,
            masac_config=self.masac_config,
            max_agents_soft=self.max_agents_soft,
        ).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for module in (self.target_critic_1, self.target_critic_2):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.safety_critic: CentralizedSafetyCritic | None = None
        self.target_safety_critic: CentralizedSafetyCritic | None = None
        self.safety_optimizer: torch.optim.Optimizer | None = None
        if bool(getattr(self.masac_config, "enable_safety_critic", False)):
            self.safety_critic = CentralizedSafetyCritic(
                obs_config=self.obs_config,
                masac_config=self.masac_config,
                max_agents_soft=self.max_agents_soft,
            ).to(self.device)
            self.target_safety_critic = CentralizedSafetyCritic(
                obs_config=self.obs_config,
                masac_config=self.masac_config,
                max_agents_soft=self.max_agents_soft,
            ).to(self.device)
            self.target_safety_critic.load_state_dict(self.safety_critic.state_dict())
            for parameter in self.target_safety_critic.parameters():
                parameter.requires_grad = False
            self.safety_optimizer = torch.optim.AdamW(
                self.safety_critic.parameters(),
                lr=self.masac_config.critic_lr,
                weight_decay=float(getattr(self.masac_config, "critic_weight_decay", 0.0)),
            )

        self.actor_optimizer = self._build_actor_optimizer()
        self.critic_optimizer = self._build_critic_optimizer()
        self.log_alpha = torch.tensor(
            np.log(self.masac_config.init_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = self._build_alpha_optimizer()
        self.replay_buffer = (
            self._build_replay_buffer(seed=self.seed)
            if bool(build_replay_buffer)
            else None
        )
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
        obs_config: MultiGraphObservationConfig | None = None,
        masac_config: MultiGraphMASACConfig | None = None,
        max_agents_soft: int | None = None,
        build_replay_buffer: bool = True,
    ) -> "GraphMASACAgent":
        return cls(
            build_context=GraphMASACBuildContext(
                obs_config=obs_config or MULTI_EXPERIMENT_CONFIG.observation,
                masac_config=masac_config or MULTI_EXPERIMENT_CONFIG.algorithm,
                obs_shapes=obs_shapes,
                max_agents_soft=max_agents_soft or MULTI_EXPERIMENT_CONFIG.max_agents_soft,
            ),
            device=device,
            seed=seed,
            build_replay_buffer=build_replay_buffer,
        )

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def _build_actor_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.masac_config.actor_lr,
            weight_decay=float(getattr(self.masac_config, "actor_weight_decay", 0.0)),
        )

    def _build_critic_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.masac_config.critic_lr,
            weight_decay=float(getattr(self.masac_config, "critic_weight_decay", 0.0)),
        )

    def _build_alpha_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [self.log_alpha],
            lr=self.masac_config.alpha_lr,
            weight_decay=float(getattr(self.masac_config, "alpha_weight_decay", 0.0)),
        )

    def _build_replay_buffer(self, *, seed: int | None = None) -> MultiGraphReplayBuffer:
        return MultiGraphReplayBuffer(
            capacity=self.masac_config.replay_buffer_capacity,
            obs_shapes=self.obs_shapes,
            joint_action_shape=(self.max_agents_soft, self.masac_config.action_dim),
            seed=self.seed if seed is None else int(seed),
            failure_replay_ratio=self._failure_replay_ratio,
            enable_failure_replay=self._failure_replay_enabled,
        )

    def configure_replay_sampling(
        self,
        *,
        enabled: bool,
        failure_replay_ratio: float,
    ) -> None:
        """Apply failure-aware replay settings without changing the dataclass shape."""

        self._failure_replay_enabled = bool(enabled)
        self._failure_replay_ratio = float(np.clip(failure_replay_ratio, 0.0, 1.0))
        if self.replay_buffer is not None:
            self.replay_buffer.enable_failure_replay = self._failure_replay_enabled
            self.replay_buffer.failure_replay_ratio = self._failure_replay_ratio

    def capture_behavior_reference(self) -> None:
        """Freeze the current actor as a conservative reference for BC-preserving RL."""

        reference = GraphPolicy(
            obs_config=self.obs_config,
            masac_config=self.masac_config,
            max_agents_soft=self.max_agents_soft,
        ).to(self.device)
        reference.load_state_dict(self.actor.state_dict())
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad = False
        self.behavior_reference_actor = reference

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
        action, _, deterministic_action, _ = self.actor.sample(obs_tensor)
        chosen = deterministic_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act_batch(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        obs_tensor = self._obs_batch_to_tensor(observation)
        action, _, deterministic_action, _ = self.actor.sample(obs_tensor)
        chosen = deterministic_action if deterministic else action
        return chosen.cpu().numpy().astype(np.float32)

    def actor_behavior_clone_loss(
        self,
        observation: dict[str, Tensor],
        expert_actions: Tensor,
        *,
        target_log_std: float | None = None,
        log_std_penalty_scale: float = 0.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute supervised behavior-cloning loss for the shared actor."""

        mean, log_std, _ = self.actor.forward(observation)
        mask = observation["action_mask"].unsqueeze(-1)
        predicted_actions = torch.tanh(mean) * mask
        action_dims = max(int(expert_actions.shape[-1]), 1)
        active_entries = mask.sum().clamp_min(1.0) * action_dims
        bc_loss = (((predicted_actions - expert_actions) ** 2) * mask).sum() / active_entries

        log_std_penalty = torch.zeros((), dtype=bc_loss.dtype, device=bc_loss.device)
        if target_log_std is not None and log_std_penalty_scale > 0.0:
            target = torch.full_like(log_std, float(target_log_std))
            log_std_penalty = (((log_std - target) ** 2) * mask).sum() / active_entries
        total_loss = bc_loss + float(log_std_penalty_scale) * log_std_penalty
        return total_loss, {
            "bc_loss": float(bc_loss.detach().item()),
            "log_std_penalty": float(log_std_penalty.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }

    def update(self, batch: dict[str, object]) -> dict[str, float]:
        self.update_call_step += 1
        update_interval = max(int(getattr(self.masac_config, "flash_update_interval", 1) or 1), 1)
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
        safety_costs = batch.get("safety_costs")
        failure_mask = batch.get("failure_mask")

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_obs)
            target_q1 = self.target_critic_1(next_obs, next_action)
            target_q2 = self.target_critic_2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = rewards + (1.0 - dones) * self.masac_config.gamma * target_q
            target_value_clip = float(getattr(self.masac_config, "target_value_clip", 0.0) or 0.0)
            if target_value_clip > 0.0:
                target_value = target_value.clamp(-target_value_clip, target_value_clip)

        current_q1 = self.critic_1(obs, actions)
        current_q2 = self.critic_2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            float(getattr(self.masac_config, "max_grad_norm", 0.0) or 1.0e9),
        )
        self.critic_optimizer.step()
        self._project_weights()

        safety_critic_loss = None
        mean_safety_cost = 0.0
        if self.safety_critic is not None and self.target_safety_critic is not None and safety_costs is not None:
            assert self.safety_optimizer is not None
            safety_cost_tensor = safety_costs
            mean_safety_cost = float(safety_cost_tensor.mean().detach().item())
            with torch.no_grad():
                next_safety = self.target_safety_critic(next_obs, next_action)
                safety_target = safety_cost_tensor + (1.0 - dones) * self.masac_config.gamma * next_safety
            predicted_safety = self.safety_critic(obs, actions)
            safety_critic_loss_tensor = F.mse_loss(predicted_safety, safety_target)
            self.safety_optimizer.zero_grad()
            safety_critic_loss_tensor.backward()
            safety_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.safety_critic.parameters(),
                float(getattr(self.masac_config, "max_grad_norm", 0.0) or 1.0e9),
            )
            self.safety_optimizer.step()
            self._project_weights()
            safety_critic_loss = safety_critic_loss_tensor
        else:
            safety_grad_norm = torch.zeros((), dtype=torch.float32, device=self.device)

        new_action, log_prob, _, _ = self.actor.sample(obs)
        q1_pi = self.critic_1(obs, new_action)
        q2_pi = self.critic_2(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        safety_actor_penalty = torch.zeros((), dtype=actor_loss.dtype, device=actor_loss.device)
        if self.safety_critic is not None and getattr(self.masac_config, "safety_penalty_scale", 0.0) > 0.0:
            safety_actor_penalty = self.safety_critic(obs, new_action).mean()
            actor_loss = actor_loss + float(self.masac_config.safety_penalty_scale) * safety_actor_penalty
        gate_entropy = torch.zeros((), dtype=actor_loss.dtype, device=actor_loss.device)
        if (
            getattr(self.masac_config, "modular_gate_entropy_scale", 0.0) > 0.0
            and self.actor.last_gate_weights is not None
        ):
            gate_weights = self.actor.last_gate_weights.clamp_min(1e-6)
            gate_entropy = -(gate_weights * gate_weights.log()).sum(dim=-1)
            active = obs["action_mask"]
            gate_entropy = (gate_entropy * active).sum() / active.sum().clamp_min(1.0)
            actor_loss = actor_loss - float(self.masac_config.modular_gate_entropy_scale) * gate_entropy
        behavior_anchor_loss = torch.zeros((), dtype=actor_loss.dtype, device=actor_loss.device)
        behavior_anchor_scale = float(getattr(self.masac_config, "behavior_anchor_loss_scale", 0.0) or 0.0)
        if behavior_anchor_scale > 0.0:
            mean, _, _ = self.actor.forward(obs)
            deterministic_action = torch.tanh(mean)
            action_mask = obs["action_mask"].unsqueeze(-1)
            anchor_weight = action_mask
            if self.behavior_reference_actor is not None:
                with torch.no_grad():
                    reference_mean, _, _ = self.behavior_reference_actor.forward(obs)
                    anchor_action = torch.tanh(reference_mean) * action_mask
            else:
                anchor_action = actions.detach()
            if (
                self.behavior_reference_actor is None
                and bool(getattr(self.masac_config, "behavior_anchor_non_failure_only", True))
                and failure_mask is not None
            ):
                anchor_weight = anchor_weight * (1.0 - failure_mask).clamp(0.0, 1.0).unsqueeze(-1)
            active_entries = anchor_weight.sum().clamp_min(1.0) * max(int(actions.shape[-1]), 1)
            behavior_anchor_loss = (((deterministic_action - anchor_action) ** 2) * anchor_weight).sum() / active_entries
            actor_loss = actor_loss + behavior_anchor_scale * behavior_anchor_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            float(getattr(self.masac_config, "max_grad_norm", 0.0) or 1.0e9),
        )
        self.actor_optimizer.step()
        self._project_weights()

        target_entropy = (
            obs["action_mask"].sum(dim=-1, keepdim=True) * self.masac_config.target_entropy_per_agent
        )
        alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
            [self.log_alpha],
            float(getattr(self.masac_config, "max_grad_norm", 0.0) or 1.0e9),
        )
        self.alpha_optimizer.step()
        self._clamp_alpha()

        self._soft_update_targets()
        self.update_step += 1
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q_pi.mean().item()),
            "behavior_anchor_loss": float(behavior_anchor_loss.detach().item()),
            "reward_scale": float(reward_scale.item()),
            "flash_update_skipped": 0.0,
            "flash_update_interval": float(update_interval),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "alpha_grad_norm": float(alpha_grad_norm.item()),
        }
        if safety_critic_loss is not None:
            metrics["safety_critic_loss"] = float(safety_critic_loss.item())
            metrics["safety_actor_penalty"] = float(safety_actor_penalty.detach().item())
            metrics["mean_safety_cost"] = mean_safety_cost
            metrics["safety_grad_norm"] = float(safety_grad_norm.item())
        if self.actor.last_gate_weights is not None:
            active = obs["action_mask"].unsqueeze(-1)
            gate_mean = (self.actor.last_gate_weights * active).sum(dim=(0, 1)) / active.sum().clamp_min(1.0)
            metrics["modular_gate_path_mean"] = float(gate_mean[0].detach().item())
            metrics["modular_gate_slot_mean"] = float(gate_mean[1].detach().item())
            metrics["modular_gate_avoid_mean"] = float(gate_mean[2].detach().item())
            metrics["modular_gate_entropy"] = float(gate_entropy.detach().item())
        return metrics

    def _normalize_rewards(self, rewards: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            batch_scale = rewards.detach().abs().mean().clamp_min(float(getattr(self.masac_config, "reward_scale_min", 1.0)))
            decay = float(getattr(self.masac_config, "reward_scale_ema_decay", 0.995))
            self.reward_scale_ema.mul_(decay).add_((1.0 - decay) * batch_scale)
            self.reward_scale_ema.clamp_(
                float(getattr(self.masac_config, "reward_scale_min", 1.0)),
                float(getattr(self.masac_config, "reward_scale_max", 1.0e6)),
            )
        return rewards / self.reward_scale_ema.detach(), self.reward_scale_ema.detach()

    def _project_weights(self) -> None:
        bound = float(getattr(self.masac_config, "weight_norm_bound", 0.0) or 0.0)
        modules: list[nn.Module] = [self.actor, self.critic_1, self.critic_2]
        if self.safety_critic is not None:
            modules.append(self.safety_critic)
        for module in modules:
            _project_module_weights(module, bound)

    def _clamp_alpha(self) -> None:
        min_alpha = float(getattr(self.masac_config, "min_alpha", 1.0e-8))
        max_alpha = float(getattr(self.masac_config, "max_alpha", 1.0e6))
        with torch.no_grad():
            self.log_alpha.clamp_(np.log(min_alpha), np.log(max_alpha))

    def _soft_update_targets(self) -> None:
        tau = self.masac_config.tau
        for source, target in (
            (self.critic_1, self.target_critic_1),
            (self.critic_2, self.target_critic_2),
        ):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)
        if self.safety_critic is not None and self.target_safety_critic is not None:
            for source_param, target_param in zip(self.safety_critic.parameters(), self.target_safety_critic.parameters()):
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
                "safety_critic": None if self.safety_critic is None else self.safety_critic.state_dict(),
                "target_safety_critic": (
                    None if self.target_safety_critic is None else self.target_safety_critic.state_dict()
                ),
                "safety_optimizer": None if self.safety_optimizer is None else self.safety_optimizer.state_dict(),
                "update_step": int(self.update_step),
                "update_call_step": int(self.update_call_step),
                "reward_scale_ema": self.reward_scale_ema.detach().cpu(),
                "algorithm_name": str(getattr(self.masac_config, "algorithm_name", "graph_flashsac")),
                "metadata": {
                    "algorithm_name": str(getattr(self.masac_config, "algorithm_name", "graph_flashsac")),
                    **(metadata or {}),
                },
            },
            target_path,
        )
        return target_path

    def save_actor_checkpoint(self, path: str | Path, metadata: dict[str, object] | None = None) -> Path:
        """Save only the actor weights for BC warm start."""

        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "algorithm_name": str(getattr(self.masac_config, "algorithm_name", "graph_flashsac")),
                "metadata": {
                    "algorithm_name": str(getattr(self.masac_config, "algorithm_name", "graph_flashsac")),
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
        current_algorithm = str(getattr(self.masac_config, "algorithm_name", "graph_flashsac"))
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
            metadata["legacy_algorithm_name"] = "unknown_graph_masac"
            metadata["applied_resets"] = reset_metadata
            return metadata
        full_state_keys = (
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
            "actor_optimizer",
            "critic_optimizer",
            "log_alpha",
            "alpha_optimizer",
        )
        missing_full_state_keys = [key for key in full_state_keys if key not in payload]
        if missing_full_state_keys:
            self.actor.load_state_dict(payload["actor"])
            reset_metadata = self.reset_training_state(
                reset_optimizer_state=True,
                reset_entropy_state=True,
                reset_replay_buffer=True,
                reset_update_step=True,
                seed=self.seed,
            )
            metadata["actor_warmstart_only"] = True
            metadata["missing_full_state_keys"] = missing_full_state_keys
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
        if self.safety_critic is not None and payload.get("safety_critic") is not None:
            self.safety_critic.load_state_dict(payload["safety_critic"])
        if self.target_safety_critic is not None and payload.get("target_safety_critic") is not None:
            self.target_safety_critic.load_state_dict(payload["target_safety_critic"])
        if self.safety_optimizer is not None and payload.get("safety_optimizer") is not None:
            self.safety_optimizer.load_state_dict(payload["safety_optimizer"])
        self.update_step = int(payload.get("update_step", 0))
        self.update_call_step = int(payload.get("update_call_step", self.update_step))
        if payload.get("reward_scale_ema") is not None:
            self.reward_scale_ema.data.copy_(payload["reward_scale_ema"].to(self.device))
        return metadata

    def load_actor_checkpoint(self, path: str | Path) -> dict[str, object]:
        """Load actor-only weights from a BC or RL checkpoint."""

        payload = torch.load(Path(path), map_location=self.device)
        if "actor" not in payload:
            raise KeyError(f"Actor weights missing from checkpoint: {path}")
        self.actor.load_state_dict(payload["actor"])
        return payload.get("metadata", {})

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
            if self.safety_critic is not None:
                self.safety_optimizer = torch.optim.AdamW(
                    self.safety_critic.parameters(),
                    lr=self.masac_config.critic_lr,
                    weight_decay=float(getattr(self.masac_config, "critic_weight_decay", 0.0)),
                )

        if reset_entropy_state:
            self.log_alpha = torch.tensor(
                np.log(self.masac_config.init_alpha),
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


GraphFlashSACAgent = GraphMASACAgent

