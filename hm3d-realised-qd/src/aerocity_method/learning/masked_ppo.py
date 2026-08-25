"""Masked PPO baseline over the same public candidate interface as SAC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts.io import finite_number
from aerocity_method.learning.replay import CandidateTransition, pad_candidate_batch

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


class PPODependencyUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MaskedPPOConfig:
    context_dim: int
    candidate_dim: int
    hidden_dim: int = 64
    gamma: float = 0.99
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("context_dim", "candidate_dim", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gamma", "clip_ratio", "learning_rate", "value_coef", "entropy_coef"):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.clip_ratio <= 1.0:
            raise ValueError("gamma and clip_ratio must be in (0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


if nn is not None:

    class _CandidateActorCritic(nn.Module):
        def __init__(self, context_dim: int, candidate_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(context_dim + candidate_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.value = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def logits(self, context: Tensor, candidates: Tensor) -> Tensor:
            count = candidates.shape[1]
            context_expanded = context.unsqueeze(1).expand(-1, count, -1)
            return self.actor(torch.cat((context_expanded, candidates), dim=-1)).squeeze(-1)

        def value_fn(self, context: Tensor) -> Tensor:
            return self.value(context).squeeze(-1)


class MaskedPPO:
    def __init__(self, config: MaskedPPOConfig, *, seed: int = 0) -> None:
        if torch is None or nn is None:
            raise PPODependencyUnavailable("PyTorch is required for masked PPO")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.model = _CandidateActorCritic(
            config.context_dim, config.candidate_dim, config.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _policy(self, context: Tensor, candidates: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.model.logits(context, candidates).masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        logs = torch.zeros_like(probs)
        logs[mask] = torch.log(probs[mask].clamp_min(1e-12))
        return probs, logs

    def action_probabilities(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
        legal_mask: tuple[bool, ...],
    ) -> tuple[float, ...]:
        batch = pad_candidate_batch([context], [candidates], [legal_mask])
        with torch.no_grad():
            probs, _ = self._policy(
                torch.tensor(batch.contexts, dtype=torch.float32, device=self.device),
                torch.tensor(batch.candidates, dtype=torch.float32, device=self.device),
                torch.tensor(batch.legal_masks, dtype=torch.bool, device=self.device),
            )
        return tuple(float(value) for value in probs[0, : batch.counts[0]].cpu().tolist())

    def select_action(
        self,
        context: tuple[float, ...],
        candidates: tuple[tuple[float, ...], ...],
        legal_mask: tuple[bool, ...],
        *,
        deterministic: bool = False,
    ) -> int:
        probabilities = self.action_probabilities(context, candidates, legal_mask)
        if deterministic:
            return max(range(len(probabilities)), key=lambda index: probabilities[index])
        return int(
            torch.multinomial(
                torch.tensor(probabilities, dtype=torch.float32), 1, generator=self.generator
            ).item()
        )

    def update(
        self,
        transitions: tuple[CandidateTransition, ...],
        *,
        old_action_log_probs: tuple[float, ...] | None = None,
        advantages: tuple[float, ...] | None = None,
        returns: tuple[float, ...] | None = None,
    ) -> dict[str, float]:
        if not transitions:
            raise ValueError("PPO update requires transitions")
        rows = tuple(transitions)
        batch = pad_candidate_batch(
            [row.context for row in rows],
            [row.candidates for row in rows],
            [row.legal_mask for row in rows],
        )
        context = torch.tensor(batch.contexts, dtype=torch.float32, device=self.device)
        candidates = torch.tensor(batch.candidates, dtype=torch.float32, device=self.device)
        mask = torch.tensor(batch.legal_masks, dtype=torch.bool, device=self.device)
        actions = torch.tensor([row.action for row in rows], dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [row.reward for row in rows], dtype=torch.float32, device=self.device
        )
        if returns is None:
            target_returns = rewards
        else:
            target_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = self.model.value_fn(context)
        if advantages is None:
            adv = target_returns - values.detach()
        else:
            adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        probs, logs = self._policy(context, candidates, mask)
        selected_logs = logs.gather(1, actions.unsqueeze(1)).squeeze(1)
        if old_action_log_probs is None:
            old_logs = selected_logs.detach()
        else:
            old_logs = torch.tensor(old_action_log_probs, dtype=torch.float32, device=self.device)
        ratio = torch.exp(selected_logs - old_logs)
        clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
        value_loss = torch.nn.functional.mse_loss(values, target_returns)
        entropy = -(probs * logs).sum(dim=1).mean()
        loss = (
            policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "ratio_mean": float(ratio.mean().detach().cpu()),
        }
        if any(not math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("PPO update produced non-finite diagnostics")
        return diagnostics


__all__ = ["MaskedPPO", "MaskedPPOConfig", "PPODependencyUnavailable"]
