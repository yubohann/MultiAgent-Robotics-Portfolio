"""Masked, duration-aware discrete SAC compatibility kernel.

The public behavior is based on tests and semantics from the locally owned
``md_qd_swarm.method.rb_sf_sac`` snapshot.  The implementation is decoupled
from its old route, scene, outcome and archive models.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from aerocity_method.contracts.io import finite_number
from aerocity_method.contracts.models import ABI_VERSION
from aerocity_method.learning.replay import CandidateTransition, pad_candidate_batch

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-report paths
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


class RLDependencyUnavailable(RuntimeError):
    pass


def _checkpoint_fingerprint(payload: Any) -> str:
    """Hash nested optimizer/model state without requiring NumPy or pickle stability."""

    digest = hashlib.sha256()

    def add(value: Any) -> None:
        if torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            add(str(tensor.dtype))
            add(tuple(tensor.shape))
            raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                add(key)
                add(value[key])
        elif isinstance(value, tuple):
            digest.update(b"tuple\0")
            for child in value:
                add(child)
        elif isinstance(value, list):
            digest.update(b"list\0")
            for child in value:
                add(child)
        elif value is None:
            digest.update(b"none\0")
        elif isinstance(value, bool):
            digest.update(b"bool\1" if value else b"bool\0")
        elif isinstance(value, int):
            digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("checkpoint contains a non-finite float")
            digest.update(b"float\0" + struct.pack("!d", value))
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(b"str\0" + len(encoded).to_bytes(8, "big") + encoded)
        else:
            raise ValueError(f"checkpoint contains unsupported type {type(value).__name__}")

    add(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RBSFSACConfig:
    context_dim: int
    candidate_dim: int
    preference_dim: int = 0
    sf_dim: int = 0
    hidden_dim: int = 64
    gamma: float = 0.99
    tau: float = 0.01
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy_ratio: float = 0.7
    cost_weight: float = 0.0
    cost_limit: float | None = None
    enable_cost_critics: bool = True
    cost_multiplier_learning_rate: float = 3e-4
    initial_cost_multiplier: float = 0.1
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("context_dim", "candidate_dim", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("preference_dim", "sf_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        gamma = finite_number(self.gamma, "gamma")
        tau = finite_number(self.tau, "tau")
        if not 0.0 < gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")
        for name in ("learning_rate", "alpha_learning_rate", "initial_alpha"):
            if finite_number(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < finite_number(self.target_entropy_ratio, "target_entropy_ratio") <= 1.0:
            raise ValueError("target_entropy_ratio must be in (0, 1]")
        if finite_number(self.cost_weight, "cost_weight") < 0.0:
            raise ValueError("cost_weight must be non-negative")
        if self.cost_limit is not None and finite_number(self.cost_limit, "cost_limit") < 0.0:
            raise ValueError("cost_limit must be non-negative when provided")
        if not isinstance(self.enable_cost_critics, bool):
            raise ValueError("enable_cost_critics must be boolean")
        if not self.enable_cost_critics and (
            self.cost_weight != 0.0 or self.cost_limit is not None
        ):
            raise ValueError("cost weights/limits require enable_cost_critics=True")
        if (
            finite_number(self.cost_multiplier_learning_rate, "cost_multiplier_learning_rate")
            <= 0.0
        ):
            raise ValueError("cost_multiplier_learning_rate must be positive")
        if finite_number(self.initial_cost_multiplier, "initial_cost_multiplier") <= 0.0:
            raise ValueError("initial_cost_multiplier must be positive")


if nn is not None:

    class _CandidateNetwork(nn.Module):
        def __init__(
            self,
            context_dim: int,
            candidate_dim: int,
            preference_dim: int,
            hidden_dim: int,
            output_dim: int,
        ) -> None:
            super().__init__()
            self.output_dim = output_dim
            self.network = nn.Sequential(
                nn.Linear(context_dim + candidate_dim + preference_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, context: Tensor, candidates: Tensor, preference: Tensor) -> Tensor:
            count = candidates.shape[1]
            context_expanded = context.unsqueeze(1).expand(-1, count, -1)
            if preference.shape[-1] > 0:
                preference_expanded = preference.unsqueeze(1).expand(-1, count, -1)
                inputs = torch.cat((context_expanded, candidates, preference_expanded), dim=-1)
            else:
                inputs = torch.cat((context_expanded, candidates), dim=-1)
            output = self.network(inputs)
            return output.squeeze(-1) if self.output_dim == 1 else output


class RBSFSAC:
    def __init__(self, config: RBSFSACConfig, *, seed: int = 0) -> None:
        if torch is None or nn is None:
            raise RLDependencyUnavailable("PyTorch is required for the RB-SF-SAC learning kernel")
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        args = (
            config.context_dim,
            config.candidate_dim,
            config.preference_dim,
            config.hidden_dim,
        )
        self.actor = _CandidateNetwork(*args, 1).to(self.device)
        self.q1 = _CandidateNetwork(*args, 1).to(self.device)
        self.q2 = _CandidateNetwork(*args, 1).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).eval()
        self.target_q2 = copy.deepcopy(self.q2).eval()
        self.sf1 = self.sf2 = self.target_sf1 = self.target_sf2 = None
        self.cost1 = self.cost2 = self.target_cost1 = self.target_cost2 = None
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        if config.enable_cost_critics:
            self.cost1 = _CandidateNetwork(*args, 1).to(self.device)
            self.cost2 = _CandidateNetwork(*args, 1).to(self.device)
            self.target_cost1 = copy.deepcopy(self.cost1).eval()
            self.target_cost2 = copy.deepcopy(self.cost2).eval()
            critic_parameters += list(self.cost1.parameters()) + list(self.cost2.parameters())
        if config.sf_dim > 0:
            self.sf1 = _CandidateNetwork(*args, config.sf_dim).to(self.device)
            self.sf2 = _CandidateNetwork(*args, config.sf_dim).to(self.device)
            self.target_sf1 = copy.deepcopy(self.sf1).eval()
            self.target_sf2 = copy.deepcopy(self.sf2).eval()
            critic_parameters += list(self.sf1.parameters()) + list(self.sf2.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.learning_rate)
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha), device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_learning_rate)
        self.log_cost_multiplier = None
        self.cost_multiplier_optimizer = None
        if config.cost_limit is not None:
            if not config.enable_cost_critics:
                raise ValueError("adaptive cost multiplier requires cost critics")
            self.log_cost_multiplier = torch.tensor(
                math.log(config.initial_cost_multiplier),
                device=self.device,
                requires_grad=True,
            )
            self.cost_multiplier_optimizer = torch.optim.Adam(
                [self.log_cost_multiplier], lr=config.cost_multiplier_learning_rate
            )
        self.update_step = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @property
    def cost_multiplier(self) -> Tensor | None:
        """Return the learned safety multiplier, if constrained mode is enabled."""

        if self.log_cost_multiplier is None:
            return None
        return self.log_cost_multiplier.exp()

    def _policy(self, logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        probabilities = torch.softmax(masked_logits, dim=-1)
        probabilities = probabilities.masked_fill(~mask, 0.0)
        log_probabilities = torch.zeros_like(probabilities)
        legal = mask
        log_probabilities[legal] = torch.log(probabilities[legal].clamp_min(1e-12))
        return probabilities, log_probabilities

    def action_probabilities(
        self,
        context: Sequence[float],
        candidates: Sequence[Sequence[float]],
        legal_mask: Sequence[bool],
        preference: Sequence[float] = (),
    ) -> tuple[float, ...]:
        transition_batch = pad_candidate_batch(
            [tuple(context)],
            [tuple(tuple(row) for row in candidates)],
            [tuple(legal_mask)],
        )
        if len(context) != self.config.context_dim:
            raise ValueError("context dimension does not match config")
        if len(transition_batch.candidates[0][0]) != self.config.candidate_dim:
            raise ValueError("candidate dimension does not match config")
        if len(preference) != self.config.preference_dim:
            raise ValueError("preference dimension does not match config")
        with torch.no_grad():
            context_tensor = torch.tensor(
                transition_batch.contexts, dtype=torch.float32, device=self.device
            )
            candidate_tensor = torch.tensor(
                transition_batch.candidates, dtype=torch.float32, device=self.device
            )
            mask_tensor = torch.tensor(
                transition_batch.legal_masks, dtype=torch.bool, device=self.device
            )
            preference_tensor = torch.tensor(
                [tuple(preference)], dtype=torch.float32, device=self.device
            ).reshape(1, self.config.preference_dim)
            probabilities, _ = self._policy(
                self.actor(context_tensor, candidate_tensor, preference_tensor), mask_tensor
            )
        count = transition_batch.counts[0]
        return tuple(float(value) for value in probabilities[0, :count].cpu().tolist())

    def select_action(
        self,
        context: Sequence[float],
        candidates: Sequence[Sequence[float]],
        legal_mask: Sequence[bool],
        preference: Sequence[float] = (),
        *,
        deterministic: bool = False,
    ) -> int:
        probabilities = self.action_probabilities(context, candidates, legal_mask, preference)
        if deterministic:
            return max(range(len(probabilities)), key=lambda index: probabilities[index])
        tensor = torch.tensor(probabilities, dtype=torch.float32, device="cpu")
        return int(torch.multinomial(tensor, 1, generator=self.generator).item())

    def _batch(self, transitions: Sequence[CandidateTransition]) -> dict[str, Tensor]:
        rows = tuple(transitions)
        if not rows:
            raise ValueError("update requires at least one transition")
        for transition in rows:
            if len(transition.context) != self.config.context_dim:
                raise ValueError("transition context dimension does not match config")
            if len(transition.candidates[0]) != self.config.candidate_dim:
                raise ValueError("transition candidate dimension does not match config")
            if len(transition.preference) != self.config.preference_dim:
                raise ValueError("transition preference dimension does not match config")
            if len(transition.behavior_features) != self.config.sf_dim:
                raise ValueError("transition behavior feature dimension does not match config")
        current = pad_candidate_batch(
            [row.context for row in rows],
            [row.candidates for row in rows],
            [row.legal_mask for row in rows],
        )
        following = pad_candidate_batch(
            [row.next_context for row in rows],
            [row.next_candidates for row in rows],
            [row.next_legal_mask for row in rows],
        )

        def tensor(value: object, dtype: Any = torch.float32) -> Tensor:
            return torch.tensor(value, dtype=dtype, device=self.device)

        return {
            "context": tensor(current.contexts),
            "candidates": tensor(current.candidates),
            "mask": tensor(current.legal_masks, torch.bool),
            "preference": tensor([row.preference for row in rows]).reshape(
                len(rows), self.config.preference_dim
            ),
            "action": tensor([row.action for row in rows], torch.long),
            "reward": tensor([row.reward for row in rows]),
            "cost": tensor([row.cost for row in rows]),
            "behavior": tensor([row.behavior_features for row in rows]).reshape(
                len(rows), self.config.sf_dim
            ),
            "next_context": tensor(following.contexts),
            "next_candidates": tensor(following.candidates),
            "next_mask": tensor(following.legal_masks, torch.bool),
            "next_preference": tensor([row.next_preference for row in rows]).reshape(
                len(rows), self.config.preference_dim
            ),
            "done": tensor([float(row.done) for row in rows]),
            "duration": tensor([row.duration for row in rows]),
        }

    @staticmethod
    def _gather(values: Tensor, action: Tensor) -> Tensor:
        if values.ndim == 2:
            return values.gather(1, action.unsqueeze(1)).squeeze(1)
        index = action.view(-1, 1, 1).expand(-1, 1, values.shape[-1])
        return values.gather(1, index).squeeze(1)

    def _soft_update(self, source: Any, target: Any) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(source_parameter, alpha=self.config.tau)

    def update(self, transitions: Sequence[CandidateTransition]) -> dict[str, float]:
        batch = self._batch(transitions)
        discount = torch.pow(
            torch.full_like(batch["duration"], self.config.gamma), batch["duration"]
        )
        with torch.no_grad():
            next_probabilities, next_logs = self._policy(
                self.actor(
                    batch["next_context"],
                    batch["next_candidates"],
                    batch["next_preference"],
                ),
                batch["next_mask"],
            )
            next_q = torch.minimum(
                self.target_q1(
                    batch["next_context"], batch["next_candidates"], batch["next_preference"]
                ),
                self.target_q2(
                    batch["next_context"], batch["next_candidates"], batch["next_preference"]
                ),
            )
            next_value = (next_probabilities * (next_q - self.alpha.detach() * next_logs)).sum(
                dim=1
            )
            task_target = batch["reward"] + (1.0 - batch["done"]) * discount * next_value
            cost_target = None
            if self.config.enable_cost_critics:
                assert self.target_cost1 is not None and self.target_cost2 is not None
                next_cost = torch.minimum(
                    self.target_cost1(
                        batch["next_context"],
                        batch["next_candidates"],
                        batch["next_preference"],
                    ),
                    self.target_cost2(
                        batch["next_context"],
                        batch["next_candidates"],
                        batch["next_preference"],
                    ),
                )
                cost_target = batch["cost"] + (1.0 - batch["done"]) * discount * (
                    next_probabilities * next_cost
                ).sum(dim=1)
            sf_target = None
            if self.config.sf_dim > 0:
                assert self.target_sf1 is not None and self.target_sf2 is not None
                next_sf = 0.5 * (
                    self.target_sf1(
                        batch["next_context"], batch["next_candidates"], batch["next_preference"]
                    )
                    + self.target_sf2(
                        batch["next_context"], batch["next_candidates"], batch["next_preference"]
                    )
                )
                expected_sf = (next_probabilities.unsqueeze(-1) * next_sf).sum(dim=1)
                sf_target = (
                    batch["behavior"]
                    + ((1.0 - batch["done"]) * discount).unsqueeze(-1) * expected_sf
                )
        current_q1 = self._gather(
            self.q1(batch["context"], batch["candidates"], batch["preference"]), batch["action"]
        )
        current_q2 = self._gather(
            self.q2(batch["context"], batch["candidates"], batch["preference"]), batch["action"]
        )
        critic_loss = torch.nn.functional.mse_loss(current_q1, task_target)
        critic_loss = critic_loss + torch.nn.functional.mse_loss(current_q2, task_target)
        cost_loss = torch.zeros((), device=self.device)
        total_critic_loss = critic_loss
        if self.config.enable_cost_critics:
            assert self.cost1 is not None and self.cost2 is not None and cost_target is not None
            current_cost1 = self._gather(
                self.cost1(batch["context"], batch["candidates"], batch["preference"]),
                batch["action"],
            )
            current_cost2 = self._gather(
                self.cost2(batch["context"], batch["candidates"], batch["preference"]),
                batch["action"],
            )
            cost_loss = torch.nn.functional.mse_loss(current_cost1, cost_target)
            cost_loss = cost_loss + torch.nn.functional.mse_loss(current_cost2, cost_target)
            total_critic_loss = total_critic_loss + cost_loss
        sf_loss = torch.zeros((), device=self.device)
        if self.config.sf_dim > 0:
            assert self.sf1 is not None and self.sf2 is not None and sf_target is not None
            current_sf1 = self._gather(
                self.sf1(batch["context"], batch["candidates"], batch["preference"]),
                batch["action"],
            )
            current_sf2 = self._gather(
                self.sf2(batch["context"], batch["candidates"], batch["preference"]),
                batch["action"],
            )
            sf_loss = torch.nn.functional.mse_loss(current_sf1, sf_target)
            sf_loss = sf_loss + torch.nn.functional.mse_loss(current_sf2, sf_target)
            total_critic_loss = total_critic_loss + sf_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        total_critic_loss.backward()
        critic_parameters = [
            parameter
            for group in self.critic_optimizer.param_groups
            for parameter in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(
            critic_parameters,
            max_norm=10.0,
        )
        self.critic_optimizer.step()

        probabilities, logs = self._policy(
            self.actor(batch["context"], batch["candidates"], batch["preference"]),
            batch["mask"],
        )
        with torch.no_grad():
            q_value = torch.minimum(
                self.q1(batch["context"], batch["candidates"], batch["preference"]),
                self.q2(batch["context"], batch["candidates"], batch["preference"]),
            )
            if self.config.enable_cost_critics:
                assert self.cost1 is not None and self.cost2 is not None
                cost_value = torch.minimum(
                    self.cost1(batch["context"], batch["candidates"], batch["preference"]),
                    self.cost2(batch["context"], batch["candidates"], batch["preference"]),
                )
            else:
                cost_value = torch.zeros_like(q_value)
        actor_terms = self.alpha.detach() * logs - q_value
        multiplier = self.cost_multiplier
        actor_cost_weight = self.config.cost_weight if multiplier is None else multiplier.detach()
        actor_terms = actor_terms + actor_cost_weight * cost_value
        actor_loss = (probabilities * actor_terms).sum(dim=1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        dual_loss = torch.zeros((), device=self.device)
        cost_violation = torch.zeros((), device=self.device)
        if (
            self.config.cost_limit is not None
            and self.log_cost_multiplier is not None
            and self.cost_multiplier_optimizer is not None
        ):
            cost_violation = cost_value.mean() - float(self.config.cost_limit)
            # Minimize -lambda * (observed cost - limit), so lambda rises only
            # when the measured execution cost exceeds the configured budget.
            dual_loss = -self.cost_multiplier * cost_violation
            self.cost_multiplier_optimizer.zero_grad(set_to_none=True)
            dual_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_cost_multiplier], max_norm=5.0)
            self.cost_multiplier_optimizer.step()
            with torch.no_grad():
                self.log_cost_multiplier.clamp_(-10.0, 10.0)

        entropy = -(probabilities * logs).sum(dim=1)
        legal_counts = batch["mask"].sum(dim=1).float()
        target_entropy = self.config.target_entropy_ratio * torch.log(legal_counts)
        alpha_loss = (self.log_alpha * (entropy.detach() - target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update_pairs = [(self.q1, self.target_q1), (self.q2, self.target_q2)]
        if self.config.enable_cost_critics:
            assert self.cost1 is not None and self.cost2 is not None
            assert self.target_cost1 is not None and self.target_cost2 is not None
            soft_update_pairs.extend(
                [(self.cost1, self.target_cost1), (self.cost2, self.target_cost2)]
            )
        for source, target in soft_update_pairs:
            self._soft_update(source, target)
        if self.config.sf_dim > 0:
            assert self.sf1 is not None and self.sf2 is not None
            assert self.target_sf1 is not None and self.target_sf2 is not None
            self._soft_update(self.sf1, self.target_sf1)
            self._soft_update(self.sf2, self.target_sf2)
        self.update_step += 1
        diagnostics = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "cost_loss": float(cost_loss.detach().cpu()),
            "sf_loss": float(sf_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "cost_multiplier": float(
                self.cost_multiplier.detach().cpu()
                if self.cost_multiplier is not None
                else self.config.cost_weight
            ),
            "cost_violation": float(cost_violation.detach().cpu()),
            "dual_loss": float(dual_loss.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
            "task_target_mean": float(task_target.mean().detach().cpu()),
        }
        if any(not math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("RB-SF-SAC update produced non-finite diagnostics")
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        state = copy.deepcopy(
            {
                "schema_version": ABI_VERSION,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "cost1": None if self.cost1 is None else self.cost1.state_dict(),
                "cost2": None if self.cost2 is None else self.cost2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "target_cost1": (
                    None if self.target_cost1 is None else self.target_cost1.state_dict()
                ),
                "target_cost2": (
                    None if self.target_cost2 is None else self.target_cost2.state_dict()
                ),
                "sf1": None if self.sf1 is None else self.sf1.state_dict(),
                "sf2": None if self.sf2 is None else self.sf2.state_dict(),
                "target_sf1": None if self.target_sf1 is None else self.target_sf1.state_dict(),
                "target_sf2": None if self.target_sf2 is None else self.target_sf2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
                "cost_multiplier_optimizer": (
                    None
                    if self.cost_multiplier_optimizer is None
                    else self.cost_multiplier_optimizer.state_dict()
                ),
                "log_cost_multiplier": (
                    None
                    if self.log_cost_multiplier is None
                    else self.log_cost_multiplier.detach().clone()
                ),
                "update_step": self.update_step,
                "generator_state": self.generator.get_state(),
                "torch_rng_state": torch.random.get_rng_state(),
            }
        )
        state["checkpoint_hash"] = _checkpoint_fingerprint(state)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != ABI_VERSION:
            raise ValueError("checkpoint schema version does not match current RB-SF-SAC ABI")
        supplied_hash = state.get("checkpoint_hash")
        unsigned = {key: value for key, value in state.items() if key != "checkpoint_hash"}
        if not isinstance(supplied_hash, str) or _checkpoint_fingerprint(unsigned) != supplied_hash:
            raise ValueError("RB-SF-SAC checkpoint content hash mismatch")
        loaded = copy.deepcopy(state)
        if loaded.get("config") != asdict(self.config):
            raise ValueError("checkpoint config does not match current RB-SF-SAC ABI")
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(loaded[name])
        if self.config.enable_cost_critics:
            for name in ("cost1", "cost2", "target_cost1", "target_cost2"):
                module = getattr(self, name)
                assert module is not None
                if loaded.get(name) is None:
                    raise ValueError("checkpoint is missing enabled cost critic state")
                module.load_state_dict(loaded[name])
        elif any(
            loaded.get(name) is not None
            for name in ("cost1", "cost2", "target_cost1", "target_cost2")
        ):
            raise ValueError("checkpoint contains cost critic state but config disables it")
        if self.config.sf_dim > 0:
            for name in ("sf1", "sf2", "target_sf1", "target_sf2"):
                module = getattr(self, name)
                assert module is not None
                module.load_state_dict(loaded[name])
        self.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
        self.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(loaded["alpha_optimizer"])
        if self.cost_multiplier_optimizer is not None:
            if (
                loaded.get("cost_multiplier_optimizer") is None
                or loaded.get("log_cost_multiplier") is None
            ):
                raise ValueError("adaptive cost multiplier checkpoint state is incomplete")
            self.cost_multiplier_optimizer.load_state_dict(loaded["cost_multiplier_optimizer"])
        elif loaded.get("cost_multiplier_optimizer") is not None:
            raise ValueError("checkpoint contains adaptive cost state but config disables it")
        with torch.no_grad():
            self.log_alpha.copy_(loaded["log_alpha"].to(self.device))
            if self.log_cost_multiplier is not None:
                self.log_cost_multiplier.copy_(loaded["log_cost_multiplier"].to(self.device))
        self.update_step = int(loaded["update_step"])
        self.generator.set_state(loaded["generator_state"].cpu())
        torch.random.set_rng_state(loaded["torch_rng_state"].cpu())
