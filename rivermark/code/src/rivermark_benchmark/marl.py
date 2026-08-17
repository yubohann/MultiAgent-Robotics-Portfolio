"""Train and execute a real public-observation multi-agent pilot policy.

The environment is a PettingZoo parallel environment over the ordinary
Rivermark kinematic runtime.  Each agent receives its own public state plus a
fixed summary of public team messages.  A small shared-parameter, decentralized
PPO-style actor-critic is trained from those rollouts.  This is a genuine local
MARL path, not a claim of skrl, RLlib, MAPPO, or Isaac training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import detect_source_provenance
from .runtime import HighLevelAction, PilotRuntimeConfig, PilotSwarmRuntime, PublicMission, PublicObservation

try:
    import gymnasium
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - exercised by CLI fail-closed behavior.
    gymnasium = None
    spaces = None
    ParallelEnv = object

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by CLI fail-closed behavior.
    torch = None
    nn = None


METADATA_SCHEMA = "org.rivermark.shared-marl-pilot.v1"
OBSERVATION_DIM = 16
ACTION_DIM = 4
ACTION_SCALE = np.asarray((2.3, 2.3, 1.25, 1.4), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_training_dependencies() -> Any:
    if gymnasium is None or spaces is None or torch is None or nn is None:
        raise RuntimeError("PettingZoo, Gymnasium, and PyTorch are required for shared MARL pilot training")
    return torch


def _message_vector(
    observation: PublicObservation,
    *,
    bounds_xy_m: tuple[float, float],
) -> np.ndarray:
    """Summarize public teammate messages without inspecting runtime internals."""

    own = np.asarray(observation.proprioception, dtype=np.float32)
    position_rows: list[np.ndarray] = []
    velocity_rows: list[np.ndarray] = []
    for message in observation.public_team_messages:
        position = message.get("position_m")
        velocity = message.get("velocity_mps")
        if isinstance(position, list) and len(position) == 3 and isinstance(velocity, list) and len(velocity) == 3:
            position_array = np.asarray(position, dtype=np.float32)
            velocity_array = np.asarray(velocity, dtype=np.float32)
            if np.all(np.isfinite(position_array)) and np.all(np.isfinite(velocity_array)):
                position_rows.append(position_array)
                velocity_rows.append(velocity_array)
    own_scale = np.asarray((bounds_xy_m[0], bounds_xy_m[1], 5.0, 2.8, 2.8, 1.55, np.pi, 1.5), dtype=np.float32)
    normalized_own = np.clip(np.nan_to_num(own) / own_scale, -2.0, 2.0)
    if not position_rows:
        return np.concatenate((normalized_own, np.zeros(8, dtype=np.float32))).astype(np.float32, copy=False)
    positions = np.stack(position_rows)
    velocities = np.stack(velocity_rows)
    relative_position = (positions - own[:3]).mean(axis=0) / np.asarray(
        (bounds_xy_m[0], bounds_xy_m[1], 5.0), dtype=np.float32
    )
    relative_velocity = (velocities - own[3:6]).mean(axis=0) / np.asarray((2.8, 2.8, 1.55), dtype=np.float32)
    nearest_distance = float(np.linalg.norm(positions[:, :2] - own[None, :2], axis=1).min())
    summary = np.concatenate(
        (
            np.clip(relative_position, -2.0, 2.0),
            np.clip(relative_velocity, -2.0, 2.0),
            np.asarray((min(1.0, nearest_distance / 8.0), min(1.0, len(positions) / 31.0)), dtype=np.float32),
        )
    )
    return np.concatenate((normalized_own, summary)).astype(np.float32, copy=False)


class RivermarkParallelStateEnv(ParallelEnv):
    """PettingZoo parallel wrapper with decentralized public observations.

    The reward is intentionally built from public trajectory movement, public
    team separation, public cells, and safety events.  It never scores hidden
    target matches and it is never attached to a policy observation.
    """

    metadata = {"name": "rivermark_parallel_state_v1", "is_parallelizable": True}

    def __init__(self, *, agent_count: int = 4, max_steps: int = 36, seed: int = 20260722) -> None:
        _require_training_dependencies()
        if agent_count < 2:
            raise ValueError("shared MARL pilot training requires at least two agents")
        self.agent_count = agent_count
        self.max_steps = max_steps
        self.seed_value = seed
        self.possible_agents = [f"drone_{agent_id}" for agent_id in range(agent_count)]
        self.agents = self.possible_agents[:]
        self.runtime = self._new_runtime(seed)
        self._observations: Mapping[int, PublicObservation] = {}
        self._previous_positions: dict[int, np.ndarray] = {}
        self._visited_cells: set[str] = set()

    def _new_runtime(self, seed: int) -> PilotSwarmRuntime:
        return PilotSwarmRuntime(
            PilotRuntimeConfig(agent_count=self.agent_count, max_steps=self.max_steps, seed=seed),
            information_profile="state_only",
        )

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Any:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        assert spaces is not None
        return spaces.Box(low=-2.0, high=2.0, shape=(OBSERVATION_DIM,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Any:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        assert spaces is not None
        return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def _name(self, agent_id: int) -> str:
        return f"drone_{agent_id}"

    def _agent_id(self, agent: str) -> int:
        if agent not in self.possible_agents:
            raise KeyError(f"unknown agent {agent!r}")
        return int(agent.removeprefix("drone_"))

    def _features(self, agent_id: int) -> np.ndarray:
        return _message_vector(
            self._observations[agent_id],
            bounds_xy_m=self.runtime.mission.bounds_xy_m,
        )

    def _cell(self, position: np.ndarray) -> str:
        width, height = self.runtime.config.world_size_xy_m
        return f"{min(7, int(np.clip(position[0] / width * 8, 0, 7)))}:{min(5, int(np.clip(position[1] / height * 6, 0, 5)))}"

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None):
        del options
        if seed is not None:
            self.seed_value = int(seed)
        self.runtime = self._new_runtime(self.seed_value)
        self._observations = self.runtime.reset()
        self.agents = self.possible_agents[:]
        self._previous_positions = {
            agent_id: observation.proprioception[:3].astype(np.float64, copy=True)
            for agent_id, observation in self._observations.items()
        }
        self._visited_cells = {self._cell(position) for position in self._previous_positions.values()}
        observations = {self._name(agent_id): self._features(agent_id) for agent_id in range(self.agent_count)}
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions: Mapping[str, np.ndarray]):
        if not self.agents:
            raise RuntimeError("reset must be called before stepping a completed parallel environment")
        if set(actions) != set(self.agents):
            raise ValueError("parallel environment requires exactly one action for every live agent")
        commands: dict[int, HighLevelAction] = {}
        for agent, raw_action in actions.items():
            vector = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            if vector.shape != (ACTION_DIM,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{agent} emitted a non-finite 4D action")
            command = np.clip(vector, -1.0, 1.0) * ACTION_SCALE
            commands[self._agent_id(agent)] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in command[:3]),
                yaw_rate_rad_s=float(command[3]),
                mode="transit",
                source="shared_marl_training_policy",
            )
        self._observations, frame = self.runtime.step(commands)
        positions = {
            agent_id: observation.proprioception[:3].astype(np.float64, copy=True)
            for agent_id, observation in self._observations.items()
        }
        newly_visited = {self._cell(position) for position in positions.values()} - self._visited_cells
        self._visited_cells.update(newly_visited)
        rewards: dict[str, float] = {}
        infos: dict[str, dict[str, float]] = {}
        for agent_id, position in positions.items():
            name = self._name(agent_id)
            movement = float(np.linalg.norm(position - self._previous_positions[agent_id]))
            peers = [other for other_id, other in positions.items() if other_id != agent_id]
            nearest = min((float(np.linalg.norm(position[:2] - other[:2])) for other in peers), default=8.0)
            safety = float(sum(event.agent_id == agent_id for event in frame.safety_events))
            novelty = 0.075 if self._cell(position) in newly_visited else 0.0
            altitude_cost = 0.025 * abs(float(position[2]) - 2.8)
            reward = 0.14 * movement + 0.06 * min(1.0, nearest / 5.0) + novelty - 1.1 * safety - altitude_cost
            rewards[name] = float(reward)
            infos[name] = {
                "public_movement_m": movement,
                "public_nearest_peer_m": nearest,
                "public_cell_novelty": novelty,
                "safety_events": safety,
            }
        self._previous_positions = positions
        finished = self.runtime.done
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: finished for agent in self.agents}
        observations = {} if finished else {
            self._name(agent_id): self._features(agent_id) for agent_id in range(self.agent_count)
        }
        if finished:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


class SharedMarlActorCritic(nn.Module if nn is not None else object):
    """A compact shared actor/critic with decentralized public inputs."""

    def __init__(self) -> None:
        _require_training_dependencies()
        super().__init__()
        assert nn is not None
        self.trunk = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, ACTION_DIM)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.65))

    def forward(self, observations: Any) -> tuple[Any, Any]:
        features = self.trunk(observations)
        return torch.tanh(self.actor(features)), self.critic(features).squeeze(-1)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    library = _require_training_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as stream:
        temporary = Path(stream.name)
    try:
        library.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class MarlTrainResult:
    checkpoint: Path
    metadata: Path
    updates: int
    final_mean_public_reward: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "metadata": str(self.metadata),
            "updates": self.updates,
            "final_mean_public_reward": self.final_mean_public_reward,
        }


def train_shared_marl(
    output: Path,
    *,
    updates: int,
    agent_count: int,
    episode_steps: int,
    learning_rate: float,
    ppo_epochs: int,
    minibatch_size: int,
    seed: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> MarlTrainResult:
    """Train a shared decentralized actor-critic from PettingZoo rollouts."""

    library = _require_training_dependencies()
    if updates < 1 or episode_steps < 2 or ppo_epochs < 1 or minibatch_size < 1 or learning_rate <= 0.0:
        raise ValueError("updates, episode_steps, ppo_epochs, minibatch_size, and learning_rate must be positive")
    random.seed(seed)
    np.random.seed(seed)
    library.manual_seed(seed)
    model = SharedMarlActorCritic()
    optimizer = library.optim.Adam(model.parameters(), lr=learning_rate)
    environment = RivermarkParallelStateEnv(agent_count=agent_count, max_steps=episode_steps, seed=seed)
    reward_history: list[float] = []
    for update in range(updates):
        observations, _ = environment.reset(seed=seed + update)
        feature_rows: list[Any] = []
        raw_action_rows: list[Any] = []
        old_log_prob_rows: list[Any] = []
        value_rows: list[Any] = []
        reward_rows: list[np.ndarray] = []
        done_rows: list[np.ndarray] = []
        while environment.agents:
            agents = environment.agents[:]
            features = library.from_numpy(np.stack([observations[agent] for agent in agents])).float()
            with library.no_grad():
                means, values = model(features)
                distribution = library.distributions.Normal(means, model.log_std.exp().expand_as(means))
                raw_actions = distribution.sample()
                log_probabilities = distribution.log_prob(raw_actions).sum(dim=1)
            bounded_actions = raw_actions.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
            next_observations, rewards, terminations, truncations, _ = environment.step(
                {agent: bounded_actions[index] for index, agent in enumerate(agents)}
            )
            feature_rows.append(features)
            raw_action_rows.append(raw_actions)
            old_log_prob_rows.append(log_probabilities)
            value_rows.append(values)
            reward_rows.append(np.asarray([rewards[agent] for agent in agents], dtype=np.float32))
            done_rows.append(
                np.asarray(
                    [float(terminations[agent] or truncations[agent]) for agent in agents],
                    dtype=np.float32,
                )
            )
            observations = next_observations
        features_tensor = library.cat(feature_rows).detach()
        raw_actions_tensor = library.cat(raw_action_rows).detach()
        old_log_probabilities = library.cat(old_log_prob_rows).detach()
        values = library.stack(value_rows).cpu().numpy().astype(np.float32)
        rewards_array = np.stack(reward_rows)
        dones_array = np.stack(done_rows)
        advantages = np.zeros_like(rewards_array, dtype=np.float32)
        generalized_advantage = np.zeros(agent_count, dtype=np.float32)
        next_values = np.zeros(agent_count, dtype=np.float32)
        for step_index in range(len(reward_rows) - 1, -1, -1):
            alive = 1.0 - dones_array[step_index]
            delta = rewards_array[step_index] + gamma * next_values * alive - values[step_index]
            generalized_advantage = delta + gamma * gae_lambda * alive * generalized_advantage
            advantages[step_index] = generalized_advantage
            next_values = values[step_index]
        returns_tensor = library.from_numpy((advantages + values).reshape(-1)).float()
        advantages_tensor = library.from_numpy(advantages.reshape(-1)).float()
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-6)
        sample_count = int(features_tensor.shape[0])
        for _ in range(ppo_epochs):
            permutation = library.randperm(sample_count)
            for start in range(0, sample_count, min(minibatch_size, sample_count)):
                indices = permutation[start : start + min(minibatch_size, sample_count)]
                means, predicted_values = model(features_tensor[indices])
                distribution = library.distributions.Normal(means, model.log_std.exp().expand_as(means))
                log_probabilities = distribution.log_prob(raw_actions_tensor[indices]).sum(dim=1)
                ratio = (log_probabilities - old_log_probabilities[indices]).exp()
                unclipped = ratio * advantages_tensor[indices]
                clipped = ratio.clamp(0.8, 1.2) * advantages_tensor[indices]
                policy_loss = -library.minimum(unclipped, clipped).mean()
                value_loss = library.nn.functional.mse_loss(predicted_values, returns_tensor[indices])
                entropy = distribution.entropy().sum(dim=1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.001 * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                library.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        reward_history.append(float(rewards_array.mean()))
    checkpoint = output.resolve()
    _atomic_torch_save(checkpoint, {"state_dict": model.state_dict()})
    metadata_path = checkpoint.with_suffix(".rivermark.json")
    source = detect_source_provenance()
    metadata = {
        "schema": METADATA_SCHEMA,
        "model_kind": "shared_decentralized_actor_critic",
        "implementation_kind": "trained_torch_marl_pilot_checkpoint",
        "training_backend": "rivermark-kinematic-pilot-v1",
        "formal_benchmark_admission": False,
        "information_profile": "state_only",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_scale": ACTION_SCALE.tolist(),
        "observation_contract": "own_public_proprioception_plus_public_team_message_summary",
        "policy_parameter_sharing": "one_shared_policy_for_all_agents",
        "centralized_critic": False,
        "training_algorithm": "shared_parameter_decentralized_ppo_style_actor_critic",
        "reward_contract": "public_movement_team_separation_public_cell_novelty_safety_only",
        "reward_uses_evaluator_private_truth": False,
        "pettingzoo_environment": RivermarkParallelStateEnv.metadata["name"],
        "pettingzoo_version": __import__("pettingzoo").__version__,
        "torch_version": library.__version__,
        "updates": updates,
        "agent_count": agent_count,
        "episode_steps": episode_steps,
        "learning_rate": learning_rate,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "seed": seed,
        "final_mean_public_training_reward": reward_history[-1],
        "source_revision": source.source_revision,
        "source_tree_sha256": source.source_tree_sha256,
        "source_worktree_dirty": source.source_worktree_dirty,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    _atomic_json(metadata_path, metadata)
    return MarlTrainResult(checkpoint, metadata_path, updates, reward_history[-1])


class SharedMarlCheckpointPolicy:
    """Provenance-checked shared policy for decentralized state-only rollout."""

    method_id = "shared_marl_actor_critic_checkpoint"

    def __init__(self, checkpoint: Path, metadata_path: Path | None = None) -> None:
        library = _require_training_dependencies()
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"shared MARL checkpoint is missing: {self.checkpoint}")
        self.metadata_path = (metadata_path or self.checkpoint.with_suffix(".rivermark.json")).resolve()
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"shared MARL metadata is missing: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != METADATA_SCHEMA:
            raise ValueError("unsupported shared MARL metadata schema")
        if self.metadata.get("model_kind") != "shared_decentralized_actor_critic":
            raise ValueError("shared MARL metadata has an unexpected model kind")
        if self.metadata.get("information_profile") != "state_only":
            raise ValueError("shared MARL checkpoint must declare state_only")
        if self.metadata.get("observation_dim") != OBSERVATION_DIM or self.metadata.get("action_dim") != ACTION_DIM:
            raise ValueError("shared MARL metadata has incompatible tensor dimensions")
        if self.metadata.get("checkpoint_sha256") != sha256_file(self.checkpoint):
            raise ValueError("shared MARL checkpoint SHA-256 does not match its metadata")
        try:
            payload = library.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # Older torch releases do not implement weights_only.
            payload = library.load(self.checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("shared MARL checkpoint lacks a state_dict")
        self.model = SharedMarlActorCritic()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.action_scale = np.asarray(self.metadata.get("action_scale"), dtype=np.float32)
        if self.action_scale.shape != (ACTION_DIM,) or np.any(self.action_scale <= 0.0):
            raise ValueError("shared MARL metadata needs positive 4D action_scale")
        self.mission: PublicMission | None = None
        self.agent_count = 0

    def reset(
        self,
        mission: PublicMission,
        agent_count: int,
        *,
        public_geometry: Mapping[str, Any] | None = None,
    ) -> None:
        del public_geometry
        self.mission = mission
        self.agent_count = agent_count

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        if self.mission is None:
            raise RuntimeError("reset must be called before a shared MARL policy acts")
        if set(observations) != set(range(self.agent_count)):
            raise RuntimeError("shared MARL policy requires one state-only observation per configured agent")
        if any(observation.information_profile != "state_only" for observation in observations.values()):
            raise RuntimeError("shared MARL checkpoint received a mismatched information profile")
        library = _require_training_dependencies()
        features = np.stack(
            [
                _message_vector(observations[agent_id], bounds_xy_m=self.mission.bounds_xy_m)
                for agent_id in sorted(observations)
            ]
        )
        with library.no_grad():
            means, _ = self.model(library.from_numpy(features).float())
        commands = means.cpu().numpy().astype(np.float32) * self.action_scale
        return {
            agent_id: HighLevelAction(
                velocity_xyz=tuple(float(value) for value in command[:3]),
                yaw_rate_rad_s=float(command[3]),
                mode="transit",
                source=self.method_id,
            )
            for agent_id, command in zip(sorted(observations), commands)
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "implementation_kind": "trained_torch_marl_pilot_checkpoint",
            "external_dependency": "torch,pettingzoo",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "adapter_metadata": str(self.metadata_path),
            "adapter_metadata_sha256": sha256_file(self.metadata_path),
            "policy_parameter_sharing": self.metadata["policy_parameter_sharing"],
            "centralized_critic": self.metadata["centralized_critic"],
            "reward_uses_evaluator_private_truth": self.metadata["reward_uses_evaluator_private_truth"],
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = train_shared_marl(
        args.output,
        updates=args.updates,
        agent_count=args.agents,
        episode_steps=args.episode_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
    )
    print(json.dumps({"status": "completed", **result.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
