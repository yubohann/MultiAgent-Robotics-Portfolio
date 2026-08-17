"""Train a real SB3 PPO or SAC policy on the state-only Rivermark pilot ABI.

This command is intentionally a training utility, not a benchmark result
generator.  It produces a checkpoint plus required adapter metadata.  Running
the trained checkpoint still requires the normal online recorder and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .provenance import detect_source_provenance
from .runtime import HighLevelAction, PilotRuntimeConfig, PilotSwarmRuntime


SB3_ADAPTER_V2_SCHEMA = "org.rivermark.sb3-adapter.v2"
STATE_ONLY_PROPRIOCEPTION_ABI = "org.rivermark.state-only-proprioception.v1"
STATE_ONLY_VELOCITY_ACTION_ABI = "org.rivermark.state-only-velocity-yaw-action.v1"
CITY_LITE_TRANSFER_COORDINATE_CONTRACT = (
    "citylite_route_anchor_heading_to_pilot_v1"
)


def _distribution_version(distribution: str) -> str:
    """Return the installed package version without making metadata optional."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required training distribution is missing: {distribution}") from exc


def _runtime_versions() -> dict[str, str]:
    """Record every runtime needed to reload this third-party checkpoint."""

    return {
        "python": platform.python_version(),
        "numpy": _distribution_version("numpy"),
        "gymnasium": _distribution_version("gymnasium"),
        "stable_baselines3": _distribution_version("stable-baselines3"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]
except ImportError:  # Keep module importable when training extras are absent.
    gym = None
    spaces = None


class _SingleAgentStateEnv(gym.Env if gym is not None else object):
    """Gymnasium adapter with six non-learning coverage peers.

    The learned agent sees only its own public state.  Peers use deterministic
    public coverage actions, so no hidden target/reward signal enters their
    policy input.  Reward is training-only and never becomes policy-visible
    rollout data during evaluation.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, seed: int, agent_count: int, max_steps: int) -> None:
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required to train an SB3 policy")
        super().__init__()
        self.seed_value = seed
        self.agent_count = agent_count
        self.max_steps = max_steps
        self.runtime = PilotSwarmRuntime(
            PilotRuntimeConfig(agent_count=agent_count, max_steps=max_steps, seed=seed),
            information_profile="state_only",
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self._observations: dict[int, Any] = {}
        self._last_position = np.zeros(3, dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
        self.runtime = PilotSwarmRuntime(
            PilotRuntimeConfig(agent_count=self.agent_count, max_steps=self.max_steps, seed=self.seed_value),
            information_profile="state_only",
        )
        self._observations = dict(self.runtime.reset())
        self._last_position = self._observations[0].proprioception[:3].astype(np.float64)
        return self._observations[0].proprioception.copy(), {}

    def _peer_action(self, agent_id: int) -> HighLevelAction:
        state = self._observations[agent_id].proprioception
        width, height = self.runtime.config.world_size_xy_m
        lane_y = 1.8 + (agent_id + 0.5) * (height - 3.6) / self.agent_count
        direction = 1.0 if (self.runtime.current_frame().step_index // 10 + agent_id) % 2 == 0 else -1.0
        desired = np.array((direction * 1.7, (lane_y - state[1]) * 0.45, (2.8 - state[2]) * 0.7))
        return HighLevelAction(tuple(float(value) for value in desired), source="training_public_peer")

    def step(self, action: np.ndarray):
        vector = np.asarray(action, dtype=np.float32).reshape(4)
        commands = {
            0: HighLevelAction(
                velocity_xyz=tuple(float(value) for value in vector[:3] * np.array((2.3, 2.3, 1.25))),
                yaw_rate_rad_s=float(vector[3] * 1.4),
                source="sb3_training_policy",
            )
        }
        commands.update({agent_id: self._peer_action(agent_id) for agent_id in range(1, self.agent_count)})
        self._observations, frame = self.runtime.step(commands)
        position = self._observations[0].proprioception[:3].astype(np.float64)
        movement = float(np.linalg.norm(position - self._last_position))
        self._last_position = position
        lidar = frame.sensor_packets[0].lidar_ranges_m
        safety_penalty = 1.5 if any(event.agent_id == 0 for event in frame.safety_events) else 0.0
        clearance_bonus = float(min(1.0, np.min(lidar) / 2.0)) * 0.06
        novelty = 0.08 * movement
        reward = novelty + clearance_bonus - safety_penalty
        terminated = False
        truncated = self.runtime.done
        return self._observations[0].proprioception.copy(), reward, terminated, truncated, {
            "training_reward": reward,
            "public_movement_m": movement,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "checkpoints" / "state_only_ppo.zip")
    parser.add_argument(
        "--transfer-coordinate-contract",
        choices=("none", CITY_LITE_TRANSFER_COORDINATE_CONTRACT),
        default="none",
        help=(
            "Declare a reviewed City-Lite-to-pilot coordinate transform for a "
            "development-only physical control-transfer smoke. This never marks "
            "the checkpoint as Isaac-trained or benchmark-admissible."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.timesteps < 1 or args.agents < 2 or args.episode_steps < 2:
        raise SystemExit("--timesteps must be positive; --agents and --episode-steps must be at least two")
    try:
        from stable_baselines3 import PPO, SAC  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("Stable-Baselines3 is unavailable in this Python environment") from exc
    environment = _SingleAgentStateEnv(seed=args.seed, agent_count=args.agents, max_steps=args.episode_steps)
    model_class = PPO if args.algorithm == "ppo" else SAC
    kwargs: dict[str, Any] = {"seed": args.seed, "verbose": 0, "device": "cpu"}
    if args.algorithm == "ppo":
        n_steps = min(512, args.episode_steps * 4)
        # PPO's rollout buffer is ``n_steps`` for this one-environment pilot.
        # Pick a divisor so every update uses complete minibatches.
        batch_size = next(candidate for candidate in (64, 32, 16, 8, 4, 2, 1) if candidate <= n_steps and n_steps % candidate == 0)
        kwargs.update({"n_steps": n_steps, "batch_size": batch_size})
    model = model_class("MlpPolicy", environment, **kwargs)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output.with_suffix("")))
    checkpoint = args.output if args.output.suffix == ".zip" else args.output.with_suffix(".zip")
    metadata_path = checkpoint.with_suffix(".rivermark.json")
    source = detect_source_provenance()
    transfer_contract = str(args.transfer_coordinate_contract)
    metadata = {
        "schema": SB3_ADAPTER_V2_SCHEMA,
        "implementation_kind": "trained_sb3_pilot_checkpoint",
        "algorithm": args.algorithm,
        "information_profile": "state_only",
        "observation_mean": [0.0] * 8,
        "observation_std": [1.0] * 8,
        "action_scale": [2.3, 2.3, 1.25, 1.4],
        "observation_abi": {
            "schema": STATE_ONLY_PROPRIOCEPTION_ABI,
            "shape": [8],
            "fields": [
                "position_x_m",
                "position_y_m",
                "position_z_m",
                "velocity_x_mps",
                "velocity_y_mps",
                "velocity_z_mps",
                "yaw_rad",
                "yaw_rate_radps",
            ],
            "coordinate_frame": "pilot_world_right_handed_z_up",
        },
        "action_abi": {
            "schema": STATE_ONLY_VELOCITY_ACTION_ABI,
            "shape": [4],
            "fields": ["velocity_x_mps", "velocity_y_mps", "velocity_z_mps", "yaw_rate_radps"],
            "normalized_range": [-1.0, 1.0],
            "frame": "pilot_world",
        },
        "training_backend": "rivermark-kinematic-pilot-v1",
        "training_timesteps": args.timesteps,
        "seed": args.seed,
        "agent_count": args.agents,
        "episode_steps": args.episode_steps,
        "runtime_versions": _runtime_versions(),
        "isaac_control_transfer": {
            "eligible": transfer_contract != "none",
            "coordinate_contract": transfer_contract,
            "physical_training": False,
            "isaac_training": False,
            "claim_boundary": "development_state_only_control_wiring_smoke_only",
        },
        "formal_benchmark_admission": False,
        "source_revision": source.source_revision,
        "source_tree_sha256": source.source_tree_sha256,
        "source_worktree_dirty": source.source_worktree_dirty,
        "checkpoint_sha256": _sha256_file(checkpoint),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "metadata": str(metadata_path), "algorithm": args.algorithm}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
