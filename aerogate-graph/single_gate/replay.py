"""Replay helpers for the single-agent 2D gate experiment."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from multi_gate.planners.global_route_planner import GlobalRoutePlanner2D
from single_gate.configs.experiment_config import SINGLE_EXPERIMENT_CONFIG
from single_gate.env.single_gate_env import SingleGate2DEnv
from single_gate.graph_rl.graph_flashsac import GraphFlashSACAgent
from single_gate.training import validate_single_checkpoint_compatibility
from shared.configs.global_config import GLOBAL_CONFIG
from shared.runtime.artifacts import allocate_replay_artifacts, default_run_name, write_json


class HeuristicSingleReplayController:
    """A simple waypoint follower used for deterministic smoke replays."""

    def __init__(self, env: SingleGate2DEnv) -> None:
        self.env = env
        self._planner = GlobalRoutePlanner2D(
            obstacle_map=env.obstacle_map,
            env_config=env.env_config,
        )
        self._path: list[tuple[float, float]] = []
        self._path_index = 1

    def reset(self) -> None:
        state = self.env.current_state()
        inflation = self.env.env_config.drone_radius_m + 0.6
        plan = self._planner.plan(
            start_xy=state.position_xy,
            goal_xy=state.goal_xy,
            inflation_radius_m=inflation,
        )
        self._path = list(plan.waypoints_xy)
        self._path_index = 1 if len(self._path) > 1 else 0

    def act(self) -> np.ndarray:
        state = self.env.current_state()
        position = np.asarray(state.position_xy, dtype=np.float32)
        if not self._path:
            self.reset()
        while self._path_index < len(self._path) - 1:
            target = np.asarray(self._path[self._path_index], dtype=np.float32)
            if float(np.linalg.norm(target - position)) <= 1.2:
                self._path_index += 1
            else:
                break
        target = np.asarray(self._path[self._path_index], dtype=np.float32)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            desired_velocity = np.zeros((2,), dtype=np.float32)
        else:
            direction = delta / distance
            desired_speed = min(self.env.env_config.max_command_speed_mps, 1.2 + 0.7 * distance)
            desired_velocity = direction * desired_speed
        action = desired_velocity / max(self.env.env_config.max_command_speed_mps, 1e-6)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_single_replay(
    *,
    mode: str = "heuristic",
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    max_steps: int | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run one single-agent replay and save a replay report."""

    env = SingleGate2DEnv(
        env_config=replace(
            SINGLE_EXPERIMENT_CONFIG.environment,
            max_episode_steps=int(max_steps or SINGLE_EXPERIMENT_CONFIG.environment.max_episode_steps),
        )
    )
    observation, _ = env.reset(seed=seed)
    max_steps = int(max_steps or env.env_config.max_episode_steps)

    controller: HeuristicSingleReplayController | None = None
    agent: GraphFlashSACAgent | None = None
    if mode == "heuristic":
        controller = HeuristicSingleReplayController(env)
        controller.reset()
    elif mode == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when mode='checkpoint'")
        agent = GraphFlashSACAgent.from_defaults(obs_shapes=env.observation_shapes, device=device, seed=seed)
        validate_single_checkpoint_compatibility(
            checkpoint_path=checkpoint_path,
            env=env,
        )
        agent.load_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unsupported replay mode: {mode}")

    trajectory: list[dict[str, object]] = []
    episode_reward = 0.0
    done_reason = None
    steps = 0

    for step in range(max_steps):
        state = env.current_state()
        trajectory.append(
            {
                "step": step,
                "t_sec": float(getattr(env._state, "t_sec", step * env.env_config.dt_s)),
                "x_m": float(state.position_xy[0]),
                "y_m": float(state.position_xy[1]),
                "z_m": float(env.env_config.fixed_height_m),
                "vx_mps": float(state.velocity_xy[0]),
                "vy_mps": float(state.velocity_xy[1]),
                "yaw_rad": float(state.yaw_rad),
                "speed_mps": float(np.linalg.norm(np.asarray(state.velocity_xy, dtype=np.float32))),
                "goal_x_m": float(state.goal_xy[0]),
                "goal_y_m": float(state.goal_xy[1]),
            }
        )
        if controller is not None:
            action = controller.act()
        else:
            assert agent is not None
            action = agent.act(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        steps = step + 1
        if terminated or truncated:
            done_reason = str(info.get("done_reason") or "unknown")
            break

    success = done_reason == "goal_reached"
    report = {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "success": success,
        "done_reason": done_reason,
        "episode_reward": float(episode_reward),
        "final_state": asdict(env.current_state()),
        "trajectory_len": len(trajectory),
    }

    if output_dir is None:
        artifacts = allocate_replay_artifacts("single", run_name=default_run_name(f"single_replay_{mode}"))
        output_path = artifacts.output_dir
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "replay_report.json"
    trajectory_path = output_path / "trajectory.json"
    report["report_path"] = str(report_path)
    report["trajectory_path"] = str(trajectory_path)
    write_json(report_path, report)
    write_json(
        trajectory_path,
        {
            "format": "aerogate_graph_single_replay_v2",
            "fixed_height_m": float(env.env_config.fixed_height_m),
            "goal_radius_m": float(env.env_config.goal_radius_m),
            "world_x_bounds_m": list(env.env_config.world_x_bounds_m),
            "world_y_bounds_m": list(env.env_config.world_y_bounds_m),
            "drone_radius_m": float(env.env_config.drone_radius_m),
            "drone_asset_path": str(GLOBAL_CONFIG.drone_asset_file),
            "gate_layout_path": str(GLOBAL_CONFIG.gate_layout_file),
            "trajectory": trajectory,
        },
    )
    return report

