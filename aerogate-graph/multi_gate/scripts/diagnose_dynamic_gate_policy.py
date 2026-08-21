"""Quick policy diagnostics for the 8-drone dynamic-gate curriculum."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _load_runner(root: Path) -> Any:
    runner_path = root / "multi_gate" / "scripts" / "run_dynamic_gate_density_8d_curriculum.py"
    spec = importlib.util.spec_from_file_location("dynamic_gate_curriculum_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner from {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_gate_curriculum_runner"] = runner
    spec.loader.exec_module(runner)
    return runner


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    episodes = max(len(rows), 1)
    done_reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("done_reason") or "unknown")
        done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
    return {
        "episodes": len(rows),
        "success_rate": sum(1 for row in rows if row.get("done_reason") == "goal_reached") / episodes,
        "dynamic_gate_collision_rate": sum(1 for row in rows if row.get("dynamic_gate_collision") is True) / episodes,
        "agent_collision_rate": done_reason_counts.get("agent_collision", 0) / episodes,
        "timeout_rate": done_reason_counts.get("timeout", 0) / episodes,
        "done_reason_counts": done_reason_counts,
        "mean_goal_distance_m": (
            float(np.mean([float(row["goal_distance_m"]) for row in rows if row.get("goal_distance_m") is not None]))
            if rows
            else None
        ),
        "mean_actual_gate_motion_range_m": (
            float(np.mean([float(row["actual_gate_motion_range_m"]) for row in rows]))
            if rows
            else None
        ),
        "episode_summaries": rows,
    }


def _nearest_dynamic_post_report(env: object) -> dict[str, object] | None:
    if not bool(getattr(env, "_dynamic_gate_enabled", False)):
        return None
    if not hasattr(env, "active_positions_xy") or not hasattr(env, "_dynamic_gate_posts_xy"):
        return None
    positions = np.asarray(env.active_positions_xy(), dtype=np.float32)
    posts = np.asarray(env._dynamic_gate_posts_xy(next_frame=False), dtype=np.float32)
    if positions.size == 0 or posts.size == 0:
        return None
    gate_cfg = getattr(env, "_dynamic_gate_config", None)
    post_radius_m = float(getattr(gate_cfg, "gate_post_radius_m", 0.32))
    drone_radius_m = float(getattr(getattr(env, "env_config", None), "drone_radius_m", 0.35))
    best: dict[str, object] | None = None
    for agent_idx, position in enumerate(positions):
        for post_idx, post in enumerate(posts):
            distance_m = float(np.linalg.norm(position - post))
            clearance_m = float(distance_m - post_radius_m - drone_radius_m)
            if best is None or clearance_m < float(best["clearance_m"]):
                best = {
                    "agent_index": int(agent_idx),
                    "post_index": int(post_idx),
                    "distance_m": distance_m,
                    "clearance_m": clearance_m,
                    "agent_xy": [float(position[0]), float(position[1])],
                    "post_xy": [float(post[0]), float(post[1])],
                }
    return best


def _nearest_pair_report(env: object) -> dict[str, object] | None:
    if not hasattr(env, "active_positions_xy"):
        return None
    positions = np.asarray(env.active_positions_xy(), dtype=np.float32)
    if positions.shape[0] <= 1:
        return None
    best: dict[str, object] | None = None
    for idx in range(int(positions.shape[0])):
        for jdx in range(idx + 1, int(positions.shape[0])):
            distance_m = float(np.linalg.norm(positions[idx] - positions[jdx]))
            if best is None or distance_m < float(best["distance_m"]):
                best = {
                    "agent_i": int(idx),
                    "agent_j": int(jdx),
                    "distance_m": distance_m,
                    "agent_i_xy": [float(positions[idx, 0]), float(positions[idx, 1])],
                    "agent_j_xy": [float(positions[jdx, 0]), float(positions[jdx, 1])],
                }
    return best


def main() -> None:
    root = _bootstrap()
    runner = _load_runner(root)

    from multi_gate.configs import get_multi_experiment_config
    from multi_gate.env.multi_gate_env import MultiGate2DEnv
    from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
    from multi_gate.replay import HeuristicFormationReplayController
    from multi_gate.training import validate_multi_checkpoint_compatibility

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("heuristic", "actor", "checkpoint"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage-index", type=int, default=3)
    parser.add_argument("--stage-name", type=str, default=None)
    parser.add_argument("--gate-count", type=int, default=None)
    parser.add_argument("--speed-mps", type=float, default=None)
    parser.add_argument("--amplitude-m", type=float, default=None)
    parser.add_argument("--drone-speed-mps", type=float, default=None)
    parser.add_argument("--drone-accel-mps2", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20265836)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    stages = runner._stages()
    stage = stages[int(args.stage_index)]
    if args.stage_name is not None:
        requested_stage_name = str(args.stage_name)
        for candidate_stage in stages:
            if str(candidate_stage.name) == requested_stage_name:
                stage = candidate_stage
                break
    stage = replace(
        stage,
        name=str(args.stage_name or stage.name),
        gate_count=int(args.gate_count) if args.gate_count is not None else int(stage.gate_count),
        speed_mps=float(args.speed_mps) if args.speed_mps is not None else float(stage.speed_mps),
        amplitude_m=float(args.amplitude_m) if args.amplitude_m is not None else float(stage.amplitude_m),
        drone_speed_mps=(
            float(args.drone_speed_mps) if args.drone_speed_mps is not None else float(stage.drone_speed_mps)
        ),
        drone_accel_mps2=(
            float(args.drone_accel_mps2) if args.drone_accel_mps2 is not None else float(stage.drone_accel_mps2)
        ),
    )
    config = runner._stage_config(base_config, stage)
    env = MultiGate2DEnv(
        multi_config=config,
        env_config=config.environment,
        observation_config=config.observation,
        formation_config=config.formation,
        planner_config=config.planner,
    )

    agent: GraphMASACAgent | None = None
    if args.mode in {"actor", "checkpoint"}:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for actor/checkpoint modes")
        agent = GraphMASACAgent.from_defaults(
            obs_shapes=env.observation_shapes,
            device=args.device,
            seed=int(args.seed),
            obs_config=config.observation,
            masac_config=config.algorithm,
            max_agents_soft=config.max_agents_soft,
            build_replay_buffer=False,
        )
        if args.mode == "actor":
            agent.load_actor_checkpoint(args.checkpoint)
        else:
            validate_multi_checkpoint_compatibility(
                checkpoint_path=args.checkpoint,
                env=env,
                experiment_config=config,
            )
            agent.load_checkpoint(args.checkpoint)

    rows: list[dict[str, object]] = []
    for episode_idx in range(max(int(args.episodes), 1)):
        observation, _ = env.reset(seed=int(args.seed) + episode_idx, num_agents=8)
        controller = HeuristicFormationReplayController(env)
        for step_idx in range(int(config.environment.max_episode_steps)):
            if args.mode == "heuristic":
                action = controller.act()
            else:
                assert agent is not None
                action = agent.act(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                snapshot = info.get("snapshot")
                active_positions = np.asarray(env.active_positions_xy(), dtype=np.float32)
                desired_slots = np.asarray(env.desired_slots_xy(), dtype=np.float32)
                dynamic_gate_posts = (
                    np.asarray(env._dynamic_gate_posts_xy(next_frame=False), dtype=np.float32)
                    if hasattr(env, "_dynamic_gate_posts_xy")
                    else np.zeros((0, 2), dtype=np.float32)
                )
                rows.append(
                    {
                        "episode_index": episode_idx,
                        "steps": step_idx + 1,
                        "done_reason": str(info.get("done_reason") or "unknown"),
                        "goal_distance_m": (
                            None if snapshot is None else float(getattr(snapshot, "goal_distance_m", 0.0))
                        ),
                        "virtual_center_xy": (
                            None if snapshot is None else list(getattr(snapshot, "virtual_center_xy", (0.0, 0.0)))
                        ),
                        "mean_slot_error_m": (
                            None if snapshot is None else float(getattr(snapshot, "mean_slot_error_m", 0.0))
                        ),
                        "min_clearance_m": (
                            None if info.get("min_clearance_m") is None else float(info.get("min_clearance_m"))
                        ),
                        "min_pair_distance_m": (
                            None if info.get("min_pair_distance_m") is None else float(info.get("min_pair_distance_m"))
                        ),
                        "dynamic_gate_collision": bool(info.get("dynamic_gate_collision", False)),
                        "dynamic_gate_count": int(info.get("dynamic_gate_count") or 0),
                        "height_contract_passed": bool(info.get("height_contract_passed", True)),
                        "height_escape_failure": bool(info.get("height_escape_failure", False)),
                        "side_bypass_failure": bool(info.get("side_bypass_failure", False)),
                        "corridor_miss_failure": bool(info.get("corridor_miss_failure", False)),
                        "corridor_completed": bool(info.get("corridor_completed", False)),
                        "corridor_through_success": bool(
                            bool(info.get("height_contract_passed", True))
                            and not bool(info.get("height_escape_failure", False))
                            and bool(info.get("corridor_completed", False))
                            and not bool(info.get("side_bypass_failure", False))
                            and not bool(info.get("corridor_miss_failure", False))
                        ),
                        "moving_gate_speed_mps": float(info.get("moving_gate_speed_mps") or 0.0),
                        "actual_gate_motion_range_m": float(info.get("actual_gate_motion_range_m") or 0.0),
                        "active_positions_xy": active_positions.astype(float).tolist(),
                        "active_velocities_xy": (
                            np.asarray(info.get("agent_velocities_xy"), dtype=np.float32)
                            [: active_positions.shape[0]]
                            .astype(float)
                            .tolist()
                            if info.get("agent_velocities_xy") is not None
                            else None
                        ),
                        "desired_slots_xy": desired_slots[: active_positions.shape[0]].astype(float).tolist(),
                        "actor_observation_desired_slots_xy": (
                            np.asarray(info.get("actor_observation_desired_slots"), dtype=np.float32)
                            [: active_positions.shape[0]]
                            .astype(float)
                            .tolist()
                            if info.get("actor_observation_desired_slots") is not None
                            else None
                        ),
                        "actor_observation_dynamic_gate_task_ratio": (
                            None
                            if info.get("actor_observation_dynamic_gate_task_ratio") is None
                            else float(info.get("actor_observation_dynamic_gate_task_ratio"))
                        ),
                        "dynamic_gate_posts_xy": dynamic_gate_posts.astype(float).tolist(),
                        "action_safety_shield": dict(info.get("action_safety_shield") or {}),
                        "nearest_dynamic_post": _nearest_dynamic_post_report(env),
                        "nearest_pair": _nearest_pair_report(env),
                    }
                )
                break

    report = {
        "mode": args.mode,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "stage": stage.name,
        "gate_count": int(stage.gate_count),
        "speed_mps": float(stage.speed_mps),
        "amplitude_m": float(stage.amplitude_m),
        "drone_speed_mps": float(stage.drone_speed_mps),
        "drone_accel_mps2": float(stage.drone_accel_mps2),
        **_summarize(rows),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

