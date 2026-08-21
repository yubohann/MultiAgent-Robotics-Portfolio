"""Train the gate-density imitation bridge checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    project_root = root.parents[1]
    for path in (root, project_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _stack_observations(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.stack([item[name] for item in items], axis=0).astype(np.float32, copy=False)
        for name in items[0]
    }


def _sample_minibatch(
    *,
    observations: list[dict[str, np.ndarray]],
    actions: np.ndarray,
    indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    batch_obs = _stack_observations([observations[int(index)] for index in indices])
    batch_actions = actions[indices].astype(np.float32, copy=False)
    return batch_obs, batch_actions


def main() -> None:
    _bootstrap_imports()

    from gate_density_single.scripts.run_gate_density_eval import (
        ALLOWED_GATE_COUNTS,
        ALLOWED_GATE_LAYOUT_VERSIONS,
        ALLOWED_SEEDS,
        DRONE_RADIUS_M,
        GATE_HALF_WIDTH_M,
        GATE_POST_RADIUS_M,
        GOAL_RADIUS_M,
        GOAL_XYZ,
        MAX_EPISODE_STEPS,
        SAFETY_MARGIN_M,
        START_XYZ,
        WORLD_X_BOUNDS_M,
        WORLD_Y_BOUNDS_M,
        GateDensityController,
        _build_gate_obstacle_map,
        _density_adaptive_controller_profile,
        _generate_gate_layout,
        _layout_profile,
        _moving_gate_centers,
        _moving_gate_swept_clearance_m,
        _resolve_moving_gate_speed_hz,
    )
    from single_gate.configs.experiment_config import SINGLE_EXPERIMENT_CONFIG
    from single_gate.env.single_gate_env import SingleGate2DEnv
    from single_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphSACAgent
    from single_gate.training import validate_single_checkpoint_compatibility
    from shared.core.dynamic_gate_density_2d import MAX_DRONE_COMMAND_ACCEL_MPS2, MAX_DRONE_COMMAND_SPEED_MPS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="gate_density_imitation_bridge.pt")
    parser.add_argument("--gate-counts", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--gate-layout-version",
        type=str,
        default="irregular_centerline_v2",
        choices=ALLOWED_GATE_LAYOUT_VERSIONS,
    )
    parser.add_argument("--moving-gates", action="store_true")
    parser.add_argument("--moving-gate-amplitude-m", type=float, default=0.0)
    parser.add_argument("--moving-gate-speed-hz", type=float, default=0.0)
    parser.add_argument("--moving-gate-speed-mps", type=float, default=0.0)
    parser.add_argument("--drone-speed-mps", type=float, default=SINGLE_EXPERIMENT_CONFIG.environment.max_command_speed_mps)
    parser.add_argument("--drone-accel-mps2", type=float, default=SINGLE_EXPERIMENT_CONFIG.environment.max_accel_mps2)
    parser.add_argument("--moving-gate-episode-phase-offset-s", type=float, default=0.0)
    parser.add_argument("--moving-gate-episode-phase-stride-s", type=float, default=0.0)
    parser.add_argument("--dynamic-replan-interval-steps", type=int, default=0)
    parser.add_argument("--dynamic-replan-clearance-threshold-m", type=float, default=0.0)
    parser.add_argument("--dynamic-gate-speed-cap-base-mps", type=float, default=0.45)
    parser.add_argument("--dynamic-gate-speed-cap-gain", type=float, default=0.60)
    parser.add_argument("--dynamic-shield-rollout-steps", type=int, default=6)
    parser.add_argument("--dynamic-planner-inflation-extra-m", type=float, default=0.0)
    parser.add_argument("--dynamic-final-goal-bias-start-x-m", type=float, default=0.0)
    parser.add_argument("--dynamic-final-goal-bias-strength", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-controller-profile",
        type=str,
        default="none",
        choices=("none", "density_adaptive_v1"),
        help="Use one density-adaptive dynamic-gate expert/controller instead of hand-tuned per-gate knobs.",
    )
    parser.add_argument(
        "--stagewise-controller-profile",
        type=str,
        default="none",
        choices=("none", "gate30_breakpoint_v1"),
        help="Use gate-count-specific expert controller settings for rehearsal curricula.",
    )
    parser.add_argument(
        "--expert-enable-agent-policy",
        action="store_true",
        help="Collect BC samples from the base actor + planner/shield controller instead of pure planner-only expert.",
    )
    parser.add_argument("--episodes-per-layout", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--train-updates", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--include-failed-episodes",
        action="store_true",
        help="Opt in to the legacy behavior that trains on collision/timeout episodes.",
    )
    args = parser.parse_args()

    invalid_counts = [count for count in args.gate_counts if int(count) not in ALLOWED_GATE_COUNTS]
    invalid_seeds = [seed for seed in args.seeds if int(seed) not in ALLOWED_SEEDS]
    if invalid_counts:
        raise SystemExit(f"Unsupported --gate-counts: {invalid_counts}; allowed={ALLOWED_GATE_COUNTS}")
    if invalid_seeds:
        raise SystemExit(f"Unsupported --seeds: {invalid_seeds}; allowed={ALLOWED_SEEDS}")
    if int(args.dynamic_replan_interval_steps) < 0:
        raise SystemExit("--dynamic-replan-interval-steps must be >= 0")
    if float(args.drone_speed_mps) <= 0.0:
        raise SystemExit("--drone-speed-mps must be positive")
    if float(args.drone_speed_mps) > MAX_DRONE_COMMAND_SPEED_MPS:
        raise SystemExit(f"--drone-speed-mps must be <= {MAX_DRONE_COMMAND_SPEED_MPS}")
    if float(args.drone_accel_mps2) <= 0.0:
        raise SystemExit("--drone-accel-mps2 must be positive")
    if float(args.drone_accel_mps2) > MAX_DRONE_COMMAND_ACCEL_MPS2:
        raise SystemExit(f"--drone-accel-mps2 must be <= {MAX_DRONE_COMMAND_ACCEL_MPS2}")
    if float(args.dynamic_replan_clearance_threshold_m) < 0.0:
        raise SystemExit("--dynamic-replan-clearance-threshold-m must be >= 0")
    if float(args.dynamic_gate_speed_cap_base_mps) <= 0.0:
        raise SystemExit("--dynamic-gate-speed-cap-base-mps must be positive")
    if float(args.dynamic_gate_speed_cap_gain) < 0.0:
        raise SystemExit("--dynamic-gate-speed-cap-gain must be >= 0")
    if int(args.dynamic_shield_rollout_steps) <= 0:
        raise SystemExit("--dynamic-shield-rollout-steps must be positive")
    if float(args.dynamic_planner_inflation_extra_m) < 0.0:
        raise SystemExit("--dynamic-planner-inflation-extra-m must be >= 0")
    if float(args.dynamic_final_goal_bias_start_x_m) < 0.0:
        raise SystemExit("--dynamic-final-goal-bias-start-x-m must be >= 0")
    if not (0.0 <= float(args.dynamic_final_goal_bias_strength) <= 1.0):
        raise SystemExit("--dynamic-final-goal-bias-strength must be in [0, 1]")
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint is missing: {args.base_checkpoint}")
    moving_gate_speed_hz = _resolve_moving_gate_speed_hz(
        amplitude_m=float(args.moving_gate_amplitude_m),
        speed_hz=float(args.moving_gate_speed_hz),
        speed_mps=float(args.moving_gate_speed_mps),
    )
    layout_profile = _layout_profile(str(args.gate_layout_version))

    def _controller_profile_for_gate(gate_count: int) -> dict[str, float | int]:
        if str(args.dynamic_controller_profile) == "density_adaptive_v1":
            return _density_adaptive_controller_profile(int(gate_count))
        profile = {
            "dynamic_replan_interval_steps": int(args.dynamic_replan_interval_steps),
            "dynamic_replan_clearance_threshold_m": float(args.dynamic_replan_clearance_threshold_m),
            "dynamic_gate_speed_cap_base_mps": float(args.dynamic_gate_speed_cap_base_mps),
            "dynamic_gate_speed_cap_gain": float(args.dynamic_gate_speed_cap_gain),
            "dynamic_shield_rollout_steps": int(args.dynamic_shield_rollout_steps),
            "dynamic_planner_inflation_extra_m": float(args.dynamic_planner_inflation_extra_m),
            "dynamic_final_goal_bias_start_x_m": float(args.dynamic_final_goal_bias_start_x_m),
            "dynamic_final_goal_bias_strength": float(args.dynamic_final_goal_bias_strength),
        }
        if str(args.stagewise_controller_profile) == "gate30_breakpoint_v1" and int(gate_count) < 30:
            profile.update(
                {
                    "dynamic_replan_interval_steps": 12,
                    "dynamic_replan_clearance_threshold_m": 0.45,
                    "dynamic_gate_speed_cap_base_mps": 0.70,
                    "dynamic_gate_speed_cap_gain": 0.75,
                    "dynamic_shield_rollout_steps": 6,
                    "dynamic_planner_inflation_extra_m": 0.40,
                    "dynamic_final_goal_bias_start_x_m": 0.0,
                    "dynamic_final_goal_bias_strength": 0.0,
                }
            )
        return profile

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, np.ndarray]] = []
    expert_actions: list[np.ndarray] = []
    layout_summaries: list[dict[str, Any]] = []
    accepted_episode_count = 0
    rejected_episode_count = 0
    accepted_step_count = 0
    rejected_step_count = 0
    global_done_reasons: dict[str, int] = {}
    global_accepted_done_reasons: dict[str, int] = {}
    global_rejected_done_reasons: dict[str, int] = {}
    expert_agent: GraphSACAgent | None = None

    for gate_count in [int(item) for item in args.gate_counts]:
        controller_profile = _controller_profile_for_gate(gate_count)
        for layout_seed in [int(item) for item in args.seeds]:
            gate_centers_xy, gate_yaws = _generate_gate_layout(
                gate_count=gate_count,
                seed=layout_seed,
                random_yaw=True,
                layout_version=str(args.gate_layout_version),
            )
            obstacle_map = _build_gate_obstacle_map(gate_centers_xy, gate_yaws)
            env_config = replace(
                SINGLE_EXPERIMENT_CONFIG.environment,
                fixed_height_m=layout_profile.start_xyz[2],
                drone_radius_m=DRONE_RADIUS_M,
                goal_radius_m=GOAL_RADIUS_M,
                max_episode_steps=int(args.max_steps),
                max_command_speed_mps=float(args.drone_speed_mps),
                max_accel_mps2=float(args.drone_accel_mps2),
                start_x_m=layout_profile.start_xyz[0],
                goal_x_m=layout_profile.goal_xyz[0],
                start_y_range_m=(layout_profile.start_xyz[1], layout_profile.start_xyz[1]),
                goal_y_range_m=(layout_profile.goal_xyz[1], layout_profile.goal_xyz[1]),
                world_x_bounds_m=layout_profile.world_x_bounds_m,
                world_y_bounds_m=layout_profile.world_y_bounds_m,
            )
            env = SingleGate2DEnv(env_config=env_config, obstacle_map=obstacle_map)
            validate_single_checkpoint_compatibility(checkpoint_path=args.base_checkpoint, env=env)
            if bool(args.expert_enable_agent_policy) and expert_agent is None:
                expert_agent = GraphSACAgent.from_defaults(
                    obs_shapes=env.observation_shapes,
                    device=args.device,
                    seed=int(args.seed),
                )
                expert_agent.load_checkpoint(args.base_checkpoint)
                expert_agent.actor.eval()
            layout_steps = 0
            layout_done_reasons: dict[str, int] = {}
            layout_accepted_episodes = 0
            layout_rejected_episodes = 0
            layout_accepted_steps = 0
            layout_rejected_steps = 0

            for episode_index in range(int(args.episodes_per_layout)):
                observation, _ = env.reset(seed=layout_seed * 1000 + episode_index)
                episode_phase_offset_s = float(args.moving_gate_episode_phase_offset_s) + (
                    float(episode_index) * float(args.moving_gate_episode_phase_stride_s)
                )
                controller_args = argparse.Namespace(
                    enable_route_guidance=False,
                    guidance_shadow_mode=False,
                    guidance_visible=False,
                    guidance_query_interval_steps=30,
                    moving_gates=bool(args.moving_gates),
                    dynamic_controller_profile=str(args.dynamic_controller_profile),
                    dynamic_replan_interval_steps=int(controller_profile["dynamic_replan_interval_steps"]),
                    dynamic_replan_clearance_threshold_m=float(
                        controller_profile["dynamic_replan_clearance_threshold_m"]
                    ),
                    dynamic_gate_speed_cap_base_mps=float(controller_profile["dynamic_gate_speed_cap_base_mps"]),
                    dynamic_gate_speed_cap_gain=float(controller_profile["dynamic_gate_speed_cap_gain"]),
                    dynamic_shield_rollout_steps=int(controller_profile["dynamic_shield_rollout_steps"]),
                    dynamic_planner_inflation_extra_m=float(controller_profile["dynamic_planner_inflation_extra_m"]),
                    dynamic_final_goal_bias_start_x_m=float(
                        controller_profile["dynamic_final_goal_bias_start_x_m"]
                    ),
                    dynamic_final_goal_bias_strength=float(controller_profile["dynamic_final_goal_bias_strength"]),
                )
                controller = GateDensityController(
                    env=env,
                    agent=expert_agent if bool(args.expert_enable_agent_policy) else None,
                    args=controller_args,
                    gate_count=gate_count,
                    enable_agent_policy=bool(args.expert_enable_agent_policy),
                    enable_global_planner=True,
                    enable_path_planner=True,
                    guidance_client=None,
                )
                controller.set_dynamic_gate_context(
                    base_centers_xy=gate_centers_xy,
                    gate_yaws=gate_yaws,
                    seed=layout_seed,
                    layout_version=str(args.gate_layout_version),
                    amplitude_m=float(args.moving_gate_amplitude_m),
                    speed_hz=float(moving_gate_speed_hz),
                    phase_offset_s=float(episode_phase_offset_s),
                )
                controller.reset()
                episode_observations: list[dict[str, np.ndarray]] = []
                episode_actions: list[np.ndarray] = []
                episode_done_reason = "max_steps_exhausted"
                episode_had_collision = False
                for step in range(int(args.max_steps)):
                    if bool(args.moving_gates):
                        moved_centers = _moving_gate_centers(
                            base_centers_xy=gate_centers_xy,
                            gate_yaws=gate_yaws,
                            seed=layout_seed,
                            t_sec=float(getattr(env._state, "t_sec", step * env.env_config.dt_s))
                            + float(episode_phase_offset_s),
                            enabled=True,
                            amplitude_m=float(args.moving_gate_amplitude_m),
                            speed_hz=float(moving_gate_speed_hz),
                            layout_version=str(args.gate_layout_version),
                        )
                        env.obstacle_map = _build_gate_obstacle_map(moved_centers, gate_yaws)
                        observation = env._build_observation()
                    action = controller.act(observation, step=step)
                    episode_observations.append({name: value.copy() for name, value in observation.items()})
                    episode_actions.append(np.asarray(action, dtype=np.float32).copy())
                    previous_position_xy = env.current_state().position_xy
                    observation, _, terminated, truncated, info = env.step(action)
                    layout_steps += 1
                    if bool(args.moving_gates):
                        next_position_xy = env.current_state().position_xy
                        next_moved_centers = _moving_gate_centers(
                            base_centers_xy=gate_centers_xy,
                            gate_yaws=gate_yaws,
                            seed=layout_seed,
                            t_sec=float(getattr(env._state, "t_sec", (step + 1) * env.env_config.dt_s))
                            + float(episode_phase_offset_s),
                            enabled=True,
                            amplitude_m=float(args.moving_gate_amplitude_m),
                            speed_hz=float(moving_gate_speed_hz),
                            layout_version=str(args.gate_layout_version),
                        )
                        swept_clearance_m = _moving_gate_swept_clearance_m(
                            drone_start_xy=previous_position_xy,
                            drone_end_xy=next_position_xy,
                            gate_centers_start_xy=moved_centers,
                            gate_centers_end_xy=next_moved_centers,
                            gate_yaws=gate_yaws,
                            drone_radius_m=DRONE_RADIUS_M,
                        )
                        endpoint_clearance_next_m = _build_gate_obstacle_map(
                            next_moved_centers,
                            gate_yaws,
                        ).min_signed_distance(next_position_xy, drone_radius_m=DRONE_RADIUS_M)
                        if min(float(swept_clearance_m), float(endpoint_clearance_next_m)) <= 0.0:
                            episode_done_reason = "collision"
                            episode_had_collision = True
                            break
                    if terminated or truncated:
                        episode_done_reason = str(info.get("done_reason") or "unknown")
                        episode_had_collision = episode_had_collision or episode_done_reason == "collision"
                        break

                layout_done_reasons[episode_done_reason] = layout_done_reasons.get(episode_done_reason, 0) + 1
                global_done_reasons[episode_done_reason] = global_done_reasons.get(episode_done_reason, 0) + 1
                episode_is_accepted = bool(episode_done_reason == "goal_reached" and not episode_had_collision)
                if episode_is_accepted or bool(args.include_failed_episodes):
                    observations.extend(episode_observations)
                    expert_actions.extend(episode_actions)
                    accepted_episode_count += 1
                    layout_accepted_episodes += 1
                    accepted_step_count += len(episode_actions)
                    layout_accepted_steps += len(episode_actions)
                    global_accepted_done_reasons[episode_done_reason] = (
                        global_accepted_done_reasons.get(episode_done_reason, 0) + 1
                    )
                else:
                    rejected_episode_count += 1
                    layout_rejected_episodes += 1
                    rejected_step_count += len(episode_actions)
                    layout_rejected_steps += len(episode_actions)
                    global_rejected_done_reasons[episode_done_reason] = (
                        global_rejected_done_reasons.get(episode_done_reason, 0) + 1
                    )

            layout_summaries.append(
                {
                    "gate_count": gate_count,
                    "seed": layout_seed,
                    "random_yaw": True,
                    "gate_layout_version": str(args.gate_layout_version),
                    "gate_yaw_policy": (
                        "random_uniform_minus5_to_plus5_deg_formation_facing"
                        if str(args.gate_layout_version) == "irregular_centerline_v7_large_arena_dynamic"
                        else "layout_default"
                    ),
                    "moving_gates_enabled": bool(args.moving_gates),
                    "moving_gate_amplitude_m": float(args.moving_gate_amplitude_m),
                    "moving_gate_speed_hz": float(moving_gate_speed_hz),
                    "moving_gate_speed_mps": float(args.moving_gate_speed_mps),
                    "drone_speed_mps": float(args.drone_speed_mps),
                    "drone_accel_mps2": float(args.drone_accel_mps2),
                    "moving_gate_episode_phase_offset_s": float(args.moving_gate_episode_phase_offset_s),
                    "moving_gate_episode_phase_stride_s": float(args.moving_gate_episode_phase_stride_s),
                    "dynamic_controller_profile": str(args.dynamic_controller_profile),
                    "stagewise_controller_profile": str(args.stagewise_controller_profile),
                    "dynamic_replan_interval_steps": int(controller_profile["dynamic_replan_interval_steps"]),
                    "dynamic_replan_clearance_threshold_m": float(
                        controller_profile["dynamic_replan_clearance_threshold_m"]
                    ),
                    "dynamic_gate_speed_cap_base_mps": float(controller_profile["dynamic_gate_speed_cap_base_mps"]),
                    "dynamic_gate_speed_cap_gain": float(controller_profile["dynamic_gate_speed_cap_gain"]),
                    "dynamic_shield_rollout_steps": int(controller_profile["dynamic_shield_rollout_steps"]),
                    "dynamic_planner_inflation_extra_m": float(controller_profile["dynamic_planner_inflation_extra_m"]),
                    "dynamic_final_goal_bias_start_x_m": float(
                        controller_profile["dynamic_final_goal_bias_start_x_m"]
                    ),
                    "dynamic_final_goal_bias_strength": float(
                        controller_profile["dynamic_final_goal_bias_strength"]
                    ),
                    "expert_enable_agent_policy": bool(args.expert_enable_agent_policy),
                    "episodes": int(args.episodes_per_layout),
                    "collected_steps": int(layout_steps),
                    "accepted_episodes": int(layout_accepted_episodes),
                    "rejected_episodes": int(layout_rejected_episodes),
                    "accepted_steps": int(layout_accepted_steps),
                    "rejected_steps": int(layout_rejected_steps),
                    "done_reason_counts": layout_done_reasons,
                    "gate_centers_xy": [list(item) for item in gate_centers_xy],
                    "gate_yaws_rad": list(gate_yaws),
                    "gate_yaws_deg": [float(np.degrees(value)) for value in gate_yaws],
                }
            )
            _append_jsonl(output_dir / "layout_progress.jsonl", layout_summaries[-1])
            _write_json(
                output_dir / "layout_progress_latest.json",
                {
                    "gate_count": int(gate_count),
                    "seed": int(layout_seed),
                    "layout_index_completed": int(len(layout_summaries)),
                    "total_layouts": int(len(args.gate_counts) * len(args.seeds)),
                    "accepted_episode_count": int(accepted_episode_count),
                    "rejected_episode_count": int(rejected_episode_count),
                    "accepted_step_count": int(accepted_step_count),
                    "rejected_step_count": int(rejected_step_count),
                    "done_reason_counts": global_done_reasons,
                    "accepted_done_reason_counts": global_accepted_done_reasons,
                    "rejected_done_reason_counts": global_rejected_done_reasons,
                    "latest_layout": layout_summaries[-1],
                },
            )

    if not observations:
        diagnostic_path = output_dir / "expert_filter_failure.json"
        _write_json(
            diagnostic_path,
            {
                "error": "No accepted imitation samples were collected.",
                "filter_policy": "only goal_reached episodes with no collision are used for BC",
                "include_failed_episodes": bool(args.include_failed_episodes),
                "accepted_episode_count": int(accepted_episode_count),
                "rejected_episode_count": int(rejected_episode_count),
                "accepted_step_count": int(accepted_step_count),
                "rejected_step_count": int(rejected_step_count),
                "done_reason_counts": global_done_reasons,
                "accepted_done_reason_counts": global_accepted_done_reasons,
                "rejected_done_reason_counts": global_rejected_done_reasons,
                "layout_summaries": layout_summaries,
            },
        )
        raise RuntimeError(
            "No accepted imitation samples were collected. The expert data is all failed under the "
            f"current contract; diagnostics written to {diagnostic_path}"
        )

    expert_action_array = np.stack(expert_actions, axis=0).astype(np.float32, copy=False)
    probe_gate_centers_xy, probe_gate_yaws = _generate_gate_layout(
        gate_count=0,
        seed=0,
        random_yaw=True,
        layout_version=str(args.gate_layout_version),
    )
    probe_env_config = replace(
        SINGLE_EXPERIMENT_CONFIG.environment,
        fixed_height_m=layout_profile.start_xyz[2],
        drone_radius_m=DRONE_RADIUS_M,
        goal_radius_m=GOAL_RADIUS_M,
        max_episode_steps=int(args.max_steps),
        max_command_speed_mps=float(args.drone_speed_mps),
        max_accel_mps2=float(args.drone_accel_mps2),
        start_x_m=layout_profile.start_xyz[0],
        goal_x_m=layout_profile.goal_xyz[0],
        start_y_range_m=(layout_profile.start_xyz[1], layout_profile.start_xyz[1]),
        goal_y_range_m=(layout_profile.goal_xyz[1], layout_profile.goal_xyz[1]),
        world_x_bounds_m=layout_profile.world_x_bounds_m,
        world_y_bounds_m=layout_profile.world_y_bounds_m,
    )
    probe_env = SingleGate2DEnv(
        env_config=probe_env_config,
        obstacle_map=_build_gate_obstacle_map(probe_gate_centers_xy, probe_gate_yaws),
    )
    agent = GraphSACAgent.from_defaults(obs_shapes=probe_env.observation_shapes, device=args.device, seed=int(args.seed))
    base_metadata = agent.load_checkpoint(args.base_checkpoint)
    agent.actor.train()
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=float(args.actor_lr))

    sample_count = len(observations)
    losses: list[float] = []
    for update_idx in range(int(args.train_updates)):
        batch_size = min(int(args.batch_size), sample_count)
        batch_indices = np.random.randint(0, sample_count, size=(batch_size,), dtype=np.int64)
        batch_obs_np, batch_action_np = _sample_minibatch(
            observations=observations,
            actions=expert_action_array,
            indices=batch_indices,
        )
        batch_obs = {
            name: torch.as_tensor(value, dtype=torch.float32, device=agent.device)
            for name, value in batch_obs_np.items()
        }
        target_action = torch.as_tensor(batch_action_np, dtype=torch.float32, device=agent.device)
        mean, _ = agent.actor(batch_obs)
        pred_action = torch.tanh(mean)
        loss = F.mse_loss(pred_action, target_action)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.item()))

    checkpoint_path = output_dir / "checkpoints" / str(args.checkpoint_name)
    base_signature = dict(base_metadata.get("training_signature") or {})
    if not base_signature:
        base_signature = {
            "experiment_id": SINGLE_EXPERIMENT_CONFIG.experiment_id,
            "control_mode": SINGLE_EXPERIMENT_CONFIG.control_mode,
            "planner_mode": SINGLE_EXPERIMENT_CONFIG.planner_mode,
            "observation_shapes": {name: list(shape) for name, shape in probe_env.observation_shapes.items()},
            "action_dim": int(SINGLE_EXPERIMENT_CONFIG.algorithm.action_dim),
            "log_std_min": float(SINGLE_EXPERIMENT_CONFIG.algorithm.log_std_min),
            "log_std_max": float(SINGLE_EXPERIMENT_CONFIG.algorithm.log_std_max),
        }
    metadata = {
        "experiment_id": SINGLE_EXPERIMENT_CONFIG.experiment_id,
        "seed": int(args.seed),
        "checkpoint_step": int(base_metadata.get("checkpoint_step", 0) or 0),
        "checkpoint_kind": "gate_density_imitation_bridge",
        "training_signature": base_signature,
        "resume_context": {"base_metadata": base_metadata},
        "branch": "gate_density_single",
        "stage": "gate_density_imitation_bridge_C4_C8",
        "base_checkpoint": str(args.base_checkpoint),
        "sample_count": int(sample_count),
        "gate_counts": [int(item) for item in args.gate_counts],
        "seeds": [int(item) for item in args.seeds],
        "episodes_per_layout": int(args.episodes_per_layout),
        "train_updates": int(args.train_updates),
        "batch_size": int(args.batch_size),
        "actor_lr": float(args.actor_lr),
        "expert_filter_policy": (
            "legacy_include_failed_episodes"
            if bool(args.include_failed_episodes)
            else "only_goal_reached_no_collision_episodes"
        ),
        "expert_controller": (
            "base_actor_plus_planner_shield"
            if bool(args.expert_enable_agent_policy)
            else "planner_shield_only"
        ),
        "expert_enable_agent_policy": bool(args.expert_enable_agent_policy),
        "include_failed_episodes": bool(args.include_failed_episodes),
        "guidance_training_policy": "External guidance is not called in the BC inner loop; use shadow diagnostics only.",
        "stage_objective": "C4/C5 planner imitation, C6 lower-speed tracking, C7/C8 raw A* expert bridge.",
        "modification_reason": "The visible-guidance variant failed from 6 gates upward due to collision/timeout.",
        "expected_behavior": "Policy actions should align with the gate-density planner before visible guidance blending.",
    }
    saved_path = agent.save_checkpoint(checkpoint_path, metadata=metadata)
    best_path = output_dir / "checkpoints" / "best_agent.pt"
    best_path.write_bytes(saved_path.read_bytes())

    summary = {
        **metadata,
        "checkpoint": str(saved_path),
        "best_checkpoint": str(best_path),
        "layout_summaries": layout_summaries,
        "accepted_episode_count": int(accepted_episode_count),
        "rejected_episode_count": int(rejected_episode_count),
        "accepted_step_count": int(accepted_step_count),
        "rejected_step_count": int(rejected_step_count),
        "done_reason_counts": global_done_reasons,
        "accepted_done_reason_counts": global_accepted_done_reasons,
        "rejected_done_reason_counts": global_rejected_done_reasons,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "loss_mean_last_100": float(np.mean(losses[-100:])) if losses else None,
        "gate_layout_version": str(args.gate_layout_version),
        "gate_yaw_policy": (
            "random_uniform_minus5_to_plus5_deg_formation_facing"
            if str(args.gate_layout_version) == "irregular_centerline_v7_large_arena_dynamic"
            else "layout_default"
        ),
        "moving_gates_enabled": bool(args.moving_gates),
        "moving_gate_amplitude_m": float(args.moving_gate_amplitude_m),
        "moving_gate_speed_hz": float(moving_gate_speed_hz),
        "moving_gate_speed_mps": float(args.moving_gate_speed_mps),
        "drone_speed_mps": float(args.drone_speed_mps),
        "drone_accel_mps2": float(args.drone_accel_mps2),
        "moving_gate_episode_phase_offset_s": float(args.moving_gate_episode_phase_offset_s),
        "moving_gate_episode_phase_stride_s": float(args.moving_gate_episode_phase_stride_s),
        "dynamic_replan_interval_steps": int(args.dynamic_replan_interval_steps),
        "dynamic_replan_clearance_threshold_m": float(args.dynamic_replan_clearance_threshold_m),
        "dynamic_gate_speed_cap_base_mps": float(args.dynamic_gate_speed_cap_base_mps),
        "dynamic_gate_speed_cap_gain": float(args.dynamic_gate_speed_cap_gain),
        "dynamic_shield_rollout_steps": int(args.dynamic_shield_rollout_steps),
        "dynamic_planner_inflation_extra_m": float(args.dynamic_planner_inflation_extra_m),
        "dynamic_final_goal_bias_start_x_m": float(args.dynamic_final_goal_bias_start_x_m),
        "dynamic_final_goal_bias_strength": float(args.dynamic_final_goal_bias_strength),
        "dynamic_controller_profile": str(args.dynamic_controller_profile),
        "stagewise_controller_profile": str(args.stagewise_controller_profile),
        "expert_enable_agent_policy": bool(args.expert_enable_agent_policy),
        "training_render_policy": layout_profile.training_render_policy,
        "obstacle_dynamics_policy": layout_profile.obstacle_dynamics_policy,
        "collision_policy": layout_profile.collision_policy,
        "start_xyz": list(layout_profile.start_xyz),
        "goal_xyz": list(layout_profile.goal_xyz),
        "gate_half_width_m": float(GATE_HALF_WIDTH_M),
        "gate_post_radius_m": float(GATE_POST_RADIUS_M),
        "safety_margin_m": float(SAFETY_MARGIN_M),
    }
    _write_json(output_dir / "stage_manifest.json", metadata)
    _write_json(output_dir / "stage_summary.json", summary)
    _write_json(output_dir / "best_stage_checkpoint_map.json", {"gate_density_imitation_bridge": str(best_path)})
    print("gate-density imitation bridge training complete")
    print(f"sample_count={sample_count}")
    print(f"loss_first={summary['loss_first']}")
    print(f"loss_last={summary['loss_last']}")
    print(f"checkpoint={saved_path}")
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()



