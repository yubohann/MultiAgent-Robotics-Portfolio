"""Training helpers for the multi-agent 2D gate experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

from multi_gate.configs.experiment_config import MultiExperimentConfig, is_exp3_kinematic_3d_scene_mode
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from shared.runtime.artifacts import write_json

MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv

def _start_live_isaaclab_preview(
    *,
    snapshot_path: Path,
    experiment_config: MultiExperimentConfig,
    follow_agent_index: int,
    headless: bool,
) -> subprocess.Popen | None:
    scene_mode = str(getattr(experiment_config.scene, "scene_mode", "")).strip().lower()
    if not is_exp3_kinematic_3d_scene_mode(scene_mode):
        return None
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "live_preview_multi_isaaclab.py"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--snapshot",
        str(snapshot_path),
        "--config-name",
        _resolve_live_preview_config_name(experiment_config),
        "--scene-mode",
        scene_mode,
        "--render-real-gate",
        "1" if bool(experiment_config.scene.render_real_gate) else "0",
        "--render-real-drone-shell",
        "1" if bool(experiment_config.scene.render_real_drone_shell) else "0",
        "--follow-agent-index",
        str(max(int(follow_agent_index), 0)),
    ]
    if bool(headless):
        command.append("--headless")
    return subprocess.Popen(command)

def _resolve_live_preview_config_name(experiment_config: MultiExperimentConfig) -> str:
    paper_variant = str(getattr(experiment_config, "paper_variant", "")).strip().lower()
    if paper_variant in {"e3_baseline", "e3_main", "e3_guidance"}:
        return paper_variant
    return "variable"

def _write_live_preview_snapshot(
    *,
    snapshot_path: Path,
    env: MultiEnvType,
    experiment_config: MultiExperimentConfig,
    transitions_collected: int,
) -> None:
    positions_xy = env.active_positions_xy()
    velocities_xy = env.active_velocities_xy()
    snapshot = env.snapshot()
    payload = {
        "experiment_id": experiment_config.experiment_id,
        "paper_track": experiment_config.paper_track,
        "paper_variant": experiment_config.paper_variant,
        "scene_mode": experiment_config.scene.scene_mode,
        "render_real_gate": bool(experiment_config.scene.render_real_gate),
        "render_real_drone_shell": bool(experiment_config.scene.render_real_drone_shell),
        "transitions_collected": int(transitions_collected),
        "num_agents": int(snapshot.num_agents),
        "max_agents": int(experiment_config.max_agents_soft),
        "fixed_height_m": float(env.env_config.fixed_height_m),
        "positions_xy": positions_xy.tolist(),
        "velocities_xy": velocities_xy.tolist(),
        "yaws_rad": [float(state.yaw_rad) for state in env._states],
        "desired_slots_xy": env.desired_slots_xy().tolist(),
        "virtual_center_xy": list(snapshot.virtual_center_xy),
        "lookahead_heading_xy": list(env.current_heading_xy()),
        "path_waypoints": [list(point) for point in env.path_waypoints()],
        "path_index": int(snapshot.path_index),
        "start_xy": list(env._start_center_xy),
        "goal_xy": list(env.path_waypoints()[-1]),
        "world_x_bounds_m": list(env.env_config.world_x_bounds_m),
        "world_y_bounds_m": list(env.env_config.world_y_bounds_m),
        "route_plan_guidance": env._route_plan_guidance_summary(snapshot.virtual_center_xy),
        "route_guidance": env._route_guidance_summary(snapshot.virtual_center_xy),
        "route_guidance_meta": dict(getattr(env, "_route_guidance_meta", {})),
    }
    write_json(snapshot_path, payload)