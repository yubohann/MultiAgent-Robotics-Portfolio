"""Scene sanity gates for dynamic gate-density experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from multi_gate.configs.experiment_config import MultiExperimentConfig
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from shared.core.dynamic_gate_density_2d import assert_dynamic_gate_density_geometry_sane


def validate_dynamic_gate_density_environment(
    *,
    experiment_config: MultiExperimentConfig,
    seed: int = 0,
    num_agents: int = 8,
    forward_steps: int = 12,
) -> dict[str, object]:
    """Check motion, rollout, and collision behavior for the scene."""

    failures: list[str] = []
    gate_cfg = experiment_config.dynamic_gate_density
    gate_count = int(getattr(gate_cfg, "gate_count", 0) or 0)
    speed_mps = float(getattr(gate_cfg, "moving_gate_speed_mps", 0.0) or 0.0)
    amplitude_m = float(getattr(gate_cfg, "moving_gate_amplitude_m", 0.0) or 0.0)

    geometry_report = assert_dynamic_gate_density_geometry_sane(
        gate_count=gate_count,
        speed_mps=speed_mps,
        amplitude_m=amplitude_m,
        seed=seed,
        config=gate_cfg,
    )

    # Keep this preflight renderer- and guidance-free; it only checks motion,
    # kinematics, and collision behavior.
    sanity_config = replace(
        experiment_config,
        reasoning=replace(
            experiment_config.reasoning,
            route_guidance_enabled=False,
            guidance_shadow_mode=False,
            guidance_async_enabled=False,
            guidance_cache_enabled=False,
            guidance_provider="none",
            inference_budget_hz=0.0,
        ),
    )

    env = MultiGate2DEnv(multi_config=sanity_config)
    try:
        _, reset_info = env.reset(seed=seed, num_agents=num_agents)
        if int(reset_info.get("dynamic_gate_count") or 0) != gate_count:
            failures.append(
                "env_gate_count_mismatch: "
                f"expected {gate_count}, got {reset_info.get('dynamic_gate_count')}"
            )

        dynamic_motion_report: dict[str, object] = {
            "checked": bool(gate_count > 0 and speed_mps > 0.0 and amplitude_m > 0.0),
            "motion_after_zero_action_m": 0.0,
        }
        if bool(dynamic_motion_report["checked"]):
            start_centers = np.asarray(reset_info.get("live_gate_centers_xy"), dtype=np.float32)
            zero_action = np.zeros(env.action_shape, dtype=np.float32)
            info = reset_info
            for _ in range(3):
                _, _, terminated, truncated, info = env.step(zero_action)
                if terminated or truncated:
                    failures.append(
                        f"dynamic_motion_probe_ended_early: done_reason={info.get('done_reason')}"
                    )
                    break
            end_centers = np.asarray(info.get("live_gate_centers_xy"), dtype=np.float32)
            if start_centers.shape == end_centers.shape and start_centers.size > 0:
                motion = float(np.max(np.linalg.norm(end_centers - start_centers, axis=1)))
            else:
                motion = 0.0
            dynamic_motion_report["motion_after_zero_action_m"] = motion
            if motion < max(0.03, 0.05 * amplitude_m):
                failures.append(f"env_dynamic_gates_not_moving: motion={motion:.6f}")

        _, start_info = env.reset(seed=seed + 17, num_agents=num_agents)
        start_center_x = float(start_info["virtual_center_xy"][0])
        forward_action = np.zeros(env.action_shape, dtype=np.float32)
        forward_action[: int(num_agents), 0] = 1.0
        info = start_info
        ended_early = False
        for _ in range(max(int(forward_steps), 1)):
            _, _, terminated, truncated, info = env.step(forward_action)
            if terminated or truncated:
                ended_early = True
                break
        end_center_x = float(info["virtual_center_xy"][0])
        progress_x = end_center_x - start_center_x
        if progress_x <= 0.20:
            failures.append(f"forward_progress_too_small: dx={progress_x:.6f}")
        if ended_early and str(info.get("done_reason") or "") != "goal_reached":
            failures.append(f"forward_probe_ended_early: done_reason={info.get('done_reason')}")

        collision_report: dict[str, object] = {"checked": bool(gate_count > 0)}
        if gate_count > 0:
            _, collision_reset_info = env.reset(seed=seed + 33, num_agents=num_agents)
            posts = np.asarray(collision_reset_info.get("live_gate_post_positions_xy"), dtype=np.float32)
            if posts.size == 0:
                failures.append("collision_probe_missing_gate_posts")
            else:
                env._states = [
                    replace(state, x_m=float(posts[0, 0]), y_m=float(posts[0, 1]), vx_mps=0.0, vy_mps=0.0)
                    if idx == 0
                    else state
                    for idx, state in enumerate(env._states)
                ]
                zero_action = np.zeros(env.action_shape, dtype=np.float32)
                _, _, terminated, truncated, collision_info = env.step(zero_action)
                collision_report.update(
                    {
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "done_reason": collision_info.get("done_reason"),
                        "dynamic_gate_collision": bool(collision_info.get("dynamic_gate_collision", False)),
                        "min_clearance_m": collision_info.get("min_clearance_m"),
                    }
                )
                if not bool(terminated):
                    failures.append(
                        "gate_post_contact_did_not_terminate: "
                        f"terminated={terminated}, truncated={truncated}, reason={collision_info.get('done_reason')}"
                    )
                if not bool(collision_info.get("dynamic_gate_collision", False)):
                    failures.append("gate_post_contact_not_marked_dynamic_gate_collision")

        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "geometry": geometry_report,
            "dynamic_motion": dynamic_motion_report,
            "forward_progress": {
                "start_center_x_m": start_center_x,
                "end_center_x_m": end_center_x,
                "delta_x_m": progress_x,
                "ended_early": ended_early,
                "done_reason": info.get("done_reason"),
            },
            "collision": collision_report,
        }
    finally:
        env.close()


def assert_dynamic_gate_density_environment_sane(**kwargs: Any) -> dict[str, object]:
    report = validate_dynamic_gate_density_environment(**kwargs)
    if not bool(report.get("passed")):
        failures = ", ".join(str(item) for item in report.get("failures", []))
        raise AssertionError(f"Dynamic gate-density environment sanity failed: {failures}")
    return report

