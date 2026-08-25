"""Route-guidance state, deterministic fallback, and request payloads."""

from __future__ import annotations

import math

import numpy as np


def _guidance_runtime_active(self) -> bool:
    reasoning = self.multi_config.reasoning
    return bool(
        getattr(reasoning, "route_guidance_enabled", False) or getattr(reasoning, "guidance_shadow_mode", False)
    )


def _route_guidance_visible(self) -> bool:
    return bool(getattr(self.multi_config.reasoning, "route_guidance_enabled", False))


def _default_route_guidance_meta(
    self,
    *,
    source: str,
    error: str | None = None,
) -> dict[str, object]:
    reasoning = self.multi_config.reasoning
    return {
        "provider": str(getattr(reasoning, "guidance_provider", "none")),
        "model_name": str(getattr(reasoning, "guidance_model_name", "")),
        "prompt_version": str(getattr(reasoning, "guidance_prompt_version", "")),
        "source": str(source),
        "latency_ms": None,
        "cache_hit": False,
        "request_submitted": False,
        "error": None if error is None else str(error),
    }


def _resolve_guidance_query_interval_steps(self) -> int:
    budget_hz = float(getattr(self.multi_config.reasoning, "inference_budget_hz", 0.0) or 0.0)
    if budget_hz <= 0.0:
        return 0
    return max(int(math.ceil(1.0 / max(budget_hz * float(self.env_config.dt_s), 1.0e-6))), 1)


def _heuristic_route_guidance_summary(
    self,
    center_xy: tuple[float, float],
) -> dict[str, float] | None:
    if not self._guidance_runtime_active():
        return None
    positions = self._active_positions_xy()
    min_clearance = self._min_clearance(positions)
    goal_distance = self._goal_distance(center_xy)
    heading_x, heading_y = self._current_guidance_heading(center_xy)
    target_xy = self._plan.waypoints_xy[min(self._path_index, len(self._plan.waypoints_xy) - 1)]
    target_rel_x = (float(target_xy[0]) - center_xy[0]) / 50.0
    target_rel_y = (float(target_xy[1]) - center_xy[1]) / 50.0
    compactness = float(
        np.clip(
            self._mean_slot_error(positions, self._desired_slots[: self._num_agents])
            / max(self.formation_config.goal_slot_tolerance_m * 4.0, 1e-6),
            0.0,
            1.0,
        )
    )
    if min_clearance < 1.0:
        mode_code = -1.0
    elif compactness > 0.6:
        mode_code = 0.0
    else:
        mode_code = 1.0
    return {
        "target_rel_x": float(np.clip(target_rel_x, -1.0, 1.0)),
        "target_rel_y": float(np.clip(target_rel_y, -1.0, 1.0)),
        "heading_x": float(heading_x),
        "heading_y": float(heading_y),
        "risk_level": float(np.clip((2.0 - min_clearance) / 2.0, 0.0, 1.0)),
        "formation_compactness": compactness,
        "speed_scale": float(np.clip(goal_distance / 50.0, 0.0, 1.0)),
        "mode_code": float(mode_code),
        "confidence": 0.85,
    }


def _build_guidance_query_payload(
    self,
    center_xy: tuple[float, float],
    *,
    heuristic_guidance: dict[str, float],
) -> dict[str, object]:
    positions = self._active_positions_xy()
    mean_slot_error_m, max_slot_error_m = self._slot_error_stats(
        positions,
        self._desired_slots[: self._num_agents],
    )
    min_pair_distance_m = self._pairwise_collision_stats(positions)[1]
    lookahead_waypoints = self._lookahead_waypoints()[:6]
    nearby_obstacles = list(self._active_obstacle_map().query_local(center_xy, 20.0))
    nearby_obstacles.sort(
        key=lambda obstacle: math.hypot(
            obstacle.center_xy[0] - center_xy[0],
            obstacle.center_xy[1] - center_xy[1],
        )
    )
    route_plan_guidance = self._route_plan_guidance_summary(center_xy)
    return {
        "prompt_version": str(getattr(self.multi_config.reasoning, "guidance_prompt_version", "exp3_v1")),
        "stage_name": str(
            getattr(self.multi_config.reasoning, "guidance_stage_name", "") or self.multi_config.paper_variant
        ),
        "scene_mode": str(getattr(self.multi_config.scene, "scene_mode", "")),
        "team_size": int(self._num_agents),
        "virtual_center_xy": [float(center_xy[0]), float(center_xy[1])],
        "goal_distance_m": float(self._goal_distance(center_xy)),
        "min_clearance_m": float(self._min_clearance(positions)),
        "min_pair_distance_m": (None if not math.isfinite(min_pair_distance_m) else float(min_pair_distance_m)),
        "mean_slot_error_m": float(mean_slot_error_m),
        "max_slot_error_m": float(max_slot_error_m),
        "path_index": int(self._path_index),
        "lookahead_waypoints_xy": [[float(x), float(y)] for x, y in lookahead_waypoints],
        "local_obstacles": [
            {
                "rel_x": float(obstacle.center_xy[0] - center_xy[0]),
                "rel_y": float(obstacle.center_xy[1] - center_xy[1]),
                "distance_m": float(
                    math.hypot(obstacle.center_xy[0] - center_xy[0], obstacle.center_xy[1] - center_xy[1])
                ),
                "radius_m": float(obstacle.collision_radius_m),
            }
            for obstacle in nearby_obstacles[:6]
        ],
        "command_limits": {
            "forward_speed_mps": float(self._resolved_forward_command_speed_mps()),
            "lateral_speed_mps": float(self._resolved_lateral_command_speed_mps()),
            "forward_accel_mps2": float(
                self.env_config.max_accel_mps2
                if self.env_config.max_forward_accel_mps2 is None
                else self.env_config.max_forward_accel_mps2
            ),
            "lateral_accel_mps2": float(
                self.env_config.max_accel_mps2
                if self.env_config.max_lateral_accel_mps2 is None
                else self.env_config.max_lateral_accel_mps2
            ),
            "inter_agent_safe_distance_m": float(self.env_config.inter_agent_safe_distance_m),
        },
        "route_plan_guidance": dict(route_plan_guidance or {}),
        "heuristic_route_guidance": dict(heuristic_guidance),
    }


def _refresh_route_guidance(
    self,
    center_xy: tuple[float, float],
    *,
    force: bool = False,
) -> None:
    if not self._guidance_runtime_active():
        self._route_guidance_state = None
        self._route_guidance_meta = self._default_route_guidance_meta(source="disabled")
        return

    heuristic_guidance = self._heuristic_route_guidance_summary(center_xy)
    if heuristic_guidance is None:
        self._route_guidance_state = None
        self._route_guidance_meta = self._default_route_guidance_meta(source="disabled")
        return
    if self._route_guidance_state is None:
        self._route_guidance_state = dict(heuristic_guidance)
        self._route_guidance_meta = self._default_route_guidance_meta(source="heuristic_bootstrap")

    if self._guidance_engine is None:
        self._route_guidance_state = dict(heuristic_guidance)
        self._route_guidance_meta = self._default_route_guidance_meta(source="heuristic_only")
        return

    allow_submit = bool(force)
    if not allow_submit and self._guidance_query_interval_steps > 0:
        allow_submit = self._step_count % self._guidance_query_interval_steps == 0
    guidance_update, meta = self._guidance_engine.request_guidance(
        session_key=self._guidance_session_key,
        payload=self._build_guidance_query_payload(center_xy, heuristic_guidance=heuristic_guidance),
        fallback_guidance=heuristic_guidance,
        allow_submit=allow_submit,
    )
    if guidance_update is not None:
        self._route_guidance_state = dict(guidance_update)
        self._route_guidance_meta = dict(meta)
        return
    if meta.get("error"):
        fallback_meta = dict(meta)
        fallback_meta["source"] = "heuristic_fallback"
        self._route_guidance_state = dict(heuristic_guidance)
        self._route_guidance_meta = fallback_meta
        return
    if bool(meta.get("request_submitted", False)) or str(meta.get("source") or "").startswith("guidance_async"):
        self._route_guidance_meta = dict(meta)
        return
    if self._route_guidance_state is None:
        pending_meta = dict(meta)
        if str(pending_meta.get("source") or "").startswith("guidance_async"):
            pending_meta["source"] = "heuristic_pending"
        self._route_guidance_state = dict(heuristic_guidance)
        self._route_guidance_meta = pending_meta
