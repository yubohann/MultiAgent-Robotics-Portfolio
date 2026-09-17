"""Local HTTP client for single-drone gate-density route guidance."""

from __future__ import annotations

import json
import math
import time
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

import numpy as np


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("empty guidance response")
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Guidance response does not contain a JSON object: {stripped[:160]}")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Guidance JSON payload is not an object")
    return payload


class LocalGateGuidanceClient:
    """HTTP client that queries one local guidance model per control step."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float,
        temperature: float,
        prompt_version: str,
        cache_enabled: bool,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.prompt_version = str(prompt_version)
        self.cache_enabled = bool(cache_enabled)
        self.cache: dict[str, dict[str, Any]] = {}
        self.query_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.fallback_count = 0
        self.cache_hit_count = 0
        self.latencies_ms: list[float] = []
        self.guidance_records: list[dict[str, Any]] = []

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def health_check(self) -> None:
        """Fail early when the guidance endpoint is unavailable."""

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return exactly this JSON object and no extra text: "
                        '{"heading_x":1.0,"heading_y":0.0,"speed_scale":0.5,"confidence":1.0,"risk_level":0.0}'
                    ),
                }
            ],
            "options": {"temperature": 0.0},
        }
        self._post_chat(payload, timeout_s=self.timeout_s)

    def query(
        self,
        *,
        step: int,
        position_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        clearance_m: float,
        gate_count: int,
        nearest_gate_posts_xy: list[tuple[float, float]],
        slow_guidance_action: np.ndarray,
        risk_context: dict[str, Any],
    ) -> dict[str, Any]:
        slow_norm = float(np.linalg.norm(slow_guidance_action))
        if slow_norm <= 1e-6:
            slow_heading = (1.0, 0.0)
        else:
            slow_heading = (
                float(slow_guidance_action[0] / slow_norm),
                float(slow_guidance_action[1] / slow_norm),
            )
        goal_dx = float(goal_xy[0] - position_xy[0])
        goal_dy = float(goal_xy[1] - position_xy[1])
        goal_distance = math.hypot(goal_dx, goal_dy)
        clearance_for_cache = float(clearance_m) if math.isfinite(float(clearance_m)) else 999.0
        cache_key = json.dumps(
            {
                "bucket_x": round(float(position_xy[0]) * 2.0) / 2.0,
                "bucket_y": round(float(position_xy[1]) * 2.0) / 2.0,
                "gate_count": int(gate_count),
                "clearance_bucket": round(clearance_for_cache * 2.0) / 2.0,
                "risk_bucket": round(float(risk_context.get("moving_gate_crossing_risk", 0.0)) * 4.0) / 4.0,
                "trend_bucket": round(float(risk_context.get("clearance_trend_m_per_s", 0.0)) * 5.0) / 5.0,
                "slow_heading": (round(slow_heading[0], 2), round(slow_heading[1], 2)),
            },
            sort_keys=True,
        )
        if self.cache_enabled and cache_key in self.cache:
            self.cache_hit_count += 1
            guidance = dict(self.cache[cache_key])
            guidance["guidance_cache_hit"] = True
            guidance["route_guidance_source"] = "guidance_cache"
            self.guidance_records.append(guidance | {"step": int(step)})
            return guidance

        prompt = {
            "task": "single_drone_gate_density_guidance",
            "prompt_version": self.prompt_version,
            "instruction": (
                "You guide one drone from start to goal in a rectangular arena with gate posts as obstacles. "
                "Return only compact JSON. Choose a safe heading close to the planner heading unless risk is high. "
                "For moving gates, reason about short-horizon risk. Prefer speed modulation, replan urgency, and "
                "small waypoint_bias_y over large heading changes. Keep speed_scale 0.85-1.0 when the planner path "
                "is still viable, 0.65-0.85 when clearance is shrinking, and only use 0.40-0.60 for imminent conflict. "
                "Use replan_urgency above 0.85 only when the current corridor is blocked within the next 1-2 seconds."
            ),
            "required_json_schema": {
                "heading_x": "float in [-1,1]",
                "heading_y": "float in [-1,1]",
                "speed_scale": "float in [0.2,1.0]",
                "confidence": "float in [0,1]",
                "risk_level": "float in [0,1]",
                "preferred_side": "one of left,right,center",
                "replan_urgency": "float in [0,1]",
                "waypoint_bias_y": "float in [-0.8,0.8]",
                "dynamic_clearance_margin_m": "float in [0,0.6]",
            },
            "state": {
                "step": int(step),
                "position_xy": [float(position_xy[0]), float(position_xy[1])],
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                "goal_distance_m": float(goal_distance),
                "clearance_m": float(clearance_m),
                "gate_count": int(gate_count),
                "nearest_gate_posts_xy": [[float(x), float(y)] for x, y in nearest_gate_posts_xy[:6]],
                "slow_planner_heading": [float(slow_heading[0]), float(slow_heading[1])],
                "dynamic_risk_context": risk_context,
            },
        }
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a precise UAV navigation guidance module. Output only JSON."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))},
            ],
            "options": {"temperature": self.temperature},
        }
        self.query_count += 1
        started_at = time.perf_counter()
        try:
            raw = self._post_chat(payload, timeout_s=self.timeout_s)
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            self.latencies_ms.append(float(latency_ms))
            content = str(raw.get("message", {}).get("content", ""))
            parsed = _extract_json_object(content)
            guidance = self._normalize_guidance(parsed)
            guidance.update(
                {
                    "guidance_latency_ms": float(latency_ms),
                    "guidance_cache_hit": False,
                    "route_guidance_source": "guidance_live",
                    "guidance_raw_content": content[:500],
                }
            )
            self.success_count += 1
        except Exception as exc:
            self.failure_count += 1
            self.fallback_count += 1
            guidance = {
                "heading_x": float(slow_heading[0]),
                "heading_y": float(slow_heading[1]),
                "speed_scale": 0.55,
                "confidence": 0.0,
                "risk_level": 1.0,
                "preferred_side": "center",
                "replan_urgency": 1.0,
                "waypoint_bias_y": 0.0,
                "dynamic_clearance_margin_m": 0.4,
                "guidance_latency_ms": float((time.perf_counter() - started_at) * 1000.0),
                "guidance_cache_hit": False,
                "route_guidance_source": "fallback_after_error",
                "guidance_error": repr(exc),
            }
        if self.cache_enabled:
            self.cache[cache_key] = dict(guidance)
        self.guidance_records.append(guidance | {"step": int(step)})
        return guidance

    def _post_chat(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = url_request.Request(
            self.chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with url_request.urlopen(req, timeout=float(timeout_s)) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, OSError, url_error.URLError) as exc:
            raise RuntimeError(f"Guidance endpoint unavailable at {self.chat_url}: {exc}") from exc

    @staticmethod
    def _normalize_guidance(payload: dict[str, Any]) -> dict[str, Any]:
        hx = float(payload.get("heading_x", 1.0))
        hy = float(payload.get("heading_y", 0.0))
        norm = math.hypot(hx, hy)
        if norm <= 1e-6:
            hx, hy = 1.0, 0.0
        else:
            hx, hy = hx / norm, hy / norm
        preferred_side = str(payload.get("preferred_side", "center")).strip().lower()
        return {
            "heading_x": float(np.clip(hx, -1.0, 1.0)),
            "heading_y": float(np.clip(hy, -1.0, 1.0)),
            "speed_scale": float(np.clip(float(payload.get("speed_scale", 0.55)), 0.2, 1.0)),
            "confidence": float(np.clip(float(payload.get("confidence", 0.0)), 0.0, 1.0)),
            "risk_level": float(np.clip(float(payload.get("risk_level", 0.5)), 0.0, 1.0)),
            "preferred_side": preferred_side if preferred_side in {"left", "right", "center"} else "center",
            "replan_urgency": float(np.clip(float(payload.get("replan_urgency", 0.0)), 0.0, 1.0)),
            "waypoint_bias_y": float(np.clip(float(payload.get("waypoint_bias_y", 0.0)), -0.8, 0.8)),
            "dynamic_clearance_margin_m": float(
                np.clip(float(payload.get("dynamic_clearance_margin_m", 0.0)), 0.0, 0.6)
            ),
        }
