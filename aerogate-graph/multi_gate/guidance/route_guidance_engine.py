"""Asynchronous low-frequency route guidance bridge with heuristic fallback."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import threading
from typing import Any, Mapping

from multi_gate.guidance.local_guidance_client import LocalGuidanceClient, LocalGuidanceClientError


GUIDANCE_FIELD_DEFAULTS: dict[str, float] = {
    "target_rel_x": 0.0,
    "target_rel_y": 0.0,
    "heading_x": 1.0,
    "heading_y": 0.0,
    "risk_level": 0.0,
    "formation_compactness": 0.0,
    "speed_scale": 0.5,
    "mode_code": 0.0,
    "confidence": 0.5,
}


@dataclass(frozen=True)
class _PendingGuidanceRequest:
    future: Future
    cache_key: str


class RouteGuidanceEngine:
    """Low-frequency guidance engine that never blocks env stepping on remote calls."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model_name: str,
        timeout_s: float,
        temperature: float,
        prompt_version: str,
        async_enabled: bool,
        cache_enabled: bool,
        client: LocalGuidanceClient | None = None,
        max_workers: int = 1,
    ) -> None:
        self.provider = str(provider or "none").strip().lower()
        self.base_url = str(base_url or "http://127.0.0.1:11434").rstrip("/")
        self.model_name = str(model_name or "local-guidance-model").strip()
        self.timeout_s = max(float(timeout_s), 0.05)
        self.temperature = float(temperature)
        self.prompt_version = str(prompt_version or "exp3_v1").strip() or "exp3_v1"
        self.async_enabled = bool(async_enabled)
        self.cache_enabled = bool(cache_enabled)
        self._client = client or LocalGuidanceClient(base_url=self.base_url)
        self._max_workers = max(int(max_workers), 1)
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, float]] = {}
        self._pending_by_session: dict[str, _PendingGuidanceRequest] = {}

    @property
    def runtime_enabled(self) -> bool:
        return self.provider == "local_http"

    def shutdown(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def request_guidance(
        self,
        *,
        session_key: str,
        payload: Mapping[str, Any],
        fallback_guidance: Mapping[str, float],
        allow_submit: bool,
    ) -> tuple[dict[str, float] | None, dict[str, object]]:
        fallback = self._sanitize_guidance({}, fallback_guidance=fallback_guidance)
        if not self.runtime_enabled:
            return None, self._meta(source="disabled", error=None)

        normalized_payload = self._normalize_payload(payload)
        cache_key = self._cache_key(normalized_payload)
        if self.cache_enabled:
            with self._lock:
                cached = copy.deepcopy(self._cache.get(cache_key))
            if cached is not None:
                return cached, self._meta(source="guidance_cache", cache_hit=True)

        pending_result = self._poll_completed_request(session_key=session_key, fallback_guidance=fallback)
        if pending_result is not None:
            return pending_result

        with self._lock:
            has_pending = str(session_key) in self._pending_by_session
        if has_pending:
            return None, self._meta(source="guidance_async_pending")
        if not bool(allow_submit):
            return None, self._meta(source="budget_hold")

        if self.async_enabled:
            future = self._ensure_executor().submit(self._execute_request, normalized_payload)
            with self._lock:
                self._pending_by_session[str(session_key)] = _PendingGuidanceRequest(
                    future=future,
                    cache_key=cache_key,
                )
            return None, self._meta(source="guidance_async_submitted", request_submitted=True)

        try:
            raw_guidance, latency_ms = self._execute_request(normalized_payload)
        except Exception as exc:  # pragma: no cover - covered via async polling / env fallback tests
            return None, self._meta(source="guidance_error", error=str(exc))
        guidance = self._sanitize_guidance(raw_guidance, fallback_guidance=fallback)
        if self.cache_enabled:
            with self._lock:
                self._cache[cache_key] = copy.deepcopy(guidance)
        return guidance, self._meta(source="guidance_live", latency_ms=latency_ms)

    def _poll_completed_request(
        self,
        *,
        session_key: str,
        fallback_guidance: Mapping[str, float],
    ) -> tuple[dict[str, float] | None, dict[str, object]] | None:
        with self._lock:
            pending = self._pending_by_session.get(str(session_key))
        if pending is None or not pending.future.done():
            return None
        with self._lock:
            self._pending_by_session.pop(str(session_key), None)
        try:
            raw_guidance, latency_ms = pending.future.result()
        except Exception as exc:
            return None, self._meta(source="guidance_error", error=str(exc))
        guidance = self._sanitize_guidance(raw_guidance, fallback_guidance=fallback_guidance)
        if self.cache_enabled:
            with self._lock:
                self._cache[pending.cache_key] = copy.deepcopy(guidance)
        return guidance, self._meta(source="guidance_live", latency_ms=latency_ms)

    def _execute_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        system_prompt = self._build_system_prompt()
        result = self._client.chat_json(
            model_name=self.model_name,
            system_prompt=system_prompt,
            user_payload=payload,
            timeout_s=self.timeout_s,
            temperature=self.temperature,
        )
        return dict(result.content), float(result.latency_ms)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="exp3_guidance",
            )
        return self._executor

    def _build_system_prompt(self) -> str:
        return (
            "You are the low-frequency strategic planner for a multi-drone formation task. "
            "Do not output low-level actions. Do not explain. "
            "Read the compressed state, the A* corridor hints, and the safety statistics. "
            "Return exactly one JSON object with numeric fields: "
            "target_rel_x, target_rel_y, heading_x, heading_y, risk_level, "
            "formation_compactness, speed_scale, mode_code, confidence. "
            "Respect safety first. If risk is high, reduce speed_scale and keep the heading conservative."
        )

    def _meta(
        self,
        *,
        source: str,
        latency_ms: float | None = None,
        cache_hit: bool = False,
        request_submitted: bool = False,
        error: str | None = None,
    ) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "source": str(source),
            "latency_ms": None if latency_ms is None else float(latency_ms),
            "cache_hit": bool(cache_hit),
            "request_submitted": bool(request_submitted),
            "error": None if error is None else str(error),
        }

    def _cache_key(self, payload: Mapping[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _normalize_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        def _normalize(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
            if isinstance(value, (list, tuple)):
                return [_normalize(item) for item in value]
            if isinstance(value, float):
                if not math.isfinite(value):
                    return 0.0
                return round(float(value), 4)
            if isinstance(value, (int, str, bool)) or value is None:
                return value
            return str(value)

        return dict(_normalize(dict(payload)))

    def _sanitize_guidance(
        self,
        response: Mapping[str, Any],
        *,
        fallback_guidance: Mapping[str, float],
    ) -> dict[str, float]:
        fallback = {
            key: float(fallback_guidance.get(key, default))
            for key, default in GUIDANCE_FIELD_DEFAULTS.items()
        }
        guidance = dict(fallback)
        for key in guidance:
            value = response.get(key, guidance[key])
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = guidance[key]
            if not math.isfinite(numeric_value):
                numeric_value = guidance[key]
            guidance[key] = numeric_value

        guidance["target_rel_x"] = float(max(min(guidance["target_rel_x"], 1.0), -1.0))
        guidance["target_rel_y"] = float(max(min(guidance["target_rel_y"], 1.0), -1.0))
        heading_x = float(guidance["heading_x"])
        heading_y = float(guidance["heading_y"])
        heading_norm = math.hypot(heading_x, heading_y)
        if heading_norm <= 1.0e-6:
            heading_x = float(fallback.get("heading_x", 1.0))
            heading_y = float(fallback.get("heading_y", 0.0))
            heading_norm = math.hypot(heading_x, heading_y)
        guidance["heading_x"] = float(heading_x / max(heading_norm, 1.0e-6))
        guidance["heading_y"] = float(heading_y / max(heading_norm, 1.0e-6))
        guidance["risk_level"] = float(max(min(guidance["risk_level"], 1.0), 0.0))
        guidance["formation_compactness"] = float(max(min(guidance["formation_compactness"], 1.0), 0.0))
        guidance["speed_scale"] = float(max(min(guidance["speed_scale"], 1.0), 0.0))
        guidance["mode_code"] = float(max(min(guidance["mode_code"], 1.0), -1.0))
        guidance["confidence"] = float(max(min(guidance["confidence"], 1.0), 0.0))
        return guidance


def build_guidance_engine_from_reasoning(reasoning_config: object) -> RouteGuidanceEngine | None:
    """Construct one shared guidance engine from the reasoning config when runtime is enabled."""

    route_guidance_enabled = bool(getattr(reasoning_config, "route_guidance_enabled", False))
    guidance_shadow_mode = bool(getattr(reasoning_config, "guidance_shadow_mode", False))
    provider = str(getattr(reasoning_config, "guidance_provider", "none")).strip().lower()
    if provider != "local_http" or not (route_guidance_enabled or guidance_shadow_mode):
        return None
    return RouteGuidanceEngine(
        provider=provider,
        base_url=str(getattr(reasoning_config, "guidance_base_url", "http://127.0.0.1:11434")),
        model_name=str(getattr(reasoning_config, "guidance_model_name", "local-guidance-model")),
        timeout_s=float(getattr(reasoning_config, "guidance_timeout_s", 1.5)),
        temperature=float(getattr(reasoning_config, "guidance_temperature", 0.1)),
        prompt_version=str(getattr(reasoning_config, "guidance_prompt_version", "exp3_v1")),
        async_enabled=bool(getattr(reasoning_config, "guidance_async_enabled", True)),
        cache_enabled=bool(getattr(reasoning_config, "guidance_cache_enabled", True)),
    )

