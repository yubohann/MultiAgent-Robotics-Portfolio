"""Thin local HTTP client used by the route guidance bridge."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib import error, request


class LocalGuidanceClientError(RuntimeError):
    """Raised when the local guidance service cannot satisfy one request."""


@dataclass(frozen=True)
class LocalGuidanceResult:
    """Structured result returned by one guidance call."""

    content: dict[str, Any]
    latency_ms: float
    raw_payload: dict[str, Any]


class LocalGuidanceClient:
    """Small synchronous client for the local `/api/chat` endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        num_ctx: int = 4096,
        num_predict: int = 128,
        keep_alive: str = "10m",
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.num_ctx = max(int(num_ctx), 512)
        self.num_predict = max(int(num_predict), 16)
        self.keep_alive = str(keep_alive)

    def _trace_enabled(self) -> bool:
        value = os.environ.get("GATE2D_GUIDANCE_TRACE", os.environ.get("GUIDANCE_TRACE", ""))
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _trace_preview(self, value: Any, *, limit: int = 1200) -> str:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    def _trace_print(self, line: str) -> None:
        if self._trace_enabled():
            print(line, flush=True)

    def chat_json(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        timeout_s: float = 1.5,
        temperature: float = 0.1,
    ) -> LocalGuidanceResult:
        request_payload = {
            "model": str(model_name),
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "options": {
                "temperature": float(temperature),
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }
        encoded_request = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}/api/chat",
            data=encoded_request,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        trace_id = f"{time.time_ns():x}"
        trace_time = datetime.now().astimezone().isoformat(timespec="seconds")
        self._trace_print(
            "[GUIDANCE_CALL start] "
            f"time={trace_time} id={trace_id} model={model_name} url={self.base_url}/api/chat "
            f"timeout_s={float(timeout_s):.2f} temperature={float(temperature):.2f} "
            f"num_ctx={self.num_ctx} num_predict={self.num_predict} keep_alive={self.keep_alive}"
        )
        self._trace_print(
            "[GUIDANCE_CALL prompt] "
            f"time={trace_time} id={trace_id} system={self._trace_preview(system_prompt, limit=500)} "
            f"user_payload={self._trace_preview(user_payload, limit=1800)}"
        )
        started_at = time.perf_counter()
        try:
            with request.urlopen(http_request, timeout=max(float(timeout_s), 0.05)) as response:
                payload_bytes = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._trace_print(
                "[GUIDANCE_CALL error] "
                f"time={trace_time} id={trace_id} http_status={exc.code} detail={self._trace_preview(detail, limit=1200)}"
            )
            raise LocalGuidanceClientError(f"Guidance HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            self._trace_print("[GUIDANCE_CALL error] " f"time={trace_time} id={trace_id} error={exc}")
            raise LocalGuidanceClientError(f"Guidance request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        try:
            raw_payload = json.loads(payload_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._trace_print("[GUIDANCE_CALL error] " f"time={trace_time} id={trace_id} error=non_json_http_payload")
            raise LocalGuidanceClientError("Guidance service returned non-JSON HTTP payload.") from exc

        message = raw_payload.get("message")
        message_content = message.get("content") if isinstance(message, dict) else raw_payload.get("response")
        if not isinstance(message_content, str) or not message_content.strip():
            self._trace_print("[GUIDANCE_CALL error] " f"time={trace_time} id={trace_id} error=empty_message_body")
            raise LocalGuidanceClientError("Guidance response did not contain a JSON message body.")

        try:
            content = json.loads(message_content)
        except json.JSONDecodeError as exc:
            self._trace_print(
                "[GUIDANCE_CALL error] "
                f"time={trace_time} id={trace_id} error=invalid_json_content content={self._trace_preview(message_content, limit=1200)}"
            )
            raise LocalGuidanceClientError(f"Guidance service returned invalid JSON content: {message_content[:240]}") from exc
        if not isinstance(content, dict):
            self._trace_print("[GUIDANCE_CALL error] " f"time={trace_time} id={trace_id} error=json_content_not_object")
            raise LocalGuidanceClientError("Guidance JSON content must be an object.")
        self._trace_print(
            "[GUIDANCE_CALL done] "
            f"time={trace_time} id={trace_id} latency_ms={latency_ms:.1f} content={self._trace_preview(content, limit=1800)}"
        )
        return LocalGuidanceResult(
            content=content,
            latency_ms=float(latency_ms),
            raw_payload=raw_payload,
        )



