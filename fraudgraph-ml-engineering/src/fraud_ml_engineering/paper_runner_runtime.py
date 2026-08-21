from __future__ import annotations

import hashlib
import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

DEFAULT_STAGE_ORDER = (
    "dataset_load",
    "static_graph_prepare",
    "community_fit",
    "forward",
    "backward",
    "eval",
    "json_write",
)


class StageTimer:
    def __init__(self, stages: tuple[str, ...] = DEFAULT_STAGE_ORDER) -> None:
        self._stages = tuple(stages)
        self._totals = {stage: 0.0 for stage in self._stages}
        self._counts = {stage: 0 for stage in self._stages}

    @contextmanager
    def track(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - start)

    def add(self, stage: str, seconds: float) -> None:
        if stage not in self._totals:
            self._totals[stage] = 0.0
            self._counts[stage] = 0
        self._totals[stage] += float(max(seconds, 0.0))
        self._counts[stage] += 1

    def merge(self, other: "StageTimer") -> None:
        for stage, seconds in other._totals.items():
            self._totals[stage] = self._totals.get(stage, 0.0) + float(seconds)
            self._counts[stage] = self._counts.get(stage, 0) + int(other._counts.get(stage, 0))

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        ordered = list(self._stages)
        for stage in self._totals:
            if stage not in ordered:
                ordered.append(stage)
        return {
            stage: {
                "seconds": round(float(self._totals.get(stage, 0.0)), 6),
                "count": int(self._counts.get(stage, 0)),
            }
            for stage in ordered
        }


def stable_cache_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def ensure_cache_dir(root: str | Path, namespace: str) -> Path:
    path = Path(root).expanduser().resolve() / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def select_amp_dtype(device: torch.device, requested: str = "auto") -> tuple[bool, torch.dtype | None, str]:
    if device.type != "cuda":
        return False, None, "disabled_non_cuda"
    mode = str(requested or "auto").strip().lower()
    if mode in {"off", "false", "none", "0"}:
        return False, None, "disabled_by_flag"
    windows_platform = platform.system().lower() == "windows"
    if mode == "bf16":
        if windows_platform:
            return True, torch.float16, "fp16_windows_bf16_fallback"
        return True, torch.bfloat16, "bf16"
    if mode == "fp16":
        return True, torch.float16, "fp16"
    if windows_platform:
        return True, torch.float16, "fp16_auto_windows_dgl"
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bf16_auto"
    return True, torch.float16, "fp16_auto"
