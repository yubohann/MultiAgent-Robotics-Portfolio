"""Dependency-light helpers shared by reproducible experiment runners."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence


CHECKPOINT_MODES = ("reuse", "continue", "fresh")


def hybrid_summary_path(result_root: str | Path, dataset_name: str) -> Path:
    """Return the canonical summary location for one hybrid experiment."""

    return Path(result_root) / str(dataset_name) / f"{dataset_name}_hybrid_summary.json"


def hybrid_checkpoint_path(result_root: str | Path, dataset_name: str) -> Path:
    """Return the canonical checkpoint location for one hybrid experiment."""

    return Path(result_root) / str(dataset_name) / f"{dataset_name}_hybrid_fraudgraph.pt"


def load_summary_payload(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON summary object, accepting either UTF-8 or UTF-8 with BOM."""

    summary_path = Path(path)
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else None


def load_hybrid_summary(path: str | Path) -> dict[str, Any] | None:
    """Extract a hybrid summary from a summary payload when it is available."""

    payload = load_summary_payload(path)
    if payload is None:
        return None
    summary = payload.get("summary", payload)
    return summary if isinstance(summary, dict) else None


def resolve_checkpoint_mode(force_rerun: bool, checkpoint_mode: str) -> str:
    """Resolve forced reruns and reject unknown modes by falling back to reuse."""

    if force_rerun:
        return "fresh"
    normalized = str(checkpoint_mode).strip().lower()
    return normalized if normalized in CHECKPOINT_MODES else "reuse"


def resolve_seeds(
    *,
    deprecated_seed: int,
    seeds: Sequence[int],
    default_seeds: Sequence[int] = (30, 31, 40),
) -> list[int]:
    """Preserve the legacy single-seed override while returning an ordered unique seed list."""

    if int(deprecated_seed) >= 0:
        return [int(deprecated_seed)]
    resolved: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        value = int(seed)
        if value not in seen:
            seen.add(value)
            resolved.append(value)
    return resolved or [int(seed) for seed in default_seeds]


def dedupe_float_values(values: Iterable[float]) -> list[float]:
    """Return ordered, numerically stable float values without duplicate CLI inputs."""

    resolved: list[float] = []
    seen: set[str] = set()
    for item in values:
        value = float(item)
        key = format(value, ".12g")
        if key not in seen:
            seen.add(key)
            resolved.append(value)
    return resolved


def label_fraction_slug(label_fraction: float) -> str:
    """Format a label fraction as a stable filesystem-safe path component."""

    normalized = format(float(label_fraction), ".12g").rstrip("0").rstrip(".")
    return (normalized or "0").replace("-", "m").replace(".", "p")


def aggregate_metric(values: Sequence[float]) -> dict[str, float]:
    """Summarize a metric across seeds without requiring ML runtime dependencies."""

    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    normalized = [float(value) for value in values]
    return {
        "mean": float(mean(normalized)),
        "std": float(pstdev(normalized)) if len(normalized) > 1 else 0.0,
        "min": float(min(normalized)),
        "max": float(max(normalized)),
    }


def mean_std_metric(values: Sequence[float]) -> dict[str, float]:
    """Return the mean and population standard deviation used in protocol tables."""

    aggregate = aggregate_metric(values)
    return {"mean": aggregate["mean"], "std": aggregate["std"]}
