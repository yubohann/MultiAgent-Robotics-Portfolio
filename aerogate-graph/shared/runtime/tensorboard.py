"""Small TensorBoard helpers for aerogate_graph training runs."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
import math
import re
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None  # type: ignore[assignment]


def create_summary_writer(log_dir: str | Path) -> tuple[Any | None, Path]:
    """Create a TensorBoard writer inside the run log directory."""

    tensorboard_dir = resolve_tensorboard_dir(log_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    if SummaryWriter is None:
        return None, tensorboard_dir
    return SummaryWriter(log_dir=str(tensorboard_dir)), tensorboard_dir


def resolve_tensorboard_dir(log_dir: str | Path) -> Path:
    """Resolve the canonical TensorBoard sub-directory for one run."""

    return Path(log_dir) / "tensorboard"


def close_summary_writer(writer: Any | None) -> None:
    """Flush and close a TensorBoard writer if it exists."""

    if writer is None:
        return
    writer.flush()
    writer.close()


def log_scalar(writer: Any | None, tag: str, value: Any, step: int) -> None:
    """Log one scalar if TensorBoard is enabled and the value is finite."""

    if writer is None:
        return
    resolved = _coerce_scalar(value)
    if resolved is None:
        return
    writer.add_scalar(_sanitize_tag(tag), resolved, int(step))


def log_scalars(writer: Any | None, prefix: str, values: dict[str, Any], step: int) -> None:
    """Log a flat scalar dictionary under one prefix."""

    if writer is None:
        return
    for key, value in values.items():
        log_scalar(writer, f"{prefix}/{key}", value, step)


def event_file_paths(tensorboard_dir: str | Path) -> list[str]:
    """Return TensorBoard event files inside a run directory."""

    return sorted(str(path) for path in Path(tensorboard_dir).glob("events.out.tfevents.*"))


def _coerce_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        resolved = float(value)
        if math.isfinite(resolved):
            return resolved
        return None
    return None


def _sanitize_tag(tag: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_./-]+", "_", str(tag).strip())
    sanitized = re.sub(r"/+", "/", sanitized).strip("/")
    return sanitized or "scalar"

