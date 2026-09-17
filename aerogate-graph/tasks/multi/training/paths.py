"""Training helpers for the multi-agent 2D gate experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from tasks.multi.env.multi_gate_env import MultiGate2DEnv
from tasks.multi.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv

MultiResumeMode = Literal["reset_train_state", "keep_optimizer_state"]
MultiEnvType = MultiGate2DEnv | MultiGateKinematic3DEnv

def _resolve_output_dirs(
    *,
    save_dir: str | Path | None,
    log_dir: str | Path | None,
    checkpoint_dir: str | Path | None,
) -> tuple[Path | None, Path | None]:
    if log_dir is not None or checkpoint_dir is not None:
        resolved_log_dir = Path(log_dir) if log_dir is not None else None
        resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if resolved_log_dir is None or resolved_checkpoint_dir is None:
            raise ValueError("log_dir and checkpoint_dir must be provided together")
        return resolved_log_dir, resolved_checkpoint_dir
    if save_dir is not None:
        base_dir = Path(save_dir)
        return base_dir / "logs", base_dir / "checkpoints"
    return None, None

def _resolve_review_interval(
    *,
    requested_interval: int | None,
    fallback_interval: int,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    if requested_interval is not None:
        resolved = max(int(requested_interval), 0)
        return resolved if resolved > 0 else None
    fallback = max(int(fallback_interval), 0)
    return fallback if fallback > 0 else None

def _run_label(log_dir: Path | None, checkpoint_dir: Path | None) -> str:
    if log_dir is not None and log_dir.name != "logs":
        return log_dir.name
    if checkpoint_dir is not None and checkpoint_dir.name != "checkpoints":
        return checkpoint_dir.name
    if log_dir is not None:
        return log_dir.parent.name or "multi_run"
    if checkpoint_dir is not None:
        return checkpoint_dir.parent.name or "multi_run"
    return "multi_run"