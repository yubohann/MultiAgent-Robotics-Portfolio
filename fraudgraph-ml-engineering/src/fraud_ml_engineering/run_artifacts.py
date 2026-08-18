from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RunArtifacts:
    run_root: Path
    manifest_path: Path
    summary_path: Path
    report_path: Path


def _timestamp_token() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)
    return f"{timestamp}-{milliseconds:03d}"


def _normalize_root(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _build_run_dir_name(prefix: str | None = None) -> str:
    token = _timestamp_token()
    if prefix:
        return f"{prefix}_{token}"
    return token


def create_run_artifacts(base_root: str | Path, *, prefix: str | None = None) -> RunArtifacts:
    root = _normalize_root(base_root)
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / _build_run_dir_name(prefix)
    while run_root.exists():
        run_root = root / _build_run_dir_name(prefix)
    run_root.mkdir(parents=True, exist_ok=False)
    return RunArtifacts(
        run_root=run_root,
        manifest_path=run_root / "manifest.json",
        summary_path=run_root / "summary.json",
        report_path=run_root / "report.md",
    )


def resume_run_artifacts(run_root: str | Path) -> RunArtifacts:
    root = _normalize_root(run_root)
    if not root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    return RunArtifacts(
        run_root=root,
        manifest_path=root / "manifest.json",
        summary_path=root / "summary.json",
        report_path=root / "report.md",
    )


def _repository_commit(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_run_manifest(
    *,
    dataset: str,
    seed: int | None,
    command: Sequence[str] | None = None,
    config_path: str | Path | None = None,
    notes: str = "",
    extra: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance record without requiring the training stack."""

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "recorded_at": timestamp,
        "repository_commit": _repository_commit(repo_root),
        "runtime": {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "experiment": {
            "dataset": str(dataset),
            "seed": int(seed) if seed is not None else None,
            "command": list(command or ()),
            "config_path": str(config_path) if config_path else "",
            "notes": str(notes),
            "extra": dict(extra or {}),
        },
    }


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON atomically so interrupted provenance capture never leaves a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(destination)
