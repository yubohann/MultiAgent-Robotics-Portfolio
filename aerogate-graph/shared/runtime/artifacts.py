"""Independent runtime directories and execution policy for aerogate_graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from shared.configs.global_config import EXPERIMENT_ROOT


RUNTIME_ROOT = EXPERIMENT_ROOT / "runtime"
LOGS_ROOT = RUNTIME_ROOT / "logs"
CHECKPOINTS_ROOT = RUNTIME_ROOT / "checkpoints"
REPLAYS_ROOT = RUNTIME_ROOT / "replays"
DATASETS_ROOT = RUNTIME_ROOT / "datasets"


@dataclass(frozen=True)
class RuntimeExecutionPolicy:
    """Hard runtime policy for the isolated experiment line."""

    connect_legacy_guardian: bool = False
    automatic_config_mutation: bool = False
    automatic_restart: bool = False
    automatic_source_patch: bool = False
    notes: str = (
        "aerogate_graph is an isolated experiment line. It does not attach to legacy "
        "guardian/watchdog loops, does not auto-mutate configs, and does not auto-restart runs."
    )


@dataclass(frozen=True)
class TrainingArtifacts:
    """Filesystem layout for one training run."""

    run_name: str
    track: str
    log_dir: Path
    checkpoint_dir: Path
    policy_manifest_path: Path


@dataclass(frozen=True)
class ReplayArtifacts:
    """Filesystem layout for one replay or smoke run."""

    run_name: str
    track: str
    output_dir: Path
    policy_manifest_path: Path


@dataclass(frozen=True)
class DatasetArtifacts:
    """Filesystem layout for one collected expert dataset or BC run."""

    run_name: str
    track: str
    output_dir: Path
    policy_manifest_path: Path


RUNTIME_POLICY = RuntimeExecutionPolicy()


def ensure_runtime_gate_post() -> None:
    """Create the independent runtime directory gate_post."""

    for path in (LOGS_ROOT, CHECKPOINTS_ROOT, REPLAYS_ROOT, DATASETS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def allocate_training_artifacts(track: str, run_name: str | None = None) -> TrainingArtifacts:
    """Allocate log and checkpoint directories for one training run."""

    ensure_runtime_gate_post()
    resolved_run_name = run_name or default_run_name(f"{track}_train")
    log_dir = LOGS_ROOT / track / resolved_run_name
    checkpoint_dir = CHECKPOINTS_ROOT / track / resolved_run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "runtime_policy.json"
    _write_json(manifest_path, asdict(RUNTIME_POLICY))
    return TrainingArtifacts(
        run_name=resolved_run_name,
        track=track,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        policy_manifest_path=manifest_path,
    )


def allocate_replay_artifacts(track: str, run_name: str | None = None) -> ReplayArtifacts:
    """Allocate replay output directories for one evaluation or smoke run."""

    ensure_runtime_gate_post()
    resolved_run_name = run_name or default_run_name(f"{track}_replay")
    output_dir = REPLAYS_ROOT / track / resolved_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "runtime_policy.json"
    _write_json(manifest_path, asdict(RUNTIME_POLICY))
    return ReplayArtifacts(
        run_name=resolved_run_name,
        track=track,
        output_dir=output_dir,
        policy_manifest_path=manifest_path,
    )


def allocate_dataset_artifacts(track: str, run_name: str | None = None) -> DatasetArtifacts:
    """Allocate dataset output directories for one expert-collection or BC run."""

    ensure_runtime_gate_post()
    resolved_run_name = run_name or default_run_name(f"{track}_dataset")
    output_dir = DATASETS_ROOT / track / resolved_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "runtime_policy.json"
    _write_json(manifest_path, asdict(RUNTIME_POLICY))
    return DatasetArtifacts(
        run_name=resolved_run_name,
        track=track,
        output_dir=output_dir,
        policy_manifest_path=manifest_path,
    )


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    """Write JSON with UTF-8 encoding and indentation."""

    return _write_json(Path(path), payload)


def default_run_name(prefix: str) -> str:
    """Build a timestamped default run name with sub-second uniqueness."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_suffix = uuid4().hex[:8]
    return f"{timestamp}_{unique_suffix}_{prefix}"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)
    return path

