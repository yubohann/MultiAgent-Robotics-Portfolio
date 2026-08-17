"""Run the bounded CPU researcher-entry smoke test.

This command is the first five-minute path for a fresh checkout. It exercises
the public fixture, loader, and evaluator contracts without Isaac Sim, Torch,
GPU access, private evaluator truth, or a formal dataset episode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
import tracemalloc
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import load_pilot_episode
from .evaluator import SUBMISSION_SCHEMA, evaluate_submission
from .fixture import create_cpu_fixture, verify_cpu_fixture
from .metrics import METRIC_VERSION
from .provenance import detect_source_provenance


RESEARCHER_SMOKE_SCHEMA = "org.rivermark.benchmark.researcher-smoke.v1"
_SHA256 = "0" * 64


class ResearcherEntryError(ValueError):
    """Raised when the bounded researcher smoke cannot complete safely."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _public_smoke_submission(source_revision: str) -> dict[str, Any]:
    """Build a public-only evaluator request with no target or reward fields."""

    return {
        "schema": SUBMISSION_SCHEMA,
        "dataset_version": "0.1.0",
        "dataset_index_sha256": _SHA256,
        "split": "validation",
        "evaluator": {
            "evaluator_id": "public-search3d",
            "evaluator_version": "1.0.0",
            "evaluator_sha256": _SHA256,
            "metric_schema": METRIC_VERSION,
        },
        "policy": {
            "method_id": "researcher-entry-smoke",
            "code_revision": source_revision,
            "checkpoint_sha256": _SHA256,
            "seed": 7,
        },
        "episodes": [
            {
                "episode_id": "validation-smoke-001",
                "split": "validation",
                "trace": {
                    "timestamps_s": [0.0, 1.0, 2.0],
                    "confirmed_counts": [0, 1, 2],
                    "target_count": 2,
                    "time_budget_s": 2.0,
                    "false_confirmations": 0,
                    "truncated": False,
                },
            }
        ],
    }


def run_researcher_smoke(output_root: Path) -> dict[str, Any]:
    """Create, load, and evaluate the bounded CPU researcher entry path."""

    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ResearcherEntryError(f"refusing to write into a non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tracemalloc.start()
    try:
        fixture = create_cpu_fixture(
            root / "fixture",
            fixture_id="researcher-entry-smoke-001",
            agent_count=2,
            max_steps=4,
            seed=7,
        )
        verification = verify_cpu_fixture(fixture.fixture_manifest_path)
        if not verification.valid:
            raise ResearcherEntryError("fixture verification failed: " + "; ".join(verification.issues))
        episode = load_pilot_episode(fixture.manifest_path)
        if episode.frame_count < 2 or episode.agent_count != 2:
            raise ResearcherEntryError("fixture loader returned an unexpected shape")
        if episode.manifest.get("split") != "pilot":
            raise ResearcherEntryError("researcher smoke must use the pilot split")
        # Exercise the bounded selective projection in the first-run contract:
        # camera payloads must remain unopened when state/action is requested.
        selective = load_pilot_episode(
            fixture.manifest_path,
            modalities=("state", "action"),
            agent_ids=(1,),
        )
        if (
            selective.agent_ids != (1,)
            or selective.states is None
            or selective.states.shape != (episode.frame_count, 1, 8)
            or selective.actions is None
            or selective.actions.shape != (episode.frame_count, 1, 4)
            or selective.rgb is not None
            or selective.depth is not None
        ):
            raise ResearcherEntryError("selective loader projection returned an unexpected shape or payload")
        provenance = detect_source_provenance()
        revision = provenance.source_revision
        if not isinstance(revision, str) or len(revision) < 7:
            revision = "0" * 40
        report = evaluate_submission(_public_smoke_submission(revision))
        if not report.valid:
            raise ResearcherEntryError("public evaluator smoke failed: " + "; ".join(issue.code for issue in report.issues))
        current, peak = tracemalloc.get_traced_memory()
        payload = {
            "schema": RESEARCHER_SMOKE_SCHEMA,
            "status": "passed",
            "claim_boundary": "cpu_loader_and_public_metric_smoke_only",
            "formal_benchmark_admission": False,
            "fixture": {
                "manifest": "fixture/fixture_manifest.json",
                "episode_manifest": "fixture/researcher-entry-smoke-001/episode_manifest.json",
                "frame_count": episode.frame_count,
                "agent_count": episode.agent_count,
                "sample_count": episode.sample_count,
                "episode_manifest_sha256": fixture.episode_manifest_sha256,
            },
            "checks": {
                "fixture_manifest": "passed",
                "payload_hashes": "passed",
                "loader_shapes": "passed",
                "selective_loader": "passed",
                "public_evaluator": "passed",
                "private_truth_present": False,
                "isaac_started": False,
            },
            "evaluation": {
                "evaluator_id": report.evaluator_id,
                "evaluator_version": report.evaluator_version,
                "metric_version": report.metric_version,
                "episode_count": report.episode_count,
                "scores": [score.__dict__ for score in report.scores],
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "optional_modules": {
                    name: _module_available(name)
                    for name in ("torch", "isaacsim", "omni.isaac.kit", "pyarrow")
                },
                "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "peak_python_allocated_bytes": int(peak),
                "current_python_allocated_bytes": int(current),
                "artifact_bytes_before_report": _artifact_bytes(root),
                "resource_measurement": "tracemalloc_python_allocations_and_output_bytes_only",
            },
            "source": {
                "revision": revision,
                "worktree_dirty": provenance.source_worktree_dirty,
            },
        }
        _write_json(root / "researcher_smoke_report.json", payload)
        return payload
    finally:
        tracemalloc.stop()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="new or empty directory for the smoke artifacts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_researcher_smoke(args.output_root)
    except (OSError, ResearcherEntryError, ValueError) as exc:
        print(json.dumps({"schema": RESEARCHER_SMOKE_SCHEMA, "status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
