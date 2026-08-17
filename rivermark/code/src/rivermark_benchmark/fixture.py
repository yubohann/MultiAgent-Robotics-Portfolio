"""Generate a tiny, explicit CPU-only pilot fixture.

The fixture is a derived loader smoke sample.  It is intentionally generated
from the existing kinematic pilot path, never enters ``rivermark/``, and cannot
be described as native Isaac data or a formal benchmark episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .dataset import collect_episode, load_pilot_episode
from .provenance import detect_source_provenance
from .schema import is_safe_relative_path


FIXTURE_SCHEMA = "org.rivermark.benchmark.cpu-fixture.v1"
FIXTURE_BACKEND = "rivermark-kinematic-pilot-v1"
CLAIM_BOUNDARY = "cpu_loader_smoke_only"
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


class FixtureError(ValueError):
    """Raised when a fixture cannot be created without risking existing data."""


@dataclass(frozen=True)
class CpuFixture:
    root: Path
    manifest_path: Path
    fixture_manifest_path: Path
    fixture_id: str
    episode_manifest_sha256: str
    frame_count: int
    agent_count: int


@dataclass(frozen=True)
class CpuFixtureVerification:
    valid: bool
    fixture_manifest_path: Path
    episode_manifest_sha256: str | None
    frame_count: int | None
    agent_count: int | None
    issues: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_cpu_fixture(
    output_root: Path,
    *,
    fixture_id: str = "cpu-fixture-0001",
    agent_count: int = 2,
    max_steps: int = 4,
    seed: int = 0,
) -> CpuFixture:
    """Create one small pilot episode and a binding fixture receipt.

    ``output_root`` must be absent or empty.  Existing files are never removed
    or overwritten, including failed or partially generated fixture artifacts.
    """

    if not isinstance(fixture_id, str) or not _ID.fullmatch(fixture_id):
        raise FixtureError("fixture_id must be a lowercase safe identifier")
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count < 1:
        raise FixtureError("agent_count must be a positive integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 2:
        raise FixtureError("max_steps must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FixtureError("seed must be a non-negative integer")
    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FixtureError(f"refusing to write into a non-empty fixture directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    episode_id = fixture_id
    manifest_path = collect_episode(
        root,
        episode_id=episode_id,
        teacher_method="action_chunk_vla_pilot",
        agent_count=agent_count,
        max_steps=max_steps,
        seed=seed,
    )
    episode = load_pilot_episode(manifest_path)
    manifest_hash = _sha256_file(manifest_path)
    provenance = detect_source_provenance()
    fixture_manifest_path = root / "fixture_manifest.json"
    _write_json(
        fixture_manifest_path,
        {
            "schema": FIXTURE_SCHEMA,
            "fixture_id": fixture_id,
            "dataset_version": "0.1.0-pilot",
            "backend": FIXTURE_BACKEND,
            "derived_sample": True,
            "formal_benchmark_admission": False,
            "episode_manifest": manifest_path.relative_to(root).as_posix(),
            "episode_manifest_sha256": manifest_hash,
            "agent_count": episode.agent_count,
            "frame_count": episode.frame_count,
            "seed": seed,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_revision": provenance.source_revision,
            "source_tree_sha256": provenance.source_tree_sha256,
            "source_worktree_dirty": provenance.source_worktree_dirty,
        },
    )
    return CpuFixture(root, manifest_path, fixture_manifest_path, fixture_id, manifest_hash, episode.frame_count, episode.agent_count)


def verify_cpu_fixture(fixture_manifest_path: Path) -> CpuFixtureVerification:
    """Independently verify one generated fixture and its pilot episode."""

    path = fixture_manifest_path.resolve()
    issues: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return CpuFixtureVerification(False, path, None, None, None, (f"fixture_manifest:{exc}",))
    if not isinstance(payload, dict):
        return CpuFixtureVerification(False, path, None, None, None, ("fixture_manifest:not_an_object",))
    allowed = {
        "schema", "fixture_id", "dataset_version", "backend",
        "derived_sample", "formal_benchmark_admission", "episode_manifest",
        "episode_manifest_sha256", "agent_count", "frame_count", "seed",
        "claim_boundary", "source_revision", "source_tree_sha256",
        "source_worktree_dirty",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        issues.append(f"unknown_fields:{','.join(unknown)}")
    constants = {
        "schema": FIXTURE_SCHEMA,
        "dataset_version": "0.1.0-pilot",
        "backend": FIXTURE_BACKEND,
        "derived_sample": True,
        "formal_benchmark_admission": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for key, expected in constants.items():
        if payload.get(key) != expected:
            issues.append(f"{key}:expected_{expected!r}")
    fixture_id = payload.get("fixture_id")
    if not isinstance(fixture_id, str) or not _ID.fullmatch(fixture_id):
        issues.append("fixture_id:invalid")
    for key in ("episode_manifest_sha256", "source_tree_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            issues.append(f"{key}:invalid")
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{7,64}", revision):
        issues.append("source_revision:invalid")
    if not isinstance(payload.get("source_worktree_dirty"), bool):
        issues.append("source_worktree_dirty:invalid")
    for key, minimum in (("agent_count", 1), ("frame_count", 1), ("seed", 0)):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            issues.append(f"{key}:invalid")
    relative = payload.get("episode_manifest")
    episode_hash: str | None = None
    frame_count: int | None = None
    agent_count: int | None = None
    if not is_safe_relative_path(relative):
        issues.append("episode_manifest:unsafe")
    else:
        episode_path = (path.parent / str(relative)).resolve()
        if not episode_path.is_relative_to(path.parent.resolve()) or not episode_path.is_file():
            issues.append("episode_manifest:missing")
        else:
            episode_hash = _sha256_file(episode_path)
            if episode_hash != payload.get("episode_manifest_sha256"):
                issues.append("episode_manifest_sha256:mismatch")
            try:
                episode = load_pilot_episode(episode_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                issues.append(f"episode:{exc}")
            else:
                frame_count = episode.frame_count
                agent_count = episode.agent_count
                if frame_count != payload.get("frame_count"):
                    issues.append("frame_count:mismatch")
                if agent_count != payload.get("agent_count"):
                    issues.append("agent_count:mismatch")
                if episode.manifest.get("split") != "pilot":
                    issues.append("episode_split:not_pilot")
    return CpuFixtureVerification(not issues, path, episode_hash, frame_count, agent_count, tuple(issues))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a tiny CPU fixture")
    create.add_argument("output_root", type=Path)
    create.add_argument("--fixture-id", default="cpu-fixture-0001")
    create.add_argument("--agents", type=int, default=2)
    create.add_argument("--max-steps", type=int, default=4)
    create.add_argument("--seed", type=int, default=0)
    verify = commands.add_parser("verify", help="verify a fixture manifest and payloads")
    verify.add_argument("fixture_manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "verify":
        result = verify_cpu_fixture(args.fixture_manifest)
        print(json.dumps({
            "schema": FIXTURE_SCHEMA,
            "status": "valid" if result.valid else "invalid",
            "fixture_manifest": str(result.fixture_manifest_path),
            "episode_manifest_sha256": result.episode_manifest_sha256,
            "frame_count": result.frame_count,
            "agent_count": result.agent_count,
            "issues": list(result.issues),
            "formal_benchmark_admission": False,
        }, indent=2, sort_keys=True))
        return 0 if result.valid else 1
    try:
        result = create_cpu_fixture(
            args.output_root,
            fixture_id=args.fixture_id,
            agent_count=args.agents,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    except (OSError, FixtureError, ValueError) as exc:
        print(json.dumps({"schema": FIXTURE_SCHEMA, "status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({
        "schema": FIXTURE_SCHEMA,
        "status": "created",
        "root": str(result.root),
        "fixture_manifest": str(result.fixture_manifest_path),
        "episode_manifest_sha256": result.episode_manifest_sha256,
        "frame_count": result.frame_count,
        "agent_count": result.agent_count,
        "formal_benchmark_admission": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
