"""Canonical source provenance for recordings, trainers, and releases."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceProvenance:
    source_revision: str
    source_tree_sha256: str
    source_worktree_dirty: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        stderr=subprocess.DEVNULL,
        text=text,
    )


def _tracked_tree_sha256(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        encoded = relative.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        path = root / relative
        if path.is_file():
            payload = path.read_bytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        else:
            digest.update((0).to_bytes(8, "big"))
    return digest.hexdigest()


def detect_source_provenance(root: Path | None = None) -> SourceProvenance:
    """Return Git revision, tracked-tree content hash, and worktree state.

    A source tree outside Git remains usable for pilot diagnostics, but it is
    marked dirty and receives a content hash as its revision. Formal capture
    callers must reject that fallback.
    """

    resolved = (root or repository_root()).resolve()
    try:
        revision = str(_git(resolved, "rev-parse", "HEAD")).strip().lower()
        raw_paths = bytes(_git(resolved, "ls-files", "-z", text=False))
        paths = [part.decode("utf-8", errors="surrogateescape") for part in raw_paths.split(b"\0") if part]
        status = str(_git(resolved, "status", "--porcelain", "--untracked-files=normal"))
        if not revision or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("git returned a malformed revision")
        return SourceProvenance(revision, _tracked_tree_sha256(resolved, paths), bool(status.strip()))
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        package = resolved / "src" / "rivermark_benchmark"
        paths = [path.relative_to(resolved).as_posix() for path in package.glob("*.py")]
        fallback = _tracked_tree_sha256(resolved, paths)
        return SourceProvenance(fallback, fallback, True)


def source_revision(root: Path | None = None) -> str:
    return detect_source_provenance(root).source_revision


def require_clean_source(root: Path | None = None) -> SourceProvenance:
    provenance = detect_source_provenance(root)
    if provenance.source_worktree_dirty:
        raise RuntimeError("formal artifact generation requires a clean Git worktree")
    return provenance
