"""Portable discovery of an optional sibling IsaacLab checkout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IsaacLabPaths:
    """Paths needed by Isaac-only tools, or ``None`` outside an Isaac checkout."""

    isaaclab_root: Path | None
    drone_project_root: Path | None
    source_root: Path | None


def discover_isaaclab_paths(bench_root: Path) -> IsaacLabPaths:
    """Find IsaacLab by structure, without assuming repository depth.

    ``AEROCITY_ISAACLAB_ROOT`` is an explicit override for CI or a detached
    checkout.  A candidate is accepted only when its IsaacLab source tree is
    present, so a clean benchmark checkout remains importable without Isaac.
    """

    candidates: list[Path] = []
    override = os.environ.get("AEROCITY_ISAACLAB_ROOT")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(bench_root.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        source_root = candidate / "source"
        if not (source_root / "isaaclab").is_dir():
            continue
        drone_project_root = candidate / "isaac_drone_racer"
        if not drone_project_root.is_dir():
            drone_project_root = None
        return IsaacLabPaths(candidate, drone_project_root, source_root)

    return IsaacLabPaths(None, None, None)
