"""Deterministic source snapshot manifests for dirty-worktree reproducibility."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import ABI_VERSION


def build_source_snapshot(
    root: str | Path,
    relative_paths: Sequence[str | Path],
    *,
    canaries: Sequence[str] = (),
) -> dict[str, Any]:
    resolved_root = Path(root).resolve()
    rows: dict[str, dict[str, int | str]] = {}
    for relative in sorted({Path(path).as_posix() for path in relative_paths}):
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"snapshot path escapes root: {relative}") from exc
        if not candidate.is_file():
            raise ValueError(f"snapshot path is not a file: {relative}")
        content = candidate.read_bytes()
        for canary in canaries:
            if canary and canary.encode("utf-8") in content:
                raise ValueError(f"private canary found in snapshot file: {relative}")
        rows[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if not rows:
        raise ValueError("source snapshot requires files")
    payload = {"schema_version": ABI_VERSION, "files": rows}
    payload["snapshot_hash"] = canonical_sha256(payload)
    return payload


def verify_source_snapshot(
    root: str | Path,
    snapshot: dict[str, Any],
    *,
    canaries: Sequence[str] = (),
) -> tuple[str, ...]:
    if snapshot.get("schema_version") != ABI_VERSION:
        raise ValueError("snapshot schema version mismatch")
    expected_hash = snapshot.get("snapshot_hash")
    unsigned = {key: value for key, value in snapshot.items() if key != "snapshot_hash"}
    if canonical_sha256(unsigned) != expected_hash:
        raise ValueError("snapshot manifest hash mismatch")
    rebuilt = build_source_snapshot(root, tuple(snapshot["files"]), canaries=canaries)
    mismatches = [
        path
        for path, expected in snapshot["files"].items()
        if rebuilt["files"].get(path) != expected
    ]
    return tuple(sorted(mismatches))
