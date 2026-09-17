"""Shared graph-cache helpers used by the dataset adapters."""

from __future__ import annotations

import json
import os
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import dgl
except Exception as error:  # pragma: no cover - runtime env dependent
    raise RuntimeError(
        "Missing dgl dependency. Install the pinned graph-learning profile and retry:"
        "\npython -m pip install -r requirements/requirements-cpu.txt"
    ) from error


def cache_signature_tag(signature: dict[str, Any]) -> str:
    """Return the stable hex tag embedded in cache filenames."""

    digest = zlib.crc32(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")) & 0xFFFFFFFF
    return f"{digest:08x}"


def resolve_graph_cache_paths(
    signature: dict[str, Any],
    *,
    prefix: str,
    cache_dir: str | Path,
    default_signature: dict[str, Any] | None = None,
    default_graph_path: str | Path | None = None,
    default_metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve graph and metadata cache paths for one cache signature."""

    if default_signature is not None and signature == default_signature:
        return Path(default_graph_path), Path(default_metadata_path)
    tag = cache_signature_tag(signature)
    resolved_dir = Path(cache_dir)
    return resolved_dir / f"{prefix}_{tag}.dgl", resolved_dir / f"{prefix}_{tag}.json"


def remove_cached_file(path: Path) -> None:
    """Delete a cache file when present, ignoring filesystem errors."""

    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def load_graph_cache(
    *,
    signature: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
    tolerate_corruption: bool = False,
    log: Callable[[str], None] | None = None,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
    """Load a cached graph and metadata when the stored signature matches."""

    if not graph_path.exists() or not metadata_path.exists():
        if log is not None:
            log("cache_read: miss_missing_files")
        return None

    def _read_payload() -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if log is not None:
            log("cache_read: metadata_loaded")
        if dict(metadata.get("cache_signature", {})) != signature:
            if log is not None:
                log("cache_read: miss_signature_mismatch")
            return None
        graph = dgl.load_graphs(str(graph_path))[0][0]
        return graph, metadata

    if not tolerate_corruption:
        return _read_payload()
    try:
        return _read_payload()
    except (OSError, TypeError, ValueError, dgl.DGLError):
        remove_cached_file(metadata_path)
        remove_cached_file(graph_path)
        return None


def store_graph_file(graph_path: Path, graph: dgl.DGLHeteroGraph) -> None:
    """Write one DGL graph to its cache location."""

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(graph_path), [graph])


def store_metadata_file(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Write cache metadata as UTF-8-with-BOM JSON."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def store_graph_cache(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
    atomic: bool = False,
) -> None:
    """Persist a graph cache, optionally through an atomic graph-file replace."""

    if not atomic:
        store_graph_file(graph_path, graph)
        store_metadata_file(metadata_path, metadata)
        return

    from .checkpointing import atomic_write_json

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_token = f"{os.getpid()}.{time.time_ns()}"
    temp_graph_path = graph_path.with_suffix(graph_path.suffix + f".{temp_token}.tmp")
    try:
        dgl.save_graphs(str(temp_graph_path), [graph])
        os.replace(temp_graph_path, graph_path)
        atomic_write_json(metadata_path, metadata)
    finally:
        remove_cached_file(temp_graph_path)


def lock_file_for_metadata(metadata_path: Path) -> Path:
    """Return the conventional lock path that guards one cache metadata file."""

    return metadata_path.with_suffix(metadata_path.suffix + ".lock")


@contextmanager
def exclusive_cache_write_lock(
    lock_path: Path,
    *,
    timeout_seconds: float,
    stale_seconds: float,
    poll_seconds: float,
    timeout_message: str,
) -> Iterator[None]:
    """Serialize cache builds through an exclusive lock file."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "pid": int(os.getpid()),
                        "created_at": float(time.time()),
                    },
                    file,
                    ensure_ascii=False,
                )
            break
        except FileExistsError:
            try:
                age_seconds = time.time() - float(lock_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if age_seconds >= stale_seconds:
                remove_cached_file(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(timeout_message)
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        remove_cached_file(lock_path)
