"""Canonical JSON I/O and deterministic seed derivation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def derived_seed(*parts: object) -> int:
    # Deterministic across runs; independent of Python's string randomization.
    text = json.dumps(list(parts), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    seed = 17
    for character in text:
        seed = (seed * 31 + ord(character)) & 0xFFFFFFFFFFFFFFFF
    return seed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_compact(path: Path, value: Any) -> None:
    # Single-line output for large report trees.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))
