"""Deterministic, finite-only serialization helpers.

The core semantics are derived from the locally owned md_qd_swarm
``method/io_contract.py`` snapshot recorded in ``manifests/reuse_manifest.json``.
This module is a clean rewrite with recursive normalization and safe cleanup.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def require_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def finite_number(value: int | float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric and not boolean")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _primitive(value: Any, path: str = "$") -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _primitive(value.to_dict(), path)
    if dataclasses.is_dataclass(value):
        return _primitive(dataclasses.asdict(value), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} object keys must be non-empty strings")
            normalized[key] = _primitive(child, f"{path}.{key}")
        return normalized
    if isinstance(value, (set, frozenset)):
        normalized = [_primitive(child, f"{path}[]") for child in value]
        return sorted(normalized, key=lambda item: canonical_json_bytes(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_primitive(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def to_primitive(value: Any) -> Any:
    """Return a JSON-safe snapshot while rejecting non-finite or ambiguous values."""

    return _primitive(value)


def canonical_json_bytes(payload: Any) -> bytes:
    encoded = json.dumps(
        to_primitive(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (encoded + "\n").encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_finite_diagnostics(
    payload: Any, *, label: str = "public diagnostics"
) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    diagnostics: dict[str, float] = {}
    for key, value in payload.items():
        require_identifier(key, f"{label} key")
        diagnostics[key] = finite_number(value, f"{label}.{key}")
    return diagnostics


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    encoded = (
        json.dumps(
            to_primitive(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    to_primitive(payload)
    return payload
