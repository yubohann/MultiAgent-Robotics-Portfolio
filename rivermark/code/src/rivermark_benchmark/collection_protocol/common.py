"""Shared issue types and validation helpers for collection protocols."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .constants import _PRIVATE_TOKENS


@dataclass(frozen=True)
class CollectionProtocolIssue:
    code: str
    path: str
    message: str


class CollectionProtocolError(ValueError):
    """Raised when a collection protocol or public ledger is unsafe."""


def _issue(issues: list[CollectionProtocolIssue], code: str, path: str, message: str) -> None:
    issues.append(CollectionProtocolIssue(code, path, message))


def _unknown_keys(value: Mapping[Any, Any], allowed: frozenset[str]) -> tuple[Any, ...]:
    return tuple(sorted((key for key in value if not isinstance(key, str) or key not in allowed), key=str))

def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")

def protocol_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

def _contains_private_token(value: Any) -> bool:
    return isinstance(value, str) and any(token in value.lower() for token in _PRIVATE_TOKENS)

def _valid_number(value: Any, *, minimum: float, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    if float(value) <= minimum:
        return False
    return maximum is None or float(value) < maximum
