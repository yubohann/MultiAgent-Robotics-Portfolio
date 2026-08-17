"""Episode-seed derivation, collection binding, and cell resolution."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from .common import (
    CollectionProtocolError,
    CollectionProtocolIssue,
    _contains_private_token,
    _issue,
    _unknown_keys,
    protocol_sha256,
)
from .constants import _ID, _SPLITS, COLLECTION_BINDING_KEYS
from .validate import validate_collection_protocol


def derive_episode_seed(
    *, protocol_id: str, cell_id: str, episode_seed_start: int, episode_index: int
) -> int:
    """Derive a portable uint32 seed from the frozen public identifiers.

    The SHA-256 input is four UTF-8 lines in this exact order: protocol ID,
    cell ID, decimal seed start, and decimal zero-based episode index. The
    first four digest bytes are interpreted as an unsigned big-endian integer.
    """

    for name, value in (("protocol_id", protocol_id), ("cell_id", cell_id)):
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise ValueError(f"{name} must be a public identifier")
    for name, value in (("episode_seed_start", episode_seed_start), ("episode_index", episode_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    seed_input = f"{protocol_id}\n{cell_id}\n{episode_seed_start}\n{episode_index}\n".encode()
    return int.from_bytes(hashlib.sha256(seed_input).digest()[:4], byteorder="big", signed=False)

def validate_collection_binding(value: Any) -> tuple[CollectionProtocolIssue, ...]:
    """Validate the portable binding shape without requiring the protocol file."""

    issues: list[CollectionProtocolIssue] = []
    if not isinstance(value, Mapping):
        return (CollectionProtocolIssue("type", "$", "collection binding must be an object"),)
    for key in _unknown_keys(value, COLLECTION_BINDING_KEYS):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of collection binding v1")
    for key in sorted(COLLECTION_BINDING_KEYS - set(value)):
        _issue(issues, "required", f"$.{key}", "required collection binding field is missing")
    for key in ("protocol_id", "cell_id"):
        raw = value.get(key)
        if (
            not isinstance(raw, str)
            or not _ID.fullmatch(raw)
            or _contains_private_token(raw)
        ):
            _issue(issues, key, f"$.{key}", "invalid public identifier")
    raw_hash = value.get("protocol_sha256")
    if not isinstance(raw_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        _issue(issues, "protocol_sha256", "$.protocol_sha256", "must be SHA-256")
    if value.get("split") not in _SPLITS:
        _issue(issues, "split", "$.split", "unknown formal collection split")
    episode_index = value.get("episode_index")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        _issue(issues, "episode_index", "$.episode_index", "must be a non-negative integer")
    episode_seed = value.get("episode_seed")
    if (
        isinstance(episode_seed, bool)
        or not isinstance(episode_seed, int)
        or not 0 <= episode_seed <= 0xFFFFFFFF
    ):
        _issue(issues, "episode_seed", "$.episode_seed", "must be an unsigned 32-bit integer")
    return tuple(issues)

def resolve_collection_binding(
    protocol: Mapping[str, Any], *, cell_id: str, episode_index: int
) -> dict[str, Any]:
    """Resolve one public cell/index into a hash- and seed-bound record.

    The returned object is safe to persist in a capture receipt or public
    marker: it contains no protocol path or condition values beyond the cell
    identifier and its declared split.  Keeping this derivation here ensures
    Isaac capture, packing, and coverage accounting use the same rules.
    """

    issues = validate_collection_protocol(protocol)
    if issues:
        raise CollectionProtocolError(
            "invalid collection protocol: " + "; ".join(issue.code for issue in issues)
        )
    if not isinstance(cell_id, str) or not _ID.fullmatch(cell_id):
        raise CollectionProtocolError("collection cell ID must be a public identifier")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise CollectionProtocolError("collection episode index must be a non-negative integer")
    cell = next((item for item in protocol["cells"] if item.get("cell_id") == cell_id), None)
    if not isinstance(cell, Mapping):
        raise CollectionProtocolError(f"unknown collection cell: {cell_id}")
    return {
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": protocol_sha256(protocol),
        "cell_id": cell_id,
        "split": str(cell["split"]),
        "episode_index": episode_index,
        "episode_seed": derive_episode_seed(
            protocol_id=str(protocol["protocol_id"]),
            cell_id=cell_id,
            episode_seed_start=int(protocol["randomization"]["episode_seed_start"]),
            episode_index=episode_index,
        ),
    }
