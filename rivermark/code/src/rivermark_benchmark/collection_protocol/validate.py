"""Protocol validation dispatch and loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import CollectionProtocolError, CollectionProtocolIssue
from .constants import (
    NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    T1_COLLECTION_PROTOCOL_SCHEMA,
)
from .t1 import _validate_t1_collection_protocol
from .t2 import (
    _validate_t2_native_canary_protocol,
    _validate_t2_native_canary_v2_protocol,
    _validate_t2_native_canary_v3_protocol,
)
from .v1 import _validate_v1_collection_protocol


def validate_collection_protocol(payload: Any) -> tuple[CollectionProtocolIssue, ...]:
    """Dispatch validation without changing the immutable v1 contract."""

    if isinstance(payload, Mapping) and payload.get("schema") == NATIVE_T2_CANARY_PROTOCOL_SCHEMA:
        return _validate_t2_native_canary_protocol(payload)
    if isinstance(payload, Mapping) and payload.get("schema") == NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA:
        return _validate_t2_native_canary_v2_protocol(payload)
    if isinstance(payload, Mapping) and payload.get("schema") == NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA:
        return _validate_t2_native_canary_v3_protocol(payload)
    if isinstance(payload, Mapping) and payload.get("schema") == T1_COLLECTION_PROTOCOL_SCHEMA:
        return _validate_t1_collection_protocol(payload)
    return _validate_v1_collection_protocol(payload)


def load_collection_protocol(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionProtocolError(f"cannot read collection protocol: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CollectionProtocolError("collection protocol must be an object")
    issues = validate_collection_protocol(payload)
    if issues:
        raise CollectionProtocolError("invalid collection protocol: " + "; ".join(issue.code for issue in issues))
    return payload
