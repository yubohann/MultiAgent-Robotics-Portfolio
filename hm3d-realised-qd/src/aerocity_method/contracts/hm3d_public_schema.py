"""Versioned public HM3D candidate/task contracts.

These values are part of the outcome identity, not documentation labels.  A
candidate-pool hash alone cannot distinguish an older producer that happened
to emit the same shaped rows from the current persistent-task semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION = "hm3d-public-candidate-pool-v7"
PUBLIC_TASK_RESERVATION_SCHEMA_VERSION = "hm3d-public-task-reservation-v1"


def require_current_public_schema(
    payload: Mapping[str, Any], *, context: str = "payload"
) -> None:
    """Reject records produced before the persistent public-task contract."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    if payload.get("candidate_pool_schema_version") != PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION:
        raise ValueError(
            f"{context} requires candidate_pool_schema_version="
            f"{PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION}"
        )
    if payload.get("task_reservation_schema_version") != PUBLIC_TASK_RESERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"{context} requires task_reservation_schema_version="
            f"{PUBLIC_TASK_RESERVATION_SCHEMA_VERSION}"
        )


def public_schema_fields() -> dict[str, str]:
    """Return the immutable schema fields to copy into a outcome/transition."""

    return {
        "candidate_pool_schema_version": PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
        "task_reservation_schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    }


__all__ = [
    "PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION",
    "PUBLIC_TASK_RESERVATION_SCHEMA_VERSION",
    "public_schema_fields",
    "require_current_public_schema",
]
