from __future__ import annotations

import pytest

from aerocity_method.contracts.hm3d_public_schema import (
    PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    public_schema_fields,
    require_current_public_schema,
)


def test_current_public_schema_is_exact_and_structured() -> None:
    payload = public_schema_fields()
    require_current_public_schema(payload)
    assert payload == {
        "candidate_pool_schema_version": PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
        "task_reservation_schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"candidate_pool_schema_version": PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION},
        {
            "candidate_pool_schema_version": "hm3d-public-candidate-pool-v5",
            "task_reservation_schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
        },
        {
            "candidate_pool_schema_version": PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
            "task_reservation_schema_version": "hm3d-public-task-reservation-legacy",
        },
    ],
)
def test_legacy_or_partial_public_schema_is_rejected(payload: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        require_current_public_schema(payload)


def test_non_mapping_public_schema_is_rejected_cleanly() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        require_current_public_schema([], context="decision[0]")  # type: ignore[arg-type]
