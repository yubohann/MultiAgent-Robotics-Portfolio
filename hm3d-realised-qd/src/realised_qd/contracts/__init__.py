"""Versioned public contracts and serialization helpers."""

from realised_qd.contracts.hm3d_public_schema import (
    PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    public_schema_fields,
    require_current_public_schema,
)
from realised_qd.contracts.io import write_json_atomic
from realised_qd.contracts.models import ABI_VERSION
from realised_qd.runtime.range_sensing import (
    DENSE_26_RAY_PATTERN,
    LEGACY_SIX_AXIS_PATTERN,
    public_range_direction_count,
    resolve_public_range_directions,
    validate_public_range_directions,
)

# The formal comparison keeps one fleet size so every method faces the same team.
FORMAL_FLEET_SIZE = 4

__all__ = [
    "ABI_VERSION",
    "DENSE_26_RAY_PATTERN",
    "FORMAL_FLEET_SIZE",
    "LEGACY_SIX_AXIS_PATTERN",
    "PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION",
    "PUBLIC_TASK_RESERVATION_SCHEMA_VERSION",
    "public_range_direction_count",
    "public_schema_fields",
    "require_current_public_schema",
    "resolve_public_range_directions",
    "validate_public_range_directions",
    "write_json_atomic",
]
