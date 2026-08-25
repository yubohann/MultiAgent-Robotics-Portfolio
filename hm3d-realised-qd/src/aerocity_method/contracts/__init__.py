"""Versioned public contracts and serialization helpers."""

from aerocity_method.contracts.io import canonical_sha256, write_json_atomic
from aerocity_method.runtime.range_sensing import (
    DENSE_26_RAY_PATTERN,
    LEGACY_SIX_AXIS_PATTERN,
    public_range_direction_count,
    resolve_public_range_directions,
    validate_public_range_directions,
)
from aerocity_method.contracts.hm3d_public_schema import (
    PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    public_schema_fields,
    require_current_public_schema,
)
from aerocity_method.contracts.models import ABI_VERSION

# The formal paper contract is intentionally single-scale.  Outcomes retain
# this value for provenance, but no runtime entry point accepts another fleet.
FORMAL_FLEET_SIZE = 4

__all__ = [
    "ABI_VERSION",
    "DENSE_26_RAY_PATTERN",
    "FORMAL_FLEET_SIZE",
    "LEGACY_SIX_AXIS_PATTERN",
    "PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION",
    "PUBLIC_TASK_RESERVATION_SCHEMA_VERSION",
    "canonical_sha256",
    "public_range_direction_count",
    "public_schema_fields",
    "require_current_public_schema",
    "resolve_public_range_directions",
    "validate_public_range_directions",
    "write_json_atomic",
]
