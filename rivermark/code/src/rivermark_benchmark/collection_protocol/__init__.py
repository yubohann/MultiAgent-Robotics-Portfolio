"""Fail-closed collection protocol, paired power gate, and coverage report."""

from .cli import main
from .common import (
    CollectionProtocolError,
    CollectionProtocolIssue,
    protocol_sha256,
)
from .constants import (
    COLLECTION_BINDING_KEYS,
    COLLECTION_PROTOCOL_SCHEMA,
    COLLECTION_SPLITS,
    COVERAGE_REPORT_SCHEMA,
    NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    POWER_METHOD,
    SEED_DERIVATION,
    T1_COLLECTION_PROTOCOL_SCHEMA,
    T1_COVERAGE_REPORT_SCHEMA,
)
from .coverage import coverage_report
from .power import required_paired_episodes
from .seeds import (
    derive_episode_seed,
    resolve_collection_binding,
    validate_collection_binding,
)
from .t1 import citylite_t1_split_certificate
from .t2 import (
    is_native_t2_canary_protocol,
    native_t2_motion_contract,
    native_t2_v2_motion_contract,
    native_t2_v3_motion_contract,
)
from .validate import load_collection_protocol, validate_collection_protocol

__all__ = [
    "COLLECTION_BINDING_KEYS",
    "COLLECTION_PROTOCOL_SCHEMA",
    "COLLECTION_SPLITS",
    "COVERAGE_REPORT_SCHEMA",
    "NATIVE_T2_CANARY_PROTOCOL_SCHEMA",
    "NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA",
    "NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA",
    "POWER_METHOD",
    "SEED_DERIVATION",
    "T1_COLLECTION_PROTOCOL_SCHEMA",
    "T1_COVERAGE_REPORT_SCHEMA",
    "CollectionProtocolError",
    "CollectionProtocolIssue",
    "citylite_t1_split_certificate",
    "coverage_report",
    "derive_episode_seed",
    "is_native_t2_canary_protocol",
    "load_collection_protocol",
    "main",
    "native_t2_motion_contract",
    "native_t2_v2_motion_contract",
    "native_t2_v3_motion_contract",
    "protocol_sha256",
    "required_paired_episodes",
    "resolve_collection_binding",
    "validate_collection_binding",
    "validate_collection_protocol",
]
