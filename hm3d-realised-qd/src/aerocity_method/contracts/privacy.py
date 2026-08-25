"""Public/private boundary checks used before method code receives a payload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aerocity_method.contracts.io import require_identifier

FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "blind_identity",
        "evaluator_private",
        "failed_agent_ids",
        "failure_truth",
        "fault_spec",
        "oracle_outcome",
        "complete_mesh",
        "evaluator_esdf",
        "private_geometry",
        "private_esdf",
        "private_witness",
        "split_id",
        "split_label",
        "seed",
        "episode_seed",
        "layout_seed",
        "rng_seed",
        "target_coordinates",
        "target_count",
        "target_distance",
        "target_family",
        "target_id",
        "target_process",
        "target_truth",
        "truth_map",
    }
)


class PublicBoundaryError(ValueError):
    """Raised when evaluator-private information crosses into method input."""


def walk_public_payload(
    payload: Any,
    *,
    canaries: Sequence[str] = (),
    path: str = "$",
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            require_identifier(key, f"{path} key")
            if key.casefold() in FORBIDDEN_PUBLIC_KEYS:
                raise PublicBoundaryError(f"forbidden public field at {path}.{key}")
            walk_public_payload(value, canaries=canaries, path=f"{path}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            walk_public_payload(value, canaries=canaries, path=f"{path}[{index}]")
        return
    if isinstance(payload, str):
        for canary in canaries:
            if canary and canary in payload:
                raise PublicBoundaryError(f"private canary found at {path}")
