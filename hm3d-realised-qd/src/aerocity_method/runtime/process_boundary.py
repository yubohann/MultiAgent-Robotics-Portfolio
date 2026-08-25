"""In-process public request contract used before a future persistent worker."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import (
    ABI_VERSION,
    CandidateFragmentManifest,
    PublicMethodContext,
)
from aerocity_method.contracts.privacy import walk_public_payload


def build_public_request(
    context: PublicMethodContext,
    manifests: Sequence[CandidateFragmentManifest],
    legal_mask: Sequence[bool],
) -> dict[str, Any]:
    rows = tuple(manifests)
    mask = tuple(legal_mask)
    if not rows or len(rows) != len(mask) or not any(mask):
        raise ValueError("public request requires matching candidates and a legal action")
    if any(not isinstance(value, bool) for value in mask):
        raise ValueError("public request legal mask must be boolean")
    if len({manifest.manifest_hash for manifest in rows}) != len(rows):
        raise ValueError("public request candidates must be unique")
    if any(allowed and not manifest.feasible for manifest, allowed in zip(rows, mask, strict=True)):
        raise ValueError("public request legal mask authorizes an infeasible candidate")
    for manifest in rows:
        if manifest.context_hash != context.digest:
            raise ValueError("candidate is bound to another context")
    payload = {
        "schema_version": ABI_VERSION,
        "context": context.to_dict(),
        "candidates": [manifest.to_dict() for manifest in rows],
        "candidate_hashes": [manifest.manifest_hash for manifest in rows],
        "legal_mask": mask,
    }
    walk_public_payload(payload)
    payload["request_hash"] = canonical_sha256(payload)
    return payload


def validate_public_request(
    payload: Mapping[str, Any], *, canaries: Sequence[str] = ()
) -> dict[str, Any]:
    resolved = dict(payload)
    walk_public_payload(resolved, canaries=canaries)
    if resolved.get("schema_version") != ABI_VERSION:
        raise ValueError("public request schema mismatch")
    supplied_hash = resolved.pop("request_hash", None)
    if canonical_sha256(resolved) != supplied_hash:
        raise ValueError("public request hash mismatch")
    candidates = resolved.get("candidates")
    hashes = resolved.get("candidate_hashes")
    mask = resolved.get("legal_mask")
    if (
        not isinstance(candidates, list)
        or not isinstance(hashes, list)
        or len(candidates) != len(hashes)
    ):
        raise ValueError("candidate payload/hash rows are incomplete")
    if (
        not isinstance(mask, (list, tuple))
        or len(mask) != len(candidates)
        or any(not isinstance(value, bool) for value in mask)
        or not any(mask)
    ):
        raise ValueError("public request legal mask is invalid")
    context = resolved.get("context")
    if not isinstance(context, Mapping) or context.get("schema_version") != ABI_VERSION:
        raise ValueError("public request context is invalid")
    context_hash = canonical_sha256(context)
    if len(set(hashes)) != len(hashes):
        raise ValueError("public request contains duplicate candidate hashes")
    for candidate, digest in zip(candidates, hashes, strict=True):
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate payload must be an object")
        if candidate.get("schema_version") != ABI_VERSION:
            raise ValueError("candidate schema mismatch")
        if canonical_sha256(candidate) != digest:
            raise ValueError("candidate content hash mismatch")
        if candidate.get("context_hash") != context_hash:
            raise ValueError("candidate is rebound to another public context")
    if any(
        allowed and candidate.get("feasible") is not True
        for candidate, allowed in zip(candidates, mask, strict=True)
    ):
        raise ValueError("public request authorizes an infeasible candidate")
    resolved["request_hash"] = supplied_hash
    return resolved


def sanitized_method_environment(
    source: Mapping[str, str] | None = None, *, canaries: Sequence[str] = ()
) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    forbidden_tokens = ("BLIND", "PRIVATE", "TARGET_TRUTH", "FAULT_TRUTH", "EVALUATOR")
    return {
        key: value
        for key, value in environment.items()
        if not any(token in key.upper() for token in forbidden_tokens)
        and not any(canary and canary in value for canary in canaries)
    }
