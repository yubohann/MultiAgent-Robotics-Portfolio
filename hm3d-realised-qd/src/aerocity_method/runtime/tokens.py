"""Public ActionToken authorization without evaluator-private inputs."""

from __future__ import annotations

from collections.abc import Sequence

from aerocity_method.contracts.io import canonical_sha256, require_identifier
from aerocity_method.contracts.models import (
    ActionToken,
    CandidateFragmentManifest,
    PublicMethodContext,
)


def authorize_manifest(
    context: PublicMethodContext,
    manifests: Sequence[CandidateFragmentManifest],
    legal_mask: Sequence[bool],
    selected_index: int,
    *,
    token_id: str,
    issued_at: float,
    duration: float,
) -> ActionToken:
    require_identifier(token_id, "token_id")
    rows = tuple(manifests)
    mask = tuple(legal_mask)
    if not rows or len(rows) != len(mask):
        raise ValueError("manifests and legal mask must have the same non-zero length")
    if any(not isinstance(value, bool) for value in mask) or not any(mask):
        raise ValueError("legal mask must be boolean and contain a legal candidate")
    if not isinstance(selected_index, int) or isinstance(selected_index, bool):
        raise ValueError("selected_index must be an integer")
    if selected_index < 0 or selected_index >= len(rows) or not mask[selected_index]:
        raise ValueError("selected_index must identify a legal candidate")
    manifest = rows[selected_index]
    if not manifest.feasible:
        raise ValueError("cannot authorize an infeasible candidate")
    if manifest.context_hash != context.digest:
        raise ValueError("candidate context hash does not match public context")
    episode_decision = {
        (fragment.episode_id, fragment.decision_id) for fragment in manifest.fragments
    }
    if episode_decision != {(context.episode_id, context.decision_id)}:
        raise ValueError("candidate fragment identity does not match context")
    earliest = min(fragment.planned_start for fragment in manifest.fragments)
    latest = max(fragment.planned_end for fragment in manifest.fragments)
    if latest - earliest > duration + 1e-9:
        raise ValueError("ActionToken duration does not cover the candidate fragment window")
    return ActionToken(
        token_id=token_id,
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        context_hash=context.digest,
        manifest_hash=manifest.manifest_hash,
        legal_mask_hash=canonical_sha256(mask),
        planned_fragment_hashes=tuple(fragment.digest for fragment in manifest.fragments),
        issued_at=issued_at,
        duration=duration,
    )
