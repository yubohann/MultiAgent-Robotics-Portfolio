from __future__ import annotations

import torch

def sanitize_legacy_hybrid_state_dict(
    state_dict: dict[str, torch.Tensor],
    current_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Adapt older hybrid checkpoints to the current multimodal fusion layout."""

    legacy_prefix = "sequence_encoder." + "b" + "ilstm."
    dropped_prefixes = ("graph_classifier.", "sequence_classifier.")
    remapped_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(legacy_prefix) or key.startswith(dropped_prefixes):
            continue
        if key.startswith("fusion."):
            # Older checkpoints stored the pre-gated concatenation head under
            # `fusion.*`; that branch now corresponds to `raw_fusion.*`.
            remapped_state[key.replace("fusion.", "raw_fusion.", 1)] = value
            continue
        remapped_state[key] = value

    if current_state_dict is None:
        return remapped_state

    upgraded_state: dict[str, torch.Tensor] = {}
    for key, current_value in current_state_dict.items():
        loaded_value = remapped_state.get(key)
        if loaded_value is None or tuple(loaded_value.shape) != tuple(current_value.shape):
            upgraded_state[key] = current_value.detach().clone()
            continue
        upgraded_state[key] = loaded_value.detach().to(dtype=current_value.dtype).clone()
    return upgraded_state

def uses_legacy_raw_fusion_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    has_legacy_fusion = any(key.startswith("fusion.") for key in state_dict)
    has_gated_fusion = any(key.startswith("gated_fusion.") for key in state_dict)
    return bool(has_legacy_fusion and not has_gated_fusion)

def checkpoint_legacy_fusion_only(checkpoint_payload: dict[str, Any] | None) -> bool:
    if not checkpoint_payload:
        return False
    if "legacy_fusion_only" in checkpoint_payload:
        return bool(checkpoint_payload.get("legacy_fusion_only", False))
    checkpoint_args = checkpoint_payload.get("args", {})
    if isinstance(checkpoint_args, dict) and "legacy_fusion_only" in checkpoint_args:
        return bool(checkpoint_args.get("legacy_fusion_only", False))
    state_dict = checkpoint_payload.get("model_state", checkpoint_payload)
    if isinstance(state_dict, dict):
        return uses_legacy_raw_fusion_checkpoint(state_dict)
    return False

def _balance_modality_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    """Remove global bias and keep each modality on a comparable scale before fusion."""
    if embeddings.ndim != 2 or embeddings.size(-1) <= 0:
        return embeddings
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    return F.normalize(embeddings, p=2, dim=-1, eps=1e-6)

def _detached_uncertainty_target_from_supervision(
    *,
    fusion_logits: torch.Tensor,
    labels: torch.Tensor,
    graph_logits: torch.Tensor | None = None,
    sequence_logits: torch.Tensor | None = None,
    raw_logits: torch.Tensor | None = None,
    conflict_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a smooth confidence target so the uncertainty head has useful supervision on Elliptic."""

    if fusion_logits.ndim != 2 or fusion_logits.size(0) == 0:
        return torch.empty(0, device=fusion_logits.device, dtype=fusion_logits.dtype)
    labels = labels.long()
    fusion_probs = F.softmax(fusion_logits.detach(), dim=-1)
    fusion_confidence = fusion_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    branch_confidences = [fusion_confidence]
    branch_agreements: list[torch.Tensor] = []
    for branch_logits in (graph_logits, sequence_logits, raw_logits):
        if branch_logits is None or branch_logits.ndim != 2 or branch_logits.size(0) != labels.size(0):
            continue
        branch_probs = F.softmax(branch_logits.detach(), dim=-1)
        branch_confidences.append(branch_probs.gather(1, labels.unsqueeze(1)).squeeze(1))
        branch_agreements.append(branch_logits.detach().argmax(dim=-1).eq(labels).float())
    mean_branch_confidence = torch.stack(branch_confidences, dim=0).mean(dim=0)
    mean_branch_agreement = (
        torch.stack(branch_agreements, dim=0).mean(dim=0)
        if branch_agreements
        else torch.ones_like(fusion_confidence)
    )
    target_confidence = 0.60 * fusion_confidence + 0.25 * mean_branch_confidence + 0.15 * mean_branch_agreement
    if conflict_score is not None and conflict_score.numel() == target_confidence.numel():
        detached_conflict = conflict_score.detach().reshape(-1).to(
            device=target_confidence.device,
            dtype=target_confidence.dtype,
        )
        target_confidence = target_confidence * (1.0 - 0.20 * detached_conflict.clamp(min=0.0, max=1.0))
    return target_confidence.clamp(min=0.0, max=1.0)
