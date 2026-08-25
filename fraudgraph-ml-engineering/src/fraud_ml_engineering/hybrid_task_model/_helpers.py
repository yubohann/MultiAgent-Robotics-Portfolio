from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRANSFORMER_BATCH_CHUNK_SIZE = 4_096


def seed_legacy_hybrid_compatibility(seed: int | None) -> int:
    """Stabilize model init when legacy checkpoints need fallback parameters."""

    effective_seed = 42 if seed is None else int(seed)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return effective_seed

def _resolve_attention_heads(model_dim: int, preferred_heads: int = 4) -> int:
    for heads in [preferred_heads, 4, 2, 1]:
        if model_dim % heads == 0:
            return heads
    return 1

def _autocast_enabled_for_device(device: torch.device) -> bool:
    if device.type == "cpu":
        cpu_checker = getattr(torch, "is_autocast_cpu_enabled", None)
        return bool(cpu_checker()) if callable(cpu_checker) else False
    return torch.is_autocast_enabled()

def _autocast_dtype_for_device(device: torch.device) -> torch.dtype | None:
    if not _autocast_enabled_for_device(device):
        return None
    if device.type == "cpu":
        cpu_getter = getattr(torch, "get_autocast_cpu_dtype", None)
        return cpu_getter() if callable(cpu_getter) else None
    gpu_getter = getattr(torch, "get_autocast_gpu_dtype", None)
    return gpu_getter() if callable(gpu_getter) else None

def _align_module_input(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return tensor
    if tensor.device != parameter.device:
        tensor = tensor.to(device=parameter.device)
    if not tensor.is_floating_point():
        return tensor.to(dtype=parameter.dtype)
    if tensor.dtype == parameter.dtype:
        autocast_dtype = _autocast_dtype_for_device(parameter.device)
        if autocast_dtype is not None and tensor.dtype == torch.float32:
            return tensor.to(dtype=autocast_dtype)
        return tensor
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return tensor.to(dtype=parameter.dtype)
    autocast_dtype = _autocast_dtype_for_device(parameter.device)
    if autocast_dtype is not None:
        if tensor.dtype == torch.float32:
            return tensor.to(dtype=autocast_dtype)
        return tensor
    return tensor.to(dtype=parameter.dtype)

def _slice_optional_batch(tensor: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]

def _concat_tensor_dict(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not chunks:
        return {}
    return {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0)
        for key in chunks[0]
    }

def _transformer_chunk_size(batch_size: int) -> int:
    return min(max(int(TRANSFORMER_BATCH_CHUNK_SIZE), 1), max(int(batch_size), 1))

def _is_cuda_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message

def _clear_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()

def _math_sdpa_context(x: torch.Tensor):
    if not x.is_cuda:
        return nullcontext()
    attention_module = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention_module, "sdpa_kernel", None) if attention_module is not None else None
    sdp_backend = getattr(attention_module, "SDPBackend", None) if attention_module is not None else None
    if callable(sdpa_kernel) and sdp_backend is not None:
        try:
            return sdpa_kernel([sdp_backend.MATH])
        except TypeError:
            try:
                return sdpa_kernel(backends=[sdp_backend.MATH])
            except TypeError:
                pass
    sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if callable(sdp_kernel):
        try:
            return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except TypeError:
            pass
    return nullcontext()

def _run_chunked_forward_with_backoff(*, batch_size: int, device: torch.device, runner):
    chunk_size = _transformer_chunk_size(batch_size)
    while True:
        try:
            return runner(chunk_size)
        except RuntimeError as error:
            if not _is_cuda_oom_error(error) or device.type != "cuda" or chunk_size <= 1:
                raise
            next_chunk_size = max(chunk_size // 2, 1)
            if next_chunk_size == chunk_size:
                raise
            _clear_cuda_cache(device)
            chunk_size = next_chunk_size

def _safe_transformer_forward(
    transformer: nn.Module,
    x: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    context = _math_sdpa_context(x)
    with context:
        return transformer(x, src_key_padding_mask=padding_mask)

def _class_balanced_weights(
    class_counts: torch.Tensor | None,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if class_counts is None:
        return None
    counts = class_counts.to(device=device, dtype=dtype).clamp(min=1.0)
    beta = float(min(max(beta, 0.0), 0.999999))
    if beta == 0.0:
        return torch.ones_like(counts)
    beta_tensor = torch.full_like(counts, beta)
    effective_num = (1.0 - torch.pow(beta_tensor, counts)).clamp(min=1e-12)
    weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(weights)

def _focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    alpha_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -torch.pow(1.0 - pt, gamma) * log_pt
    if alpha_weights is not None:
        loss = loss * alpha_weights.to(logits.device)[labels]
    return loss.mean()

def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    class_weights: torch.Tensor | None,
    class_counts: torch.Tensor | None,
    focal_gamma: float,
    class_balance_beta: float,
) -> torch.Tensor:
    normalized_loss_name = str(loss_name).lower()
    sample_weights = class_weights.to(logits.device) if class_weights is not None else None
    cb_weights = _class_balanced_weights(
        class_counts=class_counts,
        beta=class_balance_beta,
        device=logits.device,
        dtype=logits.dtype,
    )

    if normalized_loss_name == "ce":
        return F.cross_entropy(logits, labels)
    if normalized_loss_name == "weighted_ce":
        return F.cross_entropy(logits, labels, weight=sample_weights)
    if normalized_loss_name == "focal":
        return _focal_loss(logits, labels, gamma=focal_gamma, alpha_weights=sample_weights)
    if normalized_loss_name == "cb_ce":
        return F.cross_entropy(logits, labels, weight=cb_weights)
    if normalized_loss_name == "cb_focal":
        return _focal_loss(logits, labels, gamma=focal_gamma, alpha_weights=cb_weights)
    raise ValueError(f"Unsupported classification loss: {loss_name}")

def _positive_class_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits[:, 1] - logits[:, 0]

def _pairwise_auc_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_pairs: int = 4096,
) -> torch.Tensor:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    if positive_scores.numel() > max_pairs:
        positive_scores = positive_scores[torch.randperm(positive_scores.numel(), device=scores.device)[:max_pairs]]
    if negative_scores.numel() > max_pairs:
        negative_scores = negative_scores[torch.randperm(negative_scores.numel(), device=scores.device)[:max_pairs]]
    pairwise_margin = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
    return F.softplus(-pairwise_margin).mean()

def _ranking_friendly_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_counts: torch.Tensor | None,
    ranking_weight: float,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = _positive_class_scores(logits)
    positive_count = float(class_counts[1].item()) if class_counts is not None and class_counts.numel() > 1 else float((labels == 1).sum().item())
    negative_count = float(class_counts[0].item()) if class_counts is not None and class_counts.numel() > 0 else float((labels == 0).sum().item())
    pos_weight = max(negative_count / max(positive_count, 1.0), 1.0)
    bce_loss = F.binary_cross_entropy_with_logits(
        scores,
        labels.float(),
        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    ranking_loss = _pairwise_auc_ranking_loss(scores, labels, max_pairs=max_pairs)
    total = bce_loss + float(max(ranking_weight, 0.0)) * ranking_loss
    return total, bce_loss, ranking_loss

def _pseudo_label_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    class_weights: torch.Tensor | None,
    class_counts: torch.Tensor | None,
    focal_gamma: float,
    class_balance_beta: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized_loss_name = str(loss_name).lower()
    sample_weights = (
        sample_weights.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if sample_weights is not None
        else torch.ones(logits.size(0), device=logits.device, dtype=logits.dtype)
    )
    sample_weights = sample_weights.clamp(min=1e-6)
    class_alpha = None
    if normalized_loss_name in {"weighted_ce", "focal"} and class_weights is not None:
        class_alpha = class_weights.to(device=logits.device, dtype=logits.dtype)
    elif normalized_loss_name in {"cb_ce", "cb_focal"}:
        class_alpha = _class_balanced_weights(
            class_counts=class_counts,
            beta=class_balance_beta,
            device=logits.device,
            dtype=logits.dtype,
        )

    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    base_loss = -log_pt
    if normalized_loss_name in {"focal", "cb_focal"}:
        pt = log_pt.exp()
        base_loss = -torch.pow(1.0 - pt, focal_gamma) * log_pt
    if class_alpha is not None:
        base_loss = base_loss * class_alpha[labels]
    weighted = base_loss * sample_weights
    return weighted.sum() / sample_weights.sum().clamp(min=1e-6)

def _uniform_target_kl_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    num_classes = max(int(logits.size(1)), 1)
    uniform_target = torch.full_like(logits, 1.0 / float(num_classes))
    return F.kl_div(F.log_softmax(logits, dim=1), uniform_target, reduction="batchmean")

def _balanced_binary_sample_indices(labels: torch.Tensor) -> torch.Tensor | None:
    """Match SplitGNN's balanced binary sampling before classification loss."""
    if labels.numel() == 0:
        return None
    unique_labels = torch.unique(labels)
    if unique_labels.numel() != 2:
        return None
    pos_index = (labels == 1).nonzero(as_tuple=False).flatten()
    neg_index = (labels == 0).nonzero(as_tuple=False).flatten()
    if pos_index.numel() == 0 or neg_index.numel() == 0:
        return None
    sample_size = int(min(pos_index.numel(), neg_index.numel()))
    pos_choice = pos_index[torch.randperm(pos_index.numel(), device=labels.device)[:sample_size]]
    neg_choice = neg_index[torch.randperm(neg_index.numel(), device=labels.device)[:sample_size]]
    return torch.sort(torch.cat([pos_choice, neg_choice], dim=0)).values

def _balanced_subset_statistics(
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    class_counts: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    num_classes = 0
    if class_counts is not None:
        num_classes = max(num_classes, int(class_counts.numel()))
    if class_weights is not None:
        num_classes = max(num_classes, int(class_weights.numel()))
    if labels.numel() > 0:
        num_classes = max(num_classes, int(labels.max().item()) + 1)
    if num_classes <= 0:
        return class_weights, class_counts

    sampled_counts = torch.bincount(labels.detach(), minlength=num_classes).float().clamp(min=1.0)
    sampled_weights = sampled_counts.sum() / (sampled_counts * len(sampled_counts))
    return sampled_weights, sampled_counts

def _has_finite_branch_tensors(
    branch_outputs: dict[str, torch.Tensor] | None,
    *keys: str,
) -> bool:
    if not isinstance(branch_outputs, dict):
        return False
    for key in keys:
        value = branch_outputs.get(key)
        if value is None or not torch.is_tensor(value) or value.numel() == 0:
            return False
        if not torch.isfinite(value).all():
            return False
    return True

def _mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).flatten()

def _class_centroids(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if embeddings.numel() == 0 or labels.numel() == 0:
        return None
    unique_labels = torch.unique(labels)
    centroids = []
    classes = []
    for label in unique_labels:
        class_mask = labels == label
        if not class_mask.any():
            continue
        centroids.append(embeddings[class_mask].mean(dim=0))
        classes.append(label)
    if not centroids:
        return None
    return torch.stack(centroids, dim=0), torch.stack(classes, dim=0)

def _novelty_scores(
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
) -> torch.Tensor | None:
    centroid_payload = _class_centroids(reference_embeddings, reference_labels)
    if centroid_payload is None or query_embeddings.numel() == 0:
        return None
    centroids, centroid_labels = centroid_payload
    query_distances = torch.cdist(query_embeddings.float(), centroids.float(), p=2)
    query_min_distances = query_distances.min(dim=1).values

    reference_distances = torch.cdist(reference_embeddings.float(), centroids.float(), p=2)
    class_positions = []
    for label in reference_labels:
        matches = (centroid_labels == label).nonzero(as_tuple=False).flatten()
        class_positions.append(matches[0] if matches.numel() > 0 else torch.tensor(0, device=label.device))
    class_positions_tensor = torch.stack(class_positions).long()
    reference_to_own = reference_distances.gather(1, class_positions_tensor.unsqueeze(1)).squeeze(1)
    reference_mean = reference_to_own.mean()
    reference_std = reference_to_own.std(unbiased=False).clamp(min=1e-6)
    return (query_min_distances - reference_mean) / reference_std
