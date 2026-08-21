from __future__ import annotations

from ._core import HybridFraudModelCore as _Core

from ._encoding_mixin import EncodingMixin
from ._sequence_mixin import SequenceMixin
from ._forward_mixin import ForwardMixin
from ._ssl_mixin import SslMixin
from ._loss_mixin import LossMixin

class HybridFraudModel(_Core, EncodingMixin, SequenceMixin, ForwardMixin, SslMixin, LossMixin):
    """Hybrid graph+sequence fraud model (multi-mixin split)."""
    pass

from ._helpers import (
    TRANSFORMER_BATCH_CHUNK_SIZE,
    _align_module_input,
    _autocast_dtype_for_device,
    _autocast_enabled_for_device,
    _balanced_binary_sample_indices,
    _balanced_subset_statistics,
    _class_balanced_weights,
    _class_centroids,
    _classification_loss,
    _clear_cuda_cache,
    _concat_tensor_dict,
    _focal_loss,
    _has_finite_branch_tensors,
    _is_cuda_oom_error,
    _mask_to_index,
    _math_sdpa_context,
    _novelty_scores,
    _pairwise_auc_ranking_loss,
    _positive_class_scores,
    _pseudo_label_loss,
    _ranking_friendly_classification_loss,
    _resolve_attention_heads,
    _run_chunked_forward_with_backoff,
    _safe_transformer_forward,
    _slice_optional_batch,
    _transformer_chunk_size,
    _uniform_target_kl_loss,
    seed_legacy_hybrid_compatibility
)
from ._encoders import (
    SinusoidalPositionalEncoding,
    TransformerSequenceEncoder
)
from ._legacy import (
    _balance_modality_embedding,
    _detached_uncertainty_target_from_supervision,
    checkpoint_legacy_fusion_only,
    sanitize_legacy_hybrid_state_dict,
    uses_legacy_raw_fusion_checkpoint
)

__all__ = [
    'HybridFraudModel',
    'SinusoidalPositionalEncoding',
    'TRANSFORMER_BATCH_CHUNK_SIZE',
    'TransformerSequenceEncoder',
    '_align_module_input',
    '_autocast_dtype_for_device',
    '_autocast_enabled_for_device',
    '_balance_modality_embedding',
    '_balanced_binary_sample_indices',
    '_balanced_subset_statistics',
    '_class_balanced_weights',
    '_class_centroids',
    '_classification_loss',
    '_clear_cuda_cache',
    '_concat_tensor_dict',
    '_detached_uncertainty_target_from_supervision',
    '_focal_loss',
    '_has_finite_branch_tensors',
    '_is_cuda_oom_error',
    '_mask_to_index',
    '_math_sdpa_context',
    '_novelty_scores',
    '_pairwise_auc_ranking_loss',
    '_positive_class_scores',
    '_pseudo_label_loss',
    '_ranking_friendly_classification_loss',
    '_resolve_attention_heads',
    '_run_chunked_forward_with_backoff',
    '_safe_transformer_forward',
    '_slice_optional_batch',
    '_transformer_chunk_size',
    '_uniform_target_kl_loss',
    'checkpoint_legacy_fusion_only',
    'sanitize_legacy_hybrid_state_dict',
    'seed_legacy_hybrid_compatibility',
    'uses_legacy_raw_fusion_checkpoint'
]
