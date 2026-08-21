from __future__ import annotations

from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

TRANSFORMER_BATCH_CHUNK_SIZE = 4_096


def _resolve_attention_heads(model_dim: int, preferred_heads: int = 4) -> int:
    for heads in [preferred_heads, 4, 2, 1]:
        if model_dim % heads == 0:
            return heads
    return 1


def _masked_mean(
    embeddings: torch.Tensor,
    mask: torch.Tensor | None,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        return embeddings.mean(dim=1)
    weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp(min=1.0)
    pooled = (embeddings * weights).sum(dim=1) / denominator
    if fallback is not None:
        has_valid = mask.any(dim=1, keepdim=True)
        pooled = torch.where(has_valid, pooled, fallback)
    return pooled


def _masked_attention_pool(
    embeddings: torch.Tensor,
    mask: torch.Tensor | None,
    score_layer: nn.Module,
    score_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = score_layer(embeddings).squeeze(-1)
    if score_bias is not None:
        scores = scores + score_bias
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), -1e4)
    attention = torch.softmax(scores, dim=1)
    if mask is not None:
        attention = attention * mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pooled = (attention.unsqueeze(-1) * embeddings).sum(dim=1)
    return pooled, attention


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


def _transformer_chunk_size(batch_size: int, preferred_chunk_size: int | None = None) -> int:
    chunk_limit = TRANSFORMER_BATCH_CHUNK_SIZE if preferred_chunk_size is None else preferred_chunk_size
    return min(max(int(chunk_limit), 1), max(int(batch_size), 1))


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


def _run_chunked_encoder_with_backoff(
    *,
    batch_size: int,
    device: torch.device,
    runner,
    preferred_chunk_size: int | None = None,
):
    chunk_size = _transformer_chunk_size(batch_size, preferred_chunk_size=preferred_chunk_size)
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


def _transformer_forward_impl(
    transformer: nn.Module,
    x: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    context = _math_sdpa_context(x)
    with context:
        return transformer(x, src_key_padding_mask=padding_mask)


def _safe_transformer_forward(
    transformer: nn.Module,
    x: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    use_activation_checkpointing: bool = False,
) -> torch.Tensor:
    should_checkpoint = bool(
        use_activation_checkpointing
        and transformer.training
        and torch.is_grad_enabled()
        and x.requires_grad
    )
    if not should_checkpoint:
        return _transformer_forward_impl(transformer, x, padding_mask=padding_mask)

    def _forward(hidden_states: torch.Tensor) -> torch.Tensor:
        return _transformer_forward_impl(transformer, hidden_states, padding_mask=padding_mask)

    try:
        return activation_checkpoint(_forward, x, use_reentrant=False)
    except TypeError:
        return activation_checkpoint(_forward, x)


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


def _align_floating_input(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
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


def _edge_mean_aggregate(
    graph,
    node_embeddings: torch.Tensor,
    etype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if etype not in graph.etypes:
        zeros = torch.zeros_like(node_embeddings)
        degree = torch.zeros((node_embeddings.size(0), 1), dtype=node_embeddings.dtype, device=node_embeddings.device)
        return zeros, degree
    src_nodes, dst_nodes = graph.edges(etype=etype)
    if src_nodes.numel() == 0:
        zeros = torch.zeros_like(node_embeddings)
        degree = torch.zeros((node_embeddings.size(0), 1), dtype=node_embeddings.dtype, device=node_embeddings.device)
        return zeros, degree
    src_nodes = src_nodes.to(device=node_embeddings.device)
    dst_nodes = dst_nodes.to(device=node_embeddings.device)
    messages = node_embeddings.index_select(0, src_nodes)
    aggregated = torch.zeros_like(node_embeddings)
    aggregated.index_add_(0, dst_nodes, messages)
    degree = torch.zeros((node_embeddings.size(0), 1), dtype=node_embeddings.dtype, device=node_embeddings.device)
    degree.index_add_(
        0,
        dst_nodes,
        torch.ones((dst_nodes.numel(), 1), dtype=node_embeddings.dtype, device=node_embeddings.device),
    )
    aggregated = aggregated / degree.clamp(min=1.0)
    return aggregated, degree


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        harmonic_dim = max(model_dim // 2, 1)
        frequencies = torch.exp(torch.linspace(0.0, math.log(1000.0), steps=harmonic_dim))
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Linear(harmonic_dim * 2 + 1, model_dim)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        values = _align_floating_input(time_deltas, self.proj).unsqueeze(-1)
        harmonic = values / self.frequencies.view(1, 1, -1).to(device=values.device, dtype=values.dtype)
        encoded = torch.cat([values, torch.sin(harmonic), torch.cos(harmonic)], dim=-1)
        return self.proj(encoded)


class TemporalContextAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(model_dim, max(model_dim // 2, 8)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(model_dim // 2, 8), 1),
        )

    def forward(self, temporal_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context_embedding = self.context_proj(_align_floating_input(temporal_context, self.context_proj))
        time_reliability = torch.sigmoid(self.reliability_head(context_embedding))
        return context_embedding, time_reliability


class WaveletLiteHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = max(int(model_dim), int(input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(model_dim, max(model_dim // 2, 8)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(model_dim // 2, 8), 1),
        )

    def forward(self, wavelet_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectral_embedding = self.encoder(_align_floating_input(wavelet_features, self.encoder))
        spectral_gate = torch.sigmoid(self.gate(spectral_embedding))
        return spectral_embedding, spectral_gate


class CoAssociationEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        edge_type: str = "coassociation",
    ):
        super().__init__()
        self.edge_type = str(edge_type)
        self.aggregator = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(output_dim + 1, max(output_dim // 2, 8)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(output_dim // 2, 8), 1),
        )

    def forward(self, graph, node_embeddings: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aggregated, degree = _edge_mean_aggregate(graph, node_embeddings, self.edge_type)
        density = torch.log1p(degree)
        fused = self.aggregator(
            torch.cat([node_embeddings, aggregated, density.expand(-1, 1)], dim=-1)
        )
        gate = torch.sigmoid(self.gate(torch.cat([fused, density], dim=-1)))
        return gate * fused, {
            "coassociation_gate": gate,
            "coassociation_density": density,
        }


class ParameterizedDiffusionResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        relation_names: list[str],
        hidden_dim: int,
        dropout: float = 0.1,
        residual_scale: float = 0.20,
    ):
        super().__init__()
        self.relation_names = [str(name) for name in relation_names]
        self.relation_logits = nn.Parameter(torch.zeros(len(self.relation_names), dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(input_dim)
        self.residual_scale = float(max(residual_scale, 0.0))

    def forward(self, graph, node_embeddings: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_relations = [name for name in self.relation_names if name in graph.etypes]
        if not valid_relations:
            zero_gate = torch.zeros((node_embeddings.size(0), 1), dtype=node_embeddings.dtype, device=node_embeddings.device)
            return node_embeddings, {
                "diffusion_gate": zero_gate,
                "diffusion_neighbor_strength": zero_gate,
            }
        relation_states: list[torch.Tensor] = []
        relation_degrees: list[torch.Tensor] = []
        for relation_name in valid_relations:
            aggregated, degree = _edge_mean_aggregate(graph, node_embeddings, relation_name)
            relation_states.append(aggregated)
            relation_degrees.append(degree)
        stacked_states = torch.stack(relation_states, dim=1)
        stacked_degrees = torch.stack(relation_degrees, dim=1)
        relation_weights = torch.softmax(self.relation_logits[: len(valid_relations)], dim=0).to(
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        mixed_neighbors = (stacked_states * relation_weights.view(1, -1, 1)).sum(dim=1)
        neighbor_strength = (stacked_degrees * relation_weights.view(1, -1, 1)).sum(dim=1)
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        node_embeddings,
                        mixed_neighbors,
                        torch.log1p(neighbor_strength.clamp(min=0.0)),
                    ],
                    dim=-1,
                )
            )
        )
        output = self.output_norm(
            node_embeddings + self.residual_scale * gate * (mixed_neighbors - node_embeddings)
        )
        return output, {
            "diffusion_gate": gate,
            "diffusion_neighbor_strength": neighbor_strength,
        }


class UTGLiteTemporalFusion(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = max(int(model_dim), 32)
        self.anchor_proj = nn.Sequential(
            nn.Linear(model_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.view_gate = nn.Sequential(
            nn.Linear(model_dim * 6 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        *,
        relation_embeddings: torch.Tensor | None,
        event_embeddings: torch.Tensor | None,
        temporal_embeddings: torch.Tensor | None,
        time_reliability: torch.Tensor | None,
        wavelet_embeddings: torch.Tensor | None = None,
        coassociation_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference = next(
            tensor
            for tensor in (relation_embeddings, event_embeddings, temporal_embeddings, wavelet_embeddings, coassociation_embeddings)
            if tensor is not None
        )
        zeros = torch.zeros_like(reference)
        relation_embeddings = zeros if relation_embeddings is None else relation_embeddings
        event_embeddings = zeros if event_embeddings is None else event_embeddings
        temporal_embeddings = zeros if temporal_embeddings is None else temporal_embeddings
        wavelet_embeddings = zeros if wavelet_embeddings is None else wavelet_embeddings
        coassociation_embeddings = zeros if coassociation_embeddings is None else coassociation_embeddings
        temporal_anchor = self.anchor_proj(
            torch.cat(
                [
                    temporal_embeddings,
                    wavelet_embeddings,
                    coassociation_embeddings,
                ],
                dim=-1,
            )
        )
        time_signal = (
            time_reliability.view(-1, 1).to(device=reference.device, dtype=reference.dtype)
            if time_reliability is not None
            else torch.full((reference.size(0), 1), 0.5, device=reference.device, dtype=reference.dtype)
        )
        gate_logits = self.view_gate(
            torch.cat(
                [
                    relation_embeddings,
                    event_embeddings,
                    temporal_anchor,
                    torch.abs(relation_embeddings - event_embeddings),
                    torch.abs(relation_embeddings - temporal_anchor),
                    torch.abs(event_embeddings - temporal_anchor),
                    time_signal,
                ],
                dim=-1,
            )
        )
        availability = torch.stack(
            [
                relation_embeddings.abs().sum(dim=-1).gt(0),
                event_embeddings.abs().sum(dim=-1).gt(0),
                temporal_anchor.abs().sum(dim=-1).gt(0),
            ],
            dim=-1,
        )
        gate_logits = gate_logits.masked_fill(~availability, -1e4)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        unified = (
            gate_weights[:, 0:1] * relation_embeddings
            + gate_weights[:, 1:2] * event_embeddings
            + gate_weights[:, 2:3] * temporal_anchor
        )
        unified = self.output_norm(
            unified + 0.10 * (torch.abs(relation_embeddings - event_embeddings) + torch.abs(temporal_anchor - unified))
        )
        return unified, {
            "utg_relation_gate": gate_weights[:, 0:1],
            "utg_event_gate": gate_weights[:, 1:2],
            "utg_temporal_gate": gate_weights[:, 2:3],
            "utg_temporal_anchor": temporal_anchor,
        }


class RawFeatureAnchorEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(_align_floating_input(features, self.net))


class TypedTabularEncoder(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: list[int],
        output_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        categorical_embedding_dim: int | None = None,
    ):
        super().__init__()
        self.numeric_dim = max(int(numeric_dim), 0)
        self.categorical_cardinalities = [max(int(size), 1) for size in categorical_cardinalities]
        self.num_categorical = len(self.categorical_cardinalities)
        self.output_dim = int(output_dim)
        self.hidden_dim = max(int(hidden_dim), int(output_dim))
        self.categorical_embedding_dim = (
            max(int(categorical_embedding_dim), 8)
            if categorical_embedding_dim is not None
            else max(min(int(output_dim) // 2, 32), 8)
        )

        self.numeric_encoder = None
        if self.numeric_dim > 0:
            self.numeric_encoder = nn.Sequential(
                nn.Linear(self.numeric_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )

        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, self.categorical_embedding_dim)
                for cardinality in self.categorical_cardinalities
            ]
        )
        self.column_embeddings = nn.Parameter(
            torch.zeros(max(self.num_categorical, 1), self.categorical_embedding_dim)
        )
        nn.init.normal_(self.column_embeddings, mean=0.0, std=0.02)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, self.categorical_embedding_dim))
        nn.init.normal_(self.missing_token, mean=0.0, std=0.02)
        self.frequency_proj = nn.Linear(1, self.categorical_embedding_dim)
        self.categorical_pool_score = nn.Sequential(
            nn.Linear(self.categorical_embedding_dim, self.categorical_embedding_dim),
            nn.GELU(),
            nn.Linear(self.categorical_embedding_dim, 1),
        )
        self.categorical_proj = nn.Sequential(
            nn.Linear(self.categorical_embedding_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        missingness_input_dim = self.numeric_dim + self.num_categorical
        self.missingness_encoder = None
        if missingness_input_dim > 0:
            self.missingness_encoder = nn.Sequential(
                nn.Linear(missingness_input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        fusion_input_dim = self.output_dim * 3
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        numeric_values: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical_ids: torch.Tensor,
        categorical_missing: torch.Tensor,
        categorical_frequency: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = numeric_values.device if numeric_values.numel() > 0 else categorical_ids.device
        batch_size = (
            int(numeric_values.size(0))
            if numeric_values.ndim >= 2
            else int(categorical_ids.size(0))
            if categorical_ids.ndim >= 2
            else 0
        )
        dtype = (
            numeric_values.dtype
            if numeric_values.numel() > 0
            else categorical_frequency.dtype
            if categorical_frequency.numel() > 0
            else torch.float32
        )

        numeric_summary = torch.zeros((batch_size, self.output_dim), device=device, dtype=dtype)
        if self.numeric_encoder is not None and numeric_values.ndim == 2:
            numeric_input = torch.cat(
                [
                    _align_floating_input(numeric_values, self.numeric_encoder),
                    _align_floating_input(numeric_missing, self.numeric_encoder),
                ],
                dim=-1,
            )
            numeric_summary = self.numeric_encoder(numeric_input)

        categorical_summary = torch.zeros((batch_size, self.output_dim), device=device, dtype=dtype)
        categorical_attention = torch.zeros((batch_size, self.num_categorical), device=device, dtype=dtype)
        if self.num_categorical > 0 and categorical_ids.ndim == 2:
            token_list: list[torch.Tensor] = []
            for index, embedding in enumerate(self.categorical_embeddings):
                token = embedding(
                    categorical_ids[:, index].clamp(min=0, max=embedding.num_embeddings - 1).to(device=device, dtype=torch.long)
                )
                token = token + self.column_embeddings[index].unsqueeze(0)
                token = token + self.frequency_proj(
                    categorical_frequency[:, index : index + 1].to(device=device, dtype=token.dtype)
                )
                token = token + (
                    categorical_missing[:, index : index + 1].to(device=device, dtype=token.dtype)
                    * self.missing_token.squeeze(0)
                )
                token_list.append(token)
            categorical_tokens = torch.stack(token_list, dim=1)
            categorical_summary_raw, categorical_attention = _masked_attention_pool(
                categorical_tokens,
                torch.ones((batch_size, self.num_categorical), dtype=torch.bool, device=device),
                self.categorical_pool_score,
            )
            categorical_summary = self.categorical_proj(categorical_summary_raw)

        missingness_summary = torch.zeros((batch_size, self.output_dim), device=device, dtype=dtype)
        if self.missingness_encoder is not None:
            missingness_input = torch.cat(
                [
                    numeric_missing.to(device=device, dtype=dtype),
                    categorical_missing.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            missingness_summary = self.missingness_encoder(
                _align_floating_input(missingness_input, self.missingness_encoder)
            )

        shared = self.output_proj(
            torch.cat([numeric_summary, categorical_summary, missingness_summary], dim=-1)
        )
        return shared, {
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "missingness_summary": missingness_summary,
            "categorical_attention": categorical_attention,
        }


class RelationCapsuleSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        relation_vocab_size: int,
        token_type_count: int = 5,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 64,
        batch_chunk_size: int | None = None,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.batch_chunk_size = TRANSFORMER_BATCH_CHUNK_SIZE if batch_chunk_size is None else max(int(batch_chunk_size), 1)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim=model_dim, max_len=max_len)
        self.token_type_embedding = nn.Embedding(max(int(token_type_count), 5), model_dim)
        self.relation_embedding = nn.Embedding(max(int(relation_vocab_size), 1), model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=_resolve_attention_heads(model_dim),
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=max(int(num_layers), 1),
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1))
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.pool_score = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(model_dim * 6, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def _default_token_types(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        token_types = torch.full((batch_size, seq_len), 1, dtype=torch.long, device=device)
        token_types[:, 0] = 0
        if seq_len > 1:
            token_types[:, -1] = 4
        return token_types

    def _type_summary(
        self,
        embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        token_types: torch.Tensor,
        type_id: int,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        type_mask = valid_mask & token_types.eq(int(type_id))
        return _masked_mean(embeddings, type_mask, fallback=fallback)

    def _forward_impl(
        self,
        sequence_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        relation_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_proj(_align_floating_input(sequence_tokens, self.input_proj))
        if token_types is None:
            token_types = self._default_token_types(x.size(0), x.size(1), x.device)
        else:
            token_types = token_types.to(device=x.device, dtype=torch.long)
        if relation_ids is None:
            relation_ids = torch.zeros((x.size(0), x.size(1)), dtype=torch.long, device=x.device)
        else:
            relation_ids = relation_ids.to(device=x.device, dtype=torch.long)
        x = x + self.token_type_embedding(token_types.clamp(min=0, max=self.token_type_embedding.num_embeddings - 1))
        x = x + self.relation_embedding(relation_ids.clamp(min=0, max=self.relation_embedding.num_embeddings - 1))
        if token_weights is None:
            token_weights = torch.ones((x.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        else:
            token_weights = token_weights.to(device=x.device, dtype=x.dtype)
        if token_mask is not None:
            token_weights = token_weights * token_mask.to(dtype=x.dtype)
        x = x + torch.log1p(token_weights.clamp(min=0.0)).unsqueeze(-1)
        position = self.position_encoding.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)
        boundary_mask = ((token_types == 0) | (token_types == 4)).unsqueeze(-1).to(dtype=x.dtype)
        x = x + 0.10 * boundary_mask * position + 0.03 * (1.0 - boundary_mask) * position
        x = self.input_norm(x)
        padding_mask = ~token_mask.bool() if token_mask is not None else None
        x = _safe_transformer_forward(
            self.transformer,
            x,
            padding_mask=padding_mask,
            use_activation_checkpointing=self.activation_checkpointing,
        )
        x = self.norm(x)
        valid_mask = token_mask.bool() if token_mask is not None else torch.ones(
            (x.size(0), x.size(1)),
            dtype=torch.bool,
            device=x.device,
        )
        pooled_context, attention = _masked_attention_pool(
            x,
            valid_mask,
            self.pool_score,
            score_bias=torch.log(token_weights.clamp(min=1e-4)),
        )
        self_summary = self._type_summary(x, valid_mask, token_types, 0, fallback=pooled_context)
        global_summary = self._type_summary(x, valid_mask, token_types, 4, fallback=pooled_context)
        local_summary = self._type_summary(x, valid_mask, token_types, 1, fallback=pooled_context)
        motif_summary = self._type_summary(x, valid_mask, token_types, 2, fallback=pooled_context)
        reliability_summary = self._type_summary(x, valid_mask, token_types, 3, fallback=pooled_context)
        capsule_consensus = 0.35 * local_summary + 0.30 * motif_summary + 0.20 * reliability_summary + 0.15 * pooled_context
        summary = self.summary_proj(
            torch.cat(
                [
                    self_summary,
                    local_summary,
                    motif_summary,
                    reliability_summary,
                    global_summary,
                    capsule_consensus,
                ],
                dim=-1,
            )
        )
        summary = self.dropout(summary)
        summary = self.output_norm(summary)
        return summary, {
            "self_summary": self_summary,
            "local_summary": local_summary,
            "motif_summary": motif_summary,
            "reliability_summary": reliability_summary,
            "global_summary": global_summary,
            "capsule_summaries": torch.stack(
                [local_summary, motif_summary, reliability_summary],
                dim=1,
            ),
        }

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        relation_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = int(sequence_tokens.size(0))

        def _run(chunk_size: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if batch_size <= chunk_size:
                return self._forward_impl(
                    sequence_tokens,
                    token_mask=token_mask,
                    token_weights=token_weights,
                    token_types=token_types,
                    relation_ids=relation_ids,
                )

            summary_chunks: list[torch.Tensor] = []
            detail_chunks: list[dict[str, torch.Tensor]] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                summary_chunk, detail_chunk = self._forward_impl(
                    sequence_tokens[start:end],
                    token_mask=_slice_optional_batch(token_mask, start, end),
                    token_weights=_slice_optional_batch(token_weights, start, end),
                    token_types=_slice_optional_batch(token_types, start, end),
                    relation_ids=_slice_optional_batch(relation_ids, start, end),
                )
                summary_chunks.append(summary_chunk)
                detail_chunks.append(detail_chunk)
            return torch.cat(summary_chunks, dim=0), _concat_tensor_dict(detail_chunks)

        return _run_chunked_encoder_with_backoff(
            batch_size=batch_size,
            device=sequence_tokens.device,
            runner=_run,
            preferred_chunk_size=self.batch_chunk_size,
        )


class EventTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        anchor_dim: int,
        model_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 16,
        event_type_count: int = 4,
        source_count: int | None = None,
        batch_chunk_size: int | None = None,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.batch_chunk_size = TRANSFORMER_BATCH_CHUNK_SIZE if batch_chunk_size is None else max(int(batch_chunk_size), 1)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.anchor_proj = nn.Linear(anchor_dim, model_dim)
        self.time_encoding = ContinuousTimeEncoding(model_dim)
        self.temporal_context_proj = nn.Linear(model_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim=model_dim, max_len=max_len + 1)
        self.token_type_embedding = nn.Embedding(max(int(event_type_count), 4), model_dim)
        self.source_embedding = nn.Embedding(max(int(source_count or event_type_count), 6), model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=_resolve_attention_heads(model_dim),
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=max(int(num_layers), 1),
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1))
        self.norm = nn.LayerNorm(model_dim)
        self.pool_score = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.source_pool_score = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(model_dim * 6, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.summary_gate = nn.Sequential(
            nn.Linear(model_dim * 3 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def _forward_impl(
        self,
        event_sequence: torch.Tensor,
        anchor_features: torch.Tensor,
        event_mask: torch.Tensor | None = None,
        event_time_deltas: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        temporal_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_proj(_align_floating_input(event_sequence, self.input_proj))
        batch_size = x.size(0)
        if token_types is None:
            token_types = torch.zeros((batch_size, x.size(1)), dtype=torch.long, device=x.device)
        else:
            token_types = token_types.to(device=x.device, dtype=torch.long)
        if source_ids is None:
            source_ids = token_types
        else:
            source_ids = source_ids.to(device=x.device, dtype=torch.long)
        x = x + self.token_type_embedding(token_types.clamp(min=0, max=self.token_type_embedding.num_embeddings - 1))
        x = x + self.source_embedding(source_ids.clamp(min=0, max=self.source_embedding.num_embeddings - 1))
        if event_time_deltas is None:
            event_time_deltas = torch.zeros((batch_size, x.size(1)), dtype=x.dtype, device=x.device)
        else:
            event_time_deltas = event_time_deltas.to(device=x.device, dtype=x.dtype)
        x = x + self.time_encoding(event_time_deltas)
        if token_weights is None:
            token_weights = torch.ones((batch_size, x.size(1)), dtype=x.dtype, device=x.device)
        else:
            token_weights = token_weights.to(device=x.device, dtype=x.dtype)
        if event_mask is None:
            event_mask = torch.ones((batch_size, x.size(1)), dtype=torch.bool, device=x.device)
        else:
            event_mask = event_mask.to(device=x.device, dtype=torch.bool)
            token_weights = token_weights * event_mask.to(dtype=x.dtype)
        anchor_token = self.anchor_proj(_align_floating_input(anchor_features, self.anchor_proj)).unsqueeze(1)
        temporal_context_embedding = torch.zeros((batch_size, x.size(-1)), dtype=x.dtype, device=x.device)
        if temporal_context is not None:
            temporal_context_embedding = self.temporal_context_proj(
                _align_floating_input(temporal_context, self.temporal_context_proj)
            ).to(device=x.device, dtype=x.dtype)
            anchor_token = anchor_token + 0.30 * temporal_context_embedding.unsqueeze(1)
        anchor_types = torch.full((batch_size, 1), 3, dtype=torch.long, device=x.device)
        anchor_sources = torch.zeros((batch_size, 1), dtype=torch.long, device=x.device)
        anchor_token = anchor_token + self.token_type_embedding(anchor_types)
        anchor_token = anchor_token + self.source_embedding(anchor_sources)
        x = torch.cat([anchor_token, x], dim=1)
        token_weights = torch.cat([torch.ones((batch_size, 1), dtype=x.dtype, device=x.device), token_weights], dim=1)
        event_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=x.device), event_mask], dim=1)
        source_ids = torch.cat([anchor_sources, source_ids], dim=1)
        position = self.position_encoding.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)
        x = x + position + torch.log1p(token_weights.clamp(min=0.0)).unsqueeze(-1)
        x = self.input_norm(x)
        padding_mask = ~event_mask.bool()
        x = _safe_transformer_forward(
            self.transformer,
            x,
            padding_mask=padding_mask,
            use_activation_checkpointing=self.activation_checkpointing,
        )
        x = self.norm(x)
        pooled_context, attention = _masked_attention_pool(
            x,
            event_mask,
            self.pool_score,
            score_bias=torch.log(token_weights.clamp(min=1e-4)),
        )
        anchor_summary = x[:, 0, :]
        event_hidden = x[:, 1:, :]
        event_only_mask = event_mask[:, 1:]
        event_only_weights = token_weights[:, 1:]
        event_only_sources = source_ids[:, 1:]
        recent_summary = _masked_mean(event_hidden, event_only_mask, fallback=anchor_summary)
        current_positions = event_only_mask.long().sum(dim=1).clamp(min=1) - 1
        current_summary = event_hidden[
            torch.arange(event_hidden.size(0), device=x.device),
            current_positions,
        ]
        weighted_recent, _ = _masked_attention_pool(
            event_hidden,
            event_only_mask,
            self.pool_score,
            score_bias=torch.log(event_only_weights.clamp(min=1e-4)),
        )
        source_context = recent_summary
        source_attention = torch.zeros(
            (batch_size, max(self.source_embedding.num_embeddings - 1, 1)),
            dtype=x.dtype,
            device=x.device,
        )
        source_summaries: list[torch.Tensor] = []
        source_presence: list[torch.Tensor] = []
        for source_id in range(1, self.source_embedding.num_embeddings):
            source_mask = event_only_mask & event_only_sources.eq(int(source_id))
            source_summaries.append(_masked_mean(event_hidden, source_mask, fallback=anchor_summary))
            source_presence.append(source_mask.any(dim=1))
        if source_summaries:
            stacked_source_summaries = torch.stack(source_summaries, dim=1)
            stacked_source_presence = torch.stack(source_presence, dim=1)
            source_context, source_attention = _masked_attention_pool(
                stacked_source_summaries,
                stacked_source_presence,
                self.source_pool_score,
            )
        time_alignment = F.cosine_similarity(anchor_summary, temporal_context_embedding + 1e-6, dim=-1).unsqueeze(-1)
        temporal_gate = torch.sigmoid(
            self.summary_gate(
                torch.cat([anchor_summary, recent_summary, temporal_context_embedding, time_alignment], dim=-1)
            )
        )
        summary = self.summary_proj(
            torch.cat(
                [
                    anchor_summary,
                    current_summary,
                    recent_summary,
                    weighted_recent,
                    source_context,
                    temporal_gate * temporal_context_embedding + (1.0 - temporal_gate) * anchor_summary,
                ],
                dim=-1,
            )
        )
        summary = self.dropout(summary)
        summary = self.output_norm(summary)
        return summary, {
            "anchor_summary": anchor_summary,
            "current_summary": current_summary,
            "recent_summary": recent_summary,
            "weighted_recent": weighted_recent,
            "source_context": source_context,
            "source_attention": source_attention,
            "temporal_context_embedding": temporal_context_embedding,
            "temporal_gate": temporal_gate,
        }

    def forward(
        self,
        event_sequence: torch.Tensor,
        anchor_features: torch.Tensor,
        event_mask: torch.Tensor | None = None,
        event_time_deltas: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        temporal_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = int(event_sequence.size(0))

        def _run(chunk_size: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if batch_size <= chunk_size:
                return self._forward_impl(
                    event_sequence,
                    anchor_features,
                    event_mask=event_mask,
                    event_time_deltas=event_time_deltas,
                    token_weights=token_weights,
                    token_types=token_types,
                    source_ids=source_ids,
                    temporal_context=temporal_context,
                )

            summary_chunks: list[torch.Tensor] = []
            detail_chunks: list[dict[str, torch.Tensor]] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                summary_chunk, detail_chunk = self._forward_impl(
                    event_sequence[start:end],
                    anchor_features[start:end],
                    event_mask=_slice_optional_batch(event_mask, start, end),
                    event_time_deltas=_slice_optional_batch(event_time_deltas, start, end),
                    token_weights=_slice_optional_batch(token_weights, start, end),
                    token_types=_slice_optional_batch(token_types, start, end),
                    source_ids=_slice_optional_batch(source_ids, start, end),
                    temporal_context=_slice_optional_batch(temporal_context, start, end),
                )
                summary_chunks.append(summary_chunk)
                detail_chunks.append(detail_chunk)
            return torch.cat(summary_chunks, dim=0), _concat_tensor_dict(detail_chunks)

        return _run_chunked_encoder_with_backoff(
            batch_size=batch_size,
            device=event_sequence.device,
            runner=_run,
            preferred_chunk_size=self.batch_chunk_size,
        )


class PrototypeMemoryBank(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        num_classes: int,
        num_datasets: int,
        relation_dim: int | None = None,
        fraud_subtype_count: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        relation_dim = int(shared_dim if relation_dim is None else relation_dim)
        self.shared_dim = int(shared_dim)
        self.relation_dim = relation_dim
        self.fraud_subtype_count = max(int(fraud_subtype_count), 2)
        self.class_prototypes = nn.Parameter(torch.randn(max(int(num_classes), 2), shared_dim) * 0.02)
        self.fraud_sub_prototypes = nn.Parameter(torch.randn(self.fraud_subtype_count, shared_dim) * 0.02)
        self.relation_prototypes = nn.Parameter(torch.randn(3, shared_dim) * 0.02)
        self.dataset_prototypes = nn.Parameter(torch.randn(max(int(num_datasets), 1), shared_dim) * 0.02)
        self.relation_proj = (
            nn.Identity()
            if relation_dim == int(shared_dim)
            else nn.Sequential(
                nn.Linear(relation_dim, shared_dim),
                nn.LayerNorm(shared_dim),
            )
        )
        self.refine = nn.Sequential(
            nn.Linear(shared_dim * 4, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
        )
        self.norm = nn.LayerNorm(shared_dim)

    def project_relation_summaries(self, relation_summaries: torch.Tensor) -> torch.Tensor:
        return self.relation_proj(relation_summaries)

    def forward(
        self,
        shared_seed: torch.Tensor,
        relation_summaries: torch.Tensor | None = None,
        dataset_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scale = math.sqrt(max(shared_seed.size(-1), 1))
        class_attention = torch.softmax(shared_seed @ self.class_prototypes.t() / scale, dim=-1)
        class_context = class_attention @ self.class_prototypes
        fraud_sub_attention = torch.softmax(shared_seed @ self.fraud_sub_prototypes.t() / scale, dim=-1)
        fraud_sub_context = fraud_sub_attention @ self.fraud_sub_prototypes
        positive_class_confidence = class_attention[:, 1:2] if class_attention.size(1) > 1 else class_attention[:, :1]
        fraud_context = positive_class_confidence * fraud_sub_context
        if dataset_ids is None:
            dataset_context = torch.zeros_like(shared_seed)
        else:
            dataset_ids = dataset_ids.long().clamp(min=0, max=self.dataset_prototypes.size(0) - 1)
            dataset_context = self.dataset_prototypes[dataset_ids]
        relation_context = torch.zeros_like(shared_seed)
        relation_alignment = torch.zeros((shared_seed.size(0), 3), dtype=shared_seed.dtype, device=shared_seed.device)
        if relation_summaries is not None:
            relation_summaries = self.project_relation_summaries(relation_summaries)
            prototypes = self.relation_prototypes.unsqueeze(0).expand_as(relation_summaries)
            relation_alignment = 0.5 + 0.5 * F.cosine_similarity(relation_summaries, prototypes, dim=-1)
            relation_context = ((relation_summaries + prototypes) * 0.5 * relation_alignment.unsqueeze(-1)).mean(dim=1)
        refined = self.refine(torch.cat([shared_seed, class_context + fraud_context, dataset_context, relation_context], dim=-1))
        enhanced_shared = self.norm(shared_seed + refined)
        return {
            "enhanced_shared": enhanced_shared,
            "class_attention": class_attention,
            "class_context": class_context,
            "fraud_sub_attention": fraud_sub_attention,
            "fraud_sub_context": fraud_sub_context,
            "fraud_context": fraud_context,
            "dataset_context": dataset_context,
            "relation_context": relation_context,
            "relation_alignment": relation_alignment,
        }


class PrototypeReliabilityScorer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = max(int(num_classes), 2)
        self.score_head = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        probs: torch.Tensor,
        prototype_distances: torch.Tensor,
        conflict_score: torch.Tensor | None = None,
        modality_agreement: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        probs = probs.float()
        prototype_distances = prototype_distances.float()
        top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=-1)
        confidence = top2.values[:, 0]
        if top2.values.size(1) > 1:
            prob_margin = top2.values[:, 0] - top2.values[:, 1]
        else:
            prob_margin = top2.values[:, 0]
        normalized_probs = probs.clamp(min=1e-8)
        entropy = -(normalized_probs * normalized_probs.log()).sum(dim=-1) / math.log(float(probs.size(1)))
        sorted_distances = torch.sort(prototype_distances, dim=-1).values
        nearest_distance = sorted_distances[:, 0]
        if sorted_distances.size(1) > 1:
            second_distance = sorted_distances[:, 1]
            prototype_margin = second_distance - nearest_distance
        else:
            second_distance = sorted_distances[:, 0]
            # A single available prototype cannot provide a meaningful margin signal.
            prototype_margin = torch.zeros_like(nearest_distance)
        if conflict_score is None:
            conflict_score = torch.zeros_like(confidence)
        else:
            conflict_score = conflict_score.float().view(-1)
        if modality_agreement is None:
            modality_agreement = torch.ones_like(confidence)
        else:
            modality_agreement = modality_agreement.float().view(-1)
        features = torch.stack(
            [
                confidence,
                prob_margin,
                1.0 - entropy,
                -nearest_distance,
                prototype_margin,
                1.0 - conflict_score,
                modality_agreement,
            ],
            dim=-1,
        )
        raw_score = self.score_head(features).squeeze(-1)
        reliability = torch.sigmoid(raw_score)
        return {
            "reliability": reliability,
            "confidence": confidence,
            "entropy": entropy,
            "nearest_distance": nearest_distance,
            "second_distance": second_distance,
            "prototype_margin": prototype_margin,
            "conflict_score": conflict_score,
            "modality_agreement": modality_agreement,
            "raw_score": raw_score,
        }


class GraphDominantResidualFusion(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.output_dim = hidden_dim * 4
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        graph_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        graph_logits: torch.Tensor,
        sequence_logits: torch.Tensor,
        graph_gate_logit_bias: float = 0.0,
        graph_residual_min_gate: float = 0.0,
        sequence_residual_scale: float = 1.0,
        fusion_delta_scale: float = 0.35,
        force_graph_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        graph_hidden = self.graph_proj(graph_embeddings)
        context_hidden = self.context_proj(context_embeddings)
        shared_gap = torch.abs(graph_hidden - context_hidden)
        private_interaction = graph_hidden * context_hidden
        fusion_features = torch.cat([graph_hidden, context_hidden, shared_gap, private_interaction], dim=-1)
        delta_logits = self.delta_head(torch.cat([graph_hidden, context_hidden], dim=-1))
        if force_graph_only:
            graph_branch_gate = torch.ones((graph_logits.size(0), 1), dtype=graph_logits.dtype, device=graph_logits.device)
        else:
            graph_branch_gate = torch.sigmoid(self.gate_head(fusion_features) + float(graph_gate_logit_bias))
            if float(graph_residual_min_gate) > 0.0:
                graph_anchor = float(min(max(graph_residual_min_gate, 0.0), 1.0))
                graph_branch_gate = graph_anchor + (1.0 - graph_anchor) * graph_branch_gate
        sequence_branch_gate = (1.0 - graph_branch_gate) * float(sequence_residual_scale)
        fusion_delta_gate = float(fusion_delta_scale) * (1.0 - graph_branch_gate)
        total_gate = (graph_branch_gate + sequence_branch_gate + fusion_delta_gate).clamp(min=1e-6)
        graph_probs = torch.softmax(graph_logits.float(), dim=-1)
        sequence_probs = torch.softmax(sequence_logits.float(), dim=-1)
        graph_confidence = graph_probs.max(dim=-1, keepdim=True).values.to(dtype=graph_logits.dtype)
        sequence_confidence = sequence_probs.max(dim=-1, keepdim=True).values.to(dtype=graph_logits.dtype)
        agreement = F.cosine_similarity(graph_probs, sequence_probs, dim=-1, eps=1e-6).unsqueeze(-1)
        agreement = agreement.clamp(min=0.0, max=1.0).to(dtype=graph_logits.dtype)
        confidence_advantage = (
            (graph_confidence - sequence_confidence).clamp(min=0.0)
            / graph_confidence.clamp(min=torch.finfo(graph_logits.dtype).eps)
        )
        graph_correction_support = (0.65 * agreement + 0.35 * confidence_advantage).clamp(min=0.0, max=1.0)
        delta_correction_support = (0.50 + 0.50 * agreement).clamp(min=0.0, max=1.0)
        graph_correction_gate = (graph_branch_gate / total_gate) * graph_correction_support
        delta_correction_gate = (fusion_delta_gate / total_gate) * delta_correction_support
        logits = (
            sequence_logits
            + graph_correction_gate * (graph_logits - sequence_logits)
            + delta_correction_gate * (delta_logits - sequence_logits)
        )
        return {
            "graph_hidden": graph_hidden,
            "context_hidden": context_hidden,
            "shared_gap": shared_gap,
            "private_interaction": private_interaction,
            "fusion_features": fusion_features,
            "delta_logits": delta_logits,
            "graph_branch_gate": graph_branch_gate,
            "sequence_branch_gate": sequence_branch_gate,
            "fusion_delta_gate": fusion_delta_gate,
            "graph_correction_support": graph_correction_support,
            "delta_correction_support": delta_correction_support,
            "logits": logits,
        }


class TriStreamGateFusion(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        sequence_dim: int,
        raw_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        batch_chunk_size: int | None = None,
    ):
        super().__init__()
        self.batch_chunk_size = (
            TRANSFORMER_BATCH_CHUNK_SIZE if batch_chunk_size is None else max(int(batch_chunk_size), 1)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.sequence_proj = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        gate_input_dim = hidden_dim * 6 + 4
        self.gate_head = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 3),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.output_dim = hidden_dim * 3 + 4

    def _forward_impl(
        self,
        *,
        graph_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        raw_embeddings: torch.Tensor,
        graph_logits: torch.Tensor,
        sequence_logits: torch.Tensor,
        raw_logits: torch.Tensor,
        time_reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        graph_hidden = self.graph_proj(graph_embeddings)
        sequence_hidden = self.sequence_proj(sequence_embeddings)
        raw_hidden = self.raw_proj(raw_embeddings)
        graph_sequence_gap = torch.abs(graph_hidden - sequence_hidden)
        graph_raw_gap = torch.abs(graph_hidden - raw_hidden)
        sequence_raw_gap = torch.abs(sequence_hidden - raw_hidden)
        shared_gap_sum = graph_sequence_gap + graph_raw_gap + sequence_raw_gap
        branch_confidences = torch.stack(
            [
                torch.softmax(graph_logits, dim=-1).max(dim=-1).values,
                torch.softmax(sequence_logits, dim=-1).max(dim=-1).values,
                torch.softmax(raw_logits, dim=-1).max(dim=-1).values,
            ],
            dim=-1,
        )
        if time_reliability is None:
            time_reliability = torch.full(
                (graph_hidden.size(0), 1),
                0.5,
                dtype=graph_hidden.dtype,
                device=graph_hidden.device,
            )
        else:
            time_reliability = time_reliability.to(device=graph_hidden.device, dtype=graph_hidden.dtype).view(-1, 1)
        gate_inputs = torch.cat(
            [
                graph_hidden,
                sequence_hidden,
                raw_hidden,
                graph_sequence_gap,
                graph_raw_gap,
                sequence_raw_gap,
                branch_confidences,
                time_reliability,
            ],
            dim=-1,
        )
        gate_logits = self.gate_head(gate_inputs)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        graph_branch_gate = gate_weights[:, 0:1]
        sequence_branch_gate = gate_weights[:, 1:2]
        raw_branch_gate = gate_weights[:, 2:3]
        fused_embedding = (
            graph_branch_gate * graph_hidden
            + sequence_branch_gate * sequence_hidden
            + raw_branch_gate * raw_hidden
        )
        pairwise_interaction = graph_hidden * sequence_hidden + graph_hidden * raw_hidden + sequence_hidden * raw_hidden
        fusion_features = torch.cat(
            [
                fused_embedding,
                shared_gap_sum,
                pairwise_interaction,
                branch_confidences,
                time_reliability,
            ],
            dim=-1,
        )
        delta_logits = self.delta_head(torch.cat([fused_embedding, shared_gap_sum, pairwise_interaction], dim=-1))
        logits = (
            graph_branch_gate * graph_logits
            + sequence_branch_gate * sequence_logits
            + raw_branch_gate * raw_logits
            + 0.15 * delta_logits
        )
        return {
            "shared_gap": shared_gap_sum / 3.0,
            "private_interaction": pairwise_interaction / 3.0,
            "time_reliability": time_reliability,
            "graph_branch_gate": graph_branch_gate,
            "sequence_branch_gate": sequence_branch_gate,
            "raw_branch_gate": raw_branch_gate,
            "fusion_features": fusion_features,
            "fusion_delta_gate": torch.full_like(graph_branch_gate, 0.15),
            "logits": logits,
        }

    def forward(
        self,
        *,
        graph_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        raw_embeddings: torch.Tensor,
        graph_logits: torch.Tensor,
        sequence_logits: torch.Tensor,
        raw_logits: torch.Tensor,
        time_reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(graph_embeddings.size(0))

        def _run(chunk_size: int) -> dict[str, torch.Tensor]:
            if batch_size <= chunk_size:
                return self._forward_impl(
                    graph_embeddings=graph_embeddings,
                    sequence_embeddings=sequence_embeddings,
                    raw_embeddings=raw_embeddings,
                    graph_logits=graph_logits,
                    sequence_logits=sequence_logits,
                    raw_logits=raw_logits,
                    time_reliability=time_reliability,
                )

            output_chunks: list[dict[str, torch.Tensor]] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                output_chunks.append(
                    self._forward_impl(
                        graph_embeddings=graph_embeddings[start:end],
                        sequence_embeddings=sequence_embeddings[start:end],
                        raw_embeddings=raw_embeddings[start:end],
                        graph_logits=graph_logits[start:end],
                        sequence_logits=sequence_logits[start:end],
                        raw_logits=raw_logits[start:end],
                        time_reliability=_slice_optional_batch(time_reliability, start, end),
                    )
                )
            return _concat_tensor_dict(output_chunks)

        return _run_chunked_encoder_with_backoff(
            batch_size=batch_size,
            device=graph_embeddings.device,
            runner=_run,
            preferred_chunk_size=self.batch_chunk_size,
        )


class SharedPrivateFusion(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        context_dim: int,
        raw_dim: int,
        shared_dim: int,
        private_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_shared_proj = nn.Linear(graph_dim, shared_dim)
        self.context_shared_proj = nn.Linear(context_dim, shared_dim)
        self.raw_shared_proj = nn.Linear(raw_dim, shared_dim)
        self.graph_private_proj = nn.Linear(graph_dim, private_dim)
        self.context_private_proj = nn.Linear(context_dim, private_dim)
        self.raw_private_proj = nn.Linear(raw_dim, private_dim)
        self.shared_norm = nn.LayerNorm(shared_dim)
        self.conflict_detector = nn.Sequential(
            nn.Linear(shared_dim * 3 + private_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.shared_gate_head = nn.Sequential(
            nn.Linear(shared_dim * 3 + private_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_dim = shared_dim * 2 + private_dim * 4
        self.fusion_head = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def decompose(
        self,
        graph_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        raw_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph_shared = F.normalize(self.graph_shared_proj(graph_embeddings), p=2, dim=-1, eps=1e-6)
        context_shared = F.normalize(self.context_shared_proj(context_embeddings), p=2, dim=-1, eps=1e-6)
        raw_shared = F.normalize(self.raw_shared_proj(raw_embeddings), p=2, dim=-1, eps=1e-6)
        graph_private = torch.tanh(self.graph_private_proj(graph_embeddings))
        context_private = torch.tanh(self.context_private_proj(context_embeddings))
        raw_private = torch.tanh(self.raw_private_proj(raw_embeddings))
        shared_seed = (graph_shared + context_shared + raw_shared) / 3.0
        return {
            "graph_shared": graph_shared,
            "context_shared": context_shared,
            "raw_shared": raw_shared,
            "graph_private": graph_private,
            "context_private": context_private,
            "raw_private": raw_private,
            "shared_seed": shared_seed,
        }

    def forward_from_parts(
        self,
        fusion_parts: dict[str, torch.Tensor],
        prototype_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        graph_shared = fusion_parts["graph_shared"]
        context_shared = fusion_parts["context_shared"]
        raw_shared = fusion_parts["raw_shared"]
        graph_private = fusion_parts["graph_private"]
        context_private = fusion_parts["context_private"]
        raw_private = fusion_parts["raw_private"]
        shared_seed = fusion_parts["shared_seed"]
        shared_cosine_gc = F.cosine_similarity(graph_shared, context_shared, dim=-1).unsqueeze(-1)
        shared_cosine_gr = F.cosine_similarity(graph_shared, raw_shared, dim=-1).unsqueeze(-1)
        shared_cosine_cr = F.cosine_similarity(context_shared, raw_shared, dim=-1).unsqueeze(-1)
        private_cosine_gc = F.cosine_similarity(graph_private, context_private, dim=-1).unsqueeze(-1)
        private_cosine_gr = F.cosine_similarity(graph_private, raw_private, dim=-1).unsqueeze(-1)
        private_cosine_cr = F.cosine_similarity(context_private, raw_private, dim=-1).unsqueeze(-1)
        conflict_inputs = torch.cat(
            [
                graph_shared,
                context_shared,
                raw_shared,
                graph_private,
                context_private,
                raw_private,
                1.0 - (shared_cosine_gc + shared_cosine_gr + shared_cosine_cr) / 3.0,
                1.0 - (private_cosine_gc.abs() + private_cosine_gr.abs() + private_cosine_cr.abs()) / 3.0,
                0.5 * (
                    torch.abs(shared_cosine_gc - private_cosine_gc.abs())
                    + torch.abs(shared_cosine_gr - private_cosine_gr.abs())
                    + torch.abs(shared_cosine_cr - private_cosine_cr.abs())
                ),
            ],
            dim=-1,
        )
        conflict_score = torch.sigmoid(self.conflict_detector(conflict_inputs))
        shared_gate = torch.sigmoid(self.shared_gate_head(conflict_inputs))
        shared_gate = shared_gate * (1.0 - 0.50 * conflict_score)
        private_gate = (1.0 - 0.50 * shared_gate + 0.50 * conflict_score).clamp(min=0.0, max=1.0)
        if prototype_context is None:
            shared_consensus = self.shared_norm(shared_seed)
        else:
            shared_consensus = self.shared_norm(shared_seed + 0.50 * prototype_context)
        shared_consensus = shared_gate * shared_consensus
        shared_gap = (
            torch.abs(graph_shared - context_shared)
            + torch.abs(graph_shared - raw_shared)
            + torch.abs(context_shared - raw_shared)
        ) / 3.0
        private_interaction = private_gate * (
            graph_private * context_private
            + graph_private * raw_private
            + context_private * raw_private
        ) / 3.0
        fusion_features = torch.cat(
            [
                shared_consensus,
                (1.0 - 0.35 * conflict_score) * shared_gap,
                private_gate * graph_private,
                private_gate * context_private,
                private_gate * raw_private,
                private_interaction,
            ],
            dim=-1,
        )
        logits = self.fusion_head(fusion_features)
        return {
            **fusion_parts,
            "shared_consensus": shared_consensus,
            "shared_gap": shared_gap,
            "private_interaction": private_interaction,
            "shared_gate": shared_gate,
            "private_gate": private_gate,
            "conflict_score": conflict_score,
            "fusion_features": fusion_features,
            "logits": logits,
        }
