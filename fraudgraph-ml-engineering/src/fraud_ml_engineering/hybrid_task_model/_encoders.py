from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ._helpers import (
    _align_module_input,
    _resolve_attention_heads,
    _run_chunked_forward_with_backoff,
    _safe_transformer_forward,
    _slice_optional_batch
)
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

class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim=model_dim, max_len=max_len)
        self.token_role_embedding = nn.Embedding(3, model_dim)
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
        self.summary_gate = nn.Sequential(
            nn.Linear(model_dim * 3 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def _token_role_ids(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        role_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        role_ids[:, 0] = 0
        if seq_len > 1:
            role_ids[:, -1] = 2
        return role_ids

    def _forward_impl(
        self,
        sequence_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(_align_module_input(sequence_tokens, self.input_proj))
        role_ids = self._token_role_ids(batch_size=x.size(0), seq_len=x.size(1), device=x.device)
        x = x + self.token_role_embedding(role_ids)
        if token_weights is None:
            token_weights = torch.ones((x.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        else:
            token_weights = token_weights.to(device=x.device, dtype=x.dtype)
        if token_mask is not None:
            token_weights = token_weights * token_mask.to(dtype=x.dtype)
        x = x + torch.log1p(token_weights.clamp(min=0.0)).unsqueeze(-1)
        x = self.position_encoding(x)
        x = self.input_norm(x)
        padding_mask = None
        if token_mask is not None:
            padding_mask = ~token_mask.bool()
        x = _safe_transformer_forward(self.transformer, x, padding_mask=padding_mask)
        x = self.norm(x)
        if token_mask is None:
            valid_mask = torch.ones((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)
        else:
            valid_mask = token_mask.bool()
        # Keep the self token as an anchor, but let a weighted attention pool
        # recover informative relation context for denser graphs such as
        # Archive / IEEE variants.
        scores = self.pool_score(x).squeeze(-1)
        scores = scores + torch.log(token_weights.clamp(min=1e-4))
        scores = scores.masked_fill(~valid_mask, -1e4)
        pooled_attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled_context = (pooled_attention * x).sum(dim=1)
        self_token = x[:, 0, :]
        valid_lengths = valid_mask.long().sum(dim=1).clamp(min=1) - 1
        global_token = x[torch.arange(x.size(0), device=x.device), valid_lengths]
        self_context_cosine = F.cosine_similarity(self_token, pooled_context, dim=-1).unsqueeze(-1)
        summary_gate = torch.sigmoid(
            self.summary_gate(torch.cat([self_token, pooled_context, global_token, self_context_cosine], dim=-1))
        )
        summary = summary_gate * self_token + (1.0 - summary_gate) * pooled_context + 0.15 * global_token
        summary = self.dropout(summary)
        return self.output_norm(summary)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(sequence_tokens.size(0))

        def _run(chunk_size: int) -> torch.Tensor:
            if batch_size <= chunk_size:
                return self._forward_impl(
                    sequence_tokens,
                    token_mask=token_mask,
                    token_weights=token_weights,
                )

            summary_chunks: list[torch.Tensor] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                summary_chunks.append(
                    self._forward_impl(
                        sequence_tokens[start:end],
                        token_mask=_slice_optional_batch(token_mask, start, end),
                        token_weights=_slice_optional_batch(token_weights, start, end),
                    )
                )
            return torch.cat(summary_chunks, dim=0)

        return _run_chunked_forward_with_backoff(
            batch_size=batch_size,
            device=sequence_tokens.device,
            runner=_run,
        )
