from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._helpers import (
    _slice_optional_batch
)


class SequenceMixin:
    def _resolve_event_base_feature_bank(self, graph) -> torch.Tensor | None:
        if "event_base_feature" in graph.ndata:
            return graph.ndata["event_base_feature"]
        if "feature" in graph.ndata:
            return graph.ndata["feature"]
        return None
    def _runtime_relation_order(self, graph) -> list[str]:
        relation_order = [relation for relation in graph.etypes if relation not in {"homo", "self_loop"}]
        if "sequence_relation_degree" in graph.ndata and graph.ndata["sequence_relation_degree"].ndim == 2:
            relation_order = relation_order[: int(graph.ndata["sequence_relation_degree"].shape[1])]
        return relation_order
    def _gather_dynamic_relation_sequence_chunk(
        self,
        graph,
        base_features: torch.Tensor,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_order = self._runtime_relation_order(graph)
        base_chunk = base_features[start:end]
        batch_size = int(base_chunk.size(0))
        feature_dim = int(base_chunk.size(1))
        relation_degree = graph.ndata["sequence_relation_degree"][start:end].float()
        relation_topk_indices = graph.ndata["sequence_relation_topk_indices"][start:end].long()
        relation_count = min(len(relation_order), int(relation_degree.shape[1]) if relation_degree.ndim == 2 else 0)
        sequence_length = 2 + relation_count * len(self.runtime_relation_token_order)
        sequence_tokens = torch.zeros(
            (batch_size, sequence_length, feature_dim + 8),
            dtype=base_chunk.dtype,
            device=base_chunk.device,
        )
        sequence_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=base_chunk.device)
        sequence_token_weights = torch.zeros((batch_size, sequence_length), dtype=base_chunk.dtype, device=base_chunk.device)
        sequence_token_types = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=base_chunk.device)
        sequence_relation_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=base_chunk.device)
        ones = torch.ones((batch_size, 1), dtype=base_chunk.dtype, device=base_chunk.device)
        zeros = torch.zeros((batch_size, 1), dtype=base_chunk.dtype, device=base_chunk.device)
        if relation_count <= 0:
            sequence_tokens[:, 0, :-8] = base_chunk
            sequence_tokens[:, sequence_length - 1, :-8] = base_chunk
            sequence_mask[:, 0] = True
            sequence_mask[:, sequence_length - 1] = True
            sequence_token_weights[:, 0] = 1.0
            sequence_token_weights[:, sequence_length - 1] = 1.0
            sequence_token_types[:, sequence_length - 1] = 4
            return sequence_tokens, sequence_mask, sequence_token_weights, sequence_token_types, sequence_relation_ids
        degree_strength = torch.log1p(relation_degree[:, :relation_count]).unsqueeze(-1)
        strength_denominator = degree_strength.sum(dim=1).clamp(min=1e-6)
        max_strength = degree_strength.amax(dim=1).clamp(min=1e-6)
        weighted_context = torch.zeros_like(base_chunk)
        has_relation_context = relation_degree[:, :relation_count].gt(0).any(dim=1, keepdim=True)
        for relation_index in range(relation_count):
            topk_indices = relation_topk_indices[:, relation_index, :].clamp(min=0)
            topk_valid = relation_topk_indices[:, relation_index, :].ge(0)
            gathered_neighbors = base_features.index_select(0, topk_indices.reshape(-1)).reshape(
                batch_size,
                relation_topk_indices.size(2),
                feature_dim,
            )
            gathered_neighbors = gathered_neighbors * topk_valid.unsqueeze(-1).to(dtype=gathered_neighbors.dtype)
            neighbor_count = topk_valid.sum(dim=1, keepdim=True).clamp(min=1)
            local_feature = gathered_neighbors.sum(dim=1) / neighbor_count.to(dtype=gathered_neighbors.dtype)
            masked_neighbors = gathered_neighbors.masked_fill(~topk_valid.unsqueeze(-1), -1e4)
            motif_feature = masked_neighbors.max(dim=1).values
            has_neighbor = topk_valid.any(dim=1, keepdim=True)
            local_feature = torch.where(has_neighbor, local_feature, base_chunk)
            motif_feature = torch.where(has_neighbor, motif_feature, base_chunk)
            relation_degree_column = relation_degree[:, relation_index]
            relation_mask = relation_degree_column.gt(0).unsqueeze(-1)
            degree_strength_column = torch.log1p(relation_degree_column).unsqueeze(-1)
            relation_strength = torch.where(
                relation_mask,
                degree_strength_column / strength_denominator,
                torch.zeros_like(degree_strength_column),
            )
            relation_reliability = torch.where(
                relation_mask,
                0.55 * relation_strength + 0.45 * (degree_strength_column / max_strength),
                torch.zeros_like(degree_strength_column),
            )
            relation_delta = local_feature - base_chunk
            role_delta = motif_feature - local_feature
            local_shift_norm = torch.norm(relation_delta.float(), dim=-1, keepdim=True).div(max(feature_dim, 1) ** 0.5)
            role_shift_norm = torch.norm(role_delta.float(), dim=-1, keepdim=True).div(max(feature_dim, 1) ** 0.5)
            reliability_feature = (
                relation_reliability * local_feature
                + (1.0 - relation_reliability)
                * (0.60 * base_chunk + 0.40 * (motif_feature + 0.50 * relation_delta))
            )
            weighted_context = weighted_context + (
                local_feature * relation_reliability
                + 0.45 * motif_feature * relation_strength
                + 0.55 * reliability_feature * relation_reliability
            )
            relation_rank = float(relation_index + 1) / float(max(relation_count, 1))
            relation_presence = relation_mask.float()
            for token_name_index, token_name in enumerate(self.runtime_relation_token_order):
                slot_index = 1 + relation_index * len(self.runtime_relation_token_order) + token_name_index
                if token_name == "local":
                    token_stack = local_feature
                    token_type_id = 1
                    token_weight = torch.where(relation_mask, 1.0 + relation_strength, torch.zeros_like(relation_strength))
                elif token_name == "motif":
                    token_stack = motif_feature
                    token_type_id = 2
                    token_weight = torch.where(
                        relation_mask,
                        1.0 + 0.5 * (relation_strength + relation_reliability),
                        torch.zeros_like(relation_strength),
                    )
                else:
                    token_stack = reliability_feature
                    token_type_id = 3
                    token_weight = torch.where(
                        relation_mask,
                        1.0 + relation_reliability,
                        torch.zeros_like(relation_strength),
                    )
                semantic_channels = torch.cat(
                    [
                        torch.full_like(relation_strength, float(token_type_id) / 4.0),
                        torch.full_like(relation_strength, float(relation_rank)),
                        relation_strength,
                        relation_reliability,
                        relation_presence,
                        local_shift_norm.to(dtype=base_chunk.dtype),
                        role_shift_norm.to(dtype=base_chunk.dtype),
                        torch.full_like(relation_strength, float(token_name_index + 1) / float(len(self.runtime_relation_token_order) + 1)),
                    ],
                    dim=-1,
                )
                sequence_tokens[:, slot_index, :] = torch.cat([token_stack, semantic_channels.to(dtype=token_stack.dtype)], dim=-1)
                sequence_mask[:, slot_index] = relation_mask.squeeze(-1)
                sequence_token_weights[:, slot_index] = token_weight.squeeze(-1)
                sequence_token_types[:, slot_index] = int(token_type_id)
                sequence_relation_ids[:, slot_index] = int(relation_index + 1)
        global_context = torch.where(has_relation_context, weighted_context, base_chunk)
        sequence_tokens[:, 0, :] = torch.cat(
            [
                base_chunk + 0.16 * (global_context - base_chunk),
                torch.cat([zeros, zeros, ones, ones, ones, zeros, zeros, zeros], dim=-1).to(dtype=base_chunk.dtype),
            ],
            dim=-1,
        )
        sequence_tokens[:, sequence_length - 1, :] = torch.cat(
            [
                global_context,
                torch.cat([ones, ones, ones, ones, ones, zeros, zeros, ones], dim=-1).to(dtype=base_chunk.dtype),
            ],
            dim=-1,
        )
        sequence_mask[:, 0] = True
        sequence_mask[:, sequence_length - 1] = True
        sequence_token_weights[:, 0] = 1.0
        sequence_token_weights[:, sequence_length - 1] = 1.0
        sequence_token_types[:, 0] = 0
        sequence_token_types[:, sequence_length - 1] = 4
        return sequence_tokens, sequence_mask, sequence_token_weights, sequence_token_types, sequence_relation_ids
    def _gather_relation_sequence_chunk(
        self,
        graph,
        base_features: torch.Tensor | None,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.ieee_runtime_relation_sequence and base_features is not None:
            return self._gather_dynamic_relation_sequence_chunk(graph, base_features, start, end)
        if "sequence" in graph.ndata:
            return (
                graph.ndata["sequence"][start:end],
                _slice_optional_batch(graph.ndata["sequence_mask"] if "sequence_mask" in graph.ndata else None, start, end),
                _slice_optional_batch(
                    graph.ndata["sequence_token_weights"] if "sequence_token_weights" in graph.ndata else None,
                    start,
                    end,
                ),
                _slice_optional_batch(
                    graph.ndata["sequence_token_types"] if "sequence_token_types" in graph.ndata else None,
                    start,
                    end,
                ),
                _slice_optional_batch(
                    graph.ndata["sequence_relation_ids"] if "sequence_relation_ids" in graph.ndata else None,
                    start,
                    end,
                ),
            )
        if not _has_lazy_relation_sequence_payload(graph):
            return None, None, None, None, None
        return _materialize_relation_sequence_chunk(
            graph,
            dataset_name=str(getattr(self.args, "dataset", "")).strip().lower(),
            start=start,
            end=end,
        )
    def _gather_event_sequence_chunk(
        self,
        graph,
        base_features: torch.Tensor | None,
        start: int,
        end: int,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if "event_sequence" in graph.ndata:
            return (
                graph.ndata["event_sequence"][start:end],
                _slice_optional_batch(graph.ndata["event_mask"] if "event_mask" in graph.ndata else None, start, end),
                _slice_optional_batch(
                    graph.ndata["event_time_deltas"] if "event_time_deltas" in graph.ndata else None,
                    start,
                    end,
                ),
                _slice_optional_batch(
                    graph.ndata["event_token_weights"] if "event_token_weights" in graph.ndata else None,
                    start,
                    end,
                ),
                _slice_optional_batch(
                    graph.ndata["event_token_types"] if "event_token_types" in graph.ndata else None,
                    start,
                    end,
                ),
                _slice_optional_batch(
                    graph.ndata["event_source_ids"] if "event_source_ids" in graph.ndata else None,
                    start,
                    end,
                ),
            )
        if "event_history_indices" not in graph.ndata:
            return None, None, None, None, None, None
        base_features = base_features if base_features is not None else self._resolve_event_base_feature_bank(graph)
        if base_features is None:
            return None, None, None, None, None, None
        history_indices = graph.ndata["event_history_indices"][start:end].long()
        event_mask = (
            graph.ndata["event_mask"][start:end].bool()
            if "event_mask" in graph.ndata
            else history_indices.ge(0)
        )
        clamped_indices = history_indices.clamp(min=0)
        flat_indices = clamped_indices.reshape(-1)
        gathered = base_features.index_select(0, flat_indices).reshape(
            history_indices.size(0),
            history_indices.size(1),
            base_features.size(-1),
        )
        gathered = gathered * event_mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return (
            gathered,
            event_mask,
            _slice_optional_batch(
                graph.ndata["event_time_deltas"] if "event_time_deltas" in graph.ndata else None,
                start,
                end,
            ),
            _slice_optional_batch(
                graph.ndata["event_token_weights"] if "event_token_weights" in graph.ndata else None,
                start,
                end,
            ),
            _slice_optional_batch(
                graph.ndata["event_token_types"] if "event_token_types" in graph.ndata else None,
                start,
                end,
            ),
            _slice_optional_batch(
                graph.ndata["event_source_ids"] if "event_source_ids" in graph.ndata else None,
                start,
                end,
            ),
        )
