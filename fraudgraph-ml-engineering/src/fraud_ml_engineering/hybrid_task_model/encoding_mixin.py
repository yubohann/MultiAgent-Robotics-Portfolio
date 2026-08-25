from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._helpers import (
    _concat_tensor_dict,
    _slice_optional_batch
)
from ._legacy import (
    _balance_modality_embedding
)


class EncodingMixin:
    def _apply_modality_dropout(
        self,
        graph_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_prob <= 0.0:
            return graph_embeddings, sequence_embeddings
        batch_size = graph_embeddings.size(0)
        keep_probability = float(max(1.0 - self.modality_dropout_prob, 0.0))
        graph_keep = (torch.rand((batch_size, 1), device=graph_embeddings.device) < keep_probability).float()
        sequence_keep = (torch.rand((batch_size, 1), device=sequence_embeddings.device) < keep_probability).float()
        both_dropped = (graph_keep + sequence_keep).eq(0.0)
        graph_keep = torch.where(both_dropped, torch.ones_like(graph_keep), graph_keep)
        sequence_keep = torch.where(both_dropped, torch.ones_like(sequence_keep), sequence_keep)
        return graph_embeddings * graph_keep, sequence_embeddings * sequence_keep
    def _apply_tristream_modality_dropout(
        self,
        graph_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        raw_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_prob <= 0.0:
            return graph_embeddings, sequence_embeddings, raw_embeddings
        batch_size = graph_embeddings.size(0)
        keep_probability = float(max(1.0 - self.modality_dropout_prob, 0.0))
        graph_keep = (torch.rand((batch_size, 1), device=graph_embeddings.device) < keep_probability).float()
        sequence_keep = (torch.rand((batch_size, 1), device=sequence_embeddings.device) < keep_probability).float()
        raw_keep = (torch.rand((batch_size, 1), device=raw_embeddings.device) < keep_probability).float()
        all_dropped = (graph_keep + sequence_keep + raw_keep).eq(0.0)
        graph_keep = torch.where(all_dropped, torch.ones_like(graph_keep), graph_keep)
        sequence_keep = torch.where(all_dropped, torch.ones_like(sequence_keep), sequence_keep)
        raw_keep = torch.where(all_dropped, torch.ones_like(raw_keep), raw_keep)
        return graph_embeddings * graph_keep, sequence_embeddings * sequence_keep, raw_embeddings * raw_keep
    def _encode_raw_tabular_embeddings(
        self,
        graph,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.typed_tabular_encoder is None:
            raw_embeddings = (
                self.raw_anchor_encoder(features)
                if self.raw_anchor_encoder is not None
                else _balance_modality_embedding(features)
            )
            return raw_embeddings, {}
        raw_embeddings, typed_state = self.typed_tabular_encoder(
            graph.ndata["typed_numeric"],
            graph.ndata["typed_numeric_missing"],
            graph.ndata["typed_categorical"],
            graph.ndata["typed_categorical_missing"],
            graph.ndata["typed_categorical_frequency"],
        )
        return raw_embeddings, typed_state
    def _encode_relation_embeddings(
        self,
        graph,
        base_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
        if self.sequence_encoder is None:
            return None, None
        if "sequence" in graph.ndata:
            sequence_mask = graph.ndata["sequence_mask"] if "sequence_mask" in graph.ndata else None
            sequence_token_weights = graph.ndata["sequence_token_weights"] if "sequence_token_weights" in graph.ndata else None
            sequence_token_types = graph.ndata["sequence_token_types"] if "sequence_token_types" in graph.ndata else None
            sequence_relation_ids = graph.ndata["sequence_relation_ids"] if "sequence_relation_ids" in graph.ndata else None
            return self.sequence_encoder(
                graph.ndata["sequence"],
                token_mask=sequence_mask,
                token_weights=sequence_token_weights,
                token_types=sequence_token_types,
                relation_ids=sequence_relation_ids,
            )
        if not self.ieee_runtime_relation_sequence and not _has_lazy_relation_sequence_payload(graph):
            return None, None
        node_count = int(graph.num_nodes(graph.ntypes[0]))
        chunk_size = max(int(getattr(self.sequence_encoder, "batch_chunk_size", node_count)), 1)
        summary_chunks: list[torch.Tensor] = []
        detail_chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, node_count, chunk_size):
            end = min(start + chunk_size, node_count)
            (
                sequence_chunk,
                sequence_mask_chunk,
                sequence_token_weights_chunk,
                sequence_token_types_chunk,
                sequence_relation_ids_chunk,
            ) = self._gather_relation_sequence_chunk(graph, base_features, start, end)
            if sequence_chunk is None:
                return None, None
            summary_chunk, detail_chunk = self.sequence_encoder(
                sequence_chunk,
                token_mask=sequence_mask_chunk,
                token_weights=sequence_token_weights_chunk,
                token_types=sequence_token_types_chunk,
                relation_ids=sequence_relation_ids_chunk,
            )
            summary_chunks.append(summary_chunk)
            detail_chunks.append(detail_chunk)
        return torch.cat(summary_chunks, dim=0), _concat_tensor_dict(detail_chunks)
    def _encode_event_embeddings(
        self,
        graph,
        anchor_features: torch.Tensor,
        base_features: torch.Tensor | None,
        temporal_context_embeddings: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
        if self.event_encoder is None:
            return None, None
        if "event_sequence" in graph.ndata:
            event_mask = graph.ndata["event_mask"] if "event_mask" in graph.ndata else None
            event_time_deltas = graph.ndata["event_time_deltas"] if "event_time_deltas" in graph.ndata else None
            event_token_weights = graph.ndata["event_token_weights"] if "event_token_weights" in graph.ndata else None
            event_token_types = graph.ndata["event_token_types"] if "event_token_types" in graph.ndata else None
            return self.event_encoder(
                graph.ndata["event_sequence"],
                anchor_features=anchor_features,
                event_mask=event_mask,
                event_time_deltas=event_time_deltas,
                token_weights=event_token_weights,
                token_types=event_token_types,
                source_ids=graph.ndata["event_source_ids"] if "event_source_ids" in graph.ndata else None,
                temporal_context=temporal_context_embeddings,
            )
        if "event_history_indices" not in graph.ndata:
            return None, None
        batch_size = int(anchor_features.size(0))
        chunk_size = max(int(getattr(self.event_encoder, "batch_chunk_size", batch_size)), 1)
        summary_chunks: list[torch.Tensor] = []
        detail_chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            (
                event_sequence_chunk,
                event_mask_chunk,
                event_time_deltas_chunk,
                event_token_weights_chunk,
                event_token_types_chunk,
                event_source_ids_chunk,
            ) = self._gather_event_sequence_chunk(graph, base_features, start, end)
            if event_sequence_chunk is None:
                return None, None
            summary_chunk, detail_chunk = self.event_encoder(
                event_sequence_chunk,
                anchor_features=anchor_features[start:end],
                event_mask=event_mask_chunk,
                event_time_deltas=event_time_deltas_chunk,
                token_weights=event_token_weights_chunk,
                token_types=event_token_types_chunk,
                source_ids=event_source_ids_chunk,
                temporal_context=_slice_optional_batch(temporal_context_embeddings, start, end),
            )
            summary_chunks.append(summary_chunk)
            detail_chunks.append(detail_chunk)
        return torch.cat(summary_chunks, dim=0), _concat_tensor_dict(detail_chunks)
    def _encode_context_embeddings(
        self,
        graph,
        features: torch.Tensor,
        raw_embeddings: torch.Tensor,
        coassociation_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        device = features.device
        batch_size = features.size(0)
        context_embeddings = torch.zeros(batch_size, self.sequence_output_dim, device=device, dtype=features.dtype)
        relation_embeddings = None
        relation_details = None
        event_embeddings = None
        event_details = None
        context_gate = None
        temporal_context_embeddings = None
        time_reliability = None
        wavelet_embeddings = None
        wavelet_gate = None
        utg_outputs: dict[str, torch.Tensor] | None = None
        if self.temporal_context_encoder is not None and "temporal_context" in graph.ndata:
            temporal_context_embeddings, time_reliability = self.temporal_context_encoder(graph.ndata["temporal_context"])
        if self.wavelet_lite_head is not None and "wavelet_context" in graph.ndata:
            wavelet_embeddings, wavelet_gate = self.wavelet_lite_head(graph.ndata["wavelet_context"])
        if self.sequence_encoder is not None:
            relation_embeddings, relation_details = self._encode_relation_embeddings(graph, raw_embeddings)
        if self.event_encoder is not None:
            event_anchor_features = raw_embeddings if self.typed_tabular_encoder is not None else features
            event_embeddings, event_details = self._encode_event_embeddings(
                graph,
                event_anchor_features,
                self._resolve_event_base_feature_bank(graph),
                temporal_context_embeddings,
            )
        if self.utg_lite_fusion is not None and (
            relation_embeddings is not None
            or event_embeddings is not None
            or temporal_context_embeddings is not None
        ):
            context_embeddings, utg_outputs = self.utg_lite_fusion(
                relation_embeddings=relation_embeddings,
                event_embeddings=event_embeddings,
                temporal_embeddings=temporal_context_embeddings,
                time_reliability=time_reliability,
                wavelet_embeddings=wavelet_embeddings,
                coassociation_embeddings=coassociation_embeddings,
            )
            context_gate = utg_outputs.get("utg_temporal_gate")
        elif relation_embeddings is not None and event_embeddings is not None:
            balanced_relation = _balance_modality_embedding(relation_embeddings)
            balanced_event = _balance_modality_embedding(event_embeddings)
            embedding_delta = torch.abs(balanced_relation - balanced_event)
            embedding_interaction = balanced_relation * balanced_event
            embedding_similarity = F.cosine_similarity(balanced_relation, balanced_event, dim=-1).unsqueeze(-1)
            context_gate = torch.sigmoid(
                self.context_event_gate(
                    torch.cat(
                        [
                            balanced_relation,
                            balanced_event,
                            embedding_delta,
                            embedding_interaction,
                            embedding_similarity,
                        ],
                        dim=-1,
                    )
                )
            )
            context_embeddings = self.context_norm(
                context_gate * relation_embeddings
                + (1.0 - context_gate) * event_embeddings
                + 0.15 * embedding_delta
            )
        elif relation_embeddings is not None:
            context_embeddings = relation_embeddings
        elif event_embeddings is not None:
            context_embeddings = event_embeddings
        elif temporal_context_embeddings is not None:
            context_embeddings = temporal_context_embeddings
        elif wavelet_embeddings is not None:
            context_embeddings = wavelet_embeddings
        return context_embeddings, {
            "relation_embeddings": relation_embeddings,
            "relation_details": relation_details,
            "event_embeddings": event_embeddings,
            "event_details": event_details,
            "context_gate": context_gate,
            "temporal_context_embeddings": temporal_context_embeddings,
            "time_reliability": time_reliability,
            "wavelet_embeddings": wavelet_embeddings,
            "wavelet_gate": wavelet_gate,
            "coassociation_embeddings": coassociation_embeddings,
            "utg_outputs": utg_outputs,
        }
    def encode_embeddings(self, graph):
        features = graph.ndata["feature"]
        num_nodes = features.shape[0]
        device = features.device
        graph_embeddings = torch.zeros(num_nodes, self.graph_output_dim, device=device, dtype=features.dtype)
        sequence_embeddings = torch.zeros(num_nodes, self.sequence_output_dim, device=device, dtype=features.dtype)
        raw_embeddings, typed_tabular_state = self._encode_raw_tabular_embeddings(graph, features)
        edge_loss = torch.tensor(0.0, device=device, dtype=features.dtype)
        aux_state: dict[str, Any] = dict(typed_tabular_state)
        coassociation_embeddings = None
        if self.graph_encoder is not None:
            graph_embeddings, edge_loss = self.graph_encoder.forward_with_edge_loss(graph)
        if self.graph_diffusion_residual is not None:
            graph_embeddings, diffusion_state = self.graph_diffusion_residual(graph, graph_embeddings)
            aux_state.update(diffusion_state)
        if self.coassociation_encoder is not None:
            coassociation_embeddings, coassociation_state = self.coassociation_encoder(graph, graph_embeddings)
            aux_state.update(coassociation_state)
        if self.sequence_encoder is not None or self.event_encoder is not None or self.wavelet_lite_head is not None:
            sequence_embeddings, context_state = self._encode_context_embeddings(
                graph,
                features,
                raw_embeddings,
                coassociation_embeddings=coassociation_embeddings,
            )
            aux_state.update(context_state)
        elif self.temporal_context_encoder is not None and "temporal_context" in graph.ndata:
            temporal_context_embeddings, time_reliability = self.temporal_context_encoder(graph.ndata["temporal_context"])
            sequence_embeddings = temporal_context_embeddings
            aux_state["temporal_context_embeddings"] = temporal_context_embeddings
            aux_state["time_reliability"] = time_reliability
        temporal_context_embeddings = aux_state.get("temporal_context_embeddings")
        time_reliability = aux_state.get("time_reliability")
        if (
            self.graph_temporal_proj is not None
            and self.graph_temporal_gate is not None
            and temporal_context_embeddings is not None
        ):
            projected_temporal = self.graph_temporal_proj(temporal_context_embeddings)
            time_signal = (
                time_reliability
                if time_reliability is not None
                else torch.full((graph_embeddings.size(0), 1), 0.5, dtype=graph_embeddings.dtype, device=graph_embeddings.device)
            )
            graph_temporal_gate = torch.sigmoid(
                self.graph_temporal_gate(torch.cat([graph_embeddings, projected_temporal, time_signal], dim=-1))
            )
            graph_embeddings = graph_temporal_gate * graph_embeddings + (1.0 - graph_temporal_gate) * projected_temporal
            aux_state["graph_temporal_gate"] = graph_temporal_gate
        if self.use_multimodal_fusion:
            balanced_graph_embeddings = _balance_modality_embedding(graph_embeddings)
            balanced_sequence_embeddings = _balance_modality_embedding(sequence_embeddings)
            balanced_raw_embeddings = _balance_modality_embedding(raw_embeddings)
            (
                balanced_graph_embeddings,
                balanced_sequence_embeddings,
                balanced_raw_embeddings,
            ) = self._apply_tristream_modality_dropout(
                balanced_graph_embeddings,
                balanced_sequence_embeddings,
                balanced_raw_embeddings,
            )
            aux_state["balanced_graph_embeddings"] = balanced_graph_embeddings
            aux_state["balanced_sequence_embeddings"] = balanced_sequence_embeddings
            aux_state["balanced_raw_embeddings"] = balanced_raw_embeddings
        if self.use_multimodal_fusion and self.shared_private_fusion is not None and self.prototype_memory is not None:
            fusion_parts = self.shared_private_fusion.decompose(
                graph_embeddings=balanced_graph_embeddings,
                context_embeddings=balanced_sequence_embeddings,
                raw_embeddings=balanced_raw_embeddings,
            )
            relation_details = aux_state.get("relation_details")
            relation_summaries = (
                relation_details.get("capsule_summaries")
                if isinstance(relation_details, dict) and "capsule_summaries" in relation_details
                else None
            )
            prototype_outputs = self.prototype_memory(
                shared_seed=fusion_parts["shared_seed"],
                relation_summaries=relation_summaries,
                dataset_ids=graph.ndata["dataset_context_id"] if "dataset_context_id" in graph.ndata else None,
            )
            fusion_outputs = self.shared_private_fusion.forward_from_parts(
                fusion_parts,
                prototype_context=prototype_outputs["enhanced_shared"],
            )
            fused_embeddings = fusion_outputs["fusion_features"]
            aux_state["prototype_outputs"] = prototype_outputs
            aux_state["fusion_outputs"] = fusion_outputs
        elif self.use_multimodal_fusion and self.tri_stream_fusion is not None:
            fused_embeddings = torch.cat(
                [
                    balanced_graph_embeddings,
                    balanced_sequence_embeddings,
                    balanced_raw_embeddings,
                ],
                dim=-1,
            )
        elif self.use_multimodal_fusion:
            fused_embeddings = torch.cat([balanced_graph_embeddings, balanced_sequence_embeddings], dim=-1)
        elif self.graph_encoder is not None:
            fused_embeddings = _balance_modality_embedding(graph_embeddings)
        elif self.sequence_encoder is not None or self.event_encoder is not None:
            fused_embeddings = _balance_modality_embedding(sequence_embeddings)
        else:
            fused_embeddings = _balance_modality_embedding(
                self.feature_encoder(features) if self.feature_encoder is not None else raw_embeddings
            )
        aux_state["raw_embeddings"] = raw_embeddings
        return graph_embeddings, sequence_embeddings, raw_embeddings, fused_embeddings, edge_loss, aux_state
