from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._legacy import (
    _balance_modality_embedding
)


class ForwardMixin:
    def _collect_diagnostic_tensors(
        self,
        graph,
        graph_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        raw_embeddings: torch.Tensor,
        fused_embeddings: torch.Tensor,
        aux_state: dict[str, Any],
        branch_outputs: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor | None]:
        sequence_mask = graph.ndata["sequence_mask"].bool() if "sequence_mask" in graph.ndata else None
        sequence_valid_ratio = None
        sequence_valid_length = None
        if sequence_mask is not None:
            sequence_valid_ratio = sequence_mask.float().mean(dim=1)
            sequence_valid_length = sequence_mask.float().sum(dim=1)
        graph_sequence_prob_gap = None
        if branch_outputs is not None and "graph_residual_logits" in branch_outputs and "sequence_residual_logits" in branch_outputs:
            graph_probs = F.softmax(branch_outputs["graph_residual_logits"], dim=-1)
            sequence_probs = F.softmax(branch_outputs["sequence_residual_logits"], dim=-1)
            positive_index = 1 if graph_probs.size(-1) > 1 else 0
            graph_sequence_prob_gap = graph_probs[:, positive_index] - sequence_probs[:, positive_index]
        return {
            "graph_embedding_norm": graph_embeddings.float().norm(dim=-1),
            "sequence_embedding_norm": sequence_embeddings.float().norm(dim=-1),
            "raw_embedding_norm": raw_embeddings.float().norm(dim=-1),
            "fused_embedding_norm": fused_embeddings.float().norm(dim=-1),
            "shared_gap": None if branch_outputs is None else branch_outputs.get("shared_gap"),
            "private_interaction": None if branch_outputs is None else branch_outputs.get("private_interaction"),
            "context_gate": aux_state.get("context_gate"),
            "graph_branch_gate": None if branch_outputs is None else branch_outputs.get("graph_branch_gate"),
            "graph_correction_support": None if branch_outputs is None else branch_outputs.get("graph_correction_support"),
            "sequence_branch_gate": None if branch_outputs is None else branch_outputs.get("sequence_branch_gate"),
            "raw_branch_gate": None if branch_outputs is None else branch_outputs.get("raw_branch_gate"),
            "fusion_delta_gate": None if branch_outputs is None else branch_outputs.get("fusion_delta_gate"),
            "delta_correction_support": None if branch_outputs is None else branch_outputs.get("delta_correction_support"),
            "shared_gate": None if branch_outputs is None else branch_outputs.get("shared_gate"),
            "private_gate": None if branch_outputs is None else branch_outputs.get("private_gate"),
            "conflict_score": None if branch_outputs is None else branch_outputs.get("conflict_score"),
            "time_reliability": aux_state.get("time_reliability"),
            "graph_temporal_gate": aux_state.get("graph_temporal_gate"),
            "wavelet_gate": aux_state.get("wavelet_gate"),
            "coassociation_gate": aux_state.get("coassociation_gate"),
            "coassociation_density": aux_state.get("coassociation_density"),
            "diffusion_gate": aux_state.get("diffusion_gate"),
            "diffusion_neighbor_strength": aux_state.get("diffusion_neighbor_strength"),
            "utg_temporal_gate": None
            if not isinstance(aux_state.get("utg_outputs"), dict)
            else aux_state.get("utg_outputs", {}).get("utg_temporal_gate"),
            "sequence_token_valid_ratio": sequence_valid_ratio,
            "sequence_valid_length": sequence_valid_length,
            "graph_sequence_prob_gap": graph_sequence_prob_gap,
        }
    def forward_with_branch_details(self, graph) -> dict[str, Any]:
        graph_embeddings, sequence_embeddings, raw_embeddings, fused_embeddings, edge_loss, aux_state = self.encode_embeddings(graph)
        branch_outputs: dict[str, torch.Tensor] | None = None
        if self.use_multimodal_fusion:
            branch_outputs = self._multimodal_branch_outputs(
                graph_embeddings=graph_embeddings,
                sequence_embeddings=sequence_embeddings,
                raw_embeddings=raw_embeddings,
                fused_embeddings=fused_embeddings,
                aux_state=aux_state,
            )
            logits = branch_outputs["logits"]
        else:
            logits = self.fusion(fused_embeddings)
            branch_outputs = {
                "logits": logits,
                "fusion_logits": logits,
            }
        return {
            "logits": logits,
            "edge_loss": edge_loss,
            "graph_embeddings": graph_embeddings,
            "sequence_embeddings": sequence_embeddings,
            "raw_embeddings": raw_embeddings,
            "fused_embeddings": fused_embeddings,
            "aux_state": aux_state,
            "branch_outputs": branch_outputs,
            "diagnostics": self._collect_diagnostic_tensors(
                graph=graph,
                graph_embeddings=graph_embeddings,
                sequence_embeddings=sequence_embeddings,
                raw_embeddings=raw_embeddings,
                fused_embeddings=fused_embeddings,
                aux_state=aux_state,
                branch_outputs=branch_outputs,
            ),
        }
    def forward(self, graph):
        return self.forward_with_branch_details(graph)["logits"]
    def _forward_consistency_view(self, graph) -> torch.Tensor:
        augmented_graph = graph.local_var()
        if "feature" in augmented_graph.ndata:
            augmented_features = augmented_graph.ndata["feature"]
            noise_scale = 0.01
            feature_dropout = min(max(self.modality_dropout_prob * 0.5, 0.0), 0.30)
            augmented_features = F.dropout(augmented_features, p=feature_dropout, training=self.training)
            if self.training:
                augmented_features = augmented_features + torch.randn_like(augmented_features) * noise_scale
            augmented_graph.ndata["feature"] = augmented_features
        if "typed_numeric" in augmented_graph.ndata:
            augmented_numeric = F.dropout(
                augmented_graph.ndata["typed_numeric"],
                p=min(max(self.modality_dropout_prob * 0.40, 0.0), 0.25),
                training=self.training,
            )
            if self.training:
                augmented_numeric = augmented_numeric + torch.randn_like(augmented_numeric) * 0.01
            augmented_graph.ndata["typed_numeric"] = augmented_numeric
        if "typed_categorical_frequency" in augmented_graph.ndata:
            augmented_frequency = F.dropout(
                augmented_graph.ndata["typed_categorical_frequency"],
                p=min(max(self.modality_dropout_prob * 0.35, 0.0), 0.20),
                training=self.training,
            )
            if self.training:
                augmented_frequency = augmented_frequency + torch.randn_like(augmented_frequency) * 0.005
            augmented_graph.ndata["typed_categorical_frequency"] = augmented_frequency
        if "sequence" in augmented_graph.ndata:
            augmented_sequence = augmented_graph.ndata["sequence"]
            sequence_dropout = min(max(self.modality_dropout_prob * 0.35, 0.0), 0.20)
            augmented_sequence = F.dropout(augmented_sequence, p=sequence_dropout, training=self.training)
            if self.training:
                augmented_sequence = augmented_sequence + torch.randn_like(augmented_sequence) * 0.005
            augmented_graph.ndata["sequence"] = augmented_sequence
        elif _has_lazy_relation_sequence_payload(augmented_graph):
            sequence_dropout = min(max(self.modality_dropout_prob * 0.35, 0.0), 0.20)
            for field_name in ("sequence_base_feature", "sequence_relation_mean_feature", "sequence_relation_max_feature", "sequence_global_context"):
                if field_name not in augmented_graph.ndata:
                    continue
                augmented_value = F.dropout(augmented_graph.ndata[field_name], p=sequence_dropout, training=self.training)
                if self.training:
                    augmented_value = augmented_value + torch.randn_like(augmented_value) * 0.005
                augmented_graph.ndata[field_name] = augmented_value
        if "event_sequence" in augmented_graph.ndata:
            augmented_event_sequence = augmented_graph.ndata["event_sequence"]
            event_dropout = min(max(self.modality_dropout_prob * 0.35, 0.0), 0.20)
            augmented_event_sequence = F.dropout(augmented_event_sequence, p=event_dropout, training=self.training)
            if self.training:
                augmented_event_sequence = augmented_event_sequence + torch.randn_like(augmented_event_sequence) * 0.005
            augmented_graph.ndata["event_sequence"] = augmented_event_sequence
        elif "event_history_indices" in augmented_graph.ndata:
            event_base_feature = self._resolve_event_base_feature_bank(augmented_graph)
            if event_base_feature is not None:
                event_dropout = min(max(self.modality_dropout_prob * 0.35, 0.0), 0.20)
                augmented_event_base = F.dropout(event_base_feature, p=event_dropout, training=self.training)
                if self.training:
                    augmented_event_base = augmented_event_base + torch.randn_like(augmented_event_base) * 0.005
                augmented_graph.ndata["event_base_feature"] = augmented_event_base
        if "event_time_deltas" in augmented_graph.ndata:
            augmented_event_deltas = augmented_graph.ndata["event_time_deltas"]
            if self.training:
                augmented_event_deltas = (
                    augmented_event_deltas + torch.randn_like(augmented_event_deltas) * 0.002
                ).clamp(min=0.0)
            augmented_graph.ndata["event_time_deltas"] = augmented_event_deltas
        if "temporal_context" in augmented_graph.ndata:
            augmented_temporal = augmented_graph.ndata["temporal_context"]
            temporal_dropout = min(max(self.modality_dropout_prob * 0.25, 0.0), 0.20)
            augmented_temporal = F.dropout(augmented_temporal, p=temporal_dropout, training=self.training)
            if self.training:
                augmented_temporal = augmented_temporal + torch.randn_like(augmented_temporal) * 0.005
            augmented_graph.ndata["temporal_context"] = augmented_temporal
        return self.forward_with_branch_details(augmented_graph)["logits"]
    def _multimodal_branch_outputs(
        self,
        graph_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        raw_embeddings: torch.Tensor,
        fused_embeddings: torch.Tensor,
        aux_state: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        balanced_graph_embeddings = aux_state.get("balanced_graph_embeddings")
        if balanced_graph_embeddings is None:
            balanced_graph_embeddings = _balance_modality_embedding(graph_embeddings)
        balanced_sequence_embeddings = aux_state.get("balanced_sequence_embeddings")
        if balanced_sequence_embeddings is None:
            balanced_sequence_embeddings = _balance_modality_embedding(sequence_embeddings)
        balanced_raw_embeddings = aux_state.get("balanced_raw_embeddings")
        if balanced_raw_embeddings is None:
            balanced_raw_embeddings = _balance_modality_embedding(raw_embeddings)
        graph_residual_logits = self.graph_residual_head(balanced_graph_embeddings)
        sequence_residual_logits = self.sequence_residual_head(balanced_sequence_embeddings)
        raw_branch_logits = self.raw_residual_head(balanced_raw_embeddings)
        stage_name = str(getattr(self, "active_training_stage", "joint_finetune")).lower()
        stage_force_graph_only = stage_name == "graph_warmup"
        stage_skip_uncertainty = stage_name in {"graph_warmup", "fusion_bootstrap"}
        if stage_force_graph_only:
            graph_branch_gate = torch.ones(
                (graph_residual_logits.size(0), 1),
                dtype=graph_residual_logits.dtype,
                device=graph_residual_logits.device,
            )
            zero_gate = torch.zeros_like(graph_branch_gate)
            return {
                "logits": graph_residual_logits,
                "raw_logits": raw_branch_logits,
                "raw_branch_logits": raw_branch_logits,
                "fusion_logits": graph_residual_logits,
                "graph_residual_logits": graph_residual_logits,
                "sequence_residual_logits": sequence_residual_logits,
                "uncertainty_logits": torch.zeros_like(graph_branch_gate),
                "graph_branch_gate": graph_branch_gate,
                "sequence_branch_gate": zero_gate,
                "raw_branch_gate": zero_gate,
                "fusion_delta_gate": zero_gate,
                "time_reliability": aux_state.get("time_reliability"),
            }
        pair_logits = self.raw_fusion(torch.cat([balanced_graph_embeddings, balanced_sequence_embeddings], dim=-1))
        fusion_outputs = dict(aux_state.get("fusion_outputs", {}) or {})
        if self.shared_private_fusion is not None and self.prototype_memory is not None:
            fusion_logits = fusion_outputs.get("logits", pair_logits)
            uncertainty_features = fusion_outputs.pop("fusion_features", fused_embeddings)
            graph_branch_gate = None
            raw_branch_gate = None
        elif self.tri_stream_fusion is not None:
            tristream_outputs = self.tri_stream_fusion(
                graph_embeddings=balanced_graph_embeddings,
                sequence_embeddings=balanced_sequence_embeddings,
                raw_embeddings=balanced_raw_embeddings,
                graph_logits=graph_residual_logits,
                sequence_logits=sequence_residual_logits,
                raw_logits=raw_branch_logits,
                time_reliability=aux_state.get("time_reliability"),
            )
            fusion_outputs.update(tristream_outputs)
            fusion_logits = fusion_outputs.get("logits", pair_logits)
            uncertainty_features = fusion_outputs.pop("fusion_features", fused_embeddings)
            graph_branch_gate = fusion_outputs.get("graph_branch_gate")
            raw_branch_gate = fusion_outputs.get("raw_branch_gate")
        elif self.graph_dominant_fusion is not None:
            graph_gate_logit_bias = (
                float(getattr(self.args, "graph_gate_logit_bias", 0.0))
                if self.training
                else float(getattr(self.args, "eval_graph_gate_logit_bias", getattr(self.args, "graph_gate_logit_bias", 0.0)))
            )
            residual_outputs = self.graph_dominant_fusion(
                graph_embeddings=balanced_graph_embeddings,
                context_embeddings=balanced_sequence_embeddings,
                graph_logits=graph_residual_logits,
                sequence_logits=sequence_residual_logits,
                graph_gate_logit_bias=graph_gate_logit_bias,
                graph_residual_min_gate=float(getattr(self.args, "graph_residual_min_gate", 0.0)),
                sequence_residual_scale=float(getattr(self.args, "sequence_residual_scale", 1.0)),
                fusion_delta_scale=float(self.fusion_delta_scale),
                force_graph_only=bool(stage_force_graph_only),
            )
            fusion_outputs.update(residual_outputs)
            fusion_logits = residual_outputs["logits"]
            uncertainty_features = fusion_outputs.pop("fusion_features", fused_embeddings)
            graph_branch_gate = residual_outputs["graph_branch_gate"]
            raw_branch_gate = None
        else:
            fusion_logits = graph_residual_logits if stage_force_graph_only else pair_logits
            uncertainty_features = fused_embeddings
            graph_branch_gate = None
            raw_branch_gate = None
        if self.uncertainty_head is not None and not stage_skip_uncertainty:
            uncertainty_logits = self.uncertainty_head(uncertainty_features)
        else:
            uncertainty_logits = torch.zeros(
                (fusion_logits.size(0), 1),
                dtype=fusion_logits.dtype,
                device=fusion_logits.device,
            )
        logits = pair_logits if self.legacy_fusion_only else fusion_logits
        return {
            "logits": logits,
            "raw_logits": pair_logits,
            "raw_branch_logits": raw_branch_logits,
            "fusion_logits": fusion_logits,
            "graph_residual_logits": graph_residual_logits,
            "sequence_residual_logits": sequence_residual_logits,
            "uncertainty_logits": uncertainty_logits,
            "graph_branch_gate": graph_branch_gate,
            "raw_branch_gate": raw_branch_gate,
            **fusion_outputs,
        }
    def forward_with_details(self, graph):
        payload = self.forward_with_branch_details(graph)
        return (
            payload["logits"],
            payload["edge_loss"],
            payload["graph_embeddings"],
            payload["sequence_embeddings"],
            payload["fused_embeddings"],
        )
