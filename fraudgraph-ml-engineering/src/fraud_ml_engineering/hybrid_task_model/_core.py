from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ._helpers import TRANSFORMER_BATCH_CHUNK_SIZE


class HybridFraudModelCore(nn.Module):
    def __init__(self, args, graph):
        super().__init__()
        self.args = args
        self.enable_gnn = not bool(getattr(args, "disable_gnn", False))
        self.enable_transformer = not bool(getattr(args, "disable_transformer", False))
        self.disable_relation_sequence_encoder = bool(getattr(args, "disable_relation_sequence_encoder", False))
        self.disable_event_transformer_encoder = bool(getattr(args, "disable_event_transformer_encoder", False))
        self.disable_temporal_context_encoder = bool(getattr(args, "disable_temporal_context_encoder", False))
        self.disable_graph_temporal_fusion = bool(getattr(args, "disable_graph_temporal_fusion", False))
        self.node_type = graph.ntypes[0]
        self.dropout_rate = float(getattr(args, "dropout", 0.1))
        input_feature_dim = int(graph.nodes[self.node_type].data["feature"].shape[1])
        self.graph_encoder = SplitGNNEncoder(args, graph) if self.enable_gnn else None
        self.use_multimodal_fusion = self.enable_gnn and self.enable_transformer
        dataset_name = str(getattr(args, "dataset", "")).strip().lower()
        requested_fusion_variant = getattr(args, "fusion_variant", None)
        self.fusion_variant = str(
            requested_fusion_variant
            if requested_fusion_variant is not None
            else (
                "graph_dominant_residual"
                if self.use_multimodal_fusion and dataset_name == "elliptic"
                else "tri_stream_gate"
                if self.use_multimodal_fusion
                else "single_branch"
            )
        ).lower()
        self.modality_dropout_prob = float(getattr(args, "modality_dropout_prob", 0.0))
        self.fusion_delta_scale = float(getattr(args, "fusion_delta_scale", 0.35))
        self.active_training_stage = str(getattr(args, "active_training_stage", "joint_finetune")).lower()
        def _resolve_optional_chunk_size(value: Any, fallback: int) -> int:
            return max(int(fallback if value is None else value), 1)
        default_transformer_batch_chunk_size = 1_024 if dataset_name == "ieee" else TRANSFORMER_BATCH_CHUNK_SIZE
        transformer_batch_chunk_size = getattr(args, "transformer_batch_chunk_size", None)
        if transformer_batch_chunk_size is not None:
            default_transformer_batch_chunk_size = max(int(transformer_batch_chunk_size), 1)
        sequence_batch_chunk_size = _resolve_optional_chunk_size(
            getattr(args, "sequence_batch_chunk_size", default_transformer_batch_chunk_size),
            default_transformer_batch_chunk_size,
        )
        event_batch_chunk_size = _resolve_optional_chunk_size(
            getattr(args, "event_batch_chunk_size", default_transformer_batch_chunk_size),
            default_transformer_batch_chunk_size,
        )
        fusion_batch_chunk_size = max(
            min(sequence_batch_chunk_size, event_batch_chunk_size, default_transformer_batch_chunk_size),
            1,
        )
        transformer_activation_checkpointing = bool(getattr(args, "transformer_activation_checkpointing", True))
        transformer_hidden_dim = int(getattr(args, "transformer_hidden_dim", getattr(args, "seq_hidden_dim", 64)))
        transformer_num_layers = int(getattr(args, "transformer_num_layers", 1))
        fusion_hidden_dim = int(getattr(args, "fusion_hidden_dim", 64))
        multimodal_hidden_dim = int(fusion_hidden_dim)
        private_dim = int(getattr(args, "shared_private_dim", multimodal_hidden_dim))
        feature_hidden_dim = int(getattr(args, "feature_hidden_dim", max(fusion_hidden_dim, transformer_hidden_dim)))
        raw_anchor_dim = int(getattr(args, "raw_anchor_dim", transformer_hidden_dim))
        typed_numeric = graph.ndata["typed_numeric"] if "typed_numeric" in graph.ndata else None
        typed_numeric_missing = graph.ndata["typed_numeric_missing"] if "typed_numeric_missing" in graph.ndata else None
        typed_categorical = graph.ndata["typed_categorical"] if "typed_categorical" in graph.ndata else None
        typed_categorical_missing = (
            graph.ndata["typed_categorical_missing"] if "typed_categorical_missing" in graph.ndata else None
        )
        typed_categorical_frequency = (
            graph.ndata["typed_categorical_frequency"] if "typed_categorical_frequency" in graph.ndata else None
        )
        typed_categorical_cardinalities = []
        if typed_categorical is not None and typed_categorical.ndim == 2 and typed_categorical.shape[1] > 0:
            typed_categorical_cardinalities = [
                int(typed_categorical[:, column_index].max().item())
                for column_index in range(int(typed_categorical.shape[1]))
            ]
        self.typed_tabular_encoder = (
            TypedTabularEncoder(
                numeric_dim=int(typed_numeric.shape[1]) if typed_numeric is not None and typed_numeric.ndim == 2 else 0,
                categorical_cardinalities=typed_categorical_cardinalities,
                output_dim=raw_anchor_dim,
                hidden_dim=feature_hidden_dim,
                dropout=self.dropout_rate,
            )
            if (
                typed_numeric is not None
                and typed_numeric_missing is not None
                and typed_categorical is not None
                and typed_categorical_missing is not None
                and typed_categorical_frequency is not None
            )
            else None
        )
        self.runtime_relation_token_order = ("local", "motif", "reliability")
        sequence_tokens = graph.ndata["sequence"] if "sequence" in graph.ndata else None
        sequence_base_features = None
        if "sequence_base_feature" in graph.ndata:
            sequence_base_features = graph.ndata["sequence_base_feature"]
        elif "feature" in graph.ndata:
            sequence_base_features = graph.ndata["feature"]
        has_lazy_relation_inputs = _has_lazy_relation_sequence_payload(graph)
        has_dynamic_relation_inputs = (
            dataset_name == "ieee"
            and "sequence_relation_topk_indices" in graph.ndata
            and "sequence_relation_degree" in graph.ndata
        )
        shared_tabular_dim = raw_anchor_dim if self.typed_tabular_encoder is not None else input_feature_dim
        sequence_dim = (
            int(sequence_tokens.shape[-1])
            if sequence_tokens is not None
            else int(shared_tabular_dim + 8)
            if has_dynamic_relation_inputs
            else int(sequence_base_features.shape[-1]) + 8
            if has_lazy_relation_inputs and sequence_base_features is not None
            else input_feature_dim
        )
        sequence_len = (
            int(sequence_tokens.shape[1])
            if sequence_tokens is not None and sequence_tokens.ndim >= 2
            else int(2 + graph.ndata["sequence_relation_degree"].shape[1] * len(self.runtime_relation_token_order))
            if has_dynamic_relation_inputs and graph.ndata["sequence_relation_degree"].ndim >= 2
            else int(graph.ndata["sequence_mask"].shape[1])
            if has_lazy_relation_inputs and "sequence_mask" in graph.ndata and graph.ndata["sequence_mask"].ndim >= 2
            else 1
        )
        self.ieee_runtime_relation_sequence = bool(
            has_dynamic_relation_inputs and sequence_tokens is None and not has_lazy_relation_inputs
        )
        self.sequence_encoder = (
            RelationCapsuleSequenceEncoder(
                input_dim=sequence_dim,
                model_dim=transformer_hidden_dim,
                relation_vocab_size=(
                    int(graph.ndata["sequence_relation_ids"].max().item()) + 1
                    if "sequence_relation_ids" in graph.ndata
                    else max(sequence_len, 1)
                ),
                token_type_count=(
                    int(graph.ndata["sequence_token_types"].max().item()) + 1
                    if "sequence_token_types" in graph.ndata
                    else 5
                ),
                num_layers=transformer_num_layers,
                dropout=self.dropout_rate,
                max_len=max(sequence_len, 8),
                batch_chunk_size=sequence_batch_chunk_size,
                activation_checkpointing=transformer_activation_checkpointing,
            )
            if (
                self.enable_transformer
                and not self.disable_relation_sequence_encoder
                and (sequence_tokens is not None or has_lazy_relation_inputs or self.ieee_runtime_relation_sequence)
            )
            else None
        )
        self.event_encoder = None
        self.temporal_context_encoder = None
        self.graph_temporal_proj = None
        self.graph_temporal_gate = None
        self.wavelet_lite_head = None
        self.utg_lite_fusion = None
        self.coassociation_encoder = None
        self.graph_diffusion_residual = None
        temporal_context_dim = int(graph.ndata["temporal_context"].shape[1]) if "temporal_context" in graph.ndata else 0
        wavelet_context_dim = int(graph.ndata["wavelet_context"].shape[1]) if "wavelet_context" in graph.ndata else 0
        event_sequence = graph.ndata["event_sequence"] if "event_sequence" in graph.ndata else None
        event_history_indices = graph.ndata["event_history_indices"] if "event_history_indices" in graph.ndata else None
        event_base_features = None
        if "event_base_feature" in graph.ndata:
            event_base_features = graph.ndata["event_base_feature"]
        elif event_history_indices is not None and "feature" in graph.ndata:
            event_base_features = graph.ndata["feature"]
        has_indexed_event_inputs = event_history_indices is not None and (
            event_base_features is not None or self.typed_tabular_encoder is not None
        )
        if (
            self.enable_transformer
            and not self.disable_event_transformer_encoder
            and (event_sequence is not None or has_indexed_event_inputs)
        ):
            event_dim = (
                int(event_sequence.shape[-1])
                if event_sequence is not None
                else int(shared_tabular_dim)
                if self.typed_tabular_encoder is not None
                else int(event_base_features.shape[-1])
            )
            event_len = (
                int(event_sequence.shape[1])
                if event_sequence is not None and event_sequence.ndim >= 2
                else int(event_history_indices.shape[1])
                if event_history_indices is not None and event_history_indices.ndim >= 2
                else 1
            )
            self.event_encoder = EventTransformerEncoder(
                input_dim=event_dim,
                anchor_dim=shared_tabular_dim,
                model_dim=transformer_hidden_dim,
                num_layers=transformer_num_layers,
                dropout=self.dropout_rate,
                max_len=max(event_len, 8),
                event_type_count=(
                    int(graph.ndata["event_token_types"].max().item()) + 1
                    if "event_token_types" in graph.ndata
                    else 4
                ),
                source_count=(
                    int(graph.ndata["event_source_ids"].max().item()) + 1
                    if "event_source_ids" in graph.ndata
                    else 6
                ),
                batch_chunk_size=event_batch_chunk_size,
                activation_checkpointing=transformer_activation_checkpointing,
            )
        if temporal_context_dim > 0 and not self.disable_temporal_context_encoder:
            self.temporal_context_encoder = TemporalContextAggregator(
                input_dim=temporal_context_dim,
                model_dim=transformer_hidden_dim,
                dropout=self.dropout_rate,
            )
        if wavelet_context_dim > 0 and bool(getattr(args, "wavelet_lite_enabled", dataset_name == "elliptic")):
            self.wavelet_lite_head = WaveletLiteHead(
                input_dim=wavelet_context_dim,
                model_dim=transformer_hidden_dim,
                dropout=self.dropout_rate,
            )
        self.context_event_gate = None
        self.context_norm = nn.LayerNorm(transformer_hidden_dim)
        if bool(getattr(args, "utg_lite_enabled", dataset_name == "elliptic")) and self.enable_transformer:
            self.utg_lite_fusion = UTGLiteTemporalFusion(
                model_dim=transformer_hidden_dim,
                dropout=self.dropout_rate,
            )
        if self.sequence_encoder is not None and self.event_encoder is not None:
            self.context_event_gate = nn.Sequential(
                nn.Linear(transformer_hidden_dim * 4 + 1, transformer_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(transformer_hidden_dim, 1),
            )
        graph_dim = self.graph_encoder.output_dim if self.graph_encoder is not None else int(getattr(args, "intra_dim", 8))
        diffusion_relation_names = [
            relation_name
            for relation_name in graph.etypes
            if relation_name not in {"homo", "self_loop"}
        ]
        if self.graph_encoder is not None and bool(getattr(args, "diffusion_residual_enabled", dataset_name == "elliptic")):
            self.graph_diffusion_residual = ParameterizedDiffusionResidual(
                input_dim=graph_dim,
                relation_names=diffusion_relation_names,
                hidden_dim=max(graph_dim, fusion_hidden_dim),
                dropout=self.dropout_rate,
                residual_scale=float(getattr(args, "diffusion_residual_scale", 0.18 if dataset_name == "elliptic" else 0.0)),
            )
        if (
            self.graph_encoder is not None
            and "coassociation" in graph.etypes
            and bool(getattr(args, "coassociation_enabled", dataset_name == "elliptic"))
        ):
            self.coassociation_encoder = CoAssociationEncoder(
                input_dim=graph_dim,
                output_dim=transformer_hidden_dim,
                hidden_dim=max(transformer_hidden_dim, fusion_hidden_dim),
                dropout=self.dropout_rate,
            )
        if (
            self.temporal_context_encoder is not None
            and self.graph_encoder is not None
            and not self.disable_graph_temporal_fusion
        ):
            self.graph_temporal_proj = nn.Sequential(
                nn.Linear(transformer_hidden_dim, graph_dim),
                nn.LayerNorm(graph_dim),
            )
            self.graph_temporal_gate = nn.Sequential(
                nn.Linear(graph_dim * 2 + 1, graph_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(graph_dim, 1),
            )
        seq_dim = transformer_hidden_dim
        self.feature_encoder = None
        self.raw_anchor_encoder = None
        self.graph_residual_head = None
        self.sequence_residual_head = None
        self.raw_residual_head = None
        self.raw_fusion = None
        self.prototype_memory = None
        self.prototype_reliability_scorer = None
        self.shared_private_fusion = None
        self.graph_dominant_fusion = None
        self.tri_stream_fusion = None
        self.uncertainty_head = None
        def _build_branch_head(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_hidden_dim, args.n_class),
            )
        if not self.enable_gnn and not self.enable_transformer:
            if self.typed_tabular_encoder is None:
                self.feature_encoder = nn.Sequential(
                    nn.Linear(input_feature_dim, feature_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                )
                fusion_input_dim = feature_hidden_dim
            else:
                fusion_input_dim = raw_anchor_dim
        elif self.use_multimodal_fusion:
            if self.typed_tabular_encoder is None:
                self.raw_anchor_encoder = RawFeatureAnchorEncoder(
                    input_dim=input_feature_dim,
                    hidden_dim=feature_hidden_dim,
                    output_dim=raw_anchor_dim,
                    dropout=self.dropout_rate,
                )
            self.graph_residual_head = _build_branch_head(graph_dim)
            self.sequence_residual_head = _build_branch_head(seq_dim)
            self.raw_residual_head = _build_branch_head(raw_anchor_dim)
            self.raw_fusion = nn.Sequential(
                nn.Linear(graph_dim + seq_dim, fusion_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_hidden_dim, args.n_class),
            )
            if self.fusion_variant == "shared_private_prototype":
                self.prototype_memory = PrototypeMemoryBank(
                    shared_dim=multimodal_hidden_dim,
                    num_classes=args.n_class,
                    num_datasets=(
                        int(graph.ndata["dataset_context_id"].max().item()) + 1
                        if "dataset_context_id" in graph.ndata
                        else 1
                    ),
                    relation_dim=seq_dim,
                    fraud_subtype_count=int(getattr(args, "fraud_subtype_count", 3)),
                    dropout=self.dropout_rate,
                )
                self.prototype_reliability_scorer = PrototypeReliabilityScorer(
                    num_classes=args.n_class,
                    hidden_dim=max(fusion_hidden_dim // 2, 16),
                    dropout=self.dropout_rate,
                )
                self.shared_private_fusion = SharedPrivateFusion(
                    graph_dim=graph_dim,
                    context_dim=seq_dim,
                    raw_dim=raw_anchor_dim,
                    shared_dim=multimodal_hidden_dim,
                    private_dim=private_dim,
                    hidden_dim=fusion_hidden_dim,
                    num_classes=args.n_class,
                    dropout=self.dropout_rate,
                )
                uncertainty_input_dim = self.shared_private_fusion.output_dim
                fusion_input_dim = self.shared_private_fusion.output_dim
            elif self.fusion_variant == "tri_stream_gate":
                self.tri_stream_fusion = TriStreamGateFusion(
                    graph_dim=graph_dim,
                    sequence_dim=seq_dim,
                    raw_dim=raw_anchor_dim,
                    hidden_dim=fusion_hidden_dim,
                    num_classes=args.n_class,
                    dropout=self.dropout_rate,
                    batch_chunk_size=fusion_batch_chunk_size,
                )
                uncertainty_input_dim = self.tri_stream_fusion.output_dim
                fusion_input_dim = self.tri_stream_fusion.output_dim
            elif self.fusion_variant == "late_fusion":
                uncertainty_input_dim = graph_dim + seq_dim
                fusion_input_dim = graph_dim + seq_dim
            else:
                self.fusion_variant = "graph_dominant_residual"
                self.graph_dominant_fusion = GraphDominantResidualFusion(
                    graph_dim=graph_dim,
                    context_dim=seq_dim,
                    hidden_dim=fusion_hidden_dim,
                    num_classes=args.n_class,
                    dropout=self.dropout_rate,
                )
                uncertainty_input_dim = self.graph_dominant_fusion.output_dim
                fusion_input_dim = self.graph_dominant_fusion.output_dim
            self.uncertainty_head = nn.Sequential(
                nn.Linear(uncertainty_input_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_hidden_dim, 1),
            )
        elif self.enable_gnn:
            fusion_input_dim = graph_dim
        else:
            fusion_input_dim = seq_dim
        self.graph_output_dim = graph_dim
        self.sequence_output_dim = seq_dim
        self.raw_output_dim = (
            raw_anchor_dim
            if (self.raw_anchor_encoder is not None or self.typed_tabular_encoder is not None)
            else input_feature_dim
        )
        self.fusion = (
            None
            if self.use_multimodal_fusion
            else nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_hidden_dim, args.n_class),
            )
        )
        self.edge_loss_weight = float(getattr(args, "edge_loss_weight", getattr(args, "gamma", 1.0)))
        self.classification_loss_name = getattr(args, "classification_loss", "cb_focal")
        self.focal_gamma = float(getattr(args, "focal_gamma", 2.0))
        self.class_balance_beta = float(getattr(args, "class_balance_beta", 0.999))
        self.ranking_loss_weight = float(
            getattr(args, "ranking_loss_weight", 0.35 if dataset_name == "ieee" else 0.0)
        )
        self.ranking_max_pairs = int(getattr(args, "ranking_max_pairs", 4096))
        self.pseudo_label_threshold = float(getattr(args, "pseudo_label_threshold", 0.9))
        self.pseudo_label_min_threshold = float(getattr(args, "pseudo_label_min_threshold", 0.0))
        self.pseudo_label_top_fraction = float(getattr(args, "pseudo_label_top_fraction", 0.0))
        self.pseudo_label_weight = float(getattr(args, "pseudo_label_weight", 0.0))
        self.pseudo_label_novelty_threshold = float(getattr(args, "pseudo_label_novelty_threshold", 2.5))
        self.consistency_weight = float(getattr(args, "consistency_weight", 0.0))
        self.teacher_temperature = float(getattr(args, "teacher_temperature", 1.0))
        self.pseudo_warmup_rounds = int(getattr(args, "pseudo_warmup_rounds", 0))
        self.pseudo_ramp_rounds = int(getattr(args, "pseudo_ramp_rounds", 0))
        self.pseudo_reliability_threshold = float(getattr(args, "pseudo_reliability_threshold", 0.55))
        self.pseudo_modality_agreement_threshold = float(getattr(args, "pseudo_modality_agreement_threshold", 0.67))
        self.pseudo_prototype_margin_threshold = float(getattr(args, "pseudo_prototype_margin_threshold", 0.03))
        self.pseudo_max_nearest_distance = float(getattr(args, "pseudo_max_nearest_distance", 1.45))
        self.pseudo_threshold_std_scale = float(getattr(args, "pseudo_threshold_std_scale", 0.25))
        self.pseudo_uncertainty_confidence_threshold = float(
            getattr(args, "pseudo_uncertainty_confidence_threshold", 0.45)
        )
        self.uncertainty_ssl_blend = float(getattr(args, "uncertainty_ssl_blend", 0.25))
        self.open_set_novelty_threshold = float(getattr(args, "open_set_novelty_threshold", 0.0))
        self.open_set_loss_weight = float(getattr(args, "open_set_loss_weight", 0.0))
        self.target_prob_std = float(getattr(args, "target_prob_std", 0.0))
        self.prob_std_regularization_weight = float(getattr(args, "prob_std_regularization_weight", 0.0))
        self.graph_aux_loss_weight = float(
            getattr(args, "graph_aux_loss_weight", 0.0 if not self.use_multimodal_fusion else 0.10)
        )
        self.sequence_aux_loss_weight = float(
            getattr(args, "sequence_aux_loss_weight", 0.0 if not self.use_multimodal_fusion else 0.04)
        )
        self.raw_aux_loss_weight = float(
            getattr(args, "raw_aux_loss_weight", 0.0 if not self.use_multimodal_fusion else 0.06)
        )
        self.prototype_loss_weight = float(
            getattr(args, "prototype_loss_weight", 0.0 if not self.use_multimodal_fusion else 0.08)
        )
        default_shared_private_loss_weight = (
            0.05 if self.shared_private_fusion is not None and self.prototype_memory is not None else 0.0
        )
        self.shared_private_loss_weight = float(
            getattr(args, "shared_private_loss_weight", default_shared_private_loss_weight)
        )
        self.context_alignment_loss_weight = float(
            getattr(args, "context_alignment_loss_weight", 0.0 if self.event_encoder is None else 0.04)
        )
        self.uncertainty_loss_weight = float(
            getattr(args, "uncertainty_loss_weight", 0.0 if self.uncertainty_head is None else 0.04)
        )
        self.conflict_suppression_loss_weight = float(
            getattr(args, "conflict_suppression_loss_weight", 0.05 if self.shared_private_fusion is not None else 0.0)
        )
        self.prototype_margin_loss_weight = float(
            getattr(args, "prototype_margin_loss_weight", 0.06 if self.prototype_memory is not None else 0.0)
        )
        self.balance_aux_supervision_losses = bool(getattr(args, "balance_aux_supervision_losses", True))
        regularizer_sampling_mode = str(
            getattr(
                args,
                "regularizer_sampling_mode",
                "balanced" if self.balance_aux_supervision_losses else "full",
            )
        ).lower()
        if regularizer_sampling_mode not in {"balanced", "full"}:
            regularizer_sampling_mode = "balanced" if self.balance_aux_supervision_losses else "full"
        self.regularizer_sampling_mode = regularizer_sampling_mode
        self.graph_anchor_loss_weight = float(
            getattr(args, "graph_anchor_loss_weight", 0.0 if not self.use_multimodal_fusion else 0.12)
        )
        self.graph_anchor_temperature = float(getattr(args, "graph_anchor_temperature", 1.5))
        self.graph_teacher_distill_weight = float(getattr(args, "graph_teacher_distill_weight", 0.0))
        self.graph_teacher_temperature = float(getattr(args, "graph_teacher_temperature", 1.5))
        self.tabular_teacher_distill_weight = float(
            getattr(args, "tabular_teacher_distill_weight", 0.12 if dataset_name == "ieee" else 0.0)
        )
        self.tabular_teacher_temperature = float(getattr(args, "tabular_teacher_temperature", 1.0))
        self.coassociation_loss_weight = float(getattr(args, "coassociation_loss_weight", 0.0))
        self.wavelet_alignment_loss_weight = float(getattr(args, "wavelet_alignment_loss_weight", 0.0))
        self.utg_alignment_loss_weight = float(getattr(args, "utg_alignment_loss_weight", 0.0))
        self.use_pseudo_cycle_cache = bool(getattr(args, "pseudo_cycle_refresh_enabled", dataset_name == "elliptic"))
        self.pseudo_cycle_refresh_momentum = float(getattr(args, "pseudo_cycle_refresh_momentum", 0.65))
        self.legacy_fusion_only = bool(getattr(args, "legacy_fusion_only", False))
