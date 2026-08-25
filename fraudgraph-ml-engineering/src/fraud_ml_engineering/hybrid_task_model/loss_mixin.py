from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._helpers import (
    _balanced_binary_sample_indices,
    _balanced_subset_statistics,
    _classification_loss,
    _has_finite_branch_tensors,
    _mask_to_index,
    _pseudo_label_loss,
    _ranking_friendly_classification_loss,
    _uniform_target_kl_loss
)
from ._legacy import (
    _balance_modality_embedding,
    _detached_uncertainty_target_from_supervision
)


class LossMixin:
    def loss(
        self,
        graph,
        class_weights: torch.Tensor | None = None,
        class_counts: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        graph_teacher_logits: torch.Tensor | None = None,
        current_round: int = 0,
    ):
        payload = self.forward_with_branch_details(graph)
        logits = payload["logits"]
        edge_loss = payload["edge_loss"]
        graph_embeddings = payload["graph_embeddings"]
        sequence_embeddings = payload["sequence_embeddings"]
        raw_embeddings = payload["raw_embeddings"]
        fused_embeddings = payload["fused_embeddings"]
        aux_state = payload["aux_state"]
        branch_outputs: dict[str, torch.Tensor] | None = payload["branch_outputs"]
        diagnostic_tensors: dict[str, torch.Tensor | None] = payload["diagnostics"]
        train_mask = graph.ndata["train_mask"].bool()
        supervised_mask = graph.ndata["train_supervised_mask"].bool() if "train_supervised_mask" in graph.ndata else train_mask
        unlabeled_mask = graph.ndata["train_unlabeled_mask"].bool() if "train_unlabeled_mask" in graph.ndata else None
        stage_name = str(getattr(self, "active_training_stage", "joint_finetune")).lower()
        graph_only_stage = stage_name == "graph_warmup"
        bootstrap_stage = stage_name == "fusion_bootstrap"
        enable_heavy_fusion_regularizers = not graph_only_stage and not bootstrap_stage
        zero = torch.tensor(0.0, device=logits.device)
        full_supervised_index = _mask_to_index(supervised_mask)
        balanced_supervised_index = full_supervised_index
        regularizer_index = full_supervised_index
        supervised_probs = torch.empty(0, device=logits.device)
        classification_loss = torch.tensor(0.0, device=logits.device)
        classification_bce_loss = torch.tensor(0.0, device=logits.device)
        classification_ranking_loss = torch.tensor(0.0, device=logits.device)
        loss_class_weights = class_weights
        loss_class_counts = class_counts
        if full_supervised_index.numel() > 0:
            labels = graph.ndata["label"][full_supervised_index]
            train_logits = logits[full_supervised_index]
            supervised_probs = F.softmax(train_logits, dim=1)[:, 1]
            balanced_index = _balanced_binary_sample_indices(labels)
            if balanced_index is not None:
                balanced_supervised_index = full_supervised_index[balanced_index]
                train_logits = train_logits[balanced_index]
                labels = labels[balanced_index]
                loss_class_weights, loss_class_counts = _balanced_subset_statistics(
                    labels=labels,
                    class_weights=class_weights,
                    class_counts=class_counts,
                )
            else:
                balanced_index = None
            if self.regularizer_sampling_mode == "balanced" and balanced_index is not None:
                regularizer_index = balanced_supervised_index
            if str(self.classification_loss_name).lower() == "weighted_bce_auc":
                classification_loss, classification_bce_loss, classification_ranking_loss = _ranking_friendly_classification_loss(
                    logits=train_logits,
                    labels=labels,
                    class_counts=loss_class_counts,
                    ranking_weight=self.ranking_loss_weight,
                    max_pairs=self.ranking_max_pairs,
                )
            else:
                classification_loss = _classification_loss(
                    logits=train_logits,
                    labels=labels,
                    loss_name=self.classification_loss_name,
                    class_weights=loss_class_weights,
                    class_counts=loss_class_counts,
                    focal_gamma=self.focal_gamma,
                    class_balance_beta=self.class_balance_beta,
                )
        else:
            balanced_index = None
        graph_aux_loss = torch.tensor(0.0, device=logits.device)
        sequence_aux_loss = torch.tensor(0.0, device=logits.device)
        raw_aux_loss = torch.tensor(0.0, device=logits.device)
        prototype_loss = torch.tensor(0.0, device=logits.device)
        shared_private_loss = torch.tensor(0.0, device=logits.device)
        context_alignment_loss = torch.tensor(0.0, device=logits.device)
        uncertainty_loss = torch.tensor(0.0, device=logits.device)
        conflict_suppression_loss = torch.tensor(0.0, device=logits.device)
        prototype_margin_loss = torch.tensor(0.0, device=logits.device)
        graph_anchor_loss = torch.tensor(0.0, device=logits.device)
        graph_teacher_loss = torch.tensor(0.0, device=logits.device)
        tabular_teacher_loss = torch.tensor(0.0, device=logits.device)
        coassociation_loss = torch.tensor(0.0, device=logits.device)
        wavelet_alignment_loss = torch.tensor(0.0, device=logits.device)
        utg_alignment_loss = torch.tensor(0.0, device=logits.device)
        if self.use_multimodal_fusion and branch_outputs is not None and full_supervised_index.numel() > 0:
            aux_labels = graph.ndata["label"][full_supervised_index]
            full_graph_aux_logits = branch_outputs["graph_residual_logits"][full_supervised_index]
            graph_aux_logits = full_graph_aux_logits
            graph_anchor_teacher_logits = graph_aux_logits.detach()
            sequence_aux_logits = branch_outputs["sequence_residual_logits"][full_supervised_index]
            raw_aux_logits = branch_outputs["raw_branch_logits"][full_supervised_index]
            fusion_teacher_student_logits = logits[full_supervised_index]
            aux_class_weights = class_weights
            aux_class_counts = class_counts
            teacher_supervised_logits = (
                graph_teacher_logits[full_supervised_index].detach() if graph_teacher_logits is not None else None
            )
            if balanced_index is not None:
                aux_labels = aux_labels[balanced_index]
                graph_aux_logits = graph_aux_logits[balanced_index]
                graph_anchor_teacher_logits = graph_anchor_teacher_logits[balanced_index]
                sequence_aux_logits = sequence_aux_logits[balanced_index]
                raw_aux_logits = raw_aux_logits[balanced_index]
                fusion_teacher_student_logits = fusion_teacher_student_logits[balanced_index]
                aux_class_weights = loss_class_weights
                aux_class_counts = loss_class_counts
                if teacher_supervised_logits is not None:
                    teacher_supervised_logits = teacher_supervised_logits[balanced_index]
            if self.graph_aux_loss_weight > 0.0:
                if str(self.classification_loss_name).lower() == "weighted_bce_auc":
                    graph_aux_loss, _, _ = _ranking_friendly_classification_loss(
                        logits=graph_aux_logits,
                        labels=aux_labels,
                        class_counts=aux_class_counts,
                        ranking_weight=self.ranking_loss_weight,
                        max_pairs=self.ranking_max_pairs,
                    )
                else:
                    graph_aux_loss = _classification_loss(
                        logits=graph_aux_logits,
                        labels=aux_labels,
                        loss_name=self.classification_loss_name,
                        class_weights=aux_class_weights,
                        class_counts=aux_class_counts,
                        focal_gamma=self.focal_gamma,
                        class_balance_beta=self.class_balance_beta,
                    )
                graph_aux_loss = graph_aux_loss * self.graph_aux_loss_weight
            if self.sequence_aux_loss_weight > 0.0 and not graph_only_stage:
                if str(self.classification_loss_name).lower() == "weighted_bce_auc":
                    sequence_aux_loss, _, _ = _ranking_friendly_classification_loss(
                        logits=sequence_aux_logits,
                        labels=aux_labels,
                        class_counts=aux_class_counts,
                        ranking_weight=self.ranking_loss_weight,
                        max_pairs=self.ranking_max_pairs,
                    )
                else:
                    sequence_aux_loss = _classification_loss(
                        logits=sequence_aux_logits,
                        labels=aux_labels,
                        loss_name=self.classification_loss_name,
                        class_weights=aux_class_weights,
                        class_counts=aux_class_counts,
                        focal_gamma=self.focal_gamma,
                        class_balance_beta=self.class_balance_beta,
                    )
                sequence_aux_loss = sequence_aux_loss * self.sequence_aux_loss_weight
            if self.raw_aux_loss_weight > 0.0 and not graph_only_stage:
                if str(self.classification_loss_name).lower() == "weighted_bce_auc":
                    raw_aux_loss, _, _ = _ranking_friendly_classification_loss(
                        logits=raw_aux_logits,
                        labels=aux_labels,
                        class_counts=aux_class_counts,
                        ranking_weight=self.ranking_loss_weight,
                        max_pairs=self.ranking_max_pairs,
                    )
                else:
                    raw_aux_loss = _classification_loss(
                        logits=raw_aux_logits,
                        labels=aux_labels,
                        loss_name=self.classification_loss_name,
                        class_weights=aux_class_weights,
                        class_counts=aux_class_counts,
                        focal_gamma=self.focal_gamma,
                        class_balance_beta=self.class_balance_beta,
                    )
                raw_aux_loss = raw_aux_loss * self.raw_aux_loss_weight
            if enable_heavy_fusion_regularizers and self.prototype_loss_weight > 0.0 and self.prototype_memory is not None:
                shared_consensus = branch_outputs["shared_consensus"][regularizer_index]
                supervised_labels = graph.ndata["label"][regularizer_index].long()
                class_proto = self.prototype_memory.class_prototypes[
                    supervised_labels.clamp(
                        min=0,
                        max=self.prototype_memory.class_prototypes.size(0) - 1,
                    )
                ]
                class_alignment = 1.0 - F.cosine_similarity(shared_consensus, class_proto, dim=-1).mean()
                fraud_mask = supervised_labels == 1
                fraud_sub_alignment = torch.tensor(0.0, device=logits.device)
                fraud_sub_diversity = torch.tensor(0.0, device=logits.device)
                if fraud_mask.any():
                    fraud_shared = shared_consensus[fraud_mask]
                    fraud_distances = torch.cdist(
                        F.normalize(fraud_shared.float(), p=2, dim=-1, eps=1e-6),
                        F.normalize(self.prototype_memory.fraud_sub_prototypes.float(), p=2, dim=-1, eps=1e-6),
                        p=2,
                    )
                    nearest_sub = fraud_distances.min(dim=1).values
                    fraud_sub_alignment = nearest_sub.mean()
                    sub_cosine = torch.matmul(
                        F.normalize(self.prototype_memory.fraud_sub_prototypes.float(), p=2, dim=-1, eps=1e-6),
                        F.normalize(self.prototype_memory.fraud_sub_prototypes.float(), p=2, dim=-1, eps=1e-6).t(),
                    )
                    eye = torch.eye(sub_cosine.size(0), device=sub_cosine.device, dtype=sub_cosine.dtype)
                    fraud_sub_diversity = ((sub_cosine - eye) ** 2).mean()
                dataset_alignment = torch.tensor(0.0, device=logits.device)
                if "dataset_context_id" in graph.ndata:
                    dataset_ids = graph.ndata["dataset_context_id"][regularizer_index].long().clamp(
                        min=0,
                        max=self.prototype_memory.dataset_prototypes.size(0) - 1,
                    )
                    dataset_proto = self.prototype_memory.dataset_prototypes[dataset_ids]
                    dataset_alignment = 1.0 - F.cosine_similarity(shared_consensus, dataset_proto, dim=-1).mean()
                relation_alignment = torch.tensor(0.0, device=logits.device)
                relation_details = aux_state.get("relation_details")
                if isinstance(relation_details, dict) and "capsule_summaries" in relation_details:
                    relation_summaries = self.prototype_memory.project_relation_summaries(
                        relation_details["capsule_summaries"][regularizer_index]
                    )
                    relation_proto = self.prototype_memory.relation_prototypes.unsqueeze(0).expand_as(relation_summaries)
                    relation_alignment = 1.0 - F.cosine_similarity(relation_summaries, relation_proto, dim=-1).mean()
                prototype_loss = (
                    class_alignment + 0.50 * dataset_alignment + 0.35 * relation_alignment + 0.45 * fraud_sub_alignment + 0.10 * fraud_sub_diversity
                ) * self.prototype_loss_weight
                if self.prototype_margin_loss_weight > 0.0 and shared_consensus.size(0) > 0:
                    normal_proto = self.prototype_memory.class_prototypes[0:1]
                    positive_proto = self.prototype_memory.class_prototypes[1:2]
                    normal_distance = torch.cdist(shared_consensus.float(), normal_proto.float(), p=2).squeeze(1)
                    positive_distance = torch.cdist(shared_consensus.float(), positive_proto.float(), p=2).squeeze(1)
                    target = supervised_labels.float()
                    margin_gap = torch.where(target > 0.5, normal_distance - positive_distance, positive_distance - normal_distance)
                    prototype_margin_loss = F.relu(0.35 - margin_gap).mean() * self.prototype_margin_loss_weight
            if enable_heavy_fusion_regularizers and self.shared_private_loss_weight > 0.0 and _has_finite_branch_tensors(
                branch_outputs,
                "graph_shared",
                "context_shared",
                "raw_shared",
                "graph_private",
                "context_private",
                "raw_private",
            ):
                graph_shared = branch_outputs["graph_shared"][regularizer_index]
                context_shared = branch_outputs["context_shared"][regularizer_index]
                raw_shared = branch_outputs["raw_shared"][regularizer_index]
                graph_private = branch_outputs["graph_private"][regularizer_index]
                context_private = branch_outputs["context_private"][regularizer_index]
                raw_private = branch_outputs["raw_private"][regularizer_index]
                shared_alignment = (
                    (1.0 - F.cosine_similarity(graph_shared, context_shared, dim=-1).mean())
                    + (1.0 - F.cosine_similarity(graph_shared, raw_shared, dim=-1).mean())
                    + (1.0 - F.cosine_similarity(context_shared, raw_shared, dim=-1).mean())
                ) / 3.0
                orthogonality = (
                    torch.abs(F.cosine_similarity(graph_private, context_shared, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(context_private, graph_shared, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(graph_private, raw_shared, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(raw_private, graph_shared, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(context_private, raw_shared, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(raw_private, context_shared, dim=-1)).mean()
                ) / 6.0
                private_decouple = (
                    torch.abs(F.cosine_similarity(graph_private, context_private, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(graph_private, raw_private, dim=-1)).mean()
                    + torch.abs(F.cosine_similarity(context_private, raw_private, dim=-1)).mean()
                ) / 3.0
                shared_private_loss = (shared_alignment + 0.5 * orthogonality + 0.25 * private_decouple) * self.shared_private_loss_weight
            if enable_heavy_fusion_regularizers and self.conflict_suppression_loss_weight > 0.0 and _has_finite_branch_tensors(
                branch_outputs,
                "conflict_score",
                "graph_residual_logits",
                "sequence_residual_logits",
                "raw_branch_logits",
            ):
                conflict_score = branch_outputs["conflict_score"][regularizer_index]
                graph_logits_supervised = branch_outputs["graph_residual_logits"][regularizer_index]
                seq_logits_supervised = branch_outputs["sequence_residual_logits"][regularizer_index]
                raw_logits_supervised = branch_outputs["raw_branch_logits"][regularizer_index]
                graph_seq_conflict = 1.0 - F.cosine_similarity(graph_logits_supervised, seq_logits_supervised, dim=-1)
                graph_raw_conflict = 1.0 - F.cosine_similarity(graph_logits_supervised, raw_logits_supervised, dim=-1)
                seq_raw_conflict = 1.0 - F.cosine_similarity(seq_logits_supervised, raw_logits_supervised, dim=-1)
                target_conflict = ((graph_seq_conflict + graph_raw_conflict + seq_raw_conflict) / 3.0).detach().unsqueeze(-1)
                conflict_reg = F.mse_loss(conflict_score, target_conflict)
                shared_gate = branch_outputs.get("shared_gate")
                private_gate = branch_outputs.get("private_gate")
                if shared_gate is not None and private_gate is not None:
                    shared_gate = shared_gate[regularizer_index]
                    private_gate = private_gate[regularizer_index]
                    gate_consistency = F.relu(shared_gate + 0.50 * target_conflict - 1.0).mean() + F.relu(
                        target_conflict - private_gate
                    ).mean()
                else:
                    gate_consistency = torch.tensor(0.0, device=logits.device)
                conflict_suppression_loss = (conflict_reg + 0.5 * gate_consistency) * self.conflict_suppression_loss_weight
            if (
                enable_heavy_fusion_regularizers
                and
                self.context_alignment_loss_weight > 0.0
                and aux_state.get("relation_embeddings") is not None
                and aux_state.get("event_embeddings") is not None
            ):
                relation_view = _balance_modality_embedding(aux_state["relation_embeddings"])
                event_view = _balance_modality_embedding(aux_state["event_embeddings"])
                context_alignment_loss = (
                    1.0 - F.cosine_similarity(relation_view, event_view, dim=-1).mean()
                ) * self.context_alignment_loss_weight
            if enable_heavy_fusion_regularizers and self.uncertainty_loss_weight > 0.0:
                target_confidence = (
                    graph.ndata["label_confidence_target"][regularizer_index].float()
                    if "label_confidence_target" in graph.ndata
                    else torch.empty(0, device=logits.device, dtype=logits.dtype)
                )
                fallback_target_confidence = _detached_uncertainty_target_from_supervision(
                    fusion_logits=fusion_teacher_student_logits,
                    labels=graph.ndata["label"][regularizer_index].long(),
                    graph_logits=branch_outputs["graph_residual_logits"][regularizer_index],
                    sequence_logits=branch_outputs["sequence_residual_logits"][regularizer_index],
                    raw_logits=branch_outputs["raw_branch_logits"][regularizer_index],
                    conflict_score=None
                    if branch_outputs.get("conflict_score") is None
                    else branch_outputs["conflict_score"][regularizer_index],
                )
                if target_confidence.numel() > 0 and torch.isfinite(target_confidence).all():
                    if (
                        fallback_target_confidence.numel() == target_confidence.numel()
                        and float((target_confidence.max() - target_confidence.min()).item()) > 1e-6
                    ):
                        target_confidence = (
                            0.65 * target_confidence.to(
                                device=fallback_target_confidence.device,
                                dtype=fallback_target_confidence.dtype,
                            )
                            + 0.35 * fallback_target_confidence
                        ).clamp(0.0, 1.0)
                    elif fallback_target_confidence.numel() == target_confidence.numel():
                        target_confidence = fallback_target_confidence
                    else:
                        target_confidence = target_confidence.to(
                            device=branch_outputs["uncertainty_logits"].device,
                            dtype=branch_outputs["uncertainty_logits"].dtype,
                        )
                else:
                    target_confidence = fallback_target_confidence
                if target_confidence.numel() > 0:
                    predicted_confidence = torch.sigmoid(branch_outputs["uncertainty_logits"][regularizer_index].squeeze(-1))
                    uncertainty_loss = F.mse_loss(predicted_confidence, target_confidence) * self.uncertainty_loss_weight
            if self.graph_anchor_loss_weight > 0.0 and not graph_only_stage:
                anchor_scale = 1.0 if bootstrap_stage else 0.35
                anchor_temperature = max(float(self.graph_anchor_temperature), 1e-6)
                graph_anchor_loss = F.kl_div(
                    F.log_softmax(fusion_teacher_student_logits / anchor_temperature, dim=1),
                    F.softmax(graph_anchor_teacher_logits / anchor_temperature, dim=1),
                    reduction="batchmean",
                ) * (anchor_temperature ** 2)
                graph_anchor_loss = graph_anchor_loss * (self.graph_anchor_loss_weight * anchor_scale)
            if teacher_supervised_logits is not None and self.graph_teacher_distill_weight > 0.0:
                distill_temperature = max(float(self.graph_teacher_temperature), 1e-6)
                if graph_only_stage:
                    distill_scale = 0.50
                elif bootstrap_stage:
                    distill_scale = 1.00
                else:
                    distill_scale = 0.60
                graph_teacher_loss = F.kl_div(
                    F.log_softmax(graph_aux_logits / distill_temperature, dim=1),
                    F.softmax(teacher_supervised_logits / distill_temperature, dim=1),
                    reduction="batchmean",
                ) * (distill_temperature ** 2)
                if not graph_only_stage:
                    fusion_teacher_loss = F.kl_div(
                        F.log_softmax(fusion_teacher_student_logits / distill_temperature, dim=1),
                        F.softmax(teacher_supervised_logits / distill_temperature, dim=1),
                        reduction="batchmean",
                    ) * (distill_temperature ** 2)
                    graph_teacher_loss = graph_teacher_loss + 0.35 * fusion_teacher_loss
                graph_teacher_loss = graph_teacher_loss * (self.graph_teacher_distill_weight * distill_scale)
            if (
                not graph_only_stage
                and self.tabular_teacher_distill_weight > 0.0
                and "tabular_teacher_logits" in graph.ndata
            ):
                teacher_index = _mask_to_index(graph.ndata["train_mask"].bool())
                if teacher_index.numel() > 0:
                    tabular_teacher_targets = graph.ndata["tabular_teacher_logits"][teacher_index].detach()
                    raw_teacher_student = branch_outputs["raw_branch_logits"][teacher_index]
                    fusion_teacher_student = logits[teacher_index]
                    teacher_temperature = max(float(self.tabular_teacher_temperature), 1e-6)
                    raw_teacher_loss = F.kl_div(
                        F.log_softmax(raw_teacher_student / teacher_temperature, dim=1),
                        F.softmax(tabular_teacher_targets / teacher_temperature, dim=1),
                        reduction="batchmean",
                    ) * (teacher_temperature ** 2)
                    fusion_teacher_loss = F.kl_div(
                        F.log_softmax(fusion_teacher_student / teacher_temperature, dim=1),
                        F.softmax(tabular_teacher_targets / teacher_temperature, dim=1),
                        reduction="batchmean",
                    ) * (teacher_temperature ** 2)
                    tabular_teacher_loss = (
                        0.70 * raw_teacher_loss + 0.30 * fusion_teacher_loss
                    ) * self.tabular_teacher_distill_weight
        spread_loss = torch.tensor(0.0, device=logits.device)
        if (
            self.prob_std_regularization_weight > 0.0
            and self.target_prob_std > 0.0
            and supervised_probs.numel() >= 2
        ):
            supervised_prob_std = supervised_probs.std(unbiased=False)
            spread_loss = F.relu(
                torch.tensor(self.target_prob_std, device=logits.device, dtype=supervised_probs.dtype)
                - supervised_prob_std
            )
            spread_loss = spread_loss * self.prob_std_regularization_weight
        effective_pseudo_weight = 0.0 if graph_only_stage else float(self.pseudo_label_weight)
        if current_round < self.pseudo_warmup_rounds:
            effective_pseudo_weight = 0.0
        elif self.pseudo_ramp_rounds > 0:
            ramp_numerator = current_round - self.pseudo_warmup_rounds + 1
            ramp_progress = float(np.clip(ramp_numerator / max(self.pseudo_ramp_rounds, 1), 0.0, 1.0))
            effective_pseudo_weight = effective_pseudo_weight * ramp_progress
        pseudo_loss = torch.tensor(0.0, device=logits.device)
        consistency_loss = torch.tensor(0.0, device=logits.device)
        open_set_loss = torch.tensor(0.0, device=logits.device)
        pseudo_nodes = 0
        novel_nodes = 0
        open_set_nodes = 0
        pseudo_threshold_used = 0.0
        pseudo_reliability_mean = torch.tensor(0.0, device=logits.device)
        pseudo_reliability_pass_rate = torch.tensor(0.0, device=logits.device)
        pseudo_modality_agreement_mean = torch.tensor(0.0, device=logits.device)
        pseudo_uncertainty_confidence_mean = torch.tensor(0.0, device=logits.device)
        pseudo_uncertainty_pass_rate = torch.tensor(0.0, device=logits.device)
        pseudo_prototype_margin_mean = torch.tensor(0.0, device=logits.device)
        pseudo_nearest_distance_mean = torch.tensor(0.0, device=logits.device)
        ssl_reference_space = "none"
        ssl_reference_space_id = -1
        novelty_reference_norm_mean = torch.tensor(0.0, device=logits.device)
        prototype_reference_norm_mean = torch.tensor(0.0, device=logits.device)
        pseudo_cycle_agreement_mean = torch.tensor(0.0, device=logits.device)
        pseudo_cycle_support_rate = torch.tensor(0.0, device=logits.device)
        effective_consistency_weight = 0.0 if graph_only_stage else float(self.consistency_weight)
        effective_open_set_loss_weight = 0.0 if graph_only_stage else float(self.open_set_loss_weight)
        if not graph_only_stage:
            train_index = _mask_to_index(train_mask)
            if train_index.numel() > 0:
                coassociation_embeddings = aux_state.get("coassociation_embeddings")
                if self.coassociation_loss_weight > 0.0 and coassociation_embeddings is not None:
                    coassociation_view = _balance_modality_embedding(coassociation_embeddings[train_index])
                    graph_view = _balance_modality_embedding(graph_embeddings[train_index])
                    sequence_view = _balance_modality_embedding(sequence_embeddings[train_index])
                    coassociation_target_views = [sequence_view]
                    if graph_view.size(-1) == coassociation_view.size(-1):
                        coassociation_target_views.append(graph_view)
                    raw_view = _balance_modality_embedding(raw_embeddings[train_index])
                    if raw_view.size(-1) == coassociation_view.size(-1):
                        coassociation_target_views.append(raw_view)
                    coassociation_target = torch.stack(coassociation_target_views, dim=0).mean(dim=0)
                    density = aux_state.get("coassociation_density")
                    if density is not None:
                        density_scale = 0.5 + torch.sigmoid(density[train_index].float().reshape(-1))
                    else:
                        density_scale = torch.ones(coassociation_view.size(0), device=logits.device)
                    coassociation_alignment = 1.0 - F.cosine_similarity(
                        coassociation_view,
                        coassociation_target,
                        dim=-1,
                    )
                    coassociation_loss = (
                        (coassociation_alignment * density_scale).mean() * self.coassociation_loss_weight
                    )
                wavelet_embeddings = aux_state.get("wavelet_embeddings")
                if self.wavelet_alignment_loss_weight > 0.0 and wavelet_embeddings is not None:
                    wavelet_view = _balance_modality_embedding(wavelet_embeddings[train_index])
                    wavelet_targets = []
                    if aux_state.get("temporal_context_embeddings") is not None:
                        wavelet_targets.append(
                            _balance_modality_embedding(aux_state["temporal_context_embeddings"][train_index])
                        )
                    if aux_state.get("event_embeddings") is not None:
                        wavelet_targets.append(_balance_modality_embedding(aux_state["event_embeddings"][train_index]))
                    if aux_state.get("relation_embeddings") is not None:
                        wavelet_targets.append(
                            _balance_modality_embedding(aux_state["relation_embeddings"][train_index])
                        )
                    if wavelet_targets:
                        wavelet_target = torch.stack(wavelet_targets, dim=0).mean(dim=0)
                        wavelet_alignment_loss = (
                            (1.0 - F.cosine_similarity(wavelet_view, wavelet_target, dim=-1)).mean()
                            * self.wavelet_alignment_loss_weight
                        )
                utg_outputs = aux_state.get("utg_outputs")
                if (
                    self.utg_alignment_loss_weight > 0.0
                    and isinstance(utg_outputs, dict)
                    and utg_outputs.get("utg_temporal_anchor") is not None
                ):
                    utg_anchor = _balance_modality_embedding(utg_outputs["utg_temporal_anchor"][train_index])
                    utg_targets = []
                    if aux_state.get("relation_embeddings") is not None:
                        utg_targets.append(_balance_modality_embedding(aux_state["relation_embeddings"][train_index]))
                    if aux_state.get("event_embeddings") is not None:
                        utg_targets.append(_balance_modality_embedding(aux_state["event_embeddings"][train_index]))
                    if aux_state.get("wavelet_embeddings") is not None:
                        utg_targets.append(_balance_modality_embedding(aux_state["wavelet_embeddings"][train_index]))
                    if aux_state.get("coassociation_embeddings") is not None:
                        utg_targets.append(
                            _balance_modality_embedding(aux_state["coassociation_embeddings"][train_index])
                        )
                    if utg_targets:
                        utg_target = torch.stack(utg_targets, dim=0).mean(dim=0)
                        utg_alignment_loss = (
                            (1.0 - F.cosine_similarity(utg_anchor, utg_target, dim=-1)).mean()
                            * self.utg_alignment_loss_weight
                        )
        if (
            unlabeled_mask is not None
            and unlabeled_mask.any()
            and (
                effective_pseudo_weight > 0.0
                or effective_consistency_weight > 0.0
                or effective_open_set_loss_weight > 0.0
            )
        ):
            ssl_payload = self._ssl_target_payload(
                graph=graph,
                logits=logits,
                branch_outputs=branch_outputs,
                fused_embeddings=fused_embeddings,
                supervised_mask=supervised_mask,
                unlabeled_mask=unlabeled_mask,
                teacher_logits=teacher_logits,
                current_round=current_round,
            )
            pseudo_nodes = int(ssl_payload["pseudo_nodes"])
            novel_nodes = int(ssl_payload["novel_nodes"])
            open_set_nodes = int(ssl_payload["open_set_nodes"])
            pseudo_threshold_used = float(ssl_payload["pseudo_threshold_used"])
            pseudo_reliability_mean = ssl_payload["pseudo_reliability_mean"]
            pseudo_reliability_pass_rate = ssl_payload["pseudo_reliability_pass_rate"]
            pseudo_modality_agreement_mean = ssl_payload["pseudo_modality_agreement_mean"]
            pseudo_uncertainty_confidence_mean = ssl_payload["pseudo_uncertainty_confidence_mean"]
            pseudo_uncertainty_pass_rate = ssl_payload["pseudo_uncertainty_pass_rate"]
            pseudo_prototype_margin_mean = ssl_payload["pseudo_prototype_margin_mean"]
            pseudo_nearest_distance_mean = ssl_payload["pseudo_nearest_distance_mean"]
            ssl_reference_space = str(ssl_payload["ssl_reference_space"])
            ssl_reference_space_id = int(ssl_payload["ssl_reference_space_id"])
            novelty_reference_norm_mean = ssl_payload["novelty_reference_norm_mean"]
            prototype_reference_norm_mean = ssl_payload["prototype_reference_norm_mean"]
            pseudo_cycle_agreement_mean = ssl_payload["pseudo_cycle_agreement_mean"]
            pseudo_cycle_support_rate = ssl_payload["pseudo_cycle_support_rate"]
            reliable_mask = ssl_payload["reliable_mask"]
            if effective_pseudo_weight > 0.0 and reliable_mask.any():
                selected_logits = ssl_payload["unlabeled_logits"][reliable_mask]
                selected_labels = ssl_payload["selected_labels"][reliable_mask]
                selected_weights = (
                    ssl_payload["reliability_outputs"]["reliability"][reliable_mask]
                    * ssl_payload["confidence"][reliable_mask]
                    * ssl_payload["modality_agreement"][reliable_mask]
                    * ssl_payload["selected_weights"][reliable_mask]
                ).detach()
                pseudo_loss = _pseudo_label_loss(
                    logits=selected_logits,
                    labels=selected_labels,
                    loss_name=self.classification_loss_name,
                    class_weights=class_weights,
                    class_counts=class_counts,
                    focal_gamma=self.focal_gamma,
                    class_balance_beta=self.class_balance_beta,
                    sample_weights=selected_weights,
                )
                pseudo_loss = pseudo_loss * effective_pseudo_weight
            if effective_consistency_weight > 0.0 and reliable_mask.any():
                if ssl_payload["teacher_probs"] is not None:
                    consistency_loss = F.kl_div(
                        F.log_softmax(ssl_payload["unlabeled_logits"][reliable_mask], dim=1),
                        ssl_payload["teacher_probs"][reliable_mask],
                        reduction="batchmean",
                    )
                else:
                    augmented_logits = self._forward_consistency_view(graph)[unlabeled_mask][reliable_mask]
                    consistency_loss = F.kl_div(
                        F.log_softmax(augmented_logits, dim=1),
                        ssl_payload["unlabeled_probs"][reliable_mask],
                        reduction="batchmean",
                    )
                consistency_weight_scale = (
                    ssl_payload["reliability_outputs"]["reliability"][reliable_mask].mean().detach()
                    if ssl_payload["reliability_outputs"]["reliability"][reliable_mask].numel() > 0
                    else torch.tensor(1.0, device=logits.device)
                )
                consistency_loss = consistency_loss * effective_consistency_weight * consistency_weight_scale
            if effective_open_set_loss_weight > 0.0 and ssl_payload["open_set_mask"].any():
                open_set_loss = _uniform_target_kl_loss(
                    ssl_payload["unlabeled_logits"][ssl_payload["open_set_mask"]]
                )
                open_set_loss = open_set_loss * effective_open_set_loss_weight
        total_loss = (
            classification_loss
            + graph_aux_loss
            + sequence_aux_loss
            + raw_aux_loss
            + prototype_loss
            + prototype_margin_loss
            + shared_private_loss
            + conflict_suppression_loss
            + context_alignment_loss
            + uncertainty_loss
            + graph_anchor_loss
            + graph_teacher_loss
            + tabular_teacher_loss
            + coassociation_loss
            + wavelet_alignment_loss
            + utg_alignment_loss
            + self.edge_loss_weight * edge_loss
            + pseudo_loss
            + consistency_loss
            + open_set_loss
            + spread_loss
        )
        def _mean_value(name: str) -> torch.Tensor:
            value = diagnostic_tensors.get(name)
            if value is None or value.numel() == 0:
                return torch.tensor(0.0, device=logits.device)
            return value.float().reshape(-1).mean()
        def _std_value(name: str) -> torch.Tensor:
            value = diagnostic_tensors.get(name)
            if value is None or value.numel() == 0:
                return torch.tensor(0.0, device=logits.device)
            return value.float().reshape(-1).std(unbiased=False)
        return total_loss, {
            "cls_loss": classification_loss.detach(),
            "classification_bce_loss": classification_bce_loss.detach(),
            "classification_ranking_loss": classification_ranking_loss.detach(),
            "pseudo_loss": pseudo_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "open_set_loss": open_set_loss.detach(),
            "graph_aux_loss": graph_aux_loss.detach(),
            "sequence_aux_loss": sequence_aux_loss.detach(),
            "raw_aux_loss": raw_aux_loss.detach(),
            "prototype_loss": prototype_loss.detach(),
            "prototype_margin_loss": prototype_margin_loss.detach(),
            "shared_private_loss": shared_private_loss.detach(),
            "conflict_suppression_loss": conflict_suppression_loss.detach(),
            "context_alignment_loss": context_alignment_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
            "graph_anchor_loss": graph_anchor_loss.detach(),
            "graph_teacher_loss": graph_teacher_loss.detach(),
            "tabular_teacher_loss": tabular_teacher_loss.detach(),
            "coassociation_loss": coassociation_loss.detach(),
            "wavelet_alignment_loss": wavelet_alignment_loss.detach(),
            "utg_alignment_loss": utg_alignment_loss.detach(),
            "spread_loss": spread_loss.detach(),
            "edge_loss": edge_loss.detach(),
            "train_nodes": int(supervised_mask.sum().item()),
            "supervised_nodes": int(full_supervised_index.numel()),
            "balanced_supervised_nodes": int(balanced_supervised_index.numel()),
            "regularizer_nodes": int(regularizer_index.numel()),
            "effective_ssl_nodes": float(full_supervised_index.numel()) + float(pseudo_nodes),
            "pseudo_nodes": int(pseudo_nodes),
            "novel_nodes": int(novel_nodes),
            "open_set_nodes": int(open_set_nodes),
            "pseudo_threshold_used": float(pseudo_threshold_used),
            "pseudo_reliability_mean": pseudo_reliability_mean.detach(),
            "pseudo_reliability_pass_rate": pseudo_reliability_pass_rate.detach(),
            "pseudo_modality_agreement_mean": pseudo_modality_agreement_mean.detach(),
            "pseudo_uncertainty_confidence_mean": pseudo_uncertainty_confidence_mean.detach(),
            "pseudo_uncertainty_pass_rate": pseudo_uncertainty_pass_rate.detach(),
            "pseudo_prototype_margin_mean": pseudo_prototype_margin_mean.detach(),
            "pseudo_nearest_distance_mean": pseudo_nearest_distance_mean.detach(),
            "shared_gap_mean": _mean_value("shared_gap").detach(),
            "shared_gap_std": _std_value("shared_gap").detach(),
            "private_interaction_mean": _mean_value("private_interaction").detach(),
            "private_interaction_std": _std_value("private_interaction").detach(),
            "context_gate_mean": _mean_value("context_gate").detach(),
            "context_gate_std": _std_value("context_gate").detach(),
            "graph_branch_gate_mean": _mean_value("graph_branch_gate").detach(),
            "graph_branch_gate_std": _std_value("graph_branch_gate").detach(),
            "graph_correction_support_mean": _mean_value("graph_correction_support").detach(),
            "graph_correction_support_std": _std_value("graph_correction_support").detach(),
            "sequence_branch_gate_mean": _mean_value("sequence_branch_gate").detach(),
            "sequence_branch_gate_std": _std_value("sequence_branch_gate").detach(),
            "raw_branch_gate_mean": _mean_value("raw_branch_gate").detach(),
            "raw_branch_gate_std": _std_value("raw_branch_gate").detach(),
            "fusion_delta_gate_mean": _mean_value("fusion_delta_gate").detach(),
            "fusion_delta_gate_std": _std_value("fusion_delta_gate").detach(),
            "delta_correction_support_mean": _mean_value("delta_correction_support").detach(),
            "delta_correction_support_std": _std_value("delta_correction_support").detach(),
            "graph_embedding_norm_mean": _mean_value("graph_embedding_norm").detach(),
            "sequence_embedding_norm_mean": _mean_value("sequence_embedding_norm").detach(),
            "raw_embedding_norm_mean": _mean_value("raw_embedding_norm").detach(),
            "fused_embedding_norm_mean": _mean_value("fused_embedding_norm").detach(),
            "sequence_token_valid_ratio_mean": _mean_value("sequence_token_valid_ratio").detach(),
            "sequence_valid_length_mean": _mean_value("sequence_valid_length").detach(),
            "graph_sequence_prob_gap_mean": _mean_value("graph_sequence_prob_gap").detach(),
            "time_reliability_mean": _mean_value("time_reliability").detach(),
            "time_reliability_std": _std_value("time_reliability").detach(),
            "graph_temporal_gate_mean": _mean_value("graph_temporal_gate").detach(),
            "graph_temporal_gate_std": _std_value("graph_temporal_gate").detach(),
            "shared_gate_mean": _mean_value("shared_gate").detach(),
            "shared_gate_std": _std_value("shared_gate").detach(),
            "private_gate_mean": _mean_value("private_gate").detach(),
            "private_gate_std": _std_value("private_gate").detach(),
            "conflict_score_mean": _mean_value("conflict_score").detach(),
            "conflict_score_std": _std_value("conflict_score").detach(),
            "ssl_reference_space": ssl_reference_space,
            "ssl_reference_space_id": int(ssl_reference_space_id),
            "novelty_reference_norm_mean": novelty_reference_norm_mean.detach(),
            "prototype_reference_norm_mean": prototype_reference_norm_mean.detach(),
            "pseudo_cycle_agreement_mean": pseudo_cycle_agreement_mean.detach(),
            "pseudo_cycle_support_rate": pseudo_cycle_support_rate.detach(),
            "wavelet_gate_mean": _mean_value("wavelet_gate").detach(),
            "coassociation_gate_mean": _mean_value("coassociation_gate").detach(),
            "coassociation_density_mean": _mean_value("coassociation_density").detach(),
            "diffusion_gate_mean": _mean_value("diffusion_gate").detach(),
            "diffusion_neighbor_strength_mean": _mean_value("diffusion_neighbor_strength").detach(),
            "utg_temporal_gate_mean": _mean_value("utg_temporal_gate").detach(),
        }
