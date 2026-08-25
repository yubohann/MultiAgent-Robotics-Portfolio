from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._helpers import (
    _class_centroids,
    _mask_to_index,
    _novelty_scores
)


class SslMixin:
    def _ssl_target_payload(
        self,
        *,
        graph,
        logits: torch.Tensor,
        branch_outputs: dict[str, torch.Tensor] | None,
        fused_embeddings: torch.Tensor,
        supervised_mask: torch.Tensor,
        unlabeled_mask: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
        current_round: int,
        allow_pseudo_cycle_cache: bool = True,
    ) -> dict[str, Any]:
        device = logits.device
        empty_float = torch.empty(0, dtype=logits.dtype, device=device)
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        default_payload = {
            "unlabeled_logits": logits.new_zeros((0, logits.size(-1))),
            "teacher_probs": None,
            "unlabeled_probs": logits.new_zeros((0, logits.size(-1))),
            "confidence": empty_float,
            "pseudo_labels": empty_long,
            "selected_labels": empty_long,
            "selected_weights": empty_float,
            "dynamic_pseudo_threshold": float(self.pseudo_label_threshold),
            "pseudo_threshold_used": 0.0,
            "reliable_mask": torch.zeros(0, dtype=torch.bool, device=device),
            "open_set_mask": torch.zeros(0, dtype=torch.bool, device=device),
            "modality_agreement": empty_float,
            "uncertainty_confidence": empty_float,
            "reliability_outputs": {
                "reliability": empty_float,
                "prototype_margin": empty_float,
                "nearest_distance": empty_float,
                "modality_agreement": empty_float,
                "uncertainty_confidence": empty_float,
            },
            "prototype_margin_available": False,
            "pseudo_nodes": 0,
            "novel_nodes": 0,
            "open_set_nodes": 0,
            "pseudo_reliability_mean": torch.tensor(0.0, device=device),
            "pseudo_reliability_pass_rate": torch.tensor(0.0, device=device),
            "pseudo_modality_agreement_mean": torch.tensor(0.0, device=device),
            "pseudo_uncertainty_confidence_mean": torch.tensor(0.0, device=device),
            "pseudo_uncertainty_pass_rate": torch.tensor(0.0, device=device),
            "pseudo_prototype_margin_mean": torch.tensor(0.0, device=device),
            "pseudo_nearest_distance_mean": torch.tensor(0.0, device=device),
            "ssl_reference_space": "none",
            "ssl_reference_space_id": -1,
            "prototype_reference_norm_mean": torch.tensor(0.0, device=device),
            "novelty_reference_norm_mean": torch.tensor(0.0, device=device),
            "pseudo_cycle_agreement_mean": torch.tensor(0.0, device=device),
            "pseudo_cycle_support_rate": torch.tensor(0.0, device=device),
        }
        if unlabeled_mask is None or not unlabeled_mask.any():
            return default_payload
        unlabeled_logits = logits[unlabeled_mask]
        teacher_probs = None
        if teacher_logits is not None:
            teacher_unlabeled_logits = teacher_logits[unlabeled_mask].detach()
            teacher_probs = F.softmax(teacher_unlabeled_logits / max(self.teacher_temperature, 1e-6), dim=1)
        unlabeled_probs = teacher_probs if teacher_probs is not None else F.softmax(unlabeled_logits.detach(), dim=1)
        confidence, pseudo_labels = unlabeled_probs.max(dim=1)
        dynamic_pseudo_threshold = float(self.pseudo_label_threshold)
        confidence_std = confidence.std(unbiased=False) if confidence.numel() > 1 else torch.tensor(0.0, device=device)
        confidence_mean = confidence.mean() if confidence.numel() > 0 else torch.tensor(0.0, device=device)
        dynamic_candidate = (
            float(
                max(
                    self.pseudo_label_min_threshold,
                    min(
                        float(self.pseudo_label_threshold),
                        float(confidence_mean.item() + self.pseudo_threshold_std_scale * confidence_std.item()),
                    ),
                )
            )
            if confidence.numel() > 0
            else float(self.pseudo_label_threshold)
        )
        if self.pseudo_label_top_fraction > 0.0 and confidence.numel() > 0:
            top_count = int(round(confidence.numel() * self.pseudo_label_top_fraction))
            top_count = int(np.clip(top_count, 1, confidence.numel()))
            top_confidence = torch.topk(confidence, k=top_count).values[-1]
            dynamic_candidate = min(
                dynamic_candidate,
                max(float(top_confidence.item()), self.pseudo_label_min_threshold),
            )
        if self.pseudo_ramp_rounds > 0:
            ramp_progress = float(
                np.clip((current_round - self.pseudo_warmup_rounds + 1) / max(self.pseudo_ramp_rounds, 1), 0.0, 1.0)
            )
        else:
            ramp_progress = 1.0 if current_round >= self.pseudo_warmup_rounds else 0.0
        threshold_floor = float(self.pseudo_label_threshold - 0.08 * ramp_progress)
        dynamic_pseudo_threshold = max(
            self.pseudo_label_min_threshold,
            min(float(self.pseudo_label_threshold), max(threshold_floor, dynamic_candidate)),
        )
        confident_mask = confidence >= dynamic_pseudo_threshold
        pseudo_novelty_mask = torch.ones_like(confident_mask, dtype=torch.bool)
        open_set_mask = torch.zeros_like(confident_mask, dtype=torch.bool)
        graph_branch_logits = (
            branch_outputs["graph_residual_logits"][unlabeled_mask]
            if branch_outputs is not None and "graph_residual_logits" in branch_outputs
            else unlabeled_logits.detach()
        )
        sequence_branch_logits = (
            branch_outputs["sequence_residual_logits"][unlabeled_mask]
            if branch_outputs is not None and "sequence_residual_logits" in branch_outputs
            else unlabeled_logits.detach()
        )
        raw_branch_logits = (
            branch_outputs["raw_branch_logits"][unlabeled_mask]
            if branch_outputs is not None and "raw_branch_logits" in branch_outputs
            else unlabeled_logits.detach()
        )
        branch_predictions = torch.stack(
            [
                graph_branch_logits.argmax(dim=-1),
                sequence_branch_logits.argmax(dim=-1),
                raw_branch_logits.argmax(dim=-1),
            ],
            dim=-1,
        )
        agreement_count = branch_predictions.eq(pseudo_labels.unsqueeze(-1)).float().sum(dim=-1)
        modality_agreement = agreement_count / float(branch_predictions.size(1))
        modality_agreement_mask = modality_agreement >= self.pseudo_modality_agreement_threshold
        conflict_for_unlabeled = branch_outputs.get("conflict_score") if branch_outputs is not None else None
        if conflict_for_unlabeled is not None:
            conflict_for_unlabeled = conflict_for_unlabeled[unlabeled_mask].reshape(-1)
        else:
            graph_seq_conflict = 1.0 - F.cosine_similarity(graph_branch_logits, sequence_branch_logits, dim=-1)
            graph_raw_conflict = 1.0 - F.cosine_similarity(graph_branch_logits, raw_branch_logits, dim=-1)
            seq_raw_conflict = 1.0 - F.cosine_similarity(sequence_branch_logits, raw_branch_logits, dim=-1)
            conflict_for_unlabeled = (graph_seq_conflict + graph_raw_conflict + seq_raw_conflict) / 3.0
        stage_name = str(getattr(self, "active_training_stage", "joint_finetune")).lower()
        uncertainty_available = (
            stage_name not in {"graph_warmup", "fusion_bootstrap"}
            and branch_outputs is not None
            and "uncertainty_logits" in branch_outputs
        )
        if uncertainty_available:
            uncertainty_confidence = torch.sigmoid(
                branch_outputs["uncertainty_logits"][unlabeled_mask].detach().reshape(-1)
            )
            uncertainty_mask = uncertainty_confidence >= float(self.pseudo_uncertainty_confidence_threshold)
        else:
            uncertainty_confidence = torch.ones_like(confidence)
            uncertainty_mask = torch.ones_like(confident_mask, dtype=torch.bool)
        ssl_reference_embeddings = fused_embeddings
        ssl_reference_space = "fused_embeddings"
        ssl_reference_space_id = 0
        if branch_outputs is not None and "shared_consensus" in branch_outputs:
            shared_consensus_reference = branch_outputs["shared_consensus"]
            if (
                shared_consensus_reference is not None
                and shared_consensus_reference.shape[:1] == fused_embeddings.shape[:1]
                and torch.isfinite(shared_consensus_reference).all()
            ):
                ssl_reference_embeddings = shared_consensus_reference
                ssl_reference_space = "shared_consensus"
                ssl_reference_space_id = 1
            else:
                ssl_reference_space = "fused_embeddings_fallback"
                ssl_reference_space_id = 2
        supervised_reference = ssl_reference_embeddings[supervised_mask].detach()
        unlabeled_reference = ssl_reference_embeddings[unlabeled_mask].detach()
        prototype_reference_norm_mean = (
            supervised_reference.norm(dim=-1).mean() if supervised_reference.numel() > 0 else torch.tensor(0.0, device=device)
        )
        novelty_reference_norm_mean = prototype_reference_norm_mean.detach()
        supervised_labels_reference = graph.ndata["label"][supervised_mask].detach()
        prototype_bank = None
        if self.prototype_memory is not None and ssl_reference_space == "shared_consensus":
            prototype_bank = self.prototype_memory.class_prototypes.detach()
        if prototype_bank is None and supervised_reference.numel() > 0:
            centroid_payload = _class_centroids(supervised_reference, supervised_labels_reference)
            if centroid_payload is not None:
                prototype_bank = centroid_payload[0].detach()
            else:
                prototype_bank = F.normalize(
                    supervised_reference[: max(logits.size(1), 1)],
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )
        if prototype_bank is not None and prototype_bank.numel() > 0:
            prototype_distances = torch.cdist(
                F.normalize(unlabeled_reference.float(), p=2, dim=-1, eps=1e-6),
                F.normalize(prototype_bank.float(), p=2, dim=-1, eps=1e-6),
                p=2,
            )
            prototype_margin_available = prototype_bank.size(0) > 1
            reliability_outputs = (
                self.prototype_reliability_scorer(
                    probs=unlabeled_probs,
                    prototype_distances=prototype_distances,
                    conflict_score=conflict_for_unlabeled,
                    modality_agreement=modality_agreement,
                )
                if self.prototype_reliability_scorer is not None
                else {
                    "reliability": (
                        0.70 * confidence
                        + 0.20 * modality_agreement
                        + 0.10 * (1.0 - conflict_for_unlabeled.clamp(min=0.0, max=1.0))
                    ).clamp(0.0, 1.0),
                    "prototype_margin": torch.zeros_like(confidence),
                    "nearest_distance": torch.zeros_like(confidence),
                    "modality_agreement": modality_agreement,
                }
            )
        else:
            prototype_margin_available = False
            reliability_outputs = {
                "reliability": (
                    0.70 * confidence
                    + 0.20 * modality_agreement
                    + 0.10 * (1.0 - conflict_for_unlabeled.clamp(min=0.0, max=1.0))
                ).clamp(0.0, 1.0),
                "prototype_margin": torch.zeros_like(confidence),
                "nearest_distance": torch.zeros_like(confidence),
                "modality_agreement": modality_agreement,
            }
        reliability_outputs = dict(reliability_outputs)
        reliability_outputs["uncertainty_confidence"] = uncertainty_confidence
        if uncertainty_available:
            blend = float(np.clip(self.uncertainty_ssl_blend, 0.0, 0.5))
            reliability_outputs["reliability"] = (
                (1.0 - blend) * reliability_outputs["reliability"] + blend * uncertainty_confidence
            ).clamp(0.0, 1.0)
        reliability_mask = reliability_outputs["reliability"] >= self.pseudo_reliability_threshold
        if prototype_margin_available:
            prototype_margin_mask = reliability_outputs["prototype_margin"] >= self.pseudo_prototype_margin_threshold
            nearest_distance_mask = reliability_outputs["nearest_distance"] <= self.pseudo_max_nearest_distance
        else:
            prototype_margin_mask = torch.ones_like(reliability_mask, dtype=torch.bool)
            nearest_distance_mask = torch.ones_like(reliability_mask, dtype=torch.bool)
        novelty_scores = _novelty_scores(
            reference_embeddings=supervised_reference,
            reference_labels=graph.ndata["label"][supervised_mask].detach(),
            query_embeddings=unlabeled_reference,
        )
        if novelty_scores is not None and self.pseudo_label_novelty_threshold > 0.0:
            pseudo_novelty_mask = novelty_scores <= self.pseudo_label_novelty_threshold
        if novelty_scores is not None and self.open_set_novelty_threshold > 0.0:
            open_set_mask = novelty_scores >= self.open_set_novelty_threshold
        reliable_mask = (
            confident_mask
            & pseudo_novelty_mask
            & modality_agreement_mask
            & reliability_mask
            & uncertainty_mask
            & prototype_margin_mask
            & nearest_distance_mask
            & ~open_set_mask
        )
        selected_labels = pseudo_labels.clone()
        selected_weight_scale = torch.ones_like(confidence)
        pseudo_uncertainty_confidence_mean = (
            uncertainty_confidence.mean()
            if uncertainty_confidence.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        pseudo_uncertainty_pass_rate = (
            uncertainty_mask.float().mean()
            if uncertainty_mask.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        pseudo_cycle_agreement_mean = torch.tensor(0.0, device=device)
        pseudo_cycle_support_rate = torch.tensor(0.0, device=device)
        if (
            allow_pseudo_cycle_cache
            and
            self.use_pseudo_cycle_cache
            and "pseudo_cycle_mask" in graph.ndata
            and "pseudo_cycle_label" in graph.ndata
            and "pseudo_cycle_weight" in graph.ndata
        ):
            cache_mask = graph.ndata["pseudo_cycle_mask"][unlabeled_mask].bool()
            cache_labels = graph.ndata["pseudo_cycle_label"][unlabeled_mask].long().clamp(min=0)
            cache_weights = graph.ndata["pseudo_cycle_weight"][unlabeled_mask].float().clamp(min=0.0)
            cache_agreement = cache_mask & cache_labels.eq(pseudo_labels)
            pseudo_cycle_agreement_mean = cache_agreement.float().mean() if cache_agreement.numel() > 0 else torch.tensor(0.0, device=device)
            pseudo_cycle_support_rate = cache_mask.float().mean() if cache_mask.numel() > 0 else torch.tensor(0.0, device=device)
            selected_labels = torch.where(cache_mask, cache_labels, selected_labels)
            selected_weight_scale = torch.where(cache_mask, cache_weights.clamp(min=1e-6), selected_weight_scale)
            reliable_mask = torch.where(cache_mask, reliable_mask & cache_agreement, reliable_mask)
        return {
            "unlabeled_logits": unlabeled_logits,
            "teacher_probs": teacher_probs,
            "unlabeled_probs": unlabeled_probs,
            "confidence": confidence,
            "pseudo_labels": pseudo_labels,
            "selected_labels": selected_labels,
            "selected_weights": selected_weight_scale,
            "dynamic_pseudo_threshold": dynamic_pseudo_threshold,
            "pseudo_threshold_used": float(dynamic_pseudo_threshold),
            "reliable_mask": reliable_mask,
            "open_set_mask": open_set_mask,
            "modality_agreement": modality_agreement,
            "uncertainty_confidence": uncertainty_confidence,
            "reliability_outputs": reliability_outputs,
            "prototype_margin_available": prototype_margin_available,
            "pseudo_nodes": int(reliable_mask.sum().item()),
            "novel_nodes": int((confident_mask & ~pseudo_novelty_mask & ~open_set_mask).sum().item()),
            "open_set_nodes": int(open_set_mask.sum().item()),
            "pseudo_reliability_mean": reliability_outputs["reliability"].mean(),
            "pseudo_reliability_pass_rate": reliable_mask.float().mean()
            if reliable_mask.numel() > 0
            else torch.tensor(0.0, device=device),
            "pseudo_modality_agreement_mean": reliability_outputs["modality_agreement"].mean(),
            "pseudo_uncertainty_confidence_mean": pseudo_uncertainty_confidence_mean,
            "pseudo_uncertainty_pass_rate": pseudo_uncertainty_pass_rate,
            "pseudo_prototype_margin_mean": reliability_outputs["prototype_margin"].mean(),
            "pseudo_nearest_distance_mean": reliability_outputs["nearest_distance"].mean(),
            "ssl_reference_space": ssl_reference_space,
            "ssl_reference_space_id": ssl_reference_space_id,
            "prototype_reference_norm_mean": prototype_reference_norm_mean,
            "novelty_reference_norm_mean": novelty_reference_norm_mean,
            "pseudo_cycle_agreement_mean": pseudo_cycle_agreement_mean,
            "pseudo_cycle_support_rate": pseudo_cycle_support_rate,
        }
    def build_pseudo_cycle_cache(self, graph, current_round: int) -> dict[str, Any]:
        payload = self.forward_with_branch_details(graph)
        logits = payload["logits"]
        supervised_mask = graph.ndata["train_supervised_mask"].bool() if "train_supervised_mask" in graph.ndata else graph.ndata["train_mask"].bool()
        unlabeled_mask = graph.ndata["train_unlabeled_mask"].bool() if "train_unlabeled_mask" in graph.ndata else None
        ssl_payload = self._ssl_target_payload(
            graph=graph,
            logits=logits,
            branch_outputs=payload["branch_outputs"],
            fused_embeddings=payload["fused_embeddings"],
            supervised_mask=supervised_mask,
            unlabeled_mask=unlabeled_mask,
            teacher_logits=None,
            current_round=current_round,
            allow_pseudo_cycle_cache=False,
        )
        num_nodes = logits.size(0)
        cycle_mask = torch.zeros(num_nodes, dtype=torch.bool, device=logits.device)
        cycle_label = torch.full((num_nodes,), -1, dtype=torch.long, device=logits.device)
        cycle_weight = torch.zeros(num_nodes, dtype=logits.dtype, device=logits.device)
        cycle_confidence = torch.zeros(num_nodes, dtype=logits.dtype, device=logits.device)
        cycle_round = torch.full((num_nodes,), -1, dtype=torch.long, device=logits.device)
        unlabeled_index = _mask_to_index(unlabeled_mask) if unlabeled_mask is not None else torch.empty(0, dtype=torch.long, device=logits.device)
        reliable_mask = ssl_payload["reliable_mask"]
        if unlabeled_index.numel() > 0 and reliable_mask.any():
            selected_global_nodes = unlabeled_index[reliable_mask]
            cycle_mask[selected_global_nodes] = True
            cycle_label[selected_global_nodes] = ssl_payload["selected_labels"][reliable_mask]
            cycle_weight[selected_global_nodes] = (
                ssl_payload["reliability_outputs"]["reliability"][reliable_mask]
                * ssl_payload["confidence"][reliable_mask]
                * ssl_payload["modality_agreement"][reliable_mask]
                * ssl_payload["selected_weights"][reliable_mask]
            ).detach()
            cycle_confidence[selected_global_nodes] = ssl_payload["confidence"][reliable_mask].detach()
            cycle_round[selected_global_nodes] = int(current_round)
        return {
            "mask": cycle_mask,
            "label": cycle_label,
            "weight": cycle_weight,
            "confidence": cycle_confidence,
            "round": cycle_round,
            "stats": {
                "nodes": int(cycle_mask.sum().item()),
                "threshold": float(ssl_payload["pseudo_threshold_used"]),
                "reliability_mean": float(ssl_payload["pseudo_reliability_mean"].item()),
                "support_rate": float(ssl_payload["pseudo_cycle_support_rate"].item()),
            },
        }
