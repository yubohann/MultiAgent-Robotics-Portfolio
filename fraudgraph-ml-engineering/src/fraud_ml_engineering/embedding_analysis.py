from __future__ import annotations

"""Embedding export and visualization helpers for train/valid/test splits."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _subsample_indices(labels: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    total = len(labels)
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = np.arange(total, dtype=np.int64)
    label_values = np.unique(labels)
    selected = []
    for label in label_values:
        label_indices = indices[labels == label]
        if len(label_indices) == 0:
            continue
        take = max(1, int(round(max_points * (len(label_indices) / total))))
        take = min(take, len(label_indices))
        selected.append(rng.choice(label_indices, size=take, replace=False))
    merged = np.unique(np.concatenate(selected)) if selected else indices[:max_points]
    if len(merged) > max_points:
        merged = rng.choice(merged, size=max_points, replace=False)
    return np.sort(merged.astype(np.int64))


def _project_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[1] == 1:
        projection = np.concatenate([embeddings, np.zeros_like(embeddings)], axis=1)
        return projection, [1.0, 0.0]
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(embeddings)
    return projection, [float(v) for v in pca.explained_variance_ratio_]


def _separation_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    metrics = {
        "num_samples": int(len(labels)),
        "num_classes": int(len(unique_labels)),
        "positive_count": int((labels == 1).sum()) if len(unique_labels) > 0 else 0,
        "negative_count": int((labels == 0).sum()) if len(unique_labels) > 0 else 0,
        "centroid_distance": 0.0,
        "silhouette": 0.0,
    }
    if len(unique_labels) < 2:
        return metrics

    negative_embeddings = embeddings[labels == 0]
    positive_embeddings = embeddings[labels == 1]
    if len(negative_embeddings) > 0 and len(positive_embeddings) > 0:
        metrics["centroid_distance"] = float(
            np.linalg.norm(negative_embeddings.mean(axis=0) - positive_embeddings.mean(axis=0))
        )

    try:
        sample_indices = _subsample_indices(labels, max_points=min(2000, len(labels)))
        if len(np.unique(labels[sample_indices])) >= 2 and len(sample_indices) > 2:
            metrics["silhouette"] = float(silhouette_score(embeddings[sample_indices], labels[sample_indices]))
    except Exception:
        metrics["silhouette"] = 0.0
    return metrics


def _save_projection_plot(
    projection: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    colors = {0: "#1f77b4", 1: "#d62728"}
    names = {0: "negative", 1: "positive"}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=10,
            alpha=0.65,
            c=colors.get(int(label), "#7f7f7f"),
            label=names.get(int(label), f"class_{int(label)}"),
            edgecolors="none",
        )
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _resolve_analysis_mask(graph, split: str) -> tuple[torch.Tensor, str]:
    if split == "train" and "train_supervised_mask" in graph.ndata:
        return graph.ndata["train_supervised_mask"].bool(), "train_supervised_mask"
    key = f"{split}_mask"
    if key not in graph.ndata:
        raise KeyError(f"No mask named '{key}' exists for split '{split}'.")
    return graph.ndata[key].bool(), key


def _export_single_embedding_family(
    dataset: str,
    split: str,
    family: str,
    node_ids: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    split_dir: Path,
    max_plot_points: int,
) -> dict:
    family_prefix = split_dir / f"{dataset}_{split}_{family}"
    np.save(family_prefix.with_name(family_prefix.name + "_node_ids.npy"), node_ids)
    np.save(family_prefix.with_name(family_prefix.name + "_labels.npy"), labels)
    np.save(family_prefix.with_name(family_prefix.name + "_embeddings.npy"), embeddings)

    sample_indices = _subsample_indices(labels, max_points=max_plot_points)
    projection, explained_variance = _project_embeddings(embeddings[sample_indices])
    plot_path = family_prefix.with_name(family_prefix.name + "_pca.png")
    _save_projection_plot(
        projection=projection,
        labels=labels[sample_indices],
        title=f"{dataset} {split} {family} embeddings",
        output_path=plot_path,
    )

    metrics = _separation_metrics(embeddings, labels)
    metrics["pca_explained_variance_ratio"] = explained_variance
    metrics["plot_points"] = int(len(sample_indices))
    metrics["plot_file"] = str(plot_path)
    metrics["embeddings_file"] = str(family_prefix.with_name(family_prefix.name + "_embeddings.npy"))
    metrics["node_ids_file"] = str(family_prefix.with_name(family_prefix.name + "_node_ids.npy"))
    metrics["labels_file"] = str(family_prefix.with_name(family_prefix.name + "_labels.npy"))
    return metrics


def export_embedding_analysis_for_graph(
    dataset: str,
    run_id: str,
    model,
    graph,
    output_root: Path,
    device: torch.device,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    max_plot_points: int = 4000,
) -> dict:
    analysis_root = output_root / dataset / "embedding_analysis" / run_id
    analysis_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    graph_on_device = graph.to(device)
    with torch.no_grad():
        logits, _, graph_embeddings, sequence_embeddings, fused_embeddings = model.forward_with_details(graph_on_device)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    summary = {
        "dataset": dataset,
        "run_id": run_id,
        "analysis_root": str(analysis_root),
        "splits": {},
    }

    for split in splits:
        try:
            mask, mask_source = _resolve_analysis_mask(graph_on_device, split)
        except KeyError:
            continue
        node_ids = mask.nonzero(as_tuple=False).flatten().cpu().numpy().astype(np.int64)
        labels = graph_on_device.ndata["label"][mask].cpu().numpy().astype(np.int64)
        split_dir = analysis_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_summary = {
            "mask_source": mask_source,
            "num_samples": int(mask.sum().item()),
            "positive_rate": float(labels.mean()) if len(labels) else 0.0,
            "prob_mean": float(probs[mask.cpu().numpy()].mean()) if len(labels) else 0.0,
            "prob_std": float(probs[mask.cpu().numpy()].std()) if len(labels) else 0.0,
            "families": {},
        }

        families = {
            "graph": graph_embeddings[mask].cpu().numpy(),
            "sequence": sequence_embeddings[mask].cpu().numpy(),
            "fused": fused_embeddings[mask].cpu().numpy(),
        }
        for family, embeddings in families.items():
            split_summary["families"][family] = _export_single_embedding_family(
                dataset=dataset,
                split=split,
                family=family,
                node_ids=node_ids,
                labels=labels,
                embeddings=embeddings,
                split_dir=split_dir,
                max_plot_points=max_plot_points,
            )
        summary["splits"][split] = split_summary

    summary_path = analysis_root / f"{dataset}_embedding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    summary["summary_file"] = str(summary_path)
    return summary
