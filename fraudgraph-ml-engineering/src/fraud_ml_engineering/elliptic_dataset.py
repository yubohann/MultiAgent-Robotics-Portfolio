from __future__ import annotations

"""Elliptic Bitcoin transaction loader with causal sequence/event views."""

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import dgl
except Exception as error:  # pragma: no cover - runtime env dependent
    raise RuntimeError(
        "Missing dgl dependency. Install the pinned graph-learning profile and retry:"
        "\npython -m pip install -r requirements/requirements-cpu.txt"
    ) from error
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    SEQUENCE_BUILDER_VERSION,
    _apply_active_learning_feedback,
    _attach_dataset_context_defaults,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _sequence_quality_summary,
    _stratified_partition,
)
from .paths import CACHE_ROOT, DATA_ROOT

ELLIPTIC_DEFAULT_ROOT = DATA_ROOT / "elliptic"
ELLIPTIC_CLASSES_FILENAME = "elliptic_txs_classes.csv"
ELLIPTIC_FEATURES_FILENAME = "elliptic_txs_features.csv"
ELLIPTIC_EDGES_FILENAME = "elliptic_txs_edgelist.csv"
ELLIPTIC_CACHE_DIR = CACHE_ROOT
ELLIPTIC_GRAPH_BUILDER_VERSION = f"elliptic_causal_graph_v1::{SEQUENCE_BUILDER_VERSION}"
ELLIPTIC_DEFAULT_TRAIN_TIME_END = 34
ELLIPTIC_DEFAULT_VALID_TIME_END = 39
ELLIPTIC_DEFAULT_HISTORY_LEN = 8
ELLIPTIC_DEFAULT_SEQUENCE_TOPK = 8
ELLIPTIC_DEFAULT_USE_UNKNOWN_SSL = True
ELLIPTIC_DEFAULT_COASSOCIATION_TOPK = 3
ELLIPTIC_DEFAULT_COASSOCIATION_TIME_WINDOW = 2
ELLIPTIC_SEQUENCE_COMPACT_DIM = 32
ELLIPTIC_EVENT_COMPACT_DIM = 32
ELLIPTIC_MAX_CAUSAL_DEPTH = 3
NODE_TYPE = "transaction"
UNKNOWN_LABEL_ID = -1


def _resolve_elliptic_paths(data_root: Path) -> dict[str, Path]:
    resolved_root = Path(data_root).expanduser().resolve()
    classes_path = resolved_root / ELLIPTIC_CLASSES_FILENAME
    features_path = resolved_root / ELLIPTIC_FEATURES_FILENAME
    edges_path = resolved_root / ELLIPTIC_EDGES_FILENAME
    missing_paths = [path for path in (classes_path, features_path, edges_path) if not path.exists()]
    if missing_paths:
        missing_summary = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Elliptic dataset is incomplete. Missing required files:\n"
            f"{missing_summary}"
        )
    return {
        "data_root": resolved_root,
        "classes_path": classes_path,
        "features_path": features_path,
        "edges_path": edges_path,
    }


def _elliptic_cache_signature(
    *,
    data_root: Path,
    classes_path: Path,
    features_path: Path,
    edges_path: Path,
    train_time_end: int,
    valid_time_end: int,
    history_len: int,
    sequence_topk: int,
    coassociation_topk: int,
    coassociation_time_window: int,
) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "classes_path": str(classes_path),
        "classes_mtime_ns": int(classes_path.stat().st_mtime_ns),
        "classes_size": int(classes_path.stat().st_size),
        "features_path": str(features_path),
        "features_mtime_ns": int(features_path.stat().st_mtime_ns),
        "features_size": int(features_path.stat().st_size),
        "edges_path": str(edges_path),
        "edges_mtime_ns": int(edges_path.stat().st_mtime_ns),
        "edges_size": int(edges_path.stat().st_size),
        "graph_builder_version": ELLIPTIC_GRAPH_BUILDER_VERSION,
        "train_time_end": int(train_time_end),
        "valid_time_end": int(valid_time_end),
        "history_len": int(history_len),
        "sequence_topk": int(sequence_topk),
        "coassociation_topk": int(coassociation_topk),
        "coassociation_time_window": int(coassociation_time_window),
        "sequence_compact_dim": int(ELLIPTIC_SEQUENCE_COMPACT_DIM),
        "event_compact_dim": int(ELLIPTIC_EVENT_COMPACT_DIM),
        "max_causal_depth": int(ELLIPTIC_MAX_CAUSAL_DEPTH),
    }


def _resolve_cache_paths(signature: dict[str, Any]) -> tuple[Path, Path]:
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return (
        ELLIPTIC_CACHE_DIR / f"elliptic_{digest}.dgl",
        ELLIPTIC_CACHE_DIR / f"elliptic_{digest}.json",
    )


def _legacy_cache_signature_without_coassociation(signature: dict[str, Any]) -> dict[str, Any]:
    legacy_signature = dict(signature)
    legacy_signature.pop("coassociation_topk", None)
    legacy_signature.pop("coassociation_time_window", None)
    return legacy_signature


def _load_cached_graph(
    *,
    signature: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
    if not graph_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if dict(metadata.get("cache_signature", {})) != signature:
        return None
    graph = dgl.load_graphs(str(graph_path))[0][0]
    return graph, metadata


def _write_cache(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(graph_path), [graph])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def _clone_graph_for_runtime(graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    # The loader already owns a fresh graph instance from disk/build cache,
    # so cloning here only doubles memory without adding safety.
    return graph


def _elliptic_graph_has_current_mainline_fields(graph: dgl.DGLHeteroGraph) -> bool:
    node_data = graph.nodes[NODE_TYPE].data
    return (
        "wavelet_context" in node_data
        and "coassociation_stats" in node_data
        and "coassociation" in graph.etypes
    )


def _read_elliptic_tables(classes_path: Path, features_path: Path, edges_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classes_frame = pd.read_csv(
        classes_path,
        dtype={
            "txId": np.int64,
            "class": str,
        },
    )
    feature_columns = ["txId", "time_step"] + [f"feature_{index:03d}" for index in range(165)]
    feature_dtypes: dict[str, Any] = {
        "txId": np.int64,
        "time_step": np.int16,
    }
    for column_name in feature_columns[2:]:
        feature_dtypes[column_name] = np.float32
    features_frame = pd.read_csv(
        features_path,
        header=None,
        names=feature_columns,
        dtype=feature_dtypes,
    )
    edges_frame = pd.read_csv(
        edges_path,
        dtype={
            "txId1": np.int64,
            "txId2": np.int64,
        },
    )
    return classes_frame, features_frame, edges_frame


def _encode_labels(class_values: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    label_strings = class_values.fillna("unknown").astype(str).str.strip().str.lower().to_numpy()
    labels = np.full(label_strings.shape[0], UNKNOWN_LABEL_ID, dtype=np.int64)
    licit_mask = label_strings == "2"
    illicit_mask = label_strings == "1"
    labels[licit_mask] = 0
    labels[illicit_mask] = 1
    label_known_mask = labels >= 0
    counts = {
        "known": int(label_known_mask.sum()),
        "unknown": int((~label_known_mask).sum()),
        "licit": int((labels == 0).sum()),
        "illicit": int((labels == 1).sum()),
    }
    return labels, label_known_mask, counts


def _build_edge_index(edges_frame: pd.DataFrame, tx_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_index = pd.Series(np.arange(len(tx_ids), dtype=np.int64), index=pd.Index(tx_ids))
    src = edges_frame["txId1"].map(node_index)
    dst = edges_frame["txId2"].map(node_index)
    valid_mask = src.notna() & dst.notna()
    src_index = src.loc[valid_mask].to_numpy(dtype=np.int64, copy=True)
    dst_index = dst.loc[valid_mask].to_numpy(dtype=np.int64, copy=True)
    return src_index, dst_index


def _standardize_feature_matrix(feature_matrix: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if feature_matrix.ndim != 2:
        raise ValueError("Elliptic feature matrix must be 2D.")
    if fit_mask.sum() <= 0:
        raise ValueError("Elliptic training-time fit mask is empty; cannot standardize features.")
    train_features = feature_matrix[fit_mask]
    mean = train_features.mean(axis=0, dtype=np.float64)
    std = train_features.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = ((feature_matrix - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
    return normalized, {
        "feature_mean_abs": float(np.abs(mean).mean()),
        "feature_std_mean": float(std.mean()),
    }


def _build_graph_adjacency(
    num_nodes: int,
    src_index: np.ndarray,
    dst_index: np.ndarray,
) -> tuple[list[list[int]], list[list[int]]]:
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    for src_node, dst_node in zip(src_index.tolist(), dst_index.tolist()):
        incoming[int(dst_node)].append(int(src_node))
        outgoing[int(src_node)].append(int(dst_node))
    return incoming, outgoing


def _compute_structural_features(
    *,
    num_nodes: int,
    src_index: np.ndarray,
    dst_index: np.ndarray,
    time_steps: np.ndarray,
    label_ids: np.ndarray,
    label_known_mask: np.ndarray,
    train_time_end: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[list[int]], list[list[int]]]:
    incoming, outgoing = _build_graph_adjacency(num_nodes, src_index, dst_index)
    in_degree = np.bincount(dst_index, minlength=num_nodes).astype(np.float32)
    out_degree = np.bincount(src_index, minlength=num_nodes).astype(np.float32)
    total_degree = in_degree + out_degree
    same_timestep_neighbors = np.zeros(num_nodes, dtype=np.float32)
    causal_predecessor_count = np.zeros(num_nodes, dtype=np.float32)
    causal_successor_count = np.zeros(num_nodes, dtype=np.float32)
    known_predecessor_count = np.zeros(num_nodes, dtype=np.float32)
    illicit_predecessor_count = np.zeros(num_nodes, dtype=np.float32)
    train_known_mask = label_known_mask & (time_steps <= int(train_time_end))

    for src_node, dst_node in zip(src_index.tolist(), dst_index.tolist()):
        src_node = int(src_node)
        dst_node = int(dst_node)
        src_time = int(time_steps[src_node])
        dst_time = int(time_steps[dst_node])
        if src_time == dst_time:
            same_timestep_neighbors[src_node] += 1.0
            same_timestep_neighbors[dst_node] += 1.0
        if src_time < dst_time:
            causal_predecessor_count[dst_node] += 1.0
            causal_successor_count[src_node] += 1.0
            if train_known_mask[src_node]:
                known_predecessor_count[dst_node] += 1.0
                illicit_predecessor_count[dst_node] += float(label_ids[src_node] == 1)

    time_min = float(time_steps.min()) if time_steps.size > 0 else 0.0
    time_max = float(time_steps.max()) if time_steps.size > 0 else 1.0
    time_scale = max(time_max - time_min, 1.0)
    time_norm = ((time_steps.astype(np.float32) - time_min) / time_scale).astype(np.float32)
    known_predecessor_ratio = (known_predecessor_count / np.maximum(causal_predecessor_count, 1.0)).astype(np.float32)
    illicit_predecessor_ratio = (illicit_predecessor_count / np.maximum(known_predecessor_count, 1.0)).astype(np.float32)
    in_out_balance = np.log1p(in_degree + 1.0) - np.log1p(out_degree + 1.0)
    structural_matrix = np.stack(
        [
            time_norm,
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(total_degree),
            np.log1p(causal_predecessor_count),
            np.log1p(causal_successor_count),
            np.log1p(same_timestep_neighbors),
            known_predecessor_ratio,
            illicit_predecessor_ratio,
            in_out_balance.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    structural_stats = {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "causal_predecessor_count": causal_predecessor_count,
        "causal_successor_count": causal_successor_count,
        "known_predecessor_ratio": known_predecessor_ratio.astype(np.float32),
        "illicit_predecessor_ratio": illicit_predecessor_ratio.astype(np.float32),
        "same_timestep_neighbors": same_timestep_neighbors,
        "time_norm": time_norm,
    }
    return structural_matrix, structural_stats, incoming, outgoing


def _compact_feature_bank(feature_matrix: np.ndarray, compact_dim: int) -> np.ndarray:
    feature_tensor = torch.from_numpy(feature_matrix).float()
    if feature_tensor.size(1) <= int(compact_dim):
        return feature_tensor.numpy()
    compact = F.adaptive_avg_pool1d(feature_tensor.unsqueeze(1), int(compact_dim)).squeeze(1)
    return compact.cpu().numpy().astype(np.float32, copy=False)


def _collect_causal_history(
    node_id: int,
    incoming: list[list[int]],
    time_steps: np.ndarray,
    *,
    max_items: int,
    max_depth: int = ELLIPTIC_MAX_CAUSAL_DEPTH,
) -> list[tuple[int, int]]:
    target_time = int(time_steps[node_id])
    visited = {int(node_id)}
    frontier = [(int(candidate), 1) for candidate in incoming[node_id] if int(time_steps[int(candidate)]) < target_time]
    collected: list[tuple[int, int]] = []
    depth = 1
    while frontier and len(collected) < max_items and depth <= int(max_depth):
        frontier.sort(key=lambda item: (int(time_steps[item[0]]), -int(item[1]), item[0]), reverse=True)
        next_frontier: list[tuple[int, int]] = []
        for candidate, candidate_depth in frontier:
            if candidate in visited:
                continue
            visited.add(candidate)
            collected.append((candidate, int(candidate_depth)))
            if len(collected) >= max_items:
                break
            if int(candidate_depth) >= int(max_depth):
                continue
            for predecessor in incoming[int(candidate)]:
                predecessor = int(predecessor)
                if predecessor in visited or int(time_steps[predecessor]) >= target_time:
                    continue
                next_frontier.append((predecessor, int(candidate_depth) + 1))
        frontier = next_frontier
        depth += 1
    collected.sort(key=lambda item: (int(time_steps[item[0]]), int(item[1]), int(item[0])))
    return collected[:max_items]


def _sequence_semantic_channels(
    *,
    token_type_value: float,
    relation_rank: float,
    time_delta_value: float,
    depth_value: float,
    in_degree_value: float,
    out_degree_value: float,
    risk_value: float,
    position_value: float,
) -> np.ndarray:
    return np.asarray(
        [
            float(token_type_value),
            float(relation_rank),
            float(time_delta_value),
            float(depth_value),
            float(in_degree_value),
            float(out_degree_value),
            float(risk_value),
            float(position_value),
        ],
        dtype=np.float32,
    )


def _build_causal_sequence_and_event_payload(
    *,
    time_steps: np.ndarray,
    compact_sequence_features: np.ndarray,
    compact_event_features: np.ndarray,
    structural_stats: dict[str, np.ndarray],
    incoming: list[list[int]],
    history_len: int,
    sequence_topk: int,
) -> dict[str, np.ndarray]:
    num_nodes = int(compact_sequence_features.shape[0])
    compact_sequence_dim = int(compact_sequence_features.shape[1])
    sequence_length = int(max(history_len, 1) + 2)
    event_length = int(np.clip(max(sequence_topk, 1), 3, 6))
    candidate_limit = int(max(history_len, sequence_topk, 1))
    max_time_delta = float(max(int(time_steps.max()) - int(time_steps.min()), 1))
    max_causal_depth = float(max(ELLIPTIC_MAX_CAUSAL_DEPTH, 1))
    in_degree = structural_stats["in_degree"]
    out_degree = structural_stats["out_degree"]
    illicit_predecessor_ratio = structural_stats["illicit_predecessor_ratio"]
    time_norm = structural_stats["time_norm"]
    sequence = np.zeros((num_nodes, sequence_length, compact_sequence_dim + 8), dtype=np.float32)
    sequence_mask = np.zeros((num_nodes, sequence_length), dtype=np.bool_)
    sequence_token_weights = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    sequence_token_types = np.zeros((num_nodes, sequence_length), dtype=np.int64)
    sequence_relation_ids = np.zeros((num_nodes, sequence_length), dtype=np.int64)
    event_history_indices = np.full((num_nodes, event_length), -1, dtype=np.int64)
    event_mask = np.zeros((num_nodes, event_length), dtype=np.bool_)
    event_time_deltas = np.zeros((num_nodes, event_length), dtype=np.float32)
    event_token_weights = np.zeros((num_nodes, event_length), dtype=np.float32)
    event_token_types = np.zeros((num_nodes, event_length), dtype=np.int64)
    event_source_ids = np.zeros((num_nodes, event_length), dtype=np.int64)
    temporal_context = np.zeros((num_nodes, 8), dtype=np.float32)
    wavelet_context = np.zeros((num_nodes, 8), dtype=np.float32)

    for node_id in range(num_nodes):
        causal_history = _collect_causal_history(
            node_id,
            incoming,
            time_steps,
            max_items=candidate_limit,
            max_depth=ELLIPTIC_MAX_CAUSAL_DEPTH,
        )
        sequence_history = causal_history[-int(history_len) :] if len(causal_history) > int(history_len) else causal_history
        recent_event_history = causal_history[-max(int(event_length) - 1, 0) :]
        global_summary = (
            compact_sequence_features[[candidate for candidate, _ in sequence_history]].mean(axis=0)
            if sequence_history
            else compact_sequence_features[node_id]
        )

        self_channels = _sequence_semantic_channels(
            token_type_value=0.0,
            relation_rank=0.0,
            time_delta_value=0.0,
            depth_value=0.0,
            in_degree_value=float(np.log1p(in_degree[node_id])),
            out_degree_value=float(np.log1p(out_degree[node_id])),
            risk_value=float(illicit_predecessor_ratio[node_id]),
            position_value=0.0,
        )
        sequence[node_id, 0, :compact_sequence_dim] = compact_sequence_features[node_id]
        sequence[node_id, 0, compact_sequence_dim:] = self_channels
        sequence_mask[node_id, 0] = True
        sequence_token_weights[node_id, 0] = 1.0
        sequence_token_types[node_id, 0] = 0
        sequence_relation_ids[node_id, 0] = 0

        for history_offset, (predecessor_id, predecessor_depth) in enumerate(sequence_history, start=1):
            predecessor_time = float(time_steps[predecessor_id])
            current_time = float(time_steps[node_id])
            time_delta = max(current_time - predecessor_time, 0.0)
            time_delta_norm = float(np.clip(np.log1p(time_delta) / np.log1p(max_time_delta), 0.0, 1.0))
            relation_rank = float(history_offset / max(len(sequence_history), 1))
            position_value = float(history_offset / max(sequence_length - 1, 1))
            semantic_channels = _sequence_semantic_channels(
                token_type_value=0.25 if predecessor_depth <= 1 else 0.50,
                relation_rank=relation_rank,
                time_delta_value=time_delta_norm,
                depth_value=float(np.clip(predecessor_depth / max_causal_depth, 0.0, 1.0)),
                in_degree_value=float(np.clip(np.log1p(in_degree[predecessor_id]) / 6.0, 0.0, 1.0)),
                out_degree_value=float(np.clip(np.log1p(out_degree[predecessor_id]) / 6.0, 0.0, 1.0)),
                risk_value=float(illicit_predecessor_ratio[predecessor_id]),
                position_value=position_value,
            )
            sequence[node_id, history_offset, :compact_sequence_dim] = compact_sequence_features[predecessor_id]
            sequence[node_id, history_offset, compact_sequence_dim:] = semantic_channels
            sequence_mask[node_id, history_offset] = True
            sequence_token_weights[node_id, history_offset] = float(1.0 / (1.0 + time_delta_norm))
            sequence_token_types[node_id, history_offset] = 1 if predecessor_depth <= 1 else 2
            sequence_relation_ids[node_id, history_offset] = 1

        global_channels = _sequence_semantic_channels(
            token_type_value=1.0,
            relation_rank=1.0,
            time_delta_value=0.0,
            depth_value=0.0,
            in_degree_value=float(np.clip(np.log1p(in_degree[node_id]) / 6.0, 0.0, 1.0)),
            out_degree_value=float(np.clip(np.log1p(out_degree[node_id]) / 6.0, 0.0, 1.0)),
            risk_value=float(illicit_predecessor_ratio[node_id]),
            position_value=1.0,
        )
        sequence[node_id, sequence_length - 1, :compact_sequence_dim] = global_summary
        sequence[node_id, sequence_length - 1, compact_sequence_dim:] = global_channels
        sequence_mask[node_id, sequence_length - 1] = True
        sequence_token_weights[node_id, sequence_length - 1] = 1.0
        sequence_token_types[node_id, sequence_length - 1] = 4
        sequence_relation_ids[node_id, sequence_length - 1] = 0

        temporal_context[node_id, :] = np.asarray(
            [
                float(time_norm[node_id]),
                float(np.clip(np.log1p(structural_stats["causal_predecessor_count"][node_id]) / 6.0, 0.0, 1.0)),
                float(np.clip(np.log1p(structural_stats["causal_successor_count"][node_id]) / 6.0, 0.0, 1.0)),
                float(np.clip(np.log1p(in_degree[node_id]) / 6.0, 0.0, 1.0)),
                float(np.clip(np.log1p(out_degree[node_id]) / 6.0, 0.0, 1.0)),
                float(np.clip(np.log1p(structural_stats["same_timestep_neighbors"][node_id]) / 6.0, 0.0, 1.0)),
                float(structural_stats["known_predecessor_ratio"][node_id]),
                float(illicit_predecessor_ratio[node_id]),
            ],
            dtype=np.float32,
        )

        history_time_series: list[float] = []
        history_risk_series: list[float] = []
        history_weight_series: list[float] = []
        history_depth_series: list[float] = []
        for predecessor_id, predecessor_depth in sequence_history:
            predecessor_time = float(time_steps[predecessor_id])
            current_time = float(time_steps[node_id])
            time_delta = max(current_time - predecessor_time, 0.0)
            time_delta_norm = float(np.clip(np.log1p(time_delta) / np.log1p(max_time_delta), 0.0, 1.0))
            history_time_series.append(time_delta_norm)
            history_risk_series.append(float(illicit_predecessor_ratio[predecessor_id]))
            history_weight_series.append(float(1.0 / (1.0 + time_delta_norm)))
            history_depth_series.append(float(np.clip(predecessor_depth / max_causal_depth, 0.0, 1.0)))
        if history_time_series:
            history_time_array = np.asarray(history_time_series, dtype=np.float32)
            history_risk_array = np.asarray(history_risk_series, dtype=np.float32)
            history_weight_array = np.asarray(history_weight_series, dtype=np.float32)
            history_depth_array = np.asarray(history_depth_series, dtype=np.float32)
            split_index = max(int(np.ceil(history_time_array.size / 2.0)), 1)
            early_slice = slice(0, split_index)
            late_slice = slice(split_index, None)
            early_delta = float(history_time_array[early_slice].mean())
            late_delta = float(history_time_array[late_slice].mean()) if split_index < history_time_array.size else early_delta
            early_risk = float(history_risk_array[early_slice].mean())
            late_risk = float(history_risk_array[late_slice].mean()) if split_index < history_risk_array.size else early_risk
            early_weight = float(history_weight_array[early_slice].mean())
            late_weight = float(history_weight_array[late_slice].mean()) if split_index < history_weight_array.size else early_weight
            depth_mean = float(history_depth_array.mean())
            coverage = float(min(history_time_array.size / max(history_len, 1), 1.0))
            wavelet_context[node_id, :] = np.asarray(
                [
                    0.5 * (early_delta + late_delta),
                    early_delta - late_delta,
                    0.5 * (early_risk + late_risk),
                    early_risk - late_risk,
                    0.5 * (early_weight + late_weight),
                    early_weight - late_weight,
                    depth_mean,
                    coverage,
                ],
                dtype=np.float32,
            )
        else:
            wavelet_context[node_id, :] = np.asarray(
                [
                    float(time_norm[node_id]),
                    0.0,
                    float(illicit_predecessor_ratio[node_id]),
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )

        event_insert_position = max(event_length - len(recent_event_history) - 1, 0)
        for history_offset, (predecessor_id, predecessor_depth) in enumerate(recent_event_history):
            slot_index = event_insert_position + history_offset
            predecessor_time = float(time_steps[predecessor_id])
            current_time = float(time_steps[node_id])
            time_delta = max(current_time - predecessor_time, 0.0)
            time_delta_log = float(np.log1p(time_delta))
            event_history_indices[node_id, slot_index] = int(predecessor_id)
            event_mask[node_id, slot_index] = True
            event_time_deltas[node_id, slot_index] = time_delta_log
            event_token_weights[node_id, slot_index] = float(1.0 / (1.0 + time_delta_log))
            event_token_types[node_id, slot_index] = 1 if predecessor_depth <= 1 else 2
            event_source_ids[node_id, slot_index] = int(min(predecessor_depth, 4))

        event_history_indices[node_id, event_length - 1] = int(node_id)
        event_mask[node_id, event_length - 1] = True
        event_time_deltas[node_id, event_length - 1] = 0.0
        event_token_weights[node_id, event_length - 1] = 1.0
        event_token_types[node_id, event_length - 1] = 0
        event_source_ids[node_id, event_length - 1] = 0

    return {
        "sequence": sequence,
        "sequence_mask": sequence_mask,
        "sequence_token_weights": sequence_token_weights,
        "sequence_token_types": sequence_token_types,
        "sequence_relation_ids": sequence_relation_ids,
        "event_history_indices": event_history_indices,
        "event_mask": event_mask,
        "event_time_deltas": event_time_deltas,
        "event_token_weights": event_token_weights,
        "event_token_types": event_token_types,
        "event_source_ids": event_source_ids,
        "temporal_context": temporal_context,
        "wavelet_context": wavelet_context,
        "event_base_feature": compact_event_features.astype(np.float32, copy=False),
    }


def _build_coassociation_edges(
    *,
    time_steps: np.ndarray,
    train_time_end: int,
    unknown_train_mask: np.ndarray,
    known_train_mask: np.ndarray,
    structural_stats: dict[str, np.ndarray],
    topk: int,
    time_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = int(time_steps.shape[0])
    empty_int = np.empty(0, dtype=np.int64)
    empty_float = np.empty(0, dtype=np.float32)
    empty_node_stats = np.zeros((num_nodes, 2), dtype=np.float32)
    if int(topk) <= 0:
        return empty_int, empty_int, empty_float, empty_float, empty_node_stats
    candidate_mask = time_steps <= int(train_time_end)
    candidate_nodes = np.flatnonzero(candidate_mask)
    unknown_nodes = np.flatnonzero(unknown_train_mask)
    if candidate_nodes.size == 0 or unknown_nodes.size == 0:
        return empty_int, empty_int, empty_float, empty_float, empty_node_stats
    in_bucket = np.clip(np.floor(np.log1p(structural_stats["in_degree"])).astype(np.int64), 0, 7)
    out_bucket = np.clip(np.floor(np.log1p(structural_stats["out_degree"])).astype(np.int64), 0, 7)
    known_ratio_bucket = np.clip(np.round(structural_stats["known_predecessor_ratio"] * 4.0).astype(np.int64), 0, 4)
    illicit_ratio = structural_stats["illicit_predecessor_ratio"].astype(np.float32)
    known_ratio = structural_stats["known_predecessor_ratio"].astype(np.float32)
    total_log_degree = np.log1p(structural_stats["in_degree"] + structural_stats["out_degree"]).astype(np.float32)
    bucket_index: dict[tuple[int, int, int, int], list[int]] = {}
    unknown_group_index: dict[tuple[int, int, int, int], list[int]] = {}
    for node_id in candidate_nodes.tolist():
        key = (
            int(time_steps[node_id]),
            int(in_bucket[node_id]),
            int(out_bucket[node_id]),
            int(known_ratio_bucket[node_id]),
        )
        bucket_index.setdefault(key, []).append(int(node_id))
        if bool(unknown_train_mask[node_id]):
            unknown_group_index.setdefault(key, []).append(int(node_id))

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_score: list[float] = []
    edge_delta_t: list[float] = []
    node_support = np.zeros(num_nodes, dtype=np.float32)
    node_density = np.zeros(num_nodes, dtype=np.float32)
    max_log_degree_gap = float(np.log1p(structural_stats["in_degree"].max() + structural_stats["out_degree"].max()) + 1.0)
    score_threshold = 0.25
    for group_key, group_unknown_nodes in unknown_group_index.items():
        group_time, group_in_bucket, group_out_bucket, group_known_bucket = group_key
        candidate_lists = [
            bucket_index.get(
                (
                    int(group_time) + int(delta_step),
                    int(group_in_bucket),
                    int(group_out_bucket),
                    int(group_known_bucket),
                ),
                [],
            )
            for delta_step in range(-int(time_window), int(time_window) + 1)
        ]
        candidate_lists = [candidate_list for candidate_list in candidate_lists if candidate_list]
        if not candidate_lists:
            continue
        candidate_ids_np = np.concatenate(
            [np.asarray(candidate_list, dtype=np.int64) for candidate_list in candidate_lists],
            axis=0,
        )
        if candidate_ids_np.size == 0:
            continue

        candidate_ids_t = torch.from_numpy(candidate_ids_np)
        candidate_time_t = torch.from_numpy(time_steps[candidate_ids_np].astype(np.float32, copy=False)).view(1, -1)
        candidate_log_degree_t = torch.from_numpy(total_log_degree[candidate_ids_np]).view(1, -1)
        candidate_known_ratio_t = torch.from_numpy(known_ratio[candidate_ids_np]).view(1, -1)
        candidate_illicit_ratio_t = torch.from_numpy(illicit_ratio[candidate_ids_np]).view(1, -1)
        candidate_bonus_t = torch.from_numpy(known_train_mask[candidate_ids_np].astype(np.float32, copy=False)).view(1, -1)

        matrix_element_budget = 4_000_000
        batch_size = int(max(64, min(512, matrix_element_budget // max(int(candidate_ids_np.size), 1))))
        batch_size = min(batch_size, max(len(group_unknown_nodes), 1))
        topk_count = min(int(topk), int(candidate_ids_np.size))
        if topk_count <= 0:
            continue

        for start_index in range(0, len(group_unknown_nodes), batch_size):
            batch_nodes_np = np.asarray(group_unknown_nodes[start_index : start_index + batch_size], dtype=np.int64)
            if batch_nodes_np.size == 0:
                continue
            batch_node_ids_t = torch.from_numpy(batch_nodes_np).view(-1, 1)
            batch_time_t = torch.from_numpy(time_steps[batch_nodes_np].astype(np.float32, copy=False)).view(-1, 1)
            batch_log_degree_t = torch.from_numpy(total_log_degree[batch_nodes_np]).view(-1, 1)
            batch_known_ratio_t = torch.from_numpy(known_ratio[batch_nodes_np]).view(-1, 1)
            batch_illicit_ratio_t = torch.from_numpy(illicit_ratio[batch_nodes_np]).view(-1, 1)

            score_matrix = 0.45 * (1.0 / (1.0 + torch.abs(candidate_time_t - batch_time_t)))
            score_matrix = score_matrix + 0.25 * (
                1.0 - torch.clamp(torch.abs(candidate_log_degree_t - batch_log_degree_t) / max(max_log_degree_gap, 1e-6), max=1.0)
            )
            score_matrix = score_matrix + 0.15 * (
                1.0 - torch.clamp(torch.abs(candidate_known_ratio_t - batch_known_ratio_t), max=1.0)
            )
            score_matrix = score_matrix + 0.15 * (
                1.0 - torch.clamp(torch.abs(candidate_illicit_ratio_t - batch_illicit_ratio_t), max=1.0)
            )
            score_matrix = score_matrix + 0.10 * candidate_bonus_t
            score_matrix = score_matrix.masked_fill(candidate_ids_t.view(1, -1) == batch_node_ids_t, -1.0)
            topk_scores_t, topk_positions_t = torch.topk(score_matrix, k=topk_count, dim=1)

            topk_scores_np = topk_scores_t.cpu().numpy()
            topk_positions_np = topk_positions_t.cpu().numpy()
            for row_index, node_id in enumerate(batch_nodes_np.tolist()):
                valid_mask = topk_scores_np[row_index] >= float(score_threshold)
                if not valid_mask.any():
                    continue
                selected_positions = topk_positions_np[row_index][valid_mask]
                selected_scores = topk_scores_np[row_index][valid_mask].astype(np.float32, copy=False)
                selected_candidate_ids = candidate_ids_np[selected_positions]
                selected_time_gaps = np.abs(time_steps[selected_candidate_ids] - time_steps[node_id]).astype(
                    np.float32,
                    copy=False,
                )
                for candidate_id, score, time_gap in zip(
                    selected_candidate_ids.tolist(),
                    selected_scores.tolist(),
                    selected_time_gaps.tolist(),
                ):
                    edge_src.extend([int(node_id), int(candidate_id)])
                    edge_dst.extend([int(candidate_id), int(node_id)])
                    edge_score.extend([float(score), float(score)])
                    edge_delta_t.extend([float(time_gap), float(time_gap)])
                    node_support[int(node_id)] += float(score)
                    node_support[int(candidate_id)] += float(score)
                    node_density[int(node_id)] += 1.0
                    node_density[int(candidate_id)] += 1.0
    return (
        np.asarray(edge_src, dtype=np.int64),
        np.asarray(edge_dst, dtype=np.int64),
        np.asarray(edge_score, dtype=np.float32),
        np.asarray(edge_delta_t, dtype=np.float32),
        np.stack([node_support, node_density], axis=1).astype(np.float32),
    )


def _wavelet_context_from_cached_graph(graph: dgl.DGLHeteroGraph) -> np.ndarray:
    node_data = graph.nodes[NODE_TYPE].data
    if "wavelet_context" in node_data:
        return node_data["wavelet_context"].cpu().numpy().astype(np.float32, copy=False)
    temporal_context = node_data["temporal_context"].cpu().numpy().astype(np.float32, copy=False)
    sequence = node_data["sequence"][:, :, -8:].cpu().numpy().astype(np.float32, copy=False)
    sequence_mask = node_data["sequence_mask"].cpu().numpy().astype(np.bool_, copy=False)
    sequence_token_types = node_data["sequence_token_types"].cpu().numpy().astype(np.int64, copy=False)
    sequence_token_weights = node_data["sequence_token_weights"].cpu().numpy().astype(np.float32, copy=False)
    num_nodes = int(sequence.shape[0])
    wavelet_context = np.zeros((num_nodes, 8), dtype=np.float32)

    history_mask = sequence_mask & (sequence_token_types != 0) & (sequence_token_types != 4)
    time_series = sequence[:, :, 2]
    risk_series = sequence[:, :, 6]
    depth_series = sequence[:, :, 3]

    for node_id in range(num_nodes):
        valid_positions = np.flatnonzero(history_mask[node_id])
        if valid_positions.size <= 0:
            wavelet_context[node_id, :] = np.asarray(
                [
                    float(temporal_context[node_id, 0]),
                    0.0,
                    float(temporal_context[node_id, 7]),
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            continue
        history_time_array = time_series[node_id, valid_positions]
        history_risk_array = risk_series[node_id, valid_positions]
        history_weight_array = sequence_token_weights[node_id, valid_positions]
        history_depth_array = depth_series[node_id, valid_positions]
        split_index = max(int(np.ceil(history_time_array.size / 2.0)), 1)
        early_slice = slice(0, split_index)
        late_slice = slice(split_index, None)
        early_delta = float(history_time_array[early_slice].mean())
        late_delta = float(history_time_array[late_slice].mean()) if split_index < history_time_array.size else early_delta
        early_risk = float(history_risk_array[early_slice].mean())
        late_risk = float(history_risk_array[late_slice].mean()) if split_index < history_risk_array.size else early_risk
        early_weight = float(history_weight_array[early_slice].mean())
        late_weight = float(history_weight_array[late_slice].mean()) if split_index < history_weight_array.size else early_weight
        depth_mean = float(history_depth_array.mean())
        coverage = float(min(history_time_array.size / max(sequence.shape[1] - 2, 1), 1.0))
        wavelet_context[node_id, :] = np.asarray(
            [
                0.5 * (early_delta + late_delta),
                early_delta - late_delta,
                0.5 * (early_risk + late_risk),
                early_risk - late_risk,
                0.5 * (early_weight + late_weight),
                early_weight - late_weight,
                depth_mean,
                coverage,
            ],
            dtype=np.float32,
        )
    return wavelet_context


def _graph_with_coassociation_relation(
    graph: dgl.DGLHeteroGraph,
    *,
    coassoc_src: np.ndarray,
    coassoc_dst: np.ndarray,
    coassoc_score: np.ndarray,
    coassoc_delta_t: np.ndarray,
) -> dgl.DGLHeteroGraph:
    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    edge_data: dict[str, dict[str, torch.Tensor]] = {}
    for edge_type in graph.etypes:
        src_nodes, dst_nodes = graph.edges(etype=edge_type)
        edge_dict[(NODE_TYPE, edge_type, NODE_TYPE)] = (src_nodes, dst_nodes)
        edge_data[edge_type] = dict(graph.edges[edge_type].data)
    edge_dict[(NODE_TYPE, "coassociation", NODE_TYPE)] = (
        torch.from_numpy(coassoc_src.astype(np.int64, copy=True)),
        torch.from_numpy(coassoc_dst.astype(np.int64, copy=True)),
    )
    upgraded_graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: graph.num_nodes(NODE_TYPE)})
    for key, value in graph.nodes[NODE_TYPE].data.items():
        upgraded_graph.nodes[NODE_TYPE].data[key] = value
    for edge_type, data in edge_data.items():
        for key, value in data.items():
            upgraded_graph.edges[edge_type].data[key] = value
    upgraded_graph.edges["coassociation"].data["weight"] = torch.from_numpy(coassoc_score.astype(np.float32, copy=False))
    upgraded_graph.edges["coassociation"].data["delta_t"] = torch.from_numpy(coassoc_delta_t.astype(np.float32, copy=False))
    return upgraded_graph


def _upgrade_legacy_cached_graph(
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    *,
    cache_signature: dict[str, Any],
    train_time_end: int,
    coassociation_topk: int,
    coassociation_time_window: int,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]]:
    if _elliptic_graph_has_current_mainline_fields(graph):
        upgraded_metadata = dict(metadata)
        upgraded_metadata["cache_signature"] = dict(cache_signature)
        upgraded_metadata["data_summary"] = dict(metadata.get("data_summary", {}) or {})
        return graph, upgraded_metadata

    node_data = graph.nodes[NODE_TYPE].data
    time_steps = node_data["time_step"].cpu().numpy().astype(np.int64, copy=False)
    label_ids = node_data["label"].cpu().numpy().astype(np.int64, copy=False)
    label_known_mask = node_data["label_known_mask"].cpu().numpy().astype(np.bool_, copy=False)
    forward_src, forward_dst = graph.edges(etype="causal_forward")
    _, structural_stats, _, _ = _compute_structural_features(
        num_nodes=int(graph.num_nodes(NODE_TYPE)),
        src_index=forward_src.cpu().numpy().astype(np.int64, copy=False),
        dst_index=forward_dst.cpu().numpy().astype(np.int64, copy=False),
        time_steps=time_steps,
        label_ids=label_ids,
        label_known_mask=label_known_mask,
        train_time_end=int(train_time_end),
    )
    coassoc_src, coassoc_dst, coassoc_score, coassoc_delta_t, coassociation_stats = _build_coassociation_edges(
        time_steps=time_steps,
        train_time_end=int(train_time_end),
        unknown_train_mask=node_data["unknown_train_mask"].cpu().numpy().astype(np.bool_, copy=False),
        known_train_mask=node_data["known_train_mask"].cpu().numpy().astype(np.bool_, copy=False),
        structural_stats=structural_stats,
        topk=int(coassociation_topk),
        time_window=int(coassociation_time_window),
    )
    upgraded_graph = _graph_with_coassociation_relation(
        graph,
        coassoc_src=coassoc_src,
        coassoc_dst=coassoc_dst,
        coassoc_score=coassoc_score,
        coassoc_delta_t=coassoc_delta_t,
    )
    upgraded_graph.nodes[NODE_TYPE].data["wavelet_context"] = torch.from_numpy(_wavelet_context_from_cached_graph(graph)).float()
    upgraded_graph.nodes[NODE_TYPE].data["coassociation_stats"] = torch.from_numpy(coassociation_stats).float()
    _refresh_homo_edge_train_mask(upgraded_graph)

    upgraded_metadata = dict(metadata)
    upgraded_metadata["cache_signature"] = dict(cache_signature)
    data_summary = dict(metadata.get("data_summary", {}) or {})
    cache_graph_path, cache_metadata_path = _resolve_cache_paths(cache_signature)
    data_summary["wavelet_context_dim"] = int(upgraded_graph.nodes[NODE_TYPE].data["wavelet_context"].shape[1])
    data_summary["coassociation_topk"] = int(coassociation_topk)
    data_summary["coassociation_time_window"] = int(coassociation_time_window)
    data_summary["num_coassociation_edges"] = int(coassoc_src.size)
    data_summary["relation_order"] = list(upgraded_graph.etypes)
    data_summary["cache_graph_path"] = str(cache_graph_path)
    data_summary["cache_metadata_path"] = str(cache_metadata_path)
    data_summary["sequence_quality"] = _sequence_quality_summary(
        upgraded_graph,
        list(upgraded_graph.etypes),
        dataset_name="elliptic",
    )
    upgraded_metadata["data_summary"] = data_summary
    return upgraded_graph, upgraded_metadata


def _homo_edge_labels(
    src_index: np.ndarray,
    dst_index: np.ndarray,
    labels: np.ndarray,
) -> torch.Tensor:
    same_label = labels[src_index] == labels[dst_index]
    edge_labels = np.where(same_label, 1, -1).astype(np.int64)
    return torch.from_numpy(edge_labels)


def _refresh_homo_edge_train_mask(graph: dgl.DGLHeteroGraph) -> None:
    if "homo" not in graph.etypes:
        return
    node_data = graph.nodes[NODE_TYPE].data
    supervised_mask = (
        node_data["train_supervised_mask"].bool()
        if "train_supervised_mask" in node_data
        else node_data["train_mask"].bool()
    )
    src_nodes, dst_nodes = graph.edges(etype="homo")
    graph.edges["homo"].data["train_mask"] = (supervised_mask[src_nodes] & supervised_mask[dst_nodes]).bool()


def _build_runtime_train_masks(
    *,
    labels: torch.Tensor,
    known_train_mask: torch.Tensor,
    unknown_train_mask: torch.Tensor,
    label_fraction: float,
    use_unknown_ssl: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    known_train_mask = known_train_mask.bool()
    unknown_train_mask = unknown_train_mask.bool()
    supervised_mask = known_train_mask.clone()
    fraction = float(np.clip(label_fraction, 0.0, 1.0))
    if fraction < 0.999 and supervised_mask.any():
        rng = np.random.default_rng(seed)
        selected_nodes: list[int] = []
        train_labels = labels[supervised_mask].cpu().numpy()
        train_node_ids = supervised_mask.nonzero(as_tuple=False).flatten().cpu().numpy()
        for class_id in np.unique(train_labels):
            class_positions = np.flatnonzero(train_labels == class_id)
            if class_positions.size == 0:
                continue
            target_count = max(1, int(round(class_positions.size * fraction))) if fraction > 0.0 else 0
            target_count = min(target_count, int(class_positions.size))
            if target_count <= 0:
                continue
            chosen_positions = rng.choice(class_positions, size=target_count, replace=False)
            selected_nodes.extend(train_node_ids[chosen_positions].tolist())
        supervised_mask = torch.zeros_like(known_train_mask)
        if selected_nodes:
            supervised_mask[torch.as_tensor(sorted(set(selected_nodes)), dtype=torch.long)] = True
        supervised_mask &= known_train_mask
    withheld_known_mask = known_train_mask & ~supervised_mask
    unlabeled_mask = withheld_known_mask.clone()
    if bool(use_unknown_ssl):
        unlabeled_mask |= unknown_train_mask
    train_mask = supervised_mask | unlabeled_mask
    scarcity_ratio = float(supervised_mask.sum().item()) / max(float(train_mask.sum().item()), 1.0)
    label_scarcity_ratio = torch.full(
        (labels.shape[0],),
        scarcity_ratio,
        dtype=torch.float32,
    )
    return train_mask, supervised_mask, unlabeled_mask, label_scarcity_ratio


def _apply_runtime_masks(
    graph: dgl.DGLHeteroGraph,
    *,
    label_fraction: float,
    use_unknown_ssl: bool,
    seed: int,
) -> None:
    known_train_mask = graph.nodes[NODE_TYPE].data["known_train_mask"].bool()
    unknown_train_mask = graph.nodes[NODE_TYPE].data["unknown_train_mask"].bool()
    train_mask, supervised_mask, unlabeled_mask, label_scarcity_ratio = _build_runtime_train_masks(
        labels=graph.nodes[NODE_TYPE].data["label"].long(),
        known_train_mask=known_train_mask,
        unknown_train_mask=unknown_train_mask,
        label_fraction=label_fraction,
        use_unknown_ssl=use_unknown_ssl,
        seed=seed,
    )
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = supervised_mask.bool()
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = unlabeled_mask.bool()
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = label_scarcity_ratio.float()
    _refresh_homo_edge_train_mask(graph)


def _build_graph_payload(
    *,
    data_root: Path,
    classes_path: Path,
    features_path: Path,
    edges_path: Path,
    cache_signature: dict[str, Any],
    train_time_end: int,
    valid_time_end: int,
    history_len: int,
    sequence_topk: int,
    coassociation_topk: int,
    coassociation_time_window: int,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]]:
    classes_frame, features_frame, edges_frame = _read_elliptic_tables(classes_path, features_path, edges_path)
    if features_frame["txId"].duplicated().any():
        raise ValueError("Elliptic feature table contains duplicated txId values.")

    class_map = classes_frame.set_index("txId")["class"]
    class_values = class_map.reindex(features_frame["txId"]).fillna("unknown")
    label_ids, label_known_mask, class_counts = _encode_labels(class_values)
    tx_ids = features_frame["txId"].to_numpy(dtype=np.int64, copy=False)
    time_steps = features_frame["time_step"].to_numpy(dtype=np.int64, copy=False)
    raw_feature_matrix = features_frame.iloc[:, 2:].to_numpy(dtype=np.float32, copy=False)
    forward_src, forward_dst = _build_edge_index(edges_frame, tx_ids)
    num_nodes = int(raw_feature_matrix.shape[0])
    structural_features, structural_stats, incoming, _ = _compute_structural_features(
        num_nodes=num_nodes,
        src_index=forward_src,
        dst_index=forward_dst,
        time_steps=time_steps,
        label_ids=label_ids,
        label_known_mask=label_known_mask,
        train_time_end=train_time_end,
    )
    feature_matrix = np.concatenate([raw_feature_matrix, structural_features], axis=1).astype(np.float32)
    training_time_mask = time_steps <= int(train_time_end)
    normalized_feature_matrix, standardization_metadata = _standardize_feature_matrix(feature_matrix, training_time_mask)
    compact_sequence_features = _compact_feature_bank(normalized_feature_matrix, ELLIPTIC_SEQUENCE_COMPACT_DIM)
    compact_event_features = _compact_feature_bank(normalized_feature_matrix, ELLIPTIC_EVENT_COMPACT_DIM)
    sequence_event_payload = _build_causal_sequence_and_event_payload(
        time_steps=time_steps,
        compact_sequence_features=compact_sequence_features,
        compact_event_features=compact_event_features,
        structural_stats=structural_stats,
        incoming=incoming,
        history_len=history_len,
        sequence_topk=sequence_topk,
    )

    known_train_mask = (time_steps <= int(train_time_end)) & label_known_mask
    unknown_train_mask = (time_steps <= int(train_time_end)) & ~label_known_mask
    valid_mask = (time_steps > int(train_time_end)) & (time_steps <= int(valid_time_end)) & label_known_mask
    test_mask = (time_steps > int(valid_time_end)) & label_known_mask
    coassoc_src, coassoc_dst, coassoc_score, coassoc_delta_t, coassociation_stats = _build_coassociation_edges(
        time_steps=time_steps,
        train_time_end=int(train_time_end),
        unknown_train_mask=unknown_train_mask,
        known_train_mask=known_train_mask,
        structural_stats=structural_stats,
        topk=int(coassociation_topk),
        time_window=int(coassociation_time_window),
    )

    self_loop_nodes = np.arange(num_nodes, dtype=np.int64)
    homo_src = np.concatenate([forward_src, forward_dst], axis=0).astype(np.int64)
    homo_dst = np.concatenate([forward_dst, forward_src], axis=0).astype(np.int64)
    if homo_src.size == 0:
        homo_src = self_loop_nodes.copy()
        homo_dst = self_loop_nodes.copy()

    edge_dict = {
        (NODE_TYPE, "causal_forward", NODE_TYPE): (
            torch.from_numpy(forward_src.astype(np.int64, copy=True)),
            torch.from_numpy(forward_dst.astype(np.int64, copy=True)),
        ),
        (NODE_TYPE, "causal_reverse", NODE_TYPE): (
            torch.from_numpy(forward_dst.astype(np.int64, copy=True)),
            torch.from_numpy(forward_src.astype(np.int64, copy=True)),
        ),
        (NODE_TYPE, "self_loop", NODE_TYPE): (
            torch.from_numpy(self_loop_nodes),
            torch.from_numpy(self_loop_nodes.copy()),
        ),
        (NODE_TYPE, "homo", NODE_TYPE): (
            torch.from_numpy(homo_src),
            torch.from_numpy(homo_dst),
        ),
    }
    if coassoc_src.size > 0:
        edge_dict[(NODE_TYPE, "coassociation", NODE_TYPE)] = (
            torch.from_numpy(coassoc_src.astype(np.int64, copy=True)),
            torch.from_numpy(coassoc_dst.astype(np.int64, copy=True)),
        )
    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: num_nodes})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(normalized_feature_matrix).float()
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(np.where(label_ids < 0, 0, label_ids).astype(np.int64))
    graph.nodes[NODE_TYPE].data["label_known_mask"] = torch.from_numpy(label_known_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["tx_id"] = torch.from_numpy(tx_ids.astype(np.int64, copy=True))
    graph.nodes[NODE_TYPE].data["time_step"] = torch.from_numpy(time_steps.astype(np.int64))
    graph.nodes[NODE_TYPE].data["transaction_time"] = torch.from_numpy(time_steps.astype(np.float32))
    graph.nodes[NODE_TYPE].data["known_train_mask"] = torch.from_numpy(known_train_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["unknown_train_mask"] = torch.from_numpy(unknown_train_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["train_mask"] = torch.from_numpy(known_train_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = torch.from_numpy(known_train_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros(num_nodes, dtype=torch.bool)
    graph.nodes[NODE_TYPE].data["valid_mask"] = torch.from_numpy(valid_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["test_mask"] = torch.from_numpy(test_mask.astype(np.bool_))
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.ones(num_nodes, dtype=torch.float32)
    graph.nodes[NODE_TYPE].data["sequence"] = torch.from_numpy(sequence_event_payload["sequence"]).to(dtype=torch.float16)
    graph.nodes[NODE_TYPE].data["sequence_mask"] = torch.from_numpy(sequence_event_payload["sequence_mask"]).bool()
    graph.nodes[NODE_TYPE].data["sequence_token_weights"] = torch.from_numpy(
        sequence_event_payload["sequence_token_weights"]
    ).float()
    graph.nodes[NODE_TYPE].data["sequence_token_types"] = torch.from_numpy(
        sequence_event_payload["sequence_token_types"]
    ).long()
    graph.nodes[NODE_TYPE].data["sequence_relation_ids"] = torch.from_numpy(
        sequence_event_payload["sequence_relation_ids"]
    ).long()
    graph.nodes[NODE_TYPE].data["event_base_feature"] = torch.from_numpy(
        sequence_event_payload["event_base_feature"]
    ).to(dtype=torch.float16)
    graph.nodes[NODE_TYPE].data["event_history_indices"] = torch.from_numpy(
        sequence_event_payload["event_history_indices"]
    ).long()
    graph.nodes[NODE_TYPE].data["event_mask"] = torch.from_numpy(sequence_event_payload["event_mask"]).bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = torch.from_numpy(
        sequence_event_payload["event_time_deltas"]
    ).float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = torch.from_numpy(
        sequence_event_payload["event_token_weights"]
    ).float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = torch.from_numpy(
        sequence_event_payload["event_token_types"]
    ).long()
    graph.nodes[NODE_TYPE].data["event_source_ids"] = torch.from_numpy(
        sequence_event_payload["event_source_ids"]
    ).long()
    graph.nodes[NODE_TYPE].data["temporal_context"] = torch.from_numpy(sequence_event_payload["temporal_context"]).float()
    graph.nodes[NODE_TYPE].data["wavelet_context"] = torch.from_numpy(sequence_event_payload["wavelet_context"]).float()
    graph.nodes[NODE_TYPE].data["coassociation_stats"] = torch.from_numpy(coassociation_stats).float()
    graph.edges["homo"].data["label"] = _homo_edge_labels(homo_src, homo_dst, np.where(label_ids < 0, 0, label_ids))
    if "coassociation" in graph.etypes:
        graph.edges["coassociation"].data["weight"] = torch.from_numpy(coassoc_score).float()
        graph.edges["coassociation"].data["delta_t"] = torch.from_numpy(coassoc_delta_t).float()
    _refresh_homo_edge_train_mask(graph)

    relation_order = list(graph.etypes)
    cache_graph_path, cache_metadata_path = _resolve_cache_paths(cache_signature)
    data_summary = {
        "dataset": "elliptic",
        "dataset_registry_name": "elliptic",
        "dataset_display_name": "Elliptic Bitcoin Transaction Graph",
        "data_root": str(data_root),
        "num_nodes": int(num_nodes),
        "num_forward_edges": int(len(forward_src)),
        "num_homo_edges": int(len(homo_src)),
        "num_time_steps": int(np.unique(time_steps).size),
        "time_step_min": int(time_steps.min()),
        "time_step_max": int(time_steps.max()),
        "train_time_end": int(train_time_end),
        "valid_time_end": int(valid_time_end),
        "known_train_nodes": int(known_train_mask.sum()),
        "unknown_train_nodes": int(unknown_train_mask.sum()),
        "valid_nodes": int(valid_mask.sum()),
        "test_nodes": int(test_mask.sum()),
        "class_counts": class_counts,
        "feature_dim": int(normalized_feature_matrix.shape[1]),
        "raw_feature_dim": int(raw_feature_matrix.shape[1]),
        "structural_feature_dim": int(structural_features.shape[1]),
        "sequence_length": int(sequence_event_payload["sequence"].shape[1]),
        "sequence_feature_dim": int(sequence_event_payload["sequence"].shape[2]),
        "sequence_compact_dim": int(ELLIPTIC_SEQUENCE_COMPACT_DIM),
        "event_history_length": int(sequence_event_payload["event_history_indices"].shape[1]),
        "event_base_feature_dim": int(sequence_event_payload["event_base_feature"].shape[1]),
        "event_compact_dim": int(ELLIPTIC_EVENT_COMPACT_DIM),
        "temporal_context_dim": int(sequence_event_payload["temporal_context"].shape[1]),
        "wavelet_context_dim": int(sequence_event_payload["wavelet_context"].shape[1]),
        "coassociation_topk": int(coassociation_topk),
        "coassociation_time_window": int(coassociation_time_window),
        "num_coassociation_edges": int(coassoc_src.size),
        "relation_order": relation_order,
        "cache_graph_path": str(cache_graph_path),
        "cache_metadata_path": str(cache_metadata_path),
        "standardization": standardization_metadata,
        "sequence_quality": _sequence_quality_summary(graph, relation_order, dataset_name="elliptic"),
    }
    metadata = {
        "cache_signature": cache_signature,
        "data_summary": data_summary,
    }
    return graph, metadata


def load_elliptic_dataset(
    *,
    data_root: str | Path = ELLIPTIC_DEFAULT_ROOT,
    dataset_name: str = "elliptic",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    train_time_end: int = ELLIPTIC_DEFAULT_TRAIN_TIME_END,
    valid_time_end: int = ELLIPTIC_DEFAULT_VALID_TIME_END,
    history_len: int = ELLIPTIC_DEFAULT_HISTORY_LEN,
    sequence_topk: int = ELLIPTIC_DEFAULT_SEQUENCE_TOPK,
    use_unknown_ssl: bool = ELLIPTIC_DEFAULT_USE_UNKNOWN_SSL,
    coassociation_topk: int = ELLIPTIC_DEFAULT_COASSOCIATION_TOPK,
    coassociation_time_window: int = ELLIPTIC_DEFAULT_COASSOCIATION_TIME_WINDOW,
    rebuild_cache: bool = False,
) -> DatasetBundle:
    if int(train_time_end) >= int(valid_time_end):
        raise ValueError("Elliptic split boundaries must satisfy train_time_end < valid_time_end.")
    if int(history_len) <= 0:
        raise ValueError("Elliptic history_len must be positive.")
    if int(sequence_topk) <= 0:
        raise ValueError("Elliptic sequence_topk must be positive.")
    if int(coassociation_topk) < 0:
        raise ValueError("Elliptic coassociation_topk must be non-negative.")
    if int(coassociation_time_window) < 0:
        raise ValueError("Elliptic coassociation_time_window must be non-negative.")

    resolved_paths = _resolve_elliptic_paths(Path(data_root))
    signature = _elliptic_cache_signature(
        data_root=resolved_paths["data_root"],
        classes_path=resolved_paths["classes_path"],
        features_path=resolved_paths["features_path"],
        edges_path=resolved_paths["edges_path"],
        train_time_end=int(train_time_end),
        valid_time_end=int(valid_time_end),
        history_len=int(history_len),
        sequence_topk=int(sequence_topk),
        coassociation_topk=int(coassociation_topk),
        coassociation_time_window=int(coassociation_time_window),
    )
    cache_graph_path, cache_metadata_path = _resolve_cache_paths(signature)
    cached_payload = None if rebuild_cache else _load_cached_graph(
        signature=signature,
        graph_path=cache_graph_path,
        metadata_path=cache_metadata_path,
    )
    if cached_payload is not None:
        cached_graph, cached_metadata = cached_payload
        if not _elliptic_graph_has_current_mainline_fields(cached_graph):
            cached_payload = None
        else:
            graph, metadata = cached_graph, cached_metadata
    if cached_payload is None:
        legacy_signature = _legacy_cache_signature_without_coassociation(signature)
        legacy_graph_path, legacy_metadata_path = _resolve_cache_paths(legacy_signature)
        legacy_cached_payload = _load_cached_graph(
            signature=legacy_signature,
            graph_path=legacy_graph_path,
            metadata_path=legacy_metadata_path,
        )
        if legacy_cached_payload is not None:
            graph, metadata = _upgrade_legacy_cached_graph(
                legacy_cached_payload[0],
                legacy_cached_payload[1],
                cache_signature=signature,
                train_time_end=int(train_time_end),
                coassociation_topk=int(coassociation_topk),
                coassociation_time_window=int(coassociation_time_window),
            )
        else:
            graph, metadata = _build_graph_payload(
                data_root=resolved_paths["data_root"],
                classes_path=resolved_paths["classes_path"],
                features_path=resolved_paths["features_path"],
                edges_path=resolved_paths["edges_path"],
                cache_signature=signature,
                train_time_end=int(train_time_end),
                valid_time_end=int(valid_time_end),
                history_len=int(history_len),
                sequence_topk=int(sequence_topk),
                coassociation_topk=int(coassociation_topk),
                coassociation_time_window=int(coassociation_time_window),
            )
        _write_cache(
            graph=graph,
            metadata=metadata,
            graph_path=cache_graph_path,
            metadata_path=cache_metadata_path,
        )

    graph = _clone_graph_for_runtime(graph)
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)
    _apply_runtime_masks(
        graph,
        label_fraction=float(label_fraction),
        use_unknown_ssl=bool(use_unknown_ssl),
        seed=int(seed),
    )
    if active_learning_feedback_path:
        _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)
        _refresh_homo_edge_train_mask(graph)

    train_supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()
    train_unlabeled_mask = graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool()
    supervised_nodes = train_supervised_mask.nonzero(as_tuple=False).flatten()
    supervised_labels = graph.nodes[NODE_TYPE].data["label"][train_supervised_mask]
    unlabeled_nodes = train_unlabeled_mask.nonzero(as_tuple=False).flatten()
    supervised_partitions = _stratified_partition(
        supervised_nodes,
        supervised_labels,
        num_clients=max(int(num_clients), 1),
        seed=int(seed),
    )
    unlabeled_partitions = _random_partition(unlabeled_nodes, num_clients=max(int(num_clients), 1), seed=int(seed) + 1)
    owned_partitions = _merge_partitions(supervised_partitions, unlabeled_partitions)

    clients: list[ClientShard] = []
    if max(int(num_clients), 1) == 1:
        owned_nodes = owned_partitions[0] if owned_partitions else train_supervised_mask.nonzero(as_tuple=False).flatten()
        clients.append(
            ClientShard(
                client_id=0,
                owned_global_nodes=owned_nodes.long(),
                subgraph=graph,
                train_nodes=int(graph.nodes[NODE_TYPE].data["train_mask"].sum().item()),
            )
        )
    else:
        for client_id, owned_nodes in enumerate(owned_partitions):
            if len(owned_nodes) == 0:
                continue
            subgraph = _build_client_subgraph(graph, NODE_TYPE, owned_nodes, hops=max(int(client_hops), 0))
            local_train_nodes = int(subgraph.nodes[NODE_TYPE].data["train_mask"].sum().item())
            clients.append(
                ClientShard(
                    client_id=client_id,
                    owned_global_nodes=owned_nodes,
                    subgraph=subgraph,
                    train_nodes=local_train_nodes,
                )
            )

    class_labels = graph.nodes[NODE_TYPE].data["label"][graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()]
    if class_labels.numel() == 0:
        class_counts = torch.ones(2, dtype=torch.float32)
    else:
        class_counts = torch.bincount(class_labels.long(), minlength=2).float().clamp(min=1.0)
    class_weights = class_counts.sum() / (class_counts * len(class_counts))
    relation_order = list(graph.etypes)

    bundle = DatasetBundle(
        name=dataset_name,
        graph=graph,
        node_type=NODE_TYPE,
        relation_order=relation_order,
        class_weights=class_weights.float(),
        class_counts=class_counts.float(),
        clients=clients,
        base_lr=1e-3,
    )
    data_summary = dict(metadata.get("data_summary", {}) or {})
    data_summary["dataset"] = str(dataset_name)
    data_summary["dataset_registry_name"] = str(dataset_name)
    data_summary["num_clients"] = int(len(clients))
    data_summary["label_fraction"] = float(label_fraction)
    data_summary["use_unknown_ssl"] = bool(use_unknown_ssl)
    data_summary["coassociation_topk"] = int(coassociation_topk)
    data_summary["coassociation_time_window"] = int(coassociation_time_window)
    data_summary["active_learning_feedback_path"] = str(active_learning_feedback_path or "")
    data_summary["runtime_supervised_train_nodes"] = int(train_supervised_mask.sum().item())
    data_summary["runtime_unlabeled_train_nodes"] = int(train_unlabeled_mask.sum().item())
    data_summary["runtime_train_nodes"] = int(graph.nodes[NODE_TYPE].data["train_mask"].sum().item())
    data_summary["sequence_quality"] = _sequence_quality_summary(graph, relation_order, dataset_name=dataset_name)
    bundle.data_summary = data_summary
    return bundle
