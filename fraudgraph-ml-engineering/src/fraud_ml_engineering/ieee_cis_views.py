from __future__ import annotations

import copy
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

from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    _apply_active_learning_feedback,
    _apply_label_scarcity,
    _attach_dataset_context_defaults,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _stratified_partition,
)
from .ieee_cis_feature_profiles import compress_dense_feature_matrix, prepare_ieee_feature_frame
from .ieee_cis_light_builder import IEEEAssetLayout, ensure_ieee_light_assets, load_merged_subset_frame
from .ieee_cis_profiles import (
    IEEE_DEFAULT_DATA_ROOT,
    IEEE_LOADER_VIEW_GRAPH,
    IEEE_LOADER_VIEW_HYBRID,
    IEEE_LOADER_VIEW_SEQUENCE,
    IEEE_LOADER_VIEW_TABULAR,
    resolve_ieee_relation_columns,
)

NODE_TYPE = "transaction"
_VIEW_BUILD_KEYS = {
    "data_root",
    "data_profile",
    "relation_profile",
    "feature_profile",
    "history_len",
    "sampling_profile",
    "max_transactions",
    "time_bins",
    "relation_window_neighbors",
    "train_ratio",
    "valid_ratio",
    "seed",
    "rebuild_light_cache",
}
_BUNDLE_BUILD_KEYS = {
    "dataset_name",
    "num_clients",
    "seed",
    "client_hops",
    "label_fraction",
    "active_learning_feedback_path",
    "data_profile",
    "loader_view",
    "feature_profile",
    "relation_profile",
    "history_len",
}


def _split_masks_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_names = frame["split_name"].astype(str).to_numpy(dtype=object)
    return split_names == "train", split_names == "valid", split_names == "test"


def _extract_view_build_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(kwargs).items()
        if str(key) in _VIEW_BUILD_KEYS
    }


def _extract_bundle_build_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(kwargs).items()
        if str(key) in _BUNDLE_BUILD_KEYS
    }


def _normalized_data_summary(
    metadata: dict[str, Any],
    *,
    loader_view_override: str | None = None,
    source_loader_view: str | None = None,
) -> dict[str, Any]:
    if "data_summary" in metadata and isinstance(metadata.get("data_summary"), dict):
        summary = copy.deepcopy(dict(metadata.get("data_summary", {}) or {}))
    else:
        summary = copy.deepcopy(dict(metadata or {}))
    if loader_view_override:
        if source_loader_view is None:
            source_loader_view = str(summary.get("loader_view", "") or "")
        summary["loader_view"] = str(loader_view_override)
    if source_loader_view:
        summary["source_loader_view"] = str(source_loader_view)
    return summary


def _ensure_layout_and_metadata(
    *,
    data_root: str | Path,
    data_profile: str,
    loader_view: str,
    relation_profile: str,
    feature_profile: str,
    history_len: int,
    sampling_profile: str | None,
    max_transactions: int | None,
    time_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    rebuild_light_cache: bool,
) -> tuple[IEEEAssetLayout, dict[str, Any]]:
    return ensure_ieee_light_assets(
        data_root=data_root,
        data_profile=data_profile,
        loader_view=loader_view,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=history_len,
        sampling_profile=sampling_profile,
        max_transactions=max_transactions,
        time_bins=time_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        rebuild=rebuild_light_cache,
    )


def _prepare_features(
    frame: pd.DataFrame,
    *,
    feature_profile: str,
    relation_profile: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, Any], dict[str, np.ndarray]]:
    import ieee_cis_dataset as ieee_dataset

    train_mask, _, _ = _split_masks_from_frame(frame)
    prepared = prepare_ieee_feature_frame(
        frame,
        feature_profile=feature_profile,
        relation_columns=resolve_ieee_relation_columns(relation_profile),
        train_mask=np.asarray(train_mask, dtype=bool),
    )
    feature_matrix, feature_columns, feature_metadata, typed_artifacts = ieee_dataset._fit_feature_preprocessor(
        prepared.frame,
        train_mask=np.asarray(train_mask, dtype=bool),
    )
    compressed_matrix, compressed_columns, compression_meta = compress_dense_feature_matrix(
        feature_matrix,
        feature_columns,
        feature_profile=feature_profile,
        train_mask=np.asarray(train_mask, dtype=bool),
    )
    metadata = {
        **prepared.metadata,
        **feature_metadata,
        **compression_meta,
        "raw_prepared_feature_columns": [str(item) for item in feature_columns],
        "feature_columns": [str(item) for item in compressed_columns],
    }
    return prepared.frame, compressed_matrix, compressed_columns, metadata, typed_artifacts


def _write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {}
    for key, value in payload.items():
        normalized[key] = np.asarray(value) if isinstance(value, list) else value
    np.savez_compressed(path, **normalized)


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as cached:
        return {key: cached[key] for key in cached.files}


def _load_graph_from_cache(graph_path: Path, metadata_path: Path) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
    if not graph_path.exists() or not metadata_path.exists():
        return None
    graphs, _ = dgl.load_graphs(str(graph_path))
    if not graphs:
        return None
    return graphs[0], json.loads(metadata_path.read_text(encoding="utf-8-sig"))


def _write_graph_cache(
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    *,
    graph_path: Path,
    metadata_path: Path,
) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(graph_path), [graph])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def _edge_tables_from_graph(graph: dgl.DGLHeteroGraph, edge_tables_dir: Path) -> None:
    edge_tables_dir.mkdir(parents=True, exist_ok=True)
    for relation_name in graph.etypes:
        src, dst = graph.edges(etype=relation_name)
        payload = {
            "src": src.detach().cpu().numpy().astype(np.int64),
            "dst": dst.detach().cpu().numpy().astype(np.int64),
        }
        for feature_name, feature_tensor in graph.edges[relation_name].data.items():
            payload[str(feature_name)] = feature_tensor.detach().cpu().numpy()
        pd.DataFrame(payload).to_parquet(edge_tables_dir / f"{relation_name}.parquet", index=False)


def _attach_common_graph_tensors(
    graph: dgl.DGLHeteroGraph,
    *,
    frame: pd.DataFrame,
    feature_matrix: np.ndarray,
    typed_artifacts: dict[str, np.ndarray],
    edge_feature_dict: dict[str, dict[str, torch.Tensor]],
    relation_sequence_payload: dict[str, Any],
) -> None:
    train_mask, valid_mask, test_mask = _split_masks_from_frame(frame)
    labels = frame["isFraud"].fillna(0).astype(np.int64).to_numpy(dtype=np.int64)
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(np.asarray(feature_matrix, dtype=np.float32))
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(labels.astype(np.int64))
    graph.nodes[NODE_TYPE].data["train_mask"] = torch.from_numpy(train_mask.astype(bool))
    graph.nodes[NODE_TYPE].data["valid_mask"] = torch.from_numpy(valid_mask.astype(bool))
    graph.nodes[NODE_TYPE].data["test_mask"] = torch.from_numpy(test_mask.astype(bool))
    graph.nodes[NODE_TYPE].data["transaction_dt"] = torch.from_numpy(
        pd.to_numeric(frame["TransactionDT"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    )
    graph.nodes[NODE_TYPE].data["typed_numeric"] = torch.from_numpy(np.asarray(typed_artifacts["typed_numeric"], dtype=np.float32))
    graph.nodes[NODE_TYPE].data["typed_numeric_missing"] = torch.from_numpy(
        np.asarray(typed_artifacts["typed_numeric_missing"], dtype=np.float32)
    )
    graph.nodes[NODE_TYPE].data["typed_categorical"] = torch.from_numpy(
        np.asarray(typed_artifacts["typed_categorical"], dtype=np.int64)
    )
    graph.nodes[NODE_TYPE].data["typed_categorical_missing"] = torch.from_numpy(
        np.asarray(typed_artifacts["typed_categorical_missing"], dtype=np.float32)
    )
    graph.nodes[NODE_TYPE].data["typed_categorical_frequency"] = torch.from_numpy(
        np.asarray(typed_artifacts["typed_categorical_frequency"], dtype=np.float32)
    )
    graph.nodes[NODE_TYPE].data["sequence_relation_topk_indices"] = torch.from_numpy(
        np.asarray(relation_sequence_payload["topk_indices"], dtype=np.int64)
    ).long()
    graph.nodes[NODE_TYPE].data["sequence_relation_degree"] = torch.from_numpy(
        np.asarray(relation_sequence_payload["relation_degree"], dtype=np.float32)
    ).float()
    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = torch.from_numpy(train_mask.astype(bool))
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros(len(frame), dtype=torch.bool)
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.ones(len(frame), dtype=torch.float32)
    for relation_name, payload in edge_feature_dict.items():
        for feature_name, feature_tensor in payload.items():
            graph.edges[relation_name].data[str(feature_name)] = feature_tensor


def _build_graph_core(
    frame: pd.DataFrame,
    *,
    relation_profile: str,
    feature_profile: str,
    relation_window_neighbors: int,
    include_hybrid_artifacts: bool,
    history_len: int,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]]:
    import ieee_cis_dataset as ieee_dataset

    relation_columns = resolve_ieee_relation_columns(relation_profile)
    prepared_frame, feature_matrix, feature_columns, feature_metadata, typed_artifacts = _prepare_features(
        frame,
        feature_profile=feature_profile,
        relation_profile=relation_profile,
    )
    edge_dict, relation_edge_counts, relation_stats, edge_feature_dict, relation_sequence_payload = ieee_dataset._build_edge_dict(
        prepared_frame,
        relation_columns=relation_columns,
        relation_window_neighbors=int(relation_window_neighbors),
    )
    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(prepared_frame)})
    _attach_common_graph_tensors(
        graph,
        frame=prepared_frame,
        feature_matrix=feature_matrix,
        typed_artifacts=typed_artifacts,
        edge_feature_dict=edge_feature_dict,
        relation_sequence_payload=relation_sequence_payload,
    )
    temporal_context, temporal_feature_names, temporal_summary = ieee_dataset._build_temporal_context_features(prepared_frame)
    graph.nodes[NODE_TYPE].data["temporal_context"] = temporal_context.float()
    _attach_dataset_context_defaults(graph, dataset_name="ieee")
    train_mask, valid_mask, test_mask = _split_masks_from_frame(prepared_frame)
    teacher_logits = None
    teacher_summary = {"enabled": False, "teacher_name": "disabled"}
    if include_hybrid_artifacts:
        teacher_logits, teacher_summary = ieee_dataset._fit_ieee_tabular_teacher(
            typed_artifacts=typed_artifacts,
            labels=prepared_frame["isFraud"].fillna(0).astype(np.int64).to_numpy(dtype=np.int64),
            train_mask=np.asarray(train_mask, dtype=bool),
            valid_mask=np.asarray(valid_mask, dtype=bool),
        )
        if teacher_logits is not None:
            graph.nodes[NODE_TYPE].data["tabular_teacher_logits"] = torch.from_numpy(teacher_logits).float()
        event_summary = ieee_dataset._attach_ieee_event_sequence(
            graph,
            prepared_frame,
            history_len=int(history_len),
            ieee_full_compact_sequences=True,
            ieee_event_feature_dim=max(int(graph.nodes[NODE_TYPE].data["feature"].shape[1]), 1),
        )
        relation_order = ieee_dataset._attach_relation_sequence(
            graph,
            dataset_name="ieee",
            ieee_full_compact_sequences=True,
            ieee_sequence_feature_dim=max(int(graph.nodes[NODE_TYPE].data["feature"].shape[1]), 1),
        )
        sequence_summary = ieee_dataset._sequence_quality_summary(graph, relation_order, dataset_name="ieee")
    else:
        event_summary = {}
        relation_order = [
            str(item)
            for item in relation_sequence_payload["relation_order"]
            if str(item) not in {"homo", "self_loop"}
        ]
        sequence_summary = {}
    homo_src, homo_dst = graph.edges(etype="homo")
    labels_tensor = graph.nodes[NODE_TYPE].data["label"]
    graph.edges["homo"].data["label"] = torch.where(
        labels_tensor[homo_src] == labels_tensor[homo_dst],
        torch.ones_like(homo_src, dtype=torch.float32),
        -torch.ones_like(homo_src, dtype=torch.float32),
    )
    graph.edges["homo"].data["train_mask"] = (
        graph.nodes[NODE_TYPE].data["train_mask"][homo_src] & graph.nodes[NODE_TYPE].data["train_mask"][homo_dst]
    ).bool()
    metadata = {
        "data_summary": {
            "dataset": "ieee",
            "loader_view": IEEE_LOADER_VIEW_HYBRID if include_hybrid_artifacts else IEEE_LOADER_VIEW_GRAPH,
            "relation_profile": str(relation_profile),
            "feature_profile": str(feature_profile),
            "relation_columns_used": [str(item) for item in relation_order],
            "relation_candidate_columns": [str(item) for item in relation_columns],
            "feature_columns": [str(item) for item in feature_columns],
            "raw_feature_columns": [str(item) for item in feature_metadata.get("raw_feature_columns", [])],
            "feature_dim": int(feature_matrix.shape[1]),
            "typed_numeric_dim": int(typed_artifacts["typed_numeric"].shape[1]),
            "typed_categorical_dim": int(typed_artifacts["typed_categorical"].shape[1]),
            "num_nodes": int(graph.num_nodes(NODE_TYPE)),
            "relation_edge_counts": {key: int(value) for key, value in relation_edge_counts.items()},
            "relation_field_stats": copy.deepcopy(relation_stats),
            "relation_topk": int(relation_sequence_payload["relation_topk"]),
            "relation_runtime_order": [str(item) for item in relation_sequence_payload["relation_order"]],
            "train_nodes": int(np.sum(train_mask)),
            "valid_nodes": int(np.sum(valid_mask)),
            "test_nodes": int(np.sum(test_mask)),
            "positive_labels": int(prepared_frame["isFraud"].fillna(0).astype(int).sum()),
            "positive_ratio": float(prepared_frame["isFraud"].fillna(0).astype(np.float32).mean()),
            "split_strategy": "chronological_transactiondt_holdout",
            "graph_builder_version": "ieee_light_asset_graph_v1",
            "temporal_context_dim": int(temporal_summary["temporal_context_dim"]),
            "temporal_context_feature_names": [str(item) for item in temporal_feature_names],
            "temporal_windows": list(temporal_summary["temporal_windows"]),
            "tabular_teacher": teacher_summary,
            "history_len": int(history_len),
            "sequence_quality": copy.deepcopy(sequence_summary),
            "event_quality": copy.deepcopy(event_summary),
        }
    }
    return graph, metadata


def _materialize_dense_sequence_payload(graph: dgl.DGLHeteroGraph) -> dict[str, Any]:
    node_data = graph.nodes[NODE_TYPE].data
    feature_bank = node_data["feature"].detach().cpu().numpy().astype(np.float32)
    history_indices = node_data["event_history_indices"].detach().cpu().numpy().astype(np.int64)
    event_mask = node_data["event_mask"].detach().cpu().numpy().astype(bool)
    clipped_indices = np.clip(history_indices, 0, max(feature_bank.shape[0] - 1, 0))
    event_sequence = feature_bank[clipped_indices]
    event_sequence = np.where(event_mask[..., None], event_sequence, 0.0).astype(np.float32)
    return {
        "event_sequence": event_sequence,
        "event_mask": event_mask.astype(bool),
        "event_time_deltas": node_data["event_time_deltas"].detach().cpu().numpy().astype(np.float32),
        "event_token_weights": node_data["event_token_weights"].detach().cpu().numpy().astype(np.float32),
        "event_token_types": node_data["event_token_types"].detach().cpu().numpy().astype(np.int64),
        "event_source_ids": node_data["event_source_ids"].detach().cpu().numpy().astype(np.int64),
    }


def _with_basic_split_stats(payload: dict[str, Any], *, feature_key: str) -> dict[str, Any]:
    features = np.asarray(payload[feature_key])
    labels = np.asarray(payload["labels"])
    train_mask = np.asarray(payload["train_mask"]).astype(bool)
    valid_mask = np.asarray(payload["valid_mask"]).astype(bool)
    test_mask = np.asarray(payload["test_mask"]).astype(bool)
    enriched = dict(payload)
    enriched["feature_dim"] = int(features.shape[1]) if features.ndim == 2 else int(features.shape[-1])
    enriched["num_nodes"] = int(features.shape[0])
    enriched["train_size"] = int(train_mask.sum())
    enriched["valid_size"] = int(valid_mask.sum())
    enriched["test_size"] = int(test_mask.sum())
    enriched["positive_train"] = int(labels[train_mask].sum())
    enriched["positive_valid"] = int(labels[valid_mask].sum())
    enriched["positive_test"] = int(labels[test_mask].sum())
    return enriched


def build_ieee_tabular_view(
    *,
    data_root: str | Path = IEEE_DEFAULT_DATA_ROOT,
    data_profile: str,
    relation_profile: str,
    feature_profile: str,
    history_len: int,
    sampling_profile: str | None,
    max_transactions: int | None,
    time_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    rebuild_light_cache: bool = False,
) -> dict[str, Any]:
    layout, metadata = _ensure_layout_and_metadata(
        data_root=data_root,
        data_profile=data_profile,
        loader_view=IEEE_LOADER_VIEW_TABULAR,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=history_len,
        sampling_profile=sampling_profile,
        max_transactions=max_transactions,
        time_bins=time_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        rebuild_light_cache=rebuild_light_cache,
    )
    if layout.typed_static_path.exists() and not rebuild_light_cache:
        cached = _load_npz(layout.typed_static_path)
        return _with_basic_split_stats({
            "features": cached["features"].astype(np.float32),
            "labels": cached["labels"].astype(np.int32),
            "train_mask": cached["train_mask"].astype(bool),
            "valid_mask": cached["valid_mask"].astype(bool),
            "test_mask": cached["test_mask"].astype(bool),
            "typed_numeric": cached["typed_numeric"].astype(np.float32),
            "typed_numeric_missing": cached["typed_numeric_missing"].astype(np.float32),
            "typed_categorical": cached["typed_categorical"].astype(np.int64),
            "typed_categorical_missing": cached["typed_categorical_missing"].astype(np.float32),
            "typed_categorical_frequency": cached["typed_categorical_frequency"].astype(np.float32),
            "feature_names": [str(item) for item in cached["feature_names"].tolist()],
            "sample_ids": cached["sample_ids"].astype(np.int64),
            "transaction_dt": cached["transaction_dt"].astype(np.float32),
            "data_summary": _normalized_data_summary(metadata),
        }, feature_key="features")

    frame = load_merged_subset_frame(layout, metadata)
    prepared_frame, feature_matrix, feature_columns, feature_metadata, typed_artifacts = _prepare_features(
        frame,
        feature_profile=feature_profile,
        relation_profile=relation_profile,
    )
    train_mask, valid_mask, test_mask = _split_masks_from_frame(prepared_frame)
    labels = prepared_frame["isFraud"].fillna(0).astype(np.int32).to_numpy(dtype=np.int32)
    sample_ids = pd.to_numeric(prepared_frame["TransactionID"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    transaction_dt = pd.to_numeric(prepared_frame["TransactionDT"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    npz_payload = {
        "features": np.asarray(feature_matrix, dtype=np.float32),
        "labels": labels.astype(np.int32),
        "train_mask": train_mask.astype(bool),
        "valid_mask": valid_mask.astype(bool),
        "test_mask": test_mask.astype(bool),
        "typed_numeric": np.asarray(typed_artifacts["typed_numeric"], dtype=np.float32),
        "typed_numeric_missing": np.asarray(typed_artifacts["typed_numeric_missing"], dtype=np.float32),
        "typed_categorical": np.asarray(typed_artifacts["typed_categorical"], dtype=np.int64),
        "typed_categorical_missing": np.asarray(typed_artifacts["typed_categorical_missing"], dtype=np.float32),
        "typed_categorical_frequency": np.asarray(typed_artifacts["typed_categorical_frequency"], dtype=np.float32),
        "feature_names": np.asarray(feature_columns),
        "sample_ids": sample_ids.astype(np.int64),
        "transaction_dt": transaction_dt.astype(np.float32),
    }
    _write_npz(layout.typed_static_path, npz_payload)
    metadata.setdefault("feature_block_summary", {})
    metadata["feature_block_summary"]["tabular_feature_columns"] = [str(item) for item in feature_columns]
    metadata["feature_block_summary"]["tabular_feature_metadata"] = copy.deepcopy(feature_metadata)
    layout.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return _with_basic_split_stats({**npz_payload, "data_summary": _normalized_data_summary(metadata)}, feature_key="features")


def build_ieee_graph_view(
    *,
    data_root: str | Path = IEEE_DEFAULT_DATA_ROOT,
    data_profile: str,
    relation_profile: str,
    feature_profile: str,
    history_len: int,
    sampling_profile: str | None,
    max_transactions: int | None,
    time_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    rebuild_light_cache: bool = False,
    hybrid_mode: bool = False,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any], IEEEAssetLayout]:
    loader_view = IEEE_LOADER_VIEW_HYBRID if hybrid_mode else IEEE_LOADER_VIEW_GRAPH
    layout, metadata = _ensure_layout_and_metadata(
        data_root=data_root,
        data_profile=data_profile,
        loader_view=loader_view,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=history_len,
        sampling_profile=sampling_profile,
        max_transactions=max_transactions,
        time_bins=time_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        rebuild_light_cache=rebuild_light_cache,
    )
    graph_path = layout.hybrid_graph_path if hybrid_mode else layout.graph_view_path
    metadata_path = layout.hybrid_metadata_path if hybrid_mode else layout.graph_metadata_path
    if not rebuild_light_cache:
        cached = _load_graph_from_cache(graph_path, metadata_path)
        if cached is not None:
            return cached[0], cached[1], layout
    frame = load_merged_subset_frame(layout, metadata)
    graph, graph_metadata = _build_graph_core(
        frame,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        relation_window_neighbors=relation_window_neighbors,
        include_hybrid_artifacts=bool(hybrid_mode),
        history_len=history_len,
    )
    merged_summary = copy.deepcopy(metadata)
    merged_summary.update(copy.deepcopy(graph_metadata))
    merged_summary.setdefault("data_summary", {}).update(
        {
            "data_profile": str(data_profile),
            "loader_view": loader_view,
            "relation_profile": str(relation_profile),
            "feature_profile": str(feature_profile),
            "history_len": int(history_len),
            "sampling_profile": str(metadata.get("sampling_profile", "")),
            "asset_family": str(metadata.get("asset_family", "")),
        }
    )
    _write_graph_cache(
        graph,
        merged_summary,
        graph_path=graph_path,
        metadata_path=metadata_path,
    )
    if hybrid_mode:
        _edge_tables_from_graph(graph, layout.edge_tables_dir)
    return graph, merged_summary, layout


def build_ieee_sequence_view(
    *,
    data_root: str | Path = IEEE_DEFAULT_DATA_ROOT,
    data_profile: str,
    relation_profile: str,
    feature_profile: str,
    history_len: int,
    sampling_profile: str | None,
    max_transactions: int | None,
    time_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    rebuild_light_cache: bool = False,
) -> dict[str, Any]:
    layout, metadata = _ensure_layout_and_metadata(
        data_root=data_root,
        data_profile=data_profile,
        loader_view=IEEE_LOADER_VIEW_HYBRID,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=history_len,
        sampling_profile=sampling_profile,
        max_transactions=max_transactions,
        time_bins=time_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        rebuild_light_cache=rebuild_light_cache,
    )
    if layout.sequence_view_path.exists() and not rebuild_light_cache:
        cached = _load_npz(layout.sequence_view_path)
        return _with_basic_split_stats({
            "static_features": cached["static_features"].astype(np.float32),
            "labels": cached["labels"].astype(np.int32),
            "train_mask": cached["train_mask"].astype(bool),
            "valid_mask": cached["valid_mask"].astype(bool),
            "test_mask": cached["test_mask"].astype(bool),
            "event_sequence": cached["event_sequence"].astype(np.float32),
            "event_mask": cached["event_mask"].astype(bool),
            "event_time_deltas": cached["event_time_deltas"].astype(np.float32),
            "event_token_weights": cached["event_token_weights"].astype(np.float32),
            "event_token_types": cached["event_token_types"].astype(np.int64),
            "event_source_ids": cached["event_source_ids"].astype(np.int64),
            "feature_names": [str(item) for item in cached["feature_names"].tolist()],
            "sample_ids": cached["sample_ids"].astype(np.int64),
            "data_summary": _normalized_data_summary(
                metadata,
                loader_view_override=IEEE_LOADER_VIEW_SEQUENCE,
                source_loader_view=IEEE_LOADER_VIEW_HYBRID,
            ),
        }, feature_key="static_features")
    graph, metadata, layout = build_ieee_graph_view(
        data_root=data_root,
        data_profile=data_profile,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=history_len,
        sampling_profile=sampling_profile,
        max_transactions=max_transactions,
        time_bins=time_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        rebuild_light_cache=rebuild_light_cache,
        hybrid_mode=True,
    )
    node_data = graph.nodes[NODE_TYPE].data
    dense_payload = _materialize_dense_sequence_payload(graph)
    npz_payload = {
        "static_features": node_data["feature"].detach().cpu().numpy().astype(np.float32),
        "labels": node_data["label"].detach().cpu().numpy().astype(np.int32),
        "train_mask": node_data["train_mask"].detach().cpu().numpy().astype(bool),
        "valid_mask": node_data["valid_mask"].detach().cpu().numpy().astype(bool),
        "test_mask": node_data["test_mask"].detach().cpu().numpy().astype(bool),
        "event_sequence": dense_payload["event_sequence"].astype(np.float32),
        "event_mask": dense_payload["event_mask"].astype(bool),
        "event_time_deltas": dense_payload["event_time_deltas"].astype(np.float32),
        "event_token_weights": dense_payload["event_token_weights"].astype(np.float32),
        "event_token_types": dense_payload["event_token_types"].astype(np.int64),
        "event_source_ids": dense_payload["event_source_ids"].astype(np.int64),
        "feature_names": np.asarray(metadata.get("data_summary", {}).get("feature_columns", [])),
        "sample_ids": np.arange(int(graph.num_nodes(NODE_TYPE)), dtype=np.int64),
    }
    _write_npz(layout.sequence_view_path, npz_payload)
    return _with_basic_split_stats(
        {
            **npz_payload,
            "data_summary": _normalized_data_summary(
                metadata,
                loader_view_override=IEEE_LOADER_VIEW_SEQUENCE,
                source_loader_view=IEEE_LOADER_VIEW_HYBRID,
            ),
        },
        feature_key="static_features",
    )


def bundle_from_ieee_graph(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    dataset_name: str = "ieee",
    num_clients: int = 1,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    data_profile: str = "",
    loader_view: str = "",
    feature_profile: str = "",
    relation_profile: str = "",
    history_len: int = 0,
) -> DatasetBundle:
    # The graph returned by build_ieee_graph_view is already uniquely owned here.
    # Re-cloning it can double RAM use for large IEEE graphs without adding safety.
    working_graph = graph
    train_mask = working_graph.nodes[NODE_TYPE].data["train_mask"].bool()
    working_graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.clone()
    working_graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask)
    working_graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.ones(
        working_graph.num_nodes(NODE_TYPE),
        dtype=torch.float32,
    )
    if float(label_fraction) < 0.999:
        _apply_label_scarcity(working_graph, label_fraction=float(label_fraction), seed=int(seed))
    if active_learning_feedback_path:
        _apply_active_learning_feedback(working_graph, active_learning_feedback_path, dataset_name=dataset_name)
    train_supervised_mask = working_graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool() & train_mask
    train_unlabeled_mask = working_graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool() & train_mask
    supervised_nodes = train_supervised_mask.nonzero(as_tuple=False).flatten()
    supervised_labels = working_graph.nodes[NODE_TYPE].data["label"][train_supervised_mask]
    unlabeled_nodes = train_unlabeled_mask.nonzero(as_tuple=False).flatten()
    resolved_num_clients = max(int(num_clients), 1)
    clients: list[ClientShard] = []
    if resolved_num_clients == 1:
        owned_nodes = train_mask.nonzero(as_tuple=False).flatten().long()
        clients.append(
            ClientShard(
                client_id=0,
                owned_global_nodes=owned_nodes,
                subgraph=working_graph,
                train_nodes=int(owned_nodes.numel()),
            )
        )
        client_subgraph_mode = "shared_global_graph"
    else:
        supervised_partitions = _stratified_partition(
            supervised_nodes,
            supervised_labels,
            num_clients=resolved_num_clients,
            seed=int(seed),
        )
        unlabeled_partitions = _random_partition(unlabeled_nodes, num_clients=resolved_num_clients, seed=int(seed) + 1)
        owned_partitions = _merge_partitions(supervised_partitions, unlabeled_partitions)
        for client_id, owned_nodes in enumerate(owned_partitions):
            if len(owned_nodes) == 0:
                continue
            subgraph = _build_client_subgraph(working_graph, NODE_TYPE, owned_nodes, hops=int(client_hops))
            clients.append(
                ClientShard(
                    client_id=int(client_id),
                    owned_global_nodes=owned_nodes,
                    subgraph=subgraph,
                    train_nodes=int(subgraph.nodes[NODE_TYPE].data["train_mask"].sum().item()),
                )
            )
        client_subgraph_mode = "materialized_subgraph"

    class_labels = working_graph.nodes[NODE_TYPE].data["label"][working_graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()]
    if class_labels.numel() == 0:
        class_counts = torch.ones(2, dtype=torch.float32)
    else:
        class_counts = torch.bincount(class_labels.long(), minlength=2).float().clamp(min=1.0)
    class_weights = class_counts.sum() / (class_counts * len(class_counts))
    relation_order = [
        str(item)
        for item in list(dict(metadata.get("data_summary", {}) or {}).get("relation_columns_used", []))
        if str(item) not in {"homo", "self_loop"}
    ]
    bundle = DatasetBundle(
        name=str(dataset_name),
        graph=working_graph,
        node_type=NODE_TYPE,
        relation_order=relation_order,
        class_weights=class_weights,
        class_counts=class_counts,
        clients=clients,
        base_lr=1e-3,
        data_summary=copy.deepcopy(dict(metadata.get("data_summary", {}) or {})),
        data_profile=str(data_profile),
        loader_view=str(loader_view),
        feature_profile=str(feature_profile),
        relation_profile=str(relation_profile),
        history_len=int(history_len),
    )
    bundle.data_summary = copy.deepcopy(dict(metadata.get("data_summary", {}) or {}))
    bundle.data_summary["num_clients"] = int(len(clients))
    bundle.data_summary["client_subgraph_mode"] = str(client_subgraph_mode)
    bundle.data_summary["label_fraction"] = float(label_fraction)
    bundle.data_summary["active_learning_feedback_path"] = str(active_learning_feedback_path or "")
    return bundle


def load_ieee_tabular_view(**kwargs) -> dict[str, Any]:
    return build_ieee_tabular_view(**_extract_view_build_kwargs(kwargs))


def load_ieee_sequence_view(**kwargs) -> dict[str, Any]:
    return build_ieee_sequence_view(**_extract_view_build_kwargs(kwargs))


def load_ieee_graph_view(**kwargs) -> DatasetBundle:
    graph, metadata, _ = build_ieee_graph_view(**_extract_view_build_kwargs(kwargs), hybrid_mode=False)
    return bundle_from_ieee_graph(graph=graph, metadata=metadata, **_extract_bundle_build_kwargs(kwargs))


def load_ieee_hybrid_view(**kwargs) -> DatasetBundle:
    graph, metadata, _ = build_ieee_graph_view(**_extract_view_build_kwargs(kwargs), hybrid_mode=True)
    return bundle_from_ieee_graph(graph=graph, metadata=metadata, **_extract_bundle_build_kwargs(kwargs))
