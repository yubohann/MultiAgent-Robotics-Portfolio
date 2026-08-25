from __future__ import annotations

"""Draft loader for Ethereum Ponzi contracts.

This loader is intentionally strict: the public local source is positive-only,
so a usable binary classification setup still needs an external negative set.
"""

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

from .archive_dataset import (
    _compute_class_counts,
    _compute_class_weights,
    _stratified_split_masks,
    _window_edge_pairs,
)
from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    _apply_active_learning_feedback,
    _apply_label_scarcity,
    _attach_dataset_context_defaults,
    _attach_relation_sequence,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _stratified_partition,
)
from .paths import DATA_ROOT

ETHEREUM_PONZI_DEFAULT_ROOT = DATA_ROOT / "ethereum_ponzi"
NODE_TYPE = "contract"
PONZI_EVENT_SEQUENCE_LENGTH = 5
FEATURE_EXCLUDE_COLUMNS = {
    "address",
    "label",
    "label_name",
    "source_group",
    "source_path",
    "is_anchor",
    "sequence_available",
}


def _read_table(parquet_path: Path, csv_path: Path, *, force_preview: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not force_preview and parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), {"path": str(parquet_path), "format": "parquet"}
        except Exception:
            pass
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False), {"path": str(csv_path), "format": "csv"}


def _load_negative_users(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Negative user file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator, low_memory=False)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            frame = pd.DataFrame({"address": [line.strip() for line in handle if line.strip()]})

    if "address" not in frame.columns:
        raise ValueError(f"Negative user file must contain an 'address' column: {path}")
    frame = frame.copy()
    frame["address"] = frame["address"].astype(str).str.strip().str.lower()
    frame = frame[frame["address"] != ""].drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
    default_label = pd.Series(np.zeros(len(frame), dtype=np.int64), index=frame.index)
    default_label_name = pd.Series(["non_ponzi"] * len(frame), index=frame.index)
    default_source_group = pd.Series(["external_negative"] * len(frame), index=frame.index)
    frame["label"] = pd.to_numeric(frame["label"] if "label" in frame.columns else default_label, errors="coerce").fillna(0).astype(np.int64)
    frame["label_name"] = frame["label_name"] if "label_name" in frame.columns else default_label_name
    frame["source_group"] = frame["source_group"] if "source_group" in frame.columns else default_source_group
    frame["is_anchor"] = 1
    frame["sequence_available"] = 0
    return frame


def _build_feature_matrix(users: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns: list[str] = []
    for column in users.columns:
        if column in FEATURE_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(users[column]):
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError("Ethereum Ponzi combined table did not yield any numeric feature columns.")
    frame = users[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    values = frame.to_numpy(dtype=np.float32)
    values = np.sign(values) * np.log1p(np.abs(values))
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    return ((values - means) / stds).astype(np.float32), feature_columns


def _numeric_column(users: pd.DataFrame, column: str, *, default: float = 0.0) -> np.ndarray:
    if column not in users.columns:
        return np.full(len(users), float(default), dtype=np.float32)
    return pd.to_numeric(users[column], errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _window_edges_from_signal(signal: np.ndarray, neighbors: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if len(signal) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    safe_signal = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(safe_signal, kind="mergesort")
    src, dst = _window_edge_pairs(order.astype(np.int64), neighbors=neighbors)
    if len(src) == 0:
        singleton = np.arange(len(safe_signal), dtype=np.int64)
        return singleton.copy(), singleton.copy()
    return src, dst


def _group_window_edges(
    group_values: pd.Series,
    order_signal: np.ndarray,
    neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    src_all: list[np.ndarray] = []
    dst_all: list[np.ndarray] = []
    normalized_values = group_values.fillna("unknown").astype(str).str.strip().str.lower()
    for value in sorted(normalized_values.unique()):
        if not value or value == "unknown":
            continue
        group_index = np.flatnonzero(normalized_values.to_numpy() == value)
        if len(group_index) < 2:
            continue
        group_order = group_index[np.argsort(order_signal[group_index], kind="mergesort")]
        src, dst = _window_edge_pairs(group_order.astype(np.int64), neighbors=neighbors)
        if len(src) > 0:
            src_all.append(src)
            dst_all.append(dst)
    if not src_all:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_all), np.concatenate(dst_all)


def _append_relation_edges(
    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]],
    relation_edge_counts: dict[str, int],
    relation_name: str,
    src: np.ndarray,
    dst: np.ndarray,
) -> None:
    if len(src) == 0:
        return
    edge_dict[(NODE_TYPE, relation_name, NODE_TYPE)] = (
        torch.from_numpy(src.astype(np.int64)),
        torch.from_numpy(dst.astype(np.int64)),
    )
    relation_edge_counts[relation_name] = int(len(src))


def _build_synthetic_edges(users: pd.DataFrame) -> tuple[dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]], dict[str, int]]:
    code_score = (
        _numeric_column(users, "source_char_count")
        + 5.0 * _numeric_column(users, "function_count")
        + 4.0 * _numeric_column(users, "payable_count")
        + 3.0 * _numeric_column(users, "transfer_call_count")
        + 3.0 * _numeric_column(users, "value_call_count")
    )
    src, dst = _window_edges_from_signal(code_score, neighbors=4)
    edge_dict = {
        (NODE_TYPE, "homo", NODE_TYPE): (torch.from_numpy(src), torch.from_numpy(dst)),
        (NODE_TYPE, "code_peer", NODE_TYPE): (torch.from_numpy(src.copy()), torch.from_numpy(dst.copy())),
    }
    relation_edge_counts = {"homo": int(len(src)), "code_peer": int(len(src))}

    source_group_src, source_group_dst = _group_window_edges(users.get("source_group", pd.Series([], dtype=str)), code_score)
    _append_relation_edges(edge_dict, relation_edge_counts, "source_group_peer", source_group_src, source_group_dst)

    behavior_score = (
        4.0 * _numeric_column(users, "payable_count")
        + 3.0 * _numeric_column(users, "transfer_call_count")
        + 2.0 * _numeric_column(users, "send_call_count")
        + 3.0 * _numeric_column(users, "value_call_count")
        + 1.5 * _numeric_column(users, "require_count")
        + 1.0 * _numeric_column(users, "if_count")
    )
    behavior_src, behavior_dst = _window_edges_from_signal(behavior_score, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "behavior_peer", behavior_src, behavior_dst)
    return edge_dict, relation_edge_counts


def _build_code_event_tensors(
    users: pd.DataFrame,
    history_len: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(len(users))
    sequence_length = max(int(history_len), 1)
    event_dim = 8
    event_sequence = np.zeros((num_nodes, sequence_length, event_dim), dtype=np.float32)
    event_mask = np.zeros((num_nodes, sequence_length), dtype=bool)
    event_time_deltas = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    event_token_weights = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    event_token_types = np.zeros((num_nodes, sequence_length), dtype=np.int64)
    if num_nodes == 0:
        return (
            torch.from_numpy(event_sequence),
            torch.from_numpy(event_mask),
            torch.from_numpy(event_time_deltas),
            torch.from_numpy(event_token_weights),
            torch.from_numpy(event_token_types),
        )

    source_char_count = np.log1p(np.abs(_numeric_column(users, "source_char_count")))
    source_line_count = np.log1p(np.abs(_numeric_column(users, "source_line_count")))
    comment_line_count = np.log1p(np.abs(_numeric_column(users, "comment_line_count")))
    function_count = np.log1p(np.abs(_numeric_column(users, "function_count")))
    modifier_count = np.log1p(np.abs(_numeric_column(users, "modifier_count")))
    mapping_count = np.log1p(np.abs(_numeric_column(users, "mapping_count")))
    struct_count = np.log1p(np.abs(_numeric_column(users, "struct_count")))
    contract_count = np.log1p(np.abs(_numeric_column(users, "contract_count")))
    library_count = np.log1p(np.abs(_numeric_column(users, "library_count")))
    interface_count = np.log1p(np.abs(_numeric_column(users, "interface_count")))
    require_count = np.log1p(np.abs(_numeric_column(users, "require_count")))
    if_count = np.log1p(np.abs(_numeric_column(users, "if_count")))
    for_count = np.log1p(np.abs(_numeric_column(users, "for_count")))
    while_count = np.log1p(np.abs(_numeric_column(users, "while_count")))
    event_count = np.log1p(np.abs(_numeric_column(users, "event_count")))
    payable_count = np.log1p(np.abs(_numeric_column(users, "payable_count")))
    transfer_call_count = np.log1p(np.abs(_numeric_column(users, "transfer_call_count")))
    send_call_count = np.log1p(np.abs(_numeric_column(users, "send_call_count")))
    value_call_count = np.log1p(np.abs(_numeric_column(users, "value_call_count")))
    has_source_code = np.clip(_numeric_column(users, "has_source_code"), 0.0, 1.0)
    source_group = users.get("source_group", pd.Series(["unknown"] * num_nodes)).fillna("unknown").astype(str).str.lower()
    is_curated = source_group.str.contains("curated", regex=False).to_numpy(dtype=np.float32)
    is_recent = source_group.str.contains("recent", regex=False).to_numpy(dtype=np.float32)
    is_zero_day = source_group.str.contains("zero_day", regex=False).to_numpy(dtype=np.float32)
    is_external_negative = source_group.str.contains("negative", regex=False).to_numpy(dtype=np.float32)
    is_anchor = np.clip(_numeric_column(users, "is_anchor"), 0.0, 1.0)

    token_vectors = []
    token_masks = []
    token_types = [0, 1, 1, 2, 3]

    token_0 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_0[:, 0] = source_char_count
    token_0[:, 1] = source_line_count
    token_0[:, 2] = comment_line_count
    token_0[:, 3] = has_source_code
    token_vectors.append(token_0)
    token_masks.append((source_char_count > 0.0) | (source_line_count > 0.0) | (comment_line_count > 0.0) | (has_source_code > 0.0))

    token_1 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_1[:, 0] = function_count
    token_1[:, 1] = modifier_count
    token_1[:, 2] = mapping_count
    token_1[:, 3] = struct_count
    token_1[:, 4] = contract_count
    token_1[:, 5] = library_count
    token_1[:, 6] = interface_count
    token_vectors.append(token_1)
    token_masks.append(
        (function_count > 0.0)
        | (modifier_count > 0.0)
        | (mapping_count > 0.0)
        | (struct_count > 0.0)
        | (contract_count > 0.0)
        | (library_count > 0.0)
        | (interface_count > 0.0)
    )

    token_2 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_2[:, 0] = require_count
    token_2[:, 1] = if_count
    token_2[:, 2] = for_count
    token_2[:, 3] = while_count
    token_2[:, 4] = event_count
    token_vectors.append(token_2)
    token_masks.append((require_count > 0.0) | (if_count > 0.0) | (for_count > 0.0) | (while_count > 0.0) | (event_count > 0.0))

    token_3 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_3[:, 0] = payable_count
    token_3[:, 1] = transfer_call_count
    token_3[:, 2] = send_call_count
    token_3[:, 3] = value_call_count
    token_3[:, 4] = has_source_code
    token_vectors.append(token_3)
    token_masks.append(
        (payable_count > 0.0)
        | (transfer_call_count > 0.0)
        | (send_call_count > 0.0)
        | (value_call_count > 0.0)
        | (has_source_code > 0.0)
    )

    token_4 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_4[:, 0] = is_curated
    token_4[:, 1] = is_recent
    token_4[:, 2] = is_zero_day
    token_4[:, 3] = is_external_negative
    token_4[:, 4] = is_anchor
    token_vectors.append(token_4)
    token_masks.append(np.ones(num_nodes, dtype=bool))

    max_tokens = min(sequence_length, len(token_vectors))
    for token_index in range(max_tokens):
        event_sequence[:, token_index, :] = token_vectors[token_index]
        event_mask[:, token_index] = token_masks[token_index]
        event_time_deltas[:, token_index] = float(max_tokens - 1 - token_index)
        event_token_types[:, token_index] = int(token_types[token_index])

    token_strength = np.log1p(np.abs(event_sequence).sum(axis=-1))
    event_token_weights = np.where(event_mask, 1.0 + np.clip(token_strength, 0.0, 3.0), 0.0).astype(np.float32)
    return (
        torch.from_numpy(event_sequence),
        torch.from_numpy(event_mask),
        torch.from_numpy(event_time_deltas),
        torch.from_numpy(event_token_weights),
        torch.from_numpy(event_token_types),
    )


def load_ethereum_ponzi_dataset(
    *,
    data_root: str | Path = ETHEREUM_PONZI_DEFAULT_ROOT,
    dataset_name: str = "ethereum_ponzi",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    negative_users_path: str | Path | None = None,
    force_preview: bool = False,
) -> DatasetBundle:
    resolved_root = Path(data_root).expanduser().resolve()
    positive_users, users_source = _read_table(
        resolved_root / "dataset" / "data" / "users.parquet",
        resolved_root / "dataset" / "preview" / "users.csv",
        force_preview=force_preview,
    )

    positive_users = positive_users.copy()
    positive_users["address"] = positive_users["address"].astype(str).str.strip().str.lower()
    positive_users["label"] = 1
    positive_users["label_name"] = "ponzi"
    positive_users["is_anchor"] = 1
    positive_users["sequence_available"] = pd.to_numeric(
        positive_users.get("has_source_code", 0),
        errors="coerce",
    ).fillna(0).astype(np.int64)

    if negative_users_path is None:
        raise ValueError(
            "Ethereum Ponzi loader is draft-only right now: the local public dataset is positive-only. "
            "Provide `negative_users_path=...` with an external non-Ponzi address/contract set to build a binary benchmark."
        )

    negative_users = _load_negative_users(negative_users_path)
    users = pd.concat([positive_users, negative_users], ignore_index=True, sort=False)
    users = users.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)

    if users["label"].nunique() < 2:
        raise ValueError(
            "Ethereum Ponzi loader still has only one class after merging negatives. "
            "Please provide a valid negative set with label 0."
        )

    numeric_columns = [column for column in users.columns if pd.api.types.is_numeric_dtype(users[column])]
    users[numeric_columns] = users[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_matrix, feature_columns = _build_feature_matrix(users)

    train_mask, valid_mask, test_mask = _stratified_split_masks(users["label"].to_numpy(dtype=np.int64), seed=seed)
    edge_dict, relation_edge_counts = _build_synthetic_edges(users)
    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(users)})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix)
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(users["label"].to_numpy(dtype=np.int64, copy=True))
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    graph.nodes[NODE_TYPE].data["label_confidence_target"] = torch.ones(graph.num_nodes(NODE_TYPE), dtype=torch.float32)
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)

    if float(label_fraction) < 0.999:
        _apply_label_scarcity(graph, label_fraction=float(label_fraction), seed=seed)
    else:
        graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.clone()
        graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask)
        graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full(
            (graph.num_nodes(NODE_TYPE),),
            1.0,
            dtype=torch.float32,
        )
    if active_learning_feedback_path:
        _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)

    homo_src, homo_dst = graph.edges(etype="homo")
    labels = graph.nodes[NODE_TYPE].data["label"].long()
    supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()
    graph.edges["homo"].data["label"] = torch.where(
        labels[homo_src] == labels[homo_dst],
        torch.ones_like(homo_src, dtype=torch.float32),
        -torch.ones_like(homo_src, dtype=torch.float32),
    )
    graph.edges["homo"].data["train_mask"] = supervised_mask[homo_src] & supervised_mask[homo_dst]

    event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types = _build_code_event_tensors(
        users=users,
        history_len=PONZI_EVENT_SEQUENCE_LENGTH,
    )
    graph.nodes[NODE_TYPE].data["event_sequence"] = event_sequence.float()
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()
    relation_order = _attach_relation_sequence(graph, dataset_name="ethereum_ponzi")

    train_supervised_mask = (
        graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool() & graph.nodes[NODE_TYPE].data["train_mask"].bool()
    )
    train_unlabeled_mask = (
        graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool() & graph.nodes[NODE_TYPE].data["train_mask"].bool()
    )
    supervised_nodes = train_supervised_mask.nonzero(as_tuple=False).flatten()
    supervised_labels = graph.nodes[NODE_TYPE].data["label"][train_supervised_mask]
    unlabeled_nodes = train_unlabeled_mask.nonzero(as_tuple=False).flatten()
    supervised_partitions = _stratified_partition(
        supervised_nodes,
        supervised_labels,
        num_clients=max(int(num_clients), 1),
        seed=seed,
    )
    unlabeled_partitions = _random_partition(unlabeled_nodes, num_clients=max(int(num_clients), 1), seed=seed + 1)
    owned_partitions = _merge_partitions(supervised_partitions, unlabeled_partitions)

    clients: list[ClientShard] = []
    for client_id, owned_nodes in enumerate(owned_partitions):
        if len(owned_nodes) == 0:
            continue
        subgraph = _build_client_subgraph(graph, NODE_TYPE, owned_nodes, hops=client_hops)
        clients.append(
            ClientShard(
                client_id=client_id,
                owned_global_nodes=owned_nodes,
                subgraph=subgraph,
                train_nodes=int(subgraph.nodes[NODE_TYPE].data["train_mask"].sum().item()),
            )
        )

    class_labels = graph.nodes[NODE_TYPE].data["label"][graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()]
    bundle = DatasetBundle(
        name=dataset_name,
        graph=graph,
        node_type=NODE_TYPE,
        relation_order=relation_order,
        class_weights=_compute_class_weights(class_labels) if class_labels.numel() > 0 else torch.ones(2, dtype=torch.float32),
        class_counts=_compute_class_counts(class_labels) if class_labels.numel() > 0 else torch.ones(2, dtype=torch.float32),
        clients=clients,
        base_lr=1e-3,
    )
    bundle.addresses = users["address"].tolist()
    bundle.data_summary = {
        "data_root": str(resolved_root),
        "users_source": users_source,
        "negative_users_path": str(Path(negative_users_path).expanduser().resolve()),
        "feature_columns": feature_columns,
        "feature_dim": int(feature_matrix.shape[1]),
        "num_nodes": int(graph.num_nodes(NODE_TYPE)),
        "num_clients": int(len(clients)),
        "relation_edge_counts": relation_edge_counts,
        "graph_source": "synthetic_code_semantic_graph",
        "train_nodes": int(train_mask.sum().item()),
        "valid_nodes": int(valid_mask.sum().item()),
        "test_nodes": int(test_mask.sum().item()),
        "positive_labels": int(users["label"].sum()),
        "positive_ratio": float(users["label"].mean()),
        "event_sequence_length": int(event_sequence.shape[1]),
        "event_sequence_strategy": "code_metric_groups",
        "draft_loader": True,
        "force_preview": bool(force_preview),
    }
    return bundle
