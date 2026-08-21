from __future__ import annotations

"""Draft loader for the cleaned Ethereum Phishing Transaction Network dataset."""

from pathlib import Path
from typing import Any
import re

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
    ARCHIVE_EVENT_SEQUENCE_LENGTH,
    _build_archive_event_tensors,
    _build_transaction_aggregates,
    _compute_class_counts,
    _compute_class_weights,
    _prepare_transactions_frame,
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

ETHEREUM_PHISHING_DEFAULT_ROOT = DATA_ROOT / "ethereum_phishing"
NODE_TYPE = "account"
BALANCE_SEQUENCE_LENGTH = 50
FEATURE_EXCLUDE_COLUMNS = {
    "address",
    "label",
    "label_name",
    "is_anchor",
    "sequence_available",
    "first_seen_dt",
    "last_seen_dt",
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


def _build_balance_sequence(users: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    balance_columns = sorted(
        [column for column in users.columns if re.fullmatch(r"balance_\d+", column)],
        key=lambda column: int(column.split("_")[-1]),
    )
    if not balance_columns:
        sequence = torch.zeros((len(users), BALANCE_SEQUENCE_LENGTH, 1), dtype=torch.float32)
        mask = torch.zeros((len(users), BALANCE_SEQUENCE_LENGTH), dtype=torch.bool)
        deltas = torch.zeros((len(users), BALANCE_SEQUENCE_LENGTH), dtype=torch.float32)
        return sequence, mask, deltas

    values = users[balance_columns].fillna(0.0).to_numpy(dtype=np.float32)
    sequence = torch.from_numpy(values).unsqueeze(-1)
    has_sequence = users.get("sequence_available", pd.Series(np.ones(len(users), dtype=np.int64))).to_numpy(dtype=bool)
    mask = torch.from_numpy(np.repeat(has_sequence[:, None], values.shape[1], axis=1))
    reverse_steps = np.arange(values.shape[1] - 1, -1, -1, dtype=np.float32)
    deltas = torch.from_numpy(np.repeat(reverse_steps[None, :], len(users), axis=0))
    return sequence.float(), mask.bool(), deltas.float()


def _numeric_column(users: pd.DataFrame, column: str, *, default: float = 0.0) -> np.ndarray:
    if column not in users.columns:
        return np.full(len(users), float(default), dtype=np.float32)
    return pd.to_numeric(users[column], errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _window_edges_from_signal(signal: np.ndarray, neighbors: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if len(signal) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    safe_signal = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(safe_signal, kind="mergesort")
    src, dst = _window_edge_pairs(order.astype(np.int64), neighbors=neighbors)
    if len(src) == 0:
        singleton = np.arange(len(safe_signal), dtype=np.int64)
        return singleton.copy(), singleton.copy()
    return src, dst


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


def _add_context_relations(
    users: pd.DataFrame,
    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]],
    relation_edge_counts: dict[str, int],
) -> None:
    balance_signal = (
        np.log1p(np.abs(_numeric_column(users, "balance_absmax")))
        + np.log1p(np.abs(_numeric_column(users, "balance_std")))
        + np.log1p(np.abs(_numeric_column(users, "balance_nonzero_steps")))
    )
    balance_src, balance_dst = _window_edges_from_signal(balance_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "balance_profile", balance_src, balance_dst)

    anchor_signal = (
        4.0 * _numeric_column(users, "direct_anchor_hits")
        + 3.0 * _numeric_column(users, "direct_anchor_counterparties")
        + 2.0 * _numeric_column(users, "anchor_ego_hits")
        + np.log1p(np.abs(_numeric_column(users, "raw_graph_degree")))
    )
    anchor_src, anchor_dst = _window_edges_from_signal(anchor_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "anchor_context", anchor_src, anchor_dst)


def _augment_event_tensors_with_balance_context(
    users: pd.DataFrame,
    event_sequence: torch.Tensor,
    event_mask: torch.Tensor,
    event_time_deltas: torch.Tensor,
    event_token_weights: torch.Tensor,
    event_token_types: torch.Tensor,
    balance_sequence: torch.Tensor,
    balance_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if event_sequence.ndim != 3 or balance_sequence.ndim != 3:
        return event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types

    num_nodes = int(event_sequence.shape[0])
    event_dim = int(event_sequence.shape[-1])
    extra_len = 3
    if num_nodes == 0 or event_dim <= 0:
        return event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types

    balance_first = _numeric_column(users, "balance_first")
    balance_last = _numeric_column(users, "balance_last")
    balance_delta = _numeric_column(users, "balance_delta")
    balance_absmax = np.abs(_numeric_column(users, "balance_absmax"))
    balance_std = np.abs(_numeric_column(users, "balance_std"))
    balance_diff_std = np.abs(_numeric_column(users, "balance_diff_std"))
    balance_nonzero_steps = np.abs(_numeric_column(users, "balance_nonzero_steps"))
    direct_anchor_hits = np.abs(_numeric_column(users, "direct_anchor_hits"))
    direct_anchor_counterparties = np.abs(_numeric_column(users, "direct_anchor_counterparties"))
    anchor_ego_hits = np.abs(_numeric_column(users, "anchor_ego_hits"))
    sequence_available = _numeric_column(users, "sequence_available")

    extra_sequence = torch.zeros((num_nodes, extra_len, event_dim), dtype=torch.float32)
    extra_mask = torch.zeros((num_nodes, extra_len), dtype=torch.bool)
    extra_time_deltas = torch.zeros((num_nodes, extra_len), dtype=torch.float32)
    extra_token_weights = torch.zeros((num_nodes, extra_len), dtype=torch.float32)
    extra_token_types = torch.full((num_nodes, extra_len), 3, dtype=torch.long)

    valid_balance = (
        torch.from_numpy((sequence_available > 0).astype(np.bool_))
        | balance_mask.any(dim=1)
        | torch.from_numpy((balance_absmax > 0).astype(np.bool_))
    )
    if not valid_balance.any():
        return event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types

    extra_sequence[:, 0, 0] = torch.from_numpy(np.log1p(np.abs(balance_first))).float()
    extra_sequence[:, 0, 1] = torch.from_numpy(np.log1p(np.abs(balance_last))).float()
    extra_sequence[:, 0, 2] = torch.from_numpy(balance_std).float()
    extra_sequence[:, 0, 3] = torch.from_numpy((balance_delta > 0).astype(np.float32))
    extra_sequence[:, 0, 4] = torch.from_numpy((balance_delta < 0).astype(np.float32))

    extra_sequence[:, 1, 0] = torch.from_numpy(np.log1p(balance_absmax)).float()
    extra_sequence[:, 1, 1] = torch.from_numpy(balance_std).float()
    extra_sequence[:, 1, 2] = torch.from_numpy(balance_diff_std).float()
    extra_sequence[:, 1, 3] = torch.from_numpy(
        np.clip(balance_nonzero_steps / max(balance_sequence.shape[1], 1), 0.0, 1.0)
    ).float()
    extra_sequence[:, 1, 4] = torch.from_numpy(sequence_available.clip(0.0, 1.0)).float()

    extra_sequence[:, 2, 0] = torch.from_numpy(np.log1p(np.abs(balance_delta))).float()
    extra_sequence[:, 2, 1] = torch.from_numpy(np.log1p(direct_anchor_hits)).float()
    extra_sequence[:, 2, 2] = torch.from_numpy(np.log1p(direct_anchor_counterparties)).float()
    extra_sequence[:, 2, 3] = torch.from_numpy(np.log1p(anchor_ego_hits)).float()
    extra_sequence[:, 2, 4] = torch.from_numpy(sequence_available.clip(0.0, 1.0)).float()

    extra_mask[:] = valid_balance.unsqueeze(1)
    extra_time_deltas[:, 0] = 2.0
    extra_time_deltas[:, 1] = 1.0
    extra_time_deltas[:, 2] = 0.0
    extra_token_weights[:, 0] = torch.from_numpy(
        1.0 + np.clip(balance_nonzero_steps / max(balance_sequence.shape[1], 1), 0.0, 1.0)
    ).float()
    extra_token_weights[:, 1] = torch.from_numpy(1.0 + np.clip(balance_std, 0.0, 1.0)).float()
    extra_token_weights[:, 2] = torch.from_numpy(
        1.0 + np.clip(np.log1p(direct_anchor_hits + direct_anchor_counterparties), 0.0, 2.0)
    ).float()
    extra_token_weights = extra_token_weights * extra_mask.to(dtype=torch.float32)

    return (
        torch.cat([event_sequence.float(), extra_sequence], dim=1),
        torch.cat([event_mask.bool(), extra_mask], dim=1),
        torch.cat([event_time_deltas.float(), extra_time_deltas], dim=1),
        torch.cat([event_token_weights.float(), extra_token_weights], dim=1),
        torch.cat([event_token_types.long(), extra_token_types], dim=1),
    )


def _build_temporal_user_features(transactions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "address",
        "sent_count",
        "received_count",
        "transaction_count",
        "total_sent_eth",
        "total_received_eth",
        "net_flow_balance",
        "sent_counterparties",
        "received_counterparties",
        "total_counterparties",
        "first_seen_ts",
        "last_seen_ts",
        "active_span_hours",
        "tx_mean_value",
        "tx_max_value",
        "tx_min_value",
    ]
    if transactions.empty:
        return pd.DataFrame(columns=columns)

    frame = transactions.copy()
    frame["timestamp_dt"] = pd.to_datetime(frame["timestamp_dt"], errors="coerce")
    frame["value_eth"] = pd.to_numeric(frame["value_eth"], errors="coerce").fillna(0.0)
    outbound = frame.groupby("src_address").agg(
        sent_count=("src_address", "size"),
        total_sent_eth=("value_eth", "sum"),
        sent_counterparties=("dst_address", "nunique"),
        first_sent_dt=("timestamp_dt", "min"),
        last_sent_dt=("timestamp_dt", "max"),
        tx_out_max_value=("value_eth", "max"),
        tx_out_min_value=("value_eth", "min"),
    )
    inbound = frame.groupby("dst_address").agg(
        received_count=("dst_address", "size"),
        total_received_eth=("value_eth", "sum"),
        received_counterparties=("src_address", "nunique"),
        first_received_dt=("timestamp_dt", "min"),
        last_received_dt=("timestamp_dt", "max"),
        tx_in_max_value=("value_eth", "max"),
        tx_in_min_value=("value_eth", "min"),
    )
    users = outbound.join(inbound, how="outer").fillna(0.0)

    def _combine_time(left: pd.Series, right: pd.Series, mode: str) -> pd.Series:
        left_dt = pd.to_datetime(left, errors="coerce")
        right_dt = pd.to_datetime(right, errors="coerce")
        combined = pd.DataFrame({"left": left_dt, "right": right_dt})
        return combined.min(axis=1) if mode == "min" else combined.max(axis=1)

    users["first_seen_dt"] = _combine_time(users["first_sent_dt"], users["first_received_dt"], "min")
    users["last_seen_dt"] = _combine_time(users["last_sent_dt"], users["last_received_dt"], "max")
    valid_first = users["first_seen_dt"].notna()
    valid_last = users["last_seen_dt"].notna()
    users["first_seen_ts"] = np.where(
        valid_first,
        users["first_seen_dt"].astype("int64", copy=False) / 1_000_000_000,
        0.0,
    )
    users["last_seen_ts"] = np.where(
        valid_last,
        users["last_seen_dt"].astype("int64", copy=False) / 1_000_000_000,
        0.0,
    )
    users["active_span_hours"] = np.maximum(users["last_seen_ts"] - users["first_seen_ts"], 0.0) / 3600.0
    users["transaction_count"] = users["sent_count"] + users["received_count"]
    users["net_flow_balance"] = users["total_received_eth"] - users["total_sent_eth"]
    users["total_counterparties"] = users["sent_counterparties"] + users["received_counterparties"]
    users["tx_mean_value"] = (users["total_sent_eth"] + users["total_received_eth"]) / np.maximum(
        users["transaction_count"],
        1.0,
    )
    users["tx_max_value"] = users[["tx_out_max_value", "tx_in_max_value"]].max(axis=1)
    users["tx_min_value"] = users[["tx_out_min_value", "tx_in_min_value"]].replace(0.0, np.nan).min(axis=1).fillna(0.0)
    users = users.reset_index().rename(columns={"index": "address", "src_address": "address"})
    return users[columns]


def _build_feature_matrix(users: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns: list[str] = []
    for column in users.columns:
        if column in FEATURE_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(users[column]):
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError("Ethereum phishing users table did not yield any numeric feature columns.")

    feature_frame = users[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    values = feature_frame.to_numpy(dtype=np.float32)
    values = np.sign(values) * np.log1p(np.abs(values))
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    normalized = (values - means) / stds
    return normalized.astype(np.float32), feature_columns


def _select_users(users: pd.DataFrame, max_users: int | None, seed: int) -> pd.DataFrame:
    if max_users is None or len(users) <= int(max_users):
        return users.reset_index(drop=True)

    selected = users.copy()
    selected["is_anchor"] = selected["is_anchor"].fillna(0).astype(int)
    anchors = selected[selected["is_anchor"] == 1].copy()
    contexts = selected[selected["is_anchor"] == 0].copy()
    if len(anchors) >= int(max_users):
        return anchors.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    contexts["selection_score"] = (
        10.0 * contexts.get("direct_anchor_counterparties", 0.0).astype(np.float64)
        + 6.0 * contexts.get("direct_anchor_hits", 0.0).astype(np.float64)
        + 4.0 * contexts.get("anchor_ego_hits", 0.0).astype(np.float64)
        + 2.0 * contexts.get("tx_total_count", 0.0).astype(np.float64)
        + contexts.get("raw_graph_degree", 0.0).astype(np.float64)
        + contexts.get("transaction_count", 0.0).astype(np.float64)
    )
    contexts["tie_breaker"] = rng.random(len(contexts))
    contexts = contexts.sort_values(
        ["selection_score", "tie_breaker", "address"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    keep_context = max(int(max_users) - len(anchors), 0)
    kept = pd.concat([anchors, contexts.head(keep_context)], ignore_index=True)
    return kept.drop(columns=[column for column in ("selection_score", "tie_breaker") if column in kept.columns]).reset_index(
        drop=True
    )


def _build_global_masks(raw_labels: np.ndarray, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labeled_mask = raw_labels >= 0
    labeled_indices = np.flatnonzero(labeled_mask)
    train_mask = torch.zeros(len(raw_labels), dtype=torch.bool)
    valid_mask = torch.zeros(len(raw_labels), dtype=torch.bool)
    test_mask = torch.zeros(len(raw_labels), dtype=torch.bool)
    if labeled_indices.size == 0:
        return train_mask, valid_mask, test_mask
    labeled_train, labeled_valid, labeled_test = _stratified_split_masks(raw_labels[labeled_indices].astype(np.int64), seed=seed)
    train_mask[labeled_indices] = labeled_train.bool()
    valid_mask[labeled_indices] = labeled_valid.bool()
    test_mask[labeled_indices] = labeled_test.bool()
    return train_mask, valid_mask, test_mask


def _build_synthetic_edges(users: pd.DataFrame) -> tuple[dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]], dict[str, int]]:
    activity = (
        users.get("direct_anchor_hits", 0.0).to_numpy(dtype=np.float64)
        + users.get("anchor_ego_hits", 0.0).to_numpy(dtype=np.float64)
        + users.get("transaction_count", 0.0).to_numpy(dtype=np.float64)
    )
    src, dst = _window_edges_from_signal(activity, neighbors=3)
    edge_dict = {
        (NODE_TYPE, "homo", NODE_TYPE): (torch.from_numpy(src), torch.from_numpy(dst)),
        (NODE_TYPE, "transfer_out", NODE_TYPE): (torch.from_numpy(src.copy()), torch.from_numpy(dst.copy())),
        (NODE_TYPE, "transfer_in", NODE_TYPE): (torch.from_numpy(dst.copy()), torch.from_numpy(src.copy())),
    }
    relation_edge_counts = {"homo": int(len(src)), "transfer_out": int(len(src)), "transfer_in": int(len(src))}
    _add_context_relations(users, edge_dict, relation_edge_counts)
    return edge_dict, relation_edge_counts


def load_ethereum_phishing_dataset(
    *,
    data_root: str | Path = ETHEREUM_PHISHING_DEFAULT_ROOT,
    dataset_name: str = "ethereum_phishing",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    max_users: int | None = 50000,
    max_transactions: int | None = None,
    force_preview: bool = False,
) -> DatasetBundle:
    resolved_root = Path(data_root).expanduser().resolve()
    users_frame, users_source = _read_table(
        resolved_root / "dataset" / "data" / "users.parquet",
        resolved_root / "dataset" / "preview" / "users.csv",
        force_preview=force_preview,
    )
    transactions_frame, transactions_source = _read_table(
        resolved_root / "dataset" / "data" / "transactions.parquet",
        resolved_root / "dataset" / "preview" / "transactions.csv",
        force_preview=force_preview,
    )
    context_frame, context_source = _read_table(
        resolved_root / "dataset" / "data" / "address_context.parquet",
        resolved_root / "dataset" / "preview" / "address_context.csv",
        force_preview=force_preview,
    )

    if max_transactions is not None and len(transactions_frame) > int(max_transactions):
        transactions_frame = transactions_frame.iloc[: int(max_transactions)].copy()

    anchor_users = users_frame.copy()
    anchor_users["address"] = anchor_users["address"].astype(str).str.strip().str.lower()
    anchor_users["is_anchor"] = 1
    anchor_users["label"] = pd.to_numeric(anchor_users["label"], errors="coerce").fillna(0).astype(np.int64)

    transactions = _prepare_transactions_frame(transactions_frame)
    universe_addresses = sorted(
        set(anchor_users["address"].tolist())
        | set(context_frame["address"].astype(str).str.strip().str.lower().tolist())
        | set(transactions["src_address"].astype(str).str.strip().str.lower().tolist())
        | set(transactions["dst_address"].astype(str).str.strip().str.lower().tolist())
    )
    users = pd.DataFrame({"address": universe_addresses})
    users = users.merge(context_frame, on="address", how="left")
    users = users.merge(anchor_users, on="address", how="left", suffixes=("", "_anchor"))

    ranking_aggregates = _build_transaction_aggregates(transactions)
    ranking_features = _build_temporal_user_features(transactions)
    users = users.merge(ranking_aggregates, on="address", how="left")
    users = users.merge(ranking_features, on="address", how="left")
    users["is_anchor"] = users["is_anchor"].fillna(0).astype(np.int64)
    users = _select_users(users, max_users=max_users, seed=seed)

    selected_addresses = set(users["address"].tolist())
    transactions = transactions[
        transactions["src_address"].isin(selected_addresses) & transactions["dst_address"].isin(selected_addresses)
    ].copy()

    users = users.drop(
        columns=[column for column in users.columns if column.startswith("tx_") or column in {"sent_count", "received_count", "transaction_count",
        "total_sent_eth", "total_received_eth", "net_flow_balance", "sent_counterparties", "received_counterparties", "total_counterparties",
        "first_seen_ts", "last_seen_ts", "active_span_hours", "tx_mean_value", "tx_max_value", "tx_min_value"}],
        errors="ignore",
    )
    tx_aggregates = _build_transaction_aggregates(transactions)
    tx_features = _build_temporal_user_features(transactions)
    users = users.merge(tx_aggregates, on="address", how="left")
    users = users.merge(tx_features, on="address", how="left")

    users["value_out"] = users["value_out"].fillna(users["total_sent_eth"])
    users["value_in"] = users["value_in"].fillna(users["total_received_eth"])
    users["balance"] = users["balance"].fillna(users["net_flow_balance"])
    users["degree"] = users["degree"].fillna(users["transaction_count"])
    users["degree_in"] = users["degree_in"].fillna(users["received_count"])
    users["degree_out"] = users["degree_out"].fillna(users["sent_count"])
    users["max_value"] = users["max_value"].fillna(users["tx_max_value"])
    users["min_value"] = users["min_value"].fillna(users["tx_min_value"])
    users["mean_value"] = users["mean_value"].fillna(users["tx_mean_value"])
    users["median_value"] = users["median_value"].fillna(users["tx_mean_value"])
    users["std_value"] = users["std_value"].fillna(0.0)
    users["sequence_available"] = users["sequence_available"].fillna(0).astype(np.int64)

    numeric_columns = [column for column in users.columns if pd.api.types.is_numeric_dtype(users[column])]
    users[numeric_columns] = users[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    users = users.sort_values(["is_anchor", "address"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    raw_labels = np.where(users["is_anchor"].to_numpy(dtype=bool), users["label"].to_numpy(dtype=np.int64), -1)
    graph_labels = np.where(raw_labels >= 0, raw_labels, 0).astype(np.int64)
    train_mask, valid_mask, test_mask = _build_global_masks(raw_labels, seed=seed)
    feature_matrix, feature_columns = _build_feature_matrix(users)

    if transactions.empty:
        edge_dict, relation_edge_counts = _build_synthetic_edges(users)
        graph_source = "synthetic_context_graph"
    else:
        address_to_node = {address: index for index, address in enumerate(users["address"].tolist())}
        src_nodes = transactions["src_address"].map(address_to_node).to_numpy(dtype=np.int64)
        dst_nodes = transactions["dst_address"].map(address_to_node).to_numpy(dtype=np.int64)
        edge_dict = {
            (NODE_TYPE, "homo", NODE_TYPE): (
                torch.from_numpy(src_nodes.astype(np.int64)),
                torch.from_numpy(dst_nodes.astype(np.int64)),
            ),
            (NODE_TYPE, "transfer_out", NODE_TYPE): (
                torch.from_numpy(src_nodes.astype(np.int64)),
                torch.from_numpy(dst_nodes.astype(np.int64)),
            ),
            (NODE_TYPE, "transfer_in", NODE_TYPE): (
                torch.from_numpy(dst_nodes.astype(np.int64)),
                torch.from_numpy(src_nodes.astype(np.int64)),
            ),
        }
        relation_edge_counts = {
            "homo": int(len(src_nodes)),
            "transfer_out": int(len(src_nodes)),
            "transfer_in": int(len(src_nodes)),
        }
        _add_context_relations(users, edge_dict, relation_edge_counts)
        graph_source = "deduplicated_ego_transaction_graph"

    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(users)})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix)
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(graph_labels)
    graph.nodes[NODE_TYPE].data["anchor_mask"] = torch.from_numpy((raw_labels >= 0).astype(np.bool_))
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    graph.nodes[NODE_TYPE].data["label_confidence_target"] = torch.from_numpy((raw_labels >= 0).astype(np.float32))
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)

    balance_sequence, balance_mask, balance_time_deltas = _build_balance_sequence(users)
    graph.nodes[NODE_TYPE].data["balance_sequence"] = balance_sequence.float()
    graph.nodes[NODE_TYPE].data["balance_sequence_mask"] = balance_mask.bool()
    graph.nodes[NODE_TYPE].data["balance_time_deltas"] = balance_time_deltas.float()

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
    node_labels = graph.nodes[NODE_TYPE].data["label"].long()
    supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()
    graph.edges["homo"].data["label"] = torch.where(
        node_labels[homo_src] == node_labels[homo_dst],
        torch.ones_like(homo_src, dtype=torch.float32),
        -torch.ones_like(homo_src, dtype=torch.float32),
    )
    graph.edges["homo"].data["train_mask"] = supervised_mask[homo_src] & supervised_mask[homo_dst]

    event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types = _build_archive_event_tensors(
        users=users,
        transactions=transactions,
        history_len=ARCHIVE_EVENT_SEQUENCE_LENGTH,
    )
    event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types = (
        _augment_event_tensors_with_balance_context(
            users=users,
            event_sequence=event_sequence,
            event_mask=event_mask,
            event_time_deltas=event_time_deltas,
            event_token_weights=event_token_weights,
            event_token_types=event_token_types,
            balance_sequence=balance_sequence,
            balance_mask=balance_mask,
        )
    )
    graph.nodes[NODE_TYPE].data["event_sequence"] = event_sequence.float()
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()
    relation_order = _attach_relation_sequence(graph, dataset_name="ethereum_phishing")

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
        "transactions_source": transactions_source,
        "context_source": context_source,
        "feature_columns": feature_columns,
        "feature_dim": int(feature_matrix.shape[1]),
        "num_nodes": int(graph.num_nodes(NODE_TYPE)),
        "num_clients": int(len(clients)),
        "relation_edge_counts": relation_edge_counts,
        "graph_source": graph_source,
        "anchor_nodes": int((raw_labels >= 0).sum()),
        "context_nodes": int((raw_labels < 0).sum()),
        "train_nodes": int(graph.nodes[NODE_TYPE].data["train_mask"].sum().item()),
        "valid_nodes": int(graph.nodes[NODE_TYPE].data["valid_mask"].sum().item()),
        "test_nodes": int(graph.nodes[NODE_TYPE].data["test_mask"].sum().item()),
        "positive_labels": int((raw_labels == 1).sum()),
        "positive_ratio": float(np.mean(raw_labels[raw_labels >= 0] == 1)) if np.any(raw_labels >= 0) else 0.0,
        "event_sequence_length": int(event_sequence.shape[1]),
        "event_sequence_strategy": "archive_transactions_plus_balance_context",
        "balance_sequence_length": int(balance_sequence.shape[1]),
        "balance_sequence_consumed": True,
        "selection_strategy": "keep_all_anchors_plus_top_ranked_context",
        "max_users": None if max_users is None else int(max_users),
        "max_transactions": None if max_transactions is None else int(max_transactions),
        "force_preview": bool(force_preview),
    }
    return bundle
