from __future__ import annotations

"""Independent DeFi protocol dataset loader for the hybrid federated pipeline."""

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
    _attach_relation_sequence,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _stratified_partition,
)
from .paths import DATA_ROOT

ARCHIVE_DEFAULT_ROOT = DATA_ROOT / "defi_protocol_ethereum"
ARCHIVE_PUBLIC_NAME = "DeFi Protocol Data on Ethereum"
NODE_TYPE = "account"
RELATION_NAME_MAP = {
    "dex": "dex",
    "lending": "lending",
    "stablecoin": "stablecoin",
    "yieldfarming": "yield_farming",
    "yield_farming": "yield_farming",
    "nftfi": "nft_fi",
    "nft_fi": "nft_fi",
}
SUPPORTED_RELATIONS = ("dex", "lending", "stablecoin", "yield_farming", "nft_fi")
ARCHIVE_EVENT_RELATIONS = ("transfer",) + SUPPORTED_RELATIONS
ARCHIVE_EVENT_SEQUENCE_LENGTH = 10
ARCHIVE_FEATURE_LEAKAGE_GUARD_COLUMNS = frozenset(
    {
        # Direct weak-label ingredients or exact decompositions of those ingredients.
        "received_count",
        "total_received_eth",
        "sent_count",
        "total_sent_eth",
        "type_stablecoin",
        "active_span_hours",
        "first_seen_ts",
        "last_seen_ts",
        "transaction_count",
        "nested_sent_count",
        "nested_received_count",
        "nested_total_value_eth",
        "protocol_type_total",
        "tx_out_count",
        "tx_out_value_eth",
        "tx_out_counterparties",
        "tx_error_count",
        "tx_in_count",
        "tx_in_value_eth",
        "tx_in_counterparties",
        "tx_total_count",
        "tx_total_counterparties",
        "tx_error_rate",
        "tx_out_stablecoin_count",
        "tx_in_stablecoin_count",
        # Stablecoin-specific counters re-encode the same label heuristic.
        "tether_count",
        "usdc_count",
        "dai_count",
    }
)


def _maybe_import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except Exception:  # pragma: no cover - optional dependency
        return None
    return pq


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    for old, new in (
        ("(", "_"),
        (")", ""),
        ("[", "_"),
        ("]", ""),
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        (".", "_"),
    ):
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _normalize_relation_name(raw: object) -> str | None:
    text = _normalize_column_name(raw)
    if text.startswith("type_"):
        text = text[5:]
    compact = text.replace("_", "")
    if compact in RELATION_NAME_MAP:
        return RELATION_NAME_MAP[compact]
    if text in RELATION_NAME_MAP:
        return RELATION_NAME_MAP[text]
    return None


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _read_preview_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def _read_parquet_head(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    pq = _maybe_import_pyarrow_parquet()
    if pq is None:
        raise RuntimeError("pyarrow is not installed")
    parquet_file = pq.ParquetFile(path)
    frames: list[pd.DataFrame] = []
    rows_loaded = 0
    for batch in parquet_file.iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        if max_rows is not None:
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break
            if len(frame) > remaining:
                frame = frame.iloc[:remaining].copy()
        frames.append(frame)
        rows_loaded += len(frame)
        if max_rows is not None and rows_loaded >= max_rows:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_parquet_matching_addresses(
    path: Path,
    *,
    address_column: str,
    candidate_addresses: set[str],
    max_rows: int | None = None,
) -> pd.DataFrame:
    pq = _maybe_import_pyarrow_parquet()
    if pq is None:
        raise RuntimeError("pyarrow is not installed")
    if not candidate_addresses:
        return pd.DataFrame()

    normalized_target = _normalize_column_name(address_column)
    parquet_file = pq.ParquetFile(path)
    frames: list[pd.DataFrame] = []
    rows_loaded = 0

    for batch in parquet_file.iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        normalized_columns = [_normalize_column_name(column) for column in frame.columns]
        if normalized_target not in normalized_columns:
            continue
        address_index = normalized_columns.index(normalized_target)
        batch_addresses = frame.iloc[:, address_index].astype(str).str.strip().str.lower()
        matched_frame = frame[batch_addresses.isin(candidate_addresses)].copy()
        if matched_frame.empty:
            continue
        if max_rows is not None:
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break
            if len(matched_frame) > remaining:
                matched_frame = matched_frame.iloc[:remaining].copy()
        frames.append(matched_frame)
        rows_loaded += len(matched_frame)
        if max_rows is not None and rows_loaded >= max_rows:
            break

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_archive_table(
    *,
    parquet_path: Path,
    preview_path: Path,
    max_rows: int | None,
    force_preview: bool,
    table_name: str,
) -> tuple[pd.DataFrame, dict]:
    if not force_preview and parquet_path.exists():
        try:
            frame = _read_parquet_head(parquet_path, max_rows=max_rows)
            return frame, {
                "table": table_name,
                "source": "parquet",
                "path": str(parquet_path),
                "max_rows": None if max_rows is None else int(max_rows),
            }
        except Exception as error:
            print(
                f"[WARN] Failed to read archive parquet for {table_name}: {error}. "
                f"Falling back to preview CSV: {preview_path}"
            )
    if not preview_path.exists():
        raise FileNotFoundError(
            f"Archive {table_name} table is missing. Checked parquet={parquet_path} and preview={preview_path}."
        )
    frame = _read_preview_csv(preview_path, max_rows=max_rows)
    return frame, {
        "table": table_name,
        "source": "preview_csv",
        "path": str(preview_path),
        "max_rows": None if max_rows is None else int(max_rows),
    }


def _load_archive_users_for_transactions(
    *,
    parquet_path: Path,
    preview_path: Path,
    max_rows: int | None,
    force_preview: bool,
    candidate_addresses: set[str],
) -> tuple[pd.DataFrame, dict]:
    if not force_preview and parquet_path.exists() and candidate_addresses:
        try:
            frame = _read_parquet_matching_addresses(
                parquet_path,
                address_column="address",
                candidate_addresses=candidate_addresses,
                max_rows=max_rows,
            )
            if not frame.empty:
                return frame, {
                    "table": "users",
                    "source": "parquet_overlap",
                    "path": str(parquet_path),
                    "max_rows": None if max_rows is None else int(max_rows),
                    "candidate_addresses": int(len(candidate_addresses)),
                    "matched_rows": int(len(frame)),
                }
        except Exception as error:
            print(
                f"[WARN] Failed to load archive users by transaction overlap: {error}. "
                f"Falling back to standard archive table loading."
            )
    return _load_archive_table(
        parquet_path=parquet_path,
        preview_path=preview_path,
        max_rows=max_rows,
        force_preview=force_preview,
        table_name="users",
    )


def _ensure_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _prepare_users_frame(frame: pd.DataFrame) -> pd.DataFrame:
    users = frame.copy()
    users.columns = [_normalize_column_name(column) for column in users.columns]
    if "address" not in users.columns:
        raise ValueError("Archive users table must contain an 'address' column.")
    users["address"] = users["address"].astype(str).str.strip().str.lower()
    users = users[users["address"].str.match(r"^0x[a-f0-9]{40}$", na=False)].copy()
    users = users.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)

    _ensure_numeric(
        users,
        [
            "received_count",
            "total_received_eth",
            "sent_count",
            "total_sent_eth",
        ],
    )

    if "first_seen" in users.columns:
        users["first_seen_dt"] = pd.to_datetime(users["first_seen"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        users["first_seen_dt"] = pd.NaT
    if "last_seen" in users.columns:
        users["last_seen_dt"] = pd.to_datetime(users["last_seen"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        users["last_seen_dt"] = users["first_seen_dt"]

    users["active_span_hours"] = (
        (users["last_seen_dt"] - users["first_seen_dt"]).dt.total_seconds().fillna(0.0) / 3600.0
    ).clip(lower=0.0)
    users["first_seen_ts"] = (
        users["first_seen_dt"].astype("int64", copy=False).where(users["first_seen_dt"].notna(), 0) / 1_000_000_000
    ).astype(np.float64)
    users["last_seen_ts"] = (
        users["last_seen_dt"].astype("int64", copy=False).where(users["last_seen_dt"].notna(), 0) / 1_000_000_000
    ).astype(np.float64)

    for relation in SUPPORTED_RELATIONS:
        column = f"type_{relation}"
        if column not in users.columns:
            users[column] = 0.0

    if "protocol_types" in users.columns:
        protocol_type_payloads = users["protocol_types"].apply(_safe_json_loads)
        for relation in SUPPORTED_RELATIONS:
            column = f"type_{relation}"
            if users[column].abs().sum() > 0:
                continue
            users[column] = protocol_type_payloads.apply(
                lambda payload, rel=relation: float(
                    sum(
                        float(value or 0.0)
                        for key, value in (payload.items() if isinstance(payload, dict) else [])
                        if _normalize_relation_name(key) == rel
                    )
                )
            )

    transaction_count = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_sent_count = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_received_count = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_total_value_eth = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_mean_value_eth = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_mean_gas_used = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_sender_ratio = pd.Series(np.zeros(len(users), dtype=np.float64))
    nested_unique_protocols = pd.Series(np.zeros(len(users), dtype=np.float64))

    if "transactions" in users.columns:
        raw_transactions = users["transactions"]
        if pd.api.types.is_numeric_dtype(raw_transactions):
            transaction_count = pd.to_numeric(raw_transactions, errors="coerce").fillna(0.0)
        else:
            payloads = raw_transactions.apply(_safe_json_loads)

            def _summarize_user_transactions(items: Any) -> dict[str, float]:
                if not isinstance(items, list) or not items:
                    return {
                        "count": 0.0,
                        "sent_count": 0.0,
                        "received_count": 0.0,
                        "total_value_eth": 0.0,
                        "mean_value_eth": 0.0,
                        "mean_gas_used": 0.0,
                        "sender_ratio": 0.0,
                        "unique_protocols": 0.0,
                    }
                sent = 0.0
                values: list[float] = []
                gas_used_values: list[float] = []
                protocols: set[str] = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    sent += 1.0 if bool(item.get("is_sender", False)) else 0.0
                    try:
                        values.append(abs(float(item.get("value (ETH)", item.get("value_eth", 0.0)) or 0.0)))
                    except Exception:
                        values.append(0.0)
                    try:
                        gas_used_values.append(float(item.get("gas_used", 0.0) or 0.0))
                    except Exception:
                        gas_used_values.append(0.0)
                    protocol_name = str(item.get("protocol_name", "")).strip().lower()
                    if protocol_name:
                        protocols.add(protocol_name)
                count = float(len(items))
                return {
                    "count": count,
                    "sent_count": sent,
                    "received_count": count - sent,
                    "total_value_eth": float(np.sum(values)) if values else 0.0,
                    "mean_value_eth": float(np.mean(values)) if values else 0.0,
                    "mean_gas_used": float(np.mean(gas_used_values)) if gas_used_values else 0.0,
                    "sender_ratio": float(sent / max(count, 1.0)),
                    "unique_protocols": float(len(protocols)),
                }

            summaries = payloads.apply(_summarize_user_transactions)
            transaction_count = summaries.apply(lambda item: item["count"]).astype(np.float64)
            nested_sent_count = summaries.apply(lambda item: item["sent_count"]).astype(np.float64)
            nested_received_count = summaries.apply(lambda item: item["received_count"]).astype(np.float64)
            nested_total_value_eth = summaries.apply(lambda item: item["total_value_eth"]).astype(np.float64)
            nested_mean_value_eth = summaries.apply(lambda item: item["mean_value_eth"]).astype(np.float64)
            nested_mean_gas_used = summaries.apply(lambda item: item["mean_gas_used"]).astype(np.float64)
            nested_sender_ratio = summaries.apply(lambda item: item["sender_ratio"]).astype(np.float64)
            nested_unique_protocols = summaries.apply(lambda item: item["unique_protocols"]).astype(np.float64)

    users["transaction_count"] = transaction_count
    users["nested_sent_count"] = nested_sent_count
    users["nested_received_count"] = nested_received_count
    users["nested_total_value_eth"] = nested_total_value_eth
    users["nested_mean_value_eth"] = nested_mean_value_eth
    users["nested_mean_gas_used"] = nested_mean_gas_used
    users["nested_sender_ratio"] = nested_sender_ratio
    users["nested_unique_protocols"] = nested_unique_protocols
    users["protocol_type_total"] = users[[f"type_{relation}" for relation in SUPPORTED_RELATIONS]].sum(axis=1)
    return users.reset_index(drop=True)


def _prepare_transactions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    transactions = frame.copy()
    transactions.columns = [_normalize_column_name(column) for column in transactions.columns]
    transactions = transactions.rename(columns={"from": "src_address", "to": "dst_address"})
    if not {"src_address", "dst_address"}.issubset(set(transactions.columns)):
        raise ValueError("Archive transactions table must contain 'from' and 'to' columns.")

    transactions["src_address"] = transactions["src_address"].astype(str).str.strip().str.lower()
    transactions["dst_address"] = transactions["dst_address"].astype(str).str.strip().str.lower()
    transactions = transactions[
        transactions["src_address"].str.match(r"^0x[a-f0-9]{40}$", na=False)
        & transactions["dst_address"].str.match(r"^0x[a-f0-9]{40}$", na=False)
    ].copy()
    transactions = transactions[transactions["src_address"] != transactions["dst_address"]].reset_index(drop=True)

    _ensure_numeric(transactions, ["value_eth", "gas", "gas_used", "is_error"])
    if "timestamp" in transactions.columns:
        transactions["timestamp_dt"] = pd.to_datetime(
            transactions["timestamp"],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)
    else:
        transactions["timestamp_dt"] = pd.NaT

    metadata_payloads = transactions["metadata"].apply(_safe_json_loads) if "metadata" in transactions.columns else None
    if metadata_payloads is not None:
        transactions["relation_type"] = metadata_payloads.apply(
            lambda payload: _normalize_relation_name(
                payload.get("type", payload.get("protocol_type", "")) if isinstance(payload, dict) else ""
            )
        )
        transactions["protocol_name"] = metadata_payloads.apply(
            lambda payload: str(payload.get("protocol_name", "")).strip().lower() if isinstance(payload, dict) else ""
        )
    else:
        transactions["relation_type"] = None
        transactions["protocol_name"] = ""
    transactions["relation_type"] = transactions["relation_type"].fillna("transfer")
    return transactions.reset_index(drop=True)


def _build_archive_event_tensors(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    history_len: int = ARCHIVE_EVENT_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(len(users))
    sequence_length = max(int(history_len), 1)
    event_dim = 5 + len(ARCHIVE_EVENT_RELATIONS)
    event_sequence = np.zeros((num_nodes, sequence_length, event_dim), dtype=np.float32)
    event_mask = np.zeros((num_nodes, sequence_length), dtype=bool)
    event_time_deltas = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    event_token_weights = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    event_token_types = np.zeros((num_nodes, sequence_length), dtype=np.int64)
    if transactions.empty or num_nodes == 0:
        return (
            torch.from_numpy(event_sequence),
            torch.from_numpy(event_mask),
            torch.from_numpy(event_time_deltas),
            torch.from_numpy(event_token_weights),
            torch.from_numpy(event_token_types),
        )

    relation_to_index = {relation: index for index, relation in enumerate(ARCHIVE_EVENT_RELATIONS)}
    address_to_node = {address: index for index, address in enumerate(users["address"].tolist())}
    transaction_frame = transactions.copy()
    if "timestamp_dt" in transaction_frame.columns:
        transaction_frame["timestamp_dt"] = pd.to_datetime(transaction_frame["timestamp_dt"], errors="coerce")
        transaction_frame["timestamp_ts"] = (
            transaction_frame["timestamp_dt"].astype("int64", copy=False).where(transaction_frame["timestamp_dt"].notna(), 0)
            / 1_000_000_000
        ).astype(np.float64)
        transaction_frame = transaction_frame.sort_values(
            ["timestamp_dt", "src_address", "dst_address"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    else:
        transaction_frame["timestamp_ts"] = np.nan

    histories: list[list[tuple[float, np.ndarray, int]]] = [[] for _ in range(num_nodes)]
    for row in transaction_frame.itertuples(index=False):
        relation_name = str(getattr(row, "relation_type", "transfer") or "transfer").strip().lower()
        relation_index = relation_to_index.get(relation_name, 0)
        base_vector = np.zeros(event_dim, dtype=np.float32)
        base_vector[0] = float(np.log1p(abs(float(getattr(row, "value_eth", 0.0) or 0.0))))
        base_vector[1] = float(np.log1p(max(float(getattr(row, "gas_used", 0.0) or 0.0), 0.0)))
        base_vector[2] = float(getattr(row, "is_error", 0.0) or 0.0)
        base_vector[5 + relation_index] = 1.0
        timestamp_value = float(getattr(row, "timestamp_ts", np.nan))

        src_index = address_to_node.get(str(getattr(row, "src_address", "")).strip().lower())
        if src_index is not None:
            src_vector = base_vector.copy()
            src_vector[3] = 1.0
            histories[src_index].append((timestamp_value, src_vector, 1))

        dst_index = address_to_node.get(str(getattr(row, "dst_address", "")).strip().lower())
        if dst_index is not None:
            dst_vector = base_vector.copy()
            dst_vector[4] = 1.0
            histories[dst_index].append((timestamp_value, dst_vector, 2))

    for node_index, history in enumerate(histories):
        if not history:
            continue
        trimmed_history = history[-sequence_length:]
        insert_start = sequence_length - len(trimmed_history)
        valid_times = [timestamp for timestamp, _, _ in trimmed_history if np.isfinite(timestamp)]
        reference_time = max(valid_times) if valid_times else np.nan
        for offset, (timestamp, vector, token_type) in enumerate(trimmed_history):
            position = insert_start + offset
            event_sequence[node_index, position] = vector
            event_mask[node_index, position] = True
            if np.isfinite(reference_time) and np.isfinite(timestamp):
                delta_value = max(reference_time - timestamp, 0.0) / 3600.0
                log_delta = float(np.log1p(delta_value))
            else:
                log_delta = float(max(len(trimmed_history) - 1 - offset, 0))
            event_time_deltas[node_index, position] = log_delta
            event_token_weights[node_index, position] = 1.0 / (1.0 + log_delta)
            event_token_types[node_index, position] = int(token_type)

    return (
        torch.from_numpy(event_sequence),
        torch.from_numpy(event_mask),
        torch.from_numpy(event_time_deltas),
        torch.from_numpy(event_token_weights),
        torch.from_numpy(event_token_types),
    )


def _build_archive_label_confidence(risk_scores: np.ndarray, risk_threshold: float) -> np.ndarray:
    scores = np.asarray(risk_scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    margin = np.abs(scores - float(risk_threshold))
    normalizer = float(np.percentile(margin, 90)) if margin.size >= 2 else float(margin.max())
    normalizer = max(normalizer, 1e-6)
    normalized = np.clip(margin / normalizer, 0.0, 1.0)
    return (0.15 + 0.85 * normalized).astype(np.float32)


def _select_user_sample(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    max_users: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_users is None or len(users) <= int(max_users):
        return users.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    degree_scores = pd.Series(dtype=np.float64)
    if not transactions.empty:
        degree_scores = pd.concat(
            [transactions["src_address"], transactions["dst_address"]],
            axis=0,
            ignore_index=True,
        ).value_counts()
    selection = users.copy()
    selection["graph_degree_hint"] = selection["address"].map(degree_scores).fillna(0.0).astype(np.float64)
    selection["activity_hint"] = (
        selection["graph_degree_hint"] * 3.0
        + selection["transaction_count"].astype(np.float64)
        + selection["received_count"].astype(np.float64)
        + selection["sent_count"].astype(np.float64)
    )
    selection["activity_rank"] = selection["activity_hint"].rank(method="first", ascending=True)
    selection["tie_breaker"] = rng.random(len(selection))

    # Preserve the full activity distribution instead of only keeping the top
    # most-active users, which makes the task unrealistically easy.
    requested = int(max_users)
    num_bins = min(10, max(2, requested // 400, 2))
    try:
        selection["activity_bin"] = pd.qcut(
            selection["activity_rank"],
            q=min(num_bins, len(selection)),
            labels=False,
            duplicates="drop",
        ).astype(int)
    except Exception:
        selection["activity_bin"] = 0

    sampled_indices: list[int] = []
    fractional_remainders: list[tuple[float, int]] = []
    total_rows = float(len(selection))
    bin_counts = selection["activity_bin"].value_counts().sort_index()
    guaranteed_bins = int(min(len(bin_counts), requested))

    for order_index, (bin_id, bin_size) in enumerate(bin_counts.items()):
        candidate_indices = selection.index[selection["activity_bin"] == int(bin_id)].to_numpy(dtype=np.int64)
        proportional = requested * (float(bin_size) / total_rows)
        take_count = int(np.floor(proportional))
        if guaranteed_bins > 0:
            take_count = max(take_count, 1)
        take_count = min(take_count, len(candidate_indices))
        if take_count > 0:
            chosen = rng.choice(candidate_indices, size=take_count, replace=False)
            sampled_indices.extend(int(index) for index in chosen.tolist())
        fractional_remainders.append((proportional - np.floor(proportional), int(bin_id)))
        guaranteed_bins = max(guaranteed_bins - 1, 0)

    sampled_set = set(sampled_indices)
    if len(sampled_indices) < requested:
        remaining = requested - len(sampled_indices)
        for _, bin_id in sorted(fractional_remainders, reverse=True):
            if remaining <= 0:
                break
            candidate_indices = [
                int(index)
                for index in selection.index[selection["activity_bin"] == int(bin_id)].tolist()
                if int(index) not in sampled_set
            ]
            if not candidate_indices:
                continue
            take_count = min(remaining, len(candidate_indices))
            chosen = rng.choice(np.asarray(candidate_indices, dtype=np.int64), size=take_count, replace=False)
            sampled_indices.extend(int(index) for index in chosen.tolist())
            sampled_set.update(int(index) for index in chosen.tolist())
            remaining -= take_count

    if len(sampled_indices) < requested:
        candidate_indices = [
            int(index)
            for index in selection.index.tolist()
            if int(index) not in sampled_set
        ]
        if candidate_indices:
            chosen = rng.choice(
                np.asarray(candidate_indices, dtype=np.int64),
                size=min(requested - len(sampled_indices), len(candidate_indices)),
                replace=False,
            )
            sampled_indices.extend(int(index) for index in chosen.tolist())
            sampled_set.update(int(index) for index in chosen.tolist())

    if len(sampled_indices) > requested:
        chosen = rng.choice(np.asarray(sampled_indices, dtype=np.int64), size=requested, replace=False)
        sampled_indices = [int(index) for index in chosen.tolist()]

    return selection.loc[sorted(set(sampled_indices))].drop(
        columns=["graph_degree_hint", "activity_hint", "activity_rank", "activity_bin", "tie_breaker"]
    ).reset_index(drop=True)


def _build_transaction_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["address", "tx_out_count", "tx_in_count", "tx_total_count", "tx_error_rate"])

    outbound = transactions.groupby("src_address").agg(
        tx_out_count=("src_address", "size"),
        tx_out_value_eth=("value_eth", "sum"),
        tx_out_mean_gas_used=("gas_used", "mean"),
        tx_out_counterparties=("dst_address", "nunique"),
        tx_error_count=("is_error", "sum"),
    )
    inbound = transactions.groupby("dst_address").agg(
        tx_in_count=("dst_address", "size"),
        tx_in_value_eth=("value_eth", "sum"),
        tx_in_mean_gas_used=("gas_used", "mean"),
        tx_in_counterparties=("src_address", "nunique"),
    )
    aggregate = outbound.join(inbound, how="outer")

    for relation in sorted(set(transactions["relation_type"].dropna().astype(str).tolist())):
        relation_frame = transactions[transactions["relation_type"] == relation]
        out_counts = relation_frame.groupby("src_address").size().rename(f"tx_out_{relation}_count")
        in_counts = relation_frame.groupby("dst_address").size().rename(f"tx_in_{relation}_count")
        aggregate = aggregate.join(out_counts, how="outer")
        aggregate = aggregate.join(in_counts, how="outer")

    aggregate = aggregate.fillna(0.0)
    aggregate["tx_total_count"] = aggregate["tx_out_count"] + aggregate["tx_in_count"]
    aggregate["tx_total_counterparties"] = aggregate["tx_out_counterparties"] + aggregate["tx_in_counterparties"]
    aggregate["tx_error_rate"] = aggregate["tx_error_count"] / np.maximum(aggregate["tx_out_count"], 1.0)
    aggregate = aggregate.reset_index().rename(columns={"index": "address", "src_address": "address"})
    return aggregate


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    series = pd.Series(values.astype(np.float64))
    ranks = series.rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(ranks, 0.0, 1.0)


def _derive_weak_labels(
    users: pd.DataFrame,
    positive_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    ratio = float(np.clip(positive_ratio, 0.05, 0.45))
    zeros = pd.Series(np.zeros(len(users), dtype=np.float64))
    transaction_total = users["transaction_count"].to_numpy(dtype=np.float64) + users.get("tx_total_count", zeros).to_numpy(
        dtype=np.float64
    )
    active_span_hours = users["active_span_hours"].to_numpy(dtype=np.float64)
    sent_flow = users["sent_count"].to_numpy(dtype=np.float64) + users["nested_sent_count"].to_numpy(dtype=np.float64)
    recv_flow = users["received_count"].to_numpy(dtype=np.float64) + users["nested_received_count"].to_numpy(dtype=np.float64)
    sent_value = users["total_sent_eth"].to_numpy(dtype=np.float64) + users["nested_total_value_eth"].to_numpy(
        dtype=np.float64
    )
    recv_value = users["total_received_eth"].to_numpy(dtype=np.float64)
    stablecoin_focus = (
        users["type_stablecoin"].to_numpy(dtype=np.float64)
        + users.get("tx_out_stablecoin_count", zeros).to_numpy(dtype=np.float64)
        + users.get("tx_in_stablecoin_count", zeros).to_numpy(dtype=np.float64)
    ) / np.maximum(users["protocol_type_total"].to_numpy(dtype=np.float64) + transaction_total, 1.0)
    error_rate = users.get("tx_error_rate", zeros).to_numpy(dtype=np.float64)
    counterparty_concentration = 1.0 - np.clip(
        users.get("tx_total_counterparties", zeros).to_numpy(dtype=np.float64)
        / np.maximum(users.get("tx_total_count", pd.Series(np.ones(len(users)))).to_numpy(dtype=np.float64), 1.0),
        0.0,
        1.0,
    )

    activity_burst = np.log1p(transaction_total) / np.log1p(active_span_hours + 2.0)
    short_lived = 1.0 / (1.0 + active_span_hours)
    flow_imbalance = np.abs(np.log1p(sent_flow) - np.log1p(recv_flow))
    volume_imbalance = np.abs(np.log1p(sent_value + 1e-6) - np.log1p(recv_value + 1e-6))

    risk_score = (
        0.22 * _rank01(activity_burst)
        + 0.18 * _rank01(flow_imbalance)
        + 0.16 * _rank01(volume_imbalance)
        + 0.16 * _rank01(stablecoin_focus)
        + 0.12 * _rank01(error_rate)
        + 0.08 * _rank01(counterparty_concentration)
        + 0.08 * _rank01(short_lived)
    )

    num_nodes = len(users)
    if num_nodes < 2:
        return np.zeros(num_nodes, dtype=np.int64), risk_score, float(risk_score[0]) if num_nodes else 0.0

    positive_count = int(np.clip(round(num_nodes * ratio), 1, max(num_nodes - 1, 1)))
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(num_nodes)
    order = np.lexsort((tie_breaker, -risk_score))
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[order[:positive_count]] = 1
    threshold = float(risk_score[order[positive_count - 1]])
    return labels, risk_score.astype(np.float64), threshold


def _feature_matrix_from_users(users: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    excluded_columns = {
        "address",
        "first_seen",
        "last_seen",
        "first_seen_dt",
        "last_seen_dt",
        "protocol_types",
        "protocols_used",
        "transactions",
        "weak_label",
        "risk_score",
    }
    feature_columns: list[str] = []
    for column in users.columns:
        if column in excluded_columns:
            continue
        if column in ARCHIVE_FEATURE_LEAKAGE_GUARD_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(users[column]):
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError("Archive users table did not yield any numeric feature columns.")

    feature_frame = users[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    feature_values = feature_frame.to_numpy(dtype=np.float32)
    feature_values = np.sign(feature_values) * np.log1p(np.abs(feature_values))
    means = feature_values.mean(axis=0, keepdims=True)
    stds = feature_values.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    normalized = (feature_values - means) / stds
    return normalized.astype(np.float32), feature_columns


def _stratified_split_masks(labels: np.ndarray, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = len(labels)
    train_mask = np.zeros(num_nodes, dtype=bool)
    valid_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    rng = np.random.default_rng(seed)

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = len(indices)
        if count == 1:
            train_mask[indices] = True
            continue
        if count == 2:
            train_mask[indices[:1]] = True
            test_mask[indices[1:]] = True
            continue
        if count == 3:
            train_mask[indices[:1]] = True
            valid_mask[indices[1:2]] = True
            test_mask[indices[2:]] = True
            continue

        train_count = max(1, int(round(count * 0.70)))
        valid_count = max(1, int(round(count * 0.15)))
        if train_count + valid_count >= count:
            train_count = max(1, count - 2)
            valid_count = 1
        test_count = count - train_count - valid_count
        if test_count <= 0:
            test_count = 1
            if train_count > valid_count:
                train_count -= 1
            else:
                valid_count = max(valid_count - 1, 1)
        train_mask[indices[:train_count]] = True
        valid_mask[indices[train_count : train_count + valid_count]] = True
        test_mask[indices[train_count + valid_count : train_count + valid_count + test_count]] = True

    if not valid_mask.any():
        train_indices = np.flatnonzero(train_mask)
        valid_mask[train_indices[-1:]] = True
        train_mask[train_indices[-1:]] = False
    if not test_mask.any():
        train_indices = np.flatnonzero(train_mask)
        test_mask[train_indices[-1:]] = True
        train_mask[train_indices[-1:]] = False

    return (
        torch.from_numpy(train_mask),
        torch.from_numpy(valid_mask),
        torch.from_numpy(test_mask),
    )


def _compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=2).float().clamp(min=1.0)
    return counts.sum() / (counts * len(counts))


def _compute_class_counts(labels: torch.Tensor) -> torch.Tensor:
    return torch.bincount(labels.long(), minlength=2).float().clamp(min=1.0)


def _window_edge_pairs(indices: np.ndarray, neighbors: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src: list[int] = []
    dst: list[int] = []
    max_offset = max(int(neighbors), 1)
    for position, node_id in enumerate(indices):
        upper = min(position + max_offset + 1, len(indices))
        for next_position in range(position + 1, upper):
            other = int(indices[next_position])
            src.extend([int(node_id), other])
            dst.extend([other, int(node_id)])
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def _build_synthetic_edge_dict(
    users: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]], dict[str, int], str]:
    activity_signal = (
        np.log1p(users["transaction_count"].to_numpy(dtype=np.float64))
        + np.log1p(users["received_count"].to_numpy(dtype=np.float64))
        + np.log1p(users["sent_count"].to_numpy(dtype=np.float64))
        + np.log1p(users["nested_total_value_eth"].to_numpy(dtype=np.float64))
        + np.log1p(users["active_span_hours"].to_numpy(dtype=np.float64) + 1.0)
    )
    relation_diversity = users["protocol_type_total"].to_numpy(dtype=np.float64)
    global_order = np.lexsort((relation_diversity, activity_signal))
    homo_src, homo_dst = _window_edge_pairs(global_order, neighbors=3)
    if len(homo_src) == 0:
        singleton = np.arange(len(users), dtype=np.int64)
        homo_src, homo_dst = singleton.copy(), singleton.copy()

    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {
        (NODE_TYPE, "homo", NODE_TYPE): (
            torch.from_numpy(homo_src),
            torch.from_numpy(homo_dst),
        )
    }
    relation_edge_counts: dict[str, int] = {"homo": int(len(homo_src))}

    for relation in SUPPORTED_RELATIONS:
        relation_nodes = np.flatnonzero(users[f"type_{relation}"].to_numpy(dtype=np.float64) > 0.0)
        if len(relation_nodes) < 2:
            continue
        relation_signal = (
            np.log1p(users.iloc[relation_nodes][f"type_{relation}"].to_numpy(dtype=np.float64))
            + np.log1p(users.iloc[relation_nodes]["transaction_count"].to_numpy(dtype=np.float64))
            + np.log1p(users.iloc[relation_nodes]["nested_unique_protocols"].to_numpy(dtype=np.float64) + 1.0)
        )
        relation_order = relation_nodes[np.argsort(relation_signal, kind="mergesort")]
        relation_src, relation_dst = _window_edge_pairs(relation_order, neighbors=3)
        if len(relation_src) == 0:
            continue
        edge_dict[(NODE_TYPE, relation, NODE_TYPE)] = (
            torch.from_numpy(relation_src),
            torch.from_numpy(relation_dst),
        )
        relation_edge_counts[relation] = int(len(relation_src))

    if len(edge_dict) == 1:
        edge_dict[(NODE_TYPE, "transfer", NODE_TYPE)] = (
            torch.from_numpy(homo_src.copy()),
            torch.from_numpy(homo_dst.copy()),
        )
        relation_edge_counts["transfer"] = int(len(homo_src))
    return edge_dict, relation_edge_counts, "synthetic_user_graph"


def load_archive_dataset(
    *,
    data_root: str | Path = ARCHIVE_DEFAULT_ROOT,
    dataset_name: str = "archive",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    max_users: int | None = 4000,
    max_transactions: int | None = 50000,
    risk_positive_ratio: float = 0.15,
    force_preview: bool = False,
) -> DatasetBundle:
    archive_root = Path(data_root).expanduser().resolve()
    user_row_budget = None if max_users is None else max(int(max_users) * 8, int(max_users))
    transactions_frame, transactions_source = _load_archive_table(
        parquet_path=archive_root / "dataset" / "data" / "transactions.parquet",
        preview_path=archive_root / "dataset" / "preview" / "transactions.csv",
        max_rows=max_transactions,
        force_preview=force_preview,
        table_name="transactions",
    )
    transaction_candidates: set[str] = set()
    if not transactions_frame.empty:
        raw_transaction_columns = {_normalize_column_name(column): column for column in transactions_frame.columns}
        src_column = raw_transaction_columns.get("from", raw_transaction_columns.get("src_address"))
        dst_column = raw_transaction_columns.get("to", raw_transaction_columns.get("dst_address"))
        for column in (src_column, dst_column):
            if column is None:
                continue
            transaction_candidates.update(
                transactions_frame[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .loc[lambda values: values.str.match(r"^0x[a-f0-9]{40}$", na=False)]
                .tolist()
            )
    users_frame, users_source = _load_archive_users_for_transactions(
        parquet_path=archive_root / "processed" / "users_processed.parquet",
        preview_path=archive_root / "dataset" / "preview" / "users.csv",
        max_rows=user_row_budget,
        force_preview=force_preview,
        candidate_addresses=transaction_candidates,
    )

    users = _prepare_users_frame(users_frame)
    transactions = _prepare_transactions_frame(transactions_frame)
    users = _select_user_sample(users, transactions, max_users=max_users, seed=seed)

    selected_addresses = set(users["address"].tolist())
    transactions = transactions[
        transactions["src_address"].isin(selected_addresses) & transactions["dst_address"].isin(selected_addresses)
    ].copy()

    aggregates = _build_transaction_aggregates(transactions)
    users = users.merge(aggregates, on="address", how="left")
    numeric_columns = [column for column in users.columns if pd.api.types.is_numeric_dtype(users[column])]
    users[numeric_columns] = users[numeric_columns].fillna(0.0)

    weak_labels, risk_scores, risk_threshold = _derive_weak_labels(
        users,
        positive_ratio=risk_positive_ratio,
        seed=seed,
    )
    users["weak_label"] = weak_labels.astype(np.int64)
    users["risk_score"] = risk_scores.astype(np.float64)
    feature_matrix, feature_columns = _feature_matrix_from_users(users)
    train_mask, valid_mask, test_mask = _stratified_split_masks(weak_labels, seed=seed)

    graph_source = "transaction_graph"
    if transactions.empty:
        edge_dict, relation_edge_counts, graph_source = _build_synthetic_edge_dict(users)
    else:
        address_to_node = {address: index for index, address in enumerate(users["address"].tolist())}
        src_nodes = transactions["src_address"].map(address_to_node).to_numpy(dtype=np.int64)
        dst_nodes = transactions["dst_address"].map(address_to_node).to_numpy(dtype=np.int64)
        edge_dict = {
            (NODE_TYPE, "homo", NODE_TYPE): (
                torch.from_numpy(src_nodes.astype(np.int64)),
                torch.from_numpy(dst_nodes.astype(np.int64)),
            )
        }
        relation_edge_counts = {"homo": int(len(src_nodes))}
        for relation in SUPPORTED_RELATIONS:
            mask = transactions["relation_type"] == relation
            if not bool(mask.any()):
                continue
            relation_src = torch.from_numpy(src_nodes[mask.to_numpy()].astype(np.int64))
            relation_dst = torch.from_numpy(dst_nodes[mask.to_numpy()].astype(np.int64))
            edge_dict[(NODE_TYPE, f"{relation}_out", NODE_TYPE)] = (relation_src, relation_dst)
            edge_dict[(NODE_TYPE, f"{relation}_in", NODE_TYPE)] = (relation_dst, relation_src)
            relation_edge_counts[f"{relation}_out"] = int(len(relation_src))
            relation_edge_counts[f"{relation}_in"] = int(len(relation_src))

        if len(edge_dict) == 1:
            edge_dict[(NODE_TYPE, "transfer_out", NODE_TYPE)] = (
                torch.from_numpy(src_nodes.astype(np.int64)),
                torch.from_numpy(dst_nodes.astype(np.int64)),
            )
            edge_dict[(NODE_TYPE, "transfer_in", NODE_TYPE)] = (
                torch.from_numpy(dst_nodes.astype(np.int64)),
                torch.from_numpy(src_nodes.astype(np.int64)),
            )
            relation_edge_counts["transfer_out"] = int(len(src_nodes))
            relation_edge_counts["transfer_in"] = int(len(src_nodes))

    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(users)})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix)
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(weak_labels.astype(np.int64))
    graph.nodes[NODE_TYPE].data["risk_score"] = torch.from_numpy(risk_scores.astype(np.float32))
    graph.nodes[NODE_TYPE].data["label_confidence_target"] = torch.from_numpy(
        _build_archive_label_confidence(risk_scores, risk_threshold)
    )
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)

    if float(label_fraction) < 0.999:
        _apply_label_scarcity(graph, label_fraction=float(label_fraction), seed=seed)
    else:
        graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.bool().clone()
        graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask.bool())
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
    graph.nodes[NODE_TYPE].data["event_sequence"] = event_sequence.float()
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()
    relation_order = _attach_relation_sequence(graph, dataset_name="archive")
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
    bundle = DatasetBundle(
        name=dataset_name,
        graph=graph,
        node_type=NODE_TYPE,
        relation_order=relation_order,
        class_weights=_compute_class_weights(class_labels),
        class_counts=_compute_class_counts(class_labels),
        clients=clients,
        base_lr=1e-3,
    )
    bundle.addresses = users["address"].tolist()
    bundle.data_summary = {
        "dataset_display_name": ARCHIVE_PUBLIC_NAME,
        "data_root": str(archive_root),
        "users_source": users_source,
        "transactions_source": transactions_source,
        "feature_columns": feature_columns,
        "feature_dim": int(feature_matrix.shape[1]),
        "num_nodes": int(graph.num_nodes(NODE_TYPE)),
        "num_clients": int(len(clients)),
        "relation_edge_counts": relation_edge_counts,
        "graph_source": graph_source,
        "train_nodes": int(graph.nodes[NODE_TYPE].data["train_mask"].sum().item()),
        "valid_nodes": int(graph.nodes[NODE_TYPE].data["valid_mask"].sum().item()),
        "test_nodes": int(graph.nodes[NODE_TYPE].data["test_mask"].sum().item()),
        "positive_labels": int(np.sum(weak_labels)),
        "positive_ratio": float(np.mean(weak_labels.astype(np.float32))),
        "risk_positive_ratio": float(np.clip(risk_positive_ratio, 0.05, 0.45)),
        "risk_threshold": float(risk_threshold),
        "weak_label_strategy": "heuristic_risk_score_top_fraction",
        "event_sequence_length": int(ARCHIVE_EVENT_SEQUENCE_LENGTH),
        "sampling_strategy": "activity_stratified_user_sample",
        "feature_leakage_guard_columns": sorted(ARCHIVE_FEATURE_LEAKAGE_GUARD_COLUMNS),
        "force_preview": bool(force_preview),
        "max_users": None if max_users is None else int(max_users),
        "max_transactions": None if max_transactions is None else int(max_transactions),
        "user_row_budget": None if user_row_budget is None else int(user_row_budget),
    }
    return bundle
