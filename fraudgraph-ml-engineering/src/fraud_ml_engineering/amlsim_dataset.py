from __future__ import annotations

"""AMLSim account-level dataset loader."""

import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
from typing import Any, Iterator

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

from .checkpointing import atomic_write_json
from .paths import CACHE_ROOT, DATA_ROOT
from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    SEQUENCE_BUILDER_VERSION,
    _apply_active_learning_feedback,
    _apply_label_scarcity,
    _attach_dataset_context_defaults,
    _attach_relation_sequence,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _stratified_partition,
)

AMLSIM_ROOT = DATA_ROOT / "amlsim"
AMLSIM_OUTPUTS_ROOT = AMLSIM_ROOT / "outputs"
AMLSIM_FORMAL_DEFAULT_SIMULATION = "MAFRL_15K_300K_formal_v1"
AMLSIM_DEFAULT_ROOT = AMLSIM_OUTPUTS_ROOT / AMLSIM_FORMAL_DEFAULT_SIMULATION
AMLSIM_SAMPLE_OUTPUT_ROOT = AMLSIM_ROOT / "sample" / "outputs"
AMLSIM_CACHE_DIR = CACHE_ROOT
AMLSIM_PUBLIC_NAME = "AMLSim Account AML"
NODE_TYPE = "account"
AMLSIM_DEFAULT_TRAIN_RATIO = 0.70
AMLSIM_DEFAULT_VALID_RATIO = 0.15
AMLSIM_DEFAULT_RELATION_WINDOW_NEIGHBORS = 4
AMLSIM_DEFAULT_ACTIVITY_BINS = 8
AMLSIM_DEFAULT_EVENT_HISTORY_LEN = 12
AMLSIM_GRAPH_BUILDER_VERSION = f"amlsim_account_v2::{SEQUENCE_BUILDER_VERSION}"
AMLSIM_CACHE_LOCK_TIMEOUT_SECONDS = 1800.0
AMLSIM_CACHE_LOCK_STALE_SECONDS = 7200.0
AMLSIM_CACHE_LOCK_POLL_SECONDS = 0.25

ACCOUNT_FILE_NAMES = ("accounts.csv",)
TRANSACTION_FILE_NAMES = ("transactions.csv", "tx.csv")
ALERT_ACCOUNT_FILE_NAMES = ("alert_accounts.csv", "alert_members.csv", "alerts.csv")
SAR_ACCOUNT_FILE_NAMES = ("sar_accounts.csv",)
CASH_TRANSACTION_FILE_NAMES = ("cash_tx.csv",)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig", skip_blank_lines=True)


def _normalize_column_name(value: object) -> str:
    return str(value or "").replace("\ufeff", "").strip().lower()


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {_normalize_column_name(column): str(column) for column in frame.columns}


def _resolve_column(frame: pd.DataFrame, names: tuple[str, ...], *, required: bool = False) -> str | None:
    lookup = _column_lookup(frame)
    for name in names:
        column = lookup.get(_normalize_column_name(name))
        if column is not None:
            return column
    if required:
        raise KeyError(f"Missing required AMLSim column. Expected one of: {names}")
    return None


def _standardize_id_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip()
    normalized = normalized.str.replace(r"\.0+$", "", regex=True)
    return normalized


def _to_numeric_series(series: pd.Series, *, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        fill_value = float(numeric.median())
    else:
        fill_value = float(default)
    return numeric.fillna(fill_value).astype(np.float32)


def _to_time_series(series: pd.Series | None, *, size: int) -> pd.Series:
    if series is None:
        return pd.Series(np.zeros(size, dtype=np.float32))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        fill_value = float(numeric.median())
        return numeric.fillna(fill_value).astype(np.float32)
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    if dt.notna().any():
        anchor = dt.dropna().min()
        delta_days = (dt - anchor).dt.total_seconds().div(86400.0)
        fill_value = float(delta_days.dropna().median()) if delta_days.notna().any() else 0.0
        return delta_days.fillna(fill_value).astype(np.float32)
    return pd.Series(np.zeros(size, dtype=np.float32))


def _to_bool_series(series: pd.Series | None, *, size: int) -> pd.Series:
    if series is None:
        return pd.Series(np.zeros(size, dtype=bool))
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y", "sar"})


def _first_existing_file(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate.resolve()
    return None


def _candidate_directory_payload(directory: Path) -> dict[str, Path] | None:
    if not directory.exists() or not directory.is_dir():
        return None
    accounts_path = _first_existing_file(directory, ACCOUNT_FILE_NAMES)
    transactions_path = _first_existing_file(directory, TRANSACTION_FILE_NAMES)
    if accounts_path is None or transactions_path is None:
        return None
    payload = {
        "accounts": accounts_path,
        "transactions": transactions_path,
    }
    alert_accounts_path = _first_existing_file(directory, ALERT_ACCOUNT_FILE_NAMES)
    sar_accounts_path = _first_existing_file(directory, SAR_ACCOUNT_FILE_NAMES)
    cash_tx_path = _first_existing_file(directory, CASH_TRANSACTION_FILE_NAMES)
    if alert_accounts_path is not None:
        payload["alert_accounts"] = alert_accounts_path
    if sar_accounts_path is not None:
        payload["sar_accounts"] = sar_accounts_path
    if cash_tx_path is not None:
        payload["cash_transactions"] = cash_tx_path
    return payload


def _resolve_amlsim_source(
    data_root: str | Path,
    *,
    allow_sample_fallback: bool,
) -> dict[str, Any]:
    requested_root = Path(data_root).expanduser().resolve()
    direct_payload = _candidate_directory_payload(requested_root)
    if direct_payload is not None:
        source_kind = "sample" if requested_root == AMLSIM_SAMPLE_OUTPUT_ROOT.resolve() else "generated"
        return {
            "requested_root": requested_root,
            "resolved_root": requested_root,
            "files": direct_payload,
            "source_kind": source_kind,
            "sample_fallback_used": bool(source_kind == "sample"),
        }

    formal_default_root = AMLSIM_DEFAULT_ROOT.expanduser().resolve()
    if requested_root == AMLSIM_OUTPUTS_ROOT.expanduser().resolve():
        formal_payload = _candidate_directory_payload(formal_default_root)
        if formal_payload is not None:
            return {
                "requested_root": requested_root,
                "resolved_root": formal_default_root,
                "files": formal_payload,
                "source_kind": "generated",
                "sample_fallback_used": False,
            }

    candidates: list[Path] = []
    if requested_root.exists():
        for child in sorted(requested_root.iterdir()):
            if child.is_dir():
                candidates.append(child.resolve())
        outputs_child = requested_root / "outputs"
        if outputs_child.exists() and outputs_child.is_dir():
            candidates.append(outputs_child.resolve())
            for child in sorted(outputs_child.iterdir()):
                if child.is_dir():
                    candidates.append(child.resolve())
    seen: set[str] = set()
    valid_payloads: list[tuple[Path, dict[str, Path]]] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        payload = _candidate_directory_payload(candidate)
        if payload is not None:
            valid_payloads.append((candidate, payload))
    if valid_payloads:
        if len(valid_payloads) > 1:
            available = "\n  - ".join(str(path) for path, _ in valid_payloads)
            raise FileNotFoundError(
                "Ambiguous AMLSim data root: multiple candidate output directories were found under "
                f"{requested_root}.\nPass an explicit --amlsim_data_root pointing to one dataset root.\n  - {available}"
            )
        resolved_root, files = valid_payloads[0]
        source_kind = "sample" if resolved_root == AMLSIM_SAMPLE_OUTPUT_ROOT.resolve() else "generated"
        return {
            "requested_root": requested_root,
            "resolved_root": resolved_root,
            "files": files,
            "source_kind": source_kind,
            "sample_fallback_used": bool(source_kind == "sample"),
        }
    sample_payload = _candidate_directory_payload(AMLSIM_SAMPLE_OUTPUT_ROOT)
    if allow_sample_fallback and sample_payload is not None:
        return {
            "requested_root": requested_root,
            "resolved_root": AMLSIM_SAMPLE_OUTPUT_ROOT.resolve(),
            "files": sample_payload,
            "source_kind": "sample",
            "sample_fallback_used": True,
        }
    tried_locations = [str(path) for path in candidates] or [str(requested_root)]
    raise FileNotFoundError(
        "Unable to resolve AMLSim outputs. Expected a directory containing accounts.csv and "
        "transactions.csv/tx.csv.\nTried:\n  - "
        + "\n  - ".join(tried_locations)
        + (
            f"\nSample fallback checked at: {AMLSIM_SAMPLE_OUTPUT_ROOT}"
            if allow_sample_fallback
            else "\nPass --amlsim_allow_sample_fallback to use AMLSim sample outputs for smoke tests."
        )
    )


def _canonicalize_accounts(frame: pd.DataFrame) -> pd.DataFrame:
    account_id_column = _resolve_column(frame, ("acct_id", "account_id", "ACCOUNT_ID"), required=True)
    start_column = _resolve_column(frame, ("open_dt", "start", "start_step"))
    end_column = _resolve_column(frame, ("close_dt", "end", "end_step"))
    balance_column = _resolve_column(frame, ("initial_deposit", "init_balance", "INIT_BALANCE"))
    bank_column = _resolve_column(frame, ("bank_id", "branch_id"))
    business_column = _resolve_column(frame, ("business", "business_type", "type", "ACCOUNT_TYPE"))
    country_column = _resolve_column(frame, ("country",))

    canonical = pd.DataFrame()
    canonical["account_id"] = _standardize_id_series(frame[account_id_column])
    canonical["initial_balance"] = _to_numeric_series(frame[balance_column], default=0.0) if balance_column else 0.0
    canonical["start_time"] = _to_time_series(frame[start_column] if start_column else None, size=len(frame))
    canonical["end_time"] = _to_time_series(frame[end_column] if end_column else None, size=len(frame))
    canonical["bank_id"] = (
        _standardize_id_series(frame[bank_column]).replace("", "unknown_bank") if bank_column else "unknown_bank"
    )
    canonical["country"] = (
        frame[country_column].fillna("unknown_country").astype(str).str.strip().replace("", "unknown_country")
        if country_column
        else "unknown_country"
    )
    canonical["business_type"] = (
        frame[business_column].fillna("unknown_business").astype(str).str.strip().replace("", "unknown_business")
        if business_column
        else "unknown_business"
    )
    canonical = canonical[canonical["account_id"].ne("")].copy()
    canonical = canonical.drop_duplicates(subset=["account_id"], keep="first").reset_index(drop=True)
    return canonical


def _canonicalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    orig_column = _resolve_column(frame, ("orig_acct", "ACCOUNT_ID", "src"), required=True)
    bene_column = _resolve_column(frame, ("bene_acct", "COUNTER_PARTY_ACCOUNT_NUM", "dst"), required=True)
    amount_column = _resolve_column(frame, ("base_amt", "TXN_AMOUNT_ORIG"))
    time_column = _resolve_column(frame, ("tran_timestamp", "start", "RUN_DATE", "EVENT_DATE"))
    tx_type_column = _resolve_column(frame, ("tx_type", "TXN_SOURCE_TYPE_CODE", "ttype"))
    tx_id_column = _resolve_column(frame, ("tran_id", "TXN_ID", "id"))
    sar_column = _resolve_column(frame, ("is_sar",))
    alert_column = _resolve_column(frame, ("alert_id", "ALERT_KEY"))

    canonical = pd.DataFrame()
    canonical["tx_id"] = (
        _standardize_id_series(frame[tx_id_column])
        if tx_id_column
        else pd.Series(np.arange(len(frame), dtype=np.int64)).astype(str)
    )
    canonical["orig_acct"] = _standardize_id_series(frame[orig_column])
    canonical["bene_acct"] = _standardize_id_series(frame[bene_column])
    canonical["amount"] = _to_numeric_series(frame[amount_column], default=0.0) if amount_column else 0.0
    canonical["timestamp"] = _to_time_series(frame[time_column] if time_column else None, size=len(frame))
    canonical["tx_type"] = (
        frame[tx_type_column].fillna("unknown_type").astype(str).str.strip().replace("", "unknown_type")
        if tx_type_column
        else "unknown_type"
    )
    canonical["is_sar"] = _to_bool_series(frame[sar_column] if sar_column else None, size=len(frame))
    canonical["alert_id"] = (
        _standardize_id_series(frame[alert_column]) if alert_column else pd.Series([""] * len(frame))
    )
    canonical = canonical[
        canonical["orig_acct"].ne("") & canonical["bene_acct"].ne("") & canonical["timestamp"].notna()
    ].copy()
    canonical = canonical.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    return canonical


def _canonicalize_cash_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    account_column = _resolve_column(frame, ("acct_id", "account_id", "ACCOUNT_ID"))
    amount_column = _resolve_column(frame, ("base_amt", "TXN_AMOUNT_ORIG"))
    time_column = _resolve_column(frame, ("tran_timestamp", "RUN_DATE", "start", "EVENT_DATE"))
    if account_column is None or amount_column is None:
        return pd.DataFrame(columns=["account_id", "amount", "timestamp"])
    canonical = pd.DataFrame()
    canonical["account_id"] = _standardize_id_series(frame[account_column])
    canonical["amount"] = _to_numeric_series(frame[amount_column], default=0.0)
    canonical["timestamp"] = _to_time_series(frame[time_column] if time_column else None, size=len(frame))
    canonical = canonical[canonical["account_id"].ne("")].copy()
    return canonical.reset_index(drop=True)


def _infer_positive_account_ids(
    *,
    files: dict[str, Path],
    raw_accounts: pd.DataFrame,
) -> tuple[set[str], str]:
    if "sar_accounts" in files:
        sar_frame = _read_csv(files["sar_accounts"])
        account_column = _resolve_column(sar_frame, ("acct_id", "account_id", "ACCOUNT_ID", "accountID"), required=True)
        positives = set(_standardize_id_series(sar_frame[account_column]).tolist())
        positives.discard("")
        if positives:
            return positives, "sar_accounts"
    if "alert_accounts" in files:
        alert_frame = _read_csv(files["alert_accounts"])
        account_column = _resolve_column(
            alert_frame,
            ("acct_id", "account_id", "ACCOUNT_ID", "accountID"),
            required=True,
        )
        is_sar_column = _resolve_column(alert_frame, ("is_sar", "isSAR", "Escalated_To_Case_Investigation"))
        if is_sar_column is not None:
            sar_mask = _to_bool_series(alert_frame[is_sar_column], size=len(alert_frame))
            positives = set(_standardize_id_series(alert_frame.loc[sar_mask, account_column]).tolist())
        else:
            positives = set(_standardize_id_series(alert_frame[account_column]).tolist())
        positives.discard("")
        if positives:
            return positives, "alert_accounts"
    for column_group, label_source in (
        (("prior_sar_count",), "accounts_prior_sar_count"),
        (("isFraud", "is_fraud", "suspicious"), "accounts_flag"),
    ):
        label_column = _resolve_column(raw_accounts, column_group)
        if label_column is None:
            continue
        mask = _to_bool_series(raw_accounts[label_column], size=len(raw_accounts))
        account_column = _resolve_column(
            raw_accounts,
            ("acct_id", "account_id", "ACCOUNT_ID", "accountID"),
            required=True,
        )
        positives = set(_standardize_id_series(raw_accounts.loc[mask, account_column]).tolist())
        positives.discard("")
        if positives:
            return positives, label_source
    raise ValueError(
        "Unable to infer AMLSim positive labels. Provide sar_accounts.csv, alert_accounts.csv/alerts.csv, "
        "or account-level SAR flags."
    )


def _add_missing_accounts(accounts: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    known_ids = set(accounts["account_id"].tolist())
    all_tx_ids = set(transactions["orig_acct"].tolist()) | set(transactions["bene_acct"].tolist())
    missing_ids = sorted(account_id for account_id in all_tx_ids if account_id and account_id not in known_ids)
    if not missing_ids:
        return accounts
    missing_frame = pd.DataFrame(
        {
            "account_id": missing_ids,
            "initial_balance": np.zeros(len(missing_ids), dtype=np.float32),
            "start_time": np.zeros(len(missing_ids), dtype=np.float32),
            "end_time": np.zeros(len(missing_ids), dtype=np.float32),
            "bank_id": ["unknown_bank"] * len(missing_ids),
            "country": ["unknown_country"] * len(missing_ids),
            "business_type": ["unknown_business"] * len(missing_ids),
        }
    )
    merged = pd.concat([accounts, missing_frame], axis=0, ignore_index=True)
    return merged.drop_duplicates(subset=["account_id"], keep="first").reset_index(drop=True)


def _chronological_class_aware_split(
    sort_time: np.ndarray,
    labels: np.ndarray,
    *,
    train_ratio: float,
    valid_ratio: float,
    force_class_aware: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(len(labels))
    if num_nodes < 8:
        raise ValueError("AMLSim sample is too small to build train/valid/test masks.")
    order = np.argsort(sort_time, kind="mergesort")
    train_end = min(max(int(round(num_nodes * float(train_ratio))), 1), num_nodes - 2)
    valid_end = min(max(int(round(num_nodes * float(train_ratio + valid_ratio))), train_end + 1), num_nodes - 1)

    def _masks_from_order(target_order: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[target_order[:train_end]] = True
        valid_mask[target_order[train_end:valid_end]] = True
        test_mask[target_order[valid_end:]] = True
        return train_mask, valid_mask, test_mask

    train_mask, valid_mask, test_mask = _masks_from_order(order)
    train_labels = labels[train_mask.cpu().numpy()]
    if (
        not bool(force_class_aware)
        and train_labels.size > 0
        and len(np.unique(train_labels)) >= min(2, len(np.unique(labels)))
    ):
        return train_mask, valid_mask, test_mask

    partitions = {"train": [], "valid": [], "test": []}
    for label in sorted(np.unique(labels).tolist()):
        label_nodes = order[labels[order] == label]
        if label_nodes.size == 0:
            continue
        label_train_end = min(max(int(round(label_nodes.size * float(train_ratio))), 1), max(label_nodes.size - 2, 1))
        label_valid_end = min(
            max(int(round(label_nodes.size * float(train_ratio + valid_ratio))), label_train_end + 1),
            max(label_nodes.size - 1, label_train_end + 1),
        )
        partitions["train"].append(label_nodes[:label_train_end])
        partitions["valid"].append(label_nodes[label_train_end:label_valid_end])
        partitions["test"].append(label_nodes[label_valid_end:])

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for node_ids in partitions["train"]:
        if len(node_ids) > 0:
            train_mask[torch.from_numpy(node_ids.astype(np.int64))] = True
    for node_ids in partitions["valid"]:
        if len(node_ids) > 0:
            valid_mask[torch.from_numpy(node_ids.astype(np.int64))] = True
    assigned_mask = train_mask | valid_mask
    remaining = (~assigned_mask).nonzero(as_tuple=False).flatten()
    test_mask[remaining] = True
    return train_mask, valid_mask, test_mask


def _fit_feature_preprocessor(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    feature_columns = list(frame.columns)
    processed_columns: list[np.ndarray] = []
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        fill_value = float(values.loc[train_mask].median()) if values.loc[train_mask].notna().any() else 0.0
        values = values.fillna(fill_value).to_numpy(dtype=np.float32)
        train_values = values[train_mask]
        mean = float(train_values.mean()) if train_values.size > 0 else 0.0
        std = float(train_values.std()) if train_values.size > 0 else 1.0
        if std < 1e-6:
            std = 1.0
        processed_columns.append(((values - mean) / std).reshape(-1, 1).astype(np.float32))
    if not processed_columns:
        return np.zeros((len(frame), 0), dtype=np.float32), feature_columns
    return np.concatenate(processed_columns, axis=1).astype(np.float32), feature_columns


def _value_relation_edges(values: pd.Series, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    value_to_nodes: dict[str, list[int]] = {}
    for node_id, raw_value in enumerate(values.tolist()):
        text = str(raw_value).strip()
        if not text or text.startswith("unknown_"):
            continue
        value_to_nodes.setdefault(text, []).append(int(node_id))
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    neighbor_limit = max(int(max_neighbors), 1)
    for nodes in value_to_nodes.values():
        if len(nodes) < 2:
            continue
        ordered = np.asarray(nodes, dtype=np.int64)
        for offset in range(1, min(neighbor_limit, len(ordered) - 1) + 1):
            src_parts.append(ordered[:-offset])
            dst_parts.append(ordered[offset:])
            src_parts.append(ordered[offset:])
            dst_parts.append(ordered[:-offset])
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts, axis=0), np.concatenate(dst_parts, axis=0)


def _temporal_near_edges(order: np.ndarray, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if len(order) <= 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    neighbor_limit = max(int(max_neighbors), 1)
    for offset in range(1, min(neighbor_limit, len(order) - 1) + 1):
        src_parts.append(order[:-offset])
        dst_parts.append(order[offset:])
        src_parts.append(order[offset:])
        dst_parts.append(order[:-offset])
    return np.concatenate(src_parts, axis=0), np.concatenate(dst_parts, axis=0)


def _activity_bin_codes(values: np.ndarray, bins: int) -> pd.Series:
    if len(values) == 0:
        return pd.Series([], dtype=object)
    ranked = pd.Series(values).rank(method="first")
    quantiles = min(max(int(bins), 1), len(values))
    if quantiles <= 1:
        codes = pd.Series(np.zeros(len(values), dtype=np.int32))
    else:
        codes = pd.qcut(ranked, q=quantiles, labels=False, duplicates="drop")
        if codes is None:
            codes = pd.Series(np.zeros(len(values), dtype=np.int32))
    return pd.Series(codes).fillna(0).astype(np.int32).map(lambda value: f"activity_{int(value)}")


def _cache_signature(
    *,
    source_info: dict[str, Any],
    train_ratio: float,
    valid_ratio: float,
    relation_window_neighbors: int,
    activity_bins: int,
    event_history_len: int,
) -> dict[str, Any]:
    file_state = {}
    for name, path in source_info["files"].items():
        stat = Path(path).stat()
        file_state[name] = {
            "path": str(path),
            "mtime": float(stat.st_mtime),
            "size": int(stat.st_size),
        }
    return {
        "graph_builder_version": AMLSIM_GRAPH_BUILDER_VERSION,
        "resolved_root": str(source_info["resolved_root"]),
        "source_kind": str(source_info["source_kind"]),
        "sample_fallback_used": bool(source_info["sample_fallback_used"]),
        "files": file_state,
        "train_ratio": float(train_ratio),
        "valid_ratio": float(valid_ratio),
        "relation_window_neighbors": int(relation_window_neighbors),
        "activity_bins": int(activity_bins),
        "event_history_len": int(event_history_len),
    }


def _resolve_cache_paths(signature: dict[str, Any]) -> tuple[Path, Path]:
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return AMLSIM_CACHE_DIR / f"amlsim_{digest}.dgl", AMLSIM_CACHE_DIR / f"amlsim_{digest}.json"


def _cache_lock_path(metadata_path: Path) -> Path:
    return metadata_path.with_suffix(metadata_path.suffix + ".lock")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


@contextmanager
def _cache_build_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + AMLSIM_CACHE_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "pid": int(os.getpid()),
                        "created_at": float(time.time()),
                    },
                    file,
                    ensure_ascii=False,
                )
            break
        except FileExistsError:
            try:
                age_seconds = time.time() - float(lock_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if age_seconds >= AMLSIM_CACHE_LOCK_STALE_SECONDS:
                _safe_unlink(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for AMLSim cache lock: {lock_path}")
            time.sleep(AMLSIM_CACHE_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        _safe_unlink(lock_path)


def _load_cached_graph(
    *,
    signature: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
    if not graph_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if dict(metadata.get("cache_signature", {})) != signature:
            return None
        graph = dgl.load_graphs(str(graph_path))[0][0]
        return graph, metadata
    except Exception:
        _safe_unlink(metadata_path)
        _safe_unlink(graph_path)
        return None


def _write_cache(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_token = f"{os.getpid()}.{time.time_ns()}"
    temp_graph_path = graph_path.with_suffix(graph_path.suffix + f".{temp_token}.tmp")
    try:
        dgl.save_graphs(str(temp_graph_path), [graph])
        os.replace(temp_graph_path, graph_path)
        atomic_write_json(metadata_path, metadata)
    finally:
        _safe_unlink(temp_graph_path)


def _build_graph_payload(
    *,
    source_info: dict[str, Any],
    signature: dict[str, Any],
    train_ratio: float,
    valid_ratio: float,
    relation_window_neighbors: int,
    activity_bins: int,
    event_history_len: int,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]]:
    raw_accounts = _read_csv(source_info["files"]["accounts"])
    raw_transactions = _read_csv(source_info["files"]["transactions"])
    accounts = _canonicalize_accounts(raw_accounts)
    transactions = _canonicalize_transactions(raw_transactions)
    accounts = _add_missing_accounts(accounts, transactions)
    positive_account_ids, label_source = _infer_positive_account_ids(files=source_info["files"], raw_accounts=raw_accounts)

    cash_aggregates = pd.DataFrame(index=accounts["account_id"])
    cash_transaction_rows = 0
    if "cash_transactions" in source_info["files"]:
        cash_frame = _canonicalize_cash_transactions(_read_csv(source_info["files"]["cash_transactions"]))
        if not cash_frame.empty:
            cash_transaction_rows = int(len(cash_frame))
            cash_aggregates = (
                cash_frame.groupby("account_id")
                .agg(cash_tx_count=("amount", "size"), cash_tx_amount_sum=("amount", "sum"), cash_tx_amount_mean=("amount", "mean"))
                .astype(np.float32)
            )

    account_to_index = {account_id: index for index, account_id in enumerate(accounts["account_id"].tolist())}
    transactions = transactions[
        transactions["orig_acct"].isin(account_to_index) & transactions["bene_acct"].isin(account_to_index)
    ].copy()
    transactions["orig_idx"] = transactions["orig_acct"].map(account_to_index).astype(np.int64)
    transactions["bene_idx"] = transactions["bene_acct"].map(account_to_index).astype(np.int64)

    out_agg = transactions.groupby("orig_acct").agg(
        out_tx_count=("amount", "size"),
        out_amount_sum=("amount", "sum"),
        out_amount_mean=("amount", "mean"),
        out_amount_max=("amount", "max"),
        out_unique_partners=("bene_acct", "nunique"),
    )
    in_agg = transactions.groupby("bene_acct").agg(
        in_tx_count=("amount", "size"),
        in_amount_sum=("amount", "sum"),
        in_amount_mean=("amount", "mean"),
        in_amount_max=("amount", "max"),
        in_unique_partners=("orig_acct", "nunique"),
    )
    long_activity = pd.concat(
        [
            transactions[["orig_acct", "timestamp"]].rename(columns={"orig_acct": "account_id"}),
            transactions[["bene_acct", "timestamp"]].rename(columns={"bene_acct": "account_id"}),
        ],
        axis=0,
        ignore_index=True,
    )
    activity_agg = long_activity.groupby("account_id").agg(
        first_activity=("timestamp", "min"),
        last_activity=("timestamp", "max"),
        total_tx_count=("timestamp", "size"),
    )

    top_tx_types = transactions["tx_type"].value_counts().head(4).index.tolist()
    tx_type_counts = (
        pd.concat(
            [
                transactions[["orig_acct", "tx_type"]].rename(columns={"orig_acct": "account_id"}),
                transactions[["bene_acct", "tx_type"]].rename(columns={"bene_acct": "account_id"}),
            ],
            axis=0,
            ignore_index=True,
        )
        .assign(value=1.0)
        .pivot_table(index="account_id", columns="tx_type", values="value", aggfunc="sum", fill_value=0.0)
    )
    tx_type_counts = tx_type_counts[[column for column in top_tx_types if column in tx_type_counts.columns]]
    tx_type_counts.columns = [f"tx_type_count::{column}" for column in tx_type_counts.columns]

    account_frame = accounts.set_index("account_id").copy()
    account_frame["bank_frequency"] = account_frame["bank_id"].map(account_frame["bank_id"].value_counts(normalize=True))
    account_frame["country_frequency"] = account_frame["country"].map(account_frame["country"].value_counts(normalize=True))
    account_frame["business_frequency"] = account_frame["business_type"].map(
        account_frame["business_type"].value_counts(normalize=True)
    )
    account_frame = account_frame.join(out_agg, how="left").join(in_agg, how="left").join(activity_agg, how="left")
    account_frame = account_frame.join(tx_type_counts, how="left").join(cash_aggregates, how="left")
    account_frame = account_frame.fillna(0.0)
    account_frame["total_amount_sum"] = account_frame["out_amount_sum"] + account_frame["in_amount_sum"]
    account_frame["net_amount_flow"] = account_frame["in_amount_sum"] - account_frame["out_amount_sum"]
    account_frame["total_unique_partners"] = account_frame["out_unique_partners"] + account_frame["in_unique_partners"]
    account_frame["activity_span"] = (account_frame["last_activity"] - account_frame["first_activity"]).clip(lower=0.0)
    account_frame["account_span"] = (account_frame["end_time"] - account_frame["start_time"]).clip(lower=0.0)
    account_frame["tx_density"] = account_frame["total_tx_count"] / (1.0 + account_frame["activity_span"])
    account_frame["log_total_tx_count"] = np.log1p(account_frame["total_tx_count"].to_numpy(dtype=np.float32))
    labels = account_frame.index.to_series().map(lambda account_id: 1 if account_id in positive_account_ids else 0).to_numpy(
        dtype=np.int64
    )

    timeline = account_frame["first_activity"].to_numpy(dtype=np.float32, copy=True)
    fallback_timeline = account_frame["start_time"].to_numpy(dtype=np.float32, copy=True)
    missing_timeline_mask = timeline <= 0
    timeline[missing_timeline_mask] = fallback_timeline[missing_timeline_mask]
    finite_timeline = timeline[np.isfinite(timeline)]
    timeline_signal_span = float(np.ptp(finite_timeline)) if finite_timeline.size > 0 else 0.0
    force_class_aware_split = timeline_signal_span <= 1e-6
    timeline = timeline + np.arange(len(account_frame), dtype=np.float32) * 1e-6
    train_mask, valid_mask, test_mask = _chronological_class_aware_split(
        timeline,
        labels,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        force_class_aware=force_class_aware_split,
    )
    train_mask_np = train_mask.cpu().numpy().astype(bool)

    feature_frame = account_frame[
        [
            "initial_balance",
            "start_time",
            "end_time",
            "account_span",
            "bank_frequency",
            "country_frequency",
            "business_frequency",
            "out_tx_count",
            "in_tx_count",
            "total_tx_count",
            "out_amount_sum",
            "in_amount_sum",
            "total_amount_sum",
            "net_amount_flow",
            "out_amount_mean",
            "in_amount_mean",
            "out_amount_max",
            "in_amount_max",
            "out_unique_partners",
            "in_unique_partners",
            "total_unique_partners",
            "first_activity",
            "last_activity",
            "activity_span",
            "tx_density",
            "log_total_tx_count",
            *[column for column in account_frame.columns if column.startswith("tx_type_count::")],
            *[column for column in account_frame.columns if column.startswith("cash_tx_")],
        ]
    ].copy()
    feature_matrix, feature_columns = _fit_feature_preprocessor(feature_frame, train_mask=train_mask_np)

    pair_frame = transactions[["orig_idx", "bene_idx"]].drop_duplicates(ignore_index=True)
    transfer_src = pair_frame["orig_idx"].to_numpy(dtype=np.int64)
    transfer_dst = pair_frame["bene_idx"].to_numpy(dtype=np.int64)
    reverse_src = transfer_dst.copy()
    reverse_dst = transfer_src.copy()

    sort_order = np.argsort(timeline, kind="mergesort").astype(np.int64)
    temporal_src, temporal_dst = _temporal_near_edges(sort_order, relation_window_neighbors)
    activity_codes = _activity_bin_codes(account_frame["log_total_tx_count"].to_numpy(dtype=np.float32), activity_bins)
    activity_src, activity_dst = _value_relation_edges(activity_codes, relation_window_neighbors)
    bank_src, bank_dst = _value_relation_edges(account_frame["bank_id"], relation_window_neighbors)
    country_src, country_dst = _value_relation_edges(account_frame["country"], relation_window_neighbors)
    business_src, business_dst = _value_relation_edges(account_frame["business_type"], relation_window_neighbors)

    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    relation_edge_counts: dict[str, int] = {}
    for relation_name, src, dst in (
        ("transfer", transfer_src, transfer_dst),
        ("transfer_reverse", reverse_src, reverse_dst),
        ("temporal_near", temporal_src, temporal_dst),
        ("activity_bin", activity_src, activity_dst),
        ("same_bank", bank_src, bank_dst),
        ("same_country", country_src, country_dst),
        ("same_business", business_src, business_dst),
    ):
        if len(src) == 0:
            continue
        edge_dict[(NODE_TYPE, relation_name, NODE_TYPE)] = (
            torch.from_numpy(src.astype(np.int64)),
            torch.from_numpy(dst.astype(np.int64)),
        )
        relation_edge_counts[relation_name] = int(len(src))

    homo_src_parts = [item[0].cpu().numpy() for key, item in edge_dict.items() if key[1] != "homo"]
    homo_dst_parts = [item[1].cpu().numpy() for key, item in edge_dict.items() if key[1] != "homo"]
    if homo_src_parts:
        homo_src = np.concatenate(homo_src_parts, axis=0).astype(np.int64)
        homo_dst = np.concatenate(homo_dst_parts, axis=0).astype(np.int64)
    else:
        ordered_nodes = np.arange(len(account_frame), dtype=np.int64)
        homo_src = np.concatenate([ordered_nodes[:-1], ordered_nodes[1:]], axis=0)
        homo_dst = np.concatenate([ordered_nodes[1:], ordered_nodes[:-1]], axis=0)
    edge_dict[(NODE_TYPE, "homo", NODE_TYPE)] = (
        torch.from_numpy(homo_src.astype(np.int64)),
        torch.from_numpy(homo_dst.astype(np.int64)),
    )
    relation_edge_counts["homo"] = int(len(homo_src))

    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(account_frame)})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix.astype(np.float32))
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(labels.astype(np.int64))
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time"] = torch.from_numpy(timeline.astype(np.float32))
    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.bool().clone()
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask.bool())
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full((len(account_frame),), 1.0, dtype=torch.float32)
    _attach_dataset_context_defaults(graph, dataset_name="amlsim")

    amount_scale = float(np.log1p(max(float(transactions["amount"].max()), 1.0)))
    time_min = float(transactions["timestamp"].min()) if len(transactions) > 0 else 0.0
    time_span = max(float(transactions["timestamp"].max()) - time_min, 1.0) if len(transactions) > 0 else 1.0
    total_tx_count_by_index = account_frame["total_tx_count"].to_numpy(dtype=np.float32)
    total_tx_count_scale = max(float(np.log1p(total_tx_count_by_index.max())), 1.0)
    bank_lookup = account_frame["bank_id"].to_dict()
    country_lookup = account_frame["country"].to_dict()
    top_tx_type_ids = {tx_type: index + 1 for index, tx_type in enumerate(sorted(transactions["tx_type"].unique().tolist()))}
    long_events = pd.concat(
        [
            transactions[["orig_acct", "bene_acct", "amount", "timestamp", "tx_type"]]
            .rename(columns={"orig_acct": "account_id", "bene_acct": "counterparty_id"})
            .assign(direction_out=1.0, direction_in=0.0),
            transactions[["bene_acct", "orig_acct", "amount", "timestamp", "tx_type"]]
            .rename(columns={"bene_acct": "account_id", "orig_acct": "counterparty_id"})
            .assign(direction_out=0.0, direction_in=1.0),
        ],
        axis=0,
        ignore_index=True,
    )
    long_events["same_bank"] = (
        long_events["account_id"].map(bank_lookup).fillna("unknown_bank")
        == long_events["counterparty_id"].map(bank_lookup).fillna("unknown_bank")
    ).astype(np.float32)
    long_events["same_country"] = (
        long_events["account_id"].map(country_lookup).fillna("unknown_country")
        == long_events["counterparty_id"].map(country_lookup).fillna("unknown_country")
    ).astype(np.float32)
    counterparty_lookup = {account_id: float(total_tx_count_by_index[index]) for index, account_id in enumerate(account_frame.index.tolist())}
    long_events["counterparty_activity"] = long_events["counterparty_id"].map(counterparty_lookup).fillna(0.0)

    event_sequence = torch.zeros((len(account_frame), event_history_len, 8), dtype=torch.float32)
    event_mask = torch.zeros((len(account_frame), event_history_len), dtype=torch.bool)
    event_time_deltas = torch.zeros((len(account_frame), event_history_len), dtype=torch.float32)
    event_token_weights = torch.zeros((len(account_frame), event_history_len), dtype=torch.float32)
    event_token_types = torch.zeros((len(account_frame), event_history_len), dtype=torch.long)
    event_source_ids = torch.zeros((len(account_frame), event_history_len), dtype=torch.long)
    for account_id, group in long_events.groupby("account_id", sort=False):
        if account_id not in account_to_index:
            continue
        node_index = int(account_to_index[account_id])
        group = group.sort_values(["timestamp"], kind="mergesort").tail(event_history_len).reset_index(drop=True)
        valid_length = int(len(group))
        if valid_length <= 0:
            continue
        insert_start = event_history_len - valid_length
        reference_time = float(group["timestamp"].iloc[-1])
        features = np.stack(
            [
                group["direction_out"].to_numpy(dtype=np.float32),
                group["direction_in"].to_numpy(dtype=np.float32),
                np.log1p(group["amount"].to_numpy(dtype=np.float32)) / max(amount_scale, 1e-6),
                ((group["timestamp"].to_numpy(dtype=np.float32) - time_min) / max(time_span, 1e-6)).astype(np.float32),
                group["same_bank"].to_numpy(dtype=np.float32),
                group["same_country"].to_numpy(dtype=np.float32),
                (np.log1p(group["counterparty_activity"].to_numpy(dtype=np.float32)) / total_tx_count_scale).astype(np.float32),
                np.asarray([top_tx_type_ids.get(value, 0) for value in group["tx_type"].tolist()], dtype=np.float32)
                / max(float(len(top_tx_type_ids)), 1.0),
            ],
            axis=1,
        ).astype(np.float32)
        deltas = np.log1p(np.clip(reference_time - group["timestamp"].to_numpy(dtype=np.float32), a_min=0.0, a_max=None))
        event_sequence[node_index, insert_start:] = torch.from_numpy(features)
        event_mask[node_index, insert_start:] = True
        event_time_deltas[node_index, insert_start:] = torch.from_numpy(deltas.astype(np.float32))
        event_token_weights[node_index, insert_start:] = 1.0 / (1.0 + torch.from_numpy(deltas.astype(np.float32)))
        event_token_types[node_index, insert_start : event_history_len - 1] = 1
        event_token_types[node_index, event_history_len - 1] = 2
        event_source_ids[node_index, insert_start:] = torch.tensor(
            [top_tx_type_ids.get(value, 0) for value in group["tx_type"].tolist()],
            dtype=torch.long,
        )
    graph.nodes[NODE_TYPE].data["event_sequence"] = event_sequence.float()
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()
    graph.nodes[NODE_TYPE].data["event_source_ids"] = event_source_ids.long()

    homo_src_nodes, homo_dst_nodes = graph.edges(etype="homo")
    homo_edge_labels = torch.where(
        graph.nodes[NODE_TYPE].data["label"][homo_src_nodes] == graph.nodes[NODE_TYPE].data["label"][homo_dst_nodes],
        torch.ones_like(homo_src_nodes, dtype=torch.float32),
        -torch.ones_like(homo_src_nodes, dtype=torch.float32),
    )
    graph.edges["homo"].data["label"] = homo_edge_labels
    graph.edges["homo"].data["train_mask"] = (
        graph.nodes[NODE_TYPE].data["train_mask"][homo_src_nodes] & graph.nodes[NODE_TYPE].data["train_mask"][homo_dst_nodes]
    ).bool()
    relation_order = _attach_relation_sequence(graph, dataset_name="amlsim")

    metadata = {
        "cache_signature": signature,
        "data_summary": {
            "dataset_display_name": AMLSIM_PUBLIC_NAME,
            "resolved_root": str(source_info["resolved_root"]),
            "requested_root": str(source_info["requested_root"]),
            "source_kind": str(source_info["source_kind"]),
            "sample_fallback_used": bool(source_info["sample_fallback_used"]),
            "label_source": str(label_source),
            "num_nodes": int(graph.num_nodes(NODE_TYPE)),
            "feature_dim": int(feature_matrix.shape[1]),
            "feature_columns": feature_columns,
            "relation_columns_used": list(relation_order),
            "relation_edge_counts": relation_edge_counts,
            "train_nodes": int(train_mask.sum().item()),
            "valid_nodes": int(valid_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
            "positive_labels": int(labels.sum()),
            "positive_ratio": float(labels.mean()),
            "train_positive": int(labels[train_mask_np].sum()),
            "valid_positive": int(labels[valid_mask.cpu().numpy().astype(bool)].sum()),
            "test_positive": int(labels[test_mask.cpu().numpy().astype(bool)].sum()),
            "train_ratio": float(train_ratio),
            "valid_ratio": float(valid_ratio),
            "test_ratio": float(max(1.0 - train_ratio - valid_ratio, 0.0)),
            "split_strategy": "class_aware_fallback" if force_class_aware_split else "chronological",
            "timeline_signal_span": float(timeline_signal_span),
            "event_history_len": int(event_history_len),
            "activity_bins": int(activity_bins),
            "relation_window_neighbors": int(relation_window_neighbors),
            "graph_builder_version": AMLSIM_GRAPH_BUILDER_VERSION,
            "transaction_rows": int(len(transactions)),
            "cash_transaction_rows": int(cash_transaction_rows),
            "transaction_time_min": float(transactions["timestamp"].min()) if len(transactions) > 0 else 0.0,
            "transaction_time_max": float(transactions["timestamp"].max()) if len(transactions) > 0 else 0.0,
        },
    }
    return graph, metadata


def _reset_runtime_masks(graph: dgl.DGLHeteroGraph) -> None:
    train_mask = graph.nodes[NODE_TYPE].data["train_mask"].bool()
    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.clone()
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask)
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full((graph.num_nodes(NODE_TYPE),), 1.0, dtype=torch.float32)
    if "homo" in graph.etypes:
        src, dst = graph.edges(etype="homo")
        graph.edges["homo"].data["train_mask"] = (train_mask[src] & train_mask[dst]).bool()


def load_amlsim_dataset(
    *,
    data_root: str | Path = AMLSIM_DEFAULT_ROOT,
    dataset_name: str = "amlsim",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    train_ratio: float = AMLSIM_DEFAULT_TRAIN_RATIO,
    valid_ratio: float = AMLSIM_DEFAULT_VALID_RATIO,
    relation_window_neighbors: int = AMLSIM_DEFAULT_RELATION_WINDOW_NEIGHBORS,
    activity_bins: int = AMLSIM_DEFAULT_ACTIVITY_BINS,
    event_history_len: int = AMLSIM_DEFAULT_EVENT_HISTORY_LEN,
    rebuild_cache: bool = False,
    allow_sample_fallback: bool = False,
) -> DatasetBundle:
    source_info = _resolve_amlsim_source(data_root, allow_sample_fallback=allow_sample_fallback)
    signature = _cache_signature(
        source_info=source_info,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        relation_window_neighbors=relation_window_neighbors,
        activity_bins=activity_bins,
        event_history_len=event_history_len,
    )
    cache_graph_path, cache_metadata_path = _resolve_cache_paths(signature)
    lock_path = _cache_lock_path(cache_metadata_path)
    cached = None
    if not rebuild_cache:
        cached = _load_cached_graph(signature=signature, graph_path=cache_graph_path, metadata_path=cache_metadata_path)
    if cached is None:
        with _cache_build_lock(lock_path):
            if rebuild_cache:
                _safe_unlink(cache_graph_path)
                _safe_unlink(cache_metadata_path)
                cached_after_lock = None
            else:
                cached_after_lock = _load_cached_graph(
                    signature=signature,
                    graph_path=cache_graph_path,
                    metadata_path=cache_metadata_path,
                )
            if cached_after_lock is None:
                graph, metadata = _build_graph_payload(
                    source_info=source_info,
                    signature=signature,
                    train_ratio=train_ratio,
                    valid_ratio=valid_ratio,
                    relation_window_neighbors=relation_window_neighbors,
                    activity_bins=activity_bins,
                    event_history_len=event_history_len,
                )
                _write_cache(graph=graph, metadata=metadata, graph_path=cache_graph_path, metadata_path=cache_metadata_path)
            else:
                graph, metadata = cached_after_lock
    else:
        graph, metadata = cached

    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)
    _reset_runtime_masks(graph)
    if float(label_fraction) < 0.999:
        _apply_label_scarcity(graph, label_fraction=float(label_fraction), seed=seed)
    if active_learning_feedback_path:
        _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)
    relation_order = list(dict.fromkeys(metadata.get("data_summary", {}).get("relation_columns_used", []) or list(graph.etypes)))
    if (
        "sequence" not in graph.nodes[NODE_TYPE].data
        or "sequence_token_weights" not in graph.nodes[NODE_TYPE].data
        or "sequence_token_types" not in graph.nodes[NODE_TYPE].data
        or "sequence_relation_ids" not in graph.nodes[NODE_TYPE].data
    ):
        relation_order = _attach_relation_sequence(graph, dataset_name=dataset_name)

    train_supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool() & graph.nodes[NODE_TYPE].data["train_mask"].bool()
    train_unlabeled_mask = graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool() & graph.nodes[NODE_TYPE].data["train_mask"].bool()
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
    if class_labels.numel() == 0:
        class_counts = torch.ones(2, dtype=torch.float32)
    else:
        class_counts = torch.bincount(class_labels.long(), minlength=2).float().clamp(min=1.0)
    class_weights = class_counts.sum() / (class_counts * len(class_counts))

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
    bundle.data_summary = data_summary
    return bundle
