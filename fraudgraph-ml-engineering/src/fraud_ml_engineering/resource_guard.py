from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

BYTES_PER_GIB = float(1024**3)
DEFAULT_CUDA_UTILIZATION_BUDGET = 0.85
ETHEREUM_PHISHING_SAFE_FULL_MAX_USERS = 16000
ETHEREUM_PHISHING_SAFE_FULL_MAX_TRANSACTIONS = 200000
ETHEREUM_PHISHING_LITE_MAX_USERS = 12000
ETHEREUM_PHISHING_LITE_MAX_TRANSACTIONS = 150000
ETHEREUM_PHISHING_MIN_AUTO_MAX_USERS = 4000
ETHEREUM_PHISHING_MIN_AUTO_MAX_TRANSACTIONS = 50000
DEFAULT_ONCHAIN_NODE_BUDGETS = {
    "ethereum_phishing": ETHEREUM_PHISHING_SAFE_FULL_MAX_USERS,
    "ethereum_ponzi": 6000,
    "defi_rug_pull": 5000,
}
DEFAULT_ONCHAIN_TRANSACTION_BUDGETS = {
    "ethereum_phishing": ETHEREUM_PHISHING_SAFE_FULL_MAX_TRANSACTIONS,
    "ethereum_ponzi": 80000,
    "defi_rug_pull": 60000,
}


def _normalize_device_index(device: Any) -> int:
    text = str(device or "").strip().lower()
    if ":" in text:
        _, _, suffix = text.partition(":")
        try:
            return max(int(suffix), 0)
        except ValueError:
            return 0
    return 0


def resolve_device_budget_gib(
    device: Any,
    *,
    requested_budget_gib: float | None = None,
    utilization_budget: float = DEFAULT_CUDA_UTILIZATION_BUDGET,
) -> float | None:
    if requested_budget_gib is not None:
        return float(max(requested_budget_gib, 0.0))
    if not torch.cuda.is_available() or not str(device).lower().startswith("cuda"):
        return None
    device_index = _normalize_device_index(device)
    properties = torch.cuda.get_device_properties(device_index)
    return float(properties.total_memory) * float(utilization_budget) / BYTES_PER_GIB


def safe_ethereum_phishing_limits(
    *,
    lite: bool,
    requested_max_users: int | None,
    requested_max_transactions: int | None,
) -> tuple[int, int, list[str]]:
    safe_users = (
        ETHEREUM_PHISHING_LITE_MAX_USERS if lite else ETHEREUM_PHISHING_SAFE_FULL_MAX_USERS
    )
    safe_transactions = (
        ETHEREUM_PHISHING_LITE_MAX_TRANSACTIONS if lite else ETHEREUM_PHISHING_SAFE_FULL_MAX_TRANSACTIONS
    )
    notes: list[str] = []
    if requested_max_users is None:
        max_users = safe_users
        notes.append(f"Applied default max_users={safe_users} for ethereum_phishing.")
    else:
        max_users = int(requested_max_users)
    if requested_max_transactions is None:
        max_transactions = safe_transactions
        notes.append(f"Applied default max_transactions={safe_transactions} for ethereum_phishing.")
    else:
        max_transactions = int(requested_max_transactions)
    return max_users, max_transactions, notes


def clamp_ethereum_phishing_to_lite(
    max_users: int | None,
    max_transactions: int | None,
) -> tuple[int, int]:
    resolved_users = ETHEREUM_PHISHING_LITE_MAX_USERS if max_users is None else min(int(max_users), ETHEREUM_PHISHING_LITE_MAX_USERS)
    resolved_transactions = (
        ETHEREUM_PHISHING_LITE_MAX_TRANSACTIONS
        if max_transactions is None
        else min(int(max_transactions), ETHEREUM_PHISHING_LITE_MAX_TRANSACTIONS)
    )
    return max(int(resolved_users), ETHEREUM_PHISHING_MIN_AUTO_MAX_USERS), max(
        int(resolved_transactions),
        ETHEREUM_PHISHING_MIN_AUTO_MAX_TRANSACTIONS,
    )


def estimate_onchain_candidate_vram_gib(
    dataset_name: str,
    profile: Mapping[str, Any],
    *,
    max_users: int | None = None,
    max_transactions: int | None = None,
) -> float:
    dataset_key = str(dataset_name).lower()
    nodes = int(
        max_users
        if max_users is not None
        else DEFAULT_ONCHAIN_NODE_BUDGETS.get(dataset_key, ETHEREUM_PHISHING_SAFE_FULL_MAX_USERS)
    )
    transactions = int(
        max_transactions
        if max_transactions is not None
        else DEFAULT_ONCHAIN_TRANSACTION_BUDGETS.get(dataset_key, ETHEREUM_PHISHING_SAFE_FULL_MAX_TRANSACTIONS)
    )
    hidden_dim = max(
        int(profile.get("transformer_hidden_dim", 64)),
        int(profile.get("fusion_hidden_dim", 64)),
        int(profile.get("seq_hidden_dim", 64)),
    )
    layers = max(int(profile.get("transformer_num_layers", 1)), 1)
    width_scale = float(hidden_dim) / 64.0
    layer_scale = 1.0 + 0.12 * float(layers - 1)
    graph_bytes = (nodes * 170_000.0) + (transactions * 12_000.0)
    fixed_overhead_bytes = 1.1 * BYTES_PER_GIB
    estimated_bytes = (fixed_overhead_bytes + graph_bytes) * width_scale * layer_scale
    return float(estimated_bytes / BYTES_PER_GIB)


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return 0


def _graph_runtime_bytes(graph: Any) -> int:
    total_bytes = 0
    for node_type in getattr(graph, "ntypes", []):
        for value in graph.nodes[node_type].data.values():
            total_bytes += _tensor_bytes(value)
        total_bytes += int(graph.num_nodes(node_type)) * 8
    for edge_type in getattr(graph, "canonical_etypes", []):
        for value in graph.edges[edge_type].data.values():
            total_bytes += _tensor_bytes(value)
        total_bytes += int(graph.num_edges(edge_type)) * 16
    return total_bytes


def _model_parameter_bytes(model: Any) -> int:
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in model.parameters())
    buffer_bytes = sum(_tensor_bytes(buffer) for buffer in model.buffers())
    return int(parameter_bytes + buffer_bytes)


def estimate_runtime_resource_plan(
    *,
    bundle: Any,
    model: Any,
    device: Any,
    requested_budget_gib: float | None = None,
    use_teacher_model: bool = False,
) -> dict[str, Any]:
    graph_bytes = _graph_runtime_bytes(bundle.graph)
    model_bytes = _model_parameter_bytes(model)
    graph_runtime_bytes = int(graph_bytes * 2.8)
    model_runtime_multiplier = 7.0 + (1.0 if use_teacher_model else 0.0)
    model_runtime_bytes = int(model_bytes * model_runtime_multiplier)
    overhead_bytes = int(0.25 * BYTES_PER_GIB)
    estimated_total_bytes = graph_runtime_bytes + model_runtime_bytes + overhead_bytes
    estimated_vram_gib = float(estimated_total_bytes / BYTES_PER_GIB)
    budget_gib = resolve_device_budget_gib(device, requested_budget_gib=requested_budget_gib)
    fits = True if budget_gib is None else estimated_vram_gib <= budget_gib
    return {
        "estimated_vram_gib": estimated_vram_gib,
        "budget_gib": budget_gib,
        "fits": bool(fits),
        "graph_runtime_gib": float(graph_runtime_bytes / BYTES_PER_GIB),
        "model_runtime_gib": float(model_runtime_bytes / BYTES_PER_GIB),
        "device": str(device),
        "teacher_enabled": bool(use_teacher_model),
    }


def recommend_smaller_phishing_limits(
    *,
    current_num_nodes: int,
    current_num_transactions: int,
    estimated_vram_gib: float,
    budget_gib: float,
) -> dict[str, int]:
    if budget_gib <= 0.0 or estimated_vram_gib <= 0.0:
        shrink_ratio = 0.5
    else:
        shrink_ratio = max(min((budget_gib / estimated_vram_gib) * 0.85, 0.95), 0.20)
    return {
        "max_users": max(int(current_num_nodes * shrink_ratio), ETHEREUM_PHISHING_MIN_AUTO_MAX_USERS),
        "max_transactions": max(
            int(current_num_transactions * shrink_ratio),
            ETHEREUM_PHISHING_MIN_AUTO_MAX_TRANSACTIONS,
        ),
    }
