from __future__ import annotations

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
