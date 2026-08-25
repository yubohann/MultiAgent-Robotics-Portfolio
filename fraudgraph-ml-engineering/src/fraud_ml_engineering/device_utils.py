from __future__ import annotations

"""Shared device defaults and safe CUDA resolution helpers."""

import torch

DEFAULT_DEVICE_REQUEST = "cuda"

try:
    import dgl
except Exception:  # pragma: no cover - runtime env dependent
    dgl = None


def resolve_torch_device(device_name: str, *, warn: bool = True) -> torch.device:
    normalized = str(device_name or DEFAULT_DEVICE_REQUEST).strip().lower()
    if not normalized.startswith("cuda"):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if warn:
            print("[WARN] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(normalized if normalized else DEFAULT_DEVICE_REQUEST)


def resolve_dgl_training_device(device_name: str, *, warn: bool = True) -> torch.device:
    resolved = resolve_torch_device(device_name, warn=warn)
    if resolved.type != "cuda":
        return resolved
    if dgl is None:
        if warn:
            print("[WARN] CUDA requested but DGL is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    try:
        probe_graph = dgl.graph((torch.tensor([0]), torch.tensor([0])))
        probe_graph = probe_graph.to(str(resolved))
        del probe_graph
    except Exception as error:
        if warn:
            print(f"[WARN] CUDA requested but DGL CUDA backend is unavailable ({error}). Falling back to CPU.")
        return torch.device("cpu")
    return resolved
