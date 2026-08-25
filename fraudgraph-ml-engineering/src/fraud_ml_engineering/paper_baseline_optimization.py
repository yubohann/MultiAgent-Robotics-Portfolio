from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .paper_runner_runtime import ensure_cache_dir, stable_cache_key

IEEE_FULL_SAFE_CPU_THREADS = 8
IEEE_FULL_SAFE_INTEROP_THREADS = 2
_IEEE_FULL_RUNTIME_CONFIGURED = False


def dataset_max_transactions(dataset_name: str, args: Any) -> int | None:
    normalized = str(dataset_name).strip().lower()
    mapping = {
        "ieee": "ieee_max_transactions",
        "archive": "archive_max_transactions",
        "ccfd": "ccfd_max_transactions",
        "ethereum_phishing": "ethereum_phishing_max_transactions",
    }
    field_name = mapping.get(normalized)
    if field_name is None:
        return None
    value = getattr(args, field_name, None)
    if value is None:
        return None
    resolved = int(value)
    return None if resolved <= 0 else resolved


def build_static_cache_paths(
    *,
    static_cache_dir: str | Path,
    namespace: str,
    dataset_name: str,
    seed: int,
    label_fraction: float,
    graph_builder_version: str,
    max_transactions: int | None,
    extra_key: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    payload: dict[str, Any] = {
        "dataset": str(dataset_name),
        "seed": int(seed),
        "max_transactions": None if max_transactions is None else int(max_transactions),
        "label_fraction": float(label_fraction),
        "graph_builder_version": str(graph_builder_version),
    }
    if extra_key:
        payload.update(extra_key)
    cache_dir = ensure_cache_dir(static_cache_dir, namespace)
    digest = stable_cache_key(payload)
    prefix = f"{dataset_name}_{digest}"
    return cache_dir / f"{prefix}.npz", cache_dir / f"{prefix}.json", payload


def should_run_eval(
    epoch: int,
    *,
    warmup_interval: int = 3,
    steady_interval: int = 1,
    every_epoch_after: int = 12,
) -> bool:
    epoch_id = int(epoch)
    if epoch_id <= 1:
        return True
    if epoch_id >= int(every_epoch_after):
        return epoch_id % max(int(steady_interval), 1) == 0
    return epoch_id % max(int(warmup_interval), 1) == 0


def dataloader_runtime_kwargs(
    *,
    device: torch.device,
    num_workers: int = 0,
) -> dict[str, Any]:
    worker_count = max(int(num_workers), 0)
    use_cuda_loader = device.type == "cuda"
    payload: dict[str, Any] = {
        "num_workers": worker_count,
        "pin_memory": bool(use_cuda_loader),
    }
    if worker_count > 0:
        payload["persistent_workers"] = True
    return payload


def ieee_full_gpu_cuda_ready(args: Any) -> bool:
    requested = str(getattr(args, "device", "") or "").strip().lower()
    if requested:
        return requested.startswith("cuda") and torch.cuda.is_available()
    return torch.cuda.is_available()


def ieee_runtime_uses_raw_assets(args: Any) -> bool:
    data_profile = str(getattr(args, "ieee_data_profile", "raw") or "raw").strip().lower()
    return data_profile == "raw"


def stabilize_ieee_full_runtime(
    *,
    enabled: bool,
    cpu_threads: int = IEEE_FULL_SAFE_CPU_THREADS,
    interop_threads: int = IEEE_FULL_SAFE_INTEROP_THREADS,
) -> dict[str, Any]:
    if not bool(enabled):
        return {
            "enabled": False,
            "cpu_threads": int(max(cpu_threads, 1)),
            "interop_threads": int(max(interop_threads, 1)),
            "tf32_enabled": False,
            "thread_note": "disabled",
            "interop_note": "disabled",
            "env": {},
        }

    global _IEEE_FULL_RUNTIME_CONFIGURED

    resolved_cpu_threads = max(int(cpu_threads), 1)
    resolved_interop_threads = max(int(interop_threads), 1)
    env_updates: dict[str, int] = {}
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        current_text = os.environ.get(env_name, "").strip()
        try:
            current_value = int(current_text) if current_text else None
        except ValueError:
            current_value = None
        if current_value is None or current_value > resolved_cpu_threads:
            os.environ[env_name] = str(resolved_cpu_threads)
            env_updates[env_name] = resolved_cpu_threads

    thread_note = "unchanged"
    try:
        current_threads = int(torch.get_num_threads())
        target_threads = min(current_threads, resolved_cpu_threads)
        if target_threads != current_threads:
            torch.set_num_threads(target_threads)
            thread_note = f"capped_to_{target_threads}"
        else:
            thread_note = f"kept_{current_threads}"
    except Exception as error:  # pragma: no cover - runtime dependent
        thread_note = f"set_num_threads_failed:{type(error).__name__}"

    interop_note = "already_configured"
    if not _IEEE_FULL_RUNTIME_CONFIGURED:
        try:
            current_interop = int(torch.get_num_interop_threads())
            target_interop = min(current_interop, resolved_interop_threads)
            if target_interop != current_interop:
                torch.set_num_interop_threads(target_interop)
                interop_note = f"capped_to_{target_interop}"
            else:
                interop_note = f"kept_{current_interop}"
        except Exception as error:  # pragma: no cover - runtime dependent
            interop_note = f"set_num_interop_threads_failed:{type(error).__name__}"
        _IEEE_FULL_RUNTIME_CONFIGURED = True

    tf32_enabled = False
    if torch.cuda.is_available():
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:  # pragma: no cover - runtime dependent
                pass
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        tf32_enabled = True

    return {
        "enabled": True,
        "cpu_threads": int(resolved_cpu_threads),
        "interop_threads": int(resolved_interop_threads),
        "tf32_enabled": bool(tf32_enabled),
        "thread_note": thread_note,
        "interop_note": interop_note,
        "env": env_updates,
    }


def apply_ieee_full_gpu_profile(
    args: Any,
    *,
    enabled: bool,
    floors: dict[str, int | float],
    schedule_overrides: dict[str, int | float] | None = None,
    exact_overrides: dict[str, Any] | None = None,
    ceilings: dict[str, int | float] | None = None,
) -> None:
    if not bool(enabled):
        return
    stabilize_ieee_full_runtime(enabled=True)
    effective_exact_overrides = dict(exact_overrides or {})
    if "ieee_max_transactions" in effective_exact_overrides:
        override_value = effective_exact_overrides.get("ieee_max_transactions")
        try:
            normalized_override = int(override_value) if override_value is not None else None
        except (TypeError, ValueError):
            normalized_override = override_value
        if normalized_override is None or (
            isinstance(normalized_override, int) and normalized_override <= 0 and (
                not ieee_full_gpu_cuda_ready(args) or not ieee_runtime_uses_raw_assets(args)
            )
        ):
            effective_exact_overrides.pop("ieee_max_transactions", None)
    for field_name, floor_value in floors.items():
        current_value = getattr(args, field_name, None)
        if current_value is None:
            continue
        if isinstance(floor_value, int):
            setattr(args, field_name, max(int(current_value), int(floor_value)))
        else:
            setattr(args, field_name, max(float(current_value), float(floor_value)))
    for field_name, ceiling_value in (ceilings or {}).items():
        current_value = getattr(args, field_name, None)
        if current_value is None:
            continue
        if isinstance(ceiling_value, int):
            setattr(args, field_name, min(int(current_value), int(ceiling_value)))
        else:
            setattr(args, field_name, min(float(current_value), float(ceiling_value)))
    for field_name, override_value in effective_exact_overrides.items():
        setattr(args, field_name, override_value)
    for field_name, override_value in (schedule_overrides or {}).items():
        setattr(args, field_name, override_value)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return str(value)
    return value


def collect_important_parameters(
    args: Any,
    field_names: list[str] | tuple[str, ...],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        str(field_name): _json_safe_value(getattr(args, str(field_name)))
        for field_name in field_names
        if hasattr(args, str(field_name))
    }
    if extra:
        for key, value in extra.items():
            payload[str(key)] = _json_safe_value(value)
    return payload


def format_parameter_snapshot(parameters: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in parameters.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        elif isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        elif value is None:
            parts.append(f"{key}=none")
        elif isinstance(value, (list, tuple)):
            joined = ",".join(str(item) for item in value)
            parts.append(f"{key}=[{joined}]")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def build_metric_summary(
    *,
    valid_metrics: Mapping[str, Any] | None = None,
    test_metrics: Mapping[str, Any] | None = None,
    valid_prefix: str = "best_valid",
    test_prefix: str = "test",
) -> dict[str, Any]:
    field_aliases = (
        ("acc", "acc"),
        ("auc", "auc"),
        ("pr_auc", "pr_auc"),
        ("recall", "recall"),
        ("precision", "precision"),
        ("f1_score", "f1"),
        ("f1_macro", "f1_macro"),
        ("gmean", "gmean"),
        ("recall_at_precision", "recall_at_precision"),
        ("threshold", "threshold"),
    )
    summary: dict[str, Any] = {}
    if valid_metrics is not None:
        summary[f"{valid_prefix}_metrics"] = _json_safe_value(dict(valid_metrics))
        for metric_name, alias in field_aliases:
            if metric_name in valid_metrics and valid_metrics[metric_name] is not None:
                summary[f"{valid_prefix}_{alias}"] = float(valid_metrics[metric_name])
    if test_metrics is not None:
        summary[f"{test_prefix}_metrics"] = _json_safe_value(dict(test_metrics))
        for metric_name, alias in field_aliases:
            if metric_name in test_metrics and test_metrics[metric_name] is not None:
                summary[f"{test_prefix}_{alias}"] = float(test_metrics[metric_name])
    return summary


def metric_markdown_header(label_name: str = "Split") -> tuple[str, str]:
    return (
        f"| {label_name} | ACC | Recall | Precision | AUC | PR-AUC | F1 | F1-macro | GMean | Recall@Precision | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    )


def metric_markdown_row(label: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| {label} | "
        f"{float(metrics['acc']):.6f} | "
        f"{float(metrics['recall']):.6f} | "
        f"{float(metrics['precision']):.6f} | "
        f"{float(metrics['auc']):.6f} | "
        f"{float(metrics['pr_auc']):.6f} | "
        f"{float(metrics['f1_score']):.6f} | "
        f"{float(metrics['f1_macro']):.6f} | "
        f"{float(metrics['gmean']):.6f} | "
        f"{float(metrics['recall_at_precision']):.6f} | "
        f"{float(metrics['threshold']):.6f} |"
    )
