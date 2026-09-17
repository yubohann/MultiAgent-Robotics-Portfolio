from __future__ import annotations

import os
from typing import Any

import torch

IEEE_FULL_SAFE_CPU_THREADS = 8
IEEE_FULL_SAFE_INTEROP_THREADS = 2
_IEEE_FULL_RUNTIME_CONFIGURED = False


def ieee_full_gpu_cuda_ready(args: Any) -> bool:
    requested = str(getattr(args, "device", "") or "").strip().lower()
    if requested:
        return requested.startswith("cuda") and torch.cuda.is_available()
    return torch.cuda.is_available()


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
    except (RuntimeError, ValueError) as error:  # pragma: no cover - runtime dependent
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
        except (RuntimeError, ValueError) as error:  # pragma: no cover - runtime dependent
            interop_note = f"set_num_interop_threads_failed:{type(error).__name__}"
        _IEEE_FULL_RUNTIME_CONFIGURED = True

    tf32_enabled = False
    if torch.cuda.is_available():
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except (RuntimeError, ValueError):  # pragma: no cover - runtime dependent
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
