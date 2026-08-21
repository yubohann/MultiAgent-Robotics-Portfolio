"""Utilities for shutting down Isaac Sim without hanging the parent workflow."""

from __future__ import annotations

import threading


def close_simulation_app_with_timeout(
    simulation_app,
    *,
    timeout_s: float = 90.0,
    label: str = "simulation_app",
    wait_for_replicator: bool = True,
    skip_cleanup: bool = False,
) -> bool:
    """Close Isaac Sim in a daemon thread and bail out if shutdown stalls.

    Returns ``True`` when the close finished within ``timeout_s`` and ``False``
    when the caller should continue because shutdown timed out.
    """
    if simulation_app is None:
        return True

    closed = threading.Event()

    def _close() -> None:
        try:
            try:
                simulation_app.close(
                    wait_for_replicator=wait_for_replicator,
                    skip_cleanup=skip_cleanup,
                )
            except TypeError:
                simulation_app.close()
        finally:
            closed.set()

    worker = threading.Thread(target=_close, daemon=True, name="sim-close")
    worker.start()
    if closed.wait(timeout=max(float(timeout_s), 0.0)):
        return True

    print(
        f"[Warning] {label}.close() did not finish within {float(timeout_s):.0f}s; continuing shutdown.",
        flush=True,
    )
    return False
