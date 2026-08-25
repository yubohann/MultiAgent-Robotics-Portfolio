"""Smoke test helpers for the single-agent experiment."""

from __future__ import annotations

from single_gate.training import run_training


def run_single_smoke_test() -> dict[str, object]:
    """Run a very small end-to-end training smoke test."""

    return run_training(
        train_steps=48,
        seed=7,
        device="cpu",
        save_dir=None,
        learning_starts=8,
        batch_size=8,
        updates_per_step=1,
        log_every=24,
    )

