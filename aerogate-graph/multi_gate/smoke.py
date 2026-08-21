"""Smoke test helpers for the multi-agent experiment."""

from __future__ import annotations

from multi_gate.training import run_training


def run_multi_smoke_test() -> dict[str, object]:
    """Run a very small end-to-end multi-agent training smoke test."""

    return run_training(
        train_steps=48,
        seed=13,
        device="cpu",
        save_dir=None,
        num_agents=4,
        learning_starts=8,
        batch_size=8,
        updates_per_step=1,
        log_every=24,
    )

