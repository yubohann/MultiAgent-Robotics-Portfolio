"""Legacy FlashSAC compatibility entry point for the single-agent task."""

from __future__ import annotations

from single_gate.graph_rl.graph_sac import (
    FlashSACAgent,
    GraphCritic,
    GraphEncoder,
    GraphFlashSACAgent,
    GraphSACAgent,
    GraphSACBuildContext,
    SquashedGaussianGraphActor,
)

__all__ = [
    "FlashSACAgent",
    "GraphCritic",
    "GraphEncoder",
    "GraphFlashSACAgent",
    "GraphSACAgent",
    "GraphSACBuildContext",
    "SquashedGaussianGraphActor",
]

