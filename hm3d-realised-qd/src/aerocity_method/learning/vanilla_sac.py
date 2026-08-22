"""Strict masked discrete SAC baseline for shared HM3D candidate sets."""

from __future__ import annotations

from dataclasses import replace

from aerocity_method.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig


def vanilla_sac_config(
    *,
    context_dim: int,
    candidate_dim: int,
    hidden_dim: int = 64,
    gamma: float = 0.99,
    device: str = "cpu",
) -> RBSFSACConfig:
    """Return a strict no-SF/no-cost/no-preference SAC configuration."""

    return RBSFSACConfig(
        context_dim=context_dim,
        candidate_dim=candidate_dim,
        preference_dim=0,
        sf_dim=0,
        hidden_dim=hidden_dim,
        gamma=gamma,
        cost_weight=0.0,
        cost_limit=None,
        enable_cost_critics=False,
        device=device,
    )


class VanillaMaskedDiscreteSAC(RBSFSAC):
    """Compatibility wrapper proving the baseline is not RBSFSAC with a label."""

    def __init__(self, config: RBSFSACConfig, *, seed: int = 0) -> None:
        strict = replace(
            config,
            preference_dim=0,
            sf_dim=0,
            cost_weight=0.0,
            cost_limit=None,
            enable_cost_critics=False,
        )
        super().__init__(strict, seed=seed)


__all__ = ["VanillaMaskedDiscreteSAC", "vanilla_sac_config"]
