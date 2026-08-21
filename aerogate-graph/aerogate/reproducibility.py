"""Deterministic rollout checks for lightweight, repeatable research validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .scenarios import RolloutSummary, ScenarioName, run_rollout

DEFAULT_REPRODUCIBILITY_SEEDS: tuple[int, ...] = (3, 7, 11)


@dataclass(frozen=True)
class ReproducibilityReport:
    """Evidence that repeated seeded core rollouts produce identical diagnostics."""

    scenario: ScenarioName
    seeds: tuple[int, ...]
    steps: int
    agents: int | None
    deterministic: bool
    rollouts: tuple[RolloutSummary, ...]
    mismatched_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a portable report suitable for a CI artifact or experiment record."""

        return {
            "scenario": self.scenario,
            "seeds": list(self.seeds),
            "steps": self.steps,
            "agents": self.agents,
            "deterministic": self.deterministic,
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
            "mismatched_seeds": list(self.mismatched_seeds),
        }


def verify_reproducibility(
    name: ScenarioName,
    *,
    seeds: Sequence[int] = DEFAULT_REPRODUCIBILITY_SEEDS,
    steps: int = 8,
    agents: int | None = None,
) -> ReproducibilityReport:
    """Run each seeded rollout twice and compare all public diagnostics.

    The check deliberately covers the same NumPy-only environment path used in the
    quick-start commands. It does not claim deterministic training across GPU runtimes.
    """

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("at least one seed is required for a reproducibility check")

    rollouts: list[RolloutSummary] = []
    mismatched_seeds: list[int] = []
    for seed in normalized_seeds:
        first = run_rollout(name, agents=agents, seed=seed, steps=steps)
        repeated = run_rollout(name, agents=agents, seed=seed, steps=steps)
        rollouts.append(first)
        if first != repeated:
            mismatched_seeds.append(seed)

    return ReproducibilityReport(
        scenario=name,
        seeds=normalized_seeds,
        steps=steps,
        agents=agents,
        deterministic=not mismatched_seeds,
        rollouts=tuple(rollouts),
        mismatched_seeds=tuple(mismatched_seeds),
    )
