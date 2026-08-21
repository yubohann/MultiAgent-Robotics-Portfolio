"""Experiment-2 closed-loop method ablation registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodVariant:
    name: str
    planner_name: str = "detour"
    use_planner: bool = True
    use_reactive: bool = True
    use_shield: bool = True
    use_uncertainty_arbitration: bool = True
    event_triggered_replanning: bool = True
    fixed_planner_weight: float | None = None


_VARIANTS: tuple[MethodVariant, ...] = (
    MethodVariant("reactive_only", use_planner=False, use_reactive=True, use_shield=False, use_uncertainty_arbitration=False, event_triggered_replanning=False),
    MethodVariant("planner_only", use_planner=True, use_reactive=False, use_shield=False, use_uncertainty_arbitration=False, event_triggered_replanning=False, fixed_planner_weight=1.0),
    MethodVariant("planner_reactive_fixed", use_planner=True, use_reactive=True, use_shield=False, use_uncertainty_arbitration=False, event_triggered_replanning=False, fixed_planner_weight=0.5),
    MethodVariant("w_o_safety_shield", use_planner=True, use_reactive=True, use_shield=False, use_uncertainty_arbitration=True, event_triggered_replanning=True),
    MethodVariant("w_o_event_triggered_replanning", use_planner=True, use_reactive=True, use_shield=True, use_uncertainty_arbitration=True, event_triggered_replanning=False),
    MethodVariant("w_o_uncertainty_arbitration", use_planner=True, use_reactive=True, use_shield=True, use_uncertainty_arbitration=False, event_triggered_replanning=True, fixed_planner_weight=0.65),
    MethodVariant("w_o_reactive_policy", use_planner=True, use_reactive=False, use_shield=True, use_uncertainty_arbitration=False, event_triggered_replanning=True, fixed_planner_weight=1.0),
    MethodVariant("ours_full", use_planner=True, use_reactive=True, use_shield=True, use_uncertainty_arbitration=True, event_triggered_replanning=True),
)


def method_variants() -> tuple[MethodVariant, ...]:
    return _VARIANTS


def variant_names() -> tuple[str, ...]:
    return tuple(variant.name for variant in _VARIANTS)


def get_variant(name: str) -> MethodVariant:
    normalized = str(name).strip()
    for variant in _VARIANTS:
        if variant.name == normalized:
            return variant
    raise KeyError(f"Unknown experiment-2 method variant: {name}")

