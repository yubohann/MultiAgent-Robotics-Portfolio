"""Planner baseline groups for experiment-2 scripts."""

from __future__ import annotations


_GROUPS: dict[str, tuple[str, ...]] = {
    "planner_only_main": ("straight", "detour"),
    "classic_planner_only": ("astar", "theta_star", "rrt_star", "informed_rrt_star", "heuristic"),
    "strong_planner_only": ("ego_planner", "fast_planner"),
    "all": ("straight", "detour", "astar", "theta_star", "rrt_star", "informed_rrt_star", "heuristic", "ego_planner", "fast_planner"),
}


def baseline_group_names() -> tuple[str, ...]:
    return tuple(_GROUPS)


def planners_for_group(name: str) -> tuple[str, ...]:
    normalized = str(name).strip()
    if normalized not in _GROUPS:
        raise KeyError(f"Unknown planner baseline group: {name}")
    return _GROUPS[normalized]

