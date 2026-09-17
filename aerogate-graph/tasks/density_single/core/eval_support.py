"""Shared constants and small helpers for single-drone gate-density evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D
from shared.runtime.paths import ASSETS_ROOT
from tasks.density_single.core.gate_layout import (
    GATE_POST_RADIUS_M,
    GATE_TOP_HEIGHT_M,
    _gate_post_centers,
)

DRONE_RADIUS_M = 0.25
SAFETY_MARGIN_M = 0.15
SHIELD_GUARD_MARGIN_M = 0.0
GOAL_RADIUS_M = 0.50
MAX_EPISODE_STEPS = 420
MAX_GATE_COUNT = 60


def clamp01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_gate_obstacle_map(
    gate_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
    *,
    post_radius_m: float = GATE_POST_RADIUS_M,
) -> GateObstacleMap2D:
    obstacles = [
        GatePostObstacle2D(
            species="gate_post",
            center_xy=post_xy,
            collision_radius_m=float(post_radius_m),
            canopy_height_m=GATE_TOP_HEIGHT_M,
            description=f"gate_density_post_{post_idx:02d}",
            usd_path=str(ASSETS_ROOT / "gate" / "gate.usd"),
        )
        for post_idx, post_xy in enumerate(_gate_post_centers(gate_centers_xy, gate_yaws))
    ]
    return GateObstacleMap2D(tuple(obstacles))


def path_length(points_xy: list[tuple[float, float]]) -> float:
    return float(
        sum(
            math.hypot(points_xy[idx][0] - points_xy[idx - 1][0], points_xy[idx][1] - points_xy[idx - 1][1])
            for idx in range(1, len(points_xy))
        )
    )


def percentile(values: list[float], q: float, default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))
