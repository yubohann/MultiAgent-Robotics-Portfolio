from __future__ import annotations

from shared.core.collision_2d import GateObstacleMap2D


def test_default_gate_map_contains_gate_posts() -> None:
    obstacle_map = GateObstacleMap2D.from_gate()
    assert len(obstacle_map) == 12
    assert obstacle_map.collides_point(obstacle_map.obstacles[0].center_xy)
