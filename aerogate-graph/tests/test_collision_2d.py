import pytest

from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D


def _obstacle_map():
    return GateObstacleMap2D(
        obstacles=[
            GatePostObstacle2D(center_xy=(5.0, 5.0), collision_radius_m=1.0),
        ]
    )


def test_empty_map_has_no_collisions():
    m = GateObstacleMap2D.empty()
    assert m.min_signed_distance((0.0, 0.0), drone_radius_m=0.0) == float("inf")


def test_collides_point_within_radius():
    m = _obstacle_map()
    assert m.collides_point((5.0, 5.0), drone_radius_m=0.2)


def test_no_collision_far_away():
    m = _obstacle_map()
    assert not m.collides_point((50.0, 50.0), drone_radius_m=0.2)


def test_min_signed_distance_outside():
    m = _obstacle_map()
    # 2.0 away from center, 1.0 radius -> clearance 1.0
    d = m.min_signed_distance((5.0, 3.0), drone_radius_m=0.0)
    assert d == pytest.approx(1.0)


def test_min_signed_distance_inside_is_negative():
    m = _obstacle_map()
    d = m.min_signed_distance((5.2, 5.0), drone_radius_m=0.0)
    assert d < 0


def test_segment_collides_passing_through():
    m = _obstacle_map()
    assert m.segment_collides((0.0, 5.0), (10.0, 5.0), drone_radius_m=0.2)


def test_segment_no_collision_parallel():
    m = _obstacle_map()
    assert not m.segment_collides((0.0, 9.0), (10.0, 9.0), drone_radius_m=0.2)


def test_colliding_obstacles_lists_only_nearby():
    m = _obstacle_map()
    hits = m.colliding_obstacles((5.0, 5.2), drone_radius_m=0.1)
    assert len(hits) == 1
    assert hits[0].center_xy == (5.0, 5.0)