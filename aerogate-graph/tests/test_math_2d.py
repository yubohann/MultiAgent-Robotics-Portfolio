import math

import pytest

from shared.core.math_2d import (
    clip_vector_norm,
    subtract_points,
    vector_norm,
    yaw_from_velocity,
)


def test_vector_norm_basic():
    assert vector_norm((3.0, 4.0)) == pytest.approx(5.0)
    assert vector_norm((0.0, 0.0)) == 0.0


def test_vector_norm_negative_components():
    assert vector_norm((-3.0, -4.0)) == pytest.approx(5.0)


def test_clip_vector_norm_keeps_short_vectors():
    vec = (1.0, 1.0)
    clipped = clip_vector_norm(vec, max_norm=5.0)
    assert clipped == pytest.approx((1.0, 1.0))


def test_clip_vector_norm_clamps_long_vectors():
    vec = (6.0, 0.0)
    clipped = clip_vector_norm(vec, max_norm=2.0)
    assert clipped == pytest.approx((2.0, 0.0))
    assert vector_norm(clipped) == pytest.approx(2.0)


def test_clip_vector_norm_preserves_direction():
    vec = (3.0, 4.0)
    clipped = clip_vector_norm(vec, max_norm=5.0)
    # original norm is exactly 5.0, so it stays unchanged
    assert clipped == pytest.approx((3.0, 4.0))


def test_clip_vector_norm_handles_zero():
    assert clip_vector_norm((0.0, 0.0), max_norm=1.0) == (0.0, 0.0)


def test_yaw_from_velocity_positive_x():
    assert yaw_from_velocity((1.0, 0.0)) == pytest.approx(0.0)


def test_yaw_from_velocity_positive_y():
    assert yaw_from_velocity((0.0, 1.0)) == pytest.approx(math.pi / 2)


def test_yaw_from_velocity_zero_uses_fallback():
    assert yaw_from_velocity((0.0, 0.0), fallback_yaw_rad=1.234) == pytest.approx(1.234)


def test_subtract_points():
    assert subtract_points((5.0, 7.0), (2.0, 3.0)) == (3.0, 4.0)