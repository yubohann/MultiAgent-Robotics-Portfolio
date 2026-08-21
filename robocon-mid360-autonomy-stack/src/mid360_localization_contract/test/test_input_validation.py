from types import SimpleNamespace

from mid360_localization_contract.input_validation import inspect_custom_points


def point(offset_time: int, x: float = 1.0, y: float = 2.0, z: float = 3.0):
    return SimpleNamespace(offset_time=offset_time, x=x, y=y, z=z)


def test_valid_points_have_monotonic_offsets_and_finite_coordinates():
    statistics = inspect_custom_points([point(0), point(20), point(40)])

    assert statistics.point_count == 3
    assert statistics.finite_point_count == 3
    assert statistics.non_finite_point_count == 0
    assert statistics.offsets_monotonic
    assert statistics.offset_span_ns == 40


def test_non_finite_coordinates_and_offset_regression_are_reported():
    statistics = inspect_custom_points([point(30), point(10, x=float("nan")), point(50, z=float("inf"))])

    assert statistics.finite_point_count == 1
    assert statistics.non_finite_point_count == 2
    assert not statistics.offsets_monotonic
    assert statistics.offset_span_ns == 50
