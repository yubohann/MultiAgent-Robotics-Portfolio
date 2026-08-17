"""Paired-sample power calculation for declared method comparisons."""

from __future__ import annotations

import math
from statistics import NormalDist

from .common import _valid_number


def required_paired_episodes(
    *,
    familywise_alpha: float,
    power: float,
    minimum_effect_size: float,
    difference_standard_deviation: float,
    comparison_count: int,
) -> int:
    """Return the predeclared paired-sample normal approximation.

    ``difference_standard_deviation`` is the standard deviation of per-episode
    paired method differences, not the marginal score standard deviation. A
    Bonferroni-adjusted two-sided alpha is used for every declared comparison.
    Final reporting still requires confidence intervals and the frozen public
    metric implementation.
    """

    if not _valid_number(familywise_alpha, minimum=0.0, maximum=1.0):
        raise ValueError("familywise_alpha must be in (0, 1)")
    if not _valid_number(power, minimum=0.5, maximum=1.0):
        raise ValueError("power must be in (0.5, 1)")
    if not _valid_number(minimum_effect_size, minimum=0.0) or float(minimum_effect_size) > 1.0:
        raise ValueError("minimum_effect_size must be in (0, 1]")
    if not _valid_number(difference_standard_deviation, minimum=0.0):
        raise ValueError("difference_standard_deviation must be positive")
    if isinstance(comparison_count, bool) or not isinstance(comparison_count, int) or comparison_count < 1:
        raise ValueError("comparison_count must be a positive integer")
    adjusted_two_sided_tail = float(familywise_alpha) / (2.0 * comparison_count)
    z_alpha = NormalDist().inv_cdf(1.0 - adjusted_two_sided_tail)
    z_power = NormalDist().inv_cdf(float(power))
    estimate = (
        (z_alpha + z_power)
        * float(difference_standard_deviation)
        / float(minimum_effect_size)
    ) ** 2
    return max(2, math.ceil(estimate))
