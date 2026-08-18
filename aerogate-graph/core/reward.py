"""Reward design (implementation withheld).

The reward shapes the learning signal: fly fast, stay inside the safety
boundary, and — for the multi-agent track — keep formation. How the terms are
weighted and how safety penalties are applied are part of the paper.
"""

from __future__ import annotations


class RewardModel:
    """Computes dense rewards from environment transitions.

    Reward-shaping terms, weights, and safety penalties are detailed in the
    paper.
    """

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Released with the paper")