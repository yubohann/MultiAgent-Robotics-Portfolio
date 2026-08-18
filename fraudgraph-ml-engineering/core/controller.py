"""Federated-style controller (implementation withheld).

Governs training across client data partitions and aggregation rounds —
which clients participate, how models are averaged, and how checkpoints are
selected under stability constraints. These rules are part of the paper.
"""

from __future__ import annotations


class Controller:
    """Governs multi-round, multi-client training.

    Participation rule, aggregation scheme, and stability mechanism are
    detailed in the paper.
    """

    def step(self, *args, **kwargs):
        raise NotImplementedError("Released with the paper")