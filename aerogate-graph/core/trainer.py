"""Training pipeline (implementation withheld).

Ties together environment interaction and learning: sampling, replay,
expert-demonstration bootstrapping, curriculum progression, and checkpoint
selection from evaluation results. The schedule and curriculum rules are part
of the paper.
"""

from __future__ import annotations


class Trainer:
    """Orchestrates environment interaction and policy learning.

    Training schedule, curriculum rules, and checkpoint-selection criteria
    are detailed in the paper.
    """

    def train(self, *args, **kwargs):
        raise NotImplementedError("Released with the paper")