"""Dataset scheduler and label-scarcity mechanism (implementation withheld).

Decides which datasets and difficulty regimes the pipeline trains on and
when, including the active-learning-style reveal of labels when supervision
is scarce. The scheduling policy is part of the paper.
"""

from __future__ import annotations


class Scheduler:
    """Schedules datasets and label regimes across training.

    Scheduling policy, label-scarcity ladder, and active-learning reveal
    rules are detailed in the paper.
    """

    def plan(self, *args, **kwargs):
        raise NotImplementedError("Released with the paper")