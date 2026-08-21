"""Global route planning (implementation withheld).

Training and evaluation need a global traversal order over the gate field as
a reference. This component computes a full route over the gate graph. The
graph-search / optimization strategy used to rank traversal orders is part of
the paper.
"""

from __future__ import annotations


class GlobalPlanner:
    """Computes a global traversal route over the gate field.

    The graph-search / optimization strategy is detailed in the paper.
    """

    def plan(self, *args, **kwargs):
        raise NotImplementedError("Released with the paper")