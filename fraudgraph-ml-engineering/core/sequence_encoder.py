"""Sequence encoder branches (implementation withheld).

Encode ordered behavioural views — relational sequences and event sequences —
into representations that complement what the graph branch captures. The view
construction and attention mechanism are part of the paper.
"""

from __future__ import annotations


class SequenceEncoder:
    """Encodes ordered behavioural sequences into representations.

    Sequence views, tokenization, and attention mechanism are detailed in the
    paper.
    """

    def forward(self, sequences):
        raise NotImplementedError("Released with the paper")