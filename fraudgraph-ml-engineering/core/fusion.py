"""Fusion classifier (implementation withheld).

Combines graph, sequence, and event representations into one fraud decision,
with training objectives designed to keep the branches complementary rather
than redundant. The fusion strategy is part of the paper.
"""

from __future__ import annotations


class FusionClassifier:
    """Fuses multi-view representations into a fraud classification.

    Fusion strategy and multi-objective training design are detailed in the
    paper.
    """

    def forward(self, graph_emb, seq_emb, event_emb):
        raise NotImplementedError("Released with the paper")