"""Graph encoder branch (implementation withheld).

Encodes the transaction-relation graph into node embeddings. The branch is
deliberately designed for heterophily — fraud nodes often connect to innocent
nodes, so an encoder that assumes homophily would average the signal away.
The encoder design and aggregation scheme are part of the paper.
"""

from __future__ import annotations


class GraphEncoder:
    """Encodes relational structure into node representations.

    Encoder design, aggregation scheme, and heterophily handling are detailed
    in the paper.
    """

    def forward(self, graph):
        raise NotImplementedError("Released with the paper")