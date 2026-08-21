"""Graph-structured policy (implementation withheld).

This is the heart of the project. Instead of flattening the gate field into a
plain feature vector, the policy consumes a graph-structured state and emits
an action distribution. How the graph is built, how information propagates
across it, and how the actor/critic heads are wired are all part of the paper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class GraphPolicy(ABC):
    """Abstract interface of the graph-structured policy.

    Subclasses implement ``act`` and ``update``. The concrete graph encoder,
    message-passing scheme, and network heads are detailed in the paper.
    """

    @abstractmethod
    def act(self, observation):
        """Pick an action given a graph-structured observation."""
        raise NotImplementedError("Released with the paper")

    @abstractmethod
    def update(self, batch):
        """One learning update on a sampled transition batch."""
        raise NotImplementedError("Released with the paper")