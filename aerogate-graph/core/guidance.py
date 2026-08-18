"""Expert guidance (implementation withheld).

Exploring dense gate fields with reward signal alone is slow. This component
generates route hints from the global structure and injects them during
training to speed up convergence. How hints are produced — including the
LLM-backed client behind it — is part of the paper.
"""

from __future__ import annotations


class GuidanceEngine:
    """Feeds waypoint / route hints to the learner.

    The hint-generation mechanism (including the LLM client) is detailed in
    the paper.
    """

    def hint(self, *args, **kwargs):
        raise NotImplementedError("Released with the paper")