from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.graphs.builder import build_candidate_graph


def test_graph_hash_is_invariant_to_candidate_order(manifests):
    left = build_candidate_graph(manifests)
    right = build_candidate_graph(tuple(reversed(manifests)))
    assert left.graph_hash == right.graph_hash


def test_graph_contains_membership_for_every_fragment(manifests):
    graph = build_candidate_graph(manifests)
    assert len(graph.membership_edges) == sum(len(manifest.fragments) for manifest in manifests)


def test_graph_rejects_empty_input():
    with pytest.raises(ValueError):
        build_candidate_graph(())


def test_fragment_id_conflict_is_rejected(manifests):
    conflicting = replace(
        manifests[1].fragments[0],
        instance_fragment_id=manifests[0].fragments[0].instance_fragment_id,
    )
    altered = replace(manifests[1], fragments=(conflicting, manifests[1].fragments[1]))
    with pytest.raises(ValueError):
        build_candidate_graph((manifests[0], altered))


def test_overlapping_same_agent_creates_resource_edge(manifests):
    first = manifests[0].fragments[0]
    overlapping = replace(
        manifests[0].fragments[1],
        planned_start=0.5,
        planned_end=1.0,
    )
    manifest = replace(manifests[0], fragments=(first, overlapping))
    kinds = {edge.kind for edge in build_candidate_graph((manifest,)).interaction_edges}
    assert {"temporal_overlap", "resource_competition"} <= kinds


def test_negative_collision_tolerance_is_rejected(manifests):
    with pytest.raises(ValueError):
        build_candidate_graph(manifests, collision_tolerance=-1)
