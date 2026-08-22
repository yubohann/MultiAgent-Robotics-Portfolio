"""Canonical bipartite graph construction with explicit interaction edges."""

from __future__ import annotations

import math
from itertools import combinations

from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    CandidateGraphBatch,
    FragmentInstance,
    InteractionEdge,
)


def _overlap(left: FragmentInstance, right: FragmentInstance) -> float:
    return max(
        0.0,
        min(left.planned_end, right.planned_end) - max(left.planned_start, right.planned_start),
    )


def _shares_point(left: FragmentInstance, right: FragmentInstance, tolerance: float) -> bool:
    return any(
        math.dist(left_point, right_point) <= tolerance
        for left_point in left.path
        for right_point in right.path
    )


def _inferred_interactions(
    fragments: tuple[FragmentInstance, ...], *, collision_tolerance: float
) -> tuple[InteractionEdge, ...]:
    edge_rows: dict[tuple[str, str, str], InteractionEdge] = {}
    for left, right in combinations(fragments, 2):
        overlap = _overlap(left, right)
        if overlap <= 0.0:
            continue
        left_hash = left.digest
        right_hash = right.digest

        def add(
            kind: str,
            weight: float,
            source_hash: str = left_hash,
            target_hash: str = right_hash,
        ) -> None:
            edge = InteractionEdge(source_hash, target_hash, kind, weight)
            key = (edge.source_fragment_hash, edge.target_fragment_hash, edge.kind)
            previous = edge_rows.get(key)
            if previous is None or edge.weight > previous.weight:
                edge_rows[key] = edge

        add("temporal_overlap", overlap)
        if left.agent_id == right.agent_id:
            add("resource_competition", overlap)
        if (
            left.agent_id != right.agent_id
            and left.type_signature.fragment_type == "transit"
            and right.type_signature.fragment_type == "transit"
            and _shares_point(left, right, collision_tolerance)
        ):
            add("collision", overlap)
        if (
            left.type_signature.fragment_type == "observation"
            and right.type_signature.fragment_type == "observation"
            and left.type_signature.digest == right.type_signature.digest
        ):
            add("redundant_observation", overlap)
        if "communication" in {
            left.type_signature.fragment_type,
            right.type_signature.fragment_type,
        }:
            add("communication", overlap)
    return tuple(
        sorted(
            edge_rows.values(),
            key=lambda edge: (
                edge.source_fragment_hash,
                edge.target_fragment_hash,
                edge.kind,
                edge.weight,
            ),
        )
    )


def build_candidate_graph(
    manifests: tuple[CandidateFragmentManifest, ...] | list[CandidateFragmentManifest],
    *,
    collision_tolerance: float = 1e-6,
) -> CandidateGraphBatch:
    if collision_tolerance < 0.0 or not math.isfinite(collision_tolerance):
        raise ValueError("collision_tolerance must be finite and non-negative")
    manifests_tuple = tuple(manifests)
    if not manifests_tuple:
        raise ValueError("graph construction requires at least one candidate manifest")
    manifest_by_hash: dict[str, CandidateFragmentManifest] = {}
    fragment_by_id: dict[str, FragmentInstance] = {}
    fragment_by_hash: dict[str, FragmentInstance] = {}
    membership: set[tuple[str, str, int]] = set()
    all_interactions: dict[tuple[str, str, str], InteractionEdge] = {}
    for manifest in manifests_tuple:
        manifest_hash = manifest.manifest_hash
        existing_manifest = manifest_by_hash.get(manifest_hash)
        if existing_manifest is not None and existing_manifest != manifest:
            raise ValueError("candidate hash collision")
        manifest_by_hash[manifest_hash] = manifest
        for order, fragment in enumerate(manifest.fragments):
            previous = fragment_by_id.get(fragment.instance_fragment_id)
            if previous is not None and previous.digest != fragment.digest:
                raise ValueError("fragment instance ID is bound to conflicting payloads")
            fragment_by_id[fragment.instance_fragment_id] = fragment
            fragment_by_hash[fragment.digest] = fragment
            membership.add((manifest_hash, fragment.digest, order))
        for edge in _inferred_interactions(
            manifest.fragments, collision_tolerance=collision_tolerance
        ):
            key = (edge.source_fragment_hash, edge.target_fragment_hash, edge.kind)
            previous_edge = all_interactions.get(key)
            if previous_edge is None or edge.weight > previous_edge.weight:
                all_interactions[key] = edge
    return CandidateGraphBatch(
        candidate_hashes=tuple(manifest_by_hash),
        fragment_hashes=tuple(fragment_by_hash),
        membership_edges=tuple(membership),
        interaction_edges=tuple(all_interactions.values()),
    )
