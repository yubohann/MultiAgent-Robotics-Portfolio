"""Controlled external planning baselines on the public HM3D candidate interface.

This module does *not* relabel the local auction heuristic as a literature
method.  It implements the task-level graph/Voronoi allocation idea of
GVP-MREP on the shared public candidate pool.  The original paper's ROS,
RotorS, depth pipeline and trajectory optimiser are deliberately outside this
adapter, so callers must report this as a controlled ``GVP-MREP-inspired``
transfer rather than an original-environment reproduction.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

from aerocity_method.adapters.hm3d_baselines import PublicSearchState
from aerocity_method.contracts.io import finite_number, require_identifier, require_sha256
from aerocity_method.contracts.models import CandidateFragmentManifest
from aerocity_method.contracts.privacy import walk_public_payload

GVP_MREP_PORT_ID = "gvp_mrep_port"
GVP_MREP_PORT_SCHEMA_VERSION = "hm3d-gvp-mrep-controlled-transfer-v4"
GVP_MREP_AUTHOR_COMMIT = "f5865b9c9c39e9d85095555f3e04b4fa349fce40"
GVP_MREP_GRAPH_PARTITION_SHA256 = (
    "9eb02ce91f6e49184b224649ab6d6563139a82de58031ba5a60797ff36cbc846"
)
GVP_DISTANCE_DECAY_LAMBDA = 0.2
GVP_FRONTIER_ALLOWANCE = 0.1
GVP_JOB_DELAY_TAU = 0.3


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.dist(left, right)


def _transit_endpoints(
    manifest: CandidateFragmentManifest,
) -> tuple[tuple[str, tuple[float, float, float], float], ...]:
    rows: list[tuple[str, tuple[float, float, float], float]] = []
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        if len(fragment.path) < 2:
            raise ValueError("GVP-port transit fragment needs a non-empty guarded path")
        rows.append((fragment.agent_id, tuple(fragment.path[-1]), fragment.planned_end))
    if not rows:
        raise ValueError("GVP-port candidate has no transit fragments")
    return tuple(sorted(rows, key=lambda row: row[0]))


def _edge_length(
    state: PublicSearchState,
    manifest: CandidateFragmentManifest,
) -> dict[tuple[str, str], float]:
    """Build a public dynamic topology graph from admitted guarded routes.

    An edge from an agent to a frontier exists only when that public route was
    already emitted by the shared guard.  Frontier-to-frontier links model
    communication/topological adjacency, not a claim that the straight line
    between them is a collision-free flight path.
    """

    edges: dict[tuple[str, str], float] = {}

    def add(left: str, right: str, length: float) -> None:
        key = tuple(sorted((left, right)))
        previous = edges.get(key)
        edges[key] = length if previous is None else min(previous, length)

    exploration_frontiers = tuple(
        frontier for frontier in state.frontiers if frontier.task_kind == "explore"
    )
    for agent in state.agents:
        agent_node = f"agent:{agent.agent_id}"
        for frontier in exploration_frontiers:
            add(
                agent_node,
                f"frontier:{frontier.frontier_id}",
                _distance(agent.position_m, frontier.position_m),
            )
    for index, left in enumerate(exploration_frontiers):
        for right in exploration_frontiers[index + 1 :]:
            # This is a local-map adjacency edge.  It is only used for
            # Voronoi ownership, never as a command route.
            length = _distance(left.position_m, right.position_m)
            if length <= state.communication_range_m:
                add(f"frontier:{left.frontier_id}", f"frontier:{right.frontier_id}", length)
    for left_index, left in enumerate(state.agents):
        for right in state.agents[left_index + 1 :]:
            length = _distance(left.position_m, right.position_m)
            if length <= state.communication_range_m:
                add(f"agent:{left.agent_id}", f"agent:{right.agent_id}", length)
    # Bind the exact selected guarded paths to the graph.  A guard rewrite can
    # make an intended direct edge more expensive than its Euclidean chord.
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        features = dict(fragment.type_signature.public_features)
        rank = features.get("frontier_rank")
        role = features.get("assignment_role", "explore")
        if role == "hold":
            if rank != -1 or fragment.path[0] != fragment.path[-1]:
                raise ValueError("GVP-port hold transit has inconsistent public features")
            continue
        if role not in {"explore", "backtrack"}:
            raise ValueError("GVP-port transit has an unsupported assignment role")
        if not isinstance(rank, int) or not 0 <= rank < len(state.frontiers):
            raise ValueError("GVP-port transit is missing its public frontier rank")
        path_length = sum(
            _distance(tuple(start), tuple(end))
            for start, end in zip(fragment.path[:-1], fragment.path[1:], strict=True)
        )
        if role == "explore":
            add(
                f"agent:{fragment.agent_id}",
                f"frontier:{state.frontiers[rank].frontier_id}",
                path_length,
            )
    return edges


def _shortest_distances(edges: dict[tuple[str, str], float], source: str) -> dict[str, float]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for (left, right), length in edges.items():
        adjacency.setdefault(left, []).append((right, length))
        adjacency.setdefault(right, []).append((left, length))
    distances = {source: 0.0}
    queue: list[tuple[float, str]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for neighbor, length in adjacency.get(node, ()):
            proposal = distance + length
            if proposal < distances.get(neighbor, float("inf")):
                distances[neighbor] = proposal
                heapq.heappush(queue, (proposal, neighbor))
    return distances


def _frontier_regions(
    state: PublicSearchState,
    edges: dict[tuple[str, str], float],
) -> tuple[dict[str, int], dict[int, int]]:
    """Approximate GVP history-node regions on the shared public frontier graph."""

    frontier_ids = {
        frontier.frontier_id for frontier in state.frontiers if frontier.task_kind == "explore"
    }
    adjacency = {frontier_id: set() for frontier_id in frontier_ids}
    for left, right in edges:
        if not left.startswith("frontier:") or not right.startswith("frontier:"):
            continue
        left_id = left.removeprefix("frontier:")
        right_id = right.removeprefix("frontier:")
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)
    region_by_frontier: dict[str, int] = {}
    frontier_count_by_region: dict[int, int] = {}
    for root in sorted(frontier_ids):
        if root in region_by_frontier:
            continue
        region = len(frontier_count_by_region)
        pending = [root]
        members: set[str] = set()
        while pending:
            frontier_id = pending.pop()
            if frontier_id in members:
                continue
            members.add(frontier_id)
            pending.extend(sorted(adjacency[frontier_id] - members, reverse=True))
        for frontier_id in members:
            region_by_frontier[frontier_id] = region
        frontier_count_by_region[region] = len(members)
    return region_by_frontier, frontier_count_by_region


def _predicted_connected_fraction(
    endpoints: Sequence[tuple[str, tuple[float, float, float], float]],
    communication_range_m: float,
) -> float:
    if len(endpoints) <= 1:
        return 1.0
    visited = {0}
    pending = [0]
    while pending:
        index = pending.pop()
        point = endpoints[index][1]
        for other_index, (_, other, _) in enumerate(endpoints):
            if other_index not in visited and _distance(point, other) <= communication_range_m:
                visited.add(other_index)
                pending.append(other_index)
    return len(visited) / len(endpoints)


@dataclass(frozen=True, slots=True)
class GVPPortSelection:
    """Auditable output of the graph-Voronoi controlled transfer."""

    selected_manifest_hash: str
    selected_candidate_id: str
    scores: tuple[tuple[str, float], ...]
    selected_diagnostics: dict[str, object]
    schema_version: str = GVP_MREP_PORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GVP_MREP_PORT_SCHEMA_VERSION:
            raise ValueError("GVP-port selection schema mismatch")
        require_sha256(self.selected_manifest_hash, "selected_manifest_hash")
        require_identifier(self.selected_candidate_id, "selected_candidate_id")
        if not self.scores:
            raise ValueError("GVP-port selection needs candidate scores")
        for candidate_id, score in self.scores:
            require_identifier(candidate_id, "candidate_id")
            finite_number(score, "GVP-port score")
        walk_public_payload(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": GVP_MREP_PORT_ID,
            "schema_version": self.schema_version,
            "adaptation_status": "controlled_transfer_not_original_ros_reproduction",
            "author_source": {
                "repository": "https://github.com/NKU-MobFly-Robotics/GVP-MREP",
                "commit": GVP_MREP_AUTHOR_COMMIT,
                "graph_partition_sha256": GVP_MREP_GRAPH_PARTITION_SHA256,
            },
            "selected_manifest_hash": self.selected_manifest_hash,
            "selected_candidate_id": self.selected_candidate_id,
            "scores": [list(row) for row in self.scores],
            "selected_diagnostics": self.selected_diagnostics,
            "claim_limit": (
                "Public graph/Voronoi allocation inspired by GVP-MREP. It uses the same "
                "guarded candidate pool, sparse-range frontiers, CF2X runtime and communication "
                "contract as every ranked method; it is not the original ROS/RotorS pipeline."
            ),
        }


def _score_candidate(
    state: PublicSearchState,
    manifest: CandidateFragmentManifest,
) -> tuple[float, dict[str, object]]:
    endpoints = _transit_endpoints(manifest)
    edges = _edge_length(state, manifest)
    distances = {
        agent.agent_id: _shortest_distances(edges, f"agent:{agent.agent_id}")
        for agent in state.agents
    }
    by_agent = {agent_id: endpoint for agent_id, endpoint, _ in endpoints}
    transit_by_agent = {
        fragment.agent_id: fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    }
    assigned_frontiers: dict[str, str] = {}
    assigned_regions: dict[str, int] = {}
    voronoi_matches = 0
    travel_times: list[float] = []
    travel_distances: dict[str, float] = {}
    region_by_frontier, frontier_count_by_region = _frontier_regions(state, edges)
    for agent in state.agents:
        endpoint = by_agent.get(agent.agent_id)
        if endpoint is None:
            raise ValueError("GVP-port candidate omits a public agent transit")
        transit = transit_by_agent[agent.agent_id]
        features = dict(transit.type_signature.public_features)
        rank = features.get("frontier_rank")
        role = features.get("assignment_role", "explore")
        travel_times.append(
            next(end_s for agent_id, _, end_s in endpoints if agent_id == agent.agent_id)
            - state.decision_start_s
        )
        if role == "hold":
            if rank != -1:
                raise ValueError("GVP-port hold transit has an invalid frontier rank")
            assigned_frontiers[agent.agent_id] = "HOLD"
            assigned_regions[agent.agent_id] = -1
            travel_distances[agent.agent_id] = 0.0
            continue
        if role == "backtrack":
            if not isinstance(rank, int) or not 0 <= rank < len(state.frontiers):
                raise ValueError("GVP-port backtrack transit has an invalid frontier rank")
            if state.frontiers[rank].task_kind != "backtrack":
                raise ValueError("GVP-port backtrack transit does not bind a recovery frontier")
            assigned_frontiers[agent.agent_id] = "OUTCOME_BACKTRACK"
            assigned_regions[agent.agent_id] = -2
            travel_distances[agent.agent_id] = _distance(agent.position_m, endpoint)
            continue
        if not isinstance(rank, int) or not 0 <= rank < len(state.frontiers):
            raise ValueError("GVP-port explore transit has an invalid frontier rank")
        frontier = state.frontiers[rank]
        frontier_node = f"frontier:{frontier.frontier_id}"
        owner = min(
            state.agents,
            key=lambda row: (
                distances[row.agent_id].get(frontier_node, float("inf")),
                row.agent_id,
            ),
        )
        assigned_frontiers[agent.agent_id] = frontier.frontier_id
        assigned_regions[agent.agent_id] = region_by_frontier[frontier.frontier_id]
        voronoi_matches += int(owner.agent_id == agent.agent_id)
        travel_distances[agent.agent_id] = distances[agent.agent_id].get(
            frontier_node, _distance(agent.position_m, endpoint)
        )
    makespan = max(travel_times)
    mean_time = sum(travel_times) / len(travel_times)
    balance = 1.0 - min(1.0, (max(travel_times) - min(travel_times)) / max(mean_time, 1.0e-9))
    voronoi_fraction = voronoi_matches / len(state.agents)
    connectivity = _predicted_connected_fraction(endpoints, state.communication_range_m)
    arrival_by_agent = {
        agent_id: state.transit_timing_model.motion_seconds_for_distance(distance)
        for agent_id, distance in travel_distances.items()
    }
    author_gain_by_agent: dict[str, float] = {}
    delayed_competitors_by_agent: dict[str, int] = {}
    for agent in state.agents:
        agent_id = agent.agent_id
        frontier_id = assigned_frontiers[agent_id]
        if frontier_id in {"HOLD", "OUTCOME_BACKTRACK"}:
            author_gain_by_agent[agent_id] = 0.0
            delayed_competitors_by_agent[agent_id] = 0
            continue
        frontier_node = f"frontier:{frontier_id}"
        owner = min(
            state.agents,
            key=lambda row: (
                distances[row.agent_id].get(frontier_node, float("inf")),
                row.agent_id,
            ),
        )
        if owner.agent_id != agent_id:
            author_gain_by_agent[agent_id] = 0.0
            delayed_competitors_by_agent[agent_id] = 0
            continue
        region = assigned_regions[agent_id]
        frontier_count = float(frontier_count_by_region[region])
        delayed_competitors = 0
        delay_penalty = 0.0
        for other in state.agents:
            if other.agent_id == agent_id or assigned_regions[other.agent_id] != region:
                continue
            delay = arrival_by_agent[agent_id] - arrival_by_agent[other.agent_id]
            if delay > 0.0:
                delay_penalty += GVP_JOB_DELAY_TAU * delay
                delayed_competitors += 1
        gain = (
            max(frontier_count - delay_penalty, 0.0)
            + frontier_count * GVP_FRONTIER_ALLOWANCE
        ) / (delayed_competitors + 1)
        gain *= math.exp(-travel_distances[agent_id] * GVP_DISTANCE_DECAY_LAMBDA)
        author_gain_by_agent[agent_id] = gain
        delayed_competitors_by_agent[agent_id] = delayed_competitors
    score = sum(author_gain_by_agent.values())
    diagnostics: dict[str, object] = {
        "public_graph_node_count": len({node for edge in edges for node in edge}),
        "public_graph_edge_count": len(edges),
        "graph_edge_length_sum_m": sum(edges.values()),
        "voronoi_owner_match_fraction": voronoi_fraction,
        "assigned_frontiers": assigned_frontiers,
        "assigned_regions": assigned_regions,
        "predicted_makespan_s": makespan,
        "public_load_balance": balance,
        "predicted_connected_agent_fraction": connectivity,
        "author_gain_by_agent": author_gain_by_agent,
        "delayed_competitors_by_agent": delayed_competitors_by_agent,
        "frontier_count_by_region": {
            str(region): count for region, count in sorted(frontier_count_by_region.items())
        },
        "author_parameters": {
            "lambda": GVP_DISTANCE_DECAY_LAMBDA,
            "allowance": GVP_FRONTIER_ALLOWANCE,
            "tau": GVP_JOB_DELAY_TAU,
        },
        "author_formula_source": (
            "GVP-MREP GraphVoronoiPartition::GetBestTarget at commit "
            f"{GVP_MREP_AUTHOR_COMMIT}, graph_partition.cpp lines 786-822"
        ),
        "author_graph_partition_sha256": GVP_MREP_GRAPH_PARTITION_SHA256,
    }
    return score, diagnostics


def select_gvp_mrep_port(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> tuple[CandidateFragmentManifest, GVPPortSelection]:
    """Select one legal candidate through graph-distance Voronoi allocation."""

    rows = tuple(row for row in pool if row.feasible)
    if not rows:
        raise ValueError("GVP-port requires at least one legal public candidate")
    scored = tuple((row, *_score_candidate(state, row)) for row in rows)
    selected, _, diagnostics = max(
        scored,
        key=lambda row: (row[1], row[0].candidate_id),
    )
    selection = GVPPortSelection(
        selected_manifest_hash=selected.manifest_hash,
        selected_candidate_id=selected.candidate_id,
        scores=tuple(sorted((row.candidate_id, score) for row, score, _ in scored)),
        selected_diagnostics=diagnostics,
    )
    return selected, selection


__all__ = [
    "GVP_MREP_PORT_ID",
    "GVP_MREP_PORT_SCHEMA_VERSION",
    "GVP_MREP_AUTHOR_COMMIT",
    "GVP_MREP_GRAPH_PARTITION_SHA256",
    "GVPPortSelection",
    "select_gvp_mrep_port",
]
