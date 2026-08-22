"""Run one real, target-free, multi-round HM3D P07 exploration episode.

This is the only P07 entry point that emits the target-free exploration worker
schema accepted by ``hm3d_p07_matrix``.  It keeps the CF2X
fleet alive across decisions, derives its public map solely from real sparse
range outcomes, and uses the P03 ESDF only after execution to score the frozen
free-flight-volume denominator.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
import uuid
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aerocity_method.adapters.hm3d_baselines import (
    _manifest_route_tube_separation_m,
    MINIMUM_MEANINGFUL_EXPLORATION_PATH_M,
    PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M,
    PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M,
    PUBLIC_TASK_RESERVATION_MIN_NORMAL_ALIGNMENT,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SWITCH_MARGIN_GAIN,
    PUBLIC_VERTICAL_OPPORTUNITY_THRESHOLD_M,
    ConservativeTransitTimingModel,
    GuardedPath,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    PublicTaskReservation,
    build_public_candidate_pool,
    is_non_alias_exploration_path,
    public_candidate_pool_hash,
    outcome_calibrated_path_length_budget_m,
    select_public_baseline,
    task_reservation_matches_frontier,
)
from aerocity_method.adapters.hm3d_execution import execute_hm3d_manifest
from aerocity_method.adapters.hm3d_external_baselines import select_gvp_mrep_port
from aerocity_method.adapters.hm3d_marl_ipp import (
    build_marl_ipp_training_transition,
    select_marl_ipp_port,
)
from aerocity_method.adapters.hm3d_runtime import build_enclosed_esdf, reachable_component_mask
from aerocity_method.adapters.hm3d_single_rl import (
    build_single_rl_training_transition,
    select_single_rl,
)
from aerocity_method.archives.qd import AdmissionDecision, Elite, QDArchive
from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.hm3d_public_schema import public_schema_fields
from aerocity_method.contracts.io import canonical_sha256, write_json_atomic
from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentTypeSignature,
    PublicMethodContext,
)
from aerocity_method.evaluation.hm3d_communication_contract import HM3DCommunicationContract
from aerocity_method.evaluation.hm3d_evidence_classification import (
    P07_RECORD_PURPOSES,
    build_current_evidence_integrity_contract,
    require_p07_evidence_field,
)
from aerocity_method.evaluation.hm3d_exploration_metrics import (
    ExplorationMetricSample,
    score_exploration_episode,
)
from aerocity_method.evaluation.hm3d_exploration_contract import (
    load_exploration_observation_contract,
)
from aerocity_method.evaluation.hm3d_safety import (
    TimedPolyline,
    TimedStationary,
    assess_collision_avoidance_recovery,
    assess_route_tube_separation,
    assess_synchronized_separation,
)
from aerocity_method.runtime import hm3d_cf2x_execution as cf2x
from aerocity_method.runtime.hm3d_belief import (
    FREE,
    OCCUPIED,
    PublicRangeRayOutcome,
    SparseVoxelBelief,
    public_free_voxel_transition,
)
from aerocity_method.runtime.hm3d_frontiers import (
    FrontierExtractionConfig,
    extract_frontier_clusters,
    neighbors_26,
)
from aerocity_method.runtime.hm3d_realised_qd import (
    HM3D_QD_CALIBRATION_INTENT_MODES,
    HM3D_REALISED_QD_ARCHIVE_SPEC,
    HM3D_REALISED_QD_SCHEMA_VERSION,
    MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION,
    PlannedQDSelector,
    PublicExplorationNeed,
    RealisedQDDescriptor,
    OutcomeGroundedQDSelector,
    OutcomeQDFeatureVector,
    audit_intent_realised_alignment,
    audit_pre_registered_qd_descriptor_families,
    audit_public_candidate_intent_richness,
    audit_realised_qd_calibration_mode_contrasts,
    audit_realised_qd_footprint_separation,
    audit_realised_qd_reproducibility,
    audit_realised_qd_richness,
    audit_value_protected_candidate_diversity,
    public_exploration_need_from_public_belief,
    public_observation_workload_balance_from_range_outcomes,
    qd_selector_backbone_sha256,
    realised_descriptor_from_public_outcomes,
    outcome_qd_feature_vector_from_public_outcomes,
)
from aerocity_method.runtime.hm3d_start_resets import (
    P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
    P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE,
    P07_START_RESET_SCHEMA_VERSION,
)
from aerocity_method.runtime.hm3d_team_collaboration import (
    audit_translation_invariant_team_trajectories,
)
from aerocity_method.runtime.hm3d_trajectory import maximum_rest_to_rest_distance_m
from aerocity_method.runtime.physx_query_cache import MemoizedRaycastClosestQuery
from aerocity_method.runtime.range_sensing import resolve_public_range_directions
from aerocity_method.runtime.sensors import SensorProfile
from aerocity_method.runtime.tokens import authorize_manifest


class CandidateRouteGuardError(RuntimeError):
    """Carry evaluator-only route evidence into an immutable failure artifact."""

    def __init__(self, message: str, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


VERTICAL_OPPORTUNITY_THRESHOLD_M = PUBLIC_VERTICAL_OPPORTUNITY_THRESHOLD_M
MOBILITY_HEIGHT_BAND_M = 1.0
# FUEL samples observation poses 1.5--2.5 m around a frontier instead of
# commanding the vehicle centre to the free/unknown boundary itself.  These
# are sensor standoff variants, not route-length targets.  Reverted to the
# calibrated 1.5-2.5 m band: wider standoff does not create longer routes
# (poses are limited by public FREE support, which is 0.2-0.5 m at
# constrained starts and open elsewhere), and a rolling decision window
# shortens the route budget in open scenes.  Empirical A/B (2026-08-08):
# 1.5-2.5 m no-window beat 4-6 m + 6 s rolling on 00459 (52% vs 40%
# coverage) and 00626 (13% vs 9%).
FRONTIER_OBSERVATION_STANDOFF_M = 1.5
FRONTIER_OBSERVATION_MAX_STANDOFF_M = 2.5
FRONTIER_OBSERVATION_STANDOFF_VARIANTS_M = (1.5, 2.0, 2.5)
# A received-free centre voxel alone does not establish enough local evidence
# to nominate a CF2X observation point.  The radius is only the query window;
# sparse range data need not classify every voxel in that window.  The exact
# collision guard remains the authority for continuous physical clearance.
PUBLIC_ROUTE_SUPPORT_RADIUS_M = cf2x.REQUIRED_TERMINAL_CLEARANCE_M
PUBLIC_ROUTE_SUPPORT_MIN_FREE_VOXELS = 3
# A count of three received-free cells can still be a one-dimensional grazing
# ray.  Route-progress alternatives use the stronger, public-only condition
# that free evidence spans both signs of at least two axes.  This is a ranking
# and candidate-generation proxy, never a substitute for the shared PhysX
# clearance guard.
PUBLIC_ROUTE_PROGRESS_MIN_BALANCED_AXES = 2
PUBLIC_OBSERVATION_POINTS_PER_FRONTIER_VIEWPOINT = 6
PUBLIC_ROUTE_PROGRESS_VARIANTS_PER_PATH = 4
# Route-level access alternatives commit to a farther received-free point in
# the same connected component instead of only sampling a 2.5 m observation
# standoff. This is a public-map action proposal; the shared static guard
# remains the physical admission authority.
PUBLIC_REGION_ACCESS_MIN_ADVANCE_M = 3.0
PUBLIC_REGION_ACCESS_MAX_PER_CLUSTER = 1
# Keep several route-prefix alternatives for each originating vehicle.  The
# count is intentionally smaller than the total public-frontier budget: this
# protects a long, spatially distinct route when the most efficient prefix is
# later rejected by the common static guard, while leaving room for frontier
# observation poses shared by the whole team.
PUBLIC_ROUTE_PROGRESS_RETAINED_PER_SOURCE = 3
# A route prefix at least this long is an access-route action, not a sensing
# hold. It may carry full cluster-gain ranking while still being admitted by
# the shared static guard and producing real outcomes.
PUBLIC_ROUTE_PROGRESS_FULL_GAIN_MIN_M = 2.0
PUBLIC_FRONTIER_CLUSTER_SEARCH_BUDGET = 16
# Unexplored-potential gain for region-access / route-progress candidates.
# Radius and weight are frozen protocol constants; the sparse sampling count
# is deterministic (fixed seed) so the gain is reproducible.  Re-enabled only
# together with wait-period sensing (completed vehicles keep sampling during
# the synchronous hold); without it the gain is scene-dependent.
PUBLIC_POTENTIAL_GAIN_RADIUS_M = 8.0
PUBLIC_POTENTIAL_GAIN_WEIGHT = 0.02
PUBLIC_POTENTIAL_GAIN_SAMPLES = 128
PUBLIC_FRONTIER_VIEWPOINTS_PER_CLUSTER = 2
PUBLIC_FRONTIER_OBSERVATION_POINTS_PER_VIEWPOINT = 2
PUBLIC_FRONTIER_PATH_SEARCH_BUDGET_PER_DECISION = 512
# Region access routes are the long-horizon action authority. Reserve a fixed
# slice of the path-search budget before observation standoff routes consume it,
# so a dense near-frontier viewpoint set cannot silently erase every corridor
# access proposal.
PUBLIC_REGION_ACCESS_PATH_SEARCH_RESERVE_PER_DECISION = 64
# Route prefixes and the common candidate authority use the same two-voxel
# movement threshold.  A route that cannot clear this distance is a sensing
# hold, never an exploration action.
PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M = MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
# Candidate admission is intentionally bounded, but the pre-guard viewpoint
# set must leave enough spatial alternatives after the shared static guard
# rejects wall-adjacent poses. The selector still receives at most eight legal
# team candidates.
PUBLIC_FRONTIER_VIEWPOINTS_PER_AGENT = 8
# A target that produced no new *public* free voxels is not immediately
# retried from the same received-free neighbourhood.  The one-decision TTL is
# deliberately short: it prevents a micro-motion/dwell loop without declaring
# an online, partially observed frontier permanently exhausted.
PUBLIC_NO_GAIN_VIEWPOINT_COOLDOWN_DECISIONS = 1
PUBLIC_NO_GAIN_VIEWPOINT_COOLDOWN_CHEBYSHEV_RADIUS_VOXELS = 1
# A recovery route must be a previously completed exploration command.  It is
# intentionally no shorter than a normal meaningful exploration edge, so the
# matcher cannot replace a hold with settling noise.
OUTCOME_BACKTRACK_MIN_PATH_M = MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
# A failed public voxel-chain route may still have a physically safe long
# alternative that the sparse belief cannot represent.  The exact clearance
# grid planner is used only as a bounded evaluator-side rescue for public
# exploration endpoints, never for ranking or method-specific path shaping.
EXACT_CLEARANCE_GRID_RESCUE_MAX_ATTEMPTS_PER_PATH = 4
EXACT_CLEARANCE_GRID_RESCUE_MAX_PATHS_PER_DECISION = 8
EXACT_CLEARANCE_GRID_RESCUE_MIN_PATH_LENGTH_M = 2.0


class _OutcomeBacktrackRoute(NamedTuple):
    """One own-history route that may be reversed once after its success."""

    route_id: str
    agent_id: str
    source_decision_id: str
    source_manifest_hash: str
    source_transit_outcome_sha256: str
    source_minimum_static_mesh_clearance_m: float
    source_static_clearance_contract_required_m: float
    path_m: tuple[tuple[float, float, float], ...]


def _path_length_m(path_m: Sequence[tuple[float, float, float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(path_m, path_m[1:], strict=False))


def _point_at_path_arc_length_m(
    path_m: Sequence[tuple[float, float, float]],
    target_distance_m: float,
) -> tuple[float, float, float]:
    """Return the point at an arc-length distance along one polyline."""

    path = tuple(tuple(point) for point in path_m)
    total = _path_length_m(path)
    target = min(max(float(target_distance_m), 0.0), total)
    remaining = target
    for left, right in zip(path, path[1:], strict=False):
        length = math.dist(left, right)
        if length <= 1.0e-12:
            continue
        if remaining > length + 1.0e-12:
            remaining -= length
            continue
        ratio = min(1.0, max(0.0, remaining / length))
        return tuple(left[axis] + ratio * (right[axis] - left[axis]) for axis in range(3))
    return path[-1]


def _path_prefix_to_arc_length_m(
    path_m: Sequence[tuple[float, float, float]],
    target_distance_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """Truncate a polyline at an arc length while preserving original turns."""

    path = tuple(tuple(point) for point in path_m)
    total = _path_length_m(path)
    target = min(max(float(target_distance_m), 0.0), total)
    prefix: list[tuple[float, float, float]] = [path[0]]
    remaining = target
    for left, right in zip(path, path[1:], strict=False):
        length = math.dist(left, right)
        if length <= 1.0e-12:
            continue
        if remaining > length + 1.0e-12:
            remaining -= length
            if math.dist(prefix[-1], right) > 1.0e-9:
                prefix.append(right)
            continue
        ratio = min(1.0, max(0.0, remaining / length))
        point = tuple(left[axis] + ratio * (right[axis] - left[axis]) for axis in range(3))
        if math.dist(prefix[-1], point) > 1.0e-9:
            prefix.append(point)
        return tuple(prefix)
    if math.dist(prefix[-1], path[-1]) > 1.0e-9:
        prefix.append(path[-1])
    return tuple(prefix)


def _exact_clearance_grid_rescue_path(
    scene_query: Any,
    clearance_oracle: Any,
    agent_id: str,
    start_m: tuple[float, float, float],
    endpoint_candidates_m: Sequence[tuple[float, float, float]],
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *,
    minimum_path_length_m: float,
    maximum_attempts: int = EXACT_CLEARANCE_GRID_RESCUE_MAX_ATTEMPTS_PER_PATH,
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    attempt_budget: dict[str, int] | None = None,
    requested_route_length_m: float | None = None,
    corridor_path_m: Sequence[tuple[float, float, float]] | None = None,
    segment_cache: (
        dict[
            tuple[
                str,
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            Any,
        ]
        | None
    ) = None,
) -> tuple[GuardedPath | None, dict[str, object]]:
    """Route a public exploration endpoint through the exact clearance grid.

    The sparse public map can produce a long voxel chain through a corridor
    while the exact collision mesh rejects its chord or compressed waypoints.
    This helper only rescues endpoints that are already public exploration
    poses.  It uses the same exact PhysX line guard and clearance oracle as the
    shared route guard, and the resulting path is admitted only through the
    same trackability contract.  It never changes candidate ranking.
    """

    if maximum_attempts < 1:
        return None, {"attempted": False, "reason": "maximum_attempts_below_one"}
    attempts = 0
    seen_endpoints: set[tuple[float, float, float]] = set()
    fine_grid_route_audit: dict[str, object] | None = None
    for endpoint in endpoint_candidates_m:
        point = tuple(endpoint)
        if point in seen_endpoints or math.dist(start_m, point) <= 1.0e-9:
            continue
        seen_endpoints.add(point)
        if attempts >= maximum_attempts:
            break
        if attempt_budget is not None and attempt_budget.get("remaining", 0) <= 0:
            break
        attempts += 1
        if attempt_budget is not None:
            attempt_budget["remaining"] = max(0, int(attempt_budget["remaining"]) - 1)
        grid_route = cf2x._grid_route(
            scene_query,
            clearance_oracle,
            agent_id,
            start_m,
            point,
            bounds_min,
            bounds_max,
            diagnostic_sink,
            segment_cache=segment_cache,
        )
        grid_rescue_kind = "coarse_clearance_grid"
        fine_grid_route_audit = None
        if grid_route is None:
            fine_result = cf2x._fine_clearance_grid_route(
                scene_query,
                clearance_oracle,
                agent_id,
                start_m,
                point,
                bounds_min,
                bounds_max,
                diagnostic_sink,
                requested_path_m=corridor_path_m or (start_m, point),
                segment_cache=segment_cache,
            )
            if fine_result is not None:
                grid_route, fine_grid_route_audit = fine_result
                if grid_route is not None:
                    grid_rescue_kind = "fine_clearance_grid"
        if grid_route is None:
            continue
        route = tuple(tuple(waypoint) for waypoint in grid_route)
        route_length_m = _path_length_m(route)
        if route_length_m + 1.0e-9 < minimum_path_length_m:
            continue
        guarded = cf2x._admit_trackable_path(
            GuardedPath(
                True,
                route,
                reason=(
                    "public_flight_fine_grid_route"
                    if grid_rescue_kind == "fine_clearance_grid"
                    else "public_flight_grid_route"
                ),
            ),
            bounds_min,
            bounds_max,
        )
        if guarded.legal:
            audit: dict[str, object] = {
                "attempted": True,
                "admitted": True,
                "candidate_guards_attempted": attempts,
                "original_route_length_m": requested_route_length_m,
                "grid_route_length_m": route_length_m,
                "pullback_route_length_m": route_length_m,
                "requested_pullback_endpoint_m": point,
                "pullback_endpoint_m": route[-1],
                "pullback_source": "exact_clearance_grid_route",
                "grid_rescue_kind": grid_rescue_kind,
                "fine_grid_route_audit": fine_grid_route_audit,
                "pullback_retained_fraction": (
                    route_length_m / requested_route_length_m
                    if requested_route_length_m and requested_route_length_m > 1.0e-9
                    else None
                ),
            }
            return guarded, audit
    return None, {
        "attempted": True,
        "admitted": False,
        "candidate_guards_attempted": attempts,
        "fine_grid_route_audit": fine_grid_route_audit,
        "reason": "no_exact_clearance_grid_route_admitted",
    }


def _terminal_clearance_pullback_guarded_path(
    belief: SparseVoxelBelief,
    scene_query: Any,
    clearance_oracle: Any,
    agent_id: str,
    requested_path_m: Sequence[tuple[float, float, float]],
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *,
    minimum_path_length_m: float = PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M,
    maximum_candidate_guards: int = 8,
    voxel_keys: Sequence[tuple[int, int, int]] | None = None,
    received_free_support_cache: _ReceivedFreeSupportCache | None = None,
    exact_clearance_grid_rescue_budget: dict[str, int] | None = None,
    diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    segment_cache: (
        dict[
            tuple[
                str,
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            Any,
        ]
        | None
    ) = None,
) -> tuple[GuardedPath | None, dict[str, object]]:
    """Pull a wall-adjacent observation endpoint back along the same public route.

    A frontier pose can satisfy the public-map support test while sitting too
    close to static geometry for the common 0.90 m terminal planning reserve.
    Rather than discarding the whole long route, this helper searches backwards
    along the already generated received-free polyline for the farthest
    endpoint that still passes the exact PhysX route guard. The shared guard
    remains the sole authority; the pullback is only an alternative route
    proposal derived from the same public path.
    """

    path = tuple(tuple(point) for point in requested_path_m)
    if len(path) < 2:
        return None, {"attempted": False, "reason": "path_too_short"}
    total_length_m = _path_length_m(path)
    if total_length_m + 1.0e-9 < minimum_path_length_m:
        return None, {"attempted": False, "reason": "path_below_minimum_length"}
    exact_clearance_grid_rescue_audit: dict[str, object] = {
        "attempted": False,
        "reason": (
            "below_minimum_grid_rescue_length"
            if total_length_m
            < EXACT_CLEARANCE_GRID_RESCUE_MIN_PATH_LENGTH_M - 1.0e-9
            else "not_reached"
        ),
    }
    step_m = max(cf2x.ROUTE_CLEARANCE_SAMPLE_STEP_M, belief.resolution_m * 0.5)
    target_distances: set[float] = set()
    cumulative = 0.0
    for left, right in zip(path, path[1:], strict=False):
        cumulative += math.dist(left, right)
        if minimum_path_length_m + 1.0e-9 <= cumulative <= total_length_m - 1.0e-9:
            target_distances.add(round(cumulative, 9))
    maximum_samples = max(8, math.ceil((total_length_m - minimum_path_length_m) / step_m))
    for index in range(1, maximum_samples + 1):
        target = total_length_m - index * step_m
        if target < minimum_path_length_m - 1.0e-9:
            continue
        target_distances.add(round(target, 9))
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for target_distance_m in sorted(target_distances, reverse=True):
        point = _point_at_path_arc_length_m(path, target_distance_m)
        if (
            point is not None
            and clearance_oracle.admits_many((point,), cf2x.REQUIRED_TERMINAL_CLEARANCE_M)
        ):
            candidates.append((target_distance_m, point))
        if len(candidates) >= maximum_candidate_guards:
            break
    voxel_prefix_candidates: list[tuple[float, tuple[float, float, float]]] = []
    if voxel_keys is not None:
        # The requested polyline is compressed and may skip public voxel
        # centers that are the safest terminal poses.  Adding the original
        # voxel chain lets the exact guard rescue a long route whose final
        # requested waypoint is wall-adjacent without inventing a new route.
        for index in range(max(1, len(voxel_keys) - 1), 0, -1):
            point = belief.voxel_center(voxel_keys[index])
            route_prefix = _public_route_prefix_from_voxel_chain(
                belief,
                voxel_keys=voxel_keys,
                start_m=path[0],
                progress_point_m=point,
                minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                received_free_support_cache=received_free_support_cache,
            )
            if route_prefix is None:
                continue
            distance_m = _path_length_m(route_prefix)
            if distance_m + 1.0e-9 < minimum_path_length_m:
                continue
            if (
                clearance_oracle.admits_many(
                    (point,), cf2x.REQUIRED_TERMINAL_CLEARANCE_M
                )
                and point not in tuple(candidate[1] for candidate in candidates)
            ):
                voxel_prefix_candidates.append((distance_m, point))
            if len(voxel_prefix_candidates) >= maximum_candidate_guards:
                break
    combined_candidates = sorted(
        (*candidates, *voxel_prefix_candidates),
        key=lambda row: (-row[0], row[1]),
    )
    candidate_attempts = 0
    for target_distance_m, point in combined_candidates:
        candidate_attempts += 1
        voxel_prefix = (
            _public_route_prefix_from_voxel_chain(
                belief,
                voxel_keys=voxel_keys,
                start_m=path[0],
                progress_point_m=point,
                minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                received_free_support_cache=received_free_support_cache,
            )
            if voxel_keys is not None
            else None
        )
        used_public_voxel_chain = voxel_prefix is not None
        prefix = (
            voxel_prefix
            if voxel_prefix is not None
            else _path_prefix_to_arc_length_m(path, target_distance_m)
        )
        prefix_length_m = _path_length_m(prefix)
        if prefix_length_m + 1.0e-9 < minimum_path_length_m:
            continue
        guarded = cf2x._routed_guard(
            scene_query,
            clearance_oracle,
            (),
            agent_id,
            prefix,
            bounds_min,
            bounds_max,
            allow_public_reroute=False,
            segment_cache=segment_cache,
        )
        if guarded.legal and is_non_alias_exploration_path(guarded.path_m):
            return guarded, {
                "attempted": True,
                "admitted": True,
                "candidate_guards_attempted": candidate_attempts,
                "original_route_length_m": total_length_m,
                "pullback_route_length_m": prefix_length_m,
                "pullback_endpoint_m": point,
                "pullback_source": (
                    "public_voxel_chain" if used_public_voxel_chain else "arc_length"
                ),
                "pullback_retained_fraction": (
                    prefix_length_m / total_length_m if total_length_m > 1.0e-9 else 0.0
                ),
            }
    if (
        combined_candidates
        and total_length_m + 1.0e-9
        >= EXACT_CLEARANCE_GRID_RESCUE_MIN_PATH_LENGTH_M
    ):
        grid_guarded, grid_audit = _exact_clearance_grid_rescue_path(
            scene_query,
            clearance_oracle,
            agent_id,
            path[0],
            tuple(point for _, point in combined_candidates),
            bounds_min,
            bounds_max,
            minimum_path_length_m=minimum_path_length_m,
            maximum_attempts=EXACT_CLEARANCE_GRID_RESCUE_MAX_ATTEMPTS_PER_PATH,
            diagnostic_sink=diagnostic_sink,
            attempt_budget=exact_clearance_grid_rescue_budget,
            requested_route_length_m=total_length_m,
            corridor_path_m=requested_path_m,
            segment_cache=segment_cache,
        )
        exact_clearance_grid_rescue_audit = grid_audit
        if grid_guarded is not None:
            return grid_guarded, grid_audit
    return None, {
        "attempted": True,
        "admitted": False,
        "candidate_guards_attempted": candidate_attempts,
        "original_route_length_m": total_length_m,
        "exact_clearance_grid_rescue": exact_clearance_grid_rescue_audit,
        "reason": "no_terminal_clearance_pullback_guard_admitted",
    }


def _outcome_backtrack_frontier(
    *,
    current_position_m: tuple[float, float, float],
    route: _OutcomeBacktrackRoute,
    arrival_tolerance_m: float,
    occupied_endpoints_m: Sequence[tuple[float, float, float]],
) -> tuple[PublicFrontier, tuple[tuple[float, float, float], ...]] | None:
    """Expose one exact reverse route without promoting it to map evidence.

    The current physical root may be slightly offset from the last outcome
    sample.  Replacing only the first point keeps all historical interior
    vertices intact; source-clearance slack later certifies this short join.
    """

    if arrival_tolerance_m <= 0.0:
        raise ValueError("outcome-backtrack arrival tolerance must be positive")
    if len(route.path_m) < 2:
        raise ValueError("outcome-backtrack source path needs at least two points")
    if math.dist(current_position_m, route.path_m[-1]) > arrival_tolerance_m:
        return None
    path_m = (current_position_m, *tuple(reversed(route.path_m[:-1])))
    if _path_length_m(path_m) + 1.0e-9 < OUTCOME_BACKTRACK_MIN_PATH_M:
        return None
    endpoint = path_m[-1]
    # RouteGuard identifies recovery by the requested endpoint.  Do not create
    # an ambiguous route if a normal viewpoint has exactly the same endpoint.
    if any(math.dist(endpoint, other) <= 1.0e-9 for other in occupied_endpoints_m):
        return None
    frontier = PublicFrontier(
        frontier_id=f"outcome-backtrack-{route.route_id}",
        position_m=endpoint,
        information_gain=0.0,
        traversal_risk=0.0,
        source_agent_id=route.agent_id,
        task_kind="backtrack",
        exclusive_agent_id=route.agent_id,
        viewpoint_kind="outcome_backtrack",
    )
    return frontier, path_m


def _outcome_backtrack_clearance_reuse_audit(
    *,
    current_position_m: tuple[float, float, float],
    route: _OutcomeBacktrackRoute,
    requested_path_m: Sequence[tuple[float, float, float]],
) -> dict[str, object]:
    """Prove a outcome-backtrack stays inside its source clearance envelope.

    The collision USD is immutable within an episode.  Distance to a closed
    static mesh is 1-Lipschitz, so a current root offset no greater than the
    source outcome's observed clearance slack preserves the source clearance
    requirement for the entire reversed path.  This avoids re-querying the
    native triangle index for geometry that the source outcome already
    certified, while retaining a conservative, auditable safety condition.
    """

    source_minimum = float(route.source_minimum_static_mesh_clearance_m)
    required = float(route.source_static_clearance_contract_required_m)
    if not math.isfinite(source_minimum) or not math.isfinite(required) or required <= 0.0:
        return {
            "admitted": False,
            "reason": "source_clearance_outcome_malformed",
        }
    expected_tail = tuple(reversed(route.path_m[:-1]))
    path = tuple(tuple(point) for point in requested_path_m)
    if len(path) < 2 or path[0] != current_position_m or path[1:] != expected_tail:
        return {
            "admitted": False,
            "reason": "source_route_reversal_mismatch",
        }
    endpoint_offset_m = math.dist(current_position_m, route.path_m[-1])
    clearance_slack_m = source_minimum - required
    if clearance_slack_m < -1.0e-9:
        return {
            "admitted": False,
            "reason": "source_outcome_failed_clearance_contract",
            "source_minimum_static_mesh_clearance_m": source_minimum,
            "source_static_clearance_contract_required_m": required,
        }
    admitted = endpoint_offset_m <= clearance_slack_m + 1.0e-9
    return {
        "admitted": admitted,
        "reason": "source_outcome_clearance_slack" if admitted else "endpoint_offset_exceeds_source_clearance_slack",
        "endpoint_offset_m": endpoint_offset_m,
        "source_minimum_static_mesh_clearance_m": source_minimum,
        "source_static_clearance_contract_required_m": required,
        "source_clearance_slack_m": clearance_slack_m,
    }


class _PublicObservationCooldown:
    """Outcome-only, short-lived suppression of repeated empty observations."""

    def __init__(
        self,
        *,
        duration_decisions: int = PUBLIC_NO_GAIN_VIEWPOINT_COOLDOWN_DECISIONS,
        chebyshev_radius_voxels: int = PUBLIC_NO_GAIN_VIEWPOINT_COOLDOWN_CHEBYSHEV_RADIUS_VOXELS,
    ) -> None:
        if duration_decisions < 1:
            raise ValueError("public observation cooldown must last at least one decision")
        if chebyshev_radius_voxels < 0:
            raise ValueError("public observation cooldown radius must be non-negative")
        self.duration_decisions = duration_decisions
        self.chebyshev_radius_voxels = chebyshev_radius_voxels
        self._expires_after_decision: dict[tuple[int, int, int], int] = {}
        self._filtered_viewpoint_count = 0

    def begin_decision(self, decision_index: int) -> None:
        if decision_index < 0:
            raise ValueError("decision index must be non-negative")
        self._expires_after_decision = {
            key: expires_after
            for key, expires_after in self._expires_after_decision.items()
            if expires_after >= decision_index
        }
        self._filtered_viewpoint_count = 0

    def blocks(self, key: tuple[int, int, int], *, decision_index: int) -> bool:
        radius = self.chebyshev_radius_voxels
        blocked = any(
            expires_after >= decision_index
            and all(abs(key[axis] - target[axis]) <= radius for axis in range(3))
            for target, expires_after in self._expires_after_decision.items()
        )
        if blocked:
            self._filtered_viewpoint_count += 1
        return blocked

    def observe_empty_targets(
        self,
        keys: Sequence[tuple[int, int, int]],
        *,
        decision_index: int,
        public_new_free_voxel_count: int,
    ) -> dict[str, object]:
        if public_new_free_voxel_count < 0:
            raise ValueError("public new-free voxel count must be non-negative")
        unique_keys = tuple(sorted(set(keys)))
        applied = public_new_free_voxel_count == 0 and bool(unique_keys)
        if applied:
            expires_after = decision_index + self.duration_decisions
            for key in unique_keys:
                self._expires_after_decision[key] = max(
                    expires_after,
                    self._expires_after_decision.get(key, -1),
                )
        return {
            "source": "actual_public_range_outcome_transition",
            "public_new_free_voxel_count": public_new_free_voxel_count,
            "selected_exploration_target_voxel_keys": [list(key) for key in unique_keys],
            "applied": applied,
            "expires_after_decision": (decision_index + self.duration_decisions if applied else None),
        }

    def audit(self, *, decision_index: int) -> dict[str, object]:
        active = tuple(
            sorted(
                key
                for key, expires_after in self._expires_after_decision.items()
                if expires_after >= decision_index
            )
        )
        return {
            "schema_version": "hm3d-public-empty-observation-cooldown-v1",
            "decision_index": decision_index,
            "duration_decisions": self.duration_decisions,
            "chebyshev_radius_voxels": self.chebyshev_radius_voxels,
            "active_target_voxel_keys": [list(key) for key in active],
            "active_target_count": len(active),
            "filtered_viewpoint_count": self._filtered_viewpoint_count,
            "claim_limit": (
                "Uses only selected public target cells and the previous decision's "
                "fused public new-free transition; it never reads evaluator geometry."
            ),
        }


def _route_guard_record(
    *,
    agent_id: str,
    requested_path_m: tuple[tuple[float, float, float], ...],
    guarded: Any,
    events: list[dict[str, object]],
    public_route_status: str | None = None,
    terminal_pullback: dict[str, object] | None = None,
) -> dict[str, object]:
    raycast_hits = tuple(event for event in events if event.get("event_type") == "raycast_hit")
    clearance_rejections = tuple(
        event for event in events if event.get("event_type") == "static_clearance_rejection"
    )
    hit_classes = Counter(str(event["hit_class"]) for event in raycast_hits)
    hit_prims = Counter(str(event["hit_prim_path"]) for event in raycast_hits)
    clearance_stages = Counter(str(event["stage"]) for event in clearance_rejections)
    exact_clearances = tuple(
        float(event["minimum_static_mesh_clearance_m"])
        for event in clearance_rejections
        if isinstance(event.get("minimum_static_mesh_clearance_m"), (int, float))
    )
    clearance_rejections_by_stage: dict[str, dict[str, object]] = {}
    for stage in sorted(clearance_stages):
        stage_events = tuple(
            event for event in clearance_rejections if str(event["stage"]) == stage
        )
        stage_distances = tuple(
            (float(event["minimum_static_mesh_clearance_m"]), event)
            for event in stage_events
            if isinstance(event.get("minimum_static_mesh_clearance_m"), (int, float))
        )
        minimum = min(stage_distances, default=None, key=lambda row: row[0])
        clearance_rejections_by_stage[stage] = {
            "rejection_count": len(stage_events),
            "required_clearance_m": float(stage_events[0]["required_clearance_m"]),
            "minimum_static_mesh_clearance_m": None if minimum is None else minimum[0],
            "minimum_clearance_position_m": (
                None if minimum is None else minimum[1]["minimum_clearance_position_m"]
            ),
        }
    return {
        "agent_id": agent_id,
        "requested_path_m": requested_path_m,
        "public_route_status": public_route_status,
        "legal": bool(guarded.legal),
        "reason": guarded.reason,
        "guarded_path_m": guarded.path_m,
        "terminal_pullback": terminal_pullback,
        "terminal_pullback_attempted": bool(terminal_pullback is not None),
        "terminal_pullback_admitted": bool(
            terminal_pullback is not None and terminal_pullback.get("admitted") is True
        ),
        "terminal_pullback_failure_reason": (
            None
            if terminal_pullback is None
            else terminal_pullback.get("reason")
        ),
        "blocked_segment_count": len(raycast_hits),
        "hit_class_counts": dict(sorted(hit_classes.items())),
        "hit_prim_path_counts": dict(sorted(hit_prims.items())),
        "first_blocking_hit": raycast_hits[0] if raycast_hits else None,
        "representative_blocking_hits": list(raycast_hits[:32]),
        "omitted_blocking_hit_count": max(0, len(raycast_hits) - 32),
        "clearance_rejection_count": len(clearance_rejections),
        "clearance_rejection_stage_counts": dict(sorted(clearance_stages.items())),
        "clearance_rejections_by_stage": clearance_rejections_by_stage,
        "minimum_rejected_static_mesh_clearance_m": min(exact_clearances, default=None),
        "representative_clearance_rejections": list(clearance_rejections[:32]),
        "omitted_clearance_rejection_count": max(0, len(clearance_rejections) - 32),
    }


def _route_geometry_summary(path: Any) -> dict[str, object]:
    """Summarize commanded 3-D motion without retaining a simulated trace."""

    if not isinstance(path, (list, tuple)) or len(path) < 2:
        raise ValueError("calibration command path must contain at least two points")
    points = [tuple(float(value) for value in point) for point in path]
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("calibration command path contains an invalid point")
    segments = [
        tuple(right[axis] - left[axis] for axis in range(3))
        for left, right in zip(points[:-1], points[1:], strict=True)
    ]
    lengths = [math.sqrt(sum(value * value for value in segment)) for segment in segments]
    horizontal_m = sum(math.hypot(segment[0], segment[1]) for segment in segments)
    vertical_m = sum(abs(segment[2]) for segment in segments)
    turn_angles_deg: list[float] = []
    moving_segments = [
        (segment, length)
        for segment, length in zip(segments, lengths, strict=True)
        if length > 1.0e-9
    ]
    for (left, left_length), (right, right_length) in zip(
        moving_segments[:-1], moving_segments[1:], strict=True
    ):
        cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_length * right_length)
        turn_angles_deg.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    classes: list[str] = []
    if horizontal_m > 0.05:
        classes.append("horizontal")
    if vertical_m > 0.05:
        classes.append("vertical")
    if any(angle > 15.0 for angle in turn_angles_deg):
        classes.append("turn")
    if sum(lengths) <= 0.05:
        classes.append("stationary")
    return {
        "command_path_length_m": sum(lengths),
        "horizontal_path_length_m": horizontal_m,
        "vertical_path_length_m": vertical_m,
        "waypoint_segment_count": len(segments),
        "turn_count_over_15deg": sum(angle > 15.0 for angle in turn_angles_deg),
        "maximum_turn_angle_deg": max(turn_angles_deg, default=0.0),
        "route_classes": classes,
    }


def _decision_execution_calibration_summary(backend: Any, *, decision_id: str) -> dict[str, object]:
    """Keep only the physical evidence needed to recalibrate candidate timing."""

    diagnostics = backend.engineering_diagnostics
    if not isinstance(diagnostics, dict):
        raise RuntimeError("CF2X backend omitted engineering diagnostics")
    tracking = diagnostics.get("controller_tracking")
    clearance = diagnostics.get("static_trace_clearance")
    agents = diagnostics.get("agents")
    if not isinstance(tracking, dict) or not isinstance(clearance, dict):
        backend_error = diagnostics.get("backend_exception")
        raise RuntimeError(
            "CF2X backend omitted timing or clearance evidence: "
            f"{backend_error if backend_error is not None else 'no backend exception recorded'}"
        )
    if not isinstance(agents, list) or not agents:
        raise RuntimeError("CF2X backend omitted per-agent timing evidence")
    agent_fields = (
        "agent_id",
        "command_path_m",
        "transit_completed",
        "transit_completed_at_s",
        "transit_attempted",
        "transit_attempt_actual_end_s",
        "transit_execution_deadline_s",
        "transit_failure_reason",
        "observation_started_at_s",
        "observation_completed_at_s",
        "transit_collision",
        "transit_out_of_bounds",
        "observation_collision",
        "observation_out_of_bounds",
        "minimum_static_mesh_clearance_m",
        "minimum_clearance_position_m",
        "static_clearance_contract_required_m",
        "static_clearance_contract_violation",
        "waypoint_settle_speed_mps",
        "waypoint_transitions",
        "maximum_linear_speed_mps",
        "maximum_linear_acceleration_mps2",
        "controller_tracking_telemetry_hz",
        "controller_tracking_samples",
        "realized_transit_path_length_m",
        "next_unreached_waypoint_index",
    )
    selected_agents: list[dict[str, object]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            raise RuntimeError("CF2X agent timing evidence is malformed")
        selected = {key: agent.get(key) for key in agent_fields}
        selected["route_geometry"] = _route_geometry_summary(agent.get("command_path_m"))
        selected_agents.append(selected)
    selected_clearance = {
        key: clearance.get(key)
        for key in (
            "method",
            "scope",
            "static_clearance_contract_required_m",
            "static_clearance_contract_passed",
        )
    }
    summary = {
        "schema_version": "hm3d-cf2x-decision-execution-calibration-v1",
        "decision_id": decision_id,
        "backend_id": backend.backend_id,
        "evidence_class": backend.evidence_class,
        "token_authorization_duration_s": diagnostics.get("token_authorization_duration_s"),
        "execution_deadline_s": diagnostics.get("execution_deadline_s"),
        "execution_elapsed_physics_s": diagnostics.get("execution_elapsed_physics_s"),
        "action_completion_mode": diagnostics.get("action_completion_mode"),
        "calibration_only_timeout_probe": diagnostics.get("calibration_only_timeout_probe"),
        "controller_tracking": tracking,
        "static_trace_clearance": selected_clearance,
        "agents": selected_agents,
    }
    visualization_trace = diagnostics.get("physics_visualization_trace")
    if isinstance(visualization_trace, dict) and visualization_trace.get("sample_hz") is not None:
        if visualization_trace.get("purpose") != "engineering_visual_audit_only":
            raise RuntimeError("CF2X visualization trace has an invalid purpose")
        summary["physics_visualization_trace"] = visualization_trace
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _selected_exploration_target_keys(
    manifest: CandidateFragmentManifest,
    belief: SparseVoxelBelief,
) -> tuple[tuple[int, int, int], ...]:
    """Return current, public-map target cells for non-stationary assignments."""

    keys: list[tuple[int, int, int]] = []
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        features = dict(fragment.type_signature.public_features)
        role = features.get("assignment_role")
        if role not in {"explore", "backtrack", "hold"}:
            raise RuntimeError("selected public transit omits its assignment role")
        if role == "explore":
            if not is_non_alias_exploration_path(fragment.path):
                raise RuntimeError(
                    "selected exploration transit aliases the current settled endpoint"
                )
            keys.append(belief.world_to_voxel(fragment.path[-1]))
    return tuple(sorted(set(keys)))


def _decision_stationarity_supervision(
    manifest: CandidateFragmentManifest,
    execution_calibration: dict[str, object],
) -> dict[str, object]:
    """Audit every agent-second as transit, dwell, synchronization wait or hold.

    This is an execution outcome audit, not a control input.  Selection remains
    fair because every method receives the same candidate movement floor and
    cooldown.  The report makes any residual stationary time actionable for
    common route generation and assignment instead of hiding it in an episode
    duration.
    """

    elapsed = execution_calibration.get("execution_elapsed_physics_s")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise RuntimeError("execution calibration omits a positive physical elapsed time")
    elapsed_s = float(elapsed)
    transit_by_agent = {
        fragment.agent_id: fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    }
    agents = execution_calibration.get("agents")
    if not isinstance(agents, list) or set(transit_by_agent) != {
        row.get("agent_id") for row in agents if isinstance(row, dict)
    }:
        raise RuntimeError("stationarity audit cannot align selected and executed agents")

    rows: list[dict[str, object]] = []
    violations: list[str] = []
    for raw_agent in agents:
        assert isinstance(raw_agent, dict)
        agent_id = raw_agent["agent_id"]
        if not isinstance(agent_id, str):
            raise RuntimeError("stationarity audit agent identifier is malformed")
        fragment = transit_by_agent[agent_id]
        features = dict(fragment.type_signature.public_features)
        role = features.get("assignment_role")
        hold_reason = features.get("hold_reason")
        if role not in {"explore", "backtrack", "hold"}:
            raise RuntimeError("stationarity audit transit omits its assignment role")
        geometry = raw_agent.get("route_geometry")
        if not isinstance(geometry, dict):
            raise RuntimeError("stationarity audit agent omits route geometry")
        planned_length_m = float(geometry["command_path_length_m"])
        realised_length_m = float(raw_agent["realized_transit_path_length_m"])
        transit_completed_at_s = raw_agent.get("transit_completed_at_s")
        transit_phase_s = (
            elapsed_s
            if not isinstance(transit_completed_at_s, (int, float))
            else min(elapsed_s, max(0.0, float(transit_completed_at_s)))
        )
        observation_started_at_s = raw_agent.get("observation_started_at_s")
        observation_completed_at_s = raw_agent.get("observation_completed_at_s")
        dwell_s = 0.0
        synchronization_wait_s = 0.0
        if isinstance(observation_started_at_s, (int, float)) and isinstance(
            observation_completed_at_s, (int, float)
        ):
            observation_start_s = min(elapsed_s, max(0.0, float(observation_started_at_s)))
            observation_end_s = min(elapsed_s, max(observation_start_s, float(observation_completed_at_s)))
            dwell_s = observation_end_s - observation_start_s
            synchronization_wait_s = elapsed_s - observation_end_s
        elif raw_agent.get("transit_completed") is True:
            violations.append(f"{agent_id}:completed_transit_without_observation_timing")
        meaningful_planned_exploration = role == "explore" and is_non_alias_exploration_path(
            fragment.path
        )
        # The realised trajectory has no planned endpoint sequence here, so
        # this only rejects a numerical no-motion outcome. It deliberately
        # does not reintroduce a fixed 0.50 m eligibility threshold.
        meaningful_realised_exploration = (
            role == "explore"
            and raw_agent.get("transit_completed") is True
            and realised_length_m > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M
        )
        valid_hold = (
            role != "hold"
            or (
                hold_reason
                in {
                    "relay_required",
                    "collision_avoidance",
                    "collision_avoidance_recovery",
                    "no_reachable_viewpoint",
                    "waiting_for_team_completion",
                    "controller_failure",
                }
                and planned_length_m <= 1.0e-9
            )
        )
        meaningful_planned_backtrack = (
            role == "backtrack"
            and planned_length_m + 1.0e-9 >= OUTCOME_BACKTRACK_MIN_PATH_M
        )
        meaningful_realised_backtrack = (
            role == "backtrack"
            and raw_agent.get("transit_completed") is True
            and realised_length_m + 1.0e-9 >= OUTCOME_BACKTRACK_MIN_PATH_M / 2.0
        )
        if role == "explore" and not meaningful_planned_exploration:
            violations.append(f"{agent_id}:subthreshold_planned_exploration")
        if role == "explore" and raw_agent.get("transit_completed") is True and not meaningful_realised_exploration:
            violations.append(f"{agent_id}:subthreshold_realised_exploration")
        if role == "backtrack" and not meaningful_planned_backtrack:
            violations.append(f"{agent_id}:subthreshold_planned_backtrack")
        if (
            role == "backtrack"
            and raw_agent.get("transit_completed") is True
            and not meaningful_realised_backtrack
        ):
            violations.append(f"{agent_id}:subthreshold_realised_backtrack")
        if not valid_hold:
            violations.append(f"{agent_id}:invalid_hold_semantics")
        rows.append(
            {
                "agent_id": agent_id,
                "assignment_role": role,
                "hold_reason": hold_reason,
                "planned_transit_path_length_m": planned_length_m,
                "realised_transit_path_length_m": realised_length_m,
                "transit_phase_s": transit_phase_s,
                "observation_dwell_s": dwell_s,
                "post_observation_synchronization_wait_s": synchronization_wait_s,
                "meaningful_planned_exploration": meaningful_planned_exploration,
                "meaningful_realised_exploration": meaningful_realised_exploration,
                "meaningful_planned_backtrack": meaningful_planned_backtrack,
                "meaningful_realised_backtrack": meaningful_realised_backtrack,
                "valid_hold_semantics": valid_hold,
            }
        )
    total_agent_seconds = elapsed_s * len(rows)
    return {
        "schema_version": "hm3d-decision-stationarity-supervision-v1",
        "execution_elapsed_physics_s": elapsed_s,
        "total_agent_seconds": total_agent_seconds,
        "exploration_transit_phase_agent_seconds": sum(
            float(row["transit_phase_s"])
            for row in rows
            if row["assignment_role"] == "explore"
        ),
        "backtrack_transit_phase_agent_seconds": sum(
            float(row["transit_phase_s"])
            for row in rows
            if row["assignment_role"] == "backtrack"
        ),
        "observation_dwell_agent_seconds": sum(float(row["observation_dwell_s"]) for row in rows),
        "synchronization_wait_agent_seconds": sum(
            float(row["post_observation_synchronization_wait_s"]) for row in rows
        ),
        "intentional_hold_agent_seconds": sum(
            elapsed_s for row in rows if row["assignment_role"] == "hold"
        ),
        "synchronization_wait_fraction": (
            sum(float(row["post_observation_synchronization_wait_s"]) for row in rows)
            / total_agent_seconds
        ),
        "status": (
            "STATIONARITY_SUPERVISION_ADMITTED"
            if not violations
            else "STATIONARITY_SUPERVISION_NOT_ADMITTED"
        ),
        "violations": violations,
        "agents": rows,
        "claim_limit": (
            "Minimum dwell, auditable holds and post-observation synchronization wait "
            "are reported separately. Synchronization wait is not relabelled as exploration."
        ),
    }


def _vertical_opportunity_summary(
    frontiers: Sequence[PublicFrontier],
    positions: Sequence[tuple[float, float, float]],
    execution_calibration: dict[str, object],
    candidate_route_catalog: dict[str, object],
    *,
    threshold_m: float = VERTICAL_OPPORTUNITY_THRESHOLD_M,
) -> dict[str, object]:
    """Report vertical access from raw public exposure through real completion.

    A high frontier anchor alone is not a vertical opportunity: the observation
    endpoint must first be connected through received FREE space, then pass the
    static guard, then appear in a team-feasible candidate.  Keep these stages
    separate so the engineering smoke cannot claim vertical capability from an
    unreachable endpoint or from a cluster centroid behind a wall.
    """

    starts = {f"uav{index}": tuple(position) for index, position in enumerate(positions)}
    raw_frontier_deltas: list[float] = []
    for frontier in frontiers:
        if frontier.task_kind != "explore":
            continue
        if frontier.source_agent_id is None or frontier.source_agent_id not in starts:
            continue
        start = starts[frontier.source_agent_id]
        # Deliberately use the candidate endpoint, never task_anchor_m. The
        # latter is a frontier-cluster descriptor and may be far outside the
        # currently reachable public FREE component.
        raw_frontier_deltas.append(frontier.position_m[2] - start[2])

    catalog_agents = candidate_route_catalog.get("agents")
    if not isinstance(catalog_agents, list):
        raise RuntimeError("candidate route catalog omits per-agent vertical diagnostics")

    stage_deltas: dict[str, list[float]] = {
        "public_free_path_reachable": [],
        "static_guard_admitted": [],
        "team_feasible": [],
        "selected": [],
    }
    selected_vertical_deltas_by_agent: dict[str, float] = {}
    selected_edge_contract_issues: list[dict[str, object]] = []
    duplicate_selected_vertical_agent_ids: list[str] = []
    for agent_row in catalog_agents:
        if not isinstance(agent_row, dict):
            raise RuntimeError("candidate route catalog agent row is malformed")
        agent_id = agent_row.get("agent_id")
        if not isinstance(agent_id, str) or agent_id not in starts:
            raise RuntimeError("candidate route catalog has an unknown agent")
        edge_rows = agent_row.get("frontier_edges")
        if not isinstance(edge_rows, list):
            raise RuntimeError("candidate route catalog omits frontier edges")
        for edge in edge_rows:
            if not isinstance(edge, dict) or edge.get("task_kind") != "explore":
                continue
            endpoint = edge.get("endpoint_m")
            if not isinstance(endpoint, (list, tuple)) or len(endpoint) != 3:
                raise RuntimeError("candidate route catalog edge omits a valid endpoint")
            delta = float(endpoint[2]) - starts[agent_id][2]
            if abs(delta) + 1.0e-9 < threshold_m:
                continue
            route_status = edge.get("public_route_status")
            public_free_path = route_status in {
                "admitted",
                "revalidated_public_access_plan",
                # Terminal pullback and exact-clearance grid rescue are both
                # derived from an already received-free public route and are
                # then rechecked by the same static guard. They remain one
                # stage here, not a relaxation of collision safety.
                "terminal_clearance_pullback",
                "exact_clearance_grid_route",
            }
            # This is deliberately stricter than a bare static clearance verdict:
            # the route must also be non-alias and fit in the current physical
            # decision window before it is a usable individual exploration edge.
            static_guard_admitted = (
                edge.get("individual_exploration_edge_admitted") is True
            )
            team_feasible = edge.get("appears_in_feasible_team_candidate") is True
            selected = edge.get("selected") is True
            contract_issue = None
            if selected and not team_feasible:
                contract_issue = "selected_vertical_edge_absent_from_team_feasible"
            elif team_feasible and not static_guard_admitted:
                contract_issue = "team_feasible_vertical_edge_lacks_static_guard_admission"
            elif static_guard_admitted and not public_free_path:
                contract_issue = "static_guard_admitted_vertical_edge_lacks_public_free_path"
            if contract_issue is not None:
                selected_edge_contract_issues.append(
                    {
                        "agent_id": agent_id,
                        "frontier_id": edge.get("frontier_id"),
                        "issue": contract_issue,
                    }
                )
            if public_free_path:
                stage_deltas["public_free_path_reachable"].append(delta)
            if static_guard_admitted:
                stage_deltas["static_guard_admitted"].append(delta)
            if team_feasible:
                stage_deltas["team_feasible"].append(delta)
            if selected and contract_issue is None:
                prior_delta = selected_vertical_deltas_by_agent.get(agent_id)
                if prior_delta is not None:
                    duplicate_selected_vertical_agent_ids.append(agent_id)
                else:
                    stage_deltas["selected"].append(delta)
                    selected_vertical_deltas_by_agent[agent_id] = delta

    agents = execution_calibration.get("agents")
    if not isinstance(agents, list):
        raise RuntimeError("execution calibration omits agents for vertical audit")
    realised_deltas: list[float] = []
    completed_upward_count = 0
    completed_downward_count = 0
    executed_selected_vertical_agents: set[str] = set()
    execution_path_inconsistent_agent_ids: list[str] = []
    execution_telemetry_issues: list[dict[str, object]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            raise RuntimeError("execution calibration agent is malformed")
        agent_id = agent.get("agent_id")
        if not isinstance(agent_id, str):
            raise RuntimeError("execution calibration agent omits its ID")
        selected_delta = selected_vertical_deltas_by_agent.get(agent_id)
        # The executor may contain a recovery, backtrack, or hold transit.  It
        # is physically meaningful, but it is not evidence that the selected
        # exploration policy reached a vertical frontier.
        if selected_delta is None:
            continue
        executed_selected_vertical_agents.add(agent_id)
        path = agent.get("command_path_m")
        if not isinstance(path, (list, tuple)) or len(path) < 2:
            execution_telemetry_issues.append(
                {"agent_id": agent_id, "issue": "missing_command_path"}
            )
            continue
        delta = float(path[-1][2]) - float(path[0][2])
        if (selected_delta >= threshold_m and delta < threshold_m) or (
            selected_delta <= -threshold_m and delta > -threshold_m
        ):
            execution_path_inconsistent_agent_ids.append(agent_id)
            continue
        samples = agent.get("controller_tracking_samples")
        if not isinstance(samples, list) or not samples:
            execution_telemetry_issues.append(
                {"agent_id": agent_id, "issue": "missing_realised_tracking_samples"}
            )
            continue
        final_sample = samples[-1]
        if not isinstance(final_sample, dict):
            execution_telemetry_issues.append(
                {"agent_id": agent_id, "issue": "malformed_tracking_sample"}
            )
            continue
        realised_position = final_sample.get("post_step_position_m")
        if not isinstance(realised_position, (list, tuple)) or len(realised_position) != 3:
            execution_telemetry_issues.append(
                {"agent_id": agent_id, "issue": "missing_realised_position"}
            )
            continue
        realised_delta = float(realised_position[2]) - float(path[0][2])
        realised_deltas.append(realised_delta)
        completed_in_commanded_direction = (
            delta >= threshold_m and realised_delta >= threshold_m
        ) or (delta <= -threshold_m and realised_delta <= -threshold_m)
        if completed_in_commanded_direction and agent.get("transit_completed") is True:
            if delta >= threshold_m:
                completed_upward_count += 1
            else:
                completed_downward_count += 1

    missing_execution_agents = set(selected_vertical_deltas_by_agent) - executed_selected_vertical_agents
    if missing_execution_agents:
        execution_telemetry_issues.extend(
            {
                "agent_id": agent_id,
                "issue": "selected_vertical_edge_missing_execution_telemetry",
            }
            for agent_id in sorted(missing_execution_agents)
        )

    def stage_counts(deltas: Sequence[float]) -> dict[str, int]:
        return {
            "upward": sum(delta >= threshold_m for delta in deltas),
            "downward": sum(delta <= -threshold_m for delta in deltas),
        }

    raw_counts = stage_counts(raw_frontier_deltas)
    public_free_counts = stage_counts(stage_deltas["public_free_path_reachable"])
    static_guard_counts = stage_counts(stage_deltas["static_guard_admitted"])
    team_feasible_counts = stage_counts(stage_deltas["team_feasible"])
    selected_counts = stage_counts(stage_deltas["selected"])
    # Keep directional extrema independent. A completed descending transit is
    # evidence of downward access, not a negative "upward maximum".
    raw_upward_deltas = [delta for delta in raw_frontier_deltas if delta > 0.0]
    raw_downward_deltas = [delta for delta in raw_frontier_deltas if delta < 0.0]
    realised_upward_deltas = [delta for delta in realised_deltas if delta > 0.0]
    realised_downward_deltas = [delta for delta in realised_deltas if delta < 0.0]
    return {
        "schema_version": "hm3d-p07-vertical-opportunity-v3",
        "vertical_displacement_threshold_m": threshold_m,
        "raw_exposed_upward_frontier_count": raw_counts["upward"],
        "raw_exposed_downward_frontier_count": raw_counts["downward"],
        "maximum_raw_exposed_upward_endpoint_delta_m": max(raw_upward_deltas, default=0.0),
        "maximum_raw_exposed_downward_endpoint_delta_m": min(raw_downward_deltas, default=0.0),
        "public_free_path_reachable_upward_edge_count": public_free_counts["upward"],
        "public_free_path_reachable_downward_edge_count": public_free_counts["downward"],
        "static_guard_admitted_upward_edge_count": static_guard_counts["upward"],
        "static_guard_admitted_downward_edge_count": static_guard_counts["downward"],
        "team_feasible_upward_edge_count": team_feasible_counts["upward"],
        "team_feasible_downward_edge_count": team_feasible_counts["downward"],
        "selected_upward_edge_count": selected_counts["upward"],
        "selected_downward_edge_count": selected_counts["downward"],
        "selected_vertical_explore_agent_count": len(selected_vertical_deltas_by_agent),
        "completed_upward_explore_agent_count": completed_upward_count,
        "completed_downward_explore_agent_count": completed_downward_count,
        "completed_vertical_agent_count": completed_upward_count + completed_downward_count,
        "maximum_realised_upward_displacement_m": max(realised_upward_deltas, default=0.0),
        "maximum_realised_downward_displacement_m": min(realised_downward_deltas, default=0.0),
        "selected_edge_contract_issues": selected_edge_contract_issues,
        "duplicate_selected_vertical_agent_ids": duplicate_selected_vertical_agent_ids,
        "execution_path_inconsistent_agent_ids": execution_path_inconsistent_agent_ids,
        "execution_telemetry_issues": execution_telemetry_issues,
        "claim_limit": (
            "Raw exposure is a public-frontier endpoint count only. Vertical execution "
            "requires a selected team-feasible edge and a realised completed transit."
        ),
    }


def _candidate_role_summary(
    pool: Sequence[CandidateFragmentManifest],
    *,
    selected_manifest_hash: str,
) -> list[dict[str, object]]:
    """Expose active-team capacity and explicit non-relay hold reasons."""

    rows: list[dict[str, object]] = []
    for candidate in pool:
        roles: dict[str, str] = {}
        hold_reasons: dict[str, str] = {}
        viewpoint_kinds: dict[str, str] = {}
        path_lengths_m: dict[str, float] = {}
        expected_gain_proxy_by_agent: dict[str, float] = {}
        task_reservation_match_by_agent: dict[str, bool] = {}
        task_reservation_heading_alignment_by_agent: dict[str, float] = {}
        task_reservation_switch_cost_by_agent: dict[str, float] = {}
        predicted_makespans_s: list[float] = []
        for fragment in candidate.fragments:
            if fragment.type_signature.fragment_type != "transit":
                continue
            public_features = dict(fragment.type_signature.public_features)
            role = public_features.get("assignment_role")
            if role not in {"explore", "backtrack", "hold"}:
                raise RuntimeError("public candidate transit omits its assignment role")
            roles[fragment.agent_id] = str(role)
            viewpoint_kind = public_features.get("viewpoint_kind")
            if viewpoint_kind not in {
                "observation",
                "route_progress",
                "region_access",
                "outcome_backtrack",
                "collision_avoidance_recovery",
                "hold",
            }:
                raise RuntimeError("public candidate transit omits its viewpoint kind")
            viewpoint_kinds[fragment.agent_id] = str(viewpoint_kind)
            if role == "hold":
                hold_reason = public_features.get("hold_reason")
                if hold_reason not in {
                    "relay_required",
                    "collision_avoidance",
                    "collision_avoidance_recovery",
                    "no_reachable_viewpoint",
                    "waiting_for_team_completion",
                    "controller_failure",
                }:
                    raise RuntimeError("public hold transit omits an auditable hold reason")
                hold_reasons[fragment.agent_id] = str(hold_reason)
            path_lengths_m[fragment.agent_id] = _path_length_m(fragment.path)
            expected_gain_proxy_by_agent[fragment.agent_id] = float(
                public_features.get("expected_public_gain_proxy", 0.0)
            )
            task_reservation_match_by_agent[fragment.agent_id] = bool(
                public_features.get("task_reservation_matched", False)
            )
            task_reservation_heading_alignment_by_agent[fragment.agent_id] = float(
                public_features.get("task_reservation_heading_alignment", 0.0)
            )
            task_reservation_switch_cost_by_agent[fragment.agent_id] = float(
                public_features.get("task_reservation_switch_cost", 0.0)
            )
            predicted_makespans_s.append(
                float(public_features.get("predicted_physical_makespan_s", 0.0))
            )
        if not roles:
            raise RuntimeError("public candidate contains no transit assignment")
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feasible": candidate.feasible,
                "selected": candidate.manifest_hash == selected_manifest_hash,
                "minimum_route_tube_separation_m": (
                    _manifest_route_tube_separation_m(candidate)
                ),
                "moving_explorer_count": sum(role == "explore" for role in roles.values()),
                "backtrack_count": sum(role == "backtrack" for role in roles.values()),
                "backtrack_agent_ids": sorted(
                    agent_id for agent_id, role in roles.items() if role == "backtrack"
                ),
                "moving_agent_count": sum(
                    role in {"explore", "backtrack"} for role in roles.values()
                ),
                "hold_count": sum(role == "hold" for role in roles.values()),
                "hold_agent_ids": sorted(
                    agent_id for agent_id, role in roles.items() if role == "hold"
                ),
                "hold_reasons_by_agent": dict(sorted(hold_reasons.items())),
                "viewpoint_kinds_by_agent": dict(sorted(viewpoint_kinds.items())),
                "quality_hint": candidate.quality_hint,
                "cost_hint": candidate.cost_hint,
                "planned_path_length_m_by_agent": dict(sorted(path_lengths_m.items())),
                "team_planned_path_length_m": sum(path_lengths_m.values()),
                "expected_public_gain_proxy_by_agent": dict(
                    sorted(expected_gain_proxy_by_agent.items())
                ),
                "team_expected_public_gain_proxy": sum(expected_gain_proxy_by_agent.values()),
                "task_reservation_match_by_agent": dict(
                    sorted(task_reservation_match_by_agent.items())
                ),
                "task_reservation_heading_alignment_by_agent": dict(
                    sorted(task_reservation_heading_alignment_by_agent.items())
                ),
                "task_reservation_switch_cost_by_agent": dict(
                    sorted(task_reservation_switch_cost_by_agent.items())
                ),
                "task_reservation_switch_cost_total": sum(
                    task_reservation_switch_cost_by_agent.values()
                ),
                "predicted_physical_makespan_s": max(predicted_makespans_s, default=0.0),
            }
        )
    return rows


def _feasible_all_active_candidate_count(
    candidate_roles: Sequence[Mapping[str, object]],
    *,
    fleet_size: int,
) -> int:
    """Count candidates that command every agent and survive the joint guard."""

    return sum(
        bool(row.get("feasible") is True and row.get("moving_explorer_count") == fleet_size)
        for row in candidate_roles
    )


def _per_agent_candidate_edge_diagnostics(
    state: PublicSearchState,
    belief: SparseVoxelBelief,
    route_guard_records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Explain whether each robot had a public, time-feasible exploration edge.

    Candidate selection must not turn a route-generation limitation into an
    unexplained ``hold``.  The feasibility matcher intentionally omits edges
    whose straight-line lower bound cannot fit the current action deadline;
    it also calls the route guard repeatedly while forming manifests.  This
    outcome-only summary separates those two facts from rejected public paths.
    It consumes no evaluator geometry and has no authority over selection.
    """

    frontier_by_endpoint = {
        tuple(frontier.position_m): frontier for frontier in state.frontiers
    }
    frontier_by_id = {frontier.frontier_id: frontier for frontier in state.frontiers}
    records_by_agent: dict[str, dict[str, dict[str, object]]] = {
        agent.agent_id: {} for agent in state.agents
    }
    for record in route_guard_records:
        agent_id = record.get("agent_id")
        requested_path = record.get("requested_path_m")
        if (
            not isinstance(agent_id, str)
            or agent_id not in records_by_agent
            or not isinstance(requested_path, (list, tuple))
            or len(requested_path) < 2
        ):
            continue
        start = tuple(requested_path[0])
        endpoint = tuple(requested_path[-1])
        if len(start) != 3 or len(endpoint) != 3 or math.dist(start, endpoint) <= 1.0e-9:
            continue
        frontier = frontier_by_endpoint.get(endpoint)
        if frontier is not None:
            prior = records_by_agent[agent_id].get(frontier.frontier_id)
            # A cached guard result is still an observed route opportunity.
            # Prefer the original query when both forms are present so the
            # diagnostic retains the first blocking-hit details without
            # counting duplicate cache lookups as separate edges.
            if prior is None or (
                bool(prior.get("cache_hit")) and not bool(record.get("cache_hit"))
            ):
                records_by_agent[agent_id][frontier.frontier_id] = record

    deadline_s = state.decision_start_s + state.decision_duration_s
    rows: list[dict[str, object]] = []
    for agent in state.agents:
        lower_bound_rejected = 0
        lower_bound_admitted = 0
        for frontier in state.frontiers:
            if frontier.task_kind != "explore":
                continue
            lower_bound_s = (
                state.decision_start_s
                + state.transit_timing_model.estimate_seconds(
                    (agent.position_m, frontier.position_m)
                )
                + state.observe_dwell_s
            )
            if lower_bound_s <= deadline_s + 1.0e-9:
                lower_bound_admitted += 1
            else:
                lower_bound_rejected += 1

        unique_records = tuple(
            record
            for frontier_id, record in records_by_agent[agent.agent_id].items()
            if frontier_by_id[frontier_id].task_kind == "explore"
        )
        recovery_unique_records = tuple(
            record
            for frontier_id, record in records_by_agent[agent.agent_id].items()
            if frontier_by_id[frontier_id].task_kind == "backtrack"
        )
        legal_records = tuple(record for record in unique_records if record.get("legal") is True)
        reason_counts = Counter(
            str(record.get("reason") or "admitted") for record in unique_records
        )
        clearance_stage_counts = Counter(
            str(stage)
            for record in unique_records
            for stage, count in dict(record.get("clearance_rejection_stage_counts") or {}).items()
            for _ in range(int(count))
        )
        clearance_rejections_by_stage: dict[str, dict[str, object]] = {}
        for record in unique_records:
            stage_rows = record.get("clearance_rejections_by_stage")
            if not isinstance(stage_rows, dict):
                continue
            for stage, raw_summary in stage_rows.items():
                if not isinstance(stage, str) or not isinstance(raw_summary, dict):
                    continue
                count = raw_summary.get("rejection_count")
                required = raw_summary.get("required_clearance_m")
                minimum = raw_summary.get("minimum_static_mesh_clearance_m")
                position = raw_summary.get("minimum_clearance_position_m")
                if not isinstance(count, int) or count < 1:
                    raise RuntimeError("clearance rejection summary has an invalid count")
                if not isinstance(required, (int, float)):
                    raise RuntimeError("clearance rejection summary omits its threshold")
                summary = clearance_rejections_by_stage.setdefault(
                    stage,
                    {
                        "rejection_count": 0,
                        "required_clearance_m": float(required),
                        "minimum_static_mesh_clearance_m": None,
                        "minimum_clearance_position_m": None,
                    },
                )
                if not math.isclose(
                    float(summary["required_clearance_m"]), float(required), abs_tol=1.0e-12
                ):
                    raise RuntimeError("clearance rejection stage changed its threshold")
                summary["rejection_count"] = int(summary["rejection_count"]) + count
                if isinstance(minimum, (int, float)) and (
                    summary["minimum_static_mesh_clearance_m"] is None
                    or float(minimum) < float(summary["minimum_static_mesh_clearance_m"])
                ):
                    summary["minimum_static_mesh_clearance_m"] = float(minimum)
                    summary["minimum_clearance_position_m"] = position
        public_route_status_counts = Counter(
            str(record.get("public_route_status") or "not_applicable") for record in unique_records
        )
        terminal_pullback_attempted = sum(
            bool(record.get("terminal_pullback_attempted")) for record in unique_records
        )
        terminal_pullback_admitted = sum(
            bool(record.get("terminal_pullback_admitted")) for record in unique_records
        )
        terminal_pullback_failure_reason_counts = Counter(
            str(record.get("terminal_pullback_failure_reason") or "not_attempted")
            for record in unique_records
        )
        guard_legal_distances: list[tuple[float, str, bool]] = []
        backtrack_records: list[tuple[float, str, bool, str]] = []
        for frontier_id, record in records_by_agent[agent.agent_id].items():
            frontier = frontier_by_id[frontier_id]
            guarded_path = record.get("guarded_path_m")
            if not isinstance(guarded_path, (list, tuple)) or len(guarded_path) < 2:
                continue
            length_m = sum(
                math.dist(tuple(left), tuple(right))
                for left, right in zip(guarded_path, guarded_path[1:], strict=False)
            )
            if frontier.task_kind == "explore":
                if record.get("legal") is True:
                    guard_legal_distances.append(
                        (
                            length_m,
                            frontier_id,
                            is_non_alias_exploration_path(
                                tuple(tuple(point) for point in guarded_path)
                            ),
                        )
                    )
            else:
                backtrack_records.append(
                    (length_m, frontier_id, bool(record.get("legal")), str(record.get("reason") or ""))
                )
        meaningful_guard_legal_distances = tuple(
            row
            for row in guard_legal_distances
            if row[0] >= MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
        )
        non_alias_guard_legal_distances = tuple(
            row for row in guard_legal_distances if row[2]
        )
        legacy_subhalfmetre_guard_legal = tuple(
            row
            for row in guard_legal_distances
            if row[0] < MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
        )
        endpoint_alias_rejected = tuple(
            row for row in guard_legal_distances if not row[2]
        )
        nearest_guard_legal = min(guard_legal_distances, default=None)
        nearest_meaningful_guard_legal = min(meaningful_guard_legal_distances, default=None)
        start_anchor = _nearest_public_free_key(
            belief,
            agent.position_m,
            maximum_distance_m=belief.resolution_m * 1.5,
        )
        rows.append(
            {
                "agent_id": agent.agent_id,
                "public_free_start_anchor_present": start_anchor is not None,
                "public_frontier_edge_count": sum(
                    frontier.task_kind == "explore" for frontier in state.frontiers
                ),
                "time_lower_bound_admitted_edge_count": lower_bound_admitted,
                "time_lower_bound_rejected_edge_count": lower_bound_rejected,
                "route_guard_unique_frontier_query_count": len(unique_records),
                "outcome_backtrack_route_guard_query_count": len(recovery_unique_records),
                "guard_legal_frontier_edge_count": len(guard_legal_distances),
                "meaningful_guard_legal_exploration_edge_count": len(
                    meaningful_guard_legal_distances
                ),
                "non_alias_guard_legal_exploration_edge_count": len(
                    non_alias_guard_legal_distances
                ),
                "endpoint_alias_rejected_edge_count": len(endpoint_alias_rejected),
                # This legacy 0.50 m statistic remains useful for diagnosing
                # compact public maps, but it no longer determines ordinary
                # observation-edge eligibility.
                "legacy_subhalfmetre_guard_legal_exploration_edge_count": len(
                    legacy_subhalfmetre_guard_legal
                ),
                "legal_frontier_edge_count": len(legal_records),
                "rejected_frontier_edge_count": len(unique_records) - len(legal_records),
                "route_guard_reason_counts": dict(sorted(reason_counts.items())),
                "clearance_rejection_stage_counts": dict(sorted(clearance_stage_counts.items())),
                "clearance_rejections_by_stage": dict(
                    sorted(clearance_rejections_by_stage.items())
                ),
                "public_route_status_counts": dict(sorted(public_route_status_counts.items())),
                "terminal_pullback_attempted_count": terminal_pullback_attempted,
                "terminal_pullback_admitted_count": terminal_pullback_admitted,
                "terminal_pullback_failure_reason_counts": dict(
                    sorted(terminal_pullback_failure_reason_counts.items())
                ),
                # Retain the legacy names for old outcome readers.  New code
                # must use the explicit guard/legal names above.
                "nearest_legal_frontier_id": (
                    None if nearest_guard_legal is None else nearest_guard_legal[1]
                ),
                "nearest_legal_path_length_m": (
                    None if nearest_guard_legal is None else nearest_guard_legal[0]
                ),
                "nearest_guard_legal_frontier_id": (
                    None if nearest_guard_legal is None else nearest_guard_legal[1]
                ),
                "nearest_guard_legal_path_length_m": (
                    None if nearest_guard_legal is None else nearest_guard_legal[0]
                ),
                "nearest_meaningful_guard_legal_frontier_id": (
                    None
                    if nearest_meaningful_guard_legal is None
                    else nearest_meaningful_guard_legal[1]
                ),
                "nearest_meaningful_guard_legal_path_length_m": (
                    None
                    if nearest_meaningful_guard_legal is None
                    else nearest_meaningful_guard_legal[0]
                ),
                "outcome_backtrack_available": bool(backtrack_records),
                "outcome_backtrack_guard_legal": any(row[2] for row in backtrack_records),
                "outcome_backtrack_frontier_ids": [row[1] for row in backtrack_records],
                "outcome_backtrack_path_lengths_m": [row[0] for row in backtrack_records],
            }
        )
    return rows


def _catalog_initial_heading(
    path_m: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Return the first meaningful direction in one already guarded path."""

    for left, right in zip(path_m, path_m[1:], strict=False):
        length_m = math.dist(left, right)
        if length_m > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M:
            return tuple(
                (right[axis] - left[axis]) / length_m for axis in range(3)
            )  # type: ignore[return-value]
    return None


def _candidate_route_opportunity_catalog(
    state: PublicSearchState,
    route_guard_records: Sequence[dict[str, object]],
    pool: Sequence[CandidateFragmentManifest],
    *,
    selected: CandidateFragmentManifest | None = None,
) -> dict[str, object]:
    """Audit the current public route graph without changing any selection.

    The old route-cap record cannot answer whether the current physical-time
    horizon exposes a longer legal direction.  This catalog keeps that
    question falsifiable for each decision: it separates public-frontier
    availability, individual static-guard admission, and admission into a
    jointly safe team manifest.  It deliberately defines no minimum or target
    route length; ``longest`` is a descriptive maximum over the state that was
    actually offered to the selector.
    """

    frontier_by_id = {frontier.frontier_id: frontier for frontier in state.frontiers}
    frontier_by_endpoint = {
        tuple(frontier.position_m): frontier for frontier in state.frontiers
    }
    records_by_agent: dict[str, dict[str, dict[str, object]]] = {
        agent.agent_id: {} for agent in state.agents
    }
    for record in route_guard_records:
        agent_id = record.get("agent_id")
        if not isinstance(agent_id, str) or agent_id not in records_by_agent:
            continue
        frontier_id = record.get("public_access_frontier_id")
        frontier = frontier_by_id.get(frontier_id) if isinstance(frontier_id, str) else None
        if frontier is None:
            requested_path = record.get("requested_path_m")
            if isinstance(requested_path, (list, tuple)) and len(requested_path) >= 2:
                endpoint = tuple(requested_path[-1])
                frontier = frontier_by_endpoint.get(endpoint)
        if frontier is not None:
            prior = records_by_agent[agent_id].get(frontier.frontier_id)
            if prior is None or (
                bool(prior.get("cache_hit")) and not bool(record.get("cache_hit"))
            ):
                records_by_agent[agent_id][frontier.frontier_id] = record

    team_feasible_candidate_ids: dict[tuple[str, str], set[str]] = {}
    team_feasible_manifest_hashes: dict[tuple[str, str], set[str]] = {}
    for manifest in pool:
        if not manifest.feasible:
            continue
        for fragment in manifest.fragments:
            if fragment.type_signature.fragment_type != "transit":
                continue
            features = dict(fragment.type_signature.public_features)
            if features.get("assignment_role") != "explore" or len(fragment.path) < 2:
                continue
            frontier_id = features.get("frontier_id")
            if not isinstance(frontier_id, str) or frontier_id not in frontier_by_id:
                raise RuntimeError(
                    "feasible exploration transit omits a current public frontier ID"
                )
            key = (fragment.agent_id, frontier_id)
            team_feasible_candidate_ids.setdefault(key, set()).add(manifest.candidate_id)
            team_feasible_manifest_hashes.setdefault(key, set()).add(manifest.manifest_hash)

    selected_frontier_ids: dict[str, str] = {}
    if selected is not None:
        for fragment in selected.fragments:
            if fragment.type_signature.fragment_type != "transit":
                continue
            features = dict(fragment.type_signature.public_features)
            if features.get("assignment_role") != "explore":
                continue
            frontier_id = features.get("frontier_id")
            if not isinstance(frontier_id, str) or frontier_id not in frontier_by_id:
                raise RuntimeError(
                    "selected exploration transit omits a current public frontier ID"
                )
            selected_frontier_ids[fragment.agent_id] = frontier_id
    deadline_s = state.decision_start_s + state.decision_duration_s
    agent_rows: list[dict[str, object]] = []

    def route_summary(row: dict[str, object] | None) -> dict[str, object] | None:
        if row is None:
            return None
        return {
            "frontier_id": row["frontier_id"],
            "frontier_cluster_id": row["frontier_cluster_id"],
            "viewpoint_kind": row["viewpoint_kind"],
            "guarded_path_length_m": row["guarded_path_length_m"],
            "task_reservation_matched": row["task_reservation_matched"],
            "task_reservation_direction": row["task_reservation_direction"],
            "task_reservation_heading_alignment": row[
                "task_reservation_heading_alignment"
            ],
            "appears_in_feasible_team_candidate": row[
                "appears_in_feasible_team_candidate"
            ],
            "route_path_sha256": row["route_path_sha256"],
        }

    def longest(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
        if not rows:
            return None
        return max(
            rows,
            key=lambda row: (
                float(row["guarded_path_length_m"]),
                float(row["information_gain"]),
                -float(row["traversal_risk"]),
                str(row["frontier_id"]),
            ),
        )

    for agent in state.agents:
        reservation = next(
            (
                row
                for row in state.task_reservations
                if row.agent_id == agent.agent_id
            ),
            None,
        )
        rows: list[dict[str, object]] = []
        for frontier in state.frontiers:
            direct_path = (agent.position_m, frontier.position_m)
            direct_lower_bound_s = (
                state.decision_start_s
                + state.transit_timing_model.estimate_seconds(direct_path)
                + state.observe_dwell_s
            )
            record = records_by_agent[agent.agent_id].get(frontier.frontier_id)
            guarded_path: tuple[tuple[float, float, float], ...] | None = None
            if record is not None:
                raw_path = record.get("guarded_path_m")
                if isinstance(raw_path, (list, tuple)) and len(raw_path) >= 2:
                    candidate_path = tuple(tuple(point) for point in raw_path)
                    if all(len(point) == 3 for point in candidate_path):
                        guarded_path = candidate_path
            guarded_length_m = (
                None if guarded_path is None else _path_length_m(guarded_path)
            )
            within_window = (
                guarded_path is not None
                and state.decision_start_s
                + state.transit_timing_model.estimate_seconds(guarded_path)
                + state.observe_dwell_s
                <= deadline_s + 1.0e-9
            )
            non_alias = (
                guarded_path is not None and is_non_alias_exploration_path(guarded_path)
            )
            heading = (
                None if guarded_path is None else _catalog_initial_heading(guarded_path)
            )
            (
                task_reservation_matched,
                task_reservation_anchor_distance_m,
                task_reservation_normal_alignment,
            ) = task_reservation_matches_frontier(reservation, frontier)
            task_reservation_heading_alignment = (
                None
                if reservation is None or heading is None
                else min(
                    1.0,
                    max(
                        -1.0,
                        sum(
                            reservation.terminal_heading_unit[axis] * heading[axis]
                            for axis in range(3)
                        ),
                    ),
                )
            )
            task_reservation_direction = (
                "no_active_task_reservation"
                if reservation is None
                else (
                    "different_public_task"
                    if not task_reservation_matched
                    else (
                        "path_heading_unavailable"
                        if task_reservation_heading_alignment is None
                        else (
                            "task_forward"
                            if task_reservation_heading_alignment >= 0.0
                            else "task_reverse"
                        )
                    )
                )
            )
            static_guard_legal = record is not None and record.get("legal") is True
            raw_rejection_stages = (
                {} if record is None else record.get("clearance_rejection_stage_counts", {})
            )
            if not isinstance(raw_rejection_stages, dict):
                raw_rejection_stages = {}
            static_guard_rejection_stage_counts = {
                str(stage): int(count)
                for stage, count in raw_rejection_stages.items()
                if isinstance(stage, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
            }
            exploration_edge_admitted = (
                frontier.task_kind == "explore"
                and static_guard_legal
                and bool(within_window)
                and bool(non_alias)
            )
            selected_here = selected_frontier_ids.get(agent.agent_id) == frontier.frontier_id
            candidate_key = (agent.agent_id, frontier.frontier_id)
            feasible_candidate_ids = tuple(
                sorted(team_feasible_candidate_ids.get(candidate_key, ()))
            )
            feasible_manifest_hashes = tuple(
                sorted(team_feasible_manifest_hashes.get(candidate_key, ()))
            )
            selected_candidate_contains_edge = (
                selected is not None and selected.candidate_id in feasible_candidate_ids
            )
            if selected_here != selected_candidate_contains_edge:
                raise RuntimeError(
                    "selected route catalog membership disagrees with public frontier provenance"
                )
            rows.append(
                {
                    "frontier_id": frontier.frontier_id,
                    "frontier_cluster_id": frontier.frontier_cluster_id,
                    "viewpoint_kind": frontier.viewpoint_kind,
                    "task_kind": frontier.task_kind,
                    "information_gain": frontier.information_gain,
                    "traversal_risk": frontier.traversal_risk,
                    "endpoint_m": list(frontier.position_m),
                    "task_anchor_m": list(frontier.task_anchor_m),
                    "direct_distance_m": math.dist(*direct_path),
                    "direct_time_lower_bound_s": direct_lower_bound_s,
                    "direct_time_lower_bound_admitted": direct_lower_bound_s
                    <= deadline_s + 1.0e-9,
                    "route_guard_queried": record is not None,
                    "route_guard_cache_hit": (
                        None if record is None else bool(record.get("cache_hit"))
                    ),
                    "static_guard_legal": static_guard_legal,
                    "static_guard_reason": None if record is None else record.get("reason"),
                    "static_guard_rejection_stage_counts": static_guard_rejection_stage_counts,
                    "public_route_status": (
                        None if record is None else record.get("public_route_status")
                    ),
                    "guarded_path_length_m": guarded_length_m,
                    "guarded_path_waypoint_count": (
                        None if guarded_path is None else len(guarded_path)
                    ),
                    "route_path_sha256": (
                        None if guarded_path is None else canonical_sha256(guarded_path)
                    ),
                    "within_decision_window": bool(within_window),
                    "non_alias_exploration_route": bool(non_alias),
                    "individual_exploration_edge_admitted": exploration_edge_admitted,
                    "task_reservation_matched": task_reservation_matched,
                    "task_reservation_anchor_distance_m": task_reservation_anchor_distance_m,
                    "task_reservation_normal_alignment": task_reservation_normal_alignment,
                    "task_reservation_heading_alignment": task_reservation_heading_alignment,
                    "task_reservation_direction": task_reservation_direction,
                    "appears_in_feasible_team_candidate": bool(feasible_candidate_ids),
                    "feasible_team_candidate_ids": list(feasible_candidate_ids),
                    "feasible_team_manifest_hashes": list(feasible_manifest_hashes),
                    "selected_candidate_contains_edge": selected_candidate_contains_edge,
                    "selected": selected_here,
                }
            )
        eligible = [
            row
            for row in rows
            if row["individual_exploration_edge_admitted"] is True
        ]
        team_feasible = [
            row for row in eligible if row["appears_in_feasible_team_candidate"] is True
        ]
        reserved_task_edges = [
            row for row in eligible if row["task_reservation_matched"] is True
        ]
        team_feasible_reserved_task_edges = [
            row
            for row in team_feasible
            if row["task_reservation_matched"] is True
        ]
        selected_row = next((row for row in rows if row["selected"] is True), None)
        static_guard_rejected = [
            row
            for row in rows
            if row["route_guard_queried"] is True and row["static_guard_legal"] is False
        ]
        static_guard_reason_counts = Counter(
            str(row["static_guard_reason"] or "unspecified_static_guard_rejection")
            for row in static_guard_rejected
        )
        static_guard_rejection_stage_counts = Counter(
            stage
            for row in static_guard_rejected
            for stage, count in dict(row["static_guard_rejection_stage_counts"]).items()
            for _ in range(int(count))
        )
        agent_rows.append(
            {
                "agent_id": agent.agent_id,
                "task_reservation": (
                    None if reservation is None else reservation.to_dict()
                ),
                "frontier_edges": rows,
                "summary": {
                    "ordinary_explore_frontier_count": sum(
                        frontier.task_kind == "explore" for frontier in state.frontiers
                    ),
                    "route_guard_queried_count": sum(
                        row["route_guard_queried"] is True for row in rows
                    ),
                    "individual_exploration_edge_count": len(eligible),
                    "individual_reserved_task_edge_count": len(reserved_task_edges),
                    "team_feasible_exploration_edge_count": len(team_feasible),
                    "team_matching_lost_exploration_edge_count": len(eligible) - len(team_feasible),
                    "team_feasible_reserved_task_edge_count": len(
                        team_feasible_reserved_task_edges
                    ),
                    "static_guard_rejected_frontier_edge_count": len(static_guard_rejected),
                    "static_guard_rejection_reason_counts": dict(
                        sorted(static_guard_reason_counts.items())
                    ),
                    "static_guard_rejection_stage_counts": dict(
                        sorted(static_guard_rejection_stage_counts.items())
                    ),
                    "unqueried_frontier_edge_count": sum(
                        row["route_guard_queried"] is False for row in rows
                    ),
                    "longest_individual_exploration_edge": route_summary(longest(eligible)),
                    "longest_individual_reserved_task_edge": route_summary(
                        longest(reserved_task_edges)
                    ),
                    "longest_team_feasible_edge": route_summary(longest(team_feasible)),
                    "longest_team_feasible_reserved_task_edge": route_summary(
                        longest(team_feasible_reserved_task_edges)
                    ),
                    "selected_edge": route_summary(selected_row),
                },
            }
        )
    payload = {
        "schema_version": "hm3d-candidate-route-opportunity-catalog-v3",
        "claim_limit": (
            "Engineering diagnosis only. This catalog reads the current public frontier "
            "graph, individual static-guard outcomes, and already admitted team manifests; "
            "it does not alter candidate construction, ranking, safety, execution, rewards, "
            "training, QD, or OGFR. Each team-feasible edge records the admitted public "
            "candidate IDs and manifest hashes that contain it, so membership is distinct "
            "from selection."
        ),
        "decision_id": state.context.decision_id,
        "decision_duration_s": state.decision_duration_s,
        "public_frontier_count": len(state.frontiers),
        "candidate_pool_hash": public_candidate_pool_hash(pool),
        "selected_manifest_hash": None if selected is None else selected.manifest_hash,
        "agents": agent_rows,
    }
    payload["catalog_sha256"] = canonical_sha256(payload)
    return payload


def _relative_height_band(delta_m: float, band_m: float) -> int:
    """Quantize relative height without turning millimetre drift into a new band."""

    scaled = delta_m / band_m
    if scaled >= 0.0:
        return math.floor(scaled + 0.5)
    return math.ceil(scaled - 0.5)


def _episode_mobility_summary(
    decisions: Sequence[dict[str, object]],
    initial_positions: Sequence[tuple[float, float, float]],
    *,
    height_band_m: float = MOBILITY_HEIGHT_BAND_M,
) -> dict[str, object]:
    """Aggregate realised speed, distance and inter-height activity."""

    if height_band_m <= 0.0:
        raise ValueError("mobility height band must be positive")
    initial_by_agent = {
        f"uav{index}": tuple(position) for index, position in enumerate(initial_positions)
    }
    per_agent = {
        agent_id: {
            "realised_path_length_m": 0.0,
            "maximum_linear_speed_mps": 0.0,
            "height_samples_m": [position[2]],
        }
        for agent_id, position in initial_by_agent.items()
    }
    planned_path_length_m = 0.0
    attempted_count = 0
    completed_count = 0
    raw_exposed_vertical_count = 0
    public_free_path_reachable_vertical_count = 0
    static_guard_admitted_vertical_count = 0
    team_feasible_vertical_count = 0
    selected_vertical_count = 0
    completed_vertical_count = 0
    for decision in decisions:
        reachability = decision.get("candidate_reachability")
        if not isinstance(reachability, dict):
            raise RuntimeError("decision omits candidate reachability diagnostics")
        opportunity = reachability.get("vertical_opportunity")
        if not isinstance(opportunity, dict):
            raise RuntimeError("decision omits vertical opportunity diagnostics")
        if opportunity.get("schema_version") != "hm3d-p07-vertical-opportunity-v3":
            raise RuntimeError("decision uses a stale vertical opportunity schema")
        raw_exposed_vertical_count += int(opportunity["raw_exposed_upward_frontier_count"])
        raw_exposed_vertical_count += int(opportunity["raw_exposed_downward_frontier_count"])
        public_free_path_reachable_vertical_count += int(
            opportunity["public_free_path_reachable_upward_edge_count"]
        ) + int(opportunity["public_free_path_reachable_downward_edge_count"])
        static_guard_admitted_vertical_count += int(
            opportunity["static_guard_admitted_upward_edge_count"]
        ) + int(opportunity["static_guard_admitted_downward_edge_count"])
        team_feasible_vertical_count += int(
            opportunity["team_feasible_upward_edge_count"]
        ) + int(opportunity["team_feasible_downward_edge_count"])
        selected_vertical_count += int(opportunity["selected_upward_edge_count"])
        selected_vertical_count += int(opportunity["selected_downward_edge_count"])
        completed_vertical_count += int(opportunity["completed_vertical_agent_count"])

        calibration = decision.get("execution_calibration")
        if not isinstance(calibration, dict) or not isinstance(calibration.get("agents"), list):
            raise RuntimeError("decision omits execution calibration for mobility summary")
        for agent in calibration["agents"]:
            if not isinstance(agent, dict):
                raise RuntimeError("mobility summary received a malformed agent row")
            agent_id = str(agent["agent_id"])
            if agent_id not in per_agent:
                raise RuntimeError(f"unexpected mobility agent {agent_id!r}")
            geometry = agent.get("route_geometry")
            if not isinstance(geometry, dict):
                raise RuntimeError("mobility agent omits route geometry")
            planned_path_length_m += float(geometry["command_path_length_m"])
            per_agent[agent_id]["realised_path_length_m"] += float(
                agent["realized_transit_path_length_m"]
            )
            per_agent[agent_id]["maximum_linear_speed_mps"] = max(
                float(per_agent[agent_id]["maximum_linear_speed_mps"]),
                float(agent["maximum_linear_speed_mps"]),
            )
            attempted_count += int(agent["transit_attempted"] is True)
            completed_count += int(agent["transit_completed"] is True)
            samples = agent.get("controller_tracking_samples")
            if not isinstance(samples, list):
                raise RuntimeError("mobility agent omits controller tracking samples")
            for sample in samples:
                if not isinstance(sample, dict):
                    raise RuntimeError("controller tracking sample is malformed")
                position = sample.get("post_step_position_m")
                if not isinstance(position, (list, tuple)) or len(position) != 3:
                    raise RuntimeError("controller tracking sample omits realised position")
                per_agent[agent_id]["height_samples_m"].append(float(position[2]))

    agent_rows: list[dict[str, object]] = []
    for agent_id, values in sorted(per_agent.items()):
        heights = values.pop("height_samples_m")
        assert isinstance(heights, list)
        initial_z = initial_by_agent[agent_id][2]
        bands = sorted(
            {_relative_height_band(height - initial_z, height_band_m) for height in heights}
        )
        agent_rows.append(
            {
                "agent_id": agent_id,
                **values,
                "minimum_realised_height_m": min(heights),
                "maximum_realised_height_m": max(heights),
                "maximum_absolute_height_displacement_m": max(
                    abs(height - initial_z) for height in heights
                ),
                "visited_relative_height_bands": bands,
                "visited_relative_height_band_count": len(bands),
            }
        )
    realised_path_length_m = sum(float(row["realised_path_length_m"]) for row in agent_rows)
    return {
        "schema_version": "hm3d-p07-mobility-summary-v2",
        "height_band_m": height_band_m,
        "planned_fleet_path_length_m": planned_path_length_m,
        "realised_fleet_path_length_m": realised_path_length_m,
        "mean_realised_path_length_per_agent_m": realised_path_length_m / max(len(agent_rows), 1),
        "maximum_realised_speed_mps": max(
            (float(row["maximum_linear_speed_mps"]) for row in agent_rows),
            default=0.0,
        ),
        "transit_attempt_count": attempted_count,
        "transit_completed_count": completed_count,
        "transit_completion_fraction": completed_count / max(attempted_count, 1),
        "raw_exposed_vertical_frontier_count": raw_exposed_vertical_count,
        "public_free_path_reachable_vertical_edge_count": public_free_path_reachable_vertical_count,
        "static_guard_admitted_vertical_edge_count": static_guard_admitted_vertical_count,
        "team_feasible_vertical_edge_count": team_feasible_vertical_count,
        "selected_vertical_edge_count": selected_vertical_count,
        "completed_vertical_agent_count": completed_vertical_count,
        "cross_height_band_agent_count": sum(
            int(row["visited_relative_height_band_count"]) > 1 for row in agent_rows
        ),
        "agents": agent_rows,
    }


def _episode_stationarity_summary(decisions: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate the decision-level no-idle supervision without hiding waits."""

    audits: list[dict[str, object]] = []
    for decision in decisions:
        audit = decision.get("stationarity_supervision")
        if not isinstance(audit, dict):
            raise RuntimeError("decision omits stationarity supervision")
        audits.append(audit)
    aggregate_fields = (
        "total_agent_seconds",
        "exploration_transit_phase_agent_seconds",
        "observation_dwell_agent_seconds",
        "synchronization_wait_agent_seconds",
        "intentional_hold_agent_seconds",
    )
    totals = {field: sum(float(audit[field]) for audit in audits) for field in aggregate_fields}
    violations = [
        violation
        for audit in audits
        for violation in audit.get("violations", [])
        if isinstance(violation, str)
    ]
    return {
        "schema_version": "hm3d-episode-stationarity-supervision-v1",
        **totals,
        "exploration_transit_phase_fraction": (
            totals["exploration_transit_phase_agent_seconds"] / totals["total_agent_seconds"]
            if totals["total_agent_seconds"] > 0.0
            else 0.0
        ),
        "synchronization_wait_fraction": (
            totals["synchronization_wait_agent_seconds"] / totals["total_agent_seconds"]
            if totals["total_agent_seconds"] > 0.0
            else 0.0
        ),
        "intentional_hold_fraction": (
            totals["intentional_hold_agent_seconds"] / totals["total_agent_seconds"]
            if totals["total_agent_seconds"] > 0.0
            else 0.0
        ),
        "decision_count": len(audits),
        "not_admitted_decision_count": sum(
            audit.get("status") != "STATIONARITY_SUPERVISION_ADMITTED" for audit in audits
        ),
        "violations": violations,
        "status": (
            "EPISODE_STATIONARITY_SUPERVISION_ADMITTED"
            if not violations
            else "EPISODE_STATIONARITY_SUPERVISION_NOT_ADMITTED"
        ),
        "claim_limit": (
            "A high synchronization-wait fraction is an actionable route-assignment result, "
            "not evidence that the time was spent exploring."
        ),
    }


RUNNER_VERSION = "hm3d-p07-target-free-online-exploration-v4"
P07_EXECUTION_SMOKE_COMPLETE_STATUS = "P07_EXECUTION_SMOKE_COMPLETE"
P07_EXECUTION_SMOKE_FAILED_STATUS = "P07_EXECUTION_SMOKE_FAILED"


def _classify_execution_status(
    *, terminal_outcome: str, failed_fragment_count: int
) -> tuple[str, str | None]:
    """Classify a physically executed episode without hiding outcome failures.

    A budget-exhausted episode is scoreable only when every executed fragment
    completed under the safety/timing contract.  A outcome-backed terminal
    failure remains a useful engineering artifact, but it must never look like
    a completed smoke to collectors or downstream evidence readers.
    """

    if terminal_outcome == "budget_exhausted" and failed_fragment_count == 0:
        return P07_EXECUTION_SMOKE_COMPLETE_STATUS, None
    reasons = [f"terminal_outcome={terminal_outcome}"]
    if failed_fragment_count:
        reasons.append(f"failed_fragment_count={failed_fragment_count}")
    return P07_EXECUTION_SMOKE_FAILED_STATUS, ";".join(reasons)
MINIMUM_ACTION_BUDGET_UTILIZATION = 0.95
QD_UTILITY_SLACK = 0.10
_LATTICE_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
QDHistoryRow = tuple[
    tuple[float, float, float],
    RealisedQDDescriptor,
    float,
    float,
    str,
    str,
    str,
    tuple[tuple[int, int, int], ...],
    str,
]


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _artifact_payload(artifact: dict[str, Any], label: str) -> dict[str, Any]:
    """Accept a phase envelope or its signed payload without weakening checks."""

    payload = artifact.get("payload")
    if payload is None:
        return artifact
    if not isinstance(payload, dict):
        raise ValueError(f"{label} artifact payload must be an object")
    return payload


def _load_train_qd_history(
    paths: tuple[Path, ...],
    *,
    split_manifest_sha256: str,
) -> tuple[tuple[QDHistoryRow, ...], dict[str, Any]]:
    """Load only prior *train* execution outcomes for a QD mechanism run.

    The history contains descriptors and public scalar outcomes, never a prior
    trajectory.  Validation and test artifacts are rejected so a selector
    cannot learn from the evaluation partition before its own episode starts.
    """

    if not paths:
        raise ValueError("QD history requires at least one train execution record")
    if not isinstance(split_manifest_sha256, str) or len(split_manifest_sha256) != 64:
        raise ValueError("QD history requires a valid frozen split manifest hash")
    int(split_manifest_sha256, 16)
    rows: list[QDHistoryRow] = []
    candidate_descriptor_features: list[OutcomeQDFeatureVector] = []
    candidate_descriptor_feature_scenes: list[str] = []
    calibration_samples: list[tuple[str, RealisedQDDescriptor, str]] = []
    source_runtime_record_hashes: list[str] = []
    for path in paths:
        payload = _read_object(path)
        require_p07_evidence_field(payload, "realised_qd_descriptor")
        if payload.get("schema_version") != "hm3d-p07-exploration-execution-v1":
            raise ValueError("QD history must be a P07 real-execution record")
        if (
            payload.get("status") != "P07_EXECUTION_SMOKE_COMPLETE"
            or payload.get("synthetic") is not False
        ):
            raise ValueError("QD history must be a completed non-synthetic execution")
        if payload.get("selection_partition") != "train":
            raise ValueError("QD history may only use train-scene execution outcomes")
        if payload.get("calibration_only_timeout_probe") is True:
            raise ValueError("calibration-only timeout probes cannot enter QD history")
        if payload.get("split_manifest_sha256") != split_manifest_sha256:
            raise ValueError("QD history record does not belong to the frozen train split")
        if payload.get("fleet_size") != FORMAL_FLEET_SIZE:
            raise ValueError("QD history does not match the formal four-CF2X contract")
        recorded_hash = payload.get("runtime_record_sha256")
        unsigned = dict(payload)
        unsigned.pop("runtime_record_sha256", None)
        if not isinstance(recorded_hash, str) or canonical_sha256(unsigned) != recorded_hash:
            raise ValueError("QD history runtime record hash is invalid")
        source_runtime_record_hashes.append(recorded_hash)
        scene_id = payload.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("QD history scene ID is invalid")
        calibration_mode = payload.get("qd_calibration_mode")
        if (
            calibration_mode is not None
            and calibration_mode not in HM3D_QD_CALIBRATION_INTENT_MODES
        ):
            raise ValueError("QD history calibration mode is invalid")
        if payload.get("strategy") == "qd_calibration" and calibration_mode is None:
            raise ValueError("QD calibration history record lacks its public intent mode")
        qd = payload.get("realised_qd")
        if not isinstance(qd, dict) or not isinstance(qd.get("admissions"), list):
            raise ValueError("QD history lacks realised outcome admissions")
        for index, row in enumerate(qd["admissions"]):
            if not isinstance(row, dict) or row.get("feasible") is not True:
                continue
            if row.get("executed") is not True:
                raise ValueError("QD history admission must be an executed candidate")
            intent = _point(row.get("public_candidate_intent"), f"QD history intent {index}")
            raw_descriptor = row.get("descriptor")
            if not isinstance(raw_descriptor, dict):
                raise ValueError("QD history descriptor is missing")
            raw_candidate_features = row.get("candidate_descriptor_features")
            if not isinstance(raw_candidate_features, dict):
                raise ValueError("QD history lacks pre-registered descriptor-family features")
            descriptor = RealisedQDDescriptor(
                vertical_motion_ratio=float(raw_descriptor.get("vertical_motion_ratio")),
                team_spatial_dispersion=float(raw_descriptor.get("team_spatial_dispersion")),
                public_observation_complementarity=float(
                    raw_descriptor.get("public_observation_complementarity")
                ),
                schema_version=str(raw_descriptor.get("schema_version")),
            )
            public_quality = float(row.get("public_new_free_volume_m3"))
            public_cost = float(row.get("cost_energy_j"))
            candidate_id = row.get("candidate_id")
            manifest_hash = row.get("candidate_manifest_sha256")
            outcome_hash = row.get("execution_outcome_sha256")
            raw_footprint = row.get("public_new_free_voxel_keys")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("QD history candidate ID is invalid")
            for value, label in (
                (manifest_hash, "QD history candidate manifest hash"),
                (outcome_hash, "QD history execution outcome hash"),
            ):
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"{label} is invalid")
                int(value, 16)
            if not math.isfinite(public_quality) or public_quality < 0.0:
                raise ValueError("QD history public quality is invalid")
            if not math.isfinite(public_cost) or public_cost < 0.0:
                raise ValueError("QD history public cost is invalid")
            if not isinstance(raw_footprint, list) or not raw_footprint:
                raise ValueError("QD history lacks a non-empty public execution footprint")
            footprint: list[tuple[int, int, int]] = []
            for key_index, raw_key in enumerate(raw_footprint):
                if (
                    not isinstance(raw_key, list)
                    or len(raw_key) != 3
                    or any(
                        not isinstance(value, int) or isinstance(value, bool) for value in raw_key
                    )
                ):
                    raise ValueError(
                        f"QD history public execution footprint key {key_index} is invalid"
                    )
                footprint.append((raw_key[0], raw_key[1], raw_key[2]))
            rows.append(
                (
                    intent,
                    descriptor,
                    public_quality,
                    public_cost,
                    candidate_id,
                    manifest_hash,
                    outcome_hash,
                    tuple(footprint),
                    scene_id,
                )
            )
            candidate_descriptor_features.append(
                OutcomeQDFeatureVector.from_dict(raw_candidate_features)
            )
            candidate_descriptor_feature_scenes.append(scene_id)
            if calibration_mode is not None:
                calibration_samples.append((calibration_mode, descriptor, scene_id))
    if len(rows) < MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION:
        raise ValueError(
            "QD history needs at least "
            f"{MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION} feasible train execution outcomes"
        )
    if len({row[8] for row in rows}) < 2:
        raise ValueError("QD history needs outcomes from at least two train scenes")
    descriptors = tuple(row[1] for row in rows)
    intents = tuple(row[0] for row in rows)
    footprints = tuple(row[7] for row in rows)
    richness = audit_realised_qd_richness(descriptors)
    alignment = audit_intent_realised_alignment(
        intents,
        descriptors,
        scene_ids=tuple(row[8] for row in rows),
    )
    footprint_separation = audit_realised_qd_footprint_separation(descriptors, footprints)
    descriptor_family_screen = audit_pre_registered_qd_descriptor_families(
        candidate_descriptor_features,
        footprints,
        candidate_descriptor_feature_scenes,
    )
    descriptors_by_manifest: dict[str, list[RealisedQDDescriptor]] = {}
    for row in rows:
        descriptors_by_manifest.setdefault(row[5], []).append(row[1])
    reproducibility = audit_realised_qd_reproducibility(descriptors_by_manifest)
    mode_contrast = audit_realised_qd_calibration_mode_contrasts(
        tuple(row[0] for row in calibration_samples),
        tuple(row[1] for row in calibration_samples),
        tuple(row[2] for row in calibration_samples),
    )
    if richness.status != "QD_DESCRIPTOR_ADMITTED":
        raise ValueError(f"QD history descriptor richness is not admitted: {richness.reasons}")
    if footprint_separation.status != "QD_FOOTPRINT_SEPARATION_ADMITTED":
        raise ValueError(
            "QD history footprints do not validate the descriptor semantics: "
            f"{footprint_separation.reasons}"
        )
    admission: dict[str, Any] = {
        "status": "QD_TRAIN_DESCRIPTOR_ADMITTED",
        "descriptor_schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
        "archive_spec_sha256": HM3D_REALISED_QD_ARCHIVE_SPEC.digest,
        "outcome_count": len(rows),
        "scene_ids": sorted({row[8] for row in rows}),
        "split_manifest_sha256": split_manifest_sha256,
        "source_runtime_record_sha256s": sorted(source_runtime_record_hashes),
        "richness_audit": richness.to_dict(),
        "intent_outcome_alignment": alignment.to_dict(),
        "footprint_separation_audit": footprint_separation.to_dict(),
        "descriptor_family_screen": descriptor_family_screen.to_dict(),
        "reproducibility_audit": reproducibility.to_dict(),
        "calibration_mode_contrast_audit": mode_contrast.to_dict(),
    }
    admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
    return tuple(rows), admission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_transit_timing_contract(
    path: Path,
    *,
    expected_execution_profile: dict[str, object] | None = None,
) -> tuple[ConservativeTransitTimingModel, float]:
    payload = _read_object(path)
    if (
        payload.get("schema_version") != "hm3d-cf2x-transit-timing-calibration-v4"
        or payload.get("status")
        not in {"CALIBRATION_PASS", "PASS_THROUGH_ANALYTICAL"}
        or not isinstance(payload.get("time_model"), dict)
    ):
        raise ValueError(
            "transit time-model evidence is not a passed calibration or "
            "analytical pass-through artifact"
        )
    observation_dwell_s = payload.get("observation_dwell_s")
    if (
        not isinstance(observation_dwell_s, (int, float))
        or isinstance(observation_dwell_s, bool)
        or not math.isfinite(float(observation_dwell_s))
        or float(observation_dwell_s) <= 0.0
    ):
        raise ValueError("transit calibration omits a positive observation dwell")
    execution_profile = payload.get("execution_profile")
    execution_profile_sha256 = payload.get("execution_profile_sha256")
    timing_model_payload = payload["time_model"]
    # The active v4 contract must carry the long-route extrapolation fields in
    # both the artifact summary and the serialized model.  This prevents a
    # stale pre-reserve artifact from silently re-entering candidate admission.
    required_timing_fields = (
        "calibrated_max_segment_count",
        "uncovered_segment_reserve_s",
        "intermediate_waypoint_requires_settle",
        "continuous_waypoint_speed_mps",
    )
    missing_model_fields = [
        field for field in required_timing_fields if field not in timing_model_payload
    ]
    if missing_model_fields:
        raise ValueError(
            "transit calibration time_model omits required v4 fields: "
            + ", ".join(missing_model_fields)
        )
    mismatched_summary_fields = [
        field
        for field in required_timing_fields
        if payload.get(field) != timing_model_payload.get(field)
    ]
    if mismatched_summary_fields:
        raise ValueError(
            "transit calibration summary does not match time_model: "
            + ", ".join(mismatched_summary_fields)
        )
    if expected_execution_profile is not None:
        if not isinstance(execution_profile, dict):
            raise ValueError("transit calibration omits its execution profile")
        if execution_profile_sha256 != canonical_sha256(execution_profile):
            raise ValueError("transit calibration execution-profile hash is invalid")
        normalized_execution_profile = _normalized_transit_execution_profile(
            execution_profile
        )
        if normalized_execution_profile != expected_execution_profile:
            mismatches = sorted(
                key
                for key in set(normalized_execution_profile)
                | set(expected_execution_profile)
                if normalized_execution_profile.get(key)
                != expected_execution_profile.get(key)
            )
            raise ValueError(
                "transit calibration execution profile does not match the current P07 "
                "runtime: " + ", ".join(mismatches)
            )
    return (
        ConservativeTransitTimingModel.from_dict(timing_model_payload),
        float(observation_dwell_s),
    )


def _load_transit_timing_model(path: Path) -> ConservativeTransitTimingModel:
    # Compatibility helper for callers that only inspect the fitted model.
    return _load_transit_timing_contract(path)[0]


def _normalized_transit_execution_profile(
    execution_profile: dict[str, object],
) -> dict[str, object]:
    # Normalize the frozen receipt-era timing ABI to the active outcome ABI.
    normalized = dict(execution_profile)
    if "receipt_time_tolerance_s" in normalized:
        normalized.setdefault(
            "outcome_time_tolerance_s", normalized["receipt_time_tolerance_s"]
        )
        normalized.pop("receipt_time_tolerance_s", None)
    return normalized


def _normalized_p0_eligibility_contract(
    contract: dict[str, object],
) -> dict[str, object]:
    # Normalize the receipt-era eligibility ABI to the active outcome ABI.
    normalized = dict(contract)
    if "receipt_time_tolerance_s" in normalized:
        normalized.setdefault(
            "outcome_time_tolerance_s", normalized["receipt_time_tolerance_s"]
        )
        normalized.pop("receipt_time_tolerance_s", None)
    # The eligibility certificate authorizes the frozen reset poses and the
    # shared candidate pool.  The selector that will rank that pool is not
    # part of the execution contract, so the evidence recorded under the
    # transparent frontier_3d qualification may be reused by QD strategies.
    normalized.pop("strategy", None)
    # ``random_key`` perturbs observation sampling and selector randomness,
    # not the certified reset poses, collision clearance or relay graph.
    # Multi-seed repetitions of the same frozen start therefore reuse one
    # eligibility certificate instead of re-auditing every seed.
    normalized.pop("random_key", None)
    return normalized


def _current_transit_execution_profile(
    *,
    cf2x_usd_path: Path,
    fleet_size: int,
    physics_dt_s: float,
    arrival_tolerance_m: float,
    outcome_time_tolerance_s: float,
    controller_id: str = cf2x.CF2X_DEFAULT_CONTROLLER_ID,
) -> dict[str, object]:
    return {
        "cf2x_usd_sha256": _sha256(cf2x_usd_path),
        "fleet_size": fleet_size,
        "physics_dt_s": physics_dt_s,
        "arrival_tolerance_m": arrival_tolerance_m,
        "outcome_time_tolerance_s": outcome_time_tolerance_s,
        "backend_id": cf2x.CF2X_EXECUTION_BACKEND_ID,
        "evidence_class": cf2x.CF2X_EXECUTION_EVIDENCE_CLASS,
        "controller_tracking": cf2x._transit_timing_controller_tracking_profile(
            controller_id,
            physics_dt_s=physics_dt_s,
        ),
    }


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime evidence: {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _progress_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.progress.json")


def _write_decision_progress(
    output: Path,
    *,
    scene_id: str,
    strategy: str,
    action_budget_s: float,
    elapsed_physics_s: float,
    decisions: list[dict[str, Any]],
    samples: list[ExplorationMetricSample],
) -> None:
    """Preserve completed PhysX decisions before final episode assembly."""

    payload = {
        "schema_version": "hm3d-p07-decision-progress-v1",
        "status": "P07_DECISION_PROGRESS",
        "formal_result": False,
        "trainable": False,
        "claim_limit": (
            "Crash-recovery engineering evidence only; final metrics, next-state "
            "transitions and formal eligibility require the completed P07 outcome."
        ),
        "scene_id": scene_id,
        "strategy": strategy,
        "action_budget_s": action_budget_s,
        "elapsed_physics_s": elapsed_physics_s,
        "decision_count": len(decisions),
        "decisions": decisions,
        "metric_samples": [
            {
                "timestamp_s": sample.timestamp_s,
                "explored_free_volume_m3": sample.explored_free_volume_m3,
                "true_free_volume_m3": sample.true_free_volume_m3,
                "predicted_free_volume_m3": sample.predicted_free_volume_m3,
                "hallucinated_free_volume_m3": sample.hallucinated_free_volume_m3,
                "coverage_fraction": sample.coverage_fraction,
            }
            for sample in samples
        ],
    }
    payload["progress_record_sha256"] = canonical_sha256(payload)
    write_json_atomic(_progress_path(output), payload)


def _point(raw: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    point = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label} must be finite")
    return point


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "collision": args.collision_usd.expanduser().resolve(),
        "start_resets": args.start_reset_json.expanduser().resolve(),
        "flight": args.flight_space_audit.expanduser().resolve(),
        "p03": args.p03_artifact.expanduser().resolve(),
        "p04": args.p04_artifact.expanduser().resolve(),
        "p05": args.p05_artifact.expanduser().resolve(),
        "p06": args.p06_artifact.expanduser().resolve(),
        "timing": args.transit_time_model_json.expanduser().resolve(),
        "communication": args.communication_contract_json.expanduser().resolve(),
        "cf2x": args.cf2x_usd.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
    }
    for name, path in paths.items():
        if name != "output" and not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite runtime evidence: {paths['output']}")
    return paths


def _p03_row(p03: dict[str, Any], scene_id: str) -> dict[str, Any]:
    payload = _artifact_payload(p03, "P03")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("P03 artifact lacks scene rows")
    rows = [row for row in scenes if isinstance(row, dict) and row.get("scene_id") == scene_id]
    if len(rows) != 1:
        raise ValueError("P03 artifact does not contain exactly one matching scene")
    row = rows[0]
    if (
        row.get("free_flight_validated") is not True
        or row.get("collision_replay_passed") is not True
    ):
        raise ValueError("P03 scene has not passed real free-flight and collision replay")
    return row


def _contract_hashes(
    *, p04: dict[str, Any], p06: dict[str, Any], p03_row: dict[str, Any], scene_id: str
) -> tuple[str, str, SensorProfile]:
    p04_payload = _artifact_payload(p04, "P04")
    p06_payload = _artifact_payload(p06, "P06")
    public_hash = p04_payload.get("public_contract_sha256")
    denominator_hash = p04_payload.get("evaluation_denominator_sha256")
    if not isinstance(public_hash, str) or not isinstance(denominator_hash, str):
        raise ValueError("P04 hashes are missing")
    int(public_hash, 16)
    int(denominator_hash, 16)
    episodes = p04_payload.get("episodes")
    if not isinstance(episodes, list) or not any(
        isinstance(row, dict)
        and row.get("scene_id") == scene_id
        and row.get("source_observation_binding") is True
        for row in episodes
    ):
        raise ValueError("P04 has no source-bound public observation row for this scene")
    selected = p06_payload.get("selected_profile")
    if not isinstance(selected, dict):
        raise ValueError("P06 selected profile is missing")
    profile = SensorProfile.from_dict(selected)
    if profile.mode != "sparse_range_3d":
        raise ValueError("P07 formal exploration requires the P06 sparse-range selection")
    if p03_row.get("resolution_m") != 0.25 or p03_row.get("vehicle_clearance_m") != 0.3:
        raise ValueError("P03 formal scene uses an unexpected scoring geometry contract")
    return public_hash, denominator_hash, profile


def _frozen_split_manifest_hash(p05: dict[str, Any], *, scene_id: str, split: str) -> str:
    """Bind each worker and QD history row to the immutable P05 scene split."""

    payload = _artifact_payload(p05, "P05")
    assignments = payload.get("scene_assignments")
    declared_hash = payload.get("split_manifest_sha256")
    if not isinstance(assignments, list) or not isinstance(declared_hash, str):
        raise ValueError("P05 scene split artifact is incomplete")
    rows: list[dict[str, str]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("P05 scene assignment is malformed")
        row = {field: assignment.get(field) for field in ("scene_id", "split", "asset_sha256")}
        if any(not isinstance(value, str) or not value for value in row.values()):
            raise ValueError("P05 scene assignment fields are malformed")
        if len(row["asset_sha256"]) != 64:
            raise ValueError("P05 scene assignment asset hash is malformed")
        int(row["asset_sha256"], 16)
        rows.append(row)
    if canonical_sha256(sorted(rows, key=lambda row: row["scene_id"])) != declared_hash:
        raise ValueError("P05 split manifest hash does not match its scene assignments")
    matching = [row for row in rows if row["scene_id"] == scene_id]
    if len(matching) != 1 or matching[0]["split"] != split:
        raise ValueError("P07 scene and requested partition disagree with P05 freeze")
    return declared_hash


def _initial_position_candidates(
    source: dict[str, Any],
    *,
    p03_row: dict[str, Any],
    collision_usd_sha256: str,
) -> tuple[tuple[float, float, float], ...]:
    """Load only a dedicated P07 environment reset manifest.

    P04 positions calibrate sparse range outcomes and are evaluator-side.  They
    must never double as P07 starts: their count and spatial design differ,
    and that would couple environment initialization to sensor calibration.
    """

    if source.get("schema_version") != P07_START_RESET_SCHEMA_VERSION:
        raise ValueError(
            "P07 requires a dedicated start-reset manifest, never P04 calibration views"
        )
    if source.get("status") != "P07_START_RESET_CANDIDATES_READY":
        raise ValueError("P07 start-reset manifest is not ready")
    if source.get("synthetic") is not False or source.get("formal_result") is not False:
        raise ValueError("P07 start-reset manifest is not admissible environment evidence")
    if source.get("evidence_class") != "environment_reset_pre_registration":
        raise ValueError("P07 start-reset manifest has the wrong evidence class")
    if source.get("method_visible") is not False:
        raise ValueError("P07 start-reset generator details must remain method-invisible")
    if source.get("source_glb_sha256") != p03_row.get("source_geometry_sha256"):
        raise ValueError("P07 start-reset source geometry differs from P03")
    if source.get("collision_usd_sha256") != collision_usd_sha256:
        raise ValueError("P07 start-reset collision geometry differs from P03 runtime")
    if source.get("flight_space_manifest_hash") != p03_row.get("flight_space_manifest_hash"):
        raise ValueError("P07 start-reset flight-space geometry differs from P03")
    recorded_hash = source.get("start_reset_sha256")
    unsigned = dict(source)
    unsigned.pop("start_reset_sha256", None)
    if not isinstance(recorded_hash, str) or canonical_sha256(unsigned) != recorded_hash:
        raise ValueError("P07 start-reset manifest hash is invalid")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or source.get("candidate_count") != len(candidates):
        raise ValueError("P07 start-reset candidates are malformed")
    points: list[tuple[float, float, float]] = []
    candidate_ids: set[str] = set()
    for row in candidates:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise ValueError("P07 start-reset candidate is malformed")
        candidate_id = row["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError("P07 start-reset candidates are duplicated")
        candidate_ids.add(candidate_id)
        points.append(_point(row.get("position_w_m"), "P07 start-reset position_w_m"))
    separation = float(source.get("minimum_pairwise_separation_m", 0.0))
    if separation <= 0.0:
        raise ValueError("P07 start-reset separation is invalid")
    if any(
        math.dist(left, right) < separation - 1.0e-9
        for index, left in enumerate(points)
        for right in points[index + 1 :]
    ):
        raise ValueError("P07 start-reset candidates violate their declared separation")
    if len(points) < FORMAL_FLEET_SIZE * 2:
        raise ValueError("P07 start-reset manifest is too small for the formal fleet")
    return tuple(points)


def _p0_departure_envelope_audit(source: dict[str, Any]) -> dict[str, object]:
    """Validate the reset-generation contract required by a P0 route audit.

    A point that only meets the physical hold threshold can be safe to
    initialise yet have no admissible first movement sample. P0 proves a
    usable four-agent exploration action, so it requires candidates generated
    at the already-frozen interior route-sample clearance. Legacy reset
    manifests remain loadable for engineering diagnosis, but cannot close
    M4/M5.
    """

    declared_clearance = source.get("start_mobility_clearance_m")
    if (
        not isinstance(declared_clearance, (int, float))
        or isinstance(declared_clearance, bool)
        or not math.isfinite(float(declared_clearance))
    ):
        raise ValueError("P0 start-reset manifest lacks a finite mobility-clearance contract")
    required_clearance = cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M
    if float(declared_clearance) + 1.0e-9 < required_clearance:
        raise ValueError(
            "P0 start-reset manifest admits only terminal clearance; "
            "regenerate it with the route-sample clearance contract"
        )
    if source.get("selection_rule") != P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE:
        raise ValueError(
            "P0 start-reset manifest uses a legacy selection rule; "
            "regenerate it with the route-sample clearance contract"
        )
    contract = source.get("departure_witness_contract")
    if not isinstance(contract, dict):
        raise ValueError("P0 start-reset manifest lacks a departure-witness contract")
    if contract.get("schema_version") != P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION:
        raise ValueError("P0 start-reset departure-witness schema is invalid")
    if contract.get("selection_rule") != "six-neighbour-grid-tube+exact-static-samples-v1":
        raise ValueError("P0 start-reset departure-witness selection rule is invalid")
    required_route_sample_clearance = cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M
    if (
        not isinstance(contract.get("required_internal_sample_clearance_m"), (int, float))
        or float(contract["required_internal_sample_clearance_m"]) + 1.0e-9
        < required_route_sample_clearance
    ):
        raise ValueError("P0 departure-witness contract weakens the route-sample clearance")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or source.get("candidate_count") != len(candidates):
        raise ValueError("P0 departure-witness candidates are malformed")
    witness_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("P0 departure-witness candidate is malformed")
        witnesses = candidate.get("static_departure_witnesses")
        if not isinstance(witnesses, list) or not witnesses:
            raise ValueError("P0 reset candidate lacks a static departure witness")
        admitted_witnesses = [
            witness
            for witness in witnesses
            if isinstance(witness, dict)
            and witness.get("schema_version") == P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION
            and witness.get("offline_exact_admitted") is True
        ]
        if not admitted_witnesses:
            raise ValueError("P0 reset candidate has no exact-mesh-admitted departure witness")
        for witness in admitted_witnesses:
            path = witness.get("path_m")
            if (
                not isinstance(path, list)
                or len(path) != 2
                or any(not isinstance(point, list) or len(point) != 3 for point in path)
            ):
                raise ValueError("P0 departure witness path is malformed")
            if math.dist(tuple(float(value) for value in path[0]), tuple(float(value) for value in path[1])) <= 1.0e-6:
                raise ValueError("P0 departure witness must be nonzero")
            witness_count += 1
    return {
        "selection_rule": P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE,
        "declared_start_mobility_clearance_m": float(declared_clearance),
        "required_route_sample_clearance_m": required_clearance,
        "departure_witness_schema_version": P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION,
        "departure_witness_count": witness_count,
        "departure_witness_contract_sha256": canonical_sha256(contract),
        "admitted": True,
    }


def _p0_static_departure_witness_paths(
    source: dict[str, Any],
    *,
    selected_candidate_ids: Sequence[str],
    selected_positions: Sequence[tuple[float, float, float]],
) -> tuple[dict[str, object], ...]:
    """Resolve immutable reset probes without exposing their geometry to policy code.

    The reset manifest pre-registers short, nonzero departure paths.  This
    helper only validates and resolves them for the P0 evidence path.  The
    live worker still runs every path through the shared PhysX guard, and none
    of these endpoints may enter the public belief, candidate pool, reward,
    QD archive, or replay.
    """

    candidates = source.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("P0 reset manifest candidates are missing")
    candidate_by_id = {
        row.get("candidate_id"): row
        for row in candidates
        if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
    }
    if len(candidate_by_id) != len(candidates):
        raise ValueError("P0 reset manifest candidate IDs are malformed")
    if len(selected_candidate_ids) != len(selected_positions):
        raise ValueError("P0 departure witness IDs and positions are misaligned")

    resolved: list[dict[str, object]] = []
    for candidate_id, selected_position in zip(
        selected_candidate_ids, selected_positions, strict=True
    ):
        row = candidate_by_id.get(candidate_id)
        if row is None:
            raise ValueError(f"P0 departure witness names unknown candidate: {candidate_id}")
        declared_position = _point(row.get("position_w_m"), "P0 reset candidate position_w_m")
        if math.dist(declared_position, selected_position) > 1.0e-9:
            raise ValueError(f"P0 selected position differs from reset candidate: {candidate_id}")
        raw_witnesses = row.get("static_departure_witnesses")
        if not isinstance(raw_witnesses, list) or not raw_witnesses:
            raise ValueError(f"P0 reset candidate has no departure witnesses: {candidate_id}")
        witness_ids: set[str] = set()
        witnesses: list[dict[str, object]] = []
        for raw_witness in raw_witnesses:
            if not isinstance(raw_witness, dict):
                raise ValueError(f"P0 departure witness is malformed: {candidate_id}")
            witness_id = raw_witness.get("witness_id")
            if not isinstance(witness_id, str) or not witness_id or witness_id in witness_ids:
                raise ValueError(f"P0 departure witness IDs are malformed: {candidate_id}")
            witness_ids.add(witness_id)
            if raw_witness.get("schema_version") != P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION:
                raise ValueError(f"P0 departure witness schema is invalid: {candidate_id}")
            if raw_witness.get("offline_exact_admitted") is not True:
                raise ValueError(f"P0 departure witness was not admitted offline: {candidate_id}")
            raw_path = raw_witness.get("path_m")
            if not isinstance(raw_path, list) or len(raw_path) < 2:
                raise ValueError(f"P0 departure witness path is too short: {candidate_id}")
            path = tuple(
                _point(point, "P0 departure witness path point") for point in raw_path
            )
            if math.dist(path[0], declared_position) > 1.0e-9:
                raise ValueError(f"P0 departure witness does not start at candidate: {candidate_id}")
            if _path_length_m(path) <= 1.0e-6:
                raise ValueError(f"P0 departure witness is zero-length: {candidate_id}")
            witnesses.append(
                {
                    "witness_id": witness_id,
                    "path_m": path,
                    "path_sha256": canonical_sha256(path),
                }
            )
        resolved.append({"candidate_id": candidate_id, "witnesses": tuple(witnesses)})
    return tuple(resolved)


def _p0_live_departure_qualification(
    source: dict[str, Any],
    *,
    selected_candidate_ids: Sequence[str],
    selected_positions: Sequence[tuple[float, float, float]],
    observed_positions: Sequence[tuple[float, float, float]],
    scene_query: Any,
    clearance_oracle: Any,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    collision_usd_sha256: str,
    start_reset_manifest_sha256: str,
) -> dict[str, object]:
    """Certify a nonzero live departure under the exact runtime guard.

    This path is called only for P0 audit/full-evidence runs after Isaac has
    created the scene query and clearance oracle.  It is reset qualification,
    not an exploration action: the record retains IDs, hashes and verdicts,
    never the witness geometry itself.
    """

    if len(selected_positions) != len(observed_positions):
        raise ValueError("P0 selected and observed reset positions are misaligned")
    resolved = _p0_static_departure_witness_paths(
        source,
        selected_candidate_ids=selected_candidate_ids,
        selected_positions=selected_positions,
    )
    departure_contract = source.get("departure_witness_contract")
    if not isinstance(departure_contract, dict):
        raise ValueError("P0 reset manifest lacks departure-witness contract")
    guard_contract = {
        "guard_entrypoint": "aerocity_method.runtime.hm3d_cf2x_execution._routed_guard",
        "allow_public_reroute": False,
        "public_waypoints": "empty",
        "required_route_sample_clearance_m": cf2x.REQUIRED_ROUTE_SAMPLE_CLEARANCE_M,
        "required_terminal_clearance_m": cf2x.REQUIRED_TERMINAL_CLEARANCE_M,
        "flight_clearance_m": cf2x.FLIGHT_CLEARANCE_M,
        "bounds_sha256": canonical_sha256(
            {"bounds_min_m": bounds_min, "bounds_max_m": bounds_max}
        ),
        "collision_usd_sha256": collision_usd_sha256,
        "start_reset_manifest_sha256": start_reset_manifest_sha256,
        "departure_witness_contract_sha256": canonical_sha256(departure_contract),
    }
    guard_contract_sha256 = canonical_sha256(guard_contract)
    candidate_results: list[dict[str, object]] = []
    for resolved_row, observed_position in zip(resolved, observed_positions, strict=True):
        candidate_id = str(resolved_row["candidate_id"])
        raw_witnesses = resolved_row["witnesses"]
        if not isinstance(raw_witnesses, tuple):
            raise RuntimeError("P0 departure witness resolver returned an invalid witness list")
        witness_results: list[dict[str, object]] = []
        for witness in raw_witnesses:
            if not isinstance(witness, dict):
                raise RuntimeError("P0 departure witness resolver returned an invalid witness")
            witness_id = str(witness["witness_id"])
            raw_path = witness["path_m"]
            if not isinstance(raw_path, tuple):
                raise RuntimeError("P0 departure witness resolver returned an invalid path")
            path = raw_path
            live_path = (tuple(observed_position), *path[1:])
            nonzero = _path_length_m(live_path) > 1.0e-6
            guarded = None
            if nonzero:
                guarded = cf2x._routed_guard(
                    scene_query,
                    clearance_oracle,
                    (),
                    f"p0-start-{candidate_id}",
                    live_path,
                    bounds_min,
                    bounds_max,
                    None,
                    allow_public_reroute=False,
                )
            legal = bool(nonzero and guarded is not None and guarded.legal)
            witness_results.append(
                {
                    "candidate_id": candidate_id,
                    "witness_id": witness_id,
                    "static_path_sha256": str(witness["path_sha256"]),
                    "live_path_sha256": canonical_sha256(live_path),
                    "guarded_path_sha256": (
                        None if guarded is None else canonical_sha256(tuple(guarded.path_m))
                    ),
                    "nonzero_first_hop": nonzero,
                    "legal": legal,
                    "reason": (
                        "zero_length_live_path"
                        if guarded is None
                        else ("admitted" if guarded.legal else str(guarded.reason or "rejected"))
                    ),
                }
            )
        live_legal_witness_count = sum(bool(row["legal"]) for row in witness_results)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "witness_count": len(witness_results),
                "live_legal_witness_count": live_legal_witness_count,
                "passed": live_legal_witness_count > 0,
                "witnesses": witness_results,
            }
        )
    unsigned = {
        "schema_version": "hm3d-p07-live-start-departure-qualification-v1",
        "guard_contract": guard_contract,
        "guard_contract_sha256": guard_contract_sha256,
        "selected_candidate_ids": list(selected_candidate_ids),
        "candidates": candidate_results,
        "passed": all(bool(row["passed"]) for row in candidate_results),
        "claim_limit": (
            "P0 reset qualification only. It does not create public map/free-space evidence, "
            "a candidate action, reward, QD entry or replay transition."
        ),
    }
    return {**unsigned, "qualification_sha256": canonical_sha256(unsigned)}


def _select_connected_initial_positions(
    candidates: tuple[tuple[float, float, float], ...],
    graph_for_positions: Any,
) -> tuple[tuple[float, float, float], ...]:
    """Choose a pre-audited, separated and relay-connected common reset.

    The only query here is the same real range/LOS communication telemetry
    later recorded by the executor.  It reads neither the evaluator ESDF nor
    any target truth, and is run once before all methods face an identical
    reset.  A greedy connected expansion keeps the formal four-CF2X admission
    bounded rather than enumerating every view combination.
    """

    if len(candidates) < FORMAL_FLEET_SIZE * 2:
        raise ValueError("initial position candidates do not satisfy the frozen fleet contract")
    for seed_index, seed in enumerate(candidates):
        selected = [seed]
        remaining = [
            point
            for index, point in enumerate(candidates)
            if index != seed_index and math.dist(point, seed) >= 0.75
        ]
        while len(selected) < FORMAL_FLEET_SIZE:
            next_point = next(
                (
                    point
                    for point in remaining
                    if all(math.dist(point, existing) >= 0.75 for existing in selected)
                    and graph_for_positions(tuple((*selected, point))).fully_relay_connected
                ),
                None,
            )
            if next_point is None:
                break
            selected.append(next_point)
            remaining.remove(next_point)
        if len(selected) == FORMAL_FLEET_SIZE:
            return tuple(selected)
    raise ValueError("public initial-position source has no relay-connected fleet reset")


def _select_explicit_initial_positions(
    candidates: tuple[tuple[float, float, float], ...],
    candidate_ids: tuple[str, ...],
    selected_candidate_ids: Sequence[str],
    graph_for_positions: Any,
) -> tuple[tuple[float, float, float], ...]:
    """Return one auditable P0 reset without falling back to greedy selection.

    P0 eligibility audits need to test a pre-declared fleet placement under
    the exact later bootstrap and route guards.  This helper merely resolves
    IDs from the immutable reset manifest and retains the regular separation
    and relay checks; it never uses the evaluation ESDF or target geometry.
    """

    if len(candidates) != len(candidate_ids):
        raise ValueError("P07 start candidates and IDs are misaligned")
    if len(selected_candidate_ids) != FORMAL_FLEET_SIZE:
        raise ValueError("explicit P0 reset must name every formal fleet member")
    if len(set(selected_candidate_ids)) != FORMAL_FLEET_SIZE:
        raise ValueError("explicit P0 reset candidate IDs must be unique")
    positions_by_id = dict(zip(candidate_ids, candidates, strict=True))
    unknown_ids = sorted(set(selected_candidate_ids).difference(positions_by_id))
    if unknown_ids:
        raise ValueError(f"explicit P0 reset names unknown candidate IDs: {unknown_ids}")
    selected = tuple(positions_by_id[candidate_id] for candidate_id in selected_candidate_ids)
    if any(
        math.dist(left, right) < 0.75 - 1.0e-9
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ):
        raise ValueError("explicit P0 reset violates minimum fleet separation")
    if not graph_for_positions(selected).fully_relay_connected:
        raise ValueError("explicit P0 reset is not relay connected")
    return selected


def _relay_connected_start_id_combinations(
    candidates: tuple[tuple[float, float, float], ...],
    candidate_ids: tuple[str, ...],
    graph_for_positions: Any,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate only pre-registered start quartets admitted by the real relay query."""

    if len(candidates) != len(candidate_ids):
        raise ValueError("P07 start candidates and IDs are misaligned")
    if len(candidates) < FORMAL_FLEET_SIZE:
        raise ValueError("P07 start candidates cannot form the formal fleet")
    admitted: list[tuple[str, ...]] = []
    for indices in itertools.combinations(range(len(candidates)), FORMAL_FLEET_SIZE):
        positions = tuple(candidates[index] for index in indices)
        if graph_for_positions(positions).fully_relay_connected:
            admitted.append(tuple(candidate_ids[index] for index in indices))
    return tuple(admitted)


def _validated_p0_start_eligibility_evidence(
    payload: dict[str, Any],
    *,
    scene_id: str,
    start_reset_manifest_sha256: str,
    controller_id: str,
    transit_time_model_sha256: str,
    p0_eligibility_contract: dict[str, object],
    requested_candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate that a P0 full episode reuses an all-active audited reset."""

    supplied_hash = payload.get("audit_record_sha256")
    unsigned = dict(payload)
    unsigned.pop("audit_record_sha256", None)
    if not isinstance(supplied_hash, str) or canonical_sha256(unsigned) != supplied_hash:
        raise ValueError("P0 start eligibility evidence hash is invalid")
    if payload.get("schema_version") != "hm3d-p07-start-eligibility-audit-v1":
        raise ValueError("P0 start eligibility evidence schema is invalid")
    if payload.get("status") != "P07_START_ELIGIBILITY_AUDIT_COMPLETE":
        raise ValueError("P0 start eligibility evidence did not complete")
    if payload.get("scene_id") != scene_id:
        raise ValueError("P0 start eligibility evidence scene differs from the episode")
    if payload.get("controller_id") != controller_id:
        raise ValueError("P0 start eligibility evidence controller differs from the episode")
    if payload.get("transit_time_model_sha256") != transit_time_model_sha256:
        raise ValueError("P0 start eligibility evidence timing profile differs from the episode")
    evidence_contract = payload.get("p0_eligibility_contract")
    if not isinstance(evidence_contract, dict) or (
        _normalized_p0_eligibility_contract(evidence_contract)
        != _normalized_p0_eligibility_contract(p0_eligibility_contract)
    ):
        raise ValueError("P0 start eligibility evidence execution contract differs from the episode")
    if payload.get("start_reset_manifest_sha256") != start_reset_manifest_sha256:
        raise ValueError("P0 start eligibility evidence reset manifest differs from the episode")
    initial = payload.get("initial_start_reset")
    first_pool = payload.get("first_pool")
    if not isinstance(initial, dict) or not isinstance(first_pool, dict):
        raise ValueError("P0 start eligibility evidence is incomplete")
    audited_ids = initial.get("selected_start_candidate_ids")
    if not isinstance(audited_ids, list) or not all(isinstance(value, str) for value in audited_ids):
        raise ValueError("P0 start eligibility evidence has invalid candidate IDs")
    if tuple(audited_ids) != tuple(requested_candidate_ids):
        raise ValueError("P0 episode start candidate IDs differ from eligibility evidence")
    live_departure = initial.get("p0_live_departure_qualification")
    if not isinstance(live_departure, dict):
        raise ValueError("P0 start eligibility evidence lacks live departure qualification")
    if live_departure.get("schema_version") != "hm3d-p07-live-start-departure-qualification-v1":
        raise ValueError("P0 start eligibility evidence has an invalid live departure schema")
    if tuple(live_departure.get("selected_candidate_ids", ())) != tuple(requested_candidate_ids):
        raise ValueError("P0 live departure candidate IDs differ from eligibility evidence")
    if live_departure.get("passed") is not True:
        raise ValueError("P0 start eligibility evidence lacks a live departure for every agent")
    if first_pool.get("all_agents_active_candidate_exists") is not True:
        raise ValueError("P0 start eligibility evidence lacks an all-active candidate")
    return tuple(audited_ids)


def _bootstrap_manifest(
    context: PublicMethodContext,
    positions: tuple[tuple[float, float, float], ...],
    transit_timing: Any,
    *,
    observe_dwell_s: float,
) -> CandidateFragmentManifest:
    """Build the shared physical sensing action before any selector ranks.

    A cold-start map cannot honestly authorize a directional frontier.  The
    protocol therefore gives every method the same short hover-and-measure
    action and charges it to the episode clock.  The two equal path points are
    an explicit hold command for the CF2X executor, not an unrecorded reset or
    a geometry query.
    """

    fragments: list[FragmentInstance] = []
    total_cost = 0.0
    for index, position in enumerate(positions):
        agent_id = f"uav{index}"
        transit_duration = transit_timing.estimate_seconds((position, position))
        observation_end = transit_duration + observe_dwell_s
        total_cost += transit_duration
        fragments.extend(
            (
                FragmentInstance(
                    instance_fragment_id=f"bootstrap-{agent_id}-hold",
                    type_signature=FragmentTypeSignature("transit", (("bootstrap", 1.0),)),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=0.0,
                    planned_end=transit_duration,
                    path=(position, position),
                    pose_mode="guarded_waypoint",
                    context_bucket="hm3d-public-bootstrap",
                ),
                FragmentInstance(
                    instance_fragment_id=f"bootstrap-{agent_id}-observe",
                    type_signature=FragmentTypeSignature("observation", (("bootstrap", 1.0),)),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=transit_duration,
                    planned_end=observation_end,
                    path=(position,),
                    pose_mode="dwell",
                    context_bucket="hm3d-public-bootstrap",
                ),
            )
        )
    return CandidateFragmentManifest(
        candidate_id="hm3d-public-bootstrap",
        context_hash=context.digest,
        fragments=tuple(fragments),
        planned_descriptor=(0.0, 1.0, 0.0),
        feasible=True,
        quality_hint=0.0,
        cost_hint=total_cost,
        source="hm3d-public-bootstrap-v1",
        admission_reasons=(),
    )


def _min_stationary_budget_tail_s(
    *,
    observation_dwell_s: float,
    physics_dt_s: float,
) -> float:
    """Return the shortest shared stationary tail that can contain one dwell."""
    return observation_dwell_s + physics_dt_s


def _budget_tail_manifest(
    context: PublicMethodContext,
    positions: tuple[tuple[float, float, float], ...],
    *,
    duration_s: float,
    observe_dwell_s: float,
) -> CandidateFragmentManifest:

    """Spend an unrouteable final budget remainder with a shared physical hold.

    Event-driven execution may leave enough time for a valid observation but
    not enough for the smallest admitted frontier leg.  This action is not a
    candidate or a method decision: every method holds at its realised final
    positions until the common episode horizon, and its map update remains
    outcome-bound.  The scheduled boundary is deliberate here; otherwise an
    event-driven stationary hold would immediately finish and recreate the
    discarded tail.
    """

    if not math.isfinite(duration_s) or duration_s < observe_dwell_s:
        raise ValueError("budget tail must accommodate the frozen observation dwell")
    transit_end_s = duration_s - observe_dwell_s
    fragments: list[FragmentInstance] = []
    for index, position in enumerate(positions):
        agent_id = f"uav{index}"
        fragments.extend(
            (
                FragmentInstance(
                    instance_fragment_id=f"budget-tail-{agent_id}-hold",
                    type_signature=FragmentTypeSignature("transit", (("budget_tail", 1.0),)),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=0.0,
                    planned_end=transit_end_s,
                    path=(position, position),
                    pose_mode="guarded_waypoint",
                    context_bucket="hm3d-public-budget-tail",
                ),
                FragmentInstance(
                    instance_fragment_id=f"budget-tail-{agent_id}-observe",
                    type_signature=FragmentTypeSignature("observation", (("budget_tail", 1.0),)),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=transit_end_s,
                    planned_end=duration_s,
                    path=(position,),
                    pose_mode="dwell",
                    context_bucket="hm3d-public-budget-tail",
                ),
            )
        )
    return CandidateFragmentManifest(
        candidate_id="hm3d-public-budget-tail",
        context_hash=context.digest,
        fragments=tuple(fragments),
        planned_descriptor=(0.0, 1.0, 0.0),
        feasible=True,
        quality_hint=0.0,
        cost_hint=0.0,
        source="hm3d-public-budget-tail-v1",
        admission_reasons=(),
    )


def _unexecuted_budget_tail_record(
    *,
    duration_s: float,
    observe_dwell_s: float,
) -> dict[str, object]:
    """Record a final remainder that is too short to authorize an observation.

    This is deliberately not a manifest.  The remainder is accounted for by
    the fixed-horizon metric, but no physical observation, reward, or outcome
    may be fabricated for it.  ``_budget_tail_manifest`` remains strict for
    the executable case where the frozen dwell can actually be completed.
    """

    if (
        not math.isfinite(duration_s)
        or duration_s < 0.0
        or not math.isfinite(observe_dwell_s)
        or observe_dwell_s <= 0.0
        or duration_s >= observe_dwell_s
    ):
        raise ValueError("unexecuted budget tail must be non-negative and shorter than dwell")
    return {
        "manifest_hash": None,
        "elapsed_physics_s": 0.0,
        "unexecuted_remainder_s": duration_s,
        "scheduled_completion_mode": "unexecuted_budget_remainder_below_observation_dwell",
        "execution": None,
    }


def _nearest_public_free_key(
    belief: SparseVoxelBelief,
    point_m: tuple[float, float, float],
    *,
    maximum_distance_m: float,
) -> tuple[int, int, int] | None:
    """Resolve a physical point to an observed-free voxel without evaluator truth."""

    direct = belief.world_to_voxel(point_m)
    if belief.state(direct) == FREE:
        return direct
    radius = max(1, math.ceil(maximum_distance_m / belief.resolution_m))
    nearest: tuple[float, tuple[int, int, int]] | None = None
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                key = (direct[0] + dx, direct[1] + dy, direct[2] + dz)
                if belief.state(key) != FREE:
                    continue
                distance = math.dist(point_m, belief.voxel_center(key))
                if distance > maximum_distance_m + 1.0e-9:
                    continue
                candidate = (distance, key)
                if nearest is None or candidate < nearest:
                    nearest = candidate
    return None if nearest is None else nearest[1]


class _ReceivedFreeSupport(NamedTuple):
    """Local public-map support summary for one received-free voxel."""

    free_voxel_count: int
    signed_axis_count: int
    balanced_axis_count: int


_ReceivedFreeSupportCache = dict[
    tuple[tuple[int, int, int], float], _ReceivedFreeSupport
]


def _received_free_support(
    belief: SparseVoxelBelief,
    key: tuple[int, int, int],
    *,
    radius_m: float,
    cache: _ReceivedFreeSupportCache | None = None,
) -> _ReceivedFreeSupport:
    """Summarize spatially distributed received-free evidence around ``key``.

    Sparse range data cannot certify collision clearance. It can still reject
    a candidate whose command centre is only supported by a single free ray.
    The runtime guard remains the only authority that grants a route access to
    collision geometry. This helper intentionally sees only the team belief.
    """

    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("received-free support radius must be finite and non-negative")
    cache_key = (key, radius_m)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    if belief.state(key) != FREE:
        result = _ReceivedFreeSupport(0, 0, 0)
        if cache is not None:
            cache[cache_key] = result
        return result
    radius_cells = math.ceil(radius_m / belief.resolution_m)
    radius_squared = radius_m * radius_m + 1.0e-12
    free_voxel_count = 0
    signed_axes: set[tuple[int, int]] = set()
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            for dz in range(-radius_cells, radius_cells + 1):
                if (dx * belief.resolution_m) ** 2 + (dy * belief.resolution_m) ** 2 + (
                    dz * belief.resolution_m
                ) ** 2 > radius_squared:
                    continue
                neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
                if belief.state(neighbor) == FREE:
                    free_voxel_count += 1
                    for axis, offset in enumerate((dx, dy, dz)):
                        if offset:
                            signed_axes.add((axis, 1 if offset > 0 else -1))
    balanced_axis_count = sum(
        (axis, -1) in signed_axes and (axis, 1) in signed_axes for axis in range(3)
    )
    result = _ReceivedFreeSupport(
        free_voxel_count=free_voxel_count,
        signed_axis_count=len(signed_axes),
        balanced_axis_count=balanced_axis_count,
    )
    if cache is not None:
        cache[cache_key] = result
    return result


def _has_received_free_interior_support(
    belief: SparseVoxelBelief,
    key: tuple[int, int, int],
    *,
    radius_m: float,
    cache: _ReceivedFreeSupportCache | None = None,
) -> bool:
    """Require a minimum local received-free neighborhood around a route voxel."""

    return (
        _received_free_support(belief, key, radius_m=radius_m, cache=cache).free_voxel_count
        >= PUBLIC_ROUTE_SUPPORT_MIN_FREE_VOXELS
    )


class _PublicFreePathResult(NamedTuple):
    path_m: tuple[tuple[float, float, float], ...] | None
    status: str
    # The exact public FREE voxel chain used by the bounded BFS.  Keeping this
    # private evidence lets route-progress candidates reuse a verified prefix
    # instead of running a second BFS to the same route's intermediate point.
    voxel_keys: tuple[tuple[int, int, int], ...] = ()


class _PublicFreeReachabilityCache:
    """Decision-scoped component cache for public FREE path diagnostics.

    A bounded public route search needs to distinguish a target outside the
    receding horizon from one in a disconnected received-free component.  The
    old implementation repeated an unbounded 26-neighbour flood for every
    bounded miss.  This cache performs that flood at most once for each source
    component in one immutable belief version.  It is a diagnostic cache only:
    the bounded BFS and its reconstructed path remain the path authority.
    """

    def __init__(self, belief: SparseVoxelBelief) -> None:
        self._belief = belief
        self._belief_version_sha256 = belief.version().digest
        self._component_ids: dict[tuple[int, int, int], int] = {}
        self._next_component_id = 0
        self._component_flood_count = 0
        self._cached_disconnected_rejection_count = 0
        self._bounded_path_search_count = 0
        self._bounded_path_expanded_node_count = 0
        self._route_prefix_reuse_count = 0
        self._route_prefix_fallback_count = 0
        self._region_access_attempt_count = 0
        self._region_access_generated_count = 0
        self._region_access_failure_reason_counts: Counter[str] = Counter()

    def _require_same_belief(self, belief: SparseVoxelBelief) -> None:
        if belief is not self._belief:
            raise ValueError("public FREE reachability cache belongs to a different belief")

    def record_route_prefix_reuse(self) -> None:
        """Record a route-progress prefix obtained from an existing BFS chain."""

        self._route_prefix_reuse_count += 1

    def record_route_prefix_fallback(self) -> None:
        """Record a conservative fallback search for a non-chain prefix point."""

        self._route_prefix_fallback_count += 1

    def record_region_access_generation(self) -> None:
        """Record one emitted region-access viewpoint proposal."""

        self._region_access_generated_count += 1

    def record_region_access_failure(self, reason: str) -> None:
        """Record why a long region-access attempt did not emit a viewpoint."""

        if not reason:
            raise ValueError("region-access failure reason cannot be empty")
        self._region_access_failure_reason_counts[str(reason)] += 1

    def source_and_goal_share_component(
        self,
        start_key: tuple[int, int, int],
        goal_key: tuple[int, int, int],
    ) -> bool:
        """Return component connectivity after flooding an uncached source once."""

        start_component = self._component_ids.get(start_key)
        source_component_was_cached = start_component is not None
        if start_component is None:
            component_id = self._next_component_id
            self._next_component_id += 1
            queue: deque[tuple[int, int, int]] = deque((start_key,))
            self._component_ids[start_key] = component_id
            self._component_flood_count += 1
            while queue:
                current = queue.popleft()
                for neighbor in neighbors_26(current):
                    if neighbor in self._component_ids or self._belief.state(neighbor) != FREE:
                        continue
                    self._component_ids[neighbor] = component_id
                    queue.append(neighbor)
            start_component = component_id
        goal_component = self._component_ids.get(goal_key)
        connected = goal_component == start_component
        if not connected and source_component_was_cached:
            self._cached_disconnected_rejection_count += 1
        return connected

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "hm3d-public-free-reachability-cache-v1",
            "belief_version_sha256": self._belief_version_sha256,
            "component_flood_count": self._component_flood_count,
            "component_cache_free_voxel_count": len(self._component_ids),
            "component_cached_disconnected_rejections": self._cached_disconnected_rejection_count,
            "bounded_path_search_count": self._bounded_path_search_count,
            "bounded_path_expanded_node_count": self._bounded_path_expanded_node_count,
            "route_prefix_reuse_count": self._route_prefix_reuse_count,
            "route_prefix_fallback_count": self._route_prefix_fallback_count,
            "region_access_attempt_count": self._region_access_attempt_count,
            "region_access_generated_count": self._region_access_generated_count,
            "region_access_failure_reason_counts": dict(
                sorted(self._region_access_failure_reason_counts.items())
            ),
            "claim_limit": (
                "Decision-scoped public FREE connectivity diagnostic cache. It does not "
                "alter bounded route search, candidate ranking, safety, reward, or execution. "
                "A reused route-progress prefix is a prefix of an already admitted public "
                "FREE voxel chain; fallback searches retain the previous path authority. "
                "Region-access counts are generation diagnostics only and do not admit "
                "safety or performance claims."
            ),
        }


class _PublicFrontierViewpoint(NamedTuple):
    """One observation point derived solely from the shared sparse belief.

    ``route_lengths_m`` is public-map reachability evidence, not a collision
    clearance estimate. The common PhysX guard remains the only authority
    that can certify a route.
    """

    information_gain: float
    traversal_risk: float
    position_m: tuple[float, float, float]
    source_agent_index: int
    route_lengths_m: tuple[float | None, ...]
    # Each non-null route is constructed from this exact decision's public
    # belief and starts at the corresponding public agent pose.  It is carried
    # through into the common candidate pool instead of being reduced to a
    # scalar length and re-derived from the endpoint later.
    route_paths_m: tuple[tuple[tuple[float, float, float], ...] | None, ...]
    viewpoint_kind: str
    frontier_cluster_id: str = "legacy-unclustered"
    task_anchor_m: tuple[float, float, float] | None = None
    task_normal_unit: tuple[float, float, float] | None = None
    observation_standoff_m: float | None = None


def _retain_route_progress_viewpoints(
    source_rows: Sequence[tuple[tuple[int, int, int], _PublicFrontierViewpoint]],
    *,
    source_position_m: tuple[float, float, float],
    source_agent_index: int,
    maximum_count: int,
    resolution_m: float,
) -> tuple[tuple[tuple[int, int, int], _PublicFrontierViewpoint], ...]:
    """Keep efficient, long, and spatially distinct public route prefixes.

    A route-progress or region-access point is only a public-map proposal; the
    common static guard remains its physical admission authority. Retaining a
    single pre-guard proposal per source caused one rejected endpoint to erase
    other supported prefixes from the bounded action authority. This helper
    keeps a small deterministic hedge without assigning a target to that
    source vehicle or increasing the final frontier budget.
    """

    if maximum_count < 1:
        return ()
    if resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive")

    def route_length(item: tuple[tuple[int, int, int], _PublicFrontierViewpoint]) -> float:
        length_m = item[1].route_lengths_m[source_agent_index]
        if length_m is None or length_m <= 0.0:
            raise RuntimeError("route-progress source omits its source-route length")
        return float(length_m)

    def stable_key(item: tuple[tuple[int, int, int], _PublicFrontierViewpoint]) -> tuple[int, int, int]:
        # ``max`` is used below, so negate the key to choose lexicographically
        # smaller voxel keys for otherwise exact ties.
        return tuple(-coordinate for coordinate in item[0])

    remaining = list(source_rows)
    retained: list[tuple[tuple[int, int, int], _PublicFrontierViewpoint]] = []

    def take_best(key: Callable[[tuple[tuple[int, int, int], _PublicFrontierViewpoint]], tuple[object, ...]]) -> None:
        if remaining and len(retained) < maximum_count:
            chosen = max(remaining, key=key)
            retained.append(chosen)
            remaining.remove(chosen)

    # The first row prefers a committed region-access route when one exists;
    # otherwise it retains gain-efficient local progress. The next selections
    # protect non-local route alternatives without hiding short views entirely.
    take_best(
        lambda item: (
            item[1].viewpoint_kind == "region_access",
            item[1].information_gain / max(route_length(item), resolution_m),
            item[1].information_gain,
            route_length(item),
            -item[1].traversal_risk,
            stable_key(item),
        )
    )
    # Preserve a genuine route extension even when it has lower gain density.
    take_best(
        lambda item: (
            route_length(item),
            item[1].information_gain,
            -item[1].traversal_risk,
            stable_key(item),
        )
    )
    # Prefer an independently directed/vertical observation opportunity.  If
    # the scene is planar, endpoint separation still avoids a duplicate route.
    while remaining and len(retained) < maximum_count:
        take_best(
            lambda item: (
                abs(item[1].position_m[2] - source_position_m[2]),
                min(math.dist(item[1].position_m, prior[1].position_m) for prior in retained),
                item[1].information_gain / max(route_length(item), resolution_m),
                route_length(item),
                -item[1].traversal_risk,
                stable_key(item),
            )
        )
    return tuple(retained)


def _compress_public_voxel_path(
    belief: SparseVoxelBelief,
    keys: Sequence[tuple[int, int, int]],
    *,
    start_m: tuple[float, float, float],
    goal_m: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Keep voxel turns while removing grid sampling noise from a public route."""

    if len(keys) < 1:
        raise ValueError("public voxel route must contain at least one key")
    points: list[tuple[float, float, float]] = [start_m]
    previous_direction: tuple[int, int, int] | None = None
    for index, (left, right) in enumerate(zip(keys, keys[1:], strict=False), start=1):
        direction = tuple(right[axis] - left[axis] for axis in range(3))
        if previous_direction is not None and direction != previous_direction:
            turn = belief.voxel_center(keys[index - 1])
            if math.dist(points[-1], turn) > 1.0e-9:
                points.append(turn)
        previous_direction = direction
    if math.dist(points[-1], goal_m) > 1.0e-9:
        points.append(goal_m)
    if len(points) == 1:
        points.append(goal_m)
    return tuple(points)


def _public_free_space_path_result(
    belief: SparseVoxelBelief,
    start_m: tuple[float, float, float],
    goal_m: tuple[float, float, float],
    *,
    maximum_path_length_m: float,
    minimum_received_free_support_m: float = 0.0,
    received_free_support_cache: _ReceivedFreeSupportCache | None = None,
    reachability_cache: _PublicFreeReachabilityCache | None = None,
) -> _PublicFreePathResult:
    """Plan a bounded received-free route and expose the public failure class.

    Sparse range rays can advance diagonally through the voxel grid.  The
    route graph therefore uses 26-connectivity so an observed free chain is
    not relabelled as disconnected before the shared continuous PhysX guard
    has a chance to evaluate its geometry.  This changes public graph
    connectivity only; it never authorizes a route past the static-clearance
    or fleet-separation contracts.
    """

    if not math.isfinite(minimum_received_free_support_m) or minimum_received_free_support_m < 0.0:
        raise ValueError("minimum received-free support must be finite and non-negative")
    if reachability_cache is not None:
        reachability_cache._require_same_belief(belief)
        reachability_cache._bounded_path_search_count += 1

    start_key = _nearest_public_free_key(
        belief,
        start_m,
        maximum_distance_m=belief.resolution_m * 1.5,
    )
    goal_key = _nearest_public_free_key(
        belief,
        goal_m,
        maximum_distance_m=belief.resolution_m * 1.5,
    )
    if start_key is None:
        return _PublicFreePathResult(None, "start_anchor_missing")
    if goal_key is None:
        return _PublicFreePathResult(None, "goal_anchor_missing")
    maximum_steps = max(1, math.floor(maximum_path_length_m / belief.resolution_m))
    queue: deque[tuple[int, int, int]] = deque((start_key,))
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    depth: dict[tuple[int, int, int], int] = {start_key: 0}
    while queue:
        current = queue.popleft()
        if reachability_cache is not None:
            reachability_cache._bounded_path_expanded_node_count += 1
        if current == goal_key:
            break
        if depth[current] >= maximum_steps:
            continue
        for neighbor in neighbors_26(current):
            if neighbor in parent or belief.state(neighbor) != FREE:
                continue
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            queue.append(neighbor)
    if goal_key not in parent:
        if reachability_cache is not None:
            if reachability_cache.source_and_goal_share_component(start_key, goal_key):
                return _PublicFreePathResult(None, "path_exceeds_step_budget")
            return _PublicFreePathResult(None, "public_free_component_disconnected")
        # The bounded result alone cannot distinguish a genuinely disconnected
        # public graph from a route that needs more than this receding-horizon
        # action.  The second traversal is outcome-map only and produces a
        # diagnostic label; it never authorizes the longer route.
        connected: set[tuple[int, int, int]] = {start_key}
        unbounded_queue: deque[tuple[int, int, int]] = deque((start_key,))
        while unbounded_queue:
            current = unbounded_queue.popleft()
            if current == goal_key:
                return _PublicFreePathResult(None, "path_exceeds_step_budget")
            for neighbor in neighbors_26(current):
                if neighbor in connected or belief.state(neighbor) != FREE:
                    continue
                connected.add(neighbor)
                unbounded_queue.append(neighbor)
        return _PublicFreePathResult(None, "public_free_component_disconnected")
    keys: list[tuple[int, int, int]] = []
    cursor: tuple[int, int, int] | None = goal_key
    while cursor is not None:
        keys.append(cursor)
        cursor = parent[cursor]
    keys.reverse()
    # The reset pose has already passed its own physical admission contract.
    # Every *new* route voxel needs spatially distributed public evidence so a
    # frontier candidate cannot be created from one narrow, grazed free ray.
    if minimum_received_free_support_m > 0.0 and any(
        not _has_received_free_interior_support(
            belief,
            key,
            radius_m=minimum_received_free_support_m,
            cache=received_free_support_cache,
        )
        for key in keys[1:]
    ):
        return _PublicFreePathResult(None, "public_free_interior_support_missing")
    route = _compress_public_voxel_path(
        belief,
        keys,
        start_m=start_m,
        goal_m=goal_m,
    )
    route_length = sum(
        math.dist(left, right) for left, right in zip(route, route[1:], strict=False)
    )
    if route_length > maximum_path_length_m + 1.0e-9:
        return _PublicFreePathResult(None, "path_exceeds_step_budget")
    return _PublicFreePathResult(route, "admitted", tuple(keys))


def _public_free_space_path(
    belief: SparseVoxelBelief,
    start_m: tuple[float, float, float],
    goal_m: tuple[float, float, float],
    *,
    maximum_path_length_m: float,
    minimum_received_free_support_m: float = 0.0,
    received_free_support_cache: _ReceivedFreeSupportCache | None = None,
    reachability_cache: _PublicFreeReachabilityCache | None = None,
) -> tuple[tuple[float, float, float], ...] | None:
    """Return a bounded received-free route for candidate construction."""

    return _public_free_space_path_result(
        belief,
        start_m,
        goal_m,
        maximum_path_length_m=maximum_path_length_m,
        minimum_received_free_support_m=minimum_received_free_support_m,
        received_free_support_cache=received_free_support_cache,
        reachability_cache=reachability_cache,
    ).path_m


def _public_route_prefix_from_voxel_chain(
    belief: SparseVoxelBelief,
    *,
    voxel_keys: Sequence[tuple[int, int, int]],
    start_m: tuple[float, float, float],
    progress_point_m: tuple[float, float, float],
    minimum_received_free_support_m: float,
    received_free_support_cache: _ReceivedFreeSupportCache | None = None,
) -> tuple[tuple[float, float, float], ...] | None:
    """Reuse a verified public route prefix when its snapped voxel is on-chain.

    ``_public_route_progress_points`` snaps each proposal to a public FREE
    voxel.  If that voxel belongs to the already searched observation route,
    every interior voxel up to it has already passed the public support check;
    rebuilding the prefix is therefore equivalent to truncating the same
    outcome-backed route.  A point that is not on the chain returns ``None``
    and the caller retains the original conservative BFS fallback.
    """

    if len(voxel_keys) < 2:
        return None
    progress_key = _nearest_public_free_key(
        belief,
        progress_point_m,
        maximum_distance_m=belief.resolution_m * 0.75,
    )
    if progress_key is None:
        return None
    try:
        progress_index = tuple(voxel_keys).index(progress_key)
    except ValueError:
        return None
    if progress_index < 1:
        return None
    prefix_keys = tuple(voxel_keys[: progress_index + 1])
    if any(
        not _has_received_free_interior_support(
            belief,
            key,
            radius_m=minimum_received_free_support_m,
            cache=received_free_support_cache,
        )
        for key in prefix_keys[1:]
    ):
        # This should not occur for a prefix of an admitted route, but fail
        # closed if a future caller supplies a chain with weaker evidence.
        return None
    return _compress_public_voxel_path(
        belief,
        prefix_keys,
        start_m=start_m,
        goal_m=progress_point_m,
    )


def _public_component_progress_path_result(
    belief: SparseVoxelBelief,
    start_m: tuple[float, float, float],
    goal_m: tuple[float, float, float],
    *,
    maximum_path_length_m: float,
    minimum_advance_m: float,
) -> _PublicFreePathResult:
    """Advance inside the current received-free component toward a frontier.

    A sparse six-axis range sensor often exposes a corridor component without
    yet connecting it to a distant frontier cluster.  Treating that target as
    unreachable discards the useful progress already supported by the public
    map.  This helper returns a bounded route to the farthest received-free
    voxel in the source component that reduces Euclidean distance to the goal.
    It is a public-map proposal only; the common static route guard remains the
    authority that admits the exact path against frozen collision geometry.
    """

    if maximum_path_length_m <= 0.0 or minimum_advance_m < 0.0:
        raise ValueError("progress path budget and minimum advance must be valid")
    start_key = _nearest_public_free_key(
        belief,
        start_m,
        maximum_distance_m=belief.resolution_m * 1.5,
    )
    if start_key is None:
        return _PublicFreePathResult(None, "start_anchor_missing")
    goal_key = _nearest_public_free_key(
        belief,
        goal_m,
        maximum_distance_m=belief.resolution_m * 1.5,
    )
    goal_point_m = belief.voxel_center(goal_key) if goal_key is not None else goal_m
    start_point_m = belief.voxel_center(start_key)
    initial_distance_m = math.dist(start_point_m, goal_point_m)
    if initial_distance_m <= minimum_advance_m:
        return _PublicFreePathResult(None, "no_public_component_progress")
    maximum_steps = max(1, math.floor(maximum_path_length_m / belief.resolution_m))
    queue: deque[tuple[int, int, int]] = deque((start_key,))
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
        start_key: None
    }
    depth: dict[tuple[int, int, int], int] = {start_key: 0}
    best_key: tuple[int, int, int] | None = None
    while queue:
        current = queue.popleft()
        if depth[current] >= maximum_steps:
            continue
        for neighbor in neighbors_26(current):
            if neighbor in parent or belief.state(neighbor) != FREE:
                continue
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            queue.append(neighbor)
            path_length_m = depth[neighbor] * belief.resolution_m
            if path_length_m + 1.0e-9 < minimum_advance_m:
                continue
            distance_to_goal_m = math.dist(belief.voxel_center(neighbor), goal_point_m)
            if distance_to_goal_m >= initial_distance_m - 1.0e-9:
                continue
            candidate_score = (
                path_length_m,
                -distance_to_goal_m,
                tuple(-coordinate for coordinate in neighbor),
            )
            if best_key is None:
                best_key = neighbor
                continue
            best_length_m = depth[best_key] * belief.resolution_m
            best_distance_m = math.dist(belief.voxel_center(best_key), goal_point_m)
            best_score = (
                best_length_m,
                -best_distance_m,
                tuple(-coordinate for coordinate in best_key),
            )
            if candidate_score > best_score:
                best_key = neighbor
    if best_key is None:
        return _PublicFreePathResult(None, "no_public_component_progress")
    keys: list[tuple[int, int, int]] = []
    cursor: tuple[int, int, int] | None = best_key
    while cursor is not None:
        keys.append(cursor)
        cursor = parent[cursor]
    keys.reverse()
    progress_point_m = belief.voxel_center(best_key)
    route = _compress_public_voxel_path(
        belief,
        keys,
        start_m=start_m,
        goal_m=progress_point_m,
    )
    route_length_m = sum(
        math.dist(left, right) for left, right in zip(route, route[1:], strict=False)
    )
    if route_length_m > maximum_path_length_m + 1.0e-9:
        return _PublicFreePathResult(None, "path_exceeds_step_budget")
    if route_length_m + 1.0e-9 < minimum_advance_m:
        return _PublicFreePathResult(None, "no_public_component_progress")
    return _PublicFreePathResult(route, "public_component_progress", tuple(keys))


def _known_free_observation_points(
    belief: SparseVoxelBelief,
    *,
    frontier_point_m: tuple[float, float, float],
    outward_normal: tuple[float, float, float],
    minimum_received_free_support_m: float = 0.0,
    received_free_support_cache: _ReceivedFreeSupportCache | None = None,
    maximum_points: int = PUBLIC_OBSERVATION_POINTS_PER_FRONTIER_VIEWPOINT,
) -> tuple[tuple[float, float, float], ...]:
    """Return public-supported interior viewpoints across a sensing standoff range.

    The planner sees only received FREE/OCCUPIED voxels here.  Keeping a
    nominal and deeper supported view prevents a near-boundary, wall-adjacent
    point from being the only representative of an otherwise reachable
    frontier cluster.  The shared physical guard remains the sole clearance
    authority.
    """

    if (
        not isinstance(maximum_points, int)
        or isinstance(maximum_points, bool)
        or maximum_points < 1
    ):
        raise ValueError("maximum_points must be a positive integer")
    normal_length = math.sqrt(sum(value * value for value in outward_normal))
    unit = (
        (0.0, 0.0, 0.0)
        if normal_length <= 1.0e-9
        else tuple(value / normal_length for value in outward_normal)
    )
    desired = tuple(
        frontier_point_m[axis] - unit[axis] * FRONTIER_OBSERVATION_STANDOFF_M for axis in range(3)
    )
    anchor = belief.world_to_voxel(frontier_point_m)
    radius = math.ceil(
        (FRONTIER_OBSERVATION_MAX_STANDOFF_M + belief.resolution_m) / belief.resolution_m
    )
    support_radius_m = (
        minimum_received_free_support_m
        if minimum_received_free_support_m > 0.0
        else PUBLIC_ROUTE_SUPPORT_RADIUS_M
    )
    rows: list[tuple[int, int, int, float, float, tuple[float, float, float]]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                key = (anchor[0] + dx, anchor[1] + dy, anchor[2] + dz)
                if belief.state(key) != FREE:
                    continue
                support = _received_free_support(
                    belief,
                    key,
                    radius_m=support_radius_m,
                    cache=received_free_support_cache,
                )
                if (
                    minimum_received_free_support_m > 0.0
                    and support.free_voxel_count < PUBLIC_ROUTE_SUPPORT_MIN_FREE_VOXELS
                ):
                    continue
                point = belief.voxel_center(key)
                stand_off = math.dist(point, frontier_point_m)
                if not 1.0 <= stand_off <= FRONTIER_OBSERVATION_MAX_STANDOFF_M + belief.resolution_m:
                    continue
                # Prefer public points surrounded by independently observed
                # free space before choosing the closest nominal standoff.
                # No static geometry enters this ordering.
                rows.append(
                    (
                        -support.balanced_axis_count,
                        -support.signed_axis_count,
                        -support.free_voxel_count,
                        math.dist(point, desired),
                        -stand_off,
                        point,
                    )
                )
    if not rows:
        return ()
    # Retain one robust pose around each sensor standoff before falling back to
    # the normal support-first order. This diversifies geometry using the
    # public map only; it does not require a route to be longer.
    selected: list[tuple[float, float, float]] = []
    for standoff_target_m in FRONTIER_OBSERVATION_STANDOFF_VARIANTS_M:
        for row in sorted(
            rows,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                abs(-item[4] - standoff_target_m),
                item[3],
                item[5],
            ),
        ):
            if row[-1] not in selected:
                selected.append(row[-1])
                break
    for row in sorted(rows):
        if row[-1] not in selected:
            selected.append(row[-1])
        if len(selected) >= maximum_points:
            break
    return tuple(selected[:maximum_points])


def _public_route_progress_points(
    belief: SparseVoxelBelief,
    path_m: Sequence[tuple[float, float, float]],
    *,
    minimum_received_free_support_m: float,
    received_free_support_cache: _ReceivedFreeSupportCache | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Return robust, public-map-only intermediate stops on a known-free route.

    A full frontier observation pose can be physically inadmissible even when
    an earlier part of its received-free route is safe. Retaining several
    supported prefixes lets the next sparse sensing pass extend the map from a
    legal progress point instead of turning the agent into an unexplained
    hold. The shared static guard must still admit the recomputed route.
    """

    path = tuple(tuple(point) for point in path_m)
    if len(path) < 2:
        return ()
    segment_lengths = tuple(
        math.dist(left, right) for left, right in zip(path, path[1:], strict=False)
    )
    total_length_m = sum(segment_lengths)
    if total_length_m < PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M + belief.resolution_m:
        return ()

    rows: list[tuple[float, tuple[float, float, float]]] = []
    for fraction in (0.45, 0.70, 0.85, 0.95):
        target_distance_m = total_length_m * fraction
        if (
            target_distance_m < PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M
            or total_length_m - target_distance_m < belief.resolution_m
        ):
            continue
        remaining = target_distance_m
        candidate: tuple[float, float, float] | None = None
        for left, right, length in zip(path, path[1:], segment_lengths, strict=False):
            if remaining > length + 1.0e-12:
                remaining -= length
                continue
            ratio = 0.0 if length <= 1.0e-12 else min(1.0, remaining / length)
            interpolated = tuple(
                left[axis] + ratio * (right[axis] - left[axis]) for axis in range(3)
            )
            key = _nearest_public_free_key(
                belief,
                interpolated,
                maximum_distance_m=belief.resolution_m * 0.75,
            )
            if key is not None:
                support = _received_free_support(
                    belief,
                    key,
                    radius_m=minimum_received_free_support_m,
                    cache=received_free_support_cache,
                )
                if (
                    support.free_voxel_count >= PUBLIC_ROUTE_SUPPORT_MIN_FREE_VOXELS
                    and support.balanced_axis_count
                    >= PUBLIC_ROUTE_PROGRESS_MIN_BALANCED_AXES
                ):
                    point = belief.voxel_center(key)
                    if math.dist(point, path[0]) >= PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M:
                        candidate = point
            break
        if candidate is not None:
            rows.append((target_distance_m, candidate))
    rows.sort()
    unique: list[tuple[float, float, float]] = []
    for _, point in rows:
        if point not in unique:
            unique.append(point)
        if len(unique) >= PUBLIC_ROUTE_PROGRESS_VARIANTS_PER_PATH:
            break
    return tuple(unique)


def _polyline_prefix_distance_m(
    path_m: Sequence[tuple[float, float, float]],
    point_m: tuple[float, float, float],
) -> float:
    """Return the closest arc-length position of a snapped public route point."""

    path = tuple(tuple(point) for point in path_m)
    if len(path) < 2:
        raise ValueError("public route needs at least one segment")
    accumulated_m = 0.0
    best: tuple[float, float] | None = None
    for left, right in zip(path, path[1:], strict=False):
        vector = tuple(right[axis] - left[axis] for axis in range(3))
        length_squared = sum(value * value for value in vector)
        length_m = math.sqrt(length_squared)
        if length_m <= 1.0e-12:
            continue
        projection = sum(
            (point_m[axis] - left[axis]) * vector[axis] for axis in range(3)
        ) / length_squared
        projection = min(1.0, max(0.0, projection))
        nearest = tuple(left[axis] + projection * vector[axis] for axis in range(3))
        candidate = (math.dist(point_m, nearest), accumulated_m + projection * length_m)
        if best is None or candidate < best:
            best = candidate
        accumulated_m += length_m
    if best is None:
        raise ValueError("public route has no non-degenerate segment")
    return best[1]


def _public_route_progress_gain(
    *,
    cluster_gain: float,
    progress_length_m: float,
    route_length_m: float,
) -> float:
    """Weigh an access prefix as real exploration when it advances far enough.

    Route prefixes are still public-map proposals; this only affects candidate
    ranking, never outcomes or the shared static guard. Short prefixes keep
    the discounted gain so the selector does not confuse a sensor hold with
    exploration.
    """

    if route_length_m <= 0.0 or progress_length_m < 0.0:
        raise ValueError("route progress gain requires positive route evidence")
    if (
        progress_length_m >= PUBLIC_ROUTE_PROGRESS_FULL_GAIN_MIN_M
        and progress_length_m <= route_length_m + 1.0e-9
    ):
        return cluster_gain
    return cluster_gain * max(0.25, progress_length_m / route_length_m)


def _unexplored_potential_gain(
    belief: SparseVoxelBelief,
    point_m: tuple[float, float, float],
    *,
    radius_m: float,
) -> float:
    """Estimate the unexplored volume around a target pose from public data.

    Sparse range sensing only sees the frontiers it has already hit, so every
    ranked gain is concentrated near the current pose: the agent sees nearby
    clusters, ranks them high, and never advances to distant unexplored space
    (the visibility loop that keeps every route short).  This term estimates
    the *unobserved* potential around a target by sparsely sampling the public
    belief: UNKNOWN voxels (absent from the belief dictionary) carry the
    exploration value of not-yet-seen space.  It is a public-map quantity, the
    same for every method, and is added to region-access / route-progress
    candidates only (short observation viewpoints keep the observed-cluster
    gain so a sensor hold is never rewarded as exploration).
    """

    radius_m = max(radius_m, belief.resolution_m)
    samples = PUBLIC_POTENTIAL_GAIN_SAMPLES
    center = belief.world_to_voxel(point_m)
    radius_cells = max(1, math.ceil(radius_m / belief.resolution_m))
    unknown = 0
    examined = 0
    random_source = random.Random(0)
    for _ in range(samples):
        cell = tuple(
            center[axis]
            + random_source.randint(-radius_cells, radius_cells)
            for axis in range(3)
        )
        if belief.state(cell) == FREE or belief.state(cell) == OCCUPIED:
            examined += 1
        else:
            unknown += 1
            examined += 1
    if examined == 0:
        return 0.0
    volume_m3 = 4.0 / 3.0 * math.pi * radius_m**3
    return PUBLIC_POTENTIAL_GAIN_WEIGHT * volume_m3 * (unknown / examined)


def _public_frontiers_from_belief(
    positions: tuple[tuple[float, float, float], ...],
    belief: SparseVoxelBelief,
    *,
    decision_index: int,
    maximum_step_m: float,
    maximum_frontiers_per_agent: int = PUBLIC_FRONTIER_VIEWPOINTS_PER_AGENT,
    observation_cooldown: _PublicObservationCooldown | None = None,
    task_reservations: Sequence[PublicTaskReservation] = (),
    reachability_cache: _PublicFreeReachabilityCache | None = None,
) -> tuple[PublicFrontier, ...]:
    """Build a bounded receding-horizon view set from public 3D frontiers.

    FUEL-style exploration clusters the free/unknown boundary, creates an
    interior observation pose, and evaluates a bounded public-free-space route
    to that pose. Every target and path here comes only from shared sparse
    outcomes. The common runtime guard remains the sole static-geometry safety
    authority; it rechecks this public route against frozen collision geometry.
    """

    if maximum_frontiers_per_agent < 1:
        raise ValueError("maximum_frontiers_per_agent must be positive")
    clusters = extract_frontier_clusters(
        belief,
        config=FrontierExtractionConfig(
            min_cluster_voxels=1,
            max_viewpoints_per_cluster=min(
                maximum_frontiers_per_agent,
                PUBLIC_FRONTIER_VIEWPOINTS_PER_CLUSTER,
            ),
            height_band_m=1.0,
        ),
    )
    if not clusters:
        raise ValueError("public sparse belief exposes no free/unknown frontier cluster")
    clusters = tuple(
        sorted(clusters, key=lambda cluster: (-cluster.expected_gain_m3, cluster.frontier_id))
    )[:PUBLIC_FRONTIER_CLUSTER_SEARCH_BUDGET]

    maximum_gain_m3 = max(cluster.expected_gain_m3 for cluster in clusters)
    candidates: dict[tuple[int, int, int], _PublicFrontierViewpoint] = {}
    # The belief does not change while one candidate pool is constructed.
    # Reusing these local-neighborhood summaries prevents the same received-free
    # support evidence from being rescanned for every route and viewpoint.
    received_free_support_cache: _ReceivedFreeSupportCache = {}
    public_reachability_cache = reachability_cache or _PublicFreeReachabilityCache(belief)
    public_reachability_cache._require_same_belief(belief)

    def path_search_budget_exhausted(*, region_access: bool = False) -> bool:
        # Observation standoff and route-prefix searches stop before the full
        # decision budget so the reserved region-access search still runs.
        limit = (
            PUBLIC_FRONTIER_PATH_SEARCH_BUDGET_PER_DECISION
            if region_access
            else PUBLIC_FRONTIER_PATH_SEARCH_BUDGET_PER_DECISION
            - PUBLIC_REGION_ACCESS_PATH_SEARCH_RESERVE_PER_DECISION
        )
        return public_reachability_cache._bounded_path_search_count >= limit

    def consider_viewpoint(
        *,
        position_m: tuple[float, float, float],
        information_gain: float,
        traversal_risk: float,
        route_lengths_m: tuple[float | None, ...],
        route_paths_m: tuple[tuple[tuple[float, float, float], ...] | None, ...],
        source_agent_index: int,
        viewpoint_kind: str,
        frontier_cluster_id: str,
        task_anchor_m: tuple[float, float, float],
        task_normal_unit: tuple[float, float, float],
        observation_standoff_m: float | None,
    ) -> None:
        if len(route_lengths_m) != len(positions) or len(route_paths_m) != len(positions):
            raise ValueError("public frontier route evidence must align with every agent")
        for route_length, route_path in zip(route_lengths_m, route_paths_m, strict=True):
            if (route_length is None) != (route_path is None):
                raise ValueError("public frontier route path and length disagree")
        reachable = tuple(
            (index, length)
            for index, length in enumerate(route_lengths_m)
            if length is not None
        )
        if not reachable:
            return
        source_index, _ = min(reachable, key=lambda row: (row[1], row[0]))
        endpoint_key = belief.world_to_voxel(position_m)
        if observation_cooldown is not None and observation_cooldown.blocks(
            endpoint_key, decision_index=decision_index
        ):
            return
        row = _PublicFrontierViewpoint(
            information_gain=information_gain,
            traversal_risk=traversal_risk,
            position_m=position_m,
            source_agent_index=(
                source_agent_index
                if viewpoint_kind in {"route_progress", "region_access"}
                else source_index
            ),
            route_lengths_m=route_lengths_m,
            route_paths_m=route_paths_m,
            viewpoint_kind=viewpoint_kind,
            frontier_cluster_id=frontier_cluster_id,
            task_anchor_m=task_anchor_m,
            task_normal_unit=task_normal_unit,
            observation_standoff_m=observation_standoff_m,
        )
        previous = candidates.get(endpoint_key)
        if previous is None or (
            row.information_gain,
            -row.traversal_risk,
            sum(length is not None for length in row.route_lengths_m),
            -min(length for length in row.route_lengths_m if length is not None),
            row.viewpoint_kind in {"observation", "region_access"},
        ) > (
            previous.information_gain,
            -previous.traversal_risk,
            sum(length is not None for length in previous.route_lengths_m),
            -min(length for length in previous.route_lengths_m if length is not None),
            previous.viewpoint_kind in {"observation", "region_access"},
        ):
            candidates[endpoint_key] = row

    for cluster in clusters:
        if path_search_budget_exhausted(region_access=True):
            break
        gain = cluster.expected_gain_m3 / maximum_gain_m3
        risk = 0.12 if abs(cluster.outward_normal[2]) > 0.5 else 0.08
        # Route-level access alternatives commit to a farther received-free
        # point in the current component that moves toward this frontier
        # region. They are generated before observation standoff routes so a
        # dense near-frontier viewpoint set cannot consume the search budget
        # and erase every corridor access proposal. The shared static guard
        # remains the physical admission authority for these proposals.
        region_access_generated = 0
        for agent_index in range(len(positions)):
            if (
                path_search_budget_exhausted(region_access=True)
                or region_access_generated >= PUBLIC_REGION_ACCESS_MAX_PER_CLUSTER
            ):
                break
            public_reachability_cache._bounded_path_search_count += 1
            public_reachability_cache._region_access_attempt_count += 1
            region_path_result = _public_component_progress_path_result(
                belief,
                positions[agent_index],
                cluster.centroid_m,
                maximum_path_length_m=maximum_step_m,
                minimum_advance_m=PUBLIC_REGION_ACCESS_MIN_ADVANCE_M,
            )
            region_path = region_path_result.path_m
            if region_path is None:
                public_reachability_cache.record_region_access_failure(
                    str(region_path_result.status)
                )
                continue
            if region_path_result.voxel_keys is not None and any(
                not _has_received_free_interior_support(
                    belief,
                    key,
                    radius_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                    cache=received_free_support_cache,
                )
                for key in region_path_result.voxel_keys[1:]
            ):
                public_reachability_cache.record_region_access_failure(
                    "public_free_interior_support_missing"
                )
                continue
            region_length = sum(
                math.dist(left, right)
                for left, right in zip(region_path, region_path[1:], strict=False)
            )
            if region_length + 1.0e-9 < PUBLIC_REGION_ACCESS_MIN_ADVANCE_M:
                public_reachability_cache.record_region_access_failure(
                    "no_public_component_progress"
                )
                continue
            consider_viewpoint(
                position_m=region_path[-1],
                information_gain=_public_route_progress_gain(
                    cluster_gain=gain,
                    progress_length_m=region_length,
                    route_length_m=region_length,
                ),
                traversal_risk=risk,
                route_lengths_m=tuple(
                    region_length if index == agent_index else None
                    for index in range(len(positions))
                ),
                route_paths_m=tuple(
                    region_path if index == agent_index else None
                    for index in range(len(positions))
                ),
                source_agent_index=agent_index,
                viewpoint_kind="region_access",
                frontier_cluster_id=cluster.frontier_id,
                task_anchor_m=cluster.centroid_m,
                task_normal_unit=cluster.outward_normal,
                observation_standoff_m=None,
            )
            region_access_generated += 1
            public_reachability_cache.record_region_access_generation()
        if path_search_budget_exhausted():
            # The observation standoff budget is exhausted, but the region-access
            # reserve may still contain enough searches for later clusters.
            continue
        for frontier_viewpoint in cluster.viewpoint_candidates_m:
            if path_search_budget_exhausted():
                break
            for observation_point in _known_free_observation_points(
                belief,
                frontier_point_m=frontier_viewpoint,
                outward_normal=cluster.outward_normal,
                minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                received_free_support_cache=received_free_support_cache,
                maximum_points=PUBLIC_FRONTIER_OBSERVATION_POINTS_PER_VIEWPOINT,
            ):
                if path_search_budget_exhausted():
                    break
                path_results = tuple(
                    _public_free_space_path_result(
                        belief,
                        position,
                        observation_point,
                        maximum_path_length_m=maximum_step_m,
                        minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                        received_free_support_cache=received_free_support_cache,
                        reachability_cache=public_reachability_cache,
                    )
                    for position in positions
                )
                path_rows = tuple(result.path_m for result in path_results)
                route_lengths = tuple(
                    None
                    if path is None or len(path) < 2
                    else sum(
                        math.dist(left, right) for left, right in zip(path, path[1:], strict=False)
                    )
                    for path in path_rows
                )
                consider_viewpoint(
                    position_m=observation_point,
                    information_gain=gain,
                    traversal_risk=risk,
                    route_lengths_m=route_lengths,
                    route_paths_m=path_rows,
                    source_agent_index=0,
                    viewpoint_kind="observation",
                    frontier_cluster_id=cluster.frontier_id,
                    task_anchor_m=cluster.centroid_m,
                    task_normal_unit=cluster.outward_normal,
                    observation_standoff_m=math.dist(observation_point, frontier_viewpoint),
                )
                for agent_index, path_result in enumerate(path_results):
                    path = path_result.path_m
                    if path is None:
                        if path_search_budget_exhausted():
                            continue
                        component_progress = _public_component_progress_path_result(
                            belief,
                            positions[agent_index],
                            observation_point,
                            maximum_path_length_m=maximum_step_m,
                            minimum_advance_m=PUBLIC_ROUTE_PROGRESS_MIN_ADVANCE_M,
                        )
                        progress_path = component_progress.path_m
                        if progress_path is None:
                            continue
                        progress_length = sum(
                            math.dist(left, right)
                            for left, right in zip(
                                progress_path,
                                progress_path[1:],
                                strict=False,
                            )
                        )
                        consider_viewpoint(
                            position_m=progress_path[-1],
                            information_gain=_public_route_progress_gain(
                                cluster_gain=gain,
                                progress_length_m=progress_length,
                                route_length_m=max(
                                    math.dist(
                                        positions[agent_index],
                                        observation_point,
                                    ),
                                    belief.resolution_m,
                                ),
                            ),
                            traversal_risk=risk,
                            route_lengths_m=tuple(
                                progress_length if index == agent_index else None
                                for index in range(len(positions))
                            ),
                            route_paths_m=tuple(
                                progress_path if index == agent_index else None
                                for index in range(len(positions))
                            ),
                            source_agent_index=agent_index,
                            viewpoint_kind="route_progress",
                            frontier_cluster_id=cluster.frontier_id,
                            task_anchor_m=cluster.centroid_m,
                            task_normal_unit=cluster.outward_normal,
                            observation_standoff_m=None,
                        )
                        continue
                    path_length = route_lengths[agent_index]
                    if path_length is None:
                        continue
                    for progress_point in _public_route_progress_points(
                        belief,
                        path,
                        minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                        received_free_support_cache=received_free_support_cache,
                    ):
                        progress_length = _polyline_prefix_distance_m(path, progress_point)
                        progress_path = _public_route_prefix_from_voxel_chain(
                            belief,
                            voxel_keys=path_result.voxel_keys,
                            start_m=positions[agent_index],
                            progress_point_m=progress_point,
                            minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                            received_free_support_cache=received_free_support_cache,
                        )
                        if progress_path is not None:
                            public_reachability_cache.record_route_prefix_reuse()
                        else:
                            if path_search_budget_exhausted():
                                continue
                            # A snapped prefix point can fall just outside the
                            # exact BFS chain (for example after a diagonal
                            # compression). Preserve the old conservative
                            # route authority for that rare case.
                            public_reachability_cache.record_route_prefix_fallback()
                            progress_path = _public_free_space_path(
                                belief,
                                positions[agent_index],
                                progress_point,
                                maximum_path_length_m=maximum_step_m,
                                minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                                received_free_support_cache=received_free_support_cache,
                                reachability_cache=public_reachability_cache,
                            )
                        if progress_path is None:
                            continue
                        progress_length = sum(
                            math.dist(left, right)
                            for left, right in zip(progress_path, progress_path[1:], strict=False)
                        )
                        consider_viewpoint(
                            position_m=progress_point,
                            information_gain=_public_route_progress_gain(
                                cluster_gain=gain,
                                progress_length_m=progress_length,
                                route_length_m=path_length,
                            ),
                            traversal_risk=risk,
                            route_lengths_m=tuple(
                                progress_length if index == agent_index else None
                                for index in range(len(positions))
                            ),
                            route_paths_m=tuple(
                                progress_path if index == agent_index else None
                                for index in range(len(positions))
                            ),
                            source_agent_index=agent_index,
                            viewpoint_kind="route_progress",
                            frontier_cluster_id=cluster.frontier_id,
                            task_anchor_m=cluster.centroid_m,
                            task_normal_unit=cluster.outward_normal,
                            observation_standoff_m=None,
                        )
    if len(candidates) < len(positions):
        # A bootstrap-perturbed sparse belief can transiently expose fewer
        # interior observation viewpoints than agents.  Hard-failing the whole
        # episode on this transient would discard every earlier receipt and
        # make the runtime unusable on narrow scenes.  Retry once with a
        # doubled path-search budget before accepting a sparse-but-legal pool:
        # the shared route guard and joint guard still admit every row.
        if path_search_budget_exhausted():
            retry_candidates = dict(candidates)
            budget_limit = (
                public_reachability_cache._bounded_path_search_count
                + PUBLIC_FRONTIER_PATH_SEARCH_BUDGET_PER_DECISION
            )
            for cluster in clusters:
                if public_reachability_cache._bounded_path_search_count >= budget_limit:
                    break
                if cluster.frontier_id in {row.frontier_cluster_id for row in retry_candidates.values()}:
                    continue
                for frontier_viewpoint in cluster.viewpoint_candidates_m:
                    if public_reachability_cache._bounded_path_search_count >= budget_limit:
                        break
                    for observation_point in _known_free_observation_points(
                        belief,
                        frontier_point_m=frontier_viewpoint,
                        outward_normal=cluster.outward_normal,
                        minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                        received_free_support_cache=received_free_support_cache,
                        maximum_points=PUBLIC_FRONTIER_OBSERVATION_POINTS_PER_VIEWPOINT,
                    ):
                        if public_reachability_cache._bounded_path_search_count >= budget_limit:
                            break
                        path_results = tuple(
                            _public_free_space_path_result(
                                belief,
                                position,
                                observation_point,
                                maximum_path_length_m=maximum_step_m,
                                minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                                received_free_support_cache=received_free_support_cache,
                                reachability_cache=public_reachability_cache,
                            )
                            for position in positions
                        )
                        path_rows = tuple(result.path_m for result in path_results)
                        route_lengths = tuple(
                            None
                            if path is None or len(path) < 2
                            else sum(
                                math.dist(left, right)
                                for left, right in zip(path, path[1:], strict=False)
                            )
                            for path in path_rows
                        )
                        if not any(length is not None for length in route_lengths):
                            continue
                        consider_viewpoint(
                            position_m=observation_point,
                            information_gain=(
                                cluster.expected_gain_m3 / maximum_gain_m3
                                if maximum_gain_m3 > 0.0
                                else 0.0
                            ),
                            traversal_risk=0.08,
                            route_lengths_m=route_lengths,
                            route_paths_m=path_rows,
                            source_agent_index=0,
                            viewpoint_kind="observation",
                            frontier_cluster_id=cluster.frontier_id,
                            task_anchor_m=cluster.centroid_m,
                            task_normal_unit=cluster.outward_normal,
                            observation_standoff_m=math.dist(
                                observation_point, frontier_viewpoint
                            ),
                        )
            if len(candidates) < len(positions):
                raise ValueError(
                    "public sparse belief exposes too few reachable interior observation viewpoints: "
                    f"observed={len(candidates)}, required={len(positions)}"
                )
    observation_remaining = {
        key: row
        for key, row in candidates.items()
        if row.viewpoint_kind == "observation"
    }
    route_progress_remaining = {
        key: row
        for key, row in candidates.items()
        if row.viewpoint_kind in {"route_progress", "region_access"}
    }
    if len(observation_remaining) + len(route_progress_remaining) != len(candidates):
        raise RuntimeError("public frontier generator emitted an unknown viewpoint kind")
    selected: list[tuple[tuple[int, int, int], _PublicFrontierViewpoint]] = []
    maximum_count = maximum_frontiers_per_agent * len(positions)
    # Keep a small, source-balanced reserve of route prefixes alongside the
    # complete observation poses. Both are public exploration views; the common
    # matcher now gives them the same exploration priority so a continuous
    # corridor route is not discarded merely because short views also exist.
    route_progress_reservation = min(
        PUBLIC_ROUTE_PROGRESS_RETAINED_PER_SOURCE,
        max(1, maximum_frontiers_per_agent // 2),
    )
    observation_capacity = max(
        0,
        maximum_count - route_progress_reservation * len(positions),
    )

    def matches_reservation(
        reservation: PublicTaskReservation,
        row: _PublicFrontierViewpoint,
    ) -> tuple[bool, float, float]:
        task_anchor = row.task_anchor_m or row.position_m
        anchor_distance_m = math.dist(reservation.task_anchor_m, task_anchor)
        normal_alignment = 1.0
        if reservation.task_normal_unit is not None and row.task_normal_unit is not None:
            normal_alignment = min(
                1.0,
                max(
                    -1.0,
                    sum(
                        reservation.task_normal_unit[axis] * row.task_normal_unit[axis]
                        for axis in range(3)
                    ),
                ),
            )
        return (
            anchor_distance_m <= PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M + 1.0e-9
            and normal_alignment >= PUBLIC_TASK_RESERVATION_MIN_NORMAL_ALIGNMENT,
            anchor_distance_m,
            normal_alignment,
        )

    # Reserve one current observation representative for each task that still
    # exists in the latest public map. This prevents generic diversity pruning
    # from dropping an already assigned task before the common matcher can
    # inspect it. The route itself is still regenerated from the current pose.
    for reservation in sorted(task_reservations, key=lambda item: item.agent_id):
        matching_agent_indices = [
            index
            for index in range(len(positions))
            if reservation.agent_id == f"uav{index}"
        ]
        if len(matching_agent_indices) != 1 or len(selected) >= observation_capacity:
            continue
        agent_index = matching_agent_indices[0]
        matches = [
            (key, row, *matches_reservation(reservation, row))
            for key, row in observation_remaining.items()
            if row.route_paths_m[agent_index] is not None
        ]
        matches = [row for row in matches if row[2] is True]
        if not matches:
            continue
        chosen_key, chosen_row, _matched, _distance_m, _normal_alignment = min(
            matches,
            key=lambda row: (
                row[3],
                -row[1].information_gain,
                row[1].traversal_risk,
                row[0],
            ),
        )
        selected.append((chosen_key, chosen_row))
        observation_remaining.pop(chosen_key)

    def take_next(
        remaining: dict[tuple[int, int, int], _PublicFrontierViewpoint],
    ) -> tuple[tuple[int, int, int], _PublicFrontierViewpoint]:
        unseen_cluster_ids = {
            row.frontier_cluster_id
            for _key, row in remaining.items()
            if row.frontier_cluster_id not in {
                prior.frontier_cluster_id for _prior_key, prior in selected
            }
        }
        eligible = (
            {
                key: row
                for key, row in remaining.items()
                if row.frontier_cluster_id in unseen_cluster_ids
            }
            if unseen_cluster_ids
            else remaining
        )
        if not selected:
            return max(
                eligible.items(),
                key=lambda item: (item[1].information_gain, -item[1].traversal_risk, item[0]),
            )
        return max(
            eligible.items(),
            key=lambda item: (
                min(math.dist(item[1].position_m, prior[1].position_m) for prior in selected),
                item[1].information_gain,
                -item[1].traversal_risk,
                tuple(-value for value in item[0]),
            ),
        )

    # Fill the primary viewpoint budget with complete observations first for
    # stable spatial coverage. Route prefixes remain in the same public
    # exploration tier once the shared matcher constructs assignments.
    while observation_remaining and len(selected) < observation_capacity:
        chosen = take_next(observation_remaining)
        selected.append(chosen)
        observation_remaining.pop(chosen[0])

    # Retain only a small, deterministic fallback set. These rows remain in
    # the common action authority so a later static rejection cannot turn a
    # valid public route prefix into an unexplained hold.
    for source_agent_index in range(len(positions)):
        source_rows = tuple(
            item
            for item in route_progress_remaining.items()
            if item[1].source_agent_index == source_agent_index
        )
        if not source_rows or len(selected) >= maximum_count:
            continue
        chosen_rows = _retain_route_progress_viewpoints(
            source_rows,
            source_position_m=positions[source_agent_index],
            source_agent_index=source_agent_index,
            maximum_count=min(route_progress_reservation, maximum_count - len(selected)),
            resolution_m=belief.resolution_m,
        )
        selected.extend(chosen_rows)
        for key, _row in chosen_rows:
            route_progress_remaining.pop(key)
    # If a scene has fewer than the reserved number of prefixes, spend the
    # unused slots on additional complete observation viewpoints before adding
    # any non-primary prefix.
    while observation_remaining and len(selected) < maximum_count:
        chosen = take_next(observation_remaining)
        selected.append(chosen)
        observation_remaining.pop(chosen[0])
    while route_progress_remaining and len(selected) < maximum_count:
        chosen = take_next(route_progress_remaining)
        selected.append(chosen)
        route_progress_remaining.pop(chosen[0])
    rows = tuple(
        PublicFrontier(
            frontier_id=f"d{decision_index:02d}-view-r{rank:02d}",
            position_m=row.position_m,
            information_gain=row.information_gain,
            traversal_risk=row.traversal_risk,
            source_agent_id=f"uav{row.source_agent_index}",
            viewpoint_kind=row.viewpoint_kind,
            access_paths_m=tuple(
                (f"uav{agent_index}", path_m)
                for agent_index, path_m in enumerate(row.route_paths_m)
                if path_m is not None
            ),
            frontier_cluster_id=row.frontier_cluster_id,
            task_anchor_m=row.task_anchor_m,
            task_normal_unit=row.task_normal_unit,
        )
        for rank, (_key, row) in enumerate(selected)
    )
    return tuple(rows)


def _aggregate_communication(
    round_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not round_rows:
        raise ValueError("online episode has no communication rounds")
    telemetry_rates = {float(row["communication"]["telemetry_update_hz"]) for row in round_rows}
    if len(telemetry_rates) != 1:
        raise ValueError("online episode communication telemetry rate changed between decisions")
    total = sum(int(row["communication"]["relay_telemetry_sample_count"]) for row in round_rows)
    connected = sum(
        int(row["communication"]["relay_connected_telemetry_sample_count"]) for row in round_rows
    )
    delivery_counts = {
        status: sum(
            int(row["message_delivery"]["outcome_counts_after_close"][status]) for row in round_rows
        )
        for status in ("DELIVERED", "DROPPED", "EXPIRED")
    }
    expected = sum(
        int(row["message_delivery"]["expected_recipient_outcomes"]) for row in round_rows
    )
    resolved = sum(
        int(row["message_delivery"]["resolved_recipient_outcomes"]) for row in round_rows
    )
    if total < 1 or expected < 1:
        raise ValueError("online episode has no actual communication denominator")
    communication = {
        "telemetry_update_hz": telemetry_rates.pop(),
        "relay_telemetry_sample_count": total,
        "relay_connected_telemetry_sample_count": connected,
        "relay_connected_telemetry_sample_fraction": connected / total,
        "longest_sampled_disconnected_duration_s": max(
            float(row["communication"]["longest_sampled_disconnected_duration_s"])
            for row in round_rows
        ),
        "final_graph": round_rows[-1]["communication"]["final_graph"],
        "delivered_count": delivery_counts["DELIVERED"],
        "dropped_count": delivery_counts["DROPPED"],
        "expired_count": delivery_counts["EXPIRED"],
        "outcome_count": resolved,
    }
    delivery = {
        "outcome_counts_after_close": delivery_counts,
        "expected_recipient_outcomes": expected,
        "resolved_recipient_outcomes": resolved,
        "maximum_delivery_age_s": max(
            float(row["message_delivery"]["maximum_delivery_age_s"]) for row in round_rows
        ),
    }
    return communication, delivery


class EvaluatorFreeOverlap(NamedTuple):
    public_voxel_count: int
    public_volume_m3: float
    consistent_volume_m3: float
    inconsistent_volume_m3: float
    touched_evaluator_voxel_count: int
    touched_free_evaluator_voxel_count: int
    overlap_piece_count: int
    grid_phase_offset_fraction: tuple[float, float, float]

    @property
    def conservation_error_m3(self) -> float:
        return abs(self.public_volume_m3 - self.consistent_volume_m3 - self.inconsistent_volume_m3)


def _evaluator_consistent_public_free(
    *,
    component: Any,
    grid_origin: Any,
    resolution_m: float,
    team_belief: SparseVoxelBelief,
) -> EvaluatorFreeOverlap:
    """Intersect public free-voxel cubes with the frozen evaluator volume.

    The method never sees this result.  It is evaluator-only scoring that
    distinguishes genuinely explored free volume from public free volume
    outside the frozen component.  Public and evaluator grids have independent
    origins.  A nearest-centre lookup is therefore invalid: in HM3D scene
    00626 the grids are exactly half a voxel out of phase and Python's banker
    rounding merged adjacent public voxels.  Exact axis-aligned overlap keeps
    the numerator invariant to grid phase and conserves all predicted volume.
    """

    shape = tuple(int(value) for value in component.shape)
    evaluator_resolution = float(resolution_m)
    public_resolution = float(team_belief.resolution_m)
    evaluator_boundary_origin = tuple(
        float(grid_origin[axis]) - 0.5 * evaluator_resolution for axis in range(3)
    )
    consistent_volume_m3 = 0.0
    inconsistent_volume_m3 = 0.0
    touched: set[tuple[int, int, int]] = set()
    touched_free: set[tuple[int, int, int]] = set()
    overlap_piece_count = 0
    for key in team_belief.free_keys():
        center = team_belief.voxel_center(key)
        axis_overlaps: list[list[tuple[int, float]]] = []
        for axis in range(3):
            public_lower = center[axis] - 0.5 * public_resolution
            public_upper = center[axis] + 0.5 * public_resolution
            first_index = math.floor(
                (public_lower - evaluator_boundary_origin[axis]) / evaluator_resolution
            )
            last_index = math.floor(
                (public_upper - evaluator_boundary_origin[axis]) / evaluator_resolution
            )
            overlaps: list[tuple[int, float]] = []
            for index in range(first_index, last_index + 1):
                evaluator_lower = evaluator_boundary_origin[axis] + index * evaluator_resolution
                width = max(
                    0.0,
                    min(public_upper, evaluator_lower + evaluator_resolution)
                    - max(public_lower, evaluator_lower),
                )
                if width > 1.0e-12:
                    overlaps.append((index, width))
            if not overlaps:
                raise RuntimeError("public/evaluator voxel overlap vanished on one axis")
            axis_overlaps.append(overlaps)

        for x_index, x_width in axis_overlaps[0]:
            for y_index, y_width in axis_overlaps[1]:
                for z_index, z_width in axis_overlaps[2]:
                    index = (x_index, y_index, z_index)
                    overlap_volume_m3 = x_width * y_width * z_width
                    overlap_piece_count += 1
                    touched.add(index)
                    if all(0 <= index[axis] < shape[axis] for axis in range(3)) and bool(
                        component[index]
                    ):
                        consistent_volume_m3 += overlap_volume_m3
                        touched_free.add(index)
                    else:
                        inconsistent_volume_m3 += overlap_volume_m3

    public_voxel_count = team_belief.observed_free_count
    public_volume_m3 = public_voxel_count * public_resolution**3
    conservation_error_m3 = abs(public_volume_m3 - consistent_volume_m3 - inconsistent_volume_m3)
    if conservation_error_m3 > max(1.0e-10, public_volume_m3 * 1.0e-9):
        raise RuntimeError(
            "public/evaluator free-volume overlap does not conserve volume: "
            f"public={public_volume_m3}, consistent={consistent_volume_m3}, "
            f"inconsistent={inconsistent_volume_m3}"
        )
    phase = tuple(
        (
            (team_belief.origin_m[axis] + 0.5 * public_resolution - float(grid_origin[axis]))
            / evaluator_resolution
        )
        % 1.0
        for axis in range(3)
    )
    return EvaluatorFreeOverlap(
        public_voxel_count=public_voxel_count,
        public_volume_m3=public_volume_m3,
        consistent_volume_m3=consistent_volume_m3,
        inconsistent_volume_m3=inconsistent_volume_m3,
        touched_evaluator_voxel_count=len(touched),
        touched_free_evaluator_voxel_count=len(touched_free),
        overlap_piece_count=overlap_piece_count,
        grid_phase_offset_fraction=phase,
    )


def _metric_sample(
    *,
    timestamp_s: float,
    component: Any,
    grid_origin: Any,
    resolution_m: float,
    denominator_volume_m3: float,
    team_belief: SparseVoxelBelief,
) -> ExplorationMetricSample:
    overlap = _evaluator_consistent_public_free(
        component=component,
        grid_origin=grid_origin,
        resolution_m=resolution_m,
        team_belief=team_belief,
    )
    return ExplorationMetricSample(
        timestamp_s=timestamp_s,
        explored_free_volume_m3=overlap.consistent_volume_m3,
        true_free_volume_m3=denominator_volume_m3,
        predicted_free_volume_m3=overlap.public_volume_m3,
        hallucinated_free_volume_m3=overlap.inconsistent_volume_m3,
    )


PERIODIC_SUPERVISION_SCHEMA_VERSION = "hm3d-p07-periodic-supervision-v2"


@dataclass
class _PeriodicSupervisionLedger:
    interval_s: float
    agent_ids: tuple[str, ...]
    start_positions_m: tuple[tuple[float, float, float], ...]
    next_timestamp_s: float
    samples: list[dict[str, object]] = field(default_factory=list)
    realised_path_by_agent: dict[str, float] = field(default_factory=dict)
    planned_path_by_agent: dict[str, float] = field(default_factory=dict)
    min_height_by_agent: dict[str, float] = field(default_factory=dict)
    max_height_by_agent: dict[str, float] = field(default_factory=dict)
    last_positions_by_agent: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    trace_samples_by_agent: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    delivered_message_count: int = 0
    expected_message_count: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("periodic supervision interval must be finite and positive")
        if len(self.agent_ids) != len(self.start_positions_m):
            raise ValueError("periodic supervision agent ids and start positions must match")
        for agent_id, position in zip(self.agent_ids, self.start_positions_m, strict=True):
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                raise ValueError("periodic supervision start position must be finite 3D")
            self.realised_path_by_agent[agent_id] = 0.0
            self.planned_path_by_agent[agent_id] = 0.0
            self.min_height_by_agent[agent_id] = float(position[2])
            self.max_height_by_agent[agent_id] = float(position[2])
            self.last_positions_by_agent[agent_id] = position

    def accumulate_delivery(self, delivery: dict[str, object]) -> None:
        counts = delivery.get("outcome_counts_after_close")
        if isinstance(counts, dict):
            self.delivered_message_count += int(counts.get("DELIVERED", 0))
        self.expected_message_count += int(delivery.get("expected_recipient_outcomes", 0))

    def accumulate_calibration(self, execution_calibration: dict[str, object]) -> None:
        agents = execution_calibration.get("agents")
        if not isinstance(agents, list):
            raise RuntimeError("periodic supervision requires execution calibration agents")
        for agent in agents:
            if not isinstance(agent, dict):
                raise RuntimeError("periodic supervision agent row is malformed")
            agent_id = str(agent["agent_id"])
            if agent_id not in self.realised_path_by_agent:
                raise RuntimeError(f"periodic supervision saw unexpected agent {agent_id}")
            self.realised_path_by_agent[agent_id] += float(agent["realized_transit_path_length_m"])
            geometry = agent.get("route_geometry")
            if not isinstance(geometry, dict):
                raise RuntimeError("periodic supervision agent omits route geometry")
            self.planned_path_by_agent[agent_id] += float(geometry["command_path_length_m"])
            tracking = agent.get("controller_tracking_samples")
            if not isinstance(tracking, list):
                raise RuntimeError("periodic supervision agent omits controller tracking samples")
            for sample in tracking:
                if not isinstance(sample, dict):
                    continue
                position = sample.get("post_step_position_m")
                if isinstance(position, (list, tuple)) and len(position) >= 3:
                    height = float(position[2])
                    if math.isfinite(height):
                        self.min_height_by_agent[agent_id] = min(
                            self.min_height_by_agent[agent_id], height
                        )
                        self.max_height_by_agent[agent_id] = max(
                            self.max_height_by_agent[agent_id], height
                        )

    def accumulate_trace(
        self,
        trace: object,
        *,
        start_s: float,
    ) -> None:
        """Merge audit-only PhysX telemetry using episode-local timestamps."""

        if not isinstance(trace, dict):
            return
        trace_samples = trace.get("samples")
        if not isinstance(trace_samples, list):
            return
        for row in trace_samples:
            if not isinstance(row, dict):
                continue
            local_timestamp_s = row.get("physics_timestamp_s")
            agent_rows = row.get("agents")
            if (
                not isinstance(local_timestamp_s, (int, float))
                or isinstance(local_timestamp_s, bool)
                or not isinstance(agent_rows, list)
            ):
                continue
            episode_timestamp_s = float(local_timestamp_s) + float(start_s)
            minimum_separation = row.get("minimum_inter_agent_distance_m")
            for agent_row in agent_rows:
                if not isinstance(agent_row, dict):
                    continue
                agent_id = agent_row.get("agent_id")
                position = agent_row.get("position_m")
                if (
                    not isinstance(agent_id, str)
                    or agent_id not in self.realised_path_by_agent
                    or not isinstance(position, (list, tuple))
                    or len(position) < 3
                ):
                    continue
                speed = agent_row.get("linear_speed_mps")
                self.trace_samples_by_agent.setdefault(agent_id, []).append(
                    {
                        "episode_timestamp_s": episode_timestamp_s,
                        "local_timestamp_s": float(local_timestamp_s),
                        "position_m": [float(value) for value in position[:3]],
                        "linear_speed_mps": (
                            float(speed)
                            if isinstance(speed, (int, float)) and not isinstance(speed, bool)
                            else None
                        ),
                        "minimum_inter_agent_distance_m": (
                            float(minimum_separation)
                            if isinstance(minimum_separation, (int, float))
                            and not isinstance(minimum_separation, bool)
                            else None
                        ),
                        "reservation_waiting": bool(agent_row.get("reservation_waiting", False)),
                        "transit_completed": bool(agent_row.get("transit_completed", False)),
                        "failed": bool(agent_row.get("failed", False)),
                    }
                )

    def _trace_state_at(
        self,
        agent_id: str,
        timestamp_s: float,
    ) -> dict[str, object] | None:
        series = self.trace_samples_by_agent.get(agent_id)
        if not series:
            return None
        before: dict[str, object] | None = None
        after: dict[str, object] | None = None
        for sample in series:
            sample_time = float(sample["episode_timestamp_s"])
            if sample_time <= timestamp_s + 1.0e-9:
                before = sample
            elif after is None:
                after = sample
            if after is not None and sample_time > timestamp_s:
                break
        if before is None:
            before = series[0]
        if before is None:
            return None
        if after is None or after is before:
            return {
                **before,
                "source": "physics_visualization_trace",
                "sample_timestamp_error_s": abs(float(before["episode_timestamp_s"]) - timestamp_s),
            }
        before_time = float(before["episode_timestamp_s"])
        after_time = float(after["episode_timestamp_s"])
        fraction = (timestamp_s - before_time) / max(after_time - before_time, 1.0e-9)
        before_position = tuple(before["position_m"])
        after_position = tuple(after["position_m"])
        interpolated_position = tuple(
            before_value + fraction * (after_value - before_value)
            for before_value, after_value in zip(before_position, after_position, strict=True)
        )
        before_speed = before.get("linear_speed_mps")
        after_speed = after.get("linear_speed_mps")
        interpolated_speed = None
        if isinstance(before_speed, (int, float)) and isinstance(after_speed, (int, float)):
            interpolated_speed = float(before_speed) + fraction * (float(after_speed) - float(before_speed))
        return {
            **before,
            "position_m": list(interpolated_position),
            "linear_speed_mps": interpolated_speed,
            "source": "physics_visualization_trace_interpolated",
            "sample_timestamp_error_s": min(
                abs(before_time - timestamp_s),
                abs(after_time - timestamp_s),
            ),
        }

    def emit_until(
        self,
        *,
        elapsed_s: float,
        positions_m: Sequence[tuple[float, float, float]],
        linear_speeds_mps: Sequence[float],
        samples: Sequence[ExplorationMetricSample],
        horizon_s: float,
        total_energy_j: float,
        collision_count: int,
        separation_violation_count: int,
        out_of_bounds_count: int,
        static_clearance_violation_count: int,
        executed_fragment_count: int,
        failed_fragment_count: int,
        decision_count: int,
    ) -> None:
        if len(positions_m) != len(self.agent_ids) or len(linear_speeds_mps) != len(self.agent_ids):
            raise RuntimeError("periodic supervision agent telemetry shape mismatch")
        while self.next_timestamp_s <= elapsed_s + 1.0e-9:
            timestamp_s = self.next_timestamp_s
            if timestamp_s > horizon_s + 1.0e-9:
                self.next_timestamp_s = horizon_s + self.interval_s
                break
            coverage, explored_volume = _interpolated_metric_at_s(samples, timestamp_s)
            auc = _periodic_supervision_auc_at_s(samples, timestamp_s, horizon_s)
            snapshot_positions = positions_m
            snapshot_speeds = linear_speeds_mps
            position_source = "decision_boundary"
            max_trace_error_s = 0.0
            trace_states: list[dict[str, object]] = []
            trace_available = True
            for agent_id in self.agent_ids:
                state = self._trace_state_at(agent_id, timestamp_s)
                if state is None:
                    trace_available = False
                    break
                trace_states.append(state)
                max_trace_error_s = max(
                    max_trace_error_s,
                    float(state.get("sample_timestamp_error_s", 0.0)),
                )
            if trace_available and trace_states:
                snapshot_positions = tuple(
                    tuple(float(value) for value in state["position_m"])
                    for state in trace_states
                )
                snapshot_speeds = tuple(
                    float(state["linear_speed_mps"])
                    if isinstance(state.get("linear_speed_mps"), (int, float))
                    else float(linear_speeds_mps[index])
                    for index, state in enumerate(trace_states)
                )
                position_source = str(trace_states[0]["source"])
            moving_agent_count = 0
            agents: list[dict[str, object]] = []
            for index, agent_id in enumerate(self.agent_ids):
                position = snapshot_positions[index]
                previous = self.last_positions_by_agent[agent_id]
                displacement_m = math.dist(position, previous)
                moving = displacement_m > 0.1
                moving_agent_count += int(moving)
                start = self.start_positions_m[index]
                agents.append(
                    {
                        "agent_id": agent_id,
                        "position_m": list(position),
                        "linear_speed_mps": float(snapshot_speeds[index]),
                        "realised_path_length_m": self.realised_path_by_agent[agent_id],
                        "planned_path_length_m": self.planned_path_by_agent[agent_id],
                        "displacement_since_last_snapshot_m": displacement_m,
                        "displacement_since_start_m": math.dist(position, start),
                        "vertical_displacement_since_start_m": float(position[2]) - float(start[2]),
                        "min_height_m": self.min_height_by_agent[agent_id],
                        "max_height_m": self.max_height_by_agent[agent_id],
                        "moving_since_last_snapshot": moving,
                        "position_source": position_source,
                    }
                )
                self.last_positions_by_agent[agent_id] = tuple(position)
            minimum_separation_m = min(
                (
                    math.dist(snapshot_positions[left], snapshot_positions[right])
                    for left in range(len(snapshot_positions))
                    for right in range(left + 1, len(snapshot_positions))
                ),
                default=math.inf,
            )
            expected = max(self.expected_message_count, 0)
            self.samples.append(
                {
                    "timestamp_s": timestamp_s,
                    "position_source": position_source,
                    "max_position_sample_timestamp_error_s": max_trace_error_s,
                    "coverage_fraction": coverage,
                    "explored_free_volume_m3": explored_volume,
                    "explored_free_flight_volume_auc_time": auc,
                    "planned_fleet_path_length_m": sum(self.planned_path_by_agent.values()),
                    "realised_fleet_path_length_m": sum(self.realised_path_by_agent.values()),
                    "moving_agent_count": moving_agent_count,
                    "minimum_inter_agent_distance_m": minimum_separation_m,
                    "total_energy_used_j": total_energy_j,
                    "collision_count": collision_count,
                    "separation_violation_count": separation_violation_count,
                    "out_of_bounds_count": out_of_bounds_count,
                    "static_clearance_contract_violation_count": static_clearance_violation_count,
                    "executed_fragment_count": executed_fragment_count,
                    "failed_fragment_count": failed_fragment_count,
                    "communication_delivery_fraction": (
                        self.delivered_message_count / expected if expected else 0.0
                    ),
                    "decision_count_so_far": decision_count,
                    "agents": agents,
                }
            )
            self.next_timestamp_s += self.interval_s


def _interpolated_metric_at_s(
    samples: Sequence[ExplorationMetricSample],
    timestamp_s: float,
) -> tuple[float, float]:
    if not samples:
        raise RuntimeError("periodic supervision requires metric samples")
    if timestamp_s <= samples[0].timestamp_s:
        return samples[0].coverage_fraction, samples[0].explored_free_volume_m3
    previous = samples[0]
    for current in samples[1:]:
        if current.timestamp_s >= timestamp_s:
            if current.timestamp_s <= previous.timestamp_s:
                raise RuntimeError("periodic supervision metric timestamps must increase")
            fraction = (timestamp_s - previous.timestamp_s) / (
                current.timestamp_s - previous.timestamp_s
            )
            coverage = previous.coverage_fraction + fraction * (
                current.coverage_fraction - previous.coverage_fraction
            )
            volume = previous.explored_free_volume_m3 + fraction * (
                current.explored_free_volume_m3 - previous.explored_free_volume_m3
            )
            return coverage, volume
        previous = current
    return previous.coverage_fraction, previous.explored_free_volume_m3


def _periodic_supervision_auc_at_s(
    samples: Sequence[ExplorationMetricSample],
    timestamp_s: float,
    horizon_s: float,
) -> float:
    if not samples:
        raise RuntimeError("periodic supervision requires metric samples")
    if samples[0].timestamp_s != 0.0:
        raise RuntimeError("periodic supervision metric curve must start at zero")
    area = 0.0
    previous = samples[0]
    for current in samples[1:]:
        if current.timestamp_s <= previous.timestamp_s:
            raise RuntimeError("periodic supervision metric timestamps must increase")
        if previous.timestamp_s >= timestamp_s:
            break
        right = min(current.timestamp_s, timestamp_s)
        if right > previous.timestamp_s:
            fraction = (right - previous.timestamp_s) / (current.timestamp_s - previous.timestamp_s)
            interpolated = previous.coverage_fraction + fraction * (
                current.coverage_fraction - previous.coverage_fraction
            )
            area += 0.5 * (right - previous.timestamp_s) * (
                previous.coverage_fraction + interpolated
            )
        previous = current
    if previous.timestamp_s < timestamp_s:
        area += (timestamp_s - previous.timestamp_s) * previous.coverage_fraction
    return min(1.0, area / horizon_s)


def _select(
    strategy: str,
    state: PublicSearchState,
    pool: tuple[CandidateFragmentManifest, ...],
    *,
    random_key: int,
    single_rl_checkpoint: Path | None,
    marl_ipp_checkpoint: Path | None,
    marl_ipp_source_root: Path,
    split_manifest_sha256: str,
    planned_qd_selector: PlannedQDSelector | None = None,
    realised_qd_selector: OutcomeGroundedQDSelector | None = None,
    public_exploration_need: PublicExplorationNeed | None = None,
    qd_calibration_mode: str | None = None,
) -> tuple[CandidateFragmentManifest, dict[str, Any]]:
    if strategy == "qd_calibration":
        if qd_calibration_mode not in HM3D_QD_CALIBRATION_INTENT_MODES:
            raise ValueError("QD calibration requires one declared public intent mode")
        legal = tuple(candidate for candidate in pool if candidate.feasible)
        if not legal:
            raise ValueError("QD calibration requires a feasible public candidate")
        axis, direction = {
            "vertical_low": (0, -1.0),
            "vertical_high": (0, 1.0),
            "dispersion_low": (1, -1.0),
            "dispersion_high": (1, 1.0),
            "complementarity_low": (2, -1.0),
            "complementarity_high": (2, 1.0),
        }[qd_calibration_mode]
        selected = min(
            legal,
            key=lambda candidate: (
                -direction * candidate.planned_descriptor[axis],
                candidate.manifest_hash,
            ),
        )
        return selected, {
            "selector": "train_only_qd_replay_calibration",
            "calibration_only": True,
            "public_intent_mode": qd_calibration_mode,
            "public_intent_axis": axis,
            "selected_candidate_id": selected.candidate_id,
            "selected_manifest_hash": selected.manifest_hash,
        }
    if strategy == "no_qd":
        legal = tuple(candidate for candidate in pool if candidate.feasible)
        if not legal:
            raise ValueError("no-QD selector requires a feasible public candidate")
        ranked = sorted(
            legal,
            key=lambda candidate: (
                -candidate.quality_hint / max(1.0e-9, candidate.cost_hint),
                candidate.manifest_hash,
            ),
        )
        selected = ranked[0]
        return selected, {
            "selector": "no_qd_public_quality_cost",
            "selected_candidate_id": selected.candidate_id,
            "selected_manifest_hash": selected.manifest_hash,
        }
    if strategy == "planned_qd":
        if planned_qd_selector is None:
            raise ValueError("planned-QD selector is not initialized")
        selected, selection = planned_qd_selector.select(pool)
        return selected, selection.to_dict()
    if strategy == "realised_qd":
        if realised_qd_selector is None:
            raise ValueError("realised-QD selector is not initialized")
        if public_exploration_need is None:
            raise ValueError("realised-QD selection requires current public exploration need")
        selected, selection = realised_qd_selector.select(
            pool,
            public_exploration_need=public_exploration_need,
        )
        return selected, selection.to_dict()
    if strategy == "single_rl":
        if single_rl_checkpoint is None:
            raise ValueError("single_rl requires a train-provenanced checkpoint")
        selected, selection = select_single_rl(
            state,
            pool,
            checkpoint_path=single_rl_checkpoint,
            expected_split_manifest_sha256=split_manifest_sha256,
        )
    elif strategy == "marl_ipp_port":
        if marl_ipp_checkpoint is None:
            raise ValueError("marl_ipp_port requires a train-provenanced checkpoint")
        selected, selection = select_marl_ipp_port(
            state,
            pool,
            checkpoint_path=marl_ipp_checkpoint,
            source_root=marl_ipp_source_root,
            expected_split_manifest_sha256=split_manifest_sha256,
        )
    elif strategy == "gvp_mrep_port":
        selected, selection = select_gvp_mrep_port(state, pool)
    else:
        selected, selection = select_public_baseline(strategy, pool, random_key=random_key)
    return selected, selection.to_dict()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument(
        "--record-purpose",
        choices=tuple(sorted(P07_RECORD_PURPOSES)),
        default="engineering_smoke",
        help=(
            "Explicit downstream-use class. Engineering smoke records remain valid "
            "control evidence but cannot enter RL, QD, or fragment-reuse datasets."
        ),
    )
    parser.add_argument("--collision-usd", required=True, type=Path)
    parser.add_argument("--start-reset-json", required=True, type=Path)
    parser.add_argument("--flight-space-audit", required=True, type=Path)
    parser.add_argument("--p03-artifact", required=True, type=Path)
    parser.add_argument("--p04-artifact", required=True, type=Path)
    parser.add_argument("--p05-artifact", required=True, type=Path)
    parser.add_argument("--p06-artifact", required=True, type=Path)
    parser.add_argument("--transit-time-model-json", required=True, type=Path)
    parser.add_argument(
        "--communication-contract-json",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_p07_communication_contract.json",
    )
    parser.add_argument("--cf2x-usd", type=Path, default=cf2x.DRONE_USD)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=(
            "random",
            "frontier_3d",
            "auction",
            "gvp_mrep_port",
            "single_rl",
            "marl_ipp_port",
            "no_qd",
            "planned_qd",
            "realised_qd",
            "qd_calibration",
        ),
        required=True,
    )
    parser.add_argument("--single-rl-checkpoint", type=Path, default=None)
    parser.add_argument("--marl-ipp-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--marl-ipp-source-root",
        type=Path,
        default=Path(
            os.environ.get(
                "AEROCITY_MARL_IPP_ROOT",
                str(ROOT / "external" / "marl_ipp"),
            )
        ),
    )
    parser.add_argument(
        "--qd-history",
        type=Path,
        action="append",
        default=None,
        help=(
            "One completed train-partition P07 outcome record. Repeat this option until the "
            "same-fleet history has at least twelve executed outcomes across two scenes."
        ),
    )
    parser.add_argument(
        "--qd-calibration-mode",
        choices=HM3D_QD_CALIBRATION_INTENT_MODES,
        default=None,
        help=(
            "Train-only public replay mode used to calibrate realised-QD descriptor "
            "richness and repeatability. It is never a ranked baseline or a P07 result."
        ),
    )
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument(
        "--disable-route-extreme",
        action="store_true",
        help=(
            "Engineering-only differential switch that restores the pre-route-extreme "
            "shared candidate pool. It is not used for formal training, QD, or holdout runs."
        ),
    )
    parser.add_argument(
        "--max-decision-count",
        type=int,
        default=128,
        help=(
            "Safety cap for zero-duration or otherwise non-progressing action loops. "
            "The physical action budget, not this value, ends a normal episode."
        ),
    )
    parser.add_argument(
        "--rolling-decision-window-s",
        type=float,
        default=None,
        help=(
            "Optional short physical horizon for each decision. The executor remains "
            "event-driven, but the candidate authority plans only a bounded rolling "
            "window so short legal routes replan frequently instead of waiting on a "
            "single long team leg. If omitted, the remaining episode budget is used."
        ),
    )
    parser.add_argument("--action-budget-s", type=float, default=40.0)
    parser.add_argument("--outcome-time-tolerance-s", type=float, default=0.25)
    parser.add_argument("--arrival-tolerance-m", type=float, default=0.10)
    parser.add_argument(
        "--controller-id",
        choices=(
            cf2x.CF2X_DEFAULT_CONTROLLER_ID,
            cf2x.BITCRAZE_LEE_CONTROLLER_ID,
            cf2x.BITCRAZE_MELLINGER_CONTROLLER_ID,
        ),
        default=cf2x.CF2X_DEFAULT_CONTROLLER_ID,
        help=(
            "Shared trajectory tracker. Bitcraze Lee requires a timing calibration with "
            "the same execution profile before it can produce training or formal outcomes."
        ),
    )
    parser.add_argument("--physics-dt-s", type=float, default=1.0 / 120.0)
    parser.add_argument(
        "--calibration-timeout-probe-s",
        type=float,
        default=None,
        help=(
            "Stop each decision executor at this shorter real deadline. This train-only "
            "engineering probe emits no RL transition and cannot enter QD history."
        ),
    )
    parser.add_argument(
        "--visualization-trace-hz",
        type=float,
        default=None,
        help=(
            "Optional post-step PhysX pose telemetry frequency for an engineering-only "
            "visual replay. It is rejected for training, QD, and formal records and never "
            "enters the controller, candidate generator, belief, safety, or reward path."
        ),
    )
    parser.add_argument(
        "--supervision-interval-s",
        type=float,
        default=None,
        help=(
            "Optional audit-only periodic supervision interval. Every interval records "
            "coverage, AUC, path length, energy, safety and per-agent telemetry without "
            "entering the controller, candidate, belief, safety, reward, QD or replay path."
        ),
    )
    parser.add_argument("--random-key", type=int, default=20260802)
    parser.add_argument(
        "--p0-start-candidate-ids",
        nargs=FORMAL_FLEET_SIZE,
        default=None,
        metavar="START_CANDIDATE_ID",
        help=(
            "Exactly four IDs from the immutable start-reset manifest. Valid only with "
            "--p0-start-eligibility-audit; it prevents the normal greedy setup selector "
            "from changing the audited P0 start cluster."
        ),
    )
    parser.add_argument(
        "--p0-start-eligibility-audit",
        action="store_true",
        help=(
            "Run shared PhysX bootstrap and first-pool guarded-route eligibility only. "
            "It writes an engineering audit, never an RL transition or performance result."
        ),
    )
    parser.add_argument(
        "--p0-start-connectivity-audit",
        action="store_true",
        help=(
            "Enumerate four-agent relay-connected subsets of the immutable reset manifest "
            "with the same PhysX range/LOS query used by the executor. It is environment "
            "setup evidence only and does not create agents, observations or transitions."
        ),
    )
    parser.add_argument(
        "--p0-start-eligibility-evidence",
        type=Path,
        default=None,
        help=(
            "Immutable P0 eligibility-audit output that authorizes an explicit four-agent "
            "reset for one full P0 qualification episode. It is rejected for an audit run "
            "and does not authorize training or formal evaluation."
        ),
    )
    # A persistent collector owns the single Isaac application and passes an
    # explicit worker argv. AppLauncher inspects process-global sys.argv while
    # registering its flags, so only let the standalone entry point add them.
    if argv is None:
        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


@functools.lru_cache(maxsize=8)
def _static_scene_artifacts(collision_usd: str) -> tuple[Any, dict[str, Any], dict[str, Any], Any]:
    """Cache immutable collision and evaluator geometry across batched episodes."""

    collision_path = Path(collision_usd)
    mesh = cf2x._load_collision_triangle_mesh(collision_path)
    arrays, rebuilt = build_enclosed_esdf(
        mesh,
        resolution_m=0.25,
        vehicle_clearance_m=0.3,
    )
    _, clearance, _ = cf2x._build_conservative_clearance_field(collision_path)
    return mesh, arrays, rebuilt, clearance


def main(args: argparse.Namespace, simulation_app: Any) -> int:
    import numpy as np
    import omni.physx
    import omni.usd
    import torch
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
    from isaaclab_contrib.assets import Multirotor
    from pxr import Gf, UsdGeom

    runtime_wall_started = time.perf_counter()
    fleet_size = FORMAL_FLEET_SIZE
    if (
        args.candidate_limit < fleet_size
        or args.max_decision_count < 2
        or args.action_budget_s <= 0.0
        or args.physics_dt_s <= 0.0
        or args.arrival_tolerance_m <= 0.0
        or (
            args.rolling_decision_window_s is not None
            and args.rolling_decision_window_s <= 0.0
        )
        or args.outcome_time_tolerance_s < 0.0
    ):
        raise ValueError("invalid P07 online exploration budget or controller argument")
    # The multirotor actuator samples its physical response constants from
    # configured ranges. Bind every simulator-side random source before asset
    # creation so a paired method comparison starts from the same dynamics.
    random.seed(args.random_key)
    np.random.seed(args.random_key % (2**32))
    torch.manual_seed(args.random_key)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_key)
    qd_strategy = args.strategy in {"planned_qd", "realised_qd"}
    qd_calibration = args.strategy == "qd_calibration"
    timeout_probe = args.calibration_timeout_probe_s is not None
    visualization_trace_enabled = args.visualization_trace_hz is not None
    if visualization_trace_enabled and (
        not math.isfinite(args.visualization_trace_hz) or args.visualization_trace_hz <= 0.0
    ):
        raise ValueError("visualization trace frequency must be finite and positive")
    if visualization_trace_enabled and args.record_purpose != "engineering_smoke":
        raise ValueError("visualization trace is restricted to engineering_smoke records")
    supervision_enabled = args.supervision_interval_s is not None
    if supervision_enabled and (
        not math.isfinite(args.supervision_interval_s) or args.supervision_interval_s <= 0.0
    ):
        raise ValueError("periodic supervision interval must be finite and positive")
    if timeout_probe and (
        args.split != "train"
        or args.strategy not in {"random", "frontier_3d", "auction", "gvp_mrep_port"}
        or not math.isfinite(args.calibration_timeout_probe_s)
        or args.calibration_timeout_probe_s <= 0.0
    ):
        raise ValueError(
            "calibration timeout probes require a positive deadline, train split and a "
            "non-learning baseline"
        )
    if qd_calibration and args.split != "train":
        raise ValueError("QD replay calibration may only run on the train partition")
    if qd_calibration and args.record_purpose != "qd_calibration":
        raise ValueError("QD replay calibration requires record purpose qd_calibration")
    if args.record_purpose == "qd_calibration" and not qd_calibration:
        raise ValueError("record purpose qd_calibration requires strategy qd_calibration")
    if args.record_purpose == "train_outcome" and (
        args.split != "train" or qd_calibration or timeout_probe
    ):
        raise ValueError(
            "train outcomes require the train split and cannot be calibration or timeout probes"
        )
    if qd_calibration and args.qd_calibration_mode is None:
        raise ValueError("QD replay calibration requires --qd-calibration-mode")
    if not qd_calibration and args.qd_calibration_mode is not None:
        raise ValueError("--qd-calibration-mode is only valid with qd_calibration")
    if (qd_strategy or qd_calibration) and args.candidate_limit < 6:
        raise ValueError("QD mechanism runs require at least six public candidates per decision")
    if qd_strategy and args.qd_history is None:
        raise ValueError("planned/realised-QD requires a completed train outcome history")
    if args.p0_start_eligibility_audit:
        if args.p0_start_candidate_ids is None:
            raise ValueError("P0 start eligibility audit requires four explicit start candidate IDs")
        if args.strategy != "frontier_3d":
            raise ValueError("P0 start eligibility audit uses only the transparent frontier_3d pool")
        if qd_strategy or qd_calibration or timeout_probe:
            raise ValueError("P0 start eligibility audit cannot enable QD or timeout probes")
    elif args.p0_start_candidate_ids is not None:
        if args.p0_start_eligibility_evidence is None:
            raise ValueError(
                "explicit P0 start candidate IDs require an eligibility audit or its evidence"
            )
    if args.p0_start_eligibility_evidence is not None:
        if args.p0_start_eligibility_audit or args.p0_start_candidate_ids is None:
            raise ValueError(
                "P0 eligibility evidence requires a full qualification with explicit start IDs"
            )
        # The eligibility evidence certifies the frozen reset poses and the
        # shared candidate pool, not the selector.  Restricting it to
        # frontier_3d made QD strategies (whose intent-richness audit needs
        # the same well-spread all-active reset) unable to use the verified
        # starts, so they fell back to auto-selected poses that produced too
        # few feasible candidates.  Allow the same evidence for QD calibration
        # and engineering qualification; training collection still uses
        # auto-selected starts for pose diversity.
        if timeout_probe:
            raise ValueError("P0 eligibility evidence cannot enable timeout probes")
        if args.record_purpose not in {"engineering_smoke", "qd_calibration"}:
            raise ValueError(
                "P0 eligibility evidence authorizes engineering qualification or QD "
                "calibration only, not training"
            )
    if args.p0_start_connectivity_audit:
        if (
            args.p0_start_eligibility_audit
            or args.p0_start_candidate_ids is not None
            or args.p0_start_eligibility_evidence is not None
        ):
            raise ValueError("P0 connectivity audit cannot combine with an explicit eligibility audit")
        if qd_strategy or qd_calibration or timeout_probe:
            raise ValueError("P0 connectivity audit cannot enable QD or timeout probes")
    paths = _paths(args)
    p03_row = _p03_row(_read_object(paths["p03"]), args.scene_id)
    public_contract_sha256, geometry_denominator_sha256, profile = _contract_hashes(
        p04=_read_object(paths["p04"]),
        p06=_read_object(paths["p06"]),
        p03_row=p03_row,
        scene_id=args.scene_id,
    )
    split_manifest_sha256 = _frozen_split_manifest_hash(
        _read_object(paths["p05"]),
        scene_id=args.scene_id,
        split=args.split,
    )
    flight = _read_object(paths["flight"])
    if flight.get("scene_id") != args.scene_id:
        raise ValueError("flight-space audit scene mismatch")
    flight_space = flight.get("flight_space")
    if not isinstance(flight_space, dict):
        raise ValueError("flight-space audit lacks flight-space payload")
    if flight.get("flight_space_manifest_hash") != p03_row.get("flight_space_manifest_hash"):
        raise ValueError("P03 and runtime flight-space manifests differ")
    if flight.get("collision_usd_sha256") != _sha256(paths["collision"]):
        raise ValueError("collision USD differs from P03 flight-space evidence")
    source = _read_object(paths["start_resets"])
    if source.get("scene_id") != args.scene_id:
        raise ValueError("P07 start-reset evidence scene mismatch")
    p0_live_departure_required = bool(
        args.p0_start_eligibility_audit or args.p0_start_eligibility_evidence is not None
    )
    start_candidates = _initial_position_candidates(
        source,
        p03_row=p03_row,
        collision_usd_sha256=_sha256(paths["collision"]),
    )
    p0_departure_envelope: dict[str, object] | None = None
    if args.p0_start_eligibility_audit or args.p0_start_eligibility_evidence is not None:
        p0_departure_envelope = _p0_departure_envelope_audit(source)
    bounds_min = _point(flight_space.get("free_bounds_min_m"), "free_bounds_min_m")
    bounds_max = _point(flight_space.get("free_bounds_max_m"), "free_bounds_max_m")
    communication_contract = HM3DCommunicationContract.from_path(paths["communication"])
    execution_profile = _current_transit_execution_profile(
        cf2x_usd_path=paths["cf2x"],
        fleet_size=fleet_size,
        physics_dt_s=args.physics_dt_s,
        arrival_tolerance_m=args.arrival_tolerance_m,
        outcome_time_tolerance_s=args.outcome_time_tolerance_s,
        controller_id=args.controller_id,
    )
    execution_profile_sha256 = canonical_sha256(execution_profile)
    transit_timing, observation_dwell_s = _load_transit_timing_contract(
        paths["timing"],
        expected_execution_profile=execution_profile,
    )
    p0_eligibility_contract = {
        "schema_version": "hm3d-p07-start-eligibility-contract-v1",
        "strategy": args.strategy,
        "candidate_limit": args.candidate_limit,
        "action_budget_s": args.action_budget_s,
        "physics_dt_s": args.physics_dt_s,
        "arrival_tolerance_m": args.arrival_tolerance_m,
        "outcome_time_tolerance_s": args.outcome_time_tolerance_s,
        "random_key": args.random_key,
    }

    mesh, arrays, rebuilt, clearance = _static_scene_artifacts(str(paths["collision"]))
    if rebuilt["flight_space_manifest_hash"] != flight_space.get("flight_space_manifest_hash"):
        raise ValueError("rebuilt evaluator ESDF differs from frozen P03 flight space")
    free_mask = np.asarray(arrays["free_mask"], dtype=bool)
    full_free_volume_m3 = float(free_mask.sum()) * 0.25**3
    if not math.isclose(
        full_free_volume_m3,
        float(p03_row["free_flight_volume_m3"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("evaluator denominator does not match the P03 formal artifact")
    grid_origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
    clearance_oracle = cf2x._EvaluatorStaticClearance(clearance, mesh)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    collision_root = UsdGeom.Xform.Define(stage, "/World/HM3DCollision")
    collision_root.GetPrim().GetReferences().AddReference(str(paths["collision"]))
    UsdGeom.Xform.Define(stage, "/World/P07Agents")
    for _ in range(12):
        simulation_app.update()
    sim = SimulationContext(
        SimulationCfg(
            dt=args.physics_dt_s,
            device=args.device,
            enable_scene_query_support=True,
            physx=PhysxCfg(enable_enhanced_determinism=True),
        )
    )
    sim.reset()
    scene_query = MemoizedRaycastClosestQuery(omni.physx.get_physx_scene_query_interface())
    start_candidate_ids = tuple(str(row["candidate_id"]) for row in source["candidates"])
    eligibility_evidence_sha256: str | None = None
    if args.p0_start_connectivity_audit:
        connected_start_id_combinations = _relay_connected_start_id_combinations(
            start_candidates,
            start_candidate_ids,
            lambda positions: cf2x._initial_relay_graph(scene_query, positions),
        )
        connectivity_payload = {
            "schema_version": "hm3d-p07-start-connectivity-audit-v1",
            "status": "P07_START_CONNECTIVITY_AUDIT_COMPLETE",
            "synthetic": False,
            "formal_result": False,
            "trainable": False,
            "claim_limit": (
                "P0 environment-setup evidence only. It enumerates relay-connected quartets "
                "from an immutable, clearance-filtered start manifest with the executor's "
                "range/LOS query. It does not create a public map, run a selector, execute "
                "an exploration route, score coverage or produce an RL transition."
            ),
            "scene_id": args.scene_id,
            "selection_partition": args.split,
            "start_reset_manifest_sha256": _sha256(paths["start_resets"]),
            "collision_usd_sha256": _sha256(paths["collision"]),
            "candidate_count": len(start_candidates),
            "fleet_size": fleet_size,
            "candidate_combination_count": math.comb(len(start_candidates), fleet_size),
            "relay_connected_combination_count": len(connected_start_id_combinations),
            "relay_connected_candidate_id_combinations": [
                list(candidate_ids) for candidate_ids in connected_start_id_combinations
            ],
            "random_key": args.random_key,
        }
        connectivity_payload["audit_record_sha256"] = canonical_sha256(connectivity_payload)
        _write_new(paths["output"], connectivity_payload)
        print(
            json.dumps(
                {
                    "status": connectivity_payload["status"],
                    "output": str(paths["output"]),
                    "relay_connected_combination_count": len(connected_start_id_combinations),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.p0_start_candidate_ids is None:
        starts = _select_connected_initial_positions(
            start_candidates,
            lambda positions: cf2x._initial_relay_graph(scene_query, positions),
        )
        candidate_id_by_position = dict(zip(start_candidates, start_candidate_ids, strict=True))
        selected_start_candidate_ids = tuple(candidate_id_by_position[position] for position in starts)
        start_selection_mode = "relay_connected_greedy_from_immutable_candidates"
    else:
        selected_start_candidate_ids = tuple(args.p0_start_candidate_ids)
        if args.p0_start_eligibility_evidence is not None:
            evidence_path = args.p0_start_eligibility_evidence.expanduser().resolve()
            evidence = _read_object(evidence_path)
            _validated_p0_start_eligibility_evidence(
                evidence,
                scene_id=args.scene_id,
                start_reset_manifest_sha256=_sha256(paths["start_resets"]),
                controller_id=args.controller_id,
                transit_time_model_sha256=_sha256(paths["timing"]),
                p0_eligibility_contract=p0_eligibility_contract,
                requested_candidate_ids=selected_start_candidate_ids,
            )
            eligibility_evidence_sha256 = _sha256(evidence_path)
        starts = _select_explicit_initial_positions(
            start_candidates,
            start_candidate_ids,
            selected_start_candidate_ids,
            lambda positions: cf2x._initial_relay_graph(scene_query, positions),
        )
        start_selection_mode = (
            "p0_eligibility_evidence_authorized_from_immutable_candidates"
            if eligibility_evidence_sha256 is not None
            else "explicit_p0_eligibility_audit_from_immutable_candidates"
        )
    initial_start_graph = cf2x._initial_relay_graph(scene_query, starts)
    if not initial_start_graph.fully_relay_connected:
        raise RuntimeError("selected public initial positions are not relay connected")

    for index, position in enumerate(starts):
        environment = UsdGeom.Xform.Define(stage, f"/World/P07Agents/Env_{index}")
        environment.AddTranslateOp().Set(Gf.Vec3d(*position))
    robot = Multirotor(cf2x._multirotor_cfg(paths["cf2x"], args.physics_dt_s))
    contact = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/P07Agents/Env_.*/Robot/.*",
            track_pose=False,
            track_air_time=True,
            force_threshold=cf2x.CONTACT_HARD_FAIL_N,
            history_length=1,
            debug_vis=False,
        )
    )
    sim.reset()
    robot.update(float(sim.cfg.dt))
    root_pose = torch.tensor(
        [[*position, 1.0, 0.0, 0.0, 0.0] for position in starts],
        device=robot.device,
        dtype=torch.float32,
    )
    robot.reset()
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((fleet_size, 6), device=robot.device))
    robot.set_thrust_target(
        torch.full(
            (fleet_size, int(robot.num_thrusters)),
            cf2x.HOVER_THRUST_PER_ROTOR_N,
            device=robot.device,
        )
    )
    robot.write_data_to_sim()
    sim.forward()
    robot.update(float(sim.cfg.dt))
    contact.update(float(sim.cfg.dt), force_recompute=True)
    observed_start_positions = tuple(
        tuple(float(value) for value in row)
        for row in robot.data.root_pos_w.detach().cpu().tolist()
    )
    observed_start_speeds = tuple(
        float(value)
        for value in torch.linalg.norm(robot.data.root_lin_vel_w, dim=1).detach().cpu().tolist()
    )
    reset_position_errors = tuple(
        math.dist(expected, observed)
        for expected, observed in zip(starts, observed_start_positions, strict=True)
    )
    if max(reset_position_errors, default=math.inf) > 1.0e-4:
        raise RuntimeError("P07 CF2X reset does not match the pre-registered shared start poses")
    if max(observed_start_speeds, default=math.inf) > 1.0e-5:
        raise RuntimeError("P07 CF2X reset has non-zero initial linear velocity")
    p0_live_departure_qualification: dict[str, object] | None = None
    if p0_live_departure_required:
        p0_live_departure_qualification = _p0_live_departure_qualification(
            source,
            selected_candidate_ids=selected_start_candidate_ids,
            selected_positions=starts,
            observed_positions=observed_start_positions,
            scene_query=scene_query,
            clearance_oracle=clearance_oracle,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            collision_usd_sha256=_sha256(paths["collision"]),
            start_reset_manifest_sha256=_sha256(paths["start_resets"]),
        )
        if (
            args.p0_start_eligibility_evidence is not None
            and p0_live_departure_qualification.get("passed") is not True
        ):
            raise RuntimeError(
                "P0 full qualification failed live departure revalidation for the selected reset"
            )
    reachable_mask, evaluation_denominator = reachable_component_mask(
        arrays,
        start_positions_m=observed_start_positions,
    )
    evaluation_denominator = {
        **evaluation_denominator,
        "geometry_evaluation_denominator_sha256": geometry_denominator_sha256,
        "flight_space_manifest_hash": p03_row["flight_space_manifest_hash"],
        "source_geometry_sha256": p03_row["source_geometry_sha256"],
        "collision_geometry_sha256": p03_row["collision_geometry_sha256"],
        "vehicle_clearance_m": float(p03_row["vehicle_clearance_m"]),
        "start_reset_manifest_sha256": _sha256(paths["start_resets"]),
    }
    denominator_sha256 = canonical_sha256(evaluation_denominator)
    evaluation_denominator["denominator_sha256"] = denominator_sha256
    denominator_volume_m3 = float(evaluation_denominator["reachable_volume_m3"])
    initial_start_reset_witness = {
        "start_reset_manifest_sha256": _sha256(paths["start_resets"]),
        "reset_schema_version": source.get("schema_version"),
        "reset_selection_rule": source.get("selection_rule"),
        "declared_start_mobility_clearance_m": source.get("start_mobility_clearance_m"),
        "p0_departure_envelope": p0_departure_envelope,
        "p0_live_departure_qualification": p0_live_departure_qualification,
        "selected_start_candidate_ids": list(selected_start_candidate_ids),
        "selection_mode": start_selection_mode,
        "eligibility_evidence_sha256": eligibility_evidence_sha256,
        "selected_start_positions_m": [list(position) for position in starts],
        "observed_root_positions_m": [list(position) for position in observed_start_positions],
        "position_errors_m": list(reset_position_errors),
        "linear_speeds_mps": list(observed_start_speeds),
        "position_tolerance_m": 1.0e-4,
        "speed_tolerance_mps": 1.0e-5,
        "passed": True,
    }

    def new_backend(
        *,
        execution_deadline_s: float | None = None,
        event_driven_action_completion: bool = True,
        on_agent_complete: (
            Callable[[str, float, tuple[float, float, float]], tuple[Any, Any] | None] | None
        ) = None,
    ) -> Any:
        observation_contract = load_exploration_observation_contract()
        sensor_profile = observation_contract.payload["sensor_profile"]
        range_directions = resolve_public_range_directions(
            str(sensor_profile["ray_pattern"])
        )
        range_max_m = float(sensor_profile["maximum_range_m"])
        return cf2x.IsaacCF2XExecutionBackend(
            sim=sim,
            robot=robot,
            contact=contact,
            scene_query=scene_query,
            static_clearance_oracle=clearance_oracle,
            agent_order=tuple(f"uav{index}" for index in range(fleet_size)),
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
            arrival_tolerance_m=args.arrival_tolerance_m,
            execution_deadline_s=execution_deadline_s,
            communication_max_range_m=float(communication_contract.network["maximum_range_m"]),
            communication_base_latency_s=float(communication_contract.network["base_latency_s"]),
            communication_per_hop_latency_s=float(
                communication_contract.network["per_hop_latency_s"]
            ),
            communication_loss_probability=float(
                communication_contract.network["loss_probability"]
            ),
            communication_update_hz=float(communication_contract.network["telemetry_update_hz"]),
            sparse_range_update_hz=profile.update_hz,
            sparse_range_directions=range_directions,
            sparse_range_max_m=range_max_m,
            communication_message_ttl_s=float(
                communication_contract.message_policy["time_to_live_s"]
            ),
            minimum_observation_dwell_s=observation_dwell_s,
            event_driven_action_completion=event_driven_action_completion,
            controller_id=args.controller_id,
            visualization_trace_sample_hz=args.visualization_trace_hz,
            on_agent_complete=on_agent_complete,
        )

    def _final_boundary_linear_speeds(backend: Any, *, stage_id: str) -> tuple[float, ...]:
        """Read the actual end-of-stage velocities for recovery admission.

        The recovery exception is permitted only from an observed near-rest
        boundary.  Do not infer that fact from a nominal dwell or a planned
        waypoint timestamp.
        """

        raw_speeds = getattr(backend, "final_root_linear_speeds_mps", ())
        if not isinstance(raw_speeds, tuple) or len(raw_speeds) != fleet_size:
            raise RuntimeError(
                f"CF2X {stage_id} omitted final per-vehicle linear-speed evidence"
            )
        speeds = tuple(float(speed) for speed in raw_speeds)
        if any(not math.isfinite(speed) or speed < 0.0 for speed in speeds):
            raise RuntimeError(f"CF2X {stage_id} emitted invalid final linear speeds")
        return speeds

    root_context = PublicMethodContext(
        context_id=f"p07-online-{args.scene_id}-n{fleet_size}",
        episode_id=f"p07-online-{args.scene_id}-seed{args.random_key}-n{fleet_size}",
        decision_id="decision0",
        agent_features=tuple((f"uav{index}", (1.0, 1.0)) for index in range(fleet_size)),
        public_features=(("sparse_range_schedule_hz", profile.update_hz),),
        budget=(("time_remaining_s", args.action_budget_s),),
    )
    team_belief = SparseVoxelBelief(args.scene_id, "team", 0.25)
    agent_beliefs = {
        f"uav{index}": SparseVoxelBelief(args.scene_id, f"uav{index}", 0.25)
        for index in range(fleet_size)
    }
    samples = [
        _metric_sample(
            timestamp_s=0.0,
            component=reachable_mask,
            grid_origin=grid_origin,
            resolution_m=0.25,
            denominator_volume_m3=denominator_volume_m3,
            team_belief=team_belief,
        )
    ]
    current_positions = starts
    elapsed_s = 0.0
    round_debug: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    decision_wall_rows: list[dict[str, float | str]] = []
    pool_hashes: list[str] = []
    total_energy_j = 0.0
    total_collision_count = 0
    total_inter_agent_separation_violation_count = 0
    total_oob_count = 0
    total_static_clearance_contract_violation_count = 0
    total_failed_fragments = 0
    total_executed_fragments = 0
    outcome_hashes: list[str] = []
    terminal_budget_tail: dict[str, object] | None = None
    realised_qd_archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    realised_qd_descriptors: list[RealisedQDDescriptor] = []
    realised_qd_intents: list[tuple[float, float, float]] = []
    realised_qd_footprints: list[tuple[tuple[int, int, int], ...]] = []
    realised_qd_admissions: list[dict[str, Any]] = []
    candidate_intent_audits: list[dict[str, Any]] = []
    value_protected_candidate_diversity_audits: list[dict[str, Any]] = []
    team_trajectory_diversity_audits: list[dict[str, Any]] = []
    observation_cooldown = _PublicObservationCooldown()
    planned_qd_selector: PlannedQDSelector | None = None
    realised_qd_selector: OutcomeGroundedQDSelector | None = None
    qd_history_summary: dict[str, Any] = {"mode": "none", "outcome_count": 0}
    # Every selected candidate becomes one outcome-bound training transition.
    # The same real transition may be replayed for multiple gradient updates;
    # no gradient step is allowed to manufacture a new PhysX interaction.
    decision_training_rows: list[dict[str, Any]] = []
    if qd_strategy:
        assert args.qd_history is not None
        history_paths = tuple(path.expanduser().resolve() for path in args.qd_history)
        history, train_descriptor_admission = _load_train_qd_history(
            history_paths,
            split_manifest_sha256=split_manifest_sha256,
        )
        qd_history_summary = {
            "mode": "planned_intent" if args.strategy == "planned_qd" else "outcome_grounded",
            "paths": [str(path) for path in history_paths],
            "sha256s": [_sha256(path) for path in history_paths],
            "outcome_count": len(history),
            "source_partition": "train",
            "source_scene_count": len({row[8] for row in history}),
            "train_descriptor_admission": train_descriptor_admission,
        }
        if args.strategy == "planned_qd":
            planned_qd_selector = PlannedQDSelector(
                QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC),
                utility_slack=QD_UTILITY_SLACK,
            )
            for history_index, (intent, _, quality, cost, *_) in enumerate(history):
                planned_qd_selector.observe_intent(
                    intent,
                    public_quality=quality,
                    public_cost=cost,
                    source_id=f"trainhistory{history_index}",
                )
        else:
            realised_qd_selector = OutcomeGroundedQDSelector(
                realised_qd_archive,
                utility_slack=QD_UTILITY_SLACK,
            )
            for history_index, (
                intent,
                descriptor,
                quality,
                cost,
                candidate_id,
                manifest_hash,
                outcome_hash,
                _footprint,
                _,
            ) in enumerate(history):
                realised_qd_archive.add_or_update(
                    Elite(
                        candidate_id=f"trainhistory{history_index}-{candidate_id}",
                        manifest_hash=manifest_hash,
                        behavior_hash=outcome_hash,
                        realised_descriptor=descriptor.values,
                        quality=quality,
                        cost=cost,
                        feasible=True,
                        source="hm3d-train-outcome-history",
                    )
                )
                realised_qd_selector.observe_intent(
                    intent,
                    descriptor,
                    public_quality=quality,
                    public_cost=cost,
                    execution_outcome_sha256=outcome_hash,
                )
    terminal_outcome = "budget_exhausted"

    def absorb_outcomes(
        backend: Any,
        *,
        stage_id: str,
        timestamp_offset_s: float,
        communication_audit: dict[str, Any],
        integrate_into_belief: bool = True,
    ) -> tuple[PublicRangeRayOutcome, ...]:
        nonlocal team_belief
        if communication_audit.get("passed") is not True:
            raise RuntimeError("undelivered range-map updates cannot enter the shared belief")
        delivery = backend.engineering_diagnostics.get("message_delivery")
        if not isinstance(delivery, dict):
            raise RuntimeError("CF2X backend omitted decision-boundary map-delivery evidence")
        sender_ids = delivery.get("public_map_sender_ids")
        if not isinstance(sender_ids, list) or any(
            not isinstance(agent_id, str) or agent_id not in agent_beliefs
            for agent_id in sender_ids
        ):
            raise RuntimeError("CF2X map-delivery evidence has invalid public senders")
        delivered_sender_ids = frozenset(sender_ids)
        stage_outcomes: list[PublicRangeRayOutcome] = []
        for raw_outcome in backend.public_range_outcomes:
            if raw_outcome.agent_id not in delivered_sender_ids:
                continue
            # Each isolated executor starts its local frame clock at zero.  The
            # stage prefix makes the physical observation identity unique in
            # the episode-level public belief without changing its content.
            shifted = PublicRangeRayOutcome(
                observation_id=f"{stage_id}-{raw_outcome.observation_id}",
                agent_id=raw_outcome.agent_id,
                timestamp_s=timestamp_offset_s + raw_outcome.timestamp_s,
                origin_m=raw_outcome.origin_m,
                endpoint_m=raw_outcome.endpoint_m,
                hit_occupied=raw_outcome.hit_occupied,
            )
            stage_outcomes.append(shifted)
            if integrate_into_belief:
                agent_beliefs[shifted.agent_id].integrate_ray(shifted)
        if integrate_into_belief:
            team_belief = SparseVoxelBelief(args.scene_id, "team", 0.25)
            for belief in agent_beliefs.values():
                team_belief.merge_public(belief)
        if not stage_outcomes:
            # A dropped segment delta is a valid communication outcome.  The
            # next decision receives no new map evidence, rather than silently
            # receiving raw rays that never entered the public fusion state.
            return ()
        return tuple(stage_outcomes)

    setup_wall_s = time.perf_counter() - runtime_wall_started
    bootstrap_wall_started = time.perf_counter()
    bootstrap_context = replace(root_context, decision_id="bootstrap")
    bootstrap_manifest = _bootstrap_manifest(
        bootstrap_context,
        starts,
        transit_timing,
        observe_dwell_s=observation_dwell_s,
    )
    bootstrap_duration_s = max(fragment.planned_end for fragment in bootstrap_manifest.fragments)
    if bootstrap_duration_s >= args.action_budget_s:
        raise ValueError("P07 action budget cannot accommodate the public sensing bootstrap")
    bootstrap_token = authorize_manifest(
        bootstrap_context,
        (bootstrap_manifest,),
        (True,),
        0,
        token_id=f"p07-online-bootstrap-token-{uuid.uuid4().hex}",
        issued_at=0.0,
        duration=bootstrap_duration_s,
    )
    bootstrap_backend = new_backend()
    bootstrap_ledger = execute_hm3d_manifest(
        bootstrap_manifest,
        bootstrap_token,
        bootstrap_backend,
        time_tolerance_s=args.outcome_time_tolerance_s,
        command_path_tolerance_m=0.25,
    )
    bootstrap_communication = bootstrap_backend.engineering_diagnostics["communication"]
    bootstrap_delivery = bootstrap_backend.engineering_diagnostics["message_delivery"]
    if not isinstance(bootstrap_communication, dict) or not isinstance(bootstrap_delivery, dict):
        raise RuntimeError("CF2X bootstrap omitted communication denominators")
    bootstrap_communication_audit = communication_contract.audit_worker_evidence(
        bootstrap_communication, bootstrap_delivery
    )
    if bootstrap_communication_audit["passed"] is not True:
        raise RuntimeError(
            "public sparse-range bootstrap violates the frozen communication contract: "
            f"{json.dumps(bootstrap_communication_audit, sort_keys=True)}"
        )
    latest_public_outcomes = absorb_outcomes(
        bootstrap_backend,
        stage_id="bootstrap",
        timestamp_offset_s=0.0,
        communication_audit=bootstrap_communication_audit,
    )
    bootstrap_elapsed_s = max(
        (outcome.actual_end for outcome in bootstrap_ledger.outcomes if outcome.executed),
        default=bootstrap_duration_s,
    )
    if (
        bootstrap_ledger.failed_fragment_count
        or bootstrap_ledger.collision_count
        or bootstrap_ledger.inter_agent_separation_violation_count
        or bootstrap_ledger.out_of_bounds_count
        or bootstrap_ledger.static_clearance_contract_violation_count
        or bootstrap_elapsed_s <= 0.0
    ):
        raise RuntimeError("public sparse-range bootstrap did not complete physically")
    elapsed_s = bootstrap_elapsed_s
    current_positions = bootstrap_backend.final_root_positions_m
    if len(current_positions) != fleet_size:
        raise RuntimeError("CF2X bootstrap omitted final physical positions")
    current_boundary_linear_speeds_mps = _final_boundary_linear_speeds(
        bootstrap_backend, stage_id="bootstrap"
    )
    round_debug.append(
        {"communication": bootstrap_communication, "message_delivery": bootstrap_delivery}
    )
    total_energy_j += bootstrap_ledger.total_energy_used_j
    total_collision_count += bootstrap_ledger.collision_count
    total_inter_agent_separation_violation_count += (
        bootstrap_ledger.inter_agent_separation_violation_count
    )
    total_oob_count += bootstrap_ledger.out_of_bounds_count
    total_static_clearance_contract_violation_count += (
        bootstrap_ledger.static_clearance_contract_violation_count
    )
    total_failed_fragments += bootstrap_ledger.failed_fragment_count
    total_executed_fragments += bootstrap_ledger.executed_fragment_count
    outcome_hashes.extend(outcome.digest for outcome in bootstrap_ledger.outcomes)
    samples.append(
        _metric_sample(
            timestamp_s=elapsed_s,
            component=reachable_mask,
            grid_origin=grid_origin,
            resolution_m=0.25,
            denominator_volume_m3=denominator_volume_m3,
            team_belief=team_belief,
        )
    )
    periodic_supervision: _PeriodicSupervisionLedger | None = None
    if supervision_enabled:
        assert args.supervision_interval_s is not None
        periodic_supervision = _PeriodicSupervisionLedger(
            interval_s=float(args.supervision_interval_s),
            agent_ids=tuple(f"uav{index}" for index in range(fleet_size)),
            start_positions_m=tuple(tuple(float(value) for value in point) for point in starts),
            next_timestamp_s=float(args.supervision_interval_s),
        )
        periodic_supervision.accumulate_delivery(bootstrap_delivery)
        periodic_supervision.emit_until(
            elapsed_s=elapsed_s,
            positions_m=current_positions,
            linear_speeds_mps=current_boundary_linear_speeds_mps,
            samples=samples,
            horizon_s=args.action_budget_s,
            total_energy_j=total_energy_j,
            collision_count=total_collision_count,
            separation_violation_count=total_inter_agent_separation_violation_count,
            out_of_bounds_count=total_oob_count,
            static_clearance_violation_count=total_static_clearance_contract_violation_count,
            executed_fragment_count=total_executed_fragments,
            failed_fragment_count=total_failed_fragments,
            decision_count=0,
        )
    bootstrap = {
        "manifest_hash": bootstrap_manifest.manifest_hash,
        "public_observation_frame_count": len(bootstrap_backend.public_range_frames),
        "public_range_ray_count": len(latest_public_outcomes),
        "elapsed_physics_s": elapsed_s,
        "execution": bootstrap_ledger.to_public_dict(),
    }
    if visualization_trace_enabled:
        bootstrap_trace = bootstrap_backend.engineering_diagnostics.get("physics_visualization_trace")
        if not isinstance(bootstrap_trace, dict):
            raise RuntimeError("CF2X bootstrap omitted the requested visualization trace")
        bootstrap["physics_visualization_trace"] = bootstrap_trace
    else:
        bootstrap_trace = None
    if periodic_supervision is not None:
        periodic_supervision.accumulate_trace(bootstrap_trace, start_s=0.0)
    bootstrap_wall_s = time.perf_counter() - bootstrap_wall_started
    bootstrap_auc_contribution = 0.0
    for previous, current in zip(samples, samples[1:], strict=False):
        bootstrap_auc_contribution += (
            0.5
            * (current.timestamp_s - previous.timestamp_s)
            * (previous.coverage_fraction + current.coverage_fraction)
            / args.action_budget_s
    )
    decision_index = 0
    # Per-agent, in-memory recovery history. It is intentionally absent from
    # SparseVoxelBelief: a successful physical path is a guarded execution
    # outcome, not a new public FREE observation.
    backtrack_history_by_agent: dict[str, list[_OutcomeBacktrackRoute]] = {
        f"uav{index}": [] for index in range(fleet_size)
    }
    outcome_backtrack_audit: list[dict[str, object]] = []
    # Reservations retain only public task association plus successful outcome
    # provenance. They never retain a stale manifest as an executable action.
    task_reservations_by_agent: dict[str, PublicTaskReservation] = {}
    task_reservation_audit: list[dict[str, object]] = []
    terminal_margin_distance_audit_m = maximum_rest_to_rest_distance_m(
        transit_timing.terminal_tracking_margin_s,
        cruise_speed_mps=transit_timing.cruise_speed_mps,
        max_accel_mps2=transit_timing.max_accel_mps2,
    )
    while elapsed_s < args.action_budget_s - 1.0e-9:
        if decision_index >= args.max_decision_count:
            raise RuntimeError(
                "P07 reached max_decision_count before exhausting its physical budget; "
                "this is a non-progress safeguard, not a scoreable episode"
            )
        if elapsed_s >= args.action_budget_s - 1.0e-9:
            break
        decision_wall_started = time.perf_counter()
        planning_wall_started = decision_wall_started
        # Fragment timestamps are local to one action token.  The episode
        # clock is retained in the public remaining-budget field and in the
        # worker record; passing its non-zero value into the executor would
        # compare a local PhysX outcome against a previous round's timestamp.
        decision_context = replace(
            root_context,
            decision_id=f"decision{decision_index}",
            budget=(("time_remaining_s", args.action_budget_s - elapsed_s),),
        )
        belief_before_hash = team_belief.content_sha256
        initial_graph_wall_started = time.perf_counter()
        initial_graph = cf2x._initial_relay_graph(scene_query, current_positions)
        initial_graph_wall_s = time.perf_counter() - initial_graph_wall_started
        # The executor may finish early once every active route has reached its
        # observation dwell.  The remaining episode budget is only a hard
        # physical deadline, never a fixed idle window.
        decision_remaining_s = args.action_budget_s - elapsed_s
        decision_duration_s = decision_remaining_s
        if args.rolling_decision_window_s is not None:
            decision_duration_s = min(
                decision_remaining_s,
                float(args.rolling_decision_window_s),
            )
        reachable_path_length_m = outcome_calibrated_path_length_budget_m(
            decision_duration_s=decision_duration_s,
            observe_dwell_s=observation_dwell_s,
            transit_timing_model=transit_timing,
        )
        # The active route horizon is the distance that the shared timing
        # contract can still execute and dwell within. It is not a fixed-step
        # parameter: a route may be shorter when public free-space evidence
        # genuinely ends sooner.
        effective_frontier_step_m = reachable_path_length_m
        if effective_frontier_step_m < 0.25:
            if decision_duration_s < observation_dwell_s:
                # There is no legal observation left in this physical
                # remainder.  End at the last executed outcome and let the
                # fixed-horizon metric carry its coverage forward; do not
                # manufacture a dwell outcome merely to reach the deadline.
                terminal_budget_tail = _unexecuted_budget_tail_record(
                    duration_s=decision_duration_s,
                    observe_dwell_s=observation_dwell_s,
                )
                break
            tail_duration_grid_s = (
                math.floor(decision_duration_s / args.physics_dt_s)
                * args.physics_dt_s
            )
            if tail_duration_grid_s < observation_dwell_s + args.physics_dt_s:
                terminal_budget_tail = {
                    "manifest_hash": None,
                    "elapsed_physics_s": 0.0,
                    "unexecuted_remainder_s": decision_duration_s,
                    "scheduled_completion_mode": (
                        "unexecuted_budget_remainder_not_on_physics_grid"
                    ),
                    "execution": None,
                }
                break
            min_executable_tail_s = _min_stationary_budget_tail_s(
                observation_dwell_s=observation_dwell_s,
                physics_dt_s=args.physics_dt_s,
            )
            if tail_duration_grid_s < min_executable_tail_s:
                terminal_budget_tail = {
                    "manifest_hash": None,
                    "elapsed_physics_s": 0.0,
                    "unexecuted_remainder_s": decision_duration_s,
                    "scheduled_completion_mode": "unexecuted_budget_remainder_below_observation_dwell",
                    "execution": None,
                }
                break
            tail_context = replace(
                root_context,
                decision_id=f"budget_tail{decision_index}",
                budget=(("time_remaining_s", tail_duration_grid_s),),
            )
            tail_manifest = _budget_tail_manifest(
                tail_context,
                current_positions,
                duration_s=tail_duration_grid_s,
                observe_dwell_s=observation_dwell_s,
            )
            tail_token = authorize_manifest(
                tail_context,
                (tail_manifest,),
                (True,),
                0,
                token_id=f"p07-online-budget-tail-token-{uuid.uuid4().hex}",
                issued_at=0.0,
                duration=tail_duration_grid_s,
            )
            tail_backend = new_backend(event_driven_action_completion=False)
            tail_ledger = execute_hm3d_manifest(
                tail_manifest,
                tail_token,
                tail_backend,
                time_tolerance_s=args.outcome_time_tolerance_s,
                command_path_tolerance_m=0.25,
            )
            tail_communication = tail_backend.engineering_diagnostics["communication"]
            tail_delivery = tail_backend.engineering_diagnostics["message_delivery"]
            if not isinstance(tail_communication, dict) or not isinstance(tail_delivery, dict):
                raise RuntimeError("CF2X budget tail omitted communication denominators")
            tail_communication_audit = communication_contract.audit_worker_evidence(
                tail_communication,
                tail_delivery,
            )
            if tail_communication_audit["passed"] is not True:
                raise RuntimeError("CF2X budget tail violates the frozen communication contract")
            absorb_outcomes(
                tail_backend,
                stage_id="budget_tail",
                timestamp_offset_s=elapsed_s,
                communication_audit=tail_communication_audit,
            )
            tail_elapsed_s = max(
                (outcome.actual_end for outcome in tail_ledger.outcomes if outcome.executed),
                default=0.0,
            )
            if (
                tail_ledger.failed_fragment_count
                or tail_ledger.collision_count
                or tail_ledger.inter_agent_separation_violation_count
                or tail_ledger.out_of_bounds_count
                or tail_ledger.static_clearance_contract_violation_count
                or not math.isclose(
                    tail_elapsed_s, tail_duration_grid_s, abs_tol=args.physics_dt_s
                )
            ):
                raise RuntimeError(
                    "shared physical budget tail did not complete safely: "
                    f"duration_s={tail_duration_grid_s}, elapsed_s={tail_elapsed_s}, "
                    f"failed_fragments={tail_ledger.failed_fragment_count}, "
                    f"collision={tail_ledger.collision_count}, "
                    f"separation_violations={tail_ledger.inter_agent_separation_violation_count}, "
                    f"out_of_bounds={tail_ledger.out_of_bounds_count}, "
                    f"clearance_violations={tail_ledger.static_clearance_contract_violation_count}"
                )
            elapsed_s += tail_elapsed_s
            samples.append(
                _metric_sample(
                    timestamp_s=elapsed_s,
                    component=reachable_mask,
                    grid_origin=grid_origin,
                    resolution_m=0.25,
                    denominator_volume_m3=denominator_volume_m3,
                    team_belief=team_belief,
                )
            )
            round_debug.append(
                {"communication": tail_communication, "message_delivery": tail_delivery}
            )
            total_energy_j += tail_ledger.total_energy_used_j
            total_collision_count += tail_ledger.collision_count
            total_inter_agent_separation_violation_count += (
                tail_ledger.inter_agent_separation_violation_count
            )
            total_oob_count += tail_ledger.out_of_bounds_count
            total_static_clearance_contract_violation_count += (
                tail_ledger.static_clearance_contract_violation_count
            )
            total_failed_fragments += tail_ledger.failed_fragment_count
            total_executed_fragments += tail_ledger.executed_fragment_count
            outcome_hashes.extend(outcome.digest for outcome in tail_ledger.outcomes)
            tail_trace = tail_backend.engineering_diagnostics.get("physics_visualization_trace")
            if periodic_supervision is not None:
                if len(tail_backend.final_root_positions_m) == fleet_size:
                    tail_positions = tail_backend.final_root_positions_m
                    tail_speeds = _final_boundary_linear_speeds(
                        tail_backend, stage_id="budget_tail"
                    )
                else:
                    tail_positions = current_positions
                    tail_speeds = current_boundary_linear_speeds_mps
                periodic_supervision.accumulate_delivery(tail_delivery)
                periodic_supervision.accumulate_trace(
                    tail_trace,
                    start_s=elapsed_s - tail_elapsed_s,
                )
                periodic_supervision.emit_until(
                    elapsed_s=elapsed_s,
                    positions_m=tail_positions,
                    linear_speeds_mps=tail_speeds,
                    samples=samples,
                    horizon_s=args.action_budget_s,
                    total_energy_j=total_energy_j,
                    collision_count=total_collision_count,
                    separation_violation_count=total_inter_agent_separation_violation_count,
                    out_of_bounds_count=total_oob_count,
                    static_clearance_violation_count=total_static_clearance_contract_violation_count,
                    executed_fragment_count=total_executed_fragments,
                    failed_fragment_count=total_failed_fragments,
                    decision_count=decision_index,
                )
            terminal_budget_tail = {
                "manifest_hash": tail_manifest.manifest_hash,
                "elapsed_physics_s": tail_elapsed_s,
                "unexecuted_remainder_s": (
                    decision_duration_s - tail_duration_grid_s
                ),
                "executed_from_episode_s": elapsed_s - tail_elapsed_s,
                "scheduled_completion_mode": "planned_boundary_for_unrouteable_budget_tail",
                "execution": tail_ledger.to_public_dict(),
            }
            if visualization_trace_enabled:
                if not isinstance(tail_trace, dict):
                    raise RuntimeError("CF2X budget tail omitted the requested visualization trace")
                terminal_budget_tail["physics_visualization_trace"] = tail_trace
            break
        observation_cooldown.begin_decision(decision_index)
        frontier_wall_started = time.perf_counter()
        # The public belief is immutable while this decision's candidate pool is
        # constructed. Share one component diagnostic cache across frontier
        # generation and any later public-route fallback guards, then discard it
        # before the next outcome changes the belief.
        public_reachability_cache = _PublicFreeReachabilityCache(team_belief)
        public_frontiers = _public_frontiers_from_belief(
            current_positions,
            team_belief,
            decision_index=decision_index,
            maximum_step_m=effective_frontier_step_m,
            observation_cooldown=observation_cooldown,
            task_reservations=tuple(task_reservations_by_agent.values()),
            reachability_cache=public_reachability_cache,
        )
        if len(public_frontiers) < fleet_size:
            # The public belief is saturated: fewer legal frontiers remain
            # than agents, so a team candidate cannot be constructed.  Finish
            # the episode here instead of failing.  The remaining physical
            # budget is consumed by an honest stationary sensing tail (the
            # same contract as the loop-top budget tail): the fleet hovers
            # and keeps sampling, the frozen denominator and the budget-
            # exhausted terminal outcome are preserved, and no safety
            # contract is touched.
            if not samples or samples[-1].timestamp_s + 1.0e-9 < elapsed_s:
                samples.append(
                    _metric_sample(
                        timestamp_s=elapsed_s,
                        component=reachable_mask,
                        grid_origin=grid_origin,
                        resolution_m=0.25,
                        denominator_volume_m3=denominator_volume_m3,
                        team_belief=team_belief,
                    )
                )
            saturation_remainder_s = args.action_budget_s - elapsed_s
            tail_duration_grid_s = (
                math.floor(saturation_remainder_s / args.physics_dt_s)
                * args.physics_dt_s
            )
            min_executable_tail_s = _min_stationary_budget_tail_s(
                observation_dwell_s=observation_dwell_s,
                physics_dt_s=args.physics_dt_s,
            )
            if tail_duration_grid_s < min_executable_tail_s:
                terminal_budget_tail = {
                    "manifest_hash": None,
                    "elapsed_physics_s": 0.0,
                    "unexecuted_remainder_s": saturation_remainder_s,
                    "scheduled_completion_mode": "saturated_unexecuted_budget_remainder",
                    "execution": None,
                }
                elapsed_s = args.action_budget_s
                terminal_outcome = "budget_exhausted"
                break
            tail_context = replace(
                root_context,
                decision_id=f"budget_tail{decision_index}",
                budget=(("time_remaining_s", tail_duration_grid_s),),
            )
            tail_manifest = _budget_tail_manifest(
                tail_context,
                current_positions,
                duration_s=tail_duration_grid_s,
                observe_dwell_s=observation_dwell_s,
            )
            tail_token = authorize_manifest(
                tail_context,
                (tail_manifest,),
                (True,),
                0,
                token_id=f"p07-online-saturation-tail-token-{uuid.uuid4().hex}",
                issued_at=0.0,
                duration=tail_duration_grid_s,
            )
            tail_backend = new_backend(event_driven_action_completion=False)
            tail_ledger = execute_hm3d_manifest(
                tail_manifest,
                tail_token,
                tail_backend,
                time_tolerance_s=args.outcome_time_tolerance_s,
                command_path_tolerance_m=0.25,
            )
            tail_communication = tail_backend.engineering_diagnostics["communication"]
            tail_delivery = tail_backend.engineering_diagnostics["message_delivery"]
            if not isinstance(tail_communication, dict) or not isinstance(tail_delivery, dict):
                raise RuntimeError("CF2X saturation tail omitted communication denominators")
            tail_communication_audit = communication_contract.audit_worker_evidence(
                tail_communication, tail_delivery
            )
            if tail_communication_audit["passed"] is not True:
                raise RuntimeError(
                    "CF2X saturation tail violates the frozen communication contract"
                )
            absorb_outcomes(
                tail_backend,
                stage_id="saturation_tail",
                timestamp_offset_s=elapsed_s,
                communication_audit=tail_communication_audit,
            )
            tail_elapsed_s = max(
                (outcome.actual_end for outcome in tail_ledger.outcomes if outcome.executed),
                default=0.0,
            )
            if (
                tail_ledger.failed_fragment_count
                or tail_ledger.collision_count
                or tail_ledger.inter_agent_separation_violation_count
                or tail_ledger.out_of_bounds_count
                or tail_ledger.static_clearance_contract_violation_count
                or not math.isclose(
                    tail_elapsed_s, tail_duration_grid_s, abs_tol=args.physics_dt_s
                )
            ):
                raise RuntimeError(
                    "saturation tail did not complete safely: "
                    f"duration_s={tail_duration_grid_s}, elapsed_s={tail_elapsed_s}, "
                    f"failed={tail_ledger.failed_fragment_count}"
                )
            elapsed_s += tail_elapsed_s
            samples.append(
                _metric_sample(
                    timestamp_s=elapsed_s,
                    component=reachable_mask,
                    grid_origin=grid_origin,
                    resolution_m=0.25,
                    denominator_volume_m3=denominator_volume_m3,
                    team_belief=team_belief,
                )
            )
            round_debug.append(
                {"communication": tail_communication, "message_delivery": tail_delivery}
            )
            total_energy_j += tail_ledger.total_energy_used_j
            total_collision_count += tail_ledger.collision_count
            total_inter_agent_separation_violation_count += (
                tail_ledger.inter_agent_separation_violation_count
            )
            total_oob_count += tail_ledger.out_of_bounds_count
            total_static_clearance_contract_violation_count += (
                tail_ledger.static_clearance_contract_violation_count
            )
            total_failed_fragments += tail_ledger.failed_fragment_count
            total_executed_fragments += tail_ledger.executed_fragment_count
            outcome_hashes.extend(outcome.digest for outcome in tail_ledger.outcomes)
            tail_trace = tail_backend.engineering_diagnostics.get("physics_visualization_trace")
            if periodic_supervision is not None:
                if len(tail_backend.final_root_positions_m) == fleet_size:
                    tail_positions = tail_backend.final_root_positions_m
                    tail_speeds = _final_boundary_linear_speeds(
                        tail_backend, stage_id="saturation_tail"
                    )
                else:
                    tail_positions = current_positions
                    tail_speeds = current_boundary_linear_speeds_mps
                periodic_supervision.accumulate_delivery(tail_delivery)
                periodic_supervision.accumulate_trace(
                    tail_trace, start_s=elapsed_s - tail_elapsed_s
                )
                periodic_supervision.emit_until(
                    elapsed_s=elapsed_s,
                    positions_m=tail_positions,
                    linear_speeds_mps=tail_speeds,
                    samples=samples,
                    horizon_s=args.action_budget_s,
                    total_energy_j=total_energy_j,
                    collision_count=total_collision_count,
                    separation_violation_count=total_inter_agent_separation_violation_count,
                    out_of_bounds_count=total_oob_count,
                    static_clearance_violation_count=total_static_clearance_contract_violation_count,
                    executed_fragment_count=total_executed_fragments,
                    failed_fragment_count=total_failed_fragments,
                    decision_count=decision_index,
                )
            terminal_budget_tail = {
                "manifest_hash": tail_manifest.manifest_hash,
                "elapsed_physics_s": tail_elapsed_s,
                "unexecuted_remainder_s": saturation_remainder_s - tail_duration_grid_s,
                "executed_from_episode_s": elapsed_s - tail_elapsed_s,
                "scheduled_completion_mode": "saturated_stationary_sensing_tail",
                "execution": tail_ledger.to_public_dict(),
            }
            if visualization_trace_enabled:
                if not isinstance(tail_trace, dict):
                    raise RuntimeError("CF2X saturation tail omitted the visualization trace")
                terminal_budget_tail["physics_visualization_trace"] = tail_trace
            terminal_outcome = "budget_exhausted"
            break
        predecision_task_reservation_outcomes: list[dict[str, object]] = []
        for agent_id, reservation in tuple(sorted(task_reservations_by_agent.items())):
            matching_frontier_ids = [
                frontier.frontier_id
                for frontier in public_frontiers
                if task_reservation_matches_frontier(reservation, frontier)[0]
            ]
            if matching_frontier_ids:
                predecision_task_reservation_outcomes.append(
                    {
                        "agent_id": agent_id,
                        "action": "retained_after_current_public_task_revalidation",
                        "matching_frontier_ids": matching_frontier_ids,
                    }
                )
                continue
            task_reservations_by_agent.pop(agent_id)
            predecision_task_reservation_outcomes.append(
                {
                    "agent_id": agent_id,
                    "action": "released",
                    "release_reason": "task_not_revalidated_by_current_public_frontiers",
                    "matching_frontier_ids": [],
                }
            )
        outcome_backtrack_routes: dict[
            tuple[str, tuple[float, float, float]],
            tuple[_OutcomeBacktrackRoute, tuple[tuple[float, float, float], ...]],
        ] = {}
        outcome_backtrack_offers: list[dict[str, object]] = []
        ordinary_endpoints = tuple(frontier.position_m for frontier in public_frontiers)
        for agent_index, current_position_m in enumerate(current_positions):
            agent_id = f"uav{agent_index}"
            history = backtrack_history_by_agent[agent_id]
            if not history:
                continue
            candidate = _outcome_backtrack_frontier(
                current_position_m=current_position_m,
                route=history[-1],
                arrival_tolerance_m=args.arrival_tolerance_m,
                occupied_endpoints_m=ordinary_endpoints,
            )
            if candidate is None:
                outcome_backtrack_offers.append(
                    {
                        "agent_id": agent_id,
                        "route_id": history[-1].route_id,
                        "available": False,
                        "reason": "not_at_successful_route_endpoint_or_ambiguous_endpoint",
                    }
                )
                continue
            frontier, reverse_path_m = candidate
            public_frontiers = (*public_frontiers, frontier)
            outcome_backtrack_routes[(agent_id, frontier.position_m)] = (history[-1], reverse_path_m)
            outcome_backtrack_offers.append(
                {
                    "agent_id": agent_id,
                    "route_id": history[-1].route_id,
                    "available": True,
                    "source_decision_id": history[-1].source_decision_id,
                    "source_transit_outcome_sha256": history[-1].source_transit_outcome_sha256,
                    "path_length_m": _path_length_m(reverse_path_m),
                }
            )
        frontier_wall_s = time.perf_counter() - frontier_wall_started
        state = PublicSearchState(
            context=decision_context,
            agents=tuple(
                PublicAgentPose(
                    f"uav{index}",
                    current_positions[index],
                    1.0,
                    sum(initial_graph.adjacency[index]),
                )
                for index in range(fleet_size)
            ),
            frontiers=public_frontiers,
            decision_start_s=0.0,
            decision_duration_s=decision_duration_s,
            transit_timing_model=transit_timing,
            observe_dwell_s=observation_dwell_s,
            communication_range_m=float(communication_contract.network["maximum_range_m"]),
            task_reservations=tuple(task_reservations_by_agent.values()),
        )
        route_cache: dict[
            tuple[str, tuple[tuple[float, float, float], ...]],
            tuple[Any, dict[str, object]],
        ] = {}
        segment_guard_cache: dict[
            tuple[
                str,
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            Any,
        ] = {}
        exact_clearance_grid_rescue_budget = {
            "remaining": EXACT_CLEARANCE_GRID_RESCUE_MAX_PATHS_PER_DECISION
        }
        route_guard_records: list[dict[str, object]] = []
        route_guard_timing = {
            "unique_wall_s": 0.0,
            "unique_query_count": 0,
            "wall_s_by_status": {},
        }
        joint_guard_timing = {"wall_s": 0.0, "call_count": 0}
        joint_guard_records: list[dict[str, object]] = []
        public_route_nodes = tuple(frontier.position_m for frontier in state.frontiers)
        # This map is intentionally scoped to one freshly built public state.
        # A path cannot retain this authority across decisions: the next
        # candidate pool must construct and register it again from the next
        # sparse belief and current vehicle pose.
        public_access_path_authority = {
            (agent_id, path_m): frontier.frontier_id
            for frontier in state.frontiers
            for agent_id, path_m in frontier.access_paths_m
        }

        def route_guard(
            agent_id: str,
            path: tuple[tuple[float, float, float], ...],
            *,
            cached_paths: dict[
                tuple[str, tuple[tuple[float, float, float], ...]],
                tuple[Any, dict[str, object]],
            ] = route_cache,
            public_nodes: tuple[tuple[float, float, float], ...] = public_route_nodes,
            records: list[dict[str, object]] = route_guard_records,
            timing: dict[str, float | int] = route_guard_timing,
            maximum_path_length_m: float = effective_frontier_step_m,
            outcome_backtracks: dict[
                tuple[str, tuple[float, float, float]],
                tuple[_OutcomeBacktrackRoute, tuple[tuple[float, float, float], ...]],
            ] = outcome_backtrack_routes,
            access_path_authority: dict[
                tuple[str, tuple[tuple[float, float, float], ...]], str
            ] = public_access_path_authority,
            segment_cache: dict[
                tuple[
                    str,
                    tuple[float, float, float],
                    tuple[float, float, float],
                ],
                Any,
            ] = segment_guard_cache,
        ) -> Any:
            cache_key = (agent_id, path)
            cached = cached_paths.get(cache_key)
            if cached is None:
                hit_events: list[dict[str, object]] = []
                query_wall_started = time.perf_counter()
                query_wall_s = 0.0
                public_route_status: str | None = None
                public_access_frontier_id: str | None = None
                recovery_audit: dict[str, object] = {}
                terminal_pullback: dict[str, object] | None = None
                requested_path = path
                public_path_result: _PublicFreePathResult | None = None
                try:
                    if len(path) >= 2 and math.dist(path[0], path[-1]) > 1.0e-9:
                        recovery = outcome_backtracks.get((agent_id, path[-1]))
                        if recovery is not None:
                            source_route, reverse_path_m = recovery
                            requested_path = reverse_path_m
                            public_route_status = "outcome_backtrack"
                            clearance_reuse = _outcome_backtrack_clearance_reuse_audit(
                                current_position_m=path[0],
                                route=source_route,
                                requested_path_m=requested_path,
                            )
                            if clearance_reuse["admitted"] is not True:
                                guarded = GuardedPath(
                                    False,
                                    requested_path,
                                    reason=str(clearance_reuse["reason"]),
                                )
                            else:
                                connector_end = source_route.path_m[-1]
                                connector_distance_m = math.dist(path[0], connector_end)
                                connector_hit = (
                                    None
                                    if connector_distance_m <= 1.0e-9
                                    else cf2x._first_static_scene_hit(
                                        scene_query,
                                        path[0],
                                        connector_end,
                                        endpoint_margin_m=0.05,
                                    )
                                )
                                if connector_hit is not None:
                                    hit_events.append(
                                        cf2x._raycast_guard_diagnostic(
                                            agent_id=agent_id,
                                            start=path[0],
                                            end=connector_end,
                                            requested_distance_m=connector_distance_m,
                                            raycast_distance_m=max(
                                                0.0, connector_distance_m - 0.05
                                            ),
                                            hit=connector_hit,
                                        )
                                    )
                                    guarded = GuardedPath(
                                        False,
                                        requested_path,
                                        reason="outcome_backtrack_connector_blocked",
                                    )
                                else:
                                    guarded = cf2x._admit_trackable_path(
                                        GuardedPath(True, requested_path),
                                        bounds_min,
                                        bounds_max,
                                    )
                            recovery_audit = {
                                "route_authority": "outcome_backtrack_source_clearance_reuse",
                                "backtrack_route_id": source_route.route_id,
                                "source_decision_id": source_route.source_decision_id,
                                "source_manifest_hash": source_route.source_manifest_hash,
                                "source_transit_outcome_sha256": (
                                    source_route.source_transit_outcome_sha256
                                ),
                                "source_path_sha256": canonical_sha256(source_route.path_m),
                                "route_consumption_count": 1,
                                "static_clearance_reuse": clearance_reuse,
                                "connector_static_raycast_checked": True,
                            }
                        else:
                            public_access_frontier_id = access_path_authority.get(
                                (agent_id, path)
                            )
                            if public_access_frontier_id is not None:
                                # The path was generated from this decision's
                                # public sparse belief and matched its current
                                # agent pose before candidate construction.
                                # Do not collapse it to an endpoint here: the
                                # shared static guard verifies every retained
                                # segment against frozen collision geometry.
                                requested_path = path
                                public_route_status = "revalidated_public_access_plan"
                                guarded = cf2x._routed_guard(
                                    scene_query,
                                    clearance_oracle,
                                    public_nodes,
                                    agent_id,
                                    requested_path,
                                    bounds_min,
                                    bounds_max,
                                    hit_events.append,
                                    allow_public_reroute=False,
                                    segment_cache=segment_cache,
                                )
                            else:
                                public_path_result = _public_free_space_path_result(
                                    team_belief,
                                    path[0],
                                    path[-1],
                                    maximum_path_length_m=maximum_path_length_m,
                                    minimum_received_free_support_m=PUBLIC_ROUTE_SUPPORT_RADIUS_M,
                                    reachability_cache=public_reachability_cache,
                                )
                                public_route_status = public_path_result.status
                                public_free_path = public_path_result.path_m
                                if public_free_path is None:
                                    guarded = GuardedPath(
                                        False,
                                        path,
                                        reason="no_public_free_path",
                                    )
                                else:
                                    requested_path = public_free_path
                                    guarded = cf2x._routed_guard(
                                        scene_query,
                                        clearance_oracle,
                                        public_nodes,
                                        agent_id,
                                        requested_path,
                                        bounds_min,
                                        bounds_max,
                                        hit_events.append,
                                        allow_public_reroute=False,
                                        segment_cache=segment_cache,
                                    )
                    else:
                        guarded = cf2x._routed_guard(
                            scene_query,
                            clearance_oracle,
                            public_nodes,
                            agent_id,
                            path,
                            bounds_min,
                            bounds_max,
                            hit_events.append,
                            allow_public_reroute=False,
                            segment_cache=segment_cache,
                        )
                    if (
                        not guarded.legal
                        and guarded.reason == "insufficient_continuous_collision_clearance"
                        and public_route_status not in {"outcome_backtrack", None}
                        and len(requested_path) >= 2
                    ):
                        pullback_guarded, terminal_pullback = (
                            _terminal_clearance_pullback_guarded_path(
                                team_belief,
                                scene_query,
                                clearance_oracle,
                                agent_id,
                                requested_path,
                                bounds_min,
                                bounds_max,
                                voxel_keys=(
                                    public_path_result.voxel_keys
                                    if public_path_result is not None
                                    and public_path_result.path_m is not None
                                    else None
                                ),
                                exact_clearance_grid_rescue_budget=exact_clearance_grid_rescue_budget,
                                diagnostic_sink=hit_events.append,
                                segment_cache=segment_cache,
                            )
                        )
                        if pullback_guarded is not None:
                            guarded = pullback_guarded
                            public_route_status = (
                                "exact_clearance_grid_route"
                                if terminal_pullback is not None
                                and terminal_pullback.get("pullback_source")
                                == "exact_clearance_grid_route"
                                else "terminal_clearance_pullback"
                            )
                finally:
                    query_wall_s = time.perf_counter() - query_wall_started
                    timing["unique_wall_s"] += query_wall_s
                    timing["unique_query_count"] += 1
                    status_rows = timing.setdefault("wall_s_by_status", {})
                    status_key = str(public_route_status)
                    status_row = status_rows.setdefault(
                        status_key,
                        {"count": 0, "wall_s": 0.0},
                    )
                    status_row["count"] += 1
                    status_row["wall_s"] += query_wall_s
                base_record = _route_guard_record(
                    agent_id=agent_id,
                    requested_path_m=requested_path,
                    guarded=guarded,
                    events=hit_events,
                    public_route_status=public_route_status,
                    terminal_pullback=terminal_pullback,
                )
                base_record.update(recovery_audit)
                base_record["public_access_frontier_id"] = public_access_frontier_id
                base_record["route_guard_unique_wall_s"] = query_wall_s
                cached_paths[cache_key] = (guarded, base_record)
                cache_hit = False
            else:
                guarded, base_record = cached
                cache_hit = True
            records.append(
                {
                    **base_record,
                    "request_index": len(records),
                    "cache_hit": cache_hit,
                }
            )
            return guarded

        def _joint_guard(
            manifest: CandidateFragmentManifest,
            *,
            records: list[dict[str, object]] = joint_guard_records,
            decision_start_s: float = state.decision_start_s,
            boundary_linear_speeds_mps: tuple[float, ...] = current_boundary_linear_speeds_mps,
        ) -> str | None:
            transit_fragments = tuple(
                fragment
                for fragment in manifest.fragments
                if fragment.type_signature.fragment_type == "transit"
            )
            endpoint_by_agent = {
                fragment.agent_id: tuple(fragment.path[-1]) for fragment in transit_fragments
            }
            expected_agent_ids = tuple(f"uav{index}" for index in range(fleet_size))
            if tuple(sorted(endpoint_by_agent)) != expected_agent_ids:
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "reason": "incomplete_fleet_transit_endpoints",
                    }
                )
                return "incomplete_fleet_transit_endpoints"
            transit_features = {
                fragment.agent_id: dict(fragment.type_signature.public_features)
                for fragment in transit_fragments
            }
            recovery_kind_values = tuple(
                features.get("safety_recovery_kind", "")
                for features in transit_features.values()
            )
            recovery_agent_values = tuple(
                features.get("safety_recovery_agent_id", "")
                for features in transit_features.values()
            )
            declares_recovery = any(
                value not in {"", None}
                for value in (*recovery_kind_values, *recovery_agent_values)
            )
            recovery_agent_id: str | None = None
            recovery_metadata_error: str | None = None
            if declares_recovery:
                unique_kinds = set(recovery_kind_values)
                unique_agent_ids = set(recovery_agent_values)
                if unique_kinds != {"collision_avoidance_recovery"}:
                    recovery_metadata_error = "inconsistent_recovery_kind"
                elif len(unique_agent_ids) != 1 or not isinstance(
                    recovery_agent_values[0], str
                ) or not recovery_agent_values[0]:
                    recovery_metadata_error = "inconsistent_recovery_agent_id"
                else:
                    recovery_agent_id = recovery_agent_values[0]
                    for agent_id, features in transit_features.items():
                        role = features.get("assignment_role")
                        hold_reason = features.get("hold_reason")
                        delay_s = features.get("traffic_reservation_delay_s", 0.0)
                        predecessor = features.get(
                            "traffic_reservation_predecessor_agent_id", ""
                        )
                        if (
                            not isinstance(delay_s, (int, float))
                            or isinstance(delay_s, bool)
                            or not math.isfinite(float(delay_s))
                            or abs(float(delay_s)) > 1.0e-9
                            or predecessor not in {"", None}
                        ):
                            recovery_metadata_error = "recovery_cannot_use_traffic_reservation"
                            break
                        if agent_id == recovery_agent_id:
                            if role != "backtrack" or hold_reason not in {"", None}:
                                recovery_metadata_error = "recovery_agent_must_be_backtrack"
                                break
                        elif role != "hold" or hold_reason != "collision_avoidance_recovery":
                            recovery_metadata_error = "recovery_nonmoving_agents_must_hold"
                            break
            recovery_metadata = {
                "declared": declares_recovery,
                "recovery_agent_id": recovery_agent_id,
                "metadata_valid": recovery_metadata_error is None,
                "metadata_error": recovery_metadata_error,
            }
            if recovery_metadata_error is not None:
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "collision_avoidance_recovery": recovery_metadata,
                        "reason": "malformed_collision_avoidance_recovery_metadata",
                    }
                )
                return "malformed_collision_avoidance_recovery_metadata"
            planned_team_diversity = audit_translation_invariant_team_trajectories(
                {fragment.agent_id: fragment.path for fragment in transit_fragments},
                roles_by_agent={
                    fragment.agent_id: str(
                        dict(fragment.type_signature.public_features).get(
                            "assignment_role", "explore"
                        )
                    )
                    for fragment in transit_fragments
                },
                scope="planned_guarded",
            )
            if planned_team_diversity.has_translated_duplicate:
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "team_trajectory_diversity": planned_team_diversity.to_dict(),
                        "reason": "translated_explorer_trajectory_copy",
                    }
                )
                return "translated_explorer_trajectory_copy"
            transit_routes = tuple(
                (
                    TimedStationary(
                        fragment.agent_id,
                        tuple(fragment.path[0]),
                        fragment.planned_start,
                        fragment.planned_end,
                    )
                    if dict(fragment.type_signature.public_features).get("assignment_role")
                    == "hold"
                    else TimedPolyline(
                        fragment.agent_id,
                        tuple(fragment.path),
                        fragment.planned_start,
                        fragment.planned_end,
                    )
                )
                for fragment in manifest.fragments
                if fragment.type_signature.fragment_type == "transit"
            )
            assessment = assess_synchronized_separation(
                transit_routes,
                minimum_separation_m=cf2x.PLANNED_INTER_AGENT_SEPARATION_M,
            )
            route_tube_assessment = assess_route_tube_separation(
                transit_routes,
                minimum_separation_m=cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M,
            )
            endpoint_assessment = assess_synchronized_separation(
                tuple(
                    TimedStationary(
                        agent_id,
                        endpoint_by_agent[agent_id],
                        decision_start_s,
                        max(fragment.planned_end for fragment in transit_fragments),
                    )
                    for agent_id in expected_agent_ids
                ),
                minimum_separation_m=cf2x.PLANNED_INTER_AGENT_ENDPOINT_SEPARATION_M,
            )
            transit_by_agent = {fragment.agent_id: fragment for fragment in transit_fragments}
            traffic_reservation_rows: list[dict[str, object]] = []
            unauthorized_route_tube_pairs: list[dict[str, object]] = []

            def _traffic_reservation_chain_reaches(
                delayed_agent_id: str, predecessor_agent_id: str
            ) -> tuple[bool, list[str]]:
                visited: set[str] = set()
                chain: list[str] = [delayed_agent_id]
                current = delayed_agent_id
                while current != predecessor_agent_id:
                    if current in visited:
                        return False, []
                    visited.add(current)
                    features = dict(
                        transit_by_agent[current].type_signature.public_features
                    )
                    direct_predecessor = str(
                        features.get("traffic_reservation_predecessor_agent_id", "")
                    )
                    delay_s = float(features.get("traffic_reservation_delay_s", 0.0))
                    if (
                        delay_s <= 0.0
                        or direct_predecessor not in transit_by_agent
                        or direct_predecessor == current
                    ):
                        return False, []
                    current_transit = transit_by_agent[current]
                    predecessor_transit = transit_by_agent[direct_predecessor]
                    if (
                        current_transit.planned_start + 1.0e-9
                        < predecessor_transit.planned_end
                        + cf2x.TRAFFIC_RESERVATION_MINIMUM_RELEASE_MARGIN_S
                    ):
                        return False, []
                    current = direct_predecessor
                    chain.append(current)
                return True, chain

            for left_index, left_route in enumerate(transit_routes):
                for right_route in transit_routes[left_index + 1 :]:
                    pair_assessment = assess_route_tube_separation(
                        (left_route, right_route),
                        minimum_separation_m=cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M,
                    )
                    if pair_assessment.admitted:
                        continue
                    left_fragment = transit_by_agent[left_route.agent_id]
                    right_fragment = transit_by_agent[right_route.agent_id]
                    left_features = dict(left_fragment.type_signature.public_features)
                    right_features = dict(right_fragment.type_signature.public_features)
                    serialized = False
                    serialization_chain: list[str] = []
                    for delayed_fragment, predecessor_fragment in (
                        (left_fragment, right_fragment),
                        (right_fragment, left_fragment),
                    ):
                        features = dict(delayed_fragment.type_signature.public_features)
                        delay_s = float(features.get("traffic_reservation_delay_s", 0.0))
                        predecessor_agent_id = str(
                            features.get("traffic_reservation_predecessor_agent_id", "")
                        )
                        if delay_s <= 0.0:
                            continue
                        declared_delay_matches_window = math.isclose(
                            delay_s,
                            delayed_fragment.planned_start - decision_start_s,
                            abs_tol=1.0e-9,
                        )
                        if not declared_delay_matches_window:
                            continue
                        if predecessor_agent_id == predecessor_fragment.agent_id:
                            scheduled_after_predecessor = (
                                delayed_fragment.planned_start + 1.0e-9
                                >= predecessor_fragment.planned_end
                                + cf2x.TRAFFIC_RESERVATION_MINIMUM_RELEASE_MARGIN_S
                            )
                            if not scheduled_after_predecessor:
                                continue
                            serialized = True
                            serialization_chain = [
                                delayed_fragment.agent_id,
                                predecessor_fragment.agent_id,
                            ]
                        else:
                            chain_ok, reached_chain = _traffic_reservation_chain_reaches(
                                delayed_fragment.agent_id,
                                predecessor_fragment.agent_id,
                            )
                            if not chain_ok:
                                continue
                            serialized = True
                            serialization_chain = reached_chain
                        traffic_reservation_rows.append(
                            {
                                "delayed_agent_id": delayed_fragment.agent_id,
                                "predecessor_agent_id": predecessor_fragment.agent_id,
                                "planned_delay_s": delay_s,
                                "predecessor_planned_end_s": predecessor_fragment.planned_end,
                                "delayed_planned_start_s": delayed_fragment.planned_start,
                                "serialization_chain": serialization_chain,
                                "pair_route_tube_assessment": pair_assessment.to_public_dict(),
                            }
                        )
                        break
                    if not serialized:
                        unauthorized_route_tube_pairs.append(
                            {
                                "agent_ids": [left_route.agent_id, right_route.agent_id],
                                "pair_route_tube_assessment": pair_assessment.to_public_dict(),
                                "left_reservation": {
                                    "delay_s": left_features.get(
                                        "traffic_reservation_delay_s", 0.0
                                    ),
                                    "predecessor_agent_id": left_features.get(
                                        "traffic_reservation_predecessor_agent_id", ""
                                    ),
                                },
                                "right_reservation": {
                                    "delay_s": right_features.get(
                                        "traffic_reservation_delay_s", 0.0
                                    ),
                                    "predecessor_agent_id": right_features.get(
                                        "traffic_reservation_predecessor_agent_id", ""
                                    ),
                                },
                            }
                        )
            if declares_recovery:
                assert recovery_agent_id is not None
                recovery_assessment = assess_collision_avoidance_recovery(
                    transit_routes,
                    recovery_agent_id=recovery_agent_id,
                    boundary_linear_speeds_mps={
                        f"uav{index}": boundary_linear_speeds_mps[index]
                        for index in range(fleet_size)
                    },
                    physical_minimum_separation_m=cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M,
                    planned_minimum_separation_m=cf2x.PLANNED_INTER_AGENT_SEPARATION_M,
                    recovery_endpoint_minimum_separation_m=(
                        cf2x.PLANNED_INTER_AGENT_ENDPOINT_SEPARATION_M
                    ),
                    boundary_speed_limit_mps=cf2x.WAYPOINT_SETTLE_SPEED_MPS,
                )
                recovery_record = {
                    **recovery_metadata,
                    "assessment": recovery_assessment.to_public_dict(),
                    "boundary_linear_speeds_mps": {
                        f"uav{index}": boundary_linear_speeds_mps[index]
                        for index in range(fleet_size)
                    },
                }
                if assessment.admitted:
                    records.append(
                        {
                            "candidate_id": manifest.candidate_id,
                            "candidate_manifest_sha256": manifest.manifest_hash,
                            "endpoint_by_agent": endpoint_by_agent,
                            "separation_assessment": assessment.to_public_dict(),
                            "route_tube_assessment": route_tube_assessment.to_public_dict(),
                            "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                            "traffic_reservation": traffic_reservation_rows,
                            "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                            "team_trajectory_diversity": planned_team_diversity.to_dict(),
                            "collision_avoidance_recovery": recovery_record,
                            "reason": "collision_avoidance_recovery_not_required",
                        }
                    )
                    return None
                if not recovery_assessment.admitted:
                    records.append(
                        {
                            "candidate_id": manifest.candidate_id,
                            "candidate_manifest_sha256": manifest.manifest_hash,
                            "endpoint_by_agent": endpoint_by_agent,
                            "separation_assessment": assessment.to_public_dict(),
                            "route_tube_assessment": route_tube_assessment.to_public_dict(),
                            "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                            "traffic_reservation": traffic_reservation_rows,
                            "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                            "team_trajectory_diversity": planned_team_diversity.to_dict(),
                            "collision_avoidance_recovery": recovery_record,
                            "reason": "collision_avoidance_recovery_rejected",
                        }
                    )
                    return "collision_avoidance_recovery_rejected"
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "final_relay_graph": None,
                        "communication_warning": "recovery_endpoint_not_used_for_relay_claim",
                        "separation_assessment": assessment.to_public_dict(),
                        "route_tube_assessment": route_tube_assessment.to_public_dict(),
                        "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                        "traffic_reservation": traffic_reservation_rows,
                        "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                        "team_trajectory_diversity": planned_team_diversity.to_dict(),
                        "collision_avoidance_recovery": recovery_record,
                        "reason": None,
                    }
                )
                return None
            if not assessment.admitted:
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "separation_assessment": assessment.to_public_dict(),
                        "route_tube_assessment": route_tube_assessment.to_public_dict(),
                        "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                        "traffic_reservation": traffic_reservation_rows,
                        "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                        "team_trajectory_diversity": planned_team_diversity.to_dict(),
                        "collision_avoidance_recovery": recovery_metadata,
                        "reason": "synchronized_fleet_separation",
                    }
                )
                return "synchronized_fleet_separation"
            if not endpoint_assessment.admitted:
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "separation_assessment": assessment.to_public_dict(),
                        "route_tube_assessment": route_tube_assessment.to_public_dict(),
                        "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                        "traffic_reservation": traffic_reservation_rows,
                        "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                        "team_trajectory_diversity": planned_team_diversity.to_dict(),
                        "collision_avoidance_recovery": recovery_metadata,
                        "reason": "planned_endpoint_separation_margin",
                    }
                )
                return "planned_endpoint_separation_margin"
            if unauthorized_route_tube_pairs:
                records.append(
                    {
                        "candidate_id": manifest.candidate_id,
                        "candidate_manifest_sha256": manifest.manifest_hash,
                        "endpoint_by_agent": endpoint_by_agent,
                        "separation_assessment": assessment.to_public_dict(),
                        "route_tube_assessment": route_tube_assessment.to_public_dict(),
                        "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                        "traffic_reservation": traffic_reservation_rows,
                        "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                        "team_trajectory_diversity": planned_team_diversity.to_dict(),
                        "collision_avoidance_recovery": recovery_metadata,
                        "reason": "route_tube_separation",
                    }
                )
                return "route_tube_separation"
            endpoint_graph = cf2x._initial_relay_graph(
                scene_query,
                tuple(endpoint_by_agent[f"uav{index}"] for index in range(fleet_size)),
            )
            records.append(
                {
                    "candidate_id": manifest.candidate_id,
                    "candidate_manifest_sha256": manifest.manifest_hash,
                    "endpoint_by_agent": endpoint_by_agent,
                    "final_relay_graph": endpoint_graph.to_dict(),
                    "communication_warning": (
                        None if endpoint_graph.fully_relay_connected else "final_relay_disconnected"
                    ),
                    "separation_assessment": assessment.to_public_dict(),
                    "route_tube_assessment": route_tube_assessment.to_public_dict(),
                    "endpoint_separation_assessment": endpoint_assessment.to_public_dict(),
                    "traffic_reservation": traffic_reservation_rows,
                    "unauthorized_route_tube_pairs": unauthorized_route_tube_pairs,
                    "team_trajectory_diversity": planned_team_diversity.to_dict(),
                    "collision_avoidance_recovery": recovery_metadata,
                    "reason": None,
                }
            )
            return None

        def joint_guard(
            manifest: CandidateFragmentManifest,
            *,
            timing: dict[str, float | int] = joint_guard_timing,
        ) -> str | None:
            joint_wall_started = time.perf_counter()
            timing["call_count"] += 1
            try:
                return _joint_guard(manifest)
            finally:
                timing["wall_s"] += time.perf_counter() - joint_wall_started

        candidate_pool_wall_started = time.perf_counter()
        try:
            pool = tuple(
                build_public_candidate_pool(
                    state,
                    route_guard,
                    candidate_limit=args.candidate_limit,
                    joint_guard=joint_guard,
                    minimum_feasible_candidates=(cf2x.P07_MINIMUM_EXECUTABLE_CANDIDATES),
                    minimum_multi_agent_route_candidates=2,
                    include_route_extreme=not args.disable_route_extreme,
                    require_joint_prefilter=bool(
                        args.p0_start_eligibility_audit
                        or args.p0_start_eligibility_evidence is not None
                    ),
                )
            )
        except ValueError as error:
            public_frontier_debug = [frontier.to_dict() for frontier in state.frontiers]
            first_hits = [
                record["first_blocking_hit"]
                for record in route_guard_records
                if record["first_blocking_hit"] is not None
            ]
            hit_classes = Counter(str(hit["hit_class"]) for hit in first_hits)
            hit_prims = Counter(str(hit["hit_prim_path"]) for hit in first_hits)
            raise CandidateRouteGuardError(
                str(error),
                {
                    "schema_version": "hm3d-p07-route-guard-failure-v1",
                    "decision_index": decision_index,
                    "public_frontiers": public_frontier_debug,
                    "route_request_count": len(route_guard_records),
                    "unique_route_count": len(route_cache),
                    "joint_guard_records": joint_guard_records,
                    "completed_decision_count": len(decisions),
                    "completed_decisions": decisions,
                    "completed_metric_samples": [
                        {
                            "timestamp_s": sample.timestamp_s,
                            "explored_free_volume_m3": sample.explored_free_volume_m3,
                            "true_free_volume_m3": sample.true_free_volume_m3,
                            "predicted_free_volume_m3": sample.predicted_free_volume_m3,
                            "hallucinated_free_volume_m3": sample.hallucinated_free_volume_m3,
                            "coverage_fraction": sample.coverage_fraction,
                        }
                        for sample in samples
                    ],
                    "elapsed_physics_s": elapsed_s,
                    "current_positions_m": current_positions,
                    "completed_decision_wall_rows": decision_wall_rows,
                    "failed_stage_wall_timing": {
                        "initial_communication_graph_wall_s": initial_graph_wall_s,
                        "frontier_extraction_wall_s": frontier_wall_s,
                        "candidate_pool_wall_s": time.perf_counter() - candidate_pool_wall_started,
                        "route_guard_unique_wall_s": route_guard_timing["unique_wall_s"],
                        "joint_guard_wall_s": joint_guard_timing["wall_s"],
                        "joint_guard_call_count": joint_guard_timing["call_count"],
                    },
                    "public_free_reachability_cache": public_reachability_cache.audit(),
                    "first_hit_class_counts": dict(sorted(hit_classes.items())),
                    "first_hit_prim_path_counts": dict(sorted(hit_prims.items())),
                    "route_records": route_guard_records,
                },
            ) from error
        candidate_pool_wall_s = time.perf_counter() - candidate_pool_wall_started
        pool_digest = public_candidate_pool_hash(pool)
        candidate_route_catalog = _candidate_route_opportunity_catalog(
            state,
            route_guard_records,
            pool,
        )
        pool_hashes.append(pool_digest)
        candidate_intent_audit = audit_public_candidate_intent_richness(pool)
        candidate_intent_audits.append(
            {
                "decision_id": decision_context.decision_id,
                "public_candidate_pool_hash": pool_digest,
                **candidate_intent_audit.to_dict(),
            }
        )
        value_protected_diversity_audit = audit_value_protected_candidate_diversity(
            pool,
            utility_slack=QD_UTILITY_SLACK,
        )
        value_protected_candidate_diversity_audits.append(
            {
                "decision_id": decision_context.decision_id,
                "public_candidate_pool_hash": pool_digest,
                **value_protected_diversity_audit.to_dict(),
            }
        )
        if args.p0_start_eligibility_audit:
            candidate_roles = _candidate_role_summary(pool, selected_manifest_hash="")
            all_agents_active_candidate_count = sum(
                row["moving_explorer_count"] == fleet_size
                for row in candidate_roles
            )
            feasible_all_agents_active_candidate_count = (
                _feasible_all_active_candidate_count(
                    candidate_roles,
                    fleet_size=fleet_size,
                )
            )
            all_agents_active_candidate_exists = any(
                row["feasible"] is True
                and row["moving_explorer_count"] == fleet_size
                for row in candidate_roles
            )
            audit_payload = {
                "schema_version": "hm3d-p07-start-eligibility-audit-v1",
                "status": "P07_START_ELIGIBILITY_AUDIT_COMPLETE",
                "synthetic": False,
                "formal_result": False,
                "trainable": False,
                "claim_limit": (
                    "P0 engineering eligibility evidence only. It executes the shared "
                    "PhysX bootstrap, public sparse-map fusion, public route search and "
                    "static guard for one pre-declared start cluster. It has no selected "
                    "exploration action, reward, coverage score, RL transition or QD update."
                ),
                "scene_id": args.scene_id,
                "selection_partition": args.split,
                "strategy": args.strategy,
                "controller_id": args.controller_id,
                "action_budget_s": args.action_budget_s,
                "candidate_limit": args.candidate_limit,
                "p0_eligibility_contract": p0_eligibility_contract,
                **public_schema_fields(),
                "initial_start_reset": initial_start_reset_witness,
                "bootstrap": {
                    "elapsed_physics_s": bootstrap_elapsed_s,
                    "execution": bootstrap_ledger.to_public_dict(),
                    "public_observation_frame_count": len(bootstrap_backend.public_range_frames),
                    "public_range_ray_count": len(latest_public_outcomes),
                },
                "bootstrap_communication_contract_audit": bootstrap_communication_audit,
                "first_pool": {
                    "decision_duration_s": decision_duration_s,
                    "effective_frontier_step_m": effective_frontier_step_m,
                    "public_frontier_count": len(public_frontiers),
                    "public_observation_viewpoint_count": sum(
                        frontier.task_kind == "explore"
                        and frontier.viewpoint_kind == "observation"
                        for frontier in public_frontiers
                    ),
                    "public_route_progress_fallback_count": sum(
                        frontier.task_kind == "explore"
                        and frontier.viewpoint_kind == "route_progress"
                        for frontier in public_frontiers
                    ),
                    "public_region_access_count": sum(
                        frontier.task_kind == "explore"
                        and frontier.viewpoint_kind == "region_access"
                        for frontier in public_frontiers
                    ),
                    "public_candidate_pool_hash": pool_digest,
                    "clearance_oracle": clearance_oracle.to_public_dict(),
                    "wall_timing": {
                        "initial_communication_graph_wall_s": initial_graph_wall_s,
                        "frontier_extraction_wall_s": frontier_wall_s,
                        "candidate_pool_wall_s": candidate_pool_wall_s,
                        "route_guard_unique_wall_s": route_guard_timing["unique_wall_s"],
                        "joint_guard_wall_s": joint_guard_timing["wall_s"],
                        "joint_guard_call_count": joint_guard_timing["call_count"],
                    },
                    "public_free_reachability_cache": public_reachability_cache.audit(),
                    "per_agent_edge_diagnostics": _per_agent_candidate_edge_diagnostics(
                        state,
                        team_belief,
                        route_guard_records,
                    ),
                    "candidate_route_opportunity_catalog": candidate_route_catalog,
                    "candidate_roles": candidate_roles,
                    "all_agents_active_candidate_count": all_agents_active_candidate_count,
                    "feasible_all_agents_active_candidate_count": (
                        feasible_all_agents_active_candidate_count
                    ),
                    "all_agents_active_candidate_exists": all_agents_active_candidate_exists,
                    "joint_guard_reason_counts": dict(
                        sorted(
                            Counter(
                                str(record.get("reason") or "admitted")
                                for record in joint_guard_records
                            ).items()
                        )
                    ),
                    "candidate_intent_audit": candidate_intent_audit.to_dict(),
                    "value_protected_diversity_audit": value_protected_diversity_audit.to_dict(),
                },
                "route_guard_records": route_guard_records,
                "joint_guard_records": joint_guard_records,
                "transit_time_model_sha256": _sha256(paths["timing"]),
                "start_reset_manifest_sha256": _sha256(paths["start_resets"]),
                "collision_usd_sha256": _sha256(paths["collision"]),
                "cf2x_usd_sha256": _sha256(paths["cf2x"]),
                "random_key": args.random_key,
            }
            audit_payload["audit_record_sha256"] = canonical_sha256(audit_payload)
            _write_new(paths["output"], audit_payload)
            print(
                json.dumps(
                    {
                        "status": audit_payload["status"],
                        "output": str(paths["output"]),
                        "all_agents_active_candidate_exists": all_agents_active_candidate_exists,
                    },
                    sort_keys=True,
                )
            )
            return 0
        qd_intent_fallback: dict[str, object] | None = None
        if (
            qd_strategy or qd_calibration
        ) and candidate_intent_audit.status != "QD_CANDIDATE_INTENT_ADMITTED":
            # The candidate-intent richness floor is an availability check for
            # a behaviour repertoire.  In a constrained decision the shared
            # pool may be physically unable to cover six joint intent cells
            # (e.g. every agent's feasible endpoints collapse onto one
            # dispersion bin).  Hard-failing the whole episode there would
            # discard every earlier receipt and make the QD mechanism
            # unusable on narrow scenes.  Fall back to the transparent public
            # value selector for this decision, record the rejection, and
            # continue collecting real receipts.  This never relaxes safety,
            # timing, separation or diversity contracts; it only changes which
            # selector ranks the same admitted pool.
            qd_intent_fallback = {
                "status": "QD_INTENT_FALLBACK_TO_PUBLIC_VALUE",
                "candidate_intent_audit": candidate_intent_audit.to_dict(),
                "decision_id": decision_context.decision_id,
            }
            public_exploration_need = public_exploration_need_from_public_belief(
                team_belief,
                agent_free_voxel_keys={
                    agent_id: belief.free_keys() for agent_id, belief in agent_beliefs.items()
                },
                agent_ids=tuple(agent_beliefs),
                spatial_reference_m=float(communication_contract.network["maximum_range_m"]),
                height_band_m=1.0,
            )
            planning_wall_s = time.perf_counter() - planning_wall_started
            selection_wall_started = time.perf_counter()
            selected, selection = _select(
                "frontier_3d",
                state,
                pool,
                random_key=args.random_key + decision_index,
                single_rl_checkpoint=None,
                marl_ipp_checkpoint=None,
                marl_ipp_source_root=args.marl_ipp_source_root.expanduser().resolve(),
                split_manifest_sha256=split_manifest_sha256,
                planned_qd_selector=None,
                realised_qd_selector=None,
                public_exploration_need=public_exploration_need,
                qd_calibration_mode=None,
            )
            selection = {
                **selection,
                "selected_predicted_descriptor": list(selected.planned_descriptor),
                "selected_need_alignment": public_exploration_need.alignment(
                    list(selected.planned_descriptor)
                ),
                "qd_intent_fallback": qd_intent_fallback,
            }
        else:
            public_exploration_need = public_exploration_need_from_public_belief(
                team_belief,
                agent_free_voxel_keys={
                    agent_id: belief.free_keys() for agent_id, belief in agent_beliefs.items()
                },
                agent_ids=tuple(agent_beliefs),
                spatial_reference_m=float(communication_contract.network["maximum_range_m"]),
                height_band_m=1.0,
            )
            planning_wall_s = time.perf_counter() - planning_wall_started
            selection_wall_started = time.perf_counter()
            selected, selection = _select(
                args.strategy,
                state,
                pool,
                random_key=args.random_key + decision_index,
                single_rl_checkpoint=(
                    None if args.single_rl_checkpoint is None else args.single_rl_checkpoint.resolve()
                ),
                marl_ipp_checkpoint=(
                    None if args.marl_ipp_checkpoint is None else args.marl_ipp_checkpoint.resolve()
                ),
                marl_ipp_source_root=args.marl_ipp_source_root.expanduser().resolve(),
                split_manifest_sha256=split_manifest_sha256,
                planned_qd_selector=planned_qd_selector,
                realised_qd_selector=realised_qd_selector,
                public_exploration_need=public_exploration_need,
                qd_calibration_mode=args.qd_calibration_mode,
            )
        candidate_route_catalog = _candidate_route_opportunity_catalog(
            state,
            route_guard_records,
            pool,
            selected=selected,
        )
        selected_joint_guard_record = next(
            (
                record
                for record in reversed(joint_guard_records)
                if record.get("candidate_manifest_sha256") == selected.manifest_hash
                and record.get("reason")
                in (None, "collision_avoidance_recovery_not_required")
            ),
            None,
        )
        if selected_joint_guard_record is None:
            raise RuntimeError(
                "selected candidate is missing its admitted joint-route safety certificate"
            )
        selected_recovery_metadata = selected_joint_guard_record.get(
            "collision_avoidance_recovery"
        )
        if not isinstance(selected_recovery_metadata, dict):
            raise RuntimeError("selected joint-route certificate omits recovery metadata")
        selected_is_collision_avoidance_recovery = bool(
            selected_recovery_metadata.get("declared")
        )
        selected_recovery_agent_id = (
            selected_recovery_metadata.get("recovery_agent_id")
            if selected_is_collision_avoidance_recovery
            else None
        )
        if selected_is_collision_avoidance_recovery and not isinstance(
            selected_recovery_agent_id, str
        ):
            raise RuntimeError("selected recovery omits its unique recovery agent")
        selected_joint_safety = {
            "synchronized": selected_joint_guard_record["separation_assessment"],
            "route_tube": selected_joint_guard_record["route_tube_assessment"],
            "endpoint": selected_joint_guard_record["endpoint_separation_assessment"],
            "collision_avoidance_recovery": selected_recovery_metadata,
        }
        selected_exploration_target_keys = _selected_exploration_target_keys(
            selected, team_belief
        )
        selection_wall_s = time.perf_counter() - selection_wall_started
        token = authorize_manifest(
            decision_context,
            pool,
            tuple(manifest.feasible for manifest in pool),
            tuple(manifest.manifest_hash for manifest in pool).index(selected.manifest_hash),
            token_id=f"p07-online-token-{uuid.uuid4().hex}",
            issued_at=0.0,
            duration=state.decision_duration_s,
        )
        if timeout_probe and args.calibration_timeout_probe_s >= state.decision_duration_s:
            raise ValueError(
                "calibration timeout deadline must be shorter than the decision token duration"
            )
        execution_wall_started = time.perf_counter()
        backend = new_backend(execution_deadline_s=args.calibration_timeout_probe_s)
        metric_sample_before = samples[-1]
        ledger = execute_hm3d_manifest(
            selected,
            token,
            backend,
            time_tolerance_s=args.outcome_time_tolerance_s,
            command_path_tolerance_m=0.25,
            replay_exclusion_reason=(
                "COLLISION_AVOIDANCE_RECOVERY"
                if selected_is_collision_avoidance_recovery
                else None
            ),
        )
        execution_wall_s = time.perf_counter() - execution_wall_started
        post_execution_wall_started = time.perf_counter()
        execution_calibration = _decision_execution_calibration_summary(
            backend, decision_id=decision_context.decision_id
        )
        recovery_execution_audit: dict[str, object] | None = None
        recovery_execution_contract_passed = True
        if selected_is_collision_avoidance_recovery:
            if len(backend.final_root_positions_m) != fleet_size:
                raise RuntimeError("CF2X recovery omitted final physical positions")
            recovery_final_positions = backend.final_root_positions_m
            recovery_final_speeds = _final_boundary_linear_speeds(
                backend, stage_id=decision_context.decision_id
            )
            recovery_actual_minimum_separation_m = min(
                math.dist(left, right)
                for index, left in enumerate(recovery_final_positions)
                for right in recovery_final_positions[index + 1 :]
            )
            recovery_execution_contract_passed = (
                recovery_actual_minimum_separation_m + 1.0e-9
                >= cf2x.PLANNED_INTER_AGENT_SEPARATION_M
            )
            recovery_execution_audit = {
                "recovery_agent_id": selected_recovery_agent_id,
                "entry_positions_m": [list(point) for point in current_positions],
                "entry_boundary_linear_speeds_mps": {
                    f"uav{index}": current_boundary_linear_speeds_mps[index]
                    for index in range(fleet_size)
                },
                "final_positions_m": [list(point) for point in recovery_final_positions],
                "final_boundary_linear_speeds_mps": {
                    f"uav{index}": recovery_final_speeds[index]
                    for index in range(fleet_size)
                },
                "actual_final_minimum_separation_m": recovery_actual_minimum_separation_m,
                "physical_minimum_separation_m": cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M,
                "restored_planning_separation_m": cf2x.PLANNED_INTER_AGENT_SEPARATION_M,
                "actual_physical_separation_passed": (
                    recovery_actual_minimum_separation_m + 1.0e-9
                    >= cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M
                ),
                "actual_planning_margin_restored": recovery_execution_contract_passed,
            }
        stationarity_supervision = _decision_stationarity_supervision(
            selected, execution_calibration
        )
        realised_team_diversity = backend.engineering_diagnostics.get("team_trajectory_diversity")
        if not isinstance(realised_team_diversity, dict):
            raise RuntimeError("CF2X backend omitted realised team-trajectory diversity")
        duplicate_pair_count = realised_team_diversity.get("duplicate_pair_count")
        moving_explorer_count = realised_team_diversity.get("moving_explorer_count")
        if (
            not isinstance(duplicate_pair_count, int)
            or isinstance(duplicate_pair_count, bool)
            or duplicate_pair_count < 0
            or not isinstance(moving_explorer_count, int)
            or isinstance(moving_explorer_count, bool)
            or moving_explorer_count < 0
        ):
            raise RuntimeError("CF2X team-trajectory diversity denominators are malformed")
        decision_team_diversity = {
            "decision_id": decision_context.decision_id,
            **realised_team_diversity,
        }
        team_trajectory_diversity_audits.append(decision_team_diversity)
        communication = backend.engineering_diagnostics.get("communication")
        delivery = backend.engineering_diagnostics.get("message_delivery")
        if not isinstance(communication, dict) or not isinstance(delivery, dict):
            backend_error = backend.engineering_diagnostics.get("backend_exception")
            raise RuntimeError(
                "CF2X backend omitted communication denominators: "
                f"{backend_error if backend_error is not None else 'no backend exception recorded'}"
            )
        round_communication_audit = communication_contract.audit_worker_evidence(
            communication, delivery
        )
        if round_communication_audit["passed"] is not True:
            raise RuntimeError(
                "decision communication violates the frozen contract: "
                f"{json.dumps(round_communication_audit, sort_keys=True)}"
            )
        round_start_s = elapsed_s
        public_free_before = team_belief.observed_free_count
        public_free_keys_before = frozenset(team_belief.free_keys())
        latest_public_outcomes = absorb_outcomes(
            backend,
            stage_id=f"decision{decision_index}",
            timestamp_offset_s=round_start_s,
            communication_audit=round_communication_audit,
            integrate_into_belief=not selected_is_collision_avoidance_recovery,
        )
        applied_paths: dict[str, list[tuple[float, float, float]]] = {
            f"uav{index}": [current_positions[index]] for index in range(fleet_size)
        }
        for outcome in ledger.outcomes:
            applied = outcome.applied_fragment
            if not outcome.executed or applied is None:
                continue
            applied_paths[outcome.agent_id].extend(tuple(point) for point in applied.path)
        candidate_descriptor_features = outcome_qd_feature_vector_from_public_outcomes(
            scene_id=args.scene_id,
            agent_ids=tuple(applied_paths),
            applied_paths_by_agent=applied_paths,
            range_outcomes=latest_public_outcomes,
            resolution_m=0.25,
            spatial_reference_m=float(communication_contract.network["maximum_range_m"]),
        )
        # v4 remains the currently proposed family.  The full pre-registered
        # feature vector is also bound into each outcome so train-only
        # calibration can reject v4 if a less redundant three-axis family is
        # stronger.  Validation never chooses this family.
        descriptor = realised_descriptor_from_public_outcomes(
            scene_id=args.scene_id,
            agent_ids=tuple(applied_paths),
            applied_paths_by_agent=applied_paths,
            range_outcomes=latest_public_outcomes,
            resolution_m=0.25,
            spatial_reference_m=float(communication_contract.network["maximum_range_m"]),
        )
        workload_balance = public_observation_workload_balance_from_range_outcomes(
            scene_id=args.scene_id,
            agent_ids=tuple(applied_paths),
            range_outcomes=latest_public_outcomes,
            resolution_m=0.25,
        )
        # FREE is intentionally revisable: a later occupied observation wins
        # over an earlier free ray.  Compare the fused public maps at the
        # decision boundary so raw segment rays cannot manufacture QD quality.
        public_new_free_footprint, public_revised_free_footprint = public_free_voxel_transition(
            public_free_keys_before, team_belief.free_keys()
        )
        public_free_delta_m3 = len(public_new_free_footprint) * 0.25**3
        if selected_is_collision_avoidance_recovery and (
            public_new_free_footprint or public_revised_free_footprint or public_free_delta_m3
        ):
            raise RuntimeError(
                "collision-avoidance recovery changed the public exploration belief"
            )
        if team_belief.observed_free_count - public_free_before != (
            len(public_new_free_footprint) - len(public_revised_free_footprint)
        ):
            raise RuntimeError("public free-voxel delta is inconsistent with its decision snapshot")
        if selected_is_collision_avoidance_recovery:
            cooldown_update = {
                "source": "collision_avoidance_recovery",
                "public_new_free_voxel_count": 0,
                "selected_exploration_target_voxel_keys": [],
                "applied": False,
                "exclusion_reason": "COLLISION_AVOIDANCE_RECOVERY",
            }
        else:
            cooldown_update = observation_cooldown.observe_empty_targets(
                selected_exploration_target_keys,
                decision_index=decision_index,
                public_new_free_voxel_count=len(public_new_free_footprint),
            )
        cooldown_audit = observation_cooldown.audit(decision_index=decision_index)
        execution_complete = not (
            ledger.failed_fragment_count
            or ledger.collision_count
            or ledger.inter_agent_separation_violation_count
            or ledger.out_of_bounds_count
            or ledger.static_clearance_contract_violation_count
        )
        execution_complete = execution_complete and recovery_execution_contract_passed
        selected_transits = {
            fragment.agent_id: fragment
            for fragment in selected.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        agent_execution = {
            str(row["agent_id"]): row
            for row in execution_calibration["agents"]
            if isinstance(row, dict) and isinstance(row.get("agent_id"), str)
        }
        transit_outcomes = {
            outcome.agent_id: outcome
            for outcome in ledger.outcomes
            if (
                outcome.executed
                and outcome.applied_fragment is not None
                and outcome.applied_fragment.type_signature.fragment_type == "transit"
            )
        }
        selected_backtrack_agent_ids: list[str] = []
        backtrack_outcomes: list[dict[str, object]] = []
        task_reservation_outcomes: list[dict[str, object]] = []
        for agent_id, fragment in selected_transits.items():
            features = dict(fragment.type_signature.public_features)
            role = features.get("assignment_role")
            viewpoint_kind = features.get("viewpoint_kind")
            agent_row = agent_execution.get(agent_id)
            transit_outcome = transit_outcomes.get(agent_id)
            if not isinstance(agent_row, dict):
                raise RuntimeError("selected transit is missing per-agent execution evidence")
            transit_safe_and_complete = (
                agent_row.get("transit_completed") is True
                and agent_row.get("transit_collision") is not True
                and agent_row.get("transit_out_of_bounds") is not True
                and agent_row.get("static_clearance_contract_violation") is not True
                and not agent_row.get("transit_failure_reason")
            )
            previous_reservation = task_reservations_by_agent.get(agent_id)
            reservation_outcome: dict[str, object] = {
                "agent_id": agent_id,
                "previous_task_reservation_source_decision_id": (
                    None
                    if previous_reservation is None
                    else previous_reservation.source_decision_id
                ),
                "selected_assignment_role": role,
                "selected_viewpoint_kind": viewpoint_kind,
                "selected_task_reservation_matched": features.get(
                    "task_reservation_matched", False
                ),
                "selected_task_reservation_heading_alignment": features.get(
                    "task_reservation_heading_alignment", 0.0
                ),
                "selected_task_reservation_switch_cost": features.get(
                    "task_reservation_switch_cost", 0.0
                ),
                "transit_safe_and_complete": transit_safe_and_complete,
            }
            if (
                execution_complete
                and not selected_is_collision_avoidance_recovery
                and role == "explore"
                and viewpoint_kind in {
                    "observation",
                    "route_progress",
                    "region_access",
                }
                and transit_safe_and_complete
                and transit_outcome is not None
            ):
                if not is_non_alias_exploration_path(fragment.path):
                    raise RuntimeError("completed exploration transit aliases its settled endpoint")
                frontier_id = features.get("frontier_id")
                source_frontier = next(
                    (
                        frontier
                        for frontier in state.frontiers
                        if frontier.frontier_id == frontier_id
                    ),
                    None,
                )
                if source_frontier is None:
                    raise RuntimeError("completed exploration transit omits its current public frontier")
                reservation = PublicTaskReservation.from_completed_public_exploration_transit(
                    agent_id=agent_id,
                    source_decision_id=decision_context.decision_id,
                    source_manifest_hash=selected.manifest_hash,
                    source_transit_outcome_sha256=transit_outcome.digest,
                    public_path_m=fragment.path,
                    task_anchor_m=source_frontier.task_anchor_m,
                    task_normal_unit=source_frontier.task_normal_unit,
                    source_frontier_cluster_id=source_frontier.frontier_cluster_id,
                    source_viewpoint_kind=str(viewpoint_kind),
                )
                task_reservations_by_agent[agent_id] = reservation
                reservation_outcome.update(
                    {
                        "action": "created_or_replaced_after_completed_public_exploration",
                        "new_task_reservation": reservation.to_dict(),
                    }
                )
            elif (
                role == "hold"
                and not selected_is_collision_avoidance_recovery
                and execution_complete
                and transit_safe_and_complete
                and previous_reservation is not None
            ):
                reservation_outcome.update(
                    {
                        "action": "retained_during_auditable_hold",
                        "retained_task_reservation": previous_reservation.to_dict(),
                    }
                )
            else:
                removed = task_reservations_by_agent.pop(agent_id, None)
                reservation_outcome.update(
                    {
                        "action": "cleared",
                        "clear_reason": (
                            "collision_avoidance_recovery"
                            if selected_is_collision_avoidance_recovery
                            else (
                                "execution_or_safety_failure"
                                if not execution_complete or not transit_safe_and_complete
                                else (
                                    "selected_outcome_backtrack"
                                    if role == "backtrack"
                                    else (
                                        "selected_hold"
                                        if role == "hold"
                                        else "selected_nonexploration_route"
                                    )
                                )
                            )
                        ),
                        "cleared_existing_commitment": removed is not None,
                    }
                )
            task_reservation_outcomes.append(reservation_outcome)
            if role == "backtrack":
                selected_backtrack_agent_ids.append(agent_id)
                recovery = outcome_backtrack_routes.get((agent_id, tuple(fragment.path[-1])))
                if recovery is None and selected_is_collision_avoidance_recovery:
                    backtrack_outcomes.append(
                        {
                            "agent_id": agent_id,
                            "route_id": None,
                            "source_decision_id": None,
                            "selected": True,
                            "completed_safely": transit_safe_and_complete,
                            "consumed": False,
                            "authority": "collision_avoidance_geometric_recovery",
                        }
                    )
                    continue
                if recovery is None:
                    raise RuntimeError("selected outcome backtrack omits its route authority")
                source_route, _ = recovery
                consumed = False
                if transit_safe_and_complete:
                    history = backtrack_history_by_agent[agent_id]
                    if not history or history[-1].route_id != source_route.route_id:
                        raise RuntimeError("selected outcome backtrack is not the current owned history")
                    history.pop()
                    consumed = True
                backtrack_outcomes.append(
                    {
                        "agent_id": agent_id,
                        "route_id": source_route.route_id,
                        "source_decision_id": source_route.source_decision_id,
                        "selected": True,
                        "completed_safely": transit_safe_and_complete,
                        "consumed": consumed,
                    }
                )
                continue
            if role != "explore" or not transit_safe_and_complete or transit_outcome is None:
                continue
            if _path_length_m(fragment.path) + 1.0e-9 < MINIMUM_MEANINGFUL_EXPLORATION_PATH_M:
                continue
            source_path = tuple(tuple(point) for point in transit_outcome.applied_fragment.path)
            if _path_length_m(source_path) + 1.0e-9 < MINIMUM_MEANINGFUL_EXPLORATION_PATH_M:
                raise RuntimeError("completed exploration outcome violates the shared movement floor")
            source_minimum_clearance = agent_row.get("minimum_static_mesh_clearance_m")
            source_required_clearance = agent_row.get("static_clearance_contract_required_m")
            if (
                not isinstance(source_minimum_clearance, (int, float))
                or isinstance(source_minimum_clearance, bool)
                or not isinstance(source_required_clearance, (int, float))
                or isinstance(source_required_clearance, bool)
                or not math.isfinite(float(source_minimum_clearance))
                or not math.isfinite(float(source_required_clearance))
                or float(source_required_clearance) <= 0.0
                or float(source_minimum_clearance) + 1.0e-9 < float(source_required_clearance)
            ):
                # A completed path without a measured static-clearance margin
                # remains a valid exploration outcome, but cannot authorize a
                # future source-clearance-backed recovery.
                continue
            backtrack_history_by_agent[agent_id] = [
                _OutcomeBacktrackRoute(
                    route_id=(
                        f"{decision_context.decision_id}-{agent_id}-"
                        f"{transit_outcome.digest[:12]}"
                    ),
                    agent_id=agent_id,
                    source_decision_id=decision_context.decision_id,
                    source_manifest_hash=selected.manifest_hash,
                    source_transit_outcome_sha256=transit_outcome.digest,
                    source_minimum_static_mesh_clearance_m=float(source_minimum_clearance),
                    source_static_clearance_contract_required_m=float(source_required_clearance),
                    path_m=source_path,
                )
            ]
        task_reservation_audit.append(
            {
                "schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
                "decision_id": decision_context.decision_id,
                "current_public_revalidation": predecision_task_reservation_outcomes,
                "reservations_before": [
                    reservation.to_dict() for reservation in state.task_reservations
                ],
                "outcomes": task_reservation_outcomes,
                "reservations_after": [
                    reservation.to_dict()
                    for reservation in sorted(
                        task_reservations_by_agent.values(), key=lambda item: item.agent_id
                    )
                ],
                "terminal_margin_distance_audit": {
                    "value_m": terminal_margin_distance_audit_m,
                    "method": "maximum_rest_to_rest_distance_m",
                    "input_duration_s": transit_timing.terminal_tracking_margin_s,
                    "cruise_speed_mps": transit_timing.cruise_speed_mps,
                    "max_accel_mps2": transit_timing.max_accel_mps2,
                    "audit_only": True,
                    "not_used_for": [
                        "candidate_legality",
                        "route_length_target",
                        "reward",
                        "controller_command",
                    ],
                },
            }
        )
        outcome_backtrack_audit.append(
            {
                "decision_id": decision_context.decision_id,
                "offers": outcome_backtrack_offers,
                "selected_agent_ids": selected_backtrack_agent_ids,
                "outcomes": backtrack_outcomes,
                "retained_route_ids_by_agent": {
                    agent_id: [route.route_id for route in history]
                    for agent_id, history in sorted(backtrack_history_by_agent.items())
                },
            }
        )
        qd_feasible = (
            not timeout_probe
            and execution_complete
            and not selected_is_collision_avoidance_recovery
            and not selected_backtrack_agent_ids
            and stationarity_supervision["status"] == "STATIONARITY_SUPERVISION_ADMITTED"
            and bool(public_new_free_footprint)
            and duplicate_pair_count == 0
        )
        if selected_is_collision_avoidance_recovery:
            qd_exclusion_reasons = ["COLLISION_AVOIDANCE_RECOVERY"]
        elif timeout_probe:
            qd_exclusion_reasons = ["CALIBRATION_TIMEOUT_PROBE"]
        elif selected_backtrack_agent_ids:
            qd_exclusion_reasons = ["OUTCOME_BACKTRACK_RECOVERY"]
        elif qd_feasible or not execution_complete:
            qd_exclusion_reasons = []
        elif stationarity_supervision["status"] != "STATIONARITY_SUPERVISION_ADMITTED":
            qd_exclusion_reasons = ["NON_MEANINGFUL_EXECUTION"]
        elif duplicate_pair_count:
            qd_exclusion_reasons = ["TRANSLATED_EXPLORER_TRAJECTORY_COPY"]
        else:
            qd_exclusion_reasons = ["NO_NEW_PUBLIC_FREE_VOXELS"]
        behavior_hash = canonical_sha256(
            {
                "selected_manifest_hash": selected.manifest_hash,
                "outcome_hashes": [outcome.digest for outcome in ledger.outcomes],
                "descriptor": descriptor.to_dict(),
                "candidate_descriptor_features": candidate_descriptor_features.to_dict(),
                "public_new_free_footprint": sorted(public_new_free_footprint),
            }
        )
        if args.strategy == "realised_qd":
            # This is evaluated after the flight, against the exact public
            # deficit available before selection.  It proves that an online
            # QD intervention did not merely predict a useful mode and then
            # execute an unrelated one under the CF2X/PhysX safety chain.
            predicted_descriptor = selection.get("selected_predicted_descriptor")
            if not isinstance(predicted_descriptor, list) or len(predicted_descriptor) != 3:
                raise RuntimeError("realised-QD selection omitted its predicted descriptor")
            predicted_alignment = public_exploration_need.alignment(predicted_descriptor)
            selected_alignment = selection.get("selected_need_alignment")
            if not isinstance(selected_alignment, (int, float)) or isinstance(
                selected_alignment, bool
            ):
                raise RuntimeError("realised-QD selection omitted its predicted need alignment")
            if abs(float(selected_alignment) - predicted_alignment) > 1.0e-9:
                raise RuntimeError("realised-QD predicted need alignment is inconsistent")
            realised_alignment = public_exploration_need.alignment(descriptor.values)
            selection["realised_descriptor"] = descriptor.to_dict()
            selection["realised_need_alignment"] = realised_alignment
            selection["need_alignment_prediction_error"] = abs(
                realised_alignment - predicted_alignment
            )
        if selected_is_collision_avoidance_recovery:
            # A recovery outcome is physically important, but it is not a
            # realised exploration mode.  In particular, do not update the
            # selector's intent predictor or archive from an escape motion.
            admission = AdmissionDecision(
                False,
                "COLLISION_AVOIDANCE_RECOVERY",
                None,
                None,
                realised_qd_archive.revision,
            )
        elif realised_qd_selector is not None:
            # The realised-QD selector owns the online archive update.  Keeping
            # this in one method prevents a predictor-only history from being
            # mistaken for a populated outcome-grounded repertoire.
            admission = realised_qd_selector.observe(
                selected,
                descriptor,
                public_quality=public_free_delta_m3,
                public_cost=ledger.total_energy_used_j,
                execution_outcome_sha256=behavior_hash,
                execution_feasible=qd_feasible,
            )
        else:
            # Non-QD branches still emit comparable outcome diagnostics, but
            # they never consult this archive to choose an action.
            admission = realised_qd_archive.add_or_update(
                Elite(
                    candidate_id=selected.candidate_id,
                    manifest_hash=selected.manifest_hash,
                    behavior_hash=behavior_hash,
                    realised_descriptor=descriptor.values,
                    quality=public_free_delta_m3,
                    cost=ledger.total_energy_used_j,
                    feasible=qd_feasible,
                    source=HM3D_REALISED_QD_SCHEMA_VERSION,
                )
            )
        realised_qd_record = {
            "candidate_id": selected.candidate_id,
            "candidate_manifest_sha256": selected.manifest_hash,
            "execution_outcome_sha256": behavior_hash,
            "executed": execution_complete,
            "public_candidate_intent": list(selected.planned_descriptor),
            "descriptor": descriptor.to_dict(),
            "candidate_descriptor_features": candidate_descriptor_features.to_dict(),
            "process_diagnostics": {
                "public_observation_workload_balance": workload_balance,
                "note": "diagnostic_only_not_a_qd_archive_axis",
            },
            "public_new_free_voxel_keys": [list(key) for key in sorted(public_new_free_footprint)],
            "public_revised_free_voxel_keys": [
                list(key) for key in sorted(public_revised_free_footprint)
            ],
            "public_revised_free_voxel_count": len(public_revised_free_footprint),
            "public_new_free_footprint_sha256": canonical_sha256(
                [list(key) for key in sorted(public_new_free_footprint)]
            ),
            "public_new_free_volume_m3": public_free_delta_m3,
            "quality_source": "public_sparse_range_outcomes",
            "cost_energy_j": ledger.total_energy_used_j,
            "feasible": qd_feasible,
            "collision_avoidance_recovery": recovery_execution_audit,
            "archive_exclusion_reasons": qd_exclusion_reasons,
            "archive_admission": {
                "admitted": admission.admitted,
                "reason": admission.reason,
                "cell": admission.cell,
                "replaced_manifest_hash": admission.replaced_manifest_hash,
                "revision": admission.revision,
            },
        }
        realised_qd_admissions.append(realised_qd_record)
        if qd_feasible:
            realised_qd_descriptors.append(descriptor)
            realised_qd_intents.append(tuple(selected.planned_descriptor))
            realised_qd_footprints.append(tuple(sorted(public_new_free_footprint)))
        round_elapsed_s = max((outcome.actual_end for outcome in ledger.outcomes), default=0.0)
        if round_elapsed_s <= 0.0:
            raise RuntimeError("CF2X round has no positive physical duration")
        elapsed_s = min(args.action_budget_s, round_start_s + round_elapsed_s)
        samples.append(
            _metric_sample(
                timestamp_s=elapsed_s,
                component=reachable_mask,
                grid_origin=grid_origin,
                resolution_m=0.25,
                denominator_volume_m3=denominator_volume_m3,
                team_belief=team_belief,
            )
        )
        metric_sample_after = samples[-1]
        segment_duration_s = metric_sample_after.timestamp_s - metric_sample_before.timestamp_s
        if segment_duration_s <= 0.0:
            raise RuntimeError("decision-level metric interval has non-positive duration")
        segment_auc_contribution = (
            0.5
            * segment_duration_s
            * (metric_sample_before.coverage_fraction + metric_sample_after.coverage_fraction)
            / args.action_budget_s
        )
        if decision_index == 0:
            segment_auc_contribution += bootstrap_auc_contribution
        execution = ledger.to_public_dict()
        training_reward_auc_contribution = (
            0.0 if selected_is_collision_avoidance_recovery else segment_auc_contribution
        )
        decision_training_rows.append(
            {
                "state": state,
                "pool": pool,
                "selected": selected,
                "execution": execution,
                "execution_calibration": execution_calibration,
                "team_trajectory_diversity": decision_team_diversity,
                "duration_s": segment_duration_s,
                "metric_auc_contribution": segment_auc_contribution,
                "auc_contribution": training_reward_auc_contribution,
                "training_exclusion_reason": (
                    "COLLISION_AVOIDANCE_RECOVERY"
                    if selected_is_collision_avoidance_recovery
                    else None
                ),
            }
        )
        post_execution_wall_s = time.perf_counter() - post_execution_wall_started
        decision_wall_s = time.perf_counter() - decision_wall_started
        decision_wall_timing = {
            "decision_id": decision_context.decision_id,
            "planning_wall_s": planning_wall_s,
            "initial_communication_graph_wall_s": initial_graph_wall_s,
            "frontier_extraction_wall_s": frontier_wall_s,
            "candidate_pool_wall_s": candidate_pool_wall_s,
            "route_guard_unique_wall_s": route_guard_timing["unique_wall_s"],
            "joint_guard_wall_s": joint_guard_timing["wall_s"],
            "candidate_assignment_and_manifest_wall_s": max(
                0.0,
                candidate_pool_wall_s
                - route_guard_timing["unique_wall_s"]
                - joint_guard_timing["wall_s"],
            ),
            "selection_wall_s": selection_wall_s,
            "execution_wall_s": execution_wall_s,
            "post_execution_wall_s": post_execution_wall_s,
            "total_wall_s": decision_wall_s,
        }
        decision_wall_rows.append(decision_wall_timing)
        decisions.append(
            {
                "decision_id": decision_context.decision_id,
                "public_context_hash": state.context.digest,
                "public_belief_sha256_before": belief_before_hash,
                "public_candidate_pool_hash": pool_digest,
                **public_schema_fields(),
                "selected_manifest_hash": selected.manifest_hash,
                "selected_public_candidate_intent": list(selected.planned_descriptor),
                "candidate_reachability": {
                    "effective_frontier_step_m": effective_frontier_step_m,
                    "reachable_path_length_m": reachable_path_length_m,
                    "decision_duration_s": decision_duration_s,
                    "observation_dwell_s": observation_dwell_s,
                    "cruise_speed_mps": transit_timing.cruise_speed_mps,
                    "max_accel_mps2": transit_timing.max_accel_mps2,
                    "terminal_tracking_margin_s": transit_timing.terminal_tracking_margin_s,
                    "intermediate_waypoint_settle_margin_s": (
                        transit_timing.intermediate_waypoint_settle_margin_s
                    ),
                    "calibrated_max_segment_count": transit_timing.calibrated_max_segment_count,
                    "uncovered_segment_reserve_s": transit_timing.uncovered_segment_reserve_s,
                    "public_frontier_count": len(state.frontiers),
                    "ordinary_public_frontier_count": sum(
                        frontier.task_kind == "explore" for frontier in state.frontiers
                    ),
                    "public_observation_viewpoint_count": sum(
                        frontier.task_kind == "explore"
                        and frontier.viewpoint_kind == "observation"
                        for frontier in state.frontiers
                    ),
                    "public_route_progress_fallback_count": sum(
                        frontier.task_kind == "explore"
                        and frontier.viewpoint_kind == "route_progress"
                        for frontier in state.frontiers
                    ),
                    "public_region_access_count": sum(
                        frontier.task_kind == "explore"
                        and frontier.viewpoint_kind == "region_access"
                        for frontier in state.frontiers
                    ),
                    "outcome_backtrack_offers": outcome_backtrack_offers,
                    "task_reservation_count": len(state.task_reservations),
                    "task_reservation_source_decision_ids": [
                        reservation.source_decision_id
                        for reservation in state.task_reservations
                    ],
                    "terminal_margin_distance_audit_m": terminal_margin_distance_audit_m,
                    "terminal_margin_distance_audit_only": True,
                    "feasible_candidate_count": sum(candidate.feasible for candidate in pool),
                    "candidate_roles": _candidate_role_summary(
                        pool,
                        selected_manifest_hash=selected.manifest_hash,
                    ),
                    "per_agent_edge_diagnostics": _per_agent_candidate_edge_diagnostics(
                        state,
                        team_belief,
                        route_guard_records,
                    ),
                    "candidate_route_opportunity_catalog": candidate_route_catalog,
                    "all_agents_active_candidate_exists": any(
                        candidate.feasible
                        and sum(
                            dict(fragment.type_signature.public_features).get("assignment_role")
                            == "explore"
                            for fragment in candidate.fragments
                            if fragment.type_signature.fragment_type == "transit"
                        )
                        == fleet_size
                        for candidate in pool
                    ),
                    "strategy_headroom_available": (
                        sum(candidate.feasible for candidate in pool) >= 2
                    ),
                    "route_guard_request_count": len(route_guard_records),
                    "route_guard_unique_query_count": route_guard_timing["unique_query_count"],
                    "route_guard_cache_hit_count": sum(
                        bool(record["cache_hit"]) for record in route_guard_records
                    ),
                    "public_free_reachability_cache": public_reachability_cache.audit(),
                    "joint_guard_call_count": joint_guard_timing["call_count"],
                    "selected_joint_safety": selected_joint_safety,
                    "route_guard_reason_counts": dict(
                        sorted(
                            Counter(
                                str(record["reason"] or "admitted")
                                for record in route_guard_records
                            ).items()
                        )
                    ),
                    "joint_guard_reason_counts": dict(
                        sorted(
                            Counter(
                                str(record["reason"] or "admitted")
                                for record in joint_guard_records
                            ).items()
                        )
                    ),
                    "vertical_opportunity": _vertical_opportunity_summary(
                        state.frontiers,
                        current_positions,
                        execution_calibration,
                        candidate_route_catalog,
                    ),
                    "empty_observation_cooldown": cooldown_audit,
                },
                "selection": selection,
                "execution": execution,
                "execution_calibration": execution_calibration,
                "stationarity_supervision": stationarity_supervision,
                "outcome_backtrack": outcome_backtrack_audit[-1],
                "task_reservation": task_reservation_audit[-1],
                "collision_avoidance_recovery": recovery_execution_audit,
                "empty_observation_cooldown_update": cooldown_update,
                "team_trajectory_diversity": decision_team_diversity,
                "source_observation_frame_count": len(backend.public_range_frames),
                "source_range_ray_count": len(backend.public_range_outcomes),
                "realised_qd": realised_qd_record,
                "public_belief_sha256_after": team_belief.content_sha256,
                "metric_explored_free_flight_volume_auc_time_contribution": (
                    segment_auc_contribution
                ),
                "reward_explored_free_flight_volume_auc_time_contribution": (
                    training_reward_auc_contribution
                ),
                "training_exclusion_reason": (
                    "COLLISION_AVOIDANCE_RECOVERY"
                    if selected_is_collision_avoidance_recovery
                    else None
                ),
                "duration_s": segment_duration_s,
                "elapsed_physics_s": elapsed_s,
                "wall_timing": decision_wall_timing,
            }
        )
        # Validate and checkpoint at the decision boundary. A malformed
        # diagnostic must fail after one decision, not after a full episode.
        canonical_sha256(decisions[-1])
        _write_decision_progress(
            paths["output"],
            scene_id=args.scene_id,
            strategy=args.strategy,
            action_budget_s=args.action_budget_s,
            elapsed_physics_s=elapsed_s,
            decisions=decisions,
            samples=samples,
        )
        print(
            json.dumps(
                {
                    "status": "P07_DECISION_COMPLETE",
                    "decision_id": decision_context.decision_id,
                    "strategy": args.strategy,
                    "feasible_candidate_count": sum(candidate.feasible for candidate in pool),
                    "elapsed_physics_s": elapsed_s,
                    "decision_wall_s": decision_wall_s,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        round_debug.append({"communication": communication, "message_delivery": delivery})
        total_energy_j += ledger.total_energy_used_j
        total_collision_count += ledger.collision_count
        total_inter_agent_separation_violation_count += (
            ledger.inter_agent_separation_violation_count
        )
        total_oob_count += ledger.out_of_bounds_count
        total_static_clearance_contract_violation_count += (
            ledger.static_clearance_contract_violation_count
        )
        total_failed_fragments += ledger.failed_fragment_count
        total_executed_fragments += ledger.executed_fragment_count
        outcome_hashes.extend(outcome.digest for outcome in ledger.outcomes)
        if not execution_complete:
            # A outcome-backed safety failure remains a scored episode.  The
            # metric keeps its observed volume constant through the frozen T.
            terminal_outcome = "executed_terminal_safety_failure"
            break
        if len(backend.final_root_positions_m) != fleet_size:
            raise RuntimeError("CF2X backend omitted final physical positions")
        current_positions = backend.final_root_positions_m
        current_boundary_linear_speeds_mps = _final_boundary_linear_speeds(
            backend, stage_id=decision_context.decision_id
        )
        decision_index += 1
        if periodic_supervision is not None:
            periodic_supervision.accumulate_delivery(delivery)
            periodic_supervision.accumulate_calibration(execution_calibration)
            periodic_supervision.accumulate_trace(
                execution_calibration.get("physics_visualization_trace"),
                start_s=round_start_s,
            )
            periodic_supervision.emit_until(
                elapsed_s=elapsed_s,
                positions_m=current_positions,
                linear_speeds_mps=current_boundary_linear_speeds_mps,
                samples=samples,
                horizon_s=args.action_budget_s,
                total_energy_j=total_energy_j,
                collision_count=total_collision_count,
                separation_violation_count=total_inter_agent_separation_violation_count,
                out_of_bounds_count=total_oob_count,
                static_clearance_violation_count=total_static_clearance_contract_violation_count,
                executed_fragment_count=total_executed_fragments,
                failed_fragment_count=total_failed_fragments,
                decision_count=decision_index,
            )

    if terminal_outcome != "budget_exhausted" and not decisions:
        raise RuntimeError("P07 terminal outcome has no executed decision outcome")
    if periodic_supervision is not None:
        periodic_supervision.emit_until(
            elapsed_s=elapsed_s,
            positions_m=current_positions,
            linear_speeds_mps=current_boundary_linear_speeds_mps,
            samples=samples,
            horizon_s=args.action_budget_s,
            total_energy_j=total_energy_j,
            collision_count=total_collision_count,
            separation_violation_count=total_inter_agent_separation_violation_count,
            out_of_bounds_count=total_oob_count,
            static_clearance_violation_count=total_static_clearance_contract_violation_count,
            executed_fragment_count=total_executed_fragments,
            failed_fragment_count=total_failed_fragments,
            decision_count=len(decisions),
        )
    action_budget_utilization = elapsed_s / args.action_budget_s
    if (
        terminal_outcome == "budget_exhausted"
        and action_budget_utilization < MINIMUM_ACTION_BUDGET_UTILIZATION
    ):
        raise RuntimeError(
            "online P07 episode ended before the required physical-time budget was used: "
            f"elapsed={elapsed_s:.6f}s budget={args.action_budget_s:.6f}s "
            f"utilization={action_budget_utilization:.6f}"
        )
    communication, delivery = _aggregate_communication(round_debug)
    communication_audit = communication_contract.audit_worker_evidence(communication, delivery)
    if communication_audit["passed"] is not True:
        raise RuntimeError(
            "aggregate communication evidence violates the frozen contract: "
            f"{json.dumps(communication_audit, sort_keys=True)}"
        )
    metric = score_exploration_episode(
        episode_id=root_context.episode_id,
        samples=tuple(samples),
        horizon_s=args.action_budget_s,
        collision_count=total_collision_count,
        energy_j=total_energy_j,
        delivered_messages=int(delivery["outcome_counts_after_close"]["DELIVERED"]),
        attempted_messages=int(delivery["expected_recipient_outcomes"]),
    )
    if not decision_training_rows:
        raise RuntimeError("completed P07 episode produced no decision-level training transition")
    # The shared budget tail, or a safety-stop carry-forward, is attributable
    # to the final selected action but must not become an extra policy action.
    # The bootstrap contribution was already attached to the first transition.
    terminal_tail_auc = metric.explored_free_flight_volume_auc_time - sum(
        float(row["metric_auc_contribution"]) for row in decision_training_rows
    )
    if terminal_tail_auc < -1.0e-9:
        raise RuntimeError("decision metric contributions exceed the frozen episode AUC")
    decision_training_rows[-1]["metric_auc_contribution"] = (
        float(decision_training_rows[-1]["metric_auc_contribution"]) + terminal_tail_auc
    )
    if decision_training_rows[-1]["training_exclusion_reason"] is None:
        decision_training_rows[-1]["auc_contribution"] = (
            float(decision_training_rows[-1]["auc_contribution"]) + terminal_tail_auc
        )
    decisions[-1]["metric_explored_free_flight_volume_auc_time_contribution"] = float(
        decision_training_rows[-1]["metric_auc_contribution"]
    )
    decisions[-1]["reward_explored_free_flight_volume_auc_time_contribution"] = float(
        decision_training_rows[-1]["auc_contribution"]
    )
    if not math.isclose(
        sum(float(row["metric_auc_contribution"]) for row in decision_training_rows),
        metric.explored_free_flight_volume_auc_time,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("decision metric contributions do not add up to the frozen episode AUC")
    realised_qd_audit = audit_realised_qd_richness(realised_qd_descriptors)
    intent_outcome_alignment = audit_intent_realised_alignment(
        realised_qd_intents,
        realised_qd_descriptors,
    )
    footprint_separation = audit_realised_qd_footprint_separation(
        realised_qd_descriptors,
        realised_qd_footprints,
    )
    trajectory_duplicate_decision_count = sum(
        int(row["duplicate_pair_count"]) > 0 for row in team_trajectory_diversity_audits
    )
    trajectory_observable_decision_count = sum(
        int(row["moving_explorer_count"]) >= 2 for row in team_trajectory_diversity_audits
    )
    moving_explorer_agent_ids = sorted(
        {
            str(agent_id)
            for row in team_trajectory_diversity_audits
            for agent_id in row["moving_explorer_agent_ids"]
        }
    )
    team_collaboration_reasons: list[str] = []
    if trajectory_duplicate_decision_count:
        team_collaboration_reasons.append("TRANSLATED_EXPLORER_TRAJECTORY_COPY")
    if trajectory_observable_decision_count == 0:
        team_collaboration_reasons.append("NO_DECISION_WITH_TWO_MOVING_EXPLORERS")
    if len(moving_explorer_agent_ids) < 2:
        team_collaboration_reasons.append("FEWER_THAN_TWO_EPISODE_EXPLORERS")
    team_collaboration_audit = {
        "schema_version": "hm3d-episode-team-collaboration-v1",
        "decision_count": len(team_trajectory_diversity_audits),
        "trajectory_diversity_observable_decision_count": (trajectory_observable_decision_count),
        "translated_duplicate_decision_count": trajectory_duplicate_decision_count,
        "moving_explorer_agent_ids": moving_explorer_agent_ids,
        "moving_explorer_agent_count": len(moving_explorer_agent_ids),
        "fleet_size": fleet_size,
        "status": (
            "EPISODE_TEAM_COLLABORATION_ADMITTED"
            if not team_collaboration_reasons
            else "EPISODE_TEAM_COLLABORATION_NOT_ADMITTED"
        ),
        "reasons": team_collaboration_reasons,
        "decision_audits": team_trajectory_diversity_audits,
        "claim_limit": (
            "Rejects translated explorer-path copies and one-vehicle-only episodes; "
            "task-level superiority still requires paired coverage and cost metrics."
        ),
    }
    initial_pool_hash = pool_hashes[0]
    execution = {
        "collision_count": total_collision_count,
        "inter_agent_separation_violation_count": total_inter_agent_separation_violation_count,
        "out_of_bounds_count": total_oob_count,
        "failed_fragment_count": total_failed_fragments,
        "executed_fragment_count": total_executed_fragments,
        "planned_fragment_count": fleet_size
        * 2
        * (len(decisions) + 1 + int(terminal_budget_tail is not None)),
        "terminal_outcome": terminal_outcome,
        "terminal_budget_tail": terminal_budget_tail,
        "outcome_count": len(outcome_hashes),
        "outcome_hashes": outcome_hashes,
        "total_energy_used_j": total_energy_j,
    }
    execution_status, execution_status_reason = _classify_execution_status(
        terminal_outcome=terminal_outcome,
        failed_fragment_count=total_failed_fragments,
    )
    training_transitions: dict[str, Any] = {}
    if args.record_purpose == "train_outcome":
        single_rl_rows: list[dict[str, object]] = []
        marl_ipp_rows: list[dict[str, object]] = []
        for index, transition_row in enumerate(decision_training_rows):
            next_transition_row = decision_training_rows[
                min(index + 1, len(decision_training_rows) - 1)
            ]
            is_final_transition = index == len(decision_training_rows) - 1
            # A frozen episode horizon is a truncation, not a task or safety
            # terminal.  The legacy ``done`` flag collapsed these cases and
            # made a successful budget-exhausted rollout indistinguishable
            # from a outcome-backed safety stop.
            terminated = (
                is_final_transition and terminal_outcome == "executed_terminal_safety_failure"
            )
            truncated = is_final_transition and terminal_outcome == "budget_exhausted"
            if is_final_transition and not (terminated or truncated):
                raise RuntimeError(f"unknown P07 terminal outcome {terminal_outcome!r}")
            common = {
                "next_state": next_transition_row["state"],
                "next_pool": next_transition_row["pool"],
                "scene_id": args.scene_id,
                "execution": transition_row["execution"],
                "explored_free_flight_volume_auc_time_contribution": transition_row[
                    "auc_contribution"
                ],
                "duration_s": transition_row["duration_s"],
                "terminated": terminated,
                "truncated": truncated,
            }
            single_rl_rows.append(
                build_single_rl_training_transition(
                    transition_row["state"],
                    transition_row["pool"],
                    transition_row["selected"],
                    **common,
                )
            )
            marl_ipp_rows.append(
                build_marl_ipp_training_transition(
                    transition_row["state"],
                    transition_row["pool"],
                    transition_row["selected"],
                    **common,
                )
            )
        training_transitions["single_rl_training_transitions"] = single_rl_rows
        training_transitions["marl_ipp_training_transitions"] = marl_ipp_rows
    evaluator_free_overlap = _evaluator_consistent_public_free(
        component=reachable_mask,
        grid_origin=grid_origin,
        resolution_m=0.25,
        team_belief=team_belief,
    )
    runtime_wall_elapsed_s = time.perf_counter() - runtime_wall_started
    decision_execution_wall_s = sum(float(row["execution_wall_s"]) for row in decision_wall_rows)
    payload = {
        "schema_version": "hm3d-p07-exploration-execution-v1",
        "status": execution_status,
        "status_reason": execution_status_reason,
        "synthetic": False,
        "formal_result": False,
        "p07_task_validity_closed": False,
        "record_purpose": args.record_purpose,
        "evidence_integrity_contract": build_current_evidence_integrity_contract(
            runner_source_sha256=_sha256(Path(__file__).resolve()),
            execution_source_sha256=_sha256(Path(cf2x.__file__).resolve()),
        ),
        "calibration_only_timeout_probe": timeout_probe,
        "claim_limit": (
            "Train-only QD replay calibration only; it validates public-outcome descriptor "
            "richness and repeatability and is not a P07 baseline or formal result."
            if qd_calibration
            else (
                "Train-only physical timeout-censoring evidence. It emits no replay transition, "
                "cannot enter QD history and is not a baseline result."
                if timeout_probe
                else (
                    "Train-split real target-free outcome collection. It validates the online "
                    "public-outcome loop and can enter train-only RL/QD datasets only after "
                    "field-level evidence validation."
                    if args.record_purpose == "train_outcome"
                    else (
                        "Engineering-only real target-free development episode. It validates "
                        "the online public-outcome loop and may support execution diagnostics, "
                        "but cannot enter RL, QD, or fragment-reuse datasets."
                    )
                )
            )
        ),
        "runner_version": RUNNER_VERSION,
        **public_schema_fields(),
        "scene_id": args.scene_id,
        "selection_partition": args.split,
        "split_manifest_sha256": split_manifest_sha256,
        "strategy": args.strategy,
        "qd_calibration_mode": args.qd_calibration_mode,
        "random_key": args.random_key,
        "reproducibility": {
            "episode_random_seed": args.random_key,
            "python_random_seeded": True,
            "numpy_random_seeded": True,
            "torch_random_seeded": True,
            "cuda_random_seeded": torch.cuda.is_available(),
            "physx_enhanced_determinism": True,
        },
        "selector_backbone_sha256": qd_selector_backbone_sha256(utility_slack=QD_UTILITY_SLACK),
        "public_context": root_context.to_dict(),
        "public_context_hash": root_context.digest,
        "public_episode_id": root_context.episode_id,
        "fleet_size": fleet_size,
        "candidate_limit": args.candidate_limit,
        "max_decision_count": args.max_decision_count,
        "decision_count": len(decisions),
        "candidate_headroom_diagnostics": {
            "physical_minimum_executable_candidates": (cf2x.P07_MINIMUM_EXECUTABLE_CANDIDATES),
            "decisions_with_strategy_headroom": sum(
                bool(row["candidate_reachability"]["strategy_headroom_available"])
                for row in decisions
            ),
            "strategy_headroom_fraction": (
                sum(
                    bool(row["candidate_reachability"]["strategy_headroom_available"])
                    for row in decisions
                )
                / len(decisions)
                if decisions
                else 0.0
            ),
        },
        "action_budget_s": args.action_budget_s,
        "elapsed_physics_s": elapsed_s,
        "action_budget_utilization": action_budget_utilization,
        "periodic_supervision": (
            {
                "schema_version": PERIODIC_SUPERVISION_SCHEMA_VERSION,
                "interval_s": args.supervision_interval_s,
                "sample_count": len(periodic_supervision.samples),
                "samples": periodic_supervision.samples,
                "claim_limit": (
                    "Audit-only periodic snapshots for route continuity and budget "
                    "supervision. They never enter selection, control, belief, safety, "
                    "rewards, QD, or replay."
                ),
            }
            if periodic_supervision is not None
            else None
        ),
        "terminal_outcome": terminal_outcome,
        "bootstrap": bootstrap,
        "physics_dt_s": args.physics_dt_s,
        "arrival_tolerance_m": args.arrival_tolerance_m,
        "outcome_time_tolerance_s": args.outcome_time_tolerance_s,
        "controller_id": args.controller_id,
        "action_completion_mode": "event_driven_all_routes_completed_plus_minimum_dwell",
        "execution_profile": execution_profile,
        "execution_profile_sha256": execution_profile_sha256,
        "sensor_profile_sha256": profile.entitlement_hash,
        "public_contract_sha256": public_contract_sha256,
        "evaluation_denominator_sha256": denominator_sha256,
        "evaluation_geometry_denominator_sha256": geometry_denominator_sha256,
        "evaluation_denominator": evaluation_denominator,
        "communication_contract": communication_contract.to_dict(),
        "communication_contract_sha256": communication_contract.digest,
        "communication": communication,
        "communication_contract_audit": communication_audit,
        "collision_usd_sha256": _sha256(paths["collision"]),
        "cf2x_usd_sha256": _sha256(paths["cf2x"]),
        "initial_public_relay_graph": initial_start_graph.to_dict(),
        "initial_start_reset": initial_start_reset_witness,
        "initial_position_source_sha256": _sha256(paths["start_resets"]),
        "transit_time_model_sha256": _sha256(paths["timing"]),
        "public_candidate_pool_hash": initial_pool_hash,
        "public_candidate_pool_sequence_hash": canonical_sha256(pool_hashes),
        "metric_report": metric.to_dict(),
        "mobility_summary": _episode_mobility_summary(decisions, starts),
        "stationarity_supervision": _episode_stationarity_summary(decisions),
        "outcome_backtrack_protocol": {
            "enabled": True,
            "claim_limit": (
                "A recovery route is an own-agent reversal of one completed, guarded "
                "exploration outcome. Static clearance is reused only when the source "
                "outcome's measured clearance slack covers the current endpoint offset; "
                "the connector ray, team separation and real execution checks still run. "
                "It is never fused into SparseVoxelBelief as FREE and is excluded from QD "
                "archive admission."
            ),
            "minimum_path_length_m": OUTCOME_BACKTRACK_MIN_PATH_M,
            "decisions": outcome_backtrack_audit,
        },
        "task_reservation": {
            "schema_version": PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
            "claim_limit": (
                "A outcome-grounded public task association may retain one matching "
                "frontier through a fresh extraction. Every next-step access route is "
                "rebuilt from current public belief and must pass the ordinary route "
                "guard and joint safety certificate; old frontier manifests are never replayed."
            ),
            "association_radius_m": PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M,
            "switch_margin_gain": PUBLIC_TASK_RESERVATION_SWITCH_MARGIN_GAIN,
            "terminal_margin_distance_audit_only": {
                "value_m": terminal_margin_distance_audit_m,
                "method": "maximum_rest_to_rest_distance_m",
                "input_duration_s": transit_timing.terminal_tracking_margin_s,
                "not_used_for": [
                    "candidate_legality",
                    "route_length_target",
                    "reward",
                    "controller_command",
                ],
            },
            "decisions": task_reservation_audit,
        },
        "realised_qd": {
            "claim_limit": (
                "Outcome-grounded archive diagnostics only. This worker does not claim a "
                "realised-QD selection gain until the paired planned-QD/no-QD matrix is run."
            ),
            "archive_spec": realised_qd_archive.spec.to_dict(),
            "archive_spec_sha256": realised_qd_archive.spec.digest,
            "archive_metrics": realised_qd_archive.metrics(),
            "richness_audit": realised_qd_audit.to_dict(),
            "footprint_separation_audit": footprint_separation.to_dict(),
            "candidate_intent_audits": candidate_intent_audits,
            "value_protected_candidate_diversity_audits": (
                value_protected_candidate_diversity_audits
            ),
            "intent_outcome_alignment": intent_outcome_alignment.to_dict(),
            "selection_mode": args.strategy,
            "selector_backbone_sha256": qd_selector_backbone_sha256(utility_slack=QD_UTILITY_SLACK),
            "utility_slack": QD_UTILITY_SLACK,
            "history": qd_history_summary,
            "admissions": realised_qd_admissions,
        },
        "team_collaboration_audit": team_collaboration_audit,
        "execution": execution,
        "decisions": decisions,
        "public_observation_summary": {
            "source_observation_binding": True,
            "public_team_belief_sha256": team_belief.content_sha256,
            "public_observed_free_voxel_count": team_belief.observed_free_count,
            "public_predicted_free_volume_m3": evaluator_free_overlap.public_volume_m3,
            "evaluator_consistent_public_free_volume_m3": (
                evaluator_free_overlap.consistent_volume_m3
            ),
            "evaluator_inconsistent_public_free_volume_m3": (
                evaluator_free_overlap.inconsistent_volume_m3
            ),
            "touched_evaluator_voxel_count": (evaluator_free_overlap.touched_evaluator_voxel_count),
            "touched_free_evaluator_voxel_count": (
                evaluator_free_overlap.touched_free_evaluator_voxel_count
            ),
            "overlap_piece_count": evaluator_free_overlap.overlap_piece_count,
            "grid_phase_offset_fraction": list(evaluator_free_overlap.grid_phase_offset_fraction),
            "volume_conservation_error_m3": evaluator_free_overlap.conservation_error_m3,
        },
        "runtime_performance": {
            "schema_version": "hm3d-p07-wall-timing-v1",
            "measured_scope": "main_after_app_launch_before_json_serialization",
            "setup_wall_s": setup_wall_s,
            "bootstrap_wall_s": bootstrap_wall_s,
            "decision_wall_rows": decision_wall_rows,
            "decision_execution_wall_s": decision_execution_wall_s,
            "total_wall_s": runtime_wall_elapsed_s,
            "wall_to_physics_ratio": runtime_wall_elapsed_s / max(elapsed_s, 1.0e-9),
        },
        "engineering_debug": {
            "execution": {"communication": communication, "message_delivery": delivery},
            "bootstrap_communication": bootstrap_communication,
        },
        **training_transitions,
    }
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    _write_new(paths["output"], payload)
    _progress_path(paths["output"]).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(paths["output"]),
                "decision_count": len(decisions),
                "final_coverage": metric.final_coverage_at_budget,
                "auc": metric.explored_free_flight_volume_auc_time,
            },
            sort_keys=True,
        )
    )
    return 0


def _write_failure(args: argparse.Namespace, error: BaseException) -> None:
    output = args.output.expanduser().resolve()
    if output.exists():
        return
    payload = {
        "schema_version": "hm3d-p07-exploration-execution-v1",
        "status": P07_EXECUTION_SMOKE_FAILED_STATUS,
        "synthetic": False,
        "formal_result": False,
        "p07_task_validity_closed": False,
        "scene_id": args.scene_id,
        "selection_partition": args.split,
        "strategy": args.strategy,
        "failure_denominator": {"planned": 1, "executed": 0, "failed": 1},
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            )[-8000:],
            "partial_progress_path": _progress_path(args.output).name,
        },
        "status_reason": "runner_exception",
    }
    if isinstance(error, CandidateRouteGuardError):
        payload["route_guard_diagnostics"] = error.diagnostics
        completed_decisions = error.diagnostics.get("completed_decisions")
        if isinstance(completed_decisions, list) and completed_decisions:
            payload["decisions"] = completed_decisions
            payload["decision_count"] = len(completed_decisions)
            payload["elapsed_physics_s"] = error.diagnostics.get("elapsed_physics_s")
            payload["action_budget_s"] = args.action_budget_s
            payload["partial_metric_samples"] = error.diagnostics.get(
                "completed_metric_samples", []
            )
            payload["claim_limit"] = (
                "Completed PhysX decisions remain engineering and dynamics evidence. "
                "The interrupted episode is not formal performance evidence, a trainable RL "
                "trajectory, or a QD mechanism result."
            )
            payload["partial_evidence_qualification"] = {
                "formal_performance_evidence": {
                    "eligible": False,
                    "reasons": ["EPISODE_INTERRUPTED_BEFORE_FROZEN_BUDGET"],
                },
                "trainable_real_outcome": {
                    "eligible": False,
                    "reasons": ["INCOMPLETE_DECISION_SEQUENCE"],
                },
                "dynamics_calibration_evidence": {"eligible": True, "reasons": []},
                "engineering_diagnostic_evidence": {"eligible": True, "reasons": []},
            }
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    _write_new(output, payload)


def _entrypoint() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    app = AppLauncher(args)
    try:
        exit_code = main(args, app.app)
    except BaseException as error:
        try:
            _write_failure(args, error)
        except BaseException:
            traceback.print_exc()
        traceback.print_exc()
        exit_code = 2
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
