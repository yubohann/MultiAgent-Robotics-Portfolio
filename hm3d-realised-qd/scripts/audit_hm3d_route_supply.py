"""Audit where public exploration route length disappears in an immutable outcome.

This tool is deliberately outcome-only.  It does not reconstruct a belief,
modify an episode, or infer evaluator truth.  Its job is to distinguish a
short-route selector outcome from a shortage of individually or jointly legal
public FREE-space routes before changing the candidate generator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# A guarded path length is only an executable public route after the route
# guard has proved connectivity.  Disconnected opportunities intentionally
# retain a geometric fallback length for diagnosis, but that value must never
# be reported as route supply.
_CONNECTED_PUBLIC_ROUTE_STATUSES = frozenset(
    {
        "revalidated_public_access_plan",
        "admitted",
        "outcome_backtrack",
        "exact_clearance_grid_route",
    }
)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _decision_edges(decision: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    reachability = decision.get("candidate_reachability")
    if not isinstance(reachability, Mapping):
        raise ValueError("decision is missing candidate_reachability")
    catalog = reachability.get("candidate_route_opportunity_catalog")
    if not isinstance(catalog, Mapping):
        raise ValueError("decision is missing candidate_route_opportunity_catalog")
    agents = catalog.get("agents")
    if not isinstance(agents, list):
        raise ValueError("candidate_route_opportunity_catalog.agents must be a list")
    edges: list[Mapping[str, Any]] = []
    for agent in agents:
        if not isinstance(agent, Mapping):
            raise ValueError("route opportunity agent must be an object")
        rows = agent.get("frontier_edges")
        if not isinstance(rows, list):
            raise ValueError("route opportunity frontier_edges must be a list")
        for edge in rows:
            if not isinstance(edge, Mapping):
                raise ValueError("route opportunity edge must be an object")
            edges.append(edge)
    return tuple(edges)


def _longest(edges: Iterable[Mapping[str, Any]], *, field: str) -> float:
    """Return the longest present numeric route field.

    A disconnected opportunity intentionally serializes no guarded route
    length.  Treating that absence as malformed data or as a zero-metre route
    would erase the distinction this audit exists to expose.
    """

    return max(
        (
            _finite_float(edge.get(field), field=field)
            for edge in edges
            if edge.get(field) is not None
        ),
        default=0.0,
    )


def _longest_connected_public_route(edges: Iterable[Mapping[str, Any]]) -> float:
    return _longest(
        (
            edge
            for edge in edges
            if edge.get("public_route_status") in _CONNECTED_PUBLIC_ROUTE_STATUSES
        ),
        field="guarded_path_length_m",
    )


def _selected_team_rows(decision: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    reachability = decision["candidate_reachability"]
    if not isinstance(reachability, Mapping):
        raise ValueError("decision candidate_reachability must be an object")
    rows = reachability.get("candidate_roles")
    if not isinstance(rows, list):
        raise ValueError("candidate_reachability.candidate_roles must be a list")
    result = tuple(row for row in rows if isinstance(row, Mapping) and row.get("selected") is True)
    if len(result) != 1:
        raise ValueError(f"expected exactly one selected candidate role, found {len(result)}")
    return result


def audit_outcome(payload: Mapping[str, Any]) -> dict[str, object]:
    """Return a versioned, serializable route-supply audit for one outcome."""

    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("outcome must contain non-empty decisions")
    decision_rows: list[dict[str, object]] = []
    aggregate_statuses: Counter[str] = Counter()
    aggregate_reasons: Counter[str] = Counter()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError("outcome decision must be an object")
        edges = _decision_edges(decision)
        individually_admitted = tuple(
            edge for edge in edges if edge.get("individual_exploration_edge_admitted") is True
        )
        team_admitted = tuple(
            edge for edge in edges if edge.get("appears_in_feasible_team_candidate") is True
        )
        selected_edges = tuple(edge for edge in edges if edge.get("selected") is True)
        selected_role = _selected_team_rows(decision)[0]
        statuses = Counter(str(edge.get("public_route_status", "")) for edge in edges)
        reasons = Counter(str(edge.get("static_guard_reason", "")) for edge in edges)
        aggregate_statuses.update(statuses)
        aggregate_reasons.update(reasons)
        decision_rows.append(
            {
                "decision_index": index,
                "decision_id": decision.get("decision_id"),
                "duration_s": _finite_float(decision.get("duration_s"), field="duration_s"),
                "edge_count": len(edges),
                "public_route_length_unavailable_edge_count": sum(
                    edge.get("guarded_path_length_m") is None for edge in edges
                ),
                "individually_admitted_edge_count": len(individually_admitted),
                "feasible_team_edge_count": len(team_admitted),
                "selected_edge_count": len(selected_edges),
                "max_direct_opportunity_distance_m": _longest(
                    edges, field="direct_distance_m"
                ),
                "max_guarded_path_proposal_length_m": _longest(
                    edges, field="guarded_path_length_m"
                ),
                "max_public_free_connected_path_length_m": _longest_connected_public_route(
                    edges
                ),
                "max_individually_admitted_edge_length_m": _longest(
                    individually_admitted, field="guarded_path_length_m"
                ),
                "max_feasible_team_edge_length_m": _longest(
                    team_admitted, field="guarded_path_length_m"
                ),
                "max_selected_edge_length_m": _longest(
                    selected_edges, field="guarded_path_length_m"
                ),
                "selected_team_planned_path_length_m": _finite_float(
                    selected_role.get("team_planned_path_length_m"),
                    field="selected_team_planned_path_length_m",
                ),
                "selected_moving_agent_count": int(selected_role.get("moving_agent_count", 0)),
                "public_route_status_counts": dict(sorted(statuses.items())),
                "static_guard_reason_counts": dict(sorted(reasons.items())),
            }
        )
    return {
        "schema_version": "hm3d-route-supply-audit-v2",
        "outcome_runtime_record_sha256": payload.get("runtime_record_sha256"),
        "scene_id": payload.get("scene_id"),
        "strategy": payload.get("strategy"),
        "decision_count": len(decision_rows),
        "decisions": decision_rows,
        "aggregate_public_route_status_counts": dict(sorted(aggregate_statuses.items())),
        "aggregate_static_guard_reason_counts": dict(sorted(aggregate_reasons.items())),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcome", type=Path, help="immutable P07 outcome JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new audit JSON path; the tool refuses to overwrite it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(args.outcome.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("outcome root must be an object")
    audit = audit_outcome(payload)
    rendered = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite route-supply audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
