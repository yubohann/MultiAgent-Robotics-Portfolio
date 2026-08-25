"""Public-action signatures for screening scientifically redundant methods."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import content_hash
from .contracts import ActionPacket

BEHAVIOR_SUMMARY_SCHEMA = "org.aerocity.bench.public-action-behavior.v1"
PANEL_AUDIT_SCHEMA = "org.aerocity.bench.method-panel-behavior-audit.v1"
COHORT_PANEL_AUDIT_SCHEMA = "org.aerocity.bench.method-panel-behavior-cohort-audit.v2"


def _message_semantics(message: Mapping[str, Any]) -> dict[str, Any]:
    created = float(message["created_at_s"])
    expires = float(message["expires_at_s"])
    if expires <= created:
        raise ValueError("public action message has a non-positive lifetime")
    destinations = message.get("destination_drone_ids")
    if not isinstance(destinations, list) or not destinations:
        raise ValueError("public action message lacks destinations")
    return {
        "source_drone_id": str(message["source_drone_id"]),
        "destination_drone_ids": sorted(str(value) for value in destinations),
        "lifetime_s": expires - created,
        "payload_hex": str(message["payload_hex"]),
    }


def _action_semantics(action: ActionPacket | Mapping[str, Any]) -> dict[str, Any]:
    node = action.to_dict() if isinstance(action, ActionPacket) else dict(action)
    kind = str(node.get("kind", ""))
    if kind not in {"HOVER", "WAYPOINT", "VELOCITY", "OBSERVE", "RETURN"}:
        raise ValueError("public action trace contains an unknown action kind")
    messages = node.get("messages", [])
    if not isinstance(messages, list) or any(not isinstance(item, Mapping) for item in messages):
        raise ValueError("public action trace contains malformed messages")
    # Episode IDs, packet sequence numbers, absolute issue times, message IDs,
    # and source-observation IDs bind receipts but do not change the requested
    # mission behavior. Invocation order is retained by the outer trace.
    return {
        "drone_id": str(node["drone_id"]),
        "kind": kind,
        "waypoint": node.get("waypoint"),
        "velocity_body_mps": node.get("velocity_body_mps"),
        "yaw_rate_deg_s": float(node.get("yaw_rate_deg_s", 0.0)),
        "sensor_pitch_deg": node.get("sensor_pitch_deg"),
        "messages": [_message_semantics(item) for item in messages],
    }


def summarize_public_action_trace(
    trace: Sequence[Mapping[str, ActionPacket | Mapping[str, Any]]],
) -> dict[str, Any]:
    """Hash mission-level action semantics without persisting the action trace."""

    if not trace:
        raise ValueError("public action behavior requires at least one planner invocation")
    normalized: list[list[dict[str, Any]]] = []
    kinds: Counter[str] = Counter()
    expected_roster: tuple[str, ...] | None = None
    for invocation in trace:
        if not invocation:
            raise ValueError("public action behavior contains an empty invocation")
        roster = tuple(sorted(str(drone_id) for drone_id in invocation))
        if expected_roster is None:
            expected_roster = roster
        elif roster != expected_roster:
            raise ValueError("public action behavior changes the fleet roster")
        actions = []
        for drone_id in roster:
            semantics = _action_semantics(invocation[drone_id])
            if semantics["drone_id"] != drone_id:
                raise ValueError("public action behavior key and action drone differ")
            actions.append(semantics)
            kinds[str(semantics["kind"])] += 1
        normalized.append(actions)
    return {
        "schema": BEHAVIOR_SUMMARY_SCHEMA,
        "planner_invocation_count": len(normalized),
        "action_count": sum(len(actions) for actions in normalized),
        "fleet_roster": list(expected_roster or ()),
        "action_kind_counts": dict(sorted(kinds.items())),
        "mission_action_semantics_sha256": content_hash(normalized),
        "identity_and_absolute_time_fields_omitted": True,
        "private_truth_omitted": True,
    }


def audit_method_panel_behavior(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Find exact L0 behavior equivalence without claiming general algorithm identity."""

    if len(reports) < 2:
        raise ValueError("method-panel behavior audit requires at least two reports")
    contexts: set[tuple[str, str]] = set()
    signatures: dict[str, set[str]] = {}
    for report in reports:
        if report.get("formal_score_eligible") is not False:
            raise ValueError("behavior audit accepts diagnostic reports only")
        method_id = str(report.get("method_id", ""))
        layout_hash = str(report.get("layout_hash", ""))
        episode_hash = str(report.get("episode_hash", ""))
        if not method_id or not layout_hash or not episode_hash:
            raise ValueError("behavior report lacks method or public context binding")
        contexts.add((layout_hash, episode_hash))
        replicates = report.get("replicates")
        if not isinstance(replicates, list) or not replicates:
            raise ValueError("behavior report lacks replicates")
        method_signatures = signatures.setdefault(method_id, set())
        for replicate in replicates:
            if not isinstance(replicate, Mapping):
                raise ValueError("behavior report replicate is malformed")
            summary = replicate.get("public_action_behavior")
            if not isinstance(summary, Mapping) or summary.get("schema") != BEHAVIOR_SUMMARY_SCHEMA:
                raise ValueError("behavior report replicate lacks a public action signature")
            signature = str(summary.get("mission_action_semantics_sha256", ""))
            if len(signature) != 64:
                raise ValueError("behavior report contains an invalid action signature")
            method_signatures.add(signature)
    if len(contexts) != 1:
        raise ValueError("behavior reports must use the same layout and episode")

    nondeterministic = sorted(method for method, values in signatures.items() if len(values) != 1)
    by_signature: dict[str, list[str]] = {}
    for method, values in signatures.items():
        if len(values) == 1:
            by_signature.setdefault(next(iter(values)), []).append(method)
    equivalent_groups = [
        sorted(methods) for methods in by_signature.values() if len(methods) > 1
    ]
    equivalent_groups.sort()
    distinct_deterministic_signatures = len(by_signature)
    status = "PASS"
    if nondeterministic:
        status = "REVIEW_NONDETERMINISM"
    elif equivalent_groups:
        status = "REVIEW_EXACT_EQUIVALENCE"
    layout_hash, episode_hash = next(iter(contexts))
    result: dict[str, Any] = {
        "schema": PANEL_AUDIT_SCHEMA,
        "formal_score_eligible": False,
        "status": status,
        "layout_hash": layout_hash,
        "episode_hash": episode_hash,
        "method_count": len(signatures),
        "distinct_deterministic_behavior_count": distinct_deterministic_signatures,
        "exact_equivalence_groups": equivalent_groups,
        "nondeterministic_methods": nondeterministic,
        "may_count_all_methods_as_behaviorally_distinct": (
            not equivalent_groups and not nondeterministic
        ),
        "does_not_claim_general_algorithm_equivalence": True,
        "does_not_delete_or_censor_replays": True,
        "requires_justification_before_redundant_l1": bool(equivalent_groups),
    }
    result["report_hash"] = content_hash(result)
    return result


def audit_method_panel_behavior_cohort(
    reports: Sequence[Mapping[str, Any]],
    *,
    minimum_contexts: int = 3,
) -> dict[str, Any]:
    """Screen exact mechanism redundancy across a complete ancestor-method panel.

    Exact equality on one city is only a review trigger.  A method is grouped
    as redundant here only when its deterministic action semantics match on
    every audited public context.  Stochastic methods remain visible but are
    never declared distinct from one uncontrolled trace.
    """

    if minimum_contexts < 2:
        raise ValueError("cohort behavior audit minimum_contexts must be at least two")
    indexed: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for report in reports:
        method_id = str(report.get("method_id", ""))
        layout_hash = str(report.get("layout_hash", ""))
        episode_hash = str(report.get("episode_hash", ""))
        if not method_id or len(layout_hash) != 64 or len(episode_hash) != 64:
            raise ValueError("cohort behavior report lacks method or context bindings")
        context = (layout_hash, episode_hash)
        if method_id in indexed.setdefault(context, {}):
            raise ValueError("cohort behavior panel duplicates a method/context report")
        indexed[context][method_id] = report
    if len(indexed) < minimum_contexts:
        raise ValueError("cohort behavior audit has too few independent public contexts")
    method_sets = {tuple(sorted(by_method)) for by_method in indexed.values()}
    if len(method_sets) != 1:
        raise ValueError("cohort behavior audit is not a complete method-by-context panel")
    methods = next(iter(method_sets))
    if len(methods) < 2:
        raise ValueError("cohort behavior audit requires at least two methods")

    context_audits: list[dict[str, Any]] = []
    vectors: dict[str, list[str]] = {method: [] for method in methods}
    nondeterministic: set[str] = set()
    for context in sorted(indexed):
        audit = audit_method_panel_behavior(
            [indexed[context][method] for method in methods]
        )
        context_audits.append(
            {
                "layout_hash": context[0],
                "episode_hash": context[1],
                "status": audit["status"],
                "exact_equivalence_groups": audit["exact_equivalence_groups"],
                "nondeterministic_methods": audit["nondeterministic_methods"],
                "audit_report_hash": audit["report_hash"],
            }
        )
        nondeterministic.update(str(value) for value in audit["nondeterministic_methods"])
        for method in methods:
            signatures = {
                str(replicate["public_action_behavior"]["mission_action_semantics_sha256"])
                for replicate in indexed[context][method]["replicates"]
            }
            if len(signatures) == 1:
                vectors[method].append(next(iter(signatures)))

    deterministic_methods = [method for method in methods if method not in nondeterministic]
    by_vector: dict[tuple[str, ...], list[str]] = {}
    for method in deterministic_methods:
        vector = tuple(vectors[method])
        if len(vector) != len(indexed):
            raise ValueError("deterministic method behavior vector is incomplete")
        by_vector.setdefault(vector, []).append(method)
    mechanism_groups = [sorted(group) for group in by_vector.values()]
    mechanism_groups.extend([[method] for method in sorted(nondeterministic)])
    mechanism_groups.sort(key=lambda group: tuple(group))
    equivalent_groups = [group for group in mechanism_groups if len(group) > 1]
    representatives = [group[0] for group in mechanism_groups]
    excluded_redundant = sorted(
        method for group in equivalent_groups for method in group[1:]
    )
    status = "PASS"
    if nondeterministic:
        status = "REVIEW_NONDETERMINISM"
    elif equivalent_groups:
        status = "REVIEW_EXACT_EQUIVALENCE_ACROSS_COHORT"
    result: dict[str, Any] = {
        "schema": COHORT_PANEL_AUDIT_SCHEMA,
        "formal_score_eligible": False,
        "status": status,
        "context_count": len(indexed),
        "method_count": len(methods),
        "method_ids": list(methods),
        "context_audits": context_audits,
        "mechanism_groups": mechanism_groups,
        "l1_representative_method_ids": representatives,
        "excluded_redundant_method_ids": excluded_redundant,
        "nondeterministic_methods": sorted(nondeterministic),
        "distinct_mechanism_lower_bound": len(mechanism_groups),
        "may_count_all_methods_as_behaviorally_distinct": (
            not equivalent_groups and not nondeterministic
        ),
        "does_not_claim_general_algorithm_equivalence": True,
        "does_not_delete_or_censor_candidate_methods": True,
        "requires_stochastic_repeat_adjudication": bool(nondeterministic),
    }
    result["report_hash"] = content_hash(result)
    return result
