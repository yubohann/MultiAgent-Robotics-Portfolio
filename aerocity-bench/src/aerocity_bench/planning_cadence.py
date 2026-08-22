"""Versioned high-level planning cadence shared by L0 and L1 executors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .contracts import ActionPacket, ObservationPacket

PLANNING_CADENCE_SCHEMA = "org.aerocity.bench.planning-cadence.v1"
PLANNING_CADENCE_MODE = "fixed-rate-with-public-events"
PLANNING_EVENT_TRIGGERS = (
    "anonymous_confirmation",
    "safety_intervention",
    "fleet_roster_change",
    "return_reserve_entry",
)


def validate_planning_cadence(
    node: object,
    *,
    control_period_s: float,
    episode_duration_s: float,
) -> dict[str, Any]:
    """Validate a cadence without allowing method-specific scheduling knobs."""

    if not isinstance(node, dict):
        raise ValueError("execution planning cadence must be an object")
    required = {
        "schema",
        "mode",
        "period_s",
        "event_triggers",
        "held_action_rebinding",
        "retransmit_messages_on_hold",
    }
    if set(node) != required:
        raise ValueError("execution planning cadence fields differ")
    if node["schema"] != PLANNING_CADENCE_SCHEMA:
        raise ValueError("execution planning cadence schema differs")
    if node["mode"] != PLANNING_CADENCE_MODE:
        raise ValueError("execution planning cadence mode differs")
    period = float(node["period_s"])
    if not math.isfinite(period) or period <= control_period_s or period > episode_duration_s:
        raise ValueError(
            "planning period must be finite, slower than control, and within the episode"
        )
    ratio = period / control_period_s
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("planning period must be an integer multiple of control_period_s")
    triggers = node["event_triggers"]
    if not isinstance(triggers, list) or tuple(triggers) != PLANNING_EVENT_TRIGGERS:
        raise ValueError("planning public event triggers differ from the frozen contract")
    if node["held_action_rebinding"] != "latest-public-observation":
        raise ValueError("held actions must bind the latest public observation")
    if node["retransmit_messages_on_hold"] is not False:
        raise ValueError("held actions must not retransmit planner messages")
    return node


def rebind_held_action(
    action: ActionPacket,
    observation: ObservationPacket,
) -> ActionPacket:
    """Bind approved mission semantics to a new control tick without replanning."""

    if action.drone_id != observation.drone_id or action.episode_id != observation.episode_id:
        raise ValueError("held action and current public observation identities differ")
    return ActionPacket(
        episode_id=observation.episode_id,
        drone_id=observation.drone_id,
        sequence=observation.sequence,
        issued_at_s=observation.timestamp_s,
        kind=action.kind,
        waypoint=action.waypoint,
        velocity_body_mps=action.velocity_body_mps,
        yaw_rate_deg_s=action.yaw_rate_deg_s,
        sensor_pitch_deg=action.sensor_pitch_deg,
        source_observation_id=(
            observation.observation_id if action.kind == "OBSERVE" else None
        ),
        # Reissuing messages would turn a 1 Hz planning decision into a 5 Hz
        # communication policy and double-count the bandwidth budget.
        messages=(),
    )


@dataclass
class PlanningCadenceController:
    """Choose planner invocations while retaining only approved public actions."""

    control_period_s: float
    planning_period_s: float
    _approved: dict[str, ActionPacket] = field(default_factory=dict)
    _pending_events: set[str] = field(default_factory=set)
    _last_roster: tuple[str, ...] | None = None

    @classmethod
    def from_execution_contract(cls, execution_contract: dict[str, Any]):
        control_period = float(execution_contract["control_period_s"])
        episode = execution_contract["episode"]
        planning = validate_planning_cadence(
            execution_contract.get("planning"),
            control_period_s=control_period,
            episode_duration_s=float(episode["duration_s"]),
        )
        return cls(control_period_s=control_period, planning_period_s=float(planning["period_s"]))

    @property
    def interval_ticks(self) -> int:
        return int(round(self.planning_period_s / self.control_period_s))

    def request_event(self, event: str) -> None:
        if event not in PLANNING_EVENT_TRIGGERS:
            raise ValueError(f"unsupported public planning event: {event}")
        self._pending_events.add(event)

    def due_reasons(
        self,
        *,
        control_tick: int,
        active_drone_ids: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        if control_tick < 0:
            raise ValueError("control_tick must be non-negative")
        roster = tuple(sorted(str(value) for value in active_drone_ids))
        if not roster:
            raise ValueError("planning cadence requires an active fleet")
        if self._last_roster is not None and roster != self._last_roster:
            self._pending_events.add("fleet_roster_change")
        self._last_roster = roster
        reasons = set(self._pending_events)
        if not self._approved:
            reasons.add("initial")
        if control_tick % self.interval_ticks == 0:
            reasons.add("fixed_period")
        return tuple(sorted(reasons))

    def approve(self, actions: dict[str, ActionPacket]) -> None:
        if not actions:
            raise ValueError("cannot approve an empty fleet action set")
        if set(actions) != set(self._last_roster or ()):
            raise ValueError("approved action roster differs from the active fleet")
        self._approved = dict(actions)
        self._pending_events.clear()

    def reject_planning_attempt(self) -> None:
        # A deadline miss never promotes late actions. Replan on the next
        # control tick while the executor applies its frozen safe overrun rule.
        self._pending_events.add("safety_intervention")

    def held_actions(
        self,
        observations: dict[str, ObservationPacket],
    ) -> dict[str, ActionPacket]:
        if not self._approved:
            raise RuntimeError("no approved actions are available for a non-planning tick")
        if set(observations) != set(self._approved):
            raise ValueError("held action roster differs from current observations")
        return {
            drone_id: rebind_held_action(self._approved[drone_id], observation)
            for drone_id, observation in sorted(observations.items())
        }
