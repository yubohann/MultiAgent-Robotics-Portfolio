"""Public range/LOS relay-graph accounting for multi-UAV execution.

The graph deliberately contains only measured vehicle positions and a
collision-world line-of-sight predicate.  It has no target, evaluator, or
global-map privilege.  A direct peer edge is not confused with a multi-hop
relay path, which matters whenever aircraft occupy different rooms or floors.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier

Point3 = tuple[float, float, float]
LineOfSight = Callable[[Point3, Point3], bool]


def _point(value: Sequence[float], label: str) -> Point3:
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates")
    point = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{label} coordinates must be finite")
    return point  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RelayGraphSnapshot:
    """One instant of a public range-and-LOS communication graph."""

    adjacency: tuple[tuple[bool, ...], ...]
    components: tuple[tuple[int, ...], ...]
    direct_link_count: int
    fully_relay_connected: bool
    maximum_relay_hops: int | None

    def __post_init__(self) -> None:
        count = len(self.adjacency)
        if count == 0 or any(len(row) != count for row in self.adjacency):
            raise ValueError("relay adjacency must be a non-empty square matrix")
        if any(self.adjacency[index][index] for index in range(count)):
            raise ValueError("relay graph must not contain self edges")
        if any(
            self.adjacency[left][right] != self.adjacency[right][left]
            for left in range(count)
            for right in range(count)
        ):
            raise ValueError("relay adjacency must be symmetric")
        expected_links = sum(
            self.adjacency[left][right] for left in range(count) for right in range(left + 1, count)
        )
        if self.direct_link_count != expected_links:
            raise ValueError("direct link count disagrees with adjacency")
        flattened = tuple(index for component in self.components for index in component)
        if tuple(sorted(flattened)) != tuple(range(count)):
            raise ValueError("relay components must partition every agent exactly once")
        if self.fully_relay_connected != (len(self.components) == 1):
            raise ValueError("relay connected flag disagrees with components")
        if self.fully_relay_connected and self.maximum_relay_hops is None:
            raise ValueError("connected relay graph needs a maximum hop count")
        if not self.fully_relay_connected and self.maximum_relay_hops is not None:
            raise ValueError("disconnected relay graph cannot claim an all-pairs hop count")

    @property
    def agent_relay_reachable_to_all(self) -> tuple[bool, ...]:
        """Whether each agent can relay a message to every current team member."""

        return tuple(self.fully_relay_connected for _ in self.adjacency)

    def to_dict(self) -> dict[str, object]:
        return {
            "adjacency": [[int(edge) for edge in row] for row in self.adjacency],
            "components": [list(component) for component in self.components],
            "direct_link_count": self.direct_link_count,
            "fully_relay_connected": self.fully_relay_connected,
            "maximum_relay_hops": self.maximum_relay_hops,
        }

    def shortest_relay_path(self, source_index: int, receiver_index: int) -> tuple[int, ...] | None:
        """Return the current shortest public relay route, if one exists."""

        count = len(self.adjacency)
        if not 0 <= source_index < count or not 0 <= receiver_index < count:
            raise IndexError("relay endpoint lies outside the graph")
        if source_index == receiver_index:
            return (source_index,)
        predecessors = [-1] * count
        predecessors[source_index] = source_index
        queue: deque[int] = deque((source_index,))
        while queue:
            current = queue.popleft()
            for neighbor, linked in enumerate(self.adjacency[current]):
                if linked and predecessors[neighbor] < 0:
                    predecessors[neighbor] = current
                    queue.append(neighbor)
        if predecessors[receiver_index] < 0:
            return None
        path = [receiver_index]
        while path[-1] != source_index:
            path.append(predecessors[path[-1]])
        return tuple(reversed(path))


def build_range_los_relay_graph(
    positions: Sequence[Sequence[float]],
    *,
    maximum_range_m: float,
    line_of_sight_clear: LineOfSight,
) -> RelayGraphSnapshot:
    """Build an undirected public relay graph from range-limited LOS links.

    The caller owns the geometry query.  This keeps the function deterministic
    and unit-testable while making it impossible for this layer to read target
    truth.  A link exactly at the range boundary is valid; a zero-length pair
    is rejected as a duplicated vehicle pose rather than silently treated as a
    perfect communication edge.
    """

    if not math.isfinite(maximum_range_m) or maximum_range_m <= 0.0:
        raise ValueError("maximum relay range must be finite and positive")
    if not callable(line_of_sight_clear):
        raise TypeError("line_of_sight_clear must be callable")
    points = tuple(
        _point(position, f"positions[{index}]") for index, position in enumerate(positions)
    )
    if not points:
        raise ValueError("relay graph requires at least one vehicle position")

    adjacency = [[False for _ in points] for _ in points]
    for left, source in enumerate(points):
        for right in range(left + 1, len(points)):
            target = points[right]
            distance_m = math.dist(source, target)
            if distance_m <= 1.0e-6:
                raise ValueError("relay graph does not accept duplicate vehicle positions")
            if distance_m <= maximum_range_m and bool(line_of_sight_clear(source, target)):
                adjacency[left][right] = True
                adjacency[right][left] = True

    components: list[tuple[int, ...]] = []
    remaining = set(range(len(points)))
    while remaining:
        root = min(remaining)
        queue: deque[int] = deque((root,))
        component: list[int] = []
        remaining.remove(root)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor, linked in enumerate(adjacency[current]):
                if linked and neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda component: component[0])

    fully_connected = len(components) == 1
    maximum_hops: int | None = 0 if len(points) == 1 else None
    if fully_connected:
        maximum_hops = 0
        for source in range(len(points)):
            hops = [-1] * len(points)
            hops[source] = 0
            queue = deque((source,))
            while queue:
                current = queue.popleft()
                for neighbor, linked in enumerate(adjacency[current]):
                    if linked and hops[neighbor] < 0:
                        hops[neighbor] = hops[current] + 1
                        queue.append(neighbor)
            maximum_hops = max(maximum_hops, max(hops))

    frozen_adjacency = tuple(tuple(row) for row in adjacency)
    return RelayGraphSnapshot(
        adjacency=frozen_adjacency,
        components=tuple(components),
        direct_link_count=sum(
            frozen_adjacency[left][right]
            for left in range(len(points))
            for right in range(left + 1, len(points))
        ),
        fully_relay_connected=fully_connected,
        maximum_relay_hops=maximum_hops,
    )


@dataclass(frozen=True, slots=True)
class RelayMessage:
    """One public, source-timestamped shared-state update.

    A payload digest is intentionally used in place of a payload object.  The
    mission runtime owns public mapping content; the network layer neither
    interprets it nor has a path to evaluator-private target truth.
    """

    message_id: str
    sender_id: str
    source_timestamp_s: float
    payload_digest: str
    time_to_live_s: float

    def __post_init__(self) -> None:
        for name in ("message_id", "sender_id"):
            require_identifier(getattr(self, name), name)
        issued = finite_number(self.source_timestamp_s, "source_timestamp_s")
        ttl = finite_number(self.time_to_live_s, "time_to_live_s")
        if issued < 0.0 or ttl <= 0.0:
            raise ValueError("message timestamps must be non-negative and TTL positive")
        if len(self.payload_digest) != 64:
            raise ValueError("payload_digest must be a SHA-256 digest")
        object.__setattr__(self, "source_timestamp_s", issued)
        object.__setattr__(self, "time_to_live_s", ttl)


@dataclass(frozen=True, slots=True)
class RelayDelivery:
    """Auditable outcome for one recipient, including age and hop route."""

    message_id: str
    sender_id: str
    receiver_id: str
    source_timestamp_s: float
    delivered_timestamp_s: float
    relay_path_agent_indices: tuple[int, ...]
    payload_digest: str

    def __post_init__(self) -> None:
        for name in ("message_id", "sender_id", "receiver_id"):
            require_identifier(getattr(self, name), name)
        source = finite_number(self.source_timestamp_s, "source_timestamp_s")
        delivered = finite_number(self.delivered_timestamp_s, "delivered_timestamp_s")
        if source < 0.0 or delivered < source:
            raise ValueError("message delivery timestamps are invalid")
        path = tuple(self.relay_path_agent_indices)
        if len(path) < 2 or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in path
        ):
            raise ValueError("relay delivery requires a non-empty integer path")
        if len(self.payload_digest) != 64:
            raise ValueError("payload_digest must be a SHA-256 digest")
        object.__setattr__(self, "source_timestamp_s", source)
        object.__setattr__(self, "delivered_timestamp_s", delivered)
        object.__setattr__(self, "relay_path_agent_indices", path)

    @property
    def age_seconds(self) -> float:
        return self.delivered_timestamp_s - self.source_timestamp_s

    def to_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "source_timestamp_s": self.source_timestamp_s,
            "delivered_timestamp_s": self.delivered_timestamp_s,
            "age_seconds": self.age_seconds,
            "relay_path_agent_indices": list(self.relay_path_agent_indices),
            "payload_digest": self.payload_digest,
        }


@dataclass(frozen=True, slots=True)
class RelayMessageOutcome:
    """A delivered, expired, or deterministically dropped recipient attempt."""

    message_id: str
    sender_id: str
    receiver_id: str
    status: str
    timestamp_s: float
    reason: str
    delivery: RelayDelivery | None = None

    def __post_init__(self) -> None:
        for name in ("message_id", "sender_id", "receiver_id", "reason"):
            require_identifier(getattr(self, name), name)
        if self.status not in {"DELIVERED", "DROPPED", "EXPIRED"}:
            raise ValueError("unsupported relay message outcome")
        timestamp = finite_number(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("outcome timestamp must be non-negative")
        if (self.status == "DELIVERED") != (self.delivery is not None):
            raise ValueError("only delivered outcomes may contain a delivery")
        if self.delivery is not None and (
            self.delivery.message_id != self.message_id
            or self.delivery.sender_id != self.sender_id
            or self.delivery.receiver_id != self.receiver_id
        ):
            raise ValueError("delivery identity does not match its outcome")
        object.__setattr__(self, "timestamp_s", timestamp)


@dataclass(slots=True)
class RelayMessageQueue:
    """Deterministic range/LOS relay delivery with delay, loss and stale-age proof.

    A queued update becomes deliverable only after the frozen latency and only
    when a currently measured relay route exists.  If the graph remains
    partitioned until its TTL expires, the failure is recorded rather than
    inferred away.  This is a simple packet-level contract, not an RF claim.
    """

    agent_ids: tuple[str, ...]
    base_latency_s: float
    per_hop_latency_s: float
    loss_probability: float
    _pending: list[RelayMessage] = field(default_factory=list, init=False, repr=False)
    _outcomes: list[RelayMessageOutcome] = field(default_factory=list, init=False, repr=False)
    _created: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        identifiers = tuple(self.agent_ids)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("message queue needs unique non-empty agent identifiers")
        for agent_id in identifiers:
            require_identifier(agent_id, "agent_id")
        for name in ("base_latency_s", "per_hop_latency_s", "loss_probability"):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.loss_probability > 1.0:
            raise ValueError("loss_probability must be in [0, 1]")
        self.agent_ids = identifiers

    def publish(self, message: RelayMessage) -> None:
        if message.sender_id not in self.agent_ids:
            raise ValueError("message sender is not in the relay fleet")
        for receiver in self.agent_ids:
            if receiver == message.sender_id:
                continue
            key = (message.message_id, receiver)
            if key in self._created:
                raise ValueError("message ID has already been published")
            self._created.add(key)
        self._pending.append(message)

    def _is_dropped(self, message: RelayMessage, receiver_id: str) -> bool:
        digest = canonical_sha256(
            {
                "message_id": message.message_id,
                "receiver_id": receiver_id,
                "payload_digest": message.payload_digest,
            }
        )
        draw = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        return draw < self.loss_probability

    def advance(
        self, *, timestamp_s: float, graph: RelayGraphSnapshot
    ) -> tuple[RelayMessageOutcome, ...]:
        """Advance the queue against the current measured relay graph."""

        timestamp = finite_number(timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        if len(graph.adjacency) != len(self.agent_ids):
            raise ValueError("relay graph size differs from queue fleet")
        emitted: list[RelayMessageOutcome] = []
        still_pending: list[RelayMessage] = []
        known_outcomes = {(row.message_id, row.receiver_id) for row in self._outcomes}
        for message in self._pending:
            undelivered = False
            sender_index = self.agent_ids.index(message.sender_id)
            for receiver_index, receiver_id in enumerate(self.agent_ids):
                if (
                    receiver_id == message.sender_id
                    or (
                        message.message_id,
                        receiver_id,
                    )
                    in known_outcomes
                ):
                    continue
                if timestamp > message.source_timestamp_s + message.time_to_live_s + 1.0e-12:
                    outcome = RelayMessageOutcome(
                        message.message_id,
                        message.sender_id,
                        receiver_id,
                        "EXPIRED",
                        timestamp,
                        "ttl_elapsed_without_delivery",
                    )
                elif self._is_dropped(message, receiver_id):
                    outcome = RelayMessageOutcome(
                        message.message_id,
                        message.sender_id,
                        receiver_id,
                        "DROPPED",
                        timestamp,
                        "deterministic_loss_draw",
                    )
                else:
                    path = graph.shortest_relay_path(sender_index, receiver_index)
                    minimum_delivery = (
                        message.source_timestamp_s
                        + self.base_latency_s
                        + self.per_hop_latency_s * (len(path) - 1 if path else 0)
                    )
                    if path is None or timestamp + 1.0e-12 < minimum_delivery:
                        undelivered = True
                        continue
                    delivery = RelayDelivery(
                        message.message_id,
                        message.sender_id,
                        receiver_id,
                        message.source_timestamp_s,
                        timestamp,
                        path,
                        message.payload_digest,
                    )
                    outcome = RelayMessageOutcome(
                        message.message_id,
                        message.sender_id,
                        receiver_id,
                        "DELIVERED",
                        timestamp,
                        "measured_relay_path_after_latency",
                        delivery,
                    )
                emitted.append(outcome)
                known_outcomes.add((message.message_id, receiver_id))
            if undelivered:
                still_pending.append(message)
        self._pending = still_pending
        self._outcomes.extend(emitted)
        return tuple(emitted)

    @property
    def outcomes(self) -> tuple[RelayMessageOutcome, ...]:
        return tuple(self._outcomes)

    @property
    def pending_recipient_count(self) -> int:
        """Count recipient deliveries still unresolved at the episode boundary."""

        resolved = {(row.message_id, row.receiver_id) for row in self._outcomes}
        return sum(
            1
            for message in self._pending
            for receiver_id in self.agent_ids
            if receiver_id != message.sender_id
            and (message.message_id, receiver_id) not in resolved
        )

    def finalize_episode(self, *, timestamp_s: float) -> tuple[RelayMessageOutcome, ...]:
        """Record every unresolved recipient as an episode-horizon failure."""

        timestamp = finite_number(timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        resolved = {(row.message_id, row.receiver_id) for row in self._outcomes}
        emitted: list[RelayMessageOutcome] = []
        for message in self._pending:
            for receiver_id in self.agent_ids:
                if (
                    receiver_id == message.sender_id
                    or (
                        message.message_id,
                        receiver_id,
                    )
                    in resolved
                ):
                    continue
                emitted.append(
                    RelayMessageOutcome(
                        message.message_id,
                        message.sender_id,
                        receiver_id,
                        "EXPIRED",
                        timestamp,
                        "episode_horizon_elapsed_without_delivery",
                    )
                )
        self._pending = []
        self._outcomes.extend(emitted)
        return tuple(emitted)

    def stale_age_seconds(
        self, *, receiver_id: str, payload_digest: str, now_s: float
    ) -> float | None:
        """Return the age of the newest delivered public payload at ``now_s``."""

        require_identifier(receiver_id, "receiver_id")
        now = finite_number(now_s, "now_s")
        if receiver_id not in self.agent_ids or now < 0.0 or len(payload_digest) != 64:
            raise ValueError("invalid stale-belief query")
        source_times = [
            row.delivery.source_timestamp_s
            for row in self._outcomes
            if row.status == "DELIVERED"
            and row.delivery is not None
            and row.receiver_id == receiver_id
            and row.delivery.payload_digest == payload_digest
        ]
        return None if not source_times else now - max(source_times)


__all__ = [
    "LineOfSight",
    "Point3",
    "RelayDelivery",
    "RelayGraphSnapshot",
    "RelayMessage",
    "RelayMessageOutcome",
    "RelayMessageQueue",
    "build_range_los_relay_graph",
]
