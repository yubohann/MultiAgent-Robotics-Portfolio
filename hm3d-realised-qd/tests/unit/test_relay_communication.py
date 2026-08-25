from __future__ import annotations

import hashlib

import pytest

from aerocity_method.runtime.communication import (
    RelayMessage,
    RelayMessageQueue,
    build_range_los_relay_graph,
)


def _payload_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_relay_graph_connects_distant_agents_through_a_measured_intermediate() -> None:
    observed_pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    def clear(source, target) -> bool:
        observed_pairs.append((source, target))
        return True

    graph = build_range_los_relay_graph(
        ((0.0, 0.0, 0.0), (8.0, 0.0, 0.0), (16.0, 0.0, 0.0)),
        maximum_range_m=10.0,
        line_of_sight_clear=clear,
    )

    assert graph.direct_link_count == 2
    assert graph.components == ((0, 1, 2),)
    assert graph.fully_relay_connected
    assert graph.maximum_relay_hops == 2
    assert graph.agent_relay_reachable_to_all == (True, True, True)
    assert len(observed_pairs) == 2  # The out-of-range pair is never queried.


def test_relay_graph_reports_a_los_partition_without_global_knowledge() -> None:
    graph = build_range_los_relay_graph(
        ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        maximum_range_m=6.0,
        line_of_sight_clear=lambda source, target: source[0] < 1.0,
    )

    assert graph.direct_link_count == 1
    assert graph.components == ((0, 1), (2,))
    assert not graph.fully_relay_connected
    assert graph.maximum_relay_hops is None
    assert graph.agent_relay_reachable_to_all == (False, False, False)


def test_relay_message_waits_for_latency_and_a_measured_relay_path() -> None:
    disconnected = build_range_los_relay_graph(
        ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
        maximum_range_m=10.0,
        line_of_sight_clear=lambda _source, _target: True,
    )
    connected = build_range_los_relay_graph(
        ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        maximum_range_m=10.0,
        line_of_sight_clear=lambda _source, _target: True,
    )
    queue = RelayMessageQueue(("uav0", "uav1"), 0.2, 0.1, 0.0)
    digest = _payload_digest("public-range-map")
    queue.publish(RelayMessage("message0", "uav0", 0.0, digest, 2.0))
    assert not queue.advance(timestamp_s=0.1, graph=connected)
    assert not queue.advance(timestamp_s=0.3, graph=disconnected)
    outcomes = queue.advance(timestamp_s=0.4, graph=connected)
    assert len(outcomes) == 1
    assert outcomes[0].status == "DELIVERED"
    assert outcomes[0].delivery is not None
    assert outcomes[0].delivery.relay_path_agent_indices == (0, 1)
    assert queue.stale_age_seconds(receiver_id="uav1", payload_digest=digest, now_s=1.0) == 1.0


def test_relay_message_records_timeout_and_deterministic_loss_instead_of_erasing_them() -> None:
    graph = build_range_los_relay_graph(
        ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
        maximum_range_m=10.0,
        line_of_sight_clear=lambda _source, _target: True,
    )
    queue = RelayMessageQueue(("uav0", "uav1"), 0.0, 0.0, 0.0)
    queue.publish(RelayMessage("message0", "uav0", 0.0, _payload_digest("map0"), 0.5))
    outcomes = queue.advance(timestamp_s=0.6, graph=graph)
    assert outcomes[0].status == "EXPIRED"
    loss_queue = RelayMessageQueue(("uav0", "uav1"), 0.0, 0.0, 1.0)
    loss_queue.publish(RelayMessage("message1", "uav0", 0.0, _payload_digest("map1"), 1.0))
    dropped = loss_queue.advance(timestamp_s=0.0, graph=graph)
    assert dropped[0].status == "DROPPED"


def test_relay_message_finalization_preserves_unresolved_episode_denominator() -> None:
    graph = build_range_los_relay_graph(
        ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
        maximum_range_m=10.0,
        line_of_sight_clear=lambda _source, _target: True,
    )
    queue = RelayMessageQueue(("uav0", "uav1"), 1.0, 0.0, 0.0)
    queue.publish(RelayMessage("message2", "uav0", 0.0, _payload_digest("map2"), 10.0))
    assert not queue.advance(timestamp_s=0.1, graph=graph)
    assert queue.pending_recipient_count == 1
    final = queue.finalize_episode(timestamp_s=0.2)
    assert final[0].status == "EXPIRED"
    assert final[0].reason == "episode_horizon_elapsed_without_delivery"
    assert queue.pending_recipient_count == 0


@pytest.mark.parametrize(
    ("positions", "maximum_range_m", "message"),
    [
        ((), 1.0, "at least one"),
        (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 1.0, "duplicate"),
        (((0.0, 0.0, 0.0),), 0.0, "positive"),
    ],
)
def test_relay_graph_rejects_invalid_public_geometry(positions, maximum_range_m, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_range_los_relay_graph(
            positions,
            maximum_range_m=maximum_range_m,
            line_of_sight_clear=lambda _source, _target: True,
        )
