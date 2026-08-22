from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_bench.sensor_profiles import (
    G1_L1_GEOMETRY,
    P1_L2_RGBD_INSTANCE_OBSERVE_160X120,
    P1_L2_RGBD_OBSERVE_96X69,
    render_due,
    validate_single_sensor_profile,
)


def test_geometry_profile_carries_no_rendered_stream_but_has_a_stable_fingerprint() -> None:
    G1_L1_GEOMETRY.validate()
    assert G1_L1_GEOMETRY.capability_profile == "G1"
    assert G1_L1_GEOMETRY.fingerprint == G1_L1_GEOMETRY.fingerprint
    assert (
        render_due(
            G1_L1_GEOMETRY,
            action_kind="OBSERVE",
            dwell_elapsed_s=0.0,
            last_render_elapsed_s=None,
        )
        is False
    )


def test_rendered_l2_profile_is_event_triggered_and_limited_to_observe_dwell() -> None:
    profile = P1_L2_RGBD_OBSERVE_96X69
    assert (
        render_due(
            profile,
            action_kind="WAYPOINT",
            dwell_elapsed_s=0.0,
            last_render_elapsed_s=None,
        )
        is False
    )
    assert render_due(
        profile,
        action_kind="OBSERVE",
        dwell_elapsed_s=0.0,
        last_render_elapsed_s=None,
    )
    assert not render_due(
        profile,
        action_kind="OBSERVE",
        dwell_elapsed_s=0.49,
        last_render_elapsed_s=0.0,
    )
    assert render_due(
        profile,
        action_kind="OBSERVE",
        dwell_elapsed_s=0.5,
        last_render_elapsed_s=0.0,
    )


def test_result_batch_rejects_sensor_resolution_or_capability_drift() -> None:
    assert validate_single_sensor_profile([P1_L2_RGBD_OBSERVE_96X69] * 2)
    with pytest.raises(ValueError, match="may not mix"):
        validate_single_sensor_profile(
            [P1_L2_RGBD_OBSERVE_96X69, P1_L2_RGBD_INSTANCE_OBSERVE_160X120]
        )
    with pytest.raises(ValueError, match="G1 geometry"):
        replace(G1_L1_GEOMETRY, rgb_enabled=True).validate()
