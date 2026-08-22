from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.public_boundary import assert_public_fields, validate_public_task_spec


def _public_task() -> dict:
    contract = {
        "canonical_profile": "G1_occupancy_voxel",
        "formal_execution_level": "L1",
        "control_period_s": 0.2,
        "planning_deadline_s": 0.1,
        "planning": {
            "schema": "org.aerocity.bench.planning-cadence.v1",
            "mode": "fixed-rate-with-public-events",
            "period_s": 1.0,
            "event_triggers": [
                "anonymous_confirmation",
                "safety_intervention",
                "fleet_roster_change",
                "return_reserve_entry",
            ],
            "held_action_rebinding": "latest-public-observation",
            "retransmit_messages_on_hold": False,
        },
        "clock": {
            "basis": "simulated_execution_plus_compute_overrun",
            "overrun_policy": "hold_last_safe_or_hover",
            "max_consecutive_deadline_misses": 1,
        },
        "episode": {"duration_s": 30.0, "return_reserve_s": 5.0},
        "observe": {
            "exclusive_with_translation": True,
            "continuous_dwell_s": 0.5,
            "cooldown_s": 0.5,
            "max_linear_speed_mps": 0.1,
            "max_angular_speed_deg_s": 1.0,
            "max_pose_drift_m": 0.1,
            "source_freshness_s": 0.1,
            "max_range_m": 3.0,
            "horizontal_fov_deg": 60.0,
            "vertical_fov_deg": 45.0,
            "surface_facing_min_cosine": 0.5,
        },
        "sensor_rig": {
            "translation_body_m": [0.0, 0.0, 0.0],
            "forward_axis": "+X",
            "up_axis": "+Z",
            "gimbal_mode": "fixed",
        },
        "vehicle": {
            "radius_m": 0.3,
            "horizontal_speed_mps": 1.0,
            "vertical_speed_mps": 1.0,
            "acceleration_mps2": 1.0,
            "yaw_rate_deg_s": 30.0,
            "minimum_clearance_m": 0.2,
            "home_radius_m": 1.0,
            "energy_budget_j": 4000.0,
            "energy_per_meter_j": 1.0,
            "hover_power_w": 100.0,
        },
        "communication": {
            "range_m": 10.0,
            "bandwidth_bytes_s": 10.0,
            "payload_bytes": 1.0,
            "latency_s": 0.1,
            "drop_probability": 0.0,
            "ttl_s": 1.0,
        },
        "safety": {
            "hard_collision_agent_terminal": True,
            "hard_collision_rank_ineligible": True,
            "out_of_bounds_policy": "reject_and_hover",
            "max_out_of_bounds_actions": 1,
        },
    }
    task = {
        "schema": "org.aerocity.bench.task-spec-public.ordinary.v1",
        "layout_id": "city-0123456789abcdef",
        "execution_contract": contract,
        "public_execution_contract_hash": content_hash(contract),
        "target_count_public": False,
        "target_process_public": False,
        "formal_split_label_public": False,
    }
    task["task_spec_hash"] = content_hash(task)
    return task


def test_public_boundary_accepts_explicit_false_sentinels_only() -> None:
    task = _public_task()
    validate_public_task_spec(task)
    assert_public_fields({"target_count_public": False})


@pytest.mark.parametrize("key", ("fixed_target_count_private", "support_site", "split_label"))
def test_public_boundary_rejects_private_semantic_fields(key: str) -> None:
    task = copy.deepcopy(_public_task())
    task["execution_contract"]["episode"][key] = True
    task["task_spec_hash"] = content_hash(
        {field: value for field, value in task.items() if field != "task_spec_hash"}
    )
    with pytest.raises(ValueError, match="forbidden|non-public"):
        validate_public_task_spec(task)


def test_boundary_cli_writes_a_structured_failure_receipt(tmp_path: Path) -> None:
    tool_path = Path(__file__).parents[1] / "tools" / "audit_public_boundary.py"
    spec = importlib.util.spec_from_file_location("audit_public_boundary", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    invalid_layout = tmp_path / "layout"
    public_root = invalid_layout / "method_public"
    episode_root = public_root / "episodes"
    episode_root.mkdir(parents=True)
    invalid_task = _public_task()
    invalid_task["execution_contract"]["episode"]["fixed_target_count_private"] = True
    invalid_task["task_spec_hash"] = content_hash(
        {key: value for key, value in invalid_task.items() if key != "task_spec_hash"}
    )
    (public_root / "task_spec.json").write_text(
        json.dumps(invalid_task), encoding="utf-8"
    )
    (episode_root / "episode-0000.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "audit.json"

    assert module.main(["--layout-root", str(invalid_layout), "--output", str(output)]) == 2
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["formal_score_eligible"] is False
