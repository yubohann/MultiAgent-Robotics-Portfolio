from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.cf2x_fixture_aggregate import (
    aggregate_closed_private_fixtures,
    write_closed_private_fixture_aggregate,
)
from aerocity_bench.errors import ValidationError


def _report(label: str) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight.v4",
        "formal_score_eligible": False,
        "evidence_scope": "cf2x_internal_shared_world_fleet_preflight",
        "not_a_formal_l1_episode": True,
        "execution_mode": "private-witness-fixture",
        "private_evaluator_commitment": content_hash({"evaluator": label}),
        "private_fixture_commitment": content_hash({"fixture": label}),
        "execution": {
            "control_ticks": 100,
            "shared_physx_step_count": 1200,
            "simulated_time_s": 42.0,
            "wall_clock_s": 11.0,
        },
        "policy_progress": {
            "status": "PRIVATE_FIXTURE_CLOSED",
            "observe_action_count": 4,
            "confirmation_receipt_count": 1,
            "return_action_count": 49,
            "all_returned_home": True,
            "episode_budget_completed": False,
        },
        "final": {
            "safe_completion": True,
            "collision_detected": False,
            "out_of_bounds_detected": False,
            "all_returned_home": True,
        },
    }
    report["public_report_sha256"] = content_hash(report)
    return report


def _write_reports(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label in ("calibration", "train", "validation"):
        path = tmp_path / f"{label}.public.json"
        write_json(path, _report(label))
        paths[label] = path
    return paths


def test_public_fixture_aggregate_is_closed_nonformal_and_path_free(tmp_path: Path) -> None:
    paths = _write_reports(tmp_path)
    aggregate = aggregate_closed_private_fixtures(paths)

    assert aggregate["result"] == "ALL_THREE_DEVELOPMENT_FIXTURES_CLOSED"
    assert aggregate["formal_score_eligible"] is False
    assert aggregate["totals"] == {
        "control_ticks": 300,
        "shared_physx_step_count": 3600,
        "observe_action_count": 12,
        "confirmation_receipt_count": 3,
        "return_action_count": 147,
    }
    serialized = str(aggregate).lower()
    assert str(tmp_path).lower() not in serialized
    assert "target_position" not in serialized
    assert "witness_id" not in serialized
    assert content_hash(
        {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    ) == aggregate["aggregate_sha256"]

    output = tmp_path / "aggregate.json"
    written = write_closed_private_fixture_aggregate(paths, output)
    assert read_json(output) == written
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        write_closed_private_fixture_aggregate(paths, output)


def test_public_fixture_aggregate_rejects_leak_reused_evidence_and_open_status(
    tmp_path: Path,
) -> None:
    paths = _write_reports(tmp_path)
    leaking = deepcopy(_report("calibration"))
    leaking["target_position"] = [1.0, 2.0, 3.0]
    leaking["public_report_sha256"] = content_hash(
        {key: value for key, value in leaking.items() if key != "public_report_sha256"}
    )
    write_json(paths["calibration"], leaking)
    with pytest.raises(ValidationError, match="leaks private"):
        aggregate_closed_private_fixtures(paths)

    paths = _write_reports(tmp_path / "distinct")
    paths["train"] = paths["calibration"]
    with pytest.raises(ValidationError, match="paths must be distinct"):
        aggregate_closed_private_fixtures(paths)

    paths = _write_reports(tmp_path / "incomplete")
    incomplete = deepcopy(_report("validation"))
    progress = incomplete["policy_progress"]
    assert isinstance(progress, dict)
    progress["status"] = "PRIVATE_FIXTURE_INCOMPLETE"
    incomplete["public_report_sha256"] = content_hash(
        {key: value for key, value in incomplete.items() if key != "public_report_sha256"}
    )
    write_json(paths["validation"], incomplete)
    with pytest.raises(ValidationError, match="did not close"):
        aggregate_closed_private_fixtures(paths)
