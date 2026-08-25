from __future__ import annotations

import argparse
import contextlib
import types
from copy import deepcopy
from pathlib import Path

import pytest

from aerocity_bench.behavioral_distinctness import (
    audit_method_panel_behavior_cohort,
    summarize_public_action_trace,
)
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.cf2x_fleet_preflight_contract import COMPLETE_CALIBRATION_PURPOSE
from aerocity_bench.cf2x_l0_pairing_contract import (
    L0_PAIRING_SCHEMA,
    L0_PAIRING_SCOPE,
    SHARED_BINDING_FIELDS,
    l0_pair_record_evidence_hash,
    private_evaluator_commitment,
)
from aerocity_bench.contracts import ActionPacket, Pose3D
from tools.build_cf2x_b_gate_manifest import build_manifest
from tools.run_cf2x_b_gate_replays import (
    _archive_retryable_attempt,
    _archived_attempt_authorizes_retry,
    _current_evidence_pipeline_bindings,
    _is_retryable_host_failure,
    build_replay_plan,
)
from tools.verify_cf2x_b_gate import (
    A_GATE_SCHEMA,
    B_GATE_MANIFEST_SCHEMA,
    B_GATE_MANIFEST_SCHEMA_V2,
    RANKING_METHODS,
    _load_censored_attempt_records,
    _load_l0_pairing_records,
    verify_b_gate,
)


def _behavior_audit_path(tmp_path: Path) -> Path:
    reports = []
    for context_index in range(3):
        for method in RANKING_METHODS:
            route_x = (
                2.0 + context_index
                if method in {"atlas-surface-inspector", "atlas-region-greedy"}
                else 20.0 + context_index
            )
            action = ActionPacket(
                episode_id=f"episode-{context_index}",
                drone_id="drone-000",
                sequence=0,
                issued_at_s=0.0,
                kind="WAYPOINT",
                waypoint=Pose3D((route_x, 0.0, 2.0), 0.0),
            )
            reports.append(
                {
                    "formal_score_eligible": False,
                    "method_id": method,
                    "layout_hash": f"{context_index + 1:064x}",
                    "episode_hash": f"{context_index + 11:064x}",
                    "replicates": [
                        {
                            "public_action_behavior": summarize_public_action_trace(
                                [{"drone-000": action}]
                            )
                        }
                    ],
                }
            )
    audit = audit_method_panel_behavior_cohort(reports)
    path = tmp_path / "behavior-audit.json"
    write_json(path, audit)
    return path


def _v2_preflight() -> dict[str, object]:
    preflight: dict[str, object] = {
        "schema": "org.aerocity.bench.cf2x-behavior-preflight-binding.v1",
        "audit_report_hash": "a" * 64,
        "audit_file_sha256": "b" * 64,
        "context_count": 3,
        "candidate_method_ids": list(RANKING_METHODS),
        "mechanism_groups": [
            ["atlas-region-greedy", "atlas-surface-inspector"],
            ["sweep-3d"],
        ],
        "l1_representative_method_ids": ["atlas-region-greedy", "sweep-3d"],
        "excluded_redundant_method_ids": ["atlas-surface-inspector"],
        "candidate_methods_are_not_deleted": True,
        "redundant_methods_do_not_count_as_independent_mechanisms": True,
    }
    preflight["binding_hash"] = content_hash(preflight)
    return preflight


def _inputs() -> tuple[dict, dict, list[dict], list[dict]]:
    a_gate = {
        "schema": A_GATE_SCHEMA,
        "status": "VERIFIED",
        "authorizes_next_gate": True,
        "report_hash": "1" * 64,
    }
    manifest = {
        "schema": B_GATE_MANIFEST_SCHEMA,
        "formal_score_eligible": False,
        "a_gate_report_hash": "1" * 64,
        "selection_policy": "sorted-ancestor-even-quantiles-v1",
        "precommitted_before_replays": True,
        "method_ids": list(RANKING_METHODS),
        "layout_ancestors": ["ancestor-01", "ancestor-02", "ancestor-03"],
        "expected_input_bindings": {
            "baseline_source_sha256": "0" * 64,
            "geometry_source_sha256": "1" * 64,
            "controller_spec_hash": "f" * 64,
            "cf2x_usd_sha256": "c" * 64,
            "release_config_sha256": "a" * 64,
        },
        "report_hash": "2" * 64,
    }
    ancestors = ("ancestor-01", "ancestor-02", "ancestor-03")
    global_bindings = {
        "release_config_sha256": "a" * 64,
        "execution_contract_hash": "b" * 64,
        "cf2x_usd_sha256": "c" * 64,
        "cf2x_schema_sha256": "d" * 64,
        "dynamics_spec_hash": "e" * 64,
        "controller_spec_hash": "f" * 64,
        "baseline_source_sha256": "0" * 64,
        "geometry_source_sha256": "1" * 64,
    }
    replay_records = []
    l0_records = []
    for ancestor_index, ancestor in enumerate(ancestors):
        ancestor_bindings = {
            "layout_hash": f"{ancestor_index + 1:x}" * 64,
            "stage_sha256": f"{ancestor_index + 4:x}" * 64,
            "cityspec_sha256": f"{ancestor_index + 7:x}" * 64,
            "task_spec_sha256": "9" * 64,
            "task_spec_hash": "8" * 64,
            "public_episode_sha256": f"{ancestor_index + 10:x}" * 64,
            "mission_sector_hash": f"{ancestor_index + 13:x}" * 64,
            "atlas_hash": f"{ancestor_index + 2:x}" * 64,
        }
        for method_index, method in enumerate(RANKING_METHODS):
            confirmation_count = method_index + (1 if ancestor_index == 0 else 0)
            report_hash = content_hash(["L1", method, ancestor])
            replay_records.append(
                {
                    "layout_ancestor": ancestor,
                    "method_id": method,
                    "public_report_file_sha256": report_hash,
                    "private_report_file_sha256": content_hash(["private", method, ancestor]),
                    "host_guard_report_file_sha256": content_hash(["host-guard", method, ancestor]),
                    "host_guard_passed": True,
                    "formal_score_eligible": False,
                    "execution_purpose": COMPLETE_CALIBRATION_PURPOSE,
                    "complete_calibration_replay": True,
                    "method": method,
                    "input_bindings": {**global_bindings, **ancestor_bindings},
                    "policy_progress": {
                        "status": "CALIBRATION_EPISODE_CLOSED",
                        "confirmation_receipt_count": confirmation_count,
                    },
                    "planning_timing": {"deadline_miss_tick_count": 0},
                    "execution": {"failure_record_count": 0},
                    "final": {
                        "safe_completion": True,
                        "all_returned_home": True,
                    },
                }
            )
            l0_records.append(
                {
                    "method_id": method,
                    "layout_ancestor": ancestor,
                    "score": float(confirmation_count),
                    "execution_level": "L0",
                    "evidence_hash": content_hash(["L0", method, ancestor]),
                }
            )
    return a_gate, manifest, replay_records, l0_records


def _v2_inputs() -> tuple[dict, dict, list[dict], list[dict]]:
    a_gate, manifest, replay_records, l0_records = _inputs()
    representatives = ("atlas-region-greedy", "sweep-3d")
    manifest["schema"] = B_GATE_MANIFEST_SCHEMA_V2
    manifest["method_ids"] = list(representatives)
    manifest["behavior_preflight"] = _v2_preflight()
    replay_records = [
        record for record in replay_records if record["method_id"] in representatives
    ]
    l0_records = [record for record in l0_records if record["method_id"] in representatives]
    return a_gate, manifest, replay_records, l0_records


def test_b_gate_verifies_complete_bound_three_by_three_panel() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
    )
    assert report["status"] == "VERIFIED"
    assert report["failure_count"] == 0
    assert report["replay_count"] == 9
    assert report["fidelity_audit"]["status"] == "MEASURED_NOT_FROZEN"
    assert report["l0_screening"]["screening_authorized"] is True
    assert report["report_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )


def test_b_gate_v2_saves_redundant_replays_without_inventing_method_diversity() -> None:
    a_gate, manifest, replay_records, l0_records = _v2_inputs()
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
    )
    assert len(replay_records) == 6
    assert report["status"] == "NO_GO"
    assert report["checks"][
        "complete_representative_method_by_three_ancestor_matrix"
    ] is True
    assert report["checks"][
        "two_distinct_search_mechanisms_have_nonzero_l1_signal"
    ] is False
    assert report["distinct_search_mechanism_count"] == 1
    assert report["mechanism_groups"] == _v2_preflight()["mechanism_groups"]


def test_b_gate_v2_rejects_tampered_representative_mapping() -> None:
    a_gate, manifest, replay_records, l0_records = _v2_inputs()
    manifest["method_ids"] = ["atlas-surface-inspector", "sweep-3d"]
    with pytest.raises(ValueError, match="behavior preflight binding"):
        verify_b_gate(
            a_gate=a_gate,
            manifest=manifest,
            replay_records=replay_records,
            l0_records=l0_records,
        )

    _, manifest, replay_records, l0_records = _v2_inputs()
    manifest["behavior_preflight"]["binding_hash"] = "0" * 64
    with pytest.raises(ValueError, match="behavior preflight binding"):
        verify_b_gate(
            a_gate=a_gate,
            manifest=manifest,
            replay_records=replay_records,
            l0_records=l0_records,
        )


def test_b_gate_loads_only_a_hashed_l0_pairing_bound_to_its_manifest(tmp_path) -> None:
    _a_gate, manifest, _replay_records, _l0_records = _inputs()
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    records = []
    for ancestor in manifest["layout_ancestors"]:
        bindings = {
            field: content_hash([field, ancestor]) for field in SHARED_BINDING_FIELDS
        }
        for field in (
            "release_config_sha256",
            "baseline_source_sha256",
            "geometry_source_sha256",
        ):
            bindings[field] = manifest["expected_input_bindings"][field]
        private_episode_sha256 = content_hash(["private", ancestor])
        for method_id in manifest["method_ids"]:
            record = {
                "layout_ancestor": ancestor,
                "method_id": method_id,
                "score": 1.0,
                "execution_level": "L0",
                "input_bindings": bindings,
                "private_episode_sha256": private_episode_sha256,
                "private_evaluator_commitment": private_evaluator_commitment(
                    private_episode_sha256,
                    bindings["layout_hash"],
                    bindings["execution_contract_hash"],
                ),
                "execution": {
                    "all_returned_home": True,
                    "collision_count": 0,
                    "out_of_bounds_actions": 0,
                    "deadline_miss_tick_count": 0,
                    "task_time_s": 300.0,
                },
            }
            record["evidence_hash"] = l0_pair_record_evidence_hash(
                record, manifest["report_hash"]
            )
            records.append(record)
    pairing = {
        "schema": L0_PAIRING_SCHEMA,
        "evidence_scope": L0_PAIRING_SCOPE,
        "formal_score_eligible": False,
        "status": "VERIFIED_L0_PAIRING",
        "b_gate_manifest_report_hash": manifest["report_hash"],
        "b_gate_manifest_file_sha256": file_hash(manifest_path),
        "l0_implementation_hash": content_hash(["implementation"]),
        "layout_ancestors": manifest["layout_ancestors"],
        "method_ids": manifest["method_ids"],
        "expected_input_bindings": manifest["expected_input_bindings"],
        "records": records,
    }
    pairing["report_hash"] = content_hash(pairing)
    pairing_path = tmp_path / "l0-pairing.json"
    write_json(pairing_path, pairing)

    normalized = _load_l0_pairing_records(pairing_path, manifest_path, manifest)

    assert len(normalized) == 9
    expected_fields = {
        "method_id",
        "layout_ancestor",
        "score",
        "execution_level",
        "evidence_hash",
    }
    assert all(set(record) == expected_fields for record in normalized)

    pairing["records"][0]["score"] = 9.0
    write_json(pairing_path, pairing)
    with pytest.raises(ValueError, match="report does not bind|report hash"):
        _load_l0_pairing_records(pairing_path, manifest_path, manifest)


def test_b_gate_rejects_geometry_binding_substitution_after_precommitment() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    for record in replay_records:
        record["input_bindings"]["geometry_source_sha256"] = "2" * 64

    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
    )

    assert report["status"] == "NO_GO"
    assert report["checks"]["global_cf2x_controller_and_contract_bindings_frozen"] is True
    assert report["checks"]["all_replays_match_precommitted_input_bindings"] is False


def test_b_gate_keeps_zero_confirmation_run_but_requires_search_signal() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    for record in replay_records:
        if record["method_id"] == "atlas-surface-inspector":
            record["policy_progress"]["confirmation_receipt_count"] = 0
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
    )
    assert report["status"] == "NO_GO"
    assert report["checks"]["all_public_replays_closed_and_safe"] is True
    assert report["checks"]["two_search_methods_have_nonzero_l1_signal"] is False


def test_b_gate_rejects_cross_ancestor_or_cross_method_evidence_reuse() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    cross_ancestor = deepcopy(replay_records)
    copied = cross_ancestor[0]["input_bindings"]["layout_hash"]
    for record in cross_ancestor:
        if record["layout_ancestor"] == "ancestor-02":
            record["input_bindings"]["layout_hash"] = copied
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=cross_ancestor,
        l0_records=l0_records,
    )
    assert report["checks"]["three_independent_layout_ancestors"] is False

    cross_method = deepcopy(replay_records)
    cross_method[1]["input_bindings"]["mission_sector_hash"] = "0" * 64
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=cross_method,
        l0_records=l0_records,
    )
    assert report["checks"]["paired_ancestor_inputs_frozen"] is False


def test_b_gate_reports_l0_screening_false_negatives_without_dropping_l1() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    victim = next(
        record
        for record in l0_records
        if record["method_id"] == "atlas-region-greedy"
        and record["layout_ancestor"] == "ancestor-01"
    )
    victim["score"] = 0.0
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
    )
    assert report["status"] == "VERIFIED"
    assert report["l0_screening"]["false_negative_pair_count"] == 1
    assert report["l0_screening"]["screening_authorized"] is False
    assert report["l0_screening"]["ranking_disagreement_does_not_invalidate_l1_scores"] is True


def test_b_gate_requires_a_verified_a_gate_and_complete_matrix() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    a_gate["status"] = "NO_GO"
    incomplete = replay_records[:-1]
    incomplete_l0 = [
        record
        for record in l0_records
        if not (
            record["method_id"] == "atlas-region-greedy"
            and record["layout_ancestor"] == "ancestor-03"
        )
    ]
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=incomplete,
        l0_records=incomplete_l0,
    )
    assert report["status"] == "NO_GO"
    assert report["checks"]["a_gate_verified"] is False
    assert report["checks"]["complete_three_method_by_three_ancestor_matrix"] is False


def test_b_gate_rejects_a_nonpassing_host_guard_receipt() -> None:
    a_gate, manifest, replay_records, l0_records = _inputs()
    replay_records[0]["host_guard_passed"] = False
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
    )
    assert report["status"] == "NO_GO"
    assert report["checks"]["all_host_guard_receipts_pass_and_match_precommitted_attempts"] is False


def test_b_gate_manifest_freezes_even_quantile_ancestors_before_replays(tmp_path) -> None:
    from aerocity_bench.canonical import write_json

    source_records = []
    for index in range(5):
        city_name = f"city-{index}.json"
        episode_name = f"episode-{index}.json"
        write_json(tmp_path / city_name, {"layout_id": f"layout-{index}"})
        write_json(tmp_path / episode_name, {"episode_id": f"episode-{index}"})
        source_records.append(
            {
                "city_path": city_name,
                "private_episode_path": episode_name,
                "layout_ancestor": f"ancestor-{index:02d}",
                "split_label": "calibration",
            }
        )
    source = {
        "schema": "org.aerocity.bench.g2-i-scientific-audit-manifest.v1",
        "purpose": "method-independent-task-calibration",
        "records": source_records,
        "manifest_hash": "",
    }
    source["manifest_hash"] = content_hash(
        {key: value for key, value in source.items() if key != "manifest_hash"}
    )
    source_path = tmp_path / "calibration.json"
    write_json(source_path, source)
    a_gate = {
        "schema": "org.aerocity.bench.g2-i-a-gate-freeze.v1",
        "status": "VERIFIED",
        "authorizes_next_gate": True,
        "report_hash": "",
    }
    a_gate["report_hash"] = content_hash(
        {key: value for key, value in a_gate.items() if key != "report_hash"}
    )
    a_gate_path = tmp_path / "a-gate.json"
    write_json(a_gate_path, a_gate)
    manifest_path = tmp_path / "b-manifest.json"
    manifest = build_manifest(a_gate_path, source_path, manifest_path)
    assert manifest["layout_ancestors"] == [
        "ancestor-00",
        "ancestor-02",
        "ancestor-04",
    ]
    assert manifest["method_ids"] == list(RANKING_METHODS)
    assert len(manifest["records"]) == 9
    assert manifest["replay_root"] == "replays"
    assert manifest["runtime_root"] == "runtime"
    assert manifest["report_hash"] == content_hash(
        {key: value for key, value in manifest.items() if key != "report_hash"}
    )


def _b_gate_runner_fixture(tmp_path):
    source_root = tmp_path / "source"
    layouts_root = tmp_path / "layouts"
    selected = []
    records = []
    ancestors = [f"g2-i-calibration-ancestor-{index:02d}" for index in (0, 2, 4)]
    for ancestor in ancestors:
        suffix = ancestor.removeprefix("g2-i-calibration-")
        city = {"layout_id": f"city-{suffix}", "layout_hash": content_hash(suffix)}
        episode = {"episode_id": f"episode-{suffix}", "layout_hash": city["layout_hash"]}
        city_path = source_root / f"{suffix}-city.json"
        episode_path = source_root / f"{suffix}-episode.json"
        write_json(city_path, city)
        write_json(episode_path, episode)
        selected.append(
            {
                "layout_ancestor": ancestor,
                "city_path": city_path.relative_to(source_root).as_posix(),
                "private_episode_path": episode_path.relative_to(source_root).as_posix(),
            }
        )
        layout_bundle = layouts_root / suffix
        layout_root = layout_bundle / "splits" / "calibration" / city["layout_id"]
        layout_root.mkdir(parents=True)
        write_json(
            layout_bundle / "development_layout_manifest.json",
            {
                "formal_score_eligible": False,
                "task_track": "G2-I",
                "inspection_prior_level": "full-cells",
                "private_episode_source": "frozen-calibration-input",
                "split": "calibration",
                "layout_hash": city["layout_hash"],
                "layout_relative_root": layout_root.relative_to(layout_bundle).as_posix(),
                "city_source_sha256": content_hash(city),
                "private_episode_sha256": content_hash(episode),
            },
        )
        for method in RANKING_METHODS:
            stem = f"{ancestor}__{method}"
            records.append(
                {
                    "layout_ancestor": ancestor,
                    "method_id": method,
                    "public_report": f"replays/{stem}.public.json",
                    "private_report": f"replays/{stem}.private.json",
                }
            )
    a_gate = {
        "schema": A_GATE_SCHEMA,
        "status": "VERIFIED",
        "authorizes_next_gate": True,
        "authorizes_formal_test_access": False,
    }
    a_gate["report_hash"] = content_hash(a_gate)
    a_gate_path = tmp_path / "a-gate.json"
    write_json(a_gate_path, a_gate)
    manifest = {
        "schema": B_GATE_MANIFEST_SCHEMA,
        "a_gate_report_hash": a_gate["report_hash"],
        "layout_ancestors": ancestors,
        "method_ids": list(RANKING_METHODS),
        "replay_root": "replays",
        "runtime_root": "runtime",
        "selected_source_inputs": selected,
        "records": records,
    }
    manifest["report_hash"] = content_hash(manifest)
    manifest_path = tmp_path / "b-manifest.json"
    write_json(manifest_path, manifest)
    release_config = tmp_path / "release.json"
    isaac_python = tmp_path / "python.exe"
    cf2x_usd = tmp_path / "cf2x.usd"
    for path in (release_config, isaac_python, cf2x_usd):
        path.write_text("fixture\n", encoding="utf-8")
    return (
        manifest_path,
        a_gate_path,
        source_root,
        layouts_root,
        release_config,
        isaac_python,
        cf2x_usd,
    )


def _b_gate_plan_fixture(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Build a cheap runner fixture while isolating plan tests from boundary parsing.

    Public-layout validation is covered by the dedicated boundary tests and the
    rejection test below.  These tests focus on panel binding, retry and resume
    semantics, so they replace the already-validated boundary call explicitly.
    """

    monkeypatch.setattr(
        "tools.run_cf2x_b_gate_replays.audit_public_layout",
        lambda _layout_root: {"schema": "test.public-boundary", "status": "PASS"},
    )
    return _b_gate_runner_fixture(tmp_path)


def test_b_gate_runner_builds_only_the_precommitted_complete_replay_panel(
    tmp_path, monkeypatch
) -> None:
    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    plan = build_replay_plan(*inputs, device="cuda:0", resume=False)
    assert len(plan) == 9
    assert [(item["layout_ancestor"], item["method_id"]) for item in plan] == [
        (ancestor, method)
        for ancestor in (
            "g2-i-calibration-ancestor-00",
            "g2-i-calibration-ancestor-02",
            "g2-i-calibration-ancestor-04",
        )
        for method in RANKING_METHODS
    ]
    for item in plan:
        command = item["command"]
        assert command[command.index("--execution-mode") + 1] == "public-policy"
        assert command[command.index("--run-purpose") + 1] == COMPLETE_CALIBRATION_PURPOSE
        assert command[command.index("--max-sim-time-s") + 1] == "300"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--headless" in command
        assert item["runtime_root"].parent == inputs[0].parent / "runtime"


def test_b_gate_runner_v2_launches_only_bound_mechanism_representatives(
    tmp_path, monkeypatch
) -> None:
    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    manifest = read_json(inputs[0])
    representatives = {"atlas-region-greedy", "sweep-3d"}
    manifest["schema"] = B_GATE_MANIFEST_SCHEMA_V2
    manifest["method_ids"] = ["atlas-region-greedy", "sweep-3d"]
    manifest["behavior_preflight"] = _v2_preflight()
    manifest["records"] = [
        record for record in manifest["records"] if record["method_id"] in representatives
    ]
    manifest.pop("report_hash")
    manifest["report_hash"] = content_hash(manifest)
    write_json(inputs[0], manifest)

    plan = build_replay_plan(*inputs, device="cuda:0", resume=False)
    assert len(plan) == 6
    assert {item["method_id"] for item in plan} == representatives

    manifest["behavior_preflight"]["binding_hash"] = "0" * 64
    manifest.pop("report_hash")
    manifest["report_hash"] = content_hash(manifest)
    write_json(inputs[0], manifest)
    with pytest.raises(ValueError, match="behavior preflight binding"):
        build_replay_plan(*inputs, device="cuda:0", resume=False)


def test_b_gate_v2_manifest_precommits_an_isolated_runtime_root(tmp_path) -> None:
    from aerocity_bench.canonical import write_json

    source_records = []
    for index in range(3):
        city_path = tmp_path / f"city-{index}.json"
        episode_path = tmp_path / f"episode-{index}.json"
        write_json(city_path, {"layout_id": f"layout-{index}"})
        write_json(episode_path, {"episode_id": f"episode-{index}"})
        source_records.append(
            {
                "city_path": city_path.name,
                "private_episode_path": episode_path.name,
                "layout_ancestor": f"ancestor-{index:02d}",
            }
        )
    source = {
        "schema": "org.aerocity.bench.g2-i-scientific-audit-manifest.v1",
        "records": source_records,
    }
    source["manifest_hash"] = content_hash(source)
    source_path = tmp_path / "calibration.json"
    write_json(source_path, source)
    a_gate = {
        "schema": A_GATE_SCHEMA,
        "status": "VERIFIED",
        "authorizes_next_gate": True,
    }
    a_gate["report_hash"] = content_hash(a_gate)
    a_gate_path = tmp_path / "a-gate.json"
    write_json(a_gate_path, a_gate)

    manifest = build_manifest(
        a_gate_path,
        source_path,
        tmp_path / "manifest-v2.json",
        replay_root="replays-v2",
        expected_controller_spec_hash="a" * 64,
        cf2x_usd_sha256="b" * 64,
        release_config_sha256="c" * 64,
        baseline_source_sha256="d" * 64,
        geometry_source_sha256="e" * 64,
        infrastructure_attempt_limit=3,
        retry_archive_root="censored-v2",
        retry_quiescence_s=0.0,
        retry_max_wait_s=60.0,
        behavior_audit_path=_behavior_audit_path(tmp_path),
    )
    assert manifest["schema"] == B_GATE_MANIFEST_SCHEMA_V2
    assert manifest["method_ids"] == ["atlas-region-greedy", "sweep-3d"]
    assert len(manifest["records"]) == 6
    assert manifest["behavior_preflight"]["candidate_method_ids"] == list(
        RANKING_METHODS
    )
    assert manifest["behavior_preflight"]["excluded_redundant_method_ids"] == [
        "atlas-surface-inspector"
    ]
    assert manifest["replay_root"] == "replays-v2"
    assert manifest["runtime_root"] == "replays-v2-runtime"
    assert manifest["expected_input_bindings"] == {
        "baseline_source_sha256": "d" * 64,
        "geometry_source_sha256": "e" * 64,
        "controller_spec_hash": "a" * 64,
        "cf2x_usd_sha256": "b" * 64,
        "release_config_sha256": "c" * 64,
    }
    assert all(record["public_report"].startswith("replays-v2/") for record in manifest["records"])
    assert manifest["evidence_pipeline_bindings"] == _current_evidence_pipeline_bindings()
    assert "fleet_preflight_source_sha256" in manifest["evidence_pipeline_bindings"]
    assert manifest["infrastructure_censoring_policy"]["max_attempts_per_pair"] == 3


def test_b_gate_runner_rejects_a_layout_without_public_artifacts(tmp_path) -> None:
    inputs = _b_gate_runner_fixture(tmp_path)
    with pytest.raises(ValueError, match="layout lacks a public task spec"):
        build_replay_plan(*inputs, device="cuda:0", resume=False)


def test_b_gate_manifest_requires_geometry_hash_with_other_input_bindings(tmp_path) -> None:
    source_records = []
    for index in range(3):
        city_path = tmp_path / f"city-required-{index}.json"
        episode_path = tmp_path / f"episode-required-{index}.json"
        write_json(city_path, {"layout_id": f"layout-{index}"})
        write_json(episode_path, {"episode_id": f"episode-{index}"})
        source_records.append(
            {
                "city_path": city_path.name,
                "private_episode_path": episode_path.name,
                "layout_ancestor": f"ancestor-{index:02d}",
            }
        )
    source = {
        "schema": "org.aerocity.bench.g2-i-scientific-audit-manifest.v1",
        "records": source_records,
    }
    source["manifest_hash"] = content_hash(source)
    source_path = tmp_path / "calibration-required.json"
    write_json(source_path, source)
    a_gate = {
        "schema": A_GATE_SCHEMA,
        "status": "VERIFIED",
        "authorizes_next_gate": True,
    }
    a_gate["report_hash"] = content_hash(a_gate)
    a_gate_path = tmp_path / "a-gate-required.json"
    write_json(a_gate_path, a_gate)

    with pytest.raises(ValueError, match="all expected B-gate input binding hashes"):
        build_manifest(
            a_gate_path,
            source_path,
            tmp_path / "manifest-missing-geometry.json",
            replay_root="replays-v12",
            expected_controller_spec_hash="a" * 64,
            cf2x_usd_sha256="b" * 64,
            release_config_sha256="c" * 64,
            baseline_source_sha256="d" * 64,
        )


def test_b_gate_runner_rejects_geometry_source_drift_before_native_launch(tmp_path) -> None:
    inputs = list(_b_gate_runner_fixture(tmp_path))
    manifest = read_json(inputs[0])
    manifest["replay_root"] = "replays-v12"
    manifest["runtime_root"] = "replays-v12-runtime"
    for record in manifest["records"]:
        record["public_report"] = record["public_report"].replace(
            "replays/", "replays-v12/"
        )
        record["private_report"] = record["private_report"].replace(
            "replays/", "replays-v12/"
        )
    repository = Path(__file__).resolve().parents[1]
    manifest["expected_input_bindings"] = {
        "baseline_source_sha256": file_hash(
            repository / "src" / "aerocity_bench" / "baselines.py"
        ),
        "geometry_source_sha256": "0" * 64,
        "controller_spec_hash": "a" * 64,
        "cf2x_usd_sha256": file_hash(inputs[6]),
        "release_config_sha256": file_hash(inputs[4]),
    }
    manifest["report_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "report_hash"}
    )
    write_json(inputs[0], manifest)

    with pytest.raises(ValueError, match="geometry_source_sha256 differs"):
        build_replay_plan(*inputs, device="cuda:0", resume=False)


def test_b_gate_runner_rejects_unbound_runtime_root_for_versioned_replays(tmp_path) -> None:
    from aerocity_bench.canonical import read_json, write_json

    inputs = list(_b_gate_runner_fixture(tmp_path))
    manifest = read_json(inputs[0])
    manifest["replay_root"] = "replays-v2"
    manifest.pop("runtime_root")
    for record in manifest["records"]:
        record["public_report"] = record["public_report"].replace("replays/", "replays-v2/")
        record["private_report"] = record["private_report"].replace("replays/", "replays-v2/")
    manifest["report_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "report_hash"}
    )
    write_json(inputs[0], manifest)
    with pytest.raises(ValueError, match="precommitted runtime root"):
        build_replay_plan(*inputs, device="cuda:0", resume=False)


def test_b_gate_runner_rejects_versioned_replay_without_input_bindings(tmp_path) -> None:
    from aerocity_bench.canonical import read_json, write_json

    inputs = list(_b_gate_runner_fixture(tmp_path))
    manifest = read_json(inputs[0])
    manifest["replay_root"] = "replays-v2"
    manifest["runtime_root"] = "replays-v2-runtime"
    for record in manifest["records"]:
        record["public_report"] = record["public_report"].replace("replays/", "replays-v2/")
        record["private_report"] = record["private_report"].replace("replays/", "replays-v2/")
    manifest["report_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "report_hash"}
    )
    write_json(inputs[0], manifest)
    with pytest.raises(ValueError, match="lacks precommitted input bindings"):
        build_replay_plan(*inputs, device="cuda:0", resume=False)


def test_b_gate_runner_rejects_the_prohibited_legacy_airframe_path(tmp_path) -> None:
    inputs = list(_b_gate_runner_fixture(tmp_path))
    prohibited = tmp_path / "assets" / "5_in_drone" / "cf2x.usd"
    prohibited.parent.mkdir(parents=True)
    prohibited.write_text("fixture\n", encoding="utf-8")
    inputs[-1] = prohibited
    with pytest.raises(ValueError, match="prohibited"):
        build_replay_plan(*inputs, device="cuda:0", resume=False)


def _materialize_resumable_replay(tmp_path, *, host_receipt: dict | None) -> tuple:
    from aerocity_bench.canonical import read_json, write_json

    inputs = _b_gate_runner_fixture(tmp_path)
    ancestor = "g2-i-calibration-ancestor-00"
    method = RANKING_METHODS[0]
    stem = f"{ancestor}__{method}"
    write_json(tmp_path / "replays" / f"{stem}.public.json", {})
    write_json(tmp_path / "replays" / f"{stem}.private.json", {})
    if host_receipt is not None:
        manifest = read_json(inputs[0])
        host_receipt.setdefault(
            "evidence_binding",
            content_hash(
                {
                    "manifest_report_hash": manifest["report_hash"],
                    "layout_ancestor": ancestor,
                    "method_id": method,
                }
            ),
        )
        write_json(tmp_path / "runtime" / stem / "host_guard.json", host_receipt)
    return inputs


def _passing_host_receipt() -> dict:
    return {
        "schema": "org.aerocity.bench.isaac-host-guard.v3",
        "status": "PASS",
        "returncode": 0,
        "trigger": None,
        "foreign_runtime_count_before": 0,
        "foreign_runtime_count_after": 0,
    }


def test_b_gate_resume_requires_a_passing_corresponding_host_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tools.run_cf2x_b_gate_replays.audit_public_layout",
        lambda _layout_root: {"schema": "test.public-boundary", "status": "PASS"},
    )
    monkeypatch.setattr(
        "tools.run_cf2x_b_gate_replays._validate_completed_replay",
        lambda *args, **kwargs: None,
    )
    missing_inputs = _materialize_resumable_replay(tmp_path / "missing", host_receipt=None)
    with pytest.raises(FileNotFoundError):
        build_replay_plan(*missing_inputs, device="cuda:0", resume=True)

    receipt = _passing_host_receipt()
    receipt["trigger"] = "residual_runtime"
    receipt["status"] = "FAIL"
    polluted_inputs = _materialize_resumable_replay(tmp_path / "polluted", host_receipt=receipt)
    with pytest.raises(ValueError, match="host guard status"):
        build_replay_plan(*polluted_inputs, device="cuda:0", resume=True)

    valid_inputs = _materialize_resumable_replay(
        tmp_path / "valid", host_receipt=_passing_host_receipt()
    )
    plan = build_replay_plan(*valid_inputs, device="cuda:0", resume=True)
    first = next(item for item in plan if item["layout_ancestor"].endswith("ancestor-00"))
    assert first["status"] == "VERIFIED_EXISTING"


def test_b_gate_prepare_rejects_an_orphaned_failed_runtime_attempt(tmp_path, monkeypatch) -> None:
    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    orphan = tmp_path / "runtime" / "g2-i-calibration-ancestor-00__sweep-3d" / "host_guard.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="cannot be resumed"):
        build_replay_plan(*inputs, device="cuda:0", resume=True)


def _retry_policy(archive_root, *, maximum_attempts: int = 3) -> dict:
    return {
        "schema": "org.aerocity.bench.infrastructure-censoring-policy.v1",
        "selection_rule": "first-host-isolated-complete-attempt",
        "decision_reads_method_outcome": False,
        "max_attempts_per_pair": maximum_attempts,
        "retry_archive_root": "censored-attempts",
        "allowed_host_guard_triggers": (
            "foreign_runtime",
            "foreign_runtime_during_attempt",
            "residual_runtime",
        ),
        "required_quiescence_s": 0.0,
        "maximum_quiescence_wait_s": 60.0,
        "method_or_safety_failure_retry_allowed": False,
        "preserve_every_censored_attempt": True,
        "archive_root": archive_root,
    }


def _failed_host_receipt(binding: str, trigger: str | None) -> dict:
    return {
        "schema": "org.aerocity.bench.isaac-host-guard.v3",
        "status": "FAIL",
        "returncode": None if trigger else 1,
        "trigger": trigger,
        "evidence_binding": binding,
    }


def _archive_item(tmp_path, *, maximum_attempts: int = 3) -> tuple[dict, str]:
    ancestor = "g2-i-calibration-ancestor-00"
    method = "sweep-3d"
    stem = f"{ancestor}__{method}"
    binding = content_hash(["attempt", stem])
    public_path = tmp_path / "replays" / f"{stem}.public.json"
    private_path = tmp_path / "replays" / f"{stem}.private.json"
    failure_path = public_path.with_name(f"{public_path.stem}.failure.json")
    runtime_root = tmp_path / "runtime" / stem
    write_json(public_path, {"partial": True})
    write_json(failure_path, {"failure": "interrupted-by-foreign-runtime"})
    write_json(
        runtime_root / "host_guard.json",
        _failed_host_receipt(binding, "foreign_runtime_during_attempt"),
    )
    (runtime_root / "isaac.log").write_text("foreign runtime detected\n", encoding="utf-8")
    item = {
        "layout_ancestor": ancestor,
        "method_id": method,
        "attempt_binding": binding,
        "public_path": public_path,
        "private_path": private_path,
        "runtime_root": runtime_root,
        "retry_policy": _retry_policy(
            tmp_path / "censored-attempts", maximum_attempts=maximum_attempts
        ),
    }
    return item, stem


def test_infrastructure_censoring_archives_every_artifact_before_retry(tmp_path) -> None:
    item, stem = _archive_item(tmp_path)
    attempt_root = _archive_retryable_attempt(item)
    ledger = read_json(attempt_root / "attempt.json")
    assert attempt_root == tmp_path / "censored-attempts" / stem / "attempt-001"
    assert ledger["retry_authorized"] is True
    assert ledger["decision_reads_method_outcome"] is False
    assert "runtime/host_guard.json" in ledger["artifacts"]
    assert any(name.endswith(".failure.json") for name in ledger["artifacts"])
    assert all(
        file_hash(attempt_root / relative) == digest
        for relative, digest in ledger["artifacts"].items()
    )
    assert not item["runtime_root"].exists()
    assert not item["public_path"].exists()


def test_last_infrastructure_attempt_is_preserved_but_cannot_retry(tmp_path) -> None:
    item, stem = _archive_item(tmp_path, maximum_attempts=2)
    (tmp_path / "censored-attempts" / stem / "attempt-001").mkdir(parents=True)
    attempt_root = _archive_retryable_attempt(item)
    assert attempt_root.name == "attempt-002"
    assert _archived_attempt_authorizes_retry(attempt_root) is False
    assert (attempt_root / "runtime" / "host_guard.json").is_file()
    assert not item["runtime_root"].exists()


def _enable_censoring_policy(inputs: tuple, *, maximum_attempts: int = 3) -> dict:
    manifest_path = inputs[0]
    manifest = read_json(manifest_path)
    policy = _retry_policy("unused", maximum_attempts=maximum_attempts)
    policy.pop("archive_root")
    manifest["infrastructure_censoring_policy"] = policy
    manifest["evidence_pipeline_bindings"] = _current_evidence_pipeline_bindings()
    manifest["report_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "report_hash"}
    )
    write_json(manifest_path, manifest)
    return manifest


def test_runner_waits_for_precommitted_quiescence_before_first_attempt(
    tmp_path, monkeypatch
) -> None:
    import tools.run_cf2x_b_gate_replays as replay_runner

    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    _enable_censoring_policy(inputs)
    events: list[str] = []
    monkeypatch.setattr(
        replay_runner,
        "_wait_for_retry_quiescence",
        lambda policy: events.append(f"quiet:{policy['required_quiescence_s']}"),
    )
    monkeypatch.setattr(
        replay_runner,
        "isaac_host_lock",
        lambda: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        replay_runner,
        "run_guarded_process",
        lambda *args, **kwargs: (
            events.append("run") or types.SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.setattr(replay_runner, "_validate_completed_replay", lambda *args, **kwargs: None)
    arguments = argparse.Namespace(
        manifest=inputs[0],
        a_gate=inputs[1],
        source_root=inputs[2],
        layouts_root=inputs[3],
        release_config=inputs[4],
        isaac_python=inputs[5],
        cf2x_usd=inputs[6],
        timeout_s=60.0,
        limit=1,
        resume=False,
        prepare_only=False,
        device="cuda:0",
    )

    assert replay_runner.run(arguments) == 0
    assert events == ["quiet:0.0", "run"]


def test_resume_classifies_host_censoring_before_partial_failure_artifacts(
    tmp_path, monkeypatch
) -> None:
    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    manifest = _enable_censoring_policy(inputs)
    ancestor = manifest["layout_ancestors"][0]
    method = manifest["method_ids"][0]
    stem = f"{ancestor}__{method}"
    binding = content_hash(
        {
            "manifest_report_hash": manifest["report_hash"],
            "layout_ancestor": ancestor,
            "method_id": method,
        }
    )
    write_json(
        tmp_path / "runtime" / stem / "host_guard.json",
        _failed_host_receipt(binding, "foreign_runtime_during_attempt"),
    )
    write_json(tmp_path / "replays" / f"{stem}.public.failure.json", {"partial": True})
    plan = build_replay_plan(*inputs, device="cuda:0", resume=True)
    assert plan[0]["status"] == "RETRYABLE_INFRASTRUCTURE_FAILURE"


def test_method_or_safety_failure_is_never_reclassified_for_retry(tmp_path, monkeypatch) -> None:
    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    manifest = _enable_censoring_policy(inputs)
    ancestor = manifest["layout_ancestors"][0]
    method = manifest["method_ids"][0]
    stem = f"{ancestor}__{method}"
    binding = content_hash(
        {
            "manifest_report_hash": manifest["report_hash"],
            "layout_ancestor": ancestor,
            "method_id": method,
        }
    )
    host_path = tmp_path / "runtime" / stem / "host_guard.json"
    write_json(host_path, _failed_host_receipt(binding, None))
    write_json(tmp_path / "replays" / f"{stem}.public.failure.json", {"method_failure": True})
    policy = _retry_policy(tmp_path / "censored-attempts")
    assert not _is_retryable_host_failure(
        host_path,
        policy=policy,
        evidence_binding=binding,
    )
    with pytest.raises(FileExistsError, match="failed replay evidence"):
        build_replay_plan(*inputs, device="cuda:0", resume=True)
    assert not (tmp_path / "censored-attempts").exists()


def test_runner_rejects_post_commit_evidence_pipeline_source_drift(tmp_path, monkeypatch) -> None:
    inputs = _b_gate_plan_fixture(tmp_path, monkeypatch)
    manifest = _enable_censoring_policy(inputs)
    manifest["evidence_pipeline_bindings"]["replay_runner_source_sha256"] = "0" * 64
    manifest["report_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "report_hash"}
    )
    write_json(inputs[0], manifest)
    with pytest.raises(ValueError, match="evidence pipeline differs"):
        build_replay_plan(*inputs, device="cuda:0", resume=True)


def _materialize_bound_censored_archive(tmp_path) -> tuple[tuple, dict, object]:
    inputs = _b_gate_runner_fixture(tmp_path)
    manifest = _enable_censoring_policy(inputs)
    ancestor = manifest["layout_ancestors"][0]
    method = manifest["method_ids"][0]
    stem = f"{ancestor}__{method}"
    binding = content_hash(
        {
            "manifest_report_hash": manifest["report_hash"],
            "layout_ancestor": ancestor,
            "method_id": method,
        }
    )
    item, _unused = _archive_item(tmp_path)
    item.update(
        {
            "layout_ancestor": ancestor,
            "method_id": method,
            "attempt_binding": binding,
            "public_path": tmp_path / "replays" / f"{stem}.public.json",
            "private_path": tmp_path / "replays" / f"{stem}.private.json",
            "runtime_root": tmp_path / "runtime" / stem,
            "retry_policy": _retry_policy(tmp_path / "censored-attempts"),
        }
    )
    # Rebind the fixture receipt after changing the manifest-derived attempt identity.
    write_json(
        item["runtime_root"] / "host_guard.json",
        _failed_host_receipt(binding, "foreign_runtime_during_attempt"),
    )
    attempt_root = _archive_retryable_attempt(item)
    return inputs, manifest, attempt_root


def test_final_verifier_binds_all_censored_attempts_and_denominator(tmp_path) -> None:
    inputs, manifest, _attempt_root = _materialize_bound_censored_archive(tmp_path)
    records = _load_censored_attempt_records(inputs[0], manifest)
    assert len(records) == 1
    a_gate, _plain_manifest, replay_records, l0_records = _inputs()
    manifest_for_report = deepcopy(_plain_manifest)
    manifest_for_report["report_hash"] = manifest["report_hash"]
    manifest_for_report["infrastructure_censoring_policy"] = manifest[
        "infrastructure_censoring_policy"
    ]
    # Normalize the archive record to this unit-level report manifest identity.
    records[0]["evidence_binding"] = content_hash(
        {
            "manifest_report_hash": manifest_for_report["report_hash"],
            "layout_ancestor": records[0]["layout_ancestor"],
            "method_id": records[0]["method_id"],
        }
    )
    # The filesystem loader is the authority for hashes; pure aggregation checks the denominator.
    manifest_for_report["layout_ancestors"] = manifest["layout_ancestors"]
    manifest_for_report["method_ids"] = manifest["method_ids"]
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest_for_report,
        replay_records=replay_records,
        l0_records=l0_records,
        censored_attempt_records=records,
    )
    assert report["checks"]["all_infrastructure_censoring_attempts_preserved_and_bound"]
    assert report["infrastructure_censoring"]["censored_attempt_count"] == 1
    assert report["infrastructure_censoring"]["total_attempt_denominator"] == 10


@pytest.mark.parametrize("mutation", ["delete", "cross-pair"])
def test_final_verifier_rejects_missing_or_cross_pair_censored_evidence(
    tmp_path, mutation: str
) -> None:
    inputs, manifest, attempt_root = _materialize_bound_censored_archive(tmp_path)
    if mutation == "delete":
        (attempt_root / "runtime" / "isaac.log").unlink()
    else:
        wrong_stem = (
            f"{manifest['layout_ancestors'][1]}__{manifest['method_ids'][1]}"
        )
        attempt_root.parent.replace(attempt_root.parents[1] / wrong_stem)
    with pytest.raises(ValueError):
        _load_censored_attempt_records(inputs[0], manifest)
