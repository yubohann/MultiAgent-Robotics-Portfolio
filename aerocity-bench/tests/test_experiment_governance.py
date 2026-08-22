from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.experiment_governance import (
    audit_experiment_governance,
    build_external_evidence_manifest,
)


def _registry() -> dict[str, object]:
    repository = Path(__file__).parents[1]
    return read_json(repository / "configs" / "experiment-governance-v1.json")


def _rehash(registry: dict[str, object]) -> dict[str, object]:
    registry["registry_hash"] = content_hash(
        {key: value for key, value in registry.items() if key != "registry_hash"}
    )
    return registry


def _rehash_external_manifest(manifest: dict[str, object]) -> dict[str, object]:
    manifest["manifest_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    return manifest


def _hashed_report(payload: dict[str, object]) -> dict[str, object]:
    payload["report_hash"] = content_hash(payload)
    return payload


def _synthetic_external_receipt(relative: str) -> dict[str, object] | None:
    """Build schema-minimal public receipts for evidence-binding tests only.

    These files exercise hash binding and fail-closed governance in an isolated
    source fixture.  They are never used as experimental evidence or copied
    into the repository's ignored ``reason/`` tree.
    """

    if relative.endswith("current-boundary-source-manifest-v2.json"):
        payload: dict[str, object] = {
            "accepted_ancestor_count": 3,
            "formal_score_eligible": False,
        }
        payload["manifest_hash"] = content_hash(payload)
        return payload
    if relative.endswith("cf2x-b-gate-manifest-v16-current-boundary.json"):
        return _hashed_report(
            {
                "formal_score_eligible": False,
                "layout_ancestors": ["ancestor-00", "ancestor-01", "ancestor-02"],
                "method_ids": ["synthetic-public-method"],
                "purpose": "development-only-public-four-cf2x-l1-calibration",
            }
        )
    if relative.endswith("l0-pairing-v16-current-boundary.json"):
        return _hashed_report(
            {
                "formal_score_eligible": False,
                "status": "VERIFIED_L0_PAIRING",
            }
        )
    if relative.endswith("cf2x-b-gate-verification-v16-current-boundary.json"):
        manifest = _synthetic_external_receipt(
            "cf2x-b-gate-manifest-v16-current-boundary.json"
        )
        assert isinstance(manifest, dict)
        return _hashed_report(
            {
                "authorizes_formal_test_access": False,
                "checks": {"synthetic_contract_check": True},
                "failure_count": 0,
                "fidelity_audit": {"status": "MEASURED_NOT_FROZEN"},
                "formal_score_eligible": False,
                "input_report_hashes": {"manifest": manifest["report_hash"]},
                "layout_ancestors": ["ancestor-00", "ancestor-01", "ancestor-02"],
                "method_ids": ["synthetic-public-method"],
                "status": "VERIFIED",
            }
        )
    return None


def _external_evidence_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    """Materialize source-only checkout plus separate synthetic public receipts."""

    source = Path(__file__).parents[1]
    clean_root = tmp_path / "clean-source"
    registry_path = clean_root / "configs" / "experiment-governance-v1.json"
    registry_path.parent.mkdir(parents=True)
    shutil.copy2(source / "configs" / "experiment-governance-v1.json", registry_path)
    subprocess.run(["git", "init", "-q", str(clean_root)], check=True)
    subprocess.run(["git", "-C", str(clean_root), "add", "configs"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(clean_root),
            "-c",
            "user.email=aerocity-test@example.invalid",
            "-c",
            "user.name=AeroCity Test",
            "commit",
            "-qm",
            "source fixture",
        ],
        check=True,
    )
    registry = read_json(registry_path)
    evidence_root = tmp_path / "external-evidence"
    for record in registry["records"]:
        for relative in record["evidence_paths"]:
            destination = evidence_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            receipt = _synthetic_external_receipt(relative)
            if receipt is None:
                destination.write_text("synthetic binding fixture\n", encoding="utf-8")
            else:
                write_json(destination, receipt)
    manifest = build_external_evidence_manifest(
        repository_root=clean_root,
        registry_path=registry_path,
        registry=registry,
        evidence_root=evidence_root,
    )
    manifest_path = tmp_path / "external-evidence-manifest.json"
    write_json(manifest_path, manifest)
    return clean_root, registry_path, evidence_root, manifest


def test_source_only_registry_fails_closed_until_external_evidence_is_bound() -> None:
    repository = Path(__file__).parents[1]
    report = audit_experiment_governance(_registry(), repository_root=repository)

    assert report["overall_status"] == "CONTAINMENT_FAIL_FORMAL_NO_GO"
    assert report["formal_score_eligible"] is False
    assert report["external_evidence"]["status"] == "NOT_USED"
    assert any("registered evidence does not exist" in issue["issue"] for issue in report["issues"])


def test_registry_covers_every_experiment_class_and_quarantines_old_replay(
    tmp_path: Path,
) -> None:
    clean_root, registry_path, evidence_root, _manifest = _external_evidence_fixture(tmp_path)
    report = audit_experiment_governance(
        read_json(registry_path),
        repository_root=clean_root,
        registry_path=registry_path,
        external_evidence_root=evidence_root,
        external_evidence_manifest=tmp_path / "external-evidence-manifest.json",
    )

    assert report["overall_status"] == "CONTAINMENT_PASS_FORMAL_NO_GO"
    assert report["formal_score_eligible"] is False
    assert report["containment_checks"]["legacy_cf2x_replay_retired"] is True
    assert report["containment_checks"]["stale_cf2x_v15_replay_retired"] is True
    assert report["containment_checks"]["historical_cf2x_v16_replay_superseded"] is True
    assert report["containment_checks"]["formal_main_matrix_blocked"] is True
    assert report["containment_checks"]["learning_training_blocked"] is True
    assert (
        report["next_authorized_step"]
        == "RERUN_METHOD_INDEPENDENT_A_GATE_FOR_CURRENT_CONTRACT"
    )


def test_legacy_public_replay_cannot_be_promoted_after_boundary_failure() -> None:
    repository = Path(__file__).parents[1]
    registry = copy.deepcopy(_registry())
    record = next(item for item in registry["records"] if item["kind"] == "l1_public_replay")
    record["status"] = "CALIBRATION_ONLY"
    record["phase"] = "calibration"
    record["public_input_boundary"] = "enforced"
    _rehash(registry)

    report = audit_experiment_governance(registry, repository_root=repository)

    assert report["overall_status"] == "CONTAINMENT_FAIL_FORMAL_NO_GO"
    assert not report["containment_checks"]["legacy_cf2x_replay_retired"]


def test_stale_v15_replay_cannot_be_promoted_after_schema_failure() -> None:
    repository = Path(__file__).parents[1]
    registry = copy.deepcopy(_registry())
    record = next(
        item for item in registry["records"] if item["id"] == "cf2x-public-three-ancestor-v15"
    )
    record["status"] = "CALIBRATION_ONLY"
    record["phase"] = "calibration"
    record["public_input_boundary"] = "enforced"
    record["task_contract_status"] = "frozen_calibration_only"
    record["result_adaptive_change"] = "forbidden"
    _rehash(registry)

    report = audit_experiment_governance(registry, repository_root=repository)

    assert report["overall_status"] == "CONTAINMENT_FAIL_FORMAL_NO_GO"
    assert not report["containment_checks"]["stale_cf2x_v15_replay_retired"]


def test_historical_v16_replay_cannot_be_promoted_to_current_calibration() -> None:
    repository = Path(__file__).parents[1]
    registry = copy.deepcopy(_registry())
    record = next(
        item
        for item in registry["records"]
        if item["id"] == "cf2x-public-three-ancestor-v16-current-boundary"
    )
    record["phase"] = "calibration"
    record["status"] = "CALIBRATION_ONLY"
    record["task_contract_status"] = "frozen_calibration_only"
    record["result_adaptive_change"] = "forbidden"
    _rehash(registry)

    report = audit_experiment_governance(registry, repository_root=repository)

    assert report["overall_status"] == "CONTAINMENT_FAIL_FORMAL_NO_GO"
    assert not report["containment_checks"]["historical_cf2x_v16_replay_superseded"]


def test_historical_v16_replay_requires_the_bound_verified_evidence_set() -> None:
    repository = Path(__file__).parents[1]
    registry = copy.deepcopy(_registry())
    record = next(
        item
        for item in registry["records"]
        if item["id"] == "cf2x-public-three-ancestor-v16-current-boundary"
    )
    record["evidence_paths"] = ["docs/正式实验启动门禁与落实记录.md"]
    _rehash(registry)

    report = audit_experiment_governance(registry, repository_root=repository)

    assert report["overall_status"] == "CONTAINMENT_FAIL_FORMAL_NO_GO"
    assert not report["containment_checks"]["historical_cf2x_v16_replay_superseded"]


def test_development_result_cannot_claim_formal_eligibility() -> None:
    repository = Path(__file__).parents[1]
    registry = copy.deepcopy(_registry())
    record = next(item for item in registry["records"] if item["kind"] == "g2i_task_calibration")
    record["formal_score_eligible"] = True
    _rehash(registry)

    report = audit_experiment_governance(registry, repository_root=repository)

    assert report["overall_status"] == "CONTAINMENT_FAIL_FORMAL_NO_GO"
    assert not report["containment_checks"]["no_development_record_promoted"]


def test_registry_hash_is_fail_closed() -> None:
    repository = Path(__file__).parents[1]
    registry = _registry()
    registry["registry_hash"] = "0" * 64

    with pytest.raises(ValueError, match="hash mismatch"):
        audit_experiment_governance(registry, repository_root=repository)


def test_superseded_a_gate_is_quarantined_and_does_not_promote_formal_evidence(
    tmp_path: Path,
) -> None:
    clean_root, registry_path, evidence_root, _manifest = _external_evidence_fixture(tmp_path)
    registry = copy.deepcopy(read_json(registry_path))
    record = next(item for item in registry["records"] if item["kind"] == "g2i_task_calibration")
    assert record["task_contract_status"] == "obsolete"
    assert record["status"] == "RETIRED"
    assert record["formal_score_eligible"] is False
    report = audit_experiment_governance(
        registry,
        repository_root=clean_root,
        registry_path=registry_path,
        external_evidence_root=evidence_root,
        external_evidence_manifest=tmp_path / "external-evidence-manifest.json",
    )
    assert report["overall_status"] == "CONTAINMENT_PASS_FORMAL_NO_GO"
    assert report["formal_score_eligible"] is False


def test_clean_source_can_use_explicit_hash_bound_external_evidence(tmp_path: Path) -> None:
    clean_root, registry_path, evidence_root, _manifest = _external_evidence_fixture(tmp_path)

    report = audit_experiment_governance(
        read_json(registry_path),
        repository_root=clean_root,
        registry_path=registry_path,
        external_evidence_root=evidence_root,
        external_evidence_manifest=tmp_path / "external-evidence-manifest.json",
    )

    assert report["overall_status"] == "CONTAINMENT_PASS_FORMAL_NO_GO"
    assert report["external_evidence"]["status"] == "BOUND_EXTERNAL_RECEIPTS"
    assert report["external_evidence"]["registered_path_count"] > 10


def test_external_evidence_rejects_tampered_receipt(tmp_path: Path) -> None:
    clean_root, registry_path, evidence_root, manifest = _external_evidence_fixture(tmp_path)
    receipt = evidence_root / manifest["evidence"][0]["path"]
    receipt.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="external evidence hash differs"):
        audit_experiment_governance(
            read_json(registry_path),
            repository_root=clean_root,
            registry_path=registry_path,
            external_evidence_root=evidence_root,
            external_evidence_manifest=tmp_path / "external-evidence-manifest.json",
        )


def test_external_evidence_rejects_commit_or_registry_binding_change(tmp_path: Path) -> None:
    clean_root, registry_path, evidence_root, manifest = _external_evidence_fixture(tmp_path)
    manifest["source_commit"] = "0" * 40
    write_json(tmp_path / "external-evidence-manifest.json", _rehash_external_manifest(manifest))

    with pytest.raises(ValueError, match="source commit differs"):
        audit_experiment_governance(
            read_json(registry_path),
            repository_root=clean_root,
            registry_path=registry_path,
            external_evidence_root=evidence_root,
            external_evidence_manifest=tmp_path / "external-evidence-manifest.json",
        )

    _external_evidence_fixture(tmp_path / "registry-change")
    nested = tmp_path / "registry-change"
    changed_registry = nested / "clean-source" / "configs" / "experiment-governance-v1.json"
    changed = read_json(changed_registry)
    changed["registry_version"] = "tampered"
    _rehash(changed)
    write_json(changed_registry, changed)
    with pytest.raises(ValueError, match="registry file hash differs"):
        audit_experiment_governance(
            read_json(changed_registry),
            repository_root=nested / "clean-source",
            registry_path=changed_registry,
            external_evidence_root=nested / "external-evidence",
            external_evidence_manifest=nested / "external-evidence-manifest.json",
        )


def test_external_evidence_rejects_nonregistered_or_escaping_path(tmp_path: Path) -> None:
    clean_root, registry_path, evidence_root, manifest = _external_evidence_fixture(tmp_path)
    manifest["evidence"][0]["path"] = "../outside.json"
    write_json(tmp_path / "external-evidence-manifest.json", _rehash_external_manifest(manifest))

    with pytest.raises(ValueError, match="relative to the repository root"):
        audit_experiment_governance(
            read_json(registry_path),
            repository_root=clean_root,
            registry_path=registry_path,
            external_evidence_root=evidence_root,
            external_evidence_manifest=tmp_path / "external-evidence-manifest.json",
        )
