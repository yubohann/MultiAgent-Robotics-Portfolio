from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.environment_manifest import _usd_version, build_environment_manifest


def _cf2x(tmp_path: Path) -> Path:
    path = tmp_path / "assets" / "new" / "cf2x.usd"
    path.parent.mkdir(parents=True)
    path.write_text("#usda 1.0\n", encoding="utf-8")
    return path


def _runner(_: list[str]) -> tuple[int, str, str]:
    return 0, "Test GPU, 999.0", ""


def test_usd_version_probe_runs_in_a_child_interpreter() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> tuple[int, str, str]:
        commands.append(command)
        return 0, "AEROCITY_USD_VERSION=24.11", ""

    assert _usd_version(runner) == {
        "status": "AVAILABLE",
        "version": "24.11",
        "diagnostic": "",
    }
    assert commands == [
        [
            sys.executable,
            "-c",
            "from pxr import Usd\n"
            "print('AEROCITY_USD_VERSION=' + '.'.join(str(part) for part in Usd.GetVersion()))\n",
        ]
    ]


def test_usd_version_probe_records_native_failure() -> None:
    assert _usd_version(lambda _: (1, "", "Windows fatal exception: access violation")) == {
        "status": "UNAVAILABLE",
        "version": None,
        "diagnostic": "Windows fatal exception: access violation",
    }


def test_environment_manifest_marks_dirty_source_as_development_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = {
        "schema": "org.aerocity.bench.cc0-release-asset-audit.v1",
        "status": "PASS",
        "formal_score_eligible": False,
        "report_hash": "a" * 64,
        "registry_hash": "b" * 64,
        "asset_count": 5,
        "usd_dependency_closure": {"checked_usd_layers": 32},
    }
    audit_path = tmp_path / "asset-audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    config = tmp_path / "release.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "aerocity_bench.environment_manifest._git_source_state",
        lambda _: {
            "state": "DIRTY",
            "source_commit": "UNCOMMITTED-DEVELOPMENT",
            "official_release_binding": "REJECTED",
        },
    )

    report = build_environment_manifest(
        repository_root=tmp_path,
        cf2x_usd=_cf2x(tmp_path),
        release_config=config,
        asset_audit=audit_path,
        runner=_runner,
    )

    assert report["formal_score_eligible"] is False
    assert report["source_tree"]["official_release_binding"] == "REJECTED"
    assert report["gpu"]["devices"] == ["Test GPU, 999.0"]
    assert report["runtime_packages"]["pxr_usd"]["status"] == "UNAVAILABLE"
    assert "tmp_path" not in str(report)
    assert report["manifest_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )


def test_environment_manifest_rejects_forbidden_cf2x_location(tmp_path: Path) -> None:
    forbidden = tmp_path / "assets" / "5_in_drone" / "cf2x.usd"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden"):
        build_environment_manifest(repository_root=tmp_path, cf2x_usd=forbidden)
