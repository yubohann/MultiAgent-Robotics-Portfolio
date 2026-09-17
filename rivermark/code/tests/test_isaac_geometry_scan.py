from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rivermark_benchmark.citylite_scene import (
    AABB,
    SCENE_CONTRACT_PAYLOAD_IDENTITY,
    SCENE_CONTRACT_IDENTITY,
)
from rivermark_benchmark.isaac_geometry_scan import (
    IsaacGeometryScanError,
    build_native_geometry_scan_payload,
    build_parser,
)
from rivermark_benchmark.private_evaluator_manifest import (
    PrivateManifestGenerationError,
    load_native_geometry_catalog,
    native_geometry_scan_identity,
)
from rivermark_benchmark.provenance import SourceProvenance


def _authority() -> SimpleNamespace:
    return SimpleNamespace(
        contract_payload_identity=SCENE_CONTRACT_PAYLOAD_IDENTITY,
        contract_identity=SCENE_CONTRACT_IDENTITY,
        provenance=lambda: {"environment_id": "RIVERMARK_CITY_LITE_v1"},
    )


def _payload() -> dict[str, object]:
    return build_native_geometry_scan_payload(
        authority=_authority(),
        structural_aabbs=(
            AABB(
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
                source_prim="/World/StaticScene/City/Rivermark/Test",
                category="rivermark_structural_visual",
            ),
        ),
        source=SourceProvenance("b" * 40, "c" * 16, False),
        tool_identity="d" * 16,
        stage_evidence={
            "active_static_prim_count": 2,
            "native_collision_counts": {"city_task_obstacles": 1},
        },
        runtime_lock_digest="e" * 16,
        runtime_profile_id="isaac-windows-5.1",
        system_commit={"commit_percent": 25.0},
    )


def test_native_geometry_payload_is_self_identified_and_loadable(tmp_path: Path) -> None:
    payload = _payload()
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = load_native_geometry_catalog(path)

    assert len(catalog.structural_aabbs) == 1
    assert catalog.scan_identity == payload["scan_identity"]


def test_native_geometry_payload_alteration_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["runtime_lock"]["identity"] = "f" * 16
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PrivateManifestGenerationError, match="IDENTITY does not match"):
        load_native_geometry_catalog(path)


def test_md_qd_or_foreign_city_lite_catalogs_remain_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["schema"] = "md-qd-swarm.native-geometry-scan.v1"
    path = tmp_path / "md_qd.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PrivateManifestGenerationError, match="unsupported schema"):
        load_native_geometry_catalog(path)

    payload = _payload()
    payload["scene_id"] = "RIVERMARK_CITY_LITE_MDQD_v1"
    payload["scan_identity"] = native_geometry_scan_identity(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PrivateManifestGenerationError, match="not for approved City-Lite"):
        load_native_geometry_catalog(path)


def test_native_geometry_payload_refuses_dirty_source() -> None:
    with pytest.raises(IsaacGeometryScanError, match="clean Git worktree"):
        build_native_geometry_scan_payload(
            authority=_authority(),
            structural_aabbs=(
                AABB(
                    (1.0, 2.0, 3.0),
                    (4.0, 5.0, 6.0),
                    source_prim="/World/StaticScene/City/Rivermark/Test",
                    category="rivermark_structural_visual",
                ),
            ),
            source=SourceProvenance("b" * 40, "c" * 16, True),
            tool_identity="d" * 16,
            stage_evidence={
                "active_static_prim_count": 2,
                "native_collision_counts": {"city_task_obstacles": 1},
            },
            runtime_lock_digest="e" * 16,
            runtime_profile_id="isaac-windows-5.1",
            system_commit=None,
        )


def test_geometry_scanner_parser_freezes_resource_guards() -> None:
    args = build_parser().parse_args(
        [
            "--output",
            "scan.json",
            "--scene-contract",
            "scene.json",
            "--runtime-lock",
            "lock.json",
            "--isaaclab-source",
            "isaaclab",
            "--drone-usd",
            "cf2x.usd",
        ]
    )

    assert args.preflight_commit_percent == 65.0
    assert args.abort_commit_percent == 82.0
