from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rivermark_benchmark.citylite_scene import (
    AABB,
    PUBLIC_ROUTE_FAMILIES_W_M,
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SHA256,
)
from rivermark_benchmark.collection_protocol import (
    load_collection_protocol,
    native_t2_motion_contract,
    resolve_collection_binding,
)
from rivermark_benchmark.isaac_capture import (
    PrivateEvaluatorManifestError,
    _route_execution_profile,
    validate_external_private_evaluator_manifest,
    validate_private_target_geometry,
)
from rivermark_benchmark.private_evaluator_manifest import (
    NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND,
    NATIVE_GEOMETRY_SCAN_GENERATOR,
    NATIVE_GEOMETRY_SCAN_SCHEMA,
    NATIVE_GEOMETRY_SCAN_TOOL_PATH,
    NATIVE_T2_TASK_VARIANT_ID,
    NATIVE_T2_V2_TASK_VARIANT_ID,
    NATIVE_T2_V3_TASK_VARIANT_ID,
    PrivateManifestGenerationError,
    build_private_evaluator_manifest,
    load_native_geometry_catalog,
    main,
    native_geometry_scan_sha256,
    retain_private_evaluator_manifest,
    write_private_evaluator_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config" / "collection_protocol.citylite_minimal_v1.json"
T1_V2_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_t1_expert_coverage_v2.json"
)
NATIVE_T2_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v1.json"
)
NATIVE_T2_V2_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v2.json"
)
NATIVE_T2_V3_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v3.json"
)


def _write_test_geometry_scan(path: Path) -> tuple[AABB, ...]:
    # This synthetic test occluder is deliberately not an extracted asset.  It
    # makes the validation route's partial-visibility geometry reproducible in
    # CPU tests without redistributing any City-Lite mesh or scan artifact.
    boxes = (
        AABB(
            (-32.0, -15.0, 4.75),
            (-30.0, -1.0, 14.25),
            source_prim="/World/Test/partial_visibility_occluder",
            category="test_occluder",
        ),
    )
    payload = {
        "schema": NATIVE_GEOMETRY_SCAN_SCHEMA,
        "status": "passed",
        "formal": False,
        "generator": NATIVE_GEOMETRY_SCAN_GENERATOR,
        "geometry_evidence_kind": NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND,
        "tool_path": NATIVE_GEOMETRY_SCAN_TOOL_PATH,
        "tool_sha256": "b" * 64,
        "source_revision": "c" * 40,
        "source_tree_sha256": "d" * 64,
        "source_worktree_dirty": False,
        "runtime_lock": {
            "sha256": "e" * 64,
            "profile_id": "isaac-windows-5.1",
            "audit_status": "passed",
        },
        "scene_id": "RIVERMARK_CITY_LITE_v1",
        "scene_contract_sha256": SCENE_CONTRACT_SHA256,
        "scene_content_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
        "domains": [
            {
                "aabb": {
                    "min": list(box.minimum),
                    "max": list(box.maximum),
                    "path": box.source_prim,
                    "source_kind": box.category,
                }
            }
            for box in boxes
        ],
    }
    payload["scan_sha256"] = native_geometry_scan_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return boxes


@pytest.mark.parametrize(
    ("cell_id", "route_family_id", "target_region_id", "visibility_bucket"),
    (
        (
            "train-citylite-nominal",
            "citylite-route-family-a-v1",
            "citylite-target-region-b-v1",
            "direct-visible-v1",
        ),
        (
            "validation-citylite-nominal",
            "citylite-route-family-b-v1",
            "citylite-target-region-a-v1",
            "partial-visible-v1",
        ),
    ),
)
def test_generated_private_manifest_is_reproducible_and_geometry_bound(
    tmp_path: Path,
    cell_id: str,
    route_family_id: str,
    target_region_id: str,
    visibility_bucket: str,
) -> None:
    scan_path = tmp_path / "native_scan.json"
    boxes = _write_test_geometry_scan(scan_path)
    first = build_private_evaluator_manifest(
        protocol_path=PROTOCOL_PATH,
        cell_id=cell_id,
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
    )
    second = build_private_evaluator_manifest(
        protocol_path=PROTOCOL_PATH,
        cell_id=cell_id,
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
    )
    assert first == second
    assert len(first["targets"]) == 4
    assert all("visibility_evidence" not in target for target in first["targets"])
    visibility = first["target_visibility_contract"]
    assert visibility["route_family_id"] == route_family_id
    assert visibility["target_region_id"] == target_region_id
    assert visibility["visibility_bucket"] == visibility_bucket

    protocol = load_collection_protocol(PROTOCOL_PATH)
    binding = resolve_collection_binding(protocol, cell_id=cell_id, episode_index=0)
    targets = validate_external_private_evaluator_manifest(
        first,
        city_lite_scene_contract_sha256=first["city_lite_scene_contract_sha256"],
        city_lite_scene_payload_sha256=first["city_lite_scene_payload_sha256"],
        expected_collection_binding=binding,
    )
    report = validate_private_target_geometry(
        first,
        structural_aabbs=boxes,
        public_routes_w_m=PUBLIC_ROUTE_FAMILIES_W_M[route_family_id],
        city_lite_scene_contract_sha256=first["city_lite_scene_contract_sha256"],
        city_lite_scene_payload_sha256=first["city_lite_scene_payload_sha256"],
    )
    assert len(targets) == report["target_count"] == 4
    assert report["target_region_id"] == target_region_id
    assert report["visibility_bucket"] == visibility_bucket


def test_manifest_rejects_binding_and_geometry_tampering(tmp_path: Path) -> None:
    scan_path = tmp_path / "native_scan.json"
    boxes = _write_test_geometry_scan(scan_path)
    manifest = build_private_evaluator_manifest(
        protocol_path=PROTOCOL_PATH,
        cell_id="train-citylite-nominal",
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
    )
    wrong_binding = dict(manifest["collection_binding"])
    wrong_binding["episode_index"] = 1
    with pytest.raises(PrivateEvaluatorManifestError, match="does not match the capture"):
        validate_external_private_evaluator_manifest(
            manifest,
            city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
            city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
            expected_collection_binding=wrong_binding,
        )

    tampered = copy.deepcopy(manifest)
    tampered["target_visibility_contract"]["aabb_geometry_sha256"] = "0" * 64
    with pytest.raises(PrivateEvaluatorManifestError, match="runtime route/AABB geometry"):
        validate_private_target_geometry(
            tampered,
            structural_aabbs=boxes,
            public_routes_w_m=PUBLIC_ROUTE_FAMILIES_W_M[
                "citylite-route-family-a-v1"
            ],
            city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
            city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
        )


def test_native_t2_manifest_variant_is_explicit_and_not_t1_interchangeable(
    tmp_path: Path,
) -> None:
    scan_path = tmp_path / "native_scan.json"
    _write_test_geometry_scan(scan_path)
    manifest = build_private_evaluator_manifest(
        protocol_path=NATIVE_T2_PROTOCOL_PATH,
        cell_id="native-t2-canary-inner-dev-v1",
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
        task_variant_id=NATIVE_T2_TASK_VARIANT_ID,
    )
    validate_external_private_evaluator_manifest(
        manifest,
        city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
        city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
        expected_task_variant_id=NATIVE_T2_TASK_VARIANT_ID,
    )
    with pytest.raises(PrivateEvaluatorManifestError, match="task_variant_id"):
        validate_external_private_evaluator_manifest(
            manifest,
            city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
            city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
        )

    with pytest.raises(PrivateManifestGenerationError, match="dedicated native T2 canary protocol"):
        build_private_evaluator_manifest(
            protocol_path=T1_V2_PROTOCOL_PATH,
            cell_id="train-citylite-direct-v2",
            episode_index=5,
            geometry_scan_path=scan_path,
            target_seed=18,
            task_variant_id=NATIVE_T2_TASK_VARIANT_ID,
        )
    with pytest.raises(PrivateManifestGenerationError, match="cannot generate a non-T2"):
        build_private_evaluator_manifest(
            protocol_path=NATIVE_T2_PROTOCOL_PATH,
            cell_id="native-t2-canary-inner-dev-v1",
            episode_index=1,
            geometry_scan_path=scan_path,
            target_seed=19,
        )


def test_native_t2_v2_manifest_binds_yaw_aware_visibility_contract(tmp_path: Path) -> None:
    scan_path = tmp_path / "native_scan.json"
    _write_test_geometry_scan(scan_path)
    manifest = build_private_evaluator_manifest(
        protocol_path=NATIVE_T2_V2_PROTOCOL_PATH,
        cell_id="native-t2-canary-inner-dev-v2",
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
        task_variant_id=NATIVE_T2_V2_TASK_VARIANT_ID,
    )
    visibility = manifest["target_visibility_contract"]
    assert visibility["schema"] == "org.rivermark.private-target-visibility-geometry.v4"
    assert visibility["camera_heading_contract"]["model"] == "segment_horizontal_heading_yaw_limited_v1"
    assert visibility["execution_window"]["waypoint_segment_seconds"] == 6.0
    motion = native_t2_motion_contract(load_collection_protocol(NATIVE_T2_V2_PROTOCOL_PATH))
    assert motion is not None
    validate_external_private_evaluator_manifest(
        manifest,
        city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
        city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
        expected_task_variant_id=NATIVE_T2_V2_TASK_VARIANT_ID,
        expected_native_t2_motion_contract=motion,
    )
    tampered = copy.deepcopy(manifest)
    tampered["target_visibility_contract"]["camera_heading_contract"]["yaw_settle_margin_s"] = 0.1
    with pytest.raises(PrivateEvaluatorManifestError, match="frozen native T2 motion contract"):
        validate_external_private_evaluator_manifest(
            tampered,
            city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
            city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
            expected_task_variant_id=NATIVE_T2_V2_TASK_VARIANT_ID,
            expected_native_t2_motion_contract=motion,
        )


def test_native_t2_v3_manifest_is_bound_to_the_time_scaled_contract(tmp_path: Path) -> None:
    scan_path = tmp_path / "native_scan.json"
    _write_test_geometry_scan(scan_path)
    manifest = build_private_evaluator_manifest(
        protocol_path=NATIVE_T2_V3_PROTOCOL_PATH,
        cell_id="native-t2-canary-inner-dev-v3",
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
        task_variant_id=NATIVE_T2_V3_TASK_VARIANT_ID,
    )
    motion = native_t2_motion_contract(load_collection_protocol(NATIVE_T2_V3_PROTOCOL_PATH))
    assert motion is not None
    visibility = manifest["target_visibility_contract"]
    assert visibility["execution_window"]["waypoint_segment_seconds"] == 12.0
    validate_external_private_evaluator_manifest(
        manifest,
        city_lite_scene_contract_sha256=manifest["city_lite_scene_contract_sha256"],
        city_lite_scene_payload_sha256=manifest["city_lite_scene_payload_sha256"],
        expected_task_variant_id=NATIVE_T2_V3_TASK_VARIANT_ID,
        expected_native_t2_motion_contract=motion,
    )
    with pytest.raises(PrivateManifestGenerationError, match="different native T2 protocol revision"):
        build_private_evaluator_manifest(
            protocol_path=NATIVE_T2_V3_PROTOCOL_PATH,
            cell_id="native-t2-canary-inner-dev-v3",
            episode_index=0,
            geometry_scan_path=scan_path,
            target_seed=18,
            task_variant_id=NATIVE_T2_V2_TASK_VARIANT_ID,
        )


def test_geometry_scan_and_private_output_fail_closed(tmp_path: Path) -> None:
    scan_path = tmp_path / "native_scan.json"
    _write_test_geometry_scan(scan_path)
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    payload["scene_content_sha256"] = "0" * 64
    scan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PrivateManifestGenerationError, match="payload does not match"):
        load_native_geometry_catalog(scan_path)

    _write_test_geometry_scan(scan_path)
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    payload["domains"][0]["aabb"]["max"][0] += 0.1
    scan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PrivateManifestGenerationError, match="SHA-256 does not match"):
        load_native_geometry_catalog(scan_path)

    _write_test_geometry_scan(scan_path)
    manifest = build_private_evaluator_manifest(
        protocol_path=PROTOCOL_PATH,
        cell_id="train-citylite-nominal",
        episode_index=0,
        geometry_scan_path=scan_path,
        target_seed=17,
    )
    with pytest.raises(PrivateManifestGenerationError, match="outside the repository"):
        write_private_evaluator_manifest(
            ROOT / "private_manifest.json", manifest, repository_root=ROOT
        )
    output = tmp_path / "private" / "manifest.json"
    digest = write_private_evaluator_manifest(output, manifest, repository_root=ROOT)
    assert len(digest) == 64
    assert output.is_file()
    assert "position_w_m" in output.read_text(encoding="utf-8")
    with pytest.raises(PrivateManifestGenerationError, match="refusing to overwrite"):
        write_private_evaluator_manifest(output, manifest, repository_root=ROOT)


def test_cli_success_summary_redacts_private_seed_and_targets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scan_path = tmp_path / "native_scan.json"
    _write_test_geometry_scan(scan_path)
    output = tmp_path / "private" / "manifest.json"
    private_seed = 4_294_967_295

    result = main(
        [
            "--protocol",
            str(PROTOCOL_PATH),
            "--cell-id",
            "train-citylite-nominal",
            "--episode-index",
            "0",
            "--target-seed",
            str(private_seed),
            "--geometry-scan",
            str(scan_path),
            "--output",
            str(output),
            "--repository-root",
            str(ROOT),
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    assert summary["status"] == "written"
    assert summary["collection_binding"]["cell_id"] == "train-citylite-nominal"
    assert summary["collection_binding"]["episode_index"] == 0
    assert "episode_seed" not in summary["collection_binding"]
    assert str(private_seed) not in stdout
    assert "position_w_m" not in stdout
    assert "targets" not in stdout


def test_cli_can_generate_private_entropy_without_command_line_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_path = tmp_path / "native_scan.json"
    _write_test_geometry_scan(scan_path)
    output = tmp_path / "private" / "manifest.json"
    private_seed = 4_294_967_294
    requested_bits: list[int] = []

    def _randbits(bits: int) -> int:
        requested_bits.append(bits)
        return private_seed

    monkeypatch.setattr(
        "rivermark_benchmark.private_evaluator_manifest.secrets.randbits", _randbits
    )
    result = main(
        [
            "--protocol",
            str(PROTOCOL_PATH),
            "--cell-id",
            "train-citylite-nominal",
            "--episode-index",
            "0",
            "--generate-target-seed",
            "--geometry-scan",
            str(scan_path),
            "--output",
            str(output),
            "--repository-root",
            str(ROOT),
        ]
    )

    assert result == 0
    assert requested_bits == [32]
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    assert summary["status"] == "written"
    assert str(private_seed) not in stdout
    assert "position_w_m" not in stdout
    assert "targets" not in stdout


def test_content_addressed_retention_is_external_atomic_and_rejects_collision(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    capture = tmp_path / "capture"
    operator = tmp_path / "operator"
    retention = operator / "manifest-vault"
    repository.mkdir()
    capture.mkdir()
    retention.mkdir(parents=True)
    source = operator / "candidate.json"
    source.write_bytes(b'{"private":"evaluator truth"}\n')

    first = retain_private_evaluator_manifest(
        source,
        retention,
        forbidden_roots=(repository, capture),
    )
    assert first.path.parent == retention.resolve()
    assert first.path.name == f"{first.sha256}.json"
    assert first.path.read_bytes() == source.read_bytes()
    assert first.byte_count == len(source.read_bytes())
    assert not tuple(retention.glob("*.tmp"))

    second = retain_private_evaluator_manifest(
        source,
        retention,
        forbidden_roots=(repository, capture),
    )
    assert second == first

    first.path.write_bytes(b"different bytes\n")
    with pytest.raises(PrivateManifestGenerationError, match="differs"):
        retain_private_evaluator_manifest(
            source,
            retention,
            forbidden_roots=(repository, capture),
        )


def test_retention_rejects_source_or_root_under_public_boundaries(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    capture = tmp_path / "capture"
    repository.mkdir()
    capture.mkdir()
    source = tmp_path / "operator.json"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(PrivateManifestGenerationError, match="retention root"):
        retain_private_evaluator_manifest(
            source,
            repository,
            forbidden_roots=(repository, capture),
        )

    in_capture = capture / "evaluator.json"
    in_capture.write_text("{}\n", encoding="utf-8")
    retention = tmp_path / "retention"
    retention.mkdir()
    with pytest.raises(PrivateManifestGenerationError, match="manifest must be outside"):
        retain_private_evaluator_manifest(
            in_capture,
            retention,
            forbidden_roots=(repository, capture),
        )


def test_protocol_cells_match_the_executable_route_profiles() -> None:
    protocol = load_collection_protocol(PROTOCOL_PATH)
    for cell in protocol["cells"]:
        profile = _route_execution_profile(
            {"condition_request": {"conditions": cell["conditions"]}}
        )
        assert profile.route_family_id == cell["conditions"]["route_family"]
        assert profile.start_anchor_id == cell["conditions"]["start_anchor"]
        assert profile.target_region_id == cell["conditions"]["target_region"]
        assert profile.visibility_bucket == cell["conditions"]["visibility_bucket"]


def test_t1_v2_validation_route_executes_direct_visibility() -> None:
    protocol = load_collection_protocol(T1_V2_PROTOCOL_PATH)
    validation = next(cell for cell in protocol["cells"] if cell["split"] == "validation")
    profile = _route_execution_profile(
        {"condition_request": {"conditions": validation["conditions"]}}
    )

    assert profile.route_family_id == "citylite-route-family-b-v1"
    assert profile.start_anchor_id == "citylite-start-anchor-b-v1"
    assert profile.target_region_id == "citylite-target-region-a-v1"
    assert profile.visibility_bucket == "direct-visible-v1"
