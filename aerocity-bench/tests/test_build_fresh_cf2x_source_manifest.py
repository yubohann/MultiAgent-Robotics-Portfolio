from __future__ import annotations

from pathlib import Path

from aerocity_bench.canonical import content_hash, write_json
from tools.build_cf2x_b_gate_manifest import build_manifest
from tools.build_fresh_cf2x_source_manifest import build


def test_fresh_source_manifest_paths_are_relative_to_the_manifest(tmp_path: Path) -> None:
    layouts_root = tmp_path / "panel" / "layouts"
    for index in range(3):
        ancestor = layouts_root / f"ancestor-{index:02d}"
        city_root = ancestor / "splits" / "calibration" / f"city-{index:02d}"
        city_path = city_root / "scene_authority" / "cityspec.json"
        episode_path = city_root / "evaluator_private" / "episodes" / "episode-0000.json"
        write_json(city_path, {"layout_id": f"city-{index:02d}"})
        write_json(episode_path, {"episode_id": f"episode-{index:02d}"})
        write_json(
            ancestor / "development_layout_manifest.json",
            {
                "formal_score_eligible": False,
                "public_boundary_audit": {"status": "PASS"},
            },
        )

    source_path = tmp_path / "panel" / "source-v2.json"
    source = build(layouts_root, source_path)

    assert source["manifest_hash"] == content_hash(
        {key: value for key, value in source.items() if key != "manifest_hash"}
    )
    assert source["records"][0]["city_path"].startswith("layouts/ancestor-00/")
    assert all(
        (source_path.parent / record["city_path"]).is_file()
        and (source_path.parent / record["private_episode_path"]).is_file()
        for record in source["records"]
    )

    a_gate = {
        "schema": "org.aerocity.bench.g2-i-a-gate-freeze.v1",
        "status": "VERIFIED",
        "authorizes_next_gate": True,
    }
    a_gate["report_hash"] = content_hash(a_gate)
    a_gate_path = tmp_path / "panel" / "a-gate.json"
    write_json(a_gate_path, a_gate)
    manifest = build_manifest(a_gate_path, source_path, tmp_path / "panel" / "b-gate.json")

    assert manifest["layout_ancestors"] == [
        "g2-i-calibration-ancestor-00",
        "g2-i-calibration-ancestor-01",
        "g2-i-calibration-ancestor-02",
    ]


def test_fresh_source_manifest_binds_hashed_private_city_authorities(tmp_path: Path) -> None:
    layouts_root = tmp_path / "layouts"
    private_city_root = tmp_path / "private-city-authority"
    for index in range(3):
        ancestor = layouts_root / f"ancestor-{index:02d}"
        city_root = ancestor / "splits" / "calibration" / f"city-{index:02d}"
        city = {"layout_id": f"city-{index:02d}", "layout_hash": f"{index:064x}"}
        write_json(city_root / "scene_authority" / "cityspec.json", city)
        write_json(
            city_root / "evaluator_private" / "episodes" / "episode-0000.json",
            {"episode_id": f"episode-{index:02d}"},
        )
        write_json(
            ancestor / "development_layout_manifest.json",
            {
                "formal_score_eligible": False,
                "public_boundary_audit": {"status": "PASS"},
            },
        )
        private_city = {
            **city,
            "split": "calibration",
            "spawn_grammar": "intersection",
            "family_private": "grid",
            "generation_seed": index,
        }
        write_json(
            private_city_root / f"calibration-ancestor-{index:02d}-city.json",
            private_city,
        )

    source_path = tmp_path / "source-v3.json"
    source = build(layouts_root, source_path, private_city_root=private_city_root)

    for index, record in enumerate(source["records"]):
        private_path = source_path.parent / record["private_city_source_path"]
        assert private_path.is_file()
        assert record["private_city_source_sha256"] == content_hash(
            {
                "layout_id": f"city-{index:02d}",
                "layout_hash": f"{index:064x}",
                "split": "calibration",
                "spawn_grammar": "intersection",
                "family_private": "grid",
                "generation_seed": index,
            }
        )
