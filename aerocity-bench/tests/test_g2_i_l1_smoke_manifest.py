from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "build_g2_i_l1_smoke_manifest.py"
    spec = importlib.util.spec_from_file_location("g2_i_l1_smoke_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_public_layout(root: Path, *, target_count_public: bool = False) -> None:
    episode_dir = root / "method_public" / "episodes"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode-0000.json").write_text(
        json.dumps(
            {
                "schema": "org.aerocity.bench.episode-public.ordinary.v1",
                "episode_id": "episode-public-smoke",
                "target_count_public": target_count_public,
                "target_process_public": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "method_public" / "task_spec.json").write_text(
        json.dumps(
            {
                "task_track": "G2-I",
                "inspection_prior_level": "full-cells",
            }
        ),
        encoding="utf-8",
    )


def test_smoke_manifest_precommits_one_public_hidden_target_episode(tmp_path: Path) -> None:
    module = _load_module()
    layout_root = tmp_path / "layout"
    output_root = tmp_path / "evidence"
    release_config = tmp_path / "release.json"
    _write_public_layout(layout_root)
    release_config.write_text("{}", encoding="utf-8")

    result = module.build_manifest(
        layout_root=layout_root,
        release_config=release_config,
        layout_ancestor="ancestor-00",
        method_id="atlas-region-greedy",
        episode_name="episode-0000.json",
        output_root=output_root,
        purpose="test",
    )

    panel = json.loads((output_root / "panel.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "evidence-manifest.json").read_text(encoding="utf-8"))
    assert panel == result["panel"]
    assert panel["precommitted_before_execution"] is True
    assert panel["formal_score_eligible"] is False
    assert manifest == result["manifest"]
    assert manifest["episodes"][0]["layout_root"].startswith("..")
    assert not Path(manifest["episodes"][0]["layout_root"]).is_absolute()

    with pytest.raises(FileExistsError, match="overwrite"):
        module.build_manifest(
            layout_root=layout_root,
            release_config=release_config,
            layout_ancestor="ancestor-00",
            method_id="atlas-region-greedy",
            episode_name="episode-0000.json",
            output_root=output_root,
            purpose="test",
        )


def test_smoke_manifest_rejects_public_target_leak_and_private_method(tmp_path: Path) -> None:
    module = _load_module()
    layout_root = tmp_path / "layout"
    release_config = tmp_path / "release.json"
    release_config.write_text("{}", encoding="utf-8")
    _write_public_layout(layout_root, target_count_public=True)

    with pytest.raises(ValueError, match="target information"):
        module.build_manifest(
            layout_root=layout_root,
            release_config=release_config,
            layout_ancestor="ancestor-00",
            method_id="atlas-region-greedy",
            episode_name="episode-0000.json",
            output_root=tmp_path / "leaked",
            purpose="test",
        )

    _write_public_layout(tmp_path / "private-layout")
    with pytest.raises(ValueError, match="private or fixture"):
        module.build_manifest(
            layout_root=tmp_path / "private-layout",
            release_config=release_config,
            layout_ancestor="ancestor-00",
            method_id="private-oracle",
            episode_name="episode-0000.json",
            output_root=tmp_path / "private-method",
            purpose="test",
        )
