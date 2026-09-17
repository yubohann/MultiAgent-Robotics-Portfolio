from __future__ import annotations

from pathlib import Path

from aerocity_bench.bridge.isaaclab_paths import discover_isaaclab_paths


def test_explicit_isaaclab_override_requires_a_real_source_tree(
    tmp_path: Path, monkeypatch
) -> None:
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "source" / "isaaclab").mkdir(parents=True)
    (isaaclab / "isaac_drone_racer").mkdir()
    monkeypatch.setenv("AEROCITY_ISAACLAB_ROOT", str(isaaclab))

    paths = discover_isaaclab_paths(tmp_path / "detached-benchmark")

    assert paths.isaaclab_root == isaaclab.resolve()
    assert paths.drone_project_root == (isaaclab / "isaac_drone_racer").resolve()
    assert paths.source_root == (isaaclab / "source").resolve()
