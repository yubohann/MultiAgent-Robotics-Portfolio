from __future__ import annotations

import importlib.util
from pathlib import Path

from aerocity_bench.isaaclab_paths import discover_isaaclab_paths


def _load_tool(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quadrotor_tools_import_from_a_shallow_checkout() -> None:
    root = Path(__file__).parents[1]
    for index, filename in enumerate(
        (
            "quadrotor_physics_preflight.py",
            "cf2x_l1_fleet_preflight.py",
            "quadrotor_l1_vertical_slice.py",
        )
    ):
        module = _load_tool(root / "tools" / filename, f"portable_quadrotor_tool_{index}")
        assert module.BENCH_ROOT == root
        assert module.ISAACLAB_ROOT is None
        assert module.ISAACLAB_SOURCE_ROOT is None


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
