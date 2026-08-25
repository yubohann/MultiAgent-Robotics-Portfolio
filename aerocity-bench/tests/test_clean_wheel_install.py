from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash


def _tool_module() -> object:
    root = Path(__file__).parents[1]
    path = root / "tools" / "verify_clean_wheel_install.py"
    spec = importlib.util.spec_from_file_location("verify_clean_wheel_install", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_wheel_install_rejects_non_wheel_input(tmp_path: Path) -> None:
    module = _tool_module()
    artifact = tmp_path / "not-a-wheel.txt"
    artifact.write_text("invalid", encoding="utf-8")

    with pytest.raises(ValueError, match=".whl"):
        module.verify_clean_wheel_install(Path(__file__), artifact, tmp_path / "manifest.json")


def test_clean_install_uses_the_cli_json_flag() -> None:
    module = _tool_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert '"list-presets", "--json"' in source
    assert "--as-json" not in source


def test_environment_binding_rejects_tampering_and_preserves_source_state(tmp_path: Path) -> None:
    module = _tool_module()
    manifest = {
        "schema": "org.aerocity.bench.release-environment-manifest.v1",
        "source_tree": {
            "state": "DIRTY",
            "source_commit": "UNCOMMITTED-DEVELOPMENT",
            "official_release_binding": "REJECTED",
        },
    }
    manifest["manifest_hash"] = content_hash(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    binding = module._environment_binding(path)

    assert binding["source_tree"]["official_release_binding"] == "REJECTED"
    manifest["source_tree"]["state"] = "CLEAN"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        module._environment_binding(path)
