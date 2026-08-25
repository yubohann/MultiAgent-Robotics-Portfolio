from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool_module() -> object:
    root = Path(__file__).parents[1]
    path = root / "tools" / "audit_release_assets.py"
    spec = importlib.util.spec_from_file_location("audit_release_assets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_asset_audit_keeps_local_paths_and_cf2x_out_of_evidence(monkeypatch) -> None:
    module = _tool_module()

    class _Config:
        raw = {
            "assets": {
                "bundle": "open_city_cc0_assets_20260729",
                "allowlist": ["asset-a", "asset-b"],
            }
        }

    class _Lock:
        bundle = "open_city_cc0_assets_20260729"
        registry_hash = "a" * 64
        records = {"asset-a": object(), "asset-b": object()}

    class _Evidence:
        manifest_hash = "b" * 64
        license_snapshot_hash = "c" * 64

    monkeypatch.setattr(module, "load_ordinary_config", lambda _: _Config())
    monkeypatch.setattr(
        module,
        "load_official_cc0_lock",
        lambda *_: (_Lock(), _Evidence(), {"checked_usd_layers": 2}),
    )
    monkeypatch.setattr(module, "file_hash", lambda _: "d" * 64)

    report = module.audit_release_assets(Path("config.json"), Path("E:/private-assets"))

    assert report["status"] == "PASS"
    assert report["cf2x_redistributed"] is False
    assert report["nvidia_content_redistributed"] is False
    assert "private-assets" not in str(report)
