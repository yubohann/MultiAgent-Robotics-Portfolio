from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_hm3d_download_selection_avoids_redundant_habitat_archives() -> None:
    path = ROOT / "configs" / "external" / "hm3d_v0.2_download_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = {name for values in payload["selections"].values() for name in values}
    assert selected
    assert all("-habitat-" not in name for name in selected)
    assert all(name in payload["resources"] for name in selected)
    assert "hm3d-train-glb-v0.2.tar" in payload["selections"]["formal_exploration_geometry"]
    assert "hm3d-val-glb-v0.2.tar" in payload["selections"]["formal_exploration_geometry"]
    assert set(payload["resources"]) == selected
    assert all(row["kind"] == "geometry_glb" for row in payload["resources"].values())


def test_hm3d_download_selection_never_contains_credentials() -> None:
    path = ROOT / "configs" / "external" / "hm3d_v0.2_download_selection.json"
    text = path.read_text(encoding="utf-8")
    assert "token_id_environment_variable" in text
    assert "token_secret_environment_variable" in text
    assert "password" not in text.casefold()
