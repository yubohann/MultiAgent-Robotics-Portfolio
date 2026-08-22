from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_entrypoint_pins_project_virtual_environment() -> None:
    source = (ROOT / "scripts" / "run_protocol_python.ps1").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in source
    assert "sys.executable" in source
    assert "Inkscape" not in source


def test_isaac_entrypoint_pins_isaaclab_environment() -> None:
    source = (ROOT / "scripts" / "run_isaac_python.ps1").read_text(encoding="utf-8")
    assert "env_isaaclab\\python.exe" in source
    assert "sys.executable" in source
    assert "Inkscape" not in source
