"""Access small presets and schemas shipped inside the Python wheel."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

PRESETS = {
    "ordinary-v1-mini": "ordinary-v1-mini.json",
}
SCHEMAS = {
    "ordinary-v3": "release_config_ordinary_v3.schema.json",
}


def _resource_bytes(category: str, filename: str, source_fallback: Path) -> bytes:
    packaged = files("aerocity_bench").joinpath("resources", category, filename)
    if packaged.is_file():
        return packaged.read_bytes()
    if source_fallback.is_file():
        return source_fallback.read_bytes()
    raise FileNotFoundError(f"packaged AeroCityBench resource is absent: {category}/{filename}")


def preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name}; available={sorted(PRESETS)}")
    source = Path(__file__).resolve().parents[2] / "configs" / "releases" / PRESETS[name]
    return json.loads(_resource_bytes("presets", PRESETS[name], source).decode("utf-8"))


def schema(name: str) -> dict[str, Any]:
    if name not in SCHEMAS:
        raise ValueError(f"unknown schema: {name}; available={sorted(SCHEMAS)}")
    source = Path(__file__).resolve().parents[2] / "schemas" / SCHEMAS[name]
    return json.loads(_resource_bytes("schemas", SCHEMAS[name], source).decode("utf-8"))


def write_preset(name: str, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"preset output already exists: {destination}")
    value = preset(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "PASS",
        "preset": name,
        "output": str(destination.resolve()),
    }
