#!/usr/bin/env python3
"""Validate repository files without requiring a running ROS graph."""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = (
    ROOT / "README.md",
    ROOT / "tools" / "export_run_metrics.py",
    ROOT / "tools" / "plot_run_metrics.py",
)


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit("missing required project files: " + ", ".join(missing))
    packages = sorted((ROOT / "src").glob("*/package.xml"))
    if not packages:
        raise SystemExit("no ROS 2 package.xml files found")
    for package in packages:
        ET.parse(package)
    prohibited = re.compile(r"\b" + "n" + "uc" + r"\b", re.IGNORECASE)
    for path in ROOT.glob("src/**/*"):
        if not path.is_file() or "vendor_livox_ros_driver2" in path.parts:
            continue
        if path.suffix.lower() not in {".md", ".py", ".yaml", ".yml", ".json", ".txt", ".xml"}:
            continue
        if prohibited.search(path.read_text(encoding="utf-8", errors="replace")):
            raise SystemExit(f"prohibited legacy computer label found in: {path.relative_to(ROOT)}")
    for config in (ROOT / "src").glob("*/config/*local*"):
        if config.is_file() and config.suffix in {".yaml", ".json"}:
            text = config.read_text(encoding="utf-8", errors="replace")
            if "192.168.1." in text and "template" not in config.name.lower():
                raise SystemExit(f"example network address found in runtime config: {config}")
    print(f"validated {len(packages)} ROS 2 package manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
