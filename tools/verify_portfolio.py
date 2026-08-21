"""Verify the curated portfolio entry documents without project dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "PORTFOLIO_GUIDE.md",
    ROOT / "rivermark" / "README.md",
    ROOT / "rivermark" / "code" / "README.md",
    ROOT / "robocup-cbg-wm" / "README.md",
    ROOT / "robocup-cbg-wm" / "docs" / "capability_boundaries.md",
    ROOT / "robocup-cbg-wm" / "docs" / "media" / "README.md",
    ROOT / "robocon-mid360-autonomy-stack" / "README.md",
    ROOT / "aerogate-graph" / "README.md",
    ROOT / "aerogate-graph" / "docs" / "ARCHITECTURE.md",
    ROOT / "aerogate-graph" / "docs" / "REPRODUCIBILITY.md",
    ROOT / "fraudgraph-ml-engineering" / "README.md",
    ROOT / "fraudgraph-ml-engineering" / "docs" / "reproducibility-checklist.md",
    ROOT / "ros2-systematic-learning-notes" / "README.md",
    ROOT / "coursework" / "machine-learning" / "README.md",
    ROOT / "coursework" / "machine-learning" / "README.zh-CN.md",
    ROOT / "coursework" / "machine-learning" / "classic-ml-algorithms" / "README.md",
    ROOT / "coursework" / "machine-learning" / "classic-ml-algorithms" / "README.zh-CN.md",
    ROOT / "coursework" / "machine-learning" / "ml-assignment-3" / "README.md",
    ROOT / "coursework" / "machine-learning" / "ml-assignment-3" / "README.zh-CN.md",
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r'(?:href|src)="([^"]+)"')
EXTERNAL_PREFIXES = ("#", "http://", "https://", "mailto:", "data:")


def is_local_target(value: str) -> bool:
    return bool(value) and not value.startswith(EXTERNAL_PREFIXES)


def resolve_target(document: Path, value: str) -> Path:
    target = value.split("#", 1)[0].split("?", 1)[0].strip(" <>")
    return (document.parent / target).resolve()


def validate_document(document: Path) -> list[str]:
    errors: list[str] = []
    relative = document.relative_to(ROOT)
    if not document.is_file():
        return [f"missing document: {relative}"]
    text = document.read_text(encoding="utf-8")
    if "\ufffd" in text:
        errors.append(f"replacement character found: {relative}")
    if "<<<<<<<" in text or ">>>>>>>" in text or "=======" in text:
        errors.append(f"unresolved conflict marker: {relative}")
    targets = MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text)
    for value in targets:
        value = value.strip()
        if not is_local_target(value):
            continue
        target = resolve_target(document, value)
        if not target.exists():
            errors.append(f"broken local link: {relative} -> {value}")
    return errors


def main() -> int:
    errors = [error for document in DOCUMENTS for error in validate_document(document)]
    if errors:
        print("Portfolio integrity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Portfolio integrity check passed for {len(DOCUMENTS)} documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
