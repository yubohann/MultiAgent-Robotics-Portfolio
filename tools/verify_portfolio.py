"""Verify the curated portfolio entry documents without project dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from portfolio_registry import RegistryError, load_registry


ROOT = Path(__file__).resolve().parents[1]
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
    try:
        registry = load_registry(ROOT)
    except RegistryError as exc:
        print(f"Portfolio registry check failed: {exc}")
        return 1
    errors = [error for document in registry.documents for error in validate_document(document)]
    if errors:
        print("Portfolio integrity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Portfolio integrity check passed for "
        f"{len(registry.documents)} documents across {len(registry.projects)} registered projects."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
