"""Summarize saved Crossref and OpenAlex responses without altering raw evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _title_relevant(title: str) -> bool:
    task = re.search(
        r"(explor|coverage|informative path|active mapping|active slam)", title, re.I
    )
    platform = re.search(r"(multi|swarm|uav|aerial|robot)", title, re.I)
    return task is not None and platform is not None


def _openalex_rows(payload: dict[str, Any], source_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for work in payload.get("results", []):
        title = str(work.get("title") or "")
        if not _title_relevant(title):
            continue
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        access = work.get("open_access") or {}
        rows.append(
            {
                "year": work.get("publication_year"),
                "title": title,
                "venue": source.get("display_name"),
                "doi": work.get("doi"),
                "citation_count": work.get("cited_by_count", 0),
                "open_access_url": access.get("oa_url"),
                "database": "OpenAlex",
                "source_file": source_file,
            }
        )
    return rows


def _crossref_rows(payload: dict[str, Any], source_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for work in (payload.get("message") or {}).get("items", []):
        title = " ".join(str(value) for value in work.get("title", []))
        if not _title_relevant(title):
            continue
        dates = (work.get("published") or {}).get("date-parts") or [[None]]
        rows.append(
            {
                "year": dates[0][0],
                "title": title,
                "venue": " ".join(str(value) for value in work.get("container-title", [])),
                "doi": work.get("DOI"),
                "citation_count": work.get("is-referenced-by-count", 0),
                "open_access_url": None,
                "database": "Crossref",
                "source_file": source_file,
            }
        )
    return rows


def summarize(directory: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(directory.glob("openalex_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _openalex_rows(payload, path.name):
            key = (str(row["doi"] or "").casefold(), str(row["title"]).casefold())
            by_key[key] = row
    for path in sorted(directory.glob("crossref_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _crossref_rows(payload, path.name):
            key = (str(row["doi"] or "").casefold(), str(row["title"]).casefold())
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = row
            else:
                previous["crossref_source_file"] = path.name
    return sorted(
        by_key.values(),
        key=lambda row: (int(row["year"] or 0), int(row["citation_count"] or 0)),
        reverse=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(args.directory.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
