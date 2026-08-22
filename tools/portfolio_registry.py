"""Load and validate the portfolio's machine-readable project registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class RegistryError(ValueError):
    """Raised when the checked-in portfolio registry is structurally invalid."""


@dataclass(frozen=True)
class ProjectEntry:
    """One independently navigable project in the portfolio."""

    identifier: str
    title: str
    directory: Path
    readme: Path
    category: str
    verification: str | None
    documents: tuple[Path, ...]


@dataclass(frozen=True)
class PortfolioRegistry:
    """Validated root metadata used by portfolio discovery and local checks."""

    root: Path
    entry_documents: tuple[Path, ...]
    projects: tuple[ProjectEntry, ...]

    @property
    def documents(self) -> tuple[Path, ...]:
        """Return the unique, ordered set of curated documents."""

        ordered: list[Path] = []
        for document in (*self.entry_documents, *(document for project in self.projects for document in project.documents)):
            if document not in ordered:
                ordered.append(document)
        return tuple(ordered)

    @property
    def verification_keys(self) -> tuple[str, ...]:
        """Return stable keys for projects with a root-level lightweight check."""

        return tuple(project.verification for project in self.projects if project.verification is not None)

    def project_for_verification(self, verification: str) -> ProjectEntry:
        """Find a project by its root check key."""

        for project in self.projects:
            if project.verification == verification:
                return project
        raise RegistryError(f"unknown portfolio verification key: {verification}")


def load_registry(root: Path = ROOT) -> PortfolioRegistry:
    """Read the JSON registry and reject ambiguous or unsafe project metadata."""

    registry_path = root / "tools" / "portfolio_registry.json"
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read {registry_path.relative_to(root)}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RegistryError("registry root must be a JSON object")
    if raw.get("schema_version") != 1:
        raise RegistryError("registry schema_version must be 1")

    entry_documents = _paths(raw.get("entry_documents"), root=root, field="entry_documents")
    raw_projects = raw.get("projects")
    if not isinstance(raw_projects, list) or not raw_projects:
        raise RegistryError("projects must be a non-empty JSON array")

    projects = tuple(_project(entry, root=root, index=index) for index, entry in enumerate(raw_projects))
    _assert_unique((project.identifier for project in projects), "project id")
    _assert_unique(
        (project.verification for project in projects if project.verification is not None),
        "verification key",
    )
    _assert_unique((project.directory for project in projects), "project directory")
    _assert_unique((document for project in projects for document in project.documents), "project document")
    return PortfolioRegistry(root=root, entry_documents=entry_documents, projects=projects)


def _project(raw: Any, *, root: Path, index: int) -> ProjectEntry:
    if not isinstance(raw, dict):
        raise RegistryError(f"projects[{index}] must be a JSON object")
    identifier = _string(raw.get("id"), f"projects[{index}].id")
    title = _string(raw.get("title"), f"projects[{index}].title")
    category = _string(raw.get("category"), f"projects[{index}].category")
    directory = _path(raw.get("directory"), root=root, field=f"projects[{index}].directory")
    if not directory.is_dir():
        raise RegistryError(f"project directory does not exist: {directory.relative_to(root)}")
    readme = _path(raw.get("readme"), root=root, field=f"projects[{index}].readme")
    try:
        readme.relative_to(directory)
    except ValueError as exc:
        raise RegistryError(f"projects[{index}].readme must be inside its project directory") from exc
    documents = _paths(raw.get("documents"), root=root, field=f"projects[{index}].documents")
    if readme not in documents:
        raise RegistryError(f"projects[{index}].readme must be listed in documents")
    verification = raw.get("verification")
    if verification is not None:
        verification = _string(verification, f"projects[{index}].verification")
    return ProjectEntry(
        identifier=identifier,
        title=title,
        directory=directory,
        readme=readme,
        category=category,
        verification=verification,
        documents=documents,
    )


def _paths(value: Any, *, root: Path, field: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise RegistryError(f"{field} must be a non-empty JSON array")
    paths = tuple(_path(item, root=root, field=field) for item in value)
    _assert_unique(paths, field)
    return paths


def _path(value: Any, *, root: Path, field: str) -> Path:
    text = _string(value, field)
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RegistryError(f"{field} escapes the repository root: {text}") from exc
    return candidate


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{field} must be a non-empty string")
    return value


def _assert_unique(values: Any, label: str) -> None:
    collected = tuple(values)
    if len(collected) != len(set(collected)):
        raise RegistryError(f"duplicate {label} in registry")
