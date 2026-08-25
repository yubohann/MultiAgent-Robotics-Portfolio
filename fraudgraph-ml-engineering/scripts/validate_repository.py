"""Run dependency-light integrity checks before installing the training stack."""

from __future__ import annotations

import argparse
import ast
import compileall
import re
import sys
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "fraud_ml_engineering"

REQUIRED_FILES = (
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "docs/architecture.mmd",
    "docs/research-protocol.md",
    "docs/experiment-catalog.md",
    "docs/reproducibility-checklist.md",
    "docs/experiment-manifest.md",
    "docs/comparison-report-schema.md",
    "requirements/requirements-cpu.txt",
    "requirements/requirements-cu121.txt",
    ".github/workflows/quality.yml",
    "Makefile",
)
LEGACY_PATH_MARKERS = ("C:\\Users\\\\", "D:\\", "dataset.SplitGNN")
LEGACY_ARTIFACT_MARKERS = ("hybrid_mafrl",)
PLACEHOLDER_MARKERS = ("TODO", "FIXME", "PLACEHOLDER", "TBD")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (REPO_ROOT / path).is_file()]
    if missing:
        raise AssertionError(f"Missing required files: {', '.join(missing)}")


def check_configuration_inventory() -> None:
    splitgnn_configs = sorted((REPO_ROOT / "configs" / "splitgnn").glob("*.yaml"))
    experiment_configs = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
    if len(splitgnn_configs) < 7 or len(experiment_configs) < 3:
        raise AssertionError(
            f"Expected at least 7 SplitGNN and 3 experiment configs, got {len(splitgnn_configs)} and {len(experiment_configs)}"
        )
    if any(not _read(path).strip() for path in splitgnn_configs + experiment_configs):
        raise AssertionError("An experiment configuration file is empty")


def check_package_compiles() -> None:
    if not compileall.compile_dir(str(REPO_ROOT / "src"), quiet=1):
        raise AssertionError("Python compilation failed under src/")
    if not compileall.compile_dir(str(REPO_ROOT / "scripts"), quiet=1):
        raise AssertionError("Python compilation failed under scripts/")


def check_internal_imports() -> None:
    local_modules = {path.stem for path in PACKAGE_ROOT.glob("*.py")}
    violations: list[str] = []
    for source_path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(_read(source_path), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local_modules:
                violations.append(f"{source_path.name}: from {node.module} import ...")
    if violations:
        raise AssertionError("Bare internal imports: " + "; ".join(violations))


def check_source_hygiene() -> None:
    source_paths = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py"))
    for source_path in source_paths:
        if source_path.resolve() == Path(__file__).resolve():
            continue
        text = _read(source_path)
        for marker in LEGACY_PATH_MARKERS:
            if marker in text:
                raise AssertionError(f"Legacy machine-specific path {marker!r} in {source_path}")
        for marker in LEGACY_ARTIFACT_MARKERS:
            if marker in text:
                raise AssertionError(f"Retired artifact prefix {marker!r} in {source_path}")
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            raise AssertionError(f"Placeholder marker in production path: {source_path}")


def check_gitignore_contract() -> None:
    ignore_rules = _read(REPO_ROOT / ".gitignore")
    required_rules = ("data/**", "artifacts/**", ".venv/", "dist/", "build/")
    missing = [rule for rule in required_rules if rule not in ignore_rules]
    if missing:
        raise AssertionError(f"Missing ignore rules: {', '.join(missing)}")


def check_version_metadata() -> None:
    pyproject = _read(REPO_ROOT / "pyproject.toml")
    changelog = _read(REPO_ROOT / "CHANGELOG.md")
    citation = _read(REPO_ROOT / "CITATION.cff")
    version_match = re.search(r"^version\s*=\s*\"([^\"]+)\"", pyproject, flags=re.MULTILINE)
    if version_match is None:
        raise AssertionError("No project version found in pyproject.toml")
    version = version_match.group(1)
    if f"## {version} " not in changelog or f"version: {version}" not in citation:
        raise AssertionError(f"Version {version} is not consistent across metadata")


def run_checks() -> list[str]:
    checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("required files", check_required_files),
        ("configuration inventory", check_configuration_inventory),
        ("Python compilation", check_package_compiles),
        ("internal imports", check_internal_imports),
        ("source hygiene", check_source_hygiene),
        (".gitignore contract", check_gitignore_contract),
        ("version metadata", check_version_metadata),
    )
    failures: list[str] = []
    for label, check in checks:
        try:
            check()
        except Exception as error:  # pragma: no cover - exercised by CLI failures
            failures.append(f"FAIL {label}: {error}")
        else:
            print(f"PASS {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    args = parser.parse_args()
    failures = run_checks()
    if args.quiet and not failures:
        return 0
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
