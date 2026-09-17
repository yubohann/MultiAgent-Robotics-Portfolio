"""Run dependency-light integrity checks before installing the training stack."""

from __future__ import annotations

import argparse
import ast
import compileall
import re
import sys
from collections.abc import Callable
from pathlib import Path

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
    "src/fraud_ml_engineering/caching.py",
    "src/fraud_ml_engineering/experiment_tools/__init__.py",
    "src/fraud_ml_engineering/experiment_tools/generate_auditable_comparison_report.py",
    "src/fraud_ml_engineering/experiment_tools/generate_hybrid_paper_package.py",
    "src/fraud_ml_engineering/experiment_tools/record_run_manifest.py",
    "src/fraud_ml_engineering/experiment_tools/run_hybrid_fusion_ablation.py",
    "src/fraud_ml_engineering/experiment_tools/run_hybrid_low_label_mechanism_ablation.py",
    "src/fraud_ml_engineering/experiment_tools/run_hybrid_mainline_protocol.py",
    "src/fraud_ml_engineering/experiment_tools/run_ieee_acceptance_matrix.py",
    "src/fraud_ml_engineering/experiment_tools/run_ieee_splitgnn_tuning.py",
    "src/fraud_ml_engineering/experiment_tools/run_splitgnn_smoke_suite.py",
    "scripts/generate_auditable_comparison_report.py",
    "scripts/generate_hybrid_paper_package.py",
    "scripts/record_run_manifest.py",
    "scripts/run_hybrid_fusion_ablation.py",
    "scripts/run_hybrid_low_label_mechanism_ablation.py",
    "scripts/run_hybrid_mainline_protocol.py",
    "scripts/run_ieee_acceptance_matrix.py",
    "scripts/run_ieee_splitgnn_tuning.py",
    "scripts/run_splitgnn_smoke_suite.py",
)
LEGACY_PATH_MARKERS = ("C:\\Users\\\\", "D:\\", "dataset.SplitGNN")
LEGACY_ARTIFACT_MARKERS = ("hybrid_mafrl",)
PLACEHOLDER_MARKERS = ("TODO", "FIXME", "PLACEHOLDER", "TBD")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def check_required_files() -> str | None:
    missing = [path for path in REQUIRED_FILES if not (REPO_ROOT / path).is_file()]
    if missing:
        return f"Missing required files: {', '.join(missing)}"
    return None


def check_configuration_inventory() -> str | None:
    splitgnn_configs = sorted((REPO_ROOT / "configs" / "splitgnn").glob("*.yaml"))
    experiment_configs = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
    if len(splitgnn_configs) < 7 or len(experiment_configs) < 3:
        return f"Expected at least 7 SplitGNN and 3 experiment configs, got {len(splitgnn_configs)} and {len(experiment_configs)}"
    if any(not _read(path).strip() for path in splitgnn_configs + experiment_configs):
        return "An experiment configuration file is empty"
    return None


def check_package_compiles() -> str | None:
    if not compileall.compile_dir(str(REPO_ROOT / "src"), quiet=1):
        return "Python compilation failed under src/"
    if not compileall.compile_dir(str(REPO_ROOT / "scripts"), quiet=1):
        return "Python compilation failed under scripts/"
    return None


def check_internal_imports() -> str | None:
    local_modules = {path.stem for path in PACKAGE_ROOT.glob("*.py")}
    source_paths = list(PACKAGE_ROOT.glob("*.py")) + list((PACKAGE_ROOT / "experiment_tools").glob("*.py"))
    violations: list[str] = []
    for source_path in source_paths:
        tree = ast.parse(_read(source_path), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local_modules:
                violations.append(f"{source_path.name}: from {node.module} import ...")
    if violations:
        return "Bare internal imports: " + "; ".join(violations)
    return None


def check_source_hygiene() -> str | None:
    source_paths = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py"))
    for source_path in source_paths:
        if source_path.resolve() == Path(__file__).resolve():
            continue
        text = _read(source_path)
        for marker in LEGACY_PATH_MARKERS:
            if marker in text:
                return f"Legacy machine-specific path {marker!r} in {source_path}"
        for marker in LEGACY_ARTIFACT_MARKERS:
            if marker in text:
                return f"Retired artifact prefix {marker!r} in {source_path}"
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            return f"Placeholder marker in production path: {source_path}"
    return None


def check_gitignore_contract() -> str | None:
    ignore_rules = _read(REPO_ROOT / ".gitignore")
    required_rules = ("data/**", "artifacts/**", ".venv/", "dist/", "build/")
    missing = [rule for rule in required_rules if rule not in ignore_rules]
    if missing:
        return f"Missing ignore rules: {', '.join(missing)}"
    return None


def check_version_metadata() -> str | None:
    pyproject = _read(REPO_ROOT / "pyproject.toml")
    changelog = _read(REPO_ROOT / "CHANGELOG.md")
    citation = _read(REPO_ROOT / "CITATION.cff")
    version_match = re.search(r"^version\s*=\s*\"([^\"]+)\"", pyproject, flags=re.MULTILINE)
    if version_match is None:
        return "No project version found in pyproject.toml"
    version = version_match.group(1)
    if f"## {version} " not in changelog or f"version: {version}" not in citation:
        return f"Version {version} is not consistent across metadata"
    return None


def run_checks() -> list[str]:
    checks: tuple[tuple[str, Callable[[], str | None]], ...] = (
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
        failure = check()
        if failure is None:
            print(f"PASS {label}")
        else:
            failures.append(f"FAIL {label}: {failure}")
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
