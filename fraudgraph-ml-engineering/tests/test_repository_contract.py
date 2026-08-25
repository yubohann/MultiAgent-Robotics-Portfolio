from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "fraud_ml_engineering"


def test_runtime_paths_resolve_to_repository_layout() -> None:
    from fraud_ml_engineering.paths import ARTIFACTS_ROOT, CACHE_ROOT, CONFIG_ROOT, DATA_ROOT, GRAPH_ROOT, REPO_ROOT as runtime_root

    assert runtime_root == REPO_ROOT
    assert DATA_ROOT == REPO_ROOT / "data"
    assert GRAPH_ROOT == DATA_ROOT / "graphs"
    assert CACHE_ROOT == GRAPH_ROOT / "cache"
    assert CONFIG_ROOT == REPO_ROOT / "configs"
    assert ARTIFACTS_ROOT == REPO_ROOT / "artifacts"


def test_required_project_files_exist() -> None:
    expected = (
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        ".gitignore",
        "docs/architecture.mmd",
        "docs/data-and-reproduction.md",
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
    missing = [path for path in expected if not (REPO_ROOT / path).is_file()]
    assert not missing


def test_configuration_contract_is_present() -> None:
    assert sorted(path.name for path in (REPO_ROOT / "configs" / "splitgnn").glob("*.yaml"))
    assert sorted(path.name for path in (REPO_ROOT / "configs" / "experiments").glob("*.yaml"))


def test_internal_imports_are_package_relative() -> None:
    local_modules = {path.stem for path in PACKAGE_ROOT.glob("*.py")}
    violations: list[str] = []
    for source_path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local_modules:
                violations.append(f"{source_path.name}: from {node.module} import ...")
    assert not violations, "\n".join(violations)


def test_data_and_artifact_directories_are_ignored() -> None:
    ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/**" in ignore_rules
    assert "artifacts/**" in ignore_rules


def test_readme_relative_links_resolve() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)#]+)", readme)
    local_links = [link for link in links if not link.startswith(("http://", "https://", "mailto:"))]
    missing = [link for link in local_links if not (REPO_ROOT / link).is_file()]
    assert not missing, f"README links point to missing files: {missing}"


def test_citation_metadata_matches_project_identity() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'title: "FraudGraph ML Engineering"' in citation
    assert "version: 0.1.0" in citation
    assert "https://github.com/yubohann/fraudgraph-ml-engineering" in citation


def test_quality_workflow_covers_supported_python_versions() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.10", "3.12"]' in workflow
    assert "python scripts/validate_repository.py" in workflow
    assert "python -m pytest" in workflow
    assert "python -m build" in workflow
    assert "python -m pip install --force-reinstall --no-deps dist/*.whl" in workflow
    assert "python -m fraud_ml_engineering --help" in workflow


def test_cli_parser_is_split_into_responsibility_focused_helpers() -> None:
    source = (PACKAGE_ROOT / "main.py").read_text(encoding="utf-8")
    for helper_name in (
        "_add_core_training_arguments",
        "_add_ieee_arguments",
        "_add_dataset_arguments",
        "_add_runtime_control_arguments",
    ):
        assert f"def {helper_name}" in source


def test_application_code_uses_the_fraudgraph_artifact_prefix() -> None:
    source_paths = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").glob("*.py"))
    application_paths = [path for path in source_paths if path.name != "validate_repository.py"]
    stale_paths = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in application_paths
        if "hybrid_mafrl" in path.read_text(encoding="utf-8-sig")
    ]
    assert not stale_paths, f"Retired artifact prefix found in: {stale_paths}"
