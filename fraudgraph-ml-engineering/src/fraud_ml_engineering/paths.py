"""Repository-local paths used by loaders, training, and experiment utilities."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
GRAPH_ROOT = DATA_ROOT / "graphs"
CACHE_ROOT = GRAPH_ROOT / "cache"
CONFIG_ROOT = REPO_ROOT / "configs"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


def ensure_runtime_directories() -> None:
    """Create only the local directories that store generated artifacts."""

    for path in (GRAPH_ROOT, CACHE_ROOT, ARTIFACTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
