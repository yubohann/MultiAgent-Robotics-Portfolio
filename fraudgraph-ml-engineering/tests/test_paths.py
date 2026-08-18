import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_ml_engineering.paths import ARTIFACTS_ROOT, CACHE_ROOT, CONFIG_ROOT, DATA_ROOT, GRAPH_ROOT, REPO_ROOT


def test_paths_resolve_within_repo():
    for p in (REPO_ROOT, DATA_ROOT, GRAPH_ROOT, CACHE_ROOT, CONFIG_ROOT, ARTIFACTS_ROOT):
        assert str(REPO_ROOT) in str(p)


def test_package_imports():
    import fraud_ml_engineering  # noqa: F401
    import fraud_ml_engineering.experiment_protocol  # noqa: F401
    import fraud_ml_engineering.run_artifacts  # noqa: F401


def test_main_imports():
    from fraud_ml_engineering.__main__ import parse_args

    args = parse_args(["--dry-run"])
    assert args.dry_run is True