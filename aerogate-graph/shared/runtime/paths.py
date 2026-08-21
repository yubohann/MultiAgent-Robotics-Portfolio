"""Path helpers for the gate-only experiment package."""

from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = EXPERIMENT_ROOT / "assets"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
RUNTIME_ROOT = OUTPUT_ROOT / "runtime"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
REPORTS_ROOT = OUTPUT_ROOT / "reports"


def ensure_project_on_path(*extra_paths: Path | str) -> Path:
    """Make this portable package importable when scripts are run by path."""

    for path in (EXPERIMENT_ROOT, *extra_paths):
        text = str(Path(path).resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    return EXPERIMENT_ROOT

