"""Path helpers for the gate-only experiment package."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = EXPERIMENT_ROOT / "assets"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
RUNTIME_ROOT = OUTPUT_ROOT / "runtime"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
REPORTS_ROOT = OUTPUT_ROOT / "reports"

