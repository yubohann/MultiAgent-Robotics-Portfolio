"""CLI wrapper for the fraud_ml_engineering.experiment_tools.run_hybrid_fusion_ablation module."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.experiment_tools.run_hybrid_fusion_ablation import main

if __name__ == "__main__":
    main()