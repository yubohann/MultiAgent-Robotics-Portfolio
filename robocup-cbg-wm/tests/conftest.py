from __future__ import annotations

import sys
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[1] / "sim" / "rl"
sys.path.insert(0, str(RL_ROOT))
