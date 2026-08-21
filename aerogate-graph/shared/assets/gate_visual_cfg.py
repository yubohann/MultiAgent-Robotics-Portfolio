"""Visual asset configuration for the local gate USD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateVisualCfg:
    usd_path: Path = Path(__file__).resolve().parents[2] / "assets" / "gate" / "gate.usd"
    default_scale_xyz: tuple[float, float, float] = (1.25, 1.25, 1.25)
    nominal_half_width_m: float = 2.0
    nominal_post_radius_m: float = 0.55
    visual_only_notes: str = "Gate collisions stay planar; the USD asset is used for shell visualization."


GATE_VISUAL_CFG = GateVisualCfg()

# Compatibility aliases for older code paths.
LegacyGateVisualAssetCfg = GateVisualCfg
LEGACY_GATE_VISUAL_CFG = GATE_VISUAL_CFG
