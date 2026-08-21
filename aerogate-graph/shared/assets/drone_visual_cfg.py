"""Visual asset configuration for the local 5_in_drone asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FiveInDroneVisualCfg:
    """Visual metadata for the real 5_in_drone USD."""

    usd_path: Path = Path(__file__).resolve().parents[2] / "assets" / "5_in_drone" / "5_in_drone.usd"
    rotor_joint_names: tuple[str, str, str, str] = ("m1_joint", "m2_joint", "m3_joint", "m4_joint")
    rotor_spin_velocities_rad_s: tuple[float, float, float, float] = (200.0, -200.0, 200.0, -200.0)
    body_name: str = "body"
    visual_only_notes: str = "The gate package renders the real 5_in_drone USD with a fixed-height shell pose."

    def joint_velocity_targets(self, spin_scale: float = 0.0) -> dict[str, float]:
        scale = float(spin_scale)
        return {
            joint_name: base_speed * scale
            for joint_name, base_speed in zip(self.rotor_joint_names, self.rotor_spin_velocities_rad_s)
        }


FIVE_IN_DRONE_VISUAL_CFG = FiveInDroneVisualCfg()

# Compatibility aliases for older code paths.
LegacyDroneVisualAssetCfg = FiveInDroneVisualCfg
LEGACY_DRONE_VISUAL_CFG = FIVE_IN_DRONE_VISUAL_CFG
