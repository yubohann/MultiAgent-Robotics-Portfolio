"""Global config entry for the gate-only 2D experiment package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT
LOCAL_ASSETS_ROOT = EXPERIMENT_ROOT / "assets"


@dataclass(frozen=True)
class SharedExperimentConfig:
    """Top-level constants shared by the single- and multi-drone gate tracks."""

    experiment_name: str = "aerogate_graph"
    fixed_flight_height_m: float = 4.0
    planar_dt_s: float = 0.1
    planar_max_speed_mps: float = 3.50
    planar_max_accel_mps2: float = 2.45
    min_agents: int = 1
    max_agents_soft: int = 12
    max_fixed_team_agents: int = 34
    use_legacy_ppo: bool = False
    default_gate_post_collision_radius_m: float = 2.5
    default_gate_post_canopy_height_m: float = 8.0

    drone_asset_file: Path = LOCAL_ASSETS_ROOT / "5_in_drone" / "5_in_drone.usd"
    gate_asset_file: Path = LOCAL_ASSETS_ROOT / "gate" / "gate.usd"
    gate_layout_file: Path = LOCAL_ASSETS_ROOT / "gate_scene_layouts.py"


GLOBAL_CONFIG = SharedExperimentConfig()
