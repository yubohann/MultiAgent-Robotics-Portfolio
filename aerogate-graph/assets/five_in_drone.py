"""Local spawn helpers for the 5_in_drone asset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ASSETS_ROOT = Path(__file__).resolve().parent
DEFAULT_FIVE_IN_DRONE_USD = ASSETS_ROOT / "5_in_drone" / "5_in_drone.usd"


FIVE_IN_DRONE = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DEFAULT_FIVE_IN_DRONE_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


def spawn_real_drones(
    *,
    drone_positions_xyz: Sequence[Sequence[float]],
    drone_usd_path: str | Path | None = None,
    drone_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[str]:
    """Spawn one or more 5_in_drone USD prims at the requested positions."""

    resolved_drone_path = Path(drone_usd_path) if drone_usd_path is not None else DEFAULT_FIVE_IN_DRONE_USD
    if not resolved_drone_path.exists():
        raise FileNotFoundError(f"Drone USD asset is missing: {resolved_drone_path}")

    drone_cfg = sim_utils.UsdFileCfg(
        usd_path=str(resolved_drone_path),
        scale=tuple(float(value) for value in drone_scale),
    )
    prim_paths: list[str] = []
    for drone_idx, position_xyz in enumerate(drone_positions_xyz):
        prim_path = f"/World/Drones/Drone_{drone_idx:02d}"
        drone_cfg.func(
            prim_path,
            drone_cfg,
            translation=tuple(float(value) for value in position_xyz),
        )
        prim_paths.append(prim_path)
    return prim_paths


# Compatibility alias for earlier code paths.
DEFAULT_DRONE_USD = DEFAULT_FIVE_IN_DRONE_USD
