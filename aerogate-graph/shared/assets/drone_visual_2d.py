"""Bridge between planar kinematics and the 5_in_drone visual asset."""

from __future__ import annotations

from dataclasses import dataclass

from shared.assets.drone_visual_cfg import FIVE_IN_DRONE_VISUAL_CFG, FiveInDroneVisualCfg
from shared.core.fixed_height import FixedHeightConfig, planar_xy_to_fixed_xyz
from shared.core.kinematics_2d import (
    KinematicState2D,
    Kinematics2DConfig,
    Kinematics2DUpdater,
    PlanarVelocityCommand2D,
)


@dataclass(frozen=True)
class DroneVisualCommand2D:
    """Visual command for the real drone asset."""

    root_position_xyz: tuple[float, float, float]
    root_yaw_rad: float
    joint_velocity_targets: dict[str, float]
    planar_speed_mps: float


class DroneVisualKinematicController2D:
    """Apply planar kinematics while keeping the drone on a fixed height plane."""

    def __init__(
        self,
        *,
        kinematics_config: Kinematics2DConfig | None = None,
        height_config: FixedHeightConfig | None = None,
        drone_asset_cfg: FiveInDroneVisualCfg | None = None,
    ) -> None:
        self.kinematics_config = kinematics_config or Kinematics2DConfig()
        self.height_config = height_config or FixedHeightConfig()
        self.drone_asset_cfg = drone_asset_cfg or FIVE_IN_DRONE_VISUAL_CFG
        self._updater = Kinematics2DUpdater(self.kinematics_config)

    def build_visual_command(self, state: KinematicState2D, spin_scale: float = 0.0) -> DroneVisualCommand2D:
        """Convert planar state into a fixed-height visual command."""

        root_position = planar_xy_to_fixed_xyz(
            state.x_m,
            state.y_m,
            self.height_config,
        )
        planar_speed = (state.vx_mps**2 + state.vy_mps**2) ** 0.5
        return DroneVisualCommand2D(
            root_position_xyz=root_position,
            root_yaw_rad=float(state.yaw_rad),
            joint_velocity_targets=self.drone_asset_cfg.joint_velocity_targets(spin_scale),
            planar_speed_mps=float(planar_speed),
        )

    def step(
        self,
        state: KinematicState2D,
        command: PlanarVelocityCommand2D,
        spin_scale: float = 0.0,
    ) -> tuple[KinematicState2D, DroneVisualCommand2D]:
        """Advance planar motion and build the matching visual command."""

        next_state = self._updater.step(state, command)
        return next_state, self.build_visual_command(next_state, spin_scale=spin_scale)

