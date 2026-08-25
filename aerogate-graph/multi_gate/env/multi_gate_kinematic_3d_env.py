"""Paper-oriented fixed-height real-3D kinematic variant of the multi-agent gate env."""

from __future__ import annotations

from multi_gate.env.multi_gate_env import MultiGate2DEnv


class MultiGateKinematic3DEnv(MultiGate2DEnv):
    """Keep the proven 2D training dynamics while exposing real-3D scene semantics."""

    @property
    def scene_mode(self) -> str:
        return str(self.multi_config.scene.scene_mode)

    def _build_info(
        self,
        *,
        done_reason: str | None,
        reward_terms: dict[str, float] | None,
    ) -> dict[str, object]:
        info = super()._build_info(done_reason=done_reason, reward_terms=reward_terms)
        info.update(
            {
                "scene_mode": self.scene_mode,
                "render_backend": self.multi_config.scene.render_backend,
                "render_real_gate": bool(self.multi_config.scene.render_real_gate),
                "render_real_drone_shell": bool(self.multi_config.scene.render_real_drone_shell),
                "kinematic_only": bool(self.multi_config.scene.kinematic_only),
                "disable_motors": bool(self.multi_config.scene.disable_motors),
                "fixed_height_locked": bool(self.multi_config.scene.fixed_height_locked),
                "drone_asset": self.multi_config.scene.drone_asset,
                "paper_track": self.multi_config.paper_track,
                "paper_variant": self.multi_config.paper_variant,
                "global_planner_enabled": bool(self.multi_config.reasoning.global_planner_enabled),
                "route_guidance_enabled": bool(self.multi_config.reasoning.route_guidance_enabled),
                "guidance_shadow_mode": bool(getattr(self.multi_config.reasoning, "guidance_shadow_mode", False)),
            }
        )
        return info

