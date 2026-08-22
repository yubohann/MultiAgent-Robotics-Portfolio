"""Versioned sensor execution profiles and event-triggered render policy."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import content_hash


@dataclass(frozen=True)
class SensorExecutionProfile:
    """One immutable observation/render contract for a single result batch."""

    profile_id: str
    execution_level: str
    capability_profile: str
    geometry_update_hz: float
    rgb_enabled: bool
    depth_enabled: bool
    instance_segmentation_enabled: bool
    render_on_observe_only: bool
    render_hz: float | None
    width_px: int | None
    height_px: int | None
    horizontal_fov_deg: float | None
    vertical_fov_deg: float | None

    def validate(self) -> None:
        if self.execution_level not in {"L0", "L1", "L2"}:
            raise ValueError("unknown sensor execution level")
        if self.capability_profile not in {"G1", "P1"}:
            raise ValueError("unknown sensor capability profile")
        if self.geometry_update_hz <= 0.0:
            raise ValueError("geometry_update_hz must be positive")
        rendered = self.rgb_enabled or self.depth_enabled or self.instance_segmentation_enabled
        if self.capability_profile == "G1" and rendered:
            raise ValueError("G1 geometry profile may not expose rendered image streams")
        if self.execution_level == "L1" and rendered:
            raise ValueError("L1 geometry scoring may not carry L2 render streams")
        if rendered:
            if not self.render_on_observe_only:
                raise ValueError("rendered profiles must be event-triggered at OBSERVE/dwell")
            if self.render_hz is None or not 2.0 <= self.render_hz <= 5.0:
                raise ValueError("rendered OBSERVE profile must use a declared 2-5 Hz frequency")
            if not self.width_px or not self.height_px:
                raise ValueError("rendered profile must declare image dimensions")
            if not self.horizontal_fov_deg or not self.vertical_fov_deg:
                raise ValueError("rendered profile must declare both camera FoVs")
        elif any(
            value is not None
            for value in (
                self.render_hz,
                self.width_px,
                self.height_px,
                self.horizontal_fov_deg,
                self.vertical_fov_deg,
            )
        ):
            raise ValueError("non-rendered profile may not carry camera configuration")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "org.aerocity.bench.sensor-execution-profile.v1",
            "profile_id": self.profile_id,
            "execution_level": self.execution_level,
            "capability_profile": self.capability_profile,
            "geometry_update_hz": self.geometry_update_hz,
            "rgb_enabled": self.rgb_enabled,
            "depth_enabled": self.depth_enabled,
            "instance_segmentation_enabled": self.instance_segmentation_enabled,
            "render_on_observe_only": self.render_on_observe_only,
            "render_hz": self.render_hz,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
        }


G1_L1_GEOMETRY = SensorExecutionProfile(
    profile_id="g1-l1-geometry-ray-5hz-v1",
    execution_level="L1",
    capability_profile="G1",
    geometry_update_hz=5.0,
    rgb_enabled=False,
    depth_enabled=False,
    instance_segmentation_enabled=False,
    render_on_observe_only=False,
    render_hz=None,
    width_px=None,
    height_px=None,
    horizontal_fov_deg=None,
    vertical_fov_deg=None,
)

P1_L2_RGBD_OBSERVE_96X69 = SensorExecutionProfile(
    profile_id="p1-l2-rgbd-observe-2hz-96x69-v1",
    execution_level="L2",
    capability_profile="P1",
    geometry_update_hz=5.0,
    rgb_enabled=True,
    depth_enabled=True,
    instance_segmentation_enabled=False,
    render_on_observe_only=True,
    render_hz=2.0,
    width_px=96,
    height_px=69,
    horizontal_fov_deg=68.0,
    vertical_fov_deg=51.0,
)

P1_L2_RGBD_INSTANCE_OBSERVE_160X120 = SensorExecutionProfile(
    profile_id="p1-l2-rgbd-instance-observe-2hz-160x120-v1",
    execution_level="L2",
    capability_profile="P1",
    geometry_update_hz=5.0,
    rgb_enabled=True,
    depth_enabled=True,
    instance_segmentation_enabled=True,
    render_on_observe_only=True,
    render_hz=2.0,
    width_px=160,
    height_px=120,
    horizontal_fov_deg=68.0,
    vertical_fov_deg=51.0,
)


def render_due(
    profile: SensorExecutionProfile,
    *,
    action_kind: str,
    dwell_elapsed_s: float,
    last_render_elapsed_s: float | None,
) -> bool:
    """Return whether an L2 render is due during one legal OBSERVE dwell."""

    profile.validate()
    if not (profile.rgb_enabled or profile.depth_enabled or profile.instance_segmentation_enabled):
        return False
    if action_kind != "OBSERVE" or dwell_elapsed_s < 0.0:
        return False
    if last_render_elapsed_s is None:
        return True
    assert profile.render_hz is not None
    return dwell_elapsed_s - last_render_elapsed_s >= (1.0 / profile.render_hz) - 1.0e-9


def validate_single_sensor_profile(profiles: list[SensorExecutionProfile]) -> str:
    """Reject a batch that silently mixes resolutions, FoV, or render policy."""

    if not profiles:
        raise ValueError("sensor profile batch is empty")
    fingerprints = {profile.fingerprint for profile in profiles}
    if len(fingerprints) != 1:
        raise ValueError("a result batch may not mix sensor execution profiles")
    return next(iter(fingerprints))
