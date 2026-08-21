"""Live dynamic-gate layouts, obstacle maps, and per-step caches."""

from __future__ import annotations

import numpy as np

from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D
from shared.core.dynamic_gate_density_2d import gate_posts, generate_gate_layout, live_gate_centers


def _reset_dynamic_gate_layout(self, *, seed: int | None) -> None:
    self._clear_dynamic_gate_runtime_cache()
    if not self._dynamic_gate_enabled:
        self._dynamic_gates = []
        return
    if seed is None:
        layout_seed = int(self._rng.integers(0, 2**31 - 1))
    else:
        layout_seed = int(seed)
    static_layout = (
        float(getattr(self._dynamic_gate_config, "moving_gate_speed_mps", 0.0) or 0.0) <= 1.0e-6
        or float(getattr(self._dynamic_gate_config, "moving_gate_amplitude_m", 0.0) or 0.0) <= 1.0e-6
    )
    self._dynamic_gates = generate_gate_layout(
        gate_count=int(getattr(self._dynamic_gate_config, "gate_count", 0) or 0),
        seed=layout_seed,
        config=self._dynamic_gate_config,
        static_layout=static_layout,
    )
    self._clear_dynamic_gate_runtime_cache()


def _clear_dynamic_gate_runtime_cache(self) -> None:
    self._dynamic_gate_cache_step = None
    self._dynamic_gate_centers_cache = {}
    self._dynamic_gate_posts_cache = {}
    self._dynamic_gate_velocities_cache = None
    self._dynamic_gate_obstacle_map_cache = None
    self._active_obstacle_map_cache = None


def _ensure_dynamic_gate_runtime_cache(self) -> None:
    if self._dynamic_gate_cache_step == int(self._step_count):
        return
    self._dynamic_gate_cache_step = int(self._step_count)
    self._dynamic_gate_centers_cache = {}
    self._dynamic_gate_posts_cache = {}
    self._dynamic_gate_velocities_cache = None
    self._dynamic_gate_obstacle_map_cache = None
    self._active_obstacle_map_cache = None


def _dynamic_gate_time_s(self, *, next_frame: bool = False) -> float:
    step_offset = 1 if bool(next_frame) else 0
    return float(self._step_count + step_offset) * float(self.env_config.dt_s)


def _dynamic_gate_centers_xy(self, *, next_frame: bool = False) -> np.ndarray:
    if not self._dynamic_gate_enabled or not self._dynamic_gates:
        return np.zeros((0, 2), dtype=np.float32)
    self._ensure_dynamic_gate_runtime_cache()
    cache_key = bool(next_frame)
    cached = self._dynamic_gate_centers_cache.get(cache_key)
    if cached is not None:
        return cached
    centers = live_gate_centers(
        self._dynamic_gates,
        t_sec=self._dynamic_gate_time_s(next_frame=next_frame),
        amplitude_m=float(getattr(self._dynamic_gate_config, "moving_gate_amplitude_m", 0.0) or 0.0),
        speed_mps=float(getattr(self._dynamic_gate_config, "moving_gate_speed_mps", 0.0) or 0.0),
        config=self._dynamic_gate_config,
    )
    self._dynamic_gate_centers_cache[cache_key] = centers
    return centers


def _dynamic_gate_posts_xy(self, *, next_frame: bool = False) -> np.ndarray:
    if not self._dynamic_gate_enabled or not self._dynamic_gates:
        return np.zeros((0, 2), dtype=np.float32)
    self._ensure_dynamic_gate_runtime_cache()
    cache_key = bool(next_frame)
    cached = self._dynamic_gate_posts_cache.get(cache_key)
    if cached is not None:
        return cached
    posts = gate_posts(
        self._dynamic_gates,
        self._dynamic_gate_centers_xy(next_frame=next_frame),
        config=self._dynamic_gate_config,
    )
    self._dynamic_gate_posts_cache[cache_key] = posts
    return posts


def _dynamic_gate_velocities_xy(self) -> np.ndarray:
    if not self._dynamic_gate_enabled or not self._dynamic_gates:
        return np.zeros((0, 2), dtype=np.float32)
    self._ensure_dynamic_gate_runtime_cache()
    if self._dynamic_gate_velocities_cache is not None:
        return self._dynamic_gate_velocities_cache
    centers_now = self._dynamic_gate_centers_xy(next_frame=False)
    centers_next = self._dynamic_gate_centers_xy(next_frame=True)
    velocities = (centers_next - centers_now) / max(float(self.env_config.dt_s), 1.0e-6)
    self._dynamic_gate_velocities_cache = velocities.astype(np.float32, copy=False)
    return self._dynamic_gate_velocities_cache


def _dynamic_gate_obstacle_map(self) -> GateObstacleMap2D:
    self._ensure_dynamic_gate_runtime_cache()
    if self._dynamic_gate_obstacle_map_cache is not None:
        return self._dynamic_gate_obstacle_map_cache
    posts_xy = self._dynamic_gate_posts_xy(next_frame=False)
    if posts_xy.size == 0:
        self._dynamic_gate_obstacle_map_cache = GateObstacleMap2D.empty()
        return self._dynamic_gate_obstacle_map_cache
    gate_velocities_xy = self._dynamic_gate_velocities_xy()
    post_velocities_xy = (
        np.repeat(gate_velocities_xy, 2, axis=0)
        if gate_velocities_xy.size
        else np.zeros_like(posts_xy, dtype=np.float32)
    )
    obstacles = tuple(
        GatePostObstacle2D(
            species="dynamic_gate_post",
            center_xy=(float(post_xy[0]), float(post_xy[1])),
            collision_radius_m=float(self._dynamic_gate_config.gate_post_radius_m),
            canopy_height_m=float(self._dynamic_gate_config.gate_opening_top_height_m),
            description="live dynamic gate post",
            usd_path="dynamic_gate_density_2d",
            velocity_xy=(float(velocity_xy[0]), float(velocity_xy[1])),
        )
        for post_xy, velocity_xy in zip(posts_xy, post_velocities_xy, strict=True)
    )
    self._dynamic_gate_obstacle_map_cache = GateObstacleMap2D(obstacles)
    return self._dynamic_gate_obstacle_map_cache


def _active_obstacle_map(self) -> GateObstacleMap2D:
    if not self._dynamic_gate_enabled:
        return self.obstacle_map
    self._ensure_dynamic_gate_runtime_cache()
    if self._active_obstacle_map_cache is not None:
        return self._active_obstacle_map_cache
    dynamic_map = self._dynamic_gate_obstacle_map()
    if len(self.obstacle_map) == 0:
        self._active_obstacle_map_cache = dynamic_map
    else:
        self._active_obstacle_map_cache = GateObstacleMap2D(
            tuple(self.obstacle_map.obstacles) + tuple(dynamic_map.obstacles)
        )
    return self._active_obstacle_map_cache


def _dynamic_gate_motion_range_m(self) -> float:
    if not self._dynamic_gate_enabled or not self._dynamic_gates:
        return 0.0
    centers = self._dynamic_gate_centers_xy(next_frame=False)
    bases = np.asarray([gate.base_center_xy for gate in self._dynamic_gates], dtype=np.float32)
    if centers.size == 0 or bases.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(centers - bases, axis=1)))
