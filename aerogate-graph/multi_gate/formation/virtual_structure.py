"""Formation slots and virtual-structure helpers for variable-size teams."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from multi_gate.configs.experiment_config import MultiFormationConfig
from shared.runtime.exp3_formation_demo import local_formation_offsets


@dataclass(frozen=True)
class VirtualStructureSummary:
    """Metadata describing the generated slot layout."""

    num_agents: int
    column_count: int
    row_count: int
    lateral_half_span_m: float
    trailing_length_m: float


class VirtualStructure2D:
    """Generate centered slot offsets and world-space slot positions."""

    def __init__(self, config: MultiFormationConfig | None = None) -> None:
        self.config = config or MultiFormationConfig()

    def _column_count(self, num_agents: int) -> int:
        if num_agents <= 1:
            return 1
        if num_agents <= 4:
            return 2
        if num_agents <= 12:
            return 3
        return self.config.max_columns

    def slot_offsets(
        self,
        num_agents: int,
        *,
        shape_name: str | None = None,
        slot_permutation: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Return local-frame XY slot offsets for the requested team size."""

        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        bootstrap_offsets = self._bootstrap_slot_offsets(
            num_agents,
            shape_name=shape_name,
            slot_permutation=slot_permutation,
        )
        if bootstrap_offsets is not None:
            return bootstrap_offsets
        column_count = self._column_count(num_agents)
        column_offsets = self._centered_axis_offsets(column_count, self.config.lateral_spacing_m)
        row_count = int(math.ceil(num_agents / float(column_count)))
        row_offsets = self._centered_axis_offsets(row_count, self.config.longitudinal_spacing_m)

        offsets: list[tuple[float, float]] = []
        for row_idx in range(row_count):
            for col_offset in column_offsets:
                offsets.append((row_offsets[row_idx], col_offset))
                if len(offsets) == num_agents:
                    return np.asarray(offsets, dtype=np.float32)
        return np.asarray(offsets, dtype=np.float32)

    def slot_world_positions(
        self,
        *,
        center_xy: tuple[float, float],
        heading_xy: tuple[float, float],
        num_agents: int,
    ) -> np.ndarray:
        """Project local slot offsets into world coordinates."""

        local_offsets = self.slot_offsets(num_agents)
        heading = np.asarray(heading_xy, dtype=np.float32)
        norm = float(np.linalg.norm(heading))
        if norm <= 1e-6:
            heading = np.asarray((1.0, 0.0), dtype=np.float32)
        else:
            heading = heading / norm
        lateral = np.asarray((-heading[1], heading[0]), dtype=np.float32)
        center = np.asarray(center_xy, dtype=np.float32)

        world_positions = []
        for longitudinal, lateral_offset in local_offsets:
            world = center + heading * float(longitudinal) + lateral * float(lateral_offset)
            world_positions.append(world)
        return np.asarray(world_positions, dtype=np.float32)

    def summary(self, num_agents: int) -> VirtualStructureSummary:
        """Return aggregate layout metrics used by planning and testing."""

        offsets = self.slot_offsets(num_agents)
        if offsets.size == 0:
            return VirtualStructureSummary(num_agents=num_agents, column_count=0, row_count=0, lateral_half_span_m=0.0, trailing_length_m=0.0)
        unique_rows = len({round(float(x), 4) for x in offsets[:, 0]})
        unique_cols = len({round(float(y), 4) for y in offsets[:, 1]})
        return VirtualStructureSummary(
            num_agents=num_agents,
            column_count=unique_cols,
            row_count=unique_rows,
            lateral_half_span_m=float(np.max(np.abs(offsets[:, 1])) if offsets.shape[0] else 0.0),
            trailing_length_m=float(np.max(np.abs(offsets[:, 0])) if offsets.shape[0] else 0.0),
        )

    def _bootstrap_slot_offsets(
        self,
        num_agents: int,
        *,
        shape_name: str | None = None,
        slot_permutation: tuple[int, ...] | None = None,
    ) -> np.ndarray | None:
        if not bool(getattr(self.config, "bootstrap_templates_enabled", False)):
            return None
        resolved_shape_name = str(
            shape_name if shape_name is not None else getattr(self.config, "bootstrap_shape_name", "")
        ).strip().lower()
        if resolved_shape_name in {"line", "triangle", "rectangle", "diamond", "circle"} and num_agents == 8:
            offsets = np.asarray(local_formation_offsets(resolved_shape_name), dtype=np.float32)
            permutation_source = (
                slot_permutation
                if slot_permutation is not None
                else getattr(self.config, "bootstrap_slot_permutation", ())
            )
            permutation = tuple(int(idx) for idx in (permutation_source or ()))
            if len(permutation) >= num_agents and sorted(permutation[:num_agents]) == list(range(num_agents)):
                offsets = offsets[np.asarray(permutation[:num_agents], dtype=np.int64)]
            return offsets
        lateral_spacing_m = float(
            self.config.lateral_spacing_m
            if getattr(self.config, "bootstrap_lateral_spacing_m", None) is None
            else self.config.bootstrap_lateral_spacing_m
        )
        longitudinal_spacing_m = float(
            self.config.longitudinal_spacing_m
            if getattr(self.config, "bootstrap_longitudinal_spacing_m", None) is None
            else self.config.bootstrap_longitudinal_spacing_m
        )
        if num_agents == 2:
            return np.asarray(
                (
                    (0.0, -0.5 * lateral_spacing_m),
                    (0.0, 0.5 * lateral_spacing_m),
                ),
                dtype=np.float32,
            )
        if num_agents == 3:
            layout = str(getattr(self.config, "bootstrap_three_agent_layout", "vee")).strip().lower()
            if layout in {"wide_vee", "wide-vee", "wide_v", "wide-v"}:
                return np.asarray(
                    (
                        (0.75 * longitudinal_spacing_m, 0.0),
                        (-0.65 * longitudinal_spacing_m, -0.7 * lateral_spacing_m),
                        (-0.65 * longitudinal_spacing_m, 0.7 * lateral_spacing_m),
                    ),
                    dtype=np.float32,
                )
            if layout in {"vee", "v", "triangle", "triangular"}:
                return np.asarray(
                    (
                        (0.5 * longitudinal_spacing_m, 0.0),
                        (-0.5 * longitudinal_spacing_m, -0.5 * lateral_spacing_m),
                        (-0.5 * longitudinal_spacing_m, 0.5 * lateral_spacing_m),
                    ),
                    dtype=np.float32,
                )
        return None

    @staticmethod
    def _centered_axis_offsets(count: int, spacing_m: float) -> list[float]:
        center = (count - 1) / 2.0
        return [(idx - center) * spacing_m for idx in range(count)]

