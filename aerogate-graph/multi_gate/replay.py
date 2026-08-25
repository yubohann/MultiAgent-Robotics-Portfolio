"""Replay helpers for the multi-agent 2D gate experiment."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from multi_gate.configs.experiment_config import MULTI_EXPERIMENT_CONFIG, MultiExperimentConfig
from multi_gate.dynamic_gate_task_slots import dynamic_gate_task_first_slots
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.graph_rl.graph_flashsac import GraphFlashSACAgent as GraphMASACAgent
from multi_gate.training import (
    _multi_corridor_through_success_from_info,
    _multi_episode_success_from_info,
    _select_multi_env_class,
    validate_multi_checkpoint_compatibility,
)
from shared.configs.global_config import GLOBAL_CONFIG
from shared.runtime.artifacts import allocate_replay_artifacts, default_run_name, write_json


DYNAMIC_GATE_POST_VISUAL_REPULSION_MARGIN_M = 0.76


class HeuristicFormationReplayController:
    """A simple slot follower used for deterministic multi-agent smoke replays."""

    def __init__(self, env: MultiGate2DEnv, *, compact_gate_mode: bool = False) -> None:
        self.env = env
        self.compact_gate_mode = bool(compact_gate_mode)
        self._safe_gate_phase = "align"
        self._safe_gate_offsets_xy: np.ndarray | None = None
        self._safe_gate_reset_step: int | None = None
        self._safe_gate_last_gate_idx: int | None = None
        self._compact_gate_slot_template_xy: np.ndarray | None = None

    def act(self) -> np.ndarray:
        positions = self.env.active_positions_xy()
        desired_slots = self.env.desired_slots_xy()
        heading = np.asarray(self.env.current_heading_xy(), dtype=np.float32)
        center_xy = np.asarray(self.env.snapshot().virtual_center_xy, dtype=np.float32)
        target_waypoint_xy = self._current_target_waypoint_xy()
        target_delta = target_waypoint_xy - center_xy
        target_distance = float(np.linalg.norm(target_delta))
        action = np.zeros(self.env.action_shape, dtype=np.float32)
        goal_distance = self.env.snapshot().goal_distance_m
        num_agents = positions.shape[0]
        min_pair_distance = self._minimum_pair_distance(positions)
        min_clearance = self._min_active_obstacle_clearance(positions)
        feedforward_speed = self._feedforward_speed_mps(
            goal_distance_m=goal_distance,
            target_distance_m=target_distance,
            min_clearance_m=min_clearance,
            num_agents=num_agents,
        )
        pair_warning_distance = self._pair_warning_distance_m(num_agents)
        if self.compact_gate_mode and num_agents <= 8:
            pair_warning_distance = max(pair_warning_distance, 3.45)
        pair_proximity_ratio = self._pair_proximity_ratio(
            min_pair_distance_m=min_pair_distance,
            warning_distance_m=pair_warning_distance,
        )
        feedforward_speed *= self._pair_speed_scale(
            num_agents=num_agents,
            pair_proximity_ratio=pair_proximity_ratio,
        )
        if self.compact_gate_mode and num_agents == 8:
            compact_action = self._compact_corridor_dynamic_gate_action(
                positions_xy=positions,
                center_xy=center_xy,
                min_pair_distance_m=min_pair_distance,
                min_clearance_m=min_clearance,
            )
            if compact_action is not None:
                return compact_action
        slot_gain = self._slot_gain(num_agents)
        repulsion_gain = self._repulsion_gain(num_agents)
        pair_repulsion_gain = self._pair_repulsion_gain(
            num_agents=num_agents,
            goal_distance_m=goal_distance,
            pair_proximity_ratio=pair_proximity_ratio,
        )
        if not self.compact_gate_mode:
            safe_gate_action = self._safe_dynamic_gate_action(
                positions_xy=positions,
                center_xy=center_xy,
                min_pair_distance_m=min_pair_distance,
                num_agents=num_agents,
            )
            if safe_gate_action is not None:
                return safe_gate_action
        forward_slot_gain_scale = 1.0
        task_slots, gate_task_ratio = self._dynamic_gate_task_first_slots(
            positions_xy=positions,
            desired_slots_xy=desired_slots,
            center_xy=center_xy,
            feedforward_speed_mps=feedforward_speed,
        )
        if task_slots is not None:
            desired_slots = task_slots
            heading = np.asarray((1.0, 0.0), dtype=np.float32)
            if self.compact_gate_mode:
                desired_slots = self._compact_dynamic_gate_slots(desired_slots, gate_task_ratio)
                feedforward_speed *= float(1.0 - 0.74 * gate_task_ratio)
                feedforward_speed = float(np.clip(feedforward_speed, 0.20, 0.46))
                slot_gain = max(slot_gain, float(4.20 + 2.60 * gate_task_ratio))
                forward_slot_gain_scale = 1.0
                repulsion_gain = max(repulsion_gain, float(2.55 + 1.85 * gate_task_ratio))
                pair_repulsion_gain = max(pair_repulsion_gain, float(6.20 + 2.60 * gate_task_ratio))
            else:
                feedforward_speed *= float(1.0 - 0.58 * gate_task_ratio)
                feedforward_speed = max(feedforward_speed, float(0.46 + 0.18 * (1.0 - gate_task_ratio)))
                slot_gain = max(slot_gain, float(2.15 + 2.45 * gate_task_ratio))
                forward_slot_gain_scale = float(max(0.18, 1.0 - 0.86 * gate_task_ratio))
                repulsion_gain = max(repulsion_gain, float(1.90 + 1.40 * gate_task_ratio))
                pair_repulsion_gain = max(pair_repulsion_gain, float(2.35 + 2.55 * gate_task_ratio))
        if num_agents > 16 and goal_distance < 6.0:
            feedforward_speed *= 0.88
            slot_gain *= 1.15
        if num_agents > 16 and goal_distance < 3.0:
            feedforward_speed *= 0.82
            slot_gain *= 1.22
            pair_repulsion_gain = max(pair_repulsion_gain, 0.10)
        for agent_idx in range(positions.shape[0]):
            slot_error = desired_slots[agent_idx] - positions[agent_idx]
            controlled_slot_error = slot_error.astype(np.float32, copy=True)
            controlled_slot_error[0] *= float(forward_slot_gain_scale)
            repulsion_velocity = repulsion_gain * self._obstacle_repulsion(positions[agent_idx], num_agents)
            desired_velocity = heading * feedforward_speed + slot_gain * controlled_slot_error + repulsion_velocity
            if pair_repulsion_gain > 0.0:
                desired_velocity += pair_repulsion_gain * self._pair_repulsion(
                    positions_xy=positions,
                    agent_idx=agent_idx,
                    trigger_distance_m=pair_warning_distance,
                )
            action[agent_idx] = self.env.desired_velocity_to_action(desired_velocity)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _compact_corridor_dynamic_gate_action(
        self,
        *,
        positions_xy: np.ndarray,
        center_xy: np.ndarray,
        min_pair_distance_m: float,
        min_clearance_m: float,
    ) -> np.ndarray | None:
        if not bool(getattr(self.env, "_dynamic_gate_enabled", False)):
            return None
        gates = list(getattr(self.env, "_dynamic_gates", []) or [])
        if not gates or not hasattr(self.env, "_dynamic_gate_centers_xy"):
            return None

        centers_xy = np.asarray(self.env._dynamic_gate_centers_xy(next_frame=False), dtype=np.float32)
        if centers_xy.size == 0:
            return None
        formation_x_span_m = float(np.max(positions_xy[:8, 0]) - np.min(positions_xy[:8, 0]))
        center_lane_indices = [
            gate_idx
            for gate_idx, gate in enumerate(gates)
            if int(getattr(gate, "lane_index", 0)) == 0 and gate_idx < len(centers_xy)
        ]
        target_center_y = 0.0
        task_ratio = 0.0
        nearest_center_gate: tuple[int, float] | None = None
        nearest_center_gate_key: tuple[int, float] | None = None
        active_center_gate_dx_m: float | None = None
        if center_lane_indices:
            weighted_y = 0.0
            weight_sum = 0.0
            center_x = float(center_xy[0])
            rear_x = float(np.min(positions_xy[:8, 0]))
            front_x = float(np.max(positions_xy[:8, 0]))
            for gate_idx in center_lane_indices:
                gate_center = centers_xy[gate_idx]
                dx_m = float(gate_center[0]) - center_x
                gate_x_m = float(gate_center[0])
                formation_is_crossing_gate = (rear_x - 1.0) <= gate_x_m <= (front_x + 1.2)
                if formation_is_crossing_gate:
                    candidate_key = (-1, abs(dx_m))
                elif -4.0 <= dx_m <= 12.0:
                    candidate_key = (0, max(dx_m, 0.0)) if dx_m >= -0.40 else (1, abs(dx_m))
                else:
                    candidate_key = None
                if candidate_key is not None and (
                    nearest_center_gate_key is None or candidate_key < nearest_center_gate_key
                ):
                    nearest_center_gate = (int(gate_idx), float(dx_m))
                    nearest_center_gate_key = candidate_key
                if -7.0 <= dx_m <= 24.0:
                    local = float(np.exp(-abs(dx_m) / 7.5))
                    if -4.0 <= dx_m <= 8.0:
                        local = max(local, 0.95)
                    elif 8.0 < dx_m <= 16.0:
                        local = max(local, 0.62)
                    elif 16.0 < dx_m <= 24.0:
                        local = max(local, 0.34)
                    weight = local / (1.0 + max(dx_m, 0.0) * 0.08)
                    weighted_y += float(gate_center[1]) * weight
                    weight_sum += weight
                    task_ratio = max(task_ratio, float(np.clip(local, 0.0, 1.0)))
            if weight_sum > 1.0e-6:
                target_center_y = float(np.clip(weighted_y / weight_sum, -1.25, 1.25))

        ratio = float(np.clip(task_ratio, 0.0, 1.0))
        lane_y = float((1.0 - ratio) * 1.23 + ratio * 0.58)
        x_spacing = float((1.0 - ratio) * 1.98 + ratio * 1.74)
        goal_distance = float(
            np.linalg.norm(
                np.asarray((float(self.env.env_config.goal_x_m), 0.0), dtype=np.float32) - center_xy
            )
        )
        if goal_distance < 14.0:
            lane_y = max(lane_y, 0.62)
            x_spacing = min(x_spacing, 1.55)
        if nearest_center_gate is not None:
            gate_idx, gate_dx_m = nearest_center_gate
            active_center_gate_dx_m = float(gate_dx_m)
            gate_center_y = float(centers_xy[gate_idx, 1])
            gate_velocity_y = 0.0
            if hasattr(self.env, "_dynamic_gate_velocities_xy"):
                gate_velocities_xy = np.asarray(self.env._dynamic_gate_velocities_xy(), dtype=np.float32)
                if gate_idx < len(gate_velocities_xy):
                    gate_velocity_y = float(gate_velocities_xy[gate_idx, 1])
            lead_time_s = float(np.clip(max(gate_dx_m, 0.0) / 1.15, 0.0, 0.95))
            predicted_gate_y = gate_center_y + gate_velocity_y * lead_time_s
            gate_cfg = getattr(self.env, "_dynamic_gate_config", None)
            gate_half_width_m = float(getattr(gate_cfg, "gate_half_width_m", 2.40))
            gate_post_radius_m = float(getattr(gate_cfg, "gate_post_radius_m", 0.62))
            drone_radius_m = float(getattr(gate_cfg, "drone_radius_m", self.env.env_config.drone_radius_m))
            center_room_m = gate_half_width_m - lane_y - gate_post_radius_m - drone_radius_m - 0.36
            center_room_m = float(np.clip(center_room_m, 0.0, 0.42))
            predicted_gate_y = float(
                np.clip(predicted_gate_y, gate_center_y - center_room_m, gate_center_y + center_room_m)
            )
            gate_blend = 0.96 if -1.5 <= gate_dx_m <= 8.0 else 0.72
            target_center_y = float((1.0 - gate_blend) * target_center_y + gate_blend * predicted_gate_y)
            target_center_y = float(np.clip(target_center_y, -1.25, 1.25))
        if self._compact_gate_slot_template_xy is None or self._compact_gate_slot_template_xy.shape != (8, 2):
            self._compact_gate_slot_template_xy = self._initial_compact_gate_slot_template(positions_xy[:8])
        template_xy = self._compact_gate_slot_template_xy
        offsets = np.zeros((8, 2), dtype=np.float32)
        offsets[:, 0] = template_xy[:, 0] * float(x_spacing)
        offsets[:, 1] = template_xy[:, 1] * float(lane_y)
        lookahead_x = float((1.0 - ratio) * 1.35 + ratio * 0.96)
        lateral_center_error = abs(float(target_center_y) - float(center_xy[1]))
        if active_center_gate_dx_m is not None and -1.0 <= active_center_gate_dx_m <= 8.0:
            if lateral_center_error > 0.55:
                lookahead_x = min(lookahead_x, 0.12)
            elif lateral_center_error > 0.30:
                lookahead_x = min(lookahead_x, 0.32)
            elif lateral_center_error > 0.18:
                lookahead_x = min(lookahead_x, 0.55)
        if formation_x_span_m > 5.2:
            lookahead_x = min(lookahead_x, 0.65)
        elif formation_x_span_m > 4.4:
            lookahead_x = min(lookahead_x, 0.82)
        target_center = np.asarray(
            (
                min(float(self.env.env_config.goal_x_m), float(center_xy[0]) + lookahead_x),
                target_center_y,
            ),
            dtype=np.float32,
        )
        targets_xy = target_center.reshape(1, 2) + offsets
        forward_speed = float((1.0 - ratio) * 1.38 + ratio * 0.98)
        if active_center_gate_dx_m is not None and -1.0 <= active_center_gate_dx_m <= 8.0:
            if lateral_center_error > 0.55:
                forward_speed *= 0.18
            elif lateral_center_error > 0.30:
                forward_speed *= 0.34
            elif lateral_center_error > 0.18:
                forward_speed *= 0.58
        else:
            if lateral_center_error > 0.65:
                forward_speed *= 0.60
            elif lateral_center_error > 0.40:
                forward_speed *= 0.74
        if formation_x_span_m > 5.2:
            forward_speed *= 0.74
        elif formation_x_span_m > 4.4:
            forward_speed *= 0.86
        if min_clearance_m < 1.65:
            forward_speed *= 0.58
        elif min_clearance_m < 2.65:
            forward_speed *= 0.76
        if min_pair_distance_m < 1.45:
            forward_speed *= 0.74
        slot_gain = float((1.0 - ratio) * 2.80 + ratio * 4.50)
        max_forward = float(self.env.env_config.max_command_forward_speed_mps or self.env.env_config.max_command_speed_mps)
        max_lateral = float(self.env.env_config.max_command_lateral_speed_mps or self.env.env_config.max_command_speed_mps)

        action = np.zeros(self.env.action_shape, dtype=np.float32)
        for agent_idx in range(8):
            slot_error = targets_xy[agent_idx] - positions_xy[agent_idx]
            desired_velocity = np.asarray(
                (
                    float(np.clip(forward_speed + slot_gain * slot_error[0], -1.50, min(max_forward, 1.42))),
                    float(np.clip(slot_gain * slot_error[1], -min(max_lateral, 2.35), min(max_lateral, 2.35))),
                ),
                dtype=np.float32,
            )
            obstacle_push = 4.35 * self._obstacle_repulsion(positions_xy[agent_idx], 8)
            pair_push = 7.20 * self._pair_repulsion(
                positions_xy=positions_xy,
                agent_idx=agent_idx,
                trigger_distance_m=2.80,
            )
            desired_velocity += self._clip_vector(obstacle_push, max_norm=2.00)
            desired_velocity += self._clip_vector(pair_push, max_norm=2.60)
            action[agent_idx] = self.env.desired_velocity_to_action(desired_velocity)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _compact_dynamic_gate_slots(slots_xy: np.ndarray, task_ratio: float) -> np.ndarray:
        slots = np.asarray(slots_xy, dtype=np.float32).copy()
        if slots.ndim != 2 or slots.shape[0] <= 1 or slots.shape[1] != 2:
            return slots
        ratio = float(np.clip(task_ratio, 0.0, 1.0))
        center = np.mean(slots, axis=0, keepdims=True)
        offsets = slots - center
        lateral_scale = float(1.0 - 0.34 * ratio)
        longitudinal_scale = float(1.0 - 0.10 * ratio)
        offsets[:, 1] *= max(lateral_scale, 0.64)
        offsets[:, 0] *= max(longitudinal_scale, 0.86)
        return (center + offsets).astype(np.float32, copy=False)

    @staticmethod
    def _initial_compact_gate_slot_template(positions_xy: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xy, dtype=np.float32)
        template_rows = (
            np.asarray(((-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)), dtype=np.float32),
            np.asarray(((-0.5, 0.0), (0.5, 0.0)), dtype=np.float32),
            np.asarray(((-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)), dtype=np.float32),
        )
        template = np.zeros((8, 2), dtype=np.float32)
        if positions.shape[0] < 8:
            flat = np.concatenate(template_rows, axis=0)
            template[: positions.shape[0]] = flat[: positions.shape[0]]
            return template
        lateral_order = np.argsort(positions[:8, 1])
        start = 0
        for row_idx, row_size in enumerate((3, 2, 3)):
            group = lateral_order[start : start + row_size]
            start += row_size
            longitudinal_order = sorted((int(agent_idx) for agent_idx in group), key=lambda idx: float(positions[idx, 0]))
            for slot_idx, agent_idx in enumerate(longitudinal_order):
                template[agent_idx] = template_rows[row_idx][slot_idx]
        return template

    def _safe_dynamic_gate_action(
        self,
        *,
        positions_xy: np.ndarray,
        center_xy: np.ndarray,
        min_pair_distance_m: float,
        num_agents: int,
    ) -> np.ndarray | None:
        if num_agents <= 1 or num_agents > 9:
            return None
        if not bool(getattr(self.env, "_dynamic_gate_enabled", False)):
            return None
        gates = list(getattr(self.env, "_dynamic_gates", []) or [])
        if not gates or not hasattr(self.env, "_dynamic_gate_centers_xy"):
            return None
        centers_xy = np.asarray(self.env._dynamic_gate_centers_xy(next_frame=False), dtype=np.float32)
        if centers_xy.size == 0:
            return None
        lane_gate_indices = [
            gate_idx for gate_idx, gate in enumerate(gates)
            if int(getattr(gate, "lane_index", 0)) == 0 and gate_idx < len(centers_xy)
        ]
        if not lane_gate_indices:
            return None
        gate_idx = min(lane_gate_indices, key=lambda idx: abs(float(centers_xy[idx, 0]) - float(center_xy[0])))
        if gate_idx != self._safe_gate_last_gate_idx:
            self._safe_gate_offsets_xy = None
            self._safe_gate_phase = "align"
            self._safe_gate_last_gate_idx = gate_idx
        gate_center_xy = np.asarray(centers_xy[gate_idx], dtype=np.float32)
        gate_x_m = float(gate_center_xy[0])
        gate_y_m = float(gate_center_xy[1])
        if float(center_xy[0]) > gate_x_m + 7.5:
            self._safe_gate_phase = "goal"
            return None

        step_count = int(getattr(self.env, "_step_count", 0) or 0)
        if self._safe_gate_offsets_xy is None or self._safe_gate_offsets_xy.shape[0] != num_agents or step_count <= 0:
            self._safe_gate_offsets_xy = self._initial_safe_gate_offsets(positions_xy[:num_agents])
            self._safe_gate_phase = "align"
            self._safe_gate_reset_step = step_count

        offsets_xy = self._safe_gate_offsets_xy[:num_agents]
        align_center_x_m = gate_x_m - 6.4
        pass_center_x_m = gate_x_m + 8.2
        align_targets_xy = offsets_xy + np.asarray((align_center_x_m, gate_y_m), dtype=np.float32).reshape(1, 2)
        lateral_error = float(np.max(np.abs(positions_xy[:num_agents, 1] - align_targets_xy[:, 1])))
        if (
            self._safe_gate_phase == "align"
            and float(center_xy[0]) >= gate_x_m - 8.6
            and lateral_error <= 1.25
            and min_pair_distance_m >= 1.45
        ):
            self._safe_gate_phase = "pass"
        if self._safe_gate_phase == "pass" and float(np.min(positions_xy[:num_agents, 0])) >= gate_x_m + 5.8:
            self._safe_gate_phase = "goal"
            return None

        target_center_x_m = pass_center_x_m if self._safe_gate_phase == "pass" else align_center_x_m
        targets_xy = offsets_xy + np.asarray((target_center_x_m, gate_y_m), dtype=np.float32).reshape(1, 2)
        action = np.zeros(self.env.action_shape, dtype=np.float32)
        max_forward_mps = float(self.env.env_config.max_command_forward_speed_mps or self.env.env_config.max_command_speed_mps)
        max_lateral_mps = float(self.env.env_config.max_command_lateral_speed_mps or self.env.env_config.max_command_speed_mps)
        for agent_idx in range(num_agents):
            error_xy = targets_xy[agent_idx] - positions_xy[agent_idx]
            if self._safe_gate_phase == "pass":
                desired_velocity = np.asarray(
                    (
                        float(np.clip(0.48 * error_xy[0], 0.48, min(max_forward_mps, 0.86))),
                        float(np.clip(1.45 * error_xy[1], -min(max_lateral_mps, 1.00), min(max_lateral_mps, 1.00))),
                    ),
                    dtype=np.float32,
                )
            else:
                forward_ceiling = 0.54 if abs(float(error_xy[1])) <= 0.50 else 0.32
                desired_velocity = np.asarray(
                    (
                        float(np.clip(0.42 * error_xy[0], -0.12, min(max_forward_mps, forward_ceiling))),
                        float(np.clip(1.55 * error_xy[1], -min(max_lateral_mps, 1.00), min(max_lateral_mps, 1.00))),
                    ),
                    dtype=np.float32,
                )
            pair_push = 1.70 * self._pair_repulsion(
                positions_xy=positions_xy,
                agent_idx=agent_idx,
                trigger_distance_m=max(self._pair_warning_distance_m(num_agents), 2.15),
            )
            obstacle_push = 0.95 * self._obstacle_repulsion(positions_xy[agent_idx], num_agents)
            desired_velocity += self._clip_vector(pair_push, max_norm=0.42)
            desired_velocity += self._clip_vector(obstacle_push, max_norm=0.65)
            action[agent_idx] = self.env.desired_velocity_to_action(desired_velocity)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _initial_safe_gate_offsets(positions_xy: np.ndarray) -> np.ndarray:
        num_agents = int(positions_xy.shape[0])
        base_lane_size = num_agents // 3
        remainder = num_agents % 3
        if remainder == 0:
            lane_sizes = [base_lane_size, base_lane_size, base_lane_size]
        elif remainder == 1:
            lane_sizes = [base_lane_size, base_lane_size + 1, base_lane_size]
        else:
            lane_sizes = [base_lane_size + 1, base_lane_size, base_lane_size + 1]
        lane_y = np.asarray((-0.62, 0.0, 0.62), dtype=np.float32)
        offsets = np.zeros((num_agents, 2), dtype=np.float32)
        lateral_order = np.argsort(positions_xy[:, 1])
        start = 0
        for lane_idx, lane_size in enumerate(lane_sizes):
            lane_group = lateral_order[start : min(start + int(lane_size), num_agents)]
            start += int(lane_size)
            if len(lane_group) == 0:
                continue
            longitudinal_order = sorted((int(agent_idx) for agent_idx in lane_group), key=lambda idx: float(positions_xy[idx, 0]))
            x_offsets = (
                (np.arange(len(longitudinal_order), dtype=np.float32) - 0.5 * float(len(longitudinal_order) - 1))
                * 2.20
            )
            x_offsets = np.clip(x_offsets, -2.40, 2.40)
            for offset_idx, agent_idx in enumerate(longitudinal_order):
                offsets[agent_idx, 0] = x_offsets[offset_idx]
                offsets[agent_idx, 1] = lane_y[lane_idx]
        return offsets

    @staticmethod
    def _clip_vector(vector_xy: np.ndarray, *, max_norm: float) -> np.ndarray:
        vector = np.asarray(vector_xy, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm <= 1.0e-6:
            return vector
        return vector / norm * float(max_norm)

    def _current_target_waypoint_xy(self) -> np.ndarray:
        waypoints = self.env.path_waypoints()
        path_index = min(self.env.snapshot().path_index, len(waypoints) - 1)
        return np.asarray(waypoints[path_index], dtype=np.float32)

    def _feedforward_speed_mps(
        self,
        *,
        goal_distance_m: float,
        target_distance_m: float,
        min_clearance_m: float,
        num_agents: int,
    ) -> float:
        max_speed = self.env.env_config.max_command_speed_mps
        size_ratio = 0.74 if num_agents <= 4 else 0.79 if num_agents <= 16 else 0.72
        desired_speed = min(
            max_speed * size_ratio,
            1.8 + 0.045 * goal_distance_m + 0.22 * min(target_distance_m, 8.0),
        )
        if goal_distance_m <= 10.0:
            desired_speed = min(desired_speed, 1.2 + 0.35 * goal_distance_m)
        if (
            8 < num_agents <= 16
            and self.env.snapshot().path_index >= len(self.env.path_waypoints()) - 3
            and min_clearance_m >= 2.0
        ):
            desired_speed = min(max_speed * 0.98, desired_speed * 1.35)
        if min_clearance_m < 0.75:
            desired_speed *= 0.58
        elif min_clearance_m < 1.5:
            desired_speed *= 0.76
        elif min_clearance_m < 2.5:
            desired_speed *= 0.9
        if getattr(self.env, "_step_count", 0) < 10 and num_agents >= 24:
            desired_speed *= 0.7
        return max(0.8, float(desired_speed))

    @staticmethod
    def _minimum_pair_distance(positions_xy: np.ndarray) -> float:
        if positions_xy.shape[0] <= 1:
            return float("inf")
        min_distance = float("inf")
        for anchor_idx in range(positions_xy.shape[0]):
            for other_idx in range(anchor_idx + 1, positions_xy.shape[0]):
                distance = float(np.linalg.norm(positions_xy[anchor_idx] - positions_xy[other_idx]))
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def _pair_warning_distance_m(self, num_agents: int) -> float:
        base_distance = float(self.env.env_config.inter_agent_safe_distance_m)
        if num_agents <= 3:
            return base_distance * 1.90
        if num_agents <= 8:
            return max(base_distance * 2.80, 2.10)
        return base_distance * 1.45

    @staticmethod
    def _pair_proximity_ratio(
        *,
        min_pair_distance_m: float,
        warning_distance_m: float,
    ) -> float:
        if not np.isfinite(min_pair_distance_m):
            return 0.0
        safe_distance = max(warning_distance_m, 1.0e-6)
        return float(np.clip((safe_distance - float(min_pair_distance_m)) / safe_distance, 0.0, 1.0))

    @staticmethod
    def _pair_speed_scale(
        *,
        num_agents: int,
        pair_proximity_ratio: float,
    ) -> float:
        if pair_proximity_ratio <= 0.0:
            return 1.0
        if num_agents <= 3:
            return float(np.clip(1.0 - 0.75 * pair_proximity_ratio, 0.28, 1.0))
        if num_agents <= 8:
            return float(np.clip(1.0 - 0.55 * pair_proximity_ratio, 0.40, 1.0))
        return float(np.clip(1.0 - 0.35 * pair_proximity_ratio, 0.55, 1.0))

    @staticmethod
    def _slot_gain(num_agents: int) -> float:
        if num_agents <= 4:
            return 1.0
        if num_agents <= 16:
            return 0.82
        return 0.5

    @staticmethod
    def _repulsion_gain(num_agents: int) -> float:
        if num_agents <= 4:
            return 1.0
        if num_agents <= 16:
            return 1.15
        return 2.8

    @staticmethod
    def _pair_repulsion_gain(
        *,
        num_agents: int,
        goal_distance_m: float,
        pair_proximity_ratio: float,
    ) -> float:
        if num_agents <= 3:
            base_gain = 0.42
            if goal_distance_m <= 12.0:
                base_gain += 0.06
            return float(base_gain + 0.42 * pair_proximity_ratio)
        if num_agents <= 8:
            return float(0.28 + 0.28 * pair_proximity_ratio)
        if num_agents <= 16:
            return float(0.08 + 0.08 * pair_proximity_ratio)
        return float(0.0 + 0.10 * pair_proximity_ratio)

    def _obstacle_repulsion(self, position_xy: np.ndarray, num_agents: int) -> np.ndarray:
        trigger_distance = 3.4 if num_agents <= 16 else 6.0
        query_radius = trigger_distance + 2.0
        repulsion = np.zeros((2,), dtype=np.float32)
        for obstacle in self._active_obstacle_map().query_local(
            (float(position_xy[0]), float(position_xy[1])),
            radius_m=query_radius,
        ):
            obstacle_center = np.asarray(obstacle.center_xy, dtype=np.float32)
            obstacle_species = str(getattr(obstacle, "species", ""))
            delta = position_xy - obstacle_center
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-6:
                delta = np.asarray((1.0, 0.0), dtype=np.float32)
                distance = 1.0
            signed_clearance = (
                distance
                - float(obstacle.collision_radius_m)
                - self.env.env_config.drone_radius_m
            )
            effective_trigger_distance = trigger_distance
            if obstacle_species == "dynamic_gate_post" and num_agents <= 8:
                signed_clearance -= DYNAMIC_GATE_POST_VISUAL_REPULSION_MARGIN_M
                effective_trigger_distance = max(effective_trigger_distance, 8.0)
            if signed_clearance >= effective_trigger_distance:
                continue
            normalized_deficit = (effective_trigger_distance - signed_clearance) / max(effective_trigger_distance, 1e-6)
            weight = normalized_deficit if num_agents <= 16 else normalized_deficit ** 1.8
            direction = delta / distance
            if obstacle_species == "dynamic_gate_post":
                if num_agents <= 8:
                    weight *= 8.05
                    if signed_clearance < 3.25:
                        emergency_clearance_deficit = float((3.25 - signed_clearance) / 3.25)
                        weight += float(6.55 * emergency_clearance_deficit ** 0.55)
                heading = np.asarray(self.env.current_heading_xy(), dtype=np.float32)
                heading_norm = float(np.linalg.norm(heading))
                if heading_norm <= 1.0e-6:
                    heading = np.asarray((1.0, 0.0), dtype=np.float32)
                else:
                    heading = heading / heading_norm
                lateral = np.asarray((-float(heading[1]), float(heading[0])), dtype=np.float32)
                lateral_component = float(np.dot(direction, lateral))
                if abs(lateral_component) <= 1.0e-4:
                    continue
                direction = lateral * lateral_component
            repulsion += direction * float(weight)
        return repulsion

    def _active_obstacle_map(self):
        if hasattr(self.env, "_active_obstacle_map"):
            return self.env._active_obstacle_map()
        return self.env.obstacle_map

    def _min_active_obstacle_clearance(self, positions_xy: np.ndarray) -> float:
        if positions_xy.size == 0:
            return float("inf")
        if hasattr(self.env, "_min_clearance"):
            return float(self.env._min_clearance(positions_xy))
        return min(
            self.env.obstacle_map.min_signed_distance(
                (float(position[0]), float(position[1])),
                drone_radius_m=self.env.env_config.drone_radius_m,
            )
            for position in positions_xy
        )

    def _dynamic_gate_task_first_slots(
        self,
        *,
        positions_xy: np.ndarray,
        desired_slots_xy: np.ndarray,
        center_xy: np.ndarray,
        feedforward_speed_mps: float,
    ) -> tuple[np.ndarray | None, float]:
        if positions_xy.shape[0] <= 1:
            return None, 0.0
        if not bool(getattr(self.env, "_dynamic_gate_enabled", False)):
            return None, 0.0
        gates = list(getattr(self.env, "_dynamic_gates", []) or [])
        if not gates or not hasattr(self.env, "_dynamic_gate_centers_xy"):
            return None, 0.0

        centers_xy = self.env._dynamic_gate_centers_xy(next_frame=False)
        if centers_xy.size == 0:
            return None, 0.0
        gate_velocities_xy = (
            self.env._dynamic_gate_velocities_xy()
            if hasattr(self.env, "_dynamic_gate_velocities_xy")
            else np.zeros_like(centers_xy, dtype=np.float32)
        )
        gate_cfg = getattr(self.env, "_dynamic_gate_config", None)
        gate_time_s = (
            float(self.env._dynamic_gate_time_s(next_frame=False))
            if hasattr(self.env, "_dynamic_gate_time_s")
            else 0.0
        )
        return dynamic_gate_task_first_slots(
            desired_slots_xy=desired_slots_xy[: positions_xy.shape[0]],
            positions_xy=positions_xy,
            center_xy=center_xy,
            gates=gates,
            centers_xy=centers_xy,
            gate_velocities_xy=gate_velocities_xy,
            gate_cfg=gate_cfg,
            gate_time_s=gate_time_s,
            feedforward_speed_mps=feedforward_speed_mps,
            start_x_m=float(self.env.env_config.start_x_m),
            goal_x_m=float(self.env.env_config.goal_x_m),
            drone_radius_m=float(getattr(self.env.env_config, "drone_radius_m", 0.35)),
        )

    def _pair_repulsion(
        self,
        *,
        positions_xy: np.ndarray,
        agent_idx: int,
        trigger_distance_m: float,
    ) -> np.ndarray:
        repulsion = np.zeros((2,), dtype=np.float32)
        anchor_xy = positions_xy[agent_idx]
        for other_idx, other_xy in enumerate(positions_xy):
            if other_idx == agent_idx:
                continue
            delta = anchor_xy - other_xy
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-6 or distance >= trigger_distance_m:
                continue
            normalized_deficit = float((trigger_distance_m - distance) / max(trigger_distance_m, 1.0e-6))
            repulsion += (delta / distance) * float(normalized_deficit ** 1.35)
            if distance < 1.55:
                emergency_deficit = float((1.55 - distance) / 1.55)
                repulsion += (delta / distance) * float(1.15 * emergency_deficit ** 0.65)
        return repulsion

    def _compact_gate_pair_repulsion(
        self,
        *,
        positions_xy: np.ndarray,
        agent_idx: int,
        template_xy: np.ndarray,
        trigger_distance_m: float,
    ) -> np.ndarray:
        repulsion = np.zeros((2,), dtype=np.float32)
        anchor_xy = positions_xy[agent_idx]
        row_anchor = float(template_xy[agent_idx, 1]) if agent_idx < len(template_xy) else 0.0
        for other_idx, other_xy in enumerate(positions_xy[:8]):
            if other_idx == agent_idx:
                continue
            delta = anchor_xy - other_xy
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-6 or distance >= trigger_distance_m:
                continue
            direction = delta / distance
            row_other = float(template_xy[other_idx, 1]) if other_idx < len(template_xy) else 0.0
            if (
                abs(row_anchor - row_other) > 0.5
                and abs(float(delta[0])) < 0.55
                and abs(float(delta[1])) < 1.25
            ):
                x_sign = -1.0 if row_anchor < row_other else 1.0
                direction = np.asarray((x_sign, 0.0), dtype=np.float32)
            normalized_deficit = float((trigger_distance_m - distance) / max(trigger_distance_m, 1.0e-6))
            repulsion += direction * float(normalized_deficit ** 1.35)
            if distance < 1.55:
                emergency_deficit = float((1.55 - distance) / 1.55)
                repulsion += direction * float(1.15 * emergency_deficit ** 0.65)
        return repulsion


def _replay_success_from_done_reason(
    done_reason: str | None,
    *,
    timeout_counts_as_success: bool,
    info: dict[str, object] | None = None,
) -> bool:
    if info is not None:
        payload = dict(info)
        payload["done_reason"] = done_reason
        return _multi_episode_success_from_info(
            payload,
            timeout_counts_as_success=timeout_counts_as_success,
        )
    if done_reason == "goal_reached":
        return True
    return bool(timeout_counts_as_success and done_reason == "timeout")


def _dynamic_gate_replay_metadata(env: MultiGate2DEnv) -> dict[str, object] | None:
    """Serialize the dynamic-gate scene authority used by this replay rollout."""

    if not bool(getattr(env, "_dynamic_gate_enabled", False)):
        return None
    gates = list(getattr(env, "_dynamic_gates", []) or [])
    gate_cfg = getattr(env, "_dynamic_gate_config", None)
    if gate_cfg is None and not gates:
        return None

    def _to_plain_dict(value: object) -> dict[str, object]:
        if hasattr(value, "to_dict"):
            raw = value.to_dict()  # type: ignore[attr-defined]
        elif is_dataclass(value):
            raw = asdict(value)
        else:
            raw = {
                key: getattr(value, key)
                for key in dir(value)
                if not key.startswith("_") and not callable(getattr(value, key))
            }
        return {str(key): _jsonable_gate_value(item) for key, item in dict(raw).items()}

    project_root = Path(__file__).resolve().parents[1]
    metadata: dict[str, object] = {
        "enabled": True,
        "gate_asset_path": str(project_root / "assets" / "gate" / "gate.usd"),
        "gate_count": int(len(gates)),
        "gates": [_to_plain_dict(gate) for gate in gates],
        "gate_yaws_rad": [float(getattr(gate, "yaw_rad", 0.0)) for gate in gates],
        "gate_base_centers_xy": [
            [
                float(getattr(gate, "base_center_xy", (0.0, 0.0))[0]),
                float(getattr(gate, "base_center_xy", (0.0, 0.0))[1]),
            ]
            for gate in gates
        ],
    }
    if gate_cfg is not None:
        cfg_dict = _to_plain_dict(gate_cfg)
        metadata["config"] = cfg_dict
        metadata["moving_gate_speed_mps"] = float(cfg_dict.get("moving_gate_speed_mps") or 0.0)
        metadata["moving_gate_amplitude_m"] = float(cfg_dict.get("moving_gate_amplitude_m") or 0.0)
        metadata["gate_half_width_m"] = float(cfg_dict.get("gate_half_width_m") or 0.0)
        metadata["gate_post_radius_m"] = float(cfg_dict.get("gate_post_radius_m") or 0.0)
        metadata["gate_bottom_height_m"] = float(cfg_dict.get("gate_opening_bottom_height_m") or 0.0)
        metadata["gate_top_height_m"] = float(cfg_dict.get("gate_opening_top_height_m") or 0.0)
        metadata["gate_center_height_m"] = float(cfg_dict.get("gate_center_height_m") or env.env_config.fixed_height_m)
        metadata["gate_native_visual_height_m"] = 2.1335996309757235
        gate_half_width_m = float(cfg_dict.get("gate_half_width_m") or 2.0)
        metadata["gate_native_visual_half_width_m"] = 2.0
        metadata["gate_visual_scale_xyz"] = [
            float(gate_half_width_m / 2.0),
            float(gate_half_width_m / 2.0),
            float(
                (
                    float(cfg_dict.get("gate_opening_top_height_m") or 8.0)
                    - float(cfg_dict.get("gate_opening_bottom_height_m") or 0.0)
                )
                / 2.1335996309757235
            ),
        ]
        metadata["height_contract"] = (
            "multi-agent replay fixed_height_m equals gate_center_height_m; "
            "gate USD z-scale visualizes the unified gate top height"
        )
    return metadata


def _jsonable_gate_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_gate_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_gate_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dynamic_gate_frame_payload(env: MultiGate2DEnv) -> dict[str, object]:
    """Record live gate poses visible to the policy at the current replay frame."""

    if not bool(getattr(env, "_dynamic_gate_enabled", False)):
        return {}
    gates = list(getattr(env, "_dynamic_gates", []) or [])
    if not gates or not hasattr(env, "_dynamic_gate_centers_xy"):
        return {}
    centers_xy = np.asarray(env._dynamic_gate_centers_xy(next_frame=False), dtype=np.float32)
    velocities_xy = (
        np.asarray(env._dynamic_gate_velocities_xy(), dtype=np.float32)
        if hasattr(env, "_dynamic_gate_velocities_xy")
        else np.zeros_like(centers_xy, dtype=np.float32)
    )
    posts_xy = (
        np.asarray(env._dynamic_gate_posts_xy(next_frame=False), dtype=np.float32)
        if hasattr(env, "_dynamic_gate_posts_xy")
        else np.zeros((0, 2), dtype=np.float32)
    )
    return {
        "dynamic_gate_enabled": True,
        "dynamic_gate_count": int(len(gates)),
        "live_gate_centers_xy": centers_xy.tolist(),
        "live_gate_velocities_xy": velocities_xy.tolist(),
        "live_gate_post_positions_xy": posts_xy.tolist(),
        "gate_yaws_rad": [float(getattr(gate, "yaw_rad", 0.0)) for gate in gates],
    }


def run_multi_replay(
    *,
    mode: str = "heuristic",
    num_agents: int | None = None,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    max_steps: int | None = None,
    output_dir: str | Path | None = None,
    experiment_config: MultiExperimentConfig | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run one multi-agent replay and save a replay report."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    active_num_agents = selected_config.default_agents if num_agents is None else int(num_agents)
    env_cls = _select_multi_env_class(selected_config)
    env = env_cls(
        multi_config=selected_config,
        env_config=replace(
            selected_config.environment,
            max_episode_steps=int(max_steps or selected_config.environment.max_episode_steps),
        ),
        observation_config=selected_config.observation,
        formation_config=selected_config.formation,
        planner_config=selected_config.planner,
    )
    observation, _ = env.reset(seed=seed, num_agents=active_num_agents)
    max_steps = int(max_steps or env.env_config.max_episode_steps)

    controller: HeuristicFormationReplayController | None = None
    agent: GraphMASACAgent | None = None
    if mode in {"heuristic", "compact_formation"}:
        controller = HeuristicFormationReplayController(env, compact_gate_mode=(mode == "compact_formation"))
    elif mode == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when mode='checkpoint'")
        agent = GraphMASACAgent.from_defaults(
            obs_shapes=env.observation_shapes,
            device=device,
            seed=seed,
            obs_config=selected_config.observation,
            masac_config=selected_config.algorithm,
            max_agents_soft=selected_config.max_agents_soft,
            build_replay_buffer=False,
        )
        validate_multi_checkpoint_compatibility(
            checkpoint_path=checkpoint_path,
            env=env,
            experiment_config=selected_config,
        )
        agent.load_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unsupported replay mode: {mode}")

    trajectory: list[dict[str, object]] = []
    episode_reward = 0.0
    done_reason = None
    final_info: dict[str, object] = {}
    last_info: dict[str, object] = {}
    steps = 0

    def _build_replay_frame(frame_step: int, info_payload: dict[str, object]) -> dict[str, object]:
        snapshot = env.snapshot()
        active_positions = env.active_positions_xy()
        active_velocities = env.active_velocities_xy()
        active_yaws = [float(state.yaw_rad) for state in env._states]
        frame = {
            "step": int(frame_step),
            "t_sec": float(env._states[0].t_sec) if env._states else 0.0,
            "num_agents": int(snapshot.num_agents),
            "fixed_height_m": float(env.env_config.fixed_height_m),
            "positions_xy": active_positions.tolist(),
            "velocities_xy": active_velocities.tolist(),
            "yaws_rad": active_yaws,
            "desired_slots_xy": env.desired_slots_xy().tolist(),
            "virtual_center_xy": list(snapshot.virtual_center_xy),
            "mean_slot_error_m": snapshot.mean_slot_error_m,
            "max_slot_error_m": snapshot.max_slot_error_m,
            "min_pair_distance_m": env._pairwise_collision_stats(active_positions)[1],
            "goal_distance_m": snapshot.goal_distance_m,
            "goal_distance_improvement_m": max(
                0.0,
                float(getattr(env, "_initial_goal_distance", 0.0)) - float(snapshot.goal_distance_m),
            ),
            "path_index": snapshot.path_index,
            "goal_xy": list(env.path_waypoints()[-1]),
            "guidance_tracking_error_m": env._guidance_tracking_error_m(snapshot.virtual_center_xy),
            "route_plan_guidance": (
                info_payload.get("route_plan_guidance")
                if info_payload.get("route_plan_guidance") is not None
                else env._route_plan_guidance_summary(snapshot.virtual_center_xy)
            ),
            "route_guidance": (
                info_payload.get("route_guidance")
                if info_payload.get("route_guidance") is not None
                else env._route_guidance_summary(snapshot.virtual_center_xy)
            ),
        }
        frame.update(_dynamic_gate_frame_payload(env))
        return frame

    for step in range(max_steps):
        trajectory.append(_build_replay_frame(step, last_info))
        if controller is not None:
            action = controller.act()
        else:
            assert agent is not None
            action = agent.act(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        episode_reward += float(reward)
        steps = step + 1
        if terminated or truncated:
            done_reason = str(info.get("done_reason") or "unknown")
            final_info = dict(info)
            terminal_frame = _build_replay_frame(steps, final_info)
            terminal_frame["terminal"] = True
            terminal_frame["done_reason"] = done_reason
            terminal_frame["terminated"] = bool(terminated)
            terminal_frame["truncated"] = bool(truncated)
            terminal_frame["min_clearance_m"] = float(final_info.get("min_clearance_m", env._min_clearance(env.active_positions_xy())))
            trajectory.append(terminal_frame)
            break

    timeout_counts_as_success = bool(selected_config.environment.timeout_counts_as_success)
    success = _replay_success_from_done_reason(
        done_reason,
        timeout_counts_as_success=timeout_counts_as_success,
        info=final_info,
    )
    corridor_through_success = _multi_corridor_through_success_from_info(final_info) if final_info else False
    report = {
        "mode": mode,
        "experiment_id": selected_config.experiment_id,
        "seed": seed,
        "num_agents": active_num_agents,
        "path_waypoints": [list(point) for point in env.path_waypoints()],
        "route_waypoint_names": [f"P{idx + 1}" for idx, _point in enumerate(env.path_waypoints())],
        "steps": steps,
        "success": success,
        "done_reason": done_reason,
        "timeout_counts_as_success": timeout_counts_as_success,
        "height_contract_passed": bool(final_info.get("height_contract_passed", False)),
        "height_escape_failure": bool(final_info.get("height_escape_failure", False)),
        "side_bypass_failure": bool(final_info.get("side_bypass_failure", False)),
        "corridor_miss_failure": bool(final_info.get("corridor_miss_failure", False)),
        "corridor_completed": bool(final_info.get("corridor_completed", False)),
        "corridor_through_success": bool(corridor_through_success),
        "episode_reward": float(episode_reward),
        "snapshot": {
            "virtual_center_xy": list(env.snapshot().virtual_center_xy),
            "mean_slot_error_m": env.snapshot().mean_slot_error_m,
            "max_slot_error_m": env.snapshot().max_slot_error_m,
            "goal_distance_m": env.snapshot().goal_distance_m,
            "goal_distance_improvement_m": max(0.0, float(getattr(env, "_initial_goal_distance", 0.0)) - env.snapshot().goal_distance_m),
            "path_index": env.snapshot().path_index,
            "path_waypoints": [list(point) for point in env.path_waypoints()],
            "min_pair_distance_m": env._pairwise_collision_stats(env.active_positions_xy())[1],
            "guidance_tracking_error_m": env._guidance_tracking_error_m(env.snapshot().virtual_center_xy),
            "route_guidance_tracking_error_m": env._route_guidance_tracking_error_m(env.snapshot().virtual_center_xy),
        },
        "trajectory_len": len(trajectory),
    }

    if output_dir is None:
        artifacts = allocate_replay_artifacts(
            "multi",
            run_name=default_run_name(f"multi_replay_{mode}_{active_num_agents}d"),
        )
        output_path = artifacts.output_dir
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "replay_report.json"
    trajectory_path = output_path / "trajectory.json"
    report["report_path"] = str(report_path)
    report["trajectory_path"] = str(trajectory_path)
    write_json(report_path, report)
    trajectory_payload = {
            "format": "aerogate_graph_multi_replay_v2",
            "fixed_height_m": float(env.env_config.fixed_height_m),
            "goal_radius_m": float(env.env_config.goal_radius_m),
            "world_x_bounds_m": list(env.env_config.world_x_bounds_m),
            "world_y_bounds_m": list(env.env_config.world_y_bounds_m),
            "drone_radius_m": float(env.env_config.drone_radius_m),
            "inter_agent_safe_distance_m": float(env.env_config.inter_agent_safe_distance_m),
            "path_waypoints": [list(point) for point in env.path_waypoints()],
            "route_waypoint_names": [f"P{idx + 1}" for idx, _point in enumerate(env.path_waypoints())],
            "drone_asset_path": str(GLOBAL_CONFIG.drone_asset_file),
            "gate_layout_path": str(GLOBAL_CONFIG.gate_layout_file),
            "trajectory": trajectory,
    }
    dynamic_gate_metadata = _dynamic_gate_replay_metadata(env)
    if dynamic_gate_metadata is not None:
        trajectory_payload["dynamic_gate_density"] = dynamic_gate_metadata
        report["dynamic_gate_density"] = {
            "enabled": True,
            "gate_count": dynamic_gate_metadata.get("gate_count"),
            "moving_gate_speed_mps": dynamic_gate_metadata.get("moving_gate_speed_mps"),
            "moving_gate_amplitude_m": dynamic_gate_metadata.get("moving_gate_amplitude_m"),
        }
        write_json(report_path, report)
    write_json(trajectory_path, trajectory_payload)
    return report


def render_multi_replay_isaaclab_from_summary(
    *,
    replay_summary: dict[str, object],
    experiment_config: MultiExperimentConfig,
    output_dir: str | Path | None = None,
    export_video: bool = False,
    fps: int = 10,
    camera_mode: str = "picture_in_picture",
    headless: bool = True,
) -> dict[str, object]:
    """Render a saved replay summary inside IsaacLab using the real gate and drone assets."""

    from multi_gate.configs import infer_multi_config_name_from_experiment_id

    trajectory_path = str(replay_summary.get("trajectory_path") or "").strip()
    report_path = str(replay_summary.get("report_path") or "").strip()
    if not trajectory_path:
        raise ValueError("Replay summary is missing 'trajectory_path' required for IsaacLab rendering.")
    if not report_path:
        raise ValueError("Replay summary is missing 'report_path' required for IsaacLab rendering.")

    script_path = Path(__file__).resolve().parent / "scripts" / "replay_multi_isaaclab.py"
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(report_path).resolve().parent / "isaaclab"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_camera_mode = str(camera_mode or "picture_in_picture").strip().lower()
    if resolved_camera_mode not in {
        "global",
        "follow",
        "picture_in_picture",
        "height_audit",
        "top_global",
        "top_centroid_follow",
    }:
        raise ValueError(f"Unsupported IsaacLab replay camera_mode: {camera_mode}")

    config_name = infer_multi_config_name_from_experiment_id(experiment_config.experiment_id) or "variable"
    command = [
        sys.executable,
        str(script_path),
        "--trajectory",
        trajectory_path,
        "--report",
        report_path,
        "--output-dir",
        str(resolved_output_dir),
        "--config-name",
        str(config_name),
        "--scene-mode",
        str(experiment_config.scene.scene_mode),
        "--render-real-gate",
        "1" if bool(experiment_config.scene.render_real_gate) else "0",
        "--render-real-drone-shell",
        "1" if bool(experiment_config.scene.render_real_drone_shell) else "0",
        "--fps",
        str(max(int(fps), 1)),
        "--camera-mode",
        resolved_camera_mode,
    ]
    resolved_mp4_path = None
    if export_video:
        resolved_mp4_path = resolved_output_dir / "isaaclab_replay.mp4"
        command.extend(["--mp4-path", str(resolved_mp4_path)])
    if headless:
        command.append("--headless")

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Multi-agent IsaacLab replay rendering failed.\n"
            f"command={' '.join(command)}\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    summary_path = resolved_output_dir / "isaaclab_replay_summary.json"
    rendered_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {
            "summary_path": str(summary_path),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    if resolved_mp4_path is not None:
        rendered_summary["mp4_path"] = str(resolved_mp4_path)
    rendered_summary["command"] = command
    rendered_summary["camera_mode"] = rendered_summary.get("camera_mode") or resolved_camera_mode
    return rendered_summary

