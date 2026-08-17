"""Frozen identifiers and leakage vocabulary for episode manifest v1."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterator

EPISODE_SCHEMA = "org.rivermark.benchmark.episode.v1"

ALLOWED_SPLITS = frozenset(
    {"pilot", "train", "inner_dev", "validation", "blind_test", "ood_test"}
)

ALLOWED_OBSERVATION_SCOPES = frozenset(
    {"decentralized_explicit_comm", "centralized_team_state"}
)

_COMMON_PUBLIC_MODALITIES = frozenset(
    {
        "high_level_action_history",
        "proprioception",
        "public_task_state",
        "public_team_messages",
    }
)

INFORMATION_PROFILE_MODALITIES = {
    "geometry_state": _COMMON_PUBLIC_MODALITIES | {"public_geometry"},
    "state_only": _COMMON_PUBLIC_MODALITIES,
    "egocentric_rgb_state": _COMMON_PUBLIC_MODALITIES | {"rgb"},
    "egocentric_rgbd_state": _COMMON_PUBLIC_MODALITIES
    | {"rgb", "distance_to_image_plane"},
    "language_rgbd_state": _COMMON_PUBLIC_MODALITIES
    | {"rgb", "distance_to_image_plane", "language"},
    # These profiles are intentionally closed. A result produced with a
    # radar or lidar input is not comparable to an RGB-D-only policy, and an
    # RGB-D/LiDAR/IMU capture must not claim radar merely to fit a profile.
    "multisensor_rgbd_lidar_imu_state": _COMMON_PUBLIC_MODALITIES
    | {"rgb", "distance_to_image_plane", "lidar", "imu"},
    "multisensor_rgbd_lidar_radar_state": _COMMON_PUBLIC_MODALITIES
    | {"rgb", "distance_to_image_plane", "lidar", "radar", "imu"},
    "language_multisensor_rgbd_lidar_radar_state": _COMMON_PUBLIC_MODALITIES
    | {"rgb", "distance_to_image_plane", "lidar", "radar", "imu", "language"},
}

FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "current_outcome",
        "current_return",
        "discovered_target_ids",
        "discovered_target_indices",
        "episode_outcome",
        "evaluator_match",
        "evaluator_matches",
        "evaluator_private",
        "future_observation",
        "ground_truth",
        "hidden_manifest",
        "hidden_target_id",
        "hidden_target_ids",
        "hidden_target_xyz",
        "oracle_action",
        "privileged_labels",
        "representative_target_id",
        "representative_target_index",
        "reward",
        "return",
        "runtime_seed",
        "seed",
        "target_coordinates",
        "target_id",
        "target_ids",
        "target_index",
        "target_indices",
        "target_xyz",
    }
)

FORBIDDEN_POLICY_VALUE_TOKENS = (
    "evaluator_private",
    "ground_truth",
    "hidden_manifest",
    "hidden_target",
    "oracle",
    "post-hoc",
    "post_hoc",
    "target-conditioned",
    "target_conditioned",
    "target_coordinates",
    "target_xyz",
)

_FORBIDDEN_POLICY_KEY_TOKENS = frozenset(
    {
        "evaluator",
        "future",
        "groundtruth",
        "hidden",
        "oracle",
        "privileged",
        "reward",
        "return",
        "seed",
    }
)

_TARGET_TRUTH_SUFFIXES = frozenset(
    {
        "coord",
        "coordinate",
        "coordinates",
        "id",
        "ids",
        "index",
        "indices",
        "manifest",
        "pose",
        "position",
        "positions",
        "truth",
        "xyz",
    }
)

_LEGAL_POLICY_KEYS = frozenset(
    {
        "target_waypoint_xyz",
    }
)


def normalized_key(value: object) -> str:
    """Normalize a key without hiding semantic token boundaries."""

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value).strip())
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def forbidden_policy_key(value: object) -> bool:
    """Detect evaluator truth even when key style or separators change."""

    key = normalized_key(value)
    if key in _LEGAL_POLICY_KEYS:
        return False
    if key in FORBIDDEN_POLICY_KEYS:
        return True
    tokens = set(key.split("_"))
    compact = key.replace("_", "")
    if tokens & _FORBIDDEN_POLICY_KEY_TOKENS:
        return True
    if compact in {"groundtruth", "hiddentarget", "runtimegpu"}:
        return True
    truth_object = bool(tokens & {"casualty", "rescue", "target", "victim"})
    return truth_object and bool(tokens & _TARGET_TRUTH_SUFFIXES)


def forbidden_policy_value_token(value: str) -> str | None:
    """Return the forbidden provenance phrase referenced by a public string."""

    lowered = value.lower()
    normalized = normalized_key(value)
    for token in FORBIDDEN_POLICY_VALUE_TOKENS:
        if token in lowered or normalized_key(token) in normalized:
            return token
    return None


def iter_tree(value: Any, path: str = "$") -> Iterator[tuple[str, str | None, Any]]:
    """Yield every node as (path, dictionary key, value)."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, str(key), child
            yield from iter_tree(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield child_path, None, child
            yield from iter_tree(child, child_path)


def is_safe_relative_path(value: object) -> bool:
    """Reject absolute paths and parent traversal on either host convention."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\x00" in text or re.match(r"^[a-z][a-z0-9+.-]*://", text, re.IGNORECASE):
        return False
    posix = PurePosixPath(text.replace("\\", "/"))
    windows = PureWindowsPath(text)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in posix.parts
        and "~" not in posix.parts
    )


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))
