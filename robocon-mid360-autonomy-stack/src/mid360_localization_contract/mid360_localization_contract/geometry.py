"""Small dependency-free helpers for rigid transforms used by the ROS nodes."""

from __future__ import annotations

import math
from typing import Sequence, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Transform = Tuple[Vector3, Quaternion]


def vector_add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def vector_scale(vector: Vector3, scalar: float) -> Vector3:
    return tuple(value * scalar for value in vector)  # type: ignore[return-value]


def vector_norm(vector: Vector3) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def normalize_quaternion(quaternion: Sequence[float]) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm is zero.")
    return tuple(float(value / norm) for value in quaternion)  # type: ignore[return-value]


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return normalize_quaternion((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ))


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quaternion
    return -x, -y, -z, w


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    x, y, z, w = normalize_quaternion(quaternion)
    tx = 2.0 * (y * vector[2] - z * vector[1])
    ty = 2.0 * (z * vector[0] - x * vector[2])
    tz = 2.0 * (x * vector[1] - y * vector[0])
    return (
        vector[0] + w * tx + (y * tz - z * ty),
        vector[1] + w * ty + (z * tx - x * tz),
        vector[2] + w * tz + (x * ty - y * tx),
    )


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return normalize_quaternion((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ))


def yaw_from_quaternion(quaternion: Quaternion) -> float:
    x, y, z, w = normalize_quaternion(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compose_transform(first: Transform, second: Transform) -> Transform:
    first_translation, first_rotation = first
    second_translation, second_rotation = second
    return (
        vector_add(first_translation, rotate_vector(first_rotation, second_translation)),
        quaternion_multiply(first_rotation, second_rotation),
    )


def invert_transform(transform: Transform) -> Transform:
    translation, rotation = transform
    inverse_rotation = quaternion_conjugate(normalize_quaternion(rotation))
    inverse_translation = rotate_vector(inverse_rotation, vector_scale(translation, -1.0))
    return inverse_translation, inverse_rotation
