"""Pure execution contracts shared by simulation run recorders."""

from __future__ import annotations


class StreamReadiness:
    """Require several valid packets before simulated motion starts."""

    def __init__(self, required_lidar_packets: int, required_imu_packets: int) -> None:
        if required_lidar_packets <= 0 or required_imu_packets <= 0:
            raise ValueError("required packet counts must be positive")
        self.required_lidar_packets = required_lidar_packets
        self.required_imu_packets = required_imu_packets
        self.lidar_packets = 0
        self.imu_packets = 0

    def observe_lidar(self, valid: bool) -> None:
        if valid:
            self.lidar_packets += 1

    def observe_imu(self, valid: bool) -> None:
        if valid:
            self.imu_packets += 1

    @property
    def ready(self) -> bool:
        return (
            self.lidar_packets >= self.required_lidar_packets
            and self.imu_packets >= self.required_imu_packets
        )

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "required_lidar_packets": self.required_lidar_packets,
            "required_imu_packets": self.required_imu_packets,
            "accepted_lidar_packets": self.lidar_packets,
            "accepted_imu_packets": self.imu_packets,
            "ready": self.ready,
        }
