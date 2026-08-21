import math
import unittest

from mid360_localization_contract.geometry import (
    compose_transform,
    invert_transform,
    quaternion_from_rpy,
    yaw_from_quaternion,
)


class GeometryTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        for value, reference in zip(actual, expected):
            self.assertAlmostEqual(value, reference, places=9)

    def test_compose_with_inverse_is_identity(self):
        transform = ((1.2, -0.5, 0.4), quaternion_from_rpy(0.1, -0.2, 0.3))
        translation, rotation = compose_transform(transform, invert_transform(transform))
        self.assertVectorAlmostEqual(translation, (0.0, 0.0, 0.0))
        self.assertVectorAlmostEqual(rotation, (0.0, 0.0, 0.0, 1.0))

    def test_sensor_height_is_removed_when_converting_to_base(self):
        odom_to_imu = ((2.0, 3.0, 1.4), quaternion_from_rpy(0.0, 0.0, 0.0))
        base_to_imu = ((0.0, 0.0, 0.4), quaternion_from_rpy(0.0, 0.0, 0.0))
        translation, _ = compose_transform(odom_to_imu, invert_transform(base_to_imu))
        self.assertVectorAlmostEqual(translation, (2.0, 3.0, 1.0))

    def test_yaw_uses_the_full_quaternion(self):
        yaw = 1.1
        quaternion = quaternion_from_rpy(0.25, -0.35, yaw)
        self.assertAlmostEqual(yaw_from_quaternion(quaternion), yaw, places=9)
        self.assertNotAlmostEqual(quaternion[2], yaw, places=2)

    def test_non_unit_quaternion_is_normalized(self):
        yaw = 0.7
        quaternion = quaternion_from_rpy(0.0, 0.0, yaw)
        scaled = tuple(component * 4.0 for component in quaternion)
        self.assertAlmostEqual(yaw_from_quaternion(scaled), yaw, places=9)

    def test_zero_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            yaw_from_quaternion((0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
