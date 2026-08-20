import unittest

from robocon_game_supervisor.ros_node import RoboconGameSupervisorNode


class RosAdapterContractTests(unittest.TestCase):
    def test_plain_and_json_commands_are_supported(self):
        self.assertEqual(RoboconGameSupervisorNode._parse_command(" start "), "start")
        self.assertEqual(RoboconGameSupervisorNode._parse_command('{"command":"fire_shot"}'), "fire_shot")

    def test_malformed_commands_are_rejected(self):
        self.assertIsNone(RoboconGameSupervisorNode._parse_command(""))
        self.assertIsNone(RoboconGameSupervisorNode._parse_command("{"))
        self.assertIsNone(RoboconGameSupervisorNode._parse_command('{"message":"start"}'))


if __name__ == "__main__":
    unittest.main()
