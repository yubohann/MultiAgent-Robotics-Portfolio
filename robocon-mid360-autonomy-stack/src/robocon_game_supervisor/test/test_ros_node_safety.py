import time
import unittest

import rclpy
from std_msgs.msg import String

from robocon_game_supervisor.actions import ActionRequest
from robocon_game_supervisor.protocol import MessageEnvelope, envelope_to_json
from robocon_game_supervisor.ros_node import RoboconGameSupervisorNode
from robocon_game_supervisor.supervisor import SupervisorState


class RosNodeSafetyTests(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = RoboconGameSupervisorNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def _enter_active(self):
        self.node.supervisor.enter_preflight()
        self.node.supervisor.preflight_result(True)
        self.node.supervisor.start()

    def test_operator_estop_latches_before_hardware_feedback(self):
        self.node._dispatch("emergency_stop")
        self.assertEqual(self.node.supervisor.state, SupervisorState.ESTOP)
        self.assertEqual(len(self.node._pending_actions), 1)

    def test_expired_action_enters_recovery_and_is_retained(self):
        self._enter_active()
        request = ActionRequest(
            1,
            "timeout-1",
            self.node._task_id,
            "ExecutePass",
            "test",
            1,
            1,
            2,
            {},
        )
        self.node._pending_actions[request.action_id] = request
        self.node._action_status[request.action_id] = "published"
        self.node._expire_actions()
        self.assertEqual(self.node.supervisor.state, SupervisorState.RECOVERY)
        self.assertEqual(self.node._completed_actions["ExecutePass"]["state"], "expired")

    def test_other_task_peer_message_is_rejected_before_state_progression(self):
        self._enter_active()
        now = time.time_ns()
        message = MessageEnvelope(
            1,
            "receiver_ready",
            "another-task",
            "ball_handler",
            1,
            now - 1,
            now + 1_000_000_000,
            {},
        )
        self.node._team_message_callback(String(data=envelope_to_json(message)))
        self.assertEqual(self.node._team_link.task_mismatch_count, 1)
        self.assertEqual(self.node.supervisor.task_state[next(iter(self.node.supervisor.task_state))].value, "idle")


if __name__ == "__main__":
    unittest.main()
