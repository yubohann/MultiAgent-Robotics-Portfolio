import unittest

from robocon_game_supervisor.actions import ActionDeduplicator, ActionRequest
from robocon_game_supervisor.protocol import (
    Deduplicator,
    MessageEnvelope,
    TeamLink,
    envelope_from_json,
    envelope_to_json,
)
from robocon_game_supervisor.supervisor import (
    GameSupervisor,
    SafetySnapshot,
    SupervisorState,
)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = GameSupervisor()
        self.supervisor.enter_preflight()
        self.supervisor.preflight_result(True)
        self.supervisor.start()

    @staticmethod
    def _healthy_safety() -> SafetySnapshot:
        return SafetySnapshot(True, True, 0.05, True, 0.05, True, True, True)

    def test_fire_shot_requires_all_evidence(self):
        safety = SafetySnapshot(True, True, 0.05, True, 0.05, True, True, True)
        decision = self.supervisor.request_fire_shot(safety)
        self.assertTrue(decision.accepted)

        self.supervisor = GameSupervisor()
        self.supervisor.enter_preflight()
        self.supervisor.preflight_result(True)
        self.supervisor.start()
        stale = SafetySnapshot(True, True, 0.31, True, 0.05, True, True, True)
        decision = self.supervisor.request_fire_shot(stale)
        self.assertFalse(decision.accepted)
        self.assertEqual(self.supervisor.state, SupervisorState.ACTIVE)

    def test_pass_requires_receipt_evidence_in_order(self):
        self.assertFalse(self.supervisor.confirm_receipt().accepted)
        self.assertTrue(self.supervisor.set_receiver_ready().accepted)
        self.assertTrue(self.supervisor.arm_pass().accepted)
        self.assertTrue(self.supervisor.confirm_pass_executed().accepted)
        self.assertTrue(self.supervisor.confirm_receipt().accepted)

    def test_estop_is_terminal_until_external_reset(self):
        self.supervisor.emergency_stop()
        self.assertEqual(self.supervisor.state, SupervisorState.ESTOP)
        with self.assertRaises(RuntimeError):
            self.supervisor.start()

    def test_action_feedback_drives_evidence_chain(self):
        self.assertTrue(self.supervisor.can_issue_action("PrepareReceive").accepted)
        self.assertTrue(self.supervisor.apply_action_success("PrepareReceive").accepted)
        self.assertTrue(self.supervisor.can_issue_action("PreparePass", self._healthy_safety()).accepted)
        self.assertTrue(self.supervisor.apply_action_success("PreparePass").accepted)
        self.assertTrue(self.supervisor.can_issue_action("ExecutePass", self._healthy_safety()).accepted)
        self.assertTrue(self.supervisor.apply_action_success("ExecutePass").accepted)
        self.assertTrue(self.supervisor.can_issue_action("CollectBall").accepted)
        self.assertTrue(
            self.supervisor.apply_action_success("CollectBall", {"receipt_confirmed": True}).accepted
        )

    def test_action_prerequisites_block_out_of_order_execution(self):
        decision = self.supervisor.can_issue_action("ExecutePass")
        self.assertFalse(decision.accepted)
        self.assertIn("pass_armed", decision.reason)

    def test_location_gated_actions_reject_an_unlocked_map(self):
        unsafe = SafetySnapshot(True, False, 0.05, True, 0.05, True, True, True)
        decision = self.supervisor.can_issue_action("NavigateToPose", unsafe)
        self.assertFalse(decision.accepted)
        self.assertIn("map_locked is false", decision.reason)

        self.assertTrue(self.supervisor.apply_action_success("PrepareReceive").accepted)
        decision = self.supervisor.can_issue_action("PreparePass", unsafe)
        self.assertFalse(decision.accepted)
        self.assertIn("map_locked is false", decision.reason)
        self.assertTrue(self.supervisor.apply_action_success("PreparePass").accepted)
        decision = self.supervisor.can_issue_action("ExecutePass", unsafe)
        self.assertFalse(decision.accepted)
        self.assertIn("map_locked is false", decision.reason)
        decision = self.supervisor.can_issue_action("PrepareShot", unsafe)
        self.assertFalse(decision.accepted)
        self.assertIn("map_locked is false", decision.reason)

    def test_runtime_localization_loss_enters_recovery(self):
        safety = SafetySnapshot(False, True, 0.05, False, 0.05, True, True, True)
        decision = self.supervisor.monitor_safety(safety)
        self.assertFalse(decision.accepted)
        self.assertEqual(self.supervisor.state, SupervisorState.RECOVERY)
        self.assertIn("pose_valid is false", self.supervisor.last_failure_reason)

    def test_runtime_teammate_loss_enters_recovery_when_required(self):
        safety = SafetySnapshot(True, True, 0.05, False, 0.05, True, True, False)
        decision = self.supervisor.monitor_safety(safety, require_teammate=True)
        self.assertFalse(decision.accepted)
        self.assertEqual(self.supervisor.state, SupervisorState.RECOVERY)
        self.assertIn("teammate safety", self.supervisor.last_failure_reason)

    def test_target_loss_does_not_force_recovery(self):
        safety = SafetySnapshot(True, True, 0.05, False, 0.50, True, True, True)
        decision = self.supervisor.monitor_safety(safety)
        self.assertTrue(decision.accepted)
        self.assertEqual(self.supervisor.state, SupervisorState.ACTIVE)

    def test_failed_action_enters_recovery_and_blocks_further_actions(self):
        decision = self.supervisor.handle_action_failure("ExecutePass", "MCU feedback timeout")
        self.assertFalse(decision.accepted)
        self.assertEqual(self.supervisor.state, SupervisorState.RECOVERY)
        self.assertIn("ExecutePass", self.supervisor.last_failure_reason)
        self.assertFalse(self.supervisor.can_issue_action("PrepareShot").accepted)

    def test_emergency_stop_failure_keeps_estop_latched(self):
        decision = self.supervisor.handle_action_failure("EmergencyStop", "transport did not confirm")
        self.assertFalse(decision.accepted)
        self.assertEqual(self.supervisor.state, SupervisorState.ESTOP)


class ProtocolTests(unittest.TestCase):
    def test_duplicate_and_expired_messages_are_rejected(self):
        envelope = MessageEnvelope(1, "receiver_ready", "task-1", "shooter", 1, 100, 200, {})
        dedup = Deduplicator()
        self.assertTrue(dedup.accept(envelope, 150))
        self.assertFalse(dedup.accept(envelope, 150))
        expired = MessageEnvelope(1, "receiver_ready", "task-1", "shooter", 2, 100, 200, {})
        self.assertFalse(dedup.accept(expired, 201))
        self.assertEqual(dedup.duplicate_count, 1)
        self.assertEqual(dedup.stale_count, 1)

    def test_action_request_is_expiring_and_idempotent(self):
        request = ActionRequest(1, "a-1", "task-1", "PrepareReceive", "target computer", 1, 100, 200, {})
        request.validate()
        dedup = ActionDeduplicator()
        self.assertTrue(dedup.accept(request, 150))
        self.assertFalse(dedup.accept(request, 150))
        self.assertFalse(dedup.accept(request, 201))
        self.assertEqual(dedup.duplicate_count, 1)

    def test_team_link_returns_ack_and_rejects_duplicate(self):
        envelope = MessageEnvelope(1, "heartbeat", "task-1", "ball_handler", 7, 100, 200, {})
        encoded = envelope_to_json(envelope)
        self.assertEqual(envelope_from_json(encoded), envelope)
        link = TeamLink("shooter", ack_ttl_sec=1.0)
        ack, accepted, reason = link.receive(encoded, 150)
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertEqual(ack.message_type, "ack")
        duplicate_ack, duplicate_accepted, duplicate_reason = link.receive(encoded, 150)
        self.assertFalse(duplicate_accepted)
        self.assertEqual(duplicate_reason, "stale_or_duplicate")
        self.assertFalse(duplicate_ack.payload["accepted"])

    def test_team_link_rejects_messages_for_another_task(self):
        envelope = MessageEnvelope(1, "receiver_ready", "other-task", "ball_handler", 1, 100, 200, {})
        link = TeamLink("shooter", expected_task_id="active-task")
        ack, accepted, reason = link.receive(envelope_to_json(envelope), 150)
        self.assertFalse(accepted)
        self.assertEqual(reason, "task_mismatch")
        self.assertFalse(ack.payload["accepted"])
        self.assertEqual(link.task_mismatch_count, 1)


if __name__ == "__main__":
    unittest.main()

