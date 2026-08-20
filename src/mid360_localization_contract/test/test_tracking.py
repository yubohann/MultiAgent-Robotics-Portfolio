import unittest

from mid360_localization_contract.tracking import TrackingState, TrackingStateMachine


class TrackingStateMachineTests(unittest.TestCase):
    def test_initial_state_is_uninitialized(self):
        tracker = TrackingStateMachine()
        self.assertEqual(tracker.state, TrackingState.UNINITIALIZED)
        self.assertFalse(tracker.map_locked)

    def test_fresh_valid_pose_after_anchor_is_tracking(self):
        tracker = TrackingStateMachine()
        tracker.accept_anchor()
        tracker.update(pose_valid=True, odom_fresh=True)
        self.assertEqual(tracker.state, TrackingState.TRACKING)
        self.assertTrue(tracker.map_locked)

    def test_initial_pose_is_provisional_until_scan_to_map_correction(self):
        tracker = TrackingStateMachine()
        tracker.set_provisional_anchor()
        tracker.update(pose_valid=True, odom_fresh=True)
        self.assertEqual(tracker.state, TrackingState.RELOCALIZING)
        self.assertFalse(tracker.map_locked)
        self.assertTrue(tracker.anchor_exists)
        self.assertFalse(tracker.verified_anchor)

        tracker.accept_anchor()
        tracker.update(pose_valid=True, odom_fresh=True)
        self.assertEqual(tracker.state, TrackingState.TRACKING)
        self.assertTrue(tracker.map_locked)

    def test_stale_or_invalid_pose_is_lost(self):
        tracker = TrackingStateMachine()
        tracker.accept_anchor()
        tracker.update(pose_valid=True, odom_fresh=False)
        self.assertEqual(tracker.state, TrackingState.LOST)
        self.assertFalse(tracker.map_locked)

    def test_relocalization_request_blocks_map_lock(self):
        tracker = TrackingStateMachine()
        tracker.accept_anchor()
        tracker.request_relocalization()
        tracker.update(pose_valid=True, odom_fresh=True)
        self.assertEqual(tracker.state, TrackingState.RELOCALIZING)
        self.assertFalse(tracker.map_locked)
        self.assertFalse(tracker.verified_anchor)

    def test_new_anchor_restores_tracking(self):
        tracker = TrackingStateMachine()
        tracker.request_relocalization()
        tracker.accept_anchor()
        tracker.update(pose_valid=True, odom_fresh=True)
        self.assertEqual(tracker.state, TrackingState.TRACKING)
        self.assertFalse(tracker.relocalization_requested)
        self.assertTrue(tracker.verified_anchor)
        self.assertTrue(tracker.map_locked)


if __name__ == "__main__":
    unittest.main()
