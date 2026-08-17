from __future__ import annotations

import copy
import unittest

from rivermark_benchmark.search_event_evaluator import (
    EVENT_SUBMISSION_SCHEMA,
    PRIVATE_TASK_SCHEMA,
    SearchEventEvaluationError,
    evaluate_search_events,
)


def _task() -> dict[str, object]:
    return {
        "schema": PRIVATE_TASK_SCHEMA,
        "episode_id": "episode-001",
        "agent_count": 8,
        "time_budget_s": 10.0,
        "match_radius_m": 0.75,
        "maximum_false_confirmations": 0,
        "observation_time_tolerance_s": 0.0,
        "observations": [
            {"observation_id": "obs-agent0-frame002", "agent_id": 0, "timestamp_s": 2.0},
            {"observation_id": "obs-agent7-frame006", "agent_id": 7, "timestamp_s": 6.0},
            {"observation_id": "obs-agent1-frame000", "agent_id": 1, "timestamp_s": 0.5},
            {"observation_id": "obs-agent2-frame007", "agent_id": 2, "timestamp_s": 7.0},
            {"observation_id": "obs-agent3-frame003", "agent_id": 3, "timestamp_s": 3.0},
        ],
        "targets": [
            {
                "target_id": "target-001",
                "position_w_m": [2.0, 0.0, 1.0],
                "visible_observation_ids": ["obs-agent0-frame002"],
            },
            {
                "target_id": "target-002",
                "position_w_m": [6.0, 0.0, 1.0],
                "visible_observation_ids": ["obs-agent7-frame006", "obs-agent2-frame007"],
            },
        ],
        "safety_violations": {
            "collision": 0,
            "geofence": 0,
            "visual_intrusion": 0,
        },
    }


def _submission() -> dict[str, object]:
    return {
        "schema": EVENT_SUBMISSION_SCHEMA,
        "episode_id": "episode-001",
        "events": [
            {
                "event_id": "event-001",
                "timestamp_s": 2.0,
                "agent_id": 0,
                "source_observation_id": "obs-agent0-frame002",
                "position_w_m": [2.1, 0.0, 1.0],
                "confidence": 0.9,
            },
            {
                "event_id": "event-002",
                "timestamp_s": 6.0,
                "agent_id": 7,
                "source_observation_id": "obs-agent7-frame006",
                "position_w_m": [6.0, 0.1, 1.0],
                "confidence": 0.8,
            },
        ],
    }


class SearchEventEvaluatorTests(unittest.TestCase):
    def test_matches_sensor_visible_events_against_evaluator_owned_truth(self) -> None:
        report = evaluate_search_events(_submission(), private_task=_task())

        self.assertTrue(report.eligible)
        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.false_confirmation_count, 0)
        self.assertEqual(report.score.final_recall, 1.0)
        public = report.public_dict()
        self.assertNotIn("matches", public)
        self.assertNotIn("position_w_m", repr(public))

    def test_policy_cannot_self_report_task_facts_or_confirmed_counts(self) -> None:
        for key, value in (
            ("target_count", 1),
            ("time_budget_s", 1000.0),
            ("confirmed_counts", [99]),
        ):
            submission = _submission()
            submission[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                SearchEventEvaluationError, "evaluator-owned"
            ):
                evaluate_search_events(submission, private_task=_task())

    def test_duplicate_and_outside_visibility_events_are_false_confirmations(self) -> None:
        submission = _submission()
        events = submission["events"]
        assert isinstance(events, list)
        events.insert(
            0,
            {
                "event_id": "event-too-early",
                "timestamp_s": 0.5,
                "agent_id": 1,
                "source_observation_id": "obs-agent1-frame000",
                "position_w_m": [2.0, 0.0, 1.0],
            },
        )
        events.append(
            {
                "event_id": "event-duplicate",
                "timestamp_s": 7.0,
                "agent_id": 2,
                "source_observation_id": "obs-agent2-frame007",
                "position_w_m": [6.0, 0.0, 1.0],
            }
        )

        report = evaluate_search_events(submission, private_task=_task())

        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.false_confirmation_count, 2)
        self.assertEqual(report.outside_visibility_count, 1)
        self.assertEqual(report.duplicate_confirmation_count, 1)
        self.assertFalse(report.confirmation_quality_passed)
        self.assertFalse(report.eligible)
        self.assertEqual(report.maximum_false_confirmations, 0)

    def test_private_false_confirmation_budget_is_an_evaluator_owned_hard_gate(self) -> None:
        task = _task()
        task["maximum_false_confirmations"] = 1
        submission = _submission()
        events = submission["events"]
        assert isinstance(events, list)
        events.append(
            {
                "event_id": "event-false",
                "timestamp_s": 3.0,
                "agent_id": 3,
                "source_observation_id": "obs-agent3-frame003",
                "position_w_m": [20.0, 0.0, 1.0],
                "confidence": 0.2,
            }
        )

        report = evaluate_search_events(submission, private_task=task)

        self.assertTrue(report.safety_passed)
        self.assertTrue(report.confirmation_quality_passed)
        self.assertTrue(report.eligible)
        self.assertEqual(report.false_confirmation_count, 1)
        self.assertEqual(report.maximum_false_confirmations, 1)
        public = report.public_dict()
        self.assertNotIn("matches", public)
        self.assertTrue(public["confirmation_quality_passed"])

    def test_synchronous_multi_agent_matches_share_one_metric_timestamp(self) -> None:
        task = _task()
        task["targets"] = [
            {
                "target_id": "target-a",
                "position_w_m": [1.0, 0.0, 1.0],
                "visible_observation_ids": ["obs-agent0-frame002"],
            },
            {
                "target_id": "target-b",
                "position_w_m": [4.0, 0.0, 1.0],
                "visible_observation_ids": ["obs-agent1-frame002"],
            },
        ]
        task["observations"].append(
            {"observation_id": "obs-agent1-frame002", "agent_id": 1, "timestamp_s": 2.0}
        )
        submission = _submission()
        submission["events"] = [
            {
                "event_id": "event-a",
                "timestamp_s": 2.0,
                "agent_id": 0,
                "source_observation_id": "obs-agent0-frame002",
                "position_w_m": [1.0, 0.0, 1.0],
                "confidence": 1.0,
            },
            {
                "event_id": "event-b",
                "timestamp_s": 2.0,
                "agent_id": 1,
                "source_observation_id": "obs-agent1-frame002",
                "position_w_m": [4.0, 0.0, 1.0],
                "confidence": 1.0,
            },
        ]

        report = evaluate_search_events(submission, private_task=task)

        self.assertTrue(report.eligible)
        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.score.time_to_all_targets_s, 2.0)

    def test_invalid_private_false_confirmation_budget_is_rejected(self) -> None:
        for invalid in (None, -1, 0.5, True, "zero"):
            task = _task()
            task["maximum_false_confirmations"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                SearchEventEvaluationError, "maximum_false_confirmations"
            ):
                evaluate_search_events(_submission(), private_task=task)

    def test_private_task_requires_the_complete_frozen_safety_set(self) -> None:
        for mutation in ("remove", "add"):
            task = _task()
            safety = task["safety_violations"]
            assert isinstance(safety, dict)
            if mutation == "remove":
                del safety["collision"]
            else:
                safety["unspecified"] = 0
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                SearchEventEvaluationError, "must contain exactly"
            ):
                evaluate_search_events(_submission(), private_task=task)

    def test_safety_facts_are_evaluator_owned_hard_eligibility_gate(self) -> None:
        task = _task()
        safety = task["safety_violations"]
        assert isinstance(safety, dict)
        safety["collision"] = 1

        report = evaluate_search_events(_submission(), private_task=task)

        self.assertFalse(report.eligible)
        self.assertFalse(report.safety_passed)
        self.assertEqual(report.matched_count, 2)

    def test_truth_requires_nonambiguous_matching_and_known_observations(self) -> None:
        ambiguous = _task()
        targets = ambiguous["targets"]
        assert isinstance(targets, list)
        second = targets[1]
        assert isinstance(second, dict)
        second["position_w_m"] = [3.0, 0.0, 1.0]
        with self.assertRaisesRegex(SearchEventEvaluationError, "twice the match radius"):
            evaluate_search_events(_submission(), private_task=ambiguous)

        unknown_observation = copy.deepcopy(_task())
        targets = unknown_observation["targets"]
        assert isinstance(targets, list)
        first = targets[0]
        assert isinstance(first, dict)
        first["visible_observation_ids"] = ["obs-not-in-task"]
        with self.assertRaisesRegex(SearchEventEvaluationError, "unknown evaluator observations"):
            evaluate_search_events(_submission(), private_task=unknown_observation)

    def test_candidate_must_bind_to_its_own_evaluator_observation(self) -> None:
        for field, value in (
            ("source_observation_id", "obs-not-in-task"),
            ("agent_id", 1),
            ("timestamp_s", 2.1),
        ):
            submission = _submission()
            events = submission["events"]
            assert isinstance(events, list)
            first = events[0]
            assert isinstance(first, dict)
            first[field] = value
            with self.subTest(field=field):
                report = evaluate_search_events(submission, private_task=_task())
                self.assertEqual(report.matched_count, 1)
                self.assertEqual(report.observation_evidence_mismatch_count, 1)
                self.assertEqual(report.false_confirmation_count, 1)
                self.assertFalse(report.confirmation_quality_passed)
                self.assertFalse(report.eligible)

    def test_event_after_evaluator_budget_is_rejected(self) -> None:
        submission = _submission()
        events = submission["events"]
        assert isinstance(events, list)
        first = events[0]
        assert isinstance(first, dict)
        first["timestamp_s"] = 10.5

        with self.assertRaisesRegex(SearchEventEvaluationError, "exceeds evaluator time budget"):
            evaluate_search_events(submission, private_task=_task())


if __name__ == "__main__":
    unittest.main()
