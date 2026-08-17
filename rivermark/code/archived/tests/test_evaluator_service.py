from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.evaluator_service import (
    EvaluatorAuthenticationError,
    EvaluatorRateLimitError,
    EvaluatorServiceError,
    LocalEvaluatorService,
    verify_signed_result,
)
from tests.test_evaluator import _submission


@unittest.skipUnless(importlib.util.find_spec("cryptography"), "optional service signing dependency is not installed")
class EvaluatorServiceTests(unittest.TestCase):
    def test_authenticated_signed_result_and_idempotent_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            now = [100.0]
            audit = Path(temporary) / "audit.jsonl"
            service = LocalEvaluatorService(
                auth_token="local-secret",
                max_requests=3,
                window_seconds=10,
                audit_path=audit,
                clock=lambda: now[0],
                wall_clock=lambda: 123.0,
            )
            raw = json.dumps(_submission(), sort_keys=True).encode("utf-8")
            first = service.submit(raw, authorization="Bearer local-secret")
            verify_signed_result(first, service.public_key_bytes)
            self.assertEqual(first["report"]["status"], "valid")
            self.assertFalse(first["replayed"])
            second = service.submit(raw, authorization="Bearer local-secret")
            verify_signed_result(second, service.public_key_bytes)
            self.assertTrue(second["replayed"])
            self.assertEqual(second["signature"], first["signature"])
            self.assertEqual(service.replay_count, 1)
            events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["status"] for event in events], ["valid", "replayed"])
            serialized_audit = audit.read_text(encoding="utf-8")
            self.assertNotIn("local-secret", serialized_audit)
            self.assertNotIn("episodes", serialized_audit)

    def test_authentication_rate_and_replay_capacity_fail_closed(self) -> None:
        service = LocalEvaluatorService(auth_token="secret", max_requests=1, window_seconds=60)
        raw = json.dumps(_submission()).encode("utf-8")
        with self.assertRaises(EvaluatorAuthenticationError):
            service.submit(raw, authorization="Bearer wrong")
        service.submit(raw, authorization="Bearer secret")
        with self.assertRaises(EvaluatorRateLimitError):
            service.submit(raw, authorization="Bearer secret")

        now = [0.0]
        bounded = LocalEvaluatorService(
            auth_token="secret",
            max_requests=3,
            window_seconds=1,
            max_replay_entries=1,
            clock=lambda: now[0],
        )
        bounded.submit(raw, authorization="Bearer secret")
        changed = _submission()
        changed["policy"]["seed"] = 8
        with self.assertRaises(EvaluatorServiceError):
            bounded.submit(json.dumps(changed).encode("utf-8"), authorization="Bearer secret")

    def test_invalid_json_is_signed_and_tampering_is_rejected(self) -> None:
        service = LocalEvaluatorService(auth_token="secret")
        invalid = service.submit(b"not-json", authorization="Bearer secret")
        self.assertEqual(invalid["report"]["status"], "invalid")
        verify_signed_result(invalid, service.public_key_bytes)
        tampered = copy.deepcopy(invalid)
        tampered["report"]["status"] = "valid"
        with self.assertRaises(EvaluatorServiceError):
            verify_signed_result(tampered, service.public_key_bytes)


if __name__ == "__main__":
    unittest.main()
