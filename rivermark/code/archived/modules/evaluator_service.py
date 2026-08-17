"""Local authenticated evaluator-service prototype.

This module deliberately stops at a local service boundary.  It does not open
ports or provide a blind-test leaderboard.  It adds the controls that must be
tested before an independently operated service exists: bearer-token
authentication, a bounded request window, an in-memory replay store, a
path-free append-only audit log, and detached Ed25519 result signatures.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping

from .evaluator import (
    MAX_SUBMISSION_BYTES,
    RESULT_SCHEMA,
    SubmissionIssue,
    SubmissionReport,
    evaluate_submission,
)


SERVICE_RESULT_SCHEMA = "org.rivermark.benchmark.local-evaluator-result.v1"
_AUDIT_SCHEMA = "org.rivermark.benchmark.local-evaluator-audit.v1"
class EvaluatorServiceError(ValueError):
    """Raised when a local evaluator-service request cannot be accepted."""


class EvaluatorAuthenticationError(EvaluatorServiceError):
    """Raised when the bearer token is absent or incorrect."""


class EvaluatorRateLimitError(EvaluatorServiceError):
    """Raised when the authenticated request window is exhausted."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _crypto() -> tuple[Any, Any, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - optional service dependency
        raise EvaluatorServiceError(
            "signed local evaluator results require the optional 'supply-chain' extra (cryptography)"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, serialization


def _private_signing_key(value: Any | None) -> Any:
    private_key_type, _, _ = _crypto()
    if value is None:
        return private_key_type.generate()
    if isinstance(value, bytes):
        if len(value) != 32:
            raise EvaluatorServiceError("Ed25519 private key bytes must be exactly 32 bytes")
        return private_key_type.from_private_bytes(value)
    if not isinstance(value, private_key_type):
        raise EvaluatorServiceError("signing_key must be an Ed25519PrivateKey or 32 raw bytes")
    return value


def _public_key_bytes(private_key: Any) -> bytes:
    _, _, serialization = _crypto()
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _invalid_report(message: str) -> SubmissionReport:
    return SubmissionReport(
        schema=RESULT_SCHEMA,
        status="invalid",
        dataset_version=None,
        dataset_index_sha256=None,
        split=None,
        evaluator_id=None,
        evaluator_version=None,
        evaluator_sha256=None,
        metric_version=None,
        method_id=None,
        code_revision=None,
        checkpoint_sha256=None,
        seed=None,
        episode_count=0,
        scores=(),
        issues=(SubmissionIssue("input", "$", message),),
        submission_sha256=None,
    )


class LocalEvaluatorService:
    """Bounded in-process evaluator with explicit non-deployment semantics."""

    def __init__(
        self,
        *,
        auth_token: str,
        expected_dataset_version: str | None = None,
        expected_split: str | None = None,
        expected_dataset_index_sha256: str | None = None,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        max_replay_entries: int = 4096,
        signing_key: Any | None = None,
        audit_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(auth_token, str) or not auth_token:
            raise EvaluatorServiceError("auth_token must be a non-empty string")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests <= 0:
            raise EvaluatorServiceError("max_requests must be a positive integer")
        if isinstance(window_seconds, bool) or not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
            raise EvaluatorServiceError("window_seconds must be a positive number")
        if isinstance(max_replay_entries, bool) or not isinstance(max_replay_entries, int) or max_replay_entries <= 0:
            raise EvaluatorServiceError("max_replay_entries must be a positive integer")
        if expected_dataset_index_sha256 is not None and (
            not isinstance(expected_dataset_index_sha256, str)
            or len(expected_dataset_index_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_dataset_index_sha256)
        ):
            raise EvaluatorServiceError("expected_dataset_index_sha256 must be lowercase SHA-256")
        self._auth_token_digest = hashlib.sha256(auth_token.encode("utf-8")).digest()
        self._expected_dataset_version = expected_dataset_version
        self._expected_split = expected_split
        self._expected_dataset_index_sha256 = expected_dataset_index_sha256
        self._max_requests = max_requests
        self._window_seconds = float(window_seconds)
        self._max_replay_entries = max_replay_entries
        self._clock = clock
        self._wall_clock = wall_clock
        self._request_times: deque[float] = deque()
        self._replay: dict[str, dict[str, Any]] = {}
        self._audit_path = audit_path
        self._signing_key = _private_signing_key(signing_key)
        self._public_key = _public_key_bytes(self._signing_key)
        self._key_id = hashlib.sha256(self._public_key).hexdigest()

    @property
    def public_key_bytes(self) -> bytes:
        """Return raw Ed25519 public bytes for detached verification."""

        return self._public_key

    @property
    def replay_count(self) -> int:
        return len(self._replay)

    def _audit(self, *, status: str, submission_sha256: str | None = None, report_sha256: str | None = None, replayed: bool = False) -> None:
        if self._audit_path is None:
            return
        event = {
            "schema": _AUDIT_SCHEMA,
            "status": status,
            "submission_sha256": submission_sha256,
            "report_sha256": report_sha256,
            "replayed": replayed,
            "wall_time_s": float(self._wall_clock()),
        }
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json_bytes(event).decode("utf-8"))

    def _authenticate(self, authorization: str | None) -> None:
        expected = "Bearer "
        supplied = authorization[len(expected):] if isinstance(authorization, str) and authorization.startswith(expected) else ""
        supplied_digest = hashlib.sha256(supplied.encode("utf-8")).digest()
        if not hmac.compare_digest(supplied_digest, self._auth_token_digest):
            self._audit(status="authentication_rejected")
            raise EvaluatorAuthenticationError("invalid evaluator bearer token")

    def _consume_rate_limit(self) -> None:
        now = float(self._clock())
        cutoff = now - self._window_seconds
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        if len(self._request_times) >= self._max_requests:
            self._audit(status="rate_limited")
            raise EvaluatorRateLimitError("local evaluator request rate limit exceeded")
        self._request_times.append(now)

    def submit(self, raw_submission: bytes, *, authorization: str | None) -> dict[str, Any]:
        """Evaluate one UTF-8 JSON request and return a signed local result."""

        self._authenticate(authorization)
        self._consume_rate_limit()
        if not isinstance(raw_submission, bytes):
            self._audit(status="invalid_input")
            raise EvaluatorServiceError("raw_submission must be bytes")
        if len(raw_submission) > MAX_SUBMISSION_BYTES:
            self._audit(status="resource_rejected")
            raise EvaluatorServiceError(f"submission exceeds the {MAX_SUBMISSION_BYTES} byte limit")
        submission_sha256 = hashlib.sha256(raw_submission).hexdigest()
        existing = self._replay.get(submission_sha256)
        if existing is not None:
            replay = copy.deepcopy(existing)
            replay["replayed"] = True
            self._audit(status="replayed", submission_sha256=submission_sha256, report_sha256=existing["report_sha256"], replayed=True)
            return replay
        if len(self._replay) >= self._max_replay_entries:
            self._audit(status="replay_store_full", submission_sha256=submission_sha256)
            raise EvaluatorServiceError("local evaluator replay store capacity reached")
        try:
            payload = json.loads(raw_submission.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            report = _invalid_report(str(exc))
        else:
            report = evaluate_submission(
                payload,
                expected_dataset_version=self._expected_dataset_version,
                expected_split=self._expected_split,
                expected_dataset_index_sha256=self._expected_dataset_index_sha256,
                submission_sha256=submission_sha256,
            )
        report_payload = report.as_dict()
        report_sha256 = hashlib.sha256(_canonical_json_bytes(report_payload)).hexdigest()
        signed_payload = {
            "schema": SERVICE_RESULT_SCHEMA,
            "submission_sha256": submission_sha256,
            "report_sha256": report_sha256,
            "report": report_payload,
        }
        signature = base64.b64encode(self._signing_key.sign(_canonical_json_bytes(signed_payload))).decode("ascii")
        result = {
            **signed_payload,
            "key_id": self._key_id,
            "signature_algorithm": "ed25519",
            "signature": signature,
            "replayed": False,
        }
        self._replay[submission_sha256] = copy.deepcopy(result)
        self._audit(status=report.status, submission_sha256=submission_sha256, report_sha256=report_sha256)
        return result


def verify_signed_result(result: Mapping[str, Any], public_key_bytes: bytes) -> None:
    """Raise if a local service result is malformed or its signature fails."""

    _, public_key_type, _ = _crypto()
    try:
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:  # pragma: no cover - guarded by _crypto
        raise EvaluatorServiceError("cryptography is unavailable") from exc
    required = {"schema", "submission_sha256", "report_sha256", "report", "key_id", "signature_algorithm", "signature"}
    if set(result) - (required | {"replayed"}) or not required.issubset(result):
        raise EvaluatorServiceError("signed evaluator result has an invalid field set")
    if result.get("schema") != SERVICE_RESULT_SCHEMA or result.get("signature_algorithm") != "ed25519":
        raise EvaluatorServiceError("signed evaluator result has an invalid schema or algorithm")
    if not isinstance(public_key_bytes, bytes) or len(public_key_bytes) != 32:
        raise EvaluatorServiceError("Ed25519 public key bytes must be exactly 32 bytes")
    expected_key_id = hashlib.sha256(public_key_bytes).hexdigest()
    if result.get("key_id") != expected_key_id:
        raise EvaluatorServiceError("signed evaluator result key binding mismatch")
    report = result.get("report")
    if not isinstance(report, Mapping):
        raise EvaluatorServiceError("signed evaluator result report must be an object")
    report_hash = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    if result.get("report_sha256") != report_hash:
        raise EvaluatorServiceError("signed evaluator result report hash mismatch")
    try:
        signature = base64.b64decode(result["signature"], validate=True)
        public_key_type.from_public_bytes(public_key_bytes).verify(
            signature,
            _canonical_json_bytes(
                {
                    "schema": result["schema"],
                    "submission_sha256": result["submission_sha256"],
                    "report_sha256": result["report_sha256"],
                    "report": report,
                }
            ),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise EvaluatorServiceError("signed evaluator result signature verification failed") from exc


__all__ = [
    "SERVICE_RESULT_SCHEMA",
    "EvaluatorServiceError",
    "EvaluatorAuthenticationError",
    "EvaluatorRateLimitError",
    "LocalEvaluatorService",
    "verify_signed_result",
]
