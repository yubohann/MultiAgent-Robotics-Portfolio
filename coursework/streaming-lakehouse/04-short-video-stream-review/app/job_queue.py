"""Durable local queue backed by SQLite."""

import json
import uuid
from typing import Any

from .storage import connect, init_db, now_iso

TERMINAL_STATUSES = {"done", "failed"}


def _decode_job(row: Any) -> dict:
    """Decode one job row and restore the JSON payload to a Python dict."""
    item = dict(row)
    item["payload"] = json.loads(item["payload"])
    return item


def enqueue_job(
    kind: str,
    payload: dict,
    *,
    job_id: str | None = None,
    max_attempts: int = 1,
) -> dict:
    """Persist a local job; this mirrors a Kafka task event in the teaching runtime."""
    init_db()
    job_id = job_id or uuid.uuid4().hex
    timestamp = now_iso()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, kind, payload, status, attempts, max_attempts,
                worker_id, last_error, created_at, updated_at
            )
            VALUES (?, ?, ?, 'queued', 0, ?, NULL, '', ?, ?)
            """,
            (job_id, kind, json.dumps(payload, ensure_ascii=False), max_attempts, timestamp, timestamp),
        )
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _decode_job(row)


def claim_next_job(worker_id: str) -> dict | None:
    """Claim the oldest queued job for one local worker."""
    init_db()
    with connect() as connection:
        row = connection.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        timestamp = now_iso()
        updated = connection.execute(
            """
            UPDATE jobs
            SET status = 'running',
                attempts = attempts + 1,
                worker_id = ?,
                updated_at = ?
            WHERE id = ? AND status = 'queued'
            """,
            (worker_id, timestamp, row["id"]),
        )
        if updated.rowcount != 1:
            return None
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
    return _decode_job(row)


def complete_job(job_id: str) -> None:
    """Mark a job as done after the pipeline has persisted the final video result."""
    init_db()
    with connect() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'done', last_error = '', updated_at = ?
            WHERE id = ?
            """,
            (now_iso(), job_id),
        )


def fail_job(job_id: str, error: str) -> None:
    """Record a worker failure and optionally requeue the job."""
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT attempts, max_attempts FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return
        status = "queued" if row["attempts"] < row["max_attempts"] else "failed"
        connection.execute(
            """
            UPDATE jobs
            SET status = ?, last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error, now_iso(), job_id),
        )


def requeue_interrupted_jobs() -> int:
    """Move jobs left running by a previous process back to queued."""
    init_db()
    with connect() as connection:
        updated = connection.execute(
            """
            UPDATE jobs
            SET status = 'queued',
                worker_id = NULL,
                last_error = 'worker interrupted before completion',
                updated_at = ?
            WHERE status = 'running'
            """,
            (now_iso(),),
        )
    return updated.rowcount


def has_active_jobs() -> bool:
    """Return whether any queued or running work is still in flight."""
    init_db()
    with connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()["count"]
    return count > 0


def job_stats() -> dict:
    """Aggregate job counts for `/api/health` and the frontend running badge."""
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
        ).fetchall()
    by_status = {row["status"]: row["count"] for row in rows}
    return {
        "queued": by_status.get("queued", 0),
        "running": by_status.get("running", 0),
        "done": by_status.get("done", 0),
        "failed": by_status.get("failed", 0),
    }
