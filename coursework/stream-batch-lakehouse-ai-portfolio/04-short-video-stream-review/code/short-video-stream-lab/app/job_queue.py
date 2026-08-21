"""Durable local queue backed by SQLite.

工业短视频平台通常会使用 Kafka、Pulsar 或 RabbitMQ 把“上传服务”和“审核服务”解耦。
为了让学生电脑不用额外启动消息队列，本实验用 SQLite `jobs` 表实现同样的核心概念：
任务入队、worker 领取、失败重试、完成归档和健康统计。
"""

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
    """Persist a local job; this mirrors a Kafka task event in the teaching runtime.

    `kind` 表示任务类型，`payload` 是 worker 处理所需的数据。
    这里使用可选 `job_id`，是为了让同一个视频的完成任务有稳定 id，便于观察和排错。
    """
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
    """Claim the oldest queued job for one local worker.

    领取任务分两步完成：先找最早的 queued 记录，再带状态条件更新为 running。
    `WHERE id = ? AND status = 'queued'` 能避免未来多 worker 时重复领取同一任务。
    """
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
    """Record a worker failure and optionally requeue the job.

    本实验默认 max_attempts=1，失败会进入 failed；如果教师扩展重试实验，
    可以把 max_attempts 调大，此函数就会在未超过次数时重新排队。
    """
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
    """Move jobs left running by a previous process back to queued.

    如果服务在处理视频时被关闭，任务可能停留在 running。
    服务下次启动时把它们恢复为 queued，模拟工业消费者的“崩溃恢复”。
    """
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
