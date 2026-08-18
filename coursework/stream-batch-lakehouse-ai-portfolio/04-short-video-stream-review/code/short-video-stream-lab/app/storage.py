"""SQLite persistence layer for videos, events, settings, and local jobs.

课堂版使用 SQLite 是为了让 Windows/Linux/macOS 学生机无需 Docker 也能完整运行。
这里的四张表分别对应工业系统中的元数据表、审计日志、配置中心和消息队列。
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH, ensure_directories


def now_iso() -> str:
    """Return a UTC timestamp suitable for logs and deterministic ordering."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Any:
    """Open one SQLite transaction and commit it when the caller exits normally.

    所有数据库访问都通过这个上下文管理器完成，保证 row_factory、提交和关闭逻辑一致。
    对学生来说，这相当于一个很小的 Repository/DAO 层。
    """
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create the SQLite tables used by the demo website and pipeline.

    `videos` 保存前端要展示的发布结果；`events` 是可观测事件流；
    `settings` 保存当前选择的模型；`jobs` 模拟 Kafka/Pulsar 中待消费的任务。
    """
    with connect() as connection:
        # videos 是 Demo 网站的查询主表。tags/reasons/metrics 使用 JSON 字符串，
        # 便于单机教学，不需要额外设计多张关系表。
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                original_path TEXT NOT NULL,
                media_file TEXT NOT NULL,
                thumbnail_file TEXT,
                status TEXT NOT NULL,
                risk_score REAL NOT NULL,
                caption TEXT NOT NULL,
                tags TEXT NOT NULL,
                reasons TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # settings 存放少量运行时配置，例如当前后台选择的理解模型。
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # events 记录每个视频进入、抽帧、理解、审核和发布的关键节点，
        # 报告截图中的 `/api/events` 就来自这张表。
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                stage TEXT NOT NULL,
                message TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # jobs 是本实验的本地耐久队列。它保留 queued/running/done/failed 状态，
        # 让课堂版具备和工业消息队列相似的异步处理语义。
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                worker_id TEXT,
                last_error TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def clear_db() -> None:
    """Clear runtime state while keeping tables in place.

    重置演示时只清空业务状态，不删除表结构；settings 不清空，避免学生切换的模型丢失。
    """
    init_db()
    with connect() as connection:
        connection.execute("DELETE FROM events")
        connection.execute("DELETE FROM videos")
        connection.execute("DELETE FROM jobs")


def add_event(
    video_id: str | None,
    stage: str,
    message: str,
    payload: dict | None = None,
) -> None:
    """Append one observable event to the audit log.

    事件日志是学生理解流式处理的入口：同一个视频会依次产生 ingest、queued、
    worker、understanding、moderation、publish 等事件。
    """
    init_db()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO events(video_id, stage, message, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (video_id, stage, message, json.dumps(payload or {}, ensure_ascii=False), now_iso()),
        )


def upsert_video(record: dict) -> None:
    """Insert or update the materialized video record shown by the website.

    上传阶段先写入 `processing` 记录，worker 完成后再用同一个 id 更新为最终状态。
    这就是前端能“先显示视频，再异步补摘要和标签”的关键。
    """
    init_db()
    created_at = record.get("created_at") or now_iso()
    updated_at = now_iso()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO videos (
                id, title, source, original_path, media_file, thumbnail_file,
                status, risk_score, caption, tags, reasons, metrics, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                source=excluded.source,
                original_path=excluded.original_path,
                media_file=excluded.media_file,
                thumbnail_file=excluded.thumbnail_file,
                status=excluded.status,
                risk_score=excluded.risk_score,
                caption=excluded.caption,
                tags=excluded.tags,
                reasons=excluded.reasons,
                metrics=excluded.metrics,
                updated_at=excluded.updated_at
            """,
            (
                record["id"],
                record["title"],
                record["source"],
                record["original_path"],
                record["media_file"],
                record.get("thumbnail_file"),
                record["status"],
                float(record["risk_score"]),
                record["caption"],
                json.dumps(record["tags"], ensure_ascii=False),
                json.dumps(record["reasons"], ensure_ascii=False),
                json.dumps(record["metrics"], ensure_ascii=False),
                created_at,
                updated_at,
            ),
        )


def _decode_record(row: sqlite3.Row) -> dict:
    """Convert one SQLite row into the JSON shape expected by FastAPI."""
    item = dict(row)
    for key in ("tags", "reasons", "metrics"):
        item[key] = json.loads(item[key])
    return item


def list_videos(status: str | None = None) -> list[dict]:
    """Return videos ordered for the feed, optionally filtered by publication status."""
    init_db()
    with connect() as connection:
        if status:
            rows = connection.execute(
                "SELECT * FROM videos WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM videos ORDER BY created_at DESC"
            ).fetchall()
    return [_decode_record(row) for row in rows]


def get_video(video_id: str) -> dict | None:
    """Fetch one video record by id; scripts can use this for targeted inspection."""
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM videos WHERE id = ?",
            (video_id,),
        ).fetchone()
    return _decode_record(row) if row else None


def list_events(limit: int = 80) -> list[dict]:
    """Return recent events for the website timeline and report screenshots."""
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    events = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        events.append(item)
    return events


def stats() -> dict:
    """Aggregate video counts by status for the dashboard cards."""
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM videos GROUP BY status"
        ).fetchall()
        total = connection.execute("SELECT COUNT(*) AS count FROM videos").fetchone()["count"]
    by_status = {row["status"]: row["count"] for row in rows}
    return {
        "total": total,
        "published": by_status.get("published", 0),
        "review": by_status.get("review", 0),
        "rejected": by_status.get("rejected", 0),
        "processing": by_status.get("processing", 0),
    }


def database_path() -> Path:
    """Expose the SQLite path for troubleshooting and documentation."""
    return DB_PATH


def get_setting(key: str, default: Any = None) -> Any:
    """Read a JSON setting with a fallback default."""
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def set_setting(key: str, value: Any) -> None:
    """Persist a JSON setting, replacing the previous value atomically."""
    init_db()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO settings(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (key, json.dumps(value, ensure_ascii=False), now_iso()),
        )
