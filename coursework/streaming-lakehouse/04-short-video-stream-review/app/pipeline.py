"""End-to-end pipeline for ingestion, understanding, moderation, and publishing."""

import re
import shutil
import time
from pathlib import Path

from .config import MEDIA_DIR, ensure_directories
from .ffmpeg_tools import FFmpegError, create_thumbnail
from .storage import add_event, init_db, upsert_video
from .understanding_service import MultimodalUnderstandingService
from .video_understanding import moderate_analysis


def _slug(value: str) -> str:
    """Convert a filename stem to a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "video"


def _video_id(title: str) -> str:
    """Create a unique, sortable id from the ingest clock and title."""
    return f"{time.time_ns()}-{_slug(title)[:24]}"


class ShortVideoPipeline:
    """End-to-end short video ingestion, understanding, moderation, and publishing."""

    def __init__(self) -> None:
        """Initialize directories, database schema, and the understanding service."""
        ensure_directories()
        init_db()
        self.model = MultimodalUnderstandingService()

    def process_video(
        self,
        video_path: Path,
        *,
        title: str | None = None,
        source: str = "local",
        simulate_stream: bool = True,
    ) -> dict:
        """Process one video synchronously; scripts use this for one-shot verification."""
        pending_record = self.ingest_video(video_path, title=title, source=source)
        return self.complete_video(pending_record, simulate_stream=simulate_stream)

    def ingest_video(
        self,
        video_path: Path,
        *,
        title: str | None = None,
        source: str = "local",
    ) -> dict:
        """Persist the playable video immediately, before expensive model work starts."""
        video_path = video_path.expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        title = title or video_path.stem
        video_id = _video_id(title)
        media_name = f"{video_id}-{_slug(video_path.stem)}{video_path.suffix.lower() or '.mp4'}"
        media_path = MEDIA_DIR / media_name

        # Write the event before copying files so the ingest timestamp stays accurate.
        add_event(
            video_id,
            "ingest",
            "收到短视频，开始写入媒体区",
            {"title": title, "source": source, "path": str(video_path)},
        )
        shutil.copy2(video_path, media_path)
        add_event(
            video_id,
            "stream",
            "开始按帧抽样，模拟视频流进入实时处理链路",
            {"media_file": media_name},
        )
        record = {
            "id": video_id,
            "title": title,
            "source": source,
            "original_path": str(video_path),
            "media_file": media_name,
            "thumbnail_file": "",
            "status": "processing",
            "risk_score": 0,
            "caption": "",
            "tags": [],
            "reasons": [],
            "metrics": {
                "duration_sec": 0,
                "sampled_frames": 0,
                "brightness": {"avg": 0},
                "motion": {"avg": 0},
                "model": {
                    "selected_id": "",
                    "selected_name": "等待后台理解",
                    "backend": "pending",
                    "fallback_reason": "",
                },
            },
        }
        upsert_video(record)
        add_event(video_id, "queued", "视频已进入页面，等待后台理解和打标签", {"status": "processing"})
        return record

    def complete_video(
        self,
        record: dict,
        *,
        simulate_stream: bool = True,
    ) -> dict:
        """Run understanding, moderation, tagging, and final publication update."""
        video_id = record["id"]
        title = record["title"]
        media_name = record["media_file"]
        media_path = MEDIA_DIR / media_name
        thumbnail_name = f"{video_id}-thumb.jpg"
        thumbnail_path = MEDIA_DIR / thumbnail_name

        analysis = self.model.analyze(
            media_path,
            title=title,
            video_id=video_id,
            emit_event=add_event,
            simulate_delay_sec=0.03 if simulate_stream else 0.0,
        )
        add_event(
            video_id,
            "understanding",
            "完成视频理解，生成摘要、指标和候选标签",
            {
                "caption": analysis["caption"],
                "tags": analysis["tags"],
                "sampled_frames": analysis["metrics"]["sampled_frames"],
                "model": analysis.get("model", {}),
            },
        )
        moderation = moderate_analysis(analysis, title)
        add_event(
            video_id,
            "moderation",
            "完成自动审核策略判定",
            {
                "status": moderation["status"],
                "risk_score": moderation["risk_score"],
                "reasons": moderation["reasons"],
            },
        )
        try:
            # The thumbnail is presentation-only and never blocks publication.
            create_thumbnail(media_path, thumbnail_path)
        except FFmpegError as exc:
            add_event(video_id, "thumbnail", "封面抽取失败，不影响发布", {"error": str(exc)})
            thumbnail_name = ""

        record = {
            "id": video_id,
            "title": title,
            "source": record["source"],
            "original_path": record["original_path"],
            "media_file": media_name,
            "thumbnail_file": thumbnail_name,
            "status": moderation["status"],
            "risk_score": moderation["risk_score"],
            "caption": analysis["caption"],
            "tags": analysis["tags"],
            "reasons": moderation["reasons"],
            "metrics": analysis["metrics"],
        }
        upsert_video(record)
        publish_message = {
            "published": "审核通过，视频已发布到 Demo 信息流",
            "review": "视频进入人工复核队列，暂不公开发布",
            "rejected": "视频被策略拒绝，禁止发布",
        }[moderation["status"]]
        add_event(video_id, "publish", publish_message, {"status": moderation["status"]})
        return record
