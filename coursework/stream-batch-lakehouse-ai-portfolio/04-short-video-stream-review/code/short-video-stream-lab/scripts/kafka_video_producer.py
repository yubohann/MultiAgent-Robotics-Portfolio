"""Kafka producer for video-ingest events.

这个脚本把本地 demo 视频描述成事件发送到 Kafka，模拟短视频平台中的“视频进入流”。
真正的视频文件仍在本地路径中，Kafka 只承载 metadata；工业系统通常会放对象存储 URL。
"""

import json
import os
from pathlib import Path
import sys
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo_assets import ensure_demo_videos  # noqa: E402
from app.config import INCOMING_DIR  # noqa: E402


BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
# 报告加分项建议把该环境变量改成 short_video_ingest_学号。
TOPIC = os.getenv("SHORT_VIDEO_INGEST_TOPIC", "short_video_ingest")


def main() -> None:
    """Send one ingest event for each deterministic demo video."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            api_version_auto_timeout_ms=5000,
            request_timeout_ms=10000,
        )
    except NoBrokersAvailable as exc:
        raise SystemExit(
            f"Kafka broker is not reachable at {BOOTSTRAP}. "
            "Start Kafka and create the short_video_ingest topic first."
        ) from exc
    review_video = INCOMING_DIR / "upload_[REDACTED]_yubohan_drone_review.mp4"
    if os.getenv("SHORT_VIDEO_KAFKA_SINGLE_DRONE", "0") == "1" and review_video.exists():
        items = [
            {
                "title": "[REDACTED]_Bohan Yu_Kafka_无人机巡检短视频",
                "path": str(review_video),
                "source": "kafka-drone-public-video",
            }
        ]
    else:
        items = ensure_demo_videos(overwrite=False)

    for item in items:
        # event_time 使用毫秒时间戳，模拟日志系统和流处理系统常见的事件时间字段。
        event = {
            "title": item["title"],
            "path": str(Path(item["path"]).resolve()),
            "source": item["source"],
            "event_time": int(time.time() * 1000),
        }
        producer.send(TOPIC, value=event)
        print(f"sent to {TOPIC}: {event}")
    # flush 保证脚本退出前消息已经发往 broker，便于学生立刻截图验证。
    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()
