"""Kafka consumer that runs the same review pipeline for ingest events.

默认网站使用 SQLite jobs，本脚本用于连接前三章流批一体架构：
从 `short_video_ingest` 读取视频进入事件，处理后把结果写到 `short_video_result`。
主题名可通过环境变量改成带学号的 topic，便于报告原创性检查。
"""

import json
import os
from pathlib import Path
import sys

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import ShortVideoPipeline  # noqa: E402


BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
# 环境变量让同学可以使用 short_video_ingest_学号 这类 topic 名称。
INGEST_TOPIC = os.getenv("SHORT_VIDEO_INGEST_TOPIC", "short_video_ingest")
RESULT_TOPIC = os.getenv("SHORT_VIDEO_RESULT_TOPIC", "short_video_result")
MAX_MESSAGES = int(os.getenv("SHORT_VIDEO_KAFKA_MAX_MESSAGES", "0"))
GROUP_ID = os.getenv("SHORT_VIDEO_KAFKA_GROUP_ID", "short-video-ai-reviewer")


def main() -> None:
    """Consume video events forever and publish moderation results."""
    pipeline = ShortVideoPipeline()
    try:
        consumer = KafkaConsumer(
            INGEST_TOPIC,
            bootstrap_servers=BOOTSTRAP,
            # earliest 便于课堂验收：先发消息再启动消费者也能读到历史样本。
            auto_offset_reset="earliest",
            group_id=GROUP_ID,
            value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
            api_version_auto_timeout_ms=5000,
            session_timeout_ms=10000,
            request_timeout_ms=30000,
        )
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            api_version_auto_timeout_ms=5000,
            request_timeout_ms=10000,
        )
    except NoBrokersAvailable as exc:
        raise SystemExit(
            f"Kafka broker is not reachable at {BOOTSTRAP}. "
            "Start Kafka and create short_video_ingest and short_video_result topics first."
        ) from exc
    print(f"listening on {INGEST_TOPIC}, publishing decisions to {RESULT_TOPIC}")
    processed_count = 0
    for message in consumer:
        payload = message.value
        # Kafka payload 只传路径和标题；完整理解、审核、入库仍复用主 pipeline。
        record = pipeline.process_video(
            Path(payload["path"]),
            title=payload.get("title"),
            source=payload.get("source", "kafka"),
            simulate_stream=True,
        )
        result = {
            "id": record["id"],
            "title": record["title"],
            "status": record["status"],
            "risk_score": record["risk_score"],
            "tags": record["tags"],
        }
        producer.send(RESULT_TOPIC, value=result)
        producer.flush()
        print(f"processed: {result}")
        processed_count += 1
        if MAX_MESSAGES and processed_count >= MAX_MESSAGES:
            print(f"consumer exit after {processed_count} messages")
            break
    consumer.close()
    producer.close()


if __name__ == "__main__":
    main()
