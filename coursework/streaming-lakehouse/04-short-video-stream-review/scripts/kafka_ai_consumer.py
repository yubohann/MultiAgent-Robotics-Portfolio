"""Kafka consumer that runs the same review pipeline for ingest events."""

import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401 -- side-effect import: allow direct execution from scripts/
from app.pipeline import ShortVideoPipeline
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
# Environment variables allow student-suffixed topic names.
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
            # earliest lets a late consumer still read the sample messages.
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
        # The Kafka payload carries only a path and title and reuses the main pipeline.
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
