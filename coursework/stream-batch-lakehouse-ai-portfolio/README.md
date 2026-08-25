# Stream-Batch Lakehouse AI Portfolio

[English](README.md) | [简体中文](README.zh-CN.md)

An English-first engineering portfolio covering a modern lakehouse, streaming reliability challenges, a real-time recommender, and a short-video review workflow.

## Labs

| Lab | Scope | Core technologies |
|---|---|---|
| Modern lakehouse | Kafka ingestion, Flink/Dinky processing, MinIO/Paimon storage, and Spark analysis | Kafka, Flink, Dinky, MinIO, Paimon, Spark |
| Streaming challenges | Data skew, small files, state growth, late data, exactly-once behaviour, schema evolution, and object-store bottlenecks | Kafka, Flink, Paimon, MinIO |
| AI recommender | Real-time behaviour features and recommendation services | Flink, Python services, model APIs |
| Short-video review | Local video review and model comparison pipeline | FastAPI, Kafka, local VLM interface |

Each lab keeps its original instructions, reports, screenshots, and code under its own directory. The Chinese companion is [README.zh-CN.md](README.zh-CN.md).

## Repository Layout

```text
01-modern-lakehouse/
02-streaming-challenges/
03-ai-recommender/
04-short-video-stream-review/
```

The Docker, Maven, Kafka, Flink, Spark, and model-service workflows are lab-specific. Start with the README or report inside the lab you intend to reproduce.
