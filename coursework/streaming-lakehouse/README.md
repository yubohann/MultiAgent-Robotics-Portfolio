# Streaming Lakehouse Labs

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="01-modern-lakehouse/screenshots/07-flink-job.png"><img src="01-modern-lakehouse/screenshots/07-flink-job.png" alt="Flink dual stream job running on the lakehouse stack" width="49%" /></a>
  <a href="03-ai-recommender/screenshots/03-5-fusion-final-recommendations.png"><img src="03-ai-recommender/screenshots/03-5-fusion-final-recommendations.png" alt="Fused recommendations from the real-time recommender" width="49%" /></a>
</p>

**Four big data labs covering a modern lakehouse, streaming reliability challenges, a real-time recommender and a short-video review pipeline.**

The lakehouse lab runs Kafka, Flink, Dinky, Paimon, MinIO and Spark from Docker Compose and follows orders from stream ingest to lake storage to batch verification. The challenges lab walks through data skew, small files, state growth, late data, duplicate handling, schema evolution and object store access, each with before-and-after evidence. The recommender lab combines Flink fast features with a DeepFM model and a fusion service. The short-video lab runs a FastAPI review service with a local vision model and Kafka ingest.

**Status.** All four labs complete, with instructions, reports, screenshots, evidence logs and code in each directory.

## My Role

Built and ran the four labs. The lakehouse lab: the Docker Compose stack, Flink and Dinky jobs, Paimon and MinIO storage, and Spark batch verification (`01-modern-lakehouse/`). The challenges lab: seven streaming failure reproductions with before-and-after evidence (`02-streaming-challenges/`). The recommender lab: the Flink feature job, service layer and fusion output (`03-ai-recommender/`). The short-video lab: the FastAPI review service, local VLM model comparison, Kafka path and the submission package (`04-short-video-stream-review/`). Reports, screenshots and evidence logs ship in each lab directory.

## Labs

| Lab | Scope | Core technologies |
|---|---|---|
| [Modern lakehouse](01-modern-lakehouse/) | Kafka ingestion, Flink and Dinky processing, MinIO and Paimon storage and Spark analysis | Kafka, Flink, Dinky, MinIO, Paimon, Spark |
| [Streaming challenges](02-streaming-challenges/) | Data skew, small files, state growth, late data, duplicate handling, schema evolution and object store access | Kafka, Flink, Paimon, MinIO |
| [AI recommender](03-ai-recommender/) | Real-time behaviour features and recommendation services | Flink, Python services, model APIs |
| [Short-video review](04-short-video-stream-review/) | Local video review and model comparison pipeline | FastAPI, Kafka, local VLM interface |

Each lab keeps its original instructions, reports, screenshots and code, and each lab README carries its own run commands.

## Run Entry Points

```bash
cd 01-modern-lakehouse
sudo service docker start
docker compose up -d --scale spark-worker=3
```

The recommender services and the short-video lab run as Python services, and their README files carry the remaining commands.

## Documentation

- [01 modern lakehouse](01-modern-lakehouse/README.md), stack setup, run commands and the lab report.
- [02 streaming challenges](02-streaming-challenges/README.md), seven challenge folders with evidence tables and code.
- [03 AI recommender](03-ai-recommender/README.md), service layout and verification steps.
- [04 short video stream review](04-short-video-stream-review/README.md), submission package and deterministic replay notes.

The lab materials are kept as course submissions, with third-party engines under their own terms.

*Bohan Yu, big data course labs.*
