# Modern Lakehouse Order Analytics

Author: REDACTED  
Student ID: REDACTED  
Class No.: REDACTED

This lab runs a cloud-native stream-batch lakehouse stack with Docker Compose and Dev Containers.

## Components

- Kafka: real-time order stream topic `ecommerce_orders_demo000000`.
- MinIO: S3-compatible object storage bucket `paimon-data-demo000000`.
- Flink: streaming ingestion and aggregation job `DualStreamJob_demo000000`.
- Dinky: optional Flink SQL console.
- Paimon: ODS and DWS lakehouse tables.
- Spark: offline analysis and stream-batch consistency verification.

## Main Files

- `docker-compose.yml`: local big data cluster.
- `order_stream_producer.py`: e-commerce order stream generator.
- `dinky_dual_stream_job.sql`: Dinky SQL version of the streaming job.
- `lakehouse_batch_analysis.py`: Spark batch verification.
- `flink-dual-stream-job/`: Java implementation of the Flink job.
- `lib/flink-connectors/`: Flink, Kafka, Paimon, S3, and Hadoop connector JARs.
- `docker/spark-local/`: local Spark 3.3.2 image build context.
- `SCREENSHOT_GUIDE.md`: screenshot checklist and terminal commands.

## Start

```bash
cd "/path/to/stream-batch-lakehouse-ai-portfolio/01-modern-lakehouse"
sudo service docker start
docker compose up -d --scale spark-worker=3
```

## Stop

```bash
docker compose down
sudo service docker stop
```
