# Screenshot Guide

Run all terminal commands from WSL2 Ubuntu.

## Enter Project

```bash
cd "/path/to/stream-batch-lakehouse-ai-portfolio/01-modern-lakehouse"
sudo service docker start
docker compose up -d --scale spark-worker=3
```

## Install Python Dependency

If your terminal prompt shows `(base)`, install the Kafka client into the current Conda environment:

```bash
python3 -m pip install --upgrade kafka-python==2.3.1
```

Verify it:

```bash
python3 -c "from kafka import KafkaProducer; import kafka; print(kafka.__version__)"
```

## Prepare Data And Flink Job

Run this before checking Flink jobs or Kafka offsets. Otherwise `flink list` will show `No running jobs`, and Kafka offset lookup may fail because the topic has not been created yet.

```bash
MAX_MESSAGES=30 KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python3 order_stream_producer.py
```

Build the Flink job if the target JAR does not exist:

```bash
cd flink-dual-stream-job
mvn -q -DskipTests package
cd ..
```

Submit the Flink job with container-internal service addresses:

```bash
docker cp flink-dual-stream-job/target/flink-dual-stream-job-1.0-SNAPSHOT.jar bigdata-flink-jm:/tmp/flink-dual-stream-job.jar
docker exec \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:29092 \
  -e S3_ENDPOINT=http://minio:9000 \
  -e STUDENT_ID=demo000000 \
  -e class_no=0 \
  bigdata-flink-jm /opt/flink/bin/flink run -d \
  -c com.edu.bigdata.FlinkDualStream \
  /tmp/flink-dual-stream-job.jar
```

If you accidentally submitted a job with `localhost:9092`, cancel it first:

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink list
docker exec bigdata-flink-jm /opt/flink/bin/flink cancel <job_id>
```

## Quick Status Check

```bash
docker compose ps
docker exec bigdata-flink-jm /opt/flink/bin/flink list
docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
docker exec bigdata-kafka /opt/kafka/bin/kafka-get-offsets.sh --topic ecommerce_orders_demo000000 --bootstrap-server localhost:9092
docker exec bigdata-kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group flink_dual_stream_demo000000
```

Expected signs:

- `DualStreamJob_demo000000 (RUNNING)` appears in Flink.
- Topic `ecommerce_orders_demo000000` appears in Kafka.
- Kafka topic offset is greater than 0.
- Consumer group lag eventually becomes 0.

## Screenshot 1: Connector JARs

```bash
ls -lh lib/flink-connectors
```

Capture the JAR directory. It should show:

- `flink-sql-connector-kafka-3.0.1-1.18.jar`
- `paimon-flink-1.18-0.8.0.jar`
- `flink-s3-fs-hadoop-1.18.0.jar`
- `hadoop-hdfs-client-3.3.4.jar`

## Screenshot 2: Docker Containers

```bash
docker ps
```

Capture Kafka, MinIO, Flink, Dinky, Spark Master, and three Spark Workers in `Up` status.

## Screenshot 3: Flink Web UI

Open:

```text
http://localhost:8081
```

Capture the overview page showing the running job.

Terminal helper:

```bash
curl -s http://localhost:8081/jobs/overview
```

## Screenshot 4: MinIO Web Console

Open:

```text
http://localhost:9001
```

Login:

```text
admin / password123
```

Capture bucket `paimon-data-demo000000`.

## Screenshot 5: Dinky Web UI

Open:

```text
http://localhost:8888
```

Login is usually:

```text
admin / admin
```

## Screenshot 6: Order Producer Output

```bash
MAX_MESSAGES=10 KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python3 order_stream_producer.py
```

Capture topic `ecommerce_orders_demo000000` and order IDs starting with `ORD_demo000000_`.

## Screenshot 7: Flink Job Details

Open:

```text
http://localhost:8081
```

Enter `Jobs -> Running Jobs -> DualStreamJob_demo000000`.

Terminal helpers:

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink list
docker logs --tail 100 bigdata-flink-jm | grep -E "DualStreamJob_demo000000|Completed checkpoint"
docker logs --tail 80 bigdata-flink-tm | grep -E "kafka:29092|ods_orders_demo000000|dws_product_sales_demo000000"
```

Capture job name, `RUNNING` status, records, and completed checkpoint information.

## Screenshot 8: Paimon Data In MinIO

In MinIO, open:

```text
paimon-data-demo000000 / warehouse / default.db
```

Capture:

- `ods_orders_demo000000`
- `dws_product_sales_demo000000`

Terminal helper:

```bash
docker run --rm --network bigdata-network-demo000000 --entrypoint /bin/sh minio/mc -c \
  'mc alias set local http://minio:9000 admin password123 >/dev/null && mc ls -r local/paimon-data-demo000000/warehouse | head -80'
```

## Screenshot 9: Spark Verification

Open Spark UI:

```text
http://localhost:8080
```

Run the offline verification:

```bash
docker exec bigdata-spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.paimon:paimon-spark-3.3:0.8.0,org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.12.367 \
  /opt/spark-apps/lakehouse_batch_analysis.py | tee logs/spark_batch_analysis_demo000000.log
```

Show the result summary:

```bash
grep -E "total_orders=|total_sales=|streaming lakehouse offline verification finished" logs/spark_batch_analysis_demo000000.log
```

## Recommended Screenshot Names

Use `Win + Shift + S`, then save screenshots to `screenshots/`:

```text
screenshots/01-flink-connectors.png
screenshots/02-docker-ps.png
screenshots/03-flink-home.png
screenshots/04-minio-home.png
screenshots/05-dinky-home.png
screenshots/06-order-producer.png
screenshots/07-flink-job.png
screenshots/08-minio-paimon.png
screenshots/09-spark-query.png
```
