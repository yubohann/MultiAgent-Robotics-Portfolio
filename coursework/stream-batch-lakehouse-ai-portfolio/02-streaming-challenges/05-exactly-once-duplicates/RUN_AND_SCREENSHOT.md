# Experiment 5 Run And Screenshot Guide

Use a WSL terminal, not PowerShell, because Docker is available inside WSL in this environment.

## 1. Enter The Evidence Folder

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/02-streaming-challenges/05-exactly-once-duplicates/evidence
```

## 2. Check The Environment

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
docker exec bigdata-flink-jm /opt/flink/bin/flink list
python3 -m py_compile duplicate_data_producer.py exp5_query.py
```

Expected:

- `bigdata-kafka`, `bigdata-flink-jm`, `bigdata-flink-tm`, `bigdata-spark-master`, `bigdata-minio` are running.
- Flink has no old running jobs, or only jobs you intentionally keep.
- Python compile prints no error.

## 3. Recreate The Kafka Topic And Produce Duplicate Data

```bash
docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --delete \
  --topic duplicate_data_demo000000 \
  --bootstrap-server localhost:9092 || true

sleep 3

docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists \
  --topic duplicate_data_demo000000 \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

STUDENT_NAME=REDACTED timeout 70s python3 duplicate_data_producer.py | tee duplicate_producer_clean.log
```

Expected:

- The topic is created.
- The producer prints `Student ID: demo000000`.
- The producer prints increasing `new` and `duplicates` counts.

## 4. Submit The Flink Exactly-Once Jobs

```bash
docker cp exactly_once_submit_clean.sql bigdata-flink-jm:/tmp/exactly_once_submit_clean.sql
docker exec bigdata-flink-jm /opt/flink/bin/sql-client.sh -f /tmp/exactly_once_submit_clean.sql
```

Expected:

- SQL client submits two insert jobs.
- You should see two Job IDs, one for `append_table_demo000000`, one for `pk_table_demo000000`.

Then wait 60 to 90 seconds and list jobs:

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink list
```

Cancel the two experiment-5 insert jobs shown by `flink list`:

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink cancel <append_table_job_id>
docker exec bigdata-flink-jm /opt/flink/bin/flink cancel <pk_table_job_id>
```

Do not type the angle brackets. Replace them with the real Job IDs.

## 5. Verify In Spark

```bash
docker cp exp5_query.py bigdata-spark-master:/tmp/exp5_query.py
docker exec bigdata-spark-master /opt/spark/bin/spark-submit \
  --master local[*] \
  --jars /root/.ivy2/jars/org.apache.paimon_paimon-spark-3.3-0.8.0.jar,/root/.ivy2/jars/org.apache.paimon_paimon-bundle-0.8.0.jar,/root/.ivy2/jars/org.apache.hadoop_hadoop-aws-3.3.2.jar,/root/.ivy2/jars/com.amazonaws_aws-java-sdk-bundle-1.12.367.jar \
  /tmp/exp5_query.py | tee exp5_spark_results.txt
```

## 6. Screenshot 5-1

Capture the part of the terminal that shows:

- `Experiment 5 Exactly-Once / Duplicate Data - Student ID demo000000`
- `5-1 / 5-2 comparison`
- `Append Table` row with `dup_count > 0`
- `5-1 append_table_demo000000 duplicate order_id examples`
- Several `order_id` rows with `dup_count > 1`

Save as:

```text
../screenshots/5-1-append-duplicates.png
```

## 7. Screenshot 5-2

Capture the part of the terminal that shows:

- The same comparison table
- `PK Table` row with `dup_count = 0`
- `5-2 pk_table_demo000000 duplicate check, expected empty`
- Empty duplicate result table

Save as:

```text
../screenshots/5-2-pk-dedup.png
```
