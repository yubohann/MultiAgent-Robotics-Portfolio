# Experiment 7 Run And Screenshot Guide

Use a WSL terminal, not PowerShell, because Docker is available inside WSL in this environment.

## 1. Enter The Evidence Folder

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/02-streaming-challenges/07-minio-access-bottleneck/evidence
```

## 2. Check The Environment

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
docker exec bigdata-flink-jm /opt/flink/bin/flink list
python3 -m py_compile generate_partition_data.py exp7_query.py
```

Expected:

- `bigdata-kafka`, `bigdata-flink-jm`, `bigdata-flink-tm`, `bigdata-spark-master`, and `bigdata-minio` are running.
- Flink has no old running jobs, or only jobs you intentionally keep.
- Python compile prints no error.

## 3. Recreate The Kafka Topic And Generate 31,000 Records

```bash
docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --delete \
  --topic partition_demo_demo000000 \
  --bootstrap-server localhost:9092 || true

sleep 3

docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists \
  --topic partition_demo_demo000000 \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

STUDENT_NAME=REDACTED python3 generate_partition_data.py | tee partition_data_run.log
```

Expected:

- The producer prints `Student ID: demo000000`.
- It prints `Day 1/31` through `Day 31/31`.
- It prints `Total records generated: 31000`.

## 4. Submit The Flink Jobs

```bash
docker cp partition_submit.sql bigdata-flink-jm:/tmp/partition_submit.sql
docker exec bigdata-flink-jm /opt/flink/bin/sql-client.sh -f /tmp/partition_submit.sql
```

Expected:

- SQL client submits two insert jobs:
  one for `sales_unpartitioned_demo000000`,
  one for `sales_partitioned_demo000000`.

Then wait 60 to 120 seconds and list jobs:

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink list
```

Cancel the two experiment-7 insert jobs shown by `flink list`:

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink cancel <sales_unpartitioned_job_id>
docker exec bigdata-flink-jm /opt/flink/bin/flink cancel <sales_partitioned_job_id>
```

Do not type the angle brackets. Replace them with the real Job IDs.

## 5. Verify In Spark And Measure Query Time

```bash
docker cp exp7_query.py bigdata-spark-master:/tmp/exp7_query.py
docker exec bigdata-spark-master /opt/spark/bin/spark-submit \
  --master local[*] \
  --jars /root/.ivy2/jars/org.apache.paimon_paimon-spark-3.3-0.8.0.jar,/root/.ivy2/jars/org.apache.paimon_paimon-bundle-0.8.0.jar,/root/.ivy2/jars/org.apache.hadoop_hadoop-aws-3.3.2.jar,/root/.ivy2/jars/com.amazonaws_aws-java-sdk-bundle-1.12.367.jar \
  /tmp/exp7_query.py | tee exp7_spark_results.txt
```

Expected:

- Both tables have `31000` records.
- The `dt = '2023-10-15'` query returns `Row(cnt=1000)`.
- Unpartitioned query has a larger elapsed time than the partitioned query.
- The partitioned explain output shows `PaimonScan` and `PushedFilters`.

## 6. Screenshot 7-1

Capture the terminal area that shows:

- `Experiment 7 MinIO Access Bottleneck / Partition Pruning - Student ID demo000000`
- `Table row count validation`
- `sales_unpartitioned_demo000000` and `sales_partitioned_demo000000` both have `31000`
- `7-1 slow query on unpartitioned table`
- `SELECT COUNT(*) ... sales_unpartitioned_demo000000 WHERE dt = '2023-10-15'`
- `Elapsed seconds: ...`
- `Row(cnt=1000)`

Save as:

```text
../screenshots/7-1-unpartitioned-query.png
```

## 7. Screenshot 7-2

Capture the terminal area that shows:

- `7-2 fast query on partitioned table`
- `SELECT COUNT(*) ... sales_partitioned_demo000000 WHERE dt = '2023-10-15'`
- `Elapsed seconds: ...`
- `Row(cnt=1000)`
- `7-2 EXPLAIN partitioned table`
- `PaimonScan: [sales_partitioned_demo000000]`
- `PushedFilters: [IsNotNull(dt), Equal(dt, 2023-10-15)]`

Save as:

```text
../screenshots/7-2-partitioned-query.png
```
