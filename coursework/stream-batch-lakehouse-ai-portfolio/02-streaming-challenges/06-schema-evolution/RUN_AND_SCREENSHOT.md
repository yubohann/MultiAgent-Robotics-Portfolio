# Experiment 6 Run And Screenshot Guide

Use a WSL terminal, not PowerShell, because Docker is available inside WSL in this environment.

## 1. Enter The Evidence Folder

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/02-streaming-challenges/06-schema-evolution/evidence
```

## 2. Check The Environment

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
docker exec bigdata-flink-jm /opt/flink/bin/flink list
python3 -m py_compile exp6_query.py
```

Expected:

- `bigdata-flink-jm`, `bigdata-spark-master`, and `bigdata-minio` are running.
- Flink has no old running jobs, or only jobs you intentionally keep.
- Python compile prints no error.

## 3. Run The Initial Schema And Error Demo For Screenshot 6-1

```bash
docker cp schema_initial_error.sql bigdata-flink-jm:/tmp/schema_initial_error.sql
docker exec bigdata-flink-jm /opt/flink/bin/sql-client.sh -f /tmp/schema_initial_error.sql | tee exp6_initial_output.txt
```

Expected:

- `DESCRIBE user_profile_demo000000` shows only:
  `user_id`, `username`, `email`, `create_time`.
- The final query fails with:
  `Column 'age' not found in any table`.

## 4. Screenshot 6-1

Capture the terminal area that shows:

- `DESCRIBE user_profile_demo000000`
- The initial schema without `age`
- `SELECT age FROM user_profile_demo000000`
- `Column 'age' not found in any table`

Save as:

```text
../screenshots/6-1-initial-schema-error.png
```

## 5. Run The Full Schema Evolution For Screenshot 6-2

```bash
docker cp schema_evolution_run.sql bigdata-flink-jm:/tmp/schema_evolution_run.sql
docker exec bigdata-flink-jm /opt/flink/bin/sql-client.sh -f /tmp/schema_evolution_run.sql | tee exp6_evolution_output.txt
```

Expected:

- Final `DESCRIBE user_profile_demo000000` shows `age BIGINT`.
- `address` / `location` no longer exist because `location` was dropped.

## 6. Verify Old And New Rows In Spark

```bash
docker cp exp6_query.py bigdata-spark-master:/tmp/exp6_query.py
docker exec bigdata-spark-master /opt/spark/bin/spark-submit \
  --master local[*] \
  --jars /root/.ivy2/jars/org.apache.paimon_paimon-spark-3.3-0.8.0.jar,/root/.ivy2/jars/org.apache.paimon_paimon-bundle-0.8.0.jar,/root/.ivy2/jars/org.apache.hadoop_hadoop-aws-3.3.2.jar,/root/.ivy2/jars/com.amazonaws_aws-java-sdk-bundle-1.12.367.jar \
  /tmp/exp6_query.py | tee exp6_spark_results.txt
```

Expected:

- `DESCRIBE` shows `age bigint`.
- Old rows `1`, `2`, `3` have `age = null`.
- New rows `1001`, `1002` have `age = 25` and `30`.
- Count table shows `Old data = 3`, `New data = 2`.

## 7. Screenshot 6-2

Capture the terminal area from the Spark verification that shows:

- `6-2 DESCRIBE user_profile_demo000000 after schema evolution`
- `age bigint`
- `6-2 old rows keep age=NULL, new rows have age values`
- Rows `1`, `2`, `3` with `null`, and rows `1001`, `1002` with `25`, `30`
- `Old data 3` and `New data 2`

Save as:

```text
../screenshots/6-2-schema-evolution-success.png
```
