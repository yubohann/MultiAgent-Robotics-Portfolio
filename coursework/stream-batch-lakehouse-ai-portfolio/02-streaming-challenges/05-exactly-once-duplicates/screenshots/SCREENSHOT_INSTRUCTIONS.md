# Screenshot Instructions: Exactly-Once (Experiment 5)

Student: REDACTED  |  Student ID: demo000000

## Required: 2 Screenshots (5-1 through 5-2)

### Screenshot 5-1: Duplicate Data Query Result
**What to capture:** Spark SQL query result showing duplicates in append-only table.
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data-demo000000/duplicate_demo \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true
```
```sql
USE paimon_catalog;
SELECT order_id, COUNT(*) as dup_count
FROM append_table
GROUP BY order_id HAVING COUNT(*) > 1
ORDER BY dup_count DESC LIMIT 10;
```
**Check:** Shows multiple order_ids with dup_count > 1. Table name contains `demo000000`.

### Screenshot 5-2: No Duplicates After PK Dedup
**What to capture:** Same Spark SQL, query the PK table:
```sql
SELECT order_id, COUNT(*) as dup_count
FROM pk_table
GROUP BY order_id HAVING COUNT(*) > 1;
```
**Check:** Result is EMPTY (0 rows). Proves deduplication works.
Also run the comparison query:
```sql
SELECT 'Append Table' as t, COUNT(*), COUNT(DISTINCT order_id)
FROM append_table
UNION ALL
SELECT 'PK Table', COUNT(*), COUNT(DISTINCT order_id) FROM pk_table;
```
**Check:** `demo000000` visible in table names.
