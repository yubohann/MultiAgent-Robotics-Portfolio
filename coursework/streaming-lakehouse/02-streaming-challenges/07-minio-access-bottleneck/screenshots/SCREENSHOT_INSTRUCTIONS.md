# Screenshot Instructions: MinIO Access Bottleneck (Experiment 7)

Student: REDACTED  |  Student ID: demo000000

## Required: 2 Screenshots (7-1 through 7-2)

### Screenshot 7-1: Slow Query (Unpartitioned)
**What to capture:** Spark SQL showing slow query on unpartitioned table.
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data-demo000000/partition_demo \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true
```
```sql
USE paimon_catalog;
SELECT COUNT(*) FROM sales_unpartitioned WHERE dt = '2023-10-15';
-- Should take 8+ seconds
```
Also capture EXPLAIN:
```sql
EXPLAIN FORMATTED SELECT COUNT(*) FROM sales_unpartitioned WHERE dt = '2023-10-15';
-- Shows: PartitionFilters: [] (no pruning!)
```
**Check:** Slow response time visible. No partition pruning in EXPLAIN. `demo000000` in table name.

### Screenshot 7-2: Fast Query (Partitioned) + EXPLAIN
**What to capture:** Same Spark SQL, but on partitioned table:
```sql
SELECT COUNT(*) FROM sales_partitioned WHERE dt = '2023-10-15';
-- Should take <1 second

EXPLAIN FORMATTED SELECT COUNT(*) FROM sales_partitioned WHERE dt = '2023-10-15';
-- KEY: Shows PartitionFilters: [isnotnull(dt#...), (dt#... = 2023-10-15)]
-- KEY: Shows SelectedPartitions: 1
```
**Check:** Fast response time. EXPLAIN shows partition pruning active. `demo000000` visible.

### Bonus: MinIO Directory Structure Comparison
Open http://localhost:9001 → paimon-data-demo000000 → partition_demo/
Capture side-by-side showing:
- sales_unpartitioned/: flat structure, all files mixed
- sales_partitioned/: dt=2023-10-01/, dt=2023-10-02/, ..., dt=2023-10-31/ subdirectories
