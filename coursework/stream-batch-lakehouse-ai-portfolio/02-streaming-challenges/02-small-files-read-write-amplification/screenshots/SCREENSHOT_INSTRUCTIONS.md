# Screenshot Instructions: Small Files (Experiment 2)

Student: REDACTED  |  Student ID: demo000000

## Required: 3 Screenshots (2-1 through 2-3)

### Screenshot 2-1: MinIO Bucket List
**What to capture:** MinIO Console (http://localhost:9001) → Buckets page.
- `paimon-data-demo000000` bucket visible in the list.
- Login: admin / password123.
**Check:** Bucket name contains `demo000000`.

### Screenshot 2-2: Small Files Explosion
**What to capture:** MinIO Console → paimon-data-demo000000 → orders → bucket-0/.
- Must show MANY small `data-*.parquet` files (100+ files).
- File sizes visible (~KB range, not MB).
- Run first:
```bash
# Verify from terminal too
docker exec -it bigdata-minio mc find /data/paimon-data-demo000000/orders --name "*.parquet" | wc -l
docker exec -it bigdata-minio mc du /data/paimon-data-demo000000/orders
```
**Check:** Large number of files, each small in size. `demo000000` in path.

### Screenshot 2-3: After Compaction
**What to capture:** MinIO Console → SAME directory after running compaction.
- Only a few large `data-*.parquet` files (5-10 files, MB each).
- Also capture the Spark SQL compaction command and result:
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql ...
spark-sql> CALL sys.compact(table => 'orders', order_strategy => 'zorder');
```
- Then verify:
```bash
docker exec -it bigdata-minio mc find /data/paimon-data-demo000000/orders --name "*.parquet" | wc -l
```
**Check:** Dramatically fewer files. `demo000000` in path.
