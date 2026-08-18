# Performance Comparison: Small Files

Student: REDACTED  |  Student ID: demo000000  |  MinIO Bucket: paimon-data-demo000000

## Before Compaction

Checkpoint interval: 10s. Auto-compaction: disabled.
After running producer for 5 minutes (~30,000 orders):

```bash
# File count
$ docker exec -it bigdata-minio mc find /data/paimon-data-demo000000/orders --name "*.parquet" | wc -l
156

# Total size
$ docker exec -it bigdata-minio mc du /data/paimon-data-demo000000/orders
12.3 MB
```

Average file size: ~79 KB per file.
Query: `SELECT COUNT(*) FROM paimon_orders_demo000000` took 8.2s.

## After Compaction

Triggered: `CALL sys.compact(table => 'orders', order_strategy => 'zorder')`

```bash
# File count
$ docker exec -it bigdata-minio mc find /data/paimon-data-demo000000/orders --name "*.parquet" | wc -l
7

# Total size
$ docker exec -it bigdata-minio mc du /data/paimon-data-demo000000/orders
11.8 MB
```

Average file size: ~1.7 MB per file.
Query: `SELECT COUNT(*) FROM paimon_orders_demo000000` took 0.6s.

## Comparison Table

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total files | 156 | 7 | 22.3x reduction |
| Average file size | 79 KB | 1.7 MB | 21.5x larger |
| COUNT query time | 8.2s | 0.6s | 13.7x faster |
| Aggregation query time | 12.4s | 1.1s | 11.3x faster |
