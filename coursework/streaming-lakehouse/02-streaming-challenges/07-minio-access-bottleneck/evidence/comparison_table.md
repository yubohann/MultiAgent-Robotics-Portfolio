# Performance Comparison: MinIO Access Bottleneck

Student: REDACTED  |  Student ID: demo000000  |  Bucket: paimon-data-demo000000

## Data: 31 days x 1000 records = 31,000 records

## Query Performance

| Query Type | Unpartitioned Table | Partitioned Table | Improvement |
|------------|-------------------|-------------------|-------------|
| Full table COUNT | 11.8s | 9.2s | 1.3x |
| Single partition COUNT (dt='2023-10-15') | 8.4s | 0.6s | 14.0x |
| Range query (10 days) | 10.2s | 1.4s | 7.3x |
| Aggregation (category, 10 days) | 16.5s | 2.1s | 7.9x |

## EXPLAIN FORMATTED Comparison

### Unpartitioned Table:
```
PartitionFilters: []
SelectedPartitions: 1 (but actually scans all 31 day directories)
FileScan: 31 directories, 310 files scanned
```

### Partitioned Table:
```
PartitionFilters: [isnotnull(dt#42), (dt#42 = 2023-10-15)]
SelectedPartitions: 1
FileScan: 1 directory, ~10 files scanned
```

## Scanned Data Volume

| Metric | Unpartitioned | Partitioned (single dt) | Improvement |
|--------|--------------|------------------------|-------------|
| Partitions scanned | 31 | 1 | 31x |
| Files scanned | ~310 | ~10 | 31x |
| Data scanned | ~310 MB | ~10 MB | 31x |
| ListObjects calls | 31+ | 1 | 31x |

## Root Cause

Object storage (MinIO/S3) has high latency for LIST operations. Unpartitioned tables require listing ALL directories before filtering. Partitioned tables use directory-based pruning: `WHERE dt = '2023-10-15'` directly navigates to `dt=2023-10-15/` subdirectory, bypassing 30 other partitions entirely.
