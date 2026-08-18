-- MinIO Access Bottleneck Challenge: Partition Comparison
-- Student: REDACTED  Student ID: demo000000

-- ============================================================
-- Kafka source table
-- ============================================================
CREATE TABLE kafka_partition_demo_demo000000 (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'partition_demo_demo000000',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'partition-demo-demo000000',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);

-- ============================================================
-- Table A: Unpartitioned (all files in one directory)
-- ============================================================
CREATE TABLE sales_unpartitioned_demo000000 (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING,
    PRIMARY KEY (record_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/partition_demo/sales_unpartitioned',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true'
);

-- ============================================================
-- Table B: Partitioned by dt
-- ============================================================
CREATE TABLE sales_partitioned_demo000000 (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING,
    PRIMARY KEY (record_id, dt) NOT ENFORCED
) PARTITIONED BY (dt)
WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/partition_demo/sales_partitioned',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true'
);

-- Write to both
INSERT INTO sales_unpartitioned_demo000000 SELECT * FROM kafka_partition_demo_demo000000;
INSERT INTO sales_partitioned_demo000000 SELECT * FROM kafka_partition_demo_demo000000;

-- ============================================================
-- Verification queries (run in Spark SQL)
-- ============================================================

-- Query 1: Full table scan (both scan all data)
SELECT COUNT(*) FROM sales_unpartitioned_demo000000;  -- ~12s
SELECT COUNT(*) FROM sales_partitioned_demo000000;     -- ~9s

-- Query 2: Single partition query (partitioned table is MUCH faster)
SELECT COUNT(*) FROM sales_unpartitioned_demo000000
WHERE dt = '2023-10-15';  -- ~8s (scans all 31 partitions)
SELECT COUNT(*) FROM sales_partitioned_demo000000
WHERE dt = '2023-10-15';  -- ~0.5s (scans only 1 partition)

-- Query 3: Range query
SELECT category, SUM(amount) as total_amount
FROM sales_partitioned_demo000000
WHERE dt BETWEEN '2023-10-10' AND '2023-10-20'
GROUP BY category;  -- ~1.5s (scans 11 partitions)

-- Query 4: EXPLAIN to see partition pruning
EXPLAIN FORMATTED
SELECT COUNT(*) FROM sales_partitioned_demo000000
WHERE dt = '2023-10-15';
-- Look for: PartitionFilters: [isnotnull(dt), (dt = 2023-10-15)]
--           SelectedPartitions: 1
