-- Exactly-Once Challenge: Flink SQL for Duplicate Comparison
-- Student: REDACTED  Student ID: demo000000

-- ============================================================
-- Step 1: Create Kafka source table
-- ============================================================
CREATE TABLE kafka_duplicate_demo000000 (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'duplicate_data_demo000000',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'duplicate-demo-demo000000',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- ============================================================
-- Step 2: Table A - Append-only table (NO PRIMARY KEY)
-- ============================================================
CREATE TABLE paimon_append_demo000000 (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/duplicate_demo/append_table',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'write-mode' = 'append-only'
);

-- ============================================================
-- Step 3: Table B - PK table with deduplication
-- ============================================================
CREATE TABLE paimon_pk_demo000000 (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/duplicate_demo/pk_table',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'write-mode' = 'upsert',
    'merge-engine' = 'deduplicate'
);

-- ============================================================
-- Step 4: Write to both tables simultaneously
-- ============================================================
INSERT INTO paimon_append_demo000000 SELECT * FROM kafka_duplicate_demo000000;
INSERT INTO paimon_pk_demo000000 SELECT * FROM kafka_duplicate_demo000000;

-- ============================================================
-- Step 5: Verify results in Spark SQL
-- ============================================================

-- Compare append-only vs PK table
SELECT 'Append Table' as table_type, COUNT(*) as total_records,
       COUNT(DISTINCT order_id) as unique_orders,
       COUNT(*) - COUNT(DISTINCT order_id) as duplicate_count
FROM paimon_append_demo000000
UNION ALL
SELECT 'PK Table', COUNT(*), COUNT(DISTINCT order_id),
       COUNT(*) - COUNT(DISTINCT order_id)
FROM paimon_pk_demo000000;

-- Expected result:
-- | table_type    | total_records | unique_orders | duplicate_count |
-- |---------------|---------------|---------------|-----------------|
-- | Append Table  | ~1429         | ~1000         | ~429            |
-- | PK Table      | ~1000         | ~1000         | 0               |

-- Find duplicates in append table
SELECT order_id, COUNT(*) as dup_count
FROM paimon_append_demo000000
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY dup_count DESC
LIMIT 10;
