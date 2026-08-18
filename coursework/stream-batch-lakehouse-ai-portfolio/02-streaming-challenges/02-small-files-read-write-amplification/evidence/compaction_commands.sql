-- Small Files Challenge: Flink SQL for Paimon
-- Student: REDACTED  Student ID: demo000000

-- Step 1: Create Kafka source table
CREATE TABLE kafka_orders_demo000000 (
    order_id BIGINT,
    product_name STRING,
    amount DOUBLE,
    status STRING,
    user_id STRING,
    create_time STRING,
    `timestamp` BIGINT
) WITH (
    'connector' = 'kafka',
    'topic' = 'order_stream_demo000000',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'small-file-demo-demo000000',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- Step 2: Create Paimon table WITHOUT auto-compaction (to reproduce small files)
CREATE TABLE paimon_orders_demo000000 (
    order_id BIGINT,
    product_name STRING,
    amount DOUBLE,
    status STRING,
    user_id STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/orders',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'false',
    'write-buffer-size' = '16mb',
    'page-size' = '32kb'
);

-- Step 3: Continuous insert
INSERT INTO paimon_orders_demo000000
SELECT order_id, product_name, amount, status, user_id, create_time
FROM kafka_orders_demo000000;

-- Step 4: Check file count in MinIO (run in bash)
-- docker exec -it bigdata-minio mc find /data/paimon-data-demo000000/orders --name "*.parquet" | wc -l

-- Step 5: Trigger compaction via Spark SQL
-- CALL sys.compact(table => 'orders', order_strategy => 'zorder');

-- Step 6: Create table WITH auto-compaction for comparison
CREATE TABLE paimon_orders_compacted_demo000000 (
    order_id BIGINT,
    product_name STRING,
    amount DOUBLE,
    status STRING,
    user_id STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/orders_compacted',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true',
    'compaction.target-file-size' = '128mb',
    'compaction.num-sorted-run.compaction-trigger' = '5'
);
