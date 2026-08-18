SET 'execution.attached' = 'false';
SET 'execution.checkpointing.interval' = '10s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

CREATE TEMPORARY TABLE kafka_partition_unpart_demo000000 (
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
    'properties.group.id' = 'partition-demo-unpart-demo000000',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);

CREATE TEMPORARY TABLE kafka_partition_part_demo000000 (
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
    'properties.group.id' = 'partition-demo-part-demo000000',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);

CREATE CATALOG paimon_demo000000 WITH (
    'type' = 'paimon',
    'warehouse' = 's3://paimon-data-demo000000/streaming-challenges',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true'
);

USE CATALOG paimon_demo000000;
CREATE DATABASE IF NOT EXISTS c07;
USE c07;

DROP TABLE IF EXISTS sales_unpartitioned_demo000000;
DROP TABLE IF EXISTS sales_partitioned_demo000000;

CREATE TABLE sales_unpartitioned_demo000000 (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING,
    PRIMARY KEY (record_id) NOT ENFORCED
) WITH (
    'auto-compaction' = 'true'
);

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
    'auto-compaction' = 'true'
);

INSERT INTO sales_unpartitioned_demo000000
SELECT * FROM default_catalog.default_database.kafka_partition_unpart_demo000000;

INSERT INTO sales_partitioned_demo000000
SELECT * FROM default_catalog.default_database.kafka_partition_part_demo000000;
