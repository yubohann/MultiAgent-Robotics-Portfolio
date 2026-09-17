SET 'execution.attached' = 'false';
SET 'execution.checkpointing.interval' = '10s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

CREATE TEMPORARY TABLE kafka_duplicate_append_demo000000 (
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
    'properties.group.id' = 'duplicate-demo-append-clean-demo000000',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);

CREATE TEMPORARY TABLE kafka_duplicate_pk_demo000000 (
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
    'properties.group.id' = 'duplicate-demo-pk-clean-demo000000',
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
CREATE DATABASE IF NOT EXISTS c05;
USE c05;

DROP TABLE IF EXISTS append_table_demo000000;
DROP TABLE IF EXISTS pk_table_demo000000;

CREATE TABLE append_table_demo000000 (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING
);

CREATE TABLE pk_table_demo000000 (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'merge-engine' = 'deduplicate'
);

INSERT INTO append_table_demo000000
SELECT * FROM default_catalog.default_database.kafka_duplicate_append_demo000000;

INSERT INTO pk_table_demo000000
SELECT * FROM default_catalog.default_database.kafka_duplicate_pk_demo000000;
