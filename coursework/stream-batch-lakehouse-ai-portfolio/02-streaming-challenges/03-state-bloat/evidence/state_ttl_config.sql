-- State Bloat Challenge: Flink SQL
-- Student: REDACTED  Student ID: demo000000

-- ============================================================
-- Phase 1: Reproduce State Bloat (NO TTL)
-- ============================================================

SET 'state.backend' = 'rocksdb';
SET 'state.backend.rocksdb.localdir' = '/tmp/flink/rocksdb-demo000000';
SET 'execution.checkpointing.interval' = '10s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

-- Kafka source table
CREATE TABLE user_click_demo000000 (
    click_id STRING,
    user_id STRING,
    page_url STRING,
    click_time STRING,
    `timestamp` BIGINT,
    ts AS TO_TIMESTAMP(click_time, 'yyyy-MM-dd HH:mm:ss'),
    WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'user_click_demo000000',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'state-bloat-demo-demo000000',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- UV result table WITHOUT TTL
CREATE TABLE uv_result_demo000000 (
    dt STRING,
    uv BIGINT,
    PRIMARY KEY (dt) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/uv_result',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true'
);

INSERT INTO uv_result_demo000000
SELECT
    DATE_FORMAT(ts, 'yyyy-MM-dd') as dt,
    COUNT(DISTINCT user_id) as uv
FROM user_click_demo000000
GROUP BY DATE_FORMAT(ts, 'yyyy-MM-dd');

-- ============================================================
-- Phase 2: Optimize with State TTL
-- ============================================================

SET 'table.exec.state.ttl' = '24 h';

DROP TABLE IF EXISTS uv_result_demo000000;

CREATE TABLE uv_result_ttl_demo000000 (
    dt STRING,
    uv BIGINT,
    PRIMARY KEY (dt) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/uv_result_ttl',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true'
);

INSERT INTO uv_result_ttl_demo000000
SELECT
    DATE_FORMAT(ts, 'yyyy-MM-dd') as dt,
    COUNT(DISTINCT user_id) as uv
FROM user_click_demo000000
GROUP BY DATE_FORMAT(ts, 'yyyy-MM-dd');

-- TTL cleanup strategies
-- SET 'table.exec.state.ttl.cleanup.strategy' = 'incremental';
-- SET 'state.backend.rocksdb.ttl.compaction.filter.enabled' = 'true';
