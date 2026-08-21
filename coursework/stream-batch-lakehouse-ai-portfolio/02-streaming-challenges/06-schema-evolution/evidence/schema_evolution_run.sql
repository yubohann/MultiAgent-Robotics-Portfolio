SET 'execution.attached' = 'true';

CREATE CATALOG paimon_demo000000 WITH (
    'type' = 'paimon',
    'warehouse' = 's3://paimon-data-demo000000/streaming-challenges',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true'
);

USE CATALOG paimon_demo000000;
CREATE DATABASE IF NOT EXISTS c06;
USE c06;

DROP TABLE IF EXISTS user_profile_demo000000;

CREATE TABLE user_profile_demo000000 (
    user_id BIGINT,
    username STRING,
    email STRING,
    create_time STRING,
    PRIMARY KEY (user_id) NOT ENFORCED
);

INSERT INTO user_profile_demo000000 VALUES
(1, 'Alice', 'alice@example.com', '2023-10-01 10:00:00'),
(2, 'Bob', 'bob@example.com', '2023-10-01 10:00:01'),
(3, 'Charlie', 'charlie@example.com', '2023-10-01 10:00:02');

ALTER TABLE user_profile_demo000000 ADD age INT;
ALTER TABLE user_profile_demo000000 ADD address STRING;

INSERT INTO user_profile_demo000000 (user_id, username, email, create_time, age, address) VALUES
(1001, 'David', 'david@example.com', '2023-10-01 11:00:00', 25, 'New York'),
(1002, 'Eva', 'eva@example.com', '2023-10-01 11:00:01', 30, 'London');

ALTER TABLE user_profile_demo000000 MODIFY age BIGINT;
ALTER TABLE user_profile_demo000000 RENAME address TO location;
ALTER TABLE user_profile_demo000000 DROP location;

DESCRIBE user_profile_demo000000;
