-- Schema Evolution Challenge: DDL Sequence
-- Student: REDACTED  Student ID: demo000000

-- ============================================================
-- Step 1: Create initial table (3 columns)
-- ============================================================
CREATE TABLE user_profile_demo000000 (
    user_id BIGINT,
    username STRING,
    email STRING,
    create_time STRING,
    PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data-demo000000/schema_demo/user_profile',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true'
);

-- Insert initial data
INSERT INTO user_profile_demo000000 VALUES
(1, 'Alice', 'alice@example.com', '2023-10-01 10:00:00'),
(2, 'Bob', 'bob@example.com', '2023-10-01 10:00:01'),
(3, 'Charlie', 'charlie@example.com', '2023-10-01 10:00:02');

-- Verify: SELECT * FROM user_profile_demo000000;

-- ============================================================
-- Step 2: ADD COLUMN (online, no downtime)
-- ============================================================
ALTER TABLE user_profile_demo000000 ADD age INT;
ALTER TABLE user_profile_demo000000 ADD address STRING;

-- DESCRIBE user_profile_demo000000; -- verify new schema

-- Insert data with new fields
INSERT INTO user_profile_demo000000
    (user_id, username, email, create_time, age, address) VALUES
(1001, 'David', 'david@example.com', '2023-10-01 11:00:00', 25, 'New York'),
(1002, 'Eva', 'eva@example.com', '2023-10-01 11:00:01', 30, 'London');

-- Query all data (old rows show NULL for new columns)
-- SELECT * FROM user_profile_demo000000 ORDER BY user_id;

-- ============================================================
-- Step 3: MODIFY COLUMN TYPE (INT -> BIGINT)
-- ============================================================
ALTER TABLE user_profile_demo000000 MODIFY COLUMN age BIGINT;

-- DESCRIBE user_profile_demo000000;

-- ============================================================
-- Step 4: RENAME COLUMN (address -> location)
-- ============================================================
ALTER TABLE user_profile_demo000000 RENAME COLUMN address TO location;

-- DESCRIBE user_profile_demo000000;

-- ============================================================
-- Step 5: DROP COLUMN
-- ============================================================
ALTER TABLE user_profile_demo000000 DROP COLUMN location;

-- DESCRIBE user_profile_demo000000;

-- ============================================================
-- Step 6: ADD NESTED STRUCT COLUMN
-- ============================================================
ALTER TABLE user_profile_demo000000 ADD address STRUCT<city:STRING, country:STRING, zipcode:STRING>;

INSERT INTO user_profile_demo000000
    (user_id, username, email, create_time, age, address) VALUES
(2001, 'Frank', 'frank@example.com', '2023-10-01 12:00:00', 35,
 STRUCT('Paris', 'France', '75001'));

-- Query nested field: SELECT user_id, username, address.city, address.country FROM user_profile_demo000000 WHERE address IS NOT NULL;

-- ============================================================
-- Verification queries
-- ============================================================

-- Check old vs new data compatibility
SELECT
    CASE WHEN age IS NULL THEN 'Old data' ELSE 'New data' END as data_type,
    COUNT(*) as count
FROM user_profile_demo000000
GROUP BY CASE WHEN age IS NULL THEN 'Old data' ELSE 'New data' END;

-- Expected:
-- | data_type | count |
-- |-----------|-------|
-- | Old data  | 3     |
-- | New data  | 2+    |
