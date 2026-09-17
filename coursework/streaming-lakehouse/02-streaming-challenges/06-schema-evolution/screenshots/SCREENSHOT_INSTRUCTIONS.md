# Screenshot Instructions: Schema Evolution (Experiment 6)

Student: REDACTED  |  Student ID: demo000000

## Required: 2 Screenshots (6-1 through 6-2)

### Screenshot 6-1: Schema Change Error / Initial State
**What to capture:** Flink SQL DESCRIBE showing the initial schema (4 columns) and/or query error when trying to SELECT a column that doesn't exist yet.
```sql
DESCRIBE user_profile_demo000000;
SELECT age FROM user_profile_demo000000;  -- should error: column not found
```
**Check:** `user_profile_demo000000` table name visible. Initial schema with only user_id, username, email, create_time.

### Screenshot 6-2: Schema Evolution Success
**What to capture:** After running all DDL changes, show:
```sql
DESCRIBE user_profile_demo000000;
-- Must show: user_id, username, email, create_time, age (BIGINT)
-- Note: address renamed to location then dropped, so not present

SELECT * FROM user_profile_demo000000 ORDER BY user_id;
-- Must show old rows (1-3) with age=NULL, new rows with age filled

SELECT CASE WHEN age IS NULL THEN 'Old data' ELSE 'New data' END, COUNT(*)
FROM user_profile_demo000000
GROUP BY CASE WHEN age IS NULL THEN 'Old data' ELSE 'New data' END;
-- Shows old + new data coexisting
```
**Check:** Schema changes successfully applied. Old data readable. `demo000000` in table name.
