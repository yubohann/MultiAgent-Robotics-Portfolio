# 03-ai-recommender Run And Screenshot Guide

Use a WSL terminal, not PowerShell, because Docker is available inside WSL in this environment.

## 1. Enter The Project Folder

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender
```

## 2. Check The Environment

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
python3 --version
mvn -version
```

Expected:

- `bigdata-kafka` is running.
- Python and Maven are available.

## 3. Run Code Layout Check

```bash
cd recommender-services
python3 verify_all_code.py | tee verify_all_code.log
```

Expected:

- Python service files are `OK`.
- Java/Flink files are `OK`.
- Final result is `PASS`.

## 4. Run Dependency Check

```bash
python3 test_dependencies.py | tee test_dependencies.log
```

Expected:

- `kafka-python`, `torch`, `pandas`, and `numpy` are installed.

## 5. Run Python Recommendation Tests

```bash
python3 test_recommendation_system.py | tee test_recommendation_system.log
```

Expected:

- Test summary shows `success: 8`, `failed: 0`, `errors: 0`.

## 6. Build The Flink Java Job

```bash
cd ../flink-jobs/realtime-recommender-flink-job
mvn -DskipTests clean package | tee ../../recommender-services/maven_build.log
cd ../../recommender-services
```

Expected:

- Maven output contains `BUILD SUCCESS`.
- Jar is generated:
  `target/realtime-recommender-flink-job-1.0-SNAPSHOT.jar`.

## 7. Run End-To-End Kafka Smoke Test

```bash
bash run_e2e_smoke_test.sh
```

Expected:

- It recreates Kafka topics `user_behaviors` and `recommendations`.
- It starts `realtime_recommendation_service.py`.
- It starts `user_behavior_producer.py`.
- It consumes recommendation messages from Kafka.
- Final output shows `recommendation message count: 8` or at least greater than 0.

## 8. Optional One-Command Full Verification

After the first manual run works, you can use:

```bash
bash run_full_verification.sh | tee full_verification.log
```

## 9. Screenshot Content

Capture one combined screenshot if possible, or split into two screenshots if your terminal cannot fit everything.

The screenshot should include:

- `03-ai-recommender full verification` or `Verify lab 03 code layout`
- `Student ID: demo000000`
- `verify_all_code.py: PASS` or the final `PASS` from `verify_all_code.py`
- Dependency check showing `kafka-python`, `torch`, `pandas`, `numpy`
- Python recommendation test summary showing `success: 8`, `failed: 0`, `errors: 0`
- Maven `BUILD SUCCESS`
- Kafka smoke test:
  `consumer exit code: 0`
  and recommendation messages containing `user_id`, `trigger_product`, `recommendations`
- `recommendation message count: 8` or at least greater than 0

Save the screenshot as:

```text
screenshots/03-ai-recommender-verification.png
```
