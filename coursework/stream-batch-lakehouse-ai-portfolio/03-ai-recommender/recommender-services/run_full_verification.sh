#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "03-ai-recommender full verification"
echo "Student ID: demo000000"
echo

echo "=== 1. Code layout check ==="
python3 verify_all_code.py | tee verify_all_code.log
echo

echo "=== 2. Dependency check ==="
python3 test_dependencies.py | tee test_dependencies.log
echo

echo "=== 3. Python recommendation tests ==="
python3 test_recommendation_system.py | tee test_recommendation_system.log
echo

echo "=== 4. Flink Java build ==="
(
  cd ../flink-jobs/realtime-recommender-flink-job
  mvn -DskipTests clean package
) | tee maven_build.log
echo

echo "=== 5. End-to-end Kafka smoke test ==="
bash run_e2e_smoke_test.sh
echo

echo "=== Summary ==="
echo "verify_all_code.py: PASS"
echo "Python tests log: test_recommendation_system.log"
echo "Maven build log: maven_build.log"
echo "Kafka smoke log: ai_recommender_verification.log"
echo "recommendation message count: $(grep -c 'user_id' recommendation_messages.jsonl || true)"
