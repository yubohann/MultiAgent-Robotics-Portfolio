#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

rm -f realtime_service_run.log behavior_producer_run.log recommendation_messages.jsonl ai_recommender_verification.log

docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --delete --topic user_behaviors --bootstrap-server localhost:9092 >/dev/null 2>&1 || true
docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --delete --topic recommendations --bootstrap-server localhost:9092 >/dev/null 2>&1 || true
sleep 5

docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic user_behaviors --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic recommendations --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

"$PYTHON_BIN" -u realtime_recommendation_service.py > realtime_service_run.log 2>&1 &
svc_pid=$!
sleep 5

"$PYTHON_BIN" -u user_behavior_producer.py > behavior_producer_run.log 2>&1 &
prod_pid=$!

set +e
timeout 35s docker exec bigdata-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic recommendations \
  --from-beginning \
  --max-messages 8 > recommendation_messages.jsonl 2>&1
consumer_rc=$?
set -e

kill "$prod_pid" >/dev/null 2>&1 || true
kill "$svc_pid" >/dev/null 2>&1 || true
sleep 2

{
  echo "=== AI recommender verification ==="
  echo "Student ID: demo000000"
  echo "consumer exit code: $consumer_rc"
  echo "--- service log tail ---"
  tail -80 realtime_service_run.log
  echo "--- producer log tail ---"
  tail -40 behavior_producer_run.log
  echo "--- recommendation messages ---"
  cat recommendation_messages.jsonl
} | tee ai_recommender_verification.log

msg_count=$(grep -c "user_id" recommendation_messages.jsonl || true)
echo "recommendation message count: $msg_count"
test "$msg_count" -ge 1
