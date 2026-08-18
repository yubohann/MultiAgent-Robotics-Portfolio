#!/bin/bash
set -euo pipefail

TOPICS=(
  user_behaviors
  fast_recommendations
  deep_recommendations
  final_recommendations
)

for topic in "${TOPICS[@]}"; do
  docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
    --create \
    --if-not-exists \
    --topic "$topic" \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1
done

docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --list \
  --bootstrap-server localhost:9092 | sort | grep -E '^(user_behaviors|fast_recommendations|deep_recommendations|final_recommendations)$'
