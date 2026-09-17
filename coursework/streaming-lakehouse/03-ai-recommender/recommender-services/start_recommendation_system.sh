#!/bin/bash
set -e

echo "========================================="
echo "Stream-Batch AI Recommender Startup"
echo "========================================="

BASE_DIR=$(dirname "$0")
cd "$BASE_DIR"

echo ""
echo "Checking dependencies..."
command -v python3 >/dev/null 2>&1 || {
  echo "Python 3 is required."
  exit 1
}

echo ""
echo "Checking Kafka container..."
if ! docker ps | grep -q bigdata-kafka; then
  echo "Kafka container bigdata-kafka is not running."
  echo "Start the lab cluster first, for example: docker compose up -d"
  exit 1
fi

echo ""
echo "Run these commands in separate terminals:"
echo ""
echo "Terminal 1 - behavior producer:"
echo "  python3 user_behavior_producer.py"
echo ""
echo "Terminal 2 - real-time recommendation service:"
echo "  python3 realtime_recommendation_service.py"
echo ""
echo "Terminal 3 - recommendation result monitor:"
echo "  python3 advanced_recommendation_consumer.py"
echo ""
echo "Terminal 4 - algorithm demo:"
echo "  python3 recommendation_algorithms.py"
echo ""
echo "Install Python dependencies when needed:"
echo "  pip install kafka-python torch pandas numpy"
