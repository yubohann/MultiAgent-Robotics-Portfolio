import json
import time
import random
import os
from kafka import KafkaProducer
from datetime import datetime

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"user_click_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
)

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print("Strategy: 80% from existing user pool, 20% new users (state grows continuously)")
print("=" * 80)

click_id = 1
user_pool = list(range(1, 1001))  # Start with 1000 users

try:
    while True:
        if random.random() < 0.8:
            user_id = random.choice(user_pool)
        else:
            user_id = len(user_pool) + 1
            user_pool.append(user_id)

        data = {
            "click_id": f"CLK_{click_id:08d}",
            "user_id": f"USER_{user_id:05d}",
            "page_url": f"/page_{random.randint(1, 100)}",
            "click_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time() * 1000),
        }

        producer.send(TOPIC, value=data)

        if click_id % 100 == 0:
            print(f"Sent {click_id} records | user pool size: {len(user_pool)}")

        click_id += 1
        time.sleep(0.01)

except KeyboardInterrupt:
    print(f"Stopped. Total sent: {click_id - 1} records | Final user pool: {len(user_pool)}")
    producer.close()
