import json
import time
import random
import os
from kafka import KafkaProducer
from datetime import datetime

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"schema_demo_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
)

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print("Generating user profile data for schema evolution demo")
print("=" * 80)

user_id = 4  # Start after initial 3 rows
try:
    while True:
        data = {
            "user_id": user_id,
            "username": f"User_{user_id}",
            "email": f"user_{user_id}@example.com",
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        producer.send(TOPIC, value=data)

        if user_id % 10 == 0:
            print(f"Sent user_id {user_id}")

        user_id += 1
        time.sleep(0.1)

except KeyboardInterrupt:
    print(f"Stopped. Last user_id: {user_id - 1}")
    producer.close()
