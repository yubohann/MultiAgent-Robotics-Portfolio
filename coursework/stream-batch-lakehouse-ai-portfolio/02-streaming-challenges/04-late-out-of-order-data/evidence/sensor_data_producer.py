import json
import time
import random
import os
from kafka import KafkaProducer
from datetime import datetime, timedelta

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"sensor_data_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
)

sensor_ids = [f"SENSOR_{i:03d}" for i in range(1, 11)]

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print("Strategy: 90% normal timestamps, 10% late (1-300s delay)")
print("=" * 80)

record_id = 1
normal_count = 0
late_count = 0

try:
    while True:
        if random.random() < 0.9:
            event_time = datetime.now()
            normal_count += 1
        else:
            delay_seconds = random.randint(1, 300)
            event_time = datetime.now() - timedelta(seconds=delay_seconds)
            late_count += 1

        data = {
            "record_id": f"REC_{record_id:08d}",
            "sensor_id": random.choice(sensor_ids),
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(40.0, 80.0), 2),
            "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(event_time.timestamp() * 1000),
        }

        producer.send(TOPIC, value=data)

        if record_id % 100 == 0:
            print(f"Sent {record_id} records | normal: {normal_count}, late: {late_count}")

        record_id += 1
        time.sleep(0.01)

except KeyboardInterrupt:
    print(f"Stopped. Total: {record_id - 1} | normal: {normal_count}, late: {late_count}")
    producer.close()
