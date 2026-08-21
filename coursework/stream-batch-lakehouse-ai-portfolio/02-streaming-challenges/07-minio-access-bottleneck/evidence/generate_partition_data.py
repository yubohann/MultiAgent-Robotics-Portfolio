import json
import random
import os
from datetime import datetime, timedelta
from kafka import KafkaProducer

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"partition_demo_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    max_request_size=10485760,
)

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print("Generating 31 days x 1000 records = 31000 records")
print("=" * 80)

start_date = datetime(2023, 10, 1)
categories = ["Electronics", "Clothing", "Books", "Food"]

record_id = 1
for day in range(31):
    current_date = start_date + timedelta(days=day)
    dt = current_date.strftime("%Y-%m-%d")

    for i in range(1000):
        data = {
            "record_id": record_id,
            "user_id": f"USER_{random.randint(1, 1000):04d}",
            "product_id": f"PROD_{random.randint(1, 100):03d}",
            "amount": round(random.uniform(10.0, 1000.0), 2),
            "category": random.choice(categories),
            "dt": dt,
        }
        producer.send(TOPIC, value=data)
        record_id += 1

    print(f"Day {day + 1}/31: {dt} - 1000 records sent")

print(f"Total records generated: {record_id - 1}")
producer.flush()
producer.close()
