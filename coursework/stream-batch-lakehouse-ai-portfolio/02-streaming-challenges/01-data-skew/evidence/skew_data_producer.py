import json
import time
import random
import os
from kafka import KafkaProducer
from datetime import datetime

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"click_stream_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
)

items = [
    ("iPhone15", 0.90),
    ("MacBookPro", 0.03),
    ("iPadAir", 0.03),
    ("AirPods", 0.02),
    ("AppleWatch", 0.02),
]

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print(f"Data distribution: iPhone15 (90%), others (10%)")
print("=" * 80)

click_id = 1
try:
    while True:
        r = random.random()
        cumulative = 0
        selected_item = "iPhone15"
        for item, prob in items:
            cumulative += prob
            if r <= cumulative:
                selected_item = item
                break

        data = {
            "click_id": f"CLK_{click_id:08d}",
            "item_id": selected_item,
            "user_id": f"USER_{random.randint(1, 10000):05d}",
            "click_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time() * 1000),
        }

        future = producer.send(TOPIC, value=data)
        if click_id % 100 == 0:
            print(f"Sent {click_id} records | current item: {selected_item}")

        click_id += 1
        time.sleep(0.01)

except KeyboardInterrupt:
    print(f"Stopped. Total sent: {click_id - 1} records")
    producer.close()
