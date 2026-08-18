import json
import time
import random
import os
from kafka import KafkaProducer
from datetime import datetime

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"order_stream_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
)

products = ["iPhone 15", "MacBook Pro", "iPad Air", "AirPods", "Apple Watch"]
statuses = ["UNPAID", "PAID", "SHIPPED", "DELIVERED"]

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print(f"Bootstrap Servers: {BOOTSTRAP_SERVERS}")
print("=" * 80)

order_id = 1
try:
    while True:
        data = {
            "order_id": order_id,
            "product_name": random.choice(products),
            "amount": round(random.uniform(100.0, 20000.0), 2),
            "status": random.choice(statuses),
            "user_id": f"USER_{random.randint(1, 1000):04d}",
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time() * 1000),
        }
        producer.send(TOPIC, value=data)
        if order_id % 100 == 0:
            print(f"Sent {order_id} order records")
        order_id += 1
        time.sleep(0.01)
except KeyboardInterrupt:
    print(f"Stopped. Total sent: {order_id - 1} order records")
    producer.close()
