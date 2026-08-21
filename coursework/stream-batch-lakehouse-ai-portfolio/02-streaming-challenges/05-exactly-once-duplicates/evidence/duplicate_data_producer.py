import json
import time
import random
import os
from kafka import KafkaProducer
from datetime import datetime

STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
CLASS_NO = os.getenv("CLASS_NO", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"duplicate_data_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
)

sent_orders = []
order_id = 1
new_count = 0
dup_count = 0

print(f"Name: {STUDENT_NAME}  Student ID: {STUDENT_ID}  Class: {CLASS_NO}")
print(f"Kafka Topic: {TOPIC}")
print("Strategy: 70% new orders, 30% duplicates from sent pool")
print("=" * 80)

try:
    while True:
        if random.random() < 0.7 or len(sent_orders) < 10:
            data = {
                "order_id": order_id,
                "user_id": f"USER_{random.randint(1, 1000):04d}",
                "product_id": f"PROD_{random.randint(1, 100):03d}",
                "amount": round(random.uniform(100.0, 1000.0), 2),
                "status": random.choice(["PAID", "SHIPPED", "DELIVERED"]),
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            sent_orders.append(data.copy())
            order_id += 1
            new_count += 1
        else:
            data = random.choice(sent_orders).copy()
            data["create_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dup_count += 1

        producer.send(TOPIC, value=data)

        if (order_id - 1) % 50 == 0:
            print(f"Sent {order_id - 1} records | new: {new_count}, duplicates: {dup_count} | pool: {len(sent_orders)}")

        time.sleep(0.02)

except KeyboardInterrupt:
    print(f"Stopped. Total: {order_id - 1} | new: {new_count}, duplicates: {dup_count}")
    producer.close()
