import json
import os
import random
import time
from datetime import datetime

from kafka import KafkaProducer


STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
CLASS_NO = os.getenv("CLASS_NO", "0")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")
TOPIC = os.getenv("KAFKA_TOPIC", f"ecommerce_orders_{STUDENT_ID}")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

PRODUCTS = ["iPhone 15", "MacBook Pro", "iPad Air", "AirPods"]
STATUSES = ["UNPAID", "PAID", "SHIPPED"]


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS.split(","),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda value: value.encode("utf-8"),
        retries=5,
        linger_ms=50,
    )


def build_order(order_no: int) -> dict:
    return {
        "order_id": f"ORD_{STUDENT_ID}_{order_no}",
        "product_name": random.choice(PRODUCTS),
        "amount": round(random.uniform(100.0, 20000.0), 2),
        "status": random.choice(STATUSES),
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    producer = build_producer()
    order_no = int(os.getenv("ORDER_START", "1"))
    max_messages = int(os.getenv("MAX_MESSAGES", "0"))

    print("Kafka order producer started")
    print(f"student_id={STUDENT_ID}, class_no={CLASS_NO}, student_name={STUDENT_NAME}")
    print(f"topic={TOPIC}, bootstrap_servers={BOOTSTRAP_SERVERS}")

    try:
        sent_count = 0
        while True:
            order = build_order(order_no)
            future = producer.send(TOPIC, key=order["order_id"], value=order)
            metadata = future.get(timeout=10)
            sent_count += 1
            print(
                "sent "
                f"topic={metadata.topic} partition={metadata.partition} offset={metadata.offset} "
                f"order_id={order['order_id']} amount={order['amount']} status={order['status']}"
            )
            if max_messages and sent_count >= max_messages:
                print(f"sent {sent_count} messages, exiting because MAX_MESSAGES={max_messages}")
                break
            order_no += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("producer stopped by user")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()

