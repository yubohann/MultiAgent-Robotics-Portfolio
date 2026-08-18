import json
import time

from kafka import KafkaProducer


producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
)

messages = [
    {
        "user_id": "USER_00001",
        "recommendations": ["P002", "P004", "P006", "P008", "P003"],
        "source": "deepfm",
        "model": "DeepFM",
    },
    {
        "user_id": "USER_00002",
        "recommendations": ["P001", "P003", "P005", "P007", "P004"],
        "source": "deepfm",
        "model": "DeepFM",
    },
    {
        "user_id": "USER_00003",
        "recommendations": ["P004", "P001", "P006", "P007", "P002"],
        "source": "deepfm",
        "model": "DeepFM",
    },
]

print("Send DeepFM recommendations to Kafka topic: deep_recommendations")
print("Student ID: demo000000")

for msg in messages:
    msg["timestamp"] = int(time.time() * 1000)
    producer.send("deep_recommendations", value=msg)
    print(json.dumps(msg, ensure_ascii=False))

producer.flush()
producer.close()
print(f"Sent {len(messages)} deep recommendation messages")
