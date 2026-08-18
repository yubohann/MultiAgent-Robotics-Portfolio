import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

users = [f"USER_{i:05d}" for i in range(1, 1001)]
products = [
    {"product_id": "P001", "category": "手机", "price": 5999, "name": "iPhone 15"},
    {"product_id": "P002", "category": "手机", "price": 3999, "name": "小米14"},
    {"product_id": "P003", "category": "电脑", "price": 12999, "name": "MacBook Pro"},
    {"product_id": "P004", "category": "电脑", "price": 5999, "name": "联想小新"},
    {"product_id": "P005", "category": "穿戴", "price": 2499, "name": "Apple Watch"},
    {"product_id": "P006", "category": "穿戴", "price": 899, "name": "小米手环"},
    {"product_id": "P007", "category": "耳机", "price": 1999, "name": "AirPods Pro"},
    {"product_id": "P008", "category": "耳机", "price": 299, "name": "小米耳机"}
]
behavior_types = ["click", "view", "cart", "purchase"]

print("🚀 开始生成用户行为数据...")
print("="*80)

behavior_id = 1
try:
    while True:
        user_id = random.choice(users)
        product = random.choice(products)
        
        behavior_type = random.choices(
            behavior_types,
            weights=[0.5, 0.3, 0.15, 0.05]
        )[0]
        
        data = {
            "behavior_id": f"BEH_{behavior_id:08d}",
            "user_id": user_id,
            "product_id": product["product_id"],
            "category": product["category"],
            "price": product["price"],
            "behavior_type": behavior_type,
            "timestamp": int(time.time() * 1000),
            "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        producer.send('user_behaviors', value=data)
        
        if behavior_id % 50 == 0:
            print(f"✅ 已发送 {behavior_id} 条 | 用户: {user_id} | 商品: {product['name']} | 行为: {behavior_type}")
        
        behavior_id += 1
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送，共发送 {behavior_id-1} 条记录")
    producer.close()
