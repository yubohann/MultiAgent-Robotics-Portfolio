from kafka import KafkaConsumer
import json
from datetime import datetime

consumer = KafkaConsumer(
    'recommendations',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest'
)

product_names = {
    "P001": "iPhone 15",
    "P002": "小米14",
    "P003": "MacBook Pro",
    "P004": "联想小新",
    "P005": "Apple Watch",
    "P006": "小米手环",
    "P007": "AirPods Pro",
    "P008": "小米耳机"
}

print("🎯 实时推荐系统监控")
print("="*80)

for message in consumer:
    data = message.value
    time_str = datetime.fromtimestamp(data['timestamp']/1000).strftime('%H:%M:%S')
    
    print(f"\n⏰ [{time_str}] 用户: {data['user_id']}")
    print(f"   触发商品: {product_names.get(data['trigger_product'], data['trigger_product'])}")
    print(f"   📦 推荐商品:")
    for i, prod_id in enumerate(data['recommendations'], 1):
        print(f"      {i}. {product_names.get(prod_id, prod_id)}")
