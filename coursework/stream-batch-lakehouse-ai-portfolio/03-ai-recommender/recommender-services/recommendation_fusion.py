from kafka import KafkaConsumer, KafkaProducer
import json
from datetime import datetime
from collections import defaultdict
import threading
import time
import pickle


class RecommendationFusion:
    def __init__(self, kafka_bootstrap_servers='localhost:9092', 
                 deep_model_path='deepfm_model.pth',
                 metadata_path='metadata.pkl'):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.deep_model_path = deep_model_path
        self.metadata_path = metadata_path
        
        self.fast_recommendations = {}
        self.deep_recommendations = {}
        self.user_activity = {}
        
        self.fast_recommendations_lock = threading.Lock()
        self.deep_recommendations_lock = threading.Lock()
        
        self.fast_weight = 0.6
        self.deep_weight = 0.4
        
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.fast_consumer = KafkaConsumer(
            'fast_recommendations',
            bootstrap_servers=[kafka_bootstrap_servers],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='fusion-fast-group'
        )
        
        self.deep_consumer = KafkaConsumer(
            'deep_recommendations',
            bootstrap_servers=[kafka_bootstrap_servers],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='fusion-deep-group'
        )
        
        self.deep_recommender = None
        self.item_features = None
        self.is_running = False
        
        self.product_names = {
            "P001": "iPhone 15",
            "P002": "小米14",
            "P003": "MacBook Pro",
            "P004": "联想小新",
            "P005": "Apple Watch",
            "P006": "小米手环",
            "P007": "AirPods Pro",
            "P008": "小米耳机"
        }
        
        self._init_item_features()
        
        print("="*100)
        print("🎯 推荐结果融合服务初始化完成")
        print("="*100)
        print(f"⚡ 快速推荐权重: {self.fast_weight}")
        print(f"🧠 深度推荐权重: {self.deep_weight}")
        print("="*100)

    def _init_item_features(self):
        self.item_features = [
            {"product_id": "P001", "category": "手机", "price": 5999},
            {"product_id": "P002", "category": "手机", "price": 3999},
            {"product_id": "P003", "category": "电脑", "price": 12999},
            {"product_id": "P004", "category": "电脑", "price": 5999},
            {"product_id": "P005", "category": "穿戴", "price": 2499},
            {"product_id": "P006", "category": "穿戴", "price": 899},
            {"product_id": "P007", "category": "耳机", "price": 1999},
            {"product_id": "P008", "category": "耳机", "price": 299}
        ]

    def try_load_deep_model(self):
        try:
            from deepfm_recommender import DeepFMRecommender
            import pandas as pd
            
            self.deep_recommender = DeepFMRecommender(
                self.deep_model_path, 
                self.metadata_path
            )
            
            if self.deep_recommender.load():
                print("✅ DeepFM模型加载成功")
            return True
        except Exception as e:
            print(f"⚠️  DeepFM模型加载失败: {e}")
            print("   将仅使用快速推荐")
            return False

    def get_user_weights(self, user_id):
        activity_level = self.user_activity.get(user_id, 0)
        
        if activity_level < 10:
            return 0.8, 0.2
        elif activity_level < 50:
            return 0.6, 0.4
        else:
            return 0.4, 0.6

    def fuse_recommendations(self, user_id):
        with self.fast_recommendations_lock:
            fast_recs = self.fast_recommendations.get(user_id, [])
        
        with self.deep_recommendations_lock:
            deep_recs = self.deep_recommendations.get(user_id, [])
        
        fast_weight, deep_weight = self.get_user_weights(user_id)
        
        fused_scores = {}
        
        for i, prod_id in enumerate(fast_recs):
            score = fast_weight * (1.0 - i * 0.1)
            fused_scores[prod_id] = fused_scores.get(prod_id, 0) + score
        
        for i, prod_id in enumerate(deep_recs):
            score = deep_weight * (1.0 - i * 0.1)
            fused_scores[prod_id] = fused_scores.get(prod_id, 0) + score
        
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations = []
        seen = set()
        for prod_id, score in sorted_items:
            if prod_id not in seen:
                final_recommendations.append({
                    "product_id": prod_id,
                    "product_name": self.product_names.get(prod_id, prod_id),
                    "score": float(score)
                })
                seen.add(prod_id)
            if len(final_recommendations) >= 5:
                break
        
        return final_recommendations, fast_weight, deep_weight

    def process_fast_recommendation(self, data):
        user_id = data['user_id']
        
        with self.fast_recommendations_lock:
            self.fast_recommendations[user_id] = data.get('recommendations', [])
        
        self.user_activity[user_id] = self.user_activity.get(user_id, 0) + 1

    def process_deep_recommendation(self, data):
        user_id = data['user_id']
        
        with self.deep_recommendations_lock:
            self.deep_recommendations[user_id] = data.get('recommendations', [])

    def generate_and_send_final_recommendation(self, trigger_data):
        user_id = trigger_data['user_id']
        trigger_product = trigger_data.get('trigger_product')
        
        final_recs, fast_w, deep_w = self.fuse_recommendations(user_id)
        
        result = {
            "user_id": user_id,
            "trigger_product": trigger_product,
            "trigger_product_name": self.product_names.get(trigger_product, trigger_product),
            "recommendations": final_recs,
            "timestamp": int(time.time() * 1000),
            "fusion_weights": {
                "fast": fast_w,
                "deep": deep_w
            },
            "source": "hybrid_fusion"
        }
        
        self.producer.send('final_recommendations', value=result)
        
        return result

    def print_fusion_result(self, result):
        time_str = datetime.fromtimestamp(result['timestamp']/1000).strftime('%H:%M:%S')
        
        print("\n" + "="*100)
        print(f"⏰ [{time_str}] 用户: {result['user_id']}")
        print(f"   触发商品: {result['trigger_product_name']}")
        print(f"   融合权重: 快速={result['fusion_weights']['fast']:.1f}, 深度={result['fusion_weights']['deep']:.1f}")
        print(f"\n   📦 最终推荐:")
        
        for i, rec in enumerate(result['recommendations'], 1):
            star = "⭐" if rec['score'] > 0.5 else ""
            print(f"      {i}. {rec['product_name']:20s} - 得分: {rec['score']:.4f} {star}")
        
        print("="*100)

    def fast_consumer_thread(self):
        print("📡 快速推荐消费者线程启动")
        try:
            for message in self.fast_consumer:
                data = message.value
                self.process_fast_recommendation(data)
                
                final_result = self.generate_and_send_final_recommendation(data)
                self.print_fusion_result(final_result)
                
        except Exception as e:
            print(f"❌ 快速推荐消费者错误: {e}")

    def deep_consumer_thread(self):
        print("🧠 深度推荐消费者线程启动")
        try:
            for message in self.deep_consumer:
                data = message.value
                self.process_deep_recommendation(data)
        except Exception as e:
            print(f"❌ 深度推荐消费者错误: {e}")

    def start(self):
        self.is_running = True
        
        fast_thread = threading.Thread(target=self.fast_consumer_thread, daemon=True)
        deep_thread = threading.Thread(target=self.deep_consumer_thread, daemon=True)
        
        fast_thread.start()
        deep_thread.start()
        
        print("\n✅ 推荐融合服务已启动！")
        print("📡 正在监听 fast_recommendations 和 deep_recommendations 主题")
        print("="*100)
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 收到停止信号")
            self.stop()

    def stop(self):
        self.is_running = False
        self.fast_consumer.close()
        self.deep_consumer.close()
        self.producer.close()
        print("\n✅ 推荐融合服务已停止")


if __name__ == "__main__":
    fusion = RecommendationFusion()
    fusion.start()
