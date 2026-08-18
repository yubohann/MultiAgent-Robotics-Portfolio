from kafka import KafkaConsumer, KafkaProducer
import json
from datetime import datetime
from recommendation_algorithms import (
    Product, UserProfile, HybridRecommender, get_sample_products
)
import threading
import time


class RealtimeRecommendationService:
    def __init__(self, kafka_bootstrap_servers='localhost:9092'):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        
        self.products = get_sample_products()
        self.product_names = {p.product_id: p.name for p in self.products.values()}
        
        self.recommender = HybridRecommender(
            self.products,
            content_weight=0.4,
            cf_weight=0.6
        )
        
        self.user_profiles = {}
        self.user_profiles_lock = threading.Lock()
        
        self.consumer = KafkaConsumer(
            'user_behaviors',
            bootstrap_servers=[kafka_bootstrap_servers],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='realtime-recommendation-group'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.is_running = False
        self.model_trained = False
        self.interaction_count = 0
        self.last_train_interaction_count = 0
        
        print("="*80)
        print("🚀 实时推荐服务初始化完成")
        print("="*80)
        print(f"📦 商品库: {len(self.products)} 个商品")
        for prod_id, prod in self.products.items():
            print(f"   {prod_id}: {prod.name} ({prod.category}, ¥{prod.price})")
        print("="*80)
    
    def update_user_profile(self, data: dict):
        user_id = data['user_id']
        product_id = data['product_id']
        behavior_type = data['behavior_type']
        category = data['category']
        price = data['price']
        timestamp = data['timestamp']
        
        with self.user_profiles_lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id)
            
            user_profile = self.user_profiles[user_id]
            user_profile.add_interaction(
                product_id, behavior_type, category, price, timestamp
            )
            
            self.recommender.add_interaction(
                user_id, product_id, behavior_type, category, price, timestamp
            )

            self.interaction_count += 1
            self.train_model_if_needed()

    def train_model_if_needed(self, min_new_interactions: int = 10):
        if self.model_trained and self.interaction_count - self.last_train_interaction_count < min_new_interactions:
            return

        user_count = len(self.user_profiles)
        if user_count == 0 or self.interaction_count == 0:
            return

        print(f"\n🔄 [{datetime.now().strftime('%H:%M:%S')}] 训练协同过滤模型...")
        print(f"   用户数: {user_count}, 交互数: {self.interaction_count}")
        self.recommender.train()
        self.model_trained = True
        self.last_train_interaction_count = self.interaction_count
        print("   ✅ 协同过滤模型训练完成")
    
    def generate_recommendations(self, user_id: str, n: int = 5) -> dict:
        with self.user_profiles_lock:
            if user_id not in self.user_profiles:
                return {
                    "user_id": user_id,
                    "recommendations": [],
                    "strategy": "cold_start",
                    "message": "新用户，使用热门推荐"
                }
            
            user_profile = self.user_profiles[user_id]
            recommendations = self.recommender.recommend(user_profile, n)
            
            rec_list = []
            for prod_id, score in recommendations:
                prod = self.products[prod_id]
                rec_list.append({
                    "product_id": prod_id,
                    "product_name": prod.name,
                    "category": prod.category,
                    "price": prod.price,
                    "score": float(score)
                })
            
            return {
                "user_id": user_id,
                "trigger_product": None,
                "recommendations": rec_list,
                "strategy": "hybrid",
                "user_history": {
                    "viewed": [self.product_names.get(p, p) for p in user_profile.view_history[-5:]],
                    "carted": [self.product_names.get(p, p) for p in user_profile.cart_history[-3:]],
                    "purchased": [self.product_names.get(p, p) for p in user_profile.purchase_history],
                    "top_categories": user_profile.get_top_categories(3)
                },
                "timestamp": int(time.time() * 1000)
            }
    
    def train_model_periodically(self, interval_seconds: int = 60):
        while self.is_running:
            try:
                time.sleep(interval_seconds)
                
                with self.user_profiles_lock:
                    self.train_model_if_needed()
            except Exception as e:
                print(f"❌ 模型训练出错: {e}")
    
    def process_message(self, message):
        data = message.value
        user_id = data['user_id']
        product_id = data['product_id']
        behavior_type = data['behavior_type']
        product_name = self.product_names.get(product_id, product_id)
        
        time_str = datetime.fromtimestamp(data['timestamp']/1000).strftime('%H:%M:%S')
        print(f"\n📥 [{time_str}] 用户 {user_id} - {product_name} - {behavior_type}")
        
        self.update_user_profile(data)
        
        if behavior_type in ['view', 'cart', 'purchase']:
            rec_result = self.generate_recommendations(user_id, n=5)
            
            rec_result['trigger_product'] = product_id
            rec_result['trigger_product_name'] = product_name
            rec_result['trigger_behavior'] = behavior_type
            
            self.producer.send('recommendations', value=rec_result)
            
            print(f"🎯 为 {user_id} 生成推荐:")
            for i, rec in enumerate(rec_result['recommendations'], 1):
                print(f"   {i}. {rec['product_name']} ({rec['category']}) - 得分: {rec['score']:.4f}")
            
            if rec_result.get('user_history'):
                hist = rec_result['user_history']
                if hist['purchased']:
                    print(f"   🛒 购买历史: {', '.join(hist['purchased'])}")
                if hist['top_categories']:
                    print(f"   📊 偏好类别: {hist['top_categories']}")
    
    def start(self):
        self.is_running = True
        
        training_thread = threading.Thread(
            target=self.train_model_periodically,
            args=(60,),
            daemon=True
        )
        training_thread.start()
        
        print("\n✅ 实时推荐服务已启动，正在监听用户行为...")
        print("="*80)
        
        try:
            for message in self.consumer:
                self.process_message(message)
        except KeyboardInterrupt:
            print("\n👋 收到停止信号...")
        finally:
            self.stop()
    
    def stop(self):
        self.is_running = False
        self.consumer.close()
        self.producer.close()
        print("\n✅ 实时推荐服务已停止")


def main():
    service = RealtimeRecommendationService()
    service.start()


if __name__ == "__main__":
    main()
