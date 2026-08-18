from kafka import KafkaConsumer
import json
from datetime import datetime
from collections import defaultdict


class RecommendationMonitor:
    def __init__(self, kafka_bootstrap_servers='localhost:9092'):
        self.consumer = KafkaConsumer(
            'recommendations',
            bootstrap_servers=[kafka_bootstrap_servers],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest'
        )
        
        self.user_recommendation_count = defaultdict(int)
        self.product_recommendation_count = defaultdict(int)
        self.strategy_count = defaultdict(int)
        
        print("="*100)
        print("🎯 高级推荐系统监控面板")
        print("="*100)
    
    def format_user_history(self, history):
        if not history:
            return "无"
        
        lines = []
        if history.get('purchased'):
            lines.append(f"🛒 购买: {', '.join(history['purchased'])}")
        if history.get('carted'):
            lines.append(f"📦 加购: {', '.join(history['carted'])}")
        if history.get('viewed'):
            lines.append(f"👁️  浏览: {', '.join(history['viewed'])}")
        if history.get('top_categories'):
            cat_str = ', '.join([f"{cat}({score:.1f})" for cat, score in history['top_categories']])
            lines.append(f"📊 偏好: {cat_str}")
        
        return '\n      '.join(lines) if lines else "无"
    
    def process_message(self, message):
        data = message.value
        time_str = datetime.fromtimestamp(data['timestamp']/1000).strftime('%H:%M:%S')
        
        user_id = data['user_id']
        trigger_product = data.get('trigger_product_name', '未知')
        trigger_behavior = data.get('trigger_behavior', '未知')
        strategy = data.get('strategy', 'unknown')
        
        self.user_recommendation_count[user_id] += 1
        self.strategy_count[strategy] += 1
        
        for rec in data.get('recommendations', []):
            self.product_recommendation_count[rec['product_name']] += 1
        
        print("\n" + "="*100)
        print(f"⏰ [{time_str}] 用户: {user_id} | 触发: {trigger_product} ({trigger_behavior})")
        print(f"🎯 推荐策略: {strategy.upper()}")
        
        if data.get('user_history'):
            print(f"📋 用户画像:")
            print(f"      {self.format_user_history(data['user_history'])}")
        
        print(f"\n📦 推荐结果:")
        for i, rec in enumerate(data.get('recommendations', []), 1):
            score = rec.get('score', 0)
            star = "⭐" if score > 1.0 else ""
            print(f"   {i}. {rec['product_name']:20s} | {rec['category']:6s} | ¥{rec['price']:6.0f} | 得分: {score:.4f} {star}")
        
        print(f"\n📊 统计信息:")
        print(f"   该用户累计推荐: {self.user_recommendation_count[user_id]} 次")
        print(f"   策略分布: {dict(self.strategy_count)}")
        print(f"   热门推荐商品 Top 5:")
        top_products = sorted(self.product_recommendation_count.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for i, (prod, count) in enumerate(top_products, 1):
            print(f"      {i}. {prod}: {count} 次")
        
        print("="*100)
    
    def start(self):
        print("\n✅ 监控服务已启动，等待推荐结果...\n")
        try:
            for message in self.consumer:
                self.process_message(message)
        except KeyboardInterrupt:
            print("\n👋 停止监控")
        finally:
            self.consumer.close()


def main():
    monitor = RecommendationMonitor()
    monitor.start()


if __name__ == "__main__":
    main()
