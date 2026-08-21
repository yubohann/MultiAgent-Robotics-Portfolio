import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import json
from datetime import datetime


class Product:
    def __init__(self, product_id: str, name: str, category: str, price: float, 
                 tags: List[str] = None):
        self.product_id = product_id
        self.name = name
        self.category = category
        self.price = price
        self.tags = tags or []
    
    def to_dict(self):
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "tags": self.tags
        }


class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.view_history: List[str] = []
        self.cart_history: List[str] = []
        self.purchase_history: List[str] = []
        self.category_preference: Dict[str, float] = defaultdict(float)
        self.price_range_preference: Tuple[float, float] = (0, float('inf'))
        self.last_interaction_time: Dict[str, int] = {}
    
    def add_interaction(self, product_id: str, behavior_type: str, 
                       category: str, price: float, timestamp: int):
        if behavior_type == "view":
            self.view_history.append(product_id)
        elif behavior_type == "cart":
            self.cart_history.append(product_id)
        elif behavior_type == "purchase":
            self.purchase_history.append(product_id)
        
        self.category_preference[category] += self._get_behavior_weight(behavior_type)
        self.last_interaction_time[product_id] = timestamp
        self._update_price_preference(price)
    
    def _get_behavior_weight(self, behavior_type: str) -> float:
        weights = {"click": 1, "view": 2, "cart": 5, "purchase": 10}
        return weights.get(behavior_type, 1)
    
    def _update_price_preference(self, price: float):
        all_prices = []
        if self.purchase_history:
            all_prices.extend([p for p in all_prices])
        if not all_prices:
            self.price_range_preference = (price * 0.5, price * 1.5)
        else:
            avg_price = np.mean(all_prices)
            self.price_range_preference = (avg_price * 0.5, avg_price * 1.5)
    
    def get_all_interacted_products(self) -> List[str]:
        return list(set(self.view_history + self.cart_history + self.purchase_history))
    
    def get_top_categories(self, n: int = 3) -> List[Tuple[str, float]]:
        sorted_cats = sorted(self.category_preference.items(), 
                            key=lambda x: x[1], reverse=True)
        return sorted_cats[:n]


class ContentBasedRecommender:
    def __init__(self, products: Dict[str, Product]):
        self.products = products
        self.category_products = defaultdict(list)
        for prod_id, prod in products.items():
            self.category_products[prod.category].append(prod_id)
    
    def recommend(self, user_profile: UserProfile, n: int = 10) -> List[Tuple[str, float]]:
        scores = defaultdict(float)
        interacted = set(user_profile.get_all_interacted_products())
        
        top_categories = user_profile.get_top_categories(3)
        
        for category, cat_score in top_categories:
            for prod_id in self.category_products.get(category, []):
                if prod_id not in interacted:
                    scores[prod_id] += cat_score
        
        min_price, max_price = user_profile.price_range_preference
        for prod_id, score in list(scores.items()):
            prod = self.products[prod_id]
            if min_price <= prod.price <= max_price:
                scores[prod_id] *= 1.5
            else:
                scores[prod_id] *= 0.5
        
        for prod_id in interacted:
            if prod_id in scores:
                del scores[prod_id]
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n]


class ItemBasedCFRecommender:
    def __init__(self, products: Dict[str, Product]):
        self.products = products
        self.item_item_similarity = defaultdict(dict)
        self.user_item_matrix = defaultdict(lambda: defaultdict(float))
        self.is_trained = False
    
    def add_interaction(self, user_id: str, product_id: str, 
                       behavior_type: str, timestamp: int = None):
        weights = {"click": 1, "view": 2, "cart": 5, "purchase": 10}
        weight = weights.get(behavior_type, 1)
        self.user_item_matrix[user_id][product_id] += weight
    
    def train(self):
        product_users = defaultdict(set)
        for user_id, items in self.user_item_matrix.items():
            for item_id in items:
                product_users[item_id].add(user_id)
        
        all_products = list(self.products.keys())
        for i, item_i in enumerate(all_products):
            users_i = product_users.get(item_i, set())
            for j, item_j in enumerate(all_products):
                if i >= j:
                    continue
                users_j = product_users.get(item_j, set())
                intersection = len(users_i & users_j)
                union = len(users_i | users_j)
                if union > 0:
                    jaccard = intersection / union
                    self.item_item_similarity[item_i][item_j] = jaccard
                    self.item_item_similarity[item_j][item_i] = jaccard
        
        self.is_trained = True
        print(f"✅ 协同过滤模型训练完成，计算了 {len(self.item_item_similarity)} 个商品的相似度")
    
    def recommend(self, user_profile: UserProfile, n: int = 10) -> List[Tuple[str, float]]:
        if not self.is_trained:
            print("⚠️  协同过滤模型未训练，返回空推荐")
            return []
        
        scores = defaultdict(float)
        interacted = set(user_profile.get_all_interacted_products())
        
        for item_id in interacted:
            if item_id not in self.item_item_similarity:
                continue
            
            similar_items = self.item_item_similarity[item_id]
            for similar_item, sim_score in similar_items.items():
                if similar_item not in interacted:
                    scores[similar_item] += sim_score
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n]


class HybridRecommender:
    def __init__(self, products: Dict[str, Product], 
                 content_weight: float = 0.4, cf_weight: float = 0.6):
        self.products = products
        self.content_recommender = ContentBasedRecommender(products)
        self.cf_recommender = ItemBasedCFRecommender(products)
        self.content_weight = content_weight
        self.cf_weight = cf_weight
    
    def add_interaction(self, user_id: str, product_id: str, 
                       behavior_type: str, category: str = None, 
                       price: float = None, timestamp: int = None):
        self.cf_recommender.add_interaction(user_id, product_id, behavior_type, timestamp)
    
    def train(self):
        self.cf_recommender.train()
    
    def recommend(self, user_profile: UserProfile, n: int = 10) -> List[Tuple[str, float]]:
        content_recs = self.content_recommender.recommend(user_profile, n * 2)
        cf_recs = self.cf_recommender.recommend(user_profile, n * 2)
        
        combined_scores = defaultdict(float)
        
        for prod_id, score in content_recs:
            combined_scores[prod_id] += score * self.content_weight
        
        for prod_id, score in cf_recs:
            combined_scores[prod_id] += score * self.cf_weight
        
        interacted = set(user_profile.get_all_interacted_products())
        for prod_id in interacted:
            if prod_id in combined_scores:
                del combined_scores[prod_id]
        
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n]


class RecommendationEvaluator:
    def __init__(self, test_data: List[Dict]):
        self.test_data = test_data
    
    def evaluate(self, recommender, user_profiles: Dict[str, UserProfile], 
                 k: int = 5) -> Dict[str, float]:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        user_test_items = defaultdict(set)
        for data in self.test_data:
            if data.get("behavior_type") == "purchase":
                user_test_items[data["user_id"]].add(data["product_id"])
        
        for user_id, test_items in user_test_items.items():
            if user_id not in user_profiles:
                continue
            
            user_profile = user_profiles[user_id]
            recommendations = recommender.recommend(user_profile, k)
            rec_items = [item[0] for item in recommendations]
            
            hits = set(rec_items) & test_items
            
            if len(rec_items) > 0:
                precision = len(hits) / len(rec_items)
                precision_scores.append(precision)
            
            if len(test_items) > 0:
                recall = len(hits) / len(test_items)
                recall_scores.append(recall)
            
            ndcg = self._calculate_ndcg(rec_items, test_items, k)
            ndcg_scores.append(ndcg)
        
        return {
            "precision@{}".format(k): np.mean(precision_scores) if precision_scores else 0,
            "recall@{}".format(k): np.mean(recall_scores) if recall_scores else 0,
            "ndcg@{}".format(k): np.mean(ndcg_scores) if ndcg_scores else 0
        }
    
    def _calculate_ndcg(self, recommendations: List[str], relevant: Set[str], k: int) -> float:
        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)
        
        ideal_relevant = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0


def get_sample_products() -> Dict[str, Product]:
    return {
        "P001": Product("P001", "iPhone 15", "手机", 5999, ["苹果", "5G", "拍照"]),
        "P002": Product("P002", "小米14", "手机", 3999, ["小米", "5G", "性价比"]),
        "P003": Product("P003", "MacBook Pro", "电脑", 12999, ["苹果", "轻薄", "办公"]),
        "P004": Product("P004", "联想小新", "电脑", 5999, ["联想", "性价比", "学生"]),
        "P005": Product("P005", "Apple Watch", "穿戴", 2499, ["苹果", "健康", "运动"]),
        "P006": Product("P006", "小米手环", "穿戴", 899, ["小米", "性价比", "健康"]),
        "P007": Product("P007", "AirPods Pro", "耳机", 1999, ["苹果", "降噪", "无线"]),
        "P008": Product("P008", "小米耳机", "耳机", 299, ["小米", "性价比", "无线"])
    }


def demo():
    print("="*80)
    print("🎯 推荐算法演示系统")
    print("="*80)
    
    products = get_sample_products()
    product_names = {p.product_id: p.name for p in products.values()}
    
    print("\n📦 商品库:")
    for prod_id, prod in products.items():
        print(f"   {prod_id}: {prod.name} ({prod.category}, ¥{prod.price})")
    
    hybrid_rec = HybridRecommender(products, content_weight=0.3, cf_weight=0.7)
    user_profiles = {}
    
    print("\n" + "="*80)
    print("📊 步骤1: 模拟用户交互数据")
    print("="*80)
    
    interactions = [
        ("USER_00001", "P001", "view", 1000),
        ("USER_00001", "P001", "cart", 1001),
        ("USER_00001", "P001", "purchase", 1002),
        ("USER_00001", "P007", "view", 1003),
        ("USER_00001", "P005", "view", 1004),
        ("USER_00002", "P002", "view", 1000),
        ("USER_00002", "P006", "view", 1001),
        ("USER_00002", "P008", "cart", 1002),
        ("USER_00002", "P004", "view", 1003),
        ("USER_00003", "P001", "view", 1000),
        ("USER_00003", "P002", "purchase", 1001),
        ("USER_00003", "P007", "view", 1002),
        ("USER_00003", "P008", "view", 1003),
    ]
    
    for user_id, prod_id, behavior, ts in interactions:
        if user_id not in user_profiles:
            user_profiles[user_id] = UserProfile(user_id)
        
        prod = products[prod_id]
        user_profiles[user_id].add_interaction(prod_id, behavior, prod.category, prod.price, ts)
        hybrid_rec.add_interaction(user_id, prod_id, behavior, prod.category, prod.price, ts)
        print(f"   {user_id} -> {product_names[prod_id]} ({behavior})")
    
    print("\n" + "="*80)
    print("🤖 步骤2: 训练协同过滤模型")
    print("="*80)
    hybrid_rec.train()
    
    print("\n" + "="*80)
    print("🎯 步骤3: 为用户生成推荐")
    print("="*80)
    
    for user_id in ["USER_00001", "USER_00002", "USER_00003"]:
        user_profile = user_profiles[user_id]
        recommendations = hybrid_rec.recommend(user_profile, n=5)
        
        print(f"\n👤 用户 {user_id}:")
        print(f"   历史交互: {[product_names[p] for p in user_profile.get_all_interacted_products()]}")
        print(f"   偏好类别: {user_profile.get_top_categories(2)}")
        print(f"   📦 推荐商品:")
        for i, (prod_id, score) in enumerate(recommendations, 1):
            prod = products[prod_id]
            print(f"      {i}. {product_names[prod_id]} ({prod.category}) - 得分: {score:.4f}")
    
    print("\n" + "="*80)
    print("✅ 演示完成！")
    print("="*80)


if __name__ == "__main__":
    demo()
