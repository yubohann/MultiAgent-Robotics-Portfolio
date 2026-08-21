import numpy as np
import pandas as pd
import pickle
import os


class SimpleDeepFM:
    def __init__(self, model_path='simple_deepfm_model.pkl', metadata_path='simple_metadata.pkl'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metadata = None

    def train(self, interactions_df, item_features_df, epochs=5):
        print("="*80)
        print("🤖 开始训练简易DeepFM推荐模型（基于规则+统计）")
        print("="*80)
        
        user_id_map = {}
        item_id_map = {}
        category_map = {}
        item_popularity = {}
        user_category_pref = {}
        user_interacted_items = {}
        
        user_ids = interactions_df['user_id'].unique()
        item_ids = item_features_df['product_id'].unique()
        categories = item_features_df['category'].unique()
        
        user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        category_map = {cat: i for i, cat in enumerate(categories)}
        
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            rating = row.get('rating', 1.0)
            
            if item_id not in item_popularity:
                item_popularity[item_id] = 0
            item_popularity[item_id] += rating
            
            if user_id not in user_interacted_items:
                user_interacted_items[user_id] = []
            user_interacted_items[user_id].append(item_id)
            
            item_feature = item_features_df[item_features_df['product_id'] == item_id]
            if len(item_feature) > 0:
                category = item_feature.iloc[0]['category']
                if user_id not in user_category_pref:
                    user_category_pref[user_id] = {}
                if category not in user_category_pref[user_id]:
                    user_category_pref[user_id][category] = 0
                user_category_pref[user_id][category] += rating
        
        self.metadata = {
            'user_id_map': user_id_map,
            'item_id_map': item_id_map,
            'category_map': category_map,
            'item_popularity': item_popularity,
            'user_category_pref': user_category_pref,
            'user_interacted_items': user_interacted_items
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"\n✅ 模型训练完成！")
        print(f"📊 用户数: {len(user_ids)}")
        print(f"📊 商品数: {len(item_ids)}")
        print(f"💾 模型保存到: {self.metadata_path}")
        print("="*80)

    def load(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print("✅ 简易DeepFM模型加载成功")
            return True
        else:
            print("❌ 模型文件不存在，需要先训练")
            return False

    def recommend(self, user_id, item_features_df, top_k=5):
        if self.metadata is None:
            print("❌ 模型未加载")
            return []
        
        scores = []
        
        user_cat_pref = self.metadata['user_category_pref'].get(user_id, {})
        interacted_items = set(self.metadata.get('user_interacted_items', {}).get(user_id, []))
        
        for _, item in item_features_df.iterrows():
            item_id = item['product_id']
            
            if item_id in interacted_items:
                continue
                
            category = item['category']
            popularity = self.metadata['item_popularity'].get(item_id, 0)
            cat_pref = user_cat_pref.get(category, 0)
            
            score = 0.4 * (popularity / 10.0) + 0.6 * (cat_pref / 10.0)
            scores.append((item_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def prepare_training_data(interactions):
    data = []
    for interaction in interactions:
        data.append({
            'user_id': interaction['user_id'],
            'product_id': interaction['product_id'],
            'rating': interaction.get('rating', 1.0)
        })
    
    df = pd.DataFrame(data)
    
    item_features = pd.DataFrame([
        {"product_id": "P001", "category": "手机", "price": 5999},
        {"product_id": "P002", "category": "手机", "price": 3999},
        {"product_id": "P003", "category": "电脑", "price": 12999},
        {"product_id": "P004", "category": "电脑", "price": 5999},
        {"product_id": "P005", "category": "穿戴", "price": 2499},
        {"product_id": "P006", "category": "穿戴", "price": 899},
        {"product_id": "P007", "category": "耳机", "price": 1999},
        {"product_id": "P008", "category": "耳机", "price": 299}
    ])
    
    return df, item_features


if __name__ == "__main__":
    sample_interactions = [
        {"user_id": "USER_00001", "product_id": "P001", "rating": 1.0},
        {"user_id": "USER_00001", "product_id": "P007", "rating": 0.8},
        {"user_id": "USER_00001", "product_id": "P005", "rating": 0.7},
        {"user_id": "USER_00002", "product_id": "P002", "rating": 0.9},
        {"user_id": "USER_00002", "product_id": "P006", "rating": 0.6},
        {"user_id": "USER_00002", "product_id": "P008", "rating": 0.8},
        {"user_id": "USER_00003", "product_id": "P001", "rating": 0.7},
        {"user_id": "USER_00003", "product_id": "P002", "rating": 1.0},
        {"user_id": "USER_00003", "product_id": "P007", "rating": 0.8},
        {"user_id": "USER_00003", "product_id": "P008", "rating": 0.6}
    ]
    
    interactions_df, item_features_df = prepare_training_data(sample_interactions)
    
    recommender = SimpleDeepFM()
    recommender.train(interactions_df, item_features_df, epochs=5)
    
    print("\n" + "="*80)
    print("🎯 为用户生成推荐:")
    print("="*80)
    
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
    
    for user_id in ["USER_00001", "USER_00002", "USER_00003"]:
        recommendations = recommender.recommend(user_id, item_features_df, top_k=5)
        print(f"\n👤 用户 {user_id}:")
        for i, (prod_id, score) in enumerate(recommendations, 1):
            print(f"   {i}. {product_names.get(prod_id, prod_id)} - 得分: {score:.4f}")
