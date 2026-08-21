import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import os


class DeepFMDataset(Dataset):
    def __init__(self, data_df, user_features, item_features):
        self.data = data_df
        self.user_features = user_features
        self.item_features = item_features
        
        self.user_id_map = {uid: i for i, uid in enumerate(user_features['user_id'].unique())}
        self.item_id_map = {iid: i for i, iid in enumerate(item_features['product_id'].unique())}
        self.category_map = {cat: i for i, cat in enumerate(item_features['category'].unique())}
        
        self.num_users = len(self.user_id_map)
        self.num_items = len(self.item_id_map)
        self.num_categories = len(self.category_map)
        
        self.feature_dims = {
            'user_id': self.num_users,
            'item_id': self.num_items,
            'category': self.num_categories,
            'price': 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        user_id = self.user_id_map.get(row['user_id'], 0)
        item_id = self.item_id_map.get(row['product_id'], 0)
        
        item_feature = self.item_features[self.item_features['product_id'] == row['product_id']]
        if len(item_feature) > 0:
            category = self.category_map.get(item_feature.iloc[0]['category'], 0)
            price = item_feature.iloc[0]['price'] / 10000.0
        else:
            category = 0
            price = 0.0
        
        label = float(row['rating'])
        
        return {
            'user_id': torch.LongTensor([user_id]),
            'item_id': torch.LongTensor([item_id]),
            'category': torch.LongTensor([category]),
            'price': torch.FloatTensor([price]),
            'label': torch.FloatTensor([label])
        }


class FM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=8):
        super(FM, self).__init__()
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.ModuleDict()
        for feature_name, dim in feature_dims.items():
            if dim > 1:
                self.embeddings[feature_name] = nn.Embedding(dim, embedding_dim)
        
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.ModuleDict()
        for feature_name, dim in feature_dims.items():
            if dim > 1:
                self.linear[feature_name] = nn.Embedding(dim, 1)
            else:
                self.linear[feature_name] = nn.Linear(1, 1, bias=False)

    def forward(self, features):
        first_feature = next(iter(features.values()))
        batch_size = first_feature.shape[0]
        linear_term = self.bias.expand(batch_size)
        
        for feature_name in self.feature_dims:
            if self.feature_dims[feature_name] > 1:
                feat = features[feature_name].squeeze(1)
                linear_term = linear_term + self.linear[feature_name](feat).squeeze(-1)
            else:
                feat = features[feature_name]
                linear_term = linear_term + self.linear[feature_name](feat).squeeze(-1)
        
        fm_embeddings = []
        for feature_name in self.feature_dims:
            if self.feature_dims[feature_name] > 1:
                feat = features[feature_name].squeeze(1)
                fm_embeddings.append(self.embeddings[feature_name](feat))
        
        if fm_embeddings:
            fm_input = torch.stack(fm_embeddings, dim=1)
            
            sum_of_square = torch.sum(fm_input, dim=1) ** 2
            square_of_sum = torch.sum(fm_input ** 2, dim=1)
            fm_term = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1)
        else:
            fm_term = torch.zeros_like(linear_term)
        
        return linear_term + fm_term


class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=8, dnn_dims=[64, 32]):
        super(DeepFM, self).__init__()
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        
        self.fm = FM(feature_dims, embedding_dim)
        
        self.embeddings = nn.ModuleDict()
        for feature_name, dim in feature_dims.items():
            if dim > 1:
                self.embeddings[feature_name] = nn.Embedding(dim, embedding_dim)
        
        dnn_input_dim = sum(1 if dim == 1 else embedding_dim for dim in feature_dims.values())
        
        dnn_layers = []
        input_dim = dnn_input_dim
        for dim in dnn_dims:
            dnn_layers.append(nn.Linear(input_dim, dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(0.3))
            input_dim = dim
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)
        
        self.final = nn.Linear(2, 1)

    def forward(self, features):
        fm_output = self.fm(features)
        
        dnn_embeddings = []
        for feature_name in self.feature_dims:
            if self.feature_dims[feature_name] > 1:
                feat = features[feature_name].squeeze(1)
                dnn_embeddings.append(self.embeddings[feature_name](feat))
            else:
                feat = features[feature_name]
                dnn_embeddings.append(feat)
        
        dnn_input = torch.cat(dnn_embeddings, dim=1)
        dnn_output = self.dnn(dnn_input).squeeze(-1)
        
        combined = torch.stack([fm_output, dnn_output], dim=1)
        output = self.final(combined).squeeze(-1)
        
        return torch.sigmoid(output)


class DeepFMRecommender:
    def __init__(self, model_path='deepfm_model.pth', metadata_path='metadata.pkl'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, interactions_df, item_features_df, epochs=10, batch_size=256, lr=0.001):
        print("="*80)
        print("🤖 开始训练DeepFM推荐模型")
        print("="*80)
        
        user_features_df = interactions_df[['user_id']].drop_duplicates().reset_index(drop=True)
        dataset = DeepFMDataset(interactions_df, user_features_df, item_features_df)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        feature_dims = dataset.feature_dims
        self.model = DeepFM(feature_dims).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\n📊 数据集大小: 训练 {train_size}, 验证 {val_size}")
        print(f"🎯 特征维度: {feature_dims}")
        print(f"🏋️  开始训练 {epochs} 轮...\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                features = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device).squeeze(-1)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    labels = batch['label'].to(self.device).squeeze(-1)
                    
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"   ✅ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")
        
        self.metadata = {
            'user_id_map': dataset.user_id_map,
            'item_id_map': dataset.item_id_map,
            'category_map': dataset.category_map,
            'feature_dims': dataset.feature_dims
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print("\n" + "="*80)
        print(f"🎉 DeepFM模型训练完成！")
        print(f"💾 模型保存到: {self.model_path}")
        print(f"💾 元数据保存到: {self.metadata_path}")
        print("="*80)

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.model = DeepFM(self.metadata['feature_dims']).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ DeepFM模型加载成功")
            return True
        else:
            print("❌ 模型文件不存在，需要先训练")
            return False

    def recommend(self, user_id, item_features_df, top_k=5):
        if self.model is None or self.metadata is None:
            print("❌ 模型未加载")
            return []
        
        self.model.eval()
        
        user_idx = self.metadata['user_id_map'].get(user_id, 0)
        
        scores = []
        with torch.no_grad():
            for _, item in item_features_df.iterrows():
                item_idx = self.metadata['item_id_map'].get(item['product_id'], 0)
                category_idx = self.metadata['category_map'].get(item['category'], 0)
                price = item['price'] / 10000.0
                
                features = {
                    'user_id': torch.LongTensor([[user_idx]]).to(self.device),
                    'item_id': torch.LongTensor([[item_idx]]).to(self.device),
                    'category': torch.LongTensor([[category_idx]]).to(self.device),
                    'price': torch.FloatTensor([[price]]).to(self.device)
                }
                
                score = self.model(features).item()
                scores.append((item['product_id'], score))
        
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
    
    recommender = DeepFMRecommender()
    recommender.train(interactions_df, item_features_df, epochs=5)
    
    print("\n" + "="*80)
    print("🎯 为用户生成推荐:")
    print("="*80)
    
    for user_id in ["USER_00001", "USER_00002", "USER_00003"]:
        recommendations = recommender.recommend(user_id, item_features_df, top_k=5)
        print(f"\n👤 用户 {user_id}:")
        for i, (prod_id, score) in enumerate(recommendations, 1):
            print(f"   {i}. {prod_id} - 得分: {score:.4f}")
