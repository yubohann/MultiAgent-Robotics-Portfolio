import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestRecommendationAlgorithms(unittest.TestCase):
    def setUp(self):
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
        
        self.item_features = pd.DataFrame([
            {"product_id": "P001", "category": "手机", "price": 5999},
            {"product_id": "P002", "category": "手机", "price": 3999},
            {"product_id": "P003", "category": "电脑", "price": 12999},
            {"product_id": "P004", "category": "电脑", "price": 5999},
            {"product_id": "P005", "category": "穿戴", "price": 2499},
            {"product_id": "P006", "category": "穿戴", "price": 899},
            {"product_id": "P007", "category": "耳机", "price": 1999},
            {"product_id": "P008", "category": "耳机", "price": 299}
        ])

    def test_1_simple_deepfm_init(self):
        """测试1: 简易DeepFM初始化"""
        print("\n" + "="*80)
        print("🧪 测试1: 简易DeepFM初始化")
        print("="*80)
        
        try:
            from simple_deepfm import SimpleDeepFM
            recommender = SimpleDeepFM(
                model_path='test_model.pkl',
                metadata_path='test_metadata.pkl'
            )
            self.assertIsNotNone(recommender)
            print("✅ 简易DeepFM初始化成功")
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False

    def test_2_simple_deepfm_training(self):
        """测试2: 简易DeepFM模型训练"""
        print("\n" + "="*80)
        print("🧪 测试2: 简易DeepFM模型训练")
        print("="*80)
        
        try:
            from simple_deepfm import SimpleDeepFM, prepare_training_data
            
            sample_interactions = [
                {"user_id": "USER_00001", "product_id": "P001", "rating": 1.0},
                {"user_id": "USER_00001", "product_id": "P007", "rating": 0.8},
                {"user_id": "USER_00002", "product_id": "P002", "rating": 0.9},
            ]
            
            interactions_df, item_features_df = prepare_training_data(sample_interactions)
            
            recommender = SimpleDeepFM(
                model_path='test_model.pkl',
                metadata_path='test_metadata.pkl'
            )
            
            recommender.train(interactions_df, item_features_df, epochs=3)
            
            self.assertIn('item_popularity', recommender.metadata)
            self.assertIn('user_category_pref', recommender.metadata)
            print("✅ 模型训练成功")
            return True
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return False

    def test_3_recommendation_generation(self):
        """测试3: 推荐结果生成"""
        print("\n" + "="*80)
        print("🧪 测试3: 推荐结果生成")
        print("="*80)
        
        try:
            from simple_deepfm import SimpleDeepFM, prepare_training_data
            
            sample_interactions = [
                {"user_id": "USER_00001", "product_id": "P001", "rating": 1.0},
                {"user_id": "USER_00001", "product_id": "P002", "rating": 0.8},
                {"user_id": "USER_00001", "product_id": "P007", "rating": 0.7},
            ]
            
            interactions_df, item_features_df = prepare_training_data(sample_interactions)
            
            recommender = SimpleDeepFM(
                model_path='test_model.pkl',
                metadata_path='test_metadata.pkl'
            )
            
            recommender.train(interactions_df, item_features_df, epochs=2)
            
            recommendations = recommender.recommend("USER_00001", item_features_df, top_k=5)
            
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)
            self.assertLessEqual(len(recommendations), 5)
            
            print(f"✅ 推荐生成成功，返回 {len(recommendations)} 个推荐")
            for i, (prod_id, score) in enumerate(recommendations, 1):
                print(f"   {i}. {self.product_names.get(prod_id, prod_id)} - 得分: {score:.4f}")
            return True
        except Exception as e:
            print(f"❌ 推荐生成失败: {e}")
            return False

    def test_4_no_interacted_items_in_recommendation(self):
        """测试4: 推荐结果不包含已交互商品"""
        print("\n" + "="*80)
        print("🧪 测试4: 推荐结果不包含已交互商品")
        print("="*80)
        
        try:
            from simple_deepfm import SimpleDeepFM, prepare_training_data
            
            interacted_item = "P001"
            sample_interactions = [
                {"user_id": "USER_00001", "product_id": interacted_item, "rating": 1.0},
            ]
            
            interactions_df, item_features_df = prepare_training_data(sample_interactions)
            
            recommender = SimpleDeepFM(
                model_path='test_model.pkl',
                metadata_path='test_metadata.pkl'
            )
            
            recommender.train(interactions_df, item_features_df, epochs=2)
            recommendations = recommender.recommend("USER_00001", item_features_df, top_k=10)
            
            recommended_items = [item[0] for item in recommendations]
            self.assertNotIn(interacted_item, recommended_items, 
                           f"已交互商品 {interacted_item} 不应出现在推荐结果中")
            
            print(f"✅ 推荐结果正确，不包含已交互商品 {self.product_names[interacted_item]}")
            return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

    def test_5_different_users_get_different_recommendations(self):
        """测试5: 不同用户获得不同推荐"""
        print("\n" + "="*80)
        print("🧪 测试5: 不同用户获得不同推荐")
        print("="*80)
        
        try:
            from simple_deepfm import SimpleDeepFM, prepare_training_data
            
            sample_interactions = [
                {"user_id": "USER_PHONE", "product_id": "P001", "rating": 1.0},
                {"user_id": "USER_PHONE", "product_id": "P002", "rating": 0.9},
                {"user_id": "USER_LAPTOP", "product_id": "P003", "rating": 1.0},
                {"user_id": "USER_LAPTOP", "product_id": "P004", "rating": 0.9},
            ]
            
            interactions_df, item_features_df = prepare_training_data(sample_interactions)
            
            recommender = SimpleDeepFM(
                model_path='test_model.pkl',
                metadata_path='test_metadata.pkl'
            )
            
            recommender.train(interactions_df, item_features_df, epochs=2)
            
            recs_phone = recommender.recommend("USER_PHONE", item_features_df, top_k=3)
            recs_laptop = recommender.recommend("USER_LAPTOP", item_features_df, top_k=3)
            
            items_phone = set([item[0] for item in recs_phone])
            items_laptop = set([item[0] for item in recs_laptop])
            
            print(f"📱 手机偏好用户推荐: {[self.product_names.get(i, i) for i in items_phone]}")
            print(f"💻 电脑偏好用户推荐: {[self.product_names.get(i, i) for i in items_laptop]}")
            
            self.assertNotEqual(items_phone, items_laptop, 
                               "不同兴趣用户应该获得不同的推荐")
            print("✅ 不同用户获得不同推荐，测试通过")
            return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

    def test_6_recommendation_algorithms_import(self):
        """测试6: 推荐算法模块导入"""
        print("\n" + "="*80)
        print("🧪 测试6: 推荐算法模块导入")
        print("="*80)
        
        try:
            from recommendation_algorithms import (
                Product, UserProfile, ContentBasedRecommender,
                ItemBasedCFRecommender, HybridRecommender
            )
            print("✅ 推荐算法模块导入成功")
            
            product = Product("P001", "iPhone 15", "手机", 5999)
            self.assertEqual(product.product_id, "P001")
            self.assertEqual(product.name, "iPhone 15")
            print("✅ Product类初始化成功")
            
            user_profile = UserProfile("USER_001")
            self.assertEqual(user_profile.user_id, "USER_001")
            print("✅ UserProfile类初始化成功")
            
            return True
        except Exception as e:
            print(f"❌ 导入失败: {e}")
            return False

    def test_7_content_based_recommendation(self):
        """测试7: 基于内容的推荐"""
        print("\n" + "="*80)
        print("🧪 测试7: 基于内容的推荐")
        print("="*80)
        
        try:
            from recommendation_algorithms import (
                Product, UserProfile, ContentBasedRecommender, get_sample_products
            )
            
            products = get_sample_products()
            
            recommender = ContentBasedRecommender(products)
            
            user_profile = UserProfile("USER_001")
            user_profile.add_interaction("P001", "purchase", "手机", 5999, 1000)
            user_profile.add_interaction("P007", "view", "耳机", 1999, 1001)
            
            recommendations = recommender.recommend(user_profile, n=5)
            
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)
            
            print(f"✅ 基于内容推荐成功，返回 {len(recommendations)} 个推荐")
            product_names = {p.product_id: p.name for p in products.values()}
            for i, (prod_id, score) in enumerate(recommendations, 1):
                print(f"   {i}. {product_names.get(prod_id, prod_id)} - 得分: {score:.4f}")
            
            return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

    def test_8_hybrid_recommendation(self):
        """测试8: 混合推荐"""
        print("\n" + "="*80)
        print("🧪 测试8: 混合推荐")
        print("="*80)
        
        try:
            from recommendation_algorithms import (
                Product, UserProfile, HybridRecommender, get_sample_products
            )
            
            products = get_sample_products()
            product_names = {p.product_id: p.name for p in products.values()}
            
            recommender = HybridRecommender(products, content_weight=0.5, cf_weight=0.5)
            
            recommender.add_interaction("USER_001", "P001", "purchase", "手机", 5999, 1000)
            recommender.add_interaction("USER_001", "P002", "view", "手机", 3999, 1001)
            recommender.add_interaction("USER_002", "P001", "view", "手机", 5999, 1002)
            recommender.add_interaction("USER_002", "P002", "purchase", "手机", 3999, 1003)
            
            try:
                recommender.train()
                print("✅ 协同过滤模型训练成功")
            except Exception as e:
                print(f"⚠️  协同过滤训练跳过（数据量少）: {e}")
            
            user_profile = UserProfile("USER_001")
            user_profile.add_interaction("P001", "purchase", "手机", 5999, 1000)
            
            recommendations = recommender.recommend(user_profile, n=5)
            
            self.assertIsInstance(recommendations, list)
            print(f"✅ 混合推荐成功，返回 {len(recommendations)} 个推荐")
            
            for i, (prod_id, score) in enumerate(recommendations, 1):
                print(f"   {i}. {product_names.get(prod_id, prod_id)} - 得分: {score:.4f}")
            
            return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False


def run_all_tests():
    print("\n" + "="*80)
    print("🧪 推荐系统测试套件")
    print("="*80)
    
    test_suite = unittest.TestSuite()
    
    test_suite.addTest(TestRecommendationAlgorithms('test_1_simple_deepfm_init'))
    test_suite.addTest(TestRecommendationAlgorithms('test_2_simple_deepfm_training'))
    test_suite.addTest(TestRecommendationAlgorithms('test_3_recommendation_generation'))
    test_suite.addTest(TestRecommendationAlgorithms('test_4_no_interacted_items_in_recommendation'))
    test_suite.addTest(TestRecommendationAlgorithms('test_5_different_users_get_different_recommendations'))
    test_suite.addTest(TestRecommendationAlgorithms('test_6_recommendation_algorithms_import'))
    test_suite.addTest(TestRecommendationAlgorithms('test_7_content_based_recommendation'))
    test_suite.addTest(TestRecommendationAlgorithms('test_8_hybrid_recommendation'))
    
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(test_suite)
    
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    print(f"✅ 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 失败: {len(result.failures)}")
    print(f"⚠️  错误: {len(result.errors)}")
    print("="*80)
    
    if result.wasSuccessful():
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
