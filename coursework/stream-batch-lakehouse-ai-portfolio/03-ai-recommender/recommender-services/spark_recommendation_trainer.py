from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("RecommendationModelTraining") \
    .master("local[*]") \
    .getOrCreate()

print("="*80)
print("🚀 开始训练商品推荐模型...")
print("="*80)

try:
    df = spark.read \
        .format("paimon") \
        .option("warehouse", "s3://paimon-data/warehouse") \
        .option("s3.endpoint", "http://localhost:9000") \
        .option("s3.access-key", "admin") \
        .option("s3.secret-key", "password123") \
        .option("s3.path.style.access", "true") \
        .load("paimon_catalog.dwd_user_behaviors")

    print(f"✅ 加载历史数据: {df.count()} 条记录")

    def behavior_to_score(behavior_type):
        scores = {"click": 1, "view": 2, "cart": 5, "purchase": 10}
        return scores.get(behavior_type, 1)

    behavior_score_udf = F.udf(behavior_to_score)

    ratings_df = df.withColumn("rating", behavior_score_udf(F.col("behavior_type"))) \
        .groupBy("user_id", "product_id") \
        .agg(F.sum("rating").alias("rating"))

    from pyspark.ml.feature import StringIndexer

    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_int_id")
    product_indexer = StringIndexer(inputCol="product_id", outputCol="product_int_id")

    ratings_df = user_indexer.fit(ratings_df).transform(ratings_df)
    ratings_df = product_indexer.fit(ratings_df).transform(ratings_df)

    ratings_df = ratings_df.select(
        F.col("user_int_id").cast("integer").alias("user"),
        F.col("product_int_id").cast("integer").alias("item"),
        F.col("rating").cast("float")
    )

    (training, test) = ratings_df.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        maxIter=10,
        regParam=0.01,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    print("🔄 正在训练ALS协同过滤模型...")
    model = als.fit(training)
    print("✅ 模型训练完成！")

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"📊 模型RMSE: {rmse:.4f}")

    user_recs = model.recommendForAllUsers(5)
    print("✅ 为所有用户生成Top-5推荐")

    model_path = "/tmp/recommendation_model"
    model.write().overwrite().save(model_path)
    print(f"💾 模型已保存到: {model_path}")

except Exception as e:
    print(f"⚠️  注意: {e}")
    print("💡 提示: 请确保Paimon表中已有数据后再运行模型训练")

spark.stop()
print("\n" + "="*80)
print("🎉 推荐系统模型训练流程完成！")
print("="*80)
