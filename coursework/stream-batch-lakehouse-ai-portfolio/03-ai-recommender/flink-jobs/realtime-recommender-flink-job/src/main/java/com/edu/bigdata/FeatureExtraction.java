package com.edu.bigdata;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.json.JSONObject;

import java.util.Properties;

public class FeatureExtraction {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        env.enableCheckpointing(10000);

        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env, EnvironmentSettings.inStreamingMode());

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "feature-group");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "user_behaviors",
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> rawStream = env.addSource(consumer);

        DataStream<FeatureData> featureStream = rawStream.map(new MapFunction<String, FeatureData>() {
            @Override
            public FeatureData map(String jsonStr) {
                JSONObject json = new JSONObject(jsonStr);
                return new FeatureData(
                    json.getString("user_id"),
                    json.getString("product_id"),
                    json.getString("category"),
                    json.getDouble("price"),
                    json.getString("behavior_type"),
                    json.getLong("timestamp")
                );
            }
        });

        featureStream.print("特征数据: ");

        tEnv.executeSql(
            "CREATE CATALOG paimon_catalog WITH (" +
            "  'type' = 'paimon', 'warehouse' = 's3://paimon-data/warehouse', " +
            "  's3.endpoint' = 'http://localhost:9000', 's3.access-key' = 'admin', " +
            "  's3.secret-key' = 'password123', 's3.path.style.access' = 'true'" +
            ")"
        );
        tEnv.executeSql("USE CATALOG paimon_catalog");

        tEnv.executeSql(
            "CREATE TABLE IF NOT EXISTS dwd_user_behaviors (" +
            "  user_id STRING, product_id STRING, category STRING, " +
            "  price DOUBLE, behavior_type STRING, timestamp BIGINT, " +
            "  PRIMARY KEY (user_id, product_id, timestamp) NOT ENFORCED" +
            ")"
        );

        System.out.println("🚀 实时特征提取服务启动...");
        env.execute("实时特征提取作业");
    }

    public static class FeatureData {
        public String userId;
        public String productId;
        public String category;
        public double price;
        public String behaviorType;
        public long timestamp;

        public FeatureData() {}
        public FeatureData(String userId, String productId, String category, 
                          double price, String behaviorType, long timestamp) {
            this.userId = userId;
            this.productId = productId;
            this.category = category;
            this.price = price;
            this.behaviorType = behaviorType;
            this.timestamp = timestamp;
        }
    }
}
