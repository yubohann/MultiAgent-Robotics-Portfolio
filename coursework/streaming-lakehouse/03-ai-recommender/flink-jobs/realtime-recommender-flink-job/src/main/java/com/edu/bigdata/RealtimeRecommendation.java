package com.edu.bigdata;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaProducer;
import org.json.JSONObject;

import java.util.*;

public class RealtimeRecommendation {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        Properties consumerProps = new Properties();
        consumerProps.setProperty("bootstrap.servers", "localhost:9092");
        consumerProps.setProperty("group.id", "recommendation-group");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "user_behaviors",
            new SimpleStringSchema(),
            consumerProps
        );
        consumer.setStartFromLatest();

        DataStream<String> inputStream = env.addSource(consumer);

        DataStream<String> recommendationStream = inputStream
            .keyBy(json -> new JSONObject(json).getString("user_id"))
            .map(new RecommendationFunction());

        Properties producerProps = new Properties();
        producerProps.setProperty("bootstrap.servers", "localhost:9092");

        FlinkKafkaProducer<String> producer = new FlinkKafkaProducer<>(
            "recommendations",
            new SimpleStringSchema(),
            producerProps
        );

        recommendationStream.addSink(producer);
        recommendationStream.print("推荐结果: ");

        System.out.println("🚀 实时推荐服务启动...");
        env.execute("实时商品推荐作业");
    }

    public static class RecommendationFunction extends RichMapFunction<String, String> {
        private Map<String, List<String>> userHistory;
        private Map<String, List<String>> categoryProducts;

        @Override
        public void open(Configuration parameters) {
            userHistory = new HashMap<>();
            categoryProducts = new HashMap<>();
            
            categoryProducts.put("手机", Arrays.asList("P001", "P002"));
            categoryProducts.put("电脑", Arrays.asList("P003", "P004"));
            categoryProducts.put("穿戴", Arrays.asList("P005", "P006"));
            categoryProducts.put("耳机", Arrays.asList("P007", "P008"));
        }

        @Override
        public String map(String jsonStr) {
            JSONObject json = new JSONObject(jsonStr);
            String userId = json.getString("user_id");
            String currentProduct = json.getString("product_id");
            String category = json.getString("category");

            userHistory.computeIfAbsent(userId, k -> new ArrayList<>()).add(currentProduct);

            List<String> recommendations = generateRecommendations(userId, category);

            JSONObject result = new JSONObject();
            result.put("user_id", userId);
            result.put("trigger_product", currentProduct);
            result.put("recommendations", recommendations);
            result.put("timestamp", System.currentTimeMillis());
            result.put("recommendation_strategy", "category_based");

            return result.toString();
        }

        private List<String> generateRecommendations(String userId, String category) {
            List<String> recs = new ArrayList<>();
            List<String> history = userHistory.getOrDefault(userId, new ArrayList<>());
            
            List<String> categoryItems = categoryProducts.getOrDefault(category, new ArrayList<>());
            for (String item : categoryItems) {
                if (!history.contains(item) && recs.size() < 3) {
                    recs.add(item);
                }
            }
            
            for (Map.Entry<String, List<String>> entry : categoryProducts.entrySet()) {
                if (!entry.getKey().equals(category)) {
                    for (String item : entry.getValue()) {
                        if (recs.size() < 5) {
                            recs.add(item);
                        }
                    }
                }
            }
            
            return recs;
        }
    }
}
