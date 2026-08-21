package com.edu.bigdata;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaProducer;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;

public class FastRecommender {

    private static final Map<String, ProductInfo> PRODUCT_INFO = new HashMap<>();
    
    static {
        PRODUCT_INFO.put("P001", new ProductInfo("P001", "iPhone 15", "手机", 5999));
        PRODUCT_INFO.put("P002", new ProductInfo("P002", "小米14", "手机", 3999));
        PRODUCT_INFO.put("P003", new ProductInfo("P003", "MacBook Pro", "电脑", 12999));
        PRODUCT_INFO.put("P004", new ProductInfo("P004", "联想小新", "电脑", 5999));
        PRODUCT_INFO.put("P005", new ProductInfo("P005", "Apple Watch", "穿戴", 2499));
        PRODUCT_INFO.put("P006", new ProductInfo("P006", "小米手环", "穿戴", 899));
        PRODUCT_INFO.put("P007", new ProductInfo("P007", "AirPods Pro", "耳机", 1999));
        PRODUCT_INFO.put("P008", new ProductInfo("P008", "小米耳机", "耳机", 299));
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        env.enableCheckpointing(10000);

        Properties consumerProps = new Properties();
        consumerProps.setProperty("bootstrap.servers", "localhost:9092");
        consumerProps.setProperty("group.id", "fast-recommender-group");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "user_behaviors",
            new SimpleStringSchema(),
            consumerProps
        );
        consumer.setStartFromLatest();

        DataStream<String> inputStream = env.addSource(consumer);

        DataStream<String> fastRecStream = inputStream
            .keyBy(json -> new JSONObject(json).getString("user_id"))
            .map(new FastRecommendationFunction());

        Properties producerProps = new Properties();
        producerProps.setProperty("bootstrap.servers", "localhost:9092");

        FlinkKafkaProducer<String> producer = new FlinkKafkaProducer<>(
            "fast_recommendations",
            new SimpleStringSchema(),
            producerProps
        );

        fastRecStream.addSink(producer);
        fastRecStream.print("快速推荐结果: ");

        System.out.println("🚀 Flink快速推荐服务启动...");
        env.execute("快速推荐作业");
    }

    public static class FastRecommendationFunction extends RichMapFunction<String, String> {
        private transient MapState<String, Integer> userHistory;
        private transient MapState<String, Integer> productPopularity;
        private transient ValueState<Long> lastUpdateTime;

        @Override
        public void open(Configuration parameters) {
            MapStateDescriptor<String, Integer> userHistoryDesc = 
                new MapStateDescriptor<>("userHistory", String.class, Integer.class);
            userHistory = getRuntimeContext().getMapState(userHistoryDesc);

            MapStateDescriptor<String, Integer> productPopDesc = 
                new MapStateDescriptor<>("productPopularity", String.class, Integer.class);
            productPopularity = getRuntimeContext().getMapState(productPopDesc);

            ValueStateDescriptor<Long> lastUpdateDesc = 
                new ValueStateDescriptor<>("lastUpdateTime", Long.class);
            lastUpdateTime = getRuntimeContext().getState(lastUpdateDesc);
        }

        @Override
        public String map(String jsonStr) throws Exception {
            JSONObject json = new JSONObject(jsonStr);
            String userId = json.getString("user_id");
            String productId = json.getString("product_id");
            String category = json.getString("category");
            double price = json.getDouble("price");

            String productKey = userId + "_" + productId;
            int count = userHistory.contains(productKey) ? userHistory.get(productKey) : 0;
            userHistory.put(productKey, count + 1);

            int popCount = productPopularity.contains(productId) ? productPopularity.get(productId) : 0;
            productPopularity.put(productId, popCount + 1);

            List<RecommendationCandidate> candidates = new ArrayList<>();

            candidates.addAll(generateCategoryRecommendations(category, userId, 0.5));
            candidates.addAll(generatePopularityRecommendations(0.3));
            candidates.addAll(generatePriceBandRecommendations(price, userId, 0.2));

            Map<String, Double> mergedScores = mergeCandidates(candidates);
            List<String> finalRecommendations = selectTopN(mergedScores, userId, 5);

            JSONObject result = new JSONObject();
            result.put("user_id", userId);
            result.put("trigger_product", productId);
            result.put("recommendations", new JSONArray(finalRecommendations));
            result.put("timestamp", System.currentTimeMillis());
            result.put("recommendation_type", "fast");
            result.put("strategy", "rule_based");

            return result.toString();
        }

        private List<RecommendationCandidate> generateCategoryRecommendations(
                String category, String userId, double weight) {
            List<RecommendationCandidate> candidates = new ArrayList<>();
            Set<String> interacted = getInteractedProducts(userId);

            for (Map.Entry<String, ProductInfo> entry : PRODUCT_INFO.entrySet()) {
                String prodId = entry.getKey();
                ProductInfo info = entry.getValue();
                
                if (info.category.equals(category) && !interacted.contains(prodId)) {
                    candidates.add(new RecommendationCandidate(prodId, weight));
                }
            }
            return candidates;
        }

        private List<RecommendationCandidate> generatePopularityRecommendations(double weight) {
            List<RecommendationCandidate> candidates = new ArrayList<>();
            List<Map.Entry<String, Integer>> sortedProducts = new ArrayList<>();
            
            try {
                for (Iterator<Map.Entry<String, Integer>> it = productPopularity.iterator(); it.hasNext(); ) {
                    sortedProducts.add(it.next());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            sortedProducts.sort((a, b) -> b.getValue().compareTo(a.getValue()));
            
            int count = 0;
            for (Map.Entry<String, Integer> entry : sortedProducts) {
                if (count >= 10) break;
                candidates.add(new RecommendationCandidate(entry.getKey(), weight * (1.0 - count * 0.05)));
                count++;
            }
            return candidates;
        }

        private List<RecommendationCandidate> generatePriceBandRecommendations(
                double targetPrice, String userId, double weight) {
            List<RecommendationCandidate> candidates = new ArrayList<>();
            Set<String> interacted = getInteractedProducts(userId);
            double minPrice = targetPrice * 0.5;
            double maxPrice = targetPrice * 1.5;

            for (Map.Entry<String, ProductInfo> entry : PRODUCT_INFO.entrySet()) {
                String prodId = entry.getKey();
                ProductInfo info = entry.getValue();
                
                if (!interacted.contains(prodId) && 
                    info.price >= minPrice && info.price <= maxPrice) {
                    double priceDiff = Math.abs(info.price - targetPrice) / targetPrice;
                    double priceScore = Math.max(0, 1 - priceDiff);
                    candidates.add(new RecommendationCandidate(prodId, weight * priceScore));
                }
            }
            return candidates;
        }

        private Map<String, Double> mergeCandidates(List<RecommendationCandidate> candidates) {
            Map<String, Double> scores = new HashMap<>();
            for (RecommendationCandidate candidate : candidates) {
                double currentScore = scores.getOrDefault(candidate.productId, 0.0);
                scores.put(candidate.productId, currentScore + candidate.score);
            }
            return scores;
        }

        private List<String> selectTopN(Map<String, Double> scores, String userId, int n) {
            Set<String> interacted = getInteractedProducts(userId);
            
            List<Map.Entry<String, Double>> sortedList = new ArrayList<>(scores.entrySet());
            sortedList.sort((a, b) -> b.getValue().compareTo(a.getValue()));

            List<String> result = new ArrayList<>();
            for (Map.Entry<String, Double> entry : sortedList) {
                if (!interacted.contains(entry.getKey()) && result.size() < n) {
                    result.add(entry.getKey());
                }
            }
            return result;
        }

        private Set<String> getInteractedProducts(String userId) {
            Set<String> interacted = new HashSet<>();
            try {
                for (Iterator<Map.Entry<String, Integer>> it = userHistory.iterator(); it.hasNext(); ) {
                    Map.Entry<String, Integer> entry = it.next();
                    if (entry.getKey().startsWith(userId + "_")) {
                        String productId = entry.getKey().substring((userId + "_").length());
                        interacted.add(productId);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return interacted;
        }
    }

    static class ProductInfo {
        String productId;
        String name;
        String category;
        double price;

        ProductInfo(String productId, String name, String category, double price) {
            this.productId = productId;
            this.name = name;
            this.category = category;
            this.price = price;
        }
    }

    static class RecommendationCandidate {
        String productId;
        double score;

        RecommendationCandidate(String productId, double score) {
            this.productId = productId;
            this.score = score;
        }
    }
}
