package com.edu.bigdata.challenge;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.util.Collector;
import org.json.JSONObject;

import java.util.Properties;

/**
 * Data Skew Demo - Reproduce hot-key backpressure.
 * Student: REDACTED  ID: demo000000
 * Job name: SkewDemo_demo000000
 */
public class SkewDemo {

    private static final String STUDENT_ID = "demo000000";
    private static final String TOPIC = "click_stream_" + STUDENT_ID;
    private static final String JOB_NAME = "SkewDemo_" + STUDENT_ID;

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        env.enableCheckpointing(5000);
        env.getCheckpointConfig().setCheckpointStorage("file:///tmp/flink-checkpoints-" + STUDENT_ID);

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "kafka:29092");
        properties.setProperty("group.id", "skew-demo-group-" + STUDENT_ID);

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            TOPIC,
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> stream = env.addSource(consumer);

        DataStream<Tuple3<String, Long, Long>> clickStream = stream.map(
            new MapFunction<String, Tuple3<String, Long, Long>>() {
                @Override
                public Tuple3<String, Long, Long> map(String jsonStr) {
                    JSONObject json = new JSONObject(jsonStr);
                    String itemId = json.getString("item_id");
                    return new Tuple3<>(itemId, 1L, System.currentTimeMillis());
                }
            }
        );

        // Direct keyBy on item_id causes skew: iPhone15 gets ~90% of data
        DataStream<Tuple3<String, Long, Long>> resultStream = clickStream
            .keyBy(value -> value.f0)
            .window(TumblingProcessingTimeWindows.of(Time.seconds(10)))
            .apply(new WindowFunction<Tuple3<String, Long, Long>, Tuple3<String, Long, Long>, String, TimeWindow>() {
                @Override
                public void apply(String itemId, TimeWindow window,
                                  Iterable<Tuple3<String, Long, Long>> input,
                                  Collector<Tuple3<String, Long, Long>> out) {
                    long count = 0;
                    for (Tuple3<String, Long, Long> record : input) {
                        count += record.f1;
                    }
                    out.collect(new Tuple3<>(itemId, count, window.getEnd()));
                }
            });

        resultStream.print("SkewDemo_" + STUDENT_ID + " result: ");

        System.out.println("Student: REDACTED  ID: " + STUDENT_ID);
        System.out.println("Starting " + JOB_NAME);
        System.out.println("Topic: " + TOPIC + " | Parallelism: 4 | Window: 10s tumbling");
        env.execute(JOB_NAME);
    }
}
