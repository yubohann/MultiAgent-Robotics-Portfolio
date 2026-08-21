package com.edu.bigdata.challenge;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.json.JSONObject;

import java.time.Duration;
import java.util.Properties;

/**
 * Late & Out-of-Order Data Demo.
 * Student: REDACTED  ID: demo000000
 * Job name: LateDataDemo_demo000000
 *
 * Uses: Watermark (5s bounded out-of-orderness)
 *       Allowed Lateness (60s)
 *       Side Output for severely late data
 */
public class LateDataDemo {

    private static final String STUDENT_ID = "demo000000";
    private static final String TOPIC = "sensor_data_" + STUDENT_ID;
    private static final String JOB_NAME = "LateDataDemo_" + STUDENT_ID;

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);
        env.enableCheckpointing(5000);

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "kafka:29092");
        properties.setProperty("group.id", "late-data-demo-" + STUDENT_ID);

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            TOPIC,
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> stream = env.addSource(consumer);

        DataStream<Tuple4<String, String, Double, Long>> sensorStream = stream.map(
            new MapFunction<String, Tuple4<String, String, Double, Long>>() {
                @Override
                public Tuple4<String, String, Double, Long> map(String jsonStr) {
                    JSONObject json = new JSONObject(jsonStr);
                    return new Tuple4<>(
                        json.getString("sensor_id"),
                        json.getString("record_id"),
                        json.getDouble("temperature"),
                        json.getLong("timestamp")
                    );
                }
            }
        );

        final OutputTag<Tuple4<String, String, Double, Long>> lateDataTag =
            new OutputTag<Tuple4<String, String, Double, Long>>("late-data") {};

        SingleOutputStreamOperator<String> windowedStream = sensorStream
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<Tuple4<String, String, Double, Long>>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, timestamp) -> event.f3)
            )
            .keyBy(value -> value.f0)
            .window(TumblingEventTimeWindows.of(Time.seconds(30)))
            .allowedLateness(Time.seconds(60))
            .sideOutputLateData(lateDataTag)
            .apply(new WindowFunction<Tuple4<String, String, Double, Long>, String, String, TimeWindow>() {
                @Override
                public void apply(String sensorId, TimeWindow window,
                                  Iterable<Tuple4<String, String, Double, Long>> input,
                                  Collector<String> out) {
                    int count = 0;
                    double sumTemp = 0;
                    for (Tuple4<String, String, Double, Long> record : input) {
                        count++;
                        sumTemp += record.f2;
                    }
                    double avgTemp = count > 0 ? sumTemp / count : 0;
                    String result = String.format(
                        "Sensor: %s | Window: [%s - %s] | Records: %d | Avg Temp: %.2f C",
                        sensorId,
                        new java.util.Date(window.getStart()),
                        new java.util.Date(window.getEnd()),
                        count, avgTemp
                    );
                    out.collect(result);
                }
            });

        windowedStream.print("Normal: ");
        windowedStream.getSideOutput(lateDataTag).print("LATE-DATA-ALERT: ");

        System.out.println("Student: REDACTED  ID: " + STUDENT_ID);
        System.out.println("Starting " + JOB_NAME);
        System.out.println("Watermark: 5s bounded | AllowedLateness: 60s | Window: 30s event-time");
        env.execute(JOB_NAME);
    }
}
