package com.edu.bigdata;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.StatementSet;
import org.apache.flink.table.api.TableResult;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class FlinkDualStream {
    private static final String DEFAULT_STUDENT_ID = "demo000000";
    private static final String DEFAULT_CLASS_NO = "REDACTED";
    private static final String DEFAULT_STUDENT_NAME = "REDACTED";

    public static void main(String[] args) throws Exception {
        String studentId = env("STUDENT_ID", DEFAULT_STUDENT_ID);
        String classNo = env("CLASS_NO", DEFAULT_CLASS_NO);
        String studentName = env("STUDENT_NAME", DEFAULT_STUDENT_NAME);

        String topic = env("KAFKA_TOPIC", "ecommerce_orders_" + studentId);
        String bucket = env("PAIMON_BUCKET", "paimon-data-" + studentId);
        String catalog = env("PAIMON_CATALOG", "paimon_catalog_" + studentId);
        String odsTable = env("ODS_TABLE", "ods_orders_" + studentId);
        String dwsTable = env("DWS_TABLE", "dws_product_sales_" + studentId);
        String sourceTable = "kafka_source_" + studentId;
        String jobName = env("FLINK_JOB_NAME", "DualStreamJob_" + studentId);

        String kafkaBootstrapServers = env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        String s3Endpoint = env("S3_ENDPOINT", "http://localhost:9000");

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        env.enableCheckpointing(10_000L);
        env.getCheckpointConfig().setCheckpointStorage("file:///tmp/flink-checkpoints-" + studentId);

        EnvironmentSettings settings = EnvironmentSettings.newInstance().inStreamingMode().build();
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env, settings);
        tableEnv.getConfig().getConfiguration().setString("pipeline.name", jobName);

        tableEnv.executeSql(
                "CREATE TEMPORARY TABLE " + sourceTable + " ("
                        + "order_id STRING, "
                        + "product_name STRING, "
                        + "amount DOUBLE, "
                        + "status STRING, "
                        + "create_time STRING"
                        + ") WITH ("
                        + "'connector' = 'kafka', "
                        + "'topic' = '" + topic + "', "
                        + "'properties.bootstrap.servers' = '" + kafkaBootstrapServers + "', "
                        + "'properties.group.id' = 'flink_dual_stream_" + studentId + "', "
                        + "'scan.startup.mode' = 'earliest-offset', "
                        + "'format' = 'json', "
                        + "'json.ignore-parse-errors' = 'true'"
                        + ")"
        );

        tableEnv.executeSql(
                "CREATE CATALOG " + catalog + " WITH ("
                        + "'type' = 'paimon', "
                        + "'warehouse' = 's3://" + bucket + "/warehouse', "
                        + "'s3.endpoint' = '" + s3Endpoint + "', "
                        + "'s3.access-key' = 'admin', "
                        + "'s3.secret-key' = 'password123', "
                        + "'s3.path.style.access' = 'true'"
                        + ")"
        );
        tableEnv.executeSql("USE CATALOG " + catalog);
        tableEnv.executeSql("CREATE DATABASE IF NOT EXISTS `default`");
        tableEnv.executeSql("USE `default`");

        tableEnv.executeSql(
                "CREATE TABLE IF NOT EXISTS " + odsTable + " ("
                        + "order_id STRING PRIMARY KEY NOT ENFORCED, "
                        + "product_name STRING, "
                        + "amount DOUBLE, "
                        + "status STRING, "
                        + "create_time STRING"
                        + ") WITH ('bucket' = '4')"
        );

        tableEnv.executeSql(
                "CREATE TABLE IF NOT EXISTS " + dwsTable + " ("
                        + "product_name STRING PRIMARY KEY NOT ENFORCED, "
                        + "total_amount DOUBLE"
                        + ") WITH ('bucket' = '4')"
        );

        StatementSet statementSet = tableEnv.createStatementSet();
        statementSet.addInsertSql(
                "INSERT INTO " + odsTable + " "
                        + "SELECT order_id, product_name, amount, status, create_time "
                        + "FROM default_catalog.default_database." + sourceTable
        );
        statementSet.addInsertSql(
                "INSERT INTO " + dwsTable + " "
                        + "SELECT product_name, SUM(amount) AS total_amount "
                        + "FROM default_catalog.default_database." + sourceTable + " "
                        + "WHERE status = 'PAID' "
                        + "GROUP BY product_name"
        );

        System.out.println("Starting " + jobName);
        System.out.println("student_id=" + studentId + ", class_no=" + classNo + ", student_name=" + studentName);
        System.out.println("kafka_topic=" + topic + ", kafka_bootstrap_servers=" + kafkaBootstrapServers);
        System.out.println("paimon_catalog=" + catalog + ", bucket=" + bucket + ", s3_endpoint=" + s3Endpoint);
        System.out.println("ods_table=" + odsTable + ", dws_table=" + dwsTable);

        TableResult result = statementSet.execute();
        result.await();
    }

    private static String env(String key, String fallback) {
        String value = System.getenv(key);
        return value == null || value.isBlank() ? fallback : value;
    }
}

