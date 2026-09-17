SET 'pipeline.name' = 'DualStreamJob_demo000000_Dinky';

CREATE TEMPORARY TABLE kafka_source_demo000000 (
  order_id STRING,
  product_name STRING,
  amount DOUBLE,
  status STRING,
  create_time STRING
) WITH (
  'connector' = 'kafka',
  'topic' = 'ecommerce_orders_demo000000',
  'properties.bootstrap.servers' = 'kafka:29092',
  'properties.group.id' = 'dinky_dual_stream_demo000000',
  'scan.startup.mode' = 'earliest-offset',
  'format' = 'json',
  'json.ignore-parse-errors' = 'true'
);

CREATE CATALOG paimon_catalog_demo000000 WITH (
  'type' = 'paimon',
  'warehouse' = 's3://paimon-data-demo000000/warehouse',
  's3.endpoint' = 'http://minio:9000',
  's3.access-key' = 'admin',
  's3.secret-key' = 'password123',
  's3.path.style.access' = 'true'
);

USE CATALOG paimon_catalog_demo000000;
CREATE DATABASE IF NOT EXISTS `default`;
USE `default`;

CREATE TABLE IF NOT EXISTS ods_orders_demo000000 (
  order_id STRING PRIMARY KEY NOT ENFORCED,
  product_name STRING,
  amount DOUBLE,
  status STRING,
  create_time STRING
) WITH (
  'bucket' = '4'
);

CREATE TABLE IF NOT EXISTS dws_product_sales_demo000000 (
  product_name STRING PRIMARY KEY NOT ENFORCED,
  total_amount DOUBLE
) WITH (
  'bucket' = '4'
);

EXECUTE STATEMENT SET
BEGIN
  INSERT INTO ods_orders_demo000000
  SELECT order_id, product_name, amount, status, create_time
  FROM default_catalog.default_database.kafka_source_demo000000;

  INSERT INTO dws_product_sales_demo000000
  SELECT product_name, SUM(amount) AS total_amount
  FROM default_catalog.default_database.kafka_source_demo000000
  WHERE status = 'PAID'
  GROUP BY product_name;
END;
