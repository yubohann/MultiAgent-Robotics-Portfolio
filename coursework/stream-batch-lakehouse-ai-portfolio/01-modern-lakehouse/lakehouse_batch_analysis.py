import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


STUDENT_ID = os.getenv("STUDENT_ID", "demo000000")
CLASS_NO = os.getenv("CLASS_NO", "0")
STUDENT_NAME = os.getenv("STUDENT_NAME", "REDACTED")

CATALOG = os.getenv("PAIMON_CATALOG", f"paimon_catalog_{STUDENT_ID}")
BUCKET = os.getenv("PAIMON_BUCKET", f"paimon-data-{STUDENT_ID}")
WAREHOUSE = os.getenv("PAIMON_WAREHOUSE", f"s3://{BUCKET}/warehouse")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
ODS_TABLE = os.getenv("ODS_TABLE", f"ods_orders_{STUDENT_ID}")
DWS_TABLE = os.getenv("DWS_TABLE", f"dws_product_sales_{STUDENT_ID}")


def show_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    print("Submitting Spark offline analysis job")
    print(f"student_id={STUDENT_ID}, class_no={CLASS_NO}, student_name={STUDENT_NAME}")
    print(f"catalog={CATALOG}, warehouse={WAREHOUSE}, s3_endpoint={S3_ENDPOINT}")

    spark = (
        SparkSession.builder
        .appName(f"Paimon_Lakehouse_Offline_Analysis_{STUDENT_ID}")
        .master("spark://spark-master:7077")
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.paimon.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", WAREHOUSE)
        .config(f"spark.sql.catalog.{CATALOG}.s3.endpoint", S3_ENDPOINT)
        .config(f"spark.sql.catalog.{CATALOG}.s3.access-key", "admin")
        .config(f"spark.sql.catalog.{CATALOG}.s3.secret-key", "password123")
        .config(f"spark.sql.catalog.{CATALOG}.s3.path.style.access", "true")
        .config("spark.sql.extensions", "org.apache.paimon.spark.extensions.PaimonSparkSessionExtensions")
        .config(
            "spark.jars.packages",
            "org.apache.paimon:paimon-spark-3.3:0.8.0,"
            "org.apache.hadoop:hadoop-aws:3.3.2,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.367",
        )
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )

    try:
        spark.sql(f"USE {CATALOG}.default")

        show_section("1. Catalog and table check")
        spark.sql(f"SHOW DATABASES IN {CATALOG}").show(truncate=False)
        spark.sql("SHOW TABLES").show(truncate=False)

        show_section("2. ODS order detail analysis")
        spark.sql(f"DESCRIBE {ODS_TABLE}").show(truncate=False)
        total_orders = spark.sql(f"SELECT COUNT(*) AS total_orders FROM {ODS_TABLE}").collect()[0][0]
        print(f"total_orders={total_orders}")
        spark.sql(
            f"""
            SELECT *
            FROM {ODS_TABLE}
            ORDER BY order_id DESC
            LIMIT 10
            """
        ).show(truncate=False)
        spark.sql(
            f"""
            SELECT
              status,
              COUNT(*) AS order_count,
              ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {ODS_TABLE}), 2) AS percentage
            FROM {ODS_TABLE}
            GROUP BY status
            ORDER BY order_count DESC
            """
        ).show(truncate=False)

        show_section("3. DWS product sales analysis")
        spark.sql(f"DESCRIBE {DWS_TABLE}").show(truncate=False)
        spark.sql(
            f"""
            SELECT *
            FROM {DWS_TABLE}
            ORDER BY total_amount DESC
            """
        ).show(truncate=False)
        total_sales = spark.sql(f"SELECT SUM(total_amount) AS total_sales FROM {DWS_TABLE}").collect()[0][0]
        print(f"total_sales={total_sales}")

        show_section("4. Streaming and batch result consistency")
        offline_sales = spark.sql(
            f"""
            SELECT
              product_name,
              SUM(amount) AS offline_total_amount,
              COUNT(*) AS paid_order_count
            FROM {ODS_TABLE}
            WHERE status = 'PAID'
            GROUP BY product_name
            """
        )
        streaming_sales = spark.sql(f"SELECT product_name, total_amount FROM {DWS_TABLE}")

        print("offline aggregation:")
        offline_sales.orderBy(col("offline_total_amount").desc()).show(truncate=False)
        print("streaming aggregation:")
        streaming_sales.orderBy(col("total_amount").desc()).show(truncate=False)

        comparison = offline_sales.alias("offline").join(
            streaming_sales.alias("streaming"),
            col("offline.product_name") == col("streaming.product_name"),
            "outer",
        )
        comparison.select(
            col("offline.product_name").alias("product_name"),
            col("offline.offline_total_amount").alias("offline_total"),
            col("streaming.total_amount").alias("streaming_total"),
            (col("offline.offline_total_amount") - col("streaming.total_amount")).alias("difference"),
            col("offline.paid_order_count").alias("paid_order_count"),
        ).orderBy(col("offline_total").desc_nulls_last()).show(truncate=False)

        show_section("5. Data quality check")
        spark.sql(
            f"""
            SELECT
              COUNT(*) AS total_records,
              COUNT(order_id) AS valid_order_id,
              COUNT(product_name) AS valid_product_name,
              COUNT(amount) AS valid_amount,
              COUNT(status) AS valid_status,
              COUNT(create_time) AS valid_create_time,
              COUNT(*) - COUNT(order_id) AS missing_order_id,
              COUNT(*) - COUNT(product_name) AS missing_product_name,
              COUNT(*) - COUNT(amount) AS missing_amount,
              COUNT(*) - COUNT(status) AS missing_status,
              COUNT(*) - COUNT(create_time) AS missing_create_time
            FROM {ODS_TABLE}
            """
        ).show(truncate=False)
        spark.sql(
            f"""
            SELECT order_id, COUNT(*) AS duplicate_count
            FROM {ODS_TABLE}
            GROUP BY order_id
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
            """
        ).show(truncate=False)

        show_section("6. Summary")
        print(f"student_id={STUDENT_ID}, class_no={CLASS_NO}, student_name={STUDENT_NAME}")
        print(f"ods_table={ODS_TABLE}, dws_table={DWS_TABLE}")
        print(f"total_orders={total_orders}, total_sales={total_sales}")
        print("streaming lakehouse offline verification finished")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

