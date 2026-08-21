import time

from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("exp7-minio-partition-pruning-demo000000")
    .config("spark.sql.extensions", "org.apache.paimon.spark.extensions.PaimonSparkSessionExtensions")
    .config("spark.sql.catalog.paimon", "org.apache.paimon.spark.SparkCatalog")
    .config("spark.sql.catalog.paimon.warehouse", "s3a://paimon-data-demo000000/streaming-challenges")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "admin")
    .config("spark.hadoop.fs.s3a.secret.key", "password123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.sql.catalogImplementation", "in-memory")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark.sql("USE paimon.c07")

print("Experiment 7 MinIO Access Bottleneck / Partition Pruning - Student ID demo000000")

print("\n=== Table row count validation ===")
spark.sql(
    """
SELECT 'sales_unpartitioned_demo000000' AS table_name, COUNT(*) AS total_records
FROM sales_unpartitioned_demo000000
UNION ALL
SELECT 'sales_partitioned_demo000000', COUNT(*)
FROM sales_partitioned_demo000000
"""
).show(20, truncate=False)


def timed_query(label: str, sql: str):
    print("\n=== " + label + " ===")
    print(sql)
    start = time.perf_counter()
    rows = spark.sql(sql).collect()
    elapsed = time.perf_counter() - start
    print(f"Elapsed seconds: {elapsed:.3f}")
    for row in rows:
        print(row)


timed_query(
    "7-1 slow query on unpartitioned table",
    "SELECT COUNT(*) AS cnt FROM sales_unpartitioned_demo000000 WHERE dt = '2023-10-15'",
)

print("\n=== 7-1 EXPLAIN unpartitioned table ===")
for row in spark.sql(
    "EXPLAIN FORMATTED SELECT COUNT(*) FROM sales_unpartitioned_demo000000 WHERE dt = '2023-10-15'"
).collect():
    plan = row[0]
    for line in plan.splitlines():
        if "Scan" in line or "Partition" in line or "PushedFilters" in line or "Filters" in line:
            print(line)

timed_query(
    "7-2 fast query on partitioned table",
    "SELECT COUNT(*) AS cnt FROM sales_partitioned_demo000000 WHERE dt = '2023-10-15'",
)

print("\n=== 7-2 EXPLAIN partitioned table ===")
for row in spark.sql(
    "EXPLAIN FORMATTED SELECT COUNT(*) FROM sales_partitioned_demo000000 WHERE dt = '2023-10-15'"
).collect():
    plan = row[0]
    for line in plan.splitlines():
        if "Scan" in line or "Partition" in line or "PushedFilters" in line or "Filters" in line:
            print(line)

spark.stop()
