from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("exp5-exactly-once-demo000000")
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

print("Experiment 5 Exactly-Once / Duplicate Data - Student ID demo000000")
spark.sql("USE paimon.c05")

queries = [
    (
        "5-1 / 5-2 comparison: append table has duplicates, PK table deduplicates",
        """
SELECT 'Append Table' AS t,
       COUNT(*) AS total_records,
       COUNT(DISTINCT order_id) AS unique_orders,
       COUNT(*) - COUNT(DISTINCT order_id) AS dup_count
FROM append_table_demo000000
UNION ALL
SELECT 'PK Table',
       COUNT(*),
       COUNT(DISTINCT order_id),
       COUNT(*) - COUNT(DISTINCT order_id)
FROM pk_table_demo000000
""",
    ),
    (
        "5-1 append_table_demo000000 duplicate order_id examples",
        """
SELECT order_id, COUNT(*) AS dup_count
FROM append_table_demo000000
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY dup_count DESC, order_id
LIMIT 10
""",
    ),
    (
        "5-2 pk_table_demo000000 duplicate check, expected empty",
        """
SELECT order_id, COUNT(*) AS dup_count
FROM pk_table_demo000000
GROUP BY order_id
HAVING COUNT(*) > 1
LIMIT 10
""",
    ),
]

for title, sql in queries:
    print("\n=== " + title + " ===")
    spark.sql(sql).show(50, truncate=False)

spark.stop()
