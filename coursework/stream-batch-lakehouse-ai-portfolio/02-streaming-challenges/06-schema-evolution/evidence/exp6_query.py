from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("exp6-schema-evolution-demo000000")
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

print("Experiment 6 Schema Evolution - Student ID demo000000")
spark.sql("USE paimon.c06")

queries = [
    (
        "6-2 DESCRIBE user_profile_demo000000 after schema evolution",
        "DESCRIBE user_profile_demo000000",
    ),
    (
        "6-2 old rows keep age=NULL, new rows have age values",
        """
SELECT user_id, username, email, create_time, age
FROM user_profile_demo000000
ORDER BY user_id
""",
    ),
    (
        "6-2 old/new data coexistence count",
        """
SELECT CASE WHEN age IS NULL THEN 'Old data' ELSE 'New data' END AS data_type,
       COUNT(*) AS cnt
FROM user_profile_demo000000
GROUP BY CASE WHEN age IS NULL THEN 'Old data' ELSE 'New data' END
ORDER BY data_type
""",
    ),
]

for title, sql in queries:
    print("\n=== " + title + " ===")
    spark.sql(sql).show(50, truncate=False)

spark.stop()
