# Runtime connectors (not version-controlled)

The Flink / Paimon connector jars under `lib/flink-connectors/` are runtime
dependencies. They are **not** committed to the repository (see `.gitignore`)
and should be downloaded from Maven Central before running the dual-stream
lakehouse job.

Required artifacts:

| Artifact | Version |
| --- | --- |
| `flink-sql-connector-kafka` | 3.0.1-1.18 |
| `flink-s3-fs-hadoop` | 1.18.0 |
| `hadoop-hdfs-client` | 3.3.4 |
| `paimon-flink` | 1.18-0.8.0 |

Example (from the `lib/` directory):

```bash
curl -LO https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka-3.0.1-1.18.jar
curl -LO https://repo1.maven.org/maven2/org/apache/flink/flink-s3-fs-hadoop-1.18.0.jar
curl -LO https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-hdfs-client-3.3.4.jar
curl -LO https://repo1.maven.org/maven2/org/apache/paimon/paimon-flink-1.18-0.8.0.jar
```