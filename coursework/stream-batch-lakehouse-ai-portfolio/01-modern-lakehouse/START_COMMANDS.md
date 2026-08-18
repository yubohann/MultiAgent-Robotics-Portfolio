# 01-modern-lakehouse 启动命令清单

这份清单只放最常用、最容易忘的命令。默认在 WSL2 里执行。

## 0. 进入目录并启动 Docker

```bash
cd "/path/to/stream-batch-lakehouse-ai-portfolio/01-modern-lakehouse"
sudo service docker start
```

## 1. 启动实验环境

```bash
docker compose up -d --scale spark-worker=3
docker compose ps
docker ps
```

## 2. 安装 Python 生产者依赖

```bash
python3 -m pip install --upgrade kafka-python==2.3.1
python3 -c "from kafka import KafkaProducer; import kafka; print(kafka.__version__)"
```

## 3. 运行 Kafka 订单生产者

```bash
MAX_MESSAGES=30 KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python3 order_stream_producer.py
```

## 4. 检查 Kafka

```bash
docker exec bigdata-kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
docker exec bigdata-kafka /opt/kafka/bin/kafka-get-offsets.sh --topic ecommerce_orders_demo000000 --bootstrap-server localhost:9092
docker exec bigdata-kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group flink_dual_stream_demo000000
```

## 5. 打包 Flink 作业

```bash
cd flink-dual-stream-job
mvn -DskipTests package
```

## 6. 提交 Flink 作业

```bash
docker cp target/flink-dual-stream-job-1.0-SNAPSHOT.jar bigdata-flink-jm:/tmp/flink-dual-stream-job.jar
docker exec \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:29092 \
  -e S3_ENDPOINT=http://minio:9000 \
  -e STUDENT_ID=demo000000 \
  -e class_no=0 \
  bigdata-flink-jm /opt/flink/bin/flink run -d \
  -c com.edu.bigdata.FlinkDualStream \
  /tmp/flink-dual-stream-job.jar
```

## 7. 查看 Flink 状态

```bash
docker exec bigdata-flink-jm /opt/flink/bin/flink list
docker logs --tail 80 bigdata-flink-tm
docker logs --tail 80 bigdata-flink-jm
```

## 8. 运行 Spark 离线验证

```bash
docker exec bigdata-spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.paimon:paimon-spark-3.3:0.8.0,org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.12.367 \
  /opt/spark-apps/lakehouse_batch_analysis.py
```

## 9. 常见检查

```bash
docker run --rm --network bigdata-network-demo000000 --entrypoint /bin/sh minio/mc -c \
  'mc alias set local http://minio:9000 admin password123 >/dev/null && mc ls -r local/paimon-data-demo000000/warehouse | head -80'
```

## 10. 停止实验环境

```bash
docker compose down
sudo service docker stop
```

## 11. 页面入口

- Flink Web UI: `http://localhost:8081`
- MinIO Web Console: `http://localhost:9001`
- Dinky Web UI: `http://localhost:8888`

## 12. 最重要的规则

- 宿主机访问 Kafka 用 `localhost:9092`
- 容器内访问 Kafka 用 `kafka:29092`
- 容器内访问 MinIO 用 `http://minio:9000`
- 提交 Flink 作业前先确保 `docker compose ps` 全部是 `Up`
