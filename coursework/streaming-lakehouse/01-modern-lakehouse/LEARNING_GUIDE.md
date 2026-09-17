# 01-modern-lakehouse 学习导读

作者：REDACTED  
学号：REDACTED
班级序号：REDACTED

这个目录对应的是“现代流批一体数据湖”实验。你可以把它理解成一条完整的数据链路：

`Kafka 造数 -> Flink 实时处理 -> Paimon 落湖到 MinIO -> Spark 离线核对 -> 截图与报告`

如果你想真正学会它，不要先背命令，先把每个文件在链路中的角色搞清楚。

## 先看什么

建议按这个顺序看：

1. `README.md`
2. `docker-compose.yml`
3. `order_stream_producer.py`
4. `flink-dual-stream-job/src/main/java/com/edu/bigdata/FlinkDualStream.java`
5. `lakehouse_batch_analysis.py`
6. `SCREENSHOT_GUIDE.md`
7. `report.md`

如果你先看 `instruction.md` 也可以，但那是课程原始要求，偏“说明书”；上面这条顺序更像“入门路线”。

## 目录总览

- [README.md](./README.md)：这个实验的总说明，告诉你它在做什么。
- [docker-compose.yml](./docker-compose.yml)：一键启动 Kafka、MinIO、Flink、Dinky、Spark。
- [order_stream_producer.py](./order_stream_producer.py)：往 Kafka 里持续发订单数据。
- [dinky_dual_stream_job.sql](./dinky_dual_stream_job.sql)：Flink SQL 版本的作业，适合在 Dinky 里跑。
- [flink-dual-stream-job/](./flink-dual-stream-job/README.md)：Java 版本 Flink 作业，核心逻辑在这里。
- [lakehouse_batch_analysis.py](./lakehouse_batch_analysis.py)：Spark 离线分析脚本，用来核对 Flink 的结果。
- [lib/flink-connectors/](./lib/flink-connectors/)：Flink、Kafka、Paimon、S3、Hadoop 依赖 JAR。
- [docker/spark-local/](./docker/spark-local/)：本地构建 Spark 镜像的上下文。
- [SCREENSHOT_GUIDE.md](./SCREENSHOT_GUIDE.md)：截图顺序和命令。
- [report.md](./report.md)：实验报告正文。

## 这套系统在干什么

### 1. Kafka 负责“进数据”

你先用 [order_stream_producer.py](./order_stream_producer.py) 往 Kafka 发订单。它会生成带学号的订单 ID，比如 `ORD_demo000000_1`，主题名是 `ecommerce_orders_demo000000`。

它的关键点在这里：

- 学号和 Topic 在脚本里是固定拼接的
- `MAX_MESSAGES` 可以控制发多少条
- `KAFKA_BOOTSTRAP_SERVERS` 默认是 `localhost:9092`

你可以把它当成一个“实时业务模拟器”。它不负责计算，只负责持续制造流量。

### 2. Flink 负责“实时算”

Java 版 Flink 作业在 [flink-dual-stream-job/src/main/java/com/edu/bigdata/FlinkDualStream.java](./flink-dual-stream-job/src/main/java/com/edu/bigdata/FlinkDualStream.java)。

它做的事情很直接：

- 从 Kafka 读订单流
- 建 Paimon Catalog
- 把原始明细写到 `ods_orders_demo000000`
- 把按商品聚合后的销售额写到 `dws_product_sales_demo000000`
- 开 checkpoint，保证流任务提交可靠

你在这份代码里能学到三个核心概念：

- `Kafka -> Flink` 的实时消费
- `Flink -> Paimon` 的入湖写入
- `checkpoint` 和事务提交的关系

这份作业里最值得看的几行，是“Kafka 地址、S3 地址、表名、任务名”这些参数是怎么拼出来的。它们都带学号，所以截图和讲解会很干净。

### 3. MinIO 负责“存数据”

MinIO 是对象存储，实验里用它来扮演 S3 数据湖底座。

你不用把它当“数据库”，更像“一个会被 Paimon 写入大量文件的桶”。实验成功后，你会在里面看到：

- `paimon-data-demo000000`
- `warehouse/default.db/ods_orders_demo000000`
- `warehouse/default.db/dws_product_sales_demo000000`

这一步的意义是：你不仅看到了流处理结果，还看到了结果真的落在了湖里。

### 4. Spark 负责“离线核对”

离线分析脚本在 [lakehouse_batch_analysis.py](./lakehouse_batch_analysis.py)。

它会：

- 连接 Paimon Catalog
- 查看数据库和表
- 统计 ODS 的订单总数
- 统计状态分布
- 统计 DWS 的商品销售额
- 把“Flink 流式聚合结果”和“Spark 离线聚合结果”做对比

这个脚本的价值，不只是跑个结果，而是告诉你：

> 流处理和批处理读的是同一份湖上数据，最后能对齐。

### 5. Dinky 负责“图形化操作”

[dinky_dual_stream_job.sql](./dinky_dual_stream_job.sql) 是 SQL 版实现。你如果不想只靠 Java，也可以在 Dinky 里把 SQL 粘进去跑。

它更适合做两件事：

- 演示作业逻辑
- 补截图和补报告

## 一条完整的跑法

你可以把整个实验记成下面这个顺序：

1. `docker compose up -d --scale spark-worker=3`
2. 运行 `order_stream_producer.py` 发消息到 Kafka
3. 用 Maven 打包 `flink-dual-stream-job`
4. 把 JAR 提交到 Flink JobManager
5. 去 MinIO 看 Paimon 文件
6. 跑 `lakehouse_batch_analysis.py`做离线校验
7. 按 [SCREENSHOT_GUIDE.md](./SCREENSHOT_GUIDE.md)截图

如果你只想先“跑通并截图”，先照这个顺序来就行。

## 你会在代码里看到什么

### `docker-compose.yml`

它定义了整个实验环境：

- Kafka
- MinIO
- Flink JobManager
- Flink TaskManager
- Dinky
- Spark Master
- Spark Worker

它里面最值得理解的是三件事：

- Kafka 对外暴露 `localhost:9092`
- 容器内部互连用 `kafka:29092`、`minio:9000`
- Flink/Paimon/S3 相关 JAR 是通过 `lib/flink-connectors/` 挂进去的

你可以把这个文件当成“实验舞台搭建图”。

### `order_stream_producer.py`

这份脚本你要重点理解：

- `STUDENT_ID`
- `TOPIC`
- `BOOTSTRAP_SERVERS`
- `MAX_MESSAGES`
- `producer.send(...)`

它展示的是一个很朴素但很重要的思路：

“用 Python 模拟真实业务订单，把数据变成 Kafka 流。”

### `FlinkDualStream.java`

这是整份实验最核心的文件。

你可以按下面这几个动作去读它：

1. 看参数是怎么拼的
2. 看 Kafka source 是怎么建的
3. 看 Paimon catalog 是怎么建的
4. 看 ODS 和 DWS 两张表是怎么建的
5. 看 `INSERT INTO` 是怎么把数据写进去的

它最重要的结果，不是 Java 语法，而是“你能不能说清楚数据从哪里来、到哪里去、为什么能一致”。

### `lakehouse_batch_analysis.py`

这份脚本是实验后半段的“验算器”。

你重点看：

- `SparkSession.builder`
- `SHOW DATABASES`
- `SHOW TABLES`
- `SELECT COUNT(*)`
- `GROUP BY status`
- ODS 和 DWS 的聚合对比

它最后会输出：

- `total_orders=...`
- `total_sales=...`
- `streaming lakehouse offline verification finished`

这些输出很适合放进报告，也很适合截图。

### `SCREENSHOT_GUIDE.md`

这份文件是“做题路线图”。

它把最容易混乱的部分整理成了：

- 先装依赖
- 先发数据
- 再提交 Flink
- 再检查 Kafka、MinIO、Flink、Spark

如果你卡住，优先看这里，不要自己乱试命令。

## 你最容易踩的坑

1. `ModuleNotFoundError: No module named 'kafka'`

说明当前 Python 环境没装 `kafka-python`，需要在你正在用的 Conda 环境里装。

2. 在 Flink 容器里写 `localhost:9092`

这是错的。容器里要用 `kafka:29092`，因为 `localhost` 只代表容器自己。

3. 把 `paimon-data-demo000000 / warehouse / default.db` 当成终端命令

它是 MinIO 网页里的目录路径，不是 shell 命令。

4. `No running jobs`

说明 Flink 作业还没提交，或者已经挂了。

5. `Could not match any topic-partitions`

通常是 topic 还没被生产者创建，或者 Kafka 地址写错了。

## 学习顺序建议

### 如果你是第一次看

1. 先看 [README.md](./README.md)
2. 再看 [docker-compose.yml](./docker-compose.yml)
3. 再看 [order_stream_producer.py](./order_stream_producer.py)
4. 然后看 [FlinkDualStream.java](./flink-dual-stream-job/src/main/java/com/edu/bigdata/FlinkDualStream.java)
5. 最后看 [lakehouse_batch_analysis.py](./lakehouse_batch_analysis.py)

### 如果你想写报告

1. 对照 [SCREENSHOT_GUIDE.md](./SCREENSHOT_GUIDE.md) 补截图
2. 对照 [report.md](./report.md) 补文字
3. 用 `docker compose ps`、`flink list`、Spark 输出、MinIO 目录做证据

### 如果你想讲给别人听

你可以只讲这四句话：

1. 生产者把订单写进 Kafka
2. Flink 把 Kafka 的订单写进 Paimon
3. MinIO 里能看到 Paimon 的文件
4. Spark 再把同一份数据读出来核对

这四句话讲明白了，这个实验就算真的入门了。

## 外部学习网站

下面这些都建议优先看官方文档：

- Docker Compose: https://docs.docker.com/compose/gettingstarted/
- Apache Kafka Quickstart: https://kafka.apache.org/quickstart/
- Apache Flink Overview: https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/learn-flink/overview/
- Apache Paimon Docs: https://paimon.apache.org/docs/master/
- MinIO Docs: https://min.io/docs/minio/linux/index.html
- Apache Spark Quick Start: https://spark.apache.org/docs/latest/quick-start.html
- VS Code Dev Containers: https://code.visualstudio.com/docs/devcontainers/containers

如果你还想补一条 Python 依赖说明，可以再看：

- kafka-python on PyPI: https://pypi.org/project/kafka-python/

## 一个很实用的理解方式

把这整个实验想成一家小公司的数据管道：

- Kafka 是收银台
- Flink 是实时业务大脑
- MinIO 是仓库
- Paimon 是仓库里的标准货架
- Spark 是财务复核

你只要能把这五个角色讲清楚，实验 1 的大部分内容就能顺下来。
