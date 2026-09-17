# 《云计算与大数据处理》 现代流批一体数据湖 (Streaming Lakehouse) 综合实验手册

**实验环境特别说明：**
请严格遵循手册中的版本号进行操作，以避免底层组件的兼容性报错。

---

## 准备阶段、 课前环境基建：开发环境配置 (双方案可选)

在编写任何代码之前，我们需要准备好大数据的开发环境和底层通讯包。这里提供两种开发环境配置方案，**请任选其一**。

### 方案 A：传统本地开发模式 (依托宿主机 Java 环境)
这是最经典的开发模式，代码在本地编译，通过跨版本编译技术打包。
1. **安装环境：** 确保电脑已安装 JDK 17 和 Maven。
2. **VS Code 插件：** 在 VS Code 扩展商店安装 `Extension Pack for Java` 和 `Python` 插件。
3. *(由于 Java 17 的模块强封装限制，后续在本地运行 Flink 时需配置专门的反射逃逸参数，详见第三阶段)*。

### 方案 B：Dev Containers 云原生隔离模式 (工业方案，强烈推荐，彻底告别环境报错)
这是工业界目前最先进的“容器化开发”模式。您的本地电脑**不需要安装任何 JDK 和 Maven**，VS Code 会将核心引擎“注射”到一个纯净的 Java 11 容器中，实现开发环境与生产环境的 100% 统一。
1. **前置条件：** 电脑已安装 Docker Desktop 并正在运行。
2. **VS Code 插件：** 安装微软官方扩展 `Dev Containers`。
3. **开启魔法：**
   * 在您的项目根目录下，新建文件夹 `.devcontainer`。
   * 在该文件夹内新建文件 `devcontainer.json`，完整粘贴以下配置：
     ```json
     {
         "name": "BigData Java11 Env",
         "image": "mcr.microsoft.com/devcontainers/java:11",
         "features": {
             "ghcr.io/devcontainers/features/maven:1": {
                 "version": "latest"
             }
         },
         "customizations": {
             "vscode": {
                 "extensions": [
                     "vscjava.vscode-java-pack",
                     "ms-python.python"
                 ]
             }
         }
     }
     ```
   * 在 VS Code 左下角点击绿色的 `><` 图标，选择 **“Reopen in Container (在容器中重新打开)”**。
   * *效果：VS Code 界面仍在本地，但终端、JDK 11、Maven 均已在容器内准备就绪！*

### 核心依赖共享夹准备 (双方案均需执行)
1. 在电脑任意位置新建一个主文件夹，命名为 `bigdata-lab`。**接下来的所有操作均在此文件夹内完成。**
2. 在 `bigdata-lab` 内部，新建一个名为 `flink-jars` 的文件夹。
3. 请通过浏览器或 Maven 仓库，下载以下 **3 个核心 Jar 包**，并放入 `flink-jars` 文件夹中（这些包将挂载给 Docker，让集群认识数据湖格式）：
    * `flink-sql-connector-kafka-3.0.1-1.18.jar`
    * `paimon-flink-1.18-0.8.0.jar`
    * `flink-s3-fs-hadoop-1.18.0.jar`

---

## 第一阶段：全局底层集群拉起 (Docker Compose)

在 `bigdata-lab` 根目录下新建文件 `compose.yaml`。本配置不仅拉起了流计算所需的基建，还额外引入了 **Spark 独立集群 (Standalone Cluster)**，让您在单机上体验分布式算力。

```yaml
# 新版 Docker Compose 已全面废弃顶层 version 字段，直接从 services 开始定义即可
services: 

  # ==========================================
  # 1. 消息中间件：Apache Kafka (官方原版，完美兼容 Apple M 系列芯片)
  # ==========================================
  kafka:
    image: apache/kafka:3.7.0 # 使用官方纯正镜像，杜绝第三方魔改引发的兼容性故障
    container_name: bigdata-kafka # 固定容器名称，方便后期使用 docker logs 查看运行日志
    ports:
      - "9092:9092" # 端口映射：将容器内的 9092 暴露给 Mac 宿主机的 9092 端口
    environment:
      - KAFKA_NODE_ID=1 # 集群节点唯一标识，单机实验模式设为 1
      - KAFKA_PROCESS_ROLES=broker,controller # 核心机制：启用 KRaft 模式，让该节点既做数据存储(broker)又做集群调度(controller)，彻底抛弃笨重的 Zookeeper
      
      # 【核心网络隔离配置：定义三扇监听门】
      # 29092：面向 Docker 内部局域网的后门，供 Flink/Dinky 等内部容器通信
      # 9093： 面向集群内部控制器选举的专用通道
      # 9092： 面向外部 Mac 宿主机的正门，供 Python 模拟数据脚本接入
      - KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:29092,CONTROLLER://0.0.0.0:9093,EXTERNAL://0.0.0.0:9092
      
      # 【核心网络穿透配置：告诉客户端该用什么地址连进来】
      # 如果是 Docker 内的兄弟组件，请通过 kafka:29092 连我；如果是外面的电脑，请通过 localhost:9092 连我
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:29092,EXTERNAL://localhost:9092
      - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,EXTERNAL:PLAINTEXT # 协议映射：全部使用明文不加密传输
      - KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER # 指定控制器的通信通道名称
      - KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka:9093 # 选举配置：单机模式下投票节点只有自己（修正Docker内部网络解析）
      
      # 【容错与副本配置：单机模式防报错补丁】
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 # 强制将偏移量主题的副本数降为 1（否则默认需 3 台机器）
      - KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 # 强制将事务日志副本降为 1
      - KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 # 最小同步副本数降为 1
      - KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0 # 消费者组启动无延迟，立刻开始分配数据
    networks:
      - bigdata-network # 加入自定义的虚拟局域网

  # ==========================================
  # 2. 对象存储：MinIO (数据湖的物理底座，AWS S3 协议的开源平替)
  # ==========================================
  minio:
    image: minio/minio:latest # 始终拉取最新的 MinIO 镜像
    container_name: bigdata-minio
    ports:
      - "9000:9000" # API 通信端口：Flink 写文件、Spark 读文件，全部走这个端口
      - "9001:9001" # Web UI 端口：我们在浏览器查看湖仓里的目录结构，走这个端口
    environment:
      - MINIO_ROOT_USER=admin # 初始化 MinIO 超级管理员账号
      - MINIO_ROOT_PASSWORD=password123 # 初始化 MinIO 超级管理员密码
    command: server /data --console-address ":9001" # 启动命令：作为服务端运行，数据存放在容器的 /data 目录，Web控制台绑在 9001
    networks:
      - bigdata-network

  # ==========================================
  # 3. 实时计算：Flink 集群 (JobManager 核心大脑 / 包工头)
  # ==========================================
  jobmanager:
    image: flink:1.18.0-scala_2.12-java11 # 生产环境强推 Java 11 镜像，避免 JDK 17+ 的强封装反射报错
    container_name: bigdata-flink-jm
    ports:
      - "8081:8081" # 映射 Flink 极其漂亮的 Web 监控大屏
    command: jobmanager # 指定该容器作为调度节点运行
    volumes:
      # 【极其关键的插件挂载】：将宿主机的 flink-jars 目录下的Jar包直接复制到容器的 lib 目录，让 Flink 动态认识 Kafka 和 Paimon
      - ./flink-jars:/opt/flink/lib
    networks:
      - bigdata-network

  # ==========================================
  # 4. 实时计算：Flink 集群 (TaskManager 干活的工人)
  # ==========================================
  taskmanager:
    image: flink:1.18.0-scala_2.12-java11
    container_name: bigdata-flink-tm
    depends_on:
      - jobmanager # 依赖控制：必须等大脑(JobManager)启动后，工人才启动，防止失联报错
    command: taskmanager # 指定该容器作为计算执行节点运行
    volumes:
      # 工人同样需要加载这些 Jar 包才能真正执行读写操作
      - ./flink-jars:/opt/flink/lib
    networks:
      - bigdata-network

  # ==========================================
  # 5. 敏捷数据平台：Dinky (企业级一站式 Flink SQL 网页开发中台)
  # ==========================================
  dinky:
    image: dinkydocker/dinky-standalone:1.0.3 # 独立版镜像自带轻量级 H2 数据库，免去额外部署 MySQL 的麻烦，极度适合教学
    container_name: bigdata-dinky
    ports:
      - "8888:8888" # Dinky 网页端的对外访问端口
    depends_on:
      - jobmanager # 依赖 Flink 集群就绪
    volumes:
      # Dinky 在你敲击键盘做 SQL 语法检查时，也需要读取这些外部依赖包进行校验
      # 注意：Dinky 会自动加载 custom_jars 目录下的 Jar 包
      - ./flink-jars:/opt/dinky/custom_jars
    networks:
      - bigdata-network

  # ==========================================
  # 6. 离线分析：Spark 独立集群 (Master 资源大管家)
  # ==========================================
  spark-master:
    image: apache/spark:3.3.2 # 锁死 3.3.2 版本，这是与 Paimon 0.8.0 配合最稳定、无冲突的黄金版本
    container_name: bigdata-spark-master
    # 启动官方的 Master 守护进程，并明确指定主机名和绑定的两个核心端口
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master --host spark-master --port 7077 --webui-port 8080
    ports:
      - "8080:8080" # Spark 集群 Web 大屏（您可以直观看到有几个 Worker 在待命）
      - "7077:7077" # 内部 RPC 调度端口（包工头接单的专属电话号码）
    volumes:
      # 【突破 NAT 隔离墙的挂载】：将宿主机当前的实验目录挂载到容器内，方便后续我们直接用 docker exec 在内部提交脚本
      - ./:/opt/spark-apps 
    networks:
      - bigdata-network

  # ==========================================
  # 7. 离线分析：Spark 独立集群 (Worker 节点/打工人)
  # ==========================================
  spark-worker:
    image: apache/spark:3.3.2
    depends_on:
      - spark-master
    # 启动 Worker 进程，并告诉它大管家的通信地址 (向 spark-master 的 7077 端口报到)
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
    environment:
      - SPARK_WORKER_CORES=2 # 资源隔离：限制每个工人最多只能提供 2 个 CPU 核心
      - SPARK_WORKER_MEMORY=2G # 资源隔离：限制每个工人最多只能使用 2G 内存
    networks:
      - bigdata-network

# 统一声明网络拓扑
networks: 
  bigdata-network:
    driver: bridge # 创建桥接网络，让上述 7 个容器能在内部通过服务名(如 kafka, minio)互相 Ping 通
```
**启动与扩容魔法：** 在终端执行以下命令拉起所有服务，并**动态申请 3 个 Spark Worker (模拟 3 台分布式计算节点)**：
```bash
docker compose up -d --scale spark-worker=3
```

---

## 第二阶段：源头活水 (Python 模拟电商订单流入)

在 `bigdata-lab` 目录下新建 `mock_data_producer.py`。
*(执行前，请在终端执行 `pip install kafka-python`)*

```python
import json 
import time 
import random 
from kafka import KafkaProducer 
from datetime import datetime 

# 1. 初始化 Kafka 生产者实例
producer = KafkaProducer(
    # 宿主机连接 Kafka 暴露的 9092 外网大门
    bootstrap_servers=['localhost:9092'], 
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

products = ["iPhone 15", "MacBook Pro", "iPad Air", "AirPods"]
statuses = ["UNPAID", "PAID", "SHIPPED"]

print("🚀 业务系统上线，开始向 Kafka 发送实时订单...")
order_id_counter = 1 

# 2. 开启死循环产生数据
while True:
    data = {
        "order_id": f"ORD_{order_id_counter}", 
        "product_name": random.choice(products), 
        "amount": round(random.uniform(100.0, 20000.0), 2), 
        "status": random.choice(statuses), 
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    }
    producer.send('ecommerce_orders', value=data)
    print(f"发送成功: {data}") 
    order_id_counter += 1 
    time.sleep(1) # 休眠1秒，稳定输出水流
```
**开始供水：** 运行 `python mock_data_producer.py`，保持终端开启。

---

## 第三阶段：流式入湖核心 (双轨并行的 Flink 实战)

### 轨道 A：VS Code 本地 Java 底层工程化实战

**1. 初始化 Maven 项目**
在 `bigdata-lab` 目录下，新建文件夹 `flink-java-project`。通过 VS Code 单独打开该文件夹。

**2. 编写带有交叉编译配置的 `pom.xml`**
在项目根目录新建 `pom.xml`，这是解决异构环境的核心密码：
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.edu.bigdata</groupId> 
    <artifactId>flink-java-project</artifactId> 
    <version>1.0-SNAPSHOT</version> 

    <properties>
        <maven.compiler.release>11</maven.compiler.release>
        <flink.version>1.18.0</flink.version> 
    </properties>

    <dependencies>
        <dependency><groupId>org.apache.flink</groupId><artifactId>flink-clients</artifactId><version>${flink.version}</version></dependency>
        <dependency><groupId>org.apache.flink</groupId><artifactId>flink-table-api-java-bridge</artifactId><version>${flink.version}</version></dependency>
        <dependency><groupId>org.apache.flink</groupId><artifactId>flink-table-planner-loader</artifactId><version>${flink.version}</version></dependency>
        <dependency><groupId>org.apache.flink</groupId><artifactId>flink-table-runtime</artifactId><version>${flink.version}</version></dependency>
        <dependency><groupId>org.apache.flink</groupId><artifactId>flink-sql-connector-kafka</artifactId><version>3.0.1-1.18</version></dependency>
        <dependency><groupId>org.apache.paimon</groupId><artifactId>paimon-flink-1.18</artifactId><version>0.8.0</version></dependency>
        <dependency><groupId>org.apache.flink</groupId><artifactId>flink-s3-fs-hadoop</artifactId><version>${flink.version}</version></dependency>
    </dependencies>
</project>
```

**3. 配置 Java 反射逃逸 (`.vscode/launch.json`)**
*(注：如果您采用的是 Dev Containers 方案 B，您已经在 Java 11 容器内，无需此配置)*。
如果使用宿主机 Java 17，新建 `.vscode/launch.json` 突破强封装限制：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "java",
            "name": "运行 Flink 双流应用",
            "request": "launch",
            "mainClass": "com.edu.bigdata.FlinkDualStream",
            "projectName": "flink-java-project",
            "vmArgs": "--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.math=ALL-UNNAMED --add-opens java.base/java.time=ALL-UNNAMED --add-opens java.base/java.net=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED"
        }
    ]
}
```

**4. 编写流处理逻辑 `FlinkDualStream.java`**
新建 `src/main/java/com/edu/bigdata/FlinkDualStream.java`，代码支持通过环境变量灵活配置连接地址：
```java
package com.edu.bigdata;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.StatementSet;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class FlinkDualStream {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1); 
        // 开启 Checkpoint，Paimon 依赖此机制实现事务写盘
        env.enableCheckpointing(10000); 
        
        TableEnvironment tEnv = StreamTableEnvironment.create(env, EnvironmentSettings.inStreamingMode());

        // 从环境变量读取配置，支持灵活切换运行环境
        String kafkaBootstrapServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        String s3Endpoint = System.getenv().getOrDefault("S3_ENDPOINT", "http://localhost:9000");

        tEnv.executeSql(
            "CREATE TEMPORARY TABLE kafka_source (order_id STRING, product_name STRING, amount DOUBLE, status STRING, create_time STRING) " +
            "WITH ('connector' = 'kafka', 'topic' = 'ecommerce_orders', 'properties.bootstrap.servers' = '" + kafkaBootstrapServers + "', 'scan.startup.mode' = 'latest-offset', 'format' = 'json')"
        );
        
        tEnv.executeSql(
            "CREATE CATALOG paimon_catalog WITH (" +
            "  'type' = 'paimon', 'warehouse' = 's3://paimon-data/warehouse', " +
            "  's3.endpoint' = '" + s3Endpoint + "', 's3.access-key' = 'admin', 's3.secret-key' = 'password123', 's3.path.style.access' = 'true'" +
            ")"
        );
        tEnv.executeSql("USE CATALOG paimon_catalog"); 

        tEnv.executeSql("CREATE TABLE IF NOT EXISTS ods_orders (order_id STRING PRIMARY KEY NOT ENFORCED, product_name STRING, amount DOUBLE, status STRING)");
        tEnv.executeSql("CREATE TABLE IF NOT EXISTS dws_product_sales (product_name STRING PRIMARY KEY NOT ENFORCED, total_amount DOUBLE)");

        StatementSet stmtSet = tEnv.createStatementSet();
        stmtSet.addInsertSql("INSERT INTO ods_orders SELECT order_id, product_name, amount, status FROM default_catalog.default_database.kafka_source");
        stmtSet.addInsertSql("INSERT INTO dws_product_sales SELECT product_name, SUM(amount) as total_amount FROM default_catalog.default_database.kafka_source WHERE status = 'PAID' GROUP BY product_name");
        
        System.out.println("🚀 Flink 引擎启动，开始源源不断向 MinIO Paimon 写入双流数据...");
        System.out.println("📡 Kafka 连接地址: " + kafkaBootstrapServers);
        System.out.println("📦 MinIO 连接地址: " + s3Endpoint);
        stmtSet.execute(); 
    }
}
```

**5. 配置运行环境变量**
根据您的运行环境，设置以下环境变量：

- **宿主机运行（方案 A）**：无需额外配置，默认使用 `localhost:9092` 和 `http://localhost:9000`
- **Dev Containers 运行（方案 B）**：在 `.vscode/launch.json` 中添加环境变量：
  ```json
  {
      "version": "0.2.0",
      "configurations": [
          {
              "type": "java",
              "name": "运行 Flink 双流应用",
              "request": "launch",
              "mainClass": "com.edu.bigdata.FlinkDualStream",
              "projectName": "flink-java-project",
              "vmArgs": "--add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.math=ALL-UNNAMED --add-opens java.base/java.time=ALL-UNNAMED --add-opens java.base/java.net=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED",
              "env": {
                  "KAFKA_BOOTSTRAP_SERVERS": "kafka:29092",
                  "S3_ENDPOINT": "http://minio:9000"
              }
          }
      ]
  }
  ```

**运行：** 运行 `main` 方法，控制台挂起即代表数据正在成功入湖。

### 轨道 B：Dinky 网页端 GUI 交互
1. 浏览器打开 `http://localhost:8888` (默认账号 admin, 密码 admin)。
2. 【集群中心】 -> 注册集群实例：名称 `LocalFlink`，地址 `jobmanager:8081`。
3. 【数据开发】 -> 新建 Flink SQL，粘贴以下代码并点击运行。*(注意：内网必须用服务名通信)*

```sql
CREATE TEMPORARY TABLE kafka_source (order_id STRING, product_name STRING, amount DOUBLE, status STRING, create_time STRING) WITH ('connector' = 'kafka', 'topic' = 'ecommerce_orders', 'properties.bootstrap.servers' = 'kafka:29092', 'scan.startup.mode' = 'latest-offset', 'format' = 'json');
CREATE CATALOG paimon_catalog WITH ('type' = 'paimon', 'warehouse' = 's3://paimon-data/warehouse', 's3.endpoint' = 'http://minio:9000', 's3.access-key' = 'admin', 's3.secret-key' = 'password123', 's3.path.style.access' = 'true');
USE CATALOG paimon_catalog;
CREATE TABLE IF NOT EXISTS ods_orders (order_id STRING PRIMARY KEY NOT ENFORCED, product_name STRING, amount DOUBLE, status STRING);
CREATE TABLE IF NOT EXISTS dws_product_sales (product_name STRING PRIMARY KEY NOT ENFORCED, total_amount DOUBLE);
EXECUTE STATEMENT SET BEGIN
  INSERT INTO ods_orders SELECT order_id, product_name, amount, status FROM default_catalog.default_database.kafka_source;
  INSERT INTO dws_product_sales SELECT product_name, SUM(amount) as total_amount FROM default_catalog.default_database.kafka_source WHERE status = 'PAID' GROUP BY product_name;
END;
```

---

## 第四阶段：离线计算 (提交到 Spark 分布式集群)

**企业级网络破局：** 我们将 Python 脚本放在项目根目录（已被挂载进了容器），然后通过 `docker exec` 指挥容器内部的 Spark 大管家亲自执行该脚本，完美避开宿主机与内网 Worker 之间的 NAT 隔离墙。

在 `bigdata-lab` 目录下，新建 Python 文件 `spark_offline_analysis.py`：

```python
from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, sum as spark_sum, count, avg, desc

print("🚀 正在向集群提交任务，并动态下载依赖包 (首次运行需耐心等待)...")

# ==========================================
# 1. 声明分布式架构与环境初始化
# ==========================================
spark = SparkSession.builder \
    .appName("Paimon_Lakehouse_Offline_Analysis") \
    .master("spark://spark-master:7077") \
    .config("spark.sql.catalog.paimon", "org.apache.paimon.spark.SparkCatalog") \
    .config("spark.sql.catalog.paimon.warehouse", "s3://paimon-data/warehouse") \
    .config("spark.sql.catalog.paimon.s3.endpoint", "http://minio:9000") \
    .config("spark.sql.catalog.paimon.s3.access-key", "admin") \
    .config("spark.sql.catalog.paimon.s3.secret-key", "password123") \
    .config("spark.sql.catalog.paimon.s3.path.style.access", "true") \
    .config("spark.sql.extensions", "org.apache.paimon.spark.extensions.PaimonSparkSessionExtensions") \
    .config("spark.jars.packages", 
            "org.apache.paimon:paimon-spark-3.3:0.8.0,"
            "org.apache.hadoop:hadoop-aws:3.3.2,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.367") \
    .getOrCreate()

# ==========================================
# 2. 数据湖探索与分析
# ==========================================
print("\n" + "="*80)
print("📊 现代流批一体数据湖离线分析报告")
print("="*80)

# 切换到 Paimon Catalog
spark.sql("USE paimon_catalog")

# 显示所有数据库
print("\n📚 数据库列表:")
spark.sql("SHOW DATABASES").show()

# 显示所有表
print("\n📋 表列表:")
spark.sql("SHOW TABLES").show()

# ==========================================
# 3. ODS 层原始订单数据分析
# ==========================================
print("\n" + "="*80)
print("📈 1. ODS层原始订单数据分析")
print("="*80)

# 查看订单表结构
print("\n📐 订单表结构:")
spark.sql("DESCRIBE ods_orders").show()

# 查看总订单数
total_orders = spark.sql("SELECT COUNT(*) as total_orders FROM ods_orders").collect()[0][0]
print(f"\n📊 总订单数: {total_orders}")

# 查看订单数据样本
print("\n📝 订单数据样本:")
spark.sql("SELECT * FROM ods_orders ORDER BY order_id DESC LIMIT 10").show(truncate=False)

# 按状态统计订单分布
print("\n📊 订单状态分布:")
spark.sql("""
    SELECT status, COUNT(*) as count, COUNT(*)*100.0/(SELECT COUNT(*) FROM ods_orders) as percentage
    FROM ods_orders 
    GROUP BY status 
    ORDER BY count DESC
""").show(truncate=False)

# ==========================================
# 4. DWS 层产品销售聚合分析
# ==========================================
print("\n" + "="*80)
print("📈 2. DWS层产品销售聚合分析")
print("="*80)

# 查看产品销售表结构
print("\n📐 产品销售表结构:")
spark.sql("DESCRIBE dws_product_sales").show()

# 查看产品销售数据
print("\n💰 产品销售数据 (按销售额降序):")
spark.sql("SELECT * FROM dws_product_sales ORDER BY total_amount DESC").show(truncate=False)

# 计算总销售额
total_sales = spark.sql("SELECT SUM(total_amount) as total_sales FROM dws_product_sales").collect()[0][0]
print(f"\n💵 总销售额: ¥{total_sales:,.2f}")

# 按销售额排名
print("\n🏆 产品销售额排名:")
spark.sql("""
    SELECT 
        product_name, 
        total_amount,
        RANK() OVER (ORDER BY total_amount DESC) as sales_rank
    FROM dws_product_sales 
    ORDER BY total_amount DESC
""").show(truncate=False)

# ==========================================
# 5. 高级分析：流批一体验证
# ==========================================
print("\n" + "="*80)
print("🔍 3. 流批一体验证分析")
print("="*80)

# 对比流式计算与离线计算的结果一致性
print("\n🔄 验证流式计算与离线计算的一致性:")

# 离线计算产品销售额
offline_sales = spark.sql("""
    SELECT 
        product_name, 
        SUM(amount) as offline_total_amount,
        COUNT(*) as order_count
    FROM ods_orders 
    WHERE status = 'PAID'
    GROUP BY product_name 
    ORDER BY offline_total_amount DESC
""")

# 流式计算结果
streaming_sales = spark.sql("SELECT * FROM dws_product_sales")

# 对比结果
print("\n📊 离线计算结果:")
offline_sales.show(truncate=False)

print("\n📊 流式计算结果:")
streaming_sales.show(truncate=False)

# 计算差异
print("\n🔍 结果一致性验证:")
comparison = offline_sales.alias("offline").join(
    streaming_sales.alias("streaming"),
    col("offline.product_name") == col("streaming.product_name"),
    "outer"
)
comparison.select(
    col("offline.product_name").alias("product_name"),
    col("offline.offline_total_amount").alias("offline_total"),
    col("streaming.total_amount").alias("streaming_total"),
    (col("offline.offline_total_amount") - col("streaming.total_amount")).alias("difference"),
    col("offline.order_count").alias("order_count")
).orderBy(col("offline_total").desc_nulls_last()).show(truncate=False)

# ==========================================
# 6. 数据质量检查
# ==========================================
print("\n" + "="*80)
print("🔍 4. 数据质量检查")
print("="*80)

# 检查空值
print("\n⚠️  空值检查:")
spark.sql("""
    SELECT 
        COUNT(*) as total_records,
        COUNT(order_id) as valid_order_id,
        COUNT(product_name) as valid_product_name,
        COUNT(amount) as valid_amount,
        COUNT(status) as valid_status,
        COUNT(*) - COUNT(order_id) as missing_order_id,
        COUNT(*) - COUNT(product_name) as missing_product_name,
        COUNT(*) - COUNT(amount) as missing_amount,
        COUNT(*) - COUNT(status) as missing_status
    FROM ods_orders
""").show(truncate=False)

# 检查重复订单
print("\n🔄 重复订单检查:")
spark.sql("""
    SELECT order_id, COUNT(*) as duplicate_count
    FROM ods_orders
    GROUP BY order_id
    HAVING COUNT(*) > 1
    ORDER BY duplicate_count DESC
""").show(truncate=False)

# ==========================================
# 7. 分析总结
# ==========================================
print("\n" + "="*80)
print("📝 分析总结")
print("="*80)
print(f"✅ 总订单数: {total_orders}")
print(f"✅ 总销售额: ¥{total_sales:,.2f}")
print(f"✅ 流批一体计算一致性验证完成")
print(f"✅ 数据质量检查完成")
print("\n🎉 恭喜！您已成功完成现代流批一体数据湖的完整实验流程！")
print("📖 您可以继续探索更多高级功能，如：数据变更捕捉 (CDC)、物化视图、时间旅行查询等。")

# 关闭 Spark 会话
spark.stop()
print("\n👋 Spark 会话已关闭")
```

**运行离线分析任务：** 通过 Docker 容器内部执行 Spark 任务，避免网络隔离问题：
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-apps/spark_offline_analysis.py
```

**见证分布式：** 执行时，请在浏览器中打开 Spark 大屏 `http://localhost:8080`，您将看到 3 个 Worker 节点正在协同工作，随后完整的离线分析报表将打印在控制台上，标志着全链路实验圆满成功！

---

## 第五阶段：实验验证与故障排查

### 1. 集群状态验证

**1.1 检查所有容器是否正常运行**
```bash
docker compose ps
```
预期输出：所有容器的 STATUS 均为 `Up` 或 `Up (healthy)`

**1.2 查看容器日志排查问题**
```bash
# 查看 Kafka 日志
docker logs bigdata-kafka

# 查看 Flink JobManager 日志
docker logs bigdata-flink-jm

# 查看 MinIO 日志
docker logs bigdata-minio
```

### 2. Kafka 消息验证

**2.1 检查 Kafka 主题是否创建**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```
预期输出：应看到 `ecommerce_orders` 主题

**2.2 查看 Kafka 消息内容**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-console-consumer.sh --topic ecommerce_orders --from-beginning --bootstrap-server localhost:9092
```
预期输出：应看到模拟的订单数据正在实时流入

### 3. MinIO 数据湖验证

**3.1 登录 MinIO Web UI**
- 浏览器打开：http://localhost:9001
- 账号：admin
- 密码：password123

**3.2 验证数据湖目录结构**
在 MinIO Web UI 中，应能看到：
- Bucket 名称：`paimon-data`
- 目录结构：`warehouse/ods_orders/` 和 `warehouse/dws_product_sales/`
- 数据文件：应能看到 Parquet 或 ORC 格式的数据文件

**3.3 检查数据文件大小**
在 MinIO Web UI 中，查看 `paimon-data/warehouse/ods_orders` 目录下的文件大小，确认数据正在持续写入。

### 4. Flink 任务验证

**4.1 登录 Flink Web UI**
- 浏览器打开：http://localhost:8081

**4.2 检查 Running Jobs**
在 Flink Web UI 的 "Jobs" 菜单中，应能看到正在运行的 Flink 任务，状态为 "RUNNING"

**4.3 查看任务 metrics**
点击任务名称，查看以下 metrics：
- Records Received：应持续增长
- Records Sent：应持续增长
- Checkpoint：应定期完成（每 10 秒）

### 5. Dinky 验证

**5.1 登录 Dinky Web UI**
- 浏览器打开：http://localhost:8888
- 账号：admin
- 密码：admin

**5.2 验证集群连接**
- 进入 "集群中心"
- 检查 `LocalFlink` 集群状态应为 "在线"

**5.3 验证 SQL 任务**
- 进入 "数据开发"
- 查看任务状态应为 "RUNNING"

### 6. Spark 任务验证

**6.1 登录 Spark Web UI**
- 浏览器打开：http://localhost:8080

**6.2 检查 Workers**
应能看到 3 个 Worker 节点，状态为 "ALIVE"

**6.3 查看 Application**
在 "Completed Applications" 中，应能看到您提交的离线分析任务，状态为 "FINISHED"

### 7. 数据一致性验证

**7.1 对比 Kafka 和 MinIO 的数据量**
```bash
# 统计 Kafka 消息数量
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-run-class.sh kafka.tools.GetOffsetShell --topic ecommerce_orders --bootstrap-server localhost:9092

# 通过 Spark SQL 统计 MinIO 中的数据量
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data/warehouse \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true \
  -e "SELECT COUNT(*) FROM paimon_catalog.default.ods_orders"
```
预期结果：两个数据量应该接近（考虑到 Flink 处理延迟）

**7.2 验证流批计算一致性**
对比第四阶段离线分析中，离线计算结果和流式计算结果应该完全一致。

### 8. 常见问题排查

**问题 1：Kafka 无法连接**
- 检查端口映射：`docker compose ps` 确认 9092 端口已映射
- 检查网络配置：确认 `KAFKA_ADVERTISED_LISTENERS` 配置正确
- 检查防火墙：确保防火墙未阻止 9092 端口

**问题 2：Flink 任务失败**
- 查看日志：`docker logs bigdata-flink-jm` 查看错误信息
- 检查依赖：确认 `flink-jars` 目录下的 Jar 包已正确挂载
- 检查连接地址：确认 Kafka 和 MinIO 的连接地址正确

**问题 3：MinIO 无数据**
- 检查 Flink 任务状态：确认任务正在运行
- 检查 Checkpoint：确认 Checkpoint 正常完成
- 查看 Flink 日志：检查是否有写入错误

**问题 4：Spark 任务失败**
- 检查依赖：确认 `spark.jars.packages` 配置正确
- 检查网络：确认容器内部能访问 MinIO
- 查看 Spark 日志：`docker logs bigdata-spark-master` 查看错误信息

### 9. 实验完成检查清单

在完成所有实验步骤后，请确认以下检查项：

- [ ] 所有 Docker 容器正常运行
- [ ] Kafka 主题已创建并接收数据
- [ ] MinIO 中存在 `paimon-data` Bucket
- [ ] MinIO 中存在 `ods_orders` 和 `dws_product_sales` 表
- [ ] Flink 任务正在运行并持续写入数据
- [ ] Spark 离线分析任务成功执行
- [ ] 流批计算结果一致
- [ ] 所有 Web UI 均可正常访问