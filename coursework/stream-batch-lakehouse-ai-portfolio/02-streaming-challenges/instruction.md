# 《云计算与大数据处理》业务挑战实验手册

**实验环境前提约定**
1. **Kafka**: `localhost:9092`
2. **MinIO**: `localhost:9000` (AccessKey: `admin`, SecretKey: `password123`，提前创建 Bucket `paimon-data`)
3. **Flink**: 独立集群 (JobManager `localhost:8081`)
4. **Spark**: Local 模式启动 `spark-sql` 客户端

---

## 实验一：数据倾斜 (Data Skew) 的多米诺骨牌

### 1.1 实验背景与原理

#### 1.1.1 什么是数据倾斜？

数据倾斜是大数据处理领域最常见、最头疼的性能问题之一。简单来说，**数据倾斜就是数据分布不均匀**。

想象一下：你有10个工人要搬运1000箱货物，理想情况下每个工人搬100箱，大家同时完成。但如果其中一个工人要搬900箱，其他9个工人只搬100箱，那么那个搬900箱的工人就会成为瓶颈，整个团队的效率都会被他拖慢。

在大数据处理中，这10个工人就是并行处理的Task，货物就是数据，箱子上的标签就是Key。如果某个Key对应的数据量特别大，那么处理这个Key的Task就会成为瓶颈。

#### 1.1.2 数据倾斜是怎么产生的？

数据倾斜的产生通常有以下几个原因：

**原因1：业务数据本身就不均匀**
- **电商场景**：双十一期间，iPhone的销量可能是其他手机的100倍
- **社交场景**：明星发一条微博，评论量可能是普通用户的10000倍
- **游戏场景**：热门服务器的在线人数是冷门服务器的100倍
- **支付场景**：支付宝的交易量是其他小支付平台的1000倍

**原因2：数据设计不合理**
- 选择了不合适的字段作为Key
- 没有考虑到数据分布的特点
- 没有对热点Key进行特殊处理

**原因3：数据预处理不充分**
- 没有对数据进行分片打散
- 没有对热点Key进行拆分
- 没有对Null值或异常值进行处理

#### 1.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商爆款商品**
假设你在淘宝工作，要统计双十一当天每个商品的销量。数据分布可能是这样的：
- iPhone 15：1,000,000条记录
- 小米14：100,000条记录
- 华为Mate60：80,000条记录
- 其他9997个商品：总共100,000条记录

如果直接按商品ID分组，处理iPhone 15的Task会处理1,000,000条记录，而其他Task可能只处理几千条，这就会导致严重的数据倾斜。

**例子2：明星微博评论**
假设你在微博工作，要统计每条微博的评论数。数据分布可能是这样的：
- 某明星微博：10,000,000条评论
- 网红微博：100,000条评论
- 普通用户微博：平均10条评论

如果直接按微博ID分组，处理明星微博的Task会累死，而其他Task可能闲得无聊。

**例子3：Null值陷阱**
假设你要统计用户的登录次数，但有些数据的用户ID是Null。如果直接按用户ID分组，所有Null值都会被分到同一个Task，可能导致这个Task的数据量特别大。

#### 1.1.4 数据倾斜的危害

数据倾斜会导致一系列严重的问题：

**1. 性能急剧下降**
- 整个作业的执行时间由最慢的Task决定
- 其他Task都完成了，还要等那个慢Task
- 作业执行时间可能从几分钟变成几小时

**2. 反压（Backpressure）**
- 慢Task处理不过来，会向上游发送反压信号
- 上游Task也会变慢，导致整个链路都被拖慢
- 反压会像多米诺骨牌一样传播，影响整个作业

**3. 内存溢出（OOM）**
- 慢Task需要处理大量数据，内存压力大
- 如果使用内存状态后端，可能会耗尽内存
- 最终导致Task失败，作业失败

**4. 资源浪费**
- 其他Task都完成了，但因为有一个Task还在跑，整个作业占用的资源都不能释放
- 集群资源被浪费，其他作业排队等待

#### 1.1.5 如何处理数据倾斜？

处理数据倾斜的核心思想是**"分而治之"**，把大问题拆分成小问题。

**方法1：两阶段聚合（打散+预聚合）**
这是最常用的方法，分为两个阶段：
- **第一阶段（局部聚合）**：给热点Key加上随机后缀，打散到多个Task进行局部聚合
- **第二阶段（全局聚合）**：去掉随机后缀，进行全局聚合

**方法2：加盐打散**
给每个Key加上随机前缀，把数据均匀分散到各个Task。

**方法3：拆分热点Key**
把热点Key拆分成多个子Key，分散到不同Task。

**方法4：动态负载均衡**
实时监控各个Task的负载，动态调整数据分配。

#### 1.1.6 能否避免数据倾斜？

虽然不能100%避免，但可以通过以下方式减少数据倾斜：

**1. 合理选择Key**
- 避免选择分布不均匀的字段作为Key
- 如果必须选择，考虑加盐或拆分
- 考虑使用复合Key

**2. 数据预处理**
- 提前过滤掉Null值或异常值
- 对热点Key进行特殊标记
- 对数据进行分片打散

**3. 优化配置**
- 合理设置并行度
- 配置反压相关参数
- 使用合适的状态后端

**4. 监控和预警**
- 实时监控各个Task的负载
- 发现倾斜及时告警
- 建立自动调优机制

**多米诺骨牌效应：**
1. 热点 Key 的数据全部涌入同一个 Task
2. 该 Task 处理速度跟不上数据流入速度
3. Task 缓冲区满，反压向上游传播
4. 整个作业的吞吐量下降，最终导致数据延迟

### 1.2 实验目的

- 观察 Flink 中 `keyBy` 热点造成的 Task 反压现象
- 理解两阶段聚合（打散+预聚合）的原理
- 掌握数据倾斜的检测和解决方法

### 1.3 实验环境准备

**1.3.1 创建 Kafka 主题**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic click_stream \
  --bootstrap-server localhost:9092 \
  --partitions 5 \
  --replication-factor 1
```

**1.3.2 验证主题创建**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --list \
  --bootstrap-server localhost:9092
```

### 1.4 造数脚本（模拟倾斜数据）

创建 `skew_data_producer.py`：

```python
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 定义商品分布：iPhone15 占 90%，其他商品占 10%
items = [
    ('iPhone15', 0.9),
    ('MacBookPro', 0.03),
    ('iPadAir', 0.03),
    ('AirPods', 0.02),
    ('AppleWatch', 0.02)
]

print("🚀 开始发送倾斜的点击流数据...")
print(f"📊 数据分布: iPhone15 (90%), 其他商品 (10%)")
print("="*80)

click_id = 1
try:
    while True:
        # 按概率分布选择商品
        r = random.random()
        cumulative = 0
        selected_item = 'iPhone15'
        for item, prob in items:
            cumulative += prob
            if r <= cumulative:
                selected_item = item
                break
        
        data = {
            "click_id": f"CLK_{click_id:08d}",
            "item_id": selected_item,
            "user_id": f"USER_{random.randint(1, 10000):05d}",
            "click_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time() * 1000)
        }
        
        producer.send('click_stream', value=data)
        
        if click_id % 100 == 0:
            print(f"✅ 已发送 {click_id} 条记录 | 当前商品: {selected_item}")
        
        click_id += 1
        time.sleep(0.01)  # 每秒发送约 100 条数据
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送，共发送 {click_id-1} 条记录")
    producer.close()
```

**运行造数脚本：**
```bash
pip install kafka-python
python skew_data_producer.py
```

### 1.5 问题复现（Flink DataStream）

创建 `SkewDemo.java`：

```java
package com.edu.bigdata.challenge;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.util.Collector;
import org.json.JSONObject;

import java.time.Duration;
import java.util.Properties;

public class SkewDemo {

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4); // 设置并行度为4
        env.enableCheckpointing(5000);

        // 配置 Kafka 消费者
        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "skew-demo-group");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "click_stream",
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> stream = env.addSource(consumer);

        // 解析 JSON 并转换为 Tuple3<itemId, count, windowEnd>
        DataStream<Tuple3<String, Long, Long>> clickStream = stream.map(new MapFunction<String, Tuple3<String, Long, Long>>() {
            @Override
            public Tuple3<String, Long, Long> map(String jsonStr) {
                JSONObject json = new JSONObject(jsonStr);
                String itemId = json.getString("item_id");
                return new Tuple3<>(itemId, 1L, System.currentTimeMillis());
            }
        });

        // 直接按 itemId 进行 keyBy，会产生数据倾斜
        DataStream<Tuple3<String, Long, Long>> resultStream = clickStream
            .keyBy(value -> value.f0)
            .window(TumblingProcessingTimeWindows.of(Time.seconds(10)))
            .apply(new WindowFunction<Tuple3<String, Long, Long>, Tuple3<String, Long, Long>, String, TimeWindow>() {
                @Override
                public void apply(String itemId, TimeWindow window, Iterable<Tuple3<String, Long, Long>> input, Collector<Tuple3<String, Long, Long>> out) {
                    long count = 0;
                    for (Tuple3<String, Long, Long> record : input) {
                        count += record.f1;
                    }
                    out.collect(new Tuple3<>(itemId, count, window.getEnd()));
                }
            });

        // 打印结果
        resultStream.print("倾斜计算结果: ");

        env.execute("数据倾斜演示作业");
    }
}
```

### 1.6 观察反压现象

**1.6.1 启动 Flink 作业**
```bash
cd flink-java-project
mvn clean package
java -cp target/flink-java-project-1.0-SNAPSHOT.jar com.edu.bigdata.challenge.SkewDemo
```

**1.6.2 观察 Flink Web UI**
- 浏览器打开：http://localhost:8081
- 点击正在运行的 Job："数据倾斜演示作业"
- 点击 "Task Managers" 选项卡

**1.6.3 关键观察点：**

1. **Subtask 负载不均：**
   - iPhone15 对应的 Subtask 的 Records Received 应该是其他 Subtask 的 9-10 倍
   - 该 Subtask 的 Busy Time 应该接近 100%
   - 其他 Subtask 的 Busy Time 可能只有 10-20%

2. **反压（Backpressure）：**
   - 查看 "Back Pressure" 选项卡
   - 热点 Subtask 应该显示为红色（HIGH）或橙色（MEDIUM）
   - 反压会向上游传播，导致 Source 节点也出现反压

3. **Checkpoint 时间：**
   - 查看 "Checkpoints" 选项卡
   - 热点 Subtask 的 Checkpoint 时间会明显长于其他 Subtask
   - 如果反压严重，Checkpoint 可能会超时

**1.6.4 查看日志：**
```bash
docker logs bigdata-flink-tm | grep -E "iPhone15|BackPressure"
```

### 1.7 破局解决（两阶段聚合）

创建 `SkewSolution.java`：

```java
package com.edu.bigdata.challenge;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.util.Collector;
import org.json.JSONObject;

import java.util.Properties;
import java.util.concurrent.ThreadLocalRandom;

public class SkewSolution {

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        env.enableCheckpointing(5000);

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "skew-solution-group");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "click_stream",
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> stream = env.addSource(consumer);

        DataStream<Tuple3<String, Long, Long>> clickStream = stream.map(new MapFunction<String, Tuple3<String, Long, Long>>() {
            @Override
            public Tuple3<String, Long, Long> map(String jsonStr) {
                JSONObject json = new JSONObject(jsonStr);
                String itemId = json.getString("item_id");
                return new Tuple3<>(itemId, 1L, System.currentTimeMillis());
            }
        });

        // ==================== 第一阶段：加盐局部聚合 ====================
        DataStream<Tuple3<String, Long, Long>> localAggStream = clickStream
            .map(new MapFunction<Tuple3<String, Long, Long>, Tuple3<String, Long, Long>>() {
                @Override
                public Tuple3<String, Long, Long> map(Tuple3<String, Long, Long> value) {
                    // 随机加盐：将 itemId 打散到 10 个虚拟 Key
                    String saltedKey = value.f0 + "-" + ThreadLocalRandom.current().nextInt(10);
                    return new Tuple3<>(saltedKey, value.f1, value.f2);
                }
            })
            .keyBy(value -> value.f0)
            .window(TumblingProcessingTimeWindows.of(Time.seconds(10)))
            .apply(new WindowFunction<Tuple3<String, Long, Long>, Tuple3<String, Long, Long>, String, TimeWindow>() {
                @Override
                public void apply(String saltedKey, TimeWindow window, Iterable<Tuple3<String, Long, Long>> input, Collector<Tuple3<String, Long, Long>> out) {
                    long count = 0;
                    for (Tuple3<String, Long, Long> record : input) {
                        count += record.f1;
                    }
                    out.collect(new Tuple3<>(saltedKey, count, window.getEnd()));
                }
            });

        // ==================== 第二阶段：去盐全局聚合 ====================
        DataStream<Tuple3<String, Long, Long>> globalAggStream = localAggStream
            .map(new MapFunction<Tuple3<String, Long, Long>, Tuple3<String, Long, Long>>() {
                @Override
                public Tuple3<String, Long, Long> map(Tuple3<String, Long, Long> value) {
                    // 去盐：恢复原始 itemId
                    String originalKey = value.f0.split("-")[0];
                    return new Tuple3<>(originalKey, value.f1, value.f2);
                }
            })
            .keyBy(value -> value.f0)
            .window(TumblingProcessingTimeWindows.of(Time.seconds(10)))
            .apply(new WindowFunction<Tuple3<String, Long, Long>, Tuple3<String, Long, Long>, String, TimeWindow>() {
                @Override
                public void apply(String itemId, TimeWindow window, Iterable<Tuple3<String, Long, Long>> input, Collector<Tuple3<String, Long, Long>> out) {
                    long count = 0;
                    for (Tuple3<String, Long, Long> record : input) {
                        count += record.f1;
                    }
                    out.collect(new Tuple3<>(itemId, count, window.getEnd()));
                }
            });

        globalAggStream.print("优化后计算结果: ");

        env.execute("数据倾斜优化作业");
    }
}
```

### 1.8 验证优化效果

**1.8.1 启动优化后的作业**
```bash
java -cp target/flink-java-project-1.0-SNAPSHOT.jar com.edu.bigdata.challenge.SkewSolution
```

**1.8.2 对比观察：**

1. **负载均衡：**
   - 所有 Subtask 的 Records Received 应该接近均衡
   - 每个 Subtask 的 Busy Time 应该在 20-30% 左右
   - 不再有明显的热点节点

2. **反压消失：**
   - "Back Pressure" 选项卡应该全部显示为绿色（LOW）
   - 作业的整体吞吐量提升

3. **Checkpoint 优化：**
   - Checkpoint 时间明显缩短
   - 所有 Subtask 的 Checkpoint 时间接近

**1.8.3 性能对比：**
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 最大 Subtask 负载 | 90% | 10% | 9x |
| 反压状态 | HIGH (红色) | LOW (绿色) | 显著改善 |
| Checkpoint 时间 | 10-15s | 1-2s | 5-10x |
| 整体吞吐量 | 100 records/s | 500+ records/s | 5x+ |

### 1.9 实验总结

- ✅ 理解了数据倾斜的成因和危害
- ✅ 掌握了通过 Flink Web UI 检测数据倾斜的方法
- ✅ 实现了两阶段聚合（打散+预聚合）的优化方案
- ✅ 验证了优化效果，负载均衡和吞吐量显著提升

---

## 实验二：小文件灾难与读写放大

### 2.1 实验背景与原理

#### 2.1.1 什么是小文件问题？

小文件问题是数据湖和大数据存储系统中最常见的性能杀手之一。简单来说，**小文件问题就是存储系统中存在大量尺寸很小的文件**。

想象一下：你要整理一个图书馆。如果图书馆里都是1000页的厚书，你只需几本书就能装满一个书架，找书也很方便。但如果图书馆里都是只有1页的小册子，你要上万本小册子才能装满一个书架，找一本书可能要翻遍整个书架。

在数据湖中，这些"小册子"就是小文件。小文件通常指大小在几KB到几MB之间的文件，远小于推荐的块大小（通常是128MB或256MB）。

#### 2.1.2 小文件问题是怎么产生的？

小文件问题的产生通常有以下几个原因：

**原因1：实时流处理高频写入**
- Flink、Spark Streaming等流处理框架会频繁产生Checkpoint
- 每个Checkpoint可能都会生成新的数据文件
- 每秒处理一条数据，一天就会产生86400个文件

**原因2：频繁的小批量写入**
- 业务系统每5分钟写入一批数据
- 每批数据只有几MB
- 一年下来会产生10万+个小文件

**原因3：分区表设计不合理**
- 分区粒度过细（按小时甚至分钟分区）
- 每个分区内数据量很小
- 导致大量分区目录，每个目录下只有少量文件

**原因4：数据写入模式问题**
- 每次只写入少量数据
- 没有进行数据合并
- 直接追加到存储系统

#### 2.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商实时订单数据**
假设你在电商公司，要实时收集订单数据。每秒钟有100个订单，Flink作业每10秒做一次Checkpoint。
- 每个Checkpoint生成1个文件，大小约1MB
- 1小时会生成360个文件，总共360MB
- 1天会生成8640个文件，总共8.6GB
- 1年下来会产生300多万个文件！

虽然数据总量只有3TB左右，但文件数量太多，查询性能会非常差。

**例子2：IoT传感器数据**
假设你要收集10万个传感器的数据，每个传感器每5秒发送一条数据。
- 每个传感器每天产生17280条数据
- 10万个传感器每天产生17亿条数据
- 如果按传感器ID分区，每个分区每天只有17280条数据
- 一年下来会有3650万个分区目录！

**例子3：日志收集系统**
假设你要收集应用程序的日志，每小时滚动一次日志文件。
- 每个应用每小时生成1个日志文件
- 1000个应用一天生成24000个文件
- 一年下来会有876万个文件！

#### 2.1.4 小文件问题的危害

小文件问题会导致一系列严重的性能问题：

**1. 元数据压力过大**
- 每个文件都有元数据（文件名、大小、创建时间、位置等）
- NameNode需要在内存中维护所有文件的元数据
- 文件数量太多会耗尽NameNode内存
- 严重时会导致整个HDFS集群宕机

**2. 读取性能急剧下降**
- 查询时需要打开/关闭大量文件
- 每个文件都需要一次I/O操作
- 大量随机I/O操作严重影响性能
- 一个简单的COUNT查询可能需要几分钟甚至几小时

**3. 写入放大效应**
- 频繁创建/删除小文件
- 每次写入都要更新元数据
- 元数据操作开销远大于数据写入本身
- 写入吞吐量大幅下降

**4. 备份和恢复困难**
- 备份大量小文件效率极低
- 恢复时需要重建大量文件
- 备份窗口和恢复时间大幅增加
- 灾难恢复风险增大

**5. 计算资源浪费**
- 查询时需要启动大量Task来处理小文件
- 每个Task的启动开销远大于数据处理时间
- 集群资源被浪费在Task管理上
- 真正用于数据处理的资源占比很低

#### 2.1.5 如何处理小文件问题？

处理小文件问题的核心思想是**"合并"**，把大量小文件合并成少数大文件。

**方法1：Compaction（合并）**
这是最常用的方法，分为两种类型：
- **Minor Compaction**：合并小文件，不删除旧数据
- **Major Compaction**：合并所有文件，删除重复和过期数据

**方法2：写前合并**
- 在写入存储系统前先在内存中累积数据
- 达到一定大小后再写入
- 减少文件数量

**方法3：定期合并**
- 定期运行合并任务
- 将小文件合并成大文件
- 通常在业务低峰期执行

**方法4：调整分区策略**
- 增大分区粒度（按天分区而不是按小时）
- 避免过度分区
- 合理设计分区键

#### 2.1.6 能否避免小文件问题？

虽然不能100%避免，但可以通过以下方式大幅减少小文件：

**1. 优化写入策略**
- 增加Checkpoint间隔（从10秒增加到5分钟）
- 启用写前缓冲（Write Buffer）
- 增大文件块大小

**2. 启用自动Compaction**
- 配置自动Compaction触发条件
- 设置目标文件大小
- 合理安排Compaction时间

**3. 合理设计分区**
- 选择合适的分区键
- 避免过度分区
- 考虑数据分布特点

**4. 数据预处理**
- 先写入临时表
- 定期合并到正式表
- 控制文件数量

**5. 监控和调优**
- 监控文件数量和大小分布
- 发现异常及时告警
- 建立自动调优机制

Paimon 提供了 Compaction 机制来解决这个问题，通过合并小文件来优化存储和查询性能。

### 2.2 实验目的

- 观察高频 Checkpoint 下小文件爆炸现象
- 理解小文件问题对性能的影响
- 掌握 Paimon Compaction 的使用方法
- 验证 Compaction 对读写性能的优化效果

### 2.3 实验环境准备

**2.3.1 创建 Kafka 主题**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic order_stream \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

**2.3.2 清理之前的实验数据**
```bash
# 清理 MinIO 中的旧数据
docker exec -it bigdata-minio mc rm --recursive --force /data/paimon-data/orders
```

### 2.4 造数脚本（持续产生订单数据）

创建 `order_data_producer.py`：

```python
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

products = ["iPhone 15", "MacBook Pro", "iPad Air", "AirPods", "Apple Watch"]
statuses = ["UNPAID", "PAID", "SHIPPED", "DELIVERED"]

print("🚀 开始发送订单数据...")
print("="*80)

order_id = 1
try:
    while True:
        data = {
            "order_id": order_id,
            "product_name": random.choice(products),
            "amount": round(random.uniform(100.0, 20000.0), 2),
            "status": random.choice(statuses),
            "user_id": f"USER_{random.randint(1, 1000):04d}",
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time() * 1000)
        }
        
        producer.send('order_stream', value=data)
        
        if order_id % 100 == 0:
            print(f"✅ 已发送 {order_id} 条订单记录")
        
        order_id += 1
        time.sleep(0.01)  # 每秒约 100 条
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送，共发送 {order_id-1} 条订单")
    producer.close()
```

**运行造数脚本：**
```bash
python order_data_producer.py
```

### 2.5 问题复现（高频 Checkpoint）

启动 Flink SQL Client：
```bash
docker exec -it bigdata-flink-jm /opt/flink/bin/sql-client.sh
```

执行以下 SQL：

```sql
-- 设置极短的 Checkpoint 间隔，模拟高频写入
SET 'execution.checkpointing.interval' = '10s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';

-- 创建 Kafka 源表
CREATE TABLE kafka_orders (
    order_id BIGINT,
    product_name STRING,
    amount DOUBLE,
    status STRING,
    user_id STRING,
    create_time STRING,
    `timestamp` BIGINT
) WITH (
    'connector' = 'kafka',
    'topic' = 'order_stream',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'small-file-demo',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- 创建 Paimon 表（启用自动 Compaction）
CREATE TABLE paimon_orders (
    order_id BIGINT,
    product_name STRING,
    amount DOUBLE,
    status STRING,
    user_id STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/orders',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'false',  -- 关闭自动 Compaction，手动观察小文件
    'compaction.async.enabled' = 'false',
    'write-buffer-size' = '16mb',
    'page-size' = '32kb'
);

-- 持续写入数据
INSERT INTO paimon_orders 
SELECT order_id, product_name, amount, status, user_id, create_time 
FROM kafka_orders;
```

### 2.6 观察小文件爆炸现象

**2.6.1 查看 MinIO 中的文件**
- 浏览器打开：http://localhost:9001
- 登录：admin / password123
- 导航到：paimon-data -> orders -> bucket-0

**2.6.2 关键观察点：**

1. **文件数量爆炸：**
   - 每 10 秒生成一批新文件
   - 每个文件大小通常只有几 KB 到几十 KB
   - 10 分钟内可能产生上百个文件

2. **文件命名规律：**
   - 数据文件：`data-*.parquet`
   - 变更日志文件：`changelog-*.avro`
   - 清单文件：`manifest-*.avro`

3. **统计文件数量和大小：**
```bash
# 统计文件数量
docker exec -it bigdata-minio mc find /data/paimon-data/orders --name "*.parquet" | wc -l

# 统计总大小
docker exec -it bigdata-minio mc du /data/paimon-data/orders
```

**2.6.3 观察查询性能下降：**
打开另一个 Flink SQL Client，执行查询：
```sql
SELECT count(*) FROM paimon_orders;
```
记录查询耗时。随着文件数量增加，查询时间会越来越长。

### 2.7 破局解决（Paimon Compaction）

**2.7.1 启动 Spark SQL**
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data/warehouse \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true \
  --conf spark.jars.packages=org.apache.paimon:paimon-spark-3.3:0.8.0
```

**2.7.2 执行 Compaction**
```sql
-- 切换到 Paimon Catalog
USE paimon_catalog;

-- 查看表信息
DESCRIBE FORMATTED orders;

-- 执行全量 Compaction
CALL sys.compact(table => 'orders', order_strategy => 'zorder');

-- 查看 Compaction 进度
SELECT * FROM paimon_catalog.sys.compaction_history WHERE table_name = 'orders';
```

**2.7.3 高级 Compaction 配置：**
```sql
-- 指定分区 Compaction
CALL sys.compact(table => 'orders', partitions => map('dt', '2023-10-01'));

-- 指定并行度
CALL sys.compact(table => 'orders', parallelism => 4);

-- 指定目标文件大小
CALL sys.compact(table => 'orders', target_file_size => '128mb');
```

### 2.8 验证 Compaction 效果

**2.8.1 再次查看 MinIO 文件：**
- 刷新 MinIO Web UI
- 观察文件数量显著减少
- 观察单个文件大小增加到几十 MB

**2.8.2 统计对比：**
```bash
# Compaction 前
docker exec -it bigdata-minio mc find /data/paimon-data/orders --name "*.parquet" | wc -l
docker exec -it bigdata-minio mc du /data/paimon-data/orders

# Compaction 后
docker exec -it bigdata-minio mc find /data/paimon-data/orders --name "*.parquet" | wc -l
docker exec -it bigdata-minio mc du /data/paimon-data/orders
```

**2.8.3 查询性能对比：**
在 Flink SQL Client 中再次执行查询：
```sql
SELECT count(*) FROM paimon_orders;
SELECT product_name, sum(amount) FROM paimon_orders GROUP BY product_name;
```
记录查询耗时，应该明显缩短。

**2.8.4 性能对比表：**
| 指标 | Compaction 前 | Compaction 后 | 优化比例 |
|------|---------------|---------------|----------|
| 文件数量 | 150+ | 5-10 | 15-30x |
| 平均文件大小 | 10-50 KB | 64-128 MB | 1000-10000x |
| COUNT 查询时间 | 5-10s | < 1s | 5-10x |
| 聚合查询时间 | 10-15s | 1-2s | 5-15x |

### 2.9 自动 Compaction 配置

在生产环境中，建议启用 Paimon 的自动 Compaction：

```sql
-- 创建表时启用自动 Compaction
CREATE TABLE paimon_orders (
    ...
) WITH (
    ...
    'auto-compaction' = 'true',
    'compaction.async.enabled' = 'true',
    'compaction.target.file.size' = '128mb',
    'compaction.num-sorted-run.compaction-trigger' = '5'
);
```

### 2.10 实验总结

- ✅ 观察到了高频 Checkpoint 导致的小文件爆炸现象
- ✅ 验证了小文件对查询性能的负面影响
- ✅ 掌握了 Paimon Compaction 的使用方法
- ✅ 验证了 Compaction 显著优化了存储和查询性能
- ✅ 了解了自动 Compaction 的配置方法

---

## 实验三：Flink 的状态爆炸（State Bloat）

### 3.1 实验背景与原理

#### 3.1.1 什么是状态爆炸？

状态爆炸是流计算中最常见的内存问题之一。简单来说，**状态爆炸就是流计算作业的状态大小随着时间无限增长**。

想象一下：你在一家公司做前台，负责记录来访人员的信息。如果有人要求你"记住所有来过公司的人，永远不要忘记"，那么你的笔记本会越来越厚，最后可能需要一个仓库来存放这些笔记本。

在流计算中，这个"笔记本"就是状态（State）。状态用于存储中间计算结果，比如：
- 过去1小时内每个用户的点击次数
- 当天的独立访客数（UV）
- 两个流的Join中间结果

如果不加以管理，状态会像滚雪球一样越滚越大，最终导致作业崩溃。

#### 3.1.2 状态爆炸是怎么产生的？

状态爆炸的产生通常有以下几个原因：

**原因1：无界去重**
- 计算历史UV（独立访客数）
- 需要记住所有来过的用户
- 用户数越来越多，状态越来越大

**原因2：无限窗口聚合**
- 计算从作业启动以来的累计值
- 窗口永远不关闭
- 状态持续累积

**原因3：无限流Join**
- 两个无限流进行Join操作
- 需要保留两个流的所有历史数据
- 状态随时间无限增长

**原因4：状态清理不及时**
- 没有设置TTL（Time To Live）
- 过期的状态没有及时清理
- 状态只增不减

#### 3.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商网站UV统计**
假设你在电商公司，要统计网站的历史独立访客数。
- 每天有100万新用户访问
- 一年下来有3.65亿用户
- 每个用户ID占用约64字节
- 一年后状态大小约为：3.65亿 × 64字节 ≈ 23GB
- 三年后状态大小约为70GB！

**例子2：用户行为分析**
假设你要分析用户的购物行为，需要记录用户的所有历史点击。
- 每个用户每天产生100条点击记录
- 100万用户每天产生1亿条记录
- 每条记录占用约100字节
- 一年后状态大小约为：1亿 × 365 × 100字节 ≈ 3.65TB！

**例子3：实时推荐系统**
假设你要做实时推荐，需要记住用户的所有历史行为。
- 用户历史行为数据越来越多
- 状态持续增长
- 最终导致作业OOM（内存溢出）

#### 3.1.4 状态爆炸的危害

状态爆炸会导致一系列严重的问题：

**1. 内存压力**
- 状态越来越大，占用大量内存
- TaskManager的内存被耗尽
- 最终导致OOM（内存溢出）
- 作业崩溃失败

**2. Checkpoint变慢**
- Checkpoint需要序列化所有状态
- 状态越大，Checkpoint时间越长
- Checkpoint超时失败
- 作业无法正常恢复

**3. 恢复时间变长**
- 作业失败后需要重新加载所有状态
- 状态越大，恢复时间越长
- 业务中断时间增加
- 影响系统可用性

**4. 磁盘IO压力**
- 使用RocksDB状态后端时，状态会写入磁盘
- 大量状态导致频繁的磁盘IO
- 磁盘IO成为瓶颈
- 影响整个节点的性能

**5. 资源浪费**
- 状态越来越大，需要更多的内存和磁盘
- 集群资源被无效占用
- 其他作业无法获得足够资源
- 集群利用率下降

#### 3.1.5 如何处理状态爆炸？

处理状态爆炸的核心思想是**"过期清理"**，只保留需要的状态，及时清理过期状态。

**方法1：设置State TTL**
- 为状态设置过期时间（Time To Live）
- 过期的状态自动清理
- 状态大小保持在可控范围内

**方法2：合理设计窗口**
- 使用滚动窗口或滑动窗口
- 窗口结束后清理状态
- 避免无限窗口

**方法3：增量清理**
- 定期扫描并清理过期状态
- 增量清理，避免一次性清理大量状态
- 减少对作业性能的影响

**方法4：状态压缩**
- 使用高效的状态压缩算法
- 减小状态的存储空间
- 降低内存和磁盘压力

#### 3.1.6 能否避免状态爆炸？

虽然不能100%避免，但可以通过以下方式有效控制状态增长：

**1. 合理设置TTL**
- 根据业务需求设置合理的TTL
- 不要设置过长的TTL
- 定期评估和调整TTL

**2. 选择合适的状态后端**
- 状态较大时使用RocksDB状态后端
- RocksDB可以将状态 spill 到磁盘
- 避免内存状态后端的OOM风险

**3. 优化状态设计**
- 只存储必要的信息
- 避免存储冗余数据
- 使用高效的数据结构

**4. 监控状态大小**
- 实时监控状态大小
- 发现异常及时告警
- 建立自动调优机制

**5. 定期重启作业**
- 对于一些可以接受数据丢失的场景
- 定期重启作业，清理所有状态
- 从零开始计算

常见的状态爆炸场景：
- 历史去重（UV 计算）
- 长期窗口聚合
- 无限流的 Join 操作

### 3.2 实验目的

- 模拟历史去重 UV 计算场景
- 观察 RocksDB 状态无限膨胀现象
- 理解 State TTL 的原理和作用
- 掌握状态管理的最佳实践

### 3.3 实验环境准备

**3.3.1 创建 Kafka 主题**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic user_click \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

**3.3.2 清理 Flink 状态**
```bash
# 停止之前的作业，清理状态
docker exec -it bigdata-flink-jm /opt/flink/bin/flink list
docker exec -it bigdata-flink-jm /opt/flink/bin/flink stop <job-id>
```

### 3.4 造数脚本（用户点击流）

创建 `user_click_producer.py`：

```python
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 开始发送用户点击流数据...")
print("="*80)

click_id = 1
# 模拟用户池：前 1000 个用户会反复出现，新用户逐步增加
user_pool = list(range(1, 1001))

try:
    while True:
        # 80% 的概率从现有用户池中选择，20% 的概率创建新用户
        if random.random() < 0.8:
            user_id = random.choice(user_pool)
        else:
            user_id = len(user_pool) + 1
            user_pool.append(user_id)
        
        data = {
            "click_id": f"CLK_{click_id:08d}",
            "user_id": f"USER_{user_id:05d}",
            "page_url": f"/page_{random.randint(1, 100)}",
            "click_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time() * 1000)
        }
        
        producer.send('user_click', value=data)
        
        if click_id % 100 == 0:
            print(f"✅ 已发送 {click_id} 条记录 | 用户池大小: {len(user_pool)}")
        
        click_id += 1
        time.sleep(0.01)  # 每秒约 100 条
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送，共发送 {click_id-1} 条记录")
    producer.close()
```

**运行造数脚本：**
```bash
python user_click_producer.py
```

### 3.5 问题复现（无界去重）

启动 Flink SQL Client：
```bash
docker exec -it bigdata-flink-jm /opt/flink/bin/sql-client.sh
```

执行以下 SQL：

```sql
-- 设置 RocksDB 作为状态后端
SET 'state.backend' = 'rocksdb';
SET 'state.backend.rocksdb.localdir' = '/tmp/flink/rocksdb';
SET 'state.backend.rocksdb.options.compaction.style' = 'universal';
SET 'state.backend.rocksdb.metrics.monitor-disk-usage' = 'true';

-- 启用 Checkpoint
SET 'execution.checkpointing.interval' = '10s';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';
SET 'execution.checkpointing.externalized-checkpoint-retention' = 'RETAIN_ON_CANCELLATION';

-- 创建 Kafka 源表
CREATE TABLE user_click (
    click_id STRING,
    user_id STRING,
    page_url STRING,
    click_time STRING,
    `timestamp` BIGINT,
    ts AS TO_TIMESTAMP(click_time, 'yyyy-MM-dd HH:mm:ss'),
    WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'user_click',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'state-bloat-demo',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- 计算历史 UV（无状态 TTL 限制）
CREATE TABLE uv_result (
    dt STRING,
    uv BIGINT,
    PRIMARY KEY (dt) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/uv_result',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true'
);

-- 持续计算并写入结果
INSERT INTO uv_result
SELECT 
    DATE_FORMAT(ts, 'yyyy-MM-dd') as dt,
    COUNT(DISTINCT user_id) as uv
FROM user_click
GROUP BY DATE_FORMAT(ts, 'yyyy-MM-dd');
```

### 3.6 观察状态爆炸现象

**3.6.1 查看 Flink Web UI**
- 浏览器打开：http://localhost:8081
- 点击正在运行的 Job
- 点击 "Checkpoints" 选项卡

**3.6.2 关键观察点：**

1. **Checkpoint 大小增长：**
   - 观察 "Checkpoint History" 中的 "State Size"
   - 状态大小应该随着时间持续增长
   - 增长速度取决于新用户的数量

2. **Checkpoint 时间变长：**
   - 观察 "Checkpoint History" 中的 "Duration"
   - 随着状态增大，Checkpoint 时间会越来越长

3. **RocksDB 磁盘使用：**
```bash
# 查看 TaskManager 容器中的 RocksDB 数据
docker exec -it bigdata-flink-tm du -sh /tmp/flink/rocksdb
```

4. **TaskManager 内存使用：**
```bash
# 查看 TaskManager 的内存使用情况
docker stats bigdata-flink-tm
```

**3.6.3 观察业务指标：**
在另一个 Flink SQL Client 中执行：
```sql
SELECT * FROM uv_result ORDER BY dt;
```
观察 UV 数值的增长。

### 3.7 破局解决（State TTL）

停止之前的作业，然后执行以下 SQL：

```sql
-- 设置 State TTL：24 小时
SET 'table.exec.state.ttl' = '24 h';

-- 重新创建 UV 结果表（先删除旧表）
DROP TABLE IF EXISTS uv_result;

CREATE TABLE uv_result (
    dt STRING,
    uv BIGINT,
    PRIMARY KEY (dt) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/uv_result_ttl',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true'
);

-- 重新执行计算（State TTL 已生效）
INSERT INTO uv_result
SELECT 
    DATE_FORMAT(ts, 'yyyy-MM-dd') as dt,
    COUNT(DISTINCT user_id) as uv
FROM user_click
GROUP BY DATE_FORMAT(ts, 'yyyy-MM-dd');
```

### 3.8 验证 State TTL 效果

**3.8.1 观察状态稳定：**
- 观察 Checkpoint History 中的 State Size
- 状态大小应该在达到一定值后趋于稳定
- 不会再无限增长

**3.8.2 对比实验数据：**

| 时间点 | 无 TTL 状态大小 | 有 TTL 状态大小 |
|--------|----------------|----------------|
| 10 分钟 | 10 MB | 8 MB |
| 30 分钟 | 30 MB | 12 MB |
| 1 小时 | 60 MB | 15 MB |
| 2 小时 | 120 MB | 15 MB |
| 24 小时 | 1.4 GB | 15 MB |

**3.8.3 Checkpoint 时间对比：**

| 指标 | 无 TTL | 有 TTL | 优化比例 |
|------|--------|--------|----------|
| Checkpoint 时间 | 10-30s | 1-3s | 10x |
| 状态大小 | 持续增长 | 稳定 | 无限 |
| 恢复时间 | 几分钟 | 几秒 | 100x |

### 3.9 State TTL 高级配置

**3.9.1 DataStream API 中的 TTL 配置：**
```java
import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.time.Time;

StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.hours(24))
    .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
    .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
    .build();
```

**3.9.2 TTL 清理策略：**

1. **增量清理：**
```sql
SET 'state.backend.rocksdb.ttl.compaction.filter.enabled' = 'true';
SET 'table.exec.state.ttl.cleanup.strategy' = 'incremental';
```

2. **全量清理：**
```sql
SET 'table.exec.state.ttl.cleanup.strategy' = 'full';
SET 'table.exec.state.ttl.cleanup.full.snapshot' = 'true';
```

### 3.10 实验总结

- ✅ 模拟了历史去重 UV 计算场景
- ✅ 观察到了 RocksDB 状态无限膨胀现象
- ✅ 理解了 State TTL 的原理和作用
- ✅ 验证了 State TTL 有效控制了状态大小
- ✅ 掌握了 State TTL 的配置方法和最佳实践

---

## 实验四：迟到与乱序数据（Late & Out-of-Order）

### 4.1 实验背景与原理

#### 4.1.1 什么是迟到和乱序数据？

在流计算中，迟到和乱序数据是最常见的现实问题。简单来说：

**迟到数据**：数据的事件时间（Event Time）早于当前系统的处理时间（Processing Time），也就是数据"来晚了"。

**乱序数据**：数据到达的顺序与它们的事件时间顺序不一致，也就是数据"插队了"。

想象一下：你在一家快递公司工作，负责按时间顺序整理包裹。理想情况下，包裹应该按照寄出时间顺序到达。但现实中，可能会出现：
- 昨天寄出的包裹今天才到（迟到数据）
- 今天寄出的包裹比昨天寄出的包裹先到（乱序数据）

在流计算中，这些迟到和乱序的数据如果不妥善处理，会导致计算结果不准确。

#### 4.1.2 迟到和乱序数据是怎么产生的？

迟到和乱序数据的产生通常有以下几个原因：

**原因1：网络延迟**
- 数据在网络传输过程中遇到拥塞
- 不同网络路径的传输时间不同
- 导致数据到达时间不一致

**原因2：系统故障**
- 上游系统临时故障，数据缓存后重发
- 重试机制导致数据重复发送
- 故障恢复后的数据批量到达

**原因3：数据分发**
- 数据从多个数据源并行发送
- 不同数据源的处理速度不同
- 导致数据到达顺序混乱

**原因4：数据分区**
- 数据被分区到不同的Kafka Partition
- 每个Partition的消费速度不同
- 导致跨Partition的数据乱序

**原因5：业务逻辑**
- 客户端批量发送数据
- 异步处理导致数据乱序
- 重试机制导致重复数据

#### 4.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商订单支付**
假设你在电商公司，要实时统计订单支付金额。
- 用户A在10:00下单支付，网络延迟导致数据在10:05才到达
- 用户B在10:03下单支付，数据在10:04就到达了
- 如果按处理时间统计，用户B的支付会被算在用户A前面
- 导致10:00-10:05的支付金额统计不准确

**例子2：IoT传感器数据**
假设你要收集温度传感器的数据。
- 传感器A在10:00采集到25°C，网络拥塞导致数据在10:10才到达
- 传感器B在10:05采集到26°C，数据在10:06就到达了
- 如果按处理时间统计，10:00-10:05的平均温度会偏低

**例子3：用户行为跟踪**
假设你要跟踪用户的浏览路径。
- 用户在10:00点击了商品A，数据在10:03才到达
- 用户在10:02点击了商品B，数据在10:02就到达了
- 如果按处理时间排序，用户的浏览路径会变成：商品B → 商品A
- 这与实际浏览顺序不符，影响行为分析的准确性

#### 4.1.4 迟到和乱序数据的危害

迟到和乱序数据如果不妥善处理，会导致一系列问题：

**1. 计算结果不准确**
- 窗口计算遗漏迟到数据
- 聚合结果不完整
- 统计指标有偏差
- 影响业务决策

**2. 数据丢失**
- 迟到数据被直接丢弃
- 重要业务数据丢失
- 无法追溯和审计
- 违反数据完整性要求

**3. 结果重复**
- 重复触发窗口计算
- 产生重复的结果
- 下游系统收到重复数据
- 导致数据不一致

**4. 系统复杂度增加**
- 需要额外的机制处理迟到数据
- 增加系统复杂度
- 降低系统可维护性
- 增加开发和运维成本

#### 4.1.5 如何处理迟到和乱序数据？

处理迟到和乱序数据的核心思想是**"权衡"**，在准确性和实时性之间找到平衡点。

**方法1：Watermark（水印）**
- 定义事件时间的进度
- 允许一定的乱序
- 当Watermark超过窗口结束时间时，关闭窗口

**方法2：Allowed Lateness（允许迟到）**
- 窗口关闭后，仍允许一定时间内的迟到数据触发窗口
- 迟到数据可以更新窗口结果
- 平衡准确性和实时性

**方法3：Side Output（侧输出流）**
- 将严重迟到的数据输出到侧流
- 避免丢失数据
- 可以后续重新处理

**方法4：事件时间语义**
- 使用事件时间（Event Time）而非处理时间（Processing Time）
- 基于数据实际发生时间进行计算
- 提高结果的准确性

#### 4.1.6 能否避免迟到和乱序数据？

迟到和乱序数据是分布式系统的固有特性，无法完全避免，但可以通过以下方式减少影响：

**1. 优化数据传输**
- 使用更稳定的网络
- 减少网络拥塞
- 优化数据传输协议

**2. 合理设置Watermark**
- 根据业务特点设置合理的乱序容忍时间
- 不要设置过大或过小
- 定期评估和调整

**3. 分层处理**
- 实时层：快速出结果，允许一定误差
- 批处理层：定时重算，保证准确性
- Lambda架构：兼顾实时性和准确性

**4. 数据校验**
- 对迟到数据进行校验
- 识别和处理异常数据
- 保证数据质量

**5. 监控和告警**
- 监控迟到数据的比例
- 发现异常及时告警
- 建立自动调优机制

在真实的流计算场景中，数据往往会因为网络延迟、系统故障、重试等原因而迟到或乱序。Flink 提供了以下机制来处理这些问题：

- **Watermark**：定义事件时间的进度，允许一定的乱序
- **Allowed Lateness**：允许迟到的数据再次触发窗口
- **Side Output**：将严重迟到的数据输出到侧流，避免丢失

### 4.2 实验目的

- 理解 Watermark 机制的原理
- 学习如何处理迟到数据
- 掌握 Allowed Lateness 和 Side Output 的使用
- 验证不同处理策略的效果

### 4.3 实验环境准备

**4.3.1 创建 Kafka 主题**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic sensor_data \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

### 4.4 造数脚本（传感器数据）

创建 `sensor_data_producer.py`：

```python
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime, timedelta

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 开始发送传感器数据...")
print("="*80)

sensor_ids = [f"SENSOR_{i:03d}" for i in range(1, 11)]
record_id = 1

try:
    while True:
        # 90% 的数据是正常时间戳，10% 的数据是迟到的
        if random.random() < 0.9:
            # 正常数据：当前时间
            event_time = datetime.now()
        else:
            # 迟到数据：随机延迟 1-300 秒
            delay_seconds = random.randint(1, 300)
            event_time = datetime.now() - timedelta(seconds=delay_seconds)
        
        data = {
            "record_id": f"REC_{record_id:08d}",
            "sensor_id": random.choice(sensor_ids),
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(40.0, 80.0), 2),
            "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(event_time.timestamp() * 1000)
        }
        
        producer.send('sensor_data', value=data)
        
        if record_id % 100 == 0:
            print(f"✅ 已发送 {record_id} 条记录")
        
        record_id += 1
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送，共发送 {record_id-1} 条记录")
    producer.close()
```

**运行造数脚本：**
```bash
python sensor_data_producer.py
```

### 4.5 基础实现（Watermark）

创建 `LateDataDemo.java`：

```java
package com.edu.bigdata.challenge;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.json.JSONObject;

import java.time.Duration;
import java.util.Properties;

public class LateDataDemo {

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);
        env.enableCheckpointing(5000);

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "late-data-demo");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "sensor_data",
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> stream = env.addSource(consumer);

        // 解析 JSON 并提取事件时间
        DataStream<Tuple4<String, String, Double, Long>> sensorStream = stream.map(
            new MapFunction<String, Tuple4<String, String, Double, Long>>() {
                @Override
                public Tuple4<String, String, Double, Long> map(String jsonStr) {
                    JSONObject json = new JSONObject(jsonStr);
                    return new Tuple4<>(
                        json.getString("sensor_id"),
                        json.getString("record_id"),
                        json.getDouble("temperature"),
                        json.getLong("timestamp")
                    );
                }
            }
        );

        // 定义侧输出流标签，用于收集迟到数据
        final OutputTag<Tuple4<String, String, Double, Long>> lateDataTag =
            new OutputTag<Tuple4<String, String, Double, Long>>("late-data") {};

        // 应用 Watermark 和窗口
        SingleOutputStreamOperator<String> windowedStream = sensorStream
            .assignTimestampsAndWatermarks(
                org.apache.flink.api.common.eventtime.WatermarkStrategy
                    .<Tuple4<String, String, Double, Long>>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, timestamp) -> event.f3)
            )
            .keyBy(value -> value.f0)
            .window(TumblingEventTimeWindows.of(Time.seconds(30)))
            .allowedLateness(Time.seconds(60))  // 允许迟到 60 秒
            .sideOutputLateData(lateDataTag)  // 超过允许时间的数据输出到侧流
            .apply(new WindowFunction<Tuple4<String, String, Double, Long>, String, String, TimeWindow>() {
                @Override
                public void apply(String sensorId, TimeWindow window, Iterable<Tuple4<String, String, Double, Long>> input, Collector<String> out) {
                    int count = 0;
                    double sumTemp = 0;
                    double sumHumidity = 0;
                    
                    for (Tuple4<String, String, Double, Long> record : input) {
                        count++;
                        sumTemp += record.f2;
                    }
                    
                    double avgTemp = count > 0 ? sumTemp / count : 0;
                    
                    String result = String.format(
                        "传感器: %s | 窗口: [%s - %s] | 记录数: %d | 平均温度: %.2f°C",
                        sensorId,
                        new java.util.Date(window.getStart()).toString(),
                        new java.util.Date(window.getEnd()).toString(),
                        count,
                        avgTemp
                    );
                    
                    out.collect(result);
                }
            });

        // 打印正常处理的结果
        windowedStream.print("正常数据: ");

        // 打印迟到数据
        windowedStream.getSideOutput(lateDataTag).print("迟到数据告警: ");

        env.execute("迟到数据处理演示");
    }
}
```

### 4.6 运行并观察结果

**4.6.1 启动作业**
```bash
java -cp target/flink-java-project-1.0-SNAPSHOT.jar com.edu.bigdata.challenge.LateDataDemo
```

**4.6.2 观察控制台输出：**

正常输出示例：
```
正常数据: > 传感器: SENSOR_001 | 窗口: [Wed Oct 18 10:00:00 CST 2023 - Wed Oct 18 10:00:30 CST 2023] | 记录数: 45 | 平均温度: 25.34°C
```

迟到数据输出示例：
```
迟到数据告警: > (SENSOR_002,REC_00001234,26.78,1697594400000)
```

### 4.7 高级分析：不同延迟级别的处理

创建 `LateDataAnalysis.java`：

```java
package com.edu.bigdata.challenge;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.json.JSONObject;

import java.time.Duration;
import java.util.Properties;

public class LateDataAnalysis {

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);
        env.enableCheckpointing(5000);

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "late-data-analysis");

        FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
            "sensor_data",
            new SimpleStringSchema(),
            properties
        );
        consumer.setStartFromLatest();

        DataStream<String> stream = env.addSource(consumer);

        DataStream<Tuple5<String, String, Double, Long, Long>> sensorStream = stream.map(
            new MapFunction<String, Tuple5<String, String, Double, Long, Long>>() {
                @Override
                public Tuple5<String, String, Double, Long, Long> map(String jsonStr) {
                    JSONObject json = new JSONObject(jsonStr);
                    long eventTime = json.getLong("timestamp");
                    long processingTime = System.currentTimeMillis();
                    return new Tuple5<>(
                        json.getString("sensor_id"),
                        json.getString("record_id"),
                        json.getDouble("temperature"),
                        eventTime,
                        processingTime
                    );
                }
            }
        );

        // 定义多个侧输出流标签，按延迟程度分类
        final OutputTag<Tuple5<String, String, Double, Long, Long>> slightlyLateTag =
            new OutputTag<Tuple5<String, String, Double, Long, Long>>("slightly-late") {};
        
        final OutputTag<Tuple5<String, String, Double, Long, Long>> veryLateTag =
            new OutputTag<Tuple5<String, String, Double, Long, Long>>("very-late") {};
        
        final OutputTag<Tuple5<String, String, Double, Long, Long>> extremelyLateTag =
            new OutputTag<Tuple5<String, String, Double, Long, Long>>("extremely-late") {};

        SingleOutputStreamOperator<String> windowedStream = sensorStream
            .assignTimestampsAndWatermarks(
                org.apache.flink.api.common.eventtime.WatermarkStrategy
                    .<Tuple5<String, String, Double, Long, Long>>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, timestamp) -> event.f3)
            )
            .keyBy(value -> value.f0)
            .window(TumblingEventTimeWindows.of(Time.seconds(30)))
            .sideOutputLateData(slightlyLateTag)  // 基础侧输出
            .apply(new WindowFunction<Tuple5<String, String, Double, Long, Long>, String, String, TimeWindow>() {
                @Override
                public void apply(String sensorId, TimeWindow window, Iterable<Tuple5<String, String, Double, Long, Long>> input, Collector<String> out) {
                    int count = 0;
                    double sumTemp = 0;
                    
                    for (Tuple5<String, String, Double, Long, Long> record : input) {
                        count++;
                        sumTemp += record.f2;
                    }
                    
                    double avgTemp = count > 0 ? sumTemp / count : 0;
                    
                    String result = String.format(
                        "传感器: %s | 窗口: [%s - %s] | 记录数: %d | 平均温度: %.2f°C",
                        sensorId,
                        new java.util.Date(window.getStart()).toString(),
                        new java.util.Date(window.getEnd()).toString(),
                        count,
                        avgTemp
                    );
                    
                    out.collect(result);
                }
            });

        // 对基础侧输出进行二次分析，按延迟程度分类
        DataStream<Tuple5<String, String, Double, Long, Long>> slightlyLateStream = 
            windowedStream.getSideOutput(slightlyLateTag);
        
        // 按延迟时间分流
        slightlyLateStream
            .filter(new org.apache.flink.api.common.functions.FilterFunction<Tuple5<String, String, Double, Long, Long>>() {
                @Override
                public boolean filter(Tuple5<String, String, Double, Long, Long> value) {
                    long delay = value.f4 - value.f3;
                    return delay > 5000 && delay <= 30000;  // 5-30 秒
                }
            })
            .print("轻微迟到 (5-30s): ");
        
        slightlyLateStream
            .filter(new org.apache.flink.api.common.functions.FilterFunction<Tuple5<String, String, Double, Long, Long>>() {
                @Override
                public boolean filter(Tuple5<String, String, Double, Long, Long> value) {
                    long delay = value.f4 - value.f3;
                    return delay > 30000 && delay <= 120000;  // 30-120 秒
                }
            })
            .print("严重迟到 (30-120s): ");
        
        slightlyLateStream
            .filter(new org.apache.flink.api.common.functions.FilterFunction<Tuple5<String, String, Double, Long, Long>>() {
                @Override
                public boolean filter(Tuple5<String, String, Double, Long, Long> value) {
                    long delay = value.f4 - value.f3;
                    return delay > 120000;  // 超过 120 秒
                }
            })
            .print("极端迟到 (>120s): ");

        windowedStream.print("正常数据: ");

        env.execute("迟到数据分级分析");
    }
}
```

### 4.8 实验总结

- ✅ 理解了 Watermark 机制的原理
- ✅ 掌握了 Allowed Lateness 的使用
- ✅ 实现了 Side Output 收集迟到数据
- ✅ 验证了不同延迟级别的数据处理策略
- ✅ 了解了迟到数据的业务影响和应对方案

---

## 实验五：跨组件的 Exactly-Once 保证与重复数据

### 5.1 实验背景与原理

#### 5.1.1 什么是Exactly-Once语义？

在分布式系统中，Exactly-Once语义是最理想也是最难实现的处理语义。简单来说，**Exactly-Once语义保证每条数据被精确处理一次，既不丢失也不重复**。

想象一下：你在银行转账，要求：
- 钱不能少（不丢失）
- 钱不能多（不重复）
- 转账只能成功或失败，不能出现中间状态（原子性）

在流计算和数据湖场景中，Exactly-Once语义同样重要：
- 电商订单：不能重复计算，也不能漏算
- 支付系统：不能重复扣款，也不能漏扣
- 统计系统：统计结果必须准确

#### 5.1.2 重复数据是怎么产生的？

重复数据的产生通常有以下几个原因：

**原因1：重试机制**
- 上游系统发送数据后未收到确认
- 超时后自动重试
- 导致同一条数据被多次发送

**原因2：网络抖动**
- 数据在网络传输中丢失
- 上游系统重发数据
- 但实际上数据已经被处理了

**原因3：系统故障**
- 处理系统在处理数据后崩溃
- 未能更新消费位移（Offset）
- 重启后重新消费相同数据

**原因4：Checkpoint失败**
- Flink作业的Checkpoint失败
- 作业重启后从上次成功的Checkpoint恢复
- 导致数据被重复处理

**原因5：数据重放**
- 业务需要重新处理历史数据
- 但没有去重机制
- 导致数据重复写入

#### 5.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商订单支付**
假设你在电商公司，要统计订单支付金额。
- 用户支付100元，支付系统发送支付成功消息
- 消息队列收到消息，但确认消息丢失
- 支付系统超时后重发支付成功消息
- 统计系统收到两条相同的支付消息
- 如果没有去重，会统计成支付200元

**例子2：银行转账**
假设你在银行工作，要处理用户转账。
- 用户A向用户B转账1000元
- 银行系统处理转账后崩溃
- 未能更新消费位移
- 系统重启后重新处理转账
- 如果没有去重，会转出2000元

**例子3：日志收集**
假设你要收集应用程序的日志。
- 应用程序发送日志后未收到确认
- 自动重发日志
- 日志收集系统收到多条相同日志
- 如果没有去重，日志会重复

#### 5.1.4 重复数据的危害

重复数据如果不妥善处理，会导致一系列严重问题：

**1. 统计结果不准确**
- 重复统计导致结果偏大
- 影响业务决策
- 可能导致财务损失

**2. 数据不一致**
- 不同系统的数据不一致
- 数据对账困难
- 数据可信度下降

**3. 资源浪费**
- 重复处理浪费计算资源
- 重复存储浪费存储资源
- 增加系统负载

**4. 业务错误**
- 重复扣款导致用户投诉
- 重复发货导致库存损失
- 重复计算导致报表错误

**5. 系统复杂度增加**
- 需要额外的去重机制
- 增加系统复杂度
- 降低系统可维护性

#### 5.1.5 如何实现Exactly-Once？

实现Exactly-Once的核心思想是**"幂等性"**和**"事务性"**。

**方法1：主键约束**
- 为表定义主键
- 重复写入时，主键冲突自动覆盖
- 保证数据唯一性

**方法2：幂等写入**
- 重复写入不会产生重复数据
- 通过主键或唯一键实现
- 多次写入结果一致

**方法3：事务支持**
- 写入操作具有原子性
- 要么全部成功，要么全部失败
- 避免中间状态

**方法4：快照隔离**
- 读取时看到一致的快照
- 不会看到部分写入的数据
- 保证数据一致性

#### 5.1.6 能否完全避免重复数据？

重复数据是分布式系统的固有特性，无法完全避免，但可以通过以下方式减少影响：

**1. 合理设计重试机制**
- 设置合理的超时时间
- 限制重试次数
- 避免盲目重试

**2. 使用幂等操作**
- 设计幂等的API
- 重复调用结果一致
- 减少重复数据的影响

**3. 实现去重机制**
- 基于主键去重
- 基于唯一键去重
- 基于时间窗口去重

**4. 事务性写入**
- 使用事务保证原子性
- 避免部分写入
- 保证数据一致性

**5. 监控和告警**
- 监控重复数据的比例
- 发现异常及时告警
- 建立自动调优机制

在分布式系统中，Exactly-Once 语义是指每条数据被精确处理一次，既不丢失也不重复。Paimon 通过以下机制实现 Exactly-Once：

- **主键约束**：确保数据唯一性
- **幂等写入**：重复写入不会产生重复数据
- **事务支持**：写入操作具有原子性
- **快照隔离**：读取时看到一致的快照

### 5.2 实验目的

- 理解 Paimon 的 Exactly-Once 保证机制
- 对比无主键表和有主键表的差异
- 验证 Paimon 对重复数据的处理能力
- 掌握幂等写入的最佳实践

### 5.3 实验环境准备

**5.3.1 创建 Kafka 主题**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic duplicate_data \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

**5.3.2 清理旧数据**
```bash
docker exec -it bigdata-minio mc rm --recursive --force /data/paimon-data/duplicate_demo
```

### 5.4 造数脚本（包含重复数据）

创建 `duplicate_data_producer.py`：

```python
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 开始发送包含重复数据的订单流...")
print("="*80)

# 记录已发送的订单，用于生成重复
sent_orders = []
order_id = 1

try:
    while True:
        # 70% 的概率发送新订单，30% 的概率发送重复订单
        if random.random() < 0.7 or len(sent_orders) < 10:
            # 新订单
            data = {
                "order_id": order_id,
                "user_id": f"USER_{random.randint(1, 1000):04d}",
                "product_id": f"PROD_{random.randint(1, 100):03d}",
                "amount": round(random.uniform(100.0, 1000.0), 2),
                "status": random.choice(["PAID", "SHIPPED", "DELIVERED"]),
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            sent_orders.append(data.copy())
            order_id += 1
        else:
            # 重复订单：从已发送的订单中随机选择
            data = random.choice(sent_orders).copy()
            # 修改一些字段，模拟重试时的微小变化
            data["create_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        producer.send('duplicate_data', value=data)
        
        if (order_id - 1) % 50 == 0:
            print(f"✅ 已发送 {order_id-1} 条记录（含重复）| 已缓存订单数: {len(sent_orders)}")
        
        time.sleep(0.02)
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送，共发送 {order_id-1} 条记录")
    producer.close()
```

**运行造数脚本：**
```bash
python duplicate_data_producer.py
```

### 5.5 创建对比表

启动 Flink SQL Client：
```bash
docker exec -it bigdata-flink-jm /opt/flink/bin/sql-client.sh
```

执行以下 SQL：

```sql
-- 设置 Checkpoint
SET 'execution.checkpointing.interval' = '10s';

-- 创建 Kafka 源表
CREATE TABLE kafka_duplicate (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'duplicate_data',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'duplicate-demo',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- 表A：无主键表（Append Only）
CREATE TABLE paimon_append (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/duplicate_demo/append_table',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'write-mode' = 'append-only'
);

-- 表B：有主键表（Upsert 模式）
CREATE TABLE paimon_pk (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/duplicate_demo/pk_table',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'write-mode' = 'upsert',
    'merge-engine' = 'deduplicate'
);

-- 同时写入两张表
INSERT INTO paimon_append SELECT * FROM kafka_duplicate;
INSERT INTO paimon_pk SELECT * FROM kafka_duplicate;
```

### 5.6 验证重复数据处理

**5.6.1 使用 Spark SQL 查询对比**
启动 Spark SQL：
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data/duplicate_demo \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true
```

执行查询：

```sql
-- 切换到 Paimon Catalog
USE paimon_catalog;

-- 查看无主键表
SELECT 
    '无主键表' as table_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT order_id) as unique_orders,
    COUNT(*) - COUNT(DISTINCT order_id) as duplicate_count
FROM append_table;

-- 查看有主键表
SELECT 
    '有主键表' as table_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT order_id) as unique_orders,
    COUNT(*) - COUNT(DISTINCT order_id) as duplicate_count
FROM pk_table;
```

**5.6.2 预期结果：**

| table_type | total_records | unique_orders | duplicate_count |
|------------|---------------|---------------|-----------------|
| 无主键表 | 1000 | 700 | 300 |
| 有主键表 | 700 | 700 | 0 |

**5.6.3 查看重复数据详情：**
```sql
-- 查看无主键表中的重复订单
SELECT order_id, COUNT(*) as dup_count
FROM append_table
GROUP BY order_id
HAVING COUNT(*) > 1
ORDER BY dup_count DESC
LIMIT 10;

-- 对比两张表的数据差异
SELECT 
    a.order_id,
    a.user_id,
    a.amount,
    a.create_time as append_create_time,
    b.create_time as pk_create_time
FROM append_table a
JOIN pk_table b ON a.order_id = b.order_id
WHERE a.order_id IN (
    SELECT order_id FROM append_table
    GROUP BY order_id
    HAVING COUNT(*) > 1
)
ORDER BY a.order_id
LIMIT 20;
```

### 5.7 深入理解 Paimon 的去重机制

**5.7.1 查看 Paimon 的合并日志：**
```bash
# 查看表的快照信息
docker exec -it bigdata-spark-master /opt/spark/bin/spark-shell \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data/duplicate_demo

-- 在 Spark Shell 中执行
import org.apache.paimon.table.Table
import org.apache.paimon.spark.catalog.SparkCatalog

val catalog = new SparkCatalog()
catalog.setConf(spark.sparkContext.hadoopConfiguration)
catalog.initialize("paimon_catalog", new java.util.HashMap[String, String]())

val table = catalog.getTable(new org.apache.paimon.catalog.Identifier("default", "pk_table"))

// 查看快照
println("=== 快照信息 ===")
table.snapshotManager().snapshots().forEach { snapshot =>
    println(s"快照ID: ${snapshot.snapshotId()}, 记录数: ${snapshot.recordCount()}")
}

// 查看 manifest 文件
println("\n=== Manifest 文件 ===")
table.snapshotManager().latestSnapshot().ifPresent { snapshot =>
    snapshot.readManifestFiles(table.fileIO()).forEach { manifest =>
        println(manifest)
    }
}
```

**5.7.2 观察 MinIO 中的文件结构：**
- 浏览器打开：http://localhost:9001
- 导航到：paimon-data -> duplicate_demo
- 对比两张表的文件结构

### 5.8 幂等写入的高级配置

**5.8.1 配置幂等写入：**
```sql
-- 配置主键表的幂等写入
CREATE TABLE paimon_idempotent (
    order_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    status STRING,
    create_time STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/duplicate_demo/idempotent_table',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'write-mode' = 'upsert',
    'merge-engine' = 'deduplicate',
    'idempotent-write' = 'true',
    'sequence.field' = 'create_time'
);
```

**5.8.2 测试幂等写入：**
```sql
-- 重复插入相同数据
INSERT INTO paimon_idempotent VALUES
(1, 'USER_001', 'PROD_001', 100.0, 'PAID', '2023-10-01 10:00:00'),
(1, 'USER_001', 'PROD_001', 100.0, 'PAID', '2023-10-01 10:00:00');

-- 查询结果：只有一条记录
SELECT * FROM paimon_idempotent;

-- 插入更新数据（sequence.field 更大）
INSERT INTO paimon_idempotent VALUES
(1, 'USER_001', 'PROD_001', 100.0, 'SHIPPED', '2023-10-01 10:00:01');

-- 查询结果：已更新为 SHIPPED
SELECT * FROM paimon_idempotent;

-- 插入旧数据（sequence.field 更小）- 不会更新
INSERT INTO paimon_idempotent VALUES
(1, 'USER_001', 'PROD_001', 100.0, 'PAID', '2023-10-01 10:00:00');

-- 查询结果：仍然是 SHIPPED
SELECT * FROM paimon_idempotent;
```

### 5.9 实验总结

- ✅ 理解了 Paimon 的 Exactly-Once 保证机制
- ✅ 对比了无主键表和有主键表的差异
- ✅ 验证了 Paimon 对重复数据的去重能力
- ✅ 掌握了幂等写入的配置方法
- ✅ 了解了 sequence.field 的作用

---

## 实验六：Schema 演进的剧烈震荡

### 6.1 实验背景与原理

#### 6.1.1 什么是Schema演进？

Schema演进是指在不丢失数据或中断服务的情况下，修改表结构（Schema）的能力。简单来说，**Schema演进就是让表结构可以"平滑升级"**。

想象一下：你有一座房子，住了几年后想加一个卫生间。传统的数据仓库系统就像一座"钢筋混凝土房子"，想加卫生间必须：
- 把房子拆了重建（停机维护）
- 重新装修（数据重写）
- 通知所有住户暂时搬走（下游系统同步修改）

而现代数据湖系统（如Paimon）就像一座"模块化房子"，想加卫生间可以：
- 直接在旁边加建（在线修改Schema）
- 不影响原有房间使用（自动兼容新老数据）
- 住户无需搬走（下游系统无需修改）

#### 6.1.2 Schema演进的需求是怎么产生的？

Schema演进的需求通常有以下几个原因：

**原因1：业务需求变化**
- 业务发展需要新增字段
- 业务规则变化需要修改字段类型
- 业务下线需要删除字段

**原因2：数据质量提升**
- 发现字段类型不合理
- 需要增加约束条件
- 需要优化数据结构

**原因3：系统升级**
- 升级到新版本的存储格式
- 优化查询性能
- 增加新的功能

**原因4：数据集成**
- 集成新的数据源
- 需要兼容不同的数据格式
- 需要统一数据模型

#### 6.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商用户信息表**
假设你在电商公司，有一张用户信息表：
```sql
CREATE TABLE user (
    user_id BIGINT,
    username STRING,
    email STRING,
    create_time STRING
)
```

业务发展后，需要：
- 增加用户的年龄字段（新增字段）
- 增加用户的地址字段（新增字段）
- 将create_time从STRING改为TIMESTAMP（修改字段类型）
- 不再需要email字段（删除字段）

如果使用传统数据仓库，这些修改可能需要：
- 停机维护几小时
- 重写所有历史数据
- 通知所有下游系统修改

**例子2：订单表Schema演进**
假设你有一张订单表：
```sql
CREATE TABLE order (
    order_id BIGINT,
    user_id BIGINT,
    amount DOUBLE,
    status STRING,
    create_time STRING
)
```

业务发展后，需要：
- 增加订单的支付方式字段（新增字段）
- 增加订单的物流信息字段（新增嵌套字段）
- 将status从STRING改为ENUM（修改字段类型）
- 重命名amount为total_amount（重命名字段）

**例子3：IoT传感器数据表**
假设你有一张传感器数据表：
```sql
CREATE TABLE sensor (
    sensor_id BIGINT,
    temperature DOUBLE,
    humidity DOUBLE,
    collect_time STRING
)
```

业务发展后，需要：
- 增加气压字段（新增字段）
- 增加位置信息字段（新增嵌套字段）
- 将collect_time从STRING改为TIMESTAMP（修改字段类型）
- 删除不再需要的humidity字段（删除字段）

#### 6.1.4 传统Schema演进的痛点

传统数据仓库系统在Schema演进时往往有以下痛点：

**1. 停机维护**
- 修改Schema需要停机
- 业务中断，影响用户体验
- 只能在凌晨低峰期操作

**2. 数据重写**
- 修改字段类型需要重写所有数据
- 数据量大时耗时很长
- 占用大量计算资源

**3. 下游系统同步修改**
- 所有下游系统都需要同步修改
- 修改不及时会导致数据不一致
- 维护成本很高

**4. 数据丢失风险**
- 删除字段可能导致数据丢失
- 修改字段类型可能导致数据截断
- 操作不可逆，风险很大

**5. 版本管理困难**
- Schema版本难以管理
- 不同版本的数据难以兼容
- 数据追溯困难

#### 6.1.5 如何实现Schema演进？

现代数据湖系统（如Paimon）通过以下机制实现Schema演进：

**方法1：Schema版本管理**
- 每个Schema变更都记录一个版本
- 可以追溯Schema的历史变更
- 支持回滚到之前的版本

**方法2：向前兼容**
- 新Schema可以读取老数据
- 老Schema可以读取新数据
- 下游系统无需修改

**方法3：在线Schema变更**
- 无需停机即可修改Schema
- 修改立即生效
- 不影响数据读写

**方法4：自动数据转换**
- 新增字段自动填充默认值
- 修改字段类型自动转换
- 删除字段逻辑删除而非物理删除

#### 6.1.6 能否避免Schema演进？

Schema演进是业务发展的必然需求，无法完全避免，但可以通过以下方式减少影响：

**1. 提前规划Schema**
- 充分考虑未来业务发展
- 设计具有扩展性的Schema
- 避免频繁修改

**2. 使用灵活的数据模型**
- 使用JSON或AVRO等灵活格式
- 支持动态字段
- 减少Schema修改需求

**3. 分层设计**
- 原始层保留原始数据
- 加工层根据需求转换
- 减少对原始层的修改

**4. 灰度发布**
- 先在测试环境验证
- 再在生产环境灰度发布
- 降低风险

**5. 监控和告警**
- 监控Schema变更的影响
- 发现异常及时告警
- 建立回滚机制

在真实的业务场景中，表结构（Schema）经常需要修改。传统的数据仓库系统在 Schema 演进时往往需要：
- 停机维护
- 数据重写
- 下游系统同步修改

Paimon 提供了强大的 Schema 演进支持，可以：
- 在线修改 Schema
- 自动兼容新老数据
- 下游系统无需修改
- 支持复杂的 Schema 变更

### 6.2 实验目的

- 理解 Paimon 的 Schema 演进机制
- 验证在线 Schema 变更的可行性
- 测试新老数据的兼容性
- 掌握 Schema 演进的最佳实践

### 6.3 实验环境准备

**6.3.1 清理旧数据**
```bash
docker exec -it bigdata-minio mc rm --recursive --force /data/paimon-data/schema_demo
```

### 6.4 创建初始表并写入数据

启动 Flink SQL Client：
```bash
docker exec -it bigdata-flink-jm /opt/flink/bin/sql-client.sh
```

执行以下 SQL：

```sql
-- 设置 Checkpoint
SET 'execution.checkpointing.interval' = '10s';

-- 创建初始表：用户基本信息表
CREATE TABLE user_profile (
    user_id BIGINT,
    username STRING,
    email STRING,
    create_time STRING,
    PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/schema_demo/user_profile',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true'
);

-- 插入初始数据
INSERT INTO user_profile VALUES
(1, 'Alice', 'alice@example.com', '2023-10-01 10:00:00'),
(2, 'Bob', 'bob@example.com', '2023-10-01 10:00:01'),
(3, 'Charlie', 'charlie@example.com', '2023-10-01 10:00:02');

-- 查询验证
SELECT * FROM user_profile;
```

### 6.5 启动持续写入作业

创建 `schema_evolution_producer.py`：

```python
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 开始发送用户数据（用于 Schema 演进演示）...")
print("="*80)

user_id = 4
try:
    while True:
        data = {
            "user_id": user_id,
            "username": f"User_{user_id}",
            "email": f"user_{user_id}@example.com",
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        producer.send('schema_demo', value=data)
        
        if user_id % 10 == 0:
            print(f"✅ 已发送 {user_id-3} 条用户记录")
        
        user_id += 1
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print(f"\n👋 停止发送")
    producer.close()
```

**创建 Kafka 主题并运行生产者：**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic schema_demo \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

python schema_evolution_producer.py
```

**在 Flink SQL 中创建流表并持续写入：**
```sql
-- 创建 Kafka 流表
CREATE TABLE kafka_schema_demo (
    user_id BIGINT,
    username STRING,
    email STRING,
    create_time STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'schema_demo',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'schema-evolution-demo',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- 持续写入 Paimon 表
INSERT INTO user_profile SELECT * FROM kafka_schema_demo;
```

### 6.6 在线 Schema 演进

**保持 Flink 作业运行，同时执行以下 Schema 变更操作：**

**6.6.1 添加新字段**
```sql
-- 添加年龄字段
ALTER TABLE user_profile ADD age INT;

-- 添加地址字段
ALTER TABLE user_profile ADD address STRING;

-- 查看表结构
DESCRIBE user_profile;
```

**6.6.2 写入包含新字段的数据**
```sql
-- 插入包含新字段的数据
INSERT INTO user_profile (user_id, username, email, create_time, age, address) VALUES
(1001, 'David', 'david@example.com', '2023-10-01 11:00:00', 25, 'New York'),
(1002, 'Eva', 'eva@example.com', '2023-10-01 11:00:01', 30, 'London');

-- 查询所有数据（包括新老数据）
SELECT * FROM user_profile ORDER BY user_id LIMIT 20;
```

**6.6.3 修改字段类型**
```sql
-- 将 age 字段从 INT 改为 BIGINT
ALTER TABLE user_profile MODIFY COLUMN age BIGINT;

-- 查看表结构
DESCRIBE user_profile;
```

**6.6.4 重命名字段**
```sql
-- 将 address 重命名为 location
ALTER TABLE user_profile RENAME COLUMN address TO location;

-- 查看表结构
DESCRIBE user_profile;
```

**6.6.5 删除字段**
```sql
-- 删除 location 字段
ALTER TABLE user_profile DROP COLUMN location;

-- 查看表结构
DESCRIBE user_profile;
```

### 6.7 验证下游系统兼容性

**6.7.1 使用 Spark SQL 查询（无需重启）**
启动另一个 Spark SQL 客户端：
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data/schema_demo \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true
```

执行查询：
```sql
-- 切换到 Paimon Catalog
USE paimon_catalog;

-- 查询表（Schema 已自动更新）
SELECT * FROM user_profile ORDER BY user_id DESC LIMIT 10;

-- 查看表结构
DESCRIBE user_profile;
```

**6.7.2 验证数据一致性**
```sql
-- 统计新老数据
SELECT 
    CASE WHEN age IS NULL THEN '老数据' ELSE '新数据' END as data_type,
    COUNT(*) as count
FROM user_profile
GROUP BY CASE WHEN age IS NULL THEN '老数据' ELSE '新数据' END;
```

### 6.8 复杂 Schema 演进场景

**6.8.1 添加嵌套字段**
```sql
-- 添加嵌套的地址结构
ALTER TABLE user_profile ADD address STRUCT<city:STRING, country:STRING, zipcode:STRING>;

-- 插入包含嵌套字段的数据
INSERT INTO user_profile (user_id, username, email, create_time, age, address) VALUES
(2001, 'Frank', 'frank@example.com', '2023-10-01 12:00:00', 35, 
 STRUCT('Paris', 'France', '75001')),
(2002, 'Grace', 'grace@example.com', '2023-10-01 12:00:01', 28, 
 STRUCT('Tokyo', 'Japan', '100-0001'));

-- 查询嵌套字段
SELECT 
    user_id,
    username,
    address.city,
    address.country
FROM user_profile
WHERE address IS NOT NULL;
```

**6.8.2 修改嵌套字段**
```sql
-- 添加嵌套字段的子字段
ALTER TABLE user_profile MODIFY COLUMN address ADD street STRING;

-- 查看表结构
DESCRIBE user_profile;
```

### 6.9 实验总结

- ✅ 理解了 Paimon 的 Schema 演进机制
- ✅ 验证了在线 Schema 变更的可行性
- ✅ 测试了新老数据的兼容性
- ✅ 掌握了添加、修改、删除、重命名字段的方法
- ✅ 了解了嵌套字段的 Schema 演进

---

## 实验七：对象存储（MinIO）的访问瓶颈

### 7.1 实验背景与原理

#### 7.1.1 什么是对象存储的访问瓶颈？

对象存储的访问瓶颈是指在使用对象存储（如S3、MinIO）作为数据湖存储时，由于其特殊的架构和特性，导致数据访问性能低下的问题。简单来说，**对象存储的访问瓶颈就是"找文件慢"和"读文件慢"**。

想象一下：你在一个巨大的图书馆里找书。传统的HDFS就像一个有完善索引的图书馆，你可以很快找到想要的书。而对象存储就像一个没有索引的图书馆，你需要：
- 先问管理员"有哪些书？"（LIST操作）
- 管理员给你一个书单（返回所有文件列表）
- 你在书单里找你想要的书（过滤文件列表）
- 找到后再去书架取书（GET操作）

如果图书馆有100万本书，你只想找一本书，这个过程会非常慢。

#### 7.1.2 访问瓶颈是怎么产生的？

对象存储的访问瓶颈通常有以下几个原因：

**原因1：对象存储的特性**
- **元数据操作延迟高**：LIST、GET等元数据操作延迟较高
- **不支持目录**：对象存储是扁平的命名空间，没有真正的目录
- **最终一致性**：元数据操作可能不是立即一致的
- **按使用量计费**：大量元数据操作会增加成本

**原因2：数据组织方式不合理**
- **文件数量太多**：大量小文件导致元数据操作开销大
- **没有分区**：所有文件都在同一个目录下
- **分区粒度过细**：分区太多，元数据操作开销大

**原因3：查询优化不足**
- **没有分区裁剪**：每次查询都要扫描所有文件
- **没有统计信息**：查询优化器无法生成最优执行计划
- **没有索引**：需要扫描所有数据才能找到想要的

**原因4：访问模式不匹配**
- **随机读写多**：对象存储适合顺序读写，不适合随机读写
- **小文件读写多**：对象存储适合大文件，不适合小文件
- **频繁元数据操作**：频繁LIST、GET等操作

#### 7.1.3 实际例子说明

让我们看几个真实的业务场景：

**例子1：电商订单数据分析**
假设你在电商公司，要分析2023年10月15日的订单数据。
- 订单数据按天分区存储在MinIO中
- 每天有1000个文件，每个文件100MB
- 一年有365000个文件
- 如果没有分区裁剪，查询2023-10-15的数据需要扫描所有365000个文件
- 这会导致查询时间从几秒变成几分钟

**例子2：IoT传感器数据分析**
假设你要分析2023年10月的温度传感器数据。
- 传感器数据按天分区，每天按小时子分区
- 每个分区有100个文件，每个文件50MB
- 10月有31天，每天24小时，共有744个分区
- 每个分区100个文件，总共74400个文件
- 如果没有分区裁剪，查询10月的数据需要扫描所有74400个文件

**例子3：用户行为数据分析**
假设你要分析2023年10月15日10点到12点的用户点击数据。
- 用户行为数据按天分区，每天按小时子分区
- 每个分区有500个文件，每个文件200MB
- 如果没有分区裁剪，查询2小时的数据需要扫描所有文件

#### 7.1.4 访问瓶颈的危害

对象存储的访问瓶颈会导致一系列严重问题：

**1. 查询性能低下**
- 简单查询需要几分钟甚至几小时
- 交互式分析无法使用
- 影响用户体验

**2. 资源浪费**
- 大量元数据操作占用计算资源
- 扫描大量无用数据浪费IO资源
- 集群利用率低下

**3. 成本增加**
- 大量元数据操作增加对象存储成本
- 长时间运行的查询增加计算成本
- 总体拥有成本（TCO）上升

**4. 可扩展性差**
- 数据量增长，查询性能线性下降
- 无法支撑大规模数据分析
- 系统扩展性受限

**5. 业务影响**
- 数据分析延迟，影响业务决策
- 实时报表无法按时生成
- 业务创新受阻

#### 7.1.5 如何优化对象存储的访问？

优化对象存储访问的核心思想是**"减少不必要的操作"**，只访问需要的数据。

**方法1：分区裁剪（Partition Pruning）**
- 根据查询条件过滤不需要的分区
- 只扫描需要的分区
- 显著减少扫描的数据量

**方法2：合理设计分区**
- 选择合适的分区键（如日期、地区）
- 合理设置分区粒度（按天分区而不是按小时）
- 避免过度分区

**方法3：文件合并**
- 合并小文件成大文件
- 减少文件数量
- 降低元数据操作开销

**方法4：使用索引**
- 创建布隆过滤器索引
- 创建分区索引
- 加速数据定位

**方法5：缓存热点数据**
- 缓存经常访问的数据
- 减少对对象存储的访问
- 提升查询性能

#### 7.1.6 能否完全避免访问瓶颈？

对象存储的访问瓶颈是由其架构决定的，无法完全避免，但可以通过以下方式显著改善：

**1. 合理设计数据组织**
- 选择合适的分区策略
- 控制文件数量和大小
- 优化目录结构

**2. 使用分区裁剪**
- 确保查询条件包含分区键
- 启用查询优化器的分区裁剪
- 定期分析和优化查询

**3. 优化存储格式**
- 使用列式存储格式（如Parquet、ORC）
- 启用数据压缩
- 减少数据扫描量

**4. 使用缓存**
- 缓存元数据
- 缓存热点数据
- 减少对对象存储的访问

**5. 监控和调优**
- 监控查询性能
- 分析慢查询
- 持续优化数据组织和查询

对象存储（如 S3、MinIO）具有以下特点：
- **高延迟**：元数据操作（LIST、GET）延迟较高
- **成本高**：大量元数据操作会增加成本
- **可扩展性**：吞吐量高，但单请求延迟高

分区裁剪（Partition Pruning）是优化对象存储访问的关键技术，通过：
- 只访问需要的分区
- 减少不必要的元数据操作
- 显著提升查询性能

### 7.2 实验目的

- 理解分区裁剪的原理和重要性
- 对比分区表和非分区表的性能差异
- 观察 Spark 物理执行计划
- 掌握分区设计的最佳实践

### 7.3 实验环境准备

**7.3.1 创建测试数据**
创建 `generate_partition_data.py`：

```python
import json
import random
from datetime import datetime, timedelta
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 开始生成分区测试数据...")
print("="*80)

# 生成一个月的数据
start_date = datetime(2023, 10, 1)
end_date = datetime(2023, 10, 31)

record_id = 1
for day in range((end_date - start_date).days + 1):
    current_date = start_date + timedelta(days=day)
    dt = current_date.strftime("%Y-%m-%d")
    
    # 每天生成 1000 条记录
    for i in range(1000):
        data = {
            "record_id": record_id,
            "user_id": f"USER_{random.randint(1, 1000):04d}",
            "product_id": f"PROD_{random.randint(1, 100):03d}",
            "amount": round(random.uniform(10.0, 1000.0), 2),
            "category": random.choice(["Electronics", "Clothing", "Books", "Food"]),
            "dt": dt
        }
        
        producer.send('partition_demo', value=data)
        record_id += 1
        
        if (record_id - 1) % 5000 == 0:
            print(f"✅ 已生成 {record_id-1} 条记录 | 当前日期: {dt}")

print(f"\n🎉 数据生成完成！共生成 {record_id-1} 条记录")
producer.close()
```

**运行数据生成脚本：**
```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --topic partition_demo \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

python generate_partition_data.py
```

### 7.4 创建对比表

**7.4.1 创建非分区表**
启动 Flink SQL Client：
```bash
docker exec -it bigdata-flink-jm /opt/flink/bin/sql-client.sh
```

执行以下 SQL：

```sql
-- 创建 Kafka 源表
CREATE TABLE kafka_partition_demo (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'partition_demo',
    'properties.bootstrap.servers' = 'kafka:29092',
    'properties.group.id' = 'partition-demo',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);

-- 创建非分区表
CREATE TABLE sales_unpartitioned (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING,
    PRIMARY KEY (record_id) NOT ENFORCED
) WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/partition_demo/sales_unpartitioned',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true'
);

-- 创建分区表（按 dt 分区）
CREATE TABLE sales_partitioned (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    dt STRING,
    PRIMARY KEY (record_id, dt) NOT ENFORCED
) PARTITIONED BY (dt)
WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/partition_demo/sales_partitioned',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true',
    'partition.default-name' = '__DEFAULT_PARTITION__'
);

-- 同时写入两张表
INSERT INTO sales_unpartitioned SELECT * FROM kafka_partition_demo;
INSERT INTO sales_partitioned SELECT * FROM kafka_partition_demo;
```

### 7.5 观察 MinIO 中的文件结构

**7.5.1 查看非分区表的文件结构**
- 浏览器打开：http://localhost:9001
- 导航到：paimon-data -> partition_demo -> sales_unpartitioned
- 观察：所有文件都在同一个目录下

**7.5.2 查看分区表的文件结构**
- 导航到：paimon-data -> partition_demo -> sales_partitioned
- 观察：按日期分区的目录结构
- 每个日期（如 dt=2023-10-01）都有独立的目录

**7.5.3 统计文件数量**
```bash
# 统计非分区表的文件数量
docker exec -it bigdata-minio mc find /data/paimon-data/partition_demo/sales_unpartitioned --name "*.parquet" | wc -l

# 统计分区表的文件数量
docker exec -it bigdata-minio mc find /data/paimon-data/partition_demo/sales_partitioned --name "*.parquet" | wc -l
```

### 7.6 性能对比测试

**7.6.1 启动 Spark SQL**
```bash
docker exec -it bigdata-spark-master /opt/spark/bin/spark-sql \
  --master spark://spark-master:7077 \
  --conf spark.sql.catalog.paimon=org.apache.paimon.spark.SparkCatalog \
  --conf spark.sql.catalog.paimon.warehouse=s3://paimon-data/partition_demo \
  --conf spark.sql.catalog.paimon.s3.endpoint=http://minio:9000 \
  --conf spark.sql.catalog.paimon.s3.access-key=admin \
  --conf spark.sql.catalog.paimon.s3.secret-key=password123 \
  --conf spark.sql.catalog.paimon.s3.path.style.access=true \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.cbo.enabled=true
```

**7.6.2 测试查询 1：全表扫描**
```sql
-- 非分区表查询
SELECT COUNT(*) FROM paimon_catalog.default.sales_unpartitioned;

-- 分区表查询
SELECT COUNT(*) FROM paimon_catalog.default.sales_partitioned;
```

**7.6.3 测试查询 2：单分区查询（分区裁剪生效）**
```sql
-- 非分区表查询（需要扫描所有数据）
SELECT COUNT(*) FROM paimon_catalog.default.sales_unpartitioned 
WHERE dt = '2023-10-15';

-- 分区表查询（只扫描一个分区）
SELECT COUNT(*) FROM paimon_catalog.default.sales_partitioned 
WHERE dt = '2023-10-15';
```

**7.6.4 测试查询 3：范围分区查询**
```sql
-- 非分区表查询（需要扫描所有数据）
SELECT category, SUM(amount) as total_amount
FROM paimon_catalog.default.sales_unpartitioned 
WHERE dt BETWEEN '2023-10-10' AND '2023-10-20'
GROUP BY category;

-- 分区表查询（只扫描指定范围的分区）
SELECT category, SUM(amount) as total_amount
FROM paimon_catalog.default.sales_partitioned 
WHERE dt BETWEEN '2023-10-10' AND '2023-10-20'
GROUP BY category;
```

### 7.7 查看物理执行计划

**7.7.1 查看非分区表的执行计划**
```sql
EXPLAIN FORMATTED
SELECT COUNT(*) FROM paimon_catalog.default.sales_unpartitioned 
WHERE dt = '2023-10-15';
```

**7.7.2 查看分区表的执行计划**
```sql
EXPLAIN FORMATTED
SELECT COUNT(*) FROM paimon_catalog.default.sales_partitioned 
WHERE dt = '2023-10-15';
```

**7.7.3 关键观察点：**

在分区表的执行计划中，应该看到：
- `PartitionFilters: [isnotnull(dt#...), (dt#... = 2023-10-15)]`
- `SelectedPartitions: 1`

在非分区表的执行计划中，应该看到：
- `PartitionFilters: []`
- `SelectedPartitions: 31`（或全部分区）

### 7.8 性能对比结果

| 查询类型 | 非分区表耗时 | 分区表耗时 | 性能提升 |
|---------|-------------|-------------|----------|
| 全表扫描 | 10-15s | 8-12s | 1.2-1.5x |
| 单分区查询 | 8-10s | 0.5-1s | 8-20x |
| 范围查询（10天） | 9-12s | 1-2s | 4.5-12x |
| 聚合查询 | 15-20s | 2-3s | 5-10x |

### 7.9 高级分区策略

**7.9.1 多级分区**
```sql
-- 创建多级分区表（按年、月、日分区）
CREATE TABLE sales_multi_partitioned (
    record_id BIGINT,
    user_id STRING,
    product_id STRING,
    amount DOUBLE,
    category STRING,
    year STRING,
    month STRING,
    day STRING,
    PRIMARY KEY (record_id, year, month, day) NOT ENFORCED
) PARTITIONED BY (year, month, day)
WITH (
    'connector' = 'paimon',
    'path' = 's3://paimon-data/partition_demo/sales_multi_partitioned',
    's3.endpoint' = 'http://minio:9000',
    's3.access-key' = 'admin',
    's3.secret-key' = 'password123',
    's3.path.style.access' = 'true',
    'auto-compaction' = 'true'
);
```

**7.9.2 分区裁剪优化配置**
```sql
-- 启用动态分区裁剪
SET spark.sql.optimizer.dynamicPartitionPruning.enabled = true;

-- 启用 CBO（成本优化器）
SET spark.sql.cbo.enabled = true;

-- 启用自适应执行
SET spark.sql.adaptive.enabled = true;
```

### 7.10 实验总结

- ✅ 理解了分区裁剪的原理和重要性
- ✅ 对比了分区表和非分区表的性能差异
- ✅ 观察了 Spark 物理执行计划
- ✅ 验证了分区裁剪的效果
- ✅ 掌握了分区设计的最佳实践
- ✅ 了解了多级分区和高级优化配置

---

## 实验总结与回顾

### 本课程实验全景

1. **实验一：数据倾斜** - 掌握了两阶段聚合优化
2. **实验二：小文件灾难** - 掌握了 Paimon Compaction
3. **实验三：状态爆炸** - 掌握了 State TTL 管理
4. **实验四：迟到数据** - 掌握了 Watermark 和 Side Output
5. **实验五：Exactly-Once** - 掌握了主键和幂等写入
6. **实验六：Schema 演进** - 掌握了在线 Schema 变更
7. **实验七：访问瓶颈** - 掌握了分区裁剪优化

### 关键技术点总结

- **Flink 核心技术**：State TTL、Watermark、两阶段聚合
- **Paimon 核心技术**：Compaction、Schema 演进、幂等写入、分区表
- **Spark 优化技术**：分区裁剪、CBO、自适应执行
- **对象存储优化**：小文件管理、分区设计、元数据操作优化

### 大数据处理最佳实践

1. **数据倾斜**：使用两阶段聚合、加盐打散
2. **小文件**：启用自动 Compaction、定期手动合并
3. **状态管理**：合理设置 TTL、选择合适的状态后端
4. **乱序数据**：设置合理的 Watermark、使用侧输出流
5. **重复数据**：使用主键约束、幂等写入、sequence.field
6. **Schema 演进**：使用支持在线变更的存储引擎
7. **访问优化**：合理设计分区、启用分区裁剪、优化元数据操作

---

**🎉 恭喜完成所有实验！**

您已经掌握了现代流批一体数据湖的核心技术和最佳实践，可以应对真实世界中的大数据处理挑战！
