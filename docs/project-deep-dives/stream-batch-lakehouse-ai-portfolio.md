# Stream-Batch Lakehouse AI Portfolio：项目技术说明

## 300 字项目概括

Stream-Batch Lakehouse AI Portfolio 是一个把四个数据工程课程实验放在同一目录下的流批一体作品集：现代湖仓、流处理可靠性挑战、实时推荐和短视频流审核。现代湖仓实验用 Kafka 产生订单流，Flink/Dinky 做实时清洗与聚合，MinIO 提供 S3 兼容对象存储，Paimon 保存 ODS/DWS 表，Spark 做离线分析和流批一致性核对；流处理挑战把数据倾斜、小文件读写放大、状态膨胀、迟到/乱序、exactly-once 重复、schema evolution 和 MinIO 访问瓶颈拆成七个可验收子实验。推荐实验通过 Flink 处理用户行为特征，Python 服务运行简单 DeepFM/元数据模型，并以 Kafka producer/consumer 验证实时结果。短视频审核实验则以 FastAPI + SQLite + Ollama VLM 处理真实视频，保留 FFmpeg 抽帧、模型选择、队列、API、Kafka 事件和多模型 baseline 证据。项目的价值在于展示从数据采集、状态计算、湖仓存储到 AI 服务化和可观测验收的完整链路；所有模块仍是课程工程，Docker、Flink、Spark、Ollama 和本地视频/模型的运行结果必须按各实验的截图、日志和报告解释，不能合并成一个已部署生产平台。

## 项目定位

- **模块 01**：现代 lakehouse order analytics。
- **模块 02**：七类流处理/湖仓可靠性挑战。
- **模块 03**：实时行为特征与推荐服务。
- **模块 04**：真实短视频的本地多模态审核与发布流水线。

## 总体架构

```text
Kafka streams / video input
        -> Flink or FastAPI ingestion
        -> Paimon/MinIO lakehouse or SQLite job store
        -> Spark analytics / recommender / local VLM
        -> APIs, Kafka result topics, screenshots, logs and reports
```

四个实验共享“输入、处理、存储、验证、证据”的思路，但运行时互相独立；Docker Compose、Maven、Kafka、Flink、Spark 和 Ollama 参数不能跨实验直接套用。

## 模块一：现代湖仓

`01-modern-lakehouse/docker-compose.yml` 编排 Kafka、MinIO、Flink 和 Spark；`order_stream_producer.py` 生成订单流；`dinky_dual_stream_job.sql` 是 SQL 版本；`flink-dual-stream-job/` 提供 Java 实现；`lakehouse_batch_analysis.py` 用 Spark 做离线聚合和流批一致性检查。Paimon 表分为 ODS/DWS 层，MinIO bucket 和 Kafka topic 使用实验约定的 `demo000000` 后缀，避免不同提交相互污染。

## 模块二：流处理挑战

七个子目录分别对应 `01-data-skew`、`02-small-files-read-write-amplification`、`03-state-bloat`、`04-late-out-of-order-data`、`05-exactly-once-duplicates`、`06-schema-evolution` 和 `07-minio-access-bottleneck`。每个目录包含独立 README、acceptance checklist、screenshots 和 evidence；`instruction.md` 保留原始实验手册，`report.md` 汇总整体结论。验收重点是观测现象、触发条件、修复策略和截图/日志是否一致，而不是只写“问题已解决”。

## 模块三：实时推荐

`03-ai-recommender/flink-jobs/` 放置实时推荐 Flink 作业，`recommender-services/` 包含用户行为 producer、消费端、简单 DeepFM、Spark trainer、模型元数据和 `verify_all_code.py`。典型链路是行为事件进入 Kafka，Flink 窗口/特征逻辑产生实时特征，Python 服务根据模型与元数据返回推荐，再通过 topic 或 API 做结果核验。训练模型和在线服务应绑定同一特征字段，避免离线/在线语义漂移。

## 模块四：短视频审核

`04-short-video-stream-review/code/short-video-stream-lab/app/` 拆成配置、FFmpeg 工具、job queue、local worker、model registry、Ollama VLM、预处理、存储、pipeline、API server 和 understanding service；`scripts/` 提供 demo、模型下载、Kafka producer/consumer、一次性 pipeline、API evidence 和报告导出。真实视频位于 `data/incoming/`，但发布时要遵守隐私和许可证边界。`ALLOW_LOCAL_MODEL_FALLBACK`、`LOCAL_VLM_MAX_IMAGES`、`LOCAL_VLM_TIMEOUT_SEC` 等环境变量明确控制模型回退、抽帧数和超时。

## 开发过程

1. 按课程要求先搭建 Docker/依赖环境和最小输入输出闭环。
2. 把每个分布式组件的日志、截图和验收清单作为一等产物，而非事后补图。
3. 对流处理问题逐项构造故障场景，记录现象、根因、修复和复测。
4. 将推荐和审核模块拆成可独立运行的 producer、consumer、service 和 verification 脚本。
5. 最后整理英文展示 README、中文说明、报告和脱敏日志，避免把本地服务缓存或模型文件纳入公开发布。

## 涉及知识点

- Kafka topic、producer/consumer、事件时间和消息语义
- Flink state、窗口、watermark、exactly-once 和 schema evolution
- MinIO/S3、Paimon ODS/DWS、Spark batch/stream consistency
- 数据倾斜、小文件、状态增长和对象存储访问瓶颈
- 推荐系统特征工程、DeepFM、离线训练与在线服务
- FastAPI、SQLite、FFmpeg、Ollama VLM、队列和事件审计
- Docker Compose、Maven、日志证据、课程验收和数据脱敏

## 运行入口

现代湖仓：

```bash
cd 01-modern-lakehouse
docker compose up -d --scale spark-worker=3
```

停止：`docker compose down`。推荐实验：`cd 03-ai-recommender/recommender-services && python verify_all_code.py`。短视频实验：

```powershell
cd 04-short-video-stream-review/code/short-video-stream-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ALLOW_LOCAL_MODEL_FALLBACK = "0"
$env:LOCAL_VLM_MAX_IMAGES = "3"
$env:LOCAL_VLM_TIMEOUT_SEC = "300"
python scripts\verify_demo.py
python scripts\compare_local_baselines.py
python -m app.server
```

## 测试与验证

每个模块的 evidence 目录、截图和报告是不同证据源。分布式实验要记录 topic/table/job 名称、checkpoint、时间窗口和运行日志；VLM 审核要同时记录模型版本、输入视频、抽帧策略、响应、队列健康和 Kafka 事件。没有这些绑定，单张截图只能证明页面出现过。

## 学术申请中的项目价值

该作品集适合展示数据工程、ML systems 和 AI 服务化能力：能解释流批架构、状态一致性、故障模式，也能把推荐模型和多模态审核接入实际事件流。它还可以作为机器人项目的数据基础设施补充，说明你理解感知结果如何进入可观测的服务链路。

## 证据边界

四个实验的运行环境、数据、模型和验收标准不同。课程截图与本地日志不等于生产 SLA；Ollama 模型文件、真实视频、Kafka 集群和 Docker 镜像未随代码完整分发。任何性能结论必须注明是课程样例、局部 smoke 还是完整本地实验。

## 源码审计记录

| 模块 | 代表文件 | 逐文件阅读重点 |
| --- | --- | --- |
| 湖仓编排 | `01-modern-lakehouse/docker-compose.yml` | Kafka、MinIO、Flink、Spark 的服务、端口、volume、依赖和资源。 |
| 订单生成 | `order_stream_producer.py` | topic 命名、事件字段、时间戳、吞吐和重试。 |
| Flink SQL/Java | `dinky_dual_stream_job.sql`、`flink-dual-stream-job/` | source、watermark、窗口、聚合、Paimon sink 和 checkpoint。 |
| Spark 分析 | `lakehouse_batch_analysis.py` | ODS/DWS 读取、批量统计、流批一致性和结果输出。 |
| 挑战实验 | `02-streaming-challenges/01..07/` | 每个故障的触发方式、指标、截图、修复和 acceptance checklist。 |
| 推荐作业 | `03-ai-recommender/flink-jobs/` | 行为事件、实时特征、窗口和推荐结果 topic。 |
| 推荐服务 | `recommender-services/*.py` | producer/consumer、DeepFM、模型 metadata、服务接口和验证。 |
| 审核应用 | `04.../app/config.py`、`pipeline.py`、`local_worker.py` | 输入、抽帧、队列、模型调用、结果存储和状态转换。 |
| VLM/API | `ollama_vlm.py`、`understanding_service.py`、`server.py` | 模型选择、超时、fallback、API schema 和错误响应。 |
| 证据脚本 | `scripts/verify_demo.py`、`compare_local_baselines.py`、Kafka scripts | 环境、模型、事件、API、topic 和多模型对照证据。 |

## 模块一调用链

订单 producer 生成带事件时间的消息，Kafka topic 作为缓冲；Flink source 读取后做字段清洗、窗口聚合和 checkpoint，Paimon 将 ODS/DWS 写入 MinIO；Spark 读取表做离线统计，并核对流处理结果。验证时必须同步记录 topic/table/job 名称、checkpoint、watermark、时间范围和输出快照，单次查询截图不能证明 exactly-once 或流批一致性。

## 模块二故障模型

- **数据倾斜**：识别热点 key、并行度和分区分布，修复后比较吞吐和 backpressure。
- **小文件**：记录文件数量、读写放大和 compaction/rolling 策略。
- **状态膨胀**：观察 state size、TTL、checkpoint 时间和恢复成本。
- **迟到乱序**：对比 event time、watermark、allowed lateness 和窗口结果。
- **Exactly-once**：注入重试/重复消息，确认 sink 与结果计数。
- **Schema evolution**：变更字段、兼容策略和历史数据读取。
- **MinIO 瓶颈**：区分网络、请求并发、对象大小和存储端延迟。

## 推荐与审核的共同工程点

推荐服务和 VLM 审核都需要把在线请求、模型版本、输入数据、超时、fallback 和输出事件写入可检索日志；否则“模型返回结果”无法与用户行为或视频文件对应。推荐要防止离线/在线特征漂移，审核要防止抽帧策略、模型 prompt 和本地 fallback 改变结论却未在报告中记录。

## 交付检查表

- [ ] 每个实验有独立 README、命令、截图、日志和 acceptance checklist。
- [ ] topic、table、job、model 和视频输入都带有版本/命名绑定。
- [ ] failure injection 与修复前后指标成对保存。
- [ ] 真实视频、Ollama 模型和用户数据遵守脱敏/许可证边界。
- [ ] 课程证据、局部 smoke 和生产能力声明不混写。

## 原理与理论基础

### 1. 流批一体语义

流处理把无界事件序列映射成不断更新的结果，批处理则在有限数据上计算快照。两者要得到一致结果，必须统一字段、时间语义、窗口边界、聚合函数和去重规则。Kafka 负责有序分区和持久化日志，Flink 以 event time、watermark 和 checkpoint 管理状态，Paimon/MinIO 负责表文件和快照，Spark 读取快照做离线核验。组件数量不是湖仓价值，关键在于语义是否可解释。

### 2. Exactly-once 与状态

exactly-once 不是“消息只发送一次”，而是端到端效果在失败重试后等价于每条有效事件只影响一次。checkpoint 保存算子状态和 source offset，sink 需要事务/幂等/提交协议；如果 sink、外部 API 或文件写出不满足条件，系统只能声称 at-least-once。状态大小、TTL、窗口、乱序和恢复时间共同决定运行成本。

### 3. 数据倾斜、小文件和迟到

数据倾斜使某些 key 的分区负载远高于均值，导致 backpressure；小文件增加对象存储元数据与打开成本；迟到事件需要 watermark/allowed lateness 处理，过早关闭窗口会丢数据，过晚会增加状态。挑战实验把这些现象拆开，便于用吞吐、延迟、state size、文件数和重放结果验证修复。

### 4. 推荐和 VLM 服务

实时推荐本质上是事件流到特征状态再到排序/召回的函数，离线训练与在线服务必须共享 feature schema 和 model metadata。短视频审核则是视频采样、视觉模型推理、队列、持久化和 API 事件的异步 pipeline；抽帧数、prompt、模型版本和超时会改变输出，必须进入日志。Kafka 结果 topic 使审核链可观察，但不自动保证语义正确。

### 5. 理论假设和失效模式

系统假设事件 schema、时间源、checkpoint、对象存储和模型服务在约定版本内运行。重复、乱序、状态恢复、schema 演进、MinIO 延迟、模型超时和本地 fallback 都会造成结果漂移。课程 evidence 通过日志和 acceptance checklist 约束这些问题，但不能推导生产 SLA 或统一跨实验性能。
