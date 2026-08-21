# 短视频流智能审核发布 Demo

本工程实现一个单机可运行、同时尽量贴近工业界方案的短视频内容平台流水线：

1. 短视频进入本地媒体区或 Kafka 主题。
2. 上传服务先保存媒体文件和 `processing` 元数据，再把审核任务写入本地耐久队列。
3. 本地 worker 异步消费队列任务，调用 OpenCV 和 Ollama 多模态模型。
4. 审核策略根据 VLM 风险、视觉信号和标题风险词给出 `published`、`review`、`rejected`。
5. FastAPI 提供 API 和 Demo 网站，展示视频、标签、摘要、审核理由、模型选择、队列状态和流处理事件。

默认模式使用 `SQLite jobs` 模拟工业系统中的消息队列，学生不安装 Docker/Kafka 也能跑通；如果课堂环境具备 Kafka，可以使用 `scripts/kafka_*` 脚本把同一条链路切到真实消息队列。

## 快速运行

建议使用 Python 3.11 或 3.12。Python 3.14 生态里部分科学计算依赖可能没有稳定 wheel。

```bash
cd "4. 实时短视频流智能审核与发布 综合实验/code/short-video-stream-lab"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_ollama_models.py --tier baselines
python scripts/verify_demo.py
python -m app.server
```

Windows PowerShell 使用：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_ollama_models.py --tier baselines
python scripts\verify_demo.py
python -m app.server
```

打开：

```text
http://127.0.0.1:5050
```

默认后台模型是 `Ministral 3 8B Vision (Ollama)`，项目 ID 为 `ministral-3-8b-ollama`。本实验新增 `Qwen3-VL 4B` 和 `Gemma 3 4B` 两个主流视觉基线，所有主要候选都通过 Ollama 下载到学生本机，Windows、Linux、macOS 使用同一套本地 API。`scripts/verify_demo.py` 会严格检查真实本地 VLM 是否被调用；若只想在课堂上保底演示，也可以在网站后台选择 `local-baseline`。

上传视频时，网站会先把真实视频文件写入媒体区并显示在信息流中，状态为 `处理中`。后台再异步执行理解、打标签和审核；等待期间只有摘要和标签位置显示 loading 占位符，视频播放器本身不会被 loading 遮挡。

为了兼顾学生电脑可运行和工业架构思维，本实验默认采用“工业轻量版”：

| 工业组件 | 教学默认实现 | 可替换方向 |
| --- | --- | --- |
| 对象存储 | 本地 `data/media` | MinIO/S3/OSS |
| 消息队列 | SQLite `jobs` 表 + 本地 worker | Kafka/Pulsar |
| 元数据存储 | SQLite `videos/events/jobs` | PostgreSQL/ClickHouse/Paimon |
| 模型服务 | Ollama 本地 HTTP API | vLLM/SGLang/Triton/KServe |
| 流批分析 | 事件日志 + 可选 Kafka 脚本 | Flink/Paimon/Spark |

观察异步队列链路：

```bash
curl -X POST "http://127.0.0.1:5050/api/demo?overwrite=true"
curl http://127.0.0.1:5050/api/health
curl http://127.0.0.1:5050/api/videos
curl http://127.0.0.1:5050/api/events
```

刚触发样本流时，`/api/health` 会显示 `jobs.queued` 或 `jobs.running` 大于 0，视频状态会先是 `processing`；worker 完成后，状态会变成 `published` 或 `review`。

## 模型候选

后台页面可选择以下候选：

| ID | 模型 | 适用场景 |
| --- | --- | --- |
| `ministral-3-8b-ollama` | Ministral 3 8B Vision | 默认主链路模型 |
| `qwen3-vl-4b-ollama` | Qwen3-VL 4B | 主流视觉基线 1 |
| `qwen3-vl-2b-ollama` | Qwen3-VL 2B | 16GB 低配兜底 |
| `qwen2_5-vl-3b-ollama` | Qwen2.5-VL 3B | 16GB 成熟稳定备选 |
| `gemma3-4b-ollama` | Gemma 3 4B | 主流视觉基线 2 |
| `qwen3-vl-8b-ollama` | Qwen3-VL 8B | 32GB 增强档 |
| `qwen2_5-vl-7b-ollama` | Qwen2.5-VL 7B | 32GB 稳定增强档 |
| `gemma3-12b-ollama` | Gemma 3 12B | 32GB 多模态对照 |
| `minicpm-v-ollama` | MiniCPM-V | 32GB 短视频对照 |
| `local-baseline` | OpenCV Local Baseline | 无 GPU、无模型服务兜底 |

## 下载本地模型

先安装并启动 Ollama：https://ollama.com/download。Qwen3-VL 需要 Ollama 0.12.7 或更新版本，下载脚本会自动检查版本。

16GB 机器推荐只下载 16GB 档：

```bash
python scripts/download_ollama_models.py --tier 16gb
```

32GB 机器可下载增强档：

```bash
python scripts/download_ollama_models.py --tier 32gb
```

本实验推荐只检查/下载主模型和两个主流基线：

```bash
python scripts/download_ollama_models.py --tier baselines
```

教师机或 32GB 以上机器如果希望一次准备全部候选：

```bash
python scripts/download_ollama_models.py --tier all
```

网站和 Kafka 消费者会通过 `http://127.0.0.1:11434/api/chat` 调用本机 Ollama。

默认配置会给本地 VLM 传 4 张代表性关键帧，并设置 `think: false`、`format: json`。16GB 机器建议保持默认；32GB 机器可以适当提高：

```bash
LOCAL_VLM_MAX_IMAGES=6 LOCAL_VLM_MAX_TOKENS=4500 python -m app.server
```

PowerShell：

```powershell
$env:LOCAL_VLM_MAX_IMAGES="6"
$env:LOCAL_VLM_MAX_TOKENS="4500"
python -m app.server
```

## Kafka 链路

Kafka 启动后，创建主题：

```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create --if-not-exists \
  --topic short_video_ingest \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --create --if-not-exists \
  --topic short_video_result \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

两个终端分别运行：

```bash
python scripts/kafka_ai_consumer.py
python scripts/kafka_video_producer.py
```
