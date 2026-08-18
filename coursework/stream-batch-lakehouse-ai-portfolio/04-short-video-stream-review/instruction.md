# 《云计算与大数据处理》实时短视频流智能审核与发布综合实验

## 实验主题

本实验面向一个短视频平台的真实业务链路：短视频进入平台后，系统需要把视频看作一条连续的数据流，对帧进行理解，完成内容审核、自动打标签，并把审核通过的内容发布到一个自建短视频 Demo 网站。

它承接前三章内容：

- 第一章的流批一体数据湖思想：视频原始文件、理解结果、审核日志都要形成可追溯数据资产。
- 第二章的流处理工程挑战：视频帧流天然存在吞吐、延迟、反压、失败恢复、热点内容等问题。
- 第三章的 AI 推荐系统经验：内容标签和理解结果可以继续作为推荐、搜索、画像和召回的上游特征。

本实验默认使用 `FastAPI + OpenCV + SQLite 本地耐久队列 + 本地 worker + Ollama 本地多模态模型`。其中 SQLite 同时承担单机教学版的元数据存储、事件日志和任务队列，工业环境中可替换为 PostgreSQL、ElasticSearch、ClickHouse、Paimon 或湖仓表；视频文件可替换为 MinIO/S3；任务队列可替换为 Kafka/Pulsar；理解模型通过 Ollama 下载到学生本机，Windows、Linux、macOS 使用同一套本地 API，后台可在 Qwen、Gemma、MiniCPM 和本地 baseline 之间切换。

---

## 实验目标

完成实验后，同学们应能：

1. 理解短视频内容平台中“上传、流式处理、审核、打标、发布”的端到端数据链路。
2. 使用 OpenCV 对视频做关键帧采样、场景切分、运动峰值识别和音频轨抽取。
3. 将 Qwen3-VL、Qwen2.5-VL、Gemma 3、MiniCPM-V 等 Ollama 多模态模型作为可切换候选，默认使用 16GB 机器可运行的 Qwen3-VL 4B。
4. 设计一个结合 VLM 结构化理解、视觉技术指标和平台规则的审核策略，区分自动发布、人工复核和拒绝发布。
5. 使用本地耐久队列或 Kafka 将视频进入事件和审核结果解耦，理解消息队列在内容平台中的作用。
6. 使用 FastAPI 构建一个可运行的网站和 API，把审核结果发布到短视频信息流。
7. 形成可观测日志，能够解释每个视频为什么被发布、复核或拒绝。

---

## 课程概览

### 业务流程

```
短视频文件
   ↓
上传/生成/下载样本
   ↓
任务队列：SQLite jobs（默认）或 Kafka short_video_ingest（可选）
   ↓
本地审核 worker / Kafka consumer
   ↓
OpenCV 关键帧/场景切分/音频预处理
   ↓
后台选择模型：默认 Qwen3-VL 4B，可切换 16GB/32GB 档 Ollama 模型
   ↓
多模态理解：摘要、时间线、实体、动作、字幕、标签、模型风险
   ↓
内容审核：VLM 风险 + 平台规则 + 可解释证据
   ↓
元数据入库 + 媒体文件入媒体区
   ↓
FastAPI Demo 网站展示
   ↓
审核结果：SQLite videos/events（默认）或 Kafka short_video_result（可选）
```

### 为什么采用“工业轻量版”架构

真实短视频平台通常不会在上传接口中直接调用大模型。原因很朴素：视频理解耗时长、失败概率高、模型吞吐有限，如果把这些工作塞进 HTTP 请求线程，用户会等待很久，服务也容易被少量大视频拖垮。因此工业系统一般会把链路拆成几个职责清晰的组件：

| 工业职责 | 生产环境常见实现 | 本实验默认实现 | 这样选择的原因 |
| --- | --- | --- | --- |
| 媒体存储 | S3/OSS/COS/MinIO + CDN | 本地 `data/media` | 不要求学生安装对象存储，浏览器仍能真实播放视频 |
| 任务队列 | Kafka/Pulsar/RabbitMQ | SQLite `jobs` 表 | 保留“入队/消费/失败恢复”语义，同时不依赖 Docker |
| 审核 worker | 独立容器、K8s Job、Ray/KServe worker | `LocalReviewWorker` 后台线程 | 让 16GB/32GB 电脑一次只跑一个模型任务，避免内存打爆 |
| 模型服务 | vLLM/SGLang/Triton/KServe/云模型 API | Ollama 本地 HTTP API | 跨 Windows/Linux/macOS，权重在本机，课堂网络不稳定也能演示 |
| 元数据与审计 | PostgreSQL/ClickHouse/ElasticSearch/Paimon | SQLite `videos/events/jobs` | 单机可运行，同时保留可追溯记录 |
| 流批分析 | Flink 写湖仓，Spark/Flink SQL 分析 | 事件日志 + Kafka 扩展脚本 | 先跑通业务闭环，再接回前三章流批一体基础设施 |

这不是把工业架构“缩水成玩具”，而是把工业架构中最重要的边界保留下来：上传服务只负责接收和展示，队列负责削峰和解耦，worker 负责耗时处理，模型服务负责多模态理解，元数据层负责查询和审计。将来要升级时，可以替换组件，而不是推倒业务流程。

### 端到端时序

下面的时序是本实验最核心的观察对象。请同学们在网站事件面板和 `/api/events` 中对照这些阶段：

```text
1. 浏览器上传视频或点击“运行样本流”
2. FastAPI 保存视频文件到 data/media
3. FastAPI 写入 videos 记录：status = processing
4. 网站立刻显示真实视频播放器，摘要和标签区域显示 loading
5. FastAPI 写入 jobs 记录：status = queued
6. LocalReviewWorker 领取任务：queued -> running
7. OpenCV 抽样帧、计算亮度/运动/闪烁等指标
8. Ollama Qwen3-VL 读取代表性关键帧，输出摘要、标签、风险
9. 审核策略合并模型风险、视觉指标和标题规则
10. videos 更新为 published/review/rejected
11. jobs 更新为 done，events 保留完整审计轨迹
12. 网站轮询刷新文字状态，但不重建 video 节点，播放不中断
```

### 状态机与可观测性

短视频审核系统不是只有“通过/不通过”两个结果。为了让学生理解生产系统中的可观测性，本实验显式维护两个状态机。

视频状态机：

```text
processing
   ├── published  自动发布
   ├── review     进入人工复核
   └── rejected   失败关闭或策略拒绝
```

队列任务状态机：

```text
queued
   ├── running
   │     ├── done
   │     └── failed
   └── queued  服务重启后，未完成 running 任务会被重新放回队列
```

事件日志记录了系统为什么进入某个状态。典型事件包括：

| 事件阶段 | 含义 | 验证方式 |
| --- | --- | --- |
| `ingest` | 收到视频并准备写入媒体区 | `/api/events` 中能看到文件路径和标题 |
| `queued` | 视频已显示，审核任务已入队 | `/api/health` 中 `jobs.queued` 增加 |
| `worker` | 本地 worker 开始或完成任务 | 事件面板出现 worker 事件 |
| `frame_sample` | OpenCV 已抽样分析帧 | 事件 payload 包含亮度、运动等指标 |
| `vlm_understanding` | 本地多模态模型完成理解 | payload 中包含模型摘要和风险 |
| `moderation` | 平台策略完成审核 | payload 中包含风险分和理由 |
| `publish` | 最终发布、复核或拒绝 | 视频卡片状态更新 |

### 工程目录

```text
4. 实时短视频流智能审核与发布 综合实验/
├── instruction.md
└── code/short-video-stream-lab/
    ├── app/
    │   ├── config.py                 # 路径、阈值、实验配置
    │   ├── demo_assets.py            # 生成本地短视频样本
    │   ├── ffmpeg_tools.py           # ffmpeg 元数据/封面辅助能力
    │   ├── job_queue.py              # SQLite 本地耐久任务队列
    │   ├── local_worker.py           # 本地审核 worker：消费队列任务
    │   ├── model_registry.py         # 模型候选列表与后台选择状态
    │   ├── pipeline.py               # 上传、理解、审核、发布主流程
    │   ├── preprocessing.py          # 关键帧、场景切分、音频预处理
    │   ├── server.py                 # FastAPI 网站和 API
    │   ├── storage.py                # SQLite 元数据和事件日志
    │   ├── understanding_service.py  # 多模态理解编排层
    │   ├── ollama_vlm.py             # Ollama 本地多模态 HTTP 客户端
    │   └── video_understanding.py    # OpenCV 视频理解与审核策略
    ├── scripts/
    │   ├── create_demo_video.py      # 生成测试短视频
    │   ├── download_ollama_models.py # 按 16GB/32GB 档下载本地模型
    │   ├── download_sample_video.py  # 下载互联网 MP4 样本
    │   ├── kafka_ai_consumer.py      # Kafka 消费者：审核处理
    │   ├── kafka_video_producer.py   # Kafka 生产者：发送视频进入事件
    │   ├── run_pipeline_once.py      # 本地单次处理
    │   └── verify_demo.py            # 一键验收脚本
    ├── static/
    │   ├── app.js
    │   └── styles.css
    ├── templates/
    │   └── index.html
    └── requirements.txt
```

---

## 安全注意事项

1. 本实验会处理本地视频文件，同学们不要上传含有个人隐私、真实人脸敏感信息或未经授权传播的视频。
2. 实验审核策略是教学用可解释规则，不代表真实平台审核能力。真实业务应使用多模态审核模型、OCR、ASR、黑产规则、人工复核和申诉链路。
3. `download_sample_video.py` 会从互联网下载 MP4 样本，请确认网络环境允许访问外部资源。默认实验不依赖外网，会自动生成本地测试视频。
4. Kafka 脚本默认连接 `localhost:9092`，请只在本机教学环境运行，不要把未鉴权 Kafka 暴露到公网。
5. Demo 网站只监听 `127.0.0.1:5050`，默认不对外开放。

---

## 环境准备与验证

### 1. 安装系统依赖

本实验用 OpenCV 读取视频，用 ffmpeg 生成样本视频和抽取封面。

macOS 可执行：

```bash
brew install ffmpeg
```

Windows 或 Linux 同学请安装 ffmpeg，并确保命令行能执行：

```bash
ffmpeg -version
ffprobe -version
```

### 2. 创建 Python 虚拟环境

建议使用 Python 3.11 或 3.12。Python 3.14 目前部分科学计算依赖可能需要源码编译，不适合作为课堂默认环境。

进入工程目录：

```bash
cd "4. 实时短视频流智能审核与发布 综合实验/code/short-video-stream-lab"
```

创建并激活虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 使用：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 提示脚本执行策略限制，可在当前窗口临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

如果本机没有 `python3.12`，但 `python3 --version` 显示 3.11 或 3.12，也可以使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. 一键验证

先确认 Ollama 已启动，并至少下载默认模型：

```bash
python scripts/download_ollama_models.py --model qwen3-vl-4b-ollama
```

运行验收脚本：

```bash
python scripts/verify_demo.py
```

期望看到类似输出：

```text
verification passed
{'total': 3, 'published': 1, 'review': 2, 'rejected': 0, 'processing': 0}
```

这表示系统已经完成三个短视频样本的生成、抽帧理解、本地 Qwen3-VL 调用、审核打标、入库和事件记录。验收脚本会检查 `backend == local_ollama_vlm`，如果模型没有真正运行会失败。

---

## 分阶段实验步骤

## 第一阶段：准备短视频输入流

短视频平台的源头不是一行订单 JSON，而是一个媒体文件以及围绕它产生的一系列事件。为了让实验不依赖外部网络，本工程提供本地生成样本。

运行：

```bash
python scripts/create_demo_video.py
```

系统会生成三个竖屏短视频：

| 文件 | 业务含义 | 预期结果 |
| --- | --- | --- |
| `campus_sports.mp4` | 明亮、户外感、轻运动 | 自动发布 |
| `night_scene_review.mp4` | 低照度，理解置信度下降 | 人工复核 |
| `flashy_clip_review.mp4` | 强亮度跳变和高运动 | 人工复核 |

如果任课教师希望使用互联网样本，可运行：

```bash
python scripts/download_sample_video.py
```

默认下载地址为：

```text
https://filesamples.com/samples/video/mp4/sample_640x360.mp4
```

该脚本只负责下载样本，同学们仍需用后续流水线处理它。

---

## 第二阶段：上传可见与本地耐久队列

这一阶段回答一个很实际的问题：视频上传后，用户为什么应该马上看到视频，而不是等待模型理解结束？

短视频平台的体验要求是“先接收、先可见、后处理”。用户上传的视频文件已经到达平台时，网站应该先显示播放器和 `处理中` 状态；摘要、标签、审核分可以稍后补齐。这样做有三个好处：

1. 用户能确认上传成功，不会因为模型推理耗时而误以为页面卡死。
2. 上传服务不被模型延迟绑架，HTTP 请求可以快速返回。
3. 审核任务可以排队、重试、恢复，系统更接近工业里的消息队列模型。

本实验中，这个阶段由 `app/pipeline.py`、`app/job_queue.py` 和 `app/local_worker.py` 共同完成。

上传入口先调用 `ingest_video()`。这个函数只做轻量工作：复制视频、写入 `processing` 记录、产生可观测事件。它不调用大模型。

```python
def ingest_video(self, video_path: Path, *, title: str | None = None, source: str = "local") -> dict:
    """先持久化媒体文件和 processing 记录，让前端立即可见。"""
    shutil.copy2(video_path, media_path)
    record = {
        "id": video_id,
        "title": title,
        "media_file": media_name,
        "status": "processing",
        "caption": "",
        "tags": [],
        "metrics": {"model": {"backend": "pending"}},
    }
    upsert_video(record)
    add_event(video_id, "queued", "视频已进入页面，等待后台理解和打标签", {"status": "processing"})
    return record
```

随后服务端把任务写入 SQLite `jobs` 表。这里的 SQLite 不是为了模拟数据库性能，而是为了模拟“任务进入队列”这个工业边界：

```python
job = enqueue_job(
    "complete_video",
    {"record": record, "simulate_stream": True},
    job_id=f"complete-{record['id']}",
)
```

`LocalReviewWorker` 在服务启动时运行。它不断从 `jobs` 表领取最早的 `queued` 任务，把状态改为 `running`，处理完成后改为 `done`。如果服务中途停止，下一次启动时会把遗留的 `running` 任务重新放回 `queued`，这就是本实验的轻量失败恢复。

```python
job = claim_next_job("local-review-worker")
pipeline.complete_video(job["payload"]["record"])
complete_job(job["id"])
```

本阶段的验证方法：

```bash
curl -X POST http://127.0.0.1:5050/api/demo
curl http://127.0.0.1:5050/api/health
curl http://127.0.0.1:5050/api/videos
```

预期现象是：`/api/demo` 很快返回，`/api/videos` 先看到 `processing` 视频，`/api/health` 中 `jobs.queued` 或 `jobs.running` 大于 0。稍等一段时间后，状态会更新为 `published` 或 `review`。

---

## 第三阶段：用 OpenCV 做工业级视频预处理

核心代码在 `app/preprocessing.py` 和 `app/video_understanding.py`。真实短视频平台不会把整段视频每一帧都丢给大模型，这样成本太高、延迟太大。更常见的做法是先用 OpenCV/ffmpeg 做轻量预处理，抽取对理解最有价值的片段：

```python
while True:
    ok, frame = capture.read()
    if not ok:
        break
    diff = cv2.absdiff(gray, previous_gray)
    motion = float(np.mean(diff))
    scene_change = float(np.percentile(diff, 95))
```

本实验会生成三类输入给后续 VLM：

| 输入 | 说明 | 用途 |
| --- | --- | --- |
| 均匀关键帧 | 按时间覆盖整段视频 | 保证模型看到完整故事 |
| 场景切换帧 | 画面发生明显变化的帧 | 捕捉转场和新事件 |
| 运动峰值帧 | 相邻帧变化较大的帧 | 捕捉动作和风险瞬间 |
| 音频轨 | ffmpeg 抽取 16kHz 单声道 wav | 供 ASR/音频模型使用 |
| 视觉技术指标 | 亮度、运动、色彩、闪烁等 | 审核规则和兜底判断 |

本阶段的关键观察点：打开网站后，事件列表中会出现 `frame_sample` 和 `preprocess`。`preprocess` 会记录关键帧数量、音频轨是否存在、视频时长等信息。

---

## 第四阶段：下载并选择本地多模态理解模型

模型候选定义在 `app/model_registry.py`。后台默认模型是 `Qwen3-VL 4B (Ollama)`，这是为了保证 16GB 内存的 Windows、Linux、macOS 电脑都尽量能跑通本地多模态链路。

| ID | 模型 | 适用场景 |
| --- | --- | --- |
| `qwen3-vl-4b-ollama` | Qwen3-VL 4B | 16GB 默认工业路线，综合视频理解、OCR、结构化输出 |
| `qwen3-vl-2b-ollama` | Qwen3-VL 2B | 16GB 低配兜底 |
| `qwen2_5-vl-3b-ollama` | Qwen2.5-VL 3B | 16GB 成熟稳定备选 |
| `gemma3-4b-ollama` | Gemma 3 4B | 16GB 跨平台对照 |
| `qwen3-vl-8b-ollama` | Qwen3-VL 8B | 32GB 增强档 |
| `qwen2_5-vl-7b-ollama` | Qwen2.5-VL 7B | 32GB 稳定增强档 |
| `gemma3-12b-ollama` | Gemma 3 12B | 32GB 多模态对照 |
| `minicpm-v-ollama` | MiniCPM-V | 32GB 短视频理解对照 |
| `local-baseline` | OpenCV Local Baseline | 无 GPU 或无模型服务兜底 |

### 1. 安装 Ollama

Ollama 是本实验推荐的跨平台本地模型运行层：

- Windows：下载安装 https://ollama.com/download/windows，安装后重新打开 PowerShell。
- macOS：下载安装 https://ollama.com/download/mac，并启动 Ollama App。
- Linux：可执行 `curl -fsSL https://ollama.com/install.sh | sh`，然后运行 `ollama serve`。

Qwen3-VL 需要 Ollama 0.12.7 或更新版本，下载脚本会自动检查版本。

确认 Ollama 可用：

```bash
ollama --version
curl http://127.0.0.1:11434/api/version
```

### 2. 按内存档下载模型

16GB 机器建议下载 16GB 档：

```bash
python scripts/download_ollama_models.py --tier 16gb
```

32GB 机器可下载增强档：

```bash
python scripts/download_ollama_models.py --tier 32gb
```

只下载默认模型：

```bash
python scripts/download_ollama_models.py --model qwen3-vl-4b-ollama
```

教师机或 32GB 以上机器如果希望一次准备全部候选：

```bash
python scripts/download_ollama_models.py --tier all
```

如果 Ollama 没有启动或模型没有下载，系统会记录 `local_model_fallback` 事件并回退到本地 OpenCV baseline。这不是为了替代大模型，而是保证课堂环境不因为下载或显卡问题完全无法演示。

本实验默认向 Qwen3-VL 发送 4 张代表性关键帧，并在 Ollama 请求中设置 `think: false` 与 `format: json`。这样既能保留多帧理解能力，又能避免 Qwen3 系列把输出预算耗尽在 thinking 字段中。32GB 机器可以提高关键帧数量：

```bash
LOCAL_VLM_MAX_IMAGES=6 LOCAL_VLM_MAX_TOKENS=4500 python -m app.server
```

Windows PowerShell：

```powershell
$env:LOCAL_VLM_MAX_IMAGES="6"
$env:LOCAL_VLM_MAX_TOKENS="4500"
python -m app.server
```

---

## 第五阶段：视频理解、打标签与摘要生成

多模态理解编排层在 `app/understanding_service.py`。它把关键帧、时间戳、技术指标、音频/OCR 占位信息组织成模型输入，并要求模型返回严格 JSON：

```json
{
  "summary": "一段校园操场运动短视频，画面明亮，人物在跑道上移动。",
  "timeline": [
    {"start": 0, "end": 2, "event": "操场场景建立", "evidence": "绿色场地和跑道线"},
    {"start": 2, "end": 6, "event": "主体持续运动", "evidence": "关键帧中主体位置变化"}
  ],
  "visible_text": [],
  "audio_summary": "",
  "entities": ["操场", "跑道", "运动主体"],
  "actions": ["移动", "运动"],
  "tags": ["校园", "运动", "户外", "竖屏"],
  "risk": {
    "level": "pass",
    "score": 0,
    "categories": [],
    "evidence": []
  }
}
```

本实验保留 `VideoUnderstandingModel` 的本地可解释信号，是因为生产系统也需要低成本兜底特征：当模型超时、模型输出格式错误、显卡资源不足时，平台仍能根据基础风险策略做 fail-closed 处理。

---

## 第六阶段：内容审核策略

审核策略在 `moderate_analysis()` 中。它把理解结果转为平台发布决策：

| 状态 | 含义 |
| --- | --- |
| `published` | 自动发布到信息流 |
| `review` | 进入人工复核队列 |
| `rejected` | 直接拒绝发布 |

示例策略：

```python
if brightness < 25:
    score += 42
    reasons.append({
        "code": "too_dark",
        "level": "review",
        "message": "画面过暗，自动理解置信度下降，需要人工复核。",
        "evidence": brightness,
    })

if metrics["flash_ratio"] >= 0.15:
    score += 38
    reasons.append({
        "code": "flash_risk",
        "level": "review",
        "message": "检测到多次强亮度跳变，可能造成观看不适。",
        "evidence": metrics["flash_ratio"],
    })
```

同学们需要重点理解：审核系统不只输出一个布尔值，还必须输出可解释理由、证据值、模型来源和处理状态。否则平台无法做人工复核、申诉、审计和模型迭代。

---

## 第七阶段：发布到自建短视频 Demo 网站

启动网站：

```bash
python -m app.server
```

浏览器打开：

```text
http://127.0.0.1:5050
```

网站提供：

- 状态统计：全部、已发布、复核、拒绝。
- 模型选择：后台选择 Qwen、Gemma、MiniCPM 或本地 baseline。
- 视频信息流：视频播放器、标签、摘要、风险分、审核理由。
- 上传入口：上传本地 MP4/MOV/WEBM 后自动进入处理链路。
- 流处理事件：实时展示 ingest、queued、worker、frame_sample、understanding、moderation、publish 等事件。

上传交互采用“两阶段可见”设计：服务端先复制视频文件并写入一条 `processing` 记录，前端立即显示真实视频播放器；随后服务端把审核任务写入 SQLite `jobs` 本地耐久队列，由 `LocalReviewWorker` 异步消费并完成理解、标签和审核。等待期间只在摘要和标签区域显示 loading 占位符，视频未播放时仍显示真实媒体区域，不使用 loading 占位图遮挡视频。

这是一种“工业轻量版”折中：学生电脑默认不需要 Kafka、MinIO、Flink 或 Paimon 也能跑通；但系统边界已经按照工业里的上传服务、任务队列、审核 worker、模型服务、元数据存储来组织。课堂或教师机具备 Docker 环境时，可以把 SQLite jobs 替换为 Kafka 主题，把本地 `data/media` 替换为 MinIO，把审核结果写入 Paimon 或 ClickHouse。

核心 API：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/` | Demo 网站 |
| `GET` | `/api/health` | 服务健康状态 |
| `GET` | `/api/videos` | 查看全部视频及统计 |
| `GET` | `/api/videos/{status}` | 按状态查看视频 |
| `GET` | `/api/events` | 查看流处理事件 |
| `GET` | `/api/models` | 查看候选模型和当前模型 |
| `POST` | `/api/models/select` | 切换后台理解模型 |
| `POST` | `/api/demo` | 生成并处理内置样本流 |
| `POST` | `/api/upload` | 上传视频并处理 |
| `POST` | `/api/reset` | 清空演示数据库 |

---

## 第八阶段：接入 Kafka 视频进入事件

前七个阶段可以不依赖 Kafka，适合快速验证网站和审核链路。若要贴近前三章的流处理架构，请启动 Kafka 并运行生产者/消费者脚本。

### 1. 启动 Kafka

在仓库根目录可使用已有 `compose.yaml`：

```bash
docker compose up -d kafka
```

确认 Kafka 容器存在：

```bash
docker ps | grep bigdata-kafka
```

### 2. 创建主题

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

### 3. 启动消费者

终端 A：

```bash
source .venv/bin/activate
python scripts/kafka_ai_consumer.py
```

消费者会监听 `short_video_ingest`，读取视频路径，执行理解、审核、打标，并把结果写入 `short_video_result`。

### 4. 发送视频进入事件

终端 B：

```bash
source .venv/bin/activate
python scripts/kafka_video_producer.py
```

消息示例：

```json
{
  "title": "校园运动短视频",
  "path": "/absolute/path/to/campus_sports.mp4",
  "source": "generated-demo",
  "event_time": 1778718730000
}
```

此时网站仍然可以打开，观察 Kafka 消费者处理后写入的发布结果。

---

## 第九阶段：观察、测试与迭代

### 1. 本地流水线测试

```bash
python scripts/run_pipeline_once.py
```

### 2. 一键验收测试

```bash
python scripts/verify_demo.py
```

验收脚本会检查：

- 是否生成 3 个样本视频。
- 是否至少产生 1 个自动发布视频。
- 是否至少产生 1 个复核视频。
- 每个视频是否都有标签、摘要、帧级指标。
- 每个视频是否真正使用默认 `Qwen3-VL 4B (Ollama)`，而不是悄悄回退到 baseline。
- 事件日志是否记录了主要流水线步骤。

### 3. API 验证

网站启动后执行：

```bash
curl http://127.0.0.1:5050/api/health
curl http://127.0.0.1:5050/api/videos
curl http://127.0.0.1:5050/api/events
```

### 4. 迭代方向

同学们可以选择一个方向继续增强：

- 把 SQLite `jobs` 本地队列换成 Kafka 或 Pulsar。
- 把 SQLite `videos/events` 换成 PostgreSQL、ClickHouse 或 Paimon。
- 把视频文件上传到 MinIO，并只在数据库中保存对象存储 URL。
- 把帧级指标写入 Paimon，进行离线统计和热榜分析。
- 接入 OCR/ASR，审核画面文字和音频文本。
- 使用真实视觉模型替换规则标签器。
- 将 `review` 状态加入人工审核页面，支持通过/拒绝二次决策。
- 把标签结果接入第三章推荐系统，构建内容推荐流。

---

## 关键代码说明

### 1. 主流水线 `app/pipeline.py`

`ShortVideoPipeline.ingest_video()` 负责上传入口的快速返回：它把视频复制到媒体区，写入 `processing` 记录，让网站马上能展示真实视频。昂贵的理解和审核工作放到 `complete_video()` 中，由 worker 异步调用：

```python
record = pipeline.ingest_video(path, title=title, source="browser-upload")
enqueue_job("complete_video", {"record": record, "simulate_stream": True})
```

worker 消费任务后再执行：

```python
analysis = self.model.analyze(
    media_path,
    title=title,
    video_id=video_id,
    emit_event=add_event,
    simulate_delay_sec=0.03 if simulate_stream else 0.0,
)

moderation = moderate_analysis(analysis, title)
upsert_video(record)
add_event(video_id, "publish", publish_message, {"status": moderation["status"]})
```

它完成五件事：

1. 将上传文件复制到媒体区。
2. 写入 `processing` 视频记录，保证前端先显示视频。
3. 将 `complete_video` 任务写入 SQLite `jobs` 队列。
4. `LocalReviewWorker` 调用 `MultimodalUnderstandingService` 完成关键帧预处理、模型选择、Ollama 本地多模态模型调用或 OpenCV 兜底。
5. 调用审核策略，将 VLM 风险和平台规则合并，并将最终结果写回 SQLite。

### 2. 多模型理解服务 `app/understanding_service.py`

```python
candidate = get_active_model()
local = self.local_baseline.analyze(...)
preprocess = self.preprocessor.prepare(...)
vlm_payload = self.local_vlm_client.analyze_video(...)
```

这层是实验升级后的核心。它不把模型名称写死在流水线里，而是从 `model_registry.py` 读取当前后台选择；如果 Ollama 未启动或模型未下载，则记录 `local_model_fallback` 并用本地 baseline 保持演示链路可运行。

### 3. 本地任务队列与 worker

`app/job_queue.py` 提供 `enqueue_job()`、`claim_next_job()`、`complete_job()` 和 `fail_job()`。它用 SQLite 表模拟工业系统里的消息队列，具备本地耐久性：服务重启后，未完成任务可以重新进入队列。

`app/local_worker.py` 是单进程 worker，它在 FastAPI 启动时随服务启动：

```python
job = claim_next_job("local-review-worker")
pipeline.complete_video(job["payload"]["record"])
complete_job(job["id"])
```

这比直接在请求线程里跑模型更接近工业界的上传服务/审核 worker 解耦方式，同时仍然适合普通学生电脑。

### 4. 网站服务 `app/server.py`

FastAPI 提供页面、媒体访问和 JSON API：

```python
app = FastAPI(title="Short Video Stream Review Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.post("/api/demo")
def api_demo(overwrite: bool = False):
    if has_active_jobs():
        raise HTTPException(status_code=409, detail="pipeline is already running")
    jobs = _enqueue_demo(overwrite)
    return {"started": True, "processing": True, "queued": len(jobs)}
```

这里不再把模型推理挂在请求线程或临时后台任务上，而是写入本地耐久队列，再由 worker 消费。网站可以持续刷新事件列表，观察 `queued`、`worker`、`understanding`、`moderation`、`publish` 的完整过程。

模型选择 API 也在这一层：

```python
@app.post("/api/models/select")
def api_select_model(selection: ModelSelectionRequest):
    active = set_active_model(selection.model_id)
    return {"active": active.to_dict()}
```

### 5. Kafka 脚本

`scripts/kafka_video_producer.py` 只发送视频进入事件；`scripts/kafka_ai_consumer.py` 才负责真正处理视频。这种拆分体现了消息队列的核心价值：上传侧和 AI 审核侧解耦。

---

## 故障排除 / FAQ

### 1. `ModuleNotFoundError: No module named 'cv2'`

说明没有安装 OpenCV，或没有激活虚拟环境。执行：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__)"
```

### 2. Python 3.14 安装 NumPy 很慢

请选择 Python 3.11 或 3.12 创建虚拟环境。原因是 Python 3.14 上某些科学计算包可能暂时没有稳定预编译 wheel。

### 3. `ffmpeg and ffprobe are required`

安装 ffmpeg，并确认命令行可访问：

```bash
ffmpeg -version
ffprobe -version
```

### 4. 端口 `5050` 被占用

修改 `app/server.py` 中的端口：

```python
uvicorn.run("app.server:app", host="127.0.0.1", port=5051, reload=False)
```

### 5. Kafka 连接失败

检查 Kafka 是否启动：

```bash
docker ps | grep bigdata-kafka
```

检查主题是否存在：

```bash
docker exec -it bigdata-kafka /opt/kafka/bin/kafka-topics.sh \
  --list \
  --bootstrap-server localhost:9092
```

### 6. 视频能上传但不能播放

浏览器对编码格式有要求。建议使用 H.264/AAC 或常见 MP4 文件。内置样本由 ffmpeg 生成，默认可以播放。

### 7. 已选择 Qwen 但事件里出现 `local_model_fallback`

这说明后台模型配置已经选择 Qwen，但 Ollama 没有启动、模型没有下载，或模型输出不是严格 JSON。课堂演示时可以临时接受这个回退；如果要完成真实本地模型验收，请确认：

```bash
ollama list
python scripts/download_ollama_models.py --model qwen3-vl-4b-ollama
curl http://127.0.0.1:11434/api/tags
python scripts/verify_demo.py
```

如果 16GB 机器推理太慢，保持默认 `LOCAL_VLM_MAX_IMAGES=4`；如果 32GB 机器希望增强视频理解，可设置 `LOCAL_VLM_MAX_IMAGES=6` 或选择 `qwen3-vl-8b-ollama`。

真实部署中建议将 Ollama 服务、业务 API、Kafka 消费者拆成独立进程，避免模型推理影响网站稳定性。

---

## 参考资源

- FastAPI 官方文档：https://fastapi.tiangolo.com/
- OpenCV Python 文档：https://docs.opencv.org/
- Apache Kafka 文档：https://kafka.apache.org/documentation/
- ffmpeg 文档：https://ffmpeg.org/documentation.html
- Ollama Vision 文档：https://docs.ollama.com/capabilities/vision
- Ollama Qwen3-VL：https://ollama.com/library/qwen3-vl
- Ollama Qwen2.5-VL：https://ollama.com/library/qwen2.5vl
- Ollama Gemma 3：https://ollama.com/library/gemma3
- Ollama MiniCPM-V：https://ollama.com/library/minicpm-v
- 可选 MP4 样本：https://filesamples.com/formats/mp4
