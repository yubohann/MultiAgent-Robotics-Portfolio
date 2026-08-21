# 实验 4：实时短视频流智能审核与发布综合实验

姓名：Bohan Yu
学号：[REDACTED]
班级：23计算师资1班
GPU：NVIDIA GeForce RTX 4090

本目录是实验四的 GitHub 提交包，包含最终实验报告、FastAPI 审核系统代码、真实短视频样例、截图证据、日志证据和 Kafka 加分项证据。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `code/short-video-stream-lab` | FastAPI + SQLite + Ollama VLM 短视频审核 Demo |
| `code/short-video-stream-lab/data/incoming` | 4 个小体积真实短视频样例 |
| `screenshots_[REDACTED]` | 环境、模型、API、页面和 Kafka 截图 |
| `logs_[REDACTED]` | 环境、模型、验收、API、事件流和 Kafka 日志 |
| `短视频审核实验报告_[REDACTED]_Bohan Yu.docx` | 最终 Word 版实验报告 |
| `截图命令与命名清单_[REDACTED].md` | 截图命令、画面要求和文件命名 |
| `SUBMISSION_NOTES.md` | GitHub 提交范围说明 |

## 模型与工作量

- 主模型：`ministral-3:8b`，作为本地 Ollama 多模态审核模型。
- 对照模型：`qwen3-vl:4b`、`gemma3:4b`、`local-baseline`、`ministral-3:3b`。
- Web 页面模型区展示全部模型，截图见 `screenshots_[REDACTED]/11_model_selector_all_models_[REDACTED].png`。
- 基线对照结果见 `logs_[REDACTED]/04_baseline_comparison_jsonl.log`。

## 最短复现流程

运行前需要本机已安装 FFmpeg/FFprobe，并已在 Ollama 中准备 `ministral-3:8b` 等模型。提交包不包含 FFmpeg 二进制和 Ollama 模型文件，避免把本机工具缓存提交到 GitHub。

```powershell
cd code\short-video-stream-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:PYTHONIOENCODING = "utf-8"
$env:STUDENT_ID = "[REDACTED]"
$env:ALLOW_LOCAL_MODEL_FALLBACK = "0"
$env:LOCAL_VLM_MAX_IMAGES = "3"
$env:LOCAL_VLM_TIMEOUT_SEC = "300"

python scripts\verify_demo.py
python scripts\compare_local_baselines.py
python -m app.server
```

服务地址：

```text
http://127.0.0.1:5050
```

## 核心证据

| 证据 | 文件 |
| --- | --- |
| 环境验证 | `logs_[REDACTED]/01_env.log` |
| Python 依赖 | `logs_[REDACTED]/02_python_dependencies.log` |
| Ollama 模型 | `logs_[REDACTED]/03_ollama_model.log` |
| 严格验收 | `logs_[REDACTED]/04_verify_demo.log` |
| 多模型基线对照 | `logs_[REDACTED]/04_baseline_comparison_jsonl.log` |
| API 最终视频结果 | `logs_[REDACTED]/07_api_videos_final.json` |
| API 队列健康状态 | `logs_[REDACTED]/08_api_health_final.json` |
| 全链路事件 | `logs_[REDACTED]/09_api_events_final.json` |
| 模型选择区 | `logs_[REDACTED]/11_model_selector_all_models.log` |
| Kafka topic | `logs_[REDACTED]/10_kafka_topics.log` |
| Kafka producer | `logs_[REDACTED]/12_kafka_producer.log` |
| Kafka result topic | `logs_[REDACTED]/13_kafka_result.log` |

## 说明

GitHub 提交包不包含本地 `.venv`、FFmpeg 二进制包、运行缓存、SQLite 临时状态和 128 MB 原始无人机视频。提交中保留的是压缩后的真实样例视频，便于助教直接复现。
