# 截图命令与命名清单 [REDACTED]

所有截图保存到 `screenshots_[REDACTED]`。命令在 `code/short-video-stream-lab` 下执行，先设置：

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:STUDENT_ID = "[REDACTED]"
$env:ALLOW_LOCAL_MODEL_FALLBACK = "0"
```

| 截图文件名 | 命令或页面 | 画面要求 |
| --- | --- | --- |
| `00_public_drone_source_license_[REDACTED].png` | 打开公开视频来源或 `logs_[REDACTED]/00_real_public_video_sources_[REDACTED].txt` | 展示真实公开视频来源和许可说明 |
| `01_env_python_ffmpeg_ollama_[REDACTED].png` | `python --version; ffmpeg -version; ollama --version` | 展示 Python、FFmpeg、Ollama 环境 |
| `02_python_dependencies_[REDACTED].png` | `pip list` 或 `Get-Content logs_[REDACTED]/02_python_dependencies.log` | 展示 FastAPI、OpenCV、Kafka 等依赖 |
| `03_ollama_8b_vision_model_1_[REDACTED].png` | `Get-Content logs_[REDACTED]/03_ollama_model.log` | 展示 `ministral-3:8b` 模型证据 |
| `03_ollama_8b_vision_model_2_[REDACTED].png` | `ollama list` | 展示本地已安装模型列表 |
| `04_verify_demo_8b_vlm_passed_[REDACTED].png` | `python scripts\verify_demo.py` | 展示 `verification passed` |
| `04b1_baseline_comparison_[REDACTED].png` | `python scripts\compare_local_baselines.py` | 展示多模型对照摘要 |
| `05_fastapi_server_start_[REDACTED].png` | `python -m app.server` | 展示 Uvicorn 启动到 `127.0.0.1:5050` |
| `06_upload_processing_visible_[REDACTED].png` | 浏览器打开 `http://127.0.0.1:5050` | 展示上传后 processing 或 review 状态 |
| `07_api_videos_[REDACTED].png` | `Invoke-RestMethod http://127.0.0.1:5050/api/videos` | 展示视频审核结果 JSON |
| `08_api_health_jobs_[REDACTED].png` | `Invoke-RestMethod http://127.0.0.1:5050/api/health` | 展示队列和模型健康状态 |
| `09_api_events_trace_[REDACTED].png` | `Invoke-RestMethod http://127.0.0.1:5050/api/events` | 展示 ingest、preprocess、model、publish/review 事件链 |
| `10_final_review_result_[REDACTED].png` | Web 页面最终结果区 | 展示 published/review、risk score、tags |
| `11_model_selector_all_models_[REDACTED].png` | Web 页面模型选择区 | 展示主模型和所有基线模型 |
| `12_kafka_topics_[REDACTED].png` | `Get-Content logs_[REDACTED]/10_kafka_topics.log` | 展示 `short_video_ingest_[REDACTED]` 和 `short_video_result_[REDACTED]` |
| `13_kafka_ingest_and_result_[REDACTED].png` | `Get-Content logs_[REDACTED]/12_kafka_producer.log; Get-Content logs_[REDACTED]/13_kafka_result.log` | 展示 Kafka 输入消息和审核结果消息 |

Kafka 加分项只提交一条无人机视频 ingest/result 消息；其它视频的审核结果由 FastAPI、baseline 和 events 日志覆盖。
