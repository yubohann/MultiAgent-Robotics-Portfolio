"""FastAPI web service for the short-video review demo.

这个文件负责把后端 pipeline 暴露成网站和 JSON API：
上传视频、选择本地模型、查看队列状态、查看事件日志、读取发布结果。
耗时的视频理解不在请求线程里同步完成，而是交给 LocalReviewWorker 异步处理。
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
import re

import uvicorn
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .config import BASE_DIR, INCOMING_DIR, MEDIA_DIR, ensure_directories
from .demo_assets import ensure_demo_videos
from .job_queue import enqueue_job, has_active_jobs, job_stats
from .local_worker import LocalReviewWorker
from .model_registry import get_active_model, list_model_candidates, set_active_model
from .ollama_vlm import OllamaVLMClient, OllamaModelError
from .pipeline import ShortVideoPipeline
from .storage import add_event, clear_db, list_events, list_videos, stats


ensure_directories()
# 应用进程内共享一个 pipeline 和一个 worker。课堂版这样最简单；
# 工业环境通常会把 API 服务和 worker 拆成不同进程或不同容器。
pipeline = ShortVideoPipeline()
worker = LocalReviewWorker(pipeline)
ollama_client = OllamaVLMClient()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start and stop the local worker with the FastAPI application."""
    worker.start()
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(
    title="Short Video Stream Review Demo",
    description="Real-time short video understanding, moderation, tagging, and publishing demo.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ModelSelectionRequest(BaseModel):
    """Request body for selecting the active understanding model."""

    model_id: str


def _models_payload() -> dict:
    """Build model metadata for the frontend selector.

    除了注册表中的静态信息，还会查询 Ollama 本机已下载模型，
    让界面能提示“已下载/未下载”。
    """
    candidates = list_model_candidates()
    try:
        local_names = ollama_client.list_local_models()
        ollama_ready = True
    except OllamaModelError:
        local_names = set()
        ollama_ready = False
    for candidate in candidates:
        ollama_model = candidate.get("ollama_model")
        candidate["downloaded"] = True if not ollama_model else ollama_model in local_names
    active = get_active_model().to_dict()
    active["downloaded"] = True if not active.get("ollama_model") else active["ollama_model"] in local_names
    return {"active": active, "candidates": candidates, "ollama_ready": ollama_ready}


def _enqueue_review(record: dict, *, simulate_stream: bool = True) -> dict:
    """Create a local queue job that will finish understanding and moderation."""
    job = enqueue_job(
        "complete_video",
        {"record": record, "simulate_stream": simulate_stream},
        job_id=f"complete-{record['id']}",
    )
    add_event(
        record["id"],
        "queued",
        "审核任务已写入本地耐久队列",
        {"job_id": job["id"], "queue": "sqlite-local"},
    )
    return job


def _enqueue_demo(overwrite: bool = False) -> list[dict]:
    """Generate demo videos, ingest them, and enqueue their review jobs."""
    add_event(None, "system", "开始准备内置短视频样本", {"overwrite": overwrite})
    jobs = []
    for descriptor in ensure_demo_videos(overwrite=overwrite):
        record = pipeline.ingest_video(
            Path(descriptor["path"]),
            title=descriptor["title"],
            source=descriptor["source"],
        )
        jobs.append(_enqueue_review(record, simulate_stream=True))
    add_event(None, "system", "内置样本已入队，等待 worker 异步处理", {"jobs": len(jobs)})
    return jobs


def _safe_upload_name(filename: str) -> str:
    """Strip paths and unsafe characters from a browser-provided filename."""
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(name).stem).strip("-") or "upload"
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}"


@app.get("/")
def index(request: Request):
    """Render the single-page demo UI."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/media/{filename:path}")
def media(filename: str):
    """Serve processed media files and thumbnails from the local media directory."""
    path = MEDIA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(path)


@app.get("/api/health")
def health():
    """Return runtime health, active model, Ollama readiness, and queue statistics."""
    models = _models_payload()
    jobs = job_stats()
    return {
        "ok": True,
        "processing": jobs["queued"] + jobs["running"] > 0,
        "active_model": models["active"],
        "ollama_ready": models["ollama_ready"],
        "jobs": jobs,
    }


@app.get("/api/models")
def api_models():
    """Return all selectable model candidates and the current active model."""
    return _models_payload()


@app.post("/api/models/select")
def api_select_model(selection: ModelSelectionRequest):
    """Switch the active model when no review jobs are in flight."""
    if has_active_jobs():
        raise HTTPException(status_code=409, detail="pipeline is running")
    try:
        active = set_active_model(selection.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    add_event(None, "model_config", f"后台模型已切换为 {active.name}", active.to_dict())
    return {"active": active.to_dict()}


@app.get("/api/videos")
def api_videos():
    """Return every video record for the feed and dashboard counts."""
    return {"videos": list_videos(), "stats": stats()}


@app.get("/api/videos/{status}")
def api_videos_by_status(status: str):
    """Return videos filtered by processing/published/review/rejected status."""
    if status not in {"processing", "published", "review", "rejected"}:
        raise HTTPException(status_code=400, detail="invalid status")
    return {"videos": list_videos(status=status), "stats": stats()}


@app.get("/api/events")
def api_events():
    """Return recent event-log entries for the timeline panel."""
    return {"events": list_events(limit=80)}


@app.post("/api/demo")
def api_demo(overwrite: bool = False):
    """Start the built-in sample-video flow by enqueueing demo review jobs."""
    if has_active_jobs():
        raise HTTPException(status_code=409, detail="pipeline is already running")
    jobs = _enqueue_demo(overwrite)
    return {"started": True, "processing": True, "queued": len(jobs)}


@app.post("/api/upload")
async def api_upload(
    video: UploadFile = File(...),
    title: str | None = Form(default=None),
):
    """Accept a browser upload, show it immediately, then enqueue model work.

    关键点：请求返回前只做文件保存、processing 记录写入和任务入队；
    真正耗时的理解/审核由 worker 后台执行，避免用户等待模型完成。
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="missing video file")

    original_name = _safe_upload_name(video.filename) or f"upload-{int(time.time())}.mp4"
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".webm"}:
        raise HTTPException(status_code=400, detail="only mp4/mov/m4v/webm video files are accepted")

    safe_name = f"{int(time.time())}-{original_name}"
    target = INCOMING_DIR / safe_name
    try:
        # 先保存到 incoming，保留上传原始物；pipeline 会复制一份到可公开访问的 media。
        target.write_bytes(await video.read())
        pending_record = pipeline.ingest_video(
            target,
            title=title or Path(original_name).stem,
            source="browser-upload",
        )
        job = _enqueue_review(pending_record, simulate_stream=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "started": True,
        "processing": True,
        "path": str(target),
        "video": pending_record,
        "job": job,
    }


@app.post("/api/reset")
def api_reset():
    """Clear videos, events, and jobs when no active work is running."""
    if has_active_jobs():
        raise HTTPException(status_code=409, detail="pipeline is running")
    clear_db()
    add_event(None, "system", "演示数据库已清空", {})
    return {"ok": True}


def main() -> None:
    """Run the development server for local classroom use."""
    uvicorn.run("app.server:app", host="127.0.0.1", port=5050, reload=False)


if __name__ == "__main__":
    main()
