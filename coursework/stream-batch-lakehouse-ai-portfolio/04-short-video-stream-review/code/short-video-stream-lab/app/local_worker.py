"""Background worker that consumes local review jobs.

上传接口只负责让视频快速进入页面和队列；真正耗时的模型理解放在 worker 中。
这种拆分对应工业系统里的 Web API + 消费者进程，可以避免一次上传请求被模型推理拖住。
"""

import threading
import time

from .job_queue import claim_next_job, complete_job, fail_job, requeue_interrupted_jobs
from .pipeline import ShortVideoPipeline
from .storage import add_event


class LocalReviewWorker:
    """Single-process worker that mirrors an industrial queue consumer on student laptops."""

    def __init__(self, pipeline: ShortVideoPipeline, *, poll_interval_sec: float = 0.8) -> None:
        """Bind the worker to one pipeline instance and configure polling frequency."""
        self.pipeline = pipeline
        self.poll_interval_sec = poll_interval_sec
        self.worker_id = "local-review-worker"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the daemon thread if it is not already running.

        启动前先恢复上次异常退出留下的 running 任务，保证学生重启服务后不会卡住。
        """
        if self._thread and self._thread.is_alive():
            return
        requeued = requeue_interrupted_jobs()
        if requeued:
            add_event(None, "worker", "恢复上次中断的本地审核任务", {"requeued": requeued})
        self._thread = threading.Thread(target=self._run, name=self.worker_id, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker loop to stop and wait briefly for a clean exit."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        """Poll the local queue forever until FastAPI shuts down."""
        while not self._stop.is_set():
            job = claim_next_job(self.worker_id)
            if job is None:
                # 没有任务时短暂休眠，避免空循环占满 CPU。
                time.sleep(self.poll_interval_sec)
                continue
            self._handle_job(job)

    def _handle_job(self, job: dict) -> None:
        """Execute one claimed job and persist success or failure.

        当前只有 `complete_video` 一种任务类型；保留 kind 字段是为了让后续扩展
        OCR、ASR、人工复核回调等任务时不需要重写队列结构。
        """
        payload = job["payload"]
        video_id = (payload.get("record") or {}).get("id")
        add_event(
            video_id,
            "worker",
            "本地审核 worker 开始处理队列任务",
            {"job_id": job["id"], "kind": job["kind"]},
        )
        try:
            if job["kind"] != "complete_video":
                raise ValueError(f"unknown job kind: {job['kind']}")
            # complete_video 会完成多模态理解、策略审核、封面生成和最终状态更新。
            self.pipeline.complete_video(
                payload["record"],
                simulate_stream=bool(payload.get("simulate_stream", True)),
            )
            complete_job(job["id"])
            add_event(
                video_id,
                "worker",
                "本地审核 worker 完成队列任务",
                {"job_id": job["id"]},
            )
        except Exception as exc:
            fail_job(job["id"], str(exc))
            add_event(
                video_id,
                "worker_failed",
                "本地审核 worker 处理失败",
                {"job_id": job["id"], "error": str(exc)},
            )
