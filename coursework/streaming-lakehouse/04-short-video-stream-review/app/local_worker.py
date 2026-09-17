"""Background worker that consumes local review jobs."""

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
        """Start the daemon thread if it is not already running."""
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
                # Sleep briefly when no job is waiting.
                time.sleep(self.poll_interval_sec)
                continue
            self._handle_job(job)

    def _handle_job(self, job: dict) -> None:
        """Execute one claimed job and persist success or failure."""
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
