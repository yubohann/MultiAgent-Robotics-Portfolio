"""Upload the prepared drone clip and save API evidence for screenshots."""

import json
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parents[1]
LOG_DIR = LAB_ROOT / "logs_[REDACTED]"
BASE_URL = "http://127.0.0.1:5050"
STUDENT_ID = "[REDACTED]"
VIDEO_TITLE = "[REDACTED]_\u4f59\u535a\u6db5_\u81ea\u9009\u4e0a\u4f20_\u65e0\u4eba\u673a\u5de1\u68c0\u77ed\u89c6\u9891"
VIDEO_PATH = ROOT / "data" / "incoming" / "upload_[REDACTED]_yubohan_drone_review.mp4"


def _save(name: str, payload: dict) -> None:
    """Write one JSON payload as UTF-8 evidence."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_get(path: str) -> dict:
    response = requests.get(f"{BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _json_post(path: str, **kwargs) -> dict:
    response = requests.post(f"{BASE_URL}{path}", timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Reset, select the main model, upload the drone video, and save evidence."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not VIDEO_PATH.exists():
        raise SystemExit(f"video not found: {VIDEO_PATH}")

    _save("05_api_reset.json", _json_post("/api/reset"))
    _save(
        "05_api_model_select.json",
        _json_post("/api/models/select", json={"model_id": "ministral-3-8b-ollama"}),
    )
    _save("05_api_health_after_model_select.json", _json_get("/api/health"))

    with VIDEO_PATH.open("rb") as video:
        upload_payload = _json_post(
            "/api/upload",
            data={"title": VIDEO_TITLE},
            files={"video": (VIDEO_PATH.name, video, "video/mp4")},
        )
    _save("06_upload_response_processing.json", upload_payload)
    video_id = upload_payload["video"]["id"]

    _save("07_api_videos_immediate_processing.json", _json_get("/api/videos"))
    _save("08_api_health_immediate_jobs.json", _json_get("/api/health"))
    _save("09_api_events_immediate_trace.json", _json_get("/api/events"))

    final_videos = None
    for _ in range(90):
        health = _json_get("/api/health")
        videos = _json_get("/api/videos")
        target = next((item for item in videos.get("videos", []) if item.get("id") == video_id), None)
        if target and target.get("status") != "processing" and not health.get("processing"):
            final_videos = videos
            break
        time.sleep(2)
    if final_videos is None:
        raise SystemExit("worker did not finish uploaded drone video in time")

    _save("07_api_videos_final.json", final_videos)
    _save("08_api_health_final.json", _json_get("/api/health"))
    _save("09_api_events_final.json", _json_get("/api/events"))
    _save("11_api_models_selector.json", _json_get("/api/models"))

    target = next(item for item in final_videos["videos"] if item["id"] == video_id)
    model = target["metrics"]["model"]
    print(f"STUDENT_ID={STUDENT_ID}")
    print(f"uploaded_video_id={video_id}")
    print(f"title={target['title']}")
    print(f"status={target['status']}")
    print(f"risk_score={target['risk_score']}")
    print(f"backend={model['backend']}")
    print(f"selected_id={model['selected_id']}")
    print(f"caption={target['caption'][:180]}")
    print(f"tags={', '.join(target['tags'][:12])}")
    print(f"logs={LOG_DIR}")


if __name__ == "__main__":
    main()
