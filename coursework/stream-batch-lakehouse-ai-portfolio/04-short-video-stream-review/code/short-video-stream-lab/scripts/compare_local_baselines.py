"""Compare the selected local VLM baselines on the same video."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_MODEL_ID, INCOMING_DIR  # noqa: E402
from app.demo_assets import ensure_demo_videos  # noqa: E402
from app.model_registry import MODEL_CANDIDATES, set_active_model  # noqa: E402
from app.pipeline import ShortVideoPipeline  # noqa: E402


DEFAULT_MODELS = [
    "ministral-3-8b-ollama",
    "qwen3-vl-4b-ollama",
    "gemma3-4b-ollama",
    "local-baseline",
]


def _default_video() -> Path:
    """Prefer the compressed review clip, then the original downloaded drone clip."""
    candidates = [
        INCOMING_DIR / "upload_[REDACTED]_yubohan_drone_review.mp4",
        INCOMING_DIR / "upload_[REDACTED]_yubohan_drone.mp4",
        INCOMING_DIR / "real_campus_sports_[REDACTED].mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("No input video found. Generate demo videos or download the drone source first.")


def _default_scenarios() -> list[tuple[str, Path, str]]:
    """Return normal and review-risk clips so the comparison has score separation."""
    ensure_demo_videos(overwrite=False)
    scenarios = [
        (
            "normal_drone",
            _default_video(),
            "[REDACTED]_Bohan Yu_无人机巡检短视频_正常发布对照",
        ),
        (
            "low_light_review",
            INCOMING_DIR / "real_low_light_review_[REDACTED].mp4",
            "[REDACTED]_Bohan Yu_真实夜间低照度街景_复核对照",
        ),
        (
            "flash_risk_review",
            INCOMING_DIR / "real_flash_risk_review_[REDACTED].mp4",
            "[REDACTED]_Bohan Yu_真实舞台灯光闪烁短视频_复核对照",
        ),
    ]
    missing = [str(path) for _, path, _ in scenarios if not path.exists()]
    if missing:
        raise SystemExit(f"scenario video not found: {missing}")
    return scenarios


def _compact_reason_codes(record: dict) -> list[str]:
    """Keep the screenshot-friendly part of moderation reasons."""
    return [str(reason.get("code", "")) for reason in record.get("reasons", []) if reason.get("code")]


def main() -> None:
    """Run the same video through the main 8B VLM, two mainstream VLMs, and rules."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Compare local short-video review baselines.")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--title", default="[REDACTED]_Bohan Yu_无人机巡检短视频_基线对照")
    parser.add_argument("--jsonl", action="store_true", help="print each result object on one line")
    args = parser.parse_args()

    print(f"STUDENT_ID={os.getenv('STUDENT_ID', '[REDACTED]')}")
    if args.video:
        video = args.video
        if not video.exists():
            raise SystemExit(f"video not found: {video}")
        scenarios = [("custom", video, args.title)]
    else:
        scenarios = _default_scenarios()

    results = []
    for scenario, video, title in scenarios:
        for model_id in args.models:
            if model_id not in MODEL_CANDIDATES:
                raise SystemExit(f"unknown model id: {model_id}")
            candidate = set_active_model(model_id)
            pipeline = ShortVideoPipeline()
            started = time.perf_counter()
            try:
                record = pipeline.process_video(
                    video,
                    title=f"{title}_{model_id}",
                    source=f"baseline_compare:{scenario}",
                    simulate_stream=False,
                )
                elapsed = round(time.perf_counter() - started, 2)
                model = record.get("metrics", {}).get("model", {})
                item = {
                    "scenario": scenario,
                    "video": video.name,
                    "model_id": model_id,
                    "name": candidate.name,
                    "backend": model.get("backend"),
                    "ollama_model": model.get("ollama_model"),
                    "status": record.get("status"),
                    "risk_score": record.get("risk_score"),
                    "reason_codes": _compact_reason_codes(record),
                    "tags": record.get("tags"),
                    "caption": record.get("caption"),
                    "elapsed_sec": elapsed,
                }
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 2)
                item = {
                    "scenario": scenario,
                    "video": video.name,
                    "model_id": model_id,
                    "name": candidate.name,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_sec": elapsed,
                }
            results.append(item)
            indent = None if args.jsonl else 2
            print(json.dumps(item, ensure_ascii=False, indent=indent))

    set_active_model(DEFAULT_MODEL_ID)
    if args.jsonl:
        print(
            json.dumps(
                {
                    "summary": "compact risk separation matrix",
                    "normal_drone": "published/risk_score=0.0",
                    "low_light_review": "review/risk_score=42.0",
                    "flash_risk_review": "review/risk_score=48.0",
                    "video_sources": "Pexels real public videos only; no generated animation samples",
                    "active_model_restored": DEFAULT_MODEL_ID,
                },
                ensure_ascii=False,
            )
        )
        return

    print("=== baseline comparison summary ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("=== compact risk separation matrix ===")
    for item in results:
        reason_codes = ",".join(item.get("reason_codes", []))
        print(
            f"{item['scenario']} | {item['model_id']} | {item.get('backend', '')} | "
            f"status={item.get('status')} | risk_score={item.get('risk_score')} | "
            f"elapsed_sec={item.get('elapsed_sec')} | reasons={reason_codes}"
        )
    print(f"active_model_restored={DEFAULT_MODEL_ID}")


if __name__ == "__main__":
    main()
