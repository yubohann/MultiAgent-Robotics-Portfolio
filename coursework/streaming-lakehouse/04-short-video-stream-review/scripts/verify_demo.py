"""One-command verification for the short-video review lab."""

from pathlib import Path

import _bootstrap  # noqa: F401 -- side-effect import: allow direct execution from scripts/
from app.config import DEFAULT_MODEL_ID
from app.demo_assets import ensure_demo_videos
from app.model_registry import set_active_model
from app.pipeline import ShortVideoPipeline
from app.storage import clear_db, list_events, list_videos, stats


def main() -> None:
    """Run deterministic checks and fail fast if the required lab path is broken."""
    clear_db()
    # Verification runs the default local multimodal model.
    set_active_model(DEFAULT_MODEL_ID)
    pipeline = ShortVideoPipeline()
    for item in ensure_demo_videos():
        pipeline.process_video(
            Path(item["path"]),
            title=item["title"],
            source=item["source"],
            simulate_stream=False,
        )

    videos = list_videos()
    current_stats = stats()
    # The assertions cover counts, status, events, tags, summaries, frames, backend and preprocessing.
    assert len(videos) == 3, f"expected 3 videos, got {len(videos)}"
    assert current_stats["published"] >= 1, current_stats
    assert current_stats["review"] >= 1, current_stats
    assert len(list_events()) >= 12, "expected pipeline events"
    for video in videos:
        assert video["tags"], f"missing tags for {video['id']}"
        assert video["caption"], f"missing caption for {video['id']}"
        assert video["metrics"]["sampled_frames"] > 0, f"missing frame samples for {video['id']}"
        assert video["metrics"]["model"]["selected_id"] == DEFAULT_MODEL_ID
        assert video["metrics"]["model"]["backend"] == "local_ollama_vlm", video["metrics"]["model"]
        assert video["metrics"]["preprocess"]["keyframes"] > 0
    print("verification passed")
    print(current_stats)


if __name__ == "__main__":
    main()
