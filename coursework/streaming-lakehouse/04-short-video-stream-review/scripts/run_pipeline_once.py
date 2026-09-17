"""Run the full review pipeline synchronously on the built-in demo videos."""

from pathlib import Path

import _bootstrap  # noqa: F401 -- side-effect import: allow direct execution from scripts/
from app.demo_assets import ensure_demo_videos
from app.pipeline import ShortVideoPipeline
from app.storage import stats


def main() -> None:
    """Process every demo video once and print compact publication results."""
    pipeline = ShortVideoPipeline()
    videos = ensure_demo_videos()
    for item in videos:
        record = pipeline.process_video(
            Path(item["path"]),
            title=item["title"],
            source=item["source"],
            simulate_stream=False,
        )
        print(f"{record['id']} {record['status']} {record['title']} tags={record['tags']}")
    print(f"stats={stats()}")


if __name__ == "__main__":
    main()
