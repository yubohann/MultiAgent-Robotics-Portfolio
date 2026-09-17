"""Run the full review pipeline synchronously on the built-in demo videos."""

from pathlib import Path
import sys

# Allow direct execution from scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo_assets import ensure_demo_videos  # noqa: E402
from app.pipeline import ShortVideoPipeline  # noqa: E402
from app.storage import stats  # noqa: E402


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
