"""Command-line helper to verify the real public demo videos."""

from pathlib import Path
import sys

# Allow direct execution from scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo_assets import ensure_demo_videos  # noqa: E402


def main() -> None:
    """Print the real demo videos used by the report screenshots."""
    videos = ensure_demo_videos(overwrite=True)
    for item in videos:
        print(f"real-video-ready: {item['path']} ({item['title']}) source={item['source']}")


if __name__ == "__main__":
    main()
