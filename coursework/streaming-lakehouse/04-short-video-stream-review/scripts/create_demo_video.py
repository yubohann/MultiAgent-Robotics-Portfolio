"""Command-line helper to verify the real public demo videos."""

import _bootstrap  # noqa: F401 -- side-effect import: allow direct execution from scripts/
from app.demo_assets import ensure_demo_videos


def main() -> None:
    """Print the real demo videos used by the report screenshots."""
    videos = ensure_demo_videos()
    for item in videos:
        print(f"real-video-ready: {item['path']} ({item['title']}) source={item['source']}")


if __name__ == "__main__":
    main()
