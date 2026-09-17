"""Download a small public MP4 sample into the incoming directory."""

from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import INCOMING_DIR, ensure_directories  # noqa: E402


DEFAULT_URL = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"


def main() -> None:
    """Download the configured URL and save it as an incoming test video."""
    ensure_directories()
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    output_path = INCOMING_DIR / "internet-sample-640x360.mp4"
    print(f"downloading: {url}")
    # A timeout keeps the script from waiting forever on a slow network.
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
