"""Run the full review pipeline synchronously on the built-in demo videos.

这个脚本绕过网站和本地队列，适合快速检查 pipeline 本身是否能完成
“理解 -> 审核 -> 打标签 -> 入库”。网站异步链路仍以 FastAPI + worker 为准。
"""

from pathlib import Path
import sys

# 允许从 scripts/ 目录直接导入 app 包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo_assets import ensure_demo_videos  # noqa: E402
from app.pipeline import ShortVideoPipeline  # noqa: E402
from app.storage import stats  # noqa: E402


def main() -> None:
    """Process every demo video once and print compact publication results."""
    pipeline = ShortVideoPipeline()
    videos = ensure_demo_videos(overwrite=False)
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
