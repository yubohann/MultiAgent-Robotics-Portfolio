"""One-command verification for the short-video review lab.

本脚本用于提交前自检和助教快速验收：它清空运行状态、选择默认 Ministral 3 8B Vision、
处理三段样本视频，并断言结果中包含标签、摘要、关键帧和本地 VLM 后端信息。
"""

from pathlib import Path
import sys

# 允许直接执行 `python scripts/verify_demo.py`。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo_assets import ensure_demo_videos  # noqa: E402
from app.config import DEFAULT_MODEL_ID  # noqa: E402
from app.model_registry import set_active_model  # noqa: E402
from app.pipeline import ShortVideoPipeline  # noqa: E402
from app.storage import clear_db, list_events, list_videos, stats  # noqa: E402


def main() -> None:
    """Run deterministic checks and fail fast if the required lab path is broken."""
    clear_db()
    # 验收使用当前默认本地多模态模型；本机配置为已下载的 Ministral 3 8B Vision。
    set_active_model(DEFAULT_MODEL_ID)
    pipeline = ShortVideoPipeline()
    for item in ensure_demo_videos(overwrite=True):
        pipeline.process_video(
            Path(item["path"]),
            title=item["title"],
            source=item["source"],
            simulate_stream=False,
        )

    videos = list_videos()
    current_stats = stats()
    # 下面的断言覆盖“数量、状态、事件、标签、摘要、抽帧、模型后端、预处理”。
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
