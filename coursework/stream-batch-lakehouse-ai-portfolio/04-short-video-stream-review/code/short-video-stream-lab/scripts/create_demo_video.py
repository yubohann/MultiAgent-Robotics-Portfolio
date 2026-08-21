"""Command-line helper to verify the real public demo videos.

本作品集版本不再生成动画样本。运行本脚本只检查并列出 data/incoming
下已下载的三段 Pexels 公开视频，确保截图和报告使用真实素材。
"""

from pathlib import Path
import sys

# 脚本位于 scripts/，直接运行时 Python 默认找不到 app 包。
# 把项目根目录加入 sys.path 后，就可以复用正式后端模块，而不是复制一份逻辑。
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
