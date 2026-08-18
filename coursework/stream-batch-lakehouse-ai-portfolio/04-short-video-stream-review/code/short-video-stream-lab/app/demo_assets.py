"""Provide real public demo videos for the lab.

本作品集版本固定使用已经下载到 data/incoming 的 Pexels 公开视频：
运动场跑步、夜间街景和舞台灯光。验收与截图不得使用合成动画样本。
"""

from .config import INCOMING_DIR, ensure_directories


DEMO_TITLES = {
    "campus_sports": "真实运动场跑步短视频",
    "night_scene_review": "真实夜间低照度街景",
    "flashy_clip_review": "真实舞台灯光闪烁片段",
}

REAL_DEMO_FILES = {
    "campus_sports": "real_campus_sports_[REDACTED].mp4",
    "night_scene_review": "real_low_light_review_[REDACTED].mp4",
    "flashy_clip_review": "real_flash_risk_review_[REDACTED].mp4",
}


def ensure_demo_videos(overwrite: bool = False) -> list[dict]:
    """Return the three real public demo videos.

    The overwrite argument is kept for compatibility with the original classroom
    helper command, but this portfolio does not synthesize replacement videos.
    """
    ensure_directories()
    videos = []
    for kind, title in DEMO_TITLES.items():
        real_path = INCOMING_DIR / REAL_DEMO_FILES[kind]
        if not real_path.exists():
            raise FileNotFoundError(
                f"missing required Pexels real demo video for {kind}: {real_path}. "
                "Do not fall back to generated animation for this portfolio."
            )
        videos.append({"path": real_path, "title": title, "source": "pexels-public-real-video"})
    return videos
