"""Central configuration for the local short-video review runtime."""

from pathlib import Path
import os


# BASE_DIR anchors every runtime directory below it.
BASE_DIR = Path(__file__).resolve().parents[1]

# data/ is the local object store and metadata workspace.
# incoming, media, frames, audio, models and state hold the runtime artifacts.
DATA_DIR = BASE_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
MEDIA_DIR = DATA_DIR / "media"
FRAME_DIR = MEDIA_DIR / "frames"
AUDIO_DIR = MEDIA_DIR / "audio"
MODEL_DIR = DATA_DIR / "models"
STATE_DIR = DATA_DIR / "state"
DB_PATH = STATE_DIR / "short_video_demo.sqlite3"

# OpenCV baseline sampling parameters for brightness, motion, color and flash analysis.
SAMPLE_FPS = 2
MAX_SAMPLED_FRAMES = 80
ANALYSIS_WIDTH = 160

# VLM keyframe parameters. VLM_FRAME_WIDTH sets the image width sent to Ollama,
# MAX_VLM_KEYFRAMES caps the preprocessed candidate frames and
# LOCAL_VLM_MAX_IMAGES caps the frames sent to the local model.
VLM_FRAME_WIDTH = 448
MAX_VLM_KEYFRAMES = 12
LOCAL_VLM_MAX_IMAGES = int(os.getenv("LOCAL_VLM_MAX_IMAGES", "4"))

# The two thresholds below drive scene-change and motion-peak detection for keyframe selection.
SCENE_CHANGE_THRESHOLD = 42.0
MOTION_PEAK_THRESHOLD = 18.0

# The default model is the locally validated Ministral 3 8B Vision.
# Ollama settings are overridable through environment variables.
DEFAULT_MODEL_ID = "ministral-3-8b-ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LOCAL_VLM_TIMEOUT_SEC = int(os.getenv("LOCAL_VLM_TIMEOUT_SEC", "180"))
LOCAL_VLM_MAX_TOKENS = int(os.getenv("LOCAL_VLM_MAX_TOKENS", "3500"))

# Fallback stays enabled so the classroom demo always runs.
ALLOW_LOCAL_MODEL_FALLBACK = os.getenv("ALLOW_LOCAL_MODEL_FALLBACK", "1") != "0"

# High-risk title words are a minimal example of model understanding plus rule-based moderation.
BANNED_TITLE_WORDS = {
    "adult",
    "bloody",
    "gambling",
    "violent",
    "violence",
    "成人",
    "博彩",
    "赌博",
    "暴力",
    "血腥",
}


def ensure_directories() -> None:
    """Create every runtime directory needed by the demo."""
    for directory in (DATA_DIR, INCOMING_DIR, MEDIA_DIR, FRAME_DIR, AUDIO_DIR, MODEL_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
