"""Central configuration for the local short-video review runtime.

这个文件只放“全局约定”，避免路径、阈值、模型名散落在各个模块里。
课堂版默认使用本地目录和 SQLite；工业环境中这些常量通常会来自配置中心、
Kubernetes ConfigMap、环境变量或服务发现系统。
"""

from pathlib import Path
import os


# BASE_DIR 指向项目根目录，后续所有运行时目录都从这里派生，
# 这样 Windows、Linux、macOS 下只要工作目录正确，路径就不会写死。
BASE_DIR = Path(__file__).resolve().parents[1]

# data/ 是本实验的本地“对象存储 + 元数据工作区”。
# incoming 保存用户上传或下载的原始视频，media 保存网站可访问的媒体文件，
# frames/audio/models/state 分别保存预处理产物、音频、模型辅助文件和 SQLite 状态。
DATA_DIR = BASE_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
MEDIA_DIR = DATA_DIR / "media"
FRAME_DIR = MEDIA_DIR / "frames"
AUDIO_DIR = MEDIA_DIR / "audio"
MODEL_DIR = DATA_DIR / "models"
STATE_DIR = DATA_DIR / "state"
DB_PATH = STATE_DIR / "short_video_demo.sqlite3"

# OpenCV baseline 的抽样参数。baseline 只做可解释的亮度、运动、色彩和闪烁分析，
# 作用是兜底和教学对照，不等同于真正的多模态理解模型。
SAMPLE_FPS = 2
MAX_SAMPLED_FRAMES = 80
ANALYSIS_WIDTH = 160

# VLM 关键帧参数。VLM_FRAME_WIDTH 控制发送给 Ollama 的图片尺寸，
# MAX_VLM_KEYFRAMES 是预处理阶段最多保留的候选关键帧数，
# LOCAL_VLM_MAX_IMAGES 是真正发给本地模型的图片数，默认 4 张以照顾 16GB 机器。
VLM_FRAME_WIDTH = 448
MAX_VLM_KEYFRAMES = 12
LOCAL_VLM_MAX_IMAGES = int(os.getenv("LOCAL_VLM_MAX_IMAGES", "4"))

# 下面两个阈值决定哪些帧会被认为是“场景切换”或“运动峰值”。
# 它们不是内容安全规则，而是关键帧选择策略的一部分。
SCENE_CHANGE_THRESHOLD = 42.0
MOTION_PEAK_THRESHOLD = 18.0

# 默认模型选择本机已验证的 Ministral 3 8B Vision；两个 4B 级主流模型用于基线对照。
# Ollama 相关参数允许通过环境变量覆盖，便于教师机或实验室环境统一调整。
DEFAULT_MODEL_ID = "ministral-3-8b-ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LOCAL_VLM_TIMEOUT_SEC = int(os.getenv("LOCAL_VLM_TIMEOUT_SEC", "180"))
LOCAL_VLM_MAX_TOKENS = int(os.getenv("LOCAL_VLM_MAX_TOKENS", "3500"))

# 默认允许 fallback，是为了课堂环境可演示；正式验收时报告需要说明是否发生了 fallback。
ALLOW_LOCAL_MODEL_FALLBACK = os.getenv("ALLOW_LOCAL_MODEL_FALLBACK", "1") != "0"

# 标题高风险词是一个最小策略示例，用来演示“模型理解 + 规则审核”的组合。
# 真实平台会使用更完整的词表、上下文分类器、人工复核和灰度策略。
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
    """Create every runtime directory needed by the demo.

    FastAPI 服务、脚本和测试都会调用这个函数。提前创建目录可以让后续代码
    专注于业务逻辑，不必在每次写文件前重复处理父目录不存在的问题。
    """
    for directory in (DATA_DIR, INCOMING_DIR, MEDIA_DIR, FRAME_DIR, AUDIO_DIR, MODEL_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
