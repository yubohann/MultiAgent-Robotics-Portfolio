"""Registry of selectable local video-language model candidates.

所有模型候选集中写在这里，前端、下载脚本和理解服务共用同一份配置。
这种注册表模式可以避免“前端显示一个模型，后端实际调用另一个模型”的教学事故。
"""

from dataclasses import asdict, dataclass
from typing import Literal

from .config import DEFAULT_MODEL_ID
from .storage import get_setting, set_setting

ModelMode = Literal["local_ollama_vlm", "local_baseline"]


@dataclass(frozen=True)
class ModelCandidate:
    """Metadata describing one model option shown in the backend selector."""

    id: str
    name: str
    family: str
    serving_model: str
    ollama_model: str
    mode: ModelMode
    memory_tier: str
    recommended_for: str
    hardware: str
    supports_audio: bool
    supports_video: bool
    estimated_disk_gb: float
    estimated_memory_gb: str
    pull_command: str
    notes: str

    def to_dict(self) -> dict:
        """Serialize model metadata for API responses and event logs."""
        return asdict(self)


# 候选模型按本机已下载、16GB 和 32GB 三类组织。默认选择本机已验证的 Ministral 3 8B vision。
# 需要基线对照时，只额外使用两个 4B 级主流视觉模型，避免误拉 7B/8B/12B 大模型。
MODEL_CANDIDATES: dict[str, ModelCandidate] = {
    "ministral-3-3b-ollama": ModelCandidate(
        id="ministral-3-3b-ollama",
        name="Ministral 3 3B Vision (Ollama)",
        family="Mistral",
        serving_model="ollama:ministral-3:3b",
        ollama_model="ministral-3:3b",
        mode="local_ollama_vlm",
        memory_tier="local-existing-light",
        recommended_for="本机已下载的轻量 vision 对照模型；用于和 8B 模型比较速度、资源占用和理解质量",
        hardware="16GB/32GB 机器均可尝试；当前机器已通过 ollama show 验证具备 vision capability",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=3.0,
        estimated_memory_gb="约 5-9GB",
        pull_command="ollama pull ministral-3:3b",
        notes="Ollama 本地模型显示 capabilities 包含 completion、vision、tools。硬盘空间紧张时优先使用该已下载基线，不额外占用磁盘。",
    ),
    "qwen3-vl-4b-ollama": ModelCandidate(
        id="qwen3-vl-4b-ollama",
        name="Qwen3-VL 4B (Ollama)",
        family="Qwen",
        serving_model="ollama:qwen3-vl:4b",
        ollama_model="qwen3-vl:4b",
        mode="local_ollama_vlm",
        memory_tier="16GB",
        recommended_for="主流 4B 视觉基线：对照短视频理解、OCR 和结构化输出稳定性",
        hardware="Windows/Linux/macOS，16GB 内存可运行；建议一次只跑一个模型",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=3.3,
        estimated_memory_gb="约 6-10GB，随关键帧数量变化",
        pull_command="ollama pull qwen3-vl:4b",
        notes="主流对照候选。Ollama 官方库标注 Qwen3-VL 4B 为 Text/Image，大小约 3.3GB。",
    ),
    "qwen3-vl-2b-ollama": ModelCandidate(
        id="qwen3-vl-2b-ollama",
        name="Qwen3-VL 2B (Ollama)",
        family="Qwen",
        serving_model="ollama:qwen3-vl:2b",
        ollama_model="qwen3-vl:2b",
        mode="local_ollama_vlm",
        memory_tier="16GB-safe",
        recommended_for="低配兜底：优先保证所有学生电脑能跑通本地多模态链路",
        hardware="Windows/Linux/macOS，16GB 内存更稳",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=1.9,
        estimated_memory_gb="约 4-7GB",
        pull_command="ollama pull qwen3-vl:2b",
        notes="适合课堂批量演示和低配机器。",
    ),
    "qwen2_5-vl-3b-ollama": ModelCandidate(
        id="qwen2_5-vl-3b-ollama",
        name="Qwen2.5-VL 3B (Ollama)",
        family="Qwen",
        serving_model="ollama:qwen2.5vl:3b",
        ollama_model="qwen2.5vl:3b",
        mode="local_ollama_vlm",
        memory_tier="16GB",
        recommended_for="成熟稳定备选：文档/OCR/结构化输出能力好",
        hardware="Windows/Linux/macOS，16GB 内存可运行",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=3.2,
        estimated_memory_gb="约 6-10GB",
        pull_command="ollama pull qwen2.5vl:3b",
        notes="如果 Qwen3-VL 本地表现不稳定，可切换到该成熟版本。",
    ),
    "gemma3-4b-ollama": ModelCandidate(
        id="gemma3-4b-ollama",
        name="Gemma 3 4B (Ollama)",
        family="Gemma",
        serving_model="ollama:gemma3:4b",
        ollama_model="gemma3:4b",
        mode="local_ollama_vlm",
        memory_tier="16GB",
        recommended_for="跨平台轻量视觉理解对照模型",
        hardware="Windows/Linux/macOS，16GB 内存可运行",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=3.3,
        estimated_memory_gb="约 6-10GB",
        pull_command="ollama pull gemma3:4b",
        notes="Ollama 官方库提供 4B 多模态版本，适合对照实验。",
    ),
    "ministral-3-8b-ollama": ModelCandidate(
        id="ministral-3-8b-ollama",
        name="Ministral 3 8B Vision (Ollama)",
        family="Mistral",
        serving_model="ollama:ministral-3:8b",
        ollama_model="ministral-3:8b",
        mode="local_ollama_vlm",
        memory_tier="32GB",
        recommended_for="本机已下载的 8B vision 模型；用于最高分路线的本地多模态短视频审核证据",
        hardware="建议 32GB 内存或更高；当前机器已通过 ollama show 验证具备 vision capability",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=6.0,
        estimated_memory_gb="约 10-16GB",
        pull_command="ollama pull ministral-3:8b",
        notes="Ollama 本地模型显示 capabilities 包含 completion、vision、tools，可接收关键帧图片完成结构化理解。",
    ),
    "qwen3-vl-8b-ollama": ModelCandidate(
        id="qwen3-vl-8b-ollama",
        name="Qwen3-VL 8B (Ollama)",
        family="Qwen",
        serving_model="ollama:qwen3-vl:8b",
        ollama_model="qwen3-vl:8b",
        mode="local_ollama_vlm",
        memory_tier="32GB",
        recommended_for="32GB 机器默认增强档：更强的短视频理解和推理",
        hardware="建议 32GB 内存或更高",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=6.1,
        estimated_memory_gb="约 10-16GB",
        pull_command="ollama pull qwen3-vl:8b",
        notes="32GB 学生机或教师机推荐。",
    ),
    "qwen2_5-vl-7b-ollama": ModelCandidate(
        id="qwen2_5-vl-7b-ollama",
        name="Qwen2.5-VL 7B (Ollama)",
        family="Qwen",
        serving_model="ollama:qwen2.5vl:7b",
        ollama_model="qwen2.5vl:7b",
        mode="local_ollama_vlm",
        memory_tier="32GB",
        recommended_for="32GB 稳定增强档：成熟视觉代理与 OCR 能力",
        hardware="建议 32GB 内存或更高",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=6.0,
        estimated_memory_gb="约 10-16GB",
        pull_command="ollama pull qwen2.5vl:7b",
        notes="Qwen2.5-VL 7B 是很稳的本地增强对照。",
    ),
    "gemma3-12b-ollama": ModelCandidate(
        id="gemma3-12b-ollama",
        name="Gemma 3 12B (Ollama)",
        family="Gemma",
        serving_model="ollama:gemma3:12b",
        ollama_model="gemma3:12b",
        mode="local_ollama_vlm",
        memory_tier="32GB",
        recommended_for="32GB 多模态对照模型",
        hardware="建议 32GB 内存或更高",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=8.1,
        estimated_memory_gb="约 12-20GB",
        pull_command="ollama pull gemma3:12b",
        notes="更大但不是默认；适合教师机演示。",
    ),
    "minicpm-v-ollama": ModelCandidate(
        id="minicpm-v-ollama",
        name="MiniCPM-V (Ollama)",
        family="MiniCPM",
        serving_model="ollama:minicpm-v",
        ollama_model="minicpm-v",
        mode="local_ollama_vlm",
        memory_tier="32GB",
        recommended_for="网络短视频对照：多图/视频理解能力较强",
        hardware="建议 32GB 内存；16GB 可尝试但不作为默认",
        supports_audio=False,
        supports_video=True,
        estimated_disk_gb=5.5,
        estimated_memory_gb="约 10-16GB",
        pull_command="ollama pull minicpm-v",
        notes="Ollama 官方库为 MiniCPM-V 2.6 系列；适合对照。",
    ),
    "local-baseline": ModelCandidate(
        id="local-baseline",
        name="OpenCV Local Baseline",
        family="Local",
        serving_model="local-opencv-rules",
        ollama_model="",
        mode="local_baseline",
        memory_tier="any",
        recommended_for="无 Ollama、无模型权重时的教学兜底",
        hardware="CPU",
        supports_audio=False,
        supports_video=False,
        estimated_disk_gb=0.0,
        estimated_memory_gb="< 1GB",
        pull_command="",
        notes="只基于亮度、运动、色彩和闪烁规则，不代表 SOTA 能力。",
    ),
}


def list_model_candidates() -> list[dict]:
    """Return every candidate as dictionaries for `/api/models`."""
    return [candidate.to_dict() for candidate in MODEL_CANDIDATES.values()]


def get_model_candidate(model_id: str | None) -> ModelCandidate:
    """Resolve a model id, falling back to the course default when missing."""
    if model_id and model_id in MODEL_CANDIDATES:
        return MODEL_CANDIDATES[model_id]
    return MODEL_CANDIDATES[DEFAULT_MODEL_ID]


def get_active_model() -> ModelCandidate:
    """Read the currently selected model from SQLite settings."""
    return get_model_candidate(get_setting("active_model_id", DEFAULT_MODEL_ID))


def set_active_model(model_id: str) -> ModelCandidate:
    """Persist the selected model id after validating it exists in the registry."""
    if model_id not in MODEL_CANDIDATES:
        raise ValueError(f"unknown model id: {model_id}")
    set_setting("active_model_id", model_id)
    return MODEL_CANDIDATES[model_id]
