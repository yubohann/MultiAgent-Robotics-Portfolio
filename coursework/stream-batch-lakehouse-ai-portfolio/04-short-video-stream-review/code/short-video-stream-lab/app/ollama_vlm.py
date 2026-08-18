"""Ollama client for local multimodal video-frame understanding.

Ollama 通过本机 HTTP API 暴露模型能力，适合学生在 Windows/Linux/macOS 上统一运行。
本客户端只发送经过预处理的关键帧和压缩后的指标，避免把完整视频直接塞给模型。
"""

import base64
import json
import re
from pathlib import Path

import requests

from .config import (
    LOCAL_VLM_MAX_IMAGES,
    LOCAL_VLM_MAX_TOKENS,
    LOCAL_VLM_TIMEOUT_SEC,
    OLLAMA_BASE_URL,
)
from .model_registry import ModelCandidate


class OllamaModelError(RuntimeError):
    """Raised when local Ollama is unavailable or the model cannot produce JSON."""


def _extract_json(text: str) -> dict:
    """Extract JSON from a model response that may include Markdown fences.

    即使 prompt 要求只输出 JSON，部分本地模型仍可能包一层 ```json。
    这里做容错解析，但如果完全没有 JSON，仍然抛错并触发上层 fallback。
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _image_base64(path: str) -> str:
    """Read one keyframe image and encode it for Ollama's `images` field."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _representative_frames(frames: list[dict], limit: int) -> list[dict]:
    """Pick evenly spaced keyframes when the preprocessor produced too many.

    预处理阶段可能保留 12 张候选帧；为了照顾 16GB 机器，默认只向 VLM 发送 4 张。
    """
    if len(frames) <= limit:
        return frames
    if limit <= 1:
        return frames[:1]
    indexes = [round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)]
    return [frames[index] for index in indexes]


class OllamaVLMClient:
    """Use Ollama's local HTTP API for cross-platform multimodal inference."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        """Store the Ollama base URL without trailing slash for safe URL joining."""
        self.base_url = base_url.rstrip("/")

    def list_local_models(self) -> set[str]:
        """Return model names already pulled into the local Ollama runtime."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaModelError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama first."
            ) from exc
        payload = response.json()
        names = set()
        for model in payload.get("models", []):
            name = model.get("name")
            if name:
                names.add(name)
                names.add(name.split(":")[0])
        return names

    def is_model_available(self, model_name: str) -> bool:
        """Check whether the requested Ollama model is available before inference."""
        return model_name in self.list_local_models()

    def analyze_video(
        self,
        *,
        candidate: ModelCandidate,
        title: str,
        preprocess: dict,
        local_metrics: dict,
    ) -> dict:
        """Run one multimodal analysis request through Ollama.

        输入包括标题、视频 metadata、关键帧抽样原因和 baseline 指标。
        输出必须是结构化 JSON，便于后续审核策略稳定读取字段。
        """
        if not self.is_model_available(candidate.ollama_model):
            raise OllamaModelError(
                f"Ollama model is not downloaded: {candidate.ollama_model}. "
                f"Run: {candidate.pull_command}"
            )

        selected_frames = _representative_frames(
            preprocess.get("keyframes", []),
            max(1, LOCAL_VLM_MAX_IMAGES),
        )
        images = [_image_base64(frame["file"]) for frame in selected_frames]
        if not images:
            raise OllamaModelError("no keyframes available for Ollama inference")

        # Ollama chat 接口支持 images 数组；`think: False` 避免 Qwen3 系列输出冗长思考过程。
        payload = {
            "model": candidate.ollama_model,
            "stream": False,
            "think": False,
            "format": "json",
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt(title, preprocess, local_metrics, selected_frames),
                    "images": images,
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": LOCAL_VLM_MAX_TOKENS,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=LOCAL_VLM_TIMEOUT_SEC,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaModelError(f"Ollama chat failed: {exc}") from exc

        content = response.json().get("message", {}).get("content", "")
        try:
            return _extract_json(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise OllamaModelError(f"Ollama returned non-JSON content: {content[-1200:]}") from exc

    def _prompt(
        self,
        title: str,
        preprocess: dict,
        local_metrics: dict,
        selected_frames: list[dict],
    ) -> str:
        """Build the compact Chinese prompt sent to the local VLM.

        prompt 明确约束字段名、风险等级和 JSON 输出格式，这让后续 Python 代码可以
        像处理普通结构化数据一样处理模型结果。
        """
        keyframes = [
            {
                "timestamp_sec": frame["timestamp_sec"],
                "reason": frame["reason"],
                "brightness": frame["brightness"],
                "motion": frame["motion"],
                "scene_change": frame["scene_change"],
            }
            for frame in selected_frames
        ]
        metadata = preprocess.get("metadata", {})
        compact_metrics = {
            "brightness": local_metrics.get("brightness"),
            "motion": local_metrics.get("motion"),
            "flash_count": local_metrics.get("flash_count"),
            "flash_ratio": local_metrics.get("flash_ratio"),
            "red_ratio_avg": local_metrics.get("red_ratio_avg"),
            "green_ratio_avg": local_metrics.get("green_ratio_avg"),
            "blue_ratio_avg": local_metrics.get("blue_ratio_avg"),
        }
        context = {
            "title": title,
            "metadata": {
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "fps": metadata.get("fps"),
                "duration_sec": metadata.get("duration_sec"),
            },
            "keyframes": keyframes,
            "local_metrics": compact_metrics,
        }
        return (
            "/no_think\n"
            "你是短视频内容理解与审核模型。请直接输出合法 JSON，不要解释。"
            "除 visible_text 中确实来自画面的原始文字外，所有字符串值必须使用简体中文，"
            "包括 summary、timeline.event、timeline.evidence、audio_summary、entities、actions、tags、"
            "risk.categories 和 risk.evidence。tags 必须是简短中文名词短语，不要输出英文标签。"
            "JSON 字段固定为：summary(string), timeline(array), visible_text(array), "
            "audio_summary(string), entities(array), actions(array), tags(array), risk(object)。"
            "timeline 元素使用 {start,end,event,evidence}。"
            "risk 使用 {level,score,categories,evidence}，level 只能是 pass、review 或 reject。"
            f"输入信息：{json.dumps(context, ensure_ascii=False)}"
        )
