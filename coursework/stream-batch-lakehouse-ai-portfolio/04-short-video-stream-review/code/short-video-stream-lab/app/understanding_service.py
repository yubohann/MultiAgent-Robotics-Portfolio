"""Orchestrate baseline analysis, preprocessing, local VLM, and fallback behavior.

这是视频理解层的门面：pipeline 不直接关心当前选的是 Qwen、Gemma、MiniCPM 还是 baseline，
只调用本服务并获得统一结构的 caption、tags、risk、metrics 和 keyframes。
"""

from pathlib import Path
import re
from typing import Callable

from .config import ALLOW_LOCAL_MODEL_FALLBACK
from .ollama_vlm import OllamaModelError, OllamaVLMClient
from .model_registry import ModelCandidate, get_active_model
from .preprocessing import VideoPreprocessor
from .video_understanding import VideoUnderstandingModel

EventCallback = Callable[[str | None, str, str, dict | None], None]


def _as_list(value: object) -> list:
    """Normalize model fields that may be returned as string, list, null, or scalar."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


_COMMON_ENGLISH_TAGS = {
    "aerial view": "航拍视角",
    "animation": "动画",
    "circle": "圆形元素",
    "housing": "住宅建筑",
    "neighborhood": "居民区",
    "red": "红色元素",
    "residential": "住宅区",
    "street": "道路",
    "suburb": "郊区住宅",
    "white background": "白色背景",
}


def _has_chinese(value: str) -> bool:
    """Return true when the string already contains CJK characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _is_mostly_english(value: str) -> bool:
    """Detect English model output so reports do not mix caption languages."""
    letters = re.findall(r"[A-Za-z]", value)
    return bool(letters) and len(letters) >= 12 and not _has_chinese(value)


def _zh_tag(value: object) -> str:
    """Translate common English visual tags into concise Chinese labels."""
    text = str(value).strip()
    if not text:
        return ""
    normalized = text.lower()
    return _COMMON_ENGLISH_TAGS.get(normalized, text)


def _zh_summary(value: str, local_caption: str) -> str:
    """Prefer Chinese captions; fall back to deterministic Chinese local caption."""
    text = str(value or "").strip()
    if _is_mostly_english(text):
        return local_caption
    return text or local_caption


def _normalize_timeline(value: object) -> list[dict]:
    """Normalize VLM timeline output into a predictable list of event dictionaries."""
    timeline = _as_list(value)
    normalized = []
    for index, item in enumerate(timeline):
        if isinstance(item, dict):
            start = item.get("start", item.get("t", item.get("timestamp_sec", index)))
            end = item.get("end", start)
            event = item.get("event") or item.get("reason") or item.get("description") or "关键帧事件"
            evidence = item.get("evidence") or item.get("detail") or ""
        else:
            start = index
            end = index
            event = str(item)
            evidence = ""
        normalized.append({"start": start, "end": end, "event": str(event), "evidence": str(evidence)})
    return normalized


def _normalize_risk(value: object) -> dict:
    """Normalize VLM risk output into pass/review/reject plus score and evidence."""
    if not isinstance(value, dict):
        value = {}
    level = str(value.get("level", "pass")).lower()
    level_map = {
        "safe": "pass",
        "low": "pass",
        "medium": "review",
        "moderate": "review",
        "high": "reject",
        "block": "reject",
    }
    level = level_map.get(level, level)
    if level not in {"pass", "review", "reject"}:
        level = "review"
    default_score = {"pass": 0, "review": 45, "reject": 80}[level]
    try:
        score = float(value.get("score", default_score))
    except (TypeError, ValueError):
        score = float(default_score)
    return {
        "level": level,
        "score": score,
        "categories": _as_list(value.get("categories")),
        "evidence": _as_list(value.get("evidence")),
    }


class MultimodalUnderstandingService:
    """Industrial-style understanding chain with a selectable VLM backend."""

    def __init__(self) -> None:
        """Create reusable analyzer instances for one application process."""
        self.preprocessor = VideoPreprocessor()
        self.local_baseline = VideoUnderstandingModel()
        self.local_vlm_client = OllamaVLMClient()

    def analyze(
        self,
        video_path: Path,
        *,
        title: str,
        video_id: str | None = None,
        emit_event: EventCallback | None = None,
        simulate_delay_sec: float = 0.03,
    ) -> dict:
        """Analyze one video with the active model and return a unified result.

        执行顺序是：记录模型选择 -> baseline 抽样指标 -> 关键帧预处理 -> VLM 调用。
        baseline 始终先运行，因为即使 VLM 成功，它的 metrics 也会参与审核策略。
        """
        candidate = get_active_model()
        if emit_event:
            emit_event(
                video_id,
                "model_select",
                f"当前理解模型：{candidate.name}",
                candidate.to_dict(),
            )

        local = self.local_baseline.analyze(
            video_path,
            video_id=video_id,
            emit_event=emit_event,
            simulate_delay_sec=simulate_delay_sec,
        )
        preprocess = self.preprocessor.prepare(video_path, video_id=video_id or "unknown", emit_event=emit_event)

        if candidate.mode == "local_baseline":
            # 用户主动选择 baseline 时，不视为模型故障；backend 标记为 local_baseline。
            return self._merge_local(local, preprocess, candidate, fallback_reason="")

        try:
            # 远程云模型已被移除；这里所有 VLM 候选都通过本地 Ollama 权重运行。
            vlm_payload = self.local_vlm_client.analyze_video(
                candidate=candidate,
                title=title,
                preprocess=preprocess,
                local_metrics=local["metrics"],
            )
            if emit_event:
                emit_event(
                    video_id,
                    "vlm_understanding",
                    "本地多模态模型完成结构化视频理解",
                    {
                        "model": candidate.name,
                        "summary": vlm_payload.get("summary", ""),
                        "risk": vlm_payload.get("risk", {}),
                    },
                )
            return self._merge_vlm(local, preprocess, candidate, vlm_payload)
        except OllamaModelError as exc:
            if not ALLOW_LOCAL_MODEL_FALLBACK:
                raise
            # fallback 是课堂可运行性的保护网；报告中仍应说明它不是 SOTA 模型能力。
            if emit_event:
                emit_event(
                    video_id,
                    "local_model_fallback",
                    "本地多模态模型未就绪，已回退到 OpenCV baseline",
                    {"model": candidate.name, "error": str(exc)},
                )
            return self._merge_local(local, preprocess, candidate, fallback_reason=str(exc))

    def _merge_local(
        self,
        local: dict,
        preprocess: dict,
        candidate: ModelCandidate,
        *,
        fallback_reason: str,
    ) -> dict:
        """Build the common analysis shape when only the local baseline is available."""
        metrics = dict(local["metrics"])
        metrics["preprocess"] = {
            "keyframes": len(preprocess.get("keyframes", [])),
            "has_audio": bool(preprocess.get("audio_path")),
            "sampling_strategy": preprocess.get("sampling_strategy", {}),
        }
        model_info = {
            "selected_id": candidate.id,
            "selected_name": candidate.name,
            "backend": "local_baseline_fallback" if fallback_reason else "local_baseline",
            "serving_model": candidate.serving_model,
            "ollama_model": candidate.ollama_model,
            "fallback_reason": fallback_reason,
        }
        metrics["model"] = model_info
        return {
            **local,
            "caption": local["caption"],
            "tags": [_zh_tag(tag) for tag in local["tags"] if _zh_tag(tag)],
            "timeline": [
                {
                    "start": frame["timestamp_sec"],
                    "end": frame["timestamp_sec"],
                    "event": f"关键帧：{frame['reason']}",
                    "evidence": f"motion={frame['motion']}, scene_change={frame['scene_change']}",
                }
                for frame in preprocess.get("keyframes", [])[:6]
            ],
            "visible_text": [],
            "audio_summary": "未接入 ASR 或视频无音频轨。",
            "entities": [],
            "actions": [],
            "model_risk": {"level": "pass", "score": 0, "categories": [], "evidence": []},
            "model": model_info,
            "metrics": metrics,
            "keyframes": preprocess.get("keyframes", []),
        }

    def _merge_vlm(
        self,
        local: dict,
        preprocess: dict,
        candidate: ModelCandidate,
        payload: dict,
    ) -> dict:
        """Merge VLM semantic output with deterministic local metrics.

        VLM 更擅长语义摘要、OCR 和动作描述；baseline 更稳定地提供亮度、运动、闪烁等数值。
        合并后同一套审核策略就能同时利用两类信号。
        """
        tags = list(dict.fromkeys([_zh_tag(tag) for tag in [*_as_list(payload.get("tags")), *local["tags"]] if _zh_tag(tag)]))
        summary = _zh_summary(str(payload.get("summary") or ""), local["caption"])
        timeline = _normalize_timeline(payload.get("timeline"))
        risk = _normalize_risk(payload.get("risk"))
        metrics = dict(local["metrics"])
        metrics["preprocess"] = {
            "keyframes": len(preprocess.get("keyframes", [])),
            "has_audio": bool(preprocess.get("audio_path")),
            "sampling_strategy": preprocess.get("sampling_strategy", {}),
        }
        model_info = {
            "selected_id": candidate.id,
            "selected_name": candidate.name,
            "backend": "local_ollama_vlm",
            "serving_model": candidate.serving_model,
            "ollama_model": candidate.ollama_model,
            "fallback_reason": "",
        }
        metrics["model"] = model_info
        return {
            **local,
            "caption": summary,
            "tags": tags,
            "timeline": timeline,
            "visible_text": _as_list(payload.get("visible_text")),
            "audio_summary": str(payload.get("audio_summary") or ""),
            "entities": _as_list(payload.get("entities")),
            "actions": _as_list(payload.get("actions")),
            "model_risk": risk,
            "model": model_info,
            "metrics": metrics,
            "keyframes": preprocess.get("keyframes", []),
        }
