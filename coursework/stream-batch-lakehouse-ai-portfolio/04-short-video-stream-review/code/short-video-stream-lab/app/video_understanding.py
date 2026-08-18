"""Explainable local baseline for video understanding and moderation.

这个模块不是 SOTA 多模态模型，而是一个可解释的 OpenCV baseline：
它从采样帧中计算亮度、运动、色彩和闪烁，再生成基础标签和审核信号。
当 Ollama 模型未下载或不可用时，系统可以用它兜底，保证课堂演示不断链。
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import (
    ANALYSIS_WIDTH,
    BANNED_TITLE_WORDS,
    MAX_SAMPLED_FRAMES,
    SAMPLE_FPS,
)

EventCallback = Callable[[str | None, str, str, dict | None], None]


@dataclass
class FrameSignal:
    """Low-level visual features extracted from one sampled frame."""

    brightness: float
    colorfulness: float
    motion: float
    red_ratio: float
    green_ratio: float
    blue_ratio: float
    flash_delta: float


def _brightness(frame: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB channels using standard perceptual weights."""
    rgb = frame.astype(np.float32)
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _colorfulness(frame: np.ndarray) -> float:
    """Estimate colorfulness with a simple red-green and yellow-blue opponent metric."""
    rgb = frame.astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(std_root + 0.3 * mean_root)


def _channel_ratios(frame: np.ndarray) -> tuple[float, float, float]:
    """Return rough dominant-channel ratios for red, green, and blue regions."""
    rgb = frame.astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    total_pixels = frame.shape[0] * frame.shape[1]
    red_ratio = np.count_nonzero((red > green * 1.2) & (red > blue * 1.2)) / total_pixels
    green_ratio = np.count_nonzero((green > red * 1.15) & (green > blue * 1.15)) / total_pixels
    blue_ratio = np.count_nonzero((blue > red * 1.15) & (blue > green * 1.15)) / total_pixels
    return float(red_ratio), float(green_ratio), float(blue_ratio)


def _summarize(values: list[float]) -> dict:
    """Summarize a numeric signal with avg/max/min for compact JSON output."""
    if not values:
        return {"avg": 0.0, "max": 0.0, "min": 0.0}
    return {
        "avg": round(float(np.mean(values)), 2),
        "max": round(float(np.max(values)), 2),
        "min": round(float(np.min(values)), 2),
    }


class VideoUnderstandingModel:
    """A lightweight, explainable video understanding model for the lab demo."""

    def analyze(
        self,
        video_path: Path,
        *,
        video_id: str | None = None,
        emit_event: EventCallback | None = None,
        simulate_delay_sec: float = 0.03,
    ) -> dict:
        """Read a video as sampled frame events and return semantic signals.

        这里故意保留 `emit_event` 和 `simulate_delay_sec`：前者让网站能看到流式进度，
        后者让同学更容易观察“上传立即可见、后台慢慢补理解结果”的异步效果。
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")

        metadata = self._metadata(capture)
        frame_signals: list[FrameSignal] = []
        previous_gray: np.ndarray | None = None
        previous_brightness: float | None = None
        sample_interval = max(1, int(round(max(metadata["fps"], 1.0) / SAMPLE_FPS)))
        frame_index = 0
        sampled_index = 0

        while sampled_index < MAX_SAMPLED_FRAMES:
            # OpenCV 顺序读取视频帧；达到抽样间隔时才进入特征计算，降低 CPU 压力。
            ok, bgr_frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % sample_interval != 1:
                continue

            frame = self._prepare_frame(bgr_frame)
            sampled_index += 1
            gray = _brightness(frame)
            brightness_value = float(np.mean(gray))
            colorfulness_value = _colorfulness(frame)
            red_ratio, green_ratio, blue_ratio = _channel_ratios(frame)

            motion = 0.0
            if previous_gray is not None:
                # motion 使用相邻采样帧亮度差的平均值，简单但便于教学解释。
                motion = float(np.mean(np.abs(gray - previous_gray)))

            flash_delta = 0.0
            if previous_brightness is not None:
                # flash_delta 用于发现强亮度跳变，模拟闪烁风险检测。
                flash_delta = abs(brightness_value - previous_brightness)

            frame_signals.append(
                FrameSignal(
                    brightness=brightness_value,
                    colorfulness=colorfulness_value,
                    motion=motion,
                    red_ratio=red_ratio,
                    green_ratio=green_ratio,
                    blue_ratio=blue_ratio,
                    flash_delta=flash_delta,
                )
            )
            previous_gray = gray
            previous_brightness = brightness_value

            if emit_event and (sampled_index == 1 or sampled_index % 5 == 0):
                # 不对每一帧都写事件，避免事件列表被低价值日志淹没。
                emit_event(
                    video_id,
                    "frame_sample",
                    f"已抽样分析第 {sampled_index} 帧",
                    {
                        "frame_index": sampled_index,
                        "source_frame_index": frame_index,
                        "brightness": round(brightness_value, 2),
                        "motion": round(motion, 2),
                        "colorfulness": round(colorfulness_value, 2),
                    },
                )
            if simulate_delay_sec > 0:
                time.sleep(simulate_delay_sec)

        capture.release()
        if not frame_signals:
            raise ValueError("no frames could be sampled from the video")

        metrics = self._metrics(metadata, frame_signals)
        tags = self._tags(metadata, metrics)
        caption = self._caption(metadata, metrics)
        return {
            "metadata": metadata,
            "metrics": metrics,
            "tags": tags,
            "caption": caption,
        }

    def _metadata(self, capture: cv2.VideoCapture) -> dict:
        """Read basic video metadata from an opened OpenCV capture."""
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration_sec": duration,
            "frame_count": frame_count,
        }

    def _prepare_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Resize one BGR frame and convert it to RGB for feature extraction."""
        height, width = bgr_frame.shape[:2]
        target_height = max(2, int(round(height * ANALYSIS_WIDTH / max(1, width))))
        resized = cv2.resize(bgr_frame, (ANALYSIS_WIDTH, target_height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _metrics(self, metadata: dict, signals: list[FrameSignal]) -> dict:
        """Aggregate frame-level signals into video-level metrics."""
        brightness_values = [item.brightness for item in signals]
        color_values = [item.colorfulness for item in signals]
        motion_values = [item.motion for item in signals[1:]]
        flash_values = [item.flash_delta for item in signals[1:]]
        red_values = [item.red_ratio for item in signals]
        green_values = [item.green_ratio for item in signals]
        blue_values = [item.blue_ratio for item in signals]
        flash_count = sum(1 for value in flash_values if value >= 55.0)

        return {
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": round(metadata["fps"], 2),
            "duration_sec": round(metadata["duration_sec"], 2),
            "frame_count": metadata["frame_count"],
            "sampled_frames": len(signals),
            "brightness": _summarize(brightness_values),
            "colorfulness": _summarize(color_values),
            "motion": _summarize(motion_values),
            "red_ratio_avg": round(float(np.mean(red_values)), 4),
            "green_ratio_avg": round(float(np.mean(green_values)), 4),
            "blue_ratio_avg": round(float(np.mean(blue_values)), 4),
            "flash_count": flash_count,
            "flash_ratio": round(flash_count / max(1, len(flash_values)), 4),
        }

    def _tags(self, metadata: dict, metrics: dict) -> list[str]:
        """Convert numeric metrics into human-readable tags shown on the website."""
        tags = ["短视频"]
        tags.append("竖屏" if metadata["height"] >= metadata["width"] else "横屏")

        brightness = metrics["brightness"]["avg"]
        colorfulness = metrics["colorfulness"]["avg"]
        motion = metrics["motion"]["avg"]
        green_blue = metrics["green_ratio_avg"] + metrics["blue_ratio_avg"]

        if brightness >= 135:
            tags.append("明亮")
        elif brightness <= 55:
            tags.append("低照度")
        else:
            tags.append("自然光")

        if motion >= 18:
            tags.append("运动明显")
        elif motion <= 5:
            tags.append("节奏平稳")
        else:
            tags.append("轻运动")

        if colorfulness >= 55:
            tags.append("高饱和")
        if green_blue >= 0.28:
            tags.append("户外感")
        if metrics["flash_ratio"] >= 0.15:
            tags.append("闪烁画面")
        return tags

    def _caption(self, metadata: dict, metrics: dict) -> str:
        """Generate a short baseline caption from explainable metrics."""
        orientation = "竖屏" if metadata["height"] >= metadata["width"] else "横屏"
        brightness = metrics["brightness"]["avg"]
        motion = metrics["motion"]["avg"]
        tone = "明亮" if brightness >= 135 else "低照度" if brightness <= 55 else "自然光"
        pace = "运动变化明显" if motion >= 18 else "节奏较平稳" if motion <= 5 else "存在轻微运动"

        dominant = "综合色彩均衡"
        color_ratios = {
            "绿色/自然色": metrics["green_ratio_avg"],
            "蓝色/天空水面色": metrics["blue_ratio_avg"],
            "红色高占比": metrics["red_ratio_avg"],
        }
        color_name, color_value = max(color_ratios.items(), key=lambda item: item[1])
        if color_value >= 0.18:
            dominant = f"画面以{color_name}为主"

        return f"一段{orientation}{tone}短视频，{pace}，{dominant}。"


def moderate_analysis(analysis: dict, title: str) -> dict:
    """Apply deterministic moderation rules to the analysis result.

    审核策略采用“规则信号 + VLM 风险建议”的合并方式：规则负责稳定、可解释，
    VLM 负责识别更丰富的语义风险。最终状态只有 published、review、rejected 三类。
    """
    metrics = analysis["metrics"]
    score = 0.0
    reasons: list[dict] = []
    model_risk = analysis.get("model_risk") or {}

    lowered_title = title.lower()
    matched_words = [
        word for word in BANNED_TITLE_WORDS if word.lower() in lowered_title or word in title
    ]
    if matched_words:
        # 标题命中高风险词直接强烈加分，因为标题是用户主动输入的发布信号。
        score += 80
        reasons.append(
            {
                "code": "title_policy_keyword",
                "level": "reject",
                "message": "标题包含实验策略中的高风险词，需要拒绝发布。",
                "evidence": matched_words,
            }
        )

    duration = metrics["duration_sec"]
    if duration <= 0:
        # 无效时长代表媒体不可理解，按 fail-closed 思路拒绝发布。
        score += 100
        reasons.append(
            {
                "code": "invalid_duration",
                "level": "reject",
                "message": "无法识别有效视频时长。",
                "evidence": duration,
            }
        )
    elif duration > 60:
        score += 45
        reasons.append(
            {
                "code": "duration_too_long",
                "level": "review",
                "message": "实验平台只接受 60 秒以内的短视频。",
                "evidence": duration,
            }
        )

    brightness = metrics["brightness"]["avg"]
    if brightness < 25:
        # 极暗画面不一定违规，但自动理解置信度不足，因此进入复核。
        score += 42
        reasons.append(
            {
                "code": "too_dark",
                "level": "review",
                "message": "画面过暗，自动理解置信度下降，需要人工复核。",
                "evidence": brightness,
            }
        )
    elif brightness < 45:
        score += 20
        reasons.append(
            {
                "code": "low_light",
                "level": "notice",
                "message": "画面亮度偏低，发布前建议复核封面和内容。",
                "evidence": brightness,
            }
        )

    if metrics["flash_ratio"] >= 0.15:
        # 强闪烁可能造成观看不适，本实验把它作为人工复核信号。
        score += 38
        reasons.append(
            {
                "code": "flash_risk",
                "level": "review",
                "message": "检测到多次强亮度跳变，可能造成观看不适。",
                "evidence": metrics["flash_ratio"],
            }
        )

    if metrics["red_ratio_avg"] >= 0.42:
        reasons.append(
            {
                "code": "red_dominance",
                "level": "notice",
                "message": "红色高占比不等同于违规，但在真实审核中应进入更强模型复核。",
                "evidence": metrics["red_ratio_avg"],
            }
        )

    if metrics["motion"]["avg"] >= 35:
        score += 10
        reasons.append(
            {
                "code": "high_motion",
                "level": "notice",
                "message": "画面运动幅度较高，建议结合更高采样率确认内容。",
                "evidence": metrics["motion"]["avg"],
            }
        )

    risk_level = model_risk.get("level", "pass")
    model_score = float(model_risk.get("score") or 0)
    if risk_level == "reject":
        # VLM 明确拒绝时使用较高风险分，避免规则分过低导致放行。
        score = max(score, max(75.0, model_score))
        reasons.append(
            {
                "code": "vlm_reject",
                "level": "reject",
                "message": "多模态模型判定存在拒绝发布风险。",
                "evidence": {
                    "categories": model_risk.get("categories", []),
                    "evidence": model_risk.get("evidence", []),
                    "score": model_score,
                },
            }
        )
    elif risk_level == "review":
        # VLM 建议复核时至少提升到 review 阈值附近。
        score = max(score, max(40.0, model_score))
        reasons.append(
            {
                "code": "vlm_review",
                "level": "review",
                "message": "多模态模型建议进入人工复核。",
                "evidence": {
                    "categories": model_risk.get("categories", []),
                    "evidence": model_risk.get("evidence", []),
                    "score": model_score,
                },
            }
        )

    status = "published"
    if any(reason["level"] == "reject" for reason in reasons) or score >= 70:
        status = "rejected"
    elif score >= 35:
        status = "review"

    if not reasons:
        reasons.append(
            {
                "code": "policy_passed",
                "level": "pass",
                "message": "未命中本实验策略中的高风险信号，可自动发布。",
                "evidence": round(score, 2),
            }
        )

    return {
        "status": status,
        "risk_score": round(min(score, 100.0), 2),
        "reasons": reasons,
    }
