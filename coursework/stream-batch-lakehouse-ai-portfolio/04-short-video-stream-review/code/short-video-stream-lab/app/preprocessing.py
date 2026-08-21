"""Video preprocessing for multimodal understanding.

多模态模型通常不能直接吃完整视频文件，尤其是在 16GB 学生电脑上。
本模块把视频转换为一组代表性关键帧、基础 metadata 和可选音频轨，
让后续 Ollama VLM 只处理小而有信息量的输入。
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import (
    AUDIO_DIR,
    FRAME_DIR,
    MAX_VLM_KEYFRAMES,
    MOTION_PEAK_THRESHOLD,
    SCENE_CHANGE_THRESHOLD,
    VLM_FRAME_WIDTH,
)
from .ffmpeg_tools import extract_audio_track

EventCallback = Callable[[str | None, str, str, dict | None], None]


@dataclass
class KeyFrame:
    """Metadata for one frame selected as a VLM input image."""

    frame_index: int
    timestamp_sec: float
    file: str
    reason: str
    brightness: float
    motion: float
    scene_change: float

    def to_dict(self) -> dict:
        """Serialize the dataclass for JSON storage and model prompts."""
        return asdict(self)


def _resize_for_vlm(frame: np.ndarray) -> np.ndarray:
    """Resize a BGR OpenCV frame to the width expected by local VLM prompts."""
    height, width = frame.shape[:2]
    target_height = max(2, int(round(height * VLM_FRAME_WIDTH / max(1, width))))
    return cv2.resize(frame, (VLM_FRAME_WIDTH, target_height), interpolation=cv2.INTER_AREA)


def _brightness(gray: np.ndarray) -> float:
    """Return average grayscale brightness as a simple explainable signal."""
    return float(np.mean(gray))


class VideoPreprocessor:
    """Extract industrial-style artifacts before calling a VLM.

    核心策略是“均匀抽样 + 场景切换 + 运动峰值”：既覆盖整个时间线，
    又尽量保留短视频中最可能包含语义变化的帧。
    """

    def prepare(
        self,
        video_path: Path,
        *,
        video_id: str,
        emit_event: EventCallback | None = None,
    ) -> dict:
        """Create keyframes, optional audio, and metadata for one video.

        返回值会同时供 Ollama prompt、报告事件和最终 metrics 使用。
        """
        frame_dir = FRAME_DIR / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        audio_path = AUDIO_DIR / f"{video_id}.wav"

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = frame_count / fps if fps else 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # 先保证一部分帧来自均匀时间采样，避免只选到运动峰值而丢失视频上下文。
        uniform_interval = max(1, int(frame_count / max(1, MAX_VLM_KEYFRAMES // 2)))
        candidates: list[dict] = []
        previous_gray: np.ndarray | None = None
        frame_index = -1

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            small = cv2.resize(frame, (160, max(2, int(height * 160 / max(1, width)))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            brightness = _brightness(gray)
            motion = 0.0
            scene_change = 0.0
            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                motion = float(np.mean(diff))
                scene_change = float(np.percentile(diff, 95))

            # reason 记录“为什么选择这帧”，报告和 timeline 会展示这个可解释证据。
            reason = ""
            if frame_index == 0:
                reason = "first_frame"
            elif frame_index % uniform_interval == 0:
                reason = "uniform_sample"
            elif scene_change >= SCENE_CHANGE_THRESHOLD:
                reason = "scene_cut"
            elif motion >= MOTION_PEAK_THRESHOLD:
                reason = "motion_peak"

            if reason:
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": round(frame_index / fps, 2) if fps else 0.0,
                        "frame": frame.copy(),
                        "reason": reason,
                        "brightness": round(brightness, 2),
                        "motion": round(motion, 2),
                        "scene_change": round(scene_change, 2),
                    }
                )
            previous_gray = gray

        capture.release()
        selected = self._select_keyframes(candidates)
        keyframes: list[KeyFrame] = []
        for order, item in enumerate(selected, start=1):
            # 落盘为 JPEG 是为了能直接传给 Ollama，也方便学生在 data/media/frames 下查看。
            output_path = frame_dir / f"keyframe-{order:02d}-{item['frame_index']}.jpg"
            resized = _resize_for_vlm(item["frame"])
            ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise ValueError(f"cannot encode keyframe: {output_path}")
            output_path.write_bytes(encoded.tobytes())
            keyframes.append(
                KeyFrame(
                    frame_index=item["frame_index"],
                    timestamp_sec=item["timestamp_sec"],
                    file=str(output_path),
                    reason=item["reason"],
                    brightness=item["brightness"],
                    motion=item["motion"],
                    scene_change=item["scene_change"],
                )
            )

        has_audio = extract_audio_track(video_path, audio_path)
        if emit_event:
            emit_event(
                video_id,
                "preprocess",
                "完成关键帧、场景切分和音频轨预处理",
                {
                    "keyframes": len(keyframes),
                    "has_audio": has_audio,
                    "duration_sec": round(duration_sec, 2),
                },
            )

        return {
            "metadata": {
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "duration_sec": round(duration_sec, 2),
                "frame_count": frame_count,
            },
            "keyframes": [frame.to_dict() for frame in keyframes],
            "audio_path": str(audio_path) if has_audio else "",
            "sampling_strategy": {
                "max_keyframes": MAX_VLM_KEYFRAMES,
                "scene_change_threshold": SCENE_CHANGE_THRESHOLD,
                "motion_peak_threshold": MOTION_PEAK_THRESHOLD,
            },
        }

    def _select_keyframes(self, candidates: list[dict]) -> list[dict]:
        """Choose a bounded, chronological set of frames from all candidates.

        超过上限时优先保留首帧、明显场景切换、运动峰值和均匀样本。
        最后按 frame_index 排序，是为了让 prompt 中的时间线和视频顺序一致。
        """
        if len(candidates) <= MAX_VLM_KEYFRAMES:
            return candidates

        first = candidates[:1]
        scene = sorted(
            [item for item in candidates if item["reason"] == "scene_cut"],
            key=lambda item: item["scene_change"],
            reverse=True,
        )
        motion = sorted(
            [item for item in candidates if item["reason"] == "motion_peak"],
            key=lambda item: item["motion"],
            reverse=True,
        )
        uniform = [item for item in candidates if item["reason"] == "uniform_sample"]

        selected: list[dict] = []
        seen: set[int] = set()
        for bucket in (first, scene, motion, uniform, candidates):
            for item in bucket:
                if item["frame_index"] in seen:
                    continue
                selected.append(item)
                seen.add(item["frame_index"])
                if len(selected) >= MAX_VLM_KEYFRAMES:
                    return sorted(selected, key=lambda frame: frame["frame_index"])
        return sorted(selected, key=lambda frame: frame["frame_index"])
