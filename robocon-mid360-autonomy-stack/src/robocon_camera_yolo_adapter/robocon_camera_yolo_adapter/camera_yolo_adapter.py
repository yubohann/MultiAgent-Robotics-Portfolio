"""Bridge the existing YOLO/depth detector into the team target-observation contract."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(frozen=True)
class TargetObservation:
    confidence: float
    distance_m: float
    stable: bool
    observed_at_ns: int
    target_type: str
    center_x: int | None = None
    center_y: int | None = None
    bbox: tuple[int, int, int, int] | None = None


def _read_value(candidate: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def normalize_legacy_candidate(
    candidate: Mapping[str, Any] | object,
    *,
    now_ns: int,
    target_classes: Iterable[str] = ("basket", "hoop", "backboard"),
) -> TargetObservation | None:
    """Normalize one result from the legacy detector without importing its SDK."""

    target_type = str(_read_value(candidate, "class", _read_value(candidate, "target_type", "unknown")))
    allowed = {str(item).strip().lower() for item in target_classes if str(item).strip()}
    if target_type.lower() == "person" or (allowed and target_type.lower() not in allowed):
        return None

    confidence = float(_read_value(candidate, "confidence", 0.0))
    distance_mm = _read_value(candidate, "distance_mm", _read_value(candidate, "distance", 0.0))
    distance_m = float(distance_mm or 0.0) / 1000.0
    stable = bool(_read_value(candidate, "stable", _read_value(candidate, "is_stable", False)))
    timestamp = _read_value(candidate, "observed_at_ns")
    if timestamp is None:
        timestamp_seconds = _read_value(candidate, "timestamp")
        timestamp = now_ns if timestamp_seconds is None else int(float(timestamp_seconds) * 1_000_000_000)
    observed_at_ns = int(timestamp)
    if not math.isfinite(confidence) or not math.isfinite(distance_m):
        raise ValueError("legacy target confidence and distance must be finite")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("legacy target confidence must be within [0, 1]")
    if distance_m < 0.0 or observed_at_ns <= 0:
        raise ValueError("legacy target distance and timestamp are invalid")

    center = _read_value(candidate, "position")
    center_x = center_y = None
    if center is not None and len(center) >= 2:
        center_x, center_y = int(center[0]), int(center[1])
    raw_bbox = _read_value(candidate, "bbox")
    bbox = None
    if raw_bbox is not None and len(raw_bbox) >= 4:
        bbox = tuple(int(value) for value in raw_bbox[:4])
    return TargetObservation(
        confidence=confidence,
        distance_m=distance_m,
        stable=stable,
        observed_at_ns=observed_at_ns,
        target_type=target_type,
        center_x=center_x,
        center_y=center_y,
        bbox=bbox,
    )


def select_best_legacy_candidate(
    candidates: Iterable[Mapping[str, Any] | object],
    *,
    now_ns: int,
    target_classes: Iterable[str] = ("basket", "hoop", "backboard"),
) -> TargetObservation | None:
    normalized = []
    for candidate in candidates:
        try:
            observation = normalize_legacy_candidate(
                candidate, now_ns=now_ns, target_classes=target_classes
            )
        except (TypeError, ValueError, KeyError):
            continue
        if observation is not None:
            normalized.append(observation)
    return max(normalized, key=lambda item: (item.stable, item.confidence)) if normalized else None


def detect_frame(detector: Any, frame: Any, *, use_berxel: bool, now_ns: int) -> list[dict[str, Any]]:
    """Reuse the legacy detector's preprocessing and model path for one frame."""

    from utils.general import non_max_suppression, scale_boxes

    image = detector.preprocess(frame)
    prediction = detector.model(image)
    prediction = non_max_suppression(
        prediction, detector.conf_thres, detector.iou_thres, max_det=1000
    )
    detections: list[dict[str, Any]] = []
    for batch_detection in prediction:
        if not len(batch_detection):
            continue
        batch_detection[:, :4] = scale_boxes(image.shape[2:], batch_detection[:, :4], frame.shape).round()
        for *xyxy, confidence, class_id in reversed(batch_detection):
            confidence_value = float(confidence)
            class_name = str(detector.names[int(class_id)])
            x1, y1, x2, y2 = (int(value) for value in xyxy)
            distance = None
            stable = False
            if use_berxel:
                distance = detector.depth_camera.get_min_distance_in_bbox(x1, y1, x2, y2)
                stable = bool(detector.depth_camera.is_data_stable())
            detections.append({
                "class": class_name,
                "confidence": confidence_value,
                "position": ((x1 + x2) // 2, (y1 + y2) // 2),
                "distance": distance,
                "bbox": (x1, y1, x2, y2),
                "stable": stable,
                "observed_at_ns": now_ns,
            })
    return detections


def observation_to_json(observation: TargetObservation, *, evidence_level: str) -> str:
    payload: dict[str, Any] = {
        "confidence": observation.confidence,
        "distance_m": observation.distance_m,
        "stable": observation.stable,
        "observed_at_ns": observation.observed_at_ns,
        "target_type": observation.target_type,
        "evidence_level": evidence_level,
    }
    if observation.center_x is not None and observation.center_y is not None:
        payload["center_px"] = [observation.center_x, observation.center_y]
    if observation.bbox is not None:
        payload["bbox"] = list(observation.bbox)
    return json.dumps(payload, separators=(",", ":"))


class CameraYoloAdapter(Node):
    """Run the legacy detector only when explicitly enabled and configured."""

    def __init__(self) -> None:
        super().__init__("robocon_camera_yolo_adapter")
        self.declare_parameter("enabled", False)
        self.declare_parameter("legacy_module_path", "TBD")
        self.declare_parameter("weights_path", "TBD")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("use_berxel", True)
        self.declare_parameter("conf_thres", 0.60)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("target_classes", ["basket", "hoop", "backboard"])
        self.declare_parameter("observation_topic", "/camera/target_observation")
        self.declare_parameter("status_topic", "/camera/target_status")
        self.declare_parameter("diagnostic_topic", "/camera/diagnostics")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("observation_max_age_sec", 0.50)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.legacy_module_path = str(self.get_parameter("legacy_module_path").value)
        self.weights_path = str(self.get_parameter("weights_path").value)
        self.device = str(self.get_parameter("device").value)
        self.use_berxel = bool(self.get_parameter("use_berxel").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.target_classes = [str(item) for item in self.get_parameter("target_classes").value]
        self.max_age_sec = float(self.get_parameter("observation_max_age_sec").value)
        publish_hz = float(self.get_parameter("publish_hz").value)
        if publish_hz <= 0.0 or self.max_age_sec <= 0.0:
            raise ValueError("publish_hz and observation_max_age_sec must be positive")

        self._observation_pub = self.create_publisher(
            String, str(self.get_parameter("observation_topic").value), 10
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self._diagnostic_pub = self.create_publisher(
            DiagnosticArray, str(self.get_parameter("diagnostic_topic").value), 10
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest: TargetObservation | None = None
        self._status = "disabled" if not self.enabled else "starting"
        self._failure_reason = "disabled_by_configuration" if not self.enabled else ""
        self._worker: threading.Thread | None = None
        if self.enabled:
            self._worker = threading.Thread(target=self._capture_worker, name="camera-yolo-capture", daemon=True)
            self._worker.start()
        self.create_timer(1.0 / publish_hz, self._publish_status)

    def _load_detector(self) -> Any:
        module_path = Path(self.legacy_module_path).expanduser()
        weights_path = Path(self.weights_path).expanduser()
        if not module_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError("legacy_module_path and weights_path must point to existing files")
        sys.path.insert(0, str(module_path.parent))
        spec = importlib.util.spec_from_file_location("robocon_legacy_yolo_depth", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load legacy detector module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        detector_class = getattr(module, "YOLODepthDetector")
        return detector_class(
            weights=str(weights_path),
            device=self.device,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )

    def _capture_worker(self) -> None:
        detector = None
        capture = None
        try:
            detector = self._load_detector()
            if self.use_berxel:
                if not detector.depth_camera.start_camera():
                    raise RuntimeError("legacy Berxel depth camera failed to start")
            else:
                import cv2

                capture = cv2.VideoCapture(0)
                if not capture.isOpened():
                    raise RuntimeError("fallback camera could not be opened")
            with self._lock:
                self._status = "running"
                self._failure_reason = ""
            while not self._stop_event.is_set():
                if self.use_berxel:
                    frame = detector.depth_camera.get_color_frame()
                    if frame is not None:
                        detector.depth_camera.load_depth_data()
                else:
                    received, frame = capture.read()
                    if not received:
                        frame = None
                if frame is None:
                    time.sleep(0.02)
                    continue
                detections = detect_frame(
                    detector, frame, use_berxel=self.use_berxel, now_ns=time.time_ns()
                )
                observation = select_best_legacy_candidate(
                    detections, now_ns=time.time_ns(), target_classes=self.target_classes
                )
                with self._lock:
                    self._latest = observation
                    self._status = "running_with_target" if observation else "running_no_target"
                time.sleep(0.01)
        except Exception as error:
            with self._lock:
                self._status = "error"
                self._failure_reason = str(error)
        finally:
            if detector is not None and self.use_berxel:
                try:
                    detector.depth_camera.stop_camera()
                except Exception:
                    pass
            if capture is not None:
                capture.release()

    def _publish_status(self) -> None:
        now_ns = time.time_ns()
        with self._lock:
            observation = self._latest
            status = self._status
            failure_reason = self._failure_reason
        if observation is not None and (now_ns - observation.observed_at_ns) <= int(self.max_age_sec * 1e9):
            self._observation_pub.publish(
                String(data=observation_to_json(observation, evidence_level="camera_yolo_runtime"))
            )
        age_sec = None if observation is None else max(0.0, (now_ns - observation.observed_at_ns) / 1e9)
        payload = {
            "version": 1,
            "enabled": self.enabled,
            "status": status,
            "reason": failure_reason,
            "observation_age_sec": age_sec,
            "evidence_level": "camera_yolo_runtime" if observation else "unknown",
        }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        diagnostic_status = DiagnosticStatus()
        diagnostic_status.name = f"{self.get_name()}/camera"
        diagnostic_status.level = DiagnosticStatus.OK if status == "running_with_target" else DiagnosticStatus.WARN
        diagnostic_status.message = status
        diagnostic_status.values = [
            KeyValue(key="enabled", value=str(self.enabled).lower()),
            KeyValue(key="status", value=status),
            KeyValue(key="reason", value=failure_reason),
            KeyValue(key="observation_age_sec", value=str(age_sec)),
        ]
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        diagnostic.status = [diagnostic_status]
        self._diagnostic_pub.publish(diagnostic)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraYoloAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
