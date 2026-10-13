"""链路、机载计算机、感知与视觉伺服，纯 numpy。"""
import numpy as np

import spec as S


class SimpleLink:
    """定延时加丢包的链路，RC、遥测、机间通信共用。"""

    def __init__(self, rate, latency, rng, drop=0.0):
        self.dt = 1.0 / rate
        self.latency = latency
        self.rng = rng
        self.drop = drop
        self._q = []
        self._t = -1.0
        self.last = None

    def send(self, t, msg):
        if self.drop > 0 and self.rng.random() < self.drop:
            return
        self._q.append((t + self.latency, msg))

    def recv(self, t):
        while self._q and self._q[0][0] <= t:
            self.last = self._q.pop(0)[1]
        return self.last


class VideoLink:
    """图传，定延时与丢帧。"""

    def __init__(self, rng, latency=S.VIDEO_LATENCY, rate=S.VIDEO_RATE, drop=S.VIDEO_DROP):
        self.latency = latency
        self.dt = 1.0 / rate
        self.rng = rng
        self.drop = drop
        self._frames = []
        self._next = 0.0

    def push(self, t, frame):
        if self.rng.random() < self.drop:
            return
        self._frames.append((t + self.latency, frame))
        if len(self._frames) > 200:
            self._frames = self._frames[-100:]

    def latest(self, t):
        out = None
        while self._frames and self._frames[0][0] <= t:
            out = self._frames.pop(0)[1]
        return out


class OnboardComputer:
    """机载计算时序，感知周期任务与 VLM 一次性查询。"""

    def __init__(self, rng):
        self.rng = rng
        self._t_perc = -1.0
        self.last_detection = None
        self.vlm_busy_until = -1.0
        self.vlm_result = None

    def perception_tick(self, t, image, detect_fn):
        if image is None or t - self._t_perc < 1.0 / S.OBC_PERCEPTION_HZ:
            return self.last_detection
        self._t_perc = t
        self.last_detection = detect_fn(image)
        return self.last_detection

    def vlm_query(self, t, instruction_parser, text):
        self.vlm_busy_until = t + S.VLM_LATENCY
        self.vlm_result = instruction_parser(text)
        return self.vlm_result

    def vlm_ready(self, t):
        return t >= self.vlm_busy_until


def parse_instruction(text):
    """VLM 接口层，文本到目标描述，仿真后端用颜色几何先验。"""
    t = text.lower()
    spec = {"color": "orange", "shape": "cylinder", "name": "试件圆柱"}
    if "红" in t or "red" in t:
        spec["color"] = "red"
    if "蓝" in t or "blue" in t:
        spec["color"] = "blue"
    if "球" in t or "ball" in t:
        spec["shape"] = "sphere"
    return spec


COLOR_RANGES = {
    "orange": ((0.45, 0.15, 0.0), (1.0, 0.75, 0.45)),
    "red": ((0.35, 0.0, 0.0), (1.0, 0.45, 0.45)),
    "blue": ((0.0, 0.0, 0.20), (0.45, 0.55, 1.0)),
}


def detect_color_target(rgb, spec):
    """颜色阈值加几何中心检测，返回像素中心与面积占比。"""
    lo, hi = COLOR_RANGES.get(spec.get("color", "orange"), COLOR_RANGES["orange"])
    img = np.asarray(rgb, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    mask = np.all((img >= np.array(lo)) & (img <= np.array(hi)), axis=-1)
    area = float(mask.mean())
    if area < 0.0005:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return {"u": float(xs.mean()) / w, "v": float(ys.mean()) / h, "area": area}


class VisualServo:
    """图像误差到机体速度指令，深度误差给垂向，带死区与前馈。"""

    def __init__(self):
        self.err_filt = np.zeros(3)

    def reset(self):
        self.err_filt[:] = 0.0

    def compute(self, detection, depth_m_target, drone_z, target_z, dt, forward=1.0):
        if detection is None:
            return np.zeros(3), False
        e_u = detection["u"] - 0.5
        e_v = detection["v"] - 0.5
        e_z = (drone_z - target_z) - depth_m_target
        self.err_filt = self.err_filt + (np.array([e_u, e_v, 0.0]) - self.err_filt) * min(dt * 6.0, 1.0)
        vx = -S.SERVO_KP_XY * self.err_filt[1] * forward
        vy = S.SERVO_KP_XY * self.err_filt[0]
        vz = -S.SERVO_KP_Z * e_z
        v = np.array([vx, vy, vz])
        v = np.clip(v, -S.SERVO_VMAX, S.SERVO_VMAX)
        aligned = abs(e_u) < 0.04 and abs(e_v) < 0.04 and abs(e_z) < S.SERVO_DEAD_Z
        return v, aligned
