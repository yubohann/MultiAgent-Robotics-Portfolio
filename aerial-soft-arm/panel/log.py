"""日志线程，采样写 npz 与 JSON 元数据，含事件流，另提供离线回放。"""
import json
import math
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from drivers import DRONE_HOME, Driver, variant_weights
from keys import LABEL, MODE_LABEL, SEMANTICS

MODE_INT = {"combo": 0, "precise": 1, "drone": 2}


class LoggerThread(threading.Thread):
    def __init__(self, log_q, stop_evt, out_dir: Path, meta: dict, keys=None):
        super().__init__(name="log", daemon=True)
        self.q = log_q
        self.stop = stop_evt
        self.out = Path(out_dir)
        self.meta = meta
        self.keys = keys
        self.rows = []

    def run(self):
        while True:
            try:
                snap = self.q.get(timeout=0.2)
            except queue.Empty:
                if self.stop.is_set():
                    break
                continue
            self.rows.append(snap)
        self.save()

    def save(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        n = len(self.rows)
        t0 = self.rows[0]["t"]
        arr = {
            "t": np.array([r["t"] - t0 for r in self.rows]),
            "qpos": np.stack([r["qpos"] for r in self.rows]),
            "drone": np.stack([np.concatenate([r["drone_pos"], [r["drone_yaw"]]]) for r in self.rows]),
            "P": np.array([r["P"] for r in self.rows]),
            "P_meas": np.array([r["P_meas"] for r in self.rows]),
            "p_ref": np.array([r["p_ref"] for r in self.rows]),
            "pwm": np.array([r["pwm"] for r in self.rows]),
            "valve": np.array([r["valve"] for r in self.rows]),
            "grip": np.array([r["grip"] for r in self.rows]),
            "hold": np.array([r["hold"] for r in self.rows], dtype=np.uint8),
            "contacts": np.array([r["contacts"] for r in self.rows], dtype=np.int32),
            "wrist": np.stack([r["wrist"] for r in self.rows]),
            "locks": np.stack([r["locks"].astype(np.uint8) for r in self.rows]),
            "mode": np.array([MODE_INT[r["mode"]] for r in self.rows], dtype=np.uint8),
            "estop": np.array([r["estop"] for r in self.rows], dtype=np.uint8),
            "n": np.array([n]),
        }
        np.savez_compressed(self.out / "log.npz", **arr)
        if self.keys is not None:
            ev = [[round(e[0] - t0, 4), e[1], e[2]] for e in self.keys.event_log]
            (self.out / "events.json").write_text(
                json.dumps({"events": ev}, ensure_ascii=False), encoding="utf-8")
        meta = dict(self.meta)
        meta["frames"] = n
        meta["duration"] = round(float(arr["t"][-1]), 3)
        meta["saved"] = datetime.now().isoformat(timespec="seconds")
        (self.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[log] %d 帧写入 %s" % (n, self.out))


class EventPlayer:
    """按事件流重建每一帧的键态，供回放叠加。"""

    def __init__(self, events):
        self.events = sorted(events, key=lambda e: e[0])
        self.idx = 0
        self.down_t = {}
        self.latched = set()
        self.flash = {}
        self.mode = "combo"
        self.estop = False

    def state_at(self, t):
        while self.idx < len(self.events) and self.events[self.idx][0] <= t:
            te, phase, bid = self.events[self.idx]
            self.idx += 1
            if bid == "estop":
                if phase == "d":
                    self.estop = not self.estop
                continue
            if bid.startswith("mode_"):
                if phase == "d":
                    self.mode = bid[5:]
                continue
            if phase == "d":
                sem = SEMANTICS.get(bid)
                if sem == "latch" and bid in self.latched:
                    self.latched.discard(bid)
                    self.down_t.pop(bid, None)
                    continue
                self.down_t[bid] = te
                if sem == "click":
                    self.flash[bid] = te + 0.5
            else:
                t0 = self.down_t.pop(bid, None)
                if t0 is None:
                    continue
                if SEMANTICS.get(bid) == "latch" and te - t0 < 0.25:
                    self.latched.add(bid)
                elif SEMANTICS.get(bid) == "latch" and bid in self.latched:
                    self.latched.discard(bid)
        self.flash = {b: ex for b, ex in self.flash.items() if ex > t}
        return {
            "hold": set(self.down_t),
            "latch": set(self.latched),
            "flash": set(self.flash),
            "mode": self.mode,
            "estop": self.estop,
        }


ARM_ROWS = (("w", "a", "s", "d", "q"), ("e", "r", "f", "z", "c"),
            ("1", "2", "3", "4", "5"), ("6", "x", "g", "v", "b"))
DRONE_ROWS = (("up", "down", "left", "right"), ("k8", "k2", "k4", "k6"), ("k5", "k0"))
GLOBAL_IDS = ("estop", "mode_combo", "mode_precise", "mode_drone", "exit")
INT_MODE = {v: k for k, v in MODE_INT.items()}


def draw_keyboard(dr, st, ox, oy):
    """屏幕键盘，全部按键带标签，按下的高亮，其余低亮度但可读。"""
    cw, ch, gap = 34, 19, 3
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)

    def cell(x, y, bid, w=cw):
        hold = bid in st["hold"]
        latch = bid in st["latch"]
        flash = bid in st["flash"]
        active = hold or latch or flash
        if bid == "estop":
            fill = (214, 40, 48, 245) if st["estop"] else (96, 20, 24, 225)
            txt = (245, 246, 248, 255)
        elif bid.startswith("mode_"):
            on = st["mode"] == bid[5:]
            fill = (52, 78, 116, 240) if on else (18, 22, 30, 165)
            txt = (236, 240, 245, 255) if on else (150, 156, 166, 220)
        elif hold or flash:
            fill = (58, 108, 186, 245)
            txt = (245, 247, 250, 255)
        elif latch:
            fill = (198, 132, 34, 245)
            txt = (245, 247, 250, 255)
        else:
            fill = (18, 22, 30, 165)
            txt = (150, 156, 166, 220)
        dr.rounded_rectangle((x, y, x + w, y + ch), radius=4, fill=fill,
                             outline=(255, 255, 255, 30 if not active else 60), width=1)
        dr.text((x + 5, y + 3), LABEL[bid], font=font, fill=txt)

    for r, row in enumerate(ARM_ROWS):
        for c, bid in enumerate(row):
            cell(ox + c * (cw + gap), oy + r * (ch + gap), bid)
    dx = ox + 5 * (cw + gap) + 14
    for r, row in enumerate(DRONE_ROWS):
        for c, bid in enumerate(row):
            cell(dx + c * (cw + gap), oy + r * (ch + gap), bid)
    gy = oy + 4 * (ch + gap) + 6
    for c, bid in enumerate(GLOBAL_IDS):
        cell(ox + c * (52 + gap), gy, bid, w=52)


def replay(npz_path, out_mp4):
    """按日志回放位姿并叠加屏幕键盘与腔压曲线，离线渲染成视频。"""
    data = np.load(npz_path)
    n = int(data["n"][0])
    qpos = data["qpos"]
    drone = data["drone"]
    P = data["P"]
    ts = data["t"] if "t" in data.files else np.arange(n) * 0.05
    p_ref = data["p_ref"]
    pwm = data["pwm"]
    valve = data["valve"]
    npz = Path(npz_path)
    ev_path = npz.with_name("events.json")
    events = json.loads(ev_path.read_text(encoding="utf-8"))["events"] if ev_path.exists() else []
    player = EventPlayer(events)
    meta_path = npz.with_name("meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    duration = max(float(ts[-1]), 0.1)
    fps = max(1, int(round((n - 1) / duration)))
    driver = Driver(viewer=False)
    model, d = driver.model, driver.data
    W, H = 1280, 720
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    for f in (mujoco.mjtVisFlag.mjVIS_TENDON, mujoco.mjtVisFlag.mjVIS_JOINT,
              mujoco.mjtVisFlag.mjVIS_ACTUATOR, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
              mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_CONSTRAINT):
        opt.flags[f] = 0
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation = 1.35, -6.0
    cam.lookat[:] = (0.0, 0.05, 1.72)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
         str(out_mp4)], stdin=subprocess.PIPE)
    font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    for idx in range(n):
        t = float(ts[idx])
        d.qpos[:] = qpos[idx]
        model.body_pos[driver.ids["drone"]] = drone[idx, :3]
        yaw = float(drone[idx, 3])
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        model.body_quat[driver.ids["drone"]] = (cy, 0.0, 0.0, sy)
        e = float(min(max(P[idx] / 1.2, 0.0), 1.0))
        for k, w in enumerate(variant_weights(e)):
            for gid in driver.ids["g_lat"][k]:
                model.geom_rgba[gid][3] = w
            for gid in driver.ids["g_bundle"][k]:
                model.geom_rgba[gid][3] = w
            for gid in driver.ids["g_sheath"][k]:
                model.geom_rgba[gid][3] = w
        drift = float(np.linalg.norm(drone[idx, :3] - DRONE_HOME))
        tid = driver.ids["tether"]
        if tid >= 0:
            model.geom_rgba[tid][3] = min(max(1.0 - (drift - 0.03) / 0.10, 0.0), 1.0)
        mujoco.mj_forward(model, d)
        cam.azimuth = 150.0 - 25.0 * idx / max(n - 1, 1)
        cam.lookat[:] = (float(drone[idx, 0]), float(drone[idx, 1]), float(drone[idx, 2]) - 0.25)
        renderer.update_scene(d, cam, opt)
        im = Image.fromarray(renderer.render().copy())
        dr = ImageDraw.Draw(im, "RGBA")
        st = player.state_at(t)
        draw_keyboard(dr, st, 18, H - 196)
        _draw_curve(dr, ts[:idx + 1], P[:idx + 1], p_ref[:idx + 1], pwm[:idx + 1], valve[:idx + 1], W)
        _draw_caption(dr, float(P[idx]), font_mid)
        dr.text((W - 92, H - 26), "t=%.1f s" % t, font=font_mid, fill=(235, 238, 242, 110))
        proc.stdin.write(np.ascontiguousarray(np.asarray(im)).tobytes())
    proc.stdin.close()
    proc.wait()
    renderer.close()
    print("[replay] %d 帧 -> %s" % (n, out_mp4))


def _draw_caption(dr, p, font_mid):
    """左上角腔压小条。"""
    dr.rounded_rectangle((18, 18, 206, 50), radius=9, fill=(10, 12, 16, 155),
                         outline=(255, 255, 255, 36), width=1)
    dr.line((34, 34, 58, 34), fill=(235, 90, 80, 255), width=3)
    dr.text((68, 25), "腔压 %.2f bar" % p, font=font_mid, fill=(216, 221, 228, 255))


def _draw_curve(dr, ts, P, p_ref, pwm, valve, W):
    """右上角腔压曲线与泵阀指令，带小刻度与图例。"""
    if len(ts) < 2:
        return
    font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 12)
    x0, y0, w, h = W - 400, 18, 380, 152
    dr.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=9, fill=(10, 12, 16, 155),
                         outline=(255, 255, 255, 36), width=1)
    dr.line((x0 + 18, y0 + 22, x0 + 42, y0 + 22), fill=(235, 90, 80, 255), width=3)
    dr.text((x0 + 50, y0 + 13), "腔压", font=font_s, fill=(196, 202, 210, 235))
    dr.line((x0 + 100, y0 + 22, x0 + 124, y0 + 22), fill=(162, 168, 178, 220), width=3)
    dr.text((x0 + 132, y0 + 13), "指令", font=font_s, fill=(196, 202, 210, 235))
    px0, py0, pw, ph = x0 + 44, y0 + 42, w - 62, h - 100
    t_end = max(ts[-1], 1.0)
    for pv in (0.0, 0.6, 1.2):
        yy = py0 + ph * (1 - pv / 1.5)
        dr.line((px0, yy, px0 + pw, yy), fill=(255, 255, 255, 26))
        dr.text((x0 + 18, yy - 7), "%.1f" % pv, font=font_s, fill=(150, 156, 166, 210))
    pts = [(px0 + pw * float(t) / t_end, py0 + ph * (1 - min(float(p), 1.5) / 1.5))
           for t, p in zip(ts, P)]
    dr.line(pts, fill=(235, 90, 80, 255), width=2)
    ref = [(px0 + pw * float(t) / t_end, py0 + ph * (1 - min(float(pr), 1.5) / 1.5))
           for t, pr in zip(ts, p_ref)]
    for i in range(0, len(ref) - 3, 6):
        dr.line((ref[i][0], ref[i][1], ref[i + 3][0], ref[i + 3][1]), fill=(162, 168, 178, 170), width=2)
    last = len(ts) - 1
    by = y0 + h - 34
    dr.text((x0 + 18, by - 1), "泵", font=font_s, fill=(170, 176, 186, 220))
    dr.rectangle((x0 + 42, by + 2, x0 + 42 + 120 * float(pwm[last]), by + 12), fill=(90, 170, 235, 215))
    dr.rectangle((x0 + 42, by + 2, x0 + 162, by + 12), outline=(255, 255, 255, 45))
    dr.text((x0 + 178, by - 1), "阀", font=font_s, fill=(170, 176, 186, 220))
    dr.rectangle((x0 + 202, by + 2, x0 + 202 + 100 * float(valve[last]), by + 12), fill=(240, 170, 80, 215))
    dr.rectangle((x0 + 202, by + 2, x0 + 302, by + 12), outline=(255, 255, 255, 45))
