"""Soft-Spiral 手动操控抓取（键盘交互版）。

运行：python spiral_grasp_manual.py
需要图形界面（本机桌面会话）。按键：
  K / L   收卷肌腱 A1 / 抬升肌腱 A2（按住加力，松开回弹）
  Z / C   夹爪闭合 / 张开（按住持续动作）
  X       自动夹持辅助：轻合 -> 保压 -> 提升（标定配方）
  R       重置场景    空格 暂停    Q / ESC 退出
自检（无界面）：python spiral_grasp_manual.py --selftest
"""
import argparse
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_scene as G  # noqa: E402

RATE_UP, RATE_DOWN = 0.45, 0.8
GRIP_RATE = 0.35
ASSIST_CLOSE_T, ASSIST_HOLD_T, ASSIST_LIFT_T = 0.3, 1.2, 6.0


class ManualController:
    """把按键状态翻译成 (a1, a2, a3, a4)。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.a1 = 0.0
        self.a2 = 0.0
        self.grip = 0.0
        self.assist = None
        self.assist_t = 0.0
        self.latched = False

    def update(self, dt, held, pressed):
        if "r" in pressed:
            self.reset()
            return (0.0, 0.0, 0.0, 0.0), True

        if "x" in pressed and self.assist is None:
            self.assist = "close"
            self.assist_t = 0.0
            self.latched = False

        if self.assist is None and self.latched:
            if any(held[k] for k in ("k", "l", "z", "c")) or any(k in pressed for k in ("k", "l", "z", "c")):
                self.latched = False
            else:
                return (self.a1, self.a2, self.grip, self.grip), False

        if self.assist is not None:
            self.assist_t += dt
            if self.assist == "close":
                u = min(self.assist_t / ASSIST_CLOSE_T, 1.0)
                self.grip = G.CLOSE_LEVEL * u
                if self.assist_t >= ASSIST_CLOSE_T:
                    self.assist, self.assist_t = "hold", 0.0
            elif self.assist == "hold":
                self.grip = G.CLOSE_LEVEL
                if self.assist_t >= ASSIST_HOLD_T:
                    self.assist, self.assist_t = "lift", 0.0
            else:
                u = 0.5 - 0.5 * np.cos(np.pi * min(self.assist_t / ASSIST_LIFT_T, 1.0))
                self.a1 = 1.0
                self.a2 = u
                self.grip = G.CLOSE_LEVEL + (1.0 - G.CLOSE_LEVEL) * u
                if self.assist_t >= ASSIST_LIFT_T:
                    self.assist = None
                    self.latched = True
            return (self.a1, self.a2, self.grip, self.grip), False

        self.a1 += ((1.0 if held["k"] else 0.0) - self.a1) * dt / (RATE_UP if held["k"] else RATE_DOWN)
        self.a2 += ((1.0 if held["l"] else 0.0) - self.a2) * dt / (RATE_UP if held["l"] else RATE_DOWN)
        self.a1 = min(max(self.a1, 0.0), 1.0)
        self.a2 = min(max(self.a2, 0.0), 1.0)
        if held["z"]:
            self.grip = min(self.grip + GRIP_RATE * dt, 1.0)
        elif held["c"]:
            self.grip = max(self.grip - GRIP_RATE * dt, 0.0)
        return (self.a1, self.a2, self.grip, self.grip), False


def contacts(model, data, body_a, body_b):
    ca = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_a)
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_b)
    n = 0
    for i in range(data.ncon):
        b1 = model.geom_bodyid[data.contact[i].geom1]
        b2 = model.geom_bodyid[data.contact[i].geom2]
        if ca in (b1, b2) and cb in (b1, b2):
            n += 1
    return n


def status_line(model, data):
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    g1, g2, db = bid("gripper_1"), bid("gripper_2"), bid("dumbbell")
    mid = (data.xpos[g1] + data.xpos[g2]) / 2
    gap = np.linalg.norm(data.xpos[db] - mid) * 1000
    cf = contacts(model, data, "gripper_1", "dumbbell") + contacts(model, data, "gripper_2", "dumbbell")
    return data.xpos[db][2] * 1000, gap, cf


def run_selftest():
    p = G.build_scene()
    m = mujoco.MjModel.from_xml_path(str(p))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    ctrl = ManualController()
    held = defaultdict(bool)
    log = []

    def step(dur, held_keys=(), presses=()):
        held.clear()
        for k in held_keys:
            held[k] = True
        pressed = set(presses)
        n = int(dur / dt)
        for i in range(n):
            a, _ = ctrl.update(dt, held, pressed if i == 0 else set())
            d.ctrl[:] = a
            mujoco.mj_step(m, d)

    step(1.0)
    step(2.2, held_keys=("k",))          # 下探
    step(2.5, held_keys=("k",))          # 就位（保持按住）
    step(1.0, held_keys=("k",), presses=("x",))  # 自动夹持辅助
    step(16.0)                           # 辅助提升 + 回弹归位
    z, gap, cf = status_line(m, d)
    print("selftest: dumbbell z=%.0fmm gap=%.1fmm contacts=%d" % (z, gap, cf))
    ok = cf > 0 and gap < 35 and z > G.BAR[2] * 1000 + 100
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_viewer():
    p = G.build_scene()
    m = mujoco.MjModel.from_xml_path(str(p))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep

    try:
        from pynput import keyboard
    except ImportError:
        print("需要 pynput：pip install pynput")
        return 1

    held = defaultdict(bool)
    pressed = set()
    quit_flag = {"q": False}

    def on_press(key):
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            ch = None
        if ch in ("k", "l", "z", "c", "x", "r", "q"):
            if ch not in held:
                pressed.add(ch)
            held[ch] = True
        if key == keyboard.Key.space:
            held["space"] = True
            pressed.add("space")
        if key == keyboard.Key.esc:
            quit_flag["q"] = True

    def on_release(key):
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            ch = None
        if ch in held:
            held[ch] = False
        if key == keyboard.Key.space:
            held["space"] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    import mujoco.viewer
    ctrl = ManualController()
    paused = False
    steps_per_frame = 4
    frame_dt = steps_per_frame * dt

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 129
        viewer.cam.elevation = -12
        viewer.cam.lookat[:] = (0.0, 0.32, 0.32)
        next_t = time.perf_counter()
        next_print = 0.0
        while viewer.is_running() and not quit_flag["q"]:
            if "space" in pressed:
                paused = not paused
            if paused:
                a = d.ctrl.copy()
            else:
                a, _ = ctrl.update(frame_dt, held, pressed)
                for _ in range(steps_per_frame):
                    d.ctrl[:] = a
                    mujoco.mj_step(m, d)
            pressed.clear()
            z, gap, cf = status_line(m, d)
            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "K/L 收卷/抬升  Z/C 夹爪  X 自动夹持  R 重置  空格 %s  Q 退出" % ("暂停中" if paused else "运行"),
                 "A1=%.2f A2=%.2f 夹爪=%.2f" % (ctrl.a1, ctrl.a2, ctrl.grip)),
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 "哑铃杆 z=%.0f mm" % z, "夹持偏差 %.0f mm  接触 %d" % (gap, cf)),
            ])
            viewer.sync()
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += frame_dt
            if now - next_print > 1.0:
                next_print = now
                print("A1=%.2f A2=%.2f grip=%.2f | bar z=%.0fmm gap=%.0fmm contacts=%d" %
                      (ctrl.a1, ctrl.a2, ctrl.grip, z, gap, cf))
    listener.stop()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()
    return run_viewer()


if __name__ == "__main__":
    sys.exit(main())
