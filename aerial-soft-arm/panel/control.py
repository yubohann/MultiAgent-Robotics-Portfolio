"""控制线程，读键态生成指令，管一键序列与无人机运动学。"""
import math
import threading
import time

import numpy as np

from drivers import TUNE, DRONE_HOME, XY_LIMIT, Z_LIMIT, GRIPPER_N
from keys import KeyState

ARM_MOTION = {"w", "s", "a", "d", "q", "e", "r", "f", "z", "c", "1", "2", "3"}
DRONE_MOTION = {"up", "down", "left", "right", "k8", "k2", "k4", "k6"}
LOCK_GROUPS = {"4": (0, 3), "5": (3, 6), "6": (6, 8)}

SEQ_X = [("对准", 2.4), ("伸展", 4.0), ("闭合", 1.5), ("提升", 1.6), ("保持", 0.0)]
SEQ_G = [("对准", 2.4), ("伸展", 4.0), ("闭合", 1.5), ("提升", 1.6), ("放置", 2.0), ("回中", 2.4), ("保持", 0.0)]
SEQ_EXT_RATE = 0.25
GRASP_XY = (0.0, 0.30)
GRASP_Z = 2.02
LIFT_Z = 2.17
GRASP_P = 0.75


class ControlThread(threading.Thread):
    def __init__(self, driver, keys: KeyState, stop_evt):
        super().__init__(name="control", daemon=True)
        self.d = driver
        self.keys = keys
        self.stop = stop_evt
        self.seq = None
        self.teach = None

    def run(self):
        dt = 0.01
        while not self.stop.is_set():
            self.tick(dt)
            time.sleep(dt)

    def tick(self, dt):
        d, cmd = self.d, self.d.cmd
        ks = self.keys.snapshot()
        events = self.keys.consume_events()
        held = ks["held"]
        mode = ks["mode"]
        estop = ks["estop"]
        arm_ok = mode in ("combo", "precise")
        drone_ok = mode in ("combo", "drone")
        scale = 0.5 if mode == "precise" else 1.0

        with cmd.lock:
            cmd.mode = mode
            cmd.estop = estop

            if estop:
                if self.seq:
                    self.seq = None
                    cmd.seq_phase = ""
                cmd.bx_t[:] = self.d.data.qpos[self.d.ids["j_bx_qadr"]]
                cmd.by_t[:] = self.d.data.qpos[self.d.ids["j_by_qadr"]]
                cmd.tendon[:] = 0.0
                cmd.vel_local[:] = 0.0
                cmd.yaw_rate = 0.0
                return

            for ev in events:
                if ev in LOCK_GROUPS:
                    lo, hi = LOCK_GROUPS[ev]
                    cmd.locks[lo:hi] = ~cmd.locks[lo:hi]
                elif ev == "x":
                    self.seq = {"ph": 0, "t": 0.0, "steps": SEQ_X}
                    cmd.seq_phase = SEQ_X[0][0]
                elif ev == "g":
                    self.seq = {"ph": 0, "t": 0.0, "steps": SEQ_G}
                    cmd.seq_phase = SEQ_G[0][0]
                elif ev == "t":
                    TUNE["gripper"] = (int(round(TUNE["gripper"])) + 1) % GRIPPER_N
                elif ev == "v":
                    self.teach = (cmd.bx_t.copy(), cmd.by_t.copy(), cmd.grip, cmd.p_ref)
                elif ev == "b":
                    if self.teach:
                        cmd.bx_t[:] = self.teach[0]
                        cmd.by_t[:] = self.teach[1]
                        cmd.grip = self.teach[2]
                        cmd.p_ref = self.teach[3]
                elif ev == "k5":
                    cmd.hover = not cmd.hover
                elif ev == "k0":
                    cmd.home = True

            seq_active = self.seq is not None
            if seq_active and arm_ok:
                self._run_seq(dt, cmd)
            elif arm_ok:
                self._arm_manual(dt, cmd, held, scale)
            if self.seq and self.seq.get("done"):
                self.seq = None

            self._drone(dt, cmd, held, drone_ok, estop)

    def _arm_manual(self, dt, cmd, held, scale):
        cmd.seq_phase = ""
        rate = TUNE["joint_rate"] * scale * dt
        if "w" in held:
            cmd.bx_t += rate
        if "s" in held:
            cmd.bx_t -= rate
        if "a" in held:
            cmd.by_t += rate
        if "d" in held:
            cmd.by_t -= rate
        np.clip(cmd.bx_t, -0.9, 0.9, out=cmd.bx_t)
        np.clip(cmd.by_t, -0.9, 0.9, out=cmd.by_t)
        pr = TUNE["ext_rate"] * dt
        if "q" in held:
            cmd.p_ref = min(cmd.p_ref + pr, TUNE["p_ref_max"])
        if "e" in held:
            cmd.p_ref = max(cmd.p_ref - pr, 0.0)
        g = TUNE["grip_rate"] * dt
        if "c" in held:
            cmd.grip = min(cmd.grip + g, 1.0)
        if "z" in held:
            cmd.grip = max(cmd.grip - g, 0.0)
        tgt = np.zeros(3)
        if "r" in held:
            tgt[0] = 1.0
        if "f" in held:
            tgt[1] = 0.7
            tgt[2] = 0.7
        for k in (1, 2, 3):
            if str(k) in held:
                tgt[k - 1] = 1.0
        cmd.tendon += (tgt - cmd.tendon) * min(dt * 12.0, 1.0)

    def _move_to(self, cmd, target, dt):
        delta = np.asarray(target, dtype=float) - cmd.drone_pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-4:
            return True
        step = min(TUNE["v_max"] * dt * 1.4, dist)
        cmd.drone_pos += delta / dist * step
        return False

    def _run_seq(self, dt, cmd):
        ph, t, steps = self.seq["ph"], self.seq["t"], self.seq["steps"]
        name, dur = steps[ph]
        cmd.seq_phase = name
        if name == "对准":
            self._move_to(cmd, (GRASP_XY[0], GRASP_XY[1], GRASP_Z), dt)
            cmd.grip = max(cmd.grip - 1.5 * dt, 0.0)
            cmd.p_ref = max(cmd.p_ref - TUNE["ext_rate"] * dt, 0.0)
            cmd.bx_t[:] = 0.0
            cmd.by_t[:] = 0.0
        elif name == "伸展":
            cmd.p_ref = min(cmd.p_ref + SEQ_EXT_RATE * dt, GRASP_P * TUNE["p_ref_max"])
            cmd.locks[:] = True
        elif name == "闭合":
            cmd.grip = min(cmd.grip + 1.5 * dt, 1.0)
        elif name == "提升":
            cmd.p_ref = max(cmd.p_ref - SEQ_EXT_RATE * dt, 0.35 * TUNE["p_ref_max"])
        elif name == "放置":
            cmd.p_ref = min(cmd.p_ref + SEQ_EXT_RATE * dt, GRASP_P * TUNE["p_ref_max"])
            if cmd.p_ref >= GRASP_P * TUNE["p_ref_max"] - 1e-6:
                cmd.grip = max(cmd.grip - 1.5 * dt, 0.0)
        elif name == "回中":
            self._move_to(cmd, DRONE_HOME, dt)
            cmd.p_ref = max(cmd.p_ref - SEQ_EXT_RATE * dt, 0.0)
            if cmd.p_ref <= 0.0:
                cmd.locks[:] = False
        t += dt
        if dur > 0 and t < dur:
            self.seq["t"] = t
        else:
            self.seq["ph"] += 1
            self.seq["t"] = 0.0
            if self.seq["ph"] >= len(steps):
                self.seq["done"] = True

    def _drone(self, dt, cmd, held, ok, estop):
        if estop or not ok:
            cmd.vel_local[:] = 0.0
            cmd.yaw_rate = 0.0
            if estop:
                cmd.home = False
            return
        if cmd.home:
            delta = DRONE_HOME - cmd.drone_pos
            dist = float(np.linalg.norm(delta))
            if dist < 0.01 and abs(cmd.drone_yaw) < 0.01:
                cmd.drone_pos = DRONE_HOME.copy()
                cmd.drone_yaw = 0.0
                cmd.home = False
            else:
                step = TUNE["home_speed"] * dt
                cmd.drone_pos += delta / max(dist, 1e-6) * min(step, dist)
                if abs(cmd.drone_yaw) > 0.01:
                    cmd.drone_yaw -= math.copysign(min(abs(cmd.drone_yaw), step * 2.0), cmd.drone_yaw)
            cmd.vel_local[:] = 0.0
            cmd.yaw_rate = 0.0
            return
        if cmd.hover:
            cmd.vel_local[:] = 0.0
            cmd.yaw_rate = 0.0
            return
        fwd = (1.0 if "up" in held else 0.0) - (1.0 if "down" in held else 0.0)
        lat = (1.0 if "right" in held else 0.0) - (1.0 if "left" in held else 0.0)
        vert = (1.0 if "k8" in held else 0.0) - (1.0 if "k2" in held else 0.0)
        yaw = (1.0 if "k6" in held else 0.0) - (1.0 if "k4" in held else 0.0)
        cmd.vel_local[0] = fwd * TUNE["v_max"]
        cmd.vel_local[1] = -lat * TUNE["v_max"]
        cmd.vel_local[2] = vert * TUNE["v_max"]
        cmd.yaw_rate = yaw * TUNE["yaw_max"]
        c, s = math.cos(cmd.drone_yaw), math.sin(cmd.drone_yaw)
        v = cmd.vel_local
        cmd.drone_pos[0] += (v[0] * c - v[1] * s) * dt
        cmd.drone_pos[1] += (v[0] * s + v[1] * c) * dt
        cmd.drone_pos[2] += v[2] * dt
        cmd.drone_yaw += cmd.yaw_rate * dt
        np.clip(cmd.drone_pos[0], -XY_LIMIT, XY_LIMIT, out=cmd.drone_pos[0:1])
        np.clip(cmd.drone_pos[1], -XY_LIMIT, XY_LIMIT, out=cmd.drone_pos[1:2])
        np.clip(cmd.drone_pos[2], Z_LIMIT[0], Z_LIMIT[1], out=cmd.drone_pos[2:3])
        cmd.drone_yaw = (cmd.drone_yaw + math.pi) % (2 * math.pi) - math.pi
