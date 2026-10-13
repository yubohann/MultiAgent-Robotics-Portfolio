"""模型驱动层，装配桁架臂与气路，仿真线程独占 mjData。"""
import math
import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "arm_real"))
import arm_lattice as L
import arm_real as A

DRONE_HOME = np.array([0.0, 0.0, L.DRONE_Z])
XY_LIMIT = 1.6
Z_LIMIT = (1.1, 2.7)

JOINT_KP0 = 14.0
JOINT_KV0 = 2.0
LOCK_STIFF0 = 6.0
LOCK_DAMP0 = 0.5
TUNE = {"base_stiff": A.K_SOFT, "base_damp": A.C_SOFT, "joint_rate": 0.35,
        "tendon_max": A.TENDON_FORCE, "p_ref_max": A.P_STEP, "noise": 0.012,
        "ext_rate": 0.15, "grip_rate": 0.6, "gripper": 0.0, "pneu_stiff": 2.0,
        "lock_stiff": LOCK_STIFF0, "lock_damp": LOCK_DAMP0, "lock_tau": 0.25,
        "grip_kp": 12.0, "grip_kv": 0.3,
        "v_max": 0.15, "yaw_max": 0.35, "home_speed": 0.3, "log_rate": 50.0}
DEFAULTS = dict(TUNE)
GRIPPER_N = len(L.GRIPPER_DEFS)
GRIPPER_DEFS = L.GRIPPER_DEFS


def band(x, a, b):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def variant_weights(e):
    """十一档网格按相邻两档线性插值，伸缩全程连续。"""
    x = min(max(e, 0.0), 1.0) * 10.0
    i = min(int(x), 9)
    f = x - i
    w = [0.0] * 11
    w[i] = 1.0 - f
    w[i + 1] = f
    return w


def build_model():
    xml = ROOT / "arm_real" / "_panel_lattice.xml"
    L.gen_xml(xml)
    text = xml.read_text(encoding="utf-8")
    joints = []
    for i in range(L.N_SEG):
        for ax in ("bx", "by"):
            joints.append('<position name="j_%s_%d" joint="%s_%d" kp="14.0" kv="2.0" '
                          'ctrlrange="-0.9 0.9"/>' % (ax, i, ax, i))
    text = text.replace("</actuator>", "\n".join(joints) + "\n  </actuator>")
    xml.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class Cmd:
    """控制线程写、仿真线程读的指令缓冲。"""

    def __init__(self, model):
        n = L.N_SEG
        self.lock = threading.Lock()
        self.bx_t = np.zeros(n)
        self.by_t = np.zeros(n)
        self.tendon = np.zeros(3)
        self.grip = 0.0
        self.p_ref = 0.0
        self.locks = np.zeros(n, dtype=bool)
        self.drone_pos = DRONE_HOME.copy()
        self.drone_yaw = 0.0
        self.vel_local = np.zeros(3)
        self.yaw_rate = 0.0
        self.home = False
        self.hover = False
        self.mode = "combo"
        self.estop = False
        self.seq_phase = ""

    def copy(self):
        c = {}
        for k in ("bx_t", "by_t", "tendon", "grip", "p_ref", "locks", "drone_pos",
                  "drone_yaw", "vel_local", "yaw_rate", "home", "hover", "mode",
                  "estop", "seq_phase"):
            v = getattr(self, k)
            c[k] = v.copy() if isinstance(v, np.ndarray) else v
        return c


class Driver:
    def __init__(self, viewer=False):
        self.model, self.data = build_model()
        self.pneu = A.Pneumatic()
        self.cmd = Cmd(self.model)
        self.stiff_cur = np.full(L.N_SEG, TUNE["base_stiff"])
        self.damp_cur = np.full(L.N_SEG, TUNE["base_damp"])
        self._last_e = -1.0
        self._last_grip = -1
        self._gctrl = -0.40
        self.hold = False
        self._hold_rel = None
        self.ids = self._cache_ids()
        L.set_gripper(self.model, 0)
        self.viewer = viewer
        self.frames = 0

    def _cache_ids(self):
        m = self.model
        ids = {"j_bx": [], "j_by": [], "j_bx_qadr": [], "j_by_qadr": [],
               "j_bx_dadr": [], "j_by_dadr": [], "slide": [], "slide_qadr": {},
               "slide_dadr": {}, "slide_act": {}, "act_j_bx": [], "act_j_by": [],
               "g_lat": [[] for _ in range(11)], "g_bundle": [[] for _ in range(11)],
               "g_sheath": [[] for _ in range(11)]}
        for i in range(L.N_SEG):
            for ax in ("bx", "by"):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "%s_%d" % (ax, i))
                ids["j_" + ax].append(jid)
                ids["j_" + ax + "_qadr"].append(m.jnt_qposadr[jid])
                ids["j_" + ax + "_dadr"].append(m.jnt_dofadr[jid])
                ids["act_j_" + ax].append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "j_%s_%d" % (ax, i)))
            sjid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_%d" % i)
            if sjid >= 0:
                ids["slide"].append(sjid)
                ids["slide_qadr"][i] = m.jnt_qposadr[sjid]
                ids["slide_dadr"][i] = m.jnt_dofadr[sjid]
                ids["slide_act"][i] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_%d_pos" % i)
            for k in range(11):
                ids["g_lat"][k].append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_lat_v%02d_%d" % (k, i)))
                ids["g_bundle"][k].append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_bundle_v%02d_%d" % (k, i)))
                ids["g_sheath"][k].append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_sheath_v%02d_%d" % (k, i)))
        wsj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slide_wrist")
        ids["wrist_slide_qadr"] = m.jnt_qposadr[wsj]
        ids["wrist_slide_dadr"] = m.jnt_dofadr[wsj]
        ids["slide_wrist_act"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_wrist_pos")
        for nm in ("tendon_m0", "tendon_m1", "tendon_m2", "gauge_pos"):
            ids[nm] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        ids["wrist"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist")
        ids["grip_act"] = []
        ids["grip_qadr"] = []
        ids["grip_bodies"] = []
        for v in range(len(L.GRIPPER_DEFS)):
            acts, qadrs, bodies = [], [], []
            i = 0
            while True:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip%d_%d" % (v, i))
                if jid < 0:
                    break
                qadrs.append(m.jnt_qposadr[jid])
                acts.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip%d_%d_pos" % (v, i)))
                bodies.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "finger%d_%d" % (v, i)))
                i += 1
            ids["grip_act"].append(acts)
            ids["grip_qadr"].append(qadrs)
            ids["grip_bodies"].append(bodies)
        ids["drone"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
        ids["part"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")
        pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "part_free")
        ids["part_qadr"] = m.jnt_qposadr[pj]
        ids["part_dadr"] = m.jnt_dofadr[pj]
        ids["weld"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        ids["tether"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_tether")
        ids["gauge_j"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gauge_j")
        return ids

    def grip_contacts(self):
        fb = self.ids["grip_bodies"][self.active_gripper()]
        n = 0
        try:
            ncon = min(int(self.data.ncon), len(self.data.contact))
            for i in range(ncon):
                c = self.data.contact[i]
                b1 = self.model.geom_bodyid[c.geom1]
                b2 = self.model.geom_bodyid[c.geom2]
                if (b1 in fb and b2 not in fb) or (b2 in fb and b1 not in fb):
                    n += 1
        except (IndexError, ValueError):
            pass
        return n

    def active_gripper(self):
        return int(round(TUNE["gripper"])) % len(L.GRIPPER_DEFS)

    def step(self, dt):
        d, m = self.data, self.model
        with self.cmd.lock:
            cmd = self.cmd.copy()
        if cmd["estop"]:
            cmd["tendon"] = np.zeros(3)
            cmd["vel_local"] = np.zeros(3)
            cmd["yaw_rate"] = 0.0
        self.pneu.noise_sd = TUNE["noise"]
        self.pneu.step(cmd["p_ref"], dt)
        p = self.pneu.P
        e = min(max(p / TUNE["p_ref_max"], 0.0), 1.0)
        tau = max(TUNE["lock_tau"], 1e-3)
        rate = min(dt / tau, 1.0)
        for i in range(L.N_SEG):
            ks = TUNE["lock_stiff"] if cmd["locks"][i] else (TUNE["base_stiff"] + e * TUNE["pneu_stiff"])
            kd = TUNE["lock_damp"] if cmd["locks"][i] else TUNE["base_damp"]
            self.stiff_cur[i] += (ks - self.stiff_cur[i]) * rate
            self.damp_cur[i] += (kd - self.damp_cur[i]) * rate
            for ax in ("bx", "by"):
                jid = self.ids["j_" + ax][i]
                m.jnt_stiffness[jid] = self.stiff_cur[i]
                m.dof_damping[m.jnt_dofadr[jid]] = self.damp_cur[i]
            if i > 0:
                d.ctrl[self.ids["slide_act"][i]] = -0.040 * e
        d.ctrl[self.ids["slide_wrist_act"]] = -0.040 * e
        if abs(e - self._last_e) > 0.0005:
            for k, w in enumerate(variant_weights(e)):
                for gid in self.ids["g_lat"][k]:
                    m.geom_rgba[gid][3] = w
                for gid in self.ids["g_bundle"][k]:
                    m.geom_rgba[gid][3] = w
                for gid in self.ids["g_sheath"][k]:
                    m.geom_rgba[gid][3] = w
            self._last_e = e
        for i in range(L.N_SEG):
            d.ctrl[self.ids["act_j_bx"][i]] = cmd["bx_t"][i]
            d.ctrl[self.ids["act_j_by"][i]] = cmd["by_t"][i]
        tm = TUNE["tendon_max"]
        for k in range(3):
            aid = self.ids["tendon_m%d" % k]
            m.actuator_gainprm[aid, 2] = tm
            d.ctrl[aid] = cmd["tendon"][k]
        gv = self.active_gripper()
        if gv != self._last_grip:
            L.set_gripper(m, gv)
            self._last_grip = gv
        if cmd["estop"] or cmd["grip"] < 0.4:
            self.hold = False
        elif not self.hold and cmd["grip"] > 0.55:
            dp = float(np.linalg.norm(d.xpos[self.ids["part"]] - d.xpos[self.ids["wrist"]]))
            if dp < 0.18:
                bp, bw = self.ids["part"], self.ids["wrist"]
                inv = np.zeros(4)
                mujoco.mju_negQuat(inv, d.xquat[bw])
                rel = np.zeros(4)
                mujoco.mju_mulQuat(rel, inv, d.xquat[bp])
                relpos = np.zeros(3)
                mujoco.mju_rotVecQuat(relpos, d.xpos[bp] - d.xpos[bw], inv)
                self._hold_rel = (relpos.copy(), rel.copy())
                self.hold = True
        if self.hold and self._hold_rel is not None:
            bp, bw = self.ids["part"], self.ids["wrist"]
            relpos, rel = self._hold_rel
            rm = np.zeros(9)
            mujoco.mju_quat2Mat(rm, d.xquat[bw])
            pp = d.xpos[bw] + rm.reshape(3, 3) @ relpos
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, d.xquat[bw], rel)
            d.qpos[self.ids["part_qadr"]:self.ids["part_qadr"] + 3] = pp
            d.qpos[self.ids["part_qadr"] + 3:self.ids["part_qadr"] + 7] = q
            d.qvel[self.ids["part_dadr"]:self.ids["part_dadr"] + 6] = 0.0
        weld_on = self.hold
        if weld_on:
            self._gctrl += (0.02 - self._gctrl) * min(dt * 6.0, 1.0)
        else:
            self._gctrl = -0.40 + 0.70 * cmd["grip"]
        for v in range(len(L.GRIPPER_DEFS)):
            c = self._gctrl if v == gv else -0.40
            for aid in self.ids["grip_act"][v]:
                d.ctrl[aid] = c
                m.actuator_gainprm[aid, 0] = TUNE["grip_kp"]
                m.actuator_biasprm[aid, 1] = -TUNE["grip_kp"]
                m.actuator_gainprm[aid, 1] = TUNE["grip_kv"]
                m.actuator_biasprm[aid, 2] = -TUNE["grip_kv"]
        d.ctrl[self.ids["gauge_pos"]] = -1.0 + 2.0 * min(p / A.P_MAX, 1.0)
        m.body_pos[self.ids["drone"]] = cmd["drone_pos"]
        cy = math.cos(cmd["drone_yaw"] * 0.5)
        sy = math.sin(cmd["drone_yaw"] * 0.5)
        m.body_quat[self.ids["drone"]] = (cy, 0.0, 0.0, sy)
        drift = float(np.linalg.norm(cmd["drone_pos"] - DRONE_HOME))
        tid = self.ids["tether"]
        if tid >= 0:
            m.geom_rgba[tid][3] = min(max(1.0 - (drift - 0.03) / 0.10, 0.0), 1.0)
        mujoco.mj_step(m, d)
        self.frames += 1

    def snapshot(self):
        d = self.data
        wrist = d.xpos[self.ids["wrist"]]
        with self.cmd.lock:
            cmd = self.cmd.copy()
        return {
            "t": time.monotonic(),
            "qpos": d.qpos.copy(),
            "wrist": wrist.copy(),
            "contacts": self.grip_contacts(),
            "P": self.pneu.P,
            "P_meas": self.pneu.P_meas,
            "p_ref": cmd["p_ref"],
            "pwm": self.pneu.pwm,
            "valve": self.pneu.valve,
            "mode": cmd["mode"],
            "estop": cmd["estop"],
            "locks": cmd["locks"].copy(),
            "grip": cmd["grip"],
            "hold": self.hold,
            "drone_pos": cmd["drone_pos"].copy(),
            "drone_yaw": cmd["drone_yaw"],
            "seq_phase": cmd["seq_phase"],
            "home": cmd["home"],
            "hover": cmd["hover"],
        }

    def teach_pose(self):
        with self.cmd.lock:
            return (self.cmd.bx_t.copy(), self.cmd.by_t.copy(), self.cmd.grip, self.cmd.p_ref)

    def teach_apply(self, pose):
        with self.cmd.lock:
            self.cmd.bx_t[:] = pose[0]
            self.cmd.by_t[:] = pose[1]
            self.cmd.grip = pose[2]
            self.cmd.p_ref = pose[3]


def make_params():
    def g(key):
        return lambda: TUNE[key]

    def s(key):
        def f(v):
            TUNE[key] = v
        return f

    return [
        ("机械臂", "基准刚度", "base_stiff", 0.2, 4.0, 0.05),
        ("机械臂", "基准阻尼", "base_damp", 0.02, 1.0, 0.02),
        ("机械臂", "关节速率", "joint_rate", 0.05, 1.2, 0.05),
        ("机械臂", "腱力上限", "tendon_max", 5.0, 60.0, 1.0),
        ("气动", "压力上限", "p_ref_max", 0.4, 2.0, 0.05),
        ("气动", "伸缩速率", "ext_rate", 0.05, 0.6, 0.02),
        ("气动", "压力刚度", "pneu_stiff", 0.0, 6.0, 0.25),
        ("气动", "传感噪声", "noise", 0.0, 0.05, 0.005),
        ("变刚度", "锁定刚度", "lock_stiff", 2.0, 20.0, 0.25),
        ("变刚度", "锁定阻尼", "lock_damp", 0.1, 3.0, 0.05),
        ("变刚度", "切换时间", "lock_tau", 0.02, 1.0, 0.02),
        ("夹爪", "夹爪速率", "grip_rate", 0.1, 2.0, 0.1),
        ("夹爪", "夹爪型号", "gripper", 0.0, 3.0, 1.0),
        ("夹爪", "夹爪刚度", "grip_kp", 1.0, 40.0, 0.5),
        ("夹爪", "夹爪阻尼", "grip_kv", 0.0, 4.0, 0.1),
        ("无人机", "平移速度", "v_max", 0.02, 0.6, 0.02),
        ("无人机", "偏航速度", "yaw_max", 0.05, 1.0, 0.05),
        ("无人机", "回原速度", "home_speed", 0.05, 0.8, 0.05),
        ("日志", "采样率", "log_rate", 10.0, 200.0, 10.0),
    ]
