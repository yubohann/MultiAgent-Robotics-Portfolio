"""空中抓取闭环（v3）：软臂 + 两指夹爪 + 立架目标，写死与视觉两套管线。

在 v2 臂（空间弯曲、分段锁定、5 级伸缩）末端加两指夹爪；地面立架上横放
哑铃杆作为目标。抓取流程：悬停 -> 伸缩下探 -> 夹爪闭合 -> 缩回提升 -> 放回。

运行：
  python aerial_grasp.py --selftest        写死版自检
  python aerial_grasp.py --render          渲染写死版 HD 视频
  python aerial_grasp.py --vision-selftest 视觉伺服自检
  python aerial_grasp.py --vision-render   渲染视觉版 HD 视频（带腕相机画中画）
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN_DIR = Path(tempfile.gettempdir()) / "aerial_grasp"
OUT_DIR = Path(r"C:\Users\Administrator\Desktop\aerial_soft_arm\videos\newmodel")

N_SEG = 12
SEG_L = 0.045
SEG_R0, SEG_R1 = 0.020, 0.010
N_TELE = 5
TELE_TRAVEL = 0.12
TELE_SPACING = 0.032
TUBE_C, TUBE_H = -0.075, 0.09
HOUSING_R = 0.026
TENDON_ANGLES = (90.0, 210.0, 330.0)
TENDON_R = 0.016
TENDON_FORCE = 25.0
K_SOFT, C_SOFT = 1.2, 0.08
K_LOCK, C_LOCK = 25.0, 1.5
SECTIONS = ((1, 4), (5, 8), (9, 12))

DRONE_Z = 2.15
BAR_Z = 0.929
FINGER_X = 0.015
FINGER_LEN = 0.04
GRIP_OPEN, GRIP_CLOSE = 0.0, 0.012
REACH_TELE = 0.19
LIFT_TELE = 0.005


def gen_xml(path, wrist_cam=False):
    segs = []
    for i in range(N_SEG):
        r = SEG_R0 + (SEG_R1 - SEG_R0) * i / (N_SEG - 1)
        segs.append((r, SEG_L))
    s = []
    s.append('<mujoco model="aerial_grasp">')
    s.append('<compiler angle="radian"/>')
    s.append('<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>')
    s.append("""
  <asset>
    <material name="drone_mat" rgba="0.12 0.16 0.20 1"/>
    <material name="tube_mat" rgba="0.35 0.40 0.48 1"/>
    <material name="seg_mat0" rgba="0.16 0.32 0.62 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="seg_mat1" rgba="0.25 0.62 0.85 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="seg_mat2" rgba="0.85 0.45 0.15 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="mat_grip" rgba="0.93 0.62 0.24 1" specular="0.7" shininess="0.6" roughness="0.3"/>
    <material name="mat_grip2" rgba="0.88 0.58 0.22 1" specular="0.7" shininess="0.6" roughness="0.3"/>
    <material name="mat_stand" rgba="0.42 0.45 0.50 1" specular="0.5" shininess="0.45" roughness="0.4"/>
    <material name="mat_target" rgba="0.10 0.85 0.25 1" specular="0.55" shininess="0.5" roughness="0.35"/>
    <material name="mat_ball" rgba="0.95 0.42 0.10 1" specular="0.85" shininess="0.7" roughness="0.22"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".205 .215 .235" rgb2=".125 .132 .145"
             width="512" height="512" mark="cross" markrgb=".55 .57 .6"/>
    <material name="matplane" reflectance="0.24" shininess="0.25" roughness="0.60"
              texture="texplane" texrepeat="4 4" texuniform="true"/>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.46 0.53" rgb2="0.02 0.03 0.05"
             width="512" height="3072"/>
  </asset>
  <visual>
    <quality shadowsize="8192" offsamples="16"/>
    <headlight ambient="0.18 0.19 0.22" diffuse="0.30 0.30 0.32" specular="0.06 0.06 0.06"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
""")
    s.append('<worldbody>')
    s.append('<geom name="ground" type="plane" size="10 10 1" pos="0 0 0" material="matplane" condim="1"/>')
    s.append('<light name="key" directional="true" pos="0.9 -0.6 2.6" dir="-0.35 0.25 -1" '
             'diffuse="0.80 0.78 0.74" specular="0.55 0.55 0.55" castshadow="true"/>')
    s.append('<light name="fill" directional="true" pos="-1.2 0.6 1.8" dir="0.6 -0.3 -1" '
             'diffuse="0.25 0.28 0.34" specular="0.05 0.05 0.05"/>')

    # ---------------- 立架 + 目标（哑铃杆横放） ----------------
    post_top = BAR_Z - 0.004
    s.append('<body name="stand" pos="0 0 0">')
    s.append('<geom name="pedestal" type="cylinder" pos="0 0 0.44" size="0.05 0.44" material="mat_stand" mass="1.0"/>')
    s.append('<geom name="table_top" type="box" pos="0 0 0.885" size="0.08 0.08 0.005" material="mat_stand" mass="0.2"/>')
    for sy in (0.045, -0.045):
        s.append('<geom name="post_%d" type="cylinder" pos="0 %.4f %.4f" size="0.004 %.4f" material="mat_stand" mass="0.02"/>'
                 % (0 if sy > 0 else 1, sy, (post_top + 0.89) / 2, (post_top - 0.89) / 2))
    s.append('</body>')
    s.append('<body name="dumbbell" pos="0 0 %.4f">' % BAR_Z)
    s.append('<joint type="free" damping="0.3"/>')
    s.append('<geom name="bar_geom" type="capsule" fromto="0 -0.045 0 0 0.045 0" size="0.004" '
             'mass="0.006" friction="4 0.01 0.001" condim="6" solimp="0.95 0.99 0.001" solref="0.002 1" '
             'material="mat_target"/>')
    s.append('<geom name="ball_l" type="sphere" size="0.012" pos="0 -0.057 0" mass="0.005" '
             'friction="4 0.01 0.001" condim="6" solimp="0.95 0.99 0.001" solref="0.002 1" material="mat_ball"/>')
    s.append('<geom name="ball_r" type="sphere" size="0.012" pos="0 0.057 0" mass="0.005" '
             'friction="4 0.01 0.001" condim="6" solimp="0.95 0.99 0.001" solref="0.002 1" material="mat_ball"/>')
    s.append('</body>')

    # ---------------- 六旋翼 ----------------
    s.append('<body name="drone" pos="0 0 %.2f">' % DRONE_Z)
    s.append('<freejoint name="drone_free"/>')
    s.append('<site name="imu_site" pos="0 0 0" size="0.008"/>')
    s.append('<site name="manipulator_mount" pos="0 0 -0.05" size="0.01"/>')
    s.append('<geom name="drone_panel_1" type="box" pos="0 0 0" size="0.155 0.090 0.04" material="drone_mat" mass="0.50"/>')
    s.append('<geom name="drone_panel_2" type="box" pos="0 0 0" euler="0 0 1.0471975512" size="0.155 0.090 0.04" material="drone_mat" mass="0.50"/>')
    s.append('<geom name="drone_panel_3" type="box" pos="0 0 0" euler="0 0 -1.0471975512" size="0.155 0.090 0.04" material="drone_mat" mass="0.50"/>')
    rotor_xy = ((0.34, 0), (0.17, 0.294), (-0.17, 0.294), (-0.34, 0), (-0.17, -0.294), (0.17, -0.294))
    for i, (x, y) in enumerate(rotor_xy, start=1):
        s.append('<geom name="rotor%d_arm" type="capsule" fromto="0 0 0 %.3f %.3f 0" size="0.015" material="drone_mat" mass="0.03"/>' % (i, x, y))
    for i, (x, y) in enumerate(rotor_xy, start=1):
        s.append('<body name="rotor%d_body" pos="%.3f %.3f 0">' % (i, x, y))
        s.append('<site name="rotor%d_site" pos="0 0 0" size="0.01"/>' % i)
        s.append('<geom type="cylinder" size="0.07 0.005" rgba="0.08 0.08 0.08 1" mass="0.01"/>')
        s.append('</body>')

    # ---------------- 5 级伸缩 + 软臂 ----------------
    s.append('<body name="arm_base" pos="0 0 -0.07">')
    s.append('<geom name="arm_base_geom" type="cylinder" pos="0 0 -0.06" size="%.4f 0.10" contype="0" conaffinity="0" mass="0.08" material="tube_mat"/>' % HOUSING_R)
    z_attach = -0.022
    for i in range(1, N_TELE + 1):
        r = 0.022 - 0.003 * (i - 1)
        s.append('<body name="tele%d" pos="0 0 %.4f">' % (i, z_attach))
        s.append('<joint name="tele_slide%d" type="slide" axis="0 0 -1" range="0 %.4f" limited="true" '
                 'stiffness="0" damping="0.02"/>' % (i, TELE_TRAVEL))
        s.append('<geom name="tele%d_geom" type="cylinder" pos="0 0 %.4f" size="%.4f %.4f" '
                 'contype="0" conaffinity="0" mass="0.03" material="tube_mat"/>' % (i, TUBE_C, r, TUBE_H))
        z_attach = -TELE_SPACING
    tube_bottom = TUBE_C - TUBE_H
    s.append('<site name="tendon_base" pos="0 0 %.4f" size="0.002"/>' % tube_bottom)
    s.append('<body name="seg_1" pos="0 0 %.4f">' % tube_bottom)
    for i in range(1, N_SEG + 1):
        r, L = segs[i - 1]
        mat = ("seg_mat0" if i <= 4 else ("seg_mat1" if i <= 8 else "seg_mat2"))
        if i > 1:
            s.append('<body name="seg_%d" pos="0 0 %.4f">' % (i, -L))
        s.append('<joint name="bx_%d" type="hinge" axis="1 0 0" pos="0 0 0" range="-1.0 1.0" armature="0.002" '
                 'limited="true" stiffness="%.4f" damping="%.4f"/>' % (i, K_SOFT, C_SOFT))
        s.append('<joint name="by_%d" type="hinge" axis="0 1 0" pos="0 0 0" range="-1.0 1.0" armature="0.002" '
                 'limited="true" stiffness="%.4f" damping="%.4f"/>' % (i, K_SOFT, C_SOFT))
        s.append('<geom name="seg_%d_geom" type="capsule" fromto="0 0 0 0 0 %.4f" size="%.4f" '
                 'material="%s" mass="0.010"/>' % (i, -L, r, mat))
        for k, ang in enumerate(TENDON_ANGLES):
            a = np.radians(ang)
            s.append('<site name="ts_%d_%d" pos="%.4f %.4f %.4f" size="0.0015"/>' %
                     (i, k, TENDON_R * np.cos(a), TENDON_R * np.sin(a), -L / 2))
    # ---------------- 两指平行夹板（纯水平闭合） ----------------
    s.append('<body name="palm" pos="0 0 %.4f">' % (-SEG_L))
    s.append('<geom name="palm_geom" type="box" size="0.020 0.016 0.009" mass="0.02" material="mat_grip"/>')
    s.append('<body name="plate_l" pos="-0.013 0 -0.024">')
    s.append('<joint name="grip_l" type="slide" axis="1 0 0" pos="0 0 0" range="0 %.4f" armature="0.0005" '
             'limited="true" stiffness="0" damping="0.05"/>' % GRIP_CLOSE)
    s.append('<geom name="plate_l_geom" type="box" pos="0 0 0" size="0.003 0.015 0.015" mass="0.006" '
             'friction="4 0.01 0.001" condim="6" solimp="0.95 0.99 0.001" solref="0.002 1" material="mat_grip2"/>')
    s.append('</body>')
    s.append('<body name="plate_r" pos="0.013 0 -0.024">')
    s.append('<joint name="grip_r" type="slide" axis="-1 0 0" pos="0 0 0" range="0 %.4f" armature="0.0005" '
             'limited="true" stiffness="0" damping="0.05"/>' % GRIP_CLOSE)
    s.append('<geom name="plate_r_geom" type="box" pos="0 0 0" size="0.003 0.015 0.015" mass="0.006" '
             'friction="4 0.01 0.001" condim="6" solimp="0.95 0.99 0.001" solref="0.002 1" material="mat_grip2"/>')
    s.append('</body>')
    if wrist_cam:
        s.append('<camera name="wrist" pos="0 0 -0.012" quat="1 0 0 0" fovy="55"/>')
    s.append('<site name="tip_site" pos="0 0 -0.008" size="0.004"/>')
    s.append('</body>')   # palm
    for i in range(N_SEG, 0, -1):
        s.append('</body>')
    for i in range(N_TELE):
        s.append('</body>')
    s.append('</body>')   # arm_base
    s.append('</body>')   # drone
    s.append('</worldbody>')

    s.append('<tendon>')
    for k in range(3):
        s.append('<spatial name="tendon_%d" width="0.001" frictionloss="0.02" rgba="0.9 0.3 0.3 1">' % k)
        s.append('<site site="tendon_base"/>')
        for i in range(1, N_SEG + 1):
            s.append('<site site="ts_%d_%d"/>' % (i, k))
        s.append('<site site="tip_site"/>')
        s.append('</spatial>')
    s.append('</tendon>')

    s.append('<actuator>')
    for i in range(1, 7):
        sign = 0.02 if i % 2 == 1 else -0.02
        s.append('<motor name="rotor%d" site="rotor%d_site" gear="0 0 1 0 0 %.3f" ctrlrange="0 12"/>' % (i, i, sign))
    for k in range(3):
        s.append('<muscle name="tendon_m%d" tendon="tendon_%d" ctrlrange="0 1" force="%.1f" lengthrange="0.4 0.8"/>'
                 % (k, k, TENDON_FORCE))
    for i in range(1, N_TELE + 1):
        s.append('<position name="tele_pos%d" joint="tele_slide%d" kp="2000" kv="60" ctrlrange="0 %.4f"/>' % (i, i, TELE_TRAVEL))
    s.append('<position name="grip_l_pos" joint="grip_l" kp="800" kv="2.0" ctrlrange="0 %.4f"/>' % GRIP_CLOSE)
    s.append('<position name="grip_r_pos" joint="grip_r" kp="800" kv="2.0" ctrlrange="0 %.4f"/>' % GRIP_CLOSE)
    s.append('</actuator>')
    s.append('</mujoco>')
    path.write_text("\n".join(s), encoding="utf-8")
    return path


class Hover:
    def __init__(self, model, data, target=(0.0, 0.0, DRONE_Z)):
        self.m, self.d = model, data
        self.target = np.array(target, dtype=float)
        self.rotor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotor%d" % i) for i in range(1, 7)]
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.free_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_free")])
        A = []
        for i, (x, y) in enumerate(((0.34, 0), (0.17, 0.294), (-0.17, 0.294),
                                    (-0.34, 0), (-0.17, -0.294), (0.17, -0.294)), start=1):
            sign = 0.02 if i % 2 == 1 else -0.02
            A.append([1.0, y, -x, sign])
        self.A_pinv = np.linalg.pinv(np.array(A).T)
        self.integ = np.zeros(3)
        self.dt = model.opt.timestep
        # 只统计无人机子树质量（立架等静态物体不计入）
        mass = 0.0
        for b in range(model.nbody):
            if b == 0:
                continue
            j = b
            while j != 0 and j != self.body:
                j = model.body_parentid[j]
            if j == self.body:
                mass += float(model.body_mass[b])
        self.mass = mass

    def __call__(self):
        d = self.d
        p = d.xpos[self.body].copy()
        R = d.xmat[self.body].reshape(3, 3)
        fd = self.free_dof
        v = d.qvel[fd:fd + 3].copy()
        omega = d.qvel[fd + 3:fd + 6].copy()
        self.integ = np.clip(self.integ + (self.target - p) * self.dt, -0.5, 0.5)
        a_des = 6.0 * (self.target - p) + 1.5 * self.integ - 3.0 * v
        a_des[2] += 9.81
        F_world = self.mass * a_des
        z_b = R[:, 2]
        thrust = max(float(F_world @ z_b), 0.1)
        z_d = F_world / max(np.linalg.norm(F_world), 1e-6)
        err_b = R.T @ np.cross(z_b, z_d)
        tau = 10.0 * err_b - 0.8 * omega
        tau[2] = -0.2 * omega[2]
        f = np.clip(self.A_pinv @ np.array([thrust, tau[0], tau[1], tau[2]]), 0.0, 12.0)
        for j, aid in enumerate(self.rotor_ids):
            self.d.ctrl[aid] = f[j]


def set_locks(model, locks):
    for (i0, i1), lk in zip(SECTIONS, locks):
        k = K_SOFT + (K_LOCK - K_SOFT) * lk
        c = C_SOFT + (C_LOCK - C_SOFT) * lk
        for i in range(i0, i1 + 1):
            for pre in ("bx_", "by_"):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + str(i))
                model.jnt_stiffness[jid] = k
                model.dof_damping[model.jnt_dofadr[jid]] = c


def apply_ctrl(model, data, hover, tendons, locks, tele, grip):
    set_locks(model, locks)
    hover()
    for k in range(3):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tendon_m%d" % k)] = tendons[k]
    for i in range(1, N_TELE + 1):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tele_pos%d" % i)] = tele * TELE_TRAVEL
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_l_pos")] = GRIP_CLOSE * grip
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_r_pos")] = GRIP_CLOSE * grip


def grasp_metrics(model, data):
    bar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dumbbell")
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    pl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_l")
    pr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_r")
    jaw = (data.xpos[pl] + data.xpos[pr]) / 2
    d_bar = np.linalg.norm(data.xpos[bar] - jaw)
    contacts = 0
    for i in range(data.ncon):
        b1 = model.geom_bodyid[data.contact[i].geom1]
        b2 = model.geom_bodyid[data.contact[i].geom2]
        if bar in (b1, b2) and (pl in (b1, b2) or pr in (b1, b2)):
            contacts += 1
    return data.xpos[bar][2], d_bar, contacts, jaw


def ramp(t, t0, t1, a, b):
    u = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * u))


class ScriptedGrasp:
    """写死版状态机：按夹口高度闭环下探，掌部不压杆。"""

    JAW_TARGET = BAR_Z + 0.024
    TELE_MAX = 0.5

    def __init__(self):
        self.state = "settle"
        self.t = 0.0
        self.tele = 0.0
        self.grip = 0.0
        self.grasp_tele = 0.0
        self.contact_tele = None

    def _bar_palm_contact(self, model, data):
        bar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dumbbell")
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        for i in range(data.ncon):
            b1 = model.geom_bodyid[data.contact[i].geom1]
            b2 = model.geom_bodyid[data.contact[i].geom2]
            if (b1 == bar and b2 == palm) or (b2 == bar and b1 == palm):
                return True
        return False

    def update(self, dt, model, data):
        self.t += dt
        palm_z = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")][2])
        if self.state == "settle":
            if self.t >= 3.0:
                self.state, self.t = "approach", 0.0
        elif self.state == "approach":
            # 缓慢下探，直到掌面到达杆上方目标高度
            self.tele = min(self.tele + 0.012 * dt, self.TELE_MAX)
            if palm_z <= self.JAW_TARGET or self.t > 30.0:
                self.grasp_tele = self.tele
                self.state, self.t = "hold", 0.0
        elif self.state == "hold":
            if self.t >= 1.0:
                self.state, self.t = "close", 0.0
        elif self.state == "close":
            self.grip = ramp(self.t, 0.0, 2.0, 0.0, 1.0)
            if self.t >= 2.0:
                self.state, self.t = "verify", 0.0
        elif self.state == "verify":
            self.grip = 1.0
            if self.t >= 1.5:
                self.state, self.t = "lift", 0.0
        elif self.state == "lift":
            self.grip = 1.0
            self.tele = ramp(self.t, 0.0, 5.0, self.grasp_tele, LIFT_TELE)
            if self.t >= 5.0:
                self.state, self.t = "hold_high", 0.0
        elif self.state == "hold_high":
            self.grip = 1.0
            if self.t >= 4.0:
                self.state, self.t = "lower", 0.0
        elif self.state == "lower":
            self.grip = 1.0
            self.tele = ramp(self.t, 0.0, 5.0, LIFT_TELE, self.grasp_tele)
            if self.t >= 5.0:
                self.state, self.t = "open", 0.0
        elif self.state == "open":
            self.tele = self.grasp_tele
            self.grip = ramp(self.t, 0.0, 1.5, 1.0, 0.0)
            if self.t >= 1.5:
                self.state, self.t = "retract", 0.0
        else:
            self.tele = ramp(self.t, 0.0, 3.0, self.grasp_tele, 0.0)
        phase = {"settle": "悬停就位", "approach": "伸缩下探", "backoff": "回退对准", "hold": "对准保持",
                 "close": "夹爪闭合", "verify": "夹持确认", "lift": "缩回提升", "hold_high": "高位保持",
                 "lower": "下降放置", "open": "张开释放", "retract": "收拢复位"}[self.state]
        return self.tele, self.grip, phase

    @property
    def done(self):
        return self.state == "retract" and self.t > 3.0


def build(wrist_cam=False):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    xml = RUN_DIR / ("_aerial_grasp_cam.xml" if wrist_cam else "_aerial_grasp.xml")
    gen_xml(xml, wrist_cam=wrist_cam)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml


def run_selftest():
    model, data, _ = build()
    dt = model.opt.timestep
    hover = Hover(model, data)
    ctrl = ScriptedGrasp()
    n = int(45.0 / dt)
    next_t = 0.0
    for k in range(n):
        t = (k + 1) * dt
        tele, grip, phase = ctrl.update(dt, model, data)
        apply_ctrl(model, data, hover, [0, 0, 0], (0, 0, 0), tele / TELE_TRAVEL, grip)
        mujoco.mj_step(model, data)
        if t >= next_t:
            next_t += 3.0
            bz, dbar, cf, jaw = grasp_metrics(model, data)
            print("t=%5.1f %-10s 伸缩=%5.3fm 夹爪=%.2f | 杆 z=%.3f 夹口距=%.3f 接触=%d | 无人机 z=%.2f" %
                  (t, phase, ctrl.tele, grip, bz, dbar, cf, data.xpos[hover.body][2]))
        if ctrl.done:
            break
    bz, dbar, cf, jaw = grasp_metrics(model, data)
    print("最终 杆 z=%.3f 接触=%d" % (bz, cf))
    return 0


def run_render():
    W, H, FPS = 1920, 1080, 50
    SPF = 20
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, data, _ = build()
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    renderer = mujoco.Renderer(model, height=H, width=W)
    hover = Hover(model, data)
    ctrl = ScriptedGrasp()
    cam_a, cam_b = mujoco.MjvCamera(), mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam_a)
    mujoco.mjv_defaultCamera(cam_b)
    cam_a.distance, cam_a.azimuth, cam_a.elevation = 2.9, 136.0, -8.0
    cam_a.lookat[:] = (0.0, 0.0, 1.30)
    font_big = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
    font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    video = OUT_DIR / "aerial_grasp_hd.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17",
         "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
        stdin=subprocess.PIPE)
    dt = model.opt.timestep
    t0 = time.monotonic()
    frame = 0
    w_blend = 0.0
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - 0.17 * np.clip(rr - 0.52, 0, 1) / 0.48, 0, 1).astype(np.float32)
    max_steps = int(45.0 / dt)
    for k in range(max_steps):
        t = (k + 1) * dt
        tele, grip, phase = ctrl.update(dt, model, data)
        apply_ctrl(model, data, hover, [0, 0, 0], (0, 0, 0), tele / TELE_TRAVEL, grip)
        mujoco.mj_step(model, data)
        if ctrl.done:
            break
        if k % SPF == 0:
            bar_z, dbar, cf, jaw = grasp_metrics(model, data)
            target_w = 1.0 if ctrl.state in ("hold", "close", "verify") else 0.0
            w_blend += (target_w - w_blend) * (SPF * dt) / 0.3
            cam_b.distance = 0.45
            cam_b.azimuth = 145.0
            cam_b.elevation = -22.0
            cam_b.lookat[:] = (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dumbbell")]
                               + data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")]) / 2
            renderer.update_scene(data, cam_a)
            img_a = renderer.render().copy()
            renderer.update_scene(data, cam_b)
            img_b = renderer.render().copy()
            img = (img_a.astype(np.float32) * (1 - w_blend) + img_b.astype(np.float32) * w_blend).astype(np.uint8)
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im, "RGBA")
            l1 = "空中抓取闭环 · %s · t=%.1f s" % (phase, t)
            l2 = "伸缩=%.3f m · 夹爪=%.2f · 哑铃杆高度 z=%.3f m · 夹爪-杆接触 n=%d" % (tele, grip, bar_z, cf)
            l3 = "无人机 z=%.2f m · 臂端高度 %.3f m · 目标在立架顶 %.3f m" % (
                data.xpos[hover.body][2], jaw[2], BAR_Z)
            pad, lh = 18, 40
            wpx = max(dr.textlength(s2, font=f) for s2, f in ((l1, font_big), (l2, font_mid), (l3, font_mid)))
            bw = int(wpx) + 2 * pad
            bh = 3 * lh + 2 * pad - 6
            x0, y0 = 26, H - 26 - bh
            dr.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=14,
                                 fill=(10, 12, 17, 168), outline=(255, 255, 255, 36), width=1)
            dr.text((x0 + pad, y0 + pad - 4), l1, font=font_big, fill=(120, 210, 255, 255))
            dr.text((x0 + pad, y0 + pad + lh - 4), l2, font=font_mid, fill=(235, 238, 242, 255))
            dr.text((x0 + pad, y0 + pad + 2 * lh - 4), l3, font=font_mid, fill=(178, 192, 205, 255))
            img[:] = (np.asarray(im, dtype=np.float32) * vig[..., None]).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            frame += 1
            if frame % 150 == 0:
                print("  t=%.1f %s" % (t, phase))
    proc.stdin.close()
    proc.wait()
    renderer.close()
    print("wrote %s (%.1f MB, %.1f s, %d frames)" % (video, video.stat().st_size / 1e6, frame / FPS, frame))
    print("wall %.0fs" % (time.monotonic() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        return run_render()
    return run_selftest()


if __name__ == "__main__":
    sys.exit(main())
