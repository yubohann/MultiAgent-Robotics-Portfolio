"""气动-腱驱空间弯曲、分段阻塞锁定、5 倍伸缩软臂 + 六旋翼（MuJoCo）。

v2 相对 v1 的三处升级：
  1. 空间弯曲：每节 2 个正交 hinge（x/y 轴），周向 3 条腱按 120 度布置，PCC 分段近似。
  2. 分段阻塞锁定：软臂分 3 段，每段可独立切换柔性/刚性（阻塞锁定近似），
     并做末端载荷-挠度实验作为证据。
  3. 5 级伸缩：5 级嵌套滑管，模块 0.16 m 展开到 0.76 m（约 5 倍），
     并测全展状态下的末端承载（挠度-载荷曲线）。

运行：
  python soft_arm_v2.py --selftest     无界面自检
  python soft_arm_v2.py --evidence     载荷-挠度与伸展承载实验（存 PNG/CSV）
  python soft_arm_v2.py --render       渲染 HD 视频
"""
import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUN_DIR = Path(tempfile.gettempdir()) / "soft_arm_v2"
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
SECTIONS = ((1, 4), (5, 8), (9, 12))     # 三个可锁定段
TIP_LOAD_MAX = 4.0


def gen_xml(path):
    segs = []
    for i in range(N_SEG):
        r = SEG_R0 + (SEG_R1 - SEG_R0) * i / (N_SEG - 1)
        segs.append((r, SEG_L))
    s = []
    s.append('<mujoco model="soft_arm_v2">')
    s.append('<compiler angle="radian"/>')
    s.append('<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>')
    s.append("""
  <asset>
    <material name="drone_mat" rgba="0.12 0.16 0.20 1"/>
    <material name="tube_mat" rgba="0.35 0.40 0.48 1"/>
    <material name="seg_mat0" rgba="0.16 0.32 0.62 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="seg_mat1" rgba="0.25 0.62 0.85 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="seg_mat2" rgba="0.85 0.45 0.15 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="tip_mat" rgba="0.95 0.85 0.15 1" specular="0.7" shininess="0.6" roughness="0.3"/>
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
    s.append('<light name="key" directional="true" pos="0.9 -0.6 2.4" dir="-0.35 0.25 -1" '
             'diffuse="0.80 0.78 0.74" specular="0.55 0.55 0.55" castshadow="true"/>')
    s.append('<light name="fill" directional="true" pos="-1.2 0.6 1.6" dir="0.6 -0.3 -1" '
             'diffuse="0.25 0.28 0.34" specular="0.05 0.05 0.05"/>')

    # ---------------- 六旋翼 ----------------
    s.append('<body name="drone" pos="0 0 1.95">')
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
    s.append('<geom name="arm_base_geom" type="cylinder" pos="0 0 -0.06" size="%.4f 0.10" material="tube_mat" mass="0.08"/>' % HOUSING_R)
    z_attach = -0.022
    for i in range(1, N_TELE + 1):
        r = 0.022 - 0.003 * (i - 1)
        s.append('<body name="tele%d" pos="0 0 %.4f">' % (i, z_attach))
        s.append('<joint name="tele_slide%d" type="slide" axis="0 0 -1" range="0 %.4f" limited="true" '
                 'stiffness="0" damping="0.02"/>' % (i, TELE_TRAVEL))
        s.append('<geom name="tele%d_geom" type="cylinder" pos="0 0 %.4f" size="%.4f %.4f" '
                 'material="tube_mat" mass="0.03"/>' % (i, TUBE_C, r, TUBE_H))
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
    s.append('<geom name="tip_geom" type="sphere" size="0.016" pos="0 0 %.4f" material="tip_mat" mass="0.02"/>' % (-SEG_L))
    s.append('<site name="tip_site" pos="0 0 %.4f" size="0.004"/>' % (-SEG_L))
    for i in range(N_SEG, 0, -1):
        s.append('</body>')
    for i in range(N_TELE):
        s.append('</body>')
    s.append('</body>')   # arm_base
    s.append('</body>')   # drone
    s.append('</worldbody>')

    # ---------------- 3 条空间腱 ----------------
    s.append('<tendon>')
    for k in range(3):
        s.append('<spatial name="tendon_%d" width="0.001" frictionloss="0.02" rgba="0.9 0.3 0.3 1">' % k)
        s.append('<site site="tendon_base"/>')
        for i in range(1, N_SEG + 1):
            s.append('<site site="ts_%d_%d"/>' % (i, k))
        s.append('<site site="tip_site"/>')
        s.append('</spatial>')
    s.append('</tendon>')

    # ---------------- 执行器 ----------------
    s.append('<actuator>')
    for i in range(1, 7):
        sign = 0.02 if i % 2 == 1 else -0.02
        s.append('<motor name="rotor%d" site="rotor%d_site" gear="0 0 1 0 0 %.3f" ctrlrange="0 12"/>' % (i, i, sign))
    for k in range(3):
        s.append('<muscle name="tendon_m%d" tendon="tendon_%d" ctrlrange="0 1" force="%.1f" lengthrange="0.4 0.8"/>'
                 % (k, k, TENDON_FORCE))
    for i in range(1, N_TELE + 1):
        s.append('<position name="tele_pos%d" joint="tele_slide%d" kp="400" kv="40" ctrlrange="0 %.4f"/>' % (i, i, TELE_TRAVEL))
    s.append('</actuator>')
    s.append('</mujoco>')
    path.write_text("\n".join(s), encoding="utf-8")
    return path


class Hover:
    """六旋翼位置+姿态 PD，最小二乘推力分配。"""

    def __init__(self, model, data, target=(0.0, 0.0, 1.95)):
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
        self.mass = float(np.sum(model.body_mass))

    def __call__(self):
        d = self.d
        p = d.xpos[self.body].copy()
        R = d.xmat[self.body].reshape(3, 3)
        fd = self.free_dof
        v = d.qvel[fd:fd + 3].copy()
        omega = d.qvel[fd + 3:fd + 6].copy()
        a_des = 6.0 * (self.target - p) - 3.0 * v
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
    """locks = (a, b, c) 每段 0/1。"""
    for (i0, i1), lk in zip(SECTIONS, locks):
        k = K_SOFT + (K_LOCK - K_SOFT) * lk
        c = C_SOFT + (C_LOCK - C_SOFT) * lk
        for i in range(i0, i1 + 1):
            for pre in ("bx_", "by_"):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + str(i))
                model.jnt_stiffness[jid] = k
                model.dof_damping[model.jnt_dofadr[jid]] = c


def set_ctrl(model, data, hover, tendons, locks, tele, tip_load):
    set_locks(model, locks)
    hover()
    for k in range(3):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tendon_m%d" % k)
        data.ctrl[aid] = tendons[k]
    for i in range(1, N_TELE + 1):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tele_pos%d" % i)
        data.ctrl[aid] = tele * TELE_TRAVEL
    # 末端世界系横向载荷（+x 方向）
    data.xfrc_applied[:] = 0.0
    if tip_load > 0.0:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg_%d" % N_SEG)
        data.xfrc_applied[body, 0] = tip_load


def tip_metrics(model, data):
    """末端相对基座的挠度，在无人机机体系下测量（剔除机身姿态）。"""
    mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "manipulator_mount")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
    drone = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    R = data.xmat[drone].reshape(3, 3)
    vec = R.T @ (data.site_xpos[tip] - data.site_xpos[mount])
    lateral = float(np.linalg.norm(vec[:2]))
    length = float(np.linalg.norm(vec))
    ang = float(np.degrees(np.arctan2(lateral, -vec[2])))
    return length, lateral, ang, data.site_xpos[tip].copy()


def ramp(t, t0, t1, a, b):
    u = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * u))


def timeline(t):
    """返回 (tendons(3), locks(3), tele, tip_load, phase)"""
    td = [0.0, 0.0, 0.0]
    lk = (0, 0, 0)
    tele, load = 0.0, 0.0
    if t < 2.0:
        phase = "悬停待命"
    elif t < 13.0:
        phase = "空间弯曲扫描（三向）"
        seg_t = (t - 2.0) % 3.6
        idx = int((t - 2.0) // 3.6) % 3
        td[idx] = ramp(seg_t, 0.0, 1.6, 0.0, 1.0) if seg_t < 2.4 else 1.0
        if seg_t >= 3.0:
            td[idx] = ramp(seg_t, 3.0, 3.6, 1.0, 0.0)
    elif t < 22.0:
        phase = "分段阻塞锁定"
        td[0] = ramp(t, 13.0, 15.0, 0.0, 1.0) if t < 15.0 else 1.0
        if t < 16.5:
            lk = (0, 0, 0)
        elif t < 19.0:
            lk = (1, 1, 1)
        else:
            lk = (0, 0, 0)
    elif t < 30.0:
        phase = "末端载荷-挠度"
        load = ramp(t, 22.0, 24.0, 0.0, 3.0) if t < 24.0 else 3.0
        if t < 26.5:
            lk = (0, 0, 0)
        elif t < 28.5:
            lk = (1, 1, 1)
        else:
            lk = (0, 0, 0)
    elif t < 38.0:
        phase = "5 倍伸缩展开"
        tele = ramp(t, 30.0, 34.5, 0.0, 1.0) if t < 34.5 else 1.0
        if t >= 35.5:
            load = ramp(t, 35.5, 36.5, 0.0, 2.0)
    else:
        phase = "收拢复位"
        tele = ramp(t, 38.0, 40.5, 1.0, 0.0)
        load = ramp(t, 38.0, 39.0, 2.0, 0.0)
    return td, lk, tele, load, phase


T_END = 41.5


def build():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    xml = RUN_DIR / "_soft_arm_v2.xml"
    gen_xml(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml


def run_selftest():
    model, data, _ = build()
    dt = model.opt.timestep
    hover = Hover(model, data)
    n = int(T_END / dt)
    for k in range(n):
        t = (k + 1) * dt
        td, lk, tele, load, phase = timeline(t)
        set_ctrl(model, data, hover, td, lk, tele, load)
        mujoco.mj_step(model, data)
        if k % int(3.0 / dt) == 0:
            L, lat, ang, pt = tip_metrics(model, data)
            print("t=%5.1f %-18s 腱=(%.1f %.1f %.1f) 锁=%s 伸缩=%2.0f%% 载荷=%.1fN | 臂长 %.2fm 横偏 %.3fm 偏角 %4.1f° | z=%.2f" %
                  (t, phase, td[0], td[1], td[2], lk, tele * 100, load, L, lat, ang, data.xpos[hover.body][2]))
    L, lat, ang, pt = tip_metrics(model, data)
    ok = 1.4 < data.xpos[hover.body][2] < 2.4
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_evidence():
    """载荷-挠度曲线（4 种锁定构型）+ 伸展承载。输出 PNG/CSV。"""
    model, data, _ = build()
    dt = model.opt.timestep
    hover = Hover(model, data)
    loads = [0.0, 0.5, 1.0, 2.0, 3.0]
    configs = [("全柔性", (0, 0, 0)), ("锁定段A", (1, 0, 0)), ("锁定A+B", (1, 1, 0)), ("全锁定", (1, 1, 1))]
    rows = []

    def settle(lk, tele, load, t_settle=10.0, t_hold=4.0):
        data.qpos[:] = 0
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        n1 = int(t_settle / dt)
        for k in range(n1):
            set_ctrl(model, data, hover, [0.0, 0.0, 0.0], lk, tele, load * min(k / (n1 * 0.4), 1.0))
            mujoco.mj_step(model, data)
        n2 = int(t_hold / dt)
        lat_acc = 0.0
        for k in range(n2):
            set_ctrl(model, data, hover, [0.0, 0.0, 0.0], lk, tele, load)
            mujoco.mj_step(model, data)
            L, lat, ang, _ = tip_metrics(model, data)
            lat_acc += lat
        lat = lat_acc / n2
        L, _, ang, _ = tip_metrics(model, data)
        return lat, ang, L

    for name, lk in configs:
        for load in loads:
            lat, ang, L = settle(lk, 0.0, load)
            rows.append((name, load, lat * 1000, ang, L))
            print("%-8s load=%.1fN 横偏=%6.1fmm 偏角=%5.1f° 臂长=%.3fm" % (name, load, lat * 1000, ang, L))
    csv_path = HERE / "load_deflection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "load_N", "lateral_mm", "angle_deg", "length_m"])
        w.writerows(rows)

    # 伸展承载（全展，全锁定保持刚性）
    ext_rows = []
    for load in [0.0, 0.5, 1.0, 2.0, 3.0]:
        lat, ang, L = settle((1, 1, 1), 1.0, load)
        ext = sum(float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tele_slide%d" % i)]])
                  for i in range(1, N_TELE + 1))
        ext_rows.append((load, lat * 1000, ext * 1000, L))
        print("全展 load=%.1fN 横偏=%6.1fmm 滑管伸展=%5.1fmm 臂长=%.3fm" % (load, lat * 1000, ext * 1000, L))
    csv2 = HERE / "extension_load.csv"
    with csv2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["load_N", "lateral_mm", "extension_mm", "length_m"])
        w.writerows(ext_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
        for name, lk in configs:
            xs = [r[1] for r in rows if r[0] == name]
            ys = [r[2] for r in rows if r[0] == name]
            axes[0].plot(xs, ys, marker="o", label=name)
        axes[0].set_xlabel("tip load (N)")
        axes[0].set_ylabel("lateral deflection (mm)")
        axes[0].set_title("load-deflection vs section locking")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        xs = [r[0] for r in ext_rows]
        ys = [r[1] for r in ext_rows]
        ys2 = [r[2] for r in ext_rows]
        axes[1].plot(xs, ys, marker="o", label="lateral deflection")
        axes[1].plot(xs, ys2, marker="s", label="slide retract")
        axes[1].set_xlabel("tip load (N)")
        axes[1].set_ylabel("mm")
        axes[1].set_title("fully extended arm under load")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        png = HERE / "load_deflection.png"
        fig.savefig(png)
        print("plot saved:", png)
    except Exception as e:
        print("matplotlib 不可用:", e)
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
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.9
    cam.azimuth = 138.0
    cam.elevation = -9.0
    cam.lookat[:] = (0.0, 0.0, 1.15)
    font_big = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
    font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    video = OUT_DIR / "soft_arm_v2.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17",
         "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
        stdin=subprocess.PIPE)
    dt = model.opt.timestep
    total = int(T_END / dt)
    t0 = time.monotonic()
    frame = 0
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - 0.17 * np.clip(rr - 0.52, 0, 1) / 0.48, 0, 1).astype(np.float32)
    for k in range(total):
        t = (k + 1) * dt
        td, lk, tele, load, phase = timeline(t)
        set_ctrl(model, data, hover, td, lk, tele, load)
        mujoco.mj_step(model, data)
        if k % SPF == 0:
            L, lat, ang, pt = tip_metrics(model, data)
            renderer.update_scene(data, cam)
            img = renderer.render().copy()
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im, "RGBA")
            l1 = "%s · t=%.1f s" % (phase, t)
            l2 = "腱驱=(%.2f %.2f %.2f) · 分段锁定=%d%d%d · 伸缩=%3.0f%% · 末端载荷=%.1f N" % (
                td[0], td[1], td[2], lk[0], lk[1], lk[2], tele * 100, load)
            l3 = "臂长 %.2f m · 末端横偏 %.0f mm · 偏角 %.1f° · 无人机 z=%.2f m" % (
                L, lat * 1000, ang, data.xpos[hover.body][2])
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
    print("wrote %s (%.1f MB, %.1f s, %d frames)" %
          (video, video.stat().st_size / 1e6, T_END, frame))
    print("wall %.0fs" % (time.monotonic() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.evidence:
        return run_evidence()
    if args.render:
        return run_render()
    return run_selftest()


if __name__ == "__main__":
    sys.exit(main())
