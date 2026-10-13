"""气动-腱驱可变刚度伸缩软臂 + 六旋翼 demo（MuJoCo，基于 UAMSim 平台模型思路）。

自建臂特征：
  1. 分段关节（8 节，平面弯曲）
  2. 空间腱（2 条，muscle 执行器）
  3. 压力映射刚度（关节刚度/阻尼随气压 p 变化，近似充气刚化）
  4. 滑动伸缩段（3 级嵌套滑管，行程 3x）

运行：
  python soft_arm_demo.py --selftest     无界面自检（打印悬停与臂状态）
  python soft_arm_demo.py --render       渲染 HD 视频
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
RUN_DIR = Path(tempfile.gettempdir()) / "soft_arm_demo"   # MuJoCo 不支持中文路径
OUT_DIR = Path(r"C:\Users\Administrator\Desktop\aerial_soft_arm\videos\newmodel")

N_SEG = 8
SEG_L = 0.05
SEG_R0, SEG_R1 = 0.018, 0.010
N_TELE = 3
TELE_TRAVEL = 0.12
BASE_TUBE = 0.12

K_SOFT, K_STIFF = 0.5, 6.0
C_SOFT, C_STIFF = 0.02, 0.3
TENDON_FORCE = 25.0

DRONE_MASS_HINT = 2.0


def gen_xml(path):
    segs = []
    for i in range(N_SEG):
        r = SEG_R0 + (SEG_R1 - SEG_R0) * i / (N_SEG - 1)
        segs.append((r, SEG_L))
    s = []
    s.append('<mujoco model="soft_arm_demo">')
    s.append('<compiler angle="radian"/>')
    s.append('<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>')
    s.append("""
  <asset>
    <material name="drone_mat" rgba="0.12 0.16 0.20 1"/>
    <material name="tube_mat" rgba="0.35 0.40 0.48 1"/>
    <material name="seg_mat0" rgba="0.16 0.32 0.62 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="seg_mat1" rgba="0.25 0.62 0.85 1" specular="0.6" shininess="0.5" roughness="0.35"/>
    <material name="tip_mat" rgba="0.95 0.55 0.15 1" specular="0.7" shininess="0.6" roughness="0.3"/>
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

    # ---------------- 六旋翼（UAMSim 平台模型精简版） ----------------
    s.append('<body name="drone" pos="0 0 1.7">')
    s.append('<freejoint name="drone_free"/>')
    s.append('<site name="imu_site" pos="0 0 0" size="0.008"/>')
    s.append('<site name="manipulator_mount" pos="0 0 -0.05" size="0.01"/>')
    s.append('<geom name="drone_panel_1" type="box" pos="0 0 0" size="0.155 0.090 0.04" material="drone_mat" mass="0.50"/>')
    s.append('<geom name="drone_panel_2" type="box" pos="0 0 0" euler="0 0 1.0471975512" size="0.155 0.090 0.04" material="drone_mat" mass="0.50"/>')
    s.append('<geom name="drone_panel_3" type="box" pos="0 0 0" euler="0 0 -1.0471975512" size="0.155 0.090 0.04" material="drone_mat" mass="0.50"/>')
    for name, (x, y) in (("rotor1_arm", (0.34, 0)), ("rotor2_arm", (0.17, 0.294)),
                         ("rotor3_arm", (-0.17, 0.294)), ("rotor4_arm", (-0.34, 0)),
                         ("rotor5_arm", (-0.17, -0.294)), ("rotor6_arm", (0.17, -0.294))):
        s.append('<geom name="%s" type="capsule" fromto="0 0 0 %.3f %.3f 0" size="0.015" material="drone_mat" mass="0.03"/>' % (name, x, y))
    for i, (x, y) in enumerate(((0.34, 0), (0.17, 0.294), (-0.17, 0.294),
                                (-0.34, 0), (-0.17, -0.294), (0.17, -0.294)), start=1):
        s.append('<body name="rotor%d_body" pos="%.3f %.3f 0">' % (i, x, y))
        s.append('<site name="rotor%d_site" pos="0 0 0" size="0.01"/>' % i)
        s.append('<geom type="cylinder" size="0.07 0.005" rgba="0.08 0.08 0.08 1" mass="0.01"/>')
        s.append('</body>')

    # ---------------- 软臂（挂在 mount 下方） ----------------
    s.append('<body name="arm_base" pos="0 0 -0.07">')
    s.append('<geom name="arm_base_geom" type="cylinder" pos="0 0 -0.06" size="0.024 0.10" material="tube_mat" mass="0.08"/>')
    # 伸缩段（3 级嵌套滑管，逐级套娃）
    z_attach = -0.022
    tube_c = -(TELE_TRAVEL + 0.01)          # 管中心相对本级原点
    tube_h = TELE_TRAVEL + 0.03             # 管半长
    tube_bottom = tube_c - tube_h           # 管口下端（弯曲段挂载点）
    for i in range(1, N_TELE + 1):
        r = 0.020 - 0.004 * (i - 1)
        s.append('<body name="tele%d" pos="0 0 %.4f">' % (i, z_attach))
        s.append('<joint name="tele_slide%d" type="slide" axis="0 0 -1" range="0 %.4f" limited="true" '
                 'stiffness="0" damping="0.02"/>' % (i, TELE_TRAVEL))
        s.append('<geom name="tele%d_geom" type="cylinder" pos="0 0 %.4f" size="%.4f %.4f" '
                 'material="tube_mat" mass="0.03"/>' % (i, tube_c, r, tube_h))
        z_attach = -TELE_TRAVEL - 0.02
    s.append('<site name="tendon_base" pos="0 0 %.4f" size="0.002"/>' % tube_bottom)
    # 弯曲段（8 节，挂在最内层管口下端）
    s.append('<body name="seg_1" pos="0 0 %.4f">' % tube_bottom)
    for i in range(1, N_SEG + 1):
        r, L = segs[i - 1]
        mat = "seg_mat0" if i <= N_SEG // 2 else "seg_mat1"
        if i > 1:
            s.append('<body name="seg_%d" pos="0 0 %.4f">' % (i, -L))
        s.append('<joint name="bend_%d" type="hinge" axis="0 1 0" pos="0 0 0" range="-1.2 1.2" armature="0.001" '
                 'limited="true" stiffness="%.4f" damping="%.4f"/>' % (i, K_SOFT, C_SOFT))
        s.append('<geom name="seg_%d_geom" type="capsule" fromto="0 0 0 0 0 %.4f" size="%.4f" '
                 'material="%s" mass="0.012"/>' % (i, -L, r, mat))
        s.append('<site name="tsite_%d_p" pos="%.4f 0 %.4f" size="0.002"/>' % (i, r, -L / 2))
        s.append('<site name="tsite_%d_n" pos="%.4f 0 %.4f" size="0.002"/>' % (i, -r, -L / 2))
    # 末端
    s.append('<geom name="tip_geom" type="sphere" size="0.016" pos="0 0 -0.05" material="tip_mat" mass="0.02"/>')
    s.append('<site name="tip_site" pos="0 0 -0.05" size="0.004"/>')
    for i in range(N_SEG, 0, -1):
        s.append('</body>')
    for i in range(N_TELE):
        s.append('</body>')   # tele3 -> tele1
    s.append('</body>')   # arm_base 结束
    s.append('</body>')   # drone 结束
    s.append('</worldbody>')

    # ---------------- 腱（2 条空间腱，从伸缩末端到指尖） ----------------
    s.append('<tendon>')
    for name, side in (("t_p", "p"), ("t_n", "n")):
        s.append('<spatial name="%s" width="0.001" frictionloss="0.02" rgba="0.9 0.3 0.3 1">' % name)
        s.append('<site site="tendon_base"/>')
        for i in range(1, N_SEG + 1):
            s.append('<site site="tsite_%d_%s"/>' % (i, side))
        s.append('<site site="tip_site"/>')
        s.append('</spatial>')
    s.append('</tendon>')

    # ---------------- 执行器 ----------------
    s.append('<actuator>')
    for i in range(1, 7):
        sign = 0.02 if i % 2 == 1 else -0.02
        s.append('<motor name="rotor%d" site="rotor%d_site" gear="0 0 1 0 0 %.3f" ctrlrange="0 12"/>' % (i, i, sign))
    s.append('<muscle name="tendon_p" tendon="t_p" ctrlrange="0 1" force="%.1f" lengthrange="0.3 0.7"/>' % TENDON_FORCE)
    s.append('<muscle name="tendon_n" tendon="t_n" ctrlrange="0 1" force="%.1f" lengthrange="0.3 0.7"/>' % TENDON_FORCE)
    for i in range(1, N_TELE + 1):
        s.append('<position name="tele_pos%d" joint="tele_slide%d" kp="300" kv="30" ctrlrange="0 %.4f"/>' % (i, i, TELE_TRAVEL))
    s.append('</actuator>')
    s.append('</mujoco>')
    path.write_text("\n".join(s), encoding="utf-8")
    return path


class Hover:
    """六旋翼位置+姿态 PD，推力分配用最小二乘。"""

    def __init__(self, model, data, target=(0.0, 0.0, 1.85)):
        self.m = model
        self.d = data
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
        d, m = self.d, self.m
        b = self.body
        p = d.xpos[b].copy()
        R = d.xmat[b].reshape(3, 3)
        fd = self.free_dof
        v = d.qvel[fd:fd + 3].copy()
        omega = d.qvel[fd + 3:fd + 6].copy()
        e_p = self.target - p
        a_des = 6.0 * e_p - 3.0 * v
        a_des[2] += 9.81
        F_world = self.mass * a_des
        z_b = R[:, 2]
        thrust = max(float(F_world @ z_b), 0.1)
        z_d = F_world / max(np.linalg.norm(F_world), 1e-6)
        err_w = np.cross(z_b, z_d)
        err_b = R.T @ err_w
        tau = 6.0 * err_b - 0.5 * omega
        tau[2] = -0.15 * omega[2]
        f = self.A_pinv @ np.array([thrust, tau[0], tau[1], tau[2]])
        f = np.clip(f, 0.0, 12.0)
        for j, aid in enumerate(self.rotor_ids):
            self.d.ctrl[aid] = f[j]
        return thrust


def stiffness_profile(p):
    return K_SOFT + (K_STIFF - K_SOFT) * p, C_SOFT + (C_STIFF - C_SOFT) * p


def set_pressure(model, p):
    k, c = stiffness_profile(p)
    for i in range(1, N_SEG + 1):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bend_%d" % i)
        model.jnt_stiffness[jid] = k
        model.dof_damping[model.jnt_dofadr[jid]] = c


def timeline(t):
    """返回 (p, tendon_p, tendon_n, tele_frac, phase)"""
    def ramp(t0, t1, a, b):
        u = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
        return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * u))
    p, tp, tn, tele = 0.05, 0.0, 0.0, 0.0
    if t < 3.0:
        phase = "悬停待命（软态下垂）"
    elif t < 8.0:
        phase = "腱驱弯曲（软态）"
        tp = ramp(3.0, 7.0, 0.0, 1.0)
    elif t < 12.5:
        phase = "充气增刚（腱驱不变）"
        tp = 1.0
        p = ramp(8.0, 11.5, 0.05, 1.0)
    elif t < 16.5:
        phase = "放气变柔（腱驱不变）"
        tp = 1.0
        p = ramp(12.5, 15.5, 1.0, 0.05)
    elif t < 18.5:
        phase = "腱驱释放"
        tp = ramp(16.5, 18.0, 1.0, 0.0)
    elif t < 24.5:
        phase = "伸缩展开"
        tele = ramp(18.5, 23.0, 0.0, 1.0)
    elif t < 28.0:
        phase = "伸长保持"
        tele = 1.0
        tp = ramp(24.5, 27.0, 0.0, 0.6)
    else:
        phase = "收拢复位"
        tele = ramp(28.0, 30.5, 1.0, 0.0)
        tp = ramp(28.0, 30.5, 0.6, 0.0)
    return p, tp, tn, tele, phase


T_END = 31.5


def build():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    xml = RUN_DIR / "_soft_arm_demo.xml"
    gen_xml(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml


def tip_metrics(model, data):
    mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "manipulator_mount")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
    pm = data.site_xpos[mount].copy()
    pt = data.site_xpos[tip].copy()
    vec = pt - pm
    length = float(np.linalg.norm(vec))
    ang = float(np.degrees(np.arctan2(np.linalg.norm(vec[:2]), -vec[2])))
    return length, ang, pt


def run_selftest():
    model, data, _ = build()
    dt = model.opt.timestep
    hover = Hover(model, data)
    n = int(T_END / dt)
    print("质量 %.2f kg" % hover.mass)
    for k in range(n):
        t = (k + 1) * dt
        p, tp, tn, tele, phase = timeline(t)
        set_pressure(model, p)
        data.ctrl[hover.rotor_ids] = 0
        hover()
        tp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tendon_p")
        data.ctrl[tp_id] = tp
        for i in range(1, N_TELE + 1):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tele_pos%d" % i)
            data.ctrl[aid] = tele * TELE_TRAVEL
        mujoco.mj_step(model, data)
        if k % int(2.0 / dt) == 0:
            L, ang, pt = tip_metrics(model, data)
            print("t=%5.1f %-14s p=%.2f k=%.2f | 臂长 %.3fm 偏角 %4.1f° | 无人机 z=%.2f" %
                  (t, phase, p, stiffness_profile(p)[0], L, ang, data.xpos[hover.body][2]))
    L, ang, pt = tip_metrics(model, data)
    ok = 1.2 < data.xpos[hover.body][2] < 2.2
    print("SELFTEST", "PASS" if ok else "FAIL", "末端", np.round(pt, 3))
    return 0 if ok else 1


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
    cam.distance = 2.5
    cam.azimuth = 135.0
    cam.elevation = -8.0
    cam.lookat[:] = (0.0, 0.0, 1.25)
    font_big = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
    font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    video = OUT_DIR / "soft_arm_demo.mp4"
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
        p, tp, tn, tele, phase = timeline(t)
        set_pressure(model, p)
        hover()
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tendon_p")] = tp
        for i in range(1, N_TELE + 1):
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tele_pos%d" % i)] = tele * TELE_TRAVEL
        mujoco.mj_step(model, data)
        if k % SPF == 0:
            L, ang, pt = tip_metrics(model, data)
            renderer.update_scene(data, cam)
            img = renderer.render().copy()
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im, "RGBA")
            l1 = "%s · t=%.1f s" % (phase, t)
            l2 = "气压 p=%.2f · 关节刚度 k=%.2f Nm/rad · 腱驱 A=%.2f" % (p, stiffness_profile(p)[0], tp)
            l3 = "臂长 %.2f m · 末端偏角 %.1f° · 伸缩 %.0f%% · 无人机 z=%.2f m" % (
                L, ang, tele * 100, data.xpos[hover.body][2])
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
            if frame % 100 == 0:
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
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        return run_render()
    return run_selftest()


if __name__ == "__main__":
    sys.exit(main())
