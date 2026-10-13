"""高细节软臂装配与渲染（程序化网格 + 棚拍灯光）。

运行：
  python arm_real.py --stills   出三张静帧
  python arm_real.py --render   出 12 秒展示视频
"""
import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE / "meshes"
OUT_DIR = Path(r"C:\Users\Administrator\Desktop\aerial_soft_arm\videos\arm_real")

N_SEG = 8
SEG_L = 0.05
R_START, R_END = 0.0215, 0.0125
TENDON_FORCE = 22.0
K_SOFT, C_SOFT = 1.10, 0.10
K_STIFF, C_STIFF = 6.0, 0.5
TENDON_ANGLES = (90.0, 210.0, 330.0)
TABLE_TOP = 0.74
BASE_Z = 0.752

P_MAX = 2.0
P_STEP = 1.2
CV, CE, LEAK, DEAD = 2.2, 1.8, 0.015, 0.08
TAU_PUMP = 0.12


def gen_xml(path):
    s = []
    s.append('<mujoco model="soft_arm_real">')
    s.append('<compiler angle="radian" meshdir="meshes"/>')
    s.append('<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>')
    s.append("""
  <asset>
    <texture type="2d" name="tex_fabric" file="textures/fabric.png"/>
    <texture type="2d" name="tex_brushed" file="textures/brushed.png"/>
    <texture type="2d" name="tex_speckle" file="textures/speckle.png"/>
    <texture type="2d" name="tex_plate" file="textures/nameplate.png"/>
    <texture type="2d" name="tex_shadow" file="textures/shadow_soft.png"/>
    <material name="mat_silicone" texture="tex_fabric" rgba="0.90 0.88 0.84 1" roughness="0.68" metallic="0.0" specular="0.16" shininess="0.22"/>
    <material name="mat_silicone2" texture="tex_fabric" rgba="0.82 0.80 0.76 1" roughness="0.70" metallic="0.0" specular="0.15" shininess="0.21"/>
    <material name="mat_silicone_inf" texture="tex_fabric" rgba="0.93 0.91 0.87 1" roughness="0.66" metallic="0.0" specular="0.17" shininess="0.23"/>
    <material name="mat_rubber" texture="tex_speckle" rgba="0.18 0.18 0.20 1" roughness="0.90" metallic="0.0" specular="0.10" shininess="0.20"/>
    <material name="mat_aluminum" texture="tex_brushed" rgba="0.62 0.64 0.68 1" roughness="0.34" metallic="0.85" specular="0.45" shininess="0.50"/>
    <material name="mat_dark_alu" texture="tex_brushed" rgba="0.20 0.21 0.23 1" roughness="0.42" metallic="0.70" specular="0.35" shininess="0.40"/>
    <material name="mat_brass" rgba="0.78 0.62 0.30 1" roughness="0.30" metallic="0.90" specular="0.60" shininess="0.60"/>
    <material name="mat_pcb" rgba="0.06 0.22 0.12 1" roughness="0.50" metallic="0.0" specular="0.35" shininess="0.40"/>
    <material name="mat_orange" rgba="0.90 0.44 0.10 1" roughness="0.42" metallic="0.0" specular="0.45" shininess="0.50"/>
    <material name="mat_table" texture="tex_speckle" rgba="0.37 0.37 0.38 1" roughness="0.72" metallic="0.0" specular="0.25" shininess="0.30"/>
    <material name="mat_shadow1" rgba="0.0 0.0 0.0 0.30" roughness="0.9" metallic="0.0" specular="0.0" shininess="0.1"/>
    <material name="mat_shadow2" rgba="0.0 0.0 0.0 0.18" roughness="0.9" metallic="0.0" specular="0.0" shininess="0.1"/>
    <material name="mat_shadow3" rgba="0.0 0.0 0.0 0.08" roughness="0.9" metallic="0.0" specular="0.0" shininess="0.1"/>
    <material name="mat_floor" texture="tex_speckle" rgba="0.27 0.27 0.28 1" roughness="0.90" metallic="0.0" specular="0.10" shininess="0.20"/>
    <material name="mat_wall" rgba="0.20 0.21 0.22 1" roughness="0.95" metallic="0.0" specular="0.05" shininess="0.20"/>
    <material name="mat_steel" texture="tex_brushed" rgba="0.72 0.74 0.77 1" roughness="0.28" metallic="0.90" specular="0.55" shininess="0.55"/>
    <material name="mat_black_plast" rgba="0.11 0.11 0.12 1" roughness="0.55" metallic="0.0" specular="0.30" shininess="0.35"/>
    <material name="mat_plate" texture="tex_plate" rgba="1.0 1.0 1.0 1" roughness="0.55" metallic="0.15" specular="0.35" shininess="0.40"/>
    <texture type="skybox" builtin="gradient" rgb1="0.52 0.55 0.60" rgb2="0.10 0.10 0.11" width="512" height="3072"/>
""")
    for i in range(N_SEG):
        s.append('<mesh name="bellows_%d" file="bellows_%d.obj" smoothnormal="false"/>' % (i, i))
        s.append('<mesh name="bellows_%d_inf" file="bellows_%d_inf.obj" smoothnormal="false"/>' % (i, i))
        s.append('<mesh name="details_%d" file="details_%d.obj" smoothnormal="true"/>' % (i, i))
    for n in ("base_flange", "base_bolts", "pcb", "pcb_chips", "pcb_sensor", "tensioners",
              "cable_clips", "cables", "wrist_plate", "servo_body",
              "servo_horn", "finger", "push_rods", "nameplate", "kettle_handle",
              "pump_body", "tank_body", "valve_block", "fittings", "gauge_dial",
              "gauge_needle", "air_tubes"):
        s.append('<mesh name="%s" file="%s.obj" smoothnormal="false"/>' % (n, n))
    for nm, sc in (("shadow_a", 0.115), ("shadow_b", 0.135), ("shadow_c", 0.058)):
        s.append('<mesh name="%s" file="shadow_disc.obj" scale="%.4f %.4f 1" inertia="shell"/>' % (nm, sc, sc))
    s.append("""  </asset>
  <visual>
    <quality shadowsize="8192" offsamples="16"/>
    <headlight ambient="0.46 0.47 0.50" diffuse="0.32 0.32 0.34" specular="0.06 0.06 0.06"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
""")

    s.append('<worldbody>')
    s.append('<geom name="floor" type="plane" size="8 8 0.1" material="mat_floor"/>')
    s.append('<geom name="backwall" type="box" pos="0 3.2 1.5" size="7.0 0.05 1.6" material="mat_wall"/>')
    s.append('<geom name="sidewall" type="box" pos="-4.2 0.5 1.5" size="0.05 3.8 1.6" material="mat_wall"/>')
    s.append('<light name="key" directional="true" pos="1.7 -1.8 2.8" dir="-0.50 0.55 -1" '
             'diffuse="1.35 1.30 1.22" specular="0.50 0.50 0.50" castshadow="false"/>')
    s.append('<light name="fill" directional="true" pos="-2.2 -0.8 1.5" dir="0.85 0.35 -0.50" '
             'diffuse="0.72 0.78 0.90" specular="0.18 0.18 0.18"/>')
    s.append('<light name="rim" directional="true" pos="-0.6 2.5 1.9" dir="0.20 -1.0 -0.50" '
             'diffuse="0.95 0.97 1.02" specular="0.38 0.38 0.38"/>')

    # 工作台
    s.append('<geom name="table_top" type="box" pos="0 0 0.72" size="0.62 0.42 0.02" material="mat_table"/>')
    for tag, sx, sy, rm in (("arm", 0.0, 0.0, 0.115), ("station", -0.08, 0.30, 0.135), ("kettle", 0.42, 0.27, 0.058)):
        for i, (f, a) in enumerate(((0.45, 0.30), (0.75, 0.18), (1.0, 0.08))):
            s.append('<geom name="shadow_%s_%d" type="cylinder" pos="%.3f %.3f %.4f" size="%.4f 0.0003" '
                     'material="mat_shadow%d" contype="0" conaffinity="0"/>' % (tag, i, sx, sy, 0.7404 + 0.0004 * i, rm * f, i + 1))
    for sx in (1, -1):
        for sy in (1, -1):
            s.append('<geom name="leg_%d_%d" type="box" pos="%.3f %.3f 0.36" size="0.022 0.022 0.36" '
                     'material="mat_dark_alu"/>' % (sx, sy, 0.55 * sx, 0.36 * sy))
    # 安装板与螺栓
    s.append('<geom name="mount_plate" type="box" pos="0 0 0.746" size="0.082 0.082 0.006" material="mat_dark_alu"/>')
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        s.append('<geom name="mount_bolt_%d" type="cylinder" pos="%.4f %.4f 0.753" size="0.0048 0.005" '
                 'material="mat_brass"/>' % (i, 0.062 * math.cos(a), 0.062 * math.sin(a)))
    # 道具：水壶
    s.append('<body name="kettle" pos="0.42 0.27 %.2f">' % TABLE_TOP)
    s.append('<geom name="k_body" type="cylinder" pos="0 0 0.048" size="0.036 0.048" material="mat_black_plast"/>')
    s.append('<geom name="k_lid" type="cylinder" pos="0 0 0.099" size="0.023 0.004" material="mat_steel"/>')
    s.append('<geom name="k_knob" type="cylinder" pos="0 0 0.108" size="0.008 0.005" material="mat_black_plast"/>')
    s.append('<geom name="k_handle" type="mesh" mesh="kettle_handle" euler="1.5708 0 0" pos="0 0 0.122" '
             'material="mat_black_plast"/>')
    s.append('<geom name="k_spout" type="cylinder" pos="0.032 0 0.072" euler="0 0.45 0" size="0.008 0.030" '
             'material="mat_steel"/>')
    s.append('</body>')

    # 气路硬件（工作台后侧）
    s.append('<body name="pneu_station" pos="-0.08 0.30 0.74">')
    s.append('<geom name="g_pump" type="mesh" mesh="pump_body" material="mat_dark_alu"/>')
    s.append('<geom name="g_tank" type="mesh" mesh="tank_body" material="mat_steel" '
             'euler="0 1.5708 0" pos="0.10 0.03 0.024"/>')
    s.append('<geom name="g_valves" type="mesh" mesh="valve_block" material="mat_black_plast" pos="0 -0.06 0"/>')
    s.append('<geom name="g_fittings" type="mesh" mesh="fittings" material="mat_brass" pos="0 -0.06 0"/>')
    s.append('<geom name="g_gauge_post" type="cylinder" pos="-0.055 -0.075 0.038" size="0.004 0.038" '
             'material="mat_black_plast"/>')
    s.append('<geom name="g_gauge" type="mesh" mesh="gauge_dial" material="mat_aluminum" '
             'pos="-0.055 -0.075 0.076" euler="1.5708 0 0"/>')
    s.append('<geom name="g_tubes" type="mesh" mesh="air_tubes" material="mat_rubber"/>')
    s.append('<body name="gauge_needle_body" pos="-0.055 -0.088 0.076">')
    s.append('<joint name="gauge_j" type="hinge" axis="0 1 0" pos="0 0 0" range="-1.1 1.1" '
             'armature="0.00005" damping="0.6"/>')
    s.append('<geom name="g_needle" type="mesh" mesh="gauge_needle" material="mat_orange"/>')
    s.append('</body>')
    s.append('</body>')

    s.append('<body name="arm_base" pos="0 0 %.3f">' % BASE_Z)
    s.append('<geom name="g_flange" type="mesh" mesh="base_flange" material="mat_aluminum"/>')
    s.append('<geom name="g_bolts" type="mesh" mesh="base_bolts" material="mat_brass"/>')
    s.append('<geom name="g_pcb" type="mesh" mesh="pcb" material="mat_pcb"/>')
    s.append('<geom name="g_pcb_chips" type="mesh" mesh="pcb_chips" material="mat_black_plast"/>')
    s.append('<geom name="g_pcb_sensor" type="mesh" mesh="pcb_sensor" material="mat_brass"/>')
    s.append('<geom name="g_tensioners" type="mesh" mesh="tensioners" material="mat_brass"/>')
    s.append('<geom name="g_clips" type="mesh" mesh="cable_clips" material="mat_black_plast"/>')
    s.append('<geom name="g_cables" type="mesh" mesh="cables" material="mat_rubber"/>')
    s.append('<geom name="g_plate" type="box" size="0.035 0.0006 0.0055" material="mat_plate" pos="0 -0.0825 -0.0065"/>')

    # 软臂
    z = 0.010
    s.append('<body name="seg_0" pos="0 0 %.4f">' % z)
    for i in range(N_SEG):
        r = R_START + (R_END - R_START) * i / (N_SEG - 1)
        mat = "mat_silicone" if i % 2 == 0 else "mat_silicone2"
        if i > 0:
            s.append('<body name="seg_%d" pos="0 0 %.4f">' % (i, SEG_L))
        s.append('<joint name="bx_%d" type="hinge" axis="1 0 0" pos="0 0 0" range="-0.9 0.9" armature="0.002" '
                 'limited="true" stiffness="%.3f" damping="%.3f"/>' % (i, K_SOFT, C_SOFT))
        s.append('<joint name="by_%d" type="hinge" axis="0 1 0" pos="0 0 0" range="-0.9 0.9" armature="0.002" '
                 'limited="true" stiffness="%.3f" damping="%.3f"/>' % (i, K_SOFT, C_SOFT))
        s.append('<geom name="g_bellows_%d" type="mesh" mesh="bellows_%d" material="%s" contype="0" conaffinity="0"/>' % (i, i, mat))
        s.append('<geom name="g_bellows_inf_%d" type="mesh" mesh="bellows_%d_inf" material="mat_silicone_inf" contype="0" conaffinity="0"/>' % (i, i))
        s.append('<geom name="g_details_%d" type="mesh" mesh="details_%d" material="mat_rubber" contype="0" conaffinity="0"/>' % (i, i))
        for k, ang in enumerate(TENDON_ANGLES):
            a = math.radians(ang)
            rr = r + 0.0022 * 0.85
            s.append('<site name="ts_%d_%d" pos="%.4f %.4f %.4f" size="0.0012"/>' %
                     (i, k, rr * math.cos(a), rr * math.sin(a), SEG_L / 2))
    # 腕部与夹爪
    s.append('<body name="wrist" pos="0 0 %.4f">' % (SEG_L))
    s.append('<geom name="g_wrist" type="mesh" mesh="wrist_plate" material="mat_aluminum"/>')
    s.append('<geom name="g_servo" type="mesh" mesh="servo_body" material="mat_black_plast"/>')
    s.append('<geom name="g_horn" type="mesh" mesh="servo_horn" material="mat_orange"/>')
    s.append('<geom name="g_pushrods" type="mesh" mesh="push_rods" material="mat_brass"/>')
    for side, tag in ((1, "l"), (-1, "r")):
        s.append('<body name="finger_%s" pos="%.4f 0 -0.036">' % (tag, 0.011 * side))
        s.append('<joint name="grip_%s" type="hinge" axis="0 1 0" pos="0 0 0" range="%.2f %.2f" armature="0.0005" '
                 'limited="true" stiffness="0.3" damping="0.03"/>' % (tag, -0.5 if side > 0 else 0.0, 0.0 if side > 0 else 0.5))
        s.append('<geom name="g_finger_%s" type="mesh" mesh="finger" material="mat_black_plast" euler="0 0 %.3f"/>' %
                 (tag, 0.0 if side > 0 else math.pi))
        s.append('</body>')
    s.append('</body>')
    for i in range(N_SEG - 1, -1, -1):
        s.append('</body>')
    s.append('</body>')   # arm_base
    s.append('</worldbody>')

    s.append('<tendon>')
    for k in range(3):
        s.append('<spatial name="tendon_%d" width="0.0008" frictionloss="0.01">' % k)
        for i in range(N_SEG):
            s.append('<site site="ts_%d_%d"/>' % (i, k))
        s.append('</spatial>')
    s.append('</tendon>')

    s.append('<actuator>')
    for k in range(3):
        s.append('<muscle name="tendon_m%d" tendon="tendon_%d" ctrlrange="0 1" force="%.1f" lengthrange="0.2 0.6"/>'
                 % (k, k, TENDON_FORCE))
    s.append('<position name="grip_l_pos" joint="grip_l" kp="6" kv="0.3" ctrlrange="-0.5 0"/>')
    s.append('<position name="grip_r_pos" joint="grip_r" kp="6" kv="0.3" ctrlrange="0 0.5"/>')
    s.append('<position name="gauge_pos" joint="gauge_j" kp="4" kv="0.4" ctrlrange="-1.0 1.0"/>')
    s.append('</actuator>')
    s.append('</mujoco>')
    xml = "\n".join(s)
    dens = {"mat_silicone": 1150, "mat_silicone2": 1150, "mat_silicone_inf": 1150, "mat_rubber": 1200,
            "mat_aluminum": 2700, "mat_dark_alu": 2700, "mat_brass": 8500, "mat_pcb": 1900,
            "mat_orange": 1200, "mat_steel": 7800, "mat_black_plast": 1200, "mat_plate": 8500}
    for mat, d in dens.items():
        xml = xml.replace('material="%s"' % mat, 'material="%s" density="%d"' % (mat, d))

    def mesh_volume(p):
        verts, faces = [], []
        for line in p.read_text(encoding="ascii").splitlines():
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
        v = np.asarray(verts)
        s = 0.0
        for a, b, c in faces:
            s += float(np.dot(v[a], np.cross(v[b], v[c]))) / 6.0
        return abs(s)

    vols = {}

    def repl(match):
        mesh = match.group("mesh")
        dens = 1.0 if mesh.endswith("_inf") else float(match.group("dens"))
        if mesh not in vols:
            p = MESH_DIR / (mesh + ".obj")
            vols[mesh] = mesh_volume(p) if p.exists() else 0.0
        return 'mesh="%s" material="%s" mass="%.8f"' % (mesh, match.group("mat"), vols[mesh] * dens)

    xml = re.sub(r'mesh="(?P<mesh>[^"]+)" material="(?P<mat>[^"]+)" density="(?P<dens>\d+)"', repl, xml)
    path.write_text(xml, encoding="utf-8")
    return path


def build():
    xml = HERE / "_arm_real.xml"
    gen_xml(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class Pneumatic:
    """气路模型，泵、比例阀、排气阀、气腔与压力传感器。"""

    def __init__(self, seed=11):
        self.P = 0.0
        self.P_sup = 0.0
        self.P_meas = 0.0
        self.pwm = 0.0
        self.valve = 0.0
        self.exh = 0.0
        self.dir = 0
        self.hyst = 0.0
        self.valve_gain = 3.0
        self.noise_sd = 0.012
        self.rng = np.random.default_rng(seed)

    def step(self, pref, dt):
        e = pref - self.P
        if e > 0.02:
            self.pwm = 1.0
            self.valve = min(1.0, self.valve_gain * e)
            self.exh = 0.0
        elif e > -0.02:
            self.pwm = 0.2
            self.valve = 0.0
            self.exh = 0.0
        else:
            self.pwm = 0.2
            self.valve = 0.0
            self.exh = min(1.0, -2.5 * e)
        self.P_sup += dt / TAU_PUMP * (self.pwm * P_MAX - self.P_sup)
        uv = (self.valve - DEAD) / (1.0 - DEAD) if self.valve > DEAD else 0.0
        q_in = CV * uv * max(self.P_sup - self.P, 0.0)
        q_out = (CE * self.exh + LEAK) * max(self.P, 0.0)
        prev = self.P
        self.P = max(0.0, self.P + dt * (q_in - q_out))
        dP = self.P - prev
        if abs(dP) > 1e-5:
            self.dir = 1 if dP > 0 else -1
        self.hyst += dt / 0.4 * (0.045 * self.dir - self.hyst)
        self.P_meas = round((self.P + float(self.rng.normal(0, self.noise_sd))) / 0.005) * 0.005
        return self.P_meas

    def stiffness(self):
        a = min(max((self.P - self.hyst) / P_STEP, 0.0), 1.0)
        return a * a * (3.0 - 2.0 * a)

    def inflation(self):
        a = min(max(self.P / P_STEP, 0.0), 1.0)
        return a * a * (3.0 - 2.0 * a)


def set_ctrl(model, data, tendons, grip, stiff=0.0, pressure=0.0):
    for i in range(N_SEG):
        k = K_SOFT + (K_STIFF - K_SOFT) * stiff
        c = C_SOFT + (C_STIFF - C_SOFT) * stiff
        for pre in ("bx_", "by_"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + str(i))
            model.jnt_stiffness[jid] = k
            model.dof_damping[model.jnt_dofadr[jid]] = c
        infl = min(max((pressure / P_STEP - 0.35) / 0.30, 0.0), 1.0)
        infl = infl * infl * (3.0 - 2.0 * infl)
        gid_def = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_bellows_%d" % i)
        gid_inf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_bellows_inf_%d" % i)
        model.geom_rgba[gid_def][3] = 1.0 - infl
        model.geom_rgba[gid_inf][3] = infl
    for k in range(3):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tendon_m%d" % k)] = tendons[k]
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_l_pos")] = -0.5 * grip
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_r_pos")] = 0.5 * grip
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gauge_pos")] = -1.0 + 2.0 * min(pressure / P_MAX, 1.0)


def timeline(t):
    def ramp(t, t0, t1, a, b):
        u = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
        return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * u))
    td = [0.0, 0.0, 0.0]
    grip, pref = 0.0, 0.0
    if t < 1.5:
        phase = "静置（软态下垂）"
    elif t < 4.0:
        phase = "腱驱一号向"
        td[0] = ramp(t, 1.5, 3.8, 0.0, 1.0)
    elif t < 6.0:
        phase = "腱驱二号向"
        td[0] = ramp(t, 4.0, 5.0, 1.0, 0.0)
        td[1] = ramp(t, 4.5, 5.8, 0.0, 1.0)
    elif t < 8.0:
        phase = "腱驱三号向"
        td[1] = ramp(t, 6.0, 7.0, 1.0, 0.0)
        td[2] = ramp(t, 6.5, 7.8, 0.0, 1.0)
    elif t < 8.8:
        phase = "回中"
        td[2] = ramp(t, 8.0, 8.8, 1.0, 0.0)
    elif t < 10.6:
        phase = "气泵充气"
        pref = P_STEP
        td[0] = ramp(t, 8.8, 10.2, 0.0, 0.35)
    else:
        phase = "保压夹持"
        pref = P_STEP
        td[0] = 0.35
        grip = ramp(t, 10.8, 11.6, 0.0, 1.0)
    return td, grip, pref, phase


T_END = 12.0


def camera_at(cam, t):
    u = min(max(t / T_END, 0.0), 1.0)
    cam.azimuth = 152.0 - 26.0 * u
    cam.elevation = -5.5 + 3.0 * math.sin(math.pi * u)
    cam.distance = 1.10 - 0.10 * u
    cam.lookat[:] = (0.0, 0.03, 0.87 + 0.03 * u)


def draw_pneu(im, hist):
    W, H = im.size
    dr = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
    font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    x0, y0, w, h = W - 590, 26, 562, 276
    dr.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=14,
                         fill=(10, 11, 14, 178), outline=(255, 255, 255, 45), width=1)
    px0, py0, pw, ph = x0 + 66, y0 + 44, w - 104, h - 132
    for pv in (0.0, 0.5, 1.0, 1.5, 2.0):
        yy = int(py0 + ph * (1 - pv / P_MAX))
        dr.line((px0, yy, px0 + pw, yy), fill=(255, 255, 255, 34), width=1)
        dr.text((px0 - 52, yy - 10), "%.1f" % pv, font=font_s, fill=(168, 174, 184, 255))
    dr.text((px0 - 52, py0 - 30), "腔压 bar", font=font_s, fill=(205, 210, 218, 255))
    for tv in range(0, 13, 2):
        xx = int(px0 + pw * tv / T_END)
        dr.line((xx, py0, xx, py0 + ph), fill=(255, 255, 255, 24), width=1)
        dr.text((xx - 12, py0 + ph + 6), "%d" % tv, font=font_s, fill=(148, 154, 164, 255))
    dr.text((px0 + pw + 6, py0 + ph + 6), "s", font=font_s, fill=(148, 154, 164, 255))

    def mx(t):
        return px0 + pw * min(max(t / T_END, 0.0), 1.0)

    def my(p):
        return py0 + ph * (1 - min(max(p / P_MAX, 0.0), 1.0))

    if len(hist) > 1:
        ref = [(mx(t), my(pref)) for (t, p, pref, pwm, valve) in hist]
        for i in range(0, len(ref) - 4, 6):
            dr.line((ref[i][0], ref[i][1], ref[i + 3][0], ref[i + 3][1]), fill=(162, 168, 178, 190), width=2)
        pts = [(mx(t), my(p)) for (t, p, pref, pwm, valve) in hist]
        dr.line(pts, fill=(235, 90, 80, 255), width=3, joint="curve")
        dr.ellipse((pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4), fill=(255, 120, 105, 255))
        last = hist[-1]
        dr.text((x0 + w - 300, y0 + 12), "腔压 %.2f bar  指令 %.1f bar" % (last[1], last[2]),
                font=font, fill=(240, 202, 198, 255))
        by0 = py0 + ph + 34
        dr.text((px0, by0), "泵 PWM", font=font_s, fill=(205, 210, 218, 255))
        dr.rectangle((px0 + 72, by0 + 2, px0 + 192, by0 + 16), outline=(255, 255, 255, 60), width=1)
        dr.rectangle((px0 + 72, by0 + 2, px0 + 72 + 120 * last[3], by0 + 16), fill=(90, 170, 235, 220))
        dr.text((px0 + 198, by0), "%3d%%" % int(round(last[3] * 100)), font=font_s, fill=(180, 214, 240, 255))
        dr.text((px0 + 258, by0), "阀位", font=font_s, fill=(205, 210, 218, 255))
        dr.rectangle((px0 + 308, by0 + 2, px0 + 398, by0 + 16), outline=(255, 255, 255, 60), width=1)
        dr.rectangle((px0 + 308, by0 + 2, px0 + 308 + 90 * last[4], by0 + 16), fill=(240, 170, 80, 220))
        dr.text((px0 + 404, by0), "%3d%%" % int(round(last[4] * 100)), font=font_s, fill=(244, 210, 170, 255))
    return im


def render(stills=False, video_path=None):
    W, H, FPS = 1920, 1080, 50
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, data = build()
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    opt = mujoco.MjvOption()
    for flag in (mujoco.mjtVisFlag.mjVIS_TENDON, mujoco.mjtVisFlag.mjVIS_ACTUATOR,
                 mujoco.mjtVisFlag.mjVIS_JOINT, mujoco.mjtVisFlag.mjVIS_COM,
                 mujoco.mjtVisFlag.mjVIS_CAMERA, mujoco.mjtVisFlag.mjVIS_LIGHT,
                 mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
                 mujoco.mjtVisFlag.mjVIS_CONSTRAINT):
        opt.flags[flag] = 0
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.10
    cam.azimuth = 152.0
    cam.elevation = -5.5
    cam.lookat[:] = (0.0, 0.03, 0.87)
    font_big = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
    font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - 0.12 * np.clip(rr - 0.50, 0, 1) / 0.50, 0, 1).astype(np.float32)
    proc = None
    if not stills:
        video_path = video_path or (OUT_DIR / "arm_real.mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16",
             "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)],
            stdin=subprocess.PIPE)
    dt = model.opt.timestep
    n = int(T_END / dt)
    spf = 20
    frame = 0
    t0 = time.monotonic()
    shot_at = [(1.2, "rest"), (3.6, "bend1"), (9.0, "air_lo"), (10.4, "stiff")]
    shots_done = set()
    pneu = Pneumatic()
    hist = []
    log_rows = []
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    for k in range(n):
        t = (k + 1) * dt
        td, grip, pref, phase = timeline(t)
        p_meas = pneu.step(pref, dt)
        set_ctrl(model, data, td, grip, pneu.stiffness(), pneu.P)
        mujoco.mj_step(model, data)
        hist.append((t, pneu.P, pref, pneu.pwm, pneu.valve))
        log_rows.append((t, pref, pneu.P, p_meas, pneu.pwm, pneu.valve, pneu.exh, float(data.xpos[bid][2])))
        if stills:
            for ts, tag in shot_at:
                if tag not in shots_done and t >= ts:
                    shots_done.add(tag)
                    renderer.update_scene(data, cam, opt)
                    img = renderer.render().copy()
                    Image.fromarray(img).save(HERE / ("still_%s.png" % tag))
                    print("still saved", tag)
        if not stills and k % spf == 0:
            camera_at(cam, t)
            renderer.update_scene(data, cam, opt)
            img = renderer.render().copy()
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im, "RGBA")
            l1 = "%s · t=%.1f s" % (phase, t)
            l2 = "程序化重建 · 波纹管 8 节 · 三路腱驱 · 气路模型（泵阀腔传感） · 充气波纹管 · 两指夹爪"
            pad, lh = 18, 40
            wpx = max(dr.textlength(s2, font=f) for s2, f in ((l1, font_big), (l2, font_mid)))
            bw = int(wpx) + 2 * pad
            bh = 2 * lh + 2 * pad - 6
            x0, y0 = 26, H - 26 - bh
            dr.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=14,
                                 fill=(12, 13, 16, 150), outline=(255, 255, 255, 30), width=1)
            dr.text((x0 + pad, y0 + pad - 4), l1, font=font_big, fill=(235, 238, 242, 255))
            dr.text((x0 + pad, y0 + pad + lh - 4), l2, font=font_mid, fill=(190, 196, 205, 255))
            draw_pneu(im, hist)
            graded = np.clip((np.asarray(im, dtype=np.float32) / 255.0 - 0.5) * 1.05 + 0.5, 0, 1) * 255.0
            img[:] = (graded * vig[..., None]).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            frame += 1
            if frame % 150 == 0:
                print("  t=%.1f %s" % (t, phase))
    if stills:
        renderer.close()
        return 0
    proc.stdin.close()
    proc.wait()
    renderer.close()
    csv_path = OUT_DIR / "pneu_log.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("t,pref_bar,P_bar,P_meas_bar,pwm,valve,exh,tip_z_m\n")
        for row in log_rows[::10]:
            f.write(",".join("%.4f" % v for v in row) + "\n")
    print("wrote %s (%.1f MB, %.1f s)" % (video_path, Path(video_path).stat().st_size / 1e6, frame / FPS))
    print("wrote %s" % csv_path)
    print("wall %.0fs" % (time.monotonic() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    return render(stills=args.stills)


if __name__ == "__main__":
    sys.exit(main())
