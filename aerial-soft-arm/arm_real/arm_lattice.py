"""按真机结构重建的空中桁架臂，无人机悬吊，反棱柱桁架链，5 倍伸缩。

运行：
  python arm_lattice.py --stills   出四张静帧
  python arm_lattice.py --render   出 12 秒展示视频
"""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_real as A

MESH_DIR = HERE / "meshes"
OUT_DIR = A.OUT_DIR

N_SEG = 8
R_RING = 0.028
SPAN_MIN = 0.010
SPAN_MAX = 0.050
TENDON_FORCE = 22.0
K_SOFT, C_SOFT = 1.10, 0.10
K_STIFF, C_STIFF = 6.0, 0.5
TENDON_ANGLES = (90.0, 210.0, 330.0)
DRONE_Z = 2.02
TABLE_TOP = 0.74

VARIANT_HEIGHTS = [0.010 + k * 0.004 for k in range(11)]
MESHES = tuple(["lattice_v%02d" % k for k in range(11)] + ["bundle_v%02d" % k for k in range(11)]
               + ["sheath_v%02d" % k for k in range(11)]) + (
          "tether_hose", "grip_base", "finger_prox", "finger_dist", "finger_flat",
          "finger_soft", "finger_knuckle", "grip_links",
          "truss_clamp", "truss_base_ring", "truss_tensioners",
          "drone_plate", "drone_arm", "drone_motor", "drone_prop", "drone_skid",
          "base_flange", "base_bolts", "pcb",
          "pump_body", "tank_body", "valve_block", "fittings", "gauge_dial",
          "gauge_needle", "kettle_handle")

GRIPPER_DEFS = (
    {"name": "菱形两指", "angs": (0.0, 3.1416), "prox": "finger_prox", "dist": "finger_dist",
     "mats": (("mat_blue", "mat_red"), ("mat_red", "mat_blue")),
     "col": (0.0013, 0.0075, 0.040, -0.0062, -0.050)},
    {"name": "平口两指", "angs": (0.0, 3.1416), "prox": "finger_flat", "dist": None,
     "mats": (("mat_steel",), ("mat_steel",)),
     "col": (0.0013, 0.0075, 0.042, -0.0062, -0.042)},
    {"name": "三指", "angs": (1.5708, 3.6652, 5.7596), "prox": "finger_prox", "dist": "finger_dist",
     "mats": (("mat_blue", "mat_red"), ("mat_red", "mat_blue"), ("mat_blue", "mat_red")),
     "col": (0.0013, 0.0075, 0.040, -0.0062, -0.050)},
    {"name": "软体两指", "angs": (0.0, 3.1416), "prox": "finger_soft", "dist": None,
     "mats": (("mat_silicone",), ("mat_silicone",)),
     "col": (0.0012, 0.0060, 0.036, -0.0093, -0.036)},
)


def gen_xml(path):
    s = []
    s.append('<mujoco model="arm_lattice">')
    s.append('<compiler angle="radian" meshdir="meshes"/>')
    s.append('<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" iterations="150" ls_iterations="80"/>')
    s.append("""
  <asset>
    <texture type="2d" name="tex_fabric" file="textures/fabric.png"/>
    <texture type="2d" name="tex_brushed" file="textures/brushed.png"/>
    <texture type="2d" name="tex_speckle" file="textures/speckle.png"/>
    <texture type="skybox" builtin="gradient" rgb1="0.62 0.65 0.70" rgb2="0.10 0.10 0.11" width="512" height="3072"/>
    <material name="mat_truss" texture="tex_brushed" rgba="0.44 0.46 0.49 1" roughness="0.62" metallic="0.10" specular="0.28" shininess="0.36"/>
    <material name="mat_truss2" texture="tex_brushed" rgba="0.40 0.42 0.45 1" roughness="0.64" metallic="0.09" specular="0.26" shininess="0.34"/>
    <material name="mat_rubber" texture="tex_speckle" rgba="0.16 0.16 0.18 1" roughness="0.90" metallic="0.0" specular="0.10" shininess="0.20"/>
    <material name="mat_carbon" texture="tex_speckle" rgba="0.13 0.13 0.15 1" roughness="0.55" metallic="0.15" specular="0.25" shininess="0.30"/>
    <material name="mat_aluminum" texture="tex_brushed" rgba="0.62 0.64 0.68 1" roughness="0.34" metallic="0.85" specular="0.45" shininess="0.50"/>
    <material name="mat_brass" rgba="0.78 0.62 0.30 1" roughness="0.30" metallic="0.90" specular="0.60" shininess="0.60"/>
    <material name="mat_pcb" rgba="0.06 0.22 0.12 1" roughness="0.50" metallic="0.0" specular="0.35" shininess="0.40"/>
    <material name="mat_red" rgba="0.72 0.10 0.09 1" roughness="0.38" metallic="0.55" specular="0.50" shininess="0.50"/>
    <material name="mat_blue" rgba="0.10 0.24 0.60 1" roughness="0.38" metallic="0.55" specular="0.50" shininess="0.50"/>
    <material name="mat_black_plast" rgba="0.11 0.11 0.12 1" roughness="0.55" metallic="0.0" specular="0.30" shininess="0.35"/>
    <material name="mat_sheath" rgba="0.09 0.09 0.10 1" roughness="0.88" metallic="0.0" specular="0.05" shininess="0.16"/>
    <material name="mat_silicone" rgba="0.82 0.83 0.85 1" roughness="0.62" metallic="0.0" specular="0.22" shininess="0.28"/>
    <material name="mat_steel" texture="tex_brushed" rgba="0.72 0.74 0.77 1" roughness="0.28" metallic="0.90" specular="0.55" shininess="0.55"/>
    <material name="mat_orange" rgba="0.90 0.44 0.10 1" roughness="0.42" metallic="0.0" specular="0.45" shininess="0.50"/>
    <material name="mat_table" texture="tex_speckle" rgba="0.37 0.37 0.38 1" roughness="0.72" metallic="0.0" specular="0.25" shininess="0.30"/>
    <material name="mat_floor" texture="tex_speckle" rgba="0.28 0.28 0.29 1" roughness="0.90" metallic="0.0" specular="0.10" shininess="0.20"/>
    <material name="mat_wall" rgba="0.21 0.22 0.24 1" roughness="0.95" metallic="0.0" specular="0.05" shininess="0.20"/>
    <material name="mat_shadow1" rgba="0.0 0.0 0.0 0.30" roughness="0.9" metallic="0.0" specular="0.0" shininess="0.1"/>
    <material name="mat_shadow2" rgba="0.0 0.0 0.0 0.18" roughness="0.9" metallic="0.0" specular="0.0" shininess="0.1"/>
    <material name="mat_shadow3" rgba="0.0 0.0 0.0 0.08" roughness="0.9" metallic="0.0" specular="0.0" shininess="0.1"/>
""")
    for n in MESHES:
        s.append('<mesh name="%s" file="%s.obj" smoothnormal="false"/>' % (n, n))
    for nm, sc in (("shadow_a", 0.16), ("shadow_b", 0.135), ("shadow_c", 0.058)):
        s.append('<mesh name="%s" file="shadow_disc.obj" scale="%.4f %.4f 1" inertia="shell"/>' % (nm, sc, sc))
    s.append("""  </asset>
  <visual>
    <quality shadowsize="8192" offsamples="16"/>
    <headlight ambient="0.38 0.39 0.42" diffuse="0.30 0.30 0.32" specular="0.05 0.05 0.05"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
""")
    s.append('<worldbody>')
    s.append('<geom name="floor" type="plane" size="8 8 0.1" material="mat_floor"/>')
    s.append('<geom name="backwall" type="box" pos="0 3.2 1.5" size="7.0 0.05 1.6" material="mat_wall"/>')
    s.append('<geom name="sidewall" type="box" pos="-4.2 0.5 1.5" size="0.05 3.8 1.6" material="mat_wall"/>')
    s.append('<light name="key" directional="true" pos="1.7 -1.8 2.8" dir="-0.50 0.55 -1" '
             'diffuse="1.05 1.02 0.98" specular="0.45 0.45 0.45" castshadow="false"/>')
    s.append('<light name="fill" directional="true" pos="-2.2 -0.8 1.5" dir="0.85 0.35 -0.50" '
             'diffuse="0.72 0.78 0.90" specular="0.18 0.18 0.18"/>')
    s.append('<light name="rim" directional="true" pos="-0.6 2.5 1.9" dir="0.20 -1.0 -0.50" '
             'diffuse="0.95 0.97 1.02" specular="0.38 0.38 0.38"/>')
    s.append('<geom name="table_top" type="box" pos="0 0 0.72" size="0.62 0.42 0.02" material="mat_table"/>')
    for sx in (1, -1):
        for sy in (1, -1):
            s.append('<geom name="leg_%d_%d" type="box" pos="%.3f %.3f 0.36" size="0.022 0.022 0.36" '
                     'material="mat_carbon"/>' % (sx, sy, 0.55 * sx, 0.36 * sy))
    for tag, sx, sy, rm in (("arm", 0.0, 0.0, 0.16), ("station", -0.08, 0.30, 0.135), ("kettle", 0.42, 0.27, 0.058)):
        for i, (f, a) in enumerate(((0.45, 0.30), (0.75, 0.18), (1.0, 0.08))):
            s.append('<geom name="shadow_%s_%d" type="cylinder" pos="%.3f %.3f %.4f" size="%.4f 0.0003" '
                     'material="mat_shadow%d" contype="0" conaffinity="0"/>' % (tag, i, sx, sy, 0.7404 + 0.0004 * i, rm * f, i + 1))

    s.append('<body name="pneu_station" pos="-0.08 0.30 0.74">')
    s.append('<geom name="g_pump" type="mesh" mesh="pump_body" material="mat_carbon"/>')
    s.append('<geom name="g_tank" type="mesh" mesh="tank_body" material="mat_steel" '
             'euler="0 1.5708 0" pos="0.10 0.03 0.024"/>')
    s.append('<geom name="g_valves" type="mesh" mesh="valve_block" material="mat_black_plast" pos="0 -0.06 0"/>')
    s.append('<geom name="g_fittings" type="mesh" mesh="fittings" material="mat_brass" pos="0 -0.06 0"/>')
    s.append('<geom name="g_gauge_post" type="cylinder" pos="-0.055 -0.075 0.038" size="0.004 0.038" '
             'material="mat_black_plast"/>')
    s.append('<geom name="g_gauge" type="mesh" mesh="gauge_dial" material="mat_aluminum" '
             'pos="-0.055 -0.075 0.076" euler="1.5708 0 0"/>')
    s.append('<body name="gauge_needle_body" pos="-0.055 -0.088 0.076">')
    s.append('<joint name="gauge_j" type="hinge" axis="0 1 0" pos="0 0 0" range="-1.1 1.1" '
             'armature="0.00005" damping="0.6"/>')
    s.append('<geom name="g_needle" type="mesh" mesh="gauge_needle" material="mat_orange"/>')
    s.append('</body>')
    s.append('</body>')

    s.append('<body name="kettle" pos="0.42 0.27 %.2f">' % TABLE_TOP)
    s.append('<geom name="k_body" type="cylinder" pos="0 0 0.048" size="0.036 0.048" material="mat_carbon"/>')
    s.append('<geom name="k_lid" type="cylinder" pos="0 0 0.099" size="0.023 0.004" material="mat_steel"/>')
    s.append('<geom name="k_knob" type="cylinder" pos="0 0 0.108" size="0.008 0.005" material="mat_black_plast"/>')
    s.append('<geom name="k_handle" type="mesh" mesh="kettle_handle" euler="1.5708 0 0" pos="0 0 0.122" '
             'material="mat_black_plast"/>')
    s.append('<geom name="k_spout" type="cylinder" pos="0.032 0 0.072" euler="0 0.45 0" size="0.008 0.030" '
             'material="mat_steel"/>')
    s.append('</body>')

    s.append('<body name="grasp_stand" pos="0 0.30 0.74">')
    s.append('<geom name="gs_post" type="cylinder" pos="0 0 0.39" size="0.012 0.39" material="mat_aluminum"/>')
    s.append('<geom name="gs_plate" type="cylinder" pos="0 0 0.785" size="0.050 0.008" material="mat_black_plast"/>')
    s.append('</body>')
    s.append('<body name="part" pos="0 0.30 1.565">')
    s.append('<freejoint name="part_free"/>')
    s.append('<geom name="p_body" type="cylinder" pos="0 0 0" size="0.014 0.030" material="mat_orange" '
             'friction="1.4 0.02 0.001"/>')
    s.append('</body>')

    s.append('<body name="drone" pos="0 0 %.3f">' % DRONE_Z)
    s.append('<geom name="d_plate" type="mesh" mesh="drone_plate" material="mat_carbon" contype="0" conaffinity="0"/>')
    for i, ang in enumerate((0.7854, 2.3562, 3.9270, 5.4978)):
        s.append('<geom name="d_arm_%d" type="mesh" mesh="drone_arm" material="mat_carbon" euler="0 0 %.4f" contype="0" conaffinity="0"/>' % (i, ang))
        s.append('<geom name="d_motor_%d" type="mesh" mesh="drone_motor" material="mat_black_plast" '
                 'pos="%.4f %.4f 0.010" contype="0" conaffinity="0"/>' % (i, 0.125 * math.cos(ang), 0.125 * math.sin(ang)))
        s.append('<geom name="d_prop_%d" type="mesh" mesh="drone_prop" material="mat_black_plast" '
                 'pos="%.4f %.4f 0.042" euler="0 0 %.4f" contype="0" conaffinity="0"/>' % (i, 0.125 * math.cos(ang), 0.125 * math.sin(ang), ang + 0.3 * i))
    for sx in (1, -1):
        s.append('<geom name="d_skid_%d" type="mesh" mesh="drone_skid" material="mat_carbon" pos="0 %.3f -0.030" contype="0" conaffinity="0"/>' % (sx, 0.05 * sx))

    s.append('<body name="arm_mount" pos="0 0 -0.016">')
    s.append('<geom name="m_flange" type="mesh" mesh="base_flange" material="mat_aluminum" pos="0 0 -0.0055" euler="3.1416 0 0" contype="0" conaffinity="0"/>')
    s.append('<geom name="m_bolts" type="mesh" mesh="base_bolts" material="mat_brass" pos="0 0 -0.0245" contype="0" conaffinity="0"/>')
    s.append('<geom name="m_pcb" type="mesh" mesh="pcb" material="mat_pcb" pos="0 0 0.0103" euler="3.1416 0 0" contype="0" conaffinity="0"/>')
    s.append('<geom name="m_tring" type="mesh" mesh="truss_base_ring" material="mat_aluminum" pos="0 0 -0.0125" euler="3.1416 0 0" contype="0" conaffinity="0"/>')
    s.append('<geom name="m_tens" type="mesh" mesh="truss_tensioners" material="mat_brass" pos="0 0 -0.0155" euler="3.1416 0 0" contype="0" conaffinity="0"/>')

    z0 = 0.010
    for i in range(N_SEG):
        z_off = 0.0155 if i == 0 else z0
        s.append('<body name="seg_%d" pos="0 0 %.4f">' % (i, -z_off))
        if i > 0:
            s.append('<joint name="slide_%d" type="slide" axis="0 0 1" pos="0 0 0" range="-0.040 0" armature="0.002"/>' % i)
        s.append('<joint name="bx_%d" type="hinge" axis="1 0 0" pos="0 0 0" range="-0.9 0.9" armature="0.002" '
                 'limited="true" stiffness="%.3f" damping="%.3f"/>' % (i, K_SOFT, C_SOFT))
        s.append('<joint name="by_%d" type="hinge" axis="0 1 0" pos="0 0 0" range="-0.9 0.9" armature="0.002" '
                 'limited="true" stiffness="%.3f" damping="%.3f"/>' % (i, K_SOFT, C_SOFT))
        for k in range(11):
            zr = 0.0698 if i % 2 == 1 else 0.0
            s.append('<geom name="g_lat_v%02d_%d" type="mesh" mesh="lattice_v%02d" material="mat_truss" euler="3.1416 0 %.4f" contype="0" conaffinity="0"/>' % (k, i, k, 0.0 if i % 2 == 0 else 0.7854))
            s.append('<geom name="g_bundle_v%02d_%d" type="mesh" mesh="bundle_v%02d" material="mat_rubber" euler="3.1416 0 %.4f" contype="0" conaffinity="0"/>' % (k, i, k, zr))
            s.append('<geom name="g_sheath_v%02d_%d" type="mesh" mesh="sheath_v%02d" material="mat_sheath" euler="3.1416 0 %.4f" contype="0" conaffinity="0"/>' % (k, i, k, zr))
        s.append('<geom name="g_clamp_%d" type="mesh" mesh="truss_clamp" material="mat_steel" euler="3.1416 0 %.4f" contype="0" conaffinity="0"/>' % (i, zr))
        for k, ang in enumerate(TENDON_ANGLES):
            a = math.radians(ang)
            rr = R_RING * 1.12
            s.append('<site name="ts_%d_%d" pos="%.4f %.4f -0.005" size="0.0012"/>' % (i, k, rr * math.cos(a), rr * math.sin(a)))
    s.append('<body name="wrist" pos="0 0 %.4f">' % (-z0,))
    s.append('<joint name="slide_wrist" type="slide" axis="0 0 1" pos="0 0 0" range="-0.040 0" armature="0.002"/>')
    s.append('<geom name="g_grip_base" type="mesh" mesh="grip_base" material="mat_black_plast"/>')
    s.append('<geom name="g_grip_links" type="mesh" mesh="grip_links" material="mat_black_plast" contype="0" conaffinity="0"/>')
    for v, gd in enumerate(GRIPPER_DEFS):
        for i, a in enumerate(gd["angs"]):
            mats = gd["mats"][i]
            s.append('<body name="finger%d_%d" pos="%.4f %.4f -0.036" euler="0 0 %.4f">'
                     % (v, i, 0.024 * math.cos(a), 0.024 * math.sin(a), a))
            s.append('<joint name="grip%d_%d" type="hinge" axis="0 1 0" pos="0 0 0" range="-0.45 0.20" armature="0.0005" '
                     'limited="true" stiffness="0.3" damping="0.03"/>' % (v, i))
            s.append('<geom name="g_f%d_%d_p" type="mesh" mesh="%s" material="%s" contype="0" conaffinity="0"/>' % (v, i, gd["prox"], mats[0]))
            if gd["dist"]:
                s.append('<geom name="g_f%d_%d_d" type="mesh" mesh="%s" material="%s" contype="0" conaffinity="0"/>' % (v, i, gd["dist"], mats[1]))
            c = gd["col"]
            s.append('<geom name="g_f%d_%d_c" type="box" group="3" size="%.4f %.4f %.4f" pos="%.4f 0 %.4f" '
                     'friction="2.0 0.02 0.001" solref="0.002 1" solimp="0.98 0.99 0.001"/>' % (v, i, c[0], c[1], c[2], c[3], c[4]))
            s.append('<geom name="g_f%d_%d_k" type="mesh" mesh="finger_knuckle" material="mat_steel" contype="0" conaffinity="0"/>' % (v, i))
            s.append('</body>')
    s.append('</body>')
    for i in range(N_SEG - 1, -1, -1):
        s.append('</body>')
    s.append('</body>')
    s.append('</body>')
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
        s.append('<muscle name="tendon_m%d" tendon="tendon_%d" ctrlrange="0 1" force="%.1f" lengthrange="0.2 1.2"/>'
                 % (k, k, TENDON_FORCE))
    for v, gd in enumerate(GRIPPER_DEFS):
        for i in range(len(gd["angs"])):
            s.append('<position name="grip%d_%d_pos" joint="grip%d_%d" kp="6" kv="0.3" ctrlrange="-0.45 0.30"/>' % (v, i, v, i))
    for i in range(1, N_SEG):
        s.append('<position name="slide_%d_pos" joint="slide_%d" kp="20000" kv="200" ctrlrange="-0.040 0"/>' % (i, i))
    s.append('<position name="slide_wrist_pos" joint="slide_wrist" kp="20000" kv="200" ctrlrange="-0.040 0"/>')
    s.append('<position name="gauge_pos" joint="gauge_j" kp="4" kv="0.4" ctrlrange="-1.0 1.0"/>')
    s.append('</actuator>')
    s.append('</mujoco>')
    path.write_text("\n".join(s), encoding="utf-8")
    return path


def set_gripper(model, v):
    """切换可见夹爪型号：激活的显示并参与碰撞，其余透明且不碰撞。"""
    for vv, gd in enumerate(GRIPPER_DEFS):
        active = vv == int(v)
        for i in range(len(gd["angs"])):
            for suf in ("p", "d", "c"):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_f%d_%d_%s" % (vv, i, suf))
                if gid < 0:
                    continue
                model.geom_rgba[gid][3] = 1.0 if active else 0.0
                if suf == "c":
                    model.geom_contype[gid] = 1 if active else 0
                    model.geom_conaffinity[gid] = 1 if active else 0


def build():
    xml = HERE / "_arm_lattice.xml"
    gen_xml(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    set_gripper(model, 0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def variant_weights(e):
    """十一档网格按相邻两档线性插值，伸缩全程连续。"""
    x = min(max(e, 0.0), 1.0) * 10.0
    i = min(int(x), 9)
    f = x - i
    w = [0.0] * 11
    w[i] = 1.0 - f
    w[i + 1] = f
    return w


_LAST_E = [-1.0]


def set_ctrl(model, data, tendons, grip, stiff, pressure):
    e = min(max(pressure / A.P_STEP, 0.0), 1.0)
    for i in range(N_SEG):
        for pre in ("bx_", "by_"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + str(i))
            model.jnt_stiffness[jid] = K_SOFT + (K_STIFF - K_SOFT) * stiff
            model.dof_damping[model.jnt_dofadr[jid]] = C_SOFT + (C_STIFF - C_SOFT) * stiff
        if i > 0:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_%d_pos" % i)
            data.ctrl[aid] = -0.040 * e
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_wrist_pos")
    data.ctrl[aid] = -0.040 * e
    if abs(e - _LAST_E[0]) > 0.0005:
        for k, w in enumerate(variant_weights(e)):
            for i in range(N_SEG):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_lat_v%02d_%d" % (k, i))
                model.geom_rgba[gid][3] = w
                bid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_bundle_v%02d_%d" % (k, i))
                model.geom_rgba[bid2][3] = w
                sid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_sheath_v%02d_%d" % (k, i))
                model.geom_rgba[sid2][3] = w
        _LAST_E[0] = e
    for k in range(3):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tendon_m%d" % k)] = tendons[k]
    for v, gd in enumerate(GRIPPER_DEFS):
        for i in range(len(gd["angs"])):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip%d_%d_pos" % (v, i))
            data.ctrl[aid] = -0.40 + 0.70 * grip
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gauge_pos")] = -1.0 + 2.0 * pressure


def timeline(t):
    def ramp(t, t0, t1, a, b):
        u = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
        return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * u))
    td = [0.0, 0.0, 0.0]
    grip, pref = 0.0, 0.0
    if t < 1.5:
        phase = "收拢悬垂"
    elif t < 4.0:
        phase = "充气伸展 5 倍"
        pref = A.P_STEP
    elif t < 6.5:
        phase = "腱驱一号向"
        pref = A.P_STEP
        td[0] = ramp(t, 4.0, 6.2, 0.0, 0.9)
    elif t < 8.5:
        phase = "腱驱二号向"
        pref = A.P_STEP
        td[0] = ramp(t, 6.5, 7.2, 0.9, 0.0)
        td[1] = ramp(t, 7.0, 8.3, 0.0, 0.9)
    elif t < 10.0:
        phase = "回中"
        pref = A.P_STEP
        td[1] = ramp(t, 8.5, 9.3, 0.9, 0.0)
    else:
        phase = "夹爪闭合"
        pref = A.P_STEP
        grip = ramp(t, 10.2, 11.4, 0.0, 1.0)
    return td, grip, pref, phase


T_END = 12.0


def camera_at(cam, t):
    u = min(max(t / T_END, 0.0), 1.0)
    cam.azimuth = 150.0 - 20.0 * u
    cam.elevation = -4.0 + 2.5 * math.sin(math.pi * u)
    cam.distance = 1.25 - 0.08 * u
    cam.lookat[:] = (0.0, 0.02, 1.74 + 0.02 * u)


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
    cam.distance = 1.25
    cam.azimuth = 150.0
    cam.elevation = -4.0
    cam.lookat[:] = (0.0, 0.02, 1.74)
    font_big = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
    font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - 0.12 * np.clip(rr - 0.50, 0, 1) / 0.50, 0, 1).astype(np.float32)
    proc = None
    if not stills:
        video_path = video_path or (OUT_DIR / "arm_lattice.mp4")
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
    shot_at = [(1.0, "hang"), (3.6, "extend"), (5.5, "bend"), (10.8, "grip")]
    shots_done = set()
    pneu = A.Pneumatic()
    hist = []
    for k in range(n):
        t = (k + 1) * dt
        td, grip, pref, phase = timeline(t)
        pneu.step(pref, dt)
        set_ctrl(model, data, td, grip, pneu.stiffness(), pneu.P)
        mujoco.mj_step(model, data)
        hist.append((t, pneu.P, pref, pneu.pwm, pneu.valve))
        if stills:
            for ts, tag in shot_at:
                if tag not in shots_done and t >= ts:
                    shots_done.add(tag)
                    renderer.update_scene(data, cam, opt)
                    Image.fromarray(renderer.render().copy()).save(HERE / ("lattice_%s.png" % tag))
                    print("still saved", tag)
        if not stills and k % spf == 0:
            camera_at(cam, t)
            renderer.update_scene(data, cam, opt)
            im = Image.fromarray(renderer.render().copy())
            dr = ImageDraw.Draw(im, "RGBA")
            l1 = "%s · t=%.1f s" % (phase, t)
            l2 = "真机构型重建 · 反棱柱桁架链 8 节 · 气动伸缩 5 倍 · 腱驱弯曲 · 红蓝两指夹爪"
            pad, lh = 18, 40
            wpx = max(dr.textlength(s2, font=f) for s2, f in ((l1, font_big), (l2, font_mid)))
            bw = int(wpx) + 2 * pad
            bh = 2 * lh + 2 * pad - 6
            x0, y0 = 26, H - 26 - bh
            dr.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=14,
                                 fill=(12, 13, 16, 150), outline=(255, 255, 255, 30), width=1)
            dr.text((x0 + pad, y0 + pad - 4), l1, font=font_big, fill=(235, 238, 242, 255))
            dr.text((x0 + pad, y0 + pad + lh - 4), l2, font=font_mid, fill=(190, 196, 205, 255))
            A.draw_pneu(im, hist)
            graded = np.clip((np.asarray(im, dtype=np.float32) / 255.0 - 0.5) * 1.05 + 0.5, 0, 1) * 255.0
            im = Image.fromarray((graded * vig[..., None]).astype(np.uint8))
            proc.stdin.write(np.ascontiguousarray(np.asarray(im)).tobytes())
            frame += 1
            if frame % 150 == 0:
                print("  t=%.1f %s" % (t, phase))
    if stills:
        renderer.close()
        return 0
    proc.stdin.close()
    proc.wait()
    renderer.close()
    print("wrote %s (%.1f MB, %.1f s)" % (video_path, Path(video_path).stat().st_size / 1e6, frame / FPS))
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
