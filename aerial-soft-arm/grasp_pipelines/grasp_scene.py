"""Soft-Spiral 抓取场景与编排共享模块（三套脚本共用）。

- 场景：12 节螺旋连续体软臂 + 篮筐 + 双立柱 + 横杆哑铃（标定位置）。
- 编排：写死版与视觉版共用的六阶段控制序列。
- 相机：宽景/特写参数。

依赖：mujoco（3.x）、Pillow；网格资源在同目录 spiral_repo/。
"""
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / "spiral_repo"
# MuJoCo 的 C++ 文件打开不支持中文路径，网格与场景 XML 统一放到 ASCII 运行时目录
RUNTIME = Path(tempfile.gettempdir()) / "spiral_grasp_runtime"


def ensure_assets():
    """把 spiral_repo 的网格复制到 ASCII 运行时目录（只复制一次）。"""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for sub in ("3D", "3D_Gripper"):
        src, dst = REPO / sub, RUNTIME / sub
        if not dst.exists():
            shutil.copytree(src, dst)
    return RUNTIME

# 哑铃杆标定位置（三轮迭代结果，见组内文档）
BAR = (0.0, 0.3819, 0.1154)
POST_X = 0.025
POST_R = 0.0025
POST_BASE = 0.02
ROD_HALF = 0.040
ROD_R = 0.003
BALL_X = 0.043
BALL_R = 0.009


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def fmt_rgba(rgb):
    return "%.3f %.3f %.3f 1" % rgb


def build_scene(bar=BAR, bar_material="mat_ball", out="grasp_scene_build.xml"):
    """生成抓取场景 XML（棚拍材质 + 立柱 + 哑铃）到 ASCII 运行时目录，返回路径。"""
    ensure_assets()
    s = (REPO / "Long_Manipulator+Gripper.xml").read_text(encoding="utf-8")

    vis = (
        '<visual><quality shadowsize="8192" offsamples="16"/>'
        '<headlight ambient="0.18 0.19 0.22" diffuse="0.30 0.30 0.32" specular="0.06 0.06 0.06"/>'
        '<global offwidth="1920" offheight="1080"/></visual>'
    )
    s = s.replace('<option gravity="0 0 0"/>', '<option gravity="0 0 0"/>' + vis)

    s = s.replace(
        '<texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>',
        '<texture name="texplane" type="2d" builtin="checker" rgb1=".205 .215 .235" rgb2=".125 .132 .145" width="512" height="512" mark="cross" markrgb=".55 .57 .6"/>',
    )
    s = s.replace(
        '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />',
        '<texture type="skybox" builtin="gradient" rgb1="0.42 0.46 0.53" rgb2="0.02 0.03 0.05" width="512" height="3072" />',
    )
    s = s.replace(
        '<material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>',
        '<material name="matplane" reflectance="0.24" shininess="0.25" roughness="0.60" texture="texplane" texrepeat="2 2" texuniform="true"/>',
    )
    extra_mats = []
    c0, c1 = (0.115, 0.145, 0.225), (0.155, 0.62, 0.78)
    for i in range(1, 13):
        rgb = lerp(c0, c1, (i - 1) / 11.0)
        extra_mats.append(
            '<material name="mat_seg_%02d" rgba="%s" specular="0.55" shininess="0.55" reflectance="0.07" roughness="0.32"/>'
            % (i, fmt_rgba(rgb))
        )
    extra_mats += [
        '<material name="mat_grip" rgba="0.93 0.62 0.24 1" specular="0.75" shininess="0.55" roughness="0.30"/>',
        '<material name="mat_grip2" rgba="0.88 0.58 0.22 1" specular="0.75" shininess="0.55" roughness="0.30"/>',
        '<material name="mat_gripbase" rgba="0.22 0.24 0.28 1" specular="0.6" shininess="0.5" roughness="0.38"/>',
        '<material name="mat_platform" rgba="0.33 0.35 0.39 1" specular="0.5" shininess="0.45" roughness="0.40"/>',
        '<material name="mat_holder" rgba="0.42 0.45 0.50 1" specular="0.6" shininess="0.5" roughness="0.35"/>',
        '<material name="mat_ball" rgba="0.95 0.42 0.10 1" specular="0.85" shininess="0.7" roughness="0.22"/>',
        '<material name="mat_basket" rgba="0.36 0.24 0.155 1" specular="0.35" shininess="0.3" roughness="0.55"/>',
        '<material name="mat_target" rgba="0.10 0.85 0.25 1" specular="0.55" shininess="0.5" roughness="0.35"/>',
    ]
    s = s.replace("</asset>", "".join(extra_mats) + "</asset>")

    s = s.replace('<geom type="mesh" mesh="segment_mesh" rgba="0.9 0.3 0.3 1"/>',
                  '<geom type="mesh" mesh="segment_mesh" material="mat_seg_01"/>')
    s = re.sub(r'(<geom type="mesh" mesh="segment_(\d+)_mesh")[^/]*/>',
               lambda m: '%s material="mat_seg_%02d"/>' % (m.group(1), int(m.group(2))), s)
    s = s.replace('<geom type="mesh" mesh="platform_mesh" rgba="0.6 0.6 0.6 1"/>',
                  '<geom type="mesh" mesh="platform_mesh" material="mat_platform"/>')
    s = s.replace('<geom type="mesh" mesh="base_mesh" rgba="0.3 0.3 0.8 1"/>',
                  '<geom type="mesh" mesh="base_mesh" material="mat_holder"/>')
    s = s.replace('<geom type="mesh" mesh="gripper_base" rgba="0.6 0.6 0.6 1"/>',
                  '<geom type="mesh" mesh="gripper_base" material="mat_gripbase"/>')
    s = s.replace('<geom type="mesh" mesh="gripper" rgba="0.9 0.9 0.9 1"',
                  '<geom type="mesh" mesh="gripper" material="mat_grip"')
    s = s.replace('<geom type="mesh" mesh="gripper" rgba="1 1 1 1"',
                  '<geom type="mesh" mesh="gripper" material="mat_grip2"')
    s = re.sub(r'(<geom name="basket[^"]*" type="box"[^/]*rgba="0\.5 0\.3 0\.1 1")',
               lambda m: m.group(1).replace('rgba="0.5 0.3 0.1 1"', 'material="mat_basket"'), s)

    old_light = re.search(r'<light pos="1 0.5 0.7"[^/]*/>\s*', s).group(0)
    lights = (
        '<light name="key" directional="true" pos="0.9 -0.6 1.4" dir="-0.55 0.45 -1" '
        'diffuse="0.80 0.78 0.74" specular="0.55 0.55 0.55" castshadow="true"/>'
        '<light name="fill" directional="true" pos="-1.1 0.5 0.7" dir="0.65 -0.35 -1" '
        'diffuse="0.24 0.27 0.33" specular="0.05 0.05 0.05"/>'
        '<light name="rim" directional="true" pos="0.2 1.3 0.8" dir="-0.15 -0.95 -0.6" '
        'diffuse="0.34 0.34 0.40" specular="0.12 0.12 0.12"/>'
    )
    s = s.replace(old_light, lights)

    s = re.sub(r'<body name="test_ball".*?</body>\s*', "", s, flags=re.S)

    post_half = (bar[2] - ROD_R - 0.001 - POST_BASE) / 2.0
    post_cz = POST_BASE + post_half
    obj = """
    <body name="post_l" pos="-%.4f %.4f %.4f">
      <geom name="pl_geom" type="cylinder" size="%.4f %.4f" mass="0.3"
            friction="2 0.01 0.001" material="mat_gripbase"/>
    </body>
    <body name="post_r" pos="%.4f %.4f %.4f">
      <geom name="pr_geom" type="cylinder" size="%.4f %.4f" mass="0.3"
            friction="2 0.01 0.001" material="mat_gripbase"/>
    </body>
    <body name="dumbbell" pos="%.4f %.4f %.4f">
      <joint type="free" damping="0.7"/>
      <geom name="rod_geom" type="capsule" fromto="-%.4f 0 0 %.4f 0 0" size="%.4f"
            mass="0.006" friction="4 0.01 0.001" condim="6"
            solimp="0.9 0.95 0.001" solref="0.02 1" material="%s"/>
      <geom name="ball_l" type="sphere" size="%.4f" pos="-%.4f 0 0"
            mass="0.005" friction="4 0.01 0.001" condim="6"
            solimp="0.9 0.95 0.001" solref="0.02 1" material="mat_basket"/>
      <geom name="ball_r" type="sphere" size="%.4f" pos="%.4f 0 0"
            mass="0.005" friction="4 0.01 0.001" condim="6"
            solimp="0.9 0.95 0.001" solref="0.02 1" material="mat_basket"/>
    </body>
""" % (POST_X, bar[1], post_cz, POST_R, post_half,
       POST_X, bar[1], post_cz, POST_R, post_half,
       bar[0], bar[1], bar[2],
       ROD_HALF, ROD_HALF, ROD_R, bar_material,
       BALL_R, BALL_X, BALL_R, BALL_X)
    s = s.replace("</worldbody>", obj + "</worldbody>")
    p = RUNTIME / out
    p.write_text(s, encoding="utf-8")
    return p


def add_wrist_camera(xml_path, quat="0 0.7071 -0.7071 0", pos="0 0 0.055", fovy=58.0):
    """在 gripper_base 上装腕部相机（沿手指方向前视）。就地修改 XML。"""
    s = xml_path.read_text(encoding="utf-8")
    cam = '<camera name="wrist" pos="%s" quat="%s" fovy="%.1f"/>' % (pos, quat, fovy)
    s = s.replace('<body name="gripper_base" pos="0 0 0" euler="0 3.14 0">',
                  '<body name="gripper_base" pos="0 0 0" euler="0 3.14 0">' + cam)
    xml_path.write_text(s, encoding="utf-8")


# ---------------- 写死版编排 ----------------
T_REST_END = 1.0
T_DIVE = 2.0
T_SETTLE = 2.5
T_CLOSE = 0.3
CLOSE_LEVEL = 0.33
T_CLOSE_HOLD = 1.2
T_LIFT = 6.0
T_HOLD = 8.0
T_END = T_REST_END + T_DIVE + T_SETTLE + T_CLOSE + T_CLOSE_HOLD + T_LIFT + T_HOLD

PHASES = [
    (0.0, T_REST_END, "静止 REST"),
    (T_REST_END, T_REST_END + T_DIVE, "下探 DIVE"),
    (T_REST_END + T_DIVE, T_REST_END + T_DIVE + T_SETTLE, "就位 ALIGN"),
    (T_REST_END + T_DIVE + T_SETTLE, T_REST_END + T_DIVE + T_SETTLE + T_CLOSE + T_CLOSE_HOLD, "闭合 PINCH"),
    (T_REST_END + T_DIVE + T_SETTLE + T_CLOSE + T_CLOSE_HOLD,
     T_REST_END + T_DIVE + T_SETTLE + T_CLOSE + T_CLOSE_HOLD + T_LIFT, "夹持提升 LIFT"),
    (T_REST_END + T_DIVE + T_SETTLE + T_CLOSE + T_CLOSE_HOLD + T_LIFT, T_END, "稳定保持 HOLD"),
]


def phase_of(t):
    for t0, t1, name in PHASES:
        if t < t1:
            return name
    return PHASES[-1][2]


def controls(t):
    """写死版控制序列 (a1, a2, a3, a4)。"""
    if t < T_REST_END:
        return 0.0, 0.0, 0.0, 0.0
    t1 = t - T_REST_END
    if t1 < T_DIVE:
        u = 0.5 - 0.5 * np.cos(np.pi * t1 / T_DIVE)
        return u, 0.0, 0.0, 0.0
    t2 = t1 - T_DIVE
    if t2 < T_SETTLE:
        return 1.0, 0.0, 0.0, 0.0
    t3 = t2 - T_SETTLE
    if t3 < T_CLOSE:
        v = CLOSE_LEVEL * (t3 / T_CLOSE)
        return 1.0, 0.0, v, v
    t4 = t3 - T_CLOSE
    if t4 < T_CLOSE_HOLD:
        return 1.0, 0.0, CLOSE_LEVEL, CLOSE_LEVEL
    t5 = t4 - T_CLOSE_HOLD
    if t5 < T_LIFT:
        u = 0.5 - 0.5 * np.cos(np.pi * t5 / T_LIFT)
        lev = CLOSE_LEVEL + (1.0 - CLOSE_LEVEL) * u
        return 1.0, u, lev, lev
    return 1.0, 1.0, 1.0, 1.0


def wide_cam(t):
    s = min(max((t - 7.0) / 10.0, 0.0), 1.0)
    return dict(distance=1.78 - 0.12 * s, azimuth=129.0 + 5.0 * s,
                elevation=-10.0 - 4.0 * s,
                lookat=(0.0, 0.30 + 0.24 * s, 0.30 + 0.14 * s))


def close_cam(lookat):
    return dict(distance=0.52, azimuth=148.0, elevation=-24.0, lookat=tuple(lookat))
