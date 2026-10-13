"""AS-01 桁架臂共享参数表，Isaac 侧的唯一数值来源。

数值与 MuJoCo 版 `arm_real/arm_lattice.py` 一一对应，改这里两边同步。
不依赖 mujoco，Isaac 环境可直接导入。
"""

N_SEG = 8
R_RING = 0.028
SPAN_MIN = 0.010
SPAN_MAX = 0.050
VARIANT_HEIGHTS = [0.010 + k * 0.004 for k in range(11)]
TENDON_ANGLES = (90.0, 210.0, 330.0)

SEG0_DROP = 0.0155
SEG_DROP = 0.010
WRIST_DROP = 0.010
SLIDE_TRAVEL = 0.040
BX_RANGE = (-0.9, 0.9)
SLIDE_RANGE = (-0.040, 0.0)

# 分段视觉模块：取全行程档 v10（高 0.050 = SEG_DROP + SLIDE_TRAVEL）。
# 旧导出用固定中档 v04（高 0.026），滑移伸出后相邻分段间出现贯通空隙；
# 改用全行程模块后任何滑移状态均衔接（只改 visuals，不动关节与连杆长度）。
SPAN_MODULE = "v10"
# 刚体接口冻结值（2026-10-05 已验收导出）。seg 段视觉换挡后惯量不再随网格体积重算，
# 必须与旧导出逐位一致，保证动力学回归（calib_jac / fc_align）成立。
SEG_INERTIAL = {
    "mass": 0.00884,
    "com": (0.0022, 0.0021, 0.0122),
    "inertia": (3.132782e-06, 3.473547e-06, 5.264165e-06),
}

GRIP_RANGE = (-0.45, 0.20)
GRIP_OPEN_CTRL = -0.40
GRIP_CLOSE_CTRL = 0.30
GRIP_CTRL_PER = 0.70
FINGER_HINGE_R = 0.024
FINGER_HINGE_Z = -0.036

GRIPPER_DEFS = (
    {"name": "菱形两指", "key": "diamond", "angs": (0.0, 3.1416),
     "prox": "finger_prox", "dist": "finger_dist", "single": None,
     "col": (0.0013, 0.0075, 0.025, -0.0062, -0.040)},
    {"name": "平口两指", "key": "flat", "angs": (0.0, 3.1416),
     "prox": None, "dist": None, "single": "finger_flat",
     "col": (0.0013, 0.0075, 0.042, -0.0062, -0.042)},
    {"name": "三指", "key": "three", "angs": (1.5708, 3.6652, 5.7596),
     "prox": "finger_prox", "dist": "finger_dist", "single": None,
     "col": (0.0013, 0.0075, 0.040, -0.0062, -0.050)},
    {"name": "软体两指", "key": "soft", "angs": (0.0, 3.1416),
     "prox": None, "dist": None, "single": "finger_soft",
     "col": (0.0012, 0.0060, 0.036, -0.0093, -0.036)},
)

DENSITY = 1000.0

DRIVES = {
    "slide": {"stiffness": 20000.0, "damping": 200.0, "effort": 400.0},
    "bend": {"stiffness": 2.0, "damping": 8.0, "effort": 8.0},
    "grip": {"stiffness": 12.0, "damping": 0.3, "effort": 4.0},
}

ARM_MOUNT_Z = 2.004
STAND_XY = (0.0, 0.30)
STAND_TABLE_Z = 0.74
STAND_TOP_Z = 1.525
STAND_PLATE_Z = 1.533
PART_Z = 1.565
PART_R = 0.014
PART_H = 0.060
GRASP_P = 0.75
P_STEP_BAR = 1.163
