"""从共享参数表生成桁架臂 URDF，供 Isaac Lab convert_urdf 转 USD。

用法
  python urdf_gen.py --gripper 0..3 --out urdf/arm_g0.urdf
  python urdf_gen.py --all           生成四款
"""
import argparse
import math
from pathlib import Path

import numpy as np

import arm_params as P

HERE = Path(__file__).resolve().parent
MESH_DIR = (HERE / ".." / "arm_real" / "meshes").resolve()
MESH_REL = "../../arm_real/meshes"


def mesh_volume(path):
    """用散度定理算闭合网格体积。"""
    vs, fs = [], []
    for line in path.read_text(encoding="ascii").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            vs.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(x.split("/")[0]) - 1 for x in t[1:]]
            for k in range(1, len(idx) - 1):
                fs.append((idx[0], idx[k], idx[k + 1]))
    v = np.array(vs)
    vol = 0.0
    for a, b, c in fs:
        vol += float(np.dot(np.cross(v[a], v[b]), v[c])) / 6.0
    return abs(vol)


def mesh_aabb(path):
    vs = []
    for line in path.read_text(encoding="ascii").splitlines():
        t = line.split()
        if t and t[0] == "v":
            vs.append([float(x) for x in t[1:4]])
    v = np.array(vs)
    return v.min(0), v.max(0)


def link_mass(visuals):
    """按各网格体积与安装位置合成质量、质心与惯量（平行轴）。"""
    parts = []
    total_m = 0.0
    total_v = 0.0
    com = np.zeros(3)
    for v in visuals:
        nm = v[0]
        pos = np.array([float(x) for x in v[3].split()], dtype=float)
        p = MESH_DIR / (nm + ".obj")
        if not p.exists():
            continue
        vol = mesh_volume(p)
        m = max(vol * P.DENSITY, 1e-5)
        mn, mx = mesh_aabb(p)
        c = (mn + mx) / 2.0 + pos
        h = np.maximum((mx - mn) / 2.0, 0.004)
        parts.append((m, c, h))
        total_m += m
        total_v += vol
        com += m * c
    if total_m <= 0.0:
        return 1e-4, 0.0, np.array([4e-6, 4e-6, 4e-6]), np.zeros(3)
    com = com / total_m
    inertia = np.zeros(3)
    for m, c, h in parts:
        i_loc = m * np.array([h[1] ** 2 + h[2] ** 2, h[0] ** 2 + h[2] ** 2, h[0] ** 2 + h[1] ** 2]) / 3.0
        d = c - com
        inertia += i_loc + m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return max(total_m, 1e-4), total_v, inertia, com


def box_inertia(m, hx, hy, hz):
    return (m * (hy * hy + hz * hz) / 3.0,
            m * (hx * hx + hz * hz) / 3.0,
            m * (hx * hx + hy * hy) / 3.0)


def inertial(mass, inertia, com):
    ixx, iyy, izz = inertia
    return ('    <inertial><origin xyz="%.4f %.4f %.4f"/><mass value="%.5f"/>'
            '<inertia ixx="%.6e" ixy="0" ixz="0" iyy="%.6e" iyz="0" izz="%.6e"/></inertial>'
            % (com[0], com[1], com[2], mass, ixx, iyy, izz))


def visual(mesh, rgb=(0.44, 0.46, 0.49), scale=None, xyz="0 0 0", rpy="0 0 0"):
    sc = ' scale="%.4f %.4f %.4f"' % scale if scale else ""
    return ('    <visual><origin xyz="%s" rpy="%s"/><geometry><mesh filename="%s/%s.obj"%s/></geometry>'
            '<material name="m_%s_%d"><color rgba="%.3f %.3f %.3f 1"/></material></visual>'
            % (xyz, rpy, MESH_REL, mesh, sc, mesh, 0, rgb[0], rgb[1], rgb[2]))


def drone_visuals(tray=False):
    """整机机架与航电的视觉件清单，坐标为机体系。"""
    v = []
    v.append(("q_plate", (0.16, 0.16, 0.17), None, "0 0 0.0012", "0 0 0"))
    v.append(("q_plate", (0.16, 0.16, 0.17), None, "0 0 0.036", "0 0 0"))
    for a_deg in (45, 135, 225, 315):
        a = math.radians(a_deg)
        x, y = 0.115 * math.cos(a), 0.115 * math.sin(a)
        v.append(("q_arm_boom", (0.13, 0.13, 0.14), None, "%.4f %.4f 0.018" % (x, y), "0 0 %.4f" % a))
        mx, my = 0.185 * math.cos(a), 0.185 * math.sin(a)
        v.append(("q_motor", (0.30, 0.30, 0.32), None, "%.4f %.4f 0.020" % (mx, my), "0 0 0"))
    for sx in (-1, 1):
        for sy in (-1, 1):
            v.append(("q_standoff", (0.55, 0.56, 0.58), None, "%.4f %.4f 0.004" % (0.036 * sx, 0.036 * sy), "0 0 0"))
    v.append(("q_battery", (0.10, 0.10, 0.12), None, "0 0.006 0.038", "0 0 0"))
    v.append(("q_pdb", (0.12, 0.30, 0.16), None, "0 -0.024 0.002", "0 0 0"))
    v.append(("q_fc", (0.12, 0.13, 0.14), None, "0 0.004 0.009", "0 0 0"))
    v.append(("q_obc", (0.24, 0.25, 0.27), None, "0 0.020 0.038", "0 0 3.1416"))
    v.append(("q_vtx", (0.16, 0.16, 0.18), None, "-0.030 -0.030 0.038", "0 0 0.7854"))
    v.append(("q_rx", (0.14, 0.14, 0.15), None, "0.030 0.030 0.038", "0 0 2.36"))
    v.append(("q_mast", (0.55, 0.56, 0.58), None, "-0.030 0.032 0.036", "0 0 0"))
    v.append(("q_gnss", (0.80, 0.80, 0.82), None, "-0.030 0.032 0.076", "0 0 0"))
    v.append(("q_cam_front", (0.10, 0.10, 0.12), None, "0.046 0.0 0.024", "0 0 0"))
    v.append(("q_landing", (0.12, 0.12, 0.13), None, "0 0 0.0", "0 0 0"))
    if tray:
        v.append(("q_tray", (0.30, 0.32, 0.34), None, "0 0 0.082", "0 0 0"))
    else:
        v.append(("q_cam_down", (0.10, 0.10, 0.12), None, "0.020 0.020 -0.010", "0 0 3.1416"))
    return v


def build(gripper_index, out_path, airframe=None):
    gd = P.GRIPPER_DEFS[gripper_index] if gripper_index is not None else None
    L = []
    L.append('<?xml version="1.0"?>')
    L.append('<robot name="as01_truss">')
    L.append('  <material name="steel"><color rgba="0.62 0.64 0.68 1"/></material>')
    L.append('  <material name="dark"><color rgba="0.09 0.09 0.10 1"/></material>')
    L.append('  <material name="blue"><color rgba="0.10 0.24 0.60 1"/></material>')
    L.append('  <material name="red"><color rgba="0.72 0.10 0.09 1"/></material>')
    L.append('  <material name="silicone"><color rgba="0.82 0.83 0.85 1"/></material>')

    masses = {}

    def add_link(name, visuals, inertial_freeze=None):
        m, vol, inertia, com = link_mass(visuals)
        if inertial_freeze is not None:
            m = inertial_freeze["mass"]
            com = np.asarray(inertial_freeze["com"], dtype=float)
            inertia = np.asarray(inertial_freeze["inertia"], dtype=float)
        masses[name] = (m, vol)
        L.append('  <link name="%s">' % name)
        L.append(inertial(m, inertia, com))
        for v in visuals:
            L.append(visual(*v))
        L.append('  </link>')

    def add_joint(name, jtype, parent, child, xyz, axis, limits, rpy="0 0 0"):
        L.append('  <joint name="%s" type="%s">' % (name, jtype))
        L.append('    <origin xyz="%s" rpy="%s"/>' % (xyz, rpy))
        L.append('    <parent link="%s"/><child link="%s"/>' % (parent, child))
        L.append('    <axis xyz="%s"/>' % axis)
        L.append('    <limit lower="%.4f" upper="%.4f" effort="%.1f" velocity="2.0"/>'
                 % (limits[0], limits[1], P.DRIVES["bend"]["effort"]))
        L.append('  </joint>')

    def add_props():
        for i, a_deg in enumerate((45, 135, 225, 315)):
            a = math.radians(a_deg)
            mx, my = 0.185 * math.cos(a), 0.185 * math.sin(a)
            add_link("prop_%d" % i, [("q_prop", (0.14, 0.14, 0.15), None, "0 0 0", "0 0 %.4f" % (a_deg * 1.7))])
            L.append('  <joint name="prop_%d" type="continuous">'
                     '<origin xyz="%.4f %.4f 0.047" rpy="0 0 0"/>'
                     '<parent link="base_link"/><child link="prop_%d"/><axis xyz="0 0 1"/>'
                     '<limit effort="0.5" velocity="250"/></joint>' % (i, mx, my, i))

    if airframe == "a":
        add_link("base_link", drone_visuals(tray=False) + [
            ("base_flange", (0.62, 0.64, 0.68), None, "0 0 -0.008", "3.1416 0 0"),
            ("truss_base_ring", (0.62, 0.64, 0.68), None, "0 0 -0.015", "3.1416 0 0"),
            ("truss_tensioners", (0.78, 0.62, 0.30), None, "0 0 -0.018", "3.1416 0 0"),
        ])
        add_props()
        arm_drop_extra = 0.010
    elif airframe == "b":
        add_link("base_link", drone_visuals(tray=True))
        add_props()
        add_link("b_mast", [("q_mast", (0.62, 0.64, 0.68), None, "0 0 0", "0 0 0")])
        L.append('  <link name="b_tray_c">'
                 '<inertial><origin xyz="0 0 0"/><mass value="0.001"/>'
                 '<inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>'
                 '<collision><origin xyz="0 0 0.084"/><geometry><box size="0.110 0.090 0.010"/></geometry></collision>'
                 '</link>')
        L.append('  <joint name="mount_tray" type="fixed">'
                 '<origin xyz="0 0 0" rpy="0 0 0"/>'
                 '<parent link="base_link"/><child link="b_tray_c"/></joint>')
        L.append('  <joint name="mast_j" type="revolute">'
                 '<origin xyz="-0.030 0.032 0.112" rpy="0 0 0"/>'
                 '<parent link="base_link"/><child link="b_mast"/><axis xyz="0 1 0"/>'
                 '<limit lower="-0.05" upper="0.05" effort="0.5" velocity="1.0"/></joint>')
        L.append('</robot>')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(L), encoding="utf-8")
        total = sum(m for m, _ in masses.values())
        print("[urdf] %s  纯机  总质量 %.4f kg" % (out_path.name, total))
        return masses
    else:
        add_link("base_link", [("base_flange", (0.62, 0.64, 0.68), None, "0 0 -0.0055", "3.1416 0 0"),
                               ("truss_base_ring", (0.62, 0.64, 0.68), None, "0 0 -0.0125", "3.1416 0 0"),
                               ("truss_tensioners", (0.78, 0.62, 0.30), None, "0 0 -0.0155", "3.1416 0 0")])
        arm_drop_extra = 0.0

    prev = "base_link"
    for i in range(P.N_SEG):
        seg_link = "seg_%d" % i
        bx_link = "bx_%d" % i
        by_link = "by_%d" % i
        zr = 0.0698 if i % 2 == 1 else 0.0
        rot = 0.0 if i % 2 == 0 else 0.7854
        if i > 0:
            sd_link = "slide_%d" % i
            add_link(sd_link, [])
            L.append('  <joint name="slide_%d" type="prismatic">'
                     '<origin xyz="0 0 %.4f" rpy="0 0 0"/>'
                     '<parent link="%s"/><child link="%s"/><axis xyz="0 0 1"/>'
                     '<limit lower="%.4f" upper="%.4f" effort="%.1f" velocity="0.5"/></joint>'
                     % (i, -P.SEG_DROP, prev, sd_link, P.SLIDE_RANGE[0], P.SLIDE_RANGE[1],
                        P.DRIVES["slide"]["effort"]))
            prev = sd_link
        drop = (P.SEG0_DROP if i == 0 else 0.0) + (arm_drop_extra if i == 0 else 0.0)
        add_link(bx_link, [])
        add_joint("bx_%d" % i, "revolute", prev, bx_link, "0 0 %.4f" % (-drop), "1 0 0", P.BX_RANGE)
        add_link(by_link, [])
        add_joint("by_%d" % i, "revolute", bx_link, by_link, "0 0 0", "0 1 0", P.BX_RANGE)
        add_link(seg_link, [
            ("lattice_" + P.SPAN_MODULE, (0.44, 0.46, 0.49), None, "0 0 0", "3.1416 0 %.4f" % rot),
            ("sheath_" + P.SPAN_MODULE, (0.09, 0.09, 0.10), None, "0 0 0", "3.1416 0 %.4f" % zr),
            ("bundle_" + P.SPAN_MODULE, (0.16, 0.16, 0.18), None, "0 0 0", "3.1416 0 %.4f" % zr),
            ("truss_clamp", (0.62, 0.64, 0.68), None, "0 0 0", "3.1416 0 %.4f" % zr),
        ], inertial_freeze=P.SEG_INERTIAL)
        L.append('  <joint name="mount_%d" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/>'
                 '<parent link="%s"/><child link="%s"/></joint>' % (i, by_link, seg_link))
        prev = seg_link

    add_link("wrist_slide", [])
    L.append('  <joint name="slide_wrist" type="prismatic">'
             '<origin xyz="0 0 %.4f" rpy="0 0 0"/>'
             '<parent link="%s"/><child link="wrist_slide"/><axis xyz="0 0 1"/>'
             '<limit lower="%.4f" upper="%.4f" effort="%.1f" velocity="0.5"/></joint>'
             % (-P.WRIST_DROP, prev, P.SLIDE_RANGE[0], P.SLIDE_RANGE[1],
                P.DRIVES["slide"]["effort"]))
    add_link("wrist", [("grip_base", (0.11, 0.11, 0.12), None, "0 0 0", "0 0 0"),
                       ("grip_links", (0.11, 0.11, 0.12), None, "0 0 0", "0 0 0")])
    L.append('  <joint name="mount_wrist" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/>'
             '<parent link="wrist_slide"/><child link="wrist"/></joint>')

    for k, a in enumerate(gd["angs"]):
        m = "k" if gd["key"] == "diamond" else "s"
        fl = "finger_%d" % k
        if gd["single"]:
            vis = [(gd["single"], (0.82, 0.83, 0.85) if gd["key"] == "soft" else (0.62, 0.64, 0.68), None, "0 0 0", "0 0 0")]
        else:
            prox_rgb = (0.10, 0.24, 0.60) if k % 2 == 0 else (0.72, 0.10, 0.09)
            dist_rgb = (0.72, 0.10, 0.09) if k % 2 == 0 else (0.10, 0.24, 0.60)
            vis = [(gd["prox"], prox_rgb, None, "0 0 0", "0 0 0"),
                   (gd["dist"], dist_rgb, None, "0 0 0", "0 0 0")]
        add_link(fl, vis)
        x, y = P.FINGER_HINGE_R * math.cos(a), P.FINGER_HINGE_R * math.sin(a)
        L.append('  <joint name="grip_%d" type="revolute">'
                 '<origin xyz="%.4f %.4f %.4f" rpy="0 0 %.4f"/>'
                 '<parent link="wrist"/><child link="%s"/>'
                 '<axis xyz="0 1 0"/>'
                 '<limit lower="%.4f" upper="%.4f" effort="%.1f" velocity="3.0"/></joint>'
                 % (k, x, y, P.FINGER_HINGE_Z, a, fl, P.GRIP_RANGE[0], P.GRIP_RANGE[1],
                    P.DRIVES["grip"]["effort"]))
        hx, hy, hz, px, pz = gd["col"]
        L.append('  <link name="pad_%d"><collision><origin xyz="%.4f 0 %.4f"/>'
                 '<geometry><box size="%.4f %.4f %.4f"/></geometry></collision></link>'
                 % (k, px, pz, 2 * hx, 2 * hy, 2 * hz))
        L.append('  <joint name="mount_pad_%d" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/>'
                 '<parent link="%s"/><child link="pad_%d"/></joint>' % (k, fl, k))

    L.append('</robot>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")

    total = sum(m for m, _ in masses.values())
    print("[urdf] %s  关节 %d  总质量 %.4f kg" % (out_path.name, len([x for x in L if "<joint" in x]), total))
    return masses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", type=int, default=None)
    ap.add_argument("--airframe", default=None, choices=("a", "b"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.airframe == "b":
        build(None, HERE / "urdf" / "drone_b.urdf", airframe="b")
        return
    if args.airframe == "a":
        for i, gd in enumerate(P.GRIPPER_DEFS):
            build(i, HERE / "urdf" / ("drone_a_%s.urdf" % gd["key"]), airframe="a")
        return
    if args.all:
        for i, gd in enumerate(P.GRIPPER_DEFS):
            masses = build(i, HERE / "urdf" / ("arm_%s.urdf" % gd["key"]))
            for nm, (m, vol) in sorted(masses.items()):
                if m > 1e-3:
                    print("   %-12s %.5f kg  vol %.6f m3" % (nm, m, vol))
    else:
        i = 0 if args.gripper is None else args.gripper
        out = Path(args.out) if args.out else HERE / "urdf" / ("arm_%s.urdf" % P.GRIPPER_DEFS[i]["key"])
        build(i, out)


if __name__ == "__main__":
    main()
