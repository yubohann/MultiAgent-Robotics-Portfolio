"""程序化生成高细节软臂网格（OBJ），替代原开源模型的低模圆柱。

生成内容：
  1. 波纹管段（带褶皱的硅胶管，逐节渐细，含螺旋切缝示意）
  2. 卡箍环、腱鞘管（周向 120 度三条）
  3. 底座法兰、螺栓、电路板、接插件、电容、线束
  4. 腕部法兰、舵机壳体、舵盘、两指、连杆

运行：python make_meshes.py
输出：meshes/*.obj
"""
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE / "meshes"

N_THETA = 72


def write_obj(path, verts, faces, uvs=None, fuvs=None):
    lines = []
    for v in verts:
        lines.append("v %.6f %.6f %.6f" % (v[0], v[1], v[2]))
    if uvs is not None:
        for uv in uvs:
            lines.append("vt %.6f %.6f" % (uv[0], uv[1]))
    for i, fc in enumerate(faces):
        if uvs is not None and fuvs is not None:
            lines.append("f " + " ".join("%d/%d" % (vi + 1, fuvs[i][k] + 1) for k, vi in enumerate(fc)))
        else:
            lines.append("f " + " ".join(str(vi + 1) for vi in fc))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def merge(*parts):
    """合并多个 (verts, faces) 或 (verts, faces, uvs, fuvs)。"""
    verts, faces, uvs, fuvs = [], [], [], []
    vo, to = 0, 0
    has_uv = any(len(p) > 2 for p in parts)
    for p in parts:
        v = np.asarray(p[0], dtype=float)
        f = np.asarray(p[1], dtype=int)
        verts.append(v)
        faces.append(f + vo)
        if has_uv:
            if len(p) > 2:
                uvs.append(np.asarray(p[2], dtype=float))
                fuvs.append(np.asarray(p[3], dtype=int) + to)
                to += len(p[2])
            else:
                uvs.append(np.zeros((len(v), 2)))
                fuvs.append(np.zeros_like(f))
                to += len(v)
        vo += len(v)
    if has_uv:
        return np.vstack(verts), np.vstack(faces), np.vstack(uvs), np.vstack(fuvs)
    return np.vstack(verts), np.vstack(faces)


def revolve(profile, n_theta=N_THETA, close_bottom=True, close_top=True, uv=True):
    """profile 为 [(r, z), ...]，绕 z 轴旋转。"""
    prof = np.asarray(profile, dtype=float)
    nz = len(prof)
    th = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    verts = []
    for (r, z) in prof:
        for t in th:
            verts.append((r * math.cos(t), r * math.sin(t), z))
    verts = np.array(verts)
    faces = []
    for i in range(nz - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + (j + 1) % n_theta
            d = (i + 1) * n_theta + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if close_bottom and prof[0][0] > 1e-9:
        cidx = len(verts)
        verts = np.vstack([verts, [[0, 0, prof[0][1]]]])
        for j in range(n_theta):
            faces.append((cidx, (j + 1) % n_theta, j))
    if close_top and prof[-1][0] > 1e-9:
        cidx = len(verts)
        verts = np.vstack([verts, [[0, 0, prof[-1][1]]]])
        base = (nz - 1) * n_theta
        for j in range(n_theta):
            faces.append((cidx, base + j, base + (j + 1) % n_theta))
    faces = np.array(faces, dtype=int)
    if uv:
        uvs = []
        for i, (r, z) in enumerate(prof):
            for j in range(n_theta):
                uvs.append((j / n_theta, i / max(nz - 1, 1)))
        uvs = np.array(uvs)
        fuvs = faces.copy()
        return verts, faces, uvs, fuvs
    return verts, faces


def tube_path(points, radius, n_theta=24, caps=True):
    """沿折线路径的圆管。"""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tangents = np.zeros_like(pts)
    for i in range(n):
        if i == 0:
            tangents[i] = pts[1] - pts[0]
        elif i == n - 1:
            tangents[i] = pts[-1] - pts[-2]
        else:
            tangents[i] = pts[i + 1] - pts[i - 1]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    normals = []
    prev = np.cross(tangents[0], ref)
    prev /= np.linalg.norm(prev) + 1e-12
    for i in range(n):
        t = tangents[i]
        nrm = prev - t * np.dot(prev, t)
        if np.linalg.norm(nrm) < 1e-6:
            nrm = np.cross(t, ref)
        nrm /= np.linalg.norm(nrm) + 1e-12
        normals.append(nrm)
        prev = nrm
    th = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    verts = []
    for i in range(n):
        nrm = normals[i]
        binorm = np.cross(tangents[i], nrm)
        for a in th:
            verts.append(pts[i] + radius * (math.cos(a) * nrm + math.sin(a) * binorm))
    verts = np.array(verts)
    faces = []
    for i in range(n - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + (j + 1) % n_theta
            d = (i + 1) * n_theta + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if caps:
        c0 = len(verts)
        verts = np.vstack([verts, [pts[0]]])
        for j in range(n_theta):
            faces.append((c0, (j + 1) % n_theta, j))
        c1 = len(verts)
        verts = np.vstack([verts, [pts[-1]]])
        base = (n - 1) * n_theta
        for j in range(n_theta):
            faces.append((c1, base + j, base + (j + 1) % n_theta))
    return verts, np.array(faces, dtype=int)


def box(sx, sy, sz, pos=(0, 0, 0), euler=(0, 0, 0)):
    x, y, z = sx / 2, sy / 2, sz / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=float)
    f = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=int)
    rx, ry, rz = euler
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    v = v @ (Rz @ Ry @ Rx).T + np.asarray(pos, dtype=float)
    return v, f


def cyl(radius, height, pos=(0, 0, 0), n=24, taper=1.0, euler=(0, 0, 0)):
    prof = [(radius, 0.0), (radius * taper, height)]
    v, f, _, _ = revolve(prof, n_theta=n)
    rx, ry, rz = euler
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    v = v @ (Rz @ Ry @ Rx).T + np.asarray(pos, dtype=float)
    return v, f


def torus(R, r, pos=(0, 0, 0), n_theta=48, n_phi=16, euler=(0.0, 0.0, 0.0)):
    verts, faces = [], []
    for i in range(n_theta):
        a = 2 * math.pi * i / n_theta
        for j in range(n_phi):
            b = 2 * math.pi * j / n_phi
            verts.append(((R + r * math.cos(b)) * math.cos(a),
                          (R + r * math.cos(b)) * math.sin(a),
                          r * math.sin(b)))
    verts = np.array(verts)
    rx, ry, rz = euler
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    verts = verts @ (Rz @ Ry @ Rx).T + np.asarray(pos, dtype=float)
    for i in range(n_theta):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = ((i + 1) % n_theta) * n_phi + (j + 1) % n_phi
            d = ((i + 1) % n_theta) * n_phi + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, np.array(faces, dtype=int)


def hex_head(r, h, pos):
    verts, faces = [], []
    for j in range(2):
        z = j * h
        for i in range(6):
            a = 2 * math.pi * i / 6 + math.pi / 6
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(6):
        a = i
        b = (i + 1) % 6
        faces.append((a, b, 6 + b))
        faces.append((a, 6 + b, 6 + a))
    faces.append((0, 1, 2))
    faces.append((0, 2, 3))
    faces.append((0, 3, 4))
    faces.append((0, 4, 5))
    faces.append((6, 7, 8))
    faces.append((6, 8, 9))
    faces.append((6, 9, 10))
    faces.append((6, 10, 11))
    return np.array(verts) + np.asarray(pos, dtype=float), np.array(faces, dtype=int)


# ---------------- 部件生成 ----------------

def bellows_profile(r0, length, n_pleat, amp, wall, n=240):
    """薄壁波纹管的闭合截面，圆润鼓形褶皱，两端加厚箍。"""
    outer, inner = [], []
    for i in range(n + 1):
        t = i / n
        z = t * length
        s = math.cos(2 * math.pi * n_pleat * t)
        drum = math.tanh(1.25 * s) / math.tanh(1.25)
        collar = 1.0
        if t < 0.09:
            collar = 1.0 + 0.055 * (1.0 - t / 0.09)
        elif t > 0.91:
            collar = 1.0 + 0.055 * ((t - 0.91) / 0.09)
        r = r0 * collar * (1.0 + amp * drum)
        outer.append((r, z))
        inner.append((max(r - wall, r0 * 0.55), z))
    return outer + inner[::-1] + [outer[0]]


def bellows_segment(r0, length, n_pleat=6, amp=0.065, wall=0.0016):
    """薄壁硅胶波纹管段。"""
    prof = bellows_profile(r0, length, n_pleat, amp, wall)
    return revolve(prof, close_bottom=False, close_top=False)


def bellows_segment_inflated(r0, length, n_pleat=4.8, amp=0.115, wall=0.0016):
    """充气态波纹管，褶皱拉开的形态。"""
    prof = bellows_profile(r0 * 1.05, length, n_pleat, amp, wall)
    return revolve(prof, close_bottom=False, close_top=False)


def tendon_tubes(r0, length, tube_r=0.0012):
    parts = []
    for k in range(3):
        a = 2 * math.pi * k / 3 + math.pi / 6
        rr = r0 * 1.075 + tube_r * 0.5
        x, y = rr * math.cos(a), rr * math.sin(a)
        parts.append(tube_path([(x, y, -0.0015), (x, y, length + 0.0015)], tube_r, n_theta=12))
    return merge(*parts)


def clamp_ring(r0, pos_z, tube_r=0.0013):
    return torus(r0 * 1.06 + tube_r * 0.5, tube_r, pos=(0, 0, pos_z))


def base_flange():
    prof = [(0.0, 0.0), (0.034, 0.0), (0.036, 0.002), (0.036, 0.007), (0.033, 0.009), (0.020, 0.010), (0.0, 0.010)]
    return revolve(prof)


def base_bolts():
    parts = []
    for i in range(6):
        a = 2 * math.pi * i / 6
        x, y = 0.030 * math.cos(a), 0.030 * math.sin(a)
        parts.append(cyl(0.0022, 0.0035, pos=(x, y, 0.009), n=12))
        parts.append(hex_head(0.0038, 0.0022, (x, y, 0.0125)))
    return merge(*parts)


def pcb_board():
    return box(0.058, 0.040, 0.0055, pos=(0.0, 0.0, 0.013))


def pcb_chips():
    parts = [box(0.014, 0.009, 0.009, pos=(-0.018, 0.012, 0.020))]
    parts.append(box(0.010, 0.008, 0.008, pos=(-0.018, -0.010, 0.0195)))
    parts.append(box(0.011, 0.006, 0.0026, pos=(0.008, -0.008, 0.0170)))
    parts.append(box(0.006, 0.005, 0.0022, pos=(0.022, -0.006, 0.0168)))
    parts.append(box(0.009, 0.005, 0.0020, pos=(0.010, 0.006, 0.0167)))
    for dx in (0.004, 0.016):
        parts.append(cyl(0.0035, 0.010, pos=(dx, 0.012, 0.0158), n=12))
    return merge(*parts)


def pcb_sensor():
    parts = [box(0.013, 0.009, 0.0014, pos=(-0.006, -0.012, 0.0162))]
    parts.append(torus(0.0045, 0.0012, pos=(0.024, 0.010, 0.0162), n_theta=20, n_phi=8))
    return merge(*parts)


def tensioners():
    parts = []
    for k in range(3):
        a = 2 * math.pi * k / 3 + math.pi / 6
        x, y = 0.052 * math.cos(a), 0.052 * math.sin(a)
        parts.append(cyl(0.0040, 0.010, pos=(x, y, 0.005), n=16))
        parts.append(torus(0.0060, 0.0024, pos=(x, y, 0.012), n_theta=24, n_phi=10))
        parts.append(cyl(0.0050, 0.003, pos=(x, y, 0.017), n=12))
        parts.append(hex_head(0.0034, 0.0020, (x, y, 0.020)))
    return merge(*parts)


def cable_clips():
    parts = []
    for (x0, y0) in ((-0.020, 0.020), (-0.014, 0.024), (-0.026, 0.014)):
        parts.append(torus(0.0042, 0.0012, pos=(x0, y0, 0.021), n_theta=20, n_phi=8))
    return merge(*parts)


def sphere(r, pos=(0, 0, 0), n_theta=16, n_phi=10):
    prof = []
    for i in range(n_phi + 1):
        a = math.pi * i / n_phi
        prof.append((r * math.sin(a), -r * math.cos(a)))
    v, f, _, _ = revolve(prof, n_theta=n_theta)
    return v + np.asarray(pos, dtype=float), f


def _plate(pts, t, shrink_ratio=0.0):
    """按外法线加厚的板件，pts 为多边形顶点（同一平面附近）。"""
    v = [np.asarray(p, dtype=float) for p in pts]
    c = np.mean(v, axis=0)
    if shrink_ratio > 0.0:
        v = [c + (p - c) * (1.0 - shrink_ratio) for p in v]
    n = np.cross(v[1] - v[0], v[2] - v[0])
    n = n / (np.linalg.norm(n) + 1e-12)
    axis_c = np.array([0.0, 0.0, c[2]])
    if np.dot(n, c - axis_c) < 0.0:
        n = -n
    outer = [p + n * (t / 2.0) for p in v]
    inner = [p - n * (t / 2.0) for p in v]
    verts = outer + inner
    k = len(v)
    faces = []
    for i in range(1, k - 1):
        faces.append((0, i, i + 1))
        faces.append((k, k + i + 1, k + i))
    for i in range(k):
        j = (i + 1) % k
        faces.append((i, j, k + j))
        faces.append((i, k + j, k + i))
    return np.array(verts), np.array(faces, dtype=int)


def drive_bundle(r, h, air_r=0.0026):
    """一路主气管的束管，沿模块全高走线并向下多伸 8 毫米搭接下一节，腱鞘另画。"""
    a = math.radians(30.0)
    ra = r * 1.085 + air_r
    x, y = ra * math.cos(a), ra * math.sin(a)
    return tube_path([(x, y, 0.0), (x, y, h + 0.008)], air_r, n_theta=12)


def drive_sheath(r, h, tube_r=0.0015):
    """三根实体腱鞘管，120 度走线，随节高伸缩，两端伸进卡箍导向座。"""
    parts = []
    rr = r * 1.075 + tube_r * 0.8
    for ang in (90.0, 210.0, 330.0):
        a = math.radians(ang)
        x, y = rr * math.cos(a), rr * math.sin(a)
        parts.append(tube_path([(x, y, -0.004), (x, y, h + 0.006)], tube_r, n_theta=12))
    return merge(*parts)


def truss_clamp(r, tube_r=0.0012):
    """节间卡箍环，环体加三个鞘管导向环与三颗锁紧螺钉。"""
    parts = [torus(r * 1.06 + tube_r * 0.5, tube_r, n_theta=56, n_phi=12)]
    rr = r * 1.075 + 0.0015 * 0.8
    for ang in (90.0, 210.0, 330.0):
        a = math.radians(ang)
        x, y = rr * math.cos(a), rr * math.sin(a)
        parts.append(torus(0.0027, 0.0009, pos=(x, y, 0.0), n_theta=18, n_phi=8))
        parts.append(cyl(0.0020, 0.0028, pos=(x * 0.94, y * 0.94, 0.0), n=10, euler=(0.0, 1.5708, a)))
    for ang in (30.0, 150.0, 270.0):
        a = math.radians(ang)
        x, y = r * 1.02 * math.cos(a), r * 1.02 * math.sin(a)
        parts.append(hex_head(0.0017, 0.0011, (x, y, 0.0004)))
    return merge(*parts)


def truss_base_ring(r_in=0.030, r_out=0.050, t=0.004):
    """底座张紧环板，为三组预紧轮提供安装面。"""
    prof = [(r_in, -t / 2), (r_out, -t / 2), (r_out, t / 2), (r_in, t / 2), (r_in, -t / 2)]
    return revolve(prof, n_theta=64, close_bottom=False, close_top=False)


def truss_tensioners(rr=0.041, z=0.0):
    """底座三组腱预紧轮，安装块加轮体、轴、六角头与入线导向环。"""
    parts = []
    for k in range(3):
        a = math.radians(90.0 + 120.0 * k)
        x, y = rr * math.cos(a), rr * math.sin(a)
        parts.append(box(0.016, 0.011, 0.007, pos=(x, y, z - 0.0015), euler=(0.0, 0.0, a)))
        parts.append(torus(0.0062, 0.0022, pos=(x, y, z + 0.0045), n_theta=28, n_phi=10,
                           euler=(1.5708, 0.0, a)))
        parts.append(cyl(0.0022, 0.018, pos=(x, y, z + 0.0045), n=12, euler=(1.5708, 0.0, a)))
        parts.append(hex_head(0.0030, 0.0018, (x * 0.82, y * 0.82, z + 0.0045)))
        parts.append(torus(0.0034, 0.0011, pos=(x, y, z + 0.0090), n_theta=20, n_phi=8))
    return merge(*parts)


def tether_hose():
    pts = [(-0.08, 0.24, 0.81), (-0.17, 0.29, 1.20), (-0.16, 0.25, 1.62),
           (-0.12, 0.15, 1.93), (-0.09, 0.08, 2.006)]
    return tube_path(pts, 0.0042, n_theta=14)


# ---------------- 具身无人机整机与航电 ----------------

def q_plate(t=0.0025, s=0.096):
    """碳板加沉头孔阵列，上下两块共用。"""
    parts = [box(s, s, t)]
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        for rr in (0.030, 0.020):
            x, y = rr * math.cos(a), rr * math.sin(a)
            parts.append(torus(0.0018, 0.0006, pos=(x, y, t / 2 - 0.0002), n_theta=16, n_phi=6))
    return merge(*parts)


def q_arm_boom():
    """碳管机臂，根部加厚，末端电机座。"""
    parts = [box(0.088, 0.013, 0.004, pos=(0.0, 0.0, 0.0))]
    parts.append(box(0.012, 0.020, 0.006, pos=(0.040, 0.0, 0.0)))
    parts.append(cyl(0.0115, 0.003, pos=(0.040, 0.0, 0.0045), n=20))
    return merge(*parts)


def q_motor():
    """外转子无刷电机，定子座加转子钟壳加桨夹。"""
    parts = [cyl(0.0145, 0.004, pos=(0, 0, 0.0), n=24)]
    parts.append(cyl(0.0150, 0.016, pos=(0, 0, 0.004), n=24, taper=0.88))
    for k in range(6):
        a = 2 * math.pi * k / 6
        parts.append(box(0.0022, 0.014, 0.012, pos=(0.0135 * math.cos(a), 0.0135 * math.sin(a), 0.013),
                         euler=(0, 0, a)))
    parts.append(cyl(0.0050, 0.008, pos=(0, 0, 0.020), n=16))
    parts.append(cyl(0.0026, 0.006, pos=(0, 0, 0.027), n=12))
    return merge(*parts)


def q_prop(d=0.254, pitch_deg=14.0):
    """十寸两叶桨，叶片用分段扭转薄板近似。"""
    r0, blade_w = 0.020, 0.028
    parts = [cyl(0.014, 0.006, pos=(0, 0, 0), n=20), cyl(0.011, 0.004, pos=(0, 0, 0.006), n=16)]
    for side in (1, -1):
        seg = 6
        r_prev = r0
        for k in range(seg):
            r_next = r0 + (d / 2 - r0) * (k + 1) / seg
            rm = 0.5 * (r_prev + r_next)
            chord = blade_w * (1.0 - 0.35 * (rm / (d / 2)))
            tw = math.radians(pitch_deg) * (1.0 - 0.55 * (rm / (d / 2))) * side
            parts.append(box(r_next - r_prev, chord, 0.0016, pos=(side * rm, 0.0, 0.008),
                             euler=(tw, 0.0, 0.0)))
            r_prev = r_next
    return merge(*parts)


def q_landing():
    """两点式脚架，两腿加横管。"""
    parts = []
    for sx in (1, -1):
        for sy in (1, -1):
            parts.append(cyl(0.005, 0.10, pos=(0.045 * sx, 0.030 * sy, -0.052), n=14,
                             euler=(0.25 * sy, 0.0, 0.0)))
        parts.append(tube_path([(0.045 * sx, -0.030, -0.102), (0.045 * sx, 0.030, -0.102)], 0.006, n_theta=12))
    return merge(*parts)


def q_battery():
    """6S 5000 锂聚合物电池，电芯分隔线、标签、XT60 与平衡头。"""
    parts = [box(0.135, 0.048, 0.042, pos=(0, 0, 0.021))]
    for i in range(5):
        parts.append(box(0.132, 0.0022, 0.003, pos=(0, -0.024 + 0.0095 * (i + 1), 0.042)))
    parts.append(box(0.060, 0.020, 0.0016, pos=(-0.020, 0, 0.0435)))
    parts.append(cyl(0.0032, 0.010, pos=(0.050, 0.010, 0.024), n=12, euler=(0, 1.5708, 0)))
    parts.append(box(0.012, 0.008, 0.005, pos=(0.058, 0.010, 0.028)))
    parts.append(box(0.020, 0.010, 0.005, pos=(-0.058, 0.012, 0.026)))
    for i in range(4):
        parts.append(cyl(0.0012, 0.008, pos=(-0.058, -0.002 - 0.004 * i, 0.030), n=8))
    return merge(*parts)


def q_pdb():
    """电源管理板，铜排、电容组、XT60、电流计。"""
    parts = [box(0.050, 0.050, 0.004, pos=(0, 0, 0.002))]
    parts.append(box(0.008, 0.040, 0.003, pos=(-0.016, 0, 0.0055)))
    parts.append(box(0.008, 0.040, 0.003, pos=(0.016, 0, 0.0055)))
    for sx in (-0.026, 0.026):
        for i in range(2):
            parts.append(cyl(0.006, 0.014, pos=(sx, -0.014 + 0.028 * i, 0.011), n=18))
    parts.append(box(0.016, 0.010, 0.008, pos=(0.0, -0.028, 0.006)))
    parts.append(box(0.018, 0.012, 0.003, pos=(0.0, 0.028, 0.0055)))
    return merge(*parts)


def q_fc():
    """飞控堆栈，双板、IMU 芯片、USB、排针、减震柱。"""
    parts = [box(0.036, 0.036, 0.0016, pos=(0, 0, 0.001))]
    parts.append(box(0.036, 0.036, 0.0016, pos=(0, 0, 0.011)))
    for sx in (-0.013, 0.013):
        for sy in (-0.013, 0.013):
            parts.append(cyl(0.0022, 0.010, pos=(sx, sy, 0.006), n=10))
    parts.append(box(0.005, 0.005, 0.0012, pos=(0.006, 0.004, 0.0125)))
    parts.append(box(0.008, 0.006, 0.004, pos=(-0.010, 0.014, 0.002)))
    for i in range(5):
        parts.append(cyl(0.0006, 0.006, pos=(0.012, -0.012 + 0.005 * i, 0.019), n=6))
    return merge(*parts)


def q_obc():
    """机载计算机，载板加散热鳍片、MIPI 排线座、以太网、侧接口。"""
    parts = [box(0.070, 0.045, 0.003, pos=(0, 0, 0.0015))]
    parts.append(box(0.045, 0.045, 0.004, pos=(0, 0, 0.005)))
    for i in range(7):
        parts.append(box(0.045, 0.0028, 0.011, pos=(0, -0.018 + 0.006 * i, 0.0125)))
    for i in range(4):
        parts.append(box(0.006, 0.003, 0.003, pos=(-0.026 + 0.017 * i, 0.0235, 0.004)))
    parts.append(box(0.014, 0.010, 0.008, pos=(0.031, 0.010, 0.005)))
    parts.append(box(0.010, 0.008, 0.006, pos=(0.031, -0.008, 0.004)))
    parts.append(cyl(0.003, 0.006, pos=(-0.020, 0.020, 0.007), n=10))
    return merge(*parts)


def q_vtx():
    """图传发射，屏蔽壳、SMA 座、棒状天线。"""
    parts = [box(0.036, 0.020, 0.007, pos=(0, 0, 0.0035))]
    for i in range(5):
        parts.append(box(0.030, 0.0016, 0.004, pos=(0, -0.006 + 0.003 * i, 0.008)))
    parts.append(cyl(0.0045, 0.008, pos=(0.022, 0, 0.004), n=12, euler=(0, 1.5708, 0)))
    parts.append(tube_path([(0.030, 0, 0.004), (0.045, 0.010, 0.020), (0.050, 0.024, 0.034)], 0.0022, n_theta=10))
    return merge(*parts)


def q_rx():
    """遥控接收机，双天线。"""
    parts = [box(0.028, 0.018, 0.005, pos=(0, 0, 0.0025))]
    parts.append(tube_path([(-0.014, 0.006, 0.004), (-0.034, 0.020, 0.012)], 0.0009, n_theta=8))
    parts.append(tube_path([(-0.014, -0.006, 0.004), (-0.034, -0.020, 0.012)], 0.0009, n_theta=8))
    return merge(*parts)


def q_gnss():
    """GNSS 圆盘加安装杆。"""
    parts = [cyl(0.036, 0.002, pos=(0, 0, 0), n=24)]
    parts.append(cyl(0.020, 0.003, pos=(0, 0, 0.002), n=20))
    parts.append(cyl(0.0125, 0.012, pos=(0, 0, 0.003), n=24, taper=0.9))
    return merge(*parts)


def q_standoff():
    """机架支柱，柱体加六角头。"""
    parts = [cyl(0.0035, 0.030, pos=(0, 0, 0.0), n=14)]
    parts.append(hex_head(0.0044, 0.003, (0, 0, 0.030)))
    return merge(*parts)


def q_mast():
    """天线与 GNSS 立杆。"""
    parts = [cyl(0.0030, 0.036, pos=(0, 0, 0.0), n=12)]
    parts.append(cyl(0.0060, 0.003, pos=(0, 0, 0.0), n=14))
    return merge(*parts)


def q_cam(mount="front"):
    """相机模块，镜头筒加壳加支架，front 与 down 两种支架。"""
    parts = [box(0.030, 0.026, 0.020, pos=(0, 0, 0.010))]
    parts.append(cyl(0.008, 0.010, pos=(0.016, 0, 0.010), n=20, euler=(0, 1.5708, 0)))
    parts.append(cyl(0.009, 0.003, pos=(0.019, 0, 0.010), n=20, euler=(0, 1.5708, 0)))
    if mount == "front":
        parts.append(box(0.040, 0.030, 0.003, pos=(-0.004, 0, -0.0015)))
        parts.append(box(0.004, 0.030, 0.020, pos=(0.016, 0, -0.011)))
    else:
        parts.append(box(0.034, 0.034, 0.003, pos=(0, 0, -0.0015)))
        parts.append(box(0.004, 0.020, 0.016, pos=(-0.016, 0, -0.010)))
    return merge(*parts)


def q_tray():
    """B 机货盘，底板、四壁、前挡斜口、减震垫。"""
    parts = [box(0.110, 0.090, 0.004, pos=(0, 0, 0.002))]
    for sx in (-1, 1):
        parts.append(box(0.004, 0.090, 0.030, pos=(0.053 * sx, 0, 0.019)))
    parts.append(box(0.110, 0.004, 0.036, pos=(0, 0.043, 0.022)))
    parts.append(box(0.110, 0.004, 0.018, pos=(0, -0.043, 0.013)))
    for sx in (-0.035, 0.035):
        for sy in (-0.028, 0.028):
            parts.append(cyl(0.003, 0.006, pos=(sx, sy, 0.007), n=10))
    return merge(*parts)


def lattice_module(r, h, t=0.0016, gap=0.015, collar=0.009, joint_r=0.0042, panels=True):
    """方格反棱柱模块，实体折板加实心环箍板加销接节点，销轴只在顶环一组。"""
    collar = min(collar, h * 0.5)
    parts = []
    b = [(r * math.cos(i * math.pi / 2), r * math.sin(i * math.pi / 2), 0.0) for i in range(4)]
    tp = [(r * math.cos(math.pi / 4 + i * math.pi / 2), r * math.sin(math.pi / 4 + i * math.pi / 2), h) for i in range(4)]
    if panels:
        for i in range(4):
            parts.append(_plate((b[i], b[(i + 1) % 4], tp[i]), t, gap))
            parts.append(_plate((tp[i], tp[(i + 1) % 4], b[(i + 1) % 4]), t, gap))
    for ring, z0 in ((b, 0.0), (tp, h - collar)):
        quad = [(p[0], p[1], z0 + collar * 0.5) for p in ring]
        parts.append(_plate(quad, collar, 0.0))
    for (x, y, z) in tp:
        ph = math.atan2(y, x)
        l = 0.013
        pos = (x - l / 2 * math.sin(ph), y + l / 2 * math.cos(ph), z - 0.5 * collar)
        parts.append(cyl(joint_r, l, pos=pos, n=14, euler=(1.5708, 0.0, ph)))
    return merge(*parts)


def write_stl(path, verts, faces, name="part"):
    lines = ["solid %s" % name]
    for tri in faces:
        a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-12 else n
        lines.append("  facet normal %.6e %.6e %.6e" % (n[0], n[1], n[2]))
        lines.append("    outer loop")
        for p in (a, b, c):
            lines.append("      vertex %.6e %.6e %.6e" % (p[0], p[1], p[2]))
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid %s" % name)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def grip_base():
    parts = [box(0.056, 0.044, 0.020, pos=(0, 0, -0.010))]
    parts.append(box(0.042, 0.030, 0.014, pos=(0, 0, -0.026)))
    parts.append(cyl(0.006, 0.010, pos=(0, 0, 0.002), n=16))
    return merge(*parts)


def rot_y(part, ang, pivot=(0.0, 0.0, 0.0)):
    """把 (verts, faces) 绕 y 轴转 ang，绕 pivot 点。"""
    v = np.asarray(part[0], dtype=float)
    f = part[1]
    c, s = math.cos(ang), math.sin(ang)
    p = np.asarray(pivot, dtype=float)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (v - p) @ R.T + p, f


def finger_knuckle():
    """铰链销，装在各指根，视觉用。"""
    return cyl(0.0055, 0.017, pos=(0, 0, 0), n=16, euler=(1.5708, 0, 0))


def finger_prox():
    """菱形指近节，竖直扁平指板。"""
    return box(0.016, 0.009, 0.024, pos=(0.0, 0.0, -0.012))


def finger_dist():
    """菱形指远节，内折 0.18 弧度，内侧四条横棱加尖端。"""
    pivot = (0.0, 0.0, -0.024)
    parts = [box(0.014, 0.008, 0.030, pos=(0.0, 0.0, -0.039))]
    for z in (-0.030, -0.037, -0.044, -0.051):
        parts.append(box(0.003, 0.0075, 0.0025, pos=(-0.0085, 0.0, z)))
    parts.append(box(0.012, 0.0075, 0.006, pos=(0.0, 0.0, -0.056)))
    v, f = merge(*parts)
    return rot_y((v, f), 0.18, pivot)


def finger_flat():
    """平口指，整体直板，内侧横棱，钳形闭合。"""
    parts = [box(0.016, 0.009, 0.056, pos=(0.0, 0.0, -0.028))]
    for z in (-0.010, -0.021, -0.032, -0.043):
        parts.append(box(0.003, 0.0075, 0.0025, pos=(-0.0095, 0.0, z)))
    parts.append(box(0.014, 0.0075, 0.006, pos=(0.0, 0.0, -0.058)))
    return merge(*parts)


def finger_soft():
    """软体波纹指，硅胶节鼓形，单铰近似弯曲。"""
    prof = [
        (0.0006, 0.0), (0.0090, 0.0), (0.0090, -0.006), (0.0105, -0.010),
        (0.0080, -0.015), (0.0102, -0.019), (0.0076, -0.024), (0.0098, -0.028),
        (0.0070, -0.033), (0.0092, -0.037), (0.0062, -0.042), (0.0082, -0.046),
        (0.0050, -0.050), (0.0028, -0.053), (0.0006, -0.055),
    ]
    return revolve(prof, n_theta=40)


def grip_links():
    """腕部两条深色连杆，构成菱形传动外观，两侧各一对。"""
    parts = []
    for sx in (1, -1):
        parts.append(box(0.016, 0.005, 0.0035, pos=(sx * 0.014, 0.016, -0.029), euler=(0.0, sx * 0.66, 0.0)))
        parts.append(box(0.032, 0.0045, 0.0035, pos=(sx * 0.018, 0.016, -0.032), euler=(0.0, sx * 0.83, 0.0)))
    return merge(*parts)


def drone_plate():
    parts = [box(0.200, 0.200, 0.012, pos=(0, 0, 0))]
    parts.append(box(0.070, 0.100, 0.032, pos=(0, 0, 0.022)))
    parts.append(box(0.200, 0.200, 0.010, pos=(0, 0, -0.011)))
    parts.append(cyl(0.008, 0.020, pos=(0.070, 0.0, 0.014), n=12, euler=(1.5708, 0.0, 0.0)))
    return merge(*parts)


def drone_arm():
    parts = [cyl(0.011, 0.150, pos=(0.055, 0, -0.002), n=16, euler=(0.0, 1.5708, 0.0))]
    parts.append(box(0.040, 0.036, 0.016, pos=(0.090, 0, -0.002)))
    return merge(*parts)


def drone_motor():
    parts = [cyl(0.021, 0.024, n=24)]
    parts.append(cyl(0.007, 0.014, pos=(0, 0, 0.028), n=12))
    return merge(*parts)


def drone_prop():
    parts = [box(0.100, 0.013, 0.0022, pos=(0.028, 0, 0))]
    parts.append(box(0.100, 0.013, 0.0022, pos=(-0.028, 0, 0)))
    parts.append(cyl(0.010, 0.005, pos=(0, 0, 0.001), n=16))
    return merge(*parts)


def drone_skid():
    parts = [cyl(0.006, 0.090, pos=(0, 0, 0), n=12, euler=(0.0, 1.5708, 0.0))]
    for sx in (-0.035, 0.035):
        parts.append(cyl(0.005, 0.030, pos=(sx, 0, 0.018), n=10))
    return merge(*parts)


def shadow_disc(n_theta=64, n_r=24):
    """单位半径圆盘，带 UV，用于贴软阴影纹理。"""
    verts, uvs, faces = [], [], []
    verts.append((0.0, 0.0, 0.0))
    uvs.append((0.5, 0.5))
    for i in range(1, n_r + 1):
        rr = i / n_r
        for j in range(n_theta):
            a = 2 * math.pi * j / n_theta
            verts.append((rr * math.cos(a), rr * math.sin(a), 0.0))
            uvs.append((0.5 + 0.5 * rr * math.cos(a), 0.5 + 0.5 * rr * math.sin(a)))
    for j in range(n_theta):
        faces.append((0, 1 + j, 1 + (j + 1) % n_theta))
    for i in range(1, n_r):
        base0 = 1 + (i - 1) * n_theta
        base1 = 1 + i * n_theta
        for j in range(n_theta):
            a = base0 + j
            b = base0 + (j + 1) % n_theta
            c = base1 + (j + 1) % n_theta
            d = base1 + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return (np.array(verts), np.array(faces, dtype=int), np.array(uvs), np.array(faces, dtype=int))


def cables():
    parts = []
    for i, (x0, y0, s) in enumerate(((-0.020, 0.020, 1), (-0.014, 0.024, 1), (-0.026, 0.014, 1))):
        pts = []
        n = 40
        for k in range(n + 1):
            t = k / n
            ang = math.radians(35 * t)
            rr = 0.030 + 0.028 * t
            pts.append((x0 + 0.012 * t * s, y0 + 0.020 * t * s, 0.012 + 0.030 * t + 0.004 * math.sin(ang)))
        parts.append(tube_path(pts, 0.0035, n_theta=10))
    return merge(*parts)


def wrist_plate():
    prof = [(0.0, 0.0), (0.016, 0.0), (0.017, 0.002), (0.017, 0.006), (0.014, 0.008), (0.0, 0.008)]
    return revolve(prof)


def servo_body():
    parts = [box(0.042, 0.024, 0.026, pos=(0, 0, -0.017))]
    parts.append(box(0.048, 0.010, 0.006, pos=(0, 0, -0.031)))
    parts.append(box(0.048, 0.028, 0.004, pos=(0, 0, -0.002)))
    parts.append(cyl(0.0035, 0.012, pos=(0.014, 0.0, -0.006), n=12, euler=(1.5708, 0.0, 0.0)))
    return merge(*parts)


def servo_horn():
    parts = [cyl(0.006, 0.004, pos=(0, 0, -0.033), n=20)]
    parts.append(box(0.026, 0.009, 0.004, pos=(0.0, 0, -0.033)))
    parts.append(cyl(0.0012, 0.006, pos=(0.011, 0, -0.033), n=10))
    parts.append(cyl(0.0012, 0.006, pos=(-0.011, 0, -0.033), n=10))
    return merge(*parts)


def finger_assembly():
    """单指总成，含基座、近节、远节、指垫与三根销轴。"""
    parts = [box(0.014, 0.012, 0.009, pos=(0.0, 0.0, -0.0045))]
    parts.append(box(0.009, 0.011, 0.016, pos=(0.0, 0.0, -0.017)))
    parts.append(box(0.008, 0.010, 0.016, pos=(-0.0025, 0.0, -0.031), euler=(0.0, -0.20, 0.0)))
    parts.append(box(0.0042, 0.008, 0.012, pos=(-0.0064, 0.0, -0.0335), euler=(0.0, -0.20, 0.0)))
    parts.append(cyl(0.0016, 0.013, pos=(0.0, 0.0065, 0.0), n=12, euler=(1.5708, 0.0, 0.0)))
    parts.append(cyl(0.0014, 0.012, pos=(-0.0015, 0.006, -0.0255), n=12, euler=(1.5708, 0.0, 0.0)))
    parts.append(cyl(0.0012, 0.011, pos=(-0.0052, 0.0055, -0.0385), n=12, euler=(1.5708, 0.0, 0.0)))
    return merge(*parts)


def push_rods():
    """舵盘到两指基座的推杆。"""
    parts = []
    for sx in (1, -1):
        pts = [(sx * 0.010, 0.0, -0.0345), (sx * 0.008, 0.0, -0.030), (sx * 0.011, 0.0, -0.040)]
        parts.append(tube_path(pts, 0.0010, n_theta=10))
    return merge(*parts)


def nameplate():
    w, h, t = 0.070, 0.011, 0.0012
    x, y, z = w / 2, t / 2, h / 2
    verts = np.array([
        [-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z],
        [-x, y, -z], [x, y, -z], [x, y, z], [-x, y, z],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    uvs = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    ])
    fuvs = faces.copy()
    return verts, faces, uvs, fuvs


def kettle_handle():
    return torus(0.028, 0.006, n_theta=40, n_phi=14)


def pump_body():
    parts = [box(0.070, 0.048, 0.040, pos=(0, 0, 0.020))]
    parts.append(cyl(0.017, 0.030, pos=(0, 0, 0.055), n=28))
    parts.append(cyl(0.019, 0.005, pos=(0, 0, 0.087), n=28))
    parts.append(box(0.024, 0.010, 0.014, pos=(0, 0.030, 0.012)))
    return merge(*parts)


def tank_body():
    prof = [(0.0, 0.0), (0.019, 0.002), (0.024, 0.010), (0.024, 0.100), (0.019, 0.108), (0.0, 0.110)]
    parts = [revolve(prof, n_theta=48)]
    parts.append(cyl(0.006, 0.014, pos=(0, 0.026, 0.052), n=16))
    return merge(*parts)


def valve_block():
    parts = [box(0.054, 0.032, 0.016, pos=(0, 0, 0.008))]
    for x in (-0.017, 0.0, 0.017):
        parts.append(cyl(0.0075, 0.024, pos=(x, 0, 0.028), n=18))
    return merge(*parts)


def fittings():
    parts = []
    for x in (-0.017, 0.0, 0.017):
        parts.append(cyl(0.0030, 0.010, pos=(x, 0, 0.056), n=12))
        parts.append(cyl(0.0040, 0.004, pos=(x, 0, 0.066), n=12))
    return merge(*parts)


def gauge_dial():
    parts = [cyl(0.021, 0.010, pos=(0, 0, 0.005), n=36)]
    parts.append(torus(0.020, 0.0020, pos=(0, 0, 0.011), n_theta=36, n_phi=10))
    return merge(*parts)


def gauge_needle():
    parts = [box(0.0032, 0.0018, 0.019, pos=(0, 0, 0.010))]
    parts.append(cyl(0.0030, 0.005, pos=(0, 0, 0.0), n=12))
    return merge(*parts)


def air_tubes():
    parts = []
    for i, x0 in enumerate((-0.017, 0.0, 0.017)):
        pts = [
            (x0, -0.06, 0.070),
            (x0, -0.10, 0.035),
            (0.030 + 0.010 * i, -0.17, 0.014),
            (0.050 + 0.008 * i, -0.22, 0.012),
            (0.071 + 0.008 * i, -0.265, 0.016),
        ]
        parts.append(tube_path(pts, 0.0026, n_theta=12))
    return merge(*parts)


def make_textures():
    from PIL import Image
    tex_dir = HERE / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 512
    yy, xx = np.mgrid[0:n, 0:n]
    smooth_x = np.apply_along_axis(lambda m: np.convolve(m, np.ones(31) / 31.0, mode="same"), 1,
                                   rng.normal(0, 1.0, (n, n)))
    smooth_y = np.apply_along_axis(lambda m: np.convolve(m, np.ones(31) / 31.0, mode="same"), 0, smooth_x)
    fabric = 231.0 + rng.normal(0, 2.2, (n, n)) + 2.4 * smooth_y
    streak = np.apply_along_axis(lambda m: np.convolve(m, np.ones(9) / 9.0, mode="same"), 1,
                                 rng.normal(0, 1.0, (n, n)))
    brushed = 226.0 + 10.0 * streak + rng.normal(0, 1.2, (n, n))
    speckle = 128.0 + rng.normal(0, 9.0, (n, n))
    for name, arr in (("fabric", fabric), ("brushed", brushed), ("speckle", speckle)):
        a = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([a, a, a]), "RGB").save(tex_dir / (name + ".png"))
    from PIL import ImageDraw, ImageFont
    plate = Image.new("RGB", (512, 128), (206, 168, 92))
    d = ImageDraw.Draw(plate)
    try:
        f1 = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 46)
        f2 = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    d.text((18, 12), "空中刚柔臂 AS-01", font=f1, fill=(42, 32, 14))
    d.text((20, 78), "SN 2026-001   AERIAL SOFT ARM", font=f2, fill=(64, 48, 22))
    d.rectangle((0, 0, 511, 127), outline=(118, 90, 38), width=6)
    plate.save(tex_dir / "nameplate.png")
    m = 256
    yy, xx = np.mgrid[0:m, 0:m]
    rad = np.sqrt(((xx - m / 2) / (m / 2)) ** 2 + ((yy - m / 2) / (m / 2)) ** 2)
    alpha = np.clip(1.0 - rad, 0.0, 1.0) ** 1.6
    soft = np.zeros((m, m, 4), dtype=np.uint8)
    soft[..., 3] = (alpha * 255).astype(np.uint8)
    Image.fromarray(soft, "RGBA").save(tex_dir / "shadow_soft.png")
    print("textures written to", tex_dir)


def main():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    r_start, r_end = 0.0215, 0.0125
    n_seg = 8
    seg_len = 0.05
    for i in range(n_seg):
        r0 = r_start + (r_end - r_start) * i / (n_seg - 1)
        write_obj(MESH_DIR / ("bellows_%d.obj" % i), *bellows_segment(r0, seg_len))
        write_obj(MESH_DIR / ("bellows_%d_inf.obj" % i), *bellows_segment_inflated(r0, seg_len))
        detail = merge(
            tendon_tubes(r0, seg_len),
            clamp_ring(r0, seg_len, tube_r=0.0016),
            clamp_ring(r0, 0.0, tube_r=0.0016),
        )
        write_obj(MESH_DIR / ("details_%d.obj" % i), *detail)
    write_obj(MESH_DIR / "base_flange.obj", *base_flange())
    write_obj(MESH_DIR / "base_bolts.obj", *base_bolts())
    write_obj(MESH_DIR / "pcb.obj", *pcb_board())
    write_obj(MESH_DIR / "pcb_chips.obj", *pcb_chips())
    write_obj(MESH_DIR / "pcb_sensor.obj", *pcb_sensor())
    write_obj(MESH_DIR / "tensioners.obj", *tensioners())
    write_obj(MESH_DIR / "cable_clips.obj", *cable_clips())
    write_obj(MESH_DIR / "shadow_disc.obj", *shadow_disc())
    for tag, h in (("s", 0.010), ("m", 0.030), ("l", 0.050)):
        v, f = lattice_module(0.028, h, panels=(tag != "s"))
        write_stl(MESH_DIR / ("lattice_%s.stl" % tag), v, f, "lattice_" + tag)
    for k in range(11):
        h = 0.010 + k * 0.004
        v, f = lattice_module(0.028, h, panels=(k > 0))
        write_obj(MESH_DIR / ("lattice_v%02d.obj" % k), v, f)
        write_obj(MESH_DIR / ("bundle_v%02d.obj" % k), *drive_bundle(0.028, h))
        write_obj(MESH_DIR / ("sheath_v%02d.obj" % k), *drive_sheath(0.028, h))
    write_obj(MESH_DIR / "truss_clamp.obj", *truss_clamp(0.028))
    write_obj(MESH_DIR / "truss_base_ring.obj", *truss_base_ring())
    write_obj(MESH_DIR / "truss_tensioners.obj", *truss_tensioners())
    write_obj(MESH_DIR / "tether_hose.obj", *tether_hose())
    write_obj(MESH_DIR / "q_plate.obj", *q_plate())
    write_obj(MESH_DIR / "q_arm_boom.obj", *q_arm_boom())
    write_obj(MESH_DIR / "q_motor.obj", *q_motor())
    write_obj(MESH_DIR / "q_prop.obj", *q_prop())
    write_obj(MESH_DIR / "q_landing.obj", *q_landing())
    write_obj(MESH_DIR / "q_battery.obj", *q_battery())
    write_obj(MESH_DIR / "q_pdb.obj", *q_pdb())
    write_obj(MESH_DIR / "q_fc.obj", *q_fc())
    write_obj(MESH_DIR / "q_obc.obj", *q_obc())
    write_obj(MESH_DIR / "q_vtx.obj", *q_vtx())
    write_obj(MESH_DIR / "q_rx.obj", *q_rx())
    write_obj(MESH_DIR / "q_gnss.obj", *q_gnss())
    write_obj(MESH_DIR / "q_standoff.obj", *q_standoff())
    write_obj(MESH_DIR / "q_mast.obj", *q_mast())
    write_obj(MESH_DIR / "q_cam_front.obj", *q_cam("front"))
    write_obj(MESH_DIR / "q_cam_down.obj", *q_cam("down"))
    write_obj(MESH_DIR / "q_tray.obj", *q_tray())
    write_obj(MESH_DIR / "grip_base.obj", *grip_base())
    write_obj(MESH_DIR / "finger_prox.obj", *finger_prox())
    write_obj(MESH_DIR / "finger_dist.obj", *finger_dist())
    write_obj(MESH_DIR / "finger_flat.obj", *finger_flat())
    write_obj(MESH_DIR / "finger_soft.obj", *finger_soft())
    write_obj(MESH_DIR / "finger_knuckle.obj", *finger_knuckle())
    write_obj(MESH_DIR / "grip_links.obj", *grip_links())
    write_obj(MESH_DIR / "drone_plate.obj", *drone_plate())
    write_obj(MESH_DIR / "drone_arm.obj", *drone_arm())
    write_obj(MESH_DIR / "drone_motor.obj", *drone_motor())
    write_obj(MESH_DIR / "drone_prop.obj", *drone_prop())
    write_obj(MESH_DIR / "drone_skid.obj", *drone_skid())
    write_obj(MESH_DIR / "cables.obj", *cables())
    write_obj(MESH_DIR / "wrist_plate.obj", *wrist_plate())
    write_obj(MESH_DIR / "servo_body.obj", *servo_body())
    write_obj(MESH_DIR / "servo_horn.obj", *servo_horn())
    write_obj(MESH_DIR / "finger.obj", *finger_assembly())
    write_obj(MESH_DIR / "push_rods.obj", *push_rods())
    write_obj(MESH_DIR / "nameplate.obj", *nameplate())
    write_obj(MESH_DIR / "kettle_handle.obj", *kettle_handle())
    write_obj(MESH_DIR / "pump_body.obj", *pump_body())
    write_obj(MESH_DIR / "tank_body.obj", *tank_body())
    write_obj(MESH_DIR / "valve_block.obj", *valve_block())
    write_obj(MESH_DIR / "fittings.obj", *fittings())
    write_obj(MESH_DIR / "gauge_dial.obj", *gauge_dial())
    write_obj(MESH_DIR / "gauge_needle.obj", *gauge_needle())
    write_obj(MESH_DIR / "air_tubes.obj", *air_tubes())
    make_textures()
    print("meshes written to", MESH_DIR)
    for p in sorted(MESH_DIR.glob("*.obj")):
        print("  %-22s %7d bytes" % (p.name, p.stat().st_size))


if __name__ == "__main__":
    main()
