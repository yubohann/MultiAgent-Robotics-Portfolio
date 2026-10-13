"""桁架臂制造包，料单与零件尺寸图，配套 STL 在 meshes 目录。

运行：
  python arm_lattice_bom.py --all
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_real as A

OUT = A.OUT_DIR
FONT = "C:/Windows/Fonts/msyh.ttc"
R_RING = 0.028
H_EXT = 0.050
T_PANEL = 0.0016
N_SEG = 8
COLLAR = 0.009
JOINT_R = 0.0042
N_RING = N_SEG + 1


def geom():
    r, h = R_RING, H_EXT
    b = [np.array([r * math.cos(i * math.pi / 2), r * math.sin(i * math.pi / 2), 0.0]) for i in range(4)]
    tp = [np.array([r * math.cos(math.pi / 4 + i * math.pi / 2), r * math.sin(math.pi / 4 + i * math.pi / 2), h]) for i in range(4)]

    def d(p, q):
        return float(np.linalg.norm(p - q))

    t1 = (d(b[0], b[1]), d(b[0], tp[0]), d(b[1], tp[0]))
    t2 = (d(tp[0], tp[1]), d(tp[0], b[1]), d(tp[1], b[1]))
    ring = d(b[0], b[1])
    def area3(p, q, s):
        a, bb, c = p, q, s
        return float(np.linalg.norm(np.cross(bb - a, c - a)) / 2)
    a1 = area3(b[0], b[1], tp[0])
    a2 = area3(tp[0], tp[1], b[1])
    return b, tp, t1, t2, ring, a1, a2


def bom_md():
    b, tp, t1, t2, ring, a1, a2 = geom()
    rows = [
        ("三角折板 A 型", "%d" % (4 * N_SEG), "边长 %.1f / %.1f / %.1f，厚 %.1f" % (t1[0] * 1000, t1[1] * 1000, t1[2] * 1000, T_PANEL * 1000), "碳纤维板 T700，1.6 毫米", "CNC 切割，孔位与销轴配合 H7"),
        ("三角折板 B 型", "%d" % (4 * N_SEG), "边长 %.1f / %.1f / %.1f，厚 %.1f" % (t2[0] * 1000, t2[1] * 1000, t2[2] * 1000, T_PANEL * 1000), "碳纤维板 T700，1.6 毫米", "CNC 切割，孔位与销轴配合 H7"),
        ("环箍板", "%d" % (4 * N_RING), "内接方边 %.1f，高 %.1f，厚 %.1f" % (ring * 1000, COLLAR * 1000, T_PANEL * 1000), "铝合金 6061-T6，1.6 毫米", "钣金折弯，阳极氧化"),
        ("销轴", "%d" % (4 * N_RING), "直径 %.1f，长 13" % (JOINT_R * 2 * 1000), "不锈钢 304", "车削，与环箍过盈 0.02"),
        ("腱锚座", "%d" % (3 * N_RING), "内接方边 20，厚 3", "铝合金 6061-T6", "CNC，攻 M3 螺纹"),
        ("气管快插", "%d" % N_SEG, "M5 螺纹，外径 6", "黄铜镀镍", "外购"),
        ("腱鞘与腱", "3 路", "外径 4 毫米，腱径 0.8 毫米", "PTFE 管与超高分子量聚乙烯腱", "外购，预紧力 20 牛"),
    ]
    lines = ["# AS-01 桁架臂 料单与尺寸", "",
             "模块参数，环半径 %.1f 毫米，展开节距 %.1f 毫米，收拢节距 %.1f 毫米，伸缩比 %.0f 倍，整臂展开长 %.0f 毫米。" % (
                 R_RING * 1000, H_EXT * 1000, 0.010 * 1000, H_EXT / 0.010, N_SEG * H_EXT * 1000),
             "",
             "| 零件 | 数量 | 尺寸 | 材料 | 工艺 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s |" % r)
    lines.append("")
    lines.append("单模块展开状态板件面积，A 型 %.0f 平方毫米，B 型 %.0f 平方毫米，整臂板件总面积约 %.0f 平方厘米。" % (
        a1 * 1e6, a2 * 1e6, (a1 + a2) * 4 * N_SEG * 1e-4))
    md = "\n".join(lines)
    (OUT / "lattice_bom.md").write_text(md, encoding="utf-8")
    return rows


def draw_sheet(rows):
    W, H = 1860, 1220
    im = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(im)
    f_big = ImageFont.truetype(FONT, 34)
    f_mid = ImageFont.truetype(FONT, 23)
    f_small = ImageFont.truetype(FONT, 20)
    dr.text((40, 26), "AS-01 桁架臂 制造图（单位 mm）", font=f_big, fill=(28, 30, 34))
    dr.text((40, 74), "方格反棱柱模块 %.0f 节，展开节距 %.0f，收拢节距 %.0f，伸缩比 5 倍，STL 见 meshes\\lattice_s / m / l" % (
        N_SEG, H_EXT * 1000, 10), font=f_small, fill=(96, 100, 106))

    scale = 4.0
    r_mm = R_RING * 1000.0
    h_mm = H_EXT * 1000.0
    c_mm = COLLAR * 1000.0
    jr_mm = JOINT_R * 2 * 1000.0

    def ring_pts(z, phase):
        return [(r_mm * math.cos(phase + i * math.pi / 2), r_mm * math.sin(phase + i * math.pi / 2), z) for i in range(4)]

    def draw_module(cx, top_y, sc, b_pts, t_pts, y_ratio=0.35):
        def proj(p):
            return (cx + p[0] * sc, top_y + p[1] * sc * y_ratio - p[2] * sc)
        zb, zt = b_pts[0][2], t_pts[0][2]
        faces = []
        for i in range(4):
            for tri in ((b_pts[i], b_pts[(i + 1) % 4], t_pts[i]), (t_pts[i], t_pts[(i + 1) % 4], b_pts[(i + 1) % 4])):
                pts = [proj(p) for p in tri]
                cy = sum(p[1] for p in tri) / 3
                cxv = sum(p[0] for p in tri) / 3
                ang = math.atan2(cy, cxv)
                sh = 146 + 54 * math.cos(ang - 1.1)
                faces.append((cy, pts, (int(sh), int(sh), int(sh + 7))))
        for ring, z0, z1 in ((b_pts, zb, zb + c_mm), (t_pts, zt - c_mm, zt)):
            for i in range(4):
                quad = [(ring[i][0], ring[i][1], z0), (ring[(i + 1) % 4][0], ring[(i + 1) % 4][1], z0),
                        (ring[(i + 1) % 4][0], ring[(i + 1) % 4][1], z1), (ring[i][0], ring[i][1], z1)]
                pts = [proj(p) for p in quad]
                cy = sum(p[1] for p in quad) / 4
                faces.append((cy, pts, (118, 122, 128)))
        faces.sort(key=lambda f: -f[0])
        for _, pts, fill in faces:
            dr.polygon(pts, fill=fill, outline=(64, 66, 72), width=2)
        for ring, zc in ((b_pts, zb + c_mm * 0.5), (t_pts, zt - c_mm * 0.5)):
            for (x, y, _) in ring:
                p = proj((x, y, zc))
                rad = jr_mm * sc * 0.55
                dr.ellipse((p[0] - rad, p[1] - rad, p[0] + rad, p[1] + rad), fill=(96, 100, 108), outline=(56, 58, 64), width=2)

    cx = 280.0
    top_y = 430.0
    draw_module(cx, top_y, scale, ring_pts(0.0, 0.0), ring_pts(h_mm, math.pi / 4))
    dr.text((150, 500), "模块立体，折板环箍销接相连", font=f_mid, fill=(60, 64, 70))
    dr.line((cx - r_mm * scale, 470, cx + r_mm * scale, 470), fill=(60, 62, 66), width=2)
    dr.text((cx + r_mm * scale + 12, 458), "Φ%.0f" % (r_mm * 2), font=f_mid, fill=(40, 42, 46))
    dr.line((cx + r_mm * scale + 70, top_y - h_mm * scale, cx + r_mm * scale + 70, top_y), fill=(60, 62, 66), width=2)
    dr.text((cx + r_mm * scale + 84, top_y - h_mm * scale / 2 - 12), "节距 %.0f" % h_mm, font=f_mid, fill=(40, 42, 46))

    tx, ty = 760.0, 330.0
    def proj2(p):
        return (tx + p[0] * scale, ty - p[1] * scale)
    b0 = ring_pts(0.0, 0.0)
    t0 = ring_pts(h_mm, math.pi / 4)
    for i in range(4):
        dr.line([proj2(b0[i]), proj2(t0[i])], fill=(90, 94, 100), width=2)
        dr.line([proj2(b0[i]), proj2(t0[(i - 1) % 4])], fill=(90, 94, 100), width=2)
    for sq, col in ((b0, (60, 62, 66)), (t0, (150, 90, 40))):
        pts = [proj2(p) for p in sq]
        dr.line(pts + [pts[0]], fill=col, width=3)
    for p in b0 + t0:
        q = proj2(p)
        dr.ellipse((q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5), fill=(96, 100, 108), outline=(56, 58, 64), width=2)
    dr.text((tx - 230, ty + 165), "俯视，环向两方错开 45 度，八根斜边连成星", font=f_mid, fill=(60, 64, 70))
    dr.line((tx + 30, ty + 55, tx + 30 + r_mm * math.sqrt(2) * scale, ty + 55), fill=(60, 62, 66), width=2)
    dr.text((tx + 40, ty + 65), "内接方边 %.1f" % (r_mm * math.sqrt(2)), font=f_small, fill=(40, 42, 46))

    side_mm = r_mm * math.sqrt(2)
    slope_mm = 54.4
    height_mm = math.sqrt(slope_mm ** 2 - (side_mm / 2) ** 2)
    px0, py0 = 1060.0, 500.0
    tri = [(px0, py0), (px0 + side_mm * scale, py0), (px0 + side_mm * scale / 2, py0 - height_mm * scale)]
    dr.polygon(tri, fill=(232, 234, 238), outline=(60, 62, 66), width=3)
    labels = ("%.1f" % side_mm, "%.1f" % slope_mm, "%.1f" % slope_mm)
    for k in range(3):
        p0, p1 = tri[k], tri[(k + 1) % 3]
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dr.text((mx + 8, my + 8), labels[k], font=f_small, fill=(40, 42, 46))
    dr.text((px0 - 40, py0 - height_mm * scale - 60), "折板展开件，A 型与 B 型各 4 件每节，厚 %.1f" % (T_PANEL * 1000), font=f_mid, fill=(60, 64, 70))
    for p in tri:
        dr.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), outline=(180, 60, 50), width=2)
    dr.text((px0 - 40, py0 + 14), "三处销孔 Φ%.1f" % jr_mm, font=f_small, fill=(180, 60, 50))

    chx, chtop, chsc = 1580.0, 585.0, 2.7
    for k in range(3):
        draw_module(chx, chtop, chsc, ring_pts(k * h_mm, k * math.pi / 4), ring_pts((k + 1) * h_mm, (k + 1) * math.pi / 4))
    dr.text((chx - 150, 630), "3 节相连，共环共销", font=f_mid, fill=(60, 64, 70))

    ty0 = 660
    dr.text((40, ty0 - 30), "料单", font=f_mid, fill=(60, 64, 70))
    col_x = [40, 320, 470, 800, 1120]
    header = ["零件", "数量", "尺寸", "材料", "工艺"]
    for i, htxt in enumerate(header):
        dr.text((col_x[i], ty0), htxt, font=f_small, fill=(90, 94, 100))
    y = ty0 + 32
    for r in rows:
        for i, cell in enumerate(r):
            dr.text((col_x[i], y), cell, font=f_small, fill=(40, 42, 46))
        y += 34
        dr.line((40, y - 8, W - 40, y - 8), fill=(222, 224, 228), width=1)
    dr.text((40, y + 10), "整臂展开长 %.0f，收拢长 %.0f，臂体质量见 mass_table，STL 可直接切片打印验证。" % (
        N_SEG * H_EXT * 1000, N_SEG * 10), font=f_small, fill=(110, 114, 120))
    im.save(OUT / "lattice_bom.png")
    print("wrote", OUT / "lattice_bom.png", OUT / "lattice_bom.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        rows = bom_md()
        draw_sheet(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
