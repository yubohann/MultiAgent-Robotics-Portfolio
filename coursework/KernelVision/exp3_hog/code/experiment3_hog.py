# -*- coding: utf-8 -*-
"""实验三 基于HOG特征的行人检测 主脚本。

运行: python experiment3_hog.py
输出: figures/ 插图, results/ 指标 CSV, output/ 结果图
"""
import csv
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "output")

GT = {
    "左侧紫红衣女士": (125, 105, 85, 170),
    "中间浅衣女士": (205, 103, 58, 177),
    "右侧红衣女士": (282, 78, 55, 170),
    "右侧眼镜男士": (315, 85, 48, 152),
    "后侧深衣行人": (255, 75, 40, 130),
}
IOU_MATCH = 0.35
LEG = dict(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
           labelcolor="dimgrey")


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def imread(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    cv2.imencode(ext, img)[1].tofile(path)


def show_color(ax, img, title):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)


def show_gray(ax, img, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)


def despine(ax):
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.8)


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("    -> %s" % path)


def write_csv(name, rows, fields):
    path = os.path.join(RES_DIR, name)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("    -> %s" % path)


def nms_malisiewicz(rects, overlap_thresh=0.65):
    rects = np.asarray(rects, dtype=float).reshape(-1, 4)
    x1, y1 = rects[:, 0], rects[:, 1]
    x2, y2 = rects[:, 0] + rects[:, 2], rects[:, 1] + rects[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(y2)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        ov = (w * h) / np.minimum(areas[i], areas[order[1:]] + 1e-9)
        order = order[np.where(ov <= overlap_thresh)[0] + 1]
    return rects[keep]


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter)


def evaluate(rects, gt=GT, thr=IOU_MATCH):
    matched = set()
    fp = 0
    for r in rects:
        best, best_name = 0.0, None
        for name, g in gt.items():
            v = iou(r, g)
            if v > best:
                best, best_name = v, name
        if best >= thr and best_name not in matched:
            matched.add(best_name)
        else:
            fp += 1
    n_pred = len(matched) + fp
    recall = len(matched) / len(gt)
    precision = len(matched) / n_pred if n_pred else 0.0
    return len(matched), fp, round(recall, 3), round(precision, 3)


def detect(hog, img, scale, win_stride, padding):
    t0 = time.perf_counter()
    rects, weights = hog.detectMultiScale(img, winStride=win_stride, padding=padding, scale=scale)
    return rects, weights, (time.perf_counter() - t0) * 1000.0


def draw_boxes(img, rects, color, thickness=2):
    out = img.copy()
    for x, y, w, h in rects:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return out


def hog_glyph(gray, cell=8, bins=9):
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)
    h, w = gray.shape
    vis = cv2.cvtColor(np.full((h, w), 255, np.uint8), cv2.COLOR_GRAY2BGR)
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            m = mag[i:i + cell, j:j + cell]
            a = ang[i:i + cell, j:j + cell]
            idx = np.minimum((a / (180.0 / bins)).astype(int), bins - 1)
            hist = np.array([m[idx == b].sum() for b in range(bins)])
            if hist.max() <= 1e-6:
                continue
            hist = hist / hist.max()
            cx, cy = j + cell // 2, i + cell // 2
            for b in range(bins):
                if hist[b] < 0.15:
                    continue
                theta = np.deg2rad((b + 0.5) * (180.0 / bins))
                dx = np.cos(theta) * cell * 0.45 * hist[b]
                dy = -np.sin(theta) * cell * 0.45 * hist[b]
                v = int(255 * (1.0 - 0.8 * hist[b]))
                cv2.line(vis, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                         (v, v, v), 1, cv2.LINE_AA)
    return vis


def sec1_hog_viz(img):
    print("[1] HOG 特征可视化 ...")
    crop = img[95:235, 115:255]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ang_vis = ((np.mod(ang, 180.0) / 180.0) * 255).astype(np.uint8)
    ang_color = cv2.applyColorMap(ang_vis, cv2.COLORMAP_HSV)
    ang_color[mag_vis < 20] = 0
    glyph = hog_glyph(gray)

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.3))
    show_color(axes[0], crop, "行人局部")
    show_gray(axes[1], mag_vis, "梯度幅值")
    show_color(axes[2], ang_color, "梯度方向")
    show_color(axes[3], glyph, "HOG 方向直方图")
    save_fig(fig, "fig01_hog_visualization.png")
    imwrite_u(os.path.join(OUT_DIR, "hog_glyph.png"), glyph)


def sec2_detect(img):
    print("[2] HOG + SVM 检测与 NMS ...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    rects, weights, dt = detect(hog, img, 1.05, (8, 8), (16, 16))
    keep = nms_malisiewicz(rects, 0.65)

    imwrite_u(os.path.join(OUT_DIR, "detect_raw.png"), draw_boxes(img, rects, (255, 0, 0), 1))
    imwrite_u(os.path.join(OUT_DIR, "detect_nms.png"), draw_boxes(img, keep, (0, 180, 0), 2))

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.75))
    show_color(axes[0], img, "输入图像")
    show_color(axes[1], draw_boxes(img, rects, (255, 0, 0), 1), "原始检测框 %d 个" % len(rects))
    show_color(axes[2], draw_boxes(img, keep, (0, 180, 0), 2), "NMS 后 %d 个" % len(keep))
    ax = axes[3]
    ax.axis("off")
    txt = "标注框 %d 个\n原始框 %d 个\nNMS 后 %d 个\n单次耗时 %.0f ms" % (
        len(GT), len(rects), len(keep), dt)
    ax.text(0.05, 0.5, txt, fontsize=10, va="center", ha="left", linespacing=1.8,
            color="dimgrey")
    save_fig(fig, "fig02_detect_nms.png")

    rows = []
    for thr in [0.2, 0.3, 0.5, 0.65, 0.8]:
        k = nms_malisiewicz(rects, thr)
        m, fp, rec, prec = evaluate(k)
        rows.append({"overlap": thr, "n_boxes": len(k), "matched": m,
                     "recall": rec, "precision": prec})
    write_csv("metrics_01_nms.csv", rows,
              ["overlap", "n_boxes", "matched", "recall", "precision"])

    m, fp, rec, prec = evaluate(keep)
    write_csv("metrics_00_gt.csv",
              [{"id": k, "x": v[0], "y": v[1], "w": v[2], "h": v[3]} for k, v in GT.items()],
              ["id", "x", "y", "w", "h"])
    write_csv("metrics_02_base.csv",
              [{"config": "baseline", "raw": len(rects), "nms": len(keep),
                "matched": m, "recall": rec, "precision": prec, "time_ms": round(dt, 1)}],
              ["config", "raw", "nms", "matched", "recall", "precision", "time_ms"])
    return hog, rects, keep


def sec3_sweep(img, hog):
    print("[3] 参数扫描 ...")
    rows = []

    def run(group, value, scale, ws, pad):
        all_times = []
        for _ in range(3):
            rects, _, dt = detect(hog, img, scale, ws, pad)
            all_times.append(dt)
        keep = nms_malisiewicz(rects, 0.65)
        m, fp, rec, prec = evaluate(keep)
        rows.append({"group": group, "value": value, "raw": len(rects), "nms": len(keep),
                     "matched": m, "recall": rec, "precision": prec,
                     "time_ms": round(float(np.mean(all_times)), 1)})
        return keep

    scales = [1.03, 1.05, 1.10, 1.25, 1.50]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.4))
    for j, s in enumerate(scales):
        keep = run("scale", s, s, (8, 8), (8, 8))
        show_color(axes[j // 3, j % 3], draw_boxes(img, keep, (0, 180, 0), 2), "scale %.2f" % s)
    ax = axes[1, 2]
    sub = [r for r in rows if r["group"] == "scale"]
    x = np.arange(len(scales))
    ax.bar(x - 0.18, [r["recall"] for r in sub], 0.36, label="召回率", color="#d73027", alpha=.9)
    ax.bar(x + 0.18, [r["precision"] for r in sub], 0.36, label="精确率", color="#4575b4",
           alpha=.9, hatch="//", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % s for s in scales])
    ax.set_ylim(0, 1.15)
    ax.set_title("scale 对检测性能的影响")
    ax.legend(**LEG)
    despine(ax)
    save_fig(fig, "fig03_scale_sweep.png")

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.4))
    for j, ws in enumerate([(4, 4), (8, 8), (16, 16)]):
        keep = run("winStride", "%d,%d" % ws, 1.05, ws, (8, 8))
        show_color(axes[0, j], draw_boxes(img, keep, (0, 180, 0), 2), "winStride %d,%d" % ws)
    for j, pad in enumerate([(0, 0), (8, 8), (16, 16)]):
        keep = run("padding", "%d,%d" % pad, 1.05, (8, 8), pad)
        show_color(axes[1, j], draw_boxes(img, keep, (0, 180, 0), 2), "padding %d,%d" % pad)
    save_fig(fig, "fig04_stride_padding.png")

    write_csv("metrics_03_sweep.csv", rows,
              ["group", "value", "raw", "nms", "matched", "recall", "precision", "time_ms"])

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    subs = [(r["group"], r["value"], r["time_ms"]) for r in rows]
    labels = ["%s %s" % (g.replace("winStride", "stride").replace("padding", "pad"), v)
              for g, v, _ in subs]
    vals = [t for _, _, t in subs]
    cols = ["#d73027"] * len(scales) + ["#4575b4"] * 3 + ["#009E73"] * 3
    bars = ax.bar(range(len(vals)), vals, color=cols, alpha=.9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("平均检测时间 ms")
    ax.set_title("参数对检测时间的影响")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.0f" % v, ha="center", va="bottom",
                fontsize=7.5, color="dimgrey")
    despine(ax)
    save_fig(fig, "fig05_time.png")
    return rows


def main():
    for d in (FIG_DIR, RES_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
    setup_style()

    img = imread(os.path.join(IMG_DIR, "people.jpg"))
    print("图像读取完成: %s" % (img.shape,))

    sec1_hog_viz(img)
    hog, rects, keep = sec2_detect(img)
    sec3_sweep(img, hog)
    print("\n完成。图表 -> figures/ ; 数据 -> results/ ; 结果图 -> output/")


if __name__ == "__main__":
    main()
