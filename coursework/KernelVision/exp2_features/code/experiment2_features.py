# -*- coding: utf-8 -*-
"""实验二 图像特征检测 主脚本。

运行: python experiment2_features.py
输出: figures/ 插图, results/ 指标 CSV, output/ 结果图
"""
import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(ROOT, "images", "test.png")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "output")

CROP = (slice(240, 360), slice(280, 400))
SEED = 20241002
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


def imread_gray(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    cv2.imencode(ext, img)[1].tofile(path)


def show_gray(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)


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


def r4(x):
    return round(float(x), 4)


def add_gaussian_noise(img, sigma, rng):
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def gradient_magnitude(img, ksize=3):
    f = img.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=ksize)
    return cv2.magnitude(gx, gy)


def edge_metrics(edge, gray):
    ratio = float((edge > 0).mean())
    n, _ = cv2.connectedComponents((edge > 0).astype(np.uint8), connectivity=8)
    gm = gradient_magnitude(gray)
    mean_grad = float(gm[edge > 0].mean())
    return {"edge_ratio": r4(ratio), "n_components": int(n - 1), "mean_edge_grad": round(mean_grad, 1)}


def corner_marker(dst, tau_ratio):
    mask = (dst > tau_ratio * dst.max()).astype(np.uint8)
    marker = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask, marker


def sec1_canny(gray):
    print("[1] Canny 边缘检测 ...")
    rows = []

    pairs = [(30, 60), (60, 120), (100, 200), (150, 300), (200, 400)]
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.7))
    for j, (t1, t2) in enumerate(pairs):
        edge = cv2.Canny(gray, t1, t2)
        imwrite_u(os.path.join(OUT_DIR, "canny_t%d_%d.png" % (t1, t2)), edge)
        rows.append({"t1": t1, "t2": t2, **edge_metrics(edge, gray)})
        show_gray(axes[0, j], edge, "Canny %d/%d" % (t1, t2))
        show_gray(axes[1, j], edge[CROP], "局部 %d/%d" % (t1, t2))
    save_fig(fig, "fig01_canny_thresholds.png")

    g3 = float(gradient_magnitude(gray, 3).mean())
    gains = {ap: float(gradient_magnitude(gray, ap).mean()) / g3 for ap in (3, 5, 7)}
    variants = [
        ("aperture 3", 3, 100, 200, False),
        ("aperture 5 原阈值", 5, 100, 200, False),
        ("aperture 7 原阈值", 7, 100, 200, False),
        ("aperture 3 L2梯度", 3, 100, 200, True),
        ("aperture 5 增益补偿", 5, round(100 * gains[5]), round(200 * gains[5]), False),
        ("aperture 7 增益补偿", 7, round(100 * gains[7]), round(200 * gains[7]), False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.6))
    vrows = []
    for ax, (label, ap, t1, t2, l2) in zip(axes.ravel(), variants):
        edge = cv2.Canny(gray, t1, t2, apertureSize=ap, L2gradient=l2)
        vrows.append({"label": label, "aperture": ap, "L2": int(l2), "t1": t1, "t2": t2,
                      "gain": round(gains[ap], 2), **edge_metrics(edge, gray)})
        show_gray(ax, edge, label)
    save_fig(fig, "fig02_canny_variants.png")

    rng = np.random.default_rng(SEED)
    noisy = add_gaussian_noise(gray, 25.0, rng)
    edge_clean = cv2.Canny(gray, 100, 200)
    edge_direct = cv2.Canny(noisy, 100, 200)
    blur = cv2.GaussianBlur(noisy, (5, 5), 1.5)
    edge_blur = cv2.Canny(blur, 100, 200)
    med = cv2.medianBlur(noisy, 5)
    edge_med = cv2.Canny(med, 100, 200)

    nrows = []
    for name, edge in (("干净图直接检测", edge_clean),
                       ("噪声图直接检测", edge_direct),
                       ("高斯去噪后检测", edge_blur),
                       ("中值去噪后检测", edge_med)):
        nrows.append({"case": name, **edge_metrics(edge, gray)})

    fig, axes = plt.subplots(1, 4, figsize=(7.3, 2.35))
    show_gray(axes[0], noisy, "含噪声图")
    show_gray(axes[1], edge_direct, "含噪图直接检测")
    show_gray(axes[2], edge_blur, "高斯去噪后检测")
    show_gray(axes[3], edge_med, "中值去噪后检测")
    save_fig(fig, "fig03_canny_noise.png")

    write_csv("metrics_01_canny_thresholds.csv", rows,
              ["t1", "t2", "edge_ratio", "n_components", "mean_edge_grad"])
    write_csv("metrics_02_canny_variants.csv", vrows,
              ["label", "aperture", "L2", "t1", "t2", "gain", "edge_ratio",
               "n_components", "mean_edge_grad"])
    write_csv("metrics_03_canny_noise.csv", nrows,
              ["case", "edge_ratio", "n_components", "mean_edge_grad"])
    return rows


def sec2_harris(gray):
    print("[2] Harris 角点检测 ...")
    base = dict(block_size=2, ksize=3, k=0.04, tau=0.01)
    dst = cv2.cornerHarris(gray.astype(np.float32), base["block_size"],
                           base["ksize"], base["k"])
    mask, marker = corner_marker(dst, base["tau"])
    n_corners = int(mask.sum())

    lm_mask = (mask > 0) & (dst >= cv2.dilate(dst, np.ones((3, 3), np.float32)))
    lm_marker = cv2.dilate(lm_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    n_lm = int(lm_mask.sum())

    resp = np.clip(dst, 0, None)
    resp = (resp / resp.max()) ** 0.25
    marked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    marked[marker > 0] = [0, 0, 255]
    marked_lm = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    marked_lm[lm_marker > 0] = [0, 0, 255]
    imwrite_u(os.path.join(OUT_DIR, "harris_base.png"), marked)
    imwrite_u(os.path.join(OUT_DIR, "harris_localmax.png"), marked_lm)

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.5))
    show_gray(axes[0], gray, "灰度图")
    im = axes[1].imshow(resp, cmap="magma")
    axes[1].set_title("角点响应热力图", pad=6)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].grid(False)
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    axes[1].patch.set_edgecolor("lightgrey")
    axes[1].patch.set_linewidth(0.8)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8, length=0, labelcolor="dimgrey")
    show_color(axes[2], marked, "阈值标记 %d 个" % n_corners)
    show_color(axes[3], marked_lm, "非极大值抑制后 %d 个" % n_lm)
    save_fig(fig, "fig04_harris_base.png")

    sweeps = []
    for tau in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
        m, _ = corner_marker(dst, tau)
        sweeps.append({"group": "阈值比例 tau", "value": tau, "n_corners": int(m.sum())})
    for bs in [2, 3, 4, 5, 7]:
        m, _ = corner_marker(cv2.cornerHarris(gray.astype(np.float32), bs, 3, 0.04), 0.01)
        sweeps.append({"group": "窗口 blockSize", "value": bs, "n_corners": int(m.sum())})
    for ks in [3, 5, 7]:
        m, _ = corner_marker(cv2.cornerHarris(gray.astype(np.float32), 2, ks, 0.04), 0.01)
        sweeps.append({"group": "Sobel ksize", "value": ks, "n_corners": int(m.sum())})
    for kk in [0.02, 0.04, 0.06, 0.08, 0.10]:
        m, _ = corner_marker(cv2.cornerHarris(gray.astype(np.float32), 2, 3, kk), 0.01)
        sweeps.append({"group": "常数 k", "value": kk, "n_corners": int(m.sum())})

    groups = ["阈值比例 tau", "窗口 blockSize", "Sobel ksize", "常数 k"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    for ax, g in zip(axes.ravel(), groups):
        pts = [p for p in sweeps if p["group"] == g]
        xs = [str(p["value"]) for p in pts]
        ys = [p["n_corners"] for p in pts]
        bars = ax.bar(range(len(xs)), ys, color="#4575b4", alpha=.9)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_title(g + " 对角点数量的影响")
        ax.set_ylabel("角点数量")
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom",
                    fontsize=8.5, color="dimgrey")
        despine(ax)
    save_fig(fig, "fig05_harris_sweep.png")

    write_csv("metrics_04_harris_sweep.csv", sweeps, ["group", "value", "n_corners"])
    return base, {"raw": mask, "localmax": lm_mask.astype(np.uint8)}


def sec3_relation(gray, masks):
    print("[3] 边缘与角点联合分析 ...")
    edge = cv2.Canny(gray, 100, 200)
    dil = cv2.dilate((edge > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

    def near_ratio(mask):
        ys, xs = np.nonzero(mask)
        return float(dil[ys, xs].mean())

    near_raw = near_ratio(masks["raw"])
    near_lm = near_ratio(masks["localmax"])
    near_all = float(dil.mean())

    rng = np.random.default_rng(SEED)
    sample = rng.random(dil.shape) < 0.02
    near_bg = float(dil[sample].mean())

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[edge > 0] = [80, 180, 80]
    lm_marker = cv2.dilate(masks["localmax"], np.ones((3, 3), np.uint8), iterations=1)
    _, _, _, centroids = cv2.connectedComponentsWithStats(lm_marker, connectivity=8)
    for cx, cy in centroids[1:]:
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), 1)
    imwrite_u(os.path.join(OUT_DIR, "edge_corner_overlay.png"), overlay)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    show_color(axes[0], overlay, "边缘与角点叠加")
    show_color(axes[1], overlay[CROP], "局部放大")
    ax = axes[2]
    vals = [near_lm, near_raw, near_bg]
    bars = ax.bar(["非极大值抑制", "阈值法", "背景像素"], vals,
                  color=["#d73027", "#fc8d59", "#bdbdbd"], alpha=.9, width=0.55)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("位于边缘附近的比例")
    ax.set_title("角点的边缘邻近性")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.1f%%" % (100 * v),
                ha="center", va="bottom", fontsize=9, color="dimgrey")
    despine(ax)
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig06_edge_corner_relation.png")

    rows = [
        {"metric": "阈值法角点总数", "value": int(masks["raw"].sum())},
        {"metric": "非极大值抑制角点总数", "value": int(masks["localmax"].sum())},
        {"metric": "非极大值抑制角点位于边缘附近的比例", "value": r4(near_lm)},
        {"metric": "阈值法角点位于边缘附近的比例", "value": r4(near_raw)},
        {"metric": "全图像素位于边缘附近的比例", "value": r4(near_all)},
        {"metric": "随机背景像素位于边缘附近的比例", "value": r4(near_bg)},
    ]
    write_csv("metrics_05_edge_corner_relation.csv", rows, ["metric", "value"])
    return near_lm, near_bg


def main():
    for d in (FIG_DIR, RES_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
    setup_style()

    gray = imread_gray(IMG)
    print("图像读取完成: %s" % (gray.shape,))

    sec1_canny(gray)
    _, cor_masks = sec2_harris(gray)
    sec3_relation(gray, cor_masks)
    print("\n完成。图表 -> figures/ ; 数据 -> results/ ; 结果图 -> output/")


if __name__ == "__main__":
    main()
