# -*- coding: utf-8 -*-
"""实验一 图像空域滤波 主脚本。

运行: python experiment1_filtering.py
输出: figures/ 插图, results/ 指标 CSV, output/ 结果图
"""
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cvlab_utils import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    cross_image_reference,
    edge_preservation,
    edge_slope,
    estimate_noise_sigma,
    estimate_sp_density,
    flat_region_sigma,
    imread_gray,
    imwrite_u,
    psnr,
    save_fig,
    setup_plot_style,
    show_gray,
    ssim,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "output")

SEED = 20240917
CROP = (slice(230, 350), slice(300, 420))
COLORS = {"均值滤波": "#d73027", "高斯滤波": "#4575b4", "中值滤波": "#009E73", "双边滤波": "#E69F00"}
LEG = dict(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
           labelcolor="dimgrey")


def mean_filter(img, k):
    return cv2.blur(img, (k, k))


def gaussian_filter(img, k, sigma=0.0):
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, k):
    return cv2.medianBlur(img, k)


def bilateral_filter(img, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


FILTERS = {
    "均值滤波": lambda x: mean_filter(x, 5),
    "高斯滤波": lambda x: gaussian_filter(x, 5),
    "中值滤波": lambda x: median_filter(x, 5),
    "双边滤波": lambda x: bilateral_filter(x, 9, 50, 50),
}


def despine(ax):
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.8)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print("    -> %s" % path)


def r3(x):
    return round(float(x), 3)


def sec0_noise_analysis(clean, noisy_sp, noisy_gauss):
    print("[0] 噪声特性辨识 ...")
    rows = []
    for name, img, kind in (
        ("image_fft", clean, "干净参考图"),
        ("noised1", noisy_sp, "椒盐噪声"),
        ("noised2", noisy_gauss, "高斯噪声"),
    ):
        rows.append(
            {
                "image": name,
                "kind": kind,
                "size": "%dx%d" % (img.shape[1], img.shape[0]),
                "mean": r3(img.mean()),
                "std": r3(img.std()),
                "pixel_eq_0": r3((img == 0).mean()),
                "pixel_eq_255": r3((img == 255).mean()),
                "pixel_extreme_total": r3((img == 0).mean() + (img == 255).mean()),
                "sp_density_est": r3(estimate_sp_density(img)),
                "gauss_sigma_est": r3(estimate_noise_sigma(img)),
            }
        )

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.9))
    titles = (
        ("image_fft 干净参考图", "灰度直方图", "局部放大"),
        ("noised1 椒盐噪声", "灰度直方图", "局部放大"),
        ("noised2 高斯噪声", "灰度直方图", "局部放大"),
    )
    for i, (name, img) in enumerate(
        (("image_fft", clean), ("noised1", noisy_sp), ("noised2", noisy_gauss))
    ):
        show_gray(axes[i, 0], img, titles[i][0])
        hist = np.bincount(img.ravel(), minlength=256)
        ax = axes[i, 1]
        ax.plot(hist, color="#4575b4", lw=0.9)
        ax.set_title(titles[i][1], pad=6)
        ax.set_xlim(-4, 259)
        ax.set_ylabel("像素数")
        despine(ax)
        if name == "noised1":
            ax.annotate("0 处尖峰", xy=(0, hist[0]), xytext=(30, hist.max() * 0.90),
                        arrowprops=dict(arrowstyle="->", color="#bd0c0c"),
                        color="#bd0c0c", fontsize=9)
            ax.annotate("255 处尖峰", xy=(255, hist[255]), xytext=(140, hist.max() * 0.90),
                        arrowprops=dict(arrowstyle="->", color="#bd0c0c"),
                        color="#bd0c0c", fontsize=9)
        show_gray(axes[i, 2], img[CROP], titles[i][2])
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig01_noise_analysis.png"))
    return rows


def sec1_smoothing(noisy_gauss):
    print("[1] 平滑滤波实验: 均值滤波 / 高斯滤波 (输入 noised2) ...")
    rows = []
    base_flat = flat_region_sigma(noisy_gauss)
    ks = [3, 5, 7, 9]

    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.1))
    show_gray(axes[0, 0], noisy_gauss, "含噪图")
    show_gray(axes[1, 0], noisy_gauss[CROP], "局部 含噪图")
    for j, k in enumerate(ks):
        out = mean_filter(noisy_gauss, k)
        imwrite_u(os.path.join(OUT_DIR, "noised2_mean_k%d.png" % k), out)
        f = flat_region_sigma(out)
        rows.append({
            "filter": "均值滤波", "ksize": k, "param": "k=%d" % k,
            "flat_sigma": r3(f), "noise_reduction": r3(1 - f / base_flat),
            "edge_slope": r3(edge_slope(noisy_gauss, out)),
        })
        show_gray(axes[0, j + 1], out, "均值 k=%d" % k)
        show_gray(axes[1, j + 1], out[CROP], "局部 均值 k=%d" % k)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig02a_smoothing_mean.png"))

    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.1))
    show_gray(axes[0, 0], noisy_gauss, "含噪图")
    show_gray(axes[1, 0], noisy_gauss[CROP], "局部 含噪图")
    for j, k in enumerate(ks):
        out = gaussian_filter(noisy_gauss, k)
        imwrite_u(os.path.join(OUT_DIR, "noised2_gauss_k%d.png" % k), out)
        f = flat_region_sigma(out)
        rows.append({
            "filter": "高斯滤波", "ksize": k, "param": "k=%d" % k,
            "flat_sigma": r3(f), "noise_reduction": r3(1 - f / base_flat),
            "edge_slope": r3(edge_slope(noisy_gauss, out)),
        })
        show_gray(axes[0, j + 1], out, "高斯 k=%d" % k)
        show_gray(axes[1, j + 1], out[CROP], "局部 高斯 k=%d" % k)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig02b_smoothing_gauss.png"))

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    width = 0.38
    xs = np.arange(len(ks))
    mset = [r for r in rows if r["filter"] == "均值滤波"]
    gset = [r for r in rows if r["filter"] == "高斯滤波"]
    bars1 = ax.bar(xs - width / 2, [r["flat_sigma"] for r in mset], width,
                   label="均值滤波", color="#d73027", alpha=.9)
    bars2 = ax.bar(xs + width / 2, [r["flat_sigma"] for r in gset], width,
                   label="高斯滤波", color="#4575b4", alpha=.9)
    ax.axhline(base_flat, color="dimgrey", ls="--", lw=0.9)
    ax.text(-0.45, base_flat * 1.03, "含噪图 σ=%.1f" % base_flat, va="bottom", ha="left",
            fontsize=9, color="dimgrey")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    "%.2f" % b.get_height(), ha="center", va="bottom",
                    fontsize=7.2, color="dimgrey")
    ax.set_xticks(xs)
    ax.set_xticklabels(["k=%d" % k for k in ks])
    ax.set_ylabel("平坦区域噪声 σ")
    ax.set_title("核大小对噪声抑制的影响")
    ax.set_ylim(0, base_flat * 1.18)
    ax.legend(**LEG)
    despine(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig02c_smoothing_bars.png"))
    return rows


def sec2_denoising(noisy_sp):
    print("[2] 去噪滤波实验: 中值滤波 / 双边滤波 (输入 noised1) ...")
    rows = []
    base_imp = estimate_sp_density(noisy_sp)

    fig, axes = plt.subplots(2, 4, figsize=(6.8, 4.1))
    show_gray(axes[0, 0], noisy_sp, "含噪图")
    show_gray(axes[1, 0], noisy_sp[CROP], "局部 含噪图")
    for j, k in enumerate([3, 5, 7]):
        out = median_filter(noisy_sp, k)
        imwrite_u(os.path.join(OUT_DIR, "noised1_median_k%d.png" % k), out)
        f = flat_region_sigma(out)
        rows.append({
            "filter": "中值滤波", "param": "k=%d" % k,
            "residual_impulse": r3(estimate_sp_density(out)),
            "impulse_removal": r3(1 - estimate_sp_density(out) / base_imp),
            "flat_sigma": r3(f), "edge_slope": r3(edge_slope(noisy_sp, out)),
        })
        show_gray(axes[0, j + 1], out, "中值 k=%d" % k)
        show_gray(axes[1, j + 1], out[CROP], "局部 中值 k=%d" % k)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig03a_denoising_median.png"))

    fig, axes = plt.subplots(2, 4, figsize=(6.8, 4.1))
    show_gray(axes[0, 0], noisy_sp, "含噪图")
    show_gray(axes[1, 0], noisy_sp[CROP], "局部 含噪图")
    for j, (dd, sc, ss) in enumerate([(5, 50, 50), (9, 75, 75), (9, 150, 150)]):
        out = bilateral_filter(noisy_sp, dd, sc, ss)
        imwrite_u(os.path.join(OUT_DIR, "noised1_bilateral_d%d_c%d.png" % (dd, sc)), out)
        f = flat_region_sigma(out)
        rows.append({
            "filter": "双边滤波", "param": "d=%d, sigmaColor=%d, sigmaSpace=%d" % (dd, sc, ss),
            "residual_impulse": r3(estimate_sp_density(out)),
            "impulse_removal": r3(1 - estimate_sp_density(out) / base_imp),
            "flat_sigma": r3(f), "edge_slope": r3(edge_slope(noisy_sp, out)),
        })
        show_gray(axes[0, j + 1], out, "双边 d=%d\nσColor=%d" % (dd, sc))
        show_gray(axes[1, j + 1], out[CROP], "局部 双边 d=%d\nσColor=%d" % (dd, sc))
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig03b_denoising_bilateral.png"))
    return rows


def sec3_comparison(noisy_sp, noisy_gauss):
    print("[3] 四类滤波器横向对比 (真实含噪图, 无参考指标) ...")
    rows = []
    spec = [
        ("salt_pepper", noisy_sp, "residual_impulse", estimate_sp_density),
        ("gaussian", noisy_gauss, "flat_sigma", flat_region_sigma),
    ]

    for tag, img, key, measure in spec:
        base = measure(img)
        fig, axes = plt.subplots(3, 5, figsize=(7.5, 6.1))
        show_gray(axes[0, 0], img, "含噪图")
        show_gray(axes[1, 0], img[CROP], "局部 含噪图")
        axes[2, 0].axis("off")
        axes[2, 0].text(0.5, 0.5, "第三行显示残差图像\n等于含噪图减滤波结果",
                        ha="center", va="center", fontsize=9, color="dimgrey")
        rows.append({
            "noise": tag, "filter": "原始含噪图", "param": "-",
            key: r3(base), "noise_reduction": 0.0,
            "edge_slope": 1.0,
        })

        for j, (name, fn) in enumerate(FILTERS.items()):
            out = fn(img)
            imwrite_u(os.path.join(OUT_DIR, "cmp_%s_%s.png" % (tag, name)), out)
            val = measure(out)
            rows.append({
                "noise": tag, "filter": name, "param": "-",
                key: r3(val), "noise_reduction": r3(1 - val / base),
                "edge_slope": r3(edge_slope(img, out)),
            })
            show_gray(axes[0, j + 1], out, name)
            show_gray(axes[1, j + 1], out[CROP], "局部 " + name)
            resid = np.clip(img.astype(np.int16) - out.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            show_gray(axes[2, j + 1], resid[CROP], "残差 " + name)
        fig.tight_layout()
        save_fig(fig, os.path.join(FIG_DIR,
                  "fig04%s_comparison_%s.png" % ("a" if tag == "salt_pepper" else "b", tag)))
    return rows


def sec4_controlled(clean):
    print("[4] 受控实验: 合成已知噪声, 以真值评价 (PSNR / SSIM / EPI) ...")
    rng = np.random.default_rng(SEED)
    gs_noisy = add_gaussian_noise(clean, 20.0, rng)
    sp_noisy = add_salt_pepper_noise(clean, 0.05, rng)

    rows = []
    for tag, img in (("salt_pepper", sp_noisy), ("gaussian", gs_noisy)):
        rows.append({
            "noise": tag, "filter": "原始含噪图", "param": "-",
            "psnr": r3(psnr(clean, img)), "ssim": r3(ssim(clean, img)),
            "epi": r3(edge_preservation(clean, img)),
        })
        for name, fn in FILTERS.items():
            out = fn(img)
            imwrite_u(os.path.join(OUT_DIR, "ctrl_%s_%s.png" % (tag, name)), out)
            rows.append({
                "noise": tag, "filter": name, "param": "-",
                "psnr": r3(psnr(clean, out)), "ssim": r3(ssim(clean, out)),
                "epi": r3(edge_preservation(clean, out)),
            })

    sweep = []

    def add_sweep(group, x, fname, noise, out):
        sweep.append({
            "group": group, "x": x, "filter": fname, "noise": noise,
            "psnr": r3(psnr(clean, out)), "ssim": r3(ssim(clean, out)),
            "epi": r3(edge_preservation(clean, out)),
        })

    for k in [3, 5, 7, 9, 11, 13, 15]:
        add_sweep("kernel_gauss", k, "均值滤波", "gaussian", mean_filter(gs_noisy, k))
        add_sweep("kernel_gauss", k, "高斯滤波", "gaussian", gaussian_filter(gs_noisy, k))
        add_sweep("kernel_gauss", k, "中值滤波", "gaussian", median_filter(gs_noisy, k))

    for s in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]:
        add_sweep("sigma_gauss", s, "高斯滤波", "gaussian", gaussian_filter(gs_noisy, 11, s))

    for sc in [5, 10, 25, 50, 75, 100, 150, 200]:
        add_sweep("sigma_color", sc, "双边滤波", "gaussian", bilateral_filter(gs_noisy, 9, sc, 50))

    for k in [3, 5, 7, 9, 11]:
        add_sweep("kernel_sp", k, "中值滤波", "salt_pepper", median_filter(sp_noisy, k))
        add_sweep("kernel_sp", k, "均值滤波", "salt_pepper", mean_filter(sp_noisy, k))
        add_sweep("kernel_sp", k, "高斯滤波", "salt_pepper", gaussian_filter(sp_noisy, k))

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.9))

    def series(group, name):
        pts = [r for r in sweep if r["group"] == group and r["filter"] == name]
        return [p["x"] for p in pts], pts

    ax = axes[0, 0]
    for name in ("均值滤波", "高斯滤波", "中值滤波"):
        x, p = series("kernel_gauss", name)
        ax.plot(x, [q["psnr"] for q in p], "o-", color=COLORS[name], label=name, ms=4)
    ax.set_xlabel("核大小 k")
    ax.set_ylabel("PSNR / dB")
    ax.set_title("a 高斯噪声 核大小与 PSNR")
    despine(ax)
    ax.legend(**LEG)

    ax = axes[0, 1]
    for name in ("均值滤波", "高斯滤波", "中值滤波"):
        x, p = series("kernel_gauss", name)
        ax.plot(x, [q["ssim"] for q in p], "s-", color=COLORS[name], label=name, ms=4)
    ax.set_xlabel("核大小 k")
    ax.set_ylabel("SSIM")
    ax.set_title("b 高斯噪声 核大小与 SSIM")
    despine(ax)
    ax.legend(**LEG)

    ax = axes[0, 2]
    for name in ("均值滤波", "高斯滤波", "中值滤波"):
        x, p = series("kernel_gauss", name)
        ax.plot(x, [q["epi"] for q in p], "^-", color=COLORS[name], label=name, ms=4)
    ax.set_xlabel("核大小 k")
    ax.set_ylabel("EPI 边缘保持指数")
    ax.set_title("c 高斯噪声 核大小与边缘保持")
    despine(ax)
    ax.legend(**LEG)

    ax = axes[1, 0]
    x, p = series("sigma_gauss", "高斯滤波")
    l1, = ax.plot(x, [q["psnr"] for q in p], "o-", color="#4575b4", ms=4, label="PSNR")
    ax.set_xlabel("sigmaX 核固定 11x11")
    ax.set_ylabel("PSNR / dB", color="#4575b4")
    ax2 = ax.twinx()
    l2, = ax2.plot(x, [q["ssim"] for q in p], "s--", color="#009E73", ms=4, label="SSIM")
    ax2.set_ylabel("SSIM", color="#009E73")
    ax2.tick_params(axis="y", length=0, labelcolor="dimgrey")
    ax.set_title("d 高斯滤波 sigmaX 的影响")
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.8)
    ax.legend(handles=[l1, l2], loc="lower center", **LEG)

    ax = axes[1, 1]
    x, p = series("sigma_color", "双边滤波")
    l1, = ax.plot(x, [q["psnr"] for q in p], "o-", color="#E69F00", ms=4, label="PSNR")
    ax.set_xlabel("sigmaColor d=9 sigmaSpace=50")
    ax.set_ylabel("PSNR / dB", color="#E69F00")
    ax2 = ax.twinx()
    l2, = ax2.plot(x, [q["epi"] for q in p], "^--", color="#805ad5", ms=4, label="EPI")
    ax2.set_ylabel("EPI", color="#805ad5")
    ax2.tick_params(axis="y", length=0, labelcolor="dimgrey")
    ax.set_title("e 双边滤波 sigmaColor 的影响")
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.8)
    ax.legend(handles=[l1, l2], loc="lower center", **LEG)

    ax = axes[1, 2]
    for name in ("均值滤波", "高斯滤波", "中值滤波"):
        x, p = series("kernel_sp", name)
        ax.plot(x, [q["psnr"] for q in p], "o-", color=COLORS[name], label=name, ms=4)
    ax.set_xlabel("核大小 k")
    ax.set_ylabel("PSNR / dB")
    ax.set_title("f 椒盐噪声 核大小与 PSNR")
    despine(ax)
    ax.legend(**LEG)

    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig05_parameter_sweep.png"))
    return rows, sweep


def sec5_reference_check(clean, noisy_sp, noisy_gauss):
    print("[5] 交叉估计参考图与可信度校验 (补充分析) ...")
    rng = np.random.default_rng(SEED)

    sp_syn = add_salt_pepper_noise(clean, 0.05, rng)
    gs_syn = add_gaussian_noise(clean, 20.0, rng)
    est_syn = cross_image_reference(sp_syn, gs_syn)
    rows = [{
        "case": "合成数据 真值已知",
        "ref_psnr": r3(psnr(clean, est_syn)),
        "ref_ssim": r3(ssim(clean, est_syn)),
        "noisy_sp_psnr": r3(psnr(clean, sp_syn)),
        "noisy_gauss_psnr": r3(psnr(clean, gs_syn)),
    }]

    ref_real = cross_image_reference(noisy_sp, noisy_gauss)
    imwrite_u(os.path.join(OUT_DIR, "reference_estimated.png"), ref_real)

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.9))
    show_gray(axes[0], noisy_sp, "输入 noised1")
    show_gray(axes[1], noisy_gauss, "输入 noised2")
    show_gray(axes[2], ref_real, "交叉估计参考图")
    show_gray(axes[3], ref_real[CROP], "局部 估计参考图")
    fig.tight_layout()
    save_fig(fig, os.path.join(FIG_DIR, "fig06_reference_estimate.png"))
    return rows


def main():
    for d in (FIG_DIR, RES_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
    setup_plot_style()

    clean = imread_gray(os.path.join(IMG_DIR, "image_fft.png"))
    noisy_sp = imread_gray(os.path.join(IMG_DIR, "noised1.png"))
    noisy_gauss = imread_gray(os.path.join(IMG_DIR, "noised2.png"))
    print("图像读取完成: clean%s  noised1(椒盐)%s  noised2(高斯)%s\n"
          % (clean.shape, noisy_sp.shape, noisy_gauss.shape))

    m0 = sec0_noise_analysis(clean, noisy_sp, noisy_gauss)
    m1 = sec1_smoothing(noisy_gauss)
    m2 = sec2_denoising(noisy_sp)
    m3 = sec3_comparison(noisy_sp, noisy_gauss)
    m4, m4s = sec4_controlled(clean)
    m5 = sec5_reference_check(clean, noisy_sp, noisy_gauss)

    write_csv(os.path.join(RES_DIR, "metrics_00_noise_analysis.csv"), m0,
              ["image", "kind", "size", "mean", "std", "pixel_eq_0", "pixel_eq_255",
               "pixel_extreme_total", "sp_density_est", "gauss_sigma_est"])
    write_csv(os.path.join(RES_DIR, "metrics_01_smoothing.csv"), m1,
              ["filter", "ksize", "param", "flat_sigma", "noise_reduction", "edge_slope"])
    write_csv(os.path.join(RES_DIR, "metrics_02_denoising.csv"), m2,
              ["filter", "param", "residual_impulse", "impulse_removal", "flat_sigma", "edge_slope"])
    write_csv(os.path.join(RES_DIR, "metrics_03_comparison_real.csv"), m3,
              ["noise", "filter", "param", "flat_sigma", "residual_impulse", "noise_reduction", "edge_slope"])
    write_csv(os.path.join(RES_DIR, "metrics_04_controlled_truth.csv"), m4,
              ["noise", "filter", "param", "psnr", "ssim", "epi"])
    write_csv(os.path.join(RES_DIR, "metrics_05_parameter_sweep.csv"), m4s,
              ["group", "x", "filter", "noise", "psnr", "ssim", "epi"])
    write_csv(os.path.join(RES_DIR, "metrics_06_reference_check.csv"), m5,
              ["case", "ref_psnr", "ref_ssim", "noisy_sp_psnr", "noisy_gauss_psnr"])

    print("\n完成。图表 -> figures/ ; 数据 -> results/ ; 结果图 -> output/")


if __name__ == "__main__":
    main()
