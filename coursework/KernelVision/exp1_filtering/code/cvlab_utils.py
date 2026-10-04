# -*- coding: utf-8 -*-
"""实验一 图像空域滤波 公共工具模块。

包含中文路径图像读写、PSNR 与 SSIM 与 EPI 指标、噪声估计与合成、绘图风格。
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_plot_style():
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


def psnr(ref, test, data_range=255.0):
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _gauss_1d(size, sigma):
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def ssim(ref, test, data_range=255.0, k1=0.01, k2=0.03, win=11, sigma=1.5):
    a = ref.astype(np.float64)
    b = test.astype(np.float64)
    g = _gauss_1d(win, sigma)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    def blur(x):
        return cv2.sepFilter2D(x, cv2.CV_64F, g, g, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a2 = blur(a * a) - mu_a2
    sig_b2 = blur(b * b) - mu_b2
    sig_ab = blur(a * b) - mu_ab
    num = (2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    return float(np.mean(num / den))


def gradient_magnitude(img):
    f = img.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge_preservation(ref, test, ratio=0.10):
    gr = gradient_magnitude(ref)
    gt = gradient_magnitude(test)
    mask = gr >= ratio * gr.max()
    return float(np.corrcoef(gr[mask], gt[mask])[0, 1])


def estimate_noise_sigma(img):
    f = img.astype(np.float64)
    a = f[0::2, 0::2]
    b = f[0::2, 1::2]
    c = f[1::2, 0::2]
    d = f[1::2, 1::2]
    detail = (a - b - c + d) / 2.0
    return float(np.median(np.abs(detail - np.median(detail))) / 0.6745)


def estimate_sp_density(img, thresh=100):
    med = cv2.medianBlur(img, 5).astype(np.int16)
    return float(np.mean(np.abs(img.astype(np.int16) - med) > thresh))


def add_gaussian_noise(img, sigma, rng):
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, density, rng, salt_ratio=0.5):
    out = img.copy()
    m = rng.random(img.shape)
    out[m < density * salt_ratio] = 255
    out[(m >= density * salt_ratio) & (m < density)] = 0
    return out


def flat_region_sigma(img, blk=8, low_frac=0.20):
    f = img.astype(np.float64)
    gm = gradient_magnitude(img)
    H, W = img.shape
    blocks = []
    for i in range(0, H - blk + 1, blk):
        for j in range(0, W - blk + 1, blk):
            blocks.append((float(gm[i:i + blk, j:j + blk].mean()), i, j))
    blocks.sort(key=lambda t: t[0])
    keep = blocks[: max(1, int(len(blocks) * low_frac))]
    return float(np.mean([f[i:i + blk, j:j + blk].std() for _, i, j in keep]))


def edge_slope(noisy, filtered, ratio=0.25):
    gn = gradient_magnitude(noisy)
    gf = gradient_magnitude(filtered)
    mask = gn >= ratio * gn.max()
    x, y = gn[mask], gf[mask]
    return float(np.sum(x * y) / np.sum(x * x))


def cross_image_reference(img_sp, img_gauss, h=6.0):
    med5 = cv2.medianBlur(img_sp, 5)
    mask = np.abs(img_sp.astype(np.int16) - med5.astype(np.int16)) > 100
    repaired = img_sp.astype(np.float64).copy()
    repaired[mask] = med5[mask]
    fused = img_gauss.astype(np.float64).copy()
    fused[~mask] = (repaired[~mask] + img_gauss.astype(np.float64)[~mask]) / 2.0
    fused_u8 = np.clip(np.floor(fused + 0.5), 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(fused_u8, None, h, 7, 21)


def show_gray(ax, img, title, vmin=0, vmax=255):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
