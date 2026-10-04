# -*- coding: utf-8 -*-
"""实验四 卷积神经网络实践 主脚本。

运行: python experiment4_cnn.py           完整训练流程
      python experiment4_cnn.py figures   仅用已有产物重绘全部插图
输出: figures/ 插图, results/ 指标 CSV, output/ 模型权重
"""
import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "output")

SEED = 20241002
CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
PALETTE = ["#d73027", "#4575b4", "#009E73"]
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


def read_csv(name):
    with open(os.path.join(RES_DIR, name), "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def despine(ax):
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.8)


def build_loaders(aug=True, batch=128, workers=4):
    train_tf = [T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4)] if aug else []
    train_tf = T.Compose(train_tf + [T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, transform=train_tf, download=True)
    test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, transform=test_tf, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0)
    return train_set, test_set, train_loader, test_loader


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5, bn=False):
        super().__init__()
        b1 = [nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True)]
        if bn:
            b1.append(nn.BatchNorm2d(32))
        b1.append(nn.MaxPool2d(2))
        b2 = [nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True)]
        if bn:
            b2.append(nn.BatchNorm2d(64))
        b2.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*b1, *b2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (pred.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, collect=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        loss_sum += loss_fn(pred, y).item() * len(y)
        pred = pred.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
        if collect:
            preds.append(pred.cpu())
            targets.append(y.cpu())
    if collect:
        return loss_sum / total, correct / total, torch.cat(preds), torch.cat(targets)
    return loss_sum / total, correct / total


def run_experiment(name, group, setting, epochs, lr=1e-3, batch=128, aug=True,
                   dropout=0.5, bn=False, workers=4, device="cuda"):
    _, _, train_loader, test_loader = build_loaders(aug, batch, workers)
    model = SimpleCNN(dropout=dropout, bn=bn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
        history.append({"group": group, "setting": setting, "epoch": ep,
                        "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                        "test_loss": round(te_loss, 4), "test_acc": round(te_acc, 4)})
        print("      %s | epoch %2d | train %.4f | test %.4f" % (name, ep, tr_acc, te_acc))
    return model, history, time.perf_counter() - t0


def plot_samples():
    print("[fig01] 数据样例 ...")
    raw = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(raw), 40, replace=False)
    fig, axes = plt.subplots(5, 8, figsize=(7.5, 4.9))
    for k, ax in enumerate(axes.ravel()):
        img, label = raw[int(idx[k])]
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=8, pad=2, color="dimgrey")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)
    save_fig(fig, "fig01_cifar_samples.png")


def plot_model_arch(model):
    print("[fig02] 网络结构 ...")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    boxes = [
        ("输入 3x32x32", 0.2, 2.0, 1.4, 1.0, "#e2e8f0"),
        ("卷积 32 3x3\nReLU\n池化 2x2", 1.9, 1.8, 1.5, 1.4, "#bee3f8"),
        ("卷积 64 3x3\nReLU\n池化 2x2", 3.7, 1.8, 1.5, 1.4, "#bee3f8"),
        ("展平 4096", 5.5, 2.1, 1.2, 0.9, "#fefcbf"),
        ("全连接 256\nReLU\nDropout 0.5", 6.95, 1.8, 1.5, 1.4, "#c6f6d5"),
        ("全连接 10", 8.7, 2.1, 1.1, 0.9, "#fed7d7"),
    ]
    for text, x, y, w, h, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#4a5568", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)
    for x0, x1 in [(1.6, 1.9), (3.4, 3.7), (5.2, 5.5), (6.7, 6.95), (8.45, 8.7)]:
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.2))
    ax.set_title("网络结构示意 参数总量 %s" % format(count_params(model), ","))
    save_fig(fig, "fig02_model_arch.png")


def plot_curves(history):
    print("[fig03] 训练曲线 ...")
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", ms=3, color=PALETTE[1], label="训练损失")
    axes[0].plot(ep, [h["test_loss"] for h in history], "s--", ms=3, color=PALETTE[0], label="测试损失")
    axes[0].set_xlabel("轮数")
    axes[0].set_ylabel("损失")
    axes[0].set_title("损失曲线")
    axes[0].legend(**LEG)
    despine(axes[0])
    axes[1].plot(ep, [h["train_acc"] for h in history], "o-", ms=3, color=PALETTE[1], label="训练准确率")
    axes[1].plot(ep, [h["test_acc"] for h in history], "s--", ms=3, color=PALETTE[0], label="测试准确率")
    axes[1].set_xlabel("轮数")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].legend(**LEG)
    despine(axes[1])
    save_fig(fig, "fig03_baseline_curves.png")


def confusion_matrix(model, device):
    _, _, _, test_loader = build_loaders(False, 128, 4)
    _, _, preds, targets = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, collect=True)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(targets.numpy(), preds.numpy()):
        cm[t, p] += 1
    return cm, cm.diagonal() / cm.sum(1)


def plot_confusion(cm, per_class):
    print("[fig04] 混淆矩阵 ...")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("混淆矩阵")
    axes[0].set_xticks(range(10))
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(CLASSES, fontsize=8)
    axes[0].tick_params(length=0, labelcolor="dimgrey")
    axes[0].grid(False)
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8, length=0, labelcolor="dimgrey")
    order = np.argsort(per_class)[::-1]
    bars = axes[1].bar(range(10), per_class[order] * 100, color=PALETTE[1], alpha=.9)
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([CLASSES[i] for i in order], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("准确率 %")
    axes[1].set_title("各类别准确率")
    for b, v in zip(bars, per_class[order] * 100):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.5, "%.1f" % v,
                     ha="center", va="bottom", fontsize=7.5, color="dimgrey")
    despine(axes[1])
    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig04_confusion.png")


def plot_predictions(model, device):
    print("[fig06] 预测样例 ...")
    raw = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(len(raw), 16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(7.4, 7.8))
    for k, ax in enumerate(axes.ravel()):
        img, label = raw[int(idx[k])]
        with torch.no_grad():
            pred = model(tf(img).unsqueeze(0).to(device)).argmax(1).item()
        ax.imshow(img)
        color = "#2f855a" if pred == label else "#c53030"
        ax.set_title("预测 %s 实际 %s" % (CLASSES[pred], CLASSES[label]),
                     fontsize=8.5, color=color, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.patch.set_edgecolor(color)
        ax.patch.set_linewidth(1.6)
    save_fig(fig, "fig06_predictions.png")


def plot_compare():
    print("[fig05] 参数对比 ...")
    runs = read_csv("metrics_05_runs.csv")
    hist_all = read_csv("metrics_06_history_compare.csv")
    groups = ["学习率", "批大小", "数据增强", "Dropout", "BatchNorm"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.9))
    for ax, g in zip(axes.ravel()[:5], groups):
        subs = sorted({h["setting"] for h in hist_all if h["group"] == g})
        for ci, s in enumerate(subs):
            pts = [h for h in hist_all if h["group"] == g and h["setting"] == s]
            ax.plot([int(p["epoch"]) for p in pts], [float(p["test_acc"]) for p in pts],
                    "o-", ms=3, label=s, color=PALETTE[ci % len(PALETTE)])
        ax.set_xlabel("轮数")
        ax.set_ylabel("测试准确率")
        ax.set_title(g + " 对比")
        ax.legend(**LEG)
        despine(ax)
    ax = axes[1, 2]
    ax.axis("off")
    lines = []
    for g in groups:
        subs = [r for r in runs if r["group"] == g]
        best = max(subs, key=lambda r: float(r["best_test_acc"]))
        lines.append("%s 最优 %s  %.1f%%" % (g, best["setting"], 100 * float(best["best_test_acc"])))
    ax.text(0.02, 0.92, "\n".join(lines), fontsize=9.5, va="top", linespacing=2.0,
            color="dimgrey")
    fig.tight_layout(h_pad=2.2, w_pad=1.5)
    save_fig(fig, "fig05_param_compare.png")


def train_all(device):
    plot_samples()
    model = SimpleCNN(dropout=0.5, bn=False)
    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            rows.append({"layer": name, "type": type(mod).__name__,
                         "out_shape": str(mod.weight.shape),
                         "params": mod.weight.numel() + mod.bias.numel()})
    rows.append({"layer": "total", "type": "-", "out_shape": "-", "params": count_params(model)})
    write_csv("metrics_01_model_summary.csv", rows, ["layer", "type", "out_shape", "params"])
    plot_model_arch(model)

    model, history, dt = run_experiment("baseline", "baseline", "基线", 20,
                                        lr=1e-3, batch=128, aug=True, dropout=0.5,
                                        bn=False, device=device)
    write_csv("metrics_02_baseline_history.csv", history,
              ["group", "setting", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_baseline.pt"))
    write_csv("metrics_03_baseline_summary.csv",
              [{"name": "baseline", "epochs": 20,
                "best_test_acc": max(h["test_acc"] for h in history),
                "final_test_acc": history[-1]["test_acc"], "time_s": round(dt, 1)}],
              ["name", "epochs", "best_test_acc", "final_test_acc", "time_s"])
    plot_curves(history)

    cm, per_class = confusion_matrix(model, device)
    plot_confusion(cm, per_class)
    write_csv("metrics_04_per_class.csv",
              [{"class": CLASSES[i], "accuracy": round(float(per_class[i]), 4),
                "n_test": int(cm.sum(1)[i])} for i in range(10)],
              ["class", "accuracy", "n_test"])
    plot_predictions(model, device)

    runs, hist_all = [], []
    for lr in [1e-4, 1e-3, 1e-2]:
        _, hist, dt = run_experiment("学习率=%s" % lr, "学习率", str(lr), 10, lr=lr, device=device)
        runs.append({"group": "学习率", "setting": str(lr), "epochs": 10,
                     "best_test_acc": max(h["test_acc"] for h in hist),
                     "final_test_acc": hist[-1]["test_acc"], "time_s": round(dt, 1)})
        hist_all.extend(hist)
    for bs in [64, 128, 256]:
        _, hist, dt = run_experiment("批大小=%d" % bs, "批大小", str(bs), 10, batch=bs, device=device)
        runs.append({"group": "批大小", "setting": str(bs), "epochs": 10,
                     "best_test_acc": max(h["test_acc"] for h in hist),
                     "final_test_acc": hist[-1]["test_acc"], "time_s": round(dt, 1)})
        hist_all.extend(hist)
    for aug, label in ((True, "开启"), (False, "关闭")):
        _, hist, dt = run_experiment("数据增强=%s" % label, "数据增强", label, 10, aug=aug, device=device)
        runs.append({"group": "数据增强", "setting": label, "epochs": 10,
                     "best_test_acc": max(h["test_acc"] for h in hist),
                     "final_test_acc": hist[-1]["test_acc"], "time_s": round(dt, 1)})
        hist_all.extend(hist)
    for dp, label in ((0.0, "0.0"), (0.5, "0.5")):
        _, hist, dt = run_experiment("Dropout=%s" % label, "Dropout", label, 10, dropout=dp, device=device)
        runs.append({"group": "Dropout", "setting": label, "epochs": 10,
                     "best_test_acc": max(h["test_acc"] for h in hist),
                     "final_test_acc": hist[-1]["test_acc"], "time_s": round(dt, 1)})
        hist_all.extend(hist)
    for bn, label in ((True, "开启"), (False, "关闭")):
        _, hist, dt = run_experiment("BatchNorm=%s" % label, "BatchNorm", label, 10, bn=bn, device=device)
        runs.append({"group": "BatchNorm", "setting": label, "epochs": 10,
                     "best_test_acc": max(h["test_acc"] for h in hist),
                     "final_test_acc": hist[-1]["test_acc"], "time_s": round(dt, 1)})
        hist_all.extend(hist)

    write_csv("metrics_05_runs.csv", runs,
              ["group", "setting", "epochs", "best_test_acc", "final_test_acc", "time_s"])
    write_csv("metrics_06_history_compare.csv", hist_all,
              ["group", "setting", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
    plot_compare()


def redraw(device):
    plot_samples()
    model = SimpleCNN(dropout=0.5, bn=False).to(device)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model_baseline.pt"),
                                     map_location=device))
    plot_model_arch(model)
    num = ("train_loss", "train_acc", "test_loss", "test_acc")
    history = [{"epoch": int(h["epoch"]), **{k: float(h[k]) for k in num}}
               for h in read_csv("metrics_02_baseline_history.csv")]
    plot_curves(history)
    cm, per_class = confusion_matrix(model, device)
    plot_confusion(cm, per_class)
    plot_predictions(model, device)
    plot_compare()


def main():
    for d in (FIG_DIR, RES_DIR, OUT_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)
    setup_style()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备: %s" % device)
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        redraw(device)
    else:
        train_all(device)
    print("\n完成。图表 -> figures/ ; 数据 -> results/ ; 权重 -> output/")


if __name__ == "__main__":
    main()
