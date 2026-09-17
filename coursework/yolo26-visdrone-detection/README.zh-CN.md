# YOLO26 与 VisDrone 无人机目标检测实验五六

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="screenshots/01_实验五_YOLO26训练过程/08_训练指标总览曲线.png"><img src="screenshots/01_实验五_YOLO26训练过程/08_训练指标总览曲线.png" alt="YOLO26 训练曲线" width="49%" /></a>
  <a href="screenshots/03_实验六_YOLO26测试预测/01_测试图片预测结果01_visdrone_train_0000072_05827_d_0000009.jpg"><img src="screenshots/03_实验六_YOLO26测试预测/01_测试图片预测结果01_visdrone_train_0000072_05827_d_0000009.jpg" alt="VisDrone 预测结果与 obstacle 检测框" width="49%" /></a>
</p>

**使用 Ultralytics YOLO26 在 VisDrone 无人机视角子集上完成目标检测实验，服务于嵌入式系统课程设计。**

项目覆盖完整检测流程，从 VisDrone 子集准备与标签转换，到训练，验证，预测和 ONNX 导出。所有可能与无人机碰撞的目标，包括行人，车辆，摩托车和公交车，统一合并为一类 `obstacle`，贴合避障任务。训练从 `yolo26m.pt` 出发，产出 `best.pt`，训练曲线，混淆矩阵和预测图保存在 `screenshots/`。

**状态。** 实验五和实验六已完成。在 677 张验证图片和 39961 个实例上，结果为精确率 0.809，召回率 0.638，mAP50 为 0.703，mAP50-95 为 0.415。训练权重已导出为 `best.onnx`，并用 Netron 查看网络结构。

## 我的职责

独立完成 VisDrone obstacle 子集准备与标签转换（`scripts/05_prepare_visdrone_subset.py`、`01_split_yolo_dataset.py`、`02_make_data_yaml.py`），并运行 YOLO26m 训练，验证，预测与 ONNX 导出全流程（`03_train_yolo26.py`、`04_exp6_val_predict_export.py`、`06_extract_real_video_frames.py`），产出 `screenshots/` 中的过程证据与 `docs/` 中的实验说明。报告指标来自 677 张验证图片和 39961 个实例的已保存验证结果。

## 实验流程

```text
VisDrone 子集准备 → 训练集与验证集划分 → 数据集 YAML → YOLO26m 训练
→ 验证与预测 → ONNX 导出 → 可选的真实视频抽帧
```

## 脚本一览

| 脚本 | 作用 |
| --- | --- |
| `00_smoke_test_yolo26.py` | 用官方权重做环境冒烟测试 |
| `05_prepare_visdrone_subset.py` | 构建 VisDrone 子集并转换标签 |
| `01_split_yolo_dataset.py` | 训练集与验证集划分 |
| `02_make_data_yaml.py` | 生成数据集 YAML |
| `03_train_yolo26.py` | 训练 YOLO26 |
| `04_exp6_val_predict_export.py` | 验证，预测与导出 ONNX |
| `06_extract_real_video_frames.py` | 从真实无人机视频中抽帧 |

## 快速开始

Python 3.10 与 Ultralytics。

```bash
pip install -r requirements_yolo26.txt
python scripts/00_smoke_test_yolo26.py
```

完整实验按以下顺序执行。

```bash
python scripts/05_prepare_visdrone_subset.py --mode obstacle --max-images 300 --clean
python scripts/01_split_yolo_dataset.py
python scripts/02_make_data_yaml.py
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 4 --epochs 15
python scripts/04_exp6_val_predict_export.py --device 0
```

ONNX 导出也可以单独执行。

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True simplify=False
```

## 目录内容

- `scripts/`，上述实验流程脚本。
- `configs/`，类别文件与数据集 YAML。
- `detection-samples/`，预测演示图片。
- `screenshots/`，实验五和实验六的全过程证据截图。
- `docs/`，无人机图片来源与 Sim2Real 真实相机数据建议。
- `yolo26n.pt`，保留用于快速测试的轻量官方权重。

## 文档入口

- [README.md](README.md)，英文页面。
- [YOLO26实验五六项目详解.md](YOLO26实验五六项目详解.md)，项目详解。
- [VisDrone子集_YOLO26m_标注训练推理步骤.md](VisDrone子集_YOLO26m_标注训练推理步骤.md)，VisDrone 与 YOLO26m 步骤说明。

VisDrone 数据集与 Ultralytics 权重沿用各自原始条款。

*Bohan Yu，嵌入式系统课程设计。*
