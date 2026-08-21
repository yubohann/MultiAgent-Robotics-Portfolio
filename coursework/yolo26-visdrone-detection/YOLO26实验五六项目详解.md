# YOLO26 实验五六项目详解

本文档用于详细说明 `exp5_6_yolo26` 目录下的完整实验项目。该目录对应课程中的实验五和实验六，主题是使用 Ultralytics YOLO26 完成目标检测实验，包括环境配置、数据集准备、标注格式整理、模型训练、模型验证、图片推理、结果可视化和 ONNX 模型导出。

项目路径：

```text
exp5_6_yolo26
```

Ubuntu/WSL 中建议使用的实验路径：

```bash
~/yubohan_[REDACTED]_exp5_6_yolo26
```

## 1. 项目总体目标

本项目的目标是把原指导书中基于 YOLOv8 的目标检测实验替换为 YOLO26，并完成一套可截图、可复现、可写入实验报告的完整流程。

实验五主要完成：

```text
1. 创建 YOLO26 运行环境；
2. 安装 Ultralytics、PyTorch、ONNX 等依赖；
3. 验证 YOLO26 官方权重可以推理；
4. 准备真实无人机视角数据集；
5. 将 VisDrone 标注转换为 YOLO 格式；
6. 划分训练集和验证集；
7. 生成数据集 YAML；
8. 使用 yolo26m.pt 训练自定义检测模型；
9. 保存 best.pt、last.pt、results.png 等训练结果。
```

实验六主要完成：

```text
1. 使用实验五训练得到的 best.pt 进行验证；
2. 查看 Precision、Recall、mAP50、mAP50-95 等指标；
3. 对 detect_test 目录中的图片进行预测；
4. 查看预测框和置信度；
5. 使用脚本完成验证、预测和导出；
6. 将 best.pt 导出为 ONNX；
7. 使用 Netron 查看 ONNX 网络结构；
8. 整理截图材料。
```

一句话概括：

> 该项目完成了从 YOLO26 环境配置、VisDrone 数据集转换、YOLO26m 训练，到模型验证、推理和 ONNX 导出的完整目标检测实验链路。

## 2. 目录结构说明

项目根目录主要内容如下：

```text
exp5_6_yolo26/
├── README.md
├── requirements_yolo26.txt
├── yolo26m.pt
├── yolo26n.pt
├── configs/
├── data_raw/
├── my_data/
├── detect_test/
├── scripts/
├── docs/
├── screenshots/
├── assets/
├── external/
├── VisDrone子集_YOLO26m_标注训练推理步骤.md
├── 实验五_YOLO26_截图提纲.md
├── 实验六_YOLO26_截图提纲.md
├── 实验五六_YOLO26_实验步骤与截图提纲.md
└── YOLO26_实验五六_纯截图提纲汇总.md
```

各目录作用如下。

| 路径 | 作用 |
|---|---|
| `README.md` | 主操作手册，按实验五和实验六组织命令 |
| `requirements_yolo26.txt` | 简化依赖列表 |
| `yolo26m.pt` | YOLO26m 官方预训练权重，本项目训练主模型 |
| `yolo26n.pt` | YOLO26n 轻量权重，可用于快速测试 |
| `configs/` | 类别文件和数据集 YAML 配置 |
| `data_raw/` | 原始图片和原始 YOLO 标签 |
| `my_data/` | 划分后的训练集和验证集 |
| `detect_test/` | 推理测试图片 |
| `scripts/` | 自动化脚本 |
| `docs/` | 数据来源、Sim2Real 数据建议等说明 |
| `screenshots/` | 实验五、实验六截图保存目录 |
| `assets/` | 烟雾测试用图片，如 `bus.jpg` |
| `external/` | 外部下载数据，如 VisDrone 原始数据 |

## 3. 核心文件说明

### 3.1 `README.md`

这是本项目最主要的操作文档。它按实验五、实验六顺序给出命令和截图要求。

内容包括：

```text
环境检查；
Conda 环境创建；
依赖安装；
YOLO26 smoke test；
数据集准备；
类别文件配置；
训练/验证集划分；
训练 YOLO26；
验证模型；
预测测试图片；
导出 ONNX；
Netron 查看结构。
```

### 3.2 `VisDrone子集_YOLO26m_标注训练推理步骤.md`

这是专门针对 VisDrone 子集和 YOLO26m 模型的精简流程文档。

它明确了当前实验方案：

```text
真实无人机数据：VisDrone
类别策略：所有可碰撞目标合并为 obstacle
模型：yolo26m.pt
流程：下载/转换标注 -> 划分数据集 -> 训练 -> 验证/推理 -> 导出 ONNX
```

### 3.3 `YOLO26_实验五六_纯截图提纲汇总.md`

这是纯截图提纲，不包含详细命令。它用于整理报告中需要插入的截图。

其中实验五约 31 张截图，实验六约 26 张截图，总计约 57 张。

### 3.4 `configs/classes.txt`

类别文件，每行一个类别名。

在 VisDrone obstacle 单类方案中，它应为：

```text
obstacle
```

如果使用原始 VisDrone 10 类，则内容为：

```text
pedestrian
people
bicycle
car
van
truck
tricycle
awning-tricycle
bus
motor
```

### 3.5 `configs/my_detect.yaml`

这是模板 YAML，用于说明数据集配置结构。它不是最终训练时最推荐使用的文件，因为路径是相对路径。

模板内容类似：

```yaml
path: ../my_data/detection
train: images/train
val: images/val

names:
  0: person
  1: surfboard
```

实际训练应使用脚本生成的：

```text
configs/my_detect.local.yaml
```

该文件会写入当前机器的绝对路径，减少路径错误。

## 4. `scripts` 脚本详解

`scripts/` 是本项目自动化流程的核心。

### 4.1 `00_smoke_test_yolo26.py`

作用：验证 YOLO26 环境和官方权重是否可以正常推理。

主要逻辑：

```text
1. 检查 assets/bus.jpg 是否存在；
2. 若不存在，则下载 Ultralytics 示例 bus.jpg；
3. 打印 ultralytics 和 torch 版本；
4. 检查 CUDA 是否可用；
5. 优先使用本地 yolo26m.pt；
6. 调用 YOLO 模型对 bus.jpg 推理；
7. 保存结果到 runs/smoke_yolo26/predict。
```

运行命令：

```bash
python scripts/00_smoke_test_yolo26.py
```

成功结果：

```text
runs/smoke_yolo26/predict/bus.jpg
```

这一步证明：

```text
Ultralytics 可导入；
PyTorch 可用；
YOLO26 权重可加载；
predict 流程可运行；
结果图片能正常保存。
```

### 4.2 `01_split_yolo_dataset.py`

作用：将 `data_raw/images` 和 `data_raw/labels` 中的原始图片/标签划分为训练集和验证集。

输入：

```text
data_raw/images/
data_raw/labels/
```

输出：

```text
my_data/detection/images/train/
my_data/detection/images/val/
my_data/detection/labels/train/
my_data/detection/labels/val/
```

划分比例：

```text
训练集：80%
验证集：20%
随机种子：2026
```

要求：

```text
图片和标签必须同名；
例如 image_001.jpg 对应 image_001.txt。
```

运行命令：

```bash
python scripts/01_split_yolo_dataset.py
```

终端会显示：

```text
Total images
Valid pairs
Train pairs
Val pairs
Missing labels
Dataset written to
```

### 4.3 `02_make_data_yaml.py`

作用：根据 `configs/classes.txt` 和当前机器绝对路径生成 `configs/my_detect.local.yaml`。

为什么要生成 local YAML：

```text
Ultralytics 训练时需要知道数据集根目录；
不同电脑/WSL/Ubuntu 路径不同；
使用绝对路径可以避免相对路径解析错误。
```

运行命令：

```bash
python scripts/02_make_data_yaml.py
```

输出示例：

```yaml
path: '/home/ybh/yubohan_[REDACTED]_exp5_6_yolo26/my_data/detection'
train: images/train
val: images/val

names:
  0: 'obstacle'
```

### 4.4 `03_train_yolo26.py`

作用：训练 YOLO26 模型。

默认参数：

```text
模型：yolo26m.pt
epochs：15
imgsz：640
batch：4
workers：2
输出目录：runs/yolo26/train
```

运行命令：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 4 --epochs 15
```

显存不足时：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 2 --epochs 15
```

没有 GPU 时：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device cpu --batch 2 --epochs 5
```

主要输出：

```text
runs/yolo26/train/weights/best.pt
runs/yolo26/train/weights/last.pt
runs/yolo26/train/results.png
runs/yolo26/train/confusion_matrix.png
```

其中：

```text
best.pt：验证集表现最好的权重；
last.pt：最后一个 epoch 的权重；
results.png：loss、precision、recall、mAP 曲线；
confusion_matrix.png：混淆矩阵。
```

### 4.5 `04_exp6_val_predict_export.py`

作用：实验六一键完成验证、预测和 ONNX 导出。

默认输入：

```text
权重：runs/yolo26/train/weights/best.pt
数据集配置：configs/my_detect.local.yaml
预测图片：detect_test/
```

运行命令：

```bash
python scripts/04_exp6_val_predict_export.py --device 0
```

执行流程：

```text
1. 使用 best.pt 在验证集上 val；
2. 输出 Box(P)、R、mAP50、mAP50-95；
3. 对 detect_test 目录图片 predict；
4. 保存预测图片到 runs/yolo26/predict；
5. 导出 ONNX 模型；
6. 输出 ONNX 路径。
```

如果 ONNX 导出时因为 `onnxslim` 或 `onnxruntime-gpu` 卡住，可以手动安装依赖：

```bash
python -m pip install -U onnx onnxslim onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果出现 `free(): double free detected in tcache 2`，可以关闭 simplify：

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True simplify=False
```

### 4.6 `05_prepare_visdrone_subset.py`

作用：自动下载 VisDrone 数据集，并将官方标注转换为 YOLO 格式。

运行命令：

```bash
python scripts/05_prepare_visdrone_subset.py --mode obstacle --max-images 500 --clean
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--mode obstacle` | 将所有可碰撞类别合并为 obstacle |
| `--mode visdrone10` | 保留 VisDrone 原始 10 类 |
| `--max-images 500` | 最多复制 500 张有效图片 |
| `--clean` | 清空旧的 `data_raw/images` 和 `data_raw/labels` |

VisDrone 原始类别：

```text
pedestrian
people
bicycle
car
van
truck
tricycle
awning-tricycle
bus
motor
```

在无人机避障实验中，这些目标都可能成为碰撞风险，因此可以统一成：

```text
obstacle
```

输出：

```text
data_raw/images/
data_raw/labels/
configs/classes.txt
```

### 4.7 `06_extract_real_video_frames.py`

作用：从真实视频中抽帧，用于补充真实相机图像或检测测试图片。

适用场景：

```text
1. 有一段无人机/摄像头视频；
2. 想提取若干帧作为 detect_test 图片；
3. 想后续人工标注后加入训练集；
4. 想补充 Sim2Real 真实域图像。
```

## 5. 数据集设计详解

### 5.1 为什么选择 VisDrone

本项目选择 VisDrone，原因是：

```text
1. 它是真实无人机视角数据集；
2. 包含行人、车辆、非机动车等目标；
3. 有官方检测框标注；
4. 可以自动转换为 YOLO 格式；
5. 适合课程实验五/六快速训练；
6. 和无人机避障视觉感知任务有较强关联。
```

相比普通 COCO 或 VOC，VisDrone 的视角更接近无人机航拍或低空监控；相比 ODA、UZH-FPV 等真实飞行数据，它又有更完整的目标检测框标注，因此更适合 YOLO26 实验。

### 5.2 为什么合并为 obstacle

无人机避障任务关心的是：

```text
前方或视野中的哪些物体可能碰撞
```

因此行人、车辆、自行车、摩托车、公交车等类别都可以统一视为障碍物。

合并后的好处：

```text
1. 类别数少，训练更稳定；
2. 实验报告更容易解释；
3. 单类 obstacle 符合避障感知需求；
4. 数据量集中，不会被多类别样本不均衡影响太大；
5. 验证指标只关注障碍物检测能力。
```

合并后的 `classes.txt`：

```text
obstacle
```

标签格式：

```text
0 x_center y_center width height
```

### 5.3 YOLO 标签格式

YOLO 检测标签每行表示一个目标：

```text
class_id x_center y_center width height
```

所有坐标都是相对图像宽高归一化后的值，范围通常为 0 到 1。

示例：

```text
0 0.513281 0.482812 0.326563 0.421875
```

含义：

```text
类别编号：0，即 obstacle
中心点 x：图像宽度的 0.513281
中心点 y：图像高度的 0.482812
框宽：图像宽度的 0.326563
框高：图像高度的 0.421875
```

## 6. 实验五完整流程

### 6.1 环境准备

进入项目目录：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
```

创建环境：

```bash
conda create -n yolo26 python=3.10 -y
conda activate yolo26
```

安装依赖：

```bash
python -m pip install -U pillow ultralytics onnx netron opencv-python torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple
```

检查环境：

```bash
nvidia-smi
nvcc --version
yolo checks
python -c "import ultralytics, torch, cv2; print(ultralytics.__version__); print(torch.__version__); print('cv2 ok')"
```

### 6.2 YOLO26 官方权重测试

运行：

```bash
python scripts/00_smoke_test_yolo26.py
```

查看：

```bash
ls runs/smoke_yolo26/predict
```

这一步证明：

```text
YOLO26 权重可加载；
Ultralytics 预测接口可用；
环境安装正确；
项目路径没有问题。
```

### 6.3 准备 VisDrone 子集

运行：

```bash
python scripts/05_prepare_visdrone_subset.py --mode obstacle --max-images 500 --clean
```

检查：

```bash
ls data_raw/images | head
ls data_raw/labels | head
cat configs/classes.txt
head -5 data_raw/labels/*.txt
```

应看到：

```text
data_raw/images 中有无人机图片；
data_raw/labels 中有同名 txt；
classes.txt 为 obstacle；
标签为 YOLO 五列格式。
```

### 6.4 划分训练集和验证集

运行：

```bash
python scripts/01_split_yolo_dataset.py
```

检查：

```bash
find my_data/detection -maxdepth 3 -type f | head -30
```

应看到：

```text
images/train
images/val
labels/train
labels/val
```

### 6.5 生成数据集 YAML

运行：

```bash
python scripts/02_make_data_yaml.py
cat configs/my_detect.local.yaml
```

YAML 中应包含：

```text
path
train
val
names
```

### 6.6 训练 YOLO26m

推荐命令：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 4 --epochs 15
```

如果显存不足：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 2 --epochs 15
```

训练时重点关注：

```text
epoch
box loss
cls loss
precision
recall
mAP50
mAP50-95
```

训练结束后检查：

```bash
ls -lh runs/yolo26/train
ls -lh runs/yolo26/train/weights
```

核心结果：

```text
best.pt
last.pt
results.png
confusion_matrix.png
```

## 7. 实验六完整流程

### 7.1 检查 best.pt

实验六依赖实验五训练得到的权重：

```bash
ls -lh runs/yolo26/train/weights/best.pt
```

如果没有 `best.pt`，需要先完成实验五训练。

### 7.2 准备测试图片

可以从原始图片复制 30 张：

```bash
rm -f detect_test/*
find data_raw/images -type f | head -30 | xargs -I{} cp "{}" detect_test/
ls -lh detect_test | head
```

### 7.3 验证模型

命令：

```bash
yolo detect val model=runs/yolo26/train/weights/best.pt data=configs/my_detect.local.yaml imgsz=640 device=0
```

输出中重点关注：

```text
Class
Images
Instances
Box(P)
R
mAP50
mAP50-95
```

这些指标含义：

| 指标 | 含义 |
|---|---|
| Box(P) | Precision，预测为目标的框中有多少是真的 |
| R | Recall，真实目标中有多少被检测出来 |
| mAP50 | IoU=0.5 条件下的平均精度 |
| mAP50-95 | IoU=0.5 到 0.95 的综合平均精度 |

你之前运行得到的典型结果是：

```text
all 677 images 39961 instances
Box(P)=0.809
R=0.638
mAP50=0.703
mAP50-95=0.415
```

可以解释为：

```text
模型在 VisDrone obstacle 单类任务上已经具备较好的检测能力；
Precision 较高，说明误检相对可控；
Recall 仍有提升空间，说明远处小目标、遮挡目标仍可能漏检；
mAP50-95 低于 mAP50 是正常现象，因为高 IoU 阈值对框定位精度要求更严格。
```

### 7.4 预测测试图片

命令：

```bash
yolo detect predict model=runs/yolo26/train/weights/best.pt source=detect_test save=True conf=0.5 imgsz=640 device=0
```

结果目录可能为：

```text
runs/detect/predict
runs/yolo26/predict
```

查看：

```bash
ls runs/yolo26/predict
```

预测图片中应出现：

```text
obstacle 检测框
置信度
不同无人机场景目标框
```

### 7.5 脚本一键验证、预测和导出

命令：

```bash
python scripts/04_exp6_val_predict_export.py --device 0
```

它会依次完成：

```text
model.val
model.predict
model.export(format='onnx')
```

适合截图，因为终端会连续显示验证、预测和导出阶段。

### 7.6 导出 ONNX

推荐稳定命令：

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True simplify=False
```

为什么加 `simplify=False`：

```text
在当前环境中，onnxslim 或 onnxruntime-gpu 可能触发 double free 或卡住；
simplify=False 可以跳过简化过程，提高导出成功率；
对课程实验而言，只要能成功得到 best.onnx 即可。
```

检查：

```bash
ls -lh runs/yolo26/train/weights/*.onnx
```

成功结果：

```text
runs/yolo26/train/weights/best.onnx
```

### 7.7 Netron 查看 ONNX

启动 Netron：

```bash
netron runs/yolo26/train/weights/best.onnx --host 127.0.0.1 --port 8080
```

浏览器打开：

```text
http://127.0.0.1:8080
```

截图应包含：

```text
YOLO26 ONNX 网络结构；
输入节点；
输出节点；
张量 shape；
模型层级结构。
```

## 8. YOLO26 和 YOLOv8 的替换关系

原课程指导书可能使用 YOLOv8。本项目将其替换为 YOLO26。

替换点包括：

```text
yolov8n.pt / yolov8m.pt → yolo26n.pt / yolo26m.pt
训练脚本名称改为 train_yolo26
输出目录统一为 runs/yolo26
README 和截图提纲全部改为 YOLO26
导出仍使用 Ultralytics export 接口
```

为什么可以替换：

```text
YOLO26 同样由 Ultralytics 接口管理；
训练、验证、预测、导出命令形式基本一致；
数据集 YAML 和 YOLO 标签格式不变；
实验目标仍是目标检测模型训练与部署。
```

报告中可以写：

> 本实验在保持 YOLO 数据格式、训练流程和 Ultralytics 工具链一致的前提下，将原 YOLOv8 模型替换为 YOLO26 模型。替换后仍使用 `model.train`、`model.val`、`model.predict` 和 `model.export` 完成训练、验证、推理和 ONNX 导出，实验流程与指导书要求保持一致。

## 9. 输出结果如何理解

### 9.1 `runs/yolo26/train`

训练结果目录。

常见文件：

| 文件 | 作用 |
|---|---|
| `weights/best.pt` | 验证集指标最好的模型 |
| `weights/last.pt` | 最后一次训练保存的模型 |
| `results.png` | 训练损失和指标曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `PR_curve.png` | Precision-Recall 曲线 |
| `F1_curve.png` | F1-Confidence 曲线 |

### 9.2 `runs/yolo26/val`

验证结果目录。

可能包含：

```text
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
val_batch0_pred.jpg
val_batch0_labels.jpg
```

### 9.3 `runs/yolo26/predict`

预测结果目录。

里面是对 `detect_test` 图片推理后的可视化结果。

### 9.4 `best.onnx`

ONNX 是模型交换格式，可用于：

```text
Netron 可视化；
ONNX Runtime 推理；
后续部署到其他推理框架；
作为实验六导出成果。
```

注意：

```text
导出成功不等于一定能在旧版 Tengine 上直接运行；
现代 YOLO26 ONNX 包含的算子和后处理可能与 Tengine 1.3.2 不兼容；
课程实验六的重点是完成 ONNX 导出和结构查看。
```

## 10. 常见问题详解

### 10.1 `onnxruntime-gpu not found` 卡住

原因：

```text
Ultralytics export 会自动检查导出相关依赖；
如果缺少 onnxslim 或 onnxruntime-gpu，它可能尝试 AutoUpdate；
网络慢或环境冲突时会卡住。
```

解决：

```bash
python -m pip install -U onnx onnxslim onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果仍然失败：

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True simplify=False
```

### 10.2 `free(): double free detected in tcache 2`

原因：

```text
通常是 onnx / onnxslim / onnxruntime / torch 版本组合导致的底层内存释放问题；
不是训练权重损坏；
多发生在 ONNX 简化阶段。
```

解决：

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True simplify=False
```

你之前已经用这个命令成功导出：

```text
runs/yolo26/train/weights/best.onnx
```

### 10.3 为什么验证结果很多数组输出

`model.val()` 返回的是 Ultralytics 的 `DetMetrics` 对象。直接 `print(metrics)` 时，可能会打印大量曲线数组。

这不是错误。真正需要关注的是终端表格：

```text
Class Images Instances Box(P) R mAP50 mAP50-95
```

截图时保留表格即可，不需要截完整数组。

### 10.4 为什么训练速度慢

影响因素：

```text
GPU 型号；
batch size；
imgsz；
数据量；
workers；
是否使用 CPU；
是否在 WSL 中训练。
```

如果显存不足：

```bash
--batch 2
```

如果只为截图：

```bash
--epochs 5
```

如果正式结果：

```bash
--epochs 15 或更多
```

### 10.5 为什么用 obstacle 单类而不是 10 类

课程实验重点是跑通训练、验证、预测和导出。单类 obstacle 有以下优点：

```text
类别解释更贴合无人机避障；
数据分布更集中；
训练更稳定；
指标更容易解释；
截图中类别名称统一。
```

如果要展示更丰富分类效果，可以改用：

```bash
python scripts/05_prepare_visdrone_subset.py --mode visdrone10 --max-images 500 --clean
```

## 11. 截图材料组织

本项目已经准备了多个截图提纲文档。

推荐使用：

```text
YOLO26_实验五六_纯截图提纲汇总.md
```

实验五重点截图：

```text
GPU/CUDA 检查；
Conda 环境；
依赖安装；
YOLO26 smoke test；
VisDrone 数据准备；
YOLO 标签格式；
训练集/验证集划分；
my_detect.local.yaml；
训练过程；
best.pt 和 results.png。
```

实验六重点截图：

```text
best.pt 存在；
detect_test 测试图片；
验证指标表格；
预测过程输出；
预测结果图片；
04_exp6_val_predict_export.py 脚本输出；
ONNX export success；
best.onnx 文件；
Netron 网络结构。
```

## 12. 实验报告写法建议

### 12.1 实验五可以这样写

> 实验五完成 YOLO26 目标检测环境配置和自定义数据集训练。首先在 Ubuntu 中创建 Python 3.10 的 Conda 环境，安装 Ultralytics、PyTorch、ONNX、OpenCV 等依赖，并通过 `00_smoke_test_yolo26.py` 验证 YOLO26 官方权重可正常推理。随后使用 VisDrone 真实无人机视角数据集，将 pedestrian、people、car、bus、motor 等可碰撞类别统一转换为 obstacle，并生成 YOLO 五列格式标签。通过脚本划分训练集和验证集，生成 `my_detect.local.yaml` 数据集配置，最终使用 `yolo26m.pt` 进行迁移训练，得到 `best.pt`、`last.pt` 和训练曲线图。

### 12.2 实验六可以这样写

> 实验六基于实验五训练得到的 `best.pt` 完成模型验证、推理和导出。验证阶段使用 `configs/my_detect.local.yaml` 在验证集上计算 Precision、Recall、mAP50 和 mAP50-95 等指标；预测阶段对 `detect_test` 目录中的无人机视角图片进行推理，并保存带有 obstacle 检测框和置信度的结果图；导出阶段将 PyTorch 权重转换为 ONNX 模型，并使用 Netron 查看网络输入、输出和结构。该实验验证了 YOLO26 自训练模型从训练结果到部署格式转换的完整链路。

## 13. 答辩常见问题和答案

### 13.1 这个 YOLO26 项目做了什么

答：

> 这个项目完成了 YOLO26 目标检测模型的训练、验证、预测和 ONNX 导出。数据集使用 VisDrone 真实无人机视角图片，并将行人、车辆、摩托车等可碰撞目标合并为 obstacle 类，最终训练出一个用于无人机避障视觉感知的单类检测模型。

### 13.2 为什么使用 VisDrone

答：

> VisDrone 是真实无人机视角目标检测数据集，包含行人、车辆、非机动车等多类目标，并且有官方检测框标注。它比普通 COCO/VOC 更符合无人机视觉感知场景，也比很多真实飞行数据更容易直接转换成 YOLO 格式训练。

### 13.3 为什么把多个类别合并成 obstacle

答：

> 在无人机避障任务中，行人、车辆、自行车、摩托车等目标都可能成为碰撞风险。实验重点不是细分类别识别，而是检测可能影响飞行安全的障碍物，因此统一为 obstacle 更符合避障任务目标。

### 13.4 `best.pt` 和 `best.onnx` 有什么区别

答：

> `best.pt` 是 PyTorch/Ultralytics 格式的训练权重，适合继续验证和推理；`best.onnx` 是导出的开放模型交换格式，适合用 Netron 查看结构，也可以用于 ONNX Runtime 或其他推理框架部署。

### 13.5 mAP50 和 mAP50-95 有什么区别

答：

> mAP50 是 IoU 阈值为 0.5 时的平均精度，对框定位要求相对宽松；mAP50-95 是从 0.5 到 0.95 多个 IoU 阈值的平均结果，对边界框定位更严格，因此通常低于 mAP50。

### 13.6 为什么导出 ONNX 时要用 `simplify=False`

答：

> 当前环境中 ONNX 简化依赖可能和 torch、onnxruntime 版本存在兼容问题，导致 double free 或卡住。关闭 simplify 可以跳过图简化步骤，优先保证 ONNX 导出成功。

### 13.7 这个模型能不能直接上板子跑

答：

> `best.onnx` 是现代 YOLO26 ONNX 模型，能否直接在旧版 Tengine 1.3.2 上运行取决于算子支持和后处理兼容性。旧版 Tengine 对现代 YOLO ONNX 支持有限，因此课程的板端实时项目最终选择了 MobileNetSSD；YOLO26 实验主要用于 PC/Ubuntu 侧完成训练、验证、预测和模型导出。

## 14. 和板端 MobileNetSSD 项目的关系

本目录是 YOLO26 实验五六项目，主要在 PC/Ubuntu 侧完成深度学习模型训练和导出。

板端 EAIDK610 人体识别计数项目使用的是：

```text
Tengine 1.3.2 + MobileNetSSD
```

二者关系如下：

| 项目 | 运行位置 | 模型 | 主要目标 |
|---|---|---|---|
| YOLO26 实验五六 | Ubuntu/PC/GPU | YOLO26m | 训练、验证、预测、ONNX 导出 |
| EAIDK610 综合设计 | 嵌入式开发板 | MobileNetSSD | 摄像头实时人体检测、计数、回传 |

为什么最终板端不用 YOLO26：

```text
YOLO26 现代 ONNX 对旧版 Tengine 1.3.2 不够友好；
板端课程要求重点是 Tengine 部署和实时应用；
MobileNetSSD 是指导书示例链路，稳定性更高。
```

因此，两个项目各自承担不同任务：

```text
YOLO26：展示现代目标检测训练与导出能力；
MobileNetSSD：展示嵌入式端侧部署与摄像头实时系统能力。
```

## 15. 最终总结

`exp5_6_yolo26` 是一个完整的 YOLO26 目标检测实验包。它从环境配置开始，提供了数据集准备、VisDrone 标注转换、训练集划分、数据集 YAML 生成、YOLO26m 训练、验证、预测、ONNX 导出和截图提纲等完整材料。

项目的核心价值在于：

```text
1. 用真实无人机数据替代简单示例图片；
2. 将 VisDrone 多类目标合并为 obstacle，贴合避障任务；
3. 使用 YOLO26m 完成迁移训练；
4. 生成可复现实验结果和截图证据；
5. 完成 PyTorch 权重到 ONNX 模型的导出；
6. 与 EAIDK610 板端 MobileNetSSD 项目形成互补。
```

报告中可以把它定位为：

> 实验五和实验六完成了基于 YOLO26 的目标检测训练与模型导出流程，为理解现代目标检测模型训练、验证和部署格式转换提供了实验基础；综合设计阶段则选择更适配 EAIDK610 和 Tengine 1.3.2 的 MobileNetSSD 完成端侧实时人体检测系统。
