# 实验五、实验六 YOLO26 操作手册

> English overview: [README.en.md](README.en.md)

本文档按课程指导书中的实验五、实验六流程编写，但把原来的 YOLOv8 全部替换为 YOLO26。你按顺序执行即可：实验五完成环境、标注数据、训练；实验六完成验证、预测、导出 ONNX。

官方参考：

- YOLO26 模型文档：https://docs.ultralytics.com/models/yolo26/
- Ultralytics 快速开始：https://docs.ultralytics.com/quickstart/
- 模型导出文档：https://docs.ultralytics.com/modes/export/

## 0. 固定路径

Windows 资料目录，也就是本目录：

```powershell
exp5_6_yolo26
```

Ubuntu 实验目录，后续命令默认在这里执行：

```bash
~/yubohan_[REDACTED]_exp5_6_yolo26
```

截图保存位置：

```text
Windows:
exp5_6_yolo26\screenshots\实验五
exp5_6_yolo26\screenshots\实验六

Ubuntu:
~/yubohan_[REDACTED]_exp5_6_yolo26/screenshots/实验五
~/yubohan_[REDACTED]_exp5_6_yolo26/screenshots/实验六
```

如果你在 Ubuntu 里做实验，先把 Windows 的 `exp5_6_yolo26` 整个文件夹复制到 Ubuntu 用户目录，并改名为：

```bash
~/yubohan_[REDACTED]_exp5_6_yolo26
```

进入目录：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
mkdir -p screenshots/实验五 screenshots/实验六 data_raw/images data_raw/labels detect_test
```

## 1. 实验五：YOLO26 环境准备

### 1.1 检查显卡和 CUDA

在 Ubuntu 终端执行：

```bash
nvidia-smi
nvcc --version
```

如果 `nvidia-smi` 能看到显卡，后续训练命令使用 `--device 0`。如果看不到显卡，后续训练命令使用 `--device cpu`。

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `01_gpu_cuda_check.png` | `screenshots/实验五` | 终端中 `nvidia-smi` 和 `nvcc --version` 的输出；能看到 GPU 型号或报错信息 |

### 1.2 创建 Conda 环境

执行：

```bash
conda create -n yolo26 python=3.10 -y
conda activate yolo26
python --version
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `02_conda_env.png` | `screenshots/实验五` | 终端左侧出现 `(yolo26)`，并显示 Python 3.10 |

### 1.3 安装 YOLO26 相关依赖

执行：

```bash
pip install -U ultralytics onnx netron -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo checks
python -c "import ultralytics, torch; print('ultralytics=', ultralytics.__version__); print('torch=', torch.__version__)"
```

如果 `labelimg` 安装失败，不影响训练，可先跳过，用已有标注文件继续实验。

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `03_install_ultralytics.png` | `screenshots/实验五` | `pip install` 完成，`yolo checks` 和 `ultralytics/torch` 版本输出 |

### 1.4 YOLO26 预测测试

执行：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
python scripts/00_smoke_test_yolo26.py
```

成功后会生成：

```text
runs/smoke_yolo26/predict/bus.jpg
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `04_yolo26_smoke_cmd.png` | `screenshots/实验五` | 终端中 `python scripts/00_smoke_test_yolo26.py` 执行成功，能看到 `yolo26n.pt` 和保存路径 |
| `05_yolo26_smoke_result.png` | `screenshots/实验五` | 打开 `runs/smoke_yolo26/predict/bus.jpg`，图片上有检测框 |

## 2. 实验五：数据集标注和整理

如果你还没有自己的无人机原始图片，先看：

```text
docs/无人机原始图片来源.md
docs/Sim2Real真实相机数据建议.md
```

课程实验建议直接准备 VisDrone 无人机视角数据子集：

```bash
python scripts/05_prepare_visdrone_subset.py --mode obstacle --max-images 300 --clean
```

### 2.1 准备原始图片

把你的训练图片放入：

```text
data_raw/images
```

把 YOLO 格式标签放入：

```text
data_raw/labels
```

要求：

```text
data_raw/images/0001.jpg
data_raw/labels/0001.txt
```

标签 txt 内容格式：

```text
类别编号 x_center y_center width height
```

示例：

```text
0 0.513281 0.482812 0.326563 0.421875
1 0.384375 0.612500 0.218750 0.183333
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `06_raw_dataset_files.png` | `screenshots/实验五` | 文件管理器或终端 `ls data_raw/images data_raw/labels`，能看到图片和同名 txt 标签 |
| `07_label_txt_format.png` | `screenshots/实验五` | 打开一个 `.txt` 标签文件，显示 YOLO 五列格式 |

### 2.2 修改类别名

打开：

```text
configs/classes.txt
```

按你的数据集类别逐行填写。课程指导书示例是：

```text
person
surfboard
```

如果你的数据集只有 1 类，例如 `helmet`，就改成：

```text
helmet
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `08_classes_txt.png` | `screenshots/实验五` | 编辑器中显示 `configs/classes.txt`，类别数量和标签编号一致 |

### 2.3 划分训练集和验证集

执行：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
python scripts/01_split_yolo_dataset.py
python scripts/02_make_data_yaml.py
```

检查目录：

```bash
find my_data/detection -maxdepth 3 -type f | head -30
cat configs/my_detect.local.yaml
```

成功后的结构：

```text
my_data/detection/
  images/train/
  images/val/
  labels/train/
  labels/val/
configs/my_detect.local.yaml
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `09_split_dataset.png` | `screenshots/实验五` | 终端显示 train/val 数量，`find` 能看到 images 和 labels |
| `10_my_detect_yaml.png` | `screenshots/实验五` | `cat configs/my_detect.local.yaml` 输出，能看到 `path/train/val/names` |

## 3. 实验五：YOLO26 训练

推荐先训练 15 个 epoch，和指导书保持一致。

如果有 GPU：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
python scripts/03_train_yolo26.py --device 0
```

如果没有 GPU：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
python scripts/03_train_yolo26.py --device cpu
```

等价 CLI 命令：

```bash
yolo detect train model=yolo26n.pt data=configs/my_detect.local.yaml epochs=15 imgsz=640 batch=4 workers=2 project=runs/yolo26 name=train
```

训练结果重点文件：

```text
runs/yolo26/train/weights/best.pt
runs/yolo26/train/weights/last.pt
runs/yolo26/train/results.png
runs/yolo26/train/confusion_matrix.png
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `11_train_start.png` | `screenshots/实验五` | 训练刚开始，能看到 `YOLO26`、`epochs=15`、`imgsz=640` |
| `12_train_finish.png` | `screenshots/实验五` | 训练结束，终端显示 `Results saved to runs/yolo26/train` |
| `13_train_results_png.png` | `screenshots/实验五` | 打开 `runs/yolo26/train/results.png`，能看到 loss、precision、recall、mAP 曲线 |
| `14_best_pt_file.png` | `screenshots/实验五` | 文件管理器或 `ls -lh runs/yolo26/train/weights`，能看到 `best.pt` |

## 4. 实验六：验证、预测、导出 ONNX

### 4.1 准备测试图片

把要测试的图片放入：

```text
detect_test
```

检查：

```bash
ls -lh detect_test
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `15_detect_test_files.png` | `screenshots/实验六` | `detect_test` 目录里有测试图片 |

### 4.2 一键执行验证、预测、导出

如果有 GPU：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
python scripts/04_exp6_val_predict_export.py --device 0
```

如果没有 GPU：

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
python scripts/04_exp6_val_predict_export.py --device cpu
```

等价 CLI 命令分三步执行：

```bash
yolo detect val model=runs/yolo26/train/weights/best.pt data=configs/my_detect.local.yaml imgsz=640
```

```bash
yolo detect predict model=runs/yolo26/train/weights/best.pt source=detect_test save=True conf=0.5 imgsz=640
```

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True
```

YOLO26 默认是端到端检测输出，导出的 ONNX 通常更接近：

```text
[x1, y1, x2, y2, confidence, class_id]
```

如果后续要接旧版 YOLOv8 的 C++ 后处理代码，再导出一个兼容传统后处理的 ONNX：

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True end2end=False
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `16_val_metrics.png` | `screenshots/实验六` | 验证输出，能看到 `mAP50`、`mAP50-95` 等指标 |
| `17_predict_cmd.png` | `screenshots/实验六` | 预测输出，能看到保存目录 |
| `18_predict_result.png` | `screenshots/实验六` | 打开预测结果图片，能看到检测框、类别和置信度 |
| `19_export_onnx.png` | `screenshots/实验六` | 导出 ONNX 成功，能看到 `best.onnx` 路径 |

### 4.3 Netron 查看 ONNX

执行：

```bash
netron runs/yolo26/train/weights/best.onnx --host 127.0.0.1 --port 8080
```

浏览器打开：

```text
http://127.0.0.1:8080
```

截图：

| 文件名 | 保存位置 | 截图内容 |
| --- | --- | --- |
| `20_netron_onnx.png` | `screenshots/实验六` | Netron 页面，能看到模型输入 `images` 和输出节点 |

## 5. 报告写法要点

实验五可以写：

```text
本实验将指导书中的 YOLOv8 替换为 YOLO26。模型权重由 yolov8n.pt 改为 yolo26n.pt，训练数据仍采用 YOLO txt 标注格式，数据集目录仍为 images/train、images/val、labels/train、labels/val。训练使用 Ultralytics YOLO API，训练轮数为 15，输入尺寸为 640，最终生成 best.pt。
```

实验六可以写：

```text
本实验使用实验五训练得到的 best.pt 进行验证、预测和 ONNX 导出。验证阶段记录 precision、recall、mAP50、mAP50-95 等指标；预测阶段将 detect_test 中的测试图片输出到 runs/yolo26/predict；导出阶段将 best.pt 转换为 best.onnx，并使用 Netron 查看网络输入输出结构。
```

YOLOv8 到 YOLO26 替换说明可以写：

```text
YOLO26 与 YOLOv8 同属 Ultralytics YOLO 系列，Python 和 CLI 使用方式基本一致。主要替换点为：将 yolov8n.pt 替换为 yolo26n.pt；将实验目录和运行结果命名为 yolo26；导出 ONNX 时注意 YOLO26 默认端到端输出，如需兼容旧版 YOLOv8 后处理可设置 end2end=False。
```

## 6. 常见问题

### 6.1 `yolo26n.pt` 下载失败

重新执行：

```bash
python scripts/00_smoke_test_yolo26.py
```

如果网络仍失败，换网络后再执行。模型会自动下载到 Ultralytics 缓存目录。

### 6.2 显存不足

把 batch 改小：

```bash
python scripts/03_train_yolo26.py --device 0 --batch 2
```

仍然不行就用 CPU：

```bash
python scripts/03_train_yolo26.py --device cpu --batch 2
```

### 6.3 类别数量不匹配

检查：

```bash
cat configs/classes.txt
cat configs/my_detect.local.yaml
head data_raw/labels/*.txt
```

标签第一列必须从 `0` 开始，最大编号不能超过 `classes.txt` 行数减 1。

### 6.4 `detect_test` 没有图片

预测前必须放入测试图片：

```bash
cp data_raw/images/*.jpg detect_test/
ls -lh detect_test
```

如果图片后缀是 `.png` 或 `.jpeg`，按实际后缀复制。
