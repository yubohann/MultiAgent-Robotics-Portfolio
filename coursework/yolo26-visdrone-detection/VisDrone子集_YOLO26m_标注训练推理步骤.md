# VisDrone 子集 + YOLO26m 标注、训练、推理步骤

本文档用于当前方案：

```text
真实无人机数据：VisDrone
类别策略：所有可碰撞目标合并为 obstacle
模型：yolo26m.pt
流程：下载/转换标注 -> 划分数据集 -> 训练 -> 验证/推理 -> 导出 ONNX
```

## 0. 进入环境

```bash
cd ~/yubohan_[REDACTED]_exp5_6_yolo26
conda activate yolo26
export PYTHONNOUSERSITE=1
ls -lh yolo26m.pt
```

如果 `yolo26m.pt` 不存在，先从 Windows 复制：

```bash
cp "/mnt/c/Users/Administrator/Desktop/作品集/嵌入式系统课程设计/exp5_6_yolo26/yolo26m.pt" .
```

截图：

```text
终端显示当前在 yolo26 环境，且 yolo26m.pt 文件存在。
```

## 1. 准备 VisDrone 子集并自动转换标注

执行：

```bash
python scripts/05_prepare_visdrone_subset.py --mode obstacle --max-images 500 --clean
```

说明：

```text
该脚本会下载 VisDrone DET 数据，读取官方 annotations，将 pedestrian、car、bus、motor 等所有可碰撞类别统一转换为 obstacle，输出 YOLO txt 标签。
```

输出位置：

```text
data_raw/images
data_raw/labels
configs/classes.txt
```

检查：

```bash
ls data_raw/images | head
ls data_raw/labels | head
cat configs/classes.txt
head -5 data_raw/labels/*.txt
```

截图：

```text
1. 终端显示 Copied images 和 Copied labels
2. data_raw/images 中有无人机视角图片
3. data_raw/labels 中有同名 .txt 标签
4. classes.txt 显示 obstacle
5. 标签文件为 YOLO 五列格式
```

## 2. 划分训练集和验证集

执行：

```bash
python scripts/01_split_yolo_dataset.py
python scripts/02_make_data_yaml.py
```

检查：

```bash
find my_data/detection -maxdepth 3 -type f | head -30
cat configs/my_detect.local.yaml
```

截图：

```text
1. 终端显示 Train pairs 和 Val pairs 数量
2. my_data/detection 下有 images/train、images/val、labels/train、labels/val
3. my_detect.local.yaml 中 names 只有 obstacle
```

## 3. 准备推理测试图片

从原始图片复制 30 张到 `detect_test`：

```bash
rm -f detect_test/*
find data_raw/images -type f | head -30 | xargs -I{} cp "{}" detect_test/
ls -lh detect_test | head
```

截图：

```text
detect_test 中有待推理图片。
```

## 4. 使用 YOLO26m 训练

推荐先跑 15 个 epoch，课程实验足够截图：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 4 --epochs 15
```

如果显存不足：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device 0 --batch 2 --epochs 15
```

如果 WSL 里暂时不能用 GPU：

```bash
python scripts/03_train_yolo26.py --model yolo26m.pt --device cpu --batch 2 --epochs 5
```

训练输出：

```text
runs/yolo26/train/weights/best.pt
runs/yolo26/train/weights/last.pt
runs/yolo26/train/results.png
```

截图：

```text
1. 训练开始，显示 model=yolo26m.pt
2. 训练过程，显示 epoch、loss、P、R、mAP
3. 训练结束，显示 Results saved to runs/yolo26/train
4. weights 目录中有 best.pt
5. 打开 results.png
```

## 5. 验证模型

```bash
yolo detect val model=runs/yolo26/train/weights/best.pt data=configs/my_detect.local.yaml imgsz=640 device=0
```

如果没有 GPU：

```bash
yolo detect val model=runs/yolo26/train/weights/best.pt data=configs/my_detect.local.yaml imgsz=640 device=cpu
```

截图：

```text
终端显示 Box(P、R、mAP50、mAP50-95)。
```

## 6. 推理测试图片

```bash
yolo detect predict model=runs/yolo26/train/weights/best.pt source=detect_test save=True conf=0.5 imgsz=640 device=0
```

如果没有 GPU：

```bash
yolo detect predict model=runs/yolo26/train/weights/best.pt source=detect_test save=True conf=0.5 imgsz=640 device=cpu
```

查看结果目录：

```bash
ls runs/detect/predict*
```

在 Windows 打开 WSL 结果图片：

```text
\\wsl.localhost\Ubuntu-22.04\home\ybh\yubohan_[REDACTED]_exp5_6_yolo26\runs\detect
```

截图：

```text
1. 预测命令输出
2. 预测结果目录
3. 打开结果图，图上有 obstacle 检测框和置信度
```

## 7. 导出 ONNX

```bash
yolo export model=runs/yolo26/train/weights/best.pt format=onnx imgsz=640 dynamic=True
ls -lh runs/yolo26/train/weights/*.onnx
```

截图：

```text
1. 终端显示 ONNX export success
2. weights 目录中有 best.onnx
```

## 8. 报告说明

可以写：

```text
本实验采用真实无人机视角 VisDrone 数据集作为训练数据来源。VisDrone 原始标注包含 pedestrian、people、bicycle、car、van、truck、tricycle、awning-tricycle、bus、motor 等类别。由于本文场景面向无人机避障，因此将上述可碰撞目标统一合并为 obstacle 类，并自动转换为 YOLO txt 标注格式。训练阶段使用 YOLO26m 预训练权重进行迁移学习，完成模型训练、验证、推理和 ONNX 导出。
```

