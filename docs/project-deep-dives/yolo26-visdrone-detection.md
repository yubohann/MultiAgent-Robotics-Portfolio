# YOLO26 + VisDrone Detection：项目技术说明

## 300 字项目概括

YOLO26 + VisDrone Detection 是嵌入式系统课程设计中的航拍目标检测实验，完整保留从数据准备到 ONNX 导出的可执行链路。项目先从 VisDrone 子集建立 YOLO 目录结构和标签，再生成 `data.yaml`，运行 YOLO26n 训练，执行验证、预测和模型导出，并提供真实无人机视频的逐帧提取脚本作为可选输入。`00_smoke_test_yolo26.py` 检查 Ultralytics、模型权重和运行环境；`05_prepare_visdrone_subset.py` 负责子集准备；`01_split_yolo_dataset.py` 和 `02_make_data_yaml.py` 处理划分与配置；`03_train_yolo26.py` 组织训练；`04_exp6_val_predict_export.py` 把 val、predict 和 ONNX export 串起来；`06_extract_real_video_frames.py` 使用真实视频抽帧，而不是用 OpenCV 绘制拟合结果。截图、实验报告和结果目录记录课程验收过程。项目的技术价值在于把数据规范、训练参数、推理输出和边缘部署格式连接起来，适合展示计算机视觉入门、工程脚本和 sim-to-real 意识；它是课程实验，不能由单次训练或截图推导出公开数据集 SOTA 或真实无人机部署效果。

## 项目定位

- **数据**：VisDrone 航拍目标检测子集。
- **模型**：Ultralytics YOLO26，仓库中的 `yolo26n.pt` 作为训练起点。
- **输出**：训练权重、验证/预测结果和 ONNX 模型；真实视频抽帧是可选路径。
- **性质**：嵌入式系统课程实验五、六，保留截图和实验报告作为过程证据。

## 技术架构

```text
VisDrone annotations/images
        -> subset preparation
        -> train/val split + data.yaml
        -> YOLO26 training
        -> validation / prediction
        -> ONNX export
        -> optional real-video frame extraction
```

脚本把路径、类别、权重和 Ultralytics API 作为显式参数或配置读取；结果图和截图与代码分开，便于复核。真实视频脚本只负责抽帧，不能替代对视频中目标的真实标注和独立评估。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `scripts/00_smoke_test_yolo26.py` | 依赖、权重和模型加载冒烟 |
| `scripts/05_prepare_visdrone_subset.py` | VisDrone 子集准备 |
| `scripts/01_split_yolo_dataset.py` | train/val 划分 |
| `scripts/02_make_data_yaml.py` | 生成 YOLO 数据配置 |
| `scripts/03_train_yolo26.py` | 训练入口 |
| `scripts/04_exp6_val_predict_export.py` | 验证、预测和 ONNX 导出 |
| `scripts/06_extract_real_video_frames.py` | 从真实航拍视频抽帧 |
| `configs/`、`detection-samples/`、`screenshots/`、`docs/` | 类别/数据配置、示例图、课程证据和 sim-to-real 说明 |

## 开发过程

1. 先用 smoke script 确认 Ultralytics 和权重可加载。
2. 对 VisDrone 标注做子集整理和目录标准化，再固定 train/val 划分。
3. 生成 `data.yaml` 后运行训练，保留日志和 checkpoint。
4. 在同一配置下做验证、预测和 ONNX export，检查导出文件可加载。
5. 补充真实视频抽帧和截图/报告，明确课程证据与真实部署证据的差别。

## 涉及知识点

- 航拍目标检测、YOLO 标签格式和类别映射
- 数据集划分、训练/验证、过拟合和指标解释
- Ultralytics Python API、checkpoint 和推理结果
- ONNX 导出与边缘部署接口
- 视频解码、抽帧、文件命名和证据组织
- sim-to-real 数据来源与标注边界

## 运行入口

```powershell
pip install -r requirements_yolo26.txt
python scripts/00_smoke_test_yolo26.py
python scripts/05_prepare_visdrone_subset.py
python scripts/01_split_yolo_dataset.py
python scripts/02_make_data_yaml.py
python scripts/03_train_yolo26.py
python scripts/04_exp6_val_predict_export.py
python scripts/06_extract_real_video_frames.py
```

具体参数、类别和输出目录以脚本 `--help` 与 README 为准。

## 学术申请中的项目价值

它可以作为计算机视觉与嵌入式部署基础项目，证明你能把公开数据、训练脚本、部署格式和真实视频输入连成工程流程。若与 AeroCityBench/Rivermark 等多 UAV 项目共同展示，还能说明你理解检测模块在机器人系统中的位置，而不是把单张图片效果冒充完整自主系统。

## 测试与验证

当前仓库的直接验证入口是 smoke script、训练/验证/导出脚本和课程截图。需要重新生成指标时，应固定数据子集、seed、权重版本和硬件，并独立记录 mAP、召回、推理延迟和 ONNX runtime 结果。

## 证据边界

截图和课程报告证明实验流程或页面状态，不能自动证明真实无人机上的实时性能。视频抽帧不等于视频级检测评估；YOLO26n 起始权重、VisDrone 数据许可和 ONNX runtime 版本需要按各自来源声明。

## 源码审计记录

| 阶段 | 代表文件 | 逐文件复核内容 |
| --- | --- | --- |
| 环境 | `scripts/00_smoke_test_yolo26.py` | Ultralytics import、权重路径、模型加载和设备可用性。 |
| 数据 | `05_prepare_visdrone_subset.py` | 子集来源、类别映射、图片/标签复制和缺失文件处理。 |
| 划分 | `01_split_yolo_dataset.py` | train/val 比例、随机种子、重复样本和 YOLO 目录结构。 |
| 配置 | `02_make_data_yaml.py`、`configs/` | `path`、train/val、class names 和相对路径。 |
| 训练 | `03_train_yolo26.py` | model、epochs、image size、batch、device、output directory 和 resume 语义。 |
| 评测 | `04_exp6_val_predict_export.py` | val metrics、prediction images、confidence/NMS 参数和 export format。 |
| 视频 | `06_extract_real_video_frames.py` | FFmpeg/OpenCV 解码、帧间隔、命名、尺寸和原始视频来源。 |
| 证据 | `screenshots/`、`docs/`、报告 | 每一步截图/报告是否能回到脚本、输入、权重和日志。 |

## 从数据到 ONNX 的调用链

1. 子集脚本读取 VisDrone 标注并生成图像/标签对，随后 split 脚本固定 train/val 清单。
2. data yaml 只描述路径和类别；它不保存真实数据，也不替代许可证和版本记录。
3. train 脚本构造 Ultralytics model，加载 YOLO26n 起始权重，使用显式配置创建 runs 目录。
4. val/predict 脚本在固定 checkpoint 上分别计算验证指标和生成可视化，避免把预测图当作 ground truth。
5. export 将模型转成 ONNX；需要使用 ONNX Runtime 或目标推理引擎做加载/输出 shape smoke。
6. real-video frame script 只生成图片帧；若要评估视频检测，还需独立标注、帧级/轨迹级指标和 latency protocol。

## 参数与结果复核

复现实验时至少锁定 dataset subset hash、class list、split seed、YOLO/Ultralytics 版本、权重 hash、训练设备、batch、image size、epochs 和 export opset。报告中的 mAP、precision、recall、loss 和推理时间必须注明是哪个 split、哪个 checkpoint、哪个硬件和是否包含预热；不同配置的截图不能直接拼成一个“最终结果”。

## 课程知识到代码的映射

- 标签解析对应目标检测的坐标格式、类别编码和数据质量。
- train/val split 对应泛化评估和泄漏防护。
- Ultralytics 训练接口对应模型配置、优化器、增强和 checkpoint 生命周期。
- val/predict 对应离线指标与 qualitative review 的互补关系。
- ONNX export 对应训练框架与边缘推理运行时之间的接口。
- 真实视频抽帧对应 sim-to-real 输入治理，但不自动增加标注或模型性能。

## 交付检查表

- [ ] 子集、标签、split 和 data.yaml 可从脚本重建。
- [ ] 训练/验证/预测/导出使用同一权重和配置记录。
- [ ] ONNX 文件有独立 load/inference smoke。
- [ ] 真实视频来源和隐私/许可得到记录，未把视频帧当作训练真值。
- [ ] 课程截图、报告和实测指标的证据等级清楚。

## 原理与理论基础

### 1. 检测问题

目标检测同时估计类别和边界框。对每个候选框，模型输出 objectness、类别概率和坐标，训练损失通常由分类、定位和置信度项组成。NMS 再按 IoU 阈值抑制同一目标的重复框。VisDrone 的小目标、遮挡、密集排列和俯视尺度变化会让训练增强、输入分辨率和评估阈值影响很大，不能用单张可视化图替代 mAP/recall。

### 2. 数据与泛化

train/val split 的作用是估计从已见图像到未见图像的泛化。航拍数据若按连续视频随机切帧，邻近帧可能同时落入 train 与 val，导致空间/时间泄漏；因此应记录 split 清单、seed、来源和是否按视频或场景分组。类别映射、空标签、越界框和图像尺寸错误会在训练前就改变目标分布。

### 3. 置信度与部署

检测器输出的是概率近似，不是事实。阈值、NMS IoU、输入缩放和后处理共同决定 precision/recall；边缘部署还要考虑 ONNX operator、动态 shape、FP16/INT8 和端到端 latency。导出成功只说明图能被序列化，仍需在目标 runtime 上比较输出 shape、数值误差、预热和吞吐。

### 4. 理论假设和失效模式

训练假设标签正确、类别集合稳定、训练/验证分布可比；真实无人机视频可能有运动模糊、相机姿态、天气、分辨率和域差异。抽帧不会自动解决域差异，也不会生成标注；因此 sim-to-real 只能由新数据、独立评估和延迟协议证明。
