# Sim2Real 真实相机数据建议

你的目标是 sim2real，所以真实域数据优先级应当高于普通无人机航拍数据。核心原则：

```text
仿真数据：IsaacLab/Isaac Sim 生成 RGB、Depth、Segmentation、Pose
真实数据：真实无人机或真实相机拍摄的 RGB 图像/视频帧
最终适配：尽量使用和实际无人机相同或相近的相机、镜头、分辨率、曝光、运动速度
```

## 1. 最推荐的数据源

### 第一推荐：ODA Dataset

适合你的方向：

```text
无人机避障、真实机载相机、室内障碍物、sim2real 真实域
```

理由：

- 数据由真实 MAV 在真实室内飞行环境采集
- 传感器包含 Full HD RGB camera、event camera、radar、IMU
- 由 OptiTrack 提供无人机真实位姿
- 任务本身就是 obstacle detection and avoidance
- 数据包含 1 个或 2 个障碍物、正常光照和弱光条件

用途：

```text
1. 从 RGB .avi 中抽帧，作为真实相机照片
2. 用于检测/分割模型的真实域微调
3. 用于和 IsaacLab 仿真图像做 sim2real domain gap 对比
4. 用于论文中说明真实避障场景验证
```

链接：

```text
https://github.com/tudelft/ODA_Dataset
https://doi.org/10.4121/14214236.v1
```

### 第二推荐：UZH-FPV Drone Racing Dataset

适合你的方向：

```text
无人机竞速、真实 FPV 相机、IMU、真实轨迹、快速运动 sim2real
```

理由：

- 真实 FPV drone racing quadrotor 数据
- 包含 camera images、IMU、event data、ground truth
- 运动速度高，更接近 drone racer 项目的快速飞行

缺点：

```text
它更偏竞速/VIO，不是专门的避障检测数据集；没有现成 YOLO 障碍物框标注。
```

链接：

```text
https://fpv.ifi.uzh.ch/
https://fpv.ifi.uzh.ch/datasets/
```

### 第三推荐：VisDrone

适合你的方向：

```text
YOLO26 目标检测、真实无人机相机照片、行人/车辆/自行车障碍物检测
```

理由：

- 真实 drone-mounted camera 采集
- 有检测框标注
- 适合课程实验五/六，最快能训练 YOLO26

缺点：

```text
多为航拍/斜视城市视角，不是典型机载前视避障；没有深度和真实飞行位姿。
```

链接：

```text
https://github.com/VisDrone/VisDrone-Dataset
https://docs.ultralytics.com/datasets/detect/visdrone/
```

## 2. 不建议作为真实域主数据

### Mid-Air

Mid-Air 适合做仿真/合成域补充，但不适合作为真实域，因为它不是现实相机拍摄。

如果你的论文强调 sim2real，写法应当是：

```text
Mid-Air 可作为合成数据或仿真对比数据，但真实域采用 ODA、UZH-FPV、VisDrone 或自采数据。
```

## 3. 最合理的组合方案

### 课程实验五/六

用 VisDrone：

```bash
python scripts/05_prepare_visdrone_subset.py --mode obstacle --max-images 300 --clean
python scripts/01_split_yolo_dataset.py
python scripts/02_make_data_yaml.py
python scripts/03_train_yolo26.py --device 0
```

### 论文 sim2real

用 ODA + 自己采集：

```text
仿真域：IsaacLab 森林/门框/障碍物场景生成图像
真实域 1：ODA Dataset 的真实 RGB 视频抽帧
真实域 2：自己无人机/摄像头采集的真实前视图像
检测模型：YOLO26 或轻量分割模型
控制模型：IsaacLab Graph-MASAC / PX4 / 避障策略
```

## 4. 从真实视频抽帧

把真实视频放到：

```text
external/real_videos
```

支持：

```text
.avi .mp4 .mov .mkv
```

执行：

```bash
python scripts/06_extract_real_video_frames.py --input external/real_videos --stride 15 --max-frames 500 --clean
```

输出：

```text
data_raw/images/
```

然后你需要用 LabelImg 给这些真实帧标注：

```bash
labelImg data_raw/images
```

标注输出保存到：

```text
data_raw/labels/
```

再继续：

```bash
python scripts/01_split_yolo_dataset.py
python scripts/02_make_data_yaml.py
python scripts/03_train_yolo26.py --device 0
```

## 5. 论文写法

可以直接写：

```text
为缩小 IsaacLab 仿真环境与真实无人机视觉输入之间的域差距，本文在仿真数据之外引入真实相机数据。真实域数据主要来自 ODA Dataset 和自采无人机前视图像。ODA Dataset 由真实 MAV 在室内障碍物环境中采集，包含 RGB camera、event camera、radar、IMU 以及 OptiTrack 位姿真值，任务与无人机障碍物检测和避障高度相关。本文从真实 RGB 视频中抽取图像帧，并结合人工标注或已有传感器信息用于视觉模型微调，从而提升仿真训练模型迁移到真实相机输入时的鲁棒性。
```

## 6. 数据选择结论

```text
如果只做课程 YOLO26：VisDrone 最省事。
如果做 sim2real 避障论文：ODA Dataset 最贴合。
如果做 drone racing / 高速飞行视觉惯性：UZH-FPV 最贴合。
如果要最终落地：必须补一小批自己真实相机拍摄的数据。
```

