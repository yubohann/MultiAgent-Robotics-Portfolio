# 从零学习 LiDAR--IMU 定位：FAST-LIO2、Point-LIO 与机器人自主系统实战指南

> 面向对象：会一点 Python 或 C++，但尚未系统学习机器人定位、点云和 ROS 2 的学习者。
>
> 目标：不是背出 FAST-LIO2 或 Point-LIO 的名字，而是能够解释它们为什么需要 IMU、逐点时间、局部地图和迭代滤波；能够读懂本项目的关键数据流；能够设计一套不自欺欺人的验证实验；能够把定位质量安全地传递到导航和投篮动作。
>
> 本文是新建的学习手册。已有的[算法综述](robocon-mid360-lio-algorithms-zh.md)保留不动，适合在完成本教程后查阅算法横向比较与项目边界。

---

## 目录

1. [学习路线与最终能力](#1-学习路线与最终能力)
2. [先建立正确的问题观](#2-先建立正确的问题观)
3. [第一层基础：坐标系、刚体变换与角度](#3-第一层基础坐标系刚体变换与角度)
4. [第二层基础：概率、噪声与卡尔曼滤波](#4-第二层基础概率噪声与卡尔曼滤波)
5. [第三层基础：IMU 与惯性传播](#5-第三层基础imu-与惯性传播)
6. [第四层基础：LiDAR 点云与扫描时间](#6-第四层基础lidar-点云与扫描时间)
7. [第五层基础：点云匹配、ICP 与局部几何](#7-第五层基础点云匹配icp-与局部几何)
8. [从 LOAM 到紧耦合 LIO](#8-从-loam-到紧耦合-lio)
9. [FAST-LIO2：从数据到状态更新](#9-fast-lio2从数据到状态更新)
10. [Point-LIO：高带宽点级更新](#10-point-lio高带宽点级更新)
11. [全局定位、建图与回环](#11-全局定位建图与回环)
12. [其他相关算法图谱](#12-其他相关算法图谱)
13. [ROS 2 与本项目的数据合同](#13-ros-2-与本项目的数据合同)
14. [从零到可复现的实验路线](#14-从零到可复现的实验路线)
15. [调参、诊断与常见失败](#15-调参诊断与常见失败)
16. [源码阅读路线](#16-源码阅读路线)
17. [12 周学习计划](#17-12-周学习计划)
18. [面试与项目表达](#18-面试与项目表达)
19. [术语表与延伸阅读](#19-术语表与延伸阅读)

---

## 1. 学习路线与最终能力

### 1.1 不要从源码或论文第一页直接开始

初学者常见的两种低效路径是：

```text
路径 A：直接打开 5000 行 laserMapping.cpp，希望一次看懂
路径 B：直接读论文推导，希望不写代码就理解系统
```

二者都会很快卡住。更有效的顺序是：

```text
坐标系与刚体变换
    -> IMU 运动学和误差来源
    -> 点云、扫描时间与去畸变
    -> ICP 和点到平面残差
    -> EKF / ESKF / IEKF
    -> LOAM 与 LIO 的系统结构
    -> FAST-LIO2 / Point-LIO 源码与 ROS 2 集成
    -> 固定地图、导航、感知和安全状态机
```

每一层都要完成一个可检查的小练习，再进入下一层。

### 1.2 完成后应具备的能力

完成本指南后，应该能做到：

- 画出 `map -> odom -> base_link -> imu_link -> lidar` 的坐标树，并解释每条边的所有权；
- 使用齐次变换把一个 LiDAR 点从雷达系变到地图系；
- 说明为什么四元数 `z` 分量不是 yaw；
- 写出 IMU 的姿态、速度、位置传播式，说明漂移为什么必然发生；
- 解释逐点 `offset_time` 为什么是运动去畸变的必要输入；
- 从点到平面残差推导“LiDAR 如何纠正 IMU 预测”；
- 区分 EKF、ESKF、IEKF、IKFoM、因子图和位姿图；
- 说明 LOAM、FAST-LIO2、Point-LIO、LIO-SAM、LVI-SAM、ICP、GICP、NDT 的角色差异；
- 读懂当前工程中 LiDAR、IMU、局部 odometry、固定地图和状态机之间的调用关系；
- 用同一段 rosbag 设计 FAST-LIO2 与 Point-LIO 的公平对比；
- 不把仿真、单次成功或脚本存在误写成实机精度结论。

### 1.3 学习前的最小准备

建议具备：

| 领域 | 最低要求 | 学到什么程度才可继续 |
| --- | --- | --- |
| Python | NumPy、列表/字典、画图 | 能处理一个 `N x 3` 点云数组 |
| C++ | 类、指针/引用、STL、CMake | 能阅读 ROS 2 回调和 Eigen 表达式 |
| 线性代数 | 向量、矩阵乘法、特征值 | 理解旋转矩阵、协方差、平面拟合 |
| 微积分 | 导数、泰勒展开 | 理解线性化和雅可比的目的 |
| Linux | shell、路径、进程、日志 | 能运行 build、记录日志、看 topic |
| ROS 2 | node/topic/service/action、launch | 能使用 `ros2 topic echo/hz/info` |

不要求一开始精通李群、概率图模型或 Ceres/GTSAM。先理解每个工具解决什么问题，再补严格数学推导。

---

## 2. 先建立正确的问题观

### 2.1 定位、建图、里程计和导航不是同一件事

| 名称 | 回答的问题 | 典型输出 |
| --- | --- | --- |
| 里程计（odometry） | 机器人相对起点移动了多少？ | 连续 `odom -> base_link` |
| 定位（localization） | 机器人在已知地图中的哪里？ | `map -> odom` 或 `map -> base_link` |
| 建图（mapping） | 环境地图是什么样？ | 点云、栅格、占据图、surfel 地图 |
| SLAM | 同时估计轨迹和地图 | 轨迹 + 增量地图/全局图 |
| 回环（loop closure） | 是否回到已到过的位置？ | 全局轨迹校正约束 |
| 导航（navigation） | 如何安全地到达目标？ | 路径、速度命令、恢复动作 |
| 控制（control） | 电机怎样实际执行？ | PWM/CAN/UART 指令与反馈 |

FAST-LIO2 和 Point-LIO 的核心输出是**局部连续里程计**。它们可以边运行边积累局部地图，但不能自然保证长期全局无漂移。

### 2.2 本项目中的分层

```text
FAST-LIO2 / Point-LIO
    负责：连续相对运动、去畸变、局部点云地图

固定地图 scan-to-map
    负责：把局部轨迹锚定到比赛场地 map

TF / pose contract
    负责：统一 frame、时间戳、质量和状态语义

Nav2 / 任务状态机
    负责：规划、动作互锁、恢复和取消

STM32
    负责：PID、限位、过流、超时、急停等实时安全
```

这层边界是工程安全的一部分。不能让某个上游 LIO 节点直接越过 TF、状态机和 MCU 安全边界控制投篮机构。

### 2.3 贯穿全文的三个问题

学习任何算法时，都回答下面三个问题：

1. **它估计什么？** 例如姿态、位置、地图、匹配变换、类别或速度。
2. **它依赖什么假设？** 例如时间同步、静态环境、足够平面、可靠初值。
3. **假设不成立时如何暴露和降级？** 例如残差升高、协方差变大、地图失锁、拒绝动作。

只记“某算法精度高”没有工程价值。

---

## 3. 第一层基础：坐标系、刚体变换与角度

### 3.1 坐标系是所有机器人问题的共同语言

一个点坐标没有坐标系就没有意义。

例如：

```text
[1.0, 0.0, 0.2]
```

它可能表示：

- LiDAR 前方 1 m；
- 机器人底盘前方 1 m；
- 地图全局 x=1 m；
- 相机坐标中的一个点。

本项目建议使用的主要坐标树：

```text
map
 └── odom
      └── base_link
           └── imu_link
                └── lidar_mid360
```

含义：

| 坐标系 | 作用 |
| --- | --- |
| `map` | 全局/固定场地坐标系，允许重定位修正 |
| `odom` | 局部连续坐标系，短期平滑但可能累积漂移 |
| `base_link` | 机器人机体中心或控制参考点 |
| `imu_link` | IMU 安装坐标系 |
| `lidar_mid360` | MID-360 传感器坐标系 |

### 3.2 右手系与 REP-103

ROS 中通常遵循右手系约定：

```text
x：前
y：左
z：上
```

这不是所有硬件原生坐标系都遵循的方向。读取驱动时必须确认：

- 雷达输出轴方向；
- IMU 轴方向；
- 底盘前方；
- 重力方向；
- 相机 optical frame 的轴约定。

坐标轴错 90 度或符号反了，系统仍可能“有输出”，但定位和控制会以错误物理含义运行。

### 3.3 旋转矩阵

三维旋转矩阵满足：

```text
RᵀR = I
det(R) = 1
```

如果点 `p_A` 在坐标系 A 下表示，`R_BA` 表示“把 A 中向量旋转到 B”，则：

```text
p_B = R_BA p_A
```

二维 yaw 旋转容易画图理解：

```text
R(theta) = [cos(theta)  -sin(theta)]
           [sin(theta)   cos(theta)]
```

先把二维完全理解，再扩展到三维。

### 3.4 平移和齐次变换

只有旋转不够，机器人还会移动。刚体变换写成：

```text
p_B = R_BA p_A + t_BA
```

齐次矩阵形式：

```text
T_BA = [R_BA  t_BA]
       [0     1   ]

[p_B]   [T_BA] [p_A]
[ 1 ] = [    ] [ 1 ]
```

两个变换串联：

```text
T_CA = T_CB T_BA
```

这是理解 `map -> odom -> base_link` 的数学基础：

```text
T_map_base = T_map_odom T_odom_base
```

### 3.5 变换的逆

给定：

```text
T_BA = [R  t]
       [0  1]
```

其逆为：

```text
T_AB = [Rᵀ  -Rᵀt]
       [0    1  ]
```

这是常见错误来源：平移不能简单取负，必须先旋转到正确坐标系。

### 3.6 欧拉角、四元数和 yaw

欧拉角是 roll、pitch、yaw 的组合，直观但会遇到万向节锁。四元数适合累积旋转，但不直观。

从四元数得到 yaw 时，应使用完整公式或 TF2 工具。不能写：

```text
yaw = quaternion.z
```

四元数 `z` 只是四个分量之一，不等于平面航向。

### 3.7 最小练习：手算和代码验证坐标变换

任务：已知机器人在地图系中的位姿为 `(x=2, y=1, yaw=90 度)`，LiDAR 在 `base_link` 前方 0.2 m，雷达点为 `(1, 0, 0)`。请计算该点在地图系中的二维坐标。

思路：

```text
1. 点在 LiDAR 系：p_L
2. LiDAR 到车体：p_B = T_base_lidar p_L
3. 车体到地图：p_M = T_map_base p_B
```

可用 NumPy 验证：

```python
import numpy as np

def T2(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1],
    ])

T_map_base = T2(2.0, 1.0, np.pi / 2)
T_base_lidar = T2(0.2, 0.0, 0.0)
p_lidar = np.array([1.0, 0.0, 1.0])

p_map = T_map_base @ T_base_lidar @ p_lidar
print(p_map)
```

验收标准：能够解释矩阵乘法从右向左表示“先在 LiDAR 系，再到车体系，最后到地图系”。

---

## 4. 第二层基础：概率、噪声与卡尔曼滤波

### 4.1 传感器值不是事实，而是带误差的观测

更真实的模型是：

```text
z = h(x) + noise
```

其中：

- `x` 是真实状态；
- `z` 是传感器测量；
- `h(x)` 是观测模型；
- `noise` 是噪声和未建模误差。

例如 IMU 角速度：

```text
omega_m = omega_true + b_g + n_g
```

包含真实角速度、慢变化零偏和随机噪声。

### 4.2 均值、方差和协方差

一维高斯分布常写成：

```text
x ~ N(mu, sigma²)
```

三维或高维状态使用协方差矩阵 `P`：

```text
P_ij = E[(x_i - mu_i)(x_j - mu_j)]
```

协方差不仅表示每个变量的不确定度，还表示变量之间是否相关。例如姿态误差可能影响位置误差。

### 4.3 为什么不能“相信一个最稳定的传感器”

IMU 很快，但积分必然漂移；LiDAR 约束位置，但可能被遮挡、退化或延迟。融合的本质是：

```text
预测相信动力学
更新相信观测
两者权重由不确定度决定
```

### 4.4 线性卡尔曼滤波的直觉

线性系统：

```text
x_k = F x_(k-1) + w
z_k = H x_k + v
```

预测：

```text
x_hat_minus = F x_hat
P_minus = F P Fᵀ + Q
```

更新：

```text
K = P_minus Hᵀ (H P_minus Hᵀ + R)^(-1)
x_hat = x_hat_minus + K (z - H x_hat_minus)
P = (I - K H) P_minus
```

直觉：

- `Q` 大：不太相信运动模型；
- `R` 大：不太相信传感器观测；
- `K`：在预测和观测之间折中。

### 4.5 EKF、ESKF 和 IEKF

| 方法 | 核心做法 | 在 LIO 中的意义 |
| --- | --- | --- |
| EKF | 对非线性函数在当前点一阶线性化 | 可用，但直接处理旋转不够自然 |
| ESKF | 维护名义状态和小误差状态 | 适合姿态、IMU 偏置等流形状态 |
| IEKF | 一次观测更新内反复线性化 | 适合点云几何残差较强非线性的情况 |
| IKFoM | 为流形状态封装迭代滤波工具 | Point-LIO 常见的实现组织路线 |

### 4.6 误差状态为什么重要

名义姿态 `R` 是完整旋转，误差姿态只需很小的三维向量 `delta_theta`：

```text
R_true ≈ R_nominal Exp(delta_theta)
```

这样既尊重 SO(3) 的几何，又能让滤波器在局部欧氏空间里计算协方差。

### 4.7 最小练习：一维常速度卡尔曼滤波

先不要一上来写 24 维 LIO。实现状态：

```text
x = [position, velocity]
```

输入：

- 带噪 GPS 位置；
- 常速度模型。

验证：

1. 增大 `R`，观察滤波结果更平滑但更依赖预测；
2. 增大 `Q`，观察滤波结果更紧跟测量；
3. 删除一段 GPS，观察预测逐渐失去可信度。

如果不能解释这个实验，不要进入 LIO 参数调节。

---

## 5. 第三层基础：IMU 与惯性传播

### 5.1 IMU 的两个核心测量

| 传感器 | 测量 | 用途 |
| --- | --- | --- |
| 陀螺仪 | 角速度 `omega` | 积分得到姿态变化 |
| 加速度计 | 比力 | 经姿态旋转和加上重力后得到线加速度 |

加速度计静止时不输出零，因为它测到重力相关的比力。这是初学者最容易混淆的地方。

### 5.2 从 IMU 到位姿的链条

```text
gyro 积分 -> 姿态
姿态 + accelerometer -> 世界系线加速度
线加速度积分 -> 速度
速度积分 -> 位置
```

每积分一次都会累积误差，因此：

```text
姿态有小误差
    -> 重力投影有误差
    -> 速度误差快速增大
    -> 位置误差更快增大
```

这就是 LiDAR 几何约束不可缺少的原因。

### 5.3 零偏与随机游走

IMU 误差可简化为：

```text
b_(k+1) = b_k + w_b
```

偏置不是常数，它会缓慢漂移。滤波器要把 `b_g`、`b_a` 作为状态估计，而不是永远相信出厂标定。

### 5.4 IMU 初始化的实际要求

启动前最好满足：

- 机器人尽量静止；
- 传感器已经稳定供电；
- 时间戳开始单调；
- 加速度计和陀螺仪没有饱和；
- frame 已经正确；
- 不在机构剧烈震动时初始化。

初始化失败的典型症状：

- 静止时姿态缓慢旋转；
- 初始地图明显倾斜；
- 一开始走动就大幅漂移；
- 重力方向朝错误轴。

### 5.5 IMU 外参与时间偏移

LiDAR 点是 LiDAR 系下的观测，IMU 运动在 IMU 系下传播。两者关系：

```text
p_imu = R_IL p_lidar + t_IL
```

外参误差常见表现：

| 外参问题 | 典型现象 |
| --- | --- |
| 旋转外参错 | 转弯时墙面倾斜、重影明显 |
| 平移外参错 | 快速旋转时点云出现相对错位 |
| 时间偏移错 | 急加速/急转时地图变厚、位姿抖动 |
| frame 定义错 | 轨迹整体旋转、重力方向异常 |

### 5.6 练习：先看 IMU，再跑 LIO

用 rosbag 或静态传感器数据完成：

```bash
ros2 topic echo --once /livox/imu
ros2 topic hz /livox/imu
ros2 topic info -v /livox/imu
```

检查：

- 时间戳不是零；
- 频率稳定；
- 静止时 gyro 接近零但不是绝对零；
- 加速度模长接近重力常数附近；
- 没有 NaN/Inf；
- 轴方向与文档和安装方式一致。

---

## 6. 第四层基础：LiDAR 点云与扫描时间

### 6.1 什么是点云

点云是大量三维采样点：

```text
P = {p_1, p_2, ..., p_N}
p_i = (x_i, y_i, z_i, intensity_i, time_i)
```

LiDAR 不知道“这是墙”还是“这是篮筐”，它只提供距离反射形成的几何采样。

### 6.2 机械式、固态和非重复扫描

| 雷达类别 | 典型特征 | 算法影响 |
| --- | --- | --- |
| 机械式多线雷达 | 固定 ring、扫描模式较规律 | LOAM 特征提取较容易 |
| 固态/半固态雷达 | 扫描图样可能非重复 | 逐点时间与直接点匹配更重要 |
| 2D LiDAR | 只有平面扫描 | 可用 AMCL/2D SLAM，但无法完整表达 3D 几何 |

Livox MID-360 的输入要特别关注逐点时间，因为“同一帧”内的点并非同时采集。

### 6.3 什么是运动畸变

假设雷达扫描一面静止的墙，机器人在扫描期间向前移动或转动。扫描开始时的墙点和扫描结束时的墙点在传感器系中不再满足同一个几何关系。

若把整帧当成同一时刻：

```text
真实平面 -> 点云中变弯/变厚 -> 局部平面拟合变差 -> 位姿更新错误
```

这就是 deskew（去畸变）的意义。

### 6.4 点时间字段

当前工程的关键约束是：

```text
CustomMsg.timebase + 每点 offset_time 必须保留
```

绝不能在 LIO 前把消息随意转换成只保留 header stamp 的 `PointCloud2`，除非同时明确保留了每点时间字段并验证语义没有改变。

### 6.5 点云预处理

常见步骤：

1. 移除 NaN/Inf；
2. 移除盲区近距离点；
3. 移除量程外、明显无效点；
4. 根据 line/tag/return 筛掉低质量点；
5. 保留或转换点时间；
6. 体素化或抽样；
7. 可选动态物体过滤。

预处理目标不是让点云“看起来漂亮”，而是让后续的几何观测可信、计算量有界。

### 6.6 体素降采样

体素大小为 `d` 时，把空间分成边长为 `d` 的立方体，每个体素保留一个代表点。

| 体素过小 | 体素过大 |
| --- | --- |
| 点数多、细节多、CPU 高 | 点数少、速度快、几何约束变少 |
| 地图内存增长快 | 小物体和边缘结构消失 |
| 近邻查询负载高 | 点到平面拟合可能退化 |

不要追求“最小体素”，而要把处理时间、有效约束点数和轨迹质量一起评估。

### 6.7 练习：检查 LIVox 消息语义

在 ROS 2 环境中：

```bash
ros2 topic type /livox/lidar
ros2 interface show livox_ros_driver2/msg/CustomMsg
ros2 topic echo --once /livox/lidar
```

目标：确认能够在消息中定位 `timebase`、`point_num`、每点坐标、`line`、`tag` 和 `offset_time`。

验收：能够解释 `offset_time` 的单位、起点和是否单调；不能只确认“消息能打印出来”。

---

## 7. 第五层基础：点云匹配、ICP 与局部几何

### 7.1 配准问题的定义

给定当前扫描点云 `P` 和地图点云 `Q`，估计刚体变换 `(R, t)` 使它们对齐。

```text
P -- R,t --> Q
```

这叫 registration（配准）。

### 7.2 点到点 ICP

最常见的 ICP 目标：

```text
min Σ ||R p_i + t - q_i||²
```

迭代过程：

```text
初始位姿
    -> 为每个当前点找最近地图点
    -> 解最优 R,t
    -> 更新当前位姿
    -> 判断是否收敛
```

局限：

- 需要相对较好的初始位姿；
- 最近点未必是真实对应；
- 对重复结构和动态点敏感；
- 在平面上沿平面滑动时约束较弱。

### 7.3 点到平面 ICP

对每个地图点邻域拟合平面法向 `n_i`，最小化：

```text
min Σ [n_iᵀ(Rp_i + t - q_i)]²
```

为什么通常更快：点在平面法向方向上的误差更重要，沿平面切向的轻微移动不应被同等惩罚。

FAST-LIO2 和 Point-LIO 的 LiDAR 几何更新，核心也使用这种局部点到平面约束。

### 7.4 局部平面如何拟合

给定邻域点 `q_1 ... q_k`：

1. 求均值 `q_bar`；
2. 构造邻域协方差；
3. 求最小特征值对应的特征向量；
4. 该向量是平面法向候选；
5. 用点到平面距离检验邻域是否足够平。

最小特征值很小，表示点在该方向分布很薄，通常可近似为平面。

### 7.5 最近邻与 KD-Tree

朴素最近邻查询需要比较所有地图点，复杂度约为 `O(N)`。KD-Tree 通过空间划分加速查询。

LIO 中每帧要对很多点查询近邻，且地图持续变化，因此 FAST-LIO2/Point-LIO 使用支持增量插入和删除的 ikd-Tree。

### 7.6 退化不是 bug，而是几何事实

在长走廊里：

```text
前后方向有结构
横向或沿墙方向可能缺少约束
```

在单一大平面前：

```text
法向方向约束强
沿平面滑动和绕法线转动约束弱
```

因此高质量系统必须监控：

- 有效对应数量；
- 邻域平面质量；
- 残差分布；
- 雅可比/信息矩阵条件数；
- 协方差或 degeneracy 分数；
- 动态点比例。

### 7.7 GICP 与 NDT

| 方法 | 核心想法 | 适合什么 |
| --- | --- | --- |
| ICP | 最近点 + 最小距离 | 简单基线、初值较好 |
| Point-to-plane ICP | 最近邻平面约束 | 墙面/地面等结构化环境 |
| GICP | 用局部协方差表达几何 | 点密度不一致、高质量 scan-to-map |
| NDT | 体素内建高斯分布 | 大范围粗配准、无需显式点对应 |

它们都是配准工具，不会自动解决时间同步、IMU 漂移、地图版本或控制安全。

### 7.8 最小练习：二维 ICP

先在二维实现 ICP：

1. 生成一组 L 形点；
2. 对它施加已知旋转和平移；
3. 给第二组点添加高斯噪声；
4. 用最近邻和 SVD 求刚体变换；
5. 改坏初值，观察局部最优；
6. 加入离群点，观察错误匹配。

验收：能说清 ICP 成功依赖“初值足够好 + 对应关系大致正确 + 场景几何有区分度”。

---

## 8. 从 LOAM 到紧耦合 LIO

### 8.1 LOAM：理解 LiDAR SLAM 的经典入口

LOAM 将点云分成边缘特征和面特征：

```text
原始点云
    -> 曲率计算
    -> edge features / planar features
    -> lidar odometry（高频）
    -> lidar mapping（低频）
```

优点：

- 思路清晰；
- 容易理解点到线、点到面残差；
- 是许多后续 LIO 系统的背景。

局限：

- 特征数量和质量受扫描线、曲率阈值影响；
- 对非重复扫描模式适配成本高；
- 只用 LiDAR 时快速运动会导致畸变和匹配困难。

### 8.2 LiDAR-only 与 LIO 的关键差异

```text
LiDAR-only：上一帧/地图提供初值，快速运动时可能匹配失败

LIO：IMU 提供连续姿态和速度预测，LiDAR 用于校正
```

LIO 并不是“把 IMU topic 加进去”，而是让 IMU 状态传播、点云去畸变和 LiDAR 几何残差在同一估计链中共同工作。

### 8.3 松耦合与紧耦合

松耦合：

```text
LiDAR odometry -> pose A
IMU odometry   -> pose B
后端融合 A/B
```

紧耦合：

```text
IMU 直接传播状态
LiDAR 原始几何残差直接更新同一状态
```

紧耦合通常更能利用高频 IMU 和点级时间，但实现和调试难度更高。

### 8.4 关键帧、后端与实时前端

一个完整 SLAM 系统常分为：

| 层 | 任务 |
| --- | --- |
| 前端 | 高频里程计、局部匹配、短时连续性 |
| 局部地图 | 近邻查询、局部约束、降采样 |
| 后端 | 关键帧、回环、因子图、全局一致性 |
| 全局定位 | 固定地图、GPS、UWB、人工初值等锚定 |

FAST-LIO2 和 Point-LIO 主要处在前端和局部地图层。

---

## 9. FAST-LIO2：从数据到状态更新

### 9.1 用一句话概括

FAST-LIO2 用 IMU 高频传播状态，用逐点时间完成点云去畸变，用直接点到局部平面的几何残差进行迭代误差状态卡尔曼更新，并用 ikd-Tree 维护可实时查询的增量局部地图。

### 9.2 完整处理流程

```text
Livox CustomMsg + IMU
    -> 输入守门、缓存和同步
    -> IMU 初始化
    -> IMU 传播到扫描时间段
    -> 按每点时间去畸变
    -> 当前扫描降采样
    -> ikd-Tree 近邻搜索和局部平面拟合
    -> 点到平面残差与雅可比
    -> IEKF 多次迭代更新
    -> 发布局部 odometry 与已配准点云
    -> 把有效点增量写入局部地图
```

### 9.3 输入守门

进入估计器前应检查：

```text
LiDAR topic 类型正确
点数在合理范围内
点坐标和时间均有限
offset_time 语义明确
IMU 时间戳单调
IMU 覆盖当前 LiDAR 扫描区间
LiDAR/IMU 单位正确
外参和时间偏移版本明确
```

不要等到“地图歪了”才发现输入格式不对。

### 9.4 IMU 初始化和传播

FAST-LIO2 会利用 IMU 估计重力、零偏及初始协方差，随后在 LiDAR 更新之间传播状态。

概念状态：

```text
x = (R, p, v, bg, ba, g, R_LI, t_LI)
```

外参可固定，也可作为状态在线微调。在线标定不等于可以忽略初值；运动激励不足时，外参、偏置和位姿误差可能互相混淆。

### 9.5 去畸变

对每个点 `p_i`：

```text
p_i_world = R(t_i) [R_LI p_i_lidar + t_LI] + p(t_i)
```

这里需要：

- 点时间 `t_i`；
- 连续 IMU 轨迹；
- LiDAR--IMU 外参；
- 对扫描参考时刻的清晰定义。

### 9.6 直接点云匹配

FAST-LIO2 可以不使用传统 LOAM 的显式特征提取。它仍需要轻量过滤和降采样，但核心是让尽可能多的有效点直接参与局部平面约束。

这对 Livox 类雷达很重要：其扫描图样未必天然适合基于固定 scan ring 的传统特征提取。

### 9.7 ikd-Tree 局部地图

地图操作包括：

```text
Build：初始构建
Nearest_Search：寻找邻域
Add_Points：插入新的有效点
Delete_Point_Boxes：移除局部窗口外的点
```

局部地图不是“越大越好”。它要在：

- 几何覆盖；
- 查询速度；
- 内存使用；
- 动态点污染；
- 机器人速度和视场；

之间折中。

### 9.8 IEKF 更新的直觉

对于每个有效点，构造：

```text
r_i = n_iᵀ (R(R_LI p_i + t_LI) + p - q_i)
```

在当前状态附近线性化，得到对姿态、位置、外参等误差状态的敏感度 `H_i`。所有有效残差一起决定本帧的状态修正。

IEKF 会重复：

```text
当前状态
    -> 重新找对应/构造残差
    -> 线性化
    -> 计算修正
    -> 应用修正
```

直到收敛或达到迭代上限。

### 9.9 当前工程的阅读入口

当前 `robocon-mid360-autonomy-stack` 中：

| 路径 | 阅读目标 |
| --- | --- |
| `src/vendor_fast_lio/src/preprocess.cpp` | 理解 Livox 点是怎样进入内部点云的 |
| `src/vendor_fast_lio/src/IMU_Processing.hpp` | 找 `IMU_init`、`UndistortPcl`、`Process` |
| `src/vendor_fast_lio/src/laserMapping.cpp` | 找同步、`Nearest_Search`、滤波更新、地图插入 |
| `src/vendor_fast_lio/include/ikd-Tree/` | 理解增量地图数据结构 |
| `src/vendor_fast_lio/config/mid360.yaml` | 把参数和代码行为对应起来 |
| `src/mid360_localization_contract/` | 看 LIO 输出怎样变成唯一对外 pose/TF 合同 |

建议在阅读时创建一张表：

| 问题 | 答案 | 对应文件/函数 |
| --- | --- | --- |
| 点时间来自哪里？ | `CustomMsg.offset_time` | `preprocess.cpp` |
| 去畸变在哪里做？ | IMU 轨迹与点时间结合 | `IMU_Processing.hpp` |
| 近邻怎样查询？ | ikd-Tree | `laserMapping.cpp` |
| TF 谁发布？ | contract bridge，而非上游直接对外 | `mid360_localization_contract` |

### 9.10 FAST-LIO2 参数实验，不是“万能参数表”

不要直接复制参数。每次只改一个变量，并记录输入数据和结果。

| 参数类别 | 先观察什么 | 再调整什么 |
| --- | --- | --- |
| 点数太多/延迟高 | 每帧点数、CPU、输出频率 | `point_filter_num`、体素大小 |
| 地图过厚 | 转弯片段、点时间、外参 | 先检查时间/外参，再看滤波参数 |
| 匹配点太少 | 盲区、量程、场景几何 | `blind`、`det_range`、采样密度 |
| 轨迹震荡 | IMU 噪声、残差、迭代次数 | `acc_cov`、`gyr_cov`、`max_iteration` |
| 地图无限增长 | ikd-Tree 点数、内存 | 地图体素、局部裁剪、动态点策略 |

调参数之前先确认输入和时间语义正确。错误传感器数据无法用参数“调好”。

---

## 10. Point-LIO：高带宽点级更新

### 10.1 用一句话概括

Point-LIO 在紧耦合 LiDAR--IMU 与增量地图框架上，进一步强调按单点或小批量点的采样时刻进行高频状态传播和几何更新，以降低等待完整扫描的延迟。

### 10.2 为什么需要点级更新

如果一帧扫描需要几十毫秒，在高角速度或高速移动时，等到整帧结束再进行一次主要更新会带来延迟。Point-LIO 的思想是：

```text
点 p_1 到来/可处理 -> 传播到 t_1 -> 匹配 -> 更新
点 p_2 到来/可处理 -> 传播到 t_2 -> 匹配 -> 更新
...
```

这并不意味着每个点都是独立系统，也不意味着不需要去畸变。它仍然依赖完整的 IMU、外参、局部地图和几何质量判断。

### 10.3 Point-LIO 的处理链

```text
点云预处理和每点时间
    -> IMU 队列同步
    -> 传播到点时刻
    -> 点从 LiDAR 系转到世界系
    -> ikd-Tree 查询邻域
    -> 平面质量检查
    -> 点到平面残差
    -> 迭代滤波更新
    -> 地图增量更新与高带宽状态输出
```

### 10.4 Point-LIO 的额外工程难度

点级高带宽带来更严格的工程要求：

- 更精确的每点时间语义；
- 更细粒度的 IMU/LiDAR 缓冲管理；
- 更频繁的状态读取和写入；
- 地图查询、地图插入与发布线程之间的并发控制；
- 对动态点和错误邻域更快的异常传播；
- 更严谨的 CPU 和端到端延迟测量。

所以 Point-LIO 的评估不能只看“topic hz 更高”。还要看是否有丢帧、是否有输入告警、误差是否稳定、CPU 是否接近饱和、控制缓存是否使用了新鲜状态。

### 10.5 历史工程阅读入口

历史 Point-LIO 代码在：

```text
Desktop/作品集/ROBOCON2025/Lidar_Pointlio/
```

建议阅读顺序：

1. `README.md`：识别建图/纯定位模式和输入输出；
2. `launch/mapping_unilidar_l2.launch.py`：理解实际设置的参数；
3. `src/parameters.cpp`：确认参数如何被读取；
4. `src/preprocess.cpp`：确认点时间和雷达格式如何处理；
5. `src/IMU_Processing.hpp`：理解 IMU 初始化和去畸变；
6. `src/Estimator.cpp`：看平面拟合、近邻和残差；
7. `src/laserMapping.cpp`：最后读同步、循环和发布。

不要先读 `laserMapping.cpp` 的末尾日志打印和发布代码。先画清输入、状态、观测和地图之间的关系。

### 10.6 FAST-LIO2 与 Point-LIO 的学习对比

| 问题 | FAST-LIO2 | Point-LIO |
| --- | --- | --- |
| 首先理解什么 | 扫描级直接点云 LIO | 点时间与高带宽更新 |
| IMU 的角色 | 高频传播与去畸变先验 | 高频传播并更紧密地服务点时刻更新 |
| 地图 | ikd-Tree 局部增量图 | 同类增量局部图路线 |
| 难点 | 外参、时间、残差、地图维护 | 还包括点级调度与资源管理 |
| 更合适的学习顺序 | 先学 | 在 FAST-LIO2 基础上再学 |

### 10.7 当前仓库中的 A/B 入口

当前组合仓库保留：

```text
tools/run_pointlio_ab_smoke.sh
tools/plot_pointlio_ab.py
```

脚本的意图是用同一段 Gazebo 输入 bag 分别回放 FAST-LIO2 和 Point-LIO，并收集输出频率、输入告警和 ground-truth 对齐误差。它是诊断入口，不是自动生成论文级结论的工具。

运行前必须确认：

- Point-LIO 外部 install 是否存在；
- 两者使用同一输入；
- 仿真 world、参数和 ROS domain 被记录；
- 输出是否真的收到；
- evaluator 是否成功写出指标；
- 证据等级是 Gazebo/bag replay，而非实机。

---

## 11. 全局定位、建图与回环

### 11.1 为什么局部 odometry 还需要 `map`

局部 LIO 追求连续平滑，长期会有缓慢漂移。比赛场地往往有固定地图和绝对目标，因此需要估计：

```text
T_map_odom
```

再合成：

```text
T_map_base = T_map_odom T_odom_base
```

这样，局部 LIO 即使连续地前进，固定地图模块仍能把它拉回场地全局坐标。

### 11.2 固定地图 scan-to-map

流程：

```text
已有冻结地图 + 当前已配准扫描 + LIO 初值
    -> ICP/GICP/NDT 局部配准
    -> 获得 map -> odom 修正候选
    -> 检查 fitness、overlap、连续性和时间
    -> 接受/拒绝修正
```

LIO 给配准提供初值，scan-to-map 给 LIO 提供全局锚定。这比从任意初值全局 ICP 更稳定。

### 11.3 `initialpose` 的真实作用

人工初始位姿通常是告诉系统：

```text
“机器人可能在地图的这个附近。”
```

它不是：

```text
“地图匹配已经成功，之后一定可信。”
```

正确做法是设置 `INITIALIZING` 状态，等待连续 scan-to-map 的质量达到门槛后才进入 `TRACKING`。

### 11.4 回环与位姿图

如果机器人长时间在未知区域建图，可以检测“当前地点是否曾经来过”。若匹配成功，添加回环约束：

```text
x_now <---- loop constraint ----> x_past
```

再通过位姿图或因子图优化全局轨迹。LIO-SAM 是这类前端和后端结合的代表。

### 11.5 因子图与 LIO-SAM

因子图中的：

- 节点：不同时间的状态/关键帧；
- 因子：IMU 预积分、LiDAR 匹配、GPS、回环等约束。

优点：全局一致性强；缺点：依赖、内存、关键帧管理和后端调试成本更高。

对短时、固定比赛场地任务，`LIO + 固定地图 scan-to-map` 往往比完整回环系统更直接。对长时间大范围探索，LIO-SAM 或位姿图更有价值。

---

## 12. 其他相关算法图谱

### 12.1 定位与建图方法的选择表

| 需求 | 常见选择 | 为什么 |
| --- | --- | --- |
| 快速局部 3D 里程计 | FAST-LIO2 | 直接点、IMU 紧耦合、ikd-Tree |
| 更强调点级高带宽 | Point-LIO | 连续点时刻更新路线 |
| 经典特征学习基线 | LOAM | 点到线/面结构清楚 |
| 地面车辆、算力有限 | LeGO-LOAM | 利用地面假设降低开销 |
| 长时间、回环、全局一致性 | LIO-SAM | 因子图和回环后端 |
| 视觉、LiDAR、IMU 融合 | LVI-SAM、FAST-LIVO2 | 在几何/纹理间互补 |
| 已知地图的重定位 | ICP/GICP/NDT | scan-to-map 全局锚定 |
| 2D 传感器与已有栅格图 | AMCL | 粒子滤波 2D 定位 |
| LiDAR-only 快速基线 | KISS-ICP | 简单直接，但不等价于 LIO |

### 12.2 LeGO-LOAM

LeGO-LOAM 依赖地面分割、聚类和车辆场景假设。它在平整道路上很实用，但对比赛场地的道具、机构、人员和非平面结构需要谨慎验证。

### 12.3 LVI-SAM 与 FAST-LIVO2

视觉可在 LiDAR 几何退化时补充纹理约束，LiDAR 可以在暗光或低纹理时提供尺度。代价是需要处理：

- 相机内参；
- 相机--LiDAR--IMU 外参；
- 图像与点云时间同步；
- 滚动快门；
- 曝光和运动模糊；
- 特征跟踪失效；
- 更高计算负载。

已有 YOLO 检测器不代表系统已经具有视觉里程计。检测、视觉 SLAM、语义建图是不同任务。

### 12.4 AMCL 与 2D 栅格定位

AMCL 是粒子滤波定位，适合：

- 平面移动机器人；
- 2D 激光雷达；
- 已知 2D 占据地图；
- 位姿初值不完全确定。

它和 3D LIO 的区别是观测模型和状态空间。可将 3D LIO 输出投影/桥接为导航层的 2D 位姿，但不能忽略高度、坡度和 3D 外参的影响。

### 12.5 规划与控制算法

定位只是闭环的一环：

```text
pose -> costmap -> global planner -> local controller -> chassis -> feedback
```

| 层 | 常见算法 |
| --- | --- |
| 全局规划 | A*、Dijkstra、D* Lite、Hybrid A* |
| 局部控制 | DWB、TEB、MPC、MPPI |
| 低层执行 | PID、前馈、限幅、急停 |

本项目中高层 NUC 不应替代 STM32 的实时安全职责。即使规划输出异常，MCU 仍应处理限位、过流、超时和急停。

### 12.6 视觉、RGB-D 与语义算法

| 算法/模块 | 解决什么 | 不解决什么 |
| --- | --- | --- |
| YOLO | 目标类别和 2D 边界框 | 精确 3D 位姿、时序可靠性 |
| RGB-D 反投影 | 从像素 + 深度得到 3D 点 | 目标长期跟踪、全局一致性 |
| PnP/AprilTag | 已知几何目标的相机位姿 | 稠密几何建图 |
| Mobile-SAM/分割 | 像素级目标区域 | 连续定位和控制决策 |

目标检测结果进入动作层前，必须有置信度、深度有效性、连续帧稳定性、时间新鲜度与机器人位姿质量门控。

---

## 13. ROS 2 与本项目的数据合同

### 13.1 ROS 2 中最容易被忽略的四件事

1. **时间戳**：消息是什么时间测到的？是不是仿真时间？
2. **frame_id**：数据在哪个坐标系中表达？
3. **QoS**：发布与订阅是否兼容？是否会丢历史数据？
4. **所有权**：谁可以发布某条 TF 或某个控制命令？

### 13.2 推荐的数据流

```text
/livox/lidar  (CustomMsg, 含 per-point time)
/livox/imu    (Imu)
        -> FAST-LIO2 / Point-LIO
        -> private upstream odometry
        -> pose bridge
        -> /mid360/local_odometry
        -> fixed-map localizer
        -> /mid360/localization_odometry + pose_status
        -> supervisor / navigation / perception / communication
```

### 13.3 唯一 TF 所有权

两个节点同时发布：

```text
odom -> base_link
```

会造成 TF 时间竞争：读取者有时读到 A，有时读到 B，表现为位姿跳变。解决方式是：

- 上游 FAST-LIO2 使用私有 frame 或关闭公开 TF；
- bridge 统一 frame、时间戳、协方差和状态；
- 外部消费者只使用对外合同。

### 13.4 pose contract 至少应包含什么

不应只发布位置和四元数。建议还包括：

```text
stamp
frame_id / child_frame_id
pose
twist（若可用）
covariance / quality
pose_age
sequence
drop_count
tracking_state
map_locked
degeneracy / residual / overlap（若可用）
reason_code
```

### 13.5 QoS 的基本判断

| 数据类型 | 常见倾向 | 原因 |
| --- | --- | --- |
| 高频 LiDAR/IMU | sensor-data、较小队列 | 新鲜数据比补旧数据重要 |
| 状态/配置 | reliable、适当历史 | 不能静默丢失关键状态 |
| 控制命令 | reliable + 明确超时 | 需要可追踪与失效处理 |
| 地图/静态 TF | transient local | 后加入的节点也应收到 |

具体 QoS 以驱动、节点实现和真实网络为准，不能机械套模板。

### 13.6 常用 ROS 2 检查命令

```bash
ros2 topic list
ros2 topic type /livox/lidar
ros2 topic info -v /livox/lidar
ros2 topic echo --once /livox/imu
ros2 topic hz /Odometry
ros2 run tf2_ros tf2_echo map base_link
ros2 node list
ros2 node info /laserMapping
ros2 bag info <bag-directory>
```

看到 topic 存在不等于数据正确。必须继续检查类型、频率、时间、frame、字段和 QoS。

---

## 14. 从零到可复现的实验路线

### 14.1 实验原则

每次实验都保存：

```text
目的
软件 commit
参数文件 hash
地图/world 版本
传感器/rosbag 信息
启动命令
时间范围
指标
失败原因
证据等级
```

否则几天后无法解释“这个图是怎么来的”。

### 实验 0：不运行算法，先理解输入

目标：确认 LiDAR 和 IMU 的消息合同。

检查：

- LiDAR topic 的类型；
- IMU topic 的类型；
- 每点时间是否存在；
- 时间单位；
- LiDAR/IMU 频率；
- frame_id；
- QoS；
- 是否使用 `/clock`。

通过条件：写出一张输入字段表，而不是只截图 `ros2 topic list`。

### 实验 1：静止 IMU 健康检查

目标：在机器人静止时确认 IMU 物理合理。

记录：

- gyro 均值和方差；
- accelerometer 模长；
- 时间戳间隔；
- 是否有饱和、跳变、NaN。

失败时先检查安装、驱动和单位，不要先改滤波器噪声。

### 实验 2：单帧点云质量

目标：理解盲区、点密度、遮挡和动态物体。

在 RViz 中检查：

- 近距离盲区；
- 墙面是否平整；
- 点是否随时间闪烁；
- 机器人本体是否被雷达扫到；
- 是否有大量人员/球等动态点。

### 实验 3：只验证去畸变

目标：比较开启和关闭/破坏逐点时间后的点云几何。

动作：

- 静止；
- 匀速直线；
- 原地旋转；
- 急转。

观察：墙面是否变厚、直线是否弯曲、转弯时是否重影。

### 实验 4：FAST-LIO2 局部 odometry

目标：确认局部轨迹连续、输出新鲜、地图没有明显发散。

指标：

- `/Odometry` 频率；
- pose age；
- CPU/内存；
- 有效点数；
- 残差；
- 路径连续性；
- 地图厚度。

### 实验 5：受控运动序列

按顺序做：

```text
静止  ->  直线  ->  横移/转弯  ->  原地旋转  ->  返回起点
```

每段运动有明确起止时间。这样出现问题时可把误差与动作关联，而不是只看一条长轨迹。

### 实验 6：固定地图定位

目标：验证 `map -> odom` 的锚定，而非只看局部轨迹。

步骤：

1. 先生成地图并冻结版本；
2. 提供初始位姿；
3. 跑 scan-to-map；
4. 记录 fitness、overlap、连续帧一致性；
5. 只在达到门槛后设置 `map_locked=true`；
6. 失锁时禁止绝对坐标投篮和导航目标。

### 实验 7：FAST-LIO2 与 Point-LIO A/B

公平条件：

```text
同一个输入 bag
同一个 ground truth / 对齐方式
同一个雷达点字段
同一个外参初值
同一个时间基准
同一个机器与 CPU 限制
同一个输出统计窗口
```

记录：

- 输出频率；
- 端到端延迟；
- translation/yaw error 分位数；
- CPU、内存；
- 丢帧和输入告警；
- 失锁或崩溃次数；
- 参数和 commit。

结论格式应写成：

```text
“在 world X、bag Y、参数 Z、机器 W 的仿真回放中，观察到……。”
```

而不是“Point-LIO 全面优于 FAST-LIO2”。

### 实验 8：故障注入

至少模拟：

- pose 超时；
- LiDAR 断开；
- IMU 时间跳变；
- map 未锁定；
- ICP fitness 连续变差；
- 双车通信失联；
- STM32/CAN 超时；
- 目标检测过期；
- 投篮取消。

验收：系统进入 PAUSED、RECOVERY、FAULT 或 ESTOP，不能继续执行绝对坐标动作。

### 实验 9：投篮闭环

把状态机前置条件写成可审计的布尔量：

```text
stage_ok
pose_valid
pose_fresh
map_locked
target_valid
ball_present
mechanism_ok
teammate_safe
communication_alive
```

每次拒绝动作应能指出具体原因。

---

## 15. 调参、诊断与常见失败

### 15.1 首先分辨是哪一层有问题

```text
输入层：topic、字段、单位、时间、QoS
标定层：外参、时间偏移、frame
估计层：噪声、去畸变、匹配、退化
地图层：体素、局部窗口、动态点、版本
合同层：TF 所有权、pose age、map lock
任务层：状态机、ACK、超时、取消
执行层：CAN/UART、限位、急停、反馈
```

不要看到“轨迹漂了”就直接改 `filter_size_map`。

### 15.2 现象到排查路径

| 现象 | 先查 | 再查 |
| --- | --- | --- |
| 静止时轨迹漂移 | IMU gyro、重力、时间戳 | 初始化与噪声参数 |
| 转弯时墙变厚 | 每点时间、时间偏移 | 外参旋转与 deskew |
| 起步后突然跳变 | IMU 初始化、TF 重复发布 | LiDAR/IMU 缓冲覆盖 |
| 地图倾斜 | IMU 轴向、外参、重力 | base_link/imu_link 定义 |
| 走廊中定位不稳 | 退化指标、有效平面 | 引入固定地图/视觉/运动约束 |
| CPU 持续升高 | 点数、地图大小、KD-tree | 体素、局部窗口、日志频率 |
| map 位姿跳变 | ICP 初值、TF 所有权 | 门控和连续性阈值 |
| 控制使用旧 pose | pose age、队列、QoS | latest-only 缓存和超时互锁 |
| 投篮误触发 | target age、球反馈、map lock | 状态机逻辑和 ACK |

### 15.3 噪声参数的误区

把 `acc_cov`、`gyr_cov` 调小不等于系统更准确。它表示“更相信 IMU”。若真实 IMU 噪声和偏置比设定更大，滤波器可能过度自信，反而更难被 LiDAR 纠正。

正确步骤：

1. 静止记录 IMU；
2. 估计均值、方差、漂移特征；
3. 在低速 bag 验证；
4. 再用高动态动作验证；
5. 记录每一次改动。

### 15.4 外参和时间偏移的误区

| 错误做法 | 为什么不行 |
| --- | --- |
| 从旧雷达直接抄外参到 MID-360 | 安装位姿和硬件基准可能不同 |
| 用固定 sleep 对齐消息 | sleep 不能表示传感器物理采样时间 |
| 开启在线外参后不检查初值 | 不足激励下估计可能发散或吸收偏置 |
| 用视觉效果判断外参正确 | 应结合急转、残差、重投影/几何一致性 |

### 15.5 动态环境处理

人员、篮球、队友机器人和机构都会产生动态点。可用策略：

- 去除机器人自身点；
- 用时间一致性拒绝短暂点；
- 按语义分割过滤已知动态类别；
- 只向静态地图加入高置信点；
- 当动态比例过高时降低地图更新或进入保护状态；
- 保留动态比例作为质量指标。

动态过滤不是“把看起来不顺眼的点删掉”，必须避免删掉真正有几何约束的静态结构。

---

## 16. 源码阅读路线

### 16.1 先读数据合同，再读算法主循环

推荐顺序：

```text
1. README 和 package 结构
2. MID-360 驱动/输入检查
3. 参数文件
4. preprocess
5. IMU processing
6. estimator / residual construction
7. laserMapping 主循环
8. pose bridge 与 TF contract
9. fixed-map localizer
10. supervisor、通信、控制
```

### 16.2 当前工程的路径地图

| 路径 | 学习目标 |
| --- | --- |
| `robocon-mid360-autonomy-stack/src/robocon_mid360_driver/` | 传感器驱动适配和输入诊断 |
| `robocon-mid360-autonomy-stack/src/vendor_fast_lio/` | FAST-LIO2 算法和 MID-360 参数 |
| `robocon-mid360-autonomy-stack/src/mid360_localization_contract/` | private upstream pose 到唯一 TF/pose contract |
| `robocon-mid360-autonomy-stack/src/mid360_map_localizer/` | 固定地图 ICP 与 map lock |
| `robocon-mid360-autonomy-stack/src/game_supervisor/` | 比赛状态机和互锁 |
| `robocon-mid360-autonomy-stack/src/camera_yolo_adapter/` | 目标观测如何进入任务合同 |
| `robocon-mid360-autonomy-stack/src/robocon_communication/` | ACK、幂等、心跳和超时 |
| `robocon-mid360-autonomy-stack/src/robocon_robot_control/` | NUC 与 STM32/CAN 控制边界 |
| `robocon-mid360-autonomy-stack/tools/` | 仿真回放、A/B、诊断和指标脚本 |

### 16.3 阅读 FAST-LIO2 的问题清单

打开一个函数前，先问：

```text
输入是什么类型和时间基准？
输出到哪个 frame？
状态存在哪里？
有哪些缓存和锁？
什么条件会拒绝当前数据？
时间、外参和噪声从哪里读取？
错误会怎样暴露给下游？
```

例如读 `preprocess.cpp`，不只是看循环：要确认 `offset_time` 被如何转换、放到内部点类型的哪个字段、单位是否一致、哪些点被过滤。

### 16.4 阅读一次完整消息生命周期

建议选一帧 LiDAR 消息，画出它的生命周期：

```text
CustomMsg
    -> preprocess
    -> lidar buffer
    -> 与 IMU 同步
    -> IMU propagation / deskew
    -> downsampled scan
    -> nearest neighbors
    -> plane residuals
    -> IEKF update
    -> registered cloud / odometry
    -> pose bridge
    -> localization / supervisor
```

能够画完这张图，就比只会背论文模块更接近真实工程理解。

---

## 17. 12 周学习计划

每周建议至少完成一个可复现产物：笔记、脚本、图表、测试或 run manifest。

### 第 1 周：Linux、Git、ROS 2 与坐标系

- 学习 shell、环境变量、进程和日志；
- 使用 `ros2 topic/node/interface`；
- 手画坐标树；
- 用 NumPy 实现 2D 齐次变换；
- 完成 `map -> odom -> base_link` 解释。

产物：一份坐标变换 notebook 或 Python 脚本。

### 第 2 周：三维旋转与 TF2

- 旋转矩阵、四元数、roll/pitch/yaw；
- 正确计算 yaw；
- 使用 `tf2_echo`；
- 理解静态外参与动态 TF 的区别。

产物：给定多条 TF 的点变换验证脚本。

### 第 3 周：概率和一维 Kalman Filter

- 高斯、协方差、预测和更新；
- 实现一维位置/速度滤波；
- 绘制 `Q/R` 敏感性图。

产物：带噪位置测量的 Kalman Filter 图。

### 第 4 周：IMU 运动学

- gyro、accelerometer、重力和零偏；
- IMU 时间积分；
- 静态 IMU 健康检查；
- 观察积分漂移。

产物：IMU 时间序列分析图和健康检查清单。

### 第 5 周：点云基础和 PCL/Open3D

- 读取、显示、裁剪、体素下采样；
- 理解 intensity/ring/time；
- 显示单帧 LiDAR 点云；
- 观察盲区、遮挡和动态点。

产物：单帧点云探索脚本。

### 第 6 周：ICP 与局部平面

- 实现或调用二维 ICP；
- 比较点到点和点到平面；
- 用 PCA 拟合平面；
- 观察初值和离群点影响。

产物：ICP 成功/失败对比图。

### 第 7 周：LOAM、LIO 与 ESKF 概念

- 理解 LOAM 特征路线；
- 解释紧耦合与松耦合；
- 了解误差状态和 IEKF；
- 写出 LIO 数据流图。

产物：一页“LiDAR-only 到 LIO”的架构笔记。

### 第 8 周：FAST-LIO2

- 阅读 MID-360 参数；
- 追踪 `offset_time`；
- 阅读 IMU 初始化与去畸变；
- 找到 ikd-Tree 近邻查询和地图插入。

产物：FAST-LIO2 源码路径图和参数说明表。

### 第 9 周：Point-LIO

- 阅读 Point-LIO launch 和参数；
- 对比 FAST-LIO2 与 Point-LIO 的更新组织；
- 解释 `use_imu_as_input` 和 `prop_at_freq_of_imu`；
- 建立公平 A/B 的实验表。

产物：对比矩阵和实验设计。

### 第 10 周：固定地图与全局定位

- 学习 ICP/GICP/NDT；
- 区分 `map` 与 `odom`；
- 理解 initial pose、map lock、relocalization；
- 设计失锁状态机。

产物：`UNINITIALIZED -> TRACKING -> LOST` 状态图。

### 第 11 周：导航、视觉与任务安全

- 学习 Nav2 的 costmap、planner、controller；
- 学习 YOLO + RGB-D 的三维反投影；
- 写投篮动作前置条件；
- 设计取消和急停路径。

产物：动作互锁真值表。

### 第 12 周：端到端回放与复盘

- 用 rosbag/Gazebo 跑完整输入到状态机链路；
- 进行至少一次故障注入；
- 收集频率、延迟、误差、CPU 和失败原因；
- 编写 run manifest；
- 进行 5 分钟项目讲解录音或演练。

产物：一份可追溯实验报告，而不是只保存一张 RViz 截图。

---

## 18. 面试与项目表达

### 18.1 30 秒版

“FAST-LIO2 和 Point-LIO 都是紧耦合的 LiDAR--IMU 里程计。IMU 在高频传播姿态、速度和位置，LiDAR 的每个点根据自身采样时间做去畸变，再与 ikd-Tree 局部地图做点到平面匹配，最后用迭代滤波修正位姿和偏置。FAST-LIO2 是当前 MID-360 栈的默认路线，Point-LIO 作为历史实机链路和 A/B 候选。局部里程计之外，我还用固定地图匹配估计 `map -> odom`，并把 pose age、地图锁定和检测新鲜度接入投篮互锁。”

### 18.2 2 分钟版的结构

```text
任务背景：比赛机器人急加速、振动、目标对准
问题：单独 LiDAR 会畸变，单独 IMU 会漂移
算法：LIO 用 IMU 传播 + LiDAR 点到平面更新
工程：保留 per-point time、统一 TF、做固定地图锚定
安全：pose age/map lock/目标有效性影响动作是否允许
验证：同 bag 回放、故障注入、记录证据边界
```

### 18.3 常见追问与回答方向

| 追问 | 回答核心 |
| --- | --- |
| 为什么需要 IMU？ | 高频预测、去畸变、改善 LiDAR 匹配初值；但长期积分会漂移。 |
| 为什么每点时间重要？ | 扫描内机器人在运动，逐点时间决定每个点应该使用的姿态。 |
| FAST-LIO2 和 Point-LIO 差别？ | 都是紧耦合直接点 LIO；Point-LIO更强调点级高带宽更新，需固定输入做对比。 |
| 为什么还有 ICP？ | LIO 是局部连续里程计，ICP/GICP/NDT 可与固定地图对齐提供全局锚定。 |
| 为什么不让 FAST-LIO 直接发布控制 TF？ | 避免私有 frame 和多发布者竞争，统一质量和 frame 语义。 |
| 如何证明算法更好？ | 同一 bag、同一外参、同一硬件、同一指标；报告失效样本和证据等级。 |
| 定位失效怎么办？ | pose age/地图状态触发 PAUSED/RECOVERY/FAULT，禁止绝对坐标动作。 |

### 18.4 不应说的话

- “LIO 不漂移。”
- “Point-LIO 一定比 FAST-LIO2 精度高。”
- “RViz 看起来直，所以定位准确。”
- “ICP fitness 小就绝对定位成功。”
- “有 YOLO 就能完成视觉定位。”
- “仿真通过等于实机通过。”
- “命令发给 STM32 说明球一定投出去了。”

---

## 19. 术语表与延伸阅读

### 19.1 术语表

| 术语 | 含义 |
| --- | --- |
| LiDAR | Light Detection and Ranging，激光雷达 |
| IMU | Inertial Measurement Unit，惯性测量单元 |
| Odometry | 相对里程计，连续但会漂移 |
| Localization | 在已知地图中的全局定位 |
| SLAM | Simultaneous Localization and Mapping |
| Deskew | 使用运动轨迹消除扫描内点云畸变 |
| ESKF | Error-State Kalman Filter，误差状态卡尔曼滤波 |
| IEKF | Iterated Error-State Kalman Filter，迭代误差状态滤波 |
| IKFoM | 迭代卡尔曼滤波的流形状态工具路线 |
| SE(3) | 三维旋转和平移构成的刚体变换群 |
| SO(3) | 三维旋转群 |
| ICP | Iterative Closest Point，迭代最近点配准 |
| GICP | Generalized ICP，广义 ICP |
| NDT | Normal Distributions Transform，正态分布变换 |
| ikd-Tree | 支持增量插入/删除的 KD-Tree |
| QoS | ROS 2 通信的可靠性、历史和持久性策略 |
| TF | ROS 坐标变换树 |
| `map` | 全局地图坐标系，可被重定位修正 |
| `odom` | 局部连续坐标系，允许长期漂移 |
| `base_link` | 机器人机体参考坐标系 |
| Pose age | 当前时刻与位姿采样时刻的差值 |
| Degeneracy | 场景几何约束不足导致的不可观或弱可观状态 |

### 19.2 推荐阅读顺序

1. ROS 2 TF2、QoS、rosbag 与 Nav2 官方概念文档；
2. PCL/Open3D 的点云、体素下采样、KD-Tree 和 ICP 教程；
3. 卡尔曼滤波、误差状态滤波和 IMU 预积分基础材料；
4. LOAM 论文及开源实现，用来理解特征驱动 LiDAR odometry；
5. FAST-LIO2 论文与源码，结合本项目 `vendor_fast_lio` 阅读；
6. Point-LIO 论文与历史 Point-LIO 代码，重点看点时间和高带宽处理；
7. LIO-SAM、GTSAM 与因子图材料，理解局部 odometry 和全局一致性的分工；
8. 当前项目的[算法综述](robocon-mid360-lio-algorithms-zh.md)和 `robocon-mid360-autonomy-stack.md`，把知识放回工程合同和证据边界中。

### 19.3 最后的学习检查表

在说“我学会 FAST-LIO2/Point-LIO”前，检查自己能否独立回答：

- [ ] 我能从 `map`、`odom`、`base_link` 推导一个点的坐标变换吗？
- [ ] 我能解释四元数、yaw、外参和时间偏移的区别吗？
- [ ] 我能说明 IMU 为什么漂移以及 LiDAR 如何纠正它吗？
- [ ] 我能解释逐点时间和去畸变的关系吗？
- [ ] 我能写出点到平面残差并解释每个符号吗？
- [ ] 我能说明 ikd-Tree 为什么适合 LIO 吗？
- [ ] 我能区分 FAST-LIO2、Point-LIO、LOAM、LIO-SAM 和 ICP 的角色吗？
- [ ] 我能设计一个公平 A/B 对照并写清证据边界吗？
- [ ] 我能说明定位失效时控制系统为何必须降级吗？
- [ ] 我能从当前工程里找到输入、去畸变、匹配、TF 和状态机的对应模块吗？

如果多数答案仍是“我知道名字但说不清数据怎样流动”，回到前面的最小练习，不要急于堆叠更多算法名。

