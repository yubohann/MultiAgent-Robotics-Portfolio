# Robocon MID-360 Autonomy Stack：项目技术说明

> 需要了解真实装车、现场联调、跨学科协作和竞赛规则映射时，请先阅读[详尽竞赛系统说明](robocon-mid360-autonomy-stack-详尽技术说明.md)；本页保留模块级接口与源码审计视角。

## 300 字项目概括

Robocon MID-360 Autonomy Stack 是面向 ROBOCON“飞身上篮”任务的 ROS 2 Humble 系统级自主栈。它把 Livox MID-360 官方驱动、内置 IMU、FAST-LIO2、固定地图定位、RGB-D/YOLO 感知、双车通信和 STM32/CAN 执行边界放到同一条真实竞赛闭环中。输入保留 `livox_ros_driver2/msg/CustomMsg` 的 `timebase` 和每点 `offset_time`，再由 LIO 前端产生局部连续里程计；唯一适配层把上游坐标转换成 `odom -> base_link`，固定地图定位负责 `map -> odom`，外部消费者读取版本化 pose contract。系统还设计了输入时间/点数/IMU 守门、地图版本和外参版本绑定、初始位姿、失锁/重定位状态机、投篮互锁、双车 ACK/心跳/过期和可取消任务。公开工程仓库把这些接口、配置、诊断脚本和仿真 profile 整理成可复用的 ROS 2 工作空间，并与真实竞赛机器人上的联调流程相互对应。

## 项目定位

- **目标平台**：Ubuntu 22.04、ROS 2 Humble、约 2023 年 Intel NUC；实际 SKU/NIC/IP 必须现场发现。
- **传感器**：Livox MID-360 + 官方 `livox_ros_driver2` + 内置 IMU。
- **默认 LIO 路线**：`CustomMsg -> FAST-LIO2`。
- **比赛边界**：NUC 负责任务编排和感知融合，STM32/CAN 负责实时 PID、限位、过流、超时和急停。

## 技术架构

```text
MID-360 internal IMU + CustomMsg
        -> FAST-LIO2 local odometry
        -> local odometry / quality adapter
        -> fixed-map scan-to-map and relocalization
        -> pose contract and unique TF tree
        -> vision / communication / supervisor / control
        -> STM32/CAN mechanisms
```

唯一对外 TF 约定为 `map -> odom -> base_link -> imu_link -> lidar_mid360`。系统要求完整四元数计算 yaw、记录 pose age/sequence/drop count，并在定位过期、地图未锁定、队友失联、球或机构反馈缺失时阻止绝对坐标动作和 `FireShot`。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `src/robocon_mid360_driver/` | 官方驱动适配、网络/输入诊断和 `CustomMsg` 守门 |
| `src/robocon_mid360_lio/` | FAST-LIO2 profile、局部里程计适配和参数 |
| `src/robocon_mid360_localization/` | 地图元数据、initial pose、scan-to-map、失锁与重定位 |
| `src/robocon_pose_contract/` | `/mid360/local_odometry`、`/mid360/localization_odometry`、`pose_status` 等外部合同 |
| `src/camera_yolo_adapter/` | 相机/YOLO 结果到任务层的显式桥接 |
| `src/game_supervisor/` | BOOT/PREFLIGHT/ACTIVE/PAUSED/RECOVERY/FAULT/ESTOP/FINISHED 状态机 |
| `src/robocon_communication/` | 任务 ID、序列号、ACK、幂等、心跳、超时、取消和双车协议 |
| `src/robocon_robot_control/` | NUC 高层动作请求与 STM32/CAN 反馈边界 |
| `config/`、`launch/`、`tests/` | mapping/localization/competition profile、启动文件和合同测试 |

代码设计上把估计器输出与控制输出解耦：IMU 高频传播、局部估计、latest-only 控制缓存和执行器回调不共用无限队列；质量层保留 `degeneracy_score`、动态点比例、scan-map overlap、map-update veto 原因和资源水位字段，即便实测值暂时为 `unknown/TBD`。

## 开发过程

1. 先审计原有 LIO、TF、topic 和位姿消费者，保留清晰的回退路径。
2. 在隔离 ROS 2 workspace 建立输入、pose、TF、游戏和双车合同，先完成静态与模拟消息测试。
3. 接入官方驱动和 `CustomMsg` 逐点时间语义，禁止在 LIO 前无时间字段地转成 `PointCloud2`。
4. 加入 FAST-LIO2 mapping/localization、地图版本、初始位姿、失锁/恢复和唯一 TF 适配。
5. 将视觉、通信、控制接入质量状态，再用台架、回放和真实比赛流程逐级验收。

## 涉及知识点

- ROS 2 Humble、QoS、launch、systemd 和节点生命周期
- Livox `CustomMsg`、逐点时间、IMU/LiDAR 同步
- FAST-LIO2、scan-to-map 和固定地图定位
- REP-103、TF 所有权、四元数 yaw 和外参标定
- 状态机、双车协议、ACK/幂等、超时和安全降级
- YOLO/视觉桥接、CAN 实时边界和比赛规则互锁
- rosbag 回放、性能指标、A/B 对照和风险审计

## 测试与验证

先运行 workspace README 中的静态检查、参数解析、几何/TF 合同测试；有 ROS 2 环境时再执行：

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

mapping、localization 和 competition 命令使用各自 profile，并在 `run_manifest` 里绑定地图、外参、规则配置、软件 commit 和硬件事实。接口测试、仿真回放和真实机器人联调分别承担不同的验证任务，共同构成完整的工程验证链。

## 学术申请中的项目价值

它能展示从论文算法到真实比赛系统的工程转换能力：保留逐点时间、设计唯一 TF 和质量合同、把定位失效传播到投篮互锁，并把高层任务和 MCU 安全边界分开。对机器人系统、SLAM、具身智能和可靠自主方向，这种“可观测、可取消、可恢复”的闭环比单纯跑通 LIO 更有说服力。

## 运行入口

- 官方驱动/输入检查：`src/robocon_mid360_driver`
- LIO profile：`src/robocon_mid360_lio`
- 固定地图定位：`src/robocon_mid360_localization`
- 比赛总控：`src/game_supervisor`
- 部署和自动启动：`deploy/`、`systemd/`（以仓库当前文件为准）

## 工程验证层次

项目材料按三层组织：接口与单元测试用于检查消息、坐标和状态迁移；仿真与回放用于复现传感器时序、场地几何和任务流程；真实竞赛机器人联调用于检验安装、网络、机构、通信和比赛节奏。三层材料使用同一套 topic、frame、状态和反馈命名，便于从实验室定位结果追到现场行为。

## 源码审计记录

| 层次 | 复核对象 | 代码阅读重点 |
| --- | --- | --- |
| 输入 | `robocon_mid360_driver`、网络预检、bag 检查脚本 | 官方驱动发现、QoS、`CustomMsg.timebase`、逐点 `offset_time`、IMU stamp 单调性和丢包诊断。 |
| LIO | `robocon_mid360_lio` FAST-LIO2 profile | 参数、点字段、IMU 顺序、mapping/localization profile、队列边界和异常退出。 |
| 局部位姿 | local odometry adapter | 只让一个节点发布 `odom -> base_link`，完整四元数转 yaw，协方差和 pose age 不被静默丢弃。 |
| 固定地图 | `robocon_mid360_localization` | map metadata、initialpose、scan-to-map、map->odom 更新、UNINITIALIZED/TRACKING/RELOCALIZING/LOST 转换。 |
| 合同 | `robocon_pose_contract` | topic 类型、frame、单位、QoS、频率目标、错误码、兼容桥和 preflight ready 语义。 |
| 视觉 | `camera_yolo_adapter` | 相机 frame、检测置信度、目标版本、失效超时和向总控传播的错误原因。 |
| 比赛 | `game_supervisor` | BOOT/PREFLIGHT/WAIT_START/ACTIVE/PAUSED/RECOVERY/FAULT/ESTOP/FINISHED 与取消/恢复。 |
| 协同 | `robocon_communication` | task ID、协议版本、sequence、ACK、幂等、心跳、过期、重试、取消和断链安全动作。 |
| 控制 | `robocon_robot_control`、CAN bridge | NUC 只发高层动作，STM32 保留 PID、限位、过流、离线、超时和急停。 |
| 部署 | launch、config、systemd、simulation profile | mapping/localization/competition 隔离、配置占位符、日志轮转和有界重启。 |

## 位姿数据流的逐步检查

1. 驱动将 MID-360 `CustomMsg` 和内置 IMU 放到明确 topic；输入守门检查 header、timebase、offset、NaN/Inf、点数和时间单调性。
2. FAST-LIO2 在私有 frame 内工作，适配器统一输出 frame、时间戳和协方差。
3. local odometry adapter 统一 frame 和协方差，记录 sequence、drop count、pose age、degeneracy 和短时 LiDAR 缺失行为。
4. 固定地图节点以人工 `/initialpose` 建立/重置全局锚点，只更新 `map -> odom`，不破坏局部连续性。
5. pose contract 将局部/全局 odometry、pose_status 和 preflight_ready 发布给视觉、通信和控制；消费者必须检查状态、age、地图锁定和原因。

## 运行状态与动作互锁

`preflight_ready` 只有在驱动、LIO、TF、地图、初始位姿、通信、CAN 和机构健康全部新鲜时才为真。`FireShot` 还要检查比赛阶段、目标确认、球存在、机构限位、队友安全、定位 age 和有效性；“命令已发出”不等于“球已送达”。失锁、队友消息过期、CAN 超时或机构反馈缺失时，状态机必须转 PAUSED/RECOVERY/FAULT，并提供取消和人工恢复入口。

## 建议的回放顺序

先用 synthetic messages 测输入守门和 TF 唯一性，再用 rosbag 验证 CustomMsg/IMU 时间语义；随后跑 mapping profile 生成带版本的地图，冻结后用 initialpose 进入 localization；最后注入过期 pose、断 LiDAR、断网、进程退出、错误 CAN 和取消事件，检查控制是否降级。只有这些路径通过，才进入 NUC 实机的网卡、外参和系统启动验收。

## 原理与理论基础

### 1. LiDAR--IMU 状态估计

FAST-LIO2 的目标是从 IMU 传播和 LiDAR 几何约束估计状态 `x=(R,p,v,b_g,b_a)`。IMU 在高频上积分角速度与加速度，LiDAR scan-to-map 残差约束点到局部平面/特征的距离。概念上可写成 `r_i = n_i^T (R p_i + p - q_i)`，其中 `n_i` 是地图平面法向、`q_i` 是对应点；滤波器或优化器用这些残差修正传播状态。MID-360 `CustomMsg` 保留每点 offset time，是因为同一帧内的运动畸变校正需要知道点的采样时刻，复制 header stamp 会把连续扫描错误地当成同时采样。

### 2. 坐标链和位姿合同

REP-103 下每个变换都是 `T_parent_child in SE(3)`。外部全局位姿应由 `T_map_base = T_map_odom * T_odom_base` 合成，静态安装关系再连接到 IMU 和 LiDAR。唯一 TF 所有权是系统稳定性的必要条件：两个节点同时发布 `odom -> base_link` 会造成时间竞争和跳变，读取四元数的 `z` 分量也不能代表 yaw。适配层因此承担 frame、协方差、时间戳和四元数到平面航向的转换，不让上游私有 frame 直接进入控制。

### 3. 固定地图定位与状态机

局部 LIO 提供连续相对运动，固定地图模块估计全局锚点 `T_map_odom`。人工 initial pose 只初始化这个锚点，不能重置局部轨迹；scan-to-map 通过局部配准残差和协方差判断是否锁定。`UNINITIALIZED -> TRACKING -> RELOCALIZING -> LOST` 的转换需要证据、超时和恢复动作：例如 overlap 下降、残差升高、pose age 超过阈值时禁止绝对坐标动作，重定位成功后才恢复。这样“地图没锁”不会被一个手动 TF 静默掩盖。

### 4. 新鲜度与安全互锁

每个 pose 都有时间 `t_pose`，控制时刻为 `t_now`，新鲜度 `age=t_now-t_pose`。控制合同应在 `age` 超过上限时令 `pose_valid=false`，并把 drop count、sequence、QoS、degeneracy 和 map-update veto 作为诊断。`FireShot` 是多个前置条件的逻辑合取：`enabled = stage & pose_valid & map_locked & target_ok & ball_present & mechanism_ok & teammate_safe`。任一条件为假都应进入取消/等待/恢复路径，不能用固定 sleep 代替传感器反馈。

### 5. 理论假设和失效模式

## 开发过程深拆

### 阶段 A：旧链路审计

先枚举原有 LIO、camera、TF、topic 和位姿消费者，记录每个生产者/消费者、frame、单位、频率、QoS 和 yaw 取法，再为 MID-360 重新整理安装外参和时间同步配置。

### 阶段 B：输入守门与隔离 workspace

研究 workspace 先放驱动配置模板、CustomMsg/IMU 检查器、参数 schema、synthetic message 和 TF 单元测试。网络 IP、NIC、固件、地图、外参和 NUC SKU 使用占位符；实测值通过 run manifest 回写。

### 阶段 C：定位链与质量层

先让 FAST-LIO2 产生连续 local odometry，再实现唯一 TF adapter；固定地图和 relocalization 之后才开放全局动作。质量层记录 pose age、sequence/drop、残差、overlap、degeneracy、dynamic ratio、map veto 和资源水位。

### 阶段 D：比赛闭环

视觉、双车和机构接入必须通过 pose_status/preflight_ready，而不是订阅旧位姿 topic。总控先完成 normal、timeout、duplicate、cancel、disconnect、lost、CAN fault 和 ESTOP 台架测试，再进入单车、双车和完整比赛。

## 源码阅读路线

阅读顺序是 config/schema -> input gate -> LIO adapter -> TF/pose contract -> map localization -> supervisor -> communication -> CAN bridge。每层都回答输入时间基准、失败传播、资源所有权和恢复入口。

估计器假设 IMU/LiDAR 时间和外参可校准、环境在局部窗口内有足够几何约束；固定地图假设地图版本和当前场地一致；控制合同假设 STM32 能可靠执行高层动作并报告状态。动态对手、稀疏场景、网卡丢包、机构限位或 CAN 超时会打破这些假设，所以项目把质量状态、任务取消和 MCU 安全边界设计成系统接口，而不是只在报告里描述。
