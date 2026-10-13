# ROBOCON MID-360 自主系统

## 研究报告与记忆版

**项目：** ABU ROBOCON 2025“飞身上篮”双机器人自主系统  
**核心传感器：** Livox MID-360 LiDAR 与内置 IMU  
**软件栈：** ROS 2 Humble、FAST-LIO2、RGB-D/YOLO、ROS 2 launch 与诊断系统  
**执行边界：** NUC 高层协同 -> UART/ESP-NOW -> STM32/CAN 实时控制  
**源码工作区：** `G:\mid360_ws_test_20260819`、`G:\MID360_ROBOCON_C_archive_20260825`  
**项目文档：** `docs/project-deep-dives/robocon-mid360-autonomy-stack*.md`

> 本文面向科研面试、研究生申请和技术作品集评审。报告首先说明当前能够直接审计的源码，再将它与作品集中记录的完整实体机器人系统联系起来。标记为 `TBD`、`unknown` 或 `synthetic` 的内容代表待通过 Linux、数据包、标定或实机记录补齐的证据字段。

---

## 1. 先记住这些

### 一句话

我构建了一套面向规则约束双机器人比赛的自主系统：MID-360 LiDAR 与 IMU 经过 FAST-LIO2 形成带质量检查的局部位姿，固定地图提供场地坐标，ROS 2 总控把新鲜的几何与语义证据转化为经过安全门控、可确认、可恢复的篮球动作，最终交由 STM32/CAN 控制器执行。

### 五个关键词

`MID-360` · `FAST-LIO2` · `fixed-map localization` · `quality contracts` · `dual-robot task supervision`

### 30 秒复述

这个项目解决的是一个具体的比赛系统问题：两台机器人需要在持续运动、振动、视角变化、网络延迟和机构负载变化下完成移动、接球、传球、瞄准与投篮。我把系统拆成感知、估计、语义感知、任务推理、通信、嵌入式执行和安全层。MID-360 的 `CustomMsg` 保留 `timebase` 和每点 `offset_time`，让 LIO 前端能够进行运动畸变补偿。位姿桥接节点把 FAST-LIO 的跟踪坐标转换成唯一的 `odom -> base_link` 公共接口，地图锚定节点维护 `map -> odom`。只有当位姿、地图锁定、目标新鲜度、球状态、机构健康和队友安全全部满足时，总控才接受投篮请求。这样，系统可以被标定、检查、回放，并能在真实比赛机器人上恢复运行。

### 核心研究观点

竞赛自主性的可靠性来自“每次状态迁移都有证据”。每个观测都带有坐标系、时间戳、质量状态和诊断原因；每个动作都包含权限门、传输确认、物理完成信号、超时处理和恢复路径。

---

## 2. 摘要

ROBOCON 篮球任务同时包含连续运动与离散规则：持球和传球时限、场地区域、目标几何、投篮条件以及比赛计时。项目面向这一任务构建端到端自主系统。Livox MID-360 和内置 IMU 为 FAST-LIO2 提供 LiDAR-惯性数据；固定场地图提供全局参考，同时保留局部轨迹的连续性。RGB-D/YOLO 提供篮筐观测和深度信息。ROS 2 连接状态估计、视觉感知、任务总控、双机器人通信和高层控制。UART 与 ESP-NOW 承担高层计算机之间及两台机器人之间的消息传输；STM32/CAN 保留电机与机构的确定性控制，包括 PID、限位、过流保护、超时处理和急停。

当前可审计的研究工作区将上述架构落实为清晰的软件合同。`mid360_input_guard` 检查 Livox 数据包结构和 IMU 健康状态；`mid360_pose_bridge` 校验源坐标系、归一化四元数、完成跟踪原点到机器人底盘原点的变换，并发布新鲜度和序列诊断；`mid360_map_odom_anchor` 根据 `/initialpose` 或经过验证的修正计算 `T_map_odom`，并暴露四状态跟踪机；`game_supervisor` 实现比赛生命周期、按顺序确认传球证据、`FireShot` 安全门以及具备过期和幂等语义的消息协议。

系统的工程成果是一套围绕外部算法和真实硬件建立的可复现接口层。研究价值集中在：将不确定性和故障原因从原始传感器时序持续传递到任务级决策，并让同一套接口支持合成测试、数据包回放、NUC 部署和后续实地验证。

---

## 3. 证据范围与版本关系

### 3.1 可直接审计的源码

G 盘工作区目前包含以下核心内容：

| 包或依赖 | 源码证据 | 作用 |
| --- | --- | --- |
| `mid360_localization_contract` | Python ROS 2 节点、YAML 模板、launch 文件、单元测试 | 输入验证、位姿转换、地图锚定、预检就绪 |
| `game_supervisor` | Python ROS 2 适配器、确定性核心、协议和测试 | 比赛生命周期、动作门控、双车消息语义 |
| `vendor_fast_lio` | FAST-LIO2 ROS 包和 MID-360 配置 | LiDAR-惯性里程计前端 |
| `vendor_livox_ros_driver2` | 官方 Livox ROS 2 驱动 | MID-360 网络传输和 `CustomMsg` 发布 |

该工作区采用隔离设计。私有地图、网络地址、序列号、固件版本、实测外参和硬件统计保留在模板之外，通过部署配置和 `run_manifest` 管理。

### 3.2 完整实体系统的项目证据

作品集中的详尽技术说明记录了完整竞赛系统，包括 RGB-D/YOLO 篮筐感知、双机器人协同、UART/ESP-NOW、STM32/CAN、底盘与机构测试、现场调试，以及机器人篮球和机器人篮球投篮项目的全国二等奖。这些文档提供实体系统背景和现场经验；G 盘原型提供可检查的定位合同和比赛总控实现。

### 3.3 项目演化关系

当前材料由早期实体系统说明和后期隔离合同工作区组成，两者属于连续演化过程：

```text
实体竞赛系统集成
    -> 接口审计与失效分析
    -> MID-360 / FAST-LIO2 隔离合同
    -> 确定性比赛总控与部署模板
    -> 实测硬件验证与回放证据
```

本文将这些阶段连成一条研究主线，同时明确每项陈述来自源码、模板、设计文档还是现场结果。

---

## 4. 将比赛问题转化为研究系统

### 4.1 运行挑战

机器人篮球要求机器人在移动和独立感知的同时维护共同任务状态。机器人需要反复回答四个问题：

1. 我在场地坐标系中的位置是什么？
2. 当前目标观测是否新鲜、稳定且具有可用几何关系？
3. 规则是否允许此刻执行这个动作？
4. 真实机构是否完成了刚才请求的动作？

这四个问题构成闭环。位姿跳变会改变瞄准几何；目标过期会改变投篮权限；队友消息过期会改变传球条件；机构缺少反馈会阻止下一个任务启动。

### 4.2 文档采用的比赛规则数值

详尽项目报告记录了 2024-08-14 修订的 ABU ROBOCON 2025 Robot Basketball 规则。以下数值直接解释了系统架构：

| 规则项 | 文档值 | 工程影响 |
| --- | ---: | --- |
| 比赛场地 | `15.0 m x 8.0 m` | 地图尺度、边界和初始位姿共用同一坐标系 |
| 围栏 | 高 `0.10 m`、宽 `0.05 m` | 边界几何进入建图和局部运动检查 |
| 篮板 | `1.80 m x 1.05 m` | 视觉目标具有结构化物理模型 |
| 篮筐高度 | `2.43 m` | 投篮模型使用固定目标高度 |
| 篮筐内径 | `0.450-0.459 m` | 目标中心估计和机构重复性都很重要 |
| 预赛时长 | `120 s` | 比赛计时进入总控状态 |
| 淘汰赛时长 | `160 s` | 比赛阶段通过版本化规则配置区分 |
| 持球时间 | `20 s` | 球状态包含明确的截止时间和过期路径 |
| 推进时间 | `8 s` | 进入进攻区域拥有独立计时器 |
| 最小传球距离 | `1.0 m` | 传球准备需要两车同步位姿 |
| 运球手高度 | `>= 0.70 m` | 需要机构或物理反馈作为证据 |
| 普通得分 | `2` 和 `3` 分 | 投篮区域和投篮类型进入任务推理 |
| 扣篮得分 | `7` 分 | 扣篮需要额外空间与运动证据 |

实际比赛数值应放入私有、版本化的 `game_profile`。当前提交的 `game_profile.template.yaml` 使用 `TBD`，等待规则版本与机器人硬件共同确认。

### 4.3 研究问题

**RQ1：几何可信度。** 当源坐标系、时间戳、外参或里程计新鲜度存在不确定性时，如何让高频 LiDAR-惯性估计仍然对任务控制有用？

**RQ2：语义可行动性。** 如何把目标观测和机构反馈转化为显式证据，而不是松散的回调信号？

**RQ3：系统恢复。** 当定位丢失、通信过期或执行器故障发生时，如何进入边界清晰的恢复状态，并在条件恢复后继续推进任务？

项目用输入合同、状态机和诊断字段回答这些问题，让质量信息在消费者边界清晰可见。

---

## 5. 总体架构

```text
Livox MID-360 CustomMsg + 内置 IMU
                |
                v
        mid360_input_guard
                |
                v
       FAST-LIO2 局部里程计
          私有源坐标系
                |
                v
        mid360_pose_bridge
     标准 odom -> base_link
                |
                v
       mid360_map_odom_anchor
       固定地图 -> odom 锚定
                |
                +----------------------+
                |                      |
                v                      v
   定位 / preflight              RGB-D + YOLO 篮筐目标
                |                      |
                +----------+-----------+
                           v
                 game_supervisor
             规则门、角色、计时器、恢复
                           |
                           v
                 双机器人通信
             UART / ESP-NOW、ACK、过期、去重
                           |
                           v
                  STM32 / CAN 执行
              底盘、投篮机构、反馈
```

### 5.1 各层职责

| 层级 | 输入 | 输出 | 关键设计 |
| --- | --- | --- | --- |
| 传感器传输 | MID-360 数据包、IMU | ROS 2 传感器话题 | 保留原生时间语义和消息类型 |
| 输入质量 | `CustomMsg`、`Imu` | `input_valid`、诊断 | 在早期识别结构错误和过期流 |
| 局部估计 | LiDAR、IMU | FAST-LIO 里程计 | 保持局部运动连续、高频响应 |
| 位姿合同 | 源里程计、标定参数 | `/mid360/local_odometry`、TF | 只有一个 `odom -> base_link` 所有者 |
| 全局定位 | 局部位姿、`/initialpose`、修正 | `/mid360/localization_odometry`、`map -> odom` | 增加场地参考而不打断局部运动 |
| 语义感知 | RGB-D 图像 | 目标检测与新鲜度 | 将像素和深度转化为任务证据 |
| 任务总控 | 位姿、地图、目标、球、机构、队友 | 动作决策 | 动作前检查完整证据集合 |
| 通信协议 | 任务消息 | ACK、序列、心跳 | 确定性处理过期和重复 |
| 实时控制 | 高层动作 | 电机/机构反馈 | 安全关键环节保留在 STM32/CAN |

这些接口明确了模块边界。现场调试时，可以把问题定位到传输、估计、感知、任务逻辑或执行层，并保留原始原因。

---

## 6. MID-360 数据链路与时间语义

### 6.1 Livox 原生消息

驱动发布 `/livox/lidar`，类型为 `livox_ros_driver2/msg/CustomMsg`；同时发布 `/livox/imu`，类型为 `sensor_msgs/msg/Imu`。`CustomMsg` 包含：

```text
header       ROS 标准消息头
timebase     第一测量点的时间基准
point_num    点数量
lidar_id     设备标识
points[]     CustomPoint 数组
```

每个 `CustomPoint` 都包含 `offset_time`、位置、反射率、tag 和扫描线。旋转式 LiDAR 的一帧点云跨越一段时间采集；保留每点时间，FAST-LIO2 才能完成运动补偿。将整帧点复制成同一个时间戳，会丢失转弯和加速阶段的重要时序信息。

### 6.2 `mid360_input_guard`

`mid360_localization_contract/input_guard.py` 具体执行以下检查：

- `point_num` 与接收点数组长度一致；
- 点数达到 `minimum_points`；
- `timebase` 非零，并在被接受的数据包之间递增；
- LiDAR header 时间戳非零且递增；
- 每个点的 `offset_time` 单调不下降；
- IMU 时间戳非零且递增；
- 角速度和线加速度均为有限数；
- LiDAR 与 IMU 到达时间处于配置的静默阈值内。

节点发布 `/mid360/input_valid` 和 `DiagnosticArray`。诊断内容包含点数、offset 时间跨度、流新鲜度、接收序列、接受序列、丢弃计数和最新原因。这样，“传感器似乎不稳定”会变成可观测、可回放、可定位的问题。

### 6.3 配置与时序

比赛配置为 LiDAR 和 IMU 设置 `0.30 s` 的目标静默阈值。该数值属于需要在目标机器人上测量和确认的部署门限。FAST-LIO2 模板使用 `10 Hz` 扫描频率、MID-360 LiDAR 类型、`0.5 m` 盲区，并显式提供 `time_offset_lidar_to_imu` 字段。外参和时间偏移始终作为标定输入管理。

---

## 7. FAST-LIO2 与位姿合同

### 7.1 估计器职责

FAST-LIO2 负责 LiDAR-惯性里程计。概念上，估计器使用 IMU 进行高频状态传播，再使用 LiDAR 几何残差校正状态。点到平面的残差可以写成：

\[
r_i = n_i^T(Rp_i + p - q_i),
\]

其中 `p_i` 是传感器坐标系中的点，`R,p` 表示当前位姿，`n_i,q_i` 表示局部地图平面。FAST-LIO2 算法作为外部依赖存在；本项目重点负责它外围的消息、坐标系、时间和质量合同。

### 7.2 `mid360_pose_bridge`

位姿桥接节点订阅配置好的 `nav_msgs/Odometry` 源话题，通常为 `/Odometry`。节点严格检查源里程计 frame 和 tracking child frame。默认源 frame 在选定 FAST-LIO2 构建版本确认前保持 `TBD`。

当源位姿位于 tracking 原点时，底盘位姿计算为：

\[
T_{odom\rightarrow base}
    = T_{odom\rightarrow tracking}
      \; T_{base\rightarrow tracking}^{-1}.
\]

桥接节点完成以下工作：

- 归一化完整四元数，并拒绝零范数四元数；
- 复制源时间戳和协方差；
- 将角速度和线速度转换到底盘原点；
- 应用可选的位置步长、速度、偏航角速度和高度质量阈值；
- 发布 `odom -> base_link` 的 `/mid360/local_odometry`；
- 发布系统唯一的 `odom -> base_link` TF；
- 发布 `pose_valid`、位姿年龄、源序列、接受序列、丢弃计数、frame 状态和质量原因。

这一设计解决了集成中常见的 TF 竞争问题：FAST-LIO2 可以发布私有 tracking 变换或关闭自身 TF 输出，公共 `odom -> base_link` 由桥接节点统一管理。

### 7.3 为什么必须使用完整四元数

Yaw 应从归一化后的完整四元数中计算。直接读取四元数 `z` 分量，只有在非常有限的纯平面特殊情况下才成立。几何测试覆盖了含 roll/pitch 的姿态、缩放后的非单位四元数、零四元数拒绝、变换复合以及传感器高度消除。

---

## 8. 固定地图定位与恢复

### 8.1 分离局部运动和全局参考

局部里程计负责连续、快速的短时运动；固定地图提供目标位姿、传球位置、投篮区域和重复比赛几何所需要的场地坐标。因此系统估计：

\[
T_{map\rightarrow odom}
    = T_{map\rightarrow base}
      \; T_{odom\rightarrow base}^{-1}.
\]

更新地图锚定只调整全局参考，不重置局部运动轨迹。

### 8.2 `mid360_map_odom_anchor`

节点使用以下输入：

- `/mid360/local_odometry`：最新局部位姿；
- `/initialpose`：位于 `map` frame 的 `PoseWithCovarianceStamped`；
- `/mid360/map_to_odom_correction`：经过验证的 `map -> odom` 修正；
- `/mid360/pose_valid` 和 `/mid360/relocalization_request`：质量与恢复控制。

当局部里程计存在且新鲜、`pose_valid=true` 时，系统接受初始位姿。修正消息必须满足 `header.frame_id=map`、`child_frame_id=odom`。锚定成功后，节点发布：

- `/mid360/localization_odometry`，表示 `map -> base_link`；
- `/mid360/map_locked`；
- `/mid360/pose_status` 版本化 JSON；
- `DiagnosticArray`，包含跟踪状态、锚定来源、锚定年龄、位姿年龄和拒绝原因；
- 跟踪锁定期间唯一的 `map -> odom` TF。

### 8.3 跟踪状态机

```text
UNINITIALIZED --接受锚定--> TRACKING
TRACKING --位姿无效或过期--> LOST
TRACKING --请求重定位--> RELOCALIZING
LOST --新鲜位姿和有效锚定--> TRACKING
RELOCALIZING --接受新的锚定--> TRACKING
```

四个状态含义为：

- `UNINITIALIZED`：尚未接受地图锚定；
- `TRACKING`：存在锚定、位姿有效且里程计新鲜；
- `RELOCALIZING`：操作者或系统请求重新建立锚定；
- `LOST`：原有锚定暂时无法使用，原因是位姿质量或新鲜度不足。

只有 `TRACKING` 状态会设置 `map_locked=true`。绝对坐标动作因此依赖当前证据，而不是依赖一条可能已经过期的旧 TF。

### 8.4 Preflight 就绪检查

`mid360_preflight` 检查话题类型、输入有效性、位姿有效性、标准里程计新鲜度，并在配置要求时检查地图锁定。节点发布 `/mid360/preflight_ready` 和诊断原因。默认里程计静默阈值为 `0.30 s`，启动宽限期为 `10 s`。比赛总控以该信号作为进入 `WAIT_START` 的前置条件。

---

## 9. ROS 2 运行时与启动组合

### 9.1 公共话题合同

| 话题 | 类型 | 作用 |
| --- | --- | --- |
| `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` | MID-360 原生数据包流 |
| `/livox/imu` | `sensor_msgs/msg/Imu` | FAST-LIO2 和输入健康检查 |
| `/Odometry` | `nav_msgs/msg/Odometry` | FAST-LIO2 源位姿 |
| `/mid360/local_odometry` | `nav_msgs/msg/Odometry` | 标准 `odom -> base_link` 位姿 |
| `/mid360/localization_odometry` | `nav_msgs/msg/Odometry` | 标准 `map -> base_link` 位姿 |
| `/mid360/input_valid` | `std_msgs/msg/Bool` | 输入结构和新鲜度门 |
| `/mid360/pose_valid` | `std_msgs/msg/Bool` | 位姿 frame、质量和新鲜度门 |
| `/mid360/map_locked` | `std_msgs/msg/Bool` | 固定地图跟踪状态 |
| `/mid360/preflight_ready` | `std_msgs/msg/Bool` | 比赛逻辑的系统级前置条件 |
| `/mid360/pose_status` | `std_msgs/msg/String` | 版本化 JSON 跟踪摘要 |

### 9.2 启动入口

`robocon_mid360_bringup.launch.py` 组合官方 Livox 驱动、`fastlio_mapping` 可执行文件和定位合同 launch。驱动或 FAST-LIO2 进程退出时，`OnProcessExit` 处理器会触发一致的顶层关闭。

`robocon_competition.launch.py` 将 MID-360 bringup 与 `game_supervisor` 组合起来。驱动配置、FAST-LIO2 参数、合同参数和总控参数都通过 launch 参数传入，私有部署值保持在公开模板之外。

### 9.3 部署顺序

1. 确认 NUC 型号、活动网卡、MID-360 IP、驱动版本和固件版本。
2. 将公开模板复制成私有配置文件。
3. 使用官方驱动检查 `/livox/lidar` 类型、点时间、IMU 时间戳和 QoS。
4. 采集静止、转弯和快速过渡数据包，用于时间同步和标定。
5. 测量 LiDAR-IMU 时间偏移、`base_link -> imu_link` 和 `imu_link -> lidar_mid360` 外参。
6. 配置 FAST-LIO2 源 frame，并确保系统中只有一个节点发布 `odom -> base_link`。
7. 建立并验证固定地图，随后设置 `/initialpose`。
8. 在开放绝对坐标比赛动作前运行 preflight 和诊断。
9. 接入 RGB-D/YOLO 目标质量、双机器人通信和 STM32/CAN 反馈。
10. 为每次建图、定位和比赛运行保存带版本信息的 `run_manifest`。

---

## 10. 语义感知与投篮闭环

详尽实体系统文档记录了 RGB-D 相机和 YOLO 篮筐检测链路。感知适配层承担三项工作：

1. 在相机 frame 中检测篮筐或比赛目标；
2. 使用深度和标定变换估计距离与方位；
3. 发布带时间戳、置信度、稳定性和过期信息的质量观测。

目标观测通过 TF 链与场地位姿结合，系统即可根据目标距离、航向误差、篮筐高度和机构标定模型计算投篮参数。具体的弹道参数属于机器人私有配置。

系统明确区分“命令已传输”和“动作已完成”。投篮请求先进入执行层，机构释放或完成反馈确认动作，随后任务总控推进到下一步或进入恢复路径。这种区分让任务状态与物理现实保持一致。

---

## 11. 比赛总控与双机器人协议

### 11.1 生命周期状态

`GameSupervisor` 定义以下状态：

```text
BOOT -> PREFLIGHT -> WAIT_START -> ACTIVE
                         |          |
                         v          v
                       FAULT      PAUSED
                                    |
                                    v
                                 RECOVERY

ACTIVE / PAUSED -> FINISHED
任何状态 --急停--> ESTOP
```

实现中包含 `BOOT`、`PREFLIGHT`、`WAIT_START`、`ACTIVE`、`PAUSED`、`RECOVERY`、`FAULT`、`ESTOP` 和 `FINISHED`。状态迁移由 `_require` 检查，操作指令必须按前置条件逐步推进。

### 11.2 有序传球证据

投球方任务依次经过：

```text
IDLE
  -> RECEIVER_READY
  -> PASS_ARMED
  -> PASS_EXECUTED
  -> RECEIPT_CONFIRMED
```

每一步都有独立方法和失败原因。该设计使用接收方实际就绪与传球完成证据驱动任务，而不是依赖固定睡眠时间。

### 11.3 `FireShot` 安全门

`request_fire_shot` 只有在以下条件全部满足时接受投篮：

```text
总控处于 ACTIVE
pose_valid
map_locked
pose_age <= 0.30 s
target_valid
target_age <= 0.30 s
ball_present
mechanism_healthy
teammate_safe
```

系统会返回第一个未满足条件的明确原因。后续可以继续加入比赛阶段、投篮区域、目标类别、释放反馈和持球计时等规则门。

### 11.4 版本化、可过期、幂等消息

`MessageEnvelope` 包含：

```text
protocol_version
message_type
task_id
sender_id
sequence
created_at_ns
expires_at_ns
payload
```

`Deduplicator` 先验证消息新鲜度，再按照发送方、任务和消息类型建立键值，拒绝过期消息，以及重复或倒退的序列号。即使通信出现重试和临时延迟，双车任务仍能保持确定性。

作品集描述的实体通信使用 UART 和 ESP-NOW。嵌入式边界保持清楚：NUC 请求高层动作，STM32/CAN 管理实时控制与安全。

---

## 12. 研究解读

### 12.1 主要贡献

项目最具辨识度的贡献，是面向竞赛自主性的“证据驱动架构”。定位、感知、通信和控制不再只是并列数据流，而是为每个消费者定义完整信息：

- 每个位姿带有 frame、时间戳、协方差、序列、年龄和原因；
- 每个目标带有有效性、年龄和几何上下文；
- 每个任务都有顺序明确的证据链；
- 每条消息都有身份、序列、生命周期和载荷；
- 每个执行请求都有反馈和取消路径。

这样，系统可靠性能够在接口层面被观察、测试和复现。

### 12.2 工程贡献

1. **保留原生传感器时序：** `CustomMsg.timebase` 和每点 `offset_time` 继续提供给 LIO 前端。
2. **唯一 TF 所有权：** 桥接节点管理 `odom -> base_link`，地图锚定节点管理 `map -> odom`。
3. **几何正确性：** 完整四元数归一化、传感器原点到底盘原点变换、速度变换和协方差传播。
4. **全局/局部分离：** 固定地图锚定改变场地参考，不重置局部运动。
5. **显式恢复机制：** 定位状态、preflight、重定位请求、暂停、恢复、故障和急停都是一等接口。
6. **确定性协同：** 有序传球证据，以及版本化、可过期、幂等的消息。
7. **实时边界：** ROS 2 高层推理与 STM32/CAN 电机、机构安全控制相互隔离。
8. **可复现性：** 模板、启动参数、诊断、测试和运行清单保存配置来源。

### 12.3 对机器人研究的意义

许多机器人演示关注一条成功轨迹；本项目进一步研究让轨迹值得信任的条件：时间、坐标系、质量、动作权限、确认和恢复。这些条件为 SLAM 算法与具身多机器人系统之间建立了可操作的连接。

---

## 13. 验证与可复现性

### 13.1 G 盘工作区已执行测试

在 `G:\mid360_ws_test_20260819` 中执行了纯 Python 合同测试：

```text
pytest -q \
  src/mid360_localization_contract/test/test_geometry.py \
  src/mid360_localization_contract/test/test_templates.py \
  src/mid360_localization_contract/test/test_tracking.py \
  src/game_supervisor/test/test_supervisor.py

16 passed in 0.37s
```

测试覆盖：

- 变换复合和求逆；
- 传感器高度消除；
- 完整四元数航向计算；
- 非单位和零四元数处理；
- 私有配置占位符与静态 TF 默认关闭；
- 跟踪状态迁移和地图锁定语义；
- 投篮证据与新鲜度阈值；
- 有序传球证据；
- ESTOP 终态行为；
- 消息过期与重复抑制。

ROS 适配器测试依赖 `rclpy`，属于 Ubuntu/ROS 2 目标环境测试。当前 Windows 环境将这一依赖记录为部署前提，源码测试结果保持可解释。

### 13.2 ROS 2 构建与启动

在目标 Ubuntu 22.04 / ROS 2 Humble 机器上执行：

```bash
cd ~/robocon_mid360_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

ros2 launch mid360_localization_contract competition_localization.launch.py

ros2 launch mid360_localization_contract robocon_mid360_bringup.launch.py \
  driver_config_path:=/absolute/private/MID360_config.local.json \
  fast_lio_params_file:=/absolute/private/fast_lio_mid360.local.yaml \
  contract_params_file:=/absolute/private/mid360_competition.local.yaml

ros2 launch game_supervisor robocon_competition.launch.py \
  driver_config_path:=/absolute/private/MID360_config.local.json \
  fast_lio_params_file:=/absolute/private/fast_lio_mid360.local.yaml \
  contract_params_file:=/absolute/private/mid360_competition.local.yaml \
  supervisor_params_file:=/absolute/private/supervisor.local.yaml
```

### 13.3 证据阶梯

```text
合成消息测试
    -> ROS 图与话题类型 preflight
    -> 数据包时序与标定检查
    -> 固定地图回放与重定位
    -> 单车台架测试
    -> 双车任务与通信测试
    -> 带 run manifest 的完整场地运行
```

每一级验证不同问题：单元测试验证数学与状态机不变量；数据包回放验证时间和几何；硬件运行验证安装、网络、机构和比赛节奏。

---

## 14. 失效模式与恢复设计

| 失效信号 | 检测方式 | 立即行为 | 恢复路径 |
| --- | --- | --- | --- |
| LiDAR 数据包结构错误 | 点数、timebase、offset 检查 | `input_valid=false` | 检查驱动和网络，回放原始数据包 |
| LiDAR 或 IMU 过期 | 到达年龄 | preflight 保持等待 | 恢复数据流并验证 QoS |
| 源 frame 错误 | 位姿桥接合同 | 拒绝该位姿 | 配置准确的 FAST-LIO2 frame 名称 |
| 四元数为零或无效 | 四元数归一化 | 拒绝该位姿 | 检查估计器输出和标定 |
| 位姿跳变过大 | 可选质量阈值 | `pose_valid=false` | 根据实测数据调整阈值或重定位 |
| 缺少地图锚定 | 跟踪状态 | `map_locked=false` | 接受经过验证的 `/initialpose` |
| 定位丢失 | 位姿有效性与新鲜度 | 暂停绝对坐标动作 | 请求重定位并接受新锚定 |
| 目标观测过期 | 目标年龄门 | 保持投篮请求等待 | 重新获取 RGB-D/YOLO 目标 |
| 队友消息过期 | 消息 envelope 年龄 | 任务保持待定 | 心跳、重试或安全返回 |
| 机构反馈缺失 | 总控安全快照 | 阻止投篮/传球 | 机构恢复或人工介入 |
| 进程退出 | launch `OnProcessExit` | 一致关闭系统 | 诊断后由 systemd 或人工重启 |
| 急停 | 总控 `ESTOP` | 进入终止安全状态 | 外部复位并重新执行 preflight |

系统采用统一模式：发现 -> 解释 -> 限制影响 -> 恢复。每个原因都会保留下来，现场调试可以沿证据链推进。

---

## 15. 局限与下一步研究实验

公开工作区有意保留若干部署事实。每个开放项都对应具体的测量计划，有利于后续形成完整证据包。

### 15.1 待补充的测量

- 数据包标定得到的 LiDAR-IMU 时间偏移；
- `base_link -> imu_link` 与 `imu_link -> lidar_mid360` 外参；
- NUC 上的端到端位姿频率和 P95 位姿年龄；
- 相对于场地参考的位移误差和航向误差；
- 地图配准重叠率和重定位耗时；
- 比赛网络下的数据包丢失与接受序列统计；
- 目标检测准确率、召回率、深度误差和观测稳定性；
- 传球完成时间和投篮完成率；
- 机构超时、恢复和 ESTOP 计数。

### 15.2 推荐实验

**A. 时间语义消融。** 对同一数据包比较原生 `CustomMsg` 每点时间和仅保留帧时间戳的转换，报告快速转弯时的运动畸变残差和位姿漂移。

**B. 坐标系合同消融。** 注入错误 child frame、重复 TF 发布者和未归一化四元数，测量错误发现时间以及下游动作抑制效果。

**C. 地图锚定鲁棒性。** 使用多个初始位姿，主动中断里程计并请求重定位，评估恢复到 `TRACKING` 的时间。

**D. 总控安全实验。** 回放过期位姿、过期目标、队友超时、无球和机构故障，统计误放行、正确拦截、恢复延迟和原因覆盖率。

**E. 双车协议压力测试。** 对消息进行乱序、重复、延迟和过期注入，报告去重准确率、任务完成率和安全回退行为。

**F. Sim-to-Real 对比。** 在仿真、数据包回放和真实机器人上使用同一目标与动作合同，比较位姿新鲜度、目标稳定性、动作延迟和恢复统计。

### 15.3 论文表述建议

可以将工作概括为：

> 面向规则约束双机器人操作的证据驱动 ROS 2 自主合同：结合原生 LiDAR 时序、统一坐标系所有权、固定地图锚定、语义目标质量、有序任务证据和实时嵌入式安全边界。

论文级实验应报告准确规则版本、硬件配置、数据包编号、标定版本、软件 commit、运行次数、失败定义和置信区间。当前模板已为这些字段预留位置。

---

## 16. 面试回答库

### 16.1 一分钟回答

我参与开发了一套双机器人 ROBOCON 机器人篮球自主系统。核心挑战是把异构、可能过期的信号转化为合法且可执行的动作。机器人使用 Livox MID-360 和内置 IMU，FAST-LIO2 提供连续局部里程计。我围绕估计器设计了 ROS 2 合同：输入守门节点保留 `CustomMsg` 的点级时间，位姿桥接节点检查源 frame 并产生唯一的 `odom -> base_link`，地图锚定节点根据经过验证的初始位姿或修正维护 `map -> odom`。实体系统进一步使用 RGB-D/YOLO 进行篮筐观测，使用 UART/ESP-NOW 实现双车消息，使用 STM32/CAN 负责实时底盘和机构控制。比赛总控按顺序确认传球证据，并要求新鲜位姿、地图锁定、目标、球、机构和队友安全全部满足后才允许投篮。我的主要贡献是定位与总控之间的集成边界，以及从传感器时序、标定到任务反馈和恢复的现场调试流程。

### 16.2 三分钟回答

这个项目是一套规则驱动的自主栈，而不是单一算法演示。在机器人篮球比赛中，只有几何、语义、规则、通信和机构状态同时一致，机器人才能稳定推进任务。因此我把这些依赖关系做成显式合同。

在感知层，官方 Livox 驱动发布 `CustomMsg` 和 IMU 数据。输入守门检查点数、`timebase`、每点 `offset_time`、时间戳单调性、有限 IMU 数值和流新鲜度。FAST-LIO2 使用原生时序生成局部里程计。位姿桥接节点检查源 frame、归一化四元数、把跟踪原点转换到底盘原点、保留协方差和时间戳，并独占公共 `odom -> base_link` TF。固定地图锚定节点将局部位姿与 `/initialpose` 结合，计算 `map -> odom`，并通过 `UNINITIALIZED`、`TRACKING`、`RELOCALIZING` 和 `LOST` 状态表达地图是否真正锁定。

实体系统再加入 RGB-D/YOLO 目标观测和双车通信层。总控经过 preflight 后进入等待和激活状态，也支持暂停、恢复、故障、急停和结束。传球必须按 receiver-ready、armed、executed、receipt-confirmed 顺序获得证据。投篮需要激活状态、位姿和目标新鲜、地图锁定、球存在、机构健康以及队友安全。消息携带任务身份、序列和过期时间，重复和过期消息会被确定性拒绝。高层计算机只发送动作请求，STM32/CAN 保留电机 PID、限位、过流保护、超时和急停。

这项工作的研究启发是：可靠自主性本质上是信息流问题。每个消费者收到的都不是一个裸数值，而是数值加上 frame、年龄、质量和原因。这样，标定、回放、现场调试和恢复可以进入同一个工程闭环。

### 16.3 技术问题：为什么保留 `CustomMsg`？

Livox 一帧扫描跨越一段采集时间。`timebase` 表示第一点的时间基准，每个点的 `offset_time` 表示其采样时刻。FAST-LIO2 可以利用这组信息完成运动补偿，快速转弯和加速时的几何一致性更有保障。

### 16.4 技术问题：为什么需要位姿桥接节点？

上游估计器的跟踪原点和 frame 名称属于实现细节，而整车系统需要稳定的 `odom -> base_link` 合同，包括校准后的原点、归一化四元数、协方差、新鲜度和诊断。桥接节点还统一 TF 所有权，消除变换竞争。

### 16.5 技术问题：为什么分开 `map -> odom` 和 `odom -> base_link`？

局部里程计应该保持平滑，便于控制；固定地图提供场地全局参考。更新 `map -> odom` 可以改变全局对齐，同时保留局部轨迹连续性，因此初始位姿、地图修正和重定位可以在不重置估计器的情况下完成。

### 16.6 技术问题：定位丢失时系统如何处理？

当源位姿过期、格式错误或超过质量阈值时，桥接节点将位姿标记为无效。地图锚定进入 `LOST` 或 `RELOCALIZING`，清除 `map_locked` 并发布原因。preflight 和总控会保持绝对坐标动作等待。获得新鲜有效位姿并接受新的锚定后，系统回到 `TRACKING`，任务可以继续。

### 16.7 技术问题：为什么投篮需要这么多门？

投篮同时依赖几何、感知、持球、机构和队友安全。新鲜位姿提供场地关系；新鲜目标提供瞄准几何；地图锁定提供可信全局坐标；球和机构信号表明物理准备状态；队友安全避免双车动作冲突。把这些条件写成逻辑合取后，每一个未满足项都有明确解释。

### 16.8 技术问题：你个人最大的收获是什么？

我学会把接口本身当作研究对象。算法输出只有在 frame、时间戳、协方差、新鲜度、失败原因和消费者假设都被定义后，才能真正服务于实体机器人。我也形成了短证据闭环：合成测试、数据包回放、标定、preflight、台架测试和带版本清单的场地运行。

---

## 17. 记忆卡片

### 五步故事

1. **感知：** MID-360 `CustomMsg` 加 IMU，保留原生点级时间。
2. **估计：** FAST-LIO2 生成局部运动；桥接节点生成标准 `odom -> base_link`。
3. **锚定：** `/initialpose` 或验证修正生成 `map -> odom`；跟踪状态表达地图锁定。
4. **决策：** RGB-D/YOLO、规则配置、队友协议和总控组合成证据门。
5. **执行与恢复：** UART/ESP-NOW 请求到达 STM32/CAN；反馈、超时、恢复和 ESTOP 完成闭环。

### 三个公式

```text
T_odom_base = T_odom_tracking * inverse(T_base_tracking)
T_map_odom  = T_map_base * inverse(T_odom_base)
FireShot    = ACTIVE & pose_valid & map_locked & fresh_target
              & ball_present & mechanism_healthy & teammate_safe
```

### 四个跟踪状态

```text
UNINITIALIZED -> TRACKING -> LOST
                       \-> RELOCALIZING -> TRACKING
```

### 四个传球状态

```text
receiver_ready -> pass_armed -> pass_executed -> receipt_confirmed
```

### 六个调试关键词

`frame` · `time` · `freshness` · `quality` · `acknowledgement` · `recovery`

### 结尾句

我把原生 LiDAR 时序、LiDAR-惯性估计、固定地图几何、语义目标证据、规则总控、双机器人通信和嵌入式执行连接成一个可检查的 ROS 2 闭环，让每个动作都有依据，让每个主要质量故障都有恢复路径。

---

## 18. 源码与证据索引

### 直接源码

```text
G:\mid360_ws_test_20260819\src\mid360_localization_contract\README.md
G:\mid360_ws_test_20260819\src\mid360_localization_contract\INTERFACE_CONTRACT.md
G:\mid360_ws_test_20260819\src\mid360_localization_contract\config\competition.yaml
G:\mid360_ws_test_20260819\src\mid360_localization_contract\config\fast_lio_mid360.template.yaml
G:\mid360_ws_test_20260819\src\mid360_localization_contract\config\map_metadata.template.yaml
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\input_guard.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\pose_bridge.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\map_odom_anchor.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\tracking.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\preflight.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\test\
G:\mid360_ws_test_20260819\src\game_supervisor\README.md
G:\mid360_ws_test_20260819\src\game_supervisor\config\supervisor.yaml
G:\mid360_ws_test_20260819\src\game_supervisor\config\game_profile.template.yaml
G:\mid360_ws_test_20260819\src\game_supervisor\config\run_manifest.template.yaml
G:\mid360_ws_test_20260819\src\game_supervisor\game_supervisor\supervisor.py
G:\mid360_ws_test_20260819\src\game_supervisor\game_supervisor\protocol.py
G:\mid360_ws_test_20260819\src\game_supervisor\test\
G:\mid360_ws_test_20260819\src\vendor_fast_lio\
G:\mid360_ws_test_20260819\src\vendor_livox_ros_driver2\
```

### 压缩档案与作品集背景

```text
G:\MID360_ROBOCON_C_archive_20260825\MID360_ROBOCON_C_archive_20260825.tar.gz
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-autonomy-stack.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-autonomy-stack-detailed-technical-description-en.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-autonomy-stack-详尽技术说明.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-lio-algorithms-zh.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-lio-learning-guide-zh.md
```

### 理论与工程参考

1. ABU ROBOCON 2025 Robot Basketball Rulebook，2024-08-14 修订版。
2. Xu, W. et al. “FAST-LIO2: Fast Direct LiDAR-inertial Odometry.” *IEEE Transactions on Robotics*, 2022。
3. Shan, T. et al. “LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping.” *IROS*, 2020。
4. Livox ROS Driver 2 消息与配置文档。
5. ROS 2 Humble、`tf2`、`diagnostic_msgs` 和 Nav2 接口文档。

---

## 最终总结

这个项目适合被呈现为一项扎根真实机器人集成的系统研究工作。它的核心特色是证据驱动闭环：原生传感器时序进入质量感知估计器；统一 TF 和固定地图合同提供可用几何；目标、队友和机构信号成为显式任务证据；版本化协议与嵌入式安全边界把决策转化为可恢复的物理动作。由此，系统建立了从 LiDAR 数据包到比赛行为的完整路径，每个假设清晰可见，每个恢复步骤都能被测试。
