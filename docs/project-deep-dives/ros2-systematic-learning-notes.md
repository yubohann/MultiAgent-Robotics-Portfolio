# ROS 2 Learning Notes：项目技术说明

## 300 字项目概括

ROS 2 Learning Notes 是一套面向机器人系统工程的持续型技术手册，不是可安装的软件包。目录以从基础到部署的学习路径组织 ROS 2 的节点、topic、service、action、参数、launch、TF2、QoS、rosbag、Nav2、MoveIt2、DDS、嵌入式部署、安全容错和云机器人等主题，并通过四份 Markdown 入口和一份完整目录保留章节层级。内容重点不在罗列 API，而在解释通信语义、时间戳与坐标树、QoS 兼容、生命周期、调试工具和跨设备部署中的工程取舍；例如同一 topic 的可靠性/历史深度/持久性会影响传感器与控制链路，TF2 的时间旅行和多机器人 frame 命名会影响定位与导航，launch/systemd/rosbag 则决定系统是否能启动、复现和恢复。该项目适合作为学习证据和技术面试材料，能够展示对 ROS 2 系统边界的理解；它不应被包装成已运行的机器人产品或硬件 benchmark，涉及具体版本、命令和硬件结果时仍需回到对应章节和本机验证。

## 项目定位

- **类型**：中文为主、英文入口补充的 ROS 2 工程学习手册。
- **读者**：希望从概念、通信和工具逐步进入机器人系统开发的工程师/研究生。
- **覆盖面**：ROS 2 核心、TF2、Nav2、MoveIt2、DDS/QoS、仿真、调试、部署、安全和前沿集成。

## 内容结构

```text
概念与 API -> 节点通信/参数/launch -> TF2/QoS/rosbag
           -> Nav2/MoveIt2/仿真 -> DDS/嵌入式/性能
           -> 安全、容错、云机器人和工程案例
```

`完整目录.md` 是章节索引，`README.md` 和 `README.zh-CN.md` 是项目入口；其他 Markdown 文件按主题展开原理、命令、常见故障和工程建议。由于仓库当前主要是笔记文档，知识之间通过术语、命令和示例建立链接，而非通过 Python package API 建立依赖。

## 核心内容与知识点

1. **通信模型**：publisher/subscriber、service、action、参数和 lifecycle node 的边界。
2. **坐标与时间**：TF/TF2 tree、frame 语义、时间戳、buffer、跨机器人 frame 命名。
3. **QoS 与 DDS**：reliability、durability、history、depth、deadline 和不兼容诊断。
4. **工具链**：launch、rosbag、rqt、命令行 introspection、日志和参数覆盖。
5. **导航/操作**：Nav2 代价地图、AMCL/SLAM、路径规划、MoveIt2 和传感器接入。
6. **工程化**：systemd、嵌入式/网络部署、实时性、性能、容错和安全。

## 开发过程

内容按“概念解释 -> 最小命令 -> 失败模式 -> 系统级取舍”扩展。章节目录先固定范围，再逐步补齐 QoS、TF2、多机器人、嵌入式和部署主题；每次新增内容都应保持术语、版本和命令一致。与实际项目结合时，笔记可以作为 Robocon MID-360、RoboCup CBG-WM 等仓库的背景知识，但不能替代那些项目自己的运行合同。

## 阅读与验证方法

```powershell
Get-Content -Raw README.md
Get-Content -Raw 完整目录.md
```

涉及 ROS 2 命令的章节应在对应发行版和工作区中逐条执行；如果只读文档，没有 ROS 2 环境，就只能把结论标为知识整理。建议用 `ros2 topic info`、`ros2 doctor`、`tf2_tools`、`ros2 bag info` 和 launch smoke 对具体工程进行回验证。

## 学术申请中的项目价值

它体现的是系统性学习和知识组织能力：能把抽象通信语义、坐标/时间问题和实际部署故障放在同一框架中。申请机器人系统、SLAM、导航或具身智能方向时，可将本手册作为“工程基础与自驱学习”的证据，并链接到实际代码项目说明。

## 运行入口

- 英文展示：`README.md`
- 中文入口：`README.zh-CN.md`
- 完整章节导航：`完整目录.md`

## 证据边界

这是学习笔记，不是 ROS 2 发行版文档，也不等于每一节都在同一台机器上实测。版本敏感命令、DDS 行为、Nav2 参数和硬件性能必须在目标环境中复核；引用笔记内容时应给出章节和实际验证环境。

## 章节审计记录

| 学习层 | 主题 | 应掌握的工程问题 |
| --- | --- | --- |
| 基础 | node、topic、service、action、parameter | 何时用异步流、请求响应或可取消长任务，接口如何演进。 |
| 工具 | CLI、rqt、rosbag、launch、lifecycle | 如何发现图、保存回放、编排启动和处理节点退出。 |
| 坐标 | TF/TF2、frame、时间旅行、多机器人 | 谁拥有 TF、时间戳如何查询、frame 命名如何避免环和冲突。 |
| QoS | reliability、durability、history、depth、deadline | 传感器和控制链怎样在延迟、丢包和带宽之间取舍。 |
| 导航 | Nav2、AMCL、SLAM、costmap、规划器 | 地图、定位、代价地图和控制器如何构成导航闭环。 |
| 操作 | MoveIt2、轨迹、碰撞和执行器 | 规划结果如何验证、取消和安全执行。 |
| 系统 | DDS、实时性、嵌入式、systemd、网络 | 多机部署、资源限制、自动恢复和日志治理。 |
| 安全 | 权限、容错、故障诊断、Micro-ROS/云机器人 | 失联、过期、节点崩溃和不可信输入如何降级。 |

## 一条 ROS 2 系统排障链

1. 用 `ros2 doctor`、`ros2 node list`、`ros2 topic list/info` 确认 discovery 和 QoS 连接。
2. 用 `ros2 topic echo/hz`、`ros2 service list`、`ros2 action list` 看消息是否新鲜、频率是否符合目标。
3. 用 `tf2_tools` 或 RViz 检查 frame graph、时间戳和发布者所有权；不要只看静态图是否“连上”。
4. 用 `ros2 bag info/play` 重现传感器输入，再把问题缩小到 QoS、时间、坐标、参数或执行器。
5. 用 launch 参数和 lifecycle 控制重启/暂停，最后用 systemd 或容器验证自动恢复和日志路径。

## 与作品集工程的对应

- Robocon MID-360 需要 CustomMsg、TF、QoS、rosbag、固定地图、systemd 和故障状态机。
- RoboCup CBG-WM 需要 Jazzy/Nav2、视觉 topic、shooter service、launch 参数和 Sim2Real 接口。
- Rivermark/AeroCity/HM3D 的 Isaac wrapper 可用同样的启动、资源和证据治理思路。

## 阅读与复习方法

每读一章都应写出“输入、发布者、消费者、时间基准、QoS、失败动作、恢复入口”七项；读命令时记录发行版和依赖版本；读架构时画出唯一 TF/数据流；读部署章节时实际演练停止、重启、回放和日志收集。这样笔记才会从知识列表变成可用于系统设计的检查表。

## 原理与理论基础

### 1. ROS 2 图与 DDS

ROS 2 节点图建立在 DDS/RTPS 的发现和数据分发之上。publisher 与 subscriber 只有在 topic 类型和 QoS 能兼容时才能通信；topic 名称相同不保证数据会到达。reliability 决定丢包时是否重传，durability 决定新加入者是否能获得历史，history/depth 决定缓存边界，deadline/liveliness 则可用于新鲜度和在线状态监测。不同传感器、状态、控制和日志应使用不同 QoS，因为每类数据的失败代价不同。

### 2. TF2 的时间变换

TF2 保存带时间戳的坐标变换 T_parent_child(t)。查询 lookup_transform(target, source, t) 本质上是在缓存中寻找能把 source 在时刻 t 映射到 target 的链；未来/过期时间、环、重复发布者和 frame 名称错误都会导致 extrapolation 或错误姿态。机器人系统中的位姿并非一个静态矩阵，而是矩阵、时间、所有权和 frame 语义的合同。

### 3. 生命周期、动作和可恢复性

service 适合短请求，action 适合可反馈、可取消、可能超时的长任务；lifecycle node 用 configure/activate/deactivate/cleanup 管理资源。导航、建图和执行器需要把启动成功、数据新鲜、停止和故障恢复分别建模，不能用一个 is_running 布尔值覆盖所有状态。launch 负责编排，systemd/容器负责进程级重启，两层边界不同。

### 4. Nav2 与闭环控制

导航链可写成感知/定位 -> costmap -> global planner -> local controller -> actuator -> feedback。costmap 是障碍概率/代价的空间表示，规划器寻找可行路径，controller 将路径转成受限速度，反馈再更新状态。参数变化会改变安全距离、膨胀半径、控制频率和恢复行为，因此配置、地图、frame 和传感器 QoS 必须一起记录。

### 5. 理论假设和失效模式

ROS 2 文档中的命令只在发行版、RMW、网络和消息定义匹配时成立。发现失败、QoS 不兼容、时间源错误、TF 漂移、executor 饥饿、callback 阻塞和 systemd 误重启都可能让节点存在但系统不可用。笔记的工程方法是从数据流和失败证据反推根因，而不是把问题归因于 ROS 2 本身。
