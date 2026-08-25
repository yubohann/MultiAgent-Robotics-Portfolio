# 5_in_drone 资产说明与官方安全距离口径

生成日期：2026-05-04

本文说明 `assets/5_in_drone` 下这架无人机在本项目中的模型定位、几何、质量惯量、执行器与仿真配置，并单独说明“两架无人机安全距离”的官方法规口径。注意：这里的“官方安全距离”不是本项目训练奖励、避障盾或编队间距参数。

## 1. 模型定位

`5_in_drone` 是一个自定义 5 英寸级四旋翼竞速无人机仿真资产，不是 DJI、PX4 标准机型或某个公开商业机型的完整硬件 BOM。`5_in` 来自资产目录与文件名，可理解为 5 英寸桨级别的 FPV/racing quad 资产；但当前 checkout 中 `.dae` 网格是 Git LFS pointer，不包含真实 mesh 顶点，因此本文不把“5 英寸桨直径”当作从 mesh 实测得到的参数。

本资产主要由以下文件组成：

| 文件 | 作用 |
| --- | --- |
| `5_in_drone.usd` | 顶层 USD，引用/组合传感器与物理层 |
| `configuration/5_in_drone_base.usd` | 几何外观层，体积约 102.6 MB |
| `configuration/5_in_drone_physics.usd` | 物理层，包含刚体、关节、质量、碰撞近似等 USD/PhysX 信息 |
| `configuration/5_in_drone_sensor.usd` | 传感器层，目前可读 token 只显示一个 `a_5_in_drone` Xform |
| `urdf/5_in_drone.urdf` | URDF 源模型，最清晰地记录了质量、惯量和四个桨/电机位置 |
| `meshes/base_link.dae` | 机身 mesh；当前是 Git LFS pointer，声明原始大小 155,702,931 bytes |
| `meshes/prop.dae` | 桨叶 mesh；当前是 Git LFS pointer，声明原始大小 3,141,267 bytes |

## 2. 结构与几何参数

该机是 X 型四旋翼：一个 `body` 机身 link，加四个 `prop` link，通过四个绕 Z 轴旋转的 revolute joint 与机身相连。

URDF 中四个桨/电机中心相对机身原点的位置如下：

| 关节 | 子 link | 坐标 `xyz`，单位 m | 轴向 |
| --- | --- | --- | --- |
| `m1_joint` | `prop1` | `(0.0883, 0.0883, 0.015)` | `(0, 0, 1)` |
| `m2_joint` | `prop2` | `(0.0883, -0.0883, 0.015)` | `(0, 0, 1)` |
| `m3_joint` | `prop3` | `(-0.0883, 0.0883, 0.015)` | `(0, 0, 1)` |
| `m4_joint` | `prop4` | `(-0.0883, -0.0883, 0.015)` | `(0, 0, 1)` |

由此可计算：

| 参数 | 数值 |
| --- | --- |
| 单轴投影臂长 | `0.0883 m` |
| 电机中心到机体原点的几何半径 | `0.124875 m` |
| 相邻电机中心距 | `0.1766 m` |
| 对角电机中心距 / wheelbase | `0.24975 m`，约 `250 mm` |
| 桨平面高度 | `z = 0.015 m` |

项目控制分配代码中的 `arm_length = 0.035 m` 是控制分配参数，不等于 URDF 几何电机半径 `0.124875 m`。这意味着“控制力矩臂”和“外观/URDF 几何臂长”在当前代码中并不完全一致；如果要做严格物理辨识或 sim-to-real，需要先统一这两个口径。

## 3. 质量、惯量与碰撞

URDF 明确给出的惯性参数：

| 项 | 数值 |
| --- | --- |
| 主刚体质量 | `0.5 kg` |
| 质心原点 | `(0, 0, 0)` |
| `Ixx` | `0.003 kg*m^2` |
| `Iyy` | `0.003 kg*m^2` |
| `Izz` | `0.006 kg*m^2` |

`prop1` 到 `prop4` 在 URDF 中没有独立 inertial 块，因此按 URDF 源文件看，质量与转动惯量主要集中在 `body`。视觉和碰撞都使用同一组 mesh：机身使用 `base_link.dae`，四个桨使用 `prop.dae`。USD 物理层可读 token 显示包含 `PhysicsRigidBodyAPI`、`MassAPI`、`ArticulationRoot`、`RevoluteJoint`、`convexHull` 等信息，说明导出到 USD 后使用了刚体、关节与凸包碰撞近似。

## 4. 关节与旋翼方向

四个旋翼关节都是 `revolute`，转轴为 Z 轴。URDF 中 `dynamics damping="0.0" friction="0.0"`，关节 limit 被注释掉，实际不限制转角。

`assets/five_in_drone.py` 中的默认关节初速度为：

| 关节 | 初始角速度 |
| --- | --- |
| `m1_joint` | `+200 rad/s` |
| `m2_joint` | `-200 rad/s` |
| `m3_joint` | `+200 rad/s` |
| `m4_joint` | `-200 rad/s` |

也就是说，1/3 号桨同向，2/4 号桨反向，形成交替旋向，便于抵消偏航反扭矩。

## 5. 仿真 spawn 配置

`FIVE_IN_DRONE` 在 `assets/five_in_drone.py` 中封装为 IsaacLab `ArticulationCfg`：

| 配置项 | 数值/行为 |
| --- | --- |
| prim path | `{ENV_REGEX_NS}/Robot` |
| USD 路径 | `assets/5_in_drone/5_in_drone.usd` |
| 接触传感器 | canonical 配置中 `activate_contact_sensors=True` |
| 重力 | `disable_gravity=False`，即启用重力 |
| 陀螺力 | `enable_gyroscopic_forces=True` |
| 最大反穿透速度 | `max_depenetration_velocity=10.0` |
| 自碰撞 | `enabled_self_collisions=False` |
| solver position iterations | `4` |
| solver velocity iterations | `1` |
| sleep threshold | `0.005` |
| stabilization threshold | `0.001` |
| actuator | dummy implicit actuator，`stiffness=0.0`，`damping=0.0` |

`assets/five_in_drone_graph_masac_training_backup.py` 是训练用备份配置：它直接加载 `configuration/5_in_drone_physics.usd`，并关闭 contact sensors，用于隔离多机 Graph-MASAC 训练时的 PhysX 接触报告开销。

## 6. 电机、推力和力矩模型

该模型的飞行动力不是靠 URDF 关节 motor 直接驱动，而是由 action term 根据电机角速度计算总推力和机体系力矩，然后施加到 `body` 上。

单机控制默认参数来自 `tasks/drone_racer/mdp/actions.py`：

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| `thrust_coef` | `2.25e-7` | 推力系数 |
| `drag_coef` | `1.5e-9` | 偏航反扭矩系数 |
| `omega_max` | `5145 rad/s` | 最大电机角速度，约 `49,140 RPM` |
| `init` | `(2572.5, 2572.5, 2572.5, 2572.5)` | 控制模型的初始/悬停参考角速度 |
| `taus` | `0.0001 s` 每电机 | 电机一阶响应时间常数 |
| `max_rate` | `+50000 rad/s^2` | 电机角速度最大上升率 |
| `min_rate` | `-50000 rad/s^2` | 电机角速度最大下降率 |
| `use_motor_model` | 默认 `False` | 默认绕过电机延迟；部分 Graph-MASAC 主线会强制开启 |

基于上述系数：

| 工况 | 单桨推力 | 总推力 | 等效悬停/举升质量 |
| --- | --- | --- | --- |
| `omega = 2572.5 rad/s` | `1.489 N` | `5.956 N` | `0.607 kg` |
| `omega = 5145 rad/s` | `5.956 N` | `23.824 N` | `2.429 kg` |

因此，按控制模型注释中的参考质量 `0.6076 kg`，最大总推重比约为 `4:1`。如果按 URDF 写死质量 `0.5 kg` 计算，则最大总推重比约为 `4.86:1`，而 `2572.5 rad/s` 的参考推力已经约等于 `1.21 g`。这也是一个需要注意的模型口径差异。

力/力矩分配矩阵的结构为：

```text
T  = f1 + f2 + f3 + f4
Mx =  arm_length/sqrt(2) * f1 - arm_length/sqrt(2) * f2 - arm_length/sqrt(2) * f3 + arm_length/sqrt(2) * f4
My = -arm_length/sqrt(2) * f1 - arm_length/sqrt(2) * f2 + arm_length/sqrt(2) * f3 + arm_length/sqrt(2) * f4
Mz =  drag_coef/thrust_coef * f1 - drag_coef/thrust_coef * f2 + drag_coef/thrust_coef * f3 - drag_coef/thrust_coef * f4
```

其中 `fi = thrust_coef * omega_i^2`。动作输入在单机 direct motor 模式下先从 `[-1, 1]` 映射到 `[0, omega_max]`，再通过上述矩阵变成机体系总推力和三轴力矩。

## 7. 传感器与缺失信息

当前可读资产信息没有明确列出相机、IMU、GPS、气压计、电调、电池、飞控板、图传、接收机等真实硬件配置。`5_in_drone_sensor.usd` 的可读 token 只显示一个 Xform，因此本文不臆造传感器硬件参数。

当前 `.dae` 文件是 Git LFS pointer，真实网格没有落地到工作区。因此以下信息不能从当前文件实测：

- 机架材料、厚度、外形包围盒；
- 桨叶真实直径、螺距、叶型；
- 机身、桨叶、保护罩等细分 mesh 的精确几何尺寸；
- 真实硬件 BOM 与电气参数。

如果需要精确外形尺寸，应先拉取 Git LFS 实体文件，再用 mesh 工具或 USD stage 计算 bounding box。

## 8. 两架无人机的官方安全距离标准

结论：主流官方法规没有给“两架普通小型无人机之间必须保持 X 米”的统一固定数值。官方口径通常是“不得造成碰撞风险 / 保持 well clear / 按空管或批准运行条件保持必要安全间隔”。因此不能把本项目里的 `safety_shield_trigger_dist`、`hard_dist`、编队 slot 间距等训练参数称为官方标准。

### 中国大陆官方口径

《无人驾驶航空器飞行管理暂行条例》（国令第 761 号，2024-01-01 起施行，CAAC 页面显示有效）第 32 条要求操控无人驾驶航空器时，按照国家空中交通管理领导机构的规定保持必要的安全间隔；实施超视距飞行时，要掌握飞行空域内其他航空器动态并采取避免相撞措施。第 33 条规定避让规则：避让有人驾驶航空器、无动力装置航空器以及地面/水上交通工具；单架飞行避让集群飞行；微型无人驾驶航空器避让其他无人驾驶航空器。

这说明中国大陆现行公开法规给的是“必要安全间隔”和避让规则，不是公开固定米数。具体间隔应以空管批准、适飞/管制空域规则、任务申请/运行批准和国家空中交通管理领导机构后续规定为准。

官方来源：

- CAAC《无人驾驶航空器飞行管理暂行条例》：https://www.caac.gov.cn/XXGK/XXGK/FLFG/202401/t20240115_222642.html
- CAAC《民用无人驾驶航空器运行安全管理规则》（CCAR-92，交通运输部令 2024 年第 1 号）：https://app.caac.gov.cn/XXGK/XXGK/MHGZ/202401/t20240103_222566.html

### 美国 FAA Part 107 口径

FAA Part 107 同样没有给两架小型无人机之间的固定米数。14 CFR §107.37 要求小型无人机让行所有航空器/空中车辆，不得从其上方、下方或前方通过，除非已经 well clear；并且不得把小型无人机操作到足以造成碰撞危险的近距离。14 CFR §107.35 还规定，同一个人不能在同一时间操纵、担任 RPIC 或视觉观察员参与超过一架无人机的运行。

所以在美国 Part 107 下，两架无人机同时运行的官方重点是：每架机的责任人/观察能力要合规，并且任意两机不得接近到造成碰撞危险。固定的“2 m / 5 m / 10 m”并不是 Part 107 给出的法定标准。

官方来源：

- 14 CFR §107.37： https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.37
- 14 CFR §107.35： https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.35
- 14 CFR §107.31： https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107/subpart-B/section-107.31

### EASA 欧盟口径

EASA 的开放类/特定类规则也以 VLOS、避碰、风险评估和运行授权为核心，不提供两架普通无人机之间的统一编队间距。EASA 的 Easy Access Rules 指出，远程驾驶员需要保持视觉扫描并避免碰撞；若看到低空航空器可能与无人机发生交互，应立即降低无人机高度，例如降到离地小于 10 m，并使无人机距离另一航空器不小于 500 m；如果做不到，应立即降落。

这里的 `500 m` 是面向“看到另一低空航空器/空域用户时”的避碰指导，不是两架协同无人机编队的固定安全间距。

官方来源：

- EASA Easy Access Rules for Unmanned Aircraft Systems, Revision from July 2024, online publication： https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-unmanned-aircraft-systems?page=5
- EASA FAQ, remote pilot responsibilities in open category： https://www.easa.europa.eu/en/faq/116468

## 9. 工程使用建议

如果这架 `5_in_drone` 只用于仿真论文或训练说明，建议把“官方安全距离”和“项目安全距离”分开写：

- 官方标准：写“无统一固定米数；按空管/法规/运行授权保持必要安全间隔，不得造成碰撞风险”。
- 项目标准：可另行定义训练用 `hard_dist`、`trigger_dist`、slot spacing、碰撞半径等，但必须标注为项目工程假设。
- 硬件级验证：若要映射真实 5 英寸竞速机，应先统一 URDF 质量 `0.5 kg`、控制参考质量 `0.6076 kg`、几何臂长 `0.124875 m` 与控制 `arm_length=0.035 m`。
