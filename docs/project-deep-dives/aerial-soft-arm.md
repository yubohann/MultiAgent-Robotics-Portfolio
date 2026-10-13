# AerialSoftArm：项目技术说明

## 300 字项目概括

AerialSoftArm 是面向真机的刚柔耦合空中机械臂项目：四旋翼搭载 8 自由度桁架软臂（双轴弯曲、伸缩脊柱、三腱索耦合），在 Isaac Lab 中完成视觉锁定、基座粗定位后冻结、伸臂、弯-雅可比视觉伺服对位、双指持续接触确认、捕获与提升的完整自主抓取，并包含双机托盘交接。核心主张是"基座冻结、弯曲接管最后几厘米"：刚性段承载保形，软段对位与柔性捕获，配合在线复合惯量自适应、弯曲反力矩前馈和 15 维误差状态 EKF（逐传感器 NIS 门控）。README 记录的结果为 2 cm 以内最终对位误差与 73% 的速度 RMS 降低。另有 MuJoCo 侧真机桁架臂模型（网格→XML）、键盘操作面板与 sim2real 标定导出。Isaac 结果属于仿真证据；硬件证据以真机臂弯曲、夹持与标定链条为限，关键数字需绑定 run、配置与 seed 复核。

## 项目定位

- **抽象**：无降落基面场景下的悬停空中抓取，以及双机中继交接。
- **任务流程**：视觉锁定 → 基座粗定位并冻结 → 伸臂 → 弯曲对位 → 双指捕获（持续接触 0.3 s）→ 提升 / 交接。
- **主要方法**：刚柔双结构机械臂、弯-雅可比视觉伺服、在线复合惯量自适应、弯曲反力矩前馈、接触确认与刚度调度。
- **依赖分层**：Isaac Sim 5.1 + Isaac Lab 2.3 + RTX GPU（主实验）；MuJoCo + 纯 numpy 估计/控制（真机模型、面板与无 GPU 路径）。

## 技术架构

```text
wrist camera (~2 mm noise floor)
      -> vision lock (part localisation)
base: coarse pose -> frozen hold
      attitude loop with online composite inertia (recomputed every 0.5 s)
      -> spine extension (8-DOF truss arm, telescoping)
      -> tendon model (3 tendons -> 16 joints, preload / Coulomb / hysteresis)
      -> bend Jacobian visual servo (bx/by) -> deadband trigger
      -> two-finger grip, sustained-contact check (0.3 s) -> lift
estimators: Mahony / fused / 15-state ESKF (per-sensor NIS gating)
      -> pure-numpy flight controller -> Isaac Lab actuators
MuJoCo mirror: arm_real model (meshes -> XML), panel teleop, sim2real export
```

模型的柔性来自等效腱索耦合而不是完整软体 FEM：`tendon_model.py` 把三索张力反解成 16 个弯曲关节的力矩，叠加预紧、库仑摩擦与气路迟滞；参数来源与 MuJoCo 真机臂一致（`arm_real/arm_lattice.py`），两端共享同一套运动学与力臂常数。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `isaac/fc_align.py` | 主编排：停稳伸臂、视觉测偏差、弯曲伺服、死区触发、接触确认、提升；含 `--rigid`（刚臂对照）、`--no-ff`、`--no-contact` 消融开关 |
| `isaac/embodied/` | 纯 numpy 飞行控制与估计：`flight.py`、`estimator.py`（Mahony）、`estimator_fused.py`、`estimator_ekf.py`（15 维误差状态，GNSS/气压/重力更新）、`links.py`、`spec.py` |
| `isaac/tendon_model.py` | 腱索-关节耦合等效模型：张力→关节力矩、三索反解、预紧、库仑摩擦、迟滞 |
| `isaac/calib_jac.py` | 零重力步进标定，输出弯曲到指尖/质心的雅可比（`bend_jac.json`、`bend_jac_tendon.json`） |
| `isaac/run_grasp.py`、`run_hover.py`、`run_matrix.py` | 四夹爪套件、悬停基线与实验矩阵串行调度（B/C/D 组重复运行并汇总判据） |
| `isaac/urdf/`、`isaac/usd/` | 四套夹爪（diamond / flat / soft / three）与双机（drone_a / drone_b）资产，含转换配置与资产哈希 |
| `arm_real/` | 真机桁架臂 MuJoCo 重建：`arm_lattice.py`（含腱索走线）、`make_meshes.py`、`arm_lattice_bom.py`、`arm_real.py`（气动/驱动参数） |
| `panel/` | 键盘操作面板：`control.py`、`drivers.py`、`sim.py`、`sim2real.py`（标定导出与命令流）、`tui.py`、`log.py` |
| `grasp_pipelines/` | Soft-Spiral 抓取三条路径（manual / scripted / vision，腕部相机颜色检测+闭环对位）；`spiral_repo/` 为第三方子树 |
| `arm_demo/` | MuJoCo 软臂与空中抓取演示（`aerial_grasp.py`、`soft_arm_demo.py`、`soft_arm_v2.py`） |
| `docs/` | 论文初稿（`paper_full_en.docx`）与技术报告（`tech_report_en.docx`） |
| `env/` | Python 3.11 锁定依赖清单（`requirements.txt`）与版本记录 |
| `media/` | README 演示 GIF 与图件（对位误差、逐次判据、系统总览等） |

## 开发过程

1. 先在 MuJoCo 中按真机结构重建桁架臂（反棱柱桁架链、5 倍伸缩、三腱索走线），让几何、腱索力臂与气动参数有唯一来源。
2. 把同样的运动学与腱索模型搬进 Isaac Lab（URDF→USD），建立双机与四夹爪资产族，保证两条路径共享同一套常数。
3. 实现单机对位抓取状态机（`fc_align.py`）并迭代到 v7 参数快照：停稳→伸臂→弯曲伺服→死区→接触确认→提升。
4. 补上消融开关与实验矩阵（刚臂对照、无反力矩前馈、无接触确认、基座-only），用 `run_matrix.py` 串行重复并汇总判据。
5. 建立零重力步进标定（`calib_jac.py`），把弯曲雅可比固化成 JSON，供视觉伺服与真机端共用。
6. 开发键盘操作面板与 `sim2real.py` 标定/命令流导出，打通"仿真参数→真机控制器"的交付链路。
7. 在 MuJoCo 侧实现视觉伺服抓取流水线（腕部相机、颜色检测、脉冲对位、保压与夹持确认）作为独立验证路径。

## 涉及知识点

- 空中操作耦合动力学：基座—臂反力矩耦合、复合惯量随伸展变化、冻结基座协议
- 误差状态 EKF：15 维状态 [p, v, q, b_a, b_g]、NIS 软门控、重力方向三重信任门控、无磁偏航冻结
- 互补滤波与融合估计对照（速度 RMS 对比）
- 腱索驱动建模：张力反解、rank-2 子空间投影、预紧、库仑摩擦、气路迟滞
- 视觉伺服：腕部相机噪声底、死区触发、弯曲雅可比驱动
- 接触确认与刚度调度：双指持续接触 0.3 s、堵转确认
- 仿真资产工程：URDF/USD 双份资产、资产哈希、MuJoCo 网格流水线
- sim2real 交付：结构/气动/腱索/夹爪标定导出、命令流
- 实验方法：消融开关、实验矩阵、seed 固定、判据行汇总

## 测试与验证

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$py = "<env_isaaclab 中的 python>"

& $py isaac\fc_align.py --headless --enable_cameras --out runs/fc_align_v7 `
  --eye 0.38,-0.45,0.18 --seed 5 --kp-vel 2.5 --kp-pos 0.4 --ki-att 0.10 --est-g 0.35
& $py isaac\run_grasp.py --gripper diamond --headless --out runs/grasp_diamond
& $py isaac\calib_jac.py --headless
& $py isaac\run_matrix.py

python arm_demo\aerial_grasp.py
python grasp_pipelines\spiral_grasp_scripted.py
python -m panel
```

`fc_align.py` 每次运行在日志末尾输出 PASS 判据行；`run_matrix.py` 把配置组重复运行后汇总到 `matrix_results.csv`。复现任何 README 中的数字（2 cm 对位、73% 速度 RMS、0.3 s 接触窗口）都应按 run、seed、配置与资产哈希成组复核；`runs/` 不入库，需要重新执行或回到本地存储取证。MuJoCo 路径可在无 GPU 机器上运行，用于结构、腱索与视觉流水线的独立检查。

## 学术申请中的项目价值

项目覆盖了 sim-to-real 的完整链路：真机结构重建 → 仿真资产 → 控制与估计 → 标定 → 真机交付文件，而且每个环节都有可读的代码与参数出处。相比只提交一条训练曲线，这里能支撑的研究讨论包括：为什么冻结基座、弯-雅可比对位的收敛性、腱索摩擦/迟滞对末端精度的影响、以及状态估计门控设计。它也适合作为"空中操作 + 具身智能"方向的面试与文书素材：问题定义（三个难题耦合）、方法主张（刚柔分工）、证据分层（仿真/硬件/标定）都清晰可讲。

## 运行入口

| 入口 | 文件 | 说明 |
| --- | --- | --- |
| 主实验 | `isaac/fc_align.py` | 单机对位抓取，约 25 分钟 headless |
| 抓取套件 | `isaac/run_grasp.py` | `--gripper diamond/flat/soft/three` 四夹爪矩阵 |
| 标定 | `isaac/calib_jac.py` | 零重力步进，输出弯曲/质心雅可比 |
| 实验调度 | `isaac/run_matrix.py` | 消融组串行重复与判据汇总 |
| MuJoCo 演示 | `arm_demo/aerial_grasp.py`、`grasp_pipelines/spiral_grasp_*.py` | 软臂抓取三条路径 |
| 真机面板 | `python -m panel` | 键盘遥操作、调参与 sim2real 导出 |

## 证据边界

- **仿真证据**：Isaac Lab / PhysX 下的抓取、交接与消融结果；不能直接外推为真实飞行安全结论。
- **硬件证据**：`media/demo_lattice_arm.gif` 展示真机桁架臂的弯曲与夹持；`panel/sim2real.py` 的标定导出说明交付链条，但本快照不包含真实飞行抓取录像与飞行日志。
- **第三方子树**：`grasp_pipelines/spiral_repo/` 为 vendored Soft-Spiral 代码，保留自身 MIT 许可证，见 `THIRD_PARTY_NOTICES.md`。
- **结果复核**：README 中的性能数字绑定于特定 run 与配置快照；`runs/` 与真机原始数据留在本地存储。

## 术语表

- **冻结基座（frozen base）**：粗定位完成后基座保持位姿，仅姿态回路工作，把末端自由度让给软臂。
- **弯-雅可比（bend Jacobian）**：弯曲指令到指尖位姿/质心的灵敏度矩阵，零重力标定产出。
- **复合惯量（composite inertia）**：含伸展臂在内的整机等效惯量，随臂长在线变化。
- **腱索耦合（tendon coupling）**：三根腱索张力经力臂投影到 16 个弯曲关节的等效映射。
- **接触确认（contact check）**：双指持续接触 0.3 s 才判定捕获，避免单点触碰即 attach。
- **NIS 门控**：归一化新息平方检验，用于抑制异常量测进入 EKF。
