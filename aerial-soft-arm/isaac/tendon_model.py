"""腱索-关节耦合等效模型：张力、预紧、摩擦进回路。

    tau_j = T_k x r                              （张力 -> 关节力矩，r 为力臂）
    T_k   = clip(T0 + (2/(3r)) u_k.m, 0, T_MAX) （力矩 -> 三索张力反解）

m 取 16 个弯曲关节力矩指令均值，投影到 rank-2 腱索子空间；库仑摩擦幅值随张力
缩放，迟滞映射气路 0.045 bar 滞回。参数来源 arm_real/arm_lattice.py，运行时经
set_joint_effort_target 注入。
"""
import numpy as np

N_SEG = 8
TENDON_ANGLES = (90.0, 210.0, 330.0)     # arm_lattice.py:32 / arm_params.py:12
R_TENDON = 0.028 * 1.12                  # 0.03136 m 臂环半径 x 1.12（arm_lattice.py:26,199）
T_MAX = 22.0                             # N 腱驱肌肉张力上限（arm_lattice.py:29）
K_PASSIVE = 1.10                         # Nm/rad 软态关节被动刚度（arm_lattice.py:30）
C_PASSIVE = 0.10                         # Nms/rad 软态关节阻尼（arm_lattice.py:30）
PRELOAD = 3.0                            # N 等效预紧（三索等张力零空间基）
MU_FRIC = 0.20                           # 库仑摩擦系数（导向/护套集总；MuJoCo 未建模项）
ETA_HYST = 0.045 / 1.2                   # 3.75% 气动迟滞比例（arm_real.py:35 P_STEP=1.2,300）
TAU_HYST = 0.4                           # s 迟滞一阶时间常数（arm_real.py:300）
V_EPS = 0.02                             # rad/s 库仑摩擦平滑尺度


class TendonModel:
    """三索张力 -> 16 关节等效力矩，含预紧、库仑摩擦与滞回。"""

    def __init__(self, kp=10.0, kd=4.0, preload=PRELOAD, mu=MU_FRIC,
                 hyst=1.0, friction=1.0, n_seg=N_SEG):
        self.kp = float(kp)
        self.kd = float(kd)
        self.preload = float(preload)
        self.mu = float(mu)
        self.hyst = float(hyst)
        self.friction = float(friction)
        self.n_seg = int(n_seg)
        self.U = np.array([[-np.sin(np.radians(a)), np.cos(np.radians(a))]
                           for a in TENDON_ANGLES])          # (3,2)
        self.reset()

    def reset(self):
        self.T = np.full(len(self.U), self.preload)
        self.m_cmd = np.zeros(2)
        self.m_tendon = np.zeros(2)
        self.h = np.zeros(2)
        self.dir = np.zeros(2)
        self.tau_f = np.zeros(2 * self.n_seg)

    def tensions(self, m):
        """由每关节等效力矩指令 m=(mx,my) 反解三索张力（最小范数 + 预紧 + 限幅）。"""
        T = self.preload + (2.0 / (3.0 * R_TENDON)) * (self.U @ np.asarray(m, float))
        return np.clip(T, 0.0, T_MAX)

    def torques(self, q, qd, q_target, dt=0.001):
        """按伺服等效律生成力矩指令 m，经张力反解/限幅/摩擦/滞回输出 16 关节力矩。

        q, qd, q_target: (2*n_seg,) 前 n_seg 为 bx、后 n_seg 为 by。
        返回 (tau16, info)；tau16 供 set_joint_effort_target 注入。
        """
        q = np.asarray(q, float).reshape(-1)
        qd = np.asarray(qd, float).reshape(-1)
        q_target = np.asarray(q_target, float).reshape(-1)
        m = self.kp * (q_target - q) + self.kd * (0.0 - qd)      # 伺服等效关节力矩指令
        m_bar = np.array([m[:self.n_seg].mean(), m[self.n_seg:2 * self.n_seg].mean()])

        # 力矩 -> 张力（预紧为零空间基，clip 进入物理上下限）
        self.T = self.tensions(m_bar)
        m_tend = R_TENDON * (self.U.T @ self.T)                  # 限幅后实际腱索力矩

        # 迟滞：方向锁存 + 一阶滞后（映射 Pneumatic.hyst）
        dm = m_bar - self.m_cmd
        if np.abs(dm).max() > 1e-5:
            self.dir = np.where(np.abs(dm) > 1e-5, np.sign(dm), self.dir)
        if self.hyst > 0.0:
            self.h += dt / TAU_HYST * (
                ETA_HYST * T_MAX * R_TENDON * self.dir * self.hyst - self.h)
        else:
            self.h[:] = 0.0

        # 库仑摩擦（关节级，幅值随张力均值缩放）
        T_bar = float(self.T.mean())
        tau_f = self.mu * self.friction * T_bar * R_TENDON * np.tanh(qd / V_EPS)

        self.m_cmd = m_bar
        self.m_tendon = m_tend
        self.tau_f = tau_f
        tau = np.empty(2 * self.n_seg)
        tau[:self.n_seg] = m_tend[0] - self.h[0]
        tau[self.n_seg:] = m_tend[1] - self.h[1]
        tau = tau - tau_f
        info = {"T": self.T.copy(), "m_cmd": m_bar.copy(), "m_tendon": m_tend.copy(),
                "hyst": self.h.copy(), "T_bar": T_bar, "dir": self.dir.copy()}
        return tau, info

    def info(self):
        return {"T": self.T.copy(), "m_cmd": self.m_cmd.copy(),
                "m_tendon": self.m_tendon.copy(), "hyst": self.h.copy(),
                "T_bar": float(self.T.mean())}
