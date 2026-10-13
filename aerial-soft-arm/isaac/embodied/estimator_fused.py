"""融合状态估计器（纯 numpy）：Mahony 姿态 + 气压互补高度 + 水平 alpha-beta/Kalman。

接口同 estimator.StateEstimator：out(omega) -> {pos, vel, quat, omega}，
update_imu / correct_position / predict。
坐标：pos/vel 世界系 z 向上，quat wxyz body->world，acc 为 body 系比力。
姿态误差带三重信任门控（比力幅值、角速率、世界系水平加速度低通）；高度链为
临界阻尼 alpha-beta 互补加垂直加速度计偏置慢估计；水平增益由稳态 Riccati
推导并按 GNSS 残差自适应调度。
"""

import numpy as np

import spec as S

class FusionEstimator:
    """Mahony(三重自适应门控) + 高度互补/偏置估计 + 水平自适应 2 态 Kalman。"""

    def __init__(self, pos0, dt, gnss_sigma=S.GNSS_SIGMA, baro_sigma=S.BARO_SIGMA,
                 gnss_rate=S.GNSS_RATE, imu_vel_noise=0.0707):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos = np.array(pos0, dtype=float).copy()
        self.vel = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.dt = float(dt)

        self.kp_att = 0.6          # Mahony 比例增益 (rad/s per rad err)
        self.ki_att = 0.15         # 陀螺零偏积分增益 (1/s per rad err)
        self.gyro_bias_limit = 0.05
        self.acc_gate_sigma = 0.18 * S.G   # |acc|-g 高斯门控尺度
        self.rate_gate = 2.0               # rad/s, 角速率门控尺度
        self.att_trust_floor = 0.05        # 门控下限（保留慢收敛能力）
        self.dyn_accel_gate = 1.5          # m/s^2, 动态水平加速度门控尺度
        self.dyn_accel_tau = 0.5           # s, a_h 低通时间常数
        self.aw_lp = 0.0                   # |a_world_xy| 低通值

        self.baro_tau = 2.5        # 气压互补时间常数 (s)，规格 ~1.5，见 docstring
        self.baro_res_clamp = 2.0  # m
        self.baro_accel_trim = 0.0002   # b_z 慢估计增益 (每 baro 采样)
        self.accel_bias_limit = 0.2     # m/s^2, b_z 限幅
        self.accel_bias_z = 0.0         # 垂直加速度计偏置估计
        self.gnss_z_alpha = 0.15   # GNSS-z 低增益位置修正
        self.gnss_z_beta = 0.15 / (1.0 / gnss_rate) * 0.30   # v_z 修正
        self.gnss_z_res_clamp = 1.5

        self.gnss_sigma = float(gnss_sigma)
        self.baro_sigma = float(baro_sigma)
        self.gnss_rate = float(gnss_rate)
        self.imu_vel_noise = float(imu_vel_noise)
        self.gnss_res_clamp = 1.5  # m, 残差限幅（异常值保护）
        self.vel_limit = 10.0      # m/s, 反 windup
        # 稳态 Kalman 增益（每轴相同），暴露为类参数便于整定/测试
        self.gnss_h = 1.0 / self.gnss_rate
        self.gnss_q_lo = self.imu_vel_noise ** 2 / 1.0   # m^2/s^3
        self.gnss_q_hi = 0.3                             # 机动时过程噪声上界
        self.gnss_alpha_lo, self.gnss_beta_lo = self._steady_state_gains(
            self.gnss_q_lo, self.gnss_h, self.gnss_sigma ** 2)
        self.gnss_alpha_hi, self.gnss_beta_hi = self._steady_state_gains(
            self.gnss_q_hi, self.gnss_h, self.gnss_sigma ** 2)
        # 自适应调度参数
        self.gnss_innov_ref = 0.15     # m, 残差参考尺度
        self.gnss_innov_shape = 8.0    # 调度锐度
        self.gnss_innov_smooth = 0.10  # e_lp 一阶平滑 (每 GNSS 采样)
        self.gnss_pos_boost = 0.5      # 位置增益只释放一半，速度增益全释放
        self.e_lp = 0.0
        # 当前生效增益（初始为平稳档）
        self.gnss_alpha = self.gnss_alpha_lo
        self.gnss_beta = self.gnss_beta_lo

    # Riccati: 标量双积分器的稳态 Kalman 增益
    @staticmethod
    def _steady_state_gains(q_psd, h, r, n_iter=2000):
        """迭代离散 Riccati 方程求 x=[p,v] 双积分器的稳态增益 K=[alpha,beta]。

            P <- F P F^T + Q
            K  = P H^T / (H P H^T + r)
            P <- (I - K H) P
        F = [[1,h],[0,1]], H = [1,0], Q = q*[[h^3/3,h^2/2],[h^2/2,h]]。
        返回 alpha (位置增益), beta (速度增益, v += beta*r)。
        """
        f = np.array([[1.0, h], [0.0, 1.0]])
        q = q_psd * np.array([[h ** 3 / 3.0, h ** 2 / 2.0],
                              [h ** 2 / 2.0, h]])
        h_mat = np.array([[1.0, 0.0]])
        p = np.eye(2) * 1.0
        k = np.zeros(2)
        for _ in range(n_iter):
            p = f @ p @ f.T + q
            s = float((h_mat @ p @ h_mat.T).ravel()[0]) + r
            k = (p @ h_mat.T / s).ravel()
            p = p - np.outer(k, h_mat @ p)
        return float(k[0]), float(k[1])

    # 姿态：Mahony adaptive
    def att_trust(self, acc, gyro):
        """自适应加速度信任度 in [trust_floor, 1]。

        g_a: | |acc|-g | 高斯衰减；g_w: 角速率（去偏后）平方衰减；
        g_d: 低通水平加速度平方衰减（持续加速度使比力方向≠重力方向）。
        """
        a_norm = float(np.linalg.norm(acc)) + 1e-9
        mag_err = abs(a_norm - S.G)
        g_a = np.exp(-0.5 * (mag_err / self.acc_gate_sigma) ** 2)
        w_norm = float(np.linalg.norm(np.asarray(gyro, dtype=float) - self.gyro_bias))
        g_w = 1.0 / (1.0 + (w_norm / self.rate_gate) ** 2)
        g_d = np.exp(-0.5 * (self.aw_lp / self.dyn_accel_gate) ** 2)
        return float(np.clip(g_a * g_w * g_d, self.att_trust_floor, 1.0))

    def update_imu(self, acc, gyro, dt=None):
        """1 kHz IMU 步：自适应 Mahony 姿态 + 世界系加速度积分。"""
        dt = self.dt if dt is None else float(dt)
        acc = np.asarray(acc, dtype=float)
        gyro = np.asarray(gyro, dtype=float)

        # Mahony，姿态用原始比力方向
        a_norm = float(np.linalg.norm(acc)) + 1e-9
        a_hat = acc / a_norm
        u = self.quat_rotate(self.quat_conj(self.quat),
                             np.array([0.0, 0.0, 1.0]))
        trust = self.att_trust(acc, gyro)
        e = np.cross(a_hat, u) * trust
        # 陀螺零偏积分（trust 缩放防 windup）+ 限幅
        self.gyro_bias = np.clip(self.gyro_bias - self.ki_att * e * dt,
                                 -self.gyro_bias_limit, self.gyro_bias_limit)
        w = gyro - self.gyro_bias + self.kp_att * e
        dq = np.array([1.0, w[0] * dt / 2.0, w[1] * dt / 2.0, w[2] * dt / 2.0])
        qn = self.quat_mul(self.quat, dq)
        self.quat = qn / (np.linalg.norm(qn) + 1e-12)

        # 世界系加速度积分，减去垂直偏置
        acc_c = acc.copy()
        acc_c[2] -= self.accel_bias_z
        a_world = self.quat_rotate(self.quat, acc_c) - np.array([0.0, 0.0, S.G])
        self.vel = self.vel + a_world * dt
        self.vel = self._clamp_vel(self.vel)
        self.pos = self.pos + self.vel * dt

        # 动态加速度门控的低通量（用于下一步姿态信任）
        a_h = float(np.linalg.norm(a_world[:2]))
        al = 1.0 - np.exp(-dt / self.dyn_accel_tau)
        self.aw_lp += al * (a_h - self.aw_lp)

    # 高度：气压互补 + GNSS-z 低增益 + 垂直加速度计偏置慢估计
    def update_baro(self, z, dt):
        """气压高度修正：临界阻尼 alpha-beta，等效一阶互补 tau=baro_tau。"""
        z = float(z)
        if not np.isfinite(z):
            return
        h = float(np.clip(dt, 1e-3, 0.2))
        lam = float(np.exp(-h / self.baro_tau))
        a_b = 1.0 - lam * lam
        b_b = (1.0 - lam) ** 2 / h
        r = float(np.clip(z - self.pos[2], -self.baro_res_clamp,
                          self.baro_res_clamp))
        self.pos[2] += a_b * r
        self.vel[2] += b_b * r
        self.vel = self._clamp_vel(self.vel)
        # 加速度计 z 偏置慢估计：残差为正（baro 高于估计）说明估计偏低
        if self.baro_accel_trim > 0.0:
            self.accel_bias_z = float(np.clip(
                self.accel_bias_z - self.baro_accel_trim * r,
                -self.accel_bias_limit, self.accel_bias_limit))

    # 水平：2 态 Kalman(alpha-beta) 自适应增益，2D/3D 输入皆可
    def update_gnss(self, meas, dt):
        """GNSS 修正：pos 与 vel 同步推进。

        meas: 长度 2 只修 xy；长度 3 时 z 轴走低增益通道（baro 管快漂）。
        dt:   距上次 GNSS 的间隔（用于 z 通道归一，可不用）。
        """
        meas = np.asarray(meas, dtype=float).ravel()
        # 自适应增益调度：残差越大越信任 GNSS
        e = float(np.linalg.norm(meas[:2] - self.pos[:2]))
        self.e_lp += self.gnss_innov_smooth * (e - self.e_lp)
        s = min((self.e_lp / self.gnss_innov_ref) ** self.gnss_innov_shape, 1.0)
        self.gnss_alpha = self.gnss_alpha_lo + self.gnss_pos_boost * s * (
            self.gnss_alpha_hi - self.gnss_alpha_lo)
        self.gnss_beta = self.gnss_beta_lo + s * (
            self.gnss_beta_hi - self.gnss_beta_lo)
        r_xy = np.clip(meas[:2] - self.pos[:2], -self.gnss_res_clamp,
                       self.gnss_res_clamp)
        self.pos[:2] += self.gnss_alpha * r_xy
        self.vel[:2] += self.gnss_beta * r_xy
        if meas.size >= 3:
            r_z = float(np.clip(meas[2] - self.pos[2], -self.gnss_z_res_clamp,
                                self.gnss_z_res_clamp))
            self.pos[2] += self.gnss_z_alpha * r_z
            self.vel[2] += self.gnss_z_beta * r_z
        self.vel = self._clamp_vel(self.vel)

    # 与 StateEstimator 同名接口
    def predict(self, acc, gyro, dt=None):
        self.update_imu(acc, gyro, self.dt if dt is None else dt)

    def correct_attitude(self, acc):
        pass

    def correct_position(self, meas, dt_meas, alpha=None):
        """等价于 update_gnss(meas, dt_meas)。

        alpha 未使用，增益由稳态 Kalman 给出。
        """
        self.update_gnss(meas, dt_meas)

    def out(self, omega):
        return {"pos": self.pos.copy(), "vel": self.vel.copy(),
                "quat": self.quat.copy(),
                "omega": np.asarray(omega, dtype=float).copy()}

    def _clamp_vel(self, v):
        n = float(np.linalg.norm(v))
        if n > self.vel_limit:
            return v * (self.vel_limit / n)
        return v

    @staticmethod
    def quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    @staticmethod
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_rotate(q, v):
        qv = np.array([0.0, v[0], v[1], v[2]])
        r = FusionEstimator.quat_mul(
            FusionEstimator.quat_mul(q, qv), FusionEstimator.quat_conj(q))
        return r[1:]

    @staticmethod
    def quat_from_rotvec(rv):
        ang = np.linalg.norm(rv)
        if ang < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rv / ang
        s = np.sin(ang / 2.0)
        return np.array([np.cos(ang / 2.0), axis[0] * s,
                         axis[1] * s, axis[2] * s])
