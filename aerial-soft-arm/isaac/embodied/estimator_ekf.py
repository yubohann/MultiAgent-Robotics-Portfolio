"""误差状态 EKF（15 维，纯 numpy），IMU + GNSS + 气压融合，估计 IMU 偏置。

接口同 estimator.StateEstimator：update_imu / update_gnss / update_baro /
correct_position / correct_attitude / out(omega)。
状态 x = [p, v, q, b_a, b_g]；误差取世界系约定（z 向上，quat wxyz body->world，
acc 为 body 系比力）。无磁强计时偏航不可观测，默认 freeze_yaw，偏航只由陀螺积分。
量测更新：GNSS 位置（NIS 软门控）、气压高度、重力方向（比力幅值/角速率/水平
加速度三重信任门控）。
"""
import numpy as np

import spec as S

I3 = np.eye(3)

class EkfEstimator:
    """15 维误差状态 EKF，接口与 StateEstimator 一致。"""

    def __init__(self, pos0, dt, acc_sigma=S.ACC_SIGMA, gyro_sigma=S.GYRO_SIGMA,
                 gnss_sigma=S.GNSS_SIGMA, baro_sigma=S.BARO_SIGMA,
                 acc_bias_rw=0.002, gyro_bias_rw=1e-4, acc_process=0.03,
                 use_gravity_update=True, freeze_yaw=True):
        self.dt = float(dt)

        self.pos = np.asarray(pos0, dtype=float).copy()
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        self.acc_sigma = float(acc_sigma)
        self.gyro_sigma = float(gyro_sigma)
        self.gnss_sigma = float(gnss_sigma)
        self.baro_sigma = float(baro_sigma)
        self.acc_bias_rw = float(acc_bias_rw)
        self.gyro_bias_rw = float(gyro_bias_rw)
        self.acc_process = float(acc_process)
        self.use_gravity_update = bool(use_gravity_update)
        self.freeze_yaw = bool(freeze_yaw)

        self.acc_bias_limit = 0.5
        self.gyro_bias_limit = 0.05
        self.gnss_gate = 30.0
        self.baro_gate = 9.0
        self.p_diag_min = 1e-12
        self.p_diag_max = 1e4

        self.p0_pos = 0.25
        self.p0_vel = 1.0
        self.p0_att = 0.01
        self.p0_ba = S.ACC_BIAS ** 2
        self.p0_bg = (0.006 ** 2)
        self.P = self._default_P()

        self.acc_gate_sigma = 0.4
        self.rate_gate = 2.0
        self.dyn_accel_gate = 0.08
        self.dyn_accel_tau = 0.05
        self.att_sigma = 0.08
        self.att_trust_floor = 0.02
        self.att_trust_min = 0.4
        self.att_gate = 30.0
        self.aw_lp = 0.0

        self.last_gnss_t = -1.0
        self.last_baro_t = -1.0
        self.last_nis_gnss = 0.0
        self.last_nis_baro = 0.0

    # 公共接口
    def update_imu(self, acc, gyro, dt=None):
        dt = self.dt if dt is None else float(dt)
        acc = np.asarray(acc, dtype=float)
        gyro = np.asarray(gyro, dtype=float)

        a_b = acc - self.acc_bias
        r_mat = self.quat_to_mat(self.quat)
        a_w = r_mat @ a_b - np.array([0.0, 0.0, S.G])
        self.pos = self.pos + self.vel * dt + 0.5 * a_w * dt * dt
        self.vel = self.vel + a_w * dt

        w = gyro - self.gyro_bias
        self.quat = self._quat_normalize(
            self.quat_mul(self.quat, self.quat_from_rotvec(w * dt)))

        f_w = r_mat @ a_b
        f_mat = np.eye(15)
        f_mat[0:3, 3:6] = I3 * dt
        f_mat[0:3, 6:9] = -0.5 * self.skew(f_w) * dt * dt
        f_mat[0:3, 9:12] = -0.5 * r_mat * dt * dt
        f_mat[3:6, 6:9] = -self.skew(f_w) * dt
        f_mat[3:6, 9:12] = -r_mat * dt
        f_mat[6:9, 12:15] = -r_mat * dt
        self.P = f_mat @ self.P @ f_mat.T + self._q_discrete(dt)
        self.P = self._sanitize_p(self.P)

        al = 1.0 - np.exp(-dt / self.dyn_accel_tau)
        self.aw_lp += al * (float(np.linalg.norm(a_w[:2])) - self.aw_lp)

        if self.use_gravity_update:
            self._gravity_update(acc, gyro)

    def predict(self, acc, gyro, dt=None):
        self.update_imu(acc, gyro, self.dt if dt is None else dt)

    def correct_attitude(self, acc):
        pass

    def correct_position(self, meas, dt_meas, alpha=None):
        self.update_gnss(meas, dt_meas)

    def update_gnss(self, meas, dt=None):
        meas = np.asarray(meas, dtype=float).ravel()
        n = min(meas.size, 3)
        if n < 2:
            return
        h = np.zeros((n, 15))
        h[:, 0:n] = np.eye(n)
        r_cov = (self.gnss_sigma ** 2) * np.eye(n)
        self._kf_update(meas[:n], self.pos[:n], h, r_cov, self.gnss_gate)
        self.last_gnss_t = 0.0 if dt is None else float(dt)

    def update_baro(self, z, dt=None):
        z = float(z)
        if not np.isfinite(z):
            return
        h = np.zeros((1, 15))
        h[0, 2] = 1.0
        self._kf_update(np.array([z]), np.array([self.pos[2]]), h,
                        np.array([[self.baro_sigma ** 2]]), self.baro_gate)
        self.last_baro_t = 0.0 if dt is None else float(dt)

    def out(self, omega):
        return {"pos": self.pos.copy(), "vel": self.vel.copy(),
                "quat": self.quat.copy(),
                "omega": np.asarray(omega, dtype=float).copy()}

    # 重力方向更新（门控）
    def att_trust(self, acc, gyro):
        a_b = np.asarray(acc, dtype=float) - self.acc_bias
        a_norm = float(np.linalg.norm(a_b)) + 1e-9
        g_mag = float(np.exp(-0.5 * ((a_norm - S.G) / self.acc_gate_sigma) ** 2))
        w_norm = float(np.linalg.norm(np.asarray(gyro, dtype=float) - self.gyro_bias))
        g_rate = 1.0 / (1.0 + (w_norm / self.rate_gate) ** 2)
        g_dyn = float(np.exp(-0.5 * (self.aw_lp / self.dyn_accel_gate) ** 2))
        return float(np.clip(g_mag * g_rate * g_dyn, self.att_trust_floor, 1.0))

    def _gravity_update(self, acc, gyro):
        a_b = np.asarray(acc, dtype=float) - self.acc_bias
        a_norm = float(np.linalg.norm(a_b)) + 1e-9
        trust = self.att_trust(acc, gyro)
        z = a_b / a_norm
        u = self.quat_rotate(self.quat_conj(self.quat),
                             np.array([0.0, 0.0, 1.0]))
        h = np.zeros((3, 15))
        h[:, 6:9] = self.quat_to_mat(self.quat).T @ self.skew(
            np.array([0.0, 0.0, 1.0]))
        if trust < self.att_trust_min:
            return
        r_cov = ((self.att_sigma / trust) ** 2) * I3
        r = z - u
        s_mat = h @ self.P @ h.T + r_cov
        nis = float(r @ np.linalg.solve(s_mat, r))
        if nis > self.att_gate:
            return
        self._kf_update(z, u, h, r_cov, None)

    # 通用卡尔曼更新
    def _kf_update(self, z, h_pred, h_mat, r_cov, gate):
        z = np.asarray(z, dtype=float).ravel()
        h_pred = np.asarray(h_pred, dtype=float).ravel()
        r = z - h_pred
        s_mat = h_mat @ self.P @ h_mat.T + r_cov
        if gate is not None:
            nis = float(r @ np.linalg.solve(s_mat, r))
            if nis > gate:
                r_cov = r_cov * (nis / gate)
                s_mat = h_mat @ self.P @ h_mat.T + r_cov
                if gate == self.gnss_gate:
                    self.last_nis_gnss = nis
                elif gate == self.baro_gate:
                    self.last_nis_baro = nis
        k = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        if self.freeze_yaw:
            k[8, :] = 0.0
            axis = self.quat_to_mat(self.quat).T @ np.array([0.0, 0.0, 1.0])
            k[12:15, :] -= np.outer(axis, axis @ k[12:15, :])
        dx = k @ r
        self._inject(dx)
        i_kh = np.eye(15) - k @ h_mat
        self.P = i_kh @ self.P @ i_kh.T + k @ r_cov @ k.T
        self.P = self._sanitize_p(self.P)

    def _inject(self, dx):
        self.pos = self.pos + dx[0:3]
        self.vel = self.vel + dx[3:6]
        self.acc_bias = np.clip(self.acc_bias + dx[9:12],
                                -self.acc_bias_limit, self.acc_bias_limit)
        self.gyro_bias = np.clip(self.gyro_bias + dx[12:15],
                                 -self.gyro_bias_limit, self.gyro_bias_limit)
        self.quat = self._quat_normalize(
            self.quat_mul(self.quat_from_rotvec(dx[6:9]), self.quat))

    # 噪声与数值工具
    def _q_discrete(self, dt):
        q = np.zeros((15, 15))
        sig_v = (self.acc_sigma ** 2 + self.acc_process ** 2) * dt * dt
        q[0:3, 0:3] = 0.25 * sig_v * dt * dt * I3
        q[0:3, 3:6] = 0.5 * sig_v * dt * I3
        q[3:6, 0:3] = 0.5 * sig_v * dt * I3
        q[3:6, 3:6] = sig_v * I3
        q[6:9, 6:9] = (self.gyro_sigma ** 2) * dt * dt * I3
        q[9:12, 9:12] = (self.acc_bias_rw ** 2) * dt * I3
        q[12:15, 12:15] = (self.gyro_bias_rw ** 2) * dt * I3
        return q

    def _default_P(self):
        return np.diag(np.concatenate([
            np.full(3, self.p0_pos), np.full(3, self.p0_vel),
            np.full(3, self.p0_att), np.full(3, self.p0_ba),
            np.full(3, self.p0_bg)]))

    def _sanitize_p(self, p):
        if not np.isfinite(p).all():
            return self._default_P()
        p = 0.5 * (p + p.T)
        d = np.diag(p).copy()
        d = np.clip(d, self.p_diag_min, self.p_diag_max)
        p[np.diag_indices(15)] = d
        return p

    @staticmethod
    def skew(v):
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    @staticmethod
    def _quat_normalize(q):
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n

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
        r = EkfEstimator.quat_mul(
            EkfEstimator.quat_mul(q, qv), EkfEstimator.quat_conj(q))
        return r[1:]

    @staticmethod
    def quat_from_rotvec(rv):
        ang = float(np.linalg.norm(rv))
        if ang < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rv / ang
        s = np.sin(ang / 2.0)
        return np.array([np.cos(ang / 2.0), axis[0] * s,
                         axis[1] * s, axis[2] * s])

    @staticmethod
    def quat_to_mat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
