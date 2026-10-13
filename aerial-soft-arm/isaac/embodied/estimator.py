"""传感器与状态估计，纯 numpy。

IMU 含噪声偏置与量化，GNSS 与气压计低速率采样保持，
姿态用互补滤波，位置速度用 alpha-beta 滤波跑 GNSS 与加速度积分。
"""
import numpy as np

import spec as S

MAHONY_GAIN = 0.9


class SensorSim:
    def __init__(self, rng, gnss_sigma=S.GNSS_SIGMA, baro_sigma=S.BARO_SIGMA):
        self.rng = rng
        self.gnss_sigma = gnss_sigma
        self.baro_sigma = baro_sigma
        self.gyro_bias = rng.normal(0.0, 0.004, 3)
        self.acc_bias = rng.normal(0.0, S.ACC_BIAS, 3)
        self.baro_drift = 0.0
        self._t_gnss = -1.0
        self._t_baro = -1.0
        self._gnss = np.zeros(3)
        self._baro = 0.0

    def imu(self, acc_true, gyro_true, dt):
        gyro = gyro_true + self.gyro_bias + self.rng.normal(0.0, S.GYRO_SIGMA, 3)
        acc = acc_true + self.acc_bias + self.rng.normal(0.0, S.ACC_SIGMA, 3)
        return acc, gyro

    def gnss(self, pos_true, t):
        if t - self._t_gnss >= 1.0 / S.GNSS_RATE:
            self._t_gnss = t
            self._gnss = pos_true + self.rng.normal(0.0, self.gnss_sigma, 3)
        return self._gnss.copy()

    def baro(self, z_true, t):
        if t - self._t_baro >= 1.0 / S.BARO_RATE:
            self._t_baro = t
            self.baro_drift += self.rng.normal(0.0, 0.002)
            self._baro = z_true + self.baro_drift + self.rng.normal(0.0, self.baro_sigma)
        return self._baro


class StateEstimator:
    """Mahony 姿态估计加 alpha 滤波位置，输入为带噪传感器。"""

    def __init__(self, pos0, dt):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos = np.array(pos0, dtype=float)
        self.vel = np.zeros(3)
        self.dt = dt
        self.gyro_bias = np.zeros(3)

    def update_imu(self, acc, gyro, dt):
        a = acc / (np.linalg.norm(acc) + 1e-9)
        u = self.quat_rotate(self.quat_conj(self.quat), np.array([0.0, 0.0, 1.0]))
        acc_norm = float(np.linalg.norm(acc))
        w_acc = 1.0 if abs(acc_norm - S.G) < 0.25 * S.G else 0.08
        e = np.cross(a, u) * w_acc
        self.gyro_bias += -0.03 * e * dt
        w = gyro - self.gyro_bias + MAHONY_GAIN * e
        dq = np.array([1.0, w[0] * dt / 2.0, w[1] * dt / 2.0, w[2] * dt / 2.0])
        qn = self.quat_mul(self.quat, dq)
        self.quat = qn / np.linalg.norm(qn)
        a_world = self.quat_rotate(self.quat, acc) - np.array([0.0, 0.0, S.G])
        self.vel = self.vel + a_world * dt
        self.pos = self.pos + self.vel * dt

    def predict(self, acc, gyro, dt=None):
        self.update_imu(acc, gyro, self.dt if dt is None else dt)

    def correct_attitude(self, acc):
        pass

    def correct_position(self, meas, dt_meas, alpha=0.45):
        d = np.asarray(meas, dtype=float) - self.pos
        self.pos = self.pos + alpha * d
        v_meas = d / max(dt_meas, 1e-3)
        self.vel = 0.85 * self.vel + 0.15 * v_meas

    def out(self, omega):
        return {"pos": self.pos.copy(), "vel": self.vel.copy(),
                "quat": self.quat.copy(), "omega": np.asarray(omega, dtype=float).copy()}

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
        r = StateEstimator.quat_mul(StateEstimator.quat_mul(q, qv), StateEstimator.quat_conj(q))
        return r[1:]

    @staticmethod
    def quat_from_rotvec(rv):
        ang = np.linalg.norm(rv)
        if ang < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rv / ang
        s = np.sin(ang / 2.0)
        return np.array([np.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])
