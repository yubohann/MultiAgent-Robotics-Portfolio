"""动力与飞控，纯 numpy。

四电机一阶转速响应与平方推力，X 型混控，6S 电池含内阻压降与容量消耗，
串级飞控，位置速度姿态角速度，输出总推与三轴力矩。
"""
import numpy as np

import spec as S


class ThrusterBank:
    """四电机一阶模型，推力与反扭矩。"""

    def __init__(self, n=4):
        self.n = n
        self.omega = np.zeros(n)
        ang = np.radians(np.array([45.0, 135.0, 225.0, 315.0]))
        self.xy = np.stack([S.ROTOR_XY * np.cos(ang), S.ROTOR_XY * np.sin(ang)], axis=1)
        self.dir = np.asarray(S.ARM_DIRECTIONS, dtype=float)

    def step(self, omega_cmd, dt, v_scale=1.0):
        w_cmd = np.clip(omega_cmd, 0.0, S.OMEGA_MAX * v_scale)
        err = w_cmd - self.omega
        tau = np.where(err > 0, S.TAU_INC, S.TAU_DEC)
        self.omega = self.omega + err * (1.0 - np.exp(-dt / tau))
        thrust = S.K_F * self.omega ** 2
        torque_z = -self.dir * S.CT_TORQUE_RATIO * thrust
        return thrust, torque_z

    def wrench(self, thrust, torque_z):
        fz = float(thrust.sum())
        torques = np.zeros(3)
        torques[0] = float((self.xy[:, 1] * thrust).sum())
        torques[1] = float((-self.xy[:, 0] * thrust).sum())
        torques[2] = float(torque_z.sum())
        return fz, torques

    def mixer(self, fz, m):
        """期望总推与力矩到四电机推力，pseudo-inverse 常数矩阵。"""
        a = np.zeros((4, 4))
        a[0, :] = 1.0
        a[1, :] = self.xy[:, 1]
        a[2, :] = -self.xy[:, 0]
        a[3, :] = -self.dir * S.CT_TORQUE_RATIO
        t = np.linalg.solve(a, np.array([fz, m[0], m[1], m[2]]))
        return np.clip(t, 0.0, S.K_F * S.OMEGA_MAX ** 2)


class BatteryPack:
    """6S 锂聚合物，开路电压分段线性，内阻压降，容量消耗。"""

    def __init__(self):
        self.ah_left = S.BATT_AH
        self.voltage = S.V_FULL
        self.current = 0.0

    def ocv(self):
        soc = np.clip(self.ah_left / S.BATT_AH, 0.0, 1.0)
        pts = np.array([[0.0, S.V_EMPTY], [0.15, 21.4], [0.5, 22.6], [0.85, 24.2], [1.0, S.V_FULL]])
        return float(np.interp(soc, pts[:, 0], pts[:, 1]))

    def step(self, total_thrust, dt):
        self.current = S.AVIONICS_A + S.THRUST_A_PER_N * total_thrust
        self.voltage = self.ocv() - self.current * S.R_INT
        self.ah_left = max(self.ah_left - self.current * dt / 3600.0, 0.0)
        return self.voltage

    def low(self):
        return self.voltage < 20.4

    def v_scale(self):
        return float(np.clip(self.voltage / S.V_NOM, 0.75, 1.05))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_from_rotvec(rv):
    ang = np.linalg.norm(rv)
    if ang < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / ang
    s = np.sin(ang / 2.0)
    return np.array([np.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


class FlightController:
    """串级闭环，位置速度环输出推力矢量与姿态目标，姿态与角速度环输出力矩。"""

    def __init__(self, mass, inertia=None):
        self.mass = mass
        self.inertia = np.asarray(inertia if inertia is not None else [0.012, 0.012, 0.020], dtype=float)
        self.ki_state = np.zeros(3)
        self.last_mode_att = True
        self.extra_moment = np.zeros(3)
        self.att_trim = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.L_DOB = 0.0
        self.DOB_TAU_FILT = 0.02
        self.DOB_D_MAX = 0.35
        self.DOB_HP_TAU = 0.0
        self._omega_f = None
        self._z_dob = np.zeros(3)
        self._d_slow = np.zeros(3)
        self._d_comp = np.zeros(3)
        self._moments_prev = np.zeros(3)

    def reset(self):
        self.ki_state[:] = 0.0
        self.att_trim[:] = 0.0
        self.d_hat[:] = 0.0
        self._omega_f = None
        self._z_dob[:] = 0.0
        self._d_slow[:] = 0.0
        self._d_comp[:] = 0.0
        self._moments_prev[:] = 0.0

    def compute(self, est, cmd_pos, cmd_yaw, dt, mass=None, moments_actual=None):
        m = mass if mass is not None else self.mass
        pos, vel = est["pos"], est["vel"]
        q = est["quat"]
        omega = est["omega"]

        e_pos = cmd_pos - pos
        v_des = np.clip(S.KP_POS * e_pos, -2.5, 2.5)
        e_vel = v_des - vel
        self.ki_state = np.clip(self.ki_state + e_vel * dt, -1.0, 1.0)
        a_des = S.KP_VEL * e_vel + S.KI_VEL * self.ki_state
        a_des[2] += S.G
        a_des = np.clip(a_des, -16.0, 16.0)
        self.a_des_last = a_des.copy()
        self.v_des_last = v_des.copy()

        f_des = m * a_des
        fz = float(np.linalg.norm(f_des))
        fz = float(np.clip(fz, 0.05 * m * S.G, S.MAX_TOTAL_THRUST))
        z_body = f_des / (np.linalg.norm(f_des) + 1e-9)
        yaw_axis = np.array([np.cos(cmd_yaw), np.sin(cmd_yaw), 0.0])
        x_body = yaw_axis - np.dot(yaw_axis, z_body) * z_body
        x_body = x_body / (np.linalg.norm(x_body) + 1e-9)
        y_body = np.cross(z_body, x_body)
        r_mat = np.stack([x_body, y_body, z_body], axis=1)

        tr = float(np.trace(r_mat))
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            q_des = np.array([0.25 * s,
                              (r_mat[2, 1] - r_mat[1, 2]) / s,
                              (r_mat[0, 2] - r_mat[2, 0]) / s,
                              (r_mat[1, 0] - r_mat[0, 1]) / s])
        else:
            i = int(np.argmax(np.diag(r_mat)))
            if i == 0:
                s = np.sqrt(1.0 + r_mat[0, 0] - r_mat[1, 1] - r_mat[2, 2]) * 2.0
                q_des = np.array([(r_mat[2, 1] - r_mat[1, 2]) / s, 0.25 * s,
                                  (r_mat[0, 1] + r_mat[1, 0]) / s, (r_mat[0, 2] + r_mat[2, 0]) / s])
            elif i == 1:
                s = np.sqrt(1.0 + r_mat[1, 1] - r_mat[0, 0] - r_mat[2, 2]) * 2.0
                q_des = np.array([(r_mat[0, 2] - r_mat[2, 0]) / s,
                                  (r_mat[0, 1] + r_mat[1, 0]) / s, 0.25 * s,
                                  (r_mat[1, 2] + r_mat[2, 1]) / s])
            else:
                s = np.sqrt(1.0 + r_mat[2, 2] - r_mat[0, 0] - r_mat[1, 1]) * 2.0
                q_des = np.array([(r_mat[1, 0] - r_mat[0, 1]) / s,
                                  (r_mat[0, 2] + r_mat[2, 0]) / s,
                                  (r_mat[1, 2] + r_mat[2, 1]) / s, 0.25 * s])

        q_err = quat_mul(quat_conj(q), q_des)
        if q_err[0] < 0:
            q_err = -q_err
        rv = 2.0 * q_err[1:]
        self.rv_last = rv.copy()
        self.att_trim = np.clip(self.att_trim + S.KI_ATT_M * rv * dt, -S.ATT_TRIM_M_MAX, S.ATT_TRIM_M_MAX)
        omega_des = S.KP_ATT * rv
        omega_des = np.clip(omega_des, -6.0, 6.0)
        alpha = S.KP_RATE * (omega_des - omega) - S.KD_RATE * omega
        if self.L_DOB > 0.0:
            L = self.L_DOB
            om = np.asarray(omega, dtype=float)
            if self._omega_f is None:
                self._omega_f = om.copy()
            if self.DOB_TAU_FILT > 0.0:
                a_f = 1.0 - np.exp(-max(dt, 1e-6) / float(self.DOB_TAU_FILT))
            else:
                a_f = 1.0
            self._omega_f = self._omega_f + (om - self._omega_f) * a_f
            u = self._moments_prev if moments_actual is None else np.asarray(moments_actual, dtype=float)
            lw = L * self.inertia * self._omega_f
            self._z_dob = self._z_dob + dt * (-L * (self._z_dob + u + lw))
            self.d_hat = self._z_dob + lw
            d_comp = self.d_hat
            if self.DOB_HP_TAU > 0.0:
                a_s = 1.0 - np.exp(-max(dt, 1e-6) / self.DOB_HP_TAU)
                self._d_slow = self._d_slow + (self.d_hat - self._d_slow) * a_s
                d_comp = self.d_hat - self._d_slow
            self._d_comp = np.clip(d_comp, -self.DOB_D_MAX, self.DOB_D_MAX)
        else:
            self.d_hat[:] = 0.0
            self._d_comp[:] = 0.0
        moments = self.inertia * alpha + self.extra_moment + self.att_trim - self._d_comp
        self._moments_prev = moments.copy()
        return fz, moments


def moment_from_thrust_sim(fz_cmd, moments_cmd, omega_now):
    """简化直通，检查混控与限幅逻辑。"""
    return fz_cmd, moments_cmd
