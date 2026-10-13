"""具身双机 Isaac 集成层，飞行控制闭环、电池、链路、臂驱动。"""
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

import arm_params as P
import spec as S
from flight import ThrusterBank, BatteryPack, FlightController, quat_rotate, quat_conj, quat_mul
from tendon_model import TendonModel
from estimator import SensorSim, StateEstimator
from estimator_fused import FusionEstimator
from links import SimpleLink
from arm_scene import SLIDE_JOINTS, BEND_JOINTS

HERE = Path(__file__).resolve().parent
USD_DIR = HERE / "usd"


def spawn_ground_and_stand():
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=1.8)
    post = sim_utils.CylinderCfg(radius=0.012, height=0.78,
                                 collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat)
    post.func("/World/stand/post", post, translation=(0.0, 0.0, 0.74 + 0.39))
    plate = sim_utils.CylinderCfg(radius=0.050, height=0.016,
                                  collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat)
    plate.func("/World/stand/plate", plate, translation=(0.0, 0.0, 1.525))
    pad = sim_utils.CuboidCfg(size=(0.60, 0.60, 0.40),
                              collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat)
    pad.func("/World/pad", pad, translation=(0.6, 0.9, 0.20))
    light = sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.92, 0.95))
    light.func("/World/Light", light)


PAD_TOP = 0.40
PAD_XY = (0.6, 0.9)
ARM_CLEAR_Z = PAD_TOP + 0.104


def spawn_part(xy=(0.0, 0.0), mass=None):
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=2.5, restitution=0.0)
    cfg = sim_utils.CylinderCfg(
        radius=P.PART_R, height=P.PART_H,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.037 if mass is None else float(mass)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=mat,
    )
    cfg.func("/World/part", cfg, translation=(float(xy[0]), float(xy[1]), P.PART_Z))
    return RigidObject(RigidObjectCfg(prim_path="/World/part", spawn=None))


def enable_contact_reporters(prim_root, name_regex):
    """为 prim_root 下指定名称的刚体挂 PhysX 接触回报 API，供 ContactSensor 读取求解器接触力。

    返回实际挂载的刚体 prim 路径列表；必须在 sim.reset() 之前、clear_instancing 之后调用。
    """
    import re

    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    paths = []
    stack = [stage.GetPrimAtPath(prim_root)]
    while stack:
        prim = stack.pop()
        if not prim or not prim.IsValid():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if re.fullmatch(name_regex, prim.GetName()):
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
                    rb.CreateSleepThresholdAttr().Set(0.0)
                if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                else:
                    cr = PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPath())
                cr.CreateThresholdAttr().Set(0.0)
                paths.append(str(prim.GetPath()))
            continue
        stack.extend(prim.GetChildren())
    return paths


def spawn_drone(prim, usd_name, pos, kinematic=False, init_joint_pos=None, arm_actuators=True):
    if not usd_name.endswith(".usd"):
        usd_name = usd_name + ".usd"
    spawn = sim_utils.UsdFileCfg(usd_path=str(USD_DIR / usd_name))
    if kinematic:
        spawn = sim_utils.UsdFileCfg(usd_path=str(USD_DIR / usd_name),
                                     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
    init = ArticulationCfg.InitialStateCfg(pos=tuple(pos))
    if init_joint_pos is not None:
        init = ArticulationCfg.InitialStateCfg(pos=tuple(pos), joint_pos=init_joint_pos)
    if arm_actuators:
        acts = {
            "slides": ImplicitActuatorCfg(joint_names_expr=SLIDE_JOINTS,
                                          stiffness=P.DRIVES["slide"]["stiffness"],
                                          damping=P.DRIVES["slide"]["damping"],
                                          effort_limit=P.DRIVES["slide"]["effort"]),
            "bend": ImplicitActuatorCfg(joint_names_expr=BEND_JOINTS,
                                        stiffness=P.DRIVES["bend"]["stiffness"],
                                        damping=P.DRIVES["bend"]["damping"],
                                        effort_limit=P.DRIVES["bend"]["effort"]),
            "grip": ImplicitActuatorCfg(joint_names_expr=["grip_[0-9]+"],
                                        stiffness=P.DRIVES["grip"]["stiffness"],
                                        damping=P.DRIVES["grip"]["damping"],
                                        effort_limit=P.DRIVES["grip"]["effort"]),
            "props": ImplicitActuatorCfg(joint_names_expr=["prop_[0-9]+"],
                                         stiffness=0.0, damping=0.001, effort_limit=0.5),
        }
    else:
        acts = {
            "mast": ImplicitActuatorCfg(joint_names_expr=["mast_j"],
                                        stiffness=2.0, damping=0.5, effort_limit=1.0),
            "props": ImplicitActuatorCfg(joint_names_expr=["prop_[0-9]+"],
                                         stiffness=0.0, damping=0.001, effort_limit=0.5),
        }
    cfg = ArticulationCfg(
        prim_path=prim,
        spawn=spawn,
        init_state=init,
        actuators=acts,
    )
    return Articulation(cfg)


def set_root_visual(prim_path, pose):
    """把根位移同步到 USD 变换，渲染跟随，与物理写入并用。"""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return
    xf = UsdGeom.Xformable(prim)
    t_op = None
    o_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            o_op = op
    if t_op is None:
        t_op = xf.AddTranslateOp()
    if o_op is None:
        o_op = xf.AddOrientOp()
    px, py, pz, qw, qx, qy, qz = pose
    t_op.Set(Gf.Vec3d(float(px), float(py), float(pz)))
    o_op.Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))


class DroneUnit:
    """一架具身无人机的完整闭环，飞控、电池、传感器、估计、链路、机械臂。"""

    def __init__(self, arm, name, rng, has_arm=True, mass_extra=0.0, use_fusion=False,
                 tendon=False):
        self.arm = arm
        self.name = name
        self.has_arm = has_arm
        self.bank = ThrusterBank()
        self.battery = BatteryPack()
        self.fc = FlightController(mass=0.0)
        self.sensors = SensorSim(rng)
        self.use_fusion = use_fusion
        self.use_mocap = False
        if use_fusion:
            self.est = FusionEstimator(np.zeros(3), 0.001)
        else:
            self.est = StateEstimator(np.zeros(3), 0.001)
        self._last_baro = -1.0
        self.rc = SimpleLink(S.RC_RATE, S.RC_LATENCY, rng, drop=0.0)
        self.mass_extra = mass_extra
        self.mass = None
        self.hold = False
        self.bend_ff = np.array([-1.0, 1.0])
        self.bend_rest = None
        self.arm_mass_total = None
        self.cg_rest = None
        self.cg_ff_on = True
        self.cg_jac = None
        self.bend_rest_actual = None
        self.dcg_filt = np.zeros(2)
        self._ff_prev = np.zeros(3)
        self._cg_prev = None
        self._cgv = np.zeros(2)
        self.cg_damp_t = 0.03
        self._carry = None
        self._v_prev = np.zeros(3)
        self._t = 0.0
        self._gnss_prev_t = 0.0
        self._last_gnss = np.zeros(3)
        self._last_fz = 0.0
        self._applied_m = np.zeros(3)
        self._initialized = False
        self.root_id = 0
        self.dist_force_w = np.zeros(3)
        self.dist_torque_w = np.zeros(3)
        self.contact_sensors = []

        def _find(names):
            ids, found = [], []
            for nm in names:
                try:
                    i, n = arm.find_joints([nm])
                    if i:
                        ids += list(i)
                        found += list(n)
                except Exception:
                    pass
            return ids, found

        self.slide_ids, _ = _find(SLIDE_JOINTS)
        self.bend_ids, _ = _find(BEND_JOINTS)
        self.grip_ids, _ = _find(["grip_[0-9]+"])
        self.prop_ids, _ = _find(["prop_[0-9]+"])
        self.finger_bodies = [i for i, n in enumerate(arm.body_names) if n.startswith("finger0_")]

        # 腱索-关节耦合等效模式（tendon_model.py）：
        # 弯曲 PD 从仿真驱动器转交腱索张力层，运行时经 effort 注入，不改 USD。
        self.tendon_on = bool(tendon)
        self.tendon = None
        self.tendon_last = None
        self._bend_tgt = None
        self._last_dt = 0.001
        if self.tendon_on and self.bend_ids:
            self.tendon = TendonModel(kp=P.DRIVES["bend"]["stiffness"],
                                      kd=P.DRIVES["bend"]["damping"])
            z = torch.zeros((1, len(self.bend_ids)), device=self.arm.device)
            self.arm.write_joint_stiffness_to_sim(z, joint_ids=self.bend_ids)
            self.arm.write_joint_damping_to_sim(z, joint_ids=self.bend_ids)
            print("[unit] 腱索等效模式：弯曲力矩转交易腱索层（预紧 %.1fN 摩擦 %.2f 滞回 %.1f）"
                  % (self.tendon.preload, self.tendon.mu, self.tendon.hyst), flush=True)

    def arm_cg_xy(self):
        if self.arm_mass_total is None:
            masses = self.arm.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)
            self.arm_mass_total = float(masses.sum())
            self._link_masses = masses
        masses = self._link_masses
        bpos = self.arm.data.body_pos_w[0].detach().cpu().numpy()
        wsum = max(masses.sum(), 1e-9)
        return (bpos * masses[:, None]).sum(axis=0) / wsum

    def set_grip_stiffness(self, k):
        t = torch.full((1, len(self.grip_ids)), float(k), device=self.arm.device)
        self.arm.write_joint_stiffness_to_sim(t, joint_ids=self.grip_ids)

    def set_bend_stiffness(self, k):
        if self.tendon_on and self.tendon is not None:
            self.tendon.kp = float(k)
            print("[unit] 弯曲刚度调度 -> 腱索张力层 kp=%.1f" % float(k), flush=True)
            return
        t = torch.full((1, len(self.bend_ids)), float(k), device=self.arm.device)
        self.arm.write_joint_stiffness_to_sim(t, joint_ids=self.bend_ids)

    def grip_contacts(self):
        """夹爪关节停在对象区间视为有效接触，指尖互碰约零点一四以上排除。"""
        try:
            q = float(self.arm.data.joint_pos[0, self.grip_ids].mean())
            return 2 if -0.10 < q < 0.15 else 0
        except Exception:
            return 0

    def bind_contact_sensors(self, sensors):
        """绑定指垫-试件接触力传感器（isaaclab ContactSensor 列表），读数出自 PhysX 接触求解器。"""
        self.contact_sensors = list(sensors)
        return len(self.contact_sensors)

    def read_grip_forces(self, dt=0.001):
        """读指垫-试件接触力（PhysX ContactSensor 净力），返回 {ncon, fn, ft, pads:[(mag,0,nc,mag)]}。

        无绑定传感器或读取失败返回 None（调用方回退几何判据）。
        """
        sensors = getattr(self, "contact_sensors", None)
        if not sensors:
            return None
        try:
            ncon = 0
            fn = 0.0
            pads = []
            for s in sensors:
                nf = s.data.net_forces_w
                nf = nf.detach().cpu().numpy().reshape(-1, 3)
                mag = float(np.linalg.norm(nf.sum(axis=0))) if len(nf) else 0.0
                pads.append((mag, 0.0, 1 if mag > 0.005 else 0, mag))
                if mag > 0.005:
                    ncon += 1
                    fn += mag
            return {"ncon": ncon, "fn": fn, "ft": 0.0, "pads": pads}
        except Exception as e:
            print("[unit] contact force read failed:", e, flush=True)
            return None

    def read_state(self):
        d = self.arm.data
        pos = d.root_pos_w[0].detach().cpu().numpy().copy()
        quat = d.root_quat_w[0].detach().cpu().numpy().copy()
        vel = d.root_lin_vel_w[0].detach().cpu().numpy().copy()
        omega = d.root_ang_vel_w[0].detach().cpu().numpy().copy()
        return pos, quat, vel, omega

    def update_composite_inertia(self):
        try:
            masses = self.arm.root_physx_view.get_masses().detach().cpu().numpy().astype(float).reshape(-1)
            inerts = self.arm.root_physx_view.get_inertias().detach().cpu().numpy().astype(float).reshape(-1, 3, 3)
            bp = self.arm.data.body_pos_w[0].detach().cpu().numpy()
            bq = self.arm.data.body_quat_w[0].detach().cpu().numpy()
            M = float(masses.sum())
            com = (masses[:, None] * bp).sum(axis=0) / max(M, 1e-6)
            Iw = np.zeros((3, 3))
            root_R = None
            for i in range(len(masses)):
                qw, qx, qy, qz = bq[i]
                Rm = np.array([
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
                if i == int(self.root_id):
                    root_R = Rm
                d = bp[i] - com
                Iw += Rm @ inerts[i] @ Rm.T + masses[i] * ((d @ d) * np.eye(3) - np.outer(d, d))
            if root_R is None:
                root_R = np.eye(3)
            Ic = np.maximum(np.diag(root_R.T @ Iw @ root_R), 1e-4)
            self.fc.inertia = Ic
            return Ic
        except Exception:
            return None

    def step(self, dt, cmd_pos, cmd_yaw=0.0, arm_extend=None, arm_grip=None, arm_bend=None, mass=None):
        self._last_dt = float(dt)
        pos, quat, vel, omega = self.read_state()
        if not self._initialized:
            self.est.pos = pos.copy()
            if mass is None:
                self.mass = float(self.arm.root_physx_view.get_masses().sum()) + self.mass_extra
            else:
                self.mass = mass
            self.fc.mass = self.mass
            Ic0 = self.update_composite_inertia()
            if Ic0 is not None:
                print("[unit] 惯量 复合=%s" % (np.round(Ic0, 4).tolist(),), flush=True)
            self._inertia_t = 0.0
            self._initialized = True
        self._inertia_t = getattr(self, "_inertia_t", 0.0) + dt
        if self._inertia_t >= 0.5:
            self._inertia_t = 0.0
            self.update_composite_inertia()
        m = self.mass if mass is None else mass

        acc_world = (vel - self._v_prev) / max(dt, 1e-5)
        self._v_prev = vel.copy()
        acc_body = quat_rotate(quat_conj(quat), acc_world + np.array([0.0, 0.0, S.G]))
        gyro_body = quat_rotate(quat_conj(quat), omega)
        acc_m, gyro_m = self.sensors.imu(acc_body, gyro_body, dt)
        if self.use_mocap:
            omega_body = quat_rotate(quat_conj(quat), np.asarray(omega, dtype=float))
            est_out = {"pos": pos.copy(), "vel": vel.copy(), "quat": quat.copy(),
                       "omega": omega_body.copy()}
        elif self.use_fusion:
            self.est.update_imu(acc_m, gyro_m, dt)
            if self._t - self._last_baro >= 1.0 / S.BARO_RATE - 1e-9:
                self._last_baro = self._t
                self.est.update_baro(self.sensors.baro(pos[2], self._t), 1.0 / S.BARO_RATE)
            gnss = self.sensors.gnss(pos, self._t)
            if self._t - self._gnss_prev_t >= 1.0 / S.GNSS_RATE - 1e-9:
                self._gnss_prev_t = self._t
                self.est.update_gnss(gnss, 1.0 / S.GNSS_RATE)
            est_out = self.est.out(gyro_body)
        else:
            self.est.update_imu(acc_m, gyro_m, dt)
            gnss = self.sensors.gnss(pos, self._t)
            if self._t - self._gnss_prev_t >= 1.0 / S.GNSS_RATE - 1e-9:
                self._gnss_prev_t = self._t
                self.est.correct_position(gnss, 1.0 / S.GNSS_RATE)
            est_out = self.est.out(gyro_body)
        self.last_est = est_out
        fz, moments = self.fc.compute(est_out, np.asarray(cmd_pos), cmd_yaw, dt,
                                      moments_actual=self._applied_m)
        if arm_bend is not None and len(self.bend_ids) and self.cg_ff_on and self.cg_jac is not None:
            qb = self.arm.data.joint_pos[0, self.bend_ids].detach().cpu().numpy()
            bx_avg = float(np.mean(qb[:P.N_SEG]))
            by_avg = float(np.mean(qb[P.N_SEG:]))
            if abs(float(arm_bend[0])) < 1e-6 and abs(float(arm_bend[1])) < 1e-6:
                self.bend_rest_actual = np.array([bx_avg, by_avg])
                d_act = np.zeros(2)
            elif self.bend_rest_actual is not None:
                d_act = np.array([bx_avg, by_avg]) - self.bend_rest_actual
            else:
                d_act = np.zeros(2)
            cg_now = self.arm_cg_xy()[:2]
            if self.arm_mass_total is None:
                self.arm_cg_xy()
            wgt = self.arm_mass_total * 9.81
            dcg = self.cg_jac @ d_act
            damp = np.zeros(2)
            if self._cg_prev is not None:
                cgv = (cg_now - self._cg_prev) / max(dt, 1e-6)
                self._cgv = self._cgv + (cgv - self._cgv) * min(dt / 0.1, 1.0)
                damp = -self.cg_damp_t * self._cgv
            self._cg_prev = cg_now.copy()
            ff_t = np.clip(np.array([-wgt * (dcg[1] + damp[1]), wgt * (dcg[0] + damp[0]), 0.0]), -0.25, 0.25)
            self._ff_prev = self._ff_prev + np.clip(ff_t - self._ff_prev, -0.15 * dt, 0.15 * dt)
            self.fc.extra_moment = self._ff_prev.copy()
        else:
            self._ff_prev[:] = 0.0
            self._cg_prev = None
            self.fc.extra_moment = np.zeros(3)
        self._last_fz = fz
        v = self.battery.step(fz, dt)
        thrusts = self.bank.mixer(fz, moments)
        omega_cmd = np.sqrt(np.clip(thrusts / S.K_F, 0.0, S.OMEGA_MAX ** 2))
        thrust_now, torque_z = self.bank.step(omega_cmd, dt, v_scale=self.battery.v_scale())
        fz_now, m_now = self.bank.wrench(thrust_now, torque_z)
        self._applied_m = np.asarray(m_now, dtype=float).copy()
        if self.prop_ids:
            spin = torch.tensor(self.bank.omega * self.bank.dir, dtype=torch.float32,
                                device=self.arm.device).reshape(1, -1)
            self.arm.set_joint_velocity_target(spin, joint_ids=self.prop_ids)

        f_app = np.array([0.0, 0.0, fz_now], dtype=float) + np.asarray(self.dist_force_w, dtype=float)
        m_app = np.asarray(m_now, dtype=float) + np.asarray(self.dist_torque_w, dtype=float)
        forces = torch.tensor(f_app, dtype=torch.float32, device=self.arm.device).reshape(1, 1, 3)
        torques = torch.tensor(m_app, dtype=torch.float32, device=self.arm.device).reshape(1, 1, 3)
        self.arm.set_external_force_and_torque(
            forces, torques, body_ids=[self.root_id])
        self.arm.write_data_to_sim()

        if self.has_arm and (arm_extend is not None or arm_grip is not None or arm_bend is not None):
            self.set_arm(arm_extend, arm_grip, arm_bend)
        self.carry_tick()
        self._t += dt
        return {"pos": pos, "quat": quat, "fz": fz_now, "batt_v": v, "mass": self.mass,
                "est_quat": self.est.quat.copy(), "est_pos": self.est.pos.copy(),
                "moments": np.asarray(m_now, dtype=float).copy(),
                "omega": np.asarray(omega, dtype=float).copy(),
                "cmd_fz": fz}

    def set_arm(self, extend=None, grip=None, bend=None):
        if extend is not None:
            tgt = self.arm.data.joint_pos[:, self.slide_ids].clone()
            tgt[:] = -0.040 * float(np.clip(extend, 0.0, 1.0))
            self.arm.set_joint_position_target(tgt, joint_ids=self.slide_ids)
        if bend is not None and self.bend_ids:
            bxt = torch.full((1, P.N_SEG), float(bend[0]), device=self.arm.device)
            byt = torch.full((1, P.N_SEG), float(bend[1]), device=self.arm.device)
            self.arm.set_joint_position_target(bxt, joint_ids=self.bend_ids[:P.N_SEG])
            self.arm.set_joint_position_target(byt, joint_ids=self.bend_ids[P.N_SEG:])
            if self.tendon_on and self.tendon is not None:
                self._bend_tgt = (float(bend[0]), float(bend[1]))
        if self.tendon_on and self.tendon is not None and self._bend_tgt is not None:
            q = self.arm.data.joint_pos[0, self.bend_ids].detach().cpu().numpy()
            qd = self.arm.data.joint_vel[0, self.bend_ids].detach().cpu().numpy()
            qt = np.array([self._bend_tgt[0]] * P.N_SEG + [self._bend_tgt[1]] * P.N_SEG)
            tau, info = self.tendon.torques(q, qd, qt, dt=self._last_dt)
            self.tendon_last = info
            tt = torch.tensor(tau, dtype=torch.float32, device=self.arm.device).reshape(1, -1)
            self.arm.set_joint_effort_target(tt, joint_ids=self.bend_ids)
        if grip is not None:
            c = P.GRIP_OPEN_CTRL + P.GRIP_CTRL_PER * float(np.clip(grip, 0.0, 1.0))
            tgt = torch.full((1, len(self.grip_ids)), c, device=self.arm.device)
            self.arm.set_joint_position_target(tgt, joint_ids=self.grip_ids)
        self.arm.write_data_to_sim()

    def attach_part(self, part, wrist_path="/World/DroneA/wrist"):
        """抓取吸附：捕获相对位姿后逐帧随腕运动学携带。"""
        wi = self.arm.body_names.index("wrist")
        wrist_p = self.arm.data.body_pos_w[0, wi].detach().cpu().numpy()
        wrist_q = self.arm.data.body_quat_w[0, wi].detach().cpu().numpy()
        part_p = part.data.root_pos_w[0].detach().cpu().numpy()
        part_q = part.data.root_quat_w[0].detach().cpu().numpy()
        inv = np.array([wrist_q[0], -wrist_q[1], -wrist_q[2], -wrist_q[3]])
        relq = quat_mul(inv, part_q)
        relpos = quat_rotate(inv, part_p - wrist_p)
        self._carry = (part, relpos, relq)
        self.hold = True
        return "carry"

    def carry_tick(self):
        if not self.hold or self._carry is None:
            return
        part, relpos, relq = self._carry
        wi = self.arm.body_names.index("wrist")
        wrist_p = self.arm.data.body_pos_w[0, wi].detach().cpu().numpy()
        wrist_q = self.arm.data.body_quat_w[0, wi].detach().cpu().numpy()
        pos = wrist_p + quat_rotate(wrist_q, relpos)
        quat = quat_mul(wrist_q, relq)
        pose = torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
                            dtype=torch.float32, device=self.arm.device)
        part.write_root_pose_to_sim(pose)
        part.write_root_velocity_to_sim(torch.zeros((1, 6), device=part.device))

    def release_part(self):
        self.hold = False
        self._carry = None
        return True

    def refresh(self, dt):
        self.arm.update(dt)

    def tilt_deg(self):
        _, quat, _, _ = self.read_state()
        up = quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
