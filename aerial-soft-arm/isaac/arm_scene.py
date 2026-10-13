"""桁架臂 Isaac Lab 场景与驱动封装。

场景包含固定的机械臂 articulation、试件立柱与橙色试件圆柱。
抓取持物用运行时固定关节，与 MuJoCo 版的运动学吸附对应。
"""
import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

import arm_params as P

HERE = Path(__file__).resolve().parent
USD_DIR = HERE / "usd"

SLIDE_JOINTS = ["slide_%d" % i for i in range(1, P.N_SEG)] + ["slide_wrist"]
BEND_JOINTS = ["bx_%d" % i for i in range(P.N_SEG)] + ["by_%d" % i for i in range(P.N_SEG)]


def make_sim():
    sim = SimulationContext(SimulationCfg(dt=0.001, device="cuda:0"))
    return sim


def spawn_static(sim):
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    stand_mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
    post = sim_utils.CylinderCfg(
        radius=0.012, height=0.78,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=stand_mat,
    )
    post.func("/World/stand/post", post, translation=(0.0, 0.0, 0.74 + 0.39))
    plate = sim_utils.CylinderCfg(
        radius=0.050, height=0.016,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=stand_mat,
    )
    plate.func("/World/stand/plate", plate, translation=(0.0, 0.0, 1.525))
    light = sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.92, 0.95))
    light.func("/World/Light", light)


def spawn_part(sim):
    part_mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2, restitution=0.0)
    part_cfg = sim_utils.CylinderCfg(
        radius=P.PART_R, height=P.PART_H,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.037),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=part_mat,
    )
    part_cfg.func("/World/part", part_cfg, translation=(0.0, 0.0, P.PART_Z))
    return RigidObject(RigidObjectCfg(prim_path="/World/part", spawn=None))


def spawn_arm(gripper_key):
    usd = USD_DIR / ("arm_%s.usd" % gripper_key)
    cfg = ArticulationCfg(
        prim_path="/World/Arm",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, P.ARM_MOUNT_Z)),
        actuators={
            "slides": ImplicitActuatorCfg(
                joint_names_expr=SLIDE_JOINTS,
                stiffness=P.DRIVES["slide"]["stiffness"],
                damping=P.DRIVES["slide"]["damping"],
                effort_limit=P.DRIVES["slide"]["effort"],
            ),
            "bend": ImplicitActuatorCfg(
                joint_names_expr=BEND_JOINTS,
                stiffness=P.DRIVES["bend"]["stiffness"],
                damping=P.DRIVES["bend"]["damping"],
                effort_limit=P.DRIVES["bend"]["effort"],
            ),
            "grip": ImplicitActuatorCfg(
                joint_names_expr=["grip_[0-9]+"],
                stiffness=P.DRIVES["grip"]["stiffness"],
                damping=P.DRIVES["grip"]["damping"],
                effort_limit=P.DRIVES["grip"]["effort"],
            ),
        },
    )
    return Articulation(cfg)


class ArmDriver:
    """指令缓冲到关节目标的映射，保持与 MuJoCo 面板同一套口径。"""

    def __init__(self, arm: Articulation, part: RigidObject):
        self.arm = arm
        self.part = part
        self.slide_ids, _ = arm.find_joints(SLIDE_JOINTS)
        self.bend_ids, _ = arm.find_joints(BEND_JOINTS)
        self.grip_ids, self.grip_names = arm.find_joints(["grip_[0-9]+"])
        self.hinge_pos = {i: -0.036 for i in self.grip_ids}
        n = arm.num_joints
        self.offsets = torch.zeros(1, n, device=arm.device)
        self.e = 0.0

    def extend(self, e):
        """气压归一化 0..1 映射滑移目标。"""
        self.e = float(np.clip(e, 0.0, 1.0))
        tgt = self.arm.data.joint_pos[:, self.slide_ids].clone()
        tgt[:] = -0.040 * self.e
        self.arm.set_joint_position_target(tgt, joint_ids=self.slide_ids)

    def bend(self, bx, by):
        tgt = self.arm.data.joint_pos[:, self.bend_ids].clone()
        for k, jid in enumerate(self.bend_ids):
            if k < P.N_SEG:
                tgt[0, k] = bx
            else:
                tgt[0, k] = by
        self.arm.set_joint_position_target(tgt, joint_ids=self.bend_ids)

    def grip(self, g):
        c = P.GRIP_OPEN_CTRL + P.GRIP_CTRL_PER * float(np.clip(g, 0.0, 1.0))
        tgt = torch.full((1, len(self.grip_ids)), c, device=self.arm.device)
        self.arm.set_joint_position_target(tgt, joint_ids=self.grip_ids)

    def set_pressure_stiffness(self, bar):
        """压力耦合弯曲刚度，对应面板压力刚度项。"""
        e = min(bar / P.P_STEP_BAR, 1.0)
        k = P.DRIVES["bend"]["stiffness"] + 2.0 * e
        self.arm.write_joint_stiffness_to_sim(
            torch.full((1, len(self.bend_ids)), k, device=self.arm.device),
            joint_ids=self.bend_ids,
        )

    def commit(self):
        """把目标写进 PhysX。"""
        self.arm.write_data_to_sim()

    def refresh(self, dt):
        """从 PhysX 取回状态。"""
        self.arm.update(dt)
        self.part.update(dt)

    def state(self):
        return {
            "joint_pos": self.arm.data.joint_pos[0].detach().cpu().numpy().copy(),
            "wrist_z": float(self.arm.data.body_pos_w[0, self.arm.body_names.index("wrist"), 2]),
            "part_z": float(self.part.data.root_pos_w[0, 2]),
        }

    def attach_part(self):
        """运行时固定关节，把试件焊到腕部当前位置。"""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        wrist_p = self.arm.data.body_pos_w[0, self.arm.body_names.index("wrist")].detach().cpu().numpy()
        wrist_q = self.arm.data.body_quat_w[0, self.arm.body_names.index("wrist")].detach().cpu().numpy()
        part_p = self.part.data.root_pos_w[0].detach().cpu().numpy()
        part_q = self.part.data.root_quat_w[0].detach().cpu().numpy()

        def qmul(a, b):
            w1, x1, y1, z1 = a
            w2, x2, y2, z2 = b
            return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

        def qinv(q):
            return np.array([q[0], -q[1], -q[2], -q[3]])

        def qrot(q, v):
            qv = np.array([0.0, v[0], v[1], v[2]])
            return qmul(qmul(q, qv), qinv(q))[1:]

        relpos = qrot(qinv(wrist_q), part_p - wrist_p)
        relq = qmul(qinv(wrist_q), part_q)
        jp = "/World/grasp_joint"
        if stage.GetPrimAtPath(jp):
            stage.RemovePrim(jp)
        joint = UsdPhysics.FixedJoint.Define(stage, jp)
        joint.CreateBody0Rel().SetTargets(["/World/Arm/wrist"])
        joint.CreateBody1Rel().SetTargets(["/World/part"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*relpos))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(relq[0]), float(relq[1]), float(relq[2]), float(relq[3])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return jp

    def release_part(self):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        jp = "/World/grasp_joint"
        if stage.GetPrimAtPath(jp):
            stage.RemovePrim(jp)
            return True
        return False
