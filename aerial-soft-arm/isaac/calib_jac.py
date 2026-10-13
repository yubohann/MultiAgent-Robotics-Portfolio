"""零重力离线标定，测弯曲到指尖与弯曲到质心的运动学雅可比，存 bend_jac.json。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "embodied"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="bend jacobian calibration")
parser.add_argument("--tendon", action="store_true",
                    help="腱索-关节耦合等效模式下标定，输出 bend_jac_tendon.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import kit  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
kit.mark("app launched")

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

import fx  # noqa: E402


def main():
    sim = SimulationContext(SimulationCfg(dt=0.001, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
    fx.build_studio(sim_utils)
    from scene_embodied import spawn_drone, spawn_ground_and_stand, DroneUnit
    spawn_ground_and_stand()
    arm = spawn_drone("/World/DroneA", "drone_a_diamond", (0.0, 0.0, 1.5))
    sim.reset()
    kit.mark("reset")

    ua = DroneUnit(arm, "A", np.random.default_rng(5), has_arm=True, use_fusion=False,
                   tendon=args_cli.tendon)
    if args_cli.tendon:
        ua.tendon.kp = 10.0
        ua.tendon.kd = 4.0
        ua.tendon.reset()
        print("[cj] 腱索等效模式 preload=%.1f mu=%.2f hyst=%.3f" % (
            ua.tendon.preload, ua.tendon.mu, ua.tendon.hyst), flush=True)
    ts = torch.full((1, len(ua.bend_ids)), 10.0, device=arm.device)
    td = torch.full((1, len(ua.bend_ids)), 0.5, device=arm.device)
    arm.write_joint_stiffness_to_sim(ts, joint_ids=ua.bend_ids)
    arm.write_joint_damping_to_sim(td, joint_ids=ua.bend_ids)
    print("[cj] 弯曲伺服覆盖 k=10 c=0.5", flush=True)
    dist_ids = [i for i, n in enumerate(arm.body_names) if "dist" in n]
    if not dist_ids:
        dist_ids = [i for i, n in enumerate(arm.body_names) if n.startswith("finger")]
    cmd = np.array([0.0, 0.0, 1.5])
    dt = 0.001
    total = 11.0
    bend = np.array([0.0, 0.0])
    bend_t = np.array([0.0, 0.0])
    marks = [(4.0, np.array([0.10, 0.0])), (6.5, np.array([0.0, 0.10])), (9.0, np.array([0.0, 0.0]))]
    mi = 0
    tip0 = None
    cg0 = None
    Jb = None
    Jc = None
    for k in range(int(total / dt)):
        t = k * dt
        e = 0.62 * min(t / 2.0, 1.0)
        bend = bend + np.clip(bend_t - bend, -0.10 * dt, 0.10 * dt)
        if mi < len(marks) and t >= marks[mi][0]:
            root = arm.data.root_pos_w[0].detach().cpu().numpy()
            tip = arm.data.body_pos_w[0, dist_ids].detach().cpu().numpy().mean(axis=0)[:2]
            cg = ua.arm_cg_xy()[:2]
            tip_rel = tip - root[:2]
            cg_rel = cg - root[:2]
            if mi == 0:
                tip0, cg0 = tip_rel.copy(), cg_rel.copy()
                print("[cj] 基线 指尖=(%.3f %.3f) 质心=(%.3f %.3f)" % (
                    tip_rel[0], tip_rel[1], cg_rel[0], cg_rel[1]), flush=True)
            elif mi == 1:
                Jb = ((tip_rel - tip0) / 0.10).reshape(2, 1)
                Jc = ((cg_rel - cg0) / 0.10).reshape(2, 1)
                print("[cj] 弯曲x 指尖移=(%.3f %.3f) 质心移=(%.3f %.3f)" % (
                    tip_rel[0] - tip0[0], tip_rel[1] - tip0[1],
                    cg_rel[0] - cg0[0], cg_rel[1] - cg0[1]), flush=True)
            else:
                by_col = ((tip_rel - tip0 - Jb[:, 0] * 0.10) / 0.10).reshape(2, 1)
                cy_col = ((cg_rel - cg0 - Jc[:, 0] * 0.10) / 0.10).reshape(2, 1)
                Jb2 = np.hstack([Jb, by_col])
                Jc2 = np.hstack([Jc, cy_col])
                print("[cj] 弯曲y 指尖移=(%.3f %.3f) 质心移=(%.3f %.3f)" % (
                    tip_rel[0] - tip0[0], tip_rel[1] - tip0[1],
                    cg_rel[0] - cg0[0], cg_rel[1] - cg0[1]), flush=True)
                out = {"J_bend": Jb2.tolist(), "J_cg": Jc2.tolist(), "extend": 0.62}
                _jf = "bend_jac_tendon.json" if args_cli.tendon else "bend_jac.json"
                (HERE / _jf).write_text(json.dumps(out), encoding="utf-8")
                wgt = ua.arm_mass_total * 9.81
                m_bx = wgt * abs(Jc2[0, 0] * 0.10)
                m_by = wgt * abs(Jc2[1, 1] * 0.10)
                print("[cj] 雅可比完成 J_bend=%s J_cg=%s" % (
                    np.round(Jb2, 2).tolist(), np.round(Jc2, 2).tolist()), flush=True)
                print("[cj] 弯曲0.05 质心移=%.3f m 反力矩估计 x轴%.3f y轴%.3f Nm" % (
                    float(np.linalg.norm(Jc2[:, 0] * 0.05)), m_bx, m_by), flush=True)
            bend_t = marks[mi][1]
            mi += 1
        ua.step(dt, cmd, 0.0, arm_extend=e, arm_grip=0.0,
                arm_bend=[float(bend[0]), float(bend[1])])
        sim.step(render=False)
        ua.refresh(dt)
    return 0


rc = 0
try:
    rc = main()
except Exception:
    import traceback
    traceback.print_exc()
    rc = 2
finally:
    kit.bounded_close(simulation_app)
sys.exit(rc)
