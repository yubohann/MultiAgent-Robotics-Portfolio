"""桁架臂抓取场景主脚本，无头运行一次完整抓取并输出判据。

用法
  python run_grasp.py --gripper diamond --out runs/grasp_diamond
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="AS-01 grasp run")
parser.add_argument("--gripper", default="diamond", choices=("diamond", "flat", "three", "soft"))
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import kit  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
kit.mark("app launched")

import arm_scene as S  # noqa: E402
import arm_params as P  # noqa: E402


def main():
    sim = S.make_sim()
    S.spawn_static(sim)
    part = S.spawn_part(sim)
    arm = S.spawn_arm(args_cli.gripper)
    sim.reset()
    kit.mark("scene reset, joints=%d" % arm.num_joints)

    drv = S.ArmDriver(arm, part)
    names = arm.joint_names
    print("[grasp] 关节数 %d，滑移 %d，弯曲 %d，夹爪 %d" % (len(names), len(drv.slide_ids), len(drv.bend_ids), len(drv.grip_ids)))
    mass = float(arm.root_physx_view.get_masses().sum())
    print("[grasp] 总质量 %.4f kg" % mass)

    log = []

    def run(seconds, tag):
        n = int(seconds / 0.001)
        drv.commit()
        for _ in range(n):
            sim.step()
            drv.refresh(0.001)
        st = drv.state()
        log.append((tag, time.monotonic()))
        return st

    drv.grip(0.0)
    st = run(1.5, "settle")
    z0 = st["part_z"]
    print("[grasp] 初始 试件 z=%.4f 腕 z=%.4f" % (z0, st["wrist_z"]))

    drv.set_pressure_stiffness(0.9)
    drv.extend(P.GRASP_P)
    st = run(3.5, "extend")
    print("[grasp] 伸展后 腕 z=%.4f" % st["wrist_z"])

    drv.grip(1.0)
    st = run(1.8, "close")
    print("[grasp] 闭合后 试件 z=%.4f" % st["part_z"])

    jp = drv.attach_part()
    run(0.3, "attach")
    print("[grasp] 已建立固定关节 %s" % jp)

    drv.extend(0.25)
    st = run(3.5, "lift")
    z1 = st["part_z"]
    print("[grasp] 提升后 试件 z=%.4f (提升 %.4f)" % (z1, z1 - z0))

    drv.extend(P.GRASP_P)
    run(3.0, "down")
    drv.release_part()
    drv.grip(0.0)
    st = run(1.5, "release")
    z2 = st["part_z"]
    print("[grasp] 释放后 试件 z=%.4f" % z2)

    ok = (z1 - z0) > 0.08
    print("[grasp] %s 提升判据 %.4f > 0.08" % ("PASS" if ok else "FAIL", z1 - z0))
    if args_cli.out:
        out = Path(args_cli.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "state.npz", z0=z0, z1=z1, z2=z2, mass=mass)
    return 0 if ok else 1


rc = 0
try:
    rc = main()
finally:
    kit.bounded_close(simulation_app)
sys.exit(rc)
