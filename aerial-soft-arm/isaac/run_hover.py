"""具身无人机悬停闭环，工业级验证第一步。

用法
  python run_hover.py --headless [--usd drone_a_diamond] [--hover 1.5] [--seconds 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "embodied"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="AS-01 embodied hover")
parser.add_argument("--usd", default="drone_a_diamond")
parser.add_argument("--hover", type=float, default=1.35)
parser.add_argument("--seconds", type=float, default=9.0)
parser.add_argument("--air-spawn", action="store_true", help="空中出生，纯飞控隔离")
parser.add_argument("--out", default="runs/hover")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import kit  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
kit.mark("app launched")

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

import scene_embodied as SE  # noqa: E402
import spec as S  # noqa: E402


def main():
    sim = SimulationContext(SimulationCfg(dt=0.001, device="cuda:0"))
    SE.spawn_ground_and_stand()
    SE.spawn_part()
    if args_cli.air_spawn:
        start_z = args_cli.hover
    else:
        start_z = SE.ARM_CLEAR_Z
    arm = SE.spawn_drone("/World/DroneA", args_cli.usd, (SE.PAD_XY[0], SE.PAD_XY[1], start_z))
    sim.reset()
    kit.mark("scene reset, joints=%d" % arm.num_joints)

    rng = np.random.default_rng(11)
    unit = SE.DroneUnit(arm, "A", rng, has_arm=True)
    kit.mark("mass=%.3f kg inertia=%s" % (float(arm.root_physx_view.get_masses().sum()),
                                          unit.fc.inertia.round(5)))

    out = Path(args_cli.out)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    tele = []

    dt = 0.001
    start = np.array([SE.PAD_XY[0], SE.PAD_XY[1], start_z])
    n = int(args_cli.seconds / dt)
    cmd = start.copy()
    for k in range(n):
        t = k * dt
        if t < 0.8:
            cmd = start.copy()
        elif t < 1.2:
            cmd = start + (np.array([0.0, 0.0, args_cli.hover]) - start) * (t - 0.8) / 0.4
        else:
            cmd = np.array([SE.PAD_XY[0], SE.PAD_XY[1], args_cli.hover])
        st = unit.step(dt, cmd, 0.0)
        sim.step()
        unit.refresh(dt)
        if k % 200 == 0:
            tele.append((t, st["pos"].copy(), st["batt_v"], st["fz"], unit.tilt_deg()))
        if k % 1000 == 0:
            print("[hover] t=%.1f pos=(%.3f %.3f %.3f) tilt=%.1f fz=%.2f V=%.2f"
                  % (t, st["pos"][0], st["pos"][1], st["pos"][2], unit.tilt_deg(), st["fz"], st["batt_v"]), flush=True)

    tail = [p for (tt, p, v, f, tl) in tele if tt > 5.0]
    err = np.linalg.norm(np.array(tail) - np.array([SE.PAD_XY[0], SE.PAD_XY[1], args_cli.hover]), axis=1) if tail else np.array([9.9])
    tilt = [tl for (tt, p, v, f, tl) in tele if tt > 5.0] or [99.0]
    volt = [v for (tt, p, v, f, tl) in tele if tt > 5.0] or [0.0]
    print("[hover] 后段误差 max=%.3f mean=%.3f m, 倾角 max=%.1f 度, 电压 %.2f"
          % (err.max(), err.mean(), max(tilt), min(volt)), flush=True)
    ok = err.max() < 0.12 and max(tilt) < 12.0 and min(volt) > 20.5 and np.isfinite(st["pos"]).all()
    print("[hover] %s 判据 误差<0.12 倾角<12 电压>20.5" % ("PASS" if ok else "FAIL"), flush=True)
    np.savez_compressed(out / "hover.npz",
                        t=[x[0] for x in tele], pos=np.stack([x[1] for x in tele]),
                        v=[x[2] for x in tele], fz=[x[3] for x in tele], tilt=[x[4] for x in tele])
    return 0 if ok else 1


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
