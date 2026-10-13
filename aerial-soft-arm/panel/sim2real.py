"""sim2real 桥：把仿真运行日志导出为真机控制器可消费的命令流与标定文件。"""
import csv
import json
from pathlib import Path

import numpy as np

import arm_real as A
import arm_lattice as L
from drivers import DEFAULTS


def export_calib(out_json: Path):
    """导出结构、气动、肌腱、夹爪标定参数。"""
    calib = {
        "geometry": {
            "n_seg": L.N_SEG,
            "ring_radius_m": L.R_RING,
            "span_min_m": L.SPAN_MIN,
            "span_max_m": L.SPAN_MAX,
            "slide_travel_m": 0.040,
        },
        "pneumatic": {
            "p_max_bar": A.P_STEP,
            "slide_gain_m_per_e": 0.040,
        },
        "tendon": {
            "angles_deg": list(L.TENDON_ANGLES),
            "force_max_n": DEFAULTS["tendon_max"],
        },
        "structure": {
            "k_soft": L.K_SOFT,
            "c_soft": L.C_SOFT,
            "k_stiff": L.K_STIFF,
            "c_stiff": L.C_STIFF,
        },
        "gripper": {
            "variants": [g["name"] for g in L.GRIPPER_DEFS],
            "joint_range_rad": [-0.45, 0.20],
            "ctrl_open_rad": -0.40,
            "ctrl_close_rad": 0.30,
        },
        "rates": {k: DEFAULTS[k] for k in ("joint_rate", "ext_rate", "grip_rate", "v_max", "yaw_max")},
    }
    out_json.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    return calib


def export_commands(log_npz: Path, out_csv: Path):
    """把 log.npz 转为逐帧命令流 CSV。"""
    z = np.load(log_npz)
    t = z["t"]
    p_ref = z["p_ref"]
    grip = z["grip"]
    drone = z["drone"]
    locks = z["locks"]
    estop = z["estop"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "p_ref_bar", "grip", "drone_x", "drone_y", "drone_z", "drone_yaw",
                    "lock_mask", "estop"])
        for i in range(len(t)):
            mask = int(sum(int(locks[i, k]) << k for k in range(locks.shape[1])))
            w.writerow(["%.4f" % t[i], "%.4f" % p_ref[i], "%.3f" % grip[i],
                        "%.4f" % drone[i, 0], "%.4f" % drone[i, 1], "%.4f" % drone[i, 2],
                        "%.4f" % drone[i, 3], mask, int(estop[i])])
    return len(t)


def export_run(run_dir: Path):
    run_dir = Path(run_dir)
    n = export_commands(run_dir / "log.npz", run_dir / "commands.csv")
    export_calib(run_dir / "calibration.json")
    return n
