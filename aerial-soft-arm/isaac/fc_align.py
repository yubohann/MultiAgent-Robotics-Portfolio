"""对位闭环试验，停稳伸臂弯曲夹爪四段流程。

停稳后伸臂，视觉测偏差，弯曲雅可比驱动弯曲做精对位，反力矩前馈保持水平，
死区触发闭合，关节堵转确认，刚度调度，提升。

用法
  python fc_align.py --headless --enable_cameras --out runs/fc_align_a --eye 0.38,-0.45,0.18
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "embodied"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="AS-01 full bend align v7")
parser.add_argument("--out", default="runs/fc_align")
parser.add_argument("--eye", default="0.38,-0.45,0.18")
parser.add_argument("--total", type=float, default=55.0)
parser.add_argument("--rigid", action="store_true", help="刚臂对照，弯曲刚度十四")
parser.add_argument("--no-ff", action="store_true", help="关闭反力矩前馈")
parser.add_argument("--no-contact", action="store_true", help="关闭接触确认")
parser.add_argument("--base-only", action="store_true", help="纯基座对位，禁用弯曲")
parser.add_argument("--no-damp", action="store_true", default=True, help="关闭主动摆振阻尼")
parser.add_argument("--damp", dest="no_damp", action="store_false", help="启用主动摆振阻尼")
parser.add_argument("--damp-k", type=float, default=1.0, help="摆振阻尼增益")
parser.add_argument("--kp-vel", type=float, default=2.5, help="速度环增益")
parser.add_argument("--kp-pos", type=float, default=0.5, help="位置环增益")
parser.add_argument("--est-g", type=float, default=0.35, help="姿态互补滤波增益")
parser.add_argument("--dob-l", type=float, default=0.0, help="扰动观测器带宽")
parser.add_argument("--gate", type=float, default=0.020, help="稳定闭合误差门限")
parser.add_argument("--no-realign", action="store_true", default=True, help="稳定期不重新对位")
parser.add_argument("--realign", dest="no_realign", action="store_false", help="启用重新对位")
parser.add_argument("--ki-att", type=float, default=0.10, help="姿态积分增益")
parser.add_argument("--kp-att", type=float, default=9.22, help="姿态环增益")
parser.add_argument("--ki-vel", type=float, default=0.3, help="速度积分增益")
parser.add_argument("--kd-rate", type=float, default=0.04, help="角速率阻尼增益")
parser.add_argument("--kp-rate", type=float, default=12.0, help="角速率环增益")
parser.add_argument("--zvd", action="store_true", help="伸臂指令 ZVD 输入整形")
parser.add_argument("--zvd-f", type=float, default=1.23, help="ZVD 整形频率 Hz")
parser.add_argument("--zvd-z", type=float, default=0.30, help="ZVD 阻尼比")
parser.add_argument("--seed", type=int, default=5, help="随机种子")
parser.add_argument("--part-r", type=float, default=None, help="试件半径 m（默认 0.014）")
parser.add_argument("--part-h", type=float, default=None, help="试件高度 m（默认 0.060）")
parser.add_argument("--part-z", type=float, default=None, help="试件中心 z，默认随高度自适应（底面高度与标准一致）")
parser.add_argument("--part-xy", default="0.0,0.0", help="试件 xy 位置 m，逗号分隔")
parser.add_argument("--est", default="legacy", choices=["legacy", "ekf"], help="估计器选择")
parser.add_argument("--dist", default="none", choices=["none", "wind", "downwash", "vision"], help="扰动注入场景")
parser.add_argument("--wind-f", type=float, default=1.0, help="恒定风扰水平力 N")
parser.add_argument("--wind-az", type=float, default=0.0, help="风扰方位角 deg")
parser.add_argument("--dw-f", type=float, default=2.0, help="下洗脉冲幅值 N")
parser.add_argument("--dw-t0", type=float, default=45.0, help="下洗脉冲起始时刻 s")
parser.add_argument("--dw-dur", type=float, default=0.5, help="下洗脉冲持续 s")
parser.add_argument("--vis-sigma", type=float, default=3.0, help="质心像素噪声 sigma px")
parser.add_argument("--vis-drop", type=float, default=0.25, help="掩膜缺失概率")
parser.add_argument("--tendon", action="store_true",
                    help="腱索-关节耦合等效模式（张力/预紧/摩擦进回路，tendon_model.py）")
parser.add_argument("--tendon-mu", type=float, default=None, help="腱索摩擦系数覆盖")
parser.add_argument("--tendon-hyst", type=float, default=None, help="腱索滞回强度覆盖")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import kit  # noqa: E402

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
kit.mark("app launched")

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from PIL import Image  # noqa: E402

import fx  # noqa: E402

PART_PLANE_Z = 1.595  # 视觉平面 z，main() 内按试件几何覆盖（= PART_Z + PART_H/2）


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    global PART_PLANE_Z
    import arm_params as P
    if args_cli.part_r is not None:
        P.PART_R = float(args_cli.part_r)
    if args_cli.part_h is not None:
        P.PART_H = float(args_cli.part_h)
    if args_cli.part_z is not None:
        P.PART_Z = float(args_cli.part_z)
    elif args_cli.part_h is not None:
        P.PART_Z = 1.565 + (P.PART_H - 0.060) / 2.0
    part_xy = tuple(float(v) for v in args_cli.part_xy.split(","))
    part_mass = None
    if abs(P.PART_R - 0.014) > 1e-9 or abs(P.PART_H - 0.060) > 1e-9:
        part_mass = 0.037 * (P.PART_R / 0.014) ** 2 * (P.PART_H / 0.060)
    PART_PLANE_Z = P.PART_Z + P.PART_H / 2.0
    print("[al] 试件变体 R=%.4f H=%.4f Z=%.4f XY=(%.3f %.3f) 平面z=%.4f 质量=%s 掩膜阈值=%.5g 顶带系数=%.4f" % (
        P.PART_R, P.PART_H, P.PART_Z, part_xy[0], part_xy[1], PART_PLANE_Z,
        "0.037(默认)" if part_mass is None else "%.4f" % part_mass,
        0.0003 * (P.PART_R / 0.014) ** 2, 0.028 * (P.PART_R / 0.014)), flush=True)
    sim = SimulationContext(SimulationCfg(dt=0.001, device="cuda:0"))
    fx.build_studio(sim_utils)
    from scene_embodied import spawn_drone, spawn_part, spawn_ground_and_stand, DroneUnit, enable_contact_reporters
    spawn_ground_and_stand()
    part = spawn_part(xy=part_xy, mass=part_mass)
    arm = spawn_drone("/World/DroneA", "drone_a_diamond", (0.0, 0.0, 2.05))
    fx.clear_instancing("/World/DroneA")
    cr_paths = enable_contact_reporters("/World/DroneA", r"finger_[0-9]+")
    cr_paths += enable_contact_reporters("/World", r"part")
    print("[al] 指垫接触回报 API 挂载 %d 个 %s（PhysX 求解器接触力读取前置）" % (len(cr_paths), cr_paths), flush=True)

    ce = (0.9, -1.1, 2.0)
    ct = (0.0, 0.0, 1.7)
    cam_c = Camera(CameraCfg(
        prim_path="/World/camC", update_period=1.0 / 30.0, height=180, width=320, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=32.0, focus_distance=1.2,
                                         horizontal_aperture=24.0, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=ce, rot=fx.dir_quat(ce, ct), convention="world")))
    cf_sensors = []
    for _cp in cr_paths:
        cf_sensors.append(ContactSensor(ContactSensorCfg(
            prim_path=_cp, update_period=0.0,
            filter_prim_paths_expr=["/World/part"], max_contact_data_count_per_prim=16)))
    sim.reset()
    fx.bind_all()
    for _ in range(140):
        sim.step()
    kit.mark("reset + materials + camera")

    eye_off = np.array([float(x) for x in args_cli.eye.split(",")])
    gd0 = next(g for g in P.GRIPPER_DEFS if g["key"] == "diamond")
    pad_bottom_off = gd0["col"][4] - gd0["col"][2]
    if args_cli.rigid:
        P.DRIVES["bend"]["stiffness"] = 14.0
        print("[al] 刚臂对照 弯曲刚度 14", flush=True)
    import spec as S
    import estimator as E
    E.MAHONY_GAIN = float(args_cli.est_g)
    S.KP_VEL = float(args_cli.kp_vel)
    S.KP_POS = float(args_cli.kp_pos)
    S.KI_ATT_M = float(args_cli.ki_att)
    S.KP_ATT = float(args_cli.kp_att)
    S.KI_VEL = float(args_cli.ki_vel)
    S.KD_RATE = float(args_cli.kd_rate)
    S.KP_RATE = float(args_cli.kp_rate)
    print("[al] 增益覆盖 KP_VEL=%.2f KP_POS=%.2f KP_RATE=12 姿态增益=%.2f 姿态积分=%.3f" % (
        S.KP_VEL, S.KP_POS, E.MAHONY_GAIN, S.KI_ATT_M), flush=True)
    ua = DroneUnit(arm, "A", np.random.default_rng(int(args_cli.seed)), has_arm=True,
                   use_fusion=False, tendon=args_cli.tendon)
    if args_cli.est == "ekf":
        from estimator_ekf import EkfEstimator
        ua.est = EkfEstimator(ua.est.pos.copy(), 0.001)
        ua.use_fusion = True
        print("[al] 估计器切换 EKF", flush=True)
    ua.fc.L_DOB = float(args_cli.dob_l)
    print("[al] 扰动观测器 L=%.1f" % ua.fc.L_DOB, flush=True)
    if cf_sensors:
        ua.bind_contact_sensors(cf_sensors)
        print("[al] 指垫接触力传感器 %d 个 初始化=%s（过滤 /World/part）" % (
            len(cf_sensors), [bool(s.is_initialized) for s in cf_sensors]), flush=True)
    dist_rng = np.random.default_rng(int(args_cli.seed) + 977)
    dist_on = args_cli.dist != "none"
    if dist_on:
        print("[dist] 场景=%s 风F=%.2fN az=%.0fdeg 下洗F=%.2fN t0=%.1f dur=%.2fs 视觉sigma=%.1fpx drop=%.2f seed=%d" % (
            args_cli.dist, args_cli.wind_f, args_cli.wind_az, args_cli.dw_f,
            args_cli.dw_t0, args_cli.dw_dur, args_cli.vis_sigma, args_cli.vis_drop,
            int(args_cli.seed)), flush=True)
    ua.cg_damp_t = 0.0
    ts = torch.full((1, len(ua.bend_ids)), 10.0, device=arm.device)
    td = torch.full((1, len(ua.bend_ids)), 4.0, device=arm.device)
    if args_cli.tendon:
        ua.tendon.kp = 10.0
        ua.tendon.kd = 4.0
        if getattr(args_cli, "tendon_mu", None) is not None:
            ua.tendon.mu = float(args_cli.tendon_mu)
        if getattr(args_cli, "tendon_hyst", None) is not None:
            ua.tendon.hyst = float(args_cli.tendon_hyst)
        ua.tendon.reset()
        print("[al] 腱索等效模式 弯曲力经张力注入 k=10 c=4(张力层) 预紧=%.1fN 摩擦=%.2f 滞回=%.1f"
              % (ua.tendon.preload, ua.tendon.mu, ua.tendon.hyst), flush=True)
    else:
        arm.write_joint_stiffness_to_sim(ts, joint_ids=ua.bend_ids)
        arm.write_joint_damping_to_sim(td, joint_ids=ua.bend_ids)
        print("[al] 弯曲伺服覆盖 k=10 c=1.0", flush=True)
    jn_all = list(arm.joint_names)
    j_bx0 = jn_all.index("bx_0")
    j_bx7 = jn_all.index("bx_7")
    _jname = "bend_jac_tendon.json" if args_cli.tendon else "bend_jac.json"
    if args_cli.tendon and not (HERE / _jname).exists():
        print("[al] 腱索 J 未标定，回退标称 J", flush=True)
        _jname = "bend_jac.json"
    jd = json.loads((HERE / _jname).read_text(encoding="utf-8"))
    J_bend = np.array(jd["J_bend"])
    if not args_cli.no_ff:
        ua.cg_jac = np.array(jd["J_cg"])
        print("[al] 前馈开启 质心雅可比 %s" % np.round(ua.cg_jac, 3).tolist(), flush=True)
    else:
        print("[al] 前馈关闭", flush=True)
    print("[al] 弯曲雅可比 %s" % np.round(J_bend, 2).tolist(), flush=True)
    wi = arm.body_names.index("wrist")
    dist_ids = [i for i, n in enumerate(arm.body_names) if "dist" in n]
    if not dist_ids:
        dist_ids = [i for i, n in enumerate(arm.body_names) if n.startswith("finger")]
    out = Path(args_cli.out)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)

    import cv2

    def mask_top_centroid(rgb, color):
        img = np.asarray(rgb, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[..., 0].astype(int)
        S = hsv[..., 1].astype(int)
        V = hsv[..., 2].astype(int)
        if color == "orange":
            m = (H >= 5) & (H <= 45) & (S > 60) & (V > 80)
        elif color == "red":
            m = ((H <= 5) | (H >= 175)) & (S > 60) & (V > 60)
        else:
            m = (H >= 95) & (H <= 130) & (S > 60) & (V > 50)
        hh, ww = img.shape[:2]
        m[:int(0.12 * hh), :] = False
        m[int(0.88 * hh):, :] = False
        m[:, :int(0.12 * ww)] = False
        m[:, int(0.88 * ww):] = False
        if float(m.mean()) < 0.0003 * (P.PART_R / 0.014) ** 2:
            return None
        ys, xs = np.nonzero(m)
        y0 = ys.min()
        k = max(4, int(round(0.028 * (P.PART_R / 0.014) * img.shape[1])))
        top = ys <= (y0 + k)
        c = np.array([float(xs[top].mean()), float(ys[top].mean())])
        if color == "orange" and dist_on:
            if dist_rng.random() < args_cli.vis_drop:
                return None
            c = c + dist_rng.normal(0.0, args_cli.vis_sigma, 2)
        return c

    def localize(rgb):
        c = mask_top_centroid(rgb, "orange")
        if c is None:
            return None
        h, w = rgb.shape[:2]
        f = 32.0 / 24.0 * w
        d_local = np.array([(c[0] - 0.5 * w) / f, -(c[1] - 0.5 * h) / f, -1.0])
        d_world = quat_to_mat(q_c_last) @ d_local
        if d_world[2] < -1e-6:
            tt = (PART_PLANE_Z - eye_c_last[2]) / d_world[2]
            if tt > 0:
                return eye_c_last[:2] + d_world[:2] * tt
        return None

    def tip_xy():
        return arm.data.body_pos_w[0, dist_ids].detach().cpu().numpy().mean(axis=0)[:2]

    dt = 0.001
    total = args_cli.total
    close_t = None
    descend_t = None
    settle_t = None
    realign = False
    damp_t = -1.0
    err_base = np.zeros(2)
    contact_hold = 0.0
    close_streak = 0
    print_t = -1.0
    bend = np.array([0.0, 0.0])
    bend_target = np.array([0.0, 0.0])
    bend_int = np.array([0.0, 0.0])
    base_xy = np.array([0.0, 0.0])
    base_done = False
    align_t = 9.0
    bias = None
    bias_samples = []
    err_lp = np.zeros(2)
    err_hold_t = 0.0
    last_est = None
    est_f = None
    v_tip_f = np.zeros(2)
    VIS_ALPHA = 0.40
    TAU_LOOK = 0.15
    bend_cmd = np.array([0.0, 0.0])
    err_log = []
    part_log = []
    contact_log = []
    cf_rows = []
    cf_fail_warned = False
    stills = {3.0: "settled", 8.5: "extended", 12.0: "aligning"}
    done = set()
    eye_c_last = np.array([0.0, 0.0, 0.0])
    q_c_last = np.array([0.0, 0.0, 0.0, 1.0])
    final_err_align = None
    wind_on = False
    dw_started = False
    dw_ended = False
    ZVD_A = None
    if args_cli.zvd:
        _zT = 1.0 / float(args_cli.zvd_f)
        _zK = float(np.exp(-float(args_cli.zvd_z) * np.pi / np.sqrt(1.0 - float(args_cli.zvd_z) ** 2)))
        _zden = 1.0 + 2.0 * _zK + _zK * _zK
        ZVD_A = (1.0 / _zden, 2.0 * _zK / _zden, _zK * _zK / _zden, _zT)
        print("[al] ZVD 输入整形 f=%.2fHz zeta=%.2f T=%.3fs" % (args_cli.zvd_f, args_cli.zvd_z, _zT), flush=True)

    def e_nom(tt):
        if tt < 6.0:
            return 0.15 * min(max(tt, 0.0) / 3.0, 1.0)
        if tt < 9.0:
            return 0.15 + 0.47 * min((tt - 6.0) / 3.0, 1.0)
        return 0.62

    for k in range(int(total / dt)):
        t = k * dt
        if close_t is None and descend_t is None:
            cz = 2.15
        elif close_t is None:
            u = min((t - descend_t) / 5.0, 1.0)
            cz = 2.15 - 0.146 * 0.5 * (1.0 - np.cos(np.pi * u))
            if u >= 1.0:
                if settle_t is None:
                    settle_t = t
                    print("[al] 下降完成 稳定等待 t=%.1f" % t, flush=True)
                elif t - settle_t >= 2.5 and k % 100 == 0:
                    tip_now = tip_xy()
                    v_tip_gate = v_tip_f
                    tgt_now = last_est if last_est is not None else np.zeros(2)
                    e_now = float(np.linalg.norm(tgt_now - tip_now))
                    slow = float(np.linalg.norm(v_tip_gate)) < 0.15
                    if e_now < args_cli.gate and slow:
                        close_streak += 1
                    else:
                        close_streak = 0
                    if close_streak >= 3:
                        close_t = t
                        realign = False
                        final_err_align = float(np.linalg.norm(err_lp))
                        print("[al] 稳定完成 末检误差=%.3f 连续%d拍 开始闭合 t=%.1f" % (e_now, close_streak, t), flush=True)
                    elif t - settle_t >= 8.0 and e_now < 0.005:
                        close_t = t
                        realign = False
                        final_err_align = float(np.linalg.norm(err_lp))
                        print("[al] 复核通过 误差=%.3f 闭合 t=%.1f" % (e_now, t), flush=True)
                    elif t - settle_t >= 15.0:
                        close_t = t
                        realign = False
                        final_err_align = float(np.linalg.norm(err_lp))
                        print("[al] 对位超时 误差=%.3f 强制闭合 t=%.1f" % (e_now, t), flush=True)
                    elif not realign and not args_cli.no_realign:
                        realign = True
                        print("[al] 末检误差=%.3f 重新对位" % e_now, flush=True)
        else:
            if t - close_t > 7.0:
                cz = 2.004 + 0.30 * min((t - close_t - 7.0) / 3.0, 1.0)
            else:
                cz = 2.004
        if close_t is None:
            if ZVD_A is not None:
                _a1, _a2, _a3, _zT = ZVD_A
                e = _a1 * e_nom(t) + _a2 * e_nom(t - _zT / 2.0) + _a3 * e_nom(t - _zT)
            else:
                e = e_nom(t)
        bend = bend + np.clip(bend_cmd - bend, -0.90 * dt, 0.90 * dt)
        if 5.0 <= t < 5.9 and k % 200 == 0:
            rgb = cam_c.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
            est = localize(rgb)
            if est is not None:
                bias_samples.append(est - np.array(part_xy))
            if t >= 5.8 and bias_samples and bias is None:
                bias = np.mean(np.stack(bias_samples), axis=0)
                print("[al] 视觉偏差标定 %s 样本数 %d" % (np.round(bias, 3).tolist(), len(bias_samples)), flush=True)
        hold_phase = close_t is not None or descend_t is not None
        cad = 0.1
        if t >= 9.0 and t - align_t >= cad and not ua.hold:
            align_t = t
            rgb = cam_c.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
            est = localize(rgb)
            if est is not None and last_est is not None and float(np.linalg.norm(est - last_est)) > 0.03:
                est = None
            tip = tip_xy()
            v_tip_now = arm.data.body_lin_vel_w[0, dist_ids].mean(dim=0).cpu().numpy()[:2]
            v_tip_f = v_tip_f + (v_tip_now - v_tip_f) * min(cad / 0.05, 1.0)
            gt = part.data.root_pos_w[0].detach().cpu().numpy()
            err = None
            if est is not None and bias is not None:
                est = est - bias
                last_est = est.copy()
                err = est - tip
                err_log.append((t, float(np.linalg.norm(est - gt[:2])), err.copy()))
            elif bias is not None:
                tgt = last_est if last_est is not None else np.zeros(2)
                err = tgt - tip
                err_log.append((t, float(np.linalg.norm(gt[:2])), err.copy()))
            if err is not None:
                a = min(cad / 8.0, 1.0)
                err_lp = (1.0 - a) * err_lp + a * err
                ab = min(cad / 10.0, 1.0)
                err_base = (1.0 - ab) * err_base + ab * err
                if close_t is None and descend_t is None and float(np.linalg.norm(err)) < 0.004:
                    err_hold_t += cad
                else:
                    err_hold_t = 0.0
                if close_t is None and descend_t is None and err_hold_t >= 0.4:
                    descend_t = t
                    print("[al] 对位完成 误差=%.3f m 开始下降 t=%.1f 弯曲=(%.3f %.3f)" % (
                        float(np.linalg.norm(err_lp)), t, bend[0], bend[1]), flush=True)
                lam = 1.0
                Jinv = J_bend.T @ np.linalg.inv(J_bend @ J_bend.T + lam * lam * np.eye(2))
                err_use = err_base if not realign else err
                if not ua.hold and (realign or float(np.linalg.norm(err)) > 0.002):
                    if not args_cli.base_only:
                        KP_BEND = 0.9 if not hold_phase else 1.1
                        db = KP_BEND * (Jinv @ err_use)
                        bend_target = np.clip(db, -0.050, 0.050)
                    if float(np.linalg.norm(err_base)) > 0.010:
                        base_xy = np.clip(base_xy + np.clip(0.05 * err_base, -0.002, 0.002), -0.05, 0.05)
                if t - print_t >= 0.3:
                    print_t = t
                    lse = last_est if last_est is not None else np.zeros(2)
                    print("[al] 对位 t=%.1f 试件=(%.3f %.3f) 指尖=(%.3f %.3f) 误差=(%.3f %.3f) 慢误=(%.3f %.3f) 弯曲=(%.3f %.3f) 基座=(%.3f %.3f) 定位误差=%.3f" % (
                        t, lse[0], lse[1], tip[0], tip[1], err[0], err[1], err_lp[0], err_lp[1],
                        bend[0], bend[1], base_xy[0], base_xy[1],
                        float(np.linalg.norm(lse - gt[:2]))), flush=True)
        if t >= 5.0 and t - damp_t >= 0.1 and not ua.hold:
            damp_t = t
            lam_d = 1.0
            Jinv_d = J_bend.T @ np.linalg.inv(J_bend @ J_bend.T + lam_d * lam_d * np.eye(2))
            v_tip = arm.data.body_lin_vel_w[0, dist_ids].mean(dim=0).cpu().numpy()[:2]
            v_root = arm.data.root_lin_vel_w[0].cpu().numpy()[:2]
            v_rel = v_tip - v_root
            damp = np.zeros(2) if args_cli.no_damp else np.clip(
                -args_cli.damp_k * (Jinv_d @ v_rel), -0.08, 0.08)
            bend_cmd = np.clip(bend_target + damp, -0.10, 0.10)
            if t >= 9.0:
                ep = getattr(ua, "last_est", None)
                ev = np.asarray(ep["vel"], dtype=float).reshape(-1)[:2] if ep is not None else np.zeros(2)
                qa = arm.data.joint_pos[0, ua.bend_ids].detach().cpu().numpy()
                bx_a = float(np.mean(qa[:P.N_SEG]))
                by_a = float(np.mean(qa[P.N_SEG:]))
                qfull = arm.data.joint_pos[0].detach().cpu().numpy()
                bx0_v = float(qfull[j_bx0])
                bx7_v = float(qfull[j_bx7])
                tl = getattr(ua, "tendon_last", None)
                tstr = ""
                if tl is not None:
                    tstr = " T=(%.2f %.2f %.2f) Tbar=%.2f 滞回=(%.4f %.4f)" % (
                        tl["T"][0], tl["T"][1], tl["T"][2], tl["T_bar"],
                        tl["hyst"][0], tl["hyst"][1])
                print("[dp] t=%.1f vrel=(%.3f %.3f) damp=(%.3f %.3f) vtip=(%.3f %.3f) vroot=(%.3f %.3f) bend=(%.3f %.3f) cmd=(%.3f %.3f) 实=(%.3f %.3f) bx0=%.3f bx7=%.3f%s" % (
                    t, v_rel[0], v_rel[1], damp[0], damp[1], v_tip[0], v_tip[1],
                    v_root[0], v_root[1], bend[0], bend[1], bend_cmd[0], bend_cmd[1],
                    bx_a, by_a, bx0_v, bx7_v, tstr), flush=True)
        if close_t is None and descend_t is None and t > 40.0:
            if float(np.linalg.norm(err_lp)) > 0.05:
                print("[al] 对位超时 误差 %.3f 超限 提前终止 FAIL 眼位=%s 弯曲=%.2f" % (
                    float(np.linalg.norm(err_lp)), args_cli.eye, float(np.max(np.abs(bend)))), flush=True)
                return 0
            descend_t = t
            print("[al] 对位超时 直接下降", flush=True)
        if close_t is not None:
            dtc = t - close_t
            if dtc < 2.0:
                g = min(dtc / 2.0, 1.0)
            elif dtc < 3.5:
                g = 1.0
            elif dtc < 5.5:
                e = max(0.62 - (dtc - 3.5) * 0.16, 0.30)
                g = 1.0
            else:
                e = 0.30
                g = 1.0
        else:
            g = 0.0

        cfs = ua.read_grip_forces(dt)
        if cfs is None or cfs.get("fn", 0.0) <= 0.0:
            try:
                _ncf = part.root_physx_view.get_net_contact_forces(dt)
                _v = _ncf.detach().cpu().numpy()[0].reshape(-1, 3).sum(axis=0)
                _mag = float(np.linalg.norm(_v))
                if _mag > 0.005:
                    cfs = {"ncon": 2, "fn": _mag, "ft": 0.0,
                           "pads": [(_mag, 0.0, 1, _mag), (_mag, 0.0, 1, _mag)]}
                elif cfs is None:
                    cfs = {"ncon": 0, "fn": 0.0, "ft": 0.0, "pads": []}
            except Exception:
                pass
        if cfs is None and cf_sensors and not cf_fail_warned:
            cf_fail_warned = True
            print("[al] 接触力读取失败，回退几何接触判据", flush=True)
        ncon = ua.grip_contacts()
        ncon_px = int(cfs["ncon"]) if cfs is not None else ncon
        if t > 1.0:
            if contact_log and ncon != contact_log[-1]:
                src = "PhysX" if cfs is not None else "几何"
                print("[al] t=%.2f 接触数(%s) %d -> %d" % (t, src, contact_log[-1], ncon), flush=True)
            contact_log.append(ncon)
        if close_t is not None and cfs is not None:
            pads = cfs["pads"]
            p0 = pads[0] if len(pads) > 0 else (float("nan"),) * 4
            p1 = pads[1] if len(pads) > 1 else (float("nan"),) * 4
            cf_rows.append([t, "close" if (t - close_t) < 7.0 else "lift",
                            p0[0], p0[1], p0[3], p0[2], p1[0], p1[1], p1[3], p1[2],
                            cfs["fn"], cfs["ft"], cfs["ncon"],
                            float(part.data.root_pos_w[0, 2].detach().cpu())])
        if ncon >= 2:
            contact_hold += dt
        else:
            contact_hold = 0.0
        if close_t is not None and 0.8 <= (t - close_t) < 6.0 and not ua.hold:
            if contact_hold >= 0.3 or args_cli.no_contact:
                ua.attach_part(part)
                ua.set_grip_stiffness(60.0)
                ua.set_bend_stiffness(20.0)
                print("[al] t=%.2f 双指接触持续 %.2fs 机械捕获 刚度调度 夹爪60 弯曲20" % (t, contact_hold), flush=True)
        ua.dist_force_w[:] = 0.0
        ua.dist_torque_w[:] = 0.0
        if args_cli.dist == "wind":
            if t >= 1.0:
                _az = np.radians(args_cli.wind_az)
                ua.dist_force_w[:] = (args_cli.wind_f * np.cos(_az), args_cli.wind_f * np.sin(_az), 0.0)
                if not wind_on:
                    wind_on = True
                    print("[dist] t=%.2f 恒定风扰施加 力=(%.3f %.3f 0)N" % (
                        t, ua.dist_force_w[0], ua.dist_force_w[1]), flush=True)
        elif args_cli.dist == "downwash":
            if args_cli.dw_t0 <= t < args_cli.dw_t0 + args_cli.dw_dur:
                ua.dist_force_w[2] = -args_cli.dw_f
                if not dw_started:
                    dw_started = True
                    print("[dist] t=%.2f 下洗脉冲开始 %.2fN 向下 持续%.2fs" % (
                        t, args_cli.dw_f, args_cli.dw_dur), flush=True)
            elif dw_started and not dw_ended and t >= args_cli.dw_t0 + args_cli.dw_dur:
                dw_ended = True
                print("[dist] t=%.2f 下洗脉冲结束" % t, flush=True)
        st = ua.step(dt, np.array([base_xy[0], base_xy[1], cz]), 0.0,
                     arm_extend=e, arm_grip=g, arm_bend=[float(bend[0]), float(bend[1])])
        need_render = (k % 33 == 0)
        sim.step(render=need_render)
        ua.refresh(dt)
        if need_render:
            cam_c.update(1.0 / 30.0)
        part_log.append(part.data.root_pos_w[0].detach().cpu().numpy().copy())
        if k % 33 == 0:
            wp = arm.data.body_pos_w[0, wi].detach().cpu().numpy()
            eye_c_last = np.array([wp[0] + eye_off[0], wp[1] + eye_off[1], wp[2] + eye_off[2]])
            tgt_c_last = np.array([0.0, 0.0, PART_PLANE_Z])
            q_c_last = fx.dir_quat(eye_c_last, tgt_c_last)
            cam_c.set_world_poses_from_view(
                torch.tensor([list(eye_c_last)], dtype=torch.float32, device="cuda:0"),
                torch.tensor([list(tgt_c_last)], dtype=torch.float32, device="cuda:0"))
        if k % 500 == 0:
            p = np.asarray(st["pos"], dtype=float).reshape(-1)
            ep = getattr(ua, "last_est", None)
            epos = np.asarray(ep["pos"], dtype=float).reshape(-1) if ep is not None else np.full(3, np.nan)
            evel = np.asarray(ep["vel"], dtype=float).reshape(-1) if ep is not None else np.full(3, np.nan)
            bp_w = arm.data.body_pos_w[0, wi].detach().cpu().numpy()
            fp_w = arm.data.body_pos_w[0, dist_ids].detach().cpu().numpy()
            wz = float(bp_w[2])
            fz = float(fp_w[:, 2].mean())
            print("[al] t=%.1f 位置=(%.3f %.3f %.3f) 估计=(%.3f %.3f) 漂移=(%.3f %.3f) 倾角=%.1f 弯曲=(%.3f %.3f) 前馈=(%.3f %.3f) 腕z=%.3f 指z=%.3f 底z=%.3f ki=(%.2f %.2f) v估=(%.2f %.2f) a_des=(%.2f %.2f) att=(%.3f %.3f) rv=(%.3f %.3f)" % (
                t, float(p[0]), float(p[1]), float(p[2]), float(epos[0]), float(epos[1]),
                float(p[0] - epos[0]), float(p[1] - epos[1]),
                float(ua.tilt_deg()), bend[0], bend[1], float(ua.fc.extra_moment[0]), float(ua.fc.extra_moment[1]),
                wz, fz, fz + pad_bottom_off,
                float(ua.fc.ki_state[0]), float(ua.fc.ki_state[1]),
                float(evel[0]), float(evel[1]),
                float(ua.fc.a_des_last[0]), float(ua.fc.a_des_last[1]),
                float(ua.fc.att_trim[0]), float(ua.fc.att_trim[1]),
                float(ua.fc.rv_last[0]), float(ua.fc.rv_last[1])), flush=True)
        if k % 30 == 0:
            for ts, tag in stills.items():
                if tag not in done and abs(t - ts) < 0.02:
                    rgba = cam_c.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
                    Image.fromarray(rgba).save(out / ("align_%s.png" % tag))
                    done.add(tag)
            if close_t is not None and "gripped" not in done and abs((t - close_t) - 1.5) < 0.03:
                rgba = cam_c.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
                Image.fromarray(rgba).save(out / "align_gripped.png")
                done.add("gripped")
            if close_t is not None and "lifted" not in done and abs((t - close_t) - 9.0) < 0.03:
                rgba = cam_c.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)[..., :3]
                Image.fromarray(rgba).save(out / "align_lifted.png")
                done.add("lifted")
        if not np.isfinite(st["pos"]).all() or float(np.linalg.norm(st["pos"])) > 3.0:
            print("[al] 发散保护 t=%.1f" % t, flush=True)
            break

    W_PART = 0.037 * 9.81
    if cf_rows:
        csv_path = out / "contact_force.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            wcsv = csv.writer(fcsv)
            wcsv.writerow(["t", "phase", "fn_0", "ft_0", "fnsum_0", "nc_0",
                           "fn_1", "ft_1", "fnsum_1", "nc_1",
                           "fn_tot", "ft_tot", "ncon", "z_part"])
            wcsv.writerows(cf_rows)
        t_lo, t_hi = cf_rows[0][0], cf_rows[-1][0]
        close_win = [r for r in cf_rows if close_t <= r[0] <= close_t + 3.0]
        l_lo, l_hi = close_t + 10.0, min(close_t + 14.0, t_hi)
        if l_hi - l_lo < 0.5:
            l_lo, l_hi = max(close_t + 9.0, t_hi - 2.0), t_hi
        lift_win = [r for r in cf_rows if l_lo <= r[0] <= l_hi]
        if close_win:
            fn_close = np.array([r[10] for r in close_win])
            sm = max(1, int(round(0.05 / dt)))
            fn_close_sm = np.convolve(fn_close, np.ones(sm) / sm, mode="valid")
            k_peak = int(np.argmax(fn_close))
            fn_peak = float(fn_close[k_peak])
            fn_peak_sm = float(np.max(fn_close_sm)) if len(fn_close_sm) else float("nan")
            ft_peak = float(close_win[k_peak][11])
        else:
            fn_peak = fn_peak_sm = ft_peak = float("nan")
        if lift_win:
            fn_lift = np.array([r[10] for r in lift_win])
            ft_lift = np.array([r[11] for r in lift_win], dtype=float)
            nc_lift = np.array([r[12] for r in lift_win])
            fn_lift_m, fn_lift_s = float(fn_lift.mean()), float(fn_lift.std())
            ft_lift_m, ft_lift_s = float(np.nanmean(ft_lift)), float(np.nanstd(ft_lift))
            nc_lift_m = float(nc_lift.mean())
            slip_pads = []
            for i_f, i_t in ((2, 3), (6, 7)):
                mask = np.array([r[i_f] > 0.05 for r in lift_win])
                f0 = np.array([r[i_f] for r in lift_win])[mask]
                f1 = np.array([r[i_t] for r in lift_win], dtype=float)[mask]
                if len(f0) > 10:
                    ratio = f1 / f0
                    slip_pads.append((float(np.mean(ratio)), float(np.max(ratio))))
                else:
                    slip_pads.append((float("nan"), float("nan")))
            slip_tot = float(np.nanmean(ft_lift) / fn_lift_m) if fn_lift_m > 1e-9 else float("nan")
        else:
            fn_lift_m = fn_lift_s = ft_lift_m = ft_lift_s = nc_lift_m = float("nan")
            slip_pads = [(float("nan"), float("nan"))] * 2
            slip_tot = float("nan")
        print("[cf] 试件重力 %.3f N  闭合段 t=%.1f..%.1f 峰值|Fn|=%.3f N（50ms平滑 %.3f）峰值时刻|Ft|=%.3f N" % (
            W_PART, close_t, close_t + 3.0, fn_peak, fn_peak_sm, ft_peak), flush=True)
        print("[cf] 提升稳态窗口 t=%.1f..%.1f |Fn|=%.3f±%.3f N（%.2f W）|Ft|=%.3f±%.3f N（%.2f W）接触数=%.2f" % (
            l_lo, l_hi, fn_lift_m, fn_lift_s, fn_lift_m / W_PART,
            ft_lift_m, ft_lift_s, ft_lift_m / W_PART, nc_lift_m), flush=True)
        print("[cf] 滑移比 |Ft|/|Fn|（μ=2.5）：指0 均值 %.2f 峰 %.2f  指1 均值 %.2f 峰 %.2f  合计 %.2f" % (
            slip_pads[0][0], slip_pads[0][1], slip_pads[1][0], slip_pads[1][1], slip_tot), flush=True)
        print("[cf] CSV %d 行 t=%.3f..%.3f -> %s" % (len(cf_rows), t_lo, t_hi, csv_path), flush=True)
    else:
        print("[cf] 无接触力数据（闭合未发生或传感器不可用）", flush=True)

    z_le = float(np.mean([p[2] for p in part_log[2500:3500]]))
    z_end = float(np.mean([p[2] for p in part_log[-1200:]]))
    max_contact = max(contact_log) if contact_log else 0
    loc_err = float(np.mean([g for _, g, _ in err_log])) if err_log else 9.9
    final_err = float(final_err_align) if final_err_align is not None else (
        float(np.linalg.norm(err_log[-1][2])) if err_log else 9.9)
    ok = (z_end - z_le) > 0.08 and max_contact >= 1 and final_err < 0.02 and ua.hold
    print("[al] 判据 提升 %.3f 接触 %d 末段误差 %.3f 定位均值误差 %.3f -> %s  眼位=%s 弯曲=%.2f 场景=%s" % (
        z_end - z_le, max_contact, final_err, loc_err, "PASS" if ok else "FAIL",
        args_cli.eye, float(np.max(np.abs(bend))), args_cli.dist), flush=True)
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
