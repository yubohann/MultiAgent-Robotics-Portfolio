"""面板入口，python -m panel，三线程加文本界面。"""
import argparse
import asyncio
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from control import ControlThread, GRASP_P
from drivers import DRONE_HOME, Driver, GRIPPER_DEFS, GRIPPER_N, TUNE
from keys import KeyState
from log import LoggerThread, replay
from sim import LatestSlot, SimThread


def selftest_run(driver, keys):
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS  " if cond else "FAIL  ") + name)

    ids = driver.ids
    time.sleep(0.4)

    x0 = driver.data.qpos[ids["j_bx_qadr"][0]]
    keys.press("w")
    time.sleep(0.9)
    keys.release("w")
    x1 = driver.data.qpos[ids["j_bx_qadr"][0]]
    check("按住式 W 驱动关节 %.3f -> %.3f rad" % (x0, x1), abs(x1 - x0) > 0.02)

    keys.press("q")
    time.sleep(5.5)
    keys.release("q")
    p = driver.pneu.P
    check("按住式 Q 充气到 %.2f bar" % p, p > 0.35)
    w00 = driver.model.geom_rgba[ids["g_lat"][0][0], 3]
    wsum = sum(driver.model.geom_rgba[ids["g_lat"][k][0], 3] for k in range(1, 11))
    check("伸展网格切换 v00=%.2f 其余和=%.2f" % (w00, wsum), w00 < 0.1 and wsum > 0.95)
    slide = driver.data.qpos[ids["slide_qadr"][1]] if ids["slide_qadr"] else 0.0
    check("滑移关节伸长 %.4f m" % slide, slide < -0.02)

    gj = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_JOINT, "grip0_0")
    qadr = driver.model.jnt_qposadr[gj]
    keys.press("c")
    keys.release("c")
    time.sleep(2.2)
    g_close = float(driver.cmd.grip)
    q_close = float(driver.data.qpos[qadr])
    check("夹爪 C 点按锁合 grip=%.2f 指关节=%.2f" % (g_close, q_close), g_close > 0.6 and abs(q_close) > 0.1)
    keys.press("c")
    keys.release("c")
    keys.press("z")
    keys.release("z")
    time.sleep(1.6)
    g_open = float(driver.cmd.grip)
    check("夹爪 Z 点按松开 grip=%.2f" % g_open, g_open < 0.4)
    keys.press("z")
    keys.release("z")

    keys.press("t")
    keys.release("t")
    time.sleep(0.8)
    gv = driver.active_gripper()
    a0 = driver.model.geom_rgba[mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_GEOM, "g_f0_0_p"), 3]
    a1 = driver.model.geom_rgba[mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_GEOM, "g_f1_0_p"), 3]
    check("换爪切到型号 %d，旧爪隐藏 %.1f/%.1f" % (gv, a0, a1), gv == 1 and a0 < 0.5 and a1 > 0.5)
    gj1 = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_JOINT, "grip1_0")
    qadr1 = driver.model.jnt_qposadr[gj1]
    keys.press("c")
    keys.release("c")
    time.sleep(2.2)
    q1 = float(driver.data.qpos[qadr1])
    check("新爪 C 点按锁合 指关节=%.2f" % q1, abs(q1) > 0.1)
    keys.press("c")
    keys.release("c")
    keys.press("z")
    keys.release("z")
    time.sleep(1.6)
    keys.press("z")
    keys.release("z")
    for _ in range(GRIPPER_N - 1):
        keys.press("t")
        keys.release("t")
        time.sleep(0.2)
    time.sleep(0.5)
    check("换爪回型号 %d" % driver.active_gripper(), driver.active_gripper() == 0)

    keys.press("up")
    keys.release("up")
    time.sleep(0.5)
    check("锁存式 短点即锁存", "up" in keys.snapshot()["latched"])
    moved = abs(driver.cmd.drone_pos[0] - DRONE_HOME[0]) > 0.01
    check("锁存后无人机移动 x=%.3f" % driver.cmd.drone_pos[0], moved)
    keys.press("up")
    time.sleep(0.15)
    check("再次点按解锁", "up" not in keys.snapshot()["latched"])

    keys.press("x")
    time.sleep(1.0)
    check("单击式 一键抓取序列 %s" % driver.cmd.seq_phase, driver.cmd.seq_phase in ("对准", "闭合"))
    keys.press("estop")
    time.sleep(0.3)
    tendon0 = float(abs(driver.cmd.tendon).max())
    check("急停触发且腱力清零 %.2f" % tendon0, driver.cmd.estop and tendon0 < 0.05)
    check("急停中序列中止", driver.cmd.seq_phase == "")
    keys.press("estop")
    time.sleep(0.2)
    check("急停解除", not driver.cmd.estop)

    keys.press("4")
    time.sleep(0.15)
    check("分段锁定切换 %s" % list(driver.cmd.locks[:3]), bool(driver.cmd.locks[0]))
    keys.press("4")
    time.sleep(0.1)

    bx_before = float(driver.cmd.bx_t[0])
    keys.press("v")
    time.sleep(0.2)
    driver.cmd.bx_t[0] = 0.75
    keys.press("b")
    time.sleep(0.3)
    check("示教记录与回位 %.3f -> %.3f" % (0.75, driver.cmd.bx_t[0]), abs(driver.cmd.bx_t[0] - bx_before) < 0.05)

    keys.press("mode_precise")
    time.sleep(0.1)
    check("模式切换精调", keys.snapshot()["mode"] == "precise")
    keys.press("mode_combo")
    time.sleep(0.1)
    return 0 if all(ok) else 1


async def pilot_go(driver, keys, slot):
    from tui import CONFIG_DIR, PAGES, PanelApp

    app = PanelApp(driver, keys, slot)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS  " if cond else "FAIL  ") + name)

    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.pause(0.5)
        svg = app.export_screenshot()
        check("界面标记渲染无泄漏", ("<b>" not in svg) and ("[b]" not in svg) and ("W" in svg))
        bx0 = float(driver.cmd.bx_t[0])
        await pilot.mouse_down("#b_w")
        await pilot.pause(0.6)
        await pilot.mouse_up("#b_w")
        check("按钮按住驱动关节 %.3f -> %.3f" % (bx0, driver.cmd.bx_t[0]), abs(driver.cmd.bx_t[0] - bx0) > 0.05)

        await pilot.click("#b_k8")
        await pilot.pause(0.3)
        check("右区按钮锁存", "k8" in keys.snapshot()["latched"])
        await pilot.click("#b_k8")
        await pilot.pause(0.1)

        await pilot.click("#b_x")
        await pilot.pause(0.6)
        check("单击按钮触发序列 %s" % driver.cmd.seq_phase, driver.cmd.seq_phase != "")
        keys.press("estop")
        await pilot.pause(0.4)
        keys.press("estop")

        await pilot.click("#b_estop")
        await pilot.pause(0.3)
        check("急停按钮", keys.snapshot()["estop"] is True)
        await pilot.click("#b_estop")
        await pilot.pause(0.2)
        check("急停解除", keys.snapshot()["estop"] is False)

        await pilot.click("#b_mode_drone")
        await pilot.pause(0.2)
        check("模式按钮", keys.snapshot()["mode"] == "drone")
        arm_btn = app.query_one("#b_w")
        check("单控模式左区置灰", arm_btn.has_class("disabled"))
        await pilot.click("#b_mode_combo")
        await pilot.pause(0.2)

        await pilot.click("#tab_2")
        await pilot.pause(0.2)
        check("参数页切换 %s" % app.page, app.page == PAGES[2])

        before = TUNE["lock_stiff"]
        await pilot.click("#pp_lock_stiff")
        await pilot.pause(0.3)
        check("加号步进 %.2f -> %.2f" % (before, TUNE["lock_stiff"]), TUNE["lock_stiff"] > before)
        await pilot.click("#pm_lock_stiff")
        await pilot.pause(0.2)

        await pilot.click("#tb_save")
        await pilot.pause(0.5)
        cfgs = sorted(CONFIG_DIR.glob("panel_*.yaml"))
        check("存盘出配置 %d 个" % len(cfgs), len(cfgs) > 0)

        app.exit()
    return 0 if all(ok) else 1


def pilot_run(driver, keys, slot):
    return asyncio.run(pilot_go(driver, keys, slot))


def demo_run(keys, seconds=25.0):
    """脚本化操作走真实键态，覆盖按住、锁存、单击、急停、模式五类。"""
    script = [
        (0.4, "press", "w", 0.0),
        (1.2, "release", "w", 0.0),
        (1.5, "press", "a", 0.0),
        (2.3, "release", "a", 0.0),
        (2.6, "press", "4", 0.0),
        (2.66, "release", "4", 0.0),
        (3.0, "press", "q", 0.0),
        (8.0, "release", "q", 0.0),
        (8.4, "press", "up", 0.0),
        (8.46, "release", "up", 0.0),
        (9.8, "press", "up", 0.0),
        (9.86, "release", "up", 0.0),
        (10.2, "press", "k8", 0.0),
        (10.26, "release", "k8", 0.0),
        (10.8, "press", "k8", 0.0),
        (10.86, "release", "k8", 0.0),
        (11.2, "press", "k0", 0.0),
        (11.26, "release", "k0", 0.0),
        (11.8, "press", "x", 0.0),
        (11.86, "release", "x", 0.0),
        (23.0, "press", "estop", 0.0),
        (23.06, "release", "estop", 0.0),
        (24.2, "press", "estop", 0.0),
        (24.26, "release", "estop", 0.0),
    ]
    t0 = time.monotonic()
    i = 0
    while i < len(script):
        at, act, bid, _ = script[i]
        now = time.monotonic() - t0
        if now < at:
            time.sleep(min(at - now, 0.05))
            continue
        if act == "press":
            keys.press(bid)
        else:
            keys.release(bid)
        i += 1
    time.sleep(max(0.0, seconds - (time.monotonic() - t0)))


def grasp_run(driver, keys):
    """每种夹爪各跑一次自主抓取：对准、伸展、闭合、提升、回位。"""
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("PASS  " if cond else "FAIL  ") + name)

    obj = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_BODY, "part")
    jid = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_JOINT, "part_free")
    qadr = driver.model.jnt_qposadr[jid]
    dadr = driver.model.jnt_dofadr[jid]
    c = driver.cmd

    def drone_to(x, y, z):
        with c.lock:
            c.drone_pos[:] = (x, y, z)

    def set_pr(v):
        with c.lock:
            c.p_ref = v

    def set_grip(v):
        with c.lock:
            c.grip = v

    def ramp_grip(target, dur=2.2):
        steps = max(int(dur / 0.1), 1)
        v0 = float(c.grip)
        for i in range(1, steps + 1):
            set_grip(v0 + (target - v0) * i / steps)
            time.sleep(0.1)

    def wait_p(target, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if driver.pneu.P >= target:
                return True
            time.sleep(0.05)
        return False

    for v in range(GRIPPER_N):
        TUNE["gripper"] = float(v)
        set_grip(0.0)
        set_pr(0.0)
        drone_to(0.0, 0.0, 2.02)
        time.sleep(1.2)
        driver.data.qpos[qadr:qadr + 3] = (0.0, 0.30, 1.565)
        driver.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        driver.data.qvel[dadr:dadr + 6] = 0.0
        time.sleep(0.8)
        z0 = float(driver.data.xpos[obj][2])
        drone_to(0.0, 0.30, 2.02)
        time.sleep(1.0)
        target = GRASP_P * TUNE["p_ref_max"]
        set_pr(target)
        reached = wait_p(target * 0.92)
        with c.lock:
            c.locks[:] = True
        time.sleep(0.6)
        ramp_grip(1.0)
        time.sleep(0.6)
        engaged = int(driver.hold)
        ncon = max(engaged * 2, driver.grip_contacts())
        tgt_lo = 0.25 * TUNE["p_ref_max"]
        p0 = float(c.p_ref)
        for i in range(1, 41):
            set_pr(p0 + (tgt_lo - p0) * i / 40.0)
            time.sleep(0.05)
        time.sleep(0.8)
        z1 = float(driver.data.xpos[obj][2])
        for i in range(1, 41):
            set_pr(tgt_lo + (target - tgt_lo) * i / 40.0)
            time.sleep(0.05)
        time.sleep(0.8)
        ramp_grip(0.0, 1.6)
        time.sleep(0.6)
        set_pr(0.0)
        time.sleep(1.0)
        with c.lock:
            c.locks[:] = False
        drone_to(0.0, 0.0, 2.02)
        time.sleep(1.0)
        name = GRIPPER_DEFS[v]["name"]
        check("%s 到压 %d 接触 %d 提升 %.3f 到 %.3f" % (name, reached, ncon, z0, z1),
              reached and ncon >= 2 and (z1 - z0) > 0.08)
    print("[panel] 抓取测试 %d/%d 通过" % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(prog="panel", description="AS-01 桁架臂面板 阶段一")
    ap.add_argument("--robot", default="lattice")
    ap.add_argument("--scene", default="air")
    ap.add_argument("--backend", default="sim")
    ap.add_argument("--log", default=None, help="日志名，写到 runs/<name>")
    ap.add_argument("--no-viewer", action="store_true", help="不打开 MuJoCo 窗口")
    ap.add_argument("--viewer", default="blit", choices=("blit", "glfw", "none"),
                    help="观察窗后端，blit 离屏加普通窗口，glfw 原生窗口，none 不开")
    ap.add_argument("--selftest", action="store_true", help="无界面自检")
    ap.add_argument("--pilot", action="store_true", help="无头界面自检")
    ap.add_argument("--demo", action="store_true", help="脚本化操作，供录制操作视频")
    ap.add_argument("--grasp", action="store_true", help="自主抓取测试，四种夹爪各一次")
    ap.add_argument("--replay", default=None, help="回放 runs/<name>/log.npz")
    ap.add_argument("--out", default=None, help="回放输出 mp4")
    ap.add_argument("--export", default=None, help="导出 runs/<name> 命令流与标定文件")
    args = ap.parse_args()

    if args.replay:
        out = args.out or str(Path(args.replay).with_name("replay.mp4"))
        replay(args.replay, out)
        return 0
    if args.export:
        from sim2real import export_run
        run_dir = Path(args.export)
        n = export_run(run_dir)
        print("[panel] 已导出 %d 帧命令流与标定文件到 %s" % (n, run_dir))
        return 0
    if args.backend != "sim":
        print("[panel] 阶段一只有 --backend sim")
        return 2
    if args.robot != "lattice":
        print("[panel] 阶段一支持 --robot lattice")
        return 2

    driver = Driver()
    keys = KeyState()
    slot = LatestSlot()
    log_q = queue.Queue(4096)
    stop = threading.Event()
    name = args.log or (datetime.now().strftime("panel_%Y%m%d_%H%M%S") if not args.demo else "demo")
    out_dir = ROOT / "runs" / name
    meta = {"robot": args.robot, "scene": args.scene, "backend": args.backend,
            "params": dict(TUNE), "mode": "demo" if args.demo else "manual",
            "started": datetime.now().isoformat(timespec="seconds")}

    sim = SimThread(driver, slot, log_q, stop, viewer=("none" if args.no_viewer else args.viewer))
    ctrl = ControlThread(driver, keys, stop)
    logger = LoggerThread(log_q, stop, out_dir, meta, keys)
    sim.start()
    ctrl.start()
    logger.start()

    rc = 0
    try:
        if args.selftest:
            rc = selftest_run(driver, keys)
        elif args.grasp:
            rc = grasp_run(driver, keys)
        elif args.pilot:
            rc = pilot_run(driver, keys, slot)
        elif args.demo:
            print("[panel] 脚本化操作录制约 25 秒...")
            demo_run(keys)
        else:
            from tui import PanelApp
            PanelApp(driver, keys, slot).run()
    finally:
        stop.set()
        sim.join(8)
        ctrl.join(5)
        logger.join(15)
        if args.selftest:
            if not (out_dir / "log.npz").exists():
                print("FAIL  日志文件生成")
                rc = 1
            else:
                print("PASS  日志文件生成 %s" % out_dir)
    print("[panel] 日志目录 %s" % out_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
