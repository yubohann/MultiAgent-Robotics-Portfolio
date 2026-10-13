"""Soft-Spiral 视觉伺服抓取（腕部相机 + 颜色检测 + 视觉控制下探/对准/触发/确认）。

流程：
  1. 下探：A1 平滑加力，腕相机搜索绿色横杆；
  2. 视觉停止：横杆进入夹持区（v/面积带）即停止下探；
  3. 对准伺服：若出带，用 A2 脉冲做闭环对准（最多 3 次）；
  4. 触发闭合：轻合 0.33 -> 保压；
  5. 夹持确认：腕相机仍见横杆且夹爪有接触 -> 提升保持。

运行：
  python spiral_grasp_vision.py --selftest        无界面自检（打印全过程）
  python spiral_grasp_vision.py --render          渲染 HD 视频（含腕相机画中画）
  python spiral_grasp_vision.py --render --offset 0 0   标称摆放
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_scene as G  # noqa: E402

BAR_OFFSET = (-0.012, -0.010)          # 视觉演示用挑战摆放（y, z 偏移）
DAMP_SCALE = 1.0
TAU_DIVE = 0.5                         # A1 指数接近时间常数
V_TARGET = 0.09                        # 标定：对准位姿时横杆在腕视图中的纵向位置
V_BAND = (-0.15, 0.16)                 # 可夹持带（成功抓取实测包络）
A_BAND = (5.0, 45.0)                   # 横杆可见面积带（千像素）
CHECK_PERIOD = 0.4
SETTLE_T = 1.5
CORRECT_MAX = 3
PULSE_UP, PULSE_DOWN = 0.5, 0.3
LIFT_T = 6.0
HOLD_T = 12.0
OUT_DIR = Path(r"C:\Users\Administrator\Desktop\aerial_soft_arm\videos\newmodel")


def make_model(offset=BAR_OFFSET, damp_scale=DAMP_SCALE):
    bar = (0.0, G.BAR[1] + offset[0], G.BAR[2] + offset[1])
    p = G.build_scene(bar=bar, bar_material="mat_target", out="vision_scene.xml")
    s = p.read_text(encoding="utf-8")
    s = s.replace('<body name="gripper_base" pos="0 0 0" euler="0 3.14 0">',
                  '<body name="gripper_base" pos="0 0 0" euler="0 3.14 0">'
                  '<camera name="wrist" pos="0 0 0.02" quat="0 0.7071 -0.7071 0" fovy="58"/>')
    if damp_scale != 1.0:
        def scale_damp(mo):
            return mo.group(1) + "%.5f" % (float(mo.group(2)) * damp_scale) + mo.group(3)
        s = re.sub(r'(<joint name="Joint\d+"[^>]*damping=")([0-9.eE+-]+)(")', scale_damp, s)
    p.write_text(s, encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(p))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def detect(img):
    """绿色横杆检测：返回 (u, v, area_kpx) 或 None。"""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    g = (arr[..., 1] > 0.45) & (arr[..., 1] > arr[..., 0] * 1.6) & (arr[..., 1] > arr[..., 2] * 1.4)
    ys, xs = np.nonzero(g)
    if len(xs) < 30:
        return None
    h, w = arr.shape[0], arr.shape[1]
    u = 2 * xs.mean() / (w - 1) - 1
    v = 2 * ys.mean() / (h - 1) - 1
    return float(u), float(v), len(xs) / 1000.0


def in_band(v, a):
    return (V_BAND[0] <= v <= V_BAND[1]) and (A_BAND[0] <= a <= A_BAND[1])


class VisionGrasp:
    """视觉状态机。update(dt, detect_fn, contacts_fn) 每步调用。"""

    def __init__(self, log=None):
        self.state = "dive"
        self.t = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.grip = 0.0
        self.a2_0 = 0.0
        self.check_t = 0.0
        self.dive_t = 0.0
        self.correct_n = 0
        self.verified = None
        self.stop_t = None
        self.log = log if log is not None else []

    def update(self, dt, detect_fn, contacts_fn=None):
        self.t += dt
        info = {"state": self.state, "det": None, "err": None, "in_band": None}
        if self.state == "dive":
            self.dive_t += dt
            self.a1 += (1.0 - self.a1) * dt / TAU_DIVE
            self.check_t += dt
            if self.check_t >= CHECK_PERIOD:
                self.check_t = 0.0
                det = detect_fn()
                info["det"] = det
                if det is not None:
                    u_, v_, a_ = det
                    info["err"] = v_ - V_TARGET
                    info["in_band"] = in_band(v_, a_)
                    self.log.append((self.dive_t, "dive", u_, v_, a_, self.a1, v_ - V_TARGET))
                    if info["in_band"]:
                        self.stop_t = self.dive_t
                        self.state, self.t = "settle", 0.0
            if self.dive_t > 8.0:
                self.state, self.t = "settle", 0.0
        elif self.state == "settle":
            self.a1 = min(self.a1 + 0.15 * dt, 1.0)
            if self.t >= SETTLE_T:
                self.state, self.t = "align", 0.0
                self.check_t = CHECK_PERIOD
        elif self.state == "align":
            self.check_t += dt
            if self.check_t >= CHECK_PERIOD:
                self.check_t = 0.0
                det = detect_fn()
                info["det"] = det
                if det is None:
                    self.correct_n += 1
                    if self.correct_n > CORRECT_MAX:
                        self.state, self.t = "close", 0.0
                else:
                    u_, v_, a_ = det
                    ib = in_band(v_, a_)
                    info["err"] = v_ - V_TARGET
                    info["in_band"] = ib
                    self.log.append((self.t + 10.0, "align", u_, v_, a_, self.a2, v_ - V_TARGET))
                    if ib or self.correct_n >= CORRECT_MAX:
                        self.state, self.t = "close", 0.0
                    else:
                        self.correct_n += 1
                        if v_ > V_BAND[1]:
                            self.a2 = min(self.a2 + PULSE_UP, 1.0)
                        elif v_ < V_BAND[0]:
                            self.a2 = max(self.a2 - PULSE_DOWN, 0.0)
        elif self.state == "close":
            u = min(self.t / 0.3, 1.0)
            self.grip = G.CLOSE_LEVEL * u
            if self.t >= 0.3:
                self.state, self.t = "close_hold", 0.0
        elif self.state == "close_hold":
            self.grip = G.CLOSE_LEVEL
            if self.t >= 1.2:
                det = detect_fn()
                ncon = contacts_fn() if contacts_fn is not None else 0
                self.verified = (det, ncon)
                info["verify"] = self.verified
                self.state, self.t = "lift", 0.0
                self.a2_0 = self.a2
        elif self.state == "lift":
            u = 0.5 - 0.5 * np.cos(np.pi * min(self.t / LIFT_T, 1.0))
            self.a2 = self.a2_0 + (1.0 - self.a2_0) * u
            self.grip = G.CLOSE_LEVEL + (1.0 - G.CLOSE_LEVEL) * u
            if self.t >= LIFT_T:
                self.state, self.t = "hold", 0.0
        else:
            self.a2 = 1.0
            self.grip = 1.0
        return (self.a1, self.a2, self.grip, self.grip), info


def contacts_fn_for(m, d):
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    g1, g2, db = bid("gripper_1"), bid("gripper_2"), bid("dumbbell")

    def fn():
        return sum(1 for i in range(d.ncon)
                   if db in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])
                   and (g1 in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])
                        or g2 in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])))
    return fn


def run_selftest(offset):
    m, d = make_model(offset)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    r = mujoco.Renderer(m, height=360, width=480)
    dt = m.opt.timestep
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    g1, g2, db = bid("gripper_1"), bid("gripper_2"), bid("dumbbell")
    cfn = contacts_fn_for(m, d)
    log = []
    vg = VisionGrasp(log)

    def detect_fn():
        r.update_scene(d, cid)
        return detect(r.render())

    total = 45.0
    n = int(total / dt)
    for k in range(n):
        ctrl, info = vg.update(dt, detect_fn, cfn)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        if k % int(2.0 / dt) == 0:
            mid = (d.xpos[g1] + d.xpos[g2]) / 2
            gap = np.linalg.norm(d.xpos[db] - mid) * 1000
            print("t=%5.1f %-11s a1=%.2f a2=%.2f grip=%.2f gap=%5.1fmm z=%.0fmm" %
                  ((k + 1) * dt, vg.state, vg.a1, vg.a2, vg.grip, gap, d.xpos[db][2] * 1000))
    mid = (d.xpos[g1] + d.xpos[g2]) / 2
    gap = np.linalg.norm(d.xpos[db] - mid) * 1000
    cf = cfn()
    print("\n视觉日志 (t, phase, u, v, area, a, err):")
    for row in log:
        print("  t=%5.1f %-5s u=%+.2f v=%+.2f area=%4.1f a=%.2f err=%+.3f" % row)
    print("视觉停止下探时刻：%s" % ("%.1fs" % vg.stop_t if vg.stop_t else "未触发"))
    if vg.verified is not None:
        det, ncon = vg.verified
        print("夹持确认：视觉=%s 接触=%d" % ("可见" if det else "不可见", ncon))
    print("final: gap=%.1fmm contacts=%d z=%.0fmm rise=%.0fmm" %
          (gap, cf, d.xpos[db][2] * 1000, (d.xpos[db][2] - (G.BAR[2] + offset[1])) * 1000))
    ok = cf > 0 and gap < 35 and d.xpos[db][2] > G.BAR[2] + offset[1] + 0.10
    print("SELFTEST", "PASS" if ok else "FAIL")
    r.close()
    return 0 if ok else 1


def run_render(offset, video_path=None, test_n=0):
    """渲染视觉伺服 HD 视频：主视角 + 腕相机画中画。"""
    W, H, FPS = 1920, 1080, 50
    DT = 0.002
    SPF = int(round(1.0 / (FPS * DT)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if video_path is None:
        video_path = OUT_DIR / "spiral_grasp_vision_hd.mp4"

    m, d = make_model(offset)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    r_main = mujoco.Renderer(m, height=H, width=W)
    r_wrist = mujoco.Renderer(m, height=360, width=480)
    cfn = contacts_fn_for(m, d)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    g1, g2, db, gb = bid("gripper_1"), bid("gripper_2"), bid("dumbbell"), bid("gripper_base")
    log = []
    vg = VisionGrasp(log)
    cam_a, cam_b = mujoco.MjvCamera(), mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam_a)
    mujoco.mjv_defaultCamera(cam_b)

    last = {"det": None, "img": None}
    log_lines = []

    def detect_fn():
        r_wrist.update_scene(d, cid)
        img = r_wrist.render()
        det = detect(img)
        last["det"], last["img"] = det, img
        return det

    FONT_BIG = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
    FONT_MID = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    FONT_SM = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1.0 - 0.17 * np.clip(rr - 0.52, 0, 1) / 0.48, 0, 1).astype(np.float32)

    proc = None
    if not test_n:
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-crf", "17", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)],
            stdin=subprocess.PIPE)

    state_cn = {"dive": "下探搜索", "settle": "就位", "align": "对准伺服",
                "close": "闭合", "close_hold": "保压", "lift": "夹持提升", "hold": "稳定保持"}
    seg_of = {"dive": 0, "settle": 0, "align": 0, "close": 1, "close_hold": 1,
              "lift": 0, "hold": 2}
    w_blend = 0.0
    t_global = 0.0
    frame = 0
    max_t = 40.0
    while t_global < max_t:
        for _ in range(SPF):
            ctrl, info = vg.update(DT, detect_fn, cfn)
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
            t_global += DT
        if vg.state == "hold" and vg.t > HOLD_T:
            break

        seg = seg_of.get(vg.state, 0)
        target_w = 0.0 if seg == 0 else (1.0 if seg == 1 else 1.0)
        if seg == 2:
            target_w = 1.0 if vg.t > 3.0 else 0.0
        w_blend += (target_w - w_blend) * (SPF * DT) / 0.3

        s = min(max((t_global - 3.0) / 12.0, 0.0), 1.0)
        cam_a.distance = 1.78 - 0.10 * s
        cam_a.azimuth = 129.0 + 5.0 * s
        cam_a.elevation = -10.0 - 4.0 * s
        cam_a.lookat[:] = (0.0, 0.30 + 0.24 * s, 0.30 + 0.14 * s)
        cam_b.distance = 0.52
        cam_b.azimuth = 148.0
        cam_b.elevation = -24.0
        cam_b.lookat[:] = d.xpos[db]

        r_main.update_scene(d, cam_a)
        img_a = r_main.render().copy()
        r_main.update_scene(d, cam_b)
        img_b = r_main.render().copy()
        img = (img_a.astype(np.float32) * (1 - w_blend) + img_b.astype(np.float32) * w_blend).astype(np.uint8)

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")

        # 腕相机画中画
        if last["img"] is not None:
            pip = Image.fromarray(last["img"]).resize((460, 345), Image.BILINEAR)
            px, py = W - 460 - 26, 26
            dr.rectangle((px - 3, py - 3, px + 463, py + 348), outline=(255, 255, 255, 90), width=2)
            im.paste(pip, (px, py))
            dr = ImageDraw.Draw(im, "RGBA")
            # 目标十字线（标定点）
            tx = px + (V_TARGET * 0 + (-0.08 + 1) / 2) * 460
            ty = py + ((V_TARGET + 1) / 2) * 345
            dr.line((px, ty, px + 460, ty), fill=(120, 210, 255, 170), width=1)
            dr.line((tx, py, tx, py + 345), fill=(120, 210, 255, 170), width=1)
            det = last["det"]
            if det is not None:
                u_, v_, a_ = det
                mx = px + (u_ + 1) / 2 * 460
                my = py + (v_ + 1) / 2 * 345
                ok = in_band(v_, a_)
                col = (80, 255, 120, 255) if ok else (255, 190, 60, 255)
                dr.ellipse((mx - 8, my - 8, mx + 8, my + 8), outline=col, width=3)
                txt = "u=%+.2f v=%+.2f 面积%.0f千px 误差%+.2f %s" % (
                    u_, v_, a_, v_ - V_TARGET, "在带内" if ok else "出带")
                tw = dr.textlength(txt, font=FONT_SM)
                dr.text((px + 460 - tw, py + 352), txt, font=FONT_SM, fill=col)
            else:
                txt = "腕部相机：搜索目标中…"
                tw = dr.textlength(txt, font=FONT_SM)
                dr.text((px + 460 - tw, py + 352), txt, font=FONT_SM, fill=(200, 205, 215, 255))

        # 主信息栏
        det = last["det"]
        l1 = "视觉伺服抓取 · %s · t=%.1f s" % (state_cn.get(vg.state, vg.state), t_global)
        l2 = "A1=%.2f A2=%.2f 夹爪=%.2f · 哑铃杆高度 z=%.0f mm" % (vg.a1, vg.a2, vg.grip, d.xpos[db][2] * 1000)
        l3 = "夹爪-哑铃接触 n=%d · 夹持偏差 %.0f mm%s" % (
            cfn(), np.linalg.norm(d.xpos[db] - (d.xpos[g1] + d.xpos[g2]) / 2) * 1000,
            (" · 视觉停止 t=%.1f s" % vg.stop_t) if vg.stop_t else "")
        pad, lh = 18, 40
        wpx = max(dr.textlength(s2, font=f) for s2, f in ((l1, FONT_BIG), (l2, FONT_MID), (l3, FONT_MID)))
        bw = int(wpx) + 2 * pad
        bh = 3 * lh + 2 * pad - 6
        x0, y0 = 26, H - 26 - bh
        dr.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=14,
                             fill=(10, 12, 17, 168), outline=(255, 255, 255, 36), width=1)
        dr.text((x0 + pad, y0 + pad - 4), l1, font=FONT_BIG, fill=(120, 210, 255, 255))
        dr.text((x0 + pad, y0 + pad + lh - 4), l2, font=FONT_MID, fill=(235, 238, 242, 255))
        dr.text((x0 + pad, y0 + pad + 2 * lh - 4), l3, font=FONT_MID, fill=(178, 192, 205, 255))
        img[:] = (np.asarray(im, dtype=np.float32) * vig[..., None]).astype(np.uint8)

        if test_n and frame % test_n == 0:
            Image.fromarray(img).save(HERE / ("vis_f%04d.png" % frame))
        if proc:
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        frame += 1
        if frame % 100 == 0:
            print("  t=%.1f %s" % (t_global, vg.state))

    if proc:
        proc.stdin.close()
        proc.wait()
        print("wrote %s (%.1f MB, %.1f s, %d frames)" %
              (video_path, Path(video_path).stat().st_size / 1e6, frame / FPS, frame))
    print("视觉停止下探：%s" % ("%.2f s" % vg.stop_t if vg.stop_t else "未触发"))
    if vg.verified is not None:
        det, ncon = vg.verified
        print("夹持确认：视觉=%s 接触=%d" % ("可见" if det else "不可见", ncon))
    r_main.close()
    r_wrist.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--offset", type=float, nargs=2, default=list(BAR_OFFSET))
    ap.add_argument("--test", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        return run_selftest(tuple(args.offset))
    if args.render:
        return run_render(tuple(args.offset), test_n=args.test)
    print("用法：--selftest 或 --render")
    return 0


if __name__ == "__main__":
    sys.exit(main())
