"""Soft-Spiral 写死版抓取（固定时序编排，HD 渲染）。

六阶段：静止 -> 下探 -> 就位 -> 闭合(0.33) -> 夹持提升(A2+夹爪加力) -> 稳定保持。
运行：
  python spiral_grasp_scripted.py                 渲染 HD 视频
  python spiral_grasp_scripted.py --test 250      只出关键帧（不写视频）
"""
import argparse
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

OUT_DIR = Path(r"C:\Users\Administrator\Desktop\aerial_soft_arm\videos\newmodel")
RENDER_FPS = 50
DT = 0.002
STEPS_PER_FRAME = int(round(1.0 / (RENDER_FPS * DT)))
W, H = 1920, 1080
FONT_BIG = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 33)
FONT_MID = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
SEGS = [(0.0, 5.3, 0), (5.3, 9.0, 1), (9.0, 14.5, 0), (14.5, G.T_END + 1, 1)]
DISSOLVE = 0.25


def cam_weight(t):
    for t0, t1, cam in SEGS:
        if t < t1:
            if cam == 1:
                return (t - t0) / DISSOLVE if t < t0 + DISSOLVE else 1.0
            return 1.0 - (t - t0) / DISSOLVE if (t < t0 + DISSOLVE and t0 > 0) else 0.0
    return 1.0


class Demo:
    def __init__(self, video_path, test_n=0):
        p = G.build_scene()
        self.model = mujoco.MjModel.from_xml_path(str(p))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.model.vis.global_.offwidth = W
        self.model.vis.global_.offheight = H
        self.renderer = mujoco.Renderer(self.model, height=H, width=W)
        self.cam_a, self.cam_b = mujoco.MjvCamera(), mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam_a)
        mujoco.mjv_defaultCamera(self.cam_b)
        self.gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self.g1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_1")
        self.g2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_2")
        self.db = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "dumbbell")
        self.trail = []
        self.buf = []
        self.test_n = test_n
        self.saved = 0
        self.video_path = Path(video_path)
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = np.clip(1.0 - 0.17 * np.clip(rr - 0.52, 0, 1) / 0.48, 0, 1).astype(np.float32)
        self.proc = None
        if not test_n:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                 "-r", str(RENDER_FPS), "-i", "-",
                 "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.video_path)],
                stdin=subprocess.PIPE)

    def contacts(self):
        c = 0
        for i in range(self.data.ncon):
            b1 = self.model.geom_bodyid[self.data.contact[i].geom1]
            b2 = self.model.geom_bodyid[self.data.contact[i].geom2]
            if self.db in (b1, b2) and (self.g1 in (b1, b2) or self.g2 in (b1, b2)):
                c += 1
        return c

    def cams(self, t):
        p = G.wide_cam(t)
        a = self.cam_a
        a.distance, a.azimuth, a.elevation = p["distance"], p["azimuth"], p["elevation"]
        a.lookat[:] = p["lookat"]
        q = G.close_cam(self.data.xpos[self.db])
        b = self.cam_b
        b.distance, b.azimuth, b.elevation = q["distance"], q["azimuth"], q["elevation"]
        b.lookat[:] = q["lookat"]

    def tick_frame(self, frame, t):
        self.cams(t)
        self.renderer.update_scene(self.data, self.cam_a)
        self.draw_trail(self.renderer.scene)
        img_a = self.renderer.render().copy()
        self.renderer.update_scene(self.data, self.cam_b)
        self.draw_trail(self.renderer.scene)
        img_b = self.renderer.render().copy()
        if self.test_n and frame % max(self.test_n, 1) == 0:
            Image.fromarray(img_a).save(HERE / f"sc_f{frame:04d}_a.png")
            self.saved += 1
        self.buf.append((frame, img_a, img_b))

    def select(self, item):
        idx, a, b = item
        w = cam_weight(idx / RENDER_FPS)
        if w <= 0:
            return a
        if w >= 1:
            return b
        return (a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).astype(np.uint8)

    def emit_buf(self, flush_all=False):
        while self.buf and (flush_all or len(self.buf) > 24):
            item = self.buf.pop(0)
            img = self.select(item)
            self.overlay(img, item[0])
            if self.proc:
                self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def finish(self, total, fade_frames=20):
        while self.buf:
            item = self.buf.pop(0)
            img = self.select(item)
            self.overlay(img, item[0])
            idx = item[0]
            if idx >= total - fade_frames:
                k = (total - 1 - idx) / float(fade_frames)
                img = (img.astype(np.float32) * (0.06 + 0.94 * k)).astype(np.uint8)
            if self.proc:
                self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        self.renderer.close()

    def draw_trail(self, scene):
        tr = self.trail
        for i in range(len(tr) - 1):
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            age = i / max(len(tr) - 1, 1)
            rgba = (0.20, 0.80, 1.0, 0.05 + 0.55 * age)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0020,
                                 np.asarray(tr[i], dtype=np.float64),
                                 np.asarray(tr[i + 1], dtype=np.float64))
            scene.ngeom += 1

    def overlay(self, img, frame):
        t = frame / RENDER_FPS
        ph = G.phase_of(t)
        a1, a2, a3, _ = getattr(self, "last_ctrl", (0, 0, 0, 0))
        dbz = self.data.xpos[self.db][2] * 1000
        gap = np.linalg.norm(self.data.xpos[self.db] - (self.data.xpos[self.g1] + self.data.xpos[self.g2]) / 2) * 1000
        n = self.contacts()
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        l1 = "%s · t=%.1f s" % (ph, t)
        l2 = "12 节螺旋连续体软臂 · 腱驱 A1=%.2f A2=%.2f · 夹爪 A3/A4=%.2f" % (a1, a2, a3)
        l3 = "哑铃杆高度 z=%.0f mm · 夹持偏差 %.0f mm · 夹爪-哑铃接触 n=%d" % (dbz, gap, n)
        pad, lh = 18, 40
        wpx = max(d.textlength(s, font=f) for s, f in ((l1, FONT_BIG), (l2, FONT_MID), (l3, FONT_MID)))
        bw = int(wpx) + 2 * pad
        bh = 3 * lh + 2 * pad - 6
        x0, y0 = 26, H - 26 - bh
        d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=14,
                            fill=(10, 12, 17, 168), outline=(255, 255, 255, 36), width=1)
        d.text((x0 + pad, y0 + pad - 4), l1, font=FONT_BIG, fill=(120, 210, 255, 255))
        d.text((x0 + pad, y0 + pad + lh - 4), l2, font=FONT_MID, fill=(235, 238, 242, 255))
        d.text((x0 + pad, y0 + pad + 2 * lh - 4), l3, font=FONT_MID, fill=(178, 192, 205, 255))
        img[:] = (np.asarray(im, dtype=np.float32) * self.vig[..., None]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--video", default=str(OUT_DIR / "spiral_grasp_hd.mp4"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = Demo(args.video, test_n=args.test)
    total_frames = int(G.T_END * RENDER_FPS)
    t0 = time.monotonic()
    step = 0
    for frame in range(total_frames):
        for _ in range(STEPS_PER_FRAME):
            step += 1
            a = G.controls(step * DT)
            demo.data.ctrl[:] = a
            mujoco.mj_step(demo.model, demo.data)
        demo.last_ctrl = a
        demo.trail.append(demo.data.xpos[demo.gid].copy())
        demo.tick_frame(frame, frame / RENDER_FPS)
        demo.emit_buf()
        if frame % 250 == 0:
            print("  t=%4.1f %-16s dbz=%.0fmm n=%d" %
                  (frame / RENDER_FPS, G.phase_of(frame / RENDER_FPS),
                   demo.data.xpos[demo.db][2] * 1000, demo.contacts()))
    demo.finish(total_frames)
    if not args.test:
        print("wrote %s (%.1f MB, %.1f s)" % (
            args.video, Path(args.video).stat().st_size / 1e6, total_frames / RENDER_FPS))
    print("wall %.0fs saved_tests=%d" % (time.monotonic() - t0, demo.saved))


if __name__ == "__main__":
    sys.exit(main())
