"""离屏渲染加 tkinter 窗口显示。"""
import threading

import mujoco
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk


class BlitView:
    """在仿真线程里离屏渲染，用普通窗口贴图显示，支持鼠标旋转平移缩放。"""

    def __init__(self, model, data, width=1120, height=680):
        self.model = model
        self.data = data
        self.w = width
        self.h = height
        self.closed = False
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 2.6
        self.cam.elevation = -10.0
        self.cam.azimuth = 90.0
        self.cam.lookat[:] = (0.0, 0.05, 1.75)
        self.root = tk.Tk()
        self.root.title("MuJoCo : arm_lattice")
        self.root.resizable(True, True)
        self.canvas = tk.Canvas(self.root, width=width, height=height,
                                highlightthickness=0, bg="#101014")
        self.canvas.pack(fill="both", expand=True)
        self._photo = None
        self._img_id = None
        self._drag = None
        self._pending = None
        self.canvas.bind("<Configure>", self._resize)
        self.root.bind("<F11>", self._toggle_full)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.canvas.bind("<ButtonPress-1>", lambda e: self._start(e, "orbit"))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._start(e, "pan"))
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<B3-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._end())
        self.canvas.bind("<ButtonRelease-3>", lambda e: self._end())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.update()

    def _resize(self, ev):
        if ev.width < 64 or ev.height < 64:
            return
        if (ev.width, ev.height) != (self.w, self.h):
            self._pending = (ev.width, ev.height)

    def _toggle_full(self, ev=None):
        fs = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not fs)

    def _on_close(self):
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _start(self, ev, mode):
        self._drag = (mode, ev.x, ev.y)

    def _end(self):
        self._drag = None

    def _move(self, ev):
        if self._drag is None:
            return
        mode, x0, y0 = self._drag
        dx, dy = ev.x - x0, ev.y - y0
        self._drag = (mode, ev.x, ev.y)
        if mode == "orbit":
            self.cam.azimuth = (self.cam.azimuth + dx * 0.5) % 360.0
            self.cam.elevation = float(np.clip(self.cam.elevation + dy * 0.3, -89.0, 89.0))
        else:
            az = np.radians(self.cam.azimuth)
            el = np.radians(self.cam.elevation)
            right = np.array([-np.sin(az), np.cos(az), 0.0])
            fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            up = np.cross(fwd, right)
            k = self.cam.distance / 600.0
            self.cam.lookat[:] = self.cam.lookat - right * dx * k + up * dy * k

    def _wheel(self, ev):
        self.cam.distance = float(np.clip(self.cam.distance * (0.9 if ev.delta > 0 else 1.1), 0.3, 9.0))

    def sync(self):
        if self.closed:
            return
        if self._pending is not None:
            w, h = self._pending
            self._pending = None
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
            self.w, self.h = w, h
            self._img_id = None
        try:
            self.renderer.update_scene(self.data, self.cam)
            frame = self.renderer.render()
            self._photo = ImageTk.PhotoImage(Image.fromarray(frame))
            if self._img_id is None:
                self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            else:
                self.canvas.itemconfig(self._img_id, image=self._photo)
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        self._on_close()
