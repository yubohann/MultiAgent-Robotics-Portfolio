"""仿真线程，唯一访问 mjData，实时步进并发布快照。"""
import queue
import threading
import time

import mujoco
import mujoco.viewer

from drivers import Driver, TUNE


class LatestSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._v = None

    def put(self, v):
        with self._lock:
            self._v = v

    def get(self):
        with self._lock:
            return self._v


class SimThread(threading.Thread):
    def __init__(self, driver, ui_slot, log_q, stop_evt, viewer="blit"):
        super().__init__(name="sim", daemon=True)
        self.d = driver
        self.ui = ui_slot
        self.log_q = log_q
        self.stop = stop_evt
        self.viewer = viewer

    def run(self):
        if self.viewer == "glfw":
            try:
                with mujoco.viewer.launch_passive(self.d.model, self.d.data) as v:
                    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
                    self._loop(v)
                return
            except Exception as e:
                print("[sim] glfw 观察窗不可用，回退 blit", e)
                self.viewer = "blit"
        if self.viewer == "blit":
            from blitview import BlitView
            bv = BlitView(self.d.model, self.d.data)
            self._loop(bv)
        else:
            self._loop(None)

    def _loop(self, v):
        d = self.d
        dt = 0.001
        subs = 10
        frame_i = 0
        next_t = time.monotonic()
        next_log = next_t
        next_ui = next_t
        while not self.stop.is_set():
            for _ in range(subs):
                d.step(dt)
            now = time.monotonic()
            if now >= next_log:
                next_log = now + 1.0 / max(TUNE["log_rate"], 5.0)
                snap = d.snapshot()
                try:
                    self.log_q.put_nowait(snap)
                except queue.Full:
                    pass
            if now >= next_ui:
                next_ui = now + 0.05
                self.ui.put(d.snapshot())
            if v is not None:
                frame_i += 1
                if frame_i % 2 == 0:
                    v.sync()
            next_t += subs * dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
