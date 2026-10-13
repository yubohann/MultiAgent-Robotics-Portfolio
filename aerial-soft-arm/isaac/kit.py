"""Isaac 侧公共工具，阶段打点与有界关闭，stdout 强制刷新。"""
import os
import threading
import time

T0 = time.monotonic()


def mark(tag):
    print("[kit] +%.1fs %s" % (time.monotonic() - T0, tag), flush=True)


def bounded_close(app, seconds=25.0):
    """有界关闭：超过 seconds 未完成则强制退出。"""

    def _force(sec):
        time.sleep(sec)
        mark("close timeout, force exit")
        os._exit(0)

    threading.Thread(target=_force, args=(seconds,), daemon=True).start()
    try:
        app.close()
    except Exception as e:
        mark("close error %s" % e)
    mark("app closed")
    os._exit(0)
