"""屏幕键盘按钮定义与键态存储。"""
import threading
import time

HOLD = "hold"
LATCH = "latch"
CLICK = "click"

ARM = "arm"
DRONE = "drone"
GLOBAL = "global"

QUICK_LATCH = 0.4

BUTTONS = [
    # id, label, zone, semantics, hint
    ("w", "W", ARM, HOLD, "末端前"),
    ("s", "S", ARM, HOLD, "末端后"),
    ("a", "A", ARM, HOLD, "末端左"),
    ("d", "D", ARM, HOLD, "末端右"),
    ("q", "Q", ARM, HOLD, "伸出"),
    ("e", "E", ARM, HOLD, "收回"),
    ("r", "R", ARM, HOLD, "弯曲正"),
    ("f", "F", ARM, HOLD, "弯曲反"),
    ("z", "Z", ARM, LATCH, "爪开"),
    ("c", "C", ARM, LATCH, "爪合"),
    ("1", "1", ARM, HOLD, "腱一"),
    ("2", "2", ARM, HOLD, "腱二"),
    ("3", "3", ARM, HOLD, "腱三"),
    ("4", "4", ARM, CLICK, "锁段一"),
    ("5", "5", ARM, CLICK, "锁段二"),
    ("6", "6", ARM, CLICK, "锁段三"),
    ("x", "X", ARM, CLICK, "抓取"),
    ("g", "G", ARM, CLICK, "序列"),
    ("t", "T", ARM, CLICK, "换爪"),
    ("v", "V", ARM, CLICK, "示教存"),
    ("b", "B", ARM, CLICK, "示教回"),
    ("up", "↑", DRONE, LATCH, "前进"),
    ("down", "↓", DRONE, LATCH, "后退"),
    ("left", "←", DRONE, LATCH, "左移"),
    ("right", "→", DRONE, LATCH, "右移"),
    ("k8", "8", DRONE, LATCH, "上升"),
    ("k2", "2", DRONE, LATCH, "下降"),
    ("k4", "4", DRONE, LATCH, "左偏航"),
    ("k6", "6", DRONE, LATCH, "右偏航"),
    ("k5", "5", DRONE, CLICK, "悬停锁"),
    ("k0", "0", DRONE, CLICK, "回原点"),
    ("estop", "急停", GLOBAL, CLICK, "急停"),
    ("mode_combo", "组合", GLOBAL, CLICK, "组合模式"),
    ("mode_precise", "精调", GLOBAL, CLICK, "精调模式"),
    ("mode_drone", "单控", GLOBAL, CLICK, "单控模式"),
    ("exit", "退出", GLOBAL, CLICK, "退出存盘"),
]

SEMANTICS = {b[0]: b[3] for b in BUTTONS}
ZONE = {b[0]: b[2] for b in BUTTONS}
LABEL = {b[0]: b[1] for b in BUTTONS}
HINT = {b[0]: b[4] for b in BUTTONS}

MODES = ("combo", "precise", "drone")
MODE_LABEL = {"combo": "组合", "precise": "精调", "drone": "单控"}


class KeyState:
    """线程安全的键态，界面线程写，控制线程读，事件流供日志。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.down = {}
        self.latched = set()
        self.events = []
        self.event_log = []
        self.mode = "combo"
        self.estop = False
        self.quit = False

    def _log(self, phase, bid):
        self.event_log.append((time.monotonic(), phase, bid))

    def press(self, bid):
        sem = SEMANTICS[bid]
        with self._lock:
            self._log("d", bid)
            if sem == HOLD:
                self.down[bid] = time.monotonic()
            elif sem == LATCH:
                if bid in self.latched:
                    self.latched.discard(bid)
                else:
                    self.down[bid] = time.monotonic()
            elif sem == CLICK:
                if bid == "estop":
                    self.estop = not self.estop
                elif bid.startswith("mode_"):
                    self.mode = bid[5:]
                elif bid == "exit":
                    self.quit = True
                else:
                    self.events.append(bid)

    def release(self, bid):
        sem = SEMANTICS[bid]
        with self._lock:
            self._log("u", bid)
            t0 = self.down.pop(bid, None)
            if sem == LATCH and t0 is not None and time.monotonic() - t0 < QUICK_LATCH:
                self.latched.add(bid)

    def consume_events(self):
        with self._lock:
            ev, self.events = self.events, []
            return ev

    def held(self):
        with self._lock:
            return set(self.down) | set(self.latched)

    def snapshot(self):
        with self._lock:
            return {
                "held": set(self.down) | set(self.latched),
                "latched": set(self.latched),
                "mode": self.mode,
                "estop": self.estop,
                "quit": self.quit,
            }
