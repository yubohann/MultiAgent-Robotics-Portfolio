"""文本面板，屏幕键盘与参数区，全鼠标交互。"""
import time
from datetime import datetime
from pathlib import Path

import yaml
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static

import drivers as D
from keys import HINT, LABEL, MODE_LABEL, SEMANTICS, ZONE, KeyState

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

PAGES = ["机械臂", "气动", "变刚度", "夹爪", "无人机", "日志"]
PARAMS = [dict(zip(("page", "label", "key", "min", "max", "step"), t)) for t in D.make_params()]
PARAM_BY_KEY = {p["key"]: p for p in PARAMS}
LABEL_KEY = {(p["page"], p["label"]): p["key"] for p in PARAMS}
TRACK = 26


def set_param(key, v):
    p = PARAM_BY_KEY[key]
    D.TUNE[key] = min(max(v, p["min"]), p["max"])


class ParamSlider(Static):
    def __init__(self, key, lo, hi, **kw):
        super().__init__(**kw)
        self.key = key
        self.lo = lo
        self.hi = hi
        self._v = D.TUNE[key]
        self._drag = False
        self._draw()

    @property
    def value(self):
        return self._v

    def _draw(self):
        frac = 0.0 if self.hi <= self.lo else (self._v - self.lo) / (self.hi - self.lo)
        n = int(round(frac * TRACK))
        self.update("[%s%s]" % ("\u2588" * n, "\u2591" * (TRACK - n)))

    def set_value(self, v, silent=False):
        v = min(max(v, self.lo), self.hi)
        changed = abs(v - self._v) > 1e-9
        self._v = v
        self._draw()
        if changed and not silent:
            self.app.param_changed(self.key, v)

    def _from_event(self, event):
        frac = (event.x - 1) / max(TRACK, 1)
        return self.lo + min(max(frac, 0.0), 1.0) * (self.hi - self.lo)

    def on_mouse_down(self, event):
        if event.button != 1:
            return
        event.stop()
        self._drag = True
        self.capture_mouse()
        self.set_value(self._from_event(event))

    def on_mouse_move(self, event):
        if not self._drag:
            return
        event.stop()
        self.set_value(self._from_event(event))

    def on_mouse_up(self, event):
        if event.button != 1:
            return
        event.stop()
        if self._drag:
            self._drag = False
            self.release_mouse()
            self.set_value(self._from_event(event))


class ValueLabel(Static):
    def __init__(self, key, **kw):
        super().__init__(**kw)
        self.key = key
        self._last = 0.0

    def on_click(self, event):
        if event.button != 1:
            return
        now = time.monotonic()
        if now - self._last < 0.35:
            self.app.push_screen(ValueModal(self.key), self.app.value_result_cb)
        self._last = now
        event.stop()


class ValueModal(ModalScreen):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def compose(self):
        p = PARAM_BY_KEY[self.key]
        yield Vertical(
            Label("%s 精确输入" % p["label"], classes="md_title"),
            Input(value=str(round(D.TUNE[self.key], 4)), id="vm_input"),
            Horizontal(Button("确定", id="vm_ok"), Button("取消", id="vm_cancel")),
        )

    def on_button_pressed(self, event):
        if event.button.id == "vm_ok":
            try:
                v = float(self.query_one("#vm_input", Input).value)
            except ValueError:
                v = D.TUNE[self.key]
            self.dismiss((self.key, v))
        else:
            self.dismiss(None)


class LoadModal(ModalScreen):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def compose(self):
        yield Vertical(
            Label("加载配置", classes="md_title"),
            OptionList(*[f.name for f in self.files], id="lm_list"),
            Button("取消", id="lm_cancel"),
        )

    def on_option_list_option_selected(self, event):
        self.dismiss(str(self.files[event.option_index]))

    def on_button_pressed(self, event):
        self.dismiss(None)


class TabBtn(Static):
    def __init__(self, idx, label, active):
        super().__init__(label, id="tab_%d" % idx,
                         classes="tab" + (" active" if active else ""))
        self.idx = idx

    def on_click(self, event):
        if event.button != 1:
            return
        event.stop()
        self.app.show_page(PAGES[self.idx])


class KeyBtn(Static):
    def __init__(self, bid):
        super().__init__()
        self.bid = bid
        self.zone = ZONE[bid]
        self.sem = SEMANTICS[bid]
        self.id = "b_" + bid
        self.classes = "kbtn zone_%s sem_%s" % (self.zone, self.sem)
        if self.zone == "global":
            self.update("[b]%s[/]" % LABEL[bid])
        else:
            self.update("[b]%s[/] [dim]%s[/]" % (LABEL[bid], HINT[bid]))

    def _blocked(self):
        mode = self.app.key_state.mode
        if self.zone == "arm" and mode == "drone":
            return True
        if self.zone == "drone" and mode == "precise":
            return True
        return False

    def on_mouse_down(self, event):
        if event.button != 1:
            return
        event.stop()
        if self._blocked():
            return
        self.capture_mouse()
        self.add_class("down")
        self.app.key_state.press(self.bid)

    def on_mouse_up(self, event):
        if event.button != 1:
            return
        event.stop()
        self.release_mouse()
        if self.has_class("down"):
            self.remove_class("down")
            self.app.key_state.release(self.bid)


class PanelApp(App):
    CSS = """
    Screen { background: #101216; }
    #title { height: 1; background: #1c2026; color: #dfe3ea; text-align: center; }
    #topbar { height: 3; padding: 0 1; }
    #topbar .spacer { width: 1fr; }
    #topbar .tab { width: 8; height: 3; text-align: center; border: round #39414d;
                   background: #171a20; color: #aab2bf; margin: 0 1; }
    #topbar .tab.active { background: #2f5da8; color: white; border: round #4d7fd4; }
    .tb_btn { width: 7; height: 3; text-align: center; border: round #39414d;
              background: #171a20; color: #aab2bf; margin: 0 1; }
    #params { height: auto; padding: 0 1; }
    .prow { height: 1; }
    .prow Label.pl_name { width: 14; color: #aab2bf; }
    ParamSlider { width: 30; color: #6f9fe0; }
    .prow ValueLabel { width: 10; text-align: right; color: #ffd28a; }
    .prow Button { width: 3; text-align: center; }
    #zones { height: auto; padding: 0 1; }
    #zone_arm, #zone_drone { height: auto; }
    #zone_arm { background: #131a26; border: round #26405f; }
    #zone_drone { background: #131f18; border: round #265f3c; }
    .krow { height: 3; }
    .kbtn { width: 12; height: 3; padding: 0; text-align: center;
            border: round #444d5c; background: #1b202a; color: #d7dce5; margin: 0 1; }
    .kbtn.down { background: #2f5da8; color: white; }
    .kbtn.latched { background: #7a5200; color: #ffe9c2; border: round #b07417; }
    .kbtn.disabled { opacity: 0.35; }
    .kbtn.active { background: #2f5da8; color: white; }
    #b_estop { background: #a40f12; color: white; border: round #e33; width: 10; }
    #b_estop.down, #b_estop.latched { background: #ff2a33; color: white; }
    #bottom { height: 3; padding: 0 1; }
    #bottom .kbtn { margin: 0 2; }
    #status { height: 3; background: #14181f; color: #cdd4de; padding: 0 1;
              border: round #2a3140; }
    ValueModal { align: center middle; }
    LoadModal { align: center middle; }
    ValueModal > Vertical, LoadModal > Vertical { width: 64; height: auto;
        background: #1a1f28; border: round #4d7fd4; padding: 1 2; }
    .md_title { text-align: center; color: #ffd28a; }
    #vm_input { margin: 1 0; }
    """

    def __init__(self, driver, key_state: KeyState, slot):
        super().__init__()
        self.d = driver
        self.key_state = key_state
        self.slot = slot
        self.page = PAGES[0]
        self._syncing = False
        self.rows = None
        self._msg = None
        self._msg_t = 0.0

    def compose(self) -> ComposeResult:
        yield Static(" AS-01 桁架臂面板 · 屏幕键盘遥操作 ", id="title")
        with Horizontal(id="topbar"):
            for i, pg in enumerate(PAGES):
                yield TabBtn(i, pg, pg == self.page)
            yield Static("", classes="spacer")
            yield Button("存盘", id="tb_save", classes="tb_btn")
            yield Button("加载", id="tb_load", classes="tb_btn")
            yield Button("默认", id="tb_def", classes="tb_btn")
        with Vertical(id="params"):
            for p in PARAMS:
                with Horizontal(classes="prow page_%d" % PAGES.index(p["page"])):
                    yield Label(p["label"], classes="pl_name")
                    yield ParamSlider(p["key"], p["min"], p["max"], id="ps_" + p["key"])
                    yield ValueLabel(p["key"], id="pv_" + p["key"])
                    yield Button("-", id="pm_" + p["key"], classes="tb_btn")
                    yield Button("+", id="pp_" + p["key"], classes="tb_btn")
        yield Static(id="status")
        with Horizontal(id="zones"):
            with Vertical(id="zone_arm"):
                for ids in (("w", "a", "s", "d", "q"), ("e", "r", "f", "z", "c"),
                            ("1", "2", "3", "4", "5"), ("6", "x", "g", "v", "b")):
                    with Horizontal(classes="krow"):
                        for b in ids:
                            yield KeyBtn(b)
            with Vertical(id="zone_drone"):
                for ids in (("up", "down", "left", "right"),
                            ("k8", "k2", "k4", "k6"), ("k5", "k0")):
                    with Horizontal(classes="krow"):
                        for b in ids:
                            yield KeyBtn(b)
        with Horizontal(id="bottom"):
            for b in ("estop", "mode_combo", "mode_precise", "mode_drone", "exit"):
                yield KeyBtn(b)

    def on_mount(self):
        self.show_page(self.page)
        self.set_interval(0.1, self.refresh_status)

    def show_page(self, pg):
        self.page = pg
        idx = PAGES.index(pg)
        for i, tab in enumerate(self.query(".tab")):
            tab.set_class(i == idx, "active")
        if self.rows is None:
            self.rows = list(self.query(".prow"))
        for row in self.rows:
            row.display = ("page_%d" % idx) in row.classes

    @on(Button.Pressed)
    def _btn(self, event):
        bid = event.button.id or ""
        if bid.startswith("pm_") or bid.startswith("pp_"):
            key = bid[3:]
            p = PARAM_BY_KEY[key]
            delta = -p["step"] if bid[1] == "m" else p["step"]
            set_param(key, D.TUNE[key] + delta)
            self.refresh_status()
            return
        if bid.startswith("tab_"):
            self.show_page(PAGES[int(bid[4:])])
            return
        if bid == "tb_save":
            self.save_config_cb()
        elif bid == "tb_load":
            files = sorted(CONFIG_DIR.glob("panel_*.yaml"), reverse=True)
            if not files:
                self.flash("没有可加载的配置")
            else:
                self.push_screen(LoadModal(files), self.load_config_cb)
        elif bid == "tb_def":
            D.TUNE.update(D.DEFAULTS)
            self.refresh_status()
            self.flash("参数已恢复默认")

    def param_changed(self, key, value):
        set_param(key, value)
        self.refresh_status()

    def save_config_cb(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = CONFIG_DIR / ("panel_%s.yaml" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        data = {"%s/%s" % (p["page"], p["label"]): round(D.TUNE[p["key"]], 4) for p in PARAMS}
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
        self.flash("已存盘 " + path.name)

    def load_config_cb(self, result):
        if not result:
            return
        path = CONFIG_DIR / result
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        loaded = 0
        for pk, v in data.items():
            key = LABEL_KEY.get(tuple(str(pk).split("/", 1)))
            if key is not None:
                set_param(key, float(v))
                loaded += 1
        self.refresh_status()
        self.flash("已加载 %s，%d 项" % (result, loaded))

    def value_result_cb(self, result):
        if not result:
            return
        key, v = result
        set_param(key, v)
        self.refresh_status()
        self.flash("已设置 " + key)

    def flash(self, msg):
        self._msg = msg
        self._msg_t = time.time()

    def refresh_status(self):
        self._syncing = True
        try:
            for p in PARAMS:
                self.query_one("#pv_" + p["key"], ValueLabel).update("%.3f" % D.TUNE[p["key"]])
                sl = self.query_one("#ps_" + p["key"], ParamSlider)
                if abs(sl.value - D.TUNE[p["key"]]) > 1e-9:
                    sl.set_value(D.TUNE[p["key"]], silent=True)
            snap = self.slot.get()
            ks = self.key_state.snapshot()
            for bid in ("mode_combo", "mode_precise", "mode_drone"):
                self.query_one("#b_" + bid, KeyBtn).set_class(("mode_" + ks["mode"]) == bid, "active")
            for b in self.query(".kbtn"):
                zone = ZONE[b.bid]
                blocked = (zone == "arm" and ks["mode"] == "drone") or (zone == "drone" and ks["mode"] == "precise")
                b.set_class(blocked, "disabled")
                if b.bid == "estop":
                    b.set_class(ks["estop"], "latched")
                else:
                    b.set_class(b.bid in ks["latched"], "latched")
            if ks["quit"]:
                self.exit()
            if self._msg and time.time() - self._msg_t > 3.0:
                self._msg = None
            if snap is None:
                return
            acts = ",".join(LABEL[b] for b in sorted(ks["held"])) or "-"
            w = snap["wrist"]
            est = "[red b]是[/]" if ks["estop"] else "否"
            msg = self._msg or ""
            size = self.size
            if size.width < 132 or size.height < 30:
                msg = "[red]窗口偏小 %dx%d，请拉大到 132x30 以上[/]" % (size.width, size.height)
            self.query_one("#status", Static).update(
                "模式 [b]%s[/]  急停 %s  末端 %.2f %.2f %.2f  接触 %d  腔压 %.2f bar  "
                "动作 %s  序列 %s  %s" % (
                    MODE_LABEL[ks["mode"]], est, w[0], w[1], w[2], snap["contacts"],
                    snap["P_meas"], acts, snap["seq_phase"] or "-", msg))
        finally:
            self._syncing = False
