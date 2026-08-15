"""方法參數的設定視窗(Tkinter)。

純邏輯在 `ui/settings.py`,那邊沒有 tkinter 相依所以測得到;這裡只負責
把參數表畫成表單。
"""
import tkinter as tk
from tkinter import messagebox, ttk

from ui.settings import PARAMS, defaults_for, get_in


class SettingsDialog(tk.Toplevel):
    """逐項調整某個判定方法的參數。

    只列出「這個方法真的會用到」的參數:方法選了純規則卻讓人調融合權重,
    調了也沒有作用,只會讓人以為是自己設錯。
    """

    def __init__(self, master, method, base_cfg: dict,
                 overrides: dict | None = None):
        super().__init__(master)
        self.title(f"參數設定 — {method.name}" if method else "參數設定")
        # 固定寬度:說明文字是這個視窗的重點(每個參數為什麼存在、調了
        # 會怎樣),寬度不夠就會被擠成一個字,等於沒有
        self.geometry("880x640")
        self.minsize(760, 480)
        self.method = method
        self.base_cfg = base_cfg
        self.result = None            # 按「套用」才會是 dict
        self._vars = {}

        cur = defaults_for(base_cfg, method)
        cur.update({k: v for k, v in (overrides or {}).items() if k in cur})

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text=(f"「{method.name}」的判定參數。"
                         if method else "判定參數。")
            + "調整只影響這個方法,其他方法各自保有自己的設定。"
              "設定檔本身不會被改寫。",
            foreground="#555555", wraplength=830, justify="left").pack(
            anchor="w", pady=(0, 8))

        canvas = tk.Canvas(outer, height=470, highlightthickness=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-e.delta / 120), "units"))
        self._canvas = canvas

        shown = 0
        for group, params in PARAMS:
            usable = [p for p in params if p.applies_to(method)
                      and get_in(base_cfg, p.path) is not None]
            if not usable:
                continue
            box = ttk.LabelFrame(inner, text=group, padding=8)
            box.pack(fill="x", pady=4, padx=(0, 8))
            for p in usable:
                shown += 1
                row = ttk.Frame(box)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=p.label, width=24, anchor="w").pack(
                    side="left")
                var = tk.StringVar(value=_fmt(cur.get(p.key, p.lo), p))
                self._vars[p.key] = (var, p)
                if p.boolean:
                    ttk.Checkbutton(
                        row, variable=var, onvalue="1", offvalue="0",
                        text="開啟").pack(side="left")
                else:
                    ttk.Spinbox(row, from_=p.lo, to=p.hi, increment=p.step,
                                textvariable=var, width=8).pack(side="left")
                ttk.Label(row, text=p.unit, width=10, anchor="w").pack(
                    side="left", padx=(4, 8))
                if p.help:
                    ttk.Label(row, text=p.help, foreground="#777777",
                              wraplength=430, justify="left").pack(
                        side="left", fill="x", expand=True)
        if not shown:
            ttk.Label(inner, text="這個方法沒有可調參數。").pack(anchor="w")

        btns = ttk.Frame(outer)
        btns.pack(fill="x", side="bottom", pady=(10, 0))
        ttk.Button(btns, text="回復預設", command=self._reset).pack(
            side="left")
        ttk.Button(btns, text="取消", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="套用", command=self._apply).pack(
            side="right", padx=6)

        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _reset(self):
        """回到設定檔的預設值(只清這個方法的覆寫)。"""
        base = defaults_for(self.base_cfg, self.method)
        for key, (var, p) in self._vars.items():
            var.set(_fmt(base.get(key, p.lo), p))

    def _apply(self):
        out = {}
        for key, (var, p) in self._vars.items():
            try:
                out[key] = p.clamp(float(var.get()))
            except ValueError:
                messagebox.showerror("數值不正確",
                                     f"「{p.label}」不是數字:{var.get()}",
                                     parent=self)
                return
        # 只留下與設定檔不同的:覆寫檔越小,之後看得出到底動過什麼
        base = defaults_for(self.base_cfg, self.method)
        self.result = {k: v for k, v in out.items()
                       if abs(float(base.get(k, v)) - float(v)) > 1e-9}
        self.destroy()


def _fmt(value, p) -> str:
    if p.boolean:
        return "1" if value else "0"
    return str(int(round(float(value)))) if p.integer else f"{float(value):g}"
