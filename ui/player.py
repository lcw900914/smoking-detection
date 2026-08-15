"""影片播放器:為「標記抽菸資料」設計,不是為了看片。

功能的取捨全部圍繞同一件事——**確認某一段到底是不是抽菸**:

- 逐幀與 0.25x 慢速:判準是「停留 2–5 秒」與「舉手 → 停留 → 放下」的
  轉折,正速看很難確認,慢速與逐幀才看得清楚
- 骨架疊加(可開關):直接看見系統「看到的」腕-鼻幾何,誤判時一眼就知道
  是姿態估計錯了還是判定規則錯了
- 「分析影片」一次算好:整支跑一次判定,**同時**收下警報位置與逐取樣點的
  關鍵點。警報畫成時間軸紅點,按 ◀▶ 直接跳過去;骨架則進快取,播放時
  只是查表畫線(約 1ms)。
  早期版本是播放時每幀重跑偵測(約 27ms),1.0x 只跑得到 0.84x——而那趟
  分析本來就已經算過同樣的東西,不留下來等於白算。
- 截圖:把某一幀存下來當證據或論文插圖,檔名帶幀號可以再對回來

**沒有音訊**:畫面是 OpenCV 逐幀解碼的。要聲音得換播放核心
(python-vlc / ffpyplayer),而判定完全用不到聲音,逐幀與任意變速在
無音訊下也單純得多(有音訊就得處理音畫同步)。
"""
import threading
import time
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import messagebox, ttk
from typing import List, Optional

import cv2
from PIL import Image, ImageTk

SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0)
SEEK_STEP_SEC = 5.0
BAR_H = 26
SNAPSHOT_DIR = "snapshots"


def format_time(seconds: float) -> str:
    """秒 → m:ss(超過一小時才顯示小時)。"""
    if seconds is None or seconds < 0 or seconds != seconds:   # NaN
        seconds = 0
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def next_marker(marks: List[float], now: float,
                forward: bool = True) -> Optional[float]:
    """下一個/上一個標記的時間;沒有就回 None。

    比對留 0.25 秒容差:跳過去之後停在標記上,再按一次不該原地不動。
    """
    if not marks:
        return None
    ordered = sorted(marks)
    if forward:
        later = [t for t in ordered if t > now + 0.25]
        return later[0] if later else None
    earlier = [t for t in ordered if t < now - 0.25]
    return earlier[-1] if earlier else None


def nearest_pose(cache: dict, idx: int, stride: int):
    """取第 idx 幀該用的骨架:往回找最近一個有資料的取樣點。

    分析是照管線的取樣率做的(30fps 的片子預設每 3 幀一次),不是每一幀
    都有。往回找而不是四捨五入到最近的:**畫出來的必須是系統在那個時刻
    已經看過的東西**,取用「還沒發生」的後一個取樣點會讓骨架超前畫面。
    往回沒有時才退而用下一個(影片開頭那幾幀)。
    """
    if not cache:
        return None
    stride = max(1, stride)
    base = (idx // stride) * stride
    return cache.get(base) or cache.get(base + stride)


def frame_delay_ms(fps: float, speed: float) -> int:
    """播放一幀該等幾毫秒。下限 10ms:Tk 的計時器再細也沒有意義。"""
    fps = fps if fps and fps > 0 else 25.0
    speed = speed if speed and speed > 0 else 1.0
    return max(10, int(1000.0 / (fps * speed)))


class SeekBar(tk.Canvas):
    """可拖曳的進度條,並把標記畫在上面。

    用 Canvas 而不是 ttk.Scale:Scale 沒辦法在軌道上畫標記,而「抽菸出現
    在哪幾個位置」正是這個播放器最想讓人一眼看到的東西。
    """

    TRACK = "#3a3a3a"
    FILL = "#4a9eff"
    DETECT = "#ff4d4d"      # 掃描出來的抽菸位置
    MANUAL = "#ffd24d"      # 手動標的位置

    def __init__(self, parent, on_seek):
        super().__init__(parent, height=BAR_H, highlightthickness=0,
                         background="#1e1e1e", cursor="hand2")
        self.on_seek = on_seek
        self.duration = 0.0
        self.position = 0.0
        self.detected: List[float] = []
        self.manual: List[float] = []
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._click)
        self.bind("<Configure>", lambda _e: self.redraw())

    def _click(self, event):
        if self.duration > 0:
            self.on_seek(self.x_to_t(event.x))

    def t_to_x(self, t: float) -> float:
        w = max(self.winfo_width(), 1)
        return (t / self.duration) * w if self.duration > 0 else 0

    def x_to_t(self, x: float) -> float:
        w = max(self.winfo_width(), 1)
        return max(0.0, min(self.duration, (x / w) * self.duration))

    def redraw(self) -> None:
        self.delete("all")
        w, h = max(self.winfo_width(), 1), BAR_H
        mid = h // 2
        self.create_rectangle(0, mid - 3, w, mid + 3, fill=self.TRACK,
                              width=0)
        self.create_rectangle(0, mid - 3, self.t_to_x(self.position),
                              mid + 3, fill=self.FILL, width=0)
        for marks, colour in ((self.detected, self.DETECT),
                              (self.manual, self.MANUAL)):
            for t in marks:
                x = self.t_to_x(t)
                self.create_rectangle(x - 1.5, 2, x + 1.5, h - 2,
                                      fill=colour, width=0)
        x = self.t_to_x(self.position)
        self.create_oval(x - 5, mid - 5, x + 5, mid + 5, fill="white",
                         width=0)


class VideoPlayer(tk.Toplevel):
    """播放單一影片,帶標記資料集所需的控制項。

    method / infer_config 是給「掃描抽菸」用的:掃描就是把整支影片
    跑一次即時管線,所以用哪個判定方法要跟主視窗選的一致,不然掃出來
    的位置跟主畫面的行為對不起來。
    """

    def __init__(self, master, path: str, title: str = "",
                 method=None, infer_config: str = "configs/inference.yaml"):
        super().__init__(master)
        self.path = str(path)
        self.title(title or Path(self.path).name)
        self.method = method
        self.infer_config = infer_config

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            messagebox.showerror("無法播放", f"開不起來:{self.path}")
            self.destroy()
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 <= fps <= 240.0 else 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.total / self.fps if self.total else 0.0

        self.playing = True
        self.speed = 1.0
        self.loop = tk.BooleanVar(value=True)
        self.pose_on = tk.BooleanVar(value=False)
        # 姿態快取:幀號 → 該幀所有人的關鍵點。由「分析影片」那一趟
        # 順便建好——那一趟本來就在跑姿態偵測,不留下來等於白算。
        # 播放時只是查表畫線(約 1ms),不再每幀重跑偵測(約 27ms)。
        self.pose_cache = {}
        self._pose_stride = 1
        self._frame = None            # 目前這一幀(原始 BGR,截圖用)
        self._photo = None            # 防 GC
        self._alive = True
        self._scan_thread = None
        self._scan_cancel = threading.Event()
        self._scan_q: Queue = Queue()
        self._frame_idx = 0
        self._next_at = time.perf_counter()   # 下一幀該出現的時刻
        self._last_bar_draw = 0.0

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._bind_keys()
        self.attributes("-topmost", True)
        self.after(500, lambda: self.attributes("-topmost", False))
        self._tick()
        self._poll()

    # ---------- 版面 ----------

    def _build(self):
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        s = min(1.0, 900 / max(1, src_w))
        w, h = int(src_w * s), int(src_h * s)

        self.canvas = tk.Canvas(self, background="#111111", width=w, height=h,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._item = self.canvas.create_image(w // 2, h // 2, anchor="center")
        # 暫停時也要跟著視窗縮放重畫。少了這行,暫停(逐幀細看時的常態)
        # 之後改變視窗大小,畫面會卡在舊的縮放比例、也不再置中。
        self.canvas.bind("<Configure>", lambda _e: self.redraw_current())

        self.bar = SeekBar(self, on_seek=self.seek_to)
        self.bar.duration = self.duration
        self.bar.pack(fill="x", padx=6, pady=(4, 0))

        row = ttk.Frame(self, padding=(6, 4))
        row.pack(fill="x")
        self.play_btn = ttk.Button(row, text="⏸", width=3,
                                   command=self.toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(row, text="⏮", width=3,
                   command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(row, text="⏭", width=3,
                   command=lambda: self.step(1)).pack(side="left")
        self.time_lbl = ttk.Label(row, text="0:00 / 0:00", width=14)
        self.time_lbl.pack(side="left", padx=8)

        ttk.Label(row, text="速度").pack(side="left")
        self.speed_var = tk.StringVar(value="1.0")
        sp = ttk.Combobox(row, textvariable=self.speed_var, width=5,
                          state="readonly",
                          values=[f"{s:g}" for s in SPEEDS])
        sp.pack(side="left", padx=4)
        sp.bind("<<ComboboxSelected>>", lambda _e: self._set_speed())

        ttk.Checkbutton(row, text="循環", variable=self.loop).pack(
            side="left", padx=6)
        ttk.Checkbutton(row, text="骨架", variable=self.pose_on,
                        command=self._toggle_pose).pack(side="left")
        ttk.Button(row, text="截圖", command=self.snapshot).pack(
            side="left", padx=6)
        ttk.Button(row, text="全螢幕", command=self.toggle_fullscreen).pack(
            side="left")

        row2 = ttk.Frame(self, padding=(6, 0))
        row2.pack(fill="x", pady=(0, 4))
        self.scan_btn = ttk.Button(row2, text="🔍 分析影片",
                                   command=self.toggle_scan)
        self.scan_btn.pack(side="left")
        ttk.Button(row2, text="◀ 標記",
                   command=lambda: self.jump_marker(False)).pack(side="left",
                                                                 padx=4)
        ttk.Button(row2, text="標記 ▶",
                   command=lambda: self.jump_marker(True)).pack(side="left")
        ttk.Button(row2, text="＋手動標記", command=self.add_manual).pack(
            side="left", padx=6)
        ttk.Button(row2, text="清除標記", command=self.clear_marks).pack(
            side="left")
        self.status = tk.StringVar(
            value="空白=播放/暫停　←→=±5秒　,.=逐幀　F=全螢幕　"
                  "S=截圖　M=標記　N/P=下/上一個標記")
        ttk.Label(row2, textvariable=self.status,
                  foreground="#666666").pack(side="left", padx=10)

    def _bind_keys(self):
        b = self.bind
        b("<space>", lambda _e: self.toggle_play())
        b("<Left>", lambda _e: self.seek_by(-SEEK_STEP_SEC))
        b("<Right>", lambda _e: self.seek_by(SEEK_STEP_SEC))
        b("<comma>", lambda _e: self.step(-1))
        b("<period>", lambda _e: self.step(1))
        b("<f>", lambda _e: self.toggle_fullscreen())
        b("<F>", lambda _e: self.toggle_fullscreen())
        b("<s>", lambda _e: self.snapshot())
        b("<S>", lambda _e: self.snapshot())
        b("<m>", lambda _e: self.add_manual())
        b("<M>", lambda _e: self.add_manual())
        b("<n>", lambda _e: self.jump_marker(True))
        b("<p>", lambda _e: self.jump_marker(False))
        b("<Escape>", lambda _e: self._escape())
        self.focus_set()

    # ---------- 播放 ----------

    @property
    def position(self) -> float:
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "▶")
        if self.playing:
            self._next_at = time.perf_counter()   # 重新對時,不補舊的幀
            self._tick()

    def _set_speed(self):
        try:
            self.speed = float(self.speed_var.get())
        except ValueError:
            self.speed = 1.0
        self._next_at = time.perf_counter()

    def step(self, delta: int):
        """逐幀。先暫停——逐幀的用途就是停下來細看。"""
        self.playing = False
        self.play_btn.config(text="▶")
        idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + delta - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        self._show_next(force_bar=True)

    def seek_by(self, seconds: float):
        self.seek_to(self.position + seconds)

    def seek_to(self, seconds: float):
        seconds = max(0.0, min(seconds, max(self.duration - 0.05, 0.0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * self.fps))
        self._next_at = time.perf_counter()
        self._show_next(force_bar=True)

    def _show_next(self, force_bar: bool = False) -> bool:
        ok, frame = self.cap.read()
        if not ok:
            return False
        self._frame = frame
        self._frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._render(frame, self._frame_idx)
        # 進度條與時間只更新到約 10 Hz。redraw() 會 delete("all") 再重畫
        # 所有標記,每幀都做在大視窗下就吃掉可觀的時間,而人眼根本看不出
        # 進度條每秒動 30 次跟動 10 次的差別。
        pos = self.position
        now = time.perf_counter()
        if force_bar or now - self._last_bar_draw >= 0.1:
            self._last_bar_draw = now
            self.bar.position = pos
            self.bar.redraw()
            self.time_lbl.config(
                text=f"{format_time(pos)} / {format_time(self.duration)}")
        return True

    def _tick(self):
        if not self._alive or not self.playing:
            return
        if not self._show_next():
            if self.loop.get():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not self._show_next(force_bar=True):
                    self.close()
                    return
            else:
                self.playing = False
                self.play_btn.config(text="▶")
                return
        # 解碼與繪圖的時間要從等待裡扣掉,否則週期會變成
        # 「工作時間 + 名目延遲」,播放永遠比實際慢,而且視窗越大越慢
        # (實測 640×480 只有 0.80x、1280×800 剩 0.68x)。
        period = frame_delay_ms(self.fps, self.speed) / 1000.0
        now = time.perf_counter()
        self._next_at += period
        if self._next_at < now - period:
            # 落後超過一整個週期(視窗很大、或開了骨架疊加)就重新對時。
            # 不重對的話會一直用 1ms 排程狂追,追不上還把 CPU 吃滿。
            self._next_at = now + period
        self.after(max(1, int((self._next_at - now) * 1000)), self._tick)

    def redraw_current(self):
        """用目前這一幀重畫(不前進)。視窗縮放與切換骨架疊加時用。"""
        if self._alive and self._frame is not None:
            self._render(self._frame, self._frame_idx)

    def _render(self, frame, idx: Optional[int] = None):
        if self.pose_on.get() and self.pose_cache:
            if idx is None:
                idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame = self._draw_pose(frame, max(0, idx))
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.coords(self._item, cw // 2, ch // 2)
        self.canvas.itemconfigure(self._item, image=self._photo)

    # ---------- 骨架疊加 ----------

    def _toggle_pose(self):
        """骨架疊加只吃快取,不做偵測——所以開關它不會讓播放變慢。"""
        if self.pose_on.get() and not self.pose_cache:
            self.status.set("還沒有骨架資料:請先按「分析影片」")
        self.redraw_current()          # 暫停時切換也要立刻看到差別

    def _cached_pose(self, idx: int):
        """取這一幀該用的骨架。

        分析是照管線的取樣率做的(預設 10fps,不是每一幀),所以往回找
        最近一個有資料的取樣點。**這樣反而更誠實**:畫出來的就是系統
        判定時真正看到的那一組關鍵點,不是另外補算的。
        """
        return nearest_pose(self.pose_cache, idx, self._pose_stride)

    def _draw_pose(self, frame, idx: int):
        """畫快取好的骨架。純繪圖(約 1ms),不做偵測,所以開著不影響速度。"""
        kpts_list = self._cached_pose(idx)
        if not kpts_list:
            return frame
        from inference.skeleton import draw_skeleton
        vis = frame.copy()
        for k in kpts_list:
            try:
                draw_skeleton(vis, k)
            except Exception:
                pass                  # 壞資料不該讓播放停掉
        return vis

    # ---------- 標記 ----------

    def add_manual(self):
        self.bar.manual.append(self.position)
        self.bar.redraw()
        self.status.set(f"已標記 {format_time(self.position)}"
                        f"(共 {len(self.bar.manual)} 個手動標記)")

    def clear_marks(self):
        self.bar.manual.clear()
        self.bar.detected.clear()
        self.bar.redraw()
        self.status.set("標記已清除")

    def jump_marker(self, forward: bool):
        t = next_marker(self.bar.detected + self.bar.manual,
                        self.position, forward)
        if t is None:
            self.status.set("沒有更多標記了")
            return
        self.seek_to(t)
        self.status.set(f"跳到 {format_time(t)}")

    # ---------- 分析:抽菸標記 + 骨架快取 ----------

    def toggle_scan(self):
        if self._scan_thread is not None and self._scan_thread.is_alive():
            self._scan_cancel.set()
            self.status.set("分析取消中…")
            return
        self._scan_cancel.clear()
        self.bar.detected.clear()
        self.pose_cache.clear()
        self.scan_btn.config(text="■ 停止分析")
        self.status.set("分析中…")
        self._scan_thread = threading.Thread(target=self._scan, daemon=True)
        self._scan_thread.start()

    def _scan(self):
        """整支影片跑一次即時管線,把警報時間收成標記。

        另外開一個 VideoCapture:播放用的那個由主執行緒在動,兩邊共用
        會互相把讀取位置搶掉。
        """
        try:
            from inference.pipeline import SmokingDetectionPipeline
            from utils import load_config
            cfg = load_config(self.infer_config)
            model_cfg = None
            ckpt = None
            if self.method is not None and self.method.needs_appearance:
                model_cfg = load_config("configs/model.yaml")
                ckpt = "checkpoints/hmdb_e2e_best.pt"
            pipe = SmokingDetectionPipeline(
                cfg, model_cfg, ckpt_path=ckpt,
                use_model=bool(model_cfg), method=self.method)
            hits = []
            pipe.alarm.callback = lambda tid, P, t, fr: hits.append(t)

            cap = cv2.VideoCapture(self.path)
            target = cfg["sampling"]["target_fps"]
            step = max(1, round(self.fps / target))
            i = 0
            try:
                while not self._scan_cancel.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if i % step == 0:
                        res = pipe.step(frame, i / self.fps)
                        # 這一趟本來就跑了姿態偵測,關鍵點順手留下來給
                        # 骨架疊加用——不留的話播放時就得每幀重跑一次
                        kp = [r["kpts"] for r in res.values()
                              if r.get("kpts") is not None]
                        if kp:
                            self._scan_q.put(("pose", (i, kp)))
                        if hits:
                            for t in hits:
                                self._scan_q.put(("hit", t))
                            hits.clear()
                        if i % (step * 50) == 0 and self.duration:
                            self._scan_q.put(
                                ("progress", (i / self.fps) / self.duration))
                    i += 1
            finally:
                cap.release()
                pipe.close()
            self._scan_q.put(("stride", step))
            self._scan_q.put(("done", self._scan_cancel.is_set()))
        except Exception as e:
            self._scan_q.put(("error", f"分析失敗:{e}"))

    def _poll(self):
        """背景執行緒的結果一律在主執行緒消化(tkinter 非執行緒安全)。"""
        if not self._alive:
            return
        try:
            while True:
                kind, payload = self._scan_q.get_nowait()
                if kind == "hit":
                    self.bar.detected.append(payload)
                    self.bar.redraw()
                elif kind == "progress":
                    self.status.set(
                        f"分析中… {payload:.0%}"
                        f"(抽菸 {len(self.bar.detected)} 處、"
                        f"骨架 {len(self.pose_cache)} 幀)")
                elif kind == "done":
                    self.scan_btn.config(text="🔍 分析影片")
                    n = len(self.bar.detected)
                    self.status.set(
                        ("分析已取消。" if payload else "分析完成。")
                        + (f"找到 {n} 處抽菸,按「標記 ▶」逐一查看"
                           if n else "沒有找到抽菸警報")
                        + (f";骨架已備妥 {len(self.pose_cache)} 幀"
                           if self.pose_cache else ""))
                    self.redraw_current()
                elif kind == "pose":
                    idx, kp = payload
                    self.pose_cache[idx] = kp
                elif kind == "stride":
                    self._pose_stride = payload
                elif kind == "error":
                    self.scan_btn.config(text="🔍 分析影片")
                    self.status.set(payload)
        except Empty:
            pass
        self._poll_id = self.after(80, self._poll)

    # ---------- 其他 ----------

    def snapshot(self):
        if self._frame is None:
            return
        out = Path(self.path).parent / SNAPSHOT_DIR
        out.mkdir(parents=True, exist_ok=True)
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 檔名帶幀號:之後想回到這一幀、或跟標記對照,都找得回來
        dst = out / f"{Path(self.path).stem}_f{idx:06d}.jpg"
        from utils import imwrite
        if imwrite(str(dst), self._frame):
            self.status.set(f"已存 {dst.name}")
        else:
            self.status.set("截圖失敗")

    def toggle_fullscreen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _escape(self):
        if self.attributes("-fullscreen"):
            self.attributes("-fullscreen", False)
        else:
            self.close()

    def close(self):
        self._alive = False
        self._scan_cancel.set()
        try:
            self.cap.release()
        except Exception:
            pass
        self.destroy()
