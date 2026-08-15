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

from ui.analysis import Analysis, AnalysisJob, analyse_video

SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0)
SEEK_STEP_SEC = 5.0
# 一次最多補跳幾幀。設上限是為了避免「卡了一下 → 一次跳掉兩秒」,
# 那比稍微慢一點更難看清楚動作。
MAX_SKIP_FRAMES = 4
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


def progress_text(frac, found: int, done_sec: float, spent: float) -> str:
    """分析進度的一行字,含剩餘時間估計。

    一定要有 ETA:分析一支長片要好幾分鐘,只給百分比的話使用者分不出
    「很慢」與「當掉」。總幀數問不到時(.ts 錄影檔常見)改報「已處理到
    影片的第幾分鐘」——那至少看得出還在動。
    """
    found_txt = f"已找到 {found} 處抽菸"
    if not frac:
        return (f"分析中… 已處理到影片 {format_time(done_sec)}"
                f"(耗時 {format_time(spent)}),{found_txt}")
    eta = (spent / frac - spent) if frac > 0.02 else None
    tail = f",預計剩 {format_time(eta)}" if eta and eta > 1 else ""
    return f"分析中… {frac:.0%}{tail},{found_txt}"


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


class MarkerAxis(tk.Canvas):
    """播放器下方的標記軸:所有標記點都畫在上面,按一下就跳過去。

    與上面那條進度條的分工:進度條是「現在播到哪」,這條是「有哪些值得看
    的地方」。分開之後,標記不會被播放頭蓋住,點也可以畫得夠大按得到——
    進度條上的細線只夠看,按不準。
    """

    H = 46
    R = 7                     # 點的半徑;要按得到就不能太小

    def __init__(self, parent, on_seek):
        super().__init__(parent, height=self.H, highlightthickness=0,
                         background="#232323")
        self.on_seek = on_seek
        self.duration = 0.0
        self.marks = []       # [(時間, 種類)] 種類為 detect / manual
        self._hit = []        # [(x, 時間)] 供點擊比對
        self.bind("<Button-1>", self._click)
        self.bind("<Configure>", lambda _e: self.redraw())

    def _click(self, event):
        if not self._hit:
            return
        # 點最近的那一個(容差 12px);沒點中就當一般的時間軸跳轉
        x, best = event.x, None
        for hx, ht in self._hit:
            if abs(hx - x) <= 12 and (best is None or abs(hx - x) < best[0]):
                best = (abs(hx - x), ht)
        if best is not None:
            self.on_seek(best[1])
        elif self.duration > 0:
            w = max(self.winfo_width(), 1)
            self.on_seek(max(0.0, min(self.duration, x / w * self.duration)))

    def set_marks(self, marks) -> None:
        self.marks = sorted(marks, key=lambda m: m[0])
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = max(self.winfo_width(), 1)
        y = 18
        self.create_line(0, y, w, y, fill="#4a4a4a", width=2)
        self._hit = []
        if not self.marks:
            self.create_text(8, self.H - 12, anchor="w", fill="#777777",
                             text="(還沒有標記)")
            return
        for t, kind in self.marks:
            x = (t / self.duration) * w if self.duration > 0 else 0
            colour = "#ff4d4d" if kind == "detect" else "#ffd24d"
            self.create_oval(x - self.R, y - self.R, x + self.R, y + self.R,
                             fill=colour, outline="#1a1a1a")
            self.create_text(x, self.H - 12, text=format_time(t),
                             fill="#cccccc", font=("", 8))
            self._hit.append((x, t))


class VideoPlayer(tk.Toplevel):
    """播放單一影片,帶標記資料集所需的控制項。

    method / infer_config 是給「掃描抽菸」用的:掃描就是把整支影片
    跑一次即時管線,所以用哪個判定方法要跟主視窗選的一致,不然掃出來
    的位置跟主畫面的行為對不起來。
    """

    def __init__(self, master, path: str, title: str = "",
                 method=None, infer_config: str = "configs/inference.yaml",
                 analysis=None, job=None, overrides=None):
        super().__init__(master)
        self.path = str(path)
        self.title(title or Path(self.path).name)
        self.method = method
        self.infer_config = infer_config
        self.overrides = overrides or {}

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
        self._dropped = 0

        self._build()
        self._job = job
        if job is not None:
            # 模型載入要好幾秒,那段期間沒有進度可報。先講一句,
            # 免得看起來像什麼都沒發生
            self.status.set("背景分析中…(可以先看,標記會陸續出現)")
        if analysis:
            # 播放之前就分析好了(見 ui/analysis.py):標記與骨架直接就位,
            # 不必等使用者自己去按,也不必邊播邊算
            self.apply_analysis(analysis)
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

        self.axis = MarkerAxis(self, on_seek=self.seek_to)
        self.axis.duration = self.duration
        self.axis.pack(fill="x", padx=6, pady=(2, 0))

        row2 = ttk.Frame(self, padding=(6, 0))
        row2.pack(fill="x", pady=(0, 4))
        self.scan_btn = ttk.Button(row2, text="🔍 重新分析",
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
        # 追不上就丟幀——這是播放器該有的行為,不是妥協。60fps 的片子
        # 每幀只有 16.7ms 的預算,而 1600×900 解碼+繪圖要 12~25ms,
        # 不丟幀就只能愈拖愈慢(實測 1080p60 只有 0.36x)。
        # grab() 不做色彩轉換與回傳,比 read() 便宜得多。
        behind = now - self._next_at
        if behind > period:
            skip = min(int(behind / period), MAX_SKIP_FRAMES)
            for _ in range(skip):
                if not self.cap.grab():
                    break
                self._next_at += period
                self._dropped += 1
            now = time.perf_counter()
        if self._next_at < now - period * 2:
            self._next_at = now + period      # 落後太多(例如剛從背景切回來)就重新對時
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
        img = Image.fromarray(rgb)
        # 尺寸沒變就把畫素貼進既有的 PhotoImage,不要每幀重配一張——
        # 配置一張 1600×900 的 PhotoImage 約 4ms,在 60fps 的預算裡是硬傷
        if (self._photo is not None
                and (self._photo.width(), self._photo.height()) == img.size):
            self._photo.paste(img)
        else:
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfigure(self._item, image=self._photo)
        self.canvas.coords(self._item, cw // 2, ch // 2)

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

    def apply_analysis(self, a) -> None:
        """套用分析結果:抽菸位置進標記,骨架進快取。"""
        self.bar.detected = list(a.alarms)
        self.pose_cache = dict(a.poses)
        self._pose_stride = a.stride
        self._sync_marks()
        n = len(a.alarms)
        self.status.set(
            (f"分析完成:{n} 處抽菸,按下面的點可直接跳過去"
             if n else "分析完成:沒有找到抽菸警報")
            + (f";骨架 {len(self.pose_cache)} 幀已備妥"
               if self.pose_cache else ""))

    def _sync_marks(self) -> None:
        self.bar.redraw()
        self.axis.set_marks([(t, "detect") for t in self.bar.detected]
                            + [(t, "manual") for t in self.bar.manual])

    # ---------- 標記 ----------

    def add_manual(self):
        self.bar.manual.append(self.position)
        self._sync_marks()
        self.status.set(f"已標記 {format_time(self.position)}"
                        f"(共 {len(self.bar.manual)} 個手動標記)")

    def clear_marks(self):
        self.bar.manual.clear()
        self.bar.detected.clear()
        self._sync_marks()
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
        self.scan_btn.config(text="■ 停止")
        self.status.set("分析中…")
        self._scan_thread = threading.Thread(target=self._scan, daemon=True)
        self._scan_thread.start()

    def _scan(self):
        """整支跑一次分析(與播放前那一趟是同一個函式,行為一致)。"""
        try:
            a = analyse_video(
                self.path, method=self.method,
                infer_config=self.infer_config, overrides=self.overrides,
                on_progress=lambda *a: self._scan_q.put(("progress", a)),
                cancel=self._scan_cancel)
            if not self._scan_cancel.is_set():
                a.save(self.path)      # 存側車檔,下次開就不必再跑
            self._scan_q.put(("result", a))
        except Exception as e:
            self._scan_q.put(("error", f"分析失敗:{e}"))

    def _poll(self):
        """背景執行緒的結果一律在主執行緒消化(tkinter 非執行緒安全)。"""
        if not self._alive:
            return
        try:
            while True:
                kind, payload = self._scan_q.get_nowait()
                if kind == "progress":
                    self.status.set(progress_text(*payload))
                elif kind == "result":
                    self.apply_analysis(payload)
                    self.scan_btn.config(text="🔍 重新分析")
                    self.redraw_current()
                elif kind == "error":
                    self.scan_btn.config(text="🔍 重新分析")
                    self.status.set(payload)
        except Empty:
            pass
        self._watch_job()
        self._poll_id = self.after(80, self._poll)

    def _watch_job(self) -> None:
        """背景分析:跑完就把標記與骨架補上,中途顯示進度。"""
        job = getattr(self, "_job", None)
        if job is None:
            return
        if job.result is not None:
            self.apply_analysis(job.result)
            self._job = None
        elif job.error:
            self.status.set(f"背景分析失敗:{job.error}")
            self._job = None
        elif job.progress:
            self.status.set("(背景)" + progress_text(*job.progress))

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


class AnalysisDialog(tk.Toplevel):
    """播放前的分析進度視窗。

    三個出口,對應三種心情:等它跑完(標記最完整)、先播放(分析丟到背景,
    標記邊跑邊出現)、乾脆不要分析。分析大約要影片長度的兩倍時間,所以
    「先播放」這條路必須存在——不然二十分鐘的片就得乾等十分鐘。
    """

    def __init__(self, master, job):
        super().__init__(master)
        self.title("分析影片中")
        self.resizable(False, False)
        self.job = job
        self.outcome = "wait"          # wait / background / cancel

        frm = ttk.Frame(self, padding=14)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=Path(job.path).name, wraplength=460).pack(
            anchor="w")
        ttk.Label(frm, text="先跑一次抽菸判定,順便把骨架算好。"
                            "原始影片不會被改動,結果會存起來,"
                            "同一支片下次開就不用再等。",
                  foreground="#666666", wraplength=460,
                  justify="left").pack(anchor="w", pady=(2, 8))
        self.bar = ttk.Progressbar(frm, maximum=1.0, length=460)
        self.bar.pack(fill="x")
        self.msg = tk.StringVar(value="準備中…")
        ttk.Label(frm, textvariable=self.msg, wraplength=460,
                  justify="left").pack(anchor="w", pady=(6, 10))
        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="先播放(分析在背景繼續)",
                   command=lambda: self._finish("background")).pack(
            side="left")
        ttk.Button(btns, text="不分析,直接播放",
                   command=lambda: self._finish("cancel")).pack(side="left",
                                                                padx=6)

        self.protocol("WM_DELETE_WINDOW",
                      lambda: self._finish("background"))
        self.transient(master)
        self.grab_set()
        self._poll()

    def _finish(self, outcome: str):
        self.outcome = outcome
        if outcome == "cancel":
            self.job.cancel()
        self.destroy()

    def _poll(self):
        if self.job.error:
            messagebox.showerror("分析失敗", self.job.error, parent=self)
            self._finish("cancel")
            return
        if not self.job.running:
            self._finish("wait")
            return
        if self.job.progress:
            frac = self.job.progress[0]
            self.bar["value"] = frac or 0.0
            self.msg.set(progress_text(*self.job.progress))
        self.after(120, self._poll)


def open_video(master, path: str, title: str = "", method=None,
               infer_config: str = "configs/inference.yaml",
               overrides=None):
    """開一支影片:**先分析、再播放**,但不強迫你等完。

    有側車檔就直接載入(分析要影片長度的兩倍時間,每次重開都重跑沒人
    受得了);沒有才跑,而且可以選擇丟到背景先看片,標記邊跑邊出現。
    """
    a = Analysis.load(path)
    if a is not None:
        return VideoPlayer(master, path, title, method=method,
                           infer_config=infer_config, analysis=a,
                           overrides=overrides)

    job = AnalysisJob(path, method, infer_config, overrides)
    dlg = AnalysisDialog(master, job)
    master.wait_window(dlg)
    if dlg.outcome == "wait":
        return VideoPlayer(master, path, title, method=method,
                           infer_config=infer_config, analysis=job.result,
                           overrides=overrides)
    # 先播放:分析還在跑,播放器自己盯著它,好了就把標記補上
    return VideoPlayer(master, path, title, method=method,
                       infer_config=infer_config, overrides=overrides,
                       job=job if dlg.outcome == "background" else None)
