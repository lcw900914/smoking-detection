"""抽菸行為偵測 Demo GUI(Tkinter,無額外相依)。

**所有判定方法共用這一支 GUI**,由控制列的「方法」下拉選單切換
(清單來自 `inference/methods.py`,新增方法不必動這個檔案)。同一份輸入、
同一組門檻、只換選單那一格,方法之間的效果才比得起來。

視窗最下方是分頁列(像 Excel):

- **即時偵測** —— 偵測、追蹤、警報、第二階段複核
- **直播錄影** —— 原樣存檔不解碼,不掉幀,可長時間開著

兩者互不影響、可同時跑:錄影全程不解碼(見 `inference/recorder.py`),
不吃 GPU 也不跟偵測搶 CPU。

功能:
- 方法選擇:純規則 / 外觀網路 / 兩者融合 / 加不加第二階段複核
- 來源選擇:攝影機編號或影片檔(瀏覽)
- 模型權重選擇(僅需要外觀網路的方法會開放此欄)
- 即時畫面(框、track ID、階段、P_t 置信度條、警報狀態)
- 右側面板:每個 track 的置信度進度條與階段
- 觸發/解除門檻滑桿(即時生效)
- 警報事件記錄(觸發時同時截圖存 ./alarms)

用法(專案根目錄):
    python scripts/gui.py
    python scripts/gui.py --autotest smoke_run/m1_test.mp4   # 自動驗證模式
"""
import argparse
import os
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from inference import methods as methods_registry  # noqa: E402
from inference.downloader import (DEFAULT_OUT_DIR,  # noqa: E402
                                  VideoDownloader, human_duration,
                                  human_size)
from inference.recorder import (DEFAULT_KEEP_DAYS,  # noqa: E402
                                DEFAULT_ROOT as DEFAULT_REC_ROOT,
                                DEFAULT_SEGMENT_SEC, StreamRecorder,
                                day_name, prune_days, site_slug)
from inference.verifier import STATUS_NAMES  # noqa: E402
from ui.player import open_video  # noqa: E402
from utils import load_config  # noqa: E402

VIDEO_W, VIDEO_H = 800, 600
DEFAULT_CKPT = "checkpoints/hmdb_e2e_best.pt"
_STAGE_NAMES = {0: "S1 舉手", 1: "S2 嘴部", 2: "S3 放下", 3: "背景"}

# 在場型態(值的定義在 inference/state_machine.py:PresenceClassifier)
P_PASSING, P_RUNNING = "passing", "running"
P_WANDERING, P_WAITING = "wandering", "waiting"

_presence_names = None


def presence_name(key: str) -> str:
    """在場型態代號 → 中文。延後匯入 inference.pipeline(它會拉進 torch,
    啟動時就載會拖慢開窗),但仍以那邊的對照表為唯一來源。"""
    global _presence_names
    if _presence_names is None:
        from inference.pipeline import PRESENCE_NAMES
        _presence_names = PRESENCE_NAMES
    return _presence_names.get(key, "")


def triage(results: dict):
    """把當前所有 track 分成追蹤狀態面板的三欄。

    純函式:面板要顯示誰是整個畫面最常被調整的地方,抽出來才改得動、
    也測得到(tkinter 沒辦法在測試裡跑)。

    回傳 (在場, 等待待確認, 已偵測抽菸),每項為 (track_id, 顯示文字)。

    - **在場**:經過(走/跑)與徘徊的人 —— 也就是**還在動**的人
    - **等待待確認**:被判為「等待」(停下來不動)且**還沒**觸發抽菸警報。
      這一欄才是真正的候選:**站著不動的人才可疑,走過去的只是路過**。
      抽菸是站定了才做的事,所以「等待」是最值得盯的型態,不是最該忽略的
    - **已偵測抽菸**:警報成立中。第二階段複核把警報降為橘色時一併標示
      (降級不否決:警報還在,只是待人工複查)

    等待的人只出現在第二、三欄,不重複列進「在場」——三欄加起來就是全部,
    看板才不會同一個人到處都是。
    """
    present, watching, smoking = [], [], []
    for tid, r in sorted(results.items()):
        pres = r.get("presence", "unknown")
        alarm = bool(r.get("alarm"))
        events = r.get("events", 0)

        if pres != P_WAITING:
            bits = [f"ID{tid}", presence_name(pres) or "判定中",
                    _STAGE_NAMES.get(r.get("stage"), "")]
            if r.get("orientation") == "back":
                bits.append("背向")
            if events:
                bits.append(f"{events}次")
            if r.get("moving"):
                bits.append("移動中")
            if r.get("phone"):
                bits.append("講電話")
            present.append((tid, " ".join(b for b in bits if b)))

        if alarm:
            note = ""
            v = r.get("verify")
            if v:
                note = f" [{STATUS_NAMES.get(v, v)}]"
            smoking.append(
                (tid, f"ID{tid} P{r.get('P', 0.0):.2f} {events}次{note}"))
        elif pres == P_WAITING:
            lv = r.get("level", 0.0)
            tag = ("警戒高" if lv >= 0.8 else "警戒中" if lv >= 0.5
                   else "警戒低" if lv >= 0.2 else "觀察中")
            # 欄名已經寫了「等待」,不再重複;背向要標,因為背向時骨架
            # 棄權,那個人是「看不到手」而不是「確認沒抽」
            back = " 背向" if r.get("orientation") == "back" else ""
            watching.append((tid, f"ID{tid} {events}次 {tag}{back}"))
    return present, watching, smoking


class SlotPanel:
    """固定格數的清單面板:track 進出只改文字、不增刪列,版面完全靜止。

    原本只有一欄時這段邏輯散在 DemoGUI 裡;拆成三欄後獨立出來,三欄共用
    同一套「格位 + 寬限」行為 —— track 短暫消失時格位保留 GRACE_SEC 秒,
    偵測閃爍才不會讓整份清單上下跳動。
    """

    GRACE_SEC = 2.0

    def __init__(self, parent, title: str, width: int, slots: int = 8,
                 bar: bool = False):
        self.frame = ttk.LabelFrame(parent, text=title, padding=4,
                                    width=width)
        self.frame.pack(side="left", fill="y", padx=(0, 4))
        self.frame.pack_propagate(False)
        self.slots = []
        for _ in range(slots):
            row = ttk.Frame(self.frame)
            row.pack(fill="x", pady=1)
            pb = ttk.Progressbar(row, maximum=1.0, length=60) if bar else None
            if pb is not None:
                pb.pack(side="right")
            lbl = ttk.Label(row, text="—", anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            self.slots.append((lbl, pb))
        self.n = slots
        self.tid_slot = {}
        self.tid_seen = {}

    def update(self, items, now: float, values=None) -> None:
        """items 為 [(track_id, 顯示文字)];values 可給 {track_id: 0~1} 走進度條。"""
        for tid, text in items:
            if tid not in self.tid_slot:
                used = set(self.tid_slot.values())
                free = [i for i in range(self.n) if i not in used]
                if not free:
                    continue          # 格位滿:不擠掉現有列,多的暫不顯示
                self.tid_slot[tid] = free[0]
            self.tid_seen[tid] = now
            lbl, pb = self.slots[self.tid_slot[tid]]
            lbl.config(text=text)
            if pb is not None:
                pb["value"] = (values or {}).get(tid, 0.0)

        for tid in [t for t, ts in self.tid_seen.items()
                    if now - ts > self.GRACE_SEC]:
            slot = self.tid_slot.pop(tid, None)
            self.tid_seen.pop(tid, None)
            if slot is not None:
                lbl, pb = self.slots[slot]
                lbl.config(text="—")
                if pb is not None:
                    pb["value"] = 0.0


class DemoGUI:
    """主視窗:左側影像、右側追蹤狀態、下方控制列與警報記錄。"""

    def __init__(self, root: tk.Tk,
                 infer_config: str = "configs/inference.yaml"):
        self.root = root
        self.infer_config = infer_config
        # 門檻初始值只在啟動時讀一次設定檔;之後以使用者輸入為準
        try:
            alarm_cfg = load_config(infer_config)["alarm"]
            self._init_trigger = float(alarm_cfg["trigger_threshold"])
            self._init_release = float(alarm_cfg["release_threshold"])
            self._init_min_events = int(alarm_cfg.get("min_events", 3))
            cfg_all = load_config(infer_config)
            self._init_move_gate = bool(
                cfg_all.get("move_gate", {}).get("enabled", True))
            self._init_wander_alert = bool(
                cfg_all.get("presence", {}).get("alert_wandering", False))
            esc = cfg_all.get("escalation", {})
            self._init_dwell_min = float(esc.get("min_dwell", 2.0))
            self._init_dwell_max = float(esc.get("max_dwell", 5.0))
        except Exception:
            self._init_trigger, self._init_release = 0.75, 0.4
            self._init_min_events = 3
            self._init_move_gate = True
            self._init_wander_alert = False
            self._init_dwell_min, self._init_dwell_max = 2.0, 5.0
        root.title("抽菸行為偵測 Demo — channel-as-temporal-buffer")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.pipeline = None
        self.worker: threading.Thread = None
        self.running = False
        self.frame_q: "queue.Queue" = queue.Queue(maxsize=2)
        self.alarm_q: "queue.Queue" = queue.Queue()
        # 錄影分頁(獨立於偵測:不解碼、不吃 GPU,兩者可同時開著)
        self.recorder = None
        self.rec_thread: threading.Thread = None
        self.rec_q: "queue.Queue" = queue.Queue()
        self._rec_status_t = 0.0
        # 影片下載分頁(與錄影分開:影片有結尾,直播沒有)
        self.downloader = None
        self.dl_thread: threading.Thread = None
        self.dl_info = None
        self.dl_q: "queue.Queue" = queue.Queue()
        self.dl_prog_q: "queue.Queue" = queue.Queue()
        self.dl_meta_q: "queue.Queue" = queue.Queue()
        self._dl_scan = 0          # 換資料夾時作廢上一輪的縮圖解碼
        self._poll_id = None

        # 警報片段錄製:滾動保留最近 ~10 秒取樣影格(縮小節省記憶體),
        # 警報觸發時連同後續 4 秒寫成 mp4,供記錄點擊回放
        self.CLIP_PRE_FRAMES = 100     # 約 10 秒 @10fps
        self.CLIP_POST_SEC = 4.0
        self.clip_buffer: deque = deque(maxlen=self.CLIP_PRE_FRAMES)
        # 節點滾動緩衝(與 clip_buffer 同步):錄乾淨影像時畫面上沒有
        # 紅框,stage2/extract_pose.py 事後無法回抽對象,節點只能在
        # 錄影當下就落地。每幀存 {track id: (kpts, bbox)},成本近零。
        self.pose_buffer: deque = deque(maxlen=self.CLIP_PRE_FRAMES)
        self._active_recs = []         # 錄製中的警報片段
        # 錄影疊加開關(預設關 = 錄乾淨影像)。要訓練「看畫面」的模型
        # 就必須有沒烙印骨架線與文字的原始影格,印上去救不回來;
        # 畫面顯示不受影響,一律照常疊加。
        # 影像執行緒只讀這個快取的 bool,不直接碰 tk 變數(非執行緒安全)
        self._clip_overlay = False
        self._clip_overlay_applied = False

        # 條列式記錄(分頁)
        self.log_entries = []          # 最新在前;{'text', 'rec'}
        self.PAGE_SIZE = 8
        self.page = 1
        self._last_draw = 0.0          # 畫面重繪節流用

        self._build_layout()
        self._poll_ui()

        # 啟動時帶到最前(短暫 topmost 再釋放,避免永遠壓住其他視窗)
        root.lift()
        root.attributes("-topmost", True)
        root.after(1500, lambda: root.attributes("-topmost", False))

    # ---------- 版面 ----------

    def _build_layout(self):
        # 分頁列放在最下面(像 Excel)。ttk 的 tabposition 是兩個字元:
        # 第一個是貼哪一邊(s = 下),第二個是沿該邊靠哪邊對齊(w = 左)。
        # vista 主題實測支援。
        style = ttk.Style(self.root)
        style.configure("Bottom.TNotebook", tabposition="sw")
        self.tabs = ttk.Notebook(self.root, style="Bottom.TNotebook")
        self.tabs.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        detect_tab = ttk.Frame(self.tabs)
        record_tab = ttk.Frame(self.tabs)
        download_tab = ttk.Frame(self.tabs)
        self.tabs.add(detect_tab, text="  即時偵測  ")
        self.tabs.add(record_tab, text="  直播錄影  ")
        self.tabs.add(download_tab, text="  影片下載  ")
        detect_tab.columnconfigure(0, weight=1)
        detect_tab.rowconfigure(0, weight=1)

        main = ttk.Frame(detect_tab, padding=6)
        main.grid(sticky="nsew")
        # 影像格可伸縮:拉大視窗時影像跟著放大(_draw_frame 依實際尺寸縮放)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # 左:影像畫面
        # 用 tk.Canvas:要求尺寸固定為初始值,實際顯示隨視窗伸縮——
        # 若用 Label 顯示,放大後的圖會成為新的最小尺寸(ratchet),
        # 把下方控制列擠出視窗
        self.canvas = tk.Canvas(main, background="#222222",
                                width=VIDEO_W, height=VIDEO_H,
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self._canvas_item = self.canvas.create_image(
            VIDEO_W // 2, VIDEO_H // 2, anchor="center")
        self._show_placeholder("選擇來源後按「開始」(視窗可拉大)")

        # 右:追蹤狀態 + 門檻(固定寬度,文字長度變化不會抖動)
        side = ttk.LabelFrame(main, text="追蹤狀態", padding=6,
                              width=706, height=300)
        side.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        # 注意:內部子元件用 pack 排版,尺寸傳播要用 pack_propagate 關;
        # grid_propagate 只擋 grid 子元件,關錯了面板仍會隨文字伸縮
        side.pack_propagate(False)
        side.grid_propagate(False)
        self.track_panel = ttk.Frame(side)
        self.track_panel.pack(fill="both", expand=True)

        # 三欄:在場 → 觀察中 → 已判定。由左到右就是一個人被處理的順序,
        # 看板一眼就能回答「現在有誰、誰還在觀察、誰要處理」。
        # 各欄都用固定格數(見 SlotPanel):track 閃爍時清單不會上下跳動。
        self.panel_present = SlotPanel(
            self.track_panel, "在場(經過·徘徊)", width=300, bar=True)
        self.panel_watch = SlotPanel(
            self.track_panel, "等待·待確認", width=180)
        self.panel_smoke = SlotPanel(
            self.track_panel, "⚠ 偵測到抽菸", width=190)

        thr = ttk.LabelFrame(main, text="警報門檻", padding=6)
        thr.grid(row=1, column=1, sticky="sew", padx=(6, 0))
        self.trigger_var = tk.DoubleVar(value=self._init_trigger)
        self.release_var = tk.DoubleVar(value=self._init_release)
        for text, var in (("觸發", self.trigger_var), ("解除", self.release_var)):
            row = ttk.Frame(thr)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=text, width=6).pack(side="left")
            s = ttk.Scale(row, from_=0.0, to=1.0, variable=var,
                          command=self._apply_thresholds)
            s.pack(side="left", fill="x", expand=True)
            lbl = ttk.Label(row, width=5)
            lbl.pack(side="left")
            var.trace_add("write",
                          lambda *_, v=var, l=lbl: l.config(text=f"{v.get():.2f}"))
            lbl.config(text=f"{var.get():.2f}")

        # 通報次數:手到嘴事件 ≥ N 次才允許紅色警報(即時生效)
        row = ttk.Frame(thr)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="次數", width=6).pack(side="left")
        self.min_events_var = tk.IntVar(value=self._init_min_events)
        ttk.Spinbox(row, from_=1, to=10, width=4,
                    textvariable=self.min_events_var,
                    command=self._apply_thresholds).pack(side="left")
        ttk.Label(row, text=" 次手到嘴才通報").pack(side="left")
        self.min_events_var.trace_add(
            "write", lambda *_: self._apply_thresholds())

        # 停留窗口:min~max 秒的手到嘴停留才算抽菸一口(現場可校準)
        row = ttk.Frame(thr)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="停留", width=6).pack(side="left")
        self.dwell_min_var = tk.DoubleVar(value=self._init_dwell_min)
        self.dwell_max_var = tk.DoubleVar(value=self._init_dwell_max)
        ttk.Spinbox(row, from_=0.5, to=10.0, increment=0.5, width=4,
                    textvariable=self.dwell_min_var,
                    command=self._apply_thresholds).pack(side="left")
        ttk.Label(row, text=" ~ ").pack(side="left")
        ttk.Spinbox(row, from_=1.0, to=20.0, increment=0.5, width=4,
                    textvariable=self.dwell_max_var,
                    command=self._apply_thresholds).pack(side="left")
        ttk.Label(row, text=" 秒才算一口(短=扶眼鏡 長=講電話)").pack(
            side="left")
        for v in (self.dwell_min_var, self.dwell_max_var):
            v.trace_add("write", lambda *_: self._apply_thresholds())

        # 移動排除開關:走動中(累積移動 ≥ 3 倍身高)不視為抽菸
        self.move_gate_var = tk.BooleanVar(value=self._init_move_gate)
        ttk.Checkbutton(thr, text="移動排除(走動中不通報)",
                        variable=self.move_gate_var,
                        command=self._apply_thresholds).pack(
            anchor="w", pady=2)

        # 徘徊通報開關(預設關):有人在鏡頭裡一直繞但沒離開 → 橘色警示。
        # 與抽菸警報是不同語意的事件,不影響紅色警報與 P_t
        self.wander_var = tk.BooleanVar(value=self._init_wander_alert)
        ttk.Checkbutton(thr, text="徘徊時通報(橘色,非抽菸警報)",
                        variable=self.wander_var,
                        command=self._apply_thresholds).pack(
            anchor="w", pady=2)

        # 診斷訊息開關(預設關):校準時才顯示未計入/未達門檻的原因
        # 注意:工作執行緒不可直接讀 tk 變數(tkinter 非執行緒安全),
        # 由 _apply_thresholds 在主執行緒快取成 _diag_enabled
        self.diag_var = tk.BooleanVar(value=False)
        self._diag_enabled = False
        ttk.Checkbutton(thr, text="顯示診斷訊息(校準用)",
                        variable=self.diag_var,
                        command=self._apply_thresholds).pack(
            anchor="w", pady=2)

        # 錄影疊加開關(預設關):關 = 存乾淨原始影像(蒐集訓練資料用),
        # 開 = 存與畫面相同的疊加版(給人複查用)。只影響存檔,不影響顯示
        self.clip_overlay_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(thr, text="錄影疊加骨架(蒐集訓練資料請關閉)",
                        variable=self.clip_overlay_var,
                        command=self._apply_thresholds).pack(
            anchor="w", pady=2)

        # 下:控制列(上排選方法、下排選來源與權重)
        ctrl = ttk.Frame(main, padding=(0, 6))
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew")

        # 方法選單:清單直接來自 inference/methods.py 的登錄表。
        # 新增一種抽菸判定方法時這裡不用改任何一行 —— 論文要比較各方法時,
        # 換的只有這一格,輸入與門檻都不動,比較才公平
        mrow = ttk.Frame(ctrl)
        mrow.pack(fill="x")
        ttk.Label(mrow, text="方法").pack(side="left")
        self.method_var = tk.StringVar()
        self.method_box = ttk.Combobox(
            mrow, textvariable=self.method_var, state="readonly", width=40,
            values=[self._method_label(m) for m in methods_registry.METHODS])
        self.method_box.pack(side="left", padx=4)
        self.method_box.bind("<<ComboboxSelected>>",
                             lambda _e: self._on_method_change())
        self.method_desc = ttk.Label(mrow, text="", foreground="#555555",
                                     wraplength=760, justify="left")
        self.method_desc.pack(side="left", padx=8, fill="x", expand=True)

        row1 = ttk.Frame(ctrl)
        row1.pack(fill="x", pady=(4, 0))
        ttk.Label(row1, text="來源").pack(side="left")
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(row1, textvariable=self.source_var, width=32).pack(
            side="left", padx=4)
        ttk.Button(row1, text="瀏覽…", command=self._browse_video).pack(
            side="left")
        self.ckpt_label = ttk.Label(row1, text="權重")
        self.ckpt_label.pack(side="left", padx=(12, 0))
        self.ckpt_var = tk.StringVar(
            value=DEFAULT_CKPT if Path(DEFAULT_CKPT).exists() else "")
        self.ckpt_entry = ttk.Entry(row1, textvariable=self.ckpt_var, width=36)
        self.ckpt_entry.pack(side="left", padx=4)
        self.ckpt_btn = ttk.Button(row1, text="…", width=3,
                                   command=self._browse_ckpt)
        self.ckpt_btn.pack(side="left")
        self.start_btn = ttk.Button(row1, text="▶ 開始", command=self.start)
        self.start_btn.pack(side="left", padx=(12, 2))
        self.stop_btn = ttk.Button(row1, text="■ 停止", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left")
        self.status_var = tk.StringVar(value="待機")
        ttk.Label(row1, textvariable=self.status_var).pack(
            side="left", padx=12)
        # 預設選項與相依欄位狀態(權重欄只在方法需要時開放)
        self.method_var.set(self._method_label(methods_registry.default()))
        self._on_method_change()

        # 警報記錄:條列式 + 分頁;警報項目可雙擊回放片段
        log_frame = ttk.LabelFrame(main, text="警報記錄(雙擊警報項目可回放片段)",
                                   padding=4)
        log_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.log = tk.Listbox(log_frame, height=6)
        self.log.pack(fill="both", expand=True)
        self.log.bind("<Double-Button-1>", self._on_log_dclick)

        pager = ttk.Frame(log_frame)
        pager.pack(fill="x", pady=(4, 0))
        ttk.Button(pager, text="◀ 上一頁", width=8,
                   command=lambda: self._goto_page(self.page - 1)).pack(
            side="left")
        ttk.Label(pager, text=" 第").pack(side="left")
        self.page_var = tk.IntVar(value=1)
        self.page_spin = ttk.Spinbox(
            pager, from_=1, to=1, width=4, textvariable=self.page_var,
            command=lambda: self._goto_page(self.page_var.get()))
        self.page_spin.pack(side="left", padx=2)
        self.page_total_lbl = ttk.Label(pager, text="/ 1 頁")
        self.page_total_lbl.pack(side="left")
        ttk.Button(pager, text="下一頁 ▶", width=8,
                   command=lambda: self._goto_page(self.page + 1)).pack(
            side="left", padx=(8, 0))

        self._build_record_tab(record_tab)
        self._build_download_tab(download_tab)

    # ---------- 分頁二:直播錄影 ----------

    def _build_record_tab(self, parent):
        """錄影分頁。錄影與偵測是兩件不同的事,共用視窗但不共用狀態:
        錄影全程不解碼(見 inference/recorder.py),不吃 GPU,可以和偵測
        同時開著跑。"""
        f = ttk.Frame(parent, padding=8)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="直播錄影:原樣存檔不解碼,不會掉幀,可長時間開著",
                  foreground="#444444").grid(row=0, column=0, columnspan=4,
                                             sticky="w", pady=(0, 8))

        ttk.Label(f, text="直播網址").grid(row=1, column=0, sticky="w")
        self.rec_url_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.rec_url_var).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=2)

        ttk.Label(f, text="存放資料夾").grid(row=2, column=0, sticky="w")
        self.rec_root_var = tk.StringVar(value=DEFAULT_REC_ROOT)
        ttk.Entry(f, textvariable=self.rec_root_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(f, text="瀏覽…", command=self._browse_rec_root).grid(
            row=2, column=3, sticky="w")

        # 空間預估:720p 約 30 GB/天,保留三天要 90 GB。錄到一半磁碟滿了,
        # ffmpeg 只會安靜地停住,所以要在按下開始之前就講清楚
        self.rec_disk_lbl = ttk.Label(f, text="")
        self.rec_disk_lbl.grid(row=3, column=1, columnspan=3, sticky="w",
                               padx=4)
        self.rec_root_var.trace_add("write", lambda *_: self._update_disk_hint())

        opts = ttk.Frame(f)
        opts.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 2))
        ttk.Label(opts, text="每段").pack(side="left")
        self.rec_seg_var = tk.IntVar(value=DEFAULT_SEGMENT_SEC)
        ttk.Spinbox(opts, from_=30, to=3600, increment=30, width=6,
                    textvariable=self.rec_seg_var).pack(side="left", padx=4)
        ttk.Label(opts, text="秒     保留").pack(side="left")
        self.rec_keep_var = tk.IntVar(value=DEFAULT_KEEP_DAYS)
        ttk.Spinbox(opts, from_=1, to=30, width=4,
                    textvariable=self.rec_keep_var).pack(side="left", padx=4)
        ttk.Label(opts, text="天(每天一個資料夾,逾期自動刪除)     畫質上限")\
            .pack(side="left")
        self.rec_height_var = tk.StringVar(value="720")
        ttk.Combobox(opts, textvariable=self.rec_height_var, width=6,
                     state="readonly",
                     values=("360", "480", "720", "1080")).pack(side="left",
                                                                padx=4)
        self.rec_audio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="錄音(預設關)",
                        variable=self.rec_audio_var).pack(side="left", padx=8)

        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, columnspan=4, sticky="w", pady=6)
        self.rec_start_btn = ttk.Button(btns, text="● 開始錄影",
                                        command=self.start_record)
        self.rec_start_btn.pack(side="left")
        self.rec_stop_btn = ttk.Button(btns, text="■ 停止錄影",
                                       command=self.stop_record,
                                       state="disabled")
        self.rec_stop_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="開啟資料夾",
                   command=self._open_rec_folder).pack(side="left", padx=4)
        ttk.Button(btns, text="檢查保留(不刪除)",
                   command=self._preview_prune).pack(side="left", padx=4)
        self.rec_status_var = tk.StringVar(value="未開始")
        ttk.Label(btns, textvariable=self.rec_status_var).pack(
            side="left", padx=12)

        log_box = ttk.LabelFrame(f, text="錄影記錄", padding=4)
        log_box.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(6, 0))
        f.rowconfigure(6, weight=1)
        self.rec_log = tk.Listbox(log_box, height=12)
        self.rec_log.pack(fill="both", expand=True)

        self._update_disk_hint()

    def _show_placeholder(self, text: str):
        """在影像區顯示固定像素尺寸的佔位畫面(含提示文字)。"""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (VIDEO_W, VIDEO_H), (34, 34, 34))
        d = ImageDraw.Draw(img)
        try:  # PIL 預設字型不含中文,改用微軟正黑體
            font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 22)
        except OSError:
            font = ImageFont.load_default()
        d.text((VIDEO_W // 2, VIDEO_H // 2), text,
               fill=(200, 200, 200), anchor="mm", font=font)
        self._set_canvas_image(ImageTk.PhotoImage(img))

    def _set_canvas_image(self, photo) -> None:
        """把影像置中放上 canvas(保留參照防 GC)。"""
        cw = max(self.canvas.winfo_width(), VIDEO_W)
        ch = max(self.canvas.winfo_height(), VIDEO_H)
        self.canvas.coords(self._canvas_item, cw // 2, ch // 2)
        self.canvas.itemconfigure(self._canvas_item, image=photo)
        self.canvas.image = photo

    # ---------- 分頁三:影片下載 ----------

    def _build_download_tab(self, parent):
        """把 YouTube 等網站的**影片**存成檔案。

        與「直播錄影」分頁的分工只有一件事:來源會不會結束。直播沒有結尾,
        所以錄影器把「讀不到東西」當成斷線去重連;影片有結尾,那個假設會
        變成無限重錄同一支,所以下載走這一條、交給 yt-dlp。
        """
        f = ttk.Frame(parent, padding=8)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="下載影片存檔(供離線標記與訓練)。直播請改用"
                          "「直播錄影」分頁 —— 直播沒有結尾,下載會一直"
                          "下到磁碟滿。",
                  foreground="#444444", wraplength=900, justify="left").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(f, text="影片網址").grid(row=1, column=0, sticky="w")
        self.dl_url_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.dl_url_var).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(f, text="查詢資訊", command=self.probe_download).grid(
            row=1, column=3, sticky="w")

        ttk.Label(f, text="存放資料夾").grid(row=2, column=0, sticky="w")
        self.dl_dir_var = tk.StringVar(value=DEFAULT_OUT_DIR)
        ttk.Entry(f, textvariable=self.dl_dir_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(f, text="瀏覽…", command=self._browse_dl_dir).grid(
            row=2, column=3, sticky="w")

        opts = ttk.Frame(f)
        opts.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 2))
        ttk.Label(opts, text="畫質上限").pack(side="left")
        self.dl_height_var = tk.StringVar(value="720")
        ttk.Combobox(opts, textvariable=self.dl_height_var, width=6,
                     state="readonly",
                     values=("360", "480", "720", "1080")).pack(side="left",
                                                                padx=4)
        self.dl_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="含音訊",
                        variable=self.dl_audio_var).pack(side="left", padx=8)
        ttk.Label(opts, text="(720p 以上要影音合流,已內建 ffmpeg)",
                  foreground="#777777").pack(side="left")

        self.dl_info_lbl = ttk.Label(f, text="", foreground="#444444",
                                     wraplength=900, justify="left")
        self.dl_info_lbl.grid(row=4, column=0, columnspan=4, sticky="w",
                              pady=(4, 2))

        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, columnspan=4, sticky="w", pady=6)
        self.dl_start_btn = ttk.Button(btns, text="▼ 開始下載",
                                       command=self.start_download)
        self.dl_start_btn.pack(side="left")
        self.dl_cancel_btn = ttk.Button(btns, text="✕ 取消",
                                        command=self.cancel_download,
                                        state="disabled")
        self.dl_cancel_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="開啟資料夾",
                   command=self._open_dl_folder).pack(side="left", padx=4)
        self.dl_status_var = tk.StringVar(value="待機")
        ttk.Label(btns, textvariable=self.dl_status_var).pack(side="left",
                                                              padx=12)

        self.dl_bar = ttk.Progressbar(f, maximum=1.0)
        self.dl_bar.grid(row=6, column=0, columnspan=4, sticky="ew", pady=2)

        list_box = ttk.LabelFrame(f, text="這個資料夾裡的影片(雙擊播放)",
                                  padding=4)
        list_box.grid(row=7, column=0, columnspan=4, sticky="nsew",
                      pady=(6, 0))
        f.rowconfigure(7, weight=1)
        head = ttk.Frame(list_box)
        head.pack(fill="x", pady=(0, 4))
        ttk.Button(head, text="↻ 重新整理",
                   command=self.refresh_dl_list).pack(side="left")
        self.dl_count_lbl = ttk.Label(head, text="", foreground="#666666")
        self.dl_count_lbl.pack(side="left", padx=8)
        self.dl_list = VideoList(list_box, on_open=self._play_video)
        self.dl_list.pack(fill="both", expand=True)
        self.refresh_dl_list()

    def _browse_dl_dir(self):
        p = filedialog.askdirectory(title="選擇影片存放資料夾")
        if p:
            self.dl_dir_var.set(p)
            self.refresh_dl_list()

    def _play_video(self, path):
        """雙擊或按播放:沿用警報片段那個回放視窗。"""
        if not Path(path).exists():
            messagebox.showerror("找不到檔案", f"{path}\n可能已被移動或刪除。")
            self.refresh_dl_list()
            return
        open_video(self.root, str(path), Path(path).name,
                   method=self._current_method(),
                   infer_config=self.infer_config)

    def refresh_dl_list(self):
        """重掃資料夾、重建清單,縮圖交給背景執行緒解碼。

        解碼要開 VideoCapture 並 seek,檔案一多會卡住介面,所以放背景;
        但 PhotoImage 只能在主執行緒建,背景只回傳 numpy 陣列。
        """
        folder = self.dl_dir_var.get().strip() or DEFAULT_OUT_DIR
        files = list_videos(folder)
        self.dl_count_lbl.config(
            text=f"{len(files)} 部影片　{folder}")
        pending = self.dl_list.set_files(files)
        if not pending:
            return
        self._dl_scan += 1
        scan = self._dl_scan

        def work():
            for p in pending:
                if scan != self._dl_scan:
                    return          # 使用者又換了資料夾,這輪的結果作廢
                self.dl_meta_q.put(video_meta(p))
        threading.Thread(target=work, daemon=True).start()

    def _open_dl_folder(self):
        d = Path(self.dl_dir_var.get().strip() or DEFAULT_OUT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))         # noqa: S606 (Windows 專用)

    def _new_downloader(self) -> "VideoDownloader":
        return VideoDownloader(
            self.dl_url_var.get().strip(),
            out_dir=self.dl_dir_var.get().strip() or DEFAULT_OUT_DIR,
            max_height=int(self.dl_height_var.get()),
            audio=bool(self.dl_audio_var.get()),
            log=self.dl_q.put,
            on_progress=self.dl_prog_q.put)

    def probe_download(self):
        """先查清楚再下載:標題對不對、多長、多大、是不是直播。"""
        if not self.dl_url_var.get().strip():
            messagebox.showerror("缺少網址", "請先填入影片網址。")
            return
        self.dl_status_var.set("查詢中…")

        def work():
            try:
                info = self._new_downloader().probe()
            except Exception as e:
                self.dl_q.put(f"[錯誤] 查詢失敗:{e}")
                self.dl_info = None
                return
            self.dl_info = info
            self.dl_q.put(
                f"[資訊] {info['title']}｜{human_duration(info['duration'])}"
                f"｜約 {human_size(info['size'])}"
                + ("｜⚠ 這是直播" if info["is_live"] else ""))
        threading.Thread(target=work, daemon=True).start()

    def start_download(self):
        if self.downloader is not None:
            return
        if not self.dl_url_var.get().strip():
            messagebox.showerror("缺少網址", "請先填入影片網址。")
            return
        dl = self._new_downloader()
        self.downloader = dl
        self.dl_start_btn.config(state="disabled")
        self.dl_cancel_btn.config(state="normal")
        self.dl_status_var.set("準備中…")
        self.dl_bar["value"] = 0.0

        def work():
            try:
                info = dl.probe()
                if info["is_live"]:
                    # 直播沒有結尾,下載會一直下到磁碟滿。直接擋下來,
                    # 而不是讓使用者半夜發現磁碟爆了
                    self.dl_q.put("[錯誤] 這是直播,不能用下載。"
                                  "請改用「直播錄影」分頁。")
                    return
                self.dl_q.put(f"[下載] {info['title']}"
                              f"({human_duration(info['duration'])},"
                              f"約 {human_size(info['size'])})")
                path = dl.run()
                if path is not None:
                    self.dl_q.put(f"[完成] {path}")
            except Exception as e:
                self.dl_q.put(f"[錯誤] {e}")
            finally:
                # 不從工作執行緒碰 UI:tkinter 不是執行緒安全的,
                # 由 _poll_ui 在主執行緒發現變 None 後復原按鈕
                self.downloader = None
        self.dl_thread = threading.Thread(target=work, daemon=True)
        self.dl_thread.start()

    def cancel_download(self):
        if self.downloader is not None:
            self.dl_status_var.set("取消中…")
            self.downloader.cancel()

    def _reset_dl_buttons(self):
        self.dl_start_btn.config(state="normal")
        self.dl_cancel_btn.config(state="disabled")
        self.dl_status_var.set("待機")

    # ---------- 錄影分頁的行為 ----------

    def _rec_site_root(self) -> Path:
        url = self.rec_url_var.get().strip()
        root = Path(self.rec_root_var.get().strip() or DEFAULT_REC_ROOT)
        return root / site_slug(url) if url else root

    def _browse_rec_root(self):
        p = filedialog.askdirectory(title="選擇錄影存放資料夾")
        if p:
            self.rec_root_var.set(p)

    def _update_disk_hint(self):
        """顯示存放磁碟的可用空間,不足時變紅。

        720p 實測約 30 GB/天;磁碟滿的時候 ffmpeg 只會安靜地停住,
        所以這件事要在按下開始之前就看得到,不是事後才發現。
        """
        try:
            root = Path(self.rec_root_var.get().strip() or DEFAULT_REC_ROOT)
            probe = root
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            free = shutil.disk_usage(probe).free / 2**30
            need = 30.0 * max(1, int(self.rec_keep_var.get()))
            short = free < need
            self.rec_disk_lbl.config(
                text=f"可用 {free:.0f} GB;720p 約 30 GB/天,"
                     f"保留設定約需 {need:.0f} GB"
                     + ("  ⚠ 空間可能不足" if short else ""),
                foreground="#b00000" if short else "#444444")
        except (OSError, tk.TclError, ValueError):
            self.rec_disk_lbl.config(text="")

    def start_record(self):
        if self.recorder is not None:
            return
        url = self.rec_url_var.get().strip()
        if not url:
            messagebox.showerror("缺少網址", "請先填入直播網址。")
            return
        try:
            rec = StreamRecorder(
                url, root=self.rec_root_var.get().strip() or DEFAULT_REC_ROOT,
                segment_sec=int(self.rec_seg_var.get()),
                keep_days=int(self.rec_keep_var.get()),
                max_height=int(self.rec_height_var.get()),
                audio=bool(self.rec_audio_var.get()),
                log=self.rec_q.put)
        except Exception as e:
            messagebox.showerror("無法開始錄影", str(e))
            return
        self.recorder = rec
        self.rec_start_btn.config(state="disabled")
        self.rec_stop_btn.config(state="normal")
        self.rec_status_var.set("錄影中…")

        def work():
            try:
                rec.run()
            except Exception as e:                # 錄影失敗不該拖垮 GUI
                self.rec_q.put(f"[錯誤] {e}")
            finally:
                # 只放下旗標,不從這裡碰 UI:tkinter 不是執行緒安全的,
                # 連 root.after() 都不行(視窗剛好在關閉時會炸
                # "main thread is not in main loop")。由 _poll_ui 在主
                # 執行緒發現 recorder 變 None 之後自己把按鈕復原。
                self.recorder = None
        self.rec_thread = threading.Thread(target=work, daemon=True)
        self.rec_thread.start()

    def stop_record(self):
        if self.recorder is not None:
            self.rec_status_var.set("停止中…")
            self.recorder.stop()

    def _reset_rec_buttons(self):
        self.rec_start_btn.config(state="normal")
        self.rec_stop_btn.config(state="disabled")
        self.rec_status_var.set("已停止")

    def _open_rec_folder(self):
        site = self._rec_site_root()
        site.mkdir(parents=True, exist_ok=True)
        os.startfile(str(site))          # noqa: S606 (Windows 專用)

    def _preview_prune(self):
        """列出保留天數會刪掉哪些資料夾,但**不刪**。

        自動刪除資料夾是不可逆的,所以給一個先看再說的入口。
        """
        site = self._rec_site_root()
        try:
            keep = max(1, int(self.rec_keep_var.get()))
            doomed = prune_days(site, keep, dry_run=True)
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("檢查失敗", str(e))
            return
        if not doomed:
            messagebox.showinfo(
                "保留檢查", f"{site}\n\n沒有超出 {keep} 天的資料夾,"
                            f"目前不會刪除任何東西。")
        else:
            messagebox.showwarning(
                "保留檢查",
                f"{site}\n\n下次維護時會刪除這 {len(doomed)} 個資料夾:\n"
                + "\n".join(d.name for d in doomed))

    def _refresh_rec_status(self):
        """每兩秒更新一次今天的錄影量(只看檔案系統,不呼叫 ffmpeg)。"""
        if self.recorder is None:
            return
        day = self._rec_site_root() / day_name()
        try:
            files = sorted(day.glob("*.ts"))
            size = sum(p.stat().st_size for p in files) / 2**30
        except OSError:
            return
        self.rec_status_var.set(
            f"錄影中 {len(files)} 段 {size:.2f} GB"
            + (f" 重連{self.recorder.restarts}" if self.recorder.restarts
               else ""))

    # ---------- 方法選擇 ----------

    @staticmethod
    def _method_label(m) -> str:
        """選單顯示文字:缺權重的方法照樣列出,但標明缺什麼。

        不把不可用的方法藏起來:使用者要知道還有這條路存在、以及少了
        哪個檔案才能走。
        """
        return m.name if m.available else f"{m.name}(缺權重)"

    def _current_method(self):
        label = self.method_var.get()
        for m in methods_registry.METHODS:
            if self._method_label(m) == label:
                return m
        return methods_registry.default()

    def _on_method_change(self):
        """切換方法:更新說明文字,並開關「權重」欄。"""
        m = self._current_method()
        tag = "判定不含學習權重" if m.ai_free_decision else "含學習權重"
        self.method_desc.config(text=f"[{m.key} / {tag}] {m.desc}")
        state = "normal" if m.needs_appearance else "disabled"
        for w in (self.ckpt_entry, self.ckpt_btn):
            w.config(state=state)
        self.ckpt_label.config(
            text="權重" if m.needs_appearance else "權重(此方法不需)")

    # ---------- 控制 ----------

    def _browse_video(self):
        p = filedialog.askopenfilename(
            filetypes=[("影片", "*.mp4 *.avi *.mov *.mkv"), ("全部", "*.*")])
        if p:
            self.source_var.set(p)

    def _browse_ckpt(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch 權重", "*.pt")])
        if p:
            self.ckpt_var.set(p)

    def _apply_thresholds(self, _=None):
        # 主執行緒快取診斷開關,供影像執行緒安全讀取
        try:
            self._diag_enabled = bool(self.diag_var.get())
            self._clip_overlay = bool(self.clip_overlay_var.get())
        except (tk.TclError, AttributeError):
            pass  # 版面尚未建完(滑桿 trace 可能先觸發)
        if self.pipeline is not None:
            self.pipeline.alarm.trigger = self.trigger_var.get()
            # 解除線夾在 [0.01, 觸發線-0.01]:觸發線拉到 0 時
            # 解除線若為負,P≥0 永遠無法解除警報
            self.pipeline.alarm.release = max(
                0.01, min(self.release_var.get(),
                          self.trigger_var.get() - 0.01))
            try:  # Spinbox 打字中可能是空字串
                self.pipeline.min_events = max(1, int(self.min_events_var.get()))
            except (tk.TclError, ValueError):
                pass
            self.pipeline.move_gate_enabled = bool(self.move_gate_var.get())
            self.pipeline.wander_alert_enabled = bool(self.wander_var.get())
            try:
                dmin = float(self.dwell_min_var.get())
                dmax = float(self.dwell_max_var.get())
                if 0 < dmin < dmax:
                    self.pipeline.set_dwell_window(dmin, dmax)
            except (tk.TclError, ValueError):
                pass

    def start(self):
        if self.running:
            return
        method = self._current_method()
        if method.missing():
            messagebox.showerror(
                "缺少權重",
                f"方法「{method.name}」需要以下檔案:\n"
                + "\n".join(method.missing())
                + "\n\n請先訓練或搬入權重,或改選其他方法。")
            return
        if method.needs_appearance and not self.ckpt_var.get().strip():
            messagebox.showerror(
                "缺少權重",
                f"方法「{method.name}」需要外觀網路權重,請在「權重」欄指定。")
            return
        self.start_btn.config(state="disabled")
        self.status_var.set("載入模型中…")
        threading.Thread(target=self._start_worker, args=(method,),
                         daemon=True).start()

    def _start_worker(self, method):
        """在背景執行緒載入 pipeline(避免凍住 UI),然後開始處理。"""
        try:
            from inference.pipeline import SmokingDetectionPipeline
            infer_cfg = load_config(self.infer_config)
            ckpt = self.ckpt_var.get().strip()
            # 要不要載外觀網路由方法決定,不再由「權重欄有沒有填」決定 ——
            # 否則選了純規則卻忘了清空權重欄,跑的其實是融合版
            use_model = method.needs_appearance
            model_cfg = load_config("configs/model.yaml") if use_model else None

            pipeline = SmokingDetectionPipeline(
                infer_cfg, model_cfg,
                ckpt_path=(ckpt or None) if use_model else None,
                use_model=use_model, method=method)
            self.alarm_q.put(f"[方法] {method.key} — {method.name}")
            # 警報 callback 導向 GUI 記錄(同時沿用截圖行為)
            from inference.alarm import default_alarm_callback
            snap_dir = infer_cfg["alarm"]["snapshot_dir"]

            def gui_callback(tid, P, t, frame):
                default_alarm_callback(tid, P, t, frame, snapshot_dir=snap_dir)
                # 開始錄警報片段:帶入觸發前的緩衝影格,續錄 POST 秒
                rec = {"tid": tid, "trigger_t": t,
                       "end_t": t + self.CLIP_POST_SEC,
                       "frames": list(self.clip_buffer),
                       "pose": list(self.pose_buffer), "path": None}
                self._active_recs.append(rec)
                self.alarm_q.put({
                    "text": time.strftime("%H:%M:%S") +
                            f"  ⚠ track {tid} 觸發抽菸警報(P={P:.2f})"
                            f" — 雙擊回放片段",
                    "rec": rec})
            pipeline.alarm.callback = gui_callback
            # 事件結算通知:每次手放下顯示停留秒數與是否計入(可觀察校準)
            # 記錄原則:預設只顯示「警報觸發」;
            # 事件計次與未達門檻等過程訊息全部歸入診斷開關
            # 影像執行緒內執行:只讀主執行緒快取的 _diag_enabled,
            # 不可直接碰 tk 變數
            def log_event(tid, dwell, counted, reason):
                if self._diag_enabled:
                    mark = "✔" if counted else "✘"
                    self.alarm_q.put(
                        time.strftime("%H:%M:%S") +
                        f"  track {tid} 停留 {dwell:.1f} 秒 {mark} {reason}")
            pipeline.on_event = log_event

            # 徘徊通報:與抽菸警報同樣列進記錄(不受診斷開關影響),
            # 但不錄片段、不改 P_t —— 它是另一種事件,不是抽菸的證據
            def log_wander(tid, stay, path):
                self.alarm_q.put({
                    "text": time.strftime("%H:%M:%S") +
                            f"  ◆ track {tid} 徘徊(在場 {stay:.0f} 秒,"
                            f"移動 {path:.1f} 倍身高)",
                    "rec": None})
            pipeline.on_presence = log_wander

            # 第二階段複核結果:雙擊該列可看完整的基元時間軸與節律統計。
            # 這是兩層架構最大的賣點 —— 複查誤報時看得到系統憑什麼判
            def log_verify(tid, res):
                icon = {"confirmed": "✔", "review": "△",
                        "abstain": "?"}.get(res.status, "·")
                text = (time.strftime("%H:%M:%S")
                        + f"  {icon} track {tid} 二次複核:{res.status_name}"
                        + f" — {res.reason} — 雙擊看依據")
                self.alarm_q.put({"text": text, "rec": None,
                                  "detail": res.detail})
            pipeline.on_verify = log_verify
            pipeline.on_log = lambda msg: (
                self.alarm_q.put(time.strftime("%H:%M:%S") + "  " + msg)
                if self._diag_enabled else None)
            self.pipeline = pipeline
            # 套用目前滑桿值(使用者的調整優先,不被設定檔覆蓋)
            self._apply_thresholds()
        except Exception as e:  # 載入失敗回報到 UI
            self.alarm_q.put(f"[錯誤] 模型載入失敗:{e}")
            self.root.after(0, lambda: (
                self.start_btn.config(state="normal"),
                self.status_var.set("載入失敗")))
            return

        self.running = True
        self.root.after(0, lambda: (
            self.stop_btn.config(state="normal"),
            self.status_var.set("執行中")))
        self._video_loop()

    def _video_loop(self):
        src = self.source_var.get().strip()
        from inference.stream import VideoSource
        from inference.pipeline import draw_overlay
        try:
            vs = VideoSource(
                src, sample_fps=self.pipeline.cfg["sampling"]["target_fps"],
                **self.pipeline.cfg.get("stream", {}))
        except Exception as e:
            self.alarm_q.put(f"[錯誤] 無法開啟來源:{e}")
            self.running = False
            self.root.after(0, self._reset_buttons)
            return
        self.alarm_q.put(f"[資訊] 來源={vs.kind} fps={vs.fps:.1f}")

        target = self.pipeline.cfg["sampling"]["target_fps"]
        step = max(1, round(vs.fps / target))   # 檔案:幀計數取樣
        min_interval = 1.0 / target             # live:時間取樣
        idx, results = 0, {}
        last_proc = float("-inf")
        t_prev = time.time()
        # 實際推理速率:卡的時候要看得出是「來源餵不夠」還是「機器跑不動」,
        # 不然只能猜。lag<1 = 正被直播拉開,佇列見底 = 來源慢
        fps_t0, fps_n = time.time(), 0
        while self.running:
            frame, ts = vs.read(timeout=2.0)
            if frame is None:
                if vs.is_live:
                    self.alarm_q.put("[資訊] 等待串流影格(重連中)…")
                    continue
                self.alarm_q.put("[資訊] 影片播放結束")
                break
            if getattr(vs, "pre_sampled", False):
                do_step = True      # 來源已抽稀到 target_fps,不再濾一次
            elif vs.is_live:
                do_step = ts - last_proc >= min_interval
            else:
                do_step = idx % step == 0
            if do_step:
                results = self.pipeline.step(frame, ts)
                last_proc = ts
                fps_n += 1
                if time.time() - fps_t0 >= 1.0:
                    rate = fps_n / (time.time() - fps_t0)
                    extra = ""
                    if getattr(vs, "queued", False):
                        extra = (f" 佇列{len(vs._queue)}"
                                 f" 落後比{vs.lag_ratio:.2f}")
                    # tk 變數只能在主執行緒改
                    self.root.after(
                        0, lambda r=rate, e=extra:
                        self.status_var.set(f"執行中 {r:.1f} fps{e}"))
                    fps_t0, fps_n = time.time(), 0
            vis = draw_overlay(frame, results)
            if do_step:
                # 存檔用哪一份由開關決定:關 = 乾淨原始影格(可訓練外觀模型)
                self._record_clip_frame(
                    vis if self._clip_overlay else frame, ts, results)
            try:
                self.frame_q.put_nowait((vis, dict(results)))
            except queue.Full:
                pass
            idx += 1
            if not vs.is_live:  # 檔案來源:依 fps 節流,模擬即時
                dt = 1.0 / vs.fps - (time.time() - t_prev)
                if dt > 0:
                    time.sleep(dt)
            t_prev = time.time()
        vs.release()
        self.pipeline.close()   # 收掉複核執行緒池(每次「開始」都會新建一條管線)
        self.running = False
        self.root.after(0, self._reset_buttons)

    def stop(self):
        self.running = False
        self.status_var.set("停止中…")

    def _reset_buttons(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("待機")

    def on_close(self):
        self.running = False
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        # 錄影一定要等它收乾淨:ffmpeg 是獨立行程,直接關視窗會留下孤兒
        # 行程繼續寫檔案(而且沒人管得到它)。等一下下比留孤兒好。
        if self.recorder is not None:
            self.recorder.stop()
            if self.rec_thread is not None:
                self.rec_thread.join(timeout=8.0)
        if self.downloader is not None:
            self.downloader.cancel()
            if self.dl_thread is not None:
                self.dl_thread.join(timeout=5.0)
        self.root.after(150, self.root.destroy)

    # ---------- UI 更新 ----------

    # 畫面重繪上限(秒)。重繪跑在 Tk 主執行緒且持有 GIL,實測
    # 800×600 要 5.8 ms、放大到 1600×900 要 21.5 ms;不設上限的話
    # 光是畫圖就能吃掉大半個主執行緒,連帶餓死收幀執行緒 —— 串流會
    # 因此掉幀。推理不受影響(每一幀都照跑),這裡限的只是「顯示」。
    DRAW_INTERVAL = 1.0 / 15

    def _poll_ui(self):
        try:
            vis, results = self.frame_q.get_nowait()
            now = time.time()
            if now - self._last_draw >= self.DRAW_INTERVAL:
                self._draw_frame(vis)
                self._last_draw = now
            self._update_tracks(results)     # 文字面板很便宜,照常更新
        except queue.Empty:
            pass
        changed = False
        try:
            while True:
                item = self.alarm_q.get_nowait()
                if isinstance(item, str):
                    item = {"text": item, "rec": None}
                self.log_entries.insert(0, item)  # 最新在前
                changed = True
        except queue.Empty:
            pass
        if changed:
            self._render_log()
        # 錄影分頁:訊息照單全收,狀態每兩秒才刷一次(要掃資料夾,別太勤)
        try:
            while True:
                self.rec_log.insert(0, self.rec_q.get_nowait())
                if self.rec_log.size() > 300:
                    self.rec_log.delete(300, tk.END)
        except queue.Empty:
            pass
        # 錄影執行緒結束後由主執行緒復原按鈕(見 start_record 的說明)
        if self.recorder is None and str(self.rec_stop_btn["state"]) != \
                "disabled":
            self._reset_rec_buttons()
        elif time.time() - self._rec_status_t >= 2.0:
            self._rec_status_t = time.time()
            self._refresh_rec_status()

        # 下載分頁:訊息全收,進度只取最新一筆(短時間內會湧入上百筆,
        # 每一筆都寫進度條是白做工,畫面也只看得到最後一筆)
        msg = None
        try:
            while True:
                msg = self.dl_q.get_nowait()
        except queue.Empty:
            pass
        if msg is not None:
            # 記錄改成縮圖清單之後,訊息(尤其錯誤)改走狀態列,
            # 不然使用者完全不知道發生什麼事
            self.dl_status_var.set(msg)
            if msg.startswith("[完成]") or msg.startswith("[錯誤]"):
                self.refresh_dl_list()
        try:
            while True:
                self.dl_list.apply_meta(self.dl_meta_q.get_nowait())
        except queue.Empty:
            pass
        latest = None
        try:
            while True:
                latest = self.dl_prog_q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self.dl_bar["value"] = latest.get("frac", 0.0)
            self.dl_status_var.set(latest.get("text", ""))
        # 下載執行緒結束後由主執行緒復原按鈕(同 start_download 的說明)
        if self.downloader is None and str(self.dl_cancel_btn["state"]) != \
                "disabled":
            self._reset_dl_buttons()

        # 記住排程 id:關視窗時要取消,否則 destroy 之後那個 callback 還會
        # 觸發一次,在主控台留下 invalid command name "..._poll_ui"
        self._poll_id = self.root.after(30, self._poll_ui)

    # ---------- 警報片段錄製 ----------

    def _record_clip_frame(self, img, ts, results):
        """收一張取樣影格與該幀節點進滾動緩衝,並推進進行中的片段錄製。

        img 已由呼叫端依「錄影疊加」開關選好:乾淨原始影格或疊加影格
        (縮小到寬 ≤960 節省記憶體)。results 是該幀所有 track 的推理
        結果,節點從這裡取。
        """
        # 開關切換時清掉緩衝,免得同一段片子前半乾淨、後半有疊加
        # (在影像執行緒內處理,不與主執行緒搶 deque;
        #  節點緩衝一起清,兩者長度才對得起來)
        if self._clip_overlay != self._clip_overlay_applied:
            self.clip_buffer.clear()
            self.pose_buffer.clear()
            self._clip_overlay_applied = self._clip_overlay
        h, w = img.shape[:2]
        if w > 960:
            s = 960 / w
            img = cv2.resize(img, (960, int(h * s)))
        else:
            img = img.copy()   # 原始影格可能被擷取端重複使用,務必複製
        # 節點以「原始影像座標」保存,不隨上面的縮放改變 ——
        # stage2 的正規化本來就會除掉尺度,存原始座標最不失真
        snap = {
            tid: (None if r.get("kpts") is None
                  else np.asarray(r["kpts"], np.float32).copy(),
                  np.asarray(r["bbox"], np.float32).copy())
            for tid, r in results.items()
        }
        self.clip_buffer.append(img)
        self.pose_buffer.append((ts, snap))
        for rec in self._active_recs[:]:
            rec["frames"].append(img)
            rec["pose"].append((ts, snap))
            if ts >= rec["end_t"]:
                self._active_recs.remove(rec)
                threading.Thread(target=self._write_clip, args=(rec,),
                                 daemon=True).start()

    def _write_clip(self, rec):
        """把警報片段寫成 mp4(背景執行緒)。"""
        frames = rec["frames"]
        if not frames:
            return
        clip_dir = Path("alarms/clips")
        clip_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = clip_dir / f"alarm_track{rec['tid']}_{stamp}.mp4"
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (w, h))
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            vw.write(f)
        vw.release()
        rec["path"] = str(path)
        self._write_pose(rec, path)

    def _write_pose(self, rec, clip_path):
        """把警報對象的節點序列存成 annotations/pose/{片段檔名}.npz。

        輸出格式刻意與 stage2/extract_pose.py 完全一致(kpts / bbox /
        valid / fps / clip),stage2 的資料集與訓練腳本都不必改。

        存在的理由:extract_pose.py 是靠畫面上烙印的紅色警報框回頭定位
        「被警報的是哪個人」。一旦改錄乾淨影像(訓練外觀模型的前提),
        那條路就斷了 —— 節點只能在錄影的當下直接落地,而且這樣拿到的
        是追蹤器原本就認定的對象,比事後用顏色遮罩反推更準。
        """
        seq = rec.get("pose") or []
        if not seq:
            return
        tid = rec["tid"]
        T = len(seq)
        kpts = np.zeros((T, 17, 3), np.float32)
        bbox = np.zeros((T, 4), np.float32)
        valid = np.zeros(T, bool)
        for t, (_ts, snap) in enumerate(seq):
            item = snap.get(tid)
            if item is None:
                continue            # 該幀這個 track 沒被偵測到
            k, b = item
            bbox[t] = b
            if k is not None:
                kpts[t] = k
                valid[t] = True
        out_dir = Path("annotations/pose")
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (Path(clip_path).stem + ".npz")
        np.savez_compressed(
            dst, kpts=kpts, bbox=bbox, valid=valid, fps=10.0,
            clip=str(Path(clip_path).as_posix()))
        self.alarm_q.put(
            time.strftime("%H:%M:%S") +
            f"  節點已存 {dst.name}({int(valid.sum())}/{T} 幀有骨架)")

    # ---------- 條列式記錄(分頁)與回放 ----------

    def _total_pages(self) -> int:
        return max(1, -(-len(self.log_entries) // self.PAGE_SIZE))

    def _goto_page(self, page: int):
        self.page = min(max(1, int(page)), self._total_pages())
        self._render_log()

    def _render_log(self):
        total = self._total_pages()
        self.page = min(self.page, total)
        self.page_var.set(self.page)
        self.page_spin.configure(to=total)
        self.page_total_lbl.config(text=f"/ {total} 頁"
                                        f"(共 {len(self.log_entries)} 筆)")
        self.log.delete(0, tk.END)
        start = (self.page - 1) * self.PAGE_SIZE
        for entry in self.log_entries[start:start + self.PAGE_SIZE]:
            self.log.insert(tk.END, entry["text"])

    def _on_log_dclick(self, _event):
        sel = self.log.curselection()
        if not sel:
            return
        idx = (self.page - 1) * self.PAGE_SIZE + sel[0]
        if idx >= len(self.log_entries):
            return
        entry = self.log_entries[idx]
        rec = entry.get("rec")
        if rec is None:
            if entry.get("detail"):     # 複核結果:顯示判定依據
                DetailWindow(self.root, entry["text"], entry["detail"])
            return                      # 其餘非警報項目
        if rec.get("path") is None:
            messagebox.showinfo("片段錄製中",
                                "警報片段還在錄製(觸發後續錄 4 秒),"
                                "請稍候再點。")
            return
        open_video(self.root, rec["path"],
                   f"track {rec['tid']} 抽菸警報片段",
                   method=self._current_method(),
                   infer_config=self.infer_config)

    def _draw_frame(self, bgr):
        # 依影像區「目前實際尺寸」縮放:拉大視窗畫面就跟著放大
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 100 or ch < 100:          # 尚未完成佈局時的 fallback
            cw, ch = VIDEO_W, VIDEO_H
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        img = cv2.resize(bgr, (max(1, int(w * scale)),
                               max(1, int(h * scale))))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._set_canvas_image(ImageTk.PhotoImage(Image.fromarray(img)))

    def _update_tracks(self, results):
        """把當前 track 分流到三欄。分類規則在 triage()(純函式,有測試)。"""
        now = time.time()
        present, watching, smoking = triage(results)
        levels = {tid: r.get("P", 0.0) for tid, r in results.items()}
        self.panel_present.update(present, now, levels)
        self.panel_watch.update(watching, now)
        self.panel_smoke.update(smoking, now)


class DetailWindow(tk.Toplevel):
    """判定依據視窗:等寬字顯示基元時間軸、節律統計與各類別分數。"""

    def __init__(self, master, title: str, text: str):
        super().__init__(master)
        self.title(title.strip())
        box = tk.Text(self, wrap="none", font=("Consolas", 10),
                      width=64, height=22)
        sb = ttk.Scrollbar(self, orient="vertical", command=box.yview)
        box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)
        box.configure(state="disabled")   # 只讀,但仍可選取複製
        self.attributes("-topmost", True)
        self.after(500, lambda: self.attributes("-topmost", False))


VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts")
THUMB_W, THUMB_H = 160, 90


def list_videos(folder) -> list:
    """資料夾裡的影片檔,最新的排前面。"""
    d = Path(folder)
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir()
             if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def video_meta(path: Path) -> dict:
    """縮圖 + 長度 + 尺寸。縮圖取 10% 位置的那一幀。

    不取第 0 幀:很多影片開頭是黑畫面或版權卡,整排縮圖會全黑,
    等於沒有縮圖。
    """
    meta = {"path": path, "thumb": None, "seconds": 0.0, "size": 0}
    try:
        meta["size"] = path.stat().st_size
    except OSError:
        pass
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return meta
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and n > 0:
            meta["seconds"] = n / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.1))
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            s = min(THUMB_W / w, THUMB_H / h)
            small = cv2.resize(frame, (max(1, int(w * s)),
                                       max(1, int(h * s))))
            meta["thumb"] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    except Exception:
        pass                       # 壞檔就只是沒有縮圖,不該讓清單掛掉
    finally:
        cap.release()
    return meta


class VideoList(ttk.Frame):
    """可捲動的影片縮圖清單,雙擊播放。

    縮圖解碼放在背景執行緒(每個檔案要開一次 VideoCapture,檔案一多會
    卡住介面),但 PhotoImage 一定要在主執行緒建立——tkinter 的物件不能
    跨執行緒建,所以背景只回傳 numpy 陣列,由 _poll_ui 收進來再轉。
    """

    def __init__(self, parent, on_open, height: int = 300):
        super().__init__(parent)
        self.on_open = on_open
        canvas = tk.Canvas(self, height=height, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._win = canvas.create_window((0, 0), window=self.inner,
                                         anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(self._win, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas = canvas
        self._rows = {}            # path -> (frame, thumb_label)
        self._photos = {}          # path -> PhotoImage(防 GC)
        self.empty = ttk.Label(self.inner, text="(這個資料夾還沒有影片)",
                               foreground="#888888")
        self.empty.pack(anchor="w", padx=8, pady=8)

    def _wheel(self, event):
        try:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
        except tk.TclError:
            pass

    def set_files(self, files: list) -> list:
        """重建清單(縮圖先留白),回傳還需要解碼縮圖的檔案。"""
        for row, _thumb, _meta in self._rows.values():
            row.destroy()
        self._rows.clear()
        self.empty.pack_forget()
        if not files:
            self.empty.pack(anchor="w", padx=8, pady=8)
            return []
        for p in files:
            row = ttk.Frame(self.inner, padding=3)
            row.pack(fill="x")
            thumb = tk.Label(row, width=THUMB_W, height=THUMB_H,
                             background="#333333", cursor="hand2")
            thumb.pack(side="left")
            info = ttk.Frame(row)
            info.pack(side="left", fill="x", expand=True, padx=8)
            ttk.Label(info, text=p.name, anchor="w",
                      wraplength=560, justify="left").pack(anchor="w")
            meta_lbl = ttk.Label(info, text="讀取中…", foreground="#666666")
            meta_lbl.pack(anchor="w")
            ttk.Button(row, text="▶ 播放",
                       command=lambda q=p: self.on_open(q)).pack(side="right")
            for w in (row, thumb, info):
                w.bind("<Double-Button-1>", lambda _e, q=p: self.on_open(q))
            self._rows[p] = (row, thumb, meta_lbl)
        return list(files)

    def apply_meta(self, meta: dict) -> None:
        """把背景解碼好的縮圖與資訊放上去(主執行緒呼叫)。"""
        item = self._rows.get(meta["path"])
        if item is None:
            return
        _row, thumb, meta_lbl = item
        if meta.get("thumb") is not None:
            photo = ImageTk.PhotoImage(Image.fromarray(meta["thumb"]))
            self._photos[meta["path"]] = photo      # 保留參照,否則被 GC
            thumb.configure(image=photo, width=THUMB_W, height=THUMB_H)
        secs, size = meta.get("seconds", 0), meta.get("size", 0)
        parts = []
        if secs:
            parts.append(f"{int(secs) // 60}:{int(secs) % 60:02d}")
        if size:
            parts.append(f"{size / 2**20:.1f} MB")
        meta_lbl.config(text="  ".join(parts) or "(無法讀取)")


def _enable_dpi_awareness():
    """宣告 DPI aware:高分螢幕上畫面不模糊,座標與實體像素一致。"""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main():
    _enable_dpi_awareness()
    parser = argparse.ArgumentParser(description="抽菸偵測 Demo GUI")
    parser.add_argument("--autotest", default=None,
                        help="自動驗證:載入指定影片跑數秒後截圖退出")
    parser.add_argument("--infer-config", default="configs/inference_hmdb.yaml"
                        if Path("configs/inference_hmdb.yaml").exists()
                        else "configs/inference.yaml",
                        help="推理設定(HMDB 權重用 inference_hmdb.yaml)")
    parser.add_argument("--method", default=None,
                        choices=methods_registry.keys(),
                        help="預選判定方法(見 inference/methods.py);"
                             "省略則用預設,啟動後仍可在選單改")
    args = parser.parse_args()

    root = tk.Tk()
    app = DemoGUI(root, infer_config=args.infer_config)
    if args.method:
        app.method_var.set(
            app._method_label(methods_registry.get(args.method)))
        app._on_method_change()

    if args.autotest:
        app.source_var.set(args.autotest)
        root.after(500, app.start)

        def snap_and_quit():
            from PIL import ImageGrab
            root.update_idletasks()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            box = (x, y, x + root.winfo_width(), y + root.winfo_height())
            ImageGrab.grab(box).save("smoke_run/gui_autotest.png")
            print("[autotest] 截圖已存 smoke_run/gui_autotest.png")
            app.on_close()
        root.after(12000, snap_and_quit)

    root.mainloop()


if __name__ == "__main__":
    main()
