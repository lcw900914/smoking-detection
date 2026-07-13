"""抽菸行為偵測 Demo GUI(Tkinter,無額外相依)。

功能:
- 來源選擇:攝影機編號或影片檔(瀏覽)
- 模型權重選擇;留空則僅跑偵測+追蹤
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
from PIL import Image, ImageTk  # noqa: E402

from utils import load_config  # noqa: E402

VIDEO_W, VIDEO_H = 800, 600
DEFAULT_CKPT = "checkpoints/hmdb_e2e_best.pt"
_STAGE_NAMES = {0: "S1 舉手", 1: "S2 嘴部", 2: "S3 放下", 3: "背景"}


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
            esc = cfg_all.get("escalation", {})
            self._init_dwell_min = float(esc.get("min_dwell", 2.0))
            self._init_dwell_max = float(esc.get("max_dwell", 5.0))
        except Exception:
            self._init_trigger, self._init_release = 0.75, 0.4
            self._init_min_events = 3
            self._init_move_gate = True
            self._init_dwell_min, self._init_dwell_max = 2.0, 5.0
        root.title("抽菸行為偵測 Demo — channel-as-temporal-buffer")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.pipeline = None
        self.worker: threading.Thread = None
        self.running = False
        self.frame_q: "queue.Queue" = queue.Queue(maxsize=2)
        self.alarm_q: "queue.Queue" = queue.Queue()

        # 警報片段錄製:滾動保留最近 ~10 秒取樣影格(縮小節省記憶體),
        # 警報觸發時連同後續 4 秒寫成 mp4,供記錄點擊回放
        self.CLIP_PRE_FRAMES = 100     # 約 10 秒 @10fps
        self.CLIP_POST_SEC = 4.0
        self.clip_buffer: deque = deque(maxlen=self.CLIP_PRE_FRAMES)
        self._active_recs = []         # 錄製中的警報片段

        # 條列式記錄(分頁)
        self.log_entries = []          # 最新在前;{'text', 'rec'}
        self.PAGE_SIZE = 8
        self.page = 1

        self._build_layout()
        self._poll_ui()

        # 啟動時帶到最前(短暫 topmost 再釋放,避免永遠壓住其他視窗)
        root.lift()
        root.attributes("-topmost", True)
        root.after(1500, lambda: root.attributes("-topmost", False))

    # ---------- 版面 ----------

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=6)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
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
                              width=680, height=300)
        side.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        # 注意:內部子元件用 pack 排版,尺寸傳播要用 pack_propagate 關;
        # grid_propagate 只擋 grid 子元件,關錯了面板仍會隨文字伸縮
        side.pack_propagate(False)
        side.grid_propagate(False)
        self.track_panel = ttk.Frame(side)
        self.track_panel.pack(fill="both", expand=True)

        # 固定欄位制:預先建好 N 列,track 進出只改文字、不增刪列,
        # 版面完全靜止(track 閃爍時清單才不會跳動)
        self.MAX_SLOTS = 8
        self.slots = []              # [(lbl, bar), ...]
        for _ in range(self.MAX_SLOTS):
            row = ttk.Frame(self.track_panel)
            row.pack(fill="x", pady=2)
            bar = ttk.Progressbar(row, maximum=1.0, length=90)
            bar.pack(side="right")
            lbl = ttk.Label(row, text="—", anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            self.slots.append((lbl, bar))
        self.tid_slot = {}           # track_id → slot index
        self.tid_last_seen = {}      # track_id → 最後更新時間(寬限用)

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

        # 診斷訊息開關(預設關):校準時才顯示未計入/未達門檻的原因
        # 注意:工作執行緒不可直接讀 tk 變數(tkinter 非執行緒安全),
        # 由 _apply_thresholds 在主執行緒快取成 _diag_enabled
        self.diag_var = tk.BooleanVar(value=False)
        self._diag_enabled = False
        ttk.Checkbutton(thr, text="顯示診斷訊息(校準用)",
                        variable=self.diag_var,
                        command=self._apply_thresholds).pack(
            anchor="w", pady=2)

        # 下:控制列
        ctrl = ttk.Frame(main, padding=(0, 6))
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(ctrl, text="來源").pack(side="left")
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(ctrl, textvariable=self.source_var, width=32).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="瀏覽…", command=self._browse_video).pack(
            side="left")
        ttk.Label(ctrl, text="權重").pack(side="left", padx=(12, 0))
        self.ckpt_var = tk.StringVar(
            value=DEFAULT_CKPT if Path(DEFAULT_CKPT).exists() else "")
        ttk.Entry(ctrl, textvariable=self.ckpt_var, width=36).pack(
            side="left", padx=4)
        ttk.Button(ctrl, text="…", width=3, command=self._browse_ckpt).pack(
            side="left")
        self.start_btn = ttk.Button(ctrl, text="▶ 開始", command=self.start)
        self.start_btn.pack(side="left", padx=(12, 2))
        self.stop_btn = ttk.Button(ctrl, text="■ 停止", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left")
        self.status_var = tk.StringVar(value="待機")
        ttk.Label(ctrl, textvariable=self.status_var).pack(side="left",
                                                           padx=12)

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
        except tk.TclError:
            pass
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
        self.start_btn.config(state="disabled")
        self.status_var.set("載入模型中…")
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self):
        """在背景執行緒載入 pipeline(避免凍住 UI),然後開始處理。"""
        try:
            from inference.pipeline import SmokingDetectionPipeline
            infer_cfg = load_config(self.infer_config)
            ckpt = self.ckpt_var.get().strip()
            use_model = bool(ckpt)
            model_cfg = load_config("configs/model.yaml") if use_model else None

            pipeline = SmokingDetectionPipeline(
                infer_cfg, model_cfg,
                ckpt_path=ckpt or None, use_model=use_model)
            # 警報 callback 導向 GUI 記錄(同時沿用截圖行為)
            from inference.alarm import default_alarm_callback
            snap_dir = infer_cfg["alarm"]["snapshot_dir"]

            def gui_callback(tid, P, t, frame):
                default_alarm_callback(tid, P, t, frame, snapshot_dir=snap_dir)
                # 開始錄警報片段:帶入觸發前的緩衝影格,續錄 POST 秒
                rec = {"tid": tid, "trigger_t": t,
                       "end_t": t + self.CLIP_POST_SEC,
                       "frames": list(self.clip_buffer), "path": None}
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
            vs = VideoSource(src, **self.pipeline.cfg.get("stream", {}))
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
        while self.running:
            frame, ts = vs.read(timeout=2.0)
            if frame is None:
                if vs.is_live:
                    self.alarm_q.put("[資訊] 等待串流影格(重連中)…")
                    continue
                self.alarm_q.put("[資訊] 影片播放結束")
                break
            if vs.is_live:
                do_step = ts - last_proc >= min_interval
            else:
                do_step = idx % step == 0
            if do_step:
                results = self.pipeline.step(frame, ts)
                last_proc = ts
            vis = draw_overlay(frame, results)
            if do_step:
                self._record_clip_frame(vis, ts)
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
        self.root.after(150, self.root.destroy)

    # ---------- UI 更新 ----------

    def _poll_ui(self):
        try:
            vis, results = self.frame_q.get_nowait()
            self._draw_frame(vis)
            self._update_tracks(results)
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
        self.root.after(30, self._poll_ui)

    # ---------- 警報片段錄製 ----------

    def _record_clip_frame(self, vis, ts):
        """收一張取樣影格進滾動緩衝(縮小到寬 ≤960 節省記憶體),
        並推進進行中的警報片段錄製。"""
        h, w = vis.shape[:2]
        if w > 960:
            s = 960 / w
            vis = cv2.resize(vis, (960, int(h * s)))
        self.clip_buffer.append(vis)
        for rec in self._active_recs[:]:
            rec["frames"].append(vis)
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
        rec = self.log_entries[idx].get("rec")
        if rec is None:
            return                      # 非警報項目
        if rec.get("path") is None:
            messagebox.showinfo("片段錄製中",
                                "警報片段還在錄製(觸發後續錄 4 秒),"
                                "請稍候再點。")
            return
        ClipPlayer(self.root, rec["path"],
                   f"track {rec['tid']} 抽菸警報片段")

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

    GRACE_SEC = 2.0   # track 短暫消失的寬限:格位保留,避免清單跳動

    def _update_tracks(self, results):
        now = time.time()

        # 1) 更新在畫面上的 track(必要時分配空格位)
        for tid, r in sorted(results.items()):
            if tid not in self.tid_slot:
                used = set(self.tid_slot.values())
                free = [i for i in range(self.MAX_SLOTS) if i not in used]
                if not free:
                    continue  # 格位滿(>8 人)暫不顯示,不擠掉現有列
                self.tid_slot[tid] = free[0]
            self.tid_last_seen[tid] = now

            lbl, bar = self.slots[self.tid_slot[tid]]
            level = r.get("level", 0.0)
            lv = ("高" if level >= 0.8 else
                  "中" if level >= 0.5 else
                  "低" if level >= 0.2 else "")
            text = f"ID{tid} {_STAGE_NAMES.get(r['stage'], '?')}"
            if r.get("orientation") == "back":
                text += " 背向"
            if r.get("events"):
                text += f" {r['events']}次"
            if lv:
                text += f" 警戒{lv}"
            if r.get("unverified"):
                text += " 無法確認"
            if r.get("loiter"):
                text += " 逗留"
            if r.get("moving"):
                text += " 移動中"
            if r.get("phone"):
                text += " 講電話"
            if r["alarm"]:
                text += " ⚠"
            lbl.config(text=text)
            bar["value"] = r["P"]

        # 2) 消失超過寬限的 track 才釋放格位(清成「—」,列不刪除)
        for tid in [t for t, ts in self.tid_last_seen.items()
                    if now - ts > self.GRACE_SEC]:
            slot = self.tid_slot.pop(tid, None)
            self.tid_last_seen.pop(tid, None)
            if slot is not None:
                lbl, bar = self.slots[slot]
                lbl.config(text="—")
                bar["value"] = 0.0


class ClipPlayer(tk.Toplevel):
    """警報片段回放視窗:循環播放 mp4,畫面隨視窗縮放。"""

    def __init__(self, master, path: str, title: str):
        super().__init__(master)
        self.title(title)
        self.cap = cv2.VideoCapture(path)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        self.delay = max(30, int(1000 / fps))

        # 初始尺寸:片源解析度,上限 800 寬(之後可自由拉大縮小)
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        s = min(1.0, 800 / max(1, src_w))
        self._init_w, self._init_h = int(src_w * s), int(src_h * s)

        # 用 Canvas:要求尺寸固定為初始值,實際顯示依視窗現況縮放
        # (Label 會把放大後的圖變成新的最小尺寸,無法縮回)
        self.canvas = tk.Canvas(self, background="#111111",
                                width=self._init_w, height=self._init_h,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._item = self.canvas.create_image(
            self._init_w // 2, self._init_h // 2, anchor="center")

        self.protocol("WM_DELETE_WINDOW", self._close)
        self._alive = True
        self.attributes("-topmost", True)
        self.after(500, lambda: self.attributes("-topmost", False))
        self._tick()

    def _tick(self):
        if not self._alive:
            return
        ok, frame = self.cap.read()
        if not ok:  # 播完 → 從頭循環
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                self._close()
                return
        # 依視窗目前實際尺寸等比縮放(拉大視窗畫面跟著放大)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = self._init_w, self._init_h
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)
        frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.coords(self._item, cw // 2, ch // 2)
        self.canvas.itemconfigure(self._item, image=photo)
        self.canvas.image = photo
        self.after(self.delay, self._tick)

    def _close(self):
        self._alive = False
        self.cap.release()
        self.destroy()


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
    args = parser.parse_args()

    root = tk.Tk()
    app = DemoGUI(root, infer_config=args.infer_config)

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
