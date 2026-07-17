"""警報片段標記工具:逐段播放,標記是否為抽菸(訓練資料用)。

用法(專案根目錄):
    python scripts/label_tool.py                     # 標 alarms/clips
    python scripts/label_tool.py --dir <其他資料夾>

操作(按鈕或快捷鍵):
    1 = 抽菸          2 = 喝水/飲食      3 = 講電話
    4 = 桌面工作/摸臉  5 = 其他非抽菸     0 = 不確定(之後再看)
    ←/→ = 上一段/下一段    F = 播放速度 1x/2x    Q = 離開

行為:
- 標記即寫入 annotations/clip_labels.json(逐筆存檔,中斷不掉資料)
- 啟動時自動跳到第一段未標記的片段;已標記的片段會顯示目前標籤
- 標記後自動跳下一段未標記
- 標籤語意:label=smoking 為正樣本,其餘皆為負樣本
  (細分類別供第二階段多分類訓練,攤開負類別比二元更有辨別力)
"""
import argparse
import glob
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

LABELS_PATH = Path("annotations/clip_labels.json")

# (鍵, 標籤代碼, 顯示名, 是否為正樣本)
CATEGORIES = [
    ("1", "smoking",    "抽菸",         True),
    ("2", "drinking",   "喝水/飲食",    False),
    ("3", "phone",      "講電話",       False),
    ("4", "desk_work",  "桌面工作/摸臉", False),
    ("5", "other_neg",  "其他非抽菸",   False),
    ("6", "bad_pose",   "骨架錯誤(排除)", False),  # 骨架畫錯人/錯位:不進訓練
    ("0", "unsure",     "不確定",       False),
]
NAME_OF = {code: name for _, code, name, _ in CATEGORIES}

VIDEO_W, VIDEO_H = 960, 540


class LabelStore:
    """標籤存取:逐筆寫檔,鍵為相對路徑(跨機器可攜)。"""

    def __init__(self, path: Path):
        self.path = path
        self.data = {"version": 1, "labels": {}}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get(self, clip_key: str):
        return self.data["labels"].get(clip_key)

    def set(self, clip_key: str, label: str) -> None:
        self.data["labels"][clip_key] = {
            "label": label,
            "labeled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        tmp.replace(self.path)  # 原子寫入,中斷不留壞檔

    def count_labeled(self, keys) -> int:
        return sum(1 for k in keys if k in self.data["labels"])


class LabelTool:
    def __init__(self, root: tk.Tk, clip_dir: str):
        self.root = root
        root.title("抽菸片段標記工具")
        self.store = LabelStore(LABELS_PATH)

        self.clips = sorted(glob.glob(os.path.join(clip_dir, "*.mp4")),
                            key=os.path.getmtime)
        if not self.clips:
            raise SystemExit(f"{clip_dir} 下沒有 mp4 片段")
        self.keys = [str(Path(p).as_posix()) for p in self.clips]

        # 骨架關聯率(annotations/pose 回抽結果):輔助判斷骨架品質
        self.pose_rate = {}
        pose_dir = Path("annotations/pose")
        if pose_dir.is_dir():
            import numpy as np
            for p in self.clips:
                npz = pose_dir / (Path(p).stem + ".npz")
                if npz.exists():
                    d = np.load(npz, allow_pickle=True)
                    self.pose_rate[str(Path(p).as_posix())] = \
                        float(d["valid"].mean())

        self.idx = self._first_unlabeled()
        self.cap = None
        self.speed = 1
        self._build_layout()
        self._bind_keys()

        root.lift()
        root.attributes("-topmost", True)
        root.after(1200, lambda: root.attributes("-topmost", False))

        self._open_clip()
        self._tick()

    # ---------- 版面 ----------

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=6)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(main, background="#111111",
                                width=VIDEO_W, height=VIDEO_H,
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._item = self.canvas.create_image(VIDEO_W // 2, VIDEO_H // 2,
                                              anchor="center")

        self.info_var = tk.StringVar()
        ttk.Label(main, textvariable=self.info_var,
                  font=("Microsoft JhengHei", 11)).grid(
            row=1, column=0, sticky="w", pady=(6, 2))

        btns = ttk.Frame(main)
        btns.grid(row=2, column=0, sticky="ew", pady=4)
        for key, code, name, _pos in CATEGORIES:
            ttk.Button(btns, text=f"[{key}] {name}",
                       command=lambda c=code: self.label(c)).pack(
                side="left", padx=3)
        ttk.Button(btns, text="[←] 上一段",
                   command=lambda: self.goto(self.idx - 1)).pack(
            side="left", padx=(18, 3))
        ttk.Button(btns, text="[→] 下一段",
                   command=lambda: self.goto(self.idx + 1)).pack(
            side="left", padx=3)
        ttk.Button(btns, text="[F] 速度",
                   command=self.toggle_speed).pack(side="left", padx=3)

        self.progress_var = tk.StringVar()
        ttk.Label(main, textvariable=self.progress_var).grid(
            row=3, column=0, sticky="w")

    def _bind_keys(self):
        for key, code, _n, _p in CATEGORIES:
            self.root.bind(key, lambda _e, c=code: self.label(c))
        self.root.bind("<Left>", lambda _e: self.goto(self.idx - 1))
        self.root.bind("<Right>", lambda _e: self.goto(self.idx + 1))
        self.root.bind("f", lambda _e: self.toggle_speed())
        self.root.bind("F", lambda _e: self.toggle_speed())
        self.root.bind("q", lambda _e: self.root.destroy())

    # ---------- 片段切換與標記 ----------

    def _first_unlabeled(self) -> int:
        for i, k in enumerate(self.keys):
            if self.store.get(k) is None:
                return i
        return 0

    def _open_clip(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.clips[self.idx])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        self._update_info()

    def _update_info(self):
        key = self.keys[self.idx]
        cur = self.store.get(key)
        mark = f"|| 目前標記:{NAME_OF.get(cur['label'], cur['label'])}" \
            if cur else "|| 未標記"
        rate = self.pose_rate.get(key)
        pose_txt = (f"骨架關聯 {rate:.0%}" if rate is not None
                    else "無節點資料")
        self.info_var.set(
            f"第 {self.idx + 1} / {len(self.clips)} 段  "
            f"{os.path.basename(self.clips[self.idx])}  {mark}"
            f"   [{pose_txt}](速度 {self.speed}x)")
        done = self.store.count_labeled(self.keys)
        n_pos = sum(1 for k in self.keys
                    if (self.store.get(k) or {}).get("label") == "smoking")
        n_bad = sum(1 for k in self.keys
                    if (self.store.get(k) or {}).get("label") == "bad_pose")
        self.progress_var.set(
            f"進度:已標 {done} / {len(self.clips)}"
            f"(抽菸 {n_pos},非抽菸 {done - n_pos - n_bad},"
            f"骨架錯誤排除 {n_bad})   標籤檔:{LABELS_PATH}")

    def goto(self, idx: int):
        self.idx = max(0, min(len(self.clips) - 1, idx))
        self._open_clip()

    def _next_unlabeled(self) -> int:
        for off in range(1, len(self.clips) + 1):
            j = (self.idx + off) % len(self.clips)
            if self.store.get(self.keys[j]) is None:
                return j
        return self.idx  # 全標完:停在原地

    def label(self, code: str):
        self.store.set(self.keys[self.idx], code)
        done = self.store.count_labeled(self.keys)
        if done >= len(self.clips):
            self._update_info()
            self.info_var.set("★ 全部標記完成!標籤已存 "
                              f"{LABELS_PATH}(可按 ←/→ 複查修改)")
            return
        self.goto(self._next_unlabeled())

    def toggle_speed(self):
        self.speed = 2 if self.speed == 1 else 1
        self._update_info()

    # ---------- 播放 ----------

    def _tick(self):
        ok, frame = self.cap.read()
        if self.speed == 2:          # 2x:再丟一幀
            self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循環
            ok, frame = self.cap.read()
        if ok:
            cw = max(self.canvas.winfo_width(), 200)
            ch = max(self.canvas.winfo_height(), 200)
            h, w = frame.shape[:2]
            s = min(cw / w, ch / h)
            frame = cv2.resize(frame, (max(1, int(w * s)),
                                       max(1, int(h * s))))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.coords(self._item, cw // 2, ch // 2)
            self.canvas.itemconfigure(self._item, image=photo)
            self.canvas.image = photo
        self.root.after(max(20, int(1000 / self.fps)), self._tick)


def _enable_dpi_awareness():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main():
    _enable_dpi_awareness()
    parser = argparse.ArgumentParser(description="抽菸片段標記工具")
    parser.add_argument("--dir", default="alarms/clips",
                        help="片段資料夾(預設 alarms/clips)")
    args = parser.parse_args()

    root = tk.Tk()
    LabelTool(root, args.dir)
    root.mainloop()


if __name__ == "__main__":
    main()
