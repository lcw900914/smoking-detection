"""影片分析:播放**之前**先跑一次判定,把抽菸位置與骨架都算好。

流程刻意是「先分析、再播放」而不是「邊播邊算」:

- 邊播邊算的話,骨架要嘛每幀重跑偵測(60fps 的片子根本來不及),
  要嘛畫面已經過去了結果才算出來
- 抽菸位置更不可能邊播邊給:警報要看一整段的節律才成立,
  播到一半才冒出標記,使用者早就滑過去了

**原始影片完全不動。** 結果寫成旁邊的側車檔(`.analysis/<檔名>.npz`),
所以同一支片子第二次開就直接載入,不必重跑——分析一支十分鐘的片要好幾
分鐘,沒有快取的話這個流程沒人受得了。
"""
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

CACHE_DIR = ".analysis"
CACHE_VERSION = 1


def cache_path(video: str) -> Path:
    """側車檔位置。放在子資料夾而不是影片旁邊,影片清單才不會被雜檔塞滿。"""
    v = Path(video)
    return v.parent / CACHE_DIR / f"{v.stem}.npz"


class Analysis:
    """一支影片的分析結果。"""

    def __init__(self, alarms=None, poses=None, stride: int = 1,
                 fps: float = 30.0):
        self.alarms = list(alarms or [])      # 警報時間(秒)
        self.poses = dict(poses or {})        # 幀號 → [kpts, ...]
        self.stride = max(1, int(stride))     # 每幾幀分析一次
        self.fps = float(fps)

    def __bool__(self) -> bool:
        return bool(self.alarms or self.poses)

    # ---- 存取 ----

    def save(self, video: str) -> Path:
        """存成 npz。不用 JSON:骨架是 (17,3) 的浮點陣列,一支十分鐘的片
        會有上萬組,JSON 又大又慢。"""
        dst = cache_path(video)
        dst.parent.mkdir(parents=True, exist_ok=True)
        idx = sorted(self.poses)
        counts = [len(self.poses[i]) for i in idx]
        flat = ([k for i in idx for k in self.poses[i]]
                or [np.zeros((17, 3), np.float32)])
        np.savez_compressed(
            dst, version=CACHE_VERSION, alarms=np.asarray(self.alarms,
                                                          np.float32),
            pose_idx=np.asarray(idx, np.int32),
            pose_counts=np.asarray(counts, np.int32),
            pose_data=np.asarray(flat, np.float32),
            stride=self.stride, fps=self.fps)
        return dst

    @classmethod
    def load(cls, video: str) -> Optional["Analysis"]:
        src = cache_path(video)
        if not src.exists():
            return None
        try:
            d = np.load(src, allow_pickle=False)
            if int(d["version"]) != CACHE_VERSION:
                return None
            poses, at = {}, 0
            for i, n in zip(d["pose_idx"], d["pose_counts"]):
                poses[int(i)] = [d["pose_data"][at + j] for j in range(int(n))]
                at += int(n)
            return cls(alarms=[float(x) for x in d["alarms"]], poses=poses,
                       stride=int(d["stride"]), fps=float(d["fps"]))
        except Exception:
            return None          # 側車檔壞了就當沒有,重新分析即可


def analyse_video(path: str, method=None,
                  infer_config: str = "configs/inference.yaml",
                  on_progress: Optional[Callable] = None,
                  cancel: Optional[threading.Event] = None) -> Analysis:
    """整支影片跑一次判定,回傳抽菸時間與骨架。

    同一趟同時產出兩者:管線本來就要偵測姿態才能判抽菸,關鍵點順手留下來
    就好——分開跑等於把最貴的那一段做兩次。
    """
    from inference.pipeline import SmokingDetectionPipeline
    from utils import load_config

    cfg = load_config(infer_config)
    model_cfg, ckpt = None, None
    if method is not None and method.needs_appearance:
        model_cfg = load_config("configs/model.yaml")
        ckpt = "checkpoints/hmdb_e2e_best.pt"
    pipe = SmokingDetectionPipeline(cfg, model_cfg, ckpt_path=ckpt,
                                    use_model=bool(model_cfg), method=method)
    hits = []
    pipe.alarm.callback = lambda tid, P, t, fr: hits.append(t)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 <= fps <= 240.0 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / cfg["sampling"]["target_fps"]))
    out = Analysis(stride=step, fps=fps)
    i = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                res = pipe.step(frame, i / fps)
                kp = [r["kpts"] for r in res.values()
                      if r.get("kpts") is not None]
                if kp:
                    out.poses[i] = kp
                if hits:
                    out.alarms.extend(hits)
                    hits.clear()
                if on_progress is not None and total and i % (step * 20) == 0:
                    on_progress(i / total, len(out.alarms))
            i += 1
    finally:
        cap.release()
        pipe.close()
    if on_progress is not None:
        on_progress(1.0, len(out.alarms))
    return out
