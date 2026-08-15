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
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# 批次大小。4 是實測的甜蜜點:1 → 28.1ms/幀、4 → 14.7ms/幀,
# 但 8 → 50.5ms、16 → 58.3ms —— 6GB VRAM 到批次 8 就塞不下了。
BATCH = 4

CACHE_DIR = ".analysis"
CACHE_VERSION = 1


class _Batcher:
    """把管線的偵測器換成「先批次算好、再依序取用」的代理。

    管線內部是 `self.detector.detect(frame)` 一幀一呼叫;要批次化又不想
    改動即時管線的程式,最小侵入的做法就是換掉那個物件。

    **對應關係是這裡最容易錯的地方**:結果必須與送進去的影格同順序、
    一對一。錯位的話標記會整個偏掉,而且畫面上看不出來——所以取用時
    檢查是否用完,用完還被叫就直接退回真正的偵測器,不猜。
    """

    def __init__(self, pipe, batch: int):
        self.real = pipe.detector
        self.batch = batch
        self._queue = []
        pipe.detector = self

    def preload(self, frames) -> None:
        self._queue = list(self.real.detect_batch(frames))

    def detect(self, frame):
        if self._queue:
            return self._queue.pop(0)
        return self.real.detect(frame)   # 沒預載到就照常算,寧可慢不要錯


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
                  overrides: Optional[dict] = None,
                  on_progress: Optional[Callable] = None,
                  cancel: Optional[threading.Event] = None) -> Analysis:
    """整支影片跑一次判定,回傳抽菸時間與骨架。

    on_progress(比例或 None, 已找到幾處, 已處理到影片第幾秒, 已花幾秒)
    ——比例可能是 None,因為有些容器(尤其 .ts)問不到總幀數。

    同一趟同時產出兩者:管線本來就要偵測姿態才能判抽菸,關鍵點順手留下來
    就好——分開跑等於把最貴的那一段做兩次。
    """
    from inference.pipeline import SmokingDetectionPipeline
    from utils import load_config

    # 覆寫值要在這裡套進去:分析與即時偵測必須吃同一份設定,不然畫面上
    # 調好的門檻與分析標出來的位置對不起來
    from ui.settings import apply_overrides
    cfg = apply_overrides(load_config(infer_config), overrides or {})
    model_cfg, ckpt = None, None
    if method is not None and method.needs_appearance:
        model_cfg = load_config("configs/model.yaml")
        ckpt = "checkpoints/hmdb_e2e_best.pt"
    pipe = SmokingDetectionPipeline(cfg, model_cfg, ckpt_path=ckpt,
                                    use_model=bool(model_cfg), method=method)
    hits = []
    # 標記指向「證據起點」——等待狀態下促成第一次計入事件的那次抬手,
    # 而不是警報成立的時刻。觸發是累積夠了的結論,可能落在動作結束後
    # 十幾秒,點過去只會看到人站著。
    pipe.on_alarm = lambda tid, P, t, fr, ev: hits.append(ev)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 <= fps <= 240.0 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / cfg["sampling"]["target_fps"]))
    out = Analysis(stride=step, fps=fps)
    i = 0
    started = time.time()
    # 批次偵測:每次 predict() 有約 28ms 的固定開銷(換小模型、降解析度
    # 都省不掉),批次 4 攤掉之後每幀約 14.7ms。管線的狀態機必須循序,
    # 所以只把「偵測」批次化:先算好一批,再依序餵回去。
    batcher = _Batcher(pipe, BATCH) if pipe.skeleton_enabled else None
    pending = []          # [(幀號, 影格)] 等著送批次的取樣幀

    def flush():
        """把累積的取樣幀批次偵測完,再依序走管線。

        依序這件事不能省:狀態機、計數器、警報的 EMA 全都有時序狀態,
        亂序餵進去結果就不是同一回事了。批次化的只有偵測本身。
        """
        if not pending:
            return
        batcher.preload([f for _idx, f in pending])
        for idx, f in pending:
            res = pipe.step(f, idx / fps)
            kp = [r["kpts"] for r in res.values()
                  if r.get("kpts") is not None]
            if kp:
                out.poses[idx] = kp
            if hits:
                out.alarms.extend(hits)
                hits.clear()
        pending.clear()

    try:
        while True:
            if cancel is not None and cancel.is_set():
                break
            if i % step:
                # 非取樣幀只 grab 不 read:grab 不做色彩轉換也不回傳陣列。
                # 每 step 幀只有一幀會被用到,其餘全解出來是白做的——實測
                # 1080p60 一分鐘的片,全部 read 要 20.8 秒,改用 grab 只要
                # 5.8 秒。這是整個分析裡最容易省掉的一段。
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if batcher is not None:
                pending.append((i, frame))
                if len(pending) >= BATCH:
                    flush()
            else:
                res = pipe.step(frame, i / fps)
                kp = [r["kpts"] for r in res.values()
                      if r.get("kpts") is not None]
                if kp:
                    out.poses[i] = kp
                if hits:
                    out.alarms.extend(hits)
                    hits.clear()
            if on_progress is not None and i % (step * 20) == 0:
                # 總幀數不一定拿得到(.ts 錄影檔常回報 0),拿不到就回報
                # None,讓介面顯示「已處理多久」而不是卡在 0%
                frac = (i / total) if total else None
                on_progress(frac, len(out.alarms), i / fps,
                            time.time() - started)
            i += 1
        flush()               # 收尾:最後不滿一批的也要處理
    finally:
        cap.release()
        pipe.close()
    if on_progress is not None:
        on_progress(1.0, len(out.alarms), i / fps, time.time() - started)
    return out


class AnalysisJob:
    """在背景跑的一次分析。可以邊播邊跑,也可以等它跑完再播。

    存在的理由:分析大約要影片長度的兩倍時間(偵測器得跑過整支片),
    20 分鐘的片就是 10 分鐘。硬要使用者乾等完才給看是不合理的,所以
    把「等待」與「分析」拆開——同一個工作,要嘛在對話框裡等它,要嘛
    丟到背景邊播邊出標記。
    """

    def __init__(self, path: str, method=None,
                 infer_config: str = "configs/inference.yaml",
                 overrides: Optional[dict] = None):
        self.path = str(path)
        self.overrides = overrides or {}
        self.result: Optional[Analysis] = None
        self.error: Optional[str] = None
        self.progress = None          # (比例, 找到幾處, 已到第幾秒, 已花幾秒)
        self._cancel = threading.Event()
        self._thread = threading.Thread(
            target=self._work, args=(method, infer_config), daemon=True)
        self._thread.start()

    def _work(self, method, infer_config):
        try:
            a = analyse_video(self.path, method=method,
                              infer_config=infer_config,
                              overrides=self.overrides,
                              on_progress=self._on_progress,
                              cancel=self._cancel)
            if not self._cancel.is_set():
                try:
                    a.save(self.path)
                except OSError:
                    pass          # 資料夾唯讀之類:分析結果照樣可用
                self.result = a
        except Exception as e:
            self.error = str(e)

    def _on_progress(self, *args):
        self.progress = args

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()
