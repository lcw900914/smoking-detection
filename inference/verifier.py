"""第二階段複核:警報候選 → 骨架時序判定 → **降級不否決**。

第一階段的職責是高召回,誤報多是可以接受的;這裡的職責是把誤報降級。
核心原則寫死在 `decide()` 裡:複核判「非抽菸」只把警報從紅色降為橘色
「待人工複查」,**絕不取消警報**。召回由結構保證不會下降。

三種狀態,對應三種不同的事實:

    confirmed  複核也認為是抽菸        → 維持紅色
    review     複核認為是別的動作      → 降為橘色待複查
    abstain    複核沒有足以判斷的輸入  → 維持紅色(不算證據,也不扣分)

`abstain` 不是湊出來的第三態,是必要的:實測 47% 的片段鼻點可信幀不到
20%、有兩段人工標為抽菸的片段腕點可見率只有 4% 與 18%。骨架看不到手的
時候,任何判定都是憑空生成的(見 `docs/專案現況與後續計畫.md` 第 8 節
地雷 8「沒有輸入就不該有輸出」)。若把這種情況也降級,等於因為攝影機
角度不好就把真警報壓掉——那正是這個模組被明令禁止做的事。
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CONFIRMED = "confirmed"
REVIEW = "review"
ABSTAIN = "abstain"

STATUS_NAMES = {CONFIRMED: "已確認", REVIEW: "待複查",
                ABSTAIN: "無法複核", "pending": "複核中"}

# 複核判定的兩個門檻(可由 configs/inference.yaml 的 verify: 覆寫)
MIN_SMOKING = 0.25      # 抽菸分數 ≥ 此值就維持紅色(寧可留著給人看)
MIN_VALID_RATIO = 0.15  # 骨架有效幀比例低於此值 → 棄權,不做判定
MIN_SPAN_SEC = 3.0      # 可複核的最短序列長度


@dataclass
class VerifyResult:
    """一次複核的完整結果(GUI 顯示與記錄用)。"""
    status: str                       # confirmed / review / abstain
    top: str = "other"                # 最高分的深層類別
    smoking: float = 0.0              # 抽菸維度分數
    valid_ratio: float = 0.0          # 骨架有效幀比例
    span_sec: float = 0.0             # 實際複核的序列長度
    source: str = ""                  # grammar / learned
    scores: Dict[str, float] = field(default_factory=dict)
    detail: str = ""                  # explain() 的完整文字
    reason: str = ""                  # 一句話結論(記錄列用)

    @property
    def downgraded(self) -> bool:
        """是否該把紅色警報降為橘色。"""
        return self.status == REVIEW

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, self.status)


def pose_window(hist: Sequence[Tuple[float, Optional[np.ndarray]]],
                now: float, window_sec: float,
                fps: float) -> Tuple[np.ndarray, float]:
    """滾動節點歷史 → 均勻時間格上的 (T,17,3) 序列與實際跨度秒數。

    重取樣到固定 fps 的格子上,而不是直接把歷史堆成陣列:stage2 的
    速度、停留時長、間隔變異全都以「幀距 = 1/fps」為前提,漏幀直接
    堆疊會讓時間軸壓縮,停留 3 秒看起來像 1 秒。

    沒有偵測到人的幀留零(信心值 0),下游的 `K_VALID` 會把它當成
    棄權——這與 `scripts/gui.py:_write_pose` 的零填語意一致。

    跨度取「window_sec 或這個 track 實際出現多久」的較小者:track 才
    進場 12 秒就配一個 90 秒的窗,會有 78 秒的零幀把有效比例洗掉,
    然後被誤判成「骨架不可用」。
    """
    if not hist:
        return np.zeros((0, 17, 3), np.float32), 0.0
    span = min(float(window_sec), max(0.0, now - hist[0][0]))
    if span <= 0:
        return np.zeros((0, 17, 3), np.float32), 0.0
    # +1:格子涵蓋 [now-span, now] 兩端點,否則最後一幀(剛好落在 now 的
    # 那一幀,也就是觸發當下)算出的索引等於格數,會被丟掉
    n = max(1, int(round(span * fps)) + 1)
    t0 = now - span
    kpts = np.zeros((n, 17, 3), np.float32)
    for ts, k in hist:
        if k is None or ts < t0:
            continue
        idx = int(round((ts - t0) * fps))
        if 0 <= idx < n:
            kpts[idx] = k
    return kpts, span


def decide(scores: Dict[str, float], valid_ratio: float, span_sec: float,
           min_smoking: float = MIN_SMOKING,
           min_valid_ratio: float = MIN_VALID_RATIO,
           min_span_sec: float = MIN_SPAN_SEC) -> Tuple[str, str]:
    """深層分數 + 骨架品質 → (狀態, 一句話理由)。

    刻意做成不依賴任何模型的純函式:降級不否決這條原則是本專案的紅線,
    要能被單獨測試,不能藏在推論流程裡。
    """
    if span_sec < min_span_sec:
        return ABSTAIN, f"序列只有 {span_sec:.1f} 秒,不足以判斷節律"
    if valid_ratio < min_valid_ratio:
        return ABSTAIN, (f"骨架有效幀僅 {valid_ratio:.0%}"
                         f"(< {min_valid_ratio:.0%}),看不到手臂")
    smoking = float(scores.get("smoking", 0.0))
    top = max(scores, key=scores.get) if scores else "other"
    if top == "smoking":
        return CONFIRMED, f"複核同意:抽菸 {smoking:.2f} 為最高分"
    if smoking >= min_smoking:
        # 抽菸沒拿到第一,但分數還在合理範圍 → 維持紅色。
        # 這一條是「降級不否決」的具體實作:降級的門檻要比升級嚴,
        # 誤降一次真警報的代價遠高於多留一個橘色待複查。
        return CONFIRMED, (f"複核判 {top},但抽菸 {smoking:.2f} "
                           f"仍達 {min_smoking:.2f} → 不降級")
    return REVIEW, f"複核判 {top}(抽菸僅 {smoking:.2f})→ 降為待複查"


class SecondStageVerifier:
    """把 stage2 的兩層模型包成「複核一段節點序列」這一件事。

    mode 對應 `inference/methods.py` 的 stage2 欄位:
        grammar      規則基元 + 片段文法(零學習權重)
        l1+grammar   L1 網路基元 + 片段文法
        l1+l2        L1 + 學習版 L2

    模型在第一次呼叫 `verify()` 時才載入(lazy):選純規則方法的人不該
    為了一個用不到的 ST-GCN 等 torch 暖機。
    """

    def __init__(self, mode: str, l1_ckpt: Optional[str] = None,
                 l2_ckpt: Optional[str] = None, device: str = "auto",
                 min_smoking: float = MIN_SMOKING,
                 min_valid_ratio: float = MIN_VALID_RATIO,
                 min_span_sec: float = MIN_SPAN_SEC):
        if mode not in ("grammar", "l1+grammar", "l1+l2"):
            raise ValueError(f"未知的複核模式 {mode!r}")
        self.mode = mode
        self.l1_ckpt = l1_ckpt if mode in ("l1+grammar", "l1+l2") else None
        self.l2_ckpt = l2_ckpt if mode == "l1+l2" else None
        self.device = device
        self.min_smoking = min_smoking
        self.min_valid_ratio = min_valid_ratio
        self.min_span_sec = min_span_sec
        self._rec = None

    @property
    def recognizer(self):
        if self._rec is None:
            from stage2.infer_hier import HierarchicalRecognizer
            self._rec = HierarchicalRecognizer(
                l1_ckpt=self.l1_ckpt, l2_ckpt=self.l2_ckpt,
                device=self.device)
        return self._rec

    def verify(self, kpts: np.ndarray, fps: float = 10.0,
               span_sec: Optional[float] = None) -> VerifyResult:
        """複核一段節點序列 (T,17,3)。"""
        from stage2.composition import S_VALID_RATIO, explain
        from stage2.taxonomy import DEEP_NAMES

        T = 0 if kpts is None else len(kpts)
        span = float(T / max(fps, 1e-6)) if span_sec is None else float(span_sec)
        if T < 2:
            status, reason = ABSTAIN, "沒有節點序列可複核"
            return VerifyResult(status=status, span_sec=span, reason=reason)

        out = self.recognizer.predict(kpts, fps)
        a = out["analysis"]
        valid = float(a.stats[S_VALID_RATIO])
        status, reason = decide(
            out["scores"], valid, span,
            self.min_smoking, self.min_valid_ratio, self.min_span_sec)
        detail = explain(a.segments, a.stats, out["scores"])
        head = (f"複核模式:{self.mode}   序列 {span:.1f}s   "
                f"骨架有效 {valid:.0%}")
        return VerifyResult(
            status=status, top=out["top"],
            smoking=float(out["scores"].get("smoking", 0.0)),
            valid_ratio=valid, span_sec=span, source=out["source"],
            scores=out["scores"],
            detail="\n".join([head, detail,
                              f"→ {STATUS_NAMES.get(status, status)}:{reason}",
                              f"(最高分:{DEEP_NAMES.get(out['top'], out['top'])})"]),
            reason=reason)


def build(method, cfg: Optional[dict] = None) -> Optional[SecondStageVerifier]:
    """依 Method 建立複核器;方法不含 stage2 時回傳 None。"""
    if method.stage2 is None:
        return None
    v = (cfg or {}).get("verify", {})
    l1, l2 = method.ckpts
    return SecondStageVerifier(
        method.stage2, l1_ckpt=l1, l2_ckpt=l2,
        device=v.get("device", "auto"),
        min_smoking=float(v.get("min_smoking", MIN_SMOKING)),
        min_valid_ratio=float(v.get("min_valid_ratio", MIN_VALID_RATIO)),
        min_span_sec=float(v.get("min_span_sec", MIN_SPAN_SEC)))
