"""深層組合:片段 → token、節律統計、以及無訓練資料時的文法原型評分。

L2 的學習式分類頭需要片段級標籤才能訓練。標籤還沒到位之前,同一組
token/統計量餵給 grammar_scores() 也能直接出分數——用的是可以寫成
一句話的判準(「手貼耳超過 4 秒 = 講電話」),沒有參數要學。

兩者介面相同,infer_hier.py 有權重就用學的、沒有就用文法,
所以系統不會因為「還沒標資料」而停擺,也方便拿文法當學習版的基線。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from stage2.kinematics import K_VALID, SIDE_SLICE
from stage2.primitives import (A_COSELB_MEAN, A_DEAR_MIN, A_DEYE_MIN,
                               A_D_MIN, A_OVEREYE_MAX, A_SPEED_MEAN,
                               Cycle, NUM_PRIMITIVES, P_HOLD, P_REST,
                               SEG_ATTR_DIM, Segment, find_cycles)

# ---- 節律統計版面 ------------------------------------------------------
S_N_SEG = 0            # log1p(片段數)
S_N_HOLD = 1           # log1p(停留片段數)
S_N_CYCLE = 2          # log1p(完整手到臉循環數,需經舉手武裝)
S_HOLD_MEAN = 3        # 停留時長 平均(秒)
S_HOLD_STD = 4
S_HOLD_MAX = 5
S_GAP_MEAN = 6         # 相鄰兩次停留的間隔 平均(秒)
S_GAP_CV = 7           # 間隔的變異係數:小 = 規律 = 節律性動作
S_HOLD_RATIO = 8       # 停留幀數 / 有效幀數
S_D_MIN = 9            # 全段最小腕-鼻距離
S_DEAR_RATIO = 10      # 最小腕-耳 / 最小腕-鼻:<1 表示手更貼耳而非嘴
S_OVEREYE_MAX = 11     # 手高過眼睛的最大幅度
S_COSELB_HOLD = 12     # 停留時的肘彎曲程度
S_VALID_RATIO = 13     # 骨架有效幀比例
S_WINDOW_LOG = 14      # log(視窗長度秒)
S_REST_RATIO = 15      # 靜置幀比例
STAT_DIM = 16

STAT_NAMES = [
    "log片段數", "log停留數", "log循環數", "停留時長均", "停留時長標準差",
    "停留時長最大", "間隔均", "間隔變異係數", "停留佔比", "最小腕鼻距",
    "腕耳腕鼻比", "最大高過眼", "停留肘角cos", "有效比例", "log視窗長",
    "靜置佔比",
]

# 統計量用固定的常數標準化,不用 BatchNorm。
# 理由:(1) 推論時常常 batch=1,BatchNorm 在 train/eval 兩種模式下行為
# 不同,是小資料專案最常見的「訓練好好的、上線就歪掉」;(2) 這 16 個量
# 的量綱本來就已知(秒、比例、餘弦),沒必要讓模型從資料估。
STAT_OFFSET = np.array([1.5, 0.7, 0.4, 1.0, 0.5, 1.5, 3.0, 0.5, 0.2,
                        1.2, 1.0, -0.5, 0.2, 0.5, 2.5, 0.3], np.float32)
STAT_SCALE = np.array([1.0, 0.7, 0.5, 1.0, 0.7, 1.5, 3.0, 0.5, 0.2,
                       0.8, 0.4, 0.5, 0.5, 0.3, 0.5, 0.3], np.float32)


def normalize_stats(stats: np.ndarray) -> np.ndarray:
    """節律統計 → 大致零均值單位尺度(常數標準化,無須訓練)。"""
    return ((stats - STAT_OFFSET) / STAT_SCALE).astype(np.float32)


def build_tokens(segments: List[Segment], frame_embed: Optional[np.ndarray],
                 embed_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """片段序列 → (tokens (N,D), times (N,) 秒)。

    token = 基元 one-hot ‖ 側別 one-hot ‖ 片段屬性 ‖ L1 嵌入(片段內平均)

    frame_embed 為 L1 的逐幀嵌入 (T, 2, E);None 時該段補零(純規則路徑)。
    沒有任何片段時回傳一個全零 token —— 讓下游不必處理空序列,
    「什麼都沒發生」本身也是一種輸入。
    """
    dim = NUM_PRIMITIVES + 2 + SEG_ATTR_DIM + embed_dim
    if not segments:
        return np.zeros((1, dim), np.float32), np.zeros(1, np.float32)

    toks, times = [], []
    for seg in segments:
        v = np.zeros(dim, np.float32)
        v[seg.prim] = 1.0
        v[NUM_PRIMITIVES + (0 if seg.side == "L" else 1)] = 1.0
        o = NUM_PRIMITIVES + 2
        v[o:o + SEG_ATTR_DIM] = seg.attrs
        if frame_embed is not None and embed_dim > 0:
            si = 0 if seg.side == "L" else 1
            v[o + SEG_ATTR_DIM:] = frame_embed[seg.t0:seg.t1, si].mean(0)
        toks.append(v)
        times.append(seg.start_s)
    return np.stack(toks), np.asarray(times, np.float32)


def sequence_stats(segments: List[Segment], kin: np.ndarray,
                   fps: float = 10.0,
                   cycles: Optional[List[Cycle]] = None) -> np.ndarray:
    """整段的節律統計 (STAT_DIM,)。

    這些量刻意做成手工特徵直接接到分類頭前:節律(間隔的規律性)是
    抽菸最強的判準,但要 Transformer 從十幾個 token 自己學出「變異
    係數」這種二階統計,以現有資料量是不現實的。
    """
    T = kin.shape[0]
    s = np.zeros(STAT_DIM, np.float32)
    holds = [g for g in segments if g.prim == P_HOLD]
    cycles = find_cycles(segments, kin) if cycles is None else cycles
    armed = [c for c in cycles if c.armed]

    s[S_N_SEG] = np.log1p(len(segments))
    s[S_N_HOLD] = np.log1p(len(holds))
    s[S_N_CYCLE] = np.log1p(len(armed))

    if holds:
        durs = np.array([g.dur for g in holds], np.float32)
        s[S_HOLD_MEAN], s[S_HOLD_STD] = durs.mean(), durs.std()
        s[S_HOLD_MAX] = durs.max()
        peaks = np.array(sorted(g.start_s for g in holds), np.float32)
        if len(peaks) >= 2:
            gaps = np.diff(peaks)
            s[S_GAP_MEAN] = gaps.mean()
            s[S_GAP_CV] = gaps.std() / max(gaps.mean(), 1e-3)
        s[S_HOLD_RATIO] = sum(g.t1 - g.t0 for g in holds) / max(T, 1)
        s[S_D_MIN] = min(g.attrs[A_D_MIN] for g in holds)
        s[S_DEAR_RATIO] = (min(g.attrs[A_DEAR_MIN] for g in holds) /
                           max(s[S_D_MIN], 1e-3))
        s[S_COSELB_HOLD] = float(np.mean(
            [g.attrs[A_COSELB_MEAN] for g in holds]))
        # 「手有沒有高過眼睛」只在**手停在臉部時**才有意義。
        # 拿整段的最大值會出事:13 秒裡手隨便揮一下高過眼睛,
        # 就會把整段判成抓頭髮。
        s[S_OVEREYE_MAX] = max(g.attrs[A_OVEREYE_MAX] for g in holds)
    else:
        s[S_D_MIN] = 3.0        # 沒有任何停留:視為手從未靠近臉
        s[S_DEAR_RATIO] = 1.0
        s[S_OVEREYE_MAX] = -1.0
    valid = np.maximum(kin[:, SIDE_SLICE["L"]][:, K_VALID],
                       kin[:, SIDE_SLICE["R"]][:, K_VALID])
    s[S_VALID_RATIO] = float((valid > 0.5).mean())
    s[S_WINDOW_LOG] = float(np.log(max(T / fps, 1e-2)))
    s[S_REST_RATIO] = sum(g.t1 - g.t0 for g in segments
                          if g.prim == P_REST) / max(T, 1)
    return s


# ---- 文法原型評分(無參數,標籤到位前的深層判定)------------------------

def _band(x: float, lo: float, hi: float, soft: float = 0.25) -> float:
    """梯形隸屬度:落在 [lo,hi] 給 1,兩側 soft 寬度內線性衰減到 0。

    用軟邊界而不是硬門檻,是第一階段的教訓:硬門檻在邊界上抖,
    「停留 0.99 秒」跟「1.01 秒」不該是判定翻面的分水嶺。
    """
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(soft, 1e-6))
    return max(0.0, 1.0 - (x - hi) / max(soft, 1e-6))


def _le(x: float, thr: float, soft: float = 0.25) -> float:
    """x ≤ thr 的軟隸屬度。"""
    return _band(x, -1e9, thr, soft)


def _ge(x: float, thr: float, soft: float = 0.25) -> float:
    return _band(x, thr, 1e9, soft)


def _fuzzy_and(*factors: float) -> float:
    """多條件合取:取幾何平均,不是連乘。

    連乘會讓「條件多的類別」天生吃虧——抽菸有五條判準、抓頭髮只有
    兩條,同樣每條都 0.8,連乘是 0.33 對 0.64,抓頭髮永遠贏。
    幾何平均把條件數量的影響消掉,比較的才是符合程度。
    任何一條為 0 仍然整體為 0(硬否決的語意保留)。
    """
    f = [max(0.0, min(1.0, x)) for x in factors]
    if min(f) <= 0.0:
        return 0.0
    return float(np.exp(np.mean(np.log(f))))


def grammar_scores(segments: List[Segment], stats: np.ndarray,
                   cycles: Sequence[Cycle] = ()) -> Dict[str, float]:
    """以片段文法給各深層類別打分(0–1,已正規化成分布)。

    每一條都是能用一句話講清楚的判準——這是刻意的:它同時是
    「還沒有標籤時的可用系統」與「學習版 L2 的可比基線」。
    如果訓練出來的 L2 贏不過這個,那就是資料還不夠,不是模型不好。
    """
    armed = [c for c in cycles if c.armed]
    n_cycle = float(len(armed))
    hold_mean = float(stats[S_HOLD_MEAN])
    hold_max = float(stats[S_HOLD_MAX])
    d_min = float(stats[S_D_MIN])
    ear_ratio = float(stats[S_DEAR_RATIO])
    over_eye = float(stats[S_OVEREYE_MAX])
    gap_cv = float(stats[S_GAP_CV])
    cos_elb = float(stats[S_COSELB_HOLD])
    speed = float(np.mean([g.attrs[A_SPEED_MEAN] for g in segments])
                  if segments else 1.0)
    d_eye = float(np.min([g.attrs[A_DEYE_MIN] for g in segments])
                  if segments else 3.0)

    near = _le(d_min, 0.9, 0.3)                    # 手真的到過臉
    at_mouth = _fuzzy_and(near, _ge(ear_ratio, 0.85, 0.2))   # 貼嘴非貼耳
    at_ear = _fuzzy_and(near, _le(ear_ratio, 0.75, 0.15))
    # 手高過眼睛時「比較不像」抽菸,但不是硬否決:抽菸的人手本來就
    # 常停在偏高的位置,頭一低腕高就翻過眼線。軟邊界開寬一點。
    not_above_eye = _le(over_eye, 0.05, 0.40)

    scores = {
        # 抽菸:手到嘴、每口 0.4–3 秒、來回多次、肘彎、節律規律。
        # 次數只要求 ≥1(現有片段只有 13 秒,一根菸的節律根本放不進來);
        # ≥2 次且間隔規律另外加成——那才是抽菸真正的指紋,
        # 等路 B 補錄的 60–90 秒長窗到位,這一項的權重就該調高。
        "smoking": _fuzzy_and(
            at_mouth, not_above_eye,
            _band(hold_mean, 0.4, 3.0, 0.6),
            _ge(n_cycle, 1.0, 0.5),
            _ge(cos_elb, 0.0, 0.4),
            0.5 + 0.3 * _ge(n_cycle, 2.0, 1.0)
            + 0.2 * _le(gap_cv, 0.6, 0.4)),
        # 喝水:同樣到嘴,但單次較長、次數少
        "drinking": _fuzzy_and(
            at_mouth, not_above_eye,
            _band(hold_max, 1.0, 5.0, 1.0),
            _band(n_cycle, 1.0, 2.0, 1.0)),
        # 講電話:貼耳、極長停留、幾乎不放下
        "phone_call": _fuzzy_and(
            at_ear, _ge(hold_max, 4.0, 2.0),
            _band(n_cycle, 1.0, 2.0, 1.5)),
        # 扶眼鏡:眼睛高度、極短、幾乎不動
        "glasses": _fuzzy_and(
            near, not_above_eye, _le(hold_max, 1.2, 0.6),
            _le(speed, 0.8, 0.5), _le(d_eye, 0.9, 0.4)),
        # 抓頭髮:停在臉部時手高過眼睛
        "hair": _fuzzy_and(
            near, _ge(over_eye, 0.15, 0.2),
            _band(hold_mean, 0.3, 3.0, 1.0)),
        # 其他:手根本沒到臉,或到了但不符合上面任何一種型態
        "other": max(0.15, 1.0 - near),
    }
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


@dataclass
class Analysis:
    """一段影片跑完兩層之後的所有中間產物(給 GUI / 訓練 / 除錯共用)。"""
    segments: List[Segment]
    cycles: List[Cycle]
    stats: np.ndarray            # 原始節律統計(給人看)
    tokens: np.ndarray           # (N, TOKEN_DIM)
    times: np.ndarray            # (N,) 秒
    fps: float = 10.0

    @property
    def norm_stats(self) -> np.ndarray:
        return normalize_stats(self.stats)


def analyze(kin: np.ndarray, fps: float = 10.0,
            prim: Optional[np.ndarray] = None,
            frame_embed: Optional[np.ndarray] = None,
            embed_dim: int = 0, **seg_kw) -> Analysis:
    """運動學(+ 可選的 L1 輸出)→ 片段、循環、統計、token。

    prim 為 None 時走規則路徑(L1 尚未訓練);給了 L1 的逐幀預測
    (T, 2)就走學習路徑。兩條路徑之後的處理完全相同,所以
    「有沒有訓練 L1」不會改變下游任何一行程式。
    """
    from stage2.primitives import (IGNORE, segment_primitives,
                                   segments_from_kinematics)
    if prim is None:
        segments = segments_from_kinematics(kin, fps, **seg_kw)
    else:
        segments = []
        prim = np.asarray(prim).copy()
        for si, side in enumerate(("L", "R")):
            # L1 對每一幀都會給答案,包括腕點根本偵測不到的幀——那裡它
            # 沒有輸入,輸出是憑空生成的。實測有片段 96% 的幀量不到手腕,
            # 卻照樣切出六段「停留臉部」。沒有輸入就不該有輸出:
            # 這些幀退回棄權,片段自然斷開。
            #
            # 規則棄權的另一種情況(鼻點不可信、門檻模糊帶)**不遮**——
            # 那裡手臂幾何是在的,由網路補上正是 L1 存在的理由。
            no_input = kin[:, SIDE_SLICE[side]][:, K_VALID] < 0.5
            prim[no_input, si] = IGNORE
            segments += segment_primitives(prim[:, si], kin, side, fps,
                                           **seg_kw)
        segments.sort(key=lambda x: (x.t0, x.side))
    cycles = find_cycles(segments, kin)
    stats = sequence_stats(segments, kin, fps, cycles)
    tokens, times = build_tokens(segments, frame_embed, embed_dim)
    return Analysis(segments=segments, cycles=cycles, stats=stats,
                    tokens=tokens, times=times, fps=fps)


def explain(segments: Sequence[Segment], stats: np.ndarray,
            scores: Dict[str, float], top_k: int = 3) -> str:
    """人看的說明:基元時間軸 + 判定依據。GUI 與誤報複查用。"""
    from stage2.taxonomy import DEEP_NAMES
    lines = ["【淺層基元時間軸】"]
    lines += ["  " + s.describe() for s in segments] or ["  (無片段)"]
    lines.append("【節律統計】")
    lines.append(f"  停留 {np.expm1(stats[S_N_HOLD]):.0f} 次、"
                 f"完整循環 {np.expm1(stats[S_N_CYCLE]):.0f} 次、"
                 f"單次平均 {stats[S_HOLD_MEAN]:.1f}s、"
                 f"間隔變異 {stats[S_GAP_CV]:.2f}、"
                 f"最小腕鼻距 {stats[S_D_MIN]:.2f}、"
                 f"腕耳/腕鼻 {stats[S_DEAR_RATIO]:.2f}")
    lines.append("【深層判定】")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]:
        lines.append(f"  {DEEP_NAMES.get(k, k)}  {v:.2f}")
    return "\n".join(lines)
