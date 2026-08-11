"""淺層動作基元:詞彙、規則偽標籤、分段與片段屬性。

**為什麼基元用規則產生標籤,而不是叫人標?**
「舉手 / 放下 / 停留」是幾何量的直接後果——腕-鼻距離在縮小就是舉手。
這種東西人標不會比公式準,只會比較慢而且不一致。所以淺層走
weak supervision:規則產生逐幀偽標籤,網路(L1)去學它。

那為什麼還要網路?因為規則在三種情況會壞掉,而網路能從整段時序與
整張拓樸圖補回來:
  1. 腕點被遮擋/幻覺(手背在身後時姿態模型會把腕點畫在衣領上)
  2. 鼻點不可信(側面、低頭)時規則整段棄權,網路仍可由肘與肩推斷
  3. 門檻邊界抖動——規則會在 near 附近來回跳,網路輸出平滑得多

規則標不準的地方(門檻附近的模糊帶)一律標成 IGNORE(−1),不進損失。
寧可少教,不要教錯。

分段輸出的 Segment 就是「淺層動作片段」,深層(L2)吃的是它們的序列。
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from stage2.kinematics import (K_COS_ELBOW, K_D_EAR, K_D_EYE, K_D_NOSE,
                               K_FACE_OK, K_H_OVER_EYE, K_H_WRI, K_PHI_X,
                               K_PHI_Y, K_SPEED, K_VALID, SIDE_SLICE,
                               SIDES, smooth1d)

# ---- 基元詞彙 ---------------------------------------------------------
PRIMITIVES = ["rest", "raise", "hold", "lower", "free"]
PRIM_NAMES = {"rest": "靜置", "raise": "舉手", "hold": "停留臉部",
              "lower": "放下", "free": "其他活動"}
P_REST, P_RAISE, P_HOLD, P_LOWER, P_FREE = range(5)
NUM_PRIMITIVES = len(PRIMITIVES)
IGNORE = -1

# ---- 規則門檻(與第一階段的骨架分支同一套語意)-------------------------
NEAR = 0.9          # 腕-鼻距離 < NEAR × 身體尺度 → 手在臉部
NEAR_MARGIN = 0.12  # 門檻兩側的模糊帶,落在裡面的幀不給標籤
MOVE_RATE = 0.6     # 腕-鼻距離變率(每秒)超過此值 → 舉手/放下
RATE_MARGIN = 0.25
# 腕速(身體尺度/秒)低於此值視為靜止。看起來偏大是因為它不是物理速度
# 而是「量測速度」:關鍵點抖動本身就會產生約 0.3 的底噪,真正靜止的手
# 量出來是 0.2–0.5,不是 0。門檻壓到物理直覺的數字會讓 rest 幾乎不存在。
STILL_SPEED = 0.8
LOW_WRIST = -0.5    # 腕高低於肩線這麼多 → 手是放下的
LOW_MARGIN = 0.15
REACH_MAX = 3.0     # 腕-鼻距離超過這麼遠就不是在做「往臉部去」的動作


def rule_primitives(kin: np.ndarray, side: str, fps: float = 10.0,
                    near: float = NEAR, lookback_s: float = 0.5
                    ) -> np.ndarray:
    """規則偽標籤:(T, 45) 運動學特徵 → (T,) 基元索引,模糊處為 −1。

    判斷順序(前面的優先):
        腕點不可見 / 身體座標系不成立   → −1
        腕-鼻距離 < near               → hold(手在臉部)
        距離在 near 邊界的模糊帶        → −1
        距離在縮小(且已接近)           → raise
        距離在放大(且原本接近)         → lower
        變率模糊帶                      → −1
        幾乎不動 + 手低於肩             → rest
        其餘(在動、或抬在半空)         → free

    rest 與 free 只需要腕點,鼻點不可信時仍然可判——俯視角有七成的幀
    看不到鼻子,若整幀棄權,L1 會有七成沒有標籤可學。
    """
    blk = kin[:, SIDE_SLICE[side]]
    T = blk.shape[0]
    valid = blk[:, K_VALID] > 0.5
    face = blk[:, K_FACE_OK] > 0.5
    d = smooth1d(blk[:, K_D_NOSE], 3)
    h = smooth1d(blk[:, K_H_WRI], 3)
    speed = smooth1d(blk[:, K_SPEED], 3)

    lb = max(1, int(round(lookback_s * fps)))
    d_past = np.concatenate([np.full(lb, np.nan, np.float32), d[:-lb]])
    past_ok = np.concatenate([np.zeros(lb, bool), face[:-lb]])
    # 接近率:>0 表示手正在靠近臉(距離縮小)
    rate = np.where(past_ok & face, (d_past - d) / lookback_s, np.nan)

    prim = np.full(T, IGNORE, np.int8)

    near_lo, near_hi = near - NEAR_MARGIN, near + NEAR_MARGIN
    move_lo, move_hi = MOVE_RATE - RATE_MARGIN, MOVE_RATE + RATE_MARGIN

    is_hold = face & (d < near_lo)
    far = face & (d > near_hi)
    has_rate = far & ~np.isnan(rate)
    is_raise = has_rate & (rate > move_hi) & (d < near + REACH_MAX)
    is_lower = has_rate & (rate < -move_hi) & (d_past < near + REACH_MAX)

    # rest / free 不需要鼻點:只看手的高度與速度。
    # 但手若正在朝臉部移動(rate 明顯非零),就不算靜置/雜項,留給上面判。
    quiet = valid & ~(has_rate & (np.abs(rate) >= move_lo))
    is_rest = quiet & (h < LOW_WRIST - LOW_MARGIN) & (speed < STILL_SPEED)
    is_free = quiet & ~is_rest & (
        (h > LOW_WRIST + LOW_MARGIN) | (speed > STILL_SPEED))

    prim[is_free] = P_FREE
    prim[is_rest] = P_REST
    prim[is_lower] = P_LOWER
    prim[is_raise] = P_RAISE
    prim[is_hold] = P_HOLD
    return prim


def rule_primitives_both(kin: np.ndarray, fps: float = 10.0,
                         near: float = NEAR) -> np.ndarray:
    """兩側一起算:(T, 2) —— 第 0 欄左手、第 1 欄右手。"""
    return np.stack([rule_primitives(kin, s, fps, near) for s in SIDES],
                    axis=1)


# ---- 片段屬性版面 ------------------------------------------------------
A_LOG_DUR = 0        # log(片段長度秒)
A_D_MIN = 1          # 片段內最小腕-鼻距離(最接近臉的程度)
A_D_MEAN = 2
A_D_START = 3
A_D_END = 4
A_D_DELTA = 5        # 末 − 首:舉手為負、放下為正
A_DEAR_MIN = 6       # 最小腕-耳距離(講電話 << 腕-鼻)
A_DEYE_MIN = 7
A_H_MAX = 8          # 腕最高點(相對肩線)
A_H_MEAN = 9
A_OVEREYE_MAX = 10   # 腕高減眼高的最大值:>0 代表手曾高過眼睛(抓頭髮)
A_COSELB_MEAN = 11   # 肘彎曲程度
A_COSELB_MAX = 12
A_SPEED_MEAN = 13
A_SPEED_MAX = 14
A_PHIX_MEAN = 15     # 手相對鼻的平均方位(側向 / 上下)
A_PHIY_MEAN = 16
A_VALID_RATIO = 17
A_T0_NORM = 18       # 片段起點在視窗中的相對位置 0–1
A_GAP_PREV = 19      # 與同側上一片段的間隔(秒,log1p)
A_FACE_RATIO = 20    # 片段內「臉部量可用」的幀比例
SEG_ATTR_DIM = 21

# 臉部量整段都量不到時填的哨兵值:當成「手離臉很遠」而不是 0。
# 0 的語意是「手貼在鼻子上」,恰好是最強的正證據——用 0 當缺值,
# 等於每次鼻點不可信就送模型一個假陽性。
FAR = 3.0

# 一個片段要拿得出「手確實靠近臉」的證據,至少需要這麼多幀量到鼻點
MIN_FACE_FRAMES = 3

SEG_ATTR_NAMES = [
    "log時長", "最小腕鼻距", "平均腕鼻距", "起始腕鼻距", "結束腕鼻距",
    "腕鼻距變化", "最小腕耳距", "最小腕眼距", "最高腕高", "平均腕高",
    "最大高過眼", "平均肘角cos", "最大肘角cos", "平均腕速", "最大腕速",
    "平均方位x", "平均方位y", "有效比例", "起點位置", "與前段間隔",
]


@dataclass
class Segment:
    """一段淺層動作片段(深層模型的輸入單位)。"""
    side: str
    prim: int
    t0: int                     # 起始幀(含)
    t1: int                     # 結束幀(不含)
    fps: float
    attrs: np.ndarray = field(default_factory=lambda: np.zeros(SEG_ATTR_DIM,
                                                               np.float32))

    @property
    def dur(self) -> float:
        return (self.t1 - self.t0) / self.fps

    @property
    def start_s(self) -> float:
        return self.t0 / self.fps

    @property
    def name(self) -> str:
        return PRIM_NAMES[PRIMITIVES[self.prim]]

    @property
    def face_frames(self) -> int:
        """片段內鼻點可信的幀數。"""
        return int(round((self.t1 - self.t0) * self.attrs[A_FACE_RATIO]))

    @property
    def face_measured(self) -> bool:
        """這段期間鼻點是否可信到足以支撐「手在臉部」的說法。

        門檻用**幀數**不用比例:比例會隨片段長度浮動——同樣 3 幀量到,
        0.5 秒的片段是 60%、2 秒的片段是 15%,但證據是一樣多的。
        3 幀(0.3 秒)是為了排掉單幀閃現的僥倖量測。
        """
        return self.face_frames >= MIN_FACE_FRAMES

    def describe(self) -> str:
        side = "左手" if self.side == "L" else "右手"
        d = (f"腕鼻距 {self.attrs[A_D_MIN]:.2f}" if self.face_measured
             else "腕鼻距 量不到")
        return (f"{self.start_s:5.1f}s {side} {self.name}"
                f"({self.dur:.1f}s, {d})")


def _rle(labels: np.ndarray):
    """連續相同標籤的區段 [(值, 起, 迄)]。"""
    runs = []
    if len(labels) == 0:
        return runs
    start = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            runs.append((int(labels[start]), start, t))
            start = t
    return runs


def _mode_filter(labels: np.ndarray, win: int = 5) -> np.ndarray:
    """眾數濾波:壓掉單幀跳動。

    **只平滑已經有標籤的幀**。早期版本讓濾波順手把 −1 也填成鄰居的
    標籤,結果「停留臉部」會沿著遮擋空窗蔓延到腕-鼻距離 1.3 的幀上,
    片段屬性整個失真。沒量到就是沒量到,不要用鄰居的值假裝量到了。
    """
    if win <= 1:
        return labels
    T = len(labels)
    out = labels.copy()
    half = win // 2
    for t in range(T):
        if labels[t] < 0:
            continue
        seg = labels[max(0, t - half):min(T, t + half + 1)]
        seg = seg[seg >= 0]
        vals, cnt = np.unique(seg, return_counts=True)
        out[t] = vals[cnt.argmax()]
    return out


def _bridge_gaps(labels: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """短暫的 −1 空窗(關鍵點閃爍)以兩側一致的標籤補起來。

    只補「兩側標籤相同且空窗夠短」的情形——手真的消失一秒以上時,
    中間發生了什麼是未知的,硬補會製造出不存在的長片段。
    """
    out = labels.copy()
    T = len(labels)
    t = 0
    while t < T:
        if out[t] >= 0:
            t += 1
            continue
        s = t
        while t < T and out[t] < 0:
            t += 1
        if (s > 0 and t < T and t - s <= max_gap
                and out[s - 1] == out[t]):
            out[s:t] = out[s - 1]
    return out


def segment_primitives(prim: np.ndarray, kin: np.ndarray, side: str,
                       fps: float = 10.0, min_dur: float = 0.3,
                       smooth_win: int = 5) -> List[Segment]:
    """逐幀基元 → 片段序列(已濾波、已併掉過短的碎片、已算屬性)。

    min_dur 預設 0.3 秒:比這更短的「舉手」在 10fps 下只有 3 幀,
    多半是關鍵點抖動,併進鄰段而不是自成一段。
    """
    blk = kin[:, SIDE_SLICE[side]]
    labels = _bridge_gaps(_mode_filter(
        np.asarray(prim).astype(np.int16), smooth_win))
    min_len = max(1, int(round(min_dur * fps)))

    runs = [r for r in _rle(labels) if r[0] >= 0]
    # 同標籤且中間只隔 1 幀空窗 → 視為同一段
    merged: List[list] = []
    for val, s, e in runs:
        if merged and merged[-1][0] == val and s - merged[-1][2] <= 1:
            merged[-1][2] = e
            continue
        merged.append([val, s, e])
    # 過短的碎片直接丟掉,不併進鄰段——併進去會讓片段屬性(最小腕鼻距、
    # 肘角)混進不同動作的幀,那正是要靠屬性分類的東西
    keep = [r for r in merged if r[2] - r[1] >= min_len]
    if not keep:
        return []

    segs: List[Segment] = []
    T = len(labels)
    prev_end: Optional[int] = None
    for val, s, e in keep:
        a = np.zeros(SEG_ATTR_DIM, np.float32)
        w = blk[s:e]
        # 兩套遮罩:手臂幾何只要腕點在,臉部相關量另外要求鼻點可信。
        # 混用會出大事——鼻點不可信的幀,K_D_NOSE 是被清成 0 的,
        # 拿去取 min 會得到「腕鼻距 0.00」,也就是最強的抽菸證據。
        geo = w[:, K_VALID] > 0.5
        face = w[:, K_FACE_OK] > 0.5
        gw = w[geo] if geo.any() else w
        a[A_LOG_DUR] = np.log(max((e - s) / fps, 1e-2))
        if face.any():
            fw = w[face]
            a[A_D_MIN] = fw[:, K_D_NOSE].min()
            a[A_D_MEAN] = fw[:, K_D_NOSE].mean()
            a[A_D_START] = fw[0, K_D_NOSE]
            a[A_D_END] = fw[-1, K_D_NOSE]
            a[A_DEAR_MIN] = fw[:, K_D_EAR].min()
            a[A_DEYE_MIN] = fw[:, K_D_EYE].min()
            a[A_OVEREYE_MAX] = fw[:, K_H_OVER_EYE].max()
            a[A_PHIX_MEAN] = fw[:, K_PHI_X].mean()
            a[A_PHIY_MEAN] = fw[:, K_PHI_Y].mean()
        else:
            a[A_D_MIN] = a[A_D_MEAN] = a[A_D_START] = a[A_D_END] = FAR
            a[A_DEAR_MIN] = a[A_DEYE_MIN] = FAR
            a[A_OVEREYE_MAX] = -1.0
        a[A_D_DELTA] = a[A_D_END] - a[A_D_START]
        a[A_H_MAX] = gw[:, K_H_WRI].max()
        a[A_H_MEAN] = gw[:, K_H_WRI].mean()
        a[A_COSELB_MEAN] = gw[:, K_COS_ELBOW].mean()
        a[A_COSELB_MAX] = gw[:, K_COS_ELBOW].max()
        a[A_SPEED_MEAN] = gw[:, K_SPEED].mean()
        a[A_SPEED_MAX] = gw[:, K_SPEED].max()
        a[A_VALID_RATIO] = float(geo.mean())
        a[A_FACE_RATIO] = float(face.mean())
        a[A_T0_NORM] = s / max(T - 1, 1)
        a[A_GAP_PREV] = (np.log1p((s - prev_end) / fps)
                         if prev_end is not None else 0.0)
        prev_end = e
        segs.append(Segment(side=side, prim=val, t0=s, t1=e, fps=fps,
                            attrs=a))
    return segs


def segments_from_kinematics(kin: np.ndarray, fps: float = 10.0,
                             **kw) -> List[Segment]:
    """規則路徑的完整流程:運動學 → 基元 → 兩側片段(依時間排序)。

    L1 尚未訓練時的後備路徑;也是 L1 的偽標籤來源。
    """
    segs: List[Segment] = []
    for s in SIDES:
        segs += segment_primitives(rule_primitives(kin, s, fps), kin, s,
                                   fps, **kw)
    segs.sort(key=lambda x: (x.t0, x.side))
    return segs


# ---- 手到臉事件(raise → hold → lower 的完整循環)-----------------------

@dataclass
class Cycle:
    """一次完整的手到臉循環——第一階段 HandToMouthCounter 的片段版。"""
    side: str
    t_start: int
    t_peak: int          # hold 段起點
    t_end: int
    hold_dur: float
    d_min: float
    d_ear_min: float
    over_eye_max: float
    armed: bool          # 是否由 raise 進入(非「憑空出現在臉旁」)
    fps: float = 10.0

    @property
    def peak_s(self) -> float:
        return self.t_peak / self.fps


def find_cycles(segments: List[Segment], kin: np.ndarray,
                max_gap_s: float = 1.5, arm_lookback_s: float = 3.0,
                rise_margin: float = 0.5, near: float = NEAR
                ) -> List[Cycle]:
    """從片段序列抽出手到臉循環。

    「武裝」規則來自第一階段的教訓:手背在身後時姿態模型會以 0.9 的
    置信度把腕點幻覺在衣領上,腕-鼻距離剛好落在門檻內,看起來就像
    手在嘴邊。但幻覺點恆定懸停,**永遠不會先遠離再靠近**;真的把手
    舉到嘴邊一定會。所以 hold 要採信,前面得先看到手離臉夠遠。

    武裝的判定直接查腕-鼻距離的歷史(回看 arm_lookback_s 秒內是否
    出現過 ≥ near + rise_margin),而不是要求「前一個片段必須是
    raise」——舉手只有半秒,遇到遮擋很容易被切碎而消失,拿片段當
    條件會讓幾乎所有循環都判不成立。
    """
    cycles: List[Cycle] = []
    fps = segments[0].fps if segments else 10.0
    lookback = max(1, int(round(arm_lookback_s * fps)))
    for side in SIDES:
        blk = kin[:, SIDE_SLICE[side]]
        d = blk[:, K_D_NOSE]
        face = blk[:, K_FACE_OK] > 0.5
        ss = [s for s in segments if s.side == side]
        for i, seg in enumerate(ss):
            if seg.prim != P_HOLD:
                continue
            # L1 會在鼻點整段不可信的地方也預測 hold(它從手臂幾何推的)。
            # 那是合理的預測,但**不是手到臉的證據**——沒量到鼻子就無從
            # 確認手到底靠近了什麼。計入循環等於把「不知道」當成「有」,
            # 正是這個專案一路在對付的誤報來源。
            if not seg.face_measured:
                continue
            prev = ss[i - 1] if i > 0 else None
            nxt = ss[i + 1] if i + 1 < len(ss) else None
            lo = max(0, seg.t0 - lookback)
            hist = d[lo:seg.t0][face[lo:seg.t0]]
            armed = bool(len(hist) and hist.max() >= near + rise_margin)
            armed = armed or (
                prev is not None and prev.prim == P_RAISE and
                (seg.t0 - prev.t1) / seg.fps <= max_gap_s)
            t_start = (prev.t0 if prev is not None
                       and prev.prim == P_RAISE else seg.t0)
            t_end = (nxt.t1 if nxt is not None and nxt.prim == P_LOWER and
                     (nxt.t0 - seg.t1) / seg.fps <= max_gap_s else seg.t1)
            cycles.append(Cycle(
                side=side, t_start=t_start, t_peak=seg.t0, t_end=t_end,
                hold_dur=seg.dur, d_min=float(seg.attrs[A_D_MIN]),
                d_ear_min=float(seg.attrs[A_DEAR_MIN]),
                over_eye_max=float(seg.attrs[A_OVEREYE_MAX]),
                armed=armed, fps=seg.fps))
    cycles.sort(key=lambda c: c.t_peak)
    return cycles
