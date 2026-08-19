"""單幀對照組:只看一張畫面判斷人體動作。

為什麼需要這支程式
──────────────────
主線是多幀骨架時序模型(L1 ST-GCN 基元 → L2 片段組合)。要說「時序有
貢獻」,就必須有一個**除了看不到時間、其他全部一樣**的對照組:同一份
骨架、同一組標籤、同一套 k-fold、同一個指標,只把時間軸拿掉。兩者的
差距才是時序的價值。沒有這個對照組,L2 的 AUC 是 0.58 還是 0.85 都
說不出意義——說不定光看一幀就有 0.8,那整個時序架構是白做的。

「其他全部一樣」要做到什麼程度
──────────────────────────────
1. **輸入必須真的只有一幀。** `kinematic_features` 的每側 19 維裡有三維
   是差分算出來的(K_V_DNOSE / K_V_H / K_SPEED)——那是偷看鄰幀。
   `FRAME_ARM_KEEP` 把它們拿掉;`graph_features` 的 (vx, vy) 兩個通道
   同理不進單幀模型。留著的話對照組會偷偷變成兩幀模型,比較就作廢,
   而且是那種跑得出漂亮數字、看不出錯在哪的作廢。
2. **切分必須以「段」為單位。** 同一段影片的幀分到訓練與驗證兩邊等於
   直接洩題(相鄰幀幾乎一模一樣)。`FrameDataset` 保留每筆樣本的段
   索引,`train_frame.py` 依段分 fold,種子與分層邏輯與 `train_l2.py`
   完全相同,fold 對齊,兩邊數字可以直接相減。
3. **輸出必須聚合回段級。** 主線吃一整段吐一個分數;單幀模型一段會吐
   幾百個分數,要聚合成一個才比得起來。見 `aggregate()`。

為什麼只看「手舉起來」的幀
──────────────────────────
手放在桌上那些幀不含任何判別資訊:抽菸的人與打字的人在那裡長得一模
一樣。把它們餵進去,等於拿段級標籤去污染大量中性幀——模型學到的會是
「這段影片的背景姿勢」而不是動作。所以候選幀限定在「手抬起來了」:
腕點幾何可用,且腕高沒有明顯低於肩線、或腕已經進到臉部範圍。門檻直接
沿用 `primitives` 的 `LOW_WRIST` 與 `NEAR`,對照組與規則層看的是同一
批幀。

這也正好對應單幀方法在真實系統裡唯一合理的用法:不可能每幀都跑,
只在「手舉起來」這個便宜的幾何觸發成立時才跑一次分類。

已知的先天劣勢(這是結論,不是缺陷)
────────────────────────────────────
「手停在嘴邊」這一幀,抽菸與喝水與講電話在骨架上幾乎沒有差別——差別
在停多久、來回幾次、間隔規不規律,那些全部是時間軸上的量。所以單幀
對照組的天花板本來就低。它的價值是把這個天花板量出來。

參考文獻
────────
`aggregate()` 的「每類各取自己最高的 k 幀平均、k 隨幀數等比例調整」
就是弱監督動作定位常用的 k-max-mean 池化:

  Paul, S., Roy, S., Roy-Chowdhury, A.K. (2018). W-TALC:
  Weakly-supervised Temporal Activity Localization and Classification.
  ECCV 2018. arXiv:1807.10418

那篇解決的是同一個問題:只有影片級標籤,要從逐段分數湊出影片級分數。
這裡的「段級標籤 + 逐幀分數」是一樣的結構。

FrameGCN 直接沿用 L1 的 SpatialGraphConv 與同一份鄰接矩陣
(見 stage2/stgcn.py 與 stage2/graph.py 的引用),差別只有時間卷積被拿掉。
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from stage2.composition import Analysis, analyze
from stage2.graph import build_adjacency
from stage2.kinematics import (GLOBAL_DIM, K_D_NOSE, K_H_WRI, K_SPEED,
                               K_VALID, K_V_DNOSE, K_V_H, SIDE_DIM,
                               SIDE_INPUT_DIM, SIDE_SLICE, graph_features,
                               kinematic_features, side_view)
from stage2.primitives import LOW_WRIST, NEAR
from stage2.stgcn import SpatialGraphConv
from stage2.taxonomy import DEEP_CLASSES, DEEP_NAMES
from utils import resolve_device

# ---------------------------------------------------------------------
# 單幀特徵版面
# ---------------------------------------------------------------------

# 差分特徵 = 偷看鄰幀,一律移除。索引是「該側 19 維」座標系裡的位置,
# side_view() 出來的 26 維前 19 維正是該側,所以直接沿用。
VELOCITY_DIMS: Tuple[int, ...] = (K_V_DNOSE, K_V_H, K_SPEED)

# 手臂幾何(16 維):腕鼻距、方位、腕耳/腕眼距、腕高、肘角、肢段長度…
FRAME_ARM_KEEP: List[int] = [i for i in range(SIDE_DIM)
                             if i not in VELOCITY_DIMS]
# 全域(7 維):軀幹傾斜、log(身體尺度)、朝向 one-hot、有效旗標
FRAME_GLOBAL_KEEP: List[int] = list(range(SIDE_DIM, SIDE_INPUT_DIM))

FRAME_ARM_DIM = len(FRAME_ARM_KEEP)            # 16
FRAME_GLOBAL_DIM = len(FRAME_GLOBAL_KEEP)      # 7

# ---------------------------------------------------------------------
# 全域特徵預設**不用**,理由請務必看完再改
#
# 全域那 7 維描述的是「這個人離鏡頭多遠、軀幹往哪邊傾、面向哪裡」——
# 完全沒有「手在做什麼」的資訊。可是實測(2026-08-18)只餵這 7 維、
# 一個手臂特徵都不給,抽菸 AUC 照樣有 0.940。
#
# 原因在資料不在模型:6 段抽菸正樣本全部來自 2026-07-08 同一場錄影、
# 同一個機位,負樣本散在 07-09 ~ 08-16。模型不必學動作,學「身體尺度
# 落在這個範圍 + 這個傾角」就贏了。整組一起餵 AUC 0.968,拿掉全域
# 之後掉到 0.750 —— 那 0.218 全部是捷徑。
#
# 對照組的職責是量出「單幀骨架能做到多少」。學到機位不算數,而且它在
# 正樣本補進來(不同場景、不同人)之後會立刻崩掉,是最糟的那種假成績:
# 離線好看、上線不能用、還看不出哪裡錯。
#
# 補足夠多**跨場景**的正樣本之前,這個開關維持關閉。train_frame.py
# 每次都會跑「捷徑探針」把只用全域特徵的 AUC 印出來;探針分數高就代表
# 資料仍然可以靠場景猜答案,那時候任何一組數字都要打折看。
# ---------------------------------------------------------------------
USE_GLOBAL_DEFAULT = False


def frame_keep(use_global: bool = USE_GLOBAL_DEFAULT) -> List[int]:
    """單幀特徵在 side_view 的 26 維版面裡保留哪些欄位。"""
    return FRAME_ARM_KEEP + (FRAME_GLOBAL_KEEP if use_global else [])


def frame_side_dim(use_global: bool = USE_GLOBAL_DEFAULT) -> int:
    return len(frame_keep(use_global))


def mlp_input_dim(use_global: bool = USE_GLOBAL_DEFAULT) -> int:
    return frame_side_dim(use_global) + 1      # +1:側別旗標

# 圖節點通道:graph_features 的 (x, y, vx, vy, conf) 砍掉速度兩通道
FRAME_GRAPH_KEEP: Tuple[int, ...] = (0, 1, 4)
FRAME_GRAPH_CHANNELS = len(FRAME_GRAPH_KEEP)   # 3

ARCHES = ("mlp", "gcn")


def frame_side_features(kin: np.ndarray, side: str,
                        use_global: bool = USE_GLOBAL_DEFAULT
                        ) -> np.ndarray:
    """(T, 45) 運動學 → 該側單幀特徵 (T, frame_side_dim(use_global))。

    左手照 L1 的做法做鏡像正規化(side_view 內建),兩側才能共用同一顆
    分類頭——等於訓練樣本加倍,在只有幾十段的情況下不是可有可無的。
    """
    return np.ascontiguousarray(
        side_view(kin, side)[..., frame_keep(use_global)].astype(np.float32))


def frame_graph_features(kpts: np.ndarray) -> np.ndarray:
    """(T, 17, 3) → (T, 13, 3) 單幀節點特徵(x, y, conf)。"""
    return np.ascontiguousarray(
        graph_features(kpts)[..., list(FRAME_GRAPH_KEEP)].astype(np.float32))


# ---------------------------------------------------------------------
# 候選幀:手舉起來的那些
# ---------------------------------------------------------------------

def hand_raised_mask(kin: np.ndarray, side: str,
                     low_wrist: float = LOW_WRIST,
                     near: float = NEAR) -> np.ndarray:
    """(T, 45) → (T,) bool:該側這一幀算不算「手舉起來」。

    三個條件都是**單幀幾何**,沒有一個要看鄰幀:
      K_VALID     腕點可見且身體座標系成立(沒有輸入就不該有輸出)
      K_H_WRI     腕高相對肩線;低於 low_wrist 就是手垂著
      K_D_NOSE    腕-鼻距離;已經進到臉部範圍的一律算數(手就算沒抬過
                  肩線,人低頭就菸的時候腕高可能仍是負的)
    """
    blk = np.asarray(kin)[:, SIDE_SLICE[side]]
    valid = blk[:, K_VALID] >= 0.5
    raised = blk[:, K_H_WRI] > low_wrist
    at_face = blk[:, K_D_NOSE] < near
    return valid & (raised | at_face)


def candidate_frames(kin: np.ndarray, **kw) -> List[Tuple[int, int]]:
    """→ [(幀索引, 側別索引)],側別索引 0 = 左、1 = 右。"""
    out: List[Tuple[int, int]] = []
    for si, side in enumerate(("L", "R")):
        for t in np.flatnonzero(hand_raised_mask(kin, side, **kw)):
            out.append((int(t), si))
    out.sort()
    return out


# ---------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------

class FrameMLP(nn.Module):
    """單幀運動學 → 動作。每側各判一次,兩側共用權重。

    刻意做小(兩層 64):對照組要量的是「單幀資訊量的上限」,不是
    「誰的正則化調得好」。參數量比 L1 小一個數量級,在 6 段正樣本的
    régime 下反而是它的優勢,輸了不能推給容量不足。
    """

    def __init__(self, num_classes: int = len(DEEP_CLASSES),
                 hidden: int = 64, dropout: float = 0.3,
                 use_global: bool = USE_GLOBAL_DEFAULT):
        super().__init__()
        self.use_global = use_global
        self.net = nn.Sequential(
            nn.Linear(mlp_input_dim(use_global), hidden), nn.LayerNorm(hidden),
            nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
            nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, mlp_input_dim(use_global)) → logits (B, C)。"""
        return self.net(x)


class FrameGCN(nn.Module):
    """單幀 13 節點骨架圖 → 動作。**L1 的 ST-GCN 拿掉時間卷積就是它。**

    刻意直接沿用 `SpatialGraphConv` 與同一份鄰接矩陣,而不是另寫一個
    圖網路:這樣「時序」是唯一的變因,連空間分區、功能邊、邊重要性遮罩
    都一模一樣。時間維以 T=1 餵進去,時間卷積自然消失。
    """

    def __init__(self, num_classes: int = len(DEEP_CLASSES),
                 channels: Sequence[int] = (32, 48, 64),
                 dropout: float = 0.3,
                 include_functional_edges: bool = True):
        super().__init__()
        A = torch.from_numpy(build_adjacency(include_functional_edges))
        self.gcns = nn.ModuleList()
        self.posts = nn.ModuleList()
        c = FRAME_GRAPH_CHANNELS
        for c_out in channels:
            self.gcns.append(SpatialGraphConv(c, c_out, A))
            self.posts.append(nn.Sequential(
                nn.BatchNorm2d(c_out), nn.ReLU(True), nn.Dropout(dropout)))
            c = c_out
        self.head = nn.Linear(c, num_classes)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 3, 13) → logits (B, C)。"""
        h = x.unsqueeze(2)                          # (B, C, T=1, V)
        for gcn, post in zip(self.gcns, self.posts):
            h = post(gcn(h))
        return self.head(h.mean(-1).squeeze(-1))    # 節點平均池化


def build_model(arch: str, num_classes: int = len(DEEP_CLASSES),
                use_global: bool = USE_GLOBAL_DEFAULT, **kw) -> nn.Module:
    """arch → 模型。use_global 只對 mlp 有意義:gcn 吃的是節點座標,
    版面裡本來就沒有「軀幹傾斜 / 遠近 / 朝向」這幾個欄位。"""
    if arch == "mlp":
        return FrameMLP(num_classes=num_classes, use_global=use_global, **kw)
    if arch == "gcn":
        return FrameGCN(num_classes=num_classes, **kw)
    raise ValueError(f"未知的單幀架構 {arch!r};可用:{', '.join(ARCHES)}")


# ---------------------------------------------------------------------
# 逐幀分數 → 段級分數
# ---------------------------------------------------------------------

TOPK_RATIO = 0.15      # 取最高的這個比例的幀
MIN_TOPK = 3           # 但至少這麼多幀


def aggregate(frame_probs: np.ndarray, ratio: float = TOPK_RATIO,
              min_k: int = MIN_TOPK) -> np.ndarray:
    """(N, C) 逐幀機率 → (C,) 段級機率:每一類各取自己最高的 k 幀平均。

    為什麼不用 max:單一幀的最高分太脆。姿態模型會把看不見的手腕以
    conf 0.9+ 幻覺在衣領上(見 docs 的腕點幻覺教訓),一幀幻覺就足以
    讓整段變成警報——而這正是第二階段要擋掉的東西,對照組自己犯同一個
    錯就沒有比較的意義。
    為什麼不用 mean:候選幀裡本來就混著抬手途中的中性幀,平均會把真正
    貼嘴的那幾幀稀釋掉;抽菸一段裡真正「停在嘴邊」的幀本來就是少數。
    取 top-k 平均是兩者之間:要求證據**持續數幀**,但不要求佔多數。

    每一類各自取自己的 top-k,最後再正規化成機率。所以分數的上限
    不是 1.0——中性幀在其他類別上也有一點機率,那些也會被各自的
    top-k 撈進分母。要看的是類別之間的相對差距,不是絕對值。
    """
    p = np.asarray(frame_probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("frame_probs 必須是 (N, C) 且 N > 0")
    k = max(min_k, int(round(ratio * p.shape[0])))
    k = min(k, p.shape[0])
    top = np.sort(p, axis=0)[-k:]              # 每一類各自排序
    out = top.mean(axis=0)
    s = out.sum()
    return (out / s) if s > 0 else np.full_like(out, 1.0 / len(out))


def abstain_scores(classes: Sequence[str] = DEEP_CLASSES) -> Dict[str, float]:
    """沒有任何候選幀時的輸出:抽菸 0,其餘攤平。

    不回傳「抽菸 = 0.5」之類的中間值:沒有手舉起來就是**沒有輸入**,
    憑空生成一個分數會被 verifier 的門檻當成證據。抽菸給 0 之後,
    降級與否交由 verifier 的 valid_ratio 棄權閘門決定(見 verifier.py
    的 ABSTAIN),而不是由這裡假裝有判斷。
    """
    others = [c for c in classes if c != "smoking"]
    v = 1.0 / len(others) if others else 0.0
    return {c: (0.0 if c == "smoking" else v) for c in classes}


# ---------------------------------------------------------------------
# 資料集
# ---------------------------------------------------------------------

class FrameDataset(torch.utils.data.Dataset):
    """候選幀 → (單幀特徵, 段級標籤)。

    **每筆樣本都記得自己來自哪一段**(第一個欄位):k-fold 一定要依段
    切,同段的相鄰幀幾乎一模一樣,混進驗證集就是洩題。`indices_of_clips`
    是那條規則唯一的執行手段,不要繞過它。

    樣本單位隨架構而不同:
        mlp  一筆 = (幀, 單側手臂)     兩側共用權重,左手鏡像正規化
        gcn  一筆 = (幀)               整張骨架圖,任一側舉手就收
    兩者都吐 (N, C) 的逐幀機率給 `aggregate()`,下游不必知道差別。

    特徵**逐段預先算好**存起來,不在 `__getitem__` 裡算:一段的
    kinematic_features 是整條時間軸一起算的,每取一筆樣本重算一次
    等於把 O(T) 的成本乘上樣本數,四千筆樣本 × 30 epoch 會跑掉半小時。
    增強則以「每個 epoch 重抽一次」的粒度做(見 `resample`)——樣本級
    的隨機性不是必要的,段級的就夠了,而且順序仍然由 DataLoader 打散。
    """

    def __init__(self, items: Sequence[dict], arch: str = "mlp",
                 label_fn=None, augment: bool = False, seed: int = 0,
                 use_global: bool = USE_GLOBAL_DEFAULT, **select_kw):
        from stage2.taxonomy import deep_index
        if arch not in ARCHES:
            raise ValueError(f"未知的單幀架構 {arch!r};可用:{', '.join(ARCHES)}")
        self.arch = arch
        self.augment = augment
        self.select_kw = select_kw
        self.seed = seed
        self.use_global = use_global
        label_fn = label_fn or (lambda code: deep_index(code))

        self.clips: List[dict] = []
        self.samples: List[Tuple[int, int]] = []      # (段索引, 段內第幾筆)
        for it in items:
            y = label_fn(it["label"]) if it.get("label") else None
            if y is None:
                continue
            kin = kinematic_features(it["kpts"], it["fps"])
            picks = self._pick(kin)
            ci = len(self.clips)
            # 一幀手都沒舉起來的段**留在 clips 裡、不產生訓練樣本**。
            # 它訓練不了(沒有輸入),但評估必須算它:對照組在這種段上
            # 只能棄權,而主線模型照樣會給分數。把它從評估裡拿掉等於
            # 偷偷幫對照組挑掉難題,比較就不誠實了。
            self.clips.append({**it, "label_index": y, "kin": kin,
                               "picks": picks,
                               "feat": _stack_features(
                                   arch, it["kpts"], kin, picks,
                                   use_global)})
            self.samples += [(ci, j) for j in range(len(picks))]
        self._aug: Optional[List[np.ndarray]] = None

    def _pick(self, kin: np.ndarray) -> List[Tuple[int, int]]:
        picks = candidate_frames(kin, **self.select_kw)
        if self.arch == "gcn":                # 同一幀兩手都舉只收一次
            seen, out = set(), []
            for t, _ in picks:
                if t not in seen:
                    seen.add(t)
                    out.append((t, -1))
            return out
        return picks

    # ---- 增強 ----

    def resample(self, epoch: int) -> None:
        """重抽一輪增強。訓練迴圈每個 epoch 開頭呼叫一次。

        增強一律作用在**原始關鍵點**上,增強完才重算運動學——順序反過來
        會讓特徵與畫面對不上(同 hier_dataset 的規矩)。變速增強在單幀
        對照組沒有意義(它看不到時間),而且會打亂幀索引,不用。
        水平翻轉只對 gcn 有意義:mlp 的特徵已由 side_view 做過鏡像
        正規化,再翻一次幾乎是恆等變換。
        """
        if not self.augment:
            self._aug = None
            return
        import random
        from stage2.hier_dataset import FLIP17
        rng = random.Random(self.seed * 100003 + epoch)
        npr = np.random.RandomState(self.seed * 7919 + epoch)
        out = []
        for c in self.clips:
            k = c["kpts"].copy()
            if self.arch == "gcn" and rng.random() < 0.5:
                vis = k[:, :, 2] > 0.1
                if vis.any():
                    cx = float(k[:, :, 0][vis].mean())
                    k = k[:, FLIP17]
                    k[:, :, 0] = 2 * cx - k[:, :, 0]
            k[:, :, :2] += npr.normal(0, 2.0, k[:, :, :2].shape)
            k = k.astype(np.float32)
            # 候選幀維持原本那組:增強是為了讓模型對關鍵點噪聲穩健,
            # 不是為了換一批幀。換了的話同一個 epoch 裡樣本數會跳動。
            out.append(_stack_features(
                self.arch, k, kinematic_features(k, c["fps"]), c["picks"],
                self.use_global))
        self._aug = out

    # ---- 段級索引(給 k-fold 用)----

    @property
    def clip_labels(self) -> np.ndarray:
        return np.array([c["label_index"] for c in self.clips], dtype=int)

    def indices_of_clips(self, clip_ids: Sequence[int]) -> List[int]:
        want = set(int(c) for c in clip_ids)
        return [i for i, s in enumerate(self.samples) if s[0] in want]

    # ---- 特徵 ----

    def clip_features(self, ci: int) -> np.ndarray:
        """整段候選幀的特徵堆疊 —— 推論與段級評估走這裡。

        永遠回傳**未增強**的版本;沒有候選幀的段回傳 N = 0 的空陣列,
        由呼叫端當作棄權處理。
        """
        return self.clips[ci]["feat"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        ci, j = self.samples[i]
        feat = self._aug[ci] if self._aug is not None             else self.clips[ci]["feat"]
        return torch.from_numpy(feat[j]), self.clips[ci]["label_index"]



def _stack_features(arch: str, kpts: np.ndarray, kin: np.ndarray,
                    picks: Sequence[Tuple[int, int]],
                    use_global: bool = USE_GLOBAL_DEFAULT) -> np.ndarray:
    """候選 (幀, 側) 清單 → 模型輸入陣列。"""
    if arch == "gcn":
        g = frame_graph_features(kpts)                    # (T, 13, 3)
        idx = [t for t, _ in picks]
        return np.ascontiguousarray(g[idx].transpose(0, 2, 1))   # (N,3,13)
    feats = {s: frame_side_features(kin, side, use_global)
             for s, side in enumerate(("L", "R"))}
    rows = [np.concatenate([feats[si][t], [float(si)]]).astype(np.float32)
            for t, si in picks]
    return np.stack(rows) if rows else np.zeros(
        (0, mlp_input_dim(use_global)), np.float32)


# ---------------------------------------------------------------------
# 推論介面 —— 與 HierarchicalRecognizer 同介面
# ---------------------------------------------------------------------

class FrameRecognizer:
    """單幀對照組的推論介面。

    `predict()` 的回傳形狀與 `stage2.infer_hier.HierarchicalRecognizer`
    完全相同,所以 `inference/verifier.py` 只要換一個建構子就能改用
    對照組——**同一支 GUI、同一份輸入、只換選單那一格**,這是
    `inference/methods.py` 開頭那條「比較才公平」的要求。

    `analysis` 欄位仍由 `composition.analyze()` 產生。注意它只餵兩件事:
    verifier 的骨架品質棄權閘門,以及 GUI 的基元時間軸。**分數完全不
    經過它**——分數只來自單幀模型 + `aggregate()`。品質閘門兩邊共用是
    刻意的:棄權條件不一致的話,兩組的比較會混進「誰比較少棄權」這個
    無關變因。

    用法:
        rec = FrameRecognizer("checkpoints/frame_mlp.pt")
        out = rec.predict(kpts, fps=10.0)
        print(out["top"], out["scores"], out["n_frames"])
    """

    def __init__(self, ckpt: str, device="auto"):
        self.device = resolve_device(device) if isinstance(device, str) \
            else device
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.arch = ck.get("arch", "mlp")
        self.classes: List[str] = list(ck.get("classes", DEEP_CLASSES))
        self.select_kw: dict = dict(ck.get("select", {}))
        self.use_global = bool(ck.get("use_global", USE_GLOBAL_DEFAULT))
        self.topk_ratio = float(ck.get("topk_ratio", TOPK_RATIO))
        self.min_topk = int(ck.get("min_topk", MIN_TOPK))
        self.model = build_model(self.arch, num_classes=len(self.classes),
                                 use_global=self.use_global)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.ckpt_path = ckpt

    # ---- 逐幀 ----

    @torch.no_grad()
    def frame_probs(self, kpts: np.ndarray, fps: float = 10.0
                    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """→ ((N, C) 逐幀機率, [(幀索引, 側別)])。沒有候選幀時 N = 0。"""
        kin = kinematic_features(kpts, fps)
        picks = candidate_frames(kin, **self.select_kw)
        if self.arch == "gcn":
            seen, ded = set(), []
            for t, _ in picks:
                if t not in seen:
                    seen.add(t)
                    ded.append((t, -1))
            picks = ded
        if not picks:
            return np.zeros((0, len(self.classes)), np.float32), []
        x = torch.from_numpy(
            _stack_features(self.arch, kpts, kin, picks,
                            self.use_global)).to(self.device)
        p = self.model(x).softmax(-1).cpu().numpy().astype(np.float32)
        return p, picks

    # ---- 段級 ----

    def analyze(self, kpts: np.ndarray, fps: float = 10.0) -> Analysis:
        """只給品質閘門與 GUI 時間軸用,不參與分數。"""
        return analyze(kinematic_features(kpts, fps), fps)

    def predict(self, kpts: np.ndarray, fps: float = 10.0) -> dict:
        """→ {scores, top, source, analysis, n_frames}。"""
        p, picks = self.frame_probs(kpts, fps)
        if len(p) == 0:
            scores = abstain_scores(self.classes)
        else:
            agg = aggregate(p, self.topk_ratio, self.min_topk)
            scores = {c: float(v) for c, v in zip(self.classes, agg)}
        return {"scores": scores, "top": max(scores, key=scores.get),
                "source": f"frame-{self.arch}",
                "analysis": self.analyze(kpts, fps),
                "n_frames": len(p)}

    def explain(self, kpts: np.ndarray, fps: float = 10.0) -> str:
        """人看的說明。刻意講清楚「這是對照組」,免得在 GUI 上被誤讀成
        主線模型的判定。"""
        out = self.predict(kpts, fps)
        n = out["n_frames"]
        head = (f"單幀對照組({self.arch.upper()}):看不到時間軸,"
                f"只對「手舉起來」的幀各判一次再取 top-k 平均")
        if n == 0:
            body = "這一段沒有任何一幀手是舉起來的 → 無輸入,不生成判斷"
        else:
            k = min(max(self.min_topk, int(round(self.topk_ratio * n))), n)
            rank = sorted(out["scores"].items(), key=lambda kv: -kv[1])[:3]
            body = (f"候選幀 {n} 筆,每類取最高 {k} 筆平均\n"
                    + "  ".join(f"{DEEP_NAMES.get(c, c)} {v:.2f}"
                                for c, v in rank))
        tail = (f"→ 判定:{DEEP_NAMES.get(out['top'], out['top'])} "
                f"({out['scores'][out['top']]:.2f})")
        return "\n".join([head, body, tail])


def main():
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="單幀對照組推論(單段/整批)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pose", default="annotations/pose/*.npz")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    rec = FrameRecognizer(args.ckpt)
    for path in sorted(glob.glob(args.pose))[:args.limit]:
        d = np.load(path, allow_pickle=True)
        print(f"\n──── {path} ────")
        print(rec.explain(d["kpts"], float(d["fps"]) or 10.0))


if __name__ == "__main__":
    main()
