"""判定方法登錄表 —— GUI 下拉選單與 CLI `--method` 的唯一來源。

**新增一種抽菸判定方法時,唯一要改的地方就是在 `METHODS` 補一個
`Method`。** GUI 選單、可用性檢查、`--method` 參數都會自動跟上,不必在
介面裡寫 if/else。這是刻意的:論文最後要橫向比較各方法的效果,必須是
同一支 GUI、同一份輸入、只換選單那一格,比較才公平;方法散在不同入口
或不同腳本就沒辦法比。

一個「方法」= 一條完整的端到端判定路徑,由兩段組成:

    stage1(即時,線上)          stage2(候選確認,離線)
    ─────────────────           ────────────────────
    rule     骨架幾何 + 狀態機      None        不複核
    network  外觀 CNN 時序模型      grammar     片段文法(無學習權重)
    hybrid   兩者加權融合           l1+grammar  L1 網路基元 + 片段文法
                                   l1+l2       L1 + 學習版 L2
                                   l1+l2gru    L1 + L2(編碼器換成 GRU)
                                   frame-gcn   單幀對照組(單幀骨架圖)

**frame-gcn 是對照組,不是要拿來用的方法。** 它們只看「手舉起來」的那
一幀、看不到時間軸,存在的意義是量出「不用時序能做到多少」,好讓
l1+grammar / l1+l2 的分數有個參照。放進同一份清單是刻意的:論文的橫向
比較必須同一支 GUI、同一份輸入、只換選單那一格(見上面第一段)。

偵測與追蹤(YOLO / ByteTrack)是所有方法共用的前處理,不算方法的一部分。
所以「純規則」指的是**判定規則**不含學習權重;人物框與關鍵點仍然來自
YOLO——那是感測器,不是分類器。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# stage2 權重的預設位置(檔案不存在時該方法標記為不可用)
L1_CKPT = "checkpoints/hier_l1.pt"

# L2 一種序列編碼器一個權重。**除了編碼器,L2 其餘結構完全相同**
# (token 組法、時間編碼、池化、統計串接、分類頭),所以這幾格之間
# 的差距就是編碼器造成的,不會混進別的變因。見 stage2/hier_model.py
# 的 CompositionNet。
L2_CKPT = {
    "l1+l2": "checkpoints/hier_l2.pt",          # 自注意力 ×2
    "l1+l2gru": "checkpoints/hier_l2_gru.pt",   # 雙向 GRU ×2
}

# 單幀對照組(stage2/frame_baseline.py),一種架構一個權重。
#
# 只掛 gcn 一格是刻意的:frame_baseline 另外實作了 mlp 架構,離線比較
# (stage2/train_frame.py --arch mlp)還在用,但**不進選單**。理由是
# ablation 的乾淨度:gcn 就是 L1 的 ST-GCN 拿掉時間卷積,與
# rule+l1grammar 只差「有沒有時間軸」一個變因;mlp 換的是一整組手工
# 特徵,多一個變因,拿它跟時序比會說不清楚差距從哪來。
# (實測 gcn 0.661 也優於 mlp 0.609,見 docs/單幀對照組.md)
FRAME_CKPT = {
    "frame-gcn": "checkpoints/frame_gcn.pt",
}

STAGE1_MODES = ("rule", "network", "hybrid")
STAGE2_MODES = (None, "grammar", "l1+grammar",
                *sorted(L2_CKPT), *sorted(FRAME_CKPT))


@dataclass(frozen=True)
class Method:
    """一條完整的判定路徑。"""

    key: str                       # 穩定短代號(論文表格、CLI、記錄都用它)
    name: str                      # GUI 選單顯示名
    stage1: str                    # rule / network / hybrid
    stage2: Optional[str]          # None / grammar / l1+grammar / l1+l2
    desc: str                      # 一行說明(GUI 狀態列)
    count_gate: bool = True        # 是否要求「手到嘴 N 次」才允許紅色警報
    require_release: bool = False  # 事件是否要求「看得到手放下(S3)」才計入

    # ---- 前置需求 ----

    @property
    def needs_appearance(self) -> bool:
        """是否需要第一階段的外觀網路權重(GUI 的「權重」欄)。"""
        return self.stage1 in ("network", "hybrid")

    @property
    def needs_skeleton(self) -> bool:
        """是否需要 pose 模型(骨架規則與 stage2 都吃關鍵點)。"""
        return self.stage1 in ("rule", "hybrid") or self.stage2 is not None

    @property
    def ckpts(self) -> Tuple[Optional[str], Optional[str]]:
        """(L1 權重, L2 權重);不需要的回 None。"""
        s2 = self.stage2
        l1 = L1_CKPT if (s2 == "l1+grammar" or s2 in L2_CKPT) else None
        return l1, L2_CKPT.get(s2)

    @property
    def frame_ckpt(self) -> Optional[str]:
        """單幀對照組的權重;非對照組方法回 None。"""
        return FRAME_CKPT.get(self.stage2)

    @property
    def is_frame_baseline(self) -> bool:
        """這條路徑是不是「看不到時間軸」的單幀對照組。"""
        return self.stage2 in FRAME_CKPT

    def missing(self) -> List[str]:
        """回傳缺少的 stage2 權重檔清單(空 = 可用)。

        外觀網路權重不列在這裡:它由使用者在 GUI 現場指定,不是固定路徑。
        """
        need = [*self.ckpts, self.frame_ckpt]
        return [p for p in need if p and not Path(p).exists()]

    @property
    def available(self) -> bool:
        return not self.missing()

    @property
    def ai_free_decision(self) -> bool:
        """判定過程是否完全不含學習權重。

        注意「不含學習權重」只涵蓋**判定**:人物偵測與關鍵點估計仍是
        YOLO。那一段是感測器(把畫面變成座標),不是在決定「這是不是
        抽菸」——決定是誰做的才是這個旗標在區分的事。
        """
        return self.stage1 == "rule" and self.stage2 in (None, "grammar")

    # ---- 套用到設定檔 ----

    def apply(self, cfg: dict) -> dict:  # noqa: D401
        """回傳依本方法調整過的推理設定(不改動傳入的 cfg)。

        方法自己決定要不要開骨架分支,不依賴使用者挑對設定檔——
        選了「純規則」卻載到 `skeleton.enabled: false` 的設定,
        會變成沒有任何階段來源、P_t 永遠 0 的靜默失敗。
        """
        out = dict(cfg)
        skel = dict(out.get("skeleton", {}))
        skel["enabled"] = bool(self.needs_skeleton)
        out["skeleton"] = skel
        esc = dict(out.get("escalation", {}))
        esc["require_release"] = bool(self.require_release)
        out["escalation"] = esc
        return out


METHODS: Tuple[Method, ...] = (
    # ---- 純規則家族:判定完全不靠學習權重 ----
    Method(
        key="rule",
        name="純規則:骨架幾何 + 狀態機",
        stage1="rule", stage2=None,
        desc="腕-鼻距離判 S1/S2/S3,手到嘴次數決定警戒。判定零學習權重,"
             "門檻可現場調,誤判原因看得見。"),
    Method(
        key="rule+order",
        name="純規則 + 順序檢查(要求看得到放下)",
        stage1="rule", stage2=None, require_release=True,
        desc="與 rule 只差一條:S2 停留必須在 2 秒內看到 S3(手明顯拉遠)"
             "才計為一次事件。「手到嘴然後手就不見了」不算。"
             "這是 StageStateMachine 原本要守卻沒接上線的那半條規則。"),

    Method(
        key="rule+grammar",
        name="純規則 + 片段文法複核",
        stage1="rule", stage2="grammar",
        desc="警報後再用骨架時序的片段文法複核(手到嘴 0.4–3 秒、來回多次、"
             "貼嘴非貼耳)。全程無任何學習權重。"),
    Method(
        key="rule+l1grammar",
        name="純規則 + L1 網路基元 + 片段文法",
        stage1="rule", stage2="l1+grammar",
        desc="複核時的逐幀基元改由訓練好的 L1(ST-GCN)給,深層仍是文法。"
             "stage2 目前離線 AUC 最佳的組合。"),
    Method(
        key="rule+l1l2",
        name="純規則 + L1 + 學習版 L2",
        stage1="rule", stage2="l1+l2",
        desc="深層也換成學習版 L2。⚠ 現有 6 段正樣本訓不動它,"
             "離線 AUC 0.58 輸給文法 0.75,列在這裡是為了論文對照。"),

    # ---- 外觀網路家族 ----
    Method(
        key="net",
        name="純外觀網路(CNN 時序)",
        stage1="network", stage2=None,
        desc="只用 channel-as-temporal-buffer 外觀網路的 cycle score 驅動警報,"
             "不套次數規則。HMDB51 訓練,俯視監控有 domain gap。",
        count_gate=False),
    Method(
        key="hybrid",
        name="外觀網路 + 規則融合(2026-07 舊預設)",
        stage1="hybrid", stage2=None,
        desc="cycle = 0.5×次數警戒 + 0.5×網路分數。實地 57 段警報幾乎全為誤報,"
             "作為改進的基準線。"),
    # ---- 單幀對照組:看不到時間軸,量的是「不用時序能做到多少」----
    Method(
        key="rule+l1l2gru",
        name="純規則 + L1 + 學習版 L2(GRU 編碼器)",
        stage1="rule", stage2="l1+l2gru",
        desc="與 rule+l1l2 只差一件事:L2 的序列編碼器由自注意力換成雙向 "
             "GRU(21k → 14k 參數)。用來回答「編碼器選哪個比較好」,"
             "其餘結構完全相同。"),

    Method(
        key="rule+frame_gcn",
        name="純規則 + 單幀對照組(單幀骨架圖)",
        stage1="rule", stage2="frame-gcn",
        desc="對照組:只看「手舉起來」的單幀 13 節點骨架圖,每幀各判一次"
             "再取 top-k 平均。就是 L1 的 ST-GCN 拿掉時間卷積,與"
             "rule+l1grammar 只差「有沒有時間軸」。是拿來比的,不是拿來用的。"),

    Method(
        key="hybrid+l1grammar",
        name="外觀網路 + 規則 + L1 + 片段文法",
        stage1="hybrid", stage2="l1+grammar",
        desc="第一階段照舊高召回,再由兩層骨架時序模型過濾誤報(降級不否決)。"
             "目前最完整的組合。"),
)

DEFAULT_KEY = "rule"

_BY_KEY: Dict[str, Method] = {m.key: m for m in METHODS}


def get(key: str) -> Method:
    """代號 → Method;未知代號給出可用清單而不是 KeyError。"""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"未知的方法代號 {key!r};可用:{', '.join(keys())}")


def keys() -> List[str]:
    return [m.key for m in METHODS]


def names() -> List[str]:
    """GUI 選單用的顯示名(順序與 METHODS 一致)。"""
    return [m.name for m in METHODS]


def default() -> Method:
    return get(DEFAULT_KEY)


def by_name(name: str) -> Method:
    """顯示名 → Method(GUI 的 Combobox 回傳的是顯示名)。"""
    for m in METHODS:
        if m.name == name:
            return m
    raise KeyError(f"未知的方法名稱 {name!r}")
