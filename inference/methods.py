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

偵測與追蹤(YOLO / ByteTrack)是所有方法共用的前處理,不算方法的一部分。
所以「純規則」指的是**判定規則**不含學習權重;人物框與關鍵點仍然來自
YOLO——那是感測器,不是分類器。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# stage2 權重的預設位置(檔案不存在時該方法標記為不可用)
L1_CKPT = "checkpoints/hier_l1.pt"
L2_CKPT = "checkpoints/hier_l2.pt"

STAGE1_MODES = ("rule", "network", "hybrid")
STAGE2_MODES = (None, "grammar", "l1+grammar", "l1+l2")


@dataclass(frozen=True)
class Method:
    """一條完整的判定路徑。"""

    key: str                       # 穩定短代號(論文表格、CLI、記錄都用它)
    name: str                      # GUI 選單顯示名
    stage1: str                    # rule / network / hybrid
    stage2: Optional[str]          # None / grammar / l1+grammar / l1+l2
    desc: str                      # 一行說明(GUI 狀態列)
    count_gate: bool = True        # 是否要求「手到嘴 N 次」才允許紅色警報

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
        l1 = L1_CKPT if s2 in ("l1+grammar", "l1+l2") else None
        l2 = L2_CKPT if s2 == "l1+l2" else None
        return l1, l2

    def missing(self) -> List[str]:
        """回傳缺少的 stage2 權重檔清單(空 = 可用)。

        外觀網路權重不列在這裡:它由使用者在 GUI 現場指定,不是固定路徑。
        """
        return [p for p in self.ckpts if p and not Path(p).exists()]

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

    def apply(self, cfg: dict) -> dict:
        """回傳依本方法調整過的推理設定(不改動傳入的 cfg)。

        方法自己決定要不要開骨架分支,不依賴使用者挑對設定檔——
        選了「純規則」卻載到 `skeleton.enabled: false` 的設定,
        會變成沒有任何階段來源、P_t 永遠 0 的靜默失敗。
        """
        out = dict(cfg)
        skel = dict(out.get("skeleton", {}))
        skel["enabled"] = bool(self.needs_skeleton)
        out["skeleton"] = skel
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
