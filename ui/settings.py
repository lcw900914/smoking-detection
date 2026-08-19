"""判定方法的詳細參數:選方法時可以逐項調整,並且各方法各記各的。

**為什麼要各記各的**:論文要橫向比較各方法,每個方法本來就該有自己調到
最好的一組門檻——把 `rule` 調鬆好提高召回,不該連帶把 `hybrid` 也改掉,
否則比較的就不是「方法」而是「最後一次動到誰」。

**為什麼不是直接改 yaml**:yaml 是專案的預設值,是可以進版控、換機器帶得
走的東西;現場校準是使用者的個人偏好,兩者混在一起,調過一次就再也回不到
基準。所以覆寫值另存一份,yaml 保持原樣。

參數表是宣告式的:要新增一個可調參數,只要在 `PARAMS` 補一行,對話框、
存檔、套用全部自動跟上——與 `inference/methods.py` 的方法登錄表同一個
思路,介面不必為個別參數寫程式。
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

OVERRIDES_FILE = Path("configs/gui_overrides.json")


class Param:
    """一個可調參數。

    path 是它在推理設定裡的位置,例如 ("presence", "long_stay")。
    `needs` 決定這個參數對哪些方法有意義——顯示不相干的參數只會讓人以為
    調了會有作用。
    """

    def __init__(self, path: Tuple[str, ...], label: str, lo: float,
                 hi: float, step: float, unit: str = "", help: str = "",
                 integer: bool = False, needs: str = "any",
                 boolean: bool = False):
        self.path = path
        self.label = label
        self.lo, self.hi, self.step = lo, hi, step
        self.unit = unit
        self.help = help
        self.integer = integer
        self.boolean = boolean
        self.needs = needs          # any / skeleton / appearance / stage2

    @property
    def key(self) -> str:
        return ".".join(self.path)

    def applies_to(self, method) -> bool:
        if method is None or self.needs == "any":
            return True
        if self.needs == "skeleton":
            return method.needs_skeleton
        if self.needs == "appearance":
            return method.needs_appearance
        if self.needs == "stage2":
            return method.stage2 is not None
        return True

    def clamp(self, value):
        if self.boolean:
            return bool(value)
        v = max(self.lo, min(self.hi, float(value)))
        return int(round(v)) if self.integer else v


# 分組只影響版面;真正決定顯不顯示的是每個參數的 needs
PARAMS: List[Tuple[str, List[Param]]] = [
    ("在場型態(經過 / 徘徊 / 等待)", [
        Param(("presence", "smoking_requires_waiting"),
              "只有「等待」才判抽菸", 0, 1, 1, "",
              "開:經過與徘徊的人一律不發抽菸警報——抽菸是站定了才做的事,"
              "走動的人手臂擺動很像手到嘴。注意「等待」要在場滿下面那個"
              "秒數才成立,在那之前型態是「判定中」,同樣不會通報",
              boolean=True),
        Param(("presence", "long_stay"), "判徘徊/等待所需在場", 5, 120, 1,
              "秒",
              "在場超過這麼久才開始分徘徊或等待;之前一律是「判定中」"),
        Param(("presence", "wander_path"), "徘徊門檻(累積路徑)", 0.5, 10, 0.5,
              "倍身高",
              "累積移動超過這麼多倍自身身高 → 徘徊,否則 → 等待。"
              "用累積路徑而不是位移:原地來回踱步的位移接近零"),
        Param(("presence", "short_stay"), "判「經過」的在場上限", 2, 60, 1,
              "秒", "在場短於這麼久且有移動 → 經過"),
        Param(("presence", "pass_path"), "判「經過」的最小累積路徑", 0.2, 5,
              0.1, "倍身高",
              "移動不到這麼多就不算經過(可能只是站著晃)"),
        Param(("presence", "run_speed"), "記為跑步的速度", 0.5, 5, 0.1,
              "倍身高/秒", "平均速度超過此值,經過會標記為跑"),
        Param(("presence", "window_sec"), "軌跡回看視窗", 10, 300, 10,
              "秒", "只看最近這段時間的軌跡;在場時長仍從首次出現起算"),
    ]),
    ("手到嘴事件(決定什麼算「一口」)", [
        Param(("escalation", "min_dwell"), "停留下限", 0.5, 10, 0.5, "秒",
              "短於此視為扶眼鏡/摸臉,不計", needs="skeleton"),
        Param(("escalation", "max_dwell"), "停留上限", 1, 30, 0.5, "秒",
              "長於此視為講電話,不計", needs="skeleton"),
        Param(("escalation", "min_gap"), "兩次事件最小間隔", 0.5, 10, 0.5,
              "秒", "避免同一口被算成兩次", needs="skeleton"),
        Param(("escalation", "window_sec"), "次數統計視窗", 20, 300, 10,
              "秒", "在這段時間內累積的事件次數才算數", needs="skeleton"),
    ]),
    ("警報", [
        Param(("alarm", "min_events"), "通報所需次數", 1, 10, 1, "次",
              "手到嘴事件達到幾次才允許紅色警報", integer=True),
        Param(("alarm", "trigger_threshold"), "觸發線", 0.1, 1.0, 0.05, "",
              "P 超過此值並持續下面的秒數才通報"),
        Param(("alarm", "release_threshold"), "解除線", 0.05, 0.95, 0.05, "",
              "P 低於此值才解除(雙門檻,防抖)"),
        Param(("alarm", "sustain_sec"), "持續確認", 0.5, 10, 0.5, "秒",
              "超過觸發線要持續這麼久才真的通報"),
    ]),
    ("骨架規則", [
        Param(("skeleton", "near_ratio"), "手在臉部的距離門檻", 0.3, 2.0,
              0.05, "倍身體尺度",
              "腕-鼻距離小於此值視為手在臉部。持菸時指尖碰唇、腕距鼻仍有"
              "一掌+菸長,調太小會漏", needs="skeleton"),
        Param(("skeleton", "move_ratio"), "舉手/放下的變化量", 0.1, 1.0,
              0.05, "倍肩寬", "0.6 秒內距離變化超過此值 → 判舉手或放下",
              needs="skeleton"),
        Param(("skeleton", "min_scale_px"), "太遠就棄權", 8, 100, 2, "像素",
              "身體尺度小於此值不判定——關鍵點誤差會蓋過真實距離",
              needs="skeleton"),
        Param(("skeleton", "rise_margin"), "「由遠而近」武裝餘裕", 0.0, 2.0,
              0.1, "",
              "手要先離臉這麼遠才採信之後的停留,擋姿態模型的腕點幻覺",
              needs="skeleton"),
    ]),
    ("錄影", [
        Param(("alarm", "clip_overlay"), "警報片段疊加骨架", 0, 1, 1, "",
              "關 = 存乾淨原始影像(訓練外觀模型的前提,烙印上去救不回來);"
              "開 = 存與畫面相同的疊加版,給人複查用。只影響存檔,不影響顯示",
              boolean=True),
        Param(("alarm", "clip_pre_sec"), "片段保留觸發前幾秒", 5, 30, 1, "秒",
              "標記指向促成第一次計入事件的那次抬手,而抬手到警報成立常隔"
              "十幾秒。這個值小於那段間隔時,標記跳得到但片段拍不到動作的"
              "開頭。調大要付記憶體:滾動緩衝每秒約 10 張取樣影格"),
    ]),
    ("移動排除(走動中不判抽菸)", [
        Param(("move_gate", "enabled"), "啟用移動排除", 0, 1, 1, "",
              "開:走動中的人不發抽菸警報——走路時手臂擺動很容易被當成"
              "手到嘴。代價是**經過與徘徊的人幾乎不可能被判抽菸**,只有"
              "停下來的(等待)會。邊走邊抽的場景請關掉",
              boolean=True),
        Param(("move_gate", "max_heights"), "視為走動中的累積移動", 0.5, 10,
              0.5, "倍身高",
              "視窗內移動超過此值 → 不通報(走動時手部擺動容易誤判)"),
        Param(("move_gate", "window_sec"), "移動統計視窗", 2, 60, 1, "秒"),
    ]),
    ("融合權重(僅外觀網路相關方法)", [
        Param(("fusion", "count"), "次數權重", 0.0, 1.0, 0.1, "",
              "手到嘴「次數警戒」分數的權重。舊版標成「規則權重」,"
              "讓人以為乘的是狀態機的順序檢查分數——不是,一直都是次數分數",
              needs="appearance"),
        Param(("fusion", "network"), "網路權重", 0.0, 1.0,
              0.1, "", "", needs="appearance"),
    ]),
    ("第二階段複核(降級不否決)", [
        Param(("verify", "min_smoking"), "維持紅色的抽菸分數", 0.0, 1.0,
              0.05, "",
              "抽菸分數高於此值就不降級。刻意比升級寬——誤降一次真警報的"
              "代價遠高於多留一個待複查", needs="stage2"),
        Param(("verify", "min_valid_ratio"), "骨架可用率下限", 0.0, 1.0,
              0.05, "", "低於此值直接棄權不判定(看不到手就不該下結論)",
              needs="stage2"),
        Param(("verify", "min_span_sec"), "可複核的最短長度", 1, 60, 1, "秒",
              needs="stage2"),
        Param(("verify", "window_sec"), "往回複核的長度", 10, 300, 10, "秒",
              "警報時往前取這麼長的節點序列來複核", needs="stage2"),
    ]),
]


def all_params() -> List[Param]:
    return [p for _g, ps in PARAMS for p in ps]


def get_in(cfg: dict, path: Tuple[str, ...], default=None):
    node: Any = cfg
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def set_in(cfg: dict, path: Tuple[str, ...], value) -> None:
    node = cfg
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value


def apply_overrides(cfg: dict, overrides: Dict[str, Any]) -> dict:
    """把覆寫值套進設定,回傳新的一份(不動傳入的 cfg)。

    只認參數表裡有的鍵,而且會夾在該參數的範圍內:覆寫檔是純文字,
    手改壞了不該讓管線帶著荒謬的值跑起來(例如負的停留秒數)。
    """
    out = copy.deepcopy(cfg)
    by_key = {p.key: p for p in all_params()}
    for key, value in (overrides or {}).items():
        p = by_key.get(key)
        if p is None:
            continue
        try:
            set_in(out, p.path, p.clamp(value))
        except (TypeError, ValueError):
            continue
    return out


def defaults_for(cfg: dict, method=None) -> Dict[str, Any]:
    """從設定檔讀出這個方法會用到的參數現值。"""
    return {p.key: get_in(cfg, p.path)
            for p in all_params()
            if p.applies_to(method) and get_in(cfg, p.path) is not None}


def load_overrides(path: Path = OVERRIDES_FILE) -> Dict[str, Dict[str, Any]]:
    """讀出所有方法的覆寫值({方法代號: {參數: 值}})。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}          # 檔案不在或壞了就當沒有,用預設值跑


def save_overrides(all_ov: Dict[str, Dict[str, Any]],
                   path: Path = OVERRIDES_FILE) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_ov, f, ensure_ascii=False, indent=2, sort_keys=True)
        return True
    except OSError:
        return False       # 存不起來不該讓使用者卡住,調整這一次仍然有效


def diff_from(cfg: dict, overrides: Dict[str, Any]) -> List[str]:
    """列出哪些參數被改過(給介面顯示「已調整 N 項」)。"""
    changed = []
    by_key = {p.key: p for p in all_params()}
    for key, value in (overrides or {}).items():
        p = by_key.get(key)
        if p is None:
            continue
        base = get_in(cfg, p.path)
        if base is None or abs(float(base) - float(value)) > 1e-9:
            changed.append(key)
    return sorted(changed)
